use std::{sync::Arc, sync::LazyLock};

use parking_lot::Mutex;
use tokio::sync::{OwnedSemaphorePermit, Semaphore, TryAcquireError};

use crate::{Error, ErrorCode, Result};

const BYTE_QUANTUM: usize = 64 * 1024;
// 1 GiB expressed in semaphore quanta; this compile-time constant needs no fallible startup path.
const PROCESS_BROKER_MEMORY_UNITS: u32 = 16_384;

static PROCESS_BROKER_MEMORY: LazyLock<Arc<BrokerMemoryGovernor>> = LazyLock::new(|| {
    Arc::new(BrokerMemoryGovernor::from_valid_units(
        PROCESS_BROKER_MEMORY_UNITS,
    ))
});

pub(crate) fn process_broker_memory() -> Arc<BrokerMemoryGovernor> {
    Arc::clone(&PROCESS_BROKER_MEMORY)
}

/// One process-wide host-memory budget shared by every Kafka and AMQP transport.
///
/// Reservations are rounded up to a small quantum so the semaphore stays compact. The rounding
/// is deliberately conservative: admitted allocation is never larger than the reserved amount.
#[derive(Debug)]
pub(crate) struct BrokerMemoryGovernor {
    units: Arc<Semaphore>,
    maximum_units: u32,
}

impl BrokerMemoryGovernor {
    fn from_valid_units(maximum_units: u32) -> Self {
        Self {
            units: Arc::new(Semaphore::new(maximum_units as usize)),
            maximum_units,
        }
    }

    #[cfg(test)]
    pub(crate) fn new(maximum_bytes: usize) -> Result<Self> {
        let maximum_units = bytes_to_units(maximum_bytes)?;
        if maximum_units == 0 {
            return Err(Error::invalid_data("broker memory budget must be positive"));
        }
        Ok(Self::from_valid_units(maximum_units))
    }

    pub(crate) async fn reserve(self: &Arc<Self>, bytes: usize) -> Result<BrokerMemoryReservation> {
        let units = self.validate_units(bytes)?;
        let permit = Arc::clone(&self.units)
            .acquire_many_owned(units)
            .await
            .map_err(|_| Error::internal("broker memory governor closed"))?;
        Ok(BrokerMemoryReservation {
            governor: Arc::clone(self),
            state: Arc::new(Mutex::new(ReservationState {
                permit: Some(permit),
                units,
            })),
        })
    }

    pub(crate) fn try_reserve(self: &Arc<Self>, bytes: usize) -> Result<BrokerMemoryReservation> {
        let units = self.validate_units(bytes)?;
        let permit = Arc::clone(&self.units)
            .try_acquire_many_owned(units)
            .map_err(map_try_acquire)?;
        Ok(BrokerMemoryReservation {
            governor: Arc::clone(self),
            state: Arc::new(Mutex::new(ReservationState {
                permit: Some(permit),
                units,
            })),
        })
    }

    fn validate_units(&self, bytes: usize) -> Result<u32> {
        let units = bytes_to_units(bytes)?;
        if units > self.maximum_units {
            return Err(backpressure(
                "broker allocation exceeds the process memory budget",
            ));
        }
        Ok(units)
    }

    #[cfg(test)]
    pub(crate) fn available_bytes(&self) -> usize {
        self.units.available_permits() * BYTE_QUANTUM
    }

    /// Total budget this governor can ever hand out.
    ///
    /// The delivery pump sizes its batch against this so it never opens with a reservation the
    /// process could not satisfy even when completely idle.
    pub(crate) fn maximum_bytes(&self) -> usize {
        self.maximum_units as usize * BYTE_QUANTUM
    }
}

/// RAII reservation. Dropping it releases capacity on success, error, disconnect, task abort,
/// or cancellation.
#[derive(Debug)]
struct ReservationState {
    permit: Option<OwnedSemaphorePermit>,
    units: u32,
}

#[derive(Clone, Debug)]
pub(crate) struct BrokerMemoryReservation {
    governor: Arc<BrokerMemoryGovernor>,
    state: Arc<Mutex<ReservationState>>,
}

impl BrokerMemoryReservation {
    pub(crate) fn empty(governor: Arc<BrokerMemoryGovernor>) -> Self {
        Self {
            governor,
            state: Arc::new(Mutex::new(ReservationState {
                permit: None,
                units: 0,
            })),
        }
    }

    pub(crate) fn try_grow_to(&mut self, bytes: usize) -> Result<()> {
        let desired = self.governor.validate_units(bytes)?;
        let mut state = self.state.lock();
        if desired <= state.units {
            return Ok(());
        }
        let additional = desired - state.units;
        let acquired = Arc::clone(&self.governor.units)
            .try_acquire_many_owned(additional)
            .map_err(map_try_acquire)?;
        if let Some(permit) = state.permit.as_mut() {
            permit.merge(acquired);
        } else {
            state.permit = Some(acquired);
        }
        state.units = desired;
        Ok(())
    }

    /// Runs non-cancellable blocking work while a cloned reservation remains owned by the
    /// blocking closure. Aborting the async waiter therefore cannot release memory accounting
    /// before the blocking allocation and its result are dropped.
    pub(crate) async fn spawn_blocking<T, F>(
        &self,
        work: F,
    ) -> std::result::Result<T, tokio::task::JoinError>
    where
        T: Send + 'static,
        F: FnOnce() -> T + Send + 'static,
    {
        let reservation = self.clone();
        let completed = tokio::task::spawn_blocking(move || ReservedBlockingOutput {
            value: work(),
            _reservation: reservation,
        })
        .await?;
        Ok(completed.value)
    }

    #[cfg(test)]
    pub(crate) fn reserved_bytes(&self) -> usize {
        self.state.lock().units as usize * BYTE_QUANTUM
    }
}

struct ReservedBlockingOutput<T> {
    value: T,
    _reservation: BrokerMemoryReservation,
}

fn bytes_to_units(bytes: usize) -> Result<u32> {
    if bytes == 0 {
        return Ok(0);
    }
    bytes
        .checked_add(BYTE_QUANTUM - 1)
        .map(|rounded| rounded / BYTE_QUANTUM)
        .and_then(|units| u32::try_from(units).ok())
        .ok_or_else(|| Error::invalid_data("broker memory reservation size overflow"))
}

fn map_try_acquire(error: TryAcquireError) -> Error {
    match error {
        TryAcquireError::NoPermits => backpressure("broker process memory is under pressure"),
        TryAcquireError::Closed => Error::internal("broker memory governor closed"),
    }
}

fn backpressure(message: &'static str) -> Error {
    Error::retryable(ErrorCode::Backpressure, message, Some(25))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    #[test]
    fn process_governor_has_the_fixed_one_gibibyte_budget() {
        assert_eq!(process_broker_memory().maximum_bytes(), 1024 * 1024 * 1024);
    }

    #[test]
    fn invalid_configured_budgets_are_rejected_without_affecting_process_initialization() {
        assert_eq!(
            BrokerMemoryGovernor::new(0).err().map(|error| error.code),
            Some(ErrorCode::InvalidData)
        );
        assert_eq!(
            BrokerMemoryGovernor::new(usize::MAX)
                .err()
                .map(|error| error.code),
            Some(ErrorCode::InvalidData)
        );
    }

    #[tokio::test]
    async fn concurrent_reservations_are_bounded_and_raii_releases_every_path() -> Result<()> {
        let governor = Arc::new(BrokerMemoryGovernor::new(4 * BYTE_QUANTUM)?);
        let first = governor.reserve(2 * BYTE_QUANTUM).await?;
        let second = governor.try_reserve(2 * BYTE_QUANTUM)?;
        assert_eq!(governor.available_bytes(), 0);
        assert_eq!(
            governor
                .try_reserve(1)
                .expect_err("aggregate reservation must be rejected")
                .code,
            ErrorCode::Backpressure
        );

        drop(first);
        assert_eq!(governor.available_bytes(), 2 * BYTE_QUANTUM);
        let replacement = governor.try_reserve(BYTE_QUANTUM + 1)?;
        assert_eq!(replacement.reserved_bytes(), 2 * BYTE_QUANTUM);
        drop(replacement);
        drop(second);
        assert_eq!(governor.available_bytes(), 4 * BYTE_QUANTUM);
        Ok(())
    }

    #[tokio::test]
    async fn cancelled_waiter_does_not_leak_capacity() -> Result<()> {
        let governor = Arc::new(BrokerMemoryGovernor::new(BYTE_QUANTUM)?);
        let held = governor.reserve(BYTE_QUANTUM).await?;
        let waiting = {
            let governor = Arc::clone(&governor);
            tokio::spawn(async move { governor.reserve(BYTE_QUANTUM).await })
        };
        tokio::task::yield_now().await;
        waiting.abort();
        assert!(
            waiting
                .await
                .expect_err("waiter must be cancelled")
                .is_cancelled()
        );
        drop(held);
        assert_eq!(governor.available_bytes(), BYTE_QUANTUM);
        Ok(())
    }

    #[tokio::test]
    async fn adversarial_concurrent_claims_cannot_exceed_the_process_budget() -> Result<()> {
        const WORKERS: usize = 64;
        const CAPACITY: usize = 4;
        let governor = Arc::new(BrokerMemoryGovernor::new(CAPACITY * BYTE_QUANTUM)?);
        let start = Arc::new(tokio::sync::Barrier::new(WORKERS + 1));
        let attempted = Arc::new(tokio::sync::Barrier::new(WORKERS + 1));
        let release = Arc::new(tokio::sync::Barrier::new(WORKERS + 1));
        let admitted = Arc::new(AtomicUsize::new(0));
        let mut tasks = Vec::with_capacity(WORKERS);
        for _ in 0..WORKERS {
            let governor = Arc::clone(&governor);
            let start = Arc::clone(&start);
            let attempted = Arc::clone(&attempted);
            let release = Arc::clone(&release);
            let admitted = Arc::clone(&admitted);
            tasks.push(tokio::spawn(async move {
                start.wait().await;
                let permit = governor.try_reserve(BYTE_QUANTUM).ok();
                if permit.is_some() {
                    admitted.fetch_add(1, Ordering::AcqRel);
                }
                attempted.wait().await;
                release.wait().await;
                drop(permit);
            }));
        }
        start.wait().await;
        attempted.wait().await;
        assert_eq!(admitted.load(Ordering::Acquire), CAPACITY);
        assert_eq!(governor.available_bytes(), 0);
        release.wait().await;
        for task in tasks {
            task.await
                .map_err(|error| Error::internal(format!("governor worker failed: {error}")))?;
        }
        assert_eq!(governor.available_bytes(), CAPACITY * BYTE_QUANTUM);
        Ok(())
    }
}
