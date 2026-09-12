use std::{collections::BTreeMap, sync::Arc, time::Instant};

use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use irongraph_types::{Error, ErrorCode, Result};

/// Stable connection identity used only for bounded in-memory fairness accounting.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct ConnectionId(pub u64);

impl ConnectionId {
    /// Creates an identity independently from every request/idempotency key. The random value is
    /// process-local accounting state only and is never persisted in a write command.
    #[must_use]
    pub fn new() -> Self {
        loop {
            let bytes = Uuid::new_v4().into_bytes();
            let value = u64::from_be_bytes(bytes[..8].try_into().unwrap_or([0; 8]));
            if value != 0 {
                return Self(value);
            }
        }
    }

    #[must_use]
    pub const fn is_valid(self) -> bool {
        self.0 != 0
    }
}

impl Default for ConnectionId {
    fn default() -> Self {
        Self::new()
    }
}

/// Admission category. Control work may use the reserved capacity ordinary writes cannot consume.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AdmissionClass {
    Client,
    Broker,
    Control,
    Cancellation,
}

impl AdmissionClass {
    const fn is_control(self) -> bool {
        matches!(self, Self::Control | Self::Cancellation)
    }
}

/// Hard limits for the sole in-memory write admission path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AdmissionLimits {
    pub max_requests: usize,
    pub max_encoded_bytes: usize,
    pub reserved_control_requests: usize,
    pub reserved_control_bytes: usize,
    pub max_requests_per_connection: usize,
    pub max_encoded_bytes_per_connection: usize,
    pub retry_after_ms: u64,
}

#[derive(Clone, Copy, Debug, Default)]
struct ConnectionUsage {
    requests: usize,
    bytes: usize,
}

#[derive(Debug, Default)]
struct AdmissionState {
    requests: usize,
    bytes: usize,
    connections: BTreeMap<ConnectionId, ConnectionUsage>,
}

/// Thread-safe count/byte/deadline admission controller with control-plane reserve.
#[derive(Clone, Debug)]
pub struct AdmissionController {
    limits: AdmissionLimits,
    state: Arc<Mutex<AdmissionState>>,
}

impl AdmissionController {
    pub fn new(limits: AdmissionLimits) -> Result<Self> {
        if limits.max_requests == 0
            || limits.max_encoded_bytes == 0
            || limits.max_requests_per_connection == 0
            || limits.max_encoded_bytes_per_connection == 0
            || limits.max_encoded_bytes_per_connection > limits.max_encoded_bytes
            || limits.reserved_control_requests >= limits.max_requests
            || limits.reserved_control_bytes >= limits.max_encoded_bytes
        {
            return Err(Error::invalid_data("invalid write-admission limits"));
        }
        Ok(Self {
            limits,
            state: Arc::new(Mutex::new(AdmissionState::default())),
        })
    }

    pub fn try_admit(
        &self,
        connection: ConnectionId,
        class: AdmissionClass,
        encoded_bytes: usize,
        deadline: Instant,
    ) -> Result<AdmissionPermit> {
        if Instant::now() >= deadline {
            return Err(Error::retryable(
                ErrorCode::DeadlineExceeded,
                "write deadline expired before admission",
                None,
            ));
        }
        if encoded_bytes == 0 || encoded_bytes > self.limits.max_encoded_bytes {
            return Err(Error::invalid_data("invalid admitted mutation byte size"));
        }
        if !connection.is_valid() {
            return Err(Error::invalid_data(
                "write admission requires a connection identity",
            ));
        }
        let mut state = self.state.lock();
        let usage = state
            .connections
            .get(&connection)
            .copied()
            .unwrap_or_default();
        if !class.is_control() && usage.requests >= self.limits.max_requests_per_connection {
            return Err(self.full_error("connection write-admission share is full"));
        }
        if !class.is_control()
            && usage
                .bytes
                .checked_add(encoded_bytes)
                .is_none_or(|bytes| bytes > self.limits.max_encoded_bytes_per_connection)
        {
            return Err(self.full_error("connection write-admission byte share is full"));
        }

        let request_limit = if class.is_control() {
            self.limits.max_requests
        } else {
            self.limits.max_requests - self.limits.reserved_control_requests
        };
        let byte_limit = if class.is_control() {
            self.limits.max_encoded_bytes
        } else {
            self.limits.max_encoded_bytes - self.limits.reserved_control_bytes
        };
        let next_requests = state
            .requests
            .checked_add(1)
            .ok_or_else(|| Error::internal("write-admission request count overflow"))?;
        let next_bytes = state
            .bytes
            .checked_add(encoded_bytes)
            .ok_or_else(|| Error::internal("write-admission byte count overflow"))?;
        if next_requests > request_limit || next_bytes > byte_limit {
            return Err(self.full_error("write-admission capacity is full"));
        }

        state.requests = next_requests;
        state.bytes = next_bytes;
        state.connections.insert(
            connection,
            ConnectionUsage {
                requests: usage.requests + 1,
                bytes: usage
                    .bytes
                    .checked_add(encoded_bytes)
                    .ok_or_else(|| Error::internal("connection byte accounting overflow"))?,
            },
        );
        Ok(AdmissionPermit {
            connection,
            encoded_bytes,
            state: Arc::clone(&self.state),
            released: false,
        })
    }

    #[must_use]
    pub fn usage(&self) -> (usize, usize) {
        let state = self.state.lock();
        (state.requests, state.bytes)
    }

    #[must_use]
    pub const fn maximum_encoded_bytes(&self) -> usize {
        self.limits.max_encoded_bytes
    }

    fn full_error(&self, message: &'static str) -> Error {
        Error::retryable(
            ErrorCode::WriteAdmissionFull,
            message,
            Some(self.limits.retry_after_ms),
        )
    }
}

/// RAII admission reservation released on success, error, disconnect, or cancellation.
#[derive(Debug)]
pub struct AdmissionPermit {
    connection: ConnectionId,
    encoded_bytes: usize,
    state: Arc<Mutex<AdmissionState>>,
    released: bool,
}

impl AdmissionPermit {
    pub fn release(mut self) {
        self.release_inner();
    }

    fn release_inner(&mut self) {
        if self.released {
            return;
        }
        let mut state = self.state.lock();
        state.requests = state.requests.saturating_sub(1);
        state.bytes = state.bytes.saturating_sub(self.encoded_bytes);
        let remove = if let Some(usage) = state.connections.get_mut(&self.connection) {
            usage.requests = usage.requests.saturating_sub(1);
            usage.bytes = usage.bytes.saturating_sub(self.encoded_bytes);
            usage.requests == 0
        } else {
            false
        };
        if remove {
            state.connections.remove(&self.connection);
        }
        self.released = true;
    }
}

impl Drop for AdmissionPermit {
    fn drop(&mut self) {
        self.release_inner();
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{
        Arc, Barrier,
        atomic::{AtomicUsize, Ordering},
    };
    use std::thread;
    use std::time::Duration;

    use super::*;

    #[test]
    fn concurrent_clients_cannot_consume_reserved_control_capacity() -> Result<()> {
        let admission = Arc::new(AdmissionController::new(AdmissionLimits {
            max_requests: 8,
            max_encoded_bytes: 800,
            reserved_control_requests: 2,
            reserved_control_bytes: 200,
            max_requests_per_connection: 2,
            max_encoded_bytes_per_connection: 200,
            retry_after_ms: 25,
        })?);
        let barrier = Arc::new(Barrier::new(17));
        let accepted = Arc::new(AtomicUsize::new(0));
        let handles = (0..16)
            .map(|_| {
                let admission = Arc::clone(&admission);
                let barrier = Arc::clone(&barrier);
                let accepted = Arc::clone(&accepted);
                thread::spawn(move || {
                    let permit = admission.try_admit(
                        ConnectionId::new(),
                        AdmissionClass::Client,
                        100,
                        Instant::now() + Duration::from_secs(1),
                    );
                    if permit.is_ok() {
                        accepted.fetch_add(1, Ordering::AcqRel);
                    }
                    barrier.wait();
                    drop(permit);
                })
            })
            .collect::<Vec<_>>();
        barrier.wait();
        assert_eq!(accepted.load(Ordering::Acquire), 6);
        for handle in handles {
            handle.join().expect("admission test thread panicked");
        }

        let clients = (0..6)
            .map(|_| {
                admission.try_admit(
                    ConnectionId::new(),
                    AdmissionClass::Client,
                    100,
                    Instant::now() + Duration::from_secs(1),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        let controls = (0..2)
            .map(|_| {
                admission.try_admit(
                    ConnectionId::new(),
                    AdmissionClass::Control,
                    100,
                    Instant::now() + Duration::from_secs(1),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        assert_eq!(admission.usage(), (8, 800));
        assert_eq!(
            admission
                .try_admit(
                    ConnectionId::new(),
                    AdmissionClass::Control,
                    1,
                    Instant::now() + Duration::from_secs(1),
                )
                .err()
                .map(|error| error.code),
            Some(ErrorCode::WriteAdmissionFull)
        );
        drop(clients);
        drop(controls);
        assert_eq!(admission.usage(), (0, 0));
        Ok(())
    }
}
