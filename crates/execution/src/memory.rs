//! Device memory admission for graph, model, and query allocations.

use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};

use serde::{Deserialize, Serialize};

use crate::{Error, ErrorCode, Result};

/// Non-graph allocations sharing one physical device/unified-memory budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SharedMemoryClass {
    ModelWeights,
    EmbeddingWeights,
    EncoderState,
    GraphMemory,
    DeviceOverhead,
}

impl SharedMemoryClass {
    const fn index(self) -> usize {
        match self {
            Self::ModelWeights => 0,
            Self::EmbeddingWeights => 1,
            Self::EncoderState => 2,
            Self::GraphMemory => 3,
            Self::DeviceOverhead => 4,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceMemorySnapshot {
    pub limit_bytes: usize,
    pub safety_reserve_bytes: usize,
    pub persistent_graph_bytes: usize,
    pub pinned_generation_bytes: usize,
    pub model_weight_bytes: usize,
    pub embedding_weight_bytes: usize,
    pub encoder_state_bytes: usize,
    pub graph_memory_bytes: usize,
    pub device_overhead_bytes: usize,
    pub scratch_bytes: usize,
    pub staging_bytes: usize,
}

impl DeviceMemorySnapshot {
    #[must_use]
    pub fn admitted_bytes(self) -> usize {
        self.safety_reserve_bytes
            .saturating_add(self.persistent_graph_bytes)
            .saturating_add(self.pinned_generation_bytes)
            .saturating_add(self.model_weight_bytes)
            .saturating_add(self.embedding_weight_bytes)
            .saturating_add(self.encoder_state_bytes)
            .saturating_add(self.graph_memory_bytes)
            .saturating_add(self.device_overhead_bytes)
            .saturating_add(self.scratch_bytes)
            .saturating_add(self.staging_bytes)
    }
}

#[derive(Debug, Default)]
struct MemoryLedger {
    persistent_graph_bytes: usize,
    pinned_generation_bytes: usize,
    shared_bytes: [usize; 5],
    scratch_bytes: usize,
    staging_bytes: usize,
}

/// Process-wide admission for every allocation sharing one device or Apple unified memory.
///
/// A single mutex deliberately serializes the uncommon allocation/admission boundary. This makes
/// the fit check and the corresponding accounting change atomic across graph publication, model
/// loading, generation state, graph-memory banks, query scratch, and replacement staging.
#[derive(Clone)]
pub struct DeviceMemoryGovernor {
    limit_bytes: Arc<AtomicUsize>,
    reserved_bytes: usize,
    ledger: Arc<Mutex<MemoryLedger>>,
    pinned_reservation: Option<Arc<PinnedGenerationReservation>>,
    publishes_persistent_total: bool,
    /// Reports bytes the HOST still has free, or `None` when it cannot be determined.
    ///
    /// Injected rather than read here: this crate is platform-neutral and the measurement is not.
    /// The backend that knows which device it is talking to installs it.
    ///
    /// Why it exists: `limit_bytes` is derived once, at backend construction, from the device's
    /// hardware working-set size. That number is correct when the process starts and never revisited,
    /// so a process that started on an idle machine keeps its whole budget after other tenants have
    /// taken the memory. Measured on a 48 GB host: an admitted budget of 37.4 GB against 19.2-19.4 GB
    /// actually available, with swap 91% used. The governor was not wrong about its own ledger — it
    /// was answering "does this fit the hardware" when the question is "does this fit what is left".
    host_available_bytes: Option<Arc<dyn Fn() -> Option<usize> + Send + Sync>>,
}

impl std::fmt::Debug for DeviceMemoryGovernor {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("DeviceMemoryGovernor")
            .field("limit_bytes", &self.limit_bytes)
            .field("reserved_bytes", &self.reserved_bytes)
            .field(
                "publishes_persistent_total",
                &self.publishes_persistent_total,
            )
            .field("host_available_probe", &self.host_available_bytes.is_some())
            .finish_non_exhaustive()
    }
}

impl DeviceMemoryGovernor {
    #[must_use]
    pub fn new(limit_bytes: usize, reserved_bytes: usize) -> Self {
        Self {
            limit_bytes: Arc::new(AtomicUsize::new(limit_bytes)),
            reserved_bytes,
            ledger: Arc::new(Mutex::new(MemoryLedger::default())),
            pinned_reservation: None,
            publishes_persistent_total: true,
            host_available_bytes: None,
        }
    }

    /// Installs the host free-memory probe. Called by the platform-specific backend, because this
    /// crate does not know how to ask.
    ///
    /// Returning `None` from the probe means "could not determine", and admission then behaves
    /// exactly as it does without a probe. A memory check that cannot read memory must not become a
    /// new way to refuse work.
    pub fn set_host_available_probe(
        &mut self,
        probe: Arc<dyn Fn() -> Option<usize> + Send + Sync>,
    ) {
        self.host_available_bytes = Some(probe);
    }

    /// Whether admitting `growth` more bytes would take more than the host currently has free.
    ///
    /// Deliberately bounds the **increase**, not the total. `limit_bytes` already bounds the total
    /// against the device's working-set size; bytes already admitted are resident and therefore not
    /// part of what the host reports as free, so comparing a total against free memory would
    /// double-count them and refuse work that fits.
    fn host_would_be_exhausted(&self, growth: usize) -> Option<(usize, usize)> {
        if growth == 0 {
            return None;
        }
        let available = self.host_available_bytes.as_ref()?()?;
        (growth > available).then_some((growth, available))
    }

    pub fn admit_persistent(&mut self, bytes: usize) -> Result<()> {
        let mut ledger = self.lock_ledger()?;
        let (persistent, next_pinned, prior_pin) = if self.publishes_persistent_total {
            (
                bytes
                    .checked_add(ledger.pinned_generation_bytes)
                    .ok_or_else(memory_accounting_overflow)?,
                ledger.pinned_generation_bytes,
                None,
            )
        } else {
            let reservation = self
                .pinned_reservation
                .as_ref()
                .ok_or_else(|| Error::internal("pinned governor has no generation reservation"))?;
            let prior = reservation.bytes.load(Ordering::Acquire);
            let next = ledger
                .pinned_generation_bytes
                .checked_sub(prior)
                .and_then(|value| value.checked_add(bytes))
                .ok_or_else(memory_accounting_overflow)?;
            (
                ledger
                    .persistent_graph_bytes
                    .checked_add(next)
                    .ok_or_else(memory_accounting_overflow)?,
                next,
                Some((Arc::clone(reservation), prior)),
            )
        };
        if self.total_with(
            &ledger,
            persistent,
            ledger.scratch_bytes,
            ledger.staging_bytes,
        )? > self.limit_bytes()
        {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "complete graph does not fit beside the admitted model/runtime and safety reserve",
            ));
        }
        if self.publishes_persistent_total {
            ledger.persistent_graph_bytes = bytes;
            return Ok(());
        }
        ledger.pinned_generation_bytes = next_pinned;
        if let Some((reservation, prior)) = prior_pin {
            if reservation
                .bytes
                .compare_exchange(prior, bytes, Ordering::AcqRel, Ordering::Acquire)
                .is_err()
            {
                ledger.pinned_generation_bytes = ledger
                    .pinned_generation_bytes
                    .checked_sub(bytes)
                    .and_then(|value| value.checked_add(prior))
                    .ok_or_else(memory_accounting_overflow)?;
                return Err(Error::internal(
                    "pinned generation reservation changed during admission",
                ));
            }
        }
        Ok(())
    }

    #[must_use]
    pub fn limit_bytes(&self) -> usize {
        self.limit_bytes.load(Ordering::Acquire)
    }

    /// Atomically lowers the shared budget to a measured hardware working-set ceiling.
    /// Existing and future clones observe the same cap.
    pub fn cap_limit(&self, hardware_limit_bytes: usize) -> Result<usize> {
        if hardware_limit_bytes == 0 {
            return Ok(self.limit_bytes());
        }
        let ledger = self.lock_ledger()?;
        let current = self.limit_bytes();
        let effective = current.min(hardware_limit_bytes);
        let admitted = self.total_with(
            &ledger,
            self.persistent_total(&ledger)?,
            ledger.scratch_bytes,
            ledger.staging_bytes,
        )?;
        if admitted > effective {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                format!(
                    "already admitted device/unified-memory allocations ({admitted} bytes) exceed the measured hardware working-set limit ({effective} bytes)"
                ),
            ));
        }
        self.limit_bytes.store(effective, Ordering::Release);
        Ok(effective)
    }

    #[must_use]
    pub const fn safety_reserve_bytes(&self) -> usize {
        self.reserved_bytes
    }

    /// Creates a project-generation view that shares global scratch/staging accounting while
    /// conservatively reserving one immutable resident generation in addition to the live set.
    pub fn pin_generation(&self, bytes: usize) -> Result<Self> {
        let pinned = Self {
            limit_bytes: Arc::clone(&self.limit_bytes),
            reserved_bytes: self.reserved_bytes,
            ledger: Arc::clone(&self.ledger),
            pinned_reservation: None,
            publishes_persistent_total: false,
            // Inherited, not dropped. A pinned view admits through the same ledger, so a view that
            // lost the probe would be a hole in the bound rather than a separate policy.
            host_available_bytes: self.host_available_bytes.clone(),
        };
        let mut ledger = pinned.lock_ledger()?;
        let next_pinned = ledger
            .pinned_generation_bytes
            .checked_add(bytes)
            .ok_or_else(memory_accounting_overflow)?;
        let persistent = ledger
            .persistent_graph_bytes
            .checked_add(next_pinned)
            .ok_or_else(memory_accounting_overflow)?;
        if pinned.total_with(
            &ledger,
            persistent,
            ledger.scratch_bytes,
            ledger.staging_bytes,
        )? > pinned.limit_bytes()
        {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "insufficient headroom to pin an immutable resident project generation",
            ));
        }
        ledger.pinned_generation_bytes = next_pinned;
        drop(ledger);
        Ok(Self {
            pinned_reservation: Some(Arc::new(PinnedGenerationReservation {
                bytes: AtomicUsize::new(bytes),
                ledger: Arc::clone(&pinned.ledger),
            })),
            ..pinned
        })
    }

    fn persistent_total(&self, ledger: &MemoryLedger) -> Result<usize> {
        ledger
            .persistent_graph_bytes
            .checked_add(ledger.pinned_generation_bytes)
            .ok_or_else(memory_accounting_overflow)
    }

    pub fn reserve_scratch(&self, bytes: usize) -> Result<ScratchReservation> {
        let mut ledger = self.lock_ledger()?;
        let next = ledger.scratch_bytes.checked_add(bytes).ok_or_else(|| {
            Error::new(
                ErrorCode::GpuAdmissionFailure,
                "device scratch byte accounting overflow",
            )
        })?;
        if self.total_with(
            &ledger,
            self.persistent_total(&ledger)?,
            next,
            ledger.staging_bytes,
        )? > self.limit_bytes()
        {
            // Scratch is held only for the duration of one query, so a rejection here means the
            // device is momentarily full rather than permanently unable to run this work. Marking
            // it non-retryable reported the most transient condition in the system as a permanent
            // failure, and clients gave up instead of waiting for the query in front of them.
            return Err(Error::retryable(
                ErrorCode::GpuAdmissionFailure,
                "insufficient admitted query scratch beside graph and model allocations",
                Some(10),
            ));
        }
        ledger.scratch_bytes = next;
        drop(ledger);
        Ok(ScratchReservation {
            bytes,
            governor: self.clone(),
        })
    }

    /// Reserves peak replacement bytes before allocating a new complete project image.
    pub fn reserve_staging(&self, bytes: usize) -> Result<StagingReservation> {
        let mut ledger = self.lock_ledger()?;
        let next = ledger.staging_bytes.checked_add(bytes).ok_or_else(|| {
            Error::new(
                ErrorCode::GpuAdmissionFailure,
                "device staging byte accounting overflow",
            )
        })?;
        if self.total_with(
            &ledger,
            self.persistent_total(&ledger)?,
            ledger.scratch_bytes,
            next,
        )? > self.limit_bytes()
        {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "insufficient peak memory for atomic project-image replacement beside model allocations",
            ));
        }
        ledger.staging_bytes = next;
        drop(ledger);
        Ok(StagingReservation {
            bytes,
            governor: self.clone(),
            released: false,
        })
    }

    /// Reserves a measured model/runtime allocation in the same ledger used by graph admission.
    pub fn reserve_shared(
        &self,
        class: SharedMemoryClass,
        bytes: usize,
    ) -> Result<SharedMemoryReservation> {
        let mut ledger = self.lock_ledger()?;
        let index = class.index();
        let next = ledger.shared_bytes[index]
            .checked_add(bytes)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::GpuAdmissionFailure,
                    "shared model/runtime byte accounting overflow",
                )
            })?;
        let previous = ledger.shared_bytes[index];
        ledger.shared_bytes[index] = next;
        let total = self.total_with(
            &ledger,
            self.persistent_total(&ledger)?,
            ledger.scratch_bytes,
            ledger.staging_bytes,
        )?;
        if total > self.limit_bytes() {
            ledger.shared_bytes[index] = previous;
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                format!(
                    "insufficient shared device/unified memory for {class:?}: requested {bytes} bytes"
                ),
            ));
        }
        drop(ledger);
        Ok(SharedMemoryReservation {
            class,
            bytes,
            governor: self.clone(),
            released: false,
        })
    }

    #[must_use]
    pub fn shares_budget_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.ledger, &other.ledger)
            && Arc::ptr_eq(&self.limit_bytes, &other.limit_bytes)
            && self.reserved_bytes == other.reserved_bytes
    }

    /// Converts one peak replacement reservation into the newly published persistent total.
    /// The caller must have fully built the replacement before calling this method; after it
    /// succeeds, publication is a single infallible owner swap.
    pub fn commit_staging(
        &mut self,
        mut reservation: StagingReservation,
        staged_bytes: usize,
        persistent_bytes: usize,
    ) -> Result<()> {
        self.validate_staging_reservation(&reservation, staged_bytes)?;
        let mut ledger = self.lock_ledger()?;
        let publication =
            self.prepare_staging_publication(&ledger, reservation.bytes, persistent_bytes)?;
        self.apply_staging_publication(&mut ledger, &publication)?;
        reservation.released = true;
        Ok(())
    }

    /// Keeps the old persistent owner and the complete staged replacement charged while an
    /// accelerator swaps owners, drops the old generation, and trims allocator pools.
    ///
    /// The publication closure runs while the ledger mutex is held. It must not reserve memory or
    /// otherwise re-enter this governor. Metal owner drop and device synchronization satisfy that
    /// rule: neither calls the database admission layer. If publication or synchronization fails
    /// after the owner swap, the staging charge is deliberately retained so later allocations fail
    /// closed instead of assuming allocator pages were released.
    pub fn publish_staging<T>(
        &mut self,
        mut reservation: StagingReservation,
        staged_bytes: usize,
        persistent_bytes: usize,
        publish_and_trim: impl FnOnce() -> Result<T>,
    ) -> Result<T> {
        self.validate_staging_reservation(&reservation, staged_bytes)?;
        let mut ledger = self.lock_ledger()?;
        let publication =
            self.prepare_staging_publication(&ledger, reservation.bytes, persistent_bytes)?;
        let published = match publish_and_trim() {
            Ok(published) => published,
            Err(error) => {
                reservation.released = true;
                return Err(error);
            }
        };
        if let Err(error) = self.apply_staging_publication(&mut ledger, &publication) {
            reservation.released = true;
            return Err(error);
        }
        reservation.released = true;
        Ok(published)
    }

    #[must_use]
    pub fn persistent_bytes(&self) -> usize {
        self.lock_ledger()
            .and_then(|ledger| self.persistent_total(&ledger))
            .unwrap_or(usize::MAX)
    }

    /// Bytes that a newly admitted query may reserve without exceeding the device budget.
    #[must_use]
    pub fn available_scratch_bytes(&self) -> usize {
        let Ok(ledger) = self.lock_ledger() else {
            return 0;
        };
        let Ok(total) = self.total_with(
            &ledger,
            self.persistent_total(&ledger).unwrap_or(usize::MAX),
            ledger.scratch_bytes,
            ledger.staging_bytes,
        ) else {
            return 0;
        };
        self.limit_bytes().saturating_sub(total)
    }

    #[must_use]
    pub fn snapshot(&self) -> DeviceMemorySnapshot {
        let Ok(ledger) = self.lock_ledger() else {
            return DeviceMemorySnapshot {
                limit_bytes: self.limit_bytes(),
                safety_reserve_bytes: self.reserved_bytes,
                ..DeviceMemorySnapshot::default()
            };
        };
        DeviceMemorySnapshot {
            limit_bytes: self.limit_bytes(),
            safety_reserve_bytes: self.reserved_bytes,
            persistent_graph_bytes: ledger.persistent_graph_bytes,
            pinned_generation_bytes: ledger.pinned_generation_bytes,
            model_weight_bytes: ledger.shared_bytes[SharedMemoryClass::ModelWeights.index()],
            embedding_weight_bytes: ledger.shared_bytes
                [SharedMemoryClass::EmbeddingWeights.index()],
            encoder_state_bytes: ledger.shared_bytes[SharedMemoryClass::EncoderState.index()],
            graph_memory_bytes: ledger.shared_bytes[SharedMemoryClass::GraphMemory.index()],
            device_overhead_bytes: ledger.shared_bytes[SharedMemoryClass::DeviceOverhead.index()],
            scratch_bytes: ledger.scratch_bytes,
            staging_bytes: ledger.staging_bytes,
        }
    }

    fn lock_ledger(&self) -> Result<std::sync::MutexGuard<'_, MemoryLedger>> {
        self.ledger
            .lock()
            .map_err(|_| Error::internal("device memory governor lock poisoned"))
    }

    fn validate_staging_reservation(
        &self,
        reservation: &StagingReservation,
        staged_bytes: usize,
    ) -> Result<()> {
        if !self.shares_budget_with(&reservation.governor)
            || reservation.released
            || staged_bytes > reservation.bytes
        {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "device replacement exceeded or mismatched its peak reservation",
            ));
        }
        Ok(())
    }

    fn prepare_staging_publication(
        &self,
        ledger: &MemoryLedger,
        reservation_bytes: usize,
        persistent_bytes: usize,
    ) -> Result<StagingPublication> {
        let remaining_staging = ledger
            .staging_bytes
            .checked_sub(reservation_bytes)
            .ok_or_else(|| Error::internal("device staging accounting underflow"))?;
        let (persistent_total, next_pinned, prior_pin) = if self.publishes_persistent_total {
            (
                persistent_bytes
                    .checked_add(ledger.pinned_generation_bytes)
                    .ok_or_else(memory_accounting_overflow)?,
                ledger.pinned_generation_bytes,
                None,
            )
        } else {
            let pin = self
                .pinned_reservation
                .as_ref()
                .ok_or_else(|| Error::internal("pinned governor has no generation reservation"))?;
            let prior = pin.bytes.load(Ordering::Acquire);
            let next = ledger
                .pinned_generation_bytes
                .checked_sub(prior)
                .and_then(|value| value.checked_add(persistent_bytes))
                .ok_or_else(memory_accounting_overflow)?;
            (
                ledger
                    .persistent_graph_bytes
                    .checked_add(next)
                    .ok_or_else(memory_accounting_overflow)?,
                next,
                Some((Arc::clone(pin), prior)),
            )
        };
        let next_total = self.total_with(
            ledger,
            persistent_total,
            ledger.scratch_bytes,
            remaining_staging,
        )?;
        if next_total > self.limit_bytes() {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "replacement project set no longer fits the admitted device budget",
            ));
        }
        // Second, independent bound. The check above compares a total against a limit derived once,
        // at backend construction, from the device's hardware working-set size — correct when the
        // process started and never revisited. This one asks whether the host has the bytes right
        // now. Measured on a 48 GB machine: 37.4 GB admitted against 19.3 GB actually free, swap
        // 91% used. Both must hold; the first stops a project outgrowing its device budget, the
        // second stops a correctly-budgeted allocation from being the one that exhausts a loaded
        // host.
        let current_total = self.total_with(
            ledger,
            self.persistent_total(ledger)?,
            ledger.scratch_bytes,
            ledger.staging_bytes,
        )?;
        if let Some((growth, available)) =
            self.host_would_be_exhausted(next_total.saturating_sub(current_total))
        {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                format!(
                    "admitting {growth} more bytes exceeds the {available} bytes the host has free; \
                     the device budget still allows it but the machine does not"
                ),
            ));
        }
        Ok(StagingPublication {
            persistent_bytes,
            remaining_staging,
            next_pinned,
            prior_pin,
        })
    }

    fn apply_staging_publication(
        &self,
        ledger: &mut MemoryLedger,
        publication: &StagingPublication,
    ) -> Result<()> {
        if self.publishes_persistent_total {
            ledger.persistent_graph_bytes = publication.persistent_bytes;
            ledger.staging_bytes = publication.remaining_staging;
            return Ok(());
        }
        if let Some((pin, prior)) = &publication.prior_pin {
            pin.bytes
                .compare_exchange(
                    *prior,
                    publication.persistent_bytes,
                    Ordering::AcqRel,
                    Ordering::Acquire,
                )
                .map_err(|_| {
                    Error::internal("pinned generation reservation changed during publication")
                })?;
        }
        ledger.pinned_generation_bytes = publication.next_pinned;
        ledger.staging_bytes = publication.remaining_staging;
        Ok(())
    }

    fn total_with(
        &self,
        ledger: &MemoryLedger,
        persistent_graph_bytes: usize,
        scratch_bytes: usize,
        staging_bytes: usize,
    ) -> Result<usize> {
        ledger.shared_bytes.iter().try_fold(
            self.reserved_bytes
                .checked_add(persistent_graph_bytes)
                .and_then(|value| value.checked_add(scratch_bytes))
                .and_then(|value| value.checked_add(staging_bytes))
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::GpuAdmissionFailure,
                        "device byte accounting overflow",
                    )
                })?,
            |total, bytes| {
                total.checked_add(*bytes).ok_or_else(|| {
                    Error::new(
                        ErrorCode::GpuAdmissionFailure,
                        "device byte accounting overflow",
                    )
                })
            },
        )
    }
}

fn memory_accounting_overflow() -> Error {
    Error::new(
        ErrorCode::GpuAdmissionFailure,
        "device persistent byte accounting overflow",
    )
}

struct StagingPublication {
    persistent_bytes: usize,
    remaining_staging: usize,
    next_pinned: usize,
    prior_pin: Option<(Arc<PinnedGenerationReservation>, usize)>,
}

#[derive(Debug)]
struct PinnedGenerationReservation {
    bytes: AtomicUsize,
    ledger: Arc<Mutex<MemoryLedger>>,
}

impl Drop for PinnedGenerationReservation {
    fn drop(&mut self) {
        let bytes = self.bytes.swap(0, Ordering::AcqRel);
        if bytes != 0
            && let Ok(mut ledger) = self.ledger.lock()
        {
            ledger.pinned_generation_bytes = ledger.pinned_generation_bytes.saturating_sub(bytes);
        }
    }
}

/// RAII reservation held while a replacement image is allocated and populated.
#[derive(Debug)]
pub struct StagingReservation {
    bytes: usize,
    governor: DeviceMemoryGovernor,
    released: bool,
}

impl StagingReservation {
    #[must_use]
    pub const fn bytes(&self) -> usize {
        self.bytes
    }

    /// Atomically grows or shrinks this replacement reservation before allocating the next stage.
    pub fn resize(&mut self, bytes: usize) -> Result<()> {
        if self.released || bytes == self.bytes {
            return Ok(());
        }
        let mut ledger = self.governor.lock_ledger()?;
        let without_self = ledger
            .staging_bytes
            .checked_sub(self.bytes)
            .ok_or_else(|| Error::internal("device staging reservation accounting underflow"))?;
        let replacement = without_self
            .checked_add(bytes)
            .ok_or_else(memory_accounting_overflow)?;
        if self.governor.total_with(
            &ledger,
            self.governor.persistent_total(&ledger)?,
            ledger.scratch_bytes,
            replacement,
        )? > self.governor.limit_bytes()
        {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "insufficient peak memory to grow the atomic replacement stage",
            ));
        }
        ledger.staging_bytes = replacement;
        self.bytes = bytes;
        Ok(())
    }
}

impl Drop for StagingReservation {
    fn drop(&mut self) {
        if !self.released
            && let Ok(mut ledger) = self.governor.ledger.lock()
        {
            ledger.staging_bytes = ledger.staging_bytes.saturating_sub(self.bytes);
            self.released = true;
        }
    }
}

/// RAII reservation preventing concurrent queries from overcommitting device memory.
#[derive(Debug)]
pub struct ScratchReservation {
    bytes: usize,
    governor: DeviceMemoryGovernor,
}

impl ScratchReservation {
    #[must_use]
    pub const fn bytes(&self) -> usize {
        self.bytes
    }

    /// Atomically changes one live query's scratch claim. A failed growth leaves the prior claim
    /// intact; shrinking and drop release exactly the bytes owned by this guard.
    pub fn resize(&mut self, bytes: usize) -> Result<()> {
        if bytes == self.bytes {
            return Ok(());
        }
        let mut ledger = self.governor.lock_ledger()?;
        let without_self = ledger
            .scratch_bytes
            .checked_sub(self.bytes)
            .ok_or_else(|| Error::internal("device scratch reservation accounting underflow"))?;
        let replacement = without_self.checked_add(bytes).ok_or_else(|| {
            Error::new(
                ErrorCode::GpuAdmissionFailure,
                "device scratch reservation accounting overflow",
            )
        })?;
        if self.governor.total_with(
            &ledger,
            self.governor.persistent_total(&ledger)?,
            replacement,
            ledger.staging_bytes,
        )? > self.governor.limit_bytes()
        {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "insufficient admitted query scratch beside graph and model allocations",
            ));
        }
        ledger.scratch_bytes = replacement;
        self.bytes = bytes;
        Ok(())
    }
}

impl Drop for ScratchReservation {
    fn drop(&mut self) {
        if let Ok(mut ledger) = self.governor.ledger.lock() {
            ledger.scratch_bytes = ledger.scratch_bytes.saturating_sub(self.bytes);
        }
    }
}

/// RAII ownership for measured model weights, model state, or graph-memory tensors.
#[derive(Debug)]
pub struct SharedMemoryReservation {
    class: SharedMemoryClass,
    bytes: usize,
    governor: DeviceMemoryGovernor,
    released: bool,
}

impl SharedMemoryReservation {
    #[must_use]
    pub const fn bytes(&self) -> usize {
        self.bytes
    }

    /// Replaces a conservative pre-load estimate with the allocation's measured resident bytes.
    pub fn resize(&mut self, measured_bytes: usize) -> Result<()> {
        if self.released || measured_bytes == self.bytes {
            return Ok(());
        }
        let mut ledger = self.governor.lock_ledger()?;
        let index = self.class.index();
        let without_self = ledger.shared_bytes[index]
            .checked_sub(self.bytes)
            .ok_or_else(|| Error::internal("shared memory reservation accounting underflow"))?;
        let replacement = without_self.checked_add(measured_bytes).ok_or_else(|| {
            Error::new(
                ErrorCode::GpuAdmissionFailure,
                "measured shared memory byte accounting overflow",
            )
        })?;
        let previous = ledger.shared_bytes[index];
        ledger.shared_bytes[index] = replacement;
        let total = self.governor.total_with(
            &ledger,
            self.governor.persistent_total(&ledger)?,
            ledger.scratch_bytes,
            ledger.staging_bytes,
        )?;
        if total > self.governor.limit_bytes() {
            ledger.shared_bytes[index] = previous;
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                format!(
                    "measured {class:?} allocation of {measured_bytes} bytes exceeds the shared device/unified-memory budget",
                    class = self.class
                ),
            ));
        }
        self.bytes = measured_bytes;
        Ok(())
    }
}

impl Drop for SharedMemoryReservation {
    fn drop(&mut self) {
        if !self.released
            && let Ok(mut ledger) = self.governor.ledger.lock()
        {
            let counter = &mut ledger.shared_bytes[self.class.index()];
            *counter = counter.saturating_sub(self.bytes);
            self.released = true;
        }
    }
}

#[cfg(test)]
mod governor_tests {

    use std::sync::Arc as TestArc;

    fn probe(bytes: Option<usize>) -> TestArc<dyn Fn() -> Option<usize> + Send + Sync> {
        TestArc::new(move || bytes)
    }

    #[test]
    fn a_probe_that_cannot_read_memory_never_refuses() {
        // The guarantee that keeps this from making admission more fragile than it was: a failed
        // read means "no opinion", not "no". Without it a transient sysctl failure would start
        // refusing work that fits.
        let mut governor = super::DeviceMemoryGovernor::new(1_000_000, 0);
        governor.set_host_available_probe(probe(None));
        assert!(governor.host_would_be_exhausted(usize::MAX).is_none());
    }

    #[test]
    fn no_probe_at_all_behaves_exactly_as_before() {
        let governor = super::DeviceMemoryGovernor::new(1_000_000, 0);
        assert!(governor.host_would_be_exhausted(usize::MAX).is_none());
    }

    #[test]
    fn growth_beyond_free_memory_is_refused_and_growth_within_it_is_not() {
        let mut governor = super::DeviceMemoryGovernor::new(1_000_000, 0);
        governor.set_host_available_probe(probe(Some(1_000)));
        assert_eq!(
            governor.host_would_be_exhausted(1_001),
            Some((1_001, 1_000))
        );
        assert!(governor.host_would_be_exhausted(1_000).is_none());
        assert!(governor.host_would_be_exhausted(1).is_none());
    }

    #[test]
    fn a_publication_that_grows_nothing_is_never_refused_for_host_pressure() {
        // Bounding the INCREASE rather than the total is what makes this correct: bytes already
        // admitted are resident and so are not part of what the host reports free. A total-based
        // check would double-count them and refuse a write that leaves memory unchanged.
        let mut governor = super::DeviceMemoryGovernor::new(1_000_000, 0);
        governor.set_host_available_probe(probe(Some(0)));
        assert!(governor.host_would_be_exhausted(0).is_none());
    }

    #[test]
    fn a_pinned_view_inherits_the_probe() {
        // A pinned generation admits through the same ledger; a view that dropped the probe would be
        // a hole in the bound rather than a different policy.
        let mut governor = super::DeviceMemoryGovernor::new(1_000_000, 0);
        governor.set_host_available_probe(probe(Some(10)));
        let pinned = governor.pin_generation(0).expect("pin");
        assert_eq!(pinned.host_would_be_exhausted(11), Some((11, 10)));
    }
    use std::{
        sync::{Arc, Barrier, mpsc},
        thread,
        time::Duration,
    };

    use super::{DeviceMemoryGovernor, SharedMemoryClass};
    use crate::Result;

    /// Concurrent lanes reserve and release model state against one ledger, repeatedly.
    ///
    /// Generation used to be serialized, so reserve/release could not interleave and a mispaired
    /// accounting path would never have shown itself. With lanes it can, and the failure mode is the
    /// quiet one: bytes that are never given back, so the device looks progressively fuller until
    /// admission starts refusing work that would in fact have fit. Ending exactly where it started
    /// is the whole assertion.
    #[test]
    fn concurrent_lanes_return_every_byte_they_reserve() -> Result<()> {
        const LANES: usize = 4;
        const ROUNDS: usize = 200;
        let governor = DeviceMemoryGovernor::new(1_000_000, 0);
        let before = governor.available_scratch_bytes();
        let ready = Arc::new(Barrier::new(LANES));
        let mut handles = Vec::with_capacity(LANES);
        for lane in 0..LANES {
            let governor = governor.clone();
            let ready = Arc::clone(&ready);
            handles.push(thread::spawn(move || -> Result<()> {
                ready.wait();
                for round in 0..ROUNDS {
                    // Sizes differ per lane and per round so releases cannot accidentally cancel out.
                    let bytes = 1_024 + lane * 97 + round % 13;
                    let reservation =
                        governor.reserve_shared(SharedMemoryClass::EncoderState, bytes)?;
                    // Growing mid-generation is what a graph-memory bank does as it expands.
                    let mut reservation = reservation;
                    reservation.resize(bytes + 512)?;
                    drop(reservation);
                }
                Ok(())
            }));
        }
        for handle in handles {
            handle.join().expect("lane thread finishes")?;
        }
        assert_eq!(
            governor.snapshot().encoder_state_bytes,
            0,
            "every lane returned its model state"
        );
        assert_eq!(
            governor.available_scratch_bytes(),
            before,
            "and the shared budget is exactly where it started"
        );
        Ok(())
    }

    #[test]
    fn graph_encoder_and_query_admission_share_one_atomic_budget() -> Result<()> {
        let mut governor = DeviceMemoryGovernor::new(1_000, 100);
        let model = governor.reserve_shared(SharedMemoryClass::ModelWeights, 400)?;
        governor.admit_persistent(400)?;
        assert_eq!(governor.available_scratch_bytes(), 100);
        assert!(governor.reserve_scratch(101).is_err());
        assert!(
            governor
                .reserve_shared(SharedMemoryClass::EmbeddingWeights, 101)
                .is_err()
        );

        drop(model);
        let embedding = governor.reserve_shared(SharedMemoryClass::EmbeddingWeights, 450)?;
        assert_eq!(governor.snapshot().embedding_weight_bytes, 450);
        drop(embedding);
        assert_eq!(governor.snapshot().embedding_weight_bytes, 0);
        Ok(())
    }

    #[test]
    fn measured_reservation_resize_is_atomic_and_rolls_back_on_pressure() -> Result<()> {
        let mut governor = DeviceMemoryGovernor::new(1_000, 100);
        governor.admit_persistent(400)?;
        let mut model = governor.reserve_shared(SharedMemoryClass::ModelWeights, 300)?;
        assert!(model.resize(600).is_err());
        assert_eq!(model.bytes(), 300);
        assert_eq!(governor.snapshot().model_weight_bytes, 300);
        model.resize(250)?;
        assert_eq!(governor.snapshot().model_weight_bytes, 250);
        Ok(())
    }

    #[test]
    fn pinned_generations_observe_encoder_and_staging_allocations() -> Result<()> {
        let mut governor = DeviceMemoryGovernor::new(1_000, 100);
        governor.admit_persistent(300)?;
        let _model = governor.reserve_shared(SharedMemoryClass::ModelWeights, 250)?;
        let pinned = governor.pin_generation(200)?;
        let _staging = pinned.reserve_staging(100)?;
        assert_eq!(pinned.available_scratch_bytes(), 50);
        assert!(pinned.reserve_scratch(51).is_err());
        Ok(())
    }

    #[test]
    fn concurrent_pinned_generations_are_aggregated_and_released_once() -> Result<()> {
        let mut governor = DeviceMemoryGovernor::new(1_000, 100);
        governor.admit_persistent(300)?;
        let first = governor.pin_generation(200)?;
        let first_clone = first.clone();
        let second = governor.pin_generation(250)?;
        assert_eq!(governor.snapshot().pinned_generation_bytes, 450);
        assert_eq!(governor.available_scratch_bytes(), 150);
        assert!(governor.pin_generation(151).is_err());

        drop(first);
        assert_eq!(governor.snapshot().pinned_generation_bytes, 450);
        drop(first_clone);
        assert_eq!(governor.snapshot().pinned_generation_bytes, 250);
        drop(second);
        assert_eq!(governor.snapshot().pinned_generation_bytes, 0);
        Ok(())
    }

    #[test]
    fn measured_hardware_cap_is_atomic_and_shared_by_every_clone() -> Result<()> {
        let mut governor = DeviceMemoryGovernor::new(2_000, 100);
        let peer = governor.clone();
        governor.admit_persistent(400)?;
        let _model = governor.reserve_shared(SharedMemoryClass::ModelWeights, 300)?;

        assert_eq!(peer.cap_limit(1_000)?, 1_000);
        assert_eq!(governor.limit_bytes(), 1_000);
        assert_eq!(governor.available_scratch_bytes(), 200);
        assert!(governor.reserve_scratch(201).is_err());
        assert!(peer.cap_limit(700).is_err());
        assert_eq!(governor.limit_bytes(), 1_000);
        Ok(())
    }

    #[test]
    fn scratch_resize_is_atomic_rolls_back_and_releases_exactly() -> Result<()> {
        let mut governor = DeviceMemoryGovernor::new(1_000, 100);
        governor.admit_persistent(300)?;
        let mut first = governor.reserve_scratch(200)?;
        let second = governor.reserve_scratch(300)?;

        assert!(first.resize(301).is_err());
        assert_eq!(first.bytes(), 200);
        assert_eq!(governor.snapshot().scratch_bytes, 500);
        first.resize(100)?;
        assert_eq!(governor.snapshot().scratch_bytes, 400);
        drop(second);
        assert_eq!(governor.snapshot().scratch_bytes, 100);
        drop(first);
        assert_eq!(governor.snapshot().scratch_bytes, 0);
        Ok(())
    }

    #[test]
    fn concurrent_scratch_reservations_cannot_both_overcommit() -> Result<()> {
        let mut governor = DeviceMemoryGovernor::new(1_000, 100);
        governor.admit_persistent(200)?;
        let start = Arc::new(Barrier::new(3));
        let release = Arc::new(Barrier::new(3));
        let (sent, received) = mpsc::channel();
        let mut joins = Vec::new();
        for _ in 0..2 {
            let peer = governor.clone();
            let start = Arc::clone(&start);
            let release = Arc::clone(&release);
            let sent = sent.clone();
            joins.push(thread::spawn(move || {
                start.wait();
                let reservation = peer.reserve_scratch(400).ok();
                sent.send(reservation.is_some()).expect("result receiver");
                release.wait();
                drop(reservation);
            }));
        }
        start.wait();
        let accepted = usize::from(received.recv().expect("first reservation result"))
            + usize::from(received.recv().expect("second reservation result"));
        assert_eq!(accepted, 1);
        release.wait();
        for join in joins {
            join.join().expect("scratch reservation worker");
        }
        assert_eq!(governor.snapshot().scratch_bytes, 0);
        Ok(())
    }

    #[test]
    fn staging_resize_precedes_allocation_and_rolls_back_on_pressure() -> Result<()> {
        let mut governor = DeviceMemoryGovernor::new(1_000, 100);
        governor.admit_persistent(300)?;
        let mut staging = governor.reserve_staging(200)?;
        let _model = governor.reserve_shared(SharedMemoryClass::ModelWeights, 300)?;
        assert!(staging.resize(301).is_err());
        assert_eq!(staging.bytes(), 200);
        assert_eq!(governor.snapshot().staging_bytes, 200);
        staging.resize(100)?;
        assert_eq!(governor.snapshot().staging_bytes, 100);
        Ok(())
    }

    #[test]
    fn publication_blocks_concurrent_reserve_until_old_owner_is_trimmed() -> Result<()> {
        let mut governor = DeviceMemoryGovernor::new(1_000, 100);
        governor.admit_persistent(200)?;
        let staging = governor.reserve_staging(500)?;
        let peer = governor.clone();
        let (entered_tx, entered_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let publication = thread::spawn(move || {
            governor.publish_staging(staging, 500, 500, || {
                entered_tx.send(()).expect("publication observer");
                release_rx.recv().expect("publication release");
                Ok(())
            })
        });
        entered_rx.recv().expect("publication entered");

        let (reserved_tx, reserved_rx) = mpsc::channel();
        let reserve = thread::spawn(move || {
            let result = peer.reserve_scratch(250);
            reserved_tx
                .send(result.is_ok())
                .expect("reservation observer");
            result
        });
        assert!(reserved_rx.recv_timeout(Duration::from_millis(50)).is_err());
        release_tx.send(()).expect("release publication");
        publication
            .join()
            .expect("publication worker")
            .expect("publication succeeds");
        assert!(reserved_rx.recv().expect("reservation result"));
        drop(
            reserve
                .join()
                .expect("reservation worker")
                .expect("reservation succeeds"),
        );
        Ok(())
    }

    #[test]
    fn unexplained_device_allocator_bytes_remain_in_shared_ledger() -> Result<()> {
        let governor = DeviceMemoryGovernor::new(1_000, 100);
        let overhead = governor.reserve_shared(SharedMemoryClass::DeviceOverhead, 75)?;
        assert_eq!(governor.snapshot().device_overhead_bytes, 75);
        assert_eq!(governor.snapshot().admitted_bytes(), 175);
        drop(overhead);
        assert_eq!(governor.snapshot().device_overhead_bytes, 0);
        Ok(())
    }
}
