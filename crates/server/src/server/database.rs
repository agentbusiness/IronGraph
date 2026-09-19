use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{
        Arc, OnceLock, Weak,
        atomic::{AtomicBool, AtomicU8, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use async_trait::async_trait;
use parking_lot::{Mutex, RwLock};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    Bookmark, CommitAcknowledgement, Error, ErrorCode, ProjectId, Result, ScalarValue,
    broker::{
        AmqpExchangeKind, BrokerCommand, BrokerCommit, BrokerCoordinator, BrokerReply,
        BrokerStateMachine, QueueKind, RetentionPolicy,
    },
    cypher::{
        BindCapabilities, Clause, DependencyStamp, ExecutionContext, ExecutionOutput,
        ExecutionStreamItem, QueryEngine, QueryResult, ResultValue, Statement, TextEmbedding,
        TransactionDependencies, bind, parse,
    },
    engine::{
        ActivationSubject, ApplicationWait, BackendSnapshot, CommandReservation,
        CommandReservationOutcome, CredentialRecord, EmbeddingProfileActivation,
        MutationApplyResult, MutationStateBackend, NodeIdentityPublic, ProcessId, SecurityState,
        SnapshotAttachment, StoreId, TransactionFence, WriteCommand, WriteRequest, WriteResponse,
        WriteRuntime,
    },
    gpu::{ExecutionBackend, ResidentProjectDelta, ResidentProjectImage, ResidentTemporalDelta},
    graph::{
        GraphMutation, GraphStore, IndexCatalog, ResolvedVectorMutation, StatisticsSnapshot,
        TemporalStore,
    },
    protocol::{
        BatchColumn, CatalogEvent, PathValue, QueryColumn, QueryExecutor, QueryRequest,
        QueryResultBudget, QueryStatistics, QueryStreamEvent, QueryTransaction, RelationshipValue,
        ResultNode, TypedValue,
    },
    storage::{
        AdmissionClass, AdmissionController, AdmissionLimits, ConnectionId, CowArc, MutationEntry,
        MutationKind, SegmentDescriptor, SegmentPin, SegmentStore,
    },
};
// Durability is the runtime-owned WAL plus this backend's periodic self-contained snapshot.

const DATABASE_SNAPSHOT_FORMAT: u16 = 6;
const DATABASE_SNAPSHOT_MAGIC: [u8; 8] = *b"IGDBS006";
const DATABASE_SNAPSHOT_COPY_BYTES: usize = 1024 * 1024;
const REQUEST_RESULT_INDEX_WINDOW: u64 = 65_536;
const MAX_REQUEST_RESULTS: usize = 16_384;
const MAX_REQUEST_RESULT_BYTES: u64 = 64 * 1024 * 1024;
const MAX_SINGLE_REQUEST_RESULT_BYTES: usize = 16 * 1024 * 1024;
const MAX_OPEN_TRANSACTIONS: usize = 256;
const MILLIS_PER_DAY: u64 = 86_400_000;

fn retention_from_days(days: Option<u32>) -> Result<RetentionPolicy> {
    let max_age_ms = days
        .map(|days| {
            if days == 0 {
                return Err(crate::Error::invalid_data(
                    "retention days must be positive",
                ));
            }
            u64::from(days).checked_mul(MILLIS_PER_DAY).ok_or_else(|| {
                crate::Error::invalid_data("retention days exceed the supported range")
            })
        })
        .transpose()?;
    Ok(RetentionPolicy {
        max_age_ms,
        max_bytes: None,
    })
}
const MAX_OPEN_TRANSACTIONS_PER_CONNECTION: usize = 16;
const TRANSACTION_MAINTENANCE_INTERVAL: Duration = Duration::from_millis(50);
const MIN_TRANSACTION_ACCOUNTED_BYTES: usize = 256;
const OPTIMIZER_STATISTICS_MAX_REVISION_LAG: u64 = 256;
/// Below this many canonical rows, a lazy full statistics pass is cheaper than maintenance wakeup.
/// At and above it, the periodic `spawn_blocking` warmer owns the rebuild so request workers do not.
const OPTIMIZER_STATISTICS_BACKGROUND_ROWS: usize = 100_000;
const BROKER_RECLAIM_BATCH_SEGMENTS: usize = 256;
const QUERY_STREAM_BATCH_ROWS: usize = 4_096;

/// The host (CPU) backend builds no on-device command buffer and reserves no fixed device scratch —
/// its only bound is system memory — so it is not subject to the device row budget above. This is
/// large enough never to be an artificial cap (a query that genuinely exceeds it exhausts RAM rather
/// than being refused), while staying far below the arithmetic limit so intermediate scratch
/// estimates cannot overflow `usize`.
const HOST_QUERY_EXECUTION_ROW_ADDRESS_SPACE: usize = u32::MAX as usize;

/// Conservative worst-case device scratch a single result row can require (the variable-path
/// frontier is the heaviest at ~5 KiB/row). Dividing the admitted device scratch by this scales the
/// GPU row budget to the machine's actual memory while keeping the peak reservation within it.
const DEVICE_SCRATCH_BYTES_PER_ROW: usize = 5368;

struct BrokerCommittedWrite {
    response: WriteResponse,
    application: ApplicationWait,
    reply: Option<BrokerReply>,
}

struct PendingBrokerSegment {
    _pin: SegmentPin,
    owners: BTreeSet<Uuid>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct TransactionPinKey {
    project: ProjectId,
    bookmark: Bookmark,
}

#[derive(Clone, Copy, Debug)]
struct TransactionPinUsage {
    device_bytes: usize,
    references: usize,
}

struct OpenTransactionRecord {
    connection: ConnectionId,
    encoded_bytes: usize,
    pin: TransactionPinKey,
    deadline: Instant,
    fence: TransactionFence,
    finalizing: bool,
    resource: Weak<TransactionResource>,
}

#[derive(Default)]
struct TransactionAdmissionState {
    records: BTreeMap<Uuid, OpenTransactionRecord>,
    connection_counts: BTreeMap<ConnectionId, usize>,
    encoded_bytes: usize,
    pins: BTreeMap<TransactionPinKey, TransactionPinUsage>,
    pinned_device_bytes: usize,
    device_limit_bytes: usize,
}

struct TransactionAdmissionRegistry {
    max_encoded_bytes: usize,
    state: Mutex<TransactionAdmissionState>,
}

impl TransactionAdmissionRegistry {
    fn new(max_encoded_bytes: usize) -> Result<Self> {
        if max_encoded_bytes < MIN_TRANSACTION_ACCOUNTED_BYTES {
            return Err(Error::invalid_data(
                "explicit-transaction admission byte limit is too small",
            ));
        }
        Ok(Self {
            max_encoded_bytes,
            state: Mutex::new(TransactionAdmissionState {
                device_limit_bytes: usize::MAX,
                ..TransactionAdmissionState::default()
            }),
        })
    }

    fn set_device_limit(&self, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "execution backend exposes no explicit-transaction device headroom",
            ));
        }
        let mut state = self.state.lock();
        if state.pinned_device_bytes > bytes {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "existing transaction pins exceed the execution device limit",
            ));
        }
        state.device_limit_bytes = bytes;
        Ok(())
    }

    const fn maximum_encoded_bytes(&self) -> usize {
        self.max_encoded_bytes
    }

    fn register(
        &self,
        id: Uuid,
        connection: ConnectionId,
        encoded_bytes: usize,
        pin: TransactionPinKey,
        pin_device_bytes: usize,
        deadline: Instant,
        fence: TransactionFence,
        resource: &Arc<TransactionResource>,
    ) -> Result<()> {
        if !connection.is_valid()
            || encoded_bytes < MIN_TRANSACTION_ACCOUNTED_BYTES
            || deadline <= Instant::now()
        {
            return Err(Error::invalid_data(
                "invalid explicit-transaction admission request",
            ));
        }
        let mut state = self.state.lock();
        if state.records.contains_key(&id) {
            return Err(Error::internal("duplicate explicit transaction identity"));
        }
        if state.records.len() >= MAX_OPEN_TRANSACTIONS
            || state
                .connection_counts
                .get(&connection)
                .copied()
                .unwrap_or_default()
                >= MAX_OPEN_TRANSACTIONS_PER_CONNECTION
        {
            return Err(transaction_admission_full(
                "explicit-transaction count admission is full",
            ));
        }
        let next_encoded = state
            .encoded_bytes
            .checked_add(encoded_bytes)
            .ok_or_else(|| Error::internal("transaction byte accounting overflow"))?;
        if next_encoded > self.max_encoded_bytes {
            return Err(transaction_admission_full(
                "explicit-transaction byte admission is full",
            ));
        }
        let new_pin = !state.pins.contains_key(&pin);
        let next_pinned = if new_pin {
            state
                .pinned_device_bytes
                .checked_add(pin_device_bytes)
                .ok_or_else(|| Error::internal("transaction device accounting overflow"))?
        } else {
            state.pinned_device_bytes
        };
        if next_pinned > state.device_limit_bytes {
            return Err(transaction_admission_full(
                "explicit-transaction device pin admission is full",
            ));
        }
        if let Some(existing) = state.pins.get_mut(&pin) {
            if existing.device_bytes != pin_device_bytes {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "shared transaction pin has inconsistent device bytes",
                ));
            }
            existing.references = existing
                .references
                .checked_add(1)
                .ok_or_else(|| Error::internal("transaction pin reference overflow"))?;
        } else {
            state.pins.insert(
                pin,
                TransactionPinUsage {
                    device_bytes: pin_device_bytes,
                    references: 1,
                },
            );
            state.pinned_device_bytes = next_pinned;
        }
        state.encoded_bytes = next_encoded;
        *state.connection_counts.entry(connection).or_default() += 1;
        state.records.insert(
            id,
            OpenTransactionRecord {
                connection,
                encoded_bytes,
                pin,
                deadline,
                fence,
                finalizing: false,
                resource: Arc::downgrade(resource),
            },
        );
        Ok(())
    }

    fn resize(&self, id: Uuid, encoded_bytes: usize) -> Result<()> {
        if encoded_bytes < MIN_TRANSACTION_ACCOUNTED_BYTES {
            return Err(Error::invalid_data(
                "explicit transaction encoded byte count is invalid",
            ));
        }
        let mut state = self.state.lock();
        let current = state.records.get(&id).ok_or_else(transaction_expired)?;
        if current.finalizing || current.deadline <= Instant::now() {
            return Err(Error::new(
                ErrorCode::TransactionExpired,
                "explicit transaction is expired or already finalizing",
            ));
        }
        let next = state
            .encoded_bytes
            .checked_sub(current.encoded_bytes)
            .and_then(|bytes| bytes.checked_add(encoded_bytes))
            .ok_or_else(|| Error::internal("transaction byte accounting overflow"))?;
        if next > self.max_encoded_bytes {
            return Err(transaction_admission_full(
                "explicit-transaction byte admission is full",
            ));
        }
        state.encoded_bytes = next;
        if let Some(record) = state.records.get_mut(&id) {
            record.encoded_bytes = encoded_bytes;
        }
        Ok(())
    }

    fn ensure_active(&self, id: Uuid, fence: TransactionFence) -> Result<()> {
        let state = self.state.lock();
        let record = state.records.get(&id).ok_or_else(transaction_expired)?;
        if record.finalizing || record.deadline <= Instant::now() {
            return Err(transaction_expired());
        }
        if record.fence != fence {
            return Err(transaction_sequencer_changed_error());
        }
        Ok(())
    }

    fn mark_finalizing(&self, id: Uuid) -> Result<()> {
        let mut state = self.state.lock();
        let record = state.records.get_mut(&id).ok_or_else(transaction_expired)?;
        if record.deadline <= Instant::now() || record.finalizing {
            return Err(transaction_expired());
        }
        record.finalizing = true;
        Ok(())
    }

    fn finish(&self, id: Uuid) {
        let removed = {
            let mut state = self.state.lock();
            remove_transaction_record(&mut state, id)
        };
        if let Some(record) = removed
            && let Some(resource) = record.resource.upgrade()
        {
            resource.close();
        }
    }

    fn maintain(&self, now: Instant, current: Option<(u64, ProcessId)>) {
        let mut state = self.state.lock();
        let expired = state
            .records
            .iter()
            .filter_map(|(id, record)| {
                if record.finalizing {
                    return None;
                }
                let reason = if record.deadline <= now {
                    TransactionTerminal::Expired
                } else if current.is_none_or(|(term, leader)| {
                    record.fence.term != term || record.fence.sequencer != leader
                }) {
                    TransactionTerminal::SequencerChanged
                } else {
                    return None;
                };
                Some((*id, reason, record.resource.clone()))
            })
            .collect::<Vec<_>>();
        for (id, reason, resource) in expired {
            // Keep accounting and the immutable pin until an in-flight query releases the
            // resource lock. The atomic terminal request makes that query discard its staged
            // overlay before returning, and the next maintenance tick performs removal.
            let removable = resource
                .upgrade()
                .is_none_or(|resource| resource.try_terminate(reason));
            if removable {
                remove_transaction_record(&mut state, id);
            }
        }
    }

    #[cfg(test)]
    fn usage(&self) -> (usize, usize, usize) {
        let state = self.state.lock();
        (
            state.records.len(),
            state.encoded_bytes,
            state.pinned_device_bytes,
        )
    }
}

fn remove_transaction_record(
    state: &mut TransactionAdmissionState,
    id: Uuid,
) -> Option<OpenTransactionRecord> {
    let record = state.records.remove(&id)?;
    state.encoded_bytes = state.encoded_bytes.saturating_sub(record.encoded_bytes);
    let remove_connection = if let Some(count) = state.connection_counts.get_mut(&record.connection)
    {
        *count = count.saturating_sub(1);
        *count == 0
    } else {
        false
    };
    if remove_connection {
        state.connection_counts.remove(&record.connection);
    }
    let remove_pin = if let Some(pin) = state.pins.get_mut(&record.pin) {
        pin.references = pin.references.saturating_sub(1);
        pin.references == 0
    } else {
        false
    };
    if remove_pin && let Some(pin) = state.pins.remove(&record.pin) {
        state.pinned_device_bytes = state.pinned_device_bytes.saturating_sub(pin.device_bytes);
    }
    Some(record)
}

fn transaction_admission_full(message: &'static str) -> Error {
    Error::retryable(ErrorCode::WriteAdmissionFull, message, Some(25))
}

fn transaction_expired() -> Error {
    Error::new(
        ErrorCode::TransactionExpired,
        "explicit transaction is expired or no longer active",
    )
}

fn transaction_sequencer_changed_error() -> Error {
    Error::retryable(
        ErrorCode::TransactionSequencerChanged,
        "explicit transaction was aborted because the local write sequencer changed",
        None,
    )
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct ProjectState {
    pub(super) id: ProjectId,
    pub(super) display_name: String,
    pub(super) graph: CowArc<GraphStore>,
    pub(super) temporal: CowArc<TemporalStore>,
    pub(super) predicate_versions: CowArc<BTreeMap<DependencyStamp, u64>>,
    pub(super) indexes: CowArc<IndexCatalog>,
    pub(super) next_node_id: u64,
    pub(super) next_edge_id: u64,
    #[serde(default)]
    pub(super) authority_revision: u64,
    /// Ephemeral and rebuildable; never part of durable state or a checkpoint.
    #[serde(skip)]
    pub(super) optimizer_statistics: OnceLock<Arc<StatisticsSnapshot>>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(super) struct DatabaseState {
    pub(super) projects: BTreeMap<ProjectId, Arc<ProjectState>>,
    pub(super) names: BTreeMap<String, ProjectId>,
    pub(super) broker: CowArc<BrokerStateMachine>,
    #[serde(default)]
    pub(super) embedding_activation: Option<EmbeddingProfileActivation>,
    #[serde(default)]
    pub(super) security: CowArc<SecurityState>,
    #[serde(default)]
    pub(super) applied: Bookmark,
    #[serde(default)]
    request_results: CowArc<BTreeMap<Uuid, RequestResultRecord>>,
    #[serde(default)]
    request_result_order: CowArc<BTreeMap<u64, Uuid>>,
    #[serde(default)]
    request_result_bytes: u64,
    #[serde(default)]
    last_payload_checksum: Option<[u8; 32]>,
    #[serde(default)]
    last_response: CowArc<Vec<u8>>,
}

#[derive(Default)]
struct OrderedMutationOverlay {
    /// Fully staged COW generations keyed by the exact ordered position they represent. Keeping
    /// each generation lets independent queued graph writes publish in order without a later
    /// reservation overwriting the state the earlier entry must publish.
    states: BTreeMap<u64, DatabaseState>,
    reservations: BTreeMap<u64, OrderedMutationReservation>,
}

struct OrderedMutationReservation {
    token: Uuid,
    broker_segments: Vec<SegmentDescriptor>,
    /// Exact digest of the sequencer-resolved command bytes. Publication compares this inexpensive
    /// digest instead of decoding and re-encoding a large staged batch solely to bind the overlay.
    staged_payload_digest: Option<[u8; 32]>,
    is_broker: bool,
    broker_publish_cursor: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct RequestResultRecord {
    index: u64,
    #[serde(default)]
    intent_digest: [u8; 32],
    response: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct DatabaseCheckpointManifest {
    format_version: u16,
    store_id: StoreId,
    included: Bookmark,
    state_bytes: u64,
    #[serde(default)]
    broker_segments: Vec<SegmentDescriptor>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum DatabaseMutation {
    CreateProject {
        id: ProjectId,
        display_name: String,
    },
    RenameProject {
        id: ProjectId,
        display_name: String,
    },
    DropProject {
        id: ProjectId,
        #[serde(default)]
        cascade: bool,
    },
    Graph {
        project: ProjectId,
        validation: MutationValidation,
        graph: Vec<GraphMutation>,
        temporal: Vec<PersistedTemporalMutation>,
        vectors: Vec<ResolvedVectorMutation>,
        administrative: Option<AdministrativeMutation>,
    },
    Broker {
        command: BrokerCommand,
    },
    EmbeddingProfile {
        command: EmbeddingProfileCommand,
    },
    Security {
        command: SecurityCommand,
    },
}

const COMPACT_UNIFORM_AMQP_MUTATION_MAGIC: &[u8; 5] = b"IGBU1";
const COMPACT_KAFKA_BATCH_MUTATION_MAGIC: &[u8; 5] = b"IGBK1";
const COMPACT_UNIFORM_VALUE_KAFKA_MUTATION_MAGIC: &[u8; 5] = b"IGBK2";

#[derive(Serialize)]
struct CompactUniformAmqpMutationRef<'a> {
    project: ProjectId,
    resolved_time_ms: i64,
    exchange: &'a str,
    routing_key: &'a str,
    mandatory: bool,
    properties: &'a BTreeMap<String, Vec<u8>>,
    headers: &'a BTreeMap<String, Vec<u8>>,
    payloads: &'a [Vec<u8>],
}

#[derive(Deserialize)]
struct CompactUniformAmqpMutation {
    project: ProjectId,
    resolved_time_ms: i64,
    exchange: String,
    routing_key: String,
    mandatory: bool,
    properties: BTreeMap<String, Vec<u8>>,
    headers: BTreeMap<String, Vec<u8>>,
    payloads: Vec<Vec<u8>>,
}

#[derive(Serialize)]
struct CompactKafkaBatchMutationRef<'a> {
    project: ProjectId,
    topic: &'a str,
    partition: i32,
    resolved_time_ms: i64,
    records: &'a [crate::broker::KafkaBatchRecord],
}

#[derive(Deserialize)]
struct CompactKafkaBatchMutation {
    project: ProjectId,
    topic: String,
    partition: i32,
    resolved_time_ms: i64,
    records: Vec<crate::broker::KafkaBatchRecord>,
}

#[derive(Serialize)]
struct CompactUniformValueKafkaMutationRef<'a> {
    project: ProjectId,
    topic: &'a str,
    partition: i32,
    resolved_time_ms: i64,
    payload: &'a [u8],
    create_times_ms: &'a [i64],
}

#[derive(Deserialize)]
struct CompactUniformValueKafkaMutation {
    project: ProjectId,
    topic: String,
    partition: i32,
    resolved_time_ms: i64,
    payload: Vec<u8>,
    create_times_ms: Vec<i64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum EmbeddingProfileCommand {
    Begin {
        project: ProjectId,
        profile: crate::graph::EmbeddingProfile,
    },
    Acknowledge {
        project: ProjectId,
        profile_hash: [u8; 32],
    },
    Activate {
        project: ProjectId,
        profile_hash: [u8; 32],
    },
    Abort {
        project: ProjectId,
        profile_hash: [u8; 32],
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum SecurityCommand {
    RegisterCredential {
        record: CredentialRecord,
    },
    RotateCredential {
        old_fingerprint: [u8; 32],
        replacement: CredentialRecord,
    },
    RevokeCredential {
        fingerprint: [u8; 32],
    },
    CleanupExpired,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct MutationValidation {
    snapshot: Bookmark,
    dependencies: TransactionDependencies,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub(super) enum AdministrativeMutation {
    InitializeSemantic {
        profile: crate::graph::EmbeddingProfile,
        graph_revision: u64,
    },
    CreateIndex {
        name: String,
        kind: String,
        label: String,
        properties: Vec<String>,
    },
    RebuildIndex {
        name: String,
    },
    DropIndex {
        name: String,
        /// Defaulted so a log record written before this field existed still replays, as the
        /// non-idempotent drop it was.
        #[serde(default)]
        if_exists: bool,
    },
    CreateConstraint {
        name: String,
        label: String,
        property: String,
    },
    DropConstraint {
        name: String,
        if_exists: bool,
    },
    DeclareTemporal {
        entity_kind: crate::types::EntityKind,
        label_or_type: String,
        property: String,
        scalar_type: String,
        retention_nanos: i64,
    },
    CreateRollup {
        name: String,
        label: String,
        property: String,
        hopping: bool,
        width_nanos: i64,
        every_nanos: Option<i64>,
        align_nanos: i64,
        timezone: Option<String>,
        aggregates: Vec<String>,
    },
    CreateEmbedding {
        name: String,
        label: crate::types::LabelId,
        source_property: crate::types::PropertyId,
        target_property: crate::types::PropertyId,
        model: String,
        profile: crate::graph::EmbeddingProfile,
        rows: Vec<(u64, Vec<u16>, u64)>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct PersistedTemporalMutation {
    entity_kind: crate::types::EntityKind,
    target: u64,
    sample: crate::graph::TemporalSample,
    uses_commit_time: bool,
}

pub(super) struct DatabaseInner {
    pub(super) state: RwLock<DatabaseState>,
    // Serializes local sequencer writes while the mutation is applied.
    pub(super) apply: Mutex<()>,
    pub(super) admission: AdmissionController,
    transactions: TransactionAdmissionRegistry,
    pub(super) request_timeout: Duration,
    pub(super) segments: SegmentStore,
    /// Ephemeral retry set for physically obsolete broker payload segments. Canonical liveness
    /// remains in broker references/checkpoints; exclusive startup discovers crash leftovers.
    retired_broker_segments: Mutex<BTreeSet<SegmentDescriptor>>,
    /// Uncommitted immutable payload publications awaiting their write entry. This bounded,
    /// ephemeral set is only a reclamation fence; the WAL remains the durable queue.
    pending_broker_segments: Mutex<BTreeMap<SegmentDescriptor, PendingBrokerSegment>>,
    /// Local speculative state through the highest durably appended command.
    /// It is shallow/COW, bounded by sequencer admission, and never checkpointed.
    ordered_overlay: Mutex<OrderedMutationOverlay>,
    broker_changes: tokio::sync::watch::Sender<u64>,
    text_embedding: RwLock<Option<Arc<dyn TextEmbedding>>>,
    semantic_initialization: parking_lot::Mutex<()>,
    pub(super) execution: RwLock<Option<Box<dyn ExecutionBackend>>>,
    selected_backend: OnceLock<crate::gpu::BackendKind>,
    store_id: StoreId,
    write_runtime: OnceLock<WriteBinding>,
    fatal_apply: AtomicBool,
}

struct WriteBinding {
    runtime: Weak<WriteRuntime>,
    handle: tokio::runtime::Handle,
}

/// Canonical database state machine behind the standalone WAL.
#[derive(Clone)]
pub struct Database(pub(super) Arc<DatabaseInner>);

impl Database {
    pub fn open(
        _directory: impl AsRef<Path>,
        _maximum_write_bytes: usize,
        _request_timeout: Duration,
    ) -> Result<Self> {
        Err(Error::invalid_data(
            "database startup requires the standalone bootstrap path",
        ))
    }

    pub fn open_backend(
        directory: impl AsRef<Path>,
        maximum_write_bytes: usize,
        request_timeout: Duration,
        identity: NodeIdentityPublic,
    ) -> Result<Self> {
        identity.validate()?;
        std::fs::create_dir_all(directory.as_ref())?;
        let admission = AdmissionController::new(AdmissionLimits {
            max_requests: 1_024,
            max_encoded_bytes: maximum_write_bytes,
            reserved_control_requests: 64,
            reserved_control_bytes: maximum_write_bytes / 16,
            max_requests_per_connection: 64,
            // One legal maximum-sized mutation must fit; request-count fairness prevents one
            // connection from occupying every slot concurrently.
            max_encoded_bytes_per_connection: maximum_write_bytes,
            retry_after_ms: 25,
        })?;
        let segments = SegmentStore::open(directory.as_ref(), maximum_write_bytes.max(1 << 20))?;
        let transactions = TransactionAdmissionRegistry::new(maximum_write_bytes)?;
        let (broker_changes, _) = tokio::sync::watch::channel(0);
        Ok(Self(Arc::new(DatabaseInner {
            state: RwLock::new(DatabaseState::default()),
            apply: Mutex::new(()),
            admission,
            transactions,
            request_timeout,
            segments,
            retired_broker_segments: Mutex::new(BTreeSet::new()),
            pending_broker_segments: Mutex::new(BTreeMap::new()),
            ordered_overlay: Mutex::new(OrderedMutationOverlay::default()),
            broker_changes,
            text_embedding: RwLock::new(None),
            semantic_initialization: parking_lot::Mutex::new(()),
            execution: RwLock::new(None),
            selected_backend: OnceLock::new(),
            store_id: identity.store_id,
            write_runtime: OnceLock::new(),
            fatal_apply: AtomicBool::new(false),
        })))
    }

    pub fn bind_runtime(&self, runtime: Weak<WriteRuntime>) -> Result<()> {
        let live = runtime
            .upgrade()
            .ok_or_else(|| Error::new(ErrorCode::Cancelled, "write runtime is unavailable"))?;
        if live.node().identity.store_id != self.0.store_id {
            return Err(Error::new(
                ErrorCode::AuthenticationFailed,
                "database and write-runtime store identities do not match",
            ));
        }
        let handle = tokio::runtime::Handle::try_current()
            .map_err(|_| Error::internal("write binding requires an active Tokio runtime"))?;
        self.0
            .write_runtime
            .set(WriteBinding {
                runtime,
                handle: handle.clone(),
            })
            .map_err(|_| Error::invalid_data("database write runtime is already bound"))?;
        let database = Arc::downgrade(&self.0);
        let write_runtime = Arc::downgrade(&live);
        handle.spawn(async move {
            loop {
                tokio::time::sleep(TRANSACTION_MAINTENANCE_INTERVAL).await;
                let (Some(database), Some(write_runtime)) =
                    (database.upgrade(), write_runtime.upgrade())
                else {
                    return;
                };
                // A live standalone runtime remains its own sequencer. `None` means unavailable
                // to the registry and would abort every transaction on the maintenance tick.
                let current = Some((
                    database.state.read().applied.term.max(1),
                    write_runtime.node_id(),
                ));
                database.transactions.maintain(Instant::now(), current);
            }
        });
        Ok(())
    }

    pub fn bind_text_embedding(&self, embedding: Arc<dyn TextEmbedding>) -> Result<()> {
        embedding.profile().validate()?;
        let mut current = self.0.text_embedding.write();
        if let Some(bound) = current.as_ref()
            && bound.profile() != embedding.profile()
        {
            return Err(Error::new(
                ErrorCode::EmbeddingProfileImmutable,
                "a different local embedding profile is already bound",
            ));
        }
        *current = Some(embedding);
        Ok(())
    }

    pub fn bind_execution_backend(&self, mut backend: Box<dyn ExecutionBackend>) -> Result<()> {
        let mut state = self.0.state.write();
        let cancellation = tokio_util::sync::CancellationToken::new();
        for project in state.projects.values_mut() {
            let project = Arc::make_mut(project);
            project.indexes.retry_failed_vectors();
            project.indexes.rebuild_vectors_with(|source, config| {
                backend.build_ivf_pq(source, config, &cancellation)
            })?;
            project.optimizer_statistics = OnceLock::new();
        }
        let mut resident_bytes = 0_usize;
        let project_ids = state.projects.keys().copied().collect::<Vec<_>>();
        for project_id in project_ids {
            let image = {
                let project = state.projects.get(&project_id).ok_or_else(|| {
                    Error::internal("project disappeared during execution admission")
                })?;
                ResidentProjectImage::build(
                    project.id,
                    state.applied,
                    &project.graph,
                    &project.temporal,
                    &project.indexes,
                )?
            };
            resident_bytes = resident_bytes
                .checked_add(image.resident_bytes())
                .ok_or_else(|| {
                    Error::new(ErrorCode::GpuAdmissionFailure, "resident byte overflow")
                })?;
            backend.admit_project(image)?;
            rebind_shared_project(&mut state, backend.as_ref(), project_id)?;
        }
        let transaction_device_limit = resident_bytes
            .checked_add(backend.available_query_scratch_bytes())
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::GpuAdmissionFailure,
                    "transaction device admission limit overflow",
                )
            })?
            .max(1);
        self.0
            .transactions
            .set_device_limit(transaction_device_limit)?;
        let mut current = self.0.execution.write();
        if current.is_some() {
            return Err(Error::invalid_data(
                "database execution backend is already bound",
            ));
        }
        let _ = self.0.selected_backend.set(backend.kind());
        *current = Some(backend);
        Ok(())
    }

    pub(super) fn text_embedding(&self) -> Result<Arc<dyn TextEmbedding>> {
        self.0.text_embedding.read().clone().ok_or_else(|| {
            Error::new(
                ErrorCode::EmbeddingUnavailable,
                "the local text embedding artifact is not bound",
            )
        })
    }

    pub(crate) fn ensure_project(
        &self,
        project: ProjectId,
        consistency: CommitAcknowledgement,
    ) -> Result<Bookmark> {
        let bookmark = self.consistency_barrier(consistency, None)?;
        if !self.0.state.read().projects.contains_key(&project) {
            return Err(Error::new(
                ErrorCode::ProjectNotFound,
                "configured project does not exist",
            ));
        }
        Ok(bookmark)
    }

    #[must_use]
    pub fn bookmark(&self) -> Bookmark {
        self.0.state.read().applied
    }

    /// Standalone durability: write a self-contained snapshot of the current committed state into
    /// `dir` and publish it as the latest. Best-effort — if a racing write advances the applied
    /// bookmark between reading it and freezing the state, the snapshot is retried at the newer
    /// bookmark. Broker/streaming payload segments are published beside the state file in a stable,
    /// content-addressed attachment directory before `LATEST` makes the snapshot visible.
    pub async fn standalone_snapshot(&self, dir: &Path) -> Result<Bookmark> {
        let snapshot_dir = dir.to_owned();
        tokio::task::spawn_blocking(move || std::fs::create_dir_all(snapshot_dir))
            .await
            .map_err(|error| {
                Error::internal(format!("snapshot directory task failed: {error}"))
            })??;
        let mut last_error = None;
        for _ in 0..16 {
            let bookmark = self.0.state.read().applied;
            let destination = dir.join(format!(
                "snapshot-{:020}-{:020}.igdb",
                bookmark.term, bookmark.index
            ));
            let candidate = destination.clone();
            let complete = tokio::task::spawn_blocking(move || -> Result<bool> {
                if !candidate.exists() {
                    return Ok(false);
                }
                match standalone_snapshot_is_complete(&candidate) {
                    Ok(true) => return Ok(true),
                    Ok(false) => tracing::warn!(
                        path = %candidate.display(),
                        "quarantining incomplete standalone snapshot before replacement"
                    ),
                    Err(error) => tracing::warn!(
                        path = %candidate.display(),
                        code = ?error.code,
                        message = %error.message,
                        "quarantining invalid standalone snapshot before replacement"
                    ),
                }
                quarantine_standalone_snapshot(&candidate)?;
                Ok(false)
            })
            .await
            .map_err(|error| Error::internal(format!("snapshot probe task failed: {error}")))??;
            if complete {
                return Ok(bookmark);
            }
            match self.build_snapshot(bookmark, &destination).await {
                Ok(snapshot) => {
                    let published_destination = destination.clone();
                    let published_dir = dir.to_owned();
                    tokio::task::spawn_blocking(move || -> Result<()> {
                        publish_standalone_snapshot_attachments(&published_destination, &snapshot)?;
                        if let Some(name) = published_destination
                            .file_name()
                            .and_then(|name| name.to_str())
                        {
                            crate::storage::atomic_write(
                                &published_dir.join("LATEST"),
                                name.as_bytes(),
                                false,
                            )?;
                        }
                        prune_standalone_snapshots(&published_dir, &published_destination);
                        Ok(())
                    })
                    .await
                    .map_err(|error| {
                        Error::internal(format!("snapshot publication task failed: {error}"))
                    })??;
                    return Ok(bookmark);
                }
                Err(error) => {
                    let failed_destination = destination.clone();
                    let _ = tokio::task::spawn_blocking(move || {
                        std::fs::remove_file(failed_destination)
                    })
                    .await;
                    last_error = Some(error);
                }
            }
        }
        Err(last_error.unwrap_or_else(|| Error::internal("standalone snapshot did not converge")))
    }

    /// Returns the newest validated checkpoint that must remain replayable beneath `latest`.
    /// WAL compaction stops at this older bookmark, so corrupting the newest snapshot still leaves
    /// a complete previous snapshot plus its ordered durable suffix.
    pub async fn standalone_wal_compaction_bookmark(
        &self,
        dir: &Path,
        latest: Bookmark,
    ) -> Result<Bookmark> {
        let directory = dir.to_owned();
        tokio::task::spawn_blocking(move || -> Result<Bookmark> {
            for path in standalone_snapshot_paths(&directory) {
                let Ok((bookmark, _)) = read_standalone_snapshot_manifest(&path) else {
                    continue;
                };
                if bookmark < latest && standalone_snapshot_is_complete(&path).unwrap_or(false) {
                    return Ok(bookmark);
                }
            }
            Ok(Bookmark::default())
        })
        .await
        .map_err(|error| Error::internal(format!("snapshot WAL-prefix task failed: {error}")))?
    }

    /// Standalone recovery: try `LATEST`, then older snapshots newest-first. Each candidate is
    /// validated before publication; a damaged candidate never turns the process into an empty
    /// database when a previous snapshot and durable WAL suffix are available.
    pub async fn standalone_recover(&self, dir: &Path) -> Result<Option<Bookmark>> {
        let recovery_dir = dir.to_owned();
        let candidates = tokio::task::spawn_blocking(move || {
            standalone_snapshot_paths_with_latest_first(&recovery_dir)
        })
        .await
        .map_err(|error| Error::internal(format!("snapshot discovery task failed: {error}")))?;
        if candidates.is_empty() {
            return Ok(None);
        }

        let mut last_error = None;
        for snapshot_path in candidates {
            let displayed = snapshot_path.display().to_string();
            let candidate =
                tokio::task::spawn_blocking(move || -> Result<(Bookmark, BackendSnapshot)> {
                    let (bookmark, broker_segments) =
                        read_standalone_snapshot_manifest(&snapshot_path)?;
                    let attachment_dir = standalone_snapshot_attachment_directory(&snapshot_path)?;
                    let mut attachments = Vec::with_capacity(broker_segments.len());
                    for descriptor in broker_segments {
                        attachments.push(SnapshotAttachment::new(
                            attachment_dir.join(hex::encode(descriptor.checksum)),
                            descriptor.bytes,
                            descriptor.checksum,
                        )?);
                    }
                    let bytes = std::fs::metadata(&snapshot_path)?.len();
                    let snapshot = BackendSnapshot::new(snapshot_path, bytes, attachments)?;
                    Ok((bookmark, snapshot))
                })
                .await
                .map_err(|error| {
                    Error::internal(format!("snapshot recovery task failed: {error}"))
                })?;
            let (bookmark, snapshot) = match candidate {
                Ok(candidate) => candidate,
                Err(error) => {
                    tracing::warn!(
                        path = %displayed,
                        code = ?error.code,
                        message = %error.message,
                        "standalone snapshot candidate is unreadable; trying previous"
                    );
                    last_error = Some(error);
                    continue;
                }
            };
            match self.install_snapshot(bookmark, &snapshot).await {
                Ok(()) => return Ok(Some(bookmark)),
                Err(error) => {
                    tracing::warn!(
                        path = %displayed,
                        code = ?error.code,
                        message = %error.message,
                        "standalone snapshot candidate was rejected; trying previous"
                    );
                    last_error = Some(error);
                }
            }
        }
        Err(last_error.unwrap_or_else(|| {
            Error::new(
                ErrorCode::CorruptStorage,
                "no standalone snapshot candidate could be recovered",
            )
        }))
    }

    fn commit_from(
        &self,
        mutation: DatabaseMutation,
        kind: MutationKind,
        project: ProjectId,
        request_id: Option<Uuid>,
        class: AdmissionClass,
        consistency: CommitAcknowledgement,
        connection_id: ConnectionId,
        transaction_fence: Option<TransactionFence>,
        request_timeout: Duration,
    ) -> Result<(Bookmark, Option<BrokerReply>)> {
        self.commit_scoped_with_timeout_from(
            mutation,
            kind,
            Some(project),
            request_id,
            class,
            consistency,
            request_timeout,
            connection_id,
            transaction_fence,
        )
        .map(|committed| (committed.response.bookmark, committed.reply))
    }

    fn remaining_query_write_time(&self, request: &QueryRequest) -> Result<Duration> {
        if request.cancellation.is_cancelled() {
            return Err(Error::new(
                ErrorCode::Cancelled,
                "query was cancelled before write admission",
            ));
        }
        request
            .deadline
            .map_or(Ok(self.0.request_timeout), |deadline| {
                deadline
                    .checked_duration_since(Instant::now())
                    .map(|remaining| remaining.min(self.0.request_timeout))
                    .ok_or_else(|| {
                        Error::retryable(
                            ErrorCode::DeadlineExceeded,
                            "query deadline expired before write admission",
                            None,
                        )
                    })
            })
    }

    pub(super) fn commit_unscoped(
        &self,
        mutation: DatabaseMutation,
        kind: MutationKind,
        request_id: Option<Uuid>,
        class: AdmissionClass,
        consistency: CommitAcknowledgement,
    ) -> Result<(Bookmark, Option<BrokerReply>)> {
        self.commit_scoped_from(
            mutation,
            kind,
            None,
            request_id,
            class,
            consistency,
            ConnectionId::new(),
            None,
        )
    }

    pub(crate) fn security_view(&self) -> Result<SecurityState> {
        self.ensure_apply_healthy()?;
        Ok((*self.0.state.read().security).clone())
    }

    pub(crate) fn authenticate_client(
        &self,
        fingerprint: [u8; 32],
        protocol: crate::engine::ProtocolScope,
    ) -> Result<CredentialRecord> {
        let now_millis = chrono::Utc::now().timestamp_millis();
        self.security_view()?
            .client_credentials()
            .authenticate(fingerprint, protocol, now_millis)
    }

    pub(crate) fn authorize_client(
        &self,
        fingerprint: [u8; 32],
        project: ProjectId,
        protocol: crate::engine::ProtocolScope,
        operation: crate::engine::OperationScope,
        layers: crate::engine::LayerScope,
    ) -> Result<crate::engine::AuthorizedCredential> {
        let now_millis = chrono::Utc::now().timestamp_millis();
        self.security_view()?.client_credentials().authorize(
            fingerprint,
            project,
            protocol,
            operation,
            layers,
            now_millis,
        )
    }

    pub(crate) fn pending_embedding_profile(&self) -> Result<Option<EmbeddingProfileActivation>> {
        self.ensure_apply_healthy()?;
        Ok(self.0.state.read().embedding_activation.clone())
    }

    pub(crate) fn begin_embedding_profile_activation(
        &self,
        project: ProjectId,
        profile: crate::graph::EmbeddingProfile,
        request_id: Uuid,
    ) -> Result<Bookmark> {
        self.commit_control(
            DatabaseMutation::EmbeddingProfile {
                command: EmbeddingProfileCommand::Begin { project, profile },
            },
            MutationKind::Policy,
            request_id,
            CommitAcknowledgement::Published,
        )
    }

    pub(crate) fn acknowledge_embedding_profile(
        &self,
        project: ProjectId,
        profile_hash: [u8; 32],
        request_id: Uuid,
    ) -> Result<Bookmark> {
        self.commit_control(
            DatabaseMutation::EmbeddingProfile {
                command: EmbeddingProfileCommand::Acknowledge {
                    project,
                    profile_hash,
                },
            },
            MutationKind::Policy,
            request_id,
            CommitAcknowledgement::Published,
        )
    }

    pub(crate) fn activate_embedding_profile(
        &self,
        project: ProjectId,
        profile_hash: [u8; 32],
        request_id: Uuid,
    ) -> Result<Bookmark> {
        self.commit_control(
            DatabaseMutation::EmbeddingProfile {
                command: EmbeddingProfileCommand::Activate {
                    project,
                    profile_hash,
                },
            },
            MutationKind::Policy,
            request_id,
            CommitAcknowledgement::Published,
        )
    }

    fn abort_embedding_profile_activation(
        &self,
        project: ProjectId,
        profile_hash: [u8; 32],
        request_id: Uuid,
    ) -> Result<Bookmark> {
        self.commit_control(
            DatabaseMutation::EmbeddingProfile {
                command: EmbeddingProfileCommand::Abort {
                    project,
                    profile_hash,
                },
            },
            MutationKind::Policy,
            request_id,
            CommitAcknowledgement::Published,
        )
    }

    /// Verifies and records only this authenticated process's readiness, then finalizes a barrier
    /// once every member in its exact configuration has acknowledged. Other nodes must run the
    /// same routine against their own locally loaded artifacts.
    pub(crate) fn maintain_local_artifact_readiness(&self) -> Result<bool> {
        self.ensure_apply_healthy()?;
        if let Some(pending) = self.pending_embedding_profile()? {
            let subject = pending.barrier().subject().clone();
            let ActivationSubject::EmbeddingProfile {
                project,
                profile_hash,
            } = subject;
            let local_profile = self.text_embedding()?.profile().clone();
            if local_profile.profile_hash != profile_hash {
                return Ok(false);
            }
            self.validate_embedding_profile_readiness(project, &local_profile)?;
            if !pending.barrier().is_ready() {
                self.acknowledge_embedding_profile(project, profile_hash, Uuid::new_v4())?;
            }
            if self
                .pending_embedding_profile()?
                .is_some_and(|activation| activation.barrier().is_ready())
            {
                self.activate_embedding_profile(project, profile_hash, Uuid::new_v4())?;
            }
            return Ok(true);
        }
        Ok(false)
    }

    /// Initializes graph-wide semantic vectors through the same ordered WAL path as graph writes.
    fn ensure_automatic_semantic(&self, project: ProjectId) -> Result<()> {
        let Some(embedding) = self.0.text_embedding.read().clone() else {
            return Ok(());
        };
        {
            let state = self.0.state.read();
            let project = state
                .projects
                .get(&project)
                .ok_or_else(|| Error::new(ErrorCode::ProjectNotFound, "project does not exist"))?;
            if project.indexes.contains(crate::graph::SEMANTIC_NODE_INDEX)
                && project
                    .indexes
                    .contains(crate::graph::SEMANTIC_RELATIONSHIP_INDEX)
                && !project.indexes.semantic_rebuild_needed()
            {
                return Ok(());
            }
        }
        let _initialization = self.0.semantic_initialization.lock();
        self.ensure_embedding_profile_activated(project)?;
        let (snapshot, bookmark) = {
            let state = self.0.state.read();
            (
                state.projects.get(&project).cloned().ok_or_else(|| {
                    Error::new(ErrorCode::ProjectNotFound, "project does not exist")
                })?,
                state.applied,
            )
        };
        let initialized = snapshot.indexes.contains(crate::graph::SEMANTIC_NODE_INDEX)
            && snapshot
                .indexes
                .contains(crate::graph::SEMANTIC_RELATIONSHIP_INDEX);
        if initialized && !snapshot.indexes.semantic_rebuild_needed() {
            return Ok(());
        }
        let vectors = if initialized {
            Vec::new()
        } else {
            resolve_semantic_texts(
                crate::graph::semantic_texts(&snapshot.graph)?,
                embedding.as_ref(),
                bookmark.index,
            )?
        };
        self.commit_scoped_from(
            DatabaseMutation::Graph {
                project,
                validation: MutationValidation {
                    snapshot: bookmark,
                    dependencies: TransactionDependencies::default(),
                },
                graph: Vec::new(),
                temporal: Vec::new(),
                vectors,
                administrative: Some(AdministrativeMutation::InitializeSemantic {
                    profile: embedding.profile().clone(),
                    graph_revision: snapshot.graph.revision(),
                }),
            },
            MutationKind::Graph,
            Some(project),
            Some(Uuid::new_v4()),
            AdmissionClass::Control,
            CommitAcknowledgement::Published,
            ConnectionId::new(),
            None,
        )?;
        Ok(())
    }

    /// Activates the selected local encoder before graph embeddings are resolved.
    fn ensure_embedding_profile_activated(&self, project: ProjectId) -> Result<()> {
        let profile = self.text_embedding()?.profile().clone();
        self.validate_embedding_profile_readiness(project, &profile)?;
        if self.project_embedding_profile(project)?.as_ref() == Some(&profile) {
            return Ok(());
        }

        match self.pending_embedding_profile()? {
            Some(pending)
                if pending.barrier().subject()
                    == &(ActivationSubject::EmbeddingProfile {
                        project,
                        profile_hash: profile.profile_hash,
                    }) => {}
            Some(_) => {
                return Err(Error::retryable(
                    ErrorCode::TransactionConflict,
                    "another artifact activation is already in progress",
                    Some(25),
                ));
            }
            None => {
                self.begin_embedding_profile_activation(project, profile.clone(), Uuid::new_v4())?;
            }
        }

        let deadline = Instant::now() + self.0.request_timeout;
        loop {
            self.maintain_local_artifact_readiness()?;
            if self.project_embedding_profile(project)?.as_ref() == Some(&profile) {
                return Ok(());
            }
            if Instant::now() >= deadline {
                if self.pending_embedding_profile()?.is_some_and(|pending| {
                    pending.barrier().subject()
                        == &(ActivationSubject::EmbeddingProfile {
                            project,
                            profile_hash: profile.profile_hash,
                        })
                }) {
                    let _ = self.abort_embedding_profile_activation(
                        project,
                        profile.profile_hash,
                        Uuid::new_v4(),
                    );
                }
                if self.project_embedding_profile(project)?.as_ref() == Some(&profile) {
                    return Ok(());
                }
                return Err(Error::retryable(
                    ErrorCode::DeadlineExceeded,
                    "embedding profile is waiting for every active node's verified artifacts",
                    Some(25),
                ));
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    fn project_embedding_profile(
        &self,
        project: ProjectId,
    ) -> Result<Option<crate::graph::EmbeddingProfile>> {
        self.ensure_apply_healthy()?;
        self.0
            .state
            .read()
            .projects
            .get(&project)
            .map(|state| state.indexes.profile().cloned())
            .ok_or_else(|| Error::new(ErrorCode::ProjectNotFound, "project does not exist"))
    }

    fn validate_embedding_profile_readiness(
        &self,
        project: ProjectId,
        profile: &crate::graph::EmbeddingProfile,
    ) -> Result<()> {
        profile.validate()?;
        if self.0.execution.read().is_none()
            || self.text_embedding()?.profile().profile_hash != profile.profile_hash
        {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "local embedding artifacts or resident execution backend are not ready",
            ));
        }
        let state = self.0.state.read();
        let project_state = state
            .projects
            .get(&project)
            .ok_or_else(|| Error::new(ErrorCode::ProjectNotFound, "project does not exist"))?;
        if project_state.indexes.profile() == Some(profile) {
            return Ok(());
        }
        if !project_state.indexes.embedding_profile_is_mutable() {
            return Err(Error::new(
                ErrorCode::EmbeddingProfileImmutable,
                "embedding profile is fixed by existing vector state",
            ));
        }
        Ok(())
    }

    fn write_binding(&self) -> Result<&WriteBinding> {
        self.0
            .write_runtime
            .get()
            .ok_or_else(|| Error::new(ErrorCode::Cancelled, "database write runtime is not bound"))
    }

    fn commit_control(
        &self,
        mutation: DatabaseMutation,
        kind: MutationKind,
        request_id: Uuid,
        consistency: CommitAcknowledgement,
    ) -> Result<Bookmark> {
        self.commit_unscoped(
            mutation,
            kind,
            Some(request_id),
            AdmissionClass::Control,
            consistency,
        )
        .map(|(bookmark, _)| bookmark)
    }

    fn commit_scoped_from(
        &self,
        mutation: DatabaseMutation,
        kind: MutationKind,
        project: Option<ProjectId>,
        request_id: Option<Uuid>,
        class: AdmissionClass,
        consistency: CommitAcknowledgement,
        connection_id: ConnectionId,
        transaction_fence: Option<TransactionFence>,
    ) -> Result<(Bookmark, Option<BrokerReply>)> {
        self.commit_scoped_with_timeout_from(
            mutation,
            kind,
            project,
            request_id,
            class,
            consistency,
            self.0.request_timeout,
            connection_id,
            transaction_fence,
        )
        .map(|committed| (committed.response.bookmark, committed.reply))
    }

    fn commit_scoped_with_timeout_from(
        &self,
        mutation: DatabaseMutation,
        kind: MutationKind,
        project: Option<ProjectId>,
        request_id: Option<Uuid>,
        class: AdmissionClass,
        _consistency: CommitAcknowledgement,
        request_timeout: Duration,
        connection_id: ConnectionId,
        transaction_fence: Option<TransactionFence>,
    ) -> Result<BrokerCommittedWrite> {
        self.ensure_apply_healthy()?;
        let payload =
            encode_database_mutation_bounded(&mutation, self.0.admission.maximum_encoded_bytes())?;
        let _permit = self.0.admission.try_admit(
            connection_id,
            class,
            payload.len(),
            Instant::now() + request_timeout,
        )?;
        let binding = self.0.write_runtime.get().ok_or_else(|| {
            Error::new(ErrorCode::Cancelled, "database write runtime is not bound")
        })?;
        let runtime = binding
            .runtime
            .upgrade()
            .ok_or_else(|| Error::new(ErrorCode::Cancelled, "write runtime is shutting down"))?;
        let timeout_millis = u64::try_from(request_timeout.as_millis())
            .map_err(|_| Error::invalid_data("database request timeout exceeds u64"))?;
        let request = WriteRequest {
            command: WriteCommand {
                kind,
                project_id: project,
                request_id,
                // The temporary sequencer overwrites this provisional field while holding its write
                // gate. A transport worker's clock is never committed.
                commit_time_millis: 0,
                payload,
            },
            timeout_millis,
            connection_id,
            admission_class: class,
            transaction_fence,
        };
        let committed = block_on_write(binding, runtime.write(request))?;
        let reply: Option<BrokerReply> =
            decode_database_value(&committed.response.payload, "database apply response")?;
        Ok(BrokerCommittedWrite {
            response: committed.response,
            application: committed.application,
            reply,
        })
    }

    // Standalone applies writes directly on this node, so the current applied bookmark IS the
    // linearizable point — no additional read barrier is needed.
    pub(super) fn consistency_barrier(
        &self,
        _consistency: CommitAcknowledgement,
        required: Option<Bookmark>,
    ) -> Result<Bookmark> {
        self.ensure_apply_healthy()?;
        let current = self.bookmark();
        if let Some(required) = required
            && current.index < required.index
        {
            return Err(Error::retryable(
                ErrorCode::StaleLocalRead,
                "required bookmark is not applied on the receiving node",
                Some(10),
            ));
        }
        Ok(current)
    }

    pub(super) fn ensure_apply_healthy(&self) -> Result<()> {
        if self.0.fatal_apply.load(Ordering::Acquire) {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "database state machine stopped after an invariant violation",
            ));
        }
        Ok(())
    }

    fn broker_reclamation_plan_locked(
        &self,
        state: &DatabaseState,
    ) -> (Vec<SegmentDescriptor>, BTreeSet<String>) {
        let live = state
            .broker
            .payload_segments()
            .into_iter()
            .map(|descriptor| descriptor.file_name)
            .collect::<BTreeSet<_>>();
        let mut retired = self.0.retired_broker_segments.lock();
        // A stale checkpoint import or content-address collision may enqueue a segment that is
        // live again. Prune it before applying the bounded batch so live no-ops cannot starve
        // genuinely obsolete files forever.
        retired.retain(|descriptor| !live.contains(&descriptor.file_name));
        let candidates = retired
            .iter()
            .take(BROKER_RECLAIM_BATCH_SEGMENTS)
            .cloned()
            .collect::<Vec<_>>();
        (candidates, live)
    }

    fn finish_broker_reclamation(&self, reclaimed: &[SegmentDescriptor]) -> Result<u64> {
        let mut retired = self.0.retired_broker_segments.lock();
        for descriptor in reclaimed {
            retired.remove(descriptor);
        }
        u64::try_from(reclaimed.len())
            .map_err(|_| Error::internal("reclaimed broker segment count exceeds u64"))
    }

    fn resolve_project(&self, requested: Option<ProjectId>, source: &str) -> Result<ProjectId> {
        if let Some(project) = requested {
            if self.0.state.read().projects.contains_key(&project) {
                return Ok(project);
            }
            return Err(Error::new(
                ErrorCode::ProjectNotFound,
                "project does not exist",
            ));
        }
        let query = parse(source)?;
        let Some(name) = query.project else {
            return Err(Error::new(
                ErrorCode::ProjectNotFound,
                "query does not select a project",
            ));
        };
        self.resolve_project_name(&name)
    }

    pub(super) fn resolve_project_name(&self, name: &str) -> Result<ProjectId> {
        self.0
            .state
            .read()
            .names
            .get(&normalize_name(name))
            .copied()
            .ok_or_else(|| Error::new(ErrorCode::ProjectNotFound, "project does not exist"))
    }

    pub(super) fn capture_project_execution(
        &self,
        project: ProjectId,
    ) -> Result<(
        Arc<ProjectState>,
        Bookmark,
        Option<Box<dyn ExecutionBackend>>,
    )> {
        // Apply holds the state write lock through resident publication. Taking these locks in
        // the same order therefore captures one exact host/device generation without blocking
        // execution for the lifetime of the query or transaction.
        let state = self.0.state.read();
        let snapshot = state
            .projects
            .get(&project)
            .cloned()
            .ok_or_else(|| Error::new(ErrorCode::ProjectNotFound, "project does not exist"))?;
        let bookmark = state.applied;
        let execution = self.0.execution.read();
        let pinned = match execution.as_deref() {
            Some(backend)
                if backend.resident_bookmark(project) == Some(bookmark)
                    && backend.resident_graph_revision(project)
                        == Some(snapshot.graph.revision()) =>
            {
                Some(backend.pin_project(project)?)
            }
            // The resident image lags the host snapshot (or there is no device). Under Apple unified
            // memory the host store is the authority — it holds the exact committed generation — so a
            // read simply executes on the host instead of failing. This is the single read policy:
            // a stale resident never raises GpuAdmissionFailure, whatever the publish cadence, which
            // is what lets device publication move off the write path without breaking reads.
            _ => None,
        };
        Ok((snapshot, bookmark, pinned))
    }

    /// Semantic queries wait for an admitted GPU generation instead of silently scoring on CPU.
    fn capture_semantic_execution(
        &self,
        project: ProjectId,
    ) -> Result<(
        Arc<ProjectState>,
        Bookmark,
        Option<Box<dyn ExecutionBackend>>,
    )> {
        let deadline = Instant::now() + self.0.request_timeout;
        loop {
            let captured = self.capture_project_execution(project)?;
            if captured.2.is_some()
                || self
                    .0
                    .selected_backend
                    .get()
                    .is_none_or(|kind| *kind == crate::gpu::BackendKind::Cpu)
            {
                return Ok(captured);
            }
            self.republish_resident_project(project, captured.0.graph.revision())?;
            if Instant::now() >= deadline {
                return Err(Error::retryable(
                    ErrorCode::GpuAdmissionFailure,
                    "semantic search is waiting for GPU publication",
                    Some(25),
                ));
            }
            std::thread::yield_now();
        }
    }

    /// Large projects whose resident device image lags the committed host graph.
    ///
    /// Returns projects covered by an explicitly enabled device-publication deferral policy.
    /// Publication is synchronous today, so this is normally empty; the catch-up worker remains a
    /// recovery guard rather than part of ordinary write behavior.
    pub(super) fn deferred_resident_projects(&self) -> Vec<(ProjectId, u64)> {
        let state = self.0.state.read();
        let execution = self.0.execution.read();
        let Some(backend) = execution.as_deref() else {
            return Vec::new();
        };
        state
            .projects
            .iter()
            .filter(|(id, project)| {
                backend.resident_graph_revision(**id) != Some(project.graph.revision())
                    || backend.resident_bookmark(**id) != Some(state.applied)
            })
            .map(|(id, project)| (*id, project.graph.revision()))
            .collect()
    }

    /// Rebuilds and republishes one large project's resident device image off the write path.
    ///
    /// This is the counterpart to the write-path deferral in [`should_defer_device_publish`]: the
    /// expensive O(N) image assembly runs here without holding the apply or state-write lock, so it
    /// never stalls the write gate. The assembled image is installed only if the project has not
    /// changed since `expected_revision` — a concurrent write makes it stale, and the caller retries
    /// once the project quiesces. Installing advances the device fence to the current applied
    /// bookmark, so the republished project (and every already-fresh project) serves reads from the
    /// GPU again. Returns whether an image was installed.
    pub(super) fn republish_resident_project(
        &self,
        project: ProjectId,
        expected_revision: u64,
    ) -> Result<bool> {
        // Capture a consistent, immutable snapshot (Arc/COW clone). Bail if the project already moved
        // past the revision the caller observed as quiescent.
        let (snapshot, applied) = {
            let state = self.0.state.read();
            let Some(project_state) = state.projects.get(&project) else {
                return Ok(false);
            };
            if project_state.graph.revision() != expected_revision {
                return Ok(false);
            }
            (Arc::clone(project_state), state.applied)
        };
        // O(N) host assembly + device image, entirely off the apply/state-write lock.
        let image = ResidentProjectImage::build(
            project,
            applied,
            &snapshot.graph,
            &snapshot.temporal,
            &snapshot.indexes,
        )?;
        // GPU admission may allocate, encode, submit, and synchronize a complete image. Move the
        // backend out while that happens: queries deliberately route to the canonical host graph
        // and writes see no resident backend to update, so neither waits behind a supposedly
        // background cold rebuild. The short publication section below restores the backend under
        // the normal apply -> state -> execution lock order and rechecks the captured revision.
        let mut backend = {
            let mut execution = self.0.execution.write();
            let Some(backend) = execution.take() else {
                return Ok(false);
            };
            backend
        };
        if let Err(error) = backend.admit_project(image) {
            let mut execution = self.0.execution.write();
            if execution.is_some() {
                return Err(Error::internal(
                    "execution backend was replaced during resident admission failure",
                ));
            }
            *execution = Some(backend);
            return Err(error);
        }

        let _apply = self.0.apply.lock();
        let mut state = self.0.state.write();
        let unchanged = state
            .projects
            .get(&project)
            .is_some_and(|current| current.graph.revision() == expected_revision);
        let fence = state.applied;
        let mut execution = self.0.execution.write();
        if execution.is_some() {
            return Err(Error::internal(
                "execution backend was replaced during resident admission",
            ));
        }
        let published = if unchanged {
            // Advance the device fence to the committed bookmark so this project — and every
            // project already resident at this fence — passes the read-time freshness check.
            // Projects changed while admission ran still fail the independent graph-revision
            // check and remain host-routed until their own quiescent republish.
            backend.advance_bookmark(fence);
            rebind_shared_project(&mut state, backend.as_ref(), project).map(|()| true)
        } else {
            Ok(false)
        };
        *execution = Some(backend);
        published
    }

    /// Large projects whose optimizer-statistics cache is cold, with their current graph revision.
    ///
    /// Statistics are invalidated on write and recomputed lazily on the next query. On a small graph
    /// that collect is microseconds, but on a large graph it is a full O(graph) pass that lands
    /// inline on whichever query trips the cache — a latency spike. Only large graphs are worth
    /// pre-warming off the query path, so the scan is bounded to them.
    pub(super) fn cold_statistics_projects(&self) -> Vec<(ProjectId, u64)> {
        let state = self.0.state.read();
        state
            .projects
            .iter()
            .filter(|(_, project)| {
                project
                    .graph
                    .node_slot_count()
                    .saturating_add(project.graph.edge_slot_count())
                    >= OPTIMIZER_STATISTICS_BACKGROUND_ROWS
                    && project.optimizer_statistics.get().is_none()
            })
            .map(|(id, project)| (*id, project.graph.revision()))
            .collect()
    }

    /// Pre-computes a large project's optimizer statistics off the query path, so the next query does
    /// not pay the O(graph) collect inline.
    ///
    /// Captures the live project `Arc` and populates its statistics cache through
    /// `OnceLock::get_or_init` — interior mutability, so no state-write lock is taken. If the project
    /// is unchanged since `expected_revision`, this warms exactly the cache the query will read
    /// (same `Arc`, same `OnceLock`); if a concurrent write replaced the project with a fresh, empty
    /// cache, this warmed the now-dropped old `Arc` and is harmlessly discarded. Returns whether it
    /// computed a fresh snapshot.
    pub(super) fn prewarm_optimizer_statistics(
        &self,
        project: ProjectId,
        expected_revision: u64,
    ) -> bool {
        let snapshot = {
            let state = self.0.state.read();
            match state.projects.get(&project) {
                Some(project_state)
                    if project_state.graph.revision() == expected_revision
                        && project_state.optimizer_statistics.get().is_none() =>
                {
                    Arc::clone(project_state)
                }
                _ => return false,
            }
        };
        let mut computed = false;
        snapshot.optimizer_statistics.get_or_init(|| {
            computed = true;
            Arc::new(StatisticsSnapshot::collect_project(
                &snapshot.graph,
                Some(&snapshot.temporal),
                Some(&snapshot.indexes),
            ))
        });
        computed
    }

    fn wait_for_captured_all(
        &self,
        consistency: CommitAcknowledgement,
        barrier: Bookmark,
        captured: Bookmark,
    ) -> Result<()> {
        if consistency == CommitAcknowledgement::Published && captured > barrier {
            self.consistency_barrier(CommitAcknowledgement::Published, Some(captured))?;
        }
        Ok(())
    }

    pub(super) fn execute_autocommit(
        &self,
        request: QueryRequest,
        emit: &mut dyn FnMut(QueryStreamEvent) -> Result<()>,
    ) -> Result<()> {
        request.validate()?;
        let parsed = parse(&request.query)?;
        let writes = statement_writes(&parsed.statement);
        let barrier_bookmark = self.consistency_barrier(
            if writes {
                CommitAcknowledgement::Published
            } else {
                request.consistency
            },
            request.bookmark,
        )?;
        match &parsed.statement {
            Statement::CheckReadOnly => {
                let candidate = request
                    .parameters
                    .get("statement")
                    .and_then(serde_json::Value::as_str)
                    .ok_or_else(|| {
                        Error::invalid_data("CHECK READ ONLY requires string parameter $statement")
                    })?;
                let candidate_project = self.resolve_project(request.project_id, candidate)?;
                let (snapshot, bookmark, _) = self.capture_project_execution(candidate_project)?;
                let candidate = parse(candidate)?;
                let checked = bind(
                    candidate,
                    snapshot.graph.catalog(),
                    BindCapabilities {
                        write: false,
                        schema: false,
                        knowledge_write: false,
                        workspace_write: false,
                        require_native_execution: false,
                    },
                )?;
                if !checked.read_only {
                    return Err(Error::invalid_data("statement is not read-only"));
                }
                emit(QueryStreamEvent::Schema {
                    request_id: request.request_id,
                    columns: Vec::new(),
                })?;
                return emit_read_summary(request.request_id, bookmark, 0, emit);
            }
            Statement::ShowProjects => {
                return self.show_projects(
                    request.request_id,
                    request.consistency,
                    barrier_bookmark,
                    emit,
                );
            }
            Statement::ImportDataset { name } => {
                return self.import_example_dataset(&request, name, emit);
            }
            Statement::ShowIndexes => {
                let project = self.resolve_project(request.project_id, &request.query)?;
                self.ensure_automatic_semantic(project)?;
                return self.show_indexes(
                    request.request_id,
                    project,
                    request.consistency,
                    barrier_bookmark,
                    emit,
                );
            }
            Statement::ShowConstraints => {
                let project = self.resolve_project(request.project_id, &request.query)?;
                return self.show_constraints(
                    request.request_id,
                    project,
                    request.consistency,
                    barrier_bookmark,
                    emit,
                );
            }
            Statement::ShowTopics => {
                let project = self.resolve_project(request.project_id, &request.query)?;
                return self.show_topics(request.request_id, project, emit);
            }
            Statement::ShowQueues => {
                let project = self.resolve_project(request.project_id, &request.query)?;
                return self.show_queues(request.request_id, project, emit);
            }
            Statement::ShowExchanges => {
                let project = self.resolve_project(request.project_id, &request.query)?;
                return self.show_exchanges(request.request_id, project, emit);
            }
            Statement::ShowConsumerLag => {
                let project = self.resolve_project(request.project_id, &request.query)?;
                return self.show_consumer_lag(request.request_id, project, emit);
            }
            Statement::CreateProject {
                name,
                if_not_exists,
            } => {
                return self.create_project(&request, name, *if_not_exists, emit);
            }
            Statement::AlterProjectRename { name, new_name } => {
                return self.rename_project(&request, name, new_name, emit);
            }
            Statement::DropProject {
                name,
                if_exists,
                cascade,
            } => {
                return self.drop_project(&request, name, *if_exists, *cascade, emit);
            }
            Statement::CreateTopic {
                name,
                partitions,
                retention_days,
            } => {
                let project = self.resolve_project(request.project_id, &request.query)?;
                return self.execute_broker_admin(
                    &request,
                    BrokerCommand::CreateTopic {
                        project,
                        name: name.clone(),
                        partitions: *partitions,
                        retention: retention_from_days(*retention_days)?,
                    },
                    emit,
                );
            }
            Statement::AlterTopicRetention {
                name,
                retention_days,
            } => {
                let project = self.resolve_project(request.project_id, &request.query)?;
                return self.execute_broker_admin(
                    &request,
                    BrokerCommand::SetTopicRetention {
                        project,
                        name: name.clone(),
                        retention: retention_from_days(Some(*retention_days))?,
                    },
                    emit,
                );
            }
            Statement::DropTopic { name } => {
                let project = self.resolve_project(request.project_id, &request.query)?;
                return self.execute_broker_admin(
                    &request,
                    BrokerCommand::DeleteTopic {
                        project,
                        name: name.clone(),
                    },
                    emit,
                );
            }
            Statement::ClearTopic { name } => {
                let project = self.resolve_project(request.project_id, &request.query)?;
                return self.execute_broker_admin(
                    &request,
                    BrokerCommand::ClearTopic {
                        project,
                        name: name.clone(),
                    },
                    emit,
                );
            }
            Statement::CreateQueue {
                name,
                stream,
                retention_days,
            } => {
                let project = self.resolve_project(request.project_id, &request.query)?;
                return self.execute_broker_admin(
                    &request,
                    BrokerCommand::CreateQueue {
                        project,
                        name: name.clone(),
                        kind: if *stream {
                            QueueKind::Stream
                        } else {
                            QueueKind::Classic
                        },
                        durable: true,
                        passive: false,
                        dead_letter_exchange: None,
                        dead_letter_routing_key: None,
                        retention: retention_from_days(*retention_days)?,
                        exclusive_owner: None,
                        auto_delete: false,
                    },
                    emit,
                );
            }
            Statement::AlterQueueRetention {
                name,
                retention_days,
            } => {
                let project = self.resolve_project(request.project_id, &request.query)?;
                return self.execute_broker_admin(
                    &request,
                    BrokerCommand::SetQueueRetention {
                        project,
                        name: name.clone(),
                        retention: retention_from_days(Some(*retention_days))?,
                    },
                    emit,
                );
            }
            Statement::DropQueue { name } => {
                let project = self.resolve_project(request.project_id, &request.query)?;
                return self.execute_broker_admin(
                    &request,
                    BrokerCommand::DeleteQueue {
                        project,
                        name: name.clone(),
                        if_unused: false,
                        if_empty: false,
                    },
                    emit,
                );
            }
            Statement::PurgeQueue { name } => {
                let project = self.resolve_project(request.project_id, &request.query)?;
                return self.execute_broker_admin(
                    &request,
                    BrokerCommand::PurgeQueue {
                        project,
                        name: name.clone(),
                    },
                    emit,
                );
            }
            Statement::CreateExchange { name, kind } => {
                let project = self.resolve_project(request.project_id, &request.query)?;
                let kind = match kind.to_ascii_lowercase().as_str() {
                    "direct" => AmqpExchangeKind::Direct,
                    "fanout" => AmqpExchangeKind::Fanout,
                    "topic" => AmqpExchangeKind::Topic,
                    _ => {
                        return Err(Error::invalid_data(
                            "exchange type must be DIRECT, FANOUT, or TOPIC",
                        ));
                    }
                };
                return self.execute_broker_admin(
                    &request,
                    BrokerCommand::CreateExchange {
                        project,
                        name: name.clone(),
                        kind,
                        durable: true,
                        passive: false,
                    },
                    emit,
                );
            }
            Statement::DropExchange { name } => {
                let project = self.resolve_project(request.project_id, &request.query)?;
                return self.execute_broker_admin(
                    &request,
                    BrokerCommand::DeleteExchange {
                        project,
                        name: name.clone(),
                        if_unused: false,
                    },
                    emit,
                );
            }
            Statement::BindQueue {
                queue,
                exchange,
                routing_key,
            } => {
                let project = self.resolve_project(request.project_id, &request.query)?;
                return self.execute_broker_admin(
                    &request,
                    BrokerCommand::BindQueue {
                        project,
                        exchange: exchange.clone(),
                        queue: queue.clone(),
                        routing_key: routing_key.clone(),
                    },
                    emit,
                );
            }
            Statement::UnbindQueue {
                queue,
                exchange,
                routing_key,
            } => {
                let project = self.resolve_project(request.project_id, &request.query)?;
                return self.execute_broker_admin(
                    &request,
                    BrokerCommand::UnbindQueue {
                        project,
                        exchange: exchange.clone(),
                        queue: queue.clone(),
                        routing_key: routing_key.clone(),
                    },
                    emit,
                );
            }
            _ => {}
        }
        let project = self.resolve_project(request.project_id, &request.query)?;
        self.ensure_automatic_semantic(project)?;
        if matches!(&parsed.statement, Statement::CreateEmbedding(_)) {
            self.ensure_embedding_profile_activated(project)?;
        }
        let semantic_search = matches!(&parsed.statement, Statement::Query(body) if body.clauses.iter().chain(body.unions.iter().flat_map(|branch| &branch.body)).any(|clause| matches!(clause, crate::cypher::Clause::Search(_))));
        let (snapshot, captured_bookmark, captured_execution) = if semantic_search {
            self.capture_semantic_execution(project)?
        } else if writes {
            let state = self.0.state.read();
            let snapshot =
                state.projects.get(&project).cloned().ok_or_else(|| {
                    Error::new(ErrorCode::ProjectNotFound, "project does not exist")
                })?;
            (snapshot, state.applied, None)
        } else {
            self.capture_project_execution(project)?
        };
        let text_embedding = self.0.text_embedding.read().clone();
        let capabilities = full_capabilities();
        let read_only = bind(parsed, snapshot.graph.catalog(), capabilities)?.read_only;
        if read_only {
            self.wait_for_captured_all(request.consistency, barrier_bookmark, captured_bookmark)?;
            emit_catalog(
                Some(project),
                captured_bookmark,
                snapshot.graph.catalog(),
                &snapshot.indexes,
                emit,
            )?;
            let mut sequence = 0_u64;
            let mut rows = 0_u64;
            let output = {
                let mut stream = |item| {
                    emit_execution_stream_item(
                        request.request_id,
                        item,
                        &mut sequence,
                        &mut rows,
                        emit,
                    )
                };
                execute_on_project_streaming(
                    &snapshot,
                    &request,
                    captured_bookmark,
                    captured_bookmark.index,
                    capabilities,
                    text_embedding.as_deref(),
                    captured_execution.as_deref(),
                    &mut stream,
                )?
            };
            emit_query_summary(request.request_id, output.result, rows, emit)
        } else {
            let (snapshot, planning_bookmark, planning_execution) = if semantic_search {
                (snapshot, captured_bookmark, captured_execution)
            } else {
                let state = self.0.state.read();
                let snapshot = state.projects.get(&project).cloned().ok_or_else(|| {
                    Error::new(ErrorCode::ProjectNotFound, "project does not exist")
                })?;
                (snapshot, state.applied, None)
            };
            let next_index = planning_bookmark
                .index
                .checked_add(1)
                .ok_or_else(|| Error::internal("log index exhausted"))?;
            let mut output = {
                // Semantic reads in a write statement use the pinned execution generation too.
                // The canonical mutation and its derived vectors still commit together, then
                // publish one sparse resident delta.
                execute_on_project(
                    &snapshot,
                    &request,
                    planning_bookmark,
                    next_index,
                    capabilities,
                    text_embedding.as_deref(),
                    planning_execution.as_deref(),
                )?
            };
            let administrative = administrative_mutation(
                output.administrative.take(),
                &snapshot,
                &mut output.graph_mutations,
                text_embedding.as_deref(),
                next_index,
            )?;
            let vectors = resolve_embedding_mutations(
                &snapshot,
                &[],
                &output.graph_mutations,
                text_embedding.as_deref(),
                next_index,
            )?;
            let mutation = DatabaseMutation::Graph {
                project,
                validation: MutationValidation {
                    snapshot: planning_bookmark,
                    dependencies: output.dependencies.clone(),
                },
                graph: output.graph_mutations,
                temporal: output
                    .temporal_mutations
                    .into_iter()
                    .map(|mutation| PersistedTemporalMutation {
                        entity_kind: mutation.entity_kind,
                        target: mutation.target,
                        sample: mutation.sample,
                        uses_commit_time: mutation.uses_commit_time,
                    })
                    .collect(),
                vectors,
                administrative,
            };
            let (bookmark, _) = self.commit_from(
                mutation,
                MutationKind::Graph,
                project,
                Some(request.request_id),
                AdmissionClass::Client,
                request.consistency,
                request.connection_id,
                None,
                self.remaining_query_write_time(&request)?,
            )?;
            output.result.bookmark = bookmark;
            let current = self.0.state.read();
            let project_state = current
                .projects
                .get(&project)
                .ok_or_else(|| Error::internal("committed project disappeared"))?;
            emit_result(
                request.request_id,
                output.result,
                project_state.graph.catalog(),
                &project_state.indexes,
                emit,
            )
        }
    }

    fn show_projects(
        &self,
        request_id: Uuid,
        consistency: CommitAcknowledgement,
        barrier: Bookmark,
        emit: &mut dyn FnMut(QueryStreamEvent) -> Result<()>,
    ) -> Result<()> {
        let (values, bookmark) = {
            let state = self.0.state.read();
            (
                state
                    .projects
                    .values()
                    .map(|project| (project.id.to_string(), project.display_name.clone()))
                    .collect::<Vec<_>>(),
                state.applied,
            )
        };
        self.wait_for_captured_all(consistency, barrier, bookmark)?;
        emit(QueryStreamEvent::Schema {
            request_id,
            columns: vec![
                QueryColumn {
                    name: "project_id".to_owned(),
                    value_type: "STRING".to_owned(),
                    nullable: false,
                },
                QueryColumn {
                    name: "display_name".to_owned(),
                    value_type: "STRING".to_owned(),
                    nullable: false,
                },
            ],
        })?;
        emit(QueryStreamEvent::Batch {
            request_id,
            sequence: 0,
            row_count: values.len() as u64,
            columns: vec![
                BatchColumn {
                    name: "project_id".to_owned(),
                    value_type: "STRING".to_owned(),
                    values: values
                        .iter()
                        .map(|(id, _)| TypedValue::String(id.clone()))
                        .collect(),
                },
                BatchColumn {
                    name: "display_name".to_owned(),
                    value_type: "STRING".to_owned(),
                    values: values
                        .iter()
                        .map(|(_, name)| TypedValue::String(name.clone()))
                        .collect(),
                },
            ],
        })?;
        emit(QueryStreamEvent::Summary {
            request_id,
            bookmark,
            statistics: QueryStatistics {
                rows: values.len() as u64,
                ..QueryStatistics::default()
            },
            truncated: false,
            truncation_reason: None,
        })
    }

    fn show_indexes(
        &self,
        request_id: Uuid,
        project: ProjectId,
        consistency: CommitAcknowledgement,
        barrier: Bookmark,
        emit: &mut dyn FnMut(QueryStreamEvent) -> Result<()>,
    ) -> Result<()> {
        let (rows, bookmark) = {
            let state = self.0.state.read();
            let project = state
                .projects
                .get(&project)
                .ok_or_else(|| Error::new(ErrorCode::ProjectNotFound, "project does not exist"))?;
            (
                project.indexes.statuses().collect::<Vec<_>>(),
                state.applied,
            )
        };
        self.wait_for_captured_all(consistency, barrier, bookmark)?;
        emit(QueryStreamEvent::Schema {
            request_id,
            columns: vec![
                QueryColumn {
                    name: "name".to_owned(),
                    value_type: "STRING".to_owned(),
                    nullable: false,
                },
                QueryColumn {
                    name: "kind".to_owned(),
                    value_type: "STRING".to_owned(),
                    nullable: false,
                },
                QueryColumn {
                    name: "state".to_owned(),
                    value_type: "STRING".to_owned(),
                    nullable: false,
                },
                QueryColumn {
                    name: "diagnostic".to_owned(),
                    value_type: "STRING".to_owned(),
                    nullable: true,
                },
            ],
        })?;
        emit(QueryStreamEvent::Batch {
            request_id,
            sequence: 0,
            row_count: rows.len() as u64,
            columns: vec![
                BatchColumn {
                    name: "name".to_owned(),
                    value_type: "STRING".to_owned(),
                    values: rows
                        .iter()
                        .map(|status| TypedValue::String(status.name.clone()))
                        .collect(),
                },
                BatchColumn {
                    name: "kind".to_owned(),
                    value_type: "STRING".to_owned(),
                    values: rows
                        .iter()
                        .map(|status| {
                            TypedValue::String(format!("{:?}", status.kind).to_uppercase())
                        })
                        .collect(),
                },
                BatchColumn {
                    name: "state".to_owned(),
                    value_type: "STRING".to_owned(),
                    values: rows
                        .iter()
                        .map(|status| {
                            TypedValue::String(format!("{:?}", status.state).to_uppercase())
                        })
                        .collect(),
                },
                BatchColumn {
                    name: "diagnostic".to_owned(),
                    value_type: "STRING".to_owned(),
                    values: rows
                        .iter()
                        .map(|status| {
                            status
                                .diagnostic
                                .clone()
                                .map_or(TypedValue::Null, TypedValue::String)
                        })
                        .collect(),
                },
            ],
        })?;
        emit(QueryStreamEvent::Summary {
            request_id,
            bookmark,
            statistics: QueryStatistics {
                rows: rows.len() as u64,
                ..QueryStatistics::default()
            },
            truncated: false,
            truncation_reason: None,
        })
    }

    fn show_constraints(
        &self,
        request_id: Uuid,
        project: ProjectId,
        consistency: CommitAcknowledgement,
        barrier: Bookmark,
        emit: &mut dyn FnMut(QueryStreamEvent) -> Result<()>,
    ) -> Result<()> {
        let (rows, bookmark) = {
            let state = self.0.state.read();
            let project = state
                .projects
                .get(&project)
                .ok_or_else(|| Error::new(ErrorCode::ProjectNotFound, "project does not exist"))?;
            let rows = project
                .indexes
                .constraint_definitions()
                .map(|definition| {
                    let label = project
                        .graph
                        .catalog()
                        .label_name(definition.label)
                        .ok_or_else(|| {
                            Error::new(ErrorCode::CorruptStorage, "constraint label is missing")
                        })?
                        .to_owned();
                    let property = definition
                        .properties
                        .first()
                        .and_then(|property| project.graph.catalog().property_name(*property))
                        .ok_or_else(|| {
                            Error::new(ErrorCode::CorruptStorage, "constraint property is missing")
                        })?
                        .to_owned();
                    Ok((definition.name.clone(), label, property))
                })
                .collect::<Result<Vec<_>>>()?;
            (rows, state.applied)
        };
        self.wait_for_captured_all(consistency, barrier, bookmark)?;
        emit(QueryStreamEvent::Schema {
            request_id,
            columns: ["name", "label", "property"]
                .into_iter()
                .map(|name| QueryColumn {
                    name: name.to_owned(),
                    value_type: "STRING".to_owned(),
                    nullable: false,
                })
                .collect(),
        })?;
        emit(QueryStreamEvent::Batch {
            request_id,
            sequence: 0,
            row_count: rows.len() as u64,
            columns: [
                (
                    "name",
                    rows.iter().map(|row| row.0.clone()).collect::<Vec<_>>(),
                ),
                (
                    "label",
                    rows.iter().map(|row| row.1.clone()).collect::<Vec<_>>(),
                ),
                (
                    "property",
                    rows.iter().map(|row| row.2.clone()).collect::<Vec<_>>(),
                ),
            ]
            .into_iter()
            .map(|(name, values)| BatchColumn {
                name: name.to_owned(),
                value_type: "STRING".to_owned(),
                values: values.into_iter().map(TypedValue::String).collect(),
            })
            .collect(),
        })?;
        emit(QueryStreamEvent::Summary {
            request_id,
            bookmark,
            statistics: QueryStatistics {
                rows: rows.len() as u64,
                ..QueryStatistics::default()
            },
            truncated: false,
            truncation_reason: None,
        })
    }

    fn show_topics(
        &self,
        request_id: Uuid,
        project: ProjectId,
        emit: &mut dyn FnMut(QueryStreamEvent) -> Result<()>,
    ) -> Result<()> {
        let (rows, bookmark) = {
            let state = self.0.state.read();
            (state.broker.topic_metrics(project), state.applied)
        };
        emit(QueryStreamEvent::Schema {
            request_id,
            columns: [
                ("name", "STRING"),
                ("partition", "INTEGER"),
                ("base_offset", "INTEGER"),
                ("next_offset", "INTEGER"),
                ("record_count", "INTEGER"),
                ("retained_bytes", "INTEGER"),
                ("retention_ms", "INTEGER"),
            ]
            .into_iter()
            .map(|(name, value_type)| QueryColumn {
                name: name.to_owned(),
                value_type: value_type.to_owned(),
                nullable: name == "retention_ms",
            })
            .collect(),
        })?;
        emit(QueryStreamEvent::Batch {
            request_id,
            sequence: 0,
            row_count: rows.len() as u64,
            columns: vec![
                string_batch("name", rows.iter().map(|row| row.name.clone())),
                integer_batch("partition", rows.iter().map(|row| i64::from(row.partition))),
                integer_batch(
                    "base_offset",
                    rows.iter().map(|row| saturating_i64(row.base_offset)),
                ),
                integer_batch(
                    "next_offset",
                    rows.iter().map(|row| saturating_i64(row.next_offset)),
                ),
                integer_batch(
                    "record_count",
                    rows.iter().map(|row| saturating_i64(row.record_count)),
                ),
                integer_batch(
                    "retained_bytes",
                    rows.iter().map(|row| saturating_i64(row.retained_bytes)),
                ),
                BatchColumn {
                    name: "retention_ms".to_owned(),
                    value_type: "INTEGER".to_owned(),
                    values: rows
                        .iter()
                        .map(|row| {
                            row.retention_ms.map_or(TypedValue::Null, |value| {
                                TypedValue::Integer(saturating_i64(value).to_string())
                            })
                        })
                        .collect(),
                },
            ],
        })?;
        emit_read_summary(request_id, bookmark, rows.len(), emit)
    }

    fn show_queues(
        &self,
        request_id: Uuid,
        project: ProjectId,
        emit: &mut dyn FnMut(QueryStreamEvent) -> Result<()>,
    ) -> Result<()> {
        let (rows, bookmark) = {
            let state = self.0.state.read();
            (state.broker.queue_metrics(project), state.applied)
        };
        emit(QueryStreamEvent::Schema {
            request_id,
            columns: [
                ("name", "STRING"),
                ("kind", "STRING"),
                ("message_count", "INTEGER"),
                ("available_count", "INTEGER"),
                ("retained_bytes", "INTEGER"),
                ("retention_ms", "INTEGER"),
            ]
            .into_iter()
            .map(|(name, value_type)| QueryColumn {
                name: name.to_owned(),
                value_type: value_type.to_owned(),
                nullable: name == "retention_ms",
            })
            .collect(),
        })?;
        emit(QueryStreamEvent::Batch {
            request_id,
            sequence: 0,
            row_count: rows.len() as u64,
            columns: vec![
                string_batch("name", rows.iter().map(|row| row.name.clone())),
                string_batch(
                    "kind",
                    rows.iter()
                        .map(|row| format!("{:?}", row.kind).to_uppercase()),
                ),
                integer_batch(
                    "message_count",
                    rows.iter().map(|row| saturating_i64(row.message_count)),
                ),
                integer_batch(
                    "available_count",
                    rows.iter().map(|row| saturating_i64(row.available_count)),
                ),
                integer_batch(
                    "retained_bytes",
                    rows.iter().map(|row| saturating_i64(row.retained_bytes)),
                ),
                BatchColumn {
                    name: "retention_ms".to_owned(),
                    value_type: "INTEGER".to_owned(),
                    values: rows
                        .iter()
                        .map(|row| {
                            row.retention_ms.map_or(TypedValue::Null, |value| {
                                TypedValue::Integer(saturating_i64(value).to_string())
                            })
                        })
                        .collect(),
                },
            ],
        })?;
        emit_read_summary(request_id, bookmark, rows.len(), emit)
    }

    fn show_exchanges(
        &self,
        request_id: Uuid,
        project: ProjectId,
        emit: &mut dyn FnMut(QueryStreamEvent) -> Result<()>,
    ) -> Result<()> {
        let (rows, bookmark) = {
            let state = self.0.state.read();
            (state.broker.exchange_metrics(project), state.applied)
        };
        emit(QueryStreamEvent::Schema {
            request_id,
            columns: [
                ("name", "STRING"),
                ("kind", "STRING"),
                ("durable", "BOOLEAN"),
                ("binding_count", "INTEGER"),
            ]
            .into_iter()
            .map(|(name, value_type)| QueryColumn {
                name: name.to_owned(),
                value_type: value_type.to_owned(),
                nullable: false,
            })
            .collect(),
        })?;
        emit(QueryStreamEvent::Batch {
            request_id,
            sequence: 0,
            row_count: rows.len() as u64,
            columns: vec![
                string_batch("name", rows.iter().map(|row| row.name.clone())),
                string_batch(
                    "kind",
                    rows.iter()
                        .map(|row| format!("{:?}", row.kind).to_uppercase()),
                ),
                BatchColumn {
                    name: "durable".to_owned(),
                    value_type: "BOOLEAN".to_owned(),
                    values: rows
                        .iter()
                        .map(|row| TypedValue::Boolean(row.durable))
                        .collect(),
                },
                integer_batch(
                    "binding_count",
                    rows.iter().map(|row| saturating_i64(row.binding_count)),
                ),
            ],
        })?;
        emit_read_summary(request_id, bookmark, rows.len(), emit)
    }

    fn show_consumer_lag(
        &self,
        request_id: Uuid,
        project: ProjectId,
        emit: &mut dyn FnMut(QueryStreamEvent) -> Result<()>,
    ) -> Result<()> {
        let (rows, bookmark) = {
            let state = self.0.state.read();
            (state.broker.consumer_lag_metrics(project), state.applied)
        };
        emit(QueryStreamEvent::Schema {
            request_id,
            columns: [
                ("group", "STRING"),
                ("topic", "STRING"),
                ("partition", "INTEGER"),
                ("committed_offset", "INTEGER"),
                ("next_offset", "INTEGER"),
                ("lag", "INTEGER"),
            ]
            .into_iter()
            .map(|(name, value_type)| QueryColumn {
                name: name.to_owned(),
                value_type: value_type.to_owned(),
                nullable: false,
            })
            .collect(),
        })?;
        emit(QueryStreamEvent::Batch {
            request_id,
            sequence: 0,
            row_count: rows.len() as u64,
            columns: vec![
                string_batch("group", rows.iter().map(|row| row.group.clone())),
                string_batch("topic", rows.iter().map(|row| row.topic.clone())),
                integer_batch("partition", rows.iter().map(|row| i64::from(row.partition))),
                integer_batch(
                    "committed_offset",
                    rows.iter().map(|row| saturating_i64(row.committed_offset)),
                ),
                integer_batch(
                    "next_offset",
                    rows.iter().map(|row| saturating_i64(row.next_offset)),
                ),
                integer_batch("lag", rows.iter().map(|row| saturating_i64(row.lag))),
            ],
        })?;
        emit_read_summary(request_id, bookmark, rows.len(), emit)
    }

    fn execute_broker_admin(
        &self,
        request: &QueryRequest,
        command: BrokerCommand,
        emit: &mut dyn FnMut(QueryStreamEvent) -> Result<()>,
    ) -> Result<()> {
        let commit = self.submit_from(request.connection_id, command, request.consistency)?;
        emit_admin_summary(request.request_id, commit.bookmark, emit)
    }

    fn create_project(
        &self,
        request: &QueryRequest,
        name: &str,
        if_not_exists: bool,
        emit: &mut dyn FnMut(QueryStreamEvent) -> Result<()>,
    ) -> Result<()> {
        let normalized = normalize_name(name);
        if let Some(id) = self.0.state.read().names.get(&normalized).copied() {
            if if_not_exists {
                return emit_admin_summary(request.request_id, self.bookmark(), emit);
            }
            let _ = id;
            return Err(Error::new(
                ErrorCode::TransactionConflict,
                "project name already exists",
            ));
        }
        let id = deterministic_project_id(self.0.store_id, request.request_id);
        let (bookmark, _) = self.commit_from(
            DatabaseMutation::CreateProject {
                id,
                display_name: name.to_owned(),
            },
            MutationKind::Project,
            id,
            Some(request.request_id),
            AdmissionClass::Client,
            request.consistency,
            request.connection_id,
            None,
            self.remaining_query_write_time(request)?,
        )?;
        emit_admin_summary(request.request_id, bookmark, emit)
    }

    fn rename_project(
        &self,
        request: &QueryRequest,
        name: &str,
        new_name: &str,
        emit: &mut dyn FnMut(QueryStreamEvent) -> Result<()>,
    ) -> Result<()> {
        let state = self.0.state.read();
        let id = state
            .names
            .get(&normalize_name(name))
            .copied()
            .ok_or_else(|| Error::new(ErrorCode::ProjectNotFound, "project does not exist"))?;
        if state.names.contains_key(&normalize_name(new_name)) {
            return Err(Error::new(
                ErrorCode::TransactionConflict,
                "project name already exists",
            ));
        }
        drop(state);
        let (bookmark, _) = self.commit_from(
            DatabaseMutation::RenameProject {
                id,
                display_name: new_name.to_owned(),
            },
            MutationKind::Project,
            id,
            Some(request.request_id),
            AdmissionClass::Client,
            request.consistency,
            request.connection_id,
            None,
            self.remaining_query_write_time(request)?,
        )?;
        emit_admin_summary(request.request_id, bookmark, emit)
    }

    fn drop_project(
        &self,
        request: &QueryRequest,
        name: &str,
        if_exists: bool,
        cascade: bool,
        emit: &mut dyn FnMut(QueryStreamEvent) -> Result<()>,
    ) -> Result<()> {
        let id = self
            .0
            .state
            .read()
            .names
            .get(&normalize_name(name))
            .copied();
        let Some(id) = id else {
            if if_exists {
                return emit_admin_summary(request.request_id, self.bookmark(), emit);
            }
            return Err(Error::new(
                ErrorCode::ProjectNotFound,
                "project does not exist",
            ));
        };
        let (bookmark, _) = self.commit_from(
            DatabaseMutation::DropProject { id, cascade },
            MutationKind::Project,
            id,
            Some(request.request_id),
            AdmissionClass::Client,
            request.consistency,
            request.connection_id,
            None,
            self.remaining_query_write_time(request)?,
        )?;
        emit_admin_summary(request.request_id, bookmark, emit)
    }
}

fn apply_reserved_broker_entry(
    database: &Database,
    entry: &MutationEntry,
) -> Result<Option<MutationApplyResult>> {
    if entry.request_id().is_some() || entry.kind() != MutationKind::Broker {
        return Ok(None);
    }
    let payload_digest = *blake3::hash(entry.payload()).as_bytes();
    let _apply = database.0.apply.lock();
    let mut state = database.0.state.write();
    if entry.index() != state.applied.index.saturating_add(1) {
        return Ok(None);
    }
    let (mut staged, broker_publish_cursor) = {
        let mut ordered = database.0.ordered_overlay.lock();
        let Some(record) = ordered.reservations.get(&entry.index()) else {
            return Ok(None);
        };
        if !record.is_broker || record.staged_payload_digest != Some(payload_digest) {
            return Ok(None);
        }
        let broker_publish_cursor = record.broker_publish_cursor;
        let Some(staged) = ordered.states.remove(&entry.index()) else {
            database.0.fatal_apply.store(true, Ordering::Release);
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "ordered broker reservation lost its staged canonical state",
            ));
        };
        (staged, broker_publish_cursor)
    };
    if staged.applied != entry.bookmark() {
        database.0.fatal_apply.store(true, Ordering::Release);
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "ordered broker reservation bookmark differs from committed mutation",
        ));
    }
    let response = (*staged.last_response).clone();
    staged.last_payload_checksum = Some(entry.checksum());
    staged.last_response = response.clone().into();
    if let Err(error) = prune_request_results(&mut staged) {
        database.0.fatal_apply.store(true, Ordering::Release);
        return Err(error);
    }
    if let Some(execution) = database.0.execution.write().as_deref_mut() {
        execution.advance_bookmark(entry.bookmark());
    }
    let committed_broker_segment = broker_publish_cursor
        .and_then(|previous_cursor| staged.broker.newest_payload_segment_after(previous_cursor));
    *state = staged;
    drop(state);
    database.0.broker_changes.send_replace(entry.index());
    if let Some(descriptor) = committed_broker_segment {
        database
            .0
            .pending_broker_segments
            .lock()
            .remove(&descriptor);
    }
    Ok(Some(MutationApplyResult {
        response,
        duplicate: false,
    }))
}

#[async_trait]
impl MutationStateBackend for Database {
    async fn prepare_command(
        &self,
        mut command: WriteCommand,
        sequencer_time_millis: i64,
    ) -> Result<WriteCommand> {
        command.validate()?;
        if command.kind == MutationKind::Broker {
            // Broker replay already carries this canonical timestamp in the WAL entry header.
            // Keep the potentially large command bytes unchanged and resolve its in-memory view
            // during reservation/replay instead of decoding and encoding the complete batch here.
            command.commit_time_millis = sequencer_time_millis;
            return Ok(command);
        }
        let mut mutation = decode_database_mutation(&command.payload)?;
        resolve_sequencer_values(&mut mutation, sequencer_time_millis)?;
        command.commit_time_millis = sequencer_time_millis;
        command.payload = encode_database_value_bounded(
            &mutation,
            "sequencer-resolved mutation",
            self.0.admission.maximum_encoded_bytes(),
        )?;
        command.validate()?;
        Ok(command)
    }

    async fn validate_command(&self, command: &WriteCommand) -> Result<()> {
        self.ensure_apply_healthy()?;
        let _apply = self.0.apply.lock();
        let current = self.0.state.read();
        let position = Bookmark {
            term: current.applied.term.max(1),
            index: current
                .applied
                .index
                .checked_add(1)
                .ok_or_else(|| Error::internal("local write index exhausted"))?,
        };
        validate_resolved_database_command(&current, command, position).map(|_| ())
    }

    async fn reserve_command(
        &self,
        command: &WriteCommand,
        position: Bookmark,
    ) -> Result<CommandReservation> {
        self.ensure_apply_healthy()?;
        command.validate()?;
        let mut mutation = decode_database_mutation(&command.payload)?;
        resolve_sequencer_values(&mut mutation, command.commit_time_millis)?;
        validate_database_envelope(command.kind, command.project_id, &mutation)?;
        let _apply = self.0.apply.lock();
        let current = self.0.state.read();
        let mut ordered = self.0.ordered_overlay.lock();
        if ordered.reservations.contains_key(&position.index) {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "local write position already has an ordered command reservation",
            ));
        }

        if matches!(mutation, DatabaseMutation::Graph { .. }) {
            let base = ordered
                .states
                .last_key_value()
                .map_or(&*current, |(_, state)| state);
            if base.applied.index.checked_add(1) != Some(position.index) {
                return Err(Error::retryable(
                    ErrorCode::WriteAdmissionFull,
                    "ordered database state has not reached the next reserved position",
                    Some(1),
                ));
            }
            let staged_intent_digest = command
                .request_id
                .map(|_| request_intent_digest(&mutation))
                .transpose()?
                .unwrap_or([0; 32]);
            let staged = stage_ordered_database_command_with_digest(
                base,
                command,
                mutation,
                position,
                &self.0.segments,
                staged_intent_digest,
            )?;
            let reservation = CommandReservation::pipelined(position)?;
            let token = reservation.token().ok_or_else(|| {
                Error::internal("pipelined database reservation has no owner token")
            })?;
            ordered.states.insert(position.index, staged);
            ordered.reservations.insert(
                position.index,
                OrderedMutationReservation {
                    token,
                    broker_segments: Vec::new(),
                    staged_payload_digest: Some(*blake3::hash(&command.payload).as_bytes()),
                    is_broker: false,
                    broker_publish_cursor: None,
                },
            );
            return Ok(reservation);
        }

        if !ordered.reservations.is_empty() {
            return Err(Error::retryable(
                ErrorCode::WriteAdmissionFull,
                "ordered graph mutations are still awaiting local application",
                Some(1),
            ));
        }
        validate_resolved_database_command(&current, command, position)?;
        let reservation = CommandReservation::owned_serialized(position)?;
        let token = reservation
            .token()
            .ok_or_else(|| Error::internal("owned database reservation has no token"))?;
        let is_broker = matches!(&mutation, DatabaseMutation::Broker { .. });
        let (broker_segments, staged_payload_digest, broker_publish_cursor) = if is_broker {
            let broker_publish_cursor = matches!(
                &mutation,
                DatabaseMutation::Broker {
                    command: BrokerCommand::PublishKafkaBatch { .. }
                        | BrokerCommand::PublishAmqp { .. }
                        | BrokerCommand::PublishAmqpBatch { .. }
                        | BrokerCommand::PublishAmqpUniformBatch { .. }
                }
            )
            .then(|| current.broker.message_cursor());
            let previous_segments = current
                .broker
                .payload_segments()
                .into_iter()
                .collect::<BTreeSet<_>>();
            let staged_intent_digest = command
                .request_id
                .map(|_| request_intent_digest(&mutation))
                .transpose()?
                .unwrap_or([0; 32]);
            // Broker payload materialization is part of this exact intent-bound generation.
            // Publication consumes the generation below instead of encoding and writing the
            // same segment a second time. Replay has no reservation and reconstructs it from
            // the eventual-durability WAL command as before.
            let staged = stage_ordered_database_command_with_digest(
                &current,
                command,
                mutation,
                position,
                &self.0.segments,
                staged_intent_digest,
            )?;
            let descriptors = staged
                .broker
                .payload_segments()
                .into_iter()
                .filter(|descriptor| !previous_segments.contains(descriptor))
                .collect::<Vec<_>>();
            retain_pending_broker_segments(&self.0, token, &descriptors)?;
            ordered.states.insert(position.index, staged);
            (
                descriptors,
                Some(*blake3::hash(&command.payload).as_bytes()),
                broker_publish_cursor,
            )
        } else {
            (Vec::new(), None, None)
        };
        ordered.reservations.insert(
            position.index,
            OrderedMutationReservation {
                token,
                broker_segments,
                staged_payload_digest,
                is_broker,
                broker_publish_cursor,
            },
        );
        Ok(reservation)
    }

    async fn complete_command_reservation(
        &self,
        reservation: CommandReservation,
        outcome: CommandReservationOutcome,
    ) -> Result<()> {
        let Some(token) = reservation.token() else {
            return Ok(());
        };
        let mut released = Vec::<(Uuid, Vec<SegmentDescriptor>)>::new();
        {
            let _apply = self.0.apply.lock();
            let mut ordered = self.0.ordered_overlay.lock();
            let Some(record) = ordered.reservations.get(&reservation.position().index) else {
                return Ok(());
            };
            if record.token != token {
                return Ok(());
            }
            if outcome == CommandReservationOutcome::Rejected
                && reservation.may_release_after_append()
            {
                released.extend(
                    ordered
                        .reservations
                        .values()
                        .map(|record| (record.token, record.broker_segments.clone())),
                );
                ordered.reservations.clear();
                ordered.states.clear();
            } else if let Some(record) = ordered.reservations.remove(&reservation.position().index)
            {
                released.push((record.token, record.broker_segments));
                ordered.states.remove(&reservation.position().index);
            }
        }
        let rejected = outcome == CommandReservationOutcome::Rejected;
        for (owner, descriptors) in released {
            release_pending_broker_segments(&self.0, owner, &descriptors, rejected);
        }
        Ok(())
    }

    async fn applied_bookmark(&self) -> Bookmark {
        self.0.state.read().applied
    }

    async fn apply_mutation(&self, entry: &MutationEntry) -> Result<MutationApplyResult> {
        self.ensure_apply_healthy()?;
        entry.verify()?;
        if let Some(applied) = apply_reserved_broker_entry(self, entry)? {
            return Ok(applied);
        }
        let mut mutation = decode_database_mutation(entry.payload())?;
        resolve_sequencer_values(&mut mutation, entry.commit_time_millis())?;
        let intent_digest = entry
            .request_id()
            .map(|_| request_intent_digest(&mutation))
            .transpose()?
            .unwrap_or([0; 32]);
        let payload_digest = *blake3::hash(entry.payload()).as_bytes();
        let broker_may_retire_segments = matches!(
            &mutation,
            DatabaseMutation::DropProject { .. }
                | DatabaseMutation::Broker {
                    command: BrokerCommand::DeleteTopic { .. }
                        | BrokerCommand::ClearTopic { .. }
                        | BrokerCommand::Retain { .. }
                }
        );
        let is_broker_mutation = matches!(&mutation, DatabaseMutation::Broker { .. });
        let commit_time_nanos = millis_to_nanos(entry.commit_time_millis())?;
        let device_impact = device_impact(&mutation, entry.bookmark(), commit_time_nanos);
        validate_database_entry(entry, &mutation)?;
        let _apply = self.0.apply.lock();
        let mut state = self.0.state.write();

        if entry.index() < state.applied.index {
            let response = match entry.request_id().and_then(|request| {
                state
                    .request_results
                    .get(&request)
                    .map(|record| record.response.clone())
            }) {
                Some(response) => response,
                None => {
                    encode_database_value(&Option::<BrokerReply>::None, "empty apply response")?
                }
            };
            return Ok(MutationApplyResult {
                response,
                duplicate: true,
            });
        }
        if entry.index() == state.applied.index {
            if state.last_payload_checksum != Some(entry.checksum()) {
                self.0.fatal_apply.store(true, Ordering::Release);
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "database mutation differs at an applied local write index",
                ));
            }
            return Ok(MutationApplyResult {
                response: (*state.last_response).clone(),
                duplicate: true,
            });
        }
        if entry.index() != state.applied.index.saturating_add(1) {
            self.0.fatal_apply.store(true, Ordering::Release);
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "database mutation apply has a local write-index gap",
            ));
        }
        if let Some(request_id) = entry.request_id()
            && let Some(record) = state.request_results.get(&request_id)
        {
            if record.intent_digest != intent_digest {
                self.0.fatal_apply.store(true, Ordering::Release);
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "committed request ID refers to a different mutation intent",
                ));
            }
            let response = record.response.clone();
            state.applied = entry.bookmark();
            if let Some(execution) = self.0.execution.write().as_deref_mut() {
                execution.advance_bookmark(entry.bookmark());
            }
            state.last_payload_checksum = Some(entry.checksum());
            state.last_response = response.clone().into();
            if let Err(error) = prune_request_results(&mut state) {
                self.0.fatal_apply.store(true, Ordering::Release);
                return Err(error);
            }
            return Ok(MutationApplyResult {
                response,
                duplicate: true,
            });
        }

        let broker_publish_cursor = matches!(
            &mutation,
            DatabaseMutation::Broker {
                command: BrokerCommand::PublishKafkaBatch { .. }
                    | BrokerCommand::PublishAmqp { .. }
                    | BrokerCommand::PublishAmqpBatch { .. }
                    | BrokerCommand::PublishAmqpUniformBatch { .. }
            }
        )
        .then(|| state.broker.message_cursor());
        let mut committed_broker_segment = None;
        let mut retired_broker_segments = Vec::new();
        let reserved_staged_state = {
            let mut ordered = self.0.ordered_overlay.lock();
            match ordered.reservations.get(&entry.index()) {
                Some(record) if record.staged_payload_digest == Some(payload_digest) => {
                    let Some(staged) = ordered.states.remove(&entry.index()) else {
                        self.0.fatal_apply.store(true, Ordering::Release);
                        return Err(Error::new(
                            ErrorCode::CorruptStorage,
                            "ordered graph reservation lost its staged canonical state",
                        ));
                    };
                    if staged.applied != entry.bookmark() {
                        self.0.fatal_apply.store(true, Ordering::Release);
                        return Err(Error::new(
                            ErrorCode::CorruptStorage,
                            "ordered graph reservation bookmark differs from committed mutation",
                        ));
                    }
                    Some(staged)
                }
                Some(record) if record.staged_payload_digest.is_some() => {
                    self.0.fatal_apply.store(true, Ordering::Release);
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        "ordered graph reservation differs from committed mutation intent",
                    ));
                }
                _ => None,
            }
        };
        // Read-only client/data preflight. In the standalone direct-apply path this is the only
        // place a mutation is validated, so a failure here is a first-time client rejection — it
        // must NOT trip `fatal_apply`. Keep it OUTSIDE the fatal closure below: state is a clone
        // that is only committed on success, so a rejected mutation leaves canonical state intact.
        if reserved_staged_state.is_none() {
            validate_sequencer_mutation(&state, &mutation, entry.bookmark(), commit_time_nanos)?;
        }
        let result = (|| -> Result<MutationApplyResult> {
            // Any error past this point is a local invariant/device/storage failure: in-place
            // publication has begun and the preflight above already accepted the mutation.
            let previous_broker_segments = broker_may_retire_segments.then(|| {
                state
                    .broker
                    .payload_segments()
                    .into_iter()
                    .collect::<BTreeSet<_>>()
            });
            // The normal standalone path already cloned and applied this graph mutation during
            // pre-WAL reservation. Consume that exact intent-bound overlay here: work tracks the
            // changed rows once and does not repeat canonical/index mutation after fsync. Replay
            // and non-graph mutations have no reservation overlay and use the cold apply path.
            let (mut staged_state, response, state_already_staged) =
                if let Some(staged_state) = reserved_staged_state {
                    let response = (*staged_state.last_response).clone();
                    (staged_state, response, true)
                } else {
                    let mut staged_state = state.clone();
                    let reply = apply_mutation(
                        &mut staged_state,
                        mutation,
                        entry.bookmark(),
                        commit_time_nanos,
                        Some(&self.0.segments),
                    )?;
                    let response = encode_database_value(&reply, "apply response")?;
                    (staged_state, response, false)
                };
            if let Some(previous_cursor) = broker_publish_cursor {
                committed_broker_segment = staged_state
                    .broker
                    .newest_payload_segment_after(previous_cursor);
            }
            if response.len() > MAX_SINGLE_REQUEST_RESULT_BYTES {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "committed mutation response exceeds the durable result bound",
                ));
            }
            staged_state.applied = entry.bookmark();
            staged_state.last_payload_checksum = Some(entry.checksum());
            staged_state.last_response = response.clone().into();
            if !state_already_staged && let Some(request_id) = entry.request_id() {
                retain_request_result(
                    &mut staged_state,
                    request_id,
                    intent_digest,
                    response.clone(),
                )?;
            }
            prune_request_results(&mut staged_state)?;
            {
                let mut execution = self.0.execution.write();
                populate_pending_vector_indexes(
                    &mut staged_state,
                    &device_impact,
                    execution.as_deref(),
                )?;
                let resident_already_stale = match (&device_impact, execution.as_deref()) {
                    (
                        DeviceImpact::Project { project, .. } | DeviceImpact::Rebuild(project),
                        Some(backend),
                    ) => state.projects.get(project).is_none_or(|current| {
                        backend.resident_graph_revision(*project) != Some(current.graph.revision())
                    }),
                    _ => false,
                };
                let semantic_project = match &device_impact {
                    DeviceImpact::Project { project, .. } | DeviceImpact::Rebuild(project) => {
                        staged_state
                            .projects
                            .get(project)
                            .filter(|state| {
                                state.indexes.contains(crate::graph::SEMANTIC_NODE_INDEX)
                            })
                            .map(|_| *project)
                    }
                    _ => None,
                };
                let device_impact = if resident_already_stale {
                    semantic_project
                        .map(DeviceImpact::Rebuild)
                        .unwrap_or(device_impact)
                } else {
                    device_impact
                };
                if semantic_project.is_none()
                    && (resident_already_stale
                        || match &device_impact {
                            DeviceImpact::Project { project, .. }
                            | DeviceImpact::Rebuild(project) => staged_state
                                .projects
                                .get(project)
                                .is_some_and(|state| should_defer_device_publish(&state.graph)),
                            _ => false,
                        })
                {
                    // Deferred: leave the resident image at its last-published revision so the query
                    // freshness check routes reads to host execution. The bookmark is intentionally
                    // NOT advanced on the device — advancing it would let a stale resident image
                    // answer queries. Rebuild is deferred too: an administrative graph mutation (e.g.
                    // declaring a new relationship type on the first edge of a bulk seed) otherwise
                    // rebuilds the whole resident image on the GPU under the apply lock, stalling the
                    // write gate at million scale. On a large graph every read is on host anyway.
                    // See `should_defer_device_publish`.
                } else if let Err(error) = publish_execution_state(
                    &mut *execution,
                    &mut staged_state,
                    device_impact,
                    entry.bookmark(),
                ) {
                    // The resident is a derived execution image. The canonical state clone has
                    // already passed deterministic validation, and the backend publishes staged
                    // generations atomically, so a failed device delta leaves the old resident
                    // intact. Commit the canonical write, leave the resident fence behind, and let
                    // the off-path catch-up worker rebuild it. Poisoning the state machine here
                    // turned one bad GPU byte into the loss of every database protocol.
                    tracing::warn!(
                        code = ?error.code,
                        message = %error.message,
                        bookmark = entry.index(),
                        "resident publication failed; canonical mutation committed and off-path rebuild queued"
                    );
                }
            }
            if let Some(previous) = previous_broker_segments {
                let current = staged_state
                    .broker
                    .payload_segments()
                    .into_iter()
                    .collect::<BTreeSet<_>>();
                retired_broker_segments.extend(previous.difference(&current).cloned());
            }
            // This exact staged generation is the canonical publication boundary. The dedicated
            // writer acknowledges it and persists the same ordered mutation on its background WAL
            // path afterward; no full graph build appears on that delta path.
            *state = staged_state;
            Ok(MutationApplyResult {
                response,
                duplicate: false,
            })
        })();
        if let Err(error) = &result {
            tracing::error!(
                code = ?error.code,
                message = %error.message,
                bookmark = entry.index(),
                "database state-machine apply invariant failed"
            );
            self.0.fatal_apply.store(true, Ordering::Release);
        } else {
            // The graph is now committed and resident. Release canonical state before broker
            // notification and segment reclamation.
            drop(state);
            if is_broker_mutation {
                self.0.broker_changes.send_replace(entry.index());
            }
            if let Some(descriptor) = committed_broker_segment {
                self.0.pending_broker_segments.lock().remove(&descriptor);
            }
            if !retired_broker_segments.is_empty() {
                self.0
                    .retired_broker_segments
                    .lock()
                    .extend(retired_broker_segments);
            }
        }
        result
    }

    async fn build_snapshot(
        &self,
        bookmark: Bookmark,
        destination: &Path,
    ) -> Result<BackendSnapshot> {
        let database = self.clone();
        let destination = destination.to_owned();
        tokio::task::spawn_blocking(move || {
            build_database_snapshot_file(&database, bookmark, &destination)
        })
        .await
        .map_err(|error| Error::internal(format!("snapshot build task failed: {error}")))?
    }

    async fn install_snapshot(&self, bookmark: Bookmark, snapshot: &BackendSnapshot) -> Result<()> {
        let database = self.clone();
        let snapshot = snapshot.clone();
        tokio::task::spawn_blocking(move || {
            install_database_snapshot_file(&database, bookmark, &snapshot)
        })
        .await
        .map_err(|error| Error::internal(format!("snapshot install task failed: {error}")))?
    }
}

#[derive(Clone, Debug)]
enum DeviceImpact {
    None,
    Create(ProjectId),
    Rebuild(ProjectId),
    Project {
        project: ProjectId,
        temporal: Vec<ResidentTemporalDelta>,
        vectors: Vec<ResolvedVectorMutation>,
        invalidate_derived: bool,
    },
    Drop(ProjectId),
}

fn device_impact(
    mutation: &DatabaseMutation,
    bookmark: Bookmark,
    commit_time_nanos: i64,
) -> DeviceImpact {
    match mutation {
        DatabaseMutation::CreateProject { id, .. } => DeviceImpact::Create(*id),
        DatabaseMutation::Graph {
            project,
            graph,
            temporal,
            vectors,
            administrative,
            ..
        } => {
            if administrative.is_some() {
                return DeviceImpact::Rebuild(*project);
            }
            let vector_delta = vectors
                .iter()
                .cloned()
                .map(|mutation| retag_vector_mutation(mutation, bookmark.index))
                .collect::<Vec<_>>();
            DeviceImpact::Project {
                project: *project,
                temporal: temporal
                    .iter()
                    .cloned()
                    .map(|mut mutation| {
                        mutation.sample.sequence_index = bookmark.index;
                        if mutation.uses_commit_time {
                            mutation.sample.event_time_nanos = commit_time_nanos;
                        }
                        ResidentTemporalDelta {
                            entity_kind: mutation.entity_kind,
                            target: mutation.target,
                            sample: mutation.sample,
                        }
                    })
                    .collect(),
                vectors: vector_delta,
                invalidate_derived: !graph.is_empty(),
            }
        }
        DatabaseMutation::DropProject { id, .. } => DeviceImpact::Drop(*id),
        DatabaseMutation::EmbeddingProfile {
            command: EmbeddingProfileCommand::Activate { project, .. },
        } => DeviceImpact::Rebuild(*project),
        DatabaseMutation::EmbeddingProfile {
            command:
                EmbeddingProfileCommand::Begin { .. }
                | EmbeddingProfileCommand::Acknowledge { .. }
                | EmbeddingProfileCommand::Abort { .. },
        } => DeviceImpact::None,
        DatabaseMutation::RenameProject { .. }
        | DatabaseMutation::Broker { .. }
        | DatabaseMutation::Security { .. } => DeviceImpact::None,
    }
}

/// One-row mutations use the bounded sparse resident delta synchronously. A multi-row mutation is
/// source/bulk work and never waits on GPU publication under the canonical apply mutex: the
/// off-path worker builds and publishes a coherent latest resident generation after quiescence.
/// This check reads only the current change journal, so its work tracks changed rows and never
/// scales with unrelated graph content.
fn should_defer_device_publish(graph: &GraphStore) -> bool {
    graph
        .change_ids(graph.revision())
        .map(|changes| changes.nodes.len().saturating_add(changes.edges.len()) > 1)
        .unwrap_or(true)
}

fn publish_execution_state(
    execution: &mut Option<Box<dyn ExecutionBackend>>,
    state: &mut DatabaseState,
    impact: DeviceImpact,
    bookmark: Bookmark,
) -> Result<()> {
    let Some(execution) = execution.as_deref_mut() else {
        return Ok(());
    };
    match impact {
        DeviceImpact::None => {
            execution.advance_bookmark(bookmark);
            Ok(())
        }
        DeviceImpact::Drop(project) => {
            execution.evict_project(project)?;
            execution.advance_bookmark(bookmark);
            Ok(())
        }
        DeviceImpact::Create(project) => {
            let project_state = state.projects.get(&project).ok_or_else(|| {
                Error::new(
                    ErrorCode::CorruptStorage,
                    "device publication references a missing project",
                )
            })?;
            execution.admit_project(ResidentProjectImage::build(
                project,
                bookmark,
                &project_state.graph,
                &project_state.temporal,
                &project_state.indexes,
            )?)?;
            rebind_shared_project(state, execution, project)?;
            execution.advance_bookmark(bookmark);
            Ok(())
        }
        DeviceImpact::Rebuild(project) => {
            let project_state = state.projects.get(&project).ok_or_else(|| {
                Error::new(
                    ErrorCode::CorruptStorage,
                    "device rebuild references a missing project",
                )
            })?;
            execution.admit_project(ResidentProjectImage::build(
                project,
                bookmark,
                &project_state.graph,
                &project_state.temporal,
                &project_state.indexes,
            )?)?;
            rebind_shared_project(state, execution, project)?;
            execution.advance_bookmark(bookmark);
            Ok(())
        }
        DeviceImpact::Project {
            project,
            temporal,
            vectors,
            invalidate_derived,
        } => {
            {
                let project_state = state.projects.get(&project).ok_or_else(|| {
                    Error::new(
                        ErrorCode::CorruptStorage,
                        "device publication references a missing project",
                    )
                })?;
                execution.apply_project_delta(ResidentProjectDelta {
                    project,
                    bookmark,
                    graph: project_state.graph.device_delta(bookmark.index)?,
                    temporal,
                    vectors,
                    invalidate_derived,
                })?;
            }
            rebind_shared_project(state, execution, project)?;
            execution.advance_bookmark(bookmark);
            Ok(())
        }
    }
}

fn rebind_shared_project(
    state: &mut DatabaseState,
    execution: &dyn ExecutionBackend,
    project: ProjectId,
) -> Result<()> {
    let Some(backing) = execution.shared_project_backing(project) else {
        return Ok(());
    };
    let project = state.projects.get_mut(&project).ok_or_else(|| {
        Error::new(
            ErrorCode::CorruptStorage,
            "shared device publication references a missing project",
        )
    })?;
    let project = Arc::make_mut(project);
    project.graph.rebind_shared(backing.graph)?;
    project.temporal.rebind_shared(backing.temporal)?;
    project.indexes.rebind_shared_vectors(backing.vectors)?;
    // Through the lag-tolerant check, not an unconditional clear.
    //
    // This is on the device-publication path, so it runs after EVERY committed write. Clearing the
    // snapshot outright here made `expire_optimizer_statistics_if_stale` and its
    // `OPTIMIZER_STATISTICS_MAX_REVISION_LAG` unreachable: by the time that function was consulted
    // there was nothing left to expire, so the tolerance it exists to provide never applied to a
    // single write. The rebind republishes the same logical graph plus this commit's mutations, so
    // the existing snapshot still describes it to within that lag — and the same check also catches
    // a changed schema or index generation, which a rebind can carry.
    //
    // Measured on 200 000 nodes: the reset discarded a live snapshot on every commit and the next
    // request rebuilt it at ~148 ms — including `RETURN 1`, which needs no statistics at all,
    // because the request path collects whenever a backend exists rather than when the plan asks.
    expire_optimizer_statistics_if_stale(project);
    Ok(())
}

/// Populates only vector definitions staged by the current administrative mutation. The state
/// clone and old runtime remain private until every local build has either atomically published a
/// validated generation or recorded a non-destructive FAILED/old-ONLINE outcome.
fn populate_pending_vector_indexes(
    state: &mut DatabaseState,
    impact: &DeviceImpact,
    backend: Option<&dyn ExecutionBackend>,
) -> Result<()> {
    let DeviceImpact::Rebuild(project) = impact else {
        return Ok(());
    };
    let project = state.projects.get_mut(project).ok_or_else(|| {
        Error::new(
            ErrorCode::CorruptStorage,
            "vector population references a missing project",
        )
    })?;
    let project = Arc::make_mut(project);
    let cancellation = tokio_util::sync::CancellationToken::new();
    if let Some(backend) = backend {
        project.indexes.rebuild_vectors_with(|source, config| {
            backend.build_ivf_pq(source, config, &cancellation)
        })?;
    } else {
        // No device has been bound, so this is the explicit semantic CPU fallback. Its governor
        // still admits the deterministic build plan; there is no implicit GPU-to-host fallback.
        let cpu = crate::gpu::CpuBackend::new(usize::MAX, 0);
        project.indexes.rebuild_vectors_with(|source, config| {
            cpu.build_ivf_pq(source, config, &cancellation)
        })?;
    }
    project.optimizer_statistics = OnceLock::new();
    Ok(())
}

impl Database {
    fn begin_transaction(
        &self,
        connection: ConnectionId,
        project: Option<ProjectId>,
        bookmark: Option<Bookmark>,
        consistency: CommitAcknowledgement,
    ) -> Result<Box<dyn QueryTransaction>> {
        let project = project.ok_or_else(|| {
            Error::new(ErrorCode::ProjectNotFound, "transaction requires a project")
        })?;
        let barrier = self.consistency_barrier(consistency, bookmark)?;
        self.ensure_automatic_semantic(project)?;
        let (snapshot, current_bookmark, execution) = self.capture_semantic_execution(project)?;
        self.wait_for_captured_all(consistency, barrier, current_bookmark)?;
        let fence = self.current_transaction_fence(current_bookmark)?;
        let deadline = Instant::now()
            .checked_add(self.0.request_timeout)
            .ok_or_else(|| Error::invalid_data("transaction deadline overflow"))?;
        let mut state = TransactionState {
            catalog: snapshot.graph.catalog().clone(),
            working: Arc::clone(&snapshot),
            snapshot,
            execution,
            bookmark: current_bookmark,
            consistency,
            dependencies: TransactionDependencies::default(),
            graph_mutations: Vec::new(),
            temporal_mutations: Vec::new(),
            vector_mutations: Vec::new(),
            accounted_bytes: 0,
        };
        state.accounted_bytes =
            transaction_retained_bytes(&state, self.0.transactions.maximum_encoded_bytes())?;
        // The backend already tracks the admitted complete image. Reading that O(1) counter is
        // essential here: rebuilding a project image merely to account a BEGIN would make
        // transaction latency proportional to the entire database.
        let pin_device_bytes = state
            .execution
            .as_deref()
            .and_then(|execution| execution.resident_project_bytes(project))
            .unwrap_or(0);
        let resource = Arc::new(TransactionResource {
            terminal: AtomicU8::new(0),
            lifecycle: Mutex::new(TransactionLifecycle::Active(state)),
        });
        let id = Uuid::new_v4();
        self.0.transactions.register(
            id,
            connection,
            match &*resource.lifecycle.lock() {
                TransactionLifecycle::Active(state) => state.accounted_bytes,
                _ => unreachable!("new transaction resource is active"),
            },
            TransactionPinKey {
                project,
                bookmark: current_bookmark,
            },
            pin_device_bytes,
            deadline,
            fence,
            &resource,
        )?;
        Ok(Box::new(DatabaseTransaction {
            database: self.clone(),
            id,
            project,
            connection,
            fence,
            deadline,
            resource,
        }))
    }

    // Standalone is always its own sequencer: the fence is the local node and the write term, and it
    // can never change under it, so explicit-transaction fencing is trivially satisfied.
    fn current_transaction_fence(&self, snapshot: Bookmark) -> Result<TransactionFence> {
        let binding = self.write_binding()?;
        let runtime = binding
            .runtime
            .upgrade()
            .ok_or_else(|| Error::new(ErrorCode::Cancelled, "write runtime is shutting down"))?;
        Ok(TransactionFence {
            sequencer: runtime.node_id(),
            term: snapshot.term.max(1),
            snapshot,
        })
    }

    fn verify_transaction_fence(&self, expected: TransactionFence) -> Result<()> {
        let current = self.current_transaction_fence(expected.snapshot)?;
        if current.term != expected.term || current.sequencer != expected.sequencer {
            return Err(transaction_sequencer_changed_error());
        }
        Ok(())
    }
}

fn stage_transaction_project(
    current: &ProjectState,
    graph: &[GraphMutation],
    temporal: &[crate::cypher::PreparedTemporalMutation],
    vectors: &[ResolvedVectorMutation],
    now_nanos: i64,
) -> Result<ProjectState> {
    let mut staged = current.clone();
    for mutation in graph {
        update_next_ids(&mut staged, mutation);
        crate::graph::knowledge::validate_mutation(&staged.graph, mutation)?;
        staged.indexes.before_graph_apply(&staged.graph, mutation)?;
        staged.graph.apply(mutation.clone())?;
        staged.indexes.after_graph_apply(&staged.graph, mutation)?;
    }
    for mutation in temporal {
        staged.temporal.append(
            mutation.entity_kind,
            mutation.target,
            mutation.sample.clone(),
            now_nanos,
        )?;
    }
    for mutation in vectors {
        staged.indexes.apply_vector_mutation(mutation)?;
    }
    staged.optimizer_statistics = OnceLock::new();
    Ok(staged)
}

fn stage_transaction_execution(
    current: Option<&dyn ExecutionBackend>,
    project: ProjectId,
    bookmark: Bookmark,
    staged: &ProjectState,
    graph_changed: bool,
    temporal: &[crate::cypher::PreparedTemporalMutation],
    vectors: &[ResolvedVectorMutation],
) -> Result<Option<Box<dyn ExecutionBackend>>> {
    let Some(current) = current else {
        return Ok(None);
    };
    let mut next = current.pin_project(project)?;
    if graph_changed || !temporal.is_empty() || !vectors.is_empty() {
        let overlay_revision = bookmark.index.checked_add(1).ok_or_else(|| {
            Error::new(
                ErrorCode::GpuAdmissionFailure,
                "transaction resident revision exhausted",
            )
        })?;
        next.apply_project_overlay(ResidentProjectDelta {
            project,
            bookmark,
            graph: staged.graph.device_delta(overlay_revision)?,
            temporal: temporal
                .iter()
                .map(|mutation| ResidentTemporalDelta {
                    entity_kind: mutation.entity_kind,
                    target: mutation.target,
                    sample: mutation.sample.clone(),
                })
                .collect(),
            vectors: vectors.to_vec(),
            invalidate_derived: graph_changed,
        })?;
    }
    Ok(Some(next))
}

fn transaction_retained_bytes(state: &TransactionState, limit: usize) -> Result<usize> {
    #[derive(Serialize)]
    struct RetainedTransaction<'a> {
        catalog: &'a crate::graph::NameCatalog,
        dependencies: &'a TransactionDependencies,
        graph: &'a [GraphMutation],
        temporal: Vec<PersistedTemporalMutation>,
        vectors: &'a [ResolvedVectorMutation],
    }

    let temporal = state
        .temporal_mutations
        .iter()
        .map(|mutation| PersistedTemporalMutation {
            entity_kind: mutation.entity_kind,
            target: mutation.target,
            sample: mutation.sample.clone(),
            uses_commit_time: mutation.uses_commit_time,
        })
        .collect();
    encode_database_value_bounded(
        &RetainedTransaction {
            catalog: &state.catalog,
            dependencies: &state.dependencies,
            graph: &state.graph_mutations,
            temporal,
            vectors: &state.vector_mutations,
        },
        "explicit transaction admission state",
        limit,
    )
    .map(|bytes| bytes.len().max(MIN_TRANSACTION_ACCOUNTED_BYTES))
}

impl QueryExecutor for Database {
    fn resolve_project(&self, selector: &str) -> Result<ProjectId> {
        self.ensure_apply_healthy()?;
        if let Ok(id) = Uuid::parse_str(selector) {
            let project = ProjectId(id);
            if self.0.state.read().projects.contains_key(&project) {
                return Ok(project);
            }
        }
        self.0
            .state
            .read()
            .names
            .get(&normalize_name(selector))
            .copied()
            .ok_or_else(|| Error::new(ErrorCode::ProjectNotFound, "project does not exist"))
    }

    fn execute(
        &self,
        request: QueryRequest,
        emit: &mut dyn FnMut(QueryStreamEvent) -> Result<()>,
    ) -> Result<()> {
        let mut budget = QueryResultBudget::new(request.limits);
        self.execute_autocommit(request, &mut |mut event| {
            budget.observe(&event)?;
            budget.settle(&mut event);
            emit(event)
        })
    }

    fn begin(
        &self,
        project: Option<ProjectId>,
        bookmark: Option<Bookmark>,
        consistency: CommitAcknowledgement,
    ) -> Result<Box<dyn QueryTransaction>> {
        self.begin_transaction(ConnectionId::new(), project, bookmark, consistency)
    }

    fn begin_on_connection(
        &self,
        connection: ConnectionId,
        project: Option<ProjectId>,
        bookmark: Option<Bookmark>,
        consistency: CommitAcknowledgement,
    ) -> Result<Box<dyn QueryTransaction>> {
        self.begin_transaction(connection, project, bookmark, consistency)
    }
}

struct TransactionState {
    catalog: crate::graph::NameCatalog,
    snapshot: Arc<ProjectState>,
    working: Arc<ProjectState>,
    execution: Option<Box<dyn ExecutionBackend>>,
    bookmark: Bookmark,
    consistency: CommitAcknowledgement,
    dependencies: TransactionDependencies,
    graph_mutations: Vec<GraphMutation>,
    temporal_mutations: Vec<crate::cypher::PreparedTemporalMutation>,
    vector_mutations: Vec<ResolvedVectorMutation>,
    accounted_bytes: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TransactionTerminal {
    Expired,
    SequencerChanged,
}

enum TransactionLifecycle {
    Active(TransactionState),
    Finalizing,
    Terminal(TransactionTerminal),
    Closed,
}

struct TransactionResource {
    terminal: AtomicU8,
    lifecycle: Mutex<TransactionLifecycle>,
}

impl TransactionResource {
    fn try_terminate(&self, reason: TransactionTerminal) -> bool {
        if let Some(mut lifecycle) = self.lifecycle.try_lock() {
            match *lifecycle {
                TransactionLifecycle::Active(_) => {
                    self.terminal.store(
                        match reason {
                            TransactionTerminal::Expired => 1,
                            TransactionTerminal::SequencerChanged => 2,
                        },
                        Ordering::Release,
                    );
                    *lifecycle = TransactionLifecycle::Terminal(reason);
                    true
                }
                TransactionLifecycle::Terminal(_) | TransactionLifecycle::Closed => true,
                TransactionLifecycle::Finalizing => false,
            }
        } else {
            self.terminal.store(
                match reason {
                    TransactionTerminal::Expired => 1,
                    TransactionTerminal::SequencerChanged => 2,
                },
                Ordering::Release,
            );
            false
        }
    }

    fn close(&self) {
        self.terminal.store(3, Ordering::Release);
        let mut lifecycle = self.lifecycle.lock();
        if !matches!(*lifecycle, TransactionLifecycle::Terminal(_)) {
            *lifecycle = TransactionLifecycle::Closed;
        }
    }
}

struct DatabaseTransaction {
    database: Database,
    id: Uuid,
    project: ProjectId,
    connection: ConnectionId,
    fence: TransactionFence,
    deadline: Instant,
    resource: Arc<TransactionResource>,
}

impl DatabaseTransaction {
    fn ensure_active(&self) -> Result<()> {
        self.database.ensure_apply_healthy()?;
        if let Err(error) = self
            .database
            .0
            .transactions
            .ensure_active(self.id, self.fence)
        {
            return Err(self.lifecycle_error().unwrap_or(error));
        }
        if let Err(error) = self.database.verify_transaction_fence(self.fence) {
            self.database.0.transactions.finish(self.id);
            return Err(error);
        }
        self.lifecycle_error().map_or(Ok(()), Err)
    }

    fn lifecycle_error(&self) -> Option<Error> {
        transaction_lifecycle_error(&self.resource.lifecycle.lock())
    }
}

impl QueryTransaction for DatabaseTransaction {
    fn run(
        &mut self,
        request: QueryRequest,
        emit: &mut dyn FnMut(QueryStreamEvent) -> Result<()>,
    ) -> Result<()> {
        self.ensure_active()?;
        let mut budget = QueryResultBudget::new(request.limits);
        if request
            .project_id
            .is_some_and(|project| project != self.project)
        {
            return Err(Error::new(
                ErrorCode::AuthorizationDenied,
                "transaction cannot switch projects",
            ));
        }
        let mut lifecycle = self.resource.lifecycle.lock();
        let state = match &mut *lifecycle {
            TransactionLifecycle::Active(state) => state,
            other => {
                return Err(transaction_lifecycle_error(other).unwrap_or_else(transaction_expired));
            }
        };
        let provisional = state.bookmark.index.saturating_add(1);
        let text_embedding = self.database.0.text_embedding.read().clone();
        let output = execute_on_project_inner(
            &state.working,
            &request,
            state.bookmark,
            provisional,
            full_capabilities(),
            text_embedding.as_deref(),
            state.execution.as_deref(),
            None,
        )?;
        if output.administrative.is_some() {
            return Err(Error::new(
                ErrorCode::QueryType,
                "schema and index DDL is autocommit-only",
            ));
        }
        let vectors = resolve_embedding_mutations(
            &state.working,
            &[],
            &output.graph_mutations,
            text_embedding.as_deref(),
            provisional,
        )?;
        let working = Arc::new(stage_transaction_project(
            &state.working,
            &output.graph_mutations,
            &output.temporal_mutations,
            &vectors,
            unix_nanos(SystemTime::now())?,
        )?);
        let execution = stage_transaction_execution(
            state.execution.as_deref(),
            self.project,
            state.bookmark,
            &working,
            !output.graph_mutations.is_empty(),
            &output.temporal_mutations,
            &vectors,
        )?;
        let mut next = TransactionState {
            catalog: working.graph.catalog().clone(),
            snapshot: Arc::clone(&state.snapshot),
            working,
            execution,
            bookmark: state.bookmark,
            consistency: state.consistency,
            dependencies: state.dependencies.clone(),
            graph_mutations: state.graph_mutations.clone(),
            temporal_mutations: state.temporal_mutations.clone(),
            vector_mutations: state.vector_mutations.clone(),
            accounted_bytes: state.accounted_bytes,
        };
        merge_dependencies(&mut next.dependencies, output.dependencies.clone());
        next.graph_mutations.extend(output.graph_mutations.clone());
        next.temporal_mutations
            .extend(output.temporal_mutations.clone());
        next.vector_mutations.extend(vectors);
        next.accounted_bytes = transaction_retained_bytes(
            &next,
            self.database.0.transactions.maximum_encoded_bytes(),
        )?;
        self.database
            .0
            .transactions
            .resize(self.id, next.accounted_bytes)?;
        let emitted = emit_result(
            request.request_id,
            output.result,
            &next.catalog,
            &next.working.indexes,
            &mut |mut event| {
                budget.observe(&event)?;
                budget.settle(&mut event);
                emit(event)
            },
        );
        if let Err(error) = emitted {
            self.database
                .0
                .transactions
                .resize(self.id, state.accounted_bytes)?;
            return Err(error);
        }
        let terminal = self.resource.terminal.load(Ordering::Acquire);
        if terminal != 0 {
            let reason = if terminal == 2 {
                TransactionTerminal::SequencerChanged
            } else {
                TransactionTerminal::Expired
            };
            *lifecycle = TransactionLifecycle::Terminal(reason);
            return Err(match reason {
                TransactionTerminal::Expired => transaction_expired(),
                TransactionTerminal::SequencerChanged => transaction_sequencer_changed_error(),
            });
        }
        *state = next;
        Ok(())
    }

    fn commit(self: Box<Self>) -> Result<Bookmark> {
        self.ensure_active()?;
        self.database.0.transactions.mark_finalizing(self.id)?;
        let state = {
            let mut lifecycle = self.resource.lifecycle.lock();
            match std::mem::replace(&mut *lifecycle, TransactionLifecycle::Finalizing) {
                TransactionLifecycle::Active(state) => state,
                other => {
                    let error =
                        transaction_lifecycle_error(&other).unwrap_or_else(transaction_expired);
                    *lifecycle = other;
                    return Err(error);
                }
            }
        };
        let result = self.commit_state(state);
        self.database.0.transactions.finish(self.id);
        result
    }

    fn rollback(self: Box<Self>) -> Result<()> {
        self.database.0.transactions.finish(self.id);
        Ok(())
    }
}

impl DatabaseTransaction {
    fn commit_state(&self, state: TransactionState) -> Result<Bookmark> {
        self.database.verify_transaction_fence(self.fence)?;
        let current = self
            .database
            .0
            .state
            .read()
            .projects
            .get(&self.project)
            .cloned()
            .ok_or_else(|| Error::new(ErrorCode::ProjectFenced, "project was deleted"))?;
        validate_dependencies(&state.dependencies, &current, state.bookmark.index)?;
        let next_index = self
            .database
            .bookmark()
            .index
            .checked_add(1)
            .ok_or_else(|| Error::internal("log index exhausted"))?;
        let graph = state
            .graph_mutations
            .into_iter()
            .map(|mutation| retag_graph_mutation(mutation, next_index))
            .collect::<Vec<_>>();
        let temporal = state
            .temporal_mutations
            .into_iter()
            .map(|mut mutation| {
                mutation.sample.sequence_index = next_index;
                PersistedTemporalMutation {
                    entity_kind: mutation.entity_kind,
                    target: mutation.target,
                    sample: mutation.sample,
                    uses_commit_time: mutation.uses_commit_time,
                }
            })
            .collect::<Vec<_>>();
        let vectors = state
            .vector_mutations
            .into_iter()
            .map(|mutation| retag_vector_mutation(mutation, next_index))
            .collect::<Vec<_>>();
        let timeout = self
            .deadline
            .checked_duration_since(Instant::now())
            .ok_or_else(transaction_expired)?;
        let committed = self.database.commit_scoped_with_timeout_from(
            DatabaseMutation::Graph {
                project: self.project,
                validation: MutationValidation {
                    snapshot: state.bookmark,
                    dependencies: state.dependencies,
                },
                graph,
                temporal,
                vectors,
                administrative: None,
            },
            MutationKind::Graph,
            Some(self.project),
            None,
            AdmissionClass::Client,
            state.consistency,
            timeout,
            self.connection,
            Some(self.fence),
        )?;
        Ok(committed.response.bookmark)
    }
}

impl Drop for DatabaseTransaction {
    fn drop(&mut self) {
        self.database.0.transactions.finish(self.id);
    }
}

fn transaction_lifecycle_error(lifecycle: &TransactionLifecycle) -> Option<Error> {
    match lifecycle {
        TransactionLifecycle::Active(_) => None,
        TransactionLifecycle::Terminal(TransactionTerminal::Expired) => Some(transaction_expired()),
        TransactionLifecycle::Terminal(TransactionTerminal::SequencerChanged) => {
            Some(transaction_sequencer_changed_error())
        }
        TransactionLifecycle::Finalizing | TransactionLifecycle::Closed => {
            Some(transaction_expired())
        }
    }
}

impl Database {
    /// Read one finite partition prefix under the same publication view as its watermark.
    pub fn fetch_partition_bounded(
        &self,
        read: &crate::broker::PartitionRead<'_>,
    ) -> Result<(u64, Vec<(u64, Arc<crate::broker::PayloadRecord>)>)> {
        let state = self.0.state.read();
        if !state.projects.contains_key(&read.project) {
            return Err(Error::new(
                ErrorCode::ProjectNotFound,
                "project does not exist",
            ));
        }
        // Keeping the publication read lock also prevents payload reclamation until the bounded
        // load completes. No data from another publication enters this page or watermark.
        state.broker.fetch_partition_bounded(read, &self.0.segments)
    }
}

impl BrokerCoordinator for Database {
    fn submit(&self, command: BrokerCommand, wait: CommitAcknowledgement) -> Result<BrokerCommit> {
        let timeout = u32::try_from(self.0.request_timeout.as_millis()).unwrap_or(u32::MAX);
        self.submit_with_timeout_from(ConnectionId::new(), command, wait, timeout)
    }

    fn submit_from(
        &self,
        connection: ConnectionId,
        command: BrokerCommand,
        wait: CommitAcknowledgement,
    ) -> Result<BrokerCommit> {
        let timeout = u32::try_from(self.0.request_timeout.as_millis()).unwrap_or(u32::MAX);
        self.submit_with_timeout_from(connection, command, wait, timeout)
    }

    fn submit_with_timeout(
        &self,
        command: BrokerCommand,
        wait: CommitAcknowledgement,
        timeout_millis: u32,
    ) -> Result<BrokerCommit> {
        self.submit_with_timeout_from(ConnectionId::new(), command, wait, timeout_millis)
    }

    fn submit_with_timeout_from(
        &self,
        connection: ConnectionId,
        command: BrokerCommand,
        wait: CommitAcknowledgement,
        timeout_millis: u32,
    ) -> Result<BrokerCommit> {
        let project = broker_project(&command);
        let committed = self.commit_scoped_with_timeout_from(
            DatabaseMutation::Broker { command },
            MutationKind::Broker,
            Some(project),
            None,
            AdmissionClass::Broker,
            wait,
            Duration::from_millis(u64::from(timeout_millis.max(1))),
            connection,
            None,
        )?;
        let reply = committed
            .reply
            .ok_or_else(|| Error::internal("broker mutation returned no reply"))?;
        Ok(BrokerCommit {
            bookmark: committed.response.bookmark,
            reply,
            application: committed.application,
        })
    }

    fn snapshot(&self) -> Result<BrokerStateMachine> {
        Ok((*self.0.state.read().broker).clone())
    }

    fn reclaim_payload_storage(&self) -> Result<u64> {
        // Discovery must remain under the publication lock. A liveness set captured and then
        // used after releasing this lock could unlink a content-addressed file that a concurrent
        // mutation just made canonical. The batch is deliberately small, and unlink/sync does
        // no payload reads.
        let _apply = self.0.apply.lock();
        let state = self.0.state.read();
        let (candidates, live) = self.broker_reclamation_plan_locked(&state);
        let reclaimed = self.0.segments.reclaim_unreferenced(&candidates, &live)?;
        let known = self.finish_broker_reclamation(&reclaimed)?;
        let remaining = BROKER_RECLAIM_BATCH_SEGMENTS.saturating_sub(reclaimed.len());
        if remaining == 0 {
            return Ok(known);
        }
        let discovered = self
            .0
            .segments
            .reclaim_discovered_unreferenced(&live, remaining)?;
        known
            .checked_add(
                u64::try_from(discovered.len())
                    .map_err(|_| Error::internal("discovered broker segment count exceeds u64"))?,
            )
            .ok_or_else(|| Error::internal("reclaimed broker segment count overflow"))
    }

    fn subscribe_changes(&self) -> tokio::sync::watch::Receiver<u64> {
        self.0.broker_changes.subscribe()
    }

    fn projects_with_state(&self) -> Result<BTreeSet<ProjectId>> {
        Ok(self.0.state.read().broker.projects_with_state())
    }

    fn topic_metadata(&self, project: ProjectId) -> Result<Vec<(String, usize)>> {
        Ok(self.0.state.read().broker.topic_metadata(project))
    }

    fn committed_offset(
        &self,
        project: ProjectId,
        group: &str,
        topic: &str,
        partition: i32,
    ) -> Result<Option<u64>> {
        Ok(self
            .0
            .state
            .read()
            .broker
            .committed_offset(project, group, topic, partition))
    }

    fn group_leader(
        &self,
        project: ProjectId,
        group: &str,
        generation: i32,
    ) -> Result<Option<String>> {
        Ok(self
            .0
            .state
            .read()
            .broker
            .group_leader(project, group, generation)
            .map(str::to_owned))
    }

    fn group_assignment(
        &self,
        project: ProjectId,
        group: &str,
        generation: i32,
        member: &str,
    ) -> Result<Option<Vec<u8>>> {
        Ok(self
            .0
            .state
            .read()
            .broker
            .group_assignment(project, group, generation, member))
    }

    fn fetch_partition(
        &self,
        project: ProjectId,
        topic: &str,
        partition: i32,
        offset: u64,
        maximum_bytes: usize,
    ) -> Result<Vec<(u64, Arc<crate::broker::PayloadRecord>)>> {
        let (broker, _pins) = {
            let state = self.0.state.read();
            let broker = (*state.broker).clone();
            let descriptors = broker.partition_fetch_segment_descriptors(
                project,
                topic,
                partition,
                offset,
                maximum_bytes,
            )?;
            let pins = descriptors
                .iter()
                .map(|descriptor| self.0.segments.pin(descriptor))
                .collect::<Result<Vec<SegmentPin>>>()?;
            (broker, pins)
        };
        broker.fetch_partition(
            project,
            topic,
            partition,
            offset,
            maximum_bytes,
            &self.0.segments,
        )
    }

    fn list_offset(
        &self,
        project: ProjectId,
        topic: &str,
        partition: i32,
        timestamp: i64,
    ) -> Result<Option<(u64, i64)>> {
        self.0
            .state
            .read()
            .broker
            .list_offset(project, topic, partition, timestamp)
    }

    fn queue_info(
        &self,
        project: ProjectId,
        name: &str,
    ) -> Result<Option<crate::broker::QueueInfo>> {
        Ok(self.0.state.read().broker.queue_info(project, name))
    }

    fn fetch_stream_queue(
        &self,
        project: ProjectId,
        queue: &str,
        offset: crate::broker::StreamOffset,
        maximum: usize,
        consumer: u64,
        automatic_ack: bool,
    ) -> Result<Vec<crate::broker::Delivery>> {
        let (broker, _pins) = {
            let state = self.0.state.read();
            let broker = (*state.broker).clone();
            let descriptors =
                broker.stream_queue_fetch_segment_descriptors(project, queue, offset, maximum)?;
            let pins = descriptors
                .iter()
                .map(|descriptor| self.0.segments.pin(descriptor))
                .collect::<Result<Vec<SegmentPin>>>()?;
            (broker, pins)
        };
        broker.read_stream_queue(
            project,
            queue,
            offset,
            maximum,
            consumer,
            automatic_ack,
            &self.0.segments,
        )
    }
}

fn execute_on_project(
    project: &ProjectState,
    request: &QueryRequest,
    bookmark: Bookmark,
    mutation_revision: u64,
    capabilities: BindCapabilities,
    text_embedding: Option<&dyn TextEmbedding>,
    backend: Option<&dyn ExecutionBackend>,
) -> Result<ExecutionOutput> {
    execute_on_project_inner(
        project,
        request,
        bookmark,
        mutation_revision,
        capabilities,
        text_embedding,
        backend,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_on_project_streaming(
    project: &ProjectState,
    request: &QueryRequest,
    bookmark: Bookmark,
    mutation_revision: u64,
    capabilities: BindCapabilities,
    text_embedding: Option<&dyn TextEmbedding>,
    backend: Option<&dyn ExecutionBackend>,
    emit: &mut dyn FnMut(ExecutionStreamItem) -> Result<()>,
) -> Result<ExecutionOutput> {
    execute_on_project_inner(
        project,
        request,
        bookmark,
        mutation_revision,
        capabilities,
        text_embedding,
        backend,
        Some(emit),
    )
}

#[allow(clippy::too_many_arguments)]
fn execute_on_project_inner(
    project: &ProjectState,
    request: &QueryRequest,
    bookmark: Bookmark,
    mutation_revision: u64,
    capabilities: BindCapabilities,
    text_embedding: Option<&dyn TextEmbedding>,
    backend: Option<&dyn ExecutionBackend>,
    stream: Option<&mut dyn FnMut(ExecutionStreamItem) -> Result<()>>,
) -> Result<ExecutionOutput> {
    let parameters = request
        .parameters
        .iter()
        .map(|(name, value)| json_to_result(value).map(|value| (name.clone(), value)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let vector_indexes = project
        .indexes
        .definitions()
        .filter_map(|definition| {
            project
                .indexes
                .vector_search_source(&definition.name)
                .map(|(exact, approximate)| {
                    (
                        definition.name.clone(),
                        crate::cypher::VectorSearchSource {
                            property: definition.properties[0],
                            exact,
                            approximate,
                            profile_hash: project
                                .indexes
                                .profile()
                                .map_or([0_u8; 32], |profile| profile.profile_hash),
                        },
                    )
                })
        })
        .collect();
    let binding_catalog = project.graph.catalog();
    let prior_graph_mutations = &[][..];
    let prior_temporal_mutations = &[][..];
    let next_node_id = project.next_node_id;
    let next_edge_id = project.next_edge_id;
    // A canonical-host write already has a bound, point-sized mutation plan and must not pay an
    // O(graph) statistics rebuild merely because the previous commit invalidated the cache. Device
    // analytic reads use statistics; host writes and deliberately host-routed bookkeeping reads do
    // not need them for correctness and remain proportional to the rows they touch.
    let optimizer_statistics = backend.map(|_| {
        project.optimizer_statistics.get_or_init(|| {
            Arc::new(StatisticsSnapshot::collect_project(
                &project.graph,
                Some(&project.temporal),
                Some(&project.indexes),
            ))
        })
    });
    let mut context = ExecutionContext {
        project_id: project.id,
        graph: &project.graph,
        binding_catalog,
        prior_graph_mutations,
        temporal: Some(&project.temporal),
        prior_temporal_mutations,
        vector_indexes,
        scalar_indexes: Some(&project.indexes),
        text_embedding: project
            .indexes
            .profile()
            .and_then(|profile| text_embedding.filter(|embedding| embedding.profile() == profile)),
        parameters,
        bookmark,
        mutation_revision,
        resolved_time_nanos: unix_nanos(SystemTime::now())?,
        resolved_query_at_time_nanos: None,
        next_node_id,
        next_edge_id,
        predicate_versions: (*project.predicate_versions).clone(),
        capabilities,
        // The device row budget only bounds operators whose GPU command-buffer scratch is sized by
        // it. Push it to what the device can actually hold: the host backend has no command buffer
        // (its only limit is system memory), and a GPU is scaled to its admitted working set — so a
        // high-VRAM / large unified-memory machine gets a proportionally larger budget instead of a
        // fixed constant. A device with less headroom gets a smaller budget; admission never invents
        // capacity merely to preserve a historical test floor.
        max_result_rows: match backend {
            Some(backend) if backend.kind() != crate::gpu::BackendKind::Cpu => {
                (backend.available_query_scratch_bytes() / DEVICE_SCRATCH_BYTES_PER_ROW).max(1)
            }
            _ => HOST_QUERY_EXECUTION_ROW_ADDRESS_SPACE,
        },
        max_batch_rows: QUERY_STREAM_BATCH_ROWS,
        optimizer_statistics: optimizer_statistics.map(|statistics| &**statistics),
        backend,
        cancellation: request.cancellation.clone(),
        deadline: request.deadline,
    };
    match stream {
        Some(stream) => QueryEngine.execute_streaming(&request.query, &mut context, stream),
        None => QueryEngine.execute(&request.query, &mut context),
    }
}

pub(super) fn apply_mutation(
    state: &mut DatabaseState,
    mutation: DatabaseMutation,
    bookmark: Bookmark,
    commit_time_nanos: i64,
    segments: Option<&SegmentStore>,
) -> Result<Option<BrokerReply>> {
    apply_mutation_inner(state, mutation, bookmark, commit_time_nanos, segments)
}

fn apply_mutation_speculative(
    state: &mut DatabaseState,
    mutation: DatabaseMutation,
    bookmark: Bookmark,
    commit_time_nanos: i64,
    segments: Option<&SegmentStore>,
) -> Result<Option<BrokerReply>> {
    apply_mutation_inner(state, mutation, bookmark, commit_time_nanos, segments)
}

fn apply_mutation_inner(
    state: &mut DatabaseState,
    mutation: DatabaseMutation,
    bookmark: Bookmark,
    commit_time_nanos: i64,
    segments: Option<&SegmentStore>,
) -> Result<Option<BrokerReply>> {
    match mutation {
        DatabaseMutation::CreateProject { id, display_name } => {
            validate_project_name(&display_name)?;
            let normalized = normalize_name(&display_name);
            if state.projects.contains_key(&id) || state.names.contains_key(&normalized) {
                return Err(Error::new(
                    ErrorCode::TransactionConflict,
                    "project already exists",
                ));
            }
            state.names.insert(normalized, id);
            state.projects.insert(
                id,
                Arc::new(ProjectState {
                    id,
                    display_name,
                    graph: GraphStore::default().into(),
                    temporal: TemporalStore::default().into(),
                    predicate_versions: BTreeMap::new().into(),
                    indexes: IndexCatalog::default().into(),
                    next_node_id: 1,
                    next_edge_id: 1,
                    authority_revision: bookmark.index,
                    optimizer_statistics: OnceLock::new(),
                }),
            );
            Ok(None)
        }
        DatabaseMutation::RenameProject { id, display_name } => {
            validate_project_name(&display_name)?;
            let normalized = normalize_name(&display_name);
            if state
                .names
                .get(&normalized)
                .is_some_and(|existing| *existing != id)
            {
                return Err(Error::new(
                    ErrorCode::TransactionConflict,
                    "project name already exists",
                ));
            }
            let project = state
                .projects
                .get_mut(&id)
                .ok_or_else(|| Error::new(ErrorCode::ProjectNotFound, "project does not exist"))?;
            let project = Arc::make_mut(project);
            state.names.remove(&normalize_name(&project.display_name));
            project.display_name = display_name;
            state.names.insert(normalized, id);
            Ok(None)
        }
        DatabaseMutation::DropProject { id, cascade } => {
            if !cascade && project_has_data(state, id)? {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "committed non-cascade project drop targets non-empty data",
                ));
            }
            let project = state
                .projects
                .remove(&id)
                .ok_or_else(|| Error::new(ErrorCode::ProjectNotFound, "project does not exist"))?;
            state.names.remove(&normalize_name(&project.display_name));
            state.broker.drop_project(id)?;
            Ok(None)
        }
        DatabaseMutation::Graph {
            project,
            validation: _,
            graph,
            temporal,
            vectors,
            administrative,
        } => {
            let project_state = state
                .projects
                .get_mut(&project)
                .ok_or_else(|| Error::new(ErrorCode::ProjectFenced, "project does not exist"))?;
            let project_state = Arc::make_mut(project_state);
            for mutation in graph {
                let mutation = retag_graph_mutation(mutation, bookmark.index);
                update_next_ids(project_state, &mutation);
                project_state
                    .indexes
                    .before_graph_apply(&project_state.graph, &mutation)?;
                project_state.graph.apply(mutation.clone())?;
                project_state
                    .indexes
                    .after_graph_apply(&project_state.graph, &mutation)?;
            }
            for mut mutation in temporal {
                mutation.sample.sequence_index = bookmark.index;
                if mutation.uses_commit_time {
                    mutation.sample.event_time_nanos = commit_time_nanos;
                }
                project_state.temporal.append(
                    mutation.entity_kind,
                    mutation.target,
                    mutation.sample,
                    commit_time_nanos,
                )?;
            }
            if let Some(administrative) = administrative {
                apply_administrative(
                    project_state,
                    retag_administrative_mutation(administrative, bookmark.index),
                    commit_time_nanos,
                )?;
            }
            for mutation in vectors {
                project_state
                    .indexes
                    .apply_vector_mutation(&retag_vector_mutation(mutation, bookmark.index))?;
            }
            for version in project_state.predicate_versions.values_mut() {
                *version = bookmark.index;
            }
            project_state.authority_revision = bookmark.index;
            expire_optimizer_statistics_if_stale(project_state);
            Ok(None)
        }
        DatabaseMutation::Broker { command } => {
            let project = broker_project(&command);
            if !state.projects.contains_key(&project) {
                return Err(Error::new(
                    ErrorCode::ProjectFenced,
                    "broker mutation references a missing project",
                ));
            }
            let segments = segments.ok_or_else(|| {
                Error::internal("broker mutation requires the canonical payload segment store")
            })?;
            state.broker.apply(command, segments).map(Some)
        }
        DatabaseMutation::EmbeddingProfile { command } => {
            apply_embedding_profile_command(state, &command)?;
            Ok(None)
        }
        DatabaseMutation::Security { command } => {
            state.security =
                staged_security_command(state, &command, bookmark, commit_time_nanos / 1_000_000)?
                    .into();
            Ok(None)
        }
    }
}

type StagedEmbeddingProfile = (
    Option<EmbeddingProfileActivation>,
    Option<(ProjectId, IndexCatalog)>,
);

fn staged_embedding_profile_command(
    state: &DatabaseState,
    command: &EmbeddingProfileCommand,
) -> Result<StagedEmbeddingProfile> {
    let mut activation = state.embedding_activation.clone();
    let mut activated = None;
    match command {
        EmbeddingProfileCommand::Begin { project, profile } => {
            if activation.is_some() {
                return Err(Error::new(
                    ErrorCode::TransactionConflict,
                    "another embedding-profile activation is already in progress",
                ));
            }
            let project_state = state
                .projects
                .get(project)
                .ok_or_else(|| Error::new(ErrorCode::ProjectNotFound, "project does not exist"))?;
            if !project_state.indexes.embedding_profile_is_mutable() {
                return Err(Error::new(
                    ErrorCode::EmbeddingProfileImmutable,
                    "embedding profile is fixed by existing vector state",
                ));
            }
            activation = Some(EmbeddingProfileActivation::begin(
                *project,
                profile.clone(),
            )?);
        }
        EmbeddingProfileCommand::Acknowledge {
            project,
            profile_hash,
        } => {
            let pending = activation.as_mut().ok_or_else(|| {
                Error::new(
                    ErrorCode::TransactionConflict,
                    "no embedding-profile activation is in progress",
                )
            })?;
            if pending.barrier().subject()
                != &(crate::engine::ActivationSubject::EmbeddingProfile {
                    project: *project,
                    profile_hash: *profile_hash,
                })
            {
                return Err(Error::new(
                    ErrorCode::TransactionConflict,
                    "readiness acknowledgement names a different embedding profile",
                ));
            }
            pending.acknowledge()?;
        }
        EmbeddingProfileCommand::Activate {
            project,
            profile_hash,
        } => {
            let pending = activation.take().ok_or_else(|| {
                Error::new(
                    ErrorCode::TransactionConflict,
                    "no embedding-profile activation is in progress",
                )
            })?;
            if pending.barrier().subject()
                != &(crate::engine::ActivationSubject::EmbeddingProfile {
                    project: *project,
                    profile_hash: *profile_hash,
                })
            {
                return Err(Error::new(
                    ErrorCode::TransactionConflict,
                    "activation names a different embedding profile",
                ));
            }
            let profile = pending.finish()?;
            let project_state = state
                .projects
                .get(project)
                .ok_or_else(|| Error::new(ErrorCode::ProjectNotFound, "project does not exist"))?;
            let mut indexes = (*project_state.indexes).clone();
            indexes.activate_profile(profile)?;
            activated = Some((*project, indexes));
        }
        EmbeddingProfileCommand::Abort {
            project,
            profile_hash,
        } => {
            let pending = activation.as_ref().ok_or_else(|| {
                Error::new(
                    ErrorCode::TransactionConflict,
                    "no embedding-profile activation is in progress",
                )
            })?;
            if pending.barrier().subject()
                != &(crate::engine::ActivationSubject::EmbeddingProfile {
                    project: *project,
                    profile_hash: *profile_hash,
                })
            {
                return Err(Error::new(
                    ErrorCode::TransactionConflict,
                    "abort does not match the pending embedding profile",
                ));
            }
            activation = None;
        }
    }
    if let Some(pending) = &activation {
        pending.validate()?;
    }
    Ok((activation, activated))
}

fn validate_embedding_profile_command(
    state: &DatabaseState,
    command: &EmbeddingProfileCommand,
) -> Result<()> {
    staged_embedding_profile_command(state, command).map(|_| ())
}

fn apply_embedding_profile_command(
    state: &mut DatabaseState,
    command: &EmbeddingProfileCommand,
) -> Result<()> {
    let (activation, activated) = staged_embedding_profile_command(state, command)?;
    if let Some((project, indexes)) = activated {
        let project_state = state.projects.get_mut(&project).ok_or_else(|| {
            Error::new(ErrorCode::CorruptStorage, "activated project disappeared")
        })?;
        Arc::make_mut(project_state).indexes = indexes.into();
    }
    state.embedding_activation = activation;
    Ok(())
}

fn staged_security_command(
    state: &DatabaseState,
    command: &SecurityCommand,
    bookmark: Bookmark,
    now_millis: i64,
) -> Result<SecurityState> {
    let mut security = (*state.security).clone();
    match command {
        SecurityCommand::RegisterCredential { record } => security
            .client_credentials_mut()
            .register(record.clone(), now_millis)?,
        SecurityCommand::RotateCredential {
            old_fingerprint,
            replacement,
        } => security.client_credentials_mut().rotate(
            *old_fingerprint,
            replacement.clone(),
            bookmark.index,
            now_millis,
        )?,
        SecurityCommand::RevokeCredential { fingerprint } => security
            .client_credentials_mut()
            .revoke(*fingerprint, bookmark.index)?,
        SecurityCommand::CleanupExpired => {
            security
                .client_credentials_mut()
                .cleanup_expired(now_millis);
        }
    }
    security.validate()?;
    Ok(security)
}

fn validate_resolved_database_command(
    state: &DatabaseState,
    command: &WriteCommand,
    position: Bookmark,
) -> Result<DatabaseMutation> {
    command.validate()?;
    let mutation = decode_database_mutation(&command.payload)?;
    validate_database_envelope(command.kind, command.project_id, &mutation)?;
    if state.applied.index.checked_add(1) != Some(position.index) {
        return Err(Error::retryable(
            ErrorCode::WriteAdmissionFull,
            "database state is not the predecessor of the reserved local write position",
            Some(1),
        ));
    }
    let intent_digest = request_intent_digest(&mutation)?;
    if let Some(request_id) = command.request_id
        && let Some(record) = state.request_results.get(&request_id)
    {
        if record.intent_digest != intent_digest {
            return Err(Error::new(
                ErrorCode::ProtocolViolation,
                "request ID was already committed for a different mutation: the request ID is an \
                 idempotency key, and every new mutation needs its own fresh UUID (a reused or nil \
                 ID fails every write after the first)",
            ));
        }
        return Ok(mutation);
    }
    let reply = validate_sequencer_mutation(
        state,
        &mutation,
        position,
        millis_to_nanos(command.commit_time_millis)?,
    )?;
    let response = encode_database_value(&reply, "validated apply response")?;
    if response.len() > MAX_SINGLE_REQUEST_RESULT_BYTES {
        return Err(Error::new(
            ErrorCode::ResultBudgetExceeded,
            "mutation response exceeds the durable result bound",
        ));
    }
    Ok(mutation)
}

#[cfg(test)]
fn stage_ordered_database_command(
    base: &DatabaseState,
    command: &WriteCommand,
    mutation: DatabaseMutation,
    position: Bookmark,
    segments: &SegmentStore,
) -> Result<DatabaseState> {
    let intent_digest = request_intent_digest(&mutation)?;
    stage_ordered_database_command_with_digest(
        base,
        command,
        mutation,
        position,
        segments,
        intent_digest,
    )
}

fn stage_ordered_database_command_with_digest(
    base: &DatabaseState,
    command: &WriteCommand,
    mutation: DatabaseMutation,
    position: Bookmark,
    segments: &SegmentStore,
    intent_digest: [u8; 32],
) -> Result<DatabaseState> {
    if !matches!(
        mutation,
        DatabaseMutation::Graph { .. } | DatabaseMutation::Broker { .. }
    ) {
        return Err(Error::internal(
            "only graph and broker mutations may enter the ordered pending-state overlay",
        ));
    }
    if base.applied.index.checked_add(1) != Some(position.index) {
        return Err(Error::retryable(
            ErrorCode::WriteAdmissionFull,
            "ordered database state is not the reservation predecessor",
            Some(1),
        ));
    }
    let mut staged = base.clone();
    // The digest covers the planned mutation — allocated node/edge ids and the planning snapshot
    // included — so only a command-level retry of the same planned write replays. A client
    // re-issuing even a byte-identical statement plans a new mutation and lands here: reuse of a
    // request ID across /api/query calls is always this error, never a silent success.
    if let Some(request_id) = command.request_id
        && let Some(record) = staged.request_results.get(&request_id)
    {
        if record.intent_digest != intent_digest {
            return Err(Error::new(
                ErrorCode::ProtocolViolation,
                "request ID was already reserved for a different mutation: the request ID is an \
                 idempotency key, and every new mutation needs its own fresh UUID (a reused or nil \
                 ID fails every write after the first)",
            ));
        }
        staged.applied = position;
        staged.last_payload_checksum = None;
        staged.last_response = record.response.clone().into();
        prune_request_results(&mut staged)?;
        return Ok(staged);
    }

    let commit_time_nanos = millis_to_nanos(command.commit_time_millis)?;
    validate_sequencer_mutation(&staged, &mutation, position, commit_time_nanos)?;
    let reply = apply_mutation_speculative(
        &mut staged,
        mutation,
        position,
        commit_time_nanos,
        Some(segments),
    )?;
    let response = encode_database_value(&reply, "ordered mutation response")?;
    if response.len() > MAX_SINGLE_REQUEST_RESULT_BYTES {
        return Err(Error::new(
            ErrorCode::ResultBudgetExceeded,
            "ordered mutation response exceeds the durable result bound",
        ));
    }
    staged.applied = position;
    staged.last_payload_checksum = None;
    staged.last_response = response.clone().into();
    if let Some(request_id) = command.request_id {
        retain_request_result(&mut staged, request_id, intent_digest, response)?;
    }
    prune_request_results(&mut staged)?;
    Ok(staged)
}

fn retain_pending_broker_segments(
    database: &DatabaseInner,
    owner: Uuid,
    descriptors: &[SegmentDescriptor],
) -> Result<()> {
    let mut retained = database.pending_broker_segments.lock();
    let mut processed = Vec::new();
    for descriptor in descriptors {
        let result = if let Some(existing) = retained.get_mut(descriptor) {
            existing.owners.insert(owner);
            Ok(())
        } else {
            database.segments.pin(descriptor).map(|pin| {
                retained.insert(
                    descriptor.clone(),
                    PendingBrokerSegment {
                        _pin: pin,
                        owners: BTreeSet::from([owner]),
                    },
                );
            })
        };
        if let Err(error) = result {
            for processed in processed {
                let remove = retained.get_mut(&processed).is_some_and(|record| {
                    record.owners.remove(&owner);
                    record.owners.is_empty()
                });
                if remove {
                    retained.remove(&processed);
                }
            }
            drop(retained);
            database
                .retired_broker_segments
                .lock()
                .extend(descriptors.iter().cloned());
            return Err(error);
        }
        processed.push(descriptor.clone());
    }
    Ok(())
}

fn release_pending_broker_segments(
    database: &DatabaseInner,
    owner: Uuid,
    descriptors: &[SegmentDescriptor],
    rejected: bool,
) {
    let mut pending = database.pending_broker_segments.lock();
    let mut unpinned = Vec::new();
    for descriptor in descriptors {
        let remove = pending.get_mut(descriptor).is_some_and(|record| {
            record.owners.remove(&owner);
            record.owners.is_empty()
        });
        if remove {
            pending.remove(descriptor);
            unpinned.push(descriptor.clone());
        }
    }
    drop(pending);
    if rejected && !unpinned.is_empty() {
        database.retired_broker_segments.lock().extend(unpinned);
    }
}

fn validate_sequencer_mutation(
    state: &DatabaseState,
    mutation: &DatabaseMutation,
    bookmark: Bookmark,
    commit_time_nanos: i64,
) -> Result<Option<BrokerReply>> {
    match mutation {
        DatabaseMutation::CreateProject { id, display_name } => {
            validate_project_name(display_name)?;
            if state.projects.contains_key(id)
                || state.names.contains_key(&normalize_name(display_name))
            {
                return Err(Error::new(
                    ErrorCode::TransactionConflict,
                    "project already exists",
                ));
            }
            Ok(None)
        }
        DatabaseMutation::RenameProject { id, display_name } => {
            validate_project_name(display_name)?;
            if state
                .names
                .get(&normalize_name(display_name))
                .is_some_and(|existing| existing != id)
            {
                return Err(Error::new(
                    ErrorCode::TransactionConflict,
                    "project name already exists",
                ));
            }
            if !state.projects.contains_key(id) {
                return Err(Error::new(
                    ErrorCode::ProjectNotFound,
                    "project does not exist",
                ));
            }
            Ok(None)
        }
        DatabaseMutation::DropProject { id, cascade } => {
            if !state.projects.contains_key(id) {
                return Err(Error::new(
                    ErrorCode::ProjectNotFound,
                    "project does not exist",
                ));
            }
            if !*cascade && project_has_data(state, *id)? {
                return Err(Error::new(
                    ErrorCode::TransactionConflict,
                    "project contains data; DROP PROJECT requires CASCADE",
                ));
            }
            Ok(None)
        }
        DatabaseMutation::Graph {
            project,
            validation,
            temporal,
            administrative,
            ..
        } => {
            validate_mutation_plan(validation)?;
            if validation.snapshot.index > state.applied.index {
                return Err(Error::retryable(
                    ErrorCode::TransactionConflict,
                    "mutation planning snapshot is ahead of ordered state",
                    None,
                ));
            }
            let project_state = state
                .projects
                .get(project)
                .ok_or_else(|| Error::new(ErrorCode::ProjectFenced, "project does not exist"))?;
            validate_dependencies(
                &validation.dependencies,
                project_state,
                validation.snapshot.index,
            )?;
            for mutation in temporal {
                let mut sample = mutation.sample.clone();
                sample.sequence_index = bookmark.index;
                if mutation.uses_commit_time {
                    sample.event_time_nanos = commit_time_nanos;
                }
                project_state.temporal.validate_append(
                    mutation.entity_kind,
                    mutation.target,
                    &sample,
                    commit_time_nanos,
                )?;
            }
            if let Some(administrative) = administrative {
                let mut staged = project_state.as_ref().clone();
                apply_administrative(&mut staged, administrative.clone(), commit_time_nanos)?;
            }
            Ok(None)
        }
        DatabaseMutation::Broker { command } => {
            if !state.projects.contains_key(&broker_project(command)) {
                return Err(Error::new(
                    ErrorCode::ProjectFenced,
                    "broker mutation references a missing project",
                ));
            }
            state.broker.validate_command(command)?;
            Ok(None)
        }
        DatabaseMutation::EmbeddingProfile { command } => {
            validate_embedding_profile_command(state, command)?;
            Ok(None)
        }
        DatabaseMutation::Security { command } => {
            staged_security_command(state, command, bookmark, commit_time_nanos / 1_000_000)?;
            Ok(None)
        }
    }
}

fn retain_request_result(
    state: &mut DatabaseState,
    request_id: Uuid,
    intent_digest: [u8; 32],
    response: Vec<u8>,
) -> Result<()> {
    if let Some(existing) = state.request_results.get(&request_id) {
        return if existing.intent_digest == intent_digest && existing.response == response {
            Ok(())
        } else {
            Err(Error::new(
                ErrorCode::CorruptStorage,
                "idempotency result changed for an existing request ID",
            ))
        };
    }
    if response.len() > MAX_SINGLE_REQUEST_RESULT_BYTES {
        return Err(Error::new(
            ErrorCode::ResultBudgetExceeded,
            "committed response exceeds the idempotency-result bound",
        ));
    }
    if state
        .request_result_order
        .contains_key(&state.applied.index)
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "multiple idempotency results share one local write index",
        ));
    }
    let response_bytes = u64::try_from(response.len())
        .map_err(|_| Error::internal("idempotency response size overflow"))?;
    state.request_result_bytes = state
        .request_result_bytes
        .checked_add(response_bytes)
        .ok_or_else(|| Error::internal("idempotency result byte accounting overflow"))?;
    state
        .request_result_order
        .insert(state.applied.index, request_id);
    state.request_results.insert(
        request_id,
        RequestResultRecord {
            index: state.applied.index,
            intent_digest,
            response,
        },
    );
    Ok(())
}

fn build_database_snapshot_file(
    database: &Database,
    bookmark: Bookmark,
    destination: &Path,
) -> Result<BackendSnapshot> {
    // Freeze the canonical Arc-backed project roots and pin broker payload segments while holding
    // the publication lock. Encoding and file IO then run without blocking mutation apply.
    let (state, segment_pins) = {
        let _apply = database.0.apply.lock();
        let state = database.0.state.read();
        if state.applied != bookmark {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "database state does not exactly match the requested snapshot bookmark",
            ));
        }
        validate_database_state(&state)?;
        let frozen = state.clone();
        let mut segment_pins = BTreeMap::<[u8; 32], (SegmentDescriptor, SegmentPin)>::new();
        for descriptor in frozen.broker.payload_segments() {
            if let Some((existing, _)) = segment_pins.get(&descriptor.checksum) {
                if existing.bytes != descriptor.bytes {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        "snapshot content address has inconsistent segment lengths",
                    ));
                }
                continue;
            }
            let pin = database.0.segments.pin(&descriptor)?;
            segment_pins.insert(descriptor.checksum, (descriptor, pin));
        }
        (frozen, segment_pins)
    };
    let state_path = snapshot_temporary_path(destination, "state")?;
    let attachment_directory = snapshot_attachment_staging_directory(destination)?;
    let result = (|| -> Result<BackendSnapshot> {
        let mut state_file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&state_path)?;
        ciborium::ser::into_writer(&state, &mut state_file).map_err(|error| {
            Error::invalid_data(format!(
                "database checkpoint state encoding failed: {error}"
            ))
        })?;
        crate::storage::sync_durable(&state_file)?;
        let state_bytes = state_file.metadata()?.len();
        drop(state_file);

        let manifest = DatabaseCheckpointManifest {
            format_version: DATABASE_SNAPSHOT_FORMAT,
            store_id: database.0.store_id,
            included: bookmark,
            state_bytes,
            broker_segments: state.broker.payload_segments(),
        };
        let manifest_bytes = encode_database_value(&manifest, "database checkpoint manifest")?;
        if manifest_bytes.is_empty() || manifest_bytes.len() > 16 * 1024 * 1024 {
            return Err(Error::new(
                ErrorCode::ResultBudgetExceeded,
                "database checkpoint manifest is oversized",
            ));
        }
        let manifest_len = u32::try_from(manifest_bytes.len()).map_err(|_| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "database checkpoint manifest is oversized",
            )
        })?;
        let mut output = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(destination)?;
        output.write_all(&DATABASE_SNAPSHOT_MAGIC)?;
        output.write_all(&manifest_len.to_be_bytes())?;
        output.write_all(&manifest_bytes)?;
        copy_exact_file(&state_path, &mut output, state_bytes)?;
        crate::storage::sync_durable(&output)?;
        let output_bytes = output.metadata()?.len();
        drop(output);

        fs::create_dir(&attachment_directory)?;
        let mut attachments = Vec::new();
        for (checksum, (descriptor, pin)) in &segment_pins {
            let path = attachment_directory.join(hex::encode(checksum));
            pin.link_raw_to(&path)?;
            attachments.push(SnapshotAttachment::new(path, descriptor.bytes, *checksum)?);
        }
        crate::storage::sync_durable(&File::open(&attachment_directory)?)?;
        BackendSnapshot::new(destination.to_owned(), output_bytes, attachments)
    })();
    let _ignored = fs::remove_file(&state_path);
    if result.is_err() {
        let _ignored = fs::remove_file(destination);
        let _ignored = fs::remove_dir_all(&attachment_directory);
    }
    result
}

fn install_database_snapshot_file(
    database: &Database,
    bookmark: Bookmark,
    snapshot: &BackendSnapshot,
) -> Result<()> {
    let snapshot_path = snapshot.state_path();
    let mut file = File::open(snapshot_path)?;
    let mut magic = [0_u8; 8];
    file.read_exact(&mut magic)?;
    if magic != DATABASE_SNAPSHOT_MAGIC {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "database checkpoint format is unsupported",
        ));
    }
    let mut length = [0_u8; 4];
    file.read_exact(&mut length)?;
    let manifest_len = u32::from_be_bytes(length) as usize;
    if manifest_len == 0 || manifest_len > 16 * 1024 * 1024 {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "database checkpoint manifest is oversized",
        ));
    }
    let mut manifest_bytes = vec![0_u8; manifest_len];
    file.read_exact(&mut manifest_bytes)?;
    let manifest: DatabaseCheckpointManifest =
        decode_database_value(&manifest_bytes, "database checkpoint manifest")?;
    if manifest.format_version != DATABASE_SNAPSHOT_FORMAT
        || manifest.store_id != database.0.store_id
        || manifest.included != bookmark
        || manifest.state_bytes == 0
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "database checkpoint metadata is invalid",
        ));
    }
    let mut state_reader = (&mut file).take(manifest.state_bytes);
    let mut state: DatabaseState =
        ciborium::de::from_reader(&mut state_reader).map_err(|error| {
            Error::new(
                ErrorCode::CorruptStorage,
                format!("database checkpoint state is invalid: {error}"),
            )
        })?;
    if state_reader.limit() != 0 || state.applied != bookmark {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "database checkpoint state length or bookmark is invalid",
        ));
    }
    for (project_id, project) in &mut state.projects {
        let project_state = Arc::make_mut(project);
        if let Err(structure_error) = project_state.graph.validate_structure() {
            let quarantined = project_state
                .graph
                .quarantine_invalid_relationships()
                .map_err(|recovery_error| {
                    Error::new(
                        ErrorCode::CorruptStorage,
                        format!(
                            "checkpoint project {project_id} cannot be recovered: {}; relationship quarantine failed: {}",
                            structure_error.message, recovery_error.message
                        ),
                    )
                })?;
            tracing::warn!(
                project = %project_id,
                relationships = quarantined,
                reason = %structure_error.message,
                "recovered checkpoint by quarantining invalid relationship state"
            );
        }
    }
    validate_database_state(&state)?;
    let expected_broker_segments = state.broker.payload_segments();
    if expected_broker_segments != manifest.broker_segments {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "database checkpoint broker payload segment set is incomplete",
        ));
    }
    let broker_segments = expected_broker_segments;
    validate_database_checkpoint_state_extent(&manifest, file.metadata()?.len(), manifest_len)?;
    let attachments = validate_database_snapshot_attachments(snapshot, &broker_segments)?;

    if database.0.state.read().applied.index > bookmark.index {
        return Ok(());
    }

    // Immutable attachment installation is content-addressed and may safely run before the
    // publication gate. A stale snapshot can leave only reusable orphan files, never visible
    // canonical state, while large file copies do not block current reads or mutation apply.
    for descriptor in &broker_segments {
        let attachment = attachments.get(&descriptor.checksum).ok_or_else(|| {
            Error::new(
                ErrorCode::CorruptStorage,
                "database checkpoint broker attachment is absent",
            )
        })?;
        database.0.segments.import_raw_from(
            descriptor,
            &mut File::open(&attachment.path)?,
            DATABASE_SNAPSHOT_COPY_BYTES,
        )?;
    }
    state
        .broker
        .validate_payload_segments(&database.0.segments)?;
    let replacement_images = state
        .projects
        .iter()
        .map(|(project, project_state)| {
            ResidentProjectImage::build(
                *project,
                bookmark,
                &project_state.graph,
                &project_state.temporal,
                &project_state.indexes,
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let _apply = database.0.apply.lock();
    let mut current = database.0.state.write();
    if current.applied.index > bookmark.index {
        let current_broker_segments = current
            .broker
            .payload_segments()
            .into_iter()
            .map(|descriptor| descriptor.file_name)
            .collect::<BTreeSet<_>>();
        database.0.retired_broker_segments.lock().extend(
            broker_segments
                .iter()
                .filter(|descriptor| !current_broker_segments.contains(&descriptor.file_name))
                .cloned(),
        );
        return Ok(());
    }
    reset_ephemeral_after_snapshot_install(&database.0)?;
    {
        let mut execution = database.0.execution.write();
        if let Some(execution) = execution.as_deref_mut() {
            execution.replace_all_projects(replacement_images)?;
            for project in state.projects.keys().copied().collect::<Vec<_>>() {
                rebind_shared_project(&mut state, execution, project)?;
            }
        }
    }
    *current = state;
    Ok(())
}

fn validate_database_snapshot_attachments<'a>(
    snapshot: &'a BackendSnapshot,
    broker_segments: &[SegmentDescriptor],
) -> Result<BTreeMap<[u8; 32], &'a SnapshotAttachment>> {
    let mut expected = BTreeMap::<[u8; 32], u64>::new();
    for descriptor in broker_segments {
        if let Some(previous) = expected.insert(descriptor.checksum, descriptor.bytes)
            && previous != descriptor.bytes
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "database checkpoint content address has inconsistent lengths",
            ));
        }
    }
    if expected.len() != snapshot.attachments.len() {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "database checkpoint attachment set is incomplete",
        ));
    }
    let mut actual = BTreeMap::new();
    for attachment in &snapshot.attachments {
        if expected.get(&attachment.checksum) != Some(&attachment.bytes)
            || actual.insert(attachment.checksum, attachment).is_some()
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "database checkpoint attachment metadata is invalid",
            ));
        }
        let metadata = fs::symlink_metadata(&attachment.path)?;
        if !metadata.is_file()
            || metadata.file_type().is_symlink()
            || metadata.len() != attachment.bytes
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "database checkpoint attachment file is invalid",
            ));
        }
    }
    Ok(actual)
}

fn validate_database_checkpoint_state_extent(
    manifest: &DatabaseCheckpointManifest,
    file_bytes: u64,
    manifest_bytes: usize,
) -> Result<()> {
    let manifest_bytes = u64::try_from(manifest_bytes)
        .map_err(|_| Error::new(ErrorCode::CorruptStorage, "manifest size exceeds u64"))?;
    let declared = 12_u64
        .checked_add(manifest_bytes)
        .and_then(|bytes| bytes.checked_add(manifest.state_bytes))
        .ok_or_else(|| {
            Error::new(
                ErrorCode::CorruptStorage,
                "database checkpoint declared length overflow",
            )
        })?;
    if declared != file_bytes {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "database checkpoint declared extent differs from its file",
        ));
    }
    Ok(())
}

/// Snapshot publication is serialized with state-machine apply. A state-dependent sequencer
/// reservation must therefore be absent: installing over one would detach its ordered overlay
/// from the log position it validated. Any owner pins left without a reservation are definitive
/// crash/abort leftovers; the installed snapshot is authoritative, so release and enqueue them
/// for bounded reclamation before exposing the replacement state.
fn reset_ephemeral_after_snapshot_install(database: &DatabaseInner) -> Result<()> {
    let mut ordered = database.ordered_overlay.lock();
    if !ordered.reservations.is_empty() {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "cannot install a snapshot over ordered command reservations",
        ));
    }
    ordered.states.clear();
    drop(ordered);

    let abandoned = {
        let mut pending = database.pending_broker_segments.lock();
        let abandoned = pending.keys().cloned().collect::<Vec<_>>();
        pending.clear();
        abandoned
    };
    if !abandoned.is_empty() {
        database.retired_broker_segments.lock().extend(abandoned);
    }
    Ok(())
}

fn validate_database_state(state: &DatabaseState) -> Result<()> {
    validate_request_results(state)?;
    state.broker.validate_state()?;
    state.security.validate()?;
    if let Some(pending) = &state.embedding_activation {
        pending.validate()?;
    }
    for (id, project) in &state.projects {
        if id != &project.id {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "checkpoint project key differs from its immutable ID",
            ));
        }
        if state.names.get(&normalize_name(&project.display_name)) != Some(id) {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "checkpoint project name index is inconsistent",
            ));
        }
        project.graph.validate_structure()?;
    }
    if state.names.len() != state.projects.len() {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "checkpoint project catalog cardinality is inconsistent",
        ));
    }
    Ok(())
}

fn copy_exact_file(path: &Path, output: &mut impl Write, expected: u64) -> Result<()> {
    let mut input = File::open(path)?;
    let mut remaining = expected;
    let mut buffer = vec![0_u8; DATABASE_SNAPSHOT_COPY_BYTES];
    while remaining > 0 {
        let requested = usize::try_from(remaining.min(buffer.len() as u64))
            .map_err(|_| Error::internal("checkpoint copy length overflow"))?;
        input.read_exact(&mut buffer[..requested])?;
        output.write_all(&buffer[..requested])?;
        remaining -= requested as u64;
    }
    let mut trailing = [0_u8; 1];
    if input.read(&mut trailing)? != 0 {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "checkpoint state exceeds its declared length",
        ));
    }
    Ok(())
}

fn snapshot_temporary_path(path: &Path, role: &str) -> Result<PathBuf> {
    let parent = path
        .parent()
        .ok_or_else(|| Error::invalid_data("checkpoint path has no parent"))?;
    Ok(parent.join(format!(".{role}.{}.checkpoint.tmp", Uuid::new_v4())))
}

fn snapshot_attachment_staging_directory(path: &Path) -> Result<PathBuf> {
    let parent = path
        .parent()
        .ok_or_else(|| Error::invalid_data("checkpoint path has no parent"))?;
    Ok(parent.join(format!(".attachments.{}.checkpoint.tmp", Uuid::new_v4())))
}

fn standalone_snapshot_attachment_directory(path: &Path) -> Result<PathBuf> {
    let parent = path
        .parent()
        .ok_or_else(|| Error::invalid_data("standalone snapshot path has no parent"))?;
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| Error::invalid_data("standalone snapshot has no UTF-8 file name"))?;
    Ok(parent.join(format!("{name}.attachments")))
}

/// Move the verified attachment set built with the canonical state file into its durable name.
/// `LATEST` is written only after this returns, so recovery can never observe a snapshot whose
/// broker bytes have not been published yet.
fn publish_standalone_snapshot_attachments(
    destination: &Path,
    snapshot: &BackendSnapshot,
) -> Result<()> {
    if snapshot.attachments.is_empty() {
        return Ok(());
    }
    let staging = snapshot.attachments[0]
        .path
        .parent()
        .ok_or_else(|| Error::invalid_data("snapshot attachment has no parent directory"))?;
    if snapshot
        .attachments
        .iter()
        .any(|attachment| attachment.path.parent() != Some(staging))
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "standalone snapshot attachments do not share one staging directory",
        ));
    }
    let published = standalone_snapshot_attachment_directory(destination)?;
    if published.exists() {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "standalone snapshot attachment directory already exists",
        ));
    }
    std::fs::rename(staging, &published)?;
    if let Some(parent) = published.parent() {
        crate::storage::sync_durable(&File::open(parent)?)?;
    }
    Ok(())
}

fn standalone_snapshot_paths(dir: &Path) -> Vec<PathBuf> {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return Vec::new();
    };
    let mut paths = entries
        .flatten()
        .map(|entry| entry.path())
        .filter(|path| {
            path.file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("snapshot-") && name.ends_with(".igdb"))
        })
        .collect::<Vec<_>>();
    paths.sort_by(|left, right| right.file_name().cmp(&left.file_name()));
    paths
}

fn standalone_snapshot_paths_with_latest_first(dir: &Path) -> Vec<PathBuf> {
    let mut paths = standalone_snapshot_paths(dir);
    let latest = std::fs::read_to_string(dir.join("LATEST"))
        .ok()
        .map(|name| dir.join(name.trim()))
        .filter(|path| path.exists());
    if let Some(latest) = latest
        && let Some(position) = paths.iter().position(|path| *path == latest)
    {
        paths.remove(position);
        paths.insert(0, latest);
    }
    paths
}

fn quarantine_standalone_snapshot(path: &Path) -> Result<PathBuf> {
    let parent = path
        .parent()
        .ok_or_else(|| Error::invalid_data("standalone snapshot has no parent directory"))?;
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| Error::invalid_data("standalone snapshot has no UTF-8 file name"))?;
    let quarantined = parent.join(format!("{name}.corrupt-{}", Uuid::new_v4()));
    std::fs::rename(path, &quarantined)?;
    let attachments = standalone_snapshot_attachment_directory(path)?;
    if attachments.exists() {
        let quarantined_name = quarantined
            .file_name()
            .and_then(|name| name.to_str())
            .ok_or_else(|| Error::invalid_data("quarantined snapshot name is not UTF-8"))?;
        std::fs::rename(
            attachments,
            parent.join(format!("{quarantined_name}.attachments")),
        )?;
    }
    crate::storage::sync_durable(&File::open(parent)?)?;
    Ok(quarantined)
}

fn read_standalone_snapshot_state(
    path: &Path,
) -> Result<(DatabaseCheckpointManifest, DatabaseState)> {
    let mut file = File::open(path)?;
    let mut magic = [0_u8; 8];
    file.read_exact(&mut magic)?;
    if magic != DATABASE_SNAPSHOT_MAGIC {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "standalone snapshot format is unsupported",
        ));
    }
    let mut length = [0_u8; 4];
    file.read_exact(&mut length)?;
    let manifest_len = u32::from_be_bytes(length) as usize;
    if manifest_len == 0 || manifest_len > 16 * 1024 * 1024 {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "standalone snapshot manifest is oversized",
        ));
    }
    let mut manifest_bytes = vec![0_u8; manifest_len];
    file.read_exact(&mut manifest_bytes)?;
    let manifest: DatabaseCheckpointManifest =
        decode_database_value(&manifest_bytes, "standalone snapshot manifest")?;
    if manifest.format_version != DATABASE_SNAPSHOT_FORMAT || manifest.state_bytes == 0 {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "standalone snapshot format version or state length is invalid",
        ));
    }
    let mut state_reader = (&mut file).take(manifest.state_bytes);
    let state: DatabaseState = ciborium::de::from_reader(&mut state_reader).map_err(|error| {
        Error::new(
            ErrorCode::CorruptStorage,
            format!("standalone snapshot state is invalid: {error}"),
        )
    })?;
    if state_reader.limit() != 0 || state.applied != manifest.included {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "standalone snapshot state length or bookmark is invalid",
        ));
    }
    validate_database_checkpoint_state_extent(&manifest, file.metadata()?.len(), manifest_len)?;
    validate_database_state(&state)?;
    if state.broker.payload_segments() != manifest.broker_segments {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "standalone snapshot broker payload segment set is incomplete",
        ));
    }
    Ok((manifest, state))
}

fn standalone_snapshot_is_complete(path: &Path) -> Result<bool> {
    let (manifest, _) = read_standalone_snapshot_state(path)?;
    let segments = manifest.broker_segments;
    if segments.is_empty() {
        return Ok(true);
    }
    let attachments = standalone_snapshot_attachment_directory(path)?;
    for descriptor in segments {
        if SnapshotAttachment::new(
            attachments.join(hex::encode(descriptor.checksum)),
            descriptor.bytes,
            descriptor.checksum,
        )
        .is_err()
        {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Reads the manifest header of a standalone snapshot file (`MAGIC | len | manifest | state`),
/// returning the committed bookmark and any referenced broker payload segments without decoding
/// the full state body.
fn read_standalone_snapshot_manifest(path: &Path) -> Result<(Bookmark, Vec<SegmentDescriptor>)> {
    let mut file = File::open(path)?;
    let mut magic = [0_u8; 8];
    file.read_exact(&mut magic)?;
    if magic != DATABASE_SNAPSHOT_MAGIC {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "standalone snapshot format is unsupported",
        ));
    }
    let mut length = [0_u8; 4];
    file.read_exact(&mut length)?;
    let manifest_len = u32::from_be_bytes(length) as usize;
    if manifest_len == 0 || manifest_len > 16 * 1024 * 1024 {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "standalone snapshot manifest is oversized",
        ));
    }
    let mut manifest_bytes = vec![0_u8; manifest_len];
    file.read_exact(&mut manifest_bytes)?;
    let manifest: DatabaseCheckpointManifest =
        decode_database_value(&manifest_bytes, "standalone snapshot manifest")?;
    if manifest.format_version != DATABASE_SNAPSHOT_FORMAT {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "standalone snapshot format version mismatch",
        ));
    }
    Ok((manifest.included, manifest.broker_segments))
}

/// Retains the newly published snapshot and one validated predecessor. The WAL is compacted only
/// through that predecessor, so either retained snapshot is a real recovery authority rather than
/// a stale file whose suffix has already been discarded. Invalid snapshots are renamed for
/// forensic inspection; only older validated generations are pruned.
fn prune_standalone_snapshots(dir: &Path, keep: &Path) {
    let mut retained = BTreeSet::from([keep.to_owned()]);
    for path in standalone_snapshot_paths(dir) {
        if path == keep {
            continue;
        }
        match standalone_snapshot_is_complete(&path) {
            Ok(true) if retained.len() < 2 => {
                retained.insert(path);
            }
            Ok(true) => {
                let _ = std::fs::remove_file(&path);
                if let Ok(attachments) = standalone_snapshot_attachment_directory(&path) {
                    let _ = std::fs::remove_dir_all(attachments);
                }
            }
            Ok(false) | Err(_) => {
                let _ = quarantine_standalone_snapshot(&path);
            }
        }
    }

    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if name.starts_with(".attachments.") && name.ends_with(".checkpoint.tmp") {
            let _ = std::fs::remove_dir_all(&path);
        } else if name.starts_with("snapshot-") && name.ends_with(".igdb.attachments") {
            let retained_attachment = retained.iter().any(|snapshot| {
                standalone_snapshot_attachment_directory(snapshot)
                    .ok()
                    .as_ref()
                    == Some(&path)
            });
            if !retained_attachment {
                let _ = std::fs::remove_dir_all(&path);
            }
        }
    }
}

fn prune_request_results(state: &mut DatabaseState) -> Result<()> {
    let minimum_index = state
        .applied
        .index
        .saturating_sub(REQUEST_RESULT_INDEX_WINDOW);
    while let Some((&index, &request_id)) = state.request_result_order.first_key_value() {
        if index >= minimum_index
            && state.request_results.len() <= MAX_REQUEST_RESULTS
            && state.request_result_bytes <= MAX_REQUEST_RESULT_BYTES
        {
            break;
        }
        state.request_result_order.remove(&index);
        let removed = state.request_results.remove(&request_id).ok_or_else(|| {
            Error::new(
                ErrorCode::CorruptStorage,
                "idempotency result order references a missing result",
            )
        })?;
        if removed.index != index {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "idempotency result order index differs from its record",
            ));
        }
        state.request_result_bytes = state
            .request_result_bytes
            .checked_sub(
                u64::try_from(removed.response.len())
                    .map_err(|_| Error::internal("idempotency response size overflow"))?,
            )
            .ok_or_else(|| Error::internal("idempotency result byte accounting underflow"))?;
    }
    validate_request_results(state)
}

fn validate_request_results(state: &DatabaseState) -> Result<()> {
    if state.request_results.len() != state.request_result_order.len()
        || state.request_results.len() > MAX_REQUEST_RESULTS
        || state.request_result_bytes > MAX_REQUEST_RESULT_BYTES
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "idempotency result retention bounds or indexes are invalid",
        ));
    }
    let mut bytes = 0_u64;
    for (request_id, record) in &*state.request_results {
        if record.response.len() > MAX_SINGLE_REQUEST_RESULT_BYTES
            || record.intent_digest == [0; 32]
            || state.request_result_order.get(&record.index) != Some(request_id)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "idempotency result record is invalid",
            ));
        }
        bytes = bytes
            .checked_add(
                u64::try_from(record.response.len())
                    .map_err(|_| Error::internal("idempotency response size overflow"))?,
            )
            .ok_or_else(|| Error::internal("idempotency result byte accounting overflow"))?;
    }
    if bytes != state.request_result_bytes {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "idempotency result byte accounting is inconsistent",
        ));
    }
    Ok(())
}

fn expire_optimizer_statistics_if_stale(project: &mut ProjectState) {
    let should_expire = project
        .optimizer_statistics
        .get()
        .is_some_and(|statistics| {
            let lag = project
                .graph
                .revision()
                .saturating_sub(statistics.graph_revision);
            // The first transition out of an empty generation must be visible immediately. Later
            // generations may remain boundedly stale because statistics influence cost only.
            (statistics.graph_revision == 0 && lag != 0)
                || lag >= OPTIMIZER_STATISTICS_MAX_REVISION_LAG
                || statistics.schema_generation != project.graph.catalog().optimizer_generation()
                || (statistics.index_generation != [0; 32]
                    && statistics.index_generation != project.indexes.optimizer_generation())
        });
    if should_expire {
        project.optimizer_statistics = OnceLock::new();
    }
}

fn project_has_data(state: &DatabaseState, project: ProjectId) -> Result<bool> {
    let project_state = state
        .projects
        .get(&project)
        .ok_or_else(|| Error::new(ErrorCode::ProjectNotFound, "project does not exist"))?;
    Ok(project_state.graph.node_count() != 0
        || project_state.graph.edge_count() != 0
        || !project_state.temporal.is_empty()
        || !project_state.indexes.is_empty()
        || !state.broker.project_is_empty(project))
}

fn apply_administrative(
    project: &mut ProjectState,
    mutation: AdministrativeMutation,
    resolved_commit_time_nanos: i64,
) -> Result<()> {
    match mutation {
        AdministrativeMutation::InitializeSemantic {
            profile,
            graph_revision,
        } => {
            if project.graph.revision() != graph_revision {
                return Err(Error::retryable(
                    ErrorCode::TransactionConflict,
                    "graph changed during semantic initialization; retry the query",
                    None,
                ));
            }
            project.indexes.initialize_semantic(profile)?;
            for name in [
                crate::graph::SEMANTIC_NODE_INDEX,
                crate::graph::SEMANTIC_RELATIONSHIP_INDEX,
            ] {
                project.indexes.rebuild_deferred(&project.graph, name)?;
            }
        }
        AdministrativeMutation::CreateIndex {
            name,
            kind,
            label,
            properties,
        } => {
            let label =
                project.graph.catalog().label(&label).ok_or_else(|| {
                    Error::new(ErrorCode::QueryType, "index label is not declared")
                })?;
            let properties = properties
                .iter()
                .map(|property| {
                    project.graph.catalog().property(property).ok_or_else(|| {
                        Error::new(ErrorCode::QueryType, "index property is not declared")
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let kind = match kind.to_ascii_uppercase().as_str() {
                "EQUALITY" => crate::graph::GraphIndexKind::Equality,
                "RANGE" => crate::graph::GraphIndexKind::Range,
                "TEXT" => crate::graph::GraphIndexKind::Text,
                "VECTOR" => crate::graph::GraphIndexKind::Vector,
                _ => {
                    return Err(Error::new(ErrorCode::QueryType, "unknown index family"));
                }
            };
            project.indexes.create_deferred(
                &project.graph,
                crate::graph::GraphIndexDefinition {
                    name,
                    kind,
                    label,
                    properties,
                    unique: false,
                },
            )?;
        }
        AdministrativeMutation::RebuildIndex { name } => {
            project.indexes.rebuild_deferred(&project.graph, &name)?;
        }
        AdministrativeMutation::DropIndex { name, if_exists } => {
            // `IF EXISTS` is what makes a teardown script re-runnable. Parsing the clause and then
            // dropping it on the floor made the statement fail on exactly the run where the index
            // was already gone, which is the run it exists for.
            if let Err(error) = project.indexes.drop_index(&name)
                && !(if_exists && error.code == ErrorCode::IndexUnavailable)
            {
                return Err(error);
            }
        }
        AdministrativeMutation::CreateConstraint {
            name,
            label,
            property,
        } => {
            let label = project.graph.catalog().label(&label).ok_or_else(|| {
                Error::new(ErrorCode::QueryType, "constraint label is not declared")
            })?;
            let property = project.graph.catalog().property(&property).ok_or_else(|| {
                Error::new(ErrorCode::QueryType, "constraint property is not declared")
            })?;
            project.indexes.create(
                &project.graph,
                crate::graph::GraphIndexDefinition {
                    name,
                    kind: crate::graph::GraphIndexKind::Equality,
                    label,
                    properties: vec![property],
                    unique: true,
                },
            )?;
        }
        AdministrativeMutation::DropConstraint { name, if_exists } => {
            if let Err(error) = project.indexes.drop_constraint(&name)
                && !(if_exists && error.code == ErrorCode::IndexUnavailable)
            {
                return Err(error);
            }
        }
        AdministrativeMutation::DeclareTemporal {
            entity_kind,
            label_or_type,
            property,
            scalar_type,
            retention_nanos,
        } => {
            let property = project.graph.catalog().property(&property).ok_or_else(|| {
                Error::new(ErrorCode::QueryType, "temporal property is not declared")
            })?;
            let target = match entity_kind {
                crate::types::EntityKind::Node => {
                    project.graph.catalog().label(&label_or_type).map(|id| id.0)
                }
                crate::types::EntityKind::Relationship => project
                    .graph
                    .catalog()
                    .relationship_type(&label_or_type)
                    .map(|id| id.0),
            }
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::QueryType,
                    "temporal declaration target is not declared in the project schema",
                )
            })?;
            project.temporal.declare(
                crate::graph::TemporalDeclaration {
                    entity_kind,
                    target,
                    property,
                    value_type: temporal_type(&scalar_type)?,
                    retention_nanos,
                },
                resolved_commit_time_nanos,
            )?;
        }
        AdministrativeMutation::CreateRollup {
            name,
            label,
            property,
            hopping,
            width_nanos,
            every_nanos,
            align_nanos,
            timezone,
            aggregates,
        } => {
            let label =
                project.graph.catalog().label(&label).ok_or_else(|| {
                    Error::new(ErrorCode::QueryType, "rollup label is not declared")
                })?;
            let property = project.graph.catalog().property(&property).ok_or_else(|| {
                Error::new(ErrorCode::QueryType, "rollup property is not declared")
            })?;
            let mut aggregate_set = crate::graph::AggregateSet::empty();
            for aggregate in aggregates {
                match aggregate.to_ascii_uppercase().as_str() {
                    "AVG" => aggregate_set.insert(crate::graph::AggregateSet::AVG),
                    "MIN" => aggregate_set.insert(crate::graph::AggregateSet::MIN),
                    "MAX" => aggregate_set.insert(crate::graph::AggregateSet::MAX),
                    "COUNT" => aggregate_set.insert(crate::graph::AggregateSet::COUNT),
                    "SUM" => aggregate_set.insert(crate::graph::AggregateSet::SUM),
                    _ => {
                        return Err(Error::new(
                            ErrorCode::QueryType,
                            "rollup contains an unsupported aggregate",
                        ));
                    }
                }
            }
            let mut window = if hopping {
                crate::graph::WindowSpec::hopping(
                    width_nanos,
                    every_nanos.ok_or_else(|| {
                        Error::new(ErrorCode::TemporalRange, "HOPPING rollup requires EVERY")
                    })?,
                )
            } else {
                crate::graph::WindowSpec::tumbling(width_nanos)
            };
            window.align_nanos = align_nanos;
            window.timezone = timezone;
            project
                .temporal
                .create_rollup(crate::graph::TemporalRollupDefinition {
                    name,
                    entity_kind: crate::types::EntityKind::Node,
                    target: label.0,
                    property,
                    window,
                    aggregates: aggregate_set,
                })?;
        }
        AdministrativeMutation::CreateEmbedding {
            name,
            label,
            source_property,
            target_property,
            model,
            profile,
            rows,
        } => {
            project.indexes.create_embedding_deferred(
                &project.graph,
                crate::graph::EmbeddingIndexDefinition {
                    name,
                    label,
                    source_property,
                    target_property,
                    model,
                },
                profile,
                rows,
            )?;
        }
    }
    project.optimizer_statistics = OnceLock::new();
    Ok(())
}

fn administrative_mutation(
    statement: Option<Statement>,
    project: &ProjectState,
    graph_mutations: &mut Vec<GraphMutation>,
    text_embedding: Option<&dyn TextEmbedding>,
    revision: u64,
) -> Result<Option<AdministrativeMutation>> {
    match statement {
        None
        | Some(
            Statement::CheckReadOnly
            | Statement::ShowIndexes
            | Statement::ShowConstraints
            | Statement::ShowTopics
            | Statement::ShowQueues
            | Statement::ShowExchanges
            | Statement::ShowConsumerLag
            | Statement::CreateTopic { .. }
            | Statement::AlterTopicRetention { .. }
            | Statement::DropTopic { .. }
            | Statement::ClearTopic { .. }
            | Statement::CreateQueue { .. }
            | Statement::AlterQueueRetention { .. }
            | Statement::DropQueue { .. }
            | Statement::PurgeQueue { .. }
            | Statement::CreateExchange { .. }
            | Statement::DropExchange { .. }
            | Statement::BindQueue { .. }
            | Statement::UnbindQueue { .. },
        ) => Ok(None),
        Some(Statement::CreateIndex(definition)) => Ok(Some(AdministrativeMutation::CreateIndex {
            name: definition.name,
            kind: format!("{:?}", definition.kind),
            label: definition.label,
            properties: definition.properties,
        })),
        Some(Statement::RebuildIndex { name }) => {
            Ok(Some(AdministrativeMutation::RebuildIndex { name }))
        }
        Some(Statement::DropIndex { name, if_exists }) => {
            Ok(Some(AdministrativeMutation::DropIndex { name, if_exists }))
        }
        Some(Statement::CreateConstraint(definition)) => {
            Ok(Some(AdministrativeMutation::CreateConstraint {
                name: definition.name,
                label: definition.label,
                property: definition.property,
            }))
        }
        Some(Statement::DropConstraint { name, if_exists }) => {
            Ok(Some(AdministrativeMutation::DropConstraint {
                name,
                if_exists,
            }))
        }
        Some(Statement::DeclareTemporal(declaration)) => {
            Ok(Some(AdministrativeMutation::DeclareTemporal {
                entity_kind: if declaration.target == crate::cypher::TemporalTarget::Node {
                    crate::types::EntityKind::Node
                } else {
                    crate::types::EntityKind::Relationship
                },
                label_or_type: declaration.label_or_type,
                property: declaration.property,
                scalar_type: declaration.scalar_type,
                retention_nanos: duration_expression_nanos(&declaration.retention)?,
            }))
        }
        Some(Statement::CreateRollup(definition)) => {
            Ok(Some(AdministrativeMutation::CreateRollup {
                name: definition.name,
                label: definition.label,
                property: definition.property,
                hopping: definition.window == crate::cypher::WindowSyntax::Hopping,
                width_nanos: duration_expression_nanos(&definition.width)?,
                every_nanos: definition
                    .every
                    .as_ref()
                    .map(duration_expression_nanos)
                    .transpose()?,
                align_nanos: definition
                    .align
                    .as_ref()
                    .map(instant_expression_nanos)
                    .transpose()?
                    .unwrap_or(0),
                timezone: definition.timezone,
                aggregates: definition.aggregates,
            }))
        }
        Some(Statement::CreateEmbedding(definition)) => {
            let embedding = text_embedding.ok_or_else(|| {
                Error::new(
                    ErrorCode::EmbeddingUnavailable,
                    "CREATE EMBEDDING requires an active local embedding artifact",
                )
            })?;
            let local_profile = embedding.profile();
            local_profile.validate()?;
            let profile = project.indexes.profile().cloned().ok_or_else(|| {
                Error::new(
                    ErrorCode::EmbeddingProfileMismatch,
                    "project embedding profile has not passed all-node activation",
                )
            })?;
            if &profile != local_profile {
                return Err(Error::new(
                    ErrorCode::EmbeddingProfileImmutable,
                    "local embedding artifact differs from the project's active profile",
                ));
            }
            if definition.model.to_ascii_lowercase() != "default" {
                return Err(Error::new(
                    ErrorCode::EmbeddingProfileMismatch,
                    "only the active project embedding model is supported",
                ));
            }
            let requested_similarity = parse_similarity(&definition.similarity)?;
            if requested_similarity != profile.similarity {
                return Err(Error::new(
                    ErrorCode::EmbeddingProfileMismatch,
                    "CREATE EMBEDDING similarity differs from the active profile",
                ));
            }
            let label = project
                .graph
                .catalog()
                .label(&definition.label)
                .ok_or_else(|| {
                    Error::new(ErrorCode::QueryType, "embedding label is not declared")
                })?;
            let source_property = project
                .graph
                .catalog()
                .property(&definition.source_property)
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::QueryType,
                        "embedding source property is not declared",
                    )
                })?;
            let target_property = match project
                .graph
                .catalog()
                .property(&definition.target_property)
            {
                Some(property) => property,
                None => {
                    let mut catalog = project.graph.catalog().clone();
                    for mutation in graph_mutations.iter() {
                        match mutation {
                            GraphMutation::DeclareLabel { name, id } => {
                                catalog.declare_label(name.clone(), *id)?;
                            }
                            GraphMutation::DeclareProperty { name, id } => {
                                catalog.declare_property(name.clone(), *id)?;
                            }
                            GraphMutation::DeclareRelationshipType { name, id } => {
                                catalog.declare_relationship_type(name.clone(), *id)?;
                            }
                            _ => {}
                        }
                    }
                    let property = catalog.intern_property(&definition.target_property)?;
                    graph_mutations.push(GraphMutation::DeclareProperty {
                        name: definition.target_property.clone(),
                        id: property,
                    });
                    property
                }
            };
            let mut sources = Vec::new();
            for node in project
                .graph
                .nodes()
                .filter(|node| node.labels().contains(&label))
            {
                match node.property(source_property) {
                    Some(ScalarValue::String(text)) => {
                        sources.push((node.id().0, text.to_string()))
                    }
                    Some(ScalarValue::Null) | None => {}
                    Some(_) => {
                        return Err(Error::new(
                            ErrorCode::QueryType,
                            "embedding source property contains a non-string value",
                        ));
                    }
                }
            }
            // Cold population gathers every owner first and submits its derived windows in one
            // encoder batch. Incremental writes use the separate affected-row path below.
            let vectors = embed_complete_texts(
                embedding,
                &sources
                    .iter()
                    .map(|(_, text)| text.as_str())
                    .collect::<Vec<_>>(),
            )?;
            let rows = sources
                .into_iter()
                .zip(vectors)
                .map(|((entity_id, _), vector)| {
                    Ok((entity_id, profile.quantize(&vector)?, revision))
                })
                .collect::<Result<Vec<_>>>()?;
            Ok(Some(AdministrativeMutation::CreateEmbedding {
                name: definition.name,
                label,
                source_property,
                target_property,
                model: definition.model,
                profile,
                rows,
            }))
        }
        Some(_) => Err(Error::new(
            ErrorCode::QueryType,
            "project lifecycle statement reached project executor",
        )),
    }
}

fn parse_similarity(value: &str) -> Result<crate::graph::Similarity> {
    match value.to_ascii_uppercase().as_str() {
        "COSINE" => Ok(crate::graph::Similarity::Cosine),
        "DOT" => Ok(crate::graph::Similarity::Dot),
        "EUCLIDEAN" => Ok(crate::graph::Similarity::Euclidean),
        _ => Err(Error::new(
            ErrorCode::QueryType,
            "embedding similarity must be COSINE, DOT, or EUCLIDEAN",
        )),
    }
}

/// Encodes every byte of each canonical text through overlapping derived windows and folds those
/// windows into one canonical vector row. Window strings are transient and retain no identity.
/// One batched encoder call covers the complete input set, so cold work scales with all owners while
/// the caller-controlled delta set below scales only with changed owners.
fn embed_complete_texts(embedding: &dyn TextEmbedding, texts: &[&str]) -> Result<Vec<Vec<f32>>> {
    let dimension = usize::try_from(embedding.profile().dimension)
        .map_err(|_| Error::internal("embedding dimension does not fit this process"))?;
    let mut flattened = Vec::new();
    let mut ranges = Vec::with_capacity(texts.len());
    for text in texts {
        let start = flattened.len();
        let mut windows = embedding.index_windows(text)?;
        if windows.is_empty() {
            windows.push((*text).to_owned());
        }
        flattened.extend(windows);
        ranges.push(start..flattened.len());
    }
    let encoded = embedding.embed_batch(&flattened)?;
    if encoded.len() != flattened.len() || encoded.iter().any(|vector| vector.len() != dimension) {
        return Err(Error::new(
            ErrorCode::EmbeddingProfileMismatch,
            "embedding output shape differs from the active profile",
        ));
    }

    ranges
        .into_iter()
        .map(|range| {
            let count = range.len();
            let mut aggregate = vec![0.0_f64; dimension];
            for vector in &encoded[range] {
                for (target, coordinate) in aggregate.iter_mut().zip(vector) {
                    *target += f64::from(*coordinate);
                }
            }
            let divisor = count as f64;
            let mut vector = aggregate
                .into_iter()
                .map(|coordinate| (coordinate / divisor) as f32)
                .collect::<Vec<_>>();
            if embedding.profile().normalized {
                let norm = vector
                    .iter()
                    .map(|coordinate| f64::from(*coordinate).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if norm > 0.0 {
                    for coordinate in &mut vector {
                        *coordinate = (f64::from(*coordinate) / norm) as f32;
                    }
                }
            }
            Ok(vector)
        })
        .collect()
}

fn resolve_semantic_texts(
    texts: crate::graph::SemanticTextBatch,
    embedding: &dyn TextEmbedding,
    revision: u64,
) -> Result<Vec<ResolvedVectorMutation>> {
    let mut mutations = Vec::new();
    for (property, owners) in [
        (crate::graph::SEMANTIC_NODE_PROPERTY, texts.nodes),
        (
            crate::graph::SEMANTIC_RELATIONSHIP_PROPERTY,
            texts.relationships,
        ),
    ] {
        let mut pending = Vec::new();
        for owner in owners {
            match owner.text {
                Some(text) => pending.push((owner.entity_id, text)),
                None => mutations.push(ResolvedVectorMutation::Remove {
                    property,
                    entity_id: owner.entity_id,
                    revision,
                }),
            }
        }
        let encoded = embed_complete_texts(
            embedding,
            &pending
                .iter()
                .map(|(_, text)| text.as_str())
                .collect::<Vec<_>>(),
        )?;
        for ((entity_id, _), vector) in pending.into_iter().zip(encoded) {
            mutations.push(ResolvedVectorMutation::Upsert {
                property,
                entity_id,
                coordinates: embedding.profile().quantize(&vector)?,
                revision,
            });
        }
    }
    Ok(mutations)
}

fn resolve_embedding_mutations(
    project: &ProjectState,
    prior_graph_mutations: &[GraphMutation],
    graph_mutations: &[GraphMutation],
    text_embedding: Option<&dyn TextEmbedding>,
    revision: u64,
) -> Result<Vec<ResolvedVectorMutation>> {
    if graph_mutations.is_empty() {
        return Ok(Vec::new());
    }
    let mut vectors = if project.indexes.contains(crate::graph::SEMANTIC_NODE_INDEX) {
        let embedding = text_embedding.ok_or_else(|| {
            Error::new(
                ErrorCode::EmbeddingUnavailable,
                "automatic semantic updates require the active local encoder",
            )
        })?;
        resolve_semantic_texts(
            crate::graph::semantic_text_delta(
                &project.graph,
                prior_graph_mutations,
                graph_mutations,
            )?,
            embedding,
            revision,
        )?
    } else {
        Vec::new()
    };
    if project.indexes.embedding_definitions().next().is_none() {
        return Ok(vectors);
    }
    let definitions = project
        .indexes
        .embedding_definitions()
        .cloned()
        .collect::<Vec<_>>();
    let mut affected = BTreeMap::<u64, bool>::new();
    for mutation in graph_mutations {
        match mutation {
            GraphMutation::InsertNode(node) => {
                affected.insert(node.id.0, true);
            }
            GraphMutation::SetNodeProperty { node, property, .. }
                if definitions
                    .iter()
                    .any(|definition| definition.source_property == *property) =>
            {
                affected.insert(node.0, true);
            }
            GraphMutation::AddNodeLabels { node, .. }
            | GraphMutation::RemoveNodeLabels { node, .. } => {
                affected.insert(node.0, true);
            }
            GraphMutation::DeleteNode { node, .. } => {
                affected.insert(node.0, true);
            }
            _ => {}
        }
    }
    if affected.is_empty() {
        return Ok(vectors);
    }

    let states = crate::cypher::sparse_node_states_after_mutations(
        &project.graph,
        prior_graph_mutations,
        graph_mutations,
        affected.keys().copied().map(crate::NodeId),
    )?;
    let profile = project.indexes.profile().ok_or_else(|| {
        Error::new(
            ErrorCode::EmbeddingProfileMismatch,
            "embedding definitions exist without a project profile",
        )
    })?;
    profile.validate()?;
    let embedding = text_embedding.ok_or_else(|| {
        Error::new(
            ErrorCode::EmbeddingUnavailable,
            "updating an embedded source requires the active local embedding artifact",
        )
    })?;
    if embedding.profile() != profile {
        return Err(Error::new(
            ErrorCode::EmbeddingProfileMismatch,
            "local embedding artifact differs from the project's active profile",
        ));
    }

    for definition in definitions {
        let mut pending = Vec::new();
        for entity_id in affected.keys().copied() {
            let Some(node) = states.get(&crate::NodeId(entity_id)) else {
                vectors.push(ResolvedVectorMutation::Remove {
                    property: definition.target_property,
                    entity_id,
                    revision,
                });
                continue;
            };
            if !node.labels.contains(&definition.label) {
                vectors.push(ResolvedVectorMutation::Remove {
                    property: definition.target_property,
                    entity_id,
                    revision,
                });
                continue;
            }
            match node.properties.get(&definition.source_property) {
                Some(ScalarValue::String(text)) => {
                    pending.push((entity_id, text.to_string()));
                }
                Some(ScalarValue::Null) | None => {
                    vectors.push(ResolvedVectorMutation::Remove {
                        property: definition.target_property,
                        entity_id,
                        revision,
                    });
                }
                Some(_) => {
                    return Err(Error::new(
                        ErrorCode::QueryType,
                        "embedding source property must remain a string or NULL",
                    ));
                }
            }
        }
        let embedded = embed_complete_texts(
            embedding,
            &pending
                .iter()
                .map(|(_, text)| text.as_str())
                .collect::<Vec<_>>(),
        )?;
        for ((entity_id, _), vector) in pending.into_iter().zip(embedded) {
            vectors.push(ResolvedVectorMutation::Upsert {
                property: definition.target_property,
                entity_id,
                coordinates: profile.quantize(&vector)?,
                revision,
            });
        }
    }
    Ok(vectors)
}

fn emit_result(
    request_id: Uuid,
    mut result: QueryResult,
    catalog: &crate::graph::NameCatalog,
    indexes: &IndexCatalog,
    emit: &mut dyn FnMut(QueryStreamEvent) -> Result<()>,
) -> Result<()> {
    emit_catalog(None, result.bookmark, catalog, indexes, emit)?;
    emit(QueryStreamEvent::Schema {
        request_id,
        columns: result
            .schema
            .iter()
            .map(|(name, value_type)| QueryColumn {
                name: name.clone(),
                value_type: format!("{value_type:?}").to_uppercase(),
                nullable: true,
            })
            .collect(),
    })?;
    let mut rows = 0_u64;
    let mut sequence = 0_u64;
    for batch in std::mem::take(&mut result.batches) {
        emit_execution_stream_item(
            request_id,
            ExecutionStreamItem::Batch(batch),
            &mut sequence,
            &mut rows,
            emit,
        )?;
    }
    emit_query_summary(request_id, result, rows, emit)
}

fn emit_catalog(
    project_id: Option<ProjectId>,
    bookmark: Bookmark,
    catalog: &crate::graph::NameCatalog,
    indexes: &IndexCatalog,
    emit: &mut dyn FnMut(QueryStreamEvent) -> Result<()>,
) -> Result<()> {
    emit(QueryStreamEvent::Catalog {
        catalog: CatalogEvent {
            project_id,
            schema_revision: bookmark.index,
            labels: catalog.labels().map(|(_, name)| name.to_owned()).collect(),
            relationship_types: catalog
                .relationship_types()
                .map(|(_, name)| name.to_owned())
                .collect(),
            properties: catalog
                .properties()
                .map(|(_, name)| name.to_owned())
                .collect(),
            functions: builtin_functions(),
            indexes: indexes.statuses().map(|index| index.name).collect(),
        },
    })
}

fn emit_execution_stream_item(
    request_id: Uuid,
    item: ExecutionStreamItem,
    sequence: &mut u64,
    rows: &mut u64,
    emit: &mut dyn FnMut(QueryStreamEvent) -> Result<()>,
) -> Result<()> {
    match item {
        ExecutionStreamItem::Schema(schema) => emit(QueryStreamEvent::Schema {
            request_id,
            columns: schema
                .into_iter()
                .map(|(name, value_type)| QueryColumn {
                    name,
                    value_type: format!("{value_type:?}").to_uppercase(),
                    nullable: true,
                })
                .collect(),
        }),
        ExecutionStreamItem::Batch(batch) => {
            *rows = rows.saturating_add(batch.row_count as u64);
            let columns = batch
                .columns
                .into_iter()
                .map(|column| {
                    Ok(BatchColumn {
                        name: column.name,
                        value_type: format!("{:?}", column.value_type).to_uppercase(),
                        values: column
                            .values
                            .into_iter()
                            .map(result_to_typed)
                            .collect::<Result<Vec<_>>>()?,
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            emit(QueryStreamEvent::Batch {
                request_id,
                sequence: *sequence,
                row_count: batch.row_count as u64,
                columns,
            })?;
            *sequence = sequence.saturating_add(1);
            Ok(())
        }
    }
}

fn emit_query_summary(
    request_id: Uuid,
    result: QueryResult,
    rows: u64,
    emit: &mut dyn FnMut(QueryStreamEvent) -> Result<()>,
) -> Result<()> {
    emit(QueryStreamEvent::Summary {
        request_id,
        bookmark: result.bookmark,
        statistics: QueryStatistics {
            rows,
            updates: statement_update_count(result.statistics),
            ..QueryStatistics::default()
        },
        truncated: result.truncated,
        truncation_reason: result.truncated.then(|| "result_limit".to_owned()),
    })
}

fn emit_admin_summary(
    request_id: Uuid,
    bookmark: Bookmark,
    emit: &mut dyn FnMut(QueryStreamEvent) -> Result<()>,
) -> Result<()> {
    emit(QueryStreamEvent::Schema {
        request_id,
        columns: Vec::new(),
    })?;
    emit(QueryStreamEvent::Summary {
        request_id,
        bookmark,
        statistics: QueryStatistics {
            updates: 1,
            ..QueryStatistics::default()
        },
        truncated: false,
        truncation_reason: None,
    })
}

fn string_batch(name: &str, values: impl IntoIterator<Item = String>) -> BatchColumn {
    BatchColumn {
        name: name.to_owned(),
        value_type: "STRING".to_owned(),
        values: values.into_iter().map(TypedValue::String).collect(),
    }
}

fn integer_batch(name: &str, values: impl IntoIterator<Item = i64>) -> BatchColumn {
    BatchColumn {
        name: name.to_owned(),
        value_type: "INTEGER".to_owned(),
        values: values
            .into_iter()
            .map(|value| TypedValue::Integer(value.to_string()))
            .collect(),
    }
}

fn saturating_i64(value: u64) -> i64 {
    i64::try_from(value).unwrap_or(i64::MAX)
}

fn emit_read_summary(
    request_id: Uuid,
    bookmark: Bookmark,
    rows: usize,
    emit: &mut dyn FnMut(QueryStreamEvent) -> Result<()>,
) -> Result<()> {
    emit(QueryStreamEvent::Summary {
        request_id,
        bookmark,
        statistics: QueryStatistics {
            rows: rows as u64,
            ..QueryStatistics::default()
        },
        truncated: false,
        truncation_reason: None,
    })
}

fn statement_update_count(statistics: crate::cypher::StatementStats) -> u64 {
    statistics
        .nodes_created
        .saturating_add(statistics.nodes_deleted)
        .saturating_add(statistics.relationships_created)
        .saturating_add(statistics.relationships_deleted)
        .saturating_add(statistics.properties_set)
}

fn result_to_typed(value: ResultValue) -> Result<TypedValue> {
    Ok(match value {
        ResultValue::Scalar(value) => scalar_to_typed(value)?,
        ResultValue::Node(node) => TypedValue::Node(ResultNode {
            id: node.id.to_string(),
            labels: node.labels,
            properties: node
                .properties
                .into_iter()
                .map(|(name, value)| Ok((name, scalar_to_typed(value)?)))
                .collect::<Result<_>>()?,
        }),
        ResultValue::Relationship(edge) => TypedValue::Relationship(RelationshipValue {
            id: edge.id.to_string(),
            source: edge.source.to_string(),
            target: edge.target.to_string(),
            relationship_type: edge.relationship_type,
            properties: edge
                .properties
                .into_iter()
                .map(|(name, value)| Ok((name, scalar_to_typed(value)?)))
                .collect::<Result<_>>()?,
        }),
        ResultValue::Path {
            nodes,
            relationships,
        } => TypedValue::Path(PathValue {
            nodes: nodes
                .into_iter()
                .map(|node| {
                    Ok(ResultNode {
                        id: node.id.to_string(),
                        labels: node.labels,
                        properties: node
                            .properties
                            .into_iter()
                            .map(|(name, value)| Ok((name, scalar_to_typed(value)?)))
                            .collect::<Result<_>>()?,
                    })
                })
                .collect::<Result<_>>()?,
            relationships: relationships
                .into_iter()
                .map(|edge| {
                    Ok(RelationshipValue {
                        id: edge.id.to_string(),
                        source: edge.source.to_string(),
                        target: edge.target.to_string(),
                        relationship_type: edge.relationship_type,
                        properties: edge
                            .properties
                            .into_iter()
                            .map(|(name, value)| Ok((name, scalar_to_typed(value)?)))
                            .collect::<Result<_>>()?,
                    })
                })
                .collect::<Result<_>>()?,
        }),
        ResultValue::Vector(values) => TypedValue::Vector(values),
        ResultValue::List(values) => TypedValue::List(
            values
                .into_iter()
                .map(result_to_typed)
                .collect::<Result<Vec<_>>>()?,
        ),
        ResultValue::Map(values) => TypedValue::Map(
            values
                .into_iter()
                .map(|(name, value)| Ok((name, result_to_typed(value)?)))
                .collect::<Result<_>>()?,
        ),
    })
}

fn scalar_to_typed(value: ScalarValue) -> Result<TypedValue> {
    Ok(match value {
        ScalarValue::Null => TypedValue::Null,
        ScalarValue::Boolean(value) => TypedValue::Boolean(value),
        ScalarValue::Integer(value) => TypedValue::Integer(value.to_string()),
        ScalarValue::Float(value) => TypedValue::Float(value.into_inner()),
        ScalarValue::String(value) => TypedValue::String(value.to_string()),
        ScalarValue::Bytes(value) => TypedValue::Bytes(value.to_vec()),
        ScalarValue::Date(value) => TypedValue::Date(value),
        ScalarValue::LocalTime(value) => TypedValue::Time {
            nanos: value,
            offset_seconds: None,
        },
        ScalarValue::ZonedTime {
            nanos,
            offset_seconds,
        } => TypedValue::Time {
            nanos,
            offset_seconds: Some(offset_seconds),
        },
        ScalarValue::LocalDateTime { seconds, nanos } => TypedValue::DateTime {
            seconds,
            nanos,
            timezone: None,
        },
        ScalarValue::ZonedDateTime {
            seconds,
            nanos,
            timezone,
        } => TypedValue::DateTime {
            seconds,
            nanos,
            timezone: Some(timezone.to_string()),
        },
        ScalarValue::Duration {
            months,
            days,
            seconds,
            nanos,
        } => TypedValue::Duration {
            months,
            days,
            seconds,
            nanos,
        },
        value @ (ScalarValue::List(_) | ScalarValue::Map(_)) => {
            return result_to_typed(ResultValue::from_property(value)?);
        }
    })
}

pub(super) fn json_to_result(value: &serde_json::Value) -> Result<ResultValue> {
    Ok(match value {
        serde_json::Value::Null => ResultValue::Scalar(ScalarValue::Null),
        serde_json::Value::Bool(value) => ResultValue::Scalar(ScalarValue::Boolean(*value)),
        serde_json::Value::Number(value) => {
            if let Some(integer) = value.as_i64() {
                ResultValue::Scalar(ScalarValue::Integer(integer))
            } else {
                ResultValue::Scalar(ScalarValue::Float(ordered_float::OrderedFloat(
                    value
                        .as_f64()
                        .ok_or_else(|| Error::invalid_data("query number is invalid"))?,
                )))
            }
        }
        serde_json::Value::String(value) => {
            ResultValue::Scalar(ScalarValue::String(value.clone().into()))
        }
        serde_json::Value::Array(values) => ResultValue::List(
            values
                .iter()
                .map(json_to_result)
                .collect::<Result<Vec<_>>>()?,
        ),
        serde_json::Value::Object(values) if values.contains_key("$irongraph_type") => {
            tagged_json_to_result(values)?
        }
        serde_json::Value::Object(values) => ResultValue::Map(
            values
                .iter()
                .map(|(name, value)| Ok((name.clone(), json_to_result(value)?)))
                .collect::<Result<_>>()?,
        ),
    })
}

fn tagged_json_to_result(
    values: &serde_json::Map<String, serde_json::Value>,
) -> Result<ResultValue> {
    let kind = values
        .get("$irongraph_type")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| Error::invalid_data("typed parameter requires a string $irongraph_type"))?;
    let scalar = match kind {
        "bytes" => {
            exact_tag_fields(values, &["$irongraph_type", "value"])?;
            let bytes = values
                .get("value")
                .and_then(serde_json::Value::as_array)
                .ok_or_else(|| Error::invalid_data("bytes parameter requires an array"))?
                .iter()
                .map(|value| {
                    value
                        .as_u64()
                        .and_then(|value| u8::try_from(value).ok())
                        .ok_or_else(|| Error::invalid_data("byte parameter is outside 0..255"))
                })
                .collect::<Result<Vec<_>>>()?;
            ScalarValue::Bytes(bytes.into())
        }
        "date" => {
            exact_tag_fields(values, &["$irongraph_type", "days"])?;
            ScalarValue::Date(tag_i64(values, "days")?)
        }
        "local_time" => {
            exact_tag_fields(values, &["$irongraph_type", "nanos"])?;
            ScalarValue::LocalTime(tag_i64(values, "nanos")?)
        }
        "zoned_time" => {
            exact_tag_fields(values, &["$irongraph_type", "nanos", "offset_seconds"])?;
            ScalarValue::ZonedTime {
                nanos: tag_i64(values, "nanos")?,
                offset_seconds: tag_i32(values, "offset_seconds")?,
            }
        }
        "local_datetime" => {
            exact_tag_fields(values, &["$irongraph_type", "seconds", "nanos"])?;
            ScalarValue::LocalDateTime {
                seconds: tag_i64(values, "seconds")?,
                nanos: tag_u32(values, "nanos")?,
            }
        }
        "zoned_datetime" => {
            exact_tag_fields(values, &["$irongraph_type", "seconds", "nanos", "timezone"])?;
            let timezone = values
                .get("timezone")
                .and_then(serde_json::Value::as_str)
                .filter(|timezone| !timezone.is_empty() && timezone.len() <= 255)
                .ok_or_else(|| Error::invalid_data("datetime timezone is empty or oversized"))?;
            ScalarValue::ZonedDateTime {
                seconds: tag_i64(values, "seconds")?,
                nanos: tag_u32(values, "nanos")?,
                timezone: timezone.to_owned().into(),
            }
        }
        "duration" => {
            exact_tag_fields(
                values,
                &["$irongraph_type", "months", "days", "seconds", "nanos"],
            )?;
            ScalarValue::Duration {
                months: tag_i64(values, "months")?,
                days: tag_i64(values, "days")?,
                seconds: tag_i64(values, "seconds")?,
                nanos: tag_i32(values, "nanos")?,
            }
        }
        _ => {
            return Err(Error::invalid_data(
                "typed parameter has an unknown $irongraph_type",
            ));
        }
    };
    Ok(ResultValue::Scalar(scalar))
}

fn exact_tag_fields(
    values: &serde_json::Map<String, serde_json::Value>,
    fields: &[&str],
) -> Result<()> {
    if values.len() == fields.len() && fields.iter().all(|field| values.contains_key(*field)) {
        Ok(())
    } else {
        Err(Error::invalid_data(
            "typed parameter has missing or unknown fields",
        ))
    }
}

fn tag_i64(values: &serde_json::Map<String, serde_json::Value>, field: &str) -> Result<i64> {
    values
        .get(field)
        .and_then(serde_json::Value::as_i64)
        .ok_or_else(|| Error::invalid_data(format!("typed parameter {field} must be an integer")))
}

fn tag_i32(values: &serde_json::Map<String, serde_json::Value>, field: &str) -> Result<i32> {
    i32::try_from(tag_i64(values, field)?)
        .map_err(|_| Error::invalid_data(format!("typed parameter {field} exceeds 32-bit range")))
}

fn tag_u32(values: &serde_json::Map<String, serde_json::Value>, field: &str) -> Result<u32> {
    values
        .get(field)
        .and_then(serde_json::Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .ok_or_else(|| {
            Error::invalid_data(format!(
                "typed parameter {field} is negative or exceeds 32-bit range"
            ))
        })
}

fn validate_dependencies(
    dependencies: &TransactionDependencies,
    project: &ProjectState,
    planning_index: u64,
) -> Result<()> {
    for (entity, revision) in &dependencies.entities {
        let current = match entity {
            crate::cypher::EntityDependency::Node(id) => {
                project.graph.node(*id).map(|node| node.revision())
            }
            crate::cypher::EntityDependency::Relationship(id) => {
                project.graph.edge(*id).map(|edge| edge.revision())
            }
        };
        if current.is_none() && dependencies.write_targets.contains(entity) {
            continue;
        }
        if current != Some(*revision) {
            return Err(Error::retryable(
                ErrorCode::TransactionConflict,
                "transaction entity changed",
                None,
            ));
        }
    }
    for (stamp, revision) in &dependencies.predicates {
        let changed = project
            .predicate_versions
            .get(stamp)
            .map_or(project.authority_revision > planning_index, |current| {
                current != revision
            });
        if changed {
            return Err(Error::retryable(
                ErrorCode::TransactionConflict,
                "transaction predicate changed",
                None,
            ));
        }
    }
    Ok(())
}

fn merge_dependencies(target: &mut TransactionDependencies, source: TransactionDependencies) {
    for (entity, revision) in source.entities {
        target.entities.entry(entity).or_insert(revision);
    }
    for (stamp, revision) in source.predicates {
        target.predicates.entry(stamp).or_insert(revision);
    }
    target.write_targets.extend(source.write_targets);
}

fn retag_graph_mutation(mutation: GraphMutation, revision: u64) -> GraphMutation {
    match mutation {
        GraphMutation::InsertNode(mut node) => {
            node.revision = revision;
            GraphMutation::InsertNode(node)
        }
        GraphMutation::InsertEdge(mut edge) => {
            edge.revision = revision;
            GraphMutation::InsertEdge(edge)
        }
        GraphMutation::SetNodeProperty {
            node,
            property,
            value,
            ..
        } => GraphMutation::SetNodeProperty {
            node,
            property,
            value,
            revision,
        },
        GraphMutation::AddNodeLabels { node, labels, .. } => GraphMutation::AddNodeLabels {
            node,
            labels,
            revision,
        },
        GraphMutation::RemoveNodeLabels { node, labels, .. } => GraphMutation::RemoveNodeLabels {
            node,
            labels,
            revision,
        },
        GraphMutation::SetEdgeProperty {
            edge,
            property,
            value,
            ..
        } => GraphMutation::SetEdgeProperty {
            edge,
            property,
            value,
            revision,
        },
        GraphMutation::DeleteNode { node, detach, .. } => GraphMutation::DeleteNode {
            node,
            detach,
            revision,
        },
        GraphMutation::DeleteEdge { edge, .. } => GraphMutation::DeleteEdge { edge, revision },
        declaration => declaration,
    }
}

fn retag_vector_mutation(
    mutation: ResolvedVectorMutation,
    revision: u64,
) -> ResolvedVectorMutation {
    match mutation {
        ResolvedVectorMutation::Upsert {
            property,
            entity_id,
            coordinates,
            ..
        } => ResolvedVectorMutation::Upsert {
            property,
            entity_id,
            coordinates,
            revision,
        },
        ResolvedVectorMutation::Remove {
            property,
            entity_id,
            ..
        } => ResolvedVectorMutation::Remove {
            property,
            entity_id,
            revision,
        },
    }
}

fn retag_administrative_mutation(
    mutation: AdministrativeMutation,
    revision: u64,
) -> AdministrativeMutation {
    match mutation {
        AdministrativeMutation::CreateEmbedding {
            name,
            label,
            source_property,
            target_property,
            model,
            profile,
            rows,
        } => AdministrativeMutation::CreateEmbedding {
            name,
            label,
            source_property,
            target_property,
            model,
            profile,
            rows: rows
                .into_iter()
                .map(|(entity_id, coordinates, _)| (entity_id, coordinates, revision))
                .collect(),
        },
        mutation => mutation,
    }
}

fn update_next_ids(project: &mut ProjectState, mutation: &GraphMutation) {
    match mutation {
        GraphMutation::InsertNode(node) => {
            project.next_node_id = project.next_node_id.max(node.id.0.saturating_add(1))
        }
        GraphMutation::InsertEdge(edge) => {
            project.next_edge_id = project.next_edge_id.max(edge.id.0.saturating_add(1))
        }
        _ => {}
    }
}

fn full_capabilities() -> BindCapabilities {
    BindCapabilities {
        write: true,
        schema: true,
        knowledge_write: true,
        workspace_write: true,
        // Production falls back to host execution; the conformance gate does not.
        //
        // This was briefly `true`, to make the shipped configuration identical to the one the gate
        // measures. That was wrong, and it broke ordinary queries: the pinned TCK corpus is not the
        // set of queries people write, so plan shapes outside it — `count(p)` over a
        // variable-length path variable, for one — turned from a slower host evaluation into a
        // hard GPU_ADMISSION_FAILURE. A database that refuses a valid query because one execution
        // strategy does not cover it is not correct, it is just strict.
        //
        // The defect the flag was meant to fix was that the fallback was *silent*, so native
        // coverage could regress without anything saying so. That is fixed where it belongs: the
        // fallback now logs at warn with the plan shape (see `execute_plan_inner`), and the gate
        // still sets this flag, so a scenario that stops compiling natively fails there loudly.
        require_native_execution: false,
    }
}

fn statement_writes(statement: &Statement) -> bool {
    match statement {
        Statement::Query(body) => body
            .clauses
            .iter()
            .chain(body.unions.iter().flat_map(|branch| branch.body.iter()))
            .any(|clause| {
                matches!(
                    clause,
                    Clause::Create(_)
                        | Clause::Merge { .. }
                        | Clause::Set(_)
                        | Clause::Remove(_)
                        | Clause::Delete { .. }
                )
            }),
        Statement::CheckReadOnly
        | Statement::ShowProjects
        | Statement::ShowIndexes
        | Statement::ShowConstraints
        | Statement::ShowTopics
        | Statement::ShowQueues
        | Statement::ShowExchanges
        | Statement::ShowConsumerLag => false,
        Statement::ImportDataset { .. }
        | Statement::CreateProject { .. }
        | Statement::AlterProjectRename { .. }
        | Statement::DropProject { .. }
        | Statement::CreateIndex(_)
        | Statement::CreateConstraint(_)
        | Statement::RebuildIndex { .. }
        | Statement::DropIndex { .. }
        | Statement::DropConstraint { .. }
        | Statement::DeclareTemporal(_)
        | Statement::CreateRollup(_)
        | Statement::CreateEmbedding(_)
        | Statement::CreateTopic { .. }
        | Statement::AlterTopicRetention { .. }
        | Statement::DropTopic { .. }
        | Statement::ClearTopic { .. }
        | Statement::CreateQueue { .. }
        | Statement::AlterQueueRetention { .. }
        | Statement::DropQueue { .. }
        | Statement::PurgeQueue { .. }
        | Statement::CreateExchange { .. }
        | Statement::DropExchange { .. }
        | Statement::BindQueue { .. }
        | Statement::UnbindQueue { .. } => true,
    }
}

fn broker_project(command: &BrokerCommand) -> ProjectId {
    match command {
        BrokerCommand::CreateTopic { project, .. }
        | BrokerCommand::SetTopicRetention { project, .. }
        | BrokerCommand::DeleteTopic { project, .. }
        | BrokerCommand::ClearTopic { project, .. }
        | BrokerCommand::CreateExchange { project, .. }
        | BrokerCommand::CreateQueue { project, .. }
        | BrokerCommand::SetQueueRetention { project, .. }
        | BrokerCommand::BindQueue { project, .. }
        | BrokerCommand::UnbindQueue { project, .. }
        | BrokerCommand::PurgeQueue { project, .. }
        | BrokerCommand::DeleteQueue { project, .. }
        | BrokerCommand::DeleteExchange { project, .. }
        | BrokerCommand::RegisterConsumer { project, .. }
        | BrokerCommand::UnregisterConsumer { project, .. }
        | BrokerCommand::ReleaseConnection { project, .. }
        | BrokerCommand::PublishKafkaBatch { project, .. }
        | BrokerCommand::PublishAmqp { project, .. }
        | BrokerCommand::PublishAmqpBatch { project, .. }
        | BrokerCommand::PublishAmqpUniformBatch { project, .. }
        | BrokerCommand::CommitOffset { project, .. }
        | BrokerCommand::JoinGroup { project, .. }
        | BrokerCommand::SyncGroup { project, .. }
        | BrokerCommand::LeaveGroup { project, .. }
        | BrokerCommand::HeartbeatGroup { project, .. }
        | BrokerCommand::Ack { project, .. }
        | BrokerCommand::Nack { project, .. }
        | BrokerCommand::DeliverQueue { project, .. }
        | BrokerCommand::RenewDeliveryLease { project, .. }
        | BrokerCommand::ReleaseDeliveryLease { project, .. }
        | BrokerCommand::Retain { project, .. } => *project,
    }
}

fn validate_database_entry(entry: &MutationEntry, mutation: &DatabaseMutation) -> Result<()> {
    validate_database_envelope(entry.kind(), entry.project_id(), mutation)
}

fn validate_database_envelope(
    kind: MutationKind,
    project_id: Option<ProjectId>,
    mutation: &DatabaseMutation,
) -> Result<()> {
    let expected_kind = match mutation {
        DatabaseMutation::CreateProject { .. }
        | DatabaseMutation::RenameProject { .. }
        | DatabaseMutation::DropProject { .. } => MutationKind::Project,
        DatabaseMutation::Broker { .. } => MutationKind::Broker,
        DatabaseMutation::EmbeddingProfile { .. } => MutationKind::Policy,
        DatabaseMutation::Security { .. } => MutationKind::Security,
        DatabaseMutation::Graph { .. } => MutationKind::Graph,
    };
    let expected_project = match mutation {
        DatabaseMutation::CreateProject { id, .. }
        | DatabaseMutation::RenameProject { id, .. }
        | DatabaseMutation::DropProject { id, .. } => Some(*id),
        DatabaseMutation::Graph { project, .. } => Some(*project),
        DatabaseMutation::Broker { command } => Some(broker_project(command)),
        DatabaseMutation::EmbeddingProfile { .. } | DatabaseMutation::Security { .. } => None,
    };
    if kind != expected_kind || project_id != expected_project {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "database mutation envelope does not match its payload",
        ));
    }
    Ok(())
}

fn decode_database_mutation(payload: &[u8]) -> Result<DatabaseMutation> {
    if let Some(encoded) = payload.strip_prefix(COMPACT_UNIFORM_VALUE_KAFKA_MUTATION_MAGIC) {
        let compact: CompactUniformValueKafkaMutation =
            postcard::from_bytes(encoded).map_err(|error| {
                Error::new(
                    ErrorCode::CorruptStorage,
                    format!("compact uniform-value Kafka mutation is invalid: {error}"),
                )
            })?;
        if compact.create_times_ms.is_empty() {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "compact uniform-value Kafka mutation has no records",
            ));
        }
        let records = compact
            .create_times_ms
            .into_iter()
            .map(|create_time_ms| crate::broker::KafkaBatchRecord {
                create_time_ms: Some(create_time_ms),
                key: None,
                headers: BTreeMap::new(),
                payload: compact.payload.clone(),
                value_is_null: false,
            })
            .collect();
        return Ok(DatabaseMutation::Broker {
            command: BrokerCommand::PublishKafkaBatch {
                project: compact.project,
                topic: compact.topic,
                partition: compact.partition,
                resolved_time_ms: compact.resolved_time_ms,
                records,
            },
        });
    }
    if let Some(encoded) = payload.strip_prefix(COMPACT_KAFKA_BATCH_MUTATION_MAGIC) {
        let compact: CompactKafkaBatchMutation =
            postcard::from_bytes(encoded).map_err(|error| {
                Error::new(
                    ErrorCode::CorruptStorage,
                    format!("compact Kafka batch mutation is invalid: {error}"),
                )
            })?;
        return Ok(DatabaseMutation::Broker {
            command: BrokerCommand::PublishKafkaBatch {
                project: compact.project,
                topic: compact.topic,
                partition: compact.partition,
                resolved_time_ms: compact.resolved_time_ms,
                records: compact.records,
            },
        });
    }
    if let Some(encoded) = payload.strip_prefix(COMPACT_UNIFORM_AMQP_MUTATION_MAGIC) {
        let compact: CompactUniformAmqpMutation =
            postcard::from_bytes(encoded).map_err(|error| {
                Error::new(
                    ErrorCode::CorruptStorage,
                    format!("compact uniform AMQP mutation is invalid: {error}"),
                )
            })?;
        return Ok(DatabaseMutation::Broker {
            command: BrokerCommand::PublishAmqpUniformBatch {
                project: compact.project,
                resolved_time_ms: compact.resolved_time_ms,
                exchange: compact.exchange,
                routing_key: compact.routing_key,
                mandatory: compact.mandatory,
                properties: compact.properties,
                headers: compact.headers,
                payloads: compact.payloads,
            },
        });
    }
    decode_database_value(payload, "database mutation payload")
}

fn resolve_sequencer_values(
    mutation: &mut DatabaseMutation,
    sequencer_time_millis: i64,
) -> Result<()> {
    let sequencer_time_nanos = millis_to_nanos(sequencer_time_millis)?;
    if let DatabaseMutation::Graph { temporal, .. } = mutation {
        for mutation in temporal {
            if mutation.uses_commit_time {
                mutation.sample.event_time_nanos = sequencer_time_nanos;
            }
        }
    }
    if let DatabaseMutation::Broker { command } = mutation {
        match command {
            BrokerCommand::PublishKafkaBatch {
                resolved_time_ms, ..
            }
            | BrokerCommand::PublishAmqp {
                resolved_time_ms, ..
            }
            | BrokerCommand::PublishAmqpBatch {
                resolved_time_ms, ..
            }
            | BrokerCommand::PublishAmqpUniformBatch {
                resolved_time_ms, ..
            }
            | BrokerCommand::Retain {
                resolved_time_ms, ..
            }
            | BrokerCommand::JoinGroup {
                resolved_time_ms, ..
            }
            | BrokerCommand::HeartbeatGroup {
                resolved_time_ms, ..
            }
            | BrokerCommand::DeliverQueue {
                resolved_time_ms, ..
            }
            | BrokerCommand::RenewDeliveryLease {
                resolved_time_ms, ..
            } => *resolved_time_ms = sequencer_time_millis,
            _ => {}
        }
    }
    Ok(())
}

fn request_intent_digest(mutation: &DatabaseMutation) -> Result<[u8; 32]> {
    let mut normalized = mutation.clone();
    resolve_sequencer_values(&mut normalized, 0)?;
    let encoded = encode_database_value(&normalized, "idempotent request intent")?;
    let mut hasher = blake3::Hasher::new_derive_key("irongraph.request-intent.v1");
    hasher.update(&encoded);
    Ok(*hasher.finalize().as_bytes())
}

fn millis_to_nanos(millis: i64) -> Result<i64> {
    millis
        .checked_mul(1_000_000)
        .ok_or_else(|| Error::invalid_data("resolved commit time exceeds nanosecond range"))
}

fn validate_mutation_plan(validation: &MutationValidation) -> Result<()> {
    if validation
        .dependencies
        .entities
        .values()
        .chain(validation.dependencies.predicates.values())
        .any(|revision| *revision > validation.snapshot.index)
    {
        return Err(Error::new(
            ErrorCode::ProtocolViolation,
            "mutation dependency revision exceeds its planning snapshot",
        ));
    }
    Ok(())
}

fn encode_database_value<T: Serialize>(value: &T, description: &str) -> Result<Vec<u8>> {
    let mut encoded = Vec::new();
    ciborium::ser::into_writer(value, &mut encoded)
        .map_err(|error| Error::invalid_data(format!("{description} encoding failed: {error}")))?;
    Ok(encoded)
}

fn encode_database_mutation_bounded(mutation: &DatabaseMutation, limit: usize) -> Result<Vec<u8>> {
    if let DatabaseMutation::Broker {
        command:
            BrokerCommand::PublishKafkaBatch {
                project,
                topic,
                partition,
                resolved_time_ms,
                records,
            },
    } = mutation
        && let Some(first) = records.first()
        && first.create_time_ms.is_some()
        && first.key.is_none()
        && first.headers.is_empty()
        && !first.value_is_null
        && records.iter().skip(1).all(|record| {
            record.create_time_ms.is_some()
                && record.key.is_none()
                && record.headers.is_empty()
                && !record.value_is_null
                && record.payload == first.payload
        })
    {
        let create_times_ms = records
            .iter()
            .map(|record| {
                record.create_time_ms.ok_or_else(|| {
                    Error::internal("uniform Kafka record lost its validated create timestamp")
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut encoded = Vec::with_capacity(
            COMPACT_UNIFORM_VALUE_KAFKA_MUTATION_MAGIC
                .len()
                .saturating_add(first.payload.len())
                .saturating_add(create_times_ms.len().saturating_mul(8))
                .saturating_add(topic.len())
                .saturating_add(64),
        );
        encoded.extend_from_slice(COMPACT_UNIFORM_VALUE_KAFKA_MUTATION_MAGIC);
        encoded = postcard::to_extend(
            &CompactUniformValueKafkaMutationRef {
                project: *project,
                topic,
                partition: *partition,
                resolved_time_ms: *resolved_time_ms,
                payload: &first.payload,
                create_times_ms: &create_times_ms,
            },
            encoded,
        )
        .map_err(|error| {
            Error::invalid_data(format!(
                "compact uniform-value Kafka mutation encoding failed: {error}"
            ))
        })?;
        if encoded.len() > limit {
            return Err(transaction_admission_full(
                "encoded mutation exceeds write-admission bytes",
            ));
        }
        return Ok(encoded);
    }
    if let DatabaseMutation::Broker {
        command:
            BrokerCommand::PublishKafkaBatch {
                project,
                topic,
                partition,
                resolved_time_ms,
                records,
            },
    } = mutation
    {
        let mut encoded = Vec::with_capacity(
            COMPACT_KAFKA_BATCH_MUTATION_MAGIC
                .len()
                .saturating_add(
                    records
                        .iter()
                        .map(|record| record.payload.len())
                        .fold(0_usize, usize::saturating_add),
                )
                .saturating_add(records.len().saturating_mul(16))
                .saturating_add(topic.len())
                .saturating_add(64),
        );
        encoded.extend_from_slice(COMPACT_KAFKA_BATCH_MUTATION_MAGIC);
        encoded = postcard::to_extend(
            &CompactKafkaBatchMutationRef {
                project: *project,
                topic,
                partition: *partition,
                resolved_time_ms: *resolved_time_ms,
                records,
            },
            encoded,
        )
        .map_err(|error| {
            Error::invalid_data(format!(
                "compact Kafka batch mutation encoding failed: {error}"
            ))
        })?;
        if encoded.len() > limit {
            return Err(transaction_admission_full(
                "encoded mutation exceeds write-admission bytes",
            ));
        }
        return Ok(encoded);
    }
    if let DatabaseMutation::Broker {
        command:
            BrokerCommand::PublishAmqpUniformBatch {
                project,
                resolved_time_ms,
                exchange,
                routing_key,
                mandatory,
                properties,
                headers,
                payloads,
            },
    } = mutation
    {
        let mut encoded = Vec::with_capacity(
            COMPACT_UNIFORM_AMQP_MUTATION_MAGIC
                .len()
                .saturating_add(
                    payloads
                        .iter()
                        .map(Vec::len)
                        .fold(0_usize, usize::saturating_add),
                )
                .saturating_add(256),
        );
        encoded.extend_from_slice(COMPACT_UNIFORM_AMQP_MUTATION_MAGIC);
        encoded = postcard::to_extend(
            &CompactUniformAmqpMutationRef {
                project: *project,
                resolved_time_ms: *resolved_time_ms,
                exchange,
                routing_key,
                mandatory: *mandatory,
                properties,
                headers,
                payloads,
            },
            encoded,
        )
        .map_err(|error| {
            Error::invalid_data(format!(
                "compact uniform AMQP mutation encoding failed: {error}"
            ))
        })?;
        if encoded.len() > limit {
            return Err(transaction_admission_full(
                "encoded mutation exceeds write-admission bytes",
            ));
        }
        return Ok(encoded);
    }
    encode_database_value_bounded(mutation, "mutation", limit)
}

struct BoundedEncoding {
    bytes: Vec<u8>,
    limit: usize,
    exceeded: bool,
}

impl Write for BoundedEncoding {
    fn write(&mut self, input: &[u8]) -> std::io::Result<usize> {
        let next = self.bytes.len().checked_add(input.len());
        if next.is_none_or(|next| next > self.limit) {
            self.exceeded = true;
            return Err(std::io::Error::new(
                std::io::ErrorKind::OutOfMemory,
                "bounded encoding limit exceeded",
            ));
        }
        self.bytes.extend_from_slice(input);
        Ok(input.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

fn encode_database_value_bounded<T: Serialize>(
    value: &T,
    description: &str,
    limit: usize,
) -> Result<Vec<u8>> {
    let mut output = BoundedEncoding {
        bytes: Vec::with_capacity(limit.min(64 * 1024)),
        limit,
        exceeded: false,
    };
    let encoded = ciborium::ser::into_writer(value, &mut output);
    if output.exceeded {
        return Err(transaction_admission_full(
            "encoded mutation exceeds write-admission bytes",
        ));
    }
    encoded
        .map_err(|error| Error::invalid_data(format!("{description} encoding failed: {error}")))?;
    Ok(output.bytes)
}

fn decode_database_value<T: serde::de::DeserializeOwned>(
    bytes: &[u8],
    description: &str,
) -> Result<T> {
    ciborium::de::from_reader(bytes).map_err(|error| {
        Error::new(
            ErrorCode::CorruptStorage,
            format!("{description} is invalid: {error}"),
        )
    })
}

fn block_on_write<F, T>(binding: &WriteBinding, future: F) -> Result<T>
where
    F: std::future::Future<Output = Result<T>>,
{
    match tokio::runtime::Handle::try_current() {
        Ok(current) if current.runtime_flavor() == tokio::runtime::RuntimeFlavor::MultiThread => {
            tokio::task::block_in_place(|| binding.handle.block_on(future))
        }
        Ok(_) => Err(Error::internal(
            "synchronous database execution requires a multi-thread Tokio runtime",
        )),
        Err(_) => binding.handle.block_on(future),
    }
}

fn builtin_functions() -> Vec<String> {
    [
        "abs",
        "avg",
        "ceil",
        "coalesce",
        "collect",
        "count",
        "date",
        "datetime",
        "duration",
        "floor",
        "max",
        "min",
        "percentileCont",
        "percentileDisc",
        "round",
        "stDev",
        "stDevP",
        "sum",
        "time",
        "toBoolean",
        "toFloat",
        "toInteger",
        "toString",
        "vector.similarity.cosine",
        "vector.similarity.euclidean",
    ]
    .into_iter()
    .map(str::to_owned)
    .collect()
}

fn temporal_type(value: &str) -> Result<crate::graph::TemporalType> {
    match value.to_ascii_uppercase().as_str() {
        "BOOLEAN" => Ok(crate::graph::TemporalType::Boolean),
        "INTEGER" => Ok(crate::graph::TemporalType::Integer),
        "FLOAT" => Ok(crate::graph::TemporalType::Float),
        "STRING" => Ok(crate::graph::TemporalType::String),
        "DATE" => Ok(crate::graph::TemporalType::Date),
        "LOCALTIME" | "LOCAL TIME" => Ok(crate::graph::TemporalType::LocalTime),
        "ZONEDTIME" | "ZONED TIME" | "TIME" => Ok(crate::graph::TemporalType::ZonedTime),
        "LOCALDATETIME" | "LOCAL DATETIME" => Ok(crate::graph::TemporalType::LocalDateTime),
        "ZONEDDATETIME" | "ZONED DATETIME" | "DATETIME" => {
            Ok(crate::graph::TemporalType::ZonedDateTime)
        }
        "DURATION" => Ok(crate::graph::TemporalType::Duration),
        _ => Err(Error::new(
            ErrorCode::QueryType,
            "unsupported temporal scalar type",
        )),
    }
}

fn duration_expression_nanos(expression: &crate::cypher::Expression) -> Result<i64> {
    match expression {
        crate::cypher::Expression::Literal(ScalarValue::Integer(value)) if *value > 0 => Ok(*value),
        crate::cypher::Expression::Literal(ScalarValue::Duration {
            months: 0,
            days,
            seconds,
            nanos,
        }) if *days >= 0 && *seconds >= 0 && *nanos >= 0 => days
            .checked_mul(86_400)
            .and_then(|value| value.checked_add(*seconds))
            .and_then(|value| value.checked_mul(1_000_000_000))
            .and_then(|value| value.checked_add(i64::from(*nanos)))
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::TemporalRange,
                    "duration retention overflows or is empty",
                )
            }),
        crate::cypher::Expression::Function {
            name,
            arguments,
            distinct: false,
        } if name.len() == 1 && name[0].eq_ignore_ascii_case("duration") => {
            let Some(crate::cypher::Expression::Literal(ScalarValue::String(value))) =
                arguments.first()
            else {
                return Err(Error::new(
                    ErrorCode::TemporalRange,
                    "duration DDL expression must contain a constant string",
                ));
            };
            parse_fixed_duration_nanos(value)
        }
        _ => Err(Error::new(
            ErrorCode::TemporalRange,
            "retention/window width must be a positive fixed duration or nanosecond integer",
        )),
    }
}

fn instant_expression_nanos(expression: &crate::cypher::Expression) -> Result<i64> {
    match expression {
        crate::cypher::Expression::Literal(ScalarValue::Integer(value)) => Ok(*value),
        crate::cypher::Expression::Literal(ScalarValue::ZonedDateTime {
            seconds, nanos, ..
        }) => seconds
            .checked_mul(1_000_000_000)
            .and_then(|value| value.checked_add(i64::from(*nanos)))
            .ok_or_else(|| Error::new(ErrorCode::TemporalRange, "ALIGN instant overflows")),
        crate::cypher::Expression::Function {
            name,
            arguments,
            distinct: false,
        } if name.len() == 1 && name[0].eq_ignore_ascii_case("datetime") => {
            let Some(crate::cypher::Expression::Literal(ScalarValue::String(value))) =
                arguments.first()
            else {
                return Err(Error::new(
                    ErrorCode::TemporalRange,
                    "datetime DDL expression must contain a constant string",
                ));
            };
            let instant = chrono::DateTime::parse_from_rfc3339(value).map_err(|_| {
                Error::new(
                    ErrorCode::TemporalRange,
                    "ALIGN requires an RFC 3339 instant",
                )
            })?;
            instant
                .timestamp()
                .checked_mul(1_000_000_000)
                .and_then(|nanos| nanos.checked_add(i64::from(instant.timestamp_subsec_nanos())))
                .ok_or_else(|| Error::new(ErrorCode::TemporalRange, "ALIGN instant overflows"))
        }
        _ => Err(Error::new(
            ErrorCode::TemporalRange,
            "ALIGN requires a constant nanosecond integer or datetime instant",
        )),
    }
}

fn parse_fixed_duration_nanos(value: &str) -> Result<i64> {
    let rest = value.strip_prefix('P').ok_or_else(|| {
        Error::new(
            ErrorCode::TemporalRange,
            "duration must use ISO-8601 P notation",
        )
    })?;
    let mut total = 0_f64;
    let mut number = String::new();
    let mut in_time = false;
    for character in rest.chars() {
        if character == 'T' {
            if in_time || !number.is_empty() {
                return Err(Error::new(ErrorCode::TemporalRange, "invalid duration"));
            }
            in_time = true;
            continue;
        }
        if character.is_ascii_digit() || character == '.' {
            number.push(character);
            continue;
        }
        let component = number
            .parse::<f64>()
            .map_err(|_| Error::new(ErrorCode::TemporalRange, "invalid duration component"))?;
        number.clear();
        let seconds = match (character, in_time) {
            ('D', false) => component * 86_400.0,
            ('H', true) => component * 3_600.0,
            ('M', true) => component * 60.0,
            ('S', true) => component,
            _ => {
                return Err(Error::new(
                    ErrorCode::TemporalRange,
                    "fixed duration contains a calendar or unsupported component",
                ));
            }
        };
        if !seconds.is_finite() || seconds < 0.0 {
            return Err(Error::new(
                ErrorCode::TemporalRange,
                "invalid duration value",
            ));
        }
        total += seconds;
    }
    if !number.is_empty() || !total.is_finite() || total <= 0.0 {
        return Err(Error::new(
            ErrorCode::TemporalRange,
            "duration is empty or invalid",
        ));
    }
    let nanos = total * 1_000_000_000.0;
    if nanos > i64::MAX as f64 {
        return Err(Error::new(ErrorCode::TemporalRange, "duration overflows"));
    }
    Ok(nanos.round() as i64)
}

fn validate_project_name(name: &str) -> Result<()> {
    if name.trim().is_empty() || name.len() > 255 || name.chars().any(char::is_control) {
        return Err(Error::invalid_data(
            "project name is empty, oversized, or contains control characters",
        ));
    }
    Ok(())
}

fn normalize_name(name: &str) -> String {
    name.trim().to_lowercase()
}

fn deterministic_project_id(store: StoreId, request: Uuid) -> ProjectId {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"irongraph/project-id/v1");
    hasher.update(store.0.as_bytes());
    hasher.update(request.as_bytes());
    let digest = hasher.finalize();
    let mut bytes = [0_u8; 16];
    bytes.copy_from_slice(&digest.as_bytes()[..16]);
    // RFC 9562 UUIDv8 marks this as an application-defined deterministic identifier.
    bytes[6] = (bytes[6] & 0x0f) | 0x80;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    ProjectId(Uuid::from_bytes(bytes))
}

fn unix_nanos(time: SystemTime) -> Result<i64> {
    i64::try_from(
        time.duration_since(UNIX_EPOCH)
            .map_err(|_| Error::invalid_data("time precedes epoch"))?
            .as_nanos(),
    )
    .map_err(|_| Error::invalid_data("time exceeds i64 nanoseconds"))
}

#[cfg(test)]
mod automatic_semantic_tests;

#[cfg(test)]
mod project_lifecycle_tests {
    use crate::{
        EdgeId,
        Layer,
        NodeId,
        ScalarValue,
        // Tests import the private identity owner directly to construct isolated stores.
        engine::NodeIdentity,
        graph::{DocumentItem, EdgeInput, NodeInput},
        protocol::QueryLimits,
        storage::{SegmentFamily, SegmentRecord},
        types::DocumentList,
    };

    use super::*;

    struct WindowEmbedding {
        profile: crate::graph::EmbeddingProfile,
        batches: Arc<Mutex<Vec<Vec<String>>>>,
    }

    impl TextEmbedding for WindowEmbedding {
        fn profile(&self) -> &crate::graph::EmbeddingProfile {
            &self.profile
        }

        fn embed(&self, text: &str) -> Result<Vec<f32>> {
            Ok(if text == "head" {
                vec![1.0, 0.0]
            } else {
                vec![0.0, 1.0]
            })
        }

        fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
            self.batches.lock().push(texts.to_vec());
            texts.iter().map(|text| self.embed(text)).collect()
        }

        fn index_windows(&self, text: &str) -> Result<Vec<String>> {
            if text == "head interior tail" {
                Ok(vec!["head".to_owned(), "tail".to_owned()])
            } else {
                Ok(vec![text.to_owned()])
            }
        }
    }

    #[test]
    fn compact_uniform_amqp_mutation_round_trips_with_cbor_fallback() -> Result<()> {
        let project = ProjectId::random();
        let mutation = DatabaseMutation::Broker {
            command: BrokerCommand::PublishAmqpUniformBatch {
                project,
                resolved_time_ms: 0,
                exchange: "events".to_owned(),
                routing_key: "created".to_owned(),
                mandatory: false,
                properties: BTreeMap::new(),
                headers: BTreeMap::new(),
                payloads: (0..400).map(|_| vec![7; 64]).collect(),
            },
        };
        let compact = encode_database_mutation_bounded(&mutation, 1024 * 1024)?;
        assert!(compact.starts_with(COMPACT_UNIFORM_AMQP_MUTATION_MAGIC));
        let decoded = decode_database_mutation(&compact)?;
        assert!(matches!(
            decoded,
            DatabaseMutation::Broker {
                command: BrokerCommand::PublishAmqpUniformBatch {
                    project: decoded_project,
                    ref payloads,
                    ..
                }
            } if decoded_project == project && payloads.len() == 400
        ));

        let legacy = encode_database_value(&mutation, "legacy broker mutation")?;
        assert!(!legacy.starts_with(COMPACT_UNIFORM_AMQP_MUTATION_MAGIC));
        assert!(matches!(
            decode_database_mutation(&legacy)?,
            DatabaseMutation::Broker {
                command: BrokerCommand::PublishAmqpUniformBatch { .. }
            }
        ));
        assert!(compact.len() < legacy.len());
        Ok(())
    }

    #[test]
    fn compact_kafka_batch_mutation_round_trips_with_cbor_fallback() -> Result<()> {
        let project = ProjectId::random();
        let mutation = DatabaseMutation::Broker {
            command: BrokerCommand::PublishKafkaBatch {
                project,
                topic: "events".to_owned(),
                partition: 0,
                resolved_time_ms: 0,
                records: (0..400)
                    .map(|index| crate::broker::KafkaBatchRecord {
                        create_time_ms: Some(index),
                        key: None,
                        headers: BTreeMap::new(),
                        payload: vec![index as u8; 64],
                        value_is_null: false,
                    })
                    .collect(),
            },
        };
        let compact = encode_database_mutation_bounded(&mutation, 1024 * 1024)?;
        assert!(compact.starts_with(COMPACT_KAFKA_BATCH_MUTATION_MAGIC));
        assert!(matches!(
            decode_database_mutation(&compact)?,
            DatabaseMutation::Broker {
                command: BrokerCommand::PublishKafkaBatch {
                    project: decoded_project,
                    ref records,
                    ..
                }
            } if decoded_project == project && records.len() == 400
        ));

        let legacy = encode_database_value(&mutation, "legacy Kafka mutation")?;
        assert!(!legacy.starts_with(COMPACT_KAFKA_BATCH_MUTATION_MAGIC));
        assert!(matches!(
            decode_database_mutation(&legacy)?,
            DatabaseMutation::Broker {
                command: BrokerCommand::PublishKafkaBatch { .. }
            }
        ));
        assert!(
            compact.len() < legacy.len(),
            "compact Kafka mutation was not smaller: compact={} legacy={}",
            compact.len(),
            legacy.len()
        );
        Ok(())
    }

    #[test]
    fn compact_uniform_value_kafka_mutation_stores_one_payload_and_all_timestamps() -> Result<()> {
        let project = ProjectId::random();
        let mutation = DatabaseMutation::Broker {
            command: BrokerCommand::PublishKafkaBatch {
                project,
                topic: "events".to_owned(),
                partition: 0,
                resolved_time_ms: 0,
                records: (0..400)
                    .map(|index| crate::broker::KafkaBatchRecord {
                        create_time_ms: Some(10_000 + index),
                        key: None,
                        headers: BTreeMap::new(),
                        payload: vec![7; 64],
                        value_is_null: false,
                    })
                    .collect(),
            },
        };
        let encoded = encode_database_mutation_bounded(&mutation, 1024 * 1024)?;
        assert!(encoded.starts_with(COMPACT_UNIFORM_VALUE_KAFKA_MUTATION_MAGIC));
        assert!(encoded.len() < 4_096, "uniform Kafka WAL shape regressed");
        let decoded = decode_database_mutation(&encoded)?;
        let DatabaseMutation::Broker {
            command: BrokerCommand::PublishKafkaBatch { records, .. },
        } = decoded
        else {
            return Err(Error::internal(
                "uniform Kafka mutation decoded to the wrong command",
            ));
        };
        assert_eq!(records.len(), 400);
        assert_eq!(records[0].create_time_ms, Some(10_000));
        assert_eq!(records[399].create_time_ms, Some(10_399));
        assert!(records.iter().all(|record| record.payload == vec![7; 64]));
        Ok(())
    }

    fn write_unchecked_test_snapshot(
        database: &Database,
        state: &DatabaseState,
        destination: &Path,
    ) -> Result<BackendSnapshot> {
        let mut state_bytes = Vec::new();
        ciborium::ser::into_writer(state, &mut state_bytes).map_err(|error| {
            Error::invalid_data(format!("test checkpoint state encoding failed: {error}"))
        })?;
        let manifest = DatabaseCheckpointManifest {
            format_version: DATABASE_SNAPSHOT_FORMAT,
            store_id: database.0.store_id,
            included: state.applied,
            state_bytes: state_bytes.len() as u64,
            broker_segments: Vec::new(),
        };
        let manifest_bytes = encode_database_value(&manifest, "test checkpoint manifest")?;
        let mut output = File::create(destination)?;
        output.write_all(&DATABASE_SNAPSHOT_MAGIC)?;
        output.write_all(&(manifest_bytes.len() as u32).to_be_bytes())?;
        output.write_all(&manifest_bytes)?;
        output.write_all(&state_bytes)?;
        output.flush()?;
        let bytes = output.metadata()?.len();
        drop(output);
        BackendSnapshot::new(destination.to_owned(), bytes, Vec::new())
    }

    #[tokio::test]
    async fn standalone_snapshot_retains_previous_replay_authority() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let snapshot_directory = tempfile::tempdir()?;
        let database = Database::open_backend(
            directory.path(),
            2 * 1024 * 1024,
            Duration::from_secs(1),
            NodeIdentity::generate_genesis().public(),
        )?;
        for index in 1..=3 {
            database.0.state.write().applied = Bookmark { term: 1, index };
            database
                .standalone_snapshot(snapshot_directory.path())
                .await?;
        }

        let retained = standalone_snapshot_paths(snapshot_directory.path());
        assert_eq!(retained.len(), 2);
        assert_eq!(
            database
                .standalone_wal_compaction_bookmark(
                    snapshot_directory.path(),
                    Bookmark { term: 1, index: 3 },
                )
                .await?,
            Bookmark { term: 1, index: 2 }
        );
        Ok(())
    }

    #[tokio::test]
    async fn standalone_recovery_falls_back_from_damaged_latest_snapshot() -> Result<()> {
        let source_directory = tempfile::tempdir()?;
        let snapshot_directory = tempfile::tempdir()?;
        let identity = NodeIdentity::generate_genesis().public();
        let database = Database::open_backend(
            source_directory.path(),
            2 * 1024 * 1024,
            Duration::from_secs(1),
            identity,
        )?;
        for index in 1..=2 {
            database.0.state.write().applied = Bookmark { term: 1, index };
            database
                .standalone_snapshot(snapshot_directory.path())
                .await?;
        }
        let latest_name = std::fs::read_to_string(snapshot_directory.path().join("LATEST"))?;
        let latest = snapshot_directory.path().join(latest_name.trim());
        let mut bytes = std::fs::read(&latest)?;
        bytes[0] ^= 0xff;
        std::fs::write(&latest, bytes)?;

        let restored_directory = tempfile::tempdir()?;
        let restored = Database::open_backend(
            restored_directory.path(),
            2 * 1024 * 1024,
            Duration::from_secs(1),
            identity,
        )?;
        assert_eq!(
            restored
                .standalone_recover(snapshot_directory.path())
                .await?,
            Some(Bookmark { term: 1, index: 1 })
        );
        Ok(())
    }

    #[test]
    fn checkpoint_recovery_quarantines_only_invalid_relationship_rows() -> Result<()> {
        let source_directory = tempfile::tempdir()?;
        let identity = NodeIdentity::generate_genesis().public();
        let source = Database::open_backend(
            source_directory.path(),
            2 * 1024 * 1024,
            Duration::from_secs(1),
            identity,
        )?;
        let project = ProjectId::random();
        let mut project_state = ordered_graph_project(project, 1)?;
        {
            let project_state = Arc::make_mut(&mut project_state);
            let relationship_type = project_state
                .graph
                .catalog_mut()
                .intern_relationship_type("CONNECTED")?;
            project_state.graph.insert_node(NodeInput {
                id: NodeId(2),
                layer: Layer::Observed,
                revision: 2,
                labels: Vec::new(),
                properties: Vec::new(),
            })?;
            project_state.graph.insert_edge(EdgeInput {
                id: EdgeId(1),
                source: NodeId(1),
                target: NodeId(2),
                relationship_type,
                layer: Layer::Observed,
                revision: 3,
                properties: Vec::new(),
            })?;
            project_state.next_node_id = 3;
            project_state.next_edge_id = 2;
        }
        let bookmark = Bookmark { term: 1, index: 3 };
        let mut state = DatabaseState {
            applied: bookmark,
            ..DatabaseState::default()
        };
        state
            .names
            .insert(normalize_name(&project_state.display_name), project);
        state.projects.insert(project, project_state);
        validate_database_state(&state)?;

        let mut encoded = serde_json::to_value(&state)
            .map_err(|error| Error::invalid_data(format!("test JSON encoding failed: {error}")))?;
        let project_value = encoded
            .get_mut("projects")
            .and_then(serde_json::Value::as_object_mut)
            .and_then(|projects| projects.values_mut().next())
            .ok_or_else(|| Error::invalid_data("test checkpoint project is absent"))?;
        let source_value = project_value
            .get_mut("graph")
            .and_then(|graph| graph.get_mut("edge_sources"))
            .and_then(serde_json::Value::as_array_mut)
            .and_then(|sources| sources.first_mut())
            .ok_or_else(|| Error::invalid_data("test relationship source is absent"))?;
        *source_value = serde_json::Value::from(u32::MAX);
        let corrupt_state: DatabaseState = serde_json::from_value(encoded)
            .map_err(|error| Error::invalid_data(format!("test JSON decoding failed: {error}")))?;
        assert!(validate_database_state(&corrupt_state).is_err());

        let snapshot_path = source_directory.path().join("corrupt.igdb");
        let snapshot = write_unchecked_test_snapshot(&source, &corrupt_state, &snapshot_path)?;
        let restored_directory = tempfile::tempdir()?;
        let restored = Database::open_backend(
            restored_directory.path(),
            2 * 1024 * 1024,
            Duration::from_secs(1),
            identity,
        )?;
        install_database_snapshot_file(&restored, bookmark, &snapshot)?;
        let restored_state = restored.0.state.read();
        let graph = &restored_state.projects[&project].graph;
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 0);
        graph.validate_structure()
    }

    /// The device-publication path must not defeat the lag tolerance that governs everywhere else.
    ///
    /// `rebind_shared_project` runs after every committed write. It used to clear the snapshot
    /// outright, which made `OPTIMIZER_STATISTICS_MAX_REVISION_LAG` unreachable — by the time
    /// `expire_optimizer_statistics_if_stale` was consulted there was nothing left to expire, so a
    /// tolerance measured in hundreds of commits never survived a single one. The next request then
    /// rebuilt the snapshot by walking every node, and on a 200 000-node graph that cost ~148 ms —
    /// paid even by `RETURN 1`, which needs no statistics, because the request path collects
    /// whenever a backend exists rather than when the plan asks for it.
    #[test]
    fn a_publication_within_the_lag_tolerance_keeps_the_statistics_snapshot() {
        let mut project = ProjectState {
            id: ProjectId::random(),
            display_name: "rebind".to_owned(),
            graph: GraphStore::default().into(),
            temporal: TemporalStore::default().into(),
            predicate_versions: BTreeMap::new().into(),
            indexes: IndexCatalog::default().into(),
            next_node_id: 1,
            next_edge_id: 1,
            authority_revision: 1,
            optimizer_statistics: OnceLock::new(),
        };
        // A snapshot whose recorded revision is non-zero and close to the graph's: the ordinary
        // steady state after a few commits. Revision 0 is deliberately excluded — the first
        // transition out of an empty generation must still be visible immediately.
        let mut collected = StatisticsSnapshot::collect_project(&project.graph, None, None);
        collected.graph_revision = 1;
        collected.schema_generation = project.graph.catalog().optimizer_generation();
        collected.index_generation = project.indexes.optimizer_generation();
        let _ = project.optimizer_statistics.set(Arc::new(collected));

        expire_optimizer_statistics_if_stale(&mut project);
        assert!(
            project.optimizer_statistics.get().is_some(),
            "a snapshot inside the revision lag must survive publication, or every write makes the \
             next request walk the whole graph"
        );
    }

    #[test]
    fn caller_output_limit_does_not_bound_intermediate_execution_rows() -> Result<()> {
        let project = ProjectId::random();
        let snapshot = ProjectState {
            id: project,
            display_name: "result-policy-test".to_owned(),
            graph: GraphStore::default().into(),
            temporal: TemporalStore::default().into(),
            predicate_versions: BTreeMap::new().into(),
            indexes: IndexCatalog::default().into(),
            next_node_id: 1,
            next_edge_id: 1,
            authority_revision: 1,
            optimizer_statistics: OnceLock::new(),
        };
        let request = QueryRequest {
            request_id: Uuid::new_v4(),
            project_id: Some(project),
            query: "UNWIND range(1, 129) AS i RETURN count(*) AS count".to_owned(),
            parameters: BTreeMap::new(),
            consistency: CommitAcknowledgement::Published,
            bookmark: None,
            limits: QueryLimits {
                rows: 1,
                ..QueryLimits::default()
            },
            cancellation: Default::default(),
            deadline: None,
            connection_id: ConnectionId::new(),
        };

        let output = execute_on_project_inner(
            &snapshot,
            &request,
            Bookmark { term: 1, index: 1 },
            2,
            full_capabilities(),
            None,
            None,
            None,
        )?;
        let values = output
            .result
            .batches
            .iter()
            .flat_map(|batch| &batch.columns[0].values)
            .collect::<Vec<_>>();
        assert_eq!(values, [&ResultValue::Scalar(ScalarValue::Integer(129))]);
        Ok(())
    }

    #[test]
    fn extracted_fact_merge_executes_and_builds_the_graph() -> Result<()> {
        // The exact Cypher the long-term-memory write-back emits for one extracted fact, run through
        // the full parse -> bind -> plan -> execute pipeline against an empty project.
        let project = ProjectId::random();
        let snapshot = ProjectState {
            id: project,
            display_name: "memory".to_owned(),
            graph: GraphStore::default().into(),
            temporal: TemporalStore::default().into(),
            predicate_versions: BTreeMap::new().into(),
            indexes: IndexCatalog::default().into(),
            next_node_id: 1,
            next_edge_id: 1,
            authority_revision: 0,
            optimizer_statistics: OnceLock::new(),
        };
        let mut parameters = BTreeMap::new();
        parameters.insert("subject".to_owned(), serde_json::json!("Ada Lovelace"));
        parameters.insert("object".to_owned(), serde_json::json!("Charles Babbage"));
        parameters.insert(
            "text".to_owned(),
            serde_json::json!("On the Analytical Engine."),
        );
        let request = QueryRequest {
            request_id: Uuid::new_v4(),
            project_id: Some(project),
            query: "WRITE LAYER KNOWLEDGE \
                    MERGE (s:`Contact` {display: $subject}) \
                    MERGE (o:`Contact` {display: $object}) \
                    MERGE (s)-[r:`COLLABORATED_WITH`]->(o) \
                    SET r.text = $text"
                .to_owned(),
            parameters,
            consistency: CommitAcknowledgement::Published,
            bookmark: None,
            limits: QueryLimits::default(),
            cancellation: Default::default(),
            deadline: None,
            connection_id: ConnectionId::new(),
        };
        let output = execute_on_project_inner(
            &snapshot,
            &request,
            Bookmark { term: 1, index: 1 },
            2,
            full_capabilities(),
            None,
            None,
            None,
        )?;
        // MERGE on an empty graph creates both entities and the typed relationship.
        let nodes = output
            .graph_mutations
            .iter()
            .filter(|mutation| matches!(mutation, GraphMutation::InsertNode(_)))
            .count();
        let edges = output
            .graph_mutations
            .iter()
            .filter(|mutation| matches!(mutation, GraphMutation::InsertEdge(_)))
            .count();
        assert_eq!(
            nodes, 2,
            "both entities were created: {:?}",
            output.graph_mutations
        );
        assert_eq!(
            edges, 1,
            "the typed relationship was created: {:?}",
            output.graph_mutations
        );
        Ok(())
    }

    #[test]
    fn grounded_observation_write_executes_and_links_by_id() -> Result<()> {
        // A pre-existing node the observation will be grounded to.
        let project = ProjectId::random();
        let mut graph = GraphStore::default();
        let person = graph.catalog_mut().intern_label("Person")?;
        let name = graph.catalog_mut().intern_property("name")?;
        graph.apply(GraphMutation::InsertNode(NodeInput {
            id: NodeId(1),
            layer: Layer::Observed,
            revision: 1,
            labels: vec![person],
            properties: vec![(name, ScalarValue::String(Arc::from("Zorbex")))],
        }))?;
        let snapshot = ProjectState {
            id: project,
            display_name: "memory".to_owned(),
            graph: graph.into(),
            temporal: TemporalStore::default().into(),
            predicate_versions: BTreeMap::new().into(),
            indexes: IndexCatalog::default().into(),
            next_node_id: 2,
            next_edge_id: 1,
            authority_revision: 1,
            optimizer_statistics: OnceLock::new(),
        };
        let mut parameters = BTreeMap::new();
        parameters.insert("name".to_owned(), serde_json::json!("What is Zorbex?"));
        parameters.insert("text".to_owned(), serde_json::json!("Zorbex is a product."));
        parameters.insert("grounded".to_owned(), serde_json::json!([1]));
        let request = QueryRequest {
            request_id: Uuid::new_v4(),
            project_id: Some(project),
            query: "WRITE LAYER KNOWLEDGE \
                    CREATE (o:Observation {name: $name, text: $text}) \
                    WITH o UNWIND $grounded AS gid \
                    MATCH (n) WHERE id(n) = gid \
                    CREATE (o)-[:DERIVED_FROM]->(n)"
                .to_owned(),
            parameters,
            consistency: CommitAcknowledgement::Published,
            bookmark: None,
            limits: QueryLimits::default(),
            cancellation: Default::default(),
            deadline: None,
            connection_id: ConnectionId::new(),
        };
        let output = execute_on_project_inner(
            &snapshot,
            &request,
            Bookmark { term: 1, index: 2 },
            3,
            full_capabilities(),
            None,
            None,
            None,
        )?;
        // The Observation node is created; the DERIVED_FROM edge links it to the existing node 1.
        let nodes = output
            .graph_mutations
            .iter()
            .filter(|mutation| matches!(mutation, GraphMutation::InsertNode(_)))
            .count();
        let edges = output
            .graph_mutations
            .iter()
            .filter(|mutation| matches!(mutation, GraphMutation::InsertEdge(_)))
            .count();
        assert_eq!(
            nodes, 1,
            "the observation node was created: {:?}",
            output.graph_mutations
        );
        assert_eq!(
            edges, 1,
            "id()-grounded DERIVED_FROM edge was created: {:?}",
            output.graph_mutations
        );
        Ok(())
    }

    #[test]
    fn knowledge_layer_lock_holds_on_the_transaction_staging_path() -> Result<()> {
        let project = ProjectId::random();
        let mut graph = GraphStore::default();
        let product = graph.catalog_mut().intern_label("Product")?;
        let name = graph.catalog_mut().intern_property("name")?;
        let current = ProjectState {
            id: project,
            display_name: "knowledge-lock".to_owned(),
            graph: graph.into(),
            temporal: TemporalStore::default().into(),
            predicate_versions: BTreeMap::new().into(),
            indexes: IndexCatalog::default().into(),
            next_node_id: 1,
            next_edge_id: 1,
            authority_revision: 0,
            optimizer_statistics: OnceLock::new(),
        };
        // A KNOWLEDGE node without the reserved `name` is rejected at the durable write funnel.
        let anonymous = vec![GraphMutation::InsertNode(NodeInput {
            id: NodeId(1),
            layer: Layer::Knowledge,
            revision: 1,
            labels: vec![product],
            properties: vec![],
        })];
        assert!(stage_transaction_project(&current, &anonymous, &[], &[], 0).is_err());
        // The very same shape on the OBSERVED layer is unconstrained business data.
        let observed = vec![GraphMutation::InsertNode(NodeInput {
            id: NodeId(1),
            layer: Layer::Observed,
            revision: 1,
            labels: vec![product],
            properties: vec![],
        })];
        stage_transaction_project(&current, &observed, &[], &[], 0)?;
        // A named KNOWLEDGE node is accepted and staged.
        let named = vec![GraphMutation::InsertNode(NodeInput {
            id: NodeId(1),
            layer: Layer::Knowledge,
            revision: 1,
            labels: vec![product],
            properties: vec![(name, ScalarValue::String(Arc::from("Zorbex Q7")))],
        })];
        let staged = stage_transaction_project(&current, &named, &[], &[], 0)?;
        assert_eq!(
            staged
                .graph
                .node_count_in_layers(crate::graph::LayerMask::KNOWLEDGE),
            1
        );
        Ok(())
    }

    fn transaction_resource(project: ProjectId, bookmark: Bookmark) -> Arc<TransactionResource> {
        let snapshot = Arc::new(ProjectState {
            id: project,
            display_name: "transaction".to_owned(),
            graph: GraphStore::default().into(),
            temporal: TemporalStore::default().into(),
            predicate_versions: BTreeMap::new().into(),
            indexes: IndexCatalog::default().into(),
            next_node_id: 1,
            next_edge_id: 1,
            authority_revision: 0,
            optimizer_statistics: OnceLock::new(),
        });
        Arc::new(TransactionResource {
            terminal: AtomicU8::new(0),
            lifecycle: Mutex::new(TransactionLifecycle::Active(TransactionState {
                catalog: snapshot.graph.catalog().clone(),
                working: Arc::clone(&snapshot),
                snapshot,
                execution: None,
                bookmark,
                consistency: CommitAcknowledgement::Published,
                dependencies: TransactionDependencies::default(),
                graph_mutations: Vec::new(),
                temporal_mutations: Vec::new(),
                vector_mutations: Vec::new(),
                accounted_bytes: MIN_TRANSACTION_ACCOUNTED_BYTES,
            })),
        })
    }

    fn ordered_graph_project(id: ProjectId, revision: u64) -> Result<Arc<ProjectState>> {
        let mut graph = GraphStore::default();
        let property = graph.catalog_mut().intern_property("value")?;
        graph.apply(GraphMutation::InsertNode(crate::graph::NodeInput {
            id: crate::NodeId(1),
            layer: crate::Layer::Observed,
            revision,
            labels: Vec::new(),
            properties: vec![(property, ScalarValue::Integer(0))],
        }))?;
        Ok(Arc::new(ProjectState {
            id,
            display_name: id.to_string(),
            graph: graph.into(),
            temporal: TemporalStore::default().into(),
            predicate_versions: BTreeMap::new().into(),
            indexes: IndexCatalog::default().into(),
            next_node_id: 2,
            next_edge_id: 1,
            authority_revision: revision,
            optimizer_statistics: OnceLock::new(),
        }))
    }

    fn ordered_graph_command(
        project: ProjectId,
        snapshot: Bookmark,
        value: i64,
    ) -> Result<(WriteCommand, DatabaseMutation)> {
        let mut dependencies = TransactionDependencies::default();
        let entity = crate::cypher::EntityDependency::Node(crate::NodeId(1));
        dependencies.entities.insert(entity, snapshot.index);
        dependencies.write_targets.insert(entity);
        let mutation = DatabaseMutation::Graph {
            project,
            validation: MutationValidation {
                snapshot,
                dependencies,
            },
            graph: vec![GraphMutation::SetNodeProperty {
                node: crate::NodeId(1),
                property: crate::types::PropertyId(0),
                value: ScalarValue::Integer(value),
                revision: snapshot.index.saturating_add(1),
            }],
            temporal: Vec::new(),
            vectors: Vec::new(),
            administrative: None,
        };
        let command = WriteCommand {
            kind: MutationKind::Graph,
            project_id: Some(project),
            request_id: Some(Uuid::new_v4()),
            commit_time_millis: 1,
            payload: encode_database_value(&mutation, "ordered graph test mutation")?,
        };
        Ok((command, mutation))
    }

    #[test]
    fn ordered_graph_overlay_pipelines_independent_projects_and_rejects_write_conflicts()
    -> Result<()> {
        let directory = tempfile::tempdir()?;
        let segments = SegmentStore::open(directory.path(), 2 * 1024 * 1024)?;
        let snapshot = Bookmark { term: 2, index: 10 };
        let first_project = ProjectId::random();
        let second_project = ProjectId::random();
        let mut state = DatabaseState {
            applied: snapshot,
            ..DatabaseState::default()
        };
        state
            .projects
            .insert(first_project, ordered_graph_project(first_project, 10)?);
        state
            .projects
            .insert(second_project, ordered_graph_project(second_project, 10)?);

        let (first_command, first_mutation) = ordered_graph_command(first_project, snapshot, 11)?;
        let first = stage_ordered_database_command(
            &state,
            &first_command,
            first_mutation,
            Bookmark { term: 3, index: 11 },
            &segments,
        )?;
        let (second_command, second_mutation) =
            ordered_graph_command(second_project, snapshot, 12)?;
        let second = stage_ordered_database_command(
            &first,
            &second_command,
            second_mutation,
            Bookmark { term: 3, index: 12 },
            &segments,
        )?;
        assert_eq!(second.applied.index, 12);

        let (conflicting_command, conflicting_mutation) =
            ordered_graph_command(first_project, snapshot, 13)?;
        let conflict = stage_ordered_database_command(
            &first,
            &conflicting_command,
            conflicting_mutation,
            Bookmark { term: 3, index: 12 },
            &segments,
        )
        .expect_err("two writes from one entity revision both entered the ordered overlay");
        assert_eq!(conflict.code, ErrorCode::TransactionConflict);
        Ok(())
    }

    #[tokio::test]
    async fn standalone_write_hot_path_applies_canonical_delta_once() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let database = Database::open_backend(
            directory.path(),
            2 * 1024 * 1024,
            Duration::from_secs(1),
            NodeIdentity::generate_genesis().public(),
        )?;
        let snapshot = Bookmark { term: 2, index: 10 };
        let committed = Bookmark { term: 3, index: 11 };
        let project = ProjectId::random();
        {
            let mut state = database.0.state.write();
            state.applied = snapshot;
            state
                .projects
                .insert(project, ordered_graph_project(project, snapshot.index)?);
        }
        let (command, _) = ordered_graph_command(project, snapshot, 11)?;
        let reservation = database.reserve_command(&command, committed).await?;
        let staged_project = database
            .0
            .ordered_overlay
            .lock()
            .states
            .get(&committed.index)
            .and_then(|state| state.projects.get(&project))
            .cloned()
            .ok_or_else(|| Error::internal("graph reservation did not stage its project"))?;
        let entry = MutationEntry::new(
            committed.term,
            committed.index,
            command.kind,
            command.project_id,
            command.request_id,
            command.commit_time_millis,
            command.payload.clone(),
        )?;
        let shared_entry = entry.clone();
        assert_eq!(entry.payload().as_ptr(), shared_entry.payload().as_ptr());

        let applied = database.apply_mutation(&entry).await?;
        assert!(!applied.duplicate);
        assert!(database.0.ordered_overlay.lock().states.is_empty());
        let state = database.0.state.read();
        let canonical_project = state
            .projects
            .get(&project)
            .ok_or_else(|| Error::internal("committed graph project disappeared"))?;
        assert!(Arc::ptr_eq(canonical_project, &staged_project));
        drop(state);
        database
            .complete_command_reservation(reservation, CommandReservationOutcome::Applied)
            .await?;
        Ok(())
    }

    #[tokio::test]
    async fn ordered_graph_reservations_publish_their_exact_batched_generations() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let database = Database::open_backend(
            directory.path(),
            2 * 1024 * 1024,
            Duration::from_secs(1),
            NodeIdentity::generate_genesis().public(),
        )?;
        let snapshot = Bookmark { term: 2, index: 10 };
        let first_project = ProjectId::random();
        let second_project = ProjectId::random();
        {
            let mut state = database.0.state.write();
            state.applied = snapshot;
            state.projects.insert(
                first_project,
                ordered_graph_project(first_project, snapshot.index)?,
            );
            state.projects.insert(
                second_project,
                ordered_graph_project(second_project, snapshot.index)?,
            );
        }
        let (first_command, _) = ordered_graph_command(first_project, snapshot, 11)?;
        let (second_command, _) = ordered_graph_command(second_project, snapshot, 12)?;
        let first_position = Bookmark { term: 3, index: 11 };
        let second_position = Bookmark { term: 3, index: 12 };
        let first_reservation = database
            .reserve_command(&first_command, first_position)
            .await?;
        let second_reservation = database
            .reserve_command(&second_command, second_position)
            .await?;
        assert_eq!(database.0.ordered_overlay.lock().states.len(), 2);
        let first_entry = MutationEntry::new(
            first_position.term,
            first_position.index,
            first_command.kind,
            first_command.project_id,
            first_command.request_id,
            first_command.commit_time_millis,
            first_command.payload,
        )?;
        let second_entry = MutationEntry::new(
            second_position.term,
            second_position.index,
            second_command.kind,
            second_command.project_id,
            second_command.request_id,
            second_command.commit_time_millis,
            second_command.payload,
        )?;

        database.apply_mutation(&first_entry).await?;
        assert_eq!(database.0.state.read().applied, first_position);
        assert_eq!(database.0.ordered_overlay.lock().states.len(), 1);
        database.apply_mutation(&second_entry).await?;
        assert_eq!(database.0.state.read().applied, second_position);
        assert!(database.0.ordered_overlay.lock().states.is_empty());
        database
            .complete_command_reservation(first_reservation, CommandReservationOutcome::Applied)
            .await?;
        database
            .complete_command_reservation(second_reservation, CommandReservationOutcome::Applied)
            .await?;
        assert!(database.0.ordered_overlay.lock().reservations.is_empty());
        Ok(())
    }

    #[test]
    fn reused_request_id_errors_and_identical_command_replays_without_reapplying() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let segments = SegmentStore::open(directory.path(), 2 * 1024 * 1024)?;
        let snapshot = Bookmark { term: 2, index: 10 };
        let project = ProjectId::random();
        let mut state = DatabaseState {
            applied: snapshot,
            ..DatabaseState::default()
        };
        state
            .projects
            .insert(project, ordered_graph_project(project, 10)?);

        let (command, mutation) = ordered_graph_command(project, snapshot, 11)?;
        let first = stage_ordered_database_command(
            &state,
            &command,
            mutation.clone(),
            Bookmark { term: 3, index: 11 },
            &segments,
        )?;

        // Reusing a request ID for a different mutation — the fixed-id seeding pattern an external
        // /api/query client can fall into — is a typed error, never a silent success with no write.
        let (mut reused_command, reused_mutation) =
            ordered_graph_command(project, Bookmark { term: 3, index: 11 }, 12)?;
        reused_command.request_id = command.request_id;
        let reuse = stage_ordered_database_command(
            &first,
            &reused_command,
            reused_mutation,
            Bookmark { term: 3, index: 12 },
            &segments,
        )
        .expect_err("a reused request ID with a different mutation entered the overlay");
        assert_eq!(reuse.code, ErrorCode::ProtocolViolation);
        assert!(
            reuse.message.contains("idempotency key"),
            "the error must teach the request-ID contract: {}",
            reuse.message
        );

        // The identical planned command replays: the recorded response is returned, the position
        // advances, and the mutation is not applied a second time.
        let replay = stage_ordered_database_command(
            &first,
            &command,
            mutation,
            Bookmark { term: 3, index: 12 },
            &segments,
        )?;
        assert_eq!(replay.applied.index, 12);
        assert_eq!(*replay.last_response, *first.last_response);
        let first_project = first.projects.get(&project).expect("staged project");
        let replay_project = replay.projects.get(&project).expect("replayed project");
        assert!(
            Arc::ptr_eq(first_project, replay_project),
            "a replay must return the recorded response without touching project state"
        );
        Ok(())
    }

    #[test]
    fn uncommitted_broker_prestage_pin_fences_running_reclamation() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let database = Database::open_backend(
            directory.path(),
            2 * 1024 * 1024,
            Duration::from_millis(1),
            NodeIdentity::generate_genesis().public(),
        )?;
        let project = ProjectId::random();
        let command = BrokerCommand::PublishKafkaBatch {
            project,
            topic: "events".to_owned(),
            partition: 0,
            resolved_time_ms: 1,
            records: vec![crate::broker::KafkaBatchRecord {
                create_time_ms: None,
                key: None,
                headers: BTreeMap::new(),
                payload: b"uncommitted".to_vec(),
                value_is_null: false,
            }],
        };
        let staged = {
            let mut state = database.0.state.write();
            state.broker.apply(
                BrokerCommand::CreateTopic {
                    project,
                    name: "events".to_owned(),
                    partitions: 1,
                    retention: crate::broker::RetentionPolicy {
                        max_age_ms: None,
                        max_bytes: None,
                    },
                },
                &database.0.segments,
            )?;
            state
                .broker
                .prestage_command_payload(&command, &database.0.segments)?
        };
        assert_eq!(staged.len(), 1);
        let pin = database.0.segments.pin(&staged[0])?;
        database.0.pending_broker_segments.lock().insert(
            staged[0].clone(),
            PendingBrokerSegment {
                _pin: pin,
                owners: BTreeSet::from([Uuid::new_v4()]),
            },
        );
        database
            .0
            .retired_broker_segments
            .lock()
            .insert(staged[0].clone());
        assert_eq!(database.0.segments.immutable_file_names()?.len(), 1);
        assert_eq!(database.reclaim_payload_storage()?, 0);
        assert_eq!(database.0.segments.immutable_file_names()?.len(), 1);
        database.0.pending_broker_segments.lock().clear();
        assert_eq!(database.reclaim_payload_storage()?, 1);
        assert!(database.0.segments.immutable_file_names()?.is_empty());
        Ok(())
    }

    #[test]
    fn database_snapshot_keeps_broker_payload_as_verified_attachment() -> Result<()> {
        let source_directory = tempfile::tempdir()?;
        let identity = NodeIdentity::generate_genesis().public();
        let database = Database::open_backend(
            source_directory.path(),
            4 * 1024 * 1024,
            Duration::from_secs(1),
            identity,
        )?;
        let project = ProjectId::random();
        let bookmark = Bookmark { term: 3, index: 17 };
        let payload = vec![0x6d; 1024 * 1024 + 257];
        {
            let mut state = database.0.state.write();
            state.applied = bookmark;
            state.broker.apply(
                BrokerCommand::CreateTopic {
                    project,
                    name: "events".to_owned(),
                    partitions: 1,
                    retention: crate::broker::RetentionPolicy {
                        max_age_ms: None,
                        max_bytes: None,
                    },
                },
                &database.0.segments,
            )?;
            state.broker.apply(
                BrokerCommand::PublishKafkaBatch {
                    project,
                    topic: "events".to_owned(),
                    partition: 0,
                    resolved_time_ms: 10,
                    records: vec![crate::broker::KafkaBatchRecord {
                        create_time_ms: None,
                        key: None,
                        headers: BTreeMap::new(),
                        payload: payload.clone(),
                        value_is_null: false,
                    }],
                },
                &database.0.segments,
            )?;
        }

        let state_path = source_directory.path().join("database.snapshot");
        let snapshot = build_database_snapshot_file(&database, bookmark, &state_path)?;
        assert_eq!(snapshot.attachments.len(), 1);
        assert!(snapshot.state_bytes < payload.len() as u64);
        assert_eq!(
            fs::metadata(&snapshot.attachments[0].path)?.len(),
            snapshot.attachments[0].bytes
        );

        let restored_directory = tempfile::tempdir()?;
        let restored = Database::open_backend(
            restored_directory.path(),
            4 * 1024 * 1024,
            Duration::from_secs(1),
            identity,
        )?;
        let orphan = restored.0.segments.write_immutable(
            SegmentFamily::Catalog,
            None,
            &[SegmentRecord {
                kind: 99,
                payload: b"crash-leftover".to_vec(),
            }],
        )?;
        install_database_snapshot_file(&restored, bookmark, &snapshot)?;
        assert_eq!(restored.bookmark(), bookmark);
        let state = restored.0.state.read();
        state
            .broker
            .validate_payload_segments(&restored.0.segments)?;
        let fetched = state.broker.fetch_partition(
            project,
            "events",
            0,
            0,
            payload.len() + 4096,
            &restored.0.segments,
        )?;
        assert_eq!(fetched.len(), 1);
        assert_eq!(fetched[0].1.payload.as_ref(), payload.as_slice());
        drop(state);
        let live = restored.0.state.read().broker.payload_segments()[0]
            .file_name
            .clone();
        assert_ne!(live, orphan.file_name);
        assert!(
            restored
                .0
                .segments
                .immutable_file_names()?
                .contains(&orphan.file_name)
        );
        assert_eq!(restored.reclaim_payload_storage()?, 1);
        let files = restored.0.segments.immutable_file_names()?;
        assert!(files.contains(&live));
        assert!(!files.contains(&orphan.file_name));
        restored
            .0
            .state
            .read()
            .broker
            .validate_payload_segments(&restored.0.segments)?;
        Ok(())
    }

    #[tokio::test]
    async fn standalone_snapshot_recovers_live_broker_payloads() -> Result<()> {
        let source_directory = tempfile::tempdir()?;
        let snapshot_directory = tempfile::tempdir()?;
        let identity = NodeIdentity::generate_genesis().public();
        let database = Database::open_backend(
            source_directory.path(),
            4 * 1024 * 1024,
            Duration::from_secs(1),
            identity,
        )?;
        let project = ProjectId::random();
        let bookmark = Bookmark { term: 1, index: 1 };
        let payload = b"internal arrival".to_vec();
        {
            let mut state = database.0.state.write();
            state.applied = bookmark;
            state.broker.apply(
                BrokerCommand::CreateTopic {
                    project,
                    name: "channel.events".to_owned(),
                    partitions: 1,
                    retention: crate::broker::RetentionPolicy {
                        max_age_ms: None,
                        max_bytes: None,
                    },
                },
                &database.0.segments,
            )?;
            state.broker.apply(
                BrokerCommand::PublishKafkaBatch {
                    project,
                    topic: "channel.events".to_owned(),
                    partition: 0,
                    resolved_time_ms: 1,
                    records: vec![crate::broker::KafkaBatchRecord {
                        create_time_ms: None,
                        key: None,
                        headers: BTreeMap::new(),
                        payload: payload.clone(),
                        value_is_null: false,
                    }],
                },
                &database.0.segments,
            )?;
        }

        assert_eq!(
            database
                .standalone_snapshot(snapshot_directory.path())
                .await?,
            bookmark
        );

        let restored_directory = tempfile::tempdir()?;
        let restored = Database::open_backend(
            restored_directory.path(),
            4 * 1024 * 1024,
            Duration::from_secs(1),
            identity,
        )?;
        assert_eq!(
            restored
                .standalone_recover(snapshot_directory.path())
                .await?,
            Some(bookmark)
        );
        let state = restored.0.state.read();
        let records = state.broker.fetch_partition(
            project,
            "channel.events",
            0,
            0,
            4096,
            &restored.0.segments,
        )?;
        assert_eq!(records.len(), 1);
        assert_eq!(records[0].1.payload.as_ref(), payload.as_slice());
        Ok(())
    }

    #[tokio::test]
    async fn standalone_broker_publish_consumes_its_once_materialized_generation() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let database = Database::open_backend(
            directory.path(),
            2 * 1024 * 1024,
            Duration::from_secs(1),
            NodeIdentity::generate_genesis().public(),
        )?;
        let project = ProjectId::random();
        {
            let mut state = database.0.state.write();
            state.applied = Bookmark { term: 1, index: 10 };
            state
                .projects
                .insert(project, ordered_graph_project(project, 10)?);
            state.broker.apply(
                BrokerCommand::CreateTopic {
                    project,
                    name: "events".to_owned(),
                    partitions: 1,
                    retention: crate::broker::RetentionPolicy {
                        max_age_ms: None,
                        max_bytes: None,
                    },
                },
                &database.0.segments,
            )?;
        }
        let mutation = DatabaseMutation::Broker {
            command: BrokerCommand::PublishKafkaBatch {
                project,
                topic: "events".to_owned(),
                partition: 0,
                resolved_time_ms: 1,
                records: vec![crate::broker::KafkaBatchRecord {
                    create_time_ms: None,
                    key: None,
                    headers: BTreeMap::new(),
                    payload: b"materialize exactly once".to_vec(),
                    value_is_null: false,
                }],
            },
        };
        let command = WriteCommand {
            kind: MutationKind::Broker,
            project_id: Some(project),
            request_id: Some(Uuid::new_v4()),
            commit_time_millis: 1,
            payload: encode_database_value(&mutation, "broker single materialization test")?,
        };
        let committed = Bookmark { term: 2, index: 11 };
        let reservation = database.reserve_command(&command, committed).await?;
        let staged = database
            .0
            .ordered_overlay
            .lock()
            .states
            .get(&committed.index)
            .cloned()
            .ok_or_else(|| Error::internal("broker reservation did not stage its generation"))?;
        assert_eq!(staged.broker.payload_segments().len(), 1);

        let entry = MutationEntry::new(
            committed.term,
            committed.index,
            command.kind,
            command.project_id,
            command.request_id,
            command.commit_time_millis,
            command.payload.clone(),
        )?;
        let applied = database.apply_mutation(&entry).await?;
        assert!(!applied.duplicate);
        let state = database.0.state.read();
        assert!(state.broker.shared_with(&staged.broker));
        assert_eq!(state.broker.payload_segments().len(), 1);
        drop(state);
        database
            .complete_command_reservation(reservation, CommandReservationOutcome::Applied)
            .await?;
        Ok(())
    }

    #[tokio::test]
    async fn rejected_broker_reservation_releases_its_prestage_pin() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let database = Database::open_backend(
            directory.path(),
            2 * 1024 * 1024,
            Duration::from_secs(1),
            NodeIdentity::generate_genesis().public(),
        )?;
        let project = ProjectId::random();
        {
            let mut state = database.0.state.write();
            state.applied = Bookmark { term: 1, index: 10 };
            state
                .projects
                .insert(project, ordered_graph_project(project, 10)?);
            state.broker.apply(
                BrokerCommand::CreateTopic {
                    project,
                    name: "events".to_owned(),
                    partitions: 1,
                    retention: crate::broker::RetentionPolicy {
                        max_age_ms: None,
                        max_bytes: None,
                    },
                },
                &database.0.segments,
            )?;
        }
        let mutation = DatabaseMutation::Broker {
            command: BrokerCommand::PublishKafkaBatch {
                project,
                topic: "events".to_owned(),
                partition: 0,
                resolved_time_ms: 1,
                records: vec![crate::broker::KafkaBatchRecord {
                    create_time_ms: None,
                    key: None,
                    headers: BTreeMap::new(),
                    payload: b"rejected".to_vec(),
                    value_is_null: false,
                }],
            },
        };
        let command = WriteCommand {
            kind: MutationKind::Broker,
            project_id: Some(project),
            request_id: Some(Uuid::new_v4()),
            commit_time_millis: 1,
            payload: encode_database_value(&mutation, "broker reservation test")?,
        };
        let reservation = database
            .reserve_command(&command, Bookmark { term: 2, index: 11 })
            .await?;
        assert!(!reservation.may_release_after_append());
        assert_eq!(database.0.pending_broker_segments.lock().len(), 1);
        database
            .complete_command_reservation(reservation, CommandReservationOutcome::Rejected)
            .await?;
        assert!(database.0.pending_broker_segments.lock().is_empty());
        assert_eq!(database.0.retired_broker_segments.lock().len(), 1);
        Ok(())
    }

    #[test]
    fn transaction_registry_bounds_bytes_shares_pins_and_expires_without_traffic() -> Result<()> {
        let registry = TransactionAdmissionRegistry::new(1_024)?;
        registry.set_device_limit(100)?;
        let project = ProjectId::random();
        let bookmark = Bookmark { term: 4, index: 9 };
        let leader = ProcessId::random();
        let fence = TransactionFence {
            sequencer: leader,
            term: 4,
            snapshot: bookmark,
        };
        let connection = ConnectionId::new();
        let first_id = Uuid::new_v4();
        let second_id = Uuid::new_v4();
        let first = transaction_resource(project, bookmark);
        let second = transaction_resource(project, bookmark);
        let deadline = Instant::now() + Duration::from_secs(1);
        let pin = TransactionPinKey { project, bookmark };
        registry.register(first_id, connection, 256, pin, 80, deadline, fence, &first)?;
        registry.register(
            second_id, connection, 256, pin, 80, deadline, fence, &second,
        )?;
        assert_eq!(registry.usage(), (2, 512, 80));
        registry.resize(first_id, 700)?;
        let error = registry
            .resize(second_id, 700)
            .expect_err("transaction bytes exceeded the global admission limit");
        assert_eq!(error.code, ErrorCode::WriteAdmissionFull);

        registry.maintain(deadline + Duration::from_millis(1), Some((4, leader)));
        assert_eq!(registry.usage(), (0, 0, 0));
        assert!(matches!(
            *first.lifecycle.lock(),
            TransactionLifecycle::Terminal(TransactionTerminal::Expired)
        ));
        assert!(matches!(
            *second.lifecycle.lock(),
            TransactionLifecycle::Terminal(TransactionTerminal::Expired)
        ));
        Ok(())
    }

    #[test]
    fn project_capture_keeps_host_and_resident_generations_atomic_during_publication() -> Result<()>
    {
        let directory = tempfile::tempdir()?;
        let database = Database::open_backend(
            directory.path(),
            4 * 1024 * 1024,
            Duration::from_secs(1),
            NodeIdentity::generate_genesis().public(),
        )?;
        let project = ProjectId::random();
        let initial = Bookmark { term: 1, index: 1 };
        {
            let mut state = database.0.state.write();
            let mut project_state = ordered_graph_project(project, 1)?;
            Arc::make_mut(&mut project_state)
                .graph
                .apply(GraphMutation::InsertNode(crate::graph::NodeInput {
                    id: crate::NodeId(2),
                    layer: crate::Layer::Observed,
                    revision: 1,
                    labels: Vec::new(),
                    properties: Vec::new(),
                }))?;
            state.projects.insert(project, project_state);
            state.applied = initial;
        }
        database.bind_execution_backend(Box::new(crate::gpu::CpuBackend::new(
            64 * 1024 * 1024,
            1024 * 1024,
        )))?;

        let (initial_snapshot, initial_bookmark, initial_execution) =
            database.capture_project_execution(project)?;
        assert_eq!(initial_bookmark, initial);
        let initial_execution = initial_execution
            .ok_or_else(|| Error::internal("captured resident generation is absent"))?;

        let start = Arc::new(std::sync::Barrier::new(2));
        let writer_database = database.clone();
        let writer_start = Arc::clone(&start);
        let writer = std::thread::spawn(move || -> Result<()> {
            writer_start.wait();
            for index in 2_u64..=128 {
                let bookmark = Bookmark { term: 1, index };
                let mut state = writer_database.0.state.write();
                let mut staged = state.clone();
                let project_state = Arc::make_mut(
                    staged
                        .projects
                        .get_mut(&project)
                        .ok_or_else(|| Error::internal("test project disappeared"))?,
                );
                let property = project_state
                    .graph
                    .catalog()
                    .property("value")
                    .ok_or_else(|| Error::internal("test property disappeared"))?;
                let mutation = GraphMutation::SetNodeProperty {
                    node: crate::NodeId(1),
                    property,
                    value: ScalarValue::Integer(index as i64),
                    revision: index,
                };
                project_state
                    .indexes
                    .before_graph_apply(&project_state.graph, &mutation)?;
                project_state.graph.apply(mutation.clone())?;
                project_state
                    .indexes
                    .after_graph_apply(&project_state.graph, &mutation)?;
                staged.applied = bookmark;
                {
                    let mut execution = writer_database.0.execution.write();
                    publish_execution_state(
                        &mut execution,
                        &mut staged,
                        DeviceImpact::Project {
                            project,
                            temporal: Vec::new(),
                            vectors: Vec::new(),
                            invalidate_derived: true,
                        },
                        bookmark,
                    )?;
                }
                *state = staged;
                std::thread::yield_now();
            }
            Ok(())
        });

        start.wait();
        for _ in 0..512 {
            let (snapshot, bookmark, execution) = database.capture_project_execution(project)?;
            let execution = execution
                .ok_or_else(|| Error::internal("captured resident generation is absent"))?;
            assert_eq!(execution.resident_bookmark(project), Some(bookmark));
            assert_eq!(
                execution.resident_graph_revision(project),
                Some(snapshot.graph.revision())
            );
            std::thread::yield_now();
        }
        writer
            .join()
            .map_err(|_| Error::internal("generation publication thread panicked"))??;

        assert_eq!(initial_execution.resident_bookmark(project), Some(initial));
        assert_eq!(
            initial_execution.resident_graph_revision(project),
            Some(initial_snapshot.graph.revision())
        );
        assert_eq!(initial_snapshot.graph.revision(), 1);
        assert_eq!(database.bookmark().index, 128);
        Ok(())
    }

    #[test]
    fn transaction_overlay_advances_its_pinned_resident_graph_without_advancing_snapshot_bookmark()
    -> Result<()> {
        let project = ProjectId::random();
        let bookmark = Bookmark { term: 3, index: 7 };
        let snapshot = ordered_graph_project(project, bookmark.index)?;
        let mut backend = crate::gpu::CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
        backend.admit_project(ResidentProjectImage::build(
            project,
            bookmark,
            &snapshot.graph,
            &snapshot.temporal,
            &snapshot.indexes,
        )?)?;
        let pinned = backend.pin_project(project)?;

        let label = crate::types::LabelId(snapshot.graph.catalog().next_label_id());
        let mutations = vec![
            GraphMutation::DeclareLabel {
                name: "Pending".to_owned(),
                id: label,
            },
            GraphMutation::InsertNode(crate::graph::NodeInput {
                id: crate::NodeId(2),
                layer: crate::Layer::Observed,
                revision: bookmark.index + 1,
                labels: vec![label],
                properties: Vec::new(),
            }),
        ];
        let staged = stage_transaction_project(&snapshot, &mutations, &[], &[], 0)?;
        let staged_execution = stage_transaction_execution(
            Some(pinned.as_ref()),
            project,
            bookmark,
            &staged,
            true,
            &[],
            &[],
        )?
        .ok_or_else(|| Error::internal("transaction resident generation is absent"))?;

        let cancellation = tokio_util::sync::CancellationToken::new();
        assert_eq!(
            pinned
                .scan_nodes(
                    project,
                    None,
                    crate::graph::LayerMask::default(),
                    &cancellation
                )?
                .len(),
            1
        );
        assert_eq!(
            staged_execution
                .scan_nodes(
                    project,
                    None,
                    crate::graph::LayerMask::default(),
                    &cancellation
                )?
                .len(),
            2
        );
        assert_eq!(staged_execution.resident_bookmark(project), Some(bookmark));
        assert_eq!(
            staged_execution.resident_graph_revision(project),
            Some(staged.graph.revision())
        );
        assert_eq!(
            pinned.resident_graph_revision(project),
            Some(bookmark.index)
        );
        Ok(())
    }

    #[test]
    fn show_projects_reports_the_bookmark_of_the_captured_catalog() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let database = Database::open_backend(
            directory.path(),
            4 * 1024 * 1024,
            Duration::from_secs(1),
            NodeIdentity::generate_genesis().public(),
        )?;
        let project = ProjectId::random();
        let captured = Bookmark { term: 2, index: 11 };
        {
            let mut state = database.0.state.write();
            state
                .projects
                .insert(project, ordered_graph_project(project, 11)?);
            state.applied = captured;
        }
        let request = Uuid::new_v4();
        let mut events = Vec::new();
        database.show_projects(
            request,
            CommitAcknowledgement::Published,
            Bookmark { term: 2, index: 10 },
            &mut |event| {
                events.push(event);
                Ok(())
            },
        )?;
        assert!(events.iter().any(|event| {
            matches!(event, QueryStreamEvent::Summary { bookmark, .. } if *bookmark == captured)
        }));
        assert!(
            events
                .iter()
                .any(|event| { matches!(event, QueryStreamEvent::Batch { row_count: 1, .. }) })
        );
        Ok(())
    }

    #[test]
    fn check_read_only_validates_candidate_against_the_selected_project() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let database = Database::open_backend(
            directory.path(),
            4 * 1024 * 1024,
            Duration::from_secs(1),
            NodeIdentity::generate_genesis().public(),
        )?;
        let project = ProjectId::random();
        {
            let mut state = database.0.state.write();
            state
                .projects
                .insert(project, ordered_graph_project(project, 1)?);
            state.applied = Bookmark { term: 1, index: 1 };
        }
        let request = |statement: Option<&str>| QueryRequest {
            request_id: Uuid::new_v4(),
            project_id: Some(project),
            query: "CHECK READ ONLY".to_owned(),
            parameters: statement
                .map(|statement| {
                    BTreeMap::from([("statement".to_owned(), serde_json::json!(statement))])
                })
                .unwrap_or_default(),
            consistency: CommitAcknowledgement::Published,
            bookmark: None,
            limits: QueryLimits::default(),
            cancellation: Default::default(),
            deadline: None,
            connection_id: ConnectionId::new(),
        };

        let mut events = Vec::new();
        database.execute(request(Some("RETURN 1")), &mut |event| {
            events.push(event);
            Ok(())
        })?;
        assert!(
            events
                .iter()
                .any(|event| matches!(event, QueryStreamEvent::Summary { .. }))
        );
        assert!(
            database
                .execute(request(Some("CREATE (:Item)")), &mut |_| Ok(()))
                .is_err()
        );
        assert!(database.execute(request(None), &mut |_| Ok(())).is_err());
        Ok(())
    }

    #[test]
    fn transaction_registry_fences_sequencer_change_and_releases_device_pin() -> Result<()> {
        let registry = TransactionAdmissionRegistry::new(1_024)?;
        registry.set_device_limit(64)?;
        let project = ProjectId::random();
        let bookmark = Bookmark { term: 2, index: 3 };
        let leader = ProcessId::random();
        let resource = transaction_resource(project, bookmark);
        registry.register(
            Uuid::new_v4(),
            ConnectionId::new(),
            256,
            TransactionPinKey { project, bookmark },
            64,
            Instant::now() + Duration::from_secs(1),
            TransactionFence {
                sequencer: leader,
                term: 2,
                snapshot: bookmark,
            },
            &resource,
        )?;
        registry.maintain(Instant::now(), Some((3, leader)));
        assert_eq!(registry.usage(), (0, 0, 0));
        assert!(matches!(
            *resource.lifecycle.lock(),
            TransactionLifecycle::Terminal(TransactionTerminal::SequencerChanged)
        ));
        Ok(())
    }

    #[test]
    fn transaction_registry_enforces_per_connection_limit_under_concurrency() -> Result<()> {
        let registry = Arc::new(TransactionAdmissionRegistry::new(
            (MAX_OPEN_TRANSACTIONS_PER_CONNECTION + 1) * MIN_TRANSACTION_ACCOUNTED_BYTES,
        )?);
        registry.set_device_limit(1)?;
        let connection = ConnectionId::new();
        let project = ProjectId::random();
        let bookmark = Bookmark { term: 1, index: 1 };
        let leader = ProcessId::random();
        let barrier = Arc::new(std::sync::Barrier::new(
            MAX_OPEN_TRANSACTIONS_PER_CONNECTION * 2,
        ));
        let handles = (0..MAX_OPEN_TRANSACTIONS_PER_CONNECTION * 2)
            .map(|_| {
                let registry = Arc::clone(&registry);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    let resource = transaction_resource(project, bookmark);
                    let id = Uuid::new_v4();
                    barrier.wait();
                    registry
                        .register(
                            id,
                            connection,
                            MIN_TRANSACTION_ACCOUNTED_BYTES,
                            TransactionPinKey { project, bookmark },
                            1,
                            Instant::now() + Duration::from_secs(1),
                            TransactionFence {
                                sequencer: leader,
                                term: 1,
                                snapshot: bookmark,
                            },
                            &resource,
                        )
                        .map(|_| id)
                })
            })
            .collect::<Vec<_>>();
        let results = handles
            .into_iter()
            .map(|handle| handle.join().expect("admission thread panicked"))
            .collect::<Vec<_>>();
        let accepted = results.iter().filter(|result| result.is_ok()).count();
        assert_eq!(accepted, MAX_OPEN_TRANSACTIONS_PER_CONNECTION);
        assert!(
            results
                .iter()
                .filter_map(|result| result.as_ref().err())
                .all(|error| error.code == ErrorCode::WriteAdmissionFull)
        );
        for id in results.into_iter().filter_map(std::result::Result::ok) {
            registry.finish(id);
        }
        assert_eq!(registry.usage(), (0, 0, 0));
        Ok(())
    }

    #[test]
    fn request_id_intent_binds_semantics_but_not_sequencer_time() -> Result<()> {
        let project = ProjectId::random();
        let mutation = DatabaseMutation::Broker {
            command: BrokerCommand::PublishAmqp {
                project,
                exchange: "events".to_owned(),
                routing_key: "created".to_owned(),
                mandatory: true,
                resolved_time_ms: 1,
                properties: BTreeMap::new(),
                headers: BTreeMap::new(),
                payload: b"first".to_vec(),
            },
        };
        let mut later = mutation.clone();
        resolve_sequencer_values(&mut later, 9_999)?;
        assert_eq!(
            request_intent_digest(&mutation)?,
            request_intent_digest(&later)?
        );

        let DatabaseMutation::Broker {
            command: BrokerCommand::PublishAmqp { payload, .. },
        } = &mut later
        else {
            return Err(Error::internal("test mutation changed variant"));
        };
        *payload = b"different".to_vec();
        assert_ne!(
            request_intent_digest(&mutation)?,
            request_intent_digest(&later)?
        );
        Ok(())
    }

    #[test]
    fn bounded_mutation_encoding_fails_before_allocating_past_admission() -> Result<()> {
        let error =
            encode_database_value_bounded(&vec![7_u8; 4_096], "oversized test mutation", 128)
                .expect_err("oversized mutation encoding unexpectedly succeeded");
        assert_eq!(error.code, ErrorCode::WriteAdmissionFull);
        let encoded = encode_database_value_bounded(&vec![7_u8; 32], "bounded test mutation", 128)?;
        assert!(encoded.len() <= 128);
        Ok(())
    }

    #[test]
    fn project_generation_detaches_only_mutated_subsystem_root() -> Result<()> {
        let project = ProjectId::random();
        let published = ProjectState {
            id: project,
            display_name: "cow".to_owned(),
            graph: GraphStore::default().into(),
            temporal: TemporalStore::default().into(),
            predicate_versions: BTreeMap::new().into(),
            indexes: IndexCatalog::default().into(),
            next_node_id: 1,
            next_edge_id: 1,
            authority_revision: 0,
            optimizer_statistics: OnceLock::new(),
        };
        let mut staged = published.clone();
        assert!(published.graph.shared_with(&staged.graph));
        assert!(published.temporal.shared_with(&staged.temporal));
        assert!(published.indexes.shared_with(&staged.indexes));
        assert!(
            published
                .predicate_versions
                .shared_with(&staged.predicate_versions)
        );
        staged.graph.catalog_mut().intern_label("OnlyStaged")?;
        assert!(!published.graph.shared_with(&staged.graph));
        assert!(published.temporal.shared_with(&staged.temporal));
        assert!(published.indexes.shared_with(&staged.indexes));
        assert!(
            published
                .predicate_versions
                .shared_with(&staged.predicate_versions)
        );
        assert!(published.graph.catalog().label("OnlyStaged").is_none());
        Ok(())
    }

    #[test]
    fn failed_staged_graph_batch_cannot_mutate_published_generation() -> Result<()> {
        let project = ProjectId::random();
        let mut published = DatabaseState::default();
        apply_mutation(
            &mut published,
            DatabaseMutation::CreateProject {
                id: project,
                display_name: "atomic".to_owned(),
            },
            Bookmark { term: 1, index: 1 },
            1,
            None,
        )?;
        let mut staged = published.clone();
        let graph = staged
            .projects
            .get_mut(&project)
            .ok_or_else(|| Error::internal("staged project missing"))?;
        let graph = Arc::make_mut(graph);
        let label = graph.graph.catalog_mut().intern_label("Row")?;
        graph.graph.insert_node(crate::graph::NodeInput {
            id: crate::NodeId(1),
            layer: crate::Layer::Observed,
            revision: 2,
            labels: vec![label],
            properties: Vec::new(),
        })?;
        let error = graph
            .graph
            .insert_node(crate::graph::NodeInput {
                id: crate::NodeId(1),
                layer: crate::Layer::Observed,
                revision: 2,
                labels: vec![label],
                properties: Vec::new(),
            })
            .expect_err("duplicate staged insert must fail");
        assert_eq!(error.code, ErrorCode::TransactionConflict);
        let canonical = published
            .projects
            .get(&project)
            .ok_or_else(|| Error::internal("published project missing"))?;
        assert_eq!(canonical.graph.node_count(), 0);
        Ok(())
    }

    #[test]
    fn non_cascade_drop_rejects_data_and_cascade_removes_it() -> Result<()> {
        let project = ProjectId::random();
        let mut state = DatabaseState::default();
        let bookmark = Bookmark { term: 1, index: 1 };
        apply_mutation(
            &mut state,
            DatabaseMutation::CreateProject {
                id: project,
                display_name: "project".to_owned(),
            },
            bookmark,
            1,
            None,
        )?;
        let empty_drop = DatabaseMutation::DropProject {
            id: project,
            cascade: false,
        };
        validate_sequencer_mutation(&state, &empty_drop, bookmark, 1)?;

        let project_state = Arc::make_mut(
            state
                .projects
                .get_mut(&project)
                .ok_or_else(|| Error::internal("project missing"))?,
        );
        let label = project_state.graph.catalog_mut().intern_label("Data")?;
        project_state
            .graph
            .apply(GraphMutation::InsertNode(crate::graph::NodeInput {
                id: crate::NodeId(1),
                layer: crate::Layer::Observed,
                revision: 2,
                labels: vec![label],
                properties: Vec::new(),
            }))?;
        let error = validate_sequencer_mutation(
            &state,
            &DatabaseMutation::DropProject {
                id: project,
                cascade: false,
            },
            Bookmark { term: 1, index: 2 },
            2,
        )
        .expect_err("non-cascade drop must reject project data");
        assert_eq!(error.code, ErrorCode::TransactionConflict);

        let cascade = DatabaseMutation::DropProject {
            id: project,
            cascade: true,
        };
        validate_sequencer_mutation(&state, &cascade, Bookmark { term: 1, index: 2 }, 2)?;
        apply_mutation(&mut state, cascade, Bookmark { term: 1, index: 2 }, 2, None)?;
        assert!(!state.projects.contains_key(&project));
        assert!(!state.names.contains_key("project"));
        Ok(())
    }

    #[test]
    fn optimizer_statistics_are_boundedly_stale_not_rebuilt_per_write() -> Result<()> {
        let project = ProjectId::random();
        let mut project_state = ProjectState {
            id: project,
            display_name: "statistics".to_owned(),
            graph: GraphStore::default().into(),
            temporal: TemporalStore::default().into(),
            predicate_versions: BTreeMap::new().into(),
            indexes: IndexCatalog::default().into(),
            next_node_id: 1,
            next_edge_id: 1,
            authority_revision: 0,
            optimizer_statistics: OnceLock::new(),
        };
        project_state
            .optimizer_statistics
            .set(Arc::new(StatisticsSnapshot::collect(&project_state.graph)))
            .map_err(|_| Error::internal("statistics cache already initialized"))?;
        let label = project_state.graph.catalog_mut().intern_label("Data")?;
        project_state
            .graph
            .apply(GraphMutation::InsertNode(crate::graph::NodeInput {
                id: crate::NodeId(1),
                layer: crate::Layer::Observed,
                revision: 1,
                labels: vec![label],
                properties: Vec::new(),
            }))?;
        expire_optimizer_statistics_if_stale(&mut project_state);
        assert!(project_state.optimizer_statistics.get().is_none());

        project_state
            .optimizer_statistics
            .set(Arc::new(StatisticsSnapshot::collect(&project_state.graph)))
            .map_err(|_| Error::internal("statistics cache already initialized"))?;
        project_state
            .graph
            .apply(GraphMutation::InsertNode(crate::graph::NodeInput {
                id: crate::NodeId(2),
                layer: crate::Layer::Observed,
                revision: 2,
                labels: vec![label],
                properties: Vec::new(),
            }))?;
        expire_optimizer_statistics_if_stale(&mut project_state);
        assert!(project_state.optimizer_statistics.get().is_some());
        Ok(())
    }

    #[test]
    fn automatic_storage_seal_preserves_pinned_project() -> Result<()> {
        let project = ProjectId::random();
        let mut state = DatabaseState::default();
        apply_mutation(
            &mut state,
            DatabaseMutation::CreateProject {
                id: project,
                display_name: "pinned".to_owned(),
            },
            Bookmark { term: 1, index: 1 },
            1,
            None,
        )?;
        let original = ScalarValue::List(DocumentList::new(vec![DocumentItem::Scalar(
            ScalarValue::String(Arc::from("a".repeat(1_024))),
        )])?);
        let replacement = ScalarValue::List(DocumentList::new(vec![DocumentItem::Scalar(
            ScalarValue::String(Arc::from("b".repeat(1_024))),
        )])?);
        let (label, property) = {
            let project_state = Arc::make_mut(
                state
                    .projects
                    .get_mut(&project)
                    .ok_or_else(|| Error::internal("test project is missing"))?,
            );
            let label = project_state.graph.catalog_mut().intern_label("Document")?;
            let property = project_state
                .graph
                .catalog_mut()
                .intern_property("payload")?;
            project_state
                .graph
                .apply(GraphMutation::InsertNode(crate::graph::NodeInput {
                    id: crate::NodeId(1),
                    layer: crate::Layer::Observed,
                    revision: 2,
                    labels: vec![label],
                    properties: vec![(property, original.clone())],
                }))?;
            (label, property)
        };
        let pinned_project = Arc::clone(
            state
                .projects
                .get(&project)
                .ok_or_else(|| Error::internal("test project is missing"))?,
        );

        let mut staged = state.clone();
        let staged_project = Arc::make_mut(
            staged
                .projects
                .get_mut(&project)
                .ok_or_else(|| Error::internal("staged project is missing"))?,
        );
        for revision in 3..=96 {
            staged_project.graph.apply(GraphMutation::SetNodeProperty {
                node: crate::NodeId(1),
                property,
                value: replacement.clone(),
                revision,
            })?;
        }
        assert_eq!(
            pinned_project.graph.catalog().label("Document"),
            Some(label)
        );
        assert_eq!(
            pinned_project
                .graph
                .node(crate::NodeId(1))
                .and_then(|node| node.property(property)),
            Some(original)
        );
        assert_eq!(
            staged_project
                .graph
                .node(crate::NodeId(1))
                .and_then(|node| node.property(property)),
            Some(replacement)
        );
        let mut checkpoint = Vec::new();
        ciborium::ser::into_writer(&staged, &mut checkpoint)
            .map_err(|error| Error::internal(format!("test encoding failed: {error}")))?;
        let restored: DatabaseState = ciborium::de::from_reader(checkpoint.as_slice())
            .map_err(|error| Error::internal(format!("test decoding failed: {error}")))?;
        assert_eq!(
            restored
                .projects
                .get(&project)
                .and_then(|project| project.graph.node(crate::NodeId(1)))
                .and_then(|node| node.property(property)),
            Some(ScalarValue::List(DocumentList::new(vec![
                DocumentItem::Scalar(ScalarValue::String(Arc::from("b".repeat(1_024)))),
            ])?))
        );
        Ok(())
    }

    #[test]
    fn embedding_profile_activation_requires_local_readiness() -> Result<()> {
        let project = ProjectId::random();
        let profile = crate::graph::EmbeddingProfile::new(
            [7; 32],
            [8; 32],
            384,
            crate::graph::EmbeddingDType::F16,
            true,
            crate::graph::Similarity::Cosine,
        )?;
        let mut state = DatabaseState::default();
        apply_mutation(
            &mut state,
            DatabaseMutation::CreateProject {
                id: project,
                display_name: "vectors".to_owned(),
            },
            Bookmark { term: 1, index: 1 },
            1,
            None,
        )?;
        let project_state = Arc::make_mut(
            state
                .projects
                .get_mut(&project)
                .ok_or_else(|| Error::internal("project disappeared"))?,
        );
        let label = project_state.graph.catalog_mut().intern_label("Document")?;
        project_state.graph.insert_node(crate::graph::NodeInput {
            id: crate::NodeId(1),
            layer: crate::Layer::Observed,
            revision: 2,
            labels: vec![label],
            properties: Vec::new(),
        })?;

        apply_embedding_profile_command(
            &mut state,
            &EmbeddingProfileCommand::Begin {
                project,
                profile: profile.clone(),
            },
        )?;
        let error = validate_embedding_profile_command(
            &state,
            &EmbeddingProfileCommand::Activate {
                project,
                profile_hash: profile.profile_hash,
            },
        )
        .expect_err("activation requires local readiness");
        assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);

        apply_embedding_profile_command(
            &mut state,
            &EmbeddingProfileCommand::Acknowledge {
                project,
                profile_hash: profile.profile_hash,
            },
        )?;
        apply_embedding_profile_command(
            &mut state,
            &EmbeddingProfileCommand::Activate {
                project,
                profile_hash: profile.profile_hash,
            },
        )?;
        assert_eq!(
            state
                .projects
                .get(&project)
                .and_then(|project| project.indexes.profile()),
            Some(&profile)
        );
        Ok(())
    }

    #[test]
    fn document_embedding_delta_batches_complete_text_without_scanning_unrelated_rows() -> Result<()>
    {
        let project_id = ProjectId::random();
        let mut project = Arc::unwrap_or_clone(ordered_graph_project(project_id, 1)?);
        let document = project.graph.catalog_mut().intern_label("Document")?;
        let unrelated = project.graph.catalog_mut().intern_label("Unrelated")?;
        let body = project.graph.catalog_mut().intern_property("body")?;
        let vector = project.graph.catalog_mut().intern_property("embedding")?;
        for id in 1_000..1_400 {
            project.graph.insert_node(crate::graph::NodeInput {
                id: crate::NodeId(id),
                layer: crate::Layer::Observed,
                revision: id,
                labels: vec![unrelated],
                properties: vec![(
                    body,
                    ScalarValue::String(Arc::from("unrelated ".repeat(4_096))),
                )],
            })?;
        }
        let profile = crate::graph::EmbeddingProfile::new(
            [7; 32],
            [8; 32],
            2,
            crate::graph::EmbeddingDType::F16,
            true,
            crate::graph::Similarity::Cosine,
        )?;
        project.indexes.create_embedding_deferred(
            &project.graph,
            crate::graph::EmbeddingIndexDefinition {
                name: "documents".to_owned(),
                label: document,
                source_property: body,
                target_property: vector,
                model: "default".to_owned(),
            },
            profile.clone(),
            Vec::new(),
        )?;
        let batches = Arc::new(Mutex::new(Vec::new()));
        let embedding = WindowEmbedding {
            profile,
            batches: Arc::clone(&batches),
        };
        let mutation = GraphMutation::InsertNode(crate::graph::NodeInput {
            id: crate::NodeId(2_000),
            layer: crate::Layer::Knowledge,
            revision: 2_000,
            labels: vec![document],
            properties: vec![(body, ScalarValue::String(Arc::from("head interior tail")))],
        });

        let resolved =
            resolve_embedding_mutations(&project, &[], &[mutation], Some(&embedding), 2_001)?;
        assert_eq!(batches.lock().as_slice(), &[vec!["head", "tail"]]);
        let [
            ResolvedVectorMutation::Upsert {
                entity_id,
                coordinates,
                ..
            },
        ] = resolved.as_slice()
        else {
            return Err(Error::internal(
                "document delta did not produce one vector row",
            ));
        };
        assert_eq!(*entity_id, 2_000);
        assert_eq!(coordinates.len(), 2);
        assert!(coordinates.iter().all(|coordinate| *coordinate != 0));
        Ok(())
    }

    fn wal_test_project_entry(index: u64) -> Result<(ProjectId, MutationEntry)> {
        let id = ProjectId::random();
        let payload = encode_database_value(
            &DatabaseMutation::CreateProject {
                id,
                display_name: format!("wal-project-{index}"),
            },
            "wal test create project",
        )?;
        let entry = MutationEntry::new(
            1,
            index,
            MutationKind::Project,
            Some(id),
            Some(Uuid::new_v4()),
            1,
            payload,
        )?;
        Ok((id, entry))
    }

    #[tokio::test]
    async fn standalone_snapshot_recovers_committed_writes_after_restart() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let snapshot_dir = directory.path().join("standalone-snapshots");
        // The same node identity across both lifetimes — a snapshot is bound to its store ID.
        let identity = NodeIdentity::generate_genesis().public();
        let entries = (1..=3)
            .map(wal_test_project_entry)
            .collect::<Result<Vec<_>>>()?;

        // First lifetime: commit three project creates, take a periodic-style snapshot, then drop.
        {
            let database = Database::open_backend(
                directory.path(),
                2 * 1024 * 1024,
                Duration::from_secs(1),
                identity,
            )?;
            for (_, entry) in &entries {
                database.apply_mutation(entry).await?;
            }
            assert_eq!(database.bookmark().index, 3);
            database.standalone_snapshot(&snapshot_dir).await?;
        }

        // Second lifetime: a fresh backend recovers from the snapshot alone — no journal.
        let database = Database::open_backend(
            directory.path(),
            2 * 1024 * 1024,
            Duration::from_secs(1),
            identity,
        )?;
        assert_eq!(database.bookmark().index, 0);
        let recovered = database.standalone_recover(&snapshot_dir).await?;
        assert_eq!(recovered.map(|bookmark| bookmark.index), Some(3));
        assert_eq!(database.bookmark().index, 3);
        let state = database.0.state.read();
        for (id, _) in &entries {
            assert!(
                state.projects.contains_key(id),
                "committed project {id:?} was not recovered from the snapshot"
            );
        }
        Ok(())
    }

    #[test]
    fn republish_resident_project_restores_a_stale_image_off_the_write_path() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let database = Database::open_backend(
            directory.path(),
            64 * 1024 * 1024,
            Duration::from_secs(1),
            NodeIdentity::generate_genesis().public(),
        )?;
        let project = ProjectId::random();
        let initial = Bookmark { term: 1, index: 1 };
        {
            let mut state = database.0.state.write();
            let mut project_state = ordered_graph_project(project, 1)?;
            Arc::make_mut(&mut project_state)
                .graph
                .apply(GraphMutation::InsertNode(crate::graph::NodeInput {
                    id: crate::NodeId(2),
                    layer: crate::Layer::Observed,
                    revision: 1,
                    labels: Vec::new(),
                    properties: Vec::new(),
                }))?;
            state.projects.insert(project, project_state);
            state.applied = initial;
        }
        database.bind_execution_backend(Box::new(crate::gpu::CpuBackend::new(
            64 * 1024 * 1024,
            1024 * 1024,
        )))?;
        // Publish the initial image so the resident is fresh at revision 1 / bookmark 1.
        {
            let mut state = database.0.state.write();
            let mut staged = state.clone();
            let mut execution = database.0.execution.write();
            publish_execution_state(
                &mut execution,
                &mut staged,
                DeviceImpact::Create(project),
                initial,
            )?;
            *state = staged;
        }
        assert!(
            database.capture_project_execution(project)?.2.is_some(),
            "resident image should be fresh immediately after publication"
        );

        // Simulate a deferred write: mutate the host graph and advance the applied bookmark WITHOUT
        // publishing to the device, exactly as the write path does for a graph past the ceiling.
        let advanced = Bookmark { term: 1, index: 2 };
        let new_revision = {
            let mut state = database.0.state.write();
            let project_state = Arc::make_mut(
                state
                    .projects
                    .get_mut(&project)
                    .ok_or_else(|| Error::internal("test project disappeared"))?,
            );
            let property = project_state
                .graph
                .catalog()
                .property("value")
                .ok_or_else(|| Error::internal("test property disappeared"))?;
            let mutation = GraphMutation::SetNodeProperty {
                node: crate::NodeId(1),
                property,
                value: ScalarValue::Integer(42),
                revision: 2,
            };
            project_state
                .indexes
                .before_graph_apply(&project_state.graph, &mutation)?;
            project_state.graph.apply(mutation.clone())?;
            project_state
                .indexes
                .after_graph_apply(&project_state.graph, &mutation)?;
            let revision = project_state.graph.revision();
            state.applied = advanced;
            revision
        };

        // The resident image now lags, so reads route to the host (no pinned device generation).
        assert!(
            database.capture_project_execution(project)?.2.is_none(),
            "a lagging resident image must route reads to the host"
        );

        // Republish off the write path: rebuild and reinstall the image, then the device serves reads
        // again at the advanced bookmark.
        assert!(database.republish_resident_project(project, new_revision)?);
        let (snapshot, bookmark, execution) = database.capture_project_execution(project)?;
        assert_eq!(bookmark, advanced);
        let execution = execution
            .ok_or_else(|| Error::internal("resident image is still stale after republish"))?;
        assert_eq!(execution.resident_bookmark(project), Some(advanced));
        assert_eq!(
            execution.resident_graph_revision(project),
            Some(snapshot.graph.revision())
        );
        Ok(())
    }

    #[test]
    fn device_publication_deferral_tracks_changed_rows_not_graph_size() -> Result<()> {
        fn deferred_with_unrelated_rows(unrelated: u64, changed: u64) -> Result<bool> {
            let mut graph = GraphStore::default();
            for id in 1..=unrelated {
                graph.insert_node(NodeInput {
                    id: NodeId(id),
                    layer: Layer::Observed,
                    revision: 1,
                    labels: Vec::new(),
                    properties: Vec::new(),
                })?;
            }
            let revision = 2;
            for offset in 0..changed {
                graph.insert_node(NodeInput {
                    id: NodeId(unrelated + offset + 1),
                    layer: Layer::Observed,
                    revision,
                    labels: Vec::new(),
                    properties: Vec::new(),
                })?;
            }
            Ok(should_defer_device_publish(&graph))
        }

        assert!(!deferred_with_unrelated_rows(40, 1)?);
        assert!(!deferred_with_unrelated_rows(400, 1)?);
        assert!(deferred_with_unrelated_rows(40, 2)?);
        assert!(deferred_with_unrelated_rows(400, 2)?);
        Ok(())
    }

    #[test]
    fn prewarm_optimizer_statistics_populates_the_cache_off_the_query_path() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let database = Database::open_backend(
            directory.path(),
            64 * 1024 * 1024,
            Duration::from_secs(1),
            NodeIdentity::generate_genesis().public(),
        )?;
        let project = ProjectId::random();
        {
            let mut state = database.0.state.write();
            state
                .projects
                .insert(project, ordered_graph_project(project, 1)?);
            state.applied = Bookmark { term: 1, index: 1 };
        }
        let revision = {
            let state = database.0.state.read();
            let project_state = state
                .projects
                .get(&project)
                .ok_or_else(|| Error::internal("test project disappeared"))?;
            // The cache starts cold.
            assert!(project_state.optimizer_statistics.get().is_none());
            project_state.graph.revision()
        };

        // Pre-warming computes and installs the snapshot into the live project's OnceLock.
        assert!(database.prewarm_optimizer_statistics(project, revision));
        {
            let state = database.0.state.read();
            let project_state = state
                .projects
                .get(&project)
                .ok_or_else(|| Error::internal("test project disappeared"))?;
            assert!(
                project_state.optimizer_statistics.get().is_some(),
                "statistics cache should be warm after pre-warming"
            );
        }

        // A second pre-warm at the same revision is a no-op (already warm).
        assert!(!database.prewarm_optimizer_statistics(project, revision));
        // A pre-warm against a stale revision does nothing.
        assert!(!database.prewarm_optimizer_statistics(project, revision.saturating_add(1)));
        Ok(())
    }
}
