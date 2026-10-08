use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File, OpenOptions},
    io::{BufReader, BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::{
        Arc, OnceLock, Weak,
        atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use arc_swap::{ArcSwap, ArcSwapOption};
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
    graph::{
        GraphMutation, GraphStore, IndexCatalog, ResolvedVectorMutation, StatisticsSnapshot,
        TemporalStore,
    },
    protocol::{
        BatchColumn, CatalogEvent, PathValue, QueryColumn, QueryExecutor, QueryRequest,
        QueryResultStatistics, QueryStatistics, QueryStreamEvent, QueryTransaction,
        RelationshipValue, ResultNode, TypedValue,
    },
    storage::{
        AdmissionClass, AdmissionController, ConnectionId, MutationEntry, MutationKind,
        SegmentDescriptor, SegmentPin, SegmentStore,
    },
};
// Durability is the runtime-owned WAL plus this backend's periodic self-contained snapshot.

const DATABASE_SNAPSHOT_FORMAT: u16 = 6;
const DATABASE_SNAPSHOT_MAGIC: [u8; 8] = *b"IGDBS006";
const DATABASE_SNAPSHOT_COPY_BYTES: usize = 1024 * 1024;
const REQUEST_RESULT_INDEX_WINDOW: u64 = 65_536;
const MAX_REQUEST_RESULTS: usize = 16_384;
const MAX_REQUEST_RESULT_BYTES: u64 = 64 * 1024 * 1024;
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
/// Cold projects above this size may collect empirical cost samples in a background warmup.
/// Ordinary writes maintain current counts and never queue a periodic full collection.
const OPTIMIZER_STATISTICS_BACKGROUND_ROWS: usize = 100_000;
const BROKER_RECLAIM_BATCH_SEGMENTS: usize = 256;
const QUERY_STREAM_BATCH_ROWS: usize = 4_096;

thread_local! {
    /// Embedding callbacks run synchronously inside a dedicated blocking worker.
    static DATABASE_BLOCKING_WORKER: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

struct DatabaseBlockingWorker(bool);

impl DatabaseBlockingWorker {
    fn enter() -> Self {
        Self(DATABASE_BLOCKING_WORKER.with(|context| context.replace(true)))
    }
}

impl Drop for DatabaseBlockingWorker {
    fn drop(&mut self) {
        DATABASE_BLOCKING_WORKER.with(|context| context.set(self.0));
    }
}

/// CPU execution's row address-space bound stays below arithmetic overflow; requests separately
/// enforce their output limits and cancellation. Query scratch stays in host memory.
const HOST_QUERY_EXECUTION_ROW_ADDRESS_SPACE: usize = u32::MAX as usize;

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
    references: usize,
}

struct OpenTransactionRecord {
    connection: ConnectionId,
    encoded_bytes: usize,
    pin: TransactionPinKey,
    deadline: Option<Instant>,
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
}

struct TransactionAdmissionRegistry {
    max_encoded_bytes: Option<usize>,
    state: Mutex<TransactionAdmissionState>,
}

impl TransactionAdmissionRegistry {
    #[cfg(test)]
    fn new(max_encoded_bytes: usize) -> Result<Self> {
        if max_encoded_bytes < MIN_TRANSACTION_ACCOUNTED_BYTES {
            return Err(Error::invalid_data(
                "explicit-transaction admission byte limit is too small",
            ));
        }
        Ok(Self {
            max_encoded_bytes: Some(max_encoded_bytes),
            state: Mutex::new(TransactionAdmissionState {
                ..TransactionAdmissionState::default()
            }),
        })
    }

    const fn maximum_encoded_bytes(&self) -> usize {
        match self.max_encoded_bytes {
            Some(bytes) => bytes,
            None => usize::MAX,
        }
    }

    fn unrestricted() -> Self {
        Self {
            max_encoded_bytes: None,
            state: Mutex::new(TransactionAdmissionState::default()),
        }
    }

    fn register(
        &self,
        id: Uuid,
        connection: ConnectionId,
        encoded_bytes: usize,
        pin: TransactionPinKey,
        deadline: impl Into<Option<Instant>>,
        fence: TransactionFence,
        resource: &Arc<TransactionResource>,
    ) -> Result<()> {
        let deadline = deadline.into();
        if !connection.is_valid()
            || encoded_bytes < MIN_TRANSACTION_ACCOUNTED_BYTES
            || deadline.is_some_and(|deadline| deadline <= Instant::now())
        {
            return Err(Error::invalid_data(
                "invalid explicit-transaction admission request",
            ));
        }
        let mut state = self.state.lock();
        if state.records.contains_key(&id) {
            return Err(Error::internal("duplicate explicit transaction identity"));
        }
        if self.max_encoded_bytes.is_some()
            && (state.records.len() >= MAX_OPEN_TRANSACTIONS
                || state
                    .connection_counts
                    .get(&connection)
                    .copied()
                    .unwrap_or_default()
                    >= MAX_OPEN_TRANSACTIONS_PER_CONNECTION)
        {
            return Err(transaction_admission_full(
                "explicit-transaction count admission is full",
            ));
        }
        let next_encoded = state
            .encoded_bytes
            .checked_add(encoded_bytes)
            .ok_or_else(|| Error::internal("transaction byte accounting overflow"))?;
        if self
            .max_encoded_bytes
            .is_some_and(|limit| next_encoded > limit)
        {
            return Err(transaction_admission_full(
                "explicit-transaction byte admission is full",
            ));
        }
        if let Some(existing) = state.pins.get_mut(&pin) {
            existing.references = existing
                .references
                .checked_add(1)
                .ok_or_else(|| Error::internal("transaction pin reference overflow"))?;
        } else {
            state
                .pins
                .insert(pin, TransactionPinUsage { references: 1 });
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
        if current.finalizing
            || current
                .deadline
                .is_some_and(|deadline| deadline <= Instant::now())
        {
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
        if self.max_encoded_bytes.is_some_and(|limit| next > limit) {
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
        if record.finalizing
            || record
                .deadline
                .is_some_and(|deadline| deadline <= Instant::now())
        {
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
        if record
            .deadline
            .is_some_and(|deadline| deadline <= Instant::now())
            || record.finalizing
        {
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
                let reason = if record.deadline.is_some_and(|deadline| deadline <= now) {
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
            // Keep accounting and the shared owner reference until an in-flight query releases the
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
        (state.records.len(), state.encoded_bytes, state.pins.len())
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
    if remove_pin {
        state.pins.remove(&record.pin);
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

#[derive(Clone, Debug, Default)]
pub(super) struct SharedCounter(Arc<AtomicU64>);

impl SharedCounter {
    fn get(&self) -> u64 {
        self.0.load(Ordering::Acquire)
    }

    fn set(&self, value: u64) {
        self.0.store(value, Ordering::Release);
    }

    fn advance(&self, value: u64) {
        self.0.fetch_max(value, Ordering::AcqRel);
    }
}

impl From<u64> for SharedCounter {
    fn from(value: u64) -> Self {
        Self(Arc::new(AtomicU64::new(value)))
    }
}

impl Serialize for SharedCounter {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        self.get().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SharedCounter {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        u64::deserialize(deserializer).map(Into::into)
    }
}

#[derive(Clone, Debug)]
pub(super) struct SharedName(Arc<ArcSwap<String>>);

impl SharedName {
    fn get(&self) -> Arc<String> {
        self.0.load_full()
    }

    fn set(&self, value: String) {
        self.0.store(Arc::new(value));
    }
}

impl From<String> for SharedName {
    fn from(value: String) -> Self {
        Self(Arc::new(ArcSwap::from_pointee(value)))
    }
}

impl Serialize for SharedName {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        self.get().serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for SharedName {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        String::deserialize(deserializer).map(Into::into)
    }
}

#[derive(Clone, Debug, Default)]
pub(super) struct SharedStatistics(Arc<ArcSwapOption<StatisticsSnapshot>>);

impl SharedStatistics {
    fn get(&self) -> Option<Arc<StatisticsSnapshot>> {
        self.0.load_full()
    }

    fn set(
        &self,
        value: Arc<StatisticsSnapshot>,
    ) -> std::result::Result<(), Arc<StatisticsSnapshot>> {
        let previous = self
            .0
            .compare_and_swap(&None::<Arc<StatisticsSnapshot>>, Some(Arc::clone(&value)));
        if previous.is_none() {
            Ok(())
        } else {
            Err(value)
        }
    }

    fn get_or_init(
        &self,
        initialize: impl FnOnce() -> Arc<StatisticsSnapshot>,
    ) -> Arc<StatisticsSnapshot> {
        if let Some(value) = self.get() {
            return value;
        }
        let value = initialize();
        match self.set(Arc::clone(&value)) {
            Ok(()) => value,
            Err(_) => self.get().unwrap_or(value),
        }
    }

    fn clear(&self) {
        self.0.store(None);
    }
}

impl From<OnceLock<Arc<StatisticsSnapshot>>> for SharedStatistics {
    fn from(value: OnceLock<Arc<StatisticsSnapshot>>) -> Self {
        Self(Arc::new(ArcSwapOption::from(value.into_inner())))
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct ProjectState {
    pub(super) id: ProjectId,
    pub(super) display_name: SharedName,
    pub(super) graph: GraphStore,
    pub(super) temporal: TemporalStore,
    pub(super) predicate_versions: BTreeMap<DependencyStamp, u64>,
    pub(super) indexes: IndexCatalog,
    pub(super) next_node_id: SharedCounter,
    pub(super) next_edge_id: SharedCounter,
    #[serde(default)]
    pub(super) authority_revision: SharedCounter,
    /// Ephemeral and rebuildable; never part of durable state or a checkpoint.
    #[serde(skip)]
    pub(super) optimizer_statistics: SharedStatistics,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(super) struct DatabaseState {
    pub(super) projects: BTreeMap<ProjectId, Arc<ProjectState>>,
    pub(super) names: BTreeMap<String, ProjectId>,
    pub(super) broker: BrokerStateMachine,
    #[serde(default)]
    pub(super) embedding_activation: Option<EmbeddingProfileActivation>,
    #[serde(default)]
    pub(super) security: SecurityState,
    #[serde(default)]
    pub(super) applied: Bookmark,
    #[serde(default)]
    request_results: BTreeMap<Uuid, RequestResultRecord>,
    #[serde(default)]
    request_result_order: BTreeMap<u64, Uuid>,
    #[serde(default)]
    request_result_bytes: u64,
    #[serde(default)]
    last_payload_checksum: Option<[u8; 32]>,
    #[serde(default)]
    last_response: Vec<u8>,
}

#[derive(Default)]
struct OrderedMutationOverlay {
    reservations: BTreeMap<u64, OrderedMutationReservation>,
}

struct OrderedMutationReservation {
    token: Uuid,
    payload_digest: [u8; 32],
    request_id: Option<Uuid>,
    commit_time_millis: i64,
    broker_segments: Vec<SegmentDescriptor>,
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
    state_checksum: Option<[u8; 32]>,
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
    // Identity access paths to the same canonical stores, not graph images.
    reader_projects: papaya::HashMap<ProjectId, Arc<ProjectState>>,
    reader_names: papaya::HashMap<String, ProjectId>,
    reader_bookmark: ArcSwap<Bookmark>,
    reader_broker: ArcSwap<BrokerStateMachine>,
    reader_security: ArcSwap<SecurityState>,
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
    /// Bounded identity-only work reserved before the ordered WAL append.
    ordered_overlay: Mutex<OrderedMutationOverlay>,
    broker_changes: tokio::sync::watch::Sender<u64>,
    text_embedding: RwLock<Option<Arc<dyn TextEmbedding>>>,
    embedding_jobs: OnceLock<super::embedding_jobs::DurableEmbeddingWorker>,
    semantic_initialization: parking_lot::Mutex<()>,
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
        let admission = AdmissionController::unrestricted();
        let segments = SegmentStore::open(directory.as_ref(), maximum_write_bytes.max(1 << 20))?;
        let transactions = TransactionAdmissionRegistry::unrestricted();
        let (broker_changes, _) = tokio::sync::watch::channel(0);
        let state = DatabaseState::default();
        let reader_broker = ArcSwap::from_pointee(state.broker.clone());
        let reader_security = ArcSwap::from_pointee(state.security.clone());
        Ok(Self(Arc::new(DatabaseInner {
            state: RwLock::new(state),
            reader_projects: papaya::HashMap::new(),
            reader_names: papaya::HashMap::new(),
            reader_bookmark: ArcSwap::from_pointee(Bookmark::default()),
            reader_broker,
            reader_security,
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
            embedding_jobs: OnceLock::new(),
            semantic_initialization: parking_lot::Mutex::new(()),
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
        drop(current);
        self.start_embedding_jobs()?;
        Ok(())
    }

    fn start_embedding_jobs(&self) -> Result<()> {
        use super::embedding_jobs::*;
        if self.0.embedding_jobs.get().is_some() {
            return Ok(());
        }
        let load_database = Arc::downgrade(&self.0);
        let publish_database = Arc::downgrade(&self.0);
        let completed_wal = Arc::new(AtomicU64::new(0));
        self.write_binding()?
            .runtime
            .upgrade()
            .ok_or_else(|| Error::new(ErrorCode::Cancelled, "write runtime is unavailable"))?
            .register_embedding_replay_floor(Arc::clone(&completed_wal))?;
        let queue = DurableEmbeddingWorker::start(
            self.committed_embedding_owners(Arc::clone(&completed_wal)),
            EmbeddingCallbacks {
                load: Arc::new(move |job, cancellation| {
                    if cancellation.is_cancelled() {
                        return Ok(None);
                    }
                    let Some(inner) = load_database.upgrade() else {
                        return Ok(None);
                    };
                    let database = Database(inner);
                    let (project, _) = match database.capture_canonical_project(job.owner.project) {
                        Ok(project) => project,
                        Err(error) if error.code == ErrorCode::ProjectNotFound => return Ok(None),
                        Err(error) => return Err(error),
                    };
                    let (revision, _) = embedding_owner_dependencies(&project, job.owner);
                    if revision.is_some_and(|revision| revision != job.revision) {
                        return Ok(None);
                    }
                    let encoder = database.text_embedding()?;
                    let property = match job.owner.kind {
                        crate::types::EntityKind::Node => crate::graph::SEMANTIC_NODE_PROPERTY,
                        crate::types::EntityKind::Relationship => {
                            crate::graph::SEMANTIC_RELATIONSHIP_PROPERTY
                        }
                    };
                    let mut chunks = Vec::new();
                    let semantic_index = match job.owner.kind {
                        crate::types::EntityKind::Node => crate::graph::SEMANTIC_NODE_INDEX,
                        crate::types::EntityKind::Relationship => {
                            crate::graph::SEMANTIC_RELATIONSHIP_INDEX
                        }
                    };
                    let current_vector = |name: &str| {
                        project
                            .indexes
                            .vector_search_source(name)
                            .and_then(|(column, _)| column.row_revision(job.owner.entity_id))
                            .is_some_and(|revision| revision >= job.revision)
                    };
                    if project.indexes.contains(semantic_index) && !current_vector(semantic_index) {
                        chunks.push(SemanticTextChunk {
                            property,
                            text: crate::graph::semantic_owner_text(
                                &project.graph,
                                job.owner.kind,
                                job.owner.entity_id,
                            )?
                            .map(Arc::from),
                        });
                    }
                    if job.owner.kind == crate::types::EntityKind::Node {
                        for definition in project.indexes.embedding_definitions() {
                            if current_vector(&definition.name) {
                                continue;
                            }
                            let text = project
                                .graph
                                .node(crate::NodeId(job.owner.entity_id))
                                .filter(|node| node.labels().contains(&definition.label))
                                .and_then(|node| node.property(definition.source_property))
                                .and_then(|value| match value {
                                    crate::ScalarValue::String(text) => Some(text),
                                    _ => None,
                                });
                            chunks.push(SemanticTextChunk {
                                property: definition.target_property,
                                text,
                            });
                        }
                    }
                    let current_database = Arc::downgrade(&database.0);
                    let is_current = Arc::new(move || {
                        let Some(inner) = current_database.upgrade() else {
                            return false;
                        };
                        let database = Database(inner);
                        let Ok((project, _)) =
                            database.capture_canonical_project(job.owner.project)
                        else {
                            return false;
                        };
                        embedding_owner_dependencies(&project, job.owner).0 == revision
                    });
                    Ok(Some(EmbeddingWork {
                        encoder,
                        chunks,
                        is_current: Some(is_current),
                    }))
                }),
                publish: Arc::new(move |job, vectors, cancellation| {
                    let _worker = DatabaseBlockingWorker::enter();
                    if cancellation.is_cancelled() {
                        return Ok(false);
                    }
                    let Some(inner) = publish_database.upgrade() else {
                        return Ok(false);
                    };
                    let database = Database(inner);
                    let (project, bookmark) = match database
                        .capture_canonical_project(job.owner.project)
                    {
                        Ok(project) => project,
                        Err(error) if error.code == ErrorCode::ProjectNotFound => return Ok(false),
                        Err(error) => return Err(error),
                    };
                    let (revision, dependencies) =
                        embedding_owner_dependencies(&project, job.owner);
                    if revision.is_some_and(|revision| revision != job.revision) {
                        return Ok(false);
                    }
                    if revision.is_none()
                        && vectors.iter().any(|mutation| {
                            matches!(mutation, ResolvedVectorMutation::Upsert { .. })
                        })
                    {
                        return Ok(false);
                    }
                    if vectors.is_empty() {
                        return Ok(true);
                    }
                    match database.commit_scoped_from(
                        DatabaseMutation::Graph {
                            project: job.owner.project,
                            validation: MutationValidation {
                                snapshot: bookmark,
                                dependencies,
                            },
                            graph: Vec::new(),
                            temporal: Vec::new(),
                            vectors,
                            administrative: None,
                        },
                        MutationKind::Graph,
                        Some(job.owner.project),
                        Some(Uuid::new_v4()),
                        AdmissionClass::Control,
                        CommitAcknowledgement::Published,
                        ConnectionId::new(),
                        None,
                    ) {
                        Ok(_) => Ok(true),
                        Err(error) if error.code == ErrorCode::TransactionConflict => Ok(false),
                        Err(error) => Err(error),
                    }
                }),
                failed: Arc::new(
                    |job, error| tracing::error!(project = %job.owner.project, owner = job.owner.entity_id, code = ?error.code, message = %error.message, "asynchronous embedding failed"),
                ),
            },
        )?;
        self.0
            .embedding_jobs
            .set(queue)
            .map_err(|_| Error::internal("embedding queue was concurrently bound"))
    }

    pub async fn shutdown_embedding_jobs(&self) -> Result<()> {
        if let Some(queue) = self.0.embedding_jobs.get() {
            queue.shutdown().await?;
        }
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
        if !self.0.reader_projects.pin().contains_key(&project) {
            return Err(Error::new(
                ErrorCode::ProjectNotFound,
                "configured project does not exist",
            ));
        }
        Ok(bookmark)
    }

    #[must_use]
    pub fn bookmark(&self) -> Bookmark {
        **self.0.reader_bookmark.load()
    }

    /// Executes on the database's blocking worker pool while the caller's async runtime remains free.
    pub async fn execute_async(&self, request: QueryRequest) -> Result<Vec<QueryStreamEvent>> {
        let database = self.clone();
        let handle = self.write_binding()?.handle.clone();
        handle
            .spawn_blocking(move || {
                let _worker = DatabaseBlockingWorker::enter();
                let mut events = Vec::new();
                database.execute(request, &mut |event| {
                    events.push(event);
                    Ok(())
                })?;
                Ok(events)
            })
            .await
            .map_err(|error| Error::internal(format!("query worker failed: {error}")))?
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
            let bookmark = self.bookmark();
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
                    .map(|remaining| {
                        if self.0.request_timeout.is_zero() {
                            remaining
                        } else {
                            remaining.min(self.0.request_timeout)
                        }
                    })
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
        Ok(self.0.reader_security.load().as_ref().clone())
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

    fn committed_embedding_owners(
        &self,
        completed: Arc<AtomicU64>,
    ) -> Arc<super::embedding_jobs::NextCommittedOwner> {
        use super::embedding_jobs::{EmbeddingJob, EmbeddingOwner, EmbeddingSeed};
        let initial = self.bookmark();
        let projects = self
            .0
            .reader_projects
            .pin()
            .iter()
            .map(|(id, _)| *id)
            .collect::<Vec<_>>()
            .into_iter();
        let initializer = self.embedding_seed_initializer();
        let weak = Arc::downgrade(&self.0);
        // Only one recovered project cursor and one WAL mutation's owner identities are held.
        // Source text remains on the canonical graph; completed WAL records can be compacted.
        let state = Mutex::new((
            projects,
            None::<EmbeddingSeed>,
            Vec::<EmbeddingJob>::new().into_iter(),
            initial.index,
            false,
        ));
        Arc::new(move |cancellation| {
            let mut state = state.lock();
            loop {
                if cancellation.is_cancelled() {
                    return Ok(None);
                }
                if let Some(seed) = &mut state.1 {
                    if let Some(job) = seed.next() {
                        return job.map(Some);
                    }
                    state.1 = None;
                }
                if let Some(job) = state.2.next() {
                    return Ok(Some(job));
                }
                if !state.4 {
                    if let Some(project) = state.0.next() {
                        match initializer(project, cancellation) {
                            Ok(seed) => state.1 = seed,
                            Err(error) if error.code == ErrorCode::ProjectNotFound => {}
                            Err(error) => return Err(error),
                        }
                        continue;
                    }
                    state.4 = true;
                }
                completed.store(state.3, Ordering::Release);
                let Some(inner) = weak.upgrade() else {
                    return Ok(None);
                };
                let database = Database(inner);
                let runtime = database.write_binding()?.runtime.upgrade().ok_or_else(|| {
                    Error::new(ErrorCode::Cancelled, "write runtime is unavailable")
                })?;
                let Some(entry) = runtime.committed_embedding_entry(state.3.saturating_add(1))?
                else {
                    return Ok(None);
                };
                if entry.index() > database.bookmark().index {
                    return Ok(None);
                }
                let mutation = decode_database_mutation(entry.payload())?;
                if let DatabaseMutation::Graph {
                    project,
                    graph,
                    administrative,
                    ..
                } = mutation
                {
                    let live = match database.capture_canonical_project(project) {
                        Ok((live, _)) => Some(live),
                        Err(error) if error.code == ErrorCode::ProjectNotFound => None,
                        Err(error) => return Err(error),
                    };
                    if let Some(live) = live {
                        database.initialize_automatic_semantic(project)?;
                        if matches!(
                            administrative,
                            Some(AdministrativeMutation::CreateEmbedding { .. })
                        ) {
                            state.1 = initializer(project, cancellation)?;
                        } else if !graph.is_empty() {
                            let (nodes, edges) = embedding_changed_owners(&live, &graph)?;
                            state.2 = nodes
                                .into_iter()
                                .map(|id| (crate::types::EntityKind::Node, id.0))
                                .chain(
                                    edges
                                        .into_iter()
                                        .map(|id| (crate::types::EntityKind::Relationship, id.0)),
                                )
                                .map(|(kind, entity_id)| {
                                    let owner = EmbeddingOwner {
                                        project,
                                        kind,
                                        entity_id,
                                    };
                                    let (revision, _) = embedding_owner_dependencies(&live, owner);
                                    EmbeddingJob {
                                        owner,
                                        revision: revision.unwrap_or(entry.index()).max(1),
                                    }
                                })
                                .collect::<Vec<_>>()
                                .into_iter();
                        }
                    }
                }
                state.3 = entry.index();
            }
        })
    }

    fn embedding_seed_initializer(&self) -> Arc<super::embedding_jobs::InitializeProject> {
        let database = Arc::downgrade(&self.0);
        Arc::new(move |project, cancellation| {
            let _worker = DatabaseBlockingWorker::enter();
            if cancellation.is_cancelled() {
                return Ok(None);
            }
            let Some(inner) = database.upgrade() else {
                return Ok(None);
            };
            let database = Database(inner);
            database.initialize_automatic_semantic(project)?;
            let (live, _) = database.capture_canonical_project(project)?;
            let node_limit = live.graph.node_slot_count();
            let edge_limit = live.graph.edge_slot_count();
            let live = Arc::downgrade(&live);
            let mut node = 0;
            let mut edge = 0;
            let seed = std::iter::from_fn(move || {
                let live = live.upgrade()?;
                loop {
                    let owner = if node < node_limit {
                        let dense = node;
                        node += 1;
                        let Some(row) = live.graph.node_dense(dense as u32) else {
                            continue;
                        };
                        super::embedding_jobs::EmbeddingOwner {
                            project,
                            kind: crate::types::EntityKind::Node,
                            entity_id: row.id().0,
                        }
                    } else if edge < edge_limit {
                        let dense = edge;
                        edge += 1;
                        let Some(row) = live.graph.edge_dense(dense as u32) else {
                            continue;
                        };
                        super::embedding_jobs::EmbeddingOwner {
                            project,
                            kind: crate::types::EntityKind::Relationship,
                            entity_id: row.id().0,
                        }
                    } else {
                        return None;
                    };
                    let (revision, _) = embedding_owner_dependencies(&live, owner);
                    return Some(Ok(super::embedding_jobs::EmbeddingJob {
                        owner,
                        revision: revision.unwrap_or(1).max(1),
                    }));
                }
            });
            Ok(Some(Box::new(seed) as super::embedding_jobs::EmbeddingSeed))
        })
    }

    /// Initializes empty semantic access paths; the worker seeds owner identities incrementally.
    fn initialize_automatic_semantic(&self, project: ProjectId) -> Result<()> {
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
        if initialized {
            return Ok(());
        }
        self.commit_scoped_from(
            DatabaseMutation::Graph {
                project,
                validation: MutationValidation {
                    snapshot: bookmark,
                    dependencies: TransactionDependencies::default(),
                },
                graph: Vec::new(),
                temporal: Vec::new(),
                vectors: Vec::new(),
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

        let deadline = if self.0.request_timeout.is_zero() {
            None
        } else {
            Instant::now().checked_add(self.0.request_timeout)
        };
        loop {
            self.maintain_local_artifact_readiness()?;
            if self.project_embedding_profile(project)?.as_ref() == Some(&profile) {
                return Ok(());
            }
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
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
        let (live, _) = self.capture_canonical_project(project)?;
        Ok(live.indexes.profile().map(|profile| (*profile).clone()))
    }

    fn validate_embedding_profile_readiness(
        &self,
        project: ProjectId,
        profile: &crate::graph::EmbeddingProfile,
    ) -> Result<()> {
        profile.validate()?;
        if self.text_embedding()?.profile().profile_hash != profile.profile_hash {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "local embedding artifacts are not ready",
            ));
        }
        let (project_state, _) = self.capture_canonical_project(project)?;
        if project_state.indexes.profile().as_deref() == Some(profile) {
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
        let _permit = self.0.admission.try_admit_until(
            connection_id,
            class,
            payload.len(),
            if request_timeout.is_zero() {
                None
            } else {
                Instant::now().checked_add(request_timeout)
            },
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
            if self.0.reader_projects.pin().contains_key(&project) {
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
            .reader_names
            .pin()
            .get(&normalize_name(name))
            .copied()
            .ok_or_else(|| Error::new(ErrorCode::ProjectNotFound, "project does not exist"))
    }

    fn capture_canonical_project(
        &self,
        project: ProjectId,
    ) -> Result<(Arc<ProjectState>, Bookmark)> {
        let live = self
            .0
            .reader_projects
            .pin()
            .get(&project)
            .cloned()
            .ok_or_else(|| Error::new(ErrorCode::ProjectNotFound, "project does not exist"))?;
        let bookmark = **self.0.reader_bookmark.load();
        Ok((live, bookmark))
    }

    fn publish_reader_project(&self, state: &DatabaseState, project: ProjectId) {
        let projects = self.0.reader_projects.pin();
        let names = self.0.reader_names.pin();
        if let Some(live) = state.projects.get(&project) {
            projects.insert(project, Arc::clone(live));
            names.insert(normalize_name(&live.display_name.get()), project);
        } else if let Some(previous) = projects.remove(&project) {
            names.remove(&normalize_name(&previous.display_name.get()));
        }
    }

    fn publish_reader_registry(&self, state: &DatabaseState) {
        let projects = self.0.reader_projects.pin();
        let names = self.0.reader_names.pin();
        for live in state.projects.values() {
            projects.insert(live.id, Arc::clone(live));
            names.insert(normalize_name(&live.display_name.get()), live.id);
        }
        names.retain(|name, id| {
            state
                .projects
                .get(id)
                .is_some_and(|live| normalize_name(&live.display_name.get()) == *name)
        });
        projects.retain(|id, _| state.projects.contains_key(id));
        self.0.reader_bookmark.store(Arc::new(state.applied));
        self.0.reader_broker.store(Arc::new(state.broker.clone()));
        self.0
            .reader_security
            .store(Arc::new(state.security.clone()));
    }

    /// Large projects whose optimizer-statistics cache is cold, with their current graph revision.
    ///
    /// Only cold admission or an explicit administrative rebuild can leave a cache absent.
    /// Ordinary writes install or advance a bounded count summary and never request resampling.
    pub(super) fn cold_statistics_projects(&self) -> Vec<(ProjectId, u64)> {
        let projects = self.0.reader_projects.pin();
        projects
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

    /// Samples a cold project's property, index and temporal distributions off the query path.
    /// Queries can instead initialize a bounded exact-count summary without waiting for sampling.
    ///
    /// Captures the shared canonical project without taking the writer's state lock. The
    /// statistics cache publishes one derived sample; concurrent writes can publish their
    /// current count summary without replacing or copying the graph. Returns whether sampling ran.
    pub(super) fn prewarm_optimizer_statistics(
        &self,
        project: ProjectId,
        expected_revision: u64,
    ) -> bool {
        let Ok((snapshot, _)) = self.capture_canonical_project(project) else {
            return false;
        };
        if snapshot.graph.revision() != expected_revision
            || snapshot.optimizer_statistics.get().is_some()
        {
            return false;
        }
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
        mut request: QueryRequest,
        emit: &mut dyn FnMut(QueryStreamEvent) -> Result<()>,
    ) -> Result<()> {
        request.validate()?;
        let parsed = parse(&request.query)?;
        let writes = statement_writes(&parsed.statement);
        if writes && request.deadline.is_none() && !self.0.request_timeout.is_zero() {
            request.deadline = Instant::now().checked_add(self.0.request_timeout);
        }
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
                let (snapshot, bookmark) = self.capture_canonical_project(candidate_project)?;
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
        if matches!(&parsed.statement, Statement::CreateEmbedding(_)) {
            self.ensure_embedding_profile_activated(project)?;
        }
        let (snapshot, captured_bookmark) = self.capture_canonical_project(project)?;
        let text_embedding = self.0.text_embedding.read().clone();
        let capabilities = full_capabilities();
        // The parsed statement already determines whether it writes. The query engine owns
        // binding and capability validation, including cache reuse; do not bind it twice.
        let read_only = !writes;
        if read_only {
            self.ensure_apply_healthy()?;
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
                    &mut stream,
                )?
            };
            emit_query_summary(request.request_id, output.result, rows, emit)
        } else {
            loop {
                let (snapshot, planning_bookmark) = self.capture_canonical_project(project)?;
                self.remaining_query_write_time(&request)?;
                let next_index = planning_bookmark
                    .index
                    .checked_add(1)
                    .ok_or_else(|| Error::internal("log index exhausted"))?;
                let mut output = {
                    execute_on_project(
                        &snapshot,
                        &request,
                        planning_bookmark,
                        next_index,
                        capabilities,
                        text_embedding.as_deref(),
                    )?
                };
                let administrative = administrative_mutation(
                    output.administrative.take(),
                    &snapshot,
                    &mut output.graph_mutations,
                    text_embedding.as_deref(),
                    next_index,
                )?;
                let vectors = Vec::new();
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
                drop(snapshot);
                let committed = self.commit_from(
                    mutation,
                    MutationKind::Graph,
                    project,
                    Some(request.request_id),
                    AdmissionClass::Client,
                    request.consistency,
                    request.connection_id,
                    None,
                    self.remaining_query_write_time(&request)?,
                );
                let (bookmark, _) = match committed {
                    Ok(committed) => committed,
                    Err(error) if error.code == ErrorCode::TransactionConflict => {
                        self.ensure_apply_healthy()?;
                        continue;
                    }
                    Err(error) => return Err(error),
                };
                output.result.bookmark = bookmark;
                let (project_state, _) = self.capture_canonical_project(project)?;
                return emit_result(
                    request.request_id,
                    output.result,
                    project_state.graph.catalog(),
                    &project_state.indexes,
                    emit,
                );
            }
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
            let state = self.0.reader_projects.pin();
            (
                state
                    .iter()
                    .map(|(_, project)| {
                        (
                            project.id.to_string(),
                            project.display_name.get().as_ref().clone(),
                        )
                    })
                    .collect::<Vec<_>>(),
                **self.0.reader_bookmark.load(),
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
            let (project, bookmark) = self.capture_canonical_project(project)?;
            (project.indexes.statuses().collect::<Vec<_>>(), bookmark)
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
            let (project, bookmark) = self.capture_canonical_project(project)?;
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
            (rows, bookmark)
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
                    rows.iter().map(|row| row.1.to_string()).collect::<Vec<_>>(),
                ),
                (
                    "property",
                    rows.iter().map(|row| row.2.to_string()).collect::<Vec<_>>(),
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
        let rows = self.0.reader_broker.load().topic_metrics(project);
        let bookmark = self.bookmark();
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
        let rows = self.0.reader_broker.load().queue_metrics(project);
        let bookmark = self.bookmark();
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
        let rows = self.0.reader_broker.load().exchange_metrics(project);
        let bookmark = self.bookmark();
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
        let rows = self.0.reader_broker.load().consumer_lag_metrics(project);
        let bookmark = self.bookmark();
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
        let existing = self.0.reader_names.pin().get(&normalized).copied();
        if let Some(id) = existing {
            if if_not_exists {
                self.initialize_automatic_semantic(id)?;
                let bookmark =
                    self.consistency_barrier(request.consistency, Some(self.bookmark()))?;
                return emit_admin_summary(request.request_id, bookmark, emit);
            }
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
        // Publish the constant-size semantic definitions before acknowledging a new project.
        // Owner text and vectors are still seeded by the independent embedding worker; reads
        // after this acknowledgement can observe an empty ready index while that work proceeds.
        self.initialize_automatic_semantic(id)?;
        let bookmark =
            self.consistency_barrier(request.consistency, Some(self.bookmark().max(bookmark)))?;
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

#[async_trait]
impl MutationStateBackend for Database {
    async fn prepare_command(
        &self,
        mut command: WriteCommand,
        sequencer_time_millis: i64,
    ) -> Result<WriteCommand> {
        // Admission and replay resolve values from this authoritative WAL header. The original
        // journal bytes need no additional decode, clone, encode, or worker round trip here.
        millis_to_nanos(sequencer_time_millis)?;
        command.commit_time_millis = sequencer_time_millis;
        command.validate()?;
        Ok(command)
    }

    async fn validate_command(&self, command: &WriteCommand) -> Result<()> {
        let database = self.clone();
        let owned_command = command.clone();
        tokio::task::spawn_blocking(move || {
            let command = &owned_command;
            database.ensure_apply_healthy()?;
            let _apply = database.0.apply.lock();
            let current = database.0.state.read();
            let position = Bookmark {
                term: current.applied.term.max(1),
                index: current
                    .applied
                    .index
                    .checked_add(1)
                    .ok_or_else(|| Error::internal("local write index exhausted"))?,
            };
            validate_resolved_database_command(&current, command, position).map(|_| ())
        })
        .await
        .map_err(|error| Error::internal(format!("validate_command worker failed: {error}")))?
    }

    async fn reserve_command(
        &self,
        command: &WriteCommand,
        position: Bookmark,
    ) -> Result<CommandReservation> {
        let database = self.clone();
        let owned_command = command.clone();
        tokio::task::spawn_blocking(move || {
            let command = &owned_command;
            database.ensure_apply_healthy()?;
            let _apply = database.0.apply.lock();
            let current = database.0.state.read();
            let mut ordered = database.0.ordered_overlay.lock();
            if ordered.reservations.contains_key(&position.index) {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "local write position already has an ordered command reservation",
                ));
            }

            if !ordered.reservations.is_empty() {
                return Err(Error::retryable(
                    ErrorCode::WriteAdmissionFull,
                    "an ordered mutation is still awaiting local application",
                    Some(1),
                ));
            }
            validate_resolved_database_command(&current, command, position)?;
            let reservation = CommandReservation::owned_serialized(position)?;
            let token = reservation
                .token()
                .ok_or_else(|| Error::internal("owned database reservation has no token"))?;
            let broker_segments = Vec::new();
            ordered.reservations.insert(
                position.index,
                OrderedMutationReservation {
                    token,
                    payload_digest: *blake3::hash(&command.payload).as_bytes(),
                    request_id: command.request_id,
                    commit_time_millis: command.commit_time_millis,
                    broker_segments,
                },
            );
            Ok(reservation)
        })
        .await
        .map_err(|error| Error::internal(format!("reserve_command worker failed: {error}")))?
    }

    async fn complete_command_reservation(
        &self,
        reservation: CommandReservation,
        outcome: CommandReservationOutcome,
    ) -> Result<()> {
        let database = self.clone();

        tokio::task::spawn_blocking(move || {
            let Some(token) = reservation.token() else {
                return Ok(());
            };
            let mut released = Vec::<(Uuid, Vec<SegmentDescriptor>)>::new();
            {
                let _apply = database.0.apply.lock();
                let mut ordered = database.0.ordered_overlay.lock();
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
                } else if let Some(record) =
                    ordered.reservations.remove(&reservation.position().index)
                {
                    released.push((record.token, record.broker_segments));
                }
            }
            let rejected = outcome == CommandReservationOutcome::Rejected;
            for (owner, descriptors) in released {
                release_pending_broker_segments(&database.0, owner, &descriptors, rejected);
            }
            Ok(())
        })
        .await
        .map_err(|error| {
            Error::internal(format!(
                "complete_command_reservation worker failed: {error}"
            ))
        })?
    }

    async fn applied_bookmark(&self) -> Bookmark {
        self.bookmark()
    }

    async fn apply_mutation(&self, entry: &MutationEntry) -> Result<MutationApplyResult> {
        let database = self.clone();
        let owned_entry = entry.clone();
        tokio::task::spawn_blocking(move || {
            let entry = &owned_entry;
            database.ensure_apply_healthy()?;
            entry.verify()?;
            let mut mutation = decode_database_mutation(entry.payload())?;
            resolve_sequencer_values(&mut mutation, entry.commit_time_millis())?;
            let intent_digest = entry
                .request_id()
                .map(|_| request_intent_digest(&mutation))
                .transpose()?
                .unwrap_or([0; 32]);
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
            validate_database_entry(entry, &mutation)?;
            let _apply = database.0.apply.lock();
            let mut state = database.0.state.write();

            let previous_project_name = match &mutation {
                DatabaseMutation::RenameProject { id, .. } => state
                    .projects
                    .get(id)
                    .map(|project| normalize_name(&project.display_name.get())),
                _ => None,
            };
            let affected_project = match &mutation {
                DatabaseMutation::CreateProject { id, .. }
                | DatabaseMutation::RenameProject { id, .. }
                | DatabaseMutation::DropProject { id, .. } => Some(*id),
                _ => None,
            };
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
                    database.0.fatal_apply.store(true, Ordering::Release);
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        "database mutation differs at an applied local write index",
                    ));
                }
                return Ok(MutationApplyResult {
                    response: state.last_response.clone(),
                    duplicate: true,
                });
            }
            if entry.index() != state.applied.index.saturating_add(1) {
                database.0.fatal_apply.store(true, Ordering::Release);
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "database mutation apply has a local write-index gap",
                ));
            }
            if let Some(request_id) = entry.request_id()
                && let Some(record) = state.request_results.get(&request_id)
            {
                if record.intent_digest != intent_digest {
                    database.0.fatal_apply.store(true, Ordering::Release);
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        "committed request ID refers to a different mutation intent",
                    ));
                }
                let response = record.response.clone();
                state.applied = entry.bookmark();
                database.0.reader_bookmark.store(Arc::new(entry.bookmark()));
                state.last_payload_checksum = Some(entry.checksum());
                state.last_response = response.clone();
                if let Err(error) = prune_request_results(&mut state) {
                    database.0.fatal_apply.store(true, Ordering::Release);
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
            // Admission already checked the exact predecessor while excluding other writers.
            // Replaying persisted entries has no live reservation and needs its own preflight.
            let reserved = {
                let ordered = database.0.ordered_overlay.lock();
                if let Some(reservation) = ordered.reservations.get(&entry.index()) {
                    if reservation.payload_digest != *blake3::hash(entry.payload()).as_bytes()
                        || reservation.request_id != entry.request_id()
                        || reservation.commit_time_millis != entry.commit_time_millis()
                    {
                        return Err(Error::new(
                            ErrorCode::CorruptStorage,
                            "committed write differs from its validated reservation",
                        ));
                    }
                    true
                } else {
                    false
                }
            };
            if !reserved {
                validate_sequencer_mutation(
                    &state,
                    &mutation,
                    entry.bookmark(),
                    commit_time_nanos,
                )?;
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
                let reply = apply_mutation(
                    &mut state,
                    mutation,
                    entry.bookmark(),
                    commit_time_nanos,
                    Some(&database.0.segments),
                )?;
                let response = encode_database_value(&reply, "apply response")?;
                let staged_state = &mut *state;
                if let Some(previous_cursor) = broker_publish_cursor {
                    committed_broker_segment = staged_state
                        .broker
                        .newest_payload_segment_after(previous_cursor);
                }
                staged_state.applied = entry.bookmark();
                staged_state.last_payload_checksum = Some(entry.checksum());
                staged_state.last_response = response.clone();
                if let Some(request_id) = entry.request_id() {
                    retain_request_result(
                        staged_state,
                        request_id,
                        intent_digest,
                        response.clone(),
                    )?;
                }
                prune_request_results(staged_state)?;
                if let Some(previous) = previous_broker_segments {
                    let current = staged_state
                        .broker
                        .payload_segments()
                        .into_iter()
                        .collect::<BTreeSet<_>>();
                    retired_broker_segments.extend(previous.difference(&current).cloned());
                }
                if let Some(project) = affected_project {
                    database.publish_reader_project(staged_state, project);
                    if let Some(previous_name) = &previous_project_name
                        && staged_state.projects.get(&project).is_none_or(|live| {
                            normalize_name(&live.display_name.get()) != *previous_name
                        })
                    {
                        database.0.reader_names.pin().remove(previous_name);
                    }
                }
                database.0.reader_bookmark.store(Arc::new(entry.bookmark()));
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
                database.0.fatal_apply.store(true, Ordering::Release);
            } else {
                // The canonical graph mutation is committed. Release writer state before broker
                // notification and segment reclamation.
                drop(state);
                if is_broker_mutation {
                    database.0.broker_changes.send_replace(entry.index());
                }
                if let Some(descriptor) = committed_broker_segment {
                    database
                        .0
                        .pending_broker_segments
                        .lock()
                        .remove(&descriptor);
                }
                if !retired_broker_segments.is_empty() {
                    database
                        .0
                        .retired_broker_segments
                        .lock()
                        .extend(retired_broker_segments);
                }
            }
            result
        })
        .await
        .map_err(|error| Error::internal(format!("apply_mutation worker failed: {error}")))?
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
        let (snapshot, current_bookmark) = self.capture_canonical_project(project)?;
        self.wait_for_captured_all(consistency, barrier, current_bookmark)?;
        let fence = self.current_transaction_fence(current_bookmark)?;
        let deadline = if self.0.request_timeout.is_zero() {
            None
        } else {
            Some(
                Instant::now()
                    .checked_add(self.0.request_timeout)
                    .ok_or_else(|| Error::invalid_data("transaction deadline overflow"))?,
            )
        };
        let mut state = TransactionState {
            catalog: snapshot.graph.catalog().clone(),
            working: Arc::clone(&snapshot),
            bookmark: current_bookmark,
            consistency,
            batches: Vec::new(),
            accounted_bytes: 0,
        };
        state.accounted_bytes = MIN_TRANSACTION_ACCOUNTED_BYTES;
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

fn embedding_owner_dependencies(
    project: &ProjectState,
    owner: super::embedding_jobs::EmbeddingOwner,
) -> (Option<u64>, TransactionDependencies) {
    let mut dependencies = TransactionDependencies::default();
    let revision = match owner.kind {
        crate::types::EntityKind::Node => {
            let id = crate::NodeId(owner.entity_id);
            let revision = project.graph.node(id).map(|row| row.revision());
            dependencies.entities.insert(
                crate::cypher::EntityDependency::Node(id),
                revision.unwrap_or(0),
            );
            revision
        }
        crate::types::EntityKind::Relationship => {
            let id = crate::EdgeId(owner.entity_id);
            match project.graph.edge(id) {
                Some(edge) => {
                    let mut revision = edge.revision();
                    dependencies
                        .entities
                        .insert(crate::cypher::EntityDependency::Relationship(id), revision);
                    for node in [edge.source(), edge.target()] {
                        let node_revision =
                            project.graph.node(node).map_or(0, |row| row.revision());
                        revision = revision.max(node_revision);
                        dependencies
                            .entities
                            .insert(crate::cypher::EntityDependency::Node(node), node_revision);
                    }
                    Some(revision)
                }
                None => {
                    dependencies
                        .entities
                        .insert(crate::cypher::EntityDependency::Relationship(id), 0);
                    None
                }
            }
        }
    };
    (revision, dependencies)
}

fn embedding_changed_owners(
    project: &ProjectState,
    mutations: &[GraphMutation],
) -> Result<(BTreeSet<crate::NodeId>, BTreeSet<crate::EdgeId>)> {
    let (mut nodes, edges) =
        crate::graph::semantic_affected_owners(&project.graph, &[], mutations)?;
    let sources = project
        .indexes
        .embedding_definitions()
        .map(|definition| definition.source_property)
        .collect::<BTreeSet<_>>();
    for mutation in mutations {
        if let GraphMutation::SetNodeProperty { node, property, .. } = mutation
            && sources.contains(property)
        {
            nodes.insert(*node);
        }
    }
    Ok((nodes, edges))
}

/// Keeps current embeddings fresh across excluded metadata changes without rerunning inference.
/// Only a vector that already covered the previous owner state can advance its freshness stamp.
fn embedding_freshness_updates(
    project: &ProjectState,
    mutations: &[GraphMutation],
) -> Result<Vec<(super::embedding_jobs::EmbeddingOwner, String, u64)>> {
    use super::embedding_jobs::EmbeddingOwner;
    use crate::types::EntityKind;
    let (dirty_nodes, dirty_edges) = embedding_changed_owners(project, mutations)?;
    let mut candidates = BTreeSet::new();
    for mutation in mutations {
        match mutation {
            GraphMutation::SetNodeProperty { node, .. } => {
                candidates.insert(EmbeddingOwner {
                    project: project.id,
                    kind: EntityKind::Node,
                    entity_id: node.0,
                });
                // Newly created owners have no prior embedding or canonical adjacency.
                // Their inserted relationships are already covered by dirty_edges.
                if project.graph.node(*node).is_some() {
                    for edge in project.graph.incident_edge_ids(*node)? {
                        candidates.insert(EmbeddingOwner {
                            project: project.id,
                            kind: EntityKind::Relationship,
                            entity_id: edge.0,
                        });
                    }
                }
            }
            GraphMutation::SetEdgeProperty { edge, .. } => {
                candidates.insert(EmbeddingOwner {
                    project: project.id,
                    kind: EntityKind::Relationship,
                    entity_id: edge.0,
                });
            }
            _ => {}
        }
    }
    let mut updates = Vec::new();
    for owner in candidates {
        let dirty = match owner.kind {
            EntityKind::Node => dirty_nodes.contains(&crate::NodeId(owner.entity_id)),
            EntityKind::Relationship => dirty_edges.contains(&crate::EdgeId(owner.entity_id)),
        };
        if dirty {
            continue;
        }
        let Some(previous_revision) = embedding_owner_dependencies(project, owner).0 else {
            continue;
        };
        let mut names = vec![match owner.kind {
            EntityKind::Node => crate::graph::SEMANTIC_NODE_INDEX.to_owned(),
            EntityKind::Relationship => crate::graph::SEMANTIC_RELATIONSHIP_INDEX.to_owned(),
        }];
        if owner.kind == EntityKind::Node {
            names.extend(
                project
                    .indexes
                    .embedding_definitions()
                    .map(|definition| definition.name),
            );
        }
        for name in names {
            if let Some(revision) = project
                .indexes
                .vector_search_source(&name)
                .and_then(|(index, _)| index.row_revision(owner.entity_id))
                .filter(|revision| *revision >= previous_revision)
            {
                updates.push((owner, name, revision));
            }
        }
    }
    Ok(updates)
}

fn transaction_batch_bytes(batch: &TransactionBatch, limit: usize) -> Result<usize> {
    #[derive(Serialize)]
    struct RetainedTransaction<'a> {
        dependencies: &'a TransactionDependencies,
        graph: &'a [GraphMutation],
        temporal: Vec<PersistedTemporalMutation>,
        vectors: &'a [ResolvedVectorMutation],
    }

    let temporal = batch
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
            dependencies: &batch.dependencies,
            graph: &batch.graph_mutations,
            temporal,
            vectors: &batch.vector_mutations,
        },
        "explicit transaction admission state",
        limit,
    )
    .map(|bytes| bytes.len().saturating_add(256))
}

impl QueryExecutor for Database {
    fn resolve_project(&self, selector: &str) -> Result<ProjectId> {
        self.ensure_apply_healthy()?;
        if let Ok(id) = Uuid::parse_str(selector) {
            let project = ProjectId(id);
            if self.0.reader_projects.pin().contains_key(&project) {
                return Ok(project);
            }
        }
        self.0
            .reader_names
            .pin()
            .get(&normalize_name(selector))
            .copied()
            .ok_or_else(|| Error::new(ErrorCode::ProjectNotFound, "project does not exist"))
    }

    fn execute(
        &self,
        request: QueryRequest,
        emit: &mut dyn FnMut(QueryStreamEvent) -> Result<()>,
    ) -> Result<()> {
        let mut statistics = QueryResultStatistics::new();
        self.execute_autocommit(request, &mut |mut event| {
            statistics.observe(&event)?;
            statistics.settle(&mut event);
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
    working: Arc<ProjectState>,
    bookmark: Bookmark,
    consistency: CommitAcknowledgement,
    batches: Vec<Arc<TransactionBatch>>,
    accounted_bytes: usize,
}

struct TransactionBatch {
    dependencies: TransactionDependencies,
    graph_mutations: Vec<GraphMutation>,
    temporal_mutations: Vec<crate::cypher::PreparedTemporalMutation>,
    vector_mutations: Vec<ResolvedVectorMutation>,
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
    deadline: Option<Instant>,
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
        let mut statistics = QueryResultStatistics::new();
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
        let prior_graph = state
            .batches
            .iter()
            .flat_map(|batch| batch.graph_mutations.iter().cloned())
            .collect::<Vec<_>>();
        let prior_temporal = state
            .batches
            .iter()
            .flat_map(|batch| batch.temporal_mutations.iter().cloned())
            .collect::<Vec<_>>();
        let output = execute_on_project_inner(
            &state.working,
            &request,
            state.bookmark,
            provisional,
            full_capabilities(),
            text_embedding.as_deref(),
            &prior_graph,
            &prior_temporal,
            None,
        )?;
        if output.administrative.is_some() {
            return Err(Error::new(
                ErrorCode::QueryType,
                "schema and index DDL is autocommit-only",
            ));
        }
        let vectors = Vec::new();
        let batch = Arc::new(TransactionBatch {
            dependencies: output.dependencies,
            graph_mutations: output.graph_mutations,
            temporal_mutations: output.temporal_mutations,
            vector_mutations: vectors,
        });
        let next_bytes = state
            .accounted_bytes
            .checked_add(transaction_batch_bytes(
                &batch,
                self.database.0.transactions.maximum_encoded_bytes(),
            )?)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "transaction retained bytes overflow",
                )
            })?;
        self.database.0.transactions.resize(self.id, next_bytes)?;
        let emitted = emit_result(
            request.request_id,
            output.result,
            &state.catalog,
            &state.working.indexes,
            &mut |mut event| {
                statistics.observe(&event)?;
                statistics.settle(&mut event);
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
        state.batches.push(batch);
        state.accounted_bytes = next_bytes;
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
        let mut dependencies = TransactionDependencies::default();
        for batch in state.batches.iter() {
            merge_dependencies(&mut dependencies, batch.dependencies.clone());
        }
        validate_dependencies(&dependencies, &current, state.bookmark.index)?;
        let next_index = self
            .database
            .bookmark()
            .index
            .checked_add(1)
            .ok_or_else(|| Error::internal("log index exhausted"))?;
        let graph = state
            .batches
            .iter()
            .flat_map(|batch| batch.graph_mutations.iter().cloned())
            .map(|mutation| retag_graph_mutation(mutation, next_index))
            .collect::<Vec<_>>();
        let temporal = state
            .batches
            .iter()
            .flat_map(|batch| batch.temporal_mutations.iter().cloned())
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
            .batches
            .iter()
            .flat_map(|batch| batch.vector_mutations.iter().cloned())
            .map(|mutation| retag_vector_mutation(mutation, next_index))
            .collect::<Vec<_>>();
        let timeout = self
            .deadline
            .map(|deadline| {
                deadline
                    .checked_duration_since(Instant::now())
                    .ok_or_else(transaction_expired)
            })
            .transpose()?
            .unwrap_or(Duration::ZERO);
        let committed = self.database.commit_scoped_with_timeout_from(
            DatabaseMutation::Graph {
                project: self.project,
                validation: MutationValidation {
                    snapshot: state.bookmark,
                    dependencies,
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
        if !self.0.reader_projects.pin().contains_key(&read.project) {
            return Err(Error::new(
                ErrorCode::ProjectNotFound,
                "project does not exist",
            ));
        }
        self.0
            .reader_broker
            .load()
            .fetch_partition_bounded(read, &self.0.segments)
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
            Duration::from_millis(u64::from(timeout_millis)),
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
        Ok(self.0.reader_broker.load().as_ref().clone())
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
        Ok(self.0.reader_broker.load().projects_with_state())
    }

    fn topic_metadata(&self, project: ProjectId) -> Result<Vec<(String, usize)>> {
        Ok(self.0.reader_broker.load().topic_metadata(project))
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
            .reader_broker
            .load()
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
            .reader_broker
            .load()
            .group_leader(project, group, generation))
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
            .reader_broker
            .load()
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
            let broker = self.0.reader_broker.load().as_ref().clone();
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
            .reader_broker
            .load()
            .list_offset(project, topic, partition, timestamp)
    }

    fn queue_info(
        &self,
        project: ProjectId,
        name: &str,
    ) -> Result<Option<crate::broker::QueueInfo>> {
        Ok(self.0.reader_broker.load().queue_info(project, name))
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
            let broker = self.0.reader_broker.load().as_ref().clone();
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
) -> Result<ExecutionOutput> {
    execute_on_project_inner(
        project,
        request,
        bookmark,
        mutation_revision,
        capabilities,
        text_embedding,
        &[],
        &[],
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
    emit: &mut dyn FnMut(ExecutionStreamItem) -> Result<()>,
) -> Result<ExecutionOutput> {
    execute_on_project_inner(
        project,
        request,
        bookmark,
        mutation_revision,
        capabilities,
        text_embedding,
        &[],
        &[],
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
    prior_graph_mutations: &[GraphMutation],
    prior_temporal_mutations: &[crate::cypher::PreparedTemporalMutation],
    stream: Option<&mut dyn FnMut(ExecutionStreamItem) -> Result<()>>,
) -> Result<ExecutionOutput> {
    let parameters = request
        .parameters
        .iter()
        .map(|(name, value)| json_to_result(value).map(|value| (name.clone(), value)))
        .collect::<Result<BTreeMap<_, _>>>()?;
    let vector_sources = project
        .indexes
        .definitions()
        .filter_map(|definition| {
            project
                .indexes
                .vector_search_source(&definition.name)
                .map(|(exact, approximate)| (definition, exact, approximate))
        })
        .collect::<Vec<_>>();
    let profile_hash = project
        .indexes
        .profile()
        .map_or([0_u8; 32], |profile| profile.profile_hash);
    let vector_indexes = vector_sources
        .iter()
        .map(|(definition, exact, approximate)| {
            (
                definition.name.clone(),
                crate::cypher::VectorSearchSource {
                    property: definition.properties[0],
                    exact: exact.as_ref(),
                    approximate: approximate.as_deref(),
                    profile_hash,
                },
            )
        })
        .collect();
    let binding_catalog = project.graph.catalog();
    let mut next_node_id = project.next_node_id.get();
    let mut next_edge_id = project.next_edge_id.get();
    for mutation in prior_graph_mutations {
        match mutation {
            GraphMutation::InsertNode(input) => {
                next_node_id = next_node_id.max(input.id.0.saturating_add(1))
            }
            GraphMutation::InsertEdge(input) => {
                next_edge_id = next_edge_id.max(input.id.0.saturating_add(1))
            }
            _ => {}
        }
    }
    // Every route supplies current count statistics. Omitting them makes the planner collect
    // the whole graph on a plan-cache miss, including a host-routed point write.
    let optimizer_statistics = project.optimizer_statistics.get_or_init(|| {
        Arc::new(StatisticsSnapshot::count_summary(
            &project.graph,
            project.indexes.optimizer_generation(),
        ))
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
        text_embedding: project.indexes.profile().and_then(|profile| {
            text_embedding.filter(|embedding| embedding.profile() == profile.as_ref())
        }),
        parameters,
        bookmark,
        mutation_revision,
        resolved_time_nanos: unix_nanos(SystemTime::now())?,
        resolved_query_at_time_nanos: None,
        next_node_id,
        next_edge_id,
        predicate_versions: project.predicate_versions.clone(),
        capabilities,
        max_result_rows: HOST_QUERY_EXECUTION_ROW_ADDRESS_SPACE,
        max_batch_rows: QUERY_STREAM_BATCH_ROWS,
        optimizer_statistics: Some(optimizer_statistics.as_ref()),
        backend: None,
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
                    display_name: display_name.into(),
                    graph: GraphStore::default(),
                    temporal: TemporalStore::default(),
                    predicate_versions: BTreeMap::new(),
                    indexes: IndexCatalog::default(),
                    next_node_id: 1.into(),
                    next_edge_id: 1.into(),
                    authority_revision: bookmark.index.into(),
                    optimizer_statistics: OnceLock::new().into(),
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
            let project = project.as_ref();
            state
                .names
                .remove(&normalize_name(&project.display_name.get()));
            project.display_name.set(display_name);
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
            state
                .names
                .remove(&normalize_name(&project.display_name.get()));
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
            let project_state = project_state.as_ref();
            let changes_graph =
                !graph.is_empty() || !temporal.is_empty() || administrative.is_some();
            project_state
                .indexes
                .bind_catalog_graph(&project_state.graph);
            let freshness = embedding_freshness_updates(project_state, &graph)?;
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
            for (owner, index, previous_vector_revision) in freshness {
                if let (Some(index), Some(revision)) = (
                    project_state.indexes.vector_search_source(&index),
                    embedding_owner_dependencies(project_state, owner).0,
                ) {
                    index.0.advance_row_revision(
                        owner.entity_id,
                        previous_vector_revision,
                        revision,
                    );
                }
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
                project_state.indexes.apply_vector_mutation_from_graph(
                    &project_state.graph,
                    &retag_vector_mutation(mutation, bookmark.index),
                )?;
            }
            if changes_graph {
                project_state.authority_revision.set(bookmark.index);
                refresh_optimizer_statistics(project_state);
            }
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
            apply_security_command(
                &state.security,
                &command,
                bookmark,
                commit_time_nanos / 1_000_000,
            )?;
            Ok(None)
        }
    }
}

type StagedEmbeddingProfile = (
    Option<EmbeddingProfileActivation>,
    Option<(ProjectId, crate::graph::EmbeddingProfile)>,
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
            project_state.indexes.validate_activate_profile(&profile)?;
            activated = Some((*project, profile));
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
    if let Some((project, profile)) = activated {
        let project_state = state.projects.get_mut(&project).ok_or_else(|| {
            Error::new(ErrorCode::CorruptStorage, "activated project disappeared")
        })?;
        project_state.indexes.activate_profile(profile)?;
    }
    state.embedding_activation = activation;
    Ok(())
}

fn apply_security_command(
    security: &SecurityState,
    command: &SecurityCommand,
    bookmark: Bookmark,
    now_millis: i64,
) -> Result<()> {
    match command {
        SecurityCommand::RegisterCredential { record } => security
            .client_credentials()
            .register(record.clone(), now_millis)?,
        SecurityCommand::RotateCredential {
            old_fingerprint,
            replacement,
        } => security.client_credentials().rotate(
            *old_fingerprint,
            replacement.clone(),
            bookmark.index,
            now_millis,
        )?,
        SecurityCommand::RevokeCredential { fingerprint } => security
            .client_credentials()
            .revoke(*fingerprint, bookmark.index)?,
        SecurityCommand::CleanupExpired => {
            security.client_credentials().cleanup_expired(now_millis);
        }
    }
    Ok(())
}

fn validate_security_command(
    security: &SecurityState,
    command: &SecurityCommand,
    bookmark: Bookmark,
    now_millis: i64,
) -> Result<()> {
    let credentials = security.client_credentials();
    match command {
        SecurityCommand::RegisterCredential { record } => {
            credentials.validate_register(record, now_millis)
        }
        SecurityCommand::RotateCredential {
            old_fingerprint,
            replacement,
        } => credentials.validate_rotate(*old_fingerprint, replacement, bookmark.index, now_millis),
        SecurityCommand::RevokeCredential { fingerprint } => {
            credentials.validate_revoke(*fingerprint, bookmark.index)
        }
        SecurityCommand::CleanupExpired => Ok(()),
    }
}

fn validate_resolved_database_command(
    state: &DatabaseState,
    command: &WriteCommand,
    position: Bookmark,
) -> Result<DatabaseMutation> {
    command.validate()?;
    let mut mutation = decode_database_mutation(&command.payload)?;
    resolve_sequencer_values(&mut mutation, command.commit_time_millis)?;
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
    validate_sequencer_mutation(
        state,
        &mutation,
        position,
        millis_to_nanos(command.commit_time_millis)?,
    )?;
    Ok(mutation)
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
            graph,
            temporal,
            vectors,
            administrative,
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
            project_state
                .graph
                .validate_mutations_at_revision(graph, bookmark.index)?;
            crate::graph::knowledge::validate_batch(&project_state.graph, graph)?;
            project_state
                .indexes
                .validate_graph_mutations(&project_state.graph, graph)?;
            match administrative {
                Some(AdministrativeMutation::InitializeSemantic { profile, .. }) => project_state
                    .indexes
                    .validate_vector_mutations_with_planned_columns(
                        vectors,
                        Some(profile),
                        &[
                            crate::graph::SEMANTIC_NODE_PROPERTY,
                            crate::graph::SEMANTIC_RELATIONSHIP_PROPERTY,
                        ],
                    )?,
                Some(AdministrativeMutation::CreateEmbedding {
                    profile,
                    target_property,
                    ..
                }) => project_state
                    .indexes
                    .validate_vector_mutations_with_planned_columns(
                        vectors,
                        Some(profile),
                        &[*target_property],
                    )?,
                _ => project_state.indexes.validate_vector_mutations(vectors)?,
            }
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
                validate_administrative(project_state, administrative, graph)?;
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
            validate_security_command(
                &state.security,
                command,
                bookmark,
                commit_time_nanos / 1_000_000,
            )?;
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
    // Checkpoints represent an exact committed WAL prefix. Pause mutation application while
    // streaming the canonical state; do not create an in-memory graph generation for encoding.
    // Canonical concurrent graph readers do not acquire this writer ordering gate.
    let _apply = database.0.apply.lock();
    let (state, segment_pins) = {
        database.ensure_apply_healthy()?;
        let state = database.0.state.read();
        if state.applied != bookmark {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "database state does not exactly match the requested snapshot bookmark",
            ));
        }
        let mut segment_pins = BTreeMap::<[u8; 32], (SegmentDescriptor, SegmentPin)>::new();
        for descriptor in state.broker.payload_segments() {
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
        (state, segment_pins)
    };
    let state_path = snapshot_temporary_path(destination, "state")?;
    let attachment_directory = snapshot_attachment_staging_directory(destination)?;
    let result = (|| -> Result<BackendSnapshot> {
        let mut state_file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&state_path)?;
        let mut output = BufWriter::with_capacity(64 * 1024, &mut state_file);
        ciborium::ser::into_writer(&*state, &mut output).map_err(|error| {
            Error::invalid_data(format!(
                "database checkpoint state encoding failed: {error}"
            ))
        })?;
        output.flush()?;
        drop(output);
        crate::storage::sync_durable(&state_file)?;
        let state_bytes = state_file.metadata()?.len();
        drop(state_file);
        let mut manifest = DatabaseCheckpointManifest {
            format_version: DATABASE_SNAPSHOT_FORMAT,
            store_id: database.0.store_id,
            included: bookmark,
            state_bytes,
            state_checksum: None,
            broker_segments: state.broker.payload_segments(),
        };
        let mut state_hash = checkpoint_state_hasher(&manifest)?;
        state_hash.update_reader(File::open(&state_path)?)?;
        manifest.state_checksum = Some(*state_hash.finalize().as_bytes());
        let manifest_bytes = encode_database_value(&manifest, "database checkpoint manifest")?;
        if manifest_bytes.is_empty() {
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
    let (mut file, manifest, manifest_len) = open_database_checkpoint(snapshot_path)?;
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
    verify_checkpoint_state_checksum(&mut file, &manifest)?;
    let mut state_reader = (&mut file).take(manifest.state_bytes);
    let state: DatabaseState = ciborium::de::from_reader(&mut state_reader).map_err(|error| {
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
    // Graph deserialization validates canonical structure. The database owns metadata and
    // attachment consistency; it must not rescan every decoded graph again.
    validate_database_checkpoint_metadata(&state)?;
    let expected_broker_segments = state.broker.payload_segments();
    if expected_broker_segments != manifest.broker_segments {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "database checkpoint broker payload segment set is incomplete",
        ));
    }
    let broker_segments = expected_broker_segments;
    validate_database_checkpoint_state_extent(
        &manifest,
        file.get_ref().metadata()?.len(),
        manifest_len,
    )?;
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
    *current = state;
    database.publish_reader_registry(&current);
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
    let ordered = database.ordered_overlay.lock();
    if !ordered.reservations.is_empty() {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "cannot install a snapshot over ordered command reservations",
        ));
    }
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

fn validate_database_checkpoint_metadata(state: &DatabaseState) -> Result<()> {
    for project in state.projects.values() {
        project.indexes.bind_catalog_graph(&project.graph);
    }
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
        if state
            .names
            .get(&normalize_name(&project.display_name.get()))
            != Some(id)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "checkpoint project name index is inconsistent",
            ));
        }
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

fn open_database_checkpoint(
    path: &Path,
) -> Result<(BufReader<File>, DatabaseCheckpointManifest, usize)> {
    let mut file = BufReader::with_capacity(64 * 1024, File::open(path)?);
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
    if manifest_len == 0
        || manifest_len as u64 > file.get_ref().metadata()?.len().saturating_sub(12)
    {
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
    validate_database_checkpoint_state_extent(
        &manifest,
        file.get_ref().metadata()?.len(),
        manifest_len,
    )?;
    Ok((file, manifest, manifest_len))
}

fn checkpoint_state_hasher(manifest: &DatabaseCheckpointManifest) -> Result<blake3::Hasher> {
    let mut metadata = manifest.clone();
    metadata.state_checksum = None;
    let mut hash = blake3::Hasher::new_derive_key("irongraph.checkpoint-state.v1");
    hash.update(&encode_database_value(
        &metadata,
        "checkpoint checksum metadata",
    )?);
    Ok(hash)
}

fn verify_checkpoint_state_checksum(
    file: &mut BufReader<File>,
    manifest: &DatabaseCheckpointManifest,
) -> Result<()> {
    let Some(expected) = manifest.state_checksum else {
        return Ok(());
    };
    let start = file.stream_position()?;
    let mut hash = checkpoint_state_hasher(manifest)?;
    hash.update_reader((&mut *file).take(manifest.state_bytes))?;
    if hash.finalize().as_bytes() != &expected {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "checkpoint state checksum differs",
        ));
    }
    file.seek(SeekFrom::Start(start))?;
    Ok(())
}

fn standalone_snapshot_is_complete(path: &Path) -> Result<bool> {
    let (mut file, manifest, _) = open_database_checkpoint(path)?;
    // A file without a persisted checksum can be recovered by the full reader, but cannot
    // authorize online WAL compaction. Never rebuild a second graph to probe its readiness.
    if manifest.state_checksum.is_none() {
        return Ok(false);
    }
    verify_checkpoint_state_checksum(&mut file, &manifest)?;
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
    let (_, manifest, _) = open_database_checkpoint(path)?;
    Ok((manifest.included, manifest.broker_segments))
}

/// Retains the newly published snapshot and one validated predecessor. The WAL is compacted only
/// through that predecessor, so either retained snapshot is a real recovery authority rather than
/// a stale file whose suffix has already been discarded. Invalid snapshots are renamed for
/// forensic inspection; only older validated generations are pruned.
fn prune_standalone_snapshots(dir: &Path, keep: &Path) {
    let mut retained = BTreeSet::from([keep.to_owned()]);
    let mut verified = 1;
    for path in standalone_snapshot_paths(dir) {
        if path == keep {
            continue;
        }
        if open_database_checkpoint(&path)
            .is_ok_and(|(_, manifest, _)| manifest.state_checksum.is_none())
        {
            // Keep existing recovery authorities until two checksummed checkpoints exist.
            // Probing a file must not instantiate its graph beside the running database.
            if verified < 2 {
                retained.insert(path);
            } else {
                let _ = std::fs::remove_file(&path);
            }
            continue;
        }
        match standalone_snapshot_is_complete(&path) {
            Ok(true) if verified < 2 => {
                retained.insert(path);
                verified += 1;
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
    // Retention changes update both indexes and byte accounting locally. Full validation
    // belongs to checkpoint/recovery, rather than scanning every unchanged cached response
    // after each write.
    Ok(())
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
    for (request_id, record) in &state.request_results {
        if record.intent_digest == [0; 32]
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

fn refresh_optimizer_statistics(project: &ProjectState) {
    let generation = project.indexes.optimizer_generation();
    if let Some(statistics) = project.optimizer_statistics.get() {
        let mut updated = statistics.as_ref().clone();
        updated.refresh_counts(&project.graph, generation);
        project
            .optimizer_statistics
            .0
            .store(Some(Arc::new(updated)));
    } else {
        let _ = project
            .optimizer_statistics
            .set(Arc::new(StatisticsSnapshot::count_summary(
                &project.graph,
                generation,
            )));
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

fn validate_administrative(
    project: &ProjectState,
    mutation: &AdministrativeMutation,
    planned_graph: &[GraphMutation],
) -> Result<()> {
    let catalog = project.graph.catalog();
    let label_id = |name: &str| {
        catalog
            .label(name)
            .ok_or_else(|| Error::new(ErrorCode::QueryType, "label is not declared"))
    };
    let property_id = |name: &str| {
        catalog
            .property(name)
            .ok_or_else(|| Error::new(ErrorCode::QueryType, "property is not declared"))
    };
    match mutation {
        AdministrativeMutation::InitializeSemantic {
            profile,
            graph_revision,
        } => {
            if project.graph.revision() != *graph_revision {
                return Err(Error::retryable(
                    ErrorCode::TransactionConflict,
                    "graph changed during semantic initialization",
                    None,
                ));
            }
            project.indexes.validate_initialize_semantic(profile)
        }
        AdministrativeMutation::CreateIndex {
            name,
            kind,
            label,
            properties,
        } => {
            let kind = match kind.to_ascii_uppercase().as_str() {
                "EQUALITY" => crate::graph::GraphIndexKind::Equality,
                "RANGE" => crate::graph::GraphIndexKind::Range,
                "TEXT" => crate::graph::GraphIndexKind::Text,
                "VECTOR" => crate::graph::GraphIndexKind::Vector,
                _ => return Err(Error::new(ErrorCode::QueryType, "unknown index family")),
            };
            project.indexes.validate_create(
                &project.graph,
                &crate::graph::GraphIndexDefinition {
                    name: name.clone(),
                    kind,
                    label: label_id(label)?,
                    properties: properties
                        .iter()
                        .map(|name| property_id(name))
                        .collect::<Result<Vec<_>>>()?,
                    unique: false,
                },
            )
        }
        AdministrativeMutation::CreateConstraint {
            name,
            label,
            property,
        } => project.indexes.validate_create(
            &project.graph,
            &crate::graph::GraphIndexDefinition {
                name: name.clone(),
                kind: crate::graph::GraphIndexKind::Equality,
                label: label_id(label)?,
                properties: vec![property_id(property)?],
                unique: true,
            },
        ),
        AdministrativeMutation::RebuildIndex { name } => {
            if project.indexes.contains(name) {
                Ok(())
            } else {
                Err(Error::new(
                    ErrorCode::IndexUnavailable,
                    "index does not exist",
                ))
            }
        }
        AdministrativeMutation::DropIndex { name, if_exists } => {
            match project.indexes.validate_drop_index(name) {
                Err(error) if *if_exists && error.code == ErrorCode::IndexUnavailable => Ok(()),
                result => result,
            }
        }
        AdministrativeMutation::DropConstraint { name, if_exists } => {
            match project.indexes.validate_drop_constraint(name) {
                Err(error) if *if_exists && error.code == ErrorCode::IndexUnavailable => Ok(()),
                result => result,
            }
        }
        AdministrativeMutation::DeclareTemporal {
            entity_kind,
            label_or_type,
            property,
            scalar_type,
            retention_nanos,
        } => {
            let target = match entity_kind {
                crate::types::EntityKind::Node => label_id(label_or_type)?.0,
                crate::types::EntityKind::Relationship => {
                    catalog
                        .relationship_type(label_or_type)
                        .ok_or_else(|| {
                            Error::new(ErrorCode::QueryType, "relationship type is not declared")
                        })?
                        .0
                }
            };
            project
                .temporal
                .validate_declare(&crate::graph::TemporalDeclaration {
                    entity_kind: *entity_kind,
                    target,
                    property: property_id(property)?,
                    value_type: temporal_type(scalar_type)?,
                    retention_nanos: *retention_nanos,
                })
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
                            "unsupported rollup aggregate",
                        ));
                    }
                }
            }
            let mut window = if *hopping {
                crate::graph::WindowSpec::hopping(
                    *width_nanos,
                    every_nanos.ok_or_else(|| {
                        Error::new(ErrorCode::TemporalRange, "HOPPING requires EVERY")
                    })?,
                )
            } else {
                crate::graph::WindowSpec::tumbling(*width_nanos)
            };
            window.align_nanos = *align_nanos;
            window.timezone = timezone.clone();
            project
                .temporal
                .validate_create_rollup(&crate::graph::TemporalRollupDefinition {
                    name: name.clone(),
                    entity_kind: crate::types::EntityKind::Node,
                    target: label_id(label)?.0,
                    property: property_id(property)?,
                    window,
                    aggregates: aggregate_set,
                })
        }
        AdministrativeMutation::CreateEmbedding {
            name,
            label,
            source_property,
            target_property,
            model,
            profile,
            rows,
        } => project
            .indexes
            .validate_create_embedding_with_planned_schema(
                &project.graph,
                &crate::graph::EmbeddingIndexDefinition {
                    name: name.clone(),
                    label: *label,
                    source_property: *source_property,
                    target_property: *target_property,
                    model: model.clone(),
                },
                profile,
                rows,
                planned_graph,
            ),
    }
}

fn apply_administrative(
    project: &ProjectState,
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
    project.optimizer_statistics.clear();
    Ok(())
}

fn administrative_mutation(
    statement: Option<Statement>,
    project: &ProjectState,
    graph_mutations: &mut Vec<GraphMutation>,
    text_embedding: Option<&dyn TextEmbedding>,
    _revision: u64,
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
            let profile = project
                .indexes
                .profile()
                .map(|profile| (*profile).clone())
                .ok_or_else(|| {
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
                    let next = graph_mutations
                        .iter()
                        .filter_map(|mutation| match mutation {
                            GraphMutation::DeclareProperty { id, .. } => {
                                Some(id.0.saturating_add(1))
                            }
                            _ => None,
                        })
                        .max()
                        .unwrap_or(0)
                        .max(project.graph.catalog().next_property_id());
                    let property = crate::types::PropertyId(next);
                    graph_mutations.push(GraphMutation::DeclareProperty {
                        name: definition.target_property.clone(),
                        id: property,
                    });
                    property
                }
            };
            for node in project
                .graph
                .nodes()
                .filter(|node| node.labels().contains(&label))
            {
                if let Some(value) = node.property(source_property) {
                    if !matches!(value, ScalarValue::String(_) | ScalarValue::Null) {
                        return Err(Error::new(
                            ErrorCode::QueryType,
                            "embedding source property contains a non-string value",
                        ));
                    }
                }
            }
            let rows = Vec::new();
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
            labels: catalog.labels().map(|(_, name)| name.to_string()).collect(),
            relationship_types: catalog
                .relationship_types()
                .map(|(_, name)| name.to_string())
                .collect(),
            properties: catalog
                .properties()
                .map(|(_, name)| name.to_string())
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
        if current.is_none() && *revision == 0 {
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
        let changed = project.predicate_versions.get(stamp).map_or(
            project.authority_revision.get() > planning_index,
            |current| current != revision,
        );
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

fn update_next_ids(project: &ProjectState, mutation: &GraphMutation) {
    match mutation {
        GraphMutation::InsertNode(node) => {
            project.next_node_id.advance(node.id.0.saturating_add(1))
        }
        GraphMutation::InsertEdge(edge) => {
            project.next_edge_id.advance(edge.id.0.saturating_add(1))
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
    let needs_normalization = match mutation {
        DatabaseMutation::Graph { temporal, .. } => temporal.iter().any(|row| row.uses_commit_time),
        DatabaseMutation::Broker { .. } => true,
        _ => false,
    };
    let normalized = if needs_normalization {
        let mut normalized = mutation.clone();
        resolve_sequencer_values(&mut normalized, 0)?;
        Some(normalized)
    } else {
        None
    };
    let mut hasher = blake3::Hasher::new_derive_key("irongraph.request-intent.v1");
    {
        let mut output = BufWriter::with_capacity(16 * 1024, IntentHashOutput(&mut hasher));
        ciborium::ser::into_writer(normalized.as_ref().unwrap_or(mutation), &mut output).map_err(
            |error| {
                Error::invalid_data(format!(
                    "idempotent request intent encoding failed: {error}"
                ))
            },
        )?;
        output.flush()?;
    }
    Ok(*hasher.finalize().as_bytes())
}

struct IntentHashOutput<'a>(&'a mut blake3::Hasher);
impl Write for IntentHashOutput<'_> {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        self.0.update(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
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
        Ok(_) if DATABASE_BLOCKING_WORKER.with(std::cell::Cell::get) => {
            binding.handle.block_on(future)
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
        graph::{DocumentItem, EdgeInput, LayerMask, NodeInput},
        storage::{SegmentFamily, SegmentRecord},
        types::DocumentList,
    };

    use super::*;

    fn apply_canonical_test_command(
        state: &mut DatabaseState,
        command: &WriteCommand,
        position: Bookmark,
        segments: &SegmentStore,
    ) -> Result<()> {
        let mutation = validate_resolved_database_command(state, command, position)?;
        let digest = request_intent_digest(&mutation)?;
        if let Some(record) = command
            .request_id
            .and_then(|id| state.request_results.get(&id))
        {
            let response = record.response.clone();
            state.last_response = response;
            state.applied = position;
            return Ok(());
        }
        let reply = apply_mutation(
            state,
            mutation,
            position,
            millis_to_nanos(command.commit_time_millis)?,
            Some(segments),
        )?;
        let response = encode_database_value(&reply, "canonical test apply response")?;
        state.last_response = response.clone();
        state.applied = position;
        if let Some(id) = command.request_id {
            retain_request_result(state, id, digest, response)?;
        }
        prune_request_results(state)
    }

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
        state: &impl Serialize,
        included: Bookmark,
        destination: &Path,
    ) -> Result<BackendSnapshot> {
        let mut state_bytes = Vec::new();
        ciborium::ser::into_writer(state, &mut state_bytes).map_err(|error| {
            Error::invalid_data(format!("test checkpoint state encoding failed: {error}"))
        })?;
        let mut manifest = DatabaseCheckpointManifest {
            format_version: DATABASE_SNAPSHOT_FORMAT,
            store_id: database.0.store_id,
            included,
            state_bytes: state_bytes.len() as u64,
            state_checksum: None,
            broker_segments: Vec::new(),
        };
        let mut state_hash = checkpoint_state_hasher(&manifest)?;
        state_hash.update(&state_bytes);
        manifest.state_checksum = Some(*state_hash.finalize().as_bytes());
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
    async fn snapshot_readiness_streams_dirty_state_and_detects_body_corruption() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let snapshots = tempfile::tempdir()?;
        let identity = NodeIdentity::generate_genesis().public();
        let database =
            Database::open_backend(directory.path(), usize::MAX, Duration::ZERO, identity)?;
        let (project, entry) = wal_test_project_entry(1)?;
        database.apply_mutation(&entry).await?;
        let graph = database.0.state.read().projects[&project].graph.clone();
        let body = graph.catalog().intern_property("body")?;
        for id in 1..=64 {
            graph.insert_node(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: 1,
                labels: vec![],
                properties: vec![(
                    body,
                    ScalarValue::String(
                        format!("{id}:{}", "complete domain content ".repeat(8192)).into(),
                    ),
                )],
            })?;
        }
        database.0.state.read().projects[&project]
            .next_node_id
            .advance(65);
        let bytes = graph.resident_bytes();
        database.standalone_snapshot(snapshots.path()).await?;
        let path = standalone_snapshot_paths(snapshots.path()).remove(0);
        for _ in 0..8 {
            assert!(standalone_snapshot_is_complete(&path)?);
        }
        assert_eq!(graph.resident_bytes(), bytes);
        let (mut file, manifest, _) = open_database_checkpoint(&path)?;
        assert!(manifest.state_checksum.is_some());
        let position = file.stream_position()?;
        let mut altered_metadata = manifest.clone();
        altered_metadata.included.index += 1;
        assert_eq!(
            verify_checkpoint_state_checksum(&mut file, &altered_metadata)
                .unwrap_err()
                .code,
            ErrorCode::CorruptStorage
        );
        drop(file);
        let (mut original, mut unsigned_manifest, _) = open_database_checkpoint(&path)?;
        unsigned_manifest.state_checksum = None;
        let older = snapshots
            .path()
            .join("snapshot-00000000000000000001-00000000000000000000.igdb");
        let metadata = encode_database_value(&unsigned_manifest, "test existing checkpoint")?;
        let mut output = BufWriter::new(File::create(&older)?);
        output.write_all(&DATABASE_SNAPSHOT_MAGIC)?;
        output.write_all(&(metadata.len() as u32).to_be_bytes())?;
        output.write_all(&metadata)?;
        std::io::copy(&mut (&mut original).take(manifest.state_bytes), &mut output)?;
        output.flush()?;
        drop(output);
        assert!(!standalone_snapshot_is_complete(&older)?);
        prune_standalone_snapshots(snapshots.path(), &path);
        assert!(
            older.exists(),
            "existing recovery authority must remain available"
        );
        let mut writable = OpenOptions::new().read(true).write(true).open(&path)?;
        writable.seek(SeekFrom::Start(position + manifest.state_bytes - 1))?;
        let mut byte = [0];
        writable.read_exact(&mut byte)?;
        writable.seek(SeekFrom::Current(-1))?;
        writable.write_all(&[byte[0] ^ 1])?;
        drop(writable);
        assert_eq!(
            standalone_snapshot_is_complete(&path).unwrap_err().code,
            ErrorCode::CorruptStorage
        );
        assert_eq!(graph.resident_bytes(), bytes);
        let restored_directory = tempfile::tempdir()?;
        let restored = Database::open_backend(
            restored_directory.path(),
            usize::MAX,
            Duration::ZERO,
            identity,
        )?;
        assert_eq!(
            restored.standalone_recover(snapshots.path()).await?,
            Some(Bookmark { term: 1, index: 1 })
        );
        assert_eq!(
            restored.0.state.read().projects[&project]
                .graph
                .node_count(),
            64
        );
        Ok(())
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
            let (_, entry) = wal_test_project_entry(index)?;
            database.apply_mutation(&entry).await?;
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
            let (_, entry) = wal_test_project_entry(index)?;
            database.apply_mutation(&entry).await?;
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
        let project_state = ordered_graph_project(project, 1)?;
        {
            let project_state = project_state.as_ref();
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
            project_state.next_node_id.set(3);
            project_state.next_edge_id.set(2);
        }
        let bookmark = Bookmark { term: 1, index: 3 };
        let mut state = DatabaseState {
            applied: bookmark,
            ..DatabaseState::default()
        };
        state
            .names
            .insert(normalize_name(&project_state.display_name.get()), project);
        state.projects.insert(project, project_state);
        validate_database_checkpoint_metadata(&state)?;

        let mut encoded_bytes = Vec::new();
        ciborium::ser::into_writer(&state, &mut encoded_bytes)
            .map_err(|e| Error::internal(e.to_string()))?;
        let mut encoded: ciborium::value::Value =
            ciborium::de::from_reader(encoded_bytes.as_slice())
                .map_err(|e| Error::internal(e.to_string()))?;
        fn field<'a>(
            value: &'a mut ciborium::value::Value,
            key: &str,
        ) -> Result<&'a mut ciborium::value::Value> {
            let ciborium::value::Value::Map(entries) = value else {
                return Err(Error::internal("corruption fixture expected map"));
            };
            entries
                .iter_mut()
                .find_map(|(name, value)| {
                    matches!(name,ciborium::value::Value::Text(text) if text==key).then_some(value)
                })
                .ok_or_else(|| Error::internal(format!("corruption fixture missing {key}")))
        }
        let ciborium::value::Value::Map(projects) = field(&mut encoded, "projects")? else {
            return Err(Error::internal("project fixture expected map"));
        };
        let project_value = &mut projects
            .first_mut()
            .ok_or_else(|| Error::internal("project fixture absent"))?
            .1;
        let ciborium::value::Value::Array(edges) = field(field(project_value, "graph")?, "edges")?
        else {
            return Err(Error::internal("edge fixture expected array"));
        };
        *field(
            edges
                .first_mut()
                .ok_or_else(|| Error::internal("edge fixture absent"))?,
            "source",
        )? = ciborium::value::Value::Integer(u32::MAX.into());
        let snapshot_path = source_directory.path().join("corrupt.igdb");
        let snapshot = write_unchecked_test_snapshot(&source, &encoded, bookmark, &snapshot_path)?;
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

    /// Publication keeps cold samples while advancing the exact canonical count revision.
    #[test]
    fn a_publication_keeps_the_statistics_samples() {
        let project = ProjectState {
            id: ProjectId::random(),
            display_name: ("rebind".to_owned()).into(),
            graph: GraphStore::default(),
            temporal: TemporalStore::default(),
            predicate_versions: BTreeMap::new(),
            indexes: IndexCatalog::default(),
            next_node_id: (1).into(),
            next_edge_id: (1).into(),
            authority_revision: (1).into(),
            optimizer_statistics: OnceLock::new().into(),
        };
        let mut collected = StatisticsSnapshot::collect_project(&project.graph, None, None);
        collected.graph_revision = 1;
        collected.schema_generation = project.graph.catalog().optimizer_generation();
        collected.index_generation = project.indexes.optimizer_generation();
        let _ = project.optimizer_statistics.set(Arc::new(collected));

        refresh_optimizer_statistics(&project);
        assert!(
            project.optimizer_statistics.get().is_some(),
            "publication must preserve the statistics cache"
        );
    }

    #[test]
    fn caller_output_limit_does_not_bound_intermediate_execution_rows() -> Result<()> {
        let project = ProjectId::random();
        let snapshot = ProjectState {
            id: project,
            display_name: ("result-policy-test".to_owned()).into(),
            graph: GraphStore::default(),
            temporal: TemporalStore::default(),
            predicate_versions: BTreeMap::new(),
            indexes: IndexCatalog::default(),
            next_node_id: (1).into(),
            next_edge_id: (1).into(),
            authority_revision: (1).into(),
            optimizer_statistics: OnceLock::new().into(),
        };
        let request = QueryRequest {
            request_id: Uuid::new_v4(),
            project_id: Some(project),
            query: "UNWIND range(1, 129) AS i RETURN count(*) AS count".to_owned(),
            parameters: BTreeMap::new(),
            consistency: CommitAcknowledgement::Published,
            bookmark: None,
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
            &[],
            &[],
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
            display_name: ("memory".to_owned()).into(),
            graph: GraphStore::default(),
            temporal: TemporalStore::default(),
            predicate_versions: BTreeMap::new(),
            indexes: IndexCatalog::default(),
            next_node_id: (1).into(),
            next_edge_id: (1).into(),
            authority_revision: (0).into(),
            optimizer_statistics: OnceLock::new().into(),
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
            &[],
            &[],
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
        let graph = GraphStore::default();
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
            display_name: ("memory".to_owned()).into(),
            graph,
            temporal: TemporalStore::default(),
            predicate_versions: BTreeMap::new(),
            indexes: IndexCatalog::default(),
            next_node_id: (2).into(),
            next_edge_id: (1).into(),
            authority_revision: (1).into(),
            optimizer_statistics: OnceLock::new().into(),
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
            &[],
            &[],
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
    fn knowledge_layer_lock_holds_on_the_transaction_validation_path() -> Result<()> {
        let project = ProjectId::random();
        let graph = GraphStore::default();
        let product = graph.catalog_mut().intern_label("Product")?;
        let name = graph.catalog_mut().intern_property("name")?;
        let current = ProjectState {
            id: project,
            display_name: ("knowledge-lock".to_owned()).into(),
            graph,
            temporal: TemporalStore::default(),
            predicate_versions: BTreeMap::new(),
            indexes: IndexCatalog::default(),
            next_node_id: (1).into(),
            next_edge_id: (1).into(),
            authority_revision: (0).into(),
            optimizer_statistics: OnceLock::new().into(),
        };
        // A KNOWLEDGE node without the reserved `name` is rejected at the durable write funnel.
        let anonymous = vec![GraphMutation::InsertNode(NodeInput {
            id: NodeId(1),
            layer: Layer::Knowledge,
            revision: 1,
            labels: vec![product],
            properties: vec![],
        })];
        assert!(crate::graph::knowledge::validate_batch(&current.graph, &anonymous).is_err());
        // The very same shape on the OBSERVED layer is unconstrained business data.
        let observed = vec![GraphMutation::InsertNode(NodeInput {
            id: NodeId(1),
            layer: Layer::Observed,
            revision: 1,
            labels: vec![product],
            properties: vec![],
        })];
        crate::graph::knowledge::validate_batch(&current.graph, &observed)?;
        // A named KNOWLEDGE node is accepted and staged.
        let named = vec![GraphMutation::InsertNode(NodeInput {
            id: NodeId(1),
            layer: Layer::Knowledge,
            revision: 1,
            labels: vec![product],
            properties: vec![(name, ScalarValue::String(Arc::from("Zorbex Q7")))],
        })];
        current.graph.validate_mutations(&named)?;
        crate::graph::knowledge::validate_batch(&current.graph, &named)?;
        for mutation in named {
            current.graph.apply(mutation)?;
        }
        assert_eq!(
            current
                .graph
                .node_count_in_layers(crate::graph::LayerMask::KNOWLEDGE),
            1
        );
        Ok(())
    }

    fn transaction_resource(project: ProjectId, bookmark: Bookmark) -> Arc<TransactionResource> {
        let snapshot = Arc::new(ProjectState {
            id: project,
            display_name: ("transaction".to_owned()).into(),
            graph: GraphStore::default(),
            temporal: TemporalStore::default(),
            predicate_versions: BTreeMap::new(),
            indexes: IndexCatalog::default(),
            next_node_id: (1).into(),
            next_edge_id: (1).into(),
            authority_revision: (0).into(),
            optimizer_statistics: OnceLock::new().into(),
        });
        Arc::new(TransactionResource {
            terminal: AtomicU8::new(0),
            lifecycle: Mutex::new(TransactionLifecycle::Active(TransactionState {
                catalog: snapshot.graph.catalog().clone(),
                working: Arc::clone(&snapshot),
                bookmark,
                consistency: CommitAcknowledgement::Published,
                batches: Vec::new(),
                accounted_bytes: MIN_TRANSACTION_ACCOUNTED_BYTES,
            })),
        })
    }

    #[test]
    fn surgical_transaction_batch_append_shares_prior_payloads_and_accounts_only_new_batch()
    -> Result<()> {
        for batches in [4_096, 32_768] {
            let mut log = Vec::with_capacity(batches as usize + 1);
            for revision in 0..batches {
                log.push(Arc::new(TransactionBatch {
                    dependencies: TransactionDependencies::default(),
                    graph_mutations: vec![],
                    temporal_mutations: vec![],
                    vector_mutations: vec![ResolvedVectorMutation::Upsert {
                        property: crate::types::PropertyId(1),
                        entity_id: revision,
                        coordinates: (0..384)
                            .map(|coordinate| ((revision + coordinate * 3571) % 65521) as u16)
                            .collect(),
                        revision,
                    }],
                }));
            }
            let original = log.clone();
            let journal_allocation = log.as_ptr();
            let batch = Arc::new(TransactionBatch {
                dependencies: TransactionDependencies::default(),
                graph_mutations: vec![],
                temporal_mutations: vec![],
                vector_mutations: vec![],
            });
            let added = transaction_batch_bytes(&batch, 1024)?;
            assert!(added <= 1024);
            log.push(batch);
            assert!(Arc::ptr_eq(&original[0], &log[0]));
            assert!(Arc::ptr_eq(
                &original[batches as usize - 1],
                &log[batches as usize - 1]
            ));
            assert_eq!(original.len(), batches as usize);
            assert_eq!(
                journal_allocation,
                log.as_ptr(),
                "appending one intent does not copy retained payloads or allocate a graph version"
            );
        }
        Ok(())
    }

    #[test]
    fn surgical_initialized_semantic_growth_never_starts_a_cold_build() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let database = Database::open_backend(
            directory.path(),
            2 * 1024 * 1024,
            Duration::from_secs(1),
            NodeIdentity::generate_genesis().public(),
        )?;
        let profile = crate::graph::EmbeddingProfile::new(
            [1; 32],
            [2; 32],
            2,
            crate::graph::EmbeddingDType::F16,
            true,
            crate::graph::Similarity::Cosine,
        )?;
        *database.0.text_embedding.write() = Some(Arc::new(WindowEmbedding {
            profile: profile.clone(),
            batches: Arc::new(Mutex::new(vec![])),
        }));
        let project = ProjectId::random();
        let state = ordered_graph_project(project, 1)?;
        let indexes = &state.indexes;
        indexes.initialize_semantic(profile.clone())?;
        let coordinates = profile.quantize(&[1.0, 0.0])?;
        for entity_id in 0..1024 {
            indexes.apply_vector_mutation(&ResolvedVectorMutation::Upsert {
                property: crate::graph::SEMANTIC_NODE_PROPERTY,
                entity_id,
                coordinates: coordinates.clone(),
                revision: 1,
            })?;
        }
        indexes.rebuild_vectors_with(|source, mut config| {
            config.coarse_centroids = 2;
            config.subquantizers = 1;
            config.bits_per_code = 1;
            config.probes = 1;
            config.iterations = 1;
            crate::graph::IvfPqIndex::build(source, config)
        })?;
        for entity_id in 1024..2048 {
            indexes.apply_vector_mutation(&ResolvedVectorMutation::Upsert {
                property: crate::graph::SEMANTIC_NODE_PROPERTY,
                entity_id,
                coordinates: coordinates.clone(),
                revision: 2,
            })?;
        }
        assert!(indexes.semantic_rebuild_needed());
        database.0.state.write().projects.insert(project, state);
        database.publish_reader_registry(&database.0.state.read());
        let before = database.bookmark();
        // No runtime is bound: an attempted cold-build WAL command would fail this call.
        database.initialize_automatic_semantic(project)?;
        assert_eq!(database.bookmark(), before);
        Ok(())
    }

    fn ordered_graph_project(id: ProjectId, revision: u64) -> Result<Arc<ProjectState>> {
        let graph = GraphStore::default();
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
            display_name: (id.to_string()).into(),
            graph,
            temporal: TemporalStore::default(),
            predicate_versions: BTreeMap::new(),
            indexes: IndexCatalog::default(),
            next_node_id: (2).into(),
            next_edge_id: (1).into(),
            authority_revision: (revision).into(),
            optimizer_statistics: OnceLock::new().into(),
        }))
    }

    #[test]
    fn unchanged_project_publication_keeps_names_visible_to_parallel_readers() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let database = Database::open_backend(
            directory.path(),
            2 * 1024 * 1024,
            Duration::from_secs(1),
            NodeIdentity::generate_genesis().public(),
        )?;
        let id = ProjectId::random();
        let project = ordered_graph_project(id, 1)?;
        project.display_name.set("stable".to_owned());
        let body: Arc<str> = Arc::from("complete source ".repeat(131_072));
        let property = project.graph.catalog_mut().intern_property("body")?;
        project.graph.set_node_property(
            NodeId(1),
            property,
            ScalarValue::String(Arc::clone(&body)),
            2,
        )?;
        let bytes = project.graph.resident_bytes();
        let owners = Arc::strong_count(&body);
        database
            .0
            .state
            .write()
            .projects
            .insert(id, Arc::clone(&project));
        database.publish_reader_registry(&database.0.state.read());
        let start = std::sync::Barrier::new(3);
        std::thread::scope(|scope| -> Result<()> {
            let readers = (0..2)
                .map(|_| {
                    scope.spawn(|| -> Result<()> {
                        start.wait();
                        for _ in 0..30_000 {
                            assert_eq!(database.resolve_project_name("stable")?, id);
                            let (live, _) = database.capture_canonical_project(id)?;
                            assert!(Arc::ptr_eq(&live, &project));
                        }
                        Ok(())
                    })
                })
                .collect::<Vec<_>>();
            start.wait();
            let state = database.0.state.read();
            for iteration in 0..20_000 {
                if iteration % 2 == 0 {
                    database.publish_reader_project(&state, id);
                } else {
                    database.publish_reader_registry(&state);
                }
            }
            for reader in readers {
                reader
                    .join()
                    .map_err(|_| Error::internal("project reader panicked"))??;
            }
            Ok(())
        })?;
        assert_eq!(database.0.reader_projects.len(), 1);
        assert_eq!(database.0.reader_names.len(), 1);
        assert_eq!(project.graph.resident_bytes(), bytes);
        assert_eq!(Arc::strong_count(&body), owners);
        project.display_name.set("renamed".to_owned());
        database.publish_reader_registry(&database.0.state.read());
        assert_eq!(database.resolve_project_name("renamed")?, id);
        assert!(database.resolve_project_name("stable").is_err());
        assert_eq!(database.0.reader_names.len(), 1);
        database.0.state.write().projects.remove(&id);
        database.publish_reader_registry(&database.0.state.read());
        assert!(database.resolve_project_name("renamed").is_err());
        assert!(database.capture_canonical_project(id).is_err());
        assert!(database.0.reader_projects.is_empty());
        assert!(database.0.reader_names.is_empty());
        Ok(())
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
    fn independent_creates_planned_together_apply_without_false_conflicts() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let segments = SegmentStore::open(directory.path(), 2 * 1024 * 1024)?;
        let bookmark = Bookmark { term: 2, index: 10 };
        let id = ProjectId::random();
        let project = ordered_graph_project(id, bookmark.index)?;
        let request: QueryRequest = serde_json::from_value(serde_json::json!({
            "request_id": Uuid::new_v4(),
            "query": "UNWIND range(1,32) AS value CREATE (n:Parallel) SET n.value=value RETURN count(*)"
        })).map_err(|error| Error::internal(error.to_string()))?;
        let mut commands = Vec::new();
        for _ in 0..3 {
            let output =
                execute_on_project(&project, &request, bookmark, 11, full_capabilities(), None)?;
            assert!(output.dependencies.predicates.is_empty());
            let mutation = DatabaseMutation::Graph {
                project: id,
                validation: MutationValidation {
                    snapshot: bookmark,
                    dependencies: output.dependencies,
                },
                graph: output.graph_mutations,
                temporal: Vec::new(),
                vectors: Vec::new(),
                administrative: None,
            };
            commands.push(WriteCommand {
                kind: MutationKind::Graph,
                project_id: Some(id),
                request_id: Some(Uuid::new_v4()),
                commit_time_millis: 1,
                payload: encode_database_value(&mutation, "independent create test")?,
            });
        }
        let mut state = DatabaseState {
            applied: bookmark,
            ..DatabaseState::default()
        };
        state.projects.insert(id, project.clone());
        for (index, command) in commands.iter().enumerate() {
            apply_canonical_test_command(
                &mut state,
                command,
                Bookmark {
                    term: 2,
                    index: 11 + index as u64,
                },
                &segments,
            )?;
        }
        assert_eq!(project.graph.node_count(), 97);
        Ok(())
    }

    #[test]
    fn canonical_sequential_preflight_accepts_independent_projects_and_rejects_conflicts()
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
        let (first, _) = ordered_graph_command(first_project, snapshot, 11)?;
        apply_canonical_test_command(
            &mut state,
            &first,
            Bookmark { term: 3, index: 11 },
            &segments,
        )?;
        let (second, _) = ordered_graph_command(second_project, snapshot, 12)?;
        apply_canonical_test_command(
            &mut state,
            &second,
            Bookmark { term: 3, index: 12 },
            &segments,
        )?;
        assert_eq!(state.applied.index, 12);
        let (conflict, _) = ordered_graph_command(first_project, snapshot, 13)?;
        let error =
            validate_resolved_database_command(&state, &conflict, Bookmark { term: 3, index: 13 })
                .expect_err("stale entity revision accepted");
        assert_eq!(error.code, ErrorCode::TransactionConflict);
        assert_eq!(
            state.projects[&first_project]
                .graph
                .node(NodeId(1))
                .and_then(|n| n.property(crate::types::PropertyId(0))),
            Some(ScalarValue::Integer(11))
        );
        Ok(())
    }

    #[test]
    fn canonical_integer_aggregate_predicates_fence_derived_writes_per_project() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let segments = SegmentStore::open(directory.path(), 2 * 1024 * 1024)?;
        let bookmark = Bookmark { term: 2, index: 10 };
        let project_id = ProjectId::random();
        let unrelated_id = ProjectId::random();
        let project = ordered_graph_project(project_id, bookmark.index)?;
        let label = project.graph.catalog_mut().intern_label("Node")?;
        project.graph.apply(GraphMutation::AddNodeLabels {
            node: NodeId(1),
            labels: vec![label],
            revision: bookmark.index,
        })?;
        let mut state = DatabaseState {
            applied: bookmark,
            ..DatabaseState::default()
        };
        state.projects.insert(project_id, Arc::clone(&project));
        state.projects.insert(
            unrelated_id,
            ordered_graph_project(unrelated_id, bookmark.index)?,
        );
        assert!(project.predicate_versions.is_empty());
        let request = QueryRequest {
            request_id: Uuid::new_v4(),
            project_id: Some(project_id),
            query: "MATCH (n:Node) WHERE n.value >= 0 AND n.value < 2 \
                    RETURN count(n) AS count, sum(n.value) AS total"
                .to_owned(),
            parameters: BTreeMap::new(),
            consistency: CommitAcknowledgement::Published,
            bookmark: None,
            cancellation: Default::default(),
            deadline: None,
            connection_id: ConnectionId::new(),
        };
        let output = execute_on_project(
            &project,
            &request,
            bookmark,
            bookmark.index + 1,
            full_capabilities(),
            None,
        )?;
        let values = output
            .result
            .batches
            .iter()
            .flat_map(|batch| &batch.columns)
            .map(|column| (column.name.as_str(), column.values.as_slice()))
            .collect::<BTreeMap<_, _>>();
        assert_eq!(
            values.get("count").copied(),
            Some([ResultValue::Scalar(ScalarValue::Integer(1))].as_slice())
        );
        assert_eq!(
            values.get("total").copied(),
            Some([ResultValue::Scalar(ScalarValue::Integer(0))].as_slice())
        );
        assert!(
            output.dependencies.entities.is_empty(),
            "integer scans must use bounded predicate fencing instead of per-row entities"
        );
        assert!(!output.dependencies.predicates.is_empty());
        assert!(
            output
                .dependencies
                .predicates
                .values()
                .all(|revision| *revision == bookmark.index)
        );
        validate_dependencies(&output.dependencies, &project, bookmark.index)?;

        let derived_value = match (values.get("count").copied(), values.get("total").copied()) {
            (
                Some([ResultValue::Scalar(ScalarValue::Integer(count))]),
                Some([ResultValue::Scalar(ScalarValue::Integer(total))]),
            ) => count
                .checked_add(*total)
                .ok_or_else(|| Error::internal("aggregate fixture overflow"))?,
            _ => {
                return Err(Error::internal(
                    "aggregate fixture returned non-integer results",
                ));
            }
        };
        let (mut derived_write, mut derived_mutation) =
            ordered_graph_command(project_id, bookmark, derived_value)?;
        let DatabaseMutation::Graph { validation, .. } = &mut derived_mutation else {
            return Err(Error::internal(
                "aggregate fixture expected a graph command",
            ));
        };
        validation.dependencies = output.dependencies;
        validation
            .dependencies
            .write_targets
            .insert(crate::cypher::EntityDependency::Node(NodeId(1)));
        derived_write.payload =
            encode_database_value(&derived_mutation, "aggregate-derived test mutation")?;

        let (unrelated_write, _) = ordered_graph_command(unrelated_id, bookmark, 11)?;
        apply_canonical_test_command(
            &mut state,
            &unrelated_write,
            Bookmark { term: 2, index: 11 },
            &segments,
        )?;
        assert_eq!(project.authority_revision.get(), bookmark.index);
        validate_resolved_database_command(
            &state,
            &derived_write,
            Bookmark { term: 2, index: 12 },
        )?;

        let (intervening_write, _) = ordered_graph_command(project_id, bookmark, 12)?;
        apply_canonical_test_command(
            &mut state,
            &intervening_write,
            Bookmark { term: 2, index: 12 },
            &segments,
        )?;
        assert_eq!(project.authority_revision.get(), 12);
        assert!(project.predicate_versions.is_empty());
        let error = validate_resolved_database_command(
            &state,
            &derived_write,
            Bookmark { term: 2, index: 13 },
        )
        .expect_err("a stale aggregate-derived write bypassed the project predicate fence");
        assert_eq!(error.code, ErrorCode::TransactionConflict);
        assert_eq!(error.message, "transaction predicate changed");
        assert_eq!(
            project
                .graph
                .node(NodeId(1))
                .and_then(|node| node.property(crate::types::PropertyId(0))),
            Some(ScalarValue::Integer(12)),
            "rejected derived write must leave the intervening canonical value intact"
        );
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
        database.publish_reader_registry(&database.0.state.read());
        let (command, _) = ordered_graph_command(project, snapshot, 11)?;
        let reservation = database.reserve_command(&command, committed).await?;
        let retained = database.0.state.read().projects[&project].graph.clone();
        assert_eq!(
            retained.revision(),
            snapshot.index,
            "reservation must not mutate canonical rows"
        );
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

        let (different_command, _) = ordered_graph_command(project, snapshot, 12)?;
        let different_entry = MutationEntry::new(
            committed.term,
            committed.index,
            different_command.kind,
            different_command.project_id,
            different_command.request_id,
            different_command.commit_time_millis,
            different_command.payload,
        )?;
        let mismatch = database
            .apply_mutation(&different_entry)
            .await
            .expect_err("a different mutation must not reuse the reserved command's preflight");
        assert_eq!(mismatch.code, ErrorCode::CorruptStorage);
        assert_eq!(retained.revision(), snapshot.index);

        let applied = database.apply_mutation(&entry).await?;
        assert!(!applied.duplicate);
        let state = database.0.state.read();
        let canonical_project = state
            .projects
            .get(&project)
            .ok_or_else(|| Error::internal("committed graph project disappeared"))?;
        assert_eq!(retained.revision(), committed.index);
        assert_eq!(
            canonical_project
                .graph
                .node(NodeId(1))
                .and_then(|n| n.property(crate::types::PropertyId(0))),
            Some(ScalarValue::Integer(11))
        );
        drop(state);
        database
            .complete_command_reservation(reservation, CommandReservationOutcome::Applied)
            .await?;
        Ok(())
    }

    #[tokio::test]
    async fn retained_canonical_handles_observe_multiple_publications_without_generations()
    -> Result<()> {
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
            state
                .projects
                .insert(project, ordered_graph_project(project, 1)?);
            state.applied = initial;
        }
        database.publish_reader_registry(&database.0.state.read());
        let retained = database.capture_canonical_project(project)?.0;
        for index in 2..=4 {
            let (command, _) = ordered_graph_command(
                project,
                Bookmark {
                    term: 1,
                    index: index - 1,
                },
                index as i64,
            )?;
            let entry = MutationEntry::new(
                1,
                index,
                command.kind,
                command.project_id,
                command.request_id,
                command.commit_time_millis,
                command.payload,
            )?;
            database.apply_mutation(&entry).await?;
            assert_eq!(
                retained
                    .graph
                    .node(NodeId(1))
                    .and_then(|n| n.property(crate::types::PropertyId(0))),
                Some(ScalarValue::Integer(index as i64))
            );
            assert_eq!(retained.graph.revision(), index);
            assert!(Arc::ptr_eq(
                &retained,
                &database.capture_canonical_project(project)?.0
            ));
        }
        Ok(())
    }

    #[tokio::test]
    async fn ordered_graph_reservations_publish_their_exact_committed_rows() -> Result<()> {
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
        database.publish_reader_registry(&database.0.state.read());
        let (first_command, _) = ordered_graph_command(first_project, snapshot, 11)?;
        let (second_command, _) = ordered_graph_command(second_project, snapshot, 12)?;
        let first_position = Bookmark { term: 3, index: 11 };
        let second_position = Bookmark { term: 3, index: 12 };
        let first_reservation = database
            .reserve_command(&first_command, first_position)
            .await?;
        assert_eq!(database.0.ordered_overlay.lock().reservations.len(), 1);
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
            second_command.payload.clone(),
        )?;

        database.apply_mutation(&first_entry).await?;
        assert_eq!(database.0.state.read().applied, first_position);
        database
            .complete_command_reservation(first_reservation, CommandReservationOutcome::Applied)
            .await?;
        let second_reservation = database
            .reserve_command(&second_command, second_position)
            .await?;
        database.apply_mutation(&second_entry).await?;
        assert_eq!(database.0.state.read().applied, second_position);
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
        let (command, _) = ordered_graph_command(project, snapshot, 11)?;
        apply_canonical_test_command(
            &mut state,
            &command,
            Bookmark { term: 3, index: 11 },
            &segments,
        )?;
        let retained = state.projects[&project].graph.clone();
        let response = state.last_response.clone();
        let (mut reused, _) = ordered_graph_command(project, Bookmark { term: 3, index: 11 }, 12)?;
        reused.request_id = command.request_id;
        let error =
            validate_resolved_database_command(&state, &reused, Bookmark { term: 3, index: 12 })
                .expect_err("different mutation reused request ID");
        assert_eq!(error.code, ErrorCode::ProtocolViolation);
        assert!(error.message.contains("idempotency key"));
        apply_canonical_test_command(
            &mut state,
            &command,
            Bookmark { term: 3, index: 12 },
            &segments,
        )?;
        assert_eq!(state.applied.index, 12);
        assert_eq!(*state.last_response, *response);
        assert_eq!(
            retained
                .node(NodeId(1))
                .and_then(|n| n.property(crate::types::PropertyId(0))),
            Some(ScalarValue::Integer(11))
        );
        assert_eq!(
            retained.revision(),
            11,
            "identical replay did not mutate canonical rows"
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
            let state = database.0.state.write();
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
        database.publish_reader_registry(&database.0.state.read());

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
        database.publish_reader_registry(&database.0.state.read());

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
    async fn standalone_broker_publish_consumes_its_once_materialized_payload() -> Result<()> {
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
        database.publish_reader_registry(&database.0.state.read());
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
        let retained = database.0.state.read().broker.clone();
        assert!(
            retained.payload_segments().is_empty(),
            "reservation must not publish broker rows"
        );

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
        assert_eq!(
            retained.payload_segments().len(),
            1,
            "retained broker handle sees canonical publication"
        );
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
        database.publish_reader_registry(&database.0.state.read());
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
        let retained = database.0.pending_broker_segments.lock().len();
        assert!(database.0.state.read().broker.payload_segments().is_empty());
        database
            .complete_command_reservation(reservation, CommandReservationOutcome::Rejected)
            .await?;
        assert!(database.0.pending_broker_segments.lock().is_empty());
        assert_eq!(database.0.retired_broker_segments.lock().len(), retained);
        Ok(())
    }

    #[test]
    fn transaction_registry_bounds_bytes_shares_pins_and_expires_without_traffic() -> Result<()> {
        let registry = TransactionAdmissionRegistry::new(1_024)?;
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
        registry.register(first_id, connection, 256, pin, deadline, fence, &first)?;
        registry.register(second_id, connection, 256, pin, deadline, fence, &second)?;
        assert_eq!(registry.usage(), (2, 512, 1));
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
    fn canonical_reads_progress_while_writer_is_paused_between_record_changes() -> Result<()> {
        let project = ordered_graph_project(ProjectId::random(), 1)?;
        let graph = project.graph.clone();
        let property = graph
            .catalog()
            .property("value")
            .ok_or_else(|| Error::internal("value absent"))?;
        graph.insert_node(NodeInput {
            id: NodeId(2),
            layer: Layer::Observed,
            revision: 1,
            labels: vec![],
            properties: vec![(property, ScalarValue::Integer(0))],
        })?;
        let (published_tx, published_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let writer_graph = graph.clone();
        let writer = std::thread::spawn(move || -> Result<()> {
            writer_graph.set_node_property(NodeId(1), property, ScalarValue::Integer(1), 2)?;
            published_tx
                .send(())
                .map_err(|e| Error::internal(e.to_string()))?;
            release_rx
                .recv_timeout(Duration::from_secs(2))
                .map_err(|e| Error::internal(e.to_string()))?;
            writer_graph.set_node_property(NodeId(2), property, ScalarValue::Integer(1), 2)
        });
        published_rx
            .recv_timeout(Duration::from_secs(2))
            .map_err(|e| Error::internal(e.to_string()))?;
        assert_eq!(
            graph.node(NodeId(1)).and_then(|n| n.property(property)),
            Some(ScalarValue::Integer(1))
        );
        assert_eq!(
            graph.node(NodeId(2)).and_then(|n| n.property(property)),
            Some(ScalarValue::Integer(0)),
            "selected mixed multi-record visibility"
        );
        release_tx
            .send(())
            .map_err(|e| Error::internal(e.to_string()))?;
        writer
            .join()
            .map_err(|_| Error::internal("canonical writer panicked"))??;
        assert_eq!(
            project
                .graph
                .node(NodeId(2))
                .and_then(|n| n.property(property)),
            Some(ScalarValue::Integer(1))
        );
        Ok(())
    }

    #[test]
    fn transaction_journal_reads_pending_rows_without_mutating_canonical_store() -> Result<()> {
        let project = ProjectId::random();
        let bookmark = Bookmark { term: 3, index: 7 };
        let snapshot = ordered_graph_project(project, 7)?;
        let label = crate::types::LabelId(snapshot.graph.catalog().next_label_id());
        let mutations = vec![
            GraphMutation::DeclareLabel {
                name: "Pending".to_owned(),
                id: label,
            },
            GraphMutation::InsertNode(NodeInput {
                id: NodeId(2),
                layer: Layer::Observed,
                revision: 8,
                labels: vec![label],
                properties: vec![],
            }),
        ];
        let request:QueryRequest=serde_json::from_value(serde_json::json!({"request_id":Uuid::new_v4(),"project_id":project,"query":"MATCH (n) RETURN count(n)"})).map_err(|e|Error::internal(e.to_string()))?;
        let output = execute_on_project_inner(
            &snapshot,
            &request,
            bookmark,
            8,
            full_capabilities(),
            None,
            &mutations,
            &[],
            None,
        )?;
        assert_eq!(
            output.result.batches[0].columns[0].values[0],
            ResultValue::Scalar(ScalarValue::Integer(2))
        );
        assert_eq!(snapshot.graph.node_count(), 1);
        assert!(snapshot.graph.catalog().label("Pending").is_none());
        assert_eq!(snapshot.graph.revision(), 7);
        drop(mutations);
        assert_eq!(
            snapshot.graph.node_count(),
            1,
            "rollback drops journal intents only"
        );
        Ok(())
    }

    #[test]
    fn canonical_query_progresses_while_publication_metadata_writer_is_paused() -> Result<()> {
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
        database.publish_reader_registry(&database.0.state.read());
        let request:QueryRequest=serde_json::from_value(serde_json::json!({"request_id":Uuid::new_v4(),"project_id":project,"query":"MATCH (n) RETURN n.value"})).map_err(|e|Error::internal(e.to_string()))?;
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        std::thread::scope(|scope| -> Result<()> {
            let writer = database.0.state.write();
            let reader = scope.spawn(|| {
                database.execute_autocommit(request, &mut |event| {
                    if matches!(event, QueryStreamEvent::Batch { .. }) {
                        done_tx
                            .send(())
                            .map_err(|e| Error::internal(e.to_string()))?;
                    }
                    Ok(())
                })
            });
            let progressed = done_rx.recv_timeout(Duration::from_secs(2)).is_ok();
            drop(writer);
            reader
                .join()
                .map_err(|_| Error::internal("reader panicked"))??;
            assert!(
                progressed,
                "canonical query waited for publication metadata writer"
            );
            Ok(())
        })
    }

    #[test]
    fn paused_stream_readback_does_not_hold_publication_writer() -> Result<()> {
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
        database.publish_reader_registry(&database.0.state.read());
        let request = QueryRequest {
            request_id: Uuid::new_v4(),
            project_id: Some(project),
            query: "MATCH (n) RETURN n.value".to_owned(),
            parameters: BTreeMap::new(),
            consistency: CommitAcknowledgement::Published,
            bookmark: None,
            cancellation: Default::default(),
            deadline: None,
            connection_id: ConnectionId::new(),
        };
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (resume_tx, resume_rx) = std::sync::mpsc::channel();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        std::thread::scope(|scope| -> Result<()> {
            let read_database = &database;
            let reader = scope.spawn(move || {
                read_database.execute_autocommit(request, &mut |event| {
                    if matches!(event, QueryStreamEvent::Batch { .. }) {
                        assert!(read_database.0.state.try_write().is_some());
                        entered_tx
                            .send(())
                            .map_err(|error| Error::internal(error.to_string()))?;
                        resume_rx
                            .recv_timeout(Duration::from_secs(5))
                            .map_err(|error| Error::internal(error.to_string()))?;
                    }
                    Ok(())
                })
            });
            entered_rx
                .recv_timeout(Duration::from_secs(5))
                .map_err(|error| Error::internal(error.to_string()))?;
            let writer = scope.spawn(|| {
                let mut state = database.0.state.write();
                state.applied = Bookmark { term: 1, index: 2 };
                database.publish_reader_registry(&state);
                done_tx.send(())
            });
            let progressed = done_rx.recv_timeout(Duration::from_secs(2)).is_ok();
            resume_tx
                .send(())
                .map_err(|error| Error::internal(error.to_string()))?;
            reader
                .join()
                .map_err(|_| Error::internal("reader panicked"))??;
            writer
                .join()
                .map_err(|_| Error::internal("writer panicked"))?
                .map_err(|error| Error::internal(error.to_string()))?;
            assert!(
                progressed,
                "paused query readback blocked publication writer"
            );
            assert_eq!(database.bookmark().index, 2);
            Ok(())
        })
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
        database.publish_reader_registry(&database.0.state.read());
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
        database.publish_reader_registry(&database.0.state.read());
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
    fn transaction_registry_fences_sequencer_change_and_releases_shared_handles() -> Result<()> {
        let registry = TransactionAdmissionRegistry::new(1_024)?;
        let project = ProjectId::random();
        let bookmark = Bookmark { term: 2, index: 3 };
        let leader = ProcessId::random();
        let resource = transaction_resource(project, bookmark);
        registry.register(
            Uuid::new_v4(),
            ConnectionId::new(),
            256,
            TransactionPinKey { project, bookmark },
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
    fn streamed_intent_hash_matches_existing_bytes_for_dirty_and_temporal_journals() -> Result<()> {
        let project = ProjectId::random();
        let (_, mut graph) = ordered_graph_command(project, Bookmark { term: 2, index: 10 }, 42)?;
        let DatabaseMutation::Graph { graph: rows, .. } = &mut graph else {
            return Err(Error::internal("expected graph hash fixture"));
        };
        let GraphMutation::SetNodeProperty { value, .. } = &mut rows[0] else {
            return Err(Error::internal("expected property hash fixture"));
        };
        *value = ScalarValue::String(Arc::from("é漢字\n\0".repeat(40_000)));
        let mut temporal_graph = graph.clone();
        let DatabaseMutation::Graph { temporal, .. } = &mut temporal_graph else {
            return Err(Error::internal("expected temporal hash fixture"));
        };
        temporal.push(PersistedTemporalMutation {
            entity_kind: crate::types::EntityKind::Node,
            target: 0,
            sample: crate::graph::TemporalSample {
                entity_id: 1,
                property: crate::types::PropertyId(0),
                event_time_nanos: 123_456,
                sequence_index: 11,
                value: ScalarValue::Integer(42),
            },
            uses_commit_time: true,
        });
        let broker = DatabaseMutation::Broker {
            command: BrokerCommand::Retain {
                project,
                resolved_time_ms: 999,
            },
        };
        for mutation in [&graph, &temporal_graph, &broker] {
            let mut normalized = mutation.clone();
            resolve_sequencer_values(&mut normalized, 0)?;
            let bytes = encode_database_value(&normalized, "reference intent bytes")?;
            let mut reference = blake3::Hasher::new_derive_key("irongraph.request-intent.v1");
            reference.update(&bytes);
            assert_eq!(
                request_intent_digest(mutation)?,
                *reference.finalize().as_bytes()
            );
        }
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
    fn canonical_shared_handles_never_detach_on_mutation() -> Result<()> {
        let project = ordered_graph_project(ProjectId::random(), 1)?;
        let handle = project.graph.clone();
        let label = handle.catalog_mut().intern_label("Shared")?;
        handle.add_node_labels(NodeId(1), vec![label], 2)?;
        assert_eq!(project.graph.catalog().label("Shared"), Some(label));
        assert!(
            project
                .graph
                .node(NodeId(1))
                .ok_or_else(|| Error::internal("node absent"))?
                .labels()
                .contains(&label)
        );
        let property = handle
            .catalog()
            .property("value")
            .ok_or_else(|| Error::internal("property absent"))?;
        project
            .graph
            .set_node_property(NodeId(1), property, ScalarValue::Integer(42), 3)?;
        assert_eq!(
            handle.node(NodeId(1)).and_then(|n| n.property(property)),
            Some(ScalarValue::Integer(42))
        );
        Ok(())
    }

    #[test]
    fn failed_pure_batch_validation_never_changes_canonical_rows_or_schema() -> Result<()> {
        let graph = GraphStore::default();
        let label = crate::types::LabelId(0);
        let batch = vec![
            GraphMutation::DeclareLabel {
                name: "Pending".to_owned(),
                id: label,
            },
            GraphMutation::InsertNode(NodeInput {
                id: NodeId(1),
                layer: Layer::Observed,
                revision: 1,
                labels: vec![label],
                properties: vec![],
            }),
            GraphMutation::InsertNode(NodeInput {
                id: NodeId(1),
                layer: Layer::Observed,
                revision: 1,
                labels: vec![label],
                properties: vec![],
            }),
        ];
        assert!(graph.validate_mutations(&batch).is_err());
        assert_eq!(graph.node_count(), 0);
        assert!(graph.catalog().label("Pending").is_none());
        assert_eq!(graph.revision(), 0);
        Ok(())
    }

    #[test]
    fn committed_response_above_former_byte_quota_is_accepted() -> Result<()> {
        let mut state = DatabaseState {
            applied: Bookmark { term: 1, index: 1 },
            ..DatabaseState::default()
        };
        let request_id = Uuid::new_v4();
        let response = vec![b'x'; 17 * 1024 * 1024];
        retain_request_result(&mut state, request_id, [1; 32], response)?;
        prune_request_results(&mut state)?;
        assert_eq!(
            state.request_results[&request_id].response.len(),
            17 * 1024 * 1024
        );
        assert_eq!(state.request_result_bytes, 17 * 1024 * 1024);
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

        let project_state = state
            .projects
            .get(&project)
            .ok_or_else(|| Error::internal("project missing"))?;
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
    fn surgical_statistics_survive_hundreds_of_transaction_statements_and_preserve_results()
    -> Result<()> {
        let project_id = ProjectId::random();
        let project = (*ordered_graph_project(project_id, 1)?).clone();
        let value = project
            .graph
            .catalog()
            .property("value")
            .expect("value property");
        let label = project.graph.catalog_mut().intern_label("Data")?;
        let changed = project.graph.catalog_mut().intern_label("Changed")?;
        let body = project.graph.catalog_mut().intern_property("body")?;
        let payload = ScalarValue::String(Arc::from("complete unrelated document ".repeat(192)));
        project.graph.add_node_labels(NodeId(1), vec![label], 1)?;
        for id in 2..=4_096 {
            project.graph.insert_node(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: 1,
                labels: vec![label],
                properties: vec![
                    (value, ScalarValue::Integer(id as i64)),
                    (body, payload.clone()),
                ],
            })?;
        }
        project.graph.set_node_property(
            NodeId(3),
            body,
            ScalarValue::String(Arc::from("dirty source ".repeat(300))),
            2,
        )?;
        project
            .optimizer_statistics
            .set(Arc::new(StatisticsSnapshot::collect(&project.graph)))
            .map_err(|_| Error::internal("unexpected initialized statistics"))?;
        let pinned = project.clone();
        for revision in 3..=520 {
            let mut mutations = vec![GraphMutation::SetNodeProperty {
                node: NodeId(1),
                property: value,
                value: ScalarValue::Integer(revision as i64),
                revision,
            }];
            if revision == 256 {
                mutations.push(GraphMutation::AddNodeLabels {
                    node: NodeId(1),
                    labels: vec![changed],
                    revision,
                });
            }
            project.graph.validate_mutations(&mutations)?;
            for mutation in mutations {
                project.graph.apply(mutation)?;
            }
            refresh_optimizer_statistics(&project);
            let statistics = project
                .optimizer_statistics
                .get()
                .expect("statement statistics stay warm");
            assert_eq!(statistics.graph_revision, revision);
            assert_eq!(
                statistics.sampled_graph_revision,
                Some(2),
                "ordinary statement rebuilt full samples"
            );
            assert_eq!(statistics.node_count(LayerMask::ALL), 4_096);
            assert_eq!(
                statistics.label_count(changed, LayerMask::ALL),
                u64::from(revision >= 256)
            );
            assert_eq!(project.graph.node_count(), 4096);
        }
        project
            .graph
            .set_node_property(NodeId(2), value, ScalarValue::Integer(9_999_999), 520)?;
        refresh_optimizer_statistics(&project);
        assert_eq!(
            pinned.graph.node(NodeId(1)).and_then(|n| n.property(value)),
            Some(ScalarValue::Integer(520)),
            "shared handle observes canonical row updates"
        );
        let bookmark = Bookmark {
            term: 1,
            index: 520,
        };
        {
            for (query, expected) in [
                (
                    "MATCH (n:Data) WHERE n.value = 520 RETURN count(n) AS count",
                    2,
                ),
                (
                    "MATCH (n:Data) WHERE n.value > 9999998 RETURN count(n) AS count",
                    1,
                ),
            ] {
                let request = QueryRequest {
                    request_id: Uuid::new_v4(),
                    project_id: Some(project_id),
                    query: query.to_owned(),
                    parameters: BTreeMap::new(),
                    consistency: CommitAcknowledgement::Published,
                    bookmark: None,
                    cancellation: Default::default(),
                    deadline: None,
                    connection_id: ConnectionId::new(),
                };
                let output = execute_on_project(
                    &project,
                    &request,
                    bookmark,
                    521,
                    full_capabilities(),
                    None,
                )?;
                let values = output
                    .result
                    .batches
                    .iter()
                    .flat_map(|batch| &batch.columns[0].values)
                    .collect::<Vec<_>>();
                assert_eq!(
                    values,
                    [&ResultValue::Scalar(ScalarValue::Integer(expected))]
                );
                assert_eq!(
                    project
                        .optimizer_statistics
                        .get()
                        .expect("query kept statistics")
                        .sampled_graph_revision,
                    Some(2)
                );
            }
        }
        Ok(())
    }

    #[test]
    fn optimizer_statistics_advance_counts_without_rebuilding_samples() -> Result<()> {
        let project = ProjectId::random();
        let project_state = ProjectState {
            id: project,
            display_name: ("statistics".to_owned()).into(),
            graph: GraphStore::default(),
            temporal: TemporalStore::default(),
            predicate_versions: BTreeMap::new(),
            indexes: IndexCatalog::default(),
            next_node_id: (1).into(),
            next_edge_id: (1).into(),
            authority_revision: (0).into(),
            optimizer_statistics: OnceLock::new().into(),
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
        refresh_optimizer_statistics(&project_state);
        let first = project_state
            .optimizer_statistics
            .get()
            .expect("current statistics");
        assert_eq!(first.node_count(LayerMask::ALL), 1);
        assert_eq!(first.sampled_graph_revision, Some(0));
        let pinned = Arc::clone(&first);
        project_state
            .graph
            .apply(GraphMutation::InsertNode(crate::graph::NodeInput {
                id: crate::NodeId(2),
                layer: crate::Layer::Observed,
                revision: 2,
                labels: vec![label],
                properties: Vec::new(),
            }))?;
        refresh_optimizer_statistics(&project_state);
        let current = project_state
            .optimizer_statistics
            .get()
            .expect("current statistics");
        assert_eq!(current.node_count(LayerMask::ALL), 2);
        assert_eq!(current.graph_revision, 2);
        assert_eq!(current.sampled_graph_revision, Some(0));
        assert_eq!(pinned.node_count(LayerMask::ALL), 1);
        Ok(())
    }

    #[test]
    fn canonical_checkpoint_and_borrowed_payloads_survive_replacement_churn() -> Result<()> {
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
            let project_state = state
                .projects
                .get(&project)
                .ok_or_else(|| Error::internal("test project is missing"))?;
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

        let staged_project = state
            .projects
            .get(&project)
            .ok_or_else(|| Error::internal("canonical project is missing"))?;
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
            Some(replacement.clone())
        );
        assert_eq!(
            original,
            ScalarValue::List(DocumentList::new(vec![DocumentItem::Scalar(
                ScalarValue::String(Arc::from("a".repeat(1024)))
            )])?),
            "previously borrowed value remains memory safe"
        );
        assert_eq!(
            staged_project
                .graph
                .node(crate::NodeId(1))
                .and_then(|node| node.property(property)),
            Some(replacement)
        );
        let mut checkpoint = Vec::new();
        ciborium::ser::into_writer(&state, &mut checkpoint)
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
        let project_state = state
            .projects
            .get(&project)
            .ok_or_else(|| Error::internal("project disappeared"))?;
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
            Some(Arc::new(profile))
        );
        Ok(())
    }

    #[tokio::test]
    async fn document_embedding_delta_batches_complete_text_without_scanning_unrelated_rows()
    -> Result<()> {
        use crate::server::embedding_jobs::{
            DurableEmbeddingWorker, EmbeddingCallbacks, EmbeddingJob, EmbeddingOwner,
            EmbeddingWork, SemanticTextChunk,
        };
        let project_id = ProjectId::random();
        let project = ordered_graph_project(project_id, 1)?;
        let document = project.graph.catalog_mut().intern_label("Document")?;
        let unrelated = project.graph.catalog_mut().intern_label("Unrelated")?;
        let body = project.graph.catalog_mut().intern_property("body")?;
        let vector = project.graph.catalog_mut().intern_property("embedding")?;
        let source: Arc<str> = Arc::from("unrelated complete content ".repeat(4096));
        for id in 1000..1400 {
            project.graph.insert_node(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: id,
                labels: vec![unrelated],
                properties: vec![(body, ScalarValue::String(source.clone()))],
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
            vec![],
        )?;
        project.graph.insert_node(NodeInput {
            id: NodeId(2000),
            layer: Layer::Observed,
            revision: 2000,
            labels: vec![document],
            properties: vec![(body, ScalarValue::String(Arc::from("head interior tail")))],
        })?;
        let batches = Arc::new(Mutex::new(vec![]));
        let encoder = Arc::new(WindowEmbedding {
            profile,
            batches: batches.clone(),
        });
        let load_project = project.clone();
        let publish_project = project.clone();
        let pending = std::sync::atomic::AtomicBool::new(true);
        let queue = DurableEmbeddingWorker::start(
            Arc::new(move |_| {
                Ok(pending
                    .swap(false, Ordering::AcqRel)
                    .then_some(EmbeddingJob {
                        owner: EmbeddingOwner {
                            project: project_id,
                            kind: crate::types::EntityKind::Node,
                            entity_id: 2000,
                        },
                        revision: 2000,
                    }))
            }),
            EmbeddingCallbacks {
                load: Arc::new(move |job, _| {
                    assert_eq!(job.owner.entity_id, 2000);
                    let text = load_project
                        .graph
                        .node(NodeId(job.owner.entity_id))
                        .and_then(|n| n.property(body))
                        .and_then(|v| match v {
                            ScalarValue::String(s) => Some(s),
                            _ => None,
                        });
                    Ok(Some(EmbeddingWork {
                        encoder: encoder.clone(),
                        chunks: vec![SemanticTextChunk {
                            property: vector,
                            text,
                        }],
                        is_current: None,
                    }))
                }),
                publish: Arc::new(move |_, vectors, _| {
                    for mutation in vectors {
                        publish_project
                            .indexes
                            .apply_vector_mutation_from_graph(&publish_project.graph, &mutation)?;
                    }
                    Ok(true)
                }),
                failed: Arc::new(|_, error| panic!("unexpected delta queue failure: {error}")),
            },
        )?;
        tokio::time::timeout(Duration::from_secs(3), async {
            while !queue.is_idle() {
                tokio::time::sleep(Duration::from_millis(1)).await;
            }
        })
        .await
        .map_err(|_| Error::internal("document delta queue timed out"))?;
        assert_eq!(batches.lock().as_slice(), &[vec!["head", "tail"]]);
        let (exact, _) = project
            .indexes
            .vector_search_source("documents")
            .ok_or_else(|| Error::internal("document vector source absent"))?;
        let coordinates = exact
            .vector_for(2000)
            .ok_or_else(|| Error::internal("document vector absent"))?;
        assert_eq!(coordinates.len(), 2);
        assert!(coordinates.iter().all(|v| *v > 0.0));
        assert_eq!(exact.len(), 1);
        for id in 1000..1400 {
            let Some(ScalarValue::String(retained)) = project
                .graph
                .node(NodeId(id))
                .and_then(|n| n.property(body))
            else {
                return Err(Error::internal("unrelated source absent"));
            };
            assert!(Arc::ptr_eq(&source, &retained));
        }
        queue.shutdown().await?;
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
    fn canonical_captures_need_no_image_republication_after_writes() -> Result<()> {
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
        database.publish_reader_registry(&database.0.state.read());
        let (live, _) = database.capture_canonical_project(project)?;
        let property = live
            .graph
            .catalog()
            .property("value")
            .ok_or_else(|| Error::internal("value absent"))?;
        live.graph
            .set_node_property(NodeId(1), property, ScalarValue::Integer(42), 2)?;
        let (current, _) = database.capture_canonical_project(project)?;
        assert!(Arc::ptr_eq(&live, &current));
        assert_eq!(
            current
                .graph
                .node(NodeId(1))
                .and_then(|n| n.property(property)),
            Some(ScalarValue::Integer(42))
        );
        assert_eq!(live.graph.revision(), 2);
        Ok(())
    }

    #[test]
    fn canonical_property_delta_keeps_unrelated_large_source_allocations() -> Result<()> {
        for unrelated in [40, 400] {
            let graph = GraphStore::default();
            let body = graph.catalog_mut().intern_property("body")?;
            let value = graph.catalog_mut().intern_property("value")?;
            let source: Arc<str> = Arc::from("complete unrelated payload ".repeat(4096));
            for id in 1..=unrelated {
                graph.insert_node(NodeInput {
                    id: NodeId(id),
                    layer: Layer::Observed,
                    revision: 1,
                    labels: vec![],
                    properties: vec![(body, ScalarValue::String(source.clone()))],
                })?;
            }
            graph.set_node_property(NodeId(1), value, ScalarValue::Integer(7), 2)?;
            for id in 2..=unrelated {
                let Some(ScalarValue::String(retained)) =
                    graph.node(NodeId(id)).and_then(|n| n.property(body))
                else {
                    return Err(Error::internal("unrelated body disappeared"));
                };
                assert!(Arc::ptr_eq(&source, &retained));
            }
            assert_eq!(graph.node_count(), unrelated as usize);
        }
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
        database.publish_reader_registry(&database.0.state.read());
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
