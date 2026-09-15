//! In-process IronGraph with the same canonical WAL, snapshots, execution backends, and Cypher
//! semantics as the standalone server.

use std::{
    collections::HashMap,
    path::{Path, PathBuf},
    sync::Mutex,
    sync::atomic::{AtomicBool, Ordering},
    time::Duration,
};

use irongraph_client::{ClientError, Query, QueryResult};
use irongraph_server::{
    embeddings::{EmbeddingDevice, LocalEmbeddingModel, ensure_default_embedding_model},
    engine::{
        ExecutionClass, SingleNodeBootstrapConfig, WriteRuntime, WriteStorageLimits,
        open_standalone,
    },
    gpu::{
        BackendKind, DeviceMemoryGovernor, ResolvedComputeDevice,
        create_execution_backend_with_governor,
    },
    protocol::{QueryExecutor as _, QueryStreamEvent},
    server::Database,
};
use tokio_util::sync::CancellationToken;

#[cfg(test)]
mod acceptance;
#[cfg(test)]
mod standard_client;
mod streams;
pub use streams::{StreamAcknowledgement, StreamAppend, StreamFetch, StreamPage, StreamRecord};

/// Controls shared by native query and stream calls. IDs identify active calls only.
#[derive(Clone, Debug, Default, serde::Serialize, serde::Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OperationOptions {
    pub operation_id: Option<String>,
    pub timeout_ms: Option<u64>,
}

/// Readiness is returned only after recovery and configured model initialization finish.
#[derive(Clone, Debug, serde::Serialize)]
pub struct RuntimeStatus {
    pub data_dir: PathBuf,
    pub ready: bool,
    pub active_operations: usize,
    pub max_concurrent_operations: usize,
    pub worker_threads: usize,
}

struct Operation<'a> {
    database: &'a EmbeddedDatabase,
    id: String,
    cancellation: CancellationToken,
    deadline: std::time::Instant,
}

impl Operation<'_> {
    fn check(&self) -> Result<()> {
        if self.cancellation.is_cancelled() {
            return Err(irongraph_types::Error::new(
                irongraph_types::ErrorCode::Cancelled,
                "operation cancelled before dispatch",
            )
            .into());
        }
        if std::time::Instant::now() >= self.deadline {
            return Err(irongraph_types::Error::new(
                irongraph_types::ErrorCode::DeadlineExceeded,
                "operation deadline elapsed before dispatch",
            )
            .into());
        }
        Ok(())
    }
}

impl Drop for Operation<'_> {
    fn drop(&mut self) {
        self.database
            .operations
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&self.id);
    }
}

const DEFAULT_MAXIMUM_WRITE_BYTES: usize = 64 * 1024 * 1024;
const DEFAULT_RESERVED_DEVICE_BYTES: usize = 3 * 1024 * 1024 * 1024;
const DEFAULT_SNAPSHOT_INTERVAL: Duration = Duration::from_secs(30);

static EMBEDDED_INSTANCE_ACTIVE: AtomicBool = AtomicBool::new(false);

/// The complete supported execution-device set.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ExecutionDevice {
    #[default]
    Auto,
    Cpu,
    Metal(u32),
    Cuda(u32),
}

/// Local text-embedding startup policy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum EmbeddingPolicy {
    #[default]
    Automatic,
    Disabled,
}

/// Validated options for one embedded database instance.
#[derive(Clone, Debug)]
pub struct EmbeddedOptions {
    data_dir: PathBuf,
    pub execution_device: ExecutionDevice,
    pub embedding_policy: EmbeddingPolicy,
    pub device_memory_limit_bytes: usize,
    pub device_reserved_bytes: usize,
    pub max_write_bytes: usize,
    pub request_timeout: Duration,
    pub startup_timeout: Duration,
    pub snapshot_interval: Duration,
    pub max_concurrent_operations: usize,
    pub worker_threads: usize,
}

impl EmbeddedOptions {
    #[must_use]
    pub fn new(data_dir: impl Into<PathBuf>) -> Self {
        Self {
            data_dir: data_dir.into(),
            execution_device: ExecutionDevice::Auto,
            embedding_policy: EmbeddingPolicy::Automatic,
            device_memory_limit_bytes: usize::MAX,
            device_reserved_bytes: DEFAULT_RESERVED_DEVICE_BYTES,
            max_write_bytes: DEFAULT_MAXIMUM_WRITE_BYTES,
            request_timeout: Duration::from_secs(30),
            startup_timeout: Duration::from_secs(60),
            snapshot_interval: DEFAULT_SNAPSHOT_INTERVAL,
            max_concurrent_operations: 32,
            worker_threads: 2,
        }
    }

    #[must_use]
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    #[must_use]
    pub fn with_execution_device(mut self, execution_device: ExecutionDevice) -> Self {
        self.execution_device = execution_device;
        self
    }

    #[must_use]
    pub fn with_embedding_policy(mut self, embedding_policy: EmbeddingPolicy) -> Self {
        self.embedding_policy = embedding_policy;
        self
    }

    fn validate(&self) -> Result<()> {
        if self.max_concurrent_operations == 0 || self.worker_threads == 0 {
            return Err(EmbeddedError::Configuration(
                "operation and worker budgets must be positive".into(),
            ));
        }
        if self.data_dir.as_os_str().is_empty() {
            return Err(EmbeddedError::Configuration(
                "embedded data directory is empty".to_owned(),
            ));
        }
        if self.max_write_bytes < 1024 * 1024 {
            return Err(EmbeddedError::Configuration(
                "maximum write bytes must be at least 1 MiB".to_owned(),
            ));
        }
        if self.request_timeout.is_zero()
            || self.startup_timeout.is_zero()
            || self.snapshot_interval.is_zero()
        {
            return Err(EmbeddedError::Configuration(
                "embedded timeouts and snapshot interval must be positive".to_owned(),
            ));
        }
        Ok(())
    }
}

/// Embedded lifecycle or query error.
#[derive(Debug, thiserror::Error)]
pub enum EmbeddedError {
    #[error("invalid embedded configuration: {0}")]
    Configuration(String),
    #[error("another embedded IronGraph instance is already active in this process")]
    ProcessInstanceActive,
    #[error("database startup or shutdown failed: {0}")]
    Engine(#[from] irongraph_types::Error),
    #[error(transparent)]
    Query(#[from] ClientError),
    #[error("embedded runtime failed: {0}")]
    Runtime(#[from] std::io::Error),
}

pub type Result<T> = std::result::Result<T, EmbeddedError>;

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ResourceBudgets {
    device_memory_limit_bytes: Option<usize>,
    device_reserved_bytes: Option<usize>,
    max_write_bytes: Option<usize>,
    max_concurrent_operations: Option<usize>,
    worker_threads: Option<usize>,
    request_timeout_ms: Option<u64>,
    startup_timeout_ms: Option<u64>,
    snapshot_interval_ms: Option<u64>,
}

/// Validate and apply explicit host resource budgets from a language binding.
pub fn configure_budgets(options: &mut EmbeddedOptions, budgets: serde_json::Value) -> Result<()> {
    let budgets: ResourceBudgets = serde_json::from_value(budgets)
        .map_err(|error| EmbeddedError::Configuration(error.to_string()))?;
    if let Some(value) = budgets.device_memory_limit_bytes {
        options.device_memory_limit_bytes = value;
    }
    if let Some(value) = budgets.device_reserved_bytes {
        options.device_reserved_bytes = value;
    }
    if let Some(value) = budgets.max_write_bytes {
        options.max_write_bytes = value;
    }
    if let Some(value) = budgets.max_concurrent_operations {
        options.max_concurrent_operations = value;
    }
    if let Some(value) = budgets.worker_threads {
        options.worker_threads = value;
    }
    if let Some(value) = budgets.request_timeout_ms {
        options.request_timeout = Duration::from_millis(value);
    }
    if let Some(value) = budgets.startup_timeout_ms {
        options.startup_timeout = Duration::from_millis(value);
    }
    if let Some(value) = budgets.snapshot_interval_ms {
        options.snapshot_interval = Duration::from_millis(value);
    }
    options.validate()
}

struct InstanceGuard;

impl InstanceGuard {
    fn acquire() -> Result<Self> {
        EMBEDDED_INSTANCE_ACTIVE
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map_err(|_| EmbeddedError::ProcessInstanceActive)?;
        Ok(Self)
    }
}

impl Drop for InstanceGuard {
    fn drop(&mut self) {
        EMBEDDED_INSTANCE_ACTIVE.store(false, Ordering::Release);
    }
}

struct EmbeddedCore {
    boot: irongraph_server::engine::BootstrappedNode<Database>,
    database: Database,
    snapshot_directory: PathBuf,
    snapshot_shutdown: CancellationToken,
    snapshot_task: tokio::task::JoinHandle<()>,
    _embedding: Option<std::sync::Arc<LocalEmbeddingModel>>,
    _instance: InstanceGuard,
}

impl EmbeddedCore {
    async fn open(options: EmbeddedOptions, instance: InstanceGuard) -> Result<Self> {
        options.validate()?;
        let data_dir = absolute_path(options.data_dir)?;
        let resolved_device = resolve_device(options.execution_device)?;
        let governor = DeviceMemoryGovernor::new(
            options.device_memory_limit_bytes,
            options.device_reserved_bytes,
        );
        let execution_backend =
            create_execution_backend_with_governor(resolved_device, governor.clone())?;
        let max_write_bytes = options.max_write_bytes;
        let request_timeout = options.request_timeout;
        let boot = open_standalone(
            &data_dir,
            SingleNodeBootstrapConfig {
                execution_class: execution_class(resolved_device.backend),
                startup_timeout: options.startup_timeout,
                storage_limits: WriteStorageLimits {
                    max_log_record_bytes: max_write_bytes.saturating_mul(2),
                    max_log_entries_per_read: 4_096,
                    max_snapshot_bytes: 64 * 1024 * 1024 * 1024,
                },
            },
            move |directory, identity| {
                Ok(std::sync::Arc::new(Database::open_backend(
                    directory,
                    max_write_bytes,
                    request_timeout,
                    identity,
                )?))
            },
        )
        .await?;
        boot.backend()
            .bind_runtime(std::sync::Arc::downgrade(boot.runtime()))?;
        let database = boot.backend().as_ref().clone();
        let snapshot_directory = data_dir.join("standalone-snapshots");
        database.standalone_recover(&snapshot_directory).await?;
        database.bind_execution_backend(execution_backend)?;
        // Mutations committed after the restored snapshot exist only in the standalone WAL, and
        // replay must finish before the first write claims an index the WAL already holds.
        boot.runtime().replay_standalone_wal().await?;

        let embedding = if options.embedding_policy == EmbeddingPolicy::Automatic {
            let embedding_device = embedding_device(options.execution_device);
            let embedding_governor = governor.clone();
            let embedding_database = database.clone();
            Some(
                tokio::task::spawn_blocking(move || {
                    let artifacts = ensure_default_embedding_model()?;
                    let embedding = std::sync::Arc::new(LocalEmbeddingModel::load(
                        artifacts,
                        embedding_device,
                    )?);
                    embedding.bind_memory_governor(embedding_governor)?;
                    embedding.warm_up()?;
                    embedding_database.bind_text_embedding(embedding.clone())?;
                    Ok::<_, irongraph_types::Error>(embedding)
                })
                .await
                .map_err(|error| {
                    irongraph_types::Error::internal(format!(
                        "embedded embedding startup task failed: {error}"
                    ))
                })??,
            )
        } else {
            None
        };

        let snapshot_shutdown = CancellationToken::new();
        let snapshot_task = spawn_snapshot_maintenance(
            database.clone(),
            std::sync::Arc::clone(boot.runtime()),
            snapshot_directory.clone(),
            options.snapshot_interval,
            snapshot_shutdown.clone(),
        );
        Ok(Self {
            boot,
            database,
            snapshot_directory,
            snapshot_shutdown,
            snapshot_task,
            _embedding: embedding,
            _instance: instance,
        })
    }

    fn query(&self, query: Query, operation: &Operation<'_>) -> Result<QueryResult> {
        query.validate()?;
        operation.check()?;
        let mut request = query.into_protocol();
        request.cancellation = operation.cancellation.clone();
        request.deadline = Some(operation.deadline);
        let mut events = Vec::<QueryStreamEvent>::new();
        self.database.execute(request, &mut |event| {
            events.push(event);
            Ok(())
        })?;
        Ok(QueryResult::from_events(events)?)
    }

    async fn close(self) -> Result<()> {
        self.snapshot_shutdown.cancel();
        let joined = self.snapshot_task.await;
        let snapshot = self
            .database
            .standalone_snapshot(&self.snapshot_directory)
            .await;
        // Always stop the writer, including when the final snapshot fails.
        let shutdown = self.boot.runtime().shutdown().await;
        joined.map_err(|error| {
            irongraph_types::Error::internal(format!("snapshot worker failed: {error}"))
        })?;
        snapshot?;
        shutdown?;
        Ok(())
    }
}

/// Synchronous in-process database suitable for Rust, Python, and Node host runtimes.
pub struct EmbeddedDatabase {
    runtime: tokio::runtime::Runtime,
    core: Option<EmbeddedCore>,
    options: EmbeddedOptions,
    operations: Mutex<HashMap<String, CancellationToken>>,
    next_operation: std::sync::atomic::AtomicU64,
}

impl EmbeddedDatabase {
    pub fn open(mut options: EmbeddedOptions) -> Result<Self> {
        options.validate()?;
        options.data_dir = absolute_path(options.data_dir)?;
        let instance = InstanceGuard::acquire()?;
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(options.worker_threads)
            .max_blocking_threads(options.max_concurrent_operations)
            .enable_all()
            .thread_name("irongraph-embedded")
            .build()?;
        let core = runtime.block_on(EmbeddedCore::open(options.clone(), instance))?;
        Ok(Self {
            runtime,
            core: Some(core),
            options,
            operations: Mutex::new(HashMap::new()),
            next_operation: std::sync::atomic::AtomicU64::new(0),
        })
    }

    pub fn query(&self, query: Query) -> Result<QueryResult> {
        self.query_with_options(query, OperationOptions::default())
    }

    pub fn query_with_options(
        &self,
        query: Query,
        options: OperationOptions,
    ) -> Result<QueryResult> {
        let operation = self.begin_operation(options)?;
        self.core
            .as_ref()
            .ok_or_else(|| EmbeddedError::Configuration("database is closed".to_owned()))?
            .query(query, &operation)
    }

    /// Returns false if the operation has not started or has already completed.
    pub fn cancel(&self, operation_id: &str) -> bool {
        let operations = self
            .operations
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if let Some(token) = operations.get(operation_id) {
            token.cancel();
            true
        } else {
            false
        }
    }

    pub fn status(&self) -> Result<RuntimeStatus> {
        Ok(RuntimeStatus {
            data_dir: absolute_path(self.options.data_dir.clone())?,
            ready: self.core.is_some(),
            active_operations: self
                .operations
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .len(),
            max_concurrent_operations: self.options.max_concurrent_operations,
            worker_threads: self.options.worker_threads,
        })
    }

    fn begin_operation(&self, options: OperationOptions) -> Result<Operation<'_>> {
        let timeout = options
            .timeout_ms
            .map_or(self.options.request_timeout, Duration::from_millis);
        let deadline = std::time::Instant::now()
            .checked_add(timeout)
            .ok_or_else(|| EmbeddedError::Configuration("operation timeout is too large".into()))?;
        let id = options.operation_id.unwrap_or_else(|| {
            format!(
                "internal:{}",
                self.next_operation.fetch_add(1, Ordering::Relaxed)
            )
        });
        if id.is_empty() || id.len() > 256 {
            return Err(EmbeddedError::Configuration(
                "operation ID must contain 1 to 256 bytes".into(),
            ));
        }
        let mut operations = self
            .operations
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if operations.contains_key(&id) {
            return Err(EmbeddedError::Configuration(
                "operation ID is already active".into(),
            ));
        }
        if operations.len() >= self.options.max_concurrent_operations {
            return Err(irongraph_types::Error::new(
                irongraph_types::ErrorCode::Backpressure,
                "embedded concurrency budget exhausted before dispatch",
            )
            .into());
        }
        let cancellation = CancellationToken::new();
        operations.insert(id.clone(), cancellation.clone());
        Ok(Operation {
            database: self,
            id,
            cancellation,
            deadline,
        })
    }

    pub fn snapshot(&self) -> Result<irongraph_types::Bookmark> {
        let core = self
            .core
            .as_ref()
            .ok_or_else(|| EmbeddedError::Configuration("database is closed".to_owned()))?;
        Ok(self
            .runtime
            .block_on(core.database.standalone_snapshot(&core.snapshot_directory))?)
    }

    pub fn flush(&self) -> Result<()> {
        let core = self
            .core
            .as_ref()
            .ok_or_else(|| EmbeddedError::Configuration("database is closed".into()))?;
        Ok(self.runtime.block_on(core.boot.runtime().flush_wal())?)
    }

    pub fn close(mut self) -> Result<()> {
        self.close_inner()
    }

    fn close_inner(&mut self) -> Result<()> {
        if let Some(core) = self.core.take() {
            self.runtime.block_on(core.close())?;
        }
        Ok(())
    }
}

impl Drop for EmbeddedDatabase {
    fn drop(&mut self) {
        let _ = self.close_inner();
    }
}

fn absolute_path(path: PathBuf) -> Result<PathBuf> {
    if path.is_absolute() {
        return Ok(path);
    }
    Ok(std::env::current_dir()?.join(path))
}

fn resolve_device(device: ExecutionDevice) -> Result<ResolvedComputeDevice> {
    match device {
        ExecutionDevice::Cpu => Ok(ResolvedComputeDevice {
            backend: BackendKind::Cpu,
            ordinal: 0,
        }),
        ExecutionDevice::Metal(ordinal) => Ok(ResolvedComputeDevice {
            backend: BackendKind::Metal,
            ordinal,
        }),
        ExecutionDevice::Cuda(ordinal) => Ok(ResolvedComputeDevice {
            backend: BackendKind::Cuda,
            ordinal,
        }),
        ExecutionDevice::Auto => automatic_device(),
    }
}

fn automatic_device() -> Result<ResolvedComputeDevice> {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        return Ok(ResolvedComputeDevice {
            backend: BackendKind::Metal,
            ordinal: 0,
        });
    }
    #[cfg(all(feature = "cuda", not(any(target_os = "macos", target_os = "ios"))))]
    {
        return Ok(ResolvedComputeDevice {
            backend: BackendKind::Cuda,
            ordinal: 0,
        });
    }
    #[allow(unreachable_code)]
    Err(EmbeddedError::Configuration(
        "automatic execution-device selection found no compiled accelerator; select CPU explicitly"
            .to_owned(),
    ))
}

const fn execution_class(backend: BackendKind) -> ExecutionClass {
    match backend {
        BackendKind::Cpu => ExecutionClass::Cpu,
        BackendKind::Metal => ExecutionClass::Metal,
        BackendKind::Cuda => ExecutionClass::Cuda,
    }
}

const fn embedding_device(device: ExecutionDevice) -> EmbeddingDevice {
    match device {
        ExecutionDevice::Auto => EmbeddingDevice::Auto,
        ExecutionDevice::Cpu => EmbeddingDevice::Cpu,
        ExecutionDevice::Metal(ordinal) => EmbeddingDevice::Metal(ordinal),
        ExecutionDevice::Cuda(ordinal) => EmbeddingDevice::Cuda(ordinal),
    }
}

fn spawn_snapshot_maintenance(
    database: Database,
    runtime: std::sync::Arc<WriteRuntime>,
    directory: PathBuf,
    interval: Duration,
    shutdown: CancellationToken,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval_at(tokio::time::Instant::now() + interval, interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut last = None;
        loop {
            tokio::select! {
                () = shutdown.cancelled() => return,
                _ = ticker.tick() => {
                    if last == Some(database.bookmark()) {
                        continue;
                    }
                    // A failure below leaves `last` unset so the next tick retries; background
                    // upkeep never ends the embedded process.
                    match database.standalone_snapshot(&directory).await {
                        Ok(bookmark) => match database.standalone_wal_compaction_bookmark(&directory, bookmark).await {
                            Ok(prefix) if prefix != irongraph_types::Bookmark::default() => {
                                let runtime = std::sync::Arc::clone(&runtime);
                                match tokio::task::spawn_blocking(move || runtime.compact_wal_through(prefix)).await {
                                    Ok(Ok(())) => last = Some(bookmark),
                                    Ok(Err(error)) => tracing::warn!(code = ?error.code, message = %error.message, "embedded WAL compaction failed"),
                                    Err(error) => tracing::warn!(%error, "embedded WAL compaction task failed"),
                                }
                            }
                            Ok(_) => last = Some(bookmark),
                            Err(error) => tracing::warn!(code = ?error.code, message = %error.message, "embedded snapshot prefix selection failed"),
                        },
                        Err(error) => tracing::warn!(code = ?error.code, message = %error.message, "embedded periodic snapshot failed"),
                    }
                }
            }
        }
    })
}
