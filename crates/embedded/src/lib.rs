//! In-process IronGraph with the same asynchronous persistence, CPU execution, and Cypher
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
    embeddings::{
        EmbeddingStorage, LocalEmbeddingModel, ModelInstallProgress,
        ensure_default_embedding_model_with_storage,
    },
    engine::{ExecutionClass, SingleNodeBootstrapConfig, open_standalone},
    protocol::{QueryExecutor as _, QueryRequest},
    server::Database,
};
use tokio_util::sync::CancellationToken;

#[cfg(test)]
mod acceptance;
mod owner;
#[cfg(test)]
mod standard_client;
mod streams;
pub use irongraph_server::embeddings::EmbeddingDevice;
pub use owner::{AdapterDatabaseOwner, AdapterOwnerAdmission};
pub use streams::{StreamAcknowledgement, StreamAppend, StreamFetch, StreamPage, StreamRecord};

#[cfg(test)]
type QueryExecutionGate = (
    tokio::sync::oneshot::Sender<()>,
    std::sync::mpsc::Receiver<()>,
);
#[cfg(test)]
static QUERY_EXECUTION_GATE: Mutex<Option<QueryExecutionGate>> = Mutex::new(None);

fn collect_query_result(database: &Database, request: QueryRequest) -> Result<QueryResult> {
    #[cfg(test)]
    {
        let gate = QUERY_EXECUTION_GATE.lock().unwrap().take();
        if let Some((entered, release)) = gate {
            let _ = entered.send(());
            release.recv().map_err(|error| {
                irongraph_types::Error::internal(format!("query test gate closed: {error}"))
            })?;
        }
    }
    let mut result = QueryResult::default();
    let mut assembly_error = None;
    let execution = database.execute(request, &mut |event| {
        result.append_event(event).map_err(|error| {
            let cancelled = irongraph_types::Error::internal(error.to_string());
            assembly_error = Some(error);
            cancelled
        })
    });
    if let Some(error) = assembly_error {
        return Err(error.into());
    }
    execution?;
    Ok(result)
}

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
    pub worker_threads: usize,
}

struct Operation<'a> {
    database: &'a EmbeddedDatabase,
    id: String,
    cancellation: CancellationToken,
    deadline: Option<std::time::Instant>,
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
        if self
            .deadline
            .is_some_and(|deadline| std::time::Instant::now() >= deadline)
        {
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
        self.cancellation.cancel();
        self.database
            .operations
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .remove(&self.id);
    }
}

// Synchronous query workers can await ordered writer callbacks. Keep bounded callback,
// inference, and maintenance capacity available even when every query permit is occupied.
const BLOCKING_CALLBACK_HEADROOM: usize = 16;

#[cfg(test)]
mod embedding_configuration_tests {
    use super::*;

    #[test]
    fn application_paths_are_carried_independently_of_database_storage() {
        let options = EmbeddedOptions::new("app/data")
            .with_model_dir("app/models")
            .with_embedding_cache_dir("app/cache");
        assert_eq!(options.data_dir(), Path::new("app/data"));
        assert_eq!(options.model_dir.as_deref(), Some(Path::new("app/models")));
        assert_eq!(
            options.embedding_cache_dir.as_deref(),
            Some(Path::new("app/cache"))
        );
    }

    #[test]
    fn cpu_graph_keeps_independent_embedding_selection() {
        let options = EmbeddedOptions::new("test-data").with_execution_device(ExecutionDevice::Cpu);
        assert_eq!(options.embedding_device, EmbeddingDevice::Auto);
        assert!(options.validate().is_ok());
        let options = options.with_embedding_device(EmbeddingDevice::Metal(2));
        assert_eq!(options.embedding_device, EmbeddingDevice::Metal(2));
        assert!(options.validate().is_ok());
        assert!(
            options
                .with_execution_device(ExecutionDevice::Metal(0))
                .validate()
                .is_err()
        );
    }
}

static EMBEDDED_INSTANCE_ACTIVE: AtomicBool = AtomicBool::new(false);

/// Graph device request. The CPU runtime accepts Auto and Cpu; GPU requests return a configuration error.
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
    pub embedding_device: EmbeddingDevice,
    pub embedding_policy: EmbeddingPolicy,
    pub model_dir: Option<PathBuf>,
    pub embedding_cache_dir: Option<PathBuf>,
    pub worker_threads: usize,
}

impl EmbeddedOptions {
    #[must_use]
    pub fn new(data_dir: impl Into<PathBuf>) -> Self {
        Self {
            data_dir: data_dir.into(),
            execution_device: ExecutionDevice::Auto,
            embedding_device: EmbeddingDevice::Auto,
            embedding_policy: EmbeddingPolicy::Automatic,
            model_dir: None,
            embedding_cache_dir: None,
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

    #[must_use]
    pub fn with_embedding_device(mut self, embedding_device: EmbeddingDevice) -> Self {
        self.embedding_device = embedding_device;
        self
    }

    #[must_use]
    pub fn with_model_dir(mut self, directory: impl Into<PathBuf>) -> Self {
        self.model_dir = Some(directory.into());
        self
    }

    #[must_use]
    pub fn with_embedding_cache_dir(mut self, directory: impl Into<PathBuf>) -> Self {
        self.embedding_cache_dir = Some(directory.into());
        self
    }

    fn validate(&self) -> Result<()> {
        if !matches!(
            self.execution_device,
            ExecutionDevice::Auto | ExecutionDevice::Cpu
        ) {
            return Err(EmbeddedError::Configuration(
                "graph execution is CPU-only; select GPU inference with embedding_device".into(),
            ));
        }
        if self.worker_threads == 0 {
            return Err(EmbeddedError::Configuration(
                "worker_threads must be positive".into(),
            ));
        }
        if self
            .worker_threads
            .checked_mul(4)
            .and_then(|workers| workers.checked_add(BLOCKING_CALLBACK_HEADROOM))
            .is_none_or(|workers| workers > tokio::sync::Semaphore::MAX_PERMITS)
        {
            return Err(EmbeddedError::Configuration(
                "worker_threads exceeds the scheduler's representable range".into(),
            ));
        }
        if self.data_dir.as_os_str().is_empty() {
            return Err(EmbeddedError::Configuration(
                "embedded data directory is empty".to_owned(),
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
    worker_threads: Option<usize>,
}

/// Validate and apply explicit host resource budgets from a language binding.
pub fn configure_budgets(options: &mut EmbeddedOptions, budgets: serde_json::Value) -> Result<()> {
    let budgets: ResourceBudgets = serde_json::from_value(budgets)
        .map_err(|error| EmbeddedError::Configuration(error.to_string()))?;
    if let Some(value) = budgets.worker_threads {
        options.worker_threads = value;
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
    _embedding: Option<std::sync::Arc<LocalEmbeddingModel>>,
    _instance: InstanceGuard,
}

impl EmbeddedCore {
    async fn open(options: EmbeddedOptions, instance: InstanceGuard) -> Result<Self> {
        options.validate()?;
        let data_dir = absolute_path(options.data_dir)?;
        let max_write_bytes = usize::MAX;
        let request_timeout = Duration::ZERO;
        let progress = ModelInstallProgress::default();
        let preparation_progress = progress.clone();
        let recovery = async {
            let recovered = open_standalone(
                &data_dir,
                SingleNodeBootstrapConfig {
                    execution_class: ExecutionClass::Cpu,
                    startup_timeout: Duration::ZERO,
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
            .await;
            if recovered.is_err() {
                progress.cancel();
            }
            recovered
        };
        let preparation = async {
            let embedding = if options.embedding_policy == EmbeddingPolicy::Automatic {
                let embedding_device = options.embedding_device;
                let storage = EmbeddingStorage {
                    model_dir: options.model_dir.clone(),
                    cache_dir: options.embedding_cache_dir.clone(),
                };
                Some(
                    tokio::task::spawn_blocking(move || {
                        let started = std::time::Instant::now();
                        let artifacts = ensure_default_embedding_model_with_storage(
                            &storage,
                            &preparation_progress,
                        )?;
                        tracing::info!(
                            phase = "embedding_artifacts",
                            elapsed_ms = started.elapsed().as_secs_f64() * 1000.0,
                            "startup embedding phase completed"
                        );
                        let started = std::time::Instant::now();
                        let embedding =
                            std::sync::Arc::new(LocalEmbeddingModel::load_with_storage(
                                artifacts,
                                embedding_device,
                                &storage,
                            )?);
                        tracing::info!(
                            phase = "embedding_load",
                            elapsed_ms = started.elapsed().as_secs_f64() * 1000.0,
                            "startup embedding phase completed"
                        );
                        let started = std::time::Instant::now();
                        embedding.warm_up()?;
                        tracing::info!(
                            phase = "embedding_warmup",
                            elapsed_ms = started.elapsed().as_secs_f64() * 1000.0,
                            "startup embedding phase completed"
                        );
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
            Ok::<_, irongraph_types::Error>(embedding)
        };
        // Neither preparation reads graph data; both finish before the encoder is bound or the
        // embedded handle is returned. Join also settles recovery when encoder preparation fails.
        let (recovered, prepared) = tokio::join!(recovery, preparation);
        let boot = recovered?;
        let embedding = prepared?;
        boot.backend()
            .bind_runtime(std::sync::Arc::downgrade(boot.runtime()))?;
        let database = boot.backend().as_ref().clone();
        if let Some(embedding) = &embedding {
            database.bind_text_embedding(embedding.clone())?;
        }

        Ok(Self {
            boot,
            database,
            _embedding: embedding,
            _instance: instance,
        })
    }

    fn query(&self, query: Query, operation: &Operation<'_>) -> Result<QueryResult> {
        query.validate()?;
        operation.check()?;
        let mut request = query.into_protocol();
        request.cancellation = operation.cancellation.clone();
        request.deadline = operation.deadline;
        collect_query_result(&self.database, request)
    }

    async fn close(self) -> Result<()> {
        let embedding_shutdown = self.database.shutdown_embedding_jobs().await;
        let shutdown = self.boot.runtime().shutdown().await;
        shutdown?;
        embedding_shutdown?;
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
    async_workers: std::sync::Arc<tokio::sync::Semaphore>,
}

impl EmbeddedDatabase {
    pub fn open(mut options: EmbeddedOptions) -> Result<Self> {
        options.validate()?;
        options.data_dir = absolute_path(options.data_dir)?;
        let instance = InstanceGuard::acquire()?;
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(options.worker_threads)
            .max_blocking_threads(
                options.worker_threads.saturating_mul(4) + BLOCKING_CALLBACK_HEADROOM,
            )
            .enable_all()
            .thread_name("irongraph-embedded")
            .build()?;
        let core = runtime.block_on(EmbeddedCore::open(options.clone(), instance))?;
        let async_workers = std::sync::Arc::new(tokio::sync::Semaphore::new(
            options.worker_threads.saturating_mul(4),
        ));
        Ok(Self {
            runtime,
            core: Some(core),
            options,
            operations: Mutex::new(HashMap::new()),
            next_operation: std::sync::atomic::AtomicU64::new(0),
            async_workers,
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

    /// Executes without blocking the caller's async runtime. All workers share the canonical graph.
    pub async fn query_async(&self, query: Query) -> Result<QueryResult> {
        self.query_with_options_async(query, OperationOptions::default())
            .await
    }

    pub async fn query_with_options_async(
        &self,
        query: Query,
        options: OperationOptions,
    ) -> Result<QueryResult> {
        let operation = self.begin_operation(options)?;
        operation.check()?;
        let permit = self
            .async_workers
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| EmbeddedError::Configuration("embedded workers are unavailable".into()))?;
        let mut request = query.into_protocol();
        request.cancellation = operation.cancellation.clone();
        request.deadline = operation.deadline;
        let database = self
            .core
            .as_ref()
            .ok_or_else(|| EmbeddedError::Configuration("database is closed".to_owned()))?
            .database
            .clone();
        self.runtime
            .handle()
            .spawn_blocking(move || {
                let _permit = permit;
                collect_query_result(&database, request)
            })
            .await
            .map_err(|error| {
                irongraph_types::Error::internal(format!("embedded query worker failed: {error}"))
            })?
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
            worker_threads: self.options.worker_threads,
        })
    }

    fn begin_operation(&self, options: OperationOptions) -> Result<Operation<'_>> {
        let timeout = options.timeout_ms.map(Duration::from_millis);
        let deadline = timeout
            .map(|timeout| {
                std::time::Instant::now()
                    .checked_add(timeout)
                    .ok_or_else(|| {
                        EmbeddedError::Configuration("operation timeout is too large".into())
                    })
            })
            .transpose()?;
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
        let cancellation = CancellationToken::new();
        operations.insert(id.clone(), cancellation.clone());
        Ok(Operation {
            database: self,
            id,
            cancellation,
            deadline,
        })
    }

    pub fn flush(&self) -> Result<()> {
        let core = self
            .core
            .as_ref()
            .ok_or_else(|| EmbeddedError::Configuration("database is closed".into()))?;
        Ok(self
            .runtime
            .block_on(core.boot.runtime().flush_persistence())?)
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
