//! IronGraph SDK for embedded databases and remote API/Bolt connections.
//!
//! Documents, vector search, graph algorithms, projects, and administration use the same
//! [`Query`] interface. Synchronous methods block the caller; asynchronous embedded methods
//! run native work on Tokio's blocking pool and cancel active operations when dropped.

use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::{collections::BTreeMap, fmt, path::PathBuf};

#[repr(C)]
struct Buffer {
    data: *mut u8,
    len: usize,
}
unsafe extern "C" {
    fn irongraph_abi_version() -> u32;
    fn irongraph_package_version_v1() -> *const std::ffi::c_char;
    fn irongraph_call_v1(request: *const u8, len: usize, out: *mut Buffer) -> i32;
    fn irongraph_buffer_free_v1(buffer: Buffer);
}

/// Structured native or SDK error. Database error codes and retry information are preserved.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Error {
    pub code: String,
    pub message: String,
    pub retryable: bool,
    pub retry_after_ms: Option<u64>,
}
impl Error {
    fn sdk(message: impl Into<String>) -> Self {
        Self {
            code: "SDK_PROTOCOL".into(),
            message: message.into(),
            retryable: false,
            retry_after_ms: None,
        }
    }
}
impl fmt::Display for Error {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code, self.message)
    }
}
impl std::error::Error for Error {}
pub type Result<T> = std::result::Result<T, Error>;

struct OwnedBuffer(Buffer);
impl Drop for OwnedBuffer {
    fn drop(&mut self) {
        unsafe {
            irongraph_buffer_free_v1(Buffer {
                data: self.0.data,
                len: self.0.len,
            });
        }
    }
}

fn call(request: Value) -> Result<Value> {
    call_typed(request)
}

fn call_typed<T: serde::de::DeserializeOwned>(request: Value) -> Result<T> {
    if unsafe { irongraph_abi_version() } != 1 {
        return Err(Error::sdk("native library ABI version mismatch"));
    }
    let version = unsafe { std::ffi::CStr::from_ptr(irongraph_package_version_v1()) };
    if version.to_bytes() != env!("CARGO_PKG_VERSION").as_bytes() {
        return Err(Error::sdk("native library package version mismatch"));
    }
    let bytes = serde_json::to_vec(&request).map_err(|error| Error::sdk(error.to_string()))?;
    let mut buffer = Buffer {
        data: std::ptr::null_mut(),
        len: 0,
    };
    let status = unsafe { irongraph_call_v1(bytes.as_ptr(), bytes.len(), &mut buffer) };
    if status == 2 {
        return Err(Error::sdk("native library rejected request buffer"));
    }
    let buffer = OwnedBuffer(buffer);
    if buffer.0.data.is_null() || buffer.0.len > isize::MAX as usize {
        return Err(Error::sdk("native library returned an invalid buffer"));
    }
    let bytes = unsafe { std::slice::from_raw_parts(buffer.0.data, buffer.0.len) };
    let response: NativeResponse<'_> =
        serde_json::from_slice(bytes).map_err(|error| Error::sdk(error.to_string()))?;
    if response.version != env!("CARGO_PKG_VERSION") {
        return Err(Error::sdk("native library package version mismatch"));
    }
    match (status, response.ok) {
        (0, true) => serde_json::from_str(
            response
                .result
                .ok_or_else(|| Error::sdk("native result is missing"))?
                .get(),
        )
        .map_err(|error| Error::sdk(error.to_string())),
        (1, false) => Err(response
            .error
            .ok_or_else(|| Error::sdk("native error is missing"))?),
        _ => Err(Error::sdk("native response status is inconsistent")),
    }
}

#[derive(Deserialize)]
struct NativeResponse<'a> {
    ok: bool,
    version: &'a str,
    #[serde(default, borrow, deserialize_with = "present_native_result")]
    result: Option<&'a serde_json::value::RawValue>,
    error: Option<Error>,
}

fn present_native_result<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> std::result::Result<Option<&'de serde_json::value::RawValue>, D::Error> {
    // Preserve an explicitly returned null, and distinguish it from a missing result field.
    <&serde_json::value::RawValue>::deserialize(deserializer).map(Some)
}

/// Position of an applied database write.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
pub struct Bookmark {
    pub term: u64,
    pub index: u64,
}

/// Acknowledgement for a write applied by this single-node database.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum CommitAcknowledgement {
    #[default]
    Published,
}

/// One Cypher request covering the complete database query and administration surface.
#[derive(Clone, Debug, Serialize)]
pub struct Query {
    pub cypher: String,
    pub parameters: BTreeMap<String, Value>,
    /// Immutable project UUID, returned by `SHOW PROJECTS`; no default project is assumed.
    pub project_id: Option<String>,
    pub bookmark: Option<Bookmark>,
    pub consistency: CommitAcknowledgement,
}
impl Query {
    pub fn new(cypher: impl Into<String>) -> Self {
        Self {
            cypher: cypher.into(),
            parameters: BTreeMap::new(),
            project_id: None,
            bookmark: None,
            consistency: CommitAcknowledgement::Published,
        }
    }
    pub fn with_project(mut self, project_id: impl Into<String>) -> Self {
        self.project_id = Some(project_id.into());
        self
    }
    pub fn with_parameter(mut self, name: impl Into<String>, value: impl Into<Value>) -> Self {
        self.parameters.insert(name.into(), value.into());
        self
    }
    pub fn with_bookmark(mut self, bookmark: Bookmark) -> Self {
        self.bookmark = Some(bookmark);
        self
    }
}

/// Lossless Query API result. Values retain their protocol type tags; 64-bit graph integers
/// remain decimal strings, and nodes, relationships, paths, vectors, and temporal values retain
/// their complete structure. Inspect `rows[row][column]` with `serde_json`.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct QueryResult {
    pub catalog: Option<Value>,
    pub columns: Vec<Value>,
    pub rows: Vec<Vec<Value>>,
    pub summary: Value,
}

#[derive(Clone, Copy, Debug, Default)]
pub enum ExecutionDevice {
    #[default]
    Auto,
    Cpu,
    Metal(u32),
    Cuda(u32),
}
/// Independently selected text inference device.
#[derive(Clone, Copy, Debug, Default)]
pub enum EmbeddingDevice {
    #[default]
    Auto,
    Cpu,
    Metal(u32),
    Cuda(u32),
}
#[derive(Clone, Copy, Debug, Default, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum EmbeddingPolicy {
    #[default]
    Automatic,
    Disabled,
}

/// Options for one database process. Automatic embeddings install and warm the verified model.
#[derive(Clone, Debug)]
pub struct EmbeddedOptions {
    pub data_dir: PathBuf,
    pub execution_device: ExecutionDevice,
    pub embedding_device: EmbeddingDevice,
    pub embedding_policy: EmbeddingPolicy,
    pub snapshot_interval_ms: Option<u64>,
    pub worker_threads: Option<usize>,
}
impl EmbeddedOptions {
    pub fn new(data_dir: impl Into<PathBuf>) -> Self {
        Self {
            data_dir: data_dir.into(),
            execution_device: ExecutionDevice::Auto,
            embedding_device: EmbeddingDevice::Auto,
            embedding_policy: EmbeddingPolicy::Automatic,
            snapshot_interval_ms: None,
            worker_threads: None,
        }
    }
    pub fn with_execution_device(mut self, device: ExecutionDevice) -> Self {
        self.execution_device = device;
        self
    }
    pub fn with_embedding_policy(mut self, policy: EmbeddingPolicy) -> Self {
        self.embedding_policy = policy;
        self
    }
    pub fn with_embedding_device(mut self, device: EmbeddingDevice) -> Self {
        self.embedding_device = device;
        self
    }
    fn json(self) -> Value {
        let (device, ordinal) = match self.execution_device {
            ExecutionDevice::Auto => ("auto", 0),
            ExecutionDevice::Cpu => ("cpu", 0),
            ExecutionDevice::Metal(ordinal) => ("metal", ordinal),
            ExecutionDevice::Cuda(ordinal) => ("cuda", ordinal),
        };
        let (embedding_device, embedding_ordinal) = match self.embedding_device {
            EmbeddingDevice::Auto => ("auto", 0),
            EmbeddingDevice::Cpu => ("cpu", 0),
            EmbeddingDevice::Metal(ordinal) => ("metal", ordinal),
            EmbeddingDevice::Cuda(ordinal) => ("cuda", ordinal),
        };
        json!({"data_dir":self.data_dir,"execution_device":device,"device_ordinal":ordinal,"embedding_policy":self.embedding_policy,
            "embedding_device":embedding_device,"embedding_device_ordinal":embedding_ordinal,
            "snapshot_interval_ms":self.snapshot_interval_ms,"worker_threads":self.worker_threads})
    }
}

struct Handle(Option<u64>, Option<tokio::sync::OwnedSemaphorePermit>);

struct HandleTeardown {
    handle: u64,
    _permit: tokio::sync::OwnedSemaphorePermit,
}
struct HandleReaper {
    sender: std::sync::mpsc::SyncSender<HandleTeardown>,
    permits: std::sync::Arc<tokio::sync::Semaphore>,
    failed: std::sync::atomic::AtomicBool,
}
static HANDLE_REAPER: std::sync::OnceLock<std::result::Result<HandleReaper, String>> =
    std::sync::OnceLock::new();
#[cfg(test)]
type HandleTeardownGate = (std::sync::mpsc::Sender<()>, std::sync::mpsc::Receiver<()>);
#[cfg(test)]
static HANDLE_TEARDOWN_GATE: std::sync::Mutex<Option<HandleTeardownGate>> =
    std::sync::Mutex::new(None);

fn handle_reaper() -> Result<&'static HandleReaper> {
    let reaper = HANDLE_REAPER
        .get_or_init(|| {
            let (sender, receiver) = std::sync::mpsc::sync_channel::<HandleTeardown>(64);
            std::thread::Builder::new()
                .name("irongraph-sdk-teardown".into())
                .spawn(move || {
                    for teardown in receiver {
                        #[cfg(test)]
                        if let Some((entered, release)) =
                            HANDLE_TEARDOWN_GATE.lock().unwrap().take()
                        {
                            let _ = entered.send(());
                            let _ = release.recv();
                        }
                        let _ = std::panic::catch_unwind(|| {
                            let _ = call(json!({"action":"close","handle":teardown.handle}));
                        });
                        drop(teardown);
                    }
                })
                .map_err(|error| error.to_string())?;
            Ok(HandleReaper {
                sender,
                permits: std::sync::Arc::new(tokio::sync::Semaphore::new(64)),
                failed: std::sync::atomic::AtomicBool::new(false),
            })
        })
        .as_ref()
        .map_err(|error| Error::sdk(format!("SDK teardown worker failed: {error}")))?;
    if reaper.failed.load(std::sync::atomic::Ordering::Acquire) {
        return Err(Error::sdk("SDK teardown worker is unavailable"));
    }
    Ok(reaper)
}
#[cfg(test)]
mod embedding_configuration_tests {
    use super::*;

    #[test]
    fn native_envelope_borrows_result_and_distinguishes_null_from_missing() {
        let null: NativeResponse<'_> =
            serde_json::from_str(r#"{"ok":true,"version":"0.1.6","result":null}"#).unwrap();
        assert_eq!(null.result.unwrap().get(), "null");
        let missing: NativeResponse<'_> =
            serde_json::from_str(r#"{"ok":true,"version":"0.1.6"}"#).unwrap();
        assert!(missing.result.is_none());
        let dirty = serde_json::json!({"rows":[[{"type":"string","value":"Unicode 🦀\\\"\0".repeat(16_384)}]],"columns":[],"catalog":null,"summary":{}});
        let encoded = serde_json::json!({"ok":true,"version":"0.1.6","result":dirty}).to_string();
        let response: NativeResponse<'_> = serde_json::from_str(&encoded).unwrap();
        let raw = response.result.unwrap().get();
        assert!(raw.as_ptr() >= encoded.as_ptr());
        assert!(raw.as_bytes().as_ptr_range().end <= encoded.as_bytes().as_ptr_range().end);
        let result: QueryResult = serde_json::from_str(raw).unwrap();
        assert_eq!(serde_json::to_value(result).unwrap(), dirty);
    }

    #[test]
    fn cpu_graph_serializes_independent_embedding_device() {
        let options = EmbeddedOptions::new("test-data")
            .with_execution_device(ExecutionDevice::Cpu)
            .with_embedding_device(EmbeddingDevice::Metal(2))
            .json();
        assert_eq!(options["execution_device"], "cpu");
        assert_eq!(options["embedding_device"], "metal");
        assert_eq!(options["embedding_device_ordinal"], 2);
        assert_eq!(
            EmbeddedOptions::new("test-data").json()["embedding_device"],
            "auto"
        );
    }
}
impl Handle {
    fn open(request: Value) -> Result<Self> {
        let permit = handle_reaper()?
            .permits
            .clone()
            .try_acquire_owned()
            .map_err(|_| Error {
                code: "BACKPRESSURE".into(),
                message: "SDK native handle capacity exhausted before open".into(),
                retryable: true,
                retry_after_ms: None,
            })?;
        call(request)?["handle"]
            .as_u64()
            .filter(|handle| *handle != 0)
            .map(|handle| Self(Some(handle), Some(permit)))
            .ok_or_else(|| Error::sdk("native open returned an invalid handle"))
    }
    fn query(&self, query: Query) -> Result<QueryResult> {
        call_typed(json!({"action":"query","handle":self.0,"query":query}))
    }
    fn close(&mut self) -> Result<()> {
        if let Some(handle) = self.0.take() {
            let result = call(json!({"action":"close","handle":handle}));
            self.1.take();
            result?;
        }
        Ok(())
    }
}
impl Drop for Handle {
    fn drop(&mut self) {
        if let Some(handle) = self.0.take() {
            let teardown = HandleTeardown {
                handle,
                _permit: self
                    .1
                    .take()
                    .expect("live handle retains lifecycle admission"),
            };
            let reaper = HANDLE_REAPER
                .get()
                .expect("open initialized reaper")
                .as_ref()
                .expect("open admitted reaper");
            if let Err(error) = reaper.sender.try_send(teardown) {
                reaper
                    .failed
                    .store(true, std::sync::atomic::Ordering::Release);
                let (std::sync::mpsc::TrySendError::Full(teardown)
                | std::sync::mpsc::TrySendError::Disconnected(teardown)) = error;
                std::mem::forget(teardown);
            }
        }
    }
}

#[cfg(test)]
mod bounded_teardown_tests {
    use super::*;

    #[test]
    fn paused_native_teardown_bounds_handles_and_keeps_current_thread_responsive() {
        let directory = tempfile::tempdir().unwrap();
        let options =
            EmbeddedOptions::new(directory.path()).with_embedding_policy(EmbeddingPolicy::Disabled);
        let database = EmbeddedDatabase::open(options.clone()).unwrap();
        let (entered, observed) = std::sync::mpsc::channel();
        let (release, blocked) = std::sync::mpsc::channel();
        *HANDLE_TEARDOWN_GATE.lock().unwrap() = Some((entered, blocked));
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        runtime.block_on(async {
            drop(database);
            tokio::time::sleep(std::time::Duration::from_millis(1)).await;
        });
        observed
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
        let mut handles = Vec::new();
        for _ in 0..63 {
            handles.push(RemoteClient::api("http://127.0.0.1:1").unwrap());
        }
        let error = match RemoteClient::api("http://127.0.0.1:1") {
            Err(error) => error,
            Ok(_) => panic!("teardown did not retain handle admission"),
        };
        assert_eq!(error.code, "BACKPRESSURE");
        for handle in handles {
            handle.close().unwrap();
        }
        assert!(
            EmbeddedDatabase::open(options.clone()).is_err(),
            "instance guard released before actual teardown"
        );
        release.send(()).unwrap();
        runtime.block_on(async {
            tokio::time::timeout(std::time::Duration::from_secs(5), async {
                while handle_reaper().unwrap().permits.available_permits() != 64 {
                    tokio::time::sleep(std::time::Duration::from_millis(1)).await;
                }
            })
            .await
            .unwrap();
        });
        EmbeddedDatabase::open(options.clone())
            .unwrap()
            .close()
            .unwrap();
        EmbeddedDatabase::open(options).unwrap().close().unwrap();
    }
}

static ASYNC_JOBS: std::sync::LazyLock<std::sync::Arc<tokio::sync::Semaphore>> =
    std::sync::LazyLock::new(|| std::sync::Arc::new(tokio::sync::Semaphore::new(64)));
static ASYNC_CONTROLS: std::sync::LazyLock<std::sync::Arc<tokio::sync::Semaphore>> =
    std::sync::LazyLock::new(|| std::sync::Arc::new(tokio::sync::Semaphore::new(64)));
static ASYNC_CANCELLATIONS: std::sync::LazyLock<std::sync::Arc<tokio::sync::Semaphore>> =
    std::sync::LazyLock::new(|| std::sync::Arc::new(tokio::sync::Semaphore::new(64)));
static ASYNC_OPERATION_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

struct AsyncCancellation {
    operation: Option<(u64, String)>,
    cancelled: std::sync::Arc<std::sync::atomic::AtomicBool>,
    abort: tokio::task::AbortHandle,
    runtime: tokio::runtime::Handle,
    control: Option<tokio::sync::OwnedSemaphorePermit>,
    job: std::sync::Arc<tokio::sync::OwnedSemaphorePermit>,
}
impl Drop for AsyncCancellation {
    fn drop(&mut self) {
        let Some((handle, operation_id)) = self.operation.take() else {
            return;
        };
        if self.abort.is_finished() {
            return;
        }
        self.cancelled
            .store(true, std::sync::atomic::Ordering::Release);
        // Blocking tasks can only be aborted before they start. A running native request
        // may still be parsing its packet, so retry until registration or completion.
        self.abort.abort();
        let abort = self.abort.clone();
        let control = self.control.take();
        let job = self.job.clone();
        self.runtime.spawn_blocking(move || {
            let _control = control;
            let _job = job;
            while !abort.is_finished() {
                match call(json!({"action":"cancel","handle":handle,"operation_id":operation_id})) {
                    Ok(result) if result.as_bool() == Some(true) => break,
                    Err(_) => break,
                    _ => std::thread::sleep(std::time::Duration::from_millis(1)),
                }
            }
        });
    }
}
async fn blocking<T: Send + 'static>(
    operation: Option<(u64, String)>,
    bounded: bool,
    work: impl FnOnce() -> Result<T> + Send + 'static,
) -> Result<T> {
    let runtime = tokio::runtime::Handle::try_current()
        .map_err(|_| Error::sdk("asynchronous embedded calls require a Tokio runtime"))?;
    let admission = if bounded {
        &*ASYNC_JOBS
    } else {
        &*ASYNC_CONTROLS
    };
    let permit = admission
        .clone()
        .acquire_owned()
        .await
        .map_err(|_| Error::sdk("asynchronous SDK workers are closed"))?;
    let control = if operation.is_some() {
        Some(
            ASYNC_CANCELLATIONS
                .clone()
                .acquire_owned()
                .await
                .map_err(|_| Error::sdk("asynchronous SDK cancellation workers are closed"))?,
        )
    } else {
        None
    };
    let permit = std::sync::Arc::new(permit);
    let worker_permit = permit.clone();
    let cancelled = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let worker_cancelled = cancelled.clone();
    let task = runtime.spawn_blocking(move || {
        let _permit = worker_permit;
        if worker_cancelled.load(std::sync::atomic::Ordering::Acquire) {
            return Err(Error::sdk(
                "asynchronous operation cancelled before dispatch",
            ));
        }
        work()
    });
    let mut cancellation = AsyncCancellation {
        operation,
        control,
        job: permit,
        cancelled,
        abort: task.abort_handle(),
        runtime,
    };
    let result = task.await;
    cancellation.operation = None;
    result.map_err(|error| Error::sdk(format!("native worker failed: {error}")))?
}
fn async_options(mut options: OperationOptions) -> (OperationOptions, String) {
    let id = options
        .operation_id
        .get_or_insert_with(|| {
            format!(
                "sdk:{}:{}",
                std::process::id(),
                ASYNC_OPERATION_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            )
        })
        .clone();
    (options, id)
}
fn decode<T: serde::de::DeserializeOwned>(request: Value) -> Result<T> {
    call_typed(request)
}

/// Database with synchronous and Tokio asynchronous access to the same native owner.
/// Only one embedded instance can be active in a process. Close explicitly to observe shutdown errors.
pub struct EmbeddedDatabase(Handle);

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct OperationOptions {
    pub operation_id: Option<String>,
    pub timeout_ms: Option<u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamRecord {
    pub key: Option<Vec<u8>>,
    pub headers: BTreeMap<String, Vec<u8>>,
    pub value: Option<Vec<u8>>,
    pub create_time_ms: Option<i64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamAppend {
    pub project_id: String,
    pub topic: String,
    pub partition: i32,
    pub records: Vec<StreamRecord>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamAcknowledgement {
    pub bookmark: Bookmark,
    pub first_offset: u64,
    pub record_count: u32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamFetch {
    pub project_id: String,
    pub topic: String,
    pub partition: i32,
    pub offset: u64,
    pub max_records: usize,
    pub max_bytes: usize,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamPayload {
    pub id: u64,
    pub resolved_time_ms: i64,
    pub ingress: Value,
    pub payload: Vec<u8>,
    pub checksum: Vec<u8>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StreamPage {
    pub records: Vec<(u64, StreamPayload)>,
    pub high_watermark: u64,
    pub next_offset: u64,
    pub truncated: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct RuntimeStatus {
    pub data_dir: PathBuf,
    pub ready: bool,
    pub active_operations: usize,
    pub worker_threads: usize,
}

impl EmbeddedDatabase {
    pub async fn open_async(options: EmbeddedOptions) -> Result<Self> {
        blocking(None, true, move || Self::open(options)).await
    }
    pub async fn query_async(&self, query: Query) -> Result<QueryResult> {
        self.query_with_options_async(query, OperationOptions::default())
            .await
    }
    pub async fn query_with_options_async(
        &self,
        query: Query,
        options: OperationOptions,
    ) -> Result<QueryResult> {
        let handle = self.0.0.ok_or_else(|| Error::sdk("database is closed"))?;
        let (options, id) = async_options(options);
        blocking(Some((handle, id)), true, move || {
            decode(json!({"action":"query","handle":handle,"query":query,"options":options}))
        })
        .await
    }
    pub async fn stream_append_async(
        &self,
        request: StreamAppend,
        options: OperationOptions,
    ) -> Result<StreamAcknowledgement> {
        let handle = self.0.0.ok_or_else(|| Error::sdk("database is closed"))?;
        let (options, id) = async_options(options);
        blocking(Some((handle, id)), true, move || decode(json!({"action":"stream_append","handle":handle,"request":request,"options":options}))).await
    }
    pub async fn stream_fetch_async(
        &self,
        request: StreamFetch,
        options: OperationOptions,
    ) -> Result<StreamPage> {
        let handle = self.0.0.ok_or_else(|| Error::sdk("database is closed"))?;
        let (options, id) = async_options(options);
        blocking(Some((handle, id)), true, move || decode(json!({"action":"stream_fetch","handle":handle,"request":request,"options":options}))).await
    }
    pub async fn status_async(&self) -> Result<RuntimeStatus> {
        let handle = self.0.0;
        blocking(None, false, move || {
            decode(json!({"action":"status","handle":handle}))
        })
        .await
    }
    pub async fn cancel_async(&self, operation_id: &str) -> Result<bool> {
        let handle = self.0.0;
        let operation_id = operation_id.to_owned();
        blocking(None, false, move || {
            decode(json!({"action":"cancel","handle":handle,"operation_id":operation_id}))
        })
        .await
    }
    pub async fn snapshot_async(&self) -> Result<Bookmark> {
        let handle = self.0.0;
        blocking(None, true, move || {
            decode(json!({"action":"snapshot","handle":handle}))
        })
        .await
    }
    pub async fn flush_async(&self) -> Result<()> {
        let handle = self.0.0;
        blocking(None, true, move || {
            call(json!({"action":"flush","handle":handle})).map(|_| ())
        })
        .await
    }
    pub async fn close_async(self) -> Result<()> {
        blocking(None, false, move || self.close()).await
    }
    pub fn open(options: EmbeddedOptions) -> Result<Self> {
        Handle::open(json!({"action":"open_embedded","options":options.json()})).map(Self)
    }
    pub fn query(&self, query: Query) -> Result<QueryResult> {
        self.0.query(query)
    }
    pub fn query_with_options(
        &self,
        query: Query,
        options: OperationOptions,
    ) -> Result<QueryResult> {
        call_typed(json!({"action":"query","handle":self.0.0,"query":query,"options":options}))
    }
    pub fn stream_append(
        &self,
        request: StreamAppend,
        options: OperationOptions,
    ) -> Result<StreamAcknowledgement> {
        call_typed(
            json!({"action":"stream_append","handle":self.0.0,"request":request,"options":options}),
        )
    }
    pub fn stream_fetch(
        &self,
        request: StreamFetch,
        options: OperationOptions,
    ) -> Result<StreamPage> {
        call_typed(
            json!({"action":"stream_fetch","handle":self.0.0,"request":request,"options":options}),
        )
    }
    pub fn status(&self) -> Result<RuntimeStatus> {
        call_typed(json!({"action":"status","handle":self.0.0}))
    }
    pub fn cancel(&self, operation_id: &str) -> Result<bool> {
        call_typed(json!({"action":"cancel","handle":self.0.0,"operation_id":operation_id}))
    }
    pub fn snapshot(&self) -> Result<Bookmark> {
        call_typed(json!({"action":"snapshot","handle":self.0.0}))
    }
    pub fn flush(&self) -> Result<()> {
        call(json!({"action":"flush","handle":self.0.0}))?;
        Ok(())
    }
    pub fn close(mut self) -> Result<()> {
        self.0.close()
    }
}

/// PEM paths for mutual TLS. Credentials are kept outside graph data.
#[derive(Clone, Debug, Serialize)]
pub struct MutualTls {
    pub certificate: PathBuf,
    pub private_key: PathBuf,
    pub certificate_authority: PathBuf,
}
impl MutualTls {
    pub fn new(
        certificate: impl Into<PathBuf>,
        private_key: impl Into<PathBuf>,
        certificate_authority: impl Into<PathBuf>,
    ) -> Self {
        Self {
            certificate: certificate.into(),
            private_key: private_key.into(),
            certificate_authority: certificate_authority.into(),
        }
    }
}

/// Remote Query API or Bolt connection. Plain connections are restricted to loopback addresses;
/// remote connections require mutual TLS. Bolt requests require `Query::with_project`.
pub struct RemoteClient(Handle);
impl RemoteClient {
    fn connect(transport: &str, endpoint: &str, tls: Option<&MutualTls>) -> Result<Self> {
        Handle::open(json!({"action":"open_remote","options":{"transport":transport,"endpoint":endpoint,"tls":tls}})).map(Self)
    }
    pub fn api(endpoint: &str) -> Result<Self> {
        Self::connect("api", endpoint, None)
    }
    pub fn bolt(endpoint: &str) -> Result<Self> {
        Self::connect("bolt", endpoint, None)
    }
    pub fn api_mtls(endpoint: &str, tls: &MutualTls) -> Result<Self> {
        Self::connect("api", endpoint, Some(tls))
    }
    pub fn bolt_mtls(endpoint: &str, tls: &MutualTls) -> Result<Self> {
        Self::connect("bolt", endpoint, Some(tls))
    }
    pub fn query(&self, query: Query) -> Result<QueryResult> {
        self.0.query(query)
    }
    pub fn close(mut self) -> Result<()> {
        self.0.close()
    }
}
