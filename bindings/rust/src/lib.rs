//! IronGraph SDK for embedded databases and remote API/Bolt connections.
//!
//! Documents, vector search, graph algorithms, projects, and administration use the same
//! [`Query`] interface. Calls block the current thread; use your async runtime's blocking
//! task facility when calling from async applications.

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
    let response: Value =
        serde_json::from_slice(bytes).map_err(|error| Error::sdk(error.to_string()))?;
    if response.get("version").and_then(Value::as_str) != Some(env!("CARGO_PKG_VERSION")) {
        return Err(Error::sdk("native library package version mismatch"));
    }
    match (status, response.get("ok").and_then(Value::as_bool)) {
        (0, Some(true)) => response
            .get("result")
            .cloned()
            .ok_or_else(|| Error::sdk("native result is missing")),
        (1, Some(false)) => Err(serde_json::from_value(response["error"].clone())
            .map_err(|error| Error::sdk(error.to_string()))?),
        _ => Err(Error::sdk("native response status is inconsistent")),
    }
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

/// Optional result limits. Omitted fields are unbounded; explicit limits must be nonzero.
#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize)]
pub struct QueryLimits {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rows: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nodes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub edges: Option<u64>,
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
    pub limits: QueryLimits,
}
impl Query {
    pub fn new(cypher: impl Into<String>) -> Self {
        Self {
            cypher: cypher.into(),
            parameters: BTreeMap::new(),
            project_id: None,
            bookmark: None,
            consistency: CommitAcknowledgement::Published,
            limits: QueryLimits::default(),
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
    pub fn with_limits(mut self, limits: QueryLimits) -> Self {
        self.limits = limits;
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
    pub embedding_policy: EmbeddingPolicy,
    pub device_memory_limit_bytes: Option<usize>,
    pub device_reserved_bytes: Option<usize>,
    pub max_write_bytes: Option<usize>,
    pub request_timeout_ms: Option<u64>,
    pub startup_timeout_ms: Option<u64>,
    pub snapshot_interval_ms: Option<u64>,
    pub max_concurrent_operations: Option<usize>,
    pub worker_threads: Option<usize>,
}
impl EmbeddedOptions {
    pub fn new(data_dir: impl Into<PathBuf>) -> Self {
        Self {
            data_dir: data_dir.into(),
            execution_device: ExecutionDevice::Auto,
            embedding_policy: EmbeddingPolicy::Automatic,
            device_memory_limit_bytes: None,
            device_reserved_bytes: None,
            max_write_bytes: None,
            request_timeout_ms: None,
            startup_timeout_ms: None,
            snapshot_interval_ms: None,
            max_concurrent_operations: None,
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
    fn json(self) -> Value {
        let (device, ordinal) = match self.execution_device {
            ExecutionDevice::Auto => ("auto", 0),
            ExecutionDevice::Cpu => ("cpu", 0),
            ExecutionDevice::Metal(ordinal) => ("metal", ordinal),
            ExecutionDevice::Cuda(ordinal) => ("cuda", ordinal),
        };
        json!({"data_dir":self.data_dir,"execution_device":device,"device_ordinal":ordinal,"embedding_policy":self.embedding_policy,
            "device_memory_limit_bytes":self.device_memory_limit_bytes,"device_reserved_bytes":self.device_reserved_bytes,"max_write_bytes":self.max_write_bytes,
            "request_timeout_ms":self.request_timeout_ms,"startup_timeout_ms":self.startup_timeout_ms,"snapshot_interval_ms":self.snapshot_interval_ms,
            "max_concurrent_operations":self.max_concurrent_operations,"worker_threads":self.worker_threads})
    }
}

struct Handle(Option<u64>);
impl Handle {
    fn open(request: Value) -> Result<Self> {
        call(request)?["handle"]
            .as_u64()
            .filter(|handle| *handle != 0)
            .map(|handle| Self(Some(handle)))
            .ok_or_else(|| Error::sdk("native open returned an invalid handle"))
    }
    fn query(&self, query: Query) -> Result<QueryResult> {
        serde_json::from_value(call(
            json!({"action":"query","handle":self.0,"query":query}),
        )?)
        .map_err(|error| Error::sdk(error.to_string()))
    }
    fn close(&mut self) -> Result<()> {
        if let Some(handle) = self.0.take() {
            call(json!({"action":"close","handle":handle}))?;
        }
        Ok(())
    }
}
impl Drop for Handle {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

/// Synchronous database backed by the bundled native engine. Only one embedded instance can be
/// active in a process. Close explicitly to observe snapshot or shutdown errors.
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
    pub max_concurrent_operations: usize,
    pub worker_threads: usize,
}

impl EmbeddedDatabase {
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
        serde_json::from_value(call(
            json!({"action":"query","handle":self.0.0,"query":query,"options":options}),
        )?)
        .map_err(|error| Error::sdk(error.to_string()))
    }
    pub fn stream_append(
        &self,
        request: StreamAppend,
        options: OperationOptions,
    ) -> Result<StreamAcknowledgement> {
        serde_json::from_value(call(
            json!({"action":"stream_append","handle":self.0.0,"request":request,"options":options}),
        )?)
        .map_err(|error| Error::sdk(error.to_string()))
    }
    pub fn stream_fetch(
        &self,
        request: StreamFetch,
        options: OperationOptions,
    ) -> Result<StreamPage> {
        serde_json::from_value(call(
            json!({"action":"stream_fetch","handle":self.0.0,"request":request,"options":options}),
        )?)
        .map_err(|error| Error::sdk(error.to_string()))
    }
    pub fn status(&self) -> Result<RuntimeStatus> {
        serde_json::from_value(call(json!({"action":"status","handle":self.0.0}))?)
            .map_err(|error| Error::sdk(error.to_string()))
    }
    pub fn cancel(&self, operation_id: &str) -> Result<bool> {
        serde_json::from_value(call(
            json!({"action":"cancel","handle":self.0.0,"operation_id":operation_id}),
        )?)
        .map_err(|error| Error::sdk(error.to_string()))
    }
    pub fn snapshot(&self) -> Result<Bookmark> {
        serde_json::from_value(call(json!({"action":"snapshot","handle":self.0.0}))?)
            .map_err(|error| Error::sdk(error.to_string()))
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
