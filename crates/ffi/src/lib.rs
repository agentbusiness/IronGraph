//! Private, versioned binary boundary. No private Rust types cross this ABI.

use std::{
    collections::HashMap,
    panic::{AssertUnwindSafe, catch_unwind},
    path::PathBuf,
    sync::{Arc, Mutex, OnceLock},
    time::Duration,
};

use irongraph_client::{ClientError, MutualTls, Query, RemoteClient};
use irongraph_embedded::{
    EmbeddedDatabase, EmbeddedError, EmbeddedOptions, EmbeddingPolicy, ExecutionDevice,
};
use serde::Deserialize;
use serde_json::{Value, json};

const ABI_VERSION: u32 = 1;

/// Owned UTF-8 JSON bytes. Free exactly once using `irongraph_buffer_free_v1`.
#[repr(C)]
pub struct Buffer {
    pub data: *mut u8,
    pub len: usize,
}

enum Connection {
    Embedded(EmbeddedDatabase),
    Remote(RemoteClient),
}
type SharedConnection = Arc<Mutex<Option<Connection>>>;
#[derive(Default)]
struct Registry {
    next: u64,
    connections: HashMap<u64, SharedConnection>,
}

fn registry() -> &'static Mutex<Registry> {
    static REGISTRY: OnceLock<Mutex<Registry>> = OnceLock::new();
    REGISTRY.get_or_init(|| Mutex::new(Registry::default()))
}

#[derive(Debug)]
struct Failure {
    code: String,
    message: String,
    retryable: bool,
    retry_after_ms: Option<u64>,
}
impl Failure {
    fn new(code: &str, message: impl Into<String>) -> Self {
        Self {
            code: code.into(),
            message: message.into(),
            retryable: false,
            retry_after_ms: None,
        }
    }
    fn json(self) -> Value {
        json!({"code": self.code, "message": self.message, "retryable": self.retryable, "retry_after_ms": self.retry_after_ms})
    }
}
impl From<ClientError> for Failure {
    fn from(error: ClientError) -> Self {
        if let ClientError::Database {
            code,
            message,
            retryable,
            retry_after_ms,
        } = error
        {
            Self {
                code,
                message,
                retryable,
                retry_after_ms,
            }
        } else {
            let code = match &error {
                ClientError::Configuration(_) => "CONFIGURATION",
                ClientError::Http(_) | ClientError::Bolt(_) | ClientError::Io(_) => "TRANSPORT",
                _ => "RESULT_DECODING",
            };
            Self::new(code, error.to_string())
        }
    }
}
impl From<EmbeddedError> for Failure {
    fn from(error: EmbeddedError) -> Self {
        match error {
            EmbeddedError::Query(error) => error.into(),
            EmbeddedError::Configuration(message) => Self::new("CONFIGURATION", message),
            EmbeddedError::ProcessInstanceActive => Self::new("INSTANCE_ACTIVE", error.to_string()),
            EmbeddedError::Engine(error) => Self {
                code: serde_json::to_value(error.code)
                    .ok()
                    .and_then(|value| value.as_str().map(str::to_owned))
                    .unwrap_or_else(|| "ENGINE".into()),
                message: error.message.into_owned(),
                retryable: error.retryable,
                retry_after_ms: error.retry_after_ms,
            },
            EmbeddedError::Runtime(error) => Self::new("RUNTIME", error.to_string()),
        }
    }
}
impl From<serde_json::Error> for Failure {
    fn from(error: serde_json::Error) -> Self {
        Self::new("INVALID_REQUEST", error.to_string())
    }
}
type Result<T> = std::result::Result<T, Failure>;

#[derive(Deserialize)]
#[serde(tag = "action", rename_all = "snake_case", deny_unknown_fields)]
enum Request {
    OpenEmbedded { options: OpenOptions },
    OpenRemote { options: RemoteOptions },
    Query { handle: u64, query: Query },
    Snapshot { handle: u64 },
    Close { handle: u64 },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OpenOptions {
    data_dir: PathBuf,
    #[serde(default = "default_device")]
    execution_device: String,
    #[serde(default)]
    device_ordinal: u32,
    #[serde(default = "default_embedding")]
    embedding_policy: String,
    device_memory_limit_bytes: Option<usize>,
    device_reserved_bytes: Option<usize>,
    max_write_bytes: Option<usize>,
    request_timeout_ms: Option<u64>,
    startup_timeout_ms: Option<u64>,
    snapshot_interval_ms: Option<u64>,
}
fn default_device() -> String {
    "auto".into()
}
fn default_embedding() -> String {
    "automatic".into()
}

impl OpenOptions {
    fn into_options(self) -> Result<EmbeddedOptions> {
        let device = match self.execution_device.as_str() {
            "auto" => ExecutionDevice::Auto,
            "cpu" => ExecutionDevice::Cpu,
            "metal" => ExecutionDevice::Metal(self.device_ordinal),
            "cuda" => ExecutionDevice::Cuda(self.device_ordinal),
            _ => {
                return Err(Failure::new(
                    "CONFIGURATION",
                    "execution_device must be auto, cpu, metal, or cuda",
                ));
            }
        };
        let embedding = match self.embedding_policy.as_str() {
            "automatic" => EmbeddingPolicy::Automatic,
            "disabled" => EmbeddingPolicy::Disabled,
            _ => {
                return Err(Failure::new(
                    "CONFIGURATION",
                    "embedding_policy must be automatic or disabled",
                ));
            }
        };
        let mut options = EmbeddedOptions::new(self.data_dir)
            .with_execution_device(device)
            .with_embedding_policy(embedding);
        if let Some(value) = self.device_memory_limit_bytes {
            options.device_memory_limit_bytes = value;
        }
        if let Some(value) = self.device_reserved_bytes {
            options.device_reserved_bytes = value;
        }
        if let Some(value) = self.max_write_bytes {
            options.max_write_bytes = value;
        }
        if let Some(value) = self.request_timeout_ms {
            options.request_timeout = Duration::from_millis(value);
        }
        if let Some(value) = self.startup_timeout_ms {
            options.startup_timeout = Duration::from_millis(value);
        }
        if let Some(value) = self.snapshot_interval_ms {
            options.snapshot_interval = Duration::from_millis(value);
        }
        Ok(options)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RemoteOptions {
    transport: String,
    endpoint: String,
    tls: Option<TlsOptions>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TlsOptions {
    certificate: PathBuf,
    private_key: PathBuf,
    certificate_authority: PathBuf,
}
impl RemoteOptions {
    fn connect(self) -> Result<RemoteClient> {
        let tls = self
            .tls
            .map(|tls| MutualTls::new(tls.certificate, tls.private_key, tls.certificate_authority));
        Ok(match (self.transport.as_str(), tls) {
            ("api", None) => RemoteClient::api(&self.endpoint)?,
            ("bolt", None) => RemoteClient::bolt(&self.endpoint)?,
            ("api", Some(tls)) => RemoteClient::api_mtls(&self.endpoint, &tls)?,
            ("bolt", Some(tls)) => RemoteClient::bolt_mtls(&self.endpoint, &tls)?,
            _ => {
                return Err(Failure::new(
                    "CONFIGURATION",
                    "transport must be api or bolt",
                ));
            }
        })
    }
}

fn insert(connection: Connection) -> Result<Value> {
    let mut registry = registry()
        .lock()
        .map_err(|_| Failure::new("INTERNAL", "handle registry is poisoned"))?;
    registry.next = registry
        .next
        .checked_add(1)
        .ok_or_else(|| Failure::new("INTERNAL", "handle identifiers exhausted"))?;
    let handle = registry.next;
    registry
        .connections
        .insert(handle, Arc::new(Mutex::new(Some(connection))));
    Ok(json!({"handle": handle}))
}

fn connection(handle: u64, remove: bool) -> Result<SharedConnection> {
    let mut registry = registry()
        .lock()
        .map_err(|_| Failure::new("INTERNAL", "handle registry is poisoned"))?;
    let connection = if remove {
        registry.connections.remove(&handle)
    } else {
        registry.connections.get(&handle).cloned()
    };
    connection.ok_or_else(|| Failure::new("INVALID_HANDLE", "connection is closed or unknown"))
}

fn dispatch(bytes: &[u8]) -> Result<Value> {
    match serde_json::from_slice::<Request>(bytes)? {
        Request::OpenEmbedded { options } => insert(Connection::Embedded(EmbeddedDatabase::open(
            options.into_options()?,
        )?)),
        Request::OpenRemote { options } => insert(Connection::Remote(options.connect()?)),
        Request::Query { handle, query } => {
            let shared = connection(handle, false)?;
            let guard = shared
                .lock()
                .map_err(|_| Failure::new("INTERNAL", "connection is poisoned"))?;
            let connection = guard
                .as_ref()
                .ok_or_else(|| Failure::new("INVALID_HANDLE", "connection is closed"))?;
            let result = match connection {
                Connection::Embedded(database) => database.query(query).map_err(Failure::from)?,
                Connection::Remote(client) => client.query(query).map_err(Failure::from)?,
            };
            Ok(serde_json::to_value(result)?)
        }
        Request::Snapshot { handle } => {
            let shared = connection(handle, false)?;
            let guard = shared
                .lock()
                .map_err(|_| Failure::new("INTERNAL", "connection is poisoned"))?;
            match guard.as_ref() {
                Some(Connection::Embedded(database)) => Ok(serde_json::to_value(
                    database.snapshot().map_err(Failure::from)?,
                )?),
                Some(Connection::Remote(_)) => Err(Failure::new(
                    "UNSUPPORTED_OPERATION",
                    "snapshot requires an embedded database",
                )),
                None => Err(Failure::new("INVALID_HANDLE", "connection is closed")),
            }
        }
        Request::Close { handle } => {
            let shared = connection(handle, true)?;
            let mut guard = shared
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if let Some(Connection::Embedded(database)) = guard.take() {
                database.close().map_err(Failure::from)?;
            }
            Ok(Value::Null)
        }
    }
}

fn response(bytes: &[u8]) -> (i32, Vec<u8>) {
    guarded_response(|| dispatch(bytes))
}

fn guarded_response(operation: impl FnOnce() -> Result<Value>) -> (i32, Vec<u8>) {
    let result = catch_unwind(AssertUnwindSafe(operation))
        .unwrap_or_else(|_| Err(Failure::new("INTERNAL_PANIC", "native operation panicked")));
    let (status, value) = match result {
        Ok(result) => (
            0,
            json!({"ok": true, "version": env!("CARGO_PKG_VERSION"), "result": result}),
        ),
        Err(error) => (
            1,
            json!({"ok": false, "version": env!("CARGO_PKG_VERSION"), "error": error.json()}),
        ),
    };
    (status, value.to_string().into_bytes())
}

#[unsafe(no_mangle)]
pub extern "C" fn irongraph_abi_version() -> u32 {
    ABI_VERSION
}

/// NUL-terminated package version in static storage. Never free or modify this pointer.
#[unsafe(no_mangle)]
pub extern "C" fn irongraph_package_version_v1() -> *const std::ffi::c_char {
    concat!(env!("CARGO_PKG_VERSION"), "\0").as_ptr().cast()
}

/// Execute one JSON request. Status 0 means success, 1 a structured error, and 2 invalid pointers.
///
/// # Safety
/// `request` must point to `len` readable bytes (or be null when length is zero), and `out`
/// must point to a writable, aligned Buffer. They must not overlap. The returned buffer belongs
/// to the caller until freed with `irongraph_buffer_free_v1`; do not modify its pointer or length.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn irongraph_call_v1(
    request: *const u8,
    len: usize,
    out: *mut Buffer,
) -> i32 {
    if out.is_null() || (request.is_null() && len != 0) || len > isize::MAX as usize {
        return 2;
    }
    let bytes = if len == 0 {
        &[]
    } else {
        unsafe { std::slice::from_raw_parts(request, len) }
    };
    let (status, bytes) = response(bytes);
    let boxed = bytes.into_boxed_slice();
    let len = boxed.len();
    let data = Box::into_raw(boxed).cast::<u8>();
    unsafe {
        out.write(Buffer { data, len });
    }
    status
}

/// Release a response allocated by this library. A null pointer is accepted.
///
/// # Safety
/// A non-null buffer must be an unchanged, still-owned result of `irongraph_call_v1`.
/// It must be freed exactly once and never read after this call.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn irongraph_buffer_free_v1(buffer: Buffer) {
    if !buffer.data.is_null() {
        unsafe {
            drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(
                buffer.data,
                buffer.len,
            )));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn call(value: Value) -> Value {
        let bytes = value.to_string().into_bytes();
        let mut out = Buffer {
            data: std::ptr::null_mut(),
            len: 0,
        };
        let status = unsafe { irongraph_call_v1(bytes.as_ptr(), bytes.len(), &mut out) };
        assert!(status == 0 || status == 1);
        let value =
            serde_json::from_slice(unsafe { std::slice::from_raw_parts(out.data, out.len) })
                .unwrap();
        unsafe {
            irongraph_buffer_free_v1(out);
        }
        value
    }
    #[test]
    fn abi_rejects_bad_requests_and_handles() {
        assert_eq!(irongraph_abi_version(), 1);
        assert_eq!(
            unsafe { irongraph_call_v1(std::ptr::null(), 1, std::ptr::null_mut()) },
            2
        );
        assert_eq!(
            call(json!({"action":"query", "handle":0,"query":{"cypher":"SHOW PROJECTS"}}))["error"]
                ["code"],
            "INVALID_HANDLE"
        );
        assert_eq!(
            call(json!({"action":"oops"}))["error"]["code"],
            "INVALID_REQUEST"
        );
        assert_eq!(
            call(
                json!({"action":"open_remote","options":{"transport":"api","endpoint":"http://example.com"}})
            )["error"]["code"],
            "CONFIGURATION"
        );
        let (status, bytes) = guarded_response(|| panic!("private panic detail"));
        assert_eq!(status, 1);
        let error: Value = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(error["error"]["code"], "INTERNAL_PANIC");
        assert!(
            !String::from_utf8(bytes)
                .unwrap()
                .contains("private panic detail")
        );
    }
    #[test]
    fn native_database_roundtrip_and_owned_buffers() {
        let directory = tempfile::tempdir().unwrap();
        let options = json!({"data_dir":directory.path(),"execution_device":"cpu","embedding_policy":"disabled"});
        let opened = call(json!({"action":"open_embedded","options":options}));
        assert_eq!(opened["ok"], true, "{opened}");
        let handle = opened["result"]["handle"].as_u64().unwrap();
        for cypher in [
            "CREATE PROJECT sdk",
            "USE sdk CREATE (:Document {body: 'native persistence', amount: 9007199254740993})",
        ] {
            let result = call(json!({"action":"query","handle":handle,"query":{"cypher":cypher}}));
            assert_eq!(result["ok"], true, "{result}");
        }
        assert_eq!(
            call(json!({"action":"snapshot","handle":handle}))["ok"],
            true
        );
        assert_eq!(call(json!({"action":"close","handle":handle}))["ok"], true);
        assert_eq!(
            call(json!({"action":"close","handle":handle}))["error"]["code"],
            "INVALID_HANDLE"
        );
        let reopened = call(json!({"action":"open_embedded","options":options}));
        let handle = reopened["result"]["handle"].as_u64().unwrap();
        let result = call(
            json!({"action":"query","handle":handle,"query":{"cypher":"USE sdk MATCH (d:Document) RETURN d.body, d.amount"}}),
        );
        assert_eq!(result["ok"], true, "{result}");
        assert!(
            result["result"]["rows"]
                .to_string()
                .contains("9007199254740993")
        );
        assert_eq!(call(json!({"action":"close","handle":handle}))["ok"], true);
    }
}
