use std::sync::Arc;

use irongraph_client::{MutualTls, Query, RemoteClient as RustRemoteClient};
use irongraph_embedded::{
    AdapterDatabaseOwner, AdapterOwnerAdmission, EmbeddedDatabase as RustEmbeddedDatabase,
    EmbeddedOptions, EmbeddingDevice, EmbeddingPolicy, ExecutionDevice, OperationOptions,
    StreamAppend, StreamFetch,
};
use irongraph_types::ProjectId;
use napi::{Error, Result, Status};
use napi_derive::napi;
use uuid::Uuid;

/// V8 decodes ordinary rows directly; oversized rows retain the native value conversion.
pub struct QueryOutput {
    header: serde_json::Value,
    rows: Vec<RowOutput>,
}

enum RowOutput {
    Json(String),
    Value {
        value: serde_json::Value,
        charge: usize,
    },
}

const CONVERSION_BYTES: usize = 64 * 1024;
const CONVERSION_ROWS: usize = 256;

impl RowOutput {
    fn native(value: serde_json::Value) -> Self {
        // A conservative scheduling charge; it never rejects or truncates a value.
        let mut charge = 0_usize;
        conversion_charge(&value, &mut charge);
        Self::Value { value, charge }
    }
    fn charge(&self) -> usize {
        match self {
            Self::Json(text) => text.len(),
            Self::Value { charge, .. } => *charge,
        }
    }
}

fn conversion_charge(value: &serde_json::Value, charge: &mut usize) {
    if *charge >= CONVERSION_BYTES {
        return;
    }
    *charge = charge.saturating_add(match value {
        serde_json::Value::String(text) => text.len().saturating_mul(6),
        serde_json::Value::Array(_) | serde_json::Value::Object(_) => 16,
        _ => 64,
    });
    match value {
        serde_json::Value::Array(values) => {
            for value in values {
                conversion_charge(value, charge);
                if *charge >= CONVERSION_BYTES {
                    break;
                }
            }
        }
        serde_json::Value::Object(values) => {
            for (key, value) in values {
                *charge = charge.saturating_add(key.len().saturating_mul(6));
                conversion_charge(value, charge);
                if *charge >= CONVERSION_BYTES {
                    break;
                }
            }
        }
        _ => {}
    }
}

struct RowBuffer(Vec<u8>);

struct JavascriptNumbers;
impl serde_json::ser::Formatter for JavascriptNumbers {
    fn write_f32<W: ?Sized + std::io::Write>(
        &mut self,
        writer: &mut W,
        value: f32,
    ) -> std::io::Result<()> {
        // Match native N-API conversion: the original f32 is widened exactly to a JS double.
        self.write_f64(writer, f64::from(value))
    }
}

impl std::io::Write for RowBuffer {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if self.0.len().saturating_add(bytes.len()) > 1024 * 1024 {
            return Err(std::io::Error::other("use native conversion for this row"));
        }
        self.0.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl QueryOutput {
    fn prepare(mut result: irongraph_client::QueryResult) -> Result<Self> {
        // Native conversion avoids JSON encoding/parsing for small result sets. Large sets use
        // V8's decoder to avoid millions of individual N-API property operations.
        let native_rows = result.rows.len() <= 16;
        let rows = std::mem::take(&mut result.rows)
            .into_iter()
            .map(|row| {
                if native_rows {
                    return serde_json::to_value(row)
                        .map(RowOutput::native)
                        .map_err(node_error);
                }
                let mut buffer = RowBuffer(Vec::new());
                let mut serializer =
                    serde_json::Serializer::with_formatter(&mut buffer, JavascriptNumbers);
                if serde::Serialize::serialize(&row, &mut serializer).is_ok() {
                    String::from_utf8(buffer.0)
                        .map(RowOutput::Json)
                        .map_err(node_error)
                } else {
                    serde_json::to_value(row)
                        .map(RowOutput::native)
                        .map_err(node_error)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            header: serde_json::to_value(result).map_err(node_error)?,
            rows,
        })
    }
}

impl napi::bindgen_prelude::TypeName for QueryOutput {
    fn type_name() -> &'static str {
        "QueryResult"
    }
    fn value_type() -> napi::ValueType {
        napi::ValueType::Object
    }
}

impl napi::bindgen_prelude::ToNapiValue for QueryOutput {
    unsafe fn to_napi_value(
        env: napi::sys::napi_env,
        value: Self,
    ) -> Result<napi::sys::napi_value> {
        use napi::bindgen_prelude::{FromNapiValue, JsObjectValue, Object, ToNapiValue};
        let environment = napi::Env::from_raw(env);
        // SAFETY: N-API supplies a live environment, and each conversion uses that environment.
        let mut result = unsafe {
            Object::from_napi_value(env, ToNapiValue::to_napi_value(env, value.header)?)
        }?;
        let length = u32::try_from(value.rows.len()).map_err(node_error)?;
        let yield_conversion = value.rows.len() > CONVERSION_ROWS
            || value
                .rows
                .iter()
                .scan(0_usize, |total, row| {
                    *total = total.saturating_add(row.charge());
                    Some(*total)
                })
                .any(|total| total >= CONVERSION_BYTES);
        if !yield_conversion {
            // SAFETY: Conversion and the resulting array use this live JavaScript environment.
            let rows = unsafe { convert_rows(env, value.rows)? };
            result.set_named_property("rows", unsafe {
                napi::Unknown::from_napi_value(env, rows)?
            })?;
            return unsafe { ToNapiValue::to_napi_value(env, result) };
        }
        result.set_named_property("rows", environment.create_array(length)?)?;
        let remaining = std::cell::RefCell::new(value.rows.into_iter().peekable());
        let next = environment.create_function_from_closure::<(), QueryChunk, _>(
            "nextQueryChunk",
            move |_| {
                let mut remaining = remaining.try_borrow_mut().map_err(node_error)?;
                let mut rows = Vec::new();
                let mut bytes = 0_usize;
                while rows.len() < CONVERSION_ROWS {
                    let Some(row) = remaining.peek() else { break };
                    if !rows.is_empty() && bytes.saturating_add(row.charge()) > CONVERSION_BYTES {
                        break;
                    }
                    bytes = bytes.saturating_add(row.charge());
                    rows.push(remaining.next().expect("peeked row"));
                }
                Ok(QueryChunk {
                    rows,
                    done: remaining.peek().is_none(),
                })
            },
        )?;
        // Native async promises adopt this promise. Only bounded conversion turns run on V8;
        // the unconsumed rows stay owned by the closure and are dropped when it is collected.
        let decode: QueryDecode<'_> = environment.run_script(include_str!("decode_query.js"))?;
        let promise = decode.call((result, next).into())?;
        unsafe { ToNapiValue::to_napi_value(env, promise) }
    }
}

type QueryDecode<'env> = napi::bindgen_prelude::Function<
    'env,
    napi::bindgen_prelude::FnArgs<(
        napi::bindgen_prelude::Object<'env>,
        napi::bindgen_prelude::Function<'env, (), QueryChunk>,
    )>,
    napi::Unknown<'env>,
>;

struct QueryChunk {
    rows: Vec<RowOutput>,
    done: bool,
}

impl napi::bindgen_prelude::ToNapiValue for QueryChunk {
    unsafe fn to_napi_value(
        env: napi::sys::napi_env,
        value: Self,
    ) -> Result<napi::sys::napi_value> {
        use napi::bindgen_prelude::{FromNapiValue, JsObjectValue, Object, ToNapiValue};
        let environment = napi::Env::from_raw(env);
        let mut result = Object::new(&environment)?;
        result.set_named_property("done", value.done)?;
        // SAFETY: The callback and its rows are converted in the same live environment.
        result.set_named_property("rows", unsafe {
            napi::Unknown::from_napi_value(env, convert_rows(env, value.rows)?)?
        })?;
        unsafe { ToNapiValue::to_napi_value(env, result) }
    }
}

unsafe fn convert_rows(
    env: napi::sys::napi_env,
    values: Vec<RowOutput>,
) -> Result<napi::sys::napi_value> {
    use napi::bindgen_prelude::{FromNapiValue, Function, JsObjectValue, Object, ToNapiValue};
    let environment = napi::Env::from_raw(env);
    let json: Object = environment.get_global()?.get_named_property("JSON")?;
    let parse: Function<String, napi::Unknown> = json.get_named_property("parse")?;
    let mut rows = environment.create_array(u32::try_from(values.len()).map_err(node_error)?)?;
    for (index, row) in values.into_iter().enumerate() {
        let row = match row {
            RowOutput::Json(text) => parse.call(text)?,
            RowOutput::Value { value, .. } => unsafe {
                napi::Unknown::from_napi_value(env, ToNapiValue::to_napi_value(env, value)?)?
            },
        };
        rows.set(index as u32, row)?;
    }
    unsafe { ToNapiValue::to_napi_value(env, rows) }
}

fn node_error(error: impl std::fmt::Display) -> Error {
    Error::new(Status::GenericFailure, error.to_string())
}

fn query_request(
    cypher: String,
    project_id: Option<String>,
    parameters: Option<serde_json::Value>,
) -> Result<Query> {
    let mut query = Query::new(cypher);
    if let Some(project_id) = project_id {
        query.project_id = Some(ProjectId(Uuid::parse_str(&project_id).map_err(node_error)?));
    }
    if let Some(parameters) = parameters {
        query.parameters = serde_json::from_value(parameters).map_err(node_error)?;
    }
    Ok(query)
}

#[cfg(test)]
mod control_tests {
    use super::*;

    #[test]
    fn status_and_cancel_never_wait_for_close_owner_lock() {
        let directory =
            std::env::temp_dir().join(format!("irongraph-node-control-{}", Uuid::new_v4()));
        let owner = RustEmbeddedDatabase::open(
            EmbeddedOptions::new(&directory).with_embedding_policy(EmbeddingPolicy::Disabled),
        )
        .unwrap();
        let database = EmbeddedDatabase {
            database: Arc::new(AdapterDatabaseOwner::new(
                owner,
                AdapterOwnerAdmission::reserve().unwrap(),
            )),
            workers: Arc::new(napi::tokio::sync::Semaphore::new(1)),
        };
        let closing = database.database.write().unwrap();
        assert!(database.status().is_err());
        assert!(database.cancel("operation".to_owned()).is_err());
        drop(closing);
        database.database.close().unwrap();
        std::fs::remove_dir_all(directory).unwrap();
    }
}

#[napi]
pub struct EmbeddedDatabase {
    database: Arc<AdapterDatabaseOwner>,
    workers: Arc<napi::tokio::sync::Semaphore>,
}

#[napi]
impl EmbeddedDatabase {
    #[napi(factory)]
    pub async fn open(
        data_dir: String,
        device: Option<String>,
        device_ordinal: Option<u32>,
        load_embeddings: Option<bool>,
        budgets: Option<serde_json::Value>,
        embedding_device: Option<String>,
        embedding_device_ordinal: Option<u32>,
    ) -> Result<Self> {
        let ordinal = device_ordinal.unwrap_or(0);
        let execution_device = match device.as_deref().unwrap_or("auto") {
            "auto" => ExecutionDevice::Auto,
            "cpu" => ExecutionDevice::Cpu,
            _ => {
                return Err(Error::new(
                    Status::InvalidArg,
                    "graph execution is CPU-only; device must be auto or cpu",
                ));
            }
        };
        if ordinal != 0 {
            return Err(Error::new(
                Status::InvalidArg,
                "CPU graph device ordinal must be zero",
            ));
        }
        let embedding_ordinal = embedding_device_ordinal.unwrap_or(0);
        let embedding_device = match embedding_device.as_deref().unwrap_or("auto") {
            "auto" => EmbeddingDevice::Auto,
            "cpu" => EmbeddingDevice::Cpu,
            "metal" => EmbeddingDevice::Metal(embedding_ordinal),
            "cuda" => EmbeddingDevice::Cuda(embedding_ordinal),
            _ => {
                return Err(Error::new(
                    Status::InvalidArg,
                    "embedding_device must be auto, cpu, metal, or cuda",
                ));
            }
        };
        let mut options = EmbeddedOptions::new(data_dir)
            .with_execution_device(execution_device)
            .with_embedding_device(embedding_device)
            .with_embedding_policy(if load_embeddings.unwrap_or(true) {
                EmbeddingPolicy::Automatic
            } else {
                EmbeddingPolicy::Disabled
            });
        if let Some(budgets) = budgets {
            irongraph_embedded::configure_budgets(&mut options, budgets).map_err(node_error)?;
        }
        let admission = AdapterOwnerAdmission::reserve().map_err(node_error)?;
        let workers = Arc::new(napi::tokio::sync::Semaphore::new(
            options.worker_threads.saturating_mul(4),
        ));
        let database = napi::tokio::task::spawn_blocking(move || {
            RustEmbeddedDatabase::open(options)
                .map(|database| AdapterDatabaseOwner::new(database, admission))
                .map_err(node_error)
        })
        .await
        .map_err(node_error)??;
        Ok(Self {
            database: Arc::new(database),
            workers,
        })
    }

    #[napi(ts_return_type = "Promise<QueryResult>")]
    pub async fn query(
        &self,
        cypher: String,
        project_id: Option<String>,
        parameters: Option<serde_json::Value>,
        query_options: Option<serde_json::Value>,
        operation_options: Option<serde_json::Value>,
    ) -> Result<QueryOutput> {
        let permit = self
            .workers
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| node_error("embedded workers are closed"))?;
        let database = Arc::clone(&self.database);
        napi::tokio::task::spawn_blocking(move || {
            let _permit = permit;
            let mut query = query_request(cypher, project_id, parameters)?;
            if let Some(options) = query_options {
                irongraph_client::configure_query(&mut query, options).map_err(node_error)?;
            }
            let options: OperationOptions = operation_options
                .map(serde_json::from_value)
                .transpose()
                .map_err(node_error)?
                .unwrap_or_default();
            let guard = database
                .read()
                .map_err(|_| Error::new(Status::GenericFailure, "database lock is poisoned"))?;
            let result = guard
                .as_ref()
                .ok_or_else(|| Error::new(Status::GenericFailure, "database is closed"))?
                .query_with_options(query, options)
                .map_err(node_error)?;
            QueryOutput::prepare(result)
        })
        .await
        .map_err(node_error)?
    }

    #[napi]
    pub async fn snapshot(&self) -> Result<serde_json::Value> {
        let database = Arc::clone(&self.database);
        napi::tokio::task::spawn_blocking(move || {
            let guard = database
                .read()
                .map_err(|_| Error::new(Status::GenericFailure, "database lock is poisoned"))?;
            let bookmark = guard
                .as_ref()
                .ok_or_else(|| Error::new(Status::GenericFailure, "database is closed"))?
                .snapshot()
                .map_err(node_error)?;
            serde_json::to_value(bookmark).map_err(node_error)
        })
        .await
        .map_err(node_error)?
    }

    #[napi]
    pub async fn close(&self) -> Result<()> {
        let database = Arc::clone(&self.database);
        napi::tokio::task::spawn_blocking(move || database.close().map_err(node_error))
            .await
            .map_err(node_error)?
    }

    #[napi]
    pub async fn flush(&self) -> Result<()> {
        let database = Arc::clone(&self.database);
        napi::tokio::task::spawn_blocking(move || {
            let guard = database.read().map_err(node_error)?;
            guard
                .as_ref()
                .ok_or_else(|| node_error("database is closed"))?
                .flush()
                .map_err(node_error)
        })
        .await
        .map_err(node_error)?
    }

    #[napi]
    pub fn cancel(&self, operation_id: String) -> Result<bool> {
        let guard = self
            .database
            .try_read()
            .map_err(|_| node_error("database is closing"))?;
        Ok(guard
            .as_ref()
            .ok_or_else(|| node_error("database is closed"))?
            .cancel(&operation_id))
    }

    #[napi]
    pub fn status(&self) -> Result<serde_json::Value> {
        let guard = self
            .database
            .try_read()
            .map_err(|_| node_error("database is closing"))?;
        serde_json::to_value(
            guard
                .as_ref()
                .ok_or_else(|| node_error("database is closed"))?
                .status()
                .map_err(node_error)?,
        )
        .map_err(node_error)
    }

    #[napi]
    pub async fn stream_append(
        &self,
        request: serde_json::Value,
        options: Option<serde_json::Value>,
    ) -> Result<serde_json::Value> {
        let permit = self
            .workers
            .clone()
            .acquire_owned()
            .await
            .map_err(|_| node_error("embedded workers are closed"))?;
        let database = Arc::clone(&self.database);
        napi::tokio::task::spawn_blocking(move || {
            let _permit = permit;
            let request: StreamAppend = serde_json::from_value(request).map_err(node_error)?;
            let options: OperationOptions = options
                .map(serde_json::from_value)
                .transpose()
                .map_err(node_error)?
                .unwrap_or_default();
            let guard = database.read().map_err(node_error)?;
            serde_json::to_value(
                guard
                    .as_ref()
                    .ok_or_else(|| node_error("database is closed"))?
                    .stream_append(request, options)
                    .map_err(node_error)?,
            )
            .map_err(node_error)
        })
        .await
        .map_err(node_error)?
    }

    #[napi]
    pub async fn stream_fetch(
        &self,
        request: serde_json::Value,
        options: Option<serde_json::Value>,
    ) -> Result<serde_json::Value> {
        let request: StreamFetch = serde_json::from_value(request).map_err(node_error)?;
        let options: OperationOptions = options
            .map(serde_json::from_value)
            .transpose()
            .map_err(node_error)?
            .unwrap_or_default();
        let database = Arc::clone(&self.database);
        napi::tokio::task::spawn_blocking(move || {
            let guard = database.read().map_err(node_error)?;
            serde_json::to_value(
                guard
                    .as_ref()
                    .ok_or_else(|| node_error("database is closed"))?
                    .stream_fetch(request, options)
                    .map_err(node_error)?,
            )
            .map_err(node_error)
        })
        .await
        .map_err(node_error)?
    }
}

#[napi]
pub struct Client {
    client: Arc<RustRemoteClient>,
}

#[napi]
impl Client {
    #[napi(factory)]
    pub fn api(base_url: String) -> Result<Self> {
        Ok(Self {
            client: Arc::new(RustRemoteClient::api(&base_url).map_err(node_error)?),
        })
    }

    #[napi(factory)]
    pub fn bolt(uri: String) -> Result<Self> {
        Ok(Self {
            client: Arc::new(RustRemoteClient::bolt(&uri).map_err(node_error)?),
        })
    }

    #[napi(factory)]
    pub fn api_mtls(
        base_url: String,
        certificate: String,
        private_key: String,
        certificate_authority: String,
    ) -> Result<Self> {
        let tls = MutualTls::new(certificate, private_key, certificate_authority);
        Ok(Self {
            client: Arc::new(RustRemoteClient::api_mtls(&base_url, &tls).map_err(node_error)?),
        })
    }

    #[napi(factory)]
    pub fn bolt_mtls(
        uri: String,
        certificate: String,
        private_key: String,
        certificate_authority: String,
    ) -> Result<Self> {
        let tls = MutualTls::new(certificate, private_key, certificate_authority);
        Ok(Self {
            client: Arc::new(RustRemoteClient::bolt_mtls(&uri, &tls).map_err(node_error)?),
        })
    }

    #[napi(ts_return_type = "Promise<QueryResult>")]
    pub async fn query(
        &self,
        cypher: String,
        project_id: Option<String>,
        parameters: Option<serde_json::Value>,
        query_options: Option<serde_json::Value>,
    ) -> Result<QueryOutput> {
        let mut query = query_request(cypher, project_id, parameters)?;
        if let Some(options) = query_options {
            irongraph_client::configure_query(&mut query, options).map_err(node_error)?;
        }
        let client = Arc::clone(&self.client);
        napi::tokio::task::spawn_blocking(move || {
            let result = client.query(query).map_err(node_error)?;
            QueryOutput::prepare(result)
        })
        .await
        .map_err(node_error)?
    }
}
