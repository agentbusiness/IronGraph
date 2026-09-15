use std::sync::{Arc, Mutex, RwLock};

use irongraph_client::{MutualTls, Query, RemoteClient as RustRemoteClient};
use irongraph_embedded::{
    EmbeddedDatabase as RustEmbeddedDatabase, EmbeddedOptions, EmbeddingPolicy, ExecutionDevice,
    OperationOptions, StreamAppend, StreamFetch,
};
use irongraph_types::ProjectId;
use napi::{Error, Result, Status};
use napi_derive::napi;
use uuid::Uuid;

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

#[napi]
pub struct EmbeddedDatabase {
    database: Arc<RwLock<Option<RustEmbeddedDatabase>>>,
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
    ) -> Result<Self> {
        let ordinal = device_ordinal.unwrap_or(0);
        let execution_device = match device.as_deref().unwrap_or("auto") {
            "auto" => ExecutionDevice::Auto,
            "cpu" => ExecutionDevice::Cpu,
            "metal" => ExecutionDevice::Metal(ordinal),
            "cuda" => ExecutionDevice::Cuda(ordinal),
            _ => {
                return Err(Error::new(
                    Status::InvalidArg,
                    "device must be auto, cpu, metal, or cuda",
                ));
            }
        };
        let mut options = EmbeddedOptions::new(data_dir)
            .with_execution_device(execution_device)
            .with_embedding_policy(if load_embeddings.unwrap_or(true) {
                EmbeddingPolicy::Automatic
            } else {
                EmbeddingPolicy::Disabled
            });
        if let Some(budgets) = budgets {
            irongraph_embedded::configure_budgets(&mut options, budgets).map_err(node_error)?;
        }
        let database = napi::tokio::task::spawn_blocking(move || {
            RustEmbeddedDatabase::open(options).map_err(node_error)
        })
        .await
        .map_err(node_error)??;
        Ok(Self {
            database: Arc::new(RwLock::new(Some(database))),
        })
    }

    #[napi]
    pub async fn query(
        &self,
        cypher: String,
        project_id: Option<String>,
        parameters: Option<serde_json::Value>,
        query_options: Option<serde_json::Value>,
        operation_options: Option<serde_json::Value>,
    ) -> Result<serde_json::Value> {
        let mut query = query_request(cypher, project_id, parameters)?;
        if let Some(options) = query_options {
            irongraph_client::configure_query(&mut query, options).map_err(node_error)?;
        }
        let options: OperationOptions = operation_options
            .map(serde_json::from_value)
            .transpose()
            .map_err(node_error)?
            .unwrap_or_default();
        let database = Arc::clone(&self.database);
        napi::tokio::task::spawn_blocking(move || {
            let guard = database
                .read()
                .map_err(|_| Error::new(Status::GenericFailure, "database lock is poisoned"))?;
            let result = guard
                .as_ref()
                .ok_or_else(|| Error::new(Status::GenericFailure, "database is closed"))?
                .query_with_options(query, options)
                .map_err(node_error)?;
            serde_json::to_value(result).map_err(node_error)
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
        napi::tokio::task::spawn_blocking(move || {
            let database = database
                .write()
                .map_err(|_| Error::new(Status::GenericFailure, "database lock is poisoned"))?
                .take();
            if let Some(database) = database {
                database.close().map_err(node_error)?;
            }
            Ok(())
        })
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
        let guard = self.database.read().map_err(node_error)?;
        Ok(guard
            .as_ref()
            .ok_or_else(|| node_error("database is closed"))?
            .cancel(&operation_id))
    }

    #[napi]
    pub fn status(&self) -> Result<serde_json::Value> {
        let guard = self.database.read().map_err(node_error)?;
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
        let request: StreamAppend = serde_json::from_value(request).map_err(node_error)?;
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
    client: Arc<Mutex<RustRemoteClient>>,
}

#[napi]
impl Client {
    #[napi(factory)]
    pub fn api(base_url: String) -> Result<Self> {
        Ok(Self {
            client: Arc::new(Mutex::new(
                RustRemoteClient::api(&base_url).map_err(node_error)?,
            )),
        })
    }

    #[napi(factory)]
    pub fn bolt(uri: String) -> Result<Self> {
        Ok(Self {
            client: Arc::new(Mutex::new(
                RustRemoteClient::bolt(&uri).map_err(node_error)?,
            )),
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
            client: Arc::new(Mutex::new(
                RustRemoteClient::api_mtls(&base_url, &tls).map_err(node_error)?,
            )),
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
            client: Arc::new(Mutex::new(
                RustRemoteClient::bolt_mtls(&uri, &tls).map_err(node_error)?,
            )),
        })
    }

    #[napi]
    pub async fn query(
        &self,
        cypher: String,
        project_id: Option<String>,
        parameters: Option<serde_json::Value>,
        query_options: Option<serde_json::Value>,
    ) -> Result<serde_json::Value> {
        let mut query = query_request(cypher, project_id, parameters)?;
        if let Some(options) = query_options {
            irongraph_client::configure_query(&mut query, options).map_err(node_error)?;
        }
        let client = Arc::clone(&self.client);
        napi::tokio::task::spawn_blocking(move || {
            let result = client
                .lock()
                .map_err(|_| Error::new(Status::GenericFailure, "client lock is poisoned"))?
                .query(query)
                .map_err(node_error)?;
            serde_json::to_value(result).map_err(node_error)
        })
        .await
        .map_err(node_error)?
    }
}
