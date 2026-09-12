use std::sync::{Arc, Mutex};

use irongraph_client::{MutualTls, Query, RemoteClient as RustRemoteClient};
use irongraph_embedded::{
    EmbeddedDatabase as RustEmbeddedDatabase, EmbeddedOptions, EmbeddingPolicy, ExecutionDevice,
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
    database: Arc<Mutex<Option<RustEmbeddedDatabase>>>,
}

#[napi]
impl EmbeddedDatabase {
    #[napi(factory)]
    pub async fn open(
        data_dir: String,
        device: Option<String>,
        device_ordinal: Option<u32>,
        load_embeddings: Option<bool>,
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
        let options = EmbeddedOptions::new(data_dir)
            .with_execution_device(execution_device)
            .with_embedding_policy(if load_embeddings.unwrap_or(true) {
                EmbeddingPolicy::Automatic
            } else {
                EmbeddingPolicy::Disabled
            });
        let database = napi::tokio::task::spawn_blocking(move || {
            RustEmbeddedDatabase::open(options).map_err(node_error)
        })
        .await
        .map_err(node_error)??;
        Ok(Self {
            database: Arc::new(Mutex::new(Some(database))),
        })
    }

    #[napi]
    pub async fn query(
        &self,
        cypher: String,
        project_id: Option<String>,
        parameters: Option<serde_json::Value>,
    ) -> Result<serde_json::Value> {
        let query = query_request(cypher, project_id, parameters)?;
        let database = Arc::clone(&self.database);
        napi::tokio::task::spawn_blocking(move || {
            let guard = database
                .lock()
                .map_err(|_| Error::new(Status::GenericFailure, "database lock is poisoned"))?;
            let result = guard
                .as_ref()
                .ok_or_else(|| Error::new(Status::GenericFailure, "database is closed"))?
                .query(query)
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
                .lock()
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
                .lock()
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
    ) -> Result<serde_json::Value> {
        let query = query_request(cypher, project_id, parameters)?;
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
