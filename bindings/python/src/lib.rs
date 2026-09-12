use std::{path::PathBuf, sync::Mutex};

use irongraph_client::{MutualTls, Query, RemoteClient};
use irongraph_embedded::{
    EmbeddedDatabase as RustEmbeddedDatabase, EmbeddedOptions, EmbeddingPolicy, ExecutionDevice,
};
use irongraph_types::ProjectId;
use pyo3::{exceptions::PyRuntimeError, prelude::*, types::PyModule};
use uuid::Uuid;

fn python_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

fn query_request(
    cypher: String,
    project_id: Option<String>,
    parameters: Option<Bound<'_, PyAny>>,
) -> PyResult<Query> {
    let mut query = Query::new(cypher);
    if let Some(project_id) = project_id {
        query.project_id = Some(ProjectId(
            Uuid::parse_str(&project_id).map_err(python_error)?,
        ));
    }
    if let Some(parameters) = parameters {
        query.parameters = pythonize::depythonize(&parameters).map_err(python_error)?;
    }
    Ok(query)
}

#[pyclass(name = "EmbeddedDatabase")]
struct EmbeddedDatabase {
    database: Mutex<Option<RustEmbeddedDatabase>>,
}

#[pymethods]
impl EmbeddedDatabase {
    #[new]
    #[pyo3(signature = (data_dir, *, device="auto", device_ordinal=0, load_embeddings=true))]
    fn new(
        py: Python<'_>,
        data_dir: PathBuf,
        device: &str,
        device_ordinal: u32,
        load_embeddings: bool,
    ) -> PyResult<Self> {
        let execution_device = match device {
            "auto" => ExecutionDevice::Auto,
            "cpu" => ExecutionDevice::Cpu,
            "metal" => ExecutionDevice::Metal(device_ordinal),
            "cuda" => ExecutionDevice::Cuda(device_ordinal),
            _ => {
                return Err(PyRuntimeError::new_err(
                    "device must be auto, cpu, metal, or cuda",
                ));
            }
        };
        let options = EmbeddedOptions::new(data_dir)
            .with_execution_device(execution_device)
            .with_embedding_policy(if load_embeddings {
                EmbeddingPolicy::Automatic
            } else {
                EmbeddingPolicy::Disabled
            });
        Ok(Self {
            database: Mutex::new(Some(
                py.detach(move || RustEmbeddedDatabase::open(options))
                    .map_err(python_error)?,
            )),
        })
    }

    #[pyo3(signature = (cypher, *, project_id=None, parameters=None))]
    fn query<'py>(
        &self,
        py: Python<'py>,
        cypher: String,
        project_id: Option<String>,
        parameters: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let query = query_request(cypher, project_id, parameters)?;
        let result = py.detach(|| {
            let guard = self
                .database
                .lock()
                .map_err(|_| PyRuntimeError::new_err("embedded database lock is poisoned"))?;
            guard
                .as_ref()
                .ok_or_else(|| PyRuntimeError::new_err("embedded database is closed"))?
                .query(query)
                .map_err(python_error)
        })?;
        pythonize::pythonize(py, &result).map_err(python_error)
    }

    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let bookmark = py.detach(|| {
            let guard = self
                .database
                .lock()
                .map_err(|_| PyRuntimeError::new_err("embedded database lock is poisoned"))?;
            guard
                .as_ref()
                .ok_or_else(|| PyRuntimeError::new_err("embedded database is closed"))?
                .snapshot()
                .map_err(python_error)
        })?;
        pythonize::pythonize(py, &bookmark).map_err(python_error)
    }

    fn close(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            let database = self
                .database
                .lock()
                .map_err(|_| PyRuntimeError::new_err("embedded database lock is poisoned"))?
                .take();
            if let Some(database) = database {
                database.close().map_err(python_error)?;
            }
            Ok(())
        })
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __exit__(
        &self,
        py: Python<'_>,
        _exception_type: Option<Bound<'_, PyAny>>,
        _exception: Option<Bound<'_, PyAny>>,
        _traceback: Option<Bound<'_, PyAny>>,
    ) -> PyResult<bool> {
        self.close(py)?;
        Ok(false)
    }
}

#[pyclass(name = "Client")]
struct Client {
    client: Mutex<RemoteClient>,
}

#[pymethods]
impl Client {
    #[staticmethod]
    fn api(base_url: String) -> PyResult<Self> {
        Ok(Self {
            client: Mutex::new(RemoteClient::api(&base_url).map_err(python_error)?),
        })
    }

    #[staticmethod]
    fn bolt(uri: String) -> PyResult<Self> {
        Ok(Self {
            client: Mutex::new(RemoteClient::bolt(&uri).map_err(python_error)?),
        })
    }

    #[staticmethod]
    fn api_mtls(
        base_url: String,
        certificate: PathBuf,
        private_key: PathBuf,
        certificate_authority: PathBuf,
    ) -> PyResult<Self> {
        let tls = MutualTls::new(certificate, private_key, certificate_authority);
        Ok(Self {
            client: Mutex::new(RemoteClient::api_mtls(&base_url, &tls).map_err(python_error)?),
        })
    }

    #[staticmethod]
    fn bolt_mtls(
        uri: String,
        certificate: PathBuf,
        private_key: PathBuf,
        certificate_authority: PathBuf,
    ) -> PyResult<Self> {
        let tls = MutualTls::new(certificate, private_key, certificate_authority);
        Ok(Self {
            client: Mutex::new(RemoteClient::bolt_mtls(&uri, &tls).map_err(python_error)?),
        })
    }

    #[pyo3(signature = (cypher, *, project_id=None, parameters=None))]
    fn query<'py>(
        &self,
        py: Python<'py>,
        cypher: String,
        project_id: Option<String>,
        parameters: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let query = query_request(cypher, project_id, parameters)?;
        let result = py.detach(|| {
            self.client
                .lock()
                .map_err(|_| PyRuntimeError::new_err("remote client lock is poisoned"))?
                .query(query)
                .map_err(python_error)
        })?;
        pythonize::pythonize(py, &result).map_err(python_error)
    }
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<EmbeddedDatabase>()?;
    module.add_class::<Client>()?;
    Ok(())
}
