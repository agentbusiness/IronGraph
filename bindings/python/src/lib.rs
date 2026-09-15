use std::{
    path::PathBuf,
    sync::{Mutex, RwLock},
};

use irongraph_client::{MutualTls, Query, RemoteClient};
use irongraph_embedded::{
    EmbeddedDatabase as RustEmbeddedDatabase, EmbeddedOptions, EmbeddingPolicy, ExecutionDevice,
    OperationOptions, StreamAppend, StreamFetch,
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
    database: RwLock<Option<RustEmbeddedDatabase>>,
}

#[pymethods]
impl EmbeddedDatabase {
    #[new]
    #[pyo3(signature = (data_dir, *, device="auto", device_ordinal=0, load_embeddings=true, budgets=None))]
    fn new(
        py: Python<'_>,
        data_dir: PathBuf,
        device: &str,
        device_ordinal: u32,
        load_embeddings: bool,
        budgets: Option<Bound<'_, PyAny>>,
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
        let mut options = EmbeddedOptions::new(data_dir)
            .with_execution_device(execution_device)
            .with_embedding_policy(if load_embeddings {
                EmbeddingPolicy::Automatic
            } else {
                EmbeddingPolicy::Disabled
            });
        if let Some(budgets) = budgets {
            irongraph_embedded::configure_budgets(
                &mut options,
                pythonize::depythonize(&budgets).map_err(python_error)?,
            )
            .map_err(python_error)?;
        }
        Ok(Self {
            database: RwLock::new(Some(
                py.detach(move || RustEmbeddedDatabase::open(options))
                    .map_err(python_error)?,
            )),
        })
    }

    #[pyo3(signature = (cypher, *, project_id=None, parameters=None, query_options=None, operation_options=None))]
    fn query<'py>(
        &self,
        py: Python<'py>,
        cypher: String,
        project_id: Option<String>,
        parameters: Option<Bound<'py, PyAny>>,
        query_options: Option<Bound<'py, PyAny>>,
        operation_options: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut query = query_request(cypher, project_id, parameters)?;
        if let Some(options) = query_options {
            irongraph_client::configure_query(
                &mut query,
                pythonize::depythonize(&options).map_err(python_error)?,
            )
            .map_err(python_error)?;
        }
        let options: OperationOptions = operation_options
            .as_ref()
            .map(pythonize::depythonize)
            .transpose()
            .map_err(python_error)?
            .unwrap_or_default();
        let result = py.detach(|| {
            let guard = self
                .database
                .read()
                .map_err(|_| PyRuntimeError::new_err("embedded database lock is poisoned"))?;
            guard
                .as_ref()
                .ok_or_else(|| PyRuntimeError::new_err("embedded database is closed"))?
                .query_with_options(query, options)
                .map_err(python_error)
        })?;
        pythonize::pythonize(py, &result).map_err(python_error)
    }

    fn snapshot<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let bookmark = py.detach(|| {
            let guard = self
                .database
                .read()
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
                .write()
                .map_err(|_| PyRuntimeError::new_err("embedded database lock is poisoned"))?
                .take();
            if let Some(database) = database {
                database.close().map_err(python_error)?;
            }
            Ok(())
        })
    }

    fn flush(&self, py: Python<'_>) -> PyResult<()> {
        py.detach(|| {
            let guard = self.database.read().map_err(python_error)?;
            guard
                .as_ref()
                .ok_or_else(|| python_error("database is closed"))?
                .flush()
                .map_err(python_error)
        })
    }

    fn status<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let guard = self.database.read().map_err(python_error)?;
        let status = guard
            .as_ref()
            .ok_or_else(|| python_error("database is closed"))?
            .status()
            .map_err(python_error)?;
        pythonize::pythonize(py, &status).map_err(python_error)
    }

    fn cancel(&self, operation_id: &str) -> PyResult<bool> {
        let guard = self.database.read().map_err(python_error)?;
        Ok(guard
            .as_ref()
            .ok_or_else(|| python_error("database is closed"))?
            .cancel(operation_id))
    }

    #[pyo3(signature = (request, *, options=None))]
    fn stream_append<'py>(
        &self,
        py: Python<'py>,
        request: Bound<'py, PyAny>,
        options: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: StreamAppend = pythonize::depythonize(&request).map_err(python_error)?;
        let options: OperationOptions = options
            .as_ref()
            .map(pythonize::depythonize)
            .transpose()
            .map_err(python_error)?
            .unwrap_or_default();
        let result = py.detach(|| {
            let guard = self.database.read().map_err(python_error)?;
            guard
                .as_ref()
                .ok_or_else(|| python_error("database is closed"))?
                .stream_append(request, options)
                .map_err(python_error)
        })?;
        pythonize::pythonize(py, &result).map_err(python_error)
    }

    #[pyo3(signature = (request, *, options=None))]
    fn stream_fetch<'py>(
        &self,
        py: Python<'py>,
        request: Bound<'py, PyAny>,
        options: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let request: StreamFetch = pythonize::depythonize(&request).map_err(python_error)?;
        let options: OperationOptions = options
            .as_ref()
            .map(pythonize::depythonize)
            .transpose()
            .map_err(python_error)?
            .unwrap_or_default();
        let result = py.detach(|| {
            let guard = self.database.read().map_err(python_error)?;
            guard
                .as_ref()
                .ok_or_else(|| python_error("database is closed"))?
                .stream_fetch(request, options)
                .map_err(python_error)
        })?;
        pythonize::pythonize(py, &result).map_err(python_error)
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

    #[pyo3(signature = (cypher, *, project_id=None, parameters=None, query_options=None))]
    fn query<'py>(
        &self,
        py: Python<'py>,
        cypher: String,
        project_id: Option<String>,
        parameters: Option<Bound<'py, PyAny>>,
        query_options: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mut query = query_request(cypher, project_id, parameters)?;
        if let Some(options) = query_options {
            irongraph_client::configure_query(
                &mut query,
                pythonize::depythonize(&options).map_err(python_error)?,
            )
            .map_err(python_error)?;
        }
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
