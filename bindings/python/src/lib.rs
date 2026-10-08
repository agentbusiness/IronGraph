use std::path::PathBuf;

use irongraph_client::{MutualTls, Query, QueryResult, RemoteClient};
use irongraph_embedded::{
    AdapterDatabaseOwner, AdapterOwnerAdmission, EmbeddedDatabase as RustEmbeddedDatabase,
    EmbeddedOptions, EmbeddingDevice, EmbeddingPolicy, ExecutionDevice, OperationOptions,
    StreamAppend, StreamFetch,
};
use irongraph_server::protocol::{RelationshipValue, ResultNode, TypedValue};
use irongraph_types::ProjectId;
use pyo3::{
    exceptions::PyRuntimeError,
    prelude::*,
    types::{PyDict, PyList, PyModule, PyTuple, PyType},
};
use uuid::Uuid;

fn python_error(error: impl std::fmt::Display) -> PyErr {
    PyRuntimeError::new_err(error.to_string())
}

fn python_properties<'py>(
    py: Python<'py>,
    values: &std::collections::BTreeMap<String, TypedValue>,
    work: &mut usize,
) -> PyResult<Bound<'py, PyDict>> {
    let answer = PyDict::new(py);
    for (key, value) in values {
        *work = work.saturating_add(key.len().saturating_mul(4));
        answer.set_item(key, python_value(py, value, work)?)?;
    }
    Ok(answer)
}

fn python_node<'py>(
    py: Python<'py>,
    node: &ResultNode,
    work: &mut usize,
) -> PyResult<Bound<'py, PyDict>> {
    *work = work.saturating_add(node.id.len().saturating_mul(4));
    for label in &node.labels {
        *work = work.saturating_add(label.len().saturating_mul(4));
    }
    let answer = PyDict::new(py);
    answer.set_item(pyo3::intern!(py, "id"), &node.id)?;
    answer.set_item(pyo3::intern!(py, "labels"), &node.labels)?;
    answer.set_item(
        pyo3::intern!(py, "properties"),
        python_properties(py, &node.properties, work)?,
    )?;
    Ok(answer)
}

fn python_relationship<'py>(
    py: Python<'py>,
    relationship: &RelationshipValue,
    work: &mut usize,
) -> PyResult<Bound<'py, PyDict>> {
    let answer = PyDict::new(py);
    for text in [
        &relationship.id,
        &relationship.source,
        &relationship.target,
        &relationship.relationship_type,
    ] {
        *work = work.saturating_add(text.len().saturating_mul(4));
    }
    answer.set_item(pyo3::intern!(py, "id"), &relationship.id)?;
    answer.set_item(pyo3::intern!(py, "source"), &relationship.source)?;
    answer.set_item(pyo3::intern!(py, "target"), &relationship.target)?;
    answer.set_item(
        pyo3::intern!(py, "relationship_type"),
        &relationship.relationship_type,
    )?;
    answer.set_item(
        pyo3::intern!(py, "properties"),
        python_properties(py, &relationship.properties, work)?,
    )?;
    Ok(answer)
}

fn python_value<'py>(
    py: Python<'py>,
    value: &TypedValue,
    work: &mut usize,
) -> PyResult<Bound<'py, PyAny>> {
    // Account for both container construction and payload work without another traversal.
    let payload = match value {
        // Python may use four bytes per Unicode scalar; the UTF-8 length is a conservative
        // scalar-count bound, including for large strings containing a single non-ASCII mark.
        TypedValue::Integer(text) | TypedValue::String(text) => text.len().saturating_mul(4),
        TypedValue::Bytes(bytes) => bytes.len().saturating_mul(16),
        TypedValue::Vector(numbers) => numbers.len().saturating_mul(8),
        TypedValue::DateTime {
            timezone: Some(text),
            ..
        } => text.len().saturating_mul(4),
        _ => 0,
    };
    *work = work.saturating_add(128).saturating_add(payload);
    let answer = PyDict::new(py);
    macro_rules! tagged {
        ($kind:literal, $payload:expr) => {{
            answer.set_item(pyo3::intern!(py, "type"), pyo3::intern!(py, $kind))?;
            answer.set_item(pyo3::intern!(py, "value"), $payload)?;
        }};
    }
    match value {
        TypedValue::Null => {
            answer.set_item(pyo3::intern!(py, "type"), pyo3::intern!(py, "null"))?;
        }
        TypedValue::Boolean(value) => tagged!("boolean", *value),
        TypedValue::Integer(value) => tagged!("integer", value),
        TypedValue::Float(value) => tagged!("float", *value),
        TypedValue::String(value) => tagged!("string", value),
        TypedValue::Bytes(value) => tagged!("bytes", PyList::new(py, value)?),
        TypedValue::Date(value) => tagged!("date", *value),
        TypedValue::Vector(value) => tagged!(
            "vector",
            PyList::new(py, value.iter().map(|number| f64::from(*number)))?
        ),
        TypedValue::Node(value) => tagged!("node", python_node(py, value, work)?),
        TypedValue::Relationship(value) => {
            tagged!("relationship", python_relationship(py, value, work)?)
        }
        TypedValue::List(value) => {
            let items = value
                .iter()
                .map(|item| python_value(py, item, work))
                .collect::<PyResult<Vec<_>>>()?;
            tagged!("list", PyList::new(py, items)?);
        }
        TypedValue::Map(value) => tagged!("map", python_properties(py, value, work)?),
        TypedValue::Path(value) => {
            let path = PyDict::new(py);
            let nodes = value
                .nodes
                .iter()
                .map(|node| python_node(py, node, work))
                .collect::<PyResult<Vec<_>>>()?;
            let relationships = value
                .relationships
                .iter()
                .map(|relationship| python_relationship(py, relationship, work))
                .collect::<PyResult<Vec<_>>>()?;
            path.set_item(pyo3::intern!(py, "nodes"), PyList::new(py, nodes)?)?;
            path.set_item(
                pyo3::intern!(py, "relationships"),
                PyList::new(py, relationships)?,
            )?;
            tagged!("path", path);
        }
        TypedValue::Time { .. } | TypedValue::DateTime { .. } | TypedValue::Duration { .. } => {
            return pythonize::pythonize(py, value).map_err(python_error);
        }
    }
    Ok(answer.into_any())
}

fn python_query_result(py: Python<'_>, result: QueryResult) -> PyResult<Bound<'_, PyAny>> {
    if result.rows.len() <= 16 {
        return pythonize::pythonize(py, &result).map_err(python_error);
    }
    let QueryResult {
        catalog,
        columns,
        rows,
        summary,
    } = result;
    let answer = PyDict::new(py);
    answer.set_item(
        "catalog",
        pythonize::pythonize(py, &catalog).map_err(python_error)?,
    )?;
    answer.set_item(
        "columns",
        pythonize::pythonize(py, &columns).map_err(python_error)?,
    )?;
    let mut converted = Vec::with_capacity(rows.len());
    let mut consumed = Vec::with_capacity(64);
    answer.set_item("rows", py.None())?;
    answer.set_item(
        "summary",
        pythonize::pythonize(py, &summary).map_err(python_error)?,
    )?;
    let mut work = 0;
    for row in rows {
        let values = row
            .iter()
            .map(|value| python_value(py, value, &mut work))
            .collect::<PyResult<Vec<_>>>()?;
        converted.push(PyList::new(py, values)?);
        consumed.push(row);
        if work >= 4 * 1024 * 1024 {
            py.detach(|| consumed.clear());
            py.check_signals()?;
            work = 0;
        }
    }
    py.detach(|| consumed.clear());
    answer.set_item("rows", PyList::new(py, converted)?)?;
    Ok(answer.into_any())
}

fn async_call<'py>(
    py: Python<'py>,
    owner: Bound<'py, PyAny>,
    method: &str,
    args: Bound<'py, PyTuple>,
    kwargs: Bound<'py, PyDict>,
    operation_key: Option<&str>,
    bounded: bool,
) -> PyResult<Bound<'py, PyAny>> {
    PyModule::import(py, "irongraph._async")?
        .getattr("invoke")?
        .call1((
            owner.getattr(method)?,
            args,
            kwargs,
            operation_key,
            owner,
            bounded,
        ))
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
    database: AdapterDatabaseOwner,
}

#[pymethods]
impl EmbeddedDatabase {
    #[classmethod]
    #[pyo3(signature = (data_dir, **options))]
    fn open_async<'py>(
        cls: Bound<'py, PyType>,
        py: Python<'py>,
        data_dir: Bound<'py, PyAny>,
        options: Option<Bound<'py, PyDict>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        PyModule::import(py, "irongraph._async")?
            .getattr("invoke")?
            .call1((
                cls,
                (data_dir,),
                options.unwrap_or_else(|| PyDict::new(py)),
                py.None(),
                py.None(),
                true,
                true,
            ))
    }

    #[pyo3(signature = (cypher, *, project_id=None, parameters=None, query_options=None, operation_options=None))]
    fn query_async<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        cypher: String,
        project_id: Option<String>,
        parameters: Option<Bound<'py, PyAny>>,
        query_options: Option<Bound<'py, PyAny>>,
        operation_options: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("project_id", project_id)?;
        kwargs.set_item("parameters", parameters)?;
        kwargs.set_item("query_options", query_options)?;
        kwargs.set_item("operation_options", operation_options)?;
        async_call(
            py,
            slf.into_pyobject(py)?.into_any(),
            "query",
            PyTuple::new(py, [cypher])?,
            kwargs,
            Some("operation_options"),
            true,
        )
    }
    #[pyo3(signature = (request, *, options=None))]
    fn stream_append_async<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        request: Bound<'py, PyAny>,
        options: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("options", options)?;
        async_call(
            py,
            slf.into_pyobject(py)?.into_any(),
            "stream_append",
            PyTuple::new(py, [request])?,
            kwargs,
            Some("options"),
            true,
        )
    }
    #[pyo3(signature = (request, *, options=None))]
    fn stream_fetch_async<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        request: Bound<'py, PyAny>,
        options: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        let kwargs = PyDict::new(py);
        kwargs.set_item("options", options)?;
        async_call(
            py,
            slf.into_pyobject(py)?.into_any(),
            "stream_fetch",
            PyTuple::new(py, [request])?,
            kwargs,
            Some("options"),
            true,
        )
    }
    fn flush_async<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        async_call(
            py,
            slf.into_pyobject(py)?.into_any(),
            "flush",
            PyTuple::empty(py),
            PyDict::new(py),
            None,
            true,
        )
    }
    fn snapshot_async<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        async_call(
            py,
            slf.into_pyobject(py)?.into_any(),
            "snapshot",
            PyTuple::empty(py),
            PyDict::new(py),
            None,
            true,
        )
    }
    fn status_async<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        async_call(
            py,
            slf.into_pyobject(py)?.into_any(),
            "status",
            PyTuple::empty(py),
            PyDict::new(py),
            None,
            false,
        )
    }
    fn cancel_async<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        operation_id: String,
    ) -> PyResult<Bound<'py, PyAny>> {
        async_call(
            py,
            slf.into_pyobject(py)?.into_any(),
            "cancel",
            PyTuple::new(py, [operation_id])?,
            PyDict::new(py),
            None,
            false,
        )
    }
    fn close_async<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        async_call(
            py,
            slf.into_pyobject(py)?.into_any(),
            "close",
            PyTuple::empty(py),
            PyDict::new(py),
            None,
            false,
        )
    }
    fn __aenter__<'py>(slf: PyRef<'py, Self>, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        PyModule::import(py, "irongraph._async")?
            .getattr("enter")?
            .call1((slf.into_pyobject(py)?,))
    }
    fn __aexit__<'py>(
        slf: PyRef<'py, Self>,
        py: Python<'py>,
        _exception_type: Option<Bound<'py, PyAny>>,
        _exception: Option<Bound<'py, PyAny>>,
        _traceback: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyAny>> {
        PyModule::import(py, "irongraph._async")?
            .getattr("exit")?
            .call1((slf.into_pyobject(py)?,))
    }
    #[new]
    #[pyo3(signature = (data_dir, *, device="auto", device_ordinal=0, load_embeddings=true, budgets=None, embedding_device="auto", embedding_device_ordinal=0))]
    // Python exposes these as independent keyword options rather than a positional Rust interface.
    #[allow(clippy::too_many_arguments)]
    fn new(
        py: Python<'_>,
        data_dir: PathBuf,
        device: &str,
        device_ordinal: u32,
        load_embeddings: bool,
        budgets: Option<Bound<'_, PyAny>>,
        embedding_device: &str,
        embedding_device_ordinal: u32,
    ) -> PyResult<Self> {
        let execution_device = match device {
            "auto" => ExecutionDevice::Auto,
            "cpu" => ExecutionDevice::Cpu,
            _ => {
                return Err(PyRuntimeError::new_err(
                    "graph execution is CPU-only; device must be auto or cpu",
                ));
            }
        };
        if device_ordinal != 0 {
            return Err(PyRuntimeError::new_err(
                "CPU graph device ordinal must be zero",
            ));
        }
        let embedding_device = match embedding_device {
            "auto" => EmbeddingDevice::Auto,
            "cpu" => EmbeddingDevice::Cpu,
            "metal" => EmbeddingDevice::Metal(embedding_device_ordinal),
            "cuda" => EmbeddingDevice::Cuda(embedding_device_ordinal),
            _ => {
                return Err(PyRuntimeError::new_err(
                    "embedding_device must be auto, cpu, metal, or cuda",
                ));
            }
        };
        let mut options = EmbeddedOptions::new(data_dir)
            .with_execution_device(execution_device)
            .with_embedding_device(embedding_device)
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
        let admission = AdapterOwnerAdmission::reserve().map_err(python_error)?;
        Ok(Self {
            database: py
                .detach(move || {
                    RustEmbeddedDatabase::open(options)
                        .map(|database| AdapterDatabaseOwner::new(database, admission))
                })
                .map_err(python_error)?,
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
        python_query_result(py, result)
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
        py.detach(|| self.database.close().map_err(python_error))
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
        let status = py.detach(|| {
            let guard = self.database.read().map_err(python_error)?;
            guard
                .as_ref()
                .ok_or_else(|| python_error("database is closed"))?
                .status()
                .map_err(python_error)
        })?;
        pythonize::pythonize(py, &status).map_err(python_error)
    }

    fn cancel(&self, py: Python<'_>, operation_id: &str) -> PyResult<bool> {
        py.detach(|| {
            let guard = self.database.read().map_err(python_error)?;
            Ok(guard
                .as_ref()
                .ok_or_else(|| python_error("database is closed"))?
                .cancel(operation_id))
        })
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
    client: RemoteClient,
}

#[pymethods]
impl Client {
    #[staticmethod]
    fn api(base_url: String) -> PyResult<Self> {
        Ok(Self {
            client: RemoteClient::api(&base_url).map_err(python_error)?,
        })
    }

    #[staticmethod]
    fn bolt(uri: String) -> PyResult<Self> {
        Ok(Self {
            client: RemoteClient::bolt(&uri).map_err(python_error)?,
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
            client: RemoteClient::api_mtls(&base_url, &tls).map_err(python_error)?,
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
            client: RemoteClient::bolt_mtls(&uri, &tls).map_err(python_error)?,
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
        let result = py.detach(|| self.client.query(query).map_err(python_error))?;
        python_query_result(py, result)
    }
}

#[pymodule]
fn _native(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<EmbeddedDatabase>()?;
    module.add_class::<Client>()?;
    Ok(())
}
