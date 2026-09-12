// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Fresh-report identity, semantic oracle, and poison-backend route gate for String1 [1].
//!
//! The ignored strict gate requires one complete graph-free quantifier command. The outer
//! backend reports Metal until its immutable project pin and poisons every decomposed primitive;
//! the pinned backend delegates only the sealed quantifier command to the CPU semantic reference.

use std::{
    collections::BTreeMap,
    fs,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use irongraph::{
    Bookmark, Error, ErrorCode, ProjectId, Result, ScalarValue,
    cypher::{
        BindCapabilities, ColumnType, ExecutionContext, Expression, PhysicalOperator, QueryEngine,
        ResultValue, StatementStats, bind, parse, plan,
    },
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend, ResidentGroup,
        ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest, ResidentNodePipelineRequest,
        ResidentNodePipelineResult, ResidentProjectImage, ResidentQuantifierExpression,
        ResidentQuantifierProgramRequest, ResidentQuantifierProgramResult,
        ResidentQuantifierSource, ResidentQuantifierStage, ResidentQuantifierValue,
        ResidentSortRequest, ResidentSortResult, ResidentVectorQuery, ResidentVectorResult,
        ScratchReservation,
    },
    graph::{GraphStore, IndexCatalog, LayerMask, NameCatalog, TemporalStore},
    types::{LabelId, PropertyId},
};
use sha2::{Digest as _, Sha256};
use tokio_util::sync::CancellationToken;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::from_u128(
    0x5354_5249_4e47_315f_5355_4253_5452_0001,
));
const QUERY: &str = "RETURN substring('0123456789', 1) AS s";
const REPORT_INDEX: usize = 2_784;
const FEATURE: &str = "expressions/string/String1.feature";
const SCENARIO: &str = "[1] `substring()` with default second argument";
const REPORT: &str =
    "/tmp/irongraph-tck-full-20260721-after-string-predicate-precedence-merge9-3.json";
const REPORT_SHA256: &str = "77943059a3208e2adf43a771e757e1d800f135fd4a055aca076efb2aa041d213";
const MAX_RESULT_ROWS: usize = 64;
const MEMORY_LIMIT_BYTES: usize = 64 * 1024 * 1024;
const RESERVED_BYTES: usize = 8 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq)]
struct ObservedResult {
    schema: Vec<(String, ColumnType)>,
    rows: Vec<Vec<ResultValue>>,
}

#[derive(Default)]
struct Observations {
    pins: AtomicUsize,
    quantifier_calls: AtomicUsize,
    forbidden_calls: AtomicUsize,
    requests: Mutex<Vec<ResidentQuantifierProgramRequest>>,
}

struct StrictSubstringBackend {
    inner: Box<dyn ExecutionBackend>,
    pinned: bool,
    observations: Arc<Observations>,
}

impl StrictSubstringBackend {
    fn new(graph: &GraphStore) -> Result<Self> {
        let mut inner = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
        inner.admit_project(ResidentProjectImage::build(
            PROJECT,
            bookmark(graph),
            graph,
            &TemporalStore::default(),
            &IndexCatalog::default(),
        )?)?;
        Ok(Self {
            inner: Box::new(inner),
            pinned: false,
            observations: Arc::new(Observations::default()),
        })
    }

    fn observations(&self) -> Arc<Observations> {
        Arc::clone(&self.observations)
    }

    fn reject<T>(&self, route: &'static str) -> Result<T> {
        self.observations
            .forbidden_calls
            .fetch_add(1, Ordering::SeqCst);
        Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!("strict String1 substring backend rejected route `{route}`"),
        ))
    }
}

impl ExecutionBackend for StrictSubstringBackend {
    fn kind(&self) -> BackendKind {
        if self.pinned {
            self.inner.kind()
        } else {
            BackendKind::Metal
        }
    }

    fn available_query_scratch_bytes(&self) -> usize {
        self.inner.available_query_scratch_bytes()
    }

    fn reserve_query_scratch(&self, bytes: usize) -> Result<ScratchReservation> {
        self.inner.reserve_query_scratch(bytes)
    }

    fn resident_project_bytes(&self, project: ProjectId) -> Option<usize> {
        self.inner.resident_project_bytes(project)
    }

    fn pin_project(&self, project: ProjectId) -> Result<Box<dyn ExecutionBackend>> {
        if self.pinned || project != PROJECT {
            return self.reject("pin_project");
        }
        self.observations.pins.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Self {
            inner: self.inner.pin_project(project)?,
            pinned: true,
            observations: Arc::clone(&self.observations),
        }))
    }

    fn admit_project(&mut self, image: ResidentProjectImage) -> Result<()> {
        if self.pinned {
            return self.reject("admit_project");
        }
        self.inner.admit_project(image)
    }

    fn replace_all_projects(&mut self, images: Vec<ResidentProjectImage>) -> Result<()> {
        if self.pinned {
            return self.reject("replace_all_projects");
        }
        self.inner.replace_all_projects(images)
    }

    fn evict_project(&mut self, project: ProjectId) -> Result<()> {
        if self.pinned {
            return self.reject("evict_project");
        }
        self.inner.evict_project(project)
    }

    fn advance_bookmark(&mut self, bookmark: Bookmark) {
        if self.pinned {
            self.observations
                .forbidden_calls
                .fetch_add(1, Ordering::SeqCst);
        } else {
            self.inner.advance_bookmark(bookmark);
        }
    }

    fn resident_revision(&self) -> Option<u64> {
        self.inner.resident_revision()
    }

    fn resident_graph_revision(&self, project: ProjectId) -> Option<u64> {
        self.inner.resident_graph_revision(project)
    }

    fn resident_bookmark(&self, project: ProjectId) -> Option<Bookmark> {
        self.inner.resident_bookmark(project)
    }

    fn scan_nodes(
        &self,
        _project: ProjectId,
        _label: Option<LabelId>,
        _layers: LayerMask,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.reject("scan_nodes")
    }

    fn filter_node_i64(
        &self,
        _project: ProjectId,
        _property: PropertyId,
        _operation: CompareOp,
        _operand: i64,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.reject("filter_node_i64")
    }

    fn expand_project_out(
        &self,
        _project: ProjectId,
        _sources: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.reject("expand_project_out")
    }

    fn expand_project_in(
        &self,
        _project: ProjectId,
        _targets: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.reject("expand_project_in")
    }

    fn search_vectors(
        &self,
        _request: &ResidentVectorQuery,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentVectorResult> {
        self.reject("search_vectors")
    }

    fn sort_rows(
        &self,
        _request: &ResidentSortRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentSortResult> {
        self.reject("sort_rows")
    }

    fn join_node_i64(
        &self,
        _request: &ResidentJoinRequest,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentJoinPair>> {
        self.reject("join_node_i64")
    }

    fn group_node_i64(
        &self,
        _request: &ResidentGroupRequest,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentGroup>> {
        self.reject("group_node_i64")
    }

    fn execute_node_pipeline(
        &self,
        _request: &ResidentNodePipelineRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentNodePipelineResult> {
        self.reject("execute_node_pipeline")
    }

    fn supports_native_quantifier_program(&self) -> bool {
        true
    }

    fn execute_quantifier_program(
        &self,
        request: &ResidentQuantifierProgramRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentQuantifierProgramResult> {
        if !self.pinned {
            return self.reject("execute_quantifier_program_unpinned");
        }
        request.validate()?;
        self.observations
            .quantifier_calls
            .fetch_add(1, Ordering::SeqCst);
        self.observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(request.clone());
        self.inner.execute_quantifier_program(request, cancellation)
    }

    fn filter_i64(
        &self,
        _values: &[i64],
        _validity: &[bool],
        _operation: CompareOp,
        _operand: i64,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.reject("filter_i64")
    }

    fn expand_out(
        &self,
        _sources: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.reject("expand_out")
    }

    fn exact_l2(
        &self,
        _matrix: &[f32],
        _rows: usize,
        _dimension: usize,
        _query: &[f32],
        _cancellation: &CancellationToken,
    ) -> Result<DistanceBatch> {
        self.reject("exact_l2")
    }
}

fn bookmark(graph: &GraphStore) -> Bookmark {
    Bookmark {
        term: 91,
        index: graph.revision(),
    }
}

fn context<'a>(
    graph: &'a GraphStore,
    backend: Option<&'a dyn ExecutionBackend>,
    require_native_execution: bool,
) -> ExecutionContext<'a> {
    ExecutionContext {
        project_id: PROJECT,
        graph,
        binding_catalog: graph.catalog(),
        prior_graph_mutations: &[],
        temporal: None,
        prior_temporal_mutations: &[],
        vector_indexes: BTreeMap::new(),
        scalar_indexes: None,
        text_embedding: None,
        parameters: BTreeMap::new(),
        bookmark: bookmark(graph),
        mutation_revision: graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id: 1,
        next_edge_id: 1,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            require_native_execution,
            ..BindCapabilities::default()
        },
        max_result_rows: MAX_RESULT_ROWS,
        max_batch_rows: MAX_RESULT_ROWS,
        optimizer_statistics: None,
        backend,
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(10)),
        resolved_query_at_time_nanos: None,
    }
}

fn execute_query(
    query: &str,
    graph: &GraphStore,
    backend: Option<&dyn ExecutionBackend>,
    require_native_execution: bool,
) -> Result<ObservedResult> {
    let output = QueryEngine.execute(
        query,
        &mut context(graph, backend, require_native_execution),
    )?;
    if !output.graph_mutations.is_empty()
        || !output.temporal_mutations.is_empty()
        || output.result.statistics != StatementStats::default()
        || output.result.truncated
    {
        return Err(Error::internal(
            "read-only substring projection produced side effects or truncation",
        ));
    }
    let mut rows = Vec::new();
    for batch in &output.result.batches {
        for row in 0..batch.row_count {
            rows.push(
                batch
                    .columns
                    .iter()
                    .map(|column| column.values[row].clone())
                    .collect(),
            );
        }
    }
    Ok(ObservedResult {
        schema: output.result.schema,
        rows,
    })
}

fn execute(
    graph: &GraphStore,
    backend: Option<&dyn ExecutionBackend>,
    require_native_execution: bool,
) -> Result<ObservedResult> {
    execute_query(QUERY, graph, backend, require_native_execution)
}

fn expected() -> ObservedResult {
    ObservedResult {
        schema: vec![("s".to_owned(), ColumnType::String)],
        rows: vec![vec![ResultValue::Scalar(ScalarValue::String(
            "123456789".into(),
        ))]],
    }
}

fn assert_exact_request(request: &ResidentQuantifierProgramRequest) -> Result<()> {
    request.validate()?;
    if !matches!(&request.source, ResidentQuantifierSource::Unit) || request.program.slot_count != 1
    {
        return Err(Error::internal(
            "String1 [1] changed its graph-free quantifier source",
        ));
    }
    let [
        ResidentQuantifierStage::Project {
            keep_scope: false,
            bindings,
        },
    ] = request.program.stages.as_slice()
    else {
        return Err(Error::internal(
            "String1 [1] changed its one-project quantifier shape",
        ));
    };
    let [binding] = bindings.as_slice() else {
        return Err(Error::internal(
            "String1 [1] changed its substring binding cardinality",
        ));
    };
    let ResidentQuantifierExpression::Function {
        function,
        arguments,
    } = &binding.expression
    else {
        return Err(Error::internal(
            "String1 [1] did not lower substring as a native value function",
        ));
    };
    if format!("{function:?}") != "Substring"
        || !matches!(
            arguments.as_slice(),
            [
                ResidentQuantifierExpression::Literal(ResidentQuantifierValue::String(source)),
                ResidentQuantifierExpression::Literal(ResidentQuantifierValue::Integer(1)),
            ] if source == "0123456789"
        )
        || !matches!(
            request.program.outputs.as_slice(),
            [output] if output.name == "s" && output.source == binding.output
        )
    {
        return Err(Error::internal(
            "String1 [1] changed its sealed substring expression",
        ));
    }
    Ok(())
}

#[test]
fn physical_shape_and_generic_cpu_oracle_are_exact() -> Result<()> {
    let catalog = NameCatalog::default();
    let physical = plan(bind(parse(QUERY)?, &catalog, BindCapabilities::default())?)?;
    let [
        PhysicalOperator::Project {
            keep_scope: false,
            projection,
        },
    ] = physical.operators.as_slice()
    else {
        return Err(Error::internal(
            "String1 [1] is no longer one graph-free projection",
        ));
    };
    let [item] = projection.items.as_slice() else {
        return Err(Error::internal("String1 [1] changed its projection width"));
    };
    assert_eq!(item.alias.as_deref(), Some("s"));
    assert!(matches!(
        &item.expression,
        Expression::Function {
            name,
            distinct: false,
            arguments,
        } if name.len() == 1
            && name[0].eq_ignore_ascii_case("substring")
            && matches!(
                arguments.as_slice(),
                [
                    Expression::Literal(ScalarValue::String(source)),
                    Expression::Literal(ScalarValue::Integer(1)),
                ] if source.as_ref() == "0123456789"
            )
    ));

    let graph = GraphStore::default();
    assert_eq!(execute(&graph, None, false)?, expected());
    Ok(())
}

#[test]
#[ignore = "external assurance gate: requires the exact fresh 3,749-Metal full report"]
fn fresh_report_pins_the_single_string1_failure() {
    let bytes = fs::read(REPORT).expect("fresh TCK report is readable");
    assert_eq!(hex::encode(Sha256::digest(&bytes)), REPORT_SHA256);
    let report: serde_json::Value =
        serde_json::from_slice(&bytes).expect("fresh TCK report is valid JSON");
    assert_eq!(report["total"].as_u64(), Some(3_897));
    assert_eq!(report["cpu_passed"].as_u64(), Some(3_897));
    assert_eq!(report["metal_passed"].as_u64(), Some(3_749));
    let scenarios = report["scenarios"]
        .as_array()
        .expect("fresh report has a scenario array");
    assert_eq!(scenarios.len(), 3_897);
    let scenario = &scenarios[REPORT_INDEX];
    assert!(
        scenario["path"]
            .as_str()
            .is_some_and(|path| path.ends_with(FEATURE))
    );
    assert_eq!(scenario["name"].as_str(), Some(SCENARIO));
    assert_eq!(scenario["operation_count"].as_u64(), Some(1));
    assert_eq!(scenario["cpu_passed"].as_bool(), Some(true));
    assert_eq!(scenario["metal_passed"].as_bool(), Some(false));
    let failures = scenario["metal_failures"]
        .as_array()
        .expect("String1 [1] has Metal failures");
    assert_eq!(failures.len(), 1);
    let failure = failures[0].as_str().expect("Metal failure is text");
    assert!(failure.contains(QUERY));
    assert!(failure.contains("GpuAdmissionFailure"));
}

#[test]
fn substring_uses_one_complete_poisoned_quantifier_command() -> Result<()> {
    let graph = GraphStore::default();
    let reference = execute(&graph, None, false)?;
    let backend = StrictSubstringBackend::new(&graph)?;
    let observations = backend.observations();
    let native = execute(&graph, Some(&backend), true)?;

    assert_eq!(reference, expected());
    assert_eq!(native, reference);
    assert_eq!(observations.pins.load(Ordering::SeqCst), 1);
    assert_eq!(observations.quantifier_calls.load(Ordering::SeqCst), 1);
    assert_eq!(observations.forbidden_calls.load(Ordering::SeqCst), 0);
    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let [request] = requests.as_slice() else {
        return Err(Error::internal(
            "String1 [1] changed its native request cardinality",
        ));
    };
    assert_exact_request(request)
}

#[test]
fn substring_quantifier_preserves_unicode_length_clamping_and_nulls() -> Result<()> {
    let graph = GraphStore::default();
    for (query, value_type, expected_value) in [
        (
            "RETURN substring('aé😊z', 1) AS s",
            ColumnType::String,
            ScalarValue::String("é😊z".into()),
        ),
        (
            "RETURN substring('aé😊z', 1, 2) AS s",
            ColumnType::String,
            ScalarValue::String("é😊".into()),
        ),
        (
            "RETURN substring('abc', 99) AS s",
            ColumnType::String,
            ScalarValue::String("".into()),
        ),
        (
            "RETURN substring('abc', 1, 0) AS s",
            ColumnType::String,
            ScalarValue::String("".into()),
        ),
        (
            "RETURN substring(null, 1) AS s",
            ColumnType::Null,
            ScalarValue::Null,
        ),
        (
            "RETURN substring('abc', null) AS s",
            ColumnType::Null,
            ScalarValue::Null,
        ),
        (
            "RETURN substring('abc', 1, null) AS s",
            ColumnType::Null,
            ScalarValue::Null,
        ),
    ] {
        let expected = ObservedResult {
            schema: vec![("s".to_owned(), value_type)],
            rows: vec![vec![ResultValue::Scalar(expected_value)]],
        };
        let reference = execute_query(query, &graph, None, false)?;
        let backend = StrictSubstringBackend::new(&graph)?;
        let observations = backend.observations();
        let native = execute_query(query, &graph, Some(&backend), true)?;
        assert_eq!(reference, expected, "generic oracle changed for {query}");
        assert_eq!(native, reference, "native quantifier mismatch for {query}");
        assert_eq!(observations.pins.load(Ordering::SeqCst), 1, "{query}");
        assert_eq!(
            observations.quantifier_calls.load(Ordering::SeqCst),
            1,
            "{query}"
        );
        assert_eq!(
            observations.forbidden_calls.load(Ordering::SeqCst),
            0,
            "{query}"
        );
    }
    Ok(())
}
