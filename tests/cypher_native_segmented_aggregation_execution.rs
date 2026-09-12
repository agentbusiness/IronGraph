// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use irongraph::{
    Bookmark, Error, ErrorCode, ProjectId, Result, ScalarValue,
    cypher::{
        BindCapabilities, ColumnType, ExecutionContext, ExecutionOutput, QueryEngine, ResultValue,
    },
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend, ResidentGroup,
        ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest, ResidentNodePipelineRequest,
        ResidentNodePipelineResult, ResidentProjectImage, ResidentSegmentedAggregationOperation,
        ResidentSegmentedAggregationRequest, ResidentSegmentedAggregationResult,
        ResidentSegmentedAggregationSource, ResidentSortRequest, ResidentSortResult,
        ResidentVectorQuery, ResidentVectorResult, ScratchReservation,
    },
    graph::{GraphStore, IndexCatalog, LayerMask, TemporalStore},
    types::{LabelId, PropertyId},
};
use ordered_float::OrderedFloat;
use tokio_util::sync::CancellationToken;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());
const BOOKMARK: Bookmark = Bookmark { term: 91, index: 0 };
const MEMORY_LIMIT_BYTES: usize = 128 * 1024 * 1024;
const RESERVED_BYTES: usize = 16 * 1024 * 1024;
const CERTIFIED_TCK_REPORT: &str = "/tmp/irongraph-tck-full-delete-temporal-cpu-certified.json";
const OFFICIAL_LITERAL_IDENTITIES: [(usize, &str, &str); 14] = [
    (
        1253,
        "features/expressions/aggregation/Aggregation2.feature",
        "[1] `max()` over integers",
    ),
    (
        1254,
        "features/expressions/aggregation/Aggregation2.feature",
        "[2] `min()` over integers",
    ),
    (
        1255,
        "features/expressions/aggregation/Aggregation2.feature",
        "[3] `max()` over floats",
    ),
    (
        1256,
        "features/expressions/aggregation/Aggregation2.feature",
        "[4] `min()` over floats",
    ),
    (
        1257,
        "features/expressions/aggregation/Aggregation2.feature",
        "[5] `max()` over mixed numeric values",
    ),
    (
        1258,
        "features/expressions/aggregation/Aggregation2.feature",
        "[6] `min()` over mixed numeric values",
    ),
    (
        1259,
        "features/expressions/aggregation/Aggregation2.feature",
        "[7] `max()` over strings",
    ),
    (
        1260,
        "features/expressions/aggregation/Aggregation2.feature",
        "[8] `min()` over strings",
    ),
    (
        1261,
        "features/expressions/aggregation/Aggregation2.feature",
        "[9] `max()` over lists",
    ),
    (
        1262,
        "features/expressions/aggregation/Aggregation2.feature",
        "[10] `min()` over lists",
    ),
    (
        1263,
        "features/expressions/aggregation/Aggregation2.feature",
        "[11] `max()` over mixed values",
    ),
    (
        1264,
        "features/expressions/aggregation/Aggregation2.feature",
        "[12] `min()` over mixed values",
    ),
    (
        1284,
        "features/expressions/aggregation/Aggregation8.feature",
        "[3] Collect distinct nulls",
    ),
    (
        1285,
        "features/expressions/aggregation/Aggregation8.feature",
        "[4] Collect distinct values mixed with nulls",
    ),
];

fn assert_certified_report_identities<'a>(
    identities: impl IntoIterator<Item = (usize, &'a str, &'a str)>,
) {
    let report: serde_json::Value = serde_json::from_slice(
        &std::fs::read(CERTIFIED_TCK_REPORT).expect("certified TCK report is readable"),
    )
    .expect("certified TCK report is valid JSON");
    assert_eq!(report["total"].as_u64(), Some(3_897));
    assert_eq!(report["cpu_passed"].as_u64(), Some(3_897));
    assert_eq!(report["metal_passed"].as_u64(), Some(3_184));
    let scenarios = report["scenarios"]
        .as_array()
        .expect("certified report has a scenario array");
    assert_eq!(scenarios.len(), 3_897);
    let mut selected = std::collections::BTreeSet::new();
    for (stored_id, feature, expanded_name) in identities {
        assert!(
            selected.insert((feature, expanded_name)),
            "duplicate local TCK identity ({feature}, {expanded_name})"
        );
        let matches = scenarios
            .iter()
            .enumerate()
            .filter_map(|(index, scenario)| {
                let path = scenario.get("path")?.as_str()?;
                let name = scenario.get("name")?.as_str()?;
                (path.ends_with(feature) && name == expanded_name).then_some(index)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            matches.len(),
            1,
            "({feature}, {expanded_name}) resolved to {matches:?}"
        );
        assert_eq!(
            stored_id, matches[0],
            "wrong report index for {expanded_name}"
        );
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Sabotage {
    None,
    ForeignObligation,
    CpuReceiptAsMetal,
}

#[derive(Default)]
struct Observations {
    pins: AtomicUsize,
    aggregate_calls: AtomicUsize,
    unexpected_routes: AtomicUsize,
    requests: Mutex<Vec<ResidentSegmentedAggregationRequest>>,
}

/// A strict execution observer. Before pinning it advertises Metal, preventing the query engine
/// from entering its generic CPU row executor. The valid pinned form delegates exactly one
/// segmented request to the CPU semantic reference and exposes honest CPU receipt provenance.
struct ObservedAggregationBackend {
    inner: Box<dyn ExecutionBackend>,
    pinned: bool,
    sabotage: Sabotage,
    observations: Arc<Observations>,
}

impl ObservedAggregationBackend {
    fn new(inner: CpuBackend, sabotage: Sabotage) -> Self {
        Self {
            inner: Box::new(inner),
            pinned: false,
            sabotage,
            observations: Arc::new(Observations::default()),
        }
    }

    fn observations(&self) -> Arc<Observations> {
        Arc::clone(&self.observations)
    }

    fn reject<T>(&self, route: &'static str) -> Result<T> {
        self.observations
            .unexpected_routes
            .fetch_add(1, Ordering::SeqCst);
        Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!("strict segmented integration test rejected `{route}`"),
        ))
    }
}

impl ExecutionBackend for ObservedAggregationBackend {
    fn kind(&self) -> BackendKind {
        if self.pinned && self.sabotage != Sabotage::CpuReceiptAsMetal {
            BackendKind::Cpu
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
        if self.pinned {
            return self.reject("pin_project_twice");
        }
        self.observations.pins.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Self {
            inner: self.inner.pin_project(project)?,
            pinned: true,
            sabotage: self.sabotage,
            observations: Arc::clone(&self.observations),
        }))
    }

    fn admit_project(&mut self, image: ResidentProjectImage) -> Result<()> {
        self.inner.admit_project(image)
    }

    fn replace_all_projects(&mut self, images: Vec<ResidentProjectImage>) -> Result<()> {
        self.inner.replace_all_projects(images)
    }

    fn evict_project(&mut self, project: ProjectId) -> Result<()> {
        self.inner.evict_project(project)
    }

    fn advance_bookmark(&mut self, bookmark: Bookmark) {
        self.inner.advance_bookmark(bookmark);
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

    fn supports_native_segmented_aggregation(&self) -> bool {
        true
    }

    fn execute_segmented_aggregation(
        &self,
        request: &ResidentSegmentedAggregationRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentSegmentedAggregationResult> {
        if !self.pinned {
            return self.reject("execute_segmented_aggregation_without_pin");
        }
        self.observations
            .aggregate_calls
            .fetch_add(1, Ordering::SeqCst);
        self.observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(request.clone());
        if self.sabotage == Sabotage::ForeignObligation {
            let mut foreign = request.clone();
            foreign.aggregate_obligations[0].id = foreign.aggregate_obligations[0]
                .id
                .checked_add(10_000)
                .ok_or_else(|| Error::internal("test obligation identity overflow"))?;
            foreign.validate()?;
            self.inner
                .execute_segmented_aggregation(&foreign, cancellation)
        } else {
            self.inner
                .execute_segmented_aggregation(request, cancellation)
        }
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

struct Fixture {
    graph: GraphStore,
}

impl Fixture {
    fn new() -> Self {
        Self {
            graph: GraphStore::default(),
        }
    }

    fn backend(&self, sabotage: Sabotage) -> Result<ObservedAggregationBackend> {
        let image = ResidentProjectImage::build(
            PROJECT,
            BOOKMARK,
            &self.graph,
            &TemporalStore::default(),
            &IndexCatalog::default(),
        )?;
        let mut cpu = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
        cpu.admit_project(image)?;
        Ok(ObservedAggregationBackend::new(cpu, sabotage))
    }
}

fn context<'a>(
    fixture: &'a Fixture,
    backend: &'a dyn ExecutionBackend,
    parameters: BTreeMap<String, ResultValue>,
) -> ExecutionContext<'a> {
    ExecutionContext {
        project_id: PROJECT,
        graph: &fixture.graph,
        binding_catalog: fixture.graph.catalog(),
        prior_graph_mutations: &[],
        temporal: None,
        prior_temporal_mutations: &[],
        vector_indexes: BTreeMap::new(),
        scalar_indexes: None,
        text_embedding: None,
        parameters,
        bookmark: BOOKMARK,
        mutation_revision: fixture.graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id: 1,
        next_edge_id: 1,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            require_native_execution: true,
            ..BindCapabilities::default()
        },
        max_result_rows: 4096,
        max_batch_rows: 3,
        optimizer_statistics: None,
        backend: Some(backend),
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(10)),
        resolved_query_at_time_nanos: None,
    }
}

fn execute(
    fixture: &Fixture,
    backend: &dyn ExecutionBackend,
    query: &str,
    parameters: BTreeMap<String, ResultValue>,
) -> Result<ExecutionOutput> {
    QueryEngine.execute(query, &mut context(fixture, backend, parameters))
}

fn output_rows(output: &ExecutionOutput) -> Result<Vec<Vec<ResultValue>>> {
    let mut rows = Vec::new();
    for batch in &output.result.batches {
        if batch.columns.len() != output.result.schema.len()
            || batch
                .columns
                .iter()
                .any(|column| column.values.len() != batch.row_count)
        {
            return Err(Error::internal(
                "segmented integration output has invalid columnar shape",
            ));
        }
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
    Ok(rows)
}

fn null() -> ResultValue {
    ResultValue::Scalar(ScalarValue::Null)
}

fn integer(value: i64) -> ResultValue {
    ResultValue::Scalar(ScalarValue::Integer(value))
}

fn float(value: f64) -> ResultValue {
    ResultValue::Scalar(ScalarValue::Float(OrderedFloat(value)))
}

fn string(value: &'static str) -> ResultValue {
    ResultValue::Scalar(ScalarValue::String(Arc::from(value)))
}

fn list(values: Vec<ResultValue>) -> ResultValue {
    ResultValue::List(values)
}

#[test]
fn exact_literal_unwind_portfolio_routes_once_and_decodes_exact_backend_rows() -> Result<()> {
    let cases = [
        (
            1253,
            "UNWIND [1, 2, 0, null, -1] AS x RETURN max(x)",
            "max(x)",
            integer(2),
        ),
        (
            1254,
            "UNWIND [1, 2, 0, null, -1] AS x RETURN min(x)",
            "min(x)",
            integer(-1),
        ),
        (
            1255,
            "UNWIND [1.0, 2.0, 0.5, null] AS x RETURN max(x)",
            "max(x)",
            float(2.0),
        ),
        (
            1256,
            "UNWIND [1.0, 2.0, 0.5, null] AS x RETURN min(x)",
            "min(x)",
            float(0.5),
        ),
        (
            1257,
            "UNWIND [1, 2.0, 5, null, 3.2, 0.1] AS x RETURN max(x)",
            "max(x)",
            integer(5),
        ),
        (
            1258,
            "UNWIND [1, 2.0, 5, null, 3.2, 0.1] AS x RETURN min(x)",
            "min(x)",
            float(0.1),
        ),
        (
            1259,
            "UNWIND ['a', 'b', 'B', null, 'abc', 'abc1'] AS i RETURN max(i)",
            "max(i)",
            string("b"),
        ),
        (
            1260,
            "UNWIND ['a', 'b', 'B', null, 'abc', 'abc1'] AS i RETURN min(i)",
            "min(i)",
            string("B"),
        ),
        (
            1261,
            "UNWIND [[1], [2], [2, 1]] AS x RETURN max(x)",
            "max(x)",
            list(vec![integer(2), integer(1)]),
        ),
        (
            1262,
            "UNWIND [[1], [2], [2, 1]] AS x RETURN min(x)",
            "min(x)",
            list(vec![integer(1)]),
        ),
        (
            1263,
            "UNWIND [1, 'a', null, [1, 2], 0.2, 'b'] AS x RETURN max(x)",
            "max(x)",
            integer(1),
        ),
        (
            1264,
            "UNWIND [1, 'a', null, [1, 2], 0.2, 'b'] AS x RETURN min(x)",
            "min(x)",
            list(vec![integer(1), integer(2)]),
        ),
        (
            1284,
            "UNWIND [null, null] AS x RETURN collect(DISTINCT x)",
            "collect(DISTINCT x)",
            list(vec![]),
        ),
        (
            1285,
            "UNWIND [null, 1, null] AS x RETURN collect(DISTINCT x)",
            "collect(DISTINCT x)",
            list(vec![integer(1)]),
        ),
    ];
    let case_count = cases.len();
    let fixture = Fixture::new();
    let backend = fixture.backend(Sabotage::None)?;
    let observations = backend.observations();
    for (id, query, column, expected) in cases {
        let output = execute(&fixture, &backend, query, BTreeMap::new()).map_err(|error| {
            Error::new(
                error.code,
                format!("native segmented case `{query}` failed: {error}"),
            )
        })?;
        assert_eq!(
            output.result.schema,
            vec![(column.to_owned(), expected.column_type())],
            "TCK {id} lost its exact output name or type"
        );
        assert_eq!(output_rows(&output)?, vec![vec![expected]], "TCK {id}");
    }
    assert_eq!(
        observations.aggregate_calls.load(Ordering::SeqCst),
        case_count
    );
    assert_eq!(observations.pins.load(Ordering::SeqCst), case_count);
    assert_eq!(observations.unexpected_routes.load(Ordering::SeqCst), 0);
    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    for request in &requests[8..12] {
        assert!(request.program.is_none());
        assert!(request.grouping_columns.is_empty());
        assert_eq!(request.input.column_count, 1);
        assert_eq!(request.aggregates.len(), 1);
        assert!(!request.aggregates[0].distinct);
        assert_eq!(request.maximum_output_groups, 1);
        assert_eq!(request.maximum_output_cells, 3);
        assert!(request.input.cells.len() > request.input.top_level_cell_count());
    }
    assert_eq!(requests[8].input.cells.len(), 7);
    assert_eq!(requests[9].input.cells.len(), 7);
    assert_eq!(requests[10].input.cells.len(), 8);
    assert_eq!(requests[11].input.cells.len(), 8);
    Ok(())
}

#[test]
#[ignore = "external assurance gate: requires the certified full TCK report"]
fn certified_report_uniquely_resolves_all_literal_aggregation_selectors() {
    assert_certified_report_identities(OFFICIAL_LITERAL_IDENTITIES);
}

#[test]
fn immutable_empty_parameter_and_group_inputs_stay_inside_one_backend_pass() -> Result<()> {
    let fixture = Fixture::new();
    let backend = fixture.backend(Sabotage::None)?;
    let observations = backend.observations();

    let empty = execute(
        &fixture,
        &backend,
        "UNWIND [] AS x RETURN count(x) AS c, sum(x) AS s, avg(x) AS a, min(x) AS lo, max(x) AS hi, collect(x) AS xs",
        BTreeMap::new(),
    )?;
    assert_eq!(
        output_rows(&empty)?,
        vec![vec![
            integer(0),
            integer(0),
            null(),
            null(),
            null(),
            list(vec![]),
        ]]
    );

    let parameters = BTreeMap::from([(
        "values".to_owned(),
        list(vec![integer(1), integer(2), integer(2), null()]),
    )]);
    let parameter = execute(
        &fixture,
        &backend,
        "UNWIND $values AS x RETURN sum(x) AS total, count(DISTINCT x) AS unique",
        parameters,
    )?;
    assert_eq!(output_rows(&parameter)?, vec![vec![integer(5), integer(2)]]);

    let grouped = execute(
        &fixture,
        &backend,
        "UNWIND ['b', 'a', 'b'] AS x RETURN count(*) AS amount, x AS key",
        BTreeMap::new(),
    )?;
    assert_eq!(
        grouped.result.schema,
        vec![
            ("amount".to_owned(), ColumnType::Integer),
            ("key".to_owned(), ColumnType::String),
        ]
    );
    assert_eq!(
        output_rows(&grouped)?,
        vec![vec![integer(2), string("b")], vec![integer(1), string("a")]]
    );

    assert_eq!(observations.aggregate_calls.load(Ordering::SeqCst), 3);
    assert_eq!(observations.unexpected_routes.load(Ordering::SeqCst), 0);
    Ok(())
}

#[test]
fn executions_are_fresh_while_obligations_and_input_shape_are_deterministic() -> Result<()> {
    let fixture = Fixture::new();
    let backend = fixture.backend(Sabotage::None)?;
    let observations = backend.observations();
    let query = "UNWIND [1, 1, 2] AS x RETURN x AS key, count(*) AS amount";
    execute(&fixture, &backend, query, BTreeMap::new())?;
    execute(&fixture, &backend, query, BTreeMap::new())?;
    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    assert_eq!(requests.len(), 2);
    assert_ne!(requests[0].execution, requests[1].execution);
    assert_eq!(
        requests[0].segmentation_obligation,
        requests[1].segmentation_obligation
    );
    assert_eq!(
        requests[0].aggregate_obligations,
        requests[1].aggregate_obligations
    );
    assert_eq!(requests[0].input, requests[1].input);
    assert_eq!(requests[0].grouping_columns, vec![0]);
    assert_eq!(requests[0].project, PROJECT);
    assert_eq!(requests[0].expected_bookmark, BOOKMARK);
    assert_eq!(
        requests[0].expected_graph_revision,
        fixture.graph.revision()
    );
    assert_eq!(
        requests[0].expected_layout_version,
        fixture.graph.layout_version()
    );
    Ok(())
}

#[test]
fn graph_upstream_executes_while_unsupported_post_aggregate_work_fails_closed() -> Result<()> {
    let fixture = Fixture::new();
    let backend = fixture.backend(Sabotage::None)?;
    let observations = backend.observations();
    let query = "MATCH (n) RETURN collect(n)";
    let output = execute(&fixture, &backend, query, BTreeMap::new())?;
    assert_eq!(output_rows(&output)?, vec![vec![list(Vec::new())]]);
    assert_eq!(observations.pins.load(Ordering::SeqCst), 1);
    assert_eq!(observations.aggregate_calls.load(Ordering::SeqCst), 1);
    assert_eq!(observations.unexpected_routes.load(Ordering::SeqCst), 0);

    for query in [
        "UNWIND [1, 2] AS x RETURN sum(x) + 1",
        "UNWIND [1, 2] AS x RETURN sum(-x)",
    ] {
        let fixture = Fixture::new();
        let backend = fixture.backend(Sabotage::None)?;
        let observations = backend.observations();
        let error = execute(&fixture, &backend, query, BTreeMap::new())
            .expect_err("unsupported post-aggregate work entered host semantic execution");
        assert_eq!(error.code, ErrorCode::GpuAdmissionFailure, "{query}");
        assert_eq!(
            observations.aggregate_calls.load(Ordering::SeqCst),
            0,
            "{query} dispatched a partial aggregate request"
        );
    }

    let fixture = Fixture::new();
    let backend = fixture.backend(Sabotage::None)?;
    let observations = backend.observations();
    let query = "UNWIND range(1000000, 2000000) AS i WITH i LIMIT 3000 RETURN sum(i) AS total";
    let output = execute(&fixture, &backend, query, BTreeMap::new())?;
    assert_eq!(output_rows(&output)?, vec![vec![integer(3_004_498_500)]]);
    assert_eq!(observations.aggregate_calls.load(Ordering::SeqCst), 1);
    assert_eq!(observations.unexpected_routes.load(Ordering::SeqCst), 0);

    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let [request] = requests.as_slice() else {
        return Err(Error::internal(format!(
            "sealed range SUM dispatched {} segmented requests instead of one",
            requests.len()
        )));
    };
    if request.input.row_count != 0
        || request.input.column_count != 0
        || !request.input.cells.is_empty()
        || !request.input.arena.is_empty()
    {
        return Err(Error::internal(
            "sealed range SUM crossed the backend boundary as host-built rows",
        ));
    }
    let program = request
        .program
        .as_ref()
        .ok_or_else(|| Error::internal("sealed range SUM omitted its fused program"))?;
    if !matches!(
        &program.source,
        ResidentSegmentedAggregationSource::Range { .. }
    ) || !matches!(
        program.stages.as_slice(),
        [
            irongraph::gpu::ResidentSegmentedAggregationStage {
                operation: ResidentSegmentedAggregationOperation::Limit { rows: 3_000 },
                ..
            },
            irongraph::gpu::ResidentSegmentedAggregationStage {
                operation: ResidentSegmentedAggregationOperation::Aggregate { .. },
                ..
            }
        ]
    ) {
        return Err(Error::internal(
            "range SUM did not remain one sealed Range -> LIMIT -> Aggregate program",
        ));
    }
    Ok(())
}

#[test]
fn foreign_request_and_wrong_backend_receipts_are_rejected_before_decode() -> Result<()> {
    let query = "UNWIND [1, 2, 3] AS x RETURN sum(x) AS total";
    for sabotage in [Sabotage::ForeignObligation, Sabotage::CpuReceiptAsMetal] {
        let fixture = Fixture::new();
        let backend = fixture.backend(sabotage)?;
        let observations = backend.observations();
        let error = execute(&fixture, &backend, query, BTreeMap::new())
            .expect_err("sabotaged segmented result reached public decoding");
        assert_eq!(error.code, ErrorCode::CorruptStorage, "{sabotage:?}");
        assert_eq!(observations.aggregate_calls.load(Ordering::SeqCst), 1);
    }
    Ok(())
}
