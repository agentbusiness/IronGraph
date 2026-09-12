// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Strict CPU proof for the compiler-only Set4/Set5 literal-map tranche.
//!
//! The outer backend advertises Metal so the generic executor cannot run. After one immutable
//! generation pin it delegates only the already-existing complete node mutation, node read, or
//! nullable-relation command to the CPU semantic backend. Every other backend entrypoint is
//! poisoned. This proves the compiler rewrite itself without claiming a real-Metal measurement.

use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use irongraph::{
    Bookmark, Error, ErrorCode, Layer, NodeId, ProjectId, Result, ScalarValue,
    cypher::{BindCapabilities, ExecutionContext, ExecutionOutput, QueryEngine, ResultValue},
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend, ResidentGroup,
        ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest, ResidentMutationOperation,
        ResidentNodePipelineRequest, ResidentNodePipelineResult, ResidentNullableRelationRequest,
        ResidentNullableRelationResult, ResidentProjectImage, ResidentRowMutationRequest,
        ResidentRowMutationResult, ResidentSortRequest, ResidentSortResult, ResidentVectorQuery,
        ResidentVectorResult, ScratchReservation,
    },
    graph::{GraphStore, IndexCatalog, LayerMask, NodeInput, TemporalStore},
    types::{LabelId, PropertyId},
};
use tokio_util::sync::CancellationToken;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::from_u128(
    0x5345_5434_5f53_4554_355f_4d41_5053_0001,
));
const BOOKMARK: Bookmark = Bookmark { term: 47, index: 9 };

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Route {
    Mutation(usize),
    NullableNoop,
    ReadNoop,
}

#[derive(Clone, Copy, Debug)]
enum FixtureShape {
    Empty,
    Propertyless,
    Name,
    NameAndName2,
}

#[derive(Clone, Copy, Debug)]
enum ExpectedScalar {
    Integer(i64),
    String(&'static str),
}

#[derive(Clone, Copy, Debug)]
enum ExpectedResult {
    Null,
    Node(&'static [(&'static str, ExpectedScalar)]),
}

#[derive(Clone, Copy, Debug)]
struct Case {
    report_index: usize,
    feature: &'static str,
    scenario: u8,
    query: &'static str,
    fixture: FixtureShape,
    expected: ExpectedResult,
    route: Route,
    properties_set: u64,
}

const CASES: [Case; 10] = [
    Case {
        report_index: 845,
        feature: "clauses/set/Set4.feature",
        scenario: 1,
        query: "MATCH (n:X) SET n = {name: 'A', name2: 'B', num: 5} RETURN n",
        fixture: FixtureShape::Propertyless,
        expected: ExpectedResult::Node(&[
            ("name", ExpectedScalar::String("A")),
            ("name2", ExpectedScalar::String("B")),
            ("num", ExpectedScalar::Integer(5)),
        ]),
        route: Route::Mutation(3),
        properties_set: 3,
    },
    Case {
        report_index: 846,
        feature: "clauses/set/Set4.feature",
        scenario: 2,
        query: "MATCH (n:X {name: 'A'}) SET n = {name: 'B', baz: 'C'} RETURN n",
        fixture: FixtureShape::NameAndName2,
        expected: ExpectedResult::Node(&[
            ("baz", ExpectedScalar::String("C")),
            ("name", ExpectedScalar::String("B")),
        ]),
        route: Route::Mutation(3),
        properties_set: 3,
    },
    Case {
        report_index: 847,
        feature: "clauses/set/Set4.feature",
        scenario: 3,
        query: "MATCH (n:X {name: 'A'}) SET n = {name: 'B', name2: null, baz: 'C'} RETURN n",
        fixture: FixtureShape::NameAndName2,
        expected: ExpectedResult::Node(&[
            ("baz", ExpectedScalar::String("C")),
            ("name", ExpectedScalar::String("B")),
        ]),
        route: Route::Mutation(3),
        properties_set: 3,
    },
    Case {
        report_index: 848,
        feature: "clauses/set/Set4.feature",
        scenario: 4,
        query: "MATCH (n:X {name: 'A'}) SET n = {} RETURN n",
        fixture: FixtureShape::NameAndName2,
        expected: ExpectedResult::Node(&[]),
        route: Route::Mutation(2),
        properties_set: 2,
    },
    Case {
        report_index: 849,
        feature: "clauses/set/Set4.feature",
        scenario: 5,
        query: "OPTIONAL MATCH (a:DoesNotExist) SET a = {num: 42} RETURN a",
        fixture: FixtureShape::Empty,
        expected: ExpectedResult::Null,
        route: Route::NullableNoop,
        properties_set: 0,
    },
    Case {
        report_index: 850,
        feature: "clauses/set/Set5.feature",
        scenario: 1,
        query: "OPTIONAL MATCH (a:DoesNotExist) SET a += {num: 42} RETURN a",
        fixture: FixtureShape::Empty,
        expected: ExpectedResult::Null,
        route: Route::NullableNoop,
        properties_set: 0,
    },
    Case {
        report_index: 851,
        feature: "clauses/set/Set5.feature",
        scenario: 2,
        query: "MATCH (n:X {name: 'A'}) SET n += {name2: 'C'} RETURN n",
        fixture: FixtureShape::NameAndName2,
        expected: ExpectedResult::Node(&[
            ("name", ExpectedScalar::String("A")),
            ("name2", ExpectedScalar::String("C")),
        ]),
        route: Route::Mutation(1),
        properties_set: 1,
    },
    Case {
        report_index: 852,
        feature: "clauses/set/Set5.feature",
        scenario: 3,
        query: "MATCH (n:X {name: 'A'}) SET n += {name2: 'B'} RETURN n",
        fixture: FixtureShape::Name,
        expected: ExpectedResult::Node(&[
            ("name", ExpectedScalar::String("A")),
            ("name2", ExpectedScalar::String("B")),
        ]),
        route: Route::Mutation(1),
        properties_set: 1,
    },
    Case {
        report_index: 853,
        feature: "clauses/set/Set5.feature",
        scenario: 4,
        query: "MATCH (n:X {name: 'A'}) SET n += {name: null} RETURN n",
        fixture: FixtureShape::NameAndName2,
        expected: ExpectedResult::Node(&[("name2", ExpectedScalar::String("B"))]),
        route: Route::Mutation(1),
        properties_set: 1,
    },
    Case {
        report_index: 854,
        feature: "clauses/set/Set5.feature",
        scenario: 5,
        query: "MATCH (n:X {name: 'A'}) SET n += {} RETURN n",
        fixture: FixtureShape::NameAndName2,
        expected: ExpectedResult::Node(&[
            ("name", ExpectedScalar::String("A")),
            ("name2", ExpectedScalar::String("B")),
        ]),
        route: Route::ReadNoop,
        properties_set: 0,
    },
];

#[derive(Default)]
struct Observations {
    pins: AtomicUsize,
    mutation_calls: AtomicUsize,
    nullable_calls: AtomicUsize,
    read_calls: AtomicUsize,
    forbidden_calls: AtomicUsize,
    mutation_command_counts: Mutex<Vec<usize>>,
}

struct StrictLiteralMapBackend {
    inner: Box<dyn ExecutionBackend>,
    pinned: bool,
    observations: Arc<Observations>,
}

impl StrictLiteralMapBackend {
    fn new(inner: CpuBackend) -> Self {
        Self {
            inner: Box::new(inner),
            pinned: false,
            observations: Arc::new(Observations::default()),
        }
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
            format!("strict literal-map route rejected `{route}`"),
        ))
    }
}

impl ExecutionBackend for StrictLiteralMapBackend {
    fn kind(&self) -> BackendKind {
        if self.pinned {
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
        request: &ResidentNodePipelineRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentNodePipelineResult> {
        if !self.pinned {
            return self.reject("execute_node_pipeline_unpinned");
        }
        if let Some(program) = request.mutation.as_ref() {
            if program.continuation.is_none()
                || program.commands.is_empty()
                || !program.commands.iter().all(|command| {
                    matches!(
                        command.operation,
                        ResidentMutationOperation::SetProperty { .. }
                    )
                })
            {
                return self.reject("non_scalar_map_mutation");
            }
            self.observations
                .mutation_calls
                .fetch_add(1, Ordering::SeqCst);
            self.observations
                .mutation_command_counts
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .push(program.commands.len());
        } else {
            self.observations.read_calls.fetch_add(1, Ordering::SeqCst);
        }
        self.inner.execute_node_pipeline(request, cancellation)
    }

    fn execute_row_mutation(
        &self,
        _request: &ResidentRowMutationRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentRowMutationResult> {
        self.reject("execute_row_mutation")
    }

    fn execute_nullable_relation(
        &self,
        request: &ResidentNullableRelationRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentNullableRelationResult> {
        if !self.pinned {
            return self.reject("execute_nullable_relation_unpinned");
        }
        self.observations
            .nullable_calls
            .fetch_add(1, Ordering::SeqCst);
        self.inner.execute_nullable_relation(request, cancellation)
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

fn fixture(shape: FixtureShape) -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    if matches!(shape, FixtureShape::Empty) {
        return Ok(graph);
    }
    let x = graph.catalog_mut().intern_label("X")?;
    let mut properties = Vec::new();
    if matches!(shape, FixtureShape::Name | FixtureShape::NameAndName2) {
        let name = graph.catalog_mut().intern_property("name")?;
        properties.push((name, ScalarValue::String("A".into())));
    }
    if matches!(shape, FixtureShape::NameAndName2) {
        let name2 = graph.catalog_mut().intern_property("name2")?;
        properties.push((name2, ScalarValue::String("B".into())));
    }
    graph.insert_node(NodeInput {
        id: NodeId(1),
        layer: Layer::Observed,
        revision: 1,
        labels: vec![x],
        properties,
    })?;
    Ok(graph)
}

fn cpu_backend(graph: &GraphStore) -> Result<CpuBackend> {
    let image = ResidentProjectImage::build(
        PROJECT,
        BOOKMARK,
        graph,
        &TemporalStore::default(),
        &IndexCatalog::default(),
    )?;
    let mut backend = CpuBackend::new(128 * 1024 * 1024, 16 * 1024 * 1024);
    backend.admit_project(image)?;
    Ok(backend)
}

fn context<'a>(
    graph: &'a GraphStore,
    backend: Option<&'a dyn ExecutionBackend>,
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
        bookmark: BOOKMARK,
        mutation_revision: graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id: 100,
        next_edge_id: 100,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            write: true,
            require_native_execution: backend.is_some(),
            ..BindCapabilities::default()
        },
        max_result_rows: 64,
        max_batch_rows: 64,
        optimizer_statistics: None,
        backend,
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(30)),
        resolved_query_at_time_nanos: None,
    }
}

fn assert_exact_result(output: &ExecutionOutput, expected: ExpectedResult, label: &str) {
    let [batch] = output.result.batches.as_slice() else {
        panic!("{label}: expected exactly one result batch");
    };
    let [column] = batch.columns.as_slice() else {
        panic!("{label}: expected exactly one result column");
    };
    let [actual] = column.values.as_slice() else {
        panic!("{label}: expected exactly one result value");
    };
    match (expected, actual) {
        (ExpectedResult::Null, ResultValue::Scalar(ScalarValue::Null)) => {}
        (ExpectedResult::Node(properties), ResultValue::Node(node)) => {
            let properties = properties
                .iter()
                .map(|(name, value)| {
                    let value = match value {
                        ExpectedScalar::Integer(value) => ScalarValue::Integer(*value),
                        ExpectedScalar::String(value) => ScalarValue::String((*value).into()),
                    };
                    ((*name).to_owned(), value)
                })
                .collect::<BTreeMap<_, _>>();
            assert_eq!(node.id, NodeId(1), "{label}: result node ID");
            assert_eq!(node.labels, ["X".to_owned()], "{label}: result node labels");
            assert_eq!(node.properties, properties, "{label}: result properties");
        }
        _ => panic!("{label}: unexpected result value {actual:?}"),
    }
}

#[test]
fn strict_cpu_proves_exact_ten_set4_set5_literal_map_cases() -> Result<()> {
    assert_eq!(
        CASES
            .iter()
            .filter(|case| matches!(case.route, Route::Mutation(_)))
            .count(),
        7,
        "the effectful literal-map regression cluster must remain exact"
    );
    assert_eq!(
        CASES
            .iter()
            .map(|case| case.report_index)
            .collect::<Vec<_>>(),
        [845, 846, 847, 848, 849, 850, 851, 852, 853, 854]
    );
    assert_eq!(
        CASES
            .iter()
            .map(|case| (case.feature, case.scenario))
            .collect::<Vec<_>>(),
        [
            ("clauses/set/Set4.feature", 1),
            ("clauses/set/Set4.feature", 2),
            ("clauses/set/Set4.feature", 3),
            ("clauses/set/Set4.feature", 4),
            ("clauses/set/Set4.feature", 5),
            ("clauses/set/Set5.feature", 1),
            ("clauses/set/Set5.feature", 2),
            ("clauses/set/Set5.feature", 3),
            ("clauses/set/Set5.feature", 4),
            ("clauses/set/Set5.feature", 5),
        ]
    );

    for case in CASES {
        let graph = fixture(case.fixture)?;
        let reference = QueryEngine.execute(case.query, &mut context(&graph, None))?;
        let backend = StrictLiteralMapBackend::new(cpu_backend(&graph)?);
        let observations = backend.observations();
        let native = QueryEngine.execute(case.query, &mut context(&graph, Some(&backend)))?;
        let label = format!("{} [{}]", case.feature, case.scenario);

        assert_eq!(
            native.result, reference.result,
            "{} [{}]",
            case.feature, case.scenario
        );
        assert_exact_result(&native, case.expected, &label);
        assert_eq!(
            native.result.statistics.properties_set, case.properties_set,
            "{} [{}]",
            case.feature, case.scenario
        );
        assert_eq!(
            native.graph_mutations.len(),
            reference.graph_mutations.len(),
            "{} [{}]",
            case.feature,
            case.scenario
        );
        assert_eq!(
            observations.pins.load(Ordering::SeqCst),
            1,
            "{} [{}]",
            case.feature,
            case.scenario
        );
        match case.route {
            Route::Mutation(commands) => {
                assert_eq!(observations.mutation_calls.load(Ordering::SeqCst), 1);
                assert_eq!(observations.nullable_calls.load(Ordering::SeqCst), 0);
                assert_eq!(observations.read_calls.load(Ordering::SeqCst), 0);
                assert_eq!(
                    *observations
                        .mutation_command_counts
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner()),
                    [commands]
                );
            }
            Route::NullableNoop => {
                assert_eq!(observations.mutation_calls.load(Ordering::SeqCst), 0);
                assert_eq!(observations.nullable_calls.load(Ordering::SeqCst), 1);
                assert_eq!(observations.read_calls.load(Ordering::SeqCst), 0);
            }
            Route::ReadNoop => {
                assert_eq!(observations.mutation_calls.load(Ordering::SeqCst), 0);
                assert_eq!(observations.nullable_calls.load(Ordering::SeqCst), 0);
                assert_eq!(observations.read_calls.load(Ordering::SeqCst), 1);
            }
        }
        assert_eq!(observations.forbidden_calls.load(Ordering::SeqCst), 0);
    }
    Ok(())
}

#[test]
fn dynamic_document_and_multi_item_map_neighbors_fail_before_dispatch() -> Result<()> {
    let graph = fixture(FixtureShape::NameAndName2)?;
    for query in [
        "MATCH (n:X {name: 'A'}) SET n = {name: n.name, baz: 'C'} RETURN n",
        "MATCH (n:X {name: 'A'}) SET n += {name2: n.name} RETURN n",
        "MATCH (n:X {name: 'A'}) SET n += {name2: ['C']} RETURN n",
        "MATCH (n:X {name: 'A'}) SET n += {name2: 'C'}, n.name = 'B' RETURN n",
    ] {
        QueryEngine.execute(query, &mut context(&graph, None))?;
        let backend = StrictLiteralMapBackend::new(cpu_backend(&graph)?);
        let observations = backend.observations();
        let error = QueryEngine
            .execute(query, &mut context(&graph, Some(&backend)))
            .expect_err("unsupported map profile unexpectedly entered native dispatch");
        assert_eq!(error.code, ErrorCode::GpuAdmissionFailure, "{query}");
        assert_eq!(observations.pins.load(Ordering::SeqCst), 0, "{query}");
        assert_eq!(observations.mutation_calls.load(Ordering::SeqCst), 0);
        assert_eq!(observations.nullable_calls.load(Ordering::SeqCst), 0);
        assert_eq!(observations.read_calls.load(Ordering::SeqCst), 0);
        assert_eq!(observations.forbidden_calls.load(Ordering::SeqCst), 0);
    }
    Ok(())
}
