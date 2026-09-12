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
    Bookmark, EdgeId, Error, ErrorCode, Layer, NodeId, ProjectId, Result, ScalarValue,
    cypher::{
        BindCapabilities, ExecutionContext, QueryEngine, QueryResult, ResultValue, StatementStats,
    },
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend, ResidentDirection,
        ResidentGroup, ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest,
        ResidentNodePipelineRequest, ResidentNodePipelineResult, ResidentProjectImage,
        ResidentSortRequest, ResidentSortResult, ResidentVariablePathRequest,
        ResidentVariablePathResult, ResidentVectorQuery, ResidentVectorResult, ScratchReservation,
    },
    graph::{EdgeInput, GraphStore, IndexCatalog, LayerMask, NodeInput, TemporalStore},
    types::{LabelId, PropertyId},
};
use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::gpu::MetalBackend;
#[cfg(all(feature = "accelerator", target_os = "macos"))]
use std::sync::MutexGuard;

const FEATURE: &str = "features/expressions/pattern/Pattern2.feature";
const MEMORY_LIMIT_BYTES: usize = 512 * 1024 * 1024;
const RESERVED_BYTES: usize = 64 * 1024 * 1024;
const GROUPED_VARIABLE_PATH_SCENARIO: Scenario = Scenario {
    id: 9,
    query: "MATCH (a:A), (b:B) WITH [p = (a)-[*]->(b) | p] AS paths, count(a) AS c RETURN paths, c",
};
const NESTED_NODE_COUNT_SCENARIO: Scenario = Scenario {
    id: 7,
    query: "MATCH p = (n:X)-->() RETURN n, [x IN nodes(p) | size([(x)-->(:Y) | 1])] AS list",
};
const ORDERED_PARENT_LIST_SCENARIO: Scenario = Scenario {
    id: 11,
    query: "MATCH (liker) RETURN [p = (liker)--() | p] AS isNew ORDER BY liker.time",
};

#[derive(Clone, Copy, Debug)]
struct Scenario {
    id: u8,
    query: &'static str,
}

const SCENARIOS: &[Scenario] = &[
    Scenario {
        id: 1,
        query: "MATCH (n) RETURN [p = (n)-->() | p] AS list",
    },
    Scenario {
        id: 2,
        query: "MATCH (n:A) RETURN [p = (n)-->(:B) | p] AS list",
    },
    Scenario {
        id: 3,
        query: "MATCH (a:A), (b:B) RETURN [p = (a)-->(b) | p] AS list",
    },
    Scenario {
        id: 4,
        query: "MATCH (n) RETURN [(n)-[:T]->(b) | b.name] AS list",
    },
    Scenario {
        id: 5,
        query: "MATCH (n) RETURN [(n)-[r:T]->() | r.name] AS list",
    },
    Scenario {
        id: 8,
        query: "MATCH (n)-->(b) WITH [p = (n)-->() | p] AS ps, count(b) AS c RETURN ps, c",
    },
    Scenario {
        id: 10,
        query: "MATCH (n:A) RETURN [p = (n)-[:HAS]->() | p] AS ps",
    },
];

struct Fixture {
    graph: GraphStore,
    project: ProjectId,
    bookmark: Bookmark,
}

fn fixture() -> Result<Fixture> {
    let mut graph = GraphStore::default();
    let a = graph.catalog_mut().intern_label("A")?;
    let b = graph.catalog_mut().intern_label("B")?;
    let t = graph.catalog_mut().intern_relationship_type("T")?;
    let has = graph.catalog_mut().intern_relationship_type("HAS")?;
    let name = graph.catalog_mut().intern_property("name")?;

    for (id, labels, value) in [
        (NodeId(1), vec![a], Some("one")),
        (NodeId(2), vec![b], Some("two")),
        (NodeId(3), vec![a, b], None),
        (NodeId(4), Vec::new(), Some("four")),
        (NodeId(5), Vec::new(), Some("isolated")),
        (NodeId(6), vec![a], Some("isolated-a")),
    ] {
        graph.insert_node(NodeInput {
            id,
            layer: Layer::Observed,
            revision: id.0,
            labels,
            properties: value
                .map(|value| vec![(name, ScalarValue::String(value.into()))])
                .unwrap_or_default(),
        })?;
    }
    for (id, source, target, relationship_type, value) in [
        (EdgeId(10), NodeId(1), NodeId(2), t, Some("t12")),
        (EdgeId(11), NodeId(1), NodeId(3), has, Some("h13")),
        (EdgeId(12), NodeId(2), NodeId(4), t, None),
        (EdgeId(13), NodeId(3), NodeId(2), has, Some("h32")),
        (EdgeId(14), NodeId(4), NodeId(3), t, Some("t43")),
    ] {
        graph.insert_edge(EdgeInput {
            id,
            source,
            target,
            relationship_type,
            layer: Layer::Observed,
            revision: id.0,
            properties: value
                .map(|value| vec![(name, ScalarValue::String(value.into()))])
                .unwrap_or_default(),
        })?;
    }

    Ok(Fixture {
        graph,
        project: ProjectId(uuid::Uuid::from_u128(0x5041_5454_4552_4e32_4c49_5354_0001)),
        bookmark: Bookmark {
            term: 29,
            index: 20_001,
        },
    })
}

fn exact_tck_pattern2_08_fixture() -> Result<Fixture> {
    let mut graph = GraphStore::default();
    let a = graph.catalog_mut().intern_label("A")?;
    let b = graph.catalog_mut().intern_label("B")?;
    let c = graph.catalog_mut().intern_label("C")?;
    let t = graph.catalog_mut().intern_relationship_type("T")?;
    for (id, label) in [(NodeId(1), a), (NodeId(2), b), (NodeId(3), c)] {
        graph.insert_node(NodeInput {
            id,
            layer: Layer::Observed,
            revision: id.0,
            labels: vec![label],
            properties: Vec::new(),
        })?;
    }
    for (id, source, target) in [
        (EdgeId(10), NodeId(1), NodeId(2)),
        (EdgeId(11), NodeId(2), NodeId(3)),
    ] {
        graph.insert_edge(EdgeInput {
            id,
            source,
            target,
            relationship_type: t,
            layer: Layer::Observed,
            revision: id.0,
            properties: Vec::new(),
        })?;
    }
    Ok(Fixture {
        graph,
        project: ProjectId(uuid::Uuid::from_u128(0x5041_5454_4552_4e32_5443_4b30_3801)),
        bookmark: Bookmark {
            term: 29,
            index: 20_008,
        },
    })
}

fn exact_tck_pattern2_09_fixture() -> Result<Fixture> {
    let mut graph = GraphStore::default();
    let a = graph.catalog_mut().intern_label("A")?;
    let b = graph.catalog_mut().intern_label("B")?;
    let relationship_type = graph.catalog_mut().intern_relationship_type("T")?;
    for (id, label) in [(NodeId(1), a), (NodeId(2), b)] {
        graph.insert_node(NodeInput {
            id,
            layer: Layer::Observed,
            revision: id.0,
            labels: vec![label],
            properties: Vec::new(),
        })?;
    }
    graph.insert_edge(EdgeInput {
        id: EdgeId(10),
        source: NodeId(1),
        target: NodeId(2),
        relationship_type,
        layer: Layer::Observed,
        revision: 10,
        properties: Vec::new(),
    })?;
    Ok(Fixture {
        graph,
        project: ProjectId(uuid::Uuid::from_u128(0x5041_5454_4552_4e32_5443_4b30_3901)),
        bookmark: Bookmark {
            term: 29,
            index: 20_009,
        },
    })
}

fn exact_tck_pattern2_07_fixture() -> Result<Fixture> {
    let mut graph = GraphStore::default();
    let x = graph.catalog_mut().intern_label("X")?;
    let y = graph.catalog_mut().intern_label("Y")?;
    let relationship_type = graph.catalog_mut().intern_relationship_type("T")?;
    for (id, labels) in [
        (NodeId(1), vec![x]),
        (NodeId(2), vec![y]),
        (NodeId(3), vec![x]),
        (NodeId(4), Vec::new()),
        (NodeId(5), vec![y]),
        (NodeId(6), vec![y]),
        (NodeId(7), vec![y]),
    ] {
        graph.insert_node(NodeInput {
            id,
            layer: Layer::Observed,
            revision: id.0,
            labels,
            properties: Vec::new(),
        })?;
    }
    for (id, source, target) in [
        (EdgeId(10), NodeId(1), NodeId(2)),
        (EdgeId(11), NodeId(3), NodeId(4)),
        (EdgeId(12), NodeId(2), NodeId(5)),
        (EdgeId(13), NodeId(2), NodeId(6)),
        (EdgeId(14), NodeId(4), NodeId(7)),
    ] {
        graph.insert_edge(EdgeInput {
            id,
            source,
            target,
            relationship_type,
            layer: Layer::Observed,
            revision: id.0,
            properties: Vec::new(),
        })?;
    }
    Ok(Fixture {
        graph,
        project: ProjectId(uuid::Uuid::from_u128(0x5041_5454_4552_4e32_5443_4b30_3701)),
        bookmark: Bookmark {
            term: 29,
            index: 20_007,
        },
    })
}

fn exact_tck_pattern2_11_fixture() -> Result<Fixture> {
    let mut graph = GraphStore::default();
    let relationship_type = graph.catalog_mut().intern_relationship_type("T")?;
    let time = graph.catalog_mut().intern_property("time")?;
    for (id, value) in [(NodeId(1), 20), (NodeId(2), 10)] {
        graph.insert_node(NodeInput {
            id,
            layer: Layer::Observed,
            revision: id.0,
            labels: Vec::new(),
            properties: vec![(time, ScalarValue::Integer(value))],
        })?;
    }
    graph.insert_edge(EdgeInput {
        id: EdgeId(10),
        source: NodeId(1),
        target: NodeId(2),
        relationship_type,
        layer: Layer::Observed,
        revision: 10,
        properties: Vec::new(),
    })?;
    Ok(Fixture {
        graph,
        project: ProjectId(uuid::Uuid::from_u128(0x5041_5454_4552_4e32_5443_4b31_3101)),
        bookmark: Bookmark {
            term: 29,
            index: 20_011,
        },
    })
}

fn resident_image(fixture: &Fixture) -> Result<ResidentProjectImage> {
    ResidentProjectImage::build(
        fixture.project,
        fixture.bookmark,
        &fixture.graph,
        &TemporalStore::default(),
        &IndexCatalog::default(),
    )
}

fn cpu_backend(fixture: &Fixture) -> Result<CpuBackend> {
    let mut backend = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
    backend.admit_project(resident_image(fixture)?)?;
    Ok(backend)
}

fn context<'a>(
    fixture: &'a Fixture,
    backend: Option<&'a dyn ExecutionBackend>,
    require_native_execution: bool,
) -> ExecutionContext<'a> {
    ExecutionContext {
        project_id: fixture.project,
        graph: &fixture.graph,
        binding_catalog: fixture.graph.catalog(),
        prior_graph_mutations: &[],
        temporal: None,
        prior_temporal_mutations: &[],
        vector_indexes: BTreeMap::new(),
        scalar_indexes: None,
        text_embedding: None,
        parameters: BTreeMap::new(),
        bookmark: fixture.bookmark,
        mutation_revision: fixture.graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id: 10_000,
        next_edge_id: 20_000,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            require_native_execution,
            ..BindCapabilities::default()
        },
        max_result_rows: 4_096,
        max_batch_rows: 3,
        optimizer_statistics: None,
        backend,
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(30)),
        resolved_query_at_time_nanos: None,
    }
}

#[derive(Default)]
struct RouteObservations {
    pins: AtomicUsize,
    variable_paths: AtomicUsize,
    scans: AtomicUsize,
    adjacency: AtomicUsize,
    node_pipelines: AtomicUsize,
    requests: Mutex<Vec<ResidentVariablePathRequest>>,
}

struct ObservedBackend {
    inner: Box<dyn ExecutionBackend>,
    reported_kind: BackendKind,
    observations: Arc<RouteObservations>,
}

impl ObservedBackend {
    fn new(inner: impl ExecutionBackend + 'static) -> Self {
        let reported_kind = inner.kind();
        Self {
            inner: Box::new(inner),
            reported_kind,
            observations: Arc::new(RouteObservations::default()),
        }
    }

    fn reporting(inner: impl ExecutionBackend + 'static, reported_kind: BackendKind) -> Self {
        Self {
            inner: Box::new(inner),
            reported_kind,
            observations: Arc::new(RouteObservations::default()),
        }
    }

    fn observations(&self) -> Arc<RouteObservations> {
        Arc::clone(&self.observations)
    }
}

impl ExecutionBackend for ObservedBackend {
    fn kind(&self) -> BackendKind {
        self.reported_kind
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
        self.observations.pins.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Self {
            inner: self.inner.pin_project(project)?,
            reported_kind: self.reported_kind,
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
        project: ProjectId,
        label: Option<LabelId>,
        layers: LayerMask,
        cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.observations.scans.fetch_add(1, Ordering::SeqCst);
        self.inner.scan_nodes(project, label, layers, cancellation)
    }

    fn filter_node_i64(
        &self,
        project: ProjectId,
        property: PropertyId,
        operation: CompareOp,
        operand: i64,
        cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.inner
            .filter_node_i64(project, property, operation, operand, cancellation)
    }

    fn expand_project_out(
        &self,
        project: ProjectId,
        sources: &[u32],
        cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.observations.adjacency.fetch_add(1, Ordering::SeqCst);
        self.inner
            .expand_project_out(project, sources, cancellation)
    }

    fn expand_project_in(
        &self,
        project: ProjectId,
        targets: &[u32],
        cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.observations.adjacency.fetch_add(1, Ordering::SeqCst);
        self.inner.expand_project_in(project, targets, cancellation)
    }

    fn search_vectors(
        &self,
        request: &ResidentVectorQuery,
        cancellation: &CancellationToken,
    ) -> Result<ResidentVectorResult> {
        self.inner.search_vectors(request, cancellation)
    }

    fn sort_rows(
        &self,
        request: &ResidentSortRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentSortResult> {
        self.inner.sort_rows(request, cancellation)
    }

    fn join_node_i64(
        &self,
        request: &ResidentJoinRequest,
        cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentJoinPair>> {
        self.inner.join_node_i64(request, cancellation)
    }

    fn group_node_i64(
        &self,
        request: &ResidentGroupRequest,
        cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentGroup>> {
        self.inner.group_node_i64(request, cancellation)
    }

    fn execute_node_pipeline(
        &self,
        request: &ResidentNodePipelineRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentNodePipelineResult> {
        self.observations
            .node_pipelines
            .fetch_add(1, Ordering::SeqCst);
        self.inner.execute_node_pipeline(request, cancellation)
    }

    fn supports_native_variable_path(&self) -> bool {
        self.inner.supports_native_variable_path()
    }

    fn execute_variable_path(
        &self,
        request: &ResidentVariablePathRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentVariablePathResult> {
        self.observations
            .variable_paths
            .fetch_add(1, Ordering::SeqCst);
        self.observations
            .requests
            .lock()
            .expect("pattern-comprehension request observations poisoned")
            .push(request.clone());
        self.inner.execute_variable_path(request, cancellation)
    }

    fn filter_i64(
        &self,
        values: &[i64],
        validity: &[bool],
        operation: CompareOp,
        operand: i64,
        cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.inner
            .filter_i64(values, validity, operation, operand, cancellation)
    }

    fn expand_out(
        &self,
        sources: &[u32],
        cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.observations.adjacency.fetch_add(1, Ordering::SeqCst);
        self.inner.expand_out(sources, cancellation)
    }

    fn exact_l2(
        &self,
        matrix: &[f32],
        rows: usize,
        dimension: usize,
        query: &[f32],
        cancellation: &CancellationToken,
    ) -> Result<DistanceBatch> {
        self.inner
            .exact_l2(matrix, rows, dimension, query, cancellation)
    }
}

fn execute(
    fixture: &Fixture,
    backend: Option<&dyn ExecutionBackend>,
    require_native_execution: bool,
    query: &str,
) -> Result<irongraph::cypher::ExecutionOutput> {
    QueryEngine.execute(
        query,
        &mut context(fixture, backend, require_native_execution),
    )
}

fn stable_rows(result: &QueryResult) -> Vec<String> {
    let mut rows = Vec::new();
    for batch in &result.batches {
        for index in 0..batch.row_count {
            rows.push(format!(
                "{:?}",
                batch
                    .columns
                    .iter()
                    .map(|column| (&column.name, &column.values[index]))
                    .collect::<Vec<_>>()
            ));
        }
    }
    rows.sort_unstable();
    rows
}

fn result_has_empty_list(result: &QueryResult) -> bool {
    result.batches.iter().any(|batch| {
        batch.columns.iter().any(|column| {
            column
                .values
                .iter()
                .any(|value| matches!(value, ResultValue::List(values) if values.is_empty()))
        })
    })
}

fn value_contains_null(value: &ResultValue) -> bool {
    match value {
        ResultValue::Scalar(ScalarValue::Null) => true,
        ResultValue::List(values) => values.iter().any(value_contains_null),
        ResultValue::Map(values) => values.values().any(value_contains_null),
        _ => false,
    }
}

fn assert_sealed_route(observations: &RouteObservations, scenario: Scenario) {
    assert_eq!(
        observations.pins.load(Ordering::SeqCst),
        1,
        "{FEATURE} [{}] pin count",
        scenario.id
    );
    assert_eq!(
        observations.variable_paths.load(Ordering::SeqCst),
        1,
        "{FEATURE} [{}] variable-path dispatch count",
        scenario.id
    );
    assert_eq!(
        observations.scans.load(Ordering::SeqCst),
        0,
        "{FEATURE} [{}] host-visible scan",
        scenario.id
    );
    assert_eq!(
        observations.adjacency.load(Ordering::SeqCst),
        0,
        "{FEATURE} [{}] Rust-driven adjacency",
        scenario.id
    );
    assert_eq!(
        observations.node_pipelines.load(Ordering::SeqCst),
        0,
        "{FEATURE} [{}] unrelated node pipeline",
        scenario.id
    );
}

#[test]
fn manifest_is_exactly_the_seven_pattern2_first_tranche_scenarios() {
    assert_eq!(FEATURE, "features/expressions/pattern/Pattern2.feature");
    assert_eq!(
        SCENARIOS
            .iter()
            .map(|scenario| scenario.id)
            .collect::<Vec<_>>(),
        [1, 2, 3, 4, 5, 8, 10]
    );
}

#[test]
fn strict_cpu_matches_the_generic_oracle_through_one_receipted_path_command() -> Result<()> {
    let fixture = fixture()?;
    let mut saw_empty_list = false;
    let mut saw_null_list_item = false;

    for scenario in SCENARIOS {
        let oracle = execute(&fixture, None, false, scenario.query)?;
        let backend = ObservedBackend::new(cpu_backend(&fixture)?);
        let observations = backend.observations();
        let native = execute(&fixture, Some(&backend), true, scenario.query)?;

        assert_eq!(
            native.result.schema, oracle.result.schema,
            "scenario {}",
            scenario.id
        );
        assert_eq!(
            stable_rows(&native.result),
            stable_rows(&oracle.result),
            "{FEATURE} [{}] native result diverged from the generic CPU oracle",
            scenario.id
        );
        assert_eq!(native.result.bookmark, fixture.bookmark);
        assert_eq!(native.result.statistics, StatementStats::default());
        assert!(!native.result.truncated);
        assert!(native.graph_mutations.is_empty());
        assert!(native.temporal_mutations.is_empty());
        assert_sealed_route(&observations, *scenario);

        let requests = observations
            .requests
            .lock()
            .expect("pattern-comprehension request observations poisoned");
        let [request] = requests.as_slice() else {
            return Err(Error::internal(
                "sealed pattern-comprehension route did not capture exactly one request",
            ));
        };
        request.validate()?;
        assert_eq!(request.project, fixture.project);
        assert_eq!(request.expected_bookmark, fixture.bookmark);
        assert_eq!(request.expected_graph_revision, fixture.graph.revision());
        assert_eq!(
            request.expected_layout_version,
            fixture.graph.layout_version()
        );
        assert_eq!(request.expected_node_slots, fixture.graph.node_slot_count());
        assert_eq!(request.expected_edge_slots, fixture.graph.edge_slot_count());
        assert_ne!((request.execution.high, request.execution.low), (0, 0));
        assert!(request.output_limit.is_none());
        let [segment] = request.segments.as_slice() else {
            return Err(Error::internal(
                "Pattern2 first tranche did not compile to one traversal segment",
            ));
        };
        assert_eq!(segment.direction, ResidentDirection::Outgoing);
        assert_eq!((segment.minimum_hops, segment.maximum_hops), (1, Some(1)));
        if scenario.id == 8 {
            assert!(!request.optional);
            assert!(request.bound_terminal_scan.is_none());
        } else {
            assert!(request.optional);
            assert_eq!(request.bound_terminal_scan.is_some(), scenario.id == 3);
        }

        saw_empty_list |= result_has_empty_list(&native.result);
        saw_null_list_item |= native.result.batches.iter().any(|batch| {
            batch
                .columns
                .iter()
                .flat_map(|column| &column.values)
                .any(value_contains_null)
        });
    }
    assert!(
        saw_empty_list,
        "OPTIONAL parent coverage never produced an empty list"
    );
    assert!(
        saw_null_list_item,
        "a matched missing property was not preserved as a null list item"
    );
    Ok(())
}

#[test]
fn exact_tck_08_enters_the_sealed_route_under_metal_planning() -> Result<()> {
    let fixture = exact_tck_pattern2_08_fixture()?;
    let scenario = SCENARIOS
        .iter()
        .find(|scenario| scenario.id == 8)
        .copied()
        .ok_or_else(|| Error::internal("Pattern2 [8] is absent from the focused manifest"))?;

    let oracle = execute(&fixture, None, false, scenario.query)?;
    let cpu = ObservedBackend::new(cpu_backend(&fixture)?);
    let cpu_observations = cpu.observations();
    let native = execute(&fixture, Some(&cpu), true, scenario.query)?;
    assert_eq!(native.result.schema, oracle.result.schema);
    assert_eq!(stable_rows(&native.result), stable_rows(&oracle.result));
    assert_sealed_route(&cpu_observations, scenario);

    let backend = ObservedBackend::reporting(cpu_backend(&fixture)?, BackendKind::Metal);
    let observations = backend.observations();
    let error = execute(&fixture, Some(&backend), true, scenario.query)
        .expect_err("CPU completion unexpectedly proved a Metal variable-path command");
    assert_eq!(
        observations.variable_paths.load(Ordering::SeqCst),
        1,
        "the exact TCK fixture did not enter the variable-path command"
    );
    assert_eq!(observations.pins.load(Ordering::SeqCst), 1);
    assert_eq!(observations.scans.load(Ordering::SeqCst), 0);
    assert_eq!(observations.adjacency.load(Ordering::SeqCst), 0);
    assert_eq!(
        error.code,
        ErrorCode::CorruptStorage,
        "only the deliberately mismatched CPU-vs-Metal completion proof may fail"
    );
    Ok(())
}

#[test]
fn strict_cpu_certifies_remaining_pattern2_09_grouped_variable_path_comprehension() -> Result<()> {
    let fixture = exact_tck_pattern2_09_fixture()?;
    let scenario = GROUPED_VARIABLE_PATH_SCENARIO;
    let oracle = execute(&fixture, None, false, scenario.query)?;
    let backend = ObservedBackend::new(cpu_backend(&fixture)?);
    let observations = backend.observations();
    let native = execute(&fixture, Some(&backend), true, scenario.query)?;

    assert_eq!(native.result.schema, oracle.result.schema);
    assert_eq!(stable_rows(&native.result), stable_rows(&oracle.result));
    assert_eq!(native.result.bookmark, fixture.bookmark);
    assert_eq!(native.result.statistics, StatementStats::default());
    assert!(!native.result.truncated);
    assert_sealed_route(&observations, scenario);

    let requests = observations
        .requests
        .lock()
        .expect("Pattern2 [9] request observations poisoned");
    let [request] = requests.as_slice() else {
        return Err(Error::internal(
            "Pattern2 [9] did not issue exactly one variable-path request",
        ));
    };
    request.validate()?;
    assert!(request.optional);
    assert!(request.bound_terminal_scan.is_some());
    assert!(matches!(
        request.final_projection,
        irongraph::gpu::ResidentVariablePathFinalProjection::GroupedParentPathLists { .. }
    ));
    let [segment] = request.segments.as_slice() else {
        return Err(Error::internal("Pattern2 [9] did not issue one segment"));
    };
    assert_eq!(segment.direction, ResidentDirection::Outgoing);
    assert_eq!((segment.minimum_hops, segment.maximum_hops), (1, None));
    assert!(matches!(
        native.result.batches.as_slice(),
        [batch]
            if batch.row_count == 1
                && matches!(batch.columns.as_slice(), [paths, count]
                    if matches!(paths.values.as_slice(), [ResultValue::List(values)] if values.len() == 1)
                        && count.values == [ResultValue::Scalar(ScalarValue::Integer(1))])
    ));
    Ok(())
}

#[test]
fn strict_cpu_certifies_pattern2_07_and_11_sealed_reductions() -> Result<()> {
    for (fixture, scenario) in [
        (exact_tck_pattern2_07_fixture()?, NESTED_NODE_COUNT_SCENARIO),
        (
            exact_tck_pattern2_11_fixture()?,
            ORDERED_PARENT_LIST_SCENARIO,
        ),
    ] {
        let oracle = execute(&fixture, None, false, scenario.query)?;
        let backend = ObservedBackend::new(cpu_backend(&fixture)?);
        let observations = backend.observations();
        let native = execute(&fixture, Some(&backend), true, scenario.query)?;

        assert_eq!(native.result, oracle.result, "Pattern2 [{}]", scenario.id);
        assert!(native.graph_mutations.is_empty());
        assert!(native.temporal_mutations.is_empty());
        assert_sealed_route(&observations, scenario);

        let requests = observations
            .requests
            .lock()
            .expect("Pattern2 sealed-reduction observations poisoned");
        let [request] = requests.as_slice() else {
            return Err(Error::internal(
                "Pattern2 sealed reduction did not issue exactly one path request",
            ));
        };
        request.validate()?;
        match scenario.id {
            7 => assert!(matches!(
                request.final_projection,
                irongraph::gpu::ResidentVariablePathFinalProjection::PathNodeOutgoingLabelCounts { .. }
            )),
            11 => assert!(matches!(
                request.final_projection,
                irongraph::gpu::ResidentVariablePathFinalProjection::OrderedParentPathLists { .. }
            )),
            _ => return Err(Error::internal("unexpected Pattern2 sealed scenario")),
        }
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn metal_test_guard() -> MutexGuard<'static, ()> {
    static METAL_TEST: Mutex<()> = Mutex::new(());
    METAL_TEST
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "hardware acceptance gate: Pattern2 [7]/[11] require v6 Metal sealed reductions"]
fn real_metal_certifies_pattern2_07_and_11_sealed_reductions() -> Result<()> {
    let _guard = metal_test_guard();
    for (fixture, scenario) in [
        (exact_tck_pattern2_07_fixture()?, NESTED_NODE_COUNT_SCENARIO),
        (
            exact_tck_pattern2_11_fixture()?,
            ORDERED_PARENT_LIST_SCENARIO,
        ),
    ] {
        let oracle = execute(&fixture, None, false, scenario.query)?;
        let backend = ObservedBackend::new({
            let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
            metal.admit_project(resident_image(&fixture)?)?;
            metal
        });
        let observations = backend.observations();
        let actual = execute(&fixture, Some(&backend), true, scenario.query)?;
        assert_eq!(actual.result, oracle.result, "Pattern2 [{}]", scenario.id);
        assert_sealed_route(&observations, scenario);
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "hardware acceptance gate: Pattern2 [9] requires a backend-authored grouped path-list relation"]
fn real_metal_certifies_pattern2_09_sealed_grouped_path_lists() -> Result<()> {
    let _guard = metal_test_guard();
    let fixture = exact_tck_pattern2_09_fixture()?;
    let oracle = execute(&fixture, None, false, GROUPED_VARIABLE_PATH_SCENARIO.query)?;
    let backend = ObservedBackend::new({
        let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
        metal.admit_project(resident_image(&fixture)?)?;
        metal
    });
    let observations = backend.observations();
    let actual = execute(
        &fixture,
        Some(&backend),
        true,
        GROUPED_VARIABLE_PATH_SCENARIO.query,
    )?;
    assert_eq!(actual.result.schema, oracle.result.schema);
    assert_eq!(stable_rows(&actual.result), stable_rows(&oracle.result));
    assert_sealed_route(&observations, GROUPED_VARIABLE_PATH_SCENARIO);
    Ok(())
}

#[test]
fn neighboring_pattern_comprehensions_fail_closed_without_any_backend_work() -> Result<()> {
    let fixture = fixture()?;
    for query in [
        "MATCH (n) RETURN [p = (n)<--() | p] AS list",
        "MATCH (n) RETURN [p = (n)-[*1..2]->() | p] AS list",
        "MATCH (n) RETURN [p = (n)-->() WHERE true | p] AS list",
        "MATCH (n) RETURN [p = (n)-->() | p] AS list, n",
        "MATCH (a:A), (b:B) WITH [p = (a)-[*]->(b) | p] AS paths, count(b) AS c RETURN paths, c",
    ] {
        let backend = ObservedBackend::reporting(cpu_backend(&fixture)?, BackendKind::Metal);
        let observations = backend.observations();
        let error = execute(&fixture, Some(&backend), true, query)
            .expect_err("unsupported strict-Metal pattern comprehension executed");
        assert_eq!(error.code, ErrorCode::GpuAdmissionFailure, "query: {query}");
        assert_eq!(
            observations.pins.load(Ordering::SeqCst),
            0,
            "query: {query}"
        );
        assert_eq!(
            observations.variable_paths.load(Ordering::SeqCst),
            0,
            "query: {query}"
        );
        assert_eq!(
            observations.scans.load(Ordering::SeqCst),
            0,
            "query: {query}"
        );
        assert_eq!(
            observations.adjacency.load(Ordering::SeqCst),
            0,
            "query: {query}"
        );
    }
    Ok(())
}

#[test]
fn stale_resident_generation_is_rejected_before_variable_path_dispatch() -> Result<()> {
    let mut fixture = fixture()?;
    let backend = ObservedBackend::new(cpu_backend(&fixture)?);
    let observations = backend.observations();
    fixture.graph.insert_node(NodeInput {
        id: NodeId(99),
        layer: Layer::Observed,
        revision: 99,
        labels: Vec::new(),
        properties: Vec::new(),
    })?;

    let error = execute(&fixture, Some(&backend), true, SCENARIOS[0].query)
        .expect_err("stale native pattern-comprehension generation executed");
    assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
    assert_eq!(observations.pins.load(Ordering::SeqCst), 1);
    assert_eq!(observations.variable_paths.load(Ordering::SeqCst), 0);
    assert_eq!(observations.scans.load(Ordering::SeqCst), 0);
    assert_eq!(observations.adjacency.load(Ordering::SeqCst), 0);
    Ok(())
}
