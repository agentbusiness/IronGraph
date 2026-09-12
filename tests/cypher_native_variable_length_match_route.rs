// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use std::sync::{Mutex, MutexGuard};

use irongraph::{
    Bookmark, EdgeId, Error, ErrorCode, Layer, NodeId, ProjectId, Result, ScalarValue,
    cypher::{
        BindCapabilities, ExecutionContext, ExecutionOutput, ExecutionStreamItem, QueryEngine,
        ResultValue, StatementStats,
    },
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend, ResidentGroup,
        ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest, ResidentNodePipelineRequest,
        ResidentNodePipelineResult, ResidentProjectImage, ResidentSortRequest, ResidentSortResult,
        ResidentVariablePathRequest, ResidentVariablePathResult, ResidentVectorQuery,
        ResidentVectorResult, ScratchReservation,
    },
    graph::{EdgeInput, GraphStore, IndexCatalog, LayerMask, NodeInput, TemporalStore},
    types::{LabelId, PropertyId},
};
use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::gpu::MetalBackend;

const FEATURE: &str = "features/clauses/match/Match5.feature";
const MEMORY_LIMIT_BYTES: usize = 512 * 1024 * 1024;
const RESERVED_BYTES: usize = 64 * 1024 * 1024;

const DEPTH_0: &[&str] = &["n0"];
const DEPTH_1: &[&str] = &["n00", "n01"];
const DEPTH_2: &[&str] = &["n000", "n001", "n010", "n011"];
const DEPTH_3: &[&str] = &[
    "n0000", "n0001", "n0010", "n0011", "n0100", "n0101", "n0110", "n0111",
];
const DEPTH_4: &[&str] = &[
    "n00000", "n00001", "n00010", "n00011", "n00100", "n00101", "n00110", "n00111", "n01000",
    "n01001", "n01010", "n01011", "n01100", "n01101", "n01110", "n01111",
];
const DEPTHS_0_TO_2: &[&str] = &["n0", "n00", "n01", "n000", "n001", "n010", "n011"];
const DEPTHS_1_TO_2: &[&str] = &["n00", "n01", "n000", "n001", "n010", "n011"];
const DEPTHS_0_TO_3: &[&str] = &[
    "n0", "n00", "n01", "n000", "n001", "n010", "n011", "n0000", "n0001", "n0010", "n0011",
    "n0100", "n0101", "n0110", "n0111",
];
const DEPTHS_1_TO_3: &[&str] = &[
    "n00", "n01", "n000", "n001", "n010", "n011", "n0000", "n0001", "n0010", "n0011", "n0100",
    "n0101", "n0110", "n0111",
];
const DEPTHS_2_TO_3: &[&str] = &[
    "n000", "n001", "n010", "n011", "n0000", "n0001", "n0010", "n0011", "n0100", "n0101", "n0110",
    "n0111",
];
const EMPTY: &[&str] = &[];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GraphShape {
    Baseline,
    Extended,
    ReversedTopExtended,
    ReversedBelowTopExtended,
}

impl GraphShape {
    const fn maximum_depth(self) -> usize {
        match self {
            Self::Baseline => 3,
            Self::Extended | Self::ReversedTopExtended | Self::ReversedBelowTopExtended => 4,
        }
    }

    const fn project_tag(self) -> u128 {
        match self {
            Self::Baseline => 1,
            Self::Extended => 2,
            Self::ReversedTopExtended => 3,
            Self::ReversedBelowTopExtended => 4,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Scenario {
    id: u8,
    shape: GraphShape,
    query: &'static str,
    expected_names: &'static [&'static str],
}

const SCENARIOS: &[Scenario] = &[
    Scenario {
        id: 1,
        shape: GraphShape::Baseline,
        query: "MATCH (a:A) MATCH (a)-[:LIKES*]->(c) RETURN c.name",
        expected_names: DEPTHS_1_TO_3,
    },
    Scenario {
        id: 2,
        shape: GraphShape::Baseline,
        query: "MATCH (a:A) MATCH (a)-[:LIKES*..]->(c) RETURN c.name",
        expected_names: DEPTHS_1_TO_3,
    },
    Scenario {
        id: 3,
        shape: GraphShape::Baseline,
        query: "MATCH (a:A) MATCH (a)-[:LIKES*0]->(c) RETURN c.name",
        expected_names: DEPTH_0,
    },
    Scenario {
        id: 4,
        shape: GraphShape::Baseline,
        query: "MATCH (a:A) MATCH (a)-[:LIKES*1]->(c) RETURN c.name",
        expected_names: DEPTH_1,
    },
    Scenario {
        id: 5,
        shape: GraphShape::Baseline,
        query: "MATCH (a:A) MATCH (a)-[:LIKES*2]->(c) RETURN c.name",
        expected_names: DEPTH_2,
    },
    Scenario {
        id: 6,
        shape: GraphShape::Baseline,
        query: "MATCH (a:A) MATCH (a)-[:LIKES*0..2]->(c) RETURN c.name",
        expected_names: DEPTHS_0_TO_2,
    },
    Scenario {
        id: 7,
        shape: GraphShape::Baseline,
        query: "MATCH (a:A) MATCH (a)-[:LIKES*1..2]->(c) RETURN c.name",
        expected_names: DEPTHS_1_TO_2,
    },
    Scenario {
        id: 8,
        shape: GraphShape::Baseline,
        query: "MATCH (a:A) MATCH (a)-[:LIKES*0..0]->(c) RETURN c.name",
        expected_names: DEPTH_0,
    },
    Scenario {
        id: 9,
        shape: GraphShape::Baseline,
        query: "MATCH (a:A) MATCH (a)-[:LIKES*1..1]->(c) RETURN c.name",
        expected_names: DEPTH_1,
    },
    Scenario {
        id: 10,
        shape: GraphShape::Baseline,
        query: "MATCH (a:A) MATCH (a)-[:LIKES*2..2]->(c) RETURN c.name",
        expected_names: DEPTH_2,
    },
    Scenario {
        id: 11,
        shape: GraphShape::Baseline,
        query: "MATCH (a:A) MATCH (a)-[:LIKES*2..1]->(c) RETURN c.name",
        expected_names: EMPTY,
    },
    Scenario {
        id: 12,
        shape: GraphShape::Baseline,
        query: "MATCH (a:A) MATCH (a)-[:LIKES*1..0]->(c) RETURN c.name",
        expected_names: EMPTY,
    },
    Scenario {
        id: 13,
        shape: GraphShape::Baseline,
        query: "MATCH (a:A) MATCH (a)-[:LIKES*..0]->(c) RETURN c.name",
        expected_names: EMPTY,
    },
    Scenario {
        id: 14,
        shape: GraphShape::Baseline,
        query: "MATCH (a:A) MATCH (a)-[:LIKES*..1]->(c) RETURN c.name",
        expected_names: DEPTH_1,
    },
    Scenario {
        id: 15,
        shape: GraphShape::Baseline,
        query: "MATCH (a:A) MATCH (a)-[:LIKES*..2]->(c) RETURN c.name",
        expected_names: DEPTHS_1_TO_2,
    },
    Scenario {
        id: 16,
        shape: GraphShape::Baseline,
        query: "MATCH (a:A) MATCH (a)-[:LIKES*0..]->(c) RETURN c.name",
        expected_names: DEPTHS_0_TO_3,
    },
    Scenario {
        id: 17,
        shape: GraphShape::Baseline,
        query: "MATCH (a:A) MATCH (a)-[:LIKES*1..]->(c) RETURN c.name",
        expected_names: DEPTHS_1_TO_3,
    },
    Scenario {
        id: 18,
        shape: GraphShape::Baseline,
        query: "MATCH (a:A) MATCH (a)-[:LIKES*2..]->(c) RETURN c.name",
        expected_names: DEPTHS_2_TO_3,
    },
    Scenario {
        id: 19,
        shape: GraphShape::Baseline,
        query: "MATCH (a:A) MATCH (a)-[:LIKES*0]->()-[:LIKES]->(c) RETURN c.name",
        expected_names: DEPTH_1,
    },
    Scenario {
        id: 20,
        shape: GraphShape::Baseline,
        query: "MATCH (a:A) MATCH (a)-[:LIKES]->()-[:LIKES*0]->(c) RETURN c.name",
        expected_names: DEPTH_1,
    },
    Scenario {
        id: 21,
        shape: GraphShape::Baseline,
        query: "MATCH (a:A) MATCH (a)-[:LIKES*1]->()-[:LIKES]->(c) RETURN c.name",
        expected_names: DEPTH_2,
    },
    Scenario {
        id: 22,
        shape: GraphShape::Baseline,
        query: "MATCH (a:A) MATCH (a)-[:LIKES]->()-[:LIKES*1]->(c) RETURN c.name",
        expected_names: DEPTH_2,
    },
    Scenario {
        id: 23,
        shape: GraphShape::Baseline,
        query: "MATCH (a:A) MATCH (a)-[:LIKES*2]->()-[:LIKES]->(c) RETURN c.name",
        expected_names: DEPTH_3,
    },
    Scenario {
        id: 24,
        shape: GraphShape::Baseline,
        query: "MATCH (a:A) MATCH (a)-[:LIKES]->()-[:LIKES*2]->(c) RETURN c.name",
        expected_names: DEPTH_3,
    },
    Scenario {
        id: 25,
        shape: GraphShape::Extended,
        query: "MATCH (a:A) MATCH (a)-[:LIKES]->()-[:LIKES*3]->(c) RETURN c.name",
        expected_names: DEPTH_4,
    },
    Scenario {
        id: 26,
        shape: GraphShape::ReversedTopExtended,
        query: "MATCH (a:A) MATCH (a)<-[:LIKES]-()-[:LIKES*3]->(c) RETURN c.name",
        expected_names: DEPTH_4,
    },
    Scenario {
        id: 27,
        shape: GraphShape::ReversedBelowTopExtended,
        query: "MATCH (a:A) MATCH (a)-[:LIKES]->()<-[:LIKES*3]->(c) RETURN c.name",
        expected_names: DEPTH_4,
    },
    Scenario {
        id: 28,
        shape: GraphShape::Extended,
        query: "MATCH (a:A) MATCH (p)-[:LIKES*1]->()-[:LIKES]->()-[r:LIKES*2]->(c) RETURN c.name",
        expected_names: DEPTH_4,
    },
    Scenario {
        id: 29,
        shape: GraphShape::Extended,
        query: "MATCH (a:A) MATCH (p)-[:LIKES]->()-[:LIKES*2]->()-[r:LIKES]->(c) RETURN c.name",
        expected_names: DEPTH_4,
    },
];

struct Fixture {
    graph: GraphStore,
    project: ProjectId,
    bookmark: Bookmark,
}

fn node_id(depth: usize, position: usize) -> NodeId {
    NodeId((1_u64 << depth).saturating_add(position as u64))
}

fn node_name(depth: usize, position: usize) -> String {
    if depth == 0 {
        "n0".to_owned()
    } else {
        format!("n0{position:0width$b}", width = depth)
    }
}

fn fixture(shape: GraphShape) -> Result<Fixture> {
    let mut graph = GraphStore::default();
    let labels = ["A", "B", "C", "D", "E"]
        .into_iter()
        .map(|name| graph.catalog_mut().intern_label(name))
        .collect::<Result<Vec<_>>>()?;
    let likes = graph.catalog_mut().intern_relationship_type("LIKES")?;
    let name_property = graph.catalog_mut().intern_property("name")?;

    for depth in 0..=shape.maximum_depth() {
        for position in 0..(1_usize << depth) {
            let id = node_id(depth, position);
            graph.insert_node(NodeInput {
                id,
                layer: Layer::Observed,
                revision: id.0,
                labels: vec![labels[depth]],
                properties: vec![(
                    name_property,
                    ScalarValue::String(node_name(depth, position).into()),
                )],
            })?;
        }
    }

    for depth in 1..=shape.maximum_depth() {
        for position in 0..(1_usize << depth) {
            let parent = node_id(depth - 1, position / 2);
            let child = node_id(depth, position);
            let reverse = match shape {
                GraphShape::Baseline | GraphShape::Extended => false,
                GraphShape::ReversedTopExtended => depth == 1,
                GraphShape::ReversedBelowTopExtended => (2..=3).contains(&depth),
            };
            let (source, target) = if reverse {
                (child, parent)
            } else {
                (parent, child)
            };
            let edge_id = EdgeId(1_000_u64.saturating_add(child.0));
            graph.insert_edge(EdgeInput {
                id: edge_id,
                source,
                target,
                relationship_type: likes,
                layer: Layer::Observed,
                revision: edge_id.0,
                properties: Vec::new(),
            })?;
        }
    }

    Ok(Fixture {
        graph,
        project: ProjectId(uuid::Uuid::from_u128(shape.project_tag())),
        bookmark: Bookmark {
            term: 9,
            index: 100 + shape.project_tag() as u64,
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
        max_batch_rows: 7,
        optimizer_statistics: None,
        backend,
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(30)),
        resolved_query_at_time_nanos: None,
    }
}

fn names(output: &ExecutionOutput) -> Result<Vec<String>> {
    if !output.graph_mutations.is_empty() || !output.temporal_mutations.is_empty() {
        return Err(Error::internal("read-only Match5 query produced mutations"));
    }
    if output.result.statistics != StatementStats::default() {
        return Err(Error::internal(
            "read-only Match5 query reported side effects",
        ));
    }
    let mut names = Vec::new();
    for batch in &output.result.batches {
        let column = batch
            .columns
            .iter()
            .find(|column| column.name == "c.name")
            .ok_or_else(|| Error::internal("Match5 result omitted `c.name`"))?;
        for value in &column.values {
            let ResultValue::Scalar(ScalarValue::String(value)) = value else {
                return Err(Error::internal("Match5 `c.name` was not a STRING"));
            };
            names.push(value.to_string());
        }
    }
    names.sort();
    Ok(names)
}

fn expected_names(scenario: Scenario) -> Vec<String> {
    let mut expected = scenario
        .expected_names
        .iter()
        .map(|value| (*value).to_owned())
        .collect::<Vec<_>>();
    expected.sort();
    expected
}

fn execute_names(
    fixture: &Fixture,
    backend: Option<&dyn ExecutionBackend>,
    require_native_execution: bool,
    scenario: Scenario,
) -> Result<Vec<String>> {
    let output = QueryEngine.execute(
        scenario.query,
        &mut context(fixture, backend, require_native_execution),
    )?;
    names(&output)
}

#[derive(Default)]
struct RouteObservations {
    pins: AtomicUsize,
    variable_paths: AtomicUsize,
    scans: AtomicUsize,
    adjacency: AtomicUsize,
    unreceipted_node_pipelines: AtomicUsize,
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

    /// Fault injection only. If a variable-path route accepts this CPU result as Metal, it has
    /// no trustworthy backend-completion proof and this test must fail.
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
        // The existing result has no execution-scoped backend-completion receipts. Extending it
        // naively for variable paths would allow a CPU result to masquerade as Metal.
        self.observations
            .unreceipted_node_pipelines
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

fn assert_no_legacy_variable_path_route(observations: &RouteObservations, label: &str) {
    assert_eq!(
        observations.scans.load(Ordering::SeqCst),
        0,
        "{label}: variable MATCH entered a host-visible scan"
    );
    assert_eq!(
        observations.adjacency.load(Ordering::SeqCst),
        0,
        "{label}: Rust drove one-hop adjacency calls instead of one native variable-path stage"
    );
    assert_eq!(
        observations
            .unreceipted_node_pipelines
            .load(Ordering::SeqCst),
        0,
        "{label}: variable MATCH used the existing unreceipted fixed-hop pipeline"
    );
}

#[test]
fn manifest_pins_all_29_official_match5_queries_and_semantic_classes() {
    assert_eq!(SCENARIOS.len(), 29);
    assert_eq!(
        SCENARIOS
            .iter()
            .map(|scenario| scenario.id)
            .collect::<Vec<_>>(),
        (1_u8..=29).collect::<Vec<_>>()
    );
    assert!(
        SCENARIOS
            .iter()
            .all(|scenario| scenario.query.contains("MATCH ("))
    );
    assert!(
        SCENARIOS
            .iter()
            .all(|scenario| scenario.query.contains("RETURN c.name"))
    );
    assert!(
        SCENARIOS
            .iter()
            .any(|scenario| scenario.query.contains("*0"))
    );
    assert!(
        SCENARIOS
            .iter()
            .any(|scenario| scenario.query.contains("*2..1"))
    );
    assert!(
        SCENARIOS
            .iter()
            .any(|scenario| scenario.query.contains("*0.."))
    );
    assert!(
        SCENARIOS
            .iter()
            .any(|scenario| scenario.query.contains("<-[:LIKES]-"))
    );
    assert!(
        SCENARIOS
            .iter()
            .any(|scenario| scenario.query.contains("<-[:LIKES*3]->"))
    );
    assert!(
        SCENARIOS
            .iter()
            .any(|scenario| scenario.query.contains("[r:LIKES*2]"))
    );
    assert_eq!(SCENARIOS[0].expected_names.len(), 14);
    assert_eq!(SCENARIOS[10].expected_names.len(), 0);
    assert_eq!(SCENARIOS[24].expected_names.len(), 16);
    assert_eq!(FEATURE, "features/clauses/match/Match5.feature");
}

#[test]
fn generic_cpu_oracle_confirms_all_29_official_match5_semantics() -> Result<()> {
    let mut failures = Vec::new();
    for scenario in SCENARIOS {
        let fixture = fixture(scenario.shape)?;
        match execute_names(&fixture, None, false, *scenario) {
            Ok(actual) if actual == expected_names(*scenario) => {}
            Ok(actual) => failures.push(format!(
                "{FEATURE} [{}] result mismatch: expected {:?}, got {actual:?}",
                scenario.id,
                expected_names(*scenario)
            )),
            Err(error) => failures.push(format!(
                "{FEATURE} [{}] CPU oracle failed with {:?}: {error}",
                scenario.id, error.code
            )),
        }
    }
    assert!(
        failures.is_empty(),
        "CPU Match5 oracle had {} failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
    Ok(())
}

#[test]
fn strict_cpu_executes_all_29_through_one_receipted_variable_path_stage() -> Result<()> {
    let mut failures = Vec::new();
    for scenario in SCENARIOS {
        let fixture = fixture(scenario.shape)?;
        let backend = ObservedBackend::new(cpu_backend(&fixture)?);
        let observations = backend.observations();
        match execute_names(&fixture, Some(&backend), true, *scenario) {
            Ok(actual) if actual == expected_names(*scenario) => {}
            Ok(actual) => failures.push(format!(
                "{FEATURE} [{}] strict CPU mismatch: expected {:?}, got {actual:?}",
                scenario.id,
                expected_names(*scenario)
            )),
            Err(error) => failures.push(format!(
                "{FEATURE} [{}] strict CPU failed with {:?}: {error}",
                scenario.id, error.code
            )),
        }
        if observations.pins.load(Ordering::SeqCst) != 1
            || observations.variable_paths.load(Ordering::SeqCst) != 1
        {
            failures.push(format!(
                "{FEATURE} [{}] did not pin and execute exactly one native variable-path stage",
                scenario.id
            ));
        }
        assert_no_legacy_variable_path_route(&observations, "strict CPU variable path");
    }
    assert!(
        failures.is_empty(),
        "strict CPU Match5 route had {} failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
    Ok(())
}

#[test]
fn cpu_cannot_masquerade_as_metal_or_fall_back_to_host_variable_path_walking() -> Result<()> {
    let scenario = SCENARIOS[0];
    let fixture = fixture(scenario.shape)?;
    let backend = ObservedBackend::reporting(cpu_backend(&fixture)?, BackendKind::Metal);
    let observations = backend.observations();
    let mut emitted = Vec::<ExecutionStreamItem>::new();
    let error = QueryEngine
        .execute_streaming(
            scenario.query,
            &mut context(&fixture, Some(&backend), true),
            &mut |item| {
                emitted.push(item);
                Ok(())
            },
        )
        .expect_err("a CPU backend advertised as Metal must never publish a Match5 answer");

    assert!(
        matches!(
            error.code,
            ErrorCode::GpuAdmissionFailure | ErrorCode::CorruptStorage
        ),
        "unexpected strict-route failure: {error}"
    );
    assert!(
        emitted.is_empty(),
        "strict rejection emitted partial results"
    );
    assert!(
        observations.pins.load(Ordering::SeqCst) <= 1,
        "strict rejection pinned more than one resident generation"
    );
    assert_eq!(
        observations.variable_paths.load(Ordering::SeqCst),
        1,
        "CPU-as-Metal fault injection did not reach exactly one complete variable-path stage"
    );
    assert_no_legacy_variable_path_route(&observations, "CPU-as-Metal fault injection");
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
#[ignore = "hardware acceptance gate: all 29 Match5 scenarios require a receipted native Metal variable-path operator"]
fn real_metal_executes_all_29_match5_scenarios_without_fallback_or_host_traversal() -> Result<()> {
    let _guard = metal_test_guard();
    let metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
    let mut backend = ObservedBackend::new(metal);
    let observations = backend.observations();
    let mut failures = Vec::new();

    for scenario in SCENARIOS {
        let fixture = fixture(scenario.shape)?;
        backend.admit_project(resident_image(&fixture)?)?;
        let pins_before = observations.pins.load(Ordering::SeqCst);
        let variable_paths_before = observations.variable_paths.load(Ordering::SeqCst);
        let scans_before = observations.scans.load(Ordering::SeqCst);
        let adjacency_before = observations.adjacency.load(Ordering::SeqCst);
        let legacy_pipeline_before = observations
            .unreceipted_node_pipelines
            .load(Ordering::SeqCst);

        match execute_names(&fixture, Some(&backend), true, *scenario) {
            Ok(actual) if actual == expected_names(*scenario) => {}
            Ok(actual) => failures.push(format!(
                "{FEATURE} [{}] real Metal mismatch: expected {:?}, got {actual:?}",
                scenario.id,
                expected_names(*scenario)
            )),
            Err(error) => failures.push(format!(
                "{FEATURE} [{}] real Metal failed with {:?}: {error}",
                scenario.id, error.code
            )),
        }

        if observations.pins.load(Ordering::SeqCst) != pins_before + 1 {
            failures.push(format!(
                "{FEATURE} [{}] did not pin exactly one immutable Metal generation",
                scenario.id
            ));
        }
        if observations.variable_paths.load(Ordering::SeqCst) != variable_paths_before + 1 {
            failures.push(format!(
                "{FEATURE} [{}] did not execute exactly one complete Metal variable-path stage",
                scenario.id
            ));
        }
        if observations.scans.load(Ordering::SeqCst) != scans_before
            || observations.adjacency.load(Ordering::SeqCst) != adjacency_before
            || observations
                .unreceipted_node_pipelines
                .load(Ordering::SeqCst)
                != legacy_pipeline_before
        {
            failures.push(format!(
                "{FEATURE} [{}] entered a host-visible scan, Rust-driven adjacency loop, or unreceipted fixed-hop pipeline",
                scenario.id
            ));
        }
    }

    assert!(
        failures.is_empty(),
        "real Metal Match5 acceptance had {} failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
    assert_eq!(observations.pins.load(Ordering::SeqCst), SCENARIOS.len());
    assert_eq!(
        observations.variable_paths.load(Ordering::SeqCst),
        SCENARIOS.len()
    );
    assert_no_legacy_variable_path_route(&observations, "real Metal Match5 acceptance");
    Ok(())
}
