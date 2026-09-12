// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Baseline manifest for the nine remaining Metal-only Set1 gaps.
//!
//! The manifest pins exact feature, scenario, and primary-query identity from the fresh full
//! conformance report. Tranches describe non-overlapping implementation ownership; they are not
//! substitutes for native CPU/Metal semantic acceptance tests in the implementation lanes.

use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use irongraph::{
    Bookmark, EdgeId, Error, ErrorCode, Layer, NodeId, ProjectId, Result, ScalarValue,
    cypher::{BindCapabilities, ExecutionContext, QueryEngine},
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend, ResidentGroup,
        ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest, ResidentNodePipelineRequest,
        ResidentNodePipelineResult, ResidentProjectImage, ResidentRowMutationRequest,
        ResidentRowMutationResult, ResidentSortRequest, ResidentSortResult, ResidentVectorQuery,
        ResidentVectorResult, ScratchReservation,
    },
    graph::{EdgeInput, GraphStore, IndexCatalog, LayerMask, NodeInput, TemporalStore},
    types::{LabelId, PropertyId},
};
use tokio_util::sync::CancellationToken;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Tranche {
    ScalarEntityReturn,
    MatchedListPostWrite,
    CreatedListPostWrite,
    OptionalNullTarget,
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
struct Case {
    report_index: usize,
    feature: &'static str,
    scenario: u8,
    name: &'static str,
    query: &'static str,
    tranche: Tranche,
}

const CASES: [Case; 9] = [
    Case {
        report_index: 823,
        feature: "clauses/set/Set1.feature",
        scenario: 1,
        name: "[1] Set a property",
        query: "MATCH (n:A) WHERE n.name = 'Andres' SET n.name = 'Michael' RETURN n",
        tranche: Tranche::ScalarEntityReturn,
    },
    Case {
        report_index: 824,
        feature: "clauses/set/Set1.feature",
        scenario: 2,
        name: "[2] Set a property to an expression",
        query: "MATCH (n:A) WHERE n.name = 'Andres' SET n.name = n.name + ' was here' RETURN n",
        tranche: Tranche::ScalarEntityReturn,
    },
    Case {
        report_index: 825,
        feature: "clauses/set/Set1.feature",
        scenario: 3,
        name: "[3] Set a property by selecting the node using a simple expression",
        query: "MATCH (n:A) SET (n).name = 'neo4j' RETURN n",
        tranche: Tranche::ScalarEntityReturn,
    },
    Case {
        report_index: 826,
        feature: "clauses/set/Set1.feature",
        scenario: 4,
        name: "[4] Set a property by selecting the relationship using a simple expression",
        query: "MATCH ()-[r:REL]->() SET (r).name = 'neo4j' RETURN r",
        tranche: Tranche::ScalarEntityReturn,
    },
    Case {
        report_index: 827,
        feature: "clauses/set/Set1.feature",
        scenario: 5,
        name: "[5] Adding a list property",
        query: "MATCH (n:A) SET n.numbers = [1, 2, 3] RETURN [i IN n.numbers | i / 2.0] AS x",
        tranche: Tranche::MatchedListPostWrite,
    },
    Case {
        report_index: 828,
        feature: "clauses/set/Set1.feature",
        scenario: 6,
        name: "[6] Concatenate elements onto a list property",
        query: "CREATE (a {numbers: [1, 2, 3]}) SET a.numbers = a.numbers + [4, 5] RETURN a.numbers",
        tranche: Tranche::CreatedListPostWrite,
    },
    Case {
        report_index: 829,
        feature: "clauses/set/Set1.feature",
        scenario: 7,
        name: "[7] Concatenate elements in reverse onto a list property",
        query: "CREATE (a {numbers: [3, 4, 5]}) SET a.numbers = [1, 2] + a.numbers RETURN a.numbers",
        tranche: Tranche::CreatedListPostWrite,
    },
    Case {
        report_index: 830,
        feature: "clauses/set/Set1.feature",
        scenario: 8,
        name: "[8] Ignore null when setting property",
        query: "OPTIONAL MATCH (a:DoesNotExist) SET a.num = 42 RETURN a",
        tranche: Tranche::OptionalNullTarget,
    },
    Case {
        report_index: 833,
        feature: "clauses/set/Set1.feature",
        scenario: 11,
        name: "[11] Set multiple node properties",
        query: "MATCH (n:X) SET n.name = 'A', n.name2 = 'B', n.num = 5 RETURN n",
        tranche: Tranche::ScalarEntityReturn,
    },
];

#[derive(Default)]
struct RouteObservations {
    pins: AtomicUsize,
    complete_commands: AtomicUsize,
    unexpected_calls: AtomicUsize,
}

/// Advertises Metal until the immutable project generation is pinned, then reports the honest
/// CPU completion class. This makes generic execution impossible while retaining the CPU backend
/// as the semantic implementation of the one allowed complete mutation command.
struct StrictCpuMutationBackend {
    inner: Box<dyn ExecutionBackend>,
    pinned: bool,
    observations: Arc<RouteObservations>,
}

impl StrictCpuMutationBackend {
    fn new(inner: CpuBackend) -> Self {
        Self {
            inner: Box::new(inner),
            pinned: false,
            observations: Arc::new(RouteObservations::default()),
        }
    }

    fn observations(&self) -> Arc<RouteObservations> {
        Arc::clone(&self.observations)
    }

    fn reject<T>(&self, route: &'static str) -> Result<T> {
        self.observations
            .unexpected_calls
            .fetch_add(1, Ordering::SeqCst);
        Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!("strict Set1 CPU route rejected `{route}`"),
        ))
    }
}

impl ExecutionBackend for StrictCpuMutationBackend {
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
        if self.pinned {
            return self.reject("repin_project");
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
        if !self.pinned
            || request
                .mutation
                .as_ref()
                .and_then(|program| program.continuation.as_ref())
                .is_none()
        {
            return self.reject("incomplete_node_pipeline");
        }
        self.observations
            .complete_commands
            .fetch_add(1, Ordering::SeqCst);
        self.inner.execute_node_pipeline(request, cancellation)
    }

    fn execute_row_mutation(
        &self,
        _request: &ResidentRowMutationRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentRowMutationResult> {
        self.reject("execute_row_mutation")
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
    project: ProjectId,
    bookmark: Bookmark,
}

fn fixture() -> Result<Fixture> {
    let mut graph = GraphStore::default();
    let a = graph.catalog_mut().intern_label("A")?;
    let x = graph.catalog_mut().intern_label("X")?;
    let rel = graph.catalog_mut().intern_relationship_type("REL")?;
    let name = graph.catalog_mut().intern_property("name")?;
    graph.insert_node(NodeInput {
        id: NodeId(1),
        layer: Layer::Observed,
        revision: 1,
        labels: vec![a],
        properties: vec![(name, ScalarValue::String("Andres".into()))],
    })?;
    for id in [2, 3] {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: 1,
            labels: Vec::new(),
            properties: Vec::new(),
        })?;
    }
    graph.insert_node(NodeInput {
        id: NodeId(4),
        layer: Layer::Observed,
        revision: 1,
        labels: vec![x],
        properties: Vec::new(),
    })?;
    graph.insert_edge(EdgeInput {
        id: EdgeId(1),
        source: NodeId(2),
        target: NodeId(3),
        relationship_type: rel,
        layer: Layer::Observed,
        revision: 1,
        properties: Vec::new(),
    })?;
    Ok(Fixture {
        graph,
        project: ProjectId(uuid::Uuid::nil()),
        bookmark: Bookmark { term: 7, index: 11 },
    })
}

fn cpu_backend(fixture: &Fixture) -> Result<CpuBackend> {
    let image = ResidentProjectImage::build(
        fixture.project,
        fixture.bookmark,
        &fixture.graph,
        &TemporalStore::default(),
        &IndexCatalog::default(),
    )?;
    let mut backend = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    backend.admit_project(image)?;
    Ok(backend)
}

fn context<'a>(
    fixture: &'a Fixture,
    backend: Option<&'a dyn ExecutionBackend>,
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

#[test]
fn tranche_partition_is_complete_and_non_overlapping() {
    let expected = [
        (Tranche::ScalarEntityReturn, 5),
        (Tranche::MatchedListPostWrite, 1),
        (Tranche::CreatedListPostWrite, 2),
        (Tranche::OptionalNullTarget, 1),
    ];
    assert_eq!(
        expected.iter().map(|(_, count)| count).sum::<usize>(),
        CASES.len()
    );
    for (tranche, count) in expected {
        assert_eq!(
            CASES.iter().filter(|case| case.tranche == tranche).count(),
            count,
            "wrong count for {tranche:?}"
        );
    }
}

#[test]
fn strict_cpu_executes_exact_five_scalar_entity_returns_as_one_complete_command() -> Result<()> {
    let selected = CASES
        .iter()
        .filter(|case| case.tranche == Tranche::ScalarEntityReturn)
        .collect::<Vec<_>>();
    assert_eq!(
        selected
            .iter()
            .map(|case| case.report_index)
            .collect::<Vec<_>>(),
        [823, 824, 825, 826, 833]
    );

    for case in selected {
        let fixture = fixture()?;
        let reference = QueryEngine.execute(case.query, &mut context(&fixture, None))?;
        let backend = StrictCpuMutationBackend::new(cpu_backend(&fixture)?);
        let observations = backend.observations();
        let native = QueryEngine.execute(case.query, &mut context(&fixture, Some(&backend)))?;

        assert_eq!(native.result, reference.result, "{}", case.name);
        assert_eq!(
            format!("{:?}", native.graph_mutations),
            format!("{:?}", reference.graph_mutations),
            "{}",
            case.name
        );
        assert_eq!(
            observations.pins.load(Ordering::SeqCst),
            1,
            "{} did not pin exactly one immutable generation",
            case.name
        );
        assert_eq!(
            observations.complete_commands.load(Ordering::SeqCst),
            1,
            "{} did not execute exactly one complete mutation command",
            case.name
        );
        assert_eq!(
            observations.unexpected_calls.load(Ordering::SeqCst),
            0,
            "{} escaped the complete mutation route",
            case.name
        );
    }
    Ok(())
}

#[test]
fn noninteger_scalar_property_and_other_mutation_shapes_still_fail_before_dispatch() -> Result<()> {
    for query in [
        "MATCH (n:A) SET n.name = 'Michael' RETURN n.name",
        "MATCH (n:A) SET n.name = 'Michael' WITH n WHERE n.name % 2 = 0 RETURN n",
        "MATCH (n:A) SET n.name = 'Michael' RETURN sum(n.name)",
        "MATCH (n:A) SET n = {name: 'Michael'} RETURN n",
        "MATCH (n:A) SET n.values = [1, 2, 3] RETURN n",
    ] {
        let fixture = fixture()?;
        let backend = StrictCpuMutationBackend::new(cpu_backend(&fixture)?);
        let observations = backend.observations();
        let error = QueryEngine
            .execute(query, &mut context(&fixture, Some(&backend)))
            .expect_err("unsupported Set1 continuation unexpectedly executed");
        assert_eq!(error.code, ErrorCode::GpuAdmissionFailure, "{query}");
        assert_eq!(
            observations.pins.load(Ordering::SeqCst),
            0,
            "unsupported query pinned a generation: {query}"
        );
        assert_eq!(
            observations.complete_commands.load(Ordering::SeqCst),
            0,
            "unsupported query dispatched a mutation command: {query}"
        );
        assert_eq!(
            observations.unexpected_calls.load(Ordering::SeqCst),
            0,
            "unsupported query entered another native route: {query}"
        );
    }
    Ok(())
}
