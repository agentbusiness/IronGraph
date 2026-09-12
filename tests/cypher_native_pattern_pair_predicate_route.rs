// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use std::sync::MutexGuard;

use irongraph::{
    Bookmark, EdgeId, Error, ErrorCode, Layer, NodeId, ProjectId, Result,
    cypher::{
        BindCapabilities, DependencyKind, EntityDependency, ExecutionContext, ExecutionOutput,
        QueryEngine, ResultValue,
    },
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend, ResidentGroup,
        ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest, ResidentNodePipelineRequest,
        ResidentNodePipelineResult, ResidentObligationKind, ResidentObligationScope,
        ResidentPatternPairPredicateRequest, ResidentPatternPairPredicateResult,
        ResidentPatternPredicateRequest, ResidentPatternPredicateResult, ResidentProjectImage,
        ResidentSortRequest, ResidentSortResult, ResidentVectorQuery, ResidentVectorResult,
        ScratchReservation,
    },
    graph::{EdgeInput, GraphStore, IndexCatalog, LayerMask, NodeInput, TemporalStore},
    types::{LabelId, PropertyId},
};
use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::gpu::MetalBackend;

#[derive(Default)]
struct BackendObservations {
    pins: AtomicUsize,
    pair_calls: AtomicUsize,
    single_pattern_calls: AtomicUsize,
    node_pipeline_calls: AtomicUsize,
    scan_calls: AtomicUsize,
    adjacency_calls: AtomicUsize,
    last_pair_request: Mutex<Option<ResidentPatternPairPredicateRequest>>,
}

struct ObservedBackend {
    inner: Box<dyn ExecutionBackend>,
    reported_kind: BackendKind,
    observations: Arc<BackendObservations>,
}

impl ObservedBackend {
    fn new(inner: impl ExecutionBackend + 'static) -> Self {
        let reported_kind = inner.kind();
        Self {
            inner: Box::new(inner),
            reported_kind,
            observations: Arc::new(BackendObservations::default()),
        }
    }

    /// Fault-injection wrapper only: CPU executes the request, but its completion receipts are
    /// deliberately validated as though they came from `reported_kind`.
    fn reporting(inner: impl ExecutionBackend + 'static, reported_kind: BackendKind) -> Self {
        Self {
            inner: Box::new(inner),
            reported_kind,
            observations: Arc::new(BackendObservations::default()),
        }
    }

    fn observations(&self) -> Arc<BackendObservations> {
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
        self.observations.scan_calls.fetch_add(1, Ordering::SeqCst);
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
        self.observations
            .adjacency_calls
            .fetch_add(1, Ordering::SeqCst);
        self.inner
            .expand_project_out(project, sources, cancellation)
    }

    fn expand_project_in(
        &self,
        project: ProjectId,
        targets: &[u32],
        cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.observations
            .adjacency_calls
            .fetch_add(1, Ordering::SeqCst);
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
            .node_pipeline_calls
            .fetch_add(1, Ordering::SeqCst);
        self.inner.execute_node_pipeline(request, cancellation)
    }

    fn execute_pattern_predicate(
        &self,
        request: &ResidentPatternPredicateRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentPatternPredicateResult> {
        self.observations
            .single_pattern_calls
            .fetch_add(1, Ordering::SeqCst);
        self.inner.execute_pattern_predicate(request, cancellation)
    }

    fn execute_pattern_predicate_pairs(
        &self,
        request: &ResidentPatternPairPredicateRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentPatternPairPredicateResult> {
        self.observations.pair_calls.fetch_add(1, Ordering::SeqCst);
        *self
            .observations
            .last_pair_request
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(request.clone());
        self.inner
            .execute_pattern_predicate_pairs(request, cancellation)
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
        self.observations
            .adjacency_calls
            .fetch_add(1, Ordering::SeqCst);
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

struct Fixture {
    graph: GraphStore,
    project: ProjectId,
    bookmark: Bookmark,
}

fn official_fixture() -> Result<Fixture> {
    let mut graph = GraphStore::default();
    let labels = [
        graph.catalog_mut().intern_label("A")?,
        graph.catalog_mut().intern_label("B")?,
        graph.catalog_mut().intern_label("C")?,
        graph.catalog_mut().intern_label("D")?,
    ];
    let rel1 = graph.catalog_mut().intern_relationship_type("REL1")?;
    let rel2 = graph.catalog_mut().intern_relationship_type("REL2")?;
    let rel3 = graph.catalog_mut().intern_relationship_type("REL3")?;
    for (offset, label) in labels.into_iter().enumerate() {
        let id = u64::try_from(offset).unwrap_or(0) + 1;
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: id,
            labels: vec![label],
            properties: Vec::new(),
        })?;
    }
    for (id, source, target, relationship_type) in [
        (10, 1, 2, rel1),
        (11, 2, 1, rel2),
        (12, 1, 3, rel3),
        (13, 1, 4, rel1),
    ] {
        graph.insert_edge(EdgeInput {
            id: EdgeId(id),
            source: NodeId(source),
            target: NodeId(target),
            relationship_type,
            layer: Layer::Observed,
            revision: id,
            properties: Vec::new(),
        })?;
    }
    Ok(Fixture {
        graph,
        project: ProjectId(uuid::Uuid::nil()),
        bookmark: Bookmark { term: 7, index: 40 },
    })
}

/// Four disconnected components make relationship-trail correctness observable:
///
/// - one ordinary edge, which cannot be reused to return to its source;
/// - two parallel edges, which form a valid two-distinct-relationship return trail;
/// - a directed three-edge cycle, which requires real multi-hop reachability;
/// - one self-loop, which is itself a valid non-empty same-endpoint trail.
fn trail_fixture() -> Result<Fixture> {
    let mut graph = GraphStore::default();
    let label = graph.catalog_mut().intern_label("N")?;
    let rel1 = graph.catalog_mut().intern_relationship_type("REL1")?;
    for id in 1..=8 {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: id,
            labels: vec![label],
            properties: Vec::new(),
        })?;
    }
    for (id, source, target) in [
        (101, 1, 2),
        (102, 3, 4),
        (103, 3, 4),
        (104, 5, 6),
        (105, 6, 7),
        (106, 7, 5),
        (107, 8, 8),
    ] {
        graph.insert_edge(EdgeInput {
            id: EdgeId(id),
            source: NodeId(source),
            target: NodeId(target),
            relationship_type: rel1,
            layer: Layer::Observed,
            revision: id,
            properties: Vec::new(),
        })?;
    }
    Ok(Fixture {
        graph,
        project: ProjectId(uuid::Uuid::nil()),
        bookmark: Bookmark {
            term: 11,
            index: 107,
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
    let mut backend = CpuBackend::new(128 * 1024 * 1024, 32 * 1024 * 1024);
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
        next_node_id: 1_000,
        next_edge_id: 1_000,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            require_native_execution,
            ..BindCapabilities::default()
        },
        max_result_rows: 1_024,
        max_batch_rows: 2,
        optimizer_statistics: None,
        backend,
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(10)),
        resolved_query_at_time_nanos: None,
    }
}

fn output_pairs(output: &ExecutionOutput) -> Result<Vec<(NodeId, NodeId)>> {
    let mut pairs = Vec::new();
    for batch in &output.result.batches {
        let n_values = batch
            .columns
            .iter()
            .find(|column| column.name == "n")
            .ok_or_else(|| Error::internal("paired result omitted `n`"))?;
        let m_values = batch
            .columns
            .iter()
            .find(|column| column.name == "m")
            .ok_or_else(|| Error::internal("paired result omitted `m`"))?;
        if n_values.values.len() != m_values.values.len() {
            return Err(Error::internal("paired result columns are not aligned"));
        }
        for (n, m) in n_values.values.iter().zip(&m_values.values) {
            let (ResultValue::Node(n), ResultValue::Node(m)) = (n, m) else {
                return Err(Error::internal("paired result contained a non-node value"));
            };
            pairs.push((n.id, m.id));
        }
    }
    Ok(pairs)
}

fn execute_pairs(
    fixture: &Fixture,
    backend: Option<&dyn ExecutionBackend>,
    require_native_execution: bool,
    query: &str,
) -> Result<Vec<(NodeId, NodeId)>> {
    let output = QueryEngine.execute(
        query,
        &mut context(fixture, backend, require_native_execution),
    )?;
    let mut pairs = output_pairs(&output)?;
    pairs.sort_unstable();
    Ok(pairs)
}

fn node_pairs(values: &[(u64, u64)]) -> Vec<(NodeId, NodeId)> {
    values
        .iter()
        .map(|(n, m)| (NodeId(*n), NodeId(*m)))
        .collect()
}

fn assert_no_host_pair_fallback(observations: &BackendObservations) {
    assert_eq!(
        observations.single_pattern_calls.load(Ordering::SeqCst),
        0,
        "paired route fell back to the one-endpoint Pattern1 stage"
    );
    assert_eq!(
        observations.node_pipeline_calls.load(Ordering::SeqCst),
        0,
        "paired route fell back to the generic node pipeline"
    );
    assert_eq!(
        observations.scan_calls.load(Ordering::SeqCst),
        0,
        "paired route materialized a host node scan"
    );
    assert_eq!(
        observations.adjacency_calls.load(Ordering::SeqCst),
        0,
        "paired route materialized host adjacency"
    );
}

#[test]
fn fused_pair_route_uses_one_call_one_pin_and_backend_owned_row_major_scans() -> Result<()> {
    let fixture = official_fixture()?;
    let backend = ObservedBackend::new(cpu_backend(&fixture)?);
    let observations = backend.observations();
    let output = QueryEngine.execute(
        "MATCH (n), (m) WHERE (n)-[]->(m) RETURN n, m",
        &mut context(&fixture, Some(&backend), true),
    )?;

    let mut actual_pairs = output_pairs(&output)?;
    actual_pairs.sort_unstable();
    assert_eq!(actual_pairs, node_pairs(&[(1, 2), (1, 3), (1, 4), (2, 1)]));
    assert_eq!(observations.pins.load(Ordering::SeqCst), 1);
    assert_eq!(observations.pair_calls.load(Ordering::SeqCst), 1);
    assert_no_host_pair_fallback(&observations);

    let request = observations
        .last_pair_request
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
        .ok_or_else(|| Error::internal("fused paired request was not observed"))?;
    let node_slots = fixture.graph.node_slot_count();
    let pair_domain = node_slots
        .checked_mul(node_slots)
        .ok_or_else(|| Error::internal("test pair domain overflow"))?;
    assert_eq!(request.expected_bookmark, fixture.bookmark);
    assert_eq!(request.expected_graph_revision, fixture.graph.revision());
    assert_eq!(
        request.expected_layout_version,
        fixture.graph.layout_version()
    );
    assert_eq!(request.input.node_slots, node_slots);
    assert_eq!(request.pair_domain()?, pair_domain);
    assert_eq!(request.max_output_pairs, pair_domain);
    assert_eq!(request.obligations().map(|item| item.id), [1, 2, 3, 4, 5]);
    assert_eq!(
        request.input.n_obligation.kind,
        ResidentObligationKind::PatternScan
    );
    assert_eq!(
        request.input.n_obligation.scope,
        ResidentObligationScope::PatternScanN
    );
    assert_eq!(
        request.input.m_obligation.kind,
        ResidentObligationKind::PatternScan
    );
    assert_eq!(
        request.input.m_obligation.scope,
        ResidentObligationScope::PatternScanM
    );
    assert_eq!(
        request.input.cartesian_obligation.kind,
        ResidentObligationKind::PatternCartesian
    );
    assert_eq!(
        request.input.cartesian_obligation.scope,
        ResidentObligationScope::PatternCartesian
    );
    Ok(())
}

#[test]
fn fused_cpu_matches_backend_free_oracle_for_official_pattern1_scenarios_12_to_18() -> Result<()> {
    let fixture = official_fixture()?;
    let scenarios: [(&str, &[(u64, u64)]); 7] = [
        (
            "MATCH (n), (m) WHERE (n)-[]->(m) RETURN n, m",
            &[(1, 2), (1, 3), (1, 4), (2, 1)],
        ),
        (
            "MATCH (n), (m) WHERE (n)-[:REL1|REL2|REL3|REL4]-(m) RETURN n, m",
            &[(1, 2), (1, 3), (1, 4), (2, 1), (3, 1), (4, 1)],
        ),
        (
            "MATCH (n), (m) WHERE (n)-[:REL1]->(m) RETURN n, m",
            &[(1, 2), (1, 4)],
        ),
        (
            "MATCH (n), (m) WHERE (n)-[:REL1]-(m) RETURN n, m",
            &[(1, 2), (1, 4), (2, 1), (4, 1)],
        ),
        (
            "MATCH (n), (m) WHERE (n)-[:REL1*]->(m) RETURN n, m",
            &[(1, 2), (1, 4)],
        ),
        (
            "MATCH (n), (m) WHERE (n)-[:REL1*]-(m) RETURN n, m",
            &[(1, 2), (1, 4), (2, 1), (2, 4), (4, 1), (4, 2)],
        ),
        (
            "MATCH (n), (m) WHERE (n)-[:REL1*2]-(m) RETURN n, m",
            &[(2, 4), (4, 2)],
        ),
    ];

    for (query, expected) in scenarios {
        let oracle = execute_pairs(&fixture, None, false, query)?;
        assert_eq!(oracle, node_pairs(expected), "backend-free query: {query}");
        let backend = ObservedBackend::new(cpu_backend(&fixture)?);
        let observations = backend.observations();
        let resident = execute_pairs(&fixture, Some(&backend), true, query)?;
        assert_eq!(resident, oracle, "native CPU query: {query}");
        assert_eq!(
            observations.pins.load(Ordering::SeqCst),
            1,
            "native CPU query was not pinned exactly once: {query}"
        );
        assert_eq!(
            observations.pair_calls.load(Ordering::SeqCst),
            1,
            "native CPU query missed the paired stage: {query}"
        );
        assert_no_host_pair_fallback(&observations);
    }
    Ok(())
}

#[test]
fn fused_cpu_proves_reachability_and_distinct_relationship_trails() -> Result<()> {
    let fixture = trail_fixture()?;
    let directed_query = "MATCH (n), (m) WHERE (n)-[:REL1*]->(m) RETURN n, m";
    let undirected_query = "MATCH (n), (m) WHERE (n)-[:REL1*]-(m) RETURN n, m";
    let exact_two_query = "MATCH (n), (m) WHERE (n)-[:REL1*2]-(m) RETURN n, m";

    let directed_oracle = execute_pairs(&fixture, None, false, directed_query)?;
    let backend = ObservedBackend::new(cpu_backend(&fixture)?);
    let observations = backend.observations();
    let directed = execute_pairs(&fixture, Some(&backend), true, directed_query)?;
    assert_eq!(directed, directed_oracle);
    assert_eq!(observations.pins.load(Ordering::SeqCst), 1);
    assert_eq!(observations.pair_calls.load(Ordering::SeqCst), 1);
    assert_no_host_pair_fallback(&observations);
    assert!(
        directed.contains(&(NodeId(5), NodeId(7))),
        "`*` must prove a transitive path, not only a one-hop witness"
    );
    assert!(
        directed.contains(&(NodeId(5), NodeId(5))),
        "a directed cycle is a valid non-empty same-endpoint trail"
    );
    assert!(
        directed.contains(&(NodeId(8), NodeId(8))),
        "a self-loop is a valid non-empty same-endpoint trail"
    );
    assert!(!directed.contains(&(NodeId(1), NodeId(1))));

    let undirected_oracle = execute_pairs(&fixture, None, false, undirected_query)?;
    let backend = ObservedBackend::new(cpu_backend(&fixture)?);
    let observations = backend.observations();
    let undirected = execute_pairs(&fixture, Some(&backend), true, undirected_query)?;
    assert_eq!(undirected, undirected_oracle);
    assert_eq!(observations.pins.load(Ordering::SeqCst), 1);
    assert_eq!(observations.pair_calls.load(Ordering::SeqCst), 1);
    assert_no_host_pair_fallback(&observations);
    for pair in [(3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8)] {
        assert!(
            undirected.contains(&(NodeId(pair.0), NodeId(pair.1))),
            "expected a valid same-endpoint trail for {pair:?}"
        );
    }
    for pair in [(1, 1), (2, 2)] {
        assert!(
            !undirected.contains(&(NodeId(pair.0), NodeId(pair.1))),
            "one undirected relationship must not be reused out and back for {pair:?}"
        );
    }

    let exact_two_oracle = execute_pairs(&fixture, None, false, exact_two_query)?;
    let backend = ObservedBackend::new(cpu_backend(&fixture)?);
    let observations = backend.observations();
    let exact_two = execute_pairs(&fixture, Some(&backend), true, exact_two_query)?;
    assert_eq!(exact_two, exact_two_oracle);
    assert_eq!(observations.pins.load(Ordering::SeqCst), 1);
    assert_eq!(observations.pair_calls.load(Ordering::SeqCst), 1);
    assert_no_host_pair_fallback(&observations);
    for pair in [(3, 3), (4, 4)] {
        assert!(
            exact_two.contains(&(NodeId(pair.0), NodeId(pair.1))),
            "parallel relationships must form a valid exact-two return trail for {pair:?}"
        );
    }
    for pair in [(1, 1), (2, 2), (5, 5), (6, 6), (7, 7), (8, 8)] {
        assert!(
            !exact_two.contains(&(NodeId(pair.0), NodeId(pair.1))),
            "exact-two must use two distinct relationships for {pair:?}"
        );
    }
    Ok(())
}

#[test]
fn stale_bookmark_revision_layout_or_catalog_emits_nothing_and_never_falls_back() -> Result<()> {
    let query = "MATCH (n), (m) WHERE (n)-[:REL1]->(m) RETURN n, m";

    let bookmark_fixture = official_fixture()?;
    let backend = ObservedBackend::new(cpu_backend(&bookmark_fixture)?);
    let observations = backend.observations();
    let mut stale_bookmark = context(&bookmark_fixture, Some(&backend), true);
    stale_bookmark.bookmark.index = stale_bookmark.bookmark.index.saturating_add(1);
    let mut emitted = 0_usize;
    let error = QueryEngine
        .execute_streaming(query, &mut stale_bookmark, &mut |_| {
            emitted += 1;
            Ok(())
        })
        .expect_err("stale bookmark must fail");
    assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
    assert_eq!(emitted, 0);
    assert_eq!(observations.pair_calls.load(Ordering::SeqCst), 0);
    assert_no_host_pair_fallback(&observations);

    let mut revision_fixture = official_fixture()?;
    let backend = ObservedBackend::new(cpu_backend(&revision_fixture)?);
    let observations = backend.observations();
    revision_fixture.graph.insert_node(NodeInput {
        id: NodeId(99),
        layer: Layer::Observed,
        revision: revision_fixture.graph.revision().saturating_add(1),
        labels: Vec::new(),
        properties: Vec::new(),
    })?;
    let mut emitted = 0_usize;
    let error = QueryEngine
        .execute_streaming(
            query,
            &mut context(&revision_fixture, Some(&backend), true),
            &mut |_| {
                emitted += 1;
                Ok(())
            },
        )
        .expect_err("stale graph revision must fail");
    assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
    assert_eq!(emitted, 0);
    assert_eq!(observations.pair_calls.load(Ordering::SeqCst), 0);
    assert_no_host_pair_fallback(&observations);

    let mut layout_fixture = official_fixture()?;
    let backend = ObservedBackend::new(cpu_backend(&layout_fixture)?);
    let observations = backend.observations();
    let prior_layout = layout_fixture.graph.layout_version();
    let _mapping = layout_fixture.graph.compact()?;
    assert_ne!(layout_fixture.graph.layout_version(), prior_layout);
    let mut emitted = 0_usize;
    let error = QueryEngine
        .execute_streaming(
            query,
            &mut context(&layout_fixture, Some(&backend), true),
            &mut |_| {
                emitted += 1;
                Ok(())
            },
        )
        .expect_err("stale dense layout must fail");
    assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
    assert_eq!(emitted, 0);
    assert_eq!(observations.pair_calls.load(Ordering::SeqCst), 1);
    assert_no_host_pair_fallback(&observations);

    let catalog_fixture = official_fixture()?;
    let backend = ObservedBackend::new(cpu_backend(&catalog_fixture)?);
    let observations = backend.observations();
    let mut stale_catalog = catalog_fixture.graph.catalog().clone();
    stale_catalog.intern_property("stale_only")?;
    let mut execution = context(&catalog_fixture, Some(&backend), true);
    execution.binding_catalog = &stale_catalog;
    let mut emitted = 0_usize;
    let error = QueryEngine
        .execute_streaming(query, &mut execution, &mut |_| {
            emitted += 1;
            Ok(())
        })
        .expect_err("stale binding catalog must fail");
    assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
    assert_eq!(emitted, 0);
    assert_eq!(observations.pair_calls.load(Ordering::SeqCst), 0);
    assert_no_host_pair_fallback(&observations);
    Ok(())
}

#[test]
fn cpu_receipts_masquerading_as_metal_fail_without_output_or_fallback() -> Result<()> {
    let fixture = official_fixture()?;
    let backend = ObservedBackend::reporting(cpu_backend(&fixture)?, BackendKind::Metal);
    let observations = backend.observations();
    let mut emitted = 0_usize;
    let error = QueryEngine
        .execute_streaming(
            "MATCH (n), (m) WHERE (n)-[:REL1]->(m) RETURN n, m",
            &mut context(&fixture, Some(&backend), true),
            &mut |_| {
                emitted += 1;
                Ok(())
            },
        )
        .expect_err("CPU paired receipts reported as Metal must fail validation");
    assert_eq!(error.code, ErrorCode::CorruptStorage);
    assert_eq!(emitted, 0);
    assert_eq!(observations.pair_calls.load(Ordering::SeqCst), 1);
    assert_no_host_pair_fallback(&observations);
    Ok(())
}

#[test]
fn fused_pair_result_budget_fails_without_truncation_or_stream_output() -> Result<()> {
    let fixture = official_fixture()?;
    let backend = ObservedBackend::new(cpu_backend(&fixture)?);
    let observations = backend.observations();
    let mut execution = context(&fixture, Some(&backend), true);
    execution.max_result_rows = 1;
    let mut emitted = 0_usize;
    let error = QueryEngine
        .execute_streaming(
            "MATCH (n), (m) WHERE (n)-[]->(m) RETURN n, m",
            &mut execution,
            &mut |_| {
                emitted += 1;
                Ok(())
            },
        )
        .expect_err("resident paired output must not be truncated");
    assert_eq!(error.code, ErrorCode::ResultBudgetExceeded);
    assert_eq!(emitted, 0);
    assert_eq!(observations.pair_calls.load(Ordering::SeqCst), 1);
    assert_no_host_pair_fallback(&observations);
    let request = observations
        .last_pair_request
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
        .ok_or_else(|| Error::internal("bounded paired request was not observed"))?;
    assert_eq!(request.max_output_pairs, 1);
    Ok(())
}

#[test]
fn fused_pair_route_records_returned_nodes_and_relationship_predicates() -> Result<()> {
    let fixture = official_fixture()?;
    let backend = cpu_backend(&fixture)?;
    let output = QueryEngine.execute(
        "MATCH (n), (m) WHERE (n)-[:REL1]->(m) RETURN n, m",
        &mut context(&fixture, Some(&backend), true),
    )?;
    assert_eq!(output_pairs(&output)?, node_pairs(&[(1, 2), (1, 4)]));

    let expected_nodes = BTreeSet::from([NodeId(1), NodeId(2), NodeId(4)]);
    let actual_nodes = output
        .dependencies
        .entities
        .keys()
        .filter_map(|dependency| match dependency {
            EntityDependency::Node(node) => Some(*node),
            EntityDependency::Relationship(_) => None,
        })
        .collect::<BTreeSet<_>>();
    assert!(
        expected_nodes.is_subset(&actual_nodes),
        "dependencies omitted one or more returned nodes: {actual_nodes:?}"
    );
    assert!(
        output
            .dependencies
            .predicates
            .keys()
            .any(|dependency| { dependency.kind == DependencyKind::FullNodeScan })
    );
    assert!(
        output
            .dependencies
            .predicates
            .keys()
            .any(|dependency| { dependency.kind == DependencyKind::FullRelationshipScan })
    );
    assert!(output.dependencies.predicates.keys().any(|dependency| {
        dependency.kind == DependencyKind::RelationshipType && dependency.name == "REL1"
    }));
    Ok(())
}

#[test]
fn unsupported_whole_plan_shapes_make_zero_paired_calls() -> Result<()> {
    let fixture = official_fixture()?;
    for query in [
        "MATCH (n:A), (m) WHERE (n)-[]->(m) RETURN n, m",
        "MATCH (n), (m) WHERE (n)-[]->(m) RETURN n AS renamed, m",
        "MATCH (n), (m) WHERE (n)-[]->(m) RETURN DISTINCT n, m",
        "MATCH (n), (m) WHERE (n)-[]->(m) RETURN n, m LIMIT 1",
        "MATCH (n), (m) WHERE (n)-[]->(m) RETURN m, n",
    ] {
        let backend = ObservedBackend::new(cpu_backend(&fixture)?);
        let observations = backend.observations();
        QueryEngine.execute(query, &mut context(&fixture, Some(&backend), false))?;
        assert_eq!(
            observations.pair_calls.load(Ordering::SeqCst),
            0,
            "unsupported plan entered the exact paired route: {query}"
        );
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
fn assert_real_metal_parity(fixture: &Fixture, queries: &[&str]) -> Result<()> {
    let image = resident_image(fixture)?;
    let mut cpu = CpuBackend::new(256 * 1024 * 1024, 64 * 1024 * 1024);
    let mut metal = MetalBackend::new(0, 256 * 1024 * 1024, 64 * 1024 * 1024)?;
    cpu.admit_project(image.clone())?;
    metal.admit_project(image)?;
    let cpu = ObservedBackend::new(cpu);
    let metal = ObservedBackend::new(metal);
    let cpu_observations = cpu.observations();
    let metal_observations = metal.observations();
    for (index, query) in queries.iter().enumerate() {
        let cpu_pairs = execute_pairs(fixture, Some(&cpu), true, query)?;
        let metal_pairs = execute_pairs(fixture, Some(&metal), true, query)?;
        assert_eq!(metal_pairs, cpu_pairs, "real Metal query: {query}");
        let expected_calls = index + 1;
        assert_eq!(
            cpu_observations.pair_calls.load(Ordering::SeqCst),
            expected_calls,
            "CPU missed the paired stage: {query}"
        );
        assert_eq!(
            metal_observations.pair_calls.load(Ordering::SeqCst),
            expected_calls,
            "Metal missed the paired stage: {query}"
        );
        assert_eq!(
            cpu_observations.pins.load(Ordering::SeqCst),
            expected_calls,
            "CPU pin count diverged: {query}"
        );
        assert_eq!(
            metal_observations.pins.load(Ordering::SeqCst),
            expected_calls,
            "Metal pin count diverged: {query}"
        );
        assert_no_host_pair_fallback(&cpu_observations);
        assert_no_host_pair_fallback(&metal_observations);
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_fused_pair_route_matches_cpu_for_official_and_trail_shapes() -> Result<()> {
    let _guard = metal_test_guard();
    let official = official_fixture()?;
    assert_real_metal_parity(
        &official,
        &[
            "MATCH (n), (m) WHERE (n)-[]->(m) RETURN n, m",
            "MATCH (n), (m) WHERE (n)-[:REL1|REL2|REL3|REL4]-(m) RETURN n, m",
            "MATCH (n), (m) WHERE (n)-[:REL1]->(m) RETURN n, m",
            "MATCH (n), (m) WHERE (n)-[:REL1]-(m) RETURN n, m",
            "MATCH (n), (m) WHERE (n)-[:REL1*]->(m) RETURN n, m",
            "MATCH (n), (m) WHERE (n)-[:REL1*]-(m) RETURN n, m",
            "MATCH (n), (m) WHERE (n)-[:REL1*2]-(m) RETURN n, m",
        ],
    )?;

    let trails = trail_fixture()?;
    assert_real_metal_parity(
        &trails,
        &[
            "MATCH (n), (m) WHERE (n)-[:REL1*]->(m) RETURN n, m",
            "MATCH (n), (m) WHERE (n)-[:REL1*]-(m) RETURN n, m",
            "MATCH (n), (m) WHERE (n)-[:REL1*2]-(m) RETURN n, m",
        ],
    )
}
