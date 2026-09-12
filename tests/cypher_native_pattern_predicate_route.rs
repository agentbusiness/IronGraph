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
        ResidentNodePipelineResult, ResidentPatternPredicateInput, ResidentPatternPredicateRequest,
        ResidentPatternPredicateResult, ResidentProjectImage, ResidentSortRequest,
        ResidentSortResult, ResidentVectorQuery, ResidentVectorResult, ScratchReservation,
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
    pattern_calls: AtomicUsize,
    node_pipeline_calls: AtomicUsize,
    scan_calls: AtomicUsize,
    adjacency_calls: AtomicUsize,
    last_pattern_request: Mutex<Option<ResidentPatternPredicateRequest>>,
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

    /// Fault-injection wrapper only: the inner CPU completion is deliberately validated as the
    /// supplied kind. This is not a Metal test.
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
            .pattern_calls
            .fetch_add(1, Ordering::SeqCst);
        *self
            .observations
            .last_pattern_request
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(request.clone());
        self.inner.execute_pattern_predicate(request, cancellation)
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

fn fixture() -> Result<Fixture> {
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
    let mut backend = CpuBackend::new(64 * 1024 * 1024, 16 * 1024 * 1024);
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
        next_node_id: 100,
        next_edge_id: 100,
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

fn node_ids(output: &ExecutionOutput, column: &str) -> Result<Vec<NodeId>> {
    let mut ids = Vec::new();
    for batch in &output.result.batches {
        let values = batch
            .columns
            .iter()
            .find(|candidate| candidate.name == column)
            .ok_or_else(|| Error::internal(format!("result omitted `{column}`")))?;
        for value in &values.values {
            let ResultValue::Node(node) = value else {
                return Err(Error::internal(format!(
                    "result `{column}` contained a non-node value"
                )));
            };
            ids.push(node.id);
        }
    }
    Ok(ids)
}

fn execute_ids(
    fixture: &Fixture,
    backend: Option<&dyn ExecutionBackend>,
    require_native_execution: bool,
    query: &str,
) -> Result<Vec<NodeId>> {
    let output = QueryEngine.execute(
        query,
        &mut context(fixture, backend, require_native_execution),
    )?;
    node_ids(&output, "n")
}

#[test]
fn fused_pattern1_route_uses_one_call_and_no_scan_pipeline_or_host_adjacency() -> Result<()> {
    let fixture = fixture()?;
    let backend = ObservedBackend::new(cpu_backend(&fixture)?);
    let observations = backend.observations();
    let query = "MATCH (n) WHERE (n)-[:REL1]-() RETURN n";
    let output = QueryEngine.execute(query, &mut context(&fixture, Some(&backend), true))?;

    assert_eq!(
        node_ids(&output, "n")?,
        vec![NodeId(1), NodeId(2), NodeId(4)]
    );
    assert_eq!(observations.pins.load(Ordering::SeqCst), 1);
    assert_eq!(observations.pattern_calls.load(Ordering::SeqCst), 1);
    assert_eq!(observations.node_pipeline_calls.load(Ordering::SeqCst), 0);
    assert_eq!(observations.scan_calls.load(Ordering::SeqCst), 0);
    assert_eq!(observations.adjacency_calls.load(Ordering::SeqCst), 0);

    let request = observations
        .last_pattern_request
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
        .ok_or_else(|| Error::internal("fused request was not observed"))?;
    assert_eq!(request.expected_bookmark, fixture.bookmark);
    assert_eq!(request.expected_graph_revision, fixture.graph.revision());
    assert_eq!(
        request.expected_layout_version,
        fixture.graph.layout_version()
    );
    assert_eq!(request.max_output_rows, fixture.graph.node_slot_count());
    let ResidentPatternPredicateInput::VisibleNodeScan {
        node_slots,
        obligation,
    } = request.input
    else {
        return Err(Error::internal("executor supplied host Pattern1 rows"));
    };
    assert_eq!(node_slots, fixture.graph.node_slot_count());
    assert_eq!(obligation.id, 1);
    assert_eq!(
        request
            .program
            .obligations()
            .map(|item| item.id)
            .collect::<Vec<_>>(),
        vec![2, 3]
    );
    Ok(())
}

#[test]
fn fused_cpu_matches_backend_free_oracle_for_boolean_and_length_shapes() -> Result<()> {
    let fixture = fixture()?;
    let backend = cpu_backend(&fixture)?;
    for query in [
        "MATCH (n) WHERE (n)-[]->() RETURN n",
        "MATCH (n) WHERE NOT (n)-[:REL2]-() RETURN n",
        "MATCH (n) WHERE (n)-[:REL1]-() AND (n)-[:REL3]-() RETURN n",
        "MATCH (n) WHERE (n)-[:REL1]-() OR (n)-[:REL2]-() RETURN n",
        "MATCH (n) WHERE (n)-[:REL1*2]-() RETURN n",
    ] {
        let oracle = execute_ids(&fixture, None, false, query)?;
        let resident = execute_ids(&fixture, Some(&backend), true, query)?;
        assert_eq!(resident, oracle, "query: {query}");
    }
    Ok(())
}

#[test]
fn optimizer_split_conjunction_is_recombined_with_ordered_obligations() -> Result<()> {
    let fixture = fixture()?;
    let backend = ObservedBackend::new(cpu_backend(&fixture)?);
    let observations = backend.observations();
    let query = "MATCH (n) WHERE (n)-[:REL1]-() AND (n)-[:REL3]-() RETURN n";
    let output = QueryEngine.execute(query, &mut context(&fixture, Some(&backend), true))?;
    assert_eq!(node_ids(&output, "n")?, vec![NodeId(1)]);

    let request = observations
        .last_pattern_request
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
        .ok_or_else(|| Error::internal("conjunction request was not observed"))?;
    assert_eq!(request.program.leaves.len(), 2);
    assert_eq!(
        request
            .obligations()
            .map(|item| item.id)
            .collect::<Vec<_>>(),
        vec![1, 2, 3, 4]
    );
    Ok(())
}

#[test]
fn unsupported_whole_plan_shapes_make_zero_fused_calls() -> Result<()> {
    let fixture = fixture()?;
    for query in [
        "MATCH (n:A) WHERE (n)-[]->() RETURN n",
        "MATCH (n) WHERE (n)-[]->() RETURN n AS renamed",
        "MATCH (n) WHERE (n)-[]->() RETURN DISTINCT n",
        "MATCH (n) WHERE (n)-[]->() RETURN n LIMIT 1",
    ] {
        let backend = ObservedBackend::new(cpu_backend(&fixture)?);
        let observations = backend.observations();
        QueryEngine.execute(query, &mut context(&fixture, Some(&backend), false))?;
        assert_eq!(
            observations.pattern_calls.load(Ordering::SeqCst),
            0,
            "query: {query}"
        );
    }
    Ok(())
}

#[test]
fn stale_snapshot_or_layout_emits_no_stream_items() -> Result<()> {
    let query = "MATCH (n) WHERE (n)-[:REL1]-() RETURN n";

    let bookmark_fixture = fixture()?;
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
    assert_eq!(observations.pattern_calls.load(Ordering::SeqCst), 0);

    let mut revision_fixture = fixture()?;
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
    assert_eq!(observations.pattern_calls.load(Ordering::SeqCst), 0);

    let mut layout_fixture = fixture()?;
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
    assert_eq!(observations.pattern_calls.load(Ordering::SeqCst), 1);
    Ok(())
}

#[test]
fn stale_canonical_catalog_emits_nothing_before_fused_dispatch() -> Result<()> {
    let fixture = fixture()?;
    let backend = ObservedBackend::new(cpu_backend(&fixture)?);
    let observations = backend.observations();
    let mut stale_catalog = fixture.graph.catalog().clone();
    stale_catalog.intern_property("stale_only")?;
    let mut execution = context(&fixture, Some(&backend), true);
    execution.binding_catalog = &stale_catalog;
    let mut emitted = 0_usize;
    let error = QueryEngine
        .execute_streaming(
            "MATCH (n) WHERE (n)-[:REL1]-() RETURN n",
            &mut execution,
            &mut |_| {
                emitted += 1;
                Ok(())
            },
        )
        .expect_err("stale binding catalog must fail");
    assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
    assert_eq!(emitted, 0);
    assert_eq!(observations.pattern_calls.load(Ordering::SeqCst), 0);
    Ok(())
}

#[test]
fn fused_result_budget_fails_without_truncation_or_stream_output() -> Result<()> {
    let fixture = fixture()?;
    let backend = ObservedBackend::new(cpu_backend(&fixture)?);
    let observations = backend.observations();
    let mut execution = context(&fixture, Some(&backend), true);
    execution.max_result_rows = 1;
    let mut emitted = 0_usize;
    let error = QueryEngine
        .execute_streaming(
            "MATCH (n) WHERE (n)-[:REL1]-() RETURN n",
            &mut execution,
            &mut |_| {
                emitted += 1;
                Ok(())
            },
        )
        .expect_err("resident Pattern1 output must not be truncated");
    assert_eq!(error.code, ErrorCode::ResultBudgetExceeded);
    assert_eq!(emitted, 0);
    assert_eq!(observations.pattern_calls.load(Ordering::SeqCst), 1);
    let request = observations
        .last_pattern_request
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
        .ok_or_else(|| Error::internal("bounded fused request was not observed"))?;
    assert_eq!(request.max_output_rows, 1);
    Ok(())
}

#[test]
fn corrupt_completion_receipts_do_not_fallback_or_emit() -> Result<()> {
    let fixture = fixture()?;
    let backend = ObservedBackend::reporting(cpu_backend(&fixture)?, BackendKind::Metal);
    let observations = backend.observations();
    let mut emitted = 0_usize;
    let error = QueryEngine
        .execute_streaming(
            "MATCH (n) WHERE (n)-[:REL1]-() RETURN n",
            &mut context(&fixture, Some(&backend), true),
            &mut |_| {
                emitted += 1;
                Ok(())
            },
        )
        .expect_err("CPU receipts reported as Metal must fail validation");
    assert_eq!(error.code, ErrorCode::CorruptStorage);
    assert_eq!(emitted, 0);
    assert_eq!(observations.pattern_calls.load(Ordering::SeqCst), 1);
    assert_eq!(observations.node_pipeline_calls.load(Ordering::SeqCst), 0);
    assert_eq!(observations.scan_calls.load(Ordering::SeqCst), 0);
    assert_eq!(observations.adjacency_calls.load(Ordering::SeqCst), 0);
    Ok(())
}

#[test]
fn fused_pattern1_records_predicate_and_returned_node_dependencies() -> Result<()> {
    let fixture = fixture()?;
    let backend = cpu_backend(&fixture)?;
    let output = QueryEngine.execute(
        "MATCH (n) WHERE (n)-[:REL1]-() AND (n)-[:REL3]-() RETURN n",
        &mut context(&fixture, Some(&backend), true),
    )?;
    assert_eq!(node_ids(&output, "n")?, vec![NodeId(1)]);
    assert_eq!(output.dependencies.entities.len(), 1);
    assert!(
        output
            .dependencies
            .entities
            .contains_key(&EntityDependency::Node(NodeId(1)))
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
    assert!(output.dependencies.predicates.keys().any(|dependency| {
        dependency.kind == DependencyKind::RelationshipType && dependency.name == "REL3"
    }));
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
#[ignore = "requires an available physical Metal device"]
fn real_metal_fused_pattern1_matches_cpu() -> Result<()> {
    let _guard = metal_test_guard();
    let fixture = fixture()?;
    let image = resident_image(&fixture)?;
    let mut cpu = CpuBackend::new(128 * 1024 * 1024, 32 * 1024 * 1024);
    let mut metal = MetalBackend::new(0, 128 * 1024 * 1024, 32 * 1024 * 1024)?;
    cpu.admit_project(image.clone())?;
    metal.admit_project(image)?;

    for query in [
        "MATCH (n) WHERE NOT (n)-[:REL2]-() RETURN n",
        "MATCH (n) WHERE (n)-[:REL1]-() OR (n)-[:REL2]-() RETURN n",
        "MATCH (n) WHERE (n)-[:REL1*2]-() RETURN n",
    ] {
        let cpu_ids = execute_ids(&fixture, Some(&cpu), true, query)?;
        let metal_ids = execute_ids(&fixture, Some(&metal), true, query)?;
        assert_eq!(metal_ids, cpu_ids, "query: {query}");
    }
    Ok(())
}
