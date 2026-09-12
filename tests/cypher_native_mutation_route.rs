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
        BindCapabilities, EntityDependency, ExecutionContext, ExecutionOutput, QueryEngine,
        ResultValue,
    },
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend, ResidentGroup,
        ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest, ResidentNodePipelineRequest,
        ResidentNodePipelineResult, ResidentProjectImage, ResidentSortRequest, ResidentSortResult,
        ResidentVectorQuery, ResidentVectorResult, ScratchReservation,
    },
    graph::{
        EdgeInput, GraphMutation, GraphStore, IndexCatalog, LayerMask, NodeInput, TemporalStore,
    },
    types::{LabelId, PropertyId},
};
use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::gpu::MetalBackend;

struct ObservedBackend {
    inner: Box<dyn ExecutionBackend>,
    reported_kind: BackendKind,
    pipeline_calls: Arc<AtomicUsize>,
    drop_mutation_result: bool,
}

impl ObservedBackend {
    fn new<B: ExecutionBackend + 'static>(inner: B, reported_kind: BackendKind) -> Self {
        Self {
            inner: Box::new(inner),
            reported_kind,
            pipeline_calls: Arc::new(AtomicUsize::new(0)),
            drop_mutation_result: false,
        }
    }

    fn missing_mutation_result<B: ExecutionBackend + 'static>(
        inner: B,
        reported_kind: BackendKind,
    ) -> Self {
        Self {
            inner: Box::new(inner),
            reported_kind,
            pipeline_calls: Arc::new(AtomicUsize::new(0)),
            drop_mutation_result: true,
        }
    }

    fn pipeline_calls(&self) -> usize {
        self.pipeline_calls.load(Ordering::SeqCst)
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
        Ok(Box::new(Self {
            inner: self.inner.pin_project(project)?,
            reported_kind: self.reported_kind,
            pipeline_calls: Arc::clone(&self.pipeline_calls),
            drop_mutation_result: self.drop_mutation_result,
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
        self.inner
            .expand_project_out(project, sources, cancellation)
    }

    fn expand_project_in(
        &self,
        project: ProjectId,
        targets: &[u32],
        cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
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
        self.pipeline_calls.fetch_add(1, Ordering::SeqCst);
        let mut result = self.inner.execute_node_pipeline(request, cancellation)?;
        if self.drop_mutation_result {
            result.mutation = None;
        }
        Ok(result)
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
    let person = graph.catalog_mut().intern_label("Person")?;
    let target = graph.catalog_mut().intern_label("Target")?;
    let missing = graph.catalog_mut().intern_label("Missing")?;
    let knows = graph.catalog_mut().intern_relationship_type("KNOWS")?;
    let count = graph.catalog_mut().intern_property("count")?;
    let weight = graph.catalog_mut().intern_property("weight")?;
    graph.insert_node(NodeInput {
        id: NodeId(1),
        layer: Layer::Observed,
        revision: 1,
        labels: vec![person],
        properties: vec![(count, ScalarValue::Integer(5))],
    })?;
    for node in [2, 3] {
        graph.insert_node(NodeInput {
            id: NodeId(node),
            layer: Layer::Observed,
            revision: 1,
            labels: vec![target],
            properties: Vec::new(),
        })?;
    }
    graph.insert_node(NodeInput {
        id: NodeId(4),
        layer: Layer::Observed,
        revision: 1,
        labels: vec![missing],
        properties: Vec::new(),
    })?;
    for (edge, node, value) in [(11, 2, 1), (12, 3, 10)] {
        graph.insert_edge(EdgeInput {
            id: EdgeId(edge),
            source: NodeId(1),
            target: NodeId(node),
            relationship_type: knows,
            layer: Layer::Observed,
            revision: 1,
            properties: vec![(weight, ScalarValue::Integer(value))],
        })?;
    }
    Ok(Fixture {
        graph,
        project: ProjectId(uuid::Uuid::nil()),
        bookmark: Bookmark { term: 7, index: 1 },
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
    let mut backend = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    backend.admit_project(resident_image(fixture)?)?;
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
        parameters: BTreeMap::from([
            (
                "step".to_owned(),
                ResultValue::Scalar(ScalarValue::Integer(1)),
            ),
            (
                "flag".to_owned(),
                ResultValue::Scalar(ScalarValue::Boolean(true)),
            ),
        ]),
        bookmark: fixture.bookmark,
        mutation_revision: fixture.graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id: 100,
        next_edge_id: 100,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            write: true,
            require_native_execution: true,
            ..BindCapabilities::default()
        },
        max_result_rows: 1_024,
        max_batch_rows: 64,
        optimizer_statistics: None,
        backend,
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(30)),
        resolved_query_at_time_nanos: None,
    }
}

const MUTATION_QUERY: &str = "MATCH (n:Person)-[r:KNOWS]->(:Target) \
     WHERE n.count > 0 \
     SET n.count = n.count + $step, \
         r.weight = r.weight + 2, \
         n.native = $flag";

fn mutation_debug(output: &ExecutionOutput) -> Vec<String> {
    output
        .graph_mutations
        .iter()
        .map(|mutation| format!("{mutation:?}"))
        .collect()
}

fn assert_complete_mutation_output(output: &ExecutionOutput) -> Result<()> {
    assert!(output.result.schema.is_empty());
    assert!(output.result.batches.is_empty());
    // The second source row targets the same node with the same `native = true` value. The
    // selected backend receipts classify that repeated target-local assignment as a no-op, so it
    // must not publish a duplicate mutation or increment the effect statistics.
    assert_eq!(output.result.statistics.properties_set, 5);
    let count = output
        .graph_mutations
        .iter()
        .filter_map(|mutation| match mutation {
            GraphMutation::SetNodeProperty {
                node: NodeId(1),
                value,
                ..
            } if matches!(value, ScalarValue::Integer(_)) => Some(value.clone()),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        count,
        vec![ScalarValue::Integer(6), ScalarValue::Integer(7)]
    );
    let weights = output
        .graph_mutations
        .iter()
        .filter_map(|mutation| match mutation {
            GraphMutation::SetEdgeProperty { edge, value, .. } => Some((*edge, value.clone())),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        weights,
        vec![
            (EdgeId(11), ScalarValue::Integer(3)),
            (EdgeId(12), ScalarValue::Integer(12)),
        ]
    );
    let native_property = output
        .graph_mutations
        .iter()
        .find_map(|mutation| match mutation {
            GraphMutation::DeclareProperty { name, id } if name == "native" => Some(*id),
            _ => None,
        });
    let native_property = native_property
        .ok_or_else(|| Error::internal("native mutation did not declare the new property"))?;
    assert_eq!(
        output
            .graph_mutations
            .iter()
            .filter(|mutation| matches!(
                mutation,
                GraphMutation::SetNodeProperty {
                    property,
                    value: ScalarValue::Boolean(true),
                    ..
                } if *property == native_property
            ))
            .count(),
        1
    );
    for entity in [
        EntityDependency::Node(NodeId(1)),
        EntityDependency::Node(NodeId(2)),
        EntityDependency::Node(NodeId(3)),
        EntityDependency::Relationship(EdgeId(11)),
        EntityDependency::Relationship(EdgeId(12)),
    ] {
        assert!(output.dependencies.entities.contains_key(&entity));
    }
    assert_eq!(
        output.dependencies.write_targets,
        [
            EntityDependency::Node(NodeId(1)),
            EntityDependency::Relationship(EdgeId(11)),
            EntityDependency::Relationship(EdgeId(12)),
        ]
        .into_iter()
        .collect()
    );
    Ok(())
}

#[test]
fn cpu_resident_finish_route_matches_the_cpu_semantic_reference() -> Result<()> {
    let fixture = fixture()?;
    let mut host_context = context(&fixture, None);
    let reference = QueryEngine.execute(MUTATION_QUERY, &mut host_context)?;

    let backend = ObservedBackend::new(cpu_backend(&fixture)?, BackendKind::Cpu);
    let mut resident_context = context(&fixture, Some(&backend));
    let resident = QueryEngine.execute(MUTATION_QUERY, &mut resident_context)?;

    assert_eq!(backend.pipeline_calls(), 1);
    assert_eq!(mutation_debug(&resident), mutation_debug(&reference));
    assert_eq!(resident.result.statistics, reference.result.statistics);
    assert_eq!(
        resident.dependencies.entities,
        reference.dependencies.entities
    );
    assert_eq!(
        resident.dependencies.write_targets,
        reference.dependencies.write_targets
    );
    assert_complete_mutation_output(&resident)?;
    assert_eq!(
        fixture
            .graph
            .node(NodeId(1))
            .and_then(|node| node.property(fixture.graph.catalog().property("count")?)),
        Some(ScalarValue::Integer(5))
    );
    Ok(())
}

#[test]
fn optional_null_target_is_a_receipted_no_op_without_declarations() -> Result<()> {
    let fixture = fixture()?;
    let backend = ObservedBackend::new(cpu_backend(&fixture)?, BackendKind::Cpu);
    let mut execution = context(&fixture, Some(&backend));
    let output = QueryEngine.execute(
        "OPTIONAL MATCH (n:Person:Missing) SET n.neverDeclared = $flag",
        &mut execution,
    )?;
    assert_eq!(backend.pipeline_calls(), 1);
    assert!(output.graph_mutations.is_empty());
    assert!(output.result.schema.is_empty());
    assert!(output.result.batches.is_empty());
    assert_eq!(output.result.statistics.properties_set, 0);
    assert!(output.dependencies.write_targets.is_empty());
    Ok(())
}

#[test]
fn cpu_resident_remove_labels_and_map_replacement_match_reference() -> Result<()> {
    let fixture = fixture()?;
    let backend = ObservedBackend::new(cpu_backend(&fixture)?, BackendKind::Cpu);
    for query in [
        "MATCH (n:Person)-[r:KNOWS]->(:Target) REMOVE r.weight, n:Person",
        "MATCH (n:Person) SET n:NativeLabel",
        "MATCH (n:Person) SET n = {fresh: 9}",
    ] {
        let reference = QueryEngine.execute(query, &mut context(&fixture, None))?;
        let resident = QueryEngine.execute(query, &mut context(&fixture, Some(&backend)))?;
        assert_eq!(
            mutation_debug(&resident),
            mutation_debug(&reference),
            "{query}"
        );
        assert_eq!(
            resident.result.statistics, reference.result.statistics,
            "{query}"
        );
        assert_eq!(
            resident.dependencies.entities, reference.dependencies.entities,
            "{query}"
        );
        assert_eq!(
            resident.dependencies.write_targets, reference.dependencies.write_targets,
            "{query}"
        );
    }
    assert_eq!(backend.pipeline_calls(), 3);
    Ok(())
}

#[test]
fn wrong_backend_receipts_fail_before_any_canonical_batch_is_returned() -> Result<()> {
    let fixture = fixture()?;
    // The inner CPU backend emits CpuReference receipts while this fault wrapper reports Metal.
    // Executor publication must reject that provenance mismatch after one dispatch.
    let backend = ObservedBackend::new(cpu_backend(&fixture)?, BackendKind::Metal);
    let mut execution = context(&fixture, Some(&backend));
    let error = QueryEngine
        .execute(MUTATION_QUERY, &mut execution)
        .err()
        .ok_or_else(|| Error::internal("wrong-backend receipts unexpectedly published"))?;
    assert_eq!(backend.pipeline_calls(), 1);
    assert_eq!(error.code, ErrorCode::CorruptStorage);
    assert_eq!(
        fixture
            .graph
            .node(NodeId(1))
            .and_then(|node| node.property(fixture.graph.catalog().property("count")?)),
        Some(ScalarValue::Integer(5))
    );
    Ok(())
}

#[test]
fn missing_raw_mutation_result_fails_before_publication() -> Result<()> {
    let fixture = fixture()?;
    let backend =
        ObservedBackend::missing_mutation_result(cpu_backend(&fixture)?, BackendKind::Cpu);
    let mut execution = context(&fixture, Some(&backend));
    let error = QueryEngine
        .execute(MUTATION_QUERY, &mut execution)
        .err()
        .ok_or_else(|| Error::internal("missing mutation result unexpectedly published"))?;
    assert_eq!(backend.pipeline_calls(), 1);
    assert_eq!(error.code, ErrorCode::CorruptStorage);
    assert_eq!(
        fixture
            .graph
            .node(NodeId(1))
            .and_then(|node| node.property(fixture.graph.catalog().property("count")?)),
        Some(ScalarValue::Integer(5))
    );
    Ok(())
}

#[test]
fn trailing_result_work_is_complete_or_fails_before_dispatch_without_host_fallback() -> Result<()> {
    let fixture = fixture()?;
    // Complete mutation continuations now own these post-write shapes. The inner CPU backend
    // emits honest CPU receipts while this fault wrapper reports Metal, so publication must reject
    // each supported command instead of accepting it or continuing any suffix on the host. A
    // terminal bare WITH and a second mutation clause after WITH remain outside this route and
    // must fail before dispatch.
    let backend = ObservedBackend::new(cpu_backend(&fixture)?, BackendKind::Metal);
    for (query, expected_code, expected_calls) in [
        (
            "MATCH (n:Person) SET n.count = 9 RETURN n.count",
            ErrorCode::CorruptStorage,
            1,
        ),
        (
            "MATCH (n:Person) SET n.count = 9 WITH n",
            ErrorCode::GpuAdmissionFailure,
            0,
        ),
        (
            "MATCH (n:Person) SET n.count = 9 RETURN n LIMIT 0",
            ErrorCode::CorruptStorage,
            1,
        ),
        (
            "MATCH (n:Person) SET n.count = 9 WITH n SET n.count = 10",
            ErrorCode::GpuAdmissionFailure,
            0,
        ),
    ] {
        let calls_before = backend.pipeline_calls();
        let mut execution = context(&fixture, Some(&backend));
        let error = QueryEngine
            .execute(query, &mut execution)
            .err()
            .ok_or_else(|| {
                Error::internal(format!(
                    "unsupported trailing work ran on the host: {query}"
                ))
            })?;
        assert_eq!(error.code, expected_code, "{query}");
        assert_eq!(
            backend.pipeline_calls(),
            calls_before + expected_calls,
            "{query}"
        );
    }
    assert_eq!(backend.pipeline_calls(), 2);
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn metal_test_guard() -> MutexGuard<'static, ()> {
    static METAL_TEST: Mutex<()> = Mutex::new(());
    match METAL_TEST.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_finish_route_matches_cpu_resident_including_null_targets() -> Result<()> {
    let _metal = metal_test_guard();
    let fixture = fixture()?;
    let cpu = ObservedBackend::new(cpu_backend(&fixture)?, BackendKind::Cpu);
    let mut metal_inner = MetalBackend::new(0, 128 * 1024 * 1024, 1024 * 1024)?;
    metal_inner.admit_project(resident_image(&fixture)?)?;
    let metal = ObservedBackend::new(metal_inner, BackendKind::Metal);

    let cpu_output = QueryEngine.execute(MUTATION_QUERY, &mut context(&fixture, Some(&cpu)))?;
    let metal_output = QueryEngine.execute(MUTATION_QUERY, &mut context(&fixture, Some(&metal)))?;
    assert_eq!(mutation_debug(&metal_output), mutation_debug(&cpu_output));
    assert_eq!(metal_output.result.statistics, cpu_output.result.statistics);
    assert_eq!(
        metal_output.dependencies.entities,
        cpu_output.dependencies.entities
    );
    assert_eq!(
        metal_output.dependencies.write_targets,
        cpu_output.dependencies.write_targets
    );
    assert_complete_mutation_output(&metal_output)?;

    let null_query = "OPTIONAL MATCH (n:Person:Missing) SET n.neverDeclared = $flag";
    let cpu_null = QueryEngine.execute(null_query, &mut context(&fixture, Some(&cpu)))?;
    let metal_null = QueryEngine.execute(null_query, &mut context(&fixture, Some(&metal)))?;
    assert!(cpu_null.graph_mutations.is_empty());
    assert_eq!(mutation_debug(&metal_null), mutation_debug(&cpu_null));
    assert_eq!(metal_null.result.statistics, cpu_null.result.statistics);
    assert_eq!(cpu.pipeline_calls(), 2);
    assert_eq!(metal.pipeline_calls(), 2);
    Ok(())
}
