// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Isolated high-level route gate for Aggregation6 [1]-[2].
//!
//! The six official examples must compile into one sealed segmented command whose graph source
//! exports the canonical float-property lane. The observer records both percentile descriptors
//! before forwarding the command to the CPU semantic reference; a successful generic replay is
//! therefore insufficient to satisfy this gate.

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
    cypher::{BindCapabilities, ExecutionContext, ExecutionOutput, QueryEngine, ResultValue},
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend, ResidentGroup,
        ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest, ResidentNodePipelineRequest,
        ResidentNodePipelineResult, ResidentNullableRelationOutputSource,
        ResidentNullableRelationStage, ResidentProjectImage, ResidentSegmentedAggregateKind,
        ResidentSegmentedAggregationOperation, ResidentSegmentedAggregationRequest,
        ResidentSegmentedAggregationResult, ResidentSegmentedAggregationSource,
        ResidentSortRequest, ResidentSortResult, ResidentVectorQuery, ResidentVectorResult,
        ScratchReservation,
    },
    graph::{GraphStore, NodeInput},
    types::{LabelId, Layer, NodeId, PropertyId},
};
use tokio_util::sync::CancellationToken;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());
const MEMORY_LIMIT_BYTES: usize = 256 * 1024 * 1024;
const RESERVED_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct SeenPercentile {
    kind: ResidentSegmentedAggregateKind,
    reduction_percentile: Option<u64>,
    descriptor_percentile: Option<u64>,
    float_property_source: bool,
}

#[derive(Default)]
struct RouteTrace {
    segmented_calls: AtomicUsize,
    other_legacy_calls: AtomicUsize,
    percentiles: Mutex<Vec<SeenPercentile>>,
}

struct ObservedCpuBackend {
    inner: Box<dyn ExecutionBackend>,
    trace: Arc<RouteTrace>,
}

impl ObservedCpuBackend {
    fn new() -> Self {
        Self {
            inner: Box::new(CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES)),
            trace: Arc::new(RouteTrace::default()),
        }
    }

    fn trace(&self) -> Arc<RouteTrace> {
        Arc::clone(&self.trace)
    }

    fn reject<T>(&self, route: &'static str) -> Result<T> {
        self.trace.other_legacy_calls.fetch_add(1, Ordering::SeqCst);
        Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!("percentile route gate rejected `{route}`"),
        ))
    }
}

impl ExecutionBackend for ObservedCpuBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Cpu
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
            trace: Arc::clone(&self.trace),
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
        _layers: irongraph::graph::LayerMask,
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
        self.inner.supports_native_segmented_aggregation()
    }

    fn execute_segmented_aggregation(
        &self,
        request: &ResidentSegmentedAggregationRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentSegmentedAggregationResult> {
        request.validate()?;
        let program = request.program.as_ref().ok_or_else(|| {
            Error::new(
                ErrorCode::GpuAdmissionFailure,
                "percentile route used a legacy host-shaped segmented relation",
            )
        })?;
        if request.input.row_count != 0
            || request.input.column_count != 0
            || !request.input.cells.is_empty()
            || !request.input.arena.is_empty()
        {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "percentile route carried host-shaped source rows",
            ));
        }
        let ResidentSegmentedAggregationSource::GraphRelation {
            request: source, ..
        } = &program.source
        else {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "percentile route omitted its resident graph relation",
            ));
        };
        let float_property_source = matches!(
            source.program.stages.last(),
            Some(ResidentNullableRelationStage::FinalProject { bindings })
                if matches!(
                    bindings.as_slice(),
                    [binding]
                        if matches!(
                            binding.source,
                            ResidentNullableRelationOutputSource::FloatProperty { .. }
                        )
                )
        );
        let reduction = program
            .stages
            .iter()
            .find_map(|stage| match &stage.operation {
                ResidentSegmentedAggregationOperation::Aggregate { reductions, .. } => reductions
                    .as_slice()
                    .first()
                    .filter(|_| reductions.len() == 1),
                _ => None,
            })
            .ok_or_else(|| Error::internal("percentile route omitted its single reduction"))?;
        let [descriptor] = request.aggregates.as_slice() else {
            return Err(Error::internal(
                "percentile route omitted its transitional descriptor",
            ));
        };
        if reduction.kind != descriptor.kind || reduction.percentile != descriptor.percentile {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "sealed and transitional percentile descriptors disagree",
            ));
        }
        self.trace
            .percentiles
            .lock()
            .map_err(|_| Error::internal("percentile route trace lock was poisoned"))?
            .push(SeenPercentile {
                kind: reduction.kind,
                reduction_percentile: reduction.percentile,
                descriptor_percentile: descriptor.percentile,
                float_property_source,
            });
        self.trace.segmented_calls.fetch_add(1, Ordering::SeqCst);
        self.inner
            .execute_segmented_aggregation(request, cancellation)
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

fn fixture_graph() -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    let price = graph.catalog_mut().intern_property("price")?;
    for (id, value) in [(1, 10.0), (2, 20.0), (3, 30.0)] {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: id,
            labels: Vec::new(),
            properties: vec![(price, ScalarValue::Float(value.into()))],
        })?;
    }
    Ok(graph)
}

fn context<'a>(
    graph: &'a GraphStore,
    backend: Option<&'a dyn ExecutionBackend>,
    percentile: f64,
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
        parameters: BTreeMap::from([(
            "percentile".to_owned(),
            ResultValue::Scalar(ScalarValue::Float(percentile.into())),
        )]),
        bookmark: Bookmark {
            term: 0,
            index: graph.revision(),
        },
        mutation_revision: graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id: 4,
        next_edge_id: 1,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            require_native_execution,
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

fn only_float(output: &ExecutionOutput) -> Result<f64> {
    if output.result.schema.len() != 1 || output.result.schema[0].0 != "p" {
        return Err(Error::internal("percentile output schema changed"));
    }
    let values = output
        .result
        .batches
        .iter()
        .flat_map(|batch| batch.columns[0].values.iter())
        .collect::<Vec<_>>();
    let [ResultValue::Scalar(ScalarValue::Float(value))] = values.as_slice() else {
        return Err(Error::internal(
            "percentile output was not one FLOAT scalar",
        ));
    };
    Ok(value.0)
}

#[test]
fn aggregation6_exact_six_percentiles_use_one_float_graph_command_each() -> Result<()> {
    let graph = fixture_graph()?;
    let mut backend = ObservedCpuBackend::new();
    backend.admit_project(ResidentProjectImage::graph_only(Arc::new(
        graph.snapshot()?,
    )))?;
    let trace = backend.trace();
    let cases: [(usize, &str, ResidentSegmentedAggregateKind, f64, f64); 6] = [
        (
            1269,
            "percentileDisc",
            ResidentSegmentedAggregateKind::PercentileDisc,
            0.0,
            10.0,
        ),
        (
            1270,
            "percentileDisc",
            ResidentSegmentedAggregateKind::PercentileDisc,
            0.5,
            20.0,
        ),
        (
            1271,
            "percentileDisc",
            ResidentSegmentedAggregateKind::PercentileDisc,
            1.0,
            30.0,
        ),
        (
            1272,
            "percentileCont",
            ResidentSegmentedAggregateKind::PercentileCont,
            0.0,
            10.0,
        ),
        (
            1273,
            "percentileCont",
            ResidentSegmentedAggregateKind::PercentileCont,
            0.5,
            20.0,
        ),
        (
            1274,
            "percentileCont",
            ResidentSegmentedAggregateKind::PercentileCont,
            1.0,
            30.0,
        ),
    ];

    for (report_index, function, _, percentile, expected) in cases {
        let query = format!("MATCH (n) RETURN {function}(n.price, $percentile) AS p");
        let output = QueryEngine
            .execute(
                &query,
                &mut context(&graph, Some(&backend), percentile, true),
            )
            .map_err(|error| {
                Error::new(
                    error.code,
                    format!("Aggregation6 report index {report_index} failed: {error}"),
                )
            })?;
        assert_eq!(only_float(&output)?.to_bits(), expected.to_bits());
    }

    assert_eq!(trace.segmented_calls.load(Ordering::SeqCst), cases.len());
    assert_eq!(trace.other_legacy_calls.load(Ordering::SeqCst), 0);
    let seen = trace
        .percentiles
        .lock()
        .map_err(|_| Error::internal("percentile route trace lock was poisoned"))?;
    assert_eq!(seen.len(), cases.len());
    for (actual, (_, _, kind, percentile, _)) in seen.iter().zip(cases) {
        assert_eq!(
            *actual,
            SeenPercentile {
                kind,
                reduction_percentile: Some(percentile.to_bits()),
                descriptor_percentile: Some(percentile.to_bits()),
                float_property_source: true,
            }
        );
    }
    drop(seen);

    // The native compiler intentionally accepts only the exact immutable-parameter shape. A
    // literal percentile remains semantically valid through the ordinary executor but must not
    // increment the native command count.
    let output = QueryEngine.execute(
        "MATCH (n) RETURN percentileDisc(n.price, 0.5) AS p",
        &mut context(&graph, None, 0.5, false),
    )?;
    assert_eq!(only_float(&output)?.to_bits(), 20.0_f64.to_bits());
    assert_eq!(trace.segmented_calls.load(Ordering::SeqCst), cases.len());

    let error = QueryEngine
        .execute(
            "MATCH (n) RETURN percentileCont(n.price, $percentile) AS p",
            &mut context(&graph, Some(&backend), 1.1, false),
        )
        .expect_err("out-of-range percentile unexpectedly succeeded");
    assert_eq!(error.code, ErrorCode::QueryType);
    assert_eq!(
        error.message,
        "NumberOutOfRange: percentile must be finite and in 0..=1"
    );
    assert_eq!(trace.segmented_calls.load(Ordering::SeqCst), cases.len());
    Ok(())
}
