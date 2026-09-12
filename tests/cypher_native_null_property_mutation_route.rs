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
    Bookmark, Error, ErrorCode, Layer, NodeId, ProjectId, Result, ScalarValue,
    cypher::{
        BindCapabilities, ColumnType, ExecutionContext, QueryEngine, ResultValue, StatementStats,
    },
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend, ResidentGroup,
        ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest, ResidentNodePipelineRequest,
        ResidentNodePipelineResult, ResidentNullableNodeDomain,
        ResidentNullableRelationBindingKind, ResidentNullableRelationMatchMode,
        ResidentNullableRelationOutputSource, ResidentNullableRelationRequest,
        ResidentNullableRelationResult, ResidentNullableRelationStage, ResidentProjectImage,
        ResidentRowMutationRequest, ResidentRowMutationResult, ResidentRowProgramRequest,
        ResidentRowProgramResult, ResidentSortRequest, ResidentSortResult,
        ResidentVariablePathRequest, ResidentVariablePathResult, ResidentVectorQuery,
        ResidentVectorResult, ScratchReservation,
    },
    graph::{GraphStore, IndexCatalog, LayerMask, NodeInput, TemporalStore},
    types::{LabelId, PropertyId},
};
use tokio_util::sync::CancellationToken;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::from_u128(
    0x4e55_4c4c_5f50_524f_5045_5254_5900_0001,
));
const BOOKMARK: Bookmark = Bookmark { term: 41, index: 5 };
const MEMORY_LIMIT_BYTES: usize = 128 * 1024 * 1024;
const RESERVED_BYTES: usize = 16 * 1024 * 1024;

#[derive(Default)]
struct Observations {
    pins: AtomicUsize,
    nullable_calls: AtomicUsize,
    mutation_calls: AtomicUsize,
    forbidden_calls: AtomicUsize,
    requests: Mutex<Vec<ResidentNullableRelationRequest>>,
}

struct StrictNullPropertyMutationBackend {
    inner: Box<dyn ExecutionBackend>,
    pinned: bool,
    observations: Arc<Observations>,
}

impl StrictNullPropertyMutationBackend {
    fn new(graph: &GraphStore) -> Result<Self> {
        let image = ResidentProjectImage::build(
            PROJECT,
            BOOKMARK,
            graph,
            &TemporalStore::default(),
            &IndexCatalog::default(),
        )?;
        let mut inner = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
        inner.admit_project(image)?;
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
            format!("strict null-property mutation test rejected `{route}`"),
        ))
    }
}

impl ExecutionBackend for StrictNullPropertyMutationBackend {
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
        let inner = self.inner.pin_project(project)?;
        self.observations.pins.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Self {
            inner,
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

    fn execute_row_mutation(
        &self,
        _request: &ResidentRowMutationRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentRowMutationResult> {
        self.observations
            .mutation_calls
            .fetch_add(1, Ordering::SeqCst);
        self.reject("execute_row_mutation")
    }

    fn execute_row_program(
        &self,
        _request: &ResidentRowProgramRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentRowProgramResult> {
        self.reject("execute_row_program")
    }

    fn execute_nullable_relation(
        &self,
        request: &ResidentNullableRelationRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentNullableRelationResult> {
        if !self.pinned {
            return self.reject("execute_nullable_relation_unpinned");
        }
        request.validate()?;
        self.observations
            .nullable_calls
            .fetch_add(1, Ordering::SeqCst);
        self.observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(request.clone());
        self.inner.execute_nullable_relation(request, cancellation)
    }

    fn execute_variable_path(
        &self,
        _request: &ResidentVariablePathRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentVariablePathResult> {
        self.reject("execute_variable_path")
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

fn context<'a>(graph: &'a GraphStore, backend: &'a dyn ExecutionBackend) -> ExecutionContext<'a> {
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
        next_node_id: 1,
        next_edge_id: 1,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            write: true,
            require_native_execution: true,
            ..BindCapabilities::default()
        },
        max_result_rows: 32,
        max_batch_rows: 32,
        optimizer_statistics: None,
        backend: Some(backend),
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(30)),
        resolved_query_at_time_nanos: None,
    }
}

fn assert_null_entity_result(output: &irongraph::cypher::ExecutionOutput) -> Result<()> {
    assert_eq!(output.result.schema, [("a".to_owned(), ColumnType::Null)]);
    let [batch] = output.result.batches.as_slice() else {
        return Err(Error::internal(
            "known-null mutation changed result batch count",
        ));
    };
    assert_eq!(batch.row_count, 1);
    assert!(matches!(
        batch.columns.as_slice(),
        [column]
            if column.name == "a"
                && column.value_type == ColumnType::Null
                && column.values == [ResultValue::Scalar(irongraph::ScalarValue::Null)]
    ));
    assert_eq!(output.result.statistics, StatementStats::default());
    assert!(output.graph_mutations.is_empty());
    assert!(output.temporal_mutations.is_empty());
    Ok(())
}

#[test]
fn remove1_5_and_set1_8_erase_only_the_proven_null_mutation() -> Result<()> {
    for (report_index, query) in [
        (665, "OPTIONAL MATCH (a:DoesNotExist) REMOVE a.num RETURN a"),
        (
            830,
            "OPTIONAL MATCH (a:DoesNotExist) SET a.num = 42 RETURN a",
        ),
    ] {
        let graph = GraphStore::default();
        let backend = StrictNullPropertyMutationBackend::new(&graph)?;
        let observations = backend.observations();
        let output = QueryEngine.execute(query, &mut context(&graph, &backend))?;
        assert_null_entity_result(&output)?;
        assert_eq!(
            observations.pins.load(Ordering::SeqCst),
            1,
            "{report_index}"
        );
        assert_eq!(
            observations.nullable_calls.load(Ordering::SeqCst),
            1,
            "{report_index}"
        );
        assert_eq!(
            observations.mutation_calls.load(Ordering::SeqCst),
            0,
            "{report_index}"
        );
        assert_eq!(
            observations.forbidden_calls.load(Ordering::SeqCst),
            0,
            "{report_index}"
        );

        let requests = observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let [request] = requests.as_slice() else {
            return Err(Error::internal(
                "known-null mutation changed native request count",
            ));
        };
        request.validate()?;
        let [
            ResidentNullableRelationStage::NodeScan {
                mode: ResidentNullableRelationMatchMode::Optional,
                output: node,
                labels: ResidentNullableNodeDomain::KnownEmpty,
            },
            ResidentNullableRelationStage::FinalProject { bindings },
        ] = request.program.stages.as_slice()
        else {
            return Err(Error::internal(
                "known-null mutation changed sealed nullable stage shape",
            ));
        };
        assert!(matches!(
            bindings.as_slice(),
            [binding]
                if binding.name == "a"
                    && binding.source
                        == (ResidentNullableRelationOutputSource::Entity {
                            slot: *node,
                            kind: ResidentNullableRelationBindingKind::Node,
                        })
        ));
        assert!(request.predicate_program.is_empty());
        assert!(request.property_lanes().is_empty());
    }
    Ok(())
}

#[test]
fn known_null_property_noop_near_misses_fail_closed() -> Result<()> {
    let mut declared = GraphStore::default();
    let existing = declared.catalog_mut().intern_label("Existing")?;
    let num = declared.catalog_mut().intern_property("num")?;
    declared.insert_node(NodeInput {
        id: NodeId(1),
        layer: Layer::Observed,
        revision: 1,
        labels: vec![existing],
        properties: vec![(num, ScalarValue::Integer(7))],
    })?;

    // These exact one-column neighbors isolate the catalog-known-null proof. The ordinary native
    // mutation compiler is allowed to inspect or dispatch them, but the nullable no-op route must
    // never consume a real matching node and erase its SET/REMOVE effect.
    for query in [
        "OPTIONAL MATCH (a:Existing) SET a.num = 42 RETURN a",
        "OPTIONAL MATCH (a:Existing) REMOVE a.num RETURN a",
    ] {
        let backend = StrictNullPropertyMutationBackend::new(&declared)?;
        let observations = backend.observations();
        let error = QueryEngine
            .execute(query, &mut context(&declared, &backend))
            .expect_err("catalog-present mutation neighbor must fail closed in the strict harness");
        assert_eq!(error.code, ErrorCode::GpuAdmissionFailure, "{query}");
        assert_eq!(
            observations.nullable_calls.load(Ordering::SeqCst),
            0,
            "{query}"
        );
        assert!(
            observations
                .requests
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .is_empty(),
            "{query}"
        );
    }

    for (graph, query) in [
        (
            &declared,
            "OPTIONAL MATCH (a:DoesNotExist) SET a.num = rand() RETURN a",
        ),
        (
            &declared,
            "OPTIONAL MATCH (a:DoesNotExist) SET a.num = 42 RETURN a, labels(a)",
        ),
        (
            &declared,
            "OPTIONAL MATCH (a:DoesNotExist) REMOVE a.num RETURN a, labels(a)",
        ),
    ] {
        let backend = StrictNullPropertyMutationBackend::new(graph)?;
        let observations = backend.observations();
        let error = QueryEngine
            .execute(query, &mut context(graph, &backend))
            .expect_err("unsupported known-null mutation neighbor must fail closed");
        assert_eq!(error.code, ErrorCode::GpuAdmissionFailure, "{query}");
        assert_eq!(observations.pins.load(Ordering::SeqCst), 0, "{query}");
        assert_eq!(
            observations.nullable_calls.load(Ordering::SeqCst),
            0,
            "{query}"
        );
        assert_eq!(
            observations.mutation_calls.load(Ordering::SeqCst),
            0,
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
