// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Strict native-route proof for TypeConversion4 [4].
//!
//! The successful path permits one immutable project pin and one complete typed-row program.
//! Every decomposed query primitive is poisoned, so host evaluation cannot manufacture the
//! Boolean spelling or silently repair NULL validity.

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
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend, ResidentEntityBinding,
        ResidentGroup, ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest,
        ResidentNodeBinding, ResidentNodePipelineRequest, ResidentNodePipelineResult,
        ResidentProjectImage, ResidentRowColumn, ResidentRowInstruction, ResidentRowOperation,
        ResidentRowProgram, ResidentRowProgramManifest, ResidentRowProgramRequest,
        ResidentRowProgramResult, ResidentRowValueType, ResidentSortRequest, ResidentSortResult,
        ResidentVectorQuery, ResidentVectorResult, ScratchReservation,
    },
    graph::{GraphStore, IndexCatalog, LayerMask, NodeInput, TemporalStore},
    types::{LabelId, PropertyId},
};
use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::gpu::MetalBackend;
#[cfg(all(feature = "accelerator", target_os = "macos"))]
use std::sync::{MutexGuard, OnceLock};

const PROJECT: ProjectId = ProjectId(uuid::Uuid::from_u128(
    0x5459_5045_434f_4e56_345f_424f_4f4c_0004,
));
const QUERY: &str = "MATCH (m:Movie) RETURN toString(m.watched)";
const REPORT_INDEX: usize = 3_856;
const MEMORY_LIMIT_BYTES: usize = 128 * 1024 * 1024;
const RESERVED_BYTES: usize = 8 * 1024 * 1024;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn metal_test_guard() -> MutexGuard<'static, ()> {
    static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    GUARD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

struct Fixture {
    graph: GraphStore,
    movie: LabelId,
    watched: PropertyId,
}

impl Fixture {
    fn new() -> Result<Self> {
        let mut graph = GraphStore::default();
        let movie = graph.catalog_mut().intern_label("Movie")?;
        let watched = graph.catalog_mut().intern_property("watched")?;
        let rating = graph.catalog_mut().intern_property("rating")?;
        let title = graph.catalog_mut().intern_property("title")?;
        for (id, properties) in [
            (
                1_u64,
                vec![
                    (watched, ScalarValue::Boolean(true)),
                    (rating, ScalarValue::Integer(4)),
                    (title, ScalarValue::String("first".into())),
                ],
            ),
            (2, vec![(watched, ScalarValue::Boolean(false))]),
            (3, Vec::new()),
        ] {
            graph.insert_node(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: id,
                labels: vec![movie],
                properties,
            })?;
        }
        Ok(Self {
            graph,
            movie,
            watched,
        })
    }

    fn image(&self) -> Result<ResidentProjectImage> {
        ResidentProjectImage::build(
            PROJECT,
            bookmark(&self.graph),
            &self.graph,
            &TemporalStore::default(),
            &IndexCatalog::default(),
        )
    }

    fn strict_cpu(&self) -> Result<StrictRowBackend> {
        let mut inner = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
        inner.admit_project(self.image()?)?;
        Ok(StrictRowBackend::new(Box::new(inner)))
    }

    #[cfg(all(feature = "accelerator", target_os = "macos"))]
    fn strict_metal(&self) -> Result<StrictRowBackend> {
        let mut inner = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
        inner.admit_project(self.image()?)?;
        Ok(StrictRowBackend::new(Box::new(inner)))
    }
}

#[derive(Default)]
struct Observations {
    pins: AtomicUsize,
    row_program_calls: AtomicUsize,
    forbidden_calls: AtomicUsize,
    requests: Mutex<Vec<ResidentRowProgramRequest>>,
    results: Mutex<Vec<ResidentRowProgramResult>>,
}

/// Reports Metal before pinning so required-native execution cannot enter the generic CPU path.
/// After pinning, the inner backend reports its honest completion kind and owns all semantics.
struct StrictRowBackend {
    inner: Box<dyn ExecutionBackend>,
    pinned: bool,
    observations: Arc<Observations>,
}

impl StrictRowBackend {
    fn new(inner: Box<dyn ExecutionBackend>) -> Self {
        Self {
            inner,
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
            format!("strict Boolean-to-string backend rejected route `{route}`"),
        ))
    }
}

impl ExecutionBackend for StrictRowBackend {
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
        self.observations.pins.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Self {
            inner: self.inner.pin_project(project)?,
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

    fn execute_row_program(
        &self,
        request: &ResidentRowProgramRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentRowProgramResult> {
        if !self.pinned {
            return self.reject("execute_row_program_unpinned");
        }
        request.validate()?;
        self.observations
            .row_program_calls
            .fetch_add(1, Ordering::SeqCst);
        self.observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(request.clone());
        let result = self.inner.execute_row_program(request, cancellation)?;
        self.observations
            .results
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(result.clone());
        Ok(result)
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

fn bookmark(graph: &GraphStore) -> Bookmark {
    Bookmark {
        term: 84,
        index: graph.revision(),
    }
}

fn context<'a>(
    graph: &'a GraphStore,
    backend: Option<&'a dyn ExecutionBackend>,
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
        parameters: BTreeMap::new(),
        bookmark: bookmark(graph),
        mutation_revision: graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id: 100,
        next_edge_id: 100,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            write: false,
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

fn result_rows(output: &irongraph::cypher::ExecutionOutput) -> Vec<Vec<ResultValue>> {
    output
        .result
        .batches
        .iter()
        .flat_map(|batch| {
            (0..batch.row_count).map(|row| {
                batch
                    .columns
                    .iter()
                    .map(|column| column.values[row].clone())
                    .collect()
            })
        })
        .collect()
}

fn forged_null_payload_request(
    base: &ResidentRowProgramRequest,
) -> Result<ResidentRowProgramRequest> {
    let mut request = base.clone();
    request.input.labels.clear();
    request.input.max_output_rows = 1;
    request.program = ResidentRowProgram {
        instructions: vec![
            ResidentRowInstruction {
                output_type: ResidentRowValueType::Boolean,
                operation: ResidentRowOperation::InputColumn(ResidentRowColumn::Boolean {
                    values: vec![1],
                    validity: vec![0],
                }),
            },
            ResidentRowInstruction {
                output_type: ResidentRowValueType::String,
                operation: ResidentRowOperation::BooleanToString { operand: 0 },
            },
        ],
    };
    request.sort_keys.clear();
    request.offset = 0;
    request.limit = usize::MAX;
    request.max_output_rows = 1;
    request.final_registers = vec![1];
    request.manifest = ResidentRowProgramManifest::build(
        &request.program,
        &request.sort_keys,
        request.offset,
        request.limit,
        request.max_output_rows,
        &request.final_registers,
        90_001,
    )?;
    request.validate()?;
    Ok(request)
}

fn assert_strict_route(fixture: &Fixture, backend: &StrictRowBackend) -> Result<()> {
    let reference = QueryEngine.execute(QUERY, &mut context(&fixture.graph, None, false))?;
    let observations = backend.observations();
    let native = QueryEngine.execute(QUERY, &mut context(&fixture.graph, Some(backend), true))?;

    assert_eq!(native.result, reference.result, "report {REPORT_INDEX}");
    assert_eq!(
        native.result.schema,
        [("toString(m.watched)".to_owned(), ColumnType::String)]
    );
    assert_eq!(
        result_rows(&native),
        [
            vec![ResultValue::Scalar(ScalarValue::String("true".into()))],
            vec![ResultValue::Scalar(ScalarValue::String("false".into()))],
            vec![ResultValue::Scalar(ScalarValue::Null)],
        ]
    );
    assert_eq!(native.result.statistics, StatementStats::default());
    assert!(native.graph_mutations.is_empty());
    assert!(native.temporal_mutations.is_empty());
    assert_eq!(observations.pins.load(Ordering::SeqCst), 1);
    assert_eq!(observations.row_program_calls.load(Ordering::SeqCst), 1);
    assert_eq!(observations.forbidden_calls.load(Ordering::SeqCst), 0);

    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let [request] = requests.as_slice() else {
        return Err(Error::internal(
            "TypeConversion4 [4] changed typed-row request cardinality",
        ));
    };
    let forged_null = forged_null_payload_request(request)?;
    request.validate()?;
    assert_eq!(request.input.labels, [fixture.movie]);
    assert!(request.sort_keys.is_empty());
    assert_eq!(request.final_registers, [1]);
    assert_eq!(request.program.string_register_capacity(1)?, Some(5));
    assert_eq!(
        request.program.instructions,
        [
            ResidentRowInstruction {
                output_type: ResidentRowValueType::Boolean,
                operation: ResidentRowOperation::LoadBooleanProperty {
                    binding: ResidentEntityBinding::Node(ResidentNodeBinding::Start),
                    property: fixture.watched,
                },
            },
            ResidentRowInstruction {
                output_type: ResidentRowValueType::String,
                operation: ResidentRowOperation::BooleanToString { operand: 0 },
            },
        ]
    );
    assert_eq!(request.manifest.instruction_obligations.len(), 2);
    assert_ne!(request.manifest.fingerprint.0, [0; 32]);

    let results = observations
        .results
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let [result] = results.as_slice() else {
        return Err(Error::internal(
            "TypeConversion4 [4] changed typed-row result cardinality",
        ));
    };
    let parts = result.clone().into_untrusted_parts();
    assert_eq!(parts.scratch_bytes, request.scratch_bytes(3)?);
    assert_eq!(parts.receipts.len(), 3);
    assert!(matches!(
        parts.projected_columns.as_slice(),
        [projected] if projected.register == 1 && matches!(
            &projected.column,
            ResidentRowColumn::String { offsets, bytes, validity }
                if offsets == &[0, 4, 9, 9]
                    && bytes == b"truefalse"
                    && validity == &[1, 1, 0]
        )
    ));

    let mut fingerprint_tamper = request.clone();
    fingerprint_tamper.program.instructions[1].operation =
        ResidentRowOperation::StringConstant("true".to_owned());
    assert_eq!(
        fingerprint_tamper
            .validate()
            .expect_err("Boolean-to-string operation tamper escaped the manifest")
            .code,
        ErrorCode::GpuAdmissionFailure
    );
    drop(results);
    drop(requests);

    let forged_error = backend
        .inner
        .execute_row_program(&forged_null, &CancellationToken::new())
        .err()
        .expect("non-canonical NULL Boolean payload reached string publication");
    assert_eq!(forged_error.code, ErrorCode::CorruptStorage);
    Ok(())
}

#[test]
fn type_conversion4_boolean_property_to_string_is_one_strict_cpu_row_command() -> Result<()> {
    let fixture = Fixture::new()?;
    let backend = fixture.strict_cpu()?;
    assert_strict_route(&fixture, &backend)
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires serialized real Metal hardware acceptance"]
fn type_conversion4_boolean_property_to_string_is_one_real_metal_row_command() -> Result<()> {
    let _guard = metal_test_guard();
    let fixture = Fixture::new()?;
    let backend = fixture.strict_metal()?;
    assert_strict_route(&fixture, &backend)
}
