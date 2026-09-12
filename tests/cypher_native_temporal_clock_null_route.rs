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
    Bookmark, Error, ErrorCode, ProjectId, Result, ScalarValue,
    cypher::{
        BindCapabilities, ColumnType, ExecutionContext, ExecutionOutput, QueryEngine, ResultValue,
        StatementStats,
    },
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend, ResidentGroup,
        ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest, ResidentNodePipelineRequest,
        ResidentNodePipelineResult, ResidentProjectImage, ResidentRowProgramRequest,
        ResidentRowProgramResult, ResidentSortRequest, ResidentSortResult,
        ResidentTemporalPipelineRequest, ResidentTemporalPipelineResult, ResidentTemporalValue,
        ResidentTemporalValueFunction, ResidentTemporalValueInput, ResidentTemporalValueInvocation,
        ResidentTemporalValueProgramRequest, ResidentTemporalValueProgramResult,
        ResidentVectorQuery, ResidentVectorResult, ScratchReservation,
    },
    graph::{GraphStore, IndexCatalog, LayerMask, TemporalStore},
    types::{LabelId, PropertyId},
};
use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::gpu::MetalBackend;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());
const MEMORY_LIMIT_BYTES: usize = 128 * 1024 * 1024;
const RESERVED_BYTES: usize = 16 * 1024 * 1024;
const FEATURE: &str = "features/expressions/temporal/Temporal4.feature";
const MAX_RESULT_ROWS: usize = 8;

#[derive(Clone, Copy, Debug)]
struct ClockNullCase {
    report_id: u16,
    function_name: &'static str,
    function: ResidentTemporalValueFunction,
}

impl ClockNullCase {
    fn official_query(self) -> String {
        format!("RETURN {}(null) AS t", self.function_name)
    }

    fn label(self, form: QueryForm) -> String {
        match form {
            QueryForm::OfficialLiteral => format!(
                "TCK {} {FEATURE} [13] {} literal NULL",
                self.report_id, self.function_name
            ),
            QueryForm::ParameterNull => {
                format!("adversarial {} parameter NULL", self.function_name)
            }
            QueryForm::MixedCaseLiteral => {
                format!("adversarial {} mixed-case literal NULL", self.function_name)
            }
        }
    }
}

const OFFICIAL_CASES: [ClockNullCase; 15] = [
    ClockNullCase {
        report_id: 3410,
        function_name: "date.transaction",
        function: ResidentTemporalValueFunction::Date,
    },
    ClockNullCase {
        report_id: 3411,
        function_name: "date.statement",
        function: ResidentTemporalValueFunction::Date,
    },
    ClockNullCase {
        report_id: 3412,
        function_name: "date.realtime",
        function: ResidentTemporalValueFunction::Date,
    },
    ClockNullCase {
        report_id: 3414,
        function_name: "localtime.transaction",
        function: ResidentTemporalValueFunction::LocalTime,
    },
    ClockNullCase {
        report_id: 3415,
        function_name: "localtime.statement",
        function: ResidentTemporalValueFunction::LocalTime,
    },
    ClockNullCase {
        report_id: 3416,
        function_name: "localtime.realtime",
        function: ResidentTemporalValueFunction::LocalTime,
    },
    ClockNullCase {
        report_id: 3418,
        function_name: "time.transaction",
        function: ResidentTemporalValueFunction::Time,
    },
    ClockNullCase {
        report_id: 3419,
        function_name: "time.statement",
        function: ResidentTemporalValueFunction::Time,
    },
    ClockNullCase {
        report_id: 3420,
        function_name: "time.realtime",
        function: ResidentTemporalValueFunction::Time,
    },
    ClockNullCase {
        report_id: 3422,
        function_name: "localdatetime.transaction",
        function: ResidentTemporalValueFunction::LocalDateTime,
    },
    ClockNullCase {
        report_id: 3423,
        function_name: "localdatetime.statement",
        function: ResidentTemporalValueFunction::LocalDateTime,
    },
    ClockNullCase {
        report_id: 3424,
        function_name: "localdatetime.realtime",
        function: ResidentTemporalValueFunction::LocalDateTime,
    },
    ClockNullCase {
        report_id: 3426,
        function_name: "datetime.transaction",
        function: ResidentTemporalValueFunction::DateTime,
    },
    ClockNullCase {
        report_id: 3427,
        function_name: "datetime.statement",
        function: ResidentTemporalValueFunction::DateTime,
    },
    ClockNullCase {
        report_id: 3428,
        function_name: "datetime.realtime",
        function: ResidentTemporalValueFunction::DateTime,
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum QueryForm {
    OfficialLiteral,
    ParameterNull,
    MixedCaseLiteral,
}

const QUERY_FORMS: [QueryForm; 3] = [
    QueryForm::OfficialLiteral,
    QueryForm::ParameterNull,
    QueryForm::MixedCaseLiteral,
];

fn mixed_case_name(name: &str) -> String {
    let mut uppercase = true;
    name.chars()
        .map(|character| {
            if character.is_ascii_alphabetic() {
                let output = if uppercase {
                    character.to_ascii_uppercase()
                } else {
                    character.to_ascii_lowercase()
                };
                uppercase = !uppercase;
                output
            } else {
                character
            }
        })
        .collect()
}

fn query(case: ClockNullCase, form: QueryForm) -> String {
    match form {
        QueryForm::OfficialLiteral => case.official_query(),
        QueryForm::ParameterNull => format!("RETURN {}($timezone) AS t", case.function_name),
        QueryForm::MixedCaseLiteral => {
            format!("RETURN {}(null) AS t", mixed_case_name(case.function_name))
        }
    }
}

struct Fixture {
    graph: GraphStore,
    bookmark: Bookmark,
}

impl Fixture {
    fn new() -> Self {
        let graph = GraphStore::default();
        Self {
            bookmark: Bookmark {
                term: 37,
                index: graph.revision(),
            },
            graph,
        }
    }

    fn image(&self) -> Result<ResidentProjectImage> {
        ResidentProjectImage::build(
            PROJECT,
            self.bookmark,
            &self.graph,
            &TemporalStore::default(),
            &IndexCatalog::default(),
        )
    }

    fn cpu(&self) -> Result<CpuBackend> {
        let mut cpu = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
        cpu.admit_project(self.image()?)?;
        Ok(cpu)
    }

    fn strict_cpu_backend(&self) -> Result<ObservedTemporalBackend> {
        ObservedTemporalBackend::strict_cpu_reference(self.cpu()?)
    }

    fn cpu_masquerading_as_metal(&self) -> Result<ObservedTemporalBackend> {
        ObservedTemporalBackend::new(
            Box::new(self.cpu()?),
            BackendKind::Metal,
            BackendKind::Metal,
            BackendKind::Metal,
        )
    }

    #[cfg(all(feature = "accelerator", target_os = "macos"))]
    fn real_metal_backend(&self) -> Result<ObservedTemporalBackend> {
        let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
        metal.admit_project(self.image()?)?;
        ObservedTemporalBackend::real_metal(metal)
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct NativeTemporalReceipt {
    project: ProjectId,
    bookmark: Bookmark,
    graph_revision: u64,
    completion: BackendKind,
    request: ResidentTemporalValueProgramRequest,
    result: ResidentTemporalValueProgramResult,
}

#[derive(Default)]
struct TemporalObservations {
    pins: AtomicUsize,
    temporal_value_calls: AtomicUsize,
    legacy_route_calls: AtomicUsize,
    unexpected_query_calls: AtomicUsize,
    receipts: Mutex<Vec<NativeTemporalReceipt>>,
}

/// Fail-closed observer around the sole accepted execution boundary.
///
/// The CPU reference advertises Metal until the query pins its immutable generation, preventing
/// the generic CPU evaluator from becoming an accepted success path. The pinned wrapper then
/// reports honest CPU provenance. A real Metal wrapper reports Metal throughout. A successful
/// call records a test-local receipt because the production temporal-value result has no receipt
/// field; the receipt captures the concrete backend kind, immutable fence, exact raw invocation,
/// and exact value returned by that backend.
struct ObservedTemporalBackend {
    inner: Box<dyn ExecutionBackend>,
    advertised_kind: BackendKind,
    pinned_kind: BackendKind,
    actual_kind: BackendKind,
    pinned: bool,
    expected_bookmark: Bookmark,
    expected_graph_revision: u64,
    observations: Arc<TemporalObservations>,
}

impl ObservedTemporalBackend {
    fn strict_cpu_reference(inner: CpuBackend) -> Result<Self> {
        Self::new(
            Box::new(inner),
            BackendKind::Metal,
            BackendKind::Metal,
            BackendKind::Cpu,
        )
    }

    #[cfg(all(feature = "accelerator", target_os = "macos"))]
    fn real_metal(inner: MetalBackend) -> Result<Self> {
        if inner.kind() != BackendKind::Metal {
            return Err(Error::internal(
                "temporal clock NULL test did not construct a real Metal backend",
            ));
        }
        Self::new(
            Box::new(inner),
            BackendKind::Metal,
            BackendKind::Metal,
            BackendKind::Metal,
        )
    }

    fn new(
        inner: Box<dyn ExecutionBackend>,
        advertised_kind: BackendKind,
        pinned_kind: BackendKind,
        actual_kind: BackendKind,
    ) -> Result<Self> {
        let expected_bookmark = inner.resident_bookmark(PROJECT).ok_or_else(|| {
            Error::internal("temporal clock NULL backend has no admitted project bookmark")
        })?;
        let expected_graph_revision = inner.resident_graph_revision(PROJECT).ok_or_else(|| {
            Error::internal("temporal clock NULL backend has no admitted graph revision")
        })?;
        Ok(Self {
            inner,
            advertised_kind,
            pinned_kind,
            actual_kind,
            pinned: false,
            expected_bookmark,
            expected_graph_revision,
            observations: Arc::new(TemporalObservations::default()),
        })
    }

    fn observations(&self) -> Arc<TemporalObservations> {
        Arc::clone(&self.observations)
    }

    fn reject_query_route<T>(&self, route: &'static str) -> Result<T> {
        self.observations
            .unexpected_query_calls
            .fetch_add(1, Ordering::SeqCst);
        Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!("strict temporal clock NULL test rejected `{route}` execution"),
        ))
    }

    fn reject_legacy_route<T>(&self, route: &'static str) -> Result<T> {
        self.observations
            .legacy_route_calls
            .fetch_add(1, Ordering::SeqCst);
        self.reject_query_route(route)
    }

    fn refresh_fence(&mut self) -> Result<()> {
        self.expected_bookmark = self.inner.resident_bookmark(PROJECT).ok_or_else(|| {
            Error::internal("replacement temporal clock NULL project has no resident bookmark")
        })?;
        self.expected_graph_revision =
            self.inner.resident_graph_revision(PROJECT).ok_or_else(|| {
                Error::internal(
                    "replacement temporal clock NULL project has no resident graph revision",
                )
            })?;
        Ok(())
    }
}

impl ExecutionBackend for ObservedTemporalBackend {
    fn kind(&self) -> BackendKind {
        if self.pinned {
            self.pinned_kind
        } else {
            self.advertised_kind
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
            return self.reject_query_route("pin_project_twice");
        }
        if project != PROJECT {
            return self.reject_query_route("pin_wrong_project");
        }
        let inner = self.inner.pin_project(project)?;
        if inner.kind() != self.actual_kind
            || inner.resident_bookmark(project) != Some(self.expected_bookmark)
            || inner.resident_graph_revision(project) != Some(self.expected_graph_revision)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "pinned temporal clock NULL generation does not match its immutable fence or backend provenance",
            ));
        }
        self.observations.pins.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Self {
            inner,
            advertised_kind: self.advertised_kind,
            pinned_kind: self.pinned_kind,
            actual_kind: self.actual_kind,
            pinned: true,
            expected_bookmark: self.expected_bookmark,
            expected_graph_revision: self.expected_graph_revision,
            observations: Arc::clone(&self.observations),
        }))
    }

    fn admit_project(&mut self, image: ResidentProjectImage) -> Result<()> {
        self.inner.admit_project(image)?;
        self.refresh_fence()
    }

    fn replace_all_projects(&mut self, images: Vec<ResidentProjectImage>) -> Result<()> {
        self.inner.replace_all_projects(images)?;
        self.refresh_fence()
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
        self.reject_query_route("scan_nodes")
    }

    fn filter_node_i64(
        &self,
        _project: ProjectId,
        _property: PropertyId,
        _operation: CompareOp,
        _operand: i64,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.reject_query_route("filter_node_i64")
    }

    fn expand_project_out(
        &self,
        _project: ProjectId,
        _sources: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.reject_query_route("expand_project_out")
    }

    fn expand_project_in(
        &self,
        _project: ProjectId,
        _targets: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.reject_query_route("expand_project_in")
    }

    fn search_vectors(
        &self,
        _request: &ResidentVectorQuery,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentVectorResult> {
        self.reject_query_route("search_vectors")
    }

    fn sort_rows(
        &self,
        _request: &ResidentSortRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentSortResult> {
        self.reject_query_route("sort_rows")
    }

    fn join_node_i64(
        &self,
        _request: &ResidentJoinRequest,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentJoinPair>> {
        self.reject_query_route("join_node_i64")
    }

    fn group_node_i64(
        &self,
        _request: &ResidentGroupRequest,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentGroup>> {
        self.reject_query_route("group_node_i64")
    }

    fn execute_node_pipeline(
        &self,
        _request: &ResidentNodePipelineRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentNodePipelineResult> {
        self.reject_legacy_route("execute_node_pipeline")
    }

    fn execute_row_program(
        &self,
        _request: &ResidentRowProgramRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentRowProgramResult> {
        self.reject_legacy_route("execute_row_program")
    }

    fn supports_native_temporal_value_program(&self) -> bool {
        true
    }

    fn execute_temporal_value_program(
        &self,
        request: &ResidentTemporalValueProgramRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentTemporalValueProgramResult> {
        if !self.pinned {
            return self.reject_query_route("execute_temporal_value_program_without_pin");
        }
        if self.inner.kind() != self.actual_kind
            || self.inner.resident_bookmark(PROJECT) != Some(self.expected_bookmark)
            || self.inner.resident_graph_revision(PROJECT) != Some(self.expected_graph_revision)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "temporal clock NULL execution escaped its pinned generation or backend provenance",
            ));
        }
        self.observations
            .temporal_value_calls
            .fetch_add(1, Ordering::SeqCst);
        let result = self
            .inner
            .execute_temporal_value_program(request, cancellation)?;
        self.observations
            .receipts
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(NativeTemporalReceipt {
                project: PROJECT,
                bookmark: self.expected_bookmark,
                graph_revision: self.expected_graph_revision,
                completion: self.actual_kind,
                request: request.clone(),
                result: result.clone(),
            });
        Ok(result)
    }

    fn execute_temporal_pipeline(
        &self,
        _request: &ResidentTemporalPipelineRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentTemporalPipelineResult> {
        self.reject_legacy_route("execute_temporal_pipeline")
    }

    fn filter_i64(
        &self,
        _values: &[i64],
        _validity: &[bool],
        _operation: CompareOp,
        _operand: i64,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.reject_query_route("filter_i64")
    }

    fn expand_out(
        &self,
        _sources: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.reject_query_route("expand_out")
    }

    fn exact_l2(
        &self,
        _matrix: &[f32],
        _rows: usize,
        _dimension: usize,
        _query: &[f32],
        _cancellation: &CancellationToken,
    ) -> Result<DistanceBatch> {
        self.reject_query_route("exact_l2")
    }
}

fn context<'a>(
    fixture: &'a Fixture,
    backend: Option<&'a dyn ExecutionBackend>,
    parameter_null: bool,
) -> ExecutionContext<'a> {
    let mut parameters = BTreeMap::new();
    if parameter_null {
        parameters.insert(
            "timezone".to_owned(),
            ResultValue::Scalar(ScalarValue::Null),
        );
    }
    ExecutionContext {
        project_id: PROJECT,
        graph: &fixture.graph,
        binding_catalog: fixture.graph.catalog(),
        prior_graph_mutations: &[],
        temporal: None,
        prior_temporal_mutations: &[],
        vector_indexes: BTreeMap::new(),
        scalar_indexes: None,
        text_embedding: None,
        parameters,
        bookmark: fixture.bookmark,
        mutation_revision: fixture.graph.revision().saturating_add(1),
        resolved_time_nanos: i64::MIN,
        next_node_id: 1,
        next_edge_id: 1,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            require_native_execution: backend.is_some(),
            ..BindCapabilities::default()
        },
        max_result_rows: MAX_RESULT_ROWS,
        max_batch_rows: 1,
        optimizer_statistics: None,
        backend,
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(10)),
        resolved_query_at_time_nanos: None,
    }
}

fn observe_exact_null(output: &ExecutionOutput, fixture: &Fixture) -> Result<Vec<ScalarValue>> {
    if !output.graph_mutations.is_empty()
        || !output.temporal_mutations.is_empty()
        || output.result.statistics != StatementStats::default()
        || output.result.truncated
        || output.administrative.is_some()
        || !output.vector_searches.is_empty()
        || output.runtime_replans != 0
    {
        return Err(Error::internal(
            "qualified temporal clock NULL query produced side effects, truncation, or auxiliary work",
        ));
    }
    if output.result.bookmark != fixture.bookmark
        || output.result.schema != vec![("t".to_owned(), ColumnType::Null)]
        || output.result.batches.len() != 1
    {
        return Err(Error::internal(format!(
            "qualified temporal clock NULL result metadata is wrong: {:?}",
            output.result
        )));
    }
    let batch = &output.result.batches[0];
    if batch.row_count != 1 || batch.columns.len() != 1 {
        return Err(Error::internal(
            "qualified temporal clock NULL result is not exactly one row and one column",
        ));
    }
    let column = &batch.columns[0];
    if column.name != "t"
        || column.value_type != ColumnType::Null
        || column.values != vec![ResultValue::Scalar(ScalarValue::Null)]
    {
        return Err(Error::internal(format!(
            "qualified temporal clock backend did not return exact NULL: {column:?}"
        )));
    }
    Ok(vec![ScalarValue::Null])
}

fn assert_native_receipt(
    fixture: &Fixture,
    backend: &ObservedTemporalBackend,
    case: ClockNullCase,
    receipt: &NativeTemporalReceipt,
) -> Result<()> {
    let expected_request = ResidentTemporalValueProgramRequest {
        invocations: vec![ResidentTemporalValueInvocation {
            function: case.function,
            input: ResidentTemporalValueInput::Null,
        }],
        output_registers: vec![0],
    };
    if receipt.project != PROJECT
        || receipt.bookmark != fixture.bookmark
        || receipt.graph_revision != fixture.graph.revision()
        || receipt.completion != backend.actual_kind
        || receipt.request != expected_request
        || receipt.result.values != vec![ResidentTemporalValue::Null]
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "qualified temporal clock native receipt is wrong: expected backend {:?}, request {:?}, exact NULL; got {receipt:?}",
                backend.actual_kind, expected_request
            ),
        ));
    }
    Ok(())
}

fn execute_native_case(
    fixture: &Fixture,
    backend: &ObservedTemporalBackend,
    case: ClockNullCase,
    form: QueryForm,
) -> std::result::Result<Vec<ScalarValue>, String> {
    let observations = backend.observations();
    let pins_before = observations.pins.load(Ordering::SeqCst);
    let temporal_before = observations.temporal_value_calls.load(Ordering::SeqCst);
    let legacy_before = observations.legacy_route_calls.load(Ordering::SeqCst);
    let unexpected_before = observations.unexpected_query_calls.load(Ordering::SeqCst);
    let receipts_before = observations
        .receipts
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .len();
    let pinned = backend.pin_project(PROJECT).map_err(|error| {
        format!(
            "immutable native generation pin failed with {:?}: {error}",
            error.code
        )
    })?;
    if pinned.kind() != BackendKind::Metal {
        return Err(format!(
            "strict native generation advertised {:?} after pinning",
            pinned.kind()
        ));
    }
    let query = query(case, form);
    let output = QueryEngine
        .execute(
            &query,
            &mut context(
                fixture,
                Some(pinned.as_ref()),
                form == QueryForm::ParameterNull,
            ),
        )
        .map_err(|error| {
            format!(
                "query failed with {:?}: {error}; pins={}, temporal_calls={}, legacy_calls={}, unexpected_calls={}",
                error.code,
                observations.pins.load(Ordering::SeqCst) - pins_before,
                observations.temporal_value_calls.load(Ordering::SeqCst) - temporal_before,
                observations.legacy_route_calls.load(Ordering::SeqCst) - legacy_before,
                observations.unexpected_query_calls.load(Ordering::SeqCst) - unexpected_before,
            )
        })?;
    if observations.pins.load(Ordering::SeqCst) != pins_before + 1 {
        return Err("query did not pin exactly one immutable resident generation".to_owned());
    }
    if observations.temporal_value_calls.load(Ordering::SeqCst) != temporal_before + 1 {
        return Err(
            "query did not cross exactly one native temporal-value backend boundary".to_owned(),
        );
    }
    if observations.legacy_route_calls.load(Ordering::SeqCst) != legacy_before
        || observations.unexpected_query_calls.load(Ordering::SeqCst) != unexpected_before
    {
        return Err(
            "query entered a generic, legacy, graph, row, host-oriented, or fallback route"
                .to_owned(),
        );
    }
    let receipt = {
        let receipts = observations
            .receipts
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if receipts.len() != receipts_before + 1 {
            return Err("native temporal observer did not retain exactly one receipt".to_owned());
        }
        receipts
            .last()
            .cloned()
            .ok_or_else(|| "native temporal receipt disappeared".to_owned())?
    };
    assert_native_receipt(fixture, backend, case, &receipt)
        .map_err(|error| format!("native request/provenance assertion failed: {error}"))?;
    observe_exact_null(&output, fixture)
        .map_err(|error| format!("native output inspection failed: {error}"))
}

fn run_native_suite(fixture: &Fixture, backend: &ObservedTemporalBackend) -> Vec<String> {
    let mut failures = Vec::new();
    for form in QUERY_FORMS {
        for case in OFFICIAL_CASES {
            if let Err(error) = execute_native_case(fixture, backend, case, form) {
                failures.push(format!("{}: {error}", case.label(form)));
            }
        }
    }
    failures
}

fn assert_no_failures(backend: &str, failures: Vec<String>) {
    assert!(
        failures.is_empty(),
        "{backend} qualified temporal clock NULL suite had {} failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn manifest_pins_exact_temporal4_qualified_clock_null_ids_and_queries() {
    assert_eq!(OFFICIAL_CASES.len(), 15);
    assert_eq!(
        OFFICIAL_CASES
            .iter()
            .map(|case| case.report_id)
            .collect::<Vec<_>>(),
        vec![
            3410, 3411, 3412, 3414, 3415, 3416, 3418, 3419, 3420, 3422, 3423, 3424, 3426, 3427,
            3428,
        ]
    );
    assert_eq!(
        OFFICIAL_CASES
            .iter()
            .map(|case| case.official_query())
            .collect::<Vec<_>>(),
        vec![
            "RETURN date.transaction(null) AS t",
            "RETURN date.statement(null) AS t",
            "RETURN date.realtime(null) AS t",
            "RETURN localtime.transaction(null) AS t",
            "RETURN localtime.statement(null) AS t",
            "RETURN localtime.realtime(null) AS t",
            "RETURN time.transaction(null) AS t",
            "RETURN time.statement(null) AS t",
            "RETURN time.realtime(null) AS t",
            "RETURN localdatetime.transaction(null) AS t",
            "RETURN localdatetime.statement(null) AS t",
            "RETURN localdatetime.realtime(null) AS t",
            "RETURN datetime.transaction(null) AS t",
            "RETURN datetime.statement(null) AS t",
            "RETURN datetime.realtime(null) AS t",
        ]
    );
    for case in OFFICIAL_CASES {
        let mixed = mixed_case_name(case.function_name);
        assert_ne!(mixed, case.function_name);
        assert!(mixed.eq_ignore_ascii_case(case.function_name));
        assert_eq!(
            case.official_query(),
            format!("RETURN {}(null) AS t", case.function_name)
        );
    }
}

#[test]
fn strict_cpu_runs_all_15_official_and_30_adversarial_cases_natively() -> Result<()> {
    let fixture = Fixture::new();
    let backend = fixture.strict_cpu_backend()?;
    assert_eq!(backend.kind(), BackendKind::Metal);
    assert_eq!(backend.actual_kind, BackendKind::Cpu);
    assert_no_failures("strict CPU reference", run_native_suite(&fixture, &backend));
    let observations = backend.observations();
    assert_eq!(observations.pins.load(Ordering::SeqCst), 45);
    assert_eq!(observations.temporal_value_calls.load(Ordering::SeqCst), 45);
    assert_eq!(observations.legacy_route_calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        observations.unexpected_query_calls.load(Ordering::SeqCst),
        0
    );
    assert!(
        observations
            .receipts
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .iter()
            .all(|receipt| receipt.completion == BackendKind::Cpu)
    );
    Ok(())
}

#[test]
fn cpu_backend_cannot_masquerade_as_metal_or_publish_a_null() -> Result<()> {
    let fixture = Fixture::new();
    let backend = fixture.cpu_masquerading_as_metal()?;
    let observations = backend.observations();
    let error = match backend.pin_project(PROJECT) {
        Ok(_) => panic!("a CPU backend advertised as Metal unexpectedly pinned as Metal"),
        Err(error) => error,
    };
    assert_eq!(error.code, ErrorCode::CorruptStorage);
    assert_eq!(observations.pins.load(Ordering::SeqCst), 0);
    assert_eq!(observations.temporal_value_calls.load(Ordering::SeqCst), 0);
    assert_eq!(observations.legacy_route_calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        observations.unexpected_query_calls.load(Ordering::SeqCst),
        0
    );
    assert!(
        observations
            .receipts
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .is_empty()
    );
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
#[ignore = "hardware acceptance gate: 15 official, 15 parameter-NULL, and 15 mixed-case queries require real Metal"]
fn real_metal_matches_strict_cpu_with_exact_native_receipts_and_no_fallback() -> Result<()> {
    let _guard = metal_test_guard();
    let fixture = Fixture::new();
    let cpu = fixture.strict_cpu_backend()?;
    let metal = fixture.real_metal_backend()?;
    assert_eq!(cpu.kind(), BackendKind::Metal);
    assert_eq!(cpu.actual_kind, BackendKind::Cpu);
    assert_eq!(metal.kind(), BackendKind::Metal);
    assert_eq!(metal.actual_kind, BackendKind::Metal);

    let mut failures = Vec::new();
    for form in QUERY_FORMS {
        for case in OFFICIAL_CASES {
            let cpu_result = execute_native_case(&fixture, &cpu, case, form);
            let metal_result = execute_native_case(&fixture, &metal, case, form);
            match (cpu_result, metal_result) {
                (Ok(cpu_rows), Ok(metal_rows)) if cpu_rows == metal_rows => {}
                (Ok(cpu_rows), Ok(metal_rows)) => failures.push(format!(
                    "{}: CPU/Metal mismatch: CPU={cpu_rows:?}, Metal={metal_rows:?}",
                    case.label(form)
                )),
                (Err(cpu_error), Ok(_)) => failures.push(format!(
                    "{}: strict CPU failed: {cpu_error}",
                    case.label(form)
                )),
                (Ok(_), Err(metal_error)) => failures.push(format!(
                    "{}: real Metal failed: {metal_error}",
                    case.label(form)
                )),
                (Err(cpu_error), Err(metal_error)) => failures.push(format!(
                    "{}: strict CPU failed: {cpu_error}; real Metal failed: {metal_error}",
                    case.label(form)
                )),
            }
        }
    }
    assert_no_failures("strict CPU versus real Metal parity", failures);

    for (name, backend, completion) in [
        ("CPU", &cpu, BackendKind::Cpu),
        ("Metal", &metal, BackendKind::Metal),
    ] {
        let observations = backend.observations();
        assert_eq!(observations.pins.load(Ordering::SeqCst), 45, "{name}");
        assert_eq!(
            observations.temporal_value_calls.load(Ordering::SeqCst),
            45,
            "{name}"
        );
        assert_eq!(
            observations.legacy_route_calls.load(Ordering::SeqCst),
            0,
            "{name}"
        );
        assert_eq!(
            observations.unexpected_query_calls.load(Ordering::SeqCst),
            0,
            "{name}"
        );
        let receipts = observations
            .receipts
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(receipts.len(), 45, "{name}");
        assert!(
            receipts.iter().all(|receipt| {
                receipt.completion == completion
                    && receipt.bookmark == fixture.bookmark
                    && receipt.graph_revision == fixture.graph.revision()
                    && receipt.result.values == vec![ResidentTemporalValue::Null]
            }),
            "{name} receipt provenance or exact NULL result changed"
        );
    }
    Ok(())
}
