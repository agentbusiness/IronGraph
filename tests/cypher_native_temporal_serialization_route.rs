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
        ResidentTemporalValueFunction, ResidentTemporalValueInput,
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
const MAX_RESULT_ROWS: usize = 2;
const FEATURE: &str = "features/expressions/temporal/Temporal6.feature";
const BASELINE_REPORT: &str = "/tmp/irongraph-tck-full-after-regression-fix.json";
const TEMPORAL_MAP_WIRE_VERSION: u8 = 1;
const DATE_MAX_UTF8_BYTES: usize = 16;
const LOCAL_TIME_MAX_UTF8_BYTES: usize = 18;
const TIME_MAX_UTF8_BYTES: usize = 27;
const LOCAL_DATETIME_MAX_UTF8_BYTES: usize = 35;
const FIXED_DATETIME_MAX_UTF8_BYTES: usize = 44;
const NAMED_DATETIME_PREFIX_MAX_UTF8_BYTES: usize = 46;
const DURATION_MAX_UTF8_BYTES: usize = 83;

#[derive(Clone, Copy, Debug)]
struct SerializationCase {
    report_id: u16,
    report_name: &'static str,
    function: ResidentTemporalValueFunction,
    query: &'static str,
    rendered: &'static str,
    round_trip: bool,
}

impl SerializationCase {
    fn label(self) -> String {
        format!("TCK {} {FEATURE} {}", self.report_id, self.report_name)
    }

    fn expected_scalars(self) -> Vec<ScalarValue> {
        let mut values = vec![ScalarValue::String(Arc::from(self.rendered))];
        if self.round_trip {
            values.push(ScalarValue::Boolean(true));
        }
        values
    }

    fn expected_schema(self) -> Vec<(String, ColumnType)> {
        let mut schema = vec![("ts".to_owned(), ColumnType::String)];
        if self.round_trip {
            schema.push(("b".to_owned(), ColumnType::Boolean));
        }
        schema
    }
}

const OFFICIAL_CASES: [SerializationCase; 17] = [
    SerializationCase {
        report_id: 3436,
        report_name: "[1] Should serialize date",
        function: ResidentTemporalValueFunction::Date,
        query: "WITH date({year: 1984, month: 10, day: 11}) AS d\nRETURN toString(d) AS ts, date(toString(d)) = d AS b",
        rendered: "1984-10-11",
        round_trip: true,
    },
    SerializationCase {
        report_id: 3437,
        report_name: "[2] Should serialize local time",
        function: ResidentTemporalValueFunction::LocalTime,
        query: "WITH localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS d\nRETURN toString(d) AS ts, localtime(toString(d)) = d AS b",
        rendered: "12:31:14.645876123",
        round_trip: true,
    },
    SerializationCase {
        report_id: 3438,
        report_name: "[3] Should serialize time",
        function: ResidentTemporalValueFunction::Time,
        query: "WITH time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) AS d\nRETURN toString(d) AS ts, time(toString(d)) = d AS b",
        rendered: "12:31:14.645876123+01:00",
        round_trip: true,
    },
    SerializationCase {
        report_id: 3439,
        report_name: "[4] Should serialize local date time",
        function: ResidentTemporalValueFunction::LocalDateTime,
        query: "WITH localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS d\nRETURN toString(d) AS ts, localdatetime(toString(d)) = d AS b",
        rendered: "1984-10-11T12:31:14.645876123",
        round_trip: true,
    },
    SerializationCase {
        report_id: 3440,
        report_name: "[5] Should serialize date time",
        function: ResidentTemporalValueFunction::DateTime,
        query: "WITH datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) AS d\nRETURN toString(d) AS ts, datetime(toString(d)) = d AS b",
        rendered: "1984-10-11T12:31:14.645876123+01:00",
        round_trip: true,
    },
    SerializationCase {
        report_id: 3441,
        report_name: "[6] Should serialize duration [3442]",
        function: ResidentTemporalValueFunction::Duration,
        query: "WITH duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70, nanoseconds: 1}) AS d\nRETURN toString(d) AS ts, duration(toString(d)) = d AS b",
        rendered: "P12Y5M14DT16H13M10.000000001S",
        round_trip: true,
    },
    SerializationCase {
        report_id: 3442,
        report_name: "[6] Should serialize duration [3443]",
        function: ResidentTemporalValueFunction::Duration,
        query: "WITH duration({years: 12, months: 5, days: -14, hours: 16}) AS d\nRETURN toString(d) AS ts, duration(toString(d)) = d AS b",
        rendered: "P12Y5M-14DT16H",
        round_trip: true,
    },
    SerializationCase {
        report_id: 3443,
        report_name: "[6] Should serialize duration [3444]",
        function: ResidentTemporalValueFunction::Duration,
        query: "WITH duration({minutes: 12, seconds: -60}) AS d\nRETURN toString(d) AS ts, duration(toString(d)) = d AS b",
        rendered: "PT11M",
        round_trip: true,
    },
    SerializationCase {
        report_id: 3444,
        report_name: "[6] Should serialize duration [3445]",
        function: ResidentTemporalValueFunction::Duration,
        query: "WITH duration({seconds: 2, milliseconds: -1}) AS d\nRETURN toString(d) AS ts, duration(toString(d)) = d AS b",
        rendered: "PT1.999S",
        round_trip: true,
    },
    SerializationCase {
        report_id: 3445,
        report_name: "[6] Should serialize duration [3446]",
        function: ResidentTemporalValueFunction::Duration,
        query: "WITH duration({seconds: -2, milliseconds: 1}) AS d\nRETURN toString(d) AS ts, duration(toString(d)) = d AS b",
        rendered: "PT-1.999S",
        round_trip: true,
    },
    SerializationCase {
        report_id: 3446,
        report_name: "[6] Should serialize duration [3447]",
        function: ResidentTemporalValueFunction::Duration,
        query: "WITH duration({seconds: -2, milliseconds: -1}) AS d\nRETURN toString(d) AS ts, duration(toString(d)) = d AS b",
        rendered: "PT-2.001S",
        round_trip: true,
    },
    SerializationCase {
        report_id: 3447,
        report_name: "[6] Should serialize duration [3448]",
        function: ResidentTemporalValueFunction::Duration,
        query: "WITH duration({days: 1, milliseconds: 1}) AS d\nRETURN toString(d) AS ts, duration(toString(d)) = d AS b",
        rendered: "P1DT0.001S",
        round_trip: true,
    },
    SerializationCase {
        report_id: 3448,
        report_name: "[6] Should serialize duration [3449]",
        function: ResidentTemporalValueFunction::Duration,
        query: "WITH duration({days: 1, milliseconds: -1}) AS d\nRETURN toString(d) AS ts, duration(toString(d)) = d AS b",
        rendered: "P1DT-0.001S",
        round_trip: true,
    },
    SerializationCase {
        report_id: 3449,
        report_name: "[6] Should serialize duration [3450]",
        function: ResidentTemporalValueFunction::Duration,
        query: "WITH duration({seconds: 60, milliseconds: -1}) AS d\nRETURN toString(d) AS ts, duration(toString(d)) = d AS b",
        rendered: "PT59.999S",
        round_trip: true,
    },
    SerializationCase {
        report_id: 3450,
        report_name: "[6] Should serialize duration [3451]",
        function: ResidentTemporalValueFunction::Duration,
        query: "WITH duration({seconds: -60, milliseconds: 1}) AS d\nRETURN toString(d) AS ts, duration(toString(d)) = d AS b",
        rendered: "PT-59.999S",
        round_trip: true,
    },
    SerializationCase {
        report_id: 3451,
        report_name: "[6] Should serialize duration [3452]",
        function: ResidentTemporalValueFunction::Duration,
        query: "WITH duration({seconds: -60, milliseconds: -1}) AS d\nRETURN toString(d) AS ts, duration(toString(d)) = d AS b",
        rendered: "PT-1M-0.001S",
        round_trip: true,
    },
    SerializationCase {
        report_id: 3452,
        report_name: "[7] Should serialize timezones correctly",
        function: ResidentTemporalValueFunction::DateTime,
        query: "WITH datetime({year: 2017, month: 8, day: 8, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: 'Europe/Stockholm'}) AS d\nRETURN toString(d) AS ts",
        rendered: "2017-08-08T12:31:14.645876123+02:00[Europe/Stockholm]",
        round_trip: false,
    },
];

struct Fixture {
    graph: GraphStore,
    bookmark: Bookmark,
}

impl Fixture {
    fn new() -> Self {
        let graph = GraphStore::default();
        Self {
            bookmark: Bookmark {
                term: 43,
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
    generation_mutation_attempts: AtomicUsize,
    receipts: Mutex<Vec<NativeTemporalReceipt>>,
}

/// Fail-closed observer around the only accepted execution boundary. The CPU reference advertises
/// Metal so the planner cannot select the generic evaluator as a native success. A successful
/// query must pin one immutable generation, issue one temporal-value program, and publish only the
/// value returned by the concrete CPU or Metal backend.
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
                "temporal serialization test did not construct a real Metal backend",
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
            Error::internal("temporal serialization backend has no admitted project bookmark")
        })?;
        let expected_graph_revision = inner.resident_graph_revision(PROJECT).ok_or_else(|| {
            Error::internal("temporal serialization backend has no admitted graph revision")
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
            format!("strict temporal serialization test rejected `{route}` execution"),
        ))
    }

    fn reject_legacy_route<T>(&self, route: &'static str) -> Result<T> {
        self.observations
            .legacy_route_calls
            .fetch_add(1, Ordering::SeqCst);
        self.reject_query_route(route)
    }

    fn reject_generation_mutation<T>(&self, route: &'static str) -> Result<T> {
        self.observations
            .generation_mutation_attempts
            .fetch_add(1, Ordering::SeqCst);
        Err(Error::new(
            ErrorCode::CorruptStorage,
            format!("pinned temporal serialization generation rejected `{route}` mutation"),
        ))
    }

    fn refresh_fence(&mut self) -> Result<()> {
        self.expected_bookmark = self.inner.resident_bookmark(PROJECT).ok_or_else(|| {
            Error::internal("replacement temporal serialization project has no resident bookmark")
        })?;
        self.expected_graph_revision =
            self.inner.resident_graph_revision(PROJECT).ok_or_else(|| {
                Error::internal(
                    "replacement temporal serialization project has no resident graph revision",
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
                "pinned temporal serialization generation has wrong backend provenance or fence",
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
        if self.pinned {
            return self.reject_generation_mutation("admit_project");
        }
        self.inner.admit_project(image)?;
        self.refresh_fence()
    }

    fn replace_all_projects(&mut self, images: Vec<ResidentProjectImage>) -> Result<()> {
        if self.pinned {
            return self.reject_generation_mutation("replace_all_projects");
        }
        self.inner.replace_all_projects(images)?;
        self.refresh_fence()
    }

    fn evict_project(&mut self, project: ProjectId) -> Result<()> {
        if self.pinned {
            return self.reject_generation_mutation("evict_project");
        }
        self.inner.evict_project(project)
    }

    fn advance_bookmark(&mut self, bookmark: Bookmark) {
        if self.pinned {
            self.observations
                .generation_mutation_attempts
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
                "temporal serialization execution escaped its immutable generation or provenance",
            ));
        }
        self.observations
            .temporal_value_calls
            .fetch_add(1, Ordering::SeqCst);
        let result = self
            .inner
            .execute_temporal_value_program(request, cancellation)?;
        if self.inner.kind() != self.actual_kind
            || self.inner.resident_bookmark(PROJECT) != Some(self.expected_bookmark)
            || self.inner.resident_graph_revision(PROJECT) != Some(self.expected_graph_revision)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "temporal serialization dispatch mutated its pinned generation",
            ));
        }
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
) -> ExecutionContext<'a> {
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
        parameters: BTreeMap::new(),
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

fn observe_output(
    output: &ExecutionOutput,
    fixture: &Fixture,
    case: SerializationCase,
) -> Result<Vec<ScalarValue>> {
    if !output.graph_mutations.is_empty()
        || !output.temporal_mutations.is_empty()
        || output.result.statistics != StatementStats::default()
        || output.result.truncated
        || output.administrative.is_some()
        || !output.vector_searches.is_empty()
        || output.runtime_replans != 0
    {
        return Err(Error::internal(
            "temporal serialization query produced side effects, truncation, or auxiliary work",
        ));
    }
    let expected_schema = case.expected_schema();
    if output.result.bookmark != fixture.bookmark
        || output.result.schema != expected_schema
        || output.result.batches.len() != 1
    {
        return Err(Error::internal(format!(
            "temporal serialization result metadata is wrong: {:?}",
            output.result
        )));
    }
    let batch = &output.result.batches[0];
    let expected = case.expected_scalars();
    if batch.row_count != 1 || batch.columns.len() != expected.len() {
        return Err(Error::internal(
            "temporal serialization result is not exactly one complete row",
        ));
    }
    for (index, (column, scalar)) in batch.columns.iter().zip(&expected).enumerate() {
        let expected_name = if index == 0 { "ts" } else { "b" };
        let expected_type = if index == 0 {
            ColumnType::String
        } else {
            ColumnType::Boolean
        };
        if column.name != expected_name
            || column.value_type != expected_type
            || column.values != vec![ResultValue::Scalar(scalar.clone())]
        {
            return Err(Error::internal(format!(
                "temporal serialization column {index} is wrong: expected {expected_name:?}/{expected_type:?}/{scalar:?}, got {column:?}"
            )));
        }
    }
    Ok(expected)
}

fn input_is_serialization_of(input: &ResidentTemporalValueInput, source: u16) -> bool {
    matches!(input, ResidentTemporalValueInput::ToString { source: actual } if *actual == source)
}

fn assert_native_receipt(
    fixture: &Fixture,
    backend: &ObservedTemporalBackend,
    case: SerializationCase,
    receipt: &NativeTemporalReceipt,
) -> Result<()> {
    if receipt.project != PROJECT
        || receipt.bookmark != fixture.bookmark
        || receipt.graph_revision != fixture.graph.revision()
        || receipt.completion != backend.actual_kind
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "temporal serialization receipt has wrong project, fence, or backend provenance: {receipt:?}"
            ),
        ));
    }

    let request = &receipt.request;
    let expected_invocations = if case.round_trip { 5 } else { 2 };
    if request.invocations.len() != expected_invocations {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "temporal serialization must keep constructor, native serialization, reparse, and comparison in one SSA program; expected {expected_invocations} invocations, got {:?}",
                request.invocations
            ),
        ));
    }
    let source = &request.invocations[0];
    if source.function != case.function {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!("temporal source family changed: {source:?}"),
        ));
    }
    match &source.input {
        ResidentTemporalValueInput::Map(packet)
            if packet.len() >= 3 && packet[0] == TEMPORAL_MAP_WIRE_VERSION && packet[1] != 0 => {}
        input => {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                format!(
                    "temporal source was folded or bypassed instead of remaining a raw map packet: {input:?}"
                ),
            ));
        }
    }
    if request.invocations[1].function != case.function
        || !input_is_serialization_of(&request.invocations[1].input, 0)
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "first toString was not a native serialization of register 0: {:?}",
                request.invocations
            ),
        ));
    }

    if case.round_trip {
        if request.output_registers != vec![1, 4]
            || request.invocations[2].function != case.function
            || !input_is_serialization_of(&request.invocations[2].input, 0)
            || request.invocations[3].function != case.function
            || request.invocations[3].input != ResidentTemporalValueInput::Register(2)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                format!(
                    "round-trip temporal serialization used the wrong SSA registers: {:?}",
                    request
                ),
            ));
        }
        match &request.invocations[4].input {
            ResidentTemporalValueInput::Comparison {
                operation: CompareOp::Eq,
                left: 3,
                right: 0,
            } if request.invocations[4].function == case.function => {}
            comparison => {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    format!(
                        "round-trip equality was not device SSA register 3 = register 0: {comparison:?}"
                    ),
                ));
            }
        }
    } else if request.output_registers != vec![1] {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "timezone serialization exposed a host/precomputed register: {:?}",
                request.output_registers
            ),
        ));
    }

    let expected_count = if case.round_trip { 2 } else { 1 };
    if receipt.result.values.len() != expected_count {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "selected backend returned the wrong number of typed serialization registers: {:?}",
                receipt.result.values
            ),
        ));
    }
    let expected_string_debug = format!("String({:?})", case.rendered);
    if format!("{:?}", receipt.result.values[0]) != expected_string_debug {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "selected backend did not return the exact native UTF-8 string register: expected {expected_string_debug}, got {:?}",
                receipt.result.values[0]
            ),
        ));
    }
    if case.round_trip && receipt.result.values[1] != ResidentTemporalValue::Boolean(true) {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "selected backend did not compare its reparsed bytes with the source value: {:?}",
                receipt.result.values
            ),
        ));
    }
    Ok(())
}

fn execute_generic_case(
    fixture: &Fixture,
    case: SerializationCase,
) -> std::result::Result<Vec<ScalarValue>, String> {
    let output = QueryEngine
        .execute(case.query, &mut context(fixture, None))
        .map_err(|error| format!("generic CPU query failed with {:?}: {error}", error.code))?;
    observe_output(&output, fixture, case)
        .map_err(|error| format!("generic CPU output inspection failed: {error}"))
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum NativeOutcome {
    Complete(Vec<ScalarValue>),
    FailClosed(String),
}

fn attempt_native_case(
    fixture: &Fixture,
    backend: &ObservedTemporalBackend,
    case: SerializationCase,
) -> std::result::Result<NativeOutcome, String> {
    let observations = backend.observations();
    let pins_before = observations.pins.load(Ordering::SeqCst);
    let temporal_before = observations.temporal_value_calls.load(Ordering::SeqCst);
    let legacy_before = observations.legacy_route_calls.load(Ordering::SeqCst);
    let unexpected_before = observations.unexpected_query_calls.load(Ordering::SeqCst);
    let mutation_before = observations
        .generation_mutation_attempts
        .load(Ordering::SeqCst);
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
    if pinned.kind() != BackendKind::Metal
        || pinned.resident_bookmark(PROJECT) != Some(fixture.bookmark)
        || pinned.resident_graph_revision(PROJECT) != Some(fixture.graph.revision())
    {
        return Err(
            "strict native generation did not retain its advertised class and exact fence"
                .to_owned(),
        );
    }

    let result = QueryEngine.execute(case.query, &mut context(fixture, Some(pinned.as_ref())));
    if observations.pins.load(Ordering::SeqCst) != pins_before + 1 {
        return Err("query did not pin exactly one immutable resident generation".to_owned());
    }
    if observations.legacy_route_calls.load(Ordering::SeqCst) != legacy_before
        || observations.unexpected_query_calls.load(Ordering::SeqCst) != unexpected_before
    {
        return Err(
            "query entered a generic, legacy, graph, row, host-evaluation, or fallback route"
                .to_owned(),
        );
    }
    if observations
        .generation_mutation_attempts
        .load(Ordering::SeqCst)
        != mutation_before
        || pinned.resident_bookmark(PROJECT) != Some(fixture.bookmark)
        || pinned.resident_graph_revision(PROJECT) != Some(fixture.graph.revision())
    {
        return Err("query attempted to mutate or replace its pinned generation".to_owned());
    }

    match result {
        Err(error) if error.code == ErrorCode::GpuAdmissionFailure => {
            let temporal_delta =
                observations.temporal_value_calls.load(Ordering::SeqCst) - temporal_before;
            let receipt_count = observations
                .receipts
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .len();
            if temporal_delta > 1 || receipt_count != receipts_before {
                return Err(format!(
                    "fail-closed query issued {temporal_delta} temporal calls or published a receipt"
                ));
            }
            Ok(NativeOutcome::FailClosed(format!(
                "{:?}: {error}",
                error.code
            )))
        }
        Err(error) => Err(format!(
            "native query failed with non-admission error {:?}: {error}",
            error.code
        )),
        Ok(output) => {
            if observations.temporal_value_calls.load(Ordering::SeqCst) != temporal_before + 1 {
                return Err(
                    "successful query did not cross exactly one native temporal-value boundary"
                        .to_owned(),
                );
            }
            let receipt = {
                let receipts = observations
                    .receipts
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner());
                if receipts.len() != receipts_before + 1 {
                    return Err(
                        "successful query did not retain exactly one native receipt".to_owned()
                    );
                }
                receipts
                    .last()
                    .cloned()
                    .ok_or_else(|| "native temporal serialization receipt disappeared".to_owned())?
            };
            assert_native_receipt(fixture, backend, case, &receipt)
                .map_err(|error| format!("native request/provenance assertion failed: {error}"))?;
            let values = observe_output(&output, fixture, case)
                .map_err(|error| format!("native output inspection failed: {error}"))?;
            Ok(NativeOutcome::Complete(values))
        }
    }
}

fn execute_native_case(
    fixture: &Fixture,
    backend: &ObservedTemporalBackend,
    case: SerializationCase,
) -> std::result::Result<Vec<ScalarValue>, String> {
    match attempt_native_case(fixture, backend, case)? {
        NativeOutcome::Complete(values) => Ok(values),
        NativeOutcome::FailClosed(error) => Err(format!(
            "native temporal serialization failed closed: {error}"
        )),
    }
}

fn assert_no_failures(backend: &str, failures: Vec<String>) {
    assert!(
        failures.is_empty(),
        "{backend} temporal serialization suite had {} failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
}

#[test]
fn manifest_pins_exact_17_temporal6_ids_names_queries_and_results() {
    assert_eq!(OFFICIAL_CASES.len(), 17);
    assert_eq!(
        OFFICIAL_CASES
            .iter()
            .map(|case| case.report_id)
            .collect::<Vec<_>>(),
        (3436_u16..=3452).collect::<Vec<_>>()
    );
    assert_eq!(
        OFFICIAL_CASES
            .iter()
            .map(|case| case.report_name)
            .collect::<Vec<_>>(),
        vec![
            "[1] Should serialize date",
            "[2] Should serialize local time",
            "[3] Should serialize time",
            "[4] Should serialize local date time",
            "[5] Should serialize date time",
            "[6] Should serialize duration [3442]",
            "[6] Should serialize duration [3443]",
            "[6] Should serialize duration [3444]",
            "[6] Should serialize duration [3445]",
            "[6] Should serialize duration [3446]",
            "[6] Should serialize duration [3447]",
            "[6] Should serialize duration [3448]",
            "[6] Should serialize duration [3449]",
            "[6] Should serialize duration [3450]",
            "[6] Should serialize duration [3451]",
            "[6] Should serialize duration [3452]",
            "[7] Should serialize timezones correctly",
        ]
    );
    assert_eq!(
        OFFICIAL_CASES
            .iter()
            .map(|case| case.rendered)
            .collect::<Vec<_>>(),
        vec![
            "1984-10-11",
            "12:31:14.645876123",
            "12:31:14.645876123+01:00",
            "1984-10-11T12:31:14.645876123",
            "1984-10-11T12:31:14.645876123+01:00",
            "P12Y5M14DT16H13M10.000000001S",
            "P12Y5M-14DT16H",
            "PT11M",
            "PT1.999S",
            "PT-1.999S",
            "PT-2.001S",
            "P1DT0.001S",
            "P1DT-0.001S",
            "PT59.999S",
            "PT-59.999S",
            "PT-1M-0.001S",
            "2017-08-08T12:31:14.645876123+02:00[Europe/Stockholm]",
        ]
    );
    assert_eq!(
        OFFICIAL_CASES
            .iter()
            .map(|case| case.query)
            .collect::<Vec<_>>(),
        vec![
            "WITH date({year: 1984, month: 10, day: 11}) AS d\nRETURN toString(d) AS ts, date(toString(d)) = d AS b",
            "WITH localtime({hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS d\nRETURN toString(d) AS ts, localtime(toString(d)) = d AS b",
            "WITH time({hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) AS d\nRETURN toString(d) AS ts, time(toString(d)) = d AS b",
            "WITH localdatetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123}) AS d\nRETURN toString(d) AS ts, localdatetime(toString(d)) = d AS b",
            "WITH datetime({year: 1984, month: 10, day: 11, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: '+01:00'}) AS d\nRETURN toString(d) AS ts, datetime(toString(d)) = d AS b",
            "WITH duration({years: 12, months: 5, days: 14, hours: 16, minutes: 12, seconds: 70, nanoseconds: 1}) AS d\nRETURN toString(d) AS ts, duration(toString(d)) = d AS b",
            "WITH duration({years: 12, months: 5, days: -14, hours: 16}) AS d\nRETURN toString(d) AS ts, duration(toString(d)) = d AS b",
            "WITH duration({minutes: 12, seconds: -60}) AS d\nRETURN toString(d) AS ts, duration(toString(d)) = d AS b",
            "WITH duration({seconds: 2, milliseconds: -1}) AS d\nRETURN toString(d) AS ts, duration(toString(d)) = d AS b",
            "WITH duration({seconds: -2, milliseconds: 1}) AS d\nRETURN toString(d) AS ts, duration(toString(d)) = d AS b",
            "WITH duration({seconds: -2, milliseconds: -1}) AS d\nRETURN toString(d) AS ts, duration(toString(d)) = d AS b",
            "WITH duration({days: 1, milliseconds: 1}) AS d\nRETURN toString(d) AS ts, duration(toString(d)) = d AS b",
            "WITH duration({days: 1, milliseconds: -1}) AS d\nRETURN toString(d) AS ts, duration(toString(d)) = d AS b",
            "WITH duration({seconds: 60, milliseconds: -1}) AS d\nRETURN toString(d) AS ts, duration(toString(d)) = d AS b",
            "WITH duration({seconds: -60, milliseconds: 1}) AS d\nRETURN toString(d) AS ts, duration(toString(d)) = d AS b",
            "WITH duration({seconds: -60, milliseconds: -1}) AS d\nRETURN toString(d) AS ts, duration(toString(d)) = d AS b",
            "WITH datetime({year: 2017, month: 8, day: 8, hour: 12, minute: 31, second: 14, nanosecond: 645876123, timezone: 'Europe/Stockholm'}) AS d\nRETURN toString(d) AS ts",
        ]
    );
    assert!(OFFICIAL_CASES[..16].iter().all(|case| case.round_trip));
    assert!(!OFFICIAL_CASES[16].round_trip);
    assert!(OFFICIAL_CASES.iter().all(|case| {
        case.query.starts_with("WITH ")
            && case.query.contains("RETURN toString(d) AS ts")
            && case.query.lines().count() == 2
    }));
}

#[test]
fn serialization_capacity_contract_covers_expanded_years_offsets_and_named_zone_bytes() {
    assert_eq!("+999999999-12-31".len(), DATE_MAX_UTF8_BYTES);
    assert_eq!("-999999999-12-31".len(), DATE_MAX_UTF8_BYTES);
    assert_eq!(LOCAL_TIME_MAX_UTF8_BYTES, "23:59:59.999999999".len());
    assert_eq!(TIME_MAX_UTF8_BYTES, LOCAL_TIME_MAX_UTF8_BYTES + 9);
    assert_eq!(
        LOCAL_DATETIME_MAX_UTF8_BYTES,
        DATE_MAX_UTF8_BYTES + 1 + LOCAL_TIME_MAX_UTF8_BYTES
    );
    assert_eq!(
        FIXED_DATETIME_MAX_UTF8_BYTES,
        LOCAL_DATETIME_MAX_UTF8_BYTES + 9
    );
    assert_eq!(
        NAMED_DATETIME_PREFIX_MAX_UTF8_BYTES,
        FIXED_DATETIME_MAX_UTF8_BYTES + 2
    );

    let stockholm_bytes = "Europe/Stockholm".len();
    let stockholm_capacity = NAMED_DATETIME_PREFIX_MAX_UTF8_BYTES + stockholm_bytes;
    assert_eq!(stockholm_bytes, 16);
    assert_eq!(stockholm_capacity, 62);
    assert_eq!(OFFICIAL_CASES[16].rendered.len(), 53);
    assert!(OFFICIAL_CASES[16].rendered.len() <= stockholm_capacity);

    // The existing Metal named-zone table stores each UTF-8 name length in u16. Expanded years,
    // a seconds-bearing offset, and brackets therefore make 46 + u16::MAX the complete wire cap.
    assert_eq!(
        NAMED_DATETIME_PREFIX_MAX_UTF8_BYTES + usize::from(u16::MAX),
        65_581
    );

    // P + signed years + signed remainder months + signed days + T + signed hours + signed
    // minutes + signed seconds/fraction. The individual maxima are 1+20+4+21+1+18+4+14.
    assert_eq!(1 + 20 + 4 + 21 + 1 + 18 + 4 + 14, DURATION_MAX_UTF8_BYTES);
    assert!(
        OFFICIAL_CASES
            .iter()
            .filter(|case| case.function == ResidentTemporalValueFunction::Duration)
            .all(|case| case.rendered.len() <= DURATION_MAX_UTF8_BYTES)
    );

    // String rows need only tag + byte length before the inline byte payload; temporal rows keep
    // the canonical eight-word prefix. This is the bounded uniform-stride contract for the ABI.
    let register_words = |capacity: usize| 8_usize.max(2 + capacity.div_ceil(4));
    assert_eq!(register_words(DATE_MAX_UTF8_BYTES), 8);
    assert_eq!(register_words(TIME_MAX_UTF8_BYTES), 9);
    assert_eq!(register_words(stockholm_capacity), 18);
    assert_eq!(register_words(DURATION_MAX_UTF8_BYTES), 23);
}

#[test]
fn generic_cpu_is_the_independent_exact_oracle_for_all_17_cases() {
    let fixture = Fixture::new();
    let mut failures = Vec::new();
    for case in OFFICIAL_CASES {
        match execute_generic_case(&fixture, case) {
            Ok(values) if values == case.expected_scalars() => {}
            Ok(values) => failures.push(format!(
                "{}: expected {:?}, got {values:?}",
                case.label(),
                case.expected_scalars()
            )),
            Err(error) => failures.push(format!("{}: {error}", case.label())),
        }
    }
    assert_no_failures("generic CPU oracle", failures);
}

#[test]
fn active_native_boundary_is_uniformly_fail_closed_or_complete_without_fallback() -> Result<()> {
    let fixture = Fixture::new();
    let backend = fixture.strict_cpu_backend()?;
    assert_eq!(backend.kind(), BackendKind::Metal);
    assert_eq!(backend.actual_kind, BackendKind::Cpu);

    let mut complete = 0_usize;
    let mut fail_closed = 0_usize;
    let mut failures = Vec::new();
    for case in OFFICIAL_CASES {
        match attempt_native_case(&fixture, &backend, case) {
            Ok(NativeOutcome::Complete(values)) if values == case.expected_scalars() => {
                complete += 1;
            }
            Ok(NativeOutcome::Complete(values)) => failures.push(format!(
                "{}: complete route returned {:?}, expected {:?}",
                case.label(),
                values,
                case.expected_scalars()
            )),
            Ok(NativeOutcome::FailClosed(_)) => fail_closed += 1,
            Err(error) => failures.push(format!("{}: {error}", case.label())),
        }
    }
    assert_no_failures("active native boundary", failures);
    assert!(
        (complete == 17 && fail_closed == 0) || (complete == 0 && fail_closed == 17),
        "Temporal6 must be one coherent native tranche, not a partial prefix: complete={complete}, fail_closed={fail_closed}"
    );

    let observations = backend.observations();
    let temporal_calls = observations.temporal_value_calls.load(Ordering::SeqCst);
    assert_eq!(observations.pins.load(Ordering::SeqCst), 17);
    assert!(
        temporal_calls == 0 || temporal_calls == 17,
        "partial temporal dispatch is not an accepted boundary: {temporal_calls}/17"
    );
    assert_eq!(observations.legacy_route_calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        observations.unexpected_query_calls.load(Ordering::SeqCst),
        0
    );
    assert_eq!(
        observations
            .generation_mutation_attempts
            .load(Ordering::SeqCst),
        0
    );
    assert_eq!(
        observations
            .receipts
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len(),
        complete
    );
    Ok(())
}

#[test]
#[ignore = "acceptance gate: requires the complete temporal serialization/string-register ABI"]
fn strict_cpu_runs_all_17_temporal6_cases_in_one_native_program_each() -> Result<()> {
    let fixture = Fixture::new();
    let backend = fixture.strict_cpu_backend()?;
    let mut failures = Vec::new();
    for case in OFFICIAL_CASES {
        match execute_native_case(&fixture, &backend, case) {
            Ok(values) if values == case.expected_scalars() => {}
            Ok(values) => failures.push(format!(
                "{}: expected {:?}, got {values:?}",
                case.label(),
                case.expected_scalars()
            )),
            Err(error) => failures.push(format!("{}: {error}", case.label())),
        }
    }
    assert_no_failures("strict CPU reference", failures);
    let observations = backend.observations();
    assert_eq!(observations.pins.load(Ordering::SeqCst), 17);
    assert_eq!(observations.temporal_value_calls.load(Ordering::SeqCst), 17);
    assert_eq!(observations.legacy_route_calls.load(Ordering::SeqCst), 0);
    assert_eq!(
        observations.unexpected_query_calls.load(Ordering::SeqCst),
        0
    );
    let receipts = observations
        .receipts
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    assert_eq!(receipts.len(), 17);
    assert!(
        receipts
            .iter()
            .all(|receipt| receipt.completion == BackendKind::Cpu)
    );
    Ok(())
}

#[test]
#[ignore = "external assurance gate: requires the pinned full TCK JSON report"]
fn baseline_report_uniquely_resolves_all_17_temporal6_identities() {
    let report: serde_json::Value = serde_json::from_slice(
        &std::fs::read(BASELINE_REPORT).expect("baseline TCK report is readable"),
    )
    .expect("baseline TCK report is valid JSON");
    let scenarios = report["scenarios"]
        .as_array()
        .expect("baseline report has a scenario array");
    assert_eq!(scenarios.len(), 3_897);
    for case in OFFICIAL_CASES {
        let matches = scenarios
            .iter()
            .enumerate()
            .filter_map(|(index, scenario)| {
                let path = scenario.get("path")?.as_str()?;
                let name = scenario.get("name")?.as_str()?;
                (path.ends_with(FEATURE) && name == case.report_name).then_some(index)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            matches,
            vec![usize::from(case.report_id)],
            "{}",
            case.label()
        );
    }
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
#[ignore = "focused hardware smoke gate: requires real Metal"]
fn real_metal_runs_first_temporal6_case_without_fallback() -> Result<()> {
    let _guard = metal_test_guard();
    let fixture = Fixture::new();
    let metal = fixture.real_metal_backend()?;
    let case = OFFICIAL_CASES[0];
    let values = execute_native_case(&fixture, &metal, case).map_err(Error::internal)?;
    assert_eq!(values, case.expected_scalars());
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "hardware acceptance gate: exact Temporal6 suite requires real Metal"]
fn real_metal_matches_strict_cpu_with_one_native_program_and_no_fallback() -> Result<()> {
    let _guard = metal_test_guard();
    let fixture = Fixture::new();
    let cpu = fixture.strict_cpu_backend()?;
    let metal = fixture.real_metal_backend()?;
    assert_eq!(cpu.actual_kind, BackendKind::Cpu);
    assert_eq!(metal.actual_kind, BackendKind::Metal);

    let mut failures = Vec::new();
    for case in OFFICIAL_CASES {
        let cpu_result = execute_native_case(&fixture, &cpu, case);
        let metal_result = execute_native_case(&fixture, &metal, case);
        match (cpu_result, metal_result) {
            (Ok(cpu_values), Ok(metal_values))
                if cpu_values == case.expected_scalars()
                    && metal_values == case.expected_scalars()
                    && cpu_values == metal_values => {}
            (Ok(cpu_values), Ok(metal_values)) => failures.push(format!(
                "{}: CPU={cpu_values:?}, Metal={metal_values:?}, expected={:?}",
                case.label(),
                case.expected_scalars()
            )),
            (Err(cpu_error), Ok(_)) => {
                failures.push(format!("{}: strict CPU failed: {cpu_error}", case.label()))
            }
            (Ok(_), Err(metal_error)) => failures.push(format!(
                "{}: real Metal failed: {metal_error}",
                case.label()
            )),
            (Err(cpu_error), Err(metal_error)) => failures.push(format!(
                "{}: strict CPU failed: {cpu_error}; real Metal failed: {metal_error}",
                case.label()
            )),
        }
    }
    assert_no_failures("strict CPU versus real Metal", failures);

    for (name, backend, completion) in [
        ("CPU", &cpu, BackendKind::Cpu),
        ("Metal", &metal, BackendKind::Metal),
    ] {
        let observations = backend.observations();
        assert_eq!(observations.pins.load(Ordering::SeqCst), 17, "{name}");
        assert_eq!(
            observations.temporal_value_calls.load(Ordering::SeqCst),
            17,
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
        assert_eq!(receipts.len(), 17, "{name}");
        assert!(
            receipts.iter().all(|receipt| {
                receipt.completion == completion
                    && receipt.bookmark == fixture.bookmark
                    && receipt.graph_revision == fixture.graph.revision()
            }),
            "{name} receipt provenance or immutable fence changed"
        );
    }
    Ok(())
}
