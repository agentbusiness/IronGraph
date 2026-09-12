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
    Bookmark, DocumentItem, DocumentList, Error, ErrorCode, Layer, NodeId, ProjectId, Result,
    ScalarValue,
    cypher::{
        BindCapabilities, ColumnType, ExecutionContext, ExecutionOutput, QueryEngine, ResultValue,
        StatementStats,
    },
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend,
        ResidentCreateNodeRequest, ResidentCreateNodeResult, ResidentCreateNodeTemporalListElement,
        ResidentCreateNodeValueInput, ResidentDeviceCompletion, ResidentGroup,
        ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest, ResidentNodePipelineRequest,
        ResidentNodePipelineResult, ResidentProjectImage, ResidentRowProgramRequest,
        ResidentRowProgramResult, ResidentScalarProgramRequest, ResidentScalarProgramResult,
        ResidentSortRequest, ResidentSortResult, ResidentTemporalValueProgramRequest,
        ResidentTemporalValueProgramResult, ResidentVectorQuery, ResidentVectorResult,
        ScratchReservation, ValidatedResidentCreateNode,
    },
    graph::{GraphMutation, GraphStore, IndexCatalog, LayerMask, TemporalStore},
    types::{LabelId, PropertyId},
};
use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::gpu::MetalBackend;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());
const BOOKMARK: Bookmark = Bookmark {
    term: 23,
    index: 67,
};
const FIRST_NODE_ID: u64 = 700;
const MEMORY_LIMIT_BYTES: usize = 128 * 1024 * 1024;
const RESERVED_BYTES: usize = 16 * 1024 * 1024;
const MAX_RESULT_ROWS: usize = 16;
const FEATURE: &str = "features/expressions/temporal/Temporal4.feature";
const CERTIFIED_TCK_REPORT: &str = "/tmp/irongraph-tck-full-delete-temporal-cpu-certified.json";

fn assert_certified_report_identities<'a>(
    identities: impl IntoIterator<Item = (usize, &'a str, &'a str)>,
) {
    let report: serde_json::Value = serde_json::from_slice(
        &std::fs::read(CERTIFIED_TCK_REPORT).expect("certified TCK report is readable"),
    )
    .expect("certified TCK report is valid JSON");
    assert_eq!(report["total"].as_u64(), Some(3_897));
    assert_eq!(report["cpu_passed"].as_u64(), Some(3_897));
    assert_eq!(report["metal_passed"].as_u64(), Some(3_184));
    let scenarios = report["scenarios"]
        .as_array()
        .expect("certified report has a scenario array");
    assert_eq!(scenarios.len(), 3_897);
    let mut selected = std::collections::BTreeSet::new();
    for (stored_id, feature, expanded_name) in identities {
        assert!(
            selected.insert((feature, expanded_name)),
            "duplicate local TCK identity ({feature}, {expanded_name})"
        );
        let matches = scenarios
            .iter()
            .enumerate()
            .filter_map(|(index, scenario)| {
                let path = scenario.get("path")?.as_str()?;
                let name = scenario.get("name")?.as_str()?;
                (path.ends_with(feature) && name == expanded_name).then_some(index)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            matches.len(),
            1,
            "({feature}, {expanded_name}) resolved to {matches:?}"
        );
        assert_eq!(
            stored_id, matches[0],
            "wrong report index for {expanded_name}"
        );
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TemporalValue {
    Date(i64),
    LocalTime(i64),
    Time {
        nanos: i64,
        offset_seconds: i32,
    },
    LocalDateTime {
        seconds: i64,
        nanos: u32,
    },
    DateTime {
        seconds: i64,
        nanos: u32,
        timezone: &'static str,
    },
    Duration {
        months: i64,
        days: i64,
        seconds: i64,
        nanos: i32,
    },
}

impl TemporalValue {
    fn scalar(self) -> ScalarValue {
        match self {
            Self::Date(days) => ScalarValue::Date(days),
            Self::LocalTime(nanos) => ScalarValue::LocalTime(nanos),
            Self::Time {
                nanos,
                offset_seconds,
            } => ScalarValue::ZonedTime {
                nanos,
                offset_seconds,
            },
            Self::LocalDateTime { seconds, nanos } => ScalarValue::LocalDateTime { seconds, nanos },
            Self::DateTime {
                seconds,
                nanos,
                timezone,
            } => ScalarValue::ZonedDateTime {
                seconds,
                nanos,
                timezone: Arc::from(timezone),
            },
            Self::Duration {
                months,
                days,
                seconds,
                nanos,
            } => ScalarValue::Duration {
                months,
                days,
                seconds,
                nanos,
            },
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct TemporalArrayCase {
    report_id: u16,
    outline: u8,
    example: u8,
    name: &'static str,
    expression: &'static str,
    values: &'static [TemporalValue],
}

impl TemporalArrayCase {
    fn query(self) -> String {
        format!("CREATE ({{dates: {}}})", self.expression)
    }

    fn label(self) -> String {
        format!(
            "TCK {} {FEATURE} {} (outline {}, example {})",
            self.report_id, self.name, self.outline, self.example
        )
    }

    fn document(self) -> Result<DocumentList> {
        DocumentList::new(
            self.values
                .iter()
                .copied()
                .map(|value| DocumentItem::Scalar(value.scalar()))
                .collect(),
        )
    }

    fn result_value(self) -> ResultValue {
        ResultValue::List(
            self.values
                .iter()
                .copied()
                .map(|value| ResultValue::Scalar(value.scalar()))
                .collect(),
        )
    }
}

const CASES: [TemporalArrayCase; 12] = [
    TemporalArrayCase {
        report_id: 3391,
        outline: 2,
        example: 1,
        name: "[2] Should store date array [3392]",
        expression: "[date({year: 1984, month: 10, day: 12})]",
        values: &[TemporalValue::Date(5_398)],
    },
    TemporalArrayCase {
        report_id: 3392,
        outline: 2,
        example: 2,
        name: "[2] Should store date array [3393]",
        expression: "[date({year: 1984, month: 10, day: 13}), date({year: 1984, month: 10, day: 14}), date({year: 1984, month: 10, day: 15})]",
        values: &[
            TemporalValue::Date(5_399),
            TemporalValue::Date(5_400),
            TemporalValue::Date(5_401),
        ],
    },
    TemporalArrayCase {
        report_id: 3394,
        outline: 4,
        example: 1,
        name: "[4] Should store local time array [3395]",
        expression: "[localtime({hour: 13})]",
        values: &[TemporalValue::LocalTime(46_800_000_000_000)],
    },
    TemporalArrayCase {
        report_id: 3395,
        outline: 4,
        example: 2,
        name: "[4] Should store local time array [3396]",
        expression: "[localtime({hour: 14}), localtime({hour: 15}), localtime({hour: 16})]",
        values: &[
            TemporalValue::LocalTime(50_400_000_000_000),
            TemporalValue::LocalTime(54_000_000_000_000),
            TemporalValue::LocalTime(57_600_000_000_000),
        ],
    },
    TemporalArrayCase {
        report_id: 3397,
        outline: 6,
        example: 1,
        name: "[6] Should store time array [3398]",
        expression: "[time({hour: 13})]",
        values: &[TemporalValue::Time {
            nanos: 46_800_000_000_000,
            offset_seconds: 0,
        }],
    },
    TemporalArrayCase {
        report_id: 3398,
        outline: 6,
        example: 2,
        name: "[6] Should store time array [3399]",
        expression: "[time({hour: 14}), time({hour: 15}), time({hour: 16})]",
        values: &[
            TemporalValue::Time {
                nanos: 50_400_000_000_000,
                offset_seconds: 0,
            },
            TemporalValue::Time {
                nanos: 54_000_000_000_000,
                offset_seconds: 0,
            },
            TemporalValue::Time {
                nanos: 57_600_000_000_000,
                offset_seconds: 0,
            },
        ],
    },
    TemporalArrayCase {
        report_id: 3400,
        outline: 8,
        example: 1,
        name: "[8] Should store local date time array [3401]",
        expression: "[localdatetime({year: 1913})]",
        values: &[TemporalValue::LocalDateTime {
            seconds: -1_798_761_600,
            nanos: 0,
        }],
    },
    TemporalArrayCase {
        report_id: 3401,
        outline: 8,
        example: 2,
        name: "[8] Should store local date time array [3402]",
        expression: "[localdatetime({year: 1914}), localdatetime({year: 1915}), localdatetime({year: 1916})]",
        values: &[
            TemporalValue::LocalDateTime {
                seconds: -1_767_225_600,
                nanos: 0,
            },
            TemporalValue::LocalDateTime {
                seconds: -1_735_689_600,
                nanos: 0,
            },
            TemporalValue::LocalDateTime {
                seconds: -1_704_153_600,
                nanos: 0,
            },
        ],
    },
    TemporalArrayCase {
        report_id: 3403,
        outline: 10,
        example: 1,
        name: "[10] Should store date time array [3404]",
        expression: "[datetime({year: 1913})]",
        values: &[TemporalValue::DateTime {
            seconds: -1_798_761_600,
            nanos: 0,
            timezone: "UTC",
        }],
    },
    TemporalArrayCase {
        report_id: 3404,
        outline: 10,
        example: 2,
        name: "[10] Should store date time array [3405]",
        expression: "[datetime({year: 1914}), datetime({year: 1915}), datetime({year: 1916})]",
        values: &[
            TemporalValue::DateTime {
                seconds: -1_767_225_600,
                nanos: 0,
                timezone: "UTC",
            },
            TemporalValue::DateTime {
                seconds: -1_735_689_600,
                nanos: 0,
                timezone: "UTC",
            },
            TemporalValue::DateTime {
                seconds: -1_704_153_600,
                nanos: 0,
                timezone: "UTC",
            },
        ],
    },
    TemporalArrayCase {
        report_id: 3406,
        outline: 12,
        example: 1,
        name: "[12] Should store duration array [3407]",
        expression: "[duration({seconds: 13})]",
        values: &[TemporalValue::Duration {
            months: 0,
            days: 0,
            seconds: 13,
            nanos: 0,
        }],
    },
    TemporalArrayCase {
        report_id: 3407,
        outline: 12,
        example: 2,
        name: "[12] Should store duration array [3408]",
        expression: "[duration({seconds: 14}), duration({seconds: 15}), duration({seconds: 16})]",
        values: &[
            TemporalValue::Duration {
                months: 0,
                days: 0,
                seconds: 14,
                nanos: 0,
            },
            TemporalValue::Duration {
                months: 0,
                days: 0,
                seconds: 15,
                nanos: 0,
            },
            TemporalValue::Duration {
                months: 0,
                days: 0,
                seconds: 16,
                nanos: 0,
            },
        ],
    },
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AllowedRoute {
    Create,
    Readback,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FaultMode {
    None,
    ReplayFirstCreate,
}

#[derive(Default)]
struct RouteObservations {
    pins: AtomicUsize,
    create_calls: AtomicUsize,
    pipeline_calls: AtomicUsize,
    temporal_program_calls: AtomicUsize,
    scalar_program_calls: AtomicUsize,
    row_program_calls: AtomicUsize,
    unexpected_calls: AtomicUsize,
    generation_mutation_attempts: AtomicUsize,
    create_requests: Mutex<Vec<ResidentCreateNodeRequest>>,
    raw_create_results: Mutex<Vec<ResidentCreateNodeResult>>,
    validated_create_results: Mutex<Vec<ValidatedResidentCreateNode>>,
    pipeline_requests: Mutex<Vec<ResidentNodePipelineRequest>>,
    first_create_result: Mutex<Option<ResidentCreateNodeResult>>,
}

/// Fail-closed observer for either the CREATE boundary or the control MATCH boundary.
///
/// The root object advertises Metal to force strict planning. Its pinned generation reports the
/// honest completion kind, except in the explicit CPU-as-Metal sabotage case. No generic graph,
/// scalar, temporal, or typed-row entrypoint is open alongside the selected boundary.
struct ObservedBackend {
    inner: Box<dyn ExecutionBackend>,
    pinned_kind: BackendKind,
    actual_kind: BackendKind,
    pinned: bool,
    route: AllowedRoute,
    fault: FaultMode,
    expected_bookmark: Bookmark,
    expected_graph_revision: u64,
    observations: Arc<RouteObservations>,
}

impl ObservedBackend {
    fn new<B: ExecutionBackend + 'static>(
        inner: B,
        pinned_kind: BackendKind,
        actual_kind: BackendKind,
        route: AllowedRoute,
        fault: FaultMode,
    ) -> Result<Self> {
        if inner.kind() != actual_kind {
            return Err(Error::internal(
                "temporal-array observer was given the wrong physical backend",
            ));
        }
        let expected_bookmark = inner.resident_bookmark(PROJECT).ok_or_else(|| {
            Error::internal("temporal-array backend has no admitted project bookmark")
        })?;
        let expected_graph_revision = inner.resident_graph_revision(PROJECT).ok_or_else(|| {
            Error::internal("temporal-array backend has no admitted graph revision")
        })?;
        Ok(Self {
            inner: Box::new(inner),
            pinned_kind,
            actual_kind,
            pinned: false,
            route,
            fault,
            expected_bookmark,
            expected_graph_revision,
            observations: Arc::new(RouteObservations::default()),
        })
    }

    fn strict_cpu_create(graph: &GraphStore, fault: FaultMode) -> Result<Self> {
        Self::new(
            admitted_cpu(graph)?,
            BackendKind::Cpu,
            BackendKind::Cpu,
            AllowedRoute::Create,
            fault,
        )
    }

    fn cpu_masquerading_as_metal(graph: &GraphStore) -> Result<Self> {
        Self::new(
            admitted_cpu(graph)?,
            BackendKind::Metal,
            BackendKind::Cpu,
            AllowedRoute::Create,
            FaultMode::None,
        )
    }

    fn strict_cpu_readback(graph: &GraphStore) -> Result<Self> {
        Self::new(
            admitted_cpu(graph)?,
            BackendKind::Cpu,
            BackendKind::Cpu,
            AllowedRoute::Readback,
            FaultMode::None,
        )
    }

    #[cfg(all(feature = "accelerator", target_os = "macos"))]
    fn real_metal(graph: &GraphStore, route: AllowedRoute) -> Result<Self> {
        let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
        metal.admit_project(project_image(graph)?)?;
        Self::new(
            metal,
            BackendKind::Metal,
            BackendKind::Metal,
            route,
            FaultMode::None,
        )
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
            format!("strict temporal-array gate rejected `{route}` execution"),
        ))
    }

    fn reject_generation_mutation<T>(&self, route: &'static str) -> Result<T> {
        self.observations
            .generation_mutation_attempts
            .fetch_add(1, Ordering::SeqCst);
        Err(Error::new(
            ErrorCode::CorruptStorage,
            format!("pinned temporal-array generation rejected `{route}` mutation"),
        ))
    }
}

impl ExecutionBackend for ObservedBackend {
    fn kind(&self) -> BackendKind {
        if self.pinned {
            self.pinned_kind
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
            return self.reject("pin_project_twice");
        }
        if project != PROJECT {
            return self.reject("pin_wrong_project");
        }
        let inner = self.inner.pin_project(project)?;
        if inner.kind() != self.actual_kind
            || inner.resident_bookmark(project) != Some(self.expected_bookmark)
            || inner.resident_graph_revision(project) != Some(self.expected_graph_revision)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "temporal-array query pinned the wrong immutable generation",
            ));
        }
        self.observations.pins.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Self {
            inner,
            pinned_kind: self.pinned_kind,
            actual_kind: self.actual_kind,
            pinned: true,
            route: self.route,
            fault: self.fault,
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
        self.expected_bookmark = self
            .inner
            .resident_bookmark(PROJECT)
            .ok_or_else(|| Error::internal("replacement temporal-array project has no bookmark"))?;
        self.expected_graph_revision =
            self.inner.resident_graph_revision(PROJECT).ok_or_else(|| {
                Error::internal("replacement temporal-array project has no graph revision")
            })?;
        Ok(())
    }

    fn replace_all_projects(&mut self, images: Vec<ResidentProjectImage>) -> Result<()> {
        if self.pinned {
            return self.reject_generation_mutation("replace_all_projects");
        }
        self.inner.replace_all_projects(images)?;
        self.expected_bookmark = self
            .inner
            .resident_bookmark(PROJECT)
            .ok_or_else(|| Error::internal("replacement temporal-array project has no bookmark"))?;
        self.expected_graph_revision =
            self.inner.resident_graph_revision(PROJECT).ok_or_else(|| {
                Error::internal("replacement temporal-array project has no graph revision")
            })?;
        Ok(())
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
        if !self.pinned || self.route != AllowedRoute::Readback {
            return self.reject("execute_node_pipeline");
        }
        if request.project != PROJECT
            || request.mutation.is_some()
            || self.inner.resident_bookmark(PROJECT) != Some(self.expected_bookmark)
            || self.inner.resident_graph_revision(PROJECT) != Some(self.expected_graph_revision)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "temporal-array readback escaped its immutable resident generation",
            ));
        }
        self.observations
            .pipeline_calls
            .fetch_add(1, Ordering::SeqCst);
        self.observations
            .pipeline_requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(request.clone());
        self.inner.execute_node_pipeline(request, cancellation)
    }

    fn supports_native_create_node(&self) -> bool {
        self.route == AllowedRoute::Create
    }

    fn execute_create_node(
        &self,
        request: &ResidentCreateNodeRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentCreateNodeResult> {
        if !self.pinned || self.route != AllowedRoute::Create {
            return self.reject("execute_create_node");
        }
        if request.project != PROJECT
            || request.expected_bookmark != self.expected_bookmark
            || request.expected_graph_revision != self.expected_graph_revision
            || self.inner.resident_bookmark(PROJECT) != Some(self.expected_bookmark)
            || self.inner.resident_graph_revision(PROJECT) != Some(self.expected_graph_revision)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "temporal-array CREATE escaped its immutable resident generation",
            ));
        }
        request.validate()?;
        let call = self
            .observations
            .create_calls
            .fetch_add(1, Ordering::SeqCst);
        self.observations
            .create_requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(request.clone());

        if self.fault == FaultMode::ReplayFirstCreate && call > 0 {
            return self
                .observations
                .first_create_result
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone()
                .ok_or_else(|| Error::internal("first CREATE result disappeared during replay"));
        }

        let result = self.inner.execute_create_node(request, cancellation)?;
        self.observations
            .raw_create_results
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(result.clone());
        let validated = result
            .clone()
            .validate_for_publication(request, self.actual_kind)?;
        self.observations
            .validated_create_results
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(validated);
        if call == 0 {
            *self
                .observations
                .first_create_result
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(result.clone());
        }
        Ok(result)
    }

    fn execute_row_program(
        &self,
        _request: &ResidentRowProgramRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentRowProgramResult> {
        self.observations
            .row_program_calls
            .fetch_add(1, Ordering::SeqCst);
        self.reject("execute_row_program")
    }

    fn supports_native_temporal_value_program(&self) -> bool {
        false
    }

    fn execute_temporal_value_program(
        &self,
        _request: &ResidentTemporalValueProgramRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentTemporalValueProgramResult> {
        self.observations
            .temporal_program_calls
            .fetch_add(1, Ordering::SeqCst);
        self.reject("execute_temporal_value_program_outside_create")
    }

    fn supports_native_scalar_program(&self) -> bool {
        false
    }

    fn execute_scalar_program(
        &self,
        _request: &ResidentScalarProgramRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentScalarProgramResult> {
        self.observations
            .scalar_program_calls
            .fetch_add(1, Ordering::SeqCst);
        self.reject("execute_scalar_program_outside_create")
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

fn project_image(graph: &GraphStore) -> Result<ResidentProjectImage> {
    ResidentProjectImage::build(
        PROJECT,
        BOOKMARK,
        graph,
        &TemporalStore::default(),
        &IndexCatalog::default(),
    )
}

fn admitted_cpu(graph: &GraphStore) -> Result<CpuBackend> {
    let mut cpu = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
    cpu.admit_project(project_image(graph)?)?;
    Ok(cpu)
}

fn context<'a>(
    graph: &'a GraphStore,
    backend: Option<&'a dyn ExecutionBackend>,
    write: bool,
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
        bookmark: BOOKMARK,
        mutation_revision: graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id: FIRST_NODE_ID,
        next_edge_id: 1,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            write,
            require_native_execution,
            ..BindCapabilities::default()
        },
        max_result_rows: MAX_RESULT_ROWS,
        max_batch_rows: MAX_RESULT_ROWS,
        optimizer_statistics: None,
        backend,
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(30)),
        resolved_query_at_time_nanos: None,
    }
}

fn apply_mutations(graph: &mut GraphStore, output: &ExecutionOutput) -> Result<()> {
    for mutation in &output.graph_mutations {
        graph.apply(mutation.clone())?;
    }
    Ok(())
}

fn assert_exact_create_output(
    graph: &GraphStore,
    output: &ExecutionOutput,
    case: TemporalArrayCase,
) -> Result<()> {
    let label = case.label();
    let expected_stats = StatementStats {
        nodes_created: 1,
        ..StatementStats::default()
    };
    if !output.result.schema.is_empty()
        || !output.result.batches.is_empty()
        || output.result.statistics != expected_stats
        || output.result.bookmark != BOOKMARK
        || output.result.truncated
        || !output.temporal_mutations.is_empty()
        || output.administrative.is_some()
        || !output.vector_searches.is_empty()
        || output.runtime_replans != 0
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!("{label}: CREATE result, statistics, or truncation state changed: {output:#?}"),
        ));
    }

    let mut property_id = None;
    let mut inserted = None;
    for mutation in &output.graph_mutations {
        match mutation {
            GraphMutation::DeclareProperty { name, id } if name == "dates" => {
                if property_id.replace(*id).is_some() {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        format!("{label}: duplicate `dates` declaration"),
                    ));
                }
            }
            GraphMutation::InsertNode(node) => {
                if inserted.replace(node).is_some() {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        format!("{label}: CREATE emitted more than one node"),
                    ));
                }
            }
            other => {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    format!("{label}: CREATE emitted an unexpected mutation: {other:?}"),
                ));
            }
        }
    }
    let property_id = property_id.ok_or_else(|| {
        Error::new(
            ErrorCode::CorruptStorage,
            format!("{label}: CREATE omitted the `dates` property declaration"),
        )
    })?;
    let inserted = inserted.ok_or_else(|| {
        Error::new(
            ErrorCode::CorruptStorage,
            format!("{label}: CREATE omitted the canonical node mutation"),
        )
    })?;
    if inserted.id != NodeId(FIRST_NODE_ID)
        || inserted.layer != Layer::Observed
        || inserted.revision != graph.revision().saturating_add(1)
        || !inserted.labels.is_empty()
        || inserted.properties.len() != 1
        || inserted.properties[0].0 != property_id
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!("{label}: canonical node identity or property shape changed: {inserted:?}"),
        ));
    }

    let expected = case.document()?;
    let ScalarValue::List(actual) = &inserted.properties[0].1 else {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "{label}: temporal array was not stored as a canonical flat LIST: {:?}",
                inserted.properties[0].1
            ),
        ));
    };
    if actual != &expected
        || actual.as_bytes() != expected.as_bytes()
        || actual.items()?
            != case
                .values
                .iter()
                .copied()
                .map(|value| DocumentItem::Scalar(value.scalar()))
                .collect::<Vec<_>>()
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "{label}: canonical LIST bytes, element order, or temporal element types changed"
            ),
        ));
    }
    Ok(())
}

fn assert_route_closed(observations: &RouteObservations, label: &str) -> Result<()> {
    if observations.unexpected_calls.load(Ordering::SeqCst) != 0
        || observations.temporal_program_calls.load(Ordering::SeqCst) != 0
        || observations.scalar_program_calls.load(Ordering::SeqCst) != 0
        || observations.row_program_calls.load(Ordering::SeqCst) != 0
        || observations
            .generation_mutation_attempts
            .load(Ordering::SeqCst)
            != 0
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "{label}: execution entered a host/generic route, an external temporal/list builder, or mutated its pinned generation"
            ),
        ));
    }
    Ok(())
}

fn assert_control_readback(
    graph: &GraphStore,
    backend: &ObservedBackend,
    case: TemporalArrayCase,
) -> Result<ExecutionOutput> {
    let label = case.label();
    let observations = backend.observations();
    let output = QueryEngine
        .execute(
            "MATCH (n) RETURN n.dates",
            &mut context(graph, Some(backend), false, true),
        )
        .map_err(|error| {
            Error::new(
                error.code,
                format!("{label}: selected-backend control MATCH failed: {error}"),
            )
        })?;
    if observations.pins.load(Ordering::SeqCst) != 1
        || observations.pipeline_calls.load(Ordering::SeqCst) != 1
        || observations.create_calls.load(Ordering::SeqCst) != 0
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "{label}: control MATCH did not use exactly one pinned resident pipeline; pins={}, pipelines={}, creates={}",
                observations.pins.load(Ordering::SeqCst),
                observations.pipeline_calls.load(Ordering::SeqCst),
                observations.create_calls.load(Ordering::SeqCst),
            ),
        ));
    }
    assert_route_closed(&observations, &label)?;
    let requests = observations
        .pipeline_requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if requests.len() != 1
        || requests[0].project != PROJECT
        || requests[0].mutation.is_some()
        || requests[0].expansion.is_some()
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!("{label}: control MATCH request changed shape: {requests:#?}"),
        ));
    }
    let expected = case.result_value();
    if output.result.schema != vec![("n.dates".to_owned(), ColumnType::List)]
        || output.result.batches.len() != 1
        || output.result.batches[0].row_count != 1
        || output.result.batches[0].columns.len() != 1
        || output.result.batches[0].columns[0].name != "n.dates"
        || output.result.batches[0].columns[0].value_type != ColumnType::List
        || output.result.batches[0].columns[0].values != vec![expected]
        || output.result.truncated
        || !output.graph_mutations.is_empty()
        || !output.temporal_mutations.is_empty()
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "{label}: control MATCH changed list order/type or truncated the result: {output:#?}"
            ),
        ));
    }
    Ok(output)
}

fn run_cpu_case(case: TemporalArrayCase) -> Result<()> {
    let mut graph = GraphStore::default();
    let cpu = admitted_cpu(&graph)?;
    if cpu.kind() != BackendKind::Cpu {
        return Err(Error::internal("CPU Temporal4 oracle is not a CPU backend"));
    }
    let output = QueryEngine
        .execute(&case.query(), &mut context(&graph, Some(&cpu), true, false))
        .map_err(|error| {
            Error::new(
                error.code,
                format!("{}: CPU CREATE failed: {error}", case.label()),
            )
        })?;
    assert_exact_create_output(&graph, &output, case)?;
    apply_mutations(&mut graph, &output)?;

    let dates = graph.catalog().property("dates").ok_or_else(|| {
        Error::new(
            ErrorCode::CorruptStorage,
            format!("{}: published graph lost `dates`", case.label()),
        )
    })?;
    let stored = graph
        .node(NodeId(FIRST_NODE_ID))
        .and_then(|node| node.property(dates))
        .ok_or_else(|| {
            Error::new(
                ErrorCode::CorruptStorage,
                format!("{}: published graph lost the LIST property", case.label()),
            )
        })?;
    if stored != ScalarValue::List(case.document()?) {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!("{}: published canonical LIST changed", case.label()),
        ));
    }

    let readback = ObservedBackend::strict_cpu_readback(&graph)?;
    assert_control_readback(&graph, &readback, case)?;
    Ok(())
}

fn assert_native_array_boundary(
    graph: &GraphStore,
    backend: &ObservedBackend,
    case: TemporalArrayCase,
) -> Result<()> {
    let label = case.label();
    let observations = backend.observations();
    if observations.pins.load(Ordering::SeqCst) != 1
        || observations.create_calls.load(Ordering::SeqCst) != 1
        || observations.pipeline_calls.load(Ordering::SeqCst) != 0
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "{label}: CREATE did not use exactly one pinned native intent; pins={}, creates={}, pipelines={}",
                observations.pins.load(Ordering::SeqCst),
                observations.create_calls.load(Ordering::SeqCst),
                observations.pipeline_calls.load(Ordering::SeqCst),
            ),
        ));
    }
    assert_route_closed(&observations, &label)?;

    let requests = observations
        .create_requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let validated = observations
        .validated_create_results
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if requests.len() != 1 || validated.len() != 1 {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!("{label}: CREATE request or validated intent disappeared"),
        ));
    }
    let request = &requests[0];
    let validated = &validated[0];
    request.validate()?;
    if request.project != PROJECT
        || request.expected_bookmark != BOOKMARK
        || request.expected_graph_revision != graph.revision()
        || request.expected_layout_version != graph.layout_version()
        || request.operation.node_id != NodeId(FIRST_NODE_ID)
        || request.operation.properties.len() != 1
        || request.property_names != ["dates"]
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!("{label}: native array request has the wrong immutable shape: {request:#?}"),
        ));
    }

    let property = &request.operation.properties[0];
    let ResidentCreateNodeValueInput::TemporalList(syntax) = &property.value else {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "{label}: compiler did not retain one raw temporal-LIST program: {:?}",
                property.value
            ),
        ));
    };
    if syntax.elements.len() != case.values.len()
        || syntax
            .elements
            .iter()
            .any(|element| !matches!(element, ResidentCreateNodeTemporalListElement::Temporal(_)))
        || (syntax.maximum_document_bytes as usize) < case.document()?.as_bytes().len()
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "{label}: raw temporal-LIST shape/capacity lost an element or admitted host materialization"
            ),
        ));
    }

    // One proof per temporal constructor plus one proof for flat canonical LIST construction.
    // Selection and final CREATE publication are separate obligations checked below.
    let expected_expression_receipts = case.values.len() + 1;
    let expression_scopes = request
        .expression_obligations
        .iter()
        .map(|obligation| obligation.scope)
        .collect::<BTreeSet<_>>();
    if request.expression_obligations.len() != expected_expression_receipts
        || expression_scopes.len() != expected_expression_receipts
        || request.expression_obligations.iter().any(|obligation| {
            obligation.kind != irongraph::gpu::ResidentObligationKind::Expression
                || obligation.id == 0
        })
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "{label}: expected {} distinct device expression receipts ({} constructors + LIST builder), got {:?}",
                expected_expression_receipts,
                case.values.len(),
                request.expression_obligations,
            ),
        ));
    }

    let expected_completion = match backend.actual_kind {
        BackendKind::Cpu => ResidentDeviceCompletion::CpuReference,
        BackendKind::Metal => ResidentDeviceCompletion::Metal,
        BackendKind::Cuda => {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "Temporal4 array gate accepts only CPU reference or real Metal",
            ));
        }
    };
    let expected_receipt_count = expected_expression_receipts + 2;
    if validated.project() != PROJECT
        || validated.bookmark() != BOOKMARK
        || validated.graph_revision() != graph.revision()
        || validated.layout_version() != graph.layout_version()
        || validated.execution() != request.execution
        || validated.fingerprint() != request.fingerprint()?
        || validated.receipts().len() != expected_receipt_count
        || validated.receipts().iter().any(|receipt| {
            receipt.execution != request.execution
                || receipt.input_cardinality != 1
                || receipt.output_cardinality != 1
                || receipt.completion != expected_completion
        })
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!("{label}: native array receipts are missing, forged, stale, or wrong-device"),
        ));
    }

    let intent = validated.intent();
    if intent.node_id != NodeId(FIRST_NODE_ID)
        || intent.layer != Layer::Observed
        || !intent.labels.is_empty()
        || intent.properties.len() != 1
        || intent.properties[0].property_name != property.property_name
        || intent.properties[0].value != ScalarValue::List(case.document()?)
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "{label}: device intent truncated, reordered, or mistyped the canonical LIST: {intent:#?}"
            ),
        ));
    }
    Ok(())
}

fn execute_strict_create(
    graph: &GraphStore,
    backend: &ObservedBackend,
    case: TemporalArrayCase,
) -> Result<ExecutionOutput> {
    if backend.kind() != BackendKind::Metal {
        return Err(Error::internal(
            "strict temporal-array backend did not advertise Metal before planning",
        ));
    }
    let output = QueryEngine.execute(
        &case.query(),
        &mut context(graph, Some(backend), true, true),
    )?;
    assert_native_array_boundary(graph, backend, case)?;
    assert_exact_create_output(graph, &output, case)?;
    Ok(output)
}

#[test]
fn manifest_is_exactly_the_twelve_remaining_temporal4_array_create_scenarios() -> Result<()> {
    assert_eq!(CASES.len(), 12);
    assert_eq!(
        CASES.iter().map(|case| case.report_id).collect::<Vec<_>>(),
        vec![
            3391, 3392, 3394, 3395, 3397, 3398, 3400, 3401, 3403, 3404, 3406, 3407
        ]
    );
    assert_eq!(
        CASES
            .iter()
            .map(|case| case.values.len())
            .collect::<Vec<_>>(),
        vec![1, 3, 1, 3, 1, 3, 1, 3, 1, 3, 1, 3]
    );
    let families = CASES
        .chunks_exact(2)
        .map(|pair| {
            (
                std::mem::discriminant(&pair[0].values[0]),
                std::mem::discriminant(&pair[1].values[0]),
            )
        })
        .collect::<Vec<_>>();
    assert_eq!(families.len(), 6);
    assert!(families.iter().all(|(one, three)| one == three));
    for case in CASES {
        assert!(case.query().starts_with("CREATE ({dates: ["));
        assert_eq!(case.document()?.items()?.len(), case.values.len());
    }
    Ok(())
}

#[test]
#[ignore = "external assurance gate: requires the certified full TCK report"]
fn certified_report_uniquely_resolves_every_temporal_array_selector() {
    assert_certified_report_identities(
        CASES
            .iter()
            .map(|case| (usize::from(case.report_id), FEATURE, case.name)),
    );
}

macro_rules! cpu_case_test {
    ($name:ident, $index:expr) => {
        #[test]
        fn $name() -> Result<()> {
            run_cpu_case(CASES[$index])
        }
    };
}

cpu_case_test!(cpu_tck_3392_stores_one_date_and_reads_it_back, 0);
cpu_case_test!(cpu_tck_3393_stores_three_dates_and_reads_them_back, 1);
cpu_case_test!(cpu_tck_3395_stores_one_localtime_and_reads_it_back, 2);
cpu_case_test!(cpu_tck_3396_stores_three_localtimes_and_reads_them_back, 3);
cpu_case_test!(cpu_tck_3398_stores_one_time_and_reads_it_back, 4);
cpu_case_test!(cpu_tck_3399_stores_three_times_and_reads_them_back, 5);
cpu_case_test!(cpu_tck_3401_stores_one_localdatetime_and_reads_it_back, 6);
cpu_case_test!(
    cpu_tck_3402_stores_three_localdatetimes_and_reads_them_back,
    7
);
cpu_case_test!(cpu_tck_3404_stores_one_datetime_and_reads_it_back, 8);
cpu_case_test!(cpu_tck_3405_stores_three_datetimes_and_reads_them_back, 9);
cpu_case_test!(cpu_tck_3407_stores_one_duration_and_reads_it_back, 10);
cpu_case_test!(cpu_tck_3408_stores_three_durations_and_reads_them_back, 11);

#[test]
fn cpu_completion_cannot_masquerade_as_metal_or_unlock_host_list_construction() -> Result<()> {
    for case in CASES {
        let graph = GraphStore::default();
        let backend = ObservedBackend::cpu_masquerading_as_metal(&graph)?;
        let observations = backend.observations();
        let error = QueryEngine
            .execute(
                &case.query(),
                &mut context(&graph, Some(&backend), true, true),
            )
            .err()
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::CorruptStorage,
                    format!("{}: CPU work was published as Metal", case.label()),
                )
            })?;
        assert!(
            matches!(
                error.code,
                ErrorCode::GpuAdmissionFailure | ErrorCode::CorruptStorage
            ),
            "{}: unexpected failure class: {error:?}",
            case.label()
        );
        assert_route_closed(&observations, &case.label())?;
        assert_eq!(graph.node_count(), 0, "{}", case.label());
        assert_eq!(graph.revision(), 0, "{}", case.label());

        let calls = observations.create_calls.load(Ordering::SeqCst);
        if calls == 0 {
            assert_eq!(
                error.code,
                ErrorCode::GpuAdmissionFailure,
                "{}: missing native LIST support must fail admission before fallback",
                case.label()
            );
        } else {
            assert_eq!(calls, 1, "{}", case.label());
            assert_eq!(
                error.code,
                ErrorCode::CorruptStorage,
                "{}: CPU completion receipt was not rejected as forged Metal provenance",
                case.label()
            );
            assert_native_array_boundary(&graph, &backend, case)?;
        }
    }
    Ok(())
}

#[test]
fn create_receipts_reject_replay_forgery_stale_fences_and_scalar_list_inputs() -> Result<()> {
    let graph = GraphStore::default();
    let backend = ObservedBackend::strict_cpu_create(&graph, FaultMode::None)?;
    let output = QueryEngine.execute(
        "CREATE ({created: date({year: 1984, month: 10, day: 11})})",
        &mut context(&graph, Some(&backend), true, true),
    )?;
    assert_eq!(output.result.statistics.nodes_created, 1);
    let observations = backend.observations();
    assert_route_closed(&observations, "scalar receipt integrity control")?;
    let request = observations
        .create_requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .first()
        .cloned()
        .ok_or_else(|| Error::internal("scalar receipt control did not capture its request"))?;
    let raw = observations
        .raw_create_results
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .first()
        .cloned()
        .ok_or_else(|| Error::internal("scalar receipt control did not capture its result"))?;
    let scalar_validated = raw
        .clone()
        .validate_for_publication(&request, BackendKind::Cpu)?;

    assert_eq!(
        raw.clone()
            .validate_for_publication(&request, BackendKind::Metal)
            .expect_err("CPU receipt masqueraded as Metal")
            .code,
        ErrorCode::CorruptStorage
    );

    let mut forged_execution = request.clone();
    forged_execution.execution.low ^= 1;
    if forged_execution.execution.high == 0 && forged_execution.execution.low == 0 {
        forged_execution.execution.low = 1;
    }
    assert_eq!(
        raw.clone()
            .validate_for_publication(&forged_execution, BackendKind::Cpu)
            .expect_err("stale execution receipt validated for another request")
            .code,
        ErrorCode::CorruptStorage
    );

    let mut forged_obligation = request.clone();
    forged_obligation.expression_obligations[0].id = forged_obligation.expression_obligations[0]
        .id
        .saturating_add(100);
    assert_eq!(
        raw.clone()
            .validate_for_publication(&forged_obligation, BackendKind::Cpu)
            .expect_err("forged expression obligation reused an old receipt")
            .code,
        ErrorCode::CorruptStorage
    );

    for stale in 0..3 {
        let mut request_with_stale_generation = request.clone();
        match stale {
            0 => request_with_stale_generation.expected_bookmark.index += 1,
            1 => request_with_stale_generation.expected_graph_revision += 1,
            2 => request_with_stale_generation.expected_layout_version += 1,
            _ => unreachable!(),
        }
        assert_eq!(
            raw.clone()
                .validate_for_publication(&request_with_stale_generation, BackendKind::Cpu)
                .expect_err("stale generation receipt validated")
                .code,
            ErrorCode::CorruptStorage
        );
    }

    let mut scalar_list = request.clone();
    scalar_list.operation.properties[0].value =
        ResidentCreateNodeValueInput::Scalar(ScalarValue::List(CASES[0].document()?));
    assert_eq!(
        scalar_list
            .validate()
            .expect_err("compiler-owned ScalarValue::List entered native CREATE")
            .code,
        ErrorCode::GpuAdmissionFailure
    );
    assert!(
        assert_native_array_boundary_from_parts(
            &request,
            &scalar_validated,
            CASES[0],
            BackendKind::Cpu,
        )
        .is_err(),
        "one scalar temporal receipt was accepted as a one-element LIST proof"
    );

    let replay_backend = ObservedBackend::strict_cpu_create(&graph, FaultMode::ReplayFirstCreate)?;
    let replay_observations = replay_backend.observations();
    let replay_error = QueryEngine
        .execute(
            "CREATE ({created: date({year: 1984, month: 10, day: 11})}), ({created: date({year: 1984, month: 10, day: 12})})",
            &mut context(&graph, Some(&replay_backend), true, true),
        )
        .expect_err("first CREATE result and receipts were replayed for the second intent");
    assert_eq!(replay_error.code, ErrorCode::CorruptStorage);
    assert_eq!(replay_observations.create_calls.load(Ordering::SeqCst), 2);
    assert_route_closed(&replay_observations, "replayed CREATE receipt")?;
    assert_eq!(graph.node_count(), 0);
    Ok(())
}

fn assert_native_array_boundary_from_parts(
    request: &ResidentCreateNodeRequest,
    validated: &ValidatedResidentCreateNode,
    case: TemporalArrayCase,
    completion: BackendKind,
) -> Result<()> {
    if request.operation.properties.len() != 1 {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "array proof has the wrong property count",
        ));
    }
    if matches!(
        request.operation.properties[0].value,
        ResidentCreateNodeValueInput::Scalar(_) | ResidentCreateNodeValueInput::Temporal(_)
    ) {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "scalar input/receipt cannot prove native LIST construction",
        ));
    }
    let expected_expressions = case.values.len() + 1;
    if request.expression_obligations.len() != expected_expressions
        || validated.receipts().len() != expected_expressions + 2
        || validated.intent().properties.len() != 1
        || validated.intent().properties[0].value != ScalarValue::List(case.document()?)
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "array proof is missing constructor/list receipts or exact canonical bytes",
        ));
    }
    let expected = match completion {
        BackendKind::Cpu => ResidentDeviceCompletion::CpuReference,
        BackendKind::Metal => ResidentDeviceCompletion::Metal,
        BackendKind::Cuda => {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "unsupported array proof completion backend",
            ));
        }
    };
    if validated
        .receipts()
        .iter()
        .any(|receipt| receipt.completion != expected)
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "array proof has wrong-device completion receipts",
        ));
    }
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
#[ignore = "real-Metal Temporal4 array-CREATE hardware acceptance gate"]
fn real_metal_constructs_all_temporal_elements_and_flat_lists_then_reads_them_back() -> Result<()> {
    let _metal = metal_test_guard();
    let mut failures = Vec::new();
    let mut executions = BTreeSet::new();
    let mut passed = 0_usize;

    for case in CASES {
        let mut graph = GraphStore::default();
        let create = ObservedBackend::real_metal(&graph, AllowedRoute::Create)?;
        match execute_strict_create(&graph, &create, case) {
            Ok(output) => {
                let request_execution = create
                    .observations()
                    .create_requests
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .first()
                    .map(|request| request.execution);
                if request_execution.is_none_or(|execution| !executions.insert(execution)) {
                    failures.push(format!(
                        "{}: native CREATE omitted or reused an execution identity",
                        case.label()
                    ));
                    continue;
                }
                if let Err(error) = apply_mutations(&mut graph, &output) {
                    failures.push(format!("{}: publication failed: {error}", case.label()));
                    continue;
                }
                match ObservedBackend::real_metal(&graph, AllowedRoute::Readback)
                    .and_then(|readback| assert_control_readback(&graph, &readback, case))
                {
                    Ok(_) => passed += 1,
                    Err(error) => failures.push(format!(
                        "{}: Metal control MATCH/readback failed: {error}",
                        case.label()
                    )),
                }
            }
            Err(error) => {
                let observations = create.observations();
                failures.push(format!(
                    "{}: {:?}: {}; native CREATE calls={}, external temporal calls={}, scalar/list calls={}, generic/rejected calls={}",
                    case.label(),
                    error.code,
                    error.message,
                    observations.create_calls.load(Ordering::SeqCst),
                    observations.temporal_program_calls.load(Ordering::SeqCst),
                    observations.scalar_program_calls.load(Ordering::SeqCst),
                    observations.unexpected_calls.load(Ordering::SeqCst),
                ));
            }
        }
    }

    assert!(
        failures.is_empty(),
        "real Metal passed {passed}/{} Temporal4 array CREATE+MATCH scenarios; {} failed:\n{}",
        CASES.len(),
        failures.len(),
        failures.join("\n"),
    );
    Ok(())
}
