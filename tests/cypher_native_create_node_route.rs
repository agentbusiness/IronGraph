// Test-only module. Clippy's `allow-expect-in-tests` covers `#[test]` bodies but not the helper
// functions those tests call, and a failed expectation in a fixture is the intended way for a test
// to fail. The production denial of `expect` is unaffected.
#![allow(clippy::expect_used)]

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
    Bookmark, Error, ErrorCode, Layer, NodeId, ProjectId, Result, ScalarValue,
    cypher::{
        BindCapabilities, ColumnType, ExecutionContext, ExecutionOutput, QueryEngine, ResultValue,
        StatementStats,
    },
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend,
        ResidentCreateNodeRequest, ResidentCreateNodeResult, ResidentCreateNodeValueInput,
        ResidentDeviceCompletion, ResidentGroup, ResidentGroupRequest, ResidentJoinPair,
        ResidentJoinRequest, ResidentNodePipelineRequest, ResidentNodePipelineResult,
        ResidentProjectImage, ResidentRowProgramRequest, ResidentRowProgramResult,
        ResidentSortRequest, ResidentSortResult, ResidentTemporalPipelineRequest,
        ResidentTemporalPipelineResult, ResidentTemporalValueFunction, ResidentTemporalValueInput,
        ResidentTemporalValueProgramRequest, ResidentTemporalValueProgramResult,
        ResidentVectorQuery, ResidentVectorResult, ScratchReservation, ValidatedResidentCreateNode,
    },
    graph::{GraphMutation, GraphStore, IndexCatalog, LayerMask, TemporalStore},
    types::{LabelId, PropertyId},
};
use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::gpu::MetalBackend;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());
const BOOKMARK: Bookmark = Bookmark {
    term: 19,
    index: 41,
};
const FIRST_NODE_ID: u64 = 100;
const MEMORY_LIMIT_BYTES: usize = 128 * 1024 * 1024;
const RESERVED_BYTES: usize = 16 * 1024 * 1024;
const MAX_RESULT_ROWS: usize = 32;
const CREATE_FEATURE: &str = "features/clauses/create/Create1.feature";
const TEMPORAL_FEATURE: &str = "features/expressions/temporal/Temporal4.feature";
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
const TEMPORAL_MAP_WIRE_VERSION: u8 = 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExpectedValue {
    Null,
    Boolean(bool),
    Integer(i64),
    String(&'static str),
    Date(i64),
    LocalTime(i64),
    ZonedTime {
        nanos: i64,
        offset_seconds: i32,
    },
    LocalDateTime {
        seconds: i64,
        nanos: u32,
    },
    ZonedDateTime {
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

impl ExpectedValue {
    fn scalar(self) -> ScalarValue {
        match self {
            Self::Null => ScalarValue::Null,
            Self::Boolean(value) => ScalarValue::Boolean(value),
            Self::Integer(value) => ScalarValue::Integer(value),
            Self::String(value) => ScalarValue::String(Arc::from(value)),
            Self::Date(value) => ScalarValue::Date(value),
            Self::LocalTime(value) => ScalarValue::LocalTime(value),
            Self::ZonedTime {
                nanos,
                offset_seconds,
            } => ScalarValue::ZonedTime {
                nanos,
                offset_seconds,
            },
            Self::LocalDateTime { seconds, nanos } => ScalarValue::LocalDateTime { seconds, nanos },
            Self::ZonedDateTime {
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

    const fn column_type(self) -> ColumnType {
        match self {
            Self::Null => ColumnType::Null,
            Self::Boolean(_) => ColumnType::Boolean,
            Self::Integer(_) => ColumnType::Integer,
            Self::String(_) => ColumnType::String,
            Self::Date(_)
            | Self::LocalTime(_)
            | Self::ZonedTime { .. }
            | Self::LocalDateTime { .. }
            | Self::ZonedDateTime { .. } => ColumnType::Temporal,
            Self::Duration { .. } => ColumnType::Duration,
        }
    }

    const fn temporal_function(self) -> Option<ResidentTemporalValueFunction> {
        match self {
            Self::Date(_) => Some(ResidentTemporalValueFunction::Date),
            Self::LocalTime(_) => Some(ResidentTemporalValueFunction::LocalTime),
            Self::ZonedTime { .. } => Some(ResidentTemporalValueFunction::Time),
            Self::LocalDateTime { .. } => Some(ResidentTemporalValueFunction::LocalDateTime),
            Self::ZonedDateTime { .. } => Some(ResidentTemporalValueFunction::DateTime),
            Self::Duration { .. } => Some(ResidentTemporalValueFunction::Duration),
            Self::Null | Self::Boolean(_) | Self::Integer(_) | Self::String(_) => None,
        }
    }
}

type ExpectedProperty = (&'static str, ExpectedValue);

#[derive(Clone, Copy, Debug)]
struct ExpectedNode<'a> {
    labels: &'static [&'static str],
    properties: &'a [ExpectedProperty],
}

#[derive(Clone, Copy, Debug)]
struct ExpectedColumn {
    name: &'static str,
    value: ExpectedValue,
}

#[derive(Clone, Copy, Debug)]
struct CreateCase {
    report_id: u16,
    scenario: u8,
    name: &'static str,
    query: &'static str,
    nodes: &'static [ExpectedNode<'static>],
    columns: &'static [ExpectedColumn],
}

impl CreateCase {
    fn label(self) -> String {
        format!("TCK {} {CREATE_FEATURE} {}", self.report_id, self.name)
    }
}

const CREATE_CASES: [CreateCase; 12] = [
    CreateCase {
        report_id: 52,
        scenario: 1,
        name: "[1] Create a single node",
        query: "CREATE ()",
        nodes: &[ExpectedNode {
            labels: &[],
            properties: &[],
        }],
        columns: &[],
    },
    CreateCase {
        report_id: 53,
        scenario: 2,
        name: "[2] Create two nodes",
        query: "CREATE (), ()",
        nodes: &[
            ExpectedNode {
                labels: &[],
                properties: &[],
            },
            ExpectedNode {
                labels: &[],
                properties: &[],
            },
        ],
        columns: &[],
    },
    CreateCase {
        report_id: 54,
        scenario: 3,
        name: "[3] Create a single node with a label",
        query: "CREATE (:Label)",
        nodes: &[ExpectedNode {
            labels: &["Label"],
            properties: &[],
        }],
        columns: &[],
    },
    CreateCase {
        report_id: 55,
        scenario: 4,
        name: "[4] Create two nodes with same label",
        query: "CREATE (:Label), (:Label)",
        nodes: &[
            ExpectedNode {
                labels: &["Label"],
                properties: &[],
            },
            ExpectedNode {
                labels: &["Label"],
                properties: &[],
            },
        ],
        columns: &[],
    },
    CreateCase {
        report_id: 56,
        scenario: 5,
        name: "[5] Create a single node with multiple labels",
        query: "CREATE (:A:B:C:D)",
        nodes: &[ExpectedNode {
            labels: &["A", "B", "C", "D"],
            properties: &[],
        }],
        columns: &[],
    },
    CreateCase {
        report_id: 57,
        scenario: 6,
        name: "[6] Create three nodes with multiple labels",
        query: "CREATE (:B:A:D), (:B:C), (:D:E:B)",
        nodes: &[
            ExpectedNode {
                labels: &["B", "A", "D"],
                properties: &[],
            },
            ExpectedNode {
                labels: &["B", "C"],
                properties: &[],
            },
            ExpectedNode {
                labels: &["D", "E", "B"],
                properties: &[],
            },
        ],
        columns: &[],
    },
    CreateCase {
        report_id: 58,
        scenario: 7,
        name: "[7] Create a single node with a property",
        query: "CREATE ({created: true})",
        nodes: &[ExpectedNode {
            labels: &[],
            properties: &[("created", ExpectedValue::Boolean(true))],
        }],
        columns: &[],
    },
    CreateCase {
        report_id: 59,
        scenario: 8,
        name: "[8] Create a single node with a property and return it",
        query: "CREATE (n {name: 'foo'}) RETURN n.name AS p",
        nodes: &[ExpectedNode {
            labels: &[],
            properties: &[("name", ExpectedValue::String("foo"))],
        }],
        columns: &[ExpectedColumn {
            name: "p",
            value: ExpectedValue::String("foo"),
        }],
    },
    CreateCase {
        report_id: 60,
        scenario: 9,
        name: "[9] Create a single node with two properties",
        query: "CREATE (n {id: 12, name: 'foo'})",
        nodes: &[ExpectedNode {
            labels: &[],
            properties: &[
                ("id", ExpectedValue::Integer(12)),
                ("name", ExpectedValue::String("foo")),
            ],
        }],
        columns: &[],
    },
    CreateCase {
        report_id: 61,
        scenario: 10,
        name: "[10] Create a single node with two properties and return them",
        query: "CREATE (n {id: 12, name: 'foo'}) RETURN n.id AS id, n.name AS p",
        nodes: &[ExpectedNode {
            labels: &[],
            properties: &[
                ("id", ExpectedValue::Integer(12)),
                ("name", ExpectedValue::String("foo")),
            ],
        }],
        columns: &[
            ExpectedColumn {
                name: "id",
                value: ExpectedValue::Integer(12),
            },
            ExpectedColumn {
                name: "p",
                value: ExpectedValue::String("foo"),
            },
        ],
    },
    CreateCase {
        report_id: 62,
        scenario: 11,
        name: "[11] Create a single node with null properties should not return those properties",
        query: "CREATE (n {id: 12, name: null}) RETURN n.id AS id, n.name AS p",
        nodes: &[ExpectedNode {
            labels: &[],
            properties: &[
                ("id", ExpectedValue::Integer(12)),
                ("name", ExpectedValue::Null),
            ],
        }],
        columns: &[
            ExpectedColumn {
                name: "id",
                value: ExpectedValue::Integer(12),
            },
            ExpectedColumn {
                name: "p",
                value: ExpectedValue::Null,
            },
        ],
    },
    CreateCase {
        report_id: 63,
        scenario: 12,
        name: "[12] CREATE does not lose precision on large integers",
        query: "CREATE (p:TheLabel {id: 4611686018427387905}) RETURN p.id",
        nodes: &[ExpectedNode {
            labels: &["TheLabel"],
            properties: &[("id", ExpectedValue::Integer(4_611_686_018_427_387_905))],
        }],
        columns: &[ExpectedColumn {
            name: "p.id",
            value: ExpectedValue::Integer(4_611_686_018_427_387_905),
        }],
    },
];

#[derive(Clone, Copy, Debug)]
struct TemporalCase {
    report_id: u16,
    scenario: u8,
    name: &'static str,
    expression: &'static str,
    value: ExpectedValue,
}

impl TemporalCase {
    fn query(self, returning: bool) -> String {
        if returning {
            format!(
                "CREATE (n {{created: {}}}) RETURN n.created AS created",
                self.expression
            )
        } else {
            format!("CREATE ({{created: {}}})", self.expression)
        }
    }

    fn label(self, returning: bool) -> String {
        let suffix = if returning {
            "supplemental statement-local RETURN"
        } else {
            "official CREATE operation"
        };
        format!(
            "TCK {} {TEMPORAL_FEATURE} {} ({suffix})",
            self.report_id, self.name
        )
    }
}

const TEMPORAL_CASES: [TemporalCase; 6] = [
    TemporalCase {
        report_id: 3390,
        scenario: 1,
        name: "[1] Should store date [3391]",
        expression: "date({year: 1984, month: 10, day: 11})",
        value: ExpectedValue::Date(5_397),
    },
    TemporalCase {
        report_id: 3393,
        scenario: 3,
        name: "[3] Should store local time [3394]",
        expression: "localtime({hour: 12})",
        value: ExpectedValue::LocalTime(43_200_000_000_000),
    },
    TemporalCase {
        report_id: 3396,
        scenario: 5,
        name: "[5] Should store time [3397]",
        expression: "time({hour: 12})",
        value: ExpectedValue::ZonedTime {
            nanos: 43_200_000_000_000,
            offset_seconds: 0,
        },
    },
    TemporalCase {
        report_id: 3399,
        scenario: 7,
        name: "[7] Should store local date time [3400]",
        expression: "localdatetime({year: 1912})",
        value: ExpectedValue::LocalDateTime {
            seconds: -1_830_384_000,
            nanos: 0,
        },
    },
    TemporalCase {
        report_id: 3402,
        scenario: 9,
        name: "[9] Should store date time [3403]",
        expression: "datetime({year: 1912})",
        value: ExpectedValue::ZonedDateTime {
            seconds: -1_830_384_000,
            nanos: 0,
            timezone: "UTC",
        },
    },
    TemporalCase {
        report_id: 3405,
        scenario: 11,
        name: "[11] Should store duration [3406]",
        expression: "duration({seconds: 12})",
        value: ExpectedValue::Duration {
            months: 0,
            days: 0,
            seconds: 12,
            nanos: 0,
        },
    },
];

struct Fixture {
    graph: GraphStore,
}

impl Fixture {
    fn empty() -> Self {
        Self {
            graph: GraphStore::default(),
        }
    }

    fn image(&self) -> Result<ResidentProjectImage> {
        ResidentProjectImage::build(
            PROJECT,
            BOOKMARK,
            &self.graph,
            &TemporalStore::default(),
            &IndexCatalog::default(),
        )
    }

    fn cpu(&self) -> Result<CpuBackend> {
        let mut backend = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
        backend.admit_project(self.image()?)?;
        Ok(backend)
    }

    fn strict_cpu_backend(&self) -> Result<ObservedCreateBackend> {
        ObservedCreateBackend::strict_cpu_reference(self.cpu()?)
    }

    fn wrong_receipt_backend(&self) -> Result<ObservedCreateBackend> {
        ObservedCreateBackend::wrong_receipt_provenance(self.cpu()?)
    }

    fn replay_backend(&self) -> Result<ObservedCreateBackend> {
        ObservedCreateBackend::replay_first_intent(self.cpu()?)
    }

    #[cfg(all(feature = "accelerator", target_os = "macos"))]
    fn real_metal_backend(&self) -> Result<ObservedCreateBackend> {
        let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
        metal.admit_project(self.image()?)?;
        ObservedCreateBackend::real_metal(metal)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum FaultMode {
    None,
    WrongReceiptProvenance,
    ReplayFirstIntent,
}

#[derive(Clone, Debug)]
struct NativeCreateReceipt {
    completion: BackendKind,
    request: ResidentCreateNodeRequest,
    validated: ValidatedResidentCreateNode,
}

#[derive(Default)]
struct CreateObservations {
    pins: AtomicUsize,
    resident_execution_active: AtomicUsize,
    create_calls: AtomicUsize,
    temporal_program_calls: AtomicUsize,
    legacy_route_calls: AtomicUsize,
    unexpected_query_calls: AtomicUsize,
    generation_mutation_attempts: AtomicUsize,
    receipts: Mutex<Vec<NativeCreateReceipt>>,
    first_raw_result: Mutex<Option<ResidentCreateNodeResult>>,
    resident_execution: Mutex<Option<Box<dyn ExecutionBackend>>>,
}

/// Fail-closed observer around the one legal native node-CREATE boundary.
///
/// The CPU reference advertises Metal until planning has committed to native execution. Its
/// immutable pinned generation then reports honest CPU provenance, allowing the same publication
/// validator used by real Metal to distinguish the two receipt classes. Every graph, row,
/// temporal, and legacy route is closed: temporal constructors are legal only as raw nested SSA
/// inside execute_create_node.
struct ObservedCreateBackend {
    inner: Box<dyn ExecutionBackend>,
    advertised_kind: BackendKind,
    pinned_kind: BackendKind,
    actual_kind: BackendKind,
    pinned: bool,
    expected_bookmark: Bookmark,
    expected_graph_revision: u64,
    fault: FaultMode,
    observations: Arc<CreateObservations>,
}

impl ObservedCreateBackend {
    fn strict_cpu_reference(inner: CpuBackend) -> Result<Self> {
        Self::new(
            Box::new(inner),
            BackendKind::Metal,
            BackendKind::Cpu,
            BackendKind::Cpu,
            FaultMode::None,
        )
    }

    fn wrong_receipt_provenance(inner: CpuBackend) -> Result<Self> {
        Self::new(
            Box::new(inner),
            BackendKind::Metal,
            BackendKind::Metal,
            BackendKind::Cpu,
            FaultMode::WrongReceiptProvenance,
        )
    }

    fn replay_first_intent(inner: CpuBackend) -> Result<Self> {
        Self::new(
            Box::new(inner),
            BackendKind::Metal,
            BackendKind::Cpu,
            BackendKind::Cpu,
            FaultMode::ReplayFirstIntent,
        )
    }

    #[cfg(all(feature = "accelerator", target_os = "macos"))]
    fn real_metal(inner: MetalBackend) -> Result<Self> {
        if inner.kind() != BackendKind::Metal {
            return Err(Error::internal(
                "native node-CREATE test did not construct a real Metal backend",
            ));
        }
        Self::new(
            Box::new(inner),
            BackendKind::Metal,
            BackendKind::Metal,
            BackendKind::Metal,
            FaultMode::None,
        )
    }

    fn new(
        inner: Box<dyn ExecutionBackend>,
        advertised_kind: BackendKind,
        pinned_kind: BackendKind,
        actual_kind: BackendKind,
        fault: FaultMode,
    ) -> Result<Self> {
        let expected_bookmark = inner.resident_bookmark(PROJECT).ok_or_else(|| {
            Error::internal("native node-CREATE backend has no admitted project bookmark")
        })?;
        let expected_graph_revision = inner.resident_graph_revision(PROJECT).ok_or_else(|| {
            Error::internal("native node-CREATE backend has no admitted graph revision")
        })?;
        Ok(Self {
            inner,
            advertised_kind,
            pinned_kind,
            actual_kind,
            pinned: false,
            expected_bookmark,
            expected_graph_revision,
            fault,
            observations: Arc::new(CreateObservations::default()),
        })
    }

    fn observations(&self) -> Arc<CreateObservations> {
        Arc::clone(&self.observations)
    }

    fn reject_query_route<T>(&self, route: &'static str) -> Result<T> {
        self.observations
            .unexpected_query_calls
            .fetch_add(1, Ordering::SeqCst);
        Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!("strict native node-CREATE test rejected {route} execution"),
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
            format!("pinned native node-CREATE generation rejected {route} mutation"),
        ))
    }

    fn refresh_fence(&mut self) -> Result<()> {
        self.expected_bookmark = self.inner.resident_bookmark(PROJECT).ok_or_else(|| {
            Error::internal("replacement native node-CREATE project has no bookmark")
        })?;
        self.expected_graph_revision =
            self.inner.resident_graph_revision(PROJECT).ok_or_else(|| {
                Error::internal("replacement native node-CREATE project has no graph revision")
            })?;
        Ok(())
    }

    /// Mirrors the engine's resident-execution boundary while making the selected immutable
    /// generation observable to this fail-closed wrapper. Current CREATE integration enters the
    /// callback through the admitted backend; older resident routes may call pin_project first.
    /// Both shapes therefore reach exactly the same pinned inner backend and provenance checks.
    fn with_resident_execution<T>(
        &self,
        execute: impl FnOnce(&dyn ExecutionBackend) -> Result<T>,
    ) -> Result<T> {
        if self.pinned {
            return execute(self.inner.as_ref());
        }
        let mut generation = self
            .observations
            .resident_execution
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if generation.is_none() {
            let pinned = self.inner.pin_project(PROJECT)?;
            if pinned.kind() != self.actual_kind
                || pinned.resident_bookmark(PROJECT) != Some(self.expected_bookmark)
                || pinned.resident_graph_revision(PROJECT) != Some(self.expected_graph_revision)
            {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "native node-CREATE resident execution pinned the wrong generation",
                ));
            }
            *generation = Some(pinned);
            self.observations.pins.fetch_add(1, Ordering::SeqCst);
            self.observations
                .resident_execution_active
                .store(1, Ordering::SeqCst);
        }
        let pinned = generation
            .as_deref()
            .ok_or_else(|| Error::internal("native CREATE pinned generation disappeared"))?;
        execute(pinned)
    }

    fn execute_create_node_on_generation(
        &self,
        generation: &dyn ExecutionBackend,
        request: &ResidentCreateNodeRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentCreateNodeResult> {
        if generation.kind() != self.actual_kind
            || generation.resident_bookmark(PROJECT) != Some(self.expected_bookmark)
            || generation.resident_graph_revision(PROJECT) != Some(self.expected_graph_revision)
            || request.project != PROJECT
            || request.expected_bookmark != self.expected_bookmark
            || request.expected_graph_revision != self.expected_graph_revision
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "native node-CREATE escaped its immutable generation or request fence",
            ));
        }
        request.validate()?;
        let call = self
            .observations
            .create_calls
            .fetch_add(1, Ordering::SeqCst);

        if self.fault == FaultMode::ReplayFirstIntent && call > 0 {
            return self
                .observations
                .first_raw_result
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .clone()
                .ok_or_else(|| Error::internal("first native CREATE result disappeared"));
        }

        let result = generation.execute_create_node(request, cancellation)?;
        if generation.kind() != self.actual_kind
            || generation.resident_bookmark(PROJECT) != Some(self.expected_bookmark)
            || generation.resident_graph_revision(PROJECT) != Some(self.expected_graph_revision)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "native node-CREATE dispatch mutated its pinned generation",
            ));
        }
        if call == 0 {
            *self
                .observations
                .first_raw_result
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(result.clone());
        }
        let validated = result
            .clone()
            .validate_for_publication(request, self.actual_kind)?;
        self.observations
            .receipts
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(NativeCreateReceipt {
                completion: self.actual_kind,
                request: request.clone(),
                validated,
            });
        Ok(result)
    }
}

impl ExecutionBackend for ObservedCreateBackend {
    fn kind(&self) -> BackendKind {
        if self.pinned
            || self
                .observations
                .resident_execution_active
                .load(Ordering::SeqCst)
                != 0
        {
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
                "pinned native node-CREATE generation has wrong provenance or fence",
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
            fault: self.fault,
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

    fn supports_native_create_node(&self) -> bool {
        true
    }

    fn execute_create_node(
        &self,
        request: &ResidentCreateNodeRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentCreateNodeResult> {
        self.with_resident_execution(|generation| {
            self.execute_create_node_on_generation(generation, request, cancellation)
        })
    }

    fn execute_row_program(
        &self,
        _request: &ResidentRowProgramRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentRowProgramResult> {
        self.reject_legacy_route("execute_row_program")
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
        self.reject_legacy_route("execute_temporal_value_program_outside_create")
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
        bookmark: BOOKMARK,
        mutation_revision: fixture.graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id: FIRST_NODE_ID,
        next_edge_id: 1,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            write: true,
            require_native_execution: backend.is_some(),
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

fn expected_name_order(
    nodes: &[ExpectedNode<'_>],
    names: impl Fn(&ExpectedNode<'_>) -> Vec<&'static str>,
) -> Vec<&'static str> {
    let mut seen = BTreeSet::new();
    let mut ordered = Vec::new();
    for node in nodes {
        for name in names(node) {
            if seen.insert(name) {
                ordered.push(name);
            }
        }
    }
    ordered
}

fn expected_labels(nodes: &[ExpectedNode<'_>]) -> Vec<&'static str> {
    expected_name_order(nodes, |node| {
        // The native compiler canonicalizes each node's label set before dispatch. Canonical
        // schema IDs are then allocated on first use while publishing the validated node intents.
        let mut labels = node.labels.to_vec();
        labels.sort_unstable();
        labels
    })
}

fn expected_properties(nodes: &[ExpectedNode<'_>]) -> Vec<&'static str> {
    expected_name_order(nodes, |node| {
        node.properties.iter().map(|(name, _)| *name).collect()
    })
}

fn expected_node_property_map(
    node: &ExpectedNode<'_>,
    omit_null: bool,
) -> BTreeMap<String, ScalarValue> {
    node.properties
        .iter()
        .filter(|(_, value)| !omit_null || *value != ExpectedValue::Null)
        .map(|(name, value)| ((*name).to_owned(), value.scalar()))
        .collect()
}

fn assert_result(output: &ExecutionOutput, columns: &[ExpectedColumn], label: &str) -> Result<()> {
    let expected_schema = columns
        .iter()
        .map(|column| (column.name.to_owned(), column.value.column_type()))
        .collect::<Vec<_>>();
    if output.result.schema != expected_schema {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "{label}: wrong result schema; expected {expected_schema:?}, got {:?}",
                output.result.schema
            ),
        ));
    }
    if columns.is_empty() {
        if !output.result.batches.is_empty() {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                format!("{label}: CREATE without RETURN emitted rows"),
            ));
        }
        return Ok(());
    }
    if output.result.batches.len() != 1
        || output.result.batches[0].row_count != 1
        || output.result.batches[0].columns.len() != columns.len()
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "{label}: statement-local RETURN has wrong batch shape: {:?}",
                output.result.batches
            ),
        ));
    }
    for (actual, expected) in output.result.batches[0].columns.iter().zip(columns) {
        let expected_values = vec![ResultValue::Scalar(expected.value.scalar())];
        if actual.name != expected.name
            || actual.value_type != expected.value.column_type()
            || actual.values != expected_values
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                format!(
                    "{label}: RETURN did not read the exact statement-local created property; expected {expected:?}, got {actual:?}"
                ),
            ));
        }
    }
    Ok(())
}

fn assert_canonical_publication(
    fixture: &Fixture,
    output: &ExecutionOutput,
    nodes: &[ExpectedNode<'_>],
    columns: &[ExpectedColumn],
    label: &str,
) -> Result<()> {
    assert_result(output, columns, label)?;
    let expected_stats = StatementStats {
        nodes_created: nodes.len() as u64,
        labels_added: nodes.iter().map(|node| node.labels.len() as u64).sum(),
        ..StatementStats::default()
    };
    if output.result.statistics != expected_stats
        || output.result.bookmark != BOOKMARK
        || output.result.truncated
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "{label}: exact CREATE statistics/bookmark changed; expected {expected_stats:?} at {BOOKMARK}, got {:?}",
                output.result
            ),
        ));
    }
    if !output.temporal_mutations.is_empty()
        || output.administrative.is_some()
        || !output.vector_searches.is_empty()
        || output.runtime_replans != 0
        || !output.dependencies.entities.is_empty()
        || !output.dependencies.write_targets.is_empty()
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!("{label}: standalone node-CREATE emitted unrelated execution state"),
        ));
    }

    let expected_label_names = expected_labels(nodes);
    let expected_property_names = expected_properties(nodes);
    let mut label_names = BTreeMap::<LabelId, String>::new();
    let mut property_names = BTreeMap::<PropertyId, String>::new();
    let mut declared_labels = Vec::new();
    let mut declared_properties = Vec::new();
    let mut inserted = Vec::new();
    for mutation in &output.graph_mutations {
        match mutation {
            GraphMutation::DeclareLabel { name, id } => {
                if label_names.insert(*id, name.clone()).is_some() {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        format!("{label}: duplicate label declaration for {id}"),
                    ));
                }
                declared_labels.push((name.clone(), *id));
            }
            GraphMutation::DeclareProperty { name, id } => {
                if property_names.insert(*id, name.clone()).is_some() {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        format!("{label}: duplicate property declaration for {id}"),
                    ));
                }
                declared_properties.push((name.clone(), *id));
            }
            GraphMutation::InsertNode(node) => {
                if node.labels.iter().any(|id| !label_names.contains_key(id))
                    || node
                        .properties
                        .iter()
                        .any(|(id, _)| !property_names.contains_key(id))
                {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        format!("{label}: InsertNode was published before its schema declarations"),
                    ));
                }
                inserted.push(node);
            }
            other => {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    format!("{label}: standalone node-CREATE emitted {other:?}"),
                ));
            }
        }
    }

    let expected_label_declarations = expected_label_names
        .iter()
        .enumerate()
        .map(|(index, name)| ((*name).to_owned(), LabelId(index as u64)))
        .collect::<Vec<_>>();
    let expected_property_declarations = expected_property_names
        .iter()
        .enumerate()
        .map(|(index, name)| ((*name).to_owned(), PropertyId(index as u64)))
        .collect::<Vec<_>>();
    if declared_labels != expected_label_declarations
        || declared_properties != expected_property_declarations
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "{label}: schema IDs are not exact and stable; labels={declared_labels:?}, properties={declared_properties:?}"
            ),
        ));
    }
    if inserted.len() != nodes.len() {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "{label}: expected {} canonical InsertNode mutations, got {}",
                nodes.len(),
                inserted.len()
            ),
        ));
    }
    for (index, (actual, expected)) in inserted.iter().zip(nodes).enumerate() {
        let expected_id = NodeId(FIRST_NODE_ID + index as u64);
        if actual.id != expected_id
            || actual.layer != Layer::Observed
            || actual.revision != fixture.graph.revision().saturating_add(1)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                format!(
                    "{label}: canonical node identity/layer/revision changed at index {index}: {actual:?}"
                ),
            ));
        }
        let mut actual_labels = actual
            .labels
            .iter()
            .map(|id| {
                label_names.get(id).cloned().ok_or_else(|| {
                    Error::new(
                        ErrorCode::CorruptStorage,
                        "created node label is undeclared",
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let mut expected_labels = expected
            .labels
            .iter()
            .map(|name| (*name).to_owned())
            .collect::<Vec<_>>();
        actual_labels.sort();
        expected_labels.sort();
        if actual_labels != expected_labels {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                format!(
                    "{label}: node {expected_id} labels differ; expected {expected_labels:?}, got {actual_labels:?}"
                ),
            ));
        }
        if actual
            .properties
            .iter()
            .any(|(_, value)| matches!(value, ScalarValue::Null))
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                format!("{label}: NULL was stored in the canonical InsertNode property payload"),
            ));
        }
        let actual_properties = actual
            .properties
            .iter()
            .map(|(id, value)| {
                let name = property_names.get(id).cloned().ok_or_else(|| {
                    Error::new(
                        ErrorCode::CorruptStorage,
                        "created node property is undeclared",
                    )
                })?;
                Ok((name, value.clone()))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        let expected_properties = expected_node_property_map(expected, true);
        if actual_properties != expected_properties {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                format!(
                    "{label}: node {expected_id} properties differ; expected {expected_properties:?}, got {actual_properties:?}"
                ),
            ));
        }
    }
    Ok(())
}

fn semantic_request_labels(request: &ResidentCreateNodeRequest) -> Result<Vec<String>> {
    request
        .operation
        .labels
        .iter()
        .map(|index| {
            request
                .label_names
                .get(*index as usize)
                .cloned()
                .ok_or_else(|| Error::internal("validated CREATE label index disappeared"))
        })
        .collect()
}

fn assert_raw_temporal_property(
    value: &ResidentCreateNodeValueInput,
    expected: ExpectedValue,
    label: &str,
) -> Result<()> {
    let expected_function = expected.temporal_function().ok_or_else(|| {
        Error::internal("non-temporal expectation entered temporal CREATE assertion")
    })?;
    let ResidentCreateNodeValueInput::Temporal(syntax) = value else {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "{label}: temporal CREATE property was host-evaluated or precomputed: {value:?}"
            ),
        ));
    };
    if syntax.invocations.len() != 1
        || syntax.output_register != 0
        || syntax.invocations[0].function != expected_function
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "{label}: temporal CREATE did not retain one raw constructor SSA program: {syntax:?}"
            ),
        ));
    }
    let ResidentTemporalValueInput::Map(packet) = &syntax.invocations[0].input else {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "{label}: temporal CREATE constructor was not a raw literal-map packet: {:?}",
                syntax.invocations[0].input
            ),
        ));
    };
    if packet.len() < 3 || packet[0] != TEMPORAL_MAP_WIRE_VERSION || packet[1] == 0 {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!("{label}: malformed or precomputed temporal map packet: {packet:?}"),
        ));
    }
    if (expected_function == ResidentTemporalValueFunction::DateTime
        && syntax.maximum_timezone_bytes < 3)
        || (expected_function != ResidentTemporalValueFunction::DateTime
            && syntax.maximum_timezone_bytes != 0)
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "{label}: temporal CREATE has the wrong variable-width result capacity: {}",
                syntax.maximum_timezone_bytes
            ),
        ));
    }
    Ok(())
}

fn assert_native_receipts(
    fixture: &Fixture,
    backend: &ObservedCreateBackend,
    receipts: &[NativeCreateReceipt],
    nodes: &[ExpectedNode<'_>],
    label: &str,
) -> Result<()> {
    if receipts.len() != nodes.len() {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "{label}: expected one receipted CreateNode intent per node ({}), got {}",
                nodes.len(),
                receipts.len()
            ),
        ));
    }
    let mut execution_ids = BTreeSet::new();
    for (index, (receipt, expected_node)) in receipts.iter().zip(nodes).enumerate() {
        let request = &receipt.request;
        request.validate()?;
        if request.project != PROJECT
            || request.expected_bookmark != BOOKMARK
            || request.expected_graph_revision != fixture.graph.revision()
            || request.expected_layout_version != fixture.graph.layout_version()
            || request.operation.node_id != NodeId(FIRST_NODE_ID + index as u64)
            || request.operation.layer != Layer::Observed
            || receipt.completion != backend.actual_kind
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                format!(
                    "{label}: native CreateNode request has the wrong stable fence: {request:?}"
                ),
            ));
        }
        if !execution_ids.insert(request.execution) {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                format!("{label}: two CreateNode intents reused one execution identity"),
            ));
        }
        let mut actual_labels = semantic_request_labels(request)?;
        let mut expected_labels = expected_node
            .labels
            .iter()
            .map(|name| (*name).to_owned())
            .collect::<Vec<_>>();
        actual_labels.sort();
        expected_labels.sort();
        if actual_labels != expected_labels || request.label_names.len() != expected_labels.len() {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                format!(
                    "{label}: request {index} labels differ; expected {expected_labels:?}, got {actual_labels:?}"
                ),
            ));
        }
        if request.operation.properties.len() != expected_node.properties.len()
            || request.property_names.len() != expected_node.properties.len()
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                format!("{label}: request {index} has missing, duplicate, or extra properties"),
            ));
        }

        let expected_by_name = expected_node
            .properties
            .iter()
            .map(|(name, value)| ((*name).to_owned(), *value))
            .collect::<BTreeMap<_, _>>();
        let mut request_names = BTreeSet::new();
        for property in &request.operation.properties {
            let property_name = request
                .property_names
                .get(property.property_name as usize)
                .ok_or_else(|| Error::internal("validated CREATE property index disappeared"))?;
            if !request_names.insert(property_name.clone()) {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    format!("{label}: duplicate request property {property_name}"),
                ));
            }
            let expected = expected_by_name
                .get(property_name)
                .copied()
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::CorruptStorage,
                        format!("{label}: unexpected request property {property_name}"),
                    )
                })?;
            if expected.temporal_function().is_some() {
                assert_raw_temporal_property(&property.value, expected, label)?;
            } else if property.value != ResidentCreateNodeValueInput::Scalar(expected.scalar()) {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    format!(
                        "{label}: scalar request property {property_name} was changed or precomputed: {:?}",
                        property.value
                    ),
                ));
            }
        }
        let expected_temporal_count = expected_node
            .properties
            .iter()
            .filter(|(_, value)| value.temporal_function().is_some())
            .count();
        if request.expression_obligations.len() != expected_temporal_count {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                format!(
                    "{label}: temporal expression receipts do not exactly cover native constructors"
                ),
            ));
        }

        let validated = &receipt.validated;
        if validated.project() != request.project
            || validated.bookmark() != request.expected_bookmark
            || validated.graph_revision() != request.expected_graph_revision
            || validated.layout_version() != request.expected_layout_version
            || validated.execution() != request.execution
            || validated.fingerprint() != request.fingerprint()?
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                format!("{label}: validated CreateNode receipt changed its immutable request"),
            ));
        }
        let expected_completion = match backend.actual_kind {
            BackendKind::Cpu => ResidentDeviceCompletion::CpuReference,
            BackendKind::Metal => ResidentDeviceCompletion::Metal,
            BackendKind::Cuda => {
                return Err(Error::new(
                    ErrorCode::GpuAdmissionFailure,
                    "strict CREATE test accepts only CPU reference or real Metal",
                ));
            }
        };
        let expected_obligations = std::iter::once(request.selection_obligation)
            .chain(request.expression_obligations.iter().copied())
            .chain(std::iter::once(request.rhs_obligation))
            .collect::<Vec<_>>();
        if validated.receipts().len() != expected_obligations.len()
            || validated
                .receipts()
                .iter()
                .zip(expected_obligations)
                .any(|(actual, obligation)| {
                    actual.execution != request.execution
                        || actual.obligation != obligation
                        || actual.input_cardinality != 1
                        || actual.output_cardinality != 1
                        || actual.completion != expected_completion
                })
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                format!("{label}: CreateNode completion receipts are incomplete or fabricated"),
            ));
        }
        let intent = validated.intent();
        if intent.node_id != request.operation.node_id
            || intent.layer != request.operation.layer
            || intent.labels != request.operation.labels
            || intent.properties.len() != expected_node.properties.len()
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                format!("{label}: validated CreateNode intent has the wrong exact shape"),
            ));
        }
        let actual_intent_properties = intent
            .properties
            .iter()
            .map(|property| {
                let name = request
                    .property_names
                    .get(property.property_name as usize)
                    .cloned()
                    .ok_or_else(|| Error::internal("intent property name disappeared"))?;
                Ok((name, property.value.clone()))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
        let expected_intent_properties = expected_node_property_map(expected_node, false);
        if actual_intent_properties != expected_intent_properties {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                format!(
                    "{label}: backend intent properties differ; expected {expected_intent_properties:?}, got {actual_intent_properties:?}"
                ),
            ));
        }
    }
    Ok(())
}

fn assert_route_closed(observations: &CreateObservations, label: &str) -> Result<()> {
    if observations.legacy_route_calls.load(Ordering::SeqCst) != 0
        || observations.unexpected_query_calls.load(Ordering::SeqCst) != 0
        || observations.temporal_program_calls.load(Ordering::SeqCst) != 0
        || observations
            .generation_mutation_attempts
            .load(Ordering::SeqCst)
            != 0
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "{label}: query entered CPU fallback, generic/legacy execution, host temporal evaluation, or mutated its pinned generation"
            ),
        ));
    }
    Ok(())
}

fn run_success_case(
    fixture: &Fixture,
    backend: &ObservedCreateBackend,
    query: &str,
    nodes: &[ExpectedNode<'_>],
    columns: &[ExpectedColumn],
    label: &str,
) -> Result<ExecutionOutput> {
    if backend.kind() != BackendKind::Metal {
        return Err(Error::internal(
            "strict CREATE backend must advertise Metal before native planning",
        ));
    }
    let output = QueryEngine
        .execute(query, &mut context(fixture, Some(backend)))
        .map_err(|error| {
            Error::new(
                error.code,
                format!("{label}: native CREATE execution failed: {error}"),
            )
        })?;
    let observations = backend.observations();
    if observations.pins.load(Ordering::SeqCst) != 1
        || observations.create_calls.load(Ordering::SeqCst) != nodes.len()
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "{label}: expected one immutable pin and {} CreateNode calls; pins={}, calls={}",
                nodes.len(),
                observations.pins.load(Ordering::SeqCst),
                observations.create_calls.load(Ordering::SeqCst)
            ),
        ));
    }
    assert_route_closed(&observations, label)?;
    let receipts = observations
        .receipts
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone();
    assert_native_receipts(fixture, backend, &receipts, nodes, label)?;
    assert_canonical_publication(fixture, &output, nodes, columns, label)?;
    if fixture.graph.node_count() != 0 || fixture.graph.revision() != 0 {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!("{label}: query mutated canonical input before its caller published output"),
        ));
    }
    Ok(output)
}

fn mutation_debug(output: &ExecutionOutput) -> Vec<String> {
    output
        .graph_mutations
        .iter()
        .map(|mutation| format!("{mutation:?}"))
        .collect()
}

fn semantic_request_debug(observations: &CreateObservations) -> Vec<String> {
    observations
        .receipts
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .iter()
        .map(|receipt| {
            format!(
                "names={:?}/{:?}; operation={:?}",
                receipt.request.label_names,
                receipt.request.property_names,
                receipt.request.operation,
            )
        })
        .collect()
}

#[test]
fn strict_cpu_covers_exact_create1_finish_only_scenarios() -> Result<()> {
    assert_eq!(
        CREATE_CASES
            .iter()
            .map(|case| case.report_id)
            .collect::<Vec<_>>(),
        (52_u16..=63).collect::<Vec<_>>()
    );
    let finish_only = CREATE_CASES
        .into_iter()
        .filter(|case| case.columns.is_empty())
        .collect::<Vec<_>>();
    assert_eq!(
        finish_only
            .iter()
            .map(|case| case.report_id)
            .collect::<Vec<_>>(),
        vec![52, 53, 54, 55, 56, 57, 58, 60]
    );
    for case in finish_only {
        let fixture = Fixture::empty();
        let backend = fixture.strict_cpu_backend()?;
        run_success_case(
            &fixture,
            &backend,
            case.query,
            case.nodes,
            case.columns,
            &case.label(),
        )?;
    }
    Ok(())
}

#[test]
#[ignore = "external assurance gate: requires the certified full TCK report"]
fn certified_report_uniquely_resolves_every_create_selector() {
    for case in CREATE_CASES {
        assert!(case.name.starts_with(&format!("[{}] ", case.scenario)));
    }
    for case in TEMPORAL_CASES {
        assert!(case.name.starts_with(&format!("[{}] ", case.scenario)));
    }
    assert_certified_report_identities(
        CREATE_CASES
            .iter()
            .map(|case| (usize::from(case.report_id), CREATE_FEATURE, case.name))
            .chain(
                TEMPORAL_CASES
                    .iter()
                    .map(|case| (usize::from(case.report_id), TEMPORAL_FEATURE, case.name)),
            ),
    );
}

#[test]
fn strict_cpu_requires_statement_local_return_for_exact_create1_cases() -> Result<()> {
    let return_cases = CREATE_CASES
        .into_iter()
        .filter(|case| !case.columns.is_empty())
        .collect::<Vec<_>>();
    assert_eq!(
        return_cases
            .iter()
            .map(|case| case.report_id)
            .collect::<Vec<_>>(),
        vec![59, 61, 62, 63]
    );
    for case in return_cases {
        let fixture = Fixture::empty();
        let backend = fixture.strict_cpu_backend()?;
        run_success_case(
            &fixture,
            &backend,
            case.query,
            case.nodes,
            case.columns,
            &case.label(),
        )?;
    }
    Ok(())
}

#[test]
fn strict_cpu_constructs_exact_temporal4_scalar_properties_inside_create() -> Result<()> {
    assert_eq!(
        TEMPORAL_CASES
            .iter()
            .map(|case| case.report_id)
            .collect::<Vec<_>>(),
        vec![3390, 3393, 3396, 3399, 3402, 3405]
    );
    for case in TEMPORAL_CASES {
        let properties = [("created", case.value)];
        let expected_node = [ExpectedNode {
            labels: &[],
            properties: &properties,
        }];
        let fixture = Fixture::empty();
        let backend = fixture.strict_cpu_backend()?;
        run_success_case(
            &fixture,
            &backend,
            &case.query(false),
            &expected_node,
            &[],
            &case.label(false),
        )?;
    }
    Ok(())
}

#[test]
fn strict_cpu_requires_temporal_create_values_in_statement_local_return_overlay() -> Result<()> {
    for case in TEMPORAL_CASES {
        let properties = [("created", case.value)];
        let expected_node = [ExpectedNode {
            labels: &[],
            properties: &properties,
        }];
        let expected_columns = [ExpectedColumn {
            name: "created",
            value: case.value,
        }];
        let fixture = Fixture::empty();
        let backend = fixture.strict_cpu_backend()?;
        run_success_case(
            &fixture,
            &backend,
            &case.query(true),
            &expected_node,
            &expected_columns,
            &case.label(true),
        )?;
    }
    Ok(())
}

#[test]
fn cpu_receipts_masquerading_as_metal_fail_before_any_publication() -> Result<()> {
    let fixture = Fixture::empty();
    let backend = fixture.wrong_receipt_backend()?;
    let observations = backend.observations();
    let error = QueryEngine
        .execute(
            "CREATE (:NeverPublished {id: 12})",
            &mut context(&fixture, Some(&backend)),
        )
        .err()
        .ok_or_else(|| Error::internal("wrong-device CREATE receipts were published"))?;
    assert_eq!(error.code, ErrorCode::CorruptStorage);
    assert_eq!(observations.pins.load(Ordering::SeqCst), 1);
    assert_eq!(observations.create_calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        observations
            .receipts
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len(),
        1,
        "the real CPU receipt must exist before the executor rejects Metal provenance"
    );
    assert_route_closed(&observations, "wrong receipt provenance")?;
    assert_eq!(fixture.graph.node_count(), 0);
    assert_eq!(fixture.graph.revision(), 0);
    Ok(())
}

#[test]
fn duplicate_first_and_missing_second_intent_fail_the_whole_create_statement() -> Result<()> {
    let fixture = Fixture::empty();
    let backend = fixture.replay_backend()?;
    let observations = backend.observations();
    let error = QueryEngine
        .execute(
            "CREATE (:First {id: 1}), (:Second {id: 2})",
            &mut context(&fixture, Some(&backend)),
        )
        .err()
        .ok_or_else(|| Error::internal("replayed first CREATE intent was published twice"))?;
    assert_eq!(error.code, ErrorCode::CorruptStorage);
    assert_eq!(observations.pins.load(Ordering::SeqCst), 1);
    assert_eq!(observations.create_calls.load(Ordering::SeqCst), 2);
    assert_eq!(
        observations
            .receipts
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len(),
        1,
        "only the first distinct intent may validate"
    );
    assert_route_closed(&observations, "duplicate/missing intent fault")?;
    assert_eq!(fixture.graph.node_count(), 0);
    assert_eq!(fixture.graph.revision(), 0);
    Ok(())
}

#[test]
fn pinned_create_generation_rejects_every_replacement_surface() -> Result<()> {
    let fixture = Fixture::empty();
    let backend = fixture.strict_cpu_backend()?;
    let observations = backend.observations();
    let mut pinned = backend.pin_project(PROJECT)?;
    assert_eq!(pinned.kind(), BackendKind::Cpu);
    assert_eq!(
        pinned
            .admit_project(fixture.image()?)
            .expect_err("pinned CREATE generation admitted a replacement")
            .code,
        ErrorCode::CorruptStorage
    );
    assert_eq!(
        pinned
            .replace_all_projects(vec![fixture.image()?])
            .expect_err("pinned CREATE generation replaced all projects")
            .code,
        ErrorCode::CorruptStorage
    );
    assert_eq!(
        pinned
            .evict_project(PROJECT)
            .expect_err("pinned CREATE generation evicted its project")
            .code,
        ErrorCode::CorruptStorage
    );
    pinned.advance_bookmark(Bookmark {
        term: BOOKMARK.term,
        index: BOOKMARK.index + 1,
    });
    assert_eq!(
        observations
            .generation_mutation_attempts
            .load(Ordering::SeqCst),
        4
    );
    assert_eq!(pinned.resident_bookmark(PROJECT), Some(BOOKMARK));
    assert_eq!(
        pinned.resident_graph_revision(PROJECT),
        Some(fixture.graph.revision())
    );
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
#[ignore = "real-Metal acceptance gate; run explicitly after native CREATE integration settles"]
fn real_metal_exactly_matches_strict_cpu_without_fallback_or_host_temporal_work() -> Result<()> {
    let _metal = metal_test_guard();
    for case in CREATE_CASES {
        let cpu_fixture = Fixture::empty();
        let cpu = cpu_fixture.strict_cpu_backend()?;
        let cpu_output = run_success_case(
            &cpu_fixture,
            &cpu,
            case.query,
            case.nodes,
            case.columns,
            &format!("CPU {}", case.label()),
        )?;

        let metal_fixture = Fixture::empty();
        let metal = metal_fixture.real_metal_backend()?;
        let metal_output = run_success_case(
            &metal_fixture,
            &metal,
            case.query,
            case.nodes,
            case.columns,
            &format!("Metal {}", case.label()),
        )?;
        assert_eq!(metal_output.result, cpu_output.result, "{}", case.label());
        assert_eq!(
            mutation_debug(&metal_output),
            mutation_debug(&cpu_output),
            "{}",
            case.label()
        );
        assert_eq!(
            semantic_request_debug(&metal.observations()),
            semantic_request_debug(&cpu.observations()),
            "{}",
            case.label()
        );
    }

    for case in TEMPORAL_CASES {
        let properties = [("created", case.value)];
        let expected_node = [ExpectedNode {
            labels: &[],
            properties: &properties,
        }];
        for returning in [false, true] {
            let expected_columns = returning.then_some([ExpectedColumn {
                name: "created",
                value: case.value,
            }]);
            let columns = expected_columns
                .as_ref()
                .map_or(&[] as &[ExpectedColumn], |columns| columns.as_slice());
            let query = case.query(returning);

            let cpu_fixture = Fixture::empty();
            let cpu = cpu_fixture.strict_cpu_backend()?;
            let cpu_output = run_success_case(
                &cpu_fixture,
                &cpu,
                &query,
                &expected_node,
                columns,
                &format!("CPU {}", case.label(returning)),
            )?;

            let metal_fixture = Fixture::empty();
            let metal = metal_fixture.real_metal_backend()?;
            let metal_output = run_success_case(
                &metal_fixture,
                &metal,
                &query,
                &expected_node,
                columns,
                &format!("Metal {}", case.label(returning)),
            )?;
            assert_eq!(
                metal_output.result,
                cpu_output.result,
                "{}",
                case.label(returning)
            );
            assert_eq!(
                mutation_debug(&metal_output),
                mutation_debug(&cpu_output),
                "{}",
                case.label(returning)
            );
            assert_eq!(
                semantic_request_debug(&metal.observations()),
                semantic_request_debug(&cpu.observations()),
                "{}",
                case.label(returning)
            );
        }
    }
    Ok(())
}
