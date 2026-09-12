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
    Bookmark, Error, ErrorCode, ProjectId, Result, ScalarValue,
    cypher::{BindCapabilities, ExecutionContext, QueryEngine, ResultValue},
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend, ResidentGroup,
        ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest, ResidentNodePipelineRequest,
        ResidentNodePipelineResult, ResidentProjectImage, ResidentScalarCell,
        ResidentScalarCellTag, ResidentScalarListEntry, ResidentScalarProgramOperand,
        ResidentScalarProgramRequest, ResidentScalarProgramResult, ResidentSortRequest,
        ResidentSortResult, ResidentVectorQuery, ResidentVectorResult, ScratchReservation,
    },
    graph::{GraphStore, LayerMask},
    types::{LabelId, PropertyId},
};
use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::gpu::MetalBackend;

const FEATURE: &str = "features/expressions/map/Map3.feature";
const MAP_KEYS_OPCODE_WIRE: u32 = 17;
const MAX_RESULT_ROWS: usize = 64;
const MEMORY_LIMIT_BYTES: usize = 128 * 1024 * 1024;
const RESERVED_BYTES: usize = 32 * 1024 * 1024;
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
    let mut selected = BTreeSet::new();
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

#[derive(Clone, Copy, Debug)]
enum ExpectedValue {
    Null,
    Boolean(bool),
    KeySet(&'static [&'static str]),
}

#[derive(Clone, Copy, Debug)]
struct MapKeysCase {
    report_id: u16,
    name: &'static str,
    query: &'static str,
    parameter_map: bool,
    expected: &'static [ExpectedValue],
}

impl MapKeysCase {
    fn label(self) -> String {
        format!("TCK {} {FEATURE}: {}", self.report_id, self.name)
    }

    fn parameters(self) -> BTreeMap<String, ResultValue> {
        if !self.parameter_map {
            return BTreeMap::new();
        }
        BTreeMap::from([(
            "param".to_owned(),
            ResultValue::Map(BTreeMap::from([
                (
                    "address".to_owned(),
                    ResultValue::Map(BTreeMap::from([
                        ("city".to_owned(), string_value("London")),
                        ("residential".to_owned(), boolean_value(true)),
                    ])),
                ),
                ("age".to_owned(), integer_value(38)),
                ("name".to_owned(), string_value("Alice")),
            ])),
        )])
    }
}

const CASES: [MapKeysCase; 11] = [
    MapKeysCase {
        report_id: 1941,
        name: "[1] Using `keys()` on a literal map",
        query: "RETURN keys({name: 'Alice', age: 38, address: {city: 'London', residential: true}}) AS k",
        parameter_map: false,
        expected: &[ExpectedValue::KeySet(&["name", "age", "address"])],
    },
    MapKeysCase {
        report_id: 1942,
        name: "[2] Using `keys()` on a parameter map",
        query: "RETURN keys($param) AS k",
        parameter_map: true,
        expected: &[ExpectedValue::KeySet(&["address", "name", "age"])],
    },
    MapKeysCase {
        report_id: 1943,
        name: "[3] Using `keys()` on null map",
        query: "WITH null AS m RETURN keys(m), keys(null)",
        parameter_map: false,
        expected: &[ExpectedValue::Null, ExpectedValue::Null],
    },
    MapKeysCase {
        report_id: 1944,
        name: "[4] Using `keys()` on map with null values [1945]",
        query: "RETURN keys({}) AS keys",
        parameter_map: false,
        expected: &[ExpectedValue::KeySet(&[])],
    },
    MapKeysCase {
        report_id: 1945,
        name: "[4] Using `keys()` on map with null values [1946]",
        query: "RETURN keys({k: 1}) AS keys",
        parameter_map: false,
        expected: &[ExpectedValue::KeySet(&["k"])],
    },
    MapKeysCase {
        report_id: 1946,
        name: "[4] Using `keys()` on map with null values [1947]",
        query: "RETURN keys({k: null}) AS keys",
        parameter_map: false,
        expected: &[ExpectedValue::KeySet(&["k"])],
    },
    MapKeysCase {
        report_id: 1947,
        name: "[4] Using `keys()` on map with null values [1948]",
        query: "RETURN keys({k: null, l: 1}) AS keys",
        parameter_map: false,
        expected: &[ExpectedValue::KeySet(&["k", "l"])],
    },
    MapKeysCase {
        report_id: 1948,
        name: "[4] Using `keys()` on map with null values [1949]",
        query: "RETURN keys({k: 1, l: null}) AS keys",
        parameter_map: false,
        expected: &[ExpectedValue::KeySet(&["k", "l"])],
    },
    MapKeysCase {
        report_id: 1949,
        name: "[4] Using `keys()` on map with null values [1950]",
        query: "RETURN keys({k: null, l: null}) AS keys",
        parameter_map: false,
        expected: &[ExpectedValue::KeySet(&["k", "l"])],
    },
    MapKeysCase {
        report_id: 1950,
        name: "[4] Using `keys()` on map with null values [1951]",
        query: "RETURN keys({k: 1, l: null, m: 1}) AS keys",
        parameter_map: false,
        expected: &[ExpectedValue::KeySet(&["k", "l", "m"])],
    },
    MapKeysCase {
        report_id: 1951,
        name: "[5] Using `keys()` and `IN` to check field existence",
        query: "WITH {exists: 42, notMissing: null} AS map \
                RETURN 'exists' IN keys(map) AS a, \
                       'notMissing' IN keys(map) AS b, \
                       'missing' IN keys(map) AS c",
        parameter_map: false,
        expected: &[
            ExpectedValue::Boolean(true),
            ExpectedValue::Boolean(true),
            ExpectedValue::Boolean(false),
        ],
    },
];

fn string_value(value: &str) -> ResultValue {
    ResultValue::Scalar(ScalarValue::String(Arc::from(value)))
}

fn integer_value(value: i64) -> ResultValue {
    ResultValue::Scalar(ScalarValue::Integer(value))
}

fn boolean_value(value: bool) -> ResultValue {
    ResultValue::Scalar(ScalarValue::Boolean(value))
}

fn context<'a>(
    graph: &'a GraphStore,
    backend: &'a dyn ExecutionBackend,
    parameters: BTreeMap<String, ResultValue>,
    require_native_execution: bool,
) -> ExecutionContext<'a> {
    ExecutionContext {
        project_id: ProjectId::random(),
        graph,
        binding_catalog: graph.catalog(),
        prior_graph_mutations: &[],
        temporal: None,
        prior_temporal_mutations: &[],
        vector_indexes: BTreeMap::new(),
        scalar_indexes: None,
        text_embedding: None,
        parameters,
        bookmark: Bookmark {
            term: 0,
            index: graph.revision(),
        },
        mutation_revision: graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id: 1,
        next_edge_id: 1,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            require_native_execution,
            ..BindCapabilities::default()
        },
        max_result_rows: MAX_RESULT_ROWS,
        max_batch_rows: MAX_RESULT_ROWS,
        optimizer_statistics: None,
        backend: Some(backend),
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(10)),
        resolved_query_at_time_nanos: None,
    }
}

fn execute_case(
    backend: &dyn ExecutionBackend,
    case: MapKeysCase,
    require_native_execution: bool,
) -> Result<Vec<ResultValue>> {
    let graph = GraphStore::default();
    let output = QueryEngine.execute(
        case.query,
        &mut context(&graph, backend, case.parameters(), require_native_execution),
    )?;
    if !output.graph_mutations.is_empty() || !output.temporal_mutations.is_empty() {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!("{} unexpectedly produced side effects", case.label()),
        ));
    }
    let [batch] = output.result.batches.as_slice() else {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!("{} did not return exactly one result batch", case.label()),
        ));
    };
    if batch.row_count != 1
        || batch.columns.len() != case.expected.len()
        || batch.columns.iter().any(|column| column.values.len() != 1)
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "{} returned the wrong row/column shape: {batch:#?}",
                case.label()
            ),
        ));
    }
    Ok(batch
        .columns
        .iter()
        .map(|column| column.values[0].clone())
        .collect())
}

fn assert_expected(case: MapKeysCase, actual: &[ResultValue]) -> Result<()> {
    if actual.len() != case.expected.len() {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "{} returned {} values instead of {}",
                case.label(),
                actual.len(),
                case.expected.len()
            ),
        ));
    }
    for (column, (actual, expected)) in actual.iter().zip(case.expected).enumerate() {
        let matches = match expected {
            ExpectedValue::Null => {
                matches!(actual, ResultValue::Scalar(ScalarValue::Null))
            }
            ExpectedValue::Boolean(expected) => {
                matches!(actual, ResultValue::Scalar(ScalarValue::Boolean(actual)) if actual == expected)
            }
            ExpectedValue::KeySet(expected) => {
                let ResultValue::List(values) = actual else {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        format!(
                            "{} column {column} returned {actual:?}, not a LIST",
                            case.label()
                        ),
                    ));
                };
                let keys = values
                    .iter()
                    .map(|value| match value {
                        ResultValue::Scalar(ScalarValue::String(value)) => Ok(value.as_ref()),
                        _ => Err(Error::new(
                            ErrorCode::CorruptStorage,
                            format!(
                                "{} column {column} contains a non-STRING key {value:?}",
                                case.label()
                            ),
                        )),
                    })
                    .collect::<Result<Vec<_>>>()?;
                let actual_set = keys.iter().copied().collect::<BTreeSet<_>>();
                let expected_set = expected.iter().copied().collect::<BTreeSet<_>>();
                keys.len() == expected.len()
                    && actual_set.len() == keys.len()
                    && actual_set == expected_set
            }
        };
        if !matches {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                format!(
                    "{} column {column} returned {actual:?}, expected {expected:?}",
                    case.label()
                ),
            ));
        }
    }
    Ok(())
}

#[derive(Default)]
struct RouteObservations {
    scalar_calls: AtomicUsize,
    forbidden_calls: AtomicUsize,
    rejected_scalar_images: AtomicUsize,
    requests: Mutex<Vec<ResidentScalarProgramRequest>>,
}

/// A strict accelerator-facing wrapper. It never implements Cypher semantics itself: the inner
/// backend does the one admitted scalar program. CPU is deliberately allowed to report as Metal
/// only in the provenance-sabotage test; once map keys are lowered, an unreceipted CPU result must
/// not be publishable as Metal.
struct ObservedMapKeysBackend {
    inner: Box<dyn ExecutionBackend>,
    actual_kind: BackendKind,
    publication_kind: BackendKind,
    advertise_scalar: bool,
    observations: Arc<RouteObservations>,
}

impl ObservedMapKeysBackend {
    fn strict_cpu_reference() -> Self {
        Self {
            inner: Box::new(CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES)),
            actual_kind: BackendKind::Cpu,
            publication_kind: BackendKind::Cpu,
            advertise_scalar: true,
            observations: Arc::new(RouteObservations::default()),
        }
    }

    fn strict_without_native_map_keys() -> Self {
        Self {
            inner: Box::new(CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES)),
            actual_kind: BackendKind::Cpu,
            publication_kind: BackendKind::Cpu,
            advertise_scalar: false,
            observations: Arc::new(RouteObservations::default()),
        }
    }

    fn cpu_masquerading_as_metal() -> Self {
        Self {
            inner: Box::new(CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES)),
            actual_kind: BackendKind::Cpu,
            publication_kind: BackendKind::Metal,
            advertise_scalar: true,
            observations: Arc::new(RouteObservations::default()),
        }
    }

    #[cfg(all(feature = "accelerator", target_os = "macos"))]
    fn real_metal() -> Result<Self> {
        let inner = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
        if inner.kind() != BackendKind::Metal {
            return Err(Error::internal(
                "map-keys gate did not construct a real Metal backend",
            ));
        }
        Ok(Self {
            inner: Box::new(inner),
            actual_kind: BackendKind::Metal,
            publication_kind: BackendKind::Metal,
            advertise_scalar: true,
            observations: Arc::new(RouteObservations::default()),
        })
    }

    fn observations(&self) -> Arc<RouteObservations> {
        Arc::clone(&self.observations)
    }

    fn forbidden<T>(&self, route: &'static str) -> Result<T> {
        self.observations
            .forbidden_calls
            .fetch_add(1, Ordering::SeqCst);
        Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!("strict map-keys test rejected unrelated backend route `{route}`"),
        ))
    }
}

impl ExecutionBackend for ObservedMapKeysBackend {
    fn kind(&self) -> BackendKind {
        // Strict planning must treat both the semantic-reference sabotage and real hardware as an
        // accelerator. Otherwise the ordinary Rust expression evaluator could call keys().
        BackendKind::Metal
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
            actual_kind: self.actual_kind,
            publication_kind: self.publication_kind,
            advertise_scalar: self.advertise_scalar,
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
        _project: ProjectId,
        _label: Option<LabelId>,
        _layers: LayerMask,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.forbidden("scan_nodes")
    }

    fn filter_node_i64(
        &self,
        _project: ProjectId,
        _property: PropertyId,
        _operation: CompareOp,
        _operand: i64,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.forbidden("filter_node_i64")
    }

    fn expand_project_out(
        &self,
        _project: ProjectId,
        _sources: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.forbidden("expand_project_out")
    }

    fn expand_project_in(
        &self,
        _project: ProjectId,
        _targets: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.forbidden("expand_project_in")
    }

    fn search_vectors(
        &self,
        _request: &ResidentVectorQuery,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentVectorResult> {
        self.forbidden("search_vectors")
    }

    fn sort_rows(
        &self,
        _request: &ResidentSortRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentSortResult> {
        self.forbidden("sort_rows")
    }

    fn join_node_i64(
        &self,
        _request: &ResidentJoinRequest,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentJoinPair>> {
        self.forbidden("join_node_i64")
    }

    fn group_node_i64(
        &self,
        _request: &ResidentGroupRequest,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentGroup>> {
        self.forbidden("group_node_i64")
    }

    fn execute_node_pipeline(
        &self,
        _request: &ResidentNodePipelineRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentNodePipelineResult> {
        self.forbidden("execute_node_pipeline")
    }

    fn supports_native_scalar_program(&self) -> bool {
        self.advertise_scalar && self.inner.supports_native_scalar_program()
    }

    fn execute_scalar_program(
        &self,
        request: &ResidentScalarProgramRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentScalarProgramResult> {
        self.observations
            .scalar_calls
            .fetch_add(1, Ordering::SeqCst);
        self.observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(request.clone());
        if let Err(error) = validate_native_map_keys_request(request) {
            self.observations
                .rejected_scalar_images
                .fetch_add(1, Ordering::SeqCst);
            return Err(error);
        }
        let result = self.inner.execute_scalar_program(request, cancellation)?;
        result.validate_for_publication(request, self.publication_kind)?;
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
        self.forbidden("filter_i64")
    }

    fn expand_out(
        &self,
        _sources: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.forbidden("expand_out")
    }

    fn exact_l2(
        &self,
        _matrix: &[f32],
        _rows: usize,
        _dimension: usize,
        _query: &[f32],
        _cancellation: &CancellationToken,
    ) -> Result<DistanceBatch> {
        self.forbidden("exact_l2")
    }
}

fn validate_native_map_keys_request(request: &ResidentScalarProgramRequest) -> Result<()> {
    let register_cell_base = request
        .scalar_cells
        .len()
        .checked_sub(request.instructions.len())
        .ok_or_else(|| Error::internal("map-keys scalar register arena underflowed"))?;
    if request.output_values.is_empty()
        || request
            .output_values
            .iter()
            .any(|output| !matches!(output, ResidentScalarProgramOperand::Register(_)))
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "map-keys output was pre-materialized as a compiler-owned scalar cell",
        ));
    }
    if request.scalar_cells[..register_cell_base]
        .iter()
        .any(|cell| cell.tag == ResidentScalarCellTag::List)
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "map-keys request contains a prebuilt static LIST result",
        ));
    }
    if request
        .scalar_list_entries
        .iter()
        .any(|entry| entry.value != request.null_cell)
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "map-keys output arena contains host-selected key cells before execution",
        ));
    }

    let map_keys_registers = request
        .instructions
        .iter()
        .enumerate()
        .filter_map(|(register, instruction)| {
            ((instruction.opcode as u32) == MAP_KEYS_OPCODE_WIRE).then_some((register, instruction))
        })
        .collect::<Vec<_>>();
    if map_keys_registers.is_empty() {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "map-keys query did not contain the typed native MapKeys opcode",
        ));
    }
    for (register, instruction) in &map_keys_registers {
        let ResidentScalarProgramOperand::Cell(source) = instruction.left else {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "MapKeys source was not an immutable typed MAP/NULL input",
            ));
        };
        let source = usize::from(source);
        if source >= register_cell_base
            || !matches!(
                request.scalar_cells[source].tag,
                ResidentScalarCellTag::Map | ResidentScalarCellTag::Null
            )
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "MapKeys source does not identify a static typed MAP/NULL cell",
            ));
        }
        let register = u16::try_from(*register)
            .map_err(|_| Error::internal("MapKeys register exceeds u16"))?;
        let published = request
            .output_values
            .contains(&ResidentScalarProgramOperand::Register(register));
        let consumed = request
            .instructions
            .iter()
            .skip(usize::from(register) + 1)
            .any(|consumer| {
                [
                    Some(consumer.left),
                    Some(consumer.right),
                    consumer.third,
                    consumer.fourth,
                ]
                .into_iter()
                .flatten()
                .any(|operand| operand == ResidentScalarProgramOperand::Register(register))
            });
        if !published && !consumed {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "MapKeys result is dead and cannot be the public query result",
            ));
        }
    }

    // Map entry identifiers already share the scalar program's one-based UTF-8 identity table.
    // Requiring every identifier to resolve here prevents a kernel from returning opaque host
    // metadata which Rust later turns into key strings.
    for entry in &request.map_entries {
        let identity = usize::from(entry.key);
        if identity == 0 || identity >= request.string_offsets.len() {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "MapKeys input uses a key absent from the device-readable UTF-8 table",
            ));
        }
        let start = request.string_offsets[identity - 1] as usize;
        let end = request.string_offsets[identity] as usize;
        let bytes = request.string_bytes.get(start..end).ok_or_else(|| {
            Error::new(
                ErrorCode::CorruptStorage,
                "MapKeys input key range escapes the device-readable UTF-8 arena",
            )
        })?;
        std::str::from_utf8(bytes).map_err(|_| {
            Error::new(
                ErrorCode::CorruptStorage,
                "MapKeys input key is not valid UTF-8",
            )
        })?;
    }
    Ok(())
}

fn assert_route_closed(observations: &RouteObservations, label: &str) -> Result<()> {
    let forbidden = observations.forbidden_calls.load(Ordering::SeqCst);
    if forbidden != 0 {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!("{label}: entered {forbidden} fallback/graph/generic backend routes"),
        ));
    }
    Ok(())
}

fn run_cpu_case(case: MapKeysCase) -> Result<()> {
    let cpu = ObservedMapKeysBackend::strict_cpu_reference();
    if cpu.actual_kind != BackendKind::Cpu || cpu.kind() != BackendKind::Metal {
        return Err(Error::internal("Map3 semantic reference is not CPU"));
    }
    let observations = cpu.observations();
    let values = execute_case(&cpu, case, true)?;
    assert_expected(case, &values)?;
    assert_eq!(
        observations.scalar_calls.load(Ordering::SeqCst),
        1,
        "{} did not use exactly one strict CPU scalar program",
        case.label()
    );
    assert_eq!(
        observations.rejected_scalar_images.load(Ordering::SeqCst),
        0,
        "{} produced a pre-materialized or malformed MapKeys image",
        case.label()
    );
    assert_route_closed(&observations, &case.label())
}

#[test]
fn manifest_is_exactly_the_eleven_failing_map3_keys_scenarios() {
    assert_eq!(CASES.len(), 11);
    assert_eq!(
        CASES.iter().map(|case| case.report_id).collect::<Vec<_>>(),
        (1941_u16..=1951).collect::<Vec<_>>()
    );
    assert_eq!(CASES.iter().filter(|case| case.parameter_map).count(), 1);
    assert!(CASES.iter().all(|case| case.query.contains("keys(")));
}

#[test]
#[ignore = "external assurance gate: requires the certified full TCK report"]
fn certified_report_uniquely_resolves_every_map_keys_selector() {
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

cpu_case_test!(cpu_tck_1942_literal_map_keys, 0);
cpu_case_test!(cpu_tck_1943_parameter_map_keys, 1);
cpu_case_test!(cpu_tck_1944_null_map_keys, 2);
cpu_case_test!(cpu_tck_1945_empty_map_keys, 3);
cpu_case_test!(cpu_tck_1946_single_non_null_map_key, 4);
cpu_case_test!(cpu_tck_1947_single_null_valued_map_key, 5);
cpu_case_test!(cpu_tck_1948_mixed_null_and_non_null_map_keys, 6);
cpu_case_test!(cpu_tck_1949_reverse_mixed_map_keys, 7);
cpu_case_test!(cpu_tck_1950_two_null_valued_map_keys, 8);
cpu_case_test!(cpu_tck_1951_three_mixed_map_keys, 9);
cpu_case_test!(cpu_tck_1952_keys_composes_with_membership, 10);

#[test]
fn strict_gpu_mode_rejects_all_eleven_before_host_or_generic_keys_evaluation() -> Result<()> {
    for case in CASES {
        let backend = ObservedMapKeysBackend::strict_without_native_map_keys();
        let observations = backend.observations();
        let error = execute_case(&backend, case, true)
            .expect_err("strict GPU mode evaluated keys() without a native map-keys program");
        assert_eq!(
            error.code,
            ErrorCode::GpuAdmissionFailure,
            "{}",
            case.label()
        );
        assert_eq!(
            observations.scalar_calls.load(Ordering::SeqCst),
            0,
            "{}",
            case.label()
        );
        assert_route_closed(&observations, &case.label())?;
    }
    Ok(())
}

#[test]
fn pre_materialized_host_key_list_is_rejected_by_the_acceptance_probe() {
    let request = ResidentScalarProgramRequest {
        scalar_cells: vec![
            ResidentScalarCell {
                tag: ResidentScalarCellTag::Null,
                payload: 0,
                auxiliary: 0,
            },
            ResidentScalarCell {
                tag: ResidentScalarCellTag::String,
                payload: 1,
                auxiliary: 1,
            },
            ResidentScalarCell {
                tag: ResidentScalarCellTag::List,
                payload: 0,
                auxiliary: 1,
            },
        ],
        map_entries: Vec::new(),
        scalar_list_entries: vec![ResidentScalarListEntry { value: 1 }],
        string_offsets: vec![0, 1],
        string_bytes: vec![b'k'],
        null_cell: 0,
        false_cell: None,
        true_cell: None,
        instructions: Vec::new(),
        output_values: vec![ResidentScalarProgramOperand::Cell(2)],
    };
    let error = validate_native_map_keys_request(&request)
        .expect_err("a host-prebuilt ['k'] result passed as native MapKeys work");
    assert_eq!(error.code, ErrorCode::CorruptStorage);
}

#[test]
fn cpu_work_cannot_masquerade_as_metal_or_supply_a_fake_completion() -> Result<()> {
    for case in CASES {
        let backend = ObservedMapKeysBackend::cpu_masquerading_as_metal();
        let observations = backend.observations();
        let error = execute_case(&backend, case, true).err().ok_or_else(|| {
            Error::new(
                ErrorCode::CorruptStorage,
                format!(
                    "{} published CPU scalar work as a Metal map-keys completion",
                    case.label()
                ),
            )
        })?;
        assert!(
            matches!(
                error.code,
                ErrorCode::GpuAdmissionFailure | ErrorCode::CorruptStorage
            ),
            "{}: {error:?}",
            case.label()
        );
        assert_route_closed(&observations, &case.label())?;
        let calls = observations.scalar_calls.load(Ordering::SeqCst);
        if calls == 0 {
            assert_eq!(
                error.code,
                ErrorCode::GpuAdmissionFailure,
                "{}: missing native lowering did not fail admission",
                case.label()
            );
        } else {
            assert_eq!(calls, 1, "{}", case.label());
            assert_eq!(
                error.code,
                ErrorCode::CorruptStorage,
                "{}: a CPU result was not rejected for wrong-device/fake receipt provenance",
                case.label()
            );
        }
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
#[ignore = "hardware acceptance gate: requires all 11 Map3 keys() scenarios on a real Metal device"]
fn real_metal_executes_all_eleven_with_typed_map_keys_and_no_fallback() -> Result<()> {
    let _metal = metal_test_guard();
    let backend = ObservedMapKeysBackend::real_metal()?;
    assert_eq!(backend.actual_kind, BackendKind::Metal);
    let observations = backend.observations();
    let mut passed = 0_usize;
    let mut defects = Vec::new();

    for case in CASES {
        let calls_before = observations.scalar_calls.load(Ordering::SeqCst);
        let forbidden_before = observations.forbidden_calls.load(Ordering::SeqCst);
        let rejected_before = observations.rejected_scalar_images.load(Ordering::SeqCst);
        let outcome = execute_case(&backend, case, true);
        let calls = observations
            .scalar_calls
            .load(Ordering::SeqCst)
            .saturating_sub(calls_before);
        let forbidden = observations
            .forbidden_calls
            .load(Ordering::SeqCst)
            .saturating_sub(forbidden_before);
        let rejected = observations
            .rejected_scalar_images
            .load(Ordering::SeqCst)
            .saturating_sub(rejected_before);

        match outcome {
            Ok(values) if calls == 1 && forbidden == 0 && rejected == 0 => {
                match assert_expected(case, &values) {
                    Ok(()) => passed += 1,
                    Err(error) => defects.push(format!("{}: {error}", case.label())),
                }
            }
            Ok(values) => defects.push(format!(
                "{}: output {values:?} used scalar_calls={calls}, forbidden={forbidden}, \
                 rejected_images={rejected}",
                case.label()
            )),
            Err(error) => defects.push(format!(
                "{}: scalar_calls={calls}, forbidden={forbidden}, rejected_images={rejected}: \
                 {error:?}",
                case.label()
            )),
        }
    }

    assert_eq!(
        passed,
        CASES.len(),
        "real Metal map-keys acceptance is {passed}/{}; no scenario was skipped:\n{}",
        CASES.len(),
        defects.join("\n")
    );
    assert_route_closed(&observations, "real Metal Map3 portfolio")?;
    Ok(())
}
