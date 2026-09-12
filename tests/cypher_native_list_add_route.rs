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
    Bookmark, Error, ProjectId, Result, ScalarValue,
    cypher::{BindCapabilities, ExecutionContext, QueryEngine, ResultValue},
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend, ResidentGroup,
        ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest, ResidentNodePipelineRequest,
        ResidentNodePipelineResult, ResidentProjectImage, ResidentScalarProgramOpcode,
        ResidentScalarProgramOperand, ResidentScalarProgramRequest, ResidentScalarProgramResult,
        ResidentSortRequest, ResidentSortResult, ResidentVectorQuery, ResidentVectorResult,
        ScratchReservation,
    },
    graph::{GraphStore, LayerMask},
    types::{LabelId, PropertyId},
};
use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::gpu::MetalBackend;
#[cfg(all(feature = "accelerator", target_os = "macos"))]
use std::sync::MutexGuard;

const LIST4_FEATURE: &str = "features/expressions/list/List4.feature";
const PRECEDENCE3_FEATURE: &str = "features/expressions/precedence/Precedence3.feature";
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

#[derive(Default)]
struct ScalarObservations {
    calls: AtomicUsize,
    requests: Mutex<Vec<ResidentScalarProgramRequest>>,
}

/// Observes the one native scalar boundary without implementing any Cypher semantics itself.
///
/// The CPU semantic reference deliberately reports `Metal` to the query engine. That activates
/// strict accelerator admission while still delegating the canonical request to the real
/// `CpuBackend` scalar implementation. If lowering or execution falls back to the generic host
/// evaluator, `execute_scalar_program` is not called and every route assertion below fails.
struct ObservedScalarBackend {
    inner: Box<dyn ExecutionBackend>,
    actual_kind: BackendKind,
    observations: Arc<ScalarObservations>,
}

impl ObservedScalarBackend {
    fn strict_cpu_reference() -> Self {
        Self {
            inner: Box::new(CpuBackend::new(64 * 1024 * 1024, 16 * 1024 * 1024)),
            actual_kind: BackendKind::Cpu,
            observations: Arc::new(ScalarObservations::default()),
        }
    }

    #[cfg(all(feature = "accelerator", target_os = "macos"))]
    fn real_metal() -> Result<Self> {
        let inner = MetalBackend::new(0, 128 * 1024 * 1024, 32 * 1024 * 1024)?;
        if inner.kind() != BackendKind::Metal {
            return Err(Error::internal(
                "native list-add test did not construct a real Metal backend",
            ));
        }
        Ok(Self {
            inner: Box::new(inner),
            actual_kind: BackendKind::Metal,
            observations: Arc::new(ScalarObservations::default()),
        })
    }

    fn observations(&self) -> Arc<ScalarObservations> {
        Arc::clone(&self.observations)
    }
}

impl ExecutionBackend for ObservedScalarBackend {
    fn kind(&self) -> BackendKind {
        // CPU is intentionally presented as an accelerator so strict native execution cannot
        // silently enter the ordinary CPU evaluator. A real Metal backend reports Metal too.
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
        self.inner.execute_node_pipeline(request, cancellation)
    }

    fn supports_native_scalar_program(&self) -> bool {
        self.inner.supports_native_scalar_program()
    }

    fn execute_scalar_program(
        &self,
        request: &ResidentScalarProgramRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentScalarProgramResult> {
        self.observations.calls.fetch_add(1, Ordering::SeqCst);
        self.observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(request.clone());
        self.inner.execute_scalar_program(request, cancellation)
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

fn context<'a>(graph: &'a GraphStore, backend: &'a dyn ExecutionBackend) -> ExecutionContext<'a> {
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
        parameters: BTreeMap::new(),
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
            require_native_execution: true,
            ..BindCapabilities::default()
        },
        max_result_rows: 64,
        max_batch_rows: 64,
        optimizer_statistics: None,
        backend: Some(backend),
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(10)),
        resolved_query_at_time_nanos: None,
    }
}

fn execute_single_row(backend: &dyn ExecutionBackend, query: &str) -> Result<Vec<ResultValue>> {
    let graph = GraphStore::default();
    let output = QueryEngine.execute(query, &mut context(&graph, backend))?;
    if !output.graph_mutations.is_empty() || !output.temporal_mutations.is_empty() {
        return Err(Error::internal(
            "native list-add query unexpectedly produced side effects",
        ));
    }
    let [batch] = output.result.batches.as_slice() else {
        return Err(Error::internal(
            "native list-add query did not return one batch",
        ));
    };
    if batch.row_count != 1 || batch.columns.iter().any(|column| column.values.len() != 1) {
        return Err(Error::internal(
            "native list-add query did not return exactly one row",
        ));
    }
    Ok(batch
        .columns
        .iter()
        .map(|column| column.values[0].clone())
        .collect())
}

fn integer(value: i64) -> ResultValue {
    ResultValue::Scalar(ScalarValue::Integer(value))
}

fn boolean(value: bool) -> ResultValue {
    ResultValue::Scalar(ScalarValue::Boolean(value))
}

macro_rules! list_value {
    ($($value:expr),* $(,)?) => {
        ResultValue::List(vec![$($value),*])
    };
}

#[derive(Clone)]
struct NativeListCase {
    tck_identity: Option<(usize, &'static str)>,
    name: &'static str,
    query: &'static str,
    expected: Vec<ResultValue>,
}

fn list4_scenario_1() -> NativeListCase {
    NativeListCase {
        tck_identity: Some((1711, LIST4_FEATURE)),
        name: "[1] Concatenating lists of same type",
        query: "RETURN [1, 10, 100] + [4, 5] AS foo",
        expected: vec![list_value![
            integer(1),
            integer(10),
            integer(100),
            integer(4),
            integer(5),
        ]],
    }
}

fn list4_scenario_2() -> NativeListCase {
    NativeListCase {
        tck_identity: Some((1712, LIST4_FEATURE)),
        name: "[2] Concatenating a list with a scalar of same type",
        query: "RETURN [false, true] + false AS foo",
        expected: vec![list_value![boolean(false), boolean(true), boolean(false)]],
    }
}

fn precedence3_scenario_1() -> NativeListCase {
    let appended = list_value![
        list_value![integer(1)],
        list_value![integer(2), integer(3)],
        list_value![integer(4), integer(5)],
        integer(10),
    ];
    NativeListCase {
        tck_identity: Some((2157, PRECEDENCE3_FEATURE)),
        name: "[1] List element access takes precedence over list appending",
        query: "RETURN [[1], [2, 3], [4, 5]] + [5, [6, 7], [8, 9], 10][3] AS a, \
                        [[1], [2, 3], [4, 5]] + ([5, [6, 7], [8, 9], 10][3]) AS b, \
                        ([[1], [2, 3], [4, 5]] + [5, [6, 7], [8, 9], 10])[3] AS c",
        expected: vec![appended.clone(), appended, integer(5)],
    }
}

fn precedence3_scenario_2() -> NativeListCase {
    let concatenated = list_value![
        list_value![integer(1)],
        list_value![integer(2), integer(3)],
        list_value![integer(4), integer(5)],
        integer(8),
        integer(9),
    ];
    NativeListCase {
        tck_identity: Some((2158, PRECEDENCE3_FEATURE)),
        name: "[2] List element access takes precedence over list concatenation",
        query: "RETURN [[1], [2, 3], [4, 5]] + [5, [6, 7], [8, 9], 10][2] AS a, \
                        [[1], [2, 3], [4, 5]] + ([5, [6, 7], [8, 9], 10][2]) AS b, \
                        ([[1], [2, 3], [4, 5]] + [5, [6, 7], [8, 9], 10])[2] AS c",
        expected: vec![
            concatenated.clone(),
            concatenated,
            list_value![integer(4), integer(5)],
        ],
    }
}

fn precedence3_scenario_3() -> NativeListCase {
    let concatenated = list_value![
        list_value![integer(1)],
        list_value![integer(2), integer(3)],
        list_value![integer(4), integer(5)],
        list_value![integer(6), integer(7)],
        list_value![integer(8), integer(9)],
    ];
    NativeListCase {
        tck_identity: Some((2159, PRECEDENCE3_FEATURE)),
        name: "[3] List slicing takes precedence over list concatenation",
        query: "RETURN [[1], [2, 3], [4, 5]] + [5, [6, 7], [8, 9], 10][1..3] AS a, \
                        [[1], [2, 3], [4, 5]] + ([5, [6, 7], [8, 9], 10][1..3]) AS b, \
                        ([[1], [2, 3], [4, 5]] + [5, [6, 7], [8, 9], 10])[1..3] AS c",
        expected: vec![
            concatenated.clone(),
            concatenated,
            list_value![
                list_value![integer(2), integer(3)],
                list_value![integer(4), integer(5)],
            ],
        ],
    }
}

fn precedence3_scenario_4() -> NativeListCase {
    NativeListCase {
        tck_identity: Some((2160, PRECEDENCE3_FEATURE)),
        name: "[4] List appending takes precedence over list element containment",
        query: "RETURN [1]+2 IN [3]+4 AS a, \
                        ([1]+2) IN ([3]+4) AS b, \
                        [1]+(2 IN [3])+4 AS c",
        expected: vec![
            boolean(false),
            boolean(false),
            list_value![integer(1), boolean(false), integer(4)],
        ],
    }
}

fn precedence3_scenario_5() -> NativeListCase {
    NativeListCase {
        tck_identity: Some((2161, PRECEDENCE3_FEATURE)),
        name: "[5] List concatenation takes precedence over list element containment",
        query: "RETURN [1]+[2] IN [3]+[4] AS a, \
                        ([1]+[2]) IN ([3]+[4]) AS b, \
                        (([1]+[2]) IN [3])+[4] AS c, \
                        [1]+([2] IN [3])+[4] AS d",
        expected: vec![
            boolean(false),
            boolean(false),
            list_value![boolean(false), integer(4)],
            list_value![integer(1), boolean(false), integer(4)],
        ],
    }
}

fn scalar_prepend_and_nested_shape_case() -> NativeListCase {
    NativeListCase {
        tck_identity: None,
        name: "scalar + LIST prepend and one-level nested-list rules",
        query: "RETURN 0 + [1, 2] AS prepended, \
                        [1] + [[2, 3]] AS nested_element, \
                        [1] + [2, 3] AS concatenated",
        expected: vec![
            list_value![integer(0), integer(1), integer(2)],
            list_value![integer(1), list_value![integer(2), integer(3)]],
            list_value![integer(1), integer(2), integer(3)],
        ],
    }
}

fn dynamic_index_case() -> NativeListCase {
    NativeListCase {
        tck_identity: None,
        name: "dynamic list-index result consumed by LIST addition",
        query: "WITH 2 AS i RETURN [0] + [[1], [2, 3], [4, 5]][i] AS value",
        expected: vec![list_value![integer(0), integer(4), integer(5)]],
    }
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn all_cases() -> Vec<NativeListCase> {
    vec![
        list4_scenario_1(),
        list4_scenario_2(),
        precedence3_scenario_1(),
        precedence3_scenario_2(),
        precedence3_scenario_3(),
        precedence3_scenario_4(),
        precedence3_scenario_5(),
        scalar_prepend_and_nested_shape_case(),
        dynamic_index_case(),
    ]
}

fn assert_strict_cpu_case(case: NativeListCase) {
    let backend = ObservedScalarBackend::strict_cpu_reference();
    assert_eq!(backend.actual_kind, BackendKind::Cpu);
    let observations = backend.observations();
    let outcome = execute_single_row(&backend, case.query);
    let calls = observations.calls.load(Ordering::SeqCst);
    assert_eq!(
        calls, 1,
        "{} did not enter exactly one resident scalar execution under strict no-fallback \
         admission; outcome: {outcome:?}",
        case.name,
    );
    let actual = outcome.unwrap_or_else(|error| {
        panic!(
            "{} reached the native CPU semantic reference but failed instead of returning \
             {:?}: {error:?}",
            case.name, case.expected,
        )
    });
    assert_eq!(actual, case.expected, "{}", case.name);
}

fn instruction_uses_register(
    instruction: &irongraph::gpu::ResidentScalarProgramInstruction,
    register: u16,
) -> bool {
    [
        Some(instruction.left),
        Some(instruction.right),
        instruction.third,
        instruction.fourth,
    ]
    .into_iter()
    .flatten()
    .any(|operand| operand == ResidentScalarProgramOperand::Register(register))
}

fn assert_dynamic_index_dataflow(observations: &ScalarObservations) {
    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let request = requests
        .last()
        .expect("dynamic-index query did not publish a native scalar request");
    let index_registers = request
        .instructions
        .iter()
        .enumerate()
        .filter_map(|(register, instruction)| {
            (instruction.opcode == ResidentScalarProgramOpcode::ListIndex)
                .then(|| u16::try_from(register).expect("test instruction register fits in u16"))
        })
        .collect::<Vec<_>>();
    assert!(
        !index_registers.is_empty(),
        "dynamic index was not lowered to a native ListIndex instruction",
    );
    assert!(
        index_registers.iter().any(|register| {
            request
                .instructions
                .iter()
                .skip(usize::from(*register) + 1)
                .any(|instruction| instruction_uses_register(instruction, *register))
        }),
        "the ListIndex result was not wired as an SSA operand of the following list addition",
    );
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn metal_test_guard() -> MutexGuard<'static, ()> {
    static METAL_TEST: Mutex<()> = Mutex::new(());
    match METAL_TEST.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

#[test]
#[ignore = "external assurance gate: requires the certified full TCK report"]
fn certified_report_uniquely_resolves_every_official_list_add_selector() {
    let cases = [
        list4_scenario_1(),
        list4_scenario_2(),
        precedence3_scenario_1(),
        precedence3_scenario_2(),
        precedence3_scenario_3(),
        precedence3_scenario_4(),
        precedence3_scenario_5(),
    ];
    assert_certified_report_identities(cases.iter().map(|case| {
        let (report_id, feature) = case
            .tck_identity
            .expect("official list-add case has a TCK identity");
        (report_id, feature, case.name)
    }));
}

#[test]
fn official_list4_scenario_1_list_plus_list_on_strict_cpu_reference() {
    assert_strict_cpu_case(list4_scenario_1());
}

#[test]
fn official_list4_scenario_2_list_plus_scalar_on_strict_cpu_reference() {
    assert_strict_cpu_case(list4_scenario_2());
}

#[test]
fn precedence3_scenario_1_index_before_append_on_strict_cpu_reference() {
    assert_strict_cpu_case(precedence3_scenario_1());
}

#[test]
fn precedence3_scenario_2_index_before_concat_on_strict_cpu_reference() {
    assert_strict_cpu_case(precedence3_scenario_2());
}

#[test]
fn precedence3_scenario_3_slice_before_concat_on_strict_cpu_reference() {
    assert_strict_cpu_case(precedence3_scenario_3());
}

#[test]
fn precedence3_scenario_4_append_before_in_on_strict_cpu_reference() {
    assert_strict_cpu_case(precedence3_scenario_4());
}

#[test]
fn precedence3_scenario_5_concat_before_in_on_strict_cpu_reference() {
    assert_strict_cpu_case(precedence3_scenario_5());
}

#[test]
fn scalar_prepend_and_nested_list_shape_rules_on_strict_cpu_reference() {
    assert_strict_cpu_case(scalar_prepend_and_nested_shape_case());
}

#[test]
fn dynamic_list_index_is_an_ssa_operand_of_native_list_addition() {
    let backend = ObservedScalarBackend::strict_cpu_reference();
    let observations = backend.observations();
    let outcome = execute_single_row(&backend, dynamic_index_case().query);
    assert_eq!(
        observations.calls.load(Ordering::SeqCst),
        1,
        "dynamic-index addition bypassed the strict native scalar route: {outcome:?}",
    );
    assert_dynamic_index_dataflow(&observations);
}

#[test]
fn dynamic_list_index_semantics_on_strict_cpu_reference() {
    assert_strict_cpu_case(dynamic_index_case());
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_executes_the_scalar_kernel_and_receives_dynamic_list_addition() -> Result<()> {
    let _metal = metal_test_guard();
    let backend = ObservedScalarBackend::real_metal()?;
    assert_eq!(backend.actual_kind, BackendKind::Metal);
    let observations = backend.observations();

    // This supported index-only query must complete through the same actual Metal scalar kernel.
    assert_eq!(
        execute_single_row(&backend, "RETURN [[1], [2, 3]][1] AS value")?,
        vec![list_value![integer(2), integer(3)]],
    );
    let outcome = execute_single_row(&backend, dynamic_index_case().query);
    assert_eq!(
        observations.calls.load(Ordering::SeqCst),
        2,
        "real Metal was not invoked for both the control and list-add query: {outcome:?}",
    );
    assert_dynamic_index_dataflow(&observations);
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_matches_the_strict_cpu_reference_for_all_list_add_cases() -> Result<()> {
    let _metal = metal_test_guard();
    let cpu = ObservedScalarBackend::strict_cpu_reference();
    let metal = ObservedScalarBackend::real_metal()?;
    assert_eq!(cpu.actual_kind, BackendKind::Cpu);
    assert_eq!(metal.actual_kind, BackendKind::Metal);
    let cpu_observations = cpu.observations();
    let metal_observations = metal.observations();
    let mut defects = Vec::new();

    for case in all_cases() {
        let cpu_before = cpu_observations.calls.load(Ordering::SeqCst);
        let metal_before = metal_observations.calls.load(Ordering::SeqCst);
        let cpu_outcome = execute_single_row(&cpu, case.query);
        let metal_outcome = execute_single_row(&metal, case.query);
        let cpu_calls = cpu_observations
            .calls
            .load(Ordering::SeqCst)
            .saturating_sub(cpu_before);
        let metal_calls = metal_observations
            .calls
            .load(Ordering::SeqCst)
            .saturating_sub(metal_before);

        if cpu_calls != 1 || metal_calls != 1 {
            defects.push(format!(
                "{}: native dispatch count CPU={cpu_calls}, Metal={metal_calls}; \
                 CPU={cpu_outcome:?}; Metal={metal_outcome:?}",
                case.name,
            ));
            continue;
        }
        match (&cpu_outcome, &metal_outcome) {
            (Ok(cpu_values), Ok(metal_values)) => {
                if cpu_values != &case.expected {
                    defects.push(format!(
                        "{}: CPU returned {cpu_values:?}, expected {:?}",
                        case.name, case.expected,
                    ));
                }
                if metal_values != &case.expected {
                    defects.push(format!(
                        "{}: Metal returned {metal_values:?}, expected {:?}",
                        case.name, case.expected,
                    ));
                }
                if metal_values != cpu_values {
                    defects.push(format!(
                        "{}: CPU/Metal mismatch: CPU={cpu_values:?}, Metal={metal_values:?}",
                        case.name,
                    ));
                }
            }
            _ => defects.push(format!(
                "{}: native execution did not return a row; CPU={cpu_outcome:?}; \
                 Metal={metal_outcome:?}",
                case.name,
            )),
        }
    }

    assert!(
        defects.is_empty(),
        "native list addition is not conformant:\n{}",
        defects.join("\n"),
    );
    Ok(())
}
