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
    Bookmark, Error, ErrorCode, ProjectId, Result, ScalarValue,
    cypher::{BindCapabilities, ExecutionContext, QueryEngine, ResultValue},
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend, ResidentGroup,
        ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest, ResidentNodePipelineRequest,
        ResidentNodePipelineResult, ResidentProjectImage, ResidentScalarCellTag,
        ResidentScalarProgramOpcode, ResidentScalarProgramOperand, ResidentScalarProgramRequest,
        ResidentScalarProgramResult, ResidentSortRequest, ResidentSortResult, ResidentVectorQuery,
        ResidentVectorResult, ScratchReservation,
    },
    graph::{GraphStore, LayerMask},
    types::{LabelId, PropertyId},
};
use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::gpu::MetalBackend;

const FEATURE: &str = "features/expressions/conditional/Conditional2.feature";
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

const SIMPLE_CASE_TEMPLATE: &str = r#"RETURN CASE {value}
    WHEN -10 THEN 'minus ten'
    WHEN 0 THEN 'zero'
    WHEN 1 THEN 'one'
    WHEN 5 THEN 'five'
    WHEN 10 THEN 'ten'
    WHEN 3000 THEN 'three thousand'
    ELSE 'something else'
END AS result"#;

#[derive(Clone, Copy, Debug)]
struct SimpleCaseScenario {
    id: u32,
    report_name: &'static str,
    value: &'static str,
    expected: &'static str,
}

const SCENARIOS: [SimpleCaseScenario; 12] = [
    SimpleCaseScenario {
        id: 1509,
        report_name: "[1] Simple cases over integers [1510]",
        value: "-10",
        expected: "minus ten",
    },
    SimpleCaseScenario {
        id: 1510,
        report_name: "[1] Simple cases over integers [1511]",
        value: "0",
        expected: "zero",
    },
    SimpleCaseScenario {
        id: 1511,
        report_name: "[1] Simple cases over integers [1512]",
        value: "1",
        expected: "one",
    },
    SimpleCaseScenario {
        id: 1512,
        report_name: "[1] Simple cases over integers [1513]",
        value: "5",
        expected: "five",
    },
    SimpleCaseScenario {
        id: 1513,
        report_name: "[1] Simple cases over integers [1514]",
        value: "10",
        expected: "ten",
    },
    SimpleCaseScenario {
        id: 1514,
        report_name: "[1] Simple cases over integers [1515]",
        value: "3000",
        expected: "three thousand",
    },
    SimpleCaseScenario {
        id: 1515,
        report_name: "[1] Simple cases over integers [1516]",
        value: "-30",
        expected: "something else",
    },
    SimpleCaseScenario {
        id: 1516,
        report_name: "[1] Simple cases over integers [1517]",
        value: "3",
        expected: "something else",
    },
    SimpleCaseScenario {
        id: 1517,
        report_name: "[1] Simple cases over integers [1518]",
        value: "3001",
        expected: "something else",
    },
    SimpleCaseScenario {
        id: 1518,
        report_name: "[1] Simple cases over integers [1519]",
        value: "'0'",
        expected: "something else",
    },
    SimpleCaseScenario {
        id: 1519,
        report_name: "[1] Simple cases over integers [1520]",
        value: "true",
        expected: "something else",
    },
    SimpleCaseScenario {
        id: 1520,
        report_name: "[1] Simple cases over integers [1521]",
        value: "10.1",
        expected: "something else",
    },
];

impl SimpleCaseScenario {
    fn query(self) -> String {
        SIMPLE_CASE_TEMPLATE.replace("{value}", self.value)
    }

    fn label(self) -> String {
        format!(
            "TCK {} {FEATURE} {} (CASE {})",
            self.id, self.report_name, self.value
        )
    }

    fn expected(self) -> ResultValue {
        ResultValue::Scalar(ScalarValue::String(Arc::from(self.expected)))
    }
}

#[derive(Default)]
struct ScalarObservations {
    calls: AtomicUsize,
    requests: Mutex<Vec<ResidentScalarProgramRequest>>,
}

/// Reports accelerator identity to the query engine while retaining the actual backend for the
/// test oracle. The strict execution flag therefore makes any missed resident compilation fail
/// with `GpuAdmissionFailure`; it cannot drop into the generic host expression evaluator.
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
                "simple-CASE route did not construct a real Metal backend",
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
        deadline: Some(Instant::now() + Duration::from_secs(30)),
        resolved_query_at_time_nanos: None,
    }
}

fn execute_scenario(
    backend: &dyn ExecutionBackend,
    scenario: SimpleCaseScenario,
) -> Result<ResultValue> {
    let graph = GraphStore::default();
    let output = QueryEngine.execute(&scenario.query(), &mut context(&graph, backend))?;
    if !output.graph_mutations.is_empty() || !output.temporal_mutations.is_empty() {
        return Err(Error::internal(format!(
            "{} unexpectedly produced side effects",
            scenario.label(),
        )));
    }
    if output.result.schema.len() != 1 || output.result.schema[0].0 != "result" {
        return Err(Error::internal(format!(
            "{} returned the wrong schema: {:?}",
            scenario.label(),
            output.result.schema,
        )));
    }
    let [batch] = output.result.batches.as_slice() else {
        return Err(Error::internal(format!(
            "{} did not return exactly one batch",
            scenario.label(),
        )));
    };
    if batch.row_count != 1 || batch.columns.len() != 1 || batch.columns[0].values.len() != 1 {
        return Err(Error::internal(format!(
            "{} did not return exactly one value",
            scenario.label(),
        )));
    }
    Ok(batch.columns[0].values[0].clone())
}

fn assert_native_case_program(
    request: &ResidentScalarProgramRequest,
    scenario: SimpleCaseScenario,
) -> Result<()> {
    let count = |opcode| {
        request
            .instructions
            .iter()
            .filter(|instruction| instruction.opcode == opcode)
            .count()
    };
    if count(ResidentScalarProgramOpcode::ListMembership) != 6
        || count(ResidentScalarProgramOpcode::ToInteger) != 6
        || count(ResidentScalarProgramOpcode::NumericSubtract) != 6
        || count(ResidentScalarProgramOpcode::NumericAdd) != 6
        || count(ResidentScalarProgramOpcode::NumericMultiply) != 18
        || count(ResidentScalarProgramOpcode::ListIndex) != 1
    {
        return Err(Error::internal(format!(
            "{} was not lowered as six native equality branches plus one native selection: {:?}",
            scenario.label(),
            request.instructions,
        )));
    }
    if count(ResidentScalarProgramOpcode::NumericNegative) < 1
        || count(ResidentScalarProgramOpcode::BuildList) < 1
    {
        return Err(Error::internal(format!(
            "{} folded the signed WHEN -10 candidate on the host",
            scenario.label(),
        )));
    }
    if scenario.id == 1509 && count(ResidentScalarProgramOpcode::NumericNegative) < 2 {
        return Err(Error::internal(
            "TCK 1509 folded either the CASE operand or WHEN -10 on the host",
        ));
    }
    if request.false_cell.is_none() || request.true_cell.is_none() {
        return Err(Error::internal(format!(
            "{} omitted the native equality Boolean cells",
            scenario.label(),
        )));
    }
    let final_register = u16::try_from(request.instructions.len().saturating_sub(1))
        .map_err(|_| Error::internal("simple CASE register index overflowed"))?;
    if request.output_values != [ResidentScalarProgramOperand::Register(final_register)] {
        return Err(Error::internal(format!(
            "{} published a compiler-owned value instead of the final device register",
            scenario.label(),
        )));
    }
    let final_instruction = request
        .instructions
        .last()
        .ok_or_else(|| Error::internal("simple CASE emitted no scalar instructions"))?;
    if final_instruction.opcode != ResidentScalarProgramOpcode::ListIndex
        || !matches!(
            final_instruction.right,
            ResidentScalarProgramOperand::Register(_)
        )
    {
        return Err(Error::internal(format!(
            "{} did not select its branch through a device-computed list index",
            scenario.label(),
        )));
    }
    let ResidentScalarProgramOperand::Cell(branch_list) = final_instruction.left else {
        return Err(Error::internal(format!(
            "{} did not retain its seven branch values as a resident input list",
            scenario.label(),
        )));
    };
    let branch_list = request
        .scalar_cells
        .get(branch_list as usize)
        .ok_or_else(|| Error::internal("simple CASE branch list cell was out of range"))?;
    if branch_list.tag != ResidentScalarCellTag::List || branch_list.auxiliary != 7 {
        return Err(Error::internal(format!(
            "{} branch list was malformed: {branch_list:?}",
            scenario.label(),
        )));
    }
    Ok(())
}

fn run_all(backend: &ObservedScalarBackend) -> std::result::Result<(), Vec<String>> {
    let observations = backend.observations();
    let mut defects = Vec::new();
    for scenario in SCENARIOS {
        let before = observations.calls.load(Ordering::SeqCst);
        let outcome = execute_scenario(backend, scenario);
        let calls = observations
            .calls
            .load(Ordering::SeqCst)
            .saturating_sub(before);
        if calls != 1 {
            defects.push(format!(
                "{} native dispatch count was {calls}, expected 1; outcome={outcome:?}",
                scenario.label(),
            ));
            continue;
        }
        match outcome {
            Ok(value) if value == scenario.expected() => {}
            Ok(value) => defects.push(format!(
                "{} returned {value:?}, expected {:?}",
                scenario.label(),
                scenario.expected(),
            )),
            Err(error) => defects.push(format!("{} failed: {error:?}", scenario.label())),
        }
        let requests = observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        match requests.last() {
            Some(request) => {
                if let Err(error) = assert_native_case_program(request, scenario) {
                    defects.push(error.to_string());
                }
            }
            None => defects.push(format!(
                "{} native dispatch published no scalar request",
                scenario.label(),
            )),
        }
    }
    if defects.is_empty() {
        Ok(())
    } else {
        Err(defects)
    }
}

#[test]
fn manifest_is_exactly_conditional2_report_indices_1509_through_1520() {
    assert_eq!(SCENARIOS.len(), 12);
    assert_eq!(
        SCENARIOS
            .iter()
            .map(|scenario| scenario.id)
            .collect::<Vec<_>>(),
        (1509..=1520).collect::<Vec<_>>(),
    );
}

#[test]
#[ignore = "external assurance gate: requires the certified full TCK report"]
fn certified_report_uniquely_resolves_every_simple_case_selector() {
    assert_certified_report_identities(
        SCENARIOS
            .iter()
            .map(|scenario| (scenario.id as usize, FEATURE, scenario.report_name)),
    );
}

#[test]
fn strict_native_cpu_reference_passes_all_12_without_host_case_evaluation() {
    let backend = ObservedScalarBackend::strict_cpu_reference();
    assert_eq!(backend.actual_kind, BackendKind::Cpu);
    if let Err(defects) = run_all(&backend) {
        panic!(
            "strict native CPU simple-CASE defects:\n{}",
            defects.join("\n")
        );
    }
    assert_eq!(
        backend.observations().calls.load(Ordering::SeqCst),
        SCENARIOS.len(),
    );
}

#[test]
fn null_bearing_simple_case_fails_closed_until_the_scalar_abi_can_coalesce_unknown() {
    let backend = ObservedScalarBackend::strict_cpu_reference();
    let graph = GraphStore::default();
    let error = QueryEngine
        .execute(
            "RETURN CASE null WHEN null THEN 'wrong' ELSE 'default' END AS result",
            &mut context(&graph, &backend),
        )
        .expect_err("NULL-bearing simple CASE must not be admitted by the non-null native route");
    assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
    assert_eq!(backend.observations().calls.load(Ordering::SeqCst), 0);
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "hardware acceptance gate: requires a real Metal device"]
fn real_metal_passes_all_12_with_one_scalar_kernel_dispatch_each() -> Result<()> {
    static METAL_TEST: Mutex<()> = Mutex::new(());
    let _guard = METAL_TEST
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let backend = ObservedScalarBackend::real_metal()?;
    if backend.actual_kind != BackendKind::Metal {
        return Err(Error::internal(
            "simple-CASE hardware gate did not use Metal",
        ));
    }
    if let Err(defects) = run_all(&backend) {
        return Err(Error::internal(format!(
            "real Metal simple-CASE defects:\n{}",
            defects.join("\n"),
        )));
    }
    if backend.observations().calls.load(Ordering::SeqCst) != SCENARIOS.len() {
        return Err(Error::internal(
            "real Metal did not execute exactly one scalar program per scenario",
        ));
    }
    Ok(())
}
