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

const FEATURE: &str = "features/expressions/list/List2.feature";
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

#[derive(Clone, Copy, Debug)]
enum ExpectedBound {
    Integer(i64),
    Null,
}

#[derive(Clone, Copy, Debug)]
enum ExpectedValue {
    List(&'static [i64]),
    Null,
}

#[derive(Clone, Copy, Debug)]
struct SliceCase {
    report_id: usize,
    name: &'static str,
    query: &'static str,
    start: ExpectedBound,
    end: ExpectedBound,
    expected: ExpectedValue,
}

const CASES: [SliceCase; 4] = [
    SliceCase {
        report_id: 1690,
        name: "[2] List slice with implicit end",
        query: "WITH [1, 2, 3] AS list RETURN list[1..] AS r",
        start: ExpectedBound::Integer(1),
        end: ExpectedBound::Integer(i64::MAX),
        expected: ExpectedValue::List(&[2, 3]),
    },
    SliceCase {
        report_id: 1691,
        name: "[3] List slice with implicit start",
        query: "WITH [1, 2, 3] AS list RETURN list[..2] AS r",
        start: ExpectedBound::Integer(0),
        end: ExpectedBound::Integer(2),
        expected: ExpectedValue::List(&[1, 2]),
    },
    SliceCase {
        report_id: 1700,
        name: "[9] List slice with null range [1701]",
        query: "WITH [1, 2, 3] AS list RETURN list[..null] AS r",
        start: ExpectedBound::Integer(0),
        end: ExpectedBound::Null,
        expected: ExpectedValue::Null,
    },
    SliceCase {
        report_id: 1701,
        name: "[9] List slice with null range [1702]",
        query: "WITH [1, 2, 3] AS list RETURN list[null..] AS r",
        start: ExpectedBound::Null,
        end: ExpectedBound::Integer(i64::MAX),
        expected: ExpectedValue::Null,
    },
];

#[derive(Default)]
struct ScalarObservations {
    calls: AtomicUsize,
    requests: Mutex<Vec<ResidentScalarProgramRequest>>,
}

/// Reports accelerator admission even around the CPU semantic reference. Consequently, a query
/// can succeed only by entering the native scalar boundary; the generic host evaluator is not an
/// available fallback. A real-Metal instance delegates that same request to Metal unchanged.
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
                "native list-slice gate did not construct a real Metal backend",
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
        deadline: Some(Instant::now() + Duration::from_secs(10)),
        resolved_query_at_time_nanos: None,
    }
}

fn execute_single_value(backend: &dyn ExecutionBackend, case: SliceCase) -> Result<ResultValue> {
    let graph = GraphStore::default();
    let output = QueryEngine.execute(case.query, &mut context(&graph, backend))?;
    if !output.graph_mutations.is_empty() || !output.temporal_mutations.is_empty() {
        return Err(Error::internal(format!(
            "{} unexpectedly produced side effects",
            case.name,
        )));
    }
    let [batch] = output.result.batches.as_slice() else {
        return Err(Error::internal(format!(
            "{} did not return exactly one batch",
            case.name,
        )));
    };
    if batch.row_count != 1
        || batch.columns.len() != 1
        || batch.columns[0].name != "r"
        || batch.columns[0].values.len() != 1
    {
        return Err(Error::internal(format!(
            "{} did not return one column and one row",
            case.name,
        )));
    }
    Ok(batch.columns[0].values[0].clone())
}

fn expected_value(expected: ExpectedValue) -> ResultValue {
    match expected {
        ExpectedValue::Null => ResultValue::Scalar(ScalarValue::Null),
        ExpectedValue::List(values) => ResultValue::List(
            values
                .iter()
                .copied()
                .map(|value| ResultValue::Scalar(ScalarValue::Integer(value)))
                .collect(),
        ),
    }
}

fn validate_bound(
    request: &ResidentScalarProgramRequest,
    operand: ResidentScalarProgramOperand,
    expected: ExpectedBound,
    description: &str,
) -> Result<()> {
    let ResidentScalarProgramOperand::Cell(index) = operand else {
        return Err(Error::internal(format!(
            "{description} was computed or host-materialized instead of being an immutable native operand",
        )));
    };
    let cell = request
        .scalar_cells
        .get(index as usize)
        .ok_or_else(|| Error::internal(format!("{description} names an undefined scalar cell")))?;
    match expected {
        ExpectedBound::Null if cell.tag == ResidentScalarCellTag::Null => Ok(()),
        ExpectedBound::Integer(value)
            if cell.tag == ResidentScalarCellTag::Integer && cell.payload as i64 == value =>
        {
            Ok(())
        }
        _ => Err(Error::internal(format!(
            "{description} has {cell:?}, expected {expected:?}",
        ))),
    }
}

fn validate_native_program(request: &ResidentScalarProgramRequest, case: SliceCase) -> Result<()> {
    let [instruction] = request.instructions.as_slice() else {
        return Err(Error::internal(format!(
            "{} did not lower to exactly one native scalar instruction: {:?}",
            case.name, request.instructions,
        )));
    };
    if instruction.opcode != ResidentScalarProgramOpcode::ListSlice {
        return Err(Error::internal(format!(
            "{} lowered to {:?} instead of ListSlice",
            case.name, instruction.opcode,
        )));
    }
    let ResidentScalarProgramOperand::Cell(source) = instruction.left else {
        return Err(Error::internal(format!(
            "{} did not carry its source list in the native input image",
            case.name,
        )));
    };
    let source = request
        .scalar_cells
        .get(source as usize)
        .ok_or_else(|| Error::internal("native list-slice source cell is undefined"))?;
    if source.tag != ResidentScalarCellTag::List || source.auxiliary != 3 {
        return Err(Error::internal(format!(
            "{} has a malformed source list cell: {source:?}",
            case.name,
        )));
    }
    validate_bound(request, instruction.right, case.start, "list-slice start")?;
    validate_bound(
        request,
        instruction
            .third
            .ok_or_else(|| Error::internal("native list slice omitted its end operand"))?,
        case.end,
        "list-slice end",
    )?;
    let ResidentScalarProgramOperand::Cell(arena) = instruction
        .fourth
        .ok_or_else(|| Error::internal("native list slice omitted its output arena"))?
    else {
        return Err(Error::internal(
            "native list-slice output arena was not an immutable descriptor",
        ));
    };
    let arena = request
        .scalar_cells
        .get(arena as usize)
        .ok_or_else(|| Error::internal("native list-slice output arena is undefined"))?;
    if arena.tag != ResidentScalarCellTag::Integer || arena.payload as u32 != 3 {
        return Err(Error::internal(format!(
            "{} did not reserve the device output arena for all three source entries: {arena:?}",
            case.name,
        )));
    }
    if request.output_values != [ResidentScalarProgramOperand::Register(0)] {
        return Err(Error::internal(format!(
            "{} published compiler-owned input instead of the ListSlice device register",
            case.name,
        )));
    }
    Ok(())
}

fn run_all(backend: &ObservedScalarBackend) -> Result<()> {
    let observations = backend.observations();
    let mut defects = Vec::new();
    for case in CASES {
        let calls_before = observations.calls.load(Ordering::SeqCst);
        let requests_before = observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len();
        let outcome = execute_single_value(backend, case);
        let calls = observations
            .calls
            .load(Ordering::SeqCst)
            .saturating_sub(calls_before);
        if calls != 1 {
            defects.push(format!(
                "report {} {} dispatched {calls} native scalar programs instead of one; outcome={outcome:?}",
                case.report_id, case.name,
            ));
            continue;
        }
        match outcome {
            Ok(actual) if actual == expected_value(case.expected) => {}
            Ok(actual) => defects.push(format!(
                "report {} {} returned {actual:?}, expected {:?}",
                case.report_id,
                case.name,
                expected_value(case.expected),
            )),
            Err(error) => defects.push(format!(
                "report {} {} failed native execution: {error:?}",
                case.report_id, case.name,
            )),
        }
        let requests = observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if requests.len() != requests_before + 1 {
            defects.push(format!(
                "report {} {} did not publish exactly one inspectable native request",
                case.report_id, case.name,
            ));
        } else if let Err(error) = validate_native_program(&requests[requests_before], case) {
            defects.push(error.to_string());
        }
    }
    if defects.is_empty() {
        Ok(())
    } else {
        Err(Error::internal(format!(
            "native omitted-bound list-slice defects:\n{}",
            defects.join("\n"),
        )))
    }
}

#[test]
fn manifest_is_exactly_the_four_previously_failing_list2_report_cases() {
    assert_eq!(CASES.map(|case| case.report_id), [1690, 1691, 1700, 1701]);
}

#[test]
#[ignore = "external assurance gate: requires the certified full TCK report"]
fn certified_report_uniquely_resolves_every_list_slice_selector() {
    assert_certified_report_identities(
        CASES
            .iter()
            .map(|case| (case.report_id, FEATURE, case.name)),
    );
}

#[test]
fn strict_native_cpu_reference_passes_all_four_and_exposes_exact_bound_operands() -> Result<()> {
    let backend = ObservedScalarBackend::strict_cpu_reference();
    if backend.actual_kind != BackendKind::Cpu {
        return Err(Error::internal(
            "list-slice CPU gate did not use the CPU semantic reference",
        ));
    }
    run_all(&backend)?;
    if backend.observations().calls.load(Ordering::SeqCst) != CASES.len() {
        return Err(Error::internal(
            "CPU list-slice gate did not dispatch exactly once per scenario",
        ));
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "hardware acceptance gate: requires a real Metal device"]
fn real_metal_matches_cpu_for_all_four_without_fallback() -> Result<()> {
    static METAL_TEST: Mutex<()> = Mutex::new(());
    let _guard = METAL_TEST
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let cpu = ObservedScalarBackend::strict_cpu_reference();
    let metal = ObservedScalarBackend::real_metal()?;
    if cpu.actual_kind != BackendKind::Cpu || metal.actual_kind != BackendKind::Metal {
        return Err(Error::internal(
            "list-slice hardware gate did not construct distinct CPU and Metal backends",
        ));
    }
    run_all(&cpu)?;
    run_all(&metal)?;
    if cpu.observations().calls.load(Ordering::SeqCst) != CASES.len()
        || metal.observations().calls.load(Ordering::SeqCst) != CASES.len()
    {
        return Err(Error::internal(
            "CPU or Metal did not execute one native ListSlice program per scenario",
        ));
    }
    Ok(())
}
