// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Strict native-route acceptance for the remaining scalar precedence/conversion tranche.
//!
//! The six expanded Precedence3 scenario-[6] examples are pinned by their one-based expanded
//! IDs (2163-2168) and zero-based report indices (2162-2167). TypeConversion4 scenario [3] is
//! pinned separately at report index 3855. The CPU reference is deliberately presented to the
//! query engine as Metal, so `require_native_execution` turns any generic-host fallback into a
//! closed admission failure while the wrapper records the one canonical scalar request.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use irongraph::{
    Bookmark, Error, ProjectId, Result, ScalarValue,
    cypher::{
        BindCapabilities, ColumnType, ExecutionContext, QueryEngine, ResultValue, StatementStats,
    },
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

const FRESH_TCK_REPORT: &str = "/tmp/irongraph-tck-full-next.json";
const PRECEDENCE3_FEATURE: &str = "features/expressions/precedence/Precedence3.feature";
const PRECEDENCE3_SCENARIO: &str =
    "[6] List element containment takes precedence over comparison operator";
const TYPE_CONVERSION4_FEATURE: &str =
    "features/expressions/typeConversion/TypeConversion4.feature";
const TYPE_CONVERSION4_SCENARIO: &str = "[3] `toString()` handling inlined boolean";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExpectedScalar {
    Null,
    Boolean(bool),
    String(&'static str),
}

impl ExpectedScalar {
    const fn column_type(self) -> ColumnType {
        match self {
            Self::Null => ColumnType::Null,
            Self::Boolean(_) => ColumnType::Boolean,
            Self::String(_) => ColumnType::String,
        }
    }

    fn value(self) -> ResultValue {
        ResultValue::Scalar(match self {
            Self::Null => ScalarValue::Null,
            Self::Boolean(value) => ScalarValue::Boolean(value),
            Self::String(value) => ScalarValue::String(Arc::from(value)),
        })
    }
}

const FALSE_FALSE_TRUE: &[ExpectedScalar] = &[
    ExpectedScalar::Boolean(false),
    ExpectedScalar::Boolean(false),
    ExpectedScalar::Boolean(true),
];
const TRUE_TRUE_FALSE: &[ExpectedScalar] = &[
    ExpectedScalar::Boolean(true),
    ExpectedScalar::Boolean(true),
    ExpectedScalar::Boolean(false),
];
const NULL_NULL_FALSE: &[ExpectedScalar] = &[
    ExpectedScalar::Null,
    ExpectedScalar::Null,
    ExpectedScalar::Boolean(false),
];
const NULL_NULL_TRUE: &[ExpectedScalar] = &[
    ExpectedScalar::Null,
    ExpectedScalar::Null,
    ExpectedScalar::Boolean(true),
];
const STRING_FALSE: &[ExpectedScalar] = &[ExpectedScalar::String("false")];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScenarioKind {
    Precedence {
        expanded_id: u16,
        comparison: ResidentScalarProgramOpcode,
    },
    TypeConversion,
}

#[derive(Clone, Copy, Debug)]
struct Scenario {
    report_index: usize,
    feature: &'static str,
    report_name: &'static str,
    query: &'static str,
    expected: &'static [ExpectedScalar],
    kind: ScenarioKind,
}

impl Scenario {
    fn label(self) -> String {
        match self.kind {
            ScenarioKind::Precedence { expanded_id, .. } => format!(
                "Precedence3 [6] expanded ID {expanded_id} (report index {})",
                self.report_index,
            ),
            ScenarioKind::TypeConversion => {
                format!("TypeConversion4 [3] (report index {})", self.report_index,)
            }
        }
    }

    fn expected_schema(self) -> Vec<(String, ColumnType)> {
        let names: &[&str] = match self.kind {
            ScenarioKind::Precedence { .. } => &["a", "b", "c"],
            ScenarioKind::TypeConversion => &["bool"],
        };
        names
            .iter()
            .zip(self.expected)
            .map(|(name, value)| ((*name).to_owned(), value.column_type()))
            .collect()
    }

    fn expected_row(self) -> Vec<ResultValue> {
        self.expected
            .iter()
            .copied()
            .map(ExpectedScalar::value)
            .collect()
    }
}

const SCENARIOS: [Scenario; 7] = [
    Scenario {
        report_index: 2162,
        feature: PRECEDENCE3_FEATURE,
        report_name: "[6] List element containment takes precedence over comparison operator [2163]",
        query: "RETURN [1, 2] = [3, 4] IN [[3, 4], false] AS a, \
                       [1, 2] = ([3, 4] IN [[3, 4], false]) AS b, \
                       ([1, 2] = [3, 4]) IN [[3, 4], false] AS c",
        expected: FALSE_FALSE_TRUE,
        kind: ScenarioKind::Precedence {
            expanded_id: 2163,
            comparison: ResidentScalarProgramOpcode::Equal,
        },
    },
    Scenario {
        report_index: 2163,
        feature: PRECEDENCE3_FEATURE,
        report_name: "[6] List element containment takes precedence over comparison operator [2164]",
        query: "RETURN [1, 2] <> [3, 4] IN [[3, 4], false] AS a, \
                       [1, 2] <> ([3, 4] IN [[3, 4], false]) AS b, \
                       ([1, 2] <> [3, 4]) IN [[3, 4], false] AS c",
        expected: TRUE_TRUE_FALSE,
        kind: ScenarioKind::Precedence {
            expanded_id: 2164,
            comparison: ResidentScalarProgramOpcode::NotEqual,
        },
    },
    Scenario {
        report_index: 2164,
        feature: PRECEDENCE3_FEATURE,
        report_name: "[6] List element containment takes precedence over comparison operator [2165]",
        query: "RETURN [1, 2] < [3, 4] IN [[3, 4], false] AS a, \
                       [1, 2] < ([3, 4] IN [[3, 4], false]) AS b, \
                       ([1, 2] < [3, 4]) IN [[3, 4], false] AS c",
        expected: NULL_NULL_FALSE,
        kind: ScenarioKind::Precedence {
            expanded_id: 2165,
            comparison: ResidentScalarProgramOpcode::Less,
        },
    },
    Scenario {
        report_index: 2165,
        feature: PRECEDENCE3_FEATURE,
        report_name: "[6] List element containment takes precedence over comparison operator [2166]",
        query: "RETURN [1, 2] > [3, 4] IN [[3, 4], false] AS a, \
                       [1, 2] > ([3, 4] IN [[3, 4], false]) AS b, \
                       ([1, 2] > [3, 4]) IN [[3, 4], false] AS c",
        expected: NULL_NULL_TRUE,
        kind: ScenarioKind::Precedence {
            expanded_id: 2166,
            comparison: ResidentScalarProgramOpcode::Greater,
        },
    },
    Scenario {
        report_index: 2166,
        feature: PRECEDENCE3_FEATURE,
        report_name: "[6] List element containment takes precedence over comparison operator [2167]",
        query: "RETURN [1, 2] <= [3, 4] IN [[3, 4], false] AS a, \
                       [1, 2] <= ([3, 4] IN [[3, 4], false]) AS b, \
                       ([1, 2] <= [3, 4]) IN [[3, 4], false] AS c",
        expected: NULL_NULL_FALSE,
        kind: ScenarioKind::Precedence {
            expanded_id: 2167,
            comparison: ResidentScalarProgramOpcode::LessOrEqual,
        },
    },
    Scenario {
        report_index: 2167,
        feature: PRECEDENCE3_FEATURE,
        report_name: "[6] List element containment takes precedence over comparison operator [2168]",
        query: "RETURN [1, 2] >= [3, 4] IN [[3, 4], false] AS a, \
                       [1, 2] >= ([3, 4] IN [[3, 4], false]) AS b, \
                       ([1, 2] >= [3, 4]) IN [[3, 4], false] AS c",
        expected: NULL_NULL_TRUE,
        kind: ScenarioKind::Precedence {
            expanded_id: 2168,
            comparison: ResidentScalarProgramOpcode::GreaterOrEqual,
        },
    },
    Scenario {
        report_index: 3855,
        feature: TYPE_CONVERSION4_FEATURE,
        report_name: TYPE_CONVERSION4_SCENARIO,
        query: "RETURN toString(1 < 0) AS bool",
        expected: STRING_FALSE,
        kind: ScenarioKind::TypeConversion,
    },
];

#[derive(Default)]
struct ScalarObservations {
    calls: AtomicUsize,
    requests: Mutex<Vec<ResidentScalarProgramRequest>>,
}

/// Reports accelerator identity to the query engine while retaining the actual backend kind for
/// the test oracle. Strict execution therefore rejects every generic expression fallback.
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
                "scalar precedence gate did not construct a real Metal backend",
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

#[derive(Clone, Debug, PartialEq)]
struct ObservedResult {
    schema: Vec<(String, ColumnType)>,
    rows: Vec<Vec<ResultValue>>,
}

fn execute(backend: &dyn ExecutionBackend, scenario: Scenario) -> Result<ObservedResult> {
    let graph = GraphStore::default();
    let output = QueryEngine.execute(scenario.query, &mut context(&graph, backend))?;
    if !output.graph_mutations.is_empty() || !output.temporal_mutations.is_empty() {
        return Err(Error::internal(format!(
            "{} unexpectedly produced side effects",
            scenario.label(),
        )));
    }
    if output.result.statistics != StatementStats::default() || output.result.truncated {
        return Err(Error::internal(format!(
            "{} returned non-default read metadata: {:?}",
            scenario.label(),
            output.result,
        )));
    }
    let [batch] = output.result.batches.as_slice() else {
        return Err(Error::internal(format!(
            "{} did not return exactly one batch",
            scenario.label(),
        )));
    };
    if !batch.validate()
        || batch.row_count != 1
        || batch.columns.len() != output.result.schema.len()
    {
        return Err(Error::internal(format!(
            "{} did not return one valid row: {:?}",
            scenario.label(),
            batch,
        )));
    }
    let batch_schema = batch
        .columns
        .iter()
        .map(|column| (column.name.clone(), column.value_type.clone()))
        .collect::<Vec<_>>();
    if batch_schema != output.result.schema {
        return Err(Error::internal(format!(
            "{} batch/schema metadata diverged: batch={batch_schema:?}, schema={:?}",
            scenario.label(),
            output.result.schema,
        )));
    }
    let row = batch
        .columns
        .iter()
        .map(|column| column.values[0].clone())
        .collect::<Vec<_>>();
    Ok(ObservedResult {
        schema: output.result.schema,
        rows: vec![row],
    })
}

fn expected_result(scenario: Scenario) -> ObservedResult {
    ObservedResult {
        schema: scenario.expected_schema(),
        rows: vec![scenario.expected_row()],
    }
}

fn assert_native_program(request: &ResidentScalarProgramRequest, scenario: Scenario) -> Result<()> {
    let opcodes = request
        .instructions
        .iter()
        .map(|instruction| instruction.opcode)
        .collect::<Vec<_>>();
    match scenario.kind {
        ScenarioKind::Precedence { comparison, .. } => {
            let expected_opcodes = vec![
                ResidentScalarProgramOpcode::ListMembership,
                comparison,
                ResidentScalarProgramOpcode::ListMembership,
                comparison,
                comparison,
                ResidentScalarProgramOpcode::ListMembership,
            ];
            if opcodes != expected_opcodes {
                return Err(Error::internal(format!(
                    "{} did not preserve the exact membership/comparison precedence program: {opcodes:?}",
                    scenario.label(),
                )));
            }
            let expected_outputs = [
                ResidentScalarProgramOperand::Register(1),
                ResidentScalarProgramOperand::Register(3),
                ResidentScalarProgramOperand::Register(5),
            ];
            if request.output_values.as_slice() != expected_outputs {
                return Err(Error::internal(format!(
                    "{} did not publish the three device comparison roots: {:?}",
                    scenario.label(),
                    request.output_values,
                )));
            }
            let instructions = &request.instructions;
            let exact_operands = matches!(
                (instructions[0].left, instructions[0].right),
                (
                    ResidentScalarProgramOperand::Cell(_),
                    ResidentScalarProgramOperand::Cell(_)
                )
            ) && matches!(
                (instructions[1].left, instructions[1].right),
                (
                    ResidentScalarProgramOperand::Cell(_),
                    ResidentScalarProgramOperand::Register(0)
                )
            ) && matches!(
                (instructions[2].left, instructions[2].right),
                (
                    ResidentScalarProgramOperand::Cell(_),
                    ResidentScalarProgramOperand::Cell(_)
                )
            ) && matches!(
                (instructions[3].left, instructions[3].right),
                (
                    ResidentScalarProgramOperand::Cell(_),
                    ResidentScalarProgramOperand::Register(2)
                )
            ) && matches!(
                (instructions[4].left, instructions[4].right),
                (
                    ResidentScalarProgramOperand::Cell(_),
                    ResidentScalarProgramOperand::Cell(_)
                )
            ) && matches!(
                (instructions[5].left, instructions[5].right),
                (
                    ResidentScalarProgramOperand::Register(4),
                    ResidentScalarProgramOperand::Cell(_)
                )
            );
            if !exact_operands {
                return Err(Error::internal(format!(
                    "{} changed the parenthesized/unparenthesized SSA dependencies: {:?}",
                    scenario.label(),
                    request.instructions,
                )));
            }
        }
        ScenarioKind::TypeConversion => {
            if opcodes
                != [
                    ResidentScalarProgramOpcode::Less,
                    ResidentScalarProgramOpcode::ToString,
                ]
            {
                return Err(Error::internal(format!(
                    "{} did not execute comparison then conversion on the backend: {opcodes:?}",
                    scenario.label(),
                )));
            }
            if request.output_values.as_slice() != [ResidentScalarProgramOperand::Register(1)]
                || !matches!(
                    (request.instructions[0].left, request.instructions[0].right,),
                    (
                        ResidentScalarProgramOperand::Cell(_),
                        ResidentScalarProgramOperand::Cell(_)
                    )
                )
                || !matches!(
                    request.instructions[1].left,
                    ResidentScalarProgramOperand::Register(0)
                )
            {
                return Err(Error::internal(format!(
                    "{} did not feed the device comparison register directly into toString: {:?}",
                    scenario.label(),
                    request,
                )));
            }
        }
    }
    if request.false_cell.is_none() || request.true_cell.is_none() {
        return Err(Error::internal(format!(
            "{} omitted canonical Boolean result cells",
            scenario.label(),
        )));
    }
    Ok(())
}

fn run_manifest(backend: &ObservedScalarBackend) -> std::result::Result<(), Vec<String>> {
    let observations = backend.observations();
    let mut defects = Vec::new();
    for scenario in SCENARIOS {
        let calls_before = observations.calls.load(Ordering::SeqCst);
        let requests_before = observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .len();
        let outcome = execute(backend, scenario);
        let calls = observations
            .calls
            .load(Ordering::SeqCst)
            .saturating_sub(calls_before);
        let requests = observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let request_count = requests.len().saturating_sub(requests_before);
        let request = (request_count == 1)
            .then(|| requests.get(requests_before).cloned())
            .flatten();
        drop(requests);

        if calls != 1 || request_count != 1 {
            defects.push(format!(
                "{} crossed {calls} scalar calls and recorded {request_count} requests; expected exactly one; outcome={outcome:?}",
                scenario.label(),
            ));
            continue;
        }
        match outcome {
            Ok(actual) if actual == expected_result(scenario) => {}
            Ok(actual) => defects.push(format!(
                "{} returned {actual:?}, expected {:?}",
                scenario.label(),
                expected_result(scenario),
            )),
            Err(error) => defects.push(format!("{} failed: {error:?}", scenario.label())),
        }
        match request {
            Some(request) => {
                if let Err(error) = assert_native_program(&request, scenario) {
                    defects.push(error.to_string());
                }
            }
            None => defects.push(format!(
                "{} did not retain its native scalar request",
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
fn manifest_is_exactly_six_precedence_examples_plus_one_conversion_case() {
    assert_eq!(SCENARIOS.len(), 7, "scalar tranche cardinality changed");
    assert_eq!(
        SCENARIOS[..6]
            .iter()
            .map(|scenario| match scenario.kind {
                ScenarioKind::Precedence { expanded_id, .. } => {
                    (scenario.report_index, expanded_id)
                }
                ScenarioKind::TypeConversion => unreachable!("first six cases are pinned"),
            })
            .collect::<Vec<_>>(),
        vec![
            (2162, 2163),
            (2163, 2164),
            (2164, 2165),
            (2165, 2166),
            (2166, 2167),
            (2167, 2168),
        ],
    );
    assert!(SCENARIOS[..6].iter().all(|scenario| {
        scenario.feature == PRECEDENCE3_FEATURE
            && scenario.report_name.starts_with(PRECEDENCE3_SCENARIO)
            && scenario.expected.len() == 3
    }));
    let conversion = SCENARIOS[6];
    assert_eq!(conversion.report_index, 3855);
    assert_eq!(conversion.feature, TYPE_CONVERSION4_FEATURE);
    assert_eq!(conversion.report_name, TYPE_CONVERSION4_SCENARIO);
    assert_eq!(conversion.query, "RETURN toString(1 < 0) AS bool");
    assert_eq!(conversion.expected, STRING_FALSE);

    let report_indices = SCENARIOS
        .iter()
        .map(|scenario| scenario.report_index)
        .collect::<BTreeSet<_>>();
    let report_names = SCENARIOS
        .iter()
        .map(|scenario| (scenario.feature, scenario.report_name))
        .collect::<BTreeSet<_>>();
    assert_eq!(report_indices.len(), SCENARIOS.len());
    assert_eq!(report_names.len(), SCENARIOS.len());
}

#[test]
#[ignore = "external assurance gate: requires /tmp/irongraph-tck-full-next.json"]
fn fresh_report_uniquely_resolves_the_exact_seven_case_manifest() {
    let report: serde_json::Value = serde_json::from_slice(
        &std::fs::read(FRESH_TCK_REPORT).expect("fresh TCK report is readable"),
    )
    .expect("fresh TCK report is valid JSON");
    assert_eq!(report["total"].as_u64(), Some(3_897));
    let scenarios = report["scenarios"]
        .as_array()
        .expect("fresh report has a scenario array");
    assert_eq!(scenarios.len(), 3_897);

    for expected in SCENARIOS {
        let matches = scenarios
            .iter()
            .enumerate()
            .filter_map(|(index, scenario)| {
                let path = scenario.get("path")?.as_str()?;
                let name = scenario.get("name")?.as_str()?;
                (path.ends_with(expected.feature) && name == expected.report_name).then_some(index)
            })
            .collect::<Vec<_>>();
        assert_eq!(
            matches,
            vec![expected.report_index],
            "{} did not resolve uniquely",
            expected.label(),
        );
        let scenario = &scenarios[expected.report_index];
        assert_eq!(scenario["cpu_passed"].as_bool(), Some(true));
        assert_eq!(scenario["metal_passed"].as_bool(), Some(false));
    }
}

#[test]
fn strict_cpu_reference_passes_all_seven_without_generic_fallback() {
    let backend = ObservedScalarBackend::strict_cpu_reference();
    assert_eq!(backend.actual_kind, BackendKind::Cpu);
    assert_eq!(backend.kind(), BackendKind::Metal);
    if let Err(defects) = run_manifest(&backend) {
        panic!(
            "strict native CPU scalar precedence/conversion defects:\n{}",
            defects.join("\n"),
        );
    }
    assert_eq!(
        backend.observations().calls.load(Ordering::SeqCst),
        SCENARIOS.len(),
    );
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "hardware acceptance gate: requires a real Metal device"]
fn real_metal_passes_all_seven_with_one_scalar_dispatch_each() -> Result<()> {
    static METAL_TEST: Mutex<()> = Mutex::new(());
    let _guard = METAL_TEST
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let backend = ObservedScalarBackend::real_metal()?;
    if backend.actual_kind != BackendKind::Metal || backend.kind() != BackendKind::Metal {
        return Err(Error::internal(
            "scalar precedence hardware gate did not use Metal",
        ));
    }
    if let Err(defects) = run_manifest(&backend) {
        return Err(Error::internal(format!(
            "real Metal scalar precedence/conversion defects:\n{}",
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
