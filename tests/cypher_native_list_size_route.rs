// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Exact manifest and focused native-route assertions for the List6 gaps.
//!
//! The manifest pins exact feature, scenario, and primary-query identity from the fresh full
//! conformance report. Tranches describe implementation ownership; they are not substitutes for
//! native CPU/Metal semantic acceptance tests in the eventual implementation lanes. The literal
//! `size()` tranche additionally proves one complete quantifier command on the CPU semantic
//! reference and keeps non-`size()` scalar, list, comparison, and comprehension projections
//! outside that route.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use irongraph::{
    Bookmark, EdgeId, Error, ErrorCode, Layer, NodeId, ProjectId, Result, ScalarValue,
    cypher::{BindCapabilities, ExecutionContext, QueryEngine, ResultValue},
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend, ResidentDirection,
        ResidentExecutionId, ResidentExecutionObligation, ResidentGroup, ResidentGroupRequest,
        ResidentJoinPair, ResidentJoinRequest, ResidentNodePipelineRequest,
        ResidentNodePipelineResult, ResidentObligationKind, ResidentObligationScope,
        ResidentPatternCountRequest, ResidentPatternCountResult, ResidentPatternCountStartLabels,
        ResidentPatternRelationshipTypes, ResidentProjectImage, ResidentQuantifierBinary,
        ResidentQuantifierExpression, ResidentQuantifierFunction, ResidentQuantifierProgramRequest,
        ResidentQuantifierProgramResult, ResidentQuantifierSource, ResidentQuantifierStage,
        ResidentQuantifierValue, ResidentSortRequest, ResidentSortResult, ResidentVectorQuery,
        ResidentVectorResult, ScratchReservation,
    },
    graph::{EdgeInput, GraphStore, LayerMask, NodeInput},
    types::{LabelId, PropertyId},
};

use serde::Deserialize;
use tokio_util::sync::CancellationToken;

const REPORT: &str = "/tmp/irongraph-tck-full-after-regression-fix.json";
const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Tranche {
    LiteralSize,
    MutationListSize,
    PatternCount,
}

#[derive(Clone, Copy, Debug)]
struct Case {
    report_index: usize,
    feature: &'static str,
    scenario: u8,
    name: &'static str,
    query: &'static str,
    tranche: Tranche,
}

const CASES: [Case; 8] = [
    Case {
        report_index: 1759,
        feature: "expressions/list/List6.feature",
        scenario: 1,
        name: "[1] Return list size",
        query: "RETURN size([1, 2, 3]) AS n",
        tranche: Tranche::LiteralSize,
    },
    Case {
        report_index: 1760,
        feature: "expressions/list/List6.feature",
        scenario: 2,
        name: "[2] Setting and returning the size of a list property",
        query: "MATCH (n:TheLabel) SET n.numbers = [1, 2, 3] RETURN size(n.numbers)",
        tranche: Tranche::MutationListSize,
    },
    Case {
        report_index: 1761,
        feature: "expressions/list/List6.feature",
        scenario: 3,
        name: "[3] Concatenating and returning the size of literal lists",
        query: "RETURN size([[], []] + [[]]) AS l",
        tranche: Tranche::LiteralSize,
    },
    Case {
        report_index: 1762,
        feature: "expressions/list/List6.feature",
        scenario: 4,
        name: "[4] `size()` on null list",
        query: "WITH null AS l RETURN size(l), size(null)",
        tranche: Tranche::LiteralSize,
    },
    Case {
        report_index: 1772,
        feature: "expressions/list/List6.feature",
        scenario: 7,
        name: "[7] Using size of pattern comprehension to test existence",
        query: "MATCH (n:X) RETURN n, size([(n)--() | 1]) > 0 AS b",
        tranche: Tranche::PatternCount,
    },
    Case {
        report_index: 1773,
        feature: "expressions/list/List6.feature",
        scenario: 8,
        name: "[8] Get node degree via size of pattern comprehension",
        query: "MATCH (a:X) RETURN size([(a)-->() | 1]) AS length",
        tranche: Tranche::PatternCount,
    },
    Case {
        report_index: 1774,
        feature: "expressions/list/List6.feature",
        scenario: 9,
        name: "[9] Get node degree via size of pattern comprehension that specifies a relationship type",
        query: "MATCH (a:X) RETURN size([(a)-[:T]->() | 1]) AS length",
        tranche: Tranche::PatternCount,
    },
    Case {
        report_index: 1775,
        feature: "expressions/list/List6.feature",
        scenario: 10,
        name: "[10] Get node degree via size of pattern comprehension that specifies multiple relationship types",
        query: "MATCH (a:X) RETURN size([(a)-[:T|OTHER]->() | 1]) AS length",
        tranche: Tranche::PatternCount,
    },
];

#[derive(Debug, Deserialize)]
struct Report {
    scenarios: Vec<ReportScenario>,
}

#[derive(Debug, Deserialize)]
struct ReportScenario {
    path: String,
    name: String,
    cpu_passed: bool,
    metal_passed: bool,
    metal_failures: Vec<String>,
}

#[derive(Default)]
struct QuantifierObservations {
    calls: AtomicUsize,
    requests: Mutex<Vec<ResidentQuantifierProgramRequest>>,
    pattern_count_calls: AtomicUsize,
    pattern_count_requests: Mutex<Vec<ResidentPatternCountRequest>>,
}

/// Executes the real CPU semantic reference while recording the sealed quantifier boundary. A
/// successful generic evaluator fallback cannot satisfy the route assertions because it never
/// increments this backend-owned counter or publishes a quantifier request.
struct ObservedQuantifierBackend {
    inner: Box<dyn ExecutionBackend>,
    reported_kind: BackendKind,
    observations: Arc<QuantifierObservations>,
}

impl ObservedQuantifierBackend {
    fn new() -> Self {
        Self::reporting(BackendKind::Cpu)
    }

    fn reporting(reported_kind: BackendKind) -> Self {
        Self {
            inner: Box::new(CpuBackend::new(64 * 1024 * 1024, 16 * 1024 * 1024)),
            reported_kind,
            observations: Arc::new(QuantifierObservations::default()),
        }
    }

    fn observations(&self) -> Arc<QuantifierObservations> {
        Arc::clone(&self.observations)
    }
}

impl ExecutionBackend for ObservedQuantifierBackend {
    fn kind(&self) -> BackendKind {
        self.reported_kind
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
            reported_kind: self.reported_kind,
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

    fn execute_pattern_count(
        &self,
        request: &ResidentPatternCountRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentPatternCountResult> {
        self.observations
            .pattern_count_calls
            .fetch_add(1, Ordering::SeqCst);
        self.observations
            .pattern_count_requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(request.clone());
        self.inner.execute_pattern_count(request, cancellation)
    }

    fn supports_native_quantifier_program(&self) -> bool {
        self.inner.supports_native_quantifier_program()
    }

    fn execute_quantifier_program(
        &self,
        request: &ResidentQuantifierProgramRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentQuantifierProgramResult> {
        self.observations.calls.fetch_add(1, Ordering::SeqCst);
        self.observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(request.clone());
        self.inner.execute_quantifier_program(request, cancellation)
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

fn execution_context<'a>(
    graph: &'a GraphStore,
    backend: &'a dyn ExecutionBackend,
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
        deadline: Some(Instant::now() + Duration::from_secs(60)),
        resolved_query_at_time_nanos: None,
    }
}

struct PatternCountFixture {
    graph: GraphStore,
    x: LabelId,
    t: irongraph::types::RelationshipTypeId,
    other: irongraph::types::RelationshipTypeId,
}

fn pattern_count_fixture() -> Result<PatternCountFixture> {
    let mut graph = GraphStore::default();
    let x = graph.catalog_mut().intern_label("X")?;
    let y = graph.catalog_mut().intern_label("Y")?;
    let t = graph.catalog_mut().intern_relationship_type("T")?;
    let other = graph.catalog_mut().intern_relationship_type("OTHER")?;
    let unrelated = graph.catalog_mut().intern_relationship_type("U")?;
    for id in 1..=5 {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: id,
            labels: vec![x],
            properties: Vec::new(),
        })?;
    }
    graph.insert_node(NodeInput {
        id: NodeId(6),
        layer: Layer::Observed,
        revision: 6,
        labels: vec![y],
        properties: Vec::new(),
    })?;
    for (id, source, target, relationship_type, layer) in [
        (10, 1, 2, t, Layer::Observed),
        (11, 1, 3, other, Layer::Observed),
        (12, 2, 1, unrelated, Layer::Observed),
        // The same self-loop is present in both CSR orientations but counts once when undirected.
        (13, 2, 2, t, Layer::Observed),
        // Anonymous endpoints are not constrained by the outer `X` label.
        (14, 6, 1, t, Layer::Observed),
        // Direct backend tests restrict visibility to OBSERVED and must exclude this edge.
        (15, 3, 6, t, Layer::Knowledge),
    ] {
        graph.insert_edge(EdgeInput {
            id: EdgeId(id),
            source: NodeId(source),
            target: NodeId(target),
            relationship_type,
            layer,
            revision: id,
            properties: Vec::new(),
        })?;
    }
    Ok(PatternCountFixture { graph, x, t, other })
}

fn install_pattern_fixture(
    backend: &mut ObservedQuantifierBackend,
    fixture: &PatternCountFixture,
) -> Result<()> {
    backend.replace_all_projects(vec![ResidentProjectImage::graph_only(Arc::new(
        fixture.graph.snapshot()?,
    ))])
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
                    .collect::<Vec<_>>()
            })
        })
        .collect()
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct NativeListFeatures {
    size: bool,
    list: bool,
    list_add: bool,
}

fn observe_list_features(
    expression: &ResidentQuantifierExpression,
    features: &mut NativeListFeatures,
) {
    match expression {
        ResidentQuantifierExpression::Slot(_) | ResidentQuantifierExpression::Literal(_) => {}
        ResidentQuantifierExpression::Property { source, .. }
        | ResidentQuantifierExpression::Unary {
            operand: source, ..
        }
        | ResidentQuantifierExpression::IsNull {
            expression: source, ..
        } => observe_list_features(source, features),
        ResidentQuantifierExpression::List(values) => {
            features.list = true;
            for value in values {
                observe_list_features(value, features);
            }
        }
        ResidentQuantifierExpression::Map(entries) => {
            for (_, value) in entries {
                observe_list_features(value, features);
            }
        }
        ResidentQuantifierExpression::Case {
            operand,
            alternatives,
            default,
        } => {
            if let Some(operand) = operand {
                observe_list_features(operand, features);
            }
            for (when, then) in alternatives {
                observe_list_features(when, features);
                observe_list_features(then, features);
            }
            if let Some(default) = default {
                observe_list_features(default, features);
            }
        }
        ResidentQuantifierExpression::ListComprehension {
            list,
            predicate,
            projection,
            ..
        } => {
            features.list = true;
            observe_list_features(list, features);
            if let Some(predicate) = predicate {
                observe_list_features(predicate, features);
            }
            if let Some(projection) = projection {
                observe_list_features(projection, features);
            }
        }
        ResidentQuantifierExpression::Predicate {
            list, predicate, ..
        } => {
            observe_list_features(list, features);
            observe_list_features(predicate, features);
        }
        ResidentQuantifierExpression::Function {
            function,
            arguments,
        } => {
            features.size |= *function == ResidentQuantifierFunction::Size;
            for argument in arguments {
                observe_list_features(argument, features);
            }
        }
        ResidentQuantifierExpression::Binary {
            left,
            operation,
            right,
        } => {
            features.list_add |= *operation == ResidentQuantifierBinary::Add;
            observe_list_features(left, features);
            observe_list_features(right, features);
        }
    }
}

fn request_list_features(request: &ResidentQuantifierProgramRequest) -> NativeListFeatures {
    let mut features = NativeListFeatures::default();
    for stage in &request.program.stages {
        match stage {
            ResidentQuantifierStage::Project { bindings, .. }
            | ResidentQuantifierStage::GroupCount {
                groups: bindings, ..
            } => {
                for binding in bindings {
                    observe_list_features(&binding.expression, &mut features);
                }
            }
            ResidentQuantifierStage::Unwind { expression, .. } => {
                observe_list_features(expression, &mut features);
            }
            ResidentQuantifierStage::Filter { predicate } => {
                observe_list_features(predicate, &mut features);
            }
        }
    }
    features
}

#[test]
#[ignore = "external assurance gate: requires the fresh full TCK report"]
fn exact_eight_case_manifest_matches_the_fresh_full_report() {
    let report: Report = serde_json::from_slice(
        &fs::read(REPORT).unwrap_or_else(|error| panic!("failed to read {REPORT}: {error}")),
    )
    .unwrap_or_else(|error| panic!("failed to decode {REPORT}: {error}"));
    let mut identities = BTreeSet::new();
    for case in CASES {
        assert!(
            identities.insert((case.feature, case.scenario)),
            "duplicate manifest identity: {} [{}]",
            case.feature,
            case.scenario
        );
        let actual = report
            .scenarios
            .get(case.report_index)
            .unwrap_or_else(|| panic!("report index {} disappeared", case.report_index));
        assert!(
            actual.path.ends_with(case.feature),
            "wrong feature at {}",
            case.report_index
        );
        assert_eq!(
            actual.name, case.name,
            "wrong scenario at {}",
            case.report_index
        );
        assert!(
            actual.cpu_passed,
            "CPU baseline regressed for {}",
            case.name
        );
        assert!(
            !actual.metal_passed,
            "baseline unexpectedly moved for {}",
            case.name
        );
        assert!(
            actual
                .metal_failures
                .iter()
                .any(|failure| failure.contains(&format!("`{}`", case.query))),
            "exact primary query disappeared for {}",
            case.name
        );
    }
}

#[test]
fn tranche_partition_is_complete_and_non_overlapping() {
    let expected = [
        (Tranche::LiteralSize, 3),
        (Tranche::MutationListSize, 1),
        (Tranche::PatternCount, 4),
    ];
    assert_eq!(
        expected.iter().map(|(_, count)| count).sum::<usize>(),
        CASES.len()
    );
    for (tranche, count) in expected {
        assert_eq!(
            CASES.iter().filter(|case| case.tranche == tranche).count(),
            count,
            "wrong count for {tranche:?}"
        );
    }
}

#[test]
fn exact_literal_size_cases_cross_one_native_quantifier_boundary() -> Result<()> {
    let graph = GraphStore::default();
    let mut backend = ObservedQuantifierBackend::new();
    backend.replace_all_projects(vec![ResidentProjectImage::graph_only(Arc::new(
        graph.snapshot()?,
    ))])?;
    let observations = backend.observations();

    for case in CASES
        .iter()
        .copied()
        .filter(|case| case.tranche == Tranche::LiteralSize)
    {
        let (expected_headers, expected_row, expected_features) = match case.scenario {
            1 => (
                vec!["n"],
                vec![ResultValue::Scalar(ScalarValue::Integer(3))],
                NativeListFeatures {
                    size: true,
                    list: true,
                    list_add: false,
                },
            ),
            3 => (
                vec!["l"],
                vec![ResultValue::Scalar(ScalarValue::Integer(3))],
                NativeListFeatures {
                    size: true,
                    list: true,
                    list_add: true,
                },
            ),
            4 => (
                vec!["size(l)", "size(null)"],
                vec![
                    ResultValue::Scalar(ScalarValue::Null),
                    ResultValue::Scalar(ScalarValue::Null),
                ],
                NativeListFeatures {
                    size: true,
                    list: false,
                    list_add: false,
                },
            ),
            _ => unreachable!("literal-size tranche contains a foreign scenario"),
        };
        let calls_before = observations.calls.load(Ordering::SeqCst);
        let output = QueryEngine.execute(case.query, &mut execution_context(&graph, &backend))?;
        assert!(
            output.graph_mutations.is_empty(),
            "{} mutated the graph",
            case.name
        );
        assert!(
            output.temporal_mutations.is_empty(),
            "{} mutated temporal state",
            case.name
        );
        assert_eq!(
            observations.calls.load(Ordering::SeqCst),
            calls_before + 1,
            "{} did not cross exactly one quantifier boundary",
            case.name
        );

        let headers = output
            .result
            .schema
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>();
        assert_eq!(headers, expected_headers, "wrong columns for {}", case.name);
        let rows = output
            .result
            .batches
            .iter()
            .flat_map(|batch| {
                (0..batch.row_count).map(|row| {
                    batch
                        .columns
                        .iter()
                        .map(|column| column.values[row].clone())
                        .collect::<Vec<_>>()
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(rows, vec![expected_row], "wrong result for {}", case.name);

        let requests = observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let request = requests
            .last()
            .unwrap_or_else(|| panic!("{} omitted its quantifier request", case.name));
        request.validate()?;
        assert!(
            matches!(&request.source, ResidentQuantifierSource::Unit),
            "{} did not use the graph-free Unit source",
            case.name
        );
        assert_eq!(
            request_list_features(request),
            expected_features,
            "{} lowered the wrong native list/value expression",
            case.name
        );
    }
    Ok(())
}

#[test]
fn exact_string3_reverse_crosses_one_native_quantifier_boundary() -> Result<()> {
    let graph = GraphStore::default();
    let mut backend = ObservedQuantifierBackend::new();
    backend.replace_all_projects(vec![ResidentProjectImage::graph_only(Arc::new(
        graph.snapshot()?,
    ))])?;
    let observations = backend.observations();
    let query = "RETURN reverse('raksO')";

    let output = QueryEngine.execute(query, &mut execution_context(&graph, &backend))?;
    assert_eq!(observations.calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        output.result.schema[0].0, "reverse('raksO')",
        "the native route changed the unaliased Cypher column name"
    );
    assert_eq!(
        result_rows(&output),
        vec![vec![ResultValue::Scalar(ScalarValue::String(
            "Oskar".into()
        ))]]
    );

    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let request = requests
        .last()
        .expect("reverse() omitted its sealed quantifier request");
    request.validate()?;
    assert!(matches!(&request.source, ResidentQuantifierSource::Unit));
    assert!(request.program.stages.iter().any(|stage| matches!(
        stage,
        ResidentQuantifierStage::Project { bindings, .. }
            if bindings.iter().any(|binding| matches!(
                &binding.expression,
                ResidentQuantifierExpression::Function {
                    function: ResidentQuantifierFunction::Reverse,
                    ..
                }
            ))
    )));
    Ok(())
}

#[test]
fn exact_string4_split_unwind_count_crosses_one_native_quantifier_boundary() -> Result<()> {
    let graph = GraphStore::default();
    let mut backend = ObservedQuantifierBackend::new();
    backend.replace_all_projects(vec![ResidentProjectImage::graph_only(Arc::new(
        graph.snapshot()?,
    ))])?;
    let observations = backend.observations();
    let query = "UNWIND split('one1two', '1') AS item RETURN count(item) AS item";

    let output = QueryEngine.execute(query, &mut execution_context(&graph, &backend))?;
    assert_eq!(observations.calls.load(Ordering::SeqCst), 1);
    assert_eq!(output.result.schema[0].0, "item");
    assert_eq!(
        result_rows(&output),
        vec![vec![ResultValue::Scalar(ScalarValue::Integer(2))]]
    );

    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let request = requests
        .last()
        .expect("split()/UNWIND/count omitted its sealed quantifier request");
    request.validate()?;
    assert!(matches!(&request.source, ResidentQuantifierSource::Unit));
    assert!(request.program.stages.iter().any(|stage| matches!(
        stage,
        ResidentQuantifierStage::Unwind {
            expression: ResidentQuantifierExpression::Function {
                function: ResidentQuantifierFunction::Split,
                ..
            },
            ..
        }
    )));
    assert!(
        request
            .program
            .stages
            .iter()
            .any(|stage| matches!(stage, ResidentQuantifierStage::GroupCount { .. }))
    );
    Ok(())
}

#[test]
fn null_split_unwinds_no_rows_and_counts_zero_through_one_native_boundary() -> Result<()> {
    let graph = GraphStore::default();
    let mut backend = ObservedQuantifierBackend::new();
    backend.replace_all_projects(vec![ResidentProjectImage::graph_only(Arc::new(
        graph.snapshot()?,
    ))])?;
    let observations = backend.observations();
    let query = "UNWIND split(null, '1') AS item RETURN count(item) AS item";

    let output = QueryEngine.execute(query, &mut execution_context(&graph, &backend))?;
    assert_eq!(observations.calls.load(Ordering::SeqCst), 1);
    assert_eq!(output.result.schema[0].0, "item");
    assert_eq!(
        result_rows(&output),
        vec![vec![ResultValue::Scalar(ScalarValue::Integer(0))]]
    );

    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let request = requests
        .last()
        .expect("null split()/UNWIND/count omitted its sealed quantifier request");
    request.validate()?;
    assert!(matches!(&request.source, ResidentQuantifierSource::Unit));
    assert!(request.program.stages.iter().any(|stage| matches!(
        stage,
        ResidentQuantifierStage::Unwind {
            expression: ResidentQuantifierExpression::Function {
                function: ResidentQuantifierFunction::Split,
                arguments,
            },
            ..
        } if matches!(
            arguments.as_slice(),
            [
                ResidentQuantifierExpression::Literal(ResidentQuantifierValue::Null),
                ResidentQuantifierExpression::Literal(ResidentQuantifierValue::String(_)),
            ]
        )
    )));
    assert!(
        request
            .program
            .stages
            .iter()
            .any(|stage| matches!(stage, ResidentQuantifierStage::GroupCount { .. }))
    );
    Ok(())
}

#[test]
fn non_size_value_projections_do_not_enter_the_quantifier_route() -> Result<()> {
    let graph = GraphStore::default();
    let mut backend = ObservedQuantifierBackend::new();
    backend.replace_all_projects(vec![ResidentProjectImage::graph_only(Arc::new(
        graph.snapshot()?,
    ))])?;
    let observations = backend.observations();

    for query in [
        "RETURN 1 AS value",
        "WITH 1 AS value RETURN value AS alias",
        "WITH 1 AS value RETURN value + 1 AS alias",
        "RETURN coalesce(null, 1) AS value",
        "RETURN reverse(1) AS value",
        "UNWIND [1, null] AS item RETURN count(item) AS item",
        "RETURN [1, 2, 3] AS value",
        "WITH [1] AS value RETURN value AS alias",
        "RETURN [[], []] + [[]] AS value",
        "RETURN [1, 2] = [1, 2] AS value",
        "RETURN [[1], [2, 3]] AS value",
        "RETURN [x IN [1, 2] | x] AS value",
    ] {
        let calls_before = observations.calls.load(Ordering::SeqCst);
        let _ = QueryEngine.execute(query, &mut execution_context(&graph, &backend));
        assert_eq!(
            observations.calls.load(Ordering::SeqCst),
            calls_before,
            "non-size projection escaped into the quantifier route: {query}"
        );
    }
    Ok(())
}

#[test]
fn exact_pattern_count_manifest_crosses_one_native_boundary_per_case() -> Result<()> {
    let fixture = pattern_count_fixture()?;
    let mut backend = ObservedQuantifierBackend::new();
    install_pattern_fixture(&mut backend, &fixture)?;
    let observations = backend.observations();

    for case in CASES
        .iter()
        .copied()
        .filter(|case| case.tranche == Tranche::PatternCount)
    {
        let calls_before = observations.pattern_count_calls.load(Ordering::SeqCst);
        let output =
            QueryEngine.execute(case.query, &mut execution_context(&fixture.graph, &backend))?;
        assert_eq!(
            observations.pattern_count_calls.load(Ordering::SeqCst),
            calls_before + 1,
            "{} did not cross exactly one pattern-count boundary",
            case.name
        );
        assert!(output.graph_mutations.is_empty());
        assert!(output.temporal_mutations.is_empty());

        let rows = result_rows(&output);
        match case.scenario {
            7 => {
                let ids_and_flags = rows
                    .iter()
                    .map(|row| match row.as_slice() {
                        [
                            ResultValue::Node(node),
                            ResultValue::Scalar(ScalarValue::Boolean(flag)),
                        ] => Ok((node.id, *flag)),
                        _ => Err(Error::internal("pattern-count existence row changed shape")),
                    })
                    .collect::<Result<Vec<_>>>()?;
                assert_eq!(
                    ids_and_flags,
                    vec![
                        (NodeId(1), true),
                        (NodeId(2), true),
                        (NodeId(3), true),
                        (NodeId(4), false),
                        (NodeId(5), false),
                    ]
                );
            }
            8 => assert_eq!(
                rows,
                [2_i64, 2, 1, 0, 0]
                    .into_iter()
                    .map(|value| vec![ResultValue::Scalar(ScalarValue::Integer(value))])
                    .collect::<Vec<_>>()
            ),
            9 => assert_eq!(
                rows,
                [1_i64, 1, 1, 0, 0]
                    .into_iter()
                    .map(|value| vec![ResultValue::Scalar(ScalarValue::Integer(value))])
                    .collect::<Vec<_>>()
            ),
            10 => assert_eq!(
                rows,
                [2_i64, 1, 1, 0, 0]
                    .into_iter()
                    .map(|value| vec![ResultValue::Scalar(ScalarValue::Integer(value))])
                    .collect::<Vec<_>>()
            ),
            _ => unreachable!("pattern-count tranche contains a foreign scenario"),
        }

        let request = observations
            .pattern_count_requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .last()
            .cloned()
            .ok_or_else(|| Error::internal("pattern-count request was not observed"))?;
        request.validate()?;
        assert_eq!(request.node_slots, fixture.graph.node_slot_count());
        assert_eq!(request.edge_slots, fixture.graph.edge_slot_count());
        assert_eq!(request.max_output_rows, fixture.graph.node_slot_count());
        assert_eq!(
            request.obligations().map(|obligation| obligation.id),
            [1, 2, 3]
        );
        assert_eq!(
            request.start_labels,
            ResidentPatternCountStartLabels::Known(vec![fixture.x])
        );
        match case.scenario {
            7 => {
                assert_eq!(request.direction, ResidentDirection::Undirected);
                assert_eq!(
                    request.relationship_types,
                    ResidentPatternRelationshipTypes::Any
                );
            }
            8 => {
                assert_eq!(request.direction, ResidentDirection::Outgoing);
                assert_eq!(
                    request.relationship_types,
                    ResidentPatternRelationshipTypes::Any
                );
            }
            9 => assert_eq!(
                request.relationship_types,
                ResidentPatternRelationshipTypes::Known(vec![fixture.t])
            ),
            10 => {
                let mut types = vec![fixture.t, fixture.other];
                types.sort_unstable();
                assert_eq!(
                    request.relationship_types,
                    ResidentPatternRelationshipTypes::Known(types)
                );
            }
            _ => unreachable!(),
        }
    }
    Ok(())
}

fn direct_pattern_count_request(
    fixture: &PatternCountFixture,
    direction: ResidentDirection,
    relationship_types: ResidentPatternRelationshipTypes,
    nonce: u64,
) -> ResidentPatternCountRequest {
    ResidentPatternCountRequest {
        project: PROJECT,
        expected_bookmark: Bookmark {
            term: 0,
            index: fixture.graph.revision(),
        },
        expected_graph_revision: fixture.graph.revision(),
        expected_layout_version: fixture.graph.layout_version(),
        layers: LayerMask::OBSERVED,
        node_slots: fixture.graph.node_slot_count(),
        edge_slots: fixture.graph.edge_slot_count(),
        start_labels: ResidentPatternCountStartLabels::Known(vec![fixture.x]),
        direction,
        relationship_types,
        execution: ResidentExecutionId {
            high: 0x5041_5454_4552_4e43,
            low: nonce,
        },
        scan_obligation: ResidentExecutionObligation {
            id: 1,
            kind: ResidentObligationKind::PatternScan,
            scope: ResidentObligationScope::PatternScan,
        },
        traversal_obligation: ResidentExecutionObligation {
            id: 2,
            kind: ResidentObligationKind::PatternTraversal,
            scope: ResidentObligationScope::PatternLeaf(0),
        },
        final_obligation: ResidentExecutionObligation {
            id: 3,
            kind: ResidentObligationKind::PatternFilter,
            scope: ResidentObligationScope::PatternFinal,
        },
        max_output_rows: fixture.graph.node_slot_count(),
    }
}

#[test]
fn cpu_pattern_count_backend_preserves_zero_parents_types_layers_and_self_loops() -> Result<()> {
    let fixture = pattern_count_fixture()?;
    let mut backend = CpuBackend::new(64 * 1024 * 1024, 16 * 1024 * 1024);
    backend.replace_all_projects(vec![ResidentProjectImage::graph_only(Arc::new(
        fixture.graph.snapshot()?,
    ))])?;
    let cases = [
        (
            ResidentDirection::Outgoing,
            ResidentPatternRelationshipTypes::Known(vec![fixture.t]),
            vec![1, 1, 0, 0, 0],
        ),
        (
            ResidentDirection::Incoming,
            ResidentPatternRelationshipTypes::Any,
            vec![2, 2, 1, 0, 0],
        ),
        (
            ResidentDirection::Undirected,
            ResidentPatternRelationshipTypes::Any,
            vec![4, 3, 1, 0, 0],
        ),
        (
            ResidentDirection::Outgoing,
            ResidentPatternRelationshipTypes::Never,
            vec![0, 0, 0, 0, 0],
        ),
    ];
    for (index, (direction, relationship_types, expected)) in cases.into_iter().enumerate() {
        let request =
            direct_pattern_count_request(&fixture, direction, relationship_types, index as u64 + 1);
        request.validate()?;
        let validated = backend
            .execute_pattern_count(&request, &CancellationToken::new())?
            .validate(&request, BackendKind::Cpu)?;
        assert_eq!(validated.rows(), &[0, 1, 2, 3, 4]);
        assert_eq!(validated.counts(), expected);
        assert_eq!(
            validated
                .receipts()
                .iter()
                .map(|receipt| receipt.obligation.id)
                .collect::<Vec<_>>(),
            vec![1, 2, 3]
        );
    }
    Ok(())
}

#[test]
fn unsupported_pattern_comprehensions_make_zero_native_calls_and_fail_closed_for_gpu() -> Result<()>
{
    let fixture = pattern_count_fixture()?;
    let mut backend = ObservedQuantifierBackend::reporting(BackendKind::Metal);
    install_pattern_fixture(&mut backend, &fixture)?;
    let observations = backend.observations();
    for query in [
        "MATCH (a:X) RETURN size([(a)-->() | 2]) AS length",
        "MATCH (a:X) RETURN size([(a)-->() WHERE true | 1]) AS length",
        "MATCH (a:X) RETURN size([(a)-->(:Y) | 1]) AS length",
        "MATCH (a:X) RETURN size([(a)-[*]->() | 1]) AS length",
        "MATCH (a:X) RETURN size([(a)-[{p: 1}]->() | 1]) AS length",
        "MATCH (a:X) RETURN size([(a)-->() | 1]) AS length LIMIT 1",
        "MATCH (a:X) RETURN size([(a)-->() | 1]) >= 1 AS b",
        "MATCH (a:X) RETURN size([(a)--() | 1]) AS length",
        "MATCH (a:X) RETURN a, size([(a)-->() | 1]) > 0 AS b",
        "MATCH (a:X) RETURN DISTINCT size([(a)-->() | 1]) AS length",
    ] {
        let calls_before = observations.pattern_count_calls.load(Ordering::SeqCst);
        let error = QueryEngine
            .execute(query, &mut execution_context(&fixture.graph, &backend))
            .expect_err("unsupported pattern comprehension must fail closed for a GPU backend");
        assert_eq!(
            error.code,
            ErrorCode::GpuAdmissionFailure,
            "unexpected error for query: {query}"
        );
        assert_eq!(
            observations.pattern_count_calls.load(Ordering::SeqCst),
            calls_before,
            "unsupported comprehension entered the native route: {query}"
        );
    }
    Ok(())
}
