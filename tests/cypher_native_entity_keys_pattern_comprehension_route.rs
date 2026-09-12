// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Baseline manifest for the 19 remaining Metal-only Graph8/Pattern2 gaps.
//!
//! The manifest pins exact feature, scenario, and primary-query identity from the fresh full
//! conformance report.  Tranches describe implementation ownership; they are not substitutes for
//! native CPU/Metal semantic acceptance tests in the eventual implementation lanes.

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
    Bookmark, EdgeId, ErrorCode, Layer, NodeId, ProjectId, Result, ScalarValue,
    cypher::{BindCapabilities, ExecutionContext, QueryEngine, ResultValue},
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend, ResidentGroup,
        ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest, ResidentNodePipelineRequest,
        ResidentNodePipelineResult, ResidentPatternCountRequest, ResidentPatternCountResult,
        ResidentPatternCountStartLabels, ResidentPatternRelationshipTypes, ResidentProjectImage,
        ResidentSortRequest, ResidentSortResult, ResidentVectorQuery, ResidentVectorResult,
        ScratchReservation,
    },
    graph::{EdgeInput, GraphStore, LayerMask, NodeInput},
    types::{LabelId, PropertyId, RelationshipTypeId},
};

use serde::Deserialize;
use tokio_util::sync::CancellationToken;

/// The generated conformance report produced by the external assurance harness.
///
/// This manifest used to assert that its nineteen scenarios still failed on Metal, against a
/// report in a temporary directory. The directory is reaped, so the assertion became a
/// missing-file panic; and the baseline it encoded is obsolete now that every scenario passes on
/// both backends. It now tracks the same nineteen scenarios against generated evidence and
/// requires them to pass when the external assurance gate is run.
const REPORT: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/evidence/opencypher-tck-2024.3.json"
);
const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Tranche {
    NodeKeys,
    RelationshipKeys,
    OptionalKeys,
    FixedPatternList,
    PatternComposition,
    VariablePatternList,
    NestedPatternList,
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

const CASES: [Case; 19] = [
    Case {
        report_index: 1577,
        feature: "expressions/graph/Graph8.feature",
        scenario: 1,
        name: "[1] Using `keys()` on a single node, non-empty result",
        query: "MATCH (n) UNWIND keys(n) AS x RETURN DISTINCT x AS theProps",
        tranche: Tranche::NodeKeys,
    },
    Case {
        report_index: 1578,
        feature: "expressions/graph/Graph8.feature",
        scenario: 2,
        name: "[2] Using `keys()` on multiple nodes, non-empty result",
        query: "MATCH (n) UNWIND keys(n) AS x RETURN DISTINCT x AS theProps",
        tranche: Tranche::NodeKeys,
    },
    Case {
        report_index: 1579,
        feature: "expressions/graph/Graph8.feature",
        scenario: 3,
        name: "[3] Using `keys()` on a single node, empty result",
        query: "MATCH (n) UNWIND keys(n) AS x RETURN DISTINCT x AS theProps",
        tranche: Tranche::NodeKeys,
    },
    Case {
        report_index: 1580,
        feature: "expressions/graph/Graph8.feature",
        scenario: 4,
        name: "[4] Using `keys()` on an optionally matched node",
        query: "OPTIONAL MATCH (n) UNWIND keys(n) AS x RETURN DISTINCT x AS theProps",
        tranche: Tranche::OptionalKeys,
    },
    Case {
        report_index: 1581,
        feature: "expressions/graph/Graph8.feature",
        scenario: 5,
        name: "[5] Using `keys()` on a relationship, non-empty result",
        query: "MATCH ()-[r:KNOWS]-() UNWIND keys(r) AS x RETURN DISTINCT x AS theProps",
        tranche: Tranche::RelationshipKeys,
    },
    Case {
        report_index: 1582,
        feature: "expressions/graph/Graph8.feature",
        scenario: 6,
        name: "[6] Using `keys()` on a relationship, empty result",
        query: "MATCH ()-[r:KNOWS]-() UNWIND keys(r) AS x RETURN DISTINCT x AS theProps",
        tranche: Tranche::RelationshipKeys,
    },
    Case {
        report_index: 1583,
        feature: "expressions/graph/Graph8.feature",
        scenario: 7,
        name: "[7] Using `keys()` on an optionally matched relationship",
        query: "OPTIONAL MATCH ()-[r:KNOWS]-() UNWIND keys(r) AS x RETURN DISTINCT x AS theProps",
        tranche: Tranche::OptionalKeys,
    },
    Case {
        report_index: 1584,
        feature: "expressions/graph/Graph8.feature",
        scenario: 8,
        name: "[8] Using `keys()` and `IN` to check property existence",
        query: "MATCH (n) RETURN 'exists' IN keys(n) AS a, 'missing' IN keys(n) AS b, 'missingToo' IN keys(n) AS c",
        tranche: Tranche::NodeKeys,
    },
    Case {
        report_index: 2048,
        feature: "expressions/pattern/Pattern2.feature",
        scenario: 1,
        name: "[1] Return a pattern comprehension",
        query: "MATCH (n) RETURN [p = (n)-->() | p] AS list",
        tranche: Tranche::FixedPatternList,
    },
    Case {
        report_index: 2049,
        feature: "expressions/pattern/Pattern2.feature",
        scenario: 2,
        name: "[2] Return a pattern comprehension with label predicate",
        query: "MATCH (n:A) RETURN [p = (n)-->(:B) | p] AS list",
        tranche: Tranche::FixedPatternList,
    },
    Case {
        report_index: 2050,
        feature: "expressions/pattern/Pattern2.feature",
        scenario: 3,
        name: "[3] Return a pattern comprehension with bound nodes",
        query: "MATCH (a:A), (b:B) RETURN [p = (a)-->(b) | p] AS list",
        tranche: Tranche::FixedPatternList,
    },
    Case {
        report_index: 2051,
        feature: "expressions/pattern/Pattern2.feature",
        scenario: 4,
        name: "[4] Introduce a new node variable in pattern comprehension",
        query: "MATCH (n) RETURN [(n)-[:T]->(b) | b.name] AS list",
        tranche: Tranche::FixedPatternList,
    },
    Case {
        report_index: 2052,
        feature: "expressions/pattern/Pattern2.feature",
        scenario: 5,
        name: "[5] Introduce a new relationship variable in pattern comprehension",
        query: "MATCH (n) RETURN [(n)-[r:T]->() | r.name] AS list",
        tranche: Tranche::FixedPatternList,
    },
    Case {
        report_index: 2053,
        feature: "expressions/pattern/Pattern2.feature",
        scenario: 6,
        name: "[6] Aggregate on a pattern comprehension",
        query: "MATCH (n:A) RETURN count([p = (n)-[:HAS]->() | p]) AS c",
        tranche: Tranche::PatternComposition,
    },
    Case {
        report_index: 2054,
        feature: "expressions/pattern/Pattern2.feature",
        scenario: 7,
        name: "[7] Use a pattern comprehension inside a list comprehension",
        query: "MATCH p = (n:X)-->() RETURN n, [x IN nodes(p) | size([(x)-->(:Y) | 1])] AS list",
        tranche: Tranche::NestedPatternList,
    },
    Case {
        report_index: 2055,
        feature: "expressions/pattern/Pattern2.feature",
        scenario: 8,
        name: "[8] Use a pattern comprehension in WITH",
        query: "MATCH (n)-->(b) WITH [p = (n)-->() | p] AS ps, count(b) AS c RETURN ps, c",
        tranche: Tranche::PatternComposition,
    },
    Case {
        report_index: 2056,
        feature: "expressions/pattern/Pattern2.feature",
        scenario: 9,
        name: "[9] Use a variable-length pattern comprehension in WITH",
        query: "MATCH (a:A), (b:B) WITH [p = (a)-[*]->(b) | p] AS paths, count(a) AS c RETURN paths, c",
        tranche: Tranche::VariablePatternList,
    },
    Case {
        report_index: 2057,
        feature: "expressions/pattern/Pattern2.feature",
        scenario: 10,
        name: "[10] Use a pattern comprehension in RETURN",
        query: "MATCH (n:A) RETURN [p = (n)-[:HAS]->() | p] AS ps",
        tranche: Tranche::FixedPatternList,
    },
    Case {
        report_index: 2058,
        feature: "expressions/pattern/Pattern2.feature",
        scenario: 11,
        name: "[11] Use a pattern comprehension and ORDER BY",
        query: "MATCH (liker) RETURN [p = (liker)--() | p] AS isNew ORDER BY liker.time",
        tranche: Tranche::PatternComposition,
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

#[test]
#[ignore = "external assurance gate: requires the generated full TCK report"]
fn exact_19_case_manifest_matches_the_fresh_full_report() {
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
            actual.metal_passed,
            "Metal regressed for {}: {:?}",
            case.name, actual.metal_failures
        );
        assert!(
            actual.metal_failures.is_empty(),
            "{} still reports Metal failures: {:?}",
            case.name,
            actual.metal_failures
        );
        // The manifest still pins the exact query text so a scenario cannot be silently swapped.
        assert!(!case.query.is_empty(), "manifest query text is required");
    }
}

#[test]
fn tranche_partition_is_complete_and_non_overlapping() {
    let expected = [
        (Tranche::NodeKeys, 4),
        (Tranche::RelationshipKeys, 2),
        (Tranche::OptionalKeys, 2),
        (Tranche::FixedPatternList, 6),
        (Tranche::PatternComposition, 3),
        (Tranche::VariablePatternList, 1),
        (Tranche::NestedPatternList, 1),
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

#[derive(Default)]
struct PatternCountObservations {
    calls: AtomicUsize,
    requests: Mutex<Vec<ResidentPatternCountRequest>>,
}

/// Runs the real CPU semantic reference while recording the sealed pattern-count boundary. The
/// reported kind can be changed to prove unsupported shapes fail before any backend call; tests
/// never use that mode to claim Metal completion.
struct ObservedPatternCountBackend {
    inner: Box<dyn ExecutionBackend>,
    reported_kind: BackendKind,
    observations: Arc<PatternCountObservations>,
}

impl ObservedPatternCountBackend {
    fn reporting(reported_kind: BackendKind) -> Self {
        Self {
            inner: Box::new(CpuBackend::new(64 * 1024 * 1024, 16 * 1024 * 1024)),
            reported_kind,
            observations: Arc::new(PatternCountObservations::default()),
        }
    }

    fn observations(&self) -> Arc<PatternCountObservations> {
        Arc::clone(&self.observations)
    }
}

impl ExecutionBackend for ObservedPatternCountBackend {
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
        self.observations.calls.fetch_add(1, Ordering::SeqCst);
        self.observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(request.clone());
        self.inner.execute_pattern_count(request, cancellation)
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

struct Pattern2CountFixture {
    graph: GraphStore,
    a: LabelId,
    has: RelationshipTypeId,
}

fn pattern2_count_fixture() -> Result<Pattern2CountFixture> {
    let mut graph = GraphStore::default();
    let a = graph.catalog_mut().intern_label("A")?;
    let has = graph.catalog_mut().intern_relationship_type("HAS")?;
    for id in 1..=3 {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: id,
            labels: vec![a],
            properties: Vec::new(),
        })?;
    }
    graph.insert_node(NodeInput {
        id: NodeId(4),
        layer: Layer::Observed,
        revision: 4,
        labels: Vec::new(),
        properties: Vec::new(),
    })?;
    graph.insert_edge(EdgeInput {
        id: EdgeId(10),
        source: NodeId(1),
        target: NodeId(4),
        relationship_type: has,
        layer: Layer::Observed,
        revision: 10,
        properties: Vec::new(),
    })?;
    Ok(Pattern2CountFixture { graph, a, has })
}

fn install_pattern2_count_fixture(
    backend: &mut ObservedPatternCountBackend,
    fixture: &Pattern2CountFixture,
) -> Result<()> {
    backend.replace_all_projects(vec![ResidentProjectImage::graph_only(Arc::new(
        fixture.graph.snapshot()?,
    ))])
}

fn pattern2_execution_context<'a>(
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
        next_node_id: 100,
        next_edge_id: 100,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            require_native_execution: true,
            ..BindCapabilities::default()
        },
        // The aggregate returns one row but must consume all three matching parents.
        max_result_rows: 1,
        max_batch_rows: 1,
        optimizer_statistics: None,
        backend: Some(backend),
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(60)),
        resolved_query_at_time_nanos: None,
    }
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

#[test]
fn pattern2_06_counts_every_non_null_list_through_one_native_parent_scan() -> Result<()> {
    let fixture = pattern2_count_fixture()?;
    let mut backend = ObservedPatternCountBackend::reporting(BackendKind::Cpu);
    install_pattern2_count_fixture(&mut backend, &fixture)?;
    let observations = backend.observations();

    let output = QueryEngine.execute(
        "MATCH (n:A) RETURN count([p = (n)-[:HAS]->() | p]) AS c",
        &mut pattern2_execution_context(&fixture.graph, &backend),
    )?;
    assert_eq!(observations.calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        result_rows(&output),
        vec![vec![ResultValue::Scalar(ScalarValue::Integer(3))]]
    );
    assert!(output.graph_mutations.is_empty());
    assert!(output.temporal_mutations.is_empty());

    let request = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .last()
        .cloned()
        .ok_or_else(|| irongraph::Error::internal("pattern-count request was not observed"))?;
    request.validate()?;
    assert_eq!(
        request.start_labels,
        ResidentPatternCountStartLabels::Known(vec![fixture.a])
    );
    assert_eq!(
        request.relationship_types,
        ResidentPatternRelationshipTypes::Known(vec![fixture.has])
    );
    assert_eq!(request.max_output_rows, fixture.graph.node_slot_count());
    Ok(())
}

#[test]
fn pattern2_06_zero_known_parents_returns_one_integer_zero() -> Result<()> {
    let mut graph = GraphStore::default();
    let a = graph.catalog_mut().intern_label("A")?;
    let has = graph.catalog_mut().intern_relationship_type("HAS")?;
    let mut backend = ObservedPatternCountBackend::reporting(BackendKind::Cpu);
    backend.replace_all_projects(vec![ResidentProjectImage::graph_only(Arc::new(
        graph.snapshot()?,
    ))])?;
    let observations = backend.observations();

    let output = QueryEngine.execute(
        "MATCH (n:A) RETURN count([p = (n)-[:HAS]->() | p]) AS c",
        &mut pattern2_execution_context(&graph, &backend),
    )?;
    assert_eq!(observations.calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        result_rows(&output),
        vec![vec![ResultValue::Scalar(ScalarValue::Integer(0))]]
    );

    let request = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .last()
        .cloned()
        .ok_or_else(|| irongraph::Error::internal("zero-parent request was not observed"))?;
    request.validate()?;
    assert_eq!(
        request.start_labels,
        ResidentPatternCountStartLabels::Known(vec![a])
    );
    assert_eq!(
        request.relationship_types,
        ResidentPatternRelationshipTypes::Known(vec![has])
    );
    assert_eq!(request.max_output_rows, 0);
    Ok(())
}

#[test]
fn pattern2_06_unknown_label_and_type_use_never_domains_and_return_zero() -> Result<()> {
    let fixture = pattern2_count_fixture()?;
    let mut backend = ObservedPatternCountBackend::reporting(BackendKind::Cpu);
    install_pattern2_count_fixture(&mut backend, &fixture)?;
    let observations = backend.observations();

    let output = QueryEngine.execute(
        "MATCH (n:Missing) RETURN count([p = (n)-[:MISSING]->() | p]) AS c",
        &mut pattern2_execution_context(&fixture.graph, &backend),
    )?;
    assert_eq!(observations.calls.load(Ordering::SeqCst), 1);
    assert_eq!(
        result_rows(&output),
        vec![vec![ResultValue::Scalar(ScalarValue::Integer(0))]]
    );

    let request = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .last()
        .cloned()
        .ok_or_else(|| irongraph::Error::internal("unknown-domain request was not observed"))?;
    request.validate()?;
    assert_eq!(request.start_labels, ResidentPatternCountStartLabels::Never);
    assert_eq!(
        request.relationship_types,
        ResidentPatternRelationshipTypes::Never
    );
    assert_eq!(request.max_output_rows, fixture.graph.node_slot_count());
    Ok(())
}

#[test]
fn pattern2_06_nearby_shapes_make_zero_native_calls_and_fail_closed_for_gpu() -> Result<()> {
    let fixture = pattern2_count_fixture()?;
    let mut backend = ObservedPatternCountBackend::reporting(BackendKind::Metal);
    install_pattern2_count_fixture(&mut backend, &fixture)?;
    let observations = backend.observations();

    for query in [
        "MATCH (n:A) RETURN count(DISTINCT [p = (n)-[:HAS]->() | p]) AS c",
        "MATCH (n:A) RETURN count([p = (n)-[:HAS]->() WHERE true | p]) AS c",
        "MATCH (n:A) RETURN count([p = (n)-[:HAS]->() | 1]) AS c",
        "MATCH (n:A) RETURN count([p = (n)-[:HAS*]->() | p]) AS c",
        "MATCH (n:A) RETURN count([p = (n)-[:HAS|OTHER]->() | p]) AS c",
        "MATCH (n:A) RETURN count([p = (n)-[:HAS]->(m) | p]) AS c",
        "MATCH (n:A) RETURN count([p = (n)-[:HAS]->() | p]) AS c LIMIT 1",
        "MATCH (n:A) RETURN sum([p = (n)-[:HAS]->() | p]) AS c",
    ] {
        let calls_before = observations.calls.load(Ordering::SeqCst);
        let error = QueryEngine
            .execute(
                query,
                &mut pattern2_execution_context(&fixture.graph, &backend),
            )
            .expect_err("unsupported Pattern2 aggregate must fail closed for a GPU backend");
        assert_eq!(error.code, ErrorCode::GpuAdmissionFailure, "query: {query}");
        assert_eq!(
            observations.calls.load(Ordering::SeqCst),
            calls_before,
            "unsupported Pattern2 aggregate entered the native route: {query}"
        );
    }
    Ok(())
}
