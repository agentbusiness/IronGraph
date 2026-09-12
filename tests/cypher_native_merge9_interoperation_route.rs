// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Fresh-report manifest and generic semantic oracle for `Merge9.feature`.
//!
//! The four fresh-report rows are not one implementation tranche. Scenarios [1]/[2] require a
//! scheduled value bank feeding row-varying node-MERGE keys (and [2] needs multiple creation
//! groups plus a relationship MERGE). Scenario [3] already fits the generalized ordered command
//! ABI and needs only exact admission. Scenario [4] is a singleton literal pipeline whose
//! predicate and DISTINCT can be proven and folded before the existing constant node MERGE.

use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use irongraph::{
    Bookmark, Error, ErrorCode, ProjectId, Result, ScalarValue,
    cypher::{
        BindCapabilities, ColumnType, ExecutionContext, ExecutionOutput, QueryEngine, ResultValue,
        StatementStats,
    },
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend, ResidentGroup,
        ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest, ResidentNodePipelineRequest,
        ResidentNodePipelineResult, ResidentProjectImage, ResidentRowMutationRequest,
        ResidentRowMutationResult, ResidentSortRequest, ResidentSortResult, ResidentVectorQuery,
        ResidentVectorResult, ScratchReservation,
    },
    graph::{GraphStore, IndexCatalog, LayerMask, TemporalStore},
    types::{LabelId, PropertyId},
};
use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::gpu::MetalBackend;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::from_u128(
    0x4d45_5247_4539_5f49_4e54_4552_4f50_0001,
));
const MEMORY_LIMIT_BYTES: usize = 128 * 1024 * 1024;
const RESERVED_BYTES: usize = 16 * 1024 * 1024;

const QUERY_1: &str = "UNWIND [1, 2, 3, 4] AS int MERGE (n {id: int}) RETURN count(*)";
const QUERY_2: &str = "UNWIND ['Keanu Reeves', 'Hugo Weaving', 'Carrie-Anne Moss', 'Laurence Fishburne'] AS actor MERGE (m:Movie {name: 'The Matrix'}) MERGE (p:Person {name: actor}) MERGE (p)-[:ACTED_IN]->(m)";
const QUERY_3: &str =
    "CREATE (a:A), (b:B) MERGE (a)-[:KNOWS]->(b) CREATE (b)-[:KNOWS]->(c:C) RETURN count(*)";
const QUERY_4: &str = "UNWIND [42] AS props WITH props WHERE props > 32 WITH DISTINCT props AS p MERGE (a:A {num: p}) RETURN a.num AS prop";

struct StrictMerge9Backend {
    inner: Box<dyn ExecutionBackend>,
    reported_kind: BackendKind,
    row_mutation_calls: Arc<AtomicUsize>,
}

impl StrictMerge9Backend {
    fn cpu(inner: CpuBackend, reported_kind: BackendKind) -> Self {
        Self {
            inner: Box::new(inner),
            reported_kind,
            row_mutation_calls: Arc::new(AtomicUsize::new(0)),
        }
    }

    #[cfg(all(feature = "accelerator", target_os = "macos"))]
    fn metal(inner: MetalBackend) -> Self {
        Self {
            inner: Box::new(inner),
            reported_kind: BackendKind::Metal,
            row_mutation_calls: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn calls(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.row_mutation_calls)
    }

    fn reject<T>(&self, route: &'static str) -> Result<T> {
        Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!("strict Merge9 [4] backend rejected fallback route `{route}`"),
        ))
    }
}

impl ExecutionBackend for StrictMerge9Backend {
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
            row_mutation_calls: Arc::clone(&self.row_mutation_calls),
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
        _request: &ResidentNodePipelineRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentNodePipelineResult> {
        self.reject("execute_node_pipeline")
    }

    fn execute_row_mutation(
        &self,
        request: &ResidentRowMutationRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentRowMutationResult> {
        self.row_mutation_calls.fetch_add(1, Ordering::SeqCst);
        self.inner.execute_row_mutation(request, cancellation)
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Tranche {
    DynamicScheduledNodeMerge,
    OrderedComposition,
    FoldedSingletonPipeline,
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
struct Case {
    report_index: usize,
    scenario: u8,
    name: &'static str,
    query: &'static str,
    failure_marker: &'static str,
    tranche: Tranche,
}

const CASES: [Case; 4] = [
    Case {
        report_index: 657,
        scenario: 1,
        name: "[1] UNWIND with one MERGE",
        query: QUERY_1,
        failure_marker: "UNWIND [1, 2, 3, 4] AS int MERGE (n {id: int})",
        tranche: Tranche::DynamicScheduledNodeMerge,
    },
    Case {
        report_index: 658,
        scenario: 2,
        name: "[2] UNWIND with multiple MERGE",
        query: QUERY_2,
        failure_marker: "UNWIND ['Keanu Reeves', 'Hugo Weaving', 'Carrie-Anne Moss', 'Laurence Fishburne'] AS actor",
        tranche: Tranche::DynamicScheduledNodeMerge,
    },
    Case {
        report_index: 659,
        scenario: 3,
        name: "[3] Mixing MERGE with CREATE",
        query: QUERY_3,
        failure_marker: "CREATE (a:A), (b:B) MERGE (a)-[:KNOWS]->(b)",
        tranche: Tranche::OrderedComposition,
    },
    Case {
        report_index: 660,
        scenario: 4,
        name: "[4] MERGE after WITH with predicate and WITH with aggregation",
        query: QUERY_4,
        failure_marker: "UNWIND [42] AS props WITH props WHERE props > 32 WITH DISTINCT props AS p",
        tranche: Tranche::FoldedSingletonPipeline,
    },
];

#[test]
fn merge9_tranche_partition_is_complete_and_non_overlapping() {
    assert_eq!(
        CASES
            .iter()
            .filter(|case| case.tranche == Tranche::DynamicScheduledNodeMerge)
            .count(),
        2
    );
    assert_eq!(
        CASES
            .iter()
            .filter(|case| case.tranche == Tranche::OrderedComposition)
            .count(),
        1
    );
    assert_eq!(
        CASES
            .iter()
            .filter(|case| case.tranche == Tranche::FoldedSingletonPipeline)
            .count(),
        1
    );
}

fn context<'a>(
    graph: &'a GraphStore,
    backend: Option<&'a dyn ExecutionBackend>,
    require_native_execution: bool,
) -> ExecutionContext<'a> {
    let next_node_id = graph
        .nodes()
        .map(|node| node.id().0)
        .max()
        .unwrap_or(0)
        .saturating_add(1);
    let next_edge_id = graph
        .edges()
        .map(|edge| edge.id().0)
        .max()
        .unwrap_or(0)
        .saturating_add(1);
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
            term: 79,
            index: graph.revision(),
        },
        mutation_revision: graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id,
        next_edge_id,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            write: true,
            require_native_execution,
            ..BindCapabilities::default()
        },
        max_result_rows: 32,
        max_batch_rows: 32,
        optimizer_statistics: None,
        backend,
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(30)),
        resolved_query_at_time_nanos: None,
    }
}

fn execute(graph: &GraphStore, query: &str) -> Result<ExecutionOutput> {
    QueryEngine.execute(query, &mut context(graph, None, false))
}

fn apply_output(graph: &mut GraphStore, output: &ExecutionOutput) -> Result<()> {
    for mutation in &output.graph_mutations {
        graph.apply(mutation.clone())?;
    }
    Ok(())
}

fn assert_single_integer(output: &ExecutionOutput, name: &str, value: i64) {
    assert_eq!(
        output.result.schema,
        [(name.to_owned(), ColumnType::Integer)]
    );
    assert!(matches!(
        output.result.batches.as_slice(),
        [batch]
            if batch.validate()
                && batch.row_count == 1
                && matches!(batch.columns.as_slice(), [column]
                    if column.name == name
                        && column.value_type == ColumnType::Integer
                        && column.values == [ResultValue::Scalar(ScalarValue::Integer(value))])
    ));
}

#[test]
fn merge9_generic_oracle_pins_all_four_scenarios() -> Result<()> {
    for case in CASES {
        let mut graph = GraphStore::default();
        if case.scenario == 4 {
            let setup = execute(&graph, "CREATE (:A {num: 42})")?;
            apply_output(&mut graph, &setup)?;
        }
        let output = execute(&graph, case.query)?;
        assert_eq!(output.result.bookmark.index, graph.revision());
        assert!(!output.result.truncated);
        assert!(output.temporal_mutations.is_empty());
        assert!(output.administrative.is_none());
        assert!(output.vector_searches.is_empty());
        assert_eq!(output.runtime_replans, 0);
        match case.scenario {
            1 => {
                assert_single_integer(&output, "count(*)", 4);
                assert_eq!(
                    output.result.statistics,
                    StatementStats {
                        nodes_created: 4,
                        ..StatementStats::default()
                    }
                );
            }
            2 => {
                assert!(output.result.schema.is_empty());
                assert!(output.result.batches.is_empty());
                assert_eq!(
                    output.result.statistics,
                    StatementStats {
                        nodes_created: 5,
                        relationships_created: 4,
                        labels_added: 5,
                        ..StatementStats::default()
                    }
                );
            }
            3 => {
                assert_single_integer(&output, "count(*)", 1);
                assert_eq!(
                    output.result.statistics,
                    StatementStats {
                        nodes_created: 3,
                        relationships_created: 2,
                        labels_added: 3,
                        ..StatementStats::default()
                    }
                );
            }
            4 => {
                assert_single_integer(&output, "prop", 42);
                assert_eq!(output.result.statistics, StatementStats::default());
            }
            _ => unreachable!(),
        }
        apply_output(&mut graph, &output)?;
        match case.scenario {
            1 => {
                let id = graph.catalog().property("id").expect("id token");
                let mut values = graph
                    .nodes()
                    .filter_map(|node| match node.property(id) {
                        Some(ScalarValue::Integer(value)) => Some(value),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                values.sort_unstable();
                assert_eq!(values, [1, 2, 3, 4]);
            }
            2 => {
                let name = graph.catalog().property("name").expect("name token");
                let mut values = graph
                    .nodes()
                    .filter_map(|node| match node.property(name) {
                        Some(ScalarValue::String(value)) => Some(value.to_string()),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                values.sort();
                assert_eq!(
                    values,
                    [
                        "Carrie-Anne Moss",
                        "Hugo Weaving",
                        "Keanu Reeves",
                        "Laurence Fishburne",
                        "The Matrix",
                    ]
                    .into_iter()
                    .map(ToOwned::to_owned)
                    .collect::<Vec<_>>()
                );
                assert_eq!(graph.edge_count(), 4);
            }
            3 => {
                assert_eq!(graph.node_count(), 3);
                assert_eq!(graph.edge_count(), 2);
                for label in ["A", "B", "C"] {
                    let token = graph.catalog().label(label).expect("label token");
                    assert_eq!(
                        graph
                            .nodes()
                            .filter(|node| node.labels().contains(&token))
                            .count(),
                        1
                    );
                }
            }
            4 => {
                assert_eq!(graph.node_count(), 1);
                let num = graph.catalog().property("num").expect("num token");
                assert_eq!(
                    graph.nodes().next().and_then(|node| node.property(num)),
                    Some(ScalarValue::Integer(42))
                );
            }
            _ => unreachable!(),
        }
    }
    Ok(())
}

fn run_strict_merge9_dynamic_case(case: Case, backend: StrictMerge9Backend) -> Result<()> {
    let graph = GraphStore::default();
    let calls = backend.calls();
    let output = QueryEngine.execute(case.query, &mut context(&graph, Some(&backend), true))?;
    if calls.load(Ordering::SeqCst) != 1 {
        return Err(Error::internal(format!(
            "Merge9 [{}] did not execute as one native mutation command",
            case.scenario
        )));
    }
    match case.scenario {
        1 => {
            assert_single_integer(&output, "count(*)", 4);
            assert_eq!(
                output.result.statistics,
                StatementStats {
                    nodes_created: 4,
                    ..StatementStats::default()
                }
            );
        }
        2 => {
            assert!(output.result.schema.is_empty());
            assert!(output.result.batches.is_empty());
            assert_eq!(
                output.result.statistics,
                StatementStats {
                    nodes_created: 5,
                    relationships_created: 4,
                    labels_added: 5,
                    ..StatementStats::default()
                }
            );
        }
        _ => {
            return Err(Error::internal(
                "strict dynamic Merge9 case is out of range",
            ));
        }
    }
    let mut published = graph;
    apply_output(&mut published, &output)?;
    if case.scenario == 1 {
        let id = published
            .catalog()
            .property("id")
            .ok_or_else(|| Error::internal("Merge9 [1] lost its id property"))?;
        let mut values = published
            .nodes()
            .filter_map(|node| match node.property(id) {
                Some(ScalarValue::Integer(value)) => Some(value),
                _ => None,
            })
            .collect::<Vec<_>>();
        values.sort_unstable();
        if values != [1, 2, 3, 4] {
            return Err(Error::internal(format!(
                "Merge9 [1] dynamic values changed: {values:?}"
            )));
        }
    } else {
        let name = published
            .catalog()
            .property("name")
            .ok_or_else(|| Error::internal("Merge9 [2] lost its name property"))?;
        let mut values = published
            .nodes()
            .filter_map(|node| match node.property(name) {
                Some(ScalarValue::String(value)) => Some(value.to_string()),
                _ => None,
            })
            .collect::<Vec<_>>();
        values.sort();
        let expected = [
            "Carrie-Anne Moss",
            "Hugo Weaving",
            "Keanu Reeves",
            "Laurence Fishburne",
            "The Matrix",
        ]
        .into_iter()
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
        if values != expected || published.edge_count() != 4 {
            return Err(Error::internal(format!(
                "Merge9 [2] effects changed: values={values:?}, edges={}",
                published.edge_count()
            )));
        }
    }
    Ok(())
}

fn empty_merge9_cpu_backend() -> Result<CpuBackend> {
    let graph = GraphStore::default();
    let image = ResidentProjectImage::build(
        PROJECT,
        Bookmark {
            term: 79,
            index: graph.revision(),
        },
        &graph,
        &TemporalStore::default(),
        &IndexCatalog::default(),
    )?;
    let mut backend = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
    backend.admit_project(image)?;
    Ok(backend)
}

#[test]
fn strict_cpu_merge9_1_and_2_use_scheduled_dynamic_node_merges() -> Result<()> {
    for case in &CASES[..2] {
        let backend = StrictMerge9Backend::cpu(empty_merge9_cpu_backend()?, BackendKind::Cpu);
        run_strict_merge9_dynamic_case(*case, backend)?;
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires a real Metal device; proves literal-UNWIND node/relationship MERGE grouping"]
fn real_metal_merge9_1_and_2_use_scheduled_dynamic_node_merges() -> Result<()> {
    for case in &CASES[..2] {
        let graph = GraphStore::default();
        let image = ResidentProjectImage::build(
            PROJECT,
            Bookmark {
                term: 79,
                index: graph.revision(),
            },
            &graph,
            &TemporalStore::default(),
            &IndexCatalog::default(),
        )?;
        let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
        metal.admit_project(image)?;
        run_strict_merge9_dynamic_case(*case, StrictMerge9Backend::metal(metal))?;
    }
    Ok(())
}

#[test]
fn strict_cpu_merge9_3_uses_the_complete_native_ordered_mutation() -> Result<()> {
    let graph = GraphStore::default();
    let bookmark = Bookmark {
        term: 79,
        index: graph.revision(),
    };
    let image = ResidentProjectImage::build(
        PROJECT,
        bookmark,
        &graph,
        &TemporalStore::default(),
        &IndexCatalog::default(),
    )?;
    let mut backend = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
    backend.admit_project(image)?;
    let output = QueryEngine.execute(QUERY_3, &mut context(&graph, Some(&backend), true))?;
    assert_single_integer(&output, "count(*)", 1);
    assert_eq!(output.result.bookmark, bookmark);
    assert_eq!(
        output.result.statistics,
        StatementStats {
            nodes_created: 3,
            relationships_created: 2,
            labels_added: 3,
            ..StatementStats::default()
        }
    );
    assert!(!output.result.truncated);
    assert!(output.temporal_mutations.is_empty());
    assert!(output.administrative.is_none());
    assert!(output.vector_searches.is_empty());
    assert_eq!(output.runtime_replans, 0);
    let mut published = graph.clone();
    apply_output(&mut published, &output)?;
    assert_eq!(published.node_count(), 3);
    assert_eq!(published.edge_count(), 2);
    for label in ["A", "B", "C"] {
        let token = published.catalog().label(label).expect("label token");
        assert_eq!(
            published
                .nodes()
                .filter(|node| node.labels().contains(&token))
                .count(),
            1
        );
    }
    let knows = published
        .catalog()
        .relationship_type("KNOWS")
        .expect("KNOWS token");
    assert_eq!(
        published
            .edges()
            .filter(|edge| edge.relationship_type() == knows)
            .count(),
        2
    );
    Ok(())
}

fn merge9_4_graph_and_backend() -> Result<(GraphStore, CpuBackend)> {
    let mut graph = GraphStore::default();
    let setup = execute(&graph, "CREATE (:A {num: 42})")?;
    apply_output(&mut graph, &setup)?;
    let image = ResidentProjectImage::build(
        PROJECT,
        Bookmark {
            term: 79,
            index: graph.revision(),
        },
        &graph,
        &TemporalStore::default(),
        &IndexCatalog::default(),
    )?;
    let mut backend = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
    backend.admit_project(image)?;
    Ok((graph, backend))
}

#[test]
fn strict_cpu_merge9_4_folds_the_complete_singleton_pipeline() -> Result<()> {
    let (graph, cpu) = merge9_4_graph_and_backend()?;
    let backend = StrictMerge9Backend::cpu(cpu, BackendKind::Cpu);
    let calls = backend.calls();
    let bookmark = Bookmark {
        term: 79,
        index: graph.revision(),
    };
    let output = QueryEngine.execute(QUERY_4, &mut context(&graph, Some(&backend), true))?;
    assert_single_integer(&output, "prop", 42);
    assert_eq!(output.result.bookmark, bookmark);
    assert_eq!(output.result.statistics, StatementStats::default());
    assert!(output.graph_mutations.is_empty());
    assert!(output.temporal_mutations.is_empty());
    assert!(output.administrative.is_none());
    assert!(output.vector_searches.is_empty());
    assert_eq!(output.runtime_replans, 0);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    Ok(())
}

#[test]
fn strict_cpu_singleton_fold_derives_names_and_uses_canonical_comparisons() -> Result<()> {
    for predicate in [
        "candidate = 7",
        "candidate <> 8",
        "candidate < 8",
        "candidate <= 7",
        "8 > candidate",
        "7 >= candidate",
    ] {
        let (graph, cpu) = merge9_4_graph_and_backend()?;
        let backend = StrictMerge9Backend::cpu(cpu, BackendKind::Cpu);
        let calls = backend.calls();
        let query = format!(
            "UNWIND [7] AS candidate WITH candidate WHERE {predicate} WITH DISTINCT candidate AS lookup MERGE (record:Bucket {{code: lookup}}) RETURN record.code AS answer"
        );
        let output = QueryEngine.execute(&query, &mut context(&graph, Some(&backend), true))?;
        assert_single_integer(&output, "answer", 7);
        assert_eq!(
            output.result.statistics,
            StatementStats {
                nodes_created: 1,
                labels_added: 1,
                ..StatementStats::default()
            },
            "{predicate}"
        );
        assert_eq!(output.graph_mutations.len(), 3, "{predicate}");
        assert_eq!(calls.load(Ordering::SeqCst), 1, "{predicate}");
    }
    Ok(())
}

#[test]
fn strict_cpu_merge9_4_nearby_shapes_fail_closed() -> Result<()> {
    let (graph, cpu) = merge9_4_graph_and_backend()?;
    // No Metal code runs: accelerator provenance is reported only to activate the executor's
    // no-host-fallback boundary for the compile-time rejection probes below.
    let backend = StrictMerge9Backend::cpu(cpu, BackendKind::Metal);
    let calls = backend.calls();
    for query in [
        "UNWIND [7, 7] AS candidate WITH candidate WHERE candidate <= 10 WITH DISTINCT candidate AS lookup MERGE (record:Bucket {code: lookup}) RETURN record.code AS answer",
        "UNWIND [7] AS candidate WITH candidate WHERE candidate <= 10 WITH candidate AS lookup MERGE (record:Bucket {code: lookup}) RETURN record.code AS answer",
        "UNWIND [7] AS candidate WITH candidate WHERE candidate > 10 WITH DISTINCT candidate AS lookup MERGE (record:Bucket {code: lookup}) RETURN record.code AS answer",
        "UNWIND [7] AS candidate WITH candidate WHERE candidate + 1 > 0 WITH DISTINCT candidate AS lookup MERGE (record:Bucket {code: lookup}) RETURN record.code AS answer",
        "UNWIND [7] AS candidate WITH candidate WHERE candidate <= 10 WITH DISTINCT candidate AS lookup MERGE (record:Bucket {code: lookup, other: 1}) RETURN record.code AS answer",
    ] {
        let error = QueryEngine
            .execute(query, &mut context(&graph, Some(&backend), true))
            .expect_err("neighboring Merge9 [4] shape unexpectedly entered native execution");
        assert_eq!(error.code, ErrorCode::GpuAdmissionFailure, "{query}");
        assert_eq!(calls.load(Ordering::SeqCst), 0, "{query}");
    }
    Ok(())
}
