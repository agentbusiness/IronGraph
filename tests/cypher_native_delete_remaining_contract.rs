// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Exact contract inventory for the original eight remaining native DELETE scenarios.
//!
//! The generic executor is used only as the semantic oracle. Strict observer backends advertise
//! an active accelerator, pin one immutable resident generation, and poison every route except the
//! single complete DELETE command so no acceptance case can acquire a hidden host fallback.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use irongraph::{
    Bookmark, EdgeId, Error, ErrorCode, Layer, NodeId, ProjectId, Result, ScalarValue,
    cypher::{
        BindCapabilities, ExecutionContext, Expression, PhysicalOperator, QueryEngine, ResultValue,
        StatementStats, bind, parse, plan,
    },
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend, ResidentDeleteRequest,
        ResidentDeleteResult, ResidentDeleteTargetSelector, ResidentEntityBinding, ResidentGroup,
        ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest, ResidentNodeBinding,
        ResidentNodePipelineRequest, ResidentNodePipelineResult, ResidentProjectImage,
        ResidentSortRequest, ResidentSortResult, ResidentVectorQuery, ResidentVectorResult,
        ScratchReservation,
    },
    graph::{
        EdgeInput, GraphMutation, GraphStore, IndexCatalog, LayerMask, NodeInput, TemporalStore,
    },
    types::{LabelId, PropertyId},
};
use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::gpu::MetalBackend;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::from_u128(
    0x4445_4c45_5445_5f52_454d_4149_4e49_4e47,
));
const MEMORY_LIMIT_BYTES: usize = 512 * 1024 * 1024;
const RESERVED_BYTES: usize = 64 * 1024 * 1024;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
static REAL_METAL_TEST: Mutex<()> = Mutex::new(());

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Seam {
    NullableTarget,
    PathSelector,
    MixedComposition,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PlanShape {
    OptionalNode,
    MandatoryThenOptionalRelationship,
    OptionalRelationship,
    FixedPath,
    OptionalPath,
    VariablePath,
    MatchCreateDelete,
    NestedCollectedPaths,
}

#[derive(Clone, Copy, Debug)]
struct Case {
    feature: &'static str,
    scenario: u8,
    name: &'static str,
    query: &'static str,
    seam: Seam,
    shape: PlanShape,
    expected_stats: StatementStats,
    expected_nodes: usize,
    expected_relationships: usize,
}

const CASES: [Case; 8] = [
    Case {
        feature: "clauses/delete/Delete1.feature",
        scenario: 5,
        name: "[5] Ignore null when deleting node",
        query: "OPTIONAL MATCH (a:DoesNotExist) DELETE a RETURN a",
        seam: Seam::NullableTarget,
        shape: PlanShape::OptionalNode,
        expected_stats: StatementStats {
            nodes_created: 0,
            nodes_deleted: 0,
            relationships_created: 0,
            relationships_deleted: 0,
            properties_set: 0,
            labels_added: 0,
            labels_removed: 0,
        },
        expected_nodes: 0,
        expected_relationships: 0,
    },
    Case {
        feature: "clauses/delete/Delete2.feature",
        scenario: 2,
        name: "[2] Delete optionally matched relationship",
        query: "MATCH (n) OPTIONAL MATCH (n)-[r]-() DELETE n, r",
        seam: Seam::MixedComposition,
        shape: PlanShape::MandatoryThenOptionalRelationship,
        expected_stats: StatementStats {
            nodes_created: 0,
            nodes_deleted: 1,
            relationships_created: 0,
            relationships_deleted: 0,
            properties_set: 0,
            labels_added: 0,
            labels_removed: 0,
        },
        expected_nodes: 0,
        expected_relationships: 0,
    },
    Case {
        feature: "clauses/delete/Delete2.feature",
        scenario: 4,
        name: "[4] Ignore null when deleting relationship",
        query: "OPTIONAL MATCH ()-[r:DoesNotExist]-() DELETE r RETURN r",
        seam: Seam::NullableTarget,
        shape: PlanShape::OptionalRelationship,
        expected_stats: StatementStats {
            nodes_created: 0,
            nodes_deleted: 0,
            relationships_created: 0,
            relationships_deleted: 0,
            properties_set: 0,
            labels_added: 0,
            labels_removed: 0,
        },
        expected_nodes: 0,
        expected_relationships: 0,
    },
    Case {
        feature: "clauses/delete/Delete3.feature",
        scenario: 1,
        name: "[1] Detach deleting paths",
        query: "MATCH p = (:X)-->()-->()-->() DETACH DELETE p",
        seam: Seam::PathSelector,
        shape: PlanShape::FixedPath,
        expected_stats: StatementStats {
            nodes_created: 0,
            nodes_deleted: 4,
            relationships_created: 0,
            relationships_deleted: 3,
            properties_set: 0,
            labels_added: 0,
            labels_removed: 0,
        },
        expected_nodes: 0,
        expected_relationships: 0,
    },
    Case {
        feature: "clauses/delete/Delete3.feature",
        scenario: 2,
        name: "[2] Delete on null path",
        query: "OPTIONAL MATCH p = ()-->() DETACH DELETE p",
        seam: Seam::NullableTarget,
        shape: PlanShape::OptionalPath,
        expected_stats: StatementStats {
            nodes_created: 0,
            nodes_deleted: 0,
            relationships_created: 0,
            relationships_deleted: 0,
            properties_set: 0,
            labels_added: 0,
            labels_removed: 0,
        },
        expected_nodes: 0,
        expected_relationships: 0,
    },
    Case {
        feature: "clauses/delete/Delete4.feature",
        scenario: 2,
        name: "[2] Undirected variable length expand followed by delete and count",
        query: "MATCH (a)-[*]-(b) DETACH DELETE a, b RETURN count(*) AS c",
        seam: Seam::MixedComposition,
        shape: PlanShape::VariablePath,
        expected_stats: StatementStats {
            nodes_created: 0,
            nodes_deleted: 3,
            relationships_created: 0,
            // DETACH removes incident edges as a graph consequence; only explicit DELETE
            // relationship targets contribute to statement statistics.
            relationships_deleted: 0,
            properties_set: 0,
            labels_added: 0,
            labels_removed: 0,
        },
        expected_nodes: 0,
        expected_relationships: 0,
    },
    Case {
        feature: "clauses/delete/Delete4.feature",
        scenario: 3,
        name: "[3] Create and delete in same query",
        query: "MATCH () CREATE (n) DELETE n",
        seam: Seam::MixedComposition,
        shape: PlanShape::MatchCreateDelete,
        expected_stats: StatementStats {
            // The query performs both effects even though its committed graph delta is empty.
            nodes_created: 1,
            nodes_deleted: 1,
            relationships_created: 0,
            relationships_deleted: 0,
            properties_set: 0,
            labels_added: 0,
            labels_removed: 0,
        },
        expected_nodes: 1,
        expected_relationships: 0,
    },
    Case {
        feature: "clauses/delete/Delete5.feature",
        scenario: 7,
        name: "[7] Delete paths from nested map/list",
        query: concat!(
            "MATCH p = (:User)-[r]->(:User) ",
            "WITH {key: collect(p)} AS pathColls ",
            "DELETE pathColls.key[0], pathColls.key[1]"
        ),
        seam: Seam::PathSelector,
        shape: PlanShape::NestedCollectedPaths,
        expected_stats: StatementStats {
            nodes_created: 0,
            nodes_deleted: 2,
            relationships_created: 0,
            relationships_deleted: 2,
            properties_set: 0,
            labels_added: 0,
            labels_removed: 0,
        },
        expected_nodes: 0,
        expected_relationships: 0,
    },
];

fn insert_node(graph: &mut GraphStore, id: u64, labels: Vec<LabelId>) -> Result<()> {
    graph
        .insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: id,
            labels,
            properties: Vec::new(),
        })
        .map(|_| ())
}

fn insert_edge(
    graph: &mut GraphStore,
    id: u64,
    source: u64,
    target: u64,
    relationship_type: irongraph::types::RelationshipTypeId,
) -> Result<()> {
    graph
        .insert_edge(EdgeInput {
            id: EdgeId(id),
            source: NodeId(source),
            target: NodeId(target),
            relationship_type,
            layer: Layer::Observed,
            revision: 100 + id,
            properties: Vec::new(),
        })
        .map(|_| ())
}

fn fixture(case: Case) -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    match (case.feature, case.scenario) {
        ("clauses/delete/Delete2.feature", 2) | ("clauses/delete/Delete4.feature", 3) => {
            insert_node(&mut graph, 1, Vec::new())?;
        }
        ("clauses/delete/Delete3.feature", 1) => {
            let x = graph.catalog_mut().intern_label("X")?;
            let r = graph.catalog_mut().intern_relationship_type("R")?;
            insert_node(&mut graph, 1, vec![x])?;
            for id in 2..=4 {
                insert_node(&mut graph, id, Vec::new())?;
            }
            for (id, source, target) in [(1, 1, 2), (2, 2, 3), (3, 3, 4)] {
                insert_edge(&mut graph, id, source, target, r)?;
            }
        }
        ("clauses/delete/Delete4.feature", 2) => {
            let r = graph.catalog_mut().intern_relationship_type("R")?;
            for id in 1..=3 {
                insert_node(&mut graph, id, Vec::new())?;
            }
            insert_edge(&mut graph, 1, 1, 2, r)?;
            insert_edge(&mut graph, 2, 2, 3, r)?;
        }
        ("clauses/delete/Delete5.feature", 7) => {
            let user = graph.catalog_mut().intern_label("User")?;
            let r = graph.catalog_mut().intern_relationship_type("R")?;
            insert_node(&mut graph, 1, vec![user])?;
            insert_node(&mut graph, 2, vec![user])?;
            insert_edge(&mut graph, 1, 1, 2, r)?;
            insert_edge(&mut graph, 2, 2, 1, r)?;
        }
        _ => {}
    }
    Ok(graph)
}

fn context<'a>(
    graph: &'a GraphStore,
    backend: Option<&'a dyn ExecutionBackend>,
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
        bookmark: Bookmark {
            term: 71,
            index: graph.revision(),
        },
        mutation_revision: graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id: graph
            .nodes()
            .map(|node| node.id().0.saturating_add(1))
            .max()
            .unwrap_or(1),
        next_edge_id: graph
            .edges()
            .map(|edge| edge.id().0.saturating_add(1))
            .max()
            .unwrap_or(1),
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            write: true,
            require_native_execution,
            ..BindCapabilities::default()
        },
        max_result_rows: 1_024,
        max_batch_rows: 64,
        optimizer_statistics: None,
        backend,
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(30)),
        resolved_query_at_time_nanos: None,
    }
}

fn apply_mutations(graph: &mut GraphStore, mutations: &[GraphMutation]) -> Result<()> {
    for mutation in mutations {
        graph.apply(mutation.clone())?;
    }
    Ok(())
}

fn resident_image(graph: &GraphStore) -> Result<ResidentProjectImage> {
    ResidentProjectImage::build(
        PROJECT,
        Bookmark {
            term: 71,
            index: graph.revision(),
        },
        graph,
        &TemporalStore::default(),
        &IndexCatalog::default(),
    )
}

fn semantic_operators(case: Case, graph: &GraphStore) -> Result<Vec<PhysicalOperator>> {
    let query = parse(case.query)?;
    let bound = bind(
        query,
        graph.catalog(),
        BindCapabilities {
            write: true,
            ..BindCapabilities::default()
        },
    )?;
    Ok(plan(bound)?
        .operators
        .into_iter()
        .filter(|operator| {
            !matches!(
                operator,
                PhysicalOperator::CardinalityCheckpoint { .. } | PhysicalOperator::Finish
            )
        })
        .collect())
}

fn path_is_variable_length(operator: &PhysicalOperator) -> bool {
    matches!(
        operator,
        PhysicalOperator::ScanPattern { pattern, .. }
            if pattern.steps.iter().any(|step| step.relationship.variable_length)
    )
}

fn null_column(output: &irongraph::cypher::ExecutionOutput, name: &str) -> bool {
    matches!(
        output
            .result
            .batches
            .first()
            .and_then(|batch| batch.columns.iter().find(|column| column.name == name))
            .and_then(|column| column.values.first()),
        Some(ResultValue::Scalar(ScalarValue::Null))
    )
}

fn integer_column(output: &irongraph::cypher::ExecutionOutput, name: &str) -> Option<i64> {
    match output
        .result
        .batches
        .first()
        .and_then(|batch| batch.columns.iter().find(|column| column.name == name))
        .and_then(|column| column.values.first())
    {
        Some(ResultValue::Scalar(ScalarValue::Integer(value))) => Some(*value),
        _ => None,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PathDeleteCompletion {
    intents: usize,
    read_dependencies: usize,
    final_rows: Option<usize>,
}

#[derive(Default)]
struct PathDeleteObservations {
    pins: AtomicUsize,
    delete_calls: AtomicUsize,
    rejected_routes: AtomicUsize,
    requests: Mutex<Vec<ResidentDeleteRequest>>,
    completions: Mutex<Vec<PathDeleteCompletion>>,
}

struct ObservedPathDeleteBackend {
    inner: Box<dyn ExecutionBackend>,
    pinned: bool,
    expected_bookmark: Bookmark,
    expected_graph_revision: u64,
    observations: Arc<PathDeleteObservations>,
}

impl ObservedPathDeleteBackend {
    fn admitted(graph: &GraphStore) -> Result<Self> {
        let mut inner = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
        inner.admit_project(resident_image(graph)?)?;
        let expected_bookmark = inner
            .resident_bookmark(PROJECT)
            .ok_or_else(|| Error::internal("path DELETE backend omitted its resident bookmark"))?;
        let expected_graph_revision = inner
            .resident_graph_revision(PROJECT)
            .ok_or_else(|| Error::internal("path DELETE backend omitted its resident revision"))?;
        Ok(Self {
            inner: Box::new(inner),
            pinned: false,
            expected_bookmark,
            expected_graph_revision,
            observations: Arc::new(PathDeleteObservations::default()),
        })
    }

    fn observations(&self) -> Arc<PathDeleteObservations> {
        Arc::clone(&self.observations)
    }

    fn rejected<T>(&self, route: &'static str) -> Result<T> {
        self.observations
            .rejected_routes
            .fetch_add(1, Ordering::SeqCst);
        Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!("strict path DELETE observer reached `{route}`"),
        ))
    }
}

impl ExecutionBackend for ObservedPathDeleteBackend {
    fn kind(&self) -> BackendKind {
        if self.pinned {
            BackendKind::Cpu
        } else {
            // Planning sees an accelerator. Only the immutable pinned generation identifies the
            // delegated semantic-reference completion as CPU-authored.
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
        if self.pinned || project != PROJECT {
            return self.rejected("pin_project");
        }
        let inner = self.inner.pin_project(project)?;
        if inner.kind() != BackendKind::Cpu
            || inner.resident_bookmark(project) != Some(self.expected_bookmark)
            || inner.resident_graph_revision(project) != Some(self.expected_graph_revision)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "path DELETE pin changed the immutable resident generation",
            ));
        }
        self.observations.pins.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Self {
            inner,
            pinned: true,
            expected_bookmark: self.expected_bookmark,
            expected_graph_revision: self.expected_graph_revision,
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
        self.rejected("scan_nodes")
    }

    fn filter_node_i64(
        &self,
        _project: ProjectId,
        _property: PropertyId,
        _operation: CompareOp,
        _operand: i64,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.rejected("filter_node_i64")
    }

    fn expand_project_out(
        &self,
        _project: ProjectId,
        _sources: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.rejected("expand_project_out")
    }

    fn expand_project_in(
        &self,
        _project: ProjectId,
        _targets: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.rejected("expand_project_in")
    }

    fn search_vectors(
        &self,
        _request: &ResidentVectorQuery,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentVectorResult> {
        self.rejected("search_vectors")
    }

    fn sort_rows(
        &self,
        _request: &ResidentSortRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentSortResult> {
        self.rejected("sort_rows")
    }

    fn join_node_i64(
        &self,
        _request: &ResidentJoinRequest,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentJoinPair>> {
        self.rejected("join_node_i64")
    }

    fn group_node_i64(
        &self,
        _request: &ResidentGroupRequest,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentGroup>> {
        self.rejected("group_node_i64")
    }

    fn execute_node_pipeline(
        &self,
        _request: &ResidentNodePipelineRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentNodePipelineResult> {
        self.rejected("execute_node_pipeline")
    }

    fn execute_delete_pipeline(
        &self,
        request: &ResidentDeleteRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentDeleteResult> {
        if !self.pinned {
            return self.rejected("execute_delete_pipeline_on_unpinned_generation");
        }
        if request.project != PROJECT
            || self.inner.resident_bookmark(PROJECT) != Some(self.expected_bookmark)
            || self.inner.resident_graph_revision(PROJECT) != Some(self.expected_graph_revision)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "path DELETE request escaped its pinned resident generation",
            ));
        }
        self.observations
            .delete_calls
            .fetch_add(1, Ordering::SeqCst);
        self.observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(request.clone());
        let result = self.inner.execute_delete_pipeline(request, cancellation)?;
        let validated = result
            .clone()
            .validate_for_publication(request, BackendKind::Cpu)?;
        self.observations
            .completions
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(PathDeleteCompletion {
                intents: validated.intents().len(),
                read_dependencies: validated.read_dependencies().len(),
                final_rows: validated
                    .final_relation()
                    .map(|relation| relation.row_count),
            });
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
        self.rejected("filter_i64")
    }

    fn expand_out(
        &self,
        _sources: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.rejected("expand_out")
    }

    fn exact_l2(
        &self,
        _matrix: &[f32],
        _rows: usize,
        _dimension: usize,
        _query: &[f32],
        _cancellation: &CancellationToken,
    ) -> Result<DistanceBatch> {
        self.rejected("exact_l2")
    }
}

#[test]
fn exact_eight_case_partition_has_three_reusable_seams() {
    let identities = CASES
        .iter()
        .map(|case| (case.feature, case.scenario))
        .collect::<BTreeSet<_>>();
    assert_eq!(identities.len(), CASES.len());
    assert_eq!(
        CASES
            .iter()
            .filter(|case| case.seam == Seam::NullableTarget)
            .count(),
        3
    );
    assert_eq!(
        CASES
            .iter()
            .filter(|case| case.seam == Seam::PathSelector)
            .count(),
        2
    );
    assert_eq!(
        CASES
            .iter()
            .filter(|case| case.seam == Seam::MixedComposition)
            .count(),
        3
    );
}

#[test]
fn exact_source_plans_pin_the_eight_distinct_blocking_shapes() -> Result<()> {
    for case in CASES {
        let graph = fixture(case)?;
        let operators = semantic_operators(case, &graph)?;
        let delete = operators
            .iter()
            .find_map(|operator| match operator {
                PhysicalOperator::Delete {
                    detach,
                    expressions,
                } => Some((*detach, expressions)),
                _ => None,
            })
            .ok_or_else(|| Error::internal(format!("{} omitted DELETE", case.name)))?;

        match case.shape {
            PlanShape::OptionalNode => {
                assert!(matches!(
                    operators.first(),
                    Some(PhysicalOperator::ScanPattern {
                        optional: true,
                        pattern,
                        ..
                    }) if pattern.steps.is_empty() && pattern.variable.is_none()
                ));
                assert!(matches!(
                    delete,
                    (false, expressions)
                        if matches!(expressions.as_slice(), [Expression::Variable(variable)] if variable == "a")
                ));
            }
            PlanShape::MandatoryThenOptionalRelationship => {
                assert!(matches!(
                    operators.as_slice(),
                    [
                        PhysicalOperator::ScanPattern { optional: false, .. },
                        PhysicalOperator::ScanPattern { optional: true, .. },
                        PhysicalOperator::Delete { expressions, .. }
                    ] if expressions.len() == 2
                ));
            }
            PlanShape::OptionalRelationship => {
                assert!(matches!(
                    operators.first(),
                    Some(PhysicalOperator::ScanPattern {
                        optional: true,
                        pattern,
                        ..
                    }) if pattern.steps.len() == 1
                ));
                assert!(matches!(
                    delete,
                    (false, expressions)
                        if matches!(expressions.as_slice(), [Expression::Variable(variable)] if variable == "r")
                ));
            }
            PlanShape::FixedPath => {
                assert!(matches!(
                    operators.first(),
                    Some(PhysicalOperator::ScanPattern { pattern, optional: false, .. })
                        if pattern.variable.as_deref() == Some("p")
                            && pattern.steps.len() == 3
                            && !path_is_variable_length(&operators[0])
                ));
                assert!(matches!(
                    delete,
                    (true, expressions)
                        if matches!(expressions.as_slice(), [Expression::Variable(variable)] if variable == "p")
                ));
            }
            PlanShape::OptionalPath => {
                assert!(matches!(
                    operators.first(),
                    Some(PhysicalOperator::ScanPattern { pattern, optional: true, .. })
                        if pattern.variable.as_deref() == Some("p") && pattern.steps.len() == 1
                ));
                assert!(matches!(
                    delete,
                    (true, expressions)
                        if matches!(expressions.as_slice(), [Expression::Variable(variable)] if variable == "p")
                ));
            }
            PlanShape::VariablePath => {
                assert!(operators.iter().any(path_is_variable_length));
                assert!(matches!(
                    delete,
                    (true, expressions)
                        if matches!(
                            expressions.as_slice(),
                            [Expression::Variable(a), Expression::Variable(b)]
                                if a == "a" && b == "b"
                        )
                ));
                assert!(matches!(
                    operators.last(),
                    Some(PhysicalOperator::Project { .. })
                ));
            }
            PlanShape::MatchCreateDelete => {
                assert!(
                    operators
                        .iter()
                        .any(|operator| matches!(operator, PhysicalOperator::CreatePattern(_)))
                );
                assert!(matches!(
                    delete,
                    (false, expressions)
                        if matches!(expressions.as_slice(), [Expression::Variable(variable)] if variable == "n")
                ));
            }
            PlanShape::NestedCollectedPaths => {
                assert!(
                    operators
                        .iter()
                        .any(|operator| matches!(operator, PhysicalOperator::Project { .. }))
                );
                assert_eq!(delete.1.len(), 2);
                assert!(
                    delete
                        .1
                        .iter()
                        .all(|expression| matches!(expression, Expression::Index { .. }))
                );
            }
        }
    }
    Ok(())
}

#[test]
fn generic_cpu_oracle_pins_all_eight_results_and_side_effects() -> Result<()> {
    for case in CASES {
        let graph = fixture(case)?;
        let output = QueryEngine.execute(case.query, &mut context(&graph, None, false))?;
        assert_eq!(
            output.result.statistics, case.expected_stats,
            "{}",
            case.name
        );
        assert!(!output.result.truncated, "{}", case.name);
        assert!(output.temporal_mutations.is_empty(), "{}", case.name);

        match (case.feature, case.scenario) {
            ("clauses/delete/Delete1.feature", 5) => {
                assert!(null_column(&output, "a"), "{}", case.name);
            }
            ("clauses/delete/Delete2.feature", 4) => {
                assert!(null_column(&output, "r"), "{}", case.name);
            }
            ("clauses/delete/Delete4.feature", 2) => {
                assert_eq!(integer_column(&output, "c"), Some(6), "{}", case.name);
            }
            _ => assert!(output.result.schema.is_empty(), "{}", case.name),
        }

        let mut committed = graph.clone();
        apply_mutations(&mut committed, &output.graph_mutations)?;
        assert_eq!(committed.node_count(), case.expected_nodes, "{}", case.name);
        assert_eq!(
            committed.edge_count(),
            case.expected_relationships,
            "{}",
            case.name
        );
    }
    Ok(())
}

#[test]
fn strict_nullable_cases_dispatch_one_complete_resident_delete_command() -> Result<()> {
    let cases = CASES
        .into_iter()
        .filter(|case| case.seam == Seam::NullableTarget)
        .collect::<Vec<_>>();
    assert_eq!(cases.len(), 3);

    for case in cases {
        let graph = fixture(case)?;
        let oracle = QueryEngine.execute(case.query, &mut context(&graph, None, false))?;
        let backend = ObservedPathDeleteBackend::admitted(&graph)?;
        let observations = backend.observations();
        let native = QueryEngine.execute(case.query, &mut context(&graph, Some(&backend), true))?;

        assert_eq!(observations.pins.load(Ordering::SeqCst), 1, "{}", case.name);
        assert_eq!(
            observations.delete_calls.load(Ordering::SeqCst),
            1,
            "{}",
            case.name
        );
        assert_eq!(
            observations.rejected_routes.load(Ordering::SeqCst),
            0,
            "{} reached a generic backend route",
            case.name
        );
        assert_eq!(native.result, oracle.result, "{}", case.name);
        assert!(native.graph_mutations.is_empty(), "{}", case.name);
        assert!(oracle.graph_mutations.is_empty(), "{}", case.name);

        let requests = observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(requests.len(), 1, "{}", case.name);
        let request = &requests[0];
        assert!(request.selection.initial_optional, "{}", case.name);
        assert!(request.selection_is_statically_empty, "{}", case.name);
        assert!(
            request.commands.iter().all(|command| {
                command.selector == ResidentDeleteTargetSelector::EachSelectedRow
            }),
            "{}",
            case.name
        );
        let expected_targets = match case.scenario {
            5 if case.feature.ends_with("Delete1.feature") => {
                vec![ResidentEntityBinding::Node(ResidentNodeBinding::Start)]
            }
            4 => vec![ResidentEntityBinding::Relationship(0)],
            2 => vec![
                ResidentEntityBinding::Node(ResidentNodeBinding::Start),
                ResidentEntityBinding::Relationship(0),
                ResidentEntityBinding::Node(ResidentNodeBinding::End),
            ],
            _ => return Err(Error::internal("unexpected nullable DELETE scenario")),
        };
        assert_eq!(
            request
                .commands
                .iter()
                .map(|command| command.target)
                .collect::<Vec<_>>(),
            expected_targets,
            "{}",
            case.name
        );
        drop(requests);

        let completions = observations
            .completions
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(completions.len(), 1, "{}", case.name);
        assert_eq!(completions[0].intents, 0, "{}", case.name);
        assert_eq!(completions[0].read_dependencies, 0, "{}", case.name);
        assert_eq!(
            completions[0].final_rows,
            match case.scenario {
                5 | 4 => Some(1),
                2 => None,
                _ => unreachable!(),
            },
            "{}",
            case.name
        );
    }
    Ok(())
}

#[test]
fn strict_bound_optional_relationship_delete_is_one_complete_command() -> Result<()> {
    let case = CASES
        .into_iter()
        .find(|case| case.feature.ends_with("Delete2.feature") && case.scenario == 2)
        .ok_or_else(|| Error::internal("Delete2 [2] case disappeared"))?;
    let graph = fixture(case)?;
    let oracle = QueryEngine.execute(case.query, &mut context(&graph, None, false))?;
    let backend = ObservedPathDeleteBackend::admitted(&graph)?;
    let observations = backend.observations();
    let native = QueryEngine.execute(case.query, &mut context(&graph, Some(&backend), true))?;
    assert_eq!(native.result, oracle.result);
    assert_eq!(
        format!("{:?}", native.graph_mutations),
        format!("{:?}", oracle.graph_mutations)
    );
    assert_eq!(observations.pins.load(Ordering::SeqCst), 1);
    assert_eq!(observations.delete_calls.load(Ordering::SeqCst), 1);
    assert_eq!(observations.rejected_routes.load(Ordering::SeqCst), 0);
    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    assert!(
        requests[0]
            .selection
            .expansion
            .as_ref()
            .is_some_and(|expansion| expansion.optional)
    );
    assert_eq!(requests[0].commands.len(), 2);
    Ok(())
}

#[test]
fn strict_deleted_relationship_type_is_decoded_from_the_prewrite_catalog() -> Result<()> {
    let mut graph = GraphStore::default();
    let relationship_type = graph.catalog_mut().intern_relationship_type("T")?;
    insert_node(&mut graph, 1, Vec::new())?;
    insert_node(&mut graph, 2, Vec::new())?;
    insert_edge(&mut graph, 1, 1, 2, relationship_type)?;
    let query = "MATCH ()-[r]->() DELETE r RETURN type(r)";
    let oracle = QueryEngine.execute(query, &mut context(&graph, None, false))?;
    let backend = ObservedPathDeleteBackend::admitted(&graph)?;
    let native = QueryEngine.execute(query, &mut context(&graph, Some(&backend), true))?;
    assert_eq!(native.result, oracle.result);
    assert_eq!(
        format!("{:?}", native.graph_mutations),
        format!("{:?}", oracle.graph_mutations)
    );
    assert!(matches!(
        native
            .result
            .batches
            .first()
            .and_then(|batch| batch.columns.first())
            .and_then(|column| column.values.first()),
        Some(ResultValue::Scalar(ScalarValue::String(value))) if value.as_ref() == "T"
    ));
    Ok(())
}

#[test]
fn strict_match_create_delete_publishes_both_effects_with_zero_net_graph_change() -> Result<()> {
    let case = CASES
        .into_iter()
        .find(|case| case.feature.ends_with("Delete4.feature") && case.scenario == 3)
        .ok_or_else(|| Error::internal("Delete4 [3] case disappeared"))?;
    let graph = fixture(case)?;
    for query in [
        case.query,
        "MATCH () CREATE (renamed_created_node) DELETE renamed_created_node",
    ] {
        let oracle = QueryEngine.execute(query, &mut context(&graph, None, false))?;
        let mut backend = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
        backend.admit_project(resident_image(&graph)?)?;
        let native = QueryEngine.execute(query, &mut context(&graph, Some(&backend), true))?;
        assert_eq!(native.result, oracle.result, "{query}");
        assert_eq!(
            format!("{:?}", native.graph_mutations),
            format!("{:?}", oracle.graph_mutations),
            "{query}"
        );
        assert_eq!(native.result.statistics.nodes_created, 1, "{query}");
        assert_eq!(native.result.statistics.nodes_deleted, 1, "{query}");
        let mut committed = graph.clone();
        apply_mutations(&mut committed, &native.graph_mutations)?;
        assert_eq!(committed.node_count(), graph.node_count(), "{query}");
        assert_eq!(committed.edge_count(), graph.edge_count(), "{query}");
    }
    Ok(())
}

#[test]
fn strict_path_cases_dispatch_one_complete_resident_delete_command() -> Result<()> {
    let cases = CASES
        .into_iter()
        .filter(|case| case.seam == Seam::PathSelector)
        .collect::<Vec<_>>();
    assert_eq!(cases.len(), 2);

    for case in cases {
        let graph = fixture(case)?;
        let oracle = QueryEngine.execute(case.query, &mut context(&graph, None, false))?;
        let backend = ObservedPathDeleteBackend::admitted(&graph)?;
        assert_eq!(backend.kind(), BackendKind::Metal, "{}", case.name);
        let observations = backend.observations();
        let native = QueryEngine.execute(case.query, &mut context(&graph, Some(&backend), true))?;

        assert_eq!(observations.pins.load(Ordering::SeqCst), 1, "{}", case.name);
        assert_eq!(
            observations.delete_calls.load(Ordering::SeqCst),
            1,
            "{}",
            case.name
        );
        assert_eq!(
            observations.rejected_routes.load(Ordering::SeqCst),
            0,
            "{} reached a generic backend route",
            case.name
        );

        let requests = observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(requests.len(), 1, "{}", case.name);
        let request = &requests[0];
        assert_eq!(request.project, PROJECT, "{}", case.name);
        assert_eq!(
            request.expected_bookmark,
            Bookmark {
                term: 71,
                index: graph.revision(),
            },
            "{}",
            case.name
        );
        assert_eq!(
            request.expected_graph_revision,
            graph.revision(),
            "{}",
            case.name
        );
        assert!(request.continuation.is_none(), "{}", case.name);

        let (targets, selectors, detach, unique_effects) = match case.scenario {
            1 => (
                vec![
                    ResidentEntityBinding::Node(ResidentNodeBinding::Start),
                    ResidentEntityBinding::Relationship(0),
                    ResidentEntityBinding::Node(ResidentNodeBinding::Intermediate(0)),
                    ResidentEntityBinding::Relationship(1),
                    ResidentEntityBinding::Node(ResidentNodeBinding::Intermediate(1)),
                    ResidentEntityBinding::Relationship(2),
                    ResidentEntityBinding::Node(ResidentNodeBinding::End),
                ],
                vec![ResidentDeleteTargetSelector::EachSelectedRow; 7],
                true,
                7,
            ),
            7 => (
                vec![
                    ResidentEntityBinding::Node(ResidentNodeBinding::Start),
                    ResidentEntityBinding::Relationship(0),
                    ResidentEntityBinding::Node(ResidentNodeBinding::End),
                    ResidentEntityBinding::Node(ResidentNodeBinding::Start),
                    ResidentEntityBinding::Relationship(0),
                    ResidentEntityBinding::Node(ResidentNodeBinding::End),
                ],
                vec![
                    ResidentDeleteTargetSelector::CollectedOrdinal { index: 0 },
                    ResidentDeleteTargetSelector::CollectedOrdinal { index: 0 },
                    ResidentDeleteTargetSelector::CollectedOrdinal { index: 0 },
                    ResidentDeleteTargetSelector::CollectedOrdinal { index: 1 },
                    ResidentDeleteTargetSelector::CollectedOrdinal { index: 1 },
                    ResidentDeleteTargetSelector::CollectedOrdinal { index: 1 },
                ],
                false,
                4,
            ),
            scenario => {
                return Err(Error::internal(format!(
                    "unexpected path DELETE scenario {scenario}"
                )));
            }
        };
        assert_eq!(request.commands.len(), targets.len(), "{}", case.name);
        for ((command, target), selector) in request.commands.iter().zip(targets).zip(selectors) {
            assert_eq!(command.target, target, "{}", case.name);
            assert_eq!(command.selector, selector, "{}", case.name);
            assert_eq!(command.detach, detach, "{}", case.name);
        }
        drop(requests);

        let completions = observations
            .completions
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        assert_eq!(
            completions.as_slice(),
            &[PathDeleteCompletion {
                intents: unique_effects,
                read_dependencies: unique_effects,
                final_rows: None,
            }],
            "{} changed validated deduplication or dependency receipts",
            case.name
        );
        drop(completions);

        assert_eq!(native.result.schema, oracle.result.schema, "{}", case.name);
        assert_eq!(
            native.result.batches, oracle.result.batches,
            "{}",
            case.name
        );
        assert_eq!(
            native.result.statistics, oracle.result.statistics,
            "{}",
            case.name
        );
        assert_eq!(
            format!("{:?}", native.graph_mutations),
            format!("{:?}", oracle.graph_mutations),
            "{} changed stable unique effects",
            case.name
        );
        assert_eq!(
            native.graph_mutations.len(),
            unique_effects,
            "{}",
            case.name
        );
        assert_eq!(
            native.dependencies.entities.len(),
            unique_effects,
            "{}",
            case.name
        );
        assert_eq!(
            native.dependencies.write_targets.len(),
            unique_effects,
            "{}",
            case.name
        );

        let mut committed = graph.clone();
        apply_mutations(&mut committed, &native.graph_mutations)?;
        assert_eq!(committed.node_count(), case.expected_nodes, "{}", case.name);
        assert_eq!(
            committed.edge_count(),
            case.expected_relationships,
            "{}",
            case.name
        );
    }
    Ok(())
}

#[test]
fn strict_variable_path_delete_and_count_dispatches_one_complete_command() -> Result<()> {
    let case = CASES
        .into_iter()
        .find(|case| {
            case.feature.ends_with("Delete4.feature")
                && case.scenario == 2
                && case.seam == Seam::MixedComposition
        })
        .ok_or_else(|| Error::internal("Delete4 [2] case disappeared"))?;
    let graph = fixture(case)?;
    let oracle = QueryEngine.execute(case.query, &mut context(&graph, None, false))?;
    let backend = ObservedPathDeleteBackend::admitted(&graph)?;
    let observations = backend.observations();
    let native = QueryEngine.execute(case.query, &mut context(&graph, Some(&backend), true))?;

    assert_eq!(observations.pins.load(Ordering::SeqCst), 1);
    assert_eq!(observations.delete_calls.load(Ordering::SeqCst), 1);
    assert_eq!(observations.rejected_routes.load(Ordering::SeqCst), 0);
    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    assert_eq!(requests.len(), 1);
    let request = &requests[0];
    let path = request
        .variable_path_selection
        .as_ref()
        .ok_or_else(|| Error::internal("Delete4 [2] omitted its fused variable-path selector"))?;
    assert_eq!(path.segments.len(), 1);
    assert_eq!(path.segments[0].minimum_hops, 1);
    assert_eq!(path.segments[0].maximum_hops, None);
    assert_eq!(request.commands.len(), 2);
    assert!(request.commands.iter().all(|command| command.detach));
    assert_eq!(
        request.commands[0].target,
        ResidentEntityBinding::Node(ResidentNodeBinding::Start)
    );
    assert_eq!(
        request.commands[1].target,
        ResidentEntityBinding::Node(ResidentNodeBinding::End)
    );
    assert!(request.continuation.is_some());
    drop(requests);

    let completions = observations
        .completions
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    assert_eq!(
        completions.as_slice(),
        &[PathDeleteCompletion {
            intents: 5,
            read_dependencies: 5,
            final_rows: Some(1),
        }]
    );
    drop(completions);

    assert_eq!(native.result, oracle.result);
    assert_eq!(integer_column(&native, "c"), Some(6));
    assert_eq!(
        format!("{:?}", native.graph_mutations),
        format!("{:?}", oracle.graph_mutations)
    );
    let mut committed = graph;
    apply_mutations(&mut committed, &native.graph_mutations)?;
    assert_eq!(committed.node_count(), 0);
    assert_eq!(committed.edge_count(), 0);
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "hardware acceptance gate: requires a real Metal device"]
fn real_metal_variable_path_delete_and_count_consumes_the_device_path_frame() -> Result<()> {
    let _guard = REAL_METAL_TEST
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let case = CASES
        .into_iter()
        .find(|case| case.feature.ends_with("Delete4.feature") && case.scenario == 2)
        .ok_or_else(|| Error::internal("Delete4 [2] case disappeared"))?;
    let graph = fixture(case)?;
    let oracle = QueryEngine.execute(case.query, &mut context(&graph, None, false))?;
    let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
    metal.admit_project(resident_image(&graph)?)?;
    let native = QueryEngine.execute(case.query, &mut context(&graph, Some(&metal), true))?;

    assert_eq!(native.result, oracle.result);
    assert_eq!(integer_column(&native, "c"), Some(6));
    assert_eq!(native.result.statistics.nodes_deleted, 3);
    assert_eq!(native.result.statistics.relationships_deleted, 0);
    assert_eq!(
        format!("{:?}", native.graph_mutations),
        format!("{:?}", oracle.graph_mutations)
    );
    let mut committed = graph;
    apply_mutations(&mut committed, &native.graph_mutations)?;
    assert_eq!(committed.node_count(), 0);
    assert_eq!(committed.edge_count(), 0);
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "hardware acceptance gate: requires a real Metal device"]
fn real_metal_three_hop_delete_is_outgoing_only_with_incoming_asymmetry() -> Result<()> {
    let _guard = REAL_METAL_TEST
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let mut graph = GraphStore::default();
    let x = graph.catalog_mut().intern_label("X")?;
    let r = graph.catalog_mut().intern_relationship_type("R")?;
    insert_node(&mut graph, 1, vec![x])?;
    for id in 2..=4 {
        insert_node(&mut graph, id, Vec::new())?;
    }
    for id in 7..=9 {
        insert_node(&mut graph, id, Vec::new())?;
    }
    for (id, source, target) in [
        (1, 1, 2),
        (2, 2, 3),
        (3, 3, 4),
        // This equally long chain is reachable from :X only by walking incoming edges.
        // An undirected or host-selected implementation would incorrectly delete nodes 7..=9.
        (7, 7, 1),
        (8, 8, 7),
        (9, 9, 8),
    ] {
        insert_edge(&mut graph, id, source, target, r)?;
    }

    let query = "MATCH p = (:X)-->()-->()-->() DETACH DELETE p";
    let oracle = QueryEngine.execute(query, &mut context(&graph, None, false))?;
    let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
    metal.admit_project(resident_image(&graph)?)?;
    let native = QueryEngine.execute(query, &mut context(&graph, Some(&metal), true))?;

    assert_eq!(native.result, oracle.result);
    assert_eq!(
        format!("{:?}", native.graph_mutations),
        format!("{:?}", oracle.graph_mutations)
    );
    assert_eq!(native.result.statistics.nodes_deleted, 4);
    assert_eq!(native.result.statistics.relationships_deleted, 3);

    let mut committed = graph;
    apply_mutations(&mut committed, &native.graph_mutations)?;
    for id in 1..=4 {
        assert!(
            committed.node(NodeId(id)).is_none(),
            "path node {id} survived"
        );
    }
    for id in 7..=9 {
        assert!(
            committed.node(NodeId(id)).is_some(),
            "incoming-only distractor node {id} was incorrectly selected"
        );
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "hardware acceptance gate: requires a real Metal device"]
fn real_metal_deleted_relationship_type_uses_the_prewrite_catalog() -> Result<()> {
    let _guard = REAL_METAL_TEST
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());

    let mut graph = GraphStore::default();
    let relationship_type = graph.catalog_mut().intern_relationship_type("T")?;
    insert_node(&mut graph, 1, Vec::new())?;
    insert_node(&mut graph, 2, Vec::new())?;
    insert_edge(&mut graph, 1, 1, 2, relationship_type)?;
    let query = "MATCH ()-[r]->() DELETE r RETURN type(r)";
    let oracle = QueryEngine.execute(query, &mut context(&graph, None, false))?;
    let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
    metal.admit_project(resident_image(&graph)?)?;
    let native = QueryEngine.execute(query, &mut context(&graph, Some(&metal), true))?;

    assert_eq!(native.result, oracle.result);
    assert_eq!(
        format!("{:?}", native.graph_mutations),
        format!("{:?}", oracle.graph_mutations)
    );
    assert!(matches!(
        native
            .result
            .batches
            .first()
            .and_then(|batch| batch.columns.first())
            .and_then(|column| column.values.first()),
        Some(ResultValue::Scalar(ScalarValue::String(value))) if value.as_ref() == "T"
    ));
    Ok(())
}
