// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{
    collections::{BTreeMap, BTreeSet},
    mem::size_of,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use irongraph::{
    Bookmark, Error, ErrorCode, ProjectId, Result, ScalarValue,
    cypher::{
        BindCapabilities, ExecutionContext, ExecutionOutput, QueryEngine, ResultValue,
        StatementStats,
    },
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend, ResidentGroup,
        ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest, ResidentNodePipelineRequest,
        ResidentNodePipelineResult, ResidentNullableNodeDomain,
        ResidentNullableRelationBindingKind, ResidentNullableRelationEntityLabelDomain,
        ResidentNullableRelationOutputColumn, ResidentNullableRelationOutputSource,
        ResidentNullableRelationRequest, ResidentNullableRelationResult,
        ResidentNullableRelationStage, ResidentNullableRelationshipDomain, ResidentProjectImage,
        ResidentSortRequest, ResidentSortResult, ResidentVectorQuery, ResidentVectorResult,
        ScratchReservation,
    },
    graph::{GraphMutation, GraphStore, IndexCatalog, LayerMask, TemporalStore},
    types::{LabelId, PropertyId},
};
use tokio_util::sync::CancellationToken;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::from_u128(
    0x4c41_4245_4c5f_5052_4544_4943_4154_4501,
));
const MEMORY_LIMIT_BYTES: usize = 128 * 1024 * 1024;
const RESERVED_BYTES: usize = 16 * 1024 * 1024;
const BOOLEAN_OUTPUT_BYTES_PER_ROW: usize = size_of::<u32>() + size_of::<u8>() + size_of::<u8>();

const RETURN2_SETUP: &[&str] = &["CREATE (), (:Foo)"];
const GRAPH5_NODE_SETUP: &[&str] =
    &["CREATE (:A:B:C), (:A:B), (:A:C), (:B:C), (:A), (:B), (:C), ()"];
const GRAPH5_RELATIONSHIP_SETUP: &[&str] =
    &["CREATE ()-[:T1]->(), ()-[:T2]->(), ()-[:t2]->(), (:T2)-[:T3]->(), ()-[:T4]->(:T2)"];
const GRAPH5_NULL_SETUP: &[&str] = &["CREATE (s:Single)"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExpectedDomain {
    NodeKnown(&'static [&'static str]),
    NodeKnownEmpty,
    RelationshipKnown(&'static str),
    RelationshipKnownEmpty,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SemanticProof {
    TruthCounts {
        true_rows: usize,
        false_rows: usize,
        null_rows: usize,
    },
    NodeLabels(&'static [&'static str]),
    RelationshipType(&'static str),
}

#[derive(Clone, Copy)]
struct Case {
    report_index: Option<usize>,
    feature: &'static str,
    scenario: &'static str,
    setup: &'static [&'static str],
    query: &'static str,
    expected_columns: usize,
    expected_entity_columns: usize,
    expected_domain: ExpectedDomain,
    semantic_proof: SemanticProof,
}

const CASES: [Case; 8] = [
    Case {
        report_index: Some(703),
        feature: "clauses/return/Return2.feature",
        scenario: "[8] Returning label predicate expression",
        setup: RETURN2_SETUP,
        query: "MATCH (n) RETURN (n:Foo)",
        expected_columns: 1,
        expected_entity_columns: 0,
        expected_domain: ExpectedDomain::NodeKnown(&["Foo"]),
        semantic_proof: SemanticProof::TruthCounts {
            true_rows: 1,
            false_rows: 1,
            null_rows: 0,
        },
    },
    Case {
        report_index: Some(1551),
        feature: "expressions/graph/Graph5.feature",
        scenario: "[1] Single-labels expression on nodes",
        setup: GRAPH5_NODE_SETUP,
        query: "MATCH (a) RETURN a, a:B AS result",
        expected_columns: 2,
        expected_entity_columns: 1,
        expected_domain: ExpectedDomain::NodeKnown(&["B"]),
        semantic_proof: SemanticProof::NodeLabels(&["B"]),
    },
    Case {
        report_index: Some(1552),
        feature: "expressions/graph/Graph5.feature",
        scenario: "[2] Single-labels expression on relationships",
        setup: GRAPH5_RELATIONSHIP_SETUP,
        query: "MATCH ()-[r]->() RETURN r, r:T2 AS result",
        expected_columns: 2,
        expected_entity_columns: 1,
        expected_domain: ExpectedDomain::RelationshipKnown("T2"),
        semantic_proof: SemanticProof::RelationshipType("T2"),
    },
    Case {
        report_index: Some(1553),
        feature: "expressions/graph/Graph5.feature",
        scenario: "[3] Conjunctive labels expression on nodes",
        setup: GRAPH5_NODE_SETUP,
        query: "MATCH (a) RETURN a, a:A:B AS result",
        expected_columns: 2,
        expected_entity_columns: 1,
        expected_domain: ExpectedDomain::NodeKnown(&["A", "B"]),
        semantic_proof: SemanticProof::NodeLabels(&["A", "B"]),
    },
    Case {
        report_index: Some(1559),
        feature: "expressions/graph/Graph5.feature",
        scenario: "[5] Label expression on null",
        setup: GRAPH5_NULL_SETUP,
        query: "MATCH (n:Single) OPTIONAL MATCH (n)-[r:TYPE]-(m) RETURN m:TYPE",
        expected_columns: 1,
        expected_entity_columns: 0,
        expected_domain: ExpectedDomain::NodeKnownEmpty,
        semantic_proof: SemanticProof::TruthCounts {
            true_rows: 0,
            false_rows: 0,
            null_rows: 1,
        },
    },
    Case {
        report_index: None,
        feature: "strict/entity-label-predicate",
        scenario: "repeated relationship type remains satisfiable",
        setup: GRAPH5_RELATIONSHIP_SETUP,
        query: "MATCH ()-[r]->() RETURN r:T2:T2 AS result",
        expected_columns: 1,
        expected_entity_columns: 0,
        expected_domain: ExpectedDomain::RelationshipKnown("T2"),
        semantic_proof: SemanticProof::TruthCounts {
            true_rows: 1,
            false_rows: 4,
            null_rows: 0,
        },
    },
    Case {
        report_index: None,
        feature: "strict/entity-label-predicate",
        scenario: "distinct relationship type conjunction is unsatisfiable",
        setup: GRAPH5_RELATIONSHIP_SETUP,
        query: "MATCH ()-[r]->() RETURN r:T2:T3 AS result",
        expected_columns: 1,
        expected_entity_columns: 0,
        expected_domain: ExpectedDomain::RelationshipKnownEmpty,
        semantic_proof: SemanticProof::TruthCounts {
            true_rows: 0,
            false_rows: 5,
            null_rows: 0,
        },
    },
    Case {
        report_index: None,
        feature: "strict/entity-label-predicate",
        scenario: "unknown node label is false for every non-null node",
        setup: RETURN2_SETUP,
        query: "MATCH (n) RETURN n:Missing AS result",
        expected_columns: 1,
        expected_entity_columns: 0,
        expected_domain: ExpectedDomain::NodeKnownEmpty,
        semantic_proof: SemanticProof::TruthCounts {
            true_rows: 0,
            false_rows: 2,
            null_rows: 0,
        },
    },
];

fn context<'a>(
    graph: &'a GraphStore,
    backend: Option<&'a dyn ExecutionBackend>,
    require_native_execution: bool,
) -> ExecutionContext<'a> {
    let next_node_id = graph
        .nodes()
        .map(|node| node.id().0.saturating_add(1))
        .max()
        .unwrap_or(1);
    let next_edge_id = graph
        .edges()
        .map(|edge| edge.id().0.saturating_add(1))
        .max()
        .unwrap_or(1);
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
        next_node_id,
        next_edge_id,
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
        deadline: Some(Instant::now() + Duration::from_secs(60)),
        resolved_query_at_time_nanos: None,
    }
}

fn apply_mutations(graph: &mut GraphStore, mutations: &[GraphMutation]) -> Result<()> {
    for mutation in mutations {
        graph.apply(mutation.clone())?;
    }
    Ok(())
}

fn fixture(case: Case) -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    for setup in case.setup {
        let output = QueryEngine.execute(setup, &mut context(&graph, None, false))?;
        apply_mutations(&mut graph, &output.graph_mutations)?;
    }
    Ok(graph)
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

fn result_rows(output: &ExecutionOutput) -> Result<Vec<Vec<ResultValue>>> {
    if !output.graph_mutations.is_empty() || !output.temporal_mutations.is_empty() {
        return Err(Error::internal(
            "read-only entity-label predicate query produced mutations",
        ));
    }
    let mut rows = Vec::new();
    for batch in &output.result.batches {
        if !batch.validate() || batch.columns.len() != output.result.schema.len() {
            return Err(Error::internal(
                "entity-label predicate query produced an invalid result batch",
            ));
        }
        for row in 0..batch.row_count {
            rows.push(
                batch
                    .columns
                    .iter()
                    .map(|column| column.values[row].clone())
                    .collect(),
            );
        }
    }
    Ok(rows)
}

fn assert_same_result(
    case: Case,
    expected: &ExecutionOutput,
    actual: &ExecutionOutput,
) -> Result<()> {
    if expected.result.schema != actual.result.schema
        || expected.result.statistics != actual.result.statistics
        || expected.result.truncated != actual.result.truncated
    {
        return Err(Error::internal(format!(
            "{} / {} changed result metadata: expected schema={:?} statistics={:?} truncated={}, actual schema={:?} statistics={:?} truncated={}",
            case.feature,
            case.scenario,
            expected.result.schema,
            expected.result.statistics,
            expected.result.truncated,
            actual.result.schema,
            actual.result.statistics,
            actual.result.truncated,
        )));
    }
    let mut expected_rows = result_rows(expected)?
        .into_iter()
        .map(|row| format!("{row:?}"))
        .collect::<Vec<_>>();
    let mut actual_rows = result_rows(actual)?
        .into_iter()
        .map(|row| format!("{row:?}"))
        .collect::<Vec<_>>();
    expected_rows.sort();
    actual_rows.sort();
    if expected_rows != actual_rows {
        return Err(Error::internal(format!(
            "{} / {} expected {expected_rows:?}, got {actual_rows:?}",
            case.feature, case.scenario
        )));
    }
    Ok(())
}

fn boolean_value(value: &ResultValue) -> Result<Option<bool>> {
    match value {
        ResultValue::Scalar(ScalarValue::Boolean(value)) => Ok(Some(*value)),
        ResultValue::Scalar(ScalarValue::Null) => Ok(None),
        other => Err(Error::internal(format!(
            "entity-label predicate published non-Boolean value {other:?}"
        ))),
    }
}

fn assert_semantics(case: Case, output: &ExecutionOutput) -> Result<()> {
    let rows = result_rows(output)?;
    match case.semantic_proof {
        SemanticProof::TruthCounts {
            true_rows,
            false_rows,
            null_rows,
        } => {
            let mut actual = [0_usize; 3];
            for row in &rows {
                match boolean_value(row.last().ok_or_else(|| {
                    Error::internal("entity-label predicate result row was empty")
                })?)? {
                    Some(true) => actual[0] += 1,
                    Some(false) => actual[1] += 1,
                    None => actual[2] += 1,
                }
            }
            assert_eq!(actual, [true_rows, false_rows, null_rows]);
        }
        SemanticProof::NodeLabels(required) => {
            for row in &rows {
                let [ResultValue::Node(node), predicate] = row.as_slice() else {
                    return Err(Error::internal(
                        "node label-predicate proof changed its two-column row shape",
                    ));
                };
                let expected = required
                    .iter()
                    .all(|label| node.labels.iter().any(|actual| actual == label));
                assert_eq!(boolean_value(predicate)?, Some(expected), "node={node:?}");
            }
        }
        SemanticProof::RelationshipType(required) => {
            for row in &rows {
                let [ResultValue::Relationship(relationship), predicate] = row.as_slice() else {
                    return Err(Error::internal(
                        "relationship type-predicate proof changed its two-column row shape",
                    ));
                };
                assert_eq!(
                    boolean_value(predicate)?,
                    Some(relationship.relationship_type == required),
                    "relationship={relationship:?}"
                );
            }
            assert!(
                rows.iter().any(|row| matches!(
                    row.as_slice(),
                    [ResultValue::Relationship(relationship), predicate]
                        if relationship.relationship_type == "T2"
                            && boolean_value(predicate).ok() == Some(Some(true))
                )),
                "exact-case T2 relationship did not evaluate true"
            );
            assert!(
                rows.iter().any(|row| matches!(
                    row.as_slice(),
                    [ResultValue::Relationship(relationship), predicate]
                        if relationship.relationship_type == "t2"
                            && boolean_value(predicate).ok() == Some(Some(false))
                )),
                "lowercase t2 relationship did not remain distinct"
            );
        }
    }
    Ok(())
}

fn expected_domain(
    case: Case,
    graph: &GraphStore,
) -> Result<ResidentNullableRelationEntityLabelDomain> {
    Ok(match case.expected_domain {
        ExpectedDomain::NodeKnown(names) => {
            let mut labels = names
                .iter()
                .map(|name| {
                    graph.catalog().label(name).ok_or_else(|| {
                        Error::internal(format!("fixture omitted node label `{name}`"))
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            labels.sort_unstable();
            ResidentNullableRelationEntityLabelDomain::Node(ResidentNullableNodeDomain::Known(
                labels,
            ))
        }
        ExpectedDomain::NodeKnownEmpty => {
            ResidentNullableRelationEntityLabelDomain::Node(ResidentNullableNodeDomain::KnownEmpty)
        }
        ExpectedDomain::RelationshipKnown(name) => {
            let relationship_type = graph.catalog().relationship_type(name).ok_or_else(|| {
                Error::internal(format!("fixture omitted relationship type `{name}`"))
            })?;
            ResidentNullableRelationEntityLabelDomain::Relationship(
                ResidentNullableRelationshipDomain::Known(vec![relationship_type]),
            )
        }
        ExpectedDomain::RelationshipKnownEmpty => {
            ResidentNullableRelationEntityLabelDomain::Relationship(
                ResidentNullableRelationshipDomain::KnownEmpty,
            )
        }
    })
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fault {
    None,
    ValueAboveOne,
    ValidityAboveOne,
    NonCanonicalNullValue,
    NullSourceMarkedValid,
    DomainTamper,
    FingerprintTamper,
}

#[derive(Default)]
struct Observations {
    pins: AtomicUsize,
    calls: AtomicUsize,
    forbidden: AtomicUsize,
    requests: Mutex<Vec<ResidentNullableRelationRequest>>,
}

struct StrictRecordingLabelBackend {
    inner: Box<dyn ExecutionBackend>,
    pinned_kind: BackendKind,
    pinned: bool,
    bookmark: Bookmark,
    graph_revision: u64,
    fault: Fault,
    observations: Arc<Observations>,
}

impl StrictRecordingLabelBackend {
    fn new(graph: &GraphStore, fault: Fault) -> Result<Self> {
        let mut cpu = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
        cpu.admit_project(resident_image(graph)?)?;
        let bookmark = cpu
            .resident_bookmark(PROJECT)
            .ok_or_else(|| Error::internal("CPU label-predicate fixture omitted its bookmark"))?;
        let graph_revision = cpu.resident_graph_revision(PROJECT).ok_or_else(|| {
            Error::internal("CPU label-predicate fixture omitted its graph revision")
        })?;
        Ok(Self {
            inner: Box::new(cpu),
            pinned_kind: BackendKind::Cpu,
            pinned: false,
            bookmark,
            graph_revision,
            fault,
            observations: Arc::new(Observations::default()),
        })
    }

    fn observations(&self) -> Arc<Observations> {
        Arc::clone(&self.observations)
    }

    #[cfg(all(feature = "accelerator", target_os = "macos"))]
    fn real_metal(graph: &GraphStore) -> Result<Self> {
        let mut metal = irongraph::gpu::MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
        if metal.kind() != BackendKind::Metal {
            return Err(Error::internal(
                "label-predicate hardware gate did not construct a real Metal backend",
            ));
        }
        metal.admit_project(resident_image(graph)?)?;
        let bookmark = metal
            .resident_bookmark(PROJECT)
            .ok_or_else(|| Error::internal("Metal label-predicate fixture omitted its bookmark"))?;
        let graph_revision = metal.resident_graph_revision(PROJECT).ok_or_else(|| {
            Error::internal("Metal label-predicate fixture omitted its graph revision")
        })?;
        Ok(Self {
            inner: Box::new(metal),
            pinned_kind: BackendKind::Metal,
            pinned: false,
            bookmark,
            graph_revision,
            fault: Fault::None,
            observations: Arc::new(Observations::default()),
        })
    }

    fn reject<T>(&self, route: &'static str) -> Result<T> {
        self.observations.forbidden.fetch_add(1, Ordering::SeqCst);
        Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!("strict label-predicate gate rejected `{route}`"),
        ))
    }
}

impl ExecutionBackend for StrictRecordingLabelBackend {
    fn kind(&self) -> BackendKind {
        if self.pinned {
            self.pinned_kind
        } else {
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
            return self.reject("pin_project");
        }
        let inner = self.inner.pin_project(project)?;
        if inner.kind() != self.pinned_kind
            || inner.resident_bookmark(project) != Some(self.bookmark)
            || inner.resident_graph_revision(project) != Some(self.graph_revision)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "label-predicate pin changed its backend kind or immutable generation",
            ));
        }
        self.observations.pins.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Self {
            inner,
            pinned_kind: self.pinned_kind,
            pinned: true,
            bookmark: self.bookmark,
            graph_revision: self.graph_revision,
            fault: self.fault,
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

    fn supports_nullable_relation_predicates(&self) -> bool {
        self.inner.supports_nullable_relation_predicates()
    }

    fn execute_nullable_relation(
        &self,
        request: &ResidentNullableRelationRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentNullableRelationResult> {
        if !self.pinned {
            return self.reject("execute_nullable_relation_on_root");
        }
        request.validate()?;
        if request.generation.project != PROJECT
            || request.generation.bookmark != self.bookmark
            || request.generation.graph_revision != self.graph_revision
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "label-predicate request escaped its pinned generation",
            ));
        }
        self.observations.calls.fetch_add(1, Ordering::SeqCst);
        self.observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(request.clone());
        let result = self
            .inner
            .execute_nullable_relation(request, cancellation)?;
        if self.fault == Fault::None {
            return Ok(result);
        }
        let mut parts = result.into_untrusted_parts();
        if self.fault == Fault::FingerprintTamper {
            parts.fingerprint.0[0] ^= 1;
            return Ok(ResidentNullableRelationResult::from_untrusted_parts(parts));
        }
        let (domain, values, validity) = parts
            .columns
            .iter_mut()
            .find_map(|column| match column {
                ResidentNullableRelationOutputColumn::EntityLabelPredicate {
                    domain,
                    values,
                    validity,
                    ..
                } => Some((domain, values, validity)),
                _ => None,
            })
            .ok_or_else(|| Error::internal("label-predicate fault did not observe its column"))?;
        match self.fault {
            Fault::ValueAboveOne => {
                let target = validity
                    .iter()
                    .position(|valid| *valid == 1)
                    .ok_or_else(|| Error::internal("label-predicate fault found no valid row"))?;
                values[target] = 2;
            }
            Fault::ValidityAboveOne => {
                let target = validity
                    .iter()
                    .position(|valid| *valid == 1)
                    .ok_or_else(|| Error::internal("label-predicate fault found no valid row"))?;
                validity[target] = 2;
            }
            Fault::NonCanonicalNullValue => {
                let target = validity
                    .iter()
                    .position(|valid| *valid == 0)
                    .ok_or_else(|| Error::internal("label-predicate fault found no null row"))?;
                values[target] = 1;
            }
            Fault::NullSourceMarkedValid => {
                let target = validity
                    .iter()
                    .position(|valid| *valid == 0)
                    .ok_or_else(|| Error::internal("label-predicate fault found no null row"))?;
                validity[target] = 1;
            }
            Fault::DomainTamper => {
                *domain = ResidentNullableRelationEntityLabelDomain::Node(
                    ResidentNullableNodeDomain::KnownEmpty,
                );
            }
            Fault::None | Fault::FingerprintTamper => unreachable!(),
        }
        Ok(ResidentNullableRelationResult::from_untrusted_parts(parts))
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

fn assert_request(case: Case, graph: &GraphStore, observations: &Observations) -> Result<()> {
    if observations.pins.load(Ordering::SeqCst) != 1
        || observations.calls.load(Ordering::SeqCst) != 1
        || observations.forbidden.load(Ordering::SeqCst) != 0
    {
        return Err(Error::internal(format!(
            "{} / {} did not use exactly one pinned nullable-relation command",
            case.feature, case.scenario
        )));
    }
    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let [request] = requests.as_slice() else {
        return Err(Error::internal(format!(
            "{} / {} changed its native request count",
            case.feature, case.scenario
        )));
    };
    request.validate()?;
    let expected_bytes = case
        .expected_entity_columns
        .checked_mul(size_of::<u32>())
        .and_then(|bytes| bytes.checked_add(BOOLEAN_OUTPUT_BYTES_PER_ROW))
        .ok_or_else(|| Error::internal("label-predicate expected packet bytes overflowed"))?;
    if BOOLEAN_OUTPUT_BYTES_PER_ROW != 6
        || !request.property_lanes().is_empty()
        || !request.predicate_program.is_empty()
        || request.capacities.final_output_columns as usize != case.expected_columns
        || request.capacities.final_output_bytes_per_row != expected_bytes as u64
        || request.capacities.final_output_fixed_bytes != 0
    {
        return Err(Error::internal(format!(
            "{} / {} changed its sealed output packet shape",
            case.feature, case.scenario
        )));
    }
    let Some(ResidentNullableRelationStage::FinalProject { bindings }) =
        request.program.stages.last()
    else {
        return Err(Error::internal(
            "label-predicate request omitted its final projection",
        ));
    };
    if request
        .program
        .stages
        .iter()
        .filter(|stage| matches!(stage, ResidentNullableRelationStage::FinalProject { .. }))
        .count()
        != 1
        || bindings.len() != case.expected_columns
    {
        return Err(Error::internal(
            "label-predicate request changed its single final projection boundary",
        ));
    }
    let predicate_sources = bindings
        .iter()
        .filter_map(|binding| match &binding.source {
            ResidentNullableRelationOutputSource::EntityLabelPredicate { slot, domain } => {
                Some((*slot, domain))
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    let [(predicate_slot, domain)] = predicate_sources.as_slice() else {
        return Err(Error::internal(
            "label-predicate request did not seal exactly one Boolean output",
        ));
    };
    let expected_domain = expected_domain(case, graph)?;
    if *domain != &expected_domain {
        return Err(Error::internal(format!(
            "{} / {} compiled the wrong canonical label/type domain: {domain:?}",
            case.feature, case.scenario
        )));
    }
    let entity_sources = bindings
        .iter()
        .filter_map(|binding| match &binding.source {
            ResidentNullableRelationOutputSource::Entity { slot, kind } => Some((*slot, *kind)),
            ResidentNullableRelationOutputSource::EntityLabelPredicate { .. } => None,
            _ => Some((
                irongraph::gpu::ResidentNullableRelationSlot(u16::MAX),
                ResidentNullableRelationBindingKind::Node,
            )),
        })
        .collect::<Vec<_>>();
    if entity_sources.len() != case.expected_entity_columns
        || entity_sources
            .iter()
            .any(|(slot, kind)| *slot != *predicate_slot || *kind != domain.kind())
    {
        return Err(Error::internal(format!(
            "{} / {} retained an unrelated or misaligned output source",
            case.feature, case.scenario
        )));
    }
    Ok(())
}

#[test]
fn exact_entity_label_predicate_manifest_is_stable() {
    assert_eq!(BOOLEAN_OUTPUT_BYTES_PER_ROW, 6);
    assert_eq!(CASES.len(), 8);
    assert_eq!(
        CASES
            .iter()
            .filter_map(|case| case.report_index)
            .collect::<Vec<_>>(),
        [703, 1551, 1552, 1553, 1559]
    );
    assert_eq!(
        CASES
            .iter()
            .map(|case| (case.feature, case.scenario))
            .collect::<BTreeSet<_>>()
            .len(),
        CASES.len()
    );
}

#[test]
fn native_cpu_executes_all_tck_and_adversarial_label_predicates_as_one_sealed_command() -> Result<()>
{
    for case in CASES {
        let graph = fixture(case)?;
        let expected = QueryEngine.execute(case.query, &mut context(&graph, None, false))?;
        let backend = StrictRecordingLabelBackend::new(&graph, Fault::None)?;
        let observations = backend.observations();
        let actual = QueryEngine.execute(case.query, &mut context(&graph, Some(&backend), true))?;
        assert_same_result(case, &expected, &actual)?;
        assert_eq!(actual.result.statistics, StatementStats::default());
        assert_semantics(case, &actual)?;
        assert_request(case, &graph, &observations)?;
    }
    Ok(())
}

#[test]
fn label_predicate_publication_rejects_noncanonical_booleans_nulls_domains_and_fingerprint()
-> Result<()> {
    for (case, fault) in [
        (CASES[0], Fault::ValueAboveOne),
        (CASES[0], Fault::ValidityAboveOne),
        (CASES[4], Fault::NonCanonicalNullValue),
        (CASES[4], Fault::NullSourceMarkedValid),
        (CASES[0], Fault::DomainTamper),
        (CASES[0], Fault::FingerprintTamper),
    ] {
        let graph = fixture(case)?;
        let backend = StrictRecordingLabelBackend::new(&graph, fault)?;
        let observations = backend.observations();
        let error = QueryEngine
            .execute(case.query, &mut context(&graph, Some(&backend), true))
            .expect_err("forged label-predicate result must fail publication");
        assert_eq!(error.code, ErrorCode::CorruptStorage, "fault={fault:?}");
        assert_eq!(observations.pins.load(Ordering::SeqCst), 1);
        assert_eq!(observations.calls.load(Ordering::SeqCst), 1);
        assert_eq!(observations.forbidden.load(Ordering::SeqCst), 0);
    }
    Ok(())
}

#[test]
fn label_predicate_request_fingerprint_seals_the_complete_domain() -> Result<()> {
    let case = CASES[0];
    let graph = fixture(case)?;
    let backend = StrictRecordingLabelBackend::new(&graph, Fault::None)?;
    let observations = backend.observations();
    QueryEngine.execute(case.query, &mut context(&graph, Some(&backend), true))?;
    let request = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .first()
        .cloned()
        .ok_or_else(|| Error::internal("label-predicate request was not recorded"))?;

    let mut changed_domain = request.clone();
    let Some(ResidentNullableRelationStage::FinalProject { bindings }) =
        changed_domain.program.stages.last_mut()
    else {
        return Err(Error::internal(
            "label-predicate request omitted its final projection",
        ));
    };
    let domain = bindings
        .iter_mut()
        .find_map(|binding| match &mut binding.source {
            ResidentNullableRelationOutputSource::EntityLabelPredicate { domain, .. } => {
                Some(domain)
            }
            _ => None,
        })
        .ok_or_else(|| Error::internal("label-predicate output source disappeared"))?;
    *domain =
        ResidentNullableRelationEntityLabelDomain::Node(ResidentNullableNodeDomain::KnownEmpty);
    let error = changed_domain
        .validate()
        .expect_err("post-seal label-domain mutation must fail admission");
    assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
    assert_eq!(
        error.message.as_ref(),
        "resident nullable relation fingerprint does not match its immutable request"
    );

    let mut changed_fingerprint = request;
    changed_fingerprint.manifest.fingerprint.0[0] ^= 1;
    let error = changed_fingerprint
        .validate()
        .expect_err("forged label-predicate fingerprint must fail admission");
    assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
    assert_eq!(
        error.message.as_ref(),
        "resident nullable relation fingerprint does not match its immutable request"
    );
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "hardware acceptance gate: requires a real Metal device"]
fn real_metal_matches_cpu_for_every_tck_and_adversarial_label_predicate_without_fallback()
-> Result<()> {
    static METAL_TEST: Mutex<()> = Mutex::new(());
    let _guard = METAL_TEST
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    for case in CASES {
        let graph = fixture(case)?;
        let expected = QueryEngine.execute(case.query, &mut context(&graph, None, false))?;
        let backend = StrictRecordingLabelBackend::real_metal(&graph)?;
        let observations = backend.observations();
        let actual = QueryEngine.execute(case.query, &mut context(&graph, Some(&backend), true))?;
        assert_same_result(case, &expected, &actual)?;
        assert_semantics(case, &actual)?;
        assert_request(case, &graph, &observations)?;
    }
    Ok(())
}
