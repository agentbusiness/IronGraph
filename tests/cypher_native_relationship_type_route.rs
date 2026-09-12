// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use irongraph::{
    Bookmark, Error, ErrorCode, ProjectId, Result, ScalarValue,
    cypher::{BindCapabilities, ExecutionContext, ExecutionOutput, QueryEngine, ResultValue},
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend,
        RESIDENT_NULLABLE_RELATION_NULL_ROW, ResidentGroup, ResidentGroupRequest, ResidentJoinPair,
        ResidentJoinRequest, ResidentNodePipelineRequest, ResidentNodePipelineResult,
        ResidentNullableRelationFilterPlacement, ResidentNullableRelationMatchMode,
        ResidentNullableRelationOutputColumn, ResidentNullableRelationOutputSource,
        ResidentNullableRelationPredicate, ResidentNullableRelationPredicateValue,
        ResidentNullableRelationRequest, ResidentNullableRelationResult,
        ResidentNullableRelationStage, ResidentNullableRelationshipDomain, ResidentProjectImage,
        ResidentSortRequest, ResidentSortResult, ResidentVectorQuery, ResidentVectorResult,
        ScratchReservation,
    },
    graph::{GraphMutation, GraphStore, IndexCatalog, LayerMask, TemporalStore},
    types::{LabelId, PropertyId, RelationshipTypeId},
};
use tokio_util::sync::CancellationToken;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::from_u128(
    0x5459_5045_5f52_454c_4154_494f_4e5f_5443,
));
const MEMORY_LIMIT_BYTES: usize = 128 * 1024 * 1024;
const RESERVED_BYTES: usize = 16 * 1024 * 1024;

const LOOP_SETUP: &[&str] = &["CREATE (a) CREATE (a)-[:T]->(a)"];
const TYPE_SETUP: &[&str] = &["CREATE ()-[:T]->()"];
const TWO_TYPE_SETUP: &[&str] = &["CREATE ()-[:T1]->()-[:T2]->()"];
const NULL_TYPE_SETUP: &[&str] = &["CREATE ()"];
const FIXED_PATH_SETUP: &[&str] = &["CREATE (a:A {name: 'A'})-[:KNOWS]->(b:B {name: 'B'})"];

#[derive(Clone, Copy)]
struct Case {
    report_index: usize,
    feature: &'static str,
    scenario: &'static str,
    setup: &'static [&'static str],
    query: &'static str,
    output_columns: usize,
}

const CASES: [Case; 7] = [
    Case {
        report_index: 259,
        feature: "clauses/match/Match2.feature",
        scenario: "[3] Matching a self-loop with an undirected relationship pattern",
        setup: LOOP_SETUP,
        query: "MATCH ()-[r]-() RETURN type(r) AS r",
        output_columns: 1,
    },
    Case {
        report_index: 260,
        feature: "clauses/match/Match2.feature",
        scenario: "[4] Matching a self-loop with a directed relationship pattern",
        setup: LOOP_SETUP,
        query: "MATCH ()-[r]->() RETURN type(r) AS r",
        output_columns: 1,
    },
    Case {
        report_index: 1540,
        feature: "expressions/graph/Graph4.feature",
        scenario: "[1] `type()`",
        setup: TYPE_SETUP,
        query: "MATCH ()-[r]->() RETURN type(r)",
        output_columns: 1,
    },
    Case {
        report_index: 1541,
        feature: "expressions/graph/Graph4.feature",
        scenario: "[2] `type()` on two relationships",
        setup: TWO_TYPE_SETUP,
        query: "MATCH ()-[r1]->()-[r2]->() RETURN type(r1), type(r2)",
        output_columns: 2,
    },
    Case {
        report_index: 1542,
        feature: "expressions/graph/Graph4.feature",
        scenario: "[3] `type()` on null relationship",
        setup: NULL_TYPE_SETUP,
        query: "MATCH (a) OPTIONAL MATCH (a)-[r:NOT_THERE]->() RETURN type(r), type(null)",
        output_columns: 2,
    },
    Case {
        report_index: 1543,
        feature: "expressions/graph/Graph4.feature",
        scenario: "[4] `type()` on mixed null and non-null relationships",
        setup: TYPE_SETUP,
        query: "MATCH (a) OPTIONAL MATCH (a)-[r:T]->() RETURN type(r)",
        output_columns: 1,
    },
    Case {
        report_index: 1544,
        feature: "expressions/graph/Graph4.feature",
        scenario: "[5] `type()` handling Any type",
        setup: TYPE_SETUP,
        query: "MATCH (a)-[r]->() WITH [r, 1] AS list RETURN type(list[0])",
        output_columns: 1,
    },
];

#[derive(Clone, Copy)]
struct FixedPathCase {
    report_index: usize,
    feature: &'static str,
    scenario: &'static str,
    query: &'static str,
    expected_length: i64,
    expected_rows: usize,
}

const FIXED_PATH_CASES: [FixedPathCase; 2] = [
    FixedPathCase {
        report_index: 563,
        feature: "clauses/match-where/MatchWhere1.feature",
        scenario: "[12] Filter path with path length predicate on multi variables with one binding",
        query: "MATCH p = (n)-->(x) WHERE length(p) = 1 RETURN x",
        expected_length: 1,
        expected_rows: 1,
    },
    FixedPathCase {
        report_index: 564,
        feature: "clauses/match-where/MatchWhere1.feature",
        scenario: "[13] Filter path with false path length predicate on multi variables with one binding",
        query: "MATCH p = (n)-->(x) WHERE length(p) = 10 RETURN x",
        expected_length: 10,
        expected_rows: 0,
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
            term: 53,
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

fn fixed_path_fixture() -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    for setup in FIXED_PATH_SETUP {
        let output = QueryEngine.execute(setup, &mut context(&graph, None, false))?;
        apply_mutations(&mut graph, &output.graph_mutations)?;
    }
    Ok(graph)
}

fn resident_image(graph: &GraphStore) -> Result<ResidentProjectImage> {
    ResidentProjectImage::build(
        PROJECT,
        Bookmark {
            term: 53,
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
            "read-only relationship-type query produced mutations",
        ));
    }
    let mut rows = Vec::new();
    for batch in &output.result.batches {
        if !batch.validate() || batch.columns.len() != output.result.schema.len() {
            return Err(Error::internal(
                "relationship-type query produced an invalid result batch",
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

fn assert_same_fixed_path_result(
    case: FixedPathCase,
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
    if expected_rows != actual_rows || actual_rows.len() != case.expected_rows {
        return Err(Error::internal(format!(
            "{} / {} expected {expected_rows:?}, got {actual_rows:?}",
            case.feature, case.scenario
        )));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Fault {
    None,
    WrongNonNullToken,
    NonCanonicalNullToken,
    SecondNonCanonicalNullToken,
}

#[derive(Default)]
struct Observations {
    pins: AtomicUsize,
    calls: AtomicUsize,
    forbidden: AtomicUsize,
    requests: Mutex<Vec<ResidentNullableRelationRequest>>,
}

struct StrictRelationshipTypeBackend {
    inner: Box<dyn ExecutionBackend>,
    pinned_kind: BackendKind,
    pinned: bool,
    bookmark: Bookmark,
    graph_revision: u64,
    fault: Fault,
    observations: Arc<Observations>,
}

impl StrictRelationshipTypeBackend {
    fn new(graph: &GraphStore, fault: Fault) -> Result<Self> {
        let mut cpu = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
        cpu.admit_project(resident_image(graph)?)?;
        let bookmark = cpu
            .resident_bookmark(PROJECT)
            .ok_or_else(|| Error::internal("CPU type() fixture omitted its bookmark"))?;
        let graph_revision = cpu
            .resident_graph_revision(PROJECT)
            .ok_or_else(|| Error::internal("CPU type() fixture omitted its graph revision"))?;
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
                "relationship-type hardware gate did not construct a real Metal backend",
            ));
        }
        metal.admit_project(resident_image(graph)?)?;
        let bookmark = metal
            .resident_bookmark(PROJECT)
            .ok_or_else(|| Error::internal("Metal type() fixture omitted its bookmark"))?;
        let graph_revision = metal
            .resident_graph_revision(PROJECT)
            .ok_or_else(|| Error::internal("Metal type() fixture omitted its graph revision"))?;
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
            format!("strict relationship-type gate rejected `{route}`"),
        ))
    }
}

impl ExecutionBackend for StrictRelationshipTypeBackend {
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
                "relationship-type pin changed its backend kind or immutable generation",
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
        if request.generation.project != PROJECT
            || request.generation.bookmark != self.bookmark
            || request.generation.graph_revision != self.graph_revision
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "relationship-type request escaped its pinned generation",
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
        let column_index = usize::from(self.fault == Fault::SecondNonCanonicalNullToken);
        let column = parts
            .columns
            .iter_mut()
            .filter_map(|column| match column {
                ResidentNullableRelationOutputColumn::RelationshipType {
                    source_rows,
                    relationship_types,
                    ..
                } => Some((source_rows, relationship_types)),
                _ => None,
            })
            .nth(column_index)
            .ok_or_else(|| Error::internal("type() fault did not observe its output column"))?;
        let target = match self.fault {
            Fault::WrongNonNullToken => column
                .0
                .iter()
                .position(|row| *row != RESIDENT_NULLABLE_RELATION_NULL_ROW),
            Fault::NonCanonicalNullToken | Fault::SecondNonCanonicalNullToken => column
                .0
                .iter()
                .position(|row| *row == RESIDENT_NULLABLE_RELATION_NULL_ROW),
            Fault::None => None,
        }
        .ok_or_else(|| Error::internal("type() fault could not find its target row"))?;
        column.1[target] = match self.fault {
            Fault::WrongNonNullToken => RelationshipTypeId(column.1[target].0 ^ u64::MAX),
            Fault::NonCanonicalNullToken | Fault::SecondNonCanonicalNullToken => {
                RelationshipTypeId(u64::MAX)
            }
            Fault::None => column.1[target],
        };
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

fn assert_request(case: Case, observations: &Observations) -> Result<()> {
    if observations.pins.load(Ordering::SeqCst) != 1
        || observations.calls.load(Ordering::SeqCst) != 1
        || observations.forbidden.load(Ordering::SeqCst) != 0
    {
        return Err(Error::internal(format!(
            "{} / {} did not use exactly one sealed nullable-relation command",
            case.feature, case.scenario
        )));
    }
    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let request = requests
        .first()
        .ok_or_else(|| Error::internal("relationship-type request was not recorded"))?;
    request.validate()?;
    if requests.len() != 1
        || !request.property_lanes().is_empty()
        || request.capacities.final_output_columns as usize != case.output_columns
        || request.capacities.final_output_bytes_per_row
            != ((std::mem::size_of::<u32>() + std::mem::size_of::<u64>()) * case.output_columns)
                as u64
    {
        return Err(Error::internal(format!(
            "{} / {} changed its sealed type() output shape",
            case.feature, case.scenario
        )));
    }
    let sources = request
        .program
        .stages
        .last()
        .and_then(|stage| match stage {
            irongraph::gpu::ResidentNullableRelationStage::FinalProject { bindings } => Some(
                bindings
                    .iter()
                    .map(|binding| binding.source.clone())
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        })
        .ok_or_else(|| Error::internal("type() request omitted its final projection"))?;
    if sources.len() != case.output_columns
        || sources.iter().any(|source| {
            !matches!(
                source,
                ResidentNullableRelationOutputSource::RelationshipType { .. }
            )
        })
    {
        return Err(Error::internal(format!(
            "{} / {} did not seal every type() column",
            case.feature, case.scenario
        )));
    }
    if case.report_index == 1542 {
        let [
            ResidentNullableRelationStage::NodeScan { .. },
            ResidentNullableRelationStage::Expand {
                mode: ResidentNullableRelationMatchMode::Optional,
                relationship: Some(relationship),
                relationship_types: ResidentNullableRelationshipDomain::KnownEmpty,
                ..
            },
            ResidentNullableRelationStage::FinalProject { bindings },
        ] = request.program.stages.as_slice()
        else {
            return Err(Error::internal(
                "Graph4 [3] lost its known-empty relationship-domain proof",
            ));
        };
        let [relationship_type, literal_null_type] = bindings.as_slice() else {
            return Err(Error::internal(
                "Graph4 [3] changed its two-column projection",
            ));
        };
        if relationship_type.name != "type(r)"
            || literal_null_type.name != "type(null)"
            || relationship_type.source
                != (ResidentNullableRelationOutputSource::RelationshipType {
                    slot: *relationship,
                })
            || literal_null_type.source
                != (ResidentNullableRelationOutputSource::RelationshipType {
                    slot: *relationship,
                })
        {
            return Err(Error::internal(
                "Graph4 [3] did not reuse its proven-null relationship type slot",
            ));
        }
    }
    if case.report_index == 1543 {
        let [
            ResidentNullableRelationStage::NodeScan { .. },
            ResidentNullableRelationStage::Expand {
                mode: ResidentNullableRelationMatchMode::Optional,
                relationship: Some(relationship),
                relationship_types: ResidentNullableRelationshipDomain::Known(types),
                ..
            },
            ResidentNullableRelationStage::FinalProject { bindings },
        ] = request.program.stages.as_slice()
        else {
            return Err(Error::internal(
                "Graph4 [4] changed its sealed optional-expansion shape",
            ));
        };
        let [binding] = bindings.as_slice() else {
            return Err(Error::internal(
                "Graph4 [4] changed its one-column projection",
            ));
        };
        if types.len() != 1
            || binding.name != "type(r)"
            || binding.source
                != (ResidentNullableRelationOutputSource::RelationshipType {
                    slot: *relationship,
                })
        {
            return Err(Error::internal(
                "Graph4 [4] did not preserve its mixed relationship type slot",
            ));
        }
    }
    if case.report_index == 1544 {
        let [
            ResidentNullableRelationStage::NodeScan { .. },
            ResidentNullableRelationStage::Expand {
                mode: ResidentNullableRelationMatchMode::Mandatory,
                relationship: Some(relationship),
                relationship_types: ResidentNullableRelationshipDomain::Any,
                ..
            },
            ResidentNullableRelationStage::ScopeProject {
                bindings: scope_bindings,
            },
            ResidentNullableRelationStage::FinalProject {
                bindings: output_bindings,
            },
        ] = request.program.stages.as_slice()
        else {
            return Err(Error::internal(
                "Graph4 [5] changed its sealed relationship-list projection shape",
            ));
        };
        let [scope_binding] = scope_bindings.as_slice() else {
            return Err(Error::internal(
                "Graph4 [5] changed its one-column WITH boundary",
            ));
        };
        let [output_binding] = output_bindings.as_slice() else {
            return Err(Error::internal(
                "Graph4 [5] changed its one-column final projection",
            ));
        };
        if scope_binding.variable != "list"
            || scope_binding.source != *relationship
            || scope_binding.row_limit.is_some()
            || output_binding.name != "type(list[0])"
            || output_binding.source
                != (ResidentNullableRelationOutputSource::RelationshipType {
                    slot: scope_binding.output,
                })
        {
            return Err(Error::internal(
                "Graph4 [5] did not preserve the statically selected relationship slot",
            ));
        }
    }
    Ok(())
}

fn assert_fixed_path_request(case: FixedPathCase, observations: &Observations) -> Result<()> {
    if observations.pins.load(Ordering::SeqCst) != 1
        || observations.calls.load(Ordering::SeqCst) != 1
        || observations.forbidden.load(Ordering::SeqCst) != 0
    {
        return Err(Error::internal(format!(
            "{} / {} did not use exactly one sealed nullable-relation command",
            case.feature, case.scenario
        )));
    }
    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let request = requests
        .first()
        .ok_or_else(|| Error::internal("fixed-path request was not recorded"))?;
    request.validate()?;
    if requests.len() != 1
        || !request.property_lanes().is_empty()
        || request.capacities.final_output_columns != 1
        || request.capacities.final_output_bytes_per_row != std::mem::size_of::<u32>() as u64
        || !matches!(
            request.program.stages.as_slice(),
            [
                ResidentNullableRelationStage::NodeScan { .. },
                ResidentNullableRelationStage::Expand { .. },
                ResidentNullableRelationStage::FinalProject { bindings },
            ] if matches!(
                bindings.as_slice(),
                [irongraph::gpu::ResidentNullableRelationOutputBinding {
                    source: ResidentNullableRelationOutputSource::Entity {
                        kind: irongraph::gpu::ResidentNullableRelationBindingKind::Node,
                        ..
                    },
                    ..
                }]
            )
        )
    {
        return Err(Error::internal(format!(
            "{} / {} changed its sealed fixed-path relation shape",
            case.feature, case.scenario
        )));
    }
    let [filter] = request.predicate_program.filters.as_slice() else {
        return Err(Error::internal(format!(
            "{} / {} did not emit exactly one path-length filter",
            case.feature, case.scenario
        )));
    };
    if filter.placement != (ResidentNullableRelationFilterPlacement::RelationAfter { stage: 1 })
        || !matches!(
            &filter.predicate,
            ResidentNullableRelationPredicate::CompareInteger {
                left: ResidentNullableRelationPredicateValue::Integer(1),
                operation: CompareOp::Eq,
                right: ResidentNullableRelationPredicateValue::Integer(value),
            } if *value == case.expected_length
        )
    {
        return Err(Error::internal(format!(
            "{} / {} did not seal the exact fixed-path integer comparison",
            case.feature, case.scenario
        )));
    }
    Ok(())
}

#[test]
fn exact_type_scenario_manifest_is_stable() {
    assert_eq!(CASES.len(), 7);
    assert_eq!(
        CASES.map(|case| case.report_index),
        [259, 260, 1540, 1541, 1542, 1543, 1544]
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
fn exact_fixed_path_length_scenario_manifest_is_stable() {
    assert_eq!(FIXED_PATH_CASES.len(), 2);
    assert_eq!(FIXED_PATH_CASES.map(|case| case.report_index), [563, 564]);
    assert_eq!(
        FIXED_PATH_CASES
            .iter()
            .map(|case| (case.feature, case.scenario))
            .collect::<BTreeSet<_>>()
            .len(),
        FIXED_PATH_CASES.len()
    );
}

#[test]
fn native_cpu_reference_executes_both_fixed_path_length_scenarios_as_one_sealed_command()
-> Result<()> {
    let graph = fixed_path_fixture()?;
    for case in FIXED_PATH_CASES {
        let expected = QueryEngine.execute(case.query, &mut context(&graph, None, false))?;
        let backend = StrictRelationshipTypeBackend::new(&graph, Fault::None)?;
        let observations = backend.observations();
        let actual = QueryEngine.execute(case.query, &mut context(&graph, Some(&backend), true))?;
        assert_same_fixed_path_result(case, &expected, &actual)?;
        assert_fixed_path_request(case, &observations)?;
    }
    Ok(())
}

#[test]
fn native_cpu_reference_executes_all_seven_type_scenarios_as_one_sealed_command() -> Result<()> {
    for case in CASES {
        let graph = fixture(case)?;
        let expected = QueryEngine.execute(case.query, &mut context(&graph, None, false))?;
        let backend = StrictRelationshipTypeBackend::new(&graph, Fault::None)?;
        let observations = backend.observations();
        let actual = QueryEngine.execute(case.query, &mut context(&graph, Some(&backend), true))?;
        assert_same_result(case, &expected, &actual)?;
        assert_request(case, &observations)?;
        if case.report_index == 1542 {
            assert_eq!(
                result_rows(&actual)?,
                [vec![
                    ResultValue::Scalar(ScalarValue::Null),
                    ResultValue::Scalar(ScalarValue::Null),
                ]],
                "Graph4 [3] changed its exact null row"
            );
        }
        if case.report_index == 1543 {
            let rows = result_rows(&actual)?;
            assert_eq!(rows.len(), 2, "Graph4 [4] changed its row count");
            assert!(
                rows.contains(&vec![ResultValue::Scalar(ScalarValue::String("T".into()))]),
                "Graph4 [4] omitted its non-null relationship type"
            );
            assert!(
                rows.contains(&vec![ResultValue::Scalar(ScalarValue::Null)]),
                "Graph4 [4] omitted its OPTIONAL null relationship type"
            );
        }
        if case.report_index == 1544 {
            assert_eq!(
                result_rows(&actual)?,
                [vec![ResultValue::Scalar(ScalarValue::String("T".into()))]],
                "Graph4 [5] changed its exact Any/list-index result"
            );
        }
    }
    Ok(())
}

#[test]
fn native_type_publication_rejects_wrong_and_noncanonical_tokens() -> Result<()> {
    for (case, fault) in [
        (CASES[2], Fault::WrongNonNullToken),
        (CASES[5], Fault::WrongNonNullToken),
        (CASES[5], Fault::NonCanonicalNullToken),
        (CASES[4], Fault::SecondNonCanonicalNullToken),
        (CASES[6], Fault::WrongNonNullToken),
    ] {
        let graph = fixture(case)?;
        let backend = StrictRelationshipTypeBackend::new(&graph, fault)?;
        let observations = backend.observations();
        let error = QueryEngine
            .execute(case.query, &mut context(&graph, Some(&backend), true))
            .expect_err("forged relationship-type token must fail publication");
        assert_eq!(error.code, ErrorCode::CorruptStorage);
        assert_eq!(observations.pins.load(Ordering::SeqCst), 1);
        assert_eq!(observations.calls.load(Ordering::SeqCst), 1);
        assert_eq!(observations.forbidden.load(Ordering::SeqCst), 0);
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "hardware acceptance gate: requires a real Metal device"]
fn real_metal_projects_canonical_relationship_type_tokens_for_all_seven_scenarios() -> Result<()> {
    static METAL_TEST: Mutex<()> = Mutex::new(());
    let _guard = METAL_TEST
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    for case in CASES {
        let graph = fixture(case)?;
        let expected = QueryEngine.execute(case.query, &mut context(&graph, None, false))?;
        let backend = StrictRelationshipTypeBackend::real_metal(&graph)?;
        let observations = backend.observations();
        let actual = QueryEngine.execute(case.query, &mut context(&graph, Some(&backend), true))?;
        assert_same_result(case, &expected, &actual)?;
        assert_request(case, &observations)?;
    }
    Ok(())
}
