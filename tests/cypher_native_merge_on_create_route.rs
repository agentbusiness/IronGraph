// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Focused acceptance gate for authoritative full-report rows 645 and 646, the two Merge6
//! scenarios whose success does not also depend on the separate `keys(r)` control-query route.

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
    cypher::{
        BindCapabilities, ColumnType, ExecutionContext, QueryEngine, ResultValue, StatementStats,
    },
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend, ResidentGroup,
        ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest, ResidentNodePipelineRequest,
        ResidentNodePipelineResult, ResidentObligationKind, ResidentObligationScope,
        ResidentProjectImage, ResidentRowBoundMergeBranch, ResidentRowBoundMergePropertyMode,
        ResidentRowBoundMergePropertySource, ResidentRowBoundRelationshipCommand,
        ResidentRowBoundRelationshipDirection, ResidentRowBoundRelationshipOutput,
        ResidentRowBoundRelationshipStage, ResidentRowCreateCommand, ResidentRowMutationRequest,
        ResidentRowMutationResult, ResidentSortRequest, ResidentSortResult, ResidentVectorQuery,
        ResidentVectorResult, ScratchReservation,
    },
    graph::{GraphStore, IndexCatalog, LayerMask, TemporalStore},
    types::{LabelId, PropertyId},
};
use tokio_util::sync::CancellationToken;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::from_u128(
    0x4d45_5247_4536_5f4f_4e43_5245_4154_4501,
));
const MEMORY_LIMIT_BYTES: usize = 128 * 1024 * 1024;
const RESERVED_BYTES: usize = 16 * 1024 * 1024;

#[derive(Clone, Copy, Debug)]
enum Target {
    EndNode,
    Relationship,
}

#[derive(Clone, Copy, Debug)]
enum ExpectedValue {
    Integer(i64),
    String(&'static str),
}

impl ExpectedValue {
    fn scalar(self) -> ScalarValue {
        match self {
            Self::Integer(value) => ScalarValue::Integer(value),
            Self::String(value) => ScalarValue::String(value.into()),
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Case {
    scenario: u8,
    name: &'static str,
    relationship_type: &'static str,
    query: &'static str,
    target: Target,
    property: &'static str,
    expected: ExpectedValue,
    returns_count: bool,
}

const CASES: [Case; 2] = [
    Case {
        scenario: 1,
        name: "Using ON CREATE on a node",
        relationship_type: "KNOWS",
        query: "MATCH (a:A), (b:B)\n\
                MERGE (a)-[:KNOWS]->(b)\n\
                  ON CREATE SET b.created = 1",
        target: Target::EndNode,
        property: "created",
        expected: ExpectedValue::Integer(1),
        returns_count: false,
    },
    Case {
        scenario: 2,
        name: "Using ON CREATE on a relationship",
        relationship_type: "TYPE",
        query: "MATCH (a:A), (b:B)\n\
                MERGE (a)-[r:TYPE]->(b)\n\
                  ON CREATE SET r.name = 'Lola'\n\
                RETURN count(r)",
        target: Target::Relationship,
        property: "name",
        expected: ExpectedValue::String("Lola"),
        returns_count: true,
    },
];

#[derive(Clone)]
struct Fixture {
    graph: GraphStore,
    bookmark: Bookmark,
}

fn next_ids(graph: &GraphStore) -> Result<(u64, u64)> {
    let next_node_id = graph
        .nodes()
        .map(|node| node.id().0)
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or_else(|| Error::new(ErrorCode::ResultBudgetExceeded, "test node ID overflow"))?;
    let next_edge_id = graph
        .edges()
        .map(|edge| edge.id().0)
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or_else(|| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "test relationship ID overflow",
            )
        })?;
    Ok((next_node_id, next_edge_id))
}

fn context<'a>(
    graph: &'a GraphStore,
    backend: Option<&'a dyn ExecutionBackend>,
    require_native_execution: bool,
) -> ExecutionContext<'a> {
    let (next_node_id, next_edge_id) = next_ids(graph).expect("fixture IDs must be valid");
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
            term: 66,
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

impl Fixture {
    fn build(case: Case, preexisting_relationship: bool) -> Result<Self> {
        let mut graph = GraphStore::default();
        let setup = if preexisting_relationship {
            format!(
                "CREATE (a:A), (b:B) CREATE (a)-[:{}]->(b)",
                case.relationship_type
            )
        } else {
            "CREATE (:A), (:B)".to_owned()
        };
        let output = QueryEngine.execute(&setup, &mut context(&graph, None, false))?;
        for mutation in output.graph_mutations {
            graph.apply(mutation)?;
        }
        Ok(Self {
            bookmark: Bookmark {
                term: 66,
                index: graph.revision(),
            },
            graph,
        })
    }

    fn resident_image(&self) -> Result<ResidentProjectImage> {
        ResidentProjectImage::build(
            PROJECT,
            self.bookmark,
            &self.graph,
            &TemporalStore::default(),
            &IndexCatalog::default(),
        )
    }
}

fn assert_output(
    case: Case,
    fixture: &Fixture,
    preexisting_relationship: bool,
    output: &irongraph::cypher::ExecutionOutput,
) -> Result<()> {
    let created = !preexisting_relationship;
    let expected_stats = StatementStats {
        relationships_created: u64::from(created),
        properties_set: u64::from(created),
        ..StatementStats::default()
    };
    if output.result.statistics != expected_stats
        || output.result.bookmark != fixture.bookmark
        || output.result.truncated
        || !output.temporal_mutations.is_empty()
        || output.administrative.is_some()
        || !output.vector_searches.is_empty()
        || output.runtime_replans != 0
    {
        return Err(Error::internal(format!(
            "Merge6 [{}] {} changed its result envelope: {output:#?}",
            case.scenario, case.name
        )));
    }
    if case.returns_count {
        if output.result.schema != [("count(r)".to_owned(), ColumnType::Integer)]
            || !matches!(
                output.result.batches.as_slice(),
                [batch]
                    if batch.row_count == 1
                        && matches!(batch.columns.as_slice(), [column]
                            if column.values == [ResultValue::Scalar(ScalarValue::Integer(1))])
            )
        {
            return Err(Error::internal(format!(
                "Merge6 [{}] changed count(r): {:?}",
                case.scenario, output.result
            )));
        }
    } else if !output.result.schema.is_empty() || !output.result.batches.is_empty() {
        return Err(Error::internal(format!(
            "Merge6 [{}] write-only query returned rows",
            case.scenario
        )));
    }

    let mut after = fixture.graph.clone();
    for mutation in &output.graph_mutations {
        after.apply(mutation.clone())?;
    }
    let relationship_type = after
        .catalog()
        .relationship_type(case.relationship_type)
        .ok_or_else(|| Error::internal("Merge6 relationship type disappeared"))?;
    let relationships = after
        .edges()
        .filter(|edge| edge.relationship_type() == relationship_type)
        .collect::<Vec<_>>();
    let [relationship] = relationships.as_slice() else {
        return Err(Error::internal(format!(
            "Merge6 [{}] expected one {}, got {}",
            case.scenario,
            case.relationship_type,
            relationships.len()
        )));
    };
    let actual = match case.target {
        Target::EndNode => after
            .catalog()
            .property(case.property)
            .and_then(|property| {
                after
                    .node(relationship.target())
                    .and_then(|node| node.property(property))
            }),
        Target::Relationship => after
            .catalog()
            .property(case.property)
            .and_then(|property| relationship.property(property)),
    };
    let expected = created.then(|| case.expected.scalar());
    if actual != expected {
        return Err(Error::internal(format!(
            "Merge6 [{}] ON CREATE condition changed: expected {expected:?}, got {actual:?}",
            case.scenario
        )));
    }
    Ok(())
}

#[derive(Default)]
struct Observations {
    pins: AtomicUsize,
    commands: AtomicUsize,
    forbidden: AtomicUsize,
    requests: Mutex<Vec<ResidentRowMutationRequest>>,
}

/// A strict CPU semantic-reference observer: all decomposed primitives are poisoned, while the
/// one complete pinned row-mutation command is recorded and delegated to the real CPU backend.
struct StrictOnCreateBackend {
    inner: Box<dyn ExecutionBackend>,
    pinned: bool,
    observations: Arc<Observations>,
}

impl StrictOnCreateBackend {
    fn new(image: ResidentProjectImage) -> Result<Self> {
        let mut inner = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
        inner.admit_project(image)?;
        Ok(Self {
            inner: Box::new(inner),
            pinned: false,
            observations: Arc::new(Observations::default()),
        })
    }

    fn observations(&self) -> Arc<Observations> {
        Arc::clone(&self.observations)
    }

    fn reject<T>(&self, route: &'static str) -> Result<T> {
        self.observations.forbidden.fetch_add(1, Ordering::SeqCst);
        Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!("strict Merge6 ON CREATE gate rejected `{route}`"),
        ))
    }
}

impl ExecutionBackend for StrictOnCreateBackend {
    fn kind(&self) -> BackendKind {
        BackendKind::Cpu
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
        self.observations.pins.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Self {
            inner,
            pinned: true,
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

    fn execute_row_mutation(
        &self,
        request: &ResidentRowMutationRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentRowMutationResult> {
        if !self.pinned {
            return self.reject("execute_row_mutation_unpinned");
        }
        request.validate()?;
        self.observations.commands.fetch_add(1, Ordering::SeqCst);
        self.observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(request.clone());
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

fn assert_request(case: Case, request: &ResidentRowMutationRequest) -> Result<()> {
    let Some(body) = request.bound_relationship_merge_body() else {
        return Err(Error::internal(format!(
            "Merge6 [{}] did not use the bound-relationship body",
            case.scenario
        )));
    };
    let [
        ResidentRowCreateCommand::MatchNode(first),
        ResidentRowCreateCommand::MatchNode(second),
        ResidentRowCreateCommand::CreateRelationship(relationship),
    ] = body.program.commands.as_slice()
    else {
        return Err(Error::internal(format!(
            "Merge6 [{}] changed structural commands: {:?}",
            case.scenario, body.program.commands
        )));
    };
    let label_for = |node: &irongraph::gpu::ResidentRowCreateMatchNode| {
        let [label] = node.labels.as_slice() else {
            return None;
        };
        body.program
            .label_names
            .get(usize::from(*label))
            .map(String::as_str)
    };
    let (a_entity, b_entity) = match (label_for(first), label_for(second)) {
        (Some("A"), Some("B")) => (first.output_entity, second.output_entity),
        (Some("B"), Some("A")) => (second.output_entity, first.output_entity),
        _ => {
            return Err(Error::internal(format!(
                "Merge6 [{}] changed semantic MATCH labels: {body:#?}",
                case.scenario
            )));
        }
    };
    if !matches!(
        body.commands.as_slice(),
        [
            ResidentRowBoundRelationshipCommand::MatchNode { .. },
            ResidentRowBoundRelationshipCommand::MatchNode { .. },
            ResidentRowBoundRelationshipCommand::MergeRelationship {
                direction: ResidentRowBoundRelationshipDirection::Directed,
            },
        ]
    ) || first.output_entity != 0
        || second.output_entity != 1
        || relationship.output_entity != 2
        || relationship.source_entity != a_entity
        || relationship.target_entity != b_entity
        || body.program.relationship_type_names != [case.relationship_type.to_owned()]
        || !relationship.properties.is_empty()
        || body.program.property_names != [case.property.to_owned()]
    {
        return Err(Error::internal(format!(
            "Merge6 [{}] did not keep ON CREATE data separate from MERGE identity: {body:#?}",
            case.scenario
        )));
    }
    let command_stages = body
        .schedule
        .stages
        .iter()
        .filter_map(|stage| match stage {
            ResidentRowBoundRelationshipStage::Command { command } => Some(*command),
            _ => None,
        })
        .collect::<Vec<_>>();
    if command_stages != [0, 1, 2] {
        return Err(Error::internal(format!(
            "Merge6 [{}] changed command order: {:?}",
            case.scenario, body.schedule.stages
        )));
    }
    let [on_create] = body.merge_property_sets.as_slice() else {
        return Err(Error::internal(format!(
            "Merge6 [{}] did not seal exactly one conditional SET: {:?}",
            case.scenario, body.merge_property_sets
        )));
    };
    let target_entity = match case.target {
        Target::EndNode => b_entity,
        Target::Relationship => 2,
    };
    if on_create.branch != ResidentRowBoundMergeBranch::Create
        || on_create.mode != ResidentRowBoundMergePropertyMode::Set
        || on_create.trigger_command != 2
        || on_create.target_entity != target_entity
        || on_create.property_name != 0
        || on_create.source != ResidentRowBoundMergePropertySource::Constant(case.expected.scalar())
        || on_create.rhs_obligation.kind != ResidentObligationKind::MutationRhs
        || on_create.rhs_obligation.scope != ResidentObligationScope::MutationCommand(2)
        || on_create.effect_obligation.kind != ResidentObligationKind::MutationEffect
        || on_create.effect_obligation.scope != ResidentObligationScope::MutationCommand(2)
        || on_create.rhs_obligation.id == on_create.effect_obligation.id
        || body.capacities.maximum_property_entries != 1
        || body.capacities.maximum_payload_bytes
            != match case.expected {
                ExpectedValue::Integer(_) => 0,
                ExpectedValue::String(value) => u64::try_from(value.len()).unwrap_or(u64::MAX),
            }
    {
        return Err(Error::internal(format!(
            "Merge6 [{}] changed its sealed conditional SET contract: {body:#?}",
            case.scenario
        )));
    }
    match (case.returns_count, body.outputs.as_slice()) {
        (false, []) => {}
        (true, [ResidentRowBoundRelationshipOutput::CountEntity { name, entity: 2 }])
            if name == "count(r)" => {}
        _ => {
            return Err(Error::internal(format!(
                "Merge6 [{}] changed its final output: {:?}",
                case.scenario, body.outputs
            )));
        }
    }
    Ok(())
}

fn run_generic(case: Case, preexisting_relationship: bool) -> Result<()> {
    let fixture = Fixture::build(case, preexisting_relationship)?;
    let output = QueryEngine.execute(case.query, &mut context(&fixture.graph, None, false))?;
    assert_output(case, &fixture, preexisting_relationship, &output)
}

fn run_strict(case: Case, preexisting_relationship: bool) -> Result<()> {
    let fixture = Fixture::build(case, preexisting_relationship)?;
    let backend = StrictOnCreateBackend::new(fixture.resident_image()?)?;
    let observations = backend.observations();
    let output = QueryEngine.execute(
        case.query,
        &mut context(&fixture.graph, Some(&backend), true),
    )?;
    if observations.pins.load(Ordering::SeqCst) != 1
        || observations.commands.load(Ordering::SeqCst) != 1
        || observations.forbidden.load(Ordering::SeqCst) != 0
    {
        return Err(Error::internal(format!(
            "Merge6 [{}] did not execute as one complete native command",
            case.scenario
        )));
    }
    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let [request] = requests.as_slice() else {
        return Err(Error::internal("Merge6 strict request disappeared"));
    };
    assert_request(case, request)?;
    assert_output(case, &fixture, preexisting_relationship, &output)
}

#[test]
fn generic_cpu_oracle_pins_merge6_645_and_646_create_and_match_semantics() -> Result<()> {
    for case in CASES {
        run_generic(case, false)?;
        run_generic(case, true)?;
    }
    Ok(())
}

#[test]
fn strict_cpu_executes_merge6_645_and_646_as_one_native_command() -> Result<()> {
    for case in CASES {
        run_strict(case, false)?;
        run_strict(case, true)?;
    }
    Ok(())
}
