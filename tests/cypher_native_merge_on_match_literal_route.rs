// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Strict native acceptance for the literal-property tranche of `Merge7.feature` ([1]-[3]).
//!
//! Map replacement/merge in [4]/[5] is deliberately separate: those cases require a dynamic
//! property-map action and a second `keys(r)`/dynamic-property control query. This gate covers
//! only one scalar `ON MATCH SET entity.property = literal` nested inside relationship MERGE.

use std::{
    collections::BTreeMap,
    fs,
    sync::{
        Arc, Mutex,
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
use sha2::{Digest as _, Sha256};
use tokio_util::sync::CancellationToken;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::from_u128(
    0x4d45_5247_4537_5f4f_4e5f_4d41_5443_4801,
));
const MEMORY_LIMIT_BYTES: usize = 128 * 1024 * 1024;
const RESERVED_BYTES: usize = 16 * 1024 * 1024;
const REPORT: &str = "/tmp/irongraph-tck-full-20260721-after-create3-list12-delete5.json";
const REPORT_SHA256: &str = "66383b9f0075863b17778ad4348af6c729e8ccc204c8fe67b8d5b67712ffa288";
const FEATURE_SUFFIX: &str = "features/clauses/merge/Merge7.feature";
const FAILURE_INDICES: [usize; 5] = [651, 652, 653, 654, 655];
const FAILURE_NAMES: [&str; 5] = [
    "[1] Using ON MATCH on created node",
    "[2] Using ON MATCH on created relationship",
    "[3] Using ON MATCH on a relationship",
    "[4] Copying properties from node with ON MATCH",
    "[5] Copying properties from literal map with ON MATCH",
];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Target {
    EndNode,
    Relationship,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

    const fn payload_bytes(self) -> u64 {
        match self {
            Self::Integer(_) => 0,
            Self::String(value) => value.len() as u64,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct Case {
    report_index: usize,
    scenario: u8,
    name: &'static str,
    relationship_type: &'static str,
    query: &'static str,
    target: Target,
    property: &'static str,
    expected: ExpectedValue,
    returns_count: bool,
}

const CASES: [Case; 3] = [
    Case {
        report_index: 651,
        scenario: 1,
        name: "[1] Using ON MATCH on created node",
        relationship_type: "KNOWS",
        query: "MATCH (a:A), (b:B)\n\
                MERGE (a)-[:KNOWS]->(b)\n\
                  ON MATCH SET b.created = 1",
        target: Target::EndNode,
        property: "created",
        expected: ExpectedValue::Integer(1),
        returns_count: false,
    },
    Case {
        report_index: 652,
        scenario: 2,
        name: "[2] Using ON MATCH on created relationship",
        relationship_type: "KNOWS",
        query: "MATCH (a:A), (b:B)\n\
                MERGE (a)-[r:KNOWS]->(b)\n\
                  ON MATCH SET r.created = 1",
        target: Target::Relationship,
        property: "created",
        expected: ExpectedValue::Integer(1),
        returns_count: false,
    },
    Case {
        report_index: 653,
        scenario: 3,
        name: "[3] Using ON MATCH on a relationship",
        relationship_type: "TYPE",
        query: "MATCH (a:A), (b:B)\n\
                MERGE (a)-[r:TYPE]->(b)\n\
                  ON MATCH SET r.name = 'Lola'\n\
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
        .ok_or_else(|| Error::new(ErrorCode::ResultBudgetExceeded, "test edge ID overflow"))?;
    Ok((next_node_id, next_edge_id))
}

fn context<'a>(
    graph: &'a GraphStore,
    backend: Option<&'a dyn ExecutionBackend>,
    require_native_execution: bool,
) -> ExecutionContext<'a> {
    let (next_node_id, next_edge_id) = next_ids(graph).expect("fixture IDs are valid");
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
            term: 77,
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
                term: 77,
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

fn output_rows(output: &ExecutionOutput) -> Result<Vec<Vec<ResultValue>>> {
    let mut rows = Vec::new();
    for batch in &output.result.batches {
        if !batch.validate() || batch.columns.len() != output.result.schema.len() {
            return Err(Error::internal("Merge7 result has an invalid batch shape"));
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

fn assert_output(
    case: Case,
    fixture: &Fixture,
    preexisting_relationship: bool,
    output: &ExecutionOutput,
) -> Result<()> {
    let created = !preexisting_relationship;
    assert_eq!(
        output.result.statistics,
        StatementStats {
            relationships_created: u64::from(created),
            properties_set: u64::from(preexisting_relationship),
            ..StatementStats::default()
        },
        "Merge7 [{}] changed conditional effects",
        case.scenario
    );
    assert_eq!(output.result.bookmark, fixture.bookmark);
    assert!(!output.result.truncated);
    assert!(output.temporal_mutations.is_empty());
    assert!(output.administrative.is_none());
    assert!(output.vector_searches.is_empty());
    assert_eq!(output.runtime_replans, 0);
    if case.returns_count {
        assert_eq!(
            output.result.schema,
            [("count(r)".to_owned(), ColumnType::Integer)]
        );
        assert_eq!(
            output_rows(output)?,
            [vec![ResultValue::Scalar(ScalarValue::Integer(1))]]
        );
    } else {
        assert!(output.result.schema.is_empty());
        assert!(output.result.batches.is_empty());
    }

    let mut after = fixture.graph.clone();
    for mutation in &output.graph_mutations {
        after.apply(mutation.clone())?;
    }
    let relationship_type = after
        .catalog()
        .relationship_type(case.relationship_type)
        .ok_or_else(|| Error::internal("Merge7 relationship type disappeared"))?;
    let relationships = after
        .edges()
        .filter(|edge| edge.relationship_type() == relationship_type)
        .collect::<Vec<_>>();
    let [relationship] = relationships.as_slice() else {
        return Err(Error::internal(format!(
            "Merge7 [{}] expected exactly one {}, got {}",
            case.scenario,
            case.relationship_type,
            relationships.len()
        )));
    };
    let property = after.catalog().property(case.property);
    let actual = match case.target {
        Target::EndNode => property.and_then(|property| {
            after
                .node(relationship.target())
                .and_then(|node| node.property(property))
        }),
        Target::Relationship => property.and_then(|property| relationship.property(property)),
    };
    let expected = preexisting_relationship.then(|| case.expected.scalar());
    assert_eq!(
        actual, expected,
        "Merge7 [{}] fired ON MATCH on the wrong branch",
        case.scenario
    );
    Ok(())
}

#[derive(Default)]
struct Observations {
    pins: AtomicUsize,
    commands: AtomicUsize,
    forbidden: AtomicUsize,
    requests: Mutex<Vec<ResidentRowMutationRequest>>,
}

/// All decomposed primitives are poisoned. Only one complete pinned row-mutation command is
/// recorded and delegated to the real CPU semantic reference.
struct StrictOnMatchBackend {
    inner: Box<dyn ExecutionBackend>,
    pinned: bool,
    observations: Arc<Observations>,
}

impl StrictOnMatchBackend {
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
            format!("strict Merge7 ON MATCH gate rejected `{route}`"),
        ))
    }
}

impl ExecutionBackend for StrictOnMatchBackend {
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
            "Merge7 [{}] did not use the bound-relationship body",
            case.scenario
        )));
    };
    let [
        ResidentRowCreateCommand::MatchNode(start),
        ResidentRowCreateCommand::MatchNode(end),
        ResidentRowCreateCommand::CreateRelationship(relationship),
    ] = body.program.commands.as_slice()
    else {
        return Err(Error::internal(format!(
            "Merge7 [{}] changed structural commands",
            case.scenario
        )));
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
    ) || start.output_entity != 0
        || end.output_entity != 1
        || relationship.output_entity != 2
        || relationship.source_entity != 0
        || relationship.target_entity != 1
        || !relationship.properties.is_empty()
        || body.program.property_names != [case.property.to_owned()]
        || body.program.label_names != ["A".to_owned(), "B".to_owned()]
        || body.program.relationship_type_names != [case.relationship_type.to_owned()]
    {
        return Err(Error::internal(format!(
            "Merge7 [{}] did not keep ON MATCH data outside MERGE identity: {body:#?}",
            case.scenario
        )));
    }
    let [action] = body.merge_property_sets.as_slice() else {
        return Err(Error::internal(format!(
            "Merge7 [{}] did not seal exactly one ON MATCH property action: {body:#?}",
            case.scenario
        )));
    };
    let expected_target = match case.target {
        Target::EndNode => 1,
        Target::Relationship => 2,
    };
    if action.branch != ResidentRowBoundMergeBranch::Match
        || action.mode != ResidentRowBoundMergePropertyMode::Set
        || action.trigger_command != 2
        || action.target_entity != expected_target
        || action.property_name != 0
        || action.source != ResidentRowBoundMergePropertySource::Constant(case.expected.scalar())
        || action.rhs_obligation.kind != ResidentObligationKind::MutationRhs
        || action.rhs_obligation.scope != ResidentObligationScope::MutationCommand(2)
        || action.effect_obligation.kind != ResidentObligationKind::MutationEffect
        || action.effect_obligation.scope != ResidentObligationScope::MutationCommand(2)
        || action.rhs_obligation.id == action.effect_obligation.id
    {
        return Err(Error::internal(format!(
            "Merge7 [{}] changed its ON MATCH branch descriptor: {body:#?}",
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
    if command_stages != [0, 1, 2]
        || body.schedule.stages.len() != 3
        || body.schedule.value_slot_count != 0
        || !body.schedule.property_bindings.is_empty()
        || body.capacities.maximum_property_entries != 1
        || body.capacities.maximum_payload_bytes != case.expected.payload_bytes()
    {
        return Err(Error::internal(format!(
            "Merge7 [{}] changed its conditional SET envelope: {body:#?}",
            case.scenario
        )));
    }
    match (case.returns_count, body.outputs.as_slice()) {
        (false, []) => {}
        (true, [ResidentRowBoundRelationshipOutput::CountEntity { name, entity: 2 }])
            if name == "count(r)" => {}
        _ => {
            return Err(Error::internal(format!(
                "Merge7 [{}] changed its output: {:?}",
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
    let backend = StrictOnMatchBackend::new(fixture.resident_image()?)?;
    let observations = backend.observations();
    let output = QueryEngine.execute(
        case.query,
        &mut context(&fixture.graph, Some(&backend), true),
    )?;
    assert_eq!(observations.pins.load(Ordering::SeqCst), 1);
    assert_eq!(observations.commands.load(Ordering::SeqCst), 1);
    assert_eq!(observations.forbidden.load(Ordering::SeqCst), 0);
    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    let [request] = requests.as_slice() else {
        return Err(Error::internal("Merge7 strict request disappeared"));
    };
    assert_request(case, request)?;
    assert_output(case, &fixture, preexisting_relationship, &output)
}

#[test]
fn stale_manifest_splits_literal_and_map_on_match_tranches() {
    assert_eq!(CASES.map(|case| case.report_index), [651, 652, 653]);
    assert_eq!(
        CASES.map(|case| case.name),
        [FAILURE_NAMES[0], FAILURE_NAMES[1], FAILURE_NAMES[2]]
    );
    assert_eq!(FAILURE_INDICES, [651, 652, 653, 654, 655]);
    assert_eq!(FAILURE_NAMES.len() - CASES.len(), 2);
}

#[test]
#[ignore = "external assurance gate: requires the exact stale 3,703-Metal full report"]
fn stale_report_pins_exact_five_merge7_failures() {
    let bytes = fs::read(REPORT).expect("stale report is readable");
    assert_eq!(hex::encode(Sha256::digest(&bytes)), REPORT_SHA256);
    let report: serde_json::Value =
        serde_json::from_slice(&bytes).expect("stale report is valid JSON");
    assert_eq!(report["total"].as_u64(), Some(3_897));
    assert_eq!(report["cpu_passed"].as_u64(), Some(3_897));
    assert_eq!(report["metal_passed"].as_u64(), Some(3_703));
    let scenarios = report["scenarios"]
        .as_array()
        .expect("stale report has scenarios");
    let failures = scenarios
        .iter()
        .enumerate()
        .filter_map(|(index, scenario)| {
            (scenario["path"]
                .as_str()
                .is_some_and(|path| path.ends_with(FEATURE_SUFFIX))
                && scenario["cpu_passed"].as_bool() == Some(true)
                && scenario["metal_passed"].as_bool() == Some(false))
            .then_some(index)
        })
        .collect::<Vec<_>>();
    assert_eq!(failures, FAILURE_INDICES);
    for ((index, name), operation_count) in FAILURE_INDICES
        .into_iter()
        .zip(FAILURE_NAMES)
        .zip([1_u64, 1, 1, 2, 2])
    {
        let scenario = &scenarios[index];
        assert_eq!(scenario["name"].as_str(), Some(name));
        assert_eq!(scenario["operation_count"].as_u64(), Some(operation_count));
        assert_eq!(scenario["cpu_failures"].as_array().map(Vec::len), Some(0));
        assert_eq!(
            scenario["metal_failures"].as_array().map(Vec::len),
            Some(operation_count as usize)
        );
        assert!(
            scenario["metal_failures"]
                .as_array()
                .expect("Merge7 failures are an array")
                .iter()
                .all(|failure| failure
                    .as_str()
                    .is_some_and(|failure| failure.contains("GpuAdmissionFailure")))
        );
    }
}

#[test]
fn generic_cpu_oracle_proves_literal_on_match_create_and_match_branches() -> Result<()> {
    for case in CASES {
        run_generic(case, false)?;
        run_generic(case, true)?;
    }
    Ok(())
}

#[test]
fn strict_cpu_executes_literal_on_match_as_one_native_command() -> Result<()> {
    for case in CASES {
        run_strict(case, false)?;
        run_strict(case, true)?;
    }
    Ok(())
}
