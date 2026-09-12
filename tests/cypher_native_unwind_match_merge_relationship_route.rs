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
    Bookmark, EdgeId, Error, ErrorCode, Layer, NodeId, ProjectId, Result, ScalarValue,
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
    graph::{
        EdgeInput, GraphMutation, GraphStore, IndexCatalog, LayerMask, NodeInput, TemporalStore,
    },
    types::{LabelId, PropertyId},
};
use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
use irongraph::gpu::MetalBackend;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());
const MEMORY_LIMIT_BYTES: usize = 128 * 1024 * 1024;
const RESERVED_BYTES: usize = 16 * 1024 * 1024;
const MAX_RESULT_ROWS: usize = 128;
const FIRST_CREATED_NODE_ID: u64 = 100;
const FIRST_CREATED_EDGE_ID: u64 = 200;

const UNWIND1_FEATURE: &str = "features/clauses/unwind/Unwind1.feature";
const UNWIND1_SCENARIO: &str = "[6] Creating nodes from an unwound parameter list";
const TCK_SETUP_QUERY: &str = "CREATE (:Year {year: 2016})";
const TCK_QUERY: &str = "UNWIND $events AS event\n\
     MATCH (y:Year {year: event.year})\n\
     MERGE (e:Event {id: event.id})\n\
     MERGE (y)<-[:IN]-(e)\n\
     RETURN e.id AS x\n\
     ORDER BY x";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum EventId {
    Integer(i64),
    Null,
    String,
}

impl EventId {
    fn scalar(self) -> ScalarValue {
        match self {
            Self::Integer(value) => ScalarValue::Integer(value),
            Self::Null => ScalarValue::Null,
            Self::String => ScalarValue::String(Arc::from("not-an-integer")),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct EventParameter {
    year: i64,
    id: EventId,
    unrelated_scalars: bool,
}

impl EventParameter {
    const fn new(year: i64, id: i64) -> Self {
        Self {
            year,
            id: EventId::Integer(id),
            unrelated_scalars: false,
        }
    }

    const fn null_id(year: i64) -> Self {
        Self {
            year,
            id: EventId::Null,
            unrelated_scalars: false,
        }
    }

    const fn unrelated(year: i64, id: i64) -> Self {
        Self {
            year,
            id: EventId::Integer(id),
            unrelated_scalars: true,
        }
    }

    const fn unmatched_string_id(year: i64) -> Self {
        Self {
            year,
            id: EventId::String,
            unrelated_scalars: true,
        }
    }
}

#[derive(Clone, Debug)]
struct Fixture {
    graph: GraphStore,
    bookmark: Bookmark,
}

impl Fixture {
    fn from_graph(graph: GraphStore) -> Self {
        Self {
            bookmark: Bookmark {
                term: 79,
                index: graph.revision(),
            },
            graph,
        }
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

    fn cpu_backend(&self) -> Result<CpuBackend> {
        let mut backend = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
        backend.admit_project(self.resident_image()?)?;
        Ok(backend)
    }
}

fn next_revision(graph: &GraphStore) -> u64 {
    graph.revision().saturating_add(1)
}

fn insert_year(graph: &mut GraphStore, node: NodeId, year_value: i64, revision: u64) -> Result<()> {
    let year_label = graph.catalog_mut().intern_label("Year")?;
    let year_property = graph.catalog_mut().intern_property("year")?;
    graph.insert_node(NodeInput {
        id: node,
        layer: Layer::Observed,
        revision,
        labels: vec![year_label],
        properties: vec![(year_property, ScalarValue::Integer(year_value))],
    })?;
    Ok(())
}

fn insert_event(graph: &mut GraphStore, node: NodeId, id_value: i64, revision: u64) -> Result<()> {
    let event_label = graph.catalog_mut().intern_label("Event")?;
    let id_property = graph.catalog_mut().intern_property("id")?;
    graph.insert_node(NodeInput {
        id: node,
        layer: Layer::Observed,
        revision,
        labels: vec![event_label],
        properties: vec![(id_property, ScalarValue::Integer(id_value))],
    })?;
    Ok(())
}

fn insert_named_edge(
    graph: &mut GraphStore,
    edge: EdgeId,
    source: NodeId,
    target: NodeId,
    relationship_type: &str,
    revision: u64,
) -> Result<()> {
    let relationship_type = graph
        .catalog_mut()
        .intern_relationship_type(relationship_type)?;
    graph.insert_edge(EdgeInput {
        id: edge,
        source,
        target,
        relationship_type,
        layer: Layer::Observed,
        revision,
        properties: Vec::new(),
    })?;
    Ok(())
}

fn fixture_with_years(years: &[(u64, i64)]) -> Result<Fixture> {
    let mut graph = GraphStore::default();
    for (node, year) in years {
        let revision = next_revision(&graph);
        insert_year(&mut graph, NodeId(*node), *year, revision)?;
    }
    Ok(Fixture::from_graph(graph))
}

fn event_value(event: EventParameter) -> ResultValue {
    let mut value = BTreeMap::from([
        ("id".to_owned(), ResultValue::Scalar(event.id.scalar())),
        (
            "year".to_owned(),
            ResultValue::Scalar(ScalarValue::Integer(event.year)),
        ),
    ]);
    if event.unrelated_scalars {
        value.insert(
            "active".to_owned(),
            ResultValue::Scalar(ScalarValue::Boolean(true)),
        );
        value.insert(
            "note".to_owned(),
            ResultValue::Scalar(ScalarValue::String(Arc::from("ignored by the query"))),
        );
        value.insert(
            "observed_on".to_owned(),
            ResultValue::Scalar(ScalarValue::Date(20_000)),
        );
    }
    ResultValue::Map(value)
}

fn parameters(events: &[EventParameter]) -> BTreeMap<String, ResultValue> {
    BTreeMap::from([(
        "events".to_owned(),
        ResultValue::List(events.iter().copied().map(event_value).collect()),
    )])
}

fn context<'a>(
    fixture: &'a Fixture,
    backend: Option<&'a dyn ExecutionBackend>,
    events: &[EventParameter],
    require_native_execution: bool,
) -> ExecutionContext<'a> {
    ExecutionContext {
        project_id: PROJECT,
        graph: &fixture.graph,
        binding_catalog: fixture.graph.catalog(),
        prior_graph_mutations: &[],
        temporal: None,
        prior_temporal_mutations: &[],
        vector_indexes: BTreeMap::new(),
        scalar_indexes: None,
        text_embedding: None,
        parameters: parameters(events),
        bookmark: fixture.bookmark,
        mutation_revision: fixture.graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id: FIRST_CREATED_NODE_ID,
        next_edge_id: FIRST_CREATED_EDGE_ID,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            write: true,
            require_native_execution,
            ..BindCapabilities::default()
        },
        max_result_rows: MAX_RESULT_ROWS,
        max_batch_rows: 2,
        optimizer_statistics: None,
        backend,
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(30)),
        resolved_query_at_time_nanos: None,
    }
}

fn execute_cpu(case: &ScenarioCase) -> Result<ExecutionOutput> {
    QueryEngine.execute(
        TCK_QUERY,
        &mut context(&case.fixture, None, &case.events, false),
    )
}

#[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
fn execute_real_metal(case: &ScenarioCase) -> Result<ExecutionOutput> {
    let mut backend = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
    backend.admit_project(case.fixture.resident_image()?)?;
    if backend.kind() != BackendKind::Metal {
        return Err(Error::internal(
            "scenario [6] real-hardware differential did not select Metal",
        ));
    }
    QueryEngine.execute(
        TCK_QUERY,
        &mut context(&case.fixture, Some(&backend), &case.events, true),
    )
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct NodeSignature {
    id: NodeId,
    layer: Layer,
    revision: u64,
    labels: BTreeSet<String>,
    properties: BTreeMap<String, ScalarValue>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct EdgeSignature {
    id: EdgeId,
    source: NodeId,
    target: NodeId,
    relationship_type: String,
    layer: Layer,
    revision: u64,
    properties: BTreeMap<String, ScalarValue>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct GraphSignature {
    label_names: BTreeSet<String>,
    property_names: BTreeSet<String>,
    relationship_type_names: BTreeSet<String>,
    nodes: Vec<NodeSignature>,
    edges: Vec<EdgeSignature>,
}

fn graph_signature(graph: &GraphStore) -> Result<GraphSignature> {
    let nodes = graph
        .nodes()
        .map(|node| {
            let labels = node
                .labels()
                .iter()
                .map(|label| {
                    graph
                        .catalog()
                        .label_name(*label)
                        .map(str::to_owned)
                        .ok_or_else(|| Error::internal("node label has no catalog name"))
                })
                .collect::<Result<BTreeSet<_>>>()?;
            let properties = node
                .properties()
                .into_iter()
                .map(|(property, value)| {
                    Ok((
                        graph
                            .catalog()
                            .property_name(property)
                            .map(str::to_owned)
                            .ok_or_else(|| Error::internal("node property has no catalog name"))?,
                        value,
                    ))
                })
                .collect::<Result<BTreeMap<_, _>>>()?;
            Ok(NodeSignature {
                id: node.id(),
                layer: node.layer(),
                revision: node.revision(),
                labels,
                properties,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let edges = graph
        .edges()
        .map(|edge| {
            let properties = edge
                .properties()
                .into_iter()
                .map(|(property, value)| {
                    Ok((
                        graph
                            .catalog()
                            .property_name(property)
                            .map(str::to_owned)
                            .ok_or_else(|| {
                                Error::internal("relationship property has no catalog name")
                            })?,
                        value,
                    ))
                })
                .collect::<Result<BTreeMap<_, _>>>()?;
            Ok(EdgeSignature {
                id: edge.id(),
                source: edge.source(),
                target: edge.target(),
                relationship_type: graph
                    .catalog()
                    .relationship_type_name(edge.relationship_type())
                    .map(str::to_owned)
                    .ok_or_else(|| Error::internal("relationship has no catalog type name"))?,
                layer: edge.layer(),
                revision: edge.revision(),
                properties,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(GraphSignature {
        label_names: graph
            .catalog()
            .labels()
            .map(|(_, name)| name.to_owned())
            .collect(),
        property_names: graph
            .catalog()
            .properties()
            .map(|(_, name)| name.to_owned())
            .collect(),
        relationship_type_names: graph
            .catalog()
            .relationship_types()
            .map(|(_, name)| name.to_owned())
            .collect(),
        nodes,
        edges,
    })
}

fn published_graph(fixture: &Fixture, output: &ExecutionOutput) -> Result<GraphStore> {
    let mut graph = fixture.graph.clone();
    for mutation in &output.graph_mutations {
        graph.apply(mutation.clone())?;
    }
    Ok(graph)
}

fn statement_stats(nodes: u64, relationships: u64, labels: u64) -> StatementStats {
    StatementStats {
        nodes_created: nodes,
        relationships_created: relationships,
        labels_added: labels,
        ..StatementStats::default()
    }
}

#[derive(Clone, Copy, Debug)]
enum CaseKind {
    ExactTck,
    DuplicateIdenticalRows,
    SameEventDifferentYears,
    ExistingEventAndEdge,
    ExistingEventMissingEdge,
    ZeroYearMatches,
    MultipleYearMatches,
    MatchedNullId,
    UnmatchedNullId,
    UnmatchedNonIntegerIdWithUnrelatedScalars,
    WrongDirectionAndType,
}

const ALL_CASES: [CaseKind; 11] = [
    CaseKind::ExactTck,
    CaseKind::DuplicateIdenticalRows,
    CaseKind::SameEventDifferentYears,
    CaseKind::ExistingEventAndEdge,
    CaseKind::ExistingEventMissingEdge,
    CaseKind::ZeroYearMatches,
    CaseKind::MultipleYearMatches,
    CaseKind::MatchedNullId,
    CaseKind::UnmatchedNullId,
    CaseKind::UnmatchedNonIntegerIdWithUnrelatedScalars,
    CaseKind::WrongDirectionAndType,
];

enum ExpectedOutcome {
    Success {
        rows: Vec<i64>,
        statistics: StatementStats,
        graph: GraphStore,
    },
    MergeNullError,
}

struct ScenarioCase {
    name: &'static str,
    fixture: Fixture,
    events: Vec<EventParameter>,
    expected: ExpectedOutcome,
}

fn success_case(
    name: &'static str,
    fixture: Fixture,
    events: Vec<EventParameter>,
    rows: Vec<i64>,
    statistics: StatementStats,
    graph: GraphStore,
) -> ScenarioCase {
    ScenarioCase {
        name,
        fixture,
        events,
        expected: ExpectedOutcome::Success {
            rows,
            statistics,
            graph,
        },
    }
}

fn build_case(kind: CaseKind) -> Result<ScenarioCase> {
    match kind {
        CaseKind::ExactTck => {
            // This is the exact graph produced by the TCK setup query:
            // CREATE (:Year {year: 2016})
            let fixture = fixture_with_years(&[(1, 2016)])?;
            let mut expected = fixture.graph.clone();
            let revision = next_revision(&expected);
            insert_event(&mut expected, NodeId(100), 1, revision)?;
            insert_event(&mut expected, NodeId(101), 2, revision)?;
            insert_named_edge(
                &mut expected,
                EdgeId(200),
                NodeId(100),
                NodeId(1),
                "IN",
                revision,
            )?;
            insert_named_edge(
                &mut expected,
                EdgeId(201),
                NodeId(101),
                NodeId(1),
                "IN",
                revision,
            )?;
            Ok(success_case(
                "exact openCypher TCK Unwind1 [6]",
                fixture,
                vec![EventParameter::new(2016, 1), EventParameter::new(2016, 2)],
                vec![1, 2],
                statement_stats(2, 2, 2),
                expected,
            ))
        }
        CaseKind::DuplicateIdenticalRows => {
            let fixture = fixture_with_years(&[(1, 2016)])?;
            let mut expected = fixture.graph.clone();
            let revision = next_revision(&expected);
            insert_event(&mut expected, NodeId(100), 1, revision)?;
            insert_named_edge(
                &mut expected,
                EdgeId(200),
                NodeId(100),
                NodeId(1),
                "IN",
                revision,
            )?;
            Ok(success_case(
                "duplicate identical source rows",
                fixture,
                vec![EventParameter::new(2016, 1), EventParameter::new(2016, 1)],
                vec![1, 1],
                statement_stats(1, 1, 1),
                expected,
            ))
        }
        CaseKind::SameEventDifferentYears => {
            let fixture = fixture_with_years(&[(1, 2016), (2, 2017)])?;
            let mut expected = fixture.graph.clone();
            let revision = next_revision(&expected);
            insert_event(&mut expected, NodeId(100), 7, revision)?;
            insert_named_edge(
                &mut expected,
                EdgeId(200),
                NodeId(100),
                NodeId(1),
                "IN",
                revision,
            )?;
            insert_named_edge(
                &mut expected,
                EdgeId(201),
                NodeId(100),
                NodeId(2),
                "IN",
                revision,
            )?;
            Ok(success_case(
                "one Event merged across different Year matches",
                fixture,
                vec![EventParameter::new(2016, 7), EventParameter::new(2017, 7)],
                vec![7, 7],
                statement_stats(1, 2, 1),
                expected,
            ))
        }
        CaseKind::ExistingEventAndEdge => {
            let mut fixture = fixture_with_years(&[(1, 2016)])?;
            let revision = next_revision(&fixture.graph);
            insert_event(&mut fixture.graph, NodeId(50), 1, revision)?;
            let revision = next_revision(&fixture.graph);
            insert_named_edge(
                &mut fixture.graph,
                EdgeId(10),
                NodeId(50),
                NodeId(1),
                "IN",
                revision,
            )?;
            fixture.bookmark.index = fixture.graph.revision();
            Ok(success_case(
                "existing Event and exact existing IN relationship",
                fixture.clone(),
                vec![EventParameter::new(2016, 1)],
                vec![1],
                StatementStats::default(),
                fixture.graph,
            ))
        }
        CaseKind::ExistingEventMissingEdge => {
            let mut fixture = fixture_with_years(&[(1, 2016)])?;
            let revision = next_revision(&fixture.graph);
            insert_event(&mut fixture.graph, NodeId(50), 1, revision)?;
            fixture.bookmark.index = fixture.graph.revision();
            let mut expected = fixture.graph.clone();
            let revision = next_revision(&expected);
            insert_named_edge(
                &mut expected,
                EdgeId(200),
                NodeId(50),
                NodeId(1),
                "IN",
                revision,
            )?;
            Ok(success_case(
                "existing Event with missing IN relationship",
                fixture,
                vec![EventParameter::new(2016, 1)],
                vec![1],
                statement_stats(0, 1, 0),
                expected,
            ))
        }
        CaseKind::ZeroYearMatches => {
            let fixture = fixture_with_years(&[(1, 2016)])?;
            Ok(success_case(
                "zero Year matches",
                fixture.clone(),
                vec![EventParameter::new(1999, 1)],
                Vec::new(),
                StatementStats::default(),
                fixture.graph,
            ))
        }
        CaseKind::MultipleYearMatches => {
            let fixture = fixture_with_years(&[(1, 2016), (2, 2016)])?;
            let mut expected = fixture.graph.clone();
            let revision = next_revision(&expected);
            insert_event(&mut expected, NodeId(100), 1, revision)?;
            insert_named_edge(
                &mut expected,
                EdgeId(200),
                NodeId(100),
                NodeId(1),
                "IN",
                revision,
            )?;
            insert_named_edge(
                &mut expected,
                EdgeId(201),
                NodeId(100),
                NodeId(2),
                "IN",
                revision,
            )?;
            Ok(success_case(
                "multiple Year matches expand one source row",
                fixture,
                vec![EventParameter::new(2016, 1)],
                vec![1, 1],
                statement_stats(1, 2, 1),
                expected,
            ))
        }
        CaseKind::MatchedNullId => {
            let fixture = fixture_with_years(&[(1, 2016)])?;
            Ok(ScenarioCase {
                name: "matched null event.id atomically aborts an earlier valid row",
                fixture,
                events: vec![EventParameter::new(2016, 1), EventParameter::null_id(2016)],
                expected: ExpectedOutcome::MergeNullError,
            })
        }
        CaseKind::UnmatchedNullId => {
            let fixture = fixture_with_years(&[(1, 2016)])?;
            let mut expected = fixture.graph.clone();
            let revision = next_revision(&expected);
            insert_event(&mut expected, NodeId(100), 1, revision)?;
            insert_named_edge(
                &mut expected,
                EdgeId(200),
                NodeId(100),
                NodeId(1),
                "IN",
                revision,
            )?;
            Ok(success_case(
                "unmatched null event.id does not poison an earlier valid row",
                fixture,
                vec![EventParameter::new(2016, 1), EventParameter::null_id(1999)],
                vec![1],
                statement_stats(1, 1, 1),
                expected,
            ))
        }
        CaseKind::UnmatchedNonIntegerIdWithUnrelatedScalars => {
            let fixture = fixture_with_years(&[(1, 2016)])?;
            let mut expected = fixture.graph.clone();
            let revision = next_revision(&expected);
            insert_event(&mut expected, NodeId(100), 1, revision)?;
            insert_named_edge(
                &mut expected,
                EdgeId(200),
                NodeId(100),
                NodeId(1),
                "IN",
                revision,
            )?;
            Ok(success_case(
                "unrelated scalar fields and an unmatched STRING id preserve MATCH-before-MERGE semantics",
                fixture,
                vec![
                    EventParameter::unrelated(2016, 1),
                    EventParameter::unmatched_string_id(1999),
                ],
                vec![1],
                statement_stats(1, 1, 1),
                expected,
            ))
        }
        CaseKind::WrongDirectionAndType => {
            let mut fixture = fixture_with_years(&[(1, 2016)])?;
            let revision = next_revision(&fixture.graph);
            insert_event(&mut fixture.graph, NodeId(50), 1, revision)?;
            let revision = next_revision(&fixture.graph);
            insert_named_edge(
                &mut fixture.graph,
                EdgeId(10),
                NodeId(1),
                NodeId(50),
                "IN",
                revision,
            )?;
            let revision = next_revision(&fixture.graph);
            insert_named_edge(
                &mut fixture.graph,
                EdgeId(11),
                NodeId(50),
                NodeId(1),
                "OTHER",
                revision,
            )?;
            fixture.bookmark.index = fixture.graph.revision();
            let mut expected = fixture.graph.clone();
            let revision = next_revision(&expected);
            insert_named_edge(
                &mut expected,
                EdgeId(200),
                NodeId(50),
                NodeId(1),
                "IN",
                revision,
            )?;
            Ok(success_case(
                "wrong-direction IN and wrong-type edge do not satisfy MERGE",
                fixture,
                vec![EventParameter::new(2016, 1)],
                vec![1],
                statement_stats(0, 1, 0),
                expected,
            ))
        }
    }
}

fn output_rows(output: &ExecutionOutput, expected_column_type: ColumnType) -> Result<Vec<i64>> {
    if output.result.schema != [("x".to_owned(), expected_column_type.clone())] {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "{UNWIND1_SCENARIO} returned the wrong schema: {:?}",
                output.result.schema
            ),
        ));
    }
    let mut rows = Vec::new();
    for batch in &output.result.batches {
        if batch.columns.len() != 1
            || batch.columns[0].name != "x"
            || batch.columns[0].value_type != expected_column_type
            || batch.columns[0].values.len() != batch.row_count
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                format!("malformed scenario [6] result batch: {batch:?}"),
            ));
        }
        for value in &batch.columns[0].values {
            let ResultValue::Scalar(ScalarValue::Integer(value)) = value else {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    format!("scenario [6] projected a non-integer x: {value:?}"),
                ));
            };
            rows.push(*value);
        }
    }
    Ok(rows)
}

fn assert_success_output(case: &ScenarioCase, output: &ExecutionOutput) -> Result<()> {
    let ExpectedOutcome::Success {
        rows,
        statistics,
        graph,
    } = &case.expected
    else {
        return Err(Error::internal("success assertion used for an error case"));
    };
    // The generic CPU reference reports Null for a property column when the relation is empty and
    // no runtime value exists to refine it. Non-empty scenario [6] relations are Integer.
    let expected_column_type = if rows.is_empty() {
        ColumnType::Null
    } else {
        ColumnType::Integer
    };
    let actual_rows = output_rows(output, expected_column_type)?;
    if actual_rows.as_slice() != rows.as_slice() {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "{UNWIND1_FEATURE} / {UNWIND1_SCENARIO} / {} lost ORDER BY or multiplicity; expected {rows:?}, got {actual_rows:?}",
                case.name
            ),
        ));
    }
    if output.result.statistics != *statistics
        || output.result.bookmark != case.fixture.bookmark
        || output.result.truncated
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "scenario [6] / {} statistics, bookmark, or truncation differ: {:?}",
                case.name, output.result
            ),
        ));
    }
    if !output.temporal_mutations.is_empty()
        || output.administrative.is_some()
        || !output.vector_searches.is_empty()
        || output.runtime_replans != 0
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!("scenario [6] / {} emitted unrelated state", case.name),
        ));
    }
    if let Some(unexpected) = output.graph_mutations.iter().find(|mutation| {
        !matches!(
            mutation,
            GraphMutation::DeclareLabel { .. }
                | GraphMutation::DeclareProperty { .. }
                | GraphMutation::DeclareRelationshipType { .. }
                | GraphMutation::InsertNode(_)
                | GraphMutation::InsertEdge(_)
        )
    }) {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "scenario [6] / {} emitted an unrelated mutation: {unexpected:?}",
                case.name
            ),
        ));
    }
    let inserted_nodes = output
        .graph_mutations
        .iter()
        .filter(|mutation| matches!(mutation, GraphMutation::InsertNode(_)))
        .count() as u64;
    let inserted_edges = output
        .graph_mutations
        .iter()
        .filter(|mutation| matches!(mutation, GraphMutation::InsertEdge(_)))
        .count() as u64;
    if inserted_nodes != statistics.nodes_created
        || inserted_edges != statistics.relationships_created
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "scenario [6] / {} statistics do not match published inserts: nodes={inserted_nodes}, relationships={inserted_edges}",
                case.name
            ),
        ));
    }
    let actual_graph = published_graph(&case.fixture, output)?;
    let actual_signature = graph_signature(&actual_graph)?;
    let expected_signature = graph_signature(graph)?;
    if actual_signature != expected_signature {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "scenario [6] / {} applied graph differs; expected {expected_signature:#?}, got {actual_signature:#?}",
                case.name
            ),
        ));
    }
    Ok(())
}

fn assert_merge_null_error(case: &ScenarioCase, error: &Error) -> Result<()> {
    if !matches!(&case.expected, ExpectedOutcome::MergeNullError) {
        return Err(Error::internal(
            "null-error assertion used for a success case",
        ));
    }
    if error.code != ErrorCode::QueryType || !error.message.contains("MergeReadOwnWrites") {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "scenario [6] / {} changed matched-null error semantics: {error:?}",
                case.name
            ),
        ));
    }
    Ok(())
}

fn assert_case_outcome(
    case: &ScenarioCase,
    outcome: std::result::Result<ExecutionOutput, Error>,
) -> Result<()> {
    let before = graph_signature(&case.fixture.graph)?;
    match &case.expected {
        ExpectedOutcome::Success { .. } => {
            let output = outcome.map_err(|error| {
                Error::new(
                    error.code,
                    format!("scenario [6] / {} unexpectedly failed: {error}", case.name),
                )
            })?;
            assert_success_output(case, &output)?;
        }
        ExpectedOutcome::MergeNullError => {
            let error = outcome.expect_err("matched null event.id unexpectedly succeeded");
            assert_merge_null_error(case, &error)?;
        }
    }
    let after = graph_signature(&case.fixture.graph)?;
    if after != before {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "scenario [6] / {} mutated its canonical input before publication",
                case.name
            ),
        ));
    }
    Ok(())
}

fn assert_cpu_case(kind: CaseKind) -> Result<()> {
    let case = build_case(kind)?;
    let outcome = execute_cpu(&case);
    assert_case_outcome(&case, outcome)
}

#[derive(Default)]
struct RouteObservations {
    pins: AtomicUsize,
    complete_command_calls: AtomicUsize,
    forbidden_calls: AtomicUsize,
    requests: Mutex<Vec<ResidentRowMutationRequest>>,
}

/// Strict red/green boundary for Unwind1 [6].
///
/// The unpinned facade advertises Metal, so mandatory-native execution cannot enter the generic
/// CPU executor. Every primitive exposed by this facade is poisoned. Only one pinned, complete
/// `execute_row_mutation` command may delegate to the CPU semantic implementation; that command
/// must own UNWIND, correlated MATCH, node MERGE, relationship MERGE, integer projection, sorting,
/// dependencies, effects, and final rows together.
struct StrictScenario6Backend {
    inner: Box<dyn ExecutionBackend>,
    pinned: bool,
    expected_bookmark: Bookmark,
    expected_graph_revision: u64,
    observations: Arc<RouteObservations>,
}

impl StrictScenario6Backend {
    fn cpu_reference(inner: CpuBackend) -> Result<Self> {
        let expected_bookmark = inner
            .resident_bookmark(PROJECT)
            .ok_or_else(|| Error::internal("strict scenario [6] backend has no bookmark"))?;
        let expected_graph_revision = inner
            .resident_graph_revision(PROJECT)
            .ok_or_else(|| Error::internal("strict scenario [6] backend has no graph revision"))?;
        Ok(Self {
            inner: Box::new(inner),
            pinned: false,
            expected_bookmark,
            expected_graph_revision,
            observations: Arc::new(RouteObservations::default()),
        })
    }

    fn observations(&self) -> Arc<RouteObservations> {
        Arc::clone(&self.observations)
    }

    fn reject_route<T>(&self, route: &'static str) -> Result<T> {
        self.observations
            .forbidden_calls
            .fetch_add(1, Ordering::SeqCst);
        Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!("strict scenario [6] rejected partial/fallback route `{route}`"),
        ))
    }
}

impl ExecutionBackend for StrictScenario6Backend {
    fn kind(&self) -> BackendKind {
        if self.pinned {
            BackendKind::Cpu
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
            return self.reject_route("pin_project");
        }
        let inner = self.inner.pin_project(project)?;
        if inner.kind() != BackendKind::Cpu
            || inner.resident_bookmark(project) != Some(self.expected_bookmark)
            || inner.resident_graph_revision(project) != Some(self.expected_graph_revision)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "strict scenario [6] pin changed backend provenance or graph fence",
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
        self.reject_route("scan_nodes")
    }

    fn filter_node_i64(
        &self,
        _project: ProjectId,
        _property: PropertyId,
        _operation: CompareOp,
        _operand: i64,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.reject_route("filter_node_i64")
    }

    fn expand_project_out(
        &self,
        _project: ProjectId,
        _sources: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.reject_route("expand_project_out")
    }

    fn expand_project_in(
        &self,
        _project: ProjectId,
        _targets: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.reject_route("expand_project_in")
    }

    fn search_vectors(
        &self,
        _request: &ResidentVectorQuery,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentVectorResult> {
        self.reject_route("search_vectors")
    }

    fn sort_rows(
        &self,
        _request: &ResidentSortRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentSortResult> {
        self.reject_route("sort_rows")
    }

    fn join_node_i64(
        &self,
        _request: &ResidentJoinRequest,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentJoinPair>> {
        self.reject_route("join_node_i64")
    }

    fn group_node_i64(
        &self,
        _request: &ResidentGroupRequest,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentGroup>> {
        self.reject_route("group_node_i64")
    }

    fn execute_node_pipeline(
        &self,
        _request: &ResidentNodePipelineRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentNodePipelineResult> {
        self.reject_route("execute_node_pipeline")
    }

    fn execute_row_mutation(
        &self,
        request: &ResidentRowMutationRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentRowMutationResult> {
        if !self.pinned {
            return self.reject_route("execute_row_mutation_on_unpinned_generation");
        }
        request.validate()?;
        if request.generation.project != PROJECT
            || request.generation.bookmark != self.expected_bookmark
            || request.generation.graph_revision != self.expected_graph_revision
            || self.inner.resident_bookmark(PROJECT) != Some(self.expected_bookmark)
            || self.inner.resident_graph_revision(PROJECT) != Some(self.expected_graph_revision)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "scenario [6] row-mutation request escaped its pinned immutable generation",
            ));
        }
        self.observations
            .complete_command_calls
            .fetch_add(1, Ordering::SeqCst);
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
        self.reject_route("filter_i64")
    }

    fn expand_out(
        &self,
        _sources: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.reject_route("expand_out")
    }

    fn exact_l2(
        &self,
        _matrix: &[f32],
        _rows: usize,
        _dimension: usize,
        _query: &[f32],
        _cancellation: &CancellationToken,
    ) -> Result<DistanceBatch> {
        self.reject_route("exact_l2")
    }
}

fn execute_strict(
    case: &ScenarioCase,
) -> Result<(
    std::result::Result<ExecutionOutput, Error>,
    Arc<RouteObservations>,
)> {
    let backend = StrictScenario6Backend::cpu_reference(case.fixture.cpu_backend()?)?;
    let observations = backend.observations();
    if backend.kind() != BackendKind::Metal {
        return Err(Error::internal(
            "strict scenario [6] backend did not advertise accelerator execution",
        ));
    }
    let outcome = QueryEngine.execute(
        TCK_QUERY,
        &mut context(&case.fixture, Some(&backend), &case.events, true),
    );
    Ok((outcome, observations))
}

fn assert_one_complete_native_command(
    case: &ScenarioCase,
    observations: &RouteObservations,
) -> Result<ResidentRowMutationRequest> {
    let pins = observations.pins.load(Ordering::SeqCst);
    let complete = observations.complete_command_calls.load(Ordering::SeqCst);
    let forbidden = observations.forbidden_calls.load(Ordering::SeqCst);
    if pins != 1 || complete != 1 || forbidden != 0 {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "scenario [6] / {} was not one complete native command: pins={pins}, complete={complete}, forbidden={forbidden}",
                case.name
            ),
        ));
    }
    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if requests.len() != 1 {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "scenario [6] / {} captured {} complete requests instead of one",
                case.name,
                requests.len()
            ),
        ));
    }
    Ok(requests[0].clone())
}

/// Localized adapter over the currently public v1 request fields. When the production v2 command
/// adds correlated work-row or relationship descriptors, only this assertion and the strict
/// backend method above should need structural adaptation; all semantic cases use QueryEngine.
fn assert_request_generation_and_raw_input(
    case: &ScenarioCase,
    request: &ResidentRowMutationRequest,
) -> Result<()> {
    request.validate()?;
    if request.generation.project != PROJECT
        || request.generation.bookmark != case.fixture.bookmark
        || request.generation.graph_revision != case.fixture.graph.revision()
        || request.generation.layout_version != case.fixture.graph.layout_version()
        || request.generation.catalog_generation
            != case.fixture.graph.catalog().optimizer_generation()
        || request.input.len() != case.events.len()
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "scenario [6] / {} native request changed generation or source cardinality: {request:?}",
                case.name
            ),
        ));
    }
    for (row, event) in request.input.iter().zip(&case.events) {
        let actual = row.entries.iter().cloned().collect::<BTreeMap<_, _>>();
        let mut expected = BTreeMap::from([
            ("id".to_owned(), event.id.scalar()),
            ("year".to_owned(), ScalarValue::Integer(event.year)),
        ]);
        if event.unrelated_scalars {
            expected.insert("active".to_owned(), ScalarValue::Boolean(true));
            expected.insert(
                "note".to_owned(),
                ScalarValue::String(Arc::from("ignored by the query")),
            );
            expected.insert("observed_on".to_owned(), ScalarValue::Date(20_000));
        }
        if actual != expected {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                format!(
                    "scenario [6] / {} host-rewrote or lost a raw UNWIND map; expected {expected:?}, got {actual:?}",
                    case.name
                ),
            ));
        }
    }
    Ok(())
}

fn assert_strict_case(kind: CaseKind) -> Result<()> {
    let case = build_case(kind)?;
    let (outcome, observations) = execute_strict(&case)?;
    if matches!(case.expected, ExpectedOutcome::Success { .. })
        && let Err(error) = &outcome
    {
        return Err(Error::new(
            error.code,
            format!(
                "scenario [6] / {} failed before its native route could be verified: {error}",
                case.name
            ),
        ));
    }
    let request = assert_one_complete_native_command(&case, &observations)?;
    assert_request_generation_and_raw_input(&case, &request)?;
    assert_case_outcome(&case, outcome)
}

macro_rules! cpu_case_test {
    ($name:ident, $kind:ident) => {
        #[test]
        fn $name() -> Result<()> {
            assert_cpu_case(CaseKind::$kind)
        }
    };
}

macro_rules! strict_case_test {
    ($name:ident, $kind:ident) => {
        #[test]
        fn $name() -> Result<()> {
            assert_strict_case(CaseKind::$kind)
        }
    };
}

cpu_case_test!(cpu_oracle_exact_opencypher_unwind1_scenario_6, ExactTck);
cpu_case_test!(
    cpu_oracle_duplicate_identical_rows_reuse_node_and_relationship,
    DuplicateIdenticalRows
);
cpu_case_test!(
    cpu_oracle_same_event_across_different_years_creates_two_relationships,
    SameEventDifferentYears
);
cpu_case_test!(
    cpu_oracle_existing_event_and_edge_are_reused,
    ExistingEventAndEdge
);
cpu_case_test!(
    cpu_oracle_existing_event_with_missing_edge_creates_only_edge,
    ExistingEventMissingEdge
);
cpu_case_test!(
    cpu_oracle_zero_year_matches_have_no_effects,
    ZeroYearMatches
);
cpu_case_test!(
    cpu_oracle_multiple_year_matches_expand_rows_before_merge,
    MultipleYearMatches
);
cpu_case_test!(cpu_oracle_matched_null_id_fails_atomically, MatchedNullId);
cpu_case_test!(
    cpu_oracle_unmatched_null_id_never_reaches_merge,
    UnmatchedNullId
);
cpu_case_test!(
    cpu_oracle_unmatched_non_integer_id_and_unrelated_scalars_follow_stage_order,
    UnmatchedNonIntegerIdWithUnrelatedScalars
);
cpu_case_test!(
    cpu_oracle_wrong_direction_and_type_do_not_satisfy_relationship_merge,
    WrongDirectionAndType
);

strict_case_test!(
    strict_native_exact_opencypher_unwind1_scenario_6_is_one_command,
    ExactTck
);
strict_case_test!(
    strict_native_duplicate_identical_rows_are_one_command,
    DuplicateIdenticalRows
);
strict_case_test!(
    strict_native_same_event_across_different_years_is_one_command,
    SameEventDifferentYears
);
strict_case_test!(
    strict_native_existing_event_and_edge_are_one_command,
    ExistingEventAndEdge
);
strict_case_test!(
    strict_native_existing_event_missing_edge_is_one_command,
    ExistingEventMissingEdge
);
strict_case_test!(
    strict_native_zero_year_matches_are_one_command,
    ZeroYearMatches
);
strict_case_test!(
    strict_native_multiple_year_matches_are_one_command,
    MultipleYearMatches
);
strict_case_test!(
    strict_native_matched_null_id_is_one_atomic_command,
    MatchedNullId
);
strict_case_test!(
    strict_native_unmatched_null_id_is_one_command,
    UnmatchedNullId
);
strict_case_test!(
    strict_native_unmatched_non_integer_id_and_unrelated_scalars_are_one_command,
    UnmatchedNonIntegerIdWithUnrelatedScalars
);
strict_case_test!(
    strict_native_wrong_direction_and_type_are_one_command,
    WrongDirectionAndType
);

#[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
#[test]
#[ignore = "requires a real Metal device and serialized hardware execution"]
fn real_metal_scenario_6_matches_cpu_for_all_adversarial_cases() -> Result<()> {
    for kind in ALL_CASES {
        let cpu_case = build_case(kind)?;
        let cpu = execute_cpu(&cpu_case);
        assert_case_outcome(&cpu_case, cpu)?;

        let metal_case = build_case(kind)?;
        let metal = execute_real_metal(&metal_case);
        assert_case_outcome(&metal_case, metal)?;
    }
    Ok(())
}

#[test]
fn exact_tck_text_is_pinned_in_the_scenario_test() {
    assert_eq!(TCK_SETUP_QUERY, "CREATE (:Year {year: 2016})");
    assert_eq!(
        TCK_QUERY,
        "UNWIND $events AS event\n\
         MATCH (y:Year {year: event.year})\n\
         MERGE (e:Event {id: event.id})\n\
         MERGE (y)<-[:IN]-(e)\n\
         RETURN e.id AS x\n\
         ORDER BY x"
    );
}
