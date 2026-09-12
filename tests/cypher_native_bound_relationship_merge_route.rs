// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use irongraph::{
    Bookmark, DocumentItem, Error, ErrorCode, Layer, ProjectId, Result, ScalarValue,
    cypher::{
        BindCapabilities, ColumnType, ExecutionContext, ExecutionOutput, QueryEngine, ResultValue,
        StatementStats,
    },
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend,
        ResidentCreateNodeValueInput, ResidentGroup, ResidentGroupRequest, ResidentJoinPair,
        ResidentJoinRequest, ResidentNodePipelineRequest, ResidentNodePipelineResult,
        ResidentObligationKind, ResidentObligationScope, ResidentProjectImage,
        ResidentQuantifierExpression, ResidentQuantifierFunction, ResidentQuantifierProjection,
        ResidentQuantifierSlot, ResidentQuantifierValue, ResidentRowBoundPropertyBindingSource,
        ResidentRowBoundRelationshipCommand, ResidentRowBoundRelationshipDirection,
        ResidentRowBoundRelationshipOutput, ResidentRowBoundRelationshipStage,
        ResidentRowCreateCommand, ResidentRowCreateValueInput, ResidentRowMutationBody,
        ResidentRowMutationRequest, ResidentRowMutationResult, ResidentSortRequest,
        ResidentSortResult, ResidentVectorQuery, ResidentVectorResult, ScratchReservation,
    },
    graph::{GraphMutation, GraphStore, IndexCatalog, LayerMask, TemporalStore},
    types::{LabelId, PropertyId},
};
use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
use irongraph::gpu::MetalBackend;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());
const MEMORY_LIMIT_BYTES: usize = 256 * 1024 * 1024;
const RESERVED_BYTES: usize = 16 * 1024 * 1024;
const MAX_RESULT_ROWS: usize = 1_024;
const MERGE5_FEATURE: &str = "features/clauses/merge/Merge5.feature";

#[derive(Clone, Copy, Debug)]
enum ExpectedType {
    Integer,
    String,
    Relationship,
    Path,
}

impl ExpectedType {
    const fn column_type(self) -> ColumnType {
        match self {
            Self::Integer => ColumnType::Integer,
            Self::String => ColumnType::String,
            Self::Relationship => ColumnType::Relationship,
            Self::Path => ColumnType::Path,
        }
    }
}

#[derive(Clone, Copy, Debug)]
enum ExpectedCell {
    Integer(i64),
    String(&'static str),
    Relationship {
        relationship_type: &'static str,
        name: Option<&'static str>,
    },
    Path {
        start_num: i64,
        relationship_type: &'static str,
        end_num: i64,
    },
}

#[derive(Clone, Copy, Debug)]
enum ExpectedNode {
    Any,
    Label(&'static str),
    IntegerProperty(&'static str, i64),
}

#[derive(Clone, Copy, Debug)]
struct ExpectedCreatedRelationship {
    relationship_type: &'static str,
    source: ExpectedNode,
    target: ExpectedNode,
    properties: &'static [(&'static str, &'static str)],
}

#[derive(Clone, Copy, Debug)]
struct ScenarioCase {
    id: u8,
    name: &'static str,
    setup: &'static str,
    query: &'static str,
    columns: &'static [(&'static str, ExpectedType)],
    rows: &'static [&'static [ExpectedCell]],
    nodes_added: usize,
    relationships_added: usize,
    properties_added: usize,
    created_relationship: Option<ExpectedCreatedRelationship>,
}

const CASES: [ScenarioCase; 18] = [
    ScenarioCase {
        id: 1,
        name: "Creating a relationship",
        setup: "CREATE (:A), (:B)",
        query: "MATCH (a:A), (b:B)\nMERGE (a)-[r:TYPE]->(b)\nRETURN count(*)",
        columns: &[("count(*)", ExpectedType::Integer)],
        rows: &[&[ExpectedCell::Integer(1)]],
        nodes_added: 0,
        relationships_added: 1,
        properties_added: 0,
        created_relationship: Some(ExpectedCreatedRelationship {
            relationship_type: "TYPE",
            source: ExpectedNode::Label("A"),
            target: ExpectedNode::Label("B"),
            properties: &[],
        }),
    },
    ScenarioCase {
        id: 2,
        name: "Matching a relationship",
        setup: "CREATE (a:A), (b:B)\nCREATE (a)-[:TYPE]->(b)",
        query: "MATCH (a:A), (b:B)\nMERGE (a)-[r:TYPE]->(b)\nRETURN count(r)",
        columns: &[("count(r)", ExpectedType::Integer)],
        rows: &[&[ExpectedCell::Integer(1)]],
        nodes_added: 0,
        relationships_added: 0,
        properties_added: 0,
        created_relationship: None,
    },
    ScenarioCase {
        id: 3,
        name: "Matching two relationships",
        setup: "CREATE (a:A), (b:B)\nCREATE (a)-[:TYPE]->(b)\nCREATE (a)-[:TYPE]->(b)",
        query: "MATCH (a:A), (b:B)\nMERGE (a)-[r:TYPE]->(b)\nRETURN count(r)",
        columns: &[("count(r)", ExpectedType::Integer)],
        rows: &[&[ExpectedCell::Integer(2)]],
        nodes_added: 0,
        relationships_added: 0,
        properties_added: 0,
        created_relationship: None,
    },
    ScenarioCase {
        id: 4,
        name: "Using bound variables from other updating clause",
        setup: "",
        query: "CREATE (a), (b)\nMERGE (a)-[:X]->(b)\nRETURN count(a)",
        columns: &[("count(a)", ExpectedType::Integer)],
        rows: &[&[ExpectedCell::Integer(1)]],
        nodes_added: 2,
        relationships_added: 1,
        properties_added: 0,
        created_relationship: Some(ExpectedCreatedRelationship {
            relationship_type: "X",
            source: ExpectedNode::Any,
            target: ExpectedNode::Any,
            properties: &[],
        }),
    },
    ScenarioCase {
        id: 5,
        name: "Filtering relationships",
        setup: "CREATE (a:A), (b:B)\nCREATE (a)-[:TYPE {name: 'r1'}]->(b)\nCREATE (a)-[:TYPE {name: 'r2'}]->(b)",
        query: "MATCH (a:A), (b:B)\nMERGE (a)-[r:TYPE {name: 'r2'}]->(b)\nRETURN count(r)",
        columns: &[("count(r)", ExpectedType::Integer)],
        rows: &[&[ExpectedCell::Integer(1)]],
        nodes_added: 0,
        relationships_added: 0,
        properties_added: 0,
        created_relationship: None,
    },
    ScenarioCase {
        id: 6,
        name: "Creating relationship when all matches filtered out",
        setup: "CREATE (a:A), (b:B)\nCREATE (a)-[:TYPE {name: 'r1'}]->(b)",
        query: "MATCH (a:A), (b:B)\nMERGE (a)-[r:TYPE {name: 'r2'}]->(b)\nRETURN count(r)",
        columns: &[("count(r)", ExpectedType::Integer)],
        rows: &[&[ExpectedCell::Integer(1)]],
        nodes_added: 0,
        relationships_added: 1,
        properties_added: 1,
        created_relationship: Some(ExpectedCreatedRelationship {
            relationship_type: "TYPE",
            source: ExpectedNode::Label("A"),
            target: ExpectedNode::Label("B"),
            properties: &[("name", "r2")],
        }),
    },
    ScenarioCase {
        id: 7,
        name: "Matching incoming relationship",
        setup: "CREATE (a:A), (b:B)\nCREATE (b)-[:TYPE]->(a)\nCREATE (a)-[:TYPE]->(b)",
        query: "MATCH (a:A), (b:B)\nMERGE (a)<-[r:TYPE]-(b)\nRETURN count(r)",
        columns: &[("count(r)", ExpectedType::Integer)],
        rows: &[&[ExpectedCell::Integer(1)]],
        nodes_added: 0,
        relationships_added: 0,
        properties_added: 0,
        created_relationship: None,
    },
    ScenarioCase {
        id: 8,
        name: "Creating relationship with property",
        setup: "CREATE (a:A), (b:B)",
        query: "MATCH (a:A), (b:B)\nMERGE (a)-[r:TYPE {name: 'Lola'}]->(b)\nRETURN count(r)",
        columns: &[("count(r)", ExpectedType::Integer)],
        rows: &[&[ExpectedCell::Integer(1)]],
        nodes_added: 0,
        relationships_added: 1,
        properties_added: 1,
        created_relationship: Some(ExpectedCreatedRelationship {
            relationship_type: "TYPE",
            source: ExpectedNode::Label("A"),
            target: ExpectedNode::Label("B"),
            properties: &[("name", "Lola")],
        }),
    },
    ScenarioCase {
        id: 9,
        name: "Creating relationship using merged nodes",
        setup: "CREATE (a:A), (b:B)",
        query: "MERGE (a:A)\nMERGE (b:B)\nMERGE (a)-[:FOO]->(b)",
        columns: &[],
        rows: &[],
        nodes_added: 0,
        relationships_added: 1,
        properties_added: 0,
        created_relationship: Some(ExpectedCreatedRelationship {
            relationship_type: "FOO",
            source: ExpectedNode::Label("A"),
            target: ExpectedNode::Label("B"),
            properties: &[],
        }),
    },
    ScenarioCase {
        id: 10,
        name: "Merge should bind a path",
        setup: "",
        query: "MERGE (a {num: 1})\nMERGE (b {num: 2})\nMERGE p = (a)-[:R]->(b)\nRETURN p",
        columns: &[("p", ExpectedType::Path)],
        rows: &[&[ExpectedCell::Path {
            start_num: 1,
            relationship_type: "R",
            end_num: 2,
        }]],
        nodes_added: 2,
        relationships_added: 1,
        properties_added: 2,
        created_relationship: Some(ExpectedCreatedRelationship {
            relationship_type: "R",
            source: ExpectedNode::IntegerProperty("num", 1),
            target: ExpectedNode::IntegerProperty("num", 2),
            properties: &[],
        }),
    },
    ScenarioCase {
        id: 11,
        name: "Use outgoing direction when unspecified",
        setup: "",
        query: "CREATE (a {id: 2}), (b {id: 1})\nMERGE (a)-[r:KNOWS]-(b)\nRETURN startNode(r).id AS s, endNode(r).id AS e",
        columns: &[("s", ExpectedType::Integer), ("e", ExpectedType::Integer)],
        rows: &[&[ExpectedCell::Integer(2), ExpectedCell::Integer(1)]],
        nodes_added: 2,
        relationships_added: 1,
        properties_added: 2,
        created_relationship: Some(ExpectedCreatedRelationship {
            relationship_type: "KNOWS",
            source: ExpectedNode::IntegerProperty("id", 2),
            target: ExpectedNode::IntegerProperty("id", 1),
            properties: &[],
        }),
    },
    ScenarioCase {
        id: 12,
        name: "Match outgoing relationship when direction unspecified",
        setup: "CREATE (a {id: 1}), (b {id: 2})\nCREATE (a)-[:KNOWS]->(b)",
        query: "MATCH (a {id: 2}), (b {id: 1})\nMERGE (a)-[r:KNOWS]-(b)\nRETURN r",
        columns: &[("r", ExpectedType::Relationship)],
        rows: &[&[ExpectedCell::Relationship {
            relationship_type: "KNOWS",
            name: None,
        }]],
        nodes_added: 0,
        relationships_added: 0,
        properties_added: 0,
        created_relationship: None,
    },
    ScenarioCase {
        id: 13,
        name: "Match both incoming and outgoing relationships when direction unspecified",
        setup: "CREATE (a {id: 2}), (b {id: 1}), (c {id: 1}), (d {id: 2})\nCREATE (a)-[:KNOWS {name: 'ab'}]->(b)\nCREATE (c)-[:KNOWS {name: 'cd'}]->(d)",
        query: "MATCH (a {id: 2})--(b {id: 1})\nMERGE (a)-[r:KNOWS]-(b)\nRETURN r",
        columns: &[("r", ExpectedType::Relationship)],
        rows: &[
            &[ExpectedCell::Relationship {
                relationship_type: "KNOWS",
                name: Some("ab"),
            }],
            &[ExpectedCell::Relationship {
                relationship_type: "KNOWS",
                name: Some("cd"),
            }],
        ],
        nodes_added: 0,
        relationships_added: 0,
        properties_added: 0,
        created_relationship: None,
    },
    ScenarioCase {
        id: 15,
        name: "Matching using list property",
        setup: "CREATE (a:A), (b:B)\nCREATE (a)-[:T {numbers: [42, 43]}]->(b)",
        query: "MATCH (a:A), (b:B)\nMERGE (a)-[r:T {numbers: [42, 43]}]->(b)\nRETURN count(*)",
        columns: &[("count(*)", ExpectedType::Integer)],
        rows: &[&[ExpectedCell::Integer(1)]],
        nodes_added: 0,
        relationships_added: 0,
        properties_added: 0,
        created_relationship: None,
    },
    ScenarioCase {
        id: 16,
        name: "Aliasing of existing nodes 1",
        setup: "CREATE ({id: 0})",
        query: "MATCH (n)\nMATCH (m)\nWITH n AS a, m AS b\nMERGE (a)-[r:T]->(b)\nRETURN a.id AS a, b.id AS b",
        columns: &[("a", ExpectedType::Integer), ("b", ExpectedType::Integer)],
        rows: &[&[ExpectedCell::Integer(0), ExpectedCell::Integer(0)]],
        nodes_added: 0,
        relationships_added: 1,
        properties_added: 0,
        created_relationship: Some(ExpectedCreatedRelationship {
            relationship_type: "T",
            source: ExpectedNode::IntegerProperty("id", 0),
            target: ExpectedNode::IntegerProperty("id", 0),
            properties: &[],
        }),
    },
    ScenarioCase {
        id: 17,
        name: "Aliasing of existing nodes 2",
        setup: "CREATE ({id: 0})",
        query: "MATCH (n)\nWITH n AS a, n AS b\nMERGE (a)-[r:T]->(b)\nRETURN a.id AS a",
        columns: &[("a", ExpectedType::Integer)],
        rows: &[&[ExpectedCell::Integer(0)]],
        nodes_added: 0,
        relationships_added: 1,
        properties_added: 0,
        created_relationship: Some(ExpectedCreatedRelationship {
            relationship_type: "T",
            source: ExpectedNode::IntegerProperty("id", 0),
            target: ExpectedNode::IntegerProperty("id", 0),
            properties: &[],
        }),
    },
    ScenarioCase {
        id: 18,
        name: "Double aliasing of existing nodes 1",
        setup: "CREATE ({id: 0})",
        query: "MATCH (n)\nMATCH (m)\nWITH n AS a, m AS b\nMERGE (a)-[:T]->(b)\nWITH a AS x, b AS y\nMERGE (a)\nMERGE (b)\nMERGE (a)-[:T]->(b)\nRETURN x.id AS x, y.id AS y",
        columns: &[("x", ExpectedType::Integer), ("y", ExpectedType::Integer)],
        rows: &[&[ExpectedCell::Integer(0), ExpectedCell::Integer(0)]],
        nodes_added: 0,
        relationships_added: 1,
        properties_added: 0,
        created_relationship: Some(ExpectedCreatedRelationship {
            relationship_type: "T",
            source: ExpectedNode::IntegerProperty("id", 0),
            target: ExpectedNode::IntegerProperty("id", 0),
            properties: &[],
        }),
    },
    ScenarioCase {
        id: 19,
        name: "Double aliasing of existing nodes 2",
        setup: "CREATE ({id: 0})",
        query: "MATCH (n)\nWITH n AS a\nMERGE (c)\nMERGE (a)-[:T]->(c)\nWITH a AS x\nMERGE (c)\nMERGE (x)-[:T]->(c)\nRETURN x.id AS x",
        columns: &[("x", ExpectedType::Integer)],
        rows: &[&[ExpectedCell::Integer(0)]],
        nodes_added: 0,
        relationships_added: 1,
        properties_added: 0,
        created_relationship: Some(ExpectedCreatedRelationship {
            relationship_type: "T",
            source: ExpectedNode::IntegerProperty("id", 0),
            target: ExpectedNode::IntegerProperty("id", 0),
            properties: &[],
        }),
    },
];

const SCENARIO_14_CASE: ScenarioCase = ScenarioCase {
    id: 14,
    name: "Using list properties via variable",
    setup: "",
    query: "CREATE (a:Foo), (b:Bar)\nWITH a, b\nUNWIND ['a,b', 'a,b'] AS str\nWITH a, b, split(str, ',') AS roles\nMERGE (a)-[r:FB {foobar: roles}]->(b)\nRETURN count(*)",
    columns: &[("count(*)", ExpectedType::Integer)],
    rows: &[&[ExpectedCell::Integer(2)]],
    nodes_added: 2,
    relationships_added: 1,
    properties_added: 1,
    created_relationship: None,
};

const MATCH8_OPTIONAL_AFTER_MERGE_CASE: ScenarioCase = ScenarioCase {
    id: 249,
    name: "Match8 [2] counts OPTIONAL rows against the MERGE overlay",
    setup: "CREATE (a:A), (b:B)\nCREATE (a)-[:T1]->(b), (b)-[:T2]->(a)",
    query: "MATCH (a)\nMERGE (b)\nWITH *\nOPTIONAL MATCH (a)--(b)\nRETURN count(*)",
    columns: &[("count(*)", ExpectedType::Integer)],
    rows: &[&[ExpectedCell::Integer(6)]],
    nodes_added: 0,
    relationships_added: 0,
    properties_added: 0,
    created_relationship: None,
};

const SCENARIO_20_CASE: ScenarioCase = ScenarioCase {
    id: 20,
    name: "Do not match on deleted entities",
    setup: "CREATE (a:A)\nCREATE (b1:B {num: 0}), (b2:B {num: 1})\nCREATE (c1:C), (c2:C)\nCREATE (a)-[:REL]->(b1), (a)-[:REL]->(b2), (b1)-[:REL]->(c1), (b2)-[:REL]->(c2)",
    query: "MATCH (a:A)-[ab]->(b:B)-[bc]->(c:C)\nDELETE ab, bc, b, c\nMERGE (newB:B {num: 1})\nMERGE (a)-[:REL]->(newB)\nMERGE (newC:C)\nMERGE (newB)-[:REL]->(newC)",
    columns: &[],
    rows: &[],
    nodes_added: 2,
    relationships_added: 2,
    properties_added: 1,
    created_relationship: None,
};

const SCENARIO_21_CASE: ScenarioCase = ScenarioCase {
    id: 21,
    name: "Do not match on deleted relationships",
    setup: "CREATE (a:A), (b:B)\nCREATE (a)-[:T {name: 'rel1'}]->(b), (a)-[:T {name: 'rel2'}]->(b)",
    query: "MATCH (a)-[t:T]->(b)\nDELETE t\nMERGE (a)-[t2:T {name: 'rel3'}]->(b)\nRETURN t2.name",
    columns: &[("t2.name", ExpectedType::String)],
    rows: &[
        &[ExpectedCell::String("rel3")],
        &[ExpectedCell::String("rel3")],
    ],
    nodes_added: 0,
    relationships_added: 1,
    properties_added: 1,
    created_relationship: None,
};

const DELETED_ENDPOINT_CASE: ScenarioCase = ScenarioCase {
    id: 253,
    name: "Deleted endpoint rows do not reach a later relationship MERGE",
    setup: "CREATE (dead:A:B), (deadC:C), (live:A), (liveB:B), (liveC:C)\nCREATE (dead)-[:REL]->(dead), (dead)-[:REL]->(deadC), (live)-[:REL]->(liveB), (liveB)-[:REL]->(liveC)",
    query: "MATCH (a:A)-[ab]->(b:B)-[bc]->(c:C)\nDELETE ab, bc, b, c\nMERGE (newB:B {num: 1})\nMERGE (a)-[:REL]->(newB)\nMERGE (newC:C)\nMERGE (newB)-[:REL]->(newC)",
    columns: &[],
    rows: &[],
    nodes_added: 2,
    relationships_added: 2,
    properties_added: 1,
    created_relationship: None,
};

const DISTINCT_SCHEDULED_LIST_CASE: ScenarioCase = ScenarioCase {
    id: 250,
    name: "Distinct scheduled strings remain distinct relationship keys",
    setup: "",
    query: "CREATE (origin:Source), (destination:Sink)\nWITH origin AS from, destination AS to\nUNWIND ['red|blue', 'green|gold'] AS encoded\nWITH from, to, split(encoded, '|') AS grants\nMERGE (from)-[link:HAS_ACCESS {permissions: grants}]->(to)\nRETURN count(*)",
    columns: &[("count(*)", ExpectedType::Integer)],
    rows: &[&[ExpectedCell::Integer(2)]],
    nodes_added: 2,
    relationships_added: 2,
    properties_added: 2,
    created_relationship: None,
};

const DUPLICATE_PARENT_READ_OWN_WRITES_CASE: ScenarioCase = ScenarioCase {
    id: 255,
    name: "Duplicate parent rows share one statement-local MERGE effect",
    setup: "CREATE (a:A), (b:B)\nCREATE (a)-[:SEED]->(b)\nCREATE (a)-[:SEED]->(b)",
    query: "MATCH (a:A)-[:SEED]->(b:B)\nMERGE (a)-[r:T {name: 'once'}]->(b)\nRETURN count(r)",
    columns: &[("count(r)", ExpectedType::Integer)],
    rows: &[&[ExpectedCell::Integer(2)]],
    nodes_added: 0,
    relationships_added: 1,
    properties_added: 1,
    created_relationship: Some(ExpectedCreatedRelationship {
        relationship_type: "T",
        source: ExpectedNode::Label("A"),
        target: ExpectedNode::Label("B"),
        properties: &[("name", "once")],
    }),
};

const INCOMING_CREATE_CASE: ScenarioCase = ScenarioCase {
    id: 254,
    name: "Incoming MERGE creates in stored incoming orientation",
    setup: "CREATE (a:A), (b:B)",
    query: "MATCH (a:A), (b:B)\nMERGE (a)<-[r:T]-(b)\nRETURN count(r)",
    columns: &[("count(r)", ExpectedType::Integer)],
    rows: &[&[ExpectedCell::Integer(1)]],
    nodes_added: 0,
    relationships_added: 1,
    properties_added: 0,
    created_relationship: Some(ExpectedCreatedRelationship {
        relationship_type: "T",
        source: ExpectedNode::Label("B"),
        target: ExpectedNode::Label("A"),
        properties: &[],
    }),
};

const UNDIRECTED_SELF_LOOP_CASE: ScenarioCase = ScenarioCase {
    id: 252,
    name: "Undirected self-loop is scanned once",
    setup: "CREATE (a:A)\nCREATE (a)-[:T]->(a)",
    query: "MATCH (a:A)\nMERGE (a)-[r:T]-(a)\nRETURN count(r)",
    columns: &[("count(r)", ExpectedType::Integer)],
    rows: &[&[ExpectedCell::Integer(1)]],
    nodes_added: 0,
    relationships_added: 0,
    properties_added: 0,
    created_relationship: None,
};

#[derive(Clone, Debug)]
struct Fixture {
    graph: GraphStore,
    bookmark: Bookmark,
}

impl Fixture {
    fn build(case: &ScenarioCase) -> Result<Self> {
        let mut graph = GraphStore::default();
        if !case.setup.trim().is_empty() {
            let setup = QueryEngine.execute(case.setup, &mut context(&graph, None, false))?;
            if !setup.temporal_mutations.is_empty()
                || setup.administrative.is_some()
                || !setup.vector_searches.is_empty()
                || setup.runtime_replans != 0
            {
                return Err(Error::internal(format!(
                    "Merge5 [{}] setup emitted unrelated execution state",
                    case.id
                )));
            }
            for mutation in setup.graph_mutations {
                graph.apply(mutation)?;
            }
        }
        Ok(Self {
            bookmark: Bookmark {
                term: 91,
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

fn next_ids(graph: &GraphStore) -> Result<(u64, u64)> {
    let next_node = graph
        .nodes()
        .map(|node| node.id().0)
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or_else(|| Error::new(ErrorCode::ResultBudgetExceeded, "test node ID overflow"))?;
    let next_edge = graph
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
    Ok((next_node, next_edge))
}

fn context<'a>(
    graph: &'a GraphStore,
    backend: Option<&'a dyn ExecutionBackend>,
    require_native_execution: bool,
) -> ExecutionContext<'a> {
    let (next_node_id, next_edge_id) = next_ids(graph).expect("test graph IDs must be valid");
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
            term: 91,
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
        max_result_rows: MAX_RESULT_ROWS,
        max_batch_rows: 3,
        optimizer_statistics: None,
        backend,
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(30)),
        resolved_query_at_time_nanos: None,
    }
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum NormalizedCell {
    Integer(i64),
    String(String),
    Relationship {
        relationship_type: String,
        name: Option<String>,
    },
    Path {
        start_num: i64,
        relationship_type: String,
        end_num: i64,
    },
}

fn normalize_cell(value: &ResultValue) -> Result<NormalizedCell> {
    match value {
        ResultValue::Scalar(ScalarValue::Integer(value)) => Ok(NormalizedCell::Integer(*value)),
        ResultValue::Scalar(ScalarValue::String(value)) => {
            Ok(NormalizedCell::String(value.to_string()))
        }
        ResultValue::Relationship(relationship) => {
            let mut name = None;
            for (property, value) in &relationship.properties {
                if property != "name" || name.is_some() {
                    return Err(Error::internal(format!(
                        "unexpected relationship result property `{property}`"
                    )));
                }
                let ScalarValue::String(value) = value else {
                    return Err(Error::internal(
                        "relationship result name property was not STRING",
                    ));
                };
                name = Some(value.to_string());
            }
            Ok(NormalizedCell::Relationship {
                relationship_type: relationship.relationship_type.clone(),
                name,
            })
        }
        ResultValue::Path {
            nodes,
            relationships,
        } => {
            let [start, end] = nodes.as_slice() else {
                return Err(Error::internal(format!(
                    "Merge5 one-hop path returned {} nodes",
                    nodes.len()
                )));
            };
            let [relationship] = relationships.as_slice() else {
                return Err(Error::internal(format!(
                    "Merge5 one-hop path returned {} relationships",
                    relationships.len()
                )));
            };
            if start.layer != Layer::Observed
                || end.layer != Layer::Observed
                || !start.labels.is_empty()
                || !end.labels.is_empty()
                || start.properties.len() != 1
                || end.properties.len() != 1
                || relationship.layer != Layer::Observed
                || relationship.source != start.id
                || relationship.target != end.id
                || !relationship.properties.is_empty()
            {
                return Err(Error::internal(format!(
                    "Merge5 path changed lexical orientation or canonical shape: nodes={nodes:?}, relationships={relationships:?}"
                )));
            }
            let Some(ScalarValue::Integer(start_num)) = start.properties.get("num") else {
                return Err(Error::internal(format!(
                    "Merge5 path start node lost its integer `num` property: {start:?}"
                )));
            };
            let Some(ScalarValue::Integer(end_num)) = end.properties.get("num") else {
                return Err(Error::internal(format!(
                    "Merge5 path end node lost its integer `num` property: {end:?}"
                )));
            };
            Ok(NormalizedCell::Path {
                start_num: *start_num,
                relationship_type: relationship.relationship_type.clone(),
                end_num: *end_num,
            })
        }
        value => Err(Error::internal(format!(
            "unexpected Merge5 result value {value:?}"
        ))),
    }
}

fn expected_cell(value: ExpectedCell) -> NormalizedCell {
    match value {
        ExpectedCell::Integer(value) => NormalizedCell::Integer(value),
        ExpectedCell::String(value) => NormalizedCell::String(value.to_owned()),
        ExpectedCell::Relationship {
            relationship_type,
            name,
        } => NormalizedCell::Relationship {
            relationship_type: relationship_type.to_owned(),
            name: name.map(str::to_owned),
        },
        ExpectedCell::Path {
            start_num,
            relationship_type,
            end_num,
        } => NormalizedCell::Path {
            start_num,
            relationship_type: relationship_type.to_owned(),
            end_num,
        },
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GraphMetrics {
    nodes: usize,
    relationships: usize,
    properties: usize,
}

fn graph_metrics(graph: &GraphStore) -> GraphMetrics {
    GraphMetrics {
        nodes: graph.node_count(),
        relationships: graph.edge_count(),
        properties: graph
            .nodes()
            .map(|node| node.properties().len())
            .sum::<usize>()
            .saturating_add(
                graph
                    .edges()
                    .map(|edge| edge.properties().len())
                    .sum::<usize>(),
            ),
    }
}

fn node_matches(graph: &GraphStore, node: irongraph::NodeId, expected: ExpectedNode) -> bool {
    let Some(node) = graph.node(node) else {
        return false;
    };
    match expected {
        ExpectedNode::Any => true,
        ExpectedNode::Label(name) => graph
            .catalog()
            .label(name)
            .is_some_and(|label| node.labels().contains(&label)),
        ExpectedNode::IntegerProperty(name, expected) => graph
            .catalog()
            .property(name)
            .and_then(|property| node.property(property))
            .is_some_and(|value| value == ScalarValue::Integer(expected)),
    }
}

fn assert_created_relationship(
    before: &GraphStore,
    after: &GraphStore,
    expected: ExpectedCreatedRelationship,
    label: &str,
) -> Result<()> {
    let created = after
        .edges()
        .filter(|edge| before.edge(edge.id()).is_none())
        .collect::<Vec<_>>();
    let [edge] = created.as_slice() else {
        return Err(Error::internal(format!(
            "{label}: expected one created relationship, got {}",
            created.len()
        )));
    };
    let actual_type = after
        .catalog()
        .relationship_type_name(edge.relationship_type())
        .ok_or_else(|| Error::internal(format!("{label}: relationship type disappeared")))?;
    let actual_properties =
        edge.properties()
            .into_iter()
            .map(|(property, value)| {
                let name = after.catalog().property_name(property).ok_or_else(|| {
                    Error::internal(format!("{label}: property name disappeared"))
                })?;
                let ScalarValue::String(value) = value else {
                    return Err(Error::internal(format!(
                        "{label}: created relationship property was not STRING"
                    )));
                };
                Ok((name.to_owned(), value.to_string()))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
    let expected_properties = expected
        .properties
        .iter()
        .map(|(name, value)| ((*name).to_owned(), (*value).to_owned()))
        .collect::<BTreeMap<_, _>>();
    if actual_type != expected.relationship_type
        || edge.layer() != Layer::Observed
        || actual_properties != expected_properties
        || !node_matches(after, edge.source(), expected.source)
        || !node_matches(after, edge.target(), expected.target)
        || (matches!(
            (expected.source, expected.target),
            (ExpectedNode::Any, ExpectedNode::Any)
        ) && edge.source() == edge.target())
    {
        return Err(Error::internal(format!(
            "{label}: created relationship has wrong type, endpoints, or properties: {edge:?}"
        )));
    }
    Ok(())
}

fn assert_case_output(
    case: &ScenarioCase,
    fixture: &Fixture,
    output: &ExecutionOutput,
) -> Result<()> {
    let label = format!("{MERGE5_FEATURE} / [{}] {}", case.id, case.name);
    let expected_schema = case
        .columns
        .iter()
        .map(|(name, value_type)| ((*name).to_owned(), value_type.column_type()))
        .collect::<Vec<_>>();
    if output.result.schema != expected_schema
        || output.result.bookmark != fixture.bookmark
        || output.result.truncated
        || !output.temporal_mutations.is_empty()
        || output.administrative.is_some()
        || !output.vector_searches.is_empty()
        || output.runtime_replans != 0
    {
        return Err(Error::internal(format!(
            "{label}: result envelope changed: {output:#?}"
        )));
    }
    let expected_stats = StatementStats {
        nodes_created: case.nodes_added as u64,
        relationships_created: case.relationships_added as u64,
        ..StatementStats::default()
    };
    if output.result.statistics != expected_stats {
        return Err(Error::internal(format!(
            "{label}: statement statistics mismatch: expected {expected_stats:?}, got {:?}",
            output.result.statistics
        )));
    }

    let mut actual_rows = Vec::new();
    for batch in &output.result.batches {
        if !batch.validate()
            || batch.columns.len() != case.columns.len()
            || batch
                .columns
                .iter()
                .zip(case.columns)
                .any(|(actual, expected)| {
                    actual.name != expected.0 || actual.value_type != expected.1.column_type()
                })
        {
            return Err(Error::internal(format!(
                "{label}: malformed result batch {batch:?}"
            )));
        }
        for row in 0..batch.row_count {
            actual_rows.push(
                batch
                    .columns
                    .iter()
                    .map(|column| normalize_cell(&column.values[row]))
                    .collect::<Result<Vec<_>>>()?,
            );
        }
    }
    let mut expected_rows = case
        .rows
        .iter()
        .map(|row| row.iter().copied().map(expected_cell).collect::<Vec<_>>())
        .collect::<Vec<_>>();
    actual_rows.sort();
    expected_rows.sort();
    if actual_rows != expected_rows {
        return Err(Error::internal(format!(
            "{label}: result rows mismatch: expected {expected_rows:?}, got {actual_rows:?}"
        )));
    }

    let before_metrics = graph_metrics(&fixture.graph);
    let mut after = fixture.graph.clone();
    for mutation in &output.graph_mutations {
        after.apply(mutation.clone())?;
    }
    let after_metrics = graph_metrics(&after);
    if after_metrics.nodes.checked_sub(before_metrics.nodes) != Some(case.nodes_added)
        || after_metrics
            .relationships
            .checked_sub(before_metrics.relationships)
            != Some(case.relationships_added)
        || after_metrics
            .properties
            .checked_sub(before_metrics.properties)
            != Some(case.properties_added)
        || output
            .graph_mutations
            .iter()
            .filter(|mutation| matches!(mutation, GraphMutation::InsertEdge(_)))
            .count()
            != case.relationships_added
        || output
            .graph_mutations
            .iter()
            .filter(|mutation| matches!(mutation, GraphMutation::InsertNode(_)))
            .count()
            != case.nodes_added
    {
        return Err(Error::internal(format!(
            "{label}: graph effects mismatch: before={before_metrics:?}, after={after_metrics:?}, mutations={:?}",
            output.graph_mutations
        )));
    }
    match case.created_relationship {
        Some(expected) => assert_created_relationship(&fixture.graph, &after, expected, &label)?,
        None if after_metrics != before_metrics => {
            return Err(Error::internal(format!(
                "{label}: a matching MERGE changed the graph"
            )));
        }
        None => {}
    }
    Ok(())
}

fn assert_cpu_case(index: usize) -> Result<()> {
    assert_cpu_scenario(&CASES[index])
}

fn assert_cpu_scenario(case: &ScenarioCase) -> Result<()> {
    let fixture = Fixture::build(case)?;
    let backend = StrictBoundRelationshipMergeBackend::strict_cpu(fixture.resident_image()?)?;
    let observations = backend.observations();
    let output = QueryEngine.execute(
        case.query,
        &mut context(&fixture.graph, Some(&backend), true),
    )?;
    assert_one_complete_command(case, &observations)?;
    assert_case_output(case, &fixture, &output)
}

macro_rules! cpu_case_test {
    ($name:ident, $index:expr) => {
        #[test]
        fn $name() -> Result<()> {
            assert_cpu_case($index)
        }
    };
}

cpu_case_test!(cpu_merge5_1_creates_bound_relationship, 0);
cpu_case_test!(cpu_merge5_2_reuses_matching_relationship, 1);
cpu_case_test!(cpu_merge5_3_preserves_two_parallel_matches, 2);
cpu_case_test!(cpu_merge5_4_uses_nodes_created_by_prior_clause, 3);
cpu_case_test!(cpu_merge5_5_filters_relationship_properties, 4);
cpu_case_test!(cpu_merge5_6_creates_when_property_filter_misses, 5);
cpu_case_test!(cpu_merge5_7_normalizes_incoming_direction, 6);
cpu_case_test!(cpu_merge5_8_creates_relationship_property, 7);
cpu_case_test!(cpu_merge5_9_uses_nodes_bound_by_prior_merges, 8);
cpu_case_test!(cpu_merge5_10_binds_one_hop_path, 9);
cpu_case_test!(cpu_merge5_11_uses_stored_outgoing_direction, 10);
cpu_case_test!(cpu_merge5_12_matches_undirected_outgoing_edge, 11);
cpu_case_test!(cpu_merge5_13_matches_both_undirected_orientations, 12);
cpu_case_test!(cpu_merge5_15_matches_list_property, 13);
cpu_case_test!(cpu_merge5_16_preserves_two_aliases, 14);
cpu_case_test!(cpu_merge5_17_preserves_self_alias, 15);
cpu_case_test!(
    cpu_merge5_18_preserves_hidden_slots_across_double_aliasing,
    16
);
cpu_case_test!(
    cpu_merge5_19_rebinds_shadowed_merge_nodes_after_aliasing,
    17
);

#[test]
fn cpu_match8_2_counts_optional_rows_after_node_merge_without_fallback() -> Result<()> {
    assert_cpu_scenario(&MATCH8_OPTIONAL_AFTER_MERGE_CASE)
}

fn execute_cpu_delete_case(
    case: &ScenarioCase,
) -> Result<(Fixture, ExecutionOutput, Arc<RouteObservations>)> {
    let fixture = Fixture::build(case)?;
    let backend = StrictBoundRelationshipMergeBackend::strict_cpu(fixture.resident_image()?)?;
    let observations = backend.observations();
    let output = QueryEngine.execute(
        case.query,
        &mut context(&fixture.graph, Some(&backend), true),
    )?;
    assert_one_complete_command(case, &observations)?;
    Ok((fixture, output, observations))
}

fn assert_scenario_20_output(fixture: &Fixture, output: &ExecutionOutput) -> Result<()> {
    let stats = output.result.statistics;
    if !output.result.schema.is_empty()
        || !output.result.batches.is_empty()
        || stats.nodes_created != 2
        || stats.nodes_deleted != 4
        || stats.relationships_created != 2
        || stats.relationships_deleted != 4
        || output
            .graph_mutations
            .iter()
            .filter(|mutation| matches!(mutation, GraphMutation::DeleteEdge { .. }))
            .count()
            != 4
        || output
            .graph_mutations
            .iter()
            .filter(|mutation| matches!(mutation, GraphMutation::DeleteNode { .. }))
            .count()
            != 4
        || output
            .graph_mutations
            .iter()
            .filter(|mutation| matches!(mutation, GraphMutation::InsertNode(_)))
            .count()
            != 2
        || output
            .graph_mutations
            .iter()
            .filter(|mutation| matches!(mutation, GraphMutation::InsertEdge(_)))
            .count()
            != 2
    {
        return Err(Error::internal(format!(
            "Merge5 [20] changed its one-command result or effects: {output:#?}"
        )));
    }
    let mut after = fixture.graph.clone();
    for mutation in &output.graph_mutations {
        after.apply(mutation.clone())?;
    }
    let b = after
        .catalog()
        .label("B")
        .ok_or_else(|| Error::internal("Merge5 [20] lost label B"))?;
    let c = after
        .catalog()
        .label("C")
        .ok_or_else(|| Error::internal("Merge5 [20] lost label C"))?;
    let num = after
        .catalog()
        .property("num")
        .ok_or_else(|| Error::internal("Merge5 [20] lost property num"))?;
    let b_nodes = after
        .nodes()
        .filter(|node| node.labels().contains(&b))
        .collect::<Vec<_>>();
    let c_nodes = after
        .nodes()
        .filter(|node| node.labels().contains(&c))
        .collect::<Vec<_>>();
    if after.node_count() != 3
        || after.edge_count() != 2
        || b_nodes.len() != 1
        || c_nodes.len() != 1
        || b_nodes[0].property(num) != Some(ScalarValue::Integer(1))
        || after
            .edges()
            .any(|edge| fixture.graph.edge(edge.id()).is_some())
    {
        return Err(Error::internal(format!(
            "Merge5 [20] reused a tombstoned entity: nodes={:?}, edges={:?}",
            after.nodes().collect::<Vec<_>>(),
            after.edges().collect::<Vec<_>>()
        )));
    }
    Ok(())
}

#[test]
fn cpu_merge5_20_delete_tombstones_remain_in_one_ordered_merge_command() -> Result<()> {
    let (fixture, output, _) = execute_cpu_delete_case(&SCENARIO_20_CASE)?;
    assert_scenario_20_output(&fixture, &output)
}

fn assert_scenario_21_output(fixture: &Fixture, output: &ExecutionOutput) -> Result<()> {
    let mut actual = Vec::new();
    for batch in &output.result.batches {
        if !batch.validate() || batch.columns.len() != 1 {
            return Err(Error::internal(format!(
                "Merge5 [21] returned a malformed batch: {batch:?}"
            )));
        }
        actual.extend(
            batch.columns[0]
                .values
                .iter()
                .map(normalize_cell)
                .collect::<Result<Vec<_>>>()?,
        );
    }
    actual.sort();
    if output.result.schema != vec![("t2.name".to_owned(), ColumnType::String)]
        || actual
            != vec![
                NormalizedCell::String("rel3".to_owned()),
                NormalizedCell::String("rel3".to_owned()),
            ]
        || output.result.statistics.relationships_created != 1
        || output.result.statistics.relationships_deleted != 2
    {
        return Err(Error::internal(format!(
            "Merge5 [21] changed its result or effects: {output:#?}"
        )));
    }
    let mut after = fixture.graph.clone();
    for mutation in &output.graph_mutations {
        after.apply(mutation.clone())?;
    }
    let name = after
        .catalog()
        .property("name")
        .ok_or_else(|| Error::internal("Merge5 [21] lost property name"))?;
    let edges = after.edges().collect::<Vec<_>>();
    if edges.len() != 1
        || fixture.graph.edge(edges[0].id()).is_some()
        || edges[0].property(name) != Some(ScalarValue::String(Arc::from("rel3")))
    {
        return Err(Error::internal(format!(
            "Merge5 [21] reused a deleted relationship: {edges:?}"
        )));
    }
    Ok(())
}

#[test]
fn cpu_merge5_21_deleted_parallel_relationships_do_not_match_later_merge() -> Result<()> {
    let (fixture, output, _) = execute_cpu_delete_case(&SCENARIO_21_CASE)?;
    assert_scenario_21_output(&fixture, &output)
}

fn assert_deleted_endpoint_case_output(fixture: &Fixture, output: &ExecutionOutput) -> Result<()> {
    let stats = output.result.statistics;
    if !output.result.schema.is_empty()
        || !output.result.batches.is_empty()
        || output.result.bookmark != fixture.bookmark
        || output.result.truncated
        || stats.nodes_created != 2
        || stats.nodes_deleted != 4
        || stats.relationships_created != 2
        || stats.relationships_deleted != 4
        || output
            .graph_mutations
            .iter()
            .filter(|mutation| matches!(mutation, GraphMutation::DeleteNode { .. }))
            .count()
            != 4
        || output
            .graph_mutations
            .iter()
            .filter(|mutation| matches!(mutation, GraphMutation::InsertEdge(_)))
            .count()
            != 2
        || output
            .graph_mutations
            .iter()
            .filter(|mutation| matches!(mutation, GraphMutation::DeleteEdge { .. }))
            .count()
            != 4
        || output
            .graph_mutations
            .iter()
            .filter(|mutation| matches!(mutation, GraphMutation::InsertNode(_)))
            .count()
            != 2
    {
        return Err(Error::internal(format!(
            "deleted-endpoint MERGE changed its result or effects: {output:#?}"
        )));
    }

    let mut after = fixture.graph.clone();
    for mutation in &output.graph_mutations {
        after.apply(mutation.clone())?;
    }
    let a = after
        .catalog()
        .label("A")
        .ok_or_else(|| Error::internal("deleted-endpoint case lost label A"))?;
    let b = after
        .catalog()
        .label("B")
        .ok_or_else(|| Error::internal("deleted-endpoint case lost label B"))?;
    let c = after
        .catalog()
        .label("C")
        .ok_or_else(|| Error::internal("deleted-endpoint case lost label C"))?;
    let num = after
        .catalog()
        .property("num")
        .ok_or_else(|| Error::internal("deleted-endpoint case lost property num"))?;
    let relationship_type = after
        .catalog()
        .relationship_type("REL")
        .ok_or_else(|| Error::internal("deleted-endpoint case lost relationship type REL"))?;
    let edges = after.edges().collect::<Vec<_>>();
    let live_a = after
        .nodes()
        .find(|node| node.labels().contains(&a))
        .ok_or_else(|| Error::internal("deleted-endpoint case lost its A node"))?;
    let live_b = after
        .nodes()
        .find(|node| node.labels().contains(&b))
        .ok_or_else(|| Error::internal("deleted-endpoint case lost its live B node"))?;
    let live_c = after
        .nodes()
        .find(|node| node.labels().contains(&c))
        .ok_or_else(|| Error::internal("deleted-endpoint case lost its live C node"))?;
    let endpoint_pairs = edges
        .iter()
        .map(|edge| (edge.source(), edge.target()))
        .collect::<Vec<_>>();
    if after.node_count() != 3
        || edges.len() != 2
        || live_a.labels().contains(&b)
        || live_b.property(num) != Some(ScalarValue::Integer(1))
        || edges.iter().any(|edge| {
            edge.relationship_type() != relationship_type
                || !edge.properties().is_empty()
                || fixture.graph.edge(edge.id()).is_some()
        })
        || !endpoint_pairs.contains(&(live_a.id(), live_b.id()))
        || !endpoint_pairs.contains(&(live_b.id(), live_c.id()))
    {
        return Err(Error::internal(format!(
            "deleted-endpoint MERGE retained the dead source or changed its live chain: nodes={:?}, edges={edges:?}",
            after.nodes().collect::<Vec<_>>()
        )));
    }
    Ok(())
}

#[test]
fn cpu_deleted_endpoint_rows_do_not_reach_later_relationship_merge() -> Result<()> {
    let (fixture, output, _) = execute_cpu_delete_case(&DELETED_ENDPOINT_CASE)?;
    assert_deleted_endpoint_case_output(&fixture, &output)
}

fn assert_scheduled_list_property_output(
    _case: &ScenarioCase,
    fixture: &Fixture,
    output: &ExecutionOutput,
    source_label: &str,
    target_label: &str,
    relationship_type: &str,
    property_name: &str,
    expected_lists: &[&[&str]],
) -> Result<()> {
    let expected_stats = StatementStats {
        nodes_created: 2,
        relationships_created: u64::try_from(expected_lists.len()).map_err(|_| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "scheduled-list test relationship count exceeds u64",
            )
        })?,
        labels_added: 2,
        ..StatementStats::default()
    };
    if output.result.schema != vec![("count(*)".to_owned(), ColumnType::Integer)]
        || output.result.statistics != expected_stats
        || output.result.bookmark != fixture.bookmark
        || output.result.truncated
        || output.result.batches.len() != 1
        || output.result.batches[0].row_count != 1
        || output.result.batches[0].columns.len() != 1
        || output.result.batches[0].columns[0].values
            != vec![ResultValue::Scalar(ScalarValue::Integer(2))]
        || !output.temporal_mutations.is_empty()
        || output.administrative.is_some()
        || !output.vector_searches.is_empty()
        || output.runtime_replans != 0
        || output
            .graph_mutations
            .iter()
            .filter(|mutation| matches!(mutation, GraphMutation::InsertNode(_)))
            .count()
            != 2
        || output
            .graph_mutations
            .iter()
            .filter(|mutation| matches!(mutation, GraphMutation::InsertEdge(_)))
            .count()
            != expected_lists.len()
    {
        return Err(Error::internal(format!(
            "scheduled list-property MERGE changed its result, effects, or one-command envelope: {output:#?}"
        )));
    }

    let mut after = fixture.graph.clone();
    for mutation in &output.graph_mutations {
        after.apply(mutation.clone())?;
    }
    let source_label = after
        .catalog()
        .label(source_label)
        .ok_or_else(|| Error::internal("scheduled list-property source label was not published"))?;
    let target_label = after
        .catalog()
        .label(target_label)
        .ok_or_else(|| Error::internal("scheduled list-property target label was not published"))?;
    let relationship_type = after
        .catalog()
        .relationship_type(relationship_type)
        .ok_or_else(|| {
            Error::internal("scheduled list-property relationship type was not published")
        })?;
    let property = after
        .catalog()
        .property(property_name)
        .ok_or_else(|| Error::internal("scheduled list-property name was not published"))?;
    let mut actual_lists = Vec::new();
    for edge in after.edges() {
        let properties = edge.properties();
        let [(actual_property, ScalarValue::List(value))] = properties.as_slice() else {
            return Err(Error::internal(format!(
                "scheduled list-property relationship changed its property shape: {edge:?}"
            )));
        };
        if edge.relationship_type() != relationship_type
            || *actual_property != property
            || edge.layer() != Layer::Observed
            || after
                .node(edge.source())
                .is_none_or(|node| !node.labels().contains(&source_label))
            || after
                .node(edge.target())
                .is_none_or(|node| !node.labels().contains(&target_label))
        {
            return Err(Error::internal(format!(
                "scheduled list-property relationship changed type, endpoints, or layer: {edge:?}"
            )));
        }
        actual_lists.push(
            value
                .items()?
                .into_iter()
                .map(|item| match item {
                    DocumentItem::Scalar(ScalarValue::String(value)) => Ok(value.to_string()),
                    item => Err(Error::internal(format!(
                        "scheduled split produced a non-string list item: {item:?}"
                    ))),
                })
                .collect::<Result<Vec<_>>>()?,
        );
    }
    let mut expected_lists = expected_lists
        .iter()
        .map(|values| values.iter().map(|value| (*value).to_owned()).collect())
        .collect::<Vec<Vec<String>>>();
    actual_lists.sort();
    expected_lists.sort();
    if after.node_count() != 2
        || after.edge_count() != expected_lists.len()
        || actual_lists != expected_lists
    {
        return Err(Error::internal(format!(
            "scheduled list-property MERGE collapsed or changed per-work-row values: expected={expected_lists:?}, actual={actual_lists:?}"
        )));
    }
    Ok(())
}

fn assert_cpu_scheduled_list_property_case(
    case: &ScenarioCase,
    source_label: &str,
    target_label: &str,
    relationship_type: &str,
    property_name: &str,
    expected_lists: &[&[&str]],
) -> Result<()> {
    let fixture = Fixture::build(case)?;
    let backend = StrictBoundRelationshipMergeBackend::strict_cpu(fixture.resident_image()?)?;
    let observations = backend.observations();
    let output = QueryEngine.execute(
        case.query,
        &mut context(&fixture.graph, Some(&backend), true),
    )?;
    assert_one_complete_command(case, &observations)?;
    assert_scheduled_list_property_output(
        case,
        &fixture,
        &output,
        source_label,
        target_label,
        relationship_type,
        property_name,
        expected_lists,
    )
}

#[test]
fn cpu_merge5_14_uses_scheduled_list_properties_without_fallback() -> Result<()> {
    assert_cpu_scheduled_list_property_case(
        &SCENARIO_14_CASE,
        "Foo",
        "Bar",
        "FB",
        "foobar",
        &[&["a", "b"]],
    )
}

#[test]
fn cpu_scheduled_list_property_merge_keeps_distinct_unwind_values_per_work_row() -> Result<()> {
    assert_cpu_scheduled_list_property_case(
        &DISTINCT_SCHEDULED_LIST_CASE,
        "Source",
        "Sink",
        "HAS_ACCESS",
        "permissions",
        &[&["red", "blue"], &["green", "gold"]],
    )
}

fn assert_two_node_ordered_overlay(
    case_index: usize,
    expected_rows: BTreeMap<Vec<i64>, usize>,
) -> Result<()> {
    let mut case = CASES[case_index];
    case.setup = "CREATE ({id: 0}), ({id: 1})";
    case.rows = &[];
    case.relationships_added = 4;
    case.created_relationship = None;

    let fixture = Fixture::build(&case)?;
    let backend = StrictBoundRelationshipMergeBackend::strict_cpu(fixture.resident_image()?)?;
    let observations = backend.observations();
    let output = QueryEngine.execute(
        case.query,
        &mut context(&fixture.graph, Some(&backend), true),
    )?;
    assert_one_complete_command(&case, &observations)?;

    let expected_schema = case
        .columns
        .iter()
        .map(|(name, value_type)| ((*name).to_owned(), value_type.column_type()))
        .collect::<Vec<_>>();
    let expected_stats = StatementStats {
        relationships_created: 4,
        ..StatementStats::default()
    };
    if output.result.schema != expected_schema
        || output.result.bookmark != fixture.bookmark
        || output.result.statistics != expected_stats
        || output.result.truncated
        || !output.temporal_mutations.is_empty()
        || output.administrative.is_some()
        || !output.vector_searches.is_empty()
        || output.runtime_replans != 0
        || output
            .graph_mutations
            .iter()
            .filter(|mutation| matches!(mutation, GraphMutation::InsertEdge(_)))
            .count()
            != 4
        || output
            .graph_mutations
            .iter()
            .filter(|mutation| {
                matches!(mutation, GraphMutation::DeclareRelationshipType { name, .. } if name == "T")
            })
            .count()
            != 1
        || output.graph_mutations.len() != 5
    {
        return Err(Error::internal(format!(
            "Merge5 [{}] two-node overlay changed its result envelope or effects: {output:#?}",
            case.id
        )));
    }

    let mut actual_rows = BTreeMap::<Vec<i64>, usize>::new();
    for batch in &output.result.batches {
        if !batch.validate() || batch.columns.len() != case.columns.len() {
            return Err(Error::internal(format!(
                "Merge5 [{}] two-node overlay returned a malformed batch: {batch:?}",
                case.id
            )));
        }
        for row in 0..batch.row_count {
            let values = batch
                .columns
                .iter()
                .map(|column| match &column.values[row] {
                    ResultValue::Scalar(ScalarValue::Integer(value)) => Ok(*value),
                    value => Err(Error::internal(format!(
                        "Merge5 [{}] two-node overlay returned non-integer {value:?}",
                        case.id
                    ))),
                })
                .collect::<Result<Vec<_>>>()?;
            *actual_rows.entry(values).or_default() += 1;
        }
    }
    if actual_rows != expected_rows {
        return Err(Error::internal(format!(
            "Merge5 [{}] two-node overlay lost row lineage: expected {expected_rows:?}, got {actual_rows:?}",
            case.id
        )));
    }

    let mut after = fixture.graph.clone();
    for mutation in output.graph_mutations {
        after.apply(mutation)?;
    }
    let Some(id_property) = after.catalog().property("id") else {
        return Err(Error::internal(
            "two-node ordered-overlay proof lost the `id` property token",
        ));
    };
    let mut endpoint_pairs = BTreeMap::<(i64, i64), usize>::new();
    for edge in after.edges() {
        let relationship_type = after
            .catalog()
            .relationship_type_name(edge.relationship_type());
        let source = after
            .node(edge.source())
            .and_then(|node| node.property(id_property));
        let target = after
            .node(edge.target())
            .and_then(|node| node.property(id_property));
        let (Some(ScalarValue::Integer(source)), Some(ScalarValue::Integer(target))) =
            (source, target)
        else {
            return Err(Error::internal(format!(
                "Merge5 [{}] two-node overlay lost endpoint identity: {edge:?}",
                case.id
            )));
        };
        if relationship_type != Some("T")
            || edge.layer() != Layer::Observed
            || !edge.properties().is_empty()
        {
            return Err(Error::internal(format!(
                "Merge5 [{}] two-node overlay changed relationship shape: {edge:?}",
                case.id
            )));
        }
        *endpoint_pairs.entry((source, target)).or_default() += 1;
    }
    let expected_pairs = BTreeMap::from([((0, 0), 1), ((0, 1), 1), ((1, 0), 1), ((1, 1), 1)]);
    if endpoint_pairs != expected_pairs {
        return Err(Error::internal(format!(
            "Merge5 [{}] two-node overlay did not publish one edge per ordered pair: {endpoint_pairs:?}",
            case.id
        )));
    }
    Ok(())
}

#[test]
fn cpu_merge5_18_two_node_overlay_preserves_original_alias_lineage() -> Result<()> {
    assert_two_node_ordered_overlay(
        16,
        BTreeMap::from([
            (vec![0, 0], 4),
            (vec![0, 1], 4),
            (vec![1, 0], 4),
            (vec![1, 1], 4),
        ]),
    )
}

#[test]
fn cpu_merge5_19_two_node_overlay_preserves_shadowed_alias_lineage() -> Result<()> {
    assert_two_node_ordered_overlay(17, BTreeMap::from([(vec![0], 4), (vec![1], 4)]))
}

#[test]
fn cpu_duplicate_parent_rows_preserve_multiplicity_with_one_read_own_writes_effect() -> Result<()> {
    assert_cpu_scenario(&DUPLICATE_PARENT_READ_OWN_WRITES_CASE)
}

#[test]
fn cpu_incoming_zero_match_creates_with_swapped_stored_endpoints() -> Result<()> {
    assert_cpu_scenario(&INCOMING_CREATE_CASE)
}

#[test]
fn cpu_undirected_self_loop_expands_once() -> Result<()> {
    assert_cpu_scenario(&UNDIRECTED_SELF_LOOP_CASE)
}

#[test]
fn cpu_list_key_mismatch_creates_one_relationship_with_the_exact_list_value() -> Result<()> {
    const CASE: ScenarioCase = ScenarioCase {
        id: 253,
        name: "A different list value is not a complete relationship match",
        setup: "CREATE (a:A), (b:B)\nCREATE (a)-[:T {numbers: [42, 43]}]->(b)",
        query: "MATCH (a:A), (b:B)\nMERGE (a)-[r:T {numbers: [42, 44]}]->(b)\nRETURN count(r)",
        columns: &[("count(r)", ExpectedType::Integer)],
        rows: &[&[ExpectedCell::Integer(1)]],
        nodes_added: 0,
        relationships_added: 1,
        properties_added: 1,
        created_relationship: None,
    };

    let fixture = Fixture::build(&CASE)?;
    let backend = StrictBoundRelationshipMergeBackend::strict_cpu(fixture.resident_image()?)?;
    let observations = backend.observations();
    let output = QueryEngine.execute(
        CASE.query,
        &mut context(&fixture.graph, Some(&backend), true),
    )?;
    assert_one_complete_command(&CASE, &observations)?;
    if output.result.statistics.relationships_created != 1
        || output.result.batches.len() != 1
        || output.result.batches[0].row_count != 1
        || output.result.batches[0].columns[0].values
            != vec![ResultValue::Scalar(ScalarValue::Integer(1))]
    {
        return Err(Error::internal(format!(
            "list-key mismatch returned the wrong result or effect: {output:#?}"
        )));
    }
    let [GraphMutation::InsertEdge(edge)] = output.graph_mutations.as_slice() else {
        return Err(Error::internal(format!(
            "list-key mismatch did not publish exactly one relationship: {:?}",
            output.graph_mutations
        )));
    };
    let [(_, ScalarValue::List(numbers))] = edge.properties.as_slice() else {
        return Err(Error::internal(format!(
            "created relationship did not retain its list key: {:?}",
            edge.properties
        )));
    };
    if numbers.items()?
        != vec![
            DocumentItem::Scalar(ScalarValue::Integer(42)),
            DocumentItem::Scalar(ScalarValue::Integer(44)),
        ]
    {
        return Err(Error::internal(
            "created relationship list key changed canonical value",
        ));
    }
    Ok(())
}

#[derive(Default)]
struct RouteObservations {
    pins: AtomicUsize,
    complete_commands: AtomicUsize,
    forbidden_calls: AtomicUsize,
    requests: Mutex<Vec<ResidentRowMutationRequest>>,
}

/// Strict real-Metal observer. Generation inspection and scratch reservation are allowed; every
/// semantic primitive except one pinned complete row-mutation command is poisoned. With native
/// execution mandatory, neither the generic CPU row engine nor a decomposed GPU prefix can pass.
struct StrictBoundRelationshipMergeBackend {
    inner: Box<dyn ExecutionBackend>,
    backend_kind: BackendKind,
    pinned: bool,
    expected_bookmark: Bookmark,
    expected_graph_revision: u64,
    observations: Arc<RouteObservations>,
}

impl StrictBoundRelationshipMergeBackend {
    fn strict_cpu(image: ResidentProjectImage) -> Result<Self> {
        let mut cpu = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
        cpu.admit_project(image)?;
        Self::new(Box::new(cpu), BackendKind::Cpu)
    }

    #[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
    fn real_metal(inner: MetalBackend) -> Result<Self> {
        Self::new(Box::new(inner), BackendKind::Metal)
    }

    fn new(inner: Box<dyn ExecutionBackend>, backend_kind: BackendKind) -> Result<Self> {
        let expected_bookmark = inner
            .resident_bookmark(PROJECT)
            .ok_or_else(|| Error::internal("strict bound-MERGE backend has no bookmark"))?;
        let expected_graph_revision = inner
            .resident_graph_revision(PROJECT)
            .ok_or_else(|| Error::internal("strict bound-MERGE backend has no graph revision"))?;
        Ok(Self {
            inner,
            backend_kind,
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
            format!("strict bound-relationship MERGE rejected fallback route `{route}`"),
        ))
    }
}

impl ExecutionBackend for StrictBoundRelationshipMergeBackend {
    fn kind(&self) -> BackendKind {
        self.backend_kind
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
        if inner.kind() != self.backend_kind
            || inner.resident_bookmark(project) != Some(self.expected_bookmark)
            || inner.resident_graph_revision(project) != Some(self.expected_graph_revision)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "strict bound-MERGE pin changed Metal provenance or generation",
            ));
        }
        self.observations.pins.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Self {
            inner,
            backend_kind: self.backend_kind,
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
                "bound-relationship MERGE request escaped its pinned Metal generation",
            ));
        }
        self.observations
            .complete_commands
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

fn assert_scenario_10_compiler_contract(request: &ResidentRowMutationRequest) -> Result<()> {
    let Some(body) = request.bound_relationship_merge_body() else {
        return Err(Error::internal(
            "Merge5 [10] did not use the bound-relationship body",
        ));
    };
    let [
        ResidentRowCreateCommand::CreateNode(start),
        ResidentRowCreateCommand::CreateNode(end),
        ResidentRowCreateCommand::CreateRelationship(relationship),
    ] = body.program.commands.as_slice()
    else {
        return Err(Error::internal(format!(
            "Merge5 [10] compiler changed its structural command stream: {:?}",
            body.program.commands
        )));
    };
    let [
        ResidentRowBoundRelationshipCommand::MergeNode,
        ResidentRowBoundRelationshipCommand::MergeNode,
        ResidentRowBoundRelationshipCommand::MergeRelationship {
            direction: ResidentRowBoundRelationshipDirection::Directed,
        },
    ] = body.commands.as_slice()
    else {
        return Err(Error::internal(format!(
            "Merge5 [10] compiler changed its semantic command stream: {:?}",
            body.commands
        )));
    };
    let [
        ResidentRowBoundRelationshipOutput::Path {
            name,
            start_entity,
            relationship_entity,
            end_entity,
        },
    ] = body.outputs.as_slice()
    else {
        return Err(Error::internal(format!(
            "Merge5 [10] compiler did not emit exactly one path output: {:?}",
            body.outputs
        )));
    };
    let [start_property] = start.properties.as_slice() else {
        return Err(Error::internal(format!(
            "Merge5 [10] start node key changed: {:?}",
            start.properties
        )));
    };
    let [end_property] = end.properties.as_slice() else {
        return Err(Error::internal(format!(
            "Merge5 [10] end node key changed: {:?}",
            end.properties
        )));
    };
    if body.program.entity_slot_count != 3
        || !body.program.input_keys.is_empty()
        || !matches!(body.program.property_names.as_slice(), [name] if name == "num")
        || !body.program.label_names.is_empty()
        || !matches!(body.program.relationship_type_names.as_slice(), [name] if name == "R")
        || !body.program.outputs.is_empty()
        || start.output_entity != 0
        || end.output_entity != 1
        || !start.labels.is_empty()
        || !end.labels.is_empty()
        || start_property.property_name != 0
        || end_property.property_name != 0
        || !matches!(
            &start_property.value,
            ResidentRowCreateValueInput::Constant(ResidentCreateNodeValueInput::Scalar(
                ScalarValue::Integer(1)
            ))
        )
        || !matches!(
            &end_property.value,
            ResidentRowCreateValueInput::Constant(ResidentCreateNodeValueInput::Scalar(
                ScalarValue::Integer(2)
            ))
        )
        || relationship.output_entity != 2
        || relationship.source_entity != 0
        || relationship.target_entity != 1
        || relationship.relationship_type != 0
        || !relationship.properties.is_empty()
        || name != "p"
        || *start_entity != 0
        || *relationship_entity != 2
        || *end_entity != 1
    {
        return Err(Error::internal(format!(
            "Merge5 [10] compiler changed its exact lexical path ABI: {body:#?}"
        )));
    }
    Ok(())
}

fn assert_scenario_14_compiler_contract(request: &ResidentRowMutationRequest) -> Result<()> {
    let Some(body) = request.bound_relationship_merge_body() else {
        return Err(Error::internal(
            "Merge5 [14] did not use the bound-relationship body",
        ));
    };
    let [
        ResidentRowCreateCommand::CreateNode(start),
        ResidentRowCreateCommand::CreateNode(end),
        ResidentRowCreateCommand::CreateRelationship(relationship),
    ] = body.program.commands.as_slice()
    else {
        return Err(Error::internal(format!(
            "Merge5 [14] compiler changed its structural command stream: {:?}",
            body.program.commands
        )));
    };
    let [
        ResidentRowBoundRelationshipCommand::CreateNode,
        ResidentRowBoundRelationshipCommand::CreateNode,
        ResidentRowBoundRelationshipCommand::MergeRelationship {
            direction: ResidentRowBoundRelationshipDirection::Directed,
        },
    ] = body.commands.as_slice()
    else {
        return Err(Error::internal(format!(
            "Merge5 [14] compiler changed its semantic command stream: {:?}",
            body.commands
        )));
    };
    let [
        ResidentRowBoundRelationshipStage::Command { command: first },
        ResidentRowBoundRelationshipStage::Command { command: second },
        ResidentRowBoundRelationshipStage::Project {
            keep_scope: first_keep_scope,
            retained_entities: first_retained,
            bindings: first_bindings,
            obligation: first_project_obligation,
        },
        ResidentRowBoundRelationshipStage::Unwind {
            expression: unwind_expression,
            output: unwind_output,
            obligation: unwind_obligation,
        },
        ResidentRowBoundRelationshipStage::Project {
            keep_scope: split_keep_scope,
            retained_entities: split_retained,
            bindings: split_bindings,
            obligation: split_obligation,
        },
        ResidentRowBoundRelationshipStage::Command { command: merge },
    ] = body.schedule.stages.as_slice()
    else {
        return Err(Error::internal(format!(
            "Merge5 [14] compiler changed its lexical schedule: {:?}",
            body.schedule.stages
        )));
    };
    let [
        ResidentQuantifierProjection {
            output: split_output,
            expression: split_expression,
        },
    ] = split_bindings.as_slice()
    else {
        return Err(Error::internal(format!(
            "Merge5 [14] split projection changed shape: {split_bindings:?}"
        )));
    };
    let [property_binding] = body.schedule.property_bindings.as_slice() else {
        return Err(Error::internal(format!(
            "Merge5 [14] dynamic property binding changed shape: {:?}",
            body.schedule.property_bindings
        )));
    };
    let [property] = relationship.properties.as_slice() else {
        return Err(Error::internal(format!(
            "Merge5 [14] structural relationship placeholder changed: {:?}",
            relationship.properties
        )));
    };
    let [ResidentRowBoundRelationshipOutput::CountRows { name: count_name }] =
        body.outputs.as_slice()
    else {
        return Err(Error::internal(format!(
            "Merge5 [14] compiler changed its COUNT output: {:?}",
            body.outputs
        )));
    };
    let expected_unwind =
        ResidentQuantifierExpression::Literal(ResidentQuantifierValue::List(vec![
            ResidentQuantifierValue::String("a,b".to_owned()),
            ResidentQuantifierValue::String("a,b".to_owned()),
        ]));
    let expected_split = ResidentQuantifierExpression::Function {
        function: ResidentQuantifierFunction::Split,
        arguments: vec![
            ResidentQuantifierExpression::Slot(ResidentQuantifierSlot(0)),
            ResidentQuantifierExpression::Literal(ResidentQuantifierValue::String(",".to_owned())),
        ],
    };
    let scheduled_obligations = [
        (*first_project_obligation, 2_u16),
        (*unwind_obligation, 3_u16),
        (*split_obligation, 4_u16),
    ];
    if body.program.entity_slot_count != 3
        || !body.program.input_keys.is_empty()
        || !matches!(body.program.property_names.as_slice(), [name] if name == "foobar")
        || !matches!(body.program.label_names.as_slice(), [first, second] if first == "Foo" && second == "Bar")
        || !matches!(body.program.relationship_type_names.as_slice(), [name] if name == "FB")
        || !body.program.outputs.is_empty()
        || start.output_entity != 0
        || start.labels.as_slice() != [0]
        || !start.properties.is_empty()
        || end.output_entity != 1
        || end.labels.as_slice() != [1]
        || !end.properties.is_empty()
        || relationship.output_entity != 2
        || relationship.source_entity != 0
        || relationship.target_entity != 1
        || relationship.relationship_type != 0
        || property.property_name != 0
        || !matches!(
            &property.value,
            ResidentRowCreateValueInput::Constant(ResidentCreateNodeValueInput::Scalar(
                ScalarValue::Null
            ))
        )
        || body.schedule.value_slot_count != 2
        || *first != 0
        || *second != 1
        || *first_keep_scope
        || first_retained.as_slice() != [0, 1]
        || !first_bindings.is_empty()
        || *unwind_expression != expected_unwind
        || *unwind_output != ResidentQuantifierSlot(0)
        || *split_keep_scope
        || split_retained.as_slice() != [0, 1]
        || *split_output != ResidentQuantifierSlot(1)
        || *split_expression != expected_split
        || *merge != 2
        || property_binding.command != 2
        || property_binding.property_name != 0
        || property_binding.source
            != ResidentRowBoundPropertyBindingSource::ValueSlot(ResidentQuantifierSlot(1))
        || count_name != "count(*)"
        || scheduled_obligations.iter().any(|(obligation, stage)| {
            obligation.id == 0
                || obligation.kind != ResidentObligationKind::Expression
                || obligation.scope != ResidentObligationScope::Expression(*stage)
        })
    {
        return Err(Error::internal(format!(
            "Merge5 [14] compiler changed its exact scheduled value-slot ABI: {body:#?}"
        )));
    }
    Ok(())
}

fn assert_scenario_18_compiler_contract(request: &ResidentRowMutationRequest) -> Result<()> {
    let Some(body) = request.bound_relationship_merge_body() else {
        return Err(Error::internal(
            "Merge5 [18] did not use the bound-relationship body",
        ));
    };
    let [
        ResidentRowCreateCommand::MatchNode(first_match),
        ResidentRowCreateCommand::MatchNode(second_match),
        ResidentRowCreateCommand::CreateRelationship(first_relationship),
        ResidentRowCreateCommand::CreateNode(first_shadow),
        ResidentRowCreateCommand::CreateNode(second_shadow),
        ResidentRowCreateCommand::CreateRelationship(second_relationship),
    ] = body.program.commands.as_slice()
    else {
        return Err(Error::internal(format!(
            "Merge5 [18] compiler changed its structural command stream: {:?}",
            body.program.commands
        )));
    };
    let [
        ResidentRowBoundRelationshipCommand::MatchNode {
            properties: first_match_properties,
        },
        ResidentRowBoundRelationshipCommand::MatchNode {
            properties: second_match_properties,
        },
        ResidentRowBoundRelationshipCommand::MergeRelationship {
            direction: ResidentRowBoundRelationshipDirection::Directed,
        },
        ResidentRowBoundRelationshipCommand::MergeNode,
        ResidentRowBoundRelationshipCommand::MergeNode,
        ResidentRowBoundRelationshipCommand::MergeRelationship {
            direction: ResidentRowBoundRelationshipDirection::Directed,
        },
    ] = body.commands.as_slice()
    else {
        return Err(Error::internal(format!(
            "Merge5 [18] compiler changed its semantic command stream: {:?}",
            body.commands
        )));
    };
    let [
        ResidentRowBoundRelationshipOutput::Property {
            name: first_name,
            entity: first_output_entity,
            property_name: first_output_property,
        },
        ResidentRowBoundRelationshipOutput::Property {
            name: second_name,
            entity: second_output_entity,
            property_name: second_output_property,
        },
    ] = body.outputs.as_slice()
    else {
        return Err(Error::internal(format!(
            "Merge5 [18] compiler changed its output stream: {:?}",
            body.outputs
        )));
    };
    if body.program.entity_slot_count != 6
        || !body.program.input_keys.is_empty()
        || !body.program.label_names.is_empty()
        || !matches!(body.program.property_names.as_slice(), [name] if name == "id")
        || !matches!(body.program.relationship_type_names.as_slice(), [name] if name == "T")
        || !body.program.outputs.is_empty()
        || first_match.output_entity != 0
        || second_match.output_entity != 1
        || !first_match.labels.is_empty()
        || !second_match.labels.is_empty()
        || !first_match_properties.is_empty()
        || !second_match_properties.is_empty()
        || first_relationship.output_entity != 2
        || first_relationship.source_entity != 0
        || first_relationship.target_entity != 1
        || first_relationship.relationship_type != 0
        || !first_relationship.properties.is_empty()
        || first_shadow.output_entity != 3
        || second_shadow.output_entity != 4
        || !first_shadow.labels.is_empty()
        || !second_shadow.labels.is_empty()
        || !first_shadow.properties.is_empty()
        || !second_shadow.properties.is_empty()
        || second_relationship.output_entity != 5
        || second_relationship.source_entity != 3
        || second_relationship.target_entity != 4
        || second_relationship.relationship_type != 0
        || !second_relationship.properties.is_empty()
        || first_name != "x"
        || second_name != "y"
        || *first_output_entity != 0
        || *second_output_entity != 1
        || *first_output_property != 0
        || *second_output_property != 0
    {
        return Err(Error::internal(format!(
            "Merge5 [18] compiler changed its exact hidden-slot ABI: {body:#?}"
        )));
    }
    Ok(())
}

fn assert_scenario_19_compiler_contract(request: &ResidentRowMutationRequest) -> Result<()> {
    let Some(body) = request.bound_relationship_merge_body() else {
        return Err(Error::internal(
            "Merge5 [19] did not use the bound-relationship body",
        ));
    };
    let [
        ResidentRowCreateCommand::MatchNode(start),
        ResidentRowCreateCommand::CreateNode(first_shadow),
        ResidentRowCreateCommand::CreateRelationship(first_relationship),
        ResidentRowCreateCommand::CreateNode(second_shadow),
        ResidentRowCreateCommand::CreateRelationship(second_relationship),
    ] = body.program.commands.as_slice()
    else {
        return Err(Error::internal(format!(
            "Merge5 [19] compiler changed its structural command stream: {:?}",
            body.program.commands
        )));
    };
    let [
        ResidentRowBoundRelationshipCommand::MatchNode { properties },
        ResidentRowBoundRelationshipCommand::MergeNode,
        ResidentRowBoundRelationshipCommand::MergeRelationship {
            direction: ResidentRowBoundRelationshipDirection::Directed,
        },
        ResidentRowBoundRelationshipCommand::MergeNode,
        ResidentRowBoundRelationshipCommand::MergeRelationship {
            direction: ResidentRowBoundRelationshipDirection::Directed,
        },
    ] = body.commands.as_slice()
    else {
        return Err(Error::internal(format!(
            "Merge5 [19] compiler changed its semantic command stream: {:?}",
            body.commands
        )));
    };
    let [
        ResidentRowBoundRelationshipOutput::Property {
            name,
            entity,
            property_name,
        },
    ] = body.outputs.as_slice()
    else {
        return Err(Error::internal(format!(
            "Merge5 [19] compiler changed its output stream: {:?}",
            body.outputs
        )));
    };
    if body.program.entity_slot_count != 5
        || !body.program.input_keys.is_empty()
        || !body.program.label_names.is_empty()
        || !matches!(body.program.property_names.as_slice(), [name] if name == "id")
        || !matches!(body.program.relationship_type_names.as_slice(), [name] if name == "T")
        || !body.program.outputs.is_empty()
        || start.output_entity != 0
        || !start.labels.is_empty()
        || !properties.is_empty()
        || first_shadow.output_entity != 1
        || !first_shadow.labels.is_empty()
        || !first_shadow.properties.is_empty()
        || first_relationship.output_entity != 2
        || first_relationship.source_entity != 0
        || first_relationship.target_entity != 1
        || first_relationship.relationship_type != 0
        || !first_relationship.properties.is_empty()
        || second_shadow.output_entity != 3
        || !second_shadow.labels.is_empty()
        || !second_shadow.properties.is_empty()
        || second_relationship.output_entity != 4
        || second_relationship.source_entity != 0
        || second_relationship.target_entity != 3
        || second_relationship.relationship_type != 0
        || !second_relationship.properties.is_empty()
        || name != "x"
        || *entity != 0
        || *property_name != 0
    {
        return Err(Error::internal(format!(
            "Merge5 [19] compiler changed its exact hidden-slot ABI: {body:#?}"
        )));
    }
    Ok(())
}

fn assert_delete_stage_obligations(
    stage: u16,
    targets: &[irongraph::gpu::ResidentRowBoundDeleteTarget],
    expected_entities: &[u16],
    effect: irongraph::gpu::ResidentExecutionObligation,
) -> bool {
    targets.len() == expected_entities.len()
        && targets
            .iter()
            .zip(expected_entities)
            .all(|(target, entity)| {
                target.entity == *entity
                    && target.rhs_obligation.id != 0
                    && target.rhs_obligation.kind == ResidentObligationKind::MutationRhs
                    && target.rhs_obligation.scope
                        == ResidentObligationScope::MutationCommand(stage)
            })
        && effect.id != 0
        && effect.kind == ResidentObligationKind::MutationEffect
        && effect.scope == ResidentObligationScope::MutationCommand(stage)
}

fn assert_scenario_20_compiler_contract(request: &ResidentRowMutationRequest) -> Result<()> {
    let Some(body) = request.bound_relationship_merge_body() else {
        return Err(Error::internal(
            "Merge5 [20] did not use the bound-relationship body",
        ));
    };
    let [
        ResidentRowBoundRelationshipStage::Command { command: 0 },
        ResidentRowBoundRelationshipStage::Command { command: 1 },
        ResidentRowBoundRelationshipStage::Command { command: 2 },
        ResidentRowBoundRelationshipStage::Command { command: 3 },
        ResidentRowBoundRelationshipStage::Command { command: 4 },
        ResidentRowBoundRelationshipStage::Delete {
            detach,
            targets,
            effect_obligation,
        },
        ResidentRowBoundRelationshipStage::Command { command: 5 },
        ResidentRowBoundRelationshipStage::Command { command: 6 },
        ResidentRowBoundRelationshipStage::Command { command: 7 },
        ResidentRowBoundRelationshipStage::Command { command: 8 },
    ] = body.schedule.stages.as_slice()
    else {
        return Err(Error::internal(format!(
            "Merge5 [20] compiler changed its ordered DELETE/MERGE schedule: {:?}",
            body.schedule.stages
        )));
    };
    let expected_semantics = [
        "match-node",
        "match-node",
        "match-relationship",
        "match-node",
        "match-relationship",
        "merge-node",
        "merge-relationship",
        "merge-node",
        "merge-relationship",
    ];
    let actual_semantics = body
        .commands
        .iter()
        .map(|command| match command {
            ResidentRowBoundRelationshipCommand::MatchNode { .. } => "match-node",
            ResidentRowBoundRelationshipCommand::MatchNodeIncludingWrites { .. } => {
                "match-node-including-writes"
            }
            ResidentRowBoundRelationshipCommand::MatchRelationship { .. } => "match-relationship",
            ResidentRowBoundRelationshipCommand::MergeNode => "merge-node",
            ResidentRowBoundRelationshipCommand::MergeRelationship { .. } => "merge-relationship",
            ResidentRowBoundRelationshipCommand::CreateNode => "create-node",
            ResidentRowBoundRelationshipCommand::CreateRelationship => "create-relationship",
        })
        .collect::<Vec<_>>();
    if *detach
        || !assert_delete_stage_obligations(5, targets, &[2, 4, 1, 3], *effect_obligation)
        || body.program.entity_slot_count != 9
        || actual_semantics.as_slice() != expected_semantics
        || body.command_candidate_limits.as_slice() != [1, 2, 1, 2, 1, 1, 1, 2, 1]
        || body.schedule.value_slot_count != 0
        || !body.schedule.property_bindings.is_empty()
        || !body.outputs.is_empty()
        || body.program.label_names.as_slice() != ["A", "B", "C"]
        || body.program.property_names.as_slice() != ["num"]
        || body.program.relationship_type_names.as_slice() != ["REL"]
        || body.capacities.maximum_rows != 8
    {
        return Err(Error::internal(format!(
            "Merge5 [20] compiler changed its exact DELETE ABI: {body:#?}"
        )));
    }
    Ok(())
}

fn assert_scenario_21_compiler_contract(request: &ResidentRowMutationRequest) -> Result<()> {
    let Some(body) = request.bound_relationship_merge_body() else {
        return Err(Error::internal(
            "Merge5 [21] did not use the bound-relationship body",
        ));
    };
    let [
        ResidentRowBoundRelationshipStage::Command { command: 0 },
        ResidentRowBoundRelationshipStage::Command { command: 1 },
        ResidentRowBoundRelationshipStage::Command { command: 2 },
        ResidentRowBoundRelationshipStage::Delete {
            detach,
            targets,
            effect_obligation,
        },
        ResidentRowBoundRelationshipStage::Command { command: 3 },
    ] = body.schedule.stages.as_slice()
    else {
        return Err(Error::internal(format!(
            "Merge5 [21] compiler changed its ordered DELETE/MERGE schedule: {:?}",
            body.schedule.stages
        )));
    };
    let [
        ResidentRowCreateCommand::MatchNode(a),
        ResidentRowCreateCommand::MatchNode(b),
        ResidentRowCreateCommand::CreateRelationship(t),
        ResidentRowCreateCommand::CreateRelationship(t2),
    ] = body.program.commands.as_slice()
    else {
        return Err(Error::internal(format!(
            "Merge5 [21] compiler changed its structural stream: {:?}",
            body.program.commands
        )));
    };
    let [property] = t2.properties.as_slice() else {
        return Err(Error::internal(format!(
            "Merge5 [21] MERGE key changed shape: {:?}",
            t2.properties
        )));
    };
    if *detach
        || !assert_delete_stage_obligations(3, targets, &[2], *effect_obligation)
        || body.program.entity_slot_count != 4
        || a.output_entity != 0
        || b.output_entity != 1
        || t.output_entity != 2
        || t.source_entity != 0
        || t.target_entity != 1
        || t2.output_entity != 3
        || t2.source_entity != 0
        || t2.target_entity != 1
        || property.property_name != 0
        || !matches!(&property.value,
            ResidentRowCreateValueInput::Constant(ResidentCreateNodeValueInput::Scalar(
                ScalarValue::String(value)
            )) if value.as_ref() == "rel3")
        || body.command_candidate_limits.as_slice() != [2, 2, 2, 1]
        || body.program.label_names.len() != 0
        || body.program.property_names.as_slice() != ["name"]
        || body.program.relationship_type_names.as_slice() != ["T"]
        || !matches!(body.outputs.as_slice(), [ResidentRowBoundRelationshipOutput::Property {
            name,
            entity: 3,
            property_name: 0,
        }] if name == "t2.name")
        || body.capacities.maximum_rows != 8
    {
        return Err(Error::internal(format!(
            "Merge5 [21] compiler changed its exact DELETE ABI: {body:#?}"
        )));
    }
    Ok(())
}

fn assert_match8_optional_compiler_contract(request: &ResidentRowMutationRequest) -> Result<()> {
    let Some(body) = request.bound_relationship_merge_body() else {
        return Err(Error::internal(
            "Match8 [2] did not use the bound-relationship body",
        ));
    };
    if body.program.entity_slot_count != 2
        || !matches!(
            (body.program.commands.as_slice(), body.commands.as_slice()),
            (
                [
                    ResidentRowCreateCommand::MatchNode(matched),
                    ResidentRowCreateCommand::CreateNode(merged),
                ],
                [
                    ResidentRowBoundRelationshipCommand::MatchNode { properties },
                    ResidentRowBoundRelationshipCommand::MergeNode,
                ],
            ) if matched.output_entity == 0
                && matched.labels.is_empty()
                && properties.is_empty()
                && merged.output_entity == 1
                && merged.labels.is_empty()
                && merged.properties.is_empty()
        )
        || !matches!(
            body.schedule.stages.as_slice(),
            [
                ResidentRowBoundRelationshipStage::Command { command: 0 },
                ResidentRowBoundRelationshipStage::Command { command: 1 },
            ]
        )
        || !matches!(
            body.outputs.as_slice(),
            [ResidentRowBoundRelationshipOutput::CountOptionalUndirectedRelationships {
                name,
                source_entity: 0,
                target_entity: 1,
                traversal_obligation,
                aggregate_obligation,
            }] if name == "count(*)"
                && traversal_obligation.kind == ResidentObligationKind::PatternTraversal
                && traversal_obligation.scope == ResidentObligationScope::PatternLeaf(0)
                && aggregate_obligation.kind == ResidentObligationKind::Aggregate
                && aggregate_obligation.scope == ResidentObligationScope::PatternFinal
        )
    {
        return Err(Error::internal(format!(
            "Match8 [2] compiler changed its post-MERGE OPTIONAL ABI: {body:#?}"
        )));
    }
    Ok(())
}

fn assert_one_complete_command(
    case: &ScenarioCase,
    observations: &RouteObservations,
) -> Result<()> {
    let pins = observations.pins.load(Ordering::SeqCst);
    let complete = observations.complete_commands.load(Ordering::SeqCst);
    let forbidden = observations.forbidden_calls.load(Ordering::SeqCst);
    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if pins != 1 || complete != 1 || forbidden != 0 || requests.len() != 1 {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "Merge5 [{}] was not exactly one pinned GPU command: pins={pins}, complete={complete}, forbidden={forbidden}, requests={}",
                case.id,
                requests.len()
            ),
        ));
    }
    if !matches!(
        requests[0].body,
        ResidentRowMutationBody::BoundRelationshipMerge(_)
    ) {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            format!(
                "Merge5 [{}] did not compile to the strict bound-relationship ABI",
                case.id
            ),
        ));
    }
    match case.id {
        10 => assert_scenario_10_compiler_contract(&requests[0])?,
        14 => assert_scenario_14_compiler_contract(&requests[0])?,
        18 => assert_scenario_18_compiler_contract(&requests[0])?,
        19 => assert_scenario_19_compiler_contract(&requests[0])?,
        20 => assert_scenario_20_compiler_contract(&requests[0])?,
        21 => assert_scenario_21_compiler_contract(&requests[0])?,
        249 => assert_match8_optional_compiler_contract(&requests[0])?,
        _ => {}
    }
    requests[0].validate()
}

#[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
#[test]
#[ignore = "requires a real Metal device; validates all 21 pinned Merge5 shapes and distinct scheduled values through one native command"]
fn real_metal_executes_all_21_merge5_scenarios_and_distinct_schedule_without_fallback() -> Result<()>
{
    for case in CASES
        .iter()
        .chain([&SCENARIO_14_CASE, &SCENARIO_20_CASE, &SCENARIO_21_CASE])
    {
        let fixture = Fixture::build(case)?;
        let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
        metal.admit_project(fixture.resident_image()?)?;
        let backend = StrictBoundRelationshipMergeBackend::real_metal(metal)?;
        let observations = backend.observations();
        let output = QueryEngine.execute(
            case.query,
            &mut context(&fixture.graph, Some(&backend), true),
        )?;
        assert_one_complete_command(case, &observations)?;
        match case.id {
            14 => assert_scheduled_list_property_output(
                case,
                &fixture,
                &output,
                "Foo",
                "Bar",
                "FB",
                "foobar",
                &[&["a", "b"]],
            )?,
            20 => assert_scenario_20_output(&fixture, &output)?,
            21 => assert_scenario_21_output(&fixture, &output)?,
            _ => assert_case_output(case, &fixture, &output)?,
        }
    }

    let case = &DISTINCT_SCHEDULED_LIST_CASE;
    let fixture = Fixture::build(case)?;
    let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
    metal.admit_project(fixture.resident_image()?)?;
    let backend = StrictBoundRelationshipMergeBackend::real_metal(metal)?;
    let observations = backend.observations();
    let output = QueryEngine.execute(
        case.query,
        &mut context(&fixture.graph, Some(&backend), true),
    )?;
    assert_one_complete_command(case, &observations)?;
    assert_scheduled_list_property_output(
        case,
        &fixture,
        &output,
        "Source",
        "Sink",
        "HAS_ACCESS",
        "permissions",
        &[&["red", "blue"], &["green", "gold"]],
    )?;
    Ok(())
}

#[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
#[test]
#[ignore = "requires a real Metal device; validates Match8 [2] post-MERGE OPTIONAL counting without fallback"]
fn real_metal_match8_2_counts_optional_rows_after_node_merge_without_fallback() -> Result<()> {
    let case = &MATCH8_OPTIONAL_AFTER_MERGE_CASE;
    let fixture = Fixture::build(case)?;
    let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
    metal.admit_project(fixture.resident_image()?)?;
    let backend = StrictBoundRelationshipMergeBackend::real_metal(metal)?;
    let observations = backend.observations();
    let output = QueryEngine.execute(
        case.query,
        &mut context(&fixture.graph, Some(&backend), true),
    )?;
    assert_one_complete_command(case, &observations)?;
    assert_case_output(case, &fixture, &output)
}

#[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
#[test]
#[ignore = "requires a real Metal device; validates scheduled DELETE endpoint filtering without fallback"]
fn real_metal_deleted_endpoint_rows_do_not_reach_later_relationship_merge() -> Result<()> {
    let case = &DELETED_ENDPOINT_CASE;
    let fixture = Fixture::build(case)?;
    let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
    metal.admit_project(fixture.resident_image()?)?;
    let backend = StrictBoundRelationshipMergeBackend::real_metal(metal)?;
    let observations = backend.observations();
    let output = QueryEngine.execute(
        case.query,
        &mut context(&fixture.graph, Some(&backend), true),
    )?;
    assert_one_complete_command(case, &observations)?;
    assert_deleted_endpoint_case_output(&fixture, &output)
}

#[test]
fn exact_merge5_scenario_set_is_pinned() {
    let mut scenarios = CASES.iter().map(|case| case.id).collect::<Vec<_>>();
    scenarios.extend([
        SCENARIO_14_CASE.id,
        SCENARIO_20_CASE.id,
        SCENARIO_21_CASE.id,
    ]);
    scenarios.sort_unstable();
    assert_eq!(scenarios, (1..=21).collect::<Vec<_>>());
}
