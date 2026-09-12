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

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use std::sync::MutexGuard;

use irongraph::{
    Bookmark, EdgeId, Error, ErrorCode, Layer, NodeId, ProjectId, Result, ScalarValue,
    cypher::{
        BindCapabilities, ColumnType, ExecutionContext, ExecutionOutput, ExecutionStreamItem,
        QueryEngine, ResultValue, StatementStats,
    },
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend, ResidentGroup,
        ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest, ResidentNodePipelineRequest,
        ResidentNodePipelineResult, ResidentProjectImage, ResidentSortRequest, ResidentSortResult,
        ResidentVariablePathRequest, ResidentVariablePathResult, ResidentVectorQuery,
        ResidentVectorResult, ScratchReservation,
    },
    graph::{EdgeInput, GraphStore, IndexCatalog, LayerMask, NodeInput, TemporalStore},
    types::{LabelId, PropertyId},
};
use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::gpu::MetalBackend;

const FEATURE: &str = "features/clauses/match/Match4.feature";
const RELATIONSHIP_PREDICATE_SCENARIO_ID: u8 = 5;
const RELATIONSHIP_PREDICATE_QUERY: &str =
    "MATCH (a:Artist)-[:WORKED_WITH* {year: 1988}]->(b:Artist) RETURN *";
const INTERMEDIATE_BOUNDARY_QUERY: &str =
    "MATCH (a {name: 'A'})-[:CONTAINS*0..1]->(b)-[:FRIEND*0..1]->(c) RETURN a, b, c";
const BOUND_RELATIONSHIP_COUNT_QUERY: &str = "MATCH ()-[r:EDGE]-() \
    MATCH p = (n)-[*0..1]-()-[r]-()-[*0..1]-(m) RETURN count(p) AS c";
const BOUND_RELATIONSHIP_LIST_QUERY: &str = "MATCH ()-[r1]->()-[r2]->() \
    WITH [r1, r2] AS rs LIMIT 1 \
    MATCH (first)-[rs*]->(second) RETURN first, second";
const LAST_RELATIONSHIP_FEATURE: &str = "features/clauses/match/Match9.feature";
const LAST_RELATIONSHIP_SCENARIO_ID: u8 = 1;
const LAST_RELATIONSHIP_QUERY: &str = "MATCH ()-[r*0..1]-() RETURN last(r) AS l";
const MATCH9_COUNT_SCENARIO_ID: u8 = 5;
const MATCH9_COUNT_QUERY: &str = "MATCH (a:Blue)-[r*]->(b:Green) RETURN count(r)";
const MATCH9_SAME_REMATCH_QUERY: &str = "MATCH (a)-[r1]->()-[r2]->(b) WITH [r1, r2] AS rs, a AS first, b AS second LIMIT 1 MATCH (first)-[rs*]->(second) RETURN first, second";
const MATCH9_REVERSE_REMATCH_QUERY: &str = "MATCH (a)-[r1]->()-[r2]->(b) WITH [r1, r2] AS rs, a AS second, b AS first LIMIT 1 MATCH (first)-[rs*]->(second) RETURN first, second";
const MATCH9_OPTIONAL_NULL_QUERY: &str =
    "MATCH (a:A), (b:B) OPTIONAL MATCH (a)-[r*]-(b) WHERE r IS NULL AND a <> b RETURN b";
const COMPARISON1_PATH_EQUALITY_QUERY: &str =
    "MATCH p1 = (:A)-->() MATCH p2 = (:A)<--() RETURN p1 = p2";
const CORRELATED_PATH_DISJUNCTION_QUERY: &str = "MATCH (a), (b) WHERE a.id = 0 AND \
    (a)-[:T]->(b:TheLabel) OR (a)-[:T*]->(b:MissingLabel) RETURN DISTINCT b";
const CORRELATED_PATH_DISJUNCTION_WITH_QUERY: &str = "MATCH (a), (b) WITH a, b WHERE \
    a.id = 0 AND (a)-[:T]->(b:TheLabel) OR (a)-[:T*]->(b:MissingLabel) RETURN DISTINCT b";
const CYCLE_CHORD_QUERY: &str =
    "MATCH (a)--(b)--(c)--(d)--(a), (b)--(d) WHERE a.id = 1 AND c.id = 2 RETURN d";
const INDEPENDENT_PATH_SUM_QUERY: &str =
    "MATCH ()-->() WITH 1 AS x MATCH ()-[r1]->()<--() RETURN sum(r1.times)";
const MATCH9_BASELINE_REPORT: &str =
    "/tmp/irongraph-tck-full-20260721-after-merge6-set-typeconversion-return5-r1.json";
const MATCH9_BASELINE_REPORT_SHA256: &str =
    "f47070718ad35a3fade3335ec4d3e81d4ac4a77bc3e143942f92041098ea4e77";
const RETURN7_FEATURE: &str = "features/clauses/return/Return7.feature";
const RETURN7_SCENARIO_ID: u8 = 1;
const RETURN7_QUERY: &str = "MATCH p = (a:Start)-->(b) RETURN *";
const MEMORY_LIMIT_BYTES: usize = 512 * 1024 * 1024;
const RESERVED_BYTES: usize = 64 * 1024 * 1024;
const COMPLETE_PROJECTION_QUERY: &str = "MATCH p = (a:A)-[r:T*1..2]->(b) \
    RETURN p, nodes(p) AS ns, relationships(p) AS rs, length(p) AS hops, \
    a, b, a.name AS start_name, b.name AS end_name, r";
const RETURN2_ENTITY_LIST_QUERY: &str = "MATCH (n)-[r]->(m) RETURN [n, r, m] AS r";
const RETURN2_ENTITY_MAP_QUERY: &str =
    "MATCH (n)-[r]->(m) RETURN {node1: n, rel: r, node2: m} AS m";
const WITH3_RELATIONSHIP_REMATCH_QUERY: &str = "MATCH (a)-[r]->(b:X) WITH a, r, b \
    MATCH (a)-[r]->(b) RETURN r AS rel ORDER BY rel.id";
const WITH7_GROUPED_INTERMEDIATE_QUERY: &str = "MATCH (david {name: 'David'})--(otherPerson)-->() \
    WITH otherPerson, count(*) AS foaf WHERE foaf > 1 \
    WITH otherPerson WHERE otherPerson.name <> 'NotOther' RETURN count(*)";

#[derive(Clone, Copy, Debug)]
struct Scenario {
    id: u8,
    query: &'static str,
    expected_relationship_types: &'static [&'static str],
}

const SCENARIOS: &[Scenario] = &[
    Scenario {
        id: 1,
        query: "MATCH (a)-[r*1..1]->(b) RETURN r",
        expected_relationship_types: &["T"],
    },
    Scenario {
        id: 6,
        query: "MATCH (a:A) MATCH (a)-[r*2]->() RETURN r",
        expected_relationship_types: &["X", "Y"],
    },
];

struct Fixture {
    graph: GraphStore,
    project: ProjectId,
    bookmark: Bookmark,
}

fn fixture(scenario: Scenario) -> Result<Fixture> {
    let mut graph = GraphStore::default();
    let project = ProjectId(uuid::Uuid::from_u128(
        0x4d41_5443_4834_0000 + scenario.id as u128,
    ));
    let bookmark = Bookmark {
        term: 17,
        index: 4_000 + u64::from(scenario.id),
    };

    match scenario.id {
        1 => {
            let relationship_type = graph.catalog_mut().intern_relationship_type("T")?;
            for id in [NodeId(1), NodeId(2)] {
                graph.insert_node(NodeInput {
                    id,
                    layer: Layer::Observed,
                    revision: id.0,
                    labels: Vec::new(),
                    properties: Vec::new(),
                })?;
            }
            graph.insert_edge(EdgeInput {
                id: EdgeId(10),
                source: NodeId(1),
                target: NodeId(2),
                relationship_type,
                layer: Layer::Observed,
                revision: 10,
                properties: Vec::new(),
            })?;
        }
        6 => {
            let label = graph.catalog_mut().intern_label("A")?;
            let x = graph.catalog_mut().intern_relationship_type("X")?;
            let y = graph.catalog_mut().intern_relationship_type("Y")?;
            for (id, labels) in [
                (NodeId(1), vec![label]),
                (NodeId(2), Vec::new()),
                (NodeId(3), Vec::new()),
            ] {
                graph.insert_node(NodeInput {
                    id,
                    layer: Layer::Observed,
                    revision: id.0,
                    labels,
                    properties: Vec::new(),
                })?;
            }
            for (id, source, target, relationship_type) in [
                (EdgeId(10), NodeId(1), NodeId(2), x),
                (EdgeId(11), NodeId(2), NodeId(3), y),
            ] {
                graph.insert_edge(EdgeInput {
                    id,
                    source,
                    target,
                    relationship_type,
                    layer: Layer::Observed,
                    revision: id.0,
                    properties: Vec::new(),
                })?;
            }
        }
        _ => return Err(Error::internal("unknown Match4 scenario fixture")),
    }

    Ok(Fixture {
        graph,
        project,
        bookmark,
    })
}

fn complete_projection_fixture() -> Result<Fixture> {
    let mut graph = GraphStore::default();
    let label = graph.catalog_mut().intern_label("A")?;
    let relationship_type = graph.catalog_mut().intern_relationship_type("T")?;
    let name = graph.catalog_mut().intern_property("name")?;
    for (id, labels, value) in [
        (NodeId(1), vec![label], "a"),
        (NodeId(2), Vec::new(), "b"),
        (NodeId(3), Vec::new(), "c"),
    ] {
        graph.insert_node(NodeInput {
            id,
            layer: Layer::Observed,
            revision: id.0,
            labels,
            properties: vec![(name, ScalarValue::String(value.into()))],
        })?;
    }
    for (id, source, target) in [
        (EdgeId(10), NodeId(1), NodeId(2)),
        (EdgeId(11), NodeId(2), NodeId(3)),
    ] {
        graph.insert_edge(EdgeInput {
            id,
            source,
            target,
            relationship_type,
            layer: Layer::Observed,
            revision: id.0,
            properties: Vec::new(),
        })?;
    }
    Ok(Fixture {
        graph,
        project: ProjectId(uuid::Uuid::from_u128(0x5041_5448_5f4f_5554_5055_5453)),
        bookmark: Bookmark {
            term: 17,
            index: 4_100,
        },
    })
}

fn return2_entity_container_fixture() -> Result<Fixture> {
    let mut graph = GraphStore::default();
    let a = graph.catalog_mut().intern_label("A")?;
    let b = graph.catalog_mut().intern_label("B")?;
    let relationship_type = graph.catalog_mut().intern_relationship_type("T")?;
    graph.insert_node(NodeInput {
        id: NodeId(1),
        layer: Layer::Observed,
        revision: 1,
        labels: vec![a],
        properties: Vec::new(),
    })?;
    graph.insert_node(NodeInput {
        id: NodeId(2),
        layer: Layer::Observed,
        revision: 2,
        labels: vec![b],
        properties: Vec::new(),
    })?;
    graph.insert_edge(EdgeInput {
        id: EdgeId(10),
        source: NodeId(1),
        target: NodeId(2),
        relationship_type,
        layer: Layer::Observed,
        revision: 10,
        properties: Vec::new(),
    })?;
    Ok(Fixture {
        graph,
        project: ProjectId(uuid::Uuid::from_u128(0x5245_5455_524e_325f_454e_5449)),
        bookmark: Bookmark {
            term: 17,
            index: 4_200,
        },
    })
}

fn with3_relationship_rematch_fixture() -> Result<Fixture> {
    let mut graph = GraphStore::default();
    let x = graph.catalog_mut().intern_label("X")?;
    let t1 = graph.catalog_mut().intern_relationship_type("T1")?;
    let t2 = graph.catalog_mut().intern_relationship_type("T2")?;
    let id_property = graph.catalog_mut().intern_property("id")?;
    for (id, labels) in [
        (NodeId(1), Vec::new()),
        (NodeId(2), vec![x]),
        (NodeId(3), Vec::new()),
        (NodeId(4), vec![x]),
        (NodeId(5), Vec::new()),
        (NodeId(6), Vec::new()),
    ] {
        graph.insert_node(NodeInput {
            id,
            layer: Layer::Observed,
            revision: id.0,
            labels,
            properties: Vec::new(),
        })?;
    }
    for (id, source, target, relationship_type, value) in [
        (EdgeId(10), NodeId(1), NodeId(2), t1, 0),
        (EdgeId(11), NodeId(3), NodeId(4), t2, 1),
        (EdgeId(12), NodeId(5), NodeId(6), t2, 2),
    ] {
        graph.insert_edge(EdgeInput {
            id,
            source,
            target,
            relationship_type,
            layer: Layer::Observed,
            revision: id.0,
            properties: vec![(id_property, ScalarValue::Integer(value))],
        })?;
    }
    Ok(Fixture {
        graph,
        project: ProjectId(uuid::Uuid::from_u128(0x5749_5448_335f_5245_4d41_5443)),
        bookmark: Bookmark {
            term: 17,
            index: 4_300,
        },
    })
}

fn with7_grouped_intermediate_fixture() -> Result<Fixture> {
    let mut graph = GraphStore::default();
    let relationship_type = graph.catalog_mut().intern_relationship_type("REL")?;
    let name = graph.catalog_mut().intern_property("name")?;
    for (id, value) in [
        (NodeId(1), Some("David")),
        (NodeId(2), Some("Other")),
        (NodeId(3), Some("NotOther")),
        (NodeId(4), Some("NotOther2")),
        (NodeId(5), None),
        (NodeId(6), None),
        (NodeId(7), None),
        (NodeId(8), None),
        (NodeId(9), None),
    ] {
        graph.insert_node(NodeInput {
            id,
            layer: Layer::Observed,
            revision: id.0,
            labels: Vec::new(),
            properties: value
                .map(|value| (name, ScalarValue::String(Arc::from(value))))
                .into_iter()
                .collect(),
        })?;
    }
    for (id, source, target) in [
        (EdgeId(10), NodeId(1), NodeId(2)),
        (EdgeId(11), NodeId(1), NodeId(3)),
        (EdgeId(12), NodeId(1), NodeId(4)),
        (EdgeId(13), NodeId(2), NodeId(5)),
        (EdgeId(14), NodeId(2), NodeId(6)),
        (EdgeId(15), NodeId(3), NodeId(7)),
        (EdgeId(16), NodeId(3), NodeId(8)),
        (EdgeId(17), NodeId(4), NodeId(9)),
    ] {
        graph.insert_edge(EdgeInput {
            id,
            source,
            target,
            relationship_type,
            layer: Layer::Observed,
            revision: id.0,
            properties: Vec::new(),
        })?;
    }
    Ok(Fixture {
        graph,
        project: ProjectId(uuid::Uuid::from_u128(0x5749_5448_375f_4752_4f55_5045)),
        bookmark: Bookmark {
            term: 17,
            index: 4_700,
        },
    })
}

fn relationship_predicate_fixture() -> Result<Fixture> {
    let mut graph = GraphStore::default();
    let artist = graph.catalog_mut().intern_label("Artist")?;
    let a = graph.catalog_mut().intern_label("A")?;
    let b = graph.catalog_mut().intern_label("B")?;
    let c = graph.catalog_mut().intern_label("C")?;
    let worked_with = graph
        .catalog_mut()
        .intern_relationship_type("WORKED_WITH")?;
    let year = graph.catalog_mut().intern_property("year")?;
    for (id, role) in [(NodeId(1), a), (NodeId(2), b), (NodeId(3), c)] {
        graph.insert_node(NodeInput {
            id,
            layer: Layer::Observed,
            revision: id.0,
            labels: vec![artist, role],
            properties: Vec::new(),
        })?;
    }
    for (id, source, target, value) in [
        (EdgeId(10), NodeId(1), NodeId(2), 1987),
        (EdgeId(11), NodeId(2), NodeId(3), 1988),
    ] {
        graph.insert_edge(EdgeInput {
            id,
            source,
            target,
            relationship_type: worked_with,
            layer: Layer::Observed,
            revision: id.0,
            properties: vec![(year, ScalarValue::Integer(value))],
        })?;
    }
    Ok(Fixture {
        graph,
        project: ProjectId(uuid::Uuid::from_u128(0x4d41_5443_4834_0005)),
        bookmark: Bookmark {
            term: 17,
            index: 4_005,
        },
    })
}

fn intermediate_boundary_fixture() -> Result<Fixture> {
    let mut graph = GraphStore::default();
    let contains = graph.catalog_mut().intern_relationship_type("CONTAINS")?;
    let friend = graph.catalog_mut().intern_relationship_type("FRIEND")?;
    let name = graph.catalog_mut().intern_property("name")?;
    for (id, value) in [
        (NodeId(1), "A"),
        (NodeId(2), "B"),
        (NodeId(3), "C"),
        (NodeId(4), "D"),
        (NodeId(5), "E"),
    ] {
        graph.insert_node(NodeInput {
            id,
            layer: Layer::Observed,
            revision: id.0,
            labels: Vec::new(),
            properties: vec![(name, ScalarValue::String(value.into()))],
        })?;
    }
    for (id, source, target, relationship_type) in [
        (EdgeId(10), NodeId(1), NodeId(2), contains),
        (EdgeId(11), NodeId(2), NodeId(3), friend),
    ] {
        graph.insert_edge(EdgeInput {
            id,
            source,
            target,
            relationship_type,
            layer: Layer::Observed,
            revision: id.0,
            properties: Vec::new(),
        })?;
    }
    Ok(Fixture {
        graph,
        project: ProjectId(uuid::Uuid::from_u128(0x4d41_5443_4834_0003)),
        bookmark: Bookmark {
            term: 17,
            index: 4_003,
        },
    })
}

fn bound_relationship_count_fixture() -> Result<Fixture> {
    let mut graph = GraphStore::default();
    let edge = graph.catalog_mut().intern_relationship_type("EDGE")?;
    for id in [NodeId(1), NodeId(2), NodeId(3), NodeId(4)] {
        graph.insert_node(NodeInput {
            id,
            layer: Layer::Observed,
            revision: id.0,
            labels: Vec::new(),
            properties: Vec::new(),
        })?;
    }
    for (id, source, target) in [
        (EdgeId(10), NodeId(1), NodeId(2)),
        (EdgeId(11), NodeId(2), NodeId(3)),
        (EdgeId(12), NodeId(3), NodeId(4)),
    ] {
        graph.insert_edge(EdgeInput {
            id,
            source,
            target,
            relationship_type: edge,
            layer: Layer::Observed,
            revision: id.0,
            properties: Vec::new(),
        })?;
    }
    Ok(Fixture {
        graph,
        project: ProjectId(uuid::Uuid::from_u128(0x4d41_5443_4834_0007)),
        bookmark: Bookmark {
            term: 17,
            index: 4_007,
        },
    })
}

fn bound_relationship_list_fixture() -> Result<Fixture> {
    let mut graph = GraphStore::default();
    let relationship_type = graph.catalog_mut().intern_relationship_type("Y")?;
    for id in [NodeId(1), NodeId(2), NodeId(3)] {
        graph.insert_node(NodeInput {
            id,
            layer: Layer::Observed,
            revision: id.0,
            labels: Vec::new(),
            properties: Vec::new(),
        })?;
    }
    for (id, source, target) in [
        (EdgeId(10), NodeId(1), NodeId(2)),
        (EdgeId(11), NodeId(2), NodeId(3)),
    ] {
        graph.insert_edge(EdgeInput {
            id,
            source,
            target,
            relationship_type,
            layer: Layer::Observed,
            revision: id.0,
            properties: Vec::new(),
        })?;
    }
    Ok(Fixture {
        graph,
        project: ProjectId(uuid::Uuid::from_u128(0x4d41_5443_4834_0008)),
        bookmark: Bookmark {
            term: 17,
            index: 4_008,
        },
    })
}

fn last_relationship_fixture() -> Result<Fixture> {
    let mut graph = GraphStore::default();
    let relationship_type = graph.catalog_mut().intern_relationship_type("T")?;
    for id in [NodeId(1), NodeId(2), NodeId(3)] {
        graph.insert_node(NodeInput {
            id,
            layer: Layer::Observed,
            revision: id.0,
            labels: Vec::new(),
            properties: Vec::new(),
        })?;
    }
    graph.insert_edge(EdgeInput {
        id: EdgeId(10),
        source: NodeId(1),
        target: NodeId(2),
        relationship_type,
        layer: Layer::Observed,
        revision: 10,
        properties: Vec::new(),
    })?;
    Ok(Fixture {
        graph,
        project: ProjectId(uuid::Uuid::from_u128(0x4d41_5443_4839_0001)),
        bookmark: Bookmark {
            term: 17,
            index: 4_901,
        },
    })
}

fn match9_count_fixture() -> Result<Fixture> {
    let mut graph = GraphStore::default();
    let blue = graph.catalog_mut().intern_label("Blue")?;
    let red = graph.catalog_mut().intern_label("Red")?;
    let green = graph.catalog_mut().intern_label("Green")?;
    let yellow = graph.catalog_mut().intern_label("Yellow")?;
    let relationship_type = graph.catalog_mut().intern_relationship_type("T")?;
    for (id, label) in [
        (NodeId(1), blue),
        (NodeId(2), red),
        (NodeId(3), green),
        (NodeId(4), yellow),
    ] {
        graph.insert_node(NodeInput {
            id,
            layer: Layer::Observed,
            revision: id.0,
            labels: vec![label],
            properties: Vec::new(),
        })?;
    }
    for (id, source, target) in [
        (EdgeId(10), NodeId(1), NodeId(2)),
        (EdgeId(11), NodeId(2), NodeId(3)),
        (EdgeId(12), NodeId(2), NodeId(4)),
    ] {
        graph.insert_edge(EdgeInput {
            id,
            source,
            target,
            relationship_type,
            layer: Layer::Observed,
            revision: id.0,
            properties: Vec::new(),
        })?;
    }
    Ok(Fixture {
        graph,
        project: ProjectId(uuid::Uuid::from_u128(0x4d41_5443_4839_0005)),
        bookmark: Bookmark {
            term: 17,
            index: 9_005,
        },
    })
}

fn match9_optional_null_fixture() -> Result<Fixture> {
    let mut graph = GraphStore::default();
    let a = graph.catalog_mut().intern_label("A")?;
    let b = graph.catalog_mut().intern_label("B")?;
    for (id, label) in [(NodeId(1), a), (NodeId(2), b)] {
        graph.insert_node(NodeInput {
            id,
            layer: Layer::Observed,
            revision: id.0,
            labels: vec![label],
            properties: Vec::new(),
        })?;
    }
    Ok(Fixture {
        graph,
        project: ProjectId(uuid::Uuid::from_u128(0x4d41_5443_4839_0008)),
        bookmark: Bookmark {
            term: 17,
            index: 9_008,
        },
    })
}

fn comparison1_path_equality_fixture() -> Result<Fixture> {
    let mut graph = GraphStore::default();
    let a = graph.catalog_mut().intern_label("A")?;
    let relationship_type = graph.catalog_mut().intern_relationship_type("LOOP")?;
    graph.insert_node(NodeInput {
        id: NodeId(1),
        layer: Layer::Observed,
        revision: 1,
        labels: vec![a],
        properties: Vec::new(),
    })?;
    graph.insert_edge(EdgeInput {
        id: EdgeId(10),
        source: NodeId(1),
        target: NodeId(1),
        relationship_type,
        layer: Layer::Observed,
        revision: 10,
        properties: Vec::new(),
    })?;
    Ok(Fixture {
        graph,
        project: ProjectId(uuid::Uuid::from_u128(0x434f_4d50_4152_4953_4f4e_3114)),
        bookmark: Bookmark {
            term: 17,
            index: 10_114,
        },
    })
}

fn return7_fixture() -> Result<Fixture> {
    let mut graph = GraphStore::default();
    let start = graph.catalog_mut().intern_label("Start")?;
    let relationship_type = graph.catalog_mut().intern_relationship_type("T")?;
    for (id, labels) in [(NodeId(1), vec![start]), (NodeId(2), Vec::new())] {
        graph.insert_node(NodeInput {
            id,
            layer: Layer::Observed,
            revision: id.0,
            labels,
            properties: Vec::new(),
        })?;
    }
    graph.insert_edge(EdgeInput {
        id: EdgeId(10),
        source: NodeId(1),
        target: NodeId(2),
        relationship_type,
        layer: Layer::Observed,
        revision: 10,
        properties: Vec::new(),
    })?;
    Ok(Fixture {
        graph,
        project: ProjectId(uuid::Uuid::from_u128(0x5245_5455_524e_3700_0000_0001)),
        bookmark: Bookmark {
            term: 17,
            index: 7_001,
        },
    })
}

fn correlated_path_disjunction_fixture() -> Result<Fixture> {
    let mut graph = GraphStore::default();
    let terminal = graph.catalog_mut().intern_label("TheLabel")?;
    let relationship_type = graph.catalog_mut().intern_relationship_type("T")?;
    let id = graph.catalog_mut().intern_property("id")?;
    for (node_id, labels, value) in [
        (NodeId(1), Vec::new(), 0),
        (NodeId(2), vec![terminal], 1),
        (NodeId(3), vec![terminal], 2),
    ] {
        graph.insert_node(NodeInput {
            id: node_id,
            layer: Layer::Observed,
            revision: node_id.0,
            labels,
            properties: vec![(id, ScalarValue::Integer(value))],
        })?;
    }
    graph.insert_edge(EdgeInput {
        id: EdgeId(10),
        source: NodeId(1),
        target: NodeId(2),
        relationship_type,
        layer: Layer::Observed,
        revision: 10,
        properties: Vec::new(),
    })?;
    Ok(Fixture {
        graph,
        project: ProjectId(uuid::Uuid::from_u128(0x5748_4552_4534_0002)),
        bookmark: Bookmark {
            term: 17,
            index: 7_102,
        },
    })
}

fn cycle_chord_fixture() -> Result<Fixture> {
    let mut graph = GraphStore::default();
    let relationship_type = graph.catalog_mut().intern_relationship_type("R")?;
    let id = graph.catalog_mut().intern_property("id")?;
    for (node_id, value) in [
        (NodeId(1), Some(1)),
        (NodeId(2), None),
        (NodeId(3), Some(2)),
        (NodeId(4), None),
    ] {
        graph.insert_node(NodeInput {
            id: node_id,
            layer: Layer::Observed,
            revision: node_id.0,
            labels: Vec::new(),
            properties: value
                .map(|value| vec![(id, ScalarValue::Integer(value))])
                .unwrap_or_default(),
        })?;
    }
    for (edge_id, source, target) in [
        (EdgeId(10), NodeId(1), NodeId(2)),
        (EdgeId(11), NodeId(2), NodeId(3)),
        (EdgeId(12), NodeId(3), NodeId(4)),
        (EdgeId(13), NodeId(4), NodeId(1)),
        (EdgeId(14), NodeId(2), NodeId(4)),
    ] {
        graph.insert_edge(EdgeInput {
            id: edge_id,
            source,
            target,
            relationship_type,
            layer: Layer::Observed,
            revision: edge_id.0,
            properties: Vec::new(),
        })?;
    }
    Ok(Fixture {
        graph,
        project: ProjectId(uuid::Uuid::from_u128(0x4d41_5443_4857_3201)),
        bookmark: Bookmark {
            term: 17,
            index: 7_201,
        },
    })
}

fn independent_path_sum_fixture() -> Result<Fixture> {
    let mut graph = GraphStore::default();
    let relationship_type = graph.catalog_mut().intern_relationship_type("ATE")?;
    let times = graph.catalog_mut().intern_property("times")?;
    for node_id in [NodeId(1), NodeId(2), NodeId(3)] {
        graph.insert_node(NodeInput {
            id: node_id,
            layer: Layer::Observed,
            revision: node_id.0,
            labels: Vec::new(),
            properties: Vec::new(),
        })?;
    }
    for (edge_id, source, value) in [(EdgeId(10), NodeId(1), 10), (EdgeId(11), NodeId(3), 4)] {
        graph.insert_edge(EdgeInput {
            id: edge_id,
            source,
            target: NodeId(2),
            relationship_type,
            layer: Layer::Observed,
            revision: edge_id.0,
            properties: vec![(times, ScalarValue::Integer(value))],
        })?;
    }
    Ok(Fixture {
        graph,
        project: ProjectId(uuid::Uuid::from_u128(0x4d41_5443_4838_0003)),
        bookmark: Bookmark {
            term: 17,
            index: 7_803,
        },
    })
}

fn resident_image(fixture: &Fixture) -> Result<ResidentProjectImage> {
    ResidentProjectImage::build(
        fixture.project,
        fixture.bookmark,
        &fixture.graph,
        &TemporalStore::default(),
        &IndexCatalog::default(),
    )
}

fn cpu_backend(fixture: &Fixture) -> Result<CpuBackend> {
    let mut backend = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
    backend.admit_project(resident_image(fixture)?)?;
    Ok(backend)
}

fn context<'a>(
    fixture: &'a Fixture,
    backend: Option<&'a dyn ExecutionBackend>,
    require_native_execution: bool,
) -> ExecutionContext<'a> {
    ExecutionContext {
        project_id: fixture.project,
        graph: &fixture.graph,
        binding_catalog: fixture.graph.catalog(),
        prior_graph_mutations: &[],
        temporal: None,
        prior_temporal_mutations: &[],
        vector_indexes: BTreeMap::new(),
        scalar_indexes: None,
        text_embedding: None,
        parameters: BTreeMap::new(),
        bookmark: fixture.bookmark,
        mutation_revision: fixture.graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id: 10_000,
        next_edge_id: 20_000,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            require_native_execution,
            ..BindCapabilities::default()
        },
        max_result_rows: 4_096,
        max_batch_rows: 7,
        optimizer_statistics: None,
        backend,
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(30)),
        resolved_query_at_time_nanos: None,
    }
}

fn relationship_type_rows(output: &ExecutionOutput) -> Result<Vec<Vec<String>>> {
    if !output.graph_mutations.is_empty() || !output.temporal_mutations.is_empty() {
        return Err(Error::internal("read-only Match4 query produced mutations"));
    }
    if output.result.statistics != StatementStats::default() {
        return Err(Error::internal(
            "read-only Match4 query reported side effects",
        ));
    }

    let mut rows = Vec::new();
    for batch in &output.result.batches {
        let column = batch
            .columns
            .iter()
            .find(|column| column.name == "r")
            .ok_or_else(|| Error::internal("Match4 result omitted `r`"))?;
        for value in &column.values {
            let ResultValue::List(relationships) = value else {
                return Err(Error::internal("Match4 `r` was not a relationship LIST"));
            };
            rows.push(
                relationships
                    .iter()
                    .map(|relationship| match relationship {
                        ResultValue::Relationship(edge) => Ok(edge.relationship_type.clone()),
                        _ => Err(Error::internal(
                            "Match4 relationship list contained a non-relationship value",
                        )),
                    })
                    .collect::<Result<Vec<_>>>()?,
            );
        }
    }
    Ok(rows)
}

fn execute_relationship_types(
    fixture: &Fixture,
    backend: Option<&dyn ExecutionBackend>,
    require_native_execution: bool,
    scenario: Scenario,
) -> Result<Vec<Vec<String>>> {
    let output = QueryEngine.execute(
        scenario.query,
        &mut context(fixture, backend, require_native_execution),
    )?;
    relationship_type_rows(&output)
}

fn last_relationship_rows(output: &ExecutionOutput) -> Result<Vec<Option<String>>> {
    if !output.graph_mutations.is_empty() || !output.temporal_mutations.is_empty() {
        return Err(Error::internal("read-only Match9 query produced mutations"));
    }
    if output.result.statistics != StatementStats::default() {
        return Err(Error::internal(
            "read-only Match9 query reported side effects",
        ));
    }

    let mut rows = Vec::new();
    for batch in &output.result.batches {
        let column = batch
            .columns
            .iter()
            .find(|column| column.name == "l")
            .ok_or_else(|| Error::internal("Match9 result omitted `l`"))?;
        for value in &column.values {
            rows.push(match value {
                ResultValue::Relationship(relationship) => {
                    Some(relationship.relationship_type.clone())
                }
                ResultValue::Scalar(ScalarValue::Null) => None,
                _ => {
                    return Err(Error::internal(
                        "Match9 `last(r)` was neither a relationship nor null",
                    ));
                }
            });
        }
    }
    rows.sort();
    Ok(rows)
}

fn execute_last_relationship(
    fixture: &Fixture,
    backend: Option<&dyn ExecutionBackend>,
    require_native_execution: bool,
) -> Result<Vec<Option<String>>> {
    last_relationship_rows(&QueryEngine.execute(
        LAST_RELATIONSHIP_QUERY,
        &mut context(fixture, backend, require_native_execution),
    )?)
}

fn execute(
    fixture: &Fixture,
    backend: Option<&dyn ExecutionBackend>,
    require_native_execution: bool,
    query: &str,
) -> Result<ExecutionOutput> {
    QueryEngine.execute(
        query,
        &mut context(fixture, backend, require_native_execution),
    )
}

fn assert_return7_result(output: &ExecutionOutput, fixture: &Fixture) -> Result<()> {
    assert!(
        output.graph_mutations.is_empty() && output.temporal_mutations.is_empty(),
        "Return7 [1] produced mutations"
    );
    assert_eq!(output.result.statistics, StatementStats::default());
    assert_eq!(output.result.bookmark, fixture.bookmark);
    assert!(!output.result.truncated);
    assert_eq!(
        output.result.schema,
        vec![
            ("a".to_owned(), ColumnType::Node),
            ("b".to_owned(), ColumnType::Node),
            ("p".to_owned(), ColumnType::Path),
        ]
    );

    let [batch] = output.result.batches.as_slice() else {
        return Err(Error::internal("Return7 [1] did not publish one batch"));
    };
    assert_eq!(batch.row_count, 1);
    let [a_column, b_column, p_column] = batch.columns.as_slice() else {
        return Err(Error::internal("Return7 [1] did not publish three columns"));
    };
    assert_eq!(
        (a_column.name.as_str(), &a_column.value_type),
        ("a", &ColumnType::Node)
    );
    assert_eq!(
        (b_column.name.as_str(), &b_column.value_type),
        ("b", &ColumnType::Node)
    );
    assert_eq!(
        (p_column.name.as_str(), &p_column.value_type),
        ("p", &ColumnType::Path)
    );

    let [ResultValue::Node(a)] = a_column.values.as_slice() else {
        return Err(Error::internal("Return7 [1] `a` was not one node"));
    };
    let [ResultValue::Node(b)] = b_column.values.as_slice() else {
        return Err(Error::internal("Return7 [1] `b` was not one node"));
    };
    assert_eq!(a.id, NodeId(1));
    assert_eq!(a.labels, vec!["Start".to_owned()]);
    assert!(a.properties.is_empty());
    assert_eq!(b.id, NodeId(2));
    assert!(b.labels.is_empty());
    assert!(b.properties.is_empty());

    let [
        ResultValue::Path {
            nodes,
            relationships,
        },
    ] = p_column.values.as_slice()
    else {
        return Err(Error::internal("Return7 [1] `p` was not one path"));
    };
    let [path_start, path_end] = nodes.as_slice() else {
        return Err(Error::internal(
            "Return7 [1] path did not contain two nodes",
        ));
    };
    let [relationship] = relationships.as_slice() else {
        return Err(Error::internal(
            "Return7 [1] path did not contain one relationship",
        ));
    };
    assert_eq!(path_start, a);
    assert_eq!(path_end, b);
    assert_eq!(relationship.id, EdgeId(10));
    assert_eq!(relationship.source, NodeId(1));
    assert_eq!(relationship.target, NodeId(2));
    assert_eq!(relationship.relationship_type, "T");
    assert!(relationship.properties.is_empty());
    Ok(())
}

fn assert_relationship_predicate_result(output: &ExecutionOutput, fixture: &Fixture) -> Result<()> {
    if !output.graph_mutations.is_empty()
        || !output.temporal_mutations.is_empty()
        || output.result.statistics != StatementStats::default()
        || output.result.bookmark != fixture.bookmark
        || output.result.truncated
        || output.result.schema
            != [
                ("a".to_owned(), ColumnType::Node),
                ("b".to_owned(), ColumnType::Node),
            ]
    {
        return Err(Error::internal(format!(
            "Match4 [5] changed its read-only result envelope: {output:#?}"
        )));
    }
    let [batch] = output.result.batches.as_slice() else {
        return Err(Error::internal("Match4 [5] did not publish one batch"));
    };
    let [a_column, b_column] = batch.columns.as_slice() else {
        return Err(Error::internal("Match4 [5] did not publish a/b columns"));
    };
    if batch.row_count != 1
        || !matches!(a_column.values.as_slice(), [ResultValue::Node(node)] if node.id == NodeId(2))
        || !matches!(b_column.values.as_slice(), [ResultValue::Node(node)] if node.id == NodeId(3))
    {
        return Err(Error::internal(format!(
            "Match4 [5] relationship predicate selected the wrong endpoints: {:?}",
            output.result
        )));
    }
    Ok(())
}

fn assert_match9_count_result(output: &ExecutionOutput, fixture: &Fixture) -> Result<()> {
    if !output.graph_mutations.is_empty()
        || !output.temporal_mutations.is_empty()
        || output.result.statistics != StatementStats::default()
        || output.result.bookmark != fixture.bookmark
        || output.result.truncated
        || output.result.schema != [("count(r)".to_owned(), ColumnType::Integer)]
        || !matches!(
            output.result.batches.as_slice(),
            [batch]
                if batch.row_count == 1
                    && matches!(batch.columns.as_slice(), [column]
                        if column.values
                            == [ResultValue::Scalar(ScalarValue::Integer(1))])
        )
    {
        return Err(Error::internal(format!(
            "Match9 [5] changed its path-cardinality result: {output:#?}"
        )));
    }
    Ok(())
}

#[derive(Default)]
struct RouteObservations {
    pins: AtomicUsize,
    variable_paths: AtomicUsize,
    variable_path_requests: Mutex<Vec<ResidentVariablePathRequest>>,
    scans: AtomicUsize,
    adjacency: AtomicUsize,
    unreceipted_node_pipelines: AtomicUsize,
}

struct ObservedBackend {
    inner: Box<dyn ExecutionBackend>,
    reported_kind: BackendKind,
    observations: Arc<RouteObservations>,
    poison_decomposition: bool,
}

impl ObservedBackend {
    fn new(inner: impl ExecutionBackend + 'static) -> Self {
        let reported_kind = inner.kind();
        Self {
            inner: Box::new(inner),
            reported_kind,
            observations: Arc::new(RouteObservations::default()),
            poison_decomposition: false,
        }
    }

    fn reporting(inner: impl ExecutionBackend + 'static, reported_kind: BackendKind) -> Self {
        Self {
            inner: Box::new(inner),
            reported_kind,
            observations: Arc::new(RouteObservations::default()),
            poison_decomposition: false,
        }
    }

    fn poisoning(inner: impl ExecutionBackend + 'static) -> Self {
        let reported_kind = inner.kind();
        Self {
            inner: Box::new(inner),
            reported_kind,
            observations: Arc::new(RouteObservations::default()),
            poison_decomposition: true,
        }
    }

    fn observations(&self) -> Arc<RouteObservations> {
        Arc::clone(&self.observations)
    }

    fn reject_decomposition<T>(&self, route: &'static str) -> Result<T> {
        Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!("strict variable-path gate rejected decomposed `{route}`"),
        ))
    }
}

impl ExecutionBackend for ObservedBackend {
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
        self.observations.pins.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Self {
            inner: self.inner.pin_project(project)?,
            reported_kind: self.reported_kind,
            observations: Arc::clone(&self.observations),
            poison_decomposition: self.poison_decomposition,
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
        self.observations.scans.fetch_add(1, Ordering::SeqCst);
        if self.poison_decomposition {
            return self.reject_decomposition("scan_nodes");
        }
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
        if self.poison_decomposition {
            return self.reject_decomposition("filter_node_i64");
        }
        self.inner
            .filter_node_i64(project, property, operation, operand, cancellation)
    }

    fn expand_project_out(
        &self,
        project: ProjectId,
        sources: &[u32],
        cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.observations.adjacency.fetch_add(1, Ordering::SeqCst);
        if self.poison_decomposition {
            return self.reject_decomposition("expand_project_out");
        }
        self.inner
            .expand_project_out(project, sources, cancellation)
    }

    fn expand_project_in(
        &self,
        project: ProjectId,
        targets: &[u32],
        cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.observations.adjacency.fetch_add(1, Ordering::SeqCst);
        if self.poison_decomposition {
            return self.reject_decomposition("expand_project_in");
        }
        self.inner.expand_project_in(project, targets, cancellation)
    }

    fn search_vectors(
        &self,
        request: &ResidentVectorQuery,
        cancellation: &CancellationToken,
    ) -> Result<ResidentVectorResult> {
        if self.poison_decomposition {
            return self.reject_decomposition("search_vectors");
        }
        self.inner.search_vectors(request, cancellation)
    }

    fn sort_rows(
        &self,
        request: &ResidentSortRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentSortResult> {
        if self.poison_decomposition {
            return self.reject_decomposition("sort_rows");
        }
        self.inner.sort_rows(request, cancellation)
    }

    fn join_node_i64(
        &self,
        request: &ResidentJoinRequest,
        cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentJoinPair>> {
        if self.poison_decomposition {
            return self.reject_decomposition("join_node_i64");
        }
        self.inner.join_node_i64(request, cancellation)
    }

    fn group_node_i64(
        &self,
        request: &ResidentGroupRequest,
        cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentGroup>> {
        if self.poison_decomposition {
            return self.reject_decomposition("group_node_i64");
        }
        self.inner.group_node_i64(request, cancellation)
    }

    fn execute_node_pipeline(
        &self,
        request: &ResidentNodePipelineRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentNodePipelineResult> {
        self.observations
            .unreceipted_node_pipelines
            .fetch_add(1, Ordering::SeqCst);
        if self.poison_decomposition {
            return self.reject_decomposition("execute_node_pipeline");
        }
        self.inner.execute_node_pipeline(request, cancellation)
    }

    fn supports_native_variable_path(&self) -> bool {
        self.inner.supports_native_variable_path()
    }

    fn execute_variable_path(
        &self,
        request: &ResidentVariablePathRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentVariablePathResult> {
        self.observations
            .variable_paths
            .fetch_add(1, Ordering::SeqCst);
        self.observations
            .variable_path_requests
            .lock()
            .expect("variable-path request observations poisoned")
            .push(request.clone());
        self.inner.execute_variable_path(request, cancellation)
    }

    fn filter_i64(
        &self,
        values: &[i64],
        validity: &[bool],
        operation: CompareOp,
        operand: i64,
        cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        if self.poison_decomposition {
            return self.reject_decomposition("filter_i64");
        }
        self.inner
            .filter_i64(values, validity, operation, operand, cancellation)
    }

    fn expand_out(
        &self,
        sources: &[u32],
        cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.observations.adjacency.fetch_add(1, Ordering::SeqCst);
        if self.poison_decomposition {
            return self.reject_decomposition("expand_out");
        }
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
        if self.poison_decomposition {
            return self.reject_decomposition("exact_l2");
        }
        self.inner
            .exact_l2(matrix, rows, dimension, query, cancellation)
    }
}

fn assert_single_native_route(observations: &RouteObservations, label: &str) {
    assert_eq!(
        observations.pins.load(Ordering::SeqCst),
        1,
        "{label}: pin count"
    );
    assert_eq!(
        observations.variable_paths.load(Ordering::SeqCst),
        1,
        "{label}: native variable-path count"
    );
    assert_eq!(
        observations.scans.load(Ordering::SeqCst),
        0,
        "{label}: host-visible scan"
    );
    assert_eq!(
        observations.adjacency.load(Ordering::SeqCst),
        0,
        "{label}: Rust-driven adjacency"
    );
    assert_eq!(
        observations
            .unreceipted_node_pipelines
            .load(Ordering::SeqCst),
        0,
        "{label}: unreceipted node pipeline"
    );
}

#[test]
fn manifest_is_exactly_the_two_official_match4_relationship_list_scenarios() {
    assert_eq!(SCENARIOS.len(), 2);
    assert_eq!(
        SCENARIOS
            .iter()
            .map(|scenario| scenario.id)
            .collect::<Vec<_>>(),
        [1, 6]
    );
    assert_eq!(FEATURE, "features/clauses/match/Match4.feature");
    assert!(
        SCENARIOS
            .iter()
            .all(|scenario| scenario.query.ends_with("RETURN r"))
    );
}

#[test]
fn next_match4_tranche_is_exactly_scenario_5_relationship_property_equality() {
    assert_eq!(RELATIONSHIP_PREDICATE_SCENARIO_ID, 5);
    assert_eq!(
        RELATIONSHIP_PREDICATE_QUERY,
        "MATCH (a:Artist)-[:WORKED_WITH* {year: 1988}]->(b:Artist) RETURN *"
    );
}

#[test]
fn next_manifest_is_exactly_match9_scenario_1_last_relationship() {
    assert_eq!(
        LAST_RELATIONSHIP_FEATURE,
        "features/clauses/match/Match9.feature"
    );
    assert_eq!(LAST_RELATIONSHIP_SCENARIO_ID, 1);
    assert_eq!(
        LAST_RELATIONSHIP_QUERY,
        "MATCH ()-[r*0..1]-() RETURN last(r) AS l"
    );
}

#[test]
fn next_match9_tranche_is_exactly_scenario_5_path_cardinality() {
    assert_eq!(MATCH9_COUNT_SCENARIO_ID, 5);
    assert_eq!(
        MATCH9_COUNT_QUERY,
        "MATCH (a:Blue)-[r*]->(b:Green) RETURN count(r)"
    );
    assert_eq!(
        MATCH9_BASELINE_REPORT,
        "/tmp/irongraph-tck-full-20260721-after-merge6-set-typeconversion-return5-r1.json"
    );
    assert_eq!(
        MATCH9_BASELINE_REPORT_SHA256,
        "f47070718ad35a3fade3335ec4d3e81d4ac4a77bc3e143942f92041098ea4e77"
    );
}

#[test]
fn return7_manifest_is_exactly_scenario_1_named_path_wildcard() {
    assert_eq!(RETURN7_FEATURE, "features/clauses/return/Return7.feature");
    assert_eq!(RETURN7_SCENARIO_ID, 1);
    assert_eq!(RETURN7_QUERY, "MATCH p = (a:Start)-->(b) RETURN *");
}

#[test]
fn generic_cpu_oracle_confirms_the_two_official_relationship_lists() -> Result<()> {
    for scenario in SCENARIOS {
        let fixture = fixture(*scenario)?;
        assert_eq!(
            execute_relationship_types(&fixture, None, false, *scenario)?,
            vec![
                scenario
                    .expected_relationship_types
                    .iter()
                    .map(|name| (*name).to_owned())
                    .collect::<Vec<_>>()
            ],
            "{FEATURE} [{}] generic CPU oracle mismatch",
            scenario.id
        );
    }
    Ok(())
}

#[test]
fn strict_cpu_publishes_relationship_lists_from_one_receipted_path_call() -> Result<()> {
    for scenario in SCENARIOS {
        let fixture = fixture(*scenario)?;
        let backend = ObservedBackend::new(cpu_backend(&fixture)?);
        let observations = backend.observations();
        assert_eq!(
            execute_relationship_types(&fixture, Some(&backend), true, *scenario)?,
            vec![
                scenario
                    .expected_relationship_types
                    .iter()
                    .map(|name| (*name).to_owned())
                    .collect::<Vec<_>>()
            ],
            "{FEATURE} [{}] strict CPU mismatch",
            scenario.id
        );
        assert_single_native_route(&observations, "strict CPU relationship-list route");
    }
    Ok(())
}

#[test]
fn generic_cpu_oracle_confirms_match4_relationship_property_equality() -> Result<()> {
    let fixture = relationship_predicate_fixture()?;
    let output = execute(&fixture, None, false, RELATIONSHIP_PREDICATE_QUERY)?;
    assert_relationship_predicate_result(&output, &fixture)
}

#[test]
fn strict_cpu_certifies_remaining_match4_05_relationship_property_equality() -> Result<()> {
    let fixture = relationship_predicate_fixture()?;
    let oracle = execute(&fixture, None, false, RELATIONSHIP_PREDICATE_QUERY)?;
    assert_relationship_predicate_result(&oracle, &fixture)?;

    let backend = ObservedBackend::poisoning(cpu_backend(&fixture)?);
    let observations = backend.observations();
    let actual = execute(&fixture, Some(&backend), true, RELATIONSHIP_PREDICATE_QUERY)?;
    assert_relationship_predicate_result(&actual, &fixture)?;
    assert_eq!(actual.result, oracle.result);
    assert_single_native_route(&observations, "strict CPU Match4 [5] property predicate");
    Ok(())
}

#[test]
fn strict_cpu_certifies_remaining_match4_variable_path_cluster() -> Result<()> {
    for (fixture, query, label, expected_rows) in [
        (
            intermediate_boundary_fixture()?,
            INTERMEDIATE_BOUNDARY_QUERY,
            "Match4 [3] intermediate boundary",
            3,
        ),
        (
            bound_relationship_count_fixture()?,
            BOUND_RELATIONSHIP_COUNT_QUERY,
            "Match4 [7] bound relationship count",
            1,
        ),
        (
            bound_relationship_list_fixture()?,
            BOUND_RELATIONSHIP_LIST_QUERY,
            "Match4 [8] bound relationship list",
            1,
        ),
    ] {
        let oracle = execute(&fixture, None, false, query)?;
        let backend = ObservedBackend::poisoning(cpu_backend(&fixture)?);
        let observations = backend.observations();
        let actual = execute(&fixture, Some(&backend), true, query)?;
        assert_eq!(actual.result, oracle.result, "{label}");
        assert_eq!(
            actual
                .result
                .batches
                .iter()
                .map(|batch| batch.row_count)
                .sum::<usize>(),
            expected_rows,
            "{label}: row count",
        );
        if query == BOUND_RELATIONSHIP_COUNT_QUERY {
            assert!(matches!(
                actual.result.batches.as_slice(),
                [batch]
                    if matches!(batch.columns.as_slice(), [column]
                        if column.name == "c"
                            && column.values
                                == [ResultValue::Scalar(ScalarValue::Integer(32))])
            ));
        }
        assert_single_native_route(&observations, label);
    }
    Ok(())
}

#[test]
fn strict_cpu_publishes_all_direct_path_outputs_from_the_same_validated_trail() -> Result<()> {
    let fixture = complete_projection_fixture()?;
    let oracle = execute(&fixture, None, false, COMPLETE_PROJECTION_QUERY)?;
    let backend = ObservedBackend::new(cpu_backend(&fixture)?);
    let observations = backend.observations();
    let actual = execute(&fixture, Some(&backend), true, COMPLETE_PROJECTION_QUERY)?;

    assert_eq!(actual.result, oracle.result);
    assert_eq!(
        actual
            .result
            .batches
            .iter()
            .map(|batch| batch.row_count)
            .sum::<usize>(),
        2
    );
    assert_single_native_route(&observations, "strict CPU complete path projection");
    Ok(())
}

#[test]
fn strict_cpu_publishes_return2_entity_containers_from_one_receipted_path() -> Result<()> {
    let fixture = return2_entity_container_fixture()?;
    for (label, query) in [
        ("Return2 [12] entity list", RETURN2_ENTITY_LIST_QUERY),
        ("Return2 [13] entity map", RETURN2_ENTITY_MAP_QUERY),
    ] {
        let oracle = execute(&fixture, None, false, query)?;
        let backend = ObservedBackend::new(cpu_backend(&fixture)?);
        let observations = backend.observations();
        let actual = execute(&fixture, Some(&backend), true, query)?;
        assert_eq!(actual.result, oracle.result, "{label}");
        assert_single_native_route(&observations, label);
    }
    Ok(())
}

#[test]
fn strict_cpu_certifies_with3_relationship_identity_rematch_and_order() -> Result<()> {
    let fixture = with3_relationship_rematch_fixture()?;
    let oracle = execute(&fixture, None, false, WITH3_RELATIONSHIP_REMATCH_QUERY)?;
    let backend = ObservedBackend::new(cpu_backend(&fixture)?);
    let observations = backend.observations();
    let actual = execute(
        &fixture,
        Some(&backend),
        true,
        WITH3_RELATIONSHIP_REMATCH_QUERY,
    )?;
    assert_eq!(actual.result, oracle.result);
    assert_single_native_route(&observations, "strict CPU With3 relationship rematch");
    Ok(())
}

#[test]
fn strict_cpu_certifies_with7_grouped_intermediate_filters_and_count() -> Result<()> {
    let fixture = with7_grouped_intermediate_fixture()?;
    let oracle = execute(&fixture, None, false, WITH7_GROUPED_INTERMEDIATE_QUERY)?;
    let backend = ObservedBackend::new(cpu_backend(&fixture)?);
    let observations = backend.observations();
    let actual = execute(
        &fixture,
        Some(&backend),
        true,
        WITH7_GROUPED_INTERMEDIATE_QUERY,
    )?;
    assert_eq!(actual.result, oracle.result);
    assert_single_native_route(&observations, "strict CPU With7 grouped intermediate count");
    Ok(())
}

#[test]
fn strict_cpu_publishes_return7_wildcard_from_one_receipted_path_call() -> Result<()> {
    let fixture = return7_fixture()?;
    let oracle = execute(&fixture, None, false, RETURN7_QUERY)?;
    assert_return7_result(&oracle, &fixture)?;

    let backend = ObservedBackend::new(cpu_backend(&fixture)?);
    let observations = backend.observations();
    let actual = execute(&fixture, Some(&backend), true, RETURN7_QUERY)?;

    assert_return7_result(&actual, &fixture)?;
    assert_eq!(actual.result, oracle.result);
    assert_single_native_route(&observations, "strict CPU Return7 wildcard");
    Ok(())
}

#[test]
fn generic_cpu_oracle_confirms_match9_last_relationship_multiplicity() -> Result<()> {
    let fixture = last_relationship_fixture()?;
    assert_eq!(
        execute_last_relationship(&fixture, None, false)?,
        vec![None, None, None, Some("T".to_owned()), Some("T".to_owned()),]
    );
    Ok(())
}

#[test]
fn generic_cpu_oracle_confirms_match9_path_cardinality() -> Result<()> {
    let fixture = match9_count_fixture()?;
    let output = execute(&fixture, None, false, MATCH9_COUNT_QUERY)?;
    assert_match9_count_result(&output, &fixture)
}

#[test]
fn strict_cpu_match9_path_cardinality_cannot_decompose() -> Result<()> {
    let fixture = match9_count_fixture()?;
    let oracle = execute(&fixture, None, false, MATCH9_COUNT_QUERY)?;
    assert_match9_count_result(&oracle, &fixture)?;

    let backend = ObservedBackend::poisoning(cpu_backend(&fixture)?);
    let observations = backend.observations();
    let actual = execute(&fixture, Some(&backend), true, MATCH9_COUNT_QUERY)?;
    assert_match9_count_result(&actual, &fixture)?;
    assert_eq!(actual.result, oracle.result);
    assert_single_native_route(&observations, "strict CPU Match9 [5] path cardinality");
    Ok(())
}

#[test]
fn strict_cpu_certifies_match9_06_08_and_comparison1_14_sealed_path_tails() -> Result<()> {
    for (fixture, query, label, equality) in [
        (
            bound_relationship_list_fixture()?,
            MATCH9_SAME_REMATCH_QUERY,
            "Match9 [6] same rematch",
            false,
        ),
        (
            bound_relationship_list_fixture()?,
            MATCH9_REVERSE_REMATCH_QUERY,
            "Match9 [7] reverse rematch",
            false,
        ),
        (
            match9_optional_null_fixture()?,
            MATCH9_OPTIONAL_NULL_QUERY,
            "Match9 [8] optional null path",
            false,
        ),
        (
            comparison1_path_equality_fixture()?,
            COMPARISON1_PATH_EQUALITY_QUERY,
            "Comparison1 [14] path equality",
            true,
        ),
    ] {
        let oracle = execute(&fixture, None, false, query)?;
        let backend = ObservedBackend::poisoning(cpu_backend(&fixture)?);
        let observations = backend.observations();
        let actual = execute(&fixture, Some(&backend), true, query)?;

        assert_eq!(actual.result, oracle.result, "{label}");
        assert_single_native_route(&observations, label);
        let requests = observations
            .variable_path_requests
            .lock()
            .expect("sealed path-tail request observations poisoned");
        let [request] = requests.as_slice() else {
            return Err(Error::internal(
                "sealed path tail did not issue exactly one variable-path request",
            ));
        };
        request.validate()?;
        if equality {
            assert!(matches!(
                request.final_projection,
                irongraph::gpu::ResidentVariablePathFinalProjection::IndependentOneHopPathEquality {
                    ..
                }
            ));
        } else {
            assert!(matches!(
                request.final_projection,
                irongraph::gpu::ResidentVariablePathFinalProjection::SelectedPublications { .. }
            ));
        }
    }
    Ok(())
}

#[test]
fn strict_cpu_certifies_where4_cycle_chord_and_independent_sum_path_tails() -> Result<()> {
    for (fixture, query, label) in [
        (
            correlated_path_disjunction_fixture()?,
            CORRELATED_PATH_DISJUNCTION_QUERY,
            "MatchWhere4 [2] correlated path disjunction",
        ),
        (
            correlated_path_disjunction_fixture()?,
            CORRELATED_PATH_DISJUNCTION_WITH_QUERY,
            "WithWhere4 [2] correlated path disjunction",
        ),
        (
            cycle_chord_fixture()?,
            CYCLE_CHORD_QUERY,
            "MatchWhere2 [1] cycle chord",
        ),
        (
            independent_path_sum_fixture()?,
            INDEPENDENT_PATH_SUM_QUERY,
            "Match8 [3] independent path sum",
        ),
    ] {
        let oracle = execute(&fixture, None, false, query)?;
        let backend = ObservedBackend::poisoning(cpu_backend(&fixture)?);
        let observations = backend.observations();
        let actual = execute(&fixture, Some(&backend), true, query)
            .map_err(|error| Error::new(error.code, format!("{label}: {}", error.message)))?;

        assert_eq!(actual.result, oracle.result, "{label}");
        assert_single_native_route(&observations, label);
        let requests = observations
            .variable_path_requests
            .lock()
            .expect("topology-tail request observations poisoned");
        let [request] = requests.as_slice() else {
            return Err(Error::internal(
                "topology tail did not issue exactly one variable-path request",
            ));
        };
        request.validate()?;
        match query {
            CORRELATED_PATH_DISJUNCTION_QUERY | CORRELATED_PATH_DISJUNCTION_WITH_QUERY => {
                assert!(matches!(
                    request.final_projection,
                    irongraph::gpu::ResidentVariablePathFinalProjection::CorrelatedOutgoingPathDisjunction {
                        target_label: None,
                        start_value: 0,
                        ..
                    }
                ));
            }
            CYCLE_CHORD_QUERY => assert!(matches!(
                request.final_projection,
                irongraph::gpu::ResidentVariablePathFinalProjection::UndirectedCycleChordNodeFilter {
                    start_value: 1,
                    second_value: 2,
                    ..
                }
            )),
            INDEPENDENT_PATH_SUM_QUERY => assert!(matches!(
                request.final_projection,
                irongraph::gpu::ResidentVariablePathFinalProjection::IndependentPathRelationshipPropertySum {
                    relationship_segment: 0,
                    ..
                }
            )),
            _ => return Err(Error::internal("unknown topology-tail proof query")),
        }
    }
    Ok(())
}

#[test]
fn strict_cpu_publishes_match9_last_relationship_from_one_receipted_path_call() -> Result<()> {
    let fixture = last_relationship_fixture()?;
    let backend = ObservedBackend::new(cpu_backend(&fixture)?);
    let observations = backend.observations();
    assert_eq!(
        execute_last_relationship(&fixture, Some(&backend), true)?,
        vec![None, None, None, Some("T".to_owned()), Some("T".to_owned()),]
    );
    assert_single_native_route(&observations, "strict CPU Match9 last relationship");
    Ok(())
}

#[test]
fn cpu_completion_cannot_masquerade_as_metal_relationship_list_publication() -> Result<()> {
    let scenario = SCENARIOS[0];
    let fixture = fixture(scenario)?;
    let backend = ObservedBackend::reporting(cpu_backend(&fixture)?, BackendKind::Metal);
    let observations = backend.observations();
    let mut emitted = Vec::<ExecutionStreamItem>::new();
    let error = QueryEngine
        .execute_streaming(
            scenario.query,
            &mut context(&fixture, Some(&backend), true),
            &mut |item| {
                emitted.push(item);
                Ok(())
            },
        )
        .expect_err("CPU completion must not publish as Metal");

    assert!(
        matches!(
            error.code,
            ErrorCode::GpuAdmissionFailure | ErrorCode::CorruptStorage
        ),
        "unexpected provenance rejection: {error}"
    );
    assert!(
        emitted.is_empty(),
        "provenance rejection emitted partial results"
    );
    assert_single_native_route(&observations, "CPU-as-Metal provenance rejection");
    Ok(())
}

#[test]
fn cpu_completion_cannot_masquerade_as_metal_last_relationship_publication() -> Result<()> {
    let fixture = last_relationship_fixture()?;
    let backend = ObservedBackend::reporting(cpu_backend(&fixture)?, BackendKind::Metal);
    let observations = backend.observations();
    let mut emitted = Vec::<ExecutionStreamItem>::new();
    let error = QueryEngine
        .execute_streaming(
            LAST_RELATIONSHIP_QUERY,
            &mut context(&fixture, Some(&backend), true),
            &mut |item| {
                emitted.push(item);
                Ok(())
            },
        )
        .expect_err("CPU completion must not publish Match9 output as Metal");

    assert!(
        matches!(
            error.code,
            ErrorCode::GpuAdmissionFailure | ErrorCode::CorruptStorage
        ),
        "unexpected provenance rejection: {error}"
    );
    assert!(
        emitted.is_empty(),
        "provenance rejection emitted partial Match9 results"
    );
    assert_single_native_route(
        &observations,
        "CPU-as-Metal Match9 last-relationship rejection",
    );
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn metal_test_guard() -> MutexGuard<'static, ()> {
    static METAL_TEST: Mutex<()> = Mutex::new(());
    METAL_TEST
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "hardware acceptance gate: Match9 [6-8]/Comparison1 [14] require v6 Metal path tails"]
fn real_metal_certifies_match9_06_08_and_comparison1_14_path_tails() -> Result<()> {
    let _guard = metal_test_guard();
    for (fixture, query, label) in [
        (
            bound_relationship_list_fixture()?,
            MATCH9_SAME_REMATCH_QUERY,
            "Match9 [6] same rematch",
        ),
        (
            bound_relationship_list_fixture()?,
            MATCH9_REVERSE_REMATCH_QUERY,
            "Match9 [7] reverse rematch",
        ),
        (
            match9_optional_null_fixture()?,
            MATCH9_OPTIONAL_NULL_QUERY,
            "Match9 [8] optional null path",
        ),
        (
            comparison1_path_equality_fixture()?,
            COMPARISON1_PATH_EQUALITY_QUERY,
            "Comparison1 [14] path equality",
        ),
    ] {
        let oracle = execute(&fixture, None, false, query)?;
        let backend = ObservedBackend::new({
            let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
            metal.admit_project(resident_image(&fixture)?)?;
            metal
        });
        let observations = backend.observations();
        let actual = execute(&fixture, Some(&backend), true, query)?;
        assert_eq!(actual.result, oracle.result, "{label}");
        assert_single_native_route(&observations, label);
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "hardware acceptance gate: WHERE4/cycle-chord/independent-SUM require v7-v9 Metal path tails"]
fn real_metal_certifies_where4_cycle_chord_and_independent_sum_path_tails() -> Result<()> {
    let _guard = metal_test_guard();
    for (fixture, query, label) in [
        (
            correlated_path_disjunction_fixture()?,
            CORRELATED_PATH_DISJUNCTION_QUERY,
            "MatchWhere4 [2] correlated path disjunction",
        ),
        (
            correlated_path_disjunction_fixture()?,
            CORRELATED_PATH_DISJUNCTION_WITH_QUERY,
            "WithWhere4 [2] correlated path disjunction",
        ),
        (
            cycle_chord_fixture()?,
            CYCLE_CHORD_QUERY,
            "MatchWhere2 [1] cycle chord",
        ),
        (
            independent_path_sum_fixture()?,
            INDEPENDENT_PATH_SUM_QUERY,
            "Match8 [3] independent path sum",
        ),
    ] {
        let oracle = execute(&fixture, None, false, query)?;
        let backend = ObservedBackend::poisoning({
            let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
            metal.admit_project(resident_image(&fixture)?)?;
            metal
        });
        let observations = backend.observations();
        let actual = execute(&fixture, Some(&backend), true, query)?;
        assert_eq!(actual.result, oracle.result, "{label}");
        assert_single_native_route(&observations, label);
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "hardware acceptance gate: Match4 [1]/[6] require native Metal relationship-list publication"]
fn real_metal_publishes_both_official_relationship_lists_without_fallback() -> Result<()> {
    let _guard = metal_test_guard();
    let metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
    let mut backend = ObservedBackend::new(metal);

    for scenario in SCENARIOS {
        let fixture = fixture(*scenario)?;
        backend.admit_project(resident_image(&fixture)?)?;
        let observations = backend.observations();
        let pins_before = observations.pins.load(Ordering::SeqCst);
        let paths_before = observations.variable_paths.load(Ordering::SeqCst);
        let scans_before = observations.scans.load(Ordering::SeqCst);
        let adjacency_before = observations.adjacency.load(Ordering::SeqCst);
        let pipelines_before = observations
            .unreceipted_node_pipelines
            .load(Ordering::SeqCst);

        assert_eq!(
            execute_relationship_types(&fixture, Some(&backend), true, *scenario)?,
            vec![
                scenario
                    .expected_relationship_types
                    .iter()
                    .map(|name| (*name).to_owned())
                    .collect::<Vec<_>>()
            ],
            "{FEATURE} [{}] real Metal mismatch",
            scenario.id
        );
        assert_eq!(observations.pins.load(Ordering::SeqCst), pins_before + 1);
        assert_eq!(
            observations.variable_paths.load(Ordering::SeqCst),
            paths_before + 1
        );
        assert_eq!(observations.scans.load(Ordering::SeqCst), scans_before);
        assert_eq!(
            observations.adjacency.load(Ordering::SeqCst),
            adjacency_before
        );
        assert_eq!(
            observations
                .unreceipted_node_pipelines
                .load(Ordering::SeqCst),
            pipelines_before
        );
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "hardware acceptance gate: direct path values/functions require one native Metal trail"]
fn real_metal_publishes_all_direct_path_outputs_from_one_validated_trail() -> Result<()> {
    let _guard = metal_test_guard();
    let fixture = complete_projection_fixture()?;
    let oracle = execute(&fixture, None, false, COMPLETE_PROJECTION_QUERY)?;
    let backend = ObservedBackend::new({
        let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
        metal.admit_project(resident_image(&fixture)?)?;
        metal
    });
    let observations = backend.observations();
    let actual = execute(&fixture, Some(&backend), true, COMPLETE_PROJECTION_QUERY)?;

    assert_eq!(actual.result, oracle.result);
    assert_single_native_route(&observations, "real Metal complete path projection");
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "hardware acceptance gate: Return2 entity containers require one native Metal trail"]
fn real_metal_publishes_return2_entity_containers_from_one_validated_trail() -> Result<()> {
    let _guard = metal_test_guard();
    let fixture = return2_entity_container_fixture()?;
    for (label, query) in [
        ("Return2 [12] entity list", RETURN2_ENTITY_LIST_QUERY),
        ("Return2 [13] entity map", RETURN2_ENTITY_MAP_QUERY),
    ] {
        let oracle = execute(&fixture, None, false, query)?;
        let backend = ObservedBackend::new({
            let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
            metal.admit_project(resident_image(&fixture)?)?;
            metal
        });
        let observations = backend.observations();
        let actual = execute(&fixture, Some(&backend), true, query)?;
        assert_eq!(actual.result, oracle.result, "{label}");
        assert_single_native_route(&observations, label);
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "hardware acceptance gate: With3 relationship identity/order requires one native Metal trail"]
fn real_metal_certifies_with3_relationship_identity_rematch_and_order() -> Result<()> {
    let _guard = metal_test_guard();
    let fixture = with3_relationship_rematch_fixture()?;
    let oracle = execute(&fixture, None, false, WITH3_RELATIONSHIP_REMATCH_QUERY)?;
    let backend = ObservedBackend::new({
        let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
        metal.admit_project(resident_image(&fixture)?)?;
        metal
    });
    let observations = backend.observations();
    let actual = execute(
        &fixture,
        Some(&backend),
        true,
        WITH3_RELATIONSHIP_REMATCH_QUERY,
    )?;
    assert_eq!(actual.result, oracle.result);
    assert_single_native_route(&observations, "real Metal With3 relationship rematch");
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "hardware acceptance gate: With7 grouped intermediate count requires one sealed Metal trail"]
fn real_metal_certifies_with7_grouped_intermediate_filters_and_count() -> Result<()> {
    let _guard = metal_test_guard();
    let fixture = with7_grouped_intermediate_fixture()?;
    let oracle = execute(&fixture, None, false, WITH7_GROUPED_INTERMEDIATE_QUERY)?;
    let backend = ObservedBackend::new({
        let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
        metal.admit_project(resident_image(&fixture)?)?;
        metal
    });
    let observations = backend.observations();
    let actual = execute(
        &fixture,
        Some(&backend),
        true,
        WITH7_GROUPED_INTERMEDIATE_QUERY,
    )?;
    assert_eq!(actual.result, oracle.result);
    assert_single_native_route(&observations, "real Metal With7 grouped intermediate count");
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "hardware acceptance gate: Match9 [1] last(r) requires one native Metal trail"]
fn real_metal_publishes_match9_last_relationship_without_fallback() -> Result<()> {
    let _guard = metal_test_guard();
    let fixture = last_relationship_fixture()?;
    let backend = ObservedBackend::new({
        let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
        metal.admit_project(resident_image(&fixture)?)?;
        metal
    });
    let observations = backend.observations();

    assert_eq!(
        execute_last_relationship(&fixture, Some(&backend), true)?,
        vec![None, None, None, Some("T".to_owned()), Some("T".to_owned()),]
    );
    assert_single_native_route(&observations, "real Metal Match9 last relationship");
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "hardware acceptance gate: Match4 [3]/[5]/[8] require variable-path packet v4"]
fn real_metal_certifies_match4_packet_v4_boundary_predicate_and_bound_list() -> Result<()> {
    let _guard = metal_test_guard();
    for (fixture, query, label) in [
        (
            intermediate_boundary_fixture()?,
            INTERMEDIATE_BOUNDARY_QUERY,
            "Match4 [3] segment boundary",
        ),
        (
            relationship_predicate_fixture()?,
            RELATIONSHIP_PREDICATE_QUERY,
            "Match4 [5] relationship integer predicate",
        ),
        (
            bound_relationship_list_fixture()?,
            BOUND_RELATIONSHIP_LIST_QUERY,
            "Match4 [8] bound relationship list",
        ),
    ] {
        let oracle = execute(&fixture, None, false, query)?;
        let backend = ObservedBackend::new({
            let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
            metal.admit_project(resident_image(&fixture)?)?;
            metal
        });
        let observations = backend.observations();
        let actual = execute(&fixture, Some(&backend), true, query)?;
        assert_eq!(actual.result, oracle.result, "{label}");
        assert_single_native_route(&observations, label);
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "hardware acceptance gate: Match4 [7] requires a backend-authored final count relation"]
fn real_metal_certifies_match4_07_sealed_bound_relationship_count() -> Result<()> {
    let _guard = metal_test_guard();
    let fixture = bound_relationship_count_fixture()?;
    let oracle = execute(&fixture, None, false, BOUND_RELATIONSHIP_COUNT_QUERY)?;
    let backend = ObservedBackend::new({
        let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
        metal.admit_project(resident_image(&fixture)?)?;
        metal
    });
    let observations = backend.observations();
    let actual = execute(
        &fixture,
        Some(&backend),
        true,
        BOUND_RELATIONSHIP_COUNT_QUERY,
    )?;
    assert_eq!(actual.result, oracle.result);
    assert_single_native_route(&observations, "real Metal Match4 [7] sealed count");
    let requests = observations
        .variable_path_requests
        .lock()
        .expect("Match4 [7] request observations poisoned");
    assert!(matches!(
        requests.as_slice(),
        [request]
            if matches!(
                request.final_projection,
                irongraph::gpu::ResidentVariablePathFinalProjection::CountBoundUndirectedPaths {
                    segment: 1,
                    ..
                }
            )
    ));
    Ok(())
}
