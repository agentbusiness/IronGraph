// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Strict acceptance harness for the 17 successful relationship-CREATE scenarios in openCypher
//! `Create2.feature` (zero-based certified report IDs 72 through 88).
//!
//! The active, self-contained oracle executes the literal setup, primary, and control queries with
//! the generic CPU semantic reference and validates result rows, statement statistics, graph side
//! effects, direction, self-loops, endpoint identity, relationship type, and NULL-property
//! omission. The active strict CPU and Metal gates accept only the generalized, immutable,
//! generation-fenced row-CREATE command. Their observer closes every decomposed/legacy route; no
//! existing primitive is treated as a substitute receipt.

use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use std::sync::{Mutex, MutexGuard};

use irongraph::{
    Bookmark, Error, ErrorCode, Layer, NodeId, ProjectId, Result, ScalarValue,
    cypher::{
        BindCapabilities, ColumnType, ExecutionContext, ExecutionOutput, QueryEngine, ResultValue,
        StatementStats,
    },
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend,
        ResidentCreateNodeRequest, ResidentCreateNodeResult, ResidentCreateNodeValueInput,
        ResidentGroup, ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest,
        ResidentNodePipelineRequest, ResidentNodePipelineResult, ResidentProjectDelta,
        ResidentProjectImage, ResidentRowBoundRelationshipCommand,
        ResidentRowBoundRelationshipStage, ResidentRowCreateCommand, ResidentRowCreateValueInput,
        ResidentRowMutationRequest, ResidentRowMutationResult, ResidentRowProgramRequest,
        ResidentRowProgramResult, ResidentSortRequest, ResidentSortResult,
        ResidentTemporalPipelineRequest, ResidentTemporalPipelineResult, ResidentVectorQuery,
        ResidentVectorResult, ScratchReservation,
    },
    graph::{GraphMutation, GraphStore, IndexCatalog, LayerMask, TemporalStore},
    types::{LabelId, PropertyId},
};
use serde::Deserialize;
use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::gpu::MetalBackend;

const REPORT_PATH: &str = "/tmp/irongraph-tck-full-delete-temporal-cpu-certified.json";
const PINNED_FEATURE_ROOT: &str =
    "/private/tmp/irongraph-opencypher-debug.6MXlLm/openCypher/tck/features";
const FEATURE: &str = "clauses/create/Create2.feature";
const PROJECT: ProjectId = ProjectId(uuid::Uuid::from_u128(
    0x4352_4541_5445_325f_5245_4c5f_3031_0001,
));
const MEMORY_LIMIT_BYTES: usize = 512 * 1024 * 1024;
const RESERVED_BYTES: usize = 64 * 1024 * 1024;
const MAX_RESULT_ROWS: usize = 100_000;
const MISSING_GRAPH_WRITE_CONTRACT: &str =
    "active GPU execution class has no complete resident implementation for this query plan";

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Effects {
    nodes: usize,
    relationships: usize,
    properties: usize,
    labels: usize,
}

impl Effects {
    const fn statistics(self) -> StatementStats {
        StatementStats {
            nodes_created: self.nodes as u64,
            nodes_deleted: 0,
            relationships_created: self.relationships as u64,
            relationships_deleted: 0,
            // TCK side-effect properties created as part of CREATE are not SET-clause statistics.
            properties_set: 0,
            labels_added: self.labels as u64,
            labels_removed: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ExpectedScalar {
    Null,
    Integer(i64),
    String(&'static str),
}

impl ExpectedScalar {
    fn value(self) -> ScalarValue {
        match self {
            Self::Null => ScalarValue::Null,
            Self::Integer(value) => ScalarValue::Integer(value),
            Self::String(value) => ScalarValue::String(Arc::from(value)),
        }
    }

    const fn column_type(self) -> ColumnType {
        match self {
            Self::Null => ColumnType::Null,
            Self::Integer(_) => ColumnType::Integer,
            Self::String(_) => ColumnType::String,
        }
    }
}

#[derive(Clone, Copy, Debug)]
struct ExpectedColumn {
    name: &'static str,
    value: ExpectedScalar,
}

#[derive(Clone, Copy, Debug)]
struct ExpectedNodeColumn {
    name: &'static str,
    labels: &'static [&'static str],
}

#[derive(Clone, Copy, Debug)]
struct ControlCase {
    query: &'static str,
    columns: &'static [ExpectedNodeColumn],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Endpoint {
    New(usize),
    Label(&'static str),
}

#[derive(Clone, Copy, Debug)]
struct ExpectedRelationship {
    relationship_type: &'static str,
    source: Endpoint,
    target: Endpoint,
    properties: &'static [(&'static str, ExpectedScalar)],
}

#[derive(Clone, Copy, Debug)]
struct Create2Case {
    report_id: usize,
    scenario: u8,
    name: &'static str,
    setup_query: Option<&'static str>,
    setup_effects: Effects,
    query: &'static str,
    controls: &'static [ControlCase],
    effects: Effects,
    columns: &'static [ExpectedColumn],
    relationship: ExpectedRelationship,
    read_entities: usize,
}

impl Create2Case {
    fn label(self) -> String {
        format!(
            "TCK id={} {FEATURE} [{}] {}",
            self.report_id, self.scenario, self.name
        )
    }
}

const NO_EFFECTS: Effects = Effects {
    nodes: 0,
    relationships: 0,
    properties: 0,
    labels: 0,
};
const TWO_NODES_ONE_EDGE: Effects = Effects {
    nodes: 2,
    relationships: 1,
    properties: 0,
    labels: 0,
};
const ONE_NODE_ONE_EDGE: Effects = Effects {
    nodes: 1,
    relationships: 1,
    properties: 0,
    labels: 0,
};
const ONE_EDGE: Effects = Effects {
    nodes: 0,
    relationships: 1,
    properties: 0,
    labels: 0,
};

const CONTROL_AB: [ExpectedNodeColumn; 2] = [
    ExpectedNodeColumn {
        name: "a",
        labels: &["A"],
    },
    ExpectedNodeColumn {
        name: "b",
        labels: &["B"],
    },
];
const CONTROL_XY: [ExpectedNodeColumn; 2] = [
    ExpectedNodeColumn {
        name: "x",
        labels: &["X"],
    },
    ExpectedNodeColumn {
        name: "y",
        labels: &["Y"],
    },
];
const CONTROL_BEGIN_END: [ExpectedNodeColumn; 2] = [
    ExpectedNodeColumn {
        name: "x",
        labels: &["Begin"],
    },
    ExpectedNodeColumn {
        name: "y",
        labels: &["End"],
    },
];

const CASES: [Create2Case; 17] = [
    Create2Case {
        report_id: 72,
        scenario: 1,
        name: "Create two nodes and a single relationship in a single pattern",
        setup_query: None,
        setup_effects: NO_EFFECTS,
        query: "CREATE ()-[:R]->()",
        controls: &[],
        effects: TWO_NODES_ONE_EDGE,
        columns: &[],
        relationship: ExpectedRelationship {
            relationship_type: "R",
            source: Endpoint::New(0),
            target: Endpoint::New(1),
            properties: &[],
        },
        read_entities: 0,
    },
    Create2Case {
        report_id: 73,
        scenario: 2,
        name: "Create two nodes and a single relationship in separate patterns",
        setup_query: None,
        setup_effects: NO_EFFECTS,
        query: "CREATE (a), (b), (a)-[:R]->(b)",
        controls: &[],
        effects: TWO_NODES_ONE_EDGE,
        columns: &[],
        relationship: ExpectedRelationship {
            relationship_type: "R",
            source: Endpoint::New(0),
            target: Endpoint::New(1),
            properties: &[],
        },
        read_entities: 0,
    },
    Create2Case {
        report_id: 74,
        scenario: 3,
        name: "Create two nodes and a single relationship in separate clauses",
        setup_query: None,
        setup_effects: NO_EFFECTS,
        query: "CREATE (a) CREATE (b) CREATE (a)-[:R]->(b)",
        controls: &[],
        effects: TWO_NODES_ONE_EDGE,
        columns: &[],
        relationship: ExpectedRelationship {
            relationship_type: "R",
            source: Endpoint::New(0),
            target: Endpoint::New(1),
            properties: &[],
        },
        read_entities: 0,
    },
    Create2Case {
        report_id: 75,
        scenario: 4,
        name: "Create two nodes and a single relationship in the reverse direction",
        setup_query: None,
        setup_effects: NO_EFFECTS,
        query: "CREATE (:A)<-[:R]-(:B)",
        controls: &[ControlCase {
            query: "MATCH (a:A)<-[:R]-(b:B) RETURN a, b",
            columns: &CONTROL_AB,
        }],
        effects: Effects {
            labels: 2,
            ..TWO_NODES_ONE_EDGE
        },
        columns: &[],
        relationship: ExpectedRelationship {
            relationship_type: "R",
            source: Endpoint::Label("B"),
            target: Endpoint::Label("A"),
            properties: &[],
        },
        read_entities: 0,
    },
    Create2Case {
        report_id: 76,
        scenario: 5,
        name: "Create a single relationship between two existing nodes",
        setup_query: Some("CREATE (:X) CREATE (:Y)"),
        setup_effects: Effects {
            nodes: 2,
            labels: 2,
            ..NO_EFFECTS
        },
        query: "MATCH (x:X), (y:Y) CREATE (x)-[:R]->(y)",
        controls: &[],
        effects: ONE_EDGE,
        columns: &[],
        relationship: ExpectedRelationship {
            relationship_type: "R",
            source: Endpoint::Label("X"),
            target: Endpoint::Label("Y"),
            properties: &[],
        },
        read_entities: 2,
    },
    Create2Case {
        report_id: 77,
        scenario: 6,
        name: "Create a single relationship between two existing nodes in the reverse direction",
        setup_query: Some("CREATE (:X) CREATE (:Y)"),
        setup_effects: Effects {
            nodes: 2,
            labels: 2,
            ..NO_EFFECTS
        },
        query: "MATCH (x:X), (y:Y) CREATE (x)<-[:R]-(y)",
        controls: &[ControlCase {
            query: "MATCH (x:X)<-[:R]-(y:Y) RETURN x, y",
            columns: &CONTROL_XY,
        }],
        effects: ONE_EDGE,
        columns: &[],
        relationship: ExpectedRelationship {
            relationship_type: "R",
            source: Endpoint::Label("Y"),
            target: Endpoint::Label("X"),
            properties: &[],
        },
        read_entities: 2,
    },
    Create2Case {
        report_id: 78,
        scenario: 7,
        name: "Create a single node and a single self loop in a single pattern",
        setup_query: None,
        setup_effects: NO_EFFECTS,
        query: "CREATE (root)-[:LINK]->(root)",
        controls: &[],
        effects: ONE_NODE_ONE_EDGE,
        columns: &[],
        relationship: ExpectedRelationship {
            relationship_type: "LINK",
            source: Endpoint::New(0),
            target: Endpoint::New(0),
            properties: &[],
        },
        read_entities: 0,
    },
    Create2Case {
        report_id: 79,
        scenario: 8,
        name: "Create a single node and a single self loop in separate patterns",
        setup_query: None,
        setup_effects: NO_EFFECTS,
        query: "CREATE (root), (root)-[:LINK]->(root)",
        controls: &[],
        effects: ONE_NODE_ONE_EDGE,
        columns: &[],
        relationship: ExpectedRelationship {
            relationship_type: "LINK",
            source: Endpoint::New(0),
            target: Endpoint::New(0),
            properties: &[],
        },
        read_entities: 0,
    },
    Create2Case {
        report_id: 80,
        scenario: 9,
        name: "Create a single node and a single self loop in separate clauses",
        setup_query: None,
        setup_effects: NO_EFFECTS,
        query: "CREATE (root) CREATE (root)-[:LINK]->(root)",
        controls: &[],
        effects: ONE_NODE_ONE_EDGE,
        columns: &[],
        relationship: ExpectedRelationship {
            relationship_type: "LINK",
            source: Endpoint::New(0),
            target: Endpoint::New(0),
            properties: &[],
        },
        read_entities: 0,
    },
    Create2Case {
        report_id: 81,
        scenario: 10,
        name: "Create a single self loop on an existing node",
        setup_query: Some("CREATE (:Root)"),
        setup_effects: Effects {
            nodes: 1,
            labels: 1,
            ..NO_EFFECTS
        },
        query: "MATCH (root:Root) CREATE (root)-[:LINK]->(root)",
        controls: &[],
        effects: ONE_EDGE,
        columns: &[],
        relationship: ExpectedRelationship {
            relationship_type: "LINK",
            source: Endpoint::Label("Root"),
            target: Endpoint::Label("Root"),
            properties: &[],
        },
        read_entities: 1,
    },
    Create2Case {
        report_id: 82,
        scenario: 11,
        name: "Create a single relationship and an end node on an existing starting node",
        setup_query: Some("CREATE (:Begin)"),
        setup_effects: Effects {
            nodes: 1,
            labels: 1,
            ..NO_EFFECTS
        },
        query: "MATCH (x:Begin) CREATE (x)-[:TYPE]->(:End)",
        controls: &[ControlCase {
            query: "MATCH (x:Begin)-[:TYPE]->(y:End) RETURN x, y",
            columns: &CONTROL_BEGIN_END,
        }],
        effects: Effects {
            labels: 1,
            ..ONE_NODE_ONE_EDGE
        },
        columns: &[],
        relationship: ExpectedRelationship {
            relationship_type: "TYPE",
            source: Endpoint::Label("Begin"),
            target: Endpoint::Label("End"),
            properties: &[],
        },
        read_entities: 1,
    },
    Create2Case {
        report_id: 83,
        scenario: 12,
        name: "Create a single relationship and a starting node on an existing end node",
        setup_query: Some("CREATE (:End)"),
        setup_effects: Effects {
            nodes: 1,
            labels: 1,
            ..NO_EFFECTS
        },
        query: "MATCH (x:End) CREATE (:Begin)-[:TYPE]->(x)",
        controls: &[ControlCase {
            query: "MATCH (x:Begin)-[:TYPE]->(y:End) RETURN x, y",
            columns: &CONTROL_BEGIN_END,
        }],
        effects: Effects {
            labels: 1,
            ..ONE_NODE_ONE_EDGE
        },
        columns: &[],
        relationship: ExpectedRelationship {
            relationship_type: "TYPE",
            source: Endpoint::Label("Begin"),
            target: Endpoint::Label("End"),
            properties: &[],
        },
        read_entities: 1,
    },
    Create2Case {
        report_id: 84,
        scenario: 13,
        name: "Create a single relationship with a property",
        setup_query: None,
        setup_effects: NO_EFFECTS,
        query: "CREATE ()-[:R {num: 42}]->()",
        controls: &[],
        effects: Effects {
            properties: 1,
            ..TWO_NODES_ONE_EDGE
        },
        columns: &[],
        relationship: ExpectedRelationship {
            relationship_type: "R",
            source: Endpoint::New(0),
            target: Endpoint::New(1),
            properties: &[("num", ExpectedScalar::Integer(42))],
        },
        read_entities: 0,
    },
    Create2Case {
        report_id: 85,
        scenario: 14,
        name: "Create a single relationship with a property and return it",
        setup_query: None,
        setup_effects: NO_EFFECTS,
        query: "CREATE ()-[r:R {num: 42}]->() RETURN r.num AS num",
        controls: &[],
        effects: Effects {
            properties: 1,
            ..TWO_NODES_ONE_EDGE
        },
        columns: &[ExpectedColumn {
            name: "num",
            value: ExpectedScalar::Integer(42),
        }],
        relationship: ExpectedRelationship {
            relationship_type: "R",
            source: Endpoint::New(0),
            target: Endpoint::New(1),
            properties: &[("num", ExpectedScalar::Integer(42))],
        },
        read_entities: 0,
    },
    Create2Case {
        report_id: 86,
        scenario: 15,
        name: "Create a single relationship with two properties",
        setup_query: None,
        setup_effects: NO_EFFECTS,
        query: "CREATE ()-[:R {id: 12, name: 'foo'}]->()",
        controls: &[],
        effects: Effects {
            properties: 2,
            ..TWO_NODES_ONE_EDGE
        },
        columns: &[],
        relationship: ExpectedRelationship {
            relationship_type: "R",
            source: Endpoint::New(0),
            target: Endpoint::New(1),
            properties: &[
                ("id", ExpectedScalar::Integer(12)),
                ("name", ExpectedScalar::String("foo")),
            ],
        },
        read_entities: 0,
    },
    Create2Case {
        report_id: 87,
        scenario: 16,
        name: "Create a single relationship with two properties and return them",
        setup_query: None,
        setup_effects: NO_EFFECTS,
        query: "CREATE ()-[r:R {id: 12, name: 'foo'}]->() RETURN r.id AS id, r.name AS name",
        controls: &[],
        effects: Effects {
            properties: 2,
            ..TWO_NODES_ONE_EDGE
        },
        columns: &[
            ExpectedColumn {
                name: "id",
                value: ExpectedScalar::Integer(12),
            },
            ExpectedColumn {
                name: "name",
                value: ExpectedScalar::String("foo"),
            },
        ],
        relationship: ExpectedRelationship {
            relationship_type: "R",
            source: Endpoint::New(0),
            target: Endpoint::New(1),
            properties: &[
                ("id", ExpectedScalar::Integer(12)),
                ("name", ExpectedScalar::String("foo")),
            ],
        },
        read_entities: 0,
    },
    Create2Case {
        report_id: 88,
        scenario: 17,
        name: "Create a single relationship with null properties should not return those properties",
        setup_query: None,
        setup_effects: NO_EFFECTS,
        query: "CREATE ()-[r:X {id: 12, name: null}]->() RETURN r.id, r.name AS name",
        controls: &[],
        effects: Effects {
            properties: 1,
            ..TWO_NODES_ONE_EDGE
        },
        columns: &[
            ExpectedColumn {
                name: "r.id",
                value: ExpectedScalar::Integer(12),
            },
            ExpectedColumn {
                name: "name",
                value: ExpectedScalar::Null,
            },
        ],
        relationship: ExpectedRelationship {
            relationship_type: "X",
            source: Endpoint::New(0),
            target: Endpoint::New(1),
            properties: &[("id", ExpectedScalar::Integer(12))],
        },
        read_entities: 0,
    },
];

#[derive(Debug, Deserialize)]
struct CertifiedReport {
    total: usize,
    cpu_passed: usize,
    metal_passed: usize,
    scenarios: Vec<CertifiedScenario>,
}

#[derive(Debug, Deserialize)]
struct CertifiedScenario {
    path: String,
    name: String,
    cpu_passed: bool,
    metal_passed: bool,
    fully_conformant: bool,
    operation_count: usize,
    shared_failures: Vec<String>,
    cpu_failures: Vec<String>,
    metal_failures: Vec<String>,
}

#[derive(Clone, Debug)]
struct SourceScenario {
    setup_queries: Vec<String>,
    query: String,
    control_queries: Vec<String>,
}

fn certified_report() -> Result<CertifiedReport> {
    let bytes = fs::read(REPORT_PATH).map_err(|error| {
        Error::internal(format!(
            "cannot read certified Create2 baseline {REPORT_PATH}: {error}"
        ))
    })?;
    serde_json::from_slice(&bytes)
        .map_err(|error| Error::internal(format!("cannot decode certified baseline: {error}")))
}

fn uniquely_resolved_scenario<'a>(
    report: &'a CertifiedReport,
    case: Create2Case,
) -> Result<(usize, &'a CertifiedScenario)> {
    let expanded_name = format!("[{}] {}", case.scenario, case.name);
    let matches = report
        .scenarios
        .iter()
        .enumerate()
        .filter(|(_, scenario)| scenario.path.ends_with(FEATURE) && scenario.name == expanded_name)
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(Error::internal(format!(
            "{} resolved to {} certified report entries instead of exactly one",
            case.label(),
            matches.len()
        )));
    }
    Ok(matches[0])
}

fn normalize_query(query: &str) -> String {
    query.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn scenario_block(source: &str, name: &str) -> Result<String> {
    let header = format!("Scenario: {name}");
    let start = source
        .find(&header)
        .ok_or_else(|| Error::internal(format!("pinned Create2 source omitted `{header}`")))?;
    let tail = &source[start..];
    let end = tail
        .lines()
        .skip(1)
        .scan(header.len() + 1, |offset, line| {
            let current = *offset;
            *offset += line.len() + 1;
            Some((current, line))
        })
        .find_map(|(offset, line)| line.trim_start().starts_with("Scenario:").then_some(offset))
        .unwrap_or(tail.len());
    Ok(tail[..end].to_owned())
}

fn docstrings_after(block: &str, marker: &str) -> Result<Vec<String>> {
    let mut values = Vec::new();
    let mut rest = block;
    while let Some(marker_offset) = rest.find(marker) {
        let tail = &rest[marker_offset + marker.len()..];
        let open = tail
            .find("\"\"\"")
            .ok_or_else(|| Error::internal(format!("`{marker}` omitted opening docstring")))?
            + 3;
        let close = tail[open..]
            .find("\"\"\"")
            .map(|offset| open + offset)
            .ok_or_else(|| Error::internal(format!("`{marker}` omitted closing docstring")))?;
        values.push(
            tail[open..close]
                .lines()
                .map(str::trim)
                .filter(|line| !line.is_empty())
                .collect::<Vec<_>>()
                .join(" "),
        );
        rest = &tail[close + 3..];
    }
    Ok(values)
}

fn source_scenario(case: Create2Case) -> Result<SourceScenario> {
    let path = Path::new(PINNED_FEATURE_ROOT).join(FEATURE);
    let source = fs::read_to_string(&path)
        .map_err(|error| Error::internal(format!("cannot read {}: {error}", path.display())))?;
    let block = scenario_block(&source, &format!("[{}] {}", case.scenario, case.name))?;
    let setup_queries = docstrings_after(&block, "having executed:")?;
    let query = docstrings_after(&block, "executing query:")?
        .into_iter()
        .next()
        .ok_or_else(|| Error::internal(format!("{} omitted its primary query", case.label())))?;
    let control_queries = docstrings_after(&block, "executing control query:")?;
    Ok(SourceScenario {
        setup_queries,
        query,
        control_queries,
    })
}

fn manifest_scenario(case: Create2Case) -> SourceScenario {
    SourceScenario {
        setup_queries: case
            .setup_query
            .into_iter()
            .map(str::to_owned)
            .collect::<Vec<_>>(),
        query: case.query.to_owned(),
        control_queries: case
            .controls
            .iter()
            .map(|control| control.query.to_owned())
            .collect::<Vec<_>>(),
    }
}

fn assert_literal_source(case: Create2Case, source: &SourceScenario) -> Result<()> {
    let expected_setup = case.setup_query.into_iter().collect::<Vec<_>>();
    let actual_setup = source
        .setup_queries
        .iter()
        .map(String::as_str)
        .collect::<Vec<_>>();
    if actual_setup != expected_setup
        || normalize_query(&source.query) != normalize_query(case.query)
        || source.control_queries.len() != case.controls.len()
        || source
            .control_queries
            .iter()
            .zip(case.controls)
            .any(|(actual, expected)| normalize_query(actual) != normalize_query(expected.query))
    {
        return Err(Error::internal(format!(
            "{} literal setup/primary/control manifest drifted from pinned source: {source:#?}",
            case.label()
        )));
    }
    Ok(())
}

fn failure_query(failure: &str) -> Result<String> {
    let start = failure
        .find('`')
        .ok_or_else(|| Error::internal("certified Create2 failure omitted query delimiter"))?
        + 1;
    let end = failure[start..]
        .find("`:")
        .map(|offset| start + offset)
        .ok_or_else(|| Error::internal("certified Create2 failure omitted closing delimiter"))?;
    Ok(failure[start..end].to_owned())
}

fn bookmark(graph: &GraphStore) -> Bookmark {
    Bookmark {
        term: 61,
        index: graph.revision(),
    }
}

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
        bookmark: bookmark(graph),
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
        max_batch_rows: MAX_RESULT_ROWS,
        optimizer_statistics: None,
        backend,
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(60)),
        resolved_query_at_time_nanos: None,
    }
}

fn resident_image(graph: &GraphStore) -> Result<ResidentProjectImage> {
    ResidentProjectImage::build(
        PROJECT,
        bookmark(graph),
        graph,
        &TemporalStore::default(),
        &IndexCatalog::default(),
    )
}

fn apply_mutations(graph: &mut GraphStore, mutations: &[GraphMutation]) -> Result<()> {
    for mutation in mutations {
        graph.apply(mutation.clone())?;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct GraphMetrics {
    nodes: usize,
    relationships: usize,
    properties: usize,
    labels: usize,
}

fn graph_metrics(graph: &GraphStore) -> GraphMetrics {
    GraphMetrics {
        nodes: graph.node_count(),
        relationships: graph.edge_count(),
        properties: graph
            .nodes()
            .map(|node| node.properties().len())
            .sum::<usize>()
            + graph
                .edges()
                .map(|edge| edge.properties().len())
                .sum::<usize>(),
        labels: graph.nodes().map(|node| node.labels().len()).sum(),
    }
}

fn added_effects(before: GraphMetrics, after: GraphMetrics) -> Result<Effects> {
    if after.nodes < before.nodes
        || after.relationships < before.relationships
        || after.properties < before.properties
        || after.labels < before.labels
    {
        return Err(Error::internal(
            "Create2 operation removed canonical graph content",
        ));
    }
    Ok(Effects {
        nodes: after.nodes - before.nodes,
        relationships: after.relationships - before.relationships,
        properties: after.properties - before.properties,
        labels: after.labels - before.labels,
    })
}

fn assert_common_output(output: &ExecutionOutput, label: &str) -> Result<()> {
    if output.result.truncated
        || !output.temporal_mutations.is_empty()
        || output.administrative.is_some()
        || !output.vector_searches.is_empty()
        || output.runtime_replans != 0
    {
        return Err(Error::internal(format!(
            "{label}: CREATE emitted unrelated or truncated execution state"
        )));
    }
    if output.graph_mutations.iter().any(|mutation| {
        !matches!(
            mutation,
            GraphMutation::DeclareLabel { .. }
                | GraphMutation::DeclareProperty { .. }
                | GraphMutation::DeclareRelationshipType { .. }
                | GraphMutation::InsertNode(_)
                | GraphMutation::InsertEdge(_)
        )
    }) {
        return Err(Error::internal(format!(
            "{label}: CREATE emitted a non-create graph mutation: {:?}",
            output.graph_mutations
        )));
    }
    if !output.dependencies.write_targets.is_empty() {
        return Err(Error::internal(format!(
            "{label}: newly allocated entities leaked into pre-existing write targets: {:?}",
            output.dependencies.write_targets
        )));
    }
    Ok(())
}

fn assert_scalar_result(
    output: &ExecutionOutput,
    columns: &[ExpectedColumn],
    label: &str,
) -> Result<()> {
    let expected_schema = columns
        .iter()
        .map(|column| (column.name.to_owned(), column.value.column_type()))
        .collect::<Vec<_>>();
    if output.result.schema != expected_schema {
        return Err(Error::internal(format!(
            "{label}: result schema mismatch: expected {expected_schema:?}, got {:?}",
            output.result.schema
        )));
    }
    if columns.is_empty() {
        if !output.result.batches.is_empty() {
            return Err(Error::internal(format!(
                "{label}: CREATE without RETURN emitted rows"
            )));
        }
        return Ok(());
    }
    if output.result.batches.len() != 1
        || output.result.batches[0].row_count != 1
        || output.result.batches[0].columns.len() != columns.len()
        || !output.result.batches[0].validate()
    {
        return Err(Error::internal(format!(
            "{label}: statement-local relationship RETURN has wrong shape: {:?}",
            output.result.batches
        )));
    }
    for (actual, expected) in output.result.batches[0].columns.iter().zip(columns) {
        let expected_values = vec![ResultValue::Scalar(expected.value.value())];
        if actual.name != expected.name
            || actual.value_type != expected.value.column_type()
            || actual.values != expected_values
        {
            return Err(Error::internal(format!(
                "{label}: returned relationship property differs: expected {expected:?}, got {actual:?}"
            )));
        }
    }
    Ok(())
}

fn resolve_endpoint(
    graph: &GraphStore,
    new_nodes: &[NodeId],
    endpoint: Endpoint,
) -> Result<NodeId> {
    match endpoint {
        Endpoint::New(index) => new_nodes.get(index).copied().ok_or_else(|| {
            Error::internal(format!("Create2 omitted new endpoint ordinal {index}"))
        }),
        Endpoint::Label(name) => {
            let label = graph
                .catalog()
                .label(name)
                .ok_or_else(|| Error::internal(format!("Create2 omitted endpoint label {name}")))?;
            let nodes = graph
                .nodes()
                .filter(|node| node.labels().contains(&label))
                .map(|node| node.id())
                .collect::<Vec<_>>();
            if nodes.len() != 1 {
                return Err(Error::internal(format!(
                    "Create2 endpoint label {name} resolved to {nodes:?}"
                )));
            }
            Ok(nodes[0])
        }
    }
}

fn assert_relationship_shape(
    before: &GraphStore,
    after: &GraphStore,
    expected: ExpectedRelationship,
    label: &str,
) -> Result<()> {
    let before_nodes = before
        .nodes()
        .map(|node| node.id())
        .collect::<BTreeSet<_>>();
    let mut new_nodes = after
        .nodes()
        .map(|node| node.id())
        .filter(|node| !before_nodes.contains(node))
        .collect::<Vec<_>>();
    new_nodes.sort();
    let edges = after.edges().collect::<Vec<_>>();
    if edges.len() != 1 {
        return Err(Error::internal(format!(
            "{label}: expected exactly one committed relationship, got {}",
            edges.len()
        )));
    }
    let edge = edges[0];
    let source = resolve_endpoint(after, &new_nodes, expected.source)?;
    let target = resolve_endpoint(after, &new_nodes, expected.target)?;
    let relationship_type = after
        .catalog()
        .relationship_type_name(edge.relationship_type())
        .ok_or_else(|| Error::internal("created relationship type is undeclared"))?;
    if edge.source() != source
        || edge.target() != target
        || relationship_type != expected.relationship_type
        || edge.layer() != Layer::Observed
    {
        return Err(Error::internal(format!(
            "{label}: wrong relationship direction/type/layer: {edge:?}, expected {source:?}-[:{}]->{target:?}",
            expected.relationship_type
        )));
    }
    let actual_properties =
        edge.properties()
            .into_iter()
            .map(|(property, value)| {
                let name = after.catalog().property_name(property).ok_or_else(|| {
                    Error::internal("created relationship property is undeclared")
                })?;
                Ok((name.to_owned(), value))
            })
            .collect::<Result<BTreeMap<_, _>>>()?;
    let expected_properties = expected
        .properties
        .iter()
        .map(|(name, value)| ((*name).to_owned(), value.value()))
        .collect::<BTreeMap<_, _>>();
    if actual_properties != expected_properties
        || actual_properties
            .values()
            .any(|value| matches!(value, ScalarValue::Null))
    {
        return Err(Error::internal(format!(
            "{label}: relationship property payload mismatch: expected {expected_properties:?}, got {actual_properties:?}"
        )));
    }
    Ok(())
}

fn assert_primary_output(
    case: Create2Case,
    before: &GraphStore,
    output: &ExecutionOutput,
) -> Result<GraphStore> {
    let label = case.label();
    assert_common_output(output, &label)?;
    assert_scalar_result(output, case.columns, &label)?;
    if output.result.statistics != case.effects.statistics() {
        return Err(Error::internal(format!(
            "{label}: statement statistics mismatch: expected {:?}, got {:?}",
            case.effects.statistics(),
            output.result.statistics
        )));
    }
    if output.dependencies.entities.len() != case.read_entities {
        return Err(Error::internal(format!(
            "{label}: expected {} exact existing-entity dependencies, got {:?}",
            case.read_entities, output.dependencies.entities
        )));
    }
    for (entity, revision) in &output.dependencies.entities {
        let current = match entity {
            irongraph::cypher::EntityDependency::Node(node) => {
                before.node(*node).map(|node| node.revision())
            }
            irongraph::cypher::EntityDependency::Relationship(edge) => {
                before.edge(*edge).map(|edge| edge.revision())
            }
        };
        if current != Some(*revision) {
            return Err(Error::internal(format!(
                "{label}: dependency {entity:?}@{revision} is not from the input generation"
            )));
        }
    }
    let mut after = before.clone();
    apply_mutations(&mut after, &output.graph_mutations)?;
    let observed = added_effects(graph_metrics(before), graph_metrics(&after))?;
    if observed != case.effects {
        return Err(Error::internal(format!(
            "{label}: graph side effects mismatch: expected {:?}, got {observed:?}",
            case.effects
        )));
    }
    assert_relationship_shape(before, &after, case.relationship, &label)?;
    Ok(after)
}

fn assert_control(graph: &GraphStore, control: ControlCase, label: &str) -> Result<()> {
    let output = QueryEngine.execute(control.query, &mut context(graph, None, false))?;
    assert_common_output(&output, label)?;
    if !output.graph_mutations.is_empty()
        || output.result.statistics != StatementStats::default()
        || output.result.schema
            != control
                .columns
                .iter()
                .map(|column| (column.name.to_owned(), ColumnType::Node))
                .collect::<Vec<_>>()
        || output.result.batches.len() != 1
        || output.result.batches[0].row_count != 1
        || output.result.batches[0].columns.len() != control.columns.len()
        || !output.result.batches[0].validate()
    {
        return Err(Error::internal(format!(
            "{label}: control query shape/side effects changed: {output:#?}"
        )));
    }
    for (actual, expected) in output.result.batches[0].columns.iter().zip(control.columns) {
        let [ResultValue::Node(node)] = actual.values.as_slice() else {
            return Err(Error::internal(format!(
                "{label}: control column {} did not return one node: {actual:?}",
                expected.name
            )));
        };
        let mut labels = node.labels.clone();
        labels.sort();
        let mut expected_labels = expected
            .labels
            .iter()
            .map(|name| (*name).to_owned())
            .collect::<Vec<_>>();
        expected_labels.sort();
        if actual.name != expected.name || labels != expected_labels {
            return Err(Error::internal(format!(
                "{label}: control endpoint mismatch: expected {expected:?}, got {actual:?}"
            )));
        }
    }
    Ok(())
}

fn fixture_graph(case: Create2Case, source: &SourceScenario) -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    for setup in &source.setup_queries {
        let before = graph_metrics(&graph);
        let output = QueryEngine.execute(setup, &mut context(&graph, None, false))?;
        assert_common_output(&output, &format!("{} setup", case.label()))?;
        if !output.result.schema.is_empty()
            || !output.result.batches.is_empty()
            || output.result.statistics != case.setup_effects.statistics()
        {
            return Err(Error::internal(format!(
                "{} setup result/statistics mismatch: {output:#?}",
                case.label()
            )));
        }
        apply_mutations(&mut graph, &output.graph_mutations)?;
        let observed = added_effects(before, graph_metrics(&graph))?;
        if observed != case.setup_effects {
            return Err(Error::internal(format!(
                "{} setup side effects mismatch: expected {:?}, got {observed:?}",
                case.label(),
                case.setup_effects
            )));
        }
    }
    Ok(graph)
}

#[derive(Default)]
struct RouteObservations {
    pins: AtomicUsize,
    complete_graph_write_calls: AtomicUsize,
    rejected_legacy_routes: AtomicUsize,
    generation_mutation_attempts: AtomicUsize,
}

/// Fail-closed observer for the generalized relationship-CREATE command boundary.
///
/// Only one pinned, complete row-CREATE request may cross this boundary. Write-only graph-free
/// CREATE may use the reusable ordered bound body; all other statements use the dedicated CREATE
/// body. The current node-only CREATE call and every decomposed graph/row primitive remain
/// forbidden.
struct StrictRelationshipCreateBackend {
    inner: Box<dyn ExecutionBackend>,
    advertised_kind: BackendKind,
    pinned_kind: BackendKind,
    actual_kind: BackendKind,
    pinned: bool,
    expected_bookmark: Bookmark,
    expected_graph_revision: u64,
    observations: Arc<RouteObservations>,
}

fn is_exact_graph_free_bound_create(request: &ResidentRowMutationRequest) -> bool {
    let Some(body) = request.bound_relationship_merge_body() else {
        return false;
    };
    if !request.input.is_empty()
        || body.program.commands.is_empty()
        || usize::from(body.program.entity_slot_count) != body.program.commands.len()
        || !body.program.input_keys.is_empty()
        || !body.program.outputs.is_empty()
        || body.program.continuation.is_some()
        || !body.outputs.is_empty()
        || !body.merge_property_sets.is_empty()
        || !body.merge_label_sets.is_empty()
        || body.schedule.value_slot_count != 0
        || !body.schedule.property_bindings.is_empty()
        || body.schedule.stages.len() != body.program.commands.len()
        || body.commands.len() != body.program.commands.len()
        || body.command_candidate_limits.len() != body.program.commands.len()
    {
        return false;
    }

    let mut nodes = BTreeSet::new();
    let mut relationships = 0_usize;
    for (index, (((structural, semantic), limit), stage)) in body
        .program
        .commands
        .iter()
        .zip(&body.commands)
        .zip(&body.command_candidate_limits)
        .zip(&body.schedule.stages)
        .enumerate()
    {
        let Ok(output_entity) = u16::try_from(index) else {
            return false;
        };
        if *limit != 1
            || !matches!(
                stage,
                ResidentRowBoundRelationshipStage::Command { command }
                    if *command == output_entity
            )
        {
            return false;
        }
        let properties = match (structural, semantic) {
            (
                ResidentRowCreateCommand::CreateNode(command),
                ResidentRowBoundRelationshipCommand::CreateNode,
            ) if command.output_entity == output_entity => {
                nodes.insert(output_entity);
                &command.properties
            }
            (
                ResidentRowCreateCommand::CreateRelationship(command),
                ResidentRowBoundRelationshipCommand::CreateRelationship,
            ) if command.output_entity == output_entity
                && nodes.contains(&command.source_entity)
                && nodes.contains(&command.target_entity) =>
            {
                relationships += 1;
                &command.properties
            }
            _ => return false,
        };
        if properties.iter().any(|property| {
            !matches!(
                &property.value,
                ResidentRowCreateValueInput::Constant(ResidentCreateNodeValueInput::Scalar(_))
            )
        }) {
            return false;
        }
    }
    relationships != 0
}

impl StrictRelationshipCreateBackend {
    fn strict_cpu(image: ResidentProjectImage) -> Result<Self> {
        let mut cpu = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
        cpu.admit_project(image)?;
        Self::new(
            Box::new(cpu),
            BackendKind::Metal,
            BackendKind::Cpu,
            BackendKind::Cpu,
        )
    }

    #[cfg(all(feature = "accelerator", target_os = "macos"))]
    fn real_metal(image: ResidentProjectImage) -> Result<Self> {
        let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
        metal.admit_project(image)?;
        Self::new(
            Box::new(metal),
            BackendKind::Metal,
            BackendKind::Metal,
            BackendKind::Metal,
        )
    }

    fn new(
        inner: Box<dyn ExecutionBackend>,
        advertised_kind: BackendKind,
        pinned_kind: BackendKind,
        actual_kind: BackendKind,
    ) -> Result<Self> {
        let expected_bookmark = inner.resident_bookmark(PROJECT).ok_or_else(|| {
            Error::internal("strict Create2 backend has no admitted project bookmark")
        })?;
        let expected_graph_revision = inner.resident_graph_revision(PROJECT).ok_or_else(|| {
            Error::internal("strict Create2 backend has no admitted graph generation")
        })?;
        Ok(Self {
            inner,
            advertised_kind,
            pinned_kind,
            actual_kind,
            pinned: false,
            expected_bookmark,
            expected_graph_revision,
            observations: Arc::new(RouteObservations::default()),
        })
    }

    fn observations(&self) -> Arc<RouteObservations> {
        Arc::clone(&self.observations)
    }

    fn refresh_fence(&mut self) -> Result<()> {
        self.expected_bookmark = self
            .inner
            .resident_bookmark(PROJECT)
            .ok_or_else(|| Error::internal("Create2 publication lost resident bookmark"))?;
        self.expected_graph_revision = self
            .inner
            .resident_graph_revision(PROJECT)
            .ok_or_else(|| Error::internal("Create2 publication lost graph generation"))?;
        Ok(())
    }

    fn reject_legacy<T>(&self, route: &'static str) -> Result<T> {
        self.observations
            .rejected_legacy_routes
            .fetch_add(1, Ordering::SeqCst);
        Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!(
                "strict Create2 observer rejected legacy/incomplete `{route}`; exactly one complete generation-fenced graph-write command is required"
            ),
        ))
    }

    fn reject_generation_mutation<T>(&self, route: &'static str) -> Result<T> {
        self.observations
            .generation_mutation_attempts
            .fetch_add(1, Ordering::SeqCst);
        Err(Error::new(
            ErrorCode::CorruptStorage,
            format!("pinned Create2 generation rejected `{route}` mutation"),
        ))
    }
}

impl ExecutionBackend for StrictRelationshipCreateBackend {
    fn kind(&self) -> BackendKind {
        if self.pinned {
            self.pinned_kind
        } else {
            self.advertised_kind
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
        if self.pinned {
            return self.reject_legacy("pin_project_twice");
        }
        if project != PROJECT {
            return self.reject_legacy("pin_wrong_project");
        }
        let inner = self.inner.pin_project(project)?;
        if inner.kind() != self.actual_kind
            || inner.resident_bookmark(project) != Some(self.expected_bookmark)
            || inner.resident_graph_revision(project) != Some(self.expected_graph_revision)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "Create2 pin changed backend provenance or immutable generation fence",
            ));
        }
        self.observations.pins.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Self {
            inner,
            advertised_kind: self.advertised_kind,
            pinned_kind: self.pinned_kind,
            actual_kind: self.actual_kind,
            pinned: true,
            expected_bookmark: self.expected_bookmark,
            expected_graph_revision: self.expected_graph_revision,
            observations: Arc::clone(&self.observations),
        }))
    }

    fn admit_project(&mut self, image: ResidentProjectImage) -> Result<()> {
        if self.pinned {
            return self.reject_generation_mutation("admit_project");
        }
        self.inner.admit_project(image)?;
        self.refresh_fence()
    }

    fn replace_all_projects(&mut self, images: Vec<ResidentProjectImage>) -> Result<()> {
        if self.pinned {
            return self.reject_generation_mutation("replace_all_projects");
        }
        self.inner.replace_all_projects(images)?;
        self.refresh_fence()
    }

    fn apply_project_delta(&mut self, delta: ResidentProjectDelta) -> Result<()> {
        if self.pinned {
            return self.reject_generation_mutation("apply_project_delta");
        }
        self.inner.apply_project_delta(delta)?;
        self.refresh_fence()
    }

    fn evict_project(&mut self, project: ProjectId) -> Result<()> {
        if self.pinned {
            return self.reject_generation_mutation("evict_project");
        }
        self.inner.evict_project(project)
    }

    fn advance_bookmark(&mut self, bookmark: Bookmark) {
        if self.pinned {
            self.observations
                .generation_mutation_attempts
                .fetch_add(1, Ordering::SeqCst);
        } else {
            self.inner.advance_bookmark(bookmark);
        }
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
        self.reject_legacy("scan_nodes")
    }

    fn filter_node_i64(
        &self,
        _project: ProjectId,
        _property: PropertyId,
        _operation: CompareOp,
        _operand: i64,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.reject_legacy("filter_node_i64")
    }

    fn expand_project_out(
        &self,
        _project: ProjectId,
        _sources: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.reject_legacy("expand_project_out")
    }

    fn expand_project_in(
        &self,
        _project: ProjectId,
        _targets: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.reject_legacy("expand_project_in")
    }

    fn search_vectors(
        &self,
        _request: &ResidentVectorQuery,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentVectorResult> {
        self.reject_legacy("search_vectors")
    }

    fn sort_rows(
        &self,
        _request: &ResidentSortRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentSortResult> {
        self.reject_legacy("sort_rows")
    }

    fn join_node_i64(
        &self,
        _request: &ResidentJoinRequest,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentJoinPair>> {
        self.reject_legacy("join_node_i64")
    }

    fn group_node_i64(
        &self,
        _request: &ResidentGroupRequest,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentGroup>> {
        self.reject_legacy("group_node_i64")
    }

    fn execute_node_pipeline(
        &self,
        _request: &ResidentNodePipelineRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentNodePipelineResult> {
        self.reject_legacy("execute_node_pipeline")
    }

    fn execute_row_mutation(
        &self,
        request: &ResidentRowMutationRequest,
        cancellation: &CancellationToken,
    ) -> Result<ResidentRowMutationResult> {
        if !self.pinned {
            return self.reject_legacy("execute_row_mutation_on_unpinned_generation");
        }
        request.validate()?;
        if request.create_body().is_none() && !is_exact_graph_free_bound_create(request) {
            return self.reject_legacy("execute_row_mutation_without_complete_create_body");
        }
        if self.inner.kind() != self.actual_kind
            || request.generation.project != PROJECT
            || request.generation.bookmark != self.expected_bookmark
            || request.generation.graph_revision != self.expected_graph_revision
            || self.inner.resident_bookmark(PROJECT) != Some(self.expected_bookmark)
            || self.inner.resident_graph_revision(PROJECT) != Some(self.expected_graph_revision)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "generalized Create2 row mutation escaped its pinned backend or immutable generation",
            ));
        }
        if self
            .observations
            .complete_graph_write_calls
            .compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return self.reject_legacy("execute_row_mutation_more_than_once");
        }
        self.inner.execute_row_mutation(request, cancellation)
    }

    fn supports_native_create_node(&self) -> bool {
        false
    }

    fn execute_create_node(
        &self,
        _request: &ResidentCreateNodeRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentCreateNodeResult> {
        self.reject_legacy("execute_create_node_is_graph_free_and_incomplete")
    }

    fn execute_row_program(
        &self,
        _request: &ResidentRowProgramRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentRowProgramResult> {
        self.reject_legacy("execute_row_program")
    }

    fn execute_temporal_pipeline(
        &self,
        _request: &ResidentTemporalPipelineRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentTemporalPipelineResult> {
        self.reject_legacy("execute_temporal_pipeline")
    }

    fn filter_i64(
        &self,
        _values: &[i64],
        _validity: &[bool],
        _operation: CompareOp,
        _operand: i64,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.reject_legacy("filter_i64")
    }

    fn expand_out(
        &self,
        _sources: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.reject_legacy("expand_out")
    }

    fn exact_l2(
        &self,
        _matrix: &[f32],
        _rows: usize,
        _dimension: usize,
        _query: &[f32],
        _cancellation: &CancellationToken,
    ) -> Result<DistanceBatch> {
        self.reject_legacy("exact_l2")
    }
}

fn run_generic_case(case: Create2Case) -> Result<()> {
    let source = manifest_scenario(case);
    let graph = fixture_graph(case, &source)?;
    let output = QueryEngine.execute(&source.query, &mut context(&graph, None, false))?;
    let committed = assert_primary_output(case, &graph, &output)?;
    for (index, control) in case.controls.iter().copied().enumerate() {
        assert_control(
            &committed,
            control,
            &format!("{} control {}", case.label(), index + 1),
        )?;
    }
    Ok(())
}

fn strict_case_failure(
    case: Create2Case,
    error: Error,
    observations: &RouteObservations,
) -> String {
    let counts = format!(
        "pins={}, complete_graph_write_calls={}, rejected_legacy_routes={}, generation_mutation_attempts={}",
        observations.pins.load(Ordering::SeqCst),
        observations
            .complete_graph_write_calls
            .load(Ordering::SeqCst),
        observations.rejected_legacy_routes.load(Ordering::SeqCst),
        observations
            .generation_mutation_attempts
            .load(Ordering::SeqCst),
    );
    if error.code == ErrorCode::GpuAdmissionFailure
        && error.message.contains(MISSING_GRAPH_WRITE_CONTRACT)
    {
        format!(
            "{}: generalized CREATE row mutation was not admitted ({counts})",
            case.label()
        )
    } else {
        format!(
            "{}: unexpected strict-route failure `{error}` ({counts})",
            case.label()
        )
    }
}

fn run_strict_case(
    case: Create2Case,
    backend: StrictRelationshipCreateBackend,
    graph: &GraphStore,
) -> Result<()> {
    let observations = backend.observations();
    let output = match QueryEngine.execute(case.query, &mut context(graph, Some(&backend), true)) {
        Ok(output) => output,
        Err(error) => {
            return Err(Error::new(
                error.code,
                strict_case_failure(case, error, &observations),
            ));
        }
    };
    let pins = observations.pins.load(Ordering::SeqCst);
    let graph_writes = observations
        .complete_graph_write_calls
        .load(Ordering::SeqCst);
    let rejected = observations.rejected_legacy_routes.load(Ordering::SeqCst);
    let mutations = observations
        .generation_mutation_attempts
        .load(Ordering::SeqCst);
    if pins != 1 || graph_writes != 1 || rejected != 0 || mutations != 0 {
        return Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!(
                "{}: strict route requires pins=1, complete graph writes=1, legacy routes=0, generation mutations=0; got {pins}, {graph_writes}, {rejected}, {mutations}",
                case.label()
            ),
        ));
    }
    let committed = assert_primary_output(case, graph, &output)?;
    for (index, control) in case.controls.iter().copied().enumerate() {
        assert_control(
            &committed,
            control,
            &format!("{} post-native control {}", case.label(), index + 1),
        )?;
    }
    Ok(())
}

fn run_strict_cpu_suite() -> Result<()> {
    let mut failures = Vec::new();
    for case in CASES {
        let result = (|| {
            let source = manifest_scenario(case);
            let graph = fixture_graph(case, &source)?;
            let backend = StrictRelationshipCreateBackend::strict_cpu(resident_image(&graph)?)?;
            run_strict_case(case, backend, &graph)
        })();
        if let Err(error) = result {
            failures.push(error.message);
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!(
                "strict CPU Create2 acceptance failed for {} case(s):\n{}",
                failures.len(),
                failures.join("\n")
            ),
        ))
    }
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn run_real_metal_suite() -> Result<()> {
    let mut failures = Vec::new();
    for case in CASES {
        let result = (|| {
            let source = manifest_scenario(case);
            let graph = fixture_graph(case, &source)?;
            let backend = StrictRelationshipCreateBackend::real_metal(resident_image(&graph)?)?;
            run_strict_case(case, backend, &graph)
        })();
        if let Err(error) = result {
            failures.push(error.message);
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!(
                "strict Metal Create2 acceptance remains red for {} case(s):\n{}",
                failures.len(),
                failures.join("\n")
            ),
        ))
    }
}

#[test]
fn literal_manifest_is_exactly_zero_based_report_ids_72_through_88() {
    assert_eq!(CASES.len(), 17);
    assert_eq!(
        CASES.iter().map(|case| case.report_id).collect::<Vec<_>>(),
        (72_usize..=88).collect::<Vec<_>>()
    );
    assert_eq!(
        CASES.iter().map(|case| case.scenario).collect::<Vec<_>>(),
        (1_u8..=17).collect::<Vec<_>>()
    );
}

#[test]
#[ignore = "external acceptance: requires the pinned Create2 feature and certified 3,897-scenario report"]
fn literal_manifest_matches_pinned_source_and_certified_baseline() -> Result<()> {
    let report = certified_report()?;
    assert_eq!(report.total, 3_897);
    assert_eq!(report.cpu_passed, 3_897);
    assert_eq!(report.metal_passed, 3_184);
    for case in CASES {
        let source = source_scenario(case)?;
        assert_literal_source(case, &source)?;
        let (report_index, certified) = uniquely_resolved_scenario(&report, case)?;
        assert_eq!(
            report_index,
            case.report_id,
            "{} resolved at the wrong zero-based report array index",
            case.label()
        );
        assert!(certified.cpu_passed, "{} is not CPU-green", case.label());
        assert!(
            !certified.metal_passed,
            "{} is already Metal-green",
            case.label()
        );
        assert!(
            !certified.fully_conformant,
            "{} is already conformant",
            case.label()
        );
        assert!(certified.shared_failures.is_empty(), "{}", case.label());
        assert!(certified.cpu_failures.is_empty(), "{}", case.label());
        assert_eq!(
            certified.operation_count,
            1 + case.controls.len(),
            "{}",
            case.label()
        );
        assert_eq!(
            certified.metal_failures.len(),
            1 + case.controls.len(),
            "{}",
            case.label()
        );
        assert!(
            certified
                .metal_failures
                .iter()
                .all(|failure| failure.contains("GpuAdmissionFailure")
                    && failure.contains(MISSING_GRAPH_WRITE_CONTRACT)),
            "{} has a non-admission certified failure: {:?}",
            case.label(),
            certified.metal_failures
        );
        let expected_queries = std::iter::once(case.query)
            .chain(case.controls.iter().map(|control| control.query))
            .map(normalize_query)
            .collect::<Vec<_>>();
        let actual_queries = certified
            .metal_failures
            .iter()
            .map(|failure| failure_query(failure).map(|query| normalize_query(&query)))
            .collect::<Result<Vec<_>>>()?;
        assert_eq!(actual_queries, expected_queries, "{}", case.label());
    }
    Ok(())
}

#[test]
fn generic_cpu_oracle_proves_all_17_setup_primary_control_and_side_effects() -> Result<()> {
    let mut failures = Vec::new();
    for case in CASES {
        if let Err(error) = run_generic_case(case) {
            failures.push(format!("{}: {error}", case.label()));
        }
    }
    assert!(
        failures.is_empty(),
        "generic CPU Create2 oracle had {} failure(s):\n{}",
        failures.len(),
        failures.join("\n")
    );
    Ok(())
}

#[test]
fn observer_rejects_callable_legacy_scan_and_pinned_generation_replacement() -> Result<()> {
    let graph = GraphStore::default();
    let backend = StrictRelationshipCreateBackend::strict_cpu(resident_image(&graph)?)?;
    let observations = backend.observations();
    let mut pinned = backend.pin_project(PROJECT)?;
    assert_eq!(pinned.kind(), BackendKind::Cpu);
    let scan_error = pinned
        .scan_nodes(PROJECT, None, LayerMask::ALL, &CancellationToken::new())
        .expect_err("legacy scan unexpectedly entered strict Create2 route");
    assert_eq!(scan_error.code, ErrorCode::GpuAdmissionFailure);
    let mutation_error = pinned
        .replace_all_projects(vec![resident_image(&graph)?])
        .expect_err("pinned Create2 generation was replaceable");
    assert_eq!(mutation_error.code, ErrorCode::CorruptStorage);
    assert_eq!(observations.pins.load(Ordering::SeqCst), 1);
    assert_eq!(
        observations
            .complete_graph_write_calls
            .load(Ordering::SeqCst),
        0
    );
    assert_eq!(
        observations.rejected_legacy_routes.load(Ordering::SeqCst),
        1
    );
    assert_eq!(
        observations
            .generation_mutation_attempts
            .load(Ordering::SeqCst),
        1
    );
    Ok(())
}

#[test]
fn strict_cpu_reference_executes_all_17_through_generalized_create_row_mutation() -> Result<()> {
    run_strict_cpu_suite()
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
#[ignore = "requires an available physical Metal device"]
fn real_metal_executes_all_17_as_one_native_graph_write_command_without_fallback() -> Result<()> {
    let _guard = metal_test_guard();
    run_real_metal_suite()
}
