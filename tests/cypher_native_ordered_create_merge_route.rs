// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Baseline manifest for the 27 remaining Metal-only ordered CREATE/MERGE interoperation gaps.
//!
//! This file deliberately pins report identity and tranche ownership before the shared mutation
//! ABI is changed. It does not treat report position as semantic evidence: feature, scenario name,
//! and exact primary query must all agree with the fresh full-conformance report.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use irongraph::{
    Bookmark, Error, ErrorCode, Layer, ProjectId, Result, ScalarValue,
    cypher::{
        BindCapabilities, ColumnType, ExecutionContext, ExecutionOutput, QueryEngine, ResultValue,
        StatementStats,
    },
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend,
        ResidentCreateNodeRequest, ResidentCreateNodeResult, ResidentCreateNodeValueInput,
        ResidentGroup, ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest,
        ResidentNodePipelineRequest, ResidentNodePipelineResult, ResidentObligationKind,
        ResidentObligationScope, ResidentProjectImage, ResidentRowBoundMergeBranch,
        ResidentRowBoundMergePropertySource, ResidentRowBoundPropertyBindingSource,
        ResidentRowBoundRelationshipCommand, ResidentRowBoundRelationshipDirection,
        ResidentRowBoundRelationshipOutput, ResidentRowBoundRelationshipStage,
        ResidentRowCreateCommand, ResidentRowCreateValueInput, ResidentRowMutationRequest,
        ResidentRowMutationResult, ResidentRowProgramRequest, ResidentRowProgramResult,
        ResidentSortRequest, ResidentSortResult, ResidentVectorQuery, ResidentVectorResult,
        ScratchReservation,
    },
    graph::{GraphMutation, GraphStore, IndexCatalog, LayerMask, TemporalStore},
    types::{LabelId, PropertyId},
};

use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::gpu::MetalBackend;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Tranche {
    OrderedCreate,
    ConstantNodeMerge,
    DynamicCandidates,
}

#[derive(Clone, Copy, Debug)]
#[allow(dead_code)]
struct Case {
    report_index: usize,
    feature: &'static str,
    scenario: u8,
    name: &'static str,
    query: &'static str,
    tranche: Tranche,
}

const CASES: [Case; 27] = [
    Case {
        report_index: 96,
        feature: "clauses/create/Create3.feature",
        scenario: 1,
        name: "[1] MATCH-CREATE",
        query: "MATCH () CREATE ()",
        tranche: Tranche::OrderedCreate,
    },
    Case {
        report_index: 97,
        feature: "clauses/create/Create3.feature",
        scenario: 2,
        name: "[2] WITH-CREATE",
        query: "MATCH () CREATE () WITH * CREATE ()",
        tranche: Tranche::OrderedCreate,
    },
    Case {
        report_index: 98,
        feature: "clauses/create/Create3.feature",
        scenario: 3,
        name: "[3] MATCH-CREATE-WITH-CREATE",
        query: "MATCH () CREATE () WITH * MATCH () CREATE ()",
        tranche: Tranche::DynamicCandidates,
    },
    Case {
        report_index: 99,
        feature: "clauses/create/Create3.feature",
        scenario: 4,
        name: "[4] MATCH-CREATE: Newly-created nodes not visible to preceding MATCH",
        query: "MATCH () CREATE ()",
        tranche: Tranche::OrderedCreate,
    },
    Case {
        report_index: 100,
        feature: "clauses/create/Create3.feature",
        scenario: 5,
        name: "[5] WITH-CREATE: Nodes are not created when aliases are applied to variable names",
        query: "MATCH (n) MATCH (m) WITH n AS a, m AS b CREATE (a)-[:T]->(b) RETURN a, b",
        tranche: Tranche::OrderedCreate,
    },
    Case {
        report_index: 101,
        feature: "clauses/create/Create3.feature",
        scenario: 6,
        name: "[6] WITH-CREATE: Only a single node is created when an alias is applied to a variable name",
        query: "MATCH (n) WITH n AS a CREATE (a)-[:T]->() RETURN a",
        tranche: Tranche::OrderedCreate,
    },
    Case {
        report_index: 102,
        feature: "clauses/create/Create3.feature",
        scenario: 7,
        name: "[7] WITH-CREATE: Nodes are not created when aliases are applied to variable names multiple times",
        query: "MATCH (n) MATCH (m) WITH n AS a, m AS b CREATE (a)-[:T]->(b) WITH a AS x, b AS y CREATE (x)-[:T]->(y) RETURN x, y",
        tranche: Tranche::OrderedCreate,
    },
    Case {
        report_index: 103,
        feature: "clauses/create/Create3.feature",
        scenario: 8,
        name: "[8] WITH-CREATE: Only a single node is created when an alias is applied to a variable name multiple times",
        query: "MATCH (n) WITH n AS a CREATE (a)-[:T]->() WITH a AS x CREATE (x)-[:T]->() RETURN x",
        tranche: Tranche::OrderedCreate,
    },
    Case {
        report_index: 104,
        feature: "clauses/create/Create3.feature",
        scenario: 9,
        name: "[9] WITH-CREATE: A bound node should be recognized after projection with WITH + WITH",
        query: "CREATE (a) WITH a WITH * CREATE (b) CREATE (a)<-[:T]-(b)",
        tranche: Tranche::OrderedCreate,
    },
    Case {
        report_index: 105,
        feature: "clauses/create/Create3.feature",
        scenario: 10,
        name: "[10] WITH-UNWIND-CREATE: A bound node should be recognized after projection with WITH + UNWIND",
        query: "CREATE (a) WITH a UNWIND [0] AS i CREATE (b) CREATE (a)<-[:T]-(b)",
        tranche: Tranche::OrderedCreate,
    },
    Case {
        report_index: 106,
        feature: "clauses/create/Create3.feature",
        scenario: 11,
        name: "[11] WITH-MERGE-CREATE: A bound node should be recognized after projection with WITH + MERGE node",
        query: "CREATE (a) WITH a MERGE () CREATE (b) CREATE (a)<-[:T]-(b)",
        tranche: Tranche::ConstantNodeMerge,
    },
    Case {
        report_index: 107,
        feature: "clauses/create/Create3.feature",
        scenario: 12,
        name: "[12] WITH-MERGE-CREATE: A bound node should be recognized after projection with WITH + MERGE pattern",
        query: "CREATE (a) WITH a MERGE (x) MERGE (y) MERGE (x)-[:T]->(y) CREATE (b) CREATE (a)<-[:T]-(b)",
        tranche: Tranche::ConstantNodeMerge,
    },
    Case {
        report_index: 108,
        feature: "clauses/create/Create3.feature",
        scenario: 13,
        name: "[13] Merge followed by multiple creates",
        query: "MERGE (t:T {id: 42}) CREATE (f:R) CREATE (t)-[:REL]->(f)",
        tranche: Tranche::ConstantNodeMerge,
    },
    Case {
        report_index: 586,
        feature: "clauses/merge/Merge1.feature",
        scenario: 1,
        name: "[1] Merge node when no nodes exist",
        query: "MERGE (a) RETURN count(*) AS n",
        tranche: Tranche::ConstantNodeMerge,
    },
    Case {
        report_index: 587,
        feature: "clauses/merge/Merge1.feature",
        scenario: 2,
        name: "[2] Merge node with label",
        query: "MERGE (a:TheLabel) RETURN labels(a)",
        tranche: Tranche::ConstantNodeMerge,
    },
    Case {
        report_index: 588,
        feature: "clauses/merge/Merge1.feature",
        scenario: 3,
        name: "[3] Merge node with label when it exists",
        query: "MERGE (a:TheLabel) RETURN a.id",
        tranche: Tranche::ConstantNodeMerge,
    },
    Case {
        report_index: 589,
        feature: "clauses/merge/Merge1.feature",
        scenario: 4,
        name: "[4] Merge node should create when it doesn't match, properties",
        query: "MERGE (a {num: 43}) RETURN a.num",
        tranche: Tranche::ConstantNodeMerge,
    },
    Case {
        report_index: 590,
        feature: "clauses/merge/Merge1.feature",
        scenario: 5,
        name: "[5] Merge node should create when it doesn't match, properties and label",
        query: "MERGE (a:TheLabel {num: 43}) RETURN a.num",
        tranche: Tranche::ConstantNodeMerge,
    },
    Case {
        report_index: 591,
        feature: "clauses/merge/Merge1.feature",
        scenario: 6,
        name: "[6] Merge node with prop and label",
        query: "MERGE (a:TheLabel {num: 42}) RETURN a.num",
        tranche: Tranche::ConstantNodeMerge,
    },
    Case {
        report_index: 592,
        feature: "clauses/merge/Merge1.feature",
        scenario: 7,
        name: "[7] Merge should work when finding multiple elements",
        query: "CREATE (:X) CREATE (:X) MERGE (:X)",
        tranche: Tranche::ConstantNodeMerge,
    },
    Case {
        report_index: 593,
        feature: "clauses/merge/Merge1.feature",
        scenario: 8,
        name: "[8] Merge should handle argument properly",
        query: "WITH 42 AS var MERGE (c:N {var: var})",
        tranche: Tranche::ConstantNodeMerge,
    },
    Case {
        report_index: 594,
        feature: "clauses/merge/Merge1.feature",
        scenario: 9,
        name: "[9] Merge should support updates while merging",
        query: "MATCH (foo) WITH foo.x AS x, foo.y AS y MERGE (:N {x: x, y: y + 1}) MERGE (:N {x: x, y: y}) MERGE (:N {x: x + 1, y: y}) RETURN x, y",
        tranche: Tranche::DynamicCandidates,
    },
    Case {
        report_index: 595,
        feature: "clauses/merge/Merge1.feature",
        scenario: 10,
        name: "[10] Merge must properly handle multiple labels",
        query: "MERGE (test:L:B {num: 42}) RETURN labels(test) AS labels",
        tranche: Tranche::ConstantNodeMerge,
    },
    Case {
        report_index: 596,
        feature: "clauses/merge/Merge1.feature",
        scenario: 11,
        name: "[11] Merge should be able to merge using property of bound node",
        query: "MATCH (person:Person) MERGE (city:City {name: person.bornIn})",
        tranche: Tranche::DynamicCandidates,
    },
    Case {
        report_index: 597,
        feature: "clauses/merge/Merge1.feature",
        scenario: 12,
        name: "[12] Merge should be able to merge using property of freshly created node",
        query: "CREATE (a {num: 1}) MERGE ({v: a.num})",
        tranche: Tranche::ConstantNodeMerge,
    },
    Case {
        report_index: 598,
        feature: "clauses/merge/Merge1.feature",
        scenario: 13,
        name: "[13] Merge should bind a path",
        query: "MERGE p = (a {num: 1}) RETURN p",
        tranche: Tranche::ConstantNodeMerge,
    },
    Case {
        report_index: 599,
        feature: "clauses/merge/Merge1.feature",
        scenario: 14,
        name: "[14] Merges should not be able to match on deleted nodes",
        query: "MATCH (a:A) DELETE a MERGE (a2:A) RETURN a2.num",
        tranche: Tranche::ConstantNodeMerge,
    },
];

#[test]
fn tranche_partition_is_complete_and_non_overlapping() {
    assert_eq!(
        CASES
            .iter()
            .filter(|case| case.tranche == Tranche::OrderedCreate)
            .count(),
        9
    );
    assert_eq!(
        CASES
            .iter()
            .filter(|case| case.tranche == Tranche::ConstantNodeMerge)
            .count(),
        15
    );
    assert_eq!(
        CASES
            .iter()
            .filter(|case| case.tranche == Tranche::DynamicCandidates)
            .count(),
        3
    );
}

const PROJECT: ProjectId = ProjectId(uuid::Uuid::from_u128(
    0x4352_4541_5445_335f_4f52_4445_5245_0001,
));
const MEMORY_LIMIT_BYTES: usize = 256 * 1024 * 1024;
const RESERVED_BYTES: usize = 16 * 1024 * 1024;
const MAX_RESULT_ROWS: usize = 256;

#[derive(Clone, Copy, Debug)]
enum ExpectedNodeShape {
    Label(&'static str),
    IntegerProperty(&'static str, i64),
    StringProperty(&'static str, &'static str),
}

#[derive(Clone, Copy, Debug)]
struct Create3Case {
    report_index: usize,
    scenario: u8,
    name: &'static str,
    setup: &'static str,
    query: &'static str,
    columns: &'static [&'static str],
    row: &'static [ExpectedNodeShape],
    nodes_created: u64,
    relationships_created: u64,
    labels_added: u64,
    properties_set: u64,
}

impl Create3Case {
    fn label(self) -> String {
        format!(
            "Create3 report {} [{}] {}",
            self.report_index, self.scenario, self.name
        )
    }

    const fn statistics(self) -> StatementStats {
        StatementStats {
            nodes_created: self.nodes_created,
            nodes_deleted: 0,
            relationships_created: self.relationships_created,
            relationships_deleted: 0,
            properties_set: self.properties_set,
            labels_added: self.labels_added,
            labels_removed: 0,
        }
    }

    /// CREATE-time property payloads are present in canonical insert mutations but are not
    /// reported as `properties_set` by the query result statistics contract.
    const fn inserted_properties(self) -> u64 {
        if self.scenario == 13 { 1 } else { 0 }
    }
}

const CREATE3_CASES: [Create3Case; 13] = [
    Create3Case {
        report_index: 96,
        scenario: 1,
        name: "[1] MATCH-CREATE",
        setup: "CREATE (), ()",
        query: "MATCH () CREATE ()",
        columns: &[],
        row: &[],
        nodes_created: 2,
        relationships_created: 0,
        labels_added: 0,
        properties_set: 0,
    },
    Create3Case {
        report_index: 97,
        scenario: 2,
        name: "[2] WITH-CREATE",
        setup: "CREATE (), ()",
        query: "MATCH () CREATE () WITH * CREATE ()",
        columns: &[],
        row: &[],
        nodes_created: 4,
        relationships_created: 0,
        labels_added: 0,
        properties_set: 0,
    },
    Create3Case {
        report_index: 98,
        scenario: 3,
        name: "[3] MATCH-CREATE-WITH-CREATE",
        setup: "CREATE (), ()",
        query: "MATCH () CREATE () WITH * MATCH () CREATE ()",
        columns: &[],
        row: &[],
        nodes_created: 10,
        relationships_created: 0,
        labels_added: 0,
        properties_set: 0,
    },
    Create3Case {
        report_index: 99,
        scenario: 4,
        name: "[4] MATCH-CREATE: Newly-created nodes not visible to preceding MATCH",
        setup: "CREATE ()",
        query: "MATCH () CREATE ()",
        columns: &[],
        row: &[],
        nodes_created: 1,
        relationships_created: 0,
        labels_added: 0,
        properties_set: 0,
    },
    Create3Case {
        report_index: 100,
        scenario: 5,
        name: "[5] WITH-CREATE: Nodes are not created when aliases are applied to variable names",
        setup: "CREATE ({num: 1})",
        query: "MATCH (n) MATCH (m) WITH n AS a, m AS b CREATE (a)-[:T]->(b) RETURN a, b",
        columns: &["a", "b"],
        row: &[
            ExpectedNodeShape::IntegerProperty("num", 1),
            ExpectedNodeShape::IntegerProperty("num", 1),
        ],
        nodes_created: 0,
        relationships_created: 1,
        labels_added: 0,
        properties_set: 0,
    },
    Create3Case {
        report_index: 101,
        scenario: 6,
        name: "[6] WITH-CREATE: Only a single node is created when an alias is applied to a variable name",
        setup: "CREATE (:X)",
        query: "MATCH (n) WITH n AS a CREATE (a)-[:T]->() RETURN a",
        columns: &["a"],
        row: &[ExpectedNodeShape::Label("X")],
        nodes_created: 1,
        relationships_created: 1,
        labels_added: 0,
        properties_set: 0,
    },
    Create3Case {
        report_index: 102,
        scenario: 7,
        name: "[7] WITH-CREATE: Nodes are not created when aliases are applied to variable names multiple times",
        setup: "CREATE ({name: 'A'})",
        query: "MATCH (n) MATCH (m) WITH n AS a, m AS b CREATE (a)-[:T]->(b) WITH a AS x, b AS y CREATE (x)-[:T]->(y) RETURN x, y",
        columns: &["x", "y"],
        row: &[
            ExpectedNodeShape::StringProperty("name", "A"),
            ExpectedNodeShape::StringProperty("name", "A"),
        ],
        nodes_created: 0,
        relationships_created: 2,
        labels_added: 0,
        properties_set: 0,
    },
    Create3Case {
        report_index: 103,
        scenario: 8,
        name: "[8] WITH-CREATE: Only a single node is created when an alias is applied to variable names multiple times",
        setup: "CREATE ({num: 5})",
        query: "MATCH (n) WITH n AS a CREATE (a)-[:T]->() WITH a AS x CREATE (x)-[:T]->() RETURN x",
        columns: &["x"],
        row: &[ExpectedNodeShape::IntegerProperty("num", 5)],
        nodes_created: 2,
        relationships_created: 2,
        labels_added: 0,
        properties_set: 0,
    },
    Create3Case {
        report_index: 104,
        scenario: 9,
        name: "[9] WITH-CREATE: A bound node should be recognized after projection with WITH + WITH",
        setup: "",
        query: "CREATE (a) WITH a WITH * CREATE (b) CREATE (a)<-[:T]-(b)",
        columns: &[],
        row: &[],
        nodes_created: 2,
        relationships_created: 1,
        labels_added: 0,
        properties_set: 0,
    },
    Create3Case {
        report_index: 105,
        scenario: 10,
        name: "[10] WITH-UNWIND-CREATE: A bound node should be recognized after projection with WITH + UNWIND",
        setup: "",
        query: "CREATE (a) WITH a UNWIND [0] AS i CREATE (b) CREATE (a)<-[:T]-(b)",
        columns: &[],
        row: &[],
        nodes_created: 2,
        relationships_created: 1,
        labels_added: 0,
        properties_set: 0,
    },
    Create3Case {
        report_index: 106,
        scenario: 11,
        name: "[11] WITH-MERGE-CREATE: A bound node should be recognized after projection with WITH + MERGE node",
        setup: "",
        query: "CREATE (a) WITH a MERGE () CREATE (b) CREATE (a)<-[:T]-(b)",
        columns: &[],
        row: &[],
        nodes_created: 2,
        relationships_created: 1,
        labels_added: 0,
        properties_set: 0,
    },
    Create3Case {
        report_index: 107,
        scenario: 12,
        name: "[12] WITH-MERGE-CREATE: A bound node should be recognized after projection with WITH + MERGE pattern",
        setup: "",
        query: "CREATE (a) WITH a MERGE (x) MERGE (y) MERGE (x)-[:T]->(y) CREATE (b) CREATE (a)<-[:T]-(b)",
        columns: &[],
        row: &[],
        nodes_created: 2,
        relationships_created: 2,
        labels_added: 0,
        properties_set: 0,
    },
    Create3Case {
        report_index: 108,
        scenario: 13,
        name: "[13] Merge followed by multiple creates",
        setup: "",
        query: "MERGE (t:T {id: 42}) CREATE (f:R) CREATE (t)-[:REL]->(f)",
        columns: &[],
        row: &[],
        nodes_created: 2,
        relationships_created: 1,
        labels_added: 2,
        properties_set: 0,
    },
];

#[derive(Clone, Debug)]
struct Fixture {
    graph: GraphStore,
    bookmark: Bookmark,
}

impl Fixture {
    fn build(case: Create3Case) -> Result<Self> {
        let mut graph = GraphStore::default();
        if !case.setup.is_empty() {
            let setup =
                QueryEngine.execute(case.setup, &mut create3_context(&graph, None, false))?;
            if !setup.temporal_mutations.is_empty()
                || setup.administrative.is_some()
                || !setup.vector_searches.is_empty()
                || setup.runtime_replans != 0
            {
                return Err(Error::internal(format!(
                    "{} setup emitted unrelated execution state",
                    case.label()
                )));
            }
            for mutation in setup.graph_mutations {
                graph.apply(mutation)?;
            }
        }
        Ok(Self {
            bookmark: Bookmark {
                term: 103,
                index: graph.revision(),
            },
            graph,
        })
    }

    fn image(&self) -> Result<ResidentProjectImage> {
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
    let next_node_id = graph
        .nodes()
        .map(|node| node.id().0)
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or_else(|| Error::new(ErrorCode::ResultBudgetExceeded, "Create3 node ID overflow"))?;
    let next_edge_id = graph
        .edges()
        .map(|edge| edge.id().0)
        .max()
        .unwrap_or(0)
        .checked_add(1)
        .ok_or_else(|| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "Create3 relationship ID overflow",
            )
        })?;
    Ok((next_node_id, next_edge_id))
}

fn create3_context<'a>(
    graph: &'a GraphStore,
    backend: Option<&'a dyn ExecutionBackend>,
    require_native_execution: bool,
) -> ExecutionContext<'a> {
    let (next_node_id, next_edge_id) = next_ids(graph).expect("Create3 IDs must be valid");
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
            term: 103,
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

fn assert_node_shape(value: &ResultValue, expected: ExpectedNodeShape, label: &str) -> Result<()> {
    let ResultValue::Node(node) = value else {
        return Err(Error::internal(format!(
            "{label}: expected node result, got {value:?}"
        )));
    };
    if node.layer != Layer::Observed {
        return Err(Error::internal(format!(
            "{label}: result node left OBSERVED layer: {node:?}"
        )));
    }
    let (labels, properties) = match expected {
        ExpectedNodeShape::Label(label) => (vec![label.to_owned()], BTreeMap::new()),
        ExpectedNodeShape::IntegerProperty(name, value) => (
            Vec::new(),
            BTreeMap::from([(name.to_owned(), ScalarValue::Integer(value))]),
        ),
        ExpectedNodeShape::StringProperty(name, value) => (
            Vec::new(),
            BTreeMap::from([(name.to_owned(), ScalarValue::String(value.into()))]),
        ),
    };
    if node.labels != labels || node.properties != properties {
        return Err(Error::internal(format!(
            "{label}: node shape mismatch: expected {labels:?}/{properties:?}, got {node:?}"
        )));
    }
    Ok(())
}

fn assert_case_output(
    case: Create3Case,
    fixture: &Fixture,
    output: &ExecutionOutput,
) -> Result<()> {
    let label = case.label();
    let expected_schema = case
        .columns
        .iter()
        .map(|name| ((*name).to_owned(), ColumnType::Node))
        .collect::<Vec<_>>();
    if output.result.schema != expected_schema
        || output.result.bookmark != fixture.bookmark
        || output.result.statistics != case.statistics()
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

    let expected_rows = usize::from(!case.row.is_empty());
    let mut actual_rows = 0_usize;
    for batch in &output.result.batches {
        if !batch.validate()
            || batch.columns.len() != case.columns.len()
            || batch
                .columns
                .iter()
                .zip(case.columns)
                .any(|(actual, expected)| {
                    actual.name != *expected || actual.value_type != ColumnType::Node
                })
        {
            return Err(Error::internal(format!(
                "{label}: malformed result batch: {batch:?}"
            )));
        }
        for row in 0..batch.row_count {
            if actual_rows != 0 || case.row.len() != batch.columns.len() {
                return Err(Error::internal(format!(
                    "{label}: unexpected result multiplicity or width"
                )));
            }
            for (column, expected) in batch.columns.iter().zip(case.row) {
                assert_node_shape(&column.values[row], *expected, &label)?;
            }
            actual_rows = actual_rows.saturating_add(1);
        }
    }
    if actual_rows != expected_rows {
        return Err(Error::internal(format!(
            "{label}: expected {expected_rows} result rows, got {actual_rows}"
        )));
    }

    let mut nodes = 0_u64;
    let mut relationships = 0_u64;
    let mut labels = 0_u64;
    let mut properties = 0_u64;
    for mutation in &output.graph_mutations {
        match mutation {
            GraphMutation::InsertNode(node) => {
                nodes = nodes.saturating_add(1);
                labels = labels.saturating_add(node.labels.len() as u64);
                properties = properties.saturating_add(node.properties.len() as u64);
            }
            GraphMutation::InsertEdge(edge) => {
                relationships = relationships.saturating_add(1);
                properties = properties.saturating_add(edge.properties.len() as u64);
            }
            GraphMutation::DeclareLabel { .. }
            | GraphMutation::DeclareProperty { .. }
            | GraphMutation::DeclareRelationshipType { .. } => {}
            mutation => {
                return Err(Error::internal(format!(
                    "{label}: CREATE emitted non-insert graph mutation {mutation:?}"
                )));
            }
        }
    }
    if (nodes, relationships, labels, properties)
        != (
            case.nodes_created,
            case.relationships_created,
            case.labels_added,
            case.inserted_properties(),
        )
    {
        return Err(Error::internal(format!(
            "{label}: graph mutation shape mismatch: got {nodes}/{relationships}/{labels}/{properties}"
        )));
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

/// Advertises Metal during planning but permits exactly one pinned ordered row-mutation command.
/// Its inner CPU backend is solely the semantic reference for this hardware-independent gate.
struct StrictOrderedCreateBackend {
    inner: Box<dyn ExecutionBackend>,
    execution_kind: BackendKind,
    pinned: bool,
    expected_bookmark: Bookmark,
    expected_graph_revision: u64,
    observations: Arc<RouteObservations>,
}

impl StrictOrderedCreateBackend {
    fn strict_cpu(image: ResidentProjectImage) -> Result<Self> {
        let mut cpu = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
        cpu.admit_project(image)?;
        let expected_bookmark = cpu
            .resident_bookmark(PROJECT)
            .ok_or_else(|| Error::internal("strict Create3 backend lost its bookmark"))?;
        let expected_graph_revision = cpu
            .resident_graph_revision(PROJECT)
            .ok_or_else(|| Error::internal("strict Create3 backend lost its graph revision"))?;
        Ok(Self {
            inner: Box::new(cpu),
            execution_kind: BackendKind::Cpu,
            pinned: false,
            expected_bookmark,
            expected_graph_revision,
            observations: Arc::new(RouteObservations::default()),
        })
    }

    #[cfg(all(feature = "accelerator", target_os = "macos"))]
    fn real_metal(inner: MetalBackend) -> Result<Self> {
        if inner.kind() != BackendKind::Metal {
            return Err(Error::internal(
                "strict Create3 gate did not receive a real Metal backend",
            ));
        }
        let expected_bookmark = inner
            .resident_bookmark(PROJECT)
            .ok_or_else(|| Error::internal("strict Create3 Metal backend lost its bookmark"))?;
        let expected_graph_revision = inner.resident_graph_revision(PROJECT).ok_or_else(|| {
            Error::internal("strict Create3 Metal backend lost its graph revision")
        })?;
        Ok(Self {
            inner: Box::new(inner),
            execution_kind: BackendKind::Metal,
            pinned: false,
            expected_bookmark,
            expected_graph_revision,
            observations: Arc::new(RouteObservations::default()),
        })
    }

    fn observations(&self) -> Arc<RouteObservations> {
        Arc::clone(&self.observations)
    }

    fn reject<T>(&self, route: &'static str) -> Result<T> {
        self.observations
            .forbidden_calls
            .fetch_add(1, Ordering::SeqCst);
        Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!("strict Create3 gate rejected fallback `{route}`"),
        ))
    }
}

impl ExecutionBackend for StrictOrderedCreateBackend {
    fn kind(&self) -> BackendKind {
        if self.pinned {
            self.execution_kind
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
        if inner.kind() != self.execution_kind
            || inner.resident_bookmark(project) != Some(self.expected_bookmark)
            || inner.resident_graph_revision(project) != Some(self.expected_graph_revision)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "strict Create3 pin changed backend provenance or resident generation",
            ));
        }
        self.observations.pins.fetch_add(1, Ordering::SeqCst);
        Ok(Box::new(Self {
            inner,
            execution_kind: self.execution_kind,
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
            return self.reject("execute_row_mutation_on_unpinned_generation");
        }
        if request.bound_relationship_merge_body().is_none() {
            return self.reject("execute_row_mutation_without_ordered_body");
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
                "Create3 command escaped its pinned resident generation",
            ));
        }
        if self
            .observations
            .complete_commands
            .compare_exchange(0, 1, Ordering::SeqCst, Ordering::SeqCst)
            .is_err()
        {
            return self.reject("execute_row_mutation_more_than_once");
        }
        self.observations
            .requests
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .push(request.clone());
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
        self.reject("execute_create_node")
    }

    fn execute_row_program(
        &self,
        _request: &ResidentRowProgramRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentRowProgramResult> {
        self.reject("execute_row_program")
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

fn assert_one_ordered_command(
    case: Create3Case,
    fixture: &Fixture,
    observations: &RouteObservations,
) -> Result<ResidentRowMutationRequest> {
    let pins = observations.pins.load(Ordering::SeqCst);
    let commands = observations.complete_commands.load(Ordering::SeqCst);
    let forbidden = observations.forbidden_calls.load(Ordering::SeqCst);
    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if pins != 1 || commands != 1 || forbidden != 0 || requests.len() != 1 {
        return Err(Error::internal(format!(
            "{}: expected one pin/command/request and no fallback, got {pins}/{commands}/{}/{forbidden}",
            case.label(),
            requests.len()
        )));
    }
    let request = requests[0].clone();
    request.validate()?;
    if request.generation.bookmark != fixture.bookmark
        || request.generation.graph_revision != fixture.graph.revision()
        || request.bound_relationship_merge_body().is_none()
    {
        return Err(Error::internal(format!(
            "{}: ordered command lost its generation or body",
            case.label()
        )));
    }
    Ok(request)
}

fn assert_exact_match_create_request(
    case: Create3Case,
    fixture: &Fixture,
    request: &ResidentRowMutationRequest,
) -> Result<()> {
    let Some(body) = request.bound_relationship_merge_body() else {
        return Err(Error::internal("Create3 request lost ordered body"));
    };
    let expected_candidates = u64::try_from(fixture.graph.node_count())
        .map_err(|_| Error::internal("Create3 fixture node count exceeds u64"))?;
    if body.program.entity_slot_count != 2
        || !body.program.input_keys.is_empty()
        || !body.program.property_names.is_empty()
        || !body.program.label_names.is_empty()
        || !body.program.relationship_type_names.is_empty()
        || !body.outputs.is_empty()
        || body.command_candidate_limits.as_slice() != [expected_candidates, 1]
        || !matches!(
            (body.program.commands.as_slice(), body.commands.as_slice()),
            (
                [
                    ResidentRowCreateCommand::MatchNode(matched),
                    ResidentRowCreateCommand::CreateNode(created),
                ],
                [
                    ResidentRowBoundRelationshipCommand::MatchNode { properties },
                    ResidentRowBoundRelationshipCommand::CreateNode,
                ],
            ) if matched.output_entity == 0
                && matched.labels.is_empty()
                && properties.is_empty()
                && created.output_entity == 1
                && created.labels.is_empty()
                && created.properties.is_empty()
        )
        || !matches!(
            body.schedule.stages.as_slice(),
            [
                ResidentRowBoundRelationshipStage::Command { command: 0 },
                ResidentRowBoundRelationshipStage::Command { command: 1 },
            ]
        )
    {
        return Err(Error::internal(format!(
            "{}: compiler broadened the exact MATCH/CREATE command: {body:#?}",
            case.label()
        )));
    }
    Ok(())
}

fn assert_exact_match_create_star_create_request(
    case: Create3Case,
    fixture: &Fixture,
    request: &ResidentRowMutationRequest,
) -> Result<()> {
    let Some(body) = request.bound_relationship_merge_body() else {
        return Err(Error::internal("Create3 [2] request lost ordered body"));
    };
    let expected_candidates = u64::try_from(fixture.graph.node_count())
        .map_err(|_| Error::internal("Create3 [2] fixture node count exceeds u64"))?;
    if body.program.entity_slot_count != 3
        || !body.program.input_keys.is_empty()
        || !body.program.property_names.is_empty()
        || !body.program.property_tokens.is_empty()
        || !body.program.label_names.is_empty()
        || !body.program.label_tokens.is_empty()
        || !body.program.relationship_type_names.is_empty()
        || !body.program.relationship_type_tokens.is_empty()
        || !body.program.outputs.is_empty()
        || body.program.continuation.is_some()
        || !body.outputs.is_empty()
        || body.command_candidate_limits.as_slice() != [expected_candidates, 1, 1]
        || body.schedule.value_slot_count != 0
        || !body.schedule.property_bindings.is_empty()
        || !matches!(
            (body.program.commands.as_slice(), body.commands.as_slice()),
            (
                [
                    ResidentRowCreateCommand::MatchNode(matched),
                    ResidentRowCreateCommand::CreateNode(first_created),
                    ResidentRowCreateCommand::CreateNode(second_created),
                ],
                [
                    ResidentRowBoundRelationshipCommand::MatchNode { properties },
                    ResidentRowBoundRelationshipCommand::CreateNode,
                    ResidentRowBoundRelationshipCommand::CreateNode,
                ],
            ) if matched.output_entity == 0
                && matched.labels.is_empty()
                && properties.is_empty()
                && first_created.output_entity == 1
                && first_created.labels.is_empty()
                && first_created.properties.is_empty()
                && second_created.output_entity == 2
                && second_created.labels.is_empty()
                && second_created.properties.is_empty()
        )
        || !matches!(
            body.schedule.stages.as_slice(),
            [
                ResidentRowBoundRelationshipStage::Command { command: 0 },
                ResidentRowBoundRelationshipStage::Command { command: 1 },
                ResidentRowBoundRelationshipStage::Command { command: 2 },
            ]
        )
    {
        return Err(Error::internal(format!(
            "{}: compiler broadened the exact MATCH/CREATE/WITH */CREATE command: {body:#?}",
            case.label()
        )));
    }
    Ok(())
}

fn assert_exact_match_create_star_match_create_request(
    case: Create3Case,
    fixture: &Fixture,
    request: &ResidentRowMutationRequest,
) -> Result<()> {
    let Some(body) = request.bound_relationship_merge_body() else {
        return Err(Error::internal("Create3 [3] request lost ordered body"));
    };
    let immutable = u64::try_from(fixture.graph.node_count())
        .map_err(|_| Error::internal("Create3 [3] fixture node count exceeds u64"))?;
    let post_write = immutable
        .checked_mul(2)
        .ok_or_else(|| Error::internal("Create3 [3] candidate count overflow"))?;
    let expected_rows = immutable
        .checked_mul(post_write)
        .ok_or_else(|| Error::internal("Create3 [3] row count overflow"))?;
    let expected_intents = immutable
        .checked_add(expected_rows)
        .ok_or_else(|| Error::internal("Create3 [3] intent count overflow"))?;
    if body.program.entity_slot_count != 4
        || !body.program.input_keys.is_empty()
        || !body.program.property_names.is_empty()
        || !body.program.property_tokens.is_empty()
        || !body.program.label_names.is_empty()
        || !body.program.label_tokens.is_empty()
        || !body.program.relationship_type_names.is_empty()
        || !body.program.relationship_type_tokens.is_empty()
        || !body.program.outputs.is_empty()
        || body.program.continuation.is_some()
        || !body.outputs.is_empty()
        || body.command_candidate_limits.as_slice() != [immutable, 1, post_write, 1]
        || body.capacities.maximum_rows != expected_rows
        || body.capacities.maximum_intents != expected_intents
        || body.schedule.value_slot_count != 0
        || !body.schedule.property_bindings.is_empty()
        || !matches!(
            (body.program.commands.as_slice(), body.commands.as_slice()),
            (
                [
                    ResidentRowCreateCommand::MatchNode(first_match),
                    ResidentRowCreateCommand::CreateNode(first_create),
                    ResidentRowCreateCommand::MatchNode(second_match),
                    ResidentRowCreateCommand::CreateNode(second_create),
                ],
                [
                    ResidentRowBoundRelationshipCommand::MatchNode {
                        properties: first_properties,
                    },
                    ResidentRowBoundRelationshipCommand::CreateNode,
                    ResidentRowBoundRelationshipCommand::MatchNodeIncludingWrites {
                        properties: second_properties,
                    },
                    ResidentRowBoundRelationshipCommand::CreateNode,
                ],
            ) if first_match.output_entity == 0
                && first_match.labels.is_empty()
                && first_properties.is_empty()
                && first_create.output_entity == 1
                && first_create.labels.is_empty()
                && first_create.properties.is_empty()
                && second_match.output_entity == 2
                && second_match.labels.is_empty()
                && second_properties.is_empty()
                && second_create.output_entity == 3
                && second_create.labels.is_empty()
                && second_create.properties.is_empty()
        )
        || !matches!(
            body.schedule.stages.as_slice(),
            [
                ResidentRowBoundRelationshipStage::Command { command: 0 },
                ResidentRowBoundRelationshipStage::Command { command: 1 },
                ResidentRowBoundRelationshipStage::Command { command: 2 },
                ResidentRowBoundRelationshipStage::Command { command: 3 },
            ]
        )
    {
        return Err(Error::internal(format!(
            "{}: compiler broadened the exact MATCH/CREATE/WITH */MATCH/CREATE command: {body:#?}",
            case.label()
        )));
    }
    Ok(())
}

fn assert_exact_graph_free_create3_10_through_12_request(
    case: Create3Case,
    request: &ResidentRowMutationRequest,
) -> Result<()> {
    let Some(body) = request.bound_relationship_merge_body() else {
        return Err(Error::internal(format!(
            "{}: graph-free CREATE/MERGE request lost ordered body",
            case.label()
        )));
    };
    if !body.program.input_keys.is_empty()
        || !body.program.property_names.is_empty()
        || !body.program.property_tokens.is_empty()
        || !body.program.label_names.is_empty()
        || !body.program.label_tokens.is_empty()
        || body.program.relationship_type_names.as_slice() != ["T"]
        || body.program.relationship_type_tokens.len() != 1
        || !body.program.outputs.is_empty()
        || body.program.continuation.is_some()
        || !body.outputs.is_empty()
        || body.schedule.value_slot_count != 0
        || !body.schedule.property_bindings.is_empty()
    {
        return Err(Error::internal(format!(
            "{}: graph-free CREATE/MERGE metadata broadened: {body:#?}",
            case.label()
        )));
    }

    let exact = match case.scenario {
        10 => {
            body.program.entity_slot_count == 3
                && body.command_candidate_limits.as_slice() == [1, 1, 1]
                && matches!(
                    (body.program.commands.as_slice(), body.commands.as_slice()),
                    (
                        [
                            ResidentRowCreateCommand::CreateNode(first),
                            ResidentRowCreateCommand::CreateNode(second),
                            ResidentRowCreateCommand::CreateRelationship(relationship),
                        ],
                        [
                            ResidentRowBoundRelationshipCommand::CreateNode,
                            ResidentRowBoundRelationshipCommand::CreateNode,
                            ResidentRowBoundRelationshipCommand::CreateRelationship,
                        ],
                    ) if first.output_entity == 0
                        && first.labels.is_empty()
                        && first.properties.is_empty()
                        && second.output_entity == 1
                        && second.labels.is_empty()
                        && second.properties.is_empty()
                        && relationship.output_entity == 2
                        && relationship.source_entity == 1
                        && relationship.target_entity == 0
                        && relationship.relationship_type == 0
                        && relationship.properties.is_empty()
                )
                && matches!(
                    body.schedule.stages.as_slice(),
                    [
                        ResidentRowBoundRelationshipStage::Command { command: 0 },
                        ResidentRowBoundRelationshipStage::Command { command: 1 },
                        ResidentRowBoundRelationshipStage::Command { command: 2 },
                    ]
                )
        }
        11 => {
            body.program.entity_slot_count == 4
                && body.command_candidate_limits.as_slice() == [1, 1, 1, 1]
                && matches!(
                    (body.program.commands.as_slice(), body.commands.as_slice()),
                    (
                        [
                            ResidentRowCreateCommand::CreateNode(first),
                            ResidentRowCreateCommand::CreateNode(merged),
                            ResidentRowCreateCommand::CreateNode(second),
                            ResidentRowCreateCommand::CreateRelationship(relationship),
                        ],
                        [
                            ResidentRowBoundRelationshipCommand::CreateNode,
                            ResidentRowBoundRelationshipCommand::MergeNode,
                            ResidentRowBoundRelationshipCommand::CreateNode,
                            ResidentRowBoundRelationshipCommand::CreateRelationship,
                        ],
                    ) if first.output_entity == 0
                        && first.labels.is_empty()
                        && first.properties.is_empty()
                        && merged.output_entity == 1
                        && merged.labels.is_empty()
                        && merged.properties.is_empty()
                        && second.output_entity == 2
                        && second.labels.is_empty()
                        && second.properties.is_empty()
                        && relationship.output_entity == 3
                        && relationship.source_entity == 2
                        && relationship.target_entity == 0
                        && relationship.relationship_type == 0
                        && relationship.properties.is_empty()
                )
                && matches!(
                    body.schedule.stages.as_slice(),
                    [
                        ResidentRowBoundRelationshipStage::Command { command: 0 },
                        ResidentRowBoundRelationshipStage::Command { command: 1 },
                        ResidentRowBoundRelationshipStage::Command { command: 2 },
                        ResidentRowBoundRelationshipStage::Command { command: 3 },
                    ]
                )
        }
        12 => {
            body.program.entity_slot_count == 6
                && body.command_candidate_limits.as_slice() == [1, 1, 1, 1, 1, 1]
                && matches!(
                    (body.program.commands.as_slice(), body.commands.as_slice()),
                    (
                        [
                            ResidentRowCreateCommand::CreateNode(first),
                            ResidentRowCreateCommand::CreateNode(first_merged),
                            ResidentRowCreateCommand::CreateNode(second_merged),
                            ResidentRowCreateCommand::CreateRelationship(merged_relationship),
                            ResidentRowCreateCommand::CreateNode(second),
                            ResidentRowCreateCommand::CreateRelationship(created_relationship),
                        ],
                        [
                            ResidentRowBoundRelationshipCommand::CreateNode,
                            ResidentRowBoundRelationshipCommand::MergeNode,
                            ResidentRowBoundRelationshipCommand::MergeNode,
                            ResidentRowBoundRelationshipCommand::MergeRelationship {
                                direction: ResidentRowBoundRelationshipDirection::Directed,
                            },
                            ResidentRowBoundRelationshipCommand::CreateNode,
                            ResidentRowBoundRelationshipCommand::CreateRelationship,
                        ],
                    ) if first.output_entity == 0
                        && first.labels.is_empty()
                        && first.properties.is_empty()
                        && first_merged.output_entity == 1
                        && first_merged.labels.is_empty()
                        && first_merged.properties.is_empty()
                        && second_merged.output_entity == 2
                        && second_merged.labels.is_empty()
                        && second_merged.properties.is_empty()
                        && merged_relationship.output_entity == 3
                        && merged_relationship.source_entity == 1
                        && merged_relationship.target_entity == 2
                        && merged_relationship.relationship_type == 0
                        && merged_relationship.properties.is_empty()
                        && second.output_entity == 4
                        && second.labels.is_empty()
                        && second.properties.is_empty()
                        && created_relationship.output_entity == 5
                        && created_relationship.source_entity == 4
                        && created_relationship.target_entity == 0
                        && created_relationship.relationship_type == 0
                        && created_relationship.properties.is_empty()
                )
                && matches!(
                    body.schedule.stages.as_slice(),
                    [
                        ResidentRowBoundRelationshipStage::Command { command: 0 },
                        ResidentRowBoundRelationshipStage::Command { command: 1 },
                        ResidentRowBoundRelationshipStage::Command { command: 2 },
                        ResidentRowBoundRelationshipStage::Command { command: 3 },
                        ResidentRowBoundRelationshipStage::Command { command: 4 },
                        ResidentRowBoundRelationshipStage::Command { command: 5 },
                    ]
                )
        }
        _ => false,
    };
    if !exact {
        return Err(Error::internal(format!(
            "{}: compiler broadened or changed its exact graph-free CREATE/MERGE command: {body:#?}",
            case.label()
        )));
    }
    Ok(())
}

fn run_strict_case(case: Create3Case) -> Result<()> {
    let fixture = Fixture::build(case)?;
    let backend = StrictOrderedCreateBackend::strict_cpu(fixture.image()?)?;
    let observations = backend.observations();
    let output = QueryEngine.execute(
        case.query,
        &mut create3_context(&fixture.graph, Some(&backend), true),
    )?;
    let request = assert_one_ordered_command(case, &fixture, &observations)?;
    if matches!(case.scenario, 1 | 4) {
        assert_exact_match_create_request(case, &fixture, &request)?;
    } else if case.scenario == 2 {
        assert_exact_match_create_star_create_request(case, &fixture, &request)?;
    } else if case.scenario == 3 {
        assert_exact_match_create_star_match_create_request(case, &fixture, &request)?;
    } else if matches!(case.scenario, 10..=12) {
        assert_exact_graph_free_create3_10_through_12_request(case, &request)?;
    }
    assert_case_output(case, &fixture, &output)
}

#[test]
fn generic_cpu_oracle_pins_exact_rows_and_statistics_for_all_13_create3_cases() -> Result<()> {
    for case in CREATE3_CASES {
        let fixture = Fixture::build(case)?;
        let output = QueryEngine.execute(
            case.query,
            &mut create3_context(&fixture.graph, None, false),
        )?;
        assert_case_output(case, &fixture, &output)?;
    }
    Ok(())
}

#[test]
fn strict_cpu_executes_create3_96_97_and_99_as_one_ordered_native_command() -> Result<()> {
    for case in CREATE3_CASES
        .into_iter()
        .filter(|case| matches!(case.scenario, 1 | 2 | 4))
    {
        run_strict_case(case)?;
    }
    Ok(())
}

#[test]
fn strict_cpu_executes_create3_98_with_mixed_immutable_and_local_candidates() -> Result<()> {
    run_strict_case(CREATE3_CASES[2])
}

#[test]
fn strict_cpu_preserves_work_row_zero_for_one_local_candidate() -> Result<()> {
    run_strict_case(Create3Case {
        report_index: 98,
        scenario: 3,
        name: "[3] one-resident-node provenance proof",
        setup: "CREATE ()",
        query: "MATCH () CREATE () WITH * MATCH () CREATE ()",
        columns: &[],
        row: &[],
        nodes_created: 3,
        relationships_created: 0,
        labels_added: 0,
        properties_set: 0,
    })
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires a real Metal device; exercises Create3 [3] mixed immutable/local read-own-writes provenance through one native command"]
fn real_metal_executes_create3_98_with_mixed_immutable_and_local_candidates() -> Result<()> {
    let case = CREATE3_CASES[2];
    let fixture = Fixture::build(case)?;
    let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
    metal.admit_project(fixture.image()?)?;
    let backend = StrictOrderedCreateBackend::real_metal(metal)?;
    let observations = backend.observations();
    let output = QueryEngine.execute(
        case.query,
        &mut create3_context(&fixture.graph, Some(&backend), true),
    )?;
    let request = assert_one_ordered_command(case, &fixture, &observations)?;
    assert_exact_match_create_star_match_create_request(case, &fixture, &request)?;
    assert_case_output(case, &fixture, &output)
}

#[test]
fn strict_cpu_executes_create3_100_through_103_as_one_ordered_native_command() -> Result<()> {
    for case in CREATE3_CASES
        .into_iter()
        .filter(|case| matches!(case.scenario, 5..=8))
    {
        run_strict_case(case)?;
    }
    Ok(())
}

#[test]
fn strict_cpu_executes_create3_104_and_108_as_one_ordered_native_command() -> Result<()> {
    for case in CREATE3_CASES
        .into_iter()
        .filter(|case| matches!(case.scenario, 9 | 13))
    {
        run_strict_case(case)?;
    }
    Ok(())
}

#[test]
fn strict_cpu_executes_create3_105_through_107_as_one_ordered_native_command() -> Result<()> {
    for case in CREATE3_CASES
        .into_iter()
        .filter(|case| matches!(case.scenario, 10..=12))
    {
        run_strict_case(case)?;
    }
    Ok(())
}

#[test]
fn all_create3_routes_are_complete_or_fail_before_native_dispatch() -> Result<()> {
    let mut completed = 0_usize;
    let mut rejected_before_dispatch = 0_usize;
    for case in CREATE3_CASES {
        let fixture = Fixture::build(case)?;
        let backend = StrictOrderedCreateBackend::strict_cpu(fixture.image()?)?;
        let observations = backend.observations();
        match QueryEngine.execute(
            case.query,
            &mut create3_context(&fixture.graph, Some(&backend), true),
        ) {
            Ok(output) => {
                assert_one_ordered_command(case, &fixture, &observations)?;
                assert_case_output(case, &fixture, &output)?;
                completed += 1;
            }
            Err(error) => {
                let pins = observations.pins.load(Ordering::SeqCst);
                let commands = observations.complete_commands.load(Ordering::SeqCst);
                let forbidden = observations.forbidden_calls.load(Ordering::SeqCst);
                let requests = observations
                    .requests
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .len();
                if error.code != ErrorCode::GpuAdmissionFailure
                    || pins != 0
                    || commands != 0
                    || forbidden != 0
                    || requests != 0
                {
                    return Err(Error::internal(format!(
                        "{}: incomplete route did not fail before dispatch: {error}; {pins}/{commands}/{forbidden}/{requests}",
                        case.label()
                    )));
                }
                rejected_before_dispatch += 1;
            }
        }
    }
    assert_eq!(
        completed, 13,
        "every Create3 route must be one native command"
    );
    assert_eq!(rejected_before_dispatch, 0);
    Ok(())
}

#[test]
fn nearby_match_create_shapes_remain_fail_closed() -> Result<()> {
    let fixture = Fixture::build(CREATE3_CASES[3])?;
    for query in [
        "OPTIONAL MATCH () CREATE ()",
        "MATCH (n) CREATE ()",
        "MATCH () CREATE (n)",
        "MATCH (:X) CREATE ()",
        "MATCH ({id: 1}) CREATE ()",
        "MATCH () CREATE (:X)",
        "MATCH () CREATE () CREATE ()",
        "MATCH () CREATE () RETURN count(*)",
        "MATCH () WITH * CREATE () CREATE ()",
        "MATCH () CREATE () WITH DISTINCT * CREATE ()",
        "MATCH () CREATE () WITH *, 1 AS x CREATE ()",
        "MATCH () CREATE () WITH * WITH * CREATE ()",
        "MATCH () CREATE () WITH * CREATE (n)",
        "MATCH () CREATE () WITH * CREATE (:X)",
        "MATCH () CREATE () WITH * CREATE ({id: 1})",
        "MATCH () CREATE () WITH * CREATE () CREATE ()",
        "MATCH () CREATE () WITH * CREATE () RETURN count(*)",
        "MATCH () CREATE () WITH * OPTIONAL MATCH () CREATE ()",
        "MATCH () CREATE () WITH * MATCH (n) CREATE ()",
        "MATCH () CREATE () WITH * MATCH (:X) CREATE ()",
        "MATCH () CREATE () WITH * MATCH ({id: 1}) CREATE ()",
        "MATCH () CREATE () WITH * MATCH () CREATE (n)",
        "MATCH () CREATE () WITH * MATCH () CREATE (:X)",
        "MATCH () CREATE () WITH * MATCH () CREATE ({id: 1})",
        "MATCH () CREATE () WITH * MATCH () CREATE () CREATE ()",
        "MATCH () CREATE () WITH * MATCH () CREATE () RETURN count(*)",
    ] {
        let backend = StrictOrderedCreateBackend::strict_cpu(fixture.image()?)?;
        let observations = backend.observations();
        let error = QueryEngine
            .execute(
                query,
                &mut create3_context(&fixture.graph, Some(&backend), true),
            )
            .expect_err("broader MATCH/CREATE shape unexpectedly entered the exact native lane");
        if error.code != ErrorCode::GpuAdmissionFailure
            || observations.pins.load(Ordering::SeqCst) != 0
            || observations.complete_commands.load(Ordering::SeqCst) != 0
            || observations.forbidden_calls.load(Ordering::SeqCst) != 0
            || !observations
                .requests
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .is_empty()
        {
            return Err(Error::internal(format!(
                "adversarial query did not fail before native dispatch: `{query}`: {error}"
            )));
        }
    }
    Ok(())
}

#[test]
fn nearby_graph_free_create_merge_shapes_remain_fail_closed() -> Result<()> {
    let fixture = Fixture::build(CREATE3_CASES[9])?;
    for query in [
        "CREATE (a) WITH a UNWIND [] AS i CREATE (b) CREATE (a)<-[:T]-(b)",
        "CREATE (a) WITH a UNWIND [0, 1] AS i CREATE (b) CREATE (a)<-[:T]-(b)",
        "CREATE (a) WITH a UNWIND [1] AS i CREATE (b) CREATE (a)<-[:T]-(b)",
        "CREATE (a) WITH a UNWIND [0] AS i CREATE (b) CREATE (a)<-[:T]-(b) RETURN i",
        "CREATE (a) WITH a MERGE (m) CREATE (b) CREATE (a)<-[:T]-(b)",
        "CREATE (a) WITH a MERGE (:X) CREATE (b) CREATE (a)<-[:T]-(b)",
        "CREATE (a) WITH a MERGE ({id: 1}) CREATE (b) CREATE (a)<-[:T]-(b)",
        "CREATE (a) WITH a MERGE () MERGE () CREATE (b) CREATE (a)<-[:T]-(b)",
        "CREATE (a) WITH a MERGE (x) MERGE (y) MERGE (x)<-[:T]-(y) CREATE (b) CREATE (a)<-[:T]-(b)",
        "CREATE (a) WITH a MERGE (x) MERGE (y) MERGE (x)-[:R]->(y) CREATE (b) CREATE (a)<-[:T]-(b)",
    ] {
        let backend = StrictOrderedCreateBackend::strict_cpu(fixture.image()?)?;
        let observations = backend.observations();
        let error = QueryEngine
            .execute(
                query,
                &mut create3_context(&fixture.graph, Some(&backend), true),
            )
            .expect_err("graph-free Create3 near miss entered the exact native lane");
        if error.code != ErrorCode::GpuAdmissionFailure
            || observations.pins.load(Ordering::SeqCst) != 0
            || observations.complete_commands.load(Ordering::SeqCst) != 0
            || observations.forbidden_calls.load(Ordering::SeqCst) != 0
            || !observations
                .requests
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner())
                .is_empty()
        {
            return Err(Error::internal(format!(
                "graph-free Create3 near miss did not fail before native dispatch: `{query}`: {error}"
            )));
        }
    }
    Ok(())
}

#[test]
fn strict_cpu_executes_all_13_create3_cases_as_one_ordered_native_command() -> Result<()> {
    for case in CREATE3_CASES {
        run_strict_case(case)?;
    }
    Ok(())
}

// The historical 27-case manifest above remains a report-identity record. These executable Merge1
// contracts are intentionally self-contained and do not infer current status from that stale
// report: each query, setup, result, effect, and native request shape is pinned here directly.

#[derive(Clone, Copy, Debug)]
enum Merge1ExpectedOutput {
    Count {
        name: &'static str,
        value: i64,
    },
    Property {
        name: &'static str,
        property: &'static str,
        value: i64,
    },
    Labels {
        name: &'static str,
        labels: &'static [&'static str],
    },
}

impl Merge1ExpectedOutput {
    const fn name(self) -> &'static str {
        match self {
            Self::Count { name, .. } | Self::Property { name, .. } | Self::Labels { name, .. } => {
                name
            }
        }
    }

    const fn column_type(self) -> ColumnType {
        match self {
            Self::Count { .. } | Self::Property { .. } => ColumnType::Integer,
            Self::Labels { .. } => ColumnType::List,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Merge1TckEffects {
    nodes_added: u64,
    properties_added: u64,
    label_names_added: u64,
}

#[derive(Clone, Copy, Debug)]
struct Merge1Case {
    report_index: usize,
    scenario: u8,
    name: &'static str,
    setup: &'static str,
    query: &'static str,
    labels: &'static [&'static str],
    key: Option<(&'static str, i64)>,
    output: Merge1ExpectedOutput,
    nodes_created: u64,
    labels_assigned: u64,
    tck_effects: Merge1TckEffects,
}

impl Merge1Case {
    fn label(self) -> String {
        format!(
            "Merge1 report {} [{}] {}",
            self.report_index, self.scenario, self.name
        )
    }

    const fn statistics(self) -> StatementStats {
        StatementStats {
            nodes_created: self.nodes_created,
            nodes_deleted: 0,
            relationships_created: 0,
            relationships_deleted: 0,
            properties_set: 0,
            labels_added: self.labels_assigned,
            labels_removed: 0,
        }
    }
}

const MERGE1_CASES: [Merge1Case; 7] = [
    Merge1Case {
        report_index: 586,
        scenario: 1,
        name: "[1] Merge node when no nodes exist",
        setup: "",
        query: "MERGE (a) RETURN count(*) AS n",
        labels: &[],
        key: None,
        output: Merge1ExpectedOutput::Count {
            name: "n",
            value: 1,
        },
        nodes_created: 1,
        labels_assigned: 0,
        tck_effects: Merge1TckEffects {
            nodes_added: 1,
            properties_added: 0,
            label_names_added: 0,
        },
    },
    Merge1Case {
        report_index: 587,
        scenario: 2,
        name: "[2] Merge node with label",
        setup: "",
        query: "MERGE (a:TheLabel) RETURN labels(a)",
        labels: &["TheLabel"],
        key: None,
        output: Merge1ExpectedOutput::Labels {
            name: "labels(a)",
            labels: &["TheLabel"],
        },
        nodes_created: 1,
        labels_assigned: 1,
        tck_effects: Merge1TckEffects {
            nodes_added: 1,
            properties_added: 0,
            label_names_added: 1,
        },
    },
    Merge1Case {
        report_index: 588,
        scenario: 3,
        name: "[3] Merge node with label when it exists",
        setup: "CREATE (:TheLabel {id: 1})",
        query: "MERGE (a:TheLabel) RETURN a.id",
        labels: &["TheLabel"],
        key: None,
        output: Merge1ExpectedOutput::Property {
            name: "a.id",
            property: "id",
            value: 1,
        },
        nodes_created: 0,
        labels_assigned: 0,
        tck_effects: Merge1TckEffects {
            nodes_added: 0,
            properties_added: 0,
            label_names_added: 0,
        },
    },
    Merge1Case {
        report_index: 589,
        scenario: 4,
        name: "[4] Merge node should create when it doesn't match, properties",
        setup: "CREATE ({num: 42})",
        query: "MERGE (a {num: 43}) RETURN a.num",
        labels: &[],
        key: Some(("num", 43)),
        output: Merge1ExpectedOutput::Property {
            name: "a.num",
            property: "num",
            value: 43,
        },
        nodes_created: 1,
        labels_assigned: 0,
        tck_effects: Merge1TckEffects {
            nodes_added: 1,
            properties_added: 1,
            label_names_added: 0,
        },
    },
    Merge1Case {
        report_index: 590,
        scenario: 5,
        name: "[5] Merge node should create when it doesn't match, properties and label",
        setup: "CREATE (:TheLabel {num: 42})",
        query: "MERGE (a:TheLabel {num: 43}) RETURN a.num",
        labels: &["TheLabel"],
        key: Some(("num", 43)),
        output: Merge1ExpectedOutput::Property {
            name: "a.num",
            property: "num",
            value: 43,
        },
        nodes_created: 1,
        labels_assigned: 1,
        tck_effects: Merge1TckEffects {
            nodes_added: 1,
            properties_added: 1,
            label_names_added: 0,
        },
    },
    Merge1Case {
        report_index: 591,
        scenario: 6,
        name: "[6] Merge node with prop and label",
        setup: "CREATE (:TheLabel {num: 42})",
        query: "MERGE (a:TheLabel {num: 42}) RETURN a.num",
        labels: &["TheLabel"],
        key: Some(("num", 42)),
        output: Merge1ExpectedOutput::Property {
            name: "a.num",
            property: "num",
            value: 42,
        },
        nodes_created: 0,
        labels_assigned: 0,
        tck_effects: Merge1TckEffects {
            nodes_added: 0,
            properties_added: 0,
            label_names_added: 0,
        },
    },
    Merge1Case {
        report_index: 595,
        scenario: 10,
        name: "[10] Merge must properly handle multiple labels",
        setup: "CREATE (:L:A {num: 42})",
        query: "MERGE (test:L:B {num: 42}) RETURN labels(test) AS labels",
        labels: &["L", "B"],
        key: Some(("num", 42)),
        output: Merge1ExpectedOutput::Labels {
            name: "labels",
            labels: &["L", "B"],
        },
        nodes_created: 1,
        labels_assigned: 2,
        tck_effects: Merge1TckEffects {
            nodes_added: 1,
            properties_added: 1,
            label_names_added: 1,
        },
    },
];

fn build_merge1_fixture(setup: &str, label: &str) -> Result<Fixture> {
    let mut graph = GraphStore::default();
    if !setup.is_empty() {
        let output = QueryEngine.execute(setup, &mut create3_context(&graph, None, false))?;
        if !output.temporal_mutations.is_empty()
            || output.administrative.is_some()
            || !output.vector_searches.is_empty()
            || output.runtime_replans != 0
        {
            return Err(Error::internal(format!(
                "{label}: setup emitted unrelated execution state"
            )));
        }
        for mutation in output.graph_mutations {
            graph.apply(mutation)?;
        }
    }
    Ok(Fixture {
        bookmark: Bookmark {
            term: 103,
            index: graph.revision(),
        },
        graph,
    })
}

fn assert_merge1_result_value(
    actual: &ResultValue,
    expected: Merge1ExpectedOutput,
    label: &str,
) -> Result<()> {
    match expected {
        Merge1ExpectedOutput::Count { value, .. }
        | Merge1ExpectedOutput::Property { value, .. } => {
            if actual != &ResultValue::Scalar(ScalarValue::Integer(value)) {
                return Err(Error::internal(format!(
                    "{label}: expected INTEGER {value}, got {actual:?}"
                )));
            }
        }
        Merge1ExpectedOutput::Labels {
            labels: expected, ..
        } => {
            let ResultValue::List(actual) = actual else {
                return Err(Error::internal(format!(
                    "{label}: expected labels LIST, got {actual:?}"
                )));
            };
            let mut actual = actual
                .iter()
                .map(|value| match value {
                    ResultValue::Scalar(ScalarValue::String(value)) => Ok(value.to_string()),
                    value => Err(Error::internal(format!(
                        "{label}: labels LIST contains a non-STRING value {value:?}"
                    ))),
                })
                .collect::<Result<Vec<_>>>()?;
            let mut expected = expected
                .iter()
                .map(|value| (*value).to_owned())
                .collect::<Vec<_>>();
            // Merge1 [10] explicitly ignores element order for the labels list.
            actual.sort();
            expected.sort();
            if actual != expected {
                return Err(Error::internal(format!(
                    "{label}: labels differ: expected {expected:?}, got {actual:?}"
                )));
            }
        }
    }
    Ok(())
}

fn assert_merge1_output(
    case: Merge1Case,
    fixture: &Fixture,
    output: &ExecutionOutput,
) -> Result<()> {
    let label = case.label();
    let expected_schema = vec![(case.output.name().to_owned(), case.output.column_type())];
    if output.result.schema != expected_schema
        || output.result.bookmark != fixture.bookmark
        || output.result.statistics != case.statistics()
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
    let [batch] = output.result.batches.as_slice() else {
        return Err(Error::internal(format!(
            "{label}: expected exactly one result batch, got {:?}",
            output.result.batches
        )));
    };
    let [column] = batch.columns.as_slice() else {
        return Err(Error::internal(format!(
            "{label}: expected exactly one result column, got {batch:?}"
        )));
    };
    if !batch.validate()
        || batch.row_count != 1
        || column.name != case.output.name()
        || column.value_type != case.output.column_type()
        || column.values.len() != 1
    {
        return Err(Error::internal(format!(
            "{label}: malformed result batch: {batch:?}"
        )));
    }
    assert_merge1_result_value(&column.values[0], case.output, &label)?;

    let mut nodes_added = 0_u64;
    let mut relationships_added = 0_u64;
    let mut properties_added = 0_u64;
    let mut labels_assigned = 0_u64;
    let mut label_names_added = 0_u64;
    let mut property_names_added = 0_u64;
    for mutation in &output.graph_mutations {
        match mutation {
            GraphMutation::InsertNode(node) => {
                nodes_added = nodes_added.saturating_add(1);
                properties_added = properties_added.saturating_add(node.properties.len() as u64);
                labels_assigned = labels_assigned.saturating_add(node.labels.len() as u64);
            }
            GraphMutation::InsertEdge(_) => {
                relationships_added = relationships_added.saturating_add(1);
            }
            GraphMutation::DeclareLabel { .. } => {
                label_names_added = label_names_added.saturating_add(1);
            }
            GraphMutation::DeclareProperty { .. } => {
                property_names_added = property_names_added.saturating_add(1);
            }
            mutation => {
                return Err(Error::internal(format!(
                    "{label}: node MERGE emitted an unrelated mutation {mutation:?}"
                )));
            }
        }
    }
    let actual_tck_effects = Merge1TckEffects {
        nodes_added,
        properties_added,
        label_names_added,
    };
    if relationships_added != 0
        || property_names_added != 0
        || labels_assigned != case.labels_assigned
        || actual_tck_effects != case.tck_effects
    {
        return Err(Error::internal(format!(
            "{label}: exact graph effects changed: TCK={actual_tck_effects:?}, assigned_labels={labels_assigned}, relationships={relationships_added}, declared_properties={property_names_added}, output={output:#?}"
        )));
    }
    Ok(())
}

fn merge1_program_label_names(
    program: &irongraph::gpu::ResidentRowCreateProgram,
    labels: &[u16],
) -> Result<BTreeSet<String>> {
    labels
        .iter()
        .map(|label| {
            program
                .label_names
                .get(usize::from(*label))
                .cloned()
                .ok_or_else(|| Error::internal("Merge1 node label slot is out of bounds"))
        })
        .collect()
}

fn assert_exact_merge1_request(
    case: Merge1Case,
    fixture: &Fixture,
    request: &ResidentRowMutationRequest,
) -> Result<()> {
    request.validate()?;
    let label = case.label();
    let Some(body) = request.bound_relationship_merge_body() else {
        return Err(Error::internal(format!(
            "{label}: request did not use the bound-relationship body"
        )));
    };
    let [ResidentRowCreateCommand::CreateNode(node)] = body.program.commands.as_slice() else {
        return Err(Error::internal(format!(
            "{label}: structural command stream is not one CreateNode: {:?}",
            body.program.commands
        )));
    };
    if !matches!(
        body.commands.as_slice(),
        [ResidentRowBoundRelationshipCommand::MergeNode]
    ) || body.program.entity_slot_count != 1
        || node.output_entity != 0
        || !body.program.input_keys.is_empty()
        || !body.program.relationship_type_names.is_empty()
        || !body.program.outputs.is_empty()
        || !request.input.is_empty()
        || body.command_candidate_limits.as_slice() != [1]
        || body.schedule.value_slot_count != 0
        || !body.schedule.property_bindings.is_empty()
        || !matches!(
            body.schedule.stages.as_slice(),
            [ResidentRowBoundRelationshipStage::Command { command: 0 }]
        )
    {
        return Err(Error::internal(format!(
            "{label}: compiler broadened the exact one-command node MERGE profile: {body:#?}"
        )));
    }

    let actual_labels = merge1_program_label_names(&body.program, &node.labels)?;
    let expected_labels = case
        .labels
        .iter()
        .map(|label| (*label).to_owned())
        .collect::<BTreeSet<_>>();
    if node.labels.len() != expected_labels.len() || actual_labels != expected_labels {
        return Err(Error::internal(format!(
            "{label}: node MERGE labels changed: expected {expected_labels:?}, got {actual_labels:?}"
        )));
    }

    match (case.key, node.properties.as_slice()) {
        (None, []) => {}
        (Some((expected_name, expected_value)), [property]) => {
            let actual_name = body
                .program
                .property_names
                .get(usize::from(property.property_name));
            if actual_name.map(String::as_str) != Some(expected_name)
                || !matches!(
                    &property.value,
                    ResidentRowCreateValueInput::Constant(
                        ResidentCreateNodeValueInput::Scalar(ScalarValue::Integer(value))
                    ) if *value == expected_value
                )
            {
                return Err(Error::internal(format!(
                    "{label}: node MERGE constant key changed: {property:?}"
                )));
            }
        }
        _ => {
            return Err(Error::internal(format!(
                "{label}: node MERGE key cardinality changed: {:?}",
                node.properties
            )));
        }
    }

    match (case.output, body.outputs.as_slice()) {
        (
            Merge1ExpectedOutput::Count { name, .. },
            [ResidentRowBoundRelationshipOutput::CountRows { name: actual }],
        ) if actual == name => {}
        (
            Merge1ExpectedOutput::Property { name, property, .. },
            [
                ResidentRowBoundRelationshipOutput::Property {
                    name: actual_name,
                    entity,
                    property_name,
                },
            ],
        ) if actual_name == name
            && *entity == 0
            && body
                .program
                .property_names
                .get(usize::from(*property_name))
                .map(String::as_str)
                == Some(property) => {}
        (
            Merge1ExpectedOutput::Labels { name, .. },
            [
                ResidentRowBoundRelationshipOutput::Entity {
                    name: actual_name,
                    entity: 0,
                },
            ],
        ) if actual_name == name => {}
        _ => {
            return Err(Error::internal(format!(
                "{label}: exact output descriptor changed: {:?}",
                body.outputs
            )));
        }
    }

    let mut expected_property_names = BTreeSet::new();
    if let Some((name, _)) = case.key {
        expected_property_names.insert(name);
    }
    if let Merge1ExpectedOutput::Property { property, .. } = case.output {
        expected_property_names.insert(property);
    }
    let actual_property_names = body
        .program
        .property_names
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    if actual_property_names != expected_property_names {
        return Err(Error::internal(format!(
            "{label}: property-name table changed: expected {expected_property_names:?}, got {actual_property_names:?}"
        )));
    }
    if request.generation.bookmark != fixture.bookmark
        || request.generation.graph_revision != fixture.graph.revision()
    {
        return Err(Error::internal(format!(
            "{label}: request escaped its pinned canonical generation"
        )));
    }
    Ok(())
}

fn assert_one_merge1_command(
    case: Merge1Case,
    fixture: &Fixture,
    observations: &RouteObservations,
) -> Result<ResidentRowMutationRequest> {
    let pins = observations.pins.load(Ordering::SeqCst);
    let commands = observations.complete_commands.load(Ordering::SeqCst);
    let forbidden = observations.forbidden_calls.load(Ordering::SeqCst);
    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if pins != 1 || commands != 1 || forbidden != 0 || requests.len() != 1 {
        return Err(Error::internal(format!(
            "{}: expected one project pin, one complete execute_row_mutation, one request, and no fallback; got pins={pins}, commands={commands}, requests={}, forbidden={forbidden}",
            case.label(),
            requests.len()
        )));
    }
    let request = requests[0].clone();
    assert_exact_merge1_request(case, fixture, &request)?;
    Ok(request)
}

fn run_strict_merge1_case(case: Merge1Case) -> Result<()> {
    let label = case.label();
    let fixture = build_merge1_fixture(case.setup, &label)?;
    let backend = StrictOrderedCreateBackend::strict_cpu(fixture.image()?)?;
    let observations = backend.observations();
    let output = QueryEngine.execute(
        case.query,
        &mut create3_context(&fixture.graph, Some(&backend), true),
    )?;
    assert_one_merge1_command(case, &fixture, &observations)?;
    assert_merge1_output(case, &fixture, &output)
}

#[test]
fn generic_cpu_oracle_pins_exact_merge1_rows_schema_statistics_and_effects() -> Result<()> {
    for case in MERGE1_CASES {
        let label = case.label();
        let fixture = build_merge1_fixture(case.setup, &label)?;
        let output = QueryEngine.execute(
            case.query,
            &mut create3_context(&fixture.graph, None, false),
        )?;
        assert_merge1_output(case, &fixture, &output)?;
    }
    Ok(())
}

#[test]
fn strict_cpu_executes_merge1_586_through_591_and_595_as_one_native_command() -> Result<()> {
    let mut failures = Vec::new();
    for case in MERGE1_CASES {
        if let Err(error) = run_strict_merge1_case(case) {
            failures.push(format!("{}: {error}", case.label()));
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(Error::internal(format!(
            "strict Merge1 constant-node tranche is incomplete:\n{}",
            failures.join("\n")
        )))
    }
}

// The latest full report leaves five Merge2 rows red, but they are not one implementation
// tranche. [2]-[4] are the smallest complete slice: one graph-free node MERGE and one literal
// integer property action which fires only for that command's unique creation leader. [1] needs
// a conditional label-action ABI and metadata rendering; [5] needs dynamic MATCH rows, an RHS
// property binding, and cross-row MERGE leadership. Keep those broader shapes outside this gate.

#[derive(Clone, Copy, Debug)]
struct Merge2LiteralCase {
    report_index: usize,
    scenario: u8,
    name: &'static str,
    setup: &'static str,
    query: &'static str,
    variable: &'static str,
    label_name: Option<&'static str>,
    property_name: &'static str,
    value: i64,
    created: bool,
    returns_property: bool,
}

impl Merge2LiteralCase {
    fn label(self) -> String {
        format!(
            "Merge2 report {} [{}] {}",
            self.report_index, self.scenario, self.name
        )
    }

    fn statistics(self) -> StatementStats {
        StatementStats {
            nodes_created: self.created as u64,
            nodes_deleted: 0,
            relationships_created: 0,
            relationships_deleted: 0,
            properties_set: self.created as u64,
            labels_added: if self.created && self.label_name.is_some() {
                1
            } else {
                0
            },
            labels_removed: 0,
        }
    }
}

const MERGE2_LITERAL_CASES: [Merge2LiteralCase; 3] = [
    Merge2LiteralCase {
        report_index: 604,
        scenario: 2,
        name: "[2] ON CREATE on created nodes",
        setup: "",
        query: "MERGE (b) ON CREATE SET b.created = 1",
        variable: "b",
        label_name: None,
        property_name: "created",
        value: 1,
        created: true,
        returns_property: false,
    },
    Merge2LiteralCase {
        report_index: 605,
        scenario: 3,
        name: "[3] Merge node with label add property on create",
        setup: "",
        query: "MERGE (a:TheLabel) ON CREATE SET a.num = 42 RETURN a.num",
        variable: "a",
        label_name: Some("TheLabel"),
        property_name: "num",
        value: 42,
        created: true,
        returns_property: true,
    },
    Merge2LiteralCase {
        report_index: 606,
        scenario: 4,
        name: "[4] Merge node with label add property on update when it exists",
        setup: "CREATE (:TheLabel)",
        query: "MERGE (a:TheLabel) ON CREATE SET a.num = 42 RETURN a.num",
        variable: "a",
        label_name: Some("TheLabel"),
        property_name: "num",
        value: 42,
        created: false,
        returns_property: true,
    },
];

fn assert_merge2_literal_output(
    case: Merge2LiteralCase,
    fixture: &Fixture,
    output: &ExecutionOutput,
) -> Result<()> {
    let label = case.label();
    if output.result.bookmark != fixture.bookmark
        || output.result.statistics != case.statistics()
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

    if case.returns_property {
        let output_name = format!("{}.{}", case.variable, case.property_name);
        let expected_type = if case.created {
            ColumnType::Integer
        } else {
            ColumnType::Null
        };
        let expected_value = if case.created {
            ScalarValue::Integer(case.value)
        } else {
            ScalarValue::Null
        };
        if output.result.schema != [(output_name.clone(), expected_type.clone())]
            || !matches!(
                output.result.batches.as_slice(),
                [batch]
                    if batch.validate()
                        && batch.row_count == 1
                        && matches!(batch.columns.as_slice(), [column]
                            if column.name == output_name
                                && column.value_type == expected_type
                                && column.values == [ResultValue::Scalar(expected_value.clone())])
            )
        {
            return Err(Error::internal(format!(
                "{label}: conditional property output changed: {:?}",
                output.result
            )));
        }
    } else if !output.result.schema.is_empty() || !output.result.batches.is_empty() {
        return Err(Error::internal(format!(
            "{label}: write-only ON CREATE returned rows"
        )));
    }

    let mut after = fixture.graph.clone();
    for mutation in &output.graph_mutations {
        after.apply(mutation.clone())?;
    }
    if after.edges().next().is_some() {
        return Err(Error::internal(format!(
            "{label}: node ON CREATE emitted a relationship"
        )));
    }
    let nodes = after.nodes().collect::<Vec<_>>();
    let [node] = nodes.as_slice() else {
        return Err(Error::internal(format!(
            "{label}: expected one merged node, got {}",
            nodes.len()
        )));
    };
    let actual_label = case
        .label_name
        .and_then(|name| after.catalog().label(name))
        .is_some_and(|label| node.labels().contains(&label));
    if actual_label != case.label_name.is_some() {
        return Err(Error::internal(format!(
            "{label}: merged-node label changed"
        )));
    }
    let actual_property = after
        .catalog()
        .property(case.property_name)
        .and_then(|property| node.property(property));
    let expected_property = case.created.then(|| ScalarValue::Integer(case.value));
    if actual_property != expected_property {
        return Err(Error::internal(format!(
            "{label}: expected conditional property {expected_property:?}, got {actual_property:?}"
        )));
    }
    Ok(())
}

fn assert_exact_merge2_literal_request(
    case: Merge2LiteralCase,
    fixture: &Fixture,
    request: &ResidentRowMutationRequest,
) -> Result<()> {
    request.validate()?;
    let label = case.label();
    let Some(body) = request.bound_relationship_merge_body() else {
        return Err(Error::internal(format!(
            "{label}: request did not use the sealed ordered-mutation body"
        )));
    };
    let [ResidentRowCreateCommand::CreateNode(node)] = body.program.commands.as_slice() else {
        return Err(Error::internal(format!(
            "{label}: structural stream is not one node command: {:?}",
            body.program.commands
        )));
    };
    let expected_labels = usize::from(case.label_name.is_some());
    if !request.input.is_empty()
        || body.program.entity_slot_count != 1
        || !body.program.input_keys.is_empty()
        || body.program.property_names.as_slice() != [case.property_name]
        || body.program.property_tokens.as_slice() != [None]
        || body.program.label_names.len() != expected_labels
        || body.program.label_tokens.len() != expected_labels
        || case
            .label_name
            .is_some_and(|expected| body.program.label_names.as_slice() != [expected])
        || !body.program.relationship_type_names.is_empty()
        || !body.program.relationship_type_tokens.is_empty()
        || !body.program.outputs.is_empty()
        || body.program.continuation.is_some()
        || node.output_entity != 0
        || match case.label_name {
            None => !node.labels.is_empty(),
            Some(_) => node.labels.as_slice() != [0],
        }
        || !node.properties.is_empty()
        || !matches!(
            body.commands.as_slice(),
            [ResidentRowBoundRelationshipCommand::MergeNode]
        )
        || !matches!(body.command_candidate_limits.as_slice(), [limit] if *limit >= 1)
        || body.schedule.value_slot_count != 0
        || !body.schedule.property_bindings.is_empty()
        || !matches!(
            body.schedule.stages.as_slice(),
            [ResidentRowBoundRelationshipStage::Command { command: 0 }]
        )
    {
        return Err(Error::internal(format!(
            "{label}: compiler broadened the literal node-ON-CREATE profile: {body:#?}"
        )));
    }
    let [action] = body.merge_property_sets.as_slice() else {
        return Err(Error::internal(format!(
            "{label}: expected exactly one conditional property descriptor"
        )));
    };
    if action.branch != ResidentRowBoundMergeBranch::Create
        || action.trigger_command != 0
        || action.target_entity != 0
        || action.property_name != 0
        || !matches!(
            &action.source,
            ResidentRowBoundMergePropertySource::Constant(ScalarValue::Integer(value))
                if *value == case.value
        )
        || action.rhs_obligation.kind != ResidentObligationKind::MutationRhs
        || action.rhs_obligation.scope != ResidentObligationScope::MutationCommand(0)
        || action.effect_obligation.kind != ResidentObligationKind::MutationEffect
        || action.effect_obligation.scope != ResidentObligationScope::MutationCommand(0)
        || action.rhs_obligation.id == action.effect_obligation.id
        || body.capacities.maximum_property_entries != 1
        || body.capacities.maximum_payload_bytes != 0
    {
        return Err(Error::internal(format!(
            "{label}: sealed conditional descriptor changed: {body:#?}"
        )));
    }
    match (case.returns_property, body.outputs.as_slice()) {
        (false, []) => {}
        (
            true,
            [
                ResidentRowBoundRelationshipOutput::Property {
                    name,
                    entity: 0,
                    property_name: 0,
                },
            ],
        ) if name == &format!("{}.{}", case.variable, case.property_name) => {}
        _ => {
            return Err(Error::internal(format!(
                "{label}: final property output changed: {:?}",
                body.outputs
            )));
        }
    }
    if request.generation.bookmark != fixture.bookmark
        || request.generation.graph_revision != fixture.graph.revision()
    {
        return Err(Error::internal(format!(
            "{label}: request escaped its pinned graph generation"
        )));
    }
    Ok(())
}

fn run_strict_merge2_literal_case(case: Merge2LiteralCase) -> Result<()> {
    let label = case.label();
    let fixture = build_merge1_fixture(case.setup, &label)?;
    let backend = StrictOrderedCreateBackend::strict_cpu(fixture.image()?)?;
    let observations = backend.observations();
    let output = QueryEngine.execute(
        case.query,
        &mut create3_context(&fixture.graph, Some(&backend), true),
    )?;
    let pins = observations.pins.load(Ordering::SeqCst);
    let commands = observations.complete_commands.load(Ordering::SeqCst);
    let forbidden = observations.forbidden_calls.load(Ordering::SeqCst);
    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if pins != 1 || commands != 1 || forbidden != 0 || requests.len() != 1 {
        return Err(Error::internal(format!(
            "{label}: expected one pin, one complete command, one request, and zero fallback calls; got {pins}/{commands}/{}/{forbidden}",
            requests.len()
        )));
    }
    let request = requests[0].clone();
    drop(requests);
    assert_exact_merge2_literal_request(case, &fixture, &request)?;
    assert_merge2_literal_output(case, &fixture, &output)
}

#[test]
fn generic_cpu_oracle_pins_merge2_604_through_606_conditional_semantics() -> Result<()> {
    for case in MERGE2_LITERAL_CASES {
        let label = case.label();
        let fixture = build_merge1_fixture(case.setup, &label)?;
        let output = QueryEngine.execute(
            case.query,
            &mut create3_context(&fixture.graph, None, false),
        )?;
        assert_merge2_literal_output(case, &fixture, &output)?;
    }
    Ok(())
}

#[test]
fn strict_cpu_executes_merge2_604_through_606_as_one_native_command() -> Result<()> {
    for case in MERGE2_LITERAL_CASES {
        run_strict_merge2_literal_case(case)?;
    }
    Ok(())
}

// Merge3 has the inverse condition. The narrow first tranche is [3]: a literal integer property
// action over one static node MERGE. The strict gate also runs the same query against an empty
// fixture, proving that the action does not fire on the zero-match create row. [1]/[2] require a
// separate conditional-label action (and [2] metadata rendering), while [4] requires dynamic
// input rows and a bound-property RHS.

#[derive(Clone, Copy, Debug)]
struct Merge3LiteralPropertyCase {
    name: &'static str,
    setup: &'static str,
    matched: bool,
}

impl Merge3LiteralPropertyCase {
    fn label(self) -> String {
        format!(
            "Merge3 report 611 [3] {} ({})",
            self.name,
            if self.matched { "matched" } else { "created" }
        )
    }

    fn statistics(self) -> StatementStats {
        StatementStats {
            nodes_created: u64::from(!self.matched),
            nodes_deleted: 0,
            relationships_created: 0,
            relationships_deleted: 0,
            properties_set: u64::from(self.matched),
            labels_added: u64::from(!self.matched),
            labels_removed: 0,
        }
    }
}

const MERGE3_LITERAL_PROPERTY_CASES: [Merge3LiteralPropertyCase; 2] = [
    Merge3LiteralPropertyCase {
        name: "[3] Merge node and set property on match",
        setup: "CREATE (:TheLabel)",
        matched: true,
    },
    Merge3LiteralPropertyCase {
        name: "conditional non-fire control",
        setup: "",
        matched: false,
    },
];

const MERGE3_LITERAL_PROPERTY_QUERY: &str =
    "MERGE (a:TheLabel) ON MATCH SET a.num = 42 RETURN a.num";

fn assert_merge3_literal_property_output(
    case: Merge3LiteralPropertyCase,
    fixture: &Fixture,
    output: &ExecutionOutput,
) -> Result<()> {
    let label = case.label();
    let expected_type = if case.matched {
        ColumnType::Integer
    } else {
        ColumnType::Null
    };
    let expected_value = if case.matched {
        ScalarValue::Integer(42)
    } else {
        ScalarValue::Null
    };
    if output.result.bookmark != fixture.bookmark
        || output.result.statistics != case.statistics()
        || output.result.schema != [("a.num".to_owned(), expected_type.clone())]
        || output.result.truncated
        || !output.temporal_mutations.is_empty()
        || output.administrative.is_some()
        || !output.vector_searches.is_empty()
        || output.runtime_replans != 0
        || !matches!(
            output.result.batches.as_slice(),
            [batch]
                if batch.validate()
                    && batch.row_count == 1
                    && matches!(batch.columns.as_slice(), [column]
                        if column.name == "a.num"
                            && column.value_type == expected_type
                            && column.values == [ResultValue::Scalar(expected_value.clone())])
        )
    {
        return Err(Error::internal(format!(
            "{label}: literal ON MATCH output changed: {output:#?}"
        )));
    }

    let mut after = fixture.graph.clone();
    for mutation in &output.graph_mutations {
        after.apply(mutation.clone())?;
    }
    if after.edges().next().is_some() {
        return Err(Error::internal(format!(
            "{label}: node ON MATCH emitted a relationship"
        )));
    }
    let nodes = after.nodes().collect::<Vec<_>>();
    let [node] = nodes.as_slice() else {
        return Err(Error::internal(format!(
            "{label}: expected one merged node, got {}",
            nodes.len()
        )));
    };
    let the_label = after
        .catalog()
        .label("TheLabel")
        .ok_or_else(|| Error::internal(format!("{label}: TheLabel disappeared")))?;
    if !node.labels().contains(&the_label) {
        return Err(Error::internal(format!(
            "{label}: merged node lost TheLabel"
        )));
    }
    let actual_property = after
        .catalog()
        .property("num")
        .and_then(|property| node.property(property));
    let expected_property = case.matched.then_some(ScalarValue::Integer(42));
    if actual_property != expected_property {
        return Err(Error::internal(format!(
            "{label}: expected ON MATCH property {expected_property:?}, got {actual_property:?}"
        )));
    }
    Ok(())
}

fn assert_exact_merge3_literal_property_request(
    case: Merge3LiteralPropertyCase,
    fixture: &Fixture,
    request: &ResidentRowMutationRequest,
) -> Result<()> {
    request.validate()?;
    let label = case.label();
    let Some(body) = request.bound_relationship_merge_body() else {
        return Err(Error::internal(format!(
            "{label}: request did not use the sealed ordered-mutation body"
        )));
    };
    let [ResidentRowCreateCommand::CreateNode(node)] = body.program.commands.as_slice() else {
        return Err(Error::internal(format!(
            "{label}: structural stream is not one node command: {:?}",
            body.program.commands
        )));
    };
    if !request.input.is_empty()
        || body.program.entity_slot_count != 1
        || !body.program.input_keys.is_empty()
        || body.program.property_names.as_slice() != ["num"]
        || body.program.property_tokens.as_slice() != [None]
        || body.program.label_names.as_slice() != ["TheLabel"]
        || body.program.label_tokens.len() != 1
        || !body.program.relationship_type_names.is_empty()
        || !body.program.relationship_type_tokens.is_empty()
        || !body.program.outputs.is_empty()
        || body.program.continuation.is_some()
        || node.output_entity != 0
        || node.labels.as_slice() != [0]
        || !node.properties.is_empty()
        || !matches!(
            body.commands.as_slice(),
            [ResidentRowBoundRelationshipCommand::MergeNode]
        )
        || !matches!(body.command_candidate_limits.as_slice(), [limit] if *limit >= 1)
        || body.schedule.value_slot_count != 0
        || !body.schedule.property_bindings.is_empty()
        || !matches!(
            body.schedule.stages.as_slice(),
            [ResidentRowBoundRelationshipStage::Command { command: 0 }]
        )
        || !matches!(
            body.merge_property_sets.as_slice(),
            [action]
                if action.branch == ResidentRowBoundMergeBranch::Match
                    && action.trigger_command == 0
                    && action.target_entity == 0
                    && action.property_name == 0
                    && matches!(
                        &action.source,
                        ResidentRowBoundMergePropertySource::Constant(
                            ScalarValue::Integer(42)
                        )
                    )
                    && action.rhs_obligation.kind == ResidentObligationKind::MutationRhs
                    && action.rhs_obligation.scope
                        == ResidentObligationScope::MutationCommand(0)
                    && action.effect_obligation.kind
                        == ResidentObligationKind::MutationEffect
                    && action.effect_obligation.scope
                        == ResidentObligationScope::MutationCommand(0)
                    && action.rhs_obligation.id != action.effect_obligation.id
        )
        || body.capacities.maximum_property_entries != 1
        || body.capacities.maximum_payload_bytes != 0
        || !matches!(
            body.outputs.as_slice(),
            [ResidentRowBoundRelationshipOutput::Property {
                name,
                entity: 0,
                property_name: 0,
            }] if name == "a.num"
        )
    {
        return Err(Error::internal(format!(
            "{label}: compiler broadened the literal node-ON-MATCH profile: {body:#?}"
        )));
    }
    if request.generation.bookmark != fixture.bookmark
        || request.generation.graph_revision != fixture.graph.revision()
    {
        return Err(Error::internal(format!(
            "{label}: request escaped its pinned graph generation"
        )));
    }
    Ok(())
}

fn run_strict_merge3_literal_property_case(case: Merge3LiteralPropertyCase) -> Result<()> {
    let label = case.label();
    let fixture = build_merge1_fixture(case.setup, &label)?;
    let backend = StrictOrderedCreateBackend::strict_cpu(fixture.image()?)?;
    let observations = backend.observations();
    let output = QueryEngine.execute(
        MERGE3_LITERAL_PROPERTY_QUERY,
        &mut create3_context(&fixture.graph, Some(&backend), true),
    )?;
    let pins = observations.pins.load(Ordering::SeqCst);
    let commands = observations.complete_commands.load(Ordering::SeqCst);
    let forbidden = observations.forbidden_calls.load(Ordering::SeqCst);
    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if pins != 1 || commands != 1 || forbidden != 0 || requests.len() != 1 {
        return Err(Error::internal(format!(
            "{label}: expected one pin, one complete command, one request, and zero fallback calls; got {pins}/{commands}/{}/{forbidden}",
            requests.len()
        )));
    }
    let request = requests[0].clone();
    drop(requests);
    assert_exact_merge3_literal_property_request(case, &fixture, &request)?;
    assert_merge3_literal_property_output(case, &fixture, &output)
}

#[test]
fn generic_cpu_oracle_pins_merge3_611_match_and_non_match_semantics() -> Result<()> {
    for case in MERGE3_LITERAL_PROPERTY_CASES {
        let label = case.label();
        let fixture = build_merge1_fixture(case.setup, &label)?;
        let output = QueryEngine.execute(
            MERGE3_LITERAL_PROPERTY_QUERY,
            &mut create3_context(&fixture.graph, None, false),
        )?;
        assert_merge3_literal_property_output(case, &fixture, &output)?;
    }
    Ok(())
}

#[test]
fn strict_cpu_executes_merge3_611_as_one_native_command() -> Result<()> {
    for case in MERGE3_LITERAL_PROPERTY_CASES {
        run_strict_merge3_literal_property_case(case)?;
    }
    Ok(())
}

// Merge1 [7], [8], and [14] are the remaining static/profile-only node-MERGE shapes. Keep their
// route contract beside the established Merge1 backend so this tranche reuses the same poisoned
// fallback surface instead of introducing a second permissive test adapter.

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Merge1StaticProfile {
    OrderedCreateThenMultiMatch,
    ConstantWithArgument,
    DeleteThenMerge,
}

#[derive(Clone, Copy, Debug)]
struct Merge1StaticCase {
    report_index: usize,
    scenario: u8,
    name: &'static str,
    setup: &'static str,
    query: &'static str,
    profile: Merge1StaticProfile,
    statistics: StatementStats,
}

impl Merge1StaticCase {
    fn label(self) -> String {
        format!(
            "Merge1 report {} [{}] {}",
            self.report_index, self.scenario, self.name
        )
    }
}

const MERGE1_STATIC_CASES: [Merge1StaticCase; 3] = [
    Merge1StaticCase {
        report_index: 592,
        scenario: 7,
        name: "[7] Merge should work when finding multiple elements",
        setup: "",
        query: "CREATE (:X) CREATE (:X) MERGE (:X)",
        profile: Merge1StaticProfile::OrderedCreateThenMultiMatch,
        statistics: StatementStats {
            nodes_created: 2,
            nodes_deleted: 0,
            relationships_created: 0,
            relationships_deleted: 0,
            properties_set: 0,
            labels_added: 2,
            labels_removed: 0,
        },
    },
    Merge1StaticCase {
        report_index: 593,
        scenario: 8,
        name: "[8] Merge should handle argument properly",
        setup: "CREATE ({var: 42}), ({var: 'not42'})",
        query: "WITH 42 AS var MERGE (c:N {var: var})",
        profile: Merge1StaticProfile::ConstantWithArgument,
        statistics: StatementStats {
            nodes_created: 1,
            nodes_deleted: 0,
            relationships_created: 0,
            relationships_deleted: 0,
            properties_set: 0,
            labels_added: 1,
            labels_removed: 0,
        },
    },
    Merge1StaticCase {
        report_index: 599,
        scenario: 14,
        name: "[14] Merges should not be able to match on deleted nodes",
        setup: "CREATE (:A {num: 1}), (:A {num: 2})",
        query: "MATCH (a:A) DELETE a MERGE (a2:A) RETURN a2.num",
        profile: Merge1StaticProfile::DeleteThenMerge,
        statistics: StatementStats {
            nodes_created: 1,
            nodes_deleted: 2,
            relationships_created: 0,
            relationships_deleted: 0,
            properties_set: 0,
            labels_added: 1,
            labels_removed: 0,
        },
    },
];

fn assert_merge1_static_output(
    case: Merge1StaticCase,
    fixture: &Fixture,
    output: &ExecutionOutput,
) -> Result<()> {
    let label = case.label();
    let expected_schema = match case.profile {
        Merge1StaticProfile::OrderedCreateThenMultiMatch
        | Merge1StaticProfile::ConstantWithArgument => Vec::new(),
        Merge1StaticProfile::DeleteThenMerge => {
            vec![("a2.num".to_owned(), ColumnType::Null)]
        }
    };
    if output.result.schema != expected_schema
        || output.result.bookmark != fixture.bookmark
        || output.result.statistics != case.statistics
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

    let expected_rows = if case.profile == Merge1StaticProfile::DeleteThenMerge {
        2
    } else {
        0
    };
    let mut actual_rows = 0_usize;
    for batch in &output.result.batches {
        if !batch.validate() || batch.columns.len() != expected_schema.len() {
            return Err(Error::internal(format!(
                "{label}: malformed result batch: {batch:?}"
            )));
        }
        if case.profile == Merge1StaticProfile::DeleteThenMerge {
            let [column] = batch.columns.as_slice() else {
                return Err(Error::internal(format!(
                    "{label}: DELETE/MERGE output lost its only column"
                )));
            };
            if column.name != "a2.num"
                || column.value_type != ColumnType::Null
                || column
                    .values
                    .iter()
                    .any(|value| value != &ResultValue::Scalar(ScalarValue::Null))
            {
                return Err(Error::internal(format!(
                    "{label}: DELETE/MERGE output changed: {column:?}"
                )));
            }
        }
        actual_rows = actual_rows.saturating_add(batch.row_count);
    }
    if actual_rows != expected_rows {
        return Err(Error::internal(format!(
            "{label}: expected {expected_rows} rows, got {actual_rows}"
        )));
    }

    let mut nodes_added = 0_u64;
    let mut nodes_deleted = 0_u64;
    let mut properties_deleted = 0_u64;
    let mut relationships_changed = 0_u64;
    let mut properties_added = 0_u64;
    let mut labels_assigned = 0_u64;
    let mut label_names_added = 0_u64;
    let mut property_names_added = 0_u64;
    for mutation in &output.graph_mutations {
        match mutation {
            GraphMutation::InsertNode(node) => {
                nodes_added = nodes_added.saturating_add(1);
                properties_added = properties_added.saturating_add(node.properties.len() as u64);
                labels_assigned = labels_assigned.saturating_add(node.labels.len() as u64);
            }
            GraphMutation::DeleteNode { node, .. } => {
                nodes_deleted = nodes_deleted.saturating_add(1);
                let deleted = fixture.graph.node(*node).ok_or_else(|| {
                    Error::internal(format!(
                        "{label}: DELETE targeted a node outside the pinned fixture"
                    ))
                })?;
                properties_deleted =
                    properties_deleted.saturating_add(deleted.properties().len() as u64);
            }
            GraphMutation::DeclareLabel { .. } => {
                label_names_added = label_names_added.saturating_add(1);
            }
            GraphMutation::DeclareProperty { .. } => {
                property_names_added = property_names_added.saturating_add(1);
            }
            GraphMutation::InsertEdge(_)
            | GraphMutation::DeleteEdge { .. }
            | GraphMutation::DeclareRelationshipType { .. }
            | GraphMutation::SetNodeProperty { .. }
            | GraphMutation::AddNodeLabels { .. }
            | GraphMutation::RemoveNodeLabels { .. }
            | GraphMutation::SetEdgeProperty { .. } => {
                relationships_changed = relationships_changed.saturating_add(1);
            }
        }
    }
    let expected_effects = match case.profile {
        Merge1StaticProfile::OrderedCreateThenMultiMatch => (2, 0, 0, 0, 2, 1, 0),
        Merge1StaticProfile::ConstantWithArgument => (1, 0, 0, 1, 1, 1, 0),
        Merge1StaticProfile::DeleteThenMerge => (1, 2, 2, 0, 1, 0, 0),
    };
    let actual_effects = (
        nodes_added,
        nodes_deleted,
        properties_deleted,
        properties_added,
        labels_assigned,
        label_names_added,
        property_names_added,
    );
    if actual_effects != expected_effects || relationships_changed != 0 {
        return Err(Error::internal(format!(
            "{label}: exact graph effects changed: expected {expected_effects:?}, got {actual_effects:?}, unrelated={relationships_changed}, output={output:#?}"
        )));
    }
    Ok(())
}

fn assert_exact_merge1_static_request(
    case: Merge1StaticCase,
    fixture: &Fixture,
    request: &ResidentRowMutationRequest,
) -> Result<()> {
    request.validate()?;
    let label = case.label();
    let Some(body) = request.bound_relationship_merge_body() else {
        return Err(Error::internal(format!(
            "{label}: request did not use the sealed bound-relationship body"
        )));
    };
    if !request.input.is_empty()
        || !body.program.input_keys.is_empty()
        || !body.program.relationship_type_names.is_empty()
        || !body.program.outputs.is_empty()
        || body.schedule.value_slot_count != 0
        || !body.schedule.property_bindings.is_empty()
        || request.generation.bookmark != fixture.bookmark
        || request.generation.graph_revision != fixture.graph.revision()
    {
        return Err(Error::internal(format!(
            "{label}: request escaped its exact static generation/profile: {body:#?}"
        )));
    }

    let exact = match case.profile {
        Merge1StaticProfile::OrderedCreateThenMultiMatch => {
            body.program.entity_slot_count == 3
                && body.program.label_names.as_slice() == ["X"]
                && body.program.property_names.is_empty()
                && body.outputs.is_empty()
                && body.command_candidate_limits.as_slice() == [1, 1, 2]
                && matches!(
                    (body.program.commands.as_slice(), body.commands.as_slice()),
                    (
                        [
                            ResidentRowCreateCommand::CreateNode(first),
                            ResidentRowCreateCommand::CreateNode(second),
                            ResidentRowCreateCommand::CreateNode(merged),
                        ],
                        [
                            ResidentRowBoundRelationshipCommand::CreateNode,
                            ResidentRowBoundRelationshipCommand::CreateNode,
                            ResidentRowBoundRelationshipCommand::MergeNode,
                        ],
                    ) if first.output_entity == 0
                        && first.labels.as_slice() == [0]
                        && first.properties.is_empty()
                        && second.output_entity == 1
                        && second.labels.as_slice() == [0]
                        && second.properties.is_empty()
                        && merged.output_entity == 2
                        && merged.labels.as_slice() == [0]
                        && merged.properties.is_empty()
                )
                && matches!(
                    body.schedule.stages.as_slice(),
                    [
                        ResidentRowBoundRelationshipStage::Command { command: 0 },
                        ResidentRowBoundRelationshipStage::Command { command: 1 },
                        ResidentRowBoundRelationshipStage::Command { command: 2 },
                    ]
                )
        }
        Merge1StaticProfile::ConstantWithArgument => {
            body.program.entity_slot_count == 1
                && body.program.label_names.as_slice() == ["N"]
                && body.program.property_names.as_slice() == ["var"]
                && body.outputs.is_empty()
                && body.command_candidate_limits.as_slice() == [1]
                && matches!(
                    (body.program.commands.as_slice(), body.commands.as_slice()),
                    (
                        [ResidentRowCreateCommand::CreateNode(merged)],
                        [ResidentRowBoundRelationshipCommand::MergeNode],
                    ) if merged.output_entity == 0
                        && merged.labels.as_slice() == [0]
                        && matches!(merged.properties.as_slice(), [property]
                            if property.property_name == 0
                                && matches!(&property.value,
                                    ResidentRowCreateValueInput::Constant(
                                        ResidentCreateNodeValueInput::Scalar(
                                            ScalarValue::Integer(42)
                                        )
                                    )))
                )
                && matches!(
                    body.schedule.stages.as_slice(),
                    [ResidentRowBoundRelationshipStage::Command { command: 0 }]
                )
        }
        Merge1StaticProfile::DeleteThenMerge => {
            body.program.entity_slot_count == 2
                && body.program.label_names.as_slice() == ["A"]
                && body.program.property_names.as_slice() == ["num"]
                && body.command_candidate_limits.as_slice() == [2, 2]
                && matches!(
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
                        && matched.labels.as_slice() == [0]
                        && properties.is_empty()
                        && merged.output_entity == 1
                        && merged.labels.as_slice() == [0]
                        && merged.properties.is_empty()
                )
                && matches!(
                    body.schedule.stages.as_slice(),
                    [
                        ResidentRowBoundRelationshipStage::Command { command: 0 },
                        ResidentRowBoundRelationshipStage::Delete {
                            detach: false,
                            targets,
                            ..
                        },
                        ResidentRowBoundRelationshipStage::Command { command: 1 },
                    ] if matches!(targets.as_slice(), [target] if target.entity == 0)
                )
                && matches!(
                    body.outputs.as_slice(),
                    [ResidentRowBoundRelationshipOutput::Property {
                        name,
                        entity: 1,
                        property_name: 0,
                    }] if name == "a2.num"
                )
        }
    };
    if !exact {
        return Err(Error::internal(format!(
            "{label}: exact static/profile request changed: {body:#?}"
        )));
    }
    Ok(())
}

fn run_strict_merge1_static_case(case: Merge1StaticCase) -> Result<()> {
    let label = case.label();
    let fixture = build_merge1_fixture(case.setup, &label)?;
    let backend = StrictOrderedCreateBackend::strict_cpu(fixture.image()?)?;
    let observations = backend.observations();
    let output = QueryEngine.execute(
        case.query,
        &mut create3_context(&fixture.graph, Some(&backend), true),
    )?;
    let pins = observations.pins.load(Ordering::SeqCst);
    let commands = observations.complete_commands.load(Ordering::SeqCst);
    let forbidden = observations.forbidden_calls.load(Ordering::SeqCst);
    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if pins != 1 || commands != 1 || forbidden != 0 || requests.len() != 1 {
        return Err(Error::internal(format!(
            "{label}: expected one project pin, one complete row-mutation command, one request, and no fallback; got pins={pins}, commands={commands}, requests={}, forbidden={forbidden}",
            requests.len()
        )));
    }
    let request = requests[0].clone();
    drop(requests);
    assert_exact_merge1_static_request(case, &fixture, &request)?;
    assert_merge1_static_output(case, &fixture, &output)
}

#[test]
fn generic_cpu_oracle_pins_merge1_static_profile_rows_statistics_and_effects() -> Result<()> {
    for case in MERGE1_STATIC_CASES {
        let label = case.label();
        let fixture = build_merge1_fixture(case.setup, &label)?;
        let output = QueryEngine.execute(
            case.query,
            &mut create3_context(&fixture.graph, None, false),
        )?;
        assert_merge1_static_output(case, &fixture, &output)?;
    }
    Ok(())
}

#[test]
fn strict_cpu_executes_merge1_592_593_and_599_as_one_native_command() -> Result<()> {
    let mut failures = Vec::new();
    for case in MERGE1_STATIC_CASES {
        if let Err(error) = run_strict_merge1_static_case(case) {
            failures.push(format!("{}: {error}", case.label()));
        }
    }
    if failures.is_empty() {
        Ok(())
    } else {
        Err(Error::internal(format!(
            "strict Merge1 static/profile tranche is incomplete:\n{}",
            failures.join("\n")
        )))
    }
}

// The authoritative 194-gap report still lists seven Merge1 rows. The executable strict gates
// above already prove [7], [8], and [14] in the current tree, leaving exactly [9], [11], [12], and
// [13]. These four are not one backend shape: [9] is a three-command arithmetic/read-own-writes
// program, [11] and [12] are dynamic entity-property key feeds with different provenance, and
// [13] is the existing static one-node MERGE plus a node-only path render. Seal [13] first.

const MERGE1_NODE_PATH_QUERY: &str = "MERGE p = (a {num: 1}) RETURN p";

fn assert_merge1_node_path_output(fixture: &Fixture, output: &ExecutionOutput) -> Result<()> {
    const LABEL: &str = "Merge1 report 598 [13] node-only path";
    if output.result.bookmark != fixture.bookmark
        || output.result.statistics
            != (StatementStats {
                nodes_created: 1,
                nodes_deleted: 0,
                relationships_created: 0,
                relationships_deleted: 0,
                properties_set: 0,
                labels_added: 0,
                labels_removed: 0,
            })
        || output.result.schema != [("p".to_owned(), ColumnType::Path)]
        || output.result.truncated
        || !output.temporal_mutations.is_empty()
        || output.administrative.is_some()
        || !output.vector_searches.is_empty()
        || output.runtime_replans != 0
    {
        return Err(Error::internal(format!(
            "{LABEL}: result envelope changed: {output:#?}"
        )));
    }
    let [batch] = output.result.batches.as_slice() else {
        return Err(Error::internal(format!(
            "{LABEL}: expected one result batch"
        )));
    };
    let [column] = batch.columns.as_slice() else {
        return Err(Error::internal(format!(
            "{LABEL}: expected one result column"
        )));
    };
    let [
        ResultValue::Path {
            nodes,
            relationships,
        },
    ] = column.values.as_slice()
    else {
        return Err(Error::internal(format!(
            "{LABEL}: output is not one path: {column:?}"
        )));
    };
    let [path_node] = nodes.as_slice() else {
        return Err(Error::internal(format!(
            "{LABEL}: path must contain exactly one node"
        )));
    };
    if !batch.validate()
        || batch.row_count != 1
        || column.name != "p"
        || column.value_type != ColumnType::Path
        || !relationships.is_empty()
        || !path_node.labels.is_empty()
        || path_node.properties != BTreeMap::from([("num".to_owned(), ScalarValue::Integer(1))])
    {
        return Err(Error::internal(format!(
            "{LABEL}: node-only path shape changed: {batch:?}"
        )));
    }

    let mut after = fixture.graph.clone();
    for mutation in &output.graph_mutations {
        after.apply(mutation.clone())?;
    }
    if after.edges().next().is_some() {
        return Err(Error::internal(format!(
            "{LABEL}: node-only path emitted a relationship"
        )));
    }
    let graph_nodes = after.nodes().collect::<Vec<_>>();
    let [graph_node] = graph_nodes.as_slice() else {
        return Err(Error::internal(format!(
            "{LABEL}: expected one canonical node, got {}",
            graph_nodes.len()
        )));
    };
    let property = after
        .catalog()
        .property("num")
        .ok_or_else(|| Error::internal(format!("{LABEL}: num token disappeared")))?;
    if graph_node.id() != path_node.id
        || graph_node.property(property) != Some(ScalarValue::Integer(1))
    {
        return Err(Error::internal(format!(
            "{LABEL}: rendered path disagrees with the canonical created node"
        )));
    }
    Ok(())
}

fn assert_exact_merge1_node_path_request(
    fixture: &Fixture,
    request: &ResidentRowMutationRequest,
) -> Result<()> {
    const LABEL: &str = "Merge1 report 598 [13] node-only path";
    request.validate()?;
    let Some(body) = request.bound_relationship_merge_body() else {
        return Err(Error::internal(format!(
            "{LABEL}: request did not use the sealed ordered-mutation body"
        )));
    };
    let [ResidentRowCreateCommand::CreateNode(node)] = body.program.commands.as_slice() else {
        return Err(Error::internal(format!(
            "{LABEL}: structural stream is not one node command: {:?}",
            body.program.commands
        )));
    };
    if !request.input.is_empty()
        || body.program.entity_slot_count != 1
        || !body.program.input_keys.is_empty()
        || body.program.property_names.as_slice() != ["num"]
        || body.program.property_tokens.as_slice() != [None]
        || !body.program.label_names.is_empty()
        || !body.program.label_tokens.is_empty()
        || !body.program.relationship_type_names.is_empty()
        || !body.program.relationship_type_tokens.is_empty()
        || !body.program.outputs.is_empty()
        || body.program.continuation.is_some()
        || node.output_entity != 0
        || !node.labels.is_empty()
        || !matches!(
            node.properties.as_slice(),
            [property]
                if property.property_name == 0
                    && matches!(
                        &property.value,
                        ResidentRowCreateValueInput::Constant(
                            ResidentCreateNodeValueInput::Scalar(ScalarValue::Integer(1))
                        )
                    )
        )
        || !matches!(
            body.commands.as_slice(),
            [ResidentRowBoundRelationshipCommand::MergeNode]
        )
        || !matches!(body.command_candidate_limits.as_slice(), [limit] if *limit >= 1)
        || body.schedule.value_slot_count != 0
        || !body.schedule.property_bindings.is_empty()
        || !matches!(
            body.schedule.stages.as_slice(),
            [ResidentRowBoundRelationshipStage::Command { command: 0 }]
        )
        || !body.merge_property_sets.is_empty()
        || !matches!(
            body.outputs.as_slice(),
            [ResidentRowBoundRelationshipOutput::Entity {
                name,
                entity: 0,
            }] if name == "p"
        )
    {
        return Err(Error::internal(format!(
            "{LABEL}: compiler broadened the exact node-only path profile: {body:#?}"
        )));
    }
    if request.generation.bookmark != fixture.bookmark
        || request.generation.graph_revision != fixture.graph.revision()
    {
        return Err(Error::internal(format!(
            "{LABEL}: request escaped its pinned graph generation"
        )));
    }
    Ok(())
}

fn run_strict_merge1_node_path() -> Result<()> {
    const LABEL: &str = "Merge1 report 598 [13] node-only path";
    let fixture = build_merge1_fixture("", LABEL)?;
    let backend = StrictOrderedCreateBackend::strict_cpu(fixture.image()?)?;
    let observations = backend.observations();
    let output = QueryEngine.execute(
        MERGE1_NODE_PATH_QUERY,
        &mut create3_context(&fixture.graph, Some(&backend), true),
    )?;
    let pins = observations.pins.load(Ordering::SeqCst);
    let commands = observations.complete_commands.load(Ordering::SeqCst);
    let forbidden = observations.forbidden_calls.load(Ordering::SeqCst);
    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if pins != 1 || commands != 1 || forbidden != 0 || requests.len() != 1 {
        return Err(Error::internal(format!(
            "{LABEL}: expected one pin, one complete command, one request, and zero fallback calls; got {pins}/{commands}/{}/{forbidden}",
            requests.len()
        )));
    }
    let request = requests[0].clone();
    drop(requests);
    assert_exact_merge1_node_path_request(&fixture, &request)?;
    assert_merge1_node_path_output(&fixture, &output)
}

#[test]
fn generic_cpu_oracle_pins_merge1_598_node_only_path() -> Result<()> {
    const LABEL: &str = "Merge1 report 598 [13] node-only path";
    let fixture = build_merge1_fixture("", LABEL)?;
    let output = QueryEngine.execute(
        MERGE1_NODE_PATH_QUERY,
        &mut create3_context(&fixture.graph, None, false),
    )?;
    assert_merge1_node_path_output(&fixture, &output)
}

#[test]
fn strict_cpu_executes_merge1_598_as_one_native_node_path_command() -> Result<()> {
    run_strict_merge1_node_path()
}

#[derive(Clone, Copy, Debug)]
struct Merge1FailClosedCase {
    name: &'static str,
    setup: &'static str,
    query: &'static str,
}

const MERGE1_FAIL_CLOSED_CASES: [Merge1FailClosedCase; 3] = [
    Merge1FailClosedCase {
        name: "anonymous node MERGE",
        setup: "",
        query: "MERGE ()",
    },
    Merge1FailClosedCase {
        name: "two node MERGE commands",
        setup: "",
        query: "MERGE (a) MERGE (b) RETURN count(*)",
    },
    Merge1FailClosedCase {
        name: "ordered CREATE then MERGE",
        setup: "",
        query: "CREATE () MERGE (a)",
    },
];

#[derive(Clone, Copy)]
enum Merge1DynamicExpectedValue {
    Integer(i64),
    String(&'static str),
}

impl Merge1DynamicExpectedValue {
    fn scalar(self) -> ScalarValue {
        match self {
            Self::Integer(value) => ScalarValue::Integer(value),
            Self::String(value) => ScalarValue::String(value.into()),
        }
    }
}

#[derive(Clone, Copy)]
struct Merge1DynamicEntityKeyCase {
    label: &'static str,
    setup: &'static str,
    query: &'static str,
    expected_nodes_created: u64,
    expected_labels_added: u64,
    expected_values: &'static [Merge1DynamicExpectedValue],
}

const MERGE1_DYNAMIC_ENTITY_KEY_CASES: [Merge1DynamicEntityKeyCase; 2] = [
    Merge1DynamicEntityKeyCase {
        label: "Merge1 report 596 [11] bound-node dynamic key",
        setup: "CREATE (:Person {bornIn: 'New York'}), (:Person {bornIn: 'New York'}), \
                (:Person {bornIn: 'Ohio'}), (:Person {bornIn: 'Ohio'}), \
                (:Person {bornIn: 'New Jersey'}), (:Person {bornIn: 'New Jersey'})",
        query: "MATCH (person:Person) MERGE (city:City {name: person.bornIn})",
        expected_nodes_created: 3,
        expected_labels_added: 3,
        expected_values: &[
            Merge1DynamicExpectedValue::String("New Jersey"),
            Merge1DynamicExpectedValue::String("New York"),
            Merge1DynamicExpectedValue::String("Ohio"),
        ],
    },
    Merge1DynamicEntityKeyCase {
        label: "Merge1 report 597 [12] freshly-created-node dynamic key",
        setup: "",
        query: "CREATE (a {num: 1}) MERGE ({v: a.num})",
        expected_nodes_created: 2,
        expected_labels_added: 0,
        expected_values: &[
            Merge1DynamicExpectedValue::Integer(1),
            Merge1DynamicExpectedValue::Integer(1),
        ],
    },
];

fn run_merge1_dynamic_entity_key_case_with_backend(
    case: Merge1DynamicEntityKeyCase,
    fixture: &Fixture,
    backend: StrictOrderedCreateBackend,
) -> Result<()> {
    let observations = backend.observations();
    let output = QueryEngine.execute(
        case.query,
        &mut create3_context(&fixture.graph, Some(&backend), true),
    )?;
    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if observations.pins.load(Ordering::SeqCst) != 1
        || observations.complete_commands.load(Ordering::SeqCst) != 1
        || observations.forbidden_calls.load(Ordering::SeqCst) != 0
        || requests.len() != 1
    {
        return Err(Error::internal(format!(
            "{}: dynamic-key MERGE escaped the one-command native route",
            case.label
        )));
    }
    let request = &requests[0];
    request.validate()?;
    let Some(body) = request.bound_relationship_merge_body() else {
        return Err(Error::internal(format!(
            "{}: dynamic-key MERGE lost its bound row-mutation body",
            case.label
        )));
    };
    if body.schedule.property_bindings.is_empty()
        || body.schedule.property_bindings.iter().any(|binding| {
            !matches!(
                binding.source,
                ResidentRowBoundPropertyBindingSource::EntityProperty { .. }
            )
        })
    {
        return Err(Error::internal(format!(
            "{}: dynamic-key MERGE did not preserve entity-property provenance: {body:#?}",
            case.label
        )));
    }
    drop(requests);

    let expected_statistics = StatementStats {
        nodes_created: case.expected_nodes_created,
        nodes_deleted: 0,
        relationships_created: 0,
        relationships_deleted: 0,
        properties_set: 0,
        labels_added: case.expected_labels_added,
        labels_removed: 0,
    };
    if !output.result.schema.is_empty()
        || !output.result.batches.is_empty()
        || output.result.statistics != expected_statistics
        || output.result.bookmark != fixture.bookmark
        || output.result.truncated
        || !output.temporal_mutations.is_empty()
        || output.administrative.is_some()
        || !output.vector_searches.is_empty()
        || output.runtime_replans != 0
    {
        return Err(Error::internal(format!(
            "{}: dynamic-key result envelope changed: {output:#?}",
            case.label
        )));
    }
    let mut inserted_values = output
        .graph_mutations
        .iter()
        .filter_map(|mutation| match mutation {
            GraphMutation::InsertNode(node) => Some(
                node.properties
                    .iter()
                    .map(|(_, value)| value.clone())
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        })
        .flatten()
        .collect::<Vec<_>>();
    inserted_values.sort_by(|left, right| format!("{left:?}").cmp(&format!("{right:?}")));
    let mut expected_values = case
        .expected_values
        .iter()
        .map(|value| value.scalar())
        .collect::<Vec<_>>();
    expected_values.sort_by(|left, right| format!("{left:?}").cmp(&format!("{right:?}")));
    if inserted_values != expected_values {
        return Err(Error::internal(format!(
            "{}: dynamic-key node values differ: expected {expected_values:?}, got {inserted_values:?}",
            case.label
        )));
    }
    Ok(())
}

fn run_strict_merge1_dynamic_entity_key_case(case: Merge1DynamicEntityKeyCase) -> Result<()> {
    let fixture = build_merge1_fixture(case.setup, case.label)?;
    let backend = StrictOrderedCreateBackend::strict_cpu(fixture.image()?)?;
    run_merge1_dynamic_entity_key_case_with_backend(case, &fixture, backend)
}

#[test]
fn strict_cpu_executes_merge1_596_and_597_dynamic_entity_keys() -> Result<()> {
    for case in MERGE1_DYNAMIC_ENTITY_KEY_CASES {
        run_strict_merge1_dynamic_entity_key_case(case)?;
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires a real Metal device; proves per-key grouping for Merge1 [11] and [12]"]
fn real_metal_executes_merge1_596_and_597_dynamic_entity_keys() -> Result<()> {
    for case in MERGE1_DYNAMIC_ENTITY_KEY_CASES {
        let fixture = build_merge1_fixture(case.setup, case.label)?;
        let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
        metal.admit_project(fixture.image()?)?;
        let backend = StrictOrderedCreateBackend::real_metal(metal)?;
        run_merge1_dynamic_entity_key_case_with_backend(case, &fixture, backend)?;
    }
    Ok(())
}

const MERGE1_594_SETUP: &str = "UNWIND [0, 1, 2] AS x UNWIND [0, 1, 2] AS y CREATE ({x: x, y: y})";
const MERGE1_594_QUERY: &str = "MATCH (foo) WITH foo.x AS x, foo.y AS y \
    MERGE (:N {x: x, y: y + 1}) \
    MERGE (:N {x: x, y: y}) \
    MERGE (:N {x: x + 1, y: y}) RETURN x, y";

fn run_merge1_594_with_backend(
    fixture: &Fixture,
    backend: StrictOrderedCreateBackend,
) -> Result<()> {
    const LABEL: &str = "Merge1 report 594 [9] dynamic arithmetic node MERGE";
    let observations = backend.observations();
    let output = QueryEngine.execute(
        MERGE1_594_QUERY,
        &mut create3_context(&fixture.graph, Some(&backend), true),
    )?;
    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if observations.pins.load(Ordering::SeqCst) != 1
        || observations.complete_commands.load(Ordering::SeqCst) != 1
        || observations.forbidden_calls.load(Ordering::SeqCst) != 0
        || requests.len() != 1
    {
        return Err(Error::internal(format!(
            "{LABEL}: escaped the one-command native route"
        )));
    }
    requests[0].validate()?;
    let Some(body) = requests[0].bound_relationship_merge_body() else {
        return Err(Error::internal(format!(
            "{LABEL}: lost its bound row-mutation body"
        )));
    };
    if body.schedule.property_bindings.len() != 6
        || body.schedule.property_bindings.iter().any(|binding| {
            !matches!(
                binding.source,
                ResidentRowBoundPropertyBindingSource::EntityProperty { .. }
            )
        })
    {
        return Err(Error::internal(format!(
            "{LABEL}: arithmetic key provenance changed: {body:#?}"
        )));
    }
    drop(requests);

    let expected_statistics = StatementStats {
        nodes_created: 15,
        nodes_deleted: 0,
        relationships_created: 0,
        relationships_deleted: 0,
        properties_set: 0,
        labels_added: 15,
        labels_removed: 0,
    };
    if output.result.schema
        != [
            ("x".to_owned(), ColumnType::Integer),
            ("y".to_owned(), ColumnType::Integer),
        ]
        || output.result.bookmark != fixture.bookmark
        || output.result.statistics != expected_statistics
        || output.result.truncated
        || !output.temporal_mutations.is_empty()
        || output.administrative.is_some()
        || !output.vector_searches.is_empty()
        || output.runtime_replans != 0
    {
        return Err(Error::internal(format!(
            "{LABEL}: result envelope changed: {output:#?}"
        )));
    }
    let mut rows = Vec::new();
    for batch in &output.result.batches {
        if !batch.validate() || batch.columns.len() != 2 {
            return Err(Error::internal(format!(
                "{LABEL}: malformed result batch {batch:?}"
            )));
        }
        for row in 0..batch.row_count {
            let (
                Some(ResultValue::Scalar(ScalarValue::Integer(x))),
                Some(ResultValue::Scalar(ScalarValue::Integer(y))),
            ) = (
                batch.columns[0].values.get(row),
                batch.columns[1].values.get(row),
            )
            else {
                return Err(Error::internal(format!(
                    "{LABEL}: result row is not two integers"
                )));
            };
            rows.push((*x, *y));
        }
    }
    rows.sort_unstable();
    let expected_rows = (0_i64..3)
        .flat_map(|x| (0_i64..3).map(move |y| (x, y)))
        .collect::<Vec<_>>();
    if rows != expected_rows {
        return Err(Error::internal(format!(
            "{LABEL}: expected the 3x3 source grid, got {rows:?}"
        )));
    }
    let mut inserted_nodes = 0_u64;
    let mut inserted_properties = 0_u64;
    let mut assigned_labels = 0_u64;
    let mut declared_labels = 0_u64;
    for mutation in &output.graph_mutations {
        match mutation {
            GraphMutation::InsertNode(node) => {
                inserted_nodes += 1;
                inserted_properties += node.properties.len() as u64;
                assigned_labels += node.labels.len() as u64;
            }
            GraphMutation::DeclareLabel { .. } => declared_labels += 1,
            mutation => {
                return Err(Error::internal(format!(
                    "{LABEL}: emitted an unrelated graph mutation {mutation:?}"
                )));
            }
        }
    }
    if (
        inserted_nodes,
        inserted_properties,
        assigned_labels,
        declared_labels,
    ) != (15, 30, 15, 1)
    {
        return Err(Error::internal(format!(
            "{LABEL}: graph effects changed: nodes={inserted_nodes}, properties={inserted_properties}, labels={assigned_labels}, declarations={declared_labels}"
        )));
    }
    Ok(())
}

#[test]
fn strict_cpu_executes_merge1_594_dynamic_arithmetic_program() -> Result<()> {
    let fixture = build_merge1_fixture(MERGE1_594_SETUP, "Merge1 report 594 [9]")?;
    let backend = StrictOrderedCreateBackend::strict_cpu(fixture.image()?)?;
    run_merge1_594_with_backend(&fixture, backend)
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires a real Metal device; proves three sequential per-key node MERGEs"]
fn real_metal_executes_merge1_594_dynamic_arithmetic_program() -> Result<()> {
    let fixture = build_merge1_fixture(MERGE1_594_SETUP, "Merge1 report 594 [9]")?;
    let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
    metal.admit_project(fixture.image()?)?;
    let backend = StrictOrderedCreateBackend::real_metal(metal)?;
    run_merge1_594_with_backend(&fixture, backend)
}

fn assert_no_native_dispatch(
    observations: &RouteObservations,
    label: &str,
    error: &Error,
    expected_code: ErrorCode,
) -> Result<()> {
    let pins = observations.pins.load(Ordering::SeqCst);
    let commands = observations.complete_commands.load(Ordering::SeqCst);
    let forbidden = observations.forbidden_calls.load(Ordering::SeqCst);
    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .len();
    if error.code != expected_code || pins != 0 || commands != 0 || forbidden != 0 || requests != 0
    {
        return Err(Error::internal(format!(
            "{label}: expected {expected_code:?} before native dispatch, got {error}; pins={pins}, commands={commands}, forbidden={forbidden}, requests={requests}"
        )));
    }
    Ok(())
}

#[test]
fn nearby_node_merge_shapes_remain_fail_closed_before_native_dispatch() -> Result<()> {
    for case in MERGE1_FAIL_CLOSED_CASES {
        let label = format!("Merge1 fail-closed neighbor `{}`", case.name);
        let fixture = build_merge1_fixture(case.setup, &label)?;
        let backend = StrictOrderedCreateBackend::strict_cpu(fixture.image()?)?;
        let observations = backend.observations();
        let error = QueryEngine
            .execute(
                case.query,
                &mut create3_context(&fixture.graph, Some(&backend), true),
            )
            .expect_err("broader node MERGE shape unexpectedly entered the exact native lane");
        assert_no_native_dispatch(
            &observations,
            &label,
            &error,
            ErrorCode::GpuAdmissionFailure,
        )?;
    }
    Ok(())
}

#[test]
fn merge1_null_key_preserves_query_type_before_native_dispatch() -> Result<()> {
    let fixture = build_merge1_fixture("", "Merge1 null-key guard")?;
    let generic = QueryEngine
        .execute(
            "MERGE (a {num: null})",
            &mut create3_context(&fixture.graph, None, false),
        )
        .expect_err("generic MERGE unexpectedly accepted a null key");
    if generic.code != ErrorCode::QueryType || !generic.message.contains("MergeReadOwnWrites") {
        return Err(Error::internal(format!(
            "generic MERGE null-key error changed: {generic}"
        )));
    }

    let backend = StrictOrderedCreateBackend::strict_cpu(fixture.image()?)?;
    let observations = backend.observations();
    let native = QueryEngine
        .execute(
            "MERGE (a {num: null})",
            &mut create3_context(&fixture.graph, Some(&backend), true),
        )
        .expect_err("native MERGE unexpectedly accepted a null key");
    assert_no_native_dispatch(
        &observations,
        "Merge1 null-key guard",
        &native,
        ErrorCode::QueryType,
    )?;
    if !native.message.contains("MergeReadOwnWrites") || !native.message.contains("null") {
        return Err(Error::internal(format!(
            "native MERGE null-key detail changed: {native}"
        )));
    }
    Ok(())
}

const CREATE4_GRAPH_FREE_DOCUMENT_SMOKE: &str = "\
    CREATE (m:Movie {title: 'The Matrix', released: 1999, tagline: 'Welcome'}) \
    CREATE (a:Person {name: 'Keanu Reeves', born: 1964}) \
    CREATE (b:Person {name: 'Carrie-Anne Moss', born: 1967}) \
    CREATE (a)-[:ACTED_IN {roles: ['Neo']}]->(m), \
           (b)-[:ACTED_IN {roles: ['Trinity']}]->(m)";

fn run_graph_free_document_create_with_backend(backend: StrictOrderedCreateBackend) -> Result<()> {
    let graph = GraphStore::default();
    let observations = backend.observations();
    let output = QueryEngine.execute(
        CREATE4_GRAPH_FREE_DOCUMENT_SMOKE,
        &mut create3_context(&graph, Some(&backend), true),
    )?;
    let pins = observations.pins.load(Ordering::SeqCst);
    let commands = observations.complete_commands.load(Ordering::SeqCst);
    let forbidden = observations.forbidden_calls.load(Ordering::SeqCst);
    let requests = observations
        .requests
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if pins != 1 || commands != 1 || forbidden != 0 || requests.len() != 1 {
        return Err(Error::internal(format!(
            "Create4 graph-free document route changed: pins={pins}, commands={commands}, forbidden={forbidden}, requests={}",
            requests.len()
        )));
    }
    let body = requests[0]
        .bound_relationship_merge_body()
        .ok_or_else(|| Error::internal("Create4 smoke lost its generalized ordered body"))?;
    if body.program.entity_slot_count != 5
        || body.command_candidate_limits.as_slice() != [1, 1, 1, 1, 1]
        || !matches!(
            body.commands.as_slice(),
            [
                ResidentRowBoundRelationshipCommand::CreateNode,
                ResidentRowBoundRelationshipCommand::CreateNode,
                ResidentRowBoundRelationshipCommand::CreateNode,
                ResidentRowBoundRelationshipCommand::CreateRelationship,
                ResidentRowBoundRelationshipCommand::CreateRelationship,
            ]
        )
    {
        return Err(Error::internal(format!(
            "Create4 smoke changed its complete command stream: {body:#?}"
        )));
    }
    drop(requests);

    if output.result.statistics
        != (StatementStats {
            nodes_created: 3,
            relationships_created: 2,
            labels_added: 3,
            ..StatementStats::default()
        })
        || !output.result.schema.is_empty()
        || !output.result.batches.is_empty()
    {
        return Err(Error::internal(format!(
            "Create4 smoke changed its result envelope: {output:#?}"
        )));
    }
    let nodes = output
        .graph_mutations
        .iter()
        .filter(|mutation| matches!(mutation, GraphMutation::InsertNode(_)))
        .count();
    let relationships = output
        .graph_mutations
        .iter()
        .filter(|mutation| matches!(mutation, GraphMutation::InsertEdge(_)))
        .count();
    let document_properties = output
        .graph_mutations
        .iter()
        .filter_map(|mutation| match mutation {
            GraphMutation::InsertNode(node) => Some(&node.properties),
            GraphMutation::InsertEdge(edge) => Some(&edge.properties),
            _ => None,
        })
        .flatten()
        .filter(|(_, value)| matches!(value, ScalarValue::List(_)))
        .count();
    if (nodes, relationships, document_properties) != (3, 2, 2) {
        return Err(Error::internal(format!(
            "Create4 smoke changed graph effects: nodes={nodes}, relationships={relationships}, list-properties={document_properties}"
        )));
    }
    Ok(())
}

#[test]
fn strict_cpu_executes_graph_free_document_create_as_one_ordered_command() -> Result<()> {
    let fixture = Fixture {
        graph: GraphStore::default(),
        bookmark: Bookmark {
            term: 103,
            index: 0,
        },
    };
    run_graph_free_document_create_with_backend(StrictOrderedCreateBackend::strict_cpu(
        fixture.image()?,
    )?)
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires a real Metal device; proves graph-free document CREATE composition"]
fn real_metal_executes_graph_free_document_create_as_one_ordered_command() -> Result<()> {
    let fixture = Fixture {
        graph: GraphStore::default(),
        bookmark: Bookmark {
            term: 103,
            index: 0,
        },
    };
    let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
    metal.admit_project(fixture.image()?)?;
    run_graph_free_document_create_with_backend(StrictOrderedCreateBackend::real_metal(metal)?)
}
