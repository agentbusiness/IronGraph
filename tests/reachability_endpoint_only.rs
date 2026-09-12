//! Correctness of the endpoint-only variable-length reachability fast path, focused on the
//! seed-overlap case a small TCK graph may not exercise: a node that is both a start and a reachable
//! endpoint (reached back via a cycle) must be counted.

// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use irongraph::{
    Bookmark, EdgeId, Layer, ProjectId, ScalarValue,
    cypher::{BindCapabilities, ExecutionContext, ExecutionOutput, QueryEngine, ResultValue},
    gpu::{CpuBackend, ExecutionBackend, ResidentProjectImage},
    graph::{EdgeInput, GraphStore, IndexCatalog, NodeInput, StatisticsSnapshot, TemporalStore},
    types::NodeId,
};
use tokio_util::sync::CancellationToken;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());
const BOOKMARK: Bookmark = Bookmark { term: 1, index: 1 };

/// Builds a `:Node` graph with `:R` edges and returns the single integer a scalar query produces.
fn scalar(node_count: u64, edges: &[(u64, u64)], query: &str) -> i64 {
    let mut graph = GraphStore::default();
    let label = graph.catalog_mut().intern_label("Node").unwrap();
    let rel = graph.catalog_mut().intern_relationship_type("R").unwrap();
    for id in 1..=node_count {
        graph
            .insert_node(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: 1,
                labels: vec![label],
                properties: Vec::new(),
            })
            .unwrap();
    }
    for (index, (source, target)) in edges.iter().enumerate() {
        graph
            .insert_edge(EdgeInput {
                id: EdgeId(index as u64 + 1),
                source: NodeId(*source),
                target: NodeId(*target),
                relationship_type: rel,
                layer: Layer::Observed,
                revision: 1,
                properties: Vec::new(),
            })
            .unwrap();
    }
    let image = ResidentProjectImage::build(
        PROJECT,
        BOOKMARK,
        &graph,
        &TemporalStore::default(),
        &IndexCatalog::default(),
    )
    .unwrap();
    let mut cpu = CpuBackend::new(irongraph::config::UNBOUNDED_DEVICE_MEMORY_BYTES, 0);
    cpu.admit_project(image).unwrap();
    let statistics = StatisticsSnapshot::collect_project(
        &graph,
        Some(&TemporalStore::default()),
        Some(&IndexCatalog::default()),
    );
    let backend: &dyn ExecutionBackend = &cpu;
    let mut context = ExecutionContext {
        project_id: PROJECT,
        graph: &graph,
        binding_catalog: graph.catalog(),
        prior_graph_mutations: &[],
        temporal: None,
        prior_temporal_mutations: &[],
        vector_indexes: BTreeMap::new(),
        scalar_indexes: None,
        text_embedding: None,
        parameters: BTreeMap::new(),
        bookmark: BOOKMARK,
        mutation_revision: graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id: 1,
        next_edge_id: 1,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities::default(),
        max_result_rows: 8_000_000,
        max_batch_rows: 65_536,
        optimizer_statistics: Some(&statistics),
        backend: Some(backend),
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(60)),
        resolved_query_at_time_nanos: None,
    };
    let out: ExecutionOutput = QueryEngine.execute(query, &mut context).unwrap();
    let batch = out.result.batches.first().expect("one result batch");
    let column = batch.columns.first().expect("one result column");
    match column.values.first().expect("one result value") {
        ResultValue::Scalar(ScalarValue::Integer(value)) => *value,
        other => panic!("expected integer scalar, got {other:?}"),
    }
}

#[test]
fn endpoint_only_counts_cycle_reached_starts() {
    // 3-cycle 1->2->3->1. Every node is reachable from some node in 1..3 hops, including each node
    // reaching *itself* (a start) via the full cycle. count(DISTINCT m) must be 3, not 2.
    let edges = [(1, 2), (2, 3), (3, 1)];
    assert_eq!(
        scalar(
            3,
            &edges,
            "MATCH (n:Node)-[:R*1..3]->(m) RETURN count(DISTINCT m) AS c"
        ),
        3,
        "cycle-reached start endpoints must be counted"
    );
    // RETURN DISTINCT m over the same cycle also yields all three nodes.
    // (count of the distinct-row form.)
    assert_eq!(
        scalar(
            3,
            &edges,
            "MATCH (n:Node)-[:R*1..3]->(m) WITH DISTINCT m RETURN count(m) AS c"
        ),
        3
    );
}

#[test]
fn endpoint_only_line_graph_reachable_set() {
    // Line 1->2->3->4. From {1,2,3,4} with *1..2: reachable endpoints are {2,3,4} (from 1),
    // {3,4} (from 2), {4} (from 3) => distinct {2,3,4} = 3. No node reaches itself (acyclic).
    let edges = [(1, 2), (2, 3), (3, 4)];
    assert_eq!(
        scalar(
            4,
            &edges,
            "MATCH (n:Node)-[:R*1..2]->(m) RETURN count(DISTINCT m) AS c"
        ),
        3
    );
    // With *1..3, node 1 reaches 4 as well but the endpoint set is still {2,3,4}.
    assert_eq!(
        scalar(
            4,
            &edges,
            "MATCH (n:Node)-[:R*1..3]->(m) RETURN count(DISTINCT m) AS c"
        ),
        3
    );
}

#[test]
fn endpoint_only_self_loop_is_reachable() {
    // Node 1 has a self-loop 1->1; node 2 points to 1. *1..1: from 1 -> {1}, from 2 -> {1}.
    // distinct endpoints = {1} => 1.
    let edges = [(1, 1), (2, 1)];
    assert_eq!(
        scalar(
            2,
            &edges,
            "MATCH (n:Node)-[:R*1..1]->(m) RETURN count(DISTINCT m) AS c"
        ),
        1
    );
}
