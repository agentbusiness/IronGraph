// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort. In a test the opposite is true: a failed expectation is how the test
// reports, and clippy's in-test allowance does not reach the helper functions fixtures are built
// from. Scoping the allowance here keeps the production gate enforceable instead of switched off.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Aggregation over a variable-length pattern, on a GPU node.
//!
//! `MATCH p=(a)-[:R*1..3]->(b) RETURN count(*)` used to fail with `GPU_ADMISSION_FAILURE` on any
//! Metal node. A variable-length pattern compiles to `ScanPattern`, the resident variable-path
//! compiler is a chain of exact shape matchers grown from the pinned openCypher corpus, and no
//! scenario in that corpus aggregates over one — so nothing matched. `ScanPattern` was then
//! refused host execution outright, turning a missing specialization into a refused query.
//!
//! The whole openCypher TCK passed at the time, which is exactly why this file exists separately
//! from it: a corpus that never asks a question cannot notice the answer is missing. Every
//! assertion below therefore checks the *value*, not merely that no error came back. Counting
//! nothing and counting correctly both avoid an error; only one of them is right.
//!
//! The graph is a four-node chain, whose path counts are small enough to enumerate by hand:
//!
//! ```text
//! n1 -> n2 -> n3 -> n4
//!   length 1: (n1,n2) (n2,n3) (n3,n4)   = 3
//!   length 2: (n1,n3) (n2,n4)           = 2
//!   length 3: (n1,n4)                   = 1
//!   *1..3 total                         = 6
//! ```

use std::{collections::BTreeMap, sync::Arc, time::Instant};

use irongraph::{
    Bookmark, EdgeId, Layer, NodeId, ProjectId,
    cypher::{BindCapabilities, ExecutionContext, QueryEngine, ResultValue},
    gpu::ExecutionBackend,
    graph::{EdgeInput, GraphStore, NodeInput},
};
use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::gpu::MetalBackend;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());
const MEMORY_LIMIT: usize = 256 * 1024 * 1024;
const RESERVED_MEMORY: usize = 4 * 1024 * 1024;

/// `n1 -> n2 -> n3 -> n4`, all `:N` joined by `:R`.
fn chain_graph() -> irongraph::Result<GraphStore> {
    let mut graph = GraphStore::default();
    let node = graph.catalog_mut().intern_label("N")?;
    let rel = graph.catalog_mut().intern_relationship_type("R")?;
    for id in 1..=4_u64 {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: id,
            labels: vec![node],
            properties: Vec::new(),
        })?;
    }
    for (index, (source, target)) in [(1, 2), (2, 3), (3, 4)].into_iter().enumerate() {
        graph.insert_edge(EdgeInput {
            id: EdgeId(10 + index as u64),
            source: NodeId(source),
            target: NodeId(target),
            relationship_type: rel,
            layer: Layer::Observed,
            revision: 10 + index as u64,
            properties: Vec::new(),
        })?;
    }
    Ok(graph)
}

fn context<'a>(graph: &'a GraphStore, backend: &'a dyn ExecutionBackend) -> ExecutionContext<'a> {
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
        bookmark: Bookmark { term: 0, index: 5 },
        mutation_revision: graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id: 100,
        next_edge_id: 100,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities::default(),
        max_result_rows: 1_024,
        max_batch_rows: 256,
        optimizer_statistics: None,
        backend: Some(backend),
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + std::time::Duration::from_secs(30)),
        resolved_query_at_time_nanos: None,
    }
}

/// Runs `query` and returns the single scalar it projects.
fn scalar(
    graph: &GraphStore,
    backend: &dyn ExecutionBackend,
    query: &str,
) -> irongraph::Result<ResultValue> {
    let mut context = context(graph, backend);
    let result = QueryEngine.execute(query, &mut context).map_err(|error| {
        irongraph::Error::internal(format!("`{query}` failed: {}", error.message))
    })?;
    let [batch] = result.result.batches.as_slice() else {
        return Err(irongraph::Error::internal(format!(
            "`{query}` produced {} batches, expected exactly one",
            result.result.batches.len()
        )));
    };
    let [column] = batch.columns.as_slice() else {
        return Err(irongraph::Error::internal(format!(
            "`{query}` returned {} columns, expected exactly one",
            batch.columns.len()
        )));
    };
    let [value] = column.values.as_slice() else {
        return Err(irongraph::Error::internal(format!(
            "`{query}` returned {} rows, expected exactly one",
            column.values.len()
        )));
    };
    Ok(value.clone())
}

fn integer(value: &ResultValue) -> Option<i64> {
    match value {
        ResultValue::Scalar(irongraph::ScalarValue::Integer(number)) => Some(*number),
        _ => None,
    }
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn aggregates_over_a_variable_length_pattern_answer_on_a_gpu_node() -> irongraph::Result<()> {
    let graph = chain_graph()?;
    let governor = irongraph::gpu::DeviceMemoryGovernor::new(MEMORY_LIMIT, RESERVED_MEMORY);
    let mut metal = MetalBackend::with_governor(0, governor)?;
    metal.admit_graph(Arc::new(graph.snapshot()?))?;

    // Each bound checked on its own, so a count that is merely plausible in total still fails.
    for (query, expected) in [
        ("MATCH p=(a:N)-[:R*1..1]->(b:N) RETURN count(*) AS n", 3_i64),
        ("MATCH p=(a:N)-[:R*2..2]->(b:N) RETURN count(*) AS n", 2),
        ("MATCH p=(a:N)-[:R*3..3]->(b:N) RETURN count(*) AS n", 1),
        ("MATCH p=(a:N)-[:R*1..3]->(b:N) RETURN count(*) AS n", 6),
        // `count(p)` counts non-null path bindings, which is the same six paths. It is worth
        // asserting separately: it reaches the aggregate through the path variable rather than
        // through the row, and only one of the two shapes was in the original bug report.
        ("MATCH p=(a:N)-[:R*1..3]->(b:N) RETURN count(p) AS n", 6),
        (
            "MATCH p=(a:N)-[:R*1..3]->(b:N) RETURN min(length(p)) AS n",
            1,
        ),
        (
            "MATCH p=(a:N)-[:R*1..3]->(b:N) RETURN max(length(p)) AS n",
            3,
        ),
        // 1+1+1+2+2+3. A sum discriminates between the six correct paths and any six rows.
        (
            "MATCH p=(a:N)-[:R*1..3]->(b:N) RETURN sum(length(p)) AS n",
            10,
        ),
    ] {
        let value = scalar(&graph, &metal, query)?;
        assert_eq!(
            integer(&value),
            Some(expected),
            "`{query}` must return {expected}, got {value:?}"
        );
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn a_gpu_node_and_a_cpu_node_agree_on_variable_length_aggregates() -> irongraph::Result<()> {
    // The fix routes an uncovered read to the same host execution a CPU node performs, so the two
    // must produce identical answers. If they ever diverge, the fallback is not a fallback.
    let graph = chain_graph()?;
    let mut cpu = irongraph::gpu::CpuBackend::new(MEMORY_LIMIT, RESERVED_MEMORY);
    cpu.admit_graph(Arc::new(graph.snapshot()?))?;
    let governor = irongraph::gpu::DeviceMemoryGovernor::new(MEMORY_LIMIT, RESERVED_MEMORY);
    let mut metal = MetalBackend::with_governor(0, governor)?;
    metal.admit_graph(Arc::new(graph.snapshot()?))?;

    for query in [
        "MATCH p=(a:N)-[:R*1..3]->(b:N) RETURN count(*) AS n",
        "MATCH p=(a:N)-[:R*1..3]->(b:N) RETURN sum(length(p)) AS n",
        "MATCH p=(a:N)-[:R*2..3]->(b:N) RETURN count(*) AS n",
    ] {
        assert_eq!(
            scalar(&graph, &cpu, query)?,
            scalar(&graph, &metal, query)?,
            "`{query}` must agree between a CPU node and a GPU node"
        );
    }
    Ok(())
}

#[cfg(not(all(feature = "accelerator", target_os = "macos")))]
#[test]
fn variable_length_aggregate_gate_requires_a_real_gpu_build() {
    eprintln!(
        "variable-length aggregate gate skipped: run on macOS with --features accelerator and Metal \
         available. The defect it covers only reproduces on a non-CPU backend."
    );
}
