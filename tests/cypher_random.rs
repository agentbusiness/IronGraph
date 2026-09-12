// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{collections::BTreeMap, time::Instant};

use irongraph::{
    Bookmark, ProjectId, Result, ScalarValue,
    cypher::{BindCapabilities, ExecutionContext, QueryEngine, ResultValue},
    graph::GraphStore,
};
use tokio_util::sync::CancellationToken;

fn context(graph: &GraphStore) -> ExecutionContext<'_> {
    ExecutionContext {
        project_id: ProjectId(uuid::Uuid::nil()),
        graph,
        binding_catalog: graph.catalog(),
        prior_graph_mutations: &[],
        temporal: None,
        prior_temporal_mutations: &[],
        vector_indexes: BTreeMap::new(),
        scalar_indexes: None,
        text_embedding: None,
        parameters: BTreeMap::new(),
        bookmark: Bookmark { term: 1, index: 0 },
        mutation_revision: 1,
        resolved_time_nanos: 0,
        next_node_id: 1,
        next_edge_id: 1,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities::default(),
        max_result_rows: 1_024,
        max_batch_rows: 1_024,
        optimizer_statistics: None,
        backend: None,
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + std::time::Duration::from_secs(5)),
        resolved_query_at_time_nanos: None,
    }
}

#[test]
fn rand_produces_one_unit_interval_float_per_row() -> Result<()> {
    let graph = GraphStore::default();
    let output = QueryEngine.execute(
        "UNWIND range(1, 128) AS row RETURN rand() AS value",
        &mut context(&graph),
    )?;
    let values = output
        .result
        .batches
        .iter()
        .flat_map(|batch| &batch.columns)
        .find(|column| column.name == "value")
        .ok_or_else(|| irongraph::Error::internal("rand result column is absent"))?;
    assert_eq!(values.values.len(), 128);
    assert!(values.values.iter().all(|value| {
        matches!(
            value,
            ResultValue::Scalar(ScalarValue::Float(value))
                if value.into_inner() >= 0.0 && value.into_inner() < 1.0
        )
    }));
    Ok(())
}

#[test]
fn rand_rejects_arguments() -> Result<()> {
    let graph = GraphStore::default();
    let error = QueryEngine
        .execute("RETURN rand(1)", &mut context(&graph))
        .err()
        .ok_or_else(|| irongraph::Error::internal("rand accepted an argument"))?;
    assert_eq!(error.code, irongraph::ErrorCode::QueryType);
    Ok(())
}
