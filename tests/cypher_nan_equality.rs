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

fn assert_boolean(row: &BTreeMap<String, ResultValue>, name: &str, expected: bool) {
    assert_eq!(
        row.get(name),
        Some(&ResultValue::Scalar(ScalarValue::Boolean(expected))),
        "column {name}",
    );
}

#[test]
fn nan_is_unequal_to_numeric_and_cross_type_operands() -> Result<()> {
    let graph = GraphStore::default();
    let output = QueryEngine.execute(
        "RETURN \
         0.0 / 0.0 = 1 AS eq_integer, 0.0 / 0.0 <> 1 AS neq_integer, \
         0.0 / 0.0 = 1.0 AS eq_float, 0.0 / 0.0 <> 1.0 AS neq_float, \
         0.0 / 0.0 = 0.0 / 0.0 AS eq_nan, \
         0.0 / 0.0 <> 0.0 / 0.0 AS neq_nan, \
         0.0 / 0.0 = 'a' AS eq_string, 0.0 / 0.0 <> 'a' AS neq_string",
        &mut context(&graph),
    )?;
    let batch = output
        .result
        .batches
        .first()
        .ok_or_else(|| irongraph::Error::internal("NaN query returned no batch"))?;
    let row = batch
        .columns
        .iter()
        .map(|column| {
            column
                .values
                .first()
                .cloned()
                .map(|value| (column.name.clone(), value))
                .ok_or_else(|| irongraph::Error::internal("NaN query returned an empty column"))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;

    for name in ["eq_integer", "eq_float", "eq_nan", "eq_string"] {
        assert_boolean(&row, name, false);
    }
    for name in ["neq_integer", "neq_float", "neq_nan", "neq_string"] {
        assert_boolean(&row, name, true);
    }
    Ok(())
}
