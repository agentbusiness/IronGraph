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

fn first_value(query: &str) -> Result<ResultValue> {
    let graph = GraphStore::default();
    let output = QueryEngine.execute(query, &mut context(&graph))?;
    output
        .result
        .batches
        .first()
        .and_then(|batch| batch.columns.first())
        .and_then(|column| column.values.first())
        .cloned()
        .ok_or_else(|| irongraph::Error::internal("query returned no value"))
}

#[test]
fn non_string_string_predicate_operands_evaluate_to_null() -> Result<()> {
    for query in [
        "RETURN 1 STARTS WITH '1' AS value",
        "RETURN [] ENDS WITH {} AS value",
        "RETURN true CONTAINS 0 AS value",
        "RETURN 'abc' STARTS WITH null AS value",
    ] {
        assert_eq!(
            first_value(query)?,
            ResultValue::Scalar(ScalarValue::Null),
            "query: {query}",
        );
    }
    Ok(())
}

#[test]
fn string_predicates_still_compare_string_operands() -> Result<()> {
    let cases = [
        ("RETURN 'abcdef' STARTS WITH 'abc' AS value", true),
        ("RETURN 'abcdef' ENDS WITH 'def' AS value", true),
        ("RETURN 'abcdef' CONTAINS 'cd' AS value", true),
        ("RETURN 'abcdef' CONTAINS 'xy' AS value", false),
    ];
    for (query, expected) in cases {
        assert_eq!(
            first_value(query)?,
            ResultValue::Scalar(ScalarValue::Boolean(expected)),
            "query: {query}",
        );
    }
    Ok(())
}
