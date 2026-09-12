// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{collections::BTreeMap, time::Instant};

use irongraph::{
    Bookmark, ErrorCode, ProjectId, Result, ScalarValue,
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

fn row(query: &str) -> Result<BTreeMap<String, ResultValue>> {
    let graph = GraphStore::default();
    let output = QueryEngine.execute(query, &mut context(&graph))?;
    let batch = output
        .result
        .batches
        .first()
        .ok_or_else(|| irongraph::Error::internal("query returned no batch"))?;
    batch
        .columns
        .iter()
        .map(|column| {
            column
                .values
                .first()
                .cloned()
                .map(|value| (column.name.clone(), value))
                .ok_or_else(|| irongraph::Error::internal("query returned an empty column"))
        })
        .collect()
}

fn integer_list(values: &[i64]) -> ResultValue {
    ResultValue::List(
        values
            .iter()
            .copied()
            .map(|value| ResultValue::Scalar(ScalarValue::Integer(value)))
            .collect(),
    )
}

#[test]
fn list_slices_normalize_negative_omitted_and_exceeding_bounds() -> Result<()> {
    let values = row("WITH [1, 2, 3] AS xs \
         RETURN xs[-3..-1] AS negative, xs[-5..5] AS clamped, \
                xs[3..1] AS reversed, xs[1..] AS open_end, xs[..2] AS open_start")?;
    assert_eq!(values.get("negative"), Some(&integer_list(&[1, 2])));
    assert_eq!(values.get("clamped"), Some(&integer_list(&[1, 2, 3])));
    assert_eq!(values.get("reversed"), Some(&integer_list(&[])));
    assert_eq!(values.get("open_end"), Some(&integer_list(&[2, 3])));
    assert_eq!(values.get("open_start"), Some(&integer_list(&[1, 2])));
    Ok(())
}

#[test]
fn explicit_null_in_either_slice_bound_nulls_the_slice() -> Result<()> {
    let values = row("WITH [1, 2, 3] AS xs \
         RETURN xs[null..null] AS both, xs[1..null] AS upper, \
                xs[null..3] AS lower, xs[..null] AS implicit_lower, \
                xs[null..] AS implicit_upper")?;
    for name in ["both", "upper", "lower", "implicit_lower", "implicit_upper"] {
        assert_eq!(
            values.get(name),
            Some(&ResultValue::Scalar(ScalarValue::Null)),
            "column: {name}"
        );
    }
    Ok(())
}

#[test]
fn non_integer_non_null_slice_bounds_remain_type_errors() -> Result<()> {
    let graph = GraphStore::default();
    for query in ["RETURN [1, 2, 3]['1'..2]", "RETURN [1, 2, 3][0..1.5]"] {
        let error = QueryEngine
            .execute(query, &mut context(&graph))
            .err()
            .ok_or_else(|| irongraph::Error::internal("invalid slice bound succeeded"))?;
        assert_eq!(error.code, ErrorCode::QueryType, "query: {query}");
        assert_eq!(error.message, "INTEGER value required", "query: {query}");
    }
    Ok(())
}
