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
fn aggregates_compose_inside_scalar_collection_and_case_expressions() -> Result<()> {
    let graph = GraphStore::default();
    let output = QueryEngine.execute(
        "UNWIND [1, 2, 3] AS x RETURN sum(x) + 1 AS arithmetic, size(collect(x)) AS collected, CASE WHEN avg(x) > 1 THEN max(x) ELSE min(x) END AS chosen, collect(x)[1] AS indexed",
        &mut context(&graph),
    )?;
    let batch = &output.result.batches[0];
    let value = |name: &str| {
        batch
            .columns
            .iter()
            .find(|column| column.name == name)
            .and_then(|column| column.values.first())
            .cloned()
            .ok_or_else(|| irongraph::Error::internal(format!("column {name} is absent")))
    };
    assert_eq!(
        value("arithmetic")?,
        ResultValue::Scalar(ScalarValue::Integer(7))
    );
    assert_eq!(
        value("collected")?,
        ResultValue::Scalar(ScalarValue::Integer(3))
    );
    assert_eq!(
        value("chosen")?,
        ResultValue::Scalar(ScalarValue::Integer(3))
    );
    assert_eq!(
        value("indexed")?,
        ResultValue::Scalar(ScalarValue::Integer(2))
    );
    Ok(())
}

#[test]
fn nested_aggregate_is_rejected_during_binding() -> Result<()> {
    let graph = GraphStore::default();
    let error = QueryEngine
        .execute(
            "UNWIND [1, 2] AS x RETURN sum(max(x)) AS invalid",
            &mut context(&graph),
        )
        .err()
        .ok_or_else(|| irongraph::Error::internal("nested aggregate succeeded"))?;
    assert_eq!(error.code, irongraph::ErrorCode::QuerySyntax);
    assert!(error.message.contains("NestedAggregation"), "{error:?}");
    Ok(())
}

#[test]
fn with_where_can_filter_on_an_incoming_binding_before_scope_pruning() -> Result<()> {
    let graph = GraphStore::default();
    let output = QueryEngine.execute(
        "UNWIND [1, 2] AS x WITH x + 10 AS visible WHERE x = 1 RETURN visible",
        &mut context(&graph),
    )?;
    let batch = &output.result.batches[0];
    let column = batch
        .columns
        .iter()
        .find(|column| column.name == "visible")
        .ok_or_else(|| irongraph::Error::internal("visible column is absent"))?;
    assert_eq!(
        column.values,
        vec![ResultValue::Scalar(ScalarValue::Integer(11))]
    );
    Ok(())
}
