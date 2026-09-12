// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{collections::BTreeMap, time::Instant};

use irongraph::{
    Bookmark, EdgeId, Error, ErrorCode, Layer, NodeId, ProjectId, Result, ScalarValue,
    cypher::{BindCapabilities, ExecutionContext, ExecutionOutput, QueryEngine, ResultValue},
    graph::{EdgeInput, GraphStore, NodeInput},
};
use tokio_util::sync::CancellationToken;

fn graph() -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    let label = graph.catalog_mut().intern_label("N")?;
    let relationship_type = graph.catalog_mut().intern_relationship_type("T")?;
    for id in [1, 2] {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: id,
            labels: vec![label],
            properties: Vec::new(),
        })?;
    }
    graph.insert_edge(EdgeInput {
        id: EdgeId(3),
        source: NodeId(1),
        target: NodeId(2),
        relationship_type,
        layer: Layer::Observed,
        revision: 3,
        properties: Vec::new(),
    })?;
    Ok(graph)
}

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
        bookmark: Bookmark { term: 1, index: 1 },
        mutation_revision: 2,
        resolved_time_nanos: 0,
        next_node_id: 100,
        next_edge_id: 100,
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

fn require_runtime_invalid_argument_value(graph: &GraphStore, query: &str) -> Result<()> {
    let error = QueryEngine
        .execute(query, &mut context(graph))
        .err()
        .ok_or_else(|| Error::internal(format!("invalid dynamic graph call succeeded: {query}")))?;
    assert_eq!(error.code, ErrorCode::QueryType, "{query}: {error:?}");
    assert!(
        error.message.contains("InvalidArgumentValue"),
        "{query}: {error:?}"
    );
    Ok(())
}

fn require_compile_invalid_argument_type(graph: &GraphStore, query: &str) -> Result<()> {
    let error = QueryEngine
        .execute(query, &mut context(graph))
        .err()
        .ok_or_else(|| Error::internal(format!("invalid static graph call succeeded: {query}")))?;
    assert_eq!(error.code, ErrorCode::QuerySyntax, "{query}: {error:?}");
    assert!(
        error.message.contains("InvalidArgumentType"),
        "{query}: {error:?}"
    );
    Ok(())
}

fn only_value<'a>(output: &'a ExecutionOutput, column_name: &str) -> Result<&'a ResultValue> {
    let mut values = output.result.batches.iter().flat_map(|batch| {
        batch
            .columns
            .iter()
            .filter(move |column| column.name == column_name)
            .flat_map(|column| column.values.iter())
    });
    let value = values
        .next()
        .ok_or_else(|| Error::internal(format!("missing result column `{column_name}`")))?;
    if values.next().is_some() {
        return Err(Error::internal(format!(
            "result column `{column_name}` contained more than one value"
        )));
    }
    Ok(value)
}

#[test]
fn labels_rejects_a_runtime_any_scalar_as_invalid_argument_value() -> Result<()> {
    let graph = graph()?;
    require_runtime_invalid_argument_value(
        &graph,
        "MATCH (a) WITH [a, 1] AS list RETURN labels(list[1]) AS l",
    )
}

#[test]
fn type_rejects_each_tck_runtime_any_scalar_as_invalid_argument_value() -> Result<()> {
    let graph = graph()?;
    for invalid in ["0", "1.0", "true", "''", "[]"] {
        let query =
            format!("MATCH p = (n)-[r:T]->() RETURN [x IN [r, {invalid}] | type(x)] AS list");
        require_runtime_invalid_argument_value(&graph, &query)?;
    }
    Ok(())
}

#[test]
fn graph_functions_reject_runtime_any_entities_of_the_wrong_kind() -> Result<()> {
    let graph = graph()?;
    for query in [
        "MATCH (n)-[r:T]->() WITH [n, r] AS list RETURN labels(list[1]) AS value",
        "MATCH (n)-[r:T]->() WITH [n, r] AS list RETURN type(list[0]) AS value",
    ] {
        require_runtime_invalid_argument_value(&graph, query)?;
    }
    Ok(())
}

#[test]
fn null_graph_function_arguments_still_propagate_null() -> Result<()> {
    let graph = graph()?;
    for (query, column) in [
        (
            "MATCH (n) WITH [n, null] AS list LIMIT 1 RETURN labels(list[1]) AS value",
            "value",
        ),
        (
            "MATCH ()-[r:T]->() WITH [r, null] AS list RETURN type(list[1]) AS value",
            "value",
        ),
    ] {
        let output = QueryEngine.execute(query, &mut context(&graph))?;
        assert_eq!(
            only_value(&output, column)?,
            &ResultValue::Scalar(ScalarValue::Null),
            "{query}"
        );
    }
    Ok(())
}

#[test]
fn known_wrong_graph_argument_roles_remain_compile_time_errors() -> Result<()> {
    let graph = graph()?;
    for query in [
        "MATCH p = (a) RETURN labels(p) AS l",
        "MATCH (r) RETURN type(r)",
    ] {
        require_compile_invalid_argument_type(&graph, query)?;
    }
    Ok(())
}
