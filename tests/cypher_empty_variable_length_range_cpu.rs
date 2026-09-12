// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{collections::BTreeMap, time::Instant};

use irongraph::{
    Bookmark, EdgeId, Error, Layer, NodeId, ProjectId, Result, ScalarValue,
    cypher::{BindCapabilities, ExecutionContext, ExecutionOutput, QueryEngine, ResultValue},
    graph::{EdgeInput, GraphStore, NodeInput},
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
        bookmark: Bookmark { term: 1, index: 12 },
        mutation_revision: 13,
        resolved_time_nanos: 0,
        next_node_id: 5,
        next_edge_id: 13,
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

fn sample_graph() -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    let root = graph.catalog_mut().intern_label("A")?;
    let other = graph.catalog_mut().intern_label("B")?;
    let likes = graph.catalog_mut().intern_relationship_type("LIKES")?;
    let name = graph.catalog_mut().intern_property("name")?;
    for (id, label, value) in [
        (1, root, "root"),
        (2, other, "middle"),
        (3, other, "other"),
        (4, other, "leaf"),
    ] {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: id,
            labels: vec![label],
            properties: vec![(name, ScalarValue::String(value.into()))],
        })?;
    }
    for (id, source, target) in [(10, 1, 2), (11, 1, 3), (12, 2, 4)] {
        graph.insert_edge(EdgeInput {
            id: EdgeId(id),
            source: NodeId(source),
            target: NodeId(target),
            relationship_type: likes,
            layer: Layer::Observed,
            revision: id,
            properties: Vec::new(),
        })?;
    }
    Ok(graph)
}

fn row_count(output: &ExecutionOutput) -> usize {
    output
        .result
        .batches
        .iter()
        .map(|batch| batch.row_count)
        .sum()
}

fn column_values(output: &ExecutionOutput, name: &str) -> Result<Vec<ResultValue>> {
    let mut values = Vec::new();
    for batch in &output.result.batches {
        let column = batch
            .columns
            .iter()
            .find(|column| column.name == name)
            .ok_or_else(|| Error::internal(format!("result omitted column `{name}`")))?;
        values.extend(column.values.iter().cloned());
    }
    Ok(values)
}

fn one_value(output: &ExecutionOutput, name: &str) -> Result<ResultValue> {
    let values = column_values(output, name)?;
    if values.len() != 1 {
        return Err(Error::internal(format!(
            "column `{name}` returned {} values instead of one",
            values.len()
        )));
    }
    values
        .into_iter()
        .next()
        .ok_or_else(|| Error::internal(format!("column `{name}` returned no value")))
}

#[test]
fn empty_variable_length_intervals_produce_no_matches_without_errors() -> Result<()> {
    let graph = sample_graph()?;
    for interval in ["2..1", "1..0", "..0", "4.."] {
        let query = format!("MATCH (a:A) MATCH (a)-[:LIKES*{interval}]->(c) RETURN c.name AS name");
        let output = QueryEngine.execute(&query, &mut context(&graph))?;
        assert_eq!(row_count(&output), 0, "interval: {interval}");
        assert!(output.graph_mutations.is_empty(), "interval: {interval}");
    }
    Ok(())
}

#[test]
fn optional_empty_interval_preserves_one_null_bound_row() -> Result<()> {
    let graph = sample_graph()?;
    let output = QueryEngine.execute(
        "MATCH (a:A) \
         OPTIONAL MATCH p = (a)-[rs:LIKES*2..1]->(c) \
         RETURN a.name AS source, p, rs, c",
        &mut context(&graph),
    )?;

    assert_eq!(row_count(&output), 1);
    assert_eq!(
        one_value(&output, "source")?,
        ResultValue::Scalar(ScalarValue::String("root".into()))
    );
    for name in ["p", "rs", "c"] {
        assert_eq!(
            one_value(&output, name)?,
            ResultValue::Scalar(ScalarValue::Null),
            "column: {name}"
        );
    }
    Ok(())
}

#[test]
fn valid_fixed_bounded_and_unbounded_ranges_keep_existing_semantics() -> Result<()> {
    let graph = sample_graph()?;
    let bounded = QueryEngine.execute(
        "MATCH (a:A) MATCH p = (a)-[:LIKES*0..2]->(c) \
         RETURN length(p) AS hops ORDER BY hops",
        &mut context(&graph),
    )?;
    assert_eq!(
        column_values(&bounded, "hops")?,
        [0, 1, 1, 2]
            .into_iter()
            .map(|value| ResultValue::Scalar(ScalarValue::Integer(value)))
            .collect::<Vec<_>>()
    );

    for (range, expected_rows) in [("0", 1), ("1", 2), ("..1", 2), ("", 3)] {
        let query = format!("MATCH (a:A) MATCH (a)-[:LIKES*{range}]->(c) RETURN c");
        let output = QueryEngine.execute(&query, &mut context(&graph))?;
        assert_eq!(row_count(&output), expected_rows, "range: `{range}`");
    }
    Ok(())
}
