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
        bookmark: Bookmark { term: 1, index: 20 },
        mutation_revision: 21,
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

fn insert_node(
    graph: &mut GraphStore,
    id: u64,
    labels: &[&str],
    properties: &[(&str, ScalarValue)],
) -> Result<()> {
    let labels = labels
        .iter()
        .map(|label| graph.catalog_mut().intern_label(label))
        .collect::<Result<Vec<_>>>()?;
    let properties = properties
        .iter()
        .map(|(name, value)| {
            graph
                .catalog_mut()
                .intern_property(name)
                .map(|property| (property, value.clone()))
        })
        .collect::<Result<Vec<_>>>()?;
    graph
        .insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: id,
            labels,
            properties,
        })
        .map(|_| ())
}

fn insert_edge(
    graph: &mut GraphStore,
    id: u64,
    source: u64,
    target: u64,
    relationship_type: &str,
) -> Result<()> {
    let relationship_type = graph
        .catalog_mut()
        .intern_relationship_type(relationship_type)?;
    graph
        .insert_edge(EdgeInput {
            id: EdgeId(id),
            source: NodeId(source),
            target: NodeId(target),
            relationship_type,
            layer: Layer::Observed,
            revision: id,
            properties: Vec::new(),
        })
        .map(|_| ())
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

fn scalar_integer(value: i64) -> ResultValue {
    ResultValue::Scalar(ScalarValue::Integer(value))
}

fn null() -> ResultValue {
    ResultValue::Scalar(ScalarValue::Null)
}

fn optional_chain_graph() -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    for (id, label, value) in [
        (1, "X", 1),
        (2, "Y", 2),
        (3, "Z", 3),
        (4, "X", 4),
        (5, "Y", 5),
        (6, "X", 6),
    ] {
        insert_node(
            &mut graph,
            id,
            &[label],
            &[("val", ScalarValue::Integer(value))],
        )?;
    }
    insert_edge(&mut graph, 10, 1, 2, "E1")?;
    insert_edge(&mut graph, 11, 2, 3, "E2")?;
    insert_edge(&mut graph, 12, 4, 5, "E1")?;
    Ok(graph)
}

#[test]
fn optional_where_keeps_only_matching_candidates_without_an_extra_null_row() -> Result<()> {
    let mut graph = GraphStore::default();
    insert_node(&mut graph, 1, &["X"], &[("val", ScalarValue::Integer(1))])?;
    insert_node(&mut graph, 2, &["Y"], &[("val", ScalarValue::Integer(0))])?;
    insert_node(&mut graph, 3, &["Y"], &[("val", ScalarValue::Integer(2))])?;
    insert_edge(&mut graph, 10, 1, 2, "E1")?;
    insert_edge(&mut graph, 11, 1, 3, "E1")?;

    let output = QueryEngine.execute(
        "MATCH (x:X) OPTIONAL MATCH (x)-[:E1]->(y:Y) \
         WHERE x.val < y.val RETURN y.val AS value",
        &mut context(&graph),
    )?;
    assert_eq!(row_count(&output), 1);
    assert_eq!(column_values(&output, "value")?, vec![scalar_integer(2)]);
    Ok(())
}

#[test]
fn optional_where_null_extends_when_raw_candidates_all_fail() -> Result<()> {
    let mut graph = GraphStore::default();
    insert_node(&mut graph, 1, &["X"], &[("val", ScalarValue::Integer(1))])?;
    insert_node(&mut graph, 2, &["Y"], &[("val", ScalarValue::Integer(0))])?;
    insert_edge(&mut graph, 10, 1, 2, "E1")?;

    let output = QueryEngine.execute(
        "MATCH (x:X) OPTIONAL MATCH (x)-[:E1]->(y:Y) \
         WHERE x.val < y.val RETURN x.val AS x, y.val AS y",
        &mut context(&graph),
    )?;
    assert_eq!(row_count(&output), 1);
    assert_eq!(column_values(&output, "x")?, vec![scalar_integer(1)]);
    assert_eq!(column_values(&output, "y")?, vec![null()]);
    Ok(())
}

#[test]
fn missed_optional_pattern_does_not_drop_an_invalid_null_property_predicate() -> Result<()> {
    let mut graph = GraphStore::default();
    insert_node(&mut graph, 1, &[], &[])?;
    insert_node(
        &mut graph,
        2,
        &[],
        &[("name", ScalarValue::String("Mark".into()))],
    )?;
    insert_edge(&mut graph, 10, 1, 2, "T")?;

    let output = QueryEngine.execute(
        "MATCH (n)-->(x0) OPTIONAL MATCH (x0)-->(x1) \
         WHERE x1.name = 'bar' RETURN x0.name AS name",
        &mut context(&graph),
    )?;
    assert_eq!(row_count(&output), 1);
    assert_eq!(
        column_values(&output, "name")?,
        vec![ResultValue::Scalar(ScalarValue::String("Mark".into()))]
    );
    Ok(())
}

#[test]
fn reversed_equality_failure_preserves_correlated_bindings_and_nulls_new_ones() -> Result<()> {
    let mut graph = GraphStore::default();
    insert_node(&mut graph, 1, &["A"], &[])?;
    insert_node(&mut graph, 2, &["B"], &[])?;
    insert_edge(&mut graph, 10, 1, 2, "T")?;

    let output = QueryEngine.execute(
        "MATCH (a1)-[r]->() WITH r, a1 LIMIT 1 \
         OPTIONAL MATCH (a2)<-[r]-(b2) WHERE a1 = a2 \
         RETURN a1, r, b2, a2",
        &mut context(&graph),
    )?;
    assert_eq!(row_count(&output), 1);
    assert!(matches!(
        column_values(&output, "a1")?.as_slice(),
        [ResultValue::Node(_)]
    ));
    assert!(matches!(
        column_values(&output, "r")?.as_slice(),
        [ResultValue::Relationship(_)]
    ));
    assert_eq!(column_values(&output, "b2")?, vec![null()]);
    assert_eq!(column_values(&output, "a2")?, vec![null()]);
    Ok(())
}

#[test]
fn optional_where_preserves_cardinality_for_each_correlated_input_row() -> Result<()> {
    let graph = optional_chain_graph()?;
    let output = QueryEngine.execute(
        "MATCH (x:X) OPTIONAL MATCH (x)-[:E1]->(y:Y) \
         WHERE x.val < y.val RETURN x.val AS x, y.val AS y ORDER BY x",
        &mut context(&graph),
    )?;
    assert_eq!(row_count(&output), 3);
    assert_eq!(
        column_values(&output, "x")?,
        vec![scalar_integer(1), scalar_integer(4), scalar_integer(6)]
    );
    assert_eq!(
        column_values(&output, "y")?,
        vec![scalar_integer(2), scalar_integer(5), null()]
    );
    Ok(())
}

#[test]
fn optional_where_handles_one_pattern_chain_and_two_optional_boundaries() -> Result<()> {
    let graph = optional_chain_graph()?;
    let one_boundary = QueryEngine.execute(
        "MATCH (x:X) OPTIONAL MATCH (x)-[:E1]->(y:Y)-[:E2]->(z:Z) \
         WHERE x.val < z.val \
         RETURN x.val AS x, y.val AS y, z.val AS z ORDER BY x",
        &mut context(&graph),
    )?;
    assert_eq!(row_count(&one_boundary), 3);
    assert_eq!(
        column_values(&one_boundary, "y")?,
        vec![scalar_integer(2), null(), null()]
    );
    assert_eq!(
        column_values(&one_boundary, "z")?,
        vec![scalar_integer(3), null(), null()]
    );

    let two_boundaries = QueryEngine.execute(
        "MATCH (x:X) OPTIONAL MATCH (x)-[:E1]->(y:Y) \
         OPTIONAL MATCH (y)-[:E2]->(z:Z) WHERE x.val < z.val \
         RETURN x.val AS x, y.val AS y, z.val AS z ORDER BY x",
        &mut context(&graph),
    )?;
    assert_eq!(row_count(&two_boundaries), 3);
    assert_eq!(
        column_values(&two_boundaries, "y")?,
        vec![scalar_integer(2), scalar_integer(5), null()]
    );
    assert_eq!(
        column_values(&two_boundaries, "z")?,
        vec![scalar_integer(3), null(), null()]
    );
    Ok(())
}

#[test]
fn non_attached_where_clauses_remain_genuine_filters() -> Result<()> {
    let graph = optional_chain_graph()?;
    for query in [
        "MATCH (x:X) WHERE x.val < 5 RETURN x.val AS x ORDER BY x",
        "OPTIONAL MATCH (x:X) WITH x WHERE x.val < 5 RETURN x.val AS x ORDER BY x",
    ] {
        let output = QueryEngine.execute(query, &mut context(&graph))?;
        assert_eq!(row_count(&output), 2, "query: {query}");
        assert_eq!(
            column_values(&output, "x")?,
            vec![scalar_integer(1), scalar_integer(4)],
            "query: {query}"
        );
    }
    Ok(())
}
