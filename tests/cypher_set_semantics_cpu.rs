// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{collections::BTreeMap, time::Instant};

use irongraph::{
    Bookmark, Layer, NodeId, ProjectId, Result, ScalarValue,
    cypher::{BindCapabilities, ExecutionContext, ExecutionOutput, QueryEngine, ResultValue},
    graph::{GraphMutation, GraphStore, NodeInput},
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
        bookmark: Bookmark { term: 1, index: 1 },
        mutation_revision: 2,
        resolved_time_nanos: 0,
        next_node_id: 2,
        next_edge_id: 1,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            write: true,
            ..BindCapabilities::default()
        },
        max_result_rows: 1_024,
        max_batch_rows: 1_024,
        optimizer_statistics: None,
        backend: None,
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + std::time::Duration::from_secs(5)),
        resolved_query_at_time_nanos: None,
    }
}

fn graph_with_two_properties() -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    let label = graph.catalog_mut().intern_label("X")?;
    let name = graph.catalog_mut().intern_property("name")?;
    let name2 = graph.catalog_mut().intern_property("name2")?;
    graph.insert_node(NodeInput {
        id: NodeId(1),
        layer: Layer::Observed,
        revision: 1,
        labels: vec![label],
        properties: vec![
            (name, ScalarValue::String("A".into())),
            (name2, ScalarValue::String("B".into())),
        ],
    })?;
    Ok(graph)
}

fn only_value(output: &ExecutionOutput) -> Result<&ResultValue> {
    output
        .result
        .batches
        .first()
        .and_then(|batch| batch.columns.first())
        .and_then(|column| column.values.first())
        .ok_or_else(|| irongraph::Error::internal("query returned no value"))
}

#[test]
fn replacing_a_property_map_removes_absent_keys_including_for_an_empty_map() -> Result<()> {
    let graph = graph_with_two_properties()?;
    let output = QueryEngine.execute(
        "MATCH (n:X {name: 'A'}) SET n = {name: 'B', baz: 'C'} RETURN n",
        &mut context(&graph),
    )?;
    let ResultValue::Node(node) = only_value(&output)? else {
        return Err(irongraph::Error::internal("SET did not return a node"));
    };
    assert_eq!(
        node.properties,
        BTreeMap::from([
            ("baz".to_owned(), ScalarValue::String("C".into())),
            ("name".to_owned(), ScalarValue::String("B".into())),
        ])
    );
    assert_eq!(output.result.statistics.properties_set, 3);
    assert!(output.graph_mutations.iter().any(|mutation| matches!(
        mutation,
        GraphMutation::SetNodeProperty {
            value: ScalarValue::Null,
            ..
        }
    )));

    let graph = graph_with_two_properties()?;
    let output = QueryEngine.execute(
        "MATCH (n:X {name: 'A'}) SET n = {} RETURN n",
        &mut context(&graph),
    )?;
    let ResultValue::Node(node) = only_value(&output)? else {
        return Err(irongraph::Error::internal("SET did not return a node"));
    };
    assert!(node.properties.is_empty());
    assert_eq!(output.result.statistics.properties_set, 2);
    assert_eq!(
        output
            .graph_mutations
            .iter()
            .filter(|mutation| matches!(
                mutation,
                GraphMutation::SetNodeProperty {
                    value: ScalarValue::Null,
                    ..
                }
            ))
            .count(),
        2
    );
    Ok(())
}

#[test]
fn merging_a_property_map_retains_keys_that_are_not_supplied() -> Result<()> {
    let graph = graph_with_two_properties()?;
    let output = QueryEngine.execute(
        "MATCH (n:X {name: 'A'}) SET n += {name: 'C'} RETURN n",
        &mut context(&graph),
    )?;
    let ResultValue::Node(node) = only_value(&output)? else {
        return Err(irongraph::Error::internal("SET did not return a node"));
    };
    assert_eq!(
        node.properties,
        BTreeMap::from([
            ("name".to_owned(), ScalarValue::String("C".into())),
            ("name2".to_owned(), ScalarValue::String("B".into())),
        ])
    );
    assert_eq!(output.result.statistics.properties_set, 1);
    Ok(())
}

#[test]
fn every_set_form_treats_a_null_entity_as_a_no_op() -> Result<()> {
    let graph = GraphStore::default();
    for query in [
        "OPTIONAL MATCH (a:Missing) SET a.num = 42 RETURN a",
        "OPTIONAL MATCH (a:Missing) SET a = {num: 42} RETURN a",
        "OPTIONAL MATCH (a:Missing) SET a += {num: 42} RETURN a",
        "OPTIONAL MATCH (a:Missing) SET a:Label RETURN a",
    ] {
        let output = QueryEngine.execute(query, &mut context(&graph))?;
        assert_eq!(
            only_value(&output)?,
            &ResultValue::Scalar(ScalarValue::Null),
            "query: {query}"
        );
        assert!(output.graph_mutations.is_empty(), "query: {query}");
        assert_eq!(output.result.statistics.properties_set, 0, "query: {query}");
        assert_eq!(output.result.statistics.labels_added, 0, "query: {query}");
    }
    Ok(())
}
