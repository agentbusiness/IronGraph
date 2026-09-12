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
    graph::{EdgeInput, GraphMutation, GraphStore, NodeInput},
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
        bookmark: Bookmark { term: 1, index: 4 },
        mutation_revision: 5,
        resolved_time_nanos: 0,
        next_node_id: 10,
        next_edge_id: 10,
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

fn one_value<'a>(output: &'a ExecutionOutput, name: &str) -> Result<&'a ResultValue> {
    output
        .result
        .batches
        .iter()
        .flat_map(|batch| &batch.columns)
        .find(|column| column.name == name)
        .and_then(|column| column.values.first())
        .ok_or_else(|| Error::internal(format!("query returned no `{name}` value")))
}

fn merge_fixture(existing_relationship: bool) -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    let a = graph.catalog_mut().intern_label("A")?;
    let b = graph.catalog_mut().intern_label("B")?;
    let relationship_type = graph.catalog_mut().intern_relationship_type("TYPE")?;
    let name = graph.catalog_mut().intern_property("name")?;
    let rank = graph.catalog_mut().intern_property("rank")?;
    let stale = graph.catalog_mut().intern_property("stale")?;
    graph.insert_node(NodeInput {
        id: NodeId(1),
        layer: Layer::Observed,
        revision: 1,
        labels: vec![a],
        properties: vec![
            (name, ScalarValue::String("A".into())),
            (rank, ScalarValue::Integer(7)),
        ],
    })?;
    graph.insert_node(NodeInput {
        id: NodeId(2),
        layer: Layer::Observed,
        revision: 2,
        labels: vec![b],
        properties: vec![(name, ScalarValue::String("B".into()))],
    })?;
    if existing_relationship {
        graph.insert_edge(EdgeInput {
            id: EdgeId(1),
            source: NodeId(1),
            target: NodeId(2),
            relationship_type,
            layer: Layer::Observed,
            revision: 3,
            properties: vec![
                (name, ScalarValue::String("bar".into())),
                (stale, ScalarValue::Boolean(true)),
            ],
        })?;
    }
    Ok(graph)
}

fn relationship_source_fixture() -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    let a = graph.catalog_mut().intern_label("A")?;
    let b = graph.catalog_mut().intern_label("B")?;
    let source = graph.catalog_mut().intern_relationship_type("SOURCE")?;
    let name = graph.catalog_mut().intern_property("name")?;
    let stale = graph.catalog_mut().intern_property("stale")?;
    let weight = graph.catalog_mut().intern_property("weight")?;
    graph.insert_node(NodeInput {
        id: NodeId(1),
        layer: Layer::Observed,
        revision: 1,
        labels: vec![a],
        properties: vec![(weight, ScalarValue::Integer(0))],
    })?;
    graph.insert_node(NodeInput {
        id: NodeId(2),
        layer: Layer::Observed,
        revision: 2,
        labels: vec![b],
        properties: vec![
            (name, ScalarValue::String("B".into())),
            (stale, ScalarValue::Boolean(true)),
        ],
    })?;
    graph.insert_edge(EdgeInput {
        id: EdgeId(1),
        source: NodeId(1),
        target: NodeId(2),
        relationship_type: source,
        layer: Layer::Observed,
        revision: 3,
        properties: vec![(weight, ScalarValue::Integer(1))],
    })?;
    Ok(graph)
}

#[test]
fn merge_null_pattern_properties_fail_at_runtime_without_leaking_staged_mutations() -> Result<()> {
    for query in [
        "MERGE ({num: null})",
        "CREATE (a), (b) MERGE (a)-[r:X {num: null}]->(b)",
        "UNWIND [1, null] AS num MERGE (:Item {num: num})",
    ] {
        let graph = GraphStore::default();
        let error = QueryEngine
            .execute(query, &mut context(&graph))
            .expect_err("MERGE accepted a null pattern property");
        assert_eq!(error.code, ErrorCode::QueryType, "query: {query}");
        assert!(
            error.message.contains("MergeReadOwnWrites"),
            "query: {query}; error: {error:?}"
        );
        assert_eq!(graph.node_count(), 0, "query: {query}");
        assert_eq!(graph.edge_count(), 0, "query: {query}");
        assert!(graph.catalog().property("num").is_none(), "query: {query}");
        assert!(graph.catalog().label("Item").is_none(), "query: {query}");
        assert!(
            graph.catalog().relationship_type("X").is_none(),
            "query: {query}"
        );
    }
    Ok(())
}

#[test]
fn merge_on_create_and_on_match_copy_a_nodes_current_property_map() -> Result<()> {
    let expected = BTreeMap::from([
        ("name".to_owned(), ScalarValue::String("A".into())),
        ("rank".to_owned(), ScalarValue::Integer(7)),
    ]);

    let graph = merge_fixture(false)?;
    let created = QueryEngine.execute(
        "MATCH (a:A), (b:B) \
         MERGE (a)-[r:TYPE]->(b) \
         ON CREATE SET r = a \
         RETURN r",
        &mut context(&graph),
    )?;
    let ResultValue::Relationship(relationship) = one_value(&created, "r")? else {
        return Err(Error::internal("ON CREATE did not return a relationship"));
    };
    assert_eq!(relationship.properties, expected);
    assert_eq!(created.result.statistics.relationships_created, 1);
    assert_eq!(created.result.statistics.properties_set, 2);

    let graph = merge_fixture(true)?;
    let matched = QueryEngine.execute(
        "MATCH (a:A), (b:B) \
         MERGE (a)-[r:TYPE]->(b) \
         ON MATCH SET r = a \
         RETURN r",
        &mut context(&graph),
    )?;
    let ResultValue::Relationship(relationship) = one_value(&matched, "r")? else {
        return Err(Error::internal("ON MATCH did not return a relationship"));
    };
    assert_eq!(relationship.properties, expected);
    assert_eq!(matched.result.statistics.relationships_created, 0);
    assert_eq!(matched.result.statistics.properties_set, 3);
    assert!(matched.graph_mutations.iter().any(|mutation| matches!(
        mutation,
        GraphMutation::SetEdgeProperty {
            value: ScalarValue::Null,
            ..
        }
    )));
    Ok(())
}

#[test]
fn set_replace_map_copies_a_relationships_overlay_visible_property_map() -> Result<()> {
    let graph = relationship_source_fixture()?;
    let output = QueryEngine.execute(
        "MATCH (:A)-[source:SOURCE]->(b:B) \
         SET source.weight = 9, b = source \
         RETURN b",
        &mut context(&graph),
    )?;
    let ResultValue::Node(node) = one_value(&output, "b")? else {
        return Err(Error::internal("SET entity map did not return a node"));
    };
    assert_eq!(
        node.properties,
        BTreeMap::from([("weight".to_owned(), ScalarValue::Integer(9))])
    );
    assert_eq!(output.result.statistics.properties_set, 4);
    assert!(output.graph_mutations.iter().any(|mutation| matches!(
        mutation,
        GraphMutation::SetNodeProperty {
            node: NodeId(2),
            value: ScalarValue::Integer(9),
            ..
        }
    )));
    Ok(())
}

#[test]
fn path_functions_treat_projected_and_internal_null_bindings_identically() -> Result<()> {
    let graph = GraphStore::default();
    for function in ["nodes", "relationships"] {
        let query = format!(
            "WITH null AS a \
             OPTIONAL MATCH p = (a)-[r]->() \
             RETURN {function}(p) AS fromPath, {function}(null) AS fromLiteral"
        );
        let output = QueryEngine.execute(&query, &mut context(&graph))?;
        assert_eq!(
            one_value(&output, "fromPath")?,
            &ResultValue::Scalar(ScalarValue::Null),
            "query: {query}"
        );
        assert_eq!(
            one_value(&output, "fromLiteral")?,
            &ResultValue::Scalar(ScalarValue::Null),
            "query: {query}"
        );
        assert!(output.graph_mutations.is_empty(), "query: {query}");
    }
    Ok(())
}
