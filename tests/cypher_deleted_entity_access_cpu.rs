// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{collections::BTreeMap, time::Instant};

use irongraph::{
    Bookmark, EdgeId, Error, ErrorCode, Layer, NodeId, ProjectId, Result, ScalarValue,
    cypher::{
        BindCapabilities, ExecutionContext, ExecutionOutput, QueryEngine, ResultValue,
        bind_with_parameters, parse, plan,
    },
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
        bookmark: Bookmark {
            term: 1,
            index: graph.revision(),
        },
        mutation_revision: graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id: 100,
        next_edge_id: 100,
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

fn assert_reaches_runtime(query: &str, graph: &GraphStore) -> Result<()> {
    let parameters = BTreeMap::new();
    let capabilities = BindCapabilities {
        write: true,
        ..BindCapabilities::default()
    };
    plan(bind_with_parameters(
        parse(query)?,
        graph.catalog(),
        capabilities,
        &parameters,
    )?)?;
    Ok(())
}

fn require_runtime_error(query: &str, graph: &GraphStore) -> Result<Error> {
    assert_reaches_runtime(query, graph)?;
    QueryEngine
        .execute(query, &mut context(graph))
        .err()
        .ok_or_else(|| Error::internal(format!("query unexpectedly succeeded: {query}")))
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

fn node_graph(labels: &[&str], property: Option<(&str, ScalarValue)>) -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    let labels = labels
        .iter()
        .map(|label| graph.catalog_mut().intern_label(label))
        .collect::<Result<Vec<_>>>()?;
    let properties = property
        .map(|(name, value)| {
            graph
                .catalog_mut()
                .intern_property(name)
                .map(|property| vec![(property, value)])
        })
        .transpose()?
        .unwrap_or_default();
    graph.insert_node(NodeInput {
        id: NodeId(1),
        layer: Layer::Observed,
        revision: 1,
        labels,
        properties,
    })?;
    Ok(graph)
}

fn relationship_graph(property: Option<(&str, ScalarValue)>) -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    let relationship_type = graph.catalog_mut().intern_relationship_type("T")?;
    let properties = property
        .map(|(name, value)| {
            graph
                .catalog_mut()
                .intern_property(name)
                .map(|property| vec![(property, value)])
        })
        .transpose()?
        .unwrap_or_default();
    graph.insert_node(NodeInput {
        id: NodeId(1),
        layer: Layer::Observed,
        revision: 1,
        labels: Vec::new(),
        properties: Vec::new(),
    })?;
    graph.insert_node(NodeInput {
        id: NodeId(2),
        layer: Layer::Observed,
        revision: 2,
        labels: Vec::new(),
        properties: Vec::new(),
    })?;
    graph.insert_edge(EdgeInput {
        id: EdgeId(1),
        source: NodeId(1),
        target: NodeId(2),
        relationship_type,
        layer: Layer::Observed,
        revision: 3,
        properties,
    })?;
    Ok(graph)
}

fn assert_deleted_entity_access(error: &Error, query: &str) {
    assert_eq!(error.code, ErrorCode::QueryType, "{query}: {error:?}");
    assert!(
        error.message.contains("DeletedEntityAccess"),
        "{query}: {error:?}"
    );
}

#[test]
fn return2_14_type_of_deleted_relationship_remains_readable() -> Result<()> {
    let graph = relationship_graph(None)?;
    let query = "MATCH ()-[r]->() DELETE r RETURN type(r)";
    assert_reaches_runtime(query, &graph)?;

    let output = QueryEngine.execute(query, &mut context(&graph))?;
    assert_eq!(
        column_values(&output, "type(r)")?,
        vec![ResultValue::Scalar(ScalarValue::String("T".into()))]
    );
    assert_eq!(output.result.statistics.relationships_deleted, 1);
    assert!(output.graph_mutations.iter().any(|mutation| matches!(
        mutation,
        GraphMutation::DeleteEdge {
            edge: EdgeId(1),
            ..
        }
    )));
    Ok(())
}

#[test]
fn return2_15_deleted_node_property_is_a_runtime_deleted_entity_access() -> Result<()> {
    let graph = node_graph(&[], Some(("num", ScalarValue::Integer(0))))?;
    let query = "MATCH (n) DELETE n RETURN n.num";
    let error = require_runtime_error(query, &graph)?;
    assert_deleted_entity_access(&error, query);
    Ok(())
}

#[test]
fn return2_16_deleted_node_labels_are_a_runtime_deleted_entity_access() -> Result<()> {
    let graph = node_graph(&["A"], None)?;
    let query = "MATCH (n) DELETE n RETURN labels(n)";
    let error = require_runtime_error(query, &graph)?;
    assert_deleted_entity_access(&error, query);
    Ok(())
}

#[test]
fn return2_17_deleted_relationship_property_is_a_runtime_deleted_entity_access() -> Result<()> {
    let graph = relationship_graph(Some(("num", ScalarValue::Integer(0))))?;
    let query = "MATCH ()-[r]->() DELETE r RETURN r.num";
    let error = require_runtime_error(query, &graph)?;
    assert_deleted_entity_access(&error, query);
    Ok(())
}

#[test]
fn optional_null_and_live_missing_properties_keep_null_semantics() -> Result<()> {
    let empty = GraphStore::default();
    let optional = QueryEngine.execute(
        "OPTIONAL MATCH (n:Missing) RETURN n.num AS property, labels(n) AS labels",
        &mut context(&empty),
    )?;
    assert_eq!(
        column_values(&optional, "property")?,
        vec![ResultValue::Scalar(ScalarValue::Null)]
    );
    assert_eq!(
        column_values(&optional, "labels")?,
        vec![ResultValue::Scalar(ScalarValue::Null)]
    );

    let node = node_graph(&[], None)?;
    let missing = QueryEngine.execute(
        "MATCH (n) RETURN n.missing AS property",
        &mut context(&node),
    )?;
    assert_eq!(
        column_values(&missing, "property")?,
        vec![ResultValue::Scalar(ScalarValue::Null)]
    );
    Ok(())
}
