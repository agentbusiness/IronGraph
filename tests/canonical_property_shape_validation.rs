// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{collections::BTreeMap, sync::Arc};

use irongraph::{
    Bookmark, DocumentItem, DocumentList, DocumentMap, EdgeId, Error, ErrorCode, Layer, NodeId,
    ProjectId, Result, ScalarValue,
    cypher::{BindCapabilities, ExecutionContext, QueryEngine},
    graph::{EdgeInput, GraphMutation, GraphStore, NodeInput},
};
use tokio_util::sync::CancellationToken;

fn execution_context(graph: &GraphStore) -> ExecutionContext<'_> {
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
        bookmark: Bookmark { term: 0, index: 0 },
        mutation_revision: 1,
        resolved_time_nanos: 0,
        next_node_id: 1,
        next_edge_id: 1,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            write: true,
            ..BindCapabilities::default()
        },
        max_result_rows: 100,
        max_batch_rows: 100,
        optimizer_statistics: None,
        backend: None,
        cancellation: CancellationToken::new(),
        deadline: None,
        resolved_query_at_time_nanos: None,
    }
}

fn require_query_type<T>(result: Result<T>, operation: &str) -> Result<()> {
    match result {
        Err(error)
            if error.code == ErrorCode::QueryType
                && error.message.contains("InvalidPropertyType") =>
        {
            Ok(())
        }
        Err(error) => Err(Error::internal(format!(
            "{operation} returned {:?} `{}` instead of InvalidPropertyType",
            error.code, error.message
        ))),
        Ok(_) => Err(Error::internal(format!(
            "{operation} accepted an invalid property value"
        ))),
    }
}

fn require(condition: bool, message: &'static str) -> Result<()> {
    if condition {
        Ok(())
    } else {
        Err(Error::internal(message))
    }
}

fn list(values: Vec<DocumentItem>) -> Result<ScalarValue> {
    Ok(ScalarValue::List(DocumentList::new(values)?))
}

fn map() -> Result<ScalarValue> {
    Ok(ScalarValue::Map(DocumentMap::new(BTreeMap::from([
        (
            Arc::from("nested"),
            DocumentItem::Map(BTreeMap::from([(
                Arc::from("num"),
                DocumentItem::Scalar(ScalarValue::Integer(1)),
            )])),
        ),
        (
            Arc::from("values"),
            DocumentItem::List(vec![
                DocumentItem::Scalar(ScalarValue::String(Arc::from("x"))),
                DocumentItem::Scalar(ScalarValue::Null),
            ]),
        ),
    ]))?))
}

fn map_item() -> DocumentItem {
    DocumentItem::Map(BTreeMap::from([(
        Arc::from("num"),
        DocumentItem::Scalar(ScalarValue::Integer(1)),
    )]))
}

#[test]
fn every_canonical_graph_write_rejects_non_property_shapes() -> Result<()> {
    let mut graph = GraphStore::default();
    let property = graph.catalog_mut().intern_property("value")?;
    let relationship_type = graph.catalog_mut().intern_relationship_type("REL")?;

    require_query_type(
        graph.apply(GraphMutation::InsertNode(NodeInput {
            id: NodeId(1),
            layer: Layer::Observed,
            revision: 1,
            labels: Vec::new(),
            properties: vec![(property, list(vec![map_item()])?)],
        })),
        "node insertion",
    )?;
    require(
        !graph.contains_node_id(NodeId(1)),
        "rejected node insertion changed canonical state",
    )?;

    for (id, revision) in [(NodeId(1), 1), (NodeId(2), 2)] {
        graph.insert_node(NodeInput {
            id,
            layer: Layer::Observed,
            revision,
            labels: Vec::new(),
            properties: Vec::new(),
        })?;
    }
    graph.insert_edge(EdgeInput {
        id: EdgeId(1),
        source: NodeId(1),
        target: NodeId(2),
        relationship_type,
        layer: Layer::Observed,
        revision: 3,
        properties: Vec::new(),
    })?;

    require_query_type(
        graph.set_node_property(
            NodeId(1),
            property,
            list(vec![DocumentItem::List(vec![DocumentItem::Scalar(
                ScalarValue::Integer(1),
            )])])?,
            4,
        ),
        "node property update",
    )?;
    require_query_type(
        graph.set_edge_property(EdgeId(1), property, list(vec![map_item()])?, 4),
        "relationship property update",
    )?;
    require_query_type(
        graph.insert_edge(EdgeInput {
            id: EdgeId(2),
            source: NodeId(1),
            target: NodeId(2),
            relationship_type,
            layer: Layer::Observed,
            revision: 4,
            properties: vec![(
                property,
                list(vec![
                    DocumentItem::Scalar(ScalarValue::Integer(1)),
                    DocumentItem::Scalar(ScalarValue::Boolean(true)),
                ])?,
            )],
        }),
        "relationship insertion",
    )?;
    require_query_type(
        graph.apply(GraphMutation::SetNodeProperty {
            node: NodeId(1),
            property,
            value: list(vec![DocumentItem::Scalar(ScalarValue::Null)])?,
            revision: 4,
        }),
        "published node property update",
    )?;

    require(
        graph
            .node(NodeId(1))
            .is_some_and(|node| node.property(property).is_none()),
        "rejected node property update changed canonical state",
    )?;
    require(
        graph
            .edge(EdgeId(1))
            .is_some_and(|edge| edge.property(property).is_none()),
        "rejected relationship property update changed canonical state",
    )?;
    require(
        !graph.contains_edge_id(EdgeId(2)),
        "rejected relationship insertion changed canonical state",
    )
}

#[test]
fn property_lists_reject_null_and_mixed_scalar_elements() -> Result<()> {
    let mut graph = GraphStore::default();
    let property = graph.catalog_mut().intern_property("value")?;

    for (description, value) in [
        (
            "NULL list element",
            list(vec![DocumentItem::Scalar(ScalarValue::Null)])?,
        ),
        (
            "mixed numeric list",
            list(vec![
                DocumentItem::Scalar(ScalarValue::Integer(1)),
                DocumentItem::Scalar(ScalarValue::Float(1.0.into())),
            ])?,
        ),
        (
            "nested list element",
            list(vec![DocumentItem::List(vec![DocumentItem::Scalar(
                ScalarValue::Integer(1),
            )])])?,
        ),
        ("map element", list(vec![map_item()])?),
    ] {
        require_query_type(
            graph.validate_node_property_value(property, &value),
            description,
        )?;
        require_query_type(
            graph.validate_edge_property_value(property, &value),
            description,
        )?;
    }
    Ok(())
}

#[test]
fn root_maps_with_nested_document_content_remain_valid_properties() -> Result<()> {
    let mut graph = GraphStore::default();
    let property = graph.catalog_mut().intern_property("payload")?;
    let relationship_type = graph.catalog_mut().intern_relationship_type("REL")?;
    let value = map()?;

    graph.validate_node_property_value(property, &value)?;
    graph.validate_edge_property_value(property, &value)?;
    graph.insert_node(NodeInput {
        id: NodeId(1),
        layer: Layer::Observed,
        revision: 1,
        labels: Vec::new(),
        properties: vec![(property, value.clone())],
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
        properties: vec![(property, value.clone())],
    })?;

    require(
        graph
            .node(NodeId(1))
            .is_some_and(|node| node.property(property) == Some(value.clone())),
        "nested root map did not round-trip on a node",
    )?;
    require(
        graph
            .edge(EdgeId(1))
            .is_some_and(|edge| edge.property(property) == Some(value)),
        "nested root map did not round-trip on a relationship",
    )
}

#[test]
fn homogeneous_scalar_lists_remain_valid_on_nodes_and_relationships() -> Result<()> {
    let mut graph = GraphStore::default();
    let property = graph.catalog_mut().intern_property("values")?;
    let relationship_type = graph.catalog_mut().intern_relationship_type("REL")?;
    let integers = list(vec![
        DocumentItem::Scalar(ScalarValue::Integer(1)),
        DocumentItem::Scalar(ScalarValue::Integer(2)),
        DocumentItem::Scalar(ScalarValue::Integer(3)),
    ])?;
    let strings = list(vec![
        DocumentItem::Scalar(ScalarValue::String(Arc::from("a"))),
        DocumentItem::Scalar(ScalarValue::String(Arc::from("b"))),
    ])?;

    graph.insert_node(NodeInput {
        id: NodeId(1),
        layer: Layer::Observed,
        revision: 1,
        labels: Vec::new(),
        properties: vec![(property, integers.clone())],
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
        properties: vec![(property, strings.clone())],
    })?;

    require(
        graph
            .node(NodeId(1))
            .is_some_and(|node| node.property(property) == Some(integers)),
        "homogeneous integer list did not round-trip",
    )?;
    require(
        graph
            .edge(EdgeId(1))
            .is_some_and(|edge| edge.property(property) == Some(strings)),
        "homogeneous string list did not round-trip",
    )?;

    let empty = list(Vec::new())?;
    graph.set_node_property(NodeId(2), property, empty.clone(), 4)?;
    require(
        graph
            .node(NodeId(2))
            .is_some_and(|node| node.property(property) == Some(empty)),
        "empty property list did not round-trip",
    )
}

#[test]
fn query_execution_rejects_a_list_of_maps_before_returning_mutations() -> Result<()> {
    let graph = GraphStore::default();
    let mut context = execution_context(&graph);
    require_query_type(
        QueryEngine.execute("CREATE (a) SET a.maplist = [{num: 1}]", &mut context),
        "query execution",
    )
}
