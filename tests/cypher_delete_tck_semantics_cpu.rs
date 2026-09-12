// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{collections::BTreeMap, time::Instant};

use irongraph::{
    Bookmark, EdgeId, Error, ErrorCode, Layer, NodeId, ProjectId, Result, ScalarValue,
    cypher::{BindCapabilities, ExecutionContext, QueryEngine, ResultValue},
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

fn insert_node(
    graph: &mut GraphStore,
    id: u64,
    labels: Vec<irongraph::types::LabelId>,
    properties: Vec<(irongraph::types::PropertyId, ScalarValue)>,
) -> Result<()> {
    graph.insert_node(NodeInput {
        id: NodeId(id),
        layer: Layer::Observed,
        revision: id,
        labels,
        properties,
    })?;
    Ok(())
}

fn insert_edge(
    graph: &mut GraphStore,
    id: u64,
    source: u64,
    target: u64,
    relationship_type: irongraph::types::RelationshipTypeId,
) -> Result<()> {
    graph.insert_edge(EdgeInput {
        id: EdgeId(id),
        source: NodeId(source),
        target: NodeId(target),
        relationship_type,
        layer: Layer::Observed,
        revision: 10_u64.saturating_add(id),
        properties: Vec::new(),
    })?;
    Ok(())
}

fn integer_column(output: &irongraph::cypher::ExecutionOutput, name: &str) -> Result<i64> {
    let value = output
        .result
        .batches
        .first()
        .and_then(|batch| batch.columns.iter().find(|column| column.name == name))
        .and_then(|column| column.values.first())
        .ok_or_else(|| Error::internal(format!("missing `{name}` result")))?;
    match value {
        ResultValue::Scalar(ScalarValue::Integer(value)) => Ok(*value),
        _ => Err(Error::internal(format!("`{name}` is not an INTEGER"))),
    }
}

fn apply_output(graph: &GraphStore, mutations: &[GraphMutation]) -> Result<GraphStore> {
    let mut committed = graph.clone();
    for mutation in mutations {
        committed.apply(mutation.clone())?;
    }
    Ok(committed)
}

#[test]
fn undirected_duplicate_rows_delete_each_stable_entity_once() -> Result<()> {
    let mut graph = GraphStore::default();
    let relationship_type = graph.catalog_mut().intern_relationship_type("R")?;
    insert_node(&mut graph, 1, Vec::new(), Vec::new())?;
    insert_node(&mut graph, 2, Vec::new(), Vec::new())?;
    insert_edge(&mut graph, 1, 1, 2, relationship_type)?;

    let output = QueryEngine.execute(
        "MATCH (a)-[r]-(b) DELETE r, a, b RETURN count(*) AS rows",
        &mut context(&graph),
    )?;

    assert_eq!(integer_column(&output, "rows")?, 2);
    assert_eq!(output.result.statistics.nodes_deleted, 2);
    assert_eq!(output.result.statistics.relationships_deleted, 1);
    assert_eq!(output.graph_mutations.len(), 3);
    assert!(matches!(
        output.graph_mutations.first(),
        Some(GraphMutation::DeleteEdge {
            edge: EdgeId(1),
            ..
        })
    ));
    let committed = apply_output(&graph, &output.graph_mutations)?;
    assert_eq!(committed.node_count(), 0);
    assert_eq!(committed.edge_count(), 0);
    Ok(())
}

#[test]
fn repeated_path_recovered_through_nested_map_and_list_is_deduplicated() -> Result<()> {
    let mut graph = GraphStore::default();
    let user = graph.catalog_mut().intern_label("User")?;
    let relationship_type = graph.catalog_mut().intern_relationship_type("R")?;
    insert_node(&mut graph, 1, vec![user], Vec::new())?;
    insert_node(&mut graph, 2, vec![user], Vec::new())?;
    insert_edge(&mut graph, 1, 1, 2, relationship_type)?;

    let output = QueryEngine.execute(
        "MATCH p = (:User)-[:R]->(:User) \
         WITH {nested: {paths: [p, p]}} AS value \
         DELETE value.nested.paths[0], value.nested.paths[1]",
        &mut context(&graph),
    )?;

    assert_eq!(output.result.statistics.nodes_deleted, 2);
    assert_eq!(output.result.statistics.relationships_deleted, 1);
    assert_eq!(output.graph_mutations.len(), 3);
    let committed = apply_output(&graph, &output.graph_mutations)?;
    assert_eq!(committed.node_count(), 0);
    assert_eq!(committed.edge_count(), 0);
    Ok(())
}

#[test]
fn detach_delete_path_removes_incident_relationships_outside_the_path() -> Result<()> {
    let mut graph = GraphStore::default();
    let start = graph.catalog_mut().intern_label("Start")?;
    let path_type = graph.catalog_mut().intern_relationship_type("PATH")?;
    let extra_type = graph.catalog_mut().intern_relationship_type("EXTRA")?;
    insert_node(&mut graph, 1, vec![start], Vec::new())?;
    insert_node(&mut graph, 2, Vec::new(), Vec::new())?;
    insert_node(&mut graph, 3, Vec::new(), Vec::new())?;
    insert_edge(&mut graph, 1, 1, 2, path_type)?;
    insert_edge(&mut graph, 2, 2, 3, extra_type)?;

    let output = QueryEngine.execute(
        "MATCH p = (:Start)-[:PATH]->() DETACH DELETE p",
        &mut context(&graph),
    )?;
    assert_eq!(output.result.statistics.nodes_deleted, 2);
    assert_eq!(output.result.statistics.relationships_deleted, 1);

    let committed = apply_output(&graph, &output.graph_mutations)?;
    assert!(committed.node(NodeId(1)).is_none());
    assert!(committed.node(NodeId(2)).is_none());
    assert!(committed.node(NodeId(3)).is_some());
    assert_eq!(committed.edge_count(), 0);
    Ok(())
}

#[test]
fn real_connected_node_failure_has_stable_detail_and_deleted_access_stays_distinct() -> Result<()> {
    let mut graph = GraphStore::default();
    let start = graph.catalog_mut().intern_label("Start")?;
    let relationship_type = graph.catalog_mut().intern_relationship_type("R")?;
    let name = graph.catalog_mut().intern_property("name")?;
    insert_node(
        &mut graph,
        1,
        vec![start],
        vec![(name, ScalarValue::String("one".into()))],
    )?;
    insert_node(&mut graph, 2, Vec::new(), Vec::new())?;
    insert_edge(&mut graph, 1, 1, 2, relationship_type)?;

    let connected = QueryEngine
        .execute("MATCH (n:Start) DELETE n", &mut context(&graph))
        .err()
        .ok_or_else(|| Error::internal("connected node deletion unexpectedly succeeded"))?;
    assert_eq!(connected.code, ErrorCode::QueryType);
    assert!(connected.message.contains("DeleteConnectedNode"));

    let deleted_access = QueryEngine
        .execute(
            "MATCH (n:Start) DETACH DELETE n RETURN n.name",
            &mut context(&graph),
        )
        .err()
        .ok_or_else(|| Error::internal("deleted node property access unexpectedly succeeded"))?;
    assert_eq!(deleted_access.code, ErrorCode::QueryType);
    assert!(deleted_access.message.contains("DeletedEntityAccess"));
    Ok(())
}
