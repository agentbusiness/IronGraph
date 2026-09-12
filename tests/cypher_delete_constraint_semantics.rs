// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{collections::BTreeMap, time::Instant};

use irongraph::{
    Bookmark, EdgeId, Error, ErrorCode, Layer, NodeId, ProjectId, Result,
    cypher::{BindCapabilities, ExecutionContext, QueryEngine},
    graph::{EdgeInput, GraphMutation, GraphStore, NodeInput},
};
use tokio_util::sync::CancellationToken;

fn graph_with_incident_relationship() -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    let source = graph.catalog_mut().intern_label("Source")?;
    let target = graph.catalog_mut().intern_label("Target")?;
    let link = graph.catalog_mut().intern_relationship_type("LINK")?;

    graph.insert_node(NodeInput {
        id: NodeId(1),
        layer: Layer::Observed,
        revision: 1,
        labels: vec![source],
        properties: Vec::new(),
    })?;
    graph.insert_node(NodeInput {
        id: NodeId(2),
        layer: Layer::Observed,
        revision: 2,
        labels: vec![target],
        properties: Vec::new(),
    })?;
    graph.insert_edge(EdgeInput {
        id: EdgeId(1),
        source: NodeId(1),
        target: NodeId(2),
        relationship_type: link,
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
        bookmark: Bookmark { term: 1, index: 3 },
        mutation_revision: 4,
        resolved_time_nanos: 0,
        next_node_id: 3,
        next_edge_id: 2,
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

#[test]
fn statement_overlay_reports_attached_node_delete_as_query_type() -> Result<()> {
    let graph = graph_with_incident_relationship()?;
    let error = QueryEngine
        .execute("MATCH (n:Source) DELETE n", &mut context(&graph))
        .err()
        .ok_or_else(|| Error::internal("ordinary DELETE unexpectedly succeeded"))?;

    assert_eq!(error.code, ErrorCode::QueryType);
    assert!(error.message.contains("DeleteConnectedNode"));
    assert!(graph.node(NodeId(1)).is_some());
    assert!(graph.edge(EdgeId(1)).is_some());
    Ok(())
}

#[test]
fn statement_overlay_allows_detach_delete() -> Result<()> {
    let graph = graph_with_incident_relationship()?;
    let output = QueryEngine.execute("MATCH (n:Source) DETACH DELETE n", &mut context(&graph))?;

    assert!(output.graph_mutations.iter().any(|mutation| matches!(
        mutation,
        GraphMutation::DeleteNode {
            node: NodeId(1),
            detach: true,
            revision: 4,
        }
    )));
    assert_eq!(output.result.statistics.nodes_deleted, 1);
    Ok(())
}

#[test]
fn canonical_store_reports_attached_node_delete_as_query_type() -> Result<()> {
    let mut graph = graph_with_incident_relationship()?;
    let error = graph
        .delete_node(NodeId(1), false, 4)
        .err()
        .ok_or_else(|| Error::internal("canonical ordinary DELETE unexpectedly succeeded"))?;

    assert_eq!(error.code, ErrorCode::QueryType);
    assert_eq!(error.message, "node still has relationships");
    assert!(graph.node(NodeId(1)).is_some());
    assert!(graph.edge(EdgeId(1)).is_some());
    Ok(())
}

#[test]
fn canonical_store_allows_detach_delete() -> Result<()> {
    let mut graph = graph_with_incident_relationship()?;
    graph.delete_node(NodeId(1), true, 4)?;

    assert!(graph.node(NodeId(1)).is_none());
    assert!(graph.edge(EdgeId(1)).is_none());
    assert!(graph.node(NodeId(2)).is_some());
    Ok(())
}
