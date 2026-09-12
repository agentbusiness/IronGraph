// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{collections::BTreeMap, time::Instant};

use irongraph::{
    Bookmark, EdgeId, Layer, NodeId, ProjectId, Result, ScalarValue,
    cypher::{BindCapabilities, ExecutionContext, QueryEngine},
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

fn graph_with_live_and_null_optional_relationship_rows() -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    let entity = graph.catalog_mut().intern_label("Entity")?;
    let isolated = graph.catalog_mut().intern_label("Isolated")?;
    let relationship_type = graph.catalog_mut().intern_relationship_type("R")?;
    let property = graph.catalog_mut().intern_property("num")?;
    for (id, labels) in [
        (1, vec![entity]),
        (2, vec![entity]),
        (3, vec![entity, isolated]),
    ] {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: id,
            labels,
            properties: Vec::new(),
        })?;
    }
    graph.insert_edge(EdgeInput {
        id: EdgeId(1),
        source: NodeId(1),
        target: NodeId(2),
        relationship_type,
        layer: Layer::Observed,
        revision: 4,
        properties: vec![(property, ScalarValue::Integer(42))],
    })?;
    Ok(graph)
}

#[test]
fn optional_null_relationship_property_and_node_label_removals_are_noops() -> Result<()> {
    let graph = graph_with_live_and_null_optional_relationship_rows()?;

    let relationship = QueryEngine.execute(
        "MATCH (n:Isolated) OPTIONAL MATCH (n)-[r]->() REMOVE r.num RETURN n",
        &mut context(&graph),
    )?;
    assert!(relationship.graph_mutations.is_empty());
    assert_eq!(relationship.result.statistics.properties_set, 0);

    let label = QueryEngine.execute(
        "OPTIONAL MATCH (a:DoesNotExist) REMOVE a:Entity RETURN a",
        &mut context(&graph),
    )?;
    assert!(label.graph_mutations.is_empty());
    assert_eq!(label.result.statistics.labels_removed, 0);
    Ok(())
}

#[test]
fn mixed_live_and_null_optional_rows_mutate_only_the_live_relationship() -> Result<()> {
    let graph = graph_with_live_and_null_optional_relationship_rows()?;
    let output = QueryEngine.execute(
        "MATCH (n:Entity) OPTIONAL MATCH (n)-[r:R]->() REMOVE r.num RETURN count(*) AS rows",
        &mut context(&graph),
    )?;

    assert_eq!(output.result.statistics.properties_set, 1);
    assert_eq!(
        output
            .graph_mutations
            .iter()
            .filter(|mutation| matches!(
                mutation,
                GraphMutation::SetEdgeProperty {
                    edge: EdgeId(1),
                    value: ScalarValue::Null,
                    ..
                }
            ))
            .count(),
        1
    );
    assert_eq!(output.graph_mutations.len(), 1);

    let mut committed = graph.clone();
    for mutation in output.graph_mutations {
        committed.apply(mutation)?;
    }
    assert!(
        committed
            .edge(EdgeId(1))
            .is_some_and(|edge| edge.properties().is_empty())
    );
    Ok(())
}
