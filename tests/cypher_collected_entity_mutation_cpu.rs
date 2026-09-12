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

fn sample_graph() -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    let entity = graph.catalog_mut().intern_label("Entity")?;
    let link = graph.catalog_mut().intern_relationship_type("LINK")?;
    let name = graph.catalog_mut().intern_property("name")?;
    let weight = graph.catalog_mut().intern_property("weight")?;
    for (id, value) in [(1, "first"), (2, "second")] {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: id,
            labels: vec![entity],
            properties: vec![(name, ScalarValue::String(value.into()))],
        })?;
    }
    graph.insert_edge(EdgeInput {
        id: EdgeId(1),
        source: NodeId(1),
        target: NodeId(2),
        relationship_type: link,
        layer: Layer::Observed,
        revision: 3,
        properties: vec![(weight, ScalarValue::Integer(1))],
    })?;
    Ok(graph)
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
fn collected_nodes_retain_identity_through_comprehension_map_and_unwind() -> Result<()> {
    let graph = sample_graph()?;
    let output = QueryEngine.execute(
        "MATCH (a:Entity) \
         WITH [x IN collect(a) | {entity: x}] AS wrapped \
         UNWIND wrapped AS item \
         WITH item.entity AS n \
         SET n.name = 'newName' \
         RETURN n.name AS name ORDER BY name",
        &mut context(&graph),
    )?;

    assert_eq!(
        column_values(&output, "name")?,
        vec![
            ResultValue::Scalar(ScalarValue::String("newName".into())),
            ResultValue::Scalar(ScalarValue::String("newName".into())),
        ]
    );
    assert_eq!(output.result.statistics.properties_set, 2);
    assert_eq!(
        output
            .graph_mutations
            .iter()
            .filter(|mutation| matches!(mutation, GraphMutation::SetNodeProperty { .. }))
            .count(),
        2
    );
    Ok(())
}

#[test]
fn collected_relationships_retain_identity_for_set_read_and_delete() -> Result<()> {
    let graph = sample_graph()?;
    let output = QueryEngine.execute(
        "MATCH ()-[r:LINK]->() \
         WITH [x IN collect(r) | x] AS relationships \
         UNWIND relationships AS rel \
         SET rel.weight = 2 \
         WITH rel, rel.weight AS observed \
         DELETE rel \
         RETURN observed",
        &mut context(&graph),
    )?;

    assert_eq!(
        one_value(&output, "observed")?,
        ResultValue::Scalar(ScalarValue::Integer(2))
    );
    assert!(output.graph_mutations.iter().any(|mutation| matches!(
        mutation,
        GraphMutation::SetEdgeProperty {
            edge: EdgeId(1),
            value: ScalarValue::Integer(2),
            ..
        }
    )));
    assert!(output.graph_mutations.iter().any(|mutation| matches!(
        mutation,
        GraphMutation::DeleteEdge {
            edge: EdgeId(1),
            ..
        }
    )));
    assert_eq!(output.result.statistics.properties_set, 1);
    assert_eq!(output.result.statistics.relationships_deleted, 1);
    Ok(())
}

#[test]
fn public_result_entities_do_not_gain_mutation_authority_in_a_new_statement() -> Result<()> {
    let graph = sample_graph()?;
    let public_node = one_value(
        &QueryEngine.execute(
            "MATCH (n:Entity) RETURN n ORDER BY n.name LIMIT 1",
            &mut context(&graph),
        )?,
        "n",
    )?;
    let mut mutation_context = context(&graph);
    mutation_context.parameters =
        BTreeMap::from([("entities".to_owned(), ResultValue::List(vec![public_node]))]);
    let error = QueryEngine
        .execute(
            "UNWIND $entities AS n SET n.name = 'forged' RETURN n",
            &mut mutation_context,
        )
        .err()
        .ok_or_else(|| Error::internal("public ResultNode unexpectedly became mutable"))?;
    assert_eq!(error.code, ErrorCode::QueryType);
    assert_eq!(error.message, "SET target must be a node or relationship");
    Ok(())
}

#[test]
fn tolower_is_the_exact_executor_alias_of_lower() -> Result<()> {
    let graph = GraphStore::default();
    let output = QueryEngine.execute(
        "RETURN lower('MiXeD') AS canonical, toLower('MiXeD') AS alias",
        &mut context(&graph),
    )?;
    assert_eq!(
        one_value(&output, "canonical")?,
        ResultValue::Scalar(ScalarValue::String("mixed".into()))
    );
    assert_eq!(
        one_value(&output, "alias")?,
        one_value(&output, "canonical")?
    );
    Ok(())
}
