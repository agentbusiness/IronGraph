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
        BindCapabilities, ExecutionContext, ExecutionOutput, PhysicalOperator, QueryEngine,
        ResultValue, StatementStats, bind, parse, plan,
    },
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

fn assert_read_only(output: &ExecutionOutput) {
    assert!(output.graph_mutations.is_empty());
    assert!(output.temporal_mutations.is_empty());
    assert_eq!(output.result.statistics, StatementStats::default());
}

struct NodeSnapshot {
    id: NodeId,
    labels: Vec<String>,
    properties: BTreeMap<String, ScalarValue>,
}

fn only_node(output: &ExecutionOutput, column: &str) -> Result<NodeSnapshot> {
    let values = column_values(output, column)?;
    let value = values
        .into_iter()
        .next()
        .ok_or_else(|| Error::internal(format!("result column `{column}` was empty")))?;
    let ResultValue::Node(node) = value else {
        return Err(Error::internal(format!(
            "result column `{column}` did not contain a node"
        )));
    };
    Ok(NodeSnapshot {
        id: node.id,
        labels: node.labels,
        properties: node.properties,
    })
}

#[test]
fn tck_with_where1_scenario_1_filters_one_node_from_multiple_bindings() -> Result<()> {
    let mut graph = GraphStore::default();
    for (id, name) in [(1, "A"), (2, "B"), (3, "C")] {
        insert_node(
            &mut graph,
            id,
            &[],
            &[("name", ScalarValue::String(name.into()))],
        )?;
    }

    let output = QueryEngine.execute(
        "MATCH (a)\nWITH a\nWHERE a.name = 'B'\nRETURN a",
        &mut context(&graph),
    )?;

    assert_eq!(column_values(&output, "a")?.len(), 1);
    let node = only_node(&output, "a")?;
    assert_eq!(node.id, NodeId(2));
    assert!(node.labels.is_empty());
    assert_eq!(
        node.properties,
        BTreeMap::from([("name".to_owned(), ScalarValue::String("B".into()))])
    );
    assert_read_only(&output);
    Ok(())
}

#[test]
fn with_where_distinct_filters_before_deduplicating_only_visible_values() -> Result<()> {
    let mut graph = GraphStore::default();
    for (id, selected) in [(1, true), (2, true), (3, false)] {
        insert_node(
            &mut graph,
            id,
            &[],
            &[
                ("name", ScalarValue::String("B".into())),
                ("selected", ScalarValue::Boolean(selected)),
            ],
        )?;
    }

    let source =
        "MATCH (a) WITH DISTINCT a.name AS name WHERE name = 'B' AND a.selected RETURN name";
    let output = QueryEngine.execute(source, &mut context(&graph))?;
    assert_eq!(
        column_values(&output, "name")?,
        vec![ResultValue::Scalar(ScalarValue::String("B".into()))]
    );
    assert_read_only(&output);

    let bound = bind(parse(source)?, graph.catalog(), BindCapabilities::default())?;
    let physical = plan(bound)?;
    let tail = &physical.operators[physical.operators.len().saturating_sub(4)..];
    let [
        PhysicalOperator::Project {
            keep_scope: true,
            projection: materialized,
        },
        PhysicalOperator::Filter(_),
        PhysicalOperator::Project {
            keep_scope: false,
            projection: visible,
        },
        PhysicalOperator::Project { .. },
    ] = tail
    else {
        return Err(Error::internal(format!(
            "WITH DISTINCT WHERE had the wrong phase order: {tail:?}"
        )));
    };
    assert!(!materialized.distinct);
    assert!(visible.distinct);
    Ok(())
}

#[test]
fn tck_with_where1_scenario_2_filters_an_incoming_value_before_distinct_projection() -> Result<()> {
    let mut graph = GraphStore::default();
    for (id, name) in [(1, "A"), (2, "A"), (3, "B")] {
        insert_node(
            &mut graph,
            id,
            &[],
            &[("name2", ScalarValue::String(name.into()))],
        )?;
    }

    let output = QueryEngine.execute(
        "MATCH (a)\nWITH DISTINCT a.name2 AS name\nWHERE a.name2 = 'B'\nRETURN *",
        &mut context(&graph),
    )?;

    assert_eq!(
        column_values(&output, "name")?,
        vec![ResultValue::Scalar(ScalarValue::String("B".into()))]
    );
    assert_read_only(&output);
    Ok(())
}

fn optional_relationship_graph() -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    insert_node(&mut graph, 1, &["A"], &[])?;
    insert_node(&mut graph, 2, &["B"], &[("id", ScalarValue::Integer(1))])?;
    insert_node(&mut graph, 3, &["B"], &[("id", ScalarValue::Integer(2))])?;
    insert_edge(&mut graph, 10, 1, 2, "T")?;
    Ok(graph)
}

fn assert_unmatched_b_node(output: &ExecutionOutput) -> Result<()> {
    assert_eq!(column_values(output, "other")?.len(), 1);
    let node = only_node(output, "other")?;
    assert_eq!(node.id, NodeId(3));
    assert_eq!(node.labels, vec!["B"]);
    assert_eq!(
        node.properties,
        BTreeMap::from([("id".to_owned(), ScalarValue::Integer(2))])
    );
    assert_read_only(output);
    Ok(())
}

#[test]
fn tck_with_where1_scenario_3_filters_on_an_unprojected_relationship_binding() -> Result<()> {
    let graph = optional_relationship_graph()?;
    let output = QueryEngine.execute(
        "MATCH (a:A), (other:B)\nOPTIONAL MATCH (a)-[r]->(other)\nWITH other WHERE r IS NULL\nRETURN other",
        &mut context(&graph),
    )?;
    assert_unmatched_b_node(&output)
}

#[test]
fn tck_with_where1_scenario_4_filters_on_an_unprojected_node_binding() -> Result<()> {
    let graph = optional_relationship_graph()?;
    let output = QueryEngine.execute(
        "MATCH (other:B)\nOPTIONAL MATCH (a)-[r]->(other)\nWITH other WHERE a IS NULL\nRETURN other",
        &mut context(&graph),
    )?;
    assert_unmatched_b_node(&output)
}

#[test]
fn with_where_visibility_does_not_leak_an_incoming_binding_past_the_projection() -> Result<()> {
    let graph = GraphStore::default();
    let error = QueryEngine
        .execute(
            "MATCH (a) WITH a.name AS name WHERE a.name = 'B' RETURN a",
            &mut context(&graph),
        )
        .err()
        .ok_or_else(|| Error::internal("WITH leaked its input binding into RETURN"))?;
    assert_eq!(error.code, ErrorCode::QuerySyntax);
    assert!(error.message.contains("UndefinedVariable"), "{error:?}");
    Ok(())
}

#[test]
fn with_where_rejects_a_name_absent_from_both_input_and_projection_scopes() -> Result<()> {
    let graph = GraphStore::default();
    let error = QueryEngine
        .execute(
            "MATCH (a) WITH a.name AS name WHERE missing = 'B' RETURN name",
            &mut context(&graph),
        )
        .err()
        .ok_or_else(|| Error::internal("WITH WHERE accepted an undefined variable"))?;
    assert_eq!(error.code, ErrorCode::QuerySyntax);
    assert!(error.message.contains("UndefinedVariable"), "{error:?}");
    Ok(())
}
