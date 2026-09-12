// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{collections::BTreeMap, time::Instant};

use irongraph::{
    Bookmark, EdgeId, ErrorCode, Layer, NodeId, ProjectId, Result, ScalarValue,
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
        bookmark: Bookmark { term: 1, index: 0 },
        mutation_revision: 1,
        resolved_time_nanos: 0,
        next_node_id: 1,
        next_edge_id: 1,
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

fn row(query: &str) -> Result<BTreeMap<String, ResultValue>> {
    let graph = GraphStore::default();
    let output = QueryEngine.execute(query, &mut context(&graph))?;
    let batch = output
        .result
        .batches
        .first()
        .ok_or_else(|| irongraph::Error::internal("query returned no batch"))?;
    batch
        .columns
        .iter()
        .map(|column| {
            column
                .values
                .first()
                .cloned()
                .map(|value| (column.name.clone(), value))
                .ok_or_else(|| irongraph::Error::internal("query returned an empty column"))
        })
        .collect()
}

#[test]
fn dynamic_index_dispatches_by_container_type() -> Result<()> {
    let values = row(
        "RETURN [10, 20][1] AS list_value, {name: 'Mats', Name: 'Pontus'}['name'] AS lower, {name: 'Mats', Name: 'Pontus'}['Name'] AS upper",
    )?;
    assert_eq!(
        values.get("list_value"),
        Some(&ResultValue::Scalar(ScalarValue::Integer(20)))
    );
    assert_eq!(
        values.get("lower"),
        Some(&ResultValue::Scalar(ScalarValue::String("Mats".into())))
    );
    assert_eq!(
        values.get("upper"),
        Some(&ResultValue::Scalar(ScalarValue::String("Pontus".into())))
    );
    Ok(())
}

#[test]
fn dynamic_index_propagates_null_and_missing_map_keys() -> Result<()> {
    let values = row(
        "RETURN null['x'] AS container, {name: 'Mats'}[null] AS key, {name: 'Mats'}['missing'] AS missing, {Name: 'Mats'}['name'] AS case_mismatch",
    )?;
    for name in ["container", "key", "missing", "case_mismatch"] {
        assert_eq!(
            values.get(name),
            Some(&ResultValue::Scalar(ScalarValue::Null)),
            "column: {name}"
        );
    }
    Ok(())
}

#[test]
fn map_index_rejects_every_non_string_with_the_tck_detail() -> Result<()> {
    let graph = GraphStore::default();
    for query in ["RETURN {name: 'Mats'}[0]", "RETURN {name: 'Mats'}[12.3]"] {
        let error = QueryEngine
            .execute(query, &mut context(&graph))
            .err()
            .ok_or_else(|| irongraph::Error::internal("non-string map index succeeded"))?;
        assert_eq!(error.code, ErrorCode::QueryType, "query: {query}");
        assert!(
            error.message.contains("MapElementAccessByNonString"),
            "query: {query}; error: {}",
            error.message
        );
    }
    Ok(())
}

#[test]
fn list_and_scalar_index_type_errors_remain_distinct() -> Result<()> {
    let graph = GraphStore::default();
    for query in ["RETURN [1, 2]['0']", "RETURN 100[0]"] {
        let error = QueryEngine
            .execute(query, &mut context(&graph))
            .err()
            .ok_or_else(|| irongraph::Error::internal("invalid dynamic index succeeded"))?;
        assert_eq!(error.code, ErrorCode::QueryType, "query: {query}");
        assert!(
            error.message.contains("InvalidArgumentType"),
            "query: {query}; error: {}",
            error.message
        );
    }
    Ok(())
}

#[test]
fn dynamic_entity_index_reads_node_and_relationship_properties() -> Result<()> {
    let mut graph = GraphStore::default();
    let name = graph.catalog_mut().intern_property("name")?;
    let weight = graph.catalog_mut().intern_property("weight")?;
    let relationship_type = graph.catalog_mut().intern_relationship_type("R")?;
    graph.insert_node(NodeInput {
        id: NodeId(1),
        layer: Layer::Observed,
        revision: 1,
        labels: Vec::new(),
        properties: vec![(name, ScalarValue::String("Apa".into()))],
    })?;
    graph.insert_node(NodeInput {
        id: NodeId(2),
        layer: Layer::Observed,
        revision: 1,
        labels: Vec::new(),
        properties: Vec::new(),
    })?;
    graph.insert_edge(EdgeInput {
        id: EdgeId(3),
        source: NodeId(1),
        target: NodeId(2),
        relationship_type,
        layer: Layer::Observed,
        revision: 1,
        properties: vec![(weight, ScalarValue::Integer(7))],
    })?;

    let mut execution = context(&graph);
    execution.parameters.insert(
        "node_key".to_owned(),
        ResultValue::Scalar(ScalarValue::String("name".into())),
    );
    execution.parameters.insert(
        "edge_key".to_owned(),
        ResultValue::Scalar(ScalarValue::String("weight".into())),
    );
    let output = QueryEngine.execute(
        "MATCH (n)-[r]->() RETURN n[$node_key] AS node_value, r[$edge_key] AS edge_value, n['missing'] AS missing",
        &mut execution,
    )?;
    let batch = output
        .result
        .batches
        .first()
        .ok_or_else(|| irongraph::Error::internal("query returned no batch"))?;
    let values = batch
        .columns
        .iter()
        .map(|column| {
            column
                .values
                .first()
                .cloned()
                .map(|value| (column.name.clone(), value))
                .ok_or_else(|| irongraph::Error::internal("query returned an empty column"))
        })
        .collect::<Result<BTreeMap<_, _>>>()?;
    assert_eq!(
        values.get("node_value"),
        Some(&ResultValue::Scalar(ScalarValue::String("Apa".into())))
    );
    assert_eq!(
        values.get("edge_value"),
        Some(&ResultValue::Scalar(ScalarValue::Integer(7)))
    );
    assert_eq!(
        values.get("missing"),
        Some(&ResultValue::Scalar(ScalarValue::Null))
    );
    Ok(())
}

/// One node carrying two same-length property names, used to exercise dynamic property access.
fn graph_with_two_properties() -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    let label = graph.catalog_mut().intern_label("Entity")?;
    let name = graph.catalog_mut().intern_property("name")?;
    let kind = graph.catalog_mut().intern_property("type")?;
    graph.apply(GraphMutation::InsertNode(NodeInput {
        id: NodeId(1),
        layer: Layer::Observed,
        revision: 1,
        labels: vec![label],
        properties: vec![
            (name, ScalarValue::String("Mats".into())),
            (kind, ScalarValue::String("Person".into())),
        ],
    }))?;
    Ok(graph)
}

#[test]
fn a_parameterized_dynamic_property_is_not_answered_from_another_key_s_plan() -> Result<()> {
    // Planning resolves `n[$key]` to a static property access by folding the parameter's text into
    // the plan, and the plan cache keys string parameters by length class so that one plan serves
    // many bindings. `"name"` and `"type"` are both four bytes and so share a key: without a guard
    // on the folded binding the second query reused the plan built for `name` and silently
    // answered with the wrong property instead of failing or re-planning.
    let graph = graph_with_two_properties()?;
    let mut answers = Vec::new();
    for key in ["name", "type"] {
        let mut execution = context(&graph);
        execution.parameters = BTreeMap::from([(
            "key".to_owned(),
            ResultValue::Scalar(ScalarValue::String(key.into())),
        )]);
        let output =
            QueryEngine.execute("MATCH (n:Entity) RETURN n[$key] AS value", &mut execution)?;
        let value = output
            .result
            .batches
            .first()
            .and_then(|batch| batch.columns.first())
            .and_then(|column| column.values.first())
            .cloned()
            .ok_or_else(|| irongraph::Error::internal("query returned no value"))?;
        answers.push(value);
    }
    assert_eq!(
        answers,
        vec![
            ResultValue::Scalar(ScalarValue::String("Mats".into())),
            ResultValue::Scalar(ScalarValue::String("Person".into())),
        ],
        "each binding must be answered by a plan built for that binding"
    );
    Ok(())
}
