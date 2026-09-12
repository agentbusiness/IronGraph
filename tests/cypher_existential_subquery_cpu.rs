// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

use irongraph::{
    Bookmark, EdgeId, Error, ErrorCode, Layer, NodeId, ProjectId, Result, ScalarValue,
    cypher::{
        BindCapabilities, ExecutionContext, ExecutionOutput, QueryEngine, ResultValue, bind, parse,
    },
    graph::{EdgeInput, GraphStore, NodeInput},
};
use tokio_util::sync::CancellationToken;

fn context<'a>(
    graph: &'a GraphStore,
    write: bool,
    cancellation: CancellationToken,
    deadline: Option<Instant>,
) -> ExecutionContext<'a> {
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
            write,
            ..BindCapabilities::default()
        },
        max_result_rows: 4_096,
        max_batch_rows: 4_096,
        optimizer_statistics: None,
        backend: None,
        cancellation,
        deadline,
        resolved_query_at_time_nanos: None,
    }
}

fn ordinary_context(graph: &GraphStore) -> ExecutionContext<'_> {
    context(
        graph,
        false,
        CancellationToken::new(),
        Some(Instant::now() + Duration::from_secs(5)),
    )
}

fn insert_node(graph: &mut GraphStore, id: u64, label: &str, prop: i64) -> Result<()> {
    let label = graph.catalog_mut().intern_label(label)?;
    let property = graph.catalog_mut().intern_property("prop")?;
    graph.insert_node(NodeInput {
        id: NodeId(id),
        layer: Layer::Observed,
        revision: id,
        labels: vec![label],
        properties: vec![(property, ScalarValue::Integer(prop))],
    })?;
    Ok(())
}

fn insert_edge(graph: &mut GraphStore, id: u64, source: u64, target: u64) -> Result<()> {
    let relationship_type = graph.catalog_mut().intern_relationship_type("R")?;
    graph.insert_edge(EdgeInput {
        id: EdgeId(id),
        source: NodeId(source),
        target: NodeId(target),
        relationship_type,
        layer: Layer::Observed,
        revision: id.saturating_add(10),
        properties: Vec::new(),
    })?;
    Ok(())
}

fn fixture(include_b_to_d: bool) -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    for (id, label, prop) in [(1, "A", 1), (2, "B", 1), (3, "C", 2), (4, "D", 3)] {
        insert_node(&mut graph, id, label, prop)?;
    }
    insert_edge(&mut graph, 10, 1, 2)?;
    insert_edge(&mut graph, 11, 1, 3)?;
    insert_edge(&mut graph, 12, 1, 4)?;
    if include_b_to_d {
        insert_edge(&mut graph, 13, 2, 4)?;
    }
    Ok(graph)
}

fn values(output: &ExecutionOutput, name: &str) -> Result<Vec<ResultValue>> {
    let mut values = Vec::new();
    for batch in &output.result.batches {
        let column = batch
            .columns
            .iter()
            .find(|column| column.name == name)
            .ok_or_else(|| Error::internal(format!("missing result column `{name}`")))?;
        values.extend(column.values.iter().cloned());
    }
    Ok(values)
}

fn node_ids(output: &ExecutionOutput, name: &str) -> Result<Vec<NodeId>> {
    values(output, name)?
        .into_iter()
        .map(|value| match value {
            ResultValue::Node(node) => Ok(node.id),
            other => Err(Error::internal(format!(
                "column `{name}` returned a non-node value: {other:?}"
            ))),
        })
        .collect()
}

fn boolean(output: &ExecutionOutput, name: &str) -> Result<bool> {
    match values(output, name)?.as_slice() {
        [ResultValue::Scalar(ScalarValue::Boolean(value))] => Ok(*value),
        other => Err(Error::internal(format!(
            "column `{name}` did not contain one Boolean: {other:?}"
        ))),
    }
}

fn assert_one_a(graph: &GraphStore, source: &str) -> Result<()> {
    let output = QueryEngine.execute(source, &mut ordinary_context(graph))?;
    assert_eq!(node_ids(&output, "n")?, vec![NodeId(1)], "query: {source}");
    assert!(output.graph_mutations.is_empty());
    assert!(output.temporal_mutations.is_empty());
    Ok(())
}

#[test]
fn official_simple_forms_cover_correlation_where_and_empty_witnesses() -> Result<()> {
    let graph = fixture(false)?;
    assert_one_a(&graph, "MATCH (n) WHERE exists { (n)-->() } RETURN n")?;
    assert_one_a(
        &fixture(true)?,
        "MATCH (n) WHERE exists { (n)-->(m) WHERE n.prop = m.prop } RETURN n",
    )?;

    for source in [
        "MATCH (n) WHERE exists { (n)-[:NA]->() } RETURN n",
        "MATCH (n) WHERE exists { (n)-[r]->() WHERE type(r) = 'NA' } RETURN n",
    ] {
        let output = QueryEngine.execute(source, &mut ordinary_context(&graph))?;
        assert!(node_ids(&output, "n")?.is_empty(), "query: {source}");
    }
    Ok(())
}

#[test]
fn official_full_forms_execute_return_and_aggregation_semantics() -> Result<()> {
    assert_one_a(
        &fixture(false)?,
        "MATCH (n) WHERE exists { MATCH (n)-->() RETURN true } RETURN n",
    )?;
    assert_one_a(
        &fixture(true)?,
        "MATCH (n) WHERE exists { MATCH (n)-->(m) WITH n, count(*) AS numConnections WHERE numConnections = 3 RETURN true } RETURN n",
    )?;

    let graph = fixture(false)?;
    let aggregate_over_empty = QueryEngine.execute(
        "RETURN exists { MATCH (:Missing) RETURN count(*) AS ignored } AS value",
        &mut ordinary_context(&graph),
    )?;
    assert!(boolean(&aggregate_over_empty, "value")?);
    let empty = QueryEngine.execute(
        "RETURN exists { MATCH (:Missing) RETURN true } AS value",
        &mut ordinary_context(&graph),
    )?;
    assert!(!boolean(&empty, "value")?);
    Ok(())
}

#[test]
fn official_nested_simple_full_and_pattern_predicate_forms_correlate_each_level() -> Result<()> {
    let graph = fixture(false)?;
    for source in [
        "MATCH (n) WHERE exists { MATCH (m) WHERE exists { (n)-[]->(m) WHERE n.prop = m.prop } RETURN true } RETURN n",
        "MATCH (n) WHERE exists { MATCH (m) WHERE exists { MATCH (l)<-[:R]-(n)-[:R]->(m) RETURN true } RETURN true } RETURN n",
        "MATCH (n) WHERE exists { MATCH (m) WHERE exists { MATCH (l) WHERE (l)<-[:R]-(n)-[:R]->(m) RETURN true } RETURN true } RETURN n",
    ] {
        assert_one_a(&graph, source)?;
    }
    Ok(())
}

#[test]
fn local_subquery_bindings_do_not_leak_and_updates_fail_during_parsing() -> Result<()> {
    let graph = fixture(false)?;
    let error = bind(
        parse("MATCH (n) WHERE exists { MATCH (n)-->(local) RETURN local } RETURN local")?,
        graph.catalog(),
        BindCapabilities::default(),
    )
    .expect_err("subquery-local variable leaked into the outer scope");
    assert_eq!(error.code, ErrorCode::QuerySyntax);
    assert!(error.message.contains("UndefinedVariable"));

    for source in [
        "MATCH (n) WHERE exists { MATCH (n)-->(m) SET m.prop = 1 } RETURN n",
        "MATCH (n) WHERE exists { CREATE (:X) } RETURN n",
        "MATCH (n) WHERE exists { MERGE (:X) } RETURN n",
        "MATCH (n) WHERE exists { MATCH (m) DELETE m } RETURN n",
        "MATCH (n) WHERE exists { MATCH (m) REMOVE m.prop } RETURN n",
    ] {
        let error = parse(source).expect_err("updating existential subquery was accepted");
        assert_eq!(error.code, ErrorCode::QuerySyntax, "query: {source}");
        assert!(
            error.message.contains("InvalidClauseComposition"),
            "query: {source}; error: {error:?}"
        );
    }
    Ok(())
}

#[test]
fn same_statement_writes_are_visible_without_leaking_subquery_columns() -> Result<()> {
    let graph = GraphStore::default();
    let mut execution = context(
        &graph,
        true,
        CancellationToken::new(),
        Some(Instant::now() + Duration::from_secs(5)),
    );
    let output = QueryEngine.execute(
        "CREATE (a:A {prop: 1})-[:R]->(:B {prop: 1}) WITH a WHERE exists { (a)-[:R]->() } RETURN count(*) AS witnesses",
        &mut execution,
    )?;
    assert_eq!(
        values(&output, "witnesses")?,
        vec![ResultValue::Scalar(ScalarValue::Integer(1))]
    );
    assert!(!output.graph_mutations.is_empty());
    Ok(())
}

#[test]
fn nesting_has_no_fixed_depth_ceiling_and_runtime_guards_are_enforced() -> Result<()> {
    let mut nested = "true".to_owned();
    for _ in 0..=64 {
        nested = format!("exists {{ RETURN {nested} AS value }}");
    }
    let deep_query = format!("RETURN {nested} AS value");
    parse(&deep_query)?;

    let empty = GraphStore::default();
    let output = QueryEngine.execute(
        &deep_query,
        &mut context(
            &empty,
            false,
            CancellationToken::new(),
            Some(Instant::now() + Duration::from_secs(5)),
        ),
    )?;
    assert_eq!(
        values(&output, "value")?,
        vec![ResultValue::Scalar(ScalarValue::Boolean(true))]
    );

    let graph = fixture(false)?;
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    let error = QueryEngine
        .execute(
            "MATCH (n) WHERE exists { MATCH (n)-->() RETURN true } RETURN n",
            &mut context(
                &graph,
                false,
                cancellation,
                Some(Instant::now() + Duration::from_secs(5)),
            ),
        )
        .expect_err("cancelled nested execution completed");
    assert_eq!(error.code, ErrorCode::Cancelled);

    let error = QueryEngine
        .execute(
            "MATCH (n) WHERE exists { MATCH (n)-->() RETURN true } RETURN n",
            &mut context(
                &graph,
                false,
                CancellationToken::new(),
                Some(Instant::now() - Duration::from_millis(1)),
            ),
        )
        .expect_err("expired nested execution completed");
    assert_eq!(error.code, ErrorCode::DeadlineExceeded);
    Ok(())
}
