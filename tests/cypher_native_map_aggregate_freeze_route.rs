// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Focused strict-native proofs for the final Return4/Return6 map-aggregation freeze.

use std::{
    collections::BTreeMap,
    sync::Arc,
    time::{Duration, Instant},
};

use irongraph::{
    Bookmark, ProjectId, Result, ScalarValue,
    cypher::{
        BindCapabilities, ColumnType, ExecutionContext, ExecutionOutput, QueryEngine, ResultValue,
    },
    gpu::{CpuBackend, ExecutionBackend, ResidentProjectImage},
    graph::GraphStore,
};
use tokio_util::sync::CancellationToken;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());
const MEMORY_LIMIT_BYTES: usize = 512 * 1024 * 1024;
const RESERVED_BYTES: usize = 64 * 1024 * 1024;

fn context<'a>(
    graph: &'a GraphStore,
    backend: Option<&'a dyn ExecutionBackend>,
    require_native_execution: bool,
) -> ExecutionContext<'a> {
    ExecutionContext {
        project_id: PROJECT,
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
            term: 0,
            index: graph.revision(),
        },
        mutation_revision: graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id: 1,
        next_edge_id: 1,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            write: true,
            require_native_execution,
            ..BindCapabilities::default()
        },
        max_result_rows: 128,
        max_batch_rows: 128,
        optimizer_statistics: None,
        backend,
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(30)),
        resolved_query_at_time_nanos: None,
    }
}

fn fixture_graph(setup_queries: &[&str]) -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    for setup in setup_queries {
        let output = QueryEngine.execute(setup, &mut context(&graph, None, false))?;
        assert!(output.temporal_mutations.is_empty());
        for mutation in output.graph_mutations {
            graph.apply(mutation)?;
        }
    }
    Ok(graph)
}

fn execute_strict_cpu(graph: &GraphStore, query: &str) -> Result<ExecutionOutput> {
    let mut backend = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
    backend.replace_all_projects(vec![ResidentProjectImage::graph_only(Arc::new(
        graph.snapshot()?,
    ))])?;
    QueryEngine.execute(query, &mut context(graph, Some(&backend), true))
}

fn rows(output: &ExecutionOutput) -> Vec<Vec<ResultValue>> {
    let mut rows = Vec::new();
    for batch in &output.result.batches {
        assert!(batch.validate());
        for row in 0..batch.row_count {
            rows.push(
                batch
                    .columns
                    .iter()
                    .map(|column| column.values[row].clone())
                    .collect(),
            );
        }
    }
    rows
}

fn integer(value: i64) -> ResultValue {
    ResultValue::Scalar(ScalarValue::Integer(value))
}

fn string(value: &str) -> ResultValue {
    ResultValue::Scalar(ScalarValue::String(value.into()))
}

#[test]
fn strict_cpu_return4_9_publishes_coalesce_and_canonical_map() -> Result<()> {
    let graph = fixture_graph(&["CREATE (:A), (:B {num: 42})"])?;
    let output = execute_strict_cpu(
        &graph,
        "MATCH (a:A), (b:B) \
         RETURN coalesce(a.num, b.num) AS foo, b.num AS bar, {name: count(b)} AS baz",
    )?;

    assert_eq!(
        output.result.schema,
        [
            ("foo".to_owned(), ColumnType::Integer),
            ("bar".to_owned(), ColumnType::Integer),
            ("baz".to_owned(), ColumnType::Map),
        ]
    );
    assert_eq!(
        rows(&output),
        [vec![
            integer(42),
            integer(42),
            ResultValue::Map(BTreeMap::from([("name".to_owned(), integer(1))])),
        ]]
    );
    assert!(!output.result.truncated);
    assert!(output.graph_mutations.is_empty());
    Ok(())
}

#[test]
fn strict_cpu_nonlexical_map_source_is_lowered_to_canonical_key_order() -> Result<()> {
    let graph = fixture_graph(&["CREATE (:B {num: 42})"])?;
    let output = execute_strict_cpu(
        &graph,
        "MATCH (b:B) RETURN b.num AS num, {z: count(b), a: b.num} AS value",
    )?;

    assert_eq!(
        output.result.schema,
        [
            ("num".to_owned(), ColumnType::Integer),
            ("value".to_owned(), ColumnType::Map),
        ]
    );
    assert_eq!(
        rows(&output),
        [vec![
            integer(42),
            ResultValue::Map(BTreeMap::from([
                ("a".to_owned(), integer(42)),
                ("z".to_owned(), integer(1)),
            ])),
        ]]
    );
    assert!(!output.result.truncated);
    assert!(output.graph_mutations.is_empty());
    Ok(())
}

#[test]
fn strict_cpu_with4_7_keeps_map_alias_backreferences_after_create() -> Result<()> {
    let output = execute_strict_cpu(
        &GraphStore::default(),
        "CREATE (m {id: 0}) WITH {first: m.id} AS m \
         WITH {second: m.first} AS m RETURN m.second",
    )?;
    assert_eq!(
        output.result.schema,
        [("m.second".to_owned(), ColumnType::Integer)]
    );
    assert_eq!(rows(&output), [vec![integer(0)]]);
    assert!(!output.graph_mutations.is_empty());
    Ok(())
}

#[test]
fn strict_cpu_return4_11_orders_before_collecting_head() -> Result<()> {
    let official = fixture_graph(&["CREATE (a:Person), (b:Person), (m:Message {id: 10}) \
         CREATE (a)-[:LIKE {creationDate: 20160614}]->(m)-[:POSTED_BY]->(b)"])?;
    let query = "MATCH (person:Person)<--(message)<-[like]-(:Person) \
                 WITH like.creationDate AS likeTime, person AS person \
                 ORDER BY likeTime, message.id \
                 WITH head(collect({likeTime: likeTime})) AS latestLike, person AS person \
                 RETURN latestLike.likeTime AS likeTime ORDER BY likeTime";
    let output = execute_strict_cpu(&official, query)?;
    assert_eq!(
        output.result.schema,
        [("likeTime".to_owned(), ColumnType::Integer)]
    );
    assert_eq!(rows(&output), [vec![integer(20160614)]]);

    let official_with4 = "MATCH (person:Person)<--(message)<-[like]-(:Person) \
                 WITH like.creationDate AS likeTime, person AS person \
                 ORDER BY likeTime, message.id \
                 WITH head(collect({likeTime: likeTime})) AS latestLike, person AS person \
                 WITH latestLike.likeTime AS likeTime ORDER BY likeTime RETURN likeTime";
    let output = execute_strict_cpu(&official, official_with4)?;
    assert_eq!(rows(&output), [vec![integer(20160614)]]);

    // The later LIKE is created first, so source order disagrees with the requested ascending
    // order. This twin makes the pre-aggregate ORDER observable at `head(collect(...))`.
    let order_sensitive = fixture_graph(&["CREATE (person:Person), (liker:Person), \
                (late:Message {id: 20}), (early:Message {id: 10}) \
         CREATE (liker)-[:LIKE {creationDate: 20160620}]->(late)-[:POSTED_BY]->(person), \
                (liker)-[:LIKE {creationDate: 20160610}]->(early)-[:POSTED_BY]->(person)"])?;
    let output = execute_strict_cpu(&order_sensitive, query)?;
    assert_eq!(rows(&output), [vec![integer(20160610)]]);
    assert!(!output.result.truncated);
    assert!(output.graph_mutations.is_empty());
    Ok(())
}

#[test]
fn strict_cpu_return6_6_empty_grouped_map_returns_zero_rows() -> Result<()> {
    let output = execute_strict_cpu(
        &GraphStore::default(),
        "MATCH (a {name: 'Andres'})<-[:FATHER]-(child) \
         RETURN a.name, {foo: a.name = 'Andres', kids: collect(child.name)}",
    )?;

    assert_eq!(output.result.schema.len(), 2);
    assert!(rows(&output).is_empty());
    assert!(!output.result.truncated);
    assert!(output.graph_mutations.is_empty());
    Ok(())
}

#[test]
fn strict_cpu_return6_6_nonempty_grouped_map_publishes_boolean_and_list() -> Result<()> {
    let graph = fixture_graph(&["CREATE (a {name: 'Andres'}), (child {name: 'Emil'}) \
         CREATE (child)-[:FATHER]->(a)"])?;
    let output = execute_strict_cpu(
        &graph,
        "MATCH (a {name: 'Andres'})<-[:FATHER]-(child) \
         RETURN a.name, {foo: a.name = 'Andres', kids: collect(child.name)}",
    )?;

    assert_eq!(
        rows(&output),
        [vec![
            string("Andres"),
            ResultValue::Map(BTreeMap::from([
                (
                    "foo".to_owned(),
                    ResultValue::Scalar(ScalarValue::Boolean(true)),
                ),
                ("kids".to_owned(), ResultValue::List(vec![string("Emil")]),),
            ])),
        ]]
    );
    assert!(!output.result.truncated);
    assert!(output.graph_mutations.is_empty());
    Ok(())
}

#[test]
fn strict_cpu_unwind1_5_keeps_collected_entities_on_nullable_property_route() -> Result<()> {
    let graph = fixture_graph(&["CREATE ({id: 1}), ({id: 2})"])?;
    let output = execute_strict_cpu(
        &graph,
        "MATCH (row) WITH collect(row) AS rows \
         UNWIND rows AS node RETURN node.id",
    )?;

    assert_eq!(
        output.result.schema,
        [("node.id".to_owned(), ColumnType::Integer)]
    );
    assert_eq!(rows(&output), [vec![integer(1)], vec![integer(2)]]);
    assert!(!output.result.truncated);
    assert!(output.graph_mutations.is_empty());
    Ok(())
}
