// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{collections::BTreeMap, time::Instant};

use irongraph::{
    Bookmark, EdgeId, Error, Layer, NodeId, ProjectId, Result, ScalarValue,
    cypher::{BindCapabilities, ExecutionContext, ExecutionOutput, QueryEngine, ResultValue},
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
        bookmark: Bookmark {
            term: 1,
            index: graph.revision(),
        },
        mutation_revision: graph.revision().saturating_add(1),
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

fn fixture() -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    let name = graph.catalog_mut().intern_property("name")?;
    for (id, label, value) in [(1, "A", "A"), (2, "B", "B"), (3, "C", "C")] {
        let label = graph.catalog_mut().intern_label(label)?;
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: id,
            labels: vec![label],
            properties: vec![(name, ScalarValue::String(value.into()))],
        })?;
    }
    let relationship_type = graph.catalog_mut().intern_relationship_type("R")?;
    for (id, source, target) in [(10, 1, 2), (20, 2, 3)] {
        graph.insert_edge(EdgeInput {
            id: EdgeId(id),
            source: NodeId(source),
            target: NodeId(target),
            relationship_type,
            layer: Layer::Observed,
            revision: id,
            properties: Vec::new(),
        })?;
    }
    Ok(graph)
}

fn parallel_fixture() -> Result<GraphStore> {
    let mut graph = fixture()?;
    let relationship_type = graph.catalog_mut().intern_relationship_type("R")?;
    graph.insert_edge(EdgeInput {
        id: EdgeId(11),
        source: NodeId(1),
        target: NodeId(2),
        relationship_type,
        layer: Layer::Observed,
        revision: 21,
        properties: Vec::new(),
    })?;
    Ok(graph)
}

fn execute(graph: &GraphStore, query: &str) -> Result<ExecutionOutput> {
    QueryEngine.execute(query, &mut context(graph))
}

fn column(output: &ExecutionOutput, name: &str) -> Result<Vec<ResultValue>> {
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

fn names(output: &ExecutionOutput) -> Result<Vec<String>> {
    column(output, "name")?
        .into_iter()
        .map(|value| match value {
            ResultValue::Scalar(ScalarValue::String(value)) => Ok(value.to_string()),
            other => Err(Error::internal(format!(
                "expected a string name, got {other:?}"
            ))),
        })
        .collect()
}

fn relationship_ids(output: &ExecutionOutput, name: &str) -> Result<Vec<EdgeId>> {
    column(output, name)?
        .into_iter()
        .map(|value| match value {
            ResultValue::Relationship(relationship) => Ok(relationship.id),
            other => Err(Error::internal(format!(
                "expected relationship `{name}`, got {other:?}"
            ))),
        })
        .collect()
}

#[test]
fn connected_and_comma_paths_cannot_retrace_one_relationship() -> Result<()> {
    let graph = fixture()?;
    for query in [
        "MATCH (a:A)-[r1:R]-(b:B)-[r2:R]-(c) \
         RETURN c.name AS name, r1, r2 ORDER BY name",
        "MATCH (a:A)-[r1:R]-(b:B), (b)-[r2:R]-(c) \
         RETURN c.name AS name, r1, r2 ORDER BY name",
    ] {
        let output = execute(&graph, query)?;
        assert_eq!(names(&output)?, vec!["C"]);
        assert_eq!(relationship_ids(&output, "r1")?, vec![EdgeId(10)]);
        assert_eq!(relationship_ids(&output, "r2")?, vec![EdgeId(20)]);
    }
    Ok(())
}

#[test]
fn uniqueness_compares_dense_edge_identity_not_endpoints_or_type() -> Result<()> {
    let graph = parallel_fixture()?;
    let output = execute(
        &graph,
        "MATCH (a:A)-[r1:R]-(b:B), (b)-[r2:R]-(c) \
         RETURN c.name AS name, r1, r2",
    )?;
    let names = names(&output)?;
    let first = relationship_ids(&output, "r1")?;
    let second = relationship_ids(&output, "r2")?;
    let mut rows = names
        .into_iter()
        .zip(first)
        .zip(second)
        .map(|((name, first), second)| (name, first, second))
        .collect::<Vec<_>>();
    rows.sort_unstable();
    assert_eq!(
        rows,
        vec![
            ("A".to_owned(), EdgeId(10), EdgeId(11)),
            ("A".to_owned(), EdgeId(11), EdgeId(10)),
            ("C".to_owned(), EdgeId(10), EdgeId(20)),
            ("C".to_owned(), EdgeId(11), EdgeId(20)),
        ]
    );
    Ok(())
}

#[test]
fn later_match_and_optional_match_may_reuse_an_earlier_relationship() -> Result<()> {
    let graph = fixture()?;
    let separate = execute(
        &graph,
        "MATCH (a:A)-[r1:R]-(b:B) \
         MATCH (b)-[r2:R]-(c) \
         RETURN c.name AS name, r1, r2 ORDER BY name",
    )?;
    assert_eq!(names(&separate)?, vec!["A", "C"]);
    assert_eq!(
        relationship_ids(&separate, "r1")?,
        vec![EdgeId(10), EdgeId(10)]
    );
    assert_eq!(
        relationship_ids(&separate, "r2")?,
        vec![EdgeId(10), EdgeId(20)]
    );

    let optional = execute(
        &graph,
        "MATCH (a:A)-[seed:R]-(b:B) \
         OPTIONAL MATCH (b)-[again:R]-(a) \
         RETURN seed, again",
    )?;
    assert_eq!(relationship_ids(&optional, "seed")?, vec![EdgeId(10)]);
    assert_eq!(relationship_ids(&optional, "again")?, vec![EdgeId(10)]);
    Ok(())
}

#[test]
fn anonymous_relationships_obey_the_same_clause_boundary() -> Result<()> {
    let graph = fixture()?;
    let comma = execute(
        &graph,
        "MATCH (a:A)-[]-(b:B), (b)-[]-(c) \
         RETURN c.name AS name ORDER BY name",
    )?;
    assert_eq!(names(&comma)?, vec!["C"]);

    let separate = execute(
        &graph,
        "MATCH (a:A)-[]-(b:B) \
         MATCH (b)-[]-(c) \
         RETURN c.name AS name ORDER BY name",
    )?;
    assert_eq!(names(&separate)?, vec!["A", "C"]);
    Ok(())
}

#[test]
fn failed_multi_pattern_optional_match_null_extends_the_whole_clause() -> Result<()> {
    let graph = fixture()?;
    let output = execute(
        &graph,
        "MATCH (a:A)-[seed:R]-(b:B) \
         OPTIONAL MATCH (b)-[r1:R]-(a), (b)-[r2:R]-(missing:Missing) \
         RETURN r1, r2",
    )?;
    let null = ResultValue::Scalar(ScalarValue::Null);
    assert_eq!(column(&output, "r1")?, vec![null.clone()]);
    assert_eq!(column(&output, "r2")?, vec![null]);
    Ok(())
}
