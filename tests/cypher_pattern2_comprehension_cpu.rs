// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{collections::BTreeMap, time::Instant};

use irongraph::{
    Bookmark, EdgeId, Error, Layer, NodeId, ProjectId, Result, ScalarValue,
    cypher::{
        BindCapabilities, ExecutionContext, ExecutionOutput, QueryEngine, ResultValue,
        StatementStats,
    },
    graph::{EdgeInput, GraphStore, NodeInput},
};
use tokio_util::sync::CancellationToken;

#[derive(Clone, Debug, PartialEq)]
struct ExactNode {
    id: NodeId,
    layer: Layer,
    revision: u64,
    labels: Vec<String>,
    properties: BTreeMap<String, ScalarValue>,
}

#[derive(Clone, Debug, PartialEq)]
struct ExactRelationship {
    id: EdgeId,
    source: NodeId,
    target: NodeId,
    relationship_type: String,
    layer: Layer,
    revision: u64,
    properties: BTreeMap<String, ScalarValue>,
}

#[derive(Clone, Debug, PartialEq)]
enum ExactValue {
    Scalar(ScalarValue),
    Node(ExactNode),
    Relationship(ExactRelationship),
    Path {
        nodes: Vec<ExactNode>,
        relationships: Vec<ExactRelationship>,
    },
    Vector(Vec<f32>),
    List(Vec<ExactValue>),
    Map(BTreeMap<String, ExactValue>),
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
        bookmark: Bookmark {
            term: 1,
            index: 100,
        },
        mutation_revision: 101,
        resolved_time_nanos: 0,
        next_node_id: 1_000,
        next_edge_id: 1_000,
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

fn properties(entries: &[(&str, ScalarValue)]) -> BTreeMap<String, ScalarValue> {
    entries
        .iter()
        .map(|(name, value)| ((*name).to_owned(), value.clone()))
        .collect()
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
        .map(|(name, value)| Ok((graph.catalog_mut().intern_property(name)?, value.clone())))
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

fn insert_relationship(
    graph: &mut GraphStore,
    id: u64,
    source: u64,
    target: u64,
    relationship_type: &str,
    properties: &[(&str, ScalarValue)],
) -> Result<()> {
    let relationship_type = graph
        .catalog_mut()
        .intern_relationship_type(relationship_type)?;
    let properties = properties
        .iter()
        .map(|(name, value)| Ok((graph.catalog_mut().intern_property(name)?, value.clone())))
        .collect::<Result<Vec<_>>>()?;
    graph
        .insert_edge(EdgeInput {
            id: EdgeId(id),
            source: NodeId(source),
            target: NodeId(target),
            relationship_type,
            layer: Layer::Observed,
            revision: id,
            properties,
        })
        .map(|_| ())
}

fn node(id: u64, labels: &[&str], properties: &[(&str, ScalarValue)]) -> ExactNode {
    let mut labels = labels
        .iter()
        .map(|label| (*label).to_owned())
        .collect::<Vec<_>>();
    labels.sort();
    ExactNode {
        id: NodeId(id),
        layer: Layer::Observed,
        revision: id,
        labels,
        properties: self::properties(properties),
    }
}

fn relationship(
    id: u64,
    source: u64,
    target: u64,
    relationship_type: &str,
    properties: &[(&str, ScalarValue)],
) -> ExactRelationship {
    ExactRelationship {
        id: EdgeId(id),
        source: NodeId(source),
        target: NodeId(target),
        relationship_type: relationship_type.to_owned(),
        layer: Layer::Observed,
        revision: id,
        properties: self::properties(properties),
    }
}

fn path(nodes: Vec<ExactNode>, relationships: Vec<ExactRelationship>) -> ExactValue {
    ExactValue::Path {
        nodes,
        relationships,
    }
}

fn list(values: Vec<ExactValue>) -> ExactValue {
    ExactValue::List(values)
}

fn integer(value: i64) -> ExactValue {
    ExactValue::Scalar(ScalarValue::Integer(value))
}

fn string(value: &str) -> ExactValue {
    ExactValue::Scalar(ScalarValue::String(value.into()))
}

fn null() -> ExactValue {
    ExactValue::Scalar(ScalarValue::Null)
}

fn exact_value(value: &ResultValue) -> ExactValue {
    match value {
        ResultValue::Scalar(value) => ExactValue::Scalar(value.clone()),
        ResultValue::Node(node) => {
            let mut labels = node.labels.clone();
            labels.sort();
            ExactValue::Node(ExactNode {
                id: node.id,
                layer: node.layer,
                revision: node.revision,
                labels,
                properties: node.properties.clone(),
            })
        }
        ResultValue::Relationship(relationship) => ExactValue::Relationship(ExactRelationship {
            id: relationship.id,
            source: relationship.source,
            target: relationship.target,
            relationship_type: relationship.relationship_type.clone(),
            layer: relationship.layer,
            revision: relationship.revision,
            properties: relationship.properties.clone(),
        }),
        ResultValue::Path {
            nodes,
            relationships,
        } => ExactValue::Path {
            nodes: nodes
                .iter()
                .map(|node| {
                    let mut labels = node.labels.clone();
                    labels.sort();
                    ExactNode {
                        id: node.id,
                        layer: node.layer,
                        revision: node.revision,
                        labels,
                        properties: node.properties.clone(),
                    }
                })
                .collect(),
            relationships: relationships
                .iter()
                .map(|relationship| ExactRelationship {
                    id: relationship.id,
                    source: relationship.source,
                    target: relationship.target,
                    relationship_type: relationship.relationship_type.clone(),
                    layer: relationship.layer,
                    revision: relationship.revision,
                    properties: relationship.properties.clone(),
                })
                .collect(),
        },
        ResultValue::Vector(values) => ExactValue::Vector(values.clone()),
        ResultValue::List(values) => ExactValue::List(values.iter().map(exact_value).collect()),
        ResultValue::Map(values) => ExactValue::Map(
            values
                .iter()
                .map(|(name, value)| (name.clone(), exact_value(value)))
                .collect(),
        ),
    }
}

fn result_rows(output: &ExecutionOutput, columns: &[&str]) -> Result<Vec<Vec<ExactValue>>> {
    let mut rows = Vec::new();
    for batch in &output.result.batches {
        let selected = columns
            .iter()
            .map(|name| {
                batch
                    .columns
                    .iter()
                    .find(|column| column.name == *name)
                    .ok_or_else(|| Error::internal(format!("result omitted column `{name}`")))
            })
            .collect::<Result<Vec<_>>>()?;
        for row in 0..batch.row_count {
            rows.push(
                selected
                    .iter()
                    .map(|column| exact_value(&column.values[row]))
                    .collect(),
            );
        }
    }
    Ok(rows)
}

fn sorted_rows(mut rows: Vec<Vec<ExactValue>>) -> Vec<Vec<ExactValue>> {
    rows.sort_by_key(|row| format!("{row:?}"));
    rows
}

fn assert_query(
    graph: &GraphStore,
    query: &str,
    columns: &[&str],
    expected: Vec<Vec<ExactValue>>,
    ordered: bool,
) -> Result<()> {
    let output = QueryEngine.execute(query, &mut context(graph))?;
    assert_eq!(
        output
            .result
            .schema
            .iter()
            .map(|(name, _)| name.as_str())
            .collect::<Vec<_>>(),
        columns,
        "query returned the wrong columns: {query}"
    );
    assert!(
        output.graph_mutations.is_empty(),
        "read-only Pattern2 query produced graph mutations: {query}"
    );
    assert_eq!(
        output.result.statistics,
        StatementStats::default(),
        "read-only Pattern2 query reported side effects: {query}"
    );
    assert!(!output.result.truncated, "Pattern2 result was truncated");

    let actual = result_rows(&output, columns)?;
    if ordered {
        assert_eq!(actual, expected, "query: {query}");
    } else {
        assert_eq!(sorted_rows(actual), sorted_rows(expected), "query: {query}");
    }
    Ok(())
}

#[test]
fn pattern2_01_returns_one_outgoing_path_list_per_node() -> Result<()> {
    let mut graph = GraphStore::default();
    insert_node(&mut graph, 1, &["A"], &[])?;
    insert_node(&mut graph, 2, &["B"], &[])?;
    insert_node(&mut graph, 3, &["C"], &[])?;
    insert_relationship(&mut graph, 10, 1, 2, "T", &[])?;
    insert_relationship(&mut graph, 11, 2, 3, "T", &[])?;

    let a = node(1, &["A"], &[]);
    let b = node(2, &["B"], &[]);
    let c = node(3, &["C"], &[]);
    assert_query(
        &graph,
        "MATCH (n) RETURN [p = (n)-->() | p] AS list",
        &["list"],
        vec![
            vec![list(vec![path(
                vec![a.clone(), b.clone()],
                vec![relationship(10, 1, 2, "T", &[])],
            )])],
            vec![list(vec![path(
                vec![b, c],
                vec![relationship(11, 2, 3, "T", &[])],
            )])],
            vec![list(vec![])],
        ],
        false,
    )
}

#[test]
fn pattern2_02_applies_the_end_node_label_predicate() -> Result<()> {
    let mut graph = GraphStore::default();
    for (id, label) in [(1, "A"), (2, "B"), (3, "C"), (4, "D")] {
        insert_node(&mut graph, id, &[label], &[])?;
    }
    insert_relationship(&mut graph, 10, 1, 2, "T", &[])?;
    insert_relationship(&mut graph, 11, 1, 3, "T", &[])?;
    insert_relationship(&mut graph, 12, 1, 4, "T", &[])?;

    assert_query(
        &graph,
        "MATCH (n:A) RETURN [p = (n)-->(:B) | p] AS list",
        &["list"],
        vec![vec![list(vec![path(
            vec![node(1, &["A"], &[]), node(2, &["B"], &[])],
            vec![relationship(10, 1, 2, "T", &[])],
        )])]],
        false,
    )
}

#[test]
fn pattern2_03_reuses_both_bound_endpoint_nodes() -> Result<()> {
    let mut graph = GraphStore::default();
    insert_node(&mut graph, 1, &["A"], &[])?;
    insert_node(&mut graph, 2, &["B"], &[])?;
    insert_relationship(&mut graph, 10, 1, 2, "T", &[])?;

    assert_query(
        &graph,
        "MATCH (a:A), (b:B) RETURN [p = (a)-->(b) | p] AS list",
        &["list"],
        vec![vec![list(vec![path(
            vec![node(1, &["A"], &[]), node(2, &["B"], &[])],
            vec![relationship(10, 1, 2, "T", &[])],
        )])]],
        false,
    )
}

#[test]
fn pattern2_04_projects_a_local_node_variable() -> Result<()> {
    let mut graph = GraphStore::default();
    insert_node(&mut graph, 1, &[], &[])?;
    insert_node(
        &mut graph,
        2,
        &[],
        &[("name", ScalarValue::String("val".into()))],
    )?;
    insert_node(&mut graph, 3, &[], &[])?;
    insert_relationship(&mut graph, 10, 1, 2, "T", &[])?;
    insert_relationship(&mut graph, 11, 2, 3, "T", &[])?;

    assert_query(
        &graph,
        "MATCH (n) RETURN [(n)-[:T]->(b) | b.name] AS list",
        &["list"],
        vec![
            vec![list(vec![string("val")])],
            vec![list(vec![null()])],
            vec![list(vec![])],
        ],
        false,
    )
}

#[test]
fn pattern2_05_projects_a_local_relationship_variable() -> Result<()> {
    let mut graph = GraphStore::default();
    for id in 1..=3 {
        insert_node(&mut graph, id, &[], &[])?;
    }
    insert_relationship(
        &mut graph,
        10,
        1,
        2,
        "T",
        &[("name", ScalarValue::String("val".into()))],
    )?;
    insert_relationship(&mut graph, 11, 2, 3, "T", &[])?;

    assert_query(
        &graph,
        "MATCH (n) RETURN [(n)-[r:T]->() | r.name] AS list",
        &["list"],
        vec![
            vec![list(vec![string("val")])],
            vec![list(vec![null()])],
            vec![list(vec![])],
        ],
        false,
    )
}

#[test]
fn pattern2_06_counts_non_null_pattern_comprehension_lists() -> Result<()> {
    let mut graph = GraphStore::default();
    for id in 1..=3 {
        insert_node(&mut graph, id, &["A"], &[])?;
    }
    insert_node(&mut graph, 4, &[], &[])?;
    insert_relationship(&mut graph, 10, 1, 4, "HAS", &[])?;

    assert_query(
        &graph,
        "MATCH (n:A) RETURN count([p = (n)-[:HAS]->() | p]) AS c",
        &["c"],
        vec![vec![integer(3)]],
        false,
    )
}

#[test]
fn pattern2_07_nests_pattern_comprehension_inside_list_comprehension() -> Result<()> {
    let mut graph = GraphStore::default();
    insert_node(&mut graph, 1, &["X"], &[("n", ScalarValue::Integer(1))])?;
    insert_node(&mut graph, 2, &["Y"], &[])?;
    insert_node(&mut graph, 3, &["Y"], &[])?;
    insert_node(&mut graph, 4, &["Y"], &[])?;
    insert_node(&mut graph, 5, &["X"], &[("n", ScalarValue::Integer(2))])?;
    insert_node(&mut graph, 6, &[], &[])?;
    insert_node(&mut graph, 7, &["L"], &[])?;
    insert_node(&mut graph, 8, &["Y"], &[])?;
    insert_relationship(&mut graph, 10, 1, 2, "T", &[])?;
    insert_relationship(&mut graph, 11, 2, 3, "T", &[])?;
    insert_relationship(&mut graph, 12, 2, 4, "T", &[])?;
    insert_relationship(&mut graph, 13, 5, 6, "T", &[])?;
    insert_relationship(&mut graph, 14, 6, 7, "T", &[])?;
    insert_relationship(&mut graph, 15, 6, 8, "T", &[])?;

    assert_query(
        &graph,
        "MATCH p = (n:X)-->() \
         RETURN n, [x IN nodes(p) | size([(x)-->(:Y) | 1])] AS list",
        &["n", "list"],
        vec![
            vec![
                ExactValue::Node(node(1, &["X"], &[("n", ScalarValue::Integer(1))])),
                list(vec![integer(1), integer(2)]),
            ],
            vec![
                ExactValue::Node(node(5, &["X"], &[("n", ScalarValue::Integer(2))])),
                list(vec![integer(0), integer(1)]),
            ],
        ],
        false,
    )
}

#[test]
fn pattern2_08_survives_with_grouping_and_path_projection() -> Result<()> {
    let mut graph = GraphStore::default();
    insert_node(&mut graph, 1, &["A"], &[])?;
    insert_node(&mut graph, 2, &["B"], &[])?;
    insert_node(&mut graph, 3, &["C"], &[])?;
    insert_relationship(&mut graph, 10, 1, 2, "T", &[])?;
    insert_relationship(&mut graph, 11, 2, 3, "T", &[])?;

    let a = node(1, &["A"], &[]);
    let b = node(2, &["B"], &[]);
    let c = node(3, &["C"], &[]);
    assert_query(
        &graph,
        "MATCH (n)-->(b) \
         WITH [p = (n)-->() | p] AS ps, count(b) AS c \
         RETURN ps, c",
        &["ps", "c"],
        vec![
            vec![
                list(vec![path(
                    vec![a.clone(), b.clone()],
                    vec![relationship(10, 1, 2, "T", &[])],
                )]),
                integer(1),
            ],
            vec![
                list(vec![path(
                    vec![b, c],
                    vec![relationship(11, 2, 3, "T", &[])],
                )]),
                integer(1),
            ],
        ],
        false,
    )
}

#[test]
fn pattern2_09_materializes_variable_length_paths_in_with() -> Result<()> {
    let mut graph = GraphStore::default();
    insert_node(&mut graph, 1, &["A"], &[])?;
    insert_node(&mut graph, 2, &["B"], &[])?;
    insert_relationship(&mut graph, 10, 1, 2, "T", &[])?;

    assert_query(
        &graph,
        "MATCH (a:A), (b:B) \
         WITH [p = (a)-[*]->(b) | p] AS paths, count(a) AS c \
         RETURN paths, c",
        &["paths", "c"],
        vec![vec![
            list(vec![path(
                vec![node(1, &["A"], &[]), node(2, &["B"], &[])],
                vec![relationship(10, 1, 2, "T", &[])],
            )]),
            integer(1),
        ]],
        false,
    )
}

#[test]
fn pattern2_10_returns_empty_lists_for_nodes_without_matches() -> Result<()> {
    let mut graph = GraphStore::default();
    for id in 1..=3 {
        insert_node(&mut graph, id, &["A"], &[])?;
    }
    insert_node(&mut graph, 4, &[], &[])?;
    insert_relationship(&mut graph, 10, 1, 4, "HAS", &[])?;

    assert_query(
        &graph,
        "MATCH (n:A) RETURN [p = (n)-[:HAS]->() | p] AS ps",
        &["ps"],
        vec![
            vec![list(vec![path(
                vec![node(1, &["A"], &[]), node(4, &[], &[])],
                vec![relationship(10, 1, 4, "HAS", &[])],
            )])],
            vec![list(vec![])],
            vec![list(vec![])],
        ],
        false,
    )
}

#[test]
fn pattern2_11_orders_undirected_path_lists_by_hidden_input_property() -> Result<()> {
    let mut graph = GraphStore::default();
    insert_node(&mut graph, 1, &[], &[("time", ScalarValue::Integer(10))])?;
    insert_node(&mut graph, 2, &[], &[("time", ScalarValue::Integer(20))])?;
    insert_relationship(&mut graph, 10, 1, 2, "T", &[])?;

    let early = node(1, &[], &[("time", ScalarValue::Integer(10))]);
    let late = node(2, &[], &[("time", ScalarValue::Integer(20))]);
    let connection = relationship(10, 1, 2, "T", &[]);
    assert_query(
        &graph,
        "MATCH (liker) \
         RETURN [p = (liker)--() | p] AS isNew \
         ORDER BY liker.time",
        &["isNew"],
        vec![
            vec![list(vec![path(
                vec![early.clone(), late.clone()],
                vec![connection.clone()],
            )])],
            vec![list(vec![path(vec![late, early], vec![connection])])],
        ],
        true,
    )
}

#[test]
fn pattern2_filter_uses_local_bindings_and_preserves_empty_lists() -> Result<()> {
    let mut graph = GraphStore::default();
    insert_node(&mut graph, 1, &["A"], &[])?;
    insert_node(
        &mut graph,
        2,
        &["B"],
        &[("name", ScalarValue::String("kept".into()))],
    )?;
    insert_node(
        &mut graph,
        3,
        &["B"],
        &[("name", ScalarValue::String("rejected".into()))],
    )?;
    insert_node(&mut graph, 4, &["A"], &[])?;
    insert_node(
        &mut graph,
        5,
        &["B"],
        &[("name", ScalarValue::String("rejected".into()))],
    )?;
    insert_relationship(
        &mut graph,
        10,
        1,
        2,
        "T",
        &[("keep", ScalarValue::Boolean(true))],
    )?;
    insert_relationship(
        &mut graph,
        11,
        1,
        3,
        "T",
        &[("keep", ScalarValue::Boolean(false))],
    )?;
    insert_relationship(
        &mut graph,
        12,
        4,
        5,
        "T",
        &[("keep", ScalarValue::Boolean(false))],
    )?;

    assert_query(
        &graph,
        "MATCH (n:A) \
         RETURN [p = (n)-[r:T]->(m) \
                 WHERE r.keep = true AND m.name = 'kept' | p] AS paths",
        &["paths"],
        vec![
            vec![list(vec![path(
                vec![
                    node(1, &["A"], &[]),
                    node(2, &["B"], &[("name", ScalarValue::String("kept".into()))]),
                ],
                vec![relationship(
                    10,
                    1,
                    2,
                    "T",
                    &[("keep", ScalarValue::Boolean(true))],
                )],
            )])],
            vec![list(vec![])],
        ],
        false,
    )
}
