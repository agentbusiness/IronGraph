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
        bookmark: Bookmark { term: 1, index: 40 },
        mutation_revision: 41,
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

fn insert_node(graph: &mut GraphStore, id: u64, label: &str, name: &str) -> Result<()> {
    let label = graph.catalog_mut().intern_label(label)?;
    let name_property = graph.catalog_mut().intern_property("name")?;
    graph
        .insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: id,
            labels: vec![label],
            properties: vec![(name_property, ScalarValue::String(name.into()))],
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

/// The exact four-node, four-relationship fixture used by every valid Pattern1 TCK scenario.
fn pattern1_graph() -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    for (id, label) in [(1, "A"), (2, "B"), (3, "C"), (4, "D")] {
        insert_node(&mut graph, id, label, label)?;
    }
    insert_edge(&mut graph, 10, 1, 2, "REL1")?;
    insert_edge(&mut graph, 11, 2, 1, "REL2")?;
    insert_edge(&mut graph, 12, 1, 3, "REL3")?;
    insert_edge(&mut graph, 13, 1, 4, "REL1")?;
    Ok(graph)
}

fn column_values<'a>(output: &'a ExecutionOutput, name: &str) -> Result<Vec<&'a ResultValue>> {
    let mut values = Vec::new();
    for batch in &output.result.batches {
        let column = batch
            .columns
            .iter()
            .find(|column| column.name == name)
            .ok_or_else(|| Error::internal(format!("result omitted column `{name}`")))?;
        values.extend(column.values.iter());
    }
    Ok(values)
}

fn string_value(value: &ResultValue, column: &str) -> Result<String> {
    let ResultValue::Scalar(ScalarValue::String(value)) = value else {
        return Err(Error::internal(format!(
            "column `{column}` returned a non-string value: {value:?}"
        )));
    };
    Ok(value.to_string())
}

fn assert_names(graph: &GraphStore, query: &str, expected: &[&str]) -> Result<()> {
    let output = QueryEngine.execute(query, &mut context(graph))?;
    assert!(
        output.graph_mutations.is_empty(),
        "read-only pattern predicate produced mutations: {query}"
    );
    let mut actual = column_values(&output, "name")?
        .into_iter()
        .map(|value| string_value(value, "name"))
        .collect::<Result<Vec<_>>>()?;
    let mut expected = expected
        .iter()
        .map(|value| (*value).to_owned())
        .collect::<Vec<_>>();
    actual.sort();
    expected.sort();
    assert_eq!(actual, expected, "query: {query}");
    Ok(())
}

fn assert_pairs(graph: &GraphStore, query: &str, expected: &[(&str, &str)]) -> Result<()> {
    let output = QueryEngine.execute(query, &mut context(graph))?;
    assert!(
        output.graph_mutations.is_empty(),
        "read-only pattern predicate produced mutations: {query}"
    );
    let left = column_values(&output, "left")?;
    let right = column_values(&output, "right")?;
    if left.len() != right.len() {
        return Err(Error::internal(format!(
            "result columns have different lengths for `{query}`: {} and {}",
            left.len(),
            right.len()
        )));
    }
    let mut actual = left
        .into_iter()
        .zip(right)
        .map(|(left, right)| Ok((string_value(left, "left")?, string_value(right, "right")?)))
        .collect::<Result<Vec<_>>>()?;
    let mut expected = expected
        .iter()
        .map(|(left, right)| ((*left).to_owned(), (*right).to_owned()))
        .collect::<Vec<_>>();
    actual.sort();
    expected.sort();
    assert_eq!(actual, expected, "query: {query}");
    Ok(())
}

macro_rules! single_endpoint_scenario {
    ($name:ident, $query:literal, [$($expected:literal),* $(,)?]) => {
        #[test]
        fn $name() -> Result<()> {
            let graph = pattern1_graph()?;
            assert_names(&graph, $query, &[$($expected),*])
        }
    };
}

macro_rules! two_endpoint_scenario {
    ($name:ident, $query:literal, [$(($left:literal, $right:literal)),* $(,)?]) => {
        #[test]
        fn $name() -> Result<()> {
            let graph = pattern1_graph()?;
            assert_pairs(&graph, $query, &[$(($left, $right)),*])
        }
    };
}

single_endpoint_scenario!(
    pattern1_01_any_outgoing_connection,
    "MATCH (n) WHERE (n)-[]->() RETURN n.name AS name",
    ["A", "B"]
);

single_endpoint_scenario!(
    pattern1_02_any_undirected_connection,
    "MATCH (n) WHERE (n)-[]-() RETURN n.name AS name",
    ["A", "B", "C", "D"]
);

single_endpoint_scenario!(
    pattern1_03_any_incoming_connection,
    "MATCH (n) WHERE (n)<-[]-() RETURN n.name AS name",
    ["A", "B", "C", "D"]
);

single_endpoint_scenario!(
    pattern1_04_typed_outgoing_connection,
    "MATCH (n) WHERE (n)-[:REL1]->() RETURN n.name AS name",
    ["A"]
);

single_endpoint_scenario!(
    pattern1_05_typed_undirected_connection,
    "MATCH (n) WHERE (n)-[:REL1]-() RETURN n.name AS name",
    ["A", "B", "D"]
);

single_endpoint_scenario!(
    pattern1_06_typed_incoming_connection,
    "MATCH (n) WHERE (n)<-[:REL1]-() RETURN n.name AS name",
    ["B", "D"]
);

single_endpoint_scenario!(
    pattern1_07_variable_length_outgoing_connection,
    "MATCH (n) WHERE (n)-[:REL1*]->() RETURN n.name AS name",
    ["A"]
);

single_endpoint_scenario!(
    pattern1_08_variable_length_undirected_connection,
    "MATCH (n) WHERE (n)-[:REL1*]-() RETURN n.name AS name",
    ["A", "B", "D"]
);

single_endpoint_scenario!(
    pattern1_09_variable_length_incoming_connection,
    "MATCH (n) WHERE (n)<-[:REL1*]-() RETURN n.name AS name",
    ["B", "D"]
);

single_endpoint_scenario!(
    pattern1_10_exact_length_two_undirected_connection,
    "MATCH (n) WHERE (n)-[:REL1*2]-() RETURN n.name AS name",
    ["B", "D"]
);

two_endpoint_scenario!(
    pattern1_12_any_directed_connection_between_bound_nodes,
    "MATCH (n), (m) WHERE (n)-[]->(m) RETURN n.name AS left, m.name AS right",
    [("A", "B"), ("B", "A"), ("A", "C"), ("A", "D")]
);

two_endpoint_scenario!(
    pattern1_13_any_typed_undirected_connection_between_bound_nodes,
    "MATCH (n), (m) WHERE (n)-[:REL1|REL2|REL3|REL4]-(m) \
     RETURN n.name AS left, m.name AS right",
    [
        ("A", "B"),
        ("B", "A"),
        ("A", "C"),
        ("C", "A"),
        ("A", "D"),
        ("D", "A"),
    ]
);

two_endpoint_scenario!(
    pattern1_14_typed_outgoing_connection_between_bound_nodes,
    "MATCH (n), (m) WHERE (n)-[:REL1]->(m) RETURN n.name AS left, m.name AS right",
    [("A", "B"), ("A", "D")]
);

two_endpoint_scenario!(
    pattern1_15_typed_undirected_connection_between_bound_nodes,
    "MATCH (n), (m) WHERE (n)-[:REL1]-(m) RETURN n.name AS left, m.name AS right",
    [("A", "B"), ("B", "A"), ("A", "D"), ("D", "A")]
);

two_endpoint_scenario!(
    pattern1_16_variable_length_outgoing_between_bound_nodes,
    "MATCH (n), (m) WHERE (n)-[:REL1*]->(m) RETURN n.name AS left, m.name AS right",
    [("A", "B"), ("A", "D")]
);

two_endpoint_scenario!(
    pattern1_17_variable_length_undirected_between_bound_nodes,
    "MATCH (n), (m) WHERE (n)-[:REL1*]-(m) RETURN n.name AS left, m.name AS right",
    [
        ("A", "B"),
        ("A", "D"),
        ("B", "A"),
        ("B", "D"),
        ("D", "A"),
        ("D", "B"),
    ]
);

two_endpoint_scenario!(
    pattern1_18_exact_length_two_undirected_between_bound_nodes,
    "MATCH (n), (m) WHERE (n)-[:REL1*2]-(m) RETURN n.name AS left, m.name AS right",
    [("B", "D"), ("D", "B")]
);

single_endpoint_scenario!(
    pattern1_19_negated_existential_predicate,
    "MATCH (n) WHERE NOT (n)-[:REL2]-() RETURN n.name AS name",
    ["C", "D"]
);

single_endpoint_scenario!(
    pattern1_20_conjoined_existential_predicates,
    "MATCH (n) WHERE (n)-[:REL1]-() AND (n)-[:REL3]-() RETURN n.name AS name",
    ["A"]
);

single_endpoint_scenario!(
    pattern1_21_disjoined_existential_predicates,
    "MATCH (n) WHERE (n)-[:REL1]-() OR (n)-[:REL2]-() RETURN n.name AS name",
    ["A", "B", "D"]
);

fn duplicate_path_graph() -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    for (id, name) in [(1, "A"), (2, "B"), (3, "C"), (4, "D")] {
        insert_node(&mut graph, id, "N", name)?;
    }

    // Parallel relationships create duplicate one-hop witnesses.
    insert_edge(&mut graph, 20, 1, 2, "R")?;
    insert_edge(&mut graph, 21, 1, 2, "R")?;
    insert_edge(&mut graph, 22, 1, 3, "R")?;
    // The diamond creates three physical two-hop witnesses for the same (A, D) outer row.
    insert_edge(&mut graph, 23, 2, 4, "R")?;
    insert_edge(&mut graph, 24, 3, 4, "R")?;
    // Two parallel self-loops must still produce one Boolean witness for (A) or (A, A).
    insert_edge(&mut graph, 25, 1, 1, "SELF")?;
    insert_edge(&mut graph, 26, 1, 1, "SELF")?;
    Ok(graph)
}

#[test]
fn existential_predicates_do_not_duplicate_rows_for_parallel_relationships() -> Result<()> {
    let graph = duplicate_path_graph()?;
    assert_names(
        &graph,
        "MATCH (n) WHERE (n)-[:R]->() RETURN n.name AS name",
        &["A", "B", "C"],
    )?;
    assert_pairs(
        &graph,
        "MATCH (n), (m) WHERE (n)-[:R]->(m) RETURN n.name AS left, m.name AS right",
        &[("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")],
    )
}

#[test]
fn existential_predicates_do_not_duplicate_rows_for_multiple_variable_length_paths() -> Result<()> {
    let graph = duplicate_path_graph()?;
    assert_pairs(
        &graph,
        "MATCH (n), (m) WHERE (n)-[:R*2]->(m) RETURN n.name AS left, m.name AS right",
        &[("A", "D")],
    )
}

#[test]
fn existential_predicates_treat_self_loops_as_one_boolean_witness() -> Result<()> {
    let graph = duplicate_path_graph()?;
    assert_names(
        &graph,
        "MATCH (n) WHERE (n)-[:SELF]-() RETURN n.name AS name",
        &["A"],
    )?;
    assert_pairs(
        &graph,
        "MATCH (n), (m) WHERE (n)-[:SELF]-(m) RETURN n.name AS left, m.name AS right",
        &[("A", "A")],
    )
}
