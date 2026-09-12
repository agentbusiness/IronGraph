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
        bookmark: Bookmark { term: 1, index: 30 },
        mutation_revision: 31,
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

fn insert_node(graph: &mut GraphStore, id: u64, labels: &[&str]) -> Result<()> {
    let labels = labels
        .iter()
        .map(|label| graph.catalog_mut().intern_label(label))
        .collect::<Result<Vec<_>>>()?;
    graph
        .insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: id,
            labels,
            properties: Vec::new(),
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

fn node_label_key(value: &ResultValue) -> Result<String> {
    let ResultValue::Node(node) = value else {
        return Err(Error::internal("expected a node result"));
    };
    let mut labels = node.labels.clone();
    labels.sort();
    Ok(labels.join(":"))
}

fn boolean_result(value: &ResultValue) -> Result<bool> {
    let ResultValue::Scalar(ScalarValue::Boolean(value)) = value else {
        return Err(Error::internal("expected a Boolean result"));
    };
    Ok(*value)
}

fn graph5_nodes(include_abc: bool) -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    let mut definitions = vec![
        &["A", "B"][..],
        &["A", "C"][..],
        &["B", "C"][..],
        &["A"][..],
        &["B"][..],
        &["C"][..],
        &[][..],
    ];
    if include_abc {
        definitions.insert(0, &["A", "B", "C"]);
    }
    for (offset, labels) in definitions.into_iter().enumerate() {
        insert_node(
            &mut graph,
            u64::try_from(offset).unwrap_or(0).saturating_add(1),
            labels,
        )?;
    }
    Ok(graph)
}

fn node_predicate_results(output: &ExecutionOutput) -> Result<BTreeMap<String, bool>> {
    column_values(output, "entity")?
        .iter()
        .zip(column_values(output, "result")?)
        .map(|(node, result)| Ok((node_label_key(node)?, boolean_result(&result)?)))
        .collect()
}

#[test]
fn graph5_1_single_label_predicate_executes_for_nodes() -> Result<()> {
    let graph = graph5_nodes(true)?;
    let output = QueryEngine.execute(
        "MATCH (a) RETURN a AS entity, a:B AS result",
        &mut context(&graph),
    )?;
    assert_eq!(
        node_predicate_results(&output)?,
        BTreeMap::from([
            (String::new(), false),
            ("A".to_owned(), false),
            ("A:B".to_owned(), true),
            ("A:B:C".to_owned(), true),
            ("A:C".to_owned(), false),
            ("B".to_owned(), true),
            ("B:C".to_owned(), true),
            ("C".to_owned(), false),
        ])
    );
    Ok(())
}

#[test]
fn graph5_2_relationship_predicate_is_exact_case_sensitive_type_equality() -> Result<()> {
    let mut graph = GraphStore::default();
    for id in 1..=6 {
        insert_node(&mut graph, id, &[])?;
    }
    for (id, relationship_type) in [(10, "T1"), (11, "T2"), (12, "t2"), (13, "T3"), (14, "T4")] {
        insert_edge(&mut graph, id, id - 9, id - 8, relationship_type)?;
    }
    let output = QueryEngine.execute(
        "MATCH ()-[r]->() \
         RETURN r AS entity, r:T2 AS result, r:T2:T2 AS repeated, r:T2:T1 AS distinct",
        &mut context(&graph),
    )?;
    let entities = column_values(&output, "entity")?;
    let results = column_values(&output, "result")?;
    let repeated = column_values(&output, "repeated")?;
    let distinct = column_values(&output, "distinct")?;
    let mut actual = BTreeMap::new();
    for (((entity, result), repeated), distinct) in entities
        .iter()
        .zip(results.iter())
        .zip(repeated.iter())
        .zip(distinct.iter())
    {
        let ResultValue::Relationship(edge) = entity else {
            return Err(Error::internal("expected a relationship result"));
        };
        actual.insert(
            edge.relationship_type.clone(),
            (
                boolean_result(result)?,
                boolean_result(repeated)?,
                boolean_result(distinct)?,
            ),
        );
    }
    assert_eq!(
        actual,
        BTreeMap::from([
            ("T1".to_owned(), (false, false, false)),
            ("T2".to_owned(), (true, true, false)),
            ("T3".to_owned(), (false, false, false)),
            ("T4".to_owned(), (false, false, false)),
            ("t2".to_owned(), (false, false, false)),
        ])
    );
    Ok(())
}

#[test]
fn graph5_3_conjunctive_node_predicate_requires_every_label() -> Result<()> {
    let graph = graph5_nodes(true)?;
    let output = QueryEngine.execute(
        "MATCH (a) RETURN a AS entity, a:A:B AS result",
        &mut context(&graph),
    )?;
    let actual = node_predicate_results(&output)?;
    assert_eq!(actual.get("A:B"), Some(&true));
    assert_eq!(actual.get("A:B:C"), Some(&true));
    assert!(
        actual
            .iter()
            .all(|(labels, value)| matches!(labels.as_str(), "A:B" | "A:B:C") || !value)
    );
    Ok(())
}

#[test]
fn graph5_4_reordered_and_repeated_names_remain_one_conjunction() -> Result<()> {
    let graph = graph5_nodes(false)?;
    for suffix in ["A:C", "C:A", "A:C:A", "C:C:A", "C:A:A:C"] {
        let query = format!("MATCH (a) WHERE a:{suffix} RETURN a AS entity");
        let output = QueryEngine.execute(&query, &mut context(&graph))?;
        let entities = column_values(&output, "entity")?;
        assert_eq!(entities.len(), 1, "suffix: {suffix}");
        assert_eq!(node_label_key(&entities[0])?, "A:C", "suffix: {suffix}");
    }
    Ok(())
}

#[test]
fn graph5_5_null_optional_binding_propagates_null() -> Result<()> {
    let mut graph = GraphStore::default();
    insert_node(&mut graph, 1, &["Single"])?;
    let output = QueryEngine.execute(
        "MATCH (n:Single) OPTIONAL MATCH (n)-[r:TYPE]-(m) \
         RETURN m:TYPE AS result",
        &mut context(&graph),
    )?;
    assert_eq!(
        column_values(&output, "result")?,
        vec![ResultValue::Scalar(ScalarValue::Null)]
    );
    Ok(())
}

#[test]
fn scalar_label_predicate_source_is_an_invalid_argument_type() -> Result<()> {
    let graph = GraphStore::default();
    let error = QueryEngine
        .execute("RETURN 1:A AS result", &mut context(&graph))
        .expect_err("a scalar label predicate source was accepted");
    assert_eq!(error.code, ErrorCode::QueryType);
    assert!(error.message.contains("InvalidArgumentType"));
    Ok(())
}
