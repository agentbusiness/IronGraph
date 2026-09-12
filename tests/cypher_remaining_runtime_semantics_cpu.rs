// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{collections::BTreeMap, sync::Arc, time::Instant};

use irongraph::{
    Bookmark, EdgeId, Error, ErrorCode, Layer, NodeId, ProjectId, Result, ScalarValue,
    cypher::{BindCapabilities, ExecutionContext, ExecutionOutput, QueryEngine, ResultValue},
    gpu::{CpuBackend, ExecutionBackend},
    graph::{EdgeInput, GraphStore, NodeInput},
};
use tokio_util::sync::CancellationToken;

const MEMORY_LIMIT: usize = 128 * 1024 * 1024;
const RESERVED_MEMORY: usize = 1024 * 1024;

fn context<'a>(
    graph: &'a GraphStore,
    backend: Option<&'a dyn ExecutionBackend>,
    max_result_rows: usize,
    parameters: BTreeMap<String, ResultValue>,
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
        parameters,
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
        max_result_rows,
        max_batch_rows: max_result_rows.max(1),
        optimizer_statistics: None,
        backend,
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + std::time::Duration::from_secs(5)),
        resolved_query_at_time_nanos: None,
    }
}

fn cpu_backend(graph: &GraphStore) -> Result<CpuBackend> {
    let mut backend = CpuBackend::new(MEMORY_LIMIT, RESERVED_MEMORY);
    backend.admit_graph(Arc::new(graph.snapshot()?))?;
    Ok(backend)
}

fn execute_cpu(graph: &GraphStore, query: &str, max_rows: usize) -> Result<ExecutionOutput> {
    let backend = cpu_backend(graph)?;
    QueryEngine.execute(
        query,
        &mut context(graph, Some(&backend), max_rows, BTreeMap::new()),
    )
}

fn schema_names(output: &ExecutionOutput) -> Vec<String> {
    output
        .result
        .schema
        .iter()
        .map(|(name, _)| name.clone())
        .collect()
}

fn one_value<'a>(output: &'a ExecutionOutput, name: &str) -> Result<&'a ResultValue> {
    output
        .result
        .batches
        .iter()
        .flat_map(|batch| &batch.columns)
        .find(|column| column.name == name)
        .and_then(|column| column.values.first())
        .ok_or_else(|| Error::internal(format!("query returned no `{name}` value")))
}

#[test]
fn empty_pipelines_retain_logical_projection_and_star_schemas() -> Result<()> {
    let graph = GraphStore::default();
    let cases = [
        (
            "OPTIONAL MATCH (a:Start) WITH a MATCH (a)-->(b) RETURN *",
            vec!["a", "b"],
        ),
        (
            "MATCH (me: Person)--(you: Person) \
             WITH me.age AS age, you \
             WITH age, age + count(you.age) AS agg \
             RETURN *",
            vec!["age", "agg"],
        ),
        (
            "MATCH (me: Person)--(you: Person) \
             WITH me.age AS age, me.age + count(you.age) AS agg \
             RETURN *",
            vec!["age", "agg"],
        ),
        ("MATCH (n:Missing) SET n.value = 1 RETURN *", vec!["n"]),
        ("UNWIND [] AS x CREATE (n:Never) RETURN *", vec!["n", "x"]),
    ];

    for (query, expected) in cases {
        let output = execute_cpu(&graph, query, 4_096)?;
        assert_eq!(
            schema_names(&output),
            expected.into_iter().map(str::to_owned).collect::<Vec<_>>(),
            "query: {query}"
        );
        assert!(output.result.batches.is_empty(), "query: {query}");
        assert!(output.graph_mutations.is_empty(), "query: {query}");
    }
    Ok(())
}

fn self_loop_fixture(parallel_loops: usize) -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    let single = graph.catalog_mut().intern_label("Single")?;
    let a = graph.catalog_mut().intern_label("A")?;
    let b = graph.catalog_mut().intern_label("B")?;
    let c = graph.catalog_mut().intern_label("C")?;
    let rel = graph.catalog_mut().intern_relationship_type("REL")?;
    let loop_type = graph.catalog_mut().intern_relationship_type("LOOP")?;
    let num = graph.catalog_mut().intern_property("num")?;
    for (id, label, value) in [
        (NodeId(1), single, 0_i64),
        (NodeId(2), a, 42),
        (NodeId(3), b, 46),
        (NodeId(4), c, 99),
    ] {
        graph.insert_node(NodeInput {
            id,
            layer: Layer::Observed,
            revision: id.0,
            labels: vec![label],
            properties: vec![(num, ScalarValue::Integer(value))],
        })?;
    }
    for (id, source, target) in [
        (EdgeId(1), NodeId(1), NodeId(2)),
        (EdgeId(2), NodeId(1), NodeId(3)),
        (EdgeId(3), NodeId(2), NodeId(4)),
    ] {
        graph.insert_edge(EdgeInput {
            id,
            source,
            target,
            relationship_type: rel,
            layer: Layer::Observed,
            revision: 4 + id.0,
            properties: Vec::new(),
        })?;
    }
    for loop_index in 0..parallel_loops {
        let id = EdgeId(10 + loop_index as u64);
        graph.insert_edge(EdgeInput {
            id,
            source: NodeId(3),
            target: NodeId(3),
            relationship_type: loop_type,
            layer: Layer::Observed,
            revision: 10 + loop_index as u64,
            properties: Vec::new(),
        })?;
    }
    Ok(graph)
}

fn returned_relationship_ids(output: &ExecutionOutput) -> Result<Vec<EdgeId>> {
    output
        .result
        .batches
        .iter()
        .flat_map(|batch| &batch.columns)
        .find(|column| column.name == "r")
        .ok_or_else(|| Error::internal("query returned no `r` column"))?
        .values
        .iter()
        .map(|value| match value {
            ResultValue::Relationship(relationship) => Ok(relationship.id),
            other => Err(Error::internal(format!(
                "expected a relationship, got {other:?}"
            ))),
        })
        .collect()
}

#[test]
fn optional_undirected_repeated_endpoint_accepts_only_self_loops() -> Result<()> {
    let official = self_loop_fixture(1)?;
    let output = execute_cpu(
        &official,
        "MATCH (a:B) OPTIONAL MATCH (a)-[r]-(a) RETURN r",
        4_096,
    )?;
    assert_eq!(returned_relationship_ids(&output)?, vec![EdgeId(10)]);

    let parallel = self_loop_fixture(2)?;
    let output = execute_cpu(
        &parallel,
        "MATCH (a:B) OPTIONAL MATCH (a)-[r]-(a) RETURN r",
        4_096,
    )?;
    let mut ids = returned_relationship_ids(&output)?;
    ids.sort_unstable();
    assert_eq!(ids, vec![EdgeId(10), EdgeId(11)]);
    Ok(())
}

#[test]
fn downstream_limit_bounds_range_unwind_without_weakening_global_budget() -> Result<()> {
    let graph = GraphStore::default();
    let output = QueryEngine.execute(
        "UNWIND range(1000000, 2000000) AS i \
         WITH i LIMIT 3000 \
         RETURN sum(i) AS total",
        &mut context(&graph, None, 4_096, BTreeMap::new()),
    )?;
    assert_eq!(
        one_value(&output, "total")?,
        &ResultValue::Scalar(ScalarValue::Integer(3_004_498_500))
    );

    let exact_budget = QueryEngine.execute(
        "UNWIND range(0, 1000000) AS i \
         WITH i LIMIT 128 \
         RETURN count(*) AS count",
        &mut context(&graph, None, 128, BTreeMap::new()),
    )?;
    assert_eq!(
        one_value(&exact_budget, "count")?,
        &ResultValue::Scalar(ScalarValue::Integer(128))
    );

    let over_budget = QueryEngine
        .execute(
            "UNWIND range(0, 1000000) AS i WITH i LIMIT 129 RETURN count(*)",
            &mut context(&graph, None, 128, BTreeMap::new()),
        )
        .expect_err("LIMIT larger than the global row budget was accepted");
    assert_eq!(over_budget.code, ErrorCode::ResultBudgetExceeded);
    Ok(())
}

#[test]
fn range_demand_pushdown_stops_at_cardinality_changing_operators_and_keeps_errors() -> Result<()> {
    let graph = GraphStore::default();
    for query in [
        "UNWIND range(0, 10000) AS i RETURN i",
        "UNWIND range(0, 10000) AS i WITH i WHERE i < 2 LIMIT 1 RETURN i",
        "UNWIND range(0, 10000) AS i WITH DISTINCT i LIMIT 1 RETURN i",
        "UNWIND range(0, 10000) AS i WITH i ORDER BY i LIMIT 1 RETURN i",
    ] {
        let error = QueryEngine
            .execute(query, &mut context(&graph, None, 128, BTreeMap::new()))
            .expect_err("unsafe LIMIT demand crossed a cardinality-changing operator");
        assert_eq!(
            error.code,
            ErrorCode::ResultBudgetExceeded,
            "query: {query}"
        );
    }

    let zero_step = QueryEngine
        .execute(
            "UNWIND range(0, 10, 0) AS i WITH i LIMIT 1 RETURN i",
            &mut context(&graph, None, 128, BTreeMap::new()),
        )
        .expect_err("demand-bounded range accepted a zero step");
    assert_eq!(zero_step.code, ErrorCode::QueryType);

    let descending = QueryEngine.execute(
        "UNWIND range(5, -1, -2) AS i WITH i LIMIT 3 RETURN collect(i) AS values",
        &mut context(&graph, None, 128, BTreeMap::new()),
    )?;
    assert_eq!(
        one_value(&descending, "values")?,
        &ResultValue::List(vec![
            ResultValue::Scalar(ScalarValue::Integer(5)),
            ResultValue::Scalar(ScalarValue::Integer(3)),
            ResultValue::Scalar(ScalarValue::Integer(1)),
        ])
    );
    Ok(())
}

#[test]
fn merge_null_preflight_is_runtime_and_skips_row_dependent_or_volatile_values() -> Result<()> {
    let graph = GraphStore::default();
    let cases = [
        ("MERGE ({num: null})", BTreeMap::new()),
        (
            "CREATE (a), (b) MERGE (a)-[r:X {num: null}]->(b)",
            BTreeMap::new(),
        ),
        (
            "MERGE (:Item {num: $num})",
            BTreeMap::from([("num".to_owned(), ResultValue::Scalar(ScalarValue::Null))]),
        ),
        (
            "MERGE (:Item {num: CASE WHEN true THEN null ELSE 1 END})",
            BTreeMap::new(),
        ),
    ];
    for (query, parameters) in cases {
        let error = QueryEngine
            .execute(query, &mut context(&graph, None, 128, parameters))
            .expect_err("MERGE accepted a context-free null property");
        assert_eq!(error.code, ErrorCode::QueryType, "query: {query}");
        assert!(
            error.message.contains("MergeReadOwnWrites"),
            "query: {query}"
        );
    }

    let row_dependent = QueryEngine.execute(
        "UNWIND [1] AS num MERGE (n:Item {num: num}) RETURN n.num AS num",
        &mut context(&graph, None, 128, BTreeMap::new()),
    )?;
    assert_eq!(
        one_value(&row_dependent, "num")?,
        &ResultValue::Scalar(ScalarValue::Integer(1))
    );

    let volatile = QueryEngine.execute(
        "MERGE (n:Random {num: rand()}) RETURN n.num AS num",
        &mut context(&graph, None, 128, BTreeMap::new()),
    )?;
    assert!(matches!(
        one_value(&volatile, "num")?,
        ResultValue::Scalar(ScalarValue::Float(_))
    ));
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn merge_null_preflight_precedes_unadmitted_metal_execution() -> Result<()> {
    use irongraph::gpu::MetalBackend;

    let graph = GraphStore::default();
    let metal = MetalBackend::new(0, MEMORY_LIMIT, RESERVED_MEMORY)?;
    for query in [
        "MERGE ({num: null})",
        "CREATE (a), (b) MERGE (a)-[r:X {num: null}]->(b)",
    ] {
        let error = QueryEngine
            .execute(
                query,
                &mut context(&graph, Some(&metal), 128, BTreeMap::new()),
            )
            .expect_err("Metal admission hid the shared MERGE runtime error");
        assert_eq!(error.code, ErrorCode::QueryType, "query: {query}");
        assert!(
            error.message.contains("MergeReadOwnWrites"),
            "query: {query}"
        );
    }
    Ok(())
}
