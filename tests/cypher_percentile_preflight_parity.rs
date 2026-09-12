// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{collections::BTreeMap, sync::Arc, time::Instant};

use irongraph::{
    Bookmark, Error, ErrorCode, ProjectId, Result, ScalarValue,
    cypher::{
        BindCapabilities, ExecutionContext, ExecutionOutput, QueryEngine, ResultValue,
        bind_with_parameters, parse, plan,
    },
    gpu::{CpuBackend, ExecutionBackend},
    graph::GraphStore,
};
use tokio_util::sync::CancellationToken;

const MEMORY_LIMIT: usize = 64 * 1024 * 1024;
const RESERVED_MEMORY: usize = 1024 * 1024;

fn context<'a>(
    graph: &'a GraphStore,
    backend: &'a dyn ExecutionBackend,
    parameters: BTreeMap<String, ResultValue>,
    require_native_execution: bool,
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
        next_node_id: 1,
        next_edge_id: 1,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            require_native_execution,
            ..BindCapabilities::default()
        },
        max_result_rows: 1_024,
        max_batch_rows: 1_024,
        optimizer_statistics: None,
        backend: Some(backend),
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + std::time::Duration::from_secs(10)),
        resolved_query_at_time_nanos: None,
    }
}

fn cpu_backend(graph: &GraphStore) -> Result<CpuBackend> {
    let mut backend = CpuBackend::new(MEMORY_LIMIT, RESERVED_MEMORY);
    backend.admit_graph(Arc::new(graph.snapshot()?))?;
    Ok(backend)
}

fn assert_runtime_number_out_of_range(
    graph: &GraphStore,
    backend: &dyn ExecutionBackend,
    query: &str,
    parameters: BTreeMap<String, ResultValue>,
) -> Result<()> {
    plan(bind_with_parameters(
        parse(query)?,
        graph.catalog(),
        BindCapabilities::default(),
        &parameters,
    )?)?;

    let error = QueryEngine
        .execute(query, &mut context(graph, backend, parameters, false))
        .expect_err("invalid percentile unexpectedly succeeded");
    assert_eq!(error.code, ErrorCode::QueryType, "query: {query}");
    assert_eq!(
        error.message, "NumberOutOfRange: percentile must be finite and in 0..=1",
        "query: {query}"
    );
    Ok(())
}

fn one_value(output: &ExecutionOutput, column: &str) -> Result<ResultValue> {
    output
        .result
        .batches
        .iter()
        .flat_map(|batch| &batch.columns)
        .find(|candidate| candidate.name == column)
        .and_then(|column| column.values.first())
        .cloned()
        .ok_or_else(|| Error::internal(format!("query returned no `{column}` value")))
}

#[test]
fn cpu_preflights_context_free_percentile_literals_parameters_and_expressions() -> Result<()> {
    let graph = GraphStore::default();
    let cpu = cpu_backend(&graph)?;
    let cases = [
        ("percentileCont", "-0.25", BTreeMap::new()),
        (
            "percentileDisc",
            "$percentile",
            BTreeMap::from([(
                "percentile".to_owned(),
                ResultValue::Scalar(ScalarValue::Float(1.25.into())),
            )]),
        ),
        (
            "percentileCont",
            "$base + 0.25",
            BTreeMap::from([(
                "base".to_owned(),
                ResultValue::Scalar(ScalarValue::Integer(1)),
            )]),
        ),
        (
            "percentileDisc",
            "CASE WHEN $use_invalid THEN 2.0 ELSE 0.5 END",
            BTreeMap::from([(
                "use_invalid".to_owned(),
                ResultValue::Scalar(ScalarValue::Boolean(true)),
            )]),
        ),
    ];

    for (function, percentile, parameters) in cases {
        let query = format!(
            "UNWIND [10.0, 20.0, 30.0] AS value \
             RETURN {function}(value, {percentile}) AS result"
        );
        assert_runtime_number_out_of_range(&graph, &cpu, &query, parameters)?;
    }
    Ok(())
}

#[test]
fn cpu_valid_context_free_percentile_expressions_keep_results_unchanged() -> Result<()> {
    let graph = GraphStore::default();
    let cpu = cpu_backend(&graph)?;
    let parameters = BTreeMap::from([(
        "quarter".to_owned(),
        ResultValue::Scalar(ScalarValue::Float(0.25.into())),
    )]);
    let output = QueryEngine.execute(
        "UNWIND [10.0, 20.0, 30.0, 40.0] AS value \
         RETURN percentileCont(value, $quarter + 0.0) AS continuous, \
                percentileDisc(value, CASE WHEN true THEN $quarter ELSE 2.0 END) AS discrete",
        &mut context(&graph, &cpu, parameters, false),
    )?;

    assert_eq!(
        one_value(&output, "continuous")?,
        ResultValue::Scalar(ScalarValue::Float(17.5.into()))
    );
    assert_eq!(
        one_value(&output, "discrete")?,
        ResultValue::Scalar(ScalarValue::Float(10.0.into()))
    );
    Ok(())
}

#[test]
fn row_dependent_aggregation6_percentile_remains_on_normal_cpu_execution_path() -> Result<()> {
    let graph = GraphStore::default();
    let cpu = cpu_backend(&graph)?;
    for function in ["percentileCont", "percentileDisc"] {
        let query = format!(
            "UNWIND range(3, 5) AS deg \
             WITH deg WHERE deg > 2 \
             WITH deg LIMIT 100 \
             RETURN {function}(0.90, deg) AS percentile, deg"
        );
        assert_runtime_number_out_of_range(&graph, &cpu, &query, BTreeMap::new())?;
    }
    Ok(())
}

#[test]
fn rand_dependent_percentile_is_rejected_before_runtime_preflight() -> Result<()> {
    let graph = GraphStore::default();
    let query = "RETURN percentileCont(1.0, rand()) AS percentile";
    let error = bind_with_parameters(
        parse(query)?,
        graph.catalog(),
        BindCapabilities::default(),
        &BTreeMap::new(),
    )
    .expect_err("volatile aggregate argument unexpectedly bound");
    assert_eq!(error.code, ErrorCode::QuerySyntax);
    assert_eq!(
        error.message,
        "NonConstantExpression: aggregate arguments must not contain volatile expressions"
    );
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn invalid_parameter_precedes_real_metal_resident_admission() -> Result<()> {
    use irongraph::gpu::MetalBackend;

    let graph = GraphStore::default();
    let metal = MetalBackend::new(0, MEMORY_LIMIT, RESERVED_MEMORY)?;
    for function in ["percentileCont", "percentileDisc"] {
        let query = format!(
            "UNWIND [10.0, 20.0, 30.0] AS value \
             RETURN {function}(value, $percentile) AS result"
        );
        let parameters = BTreeMap::from([(
            "percentile".to_owned(),
            ResultValue::Scalar(ScalarValue::Float(1.25.into())),
        )]);
        let error = QueryEngine
            .execute(&query, &mut context(&graph, &metal, parameters, true))
            .expect_err("Metal accepted an out-of-range percentile");
        assert_eq!(error.code, ErrorCode::QueryType, "query: {query}");
        assert_eq!(
            error.message, "NumberOutOfRange: percentile must be finite and in 0..=1",
            "query: {query}"
        );
    }
    Ok(())
}
