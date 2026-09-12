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
        backend: Some(backend),
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + std::time::Duration::from_secs(5)),
        resolved_query_at_time_nanos: None,
    }
}

fn execute_cpu(query: &str, parameters: BTreeMap<String, ResultValue>) -> Result<ExecutionOutput> {
    let graph = GraphStore::default();
    let mut backend = CpuBackend::new(MEMORY_LIMIT, RESERVED_MEMORY);
    backend.admit_graph(Arc::new(graph.snapshot()?))?;
    QueryEngine.execute(query, &mut context(&graph, &backend, parameters))
}

fn require_runtime_error(query: &str, parameters: BTreeMap<String, ResultValue>) -> Result<Error> {
    let graph = GraphStore::default();
    plan(bind_with_parameters(
        parse(query)?,
        graph.catalog(),
        BindCapabilities::default(),
        &parameters,
    )?)?;
    let mut backend = CpuBackend::new(MEMORY_LIMIT, RESERVED_MEMORY);
    backend.admit_graph(Arc::new(graph.snapshot()?))?;
    QueryEngine
        .execute(query, &mut context(&graph, &backend, parameters))
        .err()
        .ok_or_else(|| Error::internal(format!("query unexpectedly succeeded: {query}")))
}

fn percentile_parameter(value: ScalarValue) -> BTreeMap<String, ResultValue> {
    BTreeMap::from([("percentile".to_owned(), ResultValue::Scalar(value))])
}

fn value(output: &ExecutionOutput, column: &str) -> Result<ResultValue> {
    output
        .result
        .batches
        .first()
        .and_then(|batch| {
            batch
                .columns
                .iter()
                .find(|candidate| candidate.name == column)
        })
        .and_then(|column| column.values.first())
        .cloned()
        .ok_or_else(|| Error::internal(format!("column `{column}` returned no value")))
}

fn require_number_out_of_range(
    function: &str,
    percentile: ScalarValue,
    description: &str,
) -> Result<()> {
    let query = format!(
        "UNWIND [10.0, 20.0, 30.0] AS value \
         RETURN {function}(value, $percentile) AS result"
    );
    let error = require_runtime_error(&query, percentile_parameter(percentile))?;
    assert_eq!(
        error.code,
        ErrorCode::QueryType,
        "{function}: {description}"
    );
    assert_eq!(
        error.message, "NumberOutOfRange: percentile must be finite and in 0..=1",
        "{function}: {description}"
    );
    Ok(())
}

#[test]
fn invalid_percentile_arguments_use_number_out_of_range_at_runtime() -> Result<()> {
    let invalid = [
        ("negative integer", ScalarValue::Integer(-1)),
        ("large integer", ScalarValue::Integer(1_000)),
        ("fraction above one", ScalarValue::Float(1.1.into())),
        ("NaN", ScalarValue::Float(f64::NAN.into())),
        (
            "positive infinity",
            ScalarValue::Float(f64::INFINITY.into()),
        ),
        (
            "negative infinity",
            ScalarValue::Float(f64::NEG_INFINITY.into()),
        ),
    ];

    for function in ["percentileCont", "percentileDisc"] {
        for (description, percentile) in &invalid {
            require_number_out_of_range(function, percentile.clone(), description)?;
        }
    }
    Ok(())
}

#[test]
fn zero_and_one_remain_valid_for_both_percentile_functions() -> Result<()> {
    for function in ["percentileCont", "percentileDisc"] {
        for (percentile, expected) in [(0.0, 10.0), (1.0, 30.0)] {
            let output = execute_cpu(
                &format!(
                    "UNWIND [10.0, 20.0, 30.0] AS value \
                     RETURN {function}(value, $percentile) AS result"
                ),
                percentile_parameter(ScalarValue::Float(percentile.into())),
            )?;
            assert_eq!(
                value(&output, "result")?,
                ResultValue::Scalar(ScalarValue::Float(expected.into())),
                "{function} at {percentile}"
            );
        }
    }
    Ok(())
}

#[test]
fn valid_continuous_and_discrete_percentile_math_is_unchanged() -> Result<()> {
    let output = execute_cpu(
        "UNWIND [10.0, 20.0, 30.0, 40.0] AS value \
         RETURN percentileCont(value, 0.25) AS continuous, \
                percentileDisc(value, 0.25) AS discrete",
        BTreeMap::new(),
    )?;
    assert_eq!(
        value(&output, "continuous")?,
        ResultValue::Scalar(ScalarValue::Float(17.5.into()))
    );
    assert_eq!(
        value(&output, "discrete")?,
        ResultValue::Scalar(ScalarValue::Float(10.0.into()))
    );
    Ok(())
}

#[test]
fn dynamic_percentile_arguments_are_range_checked_per_group() -> Result<()> {
    for function in ["percentileCont", "percentileDisc"] {
        let query = format!(
            "UNWIND range(3, 5) AS deg \
             WITH deg WHERE deg > 2 \
             WITH deg LIMIT 100 \
             RETURN {function}(0.90, deg) AS percentile, deg"
        );
        let error = require_runtime_error(&query, BTreeMap::new())?;
        assert_eq!(error.code, ErrorCode::QueryType, "{function}");
        assert_eq!(
            error.message, "NumberOutOfRange: percentile must be finite and in 0..=1",
            "{function}"
        );
    }
    Ok(())
}

#[test]
fn null_values_and_non_numeric_percentile_arguments_keep_their_existing_semantics() -> Result<()> {
    let output = execute_cpu(
        "UNWIND [null, 10.0, null] AS value \
         RETURN percentileCont(value, 0.5) AS one_value",
        BTreeMap::new(),
    )?;
    assert_eq!(
        value(&output, "one_value")?,
        ResultValue::Scalar(ScalarValue::Float(10.0.into()))
    );

    let output = execute_cpu(
        "UNWIND [null, null] AS value \
         RETURN percentileDisc(value, 0.5) AS no_values",
        BTreeMap::new(),
    )?;
    assert_eq!(
        value(&output, "no_values")?,
        ResultValue::Scalar(ScalarValue::Null)
    );

    for percentile in [ScalarValue::Null, ScalarValue::String("half".into())] {
        let error = require_runtime_error(
            "UNWIND [10.0] AS value \
             RETURN percentileCont(value, $percentile) AS result",
            percentile_parameter(percentile),
        )?;
        assert_eq!(error.code, ErrorCode::QueryType);
        assert_eq!(error.message, "numeric value required");
    }
    Ok(())
}
