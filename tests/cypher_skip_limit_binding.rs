// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{collections::BTreeMap, time::Instant};

use irongraph::{
    Bookmark, Error, ErrorCode, ProjectId, Result, ScalarValue,
    cypher::{
        BindCapabilities, BoundQuery, ExecutionContext, QueryEngine, ResultValue, bind,
        bind_with_parameters, parse,
    },
    graph::{GraphStore, NameCatalog},
};
use tokio_util::sync::CancellationToken;

fn bind_query(query: &str) -> Result<BoundQuery> {
    bind(
        parse(query)?,
        &NameCatalog::default(),
        BindCapabilities::default(),
    )
}

fn bind_parameterized(
    query: &str,
    parameters: &BTreeMap<String, ResultValue>,
) -> Result<BoundQuery> {
    bind_with_parameters(
        parse(query)?,
        &NameCatalog::default(),
        BindCapabilities::default(),
        parameters,
    )
}

fn require_error<T>(result: Result<T>, context: &str) -> Result<Error> {
    result
        .err()
        .ok_or_else(|| Error::internal(format!("{context}: operation succeeded without an error")))
}

fn assert_bind_error(query: &str, detail: &str) -> Result<()> {
    let error = require_error(bind_query(query), query)?;
    assert_eq!(error.code, ErrorCode::QuerySyntax, "{query}: {error:?}");
    assert!(error.message.contains(detail), "{query}: {error:?}");
    Ok(())
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

#[test]
fn skip_and_limit_reject_row_dependent_expressions_during_binding() -> Result<()> {
    for query in [
        "MATCH (n) RETURN n SKIP n.count",
        "MATCH (n) RETURN n LIMIT n.count",
        "MATCH (n) RETURN n SKIP count(*)",
        "MATCH (n) RETURN n LIMIT count(*)",
    ] {
        assert_bind_error(query, "NonConstantExpression")?;
    }
    Ok(())
}

#[test]
fn skip_and_limit_reject_invalid_known_literals_during_binding() -> Result<()> {
    for query in ["RETURN 1 SKIP -1", "RETURN 1 LIMIT -1"] {
        assert_bind_error(query, "NegativeIntegerArgument")?;
    }
    for query in ["RETURN 1 SKIP 1.5", "RETURN 1 LIMIT 1.5"] {
        assert_bind_error(query, "InvalidArgumentType")?;
    }
    Ok(())
}

#[test]
fn parameter_values_are_deferred_to_runtime() -> Result<()> {
    for value in [
        ResultValue::Scalar(ScalarValue::Integer(-1)),
        ResultValue::Scalar(ScalarValue::Float(1.5.into())),
    ] {
        let parameters = BTreeMap::from([("rows".to_owned(), value)]);
        for query in ["RETURN 1 SKIP $rows", "RETURN 1 LIMIT $rows"] {
            bind_parameterized(query, &parameters)?;
        }
    }

    bind_parameterized(
        "RETURN 1 SKIP $rows LIMIT $rows",
        &BTreeMap::from([(
            "rows".to_owned(),
            ResultValue::Scalar(ScalarValue::Integer(0)),
        )]),
    )?;
    bind_query("RETURN 1 SKIP $unknown LIMIT $unknown")?;
    Ok(())
}

#[test]
fn query_engine_classifies_parameter_row_count_errors_at_runtime() -> Result<()> {
    let graph = GraphStore::default();
    for (value, expected_message) in [
        (
            ResultValue::Scalar(ScalarValue::Integer(-1)),
            "NegativeIntegerArgument: SKIP/LIMIT must be non-negative",
        ),
        (
            ResultValue::Scalar(ScalarValue::Float(1.5.into())),
            "InvalidArgumentType: SKIP/LIMIT requires an INTEGER",
        ),
    ] {
        for clause in ["SKIP", "LIMIT"] {
            let mut context = context(&graph);
            context.parameters = BTreeMap::from([("rows".to_owned(), value.clone())]);
            let query = format!("UNWIND [1, 2] AS value RETURN value {clause} $rows");
            let error = require_error(QueryEngine.execute(&query, &mut context), &query)?;
            assert_eq!(error.code, ErrorCode::QuerySyntax, "{query}: {error:?}");
            assert_eq!(error.message, expected_message, "{query}: {error:?}");
        }
    }
    Ok(())
}

#[test]
fn row_independent_integer_expressions_remain_legal() -> Result<()> {
    for query in [
        "MATCH (n) WITH n SKIP toInteger(rand() * 9) RETURN n",
        "MATCH (n) WITH n LIMIT toInteger(ceil(1.7)) RETURN n",
        "RETURN 1 SKIP abs(-1) LIMIT size([1, 2])",
        "RETURN 1 LIMIT 1 + 2",
        "RETURN 1 LIMIT 4 / 2",
    ] {
        bind_query(query)?;
    }
    Ok(())
}

#[test]
fn row_count_arithmetic_uses_executor_result_types() -> Result<()> {
    for query in [
        "RETURN 1 LIMIT 1 + 2",
        "RETURN 1 LIMIT 4 - 2",
        "RETURN 1 LIMIT 2 * 3",
        "RETURN 1 LIMIT 4 / 2",
        "RETURN 1 LIMIT 5 % 2",
    ] {
        bind_query(query)?;
    }

    for query in [
        "RETURN 1 LIMIT 1.0 + 2",
        "RETURN 1 LIMIT 4 - 2.0",
        "RETURN 1 LIMIT 2.0 * 3",
        "RETURN 1 LIMIT 4 / 2.0",
        "RETURN 1 LIMIT 5.0 % 2",
        "RETURN 1 LIMIT 2 ^ 3",
    ] {
        assert_bind_error(query, "InvalidArgumentType")?;
    }
    Ok(())
}

#[test]
fn zero_divisors_remain_runtime_errors() -> Result<()> {
    for query in ["RETURN 1 LIMIT 4 / 0", "RETURN 1 LIMIT 4 % 0"] {
        bind_query(query)?;

        let graph = GraphStore::default();
        let mut context = context(&graph);
        let error = require_error(QueryEngine.execute(query, &mut context), query)?;
        assert_eq!(error.code, ErrorCode::QueryType, "{query}: {error:?}");
        assert!(error.message.contains("zero"), "{query}: {error:?}");
    }
    Ok(())
}
