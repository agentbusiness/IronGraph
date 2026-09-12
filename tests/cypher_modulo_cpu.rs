// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{collections::BTreeMap, time::Instant};

use irongraph::{
    Bookmark, ErrorCode, ProjectId, Result, ScalarValue,
    cypher::{BindCapabilities, ExecutionContext, QueryEngine, ResultValue},
    graph::GraphStore,
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

fn row(
    query: &str,
    parameters: BTreeMap<String, ResultValue>,
) -> Result<BTreeMap<String, ResultValue>> {
    let graph = GraphStore::default();
    let mut context = context(&graph);
    context.parameters = parameters;
    let output = QueryEngine.execute(query, &mut context)?;
    let batch = output
        .result
        .batches
        .first()
        .ok_or_else(|| irongraph::Error::internal("query returned no batch"))?;
    batch
        .columns
        .iter()
        .map(|column| {
            column
                .values
                .first()
                .cloned()
                .map(|value| (column.name.clone(), value))
                .ok_or_else(|| irongraph::Error::internal("query returned an empty column"))
        })
        .collect()
}

fn float(values: &BTreeMap<String, ResultValue>, name: &str) -> Result<f64> {
    let Some(ResultValue::Scalar(ScalarValue::Float(value))) = values.get(name) else {
        return Err(irongraph::Error::internal(format!(
            "column {name} did not return FLOAT"
        )));
    };
    Ok(value.into_inner())
}

#[test]
fn integer_division_preserves_integer_precision_and_truncates_toward_zero() -> Result<()> {
    let values = row(
        "RETURN 16000000000000001 / 16000 AS large, 7 / 3 AS pp, -7 / 3 AS np, 7 / -3 AS pn, -7 / -3 AS nn, 4 / 5 AS fraction",
        BTreeMap::new(),
    )?;
    let expected = [
        ("large", 1_000_000_000_000),
        ("pp", 2),
        ("np", -2),
        ("pn", -2),
        ("nn", 2),
        ("fraction", 0),
    ];
    for (name, value) in expected {
        assert_eq!(
            values.get(name),
            Some(&ResultValue::Scalar(ScalarValue::Integer(value))),
            "column: {name}"
        );
    }
    Ok(())
}

#[test]
fn integer_division_rejects_unrepresentable_minimum_quotient() -> Result<()> {
    let graph = GraphStore::default();
    let mut context = context(&graph);
    context.parameters = BTreeMap::from([(
        "min".to_owned(),
        ResultValue::Scalar(ScalarValue::Integer(i64::MIN)),
    )]);
    let error = QueryEngine
        .execute("RETURN $min / -1 AS quotient", &mut context)
        .err()
        .ok_or_else(|| irongraph::Error::internal("overflowing integer division succeeded"))?;
    assert_eq!(error.code, ErrorCode::QueryType);
    assert!(error.message.contains("integer arithmetic overflow"));
    Ok(())
}

#[test]
fn mixed_numeric_division_returns_float_and_obeys_floating_semantics() -> Result<()> {
    let values = row(
        "RETURN 7 / 2.0 AS left_integer, 7.0 / 2 AS right_integer, 1.0 / 0 AS promoted_zero, 1 / 0.0 AS positive_zero, -1 / 0.0 AS negative_zero, 0.0 / 0.0 AS nan",
        BTreeMap::new(),
    )?;
    assert_eq!(float(&values, "left_integer")?, 3.5);
    assert_eq!(float(&values, "right_integer")?, 3.5);
    assert_eq!(float(&values, "promoted_zero")?, f64::INFINITY);
    assert_eq!(float(&values, "positive_zero")?, f64::INFINITY);
    assert_eq!(float(&values, "negative_zero")?, f64::NEG_INFINITY);
    assert!(float(&values, "nan")?.is_nan());
    Ok(())
}

#[test]
fn integer_division_by_zero_is_a_query_error() -> Result<()> {
    let graph = GraphStore::default();
    let error = QueryEngine
        .execute("RETURN 1 / 0", &mut context(&graph))
        .err()
        .ok_or_else(|| irongraph::Error::internal("integer division by zero succeeded"))?;
    assert_eq!(error.code, ErrorCode::QueryType);
    assert!(error.message.contains("division by zero"));
    Ok(())
}

#[test]
fn integer_modulo_preserves_integer_precision_and_dividend_sign() -> Result<()> {
    let values = row(
        "RETURN 16000000000000001 % 16000 AS large, 7 % 3 AS pp, -7 % 3 AS np, 7 % -3 AS pn, -7 % -3 AS nn",
        BTreeMap::new(),
    )?;
    let expected = [("large", 1), ("pp", 1), ("np", -1), ("pn", 1), ("nn", -1)];
    for (name, value) in expected {
        assert_eq!(
            values.get(name),
            Some(&ResultValue::Scalar(ScalarValue::Integer(value))),
            "column: {name}"
        );
    }
    Ok(())
}

#[test]
fn integer_modulo_handles_minimum_integer_without_overflow() -> Result<()> {
    let parameters = BTreeMap::from([(
        "min".to_owned(),
        ResultValue::Scalar(ScalarValue::Integer(i64::MIN)),
    )]);
    let values = row("RETURN $min % -1 AS remainder", parameters)?;
    assert_eq!(
        values.get("remainder"),
        Some(&ResultValue::Scalar(ScalarValue::Integer(0)))
    );
    Ok(())
}

#[test]
fn mixed_numeric_modulo_returns_float_and_obeys_floating_semantics() -> Result<()> {
    let values = row(
        "RETURN 16000000000000001 % 16000.0 AS coerced, 7 % 2.5 AS left_integer, 7.5 % 2 AS right_integer, 1.0 % 0 AS zero",
        BTreeMap::new(),
    )?;
    for (name, expected) in [
        ("coerced", 0.0),
        ("left_integer", 2.0),
        ("right_integer", 1.5),
    ] {
        assert_eq!(float(&values, name)?, expected, "column: {name}");
    }
    assert!(float(&values, "zero")?.is_nan());
    Ok(())
}

#[test]
fn integer_modulo_by_zero_is_a_query_error() -> Result<()> {
    let graph = GraphStore::default();
    let error = QueryEngine
        .execute("RETURN 1 % 0", &mut context(&graph))
        .err()
        .ok_or_else(|| irongraph::Error::internal("integer modulo by zero succeeded"))?;
    assert_eq!(error.code, ErrorCode::QueryType);
    assert!(error.message.contains("modulo by zero"));
    Ok(())
}
