// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{collections::BTreeMap, time::Instant};

use irongraph::{
    Bookmark, Error, ProjectId, Result, ScalarValue,
    cypher::{
        BinaryOperator, BindCapabilities, Clause, ExecutionContext, Expression, QueryEngine,
        ResultValue, Statement, UnaryOperator, parse,
    },
    graph::GraphStore,
};
use tokio_util::sync::CancellationToken;

fn return_expression(query: &str) -> Result<Expression> {
    let parsed = parse(query)?;
    let Statement::Query(body) = parsed.statement else {
        return Err(Error::internal("expected a query statement"));
    };
    let Some(Clause::Return(projection)) = body.clauses.first() else {
        return Err(Error::internal("expected a RETURN clause"));
    };
    let Some(item) = projection.items.first() else {
        return Err(Error::internal("expected a projected expression"));
    };
    Ok(item.expression.clone())
}

fn variable(name: &str) -> Expression {
    Expression::Variable(name.to_owned())
}

fn property(variable_name: &str, property_name: &str) -> Expression {
    Expression::Property(Box::new(variable(variable_name)), property_name.to_owned())
}

fn comparison(left: Expression, operation: BinaryOperator, right: Expression) -> Expression {
    Expression::Binary {
        left: Box::new(left),
        operation,
        right: Box::new(right),
    }
}

fn conjunction(left: Expression, right: Expression) -> Expression {
    comparison(left, BinaryOperator::And, right)
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

fn scalar_result(query: &str, column_name: &str) -> Result<ScalarValue> {
    let graph = GraphStore::default();
    let output = QueryEngine.execute(query, &mut context(&graph))?;
    let batch = output
        .result
        .batches
        .first()
        .ok_or_else(|| Error::internal("query returned no batch"))?;
    let column = batch
        .columns
        .iter()
        .find(|column| column.name == column_name)
        .ok_or_else(|| Error::internal(format!("query did not return column {column_name}")))?;
    let value = column
        .values
        .first()
        .ok_or_else(|| Error::internal(format!("column {column_name} returned no value")))?;
    let ResultValue::Scalar(value) = value else {
        return Err(Error::internal(format!(
            "column {column_name} did not return a scalar"
        )));
    };
    Ok(value.clone())
}

#[test]
fn range_chain_reuses_the_middle_operand_for_each_adjacent_comparison() -> Result<()> {
    let actual = return_expression("RETURN 1 < n.num <= 3 AS inside")?;
    let middle = property("n", "num");
    let expected = conjunction(
        comparison(Expression::integer(1), BinaryOperator::Less, middle.clone()),
        comparison(middle, BinaryOperator::LessOrEqual, Expression::integer(3)),
    );
    assert_eq!(actual, expected);
    Ok(())
}

#[test]
fn all_simple_comparison_operators_form_one_adjacent_chain() -> Result<()> {
    let actual = return_expression("RETURN n.prop1 < m.prop1 = n.prop2 <> m.prop2")?;
    let m_prop1 = property("m", "prop1");
    let n_prop2 = property("n", "prop2");
    let expected = conjunction(
        conjunction(
            comparison(
                property("n", "prop1"),
                BinaryOperator::Less,
                m_prop1.clone(),
            ),
            comparison(m_prop1, BinaryOperator::Equal, n_prop2.clone()),
        ),
        comparison(n_prop2, BinaryOperator::NotEqual, property("m", "prop2")),
    );
    assert_eq!(actual, expected);
    Ok(())
}

#[test]
fn parentheses_break_a_comparison_chain_and_logical_precedence_is_preserved() -> Result<()> {
    let parenthesized = return_expression("RETURN (1 < n.num) < 3")?;
    assert!(matches!(
        parenthesized,
        Expression::Binary {
            left,
            operation: BinaryOperator::Less,
            right,
        } if matches!(
            left.as_ref(),
            Expression::Binary {
                operation: BinaryOperator::Less,
                ..
            }
        ) && matches!(
            right.as_ref(),
            Expression::Literal(ScalarValue::Integer(3))
        )
    ));

    let logical = return_expression("RETURN NOT 1 < n.num < 3 OR flag")?;
    assert!(matches!(
        logical,
        Expression::Binary {
            left,
            operation: BinaryOperator::Or,
            right,
        } if matches!(
            left.as_ref(),
            Expression::Unary {
                operation: UnaryOperator::Not,
                operand,
            } if matches!(
                operand.as_ref(),
                Expression::Binary {
                    operation: BinaryOperator::And,
                    ..
                }
            )
        ) && matches!(right.as_ref(), Expression::Variable(name) if name == "flag")
    ));
    Ok(())
}

#[test]
fn cpu_chain_evaluation_preserves_boolean_and_null_truth_values() -> Result<()> {
    let query = "RETURN \
        1 < 2 < 3 AS ascending, \
        1 < 2 > 3 AS outside, \
        1 < null < 3 AS unknown, \
        3 < 2 < null AS false_before_null, \
        'a' <= 'b' <= 'c' AS text_range, \
        1 < 2 = 2 <> 3 AS mixed";
    for (column, expected) in [
        ("ascending", ScalarValue::Boolean(true)),
        ("outside", ScalarValue::Boolean(false)),
        ("unknown", ScalarValue::Null),
        ("false_before_null", ScalarValue::Boolean(false)),
        ("text_range", ScalarValue::Boolean(true)),
        ("mixed", ScalarValue::Boolean(true)),
    ] {
        assert_eq!(scalar_result(query, column)?, expected, "column: {column}");
    }
    Ok(())
}
