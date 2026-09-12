// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use irongraph::{
    Error, Result, ScalarValue,
    cypher::{BinaryOperator, Clause, Expression, Statement, UnaryOperator, parse},
};

fn return_expressions(query: &str) -> Result<Vec<Expression>> {
    let parsed = parse(query)?;
    let Statement::Query(body) = parsed.statement else {
        return Err(Error::internal("expected a query statement"));
    };
    let Some(Clause::Return(projection)) = body.clauses.first() else {
        return Err(Error::internal("expected a RETURN clause"));
    };
    Ok(projection
        .items
        .iter()
        .map(|item| item.expression.clone())
        .collect())
}

fn is_integer(expression: &Expression, expected: i64) -> bool {
    matches!(
        expression,
        Expression::Literal(ScalarValue::Integer(actual)) if *actual == expected
    )
}

#[test]
fn exponentiation_is_left_associative_but_parentheses_remain_authoritative() -> Result<()> {
    let expressions = return_expressions(
        "RETURN 4 ^ (3 * 2) ^ 3 AS left_grouped, 2 ^ (3 ^ 4) AS explicit_right",
    )?;
    let [left_grouped, explicit_right] = expressions.as_slice() else {
        return Err(Error::internal("expected two projected expressions"));
    };

    let Expression::Binary {
        left,
        operation: BinaryOperator::Power,
        right,
    } = left_grouped
    else {
        return Err(Error::internal("expected outer left-associative power"));
    };
    assert!(is_integer(right, 3));
    assert!(matches!(
        left.as_ref(),
        Expression::Binary {
            left: base,
            operation: BinaryOperator::Power,
            right: grouped,
        } if is_integer(base, 4)
            && matches!(
                grouped.as_ref(),
                Expression::Binary {
                    operation: BinaryOperator::Multiply,
                    ..
                }
            )
    ));

    assert!(matches!(
        explicit_right,
        Expression::Binary {
            left,
            operation: BinaryOperator::Power,
            right,
        } if is_integer(left, 2)
            && matches!(
                right.as_ref(),
                Expression::Binary {
                    operation: BinaryOperator::Power,
                    ..
                }
            )
    ));
    Ok(())
}

#[test]
fn unary_negative_still_binds_more_tightly_than_exponentiation() -> Result<()> {
    let expressions = return_expressions(
        "RETURN -3 ^ 2 AS implicit, (-3) ^ 2 AS grouped, -(3 ^ 2) AS explicit_outer",
    )?;
    let [implicit, grouped, explicit_outer] = expressions.as_slice() else {
        return Err(Error::internal("expected three projected expressions"));
    };

    for expression in [implicit, grouped] {
        assert!(matches!(
            expression,
            Expression::Binary {
                left,
                operation: BinaryOperator::Power,
                right,
            } if matches!(
                left.as_ref(),
                Expression::Unary {
                    operation: UnaryOperator::Negative,
                    operand,
                } if is_integer(operand, 3)
            ) && is_integer(right, 2)
        ));
    }
    assert!(matches!(
        explicit_outer,
        Expression::Unary {
            operation: UnaryOperator::Negative,
            operand,
        } if matches!(
            operand.as_ref(),
            Expression::Binary {
                operation: BinaryOperator::Power,
                ..
            }
        )
    ));
    Ok(())
}

#[test]
fn exponentiation_still_binds_more_tightly_than_multiplication_and_addition() -> Result<()> {
    let expressions = return_expressions("RETURN 4 ^ 3 * 2 ^ 3 AS product, 4 ^ 3 + 2 ^ 3 AS sum")?;
    let [product, sum] = expressions.as_slice() else {
        return Err(Error::internal("expected two projected expressions"));
    };

    for (expression, expected) in [
        (product, BinaryOperator::Multiply),
        (sum, BinaryOperator::Add),
    ] {
        assert!(matches!(
            expression,
            Expression::Binary {
                left,
                operation,
                right,
            } if *operation == expected
                && matches!(
                    left.as_ref(),
                    Expression::Binary {
                        operation: BinaryOperator::Power,
                        ..
                    }
                )
                && matches!(
                    right.as_ref(),
                    Expression::Binary {
                        operation: BinaryOperator::Power,
                        ..
                    }
                )
        ));
    }
    Ok(())
}
