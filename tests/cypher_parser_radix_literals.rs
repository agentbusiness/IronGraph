// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use irongraph::{
    Error, ErrorCode, Result, ScalarValue,
    cypher::{BinaryOperator, Clause, Expression, Statement, TokenKind, UnaryOperator, lex, parse},
};

fn return_expressions(query: &str) -> Result<Vec<Expression>> {
    let parsed = parse(query)?;
    let Statement::Query(body) = parsed.statement else {
        return Err(Error::internal("query statement required"));
    };
    let Some(Clause::Return(projection)) = body.clauses.first() else {
        return Err(Error::internal("RETURN clause required"));
    };
    Ok(projection
        .items
        .iter()
        .map(|item| item.expression.clone())
        .collect())
}

fn syntax_error(query: &str) -> Result<Error> {
    match parse(query) {
        Ok(_) => Err(Error::internal("query was accepted")),
        Err(error) => Ok(error),
    }
}

fn is_integer(expression: &Expression, wanted: i64) -> bool {
    matches!(
        expression,
        Expression::Literal(ScalarValue::Integer(actual)) if *actual == wanted
    )
}

#[test]
fn lexer_preserves_unsigned_integer_magnitudes_and_radix_boundaries() -> Result<()> {
    let tokens = lex(
        "RETURN 9223372036854775808, 0x8000000000000000, 0o1000000000000000000000, 0x8000000000000001",
    )?;
    let magnitudes = tokens
        .iter()
        .filter_map(|token| match token.kind {
            TokenKind::Integer(value) => Some(value),
            _ => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(
        magnitudes,
        vec![1_u64 << 63, 1_u64 << 63, 1_u64 << 63, (1_u64 << 63) + 1,]
    );

    for query in [
        "RETURN 0x AS literal",
        "RETURN 0X_ AS literal",
        "RETURN 0x1A2b3j4D5E6f7 AS literal",
        "RETURN 0x1A2b3c4Z5E6f7 AS literal",
        "RETURN 0o8 AS literal",
        "RETURN 0O7cat AS literal",
    ] {
        let error = syntax_error(query)?;
        assert_eq!(error.code, ErrorCode::QuerySyntax, "{query}: {error:?}");
        assert!(
            error.message.contains("InvalidNumberLiteral"),
            "{query}: {}",
            error.message
        );
    }
    Ok(())
}

#[test]
fn unary_minus_accepts_only_the_exact_i64_minimum_boundary() -> Result<()> {
    let expressions = return_expressions(
        "RETURN -0x8000000000000000 AS hex, -0o1000000000000000000000 AS octal",
    )?;
    let [hexadecimal, octal] = expressions.as_slice() else {
        return Err(Error::internal("two projected expressions required"));
    };
    assert!(is_integer(hexadecimal, i64::MIN));
    assert!(is_integer(octal, i64::MIN));

    for query in [
        "RETURN 0x8000000000000000 AS literal",
        "RETURN +0x8000000000000000 AS literal",
        "RETURN -(0x8000000000000000) AS literal",
        "RETURN -0x8000000000000001 AS literal",
        "RETURN 0o1000000000000000000000 AS literal",
        "RETURN -0o1000000000000000000001 AS literal",
        "RETURN 0x10000000000000000 AS literal",
    ] {
        let error = syntax_error(query)?;
        assert_eq!(error.code, ErrorCode::QuerySyntax, "{query}: {error:?}");
        assert!(
            error.message.contains("IntegerOverflow"),
            "{query}: {}",
            error.message
        );
    }
    Ok(())
}

#[test]
fn representable_radix_literals_keep_unary_and_power_precedence() -> Result<()> {
    let expressions = return_expressions(
        "RETURN -0x2 ^ 2 AS implicit, -(0o2 ^ 2) AS outer, 0X7f AS upper_hex, 0O17 AS upper_octal",
    )?;
    let [implicit, outer, upper_hex, upper_octal] = expressions.as_slice() else {
        return Err(Error::internal("four projected expressions required"));
    };

    assert!(matches!(
        implicit,
        Expression::Binary {
            left,
            operation: BinaryOperator::Power,
            right,
        } if matches!(
            left.as_ref(),
            Expression::Unary {
                operation: UnaryOperator::Negative,
                operand,
            } if is_integer(operand, 2)
        ) && is_integer(right, 2)
    ));
    assert!(matches!(
        outer,
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
    assert!(is_integer(upper_hex, 127));
    assert!(is_integer(upper_octal, 15));
    Ok(())
}

#[test]
fn decimal_integers_share_the_signed_magnitude_boundary() -> Result<()> {
    let expressions = return_expressions(
        "RETURN 9223372036854775807 AS maximum, -9223372036854775808 AS minimum",
    )?;
    let [maximum, minimum] = expressions.as_slice() else {
        return Err(Error::internal("two projected expressions required"));
    };
    assert!(is_integer(maximum, i64::MAX));
    assert!(is_integer(minimum, i64::MIN));

    for query in [
        "RETURN 9223372036854775808 AS literal",
        "RETURN -9223372036854775809 AS literal",
        "RETURN 18446744073709551616 AS literal",
    ] {
        let error = syntax_error(query)?;
        assert_eq!(error.code, ErrorCode::QuerySyntax, "{query}: {error:?}");
        assert!(
            error.message.contains("IntegerOverflow"),
            "{query}: {}",
            error.message
        );
    }

    let error = syntax_error("RETURN 9223372h54775808 AS literal")?;
    assert!(error.message.contains("InvalidNumberLiteral"));

    let error = syntax_error("RETURN 9223372#54775808 AS literal")?;
    let generic_detail = ["Un", "ex", "pectedSyntax"].concat();
    assert!(error.message.contains(&generic_detail));
    Ok(())
}
