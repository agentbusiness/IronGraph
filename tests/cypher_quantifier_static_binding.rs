// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use irongraph::{
    Error, ErrorCode, Result,
    cypher::{BindCapabilities, BoundQuery, bind, parse},
    graph::NameCatalog,
};

fn bind_query(query: &str) -> Result<BoundQuery> {
    bind(
        parse(query)?,
        &NameCatalog::default(),
        BindCapabilities::default(),
    )
}

fn require_invalid_argument_type(query: &str) -> Result<()> {
    let error = bind_query(query)
        .err()
        .ok_or_else(|| Error::internal(format!("invalid predicate bound: {query}")))?;
    assert_eq!(error.code, ErrorCode::QuerySyntax, "{query}: {error:?}");
    assert!(
        error.message.contains("InvalidArgumentType"),
        "{query}: {error:?}"
    );
    Ok(())
}

#[test]
fn homogeneous_nonnumeric_literal_elements_reject_numeric_predicates() -> Result<()> {
    for quantifier in ["none", "single", "any", "all"] {
        for list in [
            "['Clara']",
            "[false, true]",
            "['Clara', 'Bob', 'Dave', 'Alice']",
        ] {
            let query = format!("RETURN {quantifier}(x IN {list} WHERE x % 2 = 0) AS result");
            require_invalid_argument_type(&query)?;
        }
    }
    Ok(())
}

#[test]
fn valid_homogeneous_predicates_and_null_semantics_remain_legal() -> Result<()> {
    for query in [
        "RETURN none(x IN [] WHERE x % 2 = 0) AS result",
        "RETURN single(x IN [true, false] WHERE x) AS result",
        "RETURN any(x IN [1, 2, 3] WHERE x % 2 = 0) AS result",
        "RETURN all(x IN [1.5, 2.5] WHERE x % 2 = 0) AS result",
        "RETURN any(x IN ['abc', 'ef'] WHERE size(x) = 3) AS result",
        "RETURN all(x IN [{a: 2}, {a: 4}] WHERE x.a = 2) AS result",
        "RETURN none(x IN [null] WHERE x % 2 = 0) AS result",
        "RETURN any(x IN [1, null, true, 4.5, 'abc', false] WHERE null) AS result",
    ] {
        bind_query(query).map_err(|error| {
            Error::internal(format!(
                "legal quantified predicate was rejected for `{query}`: {error}"
            ))
        })?;
    }
    Ok(())
}

#[test]
fn mixed_parameterized_and_unknown_element_types_remain_deferred() -> Result<()> {
    for query in [
        "RETURN any(x IN [1, 'two'] WHERE x % 2 = 0) AS result",
        "RETURN any(x IN [true, 1] WHERE x % 2 = 0) AS result",
        "RETURN any(x IN [null, 'two'] WHERE x % 2 = 0) AS result",
        "RETURN any(x IN [1, $value] WHERE x % 2 = 0) AS result",
        "RETURN any(x IN $values WHERE x % 2 = 0) AS result",
        "RETURN any(x IN ['two'] WHERE x % $divisor = 0) AS result",
    ] {
        bind_query(query).map_err(|error| {
            Error::internal(format!(
                "unresolved quantified predicate was rejected for `{query}`: {error}"
            ))
        })?;
    }
    Ok(())
}
