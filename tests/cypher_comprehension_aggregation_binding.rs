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

fn require_invalid_aggregation(query: &str) -> Result<()> {
    let error = bind_query(query).err().ok_or_else(|| {
        Error::internal(format!("query unexpectedly bound successfully: {query}"))
    })?;
    assert_eq!(error.code, ErrorCode::QuerySyntax, "{query}: {error:?}");
    assert!(
        error.message.contains("InvalidAggregation"),
        "{query}: {error:?}"
    );
    Ok(())
}

fn require_unknown_function(query: &str, name: &str) -> Result<()> {
    let error = bind_query(query)
        .err()
        .ok_or_else(|| Error::internal(format!("unknown function unexpectedly bound: {query}")))?;
    assert_eq!(error.code, ErrorCode::QuerySyntax, "{query}: {error:?}");
    assert!(
        error.message.contains("UnknownFunction"),
        "{query}: {error:?}"
    );
    assert!(error.message.contains(name), "{query}: {error:?}");
    Ok(())
}

#[test]
fn aggregation_is_rejected_inside_element_local_iteration_expressions() -> Result<()> {
    for query in [
        "MATCH (n) RETURN [x IN [1, 2, 3, 4, 5] | count(*)]",
        "MATCH (n) RETURN [x IN [1, 2, 3] WHERE count(*) > 0 | x]",
        "MATCH (n) RETURN reduce(total = 0, x IN [1, 2, 3] | total + count(*))",
        "MATCH (n) RETURN all(x IN [1, 2, 3] WHERE count(*) > 0)",
        "MATCH (n) RETURN any(x IN [1, 2, 3] WHERE sum(x) > 0)",
        "MATCH (n) RETURN none(x IN [1, 2, 3] WHERE avg(x) > 0)",
        "MATCH (n) RETURN single(x IN [1, 2, 3] WHERE collect(x) = [x])",
    ] {
        require_invalid_aggregation(query)?;
    }
    Ok(())
}

#[test]
fn aggregation_remains_legal_before_iteration_and_outside_iteration_scope() -> Result<()> {
    for query in [
        "MATCH (n) RETURN [x IN collect(n) | x] AS nodes",
        "MATCH (n) RETURN [x IN collect(n) WHERE x IS NOT NULL | toString(x)] AS nodes",
        "MATCH (n) RETURN reduce(total = count(*), x IN [1, 2, 3] | total + x) AS total",
        "MATCH (n) RETURN reduce(total = 0, x IN collect(n.value) | total + x) AS total",
        "MATCH (n) RETURN all(x IN collect(n) WHERE x IS NOT NULL) AS present",
        "MATCH (n) RETURN count(*) AS total, [x IN [1, 2, 3] | size([x])] AS sizes",
    ] {
        bind_query(query).map_err(|error| {
            Error::internal(format!(
                "legal aggregation/iteration scope was rejected for `{query}`: {error}"
            ))
        })?;
    }
    Ok(())
}

#[test]
fn known_functions_bind_and_unknown_functions_use_standard_taxonomy() -> Result<()> {
    for query in [
        "MATCH (n)-->(b) WHERE n.name IN [x IN labels(b) | toLower(x)] RETURN b",
        "RETURN ABS(-1) AS value",
        "RETURN duration.between(date('2020-01-01'), date('2020-01-02')) AS value",
    ] {
        bind_query(query).map_err(|error| {
            Error::internal(format!(
                "known built-in was rejected for `{query}`: {error}"
            ))
        })?;
    }

    for (query, name) in [
        ("MATCH (a) RETURN foo(a)", "foo"),
        (
            "RETURN [x IN ['A'] | toLowerUnknown(x)] AS values",
            "tolowerunknown",
        ),
        (
            "RETURN [x IN [1] | vendor.missing(x)] AS values",
            "vendor.missing",
        ),
    ] {
        require_unknown_function(query, name)?;
    }
    Ok(())
}
