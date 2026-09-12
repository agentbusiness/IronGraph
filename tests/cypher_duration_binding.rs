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

#[test]
fn duration_component_functions_bind_with_canonical_names() -> Result<()> {
    for query in [
        "RETURN duration.between(date('1984-10-11'), date('2015-06-24'))",
        "RETURN duration.inMonths(duration({months: 1}))",
        "RETURN duration.inDays(duration({days: 1}))",
        "RETURN duration.inSeconds(duration({seconds: 1}))",
    ] {
        bind_query(query).map_err(|error| {
            Error::internal(format!(
                "registered duration function failed binding for `{query}`: {error}"
            ))
        })?;
    }
    Ok(())
}

#[test]
fn unknown_duration_component_function_remains_rejected() -> Result<()> {
    let query = "RETURN duration.inWeeks(duration({weeks: 1}))";
    let error = bind_query(query)
        .err()
        .ok_or_else(|| Error::internal("unknown duration function passed binding"))?;
    assert_eq!(error.code, ErrorCode::QuerySyntax);
    assert_eq!(
        error.message,
        "UnknownFunction: function `duration.inweeks` is not defined"
    );
    Ok(())
}
