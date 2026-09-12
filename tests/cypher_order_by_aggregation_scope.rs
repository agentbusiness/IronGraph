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

fn require_binding_error(query: &str) -> Result<Error> {
    bind_query(query)
        .err()
        .ok_or_else(|| Error::internal(format!("query unexpectedly bound successfully: {query}")))
}

#[test]
fn aggregating_with_cannot_introduce_a_new_order_by_aggregate() -> Result<()> {
    for query in [
        "MATCH (a) WITH a, a.num + a.num2 AS sum WITH a.num2 % 3 AS mod, min(sum) AS min ORDER BY sum(sum) RETURN mod, min",
        "MATCH (a) WITH a.num2 % 3 AS mod, min(a.num + a.num2) AS min ORDER BY sum(a.num + a.num2) RETURN mod, min",
        "MATCH (a) WITH a.kind AS kind, min(a.value) AS minimum ORDER BY max(a.value) RETURN kind, minimum",
    ] {
        let error = require_binding_error(query)?;
        assert_eq!(error.code, ErrorCode::QuerySyntax, "{query}: {error:?}");
        assert!(
            error.message.contains("UndefinedVariable"),
            "{query}: {error:?}"
        );
    }
    Ok(())
}

#[test]
fn aggregating_with_preserves_legal_order_by_scope() -> Result<()> {
    for query in [
        "MATCH (a) WITH a.num2 % 3 AS mod, sum(a.num + a.num2) AS total ORDER BY sum(a.num + a.num2) RETURN mod, total",
        "MATCH (a) WITH a.num2 % 3 AS mod, sum(a.num + a.num2) AS total ORDER BY total RETURN mod, total",
        "MATCH (a) WITH a.num2 % 3 AS mod, count(*) AS total ORDER BY a.num2 % 3, count(*) RETURN mod, total",
        "MATCH (a) WITH a.name AS name ORDER BY a.age RETURN name",
    ] {
        bind_query(query).map_err(|error| {
            Error::internal(format!(
                "legal ORDER BY scope was rejected for `{query}`: {error}"
            ))
        })?;
    }
    Ok(())
}
