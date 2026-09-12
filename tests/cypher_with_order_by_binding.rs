// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use irongraph::{
    Error, ErrorCode, Result,
    cypher::{BindCapabilities, BoundQuery, bind, parse, plan},
    graph::NameCatalog,
};

fn bind_query(query: &str) -> Result<BoundQuery> {
    bind(
        parse(query)?,
        &NameCatalog::default(),
        BindCapabilities::default(),
    )
}

fn require_error<T>(result: Result<T>, context: &str) -> Result<Error> {
    result
        .err()
        .ok_or_else(|| Error::internal(format!("{context}: query succeeded without an error")))
}

#[test]
fn with_order_by_resolves_projection_inputs_and_aliases_before_pruning() -> Result<()> {
    for query in [
        "MATCH (a) WITH a.name AS name ORDER BY a.name + 'C' RETURN name",
        "MATCH (a) WITH DISTINCT a.name AS name ORDER BY a.name RETURN name",
        "MATCH (a) WITH a, a.num AS sum WITH a AS kept ORDER BY sum RETURN kept",
        "MATCH (a) WITH a.num AS num, count(*) AS cnt ORDER BY a.num + count(*) RETURN num",
        "MATCH (a) WITH a.num AS num ORDER BY num RETURN num",
    ] {
        bind_query(query).map_err(|error| {
            Error::internal(format!(
                "legal WITH ORDER BY binding `{query}` failed: {error}"
            ))
        })?;
    }
    Ok(())
}

#[test]
fn with_order_by_leaves_aggregation_classification_to_projection_validation() -> Result<()> {
    for query in [
        "MATCH (n) WITH n.num1 AS foo ORDER BY count(n) RETURN foo",
        "MATCH (n) WITH n.num1 AS foo ORDER BY max(n.num2), n.name RETURN foo",
    ] {
        let error = require_error(plan(bind_query(query)?), query)?;
        assert_eq!(error.code, ErrorCode::QuerySyntax, "{query}: {error:?}");
        assert!(
            error.message.contains("InvalidAggregation"),
            "{query}: {error:?}"
        );
    }

    let query = "MATCH (me)--(you) WITH me.age + you.age, count(*) AS cnt ORDER BY me.age + you.age + count(*) RETURN *";
    let error = require_error(plan(bind_query(query)?), query)?;
    assert_eq!(error.code, ErrorCode::QuerySyntax, "{error:?}");
    assert!(error.message.contains("AmbiguousAggregationExpression"));
    Ok(())
}

#[test]
fn aggregating_with_rejects_an_unprojected_order_by_aggregate() -> Result<()> {
    for query in [
        "MATCH (a) WITH a, a.num + a.num2 AS sum WITH a.num2 % 3 AS mod, min(sum) AS min ORDER BY sum(sum) RETURN mod, min",
        "MATCH (a) WITH a.num2 % 3 AS mod, min(a.num + a.num2) AS min ORDER BY sum(a.num + a.num2) RETURN mod, min",
    ] {
        let error = require_error(bind_query(query), query)?;
        assert_eq!(error.code, ErrorCode::QuerySyntax, "{query}: {error:?}");
        assert!(
            error.message.contains("UndefinedVariable"),
            "{query}: {error:?}"
        );
    }
    Ok(())
}

#[test]
fn with_order_by_does_not_leak_its_input_scope() -> Result<()> {
    let query = "MATCH (a) WITH a.name AS name ORDER BY a.age RETURN a";
    let error = require_error(bind_query(query), query)?;
    assert_eq!(error.code, ErrorCode::QuerySyntax);
    assert!(error.message.contains("UndefinedVariable"));

    let query = "MATCH (a) WITH a.name AS name ORDER BY missing RETURN name";
    let error = require_error(bind_query(query), query)?;
    assert_eq!(error.code, ErrorCode::QuerySyntax);
    assert!(error.message.contains("UndefinedVariable"));
    Ok(())
}

#[test]
fn projected_alias_kind_shadows_the_incoming_binding_during_and_after_order_by() -> Result<()> {
    let query = "MATCH (n) WITH n.value AS n ORDER BY n MATCH (n) RETURN n";
    let error = require_error(bind_query(query), query)?;
    assert_eq!(error.code, ErrorCode::QuerySyntax);
    assert!(error.message.contains("VariableTypeConflict"));
    Ok(())
}
