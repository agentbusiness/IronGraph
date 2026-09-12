// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use irongraph::{
    Error, ErrorCode, Result,
    cypher::{BindCapabilities, BoundQuery, PhysicalPlan, bind, parse, plan},
    graph::NameCatalog,
};

fn plan_query(query: &str) -> Result<PhysicalPlan> {
    let bound = bind(
        parse(query)?,
        &NameCatalog::default(),
        BindCapabilities::default(),
    )?;
    plan(bound)
}

fn plan_without_binding(query: &str) -> Result<PhysicalPlan> {
    plan(BoundQuery {
        query: parse(query)?,
        read_only: true,
        dependencies: Vec::new(),
    })
}

fn planning_error(query: &str) -> Result<Error> {
    plan_query(query)
        .err()
        .ok_or_else(|| Error::internal(format!("query unexpectedly planned successfully: {query}")))
}

#[test]
fn union_and_union_all_reject_different_client_columns_during_planning() -> Result<()> {
    for query in [
        "RETURN 1 AS a UNION RETURN 2 AS b",
        "RETURN 1 AS a UNION ALL RETURN 2 AS b",
        "RETURN 1 AS a, 2 AS b UNION RETURN 3 AS b, 4 AS a",
        "RETURN 1 AS a UNION RETURN 2 AS a UNION RETURN 3 AS b",
    ] {
        let error = planning_error(query)?;
        assert_eq!(error.code, ErrorCode::QuerySyntax, "{query}: {error:?}");
        assert!(
            error.message.contains("DifferentColumnsInUnion"),
            "{query}: {error:?}"
        );
    }
    Ok(())
}

#[test]
fn union_preserves_same_named_columns_across_different_expressions_and_types() -> Result<()> {
    for query in [
        "RETURN 1 AS value UNION RETURN 'two' AS value",
        "RETURN 1 AS value UNION ALL RETURN [2, 3] AS value",
        "UNWIND [1] AS x RETURN x UNION UNWIND ['two'] AS x RETURN x",
        "RETURN 1 AS a, 'left' AS b UNION RETURN false AS a, [2] AS b",
    ] {
        plan_query(query).map_err(|error| {
            Error::internal(format!(
                "legal UNION schema was rejected for `{query}`: {error}"
            ))
        })?;
    }
    Ok(())
}

#[test]
fn union_schema_uses_terminal_return_before_order_skip_and_limit() -> Result<()> {
    plan_query(
        "RETURN 2 AS value ORDER BY value SKIP 0 LIMIT 1 \
         UNION ALL \
         RETURN 'two' AS value ORDER BY value SKIP 0 LIMIT 1",
    )?;

    let query = "RETURN 1 AS a ORDER BY a LIMIT 1 UNION RETURN 2 AS b ORDER BY b SKIP 0 LIMIT 1";
    let error = planning_error(query)?;
    assert_eq!(error.code, ErrorCode::QuerySyntax, "{error:?}");
    assert!(
        error.message.contains("DifferentColumnsInUnion"),
        "{error:?}"
    );
    Ok(())
}

#[test]
fn union_star_projection_fails_closed_without_an_exact_expanded_schema() -> Result<()> {
    for query in [
        "MATCH (a) RETURN * UNION MATCH (b) RETURN *",
        "UNWIND [1] AS x RETURN * UNION ALL UNWIND [2] AS x RETURN *",
    ] {
        let error = planning_error(query)?;
        assert_eq!(error.code, ErrorCode::QuerySyntax, "{query}: {error:?}");
        assert!(
            error.message.contains("UnsupportedUnionStarProjection"),
            "{query}: {error:?}"
        );
    }
    Ok(())
}

#[test]
fn union_schema_comparison_preserves_duplicate_projection_positions() -> Result<()> {
    // The binder owns duplicate-name rejection. Bypass it here so this planner regression proves
    // that schema comparison itself does not collapse two positional columns into one.
    let query = "RETURN 1 AS a, 2 AS a UNION RETURN 3 AS a";
    let error = plan_without_binding(query)
        .err()
        .ok_or_else(|| Error::internal("planner collapsed duplicate UNION column positions"))?;
    assert_eq!(error.code, ErrorCode::QuerySyntax, "{error:?}");
    assert!(
        error.message.contains("DifferentColumnsInUnion"),
        "{error:?}"
    );
    Ok(())
}
