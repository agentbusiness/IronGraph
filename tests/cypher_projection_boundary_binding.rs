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

fn require_syntax_error(query: &str, detail: &str) -> Result<()> {
    let error = bind_query(query)
        .err()
        .ok_or_else(|| Error::internal(format!("invalid query bound successfully: {query}")))?;
    assert_eq!(error.code, ErrorCode::QuerySyntax, "{query}: {error:?}");
    assert!(
        error.message.contains(detail),
        "{query}: expected `{detail}` in `{}`",
        error.message
    );
    Ok(())
}

fn require_bindable(query: &str) -> Result<()> {
    bind_query(query).map(|_| ()).map_err(|error| {
        Error::internal(format!("legal query was rejected for `{query}`: {error}"))
    })
}

#[test]
fn projection_columns_are_unique_after_source_names_and_star_expansion() -> Result<()> {
    for query in [
        "RETURN 1 AS a, 2 AS a",
        "WITH 1 AS a, 2 AS a RETURN a",
        "RETURN 1, 1",
        "MATCH (n) RETURN *, n",
        "MATCH (n) WITH *, n RETURN n",
    ] {
        require_syntax_error(query, "ColumnNameConflict")?;
    }

    for query in [
        "RETURN 1 AS a, 2 AS b",
        "RETURN 1, 2",
        "MATCH (a), (b) RETURN *",
        "MATCH (n) WITH *, 1 AS extra RETURN n, extra",
        "MATCH (n) RETURN n, n.name",
    ] {
        require_bindable(query)?;
    }
    Ok(())
}

#[test]
fn with_requires_aliases_only_for_non_variable_expressions() -> Result<()> {
    for query in [
        "MATCH (a) WITH a, count(*) RETURN a",
        "MATCH (a) WITH a.name RETURN a.name",
        "WITH 1 RETURN 1",
        "MATCH (a) WITH [a] RETURN a",
    ] {
        require_syntax_error(query, "NoExpressionAlias")?;
    }

    for query in [
        "MATCH (a) WITH a RETURN a",
        "MATCH (a) WITH * RETURN a",
        "WITH 1 AS one RETURN one",
        "MATCH (a) WITH a, count(*) AS total RETURN a, total",
        "RETURN 1 + 2, count(*)",
    ] {
        require_bindable(query)?;
    }
    Ok(())
}

#[test]
fn aggregate_order_validation_keeps_its_more_specific_error_precedence() -> Result<()> {
    let query = "MATCH (me)--(you) \
                 WITH me.age + you.age, count(*) AS cnt \
                 ORDER BY me.age + you.age + count(*) \
                 RETURN *";
    let error = plan(bind_query(query)?)
        .err()
        .ok_or_else(|| Error::internal("ambiguous aggregate ORDER BY planned successfully"))?;
    assert_eq!(error.code, ErrorCode::QuerySyntax, "{error:?}");
    assert!(
        error.message.contains("AmbiguousAggregationExpression"),
        "{error:?}"
    );
    Ok(())
}

#[test]
fn return_star_requires_a_non_empty_binding_scope() -> Result<()> {
    for query in ["RETURN *", "MATCH () RETURN *"] {
        require_syntax_error(query, "NoVariablesInScope")?;
    }

    for query in ["MATCH (a) RETURN *", "MATCH p = (a)-->(b) RETURN *"] {
        require_bindable(query)?;
    }
    Ok(())
}

#[test]
fn size_rejects_known_paths_without_restricting_its_polymorphic_inputs() -> Result<()> {
    for query in [
        "MATCH p = (a)-[*]->(b) RETURN size(p)",
        "MATCH p = (a)-->(b) WITH p AS path RETURN size(path)",
    ] {
        require_syntax_error(query, "InvalidArgumentType")?;
    }

    for query in [
        "MATCH p = (a)-->(b) RETURN length(p)",
        "RETURN size([1, 2]), size({answer: 42}), size('Cypher'), size(null)",
        "WITH $value AS value RETURN size(value)",
    ] {
        require_bindable(query)?;
    }
    Ok(())
}
