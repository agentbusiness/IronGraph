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

fn require_compile_invalid_argument(query: &str) -> Result<()> {
    let parsed = parse(query).map_err(|error| {
        Error::internal(format!(
            "graph function query should reach binding, but parsing `{query}` failed: {error}"
        ))
    })?;
    let error = bind(parsed, &NameCatalog::default(), BindCapabilities::default())
        .err()
        .ok_or_else(|| Error::internal(format!("invalid graph function call bound: {query}")))?;
    assert_eq!(error.code, ErrorCode::QuerySyntax, "{query}: {error:?}");
    assert!(
        error.message.contains("InvalidArgumentType"),
        "{query}: {error:?}"
    );
    Ok(())
}

#[test]
fn known_invalid_graph_function_argument_roles_fail_during_binding() -> Result<()> {
    for query in [
        "MATCH p = (a) RETURN labels(p)",
        "MATCH (r) RETURN type(r)",
        "MATCH (n) RETURN length(n)",
        "MATCH ()-[r]->() RETURN length(r)",
        "RETURN properties(1)",
        "RETURN properties('Cypher')",
        "RETURN properties([true, false])",
    ] {
        require_compile_invalid_argument(query)?;
    }
    Ok(())
}

#[test]
fn static_role_checks_survive_projection_aliases() -> Result<()> {
    for query in [
        "MATCH p = (a) WITH p AS path RETURN labels(path)",
        "MATCH (n) WITH n AS node RETURN type(node)",
        "MATCH ()-[r]->() WITH r AS relationship RETURN length(relationship)",
        "WITH 1 AS value RETURN properties(value)",
    ] {
        require_compile_invalid_argument(query)?;
    }
    Ok(())
}

#[test]
fn legal_graph_arguments_nulls_and_dynamic_values_remain_bindable() -> Result<()> {
    for query in [
        "MATCH p = (a)-[r]->() RETURN labels(a), type(r), length(p), properties(a), properties(r)",
        "RETURN labels(null), type(null), length(null), properties(null), properties({answer: 42})",
        "WITH $value AS value RETURN labels(value), type(value), length(value), properties(value)",
        "MATCH (a) WITH [a, 1] AS list RETURN labels(list[0])",
        "MATCH ()-[r]->() WITH [r, 1] AS list RETURN type(list[0])",
    ] {
        bind_query(query).map_err(|error| {
            Error::internal(format!(
                "legal graph function call was rejected for `{query}`: {error}"
            ))
        })?;
    }
    Ok(())
}

#[test]
fn null_aliases_can_correlate_nullable_read_patterns() -> Result<()> {
    for query in [
        "WITH null AS a OPTIONAL MATCH p = (a)-[r]->() RETURN nodes(p), nodes(null)",
        "WITH null AS a OPTIONAL MATCH p = (a)-[r]->() RETURN relationships(p), relationships(null)",
        "WITH null AS a WITH a AS start OPTIONAL MATCH p = (start)-[r]->() RETURN p, start, r",
        "WITH null AS r OPTIONAL MATCH ()-[r]->() RETURN r",
    ] {
        bind_query(query).map_err(|error| {
            Error::internal(format!(
                "nullable pattern correlation was rejected for `{query}`: {error}"
            ))
        })?;
    }
    Ok(())
}

#[test]
fn non_null_scalar_aliases_still_cannot_masquerade_as_graph_entities() -> Result<()> {
    for query in [
        "WITH 1 AS a OPTIONAL MATCH (a)-[r]->() RETURN a",
        "WITH 'not a relationship' AS r OPTIONAL MATCH ()-[r]->() RETURN r",
    ] {
        let error = bind_query(query)
            .err()
            .ok_or_else(|| Error::internal(format!("scalar pattern correlation bound: {query}")))?;
        assert_eq!(error.code, ErrorCode::QuerySyntax, "{query}: {error:?}");
        assert!(
            error.message.contains("VariableTypeConflict"),
            "{query}: {error:?}"
        );
    }
    Ok(())
}
