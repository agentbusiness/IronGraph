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

fn require_bind_error(query: &str, code: ErrorCode, detail: &str) -> Result<()> {
    let error = bind_query(query)
        .err()
        .ok_or_else(|| Error::internal(format!("invalid expression bound: {query}")))?;
    assert_eq!(error.code, code, "{query}: {error:?}");
    assert!(error.message.contains(detail), "{query}: {error:?}");
    Ok(())
}

#[test]
fn literal_aliases_reject_property_access_at_compile_time() -> Result<()> {
    for alias in ["nonMap", "nonGraphElement"] {
        for expression in ["123", "42.45", "true", "false", "'string'", "[123, true]"] {
            let query = format!("WITH {expression} AS {alias} RETURN {alias}.num");
            require_bind_error(&query, ErrorCode::QueryType, "InvalidArgumentType")?;
        }
    }
    Ok(())
}

#[test]
fn path_bindings_reject_property_access_with_the_tck_syntax_taxonomy() -> Result<()> {
    require_bind_error(
        "MATCH (n) MATCH r = (n)-[*]->() WHERE r.name = 'apa' RETURN r",
        ErrorCode::QuerySyntax,
        "InvalidArgumentType",
    )
}

#[test]
fn properties_rejects_only_definitely_invalid_literal_sources() -> Result<()> {
    for query in [
        "RETURN properties(1)",
        "RETURN properties('Cypher')",
        "RETURN properties([true, false])",
    ] {
        require_bind_error(query, ErrorCode::QuerySyntax, "InvalidArgumentType")?;
    }
    Ok(())
}

#[test]
fn property_compatible_and_runtime_resolved_sources_remain_legal() -> Result<()> {
    for query in [
        "WITH {existing: 42, notMissing: null} AS m RETURN m.missing, m.existing",
        "WITH null AS m RETURN m.missing",
        "MATCH (n) RETURN n.missing",
        "MATCH ()-[r]->() RETURN r.missing",
        "MATCH (n) WITH [123, n] AS list RETURN (list[1]).missing",
        "WITH [123, {existing: 42}] AS list RETURN (list[1]).existing",
        "WITH $value AS source RETURN source.missing",
        "WITH CASE WHEN $flag THEN 1 ELSE {missing: 42} END AS source RETURN source.missing",
        "WITH CASE WHEN $flag THEN 1 END AS source RETURN source.missing",
        "WITH [123, {missing: 42}] AS list RETURN (list[$index]).missing",
        "WITH [x IN $values | x] AS source RETURN source.missing",
        "WITH 1 AS x RETURN [x IN $values | x.missing] AS projected",
        "RETURN properties($value)",
        "RETURN properties(null), properties({existing: 42})",
        "RETURN properties({existing: 42}).existing",
        "MATCH (n) RETURN properties(n)",
        "MATCH ()-[r]->() RETURN properties(r)",
        "MATCH ()-[r]->() RETURN startNode(r).missing, endNode(r).missing",
        "RETURN date('1984-10-11').year",
        "RETURN datetime('1984-10-11T12:31:14Z').year",
        "RETURN duration('P1M').months",
    ] {
        bind_query(query).map_err(|error| {
            Error::internal(format!(
                "potentially valid property source was rejected for `{query}`: {error}"
            ))
        })?;
    }
    Ok(())
}

#[test]
fn source_shape_survives_value_aliases_without_guessing_through_indexing() -> Result<()> {
    require_bind_error(
        "WITH 1 AS original WITH original AS copied RETURN copied.missing",
        ErrorCode::QueryType,
        "InvalidArgumentType",
    )?;
    bind_query(
        "WITH [1, {missing: 42}] AS values WITH values[0] AS selected RETURN selected.missing",
    )
    .map_err(|error| {
        Error::internal(format!(
            "indexed heterogeneous value should remain runtime-resolved: {error}"
        ))
    })?;
    Ok(())
}

#[test]
fn nested_aggregates_use_the_tck_syntax_taxonomy_without_rejecting_legal_aggregation() -> Result<()>
{
    for query in ["RETURN count(count(*))", "RETURN sum(count(*))"] {
        require_bind_error(query, ErrorCode::QuerySyntax, "NestedAggregation")?;
    }
    for query in [
        "RETURN count(*) AS total",
        "RETURN count(*) + count(*) AS total",
        "MATCH (n) RETURN count(n.missing) AS total",
        "WITH count(*) AS total RETURN sum(total)",
    ] {
        bind_query(query).map_err(|error| {
            Error::internal(format!(
                "legal aggregation was rejected for `{query}`: {error}"
            ))
        })?;
    }
    Ok(())
}
