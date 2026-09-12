// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::collections::BTreeMap;

use irongraph::{
    Error, ErrorCode, Result, ScalarValue,
    cypher::{
        BindCapabilities, BoundQuery, ProcedureCatalog, ProcedureDefinition, ProcedureField,
        ProcedureValueType, ResultValue, bind, bind_with_procedures, parse,
    },
    graph::NameCatalog,
};

fn bind_query(query: &str) -> Result<BoundQuery> {
    bind(
        parse(query)?,
        &NameCatalog::default(),
        BindCapabilities::default(),
    )
}

fn assert_bind_error(query: &str, detail: &str) -> Result<()> {
    let error = bind_query(query)
        .err()
        .ok_or_else(|| Error::internal(format!("`{query}` passed binding without an error")))?;
    assert_eq!(error.code, ErrorCode::QuerySyntax, "{query}: {error:?}");
    assert!(
        error.message.contains(detail),
        "{query}: expected `{detail}` in `{}`",
        error.message
    );
    Ok(())
}

#[test]
fn rejects_entity_role_changes_within_and_across_match_clauses() -> Result<()> {
    for query in [
        "MATCH ()-[r]-() MATCH (r) RETURN r",
        "MATCH p = ()-[]-() MATCH (p) RETURN p",
        "MATCH (r), ()-[r]-() RETURN r",
        "MATCH ()-[r]-(), (r) RETURN r",
        "MATCH (r) MATCH ()-[r]-() RETURN r",
        "MATCH p = ()-[]-() MATCH ()-[p]-() RETURN p",
        "WITH true AS n MATCH (n) RETURN n",
        "WITH [1] AS r MATCH ()-[r]-() RETURN r",
    ] {
        assert_bind_error(query, "VariableTypeConflict")?;
    }
    Ok(())
}

#[test]
fn named_paths_require_a_fresh_variable() -> Result<()> {
    for query in [
        "MATCH (p) MATCH p = ()-[]-() RETURN p",
        "MATCH ()-[p]-(), p = ()-[]-() RETURN p",
        "MATCH p = ()-[]-(), p = ()-[]-() RETURN p",
        "WITH 1 AS p MATCH p = ()-[]-() RETURN p",
    ] {
        assert_bind_error(query, "VariableAlreadyBound")?;
    }
    Ok(())
}

#[test]
fn repeated_nodes_and_relationships_keep_their_legal_roles() -> Result<()> {
    for query in [
        "MATCH (n), (n) MATCH (n) RETURN n",
        "MATCH ()-[r]-(), ()-[r]-() MATCH ()-[r]-() RETURN r",
        "MATCH (n) WITH n AS m MATCH (m) RETURN m",
        "MATCH ()-[r]-() WITH r AS s MATCH ()-[s]-() RETURN s",
        "MATCH (a), (b) WITH coalesce(a, b) AS n MATCH (n) RETURN n",
    ] {
        bind_query(query).map_err(|error| {
            Error::internal(format!(
                "legal typed binding query `{query}` failed: {error}"
            ))
        })?;
    }
    Ok(())
}

#[test]
fn value_aliases_do_not_masquerade_as_graph_entities() -> Result<()> {
    for query in [
        "MATCH (n) WITH n.value AS value MATCH (value) RETURN value",
        "UNWIND [1] AS n MATCH (n) RETURN n",
        "MATCH (n) LET value = n.value MATCH ()-[value]-() RETURN value",
    ] {
        assert_bind_error(query, "VariableTypeConflict")?;
    }
    Ok(())
}

#[test]
fn procedure_yields_retain_declared_scope_semantics() -> Result<()> {
    let mut procedures = ProcedureCatalog::default();
    procedures.register(ProcedureDefinition::new(
        "test.scalar",
        Vec::new(),
        vec![ProcedureField::new(
            "value",
            ProcedureValueType::Integer,
            false,
        )?],
        vec![vec![ResultValue::Scalar(ScalarValue::Integer(1))]],
    )?)?;

    let error = bind_with_procedures(
        parse("CALL test.scalar() YIELD value AS n MATCH (n) RETURN n")?,
        &NameCatalog::default(),
        BindCapabilities::default(),
        &procedures,
        &BTreeMap::new(),
    )
    .err()
    .ok_or_else(|| Error::internal("scalar procedure output was accepted as a node"))?;
    assert_eq!(error.code, ErrorCode::QuerySyntax);
    assert!(error.message.contains("VariableTypeConflict"));

    bind_query("CALL graph.degree() YIELD node MATCH (node) RETURN node")?;
    assert_bind_error(
        "MATCH (n) CALL graph.degree() YIELD node AS n RETURN n",
        "VariableAlreadyBound",
    )?;
    Ok(())
}
