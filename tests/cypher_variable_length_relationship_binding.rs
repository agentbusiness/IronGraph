// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use irongraph::{
    Error, ErrorCode, Result,
    cypher::{BindCapabilities, bind, parse},
    graph::NameCatalog,
};

fn bind_query(query: &str) -> Result<()> {
    bind(
        parse(query)?,
        &NameCatalog::default(),
        BindCapabilities::default(),
    )?;
    Ok(())
}

fn assert_type_conflict(query: &str) -> Result<()> {
    let error = bind_query(query)
        .err()
        .ok_or_else(|| Error::internal(format!("`{query}` unexpectedly bound")))?;
    assert_eq!(error.code, ErrorCode::QuerySyntax, "query: {query}");
    assert!(
        error.message.contains("VariableTypeConflict"),
        "query: {query}; error: {error:?}",
    );
    Ok(())
}

#[test]
fn relationship_lists_can_constrain_variable_length_patterns() -> Result<()> {
    for query in [
        "MATCH ()-[r1]->()-[r2]->() WITH [r1, r2] AS rs MATCH (first)-[rs*]->(second) RETURN first, second",
        "MATCH ()-[r*]->() WITH r AS rs MATCH (first)-[rs*]->(second) RETURN first, second",
    ] {
        bind_query(query).map_err(|error| {
            Error::internal(format!(
                "valid relationship-list pattern failed: {query}: {error}"
            ))
        })?;
    }
    Ok(())
}

#[test]
fn relationship_lists_and_single_relationships_remain_distinct() -> Result<()> {
    for query in [
        "WITH [1, 2] AS rs MATCH ()-[rs*]->() RETURN rs",
        "MATCH ()-[r]->() WITH [r] AS rs MATCH ()-[rs]->() RETURN rs",
        "MATCH ()-[rs*]->() WITH rs MATCH ()-[rs]->() RETURN rs",
    ] {
        assert_type_conflict(query)?;
    }
    Ok(())
}
