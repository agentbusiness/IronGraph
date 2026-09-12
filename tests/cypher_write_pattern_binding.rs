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
        BindCapabilities {
            write: true,
            ..BindCapabilities::default()
        },
    )
}

fn require_syntax_detail(query: &str, detail: &str) -> Result<()> {
    let error = bind_query(query)
        .err()
        .ok_or_else(|| Error::internal(format!("invalid write pattern bound: {query}")))?;
    assert_eq!(error.code, ErrorCode::QuerySyntax, "{query}: {error:?}");
    assert!(error.message.contains(detail), "{query}: {error:?}");
    Ok(())
}

#[test]
fn create_relationships_require_one_type_fixed_length_and_a_fresh_binding() -> Result<()> {
    for (query, detail) in [
        ("CREATE ()-->()", "NoSingleRelationshipType"),
        ("CREATE ()-[:A|:B]->()", "NoSingleRelationshipType"),
        ("CREATE ()-[:FOO*2]->()", "CreatingVarLength"),
        ("MATCH ()-[r]->() CREATE ()-[r]->()", "VariableAlreadyBound"),
    ] {
        require_syntax_detail(query, detail)?;
    }
    Ok(())
}

#[test]
fn merge_has_the_official_equivalent_relationship_restrictions() -> Result<()> {
    for (query, detail) in [
        (
            "CREATE (a), (b) MERGE (a)-->(b)",
            "NoSingleRelationshipType",
        ),
        (
            "CREATE (a), (b) MERGE (a)-[:A|:B]->(b)",
            "NoSingleRelationshipType",
        ),
        (
            "MERGE (a) MERGE (b) MERGE (a)-[:FOO*2]->(b)",
            "CreatingVarLength",
        ),
        (
            "MATCH (a)-[r]->(b) MERGE (a)-[r]->(b)",
            "VariableAlreadyBound",
        ),
    ] {
        require_syntax_detail(query, detail)?;
    }
    Ok(())
}

#[test]
fn create_reuses_nodes_and_merge_preserves_undirected_relationships() -> Result<()> {
    for query in [
        "CREATE (a), (b), (a)-[:R]->(b)",
        "CREATE (root)-[:LINK]->(root)",
        "MATCH (a), (b) CREATE (a)-[:R]->(b)",
        "CREATE (a), (b) MERGE (a)-[r:KNOWS]-(b) RETURN r",
        "MATCH (a), (b) MERGE (a)-[r:KNOWS]-(b) RETURN r",
        "MATCH (a)-[r:R]->(b) MATCH (b)-[r:R]->(a) RETURN r",
    ] {
        bind_query(query).map_err(|error| {
            Error::internal(format!("legal pattern was rejected for `{query}`: {error}"))
        })?;
    }
    Ok(())
}
