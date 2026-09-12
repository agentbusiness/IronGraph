// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use irongraph::{Error, Result, cypher::parse};

fn require_invalid_relationship_pattern(query: &str) -> Result<()> {
    let error = parse(query)
        .err()
        .ok_or_else(|| Error::internal("query was accepted"))?;
    let detail = error.to_string();
    if !detail.contains("InvalidRelationshipPattern") {
        return Err(Error::internal(format!(
            "expected InvalidRelationshipPattern detail, got `{detail}`"
        )));
    }
    Ok(())
}

#[test]
fn malformed_variable_length_relationships_have_the_standard_detail() -> Result<()> {
    require_invalid_relationship_pattern("MATCH (a:A) MATCH (a)-[:LIKES..]->(c) RETURN c.name")?;
    require_invalid_relationship_pattern("MATCH (a:A) MATCH (a)-[:LIKES*-2]->(c) RETURN c.name")?;
    Ok(())
}
