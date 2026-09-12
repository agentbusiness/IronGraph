// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use irongraph::{Error, Result, cypher::parse};

fn require_invalid_delete(query: &str) -> Result<()> {
    let error = parse(query)
        .err()
        .ok_or_else(|| Error::internal("query was accepted"))?;
    let detail = error.to_string();
    if !detail.contains("InvalidDelete") {
        return Err(Error::internal(format!(
            "expected InvalidDelete detail, got `{detail}`"
        )));
    }
    Ok(())
}

#[test]
fn labels_and_relationship_types_are_not_delete_expressions() -> Result<()> {
    require_invalid_delete("MATCH (n) DELETE n:Person")?;
    require_invalid_delete("MATCH ()-[r:T]-() DELETE r:T")?;
    Ok(())
}
