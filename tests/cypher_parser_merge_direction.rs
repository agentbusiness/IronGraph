// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use irongraph::{
    Error, Result,
    cypher::{Clause, Direction, Statement, parse},
};

#[test]
fn merge_accepts_an_undirected_relationship_but_create_does_not() -> Result<()> {
    let parsed = parse("MERGE (a)-[:T]-(b)")?;
    let Statement::Query(body) = parsed.statement else {
        return Err(Error::internal("query statement required"));
    };
    let Some(Clause::Merge { pattern, .. }) = body.clauses.first() else {
        return Err(Error::internal("MERGE clause required"));
    };
    let Some(step) = pattern.steps.first() else {
        return Err(Error::internal("relationship step required"));
    };
    assert_eq!(step.relationship.direction, Direction::Undirected);

    let create_error = parse("CREATE (a)-[:T]-(b)")
        .err()
        .ok_or_else(|| Error::internal("undirected CREATE was accepted"))?;
    assert!(
        create_error
            .to_string()
            .contains("RequiresDirectedRelationship")
    );
    Ok(())
}
