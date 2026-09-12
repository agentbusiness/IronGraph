// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use irongraph::{
    Error, ErrorCode, Result,
    cypher::{BindCapabilities, bind, parse, plan},
    graph::NameCatalog,
};

fn compile(query: &str) -> Result<()> {
    let bound = bind(
        parse(query)?,
        &NameCatalog::default(),
        BindCapabilities::default(),
    )?;
    plan(bound).map(|_| ())
}

fn require_compile_error(query: &str, category: &str) -> Result<()> {
    let error = compile(query)
        .err()
        .ok_or_else(|| Error::internal(format!("query unexpectedly compiled: {query}")))?;
    assert_eq!(error.code, ErrorCode::QuerySyntax, "{query}: {error:?}");
    assert!(error.message.contains(category), "{query}: {error:?}");
    Ok(())
}

#[test]
fn aggregate_calls_are_rejected_in_predicates_and_volatile_arguments() -> Result<()> {
    require_compile_error(
        "MATCH (a) WHERE count(a) > 10 RETURN a",
        "InvalidAggregation",
    )?;
    require_compile_error("RETURN count(rand())", "NonConstantExpression")?;
    Ok(())
}

#[test]
fn mixed_aggregate_projection_requires_separate_simple_grouping_keys() -> Result<()> {
    for query in [
        "MATCH (me: Person)--(you: Person) RETURN me.age + count(you.age)",
        "MATCH (me: Person)--(you: Person) RETURN me.age + you.age, me.age + you.age + count(*)",
        "MATCH (me: Person)--(you: Person) WITH me.age + count(you.age) AS agg RETURN *",
        "MATCH (me: Person)--(you: Person) WITH me.age + you.age AS grp, me.age + you.age + count(*) AS agg RETURN *",
    ] {
        require_compile_error(query, "AmbiguousAggregationExpression")?;
    }
    Ok(())
}

#[test]
fn valid_grouping_constants_and_local_iteration_bindings_remain_legal() -> Result<()> {
    for query in [
        "MATCH (person) RETURN $age + avg(person.age) - 1000",
        "MATCH (me: Person)--(you: Person) WITH me.age AS age, you RETURN age, age + count(you.age)",
        "MATCH (me: Person)--(you: Person) RETURN me.age, me.age + count(you.age)",
        "MATCH (me: Person)--(you: Person) WITH me.age AS age, you WITH age, age + count(you.age) AS agg RETURN *",
        "MATCH (me: Person)--(you: Person) WITH me.age AS age, me.age + count(you.age) AS agg RETURN *",
        "MATCH (n) RETURN [x IN collect(n) | x] AS collected",
        "MATCH (n) RETURN n.age + n.score AS complexGroup, count(*) AS total",
        "MATCH (n) RETURN n, n.age + count(*) AS derivedFromNode",
    ] {
        compile(query).map_err(|error| {
            Error::internal(format!(
                "legal aggregate projection was rejected for `{query}`: {error}"
            ))
        })?;
    }
    Ok(())
}
