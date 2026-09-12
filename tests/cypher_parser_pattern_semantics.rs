// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use irongraph::{
    Error, ErrorCode, Result,
    cypher::{Clause, Direction, Statement, parse},
};

fn parsed_pattern(query: &str) -> Result<irongraph::cypher::Pattern> {
    let parsed = parse(query)?;
    let Statement::Query(body) = parsed.statement else {
        return Err(Error::internal("expected a query statement"));
    };
    let clause = body
        .clauses
        .into_iter()
        .next()
        .ok_or_else(|| Error::internal("expected a query clause"))?;
    let Clause::Match { patterns, .. } = clause else {
        return Err(Error::internal("expected a MATCH clause"));
    };
    patterns
        .into_iter()
        .next()
        .ok_or_else(|| Error::internal("expected a MATCH pattern"))
}

fn parse_error(query: &str) -> Result<Error> {
    match parse(query) {
        Ok(_) => Err(Error::internal("query unexpectedly parsed successfully")),
        Err(error) => Ok(error),
    }
}

#[test]
fn parameter_pattern_predicates_report_the_official_detail() -> Result<()> {
    for query in [
        "MATCH (n $param) RETURN n",
        "MATCH ()-[r:FOO $param]->() RETURN r",
        "MERGE (n $param) RETURN n",
        "MERGE (a)-[r:FOO $param]->(b) RETURN r",
    ] {
        let error = parse_error(query)?;
        assert_eq!(error.code, ErrorCode::QuerySyntax, "{query}: {error:?}");
        assert!(
            error.message.contains("InvalidParameterUse"),
            "{query}: {}",
            error.message
        );
        assert!(
            !error.message.starts_with("UnexpectedSyntax"),
            "{query}: {}",
            error.message
        );
    }
    Ok(())
}

#[test]
fn bidirectional_read_patterns_are_undirected() -> Result<()> {
    let pattern = parsed_pattern("MATCH p=(n)<-->(k)<-->(n) RETURN p")?;
    assert_eq!(
        pattern
            .steps
            .iter()
            .map(|step| step.relationship.direction)
            .collect::<Vec<_>>(),
        vec![Direction::Undirected, Direction::Undirected]
    );

    let pattern = parsed_pattern("MATCH p=(n)<-->(k)<--(n) RETURN p")?;
    assert_eq!(
        pattern
            .steps
            .iter()
            .map(|step| step.relationship.direction)
            .collect::<Vec<_>>(),
        vec![Direction::Undirected, Direction::Incoming]
    );
    Ok(())
}

#[test]
fn write_patterns_require_exactly_one_direction() -> Result<()> {
    for query in ["CREATE (a)-[:FOO]-(b)", "CREATE (a)<-[:FOO]->(b)"] {
        let error = parse_error(query)?;
        assert_eq!(error.code, ErrorCode::QuerySyntax, "{query}: {error:?}");
        assert!(
            error.message.contains("RequiresDirectedRelationship"),
            "{query}: {}",
            error.message
        );
    }
    assert!(parse("CREATE (a)-[:FOO]->(b)").is_ok());
    assert!(parse("CREATE (a)<-[:FOO]-(b)").is_ok());
    for query in ["MERGE (a)-[:FOO]-(b)", "MERGE (a)<-[:FOO]->(b)"] {
        let parsed = parse(query)?;
        let Statement::Query(body) = parsed.statement else {
            return Err(Error::internal("expected a MERGE query statement"));
        };
        let Some(Clause::Merge { pattern, .. }) = body.clauses.first() else {
            return Err(Error::internal("expected a MERGE clause"));
        };
        assert_eq!(
            pattern.steps[0].relationship.direction,
            Direction::Undirected,
            "{query}"
        );
    }
    Ok(())
}
