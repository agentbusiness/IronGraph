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
    require_bind_detail(query, ErrorCode::QuerySyntax, detail)
}

fn require_bind_detail(query: &str, code: ErrorCode, detail: &str) -> Result<()> {
    let error = bind_query(query)
        .err()
        .ok_or_else(|| Error::internal(format!("invalid pattern was bound: {query}")))?;
    assert_eq!(error.code, code, "{query}: {error:?}");
    assert!(error.message.contains(detail), "{query}: {error:?}");
    Ok(())
}

#[test]
fn bound_nodes_cannot_be_redeclared_or_constrained_by_write_patterns() -> Result<()> {
    for query in [
        "MATCH (a) CREATE (a)",
        "MATCH (a) CREATE (a {name: 'foo'}) RETURN a",
        "CREATE (n:Foo)-[:T1]->(), (n:Bar)-[:T2]->()",
        "CREATE ()<-[:T2]-(n:Foo), (n:Bar)<-[:T1]-()",
        "CREATE (n:Foo) CREATE (n:Bar)-[:OWNS]->(:Dog)",
        "CREATE (n {}) CREATE (n:Bar)-[:OWNS]->(:Dog)",
        "CREATE (n:Foo) CREATE (n {})-[:OWNS]->(:Dog)",
        "CREATE (n:Foo) CREATE (n {id: 1})-[:OWNS]->(:Dog)",
        "CREATE (n:Foo) CREATE (n {id: $id})-[:OWNS]->(:Dog)",
        "MATCH (a) MERGE (a)",
        "CREATE (a:Foo) MERGE (a)-[r:KNOWS]->(a:Bar)",
        "CREATE (a:Foo) MERGE (a {})-[r:KNOWS]->(:Bar)",
    ] {
        require_syntax_detail(query, "VariableAlreadyBound")?;
    }
    Ok(())
}

#[test]
fn bound_nodes_remain_legal_as_unconstrained_relationship_endpoints() -> Result<()> {
    for query in [
        "CREATE (a), (b), (a)-[:R]->(b)",
        "CREATE (a) CREATE (b) CREATE (a)-[:R]->(b)",
        "MATCH (a), (b) CREATE (a)-[:R]->(b)",
        "MATCH (a), (b) MERGE (a)-[r:R]->(b) RETURN r",
        "MATCH (a) CREATE (a)-[:OWNS]->(car:Car {id: 1})",
        "MATCH (a) MERGE (a)-[:OWNS]->(car:Car {id: 1})",
        "CREATE (root)-[:LINK]->(root)",
        "CREATE (a:Foo)-[:R]->(a)",
        "CREATE (a:Foo) CREATE (a)-[:OWNS]->(:Dog)",
        "CREATE (a:Foo) MERGE (a)-[:OWNS]->(:Dog)",
        "MATCH (a) MATCH (a {}) RETURN a",
    ] {
        bind_query(query).map_err(|error| {
            Error::internal(format!(
                "legal endpoint reuse was rejected for `{query}`: {error}"
            ))
        })?;
    }
    Ok(())
}

#[test]
fn relationship_uniqueness_is_owned_by_one_match_pattern() -> Result<()> {
    require_syntax_detail(
        "MATCH (a)-[r]->()-[r]->(a) RETURN r",
        "RelationshipUniquenessViolation",
    )?;

    for query in [
        "MATCH (a)-[r:R]->(b), (b)-[r:R]->(a) RETURN r",
        "MATCH (a)-[r:R]->(b) MATCH (b)-[r:R]->(a) RETURN r",
    ] {
        bind_query(query).map_err(|error| {
            Error::internal(format!(
                "legal cross-pattern relationship correlation was rejected for `{query}`: {error}"
            ))
        })?;
    }
    Ok(())
}

#[test]
fn delete_rejects_only_statically_non_graph_targets() -> Result<()> {
    for query in [
        "MATCH () DELETE 1 + 1",
        "MATCH (n) DELETE 42",
        "MATCH (n) DELETE [n]",
        "MATCH (n) WITH 1 AS target DELETE target",
    ] {
        require_bind_detail(query, ErrorCode::QuerySyntax, "InvalidArgumentType")?;
    }

    for query in [
        "MATCH (n) DELETE n",
        "MATCH ()-[r]->() DELETE r",
        "MATCH p = ()-->() DETACH DELETE p",
        "MATCH (n) WITH n AS target DELETE target",
        "MATCH ()-[r]->() WITH r AS target DELETE target",
        "MATCH p = ()-->() WITH p AS target DETACH DELETE target",
        "MATCH (n) WITH {target: n} AS value DELETE value.target",
        "MATCH (n) WITH collect(n) AS values DELETE values[$index]",
        "WITH $target AS value DELETE value",
        "WITH null AS value DELETE value",
    ] {
        bind_query(query).map_err(|error| {
            Error::internal(format!(
                "potentially valid DELETE target was rejected for `{query}`: {error}"
            ))
        })?;
    }
    Ok(())
}
