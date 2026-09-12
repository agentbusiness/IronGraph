// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use irongraph::{
    Error, ErrorCode, Result,
    cypher::{BindCapabilities, DependencyKind, bind, parse},
    graph::NameCatalog,
};

fn bind_query(query: &str) -> Result<irongraph::cypher::BoundQuery> {
    bind(
        parse(query)?,
        &NameCatalog::default(),
        BindCapabilities::default(),
    )
}

#[test]
fn node_and_relationship_colon_predicates_bind_as_one_polymorphic_operation() -> Result<()> {
    for query in [
        "MATCH (n) WHERE n:A RETURN n",
        "MATCH ()-[r]->() WHERE r:T RETURN r",
        "MATCH (n) RETURN n:A:B AS result",
        "MATCH (n) RETURN true AND (n:A) AS result",
        "RETURN 1:A AS deferred_runtime_type_error",
    ] {
        bind_query(query).map_err(|error| {
            Error::internal(format!(
                "valid colon predicate failed to bind: {query}: {error}"
            ))
        })?;
    }
    Ok(())
}

#[test]
fn colon_predicates_stamp_both_possible_entity_catalog_dependencies() -> Result<()> {
    let bound = bind_query("MATCH (value) WHERE value:A:B RETURN value")?;
    for (kind, name) in [
        (DependencyKind::Label, "A"),
        (DependencyKind::RelationshipType, "A"),
        (DependencyKind::Label, "B"),
        (DependencyKind::RelationshipType, "B"),
    ] {
        assert!(
            bound
                .dependencies
                .iter()
                .any(|dependency| dependency.kind == kind && dependency.name == name),
            "missing {kind:?} dependency for {name}",
        );
    }
    Ok(())
}

#[test]
fn undefined_colon_predicate_sources_keep_the_standard_variable_error() -> Result<()> {
    let error = bind_query("RETURN missing:A")
        .err()
        .ok_or_else(|| Error::internal("undefined colon-predicate source unexpectedly bound"))?;
    assert_eq!(error.code, ErrorCode::QuerySyntax);
    assert!(error.message.contains("UndefinedVariable"), "{error:?}");
    assert!(!error.message.contains("UnknownFunction"), "{error:?}");
    Ok(())
}
