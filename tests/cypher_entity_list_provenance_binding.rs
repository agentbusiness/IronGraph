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

fn require_variable_type_conflict(query: &str) -> Result<()> {
    let error = bind_query(query)
        .err()
        .ok_or_else(|| Error::internal(format!("`{query}` unexpectedly passed binding")))?;
    assert_eq!(error.code, ErrorCode::QuerySyntax, "{query}: {error:?}");
    assert!(
        error.message.contains("VariableTypeConflict"),
        "{query}: {error:?}"
    );
    Ok(())
}

#[test]
fn collected_nodes_survive_with_aliases_and_unwind_as_nodes() -> Result<()> {
    for query in [
        "MATCH (a:S)-[:X]->(b1) \
         WITH a, collect(b1) AS bees \
         UNWIND bees AS b2 \
         MATCH (a)-[:Y]->(b2) \
         RETURN a, b2",
        "MATCH (n) \
         WITH collect(n) AS nodes \
         WITH nodes AS aliases \
         UNWIND aliases AS selected \
         MATCH (selected) \
         RETURN selected",
    ] {
        bind_query(query).map_err(|error| {
            Error::internal(format!(
                "node-list provenance query `{query}` failed binding: {error}"
            ))
        })?;
    }
    Ok(())
}

#[test]
fn collected_relationships_survive_with_aliases_and_unwind_as_relationships() -> Result<()> {
    bind_query(
        "MATCH ()-[r:X]->() \
         WITH collect(r) AS relationships \
         WITH relationships AS aliases \
         UNWIND aliases AS selected \
         MATCH ()-[selected:Y]->() \
         RETURN selected",
    )?;
    Ok(())
}

#[test]
fn node_and_relationship_element_roles_cannot_be_swapped() -> Result<()> {
    for query in [
        "MATCH (n) \
         WITH collect(n) AS entities \
         UNWIND entities AS r \
         MATCH ()-[r]->() \
         RETURN r",
        "MATCH ()-[r]->() \
         WITH collect(r) AS entities \
         UNWIND entities AS n \
         MATCH (n) \
         RETURN n",
    ] {
        require_variable_type_conflict(query)?;
    }
    Ok(())
}

#[test]
fn definitely_scalar_lists_cannot_masquerade_as_graph_entities() -> Result<()> {
    for query in [
        "WITH [1, 2] AS values UNWIND values AS n MATCH (n) RETURN n",
        "UNWIND [1, 2] AS value \
         WITH collect(value) AS values \
         UNWIND values AS r \
         MATCH ()-[r]->() \
         RETURN r",
        "MATCH (n) \
         WITH collect(n.name) AS values \
         UNWIND values AS entity \
         MATCH (entity) \
         RETURN entity",
    ] {
        require_variable_type_conflict(query)?;
    }
    Ok(())
}

#[test]
fn entity_lists_must_be_unwound_before_pattern_reuse() -> Result<()> {
    for query in [
        "MATCH (n) WITH collect(n) AS nodes MATCH (nodes) RETURN nodes",
        "MATCH ()-[r]->() \
         WITH collect(r) AS relationships \
         MATCH ()-[relationships]->() \
         RETURN relationships",
    ] {
        require_variable_type_conflict(query)?;
    }
    Ok(())
}

#[test]
fn unknown_list_elements_are_deferred_to_runtime_role_checks() -> Result<()> {
    for query in [
        "WITH $entities AS entities \
         UNWIND entities AS node \
         MATCH (node) \
         RETURN node",
        "WITH $entities AS entities \
         UNWIND entities AS relationship \
         MATCH ()-[relationship]->() \
         RETURN relationship",
    ] {
        bind_query(query).map_err(|error| {
            Error::internal(format!(
                "dynamic-list query `{query}` was rejected before runtime: {error}"
            ))
        })?;
    }
    Ok(())
}

#[test]
fn concatenated_node_lists_can_be_indexed_as_create_endpoints() -> Result<()> {
    let query = "CREATE (a {var:'start'}), (b {var:'end'})
                 WITH *
                 UNWIND range(1,20) AS i
                 CREATE (n {var:i})
                 WITH a,b,[a] + collect(n) + [b] AS nodeList
                 UNWIND range(0,size(nodeList)-2,1) AS i
                 WITH nodeList[i] AS n1, nodeList[i+1] AS n2
                 CREATE (n1)-[:T]->(n2)";

    bind_query(query).map_err(|error| {
        Error::internal(format!(
            "concatenated node-list TCK query failed binding: {error}"
        ))
    })?;
    Ok(())
}

#[test]
fn entity_list_addition_concatenation_and_slicing_preserve_proven_roles() -> Result<()> {
    for query in [
        "MATCH (a), (b)
         WITH collect(a) AS nodes, b
         WITH nodes + b AS nodes
         WITH nodes[1..] AS tail
         UNWIND tail AS selected
         CREATE (selected)-[:T]->()",
        "MATCH (a), (b)
         WITH a, collect(b) AS nodes
         WITH a + nodes AS nodes
         WITH nodes[..1] AS head
         UNWIND head AS selected
         CREATE (selected)-[:T]->()",
        "MATCH (a), (b)
         WITH a, collect(b) AS remaining
         WITH [a] || remaining AS nodes
         WITH nodes[0] AS selected
         CREATE (selected)-[:T]->()",
        "MATCH ()-[first:T]->(), ()-[second:T]->()
         WITH first, collect(second) AS remaining
         WITH [first] + remaining AS relationships
         WITH relationships[0..] AS copied
         UNWIND copied AS selected
         MATCH ()-[selected]->()
         RETURN selected",
        "MATCH (n)
         WITH [] + collect(n) + [] AS nodes
         WITH nodes[0] AS selected
         CREATE (selected)-[:T]->()",
    ] {
        bind_query(query).map_err(|error| {
            Error::internal(format!(
                "proven entity-list operation `{query}` failed binding: {error}"
            ))
        })?;
    }
    Ok(())
}

#[test]
fn mixed_or_scalar_list_indexes_cannot_masquerade_as_write_entities() -> Result<()> {
    for query in [
        "MATCH (n), ()-[r:T]->()
         WITH [n] + [r] AS mixed
         WITH mixed[0] AS endpoint
         CREATE (endpoint)-[:X]->()",
        "MATCH (n)
         WITH collect(n) + [1] AS mixed
         WITH mixed[0] AS endpoint
         CREATE (endpoint)-[:X]->()",
        "WITH [1, 2] AS values
         WITH values[0] AS endpoint
         CREATE (endpoint)-[:X]->()",
        "WITH $values AS values
         WITH values[0] AS endpoint
         CREATE (endpoint)-[:X]->()",
    ] {
        require_variable_type_conflict(query)?;
    }
    Ok(())
}

#[test]
fn indexed_entity_list_roles_cannot_be_swapped() -> Result<()> {
    for query in [
        "MATCH (n)
         WITH collect(n) AS entities
         WITH entities[0] AS r
         MATCH ()-[r]->()
         RETURN r",
        "MATCH ()-[r:T]->()
         WITH collect(r) AS entities
         WITH entities[0] AS n
         MATCH (n)
         RETURN n",
    ] {
        require_variable_type_conflict(query)?;
    }
    Ok(())
}

#[test]
fn heterogeneous_list_indexes_remain_runtime_resolved_for_reads() -> Result<()> {
    bind_query(
        "MATCH (n), ()-[r:T]->()
         WITH [n, r] AS mixed
         WITH mixed[$index] AS selected
         MATCH (selected)
         RETURN selected",
    )?;
    Ok(())
}
