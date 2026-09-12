// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use irongraph::{
    Error, Result, ScalarValue,
    cypher::{Clause, Expression, SetItem, Statement, parse},
};

fn set_items(query: &str) -> Result<Vec<SetItem>> {
    let parsed = parse(query)?;
    let Statement::Query(body) = parsed.statement else {
        return Err(Error::internal("query statement required"));
    };
    body.clauses
        .into_iter()
        .find_map(|clause| match clause {
            Clause::Set(items) => Some(items),
            _ => None,
        })
        .ok_or_else(|| Error::internal("SET clause required"))
}

fn query_failure(query: &str) -> Result<()> {
    match parse(query) {
        Ok(_) => Err(Error::internal("query was accepted")),
        Err(_) => Ok(()),
    }
}

#[test]
fn simple_parenthesized_variables_normalize_to_property_targets() -> Result<()> {
    let items =
        set_items("MATCH (n)-[r]->() SET (n).name = 'neo4j', (r).name = 'graph' RETURN n, r")?;
    let [node_item, relationship_item] = items.as_slice() else {
        return Err(Error::internal("two SET items required"));
    };

    assert!(matches!(
        node_item,
        SetItem::Property {
            target,
            value: Expression::Literal(ScalarValue::String(value)),
            event_time: None,
        } if target.variable == "n" && target.property == "name" && value.as_ref() == "neo4j"
    ));
    assert!(matches!(
        relationship_item,
        SetItem::Property {
            target,
            value: Expression::Literal(ScalarValue::String(value)),
            event_time: None,
        } if target.variable == "r" && target.property == "name" && value.as_ref() == "graph"
    ));
    Ok(())
}

#[test]
fn parenthesized_set_targets_remain_limited_to_one_variable_property_selector() -> Result<()> {
    for query in [
        "SET (n + 1).name = 'value'",
        "SET (n.name).other = 'value'",
        "SET (function()).name = 'value'",
        "SET ((n)).name = 'value'",
        "SET (n) = {name: 'value'}",
        "SET (n):Label",
    ] {
        query_failure(query)?;
    }

    assert!(parse("SET n.name = 'value'").is_ok());
    assert!(parse("SET n += {name: 'value'}").is_ok());
    assert!(parse("SET n = {name: 'value'}").is_ok());
    assert!(parse("SET n:Label").is_ok());
    Ok(())
}
