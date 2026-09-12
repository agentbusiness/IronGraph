// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use irongraph::{
    Error, ErrorCode, Result, ScalarValue,
    cypher::{Clause, Expression, ProjectionItem, Statement, parse},
};

const ENTITY_LABEL_PREDICATE_INTRINSIC: &str = "\0irongraph.entity_label_predicate";

fn query_clauses(source: &str) -> Result<Vec<Clause>> {
    let parsed = parse(source)?;
    let Statement::Query(body) = parsed.statement else {
        return Err(Error::internal("query statement required"));
    };
    Ok(body.clauses)
}

fn return_expression(source: &str, item_index: usize) -> Result<Expression> {
    let clauses = query_clauses(source)?;
    let projection = clauses
        .into_iter()
        .find_map(|clause| match clause {
            Clause::Return(projection) => Some(projection),
            _ => None,
        })
        .ok_or_else(|| Error::internal("RETURN clause required"))?;
    projection
        .items
        .get(item_index)
        .map(|item| item.expression.clone())
        .ok_or_else(|| Error::internal("RETURN item is missing"))
}

fn where_expression(source: &str) -> Result<Expression> {
    query_clauses(source)?
        .into_iter()
        .find_map(|clause| match clause {
            Clause::Where(expression) => Some(expression),
            _ => None,
        })
        .ok_or_else(|| Error::internal("WHERE clause required"))
}

fn assert_entity_label_predicate(
    expression: &Expression,
    expected_variable: &str,
    expected_names: &[&str],
) -> Result<()> {
    let Expression::Function {
        name,
        distinct: false,
        arguments,
    } = expression
    else {
        return Err(Error::internal(
            "colon syntax must remain one polymorphic AST intrinsic",
        ));
    };
    assert_eq!(name.as_slice(), [ENTITY_LABEL_PREDICATE_INTRINSIC]);
    let Some((source, names)) = arguments.split_first() else {
        return Err(Error::internal("label predicate source is missing"));
    };
    assert!(matches!(
        source,
        Expression::Variable(variable) if variable == expected_variable
    ));
    let actual_names = names
        .iter()
        .map(|name| match name {
            Expression::Literal(ScalarValue::String(name)) => Ok(name.as_ref()),
            _ => Err(Error::internal(
                "label/type names must remain exact string literals",
            )),
        })
        .collect::<Result<Vec<_>>>()?;
    assert_eq!(actual_names, expected_names);
    Ok(())
}

#[test]
fn graph5_scenarios_1_through_5_keep_polymorphic_semantics_in_the_ast() -> Result<()> {
    // Graph5 [1]: node labels.
    let expression = return_expression("MATCH (a) RETURN a, a:B AS result", 1)?;
    assert_entity_label_predicate(&expression, "a", &["B"])?;

    // Graph5 [2]: the same syntax over a relationship means exact, case-sensitive type equality.
    let expression = return_expression("MATCH ()-[r]->() RETURN r, r:T2 AS result", 1)?;
    assert_entity_label_predicate(&expression, "r", &["T2"])?;
    let lower_case = return_expression("MATCH ()-[r]->() RETURN r, r:t2 AS result", 1)?;
    assert_entity_label_predicate(&lower_case, "r", &["t2"])?;
    assert_ne!(
        expression, lower_case,
        "relationship type names retain case"
    );

    // Graph5 [3]: multiple names are one conjunctive predicate, not nested labels() calls.
    let expression = return_expression("MATCH (a) RETURN a, a:A:B AS result", 1)?;
    assert_entity_label_predicate(&expression, "a", &["A", "B"])?;

    // Graph5 [4]: order and repeated names are preserved exactly for conjunctive evaluation.
    for (suffix, expected) in [
        ("A:C", &["A", "C"][..]),
        ("C:A", &["C", "A"][..]),
        ("A:C:A", &["A", "C", "A"][..]),
        ("C:C:A", &["C", "C", "A"][..]),
        ("C:A:A:C", &["C", "A", "A", "C"][..]),
    ] {
        let query = format!("MATCH (a) WHERE a:{suffix} RETURN a");
        let expression = where_expression(&query)?;
        assert_entity_label_predicate(&expression, "a", expected)?;
    }

    // Graph5 [5]: the source remains intact so runtime can propagate a NULL optional binding.
    let expression = return_expression(
        "MATCH (n:Single) OPTIONAL MATCH (n)-[r:TYPE]-(m) RETURN m:TYPE",
        0,
    )?;
    assert_entity_label_predicate(&expression, "m", &["TYPE"])?;
    Ok(())
}

#[test]
fn polymorphic_intrinsic_has_a_canonical_cypher_display_name() -> Result<()> {
    let expression = return_expression("MATCH ()-[r]->() RETURN r:T2", 0)?;
    let synthesized_projection = ProjectionItem {
        expression,
        alias: None,
        source_text: None,
    };

    assert_eq!(synthesized_projection.column_name(0), "r:T2");
    assert!(
        !synthesized_projection
            .column_name(0)
            .contains(ENTITY_LABEL_PREDICATE_INTRINSIC),
        "the unspellable intrinsic marker must not become client-visible",
    );
    Ok(())
}

#[test]
fn accepts_label_predicates_in_shared_expression_positions() -> Result<()> {
    for query in [
        "MATCH (a)-[:ADMIN]-(b) WHERE a:A RETURN a.id, b.id",
        "MATCH (:Root {name: 'x'})-->(i:TextNode) WHERE i.var > 'te' AND i:TextNode RETURN i",
        "MATCH (a) WHERE NOT a:A RETURN a",
        "MATCH (a) WHERE NOT (a:B) RETURN a",
        "MATCH (n) RETURN (n:Foo)",
        "MATCH (n) RETURN CASE WHEN n:Foo THEN 1 ELSE 0 END AS value",
        "MATCH (n) RETURN any(x IN [n] WHERE x:Foo) AS value",
        "MATCH (n) WHERE n:`Label with spaces` RETURN n",
    ] {
        parse(query).map_err(|error| {
            Error::internal(format!(
                "valid label predicate was rejected in `{query}`: {error}"
            ))
        })?;
    }
    Ok(())
}

#[test]
fn malformed_label_predicates_remain_syntax_errors() -> Result<()> {
    for query in [
        "MATCH (a) WHERE a: RETURN a",
        "MATCH (a) WHERE a::A RETURN a",
        "MATCH (a) WHERE a:42 RETURN a",
        "MATCH (a) WHERE a:A: RETURN a",
        "MATCH (a) WHERE a:A::B RETURN a",
    ] {
        let error = parse(query).expect_err("malformed label predicate was accepted");
        assert_eq!(error.code, ErrorCode::QuerySyntax, "query: {query}");
    }
    Ok(())
}

#[test]
fn delete_still_rejects_a_label_predicate_as_its_target() -> Result<()> {
    let error = parse("MATCH (n) DELETE n:Person")
        .expect_err("a node label predicate is not a deletable entity");
    assert_eq!(error.code, ErrorCode::QuerySyntax);
    assert!(error.message.contains("InvalidDelete"));

    parse("MATCH (n) WHERE n:Person DELETE n")?;
    Ok(())
}
