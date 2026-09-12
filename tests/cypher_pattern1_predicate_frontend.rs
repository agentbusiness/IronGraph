// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use irongraph::{
    Error, ErrorCode, Result, ScalarValue,
    cypher::{
        BinaryOperator, BindCapabilities, Clause, Expression, Statement, UnaryOperator, bind, parse,
    },
    graph::NameCatalog,
};

fn bind_query(query: &str) -> Result<()> {
    bind(
        parse(query)?,
        &NameCatalog::default(),
        BindCapabilities::default(),
    )?;
    Ok(())
}

fn query_body(query: &str) -> Result<irongraph::cypher::QueryBody> {
    let query = parse(query)?;
    let Statement::Query(body) = query.statement else {
        return Err(Error::internal("expected query body"));
    };
    Ok(body)
}

fn is_pattern_predicate(expression: &Expression) -> bool {
    matches!(
        expression,
        Expression::Function { name, distinct: false, .. }
            if matches!(name.as_slice(), [intrinsic]
                if intrinsic.starts_with('\0') && intrinsic.ends_with(".pattern_predicate"))
    )
}

fn pattern_predicate_count(expression: &Expression) -> usize {
    let own = usize::from(is_pattern_predicate(expression));
    own + match expression {
        Expression::Unary { operand, .. } => pattern_predicate_count(operand),
        Expression::Binary { left, right, .. } => {
            pattern_predicate_count(left) + pattern_predicate_count(right)
        }
        Expression::Function { arguments, .. } | Expression::List(arguments) => {
            arguments.iter().map(pattern_predicate_count).sum()
        }
        _ => 0,
    }
}

#[test]
fn parses_and_binds_all_pattern1_predicate_shapes() -> Result<()> {
    for query in [
        "MATCH (n) WHERE (n)-[]->() RETURN n",
        "MATCH (n) WHERE (n)-[]-() RETURN n",
        "MATCH (n) WHERE (n)<-[]-() RETURN n",
        "MATCH (n) WHERE (n)-[:REL1]->() RETURN n",
        "MATCH (n) WHERE (n)-[:REL1]-() RETURN n",
        "MATCH (n) WHERE (n)<-[:REL1]-() RETURN n",
        "MATCH (n) WHERE (n)-[:REL1*]->() RETURN n",
        "MATCH (n) WHERE (n)-[:REL1*]-() RETURN n",
        "MATCH (n) WHERE (n)<-[:REL1*]-() RETURN n",
        "MATCH (n) WHERE (n)-[:REL1*2]-() RETURN n",
        "MATCH (n), (m) WHERE (n)-[:REL1|REL2|REL3|REL4]-(m) RETURN n, m",
        "MATCH (n), (m) WHERE (n)-[:REL1*0..2]->(m) RETURN n, m",
        "MATCH (n) WHERE NOT (n)-[:REL2]-() RETURN n",
        "MATCH (n) WHERE (n)-[:REL1]-() AND (n)-[:REL3]-() RETURN n",
        "MATCH (n) WHERE (n)-[:REL1]-() OR (n)-[:REL2]-() RETURN n",
    ] {
        let body = query_body(query)?;
        let where_expression = body
            .clauses
            .iter()
            .find_map(|clause| match clause {
                Clause::Where(expression) => Some(expression),
                _ => None,
            })
            .ok_or_else(|| Error::internal(format!("missing WHERE expression for `{query}`")))?;
        assert!(
            pattern_predicate_count(where_expression) >= 1,
            "pattern predicate was not preserved for `{query}`: {where_expression:?}",
        );
        bind_query(query).map_err(|error| {
            Error::internal(format!(
                "valid pattern predicate failed to bind: {query}: {error}"
            ))
        })?;
    }
    Ok(())
}

#[test]
fn predicate_envelope_preserves_direction_range_types_and_properties() -> Result<()> {
    let body =
        query_body("MATCH (a), (b) WHERE (a)<-[:R|S*0..2 {weight: 7}]-(b:B {id: 9}) RETURN a")?;
    let Some(Clause::Where(Expression::Function {
        name,
        distinct: false,
        arguments,
    })) = body.clauses.get(1)
    else {
        return Err(Error::internal("expected encoded pattern predicate"));
    };
    assert!(matches!(name.as_slice(), [name] if name.starts_with('\0')));
    let [path_name, _selector, _mode, start, steps] = arguments.as_slice() else {
        return Err(Error::internal("malformed top-level pattern envelope"));
    };
    assert!(matches!(path_name, Expression::Literal(ScalarValue::Null)));
    assert!(matches!(
        start,
        Expression::Function { arguments, .. }
            if matches!(arguments.first(), Some(Expression::Literal(ScalarValue::String(name))) if name.as_ref() == "a")
    ));
    let Expression::List(steps) = steps else {
        return Err(Error::internal("expected encoded pattern steps"));
    };
    let [Expression::List(step)] = steps.as_slice() else {
        return Err(Error::internal("expected one encoded pattern step"));
    };
    let [
        Expression::Function {
            arguments: relationship,
            ..
        },
        Expression::Function {
            arguments: node, ..
        },
    ] = step.as_slice()
    else {
        return Err(Error::internal("expected relationship and node envelopes"));
    };
    let [
        _variable,
        Expression::List(types),
        direction,
        variable_length,
        minimum,
        maximum,
        Expression::Map(properties),
    ] = relationship.as_slice()
    else {
        return Err(Error::internal("malformed relationship envelope"));
    };
    assert_eq!(types, &[Expression::string("R"), Expression::string("S"),]);
    assert!(matches!(
        direction,
        Expression::Literal(ScalarValue::Integer(1))
    ));
    assert!(matches!(
        variable_length,
        Expression::Literal(ScalarValue::Boolean(true))
    ));
    assert!(matches!(
        minimum,
        Expression::Literal(ScalarValue::Integer(0))
    ));
    assert!(matches!(
        maximum,
        Expression::Literal(ScalarValue::Integer(2))
    ));
    assert!(matches!(
        properties.as_slice(),
        [(name, Expression::Literal(ScalarValue::Integer(7)))] if name == "weight"
    ));
    assert!(matches!(
        node.as_slice(),
        [Expression::Literal(ScalarValue::String(variable)), Expression::List(labels), Expression::Literal(ScalarValue::Boolean(true)), Expression::Map(properties)]
            if variable.as_ref() == "b"
                && matches!(labels.as_slice(), [Expression::Literal(ScalarValue::String(label))] if label.as_ref() == "B")
                && matches!(properties.as_slice(), [(name, Expression::Literal(ScalarValue::Integer(9)))] if name == "id")
    ));
    Ok(())
}

#[test]
fn every_named_pattern_element_must_come_from_outer_scope() -> Result<()> {
    for pattern in [
        "(a)",
        "(n)-[r]->(a)",
        "(a)-[r]->(n)",
        "(n)<-[r {}]-(a)",
        "(n)-[r {}]-(a)",
        "(n)-[r]->()",
        "()-[r]->(n)",
        "(n)<-[r]-()",
        "(n)-[r]-()",
        "()-[r]->()",
        "()<-[r]-()",
        "()-[r]-()",
        "(n)-[r:REL]->(a {num: 5})",
        "(n)-[r:REL*0..2]->(a {num: 5})",
        "(n)-[r:REL]->(:C)<-[s:REL]-(a {num: 5})",
    ] {
        let query = format!("MATCH (n) WHERE {pattern} RETURN n");
        let error = bind_query(&query)
            .err()
            .ok_or_else(|| Error::internal(format!("unbounded predicate bound: {query}")))?;
        assert_eq!(error.code, ErrorCode::QuerySyntax, "{query}: {error:?}");
        assert!(
            error.message.contains("UndefinedVariable"),
            "{query}: {error:?}",
        );
    }
    Ok(())
}

#[test]
fn outer_relationship_roles_are_reused_but_never_redeclared() -> Result<()> {
    for query in [
        "MATCH (n)-[r:R]->(m) WHERE (n)-[r:R]->(m) RETURN r",
        "MATCH (n)-[rs:R*]->(m) WHERE (n)-[rs:R*]->(m) RETURN rs",
        "MATCH (n), (m) WHERE (n)-[:R {num: n.num}]->(m) RETURN n",
    ] {
        bind_query(query)?;
    }

    let error = bind_query("MATCH ()-[r:R]->(), (n), (m) WHERE (n)-[r:R*]->(m) RETURN n")
        .err()
        .ok_or_else(|| Error::internal("relationship role conflict unexpectedly bound"))?;
    assert_eq!(error.code, ErrorCode::QuerySyntax);
    assert!(error.message.contains("VariableTypeConflict"), "{error:?}");
    Ok(())
}

#[test]
fn named_match_paths_and_predicate_expressions_remain_distinct() -> Result<()> {
    let body = query_body("MATCH p = (n)-[:R]->(m) WHERE (n)-[:R]->(m) RETURN p")?;
    let Some(Clause::Match { patterns, .. }) = body.clauses.first() else {
        return Err(Error::internal("expected MATCH"));
    };
    assert_eq!(patterns[0].variable.as_deref(), Some("p"));
    let Some(Clause::Where(predicate)) = body.clauses.get(1) else {
        return Err(Error::internal("expected WHERE"));
    };
    assert!(is_pattern_predicate(predicate));
    bind_query("MATCH p = (n)-[:R]->(m) WHERE (n)-[:R]->(m) RETURN p")?;

    let error = bind_query("MATCH (n) WHERE p = (n)-[:R]->() RETURN n")
        .err()
        .ok_or_else(|| Error::internal("undefined equality source unexpectedly became a path"))?;
    assert_eq!(error.code, ErrorCode::QuerySyntax);
    assert!(error.message.contains("UndefinedVariable"), "{error:?}");
    Ok(())
}

#[test]
fn node_values_and_non_predicate_contexts_keep_exact_compile_errors() -> Result<()> {
    let error = bind_query("MATCH (n) WHERE (n) RETURN n")
        .err()
        .ok_or_else(|| Error::internal("node value unexpectedly accepted as a predicate"))?;
    assert_eq!(error.code, ErrorCode::QuerySyntax);
    assert!(error.message.contains("InvalidArgumentType"), "{error:?}");

    for query in [
        "MATCH (n) RETURN (n)-[]->()",
        "MATCH (n) WITH (n)-[]->() AS x RETURN x",
        "MATCH (n) SET n.prop = head(nodes(head((n)-[:REL]->()))).foo",
    ] {
        let error = parse(query)
            .err()
            .ok_or_else(|| Error::internal(format!("misplaced pattern parsed: {query}")))?;
        assert_eq!(error.code, ErrorCode::QuerySyntax, "{query}: {error:?}");
        assert!(
            error.message.contains("UnexpectedSyntax"),
            "{query}: {error:?}"
        );
    }

    let body = query_body("MATCH (n) WHERE NOT (n)-[:R]-() AND (n)-[:S]-() RETURN n")?;
    let Some(Clause::Where(Expression::Binary {
        operation: BinaryOperator::And,
        left,
        right,
    })) = body.clauses.get(1)
    else {
        return Err(Error::internal("expected conjunction"));
    };
    assert!(matches!(
        left.as_ref(),
        Expression::Unary {
            operation: UnaryOperator::Not,
            operand,
        } if is_pattern_predicate(operand)
    ));
    assert!(is_pattern_predicate(right));
    Ok(())
}
