// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use irongraph::{
    Error, ErrorCode, Result, ScalarValue,
    cypher::{BindCapabilities, Clause, Expression, Statement, bind, parse},
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

fn query_error(query: &str) -> Result<Error> {
    match bind_query(query) {
        Ok(()) => Err(Error::internal(format!(
            "query unexpectedly passed semantic binding: {query}"
        ))),
        Err(error) => Ok(error),
    }
}

fn return_expression(query: &str) -> Result<Expression> {
    let query = parse(query)?;
    let Statement::Query(body) = query.statement else {
        return Err(Error::internal("expected a query statement"));
    };
    body.clauses
        .iter()
        .find_map(|clause| match clause {
            Clause::Return(projection) => projection.items.first(),
            _ => None,
        })
        .map(|item| item.expression.clone())
        .ok_or_else(|| Error::internal("expected a RETURN projection"))
}

fn query_contains_pattern_comprehension(query: &str) -> Result<bool> {
    let query = parse(query)?;
    let Statement::Query(body) = query.statement else {
        return Err(Error::internal("expected a query statement"));
    };
    Ok(body.clauses.iter().any(|clause| match clause {
        Clause::With(projection) | Clause::Return(projection) => projection
            .items
            .iter()
            .any(|item| find_pattern_comprehension(&item.expression).is_some()),
        _ => false,
    }))
}

fn is_pattern_comprehension(expression: &Expression) -> bool {
    matches!(
        expression,
        Expression::Function { name, distinct: false, arguments }
            if arguments.len() == 7
                && matches!(name.as_slice(), [intrinsic]
                    if intrinsic.starts_with('\0')
                        && intrinsic.ends_with(".pattern_comprehension"))
    )
}

fn find_pattern_comprehension(expression: &Expression) -> Option<&Expression> {
    if is_pattern_comprehension(expression) {
        return Some(expression);
    }
    match expression {
        Expression::Property(value, _)
        | Expression::Unary { operand: value, .. }
        | Expression::IsNull {
            expression: value, ..
        } => find_pattern_comprehension(value),
        Expression::List(values)
        | Expression::Function {
            arguments: values, ..
        } => values.iter().find_map(find_pattern_comprehension),
        Expression::Map(values) => values
            .iter()
            .find_map(|(_, value)| find_pattern_comprehension(value)),
        Expression::MapProjection { source, items } => {
            find_pattern_comprehension(source).or_else(|| {
                items.iter().find_map(|item| match item {
                    irongraph::cypher::MapProjectionItem::Entry(_, value) => {
                        find_pattern_comprehension(value)
                    }
                    _ => None,
                })
            })
        }
        Expression::Case {
            operand,
            alternatives,
            default,
        } => operand
            .as_deref()
            .and_then(find_pattern_comprehension)
            .or_else(|| {
                alternatives.iter().find_map(|alternative| {
                    find_pattern_comprehension(&alternative.when)
                        .or_else(|| find_pattern_comprehension(&alternative.then))
                })
            })
            .or_else(|| default.as_deref().and_then(find_pattern_comprehension)),
        Expression::ListComprehension {
            list,
            predicate,
            projection,
            ..
        } => find_pattern_comprehension(list)
            .or_else(|| predicate.as_deref().and_then(find_pattern_comprehension))
            .or_else(|| projection.as_deref().and_then(find_pattern_comprehension)),
        Expression::Reduce {
            initial,
            list,
            expression,
            ..
        } => find_pattern_comprehension(initial)
            .or_else(|| find_pattern_comprehension(list))
            .or_else(|| find_pattern_comprehension(expression)),
        Expression::ListPredicate {
            list, predicate, ..
        }
        | Expression::Binary {
            left: list,
            right: predicate,
            ..
        }
        | Expression::Index {
            expression: list,
            index: predicate,
        } => find_pattern_comprehension(list).or_else(|| find_pattern_comprehension(predicate)),
        Expression::Slice {
            expression,
            start,
            end,
        } => find_pattern_comprehension(expression)
            .or_else(|| start.as_deref().and_then(find_pattern_comprehension))
            .or_else(|| end.as_deref().and_then(find_pattern_comprehension)),
        Expression::Literal(_)
        | Expression::Parameter(_)
        | Expression::Variable(_)
        | Expression::ExistentialSubquery(_)
        | Expression::Star => None,
    }
}

#[test]
fn parses_and_binds_all_eleven_pattern2_feature_query_shapes() -> Result<()> {
    for query in [
        "MATCH (n) RETURN [p = (n)-->() | p] AS list",
        "MATCH (n:A) RETURN [p = (n)-->(:B) | p] AS list",
        "MATCH (a:A), (b:B) RETURN [p = (a)-->(b) | p] AS list",
        "MATCH (n) RETURN [(n)-[:T]->(b) | b.name] AS list",
        "MATCH (n) RETURN [(n)-[r:T]->() | r.name] AS list",
        "MATCH (n:A) RETURN count([p = (n)-[:HAS]->() | p]) AS c",
        "MATCH p = (n:X)-->() RETURN n, [x IN nodes(p) | size([(x)-->(:Y) | 1])] AS list",
        "MATCH (n)-->(b) WITH [p = (n)-->() | p] AS ps, count(b) AS c RETURN ps, c",
        "MATCH (a:A), (b:B) WITH [p = (a)-[*]->(b) | p] AS paths, count(a) AS c RETURN paths, c",
        "MATCH (n:A) RETURN [p = (n)-[:HAS]->() | p] AS ps",
        "MATCH (liker) RETURN [p = (liker)--() | p] AS isNew ORDER BY liker.time",
    ] {
        assert!(
            query_contains_pattern_comprehension(query)?,
            "pattern comprehension was not preserved for `{query}`"
        );
        bind_query(query).map_err(|error| {
            Error::internal(format!(
                "valid Pattern2 query failed to bind: {query}: {error}"
            ))
        })?;
    }
    Ok(())
}

#[test]
fn envelope_preserves_path_filter_direction_types_range_properties_and_projection() -> Result<()> {
    let expression = return_expression(
        "MATCH (a) RETURN [p = (a:A {id: 1})<-[r:R|S*0..2 {weight: a.id}]-(b:B {id: 2}) WHERE b.id = 2 | {path: p, node: b, rels: r}] AS values",
    )?;
    let Expression::Function {
        name,
        distinct: false,
        arguments,
    } = expression
    else {
        return Err(Error::internal("expected pattern-comprehension envelope"));
    };
    assert!(
        matches!(name.as_slice(), [name] if name.starts_with('\0') && name.ends_with(".pattern_comprehension"))
    );
    let [
        path_name,
        selector,
        mode,
        start,
        steps,
        predicate,
        projection,
    ] = arguments.as_slice()
    else {
        return Err(Error::internal("malformed pattern-comprehension envelope"));
    };
    assert!(
        matches!(path_name, Expression::Literal(ScalarValue::String(name)) if name.as_ref() == "p")
    );
    assert!(matches!(
        selector,
        Expression::List(values)
            if matches!(values.as_slice(), [Expression::Literal(ScalarValue::Integer(0)), Expression::Literal(ScalarValue::Null), Expression::Literal(ScalarValue::Boolean(false))])
    ));
    assert!(matches!(mode, Expression::Literal(ScalarValue::Integer(0))));

    let Expression::Function {
        arguments: start, ..
    } = start
    else {
        return Err(Error::internal("expected encoded start node"));
    };
    assert!(matches!(
        start.as_slice(),
        [Expression::Literal(ScalarValue::String(variable)), Expression::List(labels), Expression::Literal(ScalarValue::Boolean(true)), Expression::Map(properties)]
            if variable.as_ref() == "a"
                && matches!(labels.as_slice(), [Expression::Literal(ScalarValue::String(label))] if label.as_ref() == "A")
                && matches!(properties.as_slice(), [(name, Expression::Literal(ScalarValue::Integer(1)))] if name == "id")
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
        Expression::Function { arguments: end, .. },
    ] = step.as_slice()
    else {
        return Err(Error::internal(
            "expected encoded relationship and end node",
        ));
    };
    let [
        variable,
        Expression::List(types),
        direction,
        variable_length,
        minimum,
        maximum,
        Expression::Map(properties),
    ] = relationship.as_slice()
    else {
        return Err(Error::internal("malformed encoded relationship"));
    };
    assert!(
        matches!(variable, Expression::Literal(ScalarValue::String(name)) if name.as_ref() == "r")
    );
    assert_eq!(types, &[Expression::string("R"), Expression::string("S")]);
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
        [(name, Expression::Property(source, property))]
            if name == "weight"
                && property == "id"
                && matches!(source.as_ref(), Expression::Variable(variable) if variable == "a")
    ));
    assert!(matches!(
        end.as_slice(),
        [Expression::Literal(ScalarValue::String(variable)), Expression::List(labels), Expression::Literal(ScalarValue::Boolean(true)), Expression::Map(properties)]
            if variable.as_ref() == "b"
                && matches!(labels.as_slice(), [Expression::Literal(ScalarValue::String(label))] if label.as_ref() == "B")
                && matches!(properties.as_slice(), [(name, Expression::Literal(ScalarValue::Integer(2)))] if name == "id")
    ));
    assert!(matches!(
        predicate,
        Expression::List(values)
            if matches!(values.as_slice(), [Expression::Binary { .. }])
    ));
    assert!(matches!(
        projection,
        Expression::Map(entries)
            if entries.len() == 3
                && matches!(&entries[0], (name, Expression::Variable(variable)) if name == "path" && variable == "p")
                && matches!(&entries[1], (name, Expression::Variable(variable)) if name == "node" && variable == "b")
                && matches!(&entries[2], (name, Expression::Variable(variable)) if name == "rels" && variable == "r")
    ));
    bind_query(
        "MATCH (a) RETURN [p = (a:A {id: 1})<-[r:R|S*0..2 {weight: a.id}]-(b:B {id: 2}) WHERE b.id = 2 | {path: p, node: b, rels: r}] AS values",
    )?;
    Ok(())
}

#[test]
fn outer_graph_variables_correlate_and_local_graph_variables_are_projection_visible() -> Result<()>
{
    for query in [
        "MATCH (n)-[r:T]->(b) RETURN [(n)-[r:T]->(b) | r] AS relationships",
        "MATCH (n) RETURN [p = (n)-[r:T]->(b) | {path: p, relationship: r, node: b, outer: n}] AS values",
        "MATCH (n) RETURN [(n)-[rs:T*0..2]->(b) | [rs, b]] AS values",
        "MATCH (n) RETURN [(n)-[r:T]->(b) WHERE b.name IS NOT NULL | r.name] AS names",
        "MATCH p = (n:X)-->() RETURN [x IN nodes(p) | size([(x)-->(y:Y) | y])] AS values",
    ] {
        bind_query(query).map_err(|error| {
            Error::internal(format!(
                "valid correlated/local pattern-comprehension scope failed: {query}: {error}"
            ))
        })?;
    }
    Ok(())
}

#[test]
fn node_relationship_and_path_locals_do_not_leak_out_of_the_comprehension() -> Result<()> {
    for (query, variable) in [
        ("MATCH (n) WITH [(n)-[:T]->(b) | b] AS nodes RETURN b", "b"),
        (
            "MATCH (n) WITH [(n)-[r:T]->() | r] AS relationships RETURN r",
            "r",
        ),
        ("MATCH (n) WITH [p = (n)-->() | p] AS paths RETURN p", "p"),
        ("MATCH (n) RETURN [(n)-->(b) | b] AS nodes, b", "b"),
        ("MATCH (n) RETURN [(n)-->() | missing] AS values", "missing"),
    ] {
        let error = query_error(query)?;
        assert_eq!(error.code, ErrorCode::QuerySyntax, "{query}: {error:?}");
        assert!(
            error.message.contains("UndefinedVariable"),
            "{query}: {error:?}"
        );
        assert!(error.message.contains(variable), "{query}: {error:?}");
    }

    bind_query("MATCH (n) WITH [(n)-->(b) | b] AS nodes, n RETURN n")?;
    Ok(())
}

#[test]
fn correlated_variables_keep_their_graph_roles() -> Result<()> {
    for query in [
        "MATCH (n), (b) RETURN [(n)-[b:T]->() | b] AS values",
        "MATCH ()-[r:T]->(), (n) RETURN [(n)-->(r) | r] AS values",
        "MATCH ()-[r:T]->(), (n) RETURN [(n)-[r:T*]->() | r] AS values",
    ] {
        let error = query_error(query)?;
        assert_eq!(error.code, ErrorCode::QuerySyntax, "{query}: {error:?}");
        assert!(
            error.message.contains("VariableTypeConflict"),
            "{query}: {error:?}"
        );
    }
    Ok(())
}

#[test]
fn pattern_comprehension_grammar_stays_distinct_from_lists_and_pattern_predicates() -> Result<()> {
    let list = return_expression("RETURN [(1) - 1] AS values")?;
    assert!(matches!(list, Expression::List(_)));
    assert!(!is_pattern_comprehension(&list));
    bind_query("RETURN [x IN [1, 2] | x + 1] AS values")?;

    let query = parse("MATCH (n) WHERE (n)-->() RETURN [(n)-->() | 1] AS values")?;
    let Statement::Query(body) = query.statement else {
        return Err(Error::internal("expected query body"));
    };
    let Some(Clause::Where(Expression::Function { name, .. })) = body.clauses.get(1) else {
        return Err(Error::internal(
            "expected distinct pattern-predicate envelope",
        ));
    };
    assert!(matches!(name.as_slice(), [name] if name.ends_with(".pattern_predicate")));
    let Some(Clause::Return(projection)) = body.clauses.get(2) else {
        return Err(Error::internal("expected RETURN clause"));
    };
    assert!(is_pattern_comprehension(&projection.items[0].expression));

    for malformed in [
        "MATCH (n) RETURN [(n)-->()] AS values",
        "MATCH (n) RETURN [(n)-->() |] AS values",
        "MATCH (n) RETURN [p = (n)-->() p] AS values",
    ] {
        let error = parse(malformed)
            .err()
            .ok_or_else(|| Error::internal(format!("malformed query parsed: {malformed}")))?;
        assert_eq!(error.code, ErrorCode::QuerySyntax, "{malformed}: {error:?}");
    }
    Ok(())
}
