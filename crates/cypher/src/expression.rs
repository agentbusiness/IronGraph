//! Shared expression classification used by semantic binding and execution.

use std::collections::BTreeSet;

use crate::{Error, ErrorCode, Result};

use super::{Expression, MapProjectionItem};

pub fn is_aggregate_function(name: &[String]) -> bool {
    matches!(
        name.join(".").to_ascii_lowercase().as_str(),
        "count"
            | "sum"
            | "avg"
            | "min"
            | "max"
            | "collect"
            | "variance"
            | "variancep"
            | "stdev"
            | "stdevp"
            | "percentilecont"
            | "percentiledisc"
    )
}

/// True when `expression` reads any property whose name satisfies `matches`.
///
/// The walk covers every nested expression, so a property read inside a function argument, a list,
/// a `CASE` arm or a comparison is found as readily as a bare projection.
pub fn reads_property_where(
    expression: &Expression,
    matches: &(impl Fn(&str) -> bool + Copy),
) -> bool {
    if let Expression::Property(_, name) = expression
        && matches(name)
    {
        return true;
    }
    visit_children(expression, |child| reads_property_where(child, matches))
}

pub fn contains_aggregate(expression: &Expression) -> bool {
    visit_children(expression, contains_aggregate)
        || matches!(expression, Expression::Function { name, .. } if is_aggregate_function(name))
}

/// True if an aggregate call appears inside a `Map` or `List` literal anywhere in `expression`
/// (e.g. `{name: count(b)}`, `[collect(x)]`, `{kids: collect(child.name)}`).
///
/// The native segmented-aggregation packet builder represents each aggregate as a top-level output
/// column, not as an element of a constructed container. A projection that wraps an aggregate in a
/// map/list literal therefore produces a malformed device packet — the Metal segmented graph program
/// rejects it at runtime (`CorruptStorage`, status 23). Such plans must decline the resident path and
/// fall back to host execution, which evaluates the container inline and answers correctly.
pub fn aggregate_nested_in_container_literal(expression: &Expression) -> bool {
    match expression {
        Expression::Map(_) | Expression::List(_) if contains_aggregate(expression) => true,
        _ => visit_children(expression, aggregate_nested_in_container_literal),
    }
}

pub fn validate_aggregate_nesting(expression: &Expression) -> Result<()> {
    validate_nesting(expression, false)
}

/// Rejects aggregate evaluation in a row-local context such as a predicate or comprehension
/// body. Aggregates consume a complete incoming row set, so evaluating one once per current row
/// has no well-defined Cypher meaning.
pub fn reject_aggregate_in_row_context(expression: &Expression, context: &str) -> Result<()> {
    if contains_aggregate(expression) {
        Err(Error::new(
            ErrorCode::QuerySyntax,
            format!("InvalidAggregation: aggregate functions are not allowed in {context}"),
        ))
    } else {
        Ok(())
    }
}

/// A simple grouping expression can be referenced directly from another projection expression
/// which also contains an aggregate. More complex expressions are valid grouping columns in their
/// own right, but openCypher deliberately does not decompose them into implicit grouping keys.
pub fn is_simple_grouping_expression(expression: &Expression) -> bool {
    matches!(
        expression,
        Expression::Variable(_) | Expression::Property(_, _)
    )
}

/// Validates the semantics shared by aggregating `WITH` and `RETURN` projections.
///
/// Aggregate calls themselves own their arguments. Outside those calls, row-dependent fragments
/// must be derived from a separately projected variable/property (or from `*`), while literals and
/// parameters are row-independent. This distinction is what makes `age, age + count(*)` legal but
/// rejects both `age + count(*)` and `a + b, a + b + count(*)`.
pub fn validate_projection_aggregation<'expression, 'scope>(
    expressions: impl IntoIterator<Item = &'expression Expression>,
    star_variables: impl IntoIterator<Item = &'scope str>,
) -> Result<()> {
    let expressions = expressions.into_iter().collect::<Vec<_>>();
    if !expressions
        .iter()
        .any(|expression| contains_aggregate(expression))
    {
        return Ok(());
    }

    let grouping_expressions = expressions
        .iter()
        .copied()
        .filter(|expression| {
            !contains_aggregate(expression) && is_simple_grouping_expression(expression)
        })
        .collect::<Vec<_>>();
    let star_variables = if expressions
        .iter()
        .any(|expression| matches!(expression, Expression::Star))
    {
        star_variables
            .into_iter()
            .map(str::to_owned)
            .collect::<BTreeSet<_>>()
    } else {
        BTreeSet::new()
    };
    let local_variables = BTreeSet::new();

    for expression in expressions
        .iter()
        .copied()
        .filter(|expression| contains_aggregate(expression))
    {
        validate_aggregate_argument_stability(expression)?;
        if !mixed_aggregate_expression_is_legal(
            expression,
            &grouping_expressions,
            &star_variables,
            &local_variables,
        ) {
            return Err(Error::new(
                ErrorCode::QuerySyntax,
                "AmbiguousAggregationExpression: non-aggregate fragments inside an aggregating projection must be separately projected as simple grouping keys",
            ));
        }
    }
    Ok(())
}

/// The canonical function set currently has one per-row volatile scalar: `rand()`. Query clock
/// functions are statement-stable and are intentionally not classified as volatile here.
fn is_volatile_function(name: &[String]) -> bool {
    name.len() == 1 && name[0].eq_ignore_ascii_case("rand")
}

fn contains_volatile_function(expression: &Expression) -> bool {
    matches!(expression, Expression::Function { name, .. } if is_volatile_function(name))
        || visit_children(expression, contains_volatile_function)
}

fn validate_aggregate_argument_stability(expression: &Expression) -> Result<()> {
    if let Expression::Function {
        name, arguments, ..
    } = expression
        && is_aggregate_function(name)
        && arguments.iter().any(contains_volatile_function)
    {
        return Err(Error::new(
            ErrorCode::QuerySyntax,
            "NonConstantExpression: aggregate arguments must not contain volatile expressions",
        ));
    }
    for child in children(expression) {
        validate_aggregate_argument_stability(child)?;
    }
    Ok(())
}

fn mixed_aggregate_expression_is_legal(
    expression: &Expression,
    grouping_expressions: &[&Expression],
    star_variables: &BTreeSet<String>,
    local_variables: &BTreeSet<String>,
) -> bool {
    if matches!(expression, Expression::Function { name, .. } if is_aggregate_function(name)) {
        return true;
    }
    if !contains_aggregate(expression) {
        return nonaggregate_fragment_is_legal(
            expression,
            grouping_expressions,
            star_variables,
            local_variables,
        );
    }

    let recurse = |expression: &Expression, local_variables: &BTreeSet<String>| {
        mixed_aggregate_expression_is_legal(
            expression,
            grouping_expressions,
            star_variables,
            local_variables,
        )
    };
    match expression {
        Expression::Property(value, _)
        | Expression::Unary { operand: value, .. }
        | Expression::IsNull {
            expression: value, ..
        } => recurse(value, local_variables),
        Expression::List(values)
        | Expression::Function {
            arguments: values, ..
        } => values.iter().all(|value| recurse(value, local_variables)),
        Expression::Map(values) => values
            .iter()
            .all(|(_, value)| recurse(value, local_variables)),
        Expression::MapProjection { source, items } => {
            recurse(source, local_variables)
                && items.iter().all(|item| match item {
                    MapProjectionItem::AllProperties | MapProjectionItem::Property(_) => true,
                    MapProjectionItem::Variable(variable) => variable_is_legal(
                        variable,
                        grouping_expressions,
                        star_variables,
                        local_variables,
                    ),
                    MapProjectionItem::Entry(_, value) => recurse(value, local_variables),
                })
        }
        Expression::Case {
            operand,
            alternatives,
            default,
        } => {
            operand
                .as_deref()
                .is_none_or(|value| recurse(value, local_variables))
                && alternatives.iter().all(|alternative| {
                    recurse(&alternative.when, local_variables)
                        && recurse(&alternative.then, local_variables)
                })
                && default
                    .as_deref()
                    .is_none_or(|value| recurse(value, local_variables))
        }
        Expression::ListComprehension {
            variable,
            list,
            predicate,
            projection,
        } => {
            if !recurse(list, local_variables) {
                return false;
            }
            let mut nested = local_variables.clone();
            nested.insert(variable.clone());
            predicate
                .as_deref()
                .is_none_or(|value| recurse(value, &nested))
                && projection
                    .as_deref()
                    .is_none_or(|value| recurse(value, &nested))
        }
        Expression::Reduce {
            accumulator,
            initial,
            variable,
            list,
            expression,
        } => {
            if !recurse(initial, local_variables) || !recurse(list, local_variables) {
                return false;
            }
            let mut nested = local_variables.clone();
            nested.insert(accumulator.clone());
            nested.insert(variable.clone());
            recurse(expression, &nested)
        }
        Expression::ListPredicate {
            variable,
            list,
            predicate,
            ..
        } => {
            if !recurse(list, local_variables) {
                return false;
            }
            let mut nested = local_variables.clone();
            nested.insert(variable.clone());
            recurse(predicate, &nested)
        }
        Expression::Binary { left, right, .. }
        | Expression::Index {
            expression: left,
            index: right,
        } => recurse(left, local_variables) && recurse(right, local_variables),
        Expression::Slice {
            expression,
            start,
            end,
        } => {
            recurse(expression, local_variables)
                && start
                    .as_deref()
                    .is_none_or(|value| recurse(value, local_variables))
                && end
                    .as_deref()
                    .is_none_or(|value| recurse(value, local_variables))
        }
        Expression::Literal(_)
        | Expression::Parameter(_)
        | Expression::Variable(_)
        | Expression::ExistentialSubquery(_)
        | Expression::Star => true,
    }
}

fn nonaggregate_fragment_is_legal(
    expression: &Expression,
    grouping_expressions: &[&Expression],
    star_variables: &BTreeSet<String>,
    local_variables: &BTreeSet<String>,
) -> bool {
    if grouping_expressions
        .iter()
        .any(|grouping| **grouping == *expression)
    {
        return true;
    }

    let recurse = |expression: &Expression, local_variables: &BTreeSet<String>| {
        nonaggregate_fragment_is_legal(
            expression,
            grouping_expressions,
            star_variables,
            local_variables,
        )
    };
    match expression {
        Expression::Literal(_)
        | Expression::Parameter(_)
        | Expression::ExistentialSubquery(_)
        | Expression::Star => true,
        Expression::Variable(variable) => variable_is_legal(
            variable,
            grouping_expressions,
            star_variables,
            local_variables,
        ),
        Expression::Property(value, _)
        | Expression::Unary { operand: value, .. }
        | Expression::IsNull {
            expression: value, ..
        } => recurse(value, local_variables),
        Expression::List(values) => values.iter().all(|value| recurse(value, local_variables)),
        Expression::Map(values) => values
            .iter()
            .all(|(_, value)| recurse(value, local_variables)),
        Expression::MapProjection { source, items } => {
            recurse(source, local_variables)
                && items.iter().all(|item| match item {
                    MapProjectionItem::AllProperties | MapProjectionItem::Property(_) => true,
                    MapProjectionItem::Variable(variable) => variable_is_legal(
                        variable,
                        grouping_expressions,
                        star_variables,
                        local_variables,
                    ),
                    MapProjectionItem::Entry(_, value) => recurse(value, local_variables),
                })
        }
        Expression::Case {
            operand,
            alternatives,
            default,
        } => {
            operand
                .as_deref()
                .is_none_or(|value| recurse(value, local_variables))
                && alternatives.iter().all(|alternative| {
                    recurse(&alternative.when, local_variables)
                        && recurse(&alternative.then, local_variables)
                })
                && default
                    .as_deref()
                    .is_none_or(|value| recurse(value, local_variables))
        }
        Expression::ListComprehension {
            variable,
            list,
            predicate,
            projection,
        } => {
            if !recurse(list, local_variables) {
                return false;
            }
            let mut nested = local_variables.clone();
            nested.insert(variable.clone());
            predicate
                .as_deref()
                .is_none_or(|value| recurse(value, &nested))
                && projection
                    .as_deref()
                    .is_none_or(|value| recurse(value, &nested))
        }
        Expression::Reduce {
            accumulator,
            initial,
            variable,
            list,
            expression,
        } => {
            if !recurse(initial, local_variables) || !recurse(list, local_variables) {
                return false;
            }
            let mut nested = local_variables.clone();
            nested.insert(accumulator.clone());
            nested.insert(variable.clone());
            recurse(expression, &nested)
        }
        Expression::ListPredicate {
            variable,
            list,
            predicate,
            ..
        } => {
            if !recurse(list, local_variables) {
                return false;
            }
            let mut nested = local_variables.clone();
            nested.insert(variable.clone());
            recurse(predicate, &nested)
        }
        Expression::Function {
            name, arguments, ..
        } => {
            !is_volatile_function(name)
                && arguments
                    .iter()
                    .all(|value| recurse(value, local_variables))
        }
        Expression::Binary { left, right, .. }
        | Expression::Index {
            expression: left,
            index: right,
        } => recurse(left, local_variables) && recurse(right, local_variables),
        Expression::Slice {
            expression,
            start,
            end,
        } => {
            recurse(expression, local_variables)
                && start
                    .as_deref()
                    .is_none_or(|value| recurse(value, local_variables))
                && end
                    .as_deref()
                    .is_none_or(|value| recurse(value, local_variables))
        }
    }
}

fn variable_is_legal(
    variable: &str,
    grouping_expressions: &[&Expression],
    star_variables: &BTreeSet<String>,
    local_variables: &BTreeSet<String>,
) -> bool {
    local_variables.contains(variable)
        || star_variables.contains(variable)
        || grouping_expressions.iter().any(
            |grouping| matches!(grouping, Expression::Variable(grouped) if grouped == variable),
        )
}

fn validate_nesting(expression: &Expression, inside_aggregate: bool) -> Result<()> {
    let aggregate =
        matches!(expression, Expression::Function { name, .. } if is_aggregate_function(name));
    if aggregate && inside_aggregate {
        return Err(Error::new(
            ErrorCode::QueryType,
            "aggregate functions cannot be nested",
        ));
    }
    for child in children(expression) {
        validate_nesting(child, inside_aggregate || aggregate)?;
    }
    Ok(())
}

fn visit_children(expression: &Expression, predicate: impl Fn(&Expression) -> bool + Copy) -> bool {
    children(expression).into_iter().any(predicate)
}

fn children(expression: &Expression) -> Vec<&Expression> {
    match expression {
        Expression::Property(value, _) | Expression::Unary { operand: value, .. } => vec![value],
        Expression::List(values) => values.iter().collect(),
        Expression::Map(values) => values.iter().map(|(_, value)| value).collect(),
        Expression::MapProjection { source, items } => std::iter::once(source.as_ref())
            .chain(items.iter().filter_map(|item| match item {
                MapProjectionItem::Entry(_, value) => Some(value),
                MapProjectionItem::AllProperties
                | MapProjectionItem::Property(_)
                | MapProjectionItem::Variable(_) => None,
            }))
            .collect(),
        Expression::Case {
            operand,
            alternatives,
            default,
        } => operand
            .iter()
            .map(Box::as_ref)
            .chain(
                alternatives
                    .iter()
                    .flat_map(|alternative| [&alternative.when, &alternative.then]),
            )
            .chain(default.iter().map(Box::as_ref))
            .collect(),
        Expression::ListComprehension {
            list,
            predicate,
            projection,
            ..
        } => std::iter::once(list.as_ref())
            .chain(predicate.iter().map(Box::as_ref))
            .chain(projection.iter().map(Box::as_ref))
            .collect(),
        Expression::Reduce {
            initial,
            list,
            expression,
            ..
        } => vec![initial, list, expression],
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
        } => vec![list, predicate],
        Expression::Function { arguments, .. } => arguments.iter().collect(),
        Expression::IsNull { expression, .. } => vec![expression],
        Expression::Slice {
            expression,
            start,
            end,
        } => std::iter::once(expression.as_ref())
            .chain(start.iter().map(Box::as_ref))
            .chain(end.iter().map(Box::as_ref))
            .collect(),
        Expression::Literal(_)
        | Expression::Parameter(_)
        | Expression::Variable(_)
        | Expression::ExistentialSubquery(_)
        | Expression::Star => Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aggregate_classification_is_recursive_and_rejects_nesting() {
        let expression = Expression::Binary {
            left: Box::new(Expression::Function {
                name: vec!["sum".to_owned()],
                distinct: false,
                arguments: vec![Expression::Variable("x".to_owned())],
            }),
            operation: super::super::BinaryOperator::Add,
            right: Box::new(Expression::integer(1)),
        };
        assert!(contains_aggregate(&expression));
        assert!(validate_aggregate_nesting(&expression).is_ok());

        let nested = Expression::Function {
            name: vec!["sum".to_owned()],
            distinct: false,
            arguments: vec![expression],
        };
        assert!(validate_aggregate_nesting(&nested).is_err());
    }

    #[test]
    fn container_nested_aggregate_is_detected_but_top_level_aggregate_is_not() {
        let count_b = || Expression::Function {
            name: vec!["count".to_owned()],
            distinct: false,
            arguments: vec![Expression::Variable("b".to_owned())],
        };

        // `{name: count(b)}` and `[collect(x)]` wrap the aggregate in a container literal — the
        // native segmented-aggregation packet cannot represent them, so they must be detected and
        // declined to host execution.
        let map_literal = Expression::Map(vec![("name".to_owned(), count_b())]);
        assert!(aggregate_nested_in_container_literal(&map_literal));
        let list_literal = Expression::List(vec![count_b()]);
        assert!(aggregate_nested_in_container_literal(&list_literal));
        // Nested one level deeper inside another map is still detected.
        let deep = Expression::Map(vec![(
            "outer".to_owned(),
            Expression::List(vec![count_b()]),
        )]);
        assert!(aggregate_nested_in_container_literal(&deep));

        // A bare or arithmetic top-level aggregate is a normal segmented-aggregation output column
        // and must NOT be declined.
        assert!(!aggregate_nested_in_container_literal(&count_b()));
        let arithmetic = Expression::Binary {
            left: Box::new(count_b()),
            operation: super::super::BinaryOperator::Add,
            right: Box::new(Expression::integer(1)),
        };
        assert!(!aggregate_nested_in_container_literal(&arithmetic));
        // A container literal with no aggregate inside is untouched.
        let plain_map = Expression::Map(vec![(
            "name".to_owned(),
            Expression::Variable("b".to_owned()),
        )]);
        assert!(!aggregate_nested_in_container_literal(&plain_map));
    }
}
