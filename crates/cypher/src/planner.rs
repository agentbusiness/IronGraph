//! Backend-neutral vectorized physical operator IR.

use std::collections::{BTreeMap, BTreeSet};

use crate::{Error, ErrorCode, Layer, Result, graph::LayerMask};

use super::{
    BoundQuery, DependencyStamp,
    ast::*,
    expression::{contains_aggregate, is_aggregate_function, is_simple_grouping_expression},
};

#[derive(Clone, Debug)]
pub struct PhysicalPlan {
    pub project: Option<String>,
    pub read_layers: LayerMask,
    pub write_layer: Layer,
    pub at_time: Option<Expression>,
    pub operators: Vec<PhysicalOperator>,
    pub unions: Vec<(bool, Vec<PhysicalOperator>)>,
    pub read_only: bool,
    pub dependencies: Vec<DependencyStamp>,
}

/// Read-only physical operators owned by one correlated `EXISTS { ... }` expression.
#[derive(Clone, Debug)]
pub struct ExistentialSubqueryPlan {
    pub operators: Vec<PhysicalOperator>,
    pub unions: Vec<(bool, Vec<PhysicalOperator>)>,
}

/// Compact identity of the source `MATCH` or `OPTIONAL MATCH` clause which owns a pattern scan.
///
/// All relationship occurrences in one group share openCypher's `DifferentRelationships`
/// uniqueness domain. A later clause always receives a new group, even when its scans are adjacent
/// in the physical plan, so relationships bound by an earlier clause may be reused.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct MatchGroupId(u32);

impl MatchGroupId {
    /// Stable compact value used by sealed resident-query ABIs.
    pub const fn get(self) -> u32 {
        self.0
    }
}

#[derive(Clone, Debug, PartialEq)]
pub enum PhysicalOperator {
    ScanPattern {
        match_group: MatchGroupId,
        optional: bool,
        pattern: Pattern,
        access: ScanAccessPath,
    },
    /// Side-effect-free observation point inserted only before any client-visible output. The
    /// executor may restart the read-only plan once when actual cardinality is badly misestimated.
    CardinalityCheckpoint {
        pattern_key: [u8; 32],
        estimated_rows: u64,
    },
    /// Generic-join lowering for a fixed-hop cyclic pattern component. Node domains and sorted
    /// adjacency candidates are intersected before relationship rows are materialized.
    CyclicMultiwayJoin {
        match_group: MatchGroupId,
        patterns: Vec<(Pattern, ScanAccessPath)>,
    },
    Filter(Expression),
    Unwind {
        expression: Expression,
        variable: String,
    },
    CreatePattern(Pattern),
    MergePattern {
        pattern: Pattern,
        on_create: Vec<SetItem>,
        on_match: Vec<SetItem>,
    },
    Set(Vec<SetItem>),
    Remove(Vec<RemoveItem>),
    Delete {
        detach: bool,
        expressions: Vec<Expression>,
    },
    Project {
        keep_scope: bool,
        projection: Projection,
    },
    Sort(Vec<SortItem>),
    /// Stable bounded sort selected only when an adjacent literal LIMIT preserves semantics.
    TopK {
        items: Vec<SortItem>,
        limit: usize,
    },
    Skip(Expression),
    Limit(Expression),
    TemporalHistory(HistoryClause),
    TemporalWindow(WindowClause),
    TemporalHistoryWindow {
        history: HistoryClause,
        window: WindowClause,
    },
    VectorSearch {
        search: SearchClause,
        access: VectorAccessPath,
    },
    BuiltinProcedure(CallClause),
    Finish,
    Administrative(Statement),
}

/// Optimizer-owned physical access for one native embedding search. `Unspecified` exists only
/// between logical lowering and optimization; execution treats it as the exact correctness path.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum VectorAccessPath {
    Unspecified,
    Exact {
        filtered_rows: u64,
        query_batch: u32,
        scratch_bytes: u64,
    },
    IvfPq {
        filtered_rows: u64,
        query_batch: u32,
        candidate_budget: u32,
        scratch_bytes: u64,
    },
}

impl VectorAccessPath {
    #[must_use]
    pub const fn uses_ivf_pq(self) -> bool {
        matches!(self, Self::IvfPq { .. })
    }
}

/// Device-neutral access decision. Runtime lifecycle checks always retain a canonical scan
/// fallback, so a derived index becoming unavailable cannot change query results.
#[derive(Clone, Debug, PartialEq)]
pub enum ScanAccessPath {
    Unspecified,
    BoundVariable,
    AllNodes,
    Label {
        label: String,
    },
    StableId {
        variable: String,
        value: Expression,
    },
    EqualityIndex {
        name: String,
        label: String,
        values: Vec<(String, Expression)>,
        estimated_rows: u64,
    },
    ResidentProperty {
        variable: String,
        property: String,
        operation: BinaryOperator,
        value: Expression,
        estimated_rows: u64,
    },
}

pub fn plan(bound: BoundQuery) -> Result<PhysicalPlan> {
    let BoundQuery {
        query,
        read_only,
        dependencies,
    } = bound;
    let Query {
        project,
        read_layers,
        write_layer,
        at_time,
        statement,
    } = query;
    let (operators, unions) = match statement {
        Statement::Query(body) => {
            validate_union_schemas(&body)?;
            validate_order_by_projections(&body.clauses)?;
            // `ORDER BY`, `SKIP`, and `LIMIT` are parsed as their own clauses after RETURN.
            // A query has a client result whenever it contains that terminal projection, not
            // only when RETURN happens to be the final physical clause.
            let returns_rows = body
                .clauses
                .iter()
                .any(|clause| matches!(clause, Clause::Return(_)));
            let mut operators = lower(&body.clauses);
            // A standalone updating query has no implicit result projection. Its internal
            // variable bindings remain available to later clauses, but never become client
            // columns unless an explicit RETURN closes the statement.
            if !read_only && !returns_rows {
                operators.push(PhysicalOperator::Finish);
            }
            let unions = body
                .unions
                .into_iter()
                .map(|branch| {
                    validate_order_by_projections(&branch.body)?;
                    Ok((branch.all, lower(&branch.body)))
                })
                .collect::<Result<Vec<_>>>()?;
            (operators, unions)
        }
        statement => (
            vec![PhysicalOperator::Administrative(statement)],
            Vec::new(),
        ),
    };
    if let Some(at_time) = &at_time {
        validate_expression_existential_subqueries(at_time, 0)?;
    }
    validate_operator_existential_subqueries(&operators, 0)?;
    for (_, branch) in &unions {
        validate_operator_existential_subqueries(branch, 0)?;
    }
    Ok(PhysicalPlan {
        project,
        read_layers,
        write_layer,
        at_time,
        operators,
        unions,
        read_only,
        dependencies,
    })
}

/// Lower and validate a correlated existential body without constructing a second query engine.
/// The executor supplies the outer row and runs these operators against its current statement
/// state, so graph snapshots and statement-local overlays remain shared.
pub fn plan_existential_subquery(
    subquery: &ExistentialSubquery,
    depth: usize,
) -> Result<ExistentialSubqueryPlan> {
    stacker::maybe_grow(256 * 1024, 2 * 1024 * 1024, || {
        plan_existential_subquery_inner(subquery, depth)
    })
}

fn plan_existential_subquery_inner(
    subquery: &ExistentialSubquery,
    depth: usize,
) -> Result<ExistentialSubqueryPlan> {
    if subquery.body.clauses.iter().any(clause_updates_graph)
        || subquery
            .body
            .unions
            .iter()
            .flat_map(|branch| &branch.body)
            .any(clause_updates_graph)
    {
        return Err(Error::new(
            ErrorCode::QuerySyntax,
            "InvalidClauseComposition: existential subqueries are read-only",
        ));
    }
    validate_union_schemas(&subquery.body)?;
    validate_order_by_projections(&subquery.body.clauses)?;
    let operators = lower(&subquery.body.clauses);
    let unions = subquery
        .body
        .unions
        .iter()
        .map(|branch| {
            validate_order_by_projections(&branch.body)?;
            Ok((branch.all, lower(&branch.body)))
        })
        .collect::<Result<Vec<_>>>()?;
    validate_operator_existential_subqueries(&operators, depth)?;
    for (_, branch) in &unions {
        validate_operator_existential_subqueries(branch, depth)?;
    }
    Ok(ExistentialSubqueryPlan { operators, unions })
}

fn clause_updates_graph(clause: &Clause) -> bool {
    matches!(
        clause,
        Clause::Create(_)
            | Clause::Merge { .. }
            | Clause::Set(_)
            | Clause::Remove(_)
            | Clause::Delete { .. }
    )
}

fn validate_operator_existential_subqueries(
    operators: &[PhysicalOperator],
    depth: usize,
) -> Result<()> {
    for operator in operators {
        match operator {
            PhysicalOperator::ScanPattern { pattern, .. }
            | PhysicalOperator::CreatePattern(pattern) => {
                validate_pattern_existential_subqueries(pattern, depth)?;
            }
            PhysicalOperator::CyclicMultiwayJoin { patterns, .. } => {
                for (pattern, _) in patterns {
                    validate_pattern_existential_subqueries(pattern, depth)?;
                }
            }
            PhysicalOperator::Filter(expression)
            | PhysicalOperator::Unwind { expression, .. }
            | PhysicalOperator::Skip(expression)
            | PhysicalOperator::Limit(expression) => {
                validate_expression_existential_subqueries(expression, depth)?;
            }
            PhysicalOperator::MergePattern {
                pattern,
                on_create,
                on_match,
            } => {
                validate_pattern_existential_subqueries(pattern, depth)?;
                for item in on_create.iter().chain(on_match) {
                    validate_set_item_existential_subqueries(item, depth)?;
                }
            }
            PhysicalOperator::Set(items) => {
                for item in items {
                    validate_set_item_existential_subqueries(item, depth)?;
                }
            }
            PhysicalOperator::Remove(items) => {
                for item in items {
                    if let RemoveItem::Labels { labels, .. } = item {
                        for label in labels {
                            if let LabelName::Dynamic(expression)
                            | LabelName::DynamicAll(expression) = label
                            {
                                validate_expression_existential_subqueries(expression, depth)?;
                            }
                        }
                    }
                }
            }
            PhysicalOperator::Delete { expressions, .. } => {
                for expression in expressions {
                    validate_expression_existential_subqueries(expression, depth)?;
                }
            }
            PhysicalOperator::Project { projection, .. } => {
                for item in &projection.items {
                    validate_expression_existential_subqueries(&item.expression, depth)?;
                }
            }
            PhysicalOperator::Sort(items) | PhysicalOperator::TopK { items, .. } => {
                for item in items {
                    validate_expression_existential_subqueries(&item.expression, depth)?;
                }
            }
            PhysicalOperator::TemporalHistory(history) => {
                validate_expression_existential_subqueries(&history.from, depth)?;
                validate_expression_existential_subqueries(&history.to, depth)?;
            }
            PhysicalOperator::TemporalWindow(window) => {
                validate_window_existential_subqueries(window, depth)?;
            }
            PhysicalOperator::TemporalHistoryWindow { history, window } => {
                validate_expression_existential_subqueries(&history.from, depth)?;
                validate_expression_existential_subqueries(&history.to, depth)?;
                validate_window_existential_subqueries(window, depth)?;
            }
            PhysicalOperator::VectorSearch { search, .. } => {
                let input = match &search.input {
                    SearchInput::Text(expression) | SearchInput::Vector(expression) => expression,
                };
                validate_expression_existential_subqueries(input, depth)?;
                validate_expression_existential_subqueries(&search.limit, depth)?;
            }
            PhysicalOperator::BuiltinProcedure(call) => {
                for expression in &call.arguments {
                    validate_expression_existential_subqueries(expression, depth)?;
                }
                for item in &call.yields {
                    validate_expression_existential_subqueries(&item.expression, depth)?;
                }
            }
            PhysicalOperator::CardinalityCheckpoint { .. }
            | PhysicalOperator::Finish
            | PhysicalOperator::Administrative(_) => {}
        }
    }
    Ok(())
}

fn validate_pattern_existential_subqueries(pattern: &Pattern, depth: usize) -> Result<()> {
    for (_, expression) in &pattern.start.properties {
        validate_expression_existential_subqueries(expression, depth)?;
    }
    for step in &pattern.steps {
        for (_, expression) in &step.relationship.properties {
            validate_expression_existential_subqueries(expression, depth)?;
        }
        for (_, expression) in &step.node.properties {
            validate_expression_existential_subqueries(expression, depth)?;
        }
    }
    Ok(())
}

fn validate_set_item_existential_subqueries(item: &SetItem, depth: usize) -> Result<()> {
    match item {
        SetItem::Property {
            value, event_time, ..
        } => {
            validate_expression_existential_subqueries(value, depth)?;
            if let Some(event_time) = event_time {
                validate_expression_existential_subqueries(event_time, depth)?;
            }
        }
        SetItem::MergeMap { value, .. } | SetItem::ReplaceMap { value, .. } => {
            validate_expression_existential_subqueries(value, depth)?;
        }
        SetItem::Labels { labels, .. } => {
            for label in labels {
                if let LabelName::Dynamic(expression) | LabelName::DynamicAll(expression) = label {
                    validate_expression_existential_subqueries(expression, depth)?;
                }
            }
        }
    }
    Ok(())
}

fn validate_window_existential_subqueries(window: &WindowClause, depth: usize) -> Result<()> {
    validate_expression_existential_subqueries(&window.width, depth)?;
    if let Some(every) = &window.every {
        validate_expression_existential_subqueries(every, depth)?;
    }
    validate_expression_existential_subqueries(&window.event_expression, depth)?;
    if let Some(align) = &window.align {
        validate_expression_existential_subqueries(align, depth)?;
    }
    Ok(())
}

fn validate_expression_existential_subqueries(expression: &Expression, depth: usize) -> Result<()> {
    if let Some(subquery) = expression.existential_subquery_parts() {
        let _ = plan_existential_subquery(subquery, depth.saturating_add(1))?;
        return Ok(());
    }
    let validate =
        |expression: &Expression| validate_expression_existential_subqueries(expression, depth);
    match expression {
        Expression::Literal(_)
        | Expression::Parameter(_)
        | Expression::Variable(_)
        | Expression::ExistentialSubquery(_)
        | Expression::Star => {}
        Expression::Property(source, _)
        | Expression::Unary {
            operand: source, ..
        } => {
            validate(source)?;
        }
        Expression::List(values)
        | Expression::Function {
            arguments: values, ..
        } => {
            for value in values {
                validate(value)?;
            }
        }
        Expression::Map(values) => {
            for (_, value) in values {
                validate(value)?;
            }
        }
        Expression::MapProjection { source, items } => {
            validate(source)?;
            for item in items {
                if let MapProjectionItem::Entry(_, value) = item {
                    validate(value)?;
                }
            }
        }
        Expression::Case {
            operand,
            alternatives,
            default,
        } => {
            if let Some(operand) = operand {
                validate(operand)?;
            }
            for alternative in alternatives {
                validate(&alternative.when)?;
                validate(&alternative.then)?;
            }
            if let Some(default) = default {
                validate(default)?;
            }
        }
        Expression::ListComprehension {
            list,
            predicate,
            projection,
            ..
        } => {
            validate(list)?;
            if let Some(predicate) = predicate {
                validate(predicate)?;
            }
            if let Some(projection) = projection {
                validate(projection)?;
            }
        }
        Expression::Reduce {
            initial,
            list,
            expression,
            ..
        } => {
            validate(initial)?;
            validate(list)?;
            validate(expression)?;
        }
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
        } => {
            validate(list)?;
            validate(predicate)?;
        }
        Expression::IsNull { expression, .. } => validate(expression)?,
        Expression::Slice {
            expression,
            start,
            end,
        } => {
            validate(expression)?;
            if let Some(start) = start {
                validate(start)?;
            }
            if let Some(end) = end {
                validate(end)?;
            }
        }
    }
    Ok(())
}

/// UNION combines rows positionally, but Cypher exposes one shared client schema. Validate that
/// schema while every branch still has its source projection, before lowering can add hidden
/// ORDER BY columns or otherwise obscure the client-visible names.
fn validate_union_schemas(body: &QueryBody) -> Result<()> {
    let Some(first_branch) = body.unions.first() else {
        return Ok(());
    };

    let expected = terminal_return_columns(&body.clauses)?;
    for (branch_index, branch) in std::iter::once(first_branch)
        .chain(body.unions.iter().skip(1))
        .enumerate()
    {
        let actual = terminal_return_columns(&branch.body)?;
        if actual != expected {
            return Err(Error::new(
                ErrorCode::QuerySyntax,
                format!(
                    "DifferentColumnsInUnion: branch {} exposes {:?}, but the first branch exposes {:?}",
                    branch_index.saturating_add(2),
                    actual,
                    expected
                ),
            ));
        }
    }
    Ok(())
}

fn terminal_return_columns(clauses: &[Clause]) -> Result<Vec<String>> {
    let projection = clauses
        .iter()
        .rev()
        .skip_while(|clause| {
            matches!(
                clause,
                Clause::OrderBy(_) | Clause::Skip(_) | Clause::Limit(_)
            )
        })
        .next()
        .and_then(|clause| match clause {
            Clause::Return(projection) => Some(projection),
            _ => None,
        })
        .ok_or_else(|| {
            Error::new(
                ErrorCode::QuerySyntax,
                "InvalidClauseComposition: every UNION branch must end with RETURN, optionally followed by ORDER BY, SKIP, or LIMIT",
            )
        })?;

    if projection
        .items
        .iter()
        .any(|item| matches!(item.expression, Expression::Star))
    {
        return Err(Error::new(
            ErrorCode::QuerySyntax,
            "UnsupportedUnionStarProjection: RETURN * cannot be used in UNION until its exact expanded branch schema is available to the planner",
        ));
    }

    Ok(projection
        .items
        .iter()
        .enumerate()
        .map(|(index, item)| item.column_name(index))
        .collect())
}

/// Enforces the projection boundary rules which make ORDER BY deterministic after DISTINCT or
/// aggregation. A non-distinct, non-aggregating projection may still sort by its incoming scope;
/// DISTINCT and aggregation may only sort by values represented by their output grouping table.
fn validate_order_by_projections(clauses: &[Clause]) -> Result<()> {
    for pair in clauses.windows(2) {
        let [
            Clause::With(projection) | Clause::Return(projection),
            Clause::OrderBy(items),
        ] = pair
        else {
            continue;
        };

        if projection.distinct {
            for item in items {
                if !expression_is_derived_from_projection(&item.expression, projection) {
                    return Err(Error::new(
                        ErrorCode::QuerySyntax,
                        "UndefinedVariable: ORDER BY after DISTINCT may only use projected values",
                    ));
                }
            }
        }

        let projection_aggregates = projection
            .items
            .iter()
            .any(|item| contains_aggregate(&item.expression));
        for item in items {
            if !contains_aggregate(&item.expression) {
                continue;
            }
            if !projection_aggregates {
                return Err(Error::new(
                    ErrorCode::QuerySyntax,
                    "InvalidAggregation: ORDER BY cannot introduce aggregation after a non-aggregating projection",
                ));
            }
            validate_aggregate_order_expression(&item.expression, projection)?;
        }
    }
    Ok(())
}

fn expression_is_derived_from_projection(expression: &Expression, projection: &Projection) -> bool {
    if projection
        .items
        .iter()
        .any(|item| item.expression == *expression)
    {
        return true;
    }
    match expression {
        Expression::Literal(_) | Expression::Parameter(_) | Expression::Star => true,
        Expression::ExistentialSubquery(_) => false,
        Expression::Variable(variable) => projection.items.iter().any(|item| {
            item.alias.as_deref() == Some(variable)
                || matches!(&item.expression, Expression::Variable(projected) if projected == variable)
                || matches!(item.expression, Expression::Star)
        }),
        Expression::Property(value, _) => expression_is_derived_from_projection(value, projection),
        Expression::List(values) => values
            .iter()
            .all(|value| expression_is_derived_from_projection(value, projection)),
        Expression::Map(values) => values
            .iter()
            .all(|(_, value)| expression_is_derived_from_projection(value, projection)),
        Expression::MapProjection { source, items } => {
            expression_is_derived_from_projection(source, projection)
                && items.iter().all(|item| match item {
                    MapProjectionItem::AllProperties | MapProjectionItem::Property(_) => true,
                    MapProjectionItem::Variable(variable) => {
                        expression_is_derived_from_projection(
                            &Expression::Variable(variable.clone()),
                            projection,
                        )
                    }
                    MapProjectionItem::Entry(_, value) => {
                        expression_is_derived_from_projection(value, projection)
                    }
                })
        }
        Expression::Case {
            operand,
            alternatives,
            default,
        } => {
            operand.as_deref().is_none_or(|value| {
                expression_is_derived_from_projection(value, projection)
            }) && alternatives.iter().all(|alternative| {
                expression_is_derived_from_projection(&alternative.when, projection)
                    && expression_is_derived_from_projection(&alternative.then, projection)
            }) && default.as_deref().is_none_or(|value| {
                expression_is_derived_from_projection(value, projection)
            })
        }
        Expression::ListComprehension {
            list,
            predicate,
            projection: item_projection,
            ..
        } => {
            expression_is_derived_from_projection(list, projection)
                && predicate.as_deref().is_none_or(|value| {
                    expression_is_derived_from_projection(value, projection)
                })
                && item_projection.as_deref().is_none_or(|value| {
                    expression_is_derived_from_projection(value, projection)
                })
        }
        Expression::Reduce {
            initial,
            list,
            expression,
            ..
        } => {
            expression_is_derived_from_projection(initial, projection)
                && expression_is_derived_from_projection(list, projection)
                && expression_is_derived_from_projection(expression, projection)
        }
        Expression::ListPredicate {
            list, predicate, ..
        } => {
            expression_is_derived_from_projection(list, projection)
                && expression_is_derived_from_projection(predicate, projection)
        }
        Expression::Function { arguments, .. } => arguments
            .iter()
            .all(|value| expression_is_derived_from_projection(value, projection)),
        Expression::Unary { operand, .. } => {
            expression_is_derived_from_projection(operand, projection)
        }
        Expression::Binary { left, right, .. } => {
            expression_is_derived_from_projection(left, projection)
                && expression_is_derived_from_projection(right, projection)
        }
        Expression::IsNull { expression, .. } => {
            expression_is_derived_from_projection(expression, projection)
        }
        Expression::Index { expression, index } => {
            expression_is_derived_from_projection(expression, projection)
                && expression_is_derived_from_projection(index, projection)
        }
        Expression::Slice {
            expression,
            start,
            end,
        } => {
            expression_is_derived_from_projection(expression, projection)
                && start.as_deref().is_none_or(|value| {
                    expression_is_derived_from_projection(value, projection)
                })
                && end.as_deref().is_none_or(|value| {
                    expression_is_derived_from_projection(value, projection)
                })
        }
    }
}

fn validate_aggregate_order_expression(
    expression: &Expression,
    projection: &Projection,
) -> Result<()> {
    if matches!(expression, Expression::Function { name, .. } if is_aggregate_function(name)) {
        return Ok(());
    }
    if !contains_aggregate(expression) {
        return validate_nonaggregate_order_fragment(expression, projection);
    }

    let recurse =
        |expression: &Expression| validate_aggregate_order_expression(expression, projection);
    match expression {
        Expression::Case {
            operand,
            alternatives,
            default,
        } => {
            if let Some(operand) = operand {
                recurse(operand)?;
            }
            for alternative in alternatives {
                recurse(&alternative.when)?;
                recurse(&alternative.then)?;
            }
            if let Some(default) = default {
                recurse(default)?;
            }
        }
        Expression::ListComprehension {
            list,
            predicate,
            projection,
            ..
        } => {
            recurse(list)?;
            if let Some(predicate) = predicate {
                recurse(predicate)?;
            }
            if let Some(projection) = projection {
                recurse(projection)?;
            }
        }
        Expression::Reduce {
            initial,
            list,
            expression,
            ..
        } => {
            recurse(initial)?;
            recurse(list)?;
            recurse(expression)?;
        }
        Expression::ListPredicate {
            list, predicate, ..
        } => {
            recurse(list)?;
            recurse(predicate)?;
        }
        Expression::Function { arguments, .. } | Expression::List(arguments) => {
            for argument in arguments {
                recurse(argument)?;
            }
        }
        Expression::Map(values) => {
            for (_, value) in values {
                recurse(value)?;
            }
        }
        Expression::MapProjection { source, items } => {
            recurse(source)?;
            for item in items {
                if let MapProjectionItem::Entry(_, value) = item {
                    recurse(value)?;
                }
            }
        }
        Expression::Unary { operand, .. }
        | Expression::IsNull {
            expression: operand,
            ..
        } => recurse(operand)?,
        Expression::Binary { left, right, .. }
        | Expression::Index {
            expression: left,
            index: right,
        } => {
            recurse(left)?;
            recurse(right)?;
        }
        Expression::Property(value, _) => recurse(value)?,
        Expression::Slice {
            expression,
            start,
            end,
        } => {
            recurse(expression)?;
            if let Some(start) = start {
                recurse(start)?;
            }
            if let Some(end) = end {
                recurse(end)?;
            }
        }
        Expression::Literal(_)
        | Expression::Parameter(_)
        | Expression::Variable(_)
        | Expression::ExistentialSubquery(_)
        | Expression::Star => {}
    }
    Ok(())
}

fn validate_nonaggregate_order_fragment(
    expression: &Expression,
    projection: &Projection,
) -> Result<()> {
    if expression_contains_only_constants(expression)
        || projection.items.iter().any(|item| {
            item.alias
                .as_ref()
                .is_some_and(|alias| expression == &Expression::Variable(alias.clone()))
        })
    {
        return Ok(());
    }

    if let Some(item) = projection
        .items
        .iter()
        .find(|item| !contains_aggregate(&item.expression) && item.expression == *expression)
    {
        if is_simple_grouping_expression(&item.expression) {
            return Ok(());
        }
        return Err(Error::new(
            ErrorCode::QuerySyntax,
            "AmbiguousAggregationExpression: a complex grouping expression must be referenced through an alias inside aggregate ORDER BY",
        ));
    }

    Err(Error::new(
        ErrorCode::QuerySyntax,
        "UndefinedVariable: aggregate ORDER BY uses a value which is not a projected grouping key",
    ))
}

fn expression_contains_only_constants(expression: &Expression) -> bool {
    if expression.existential_subquery_parts().is_some() {
        return false;
    }
    match expression {
        Expression::Literal(_) | Expression::Parameter(_) => true,
        Expression::List(values) => values.iter().all(expression_contains_only_constants),
        Expression::Map(values) => values
            .iter()
            .all(|(_, value)| expression_contains_only_constants(value)),
        Expression::Case {
            operand,
            alternatives,
            default,
        } => {
            operand
                .as_deref()
                .is_none_or(expression_contains_only_constants)
                && alternatives.iter().all(|alternative| {
                    expression_contains_only_constants(&alternative.when)
                        && expression_contains_only_constants(&alternative.then)
                })
                && default
                    .as_deref()
                    .is_none_or(expression_contains_only_constants)
        }
        Expression::Function { arguments, .. } => {
            arguments.iter().all(expression_contains_only_constants)
        }
        Expression::Unary { operand, .. } => expression_contains_only_constants(operand),
        Expression::Binary { left, right, .. } => {
            expression_contains_only_constants(left) && expression_contains_only_constants(right)
        }
        Expression::IsNull { expression, .. } => expression_contains_only_constants(expression),
        Expression::Index { expression, index } => {
            expression_contains_only_constants(expression)
                && expression_contains_only_constants(index)
        }
        Expression::Slice {
            expression,
            start,
            end,
        } => {
            expression_contains_only_constants(expression)
                && start
                    .as_deref()
                    .is_none_or(expression_contains_only_constants)
                && end
                    .as_deref()
                    .is_none_or(expression_contains_only_constants)
        }
        Expression::Variable(_)
        | Expression::Property(_, _)
        | Expression::MapProjection { .. }
        | Expression::ListComprehension { .. }
        | Expression::Reduce { .. }
        | Expression::ListPredicate { .. }
        | Expression::ExistentialSubquery(_)
        | Expression::Star => false,
    }
}

fn lower(clauses: &[Clause]) -> Vec<PhysicalOperator> {
    let mut operators = Vec::new();
    let mut cursor = 0;
    let mut next_match_group = 0_u32;
    while cursor < clauses.len() {
        if let (
            Some(Clause::With(projection) | Clause::Return(projection)),
            Some(Clause::OrderBy(items)),
        ) = (clauses.get(cursor), clauses.get(cursor.saturating_add(1)))
        {
            // ORDER BY is evaluated against the projection's input scope plus its aliases, while
            // the result exposes only the requested projection. Materialize source-only sort
            // expressions once as inaccessible columns, sort, then trim only when such columns
            // exist. Projected keys keep the smaller project/sort plan. This preserves aliases,
            // aggregates, DISTINCT, and volatile expressions without reevaluation.
            let (expanded, rewritten_items) = projection_with_order_keys(projection, items);
            let has_hidden_order_keys = expanded.items.len() != projection.items.len();
            operators.push(PhysicalOperator::Project {
                keep_scope: false,
                projection: expanded,
            });
            operators.push(PhysicalOperator::Sort(rewritten_items));
            if has_hidden_order_keys {
                operators.push(PhysicalOperator::Project {
                    keep_scope: false,
                    projection: projection_binding_projection(projection),
                });
            }
            cursor = cursor.saturating_add(2);
            continue;
        }
        if let (Some(Clause::With(projection)), Some(Clause::Where(expression))) =
            (clauses.get(cursor), clauses.get(cursor.saturating_add(1)))
        {
            // WHERE following WITH sees both projected aliases and the bindings which fed that
            // projection. Materialize aliases without applying DISTINCT, keep the source bindings
            // through the filter, then trim to the projected values and apply DISTINCT there.
            // This prevents hidden source bindings from becoming part of the DISTINCT key.
            let mut materialized = projection.clone();
            materialized.distinct = false;
            operators.push(PhysicalOperator::Project {
                keep_scope: true,
                projection: materialized,
            });
            operators.push(PhysicalOperator::Filter(expression.clone()));
            let mut visible = projection_binding_projection(projection);
            visible.distinct = projection.distinct;
            operators.push(PhysicalOperator::Project {
                keep_scope: false,
                projection: visible,
            });
            cursor = cursor.saturating_add(2);
            continue;
        }
        if let (Some(Clause::History(history)), Some(Clause::Window(window))) =
            (clauses.get(cursor), clauses.get(cursor.saturating_add(1)))
        {
            operators.push(PhysicalOperator::TemporalHistoryWindow {
                history: history.clone(),
                window: window.clone(),
            });
            cursor = cursor.saturating_add(2);
            continue;
        }
        let clause = &clauses[cursor];
        match clause {
            Clause::Match { optional, patterns } => {
                let match_group = MatchGroupId(next_match_group);
                let Some(next) = next_match_group.checked_add(1) else {
                    // Saturating would alias two MATCH clauses into one relationship-uniqueness
                    // domain and silently change results, so lowering stops instead. The binder
                    // rejects queries long before this is reachable.
                    return operators;
                };
                next_match_group = next;
                operators.extend(patterns.iter().cloned().map(|pattern| {
                    PhysicalOperator::ScanPattern {
                        match_group,
                        optional: *optional,
                        pattern,
                        access: ScanAccessPath::Unspecified,
                    }
                }));
            }
            Clause::Where(expression) => {
                operators.push(PhysicalOperator::Filter(expression.clone()))
            }
            Clause::Unwind {
                expression,
                variable,
            } => operators.push(PhysicalOperator::Unwind {
                expression: expression.clone(),
                variable: variable.clone(),
            }),
            Clause::For {
                variable,
                expression,
            } => operators.push(PhysicalOperator::Unwind {
                expression: expression.clone(),
                variable: variable.clone(),
            }),
            Clause::Let(items) => {
                for item in items {
                    operators.push(PhysicalOperator::Project {
                        keep_scope: true,
                        projection: Projection {
                            distinct: false,
                            items: vec![ProjectionItem {
                                expression: item.expression.clone(),
                                alias: Some(item.variable.clone()),
                                source_text: None,
                            }],
                        },
                    });
                }
            }
            Clause::Filter(expression) => {
                operators.push(PhysicalOperator::Filter(expression.clone()))
            }
            Clause::Create(patterns) => operators.extend(
                patterns
                    .iter()
                    .cloned()
                    .map(PhysicalOperator::CreatePattern),
            ),
            Clause::Merge {
                pattern,
                on_create,
                on_match,
            } => operators.push(PhysicalOperator::MergePattern {
                pattern: pattern.clone(),
                on_create: on_create.clone(),
                on_match: on_match.clone(),
            }),
            Clause::Set(items) => operators.push(PhysicalOperator::Set(items.clone())),
            Clause::Remove(items) => operators.push(PhysicalOperator::Remove(items.clone())),
            Clause::Delete {
                detach,
                expressions,
            } => operators.push(PhysicalOperator::Delete {
                detach: *detach,
                expressions: expressions.clone(),
            }),
            Clause::With(projection) => operators.push(PhysicalOperator::Project {
                keep_scope: false,
                projection: projection.clone(),
            }),
            Clause::Return(projection) => operators.push(PhysicalOperator::Project {
                keep_scope: false,
                projection: projection.clone(),
            }),
            Clause::OrderBy(items) => operators.push(PhysicalOperator::Sort(items.clone())),
            Clause::Skip(value) => operators.push(PhysicalOperator::Skip(value.clone())),
            Clause::Limit(value) => operators.push(PhysicalOperator::Limit(value.clone())),
            Clause::History(value) => {
                operators.push(PhysicalOperator::TemporalHistory(value.clone()))
            }
            Clause::Window(value) => {
                operators.push(PhysicalOperator::TemporalWindow(value.clone()))
            }
            Clause::Search(value) => operators.push(PhysicalOperator::VectorSearch {
                search: value.clone(),
                access: VectorAccessPath::Unspecified,
            }),
            Clause::Call(value) => {
                operators.push(PhysicalOperator::BuiltinProcedure(value.clone()))
            }
            Clause::Finish => operators.push(PhysicalOperator::Finish),
        }
        cursor = cursor.saturating_add(1);
    }
    operators
}

fn projection_binding_projection(projection: &Projection) -> Projection {
    Projection {
        distinct: false,
        items: projection
            .items
            .iter()
            .enumerate()
            .map(|(index, item)| {
                if matches!(item.expression, Expression::Star) {
                    return item.clone();
                }
                let name = item.column_name(index);
                ProjectionItem {
                    expression: Expression::Variable(name.clone()),
                    alias: Some(name),
                    source_text: None,
                }
            })
            .collect(),
    }
}

fn projection_with_order_keys(
    projection: &Projection,
    items: &[SortItem],
) -> (Projection, Vec<SortItem>) {
    let aliases = projection
        .items
        .iter()
        .filter_map(|item| {
            item.alias
                .as_ref()
                .map(|alias| (alias.clone(), item.expression.clone()))
        })
        .collect::<BTreeMap<_, _>>();
    let mut expanded = projection.clone();
    let rewritten = items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            if let Some(expression) = projected_order_key(projection, &item.expression) {
                return SortItem {
                    expression,
                    ascending: item.ascending,
                };
            }
            let alias = format!("\0order_{index}");
            expanded.items.push(ProjectionItem {
                expression: substitute_projection_aliases(
                    &item.expression,
                    &aliases,
                    &BTreeSet::new(),
                ),
                alias: Some(alias.clone()),
                source_text: None,
            });
            SortItem {
                expression: Expression::Variable(alias),
                ascending: item.ascending,
            }
        })
        .collect();
    (expanded, rewritten)
}

fn projected_order_key(projection: &Projection, expression: &Expression) -> Option<Expression> {
    if projection
        .items
        .iter()
        .any(|item| matches!(item.expression, Expression::Star))
    {
        return Some(expression.clone());
    }
    if let Expression::Variable(variable) = expression && projection.items.iter().any(|item| {
        item.alias.as_deref() == Some(variable)
            || matches!(&item.expression, Expression::Variable(projected) if projected == variable)
    }) {
        return Some(expression.clone());
    }
    projection
        .items
        .iter()
        .enumerate()
        .find(|(_, item)| item.expression == *expression)
        .map(|(index, item)| Expression::Variable(item.column_name(index)))
}

/// Rewrites projection aliases back to their source expression so hidden ORDER BY keys can be
/// evaluated in the incoming scope. Locally bound comprehension/reduce variables shadow aliases.
fn substitute_projection_aliases(
    expression: &Expression,
    aliases: &BTreeMap<String, Expression>,
    shadowed: &BTreeSet<String>,
) -> Expression {
    let recurse = |expression: &Expression, shadowed: &BTreeSet<String>| {
        substitute_projection_aliases(expression, aliases, shadowed)
    };
    match expression {
        Expression::Literal(_)
        | Expression::Parameter(_)
        | Expression::ExistentialSubquery(_)
        | Expression::Star => expression.clone(),
        Expression::Variable(variable) => {
            if shadowed.contains(variable) {
                expression.clone()
            } else {
                match aliases.get(variable) {
                    Some(alias) => alias.clone(),
                    None => expression.clone(),
                }
            }
        }
        Expression::Property(value, property) => {
            Expression::Property(Box::new(recurse(value, shadowed)), property.clone())
        }
        Expression::List(values) => Expression::List(
            values
                .iter()
                .map(|value| recurse(value, shadowed))
                .collect(),
        ),
        Expression::Map(values) => Expression::Map(
            values
                .iter()
                .map(|(name, value)| (name.clone(), recurse(value, shadowed)))
                .collect(),
        ),
        Expression::MapProjection { source, items } => Expression::MapProjection {
            source: Box::new(recurse(source, shadowed)),
            items: items
                .iter()
                .map(|item| match item {
                    MapProjectionItem::AllProperties => MapProjectionItem::AllProperties,
                    MapProjectionItem::Property(name) => MapProjectionItem::Property(name.clone()),
                    MapProjectionItem::Variable(name) => MapProjectionItem::Variable(name.clone()),
                    MapProjectionItem::Entry(name, value) => {
                        MapProjectionItem::Entry(name.clone(), recurse(value, shadowed))
                    }
                })
                .collect(),
        },
        Expression::Case {
            operand,
            alternatives,
            default,
        } => Expression::Case {
            operand: operand
                .as_ref()
                .map(|operand| Box::new(recurse(operand, shadowed))),
            alternatives: alternatives
                .iter()
                .map(|alternative| CaseAlternative {
                    when: recurse(&alternative.when, shadowed),
                    then: recurse(&alternative.then, shadowed),
                })
                .collect(),
            default: default
                .as_ref()
                .map(|default| Box::new(recurse(default, shadowed))),
        },
        Expression::ListComprehension {
            variable,
            list,
            predicate,
            projection,
        } => {
            let mut inner = shadowed.clone();
            inner.insert(variable.clone());
            Expression::ListComprehension {
                variable: variable.clone(),
                list: Box::new(recurse(list, shadowed)),
                predicate: predicate
                    .as_ref()
                    .map(|predicate| Box::new(recurse(predicate, &inner))),
                projection: projection
                    .as_ref()
                    .map(|projection| Box::new(recurse(projection, &inner))),
            }
        }
        Expression::Reduce {
            accumulator,
            initial,
            variable,
            list,
            expression,
        } => {
            let mut inner = shadowed.clone();
            inner.insert(accumulator.clone());
            inner.insert(variable.clone());
            Expression::Reduce {
                accumulator: accumulator.clone(),
                initial: Box::new(recurse(initial, shadowed)),
                variable: variable.clone(),
                list: Box::new(recurse(list, shadowed)),
                expression: Box::new(recurse(expression, &inner)),
            }
        }
        Expression::ListPredicate {
            kind,
            variable,
            list,
            predicate,
        } => {
            let mut inner = shadowed.clone();
            inner.insert(variable.clone());
            Expression::ListPredicate {
                kind: *kind,
                variable: variable.clone(),
                list: Box::new(recurse(list, shadowed)),
                predicate: Box::new(recurse(predicate, &inner)),
            }
        }
        Expression::Function {
            name,
            distinct,
            arguments,
        } => Expression::Function {
            name: name.clone(),
            distinct: *distinct,
            arguments: arguments
                .iter()
                .map(|argument| recurse(argument, shadowed))
                .collect(),
        },
        Expression::Unary { operation, operand } => Expression::Unary {
            operation: *operation,
            operand: Box::new(recurse(operand, shadowed)),
        },
        Expression::Binary {
            left,
            operation,
            right,
        } => Expression::Binary {
            left: Box::new(recurse(left, shadowed)),
            operation: *operation,
            right: Box::new(recurse(right, shadowed)),
        },
        Expression::IsNull {
            expression,
            negated,
        } => Expression::IsNull {
            expression: Box::new(recurse(expression, shadowed)),
            negated: *negated,
        },
        Expression::Index { expression, index } => Expression::Index {
            expression: Box::new(recurse(expression, shadowed)),
            index: Box::new(recurse(index, shadowed)),
        },
        Expression::Slice {
            expression,
            start,
            end,
        } => Expression::Slice {
            expression: Box::new(recurse(expression, shadowed)),
            start: start
                .as_ref()
                .map(|start| Box::new(recurse(start, shadowed))),
            end: end.as_ref().map(|end| Box::new(recurse(end, shadowed))),
        },
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        ErrorCode,
        cypher::{BindCapabilities, bind, parse},
        graph::NameCatalog,
    };

    use super::{Expression, PhysicalOperator, SortItem, plan};

    fn planning_error(source: &str) -> crate::Result<crate::Error> {
        let bound = bind(
            parse(source)?,
            &NameCatalog::default(),
            BindCapabilities::default(),
        )?;
        match plan(bound) {
            Ok(_) => Err(crate::Error::internal(
                "planner accepted a query which should fail validation",
            )),
            Err(error) => Ok(error),
        }
    }

    fn planned_scan_groups(source: &str) -> crate::Result<Vec<(super::MatchGroupId, bool)>> {
        let bound = bind(
            parse(source)?,
            &NameCatalog::default(),
            BindCapabilities::default(),
        )?;
        Ok(plan(bound)?
            .operators
            .into_iter()
            .filter_map(|operator| match operator {
                PhysicalOperator::ScanPattern {
                    match_group,
                    optional,
                    ..
                } => Some((match_group, optional)),
                _ => None,
            })
            .collect())
    }

    #[test]
    fn comma_patterns_share_their_source_match_group() -> crate::Result<()> {
        let groups = planned_scan_groups("MATCH (a)-[r1]->(b), (b)-[r2]->(c) RETURN r1, r2")?;
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].0, groups[1].0);
        assert!(!groups[0].1);
        assert!(!groups[1].1);
        Ok(())
    }

    #[test]
    fn every_match_and_optional_match_starts_a_new_group() -> crate::Result<()> {
        let groups = planned_scan_groups(
            "MATCH (a)-[r1]->(b) MATCH (b)-[r2]->(c) OPTIONAL MATCH (c)-[r3]->(d) RETURN r1, r2, r3",
        )?;
        assert_eq!(groups.len(), 3);
        assert_ne!(groups[0].0, groups[1].0);
        assert_ne!(groups[1].0, groups[2].0);
        assert_eq!(
            groups
                .iter()
                .map(|(group, _)| group.get())
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(
            groups
                .iter()
                .map(|(_, optional)| *optional)
                .collect::<Vec<_>>(),
            vec![false, false, true]
        );
        Ok(())
    }

    #[test]
    fn order_by_rejects_values_removed_by_distinct() -> crate::Result<()> {
        let error = planning_error("MATCH (a) RETURN DISTINCT a.name ORDER BY a.age")?;
        assert_eq!(error.code, ErrorCode::QuerySyntax);
        assert!(error.message.contains("UndefinedVariable"));

        let bound = bind(
            parse("MATCH (a) RETURN DISTINCT a ORDER BY a.name")?,
            &NameCatalog::default(),
            BindCapabilities::default(),
        )?;
        plan(bound)?;
        Ok(())
    }

    #[test]
    fn order_by_rejects_aggregation_introduced_after_projection() -> crate::Result<()> {
        let error = planning_error("MATCH (n) RETURN n.num1 ORDER BY max(n.num2)")?;
        assert_eq!(error.code, ErrorCode::QuerySyntax);
        assert!(error.message.contains("InvalidAggregation"));

        let error = planning_error("MATCH (n) WITH n.num1 AS foo ORDER BY count(1) RETURN foo")?;
        assert_eq!(error.code, ErrorCode::QuerySyntax);
        assert!(error.message.contains("InvalidAggregation"));
        Ok(())
    }

    #[test]
    fn aggregate_order_by_accepts_grouping_keys_and_rejects_ambiguous_fragments()
    -> crate::Result<()> {
        for source in [
            "MATCH (person) RETURN avg(person.age) AS avgAge ORDER BY $age + avg(person.age) - 1000",
            "MATCH (me)--(you) RETURN me.age AS age, count(you.age) AS cnt ORDER BY age + count(you.age)",
            "MATCH (me)--(you) RETURN me.age AS age, count(you.age) AS cnt ORDER BY me.age + count(you.age)",
        ] {
            let bound = bind(
                parse(source)?,
                &NameCatalog::default(),
                BindCapabilities::default(),
            )?;
            plan(bound)?;
        }

        let undefined = planning_error(
            "MATCH (me)--(you) RETURN count(you.age) AS agg ORDER BY me.age + count(you.age)",
        )?;
        assert!(undefined.message.contains("UndefinedVariable"));

        let ambiguous = planning_error(
            "MATCH (me)--(you) RETURN me.age + you.age, count(*) AS cnt ORDER BY me.age + you.age + count(*)",
        )?;
        assert!(ambiguous.message.contains("AmbiguousAggregationExpression"));
        Ok(())
    }

    #[test]
    fn order_by_materializes_source_key_before_trimming_return_scope() -> crate::Result<()> {
        let bound = bind(
            parse("MATCH (n) RETURN n AS node ORDER BY n.num DESC")?,
            &NameCatalog::default(),
            BindCapabilities::default(),
        )?;
        let plan = plan(bound)?;
        let tail = &plan.operators[plan.operators.len().saturating_sub(3)..];
        let [
            PhysicalOperator::Project {
                projection: expanded,
                ..
            },
            PhysicalOperator::Sort(items),
            PhysicalOperator::Project {
                projection: visible,
                ..
            },
        ] = tail
        else {
            return Err(crate::Error::internal(
                "RETURN ORDER BY did not lower to materialize/sort/trim",
            ));
        };
        assert_eq!(expanded.items.len(), 2);
        assert_eq!(expanded.items[1].alias.as_deref(), Some("\0order_0"));
        assert!(matches!(
            items.as_slice(),
            [SortItem {
                expression: Expression::Variable(variable),
                ascending: false,
            }] if variable == "\0order_0"
        ));
        assert_eq!(visible.items.len(), 1);
        assert_eq!(visible.items[0].alias.as_deref(), Some("node"));
        Ok(())
    }

    #[test]
    fn order_by_reuses_an_identical_projected_value() -> crate::Result<()> {
        let bound = bind(
            parse("MATCH (n) RETURN n.num AS prop ORDER BY n.num")?,
            &NameCatalog::default(),
            BindCapabilities::default(),
        )?;
        let plan = plan(bound)?;
        let tail = &plan.operators[plan.operators.len().saturating_sub(2)..];
        let [
            PhysicalOperator::Project {
                projection: expanded,
                ..
            },
            PhysicalOperator::Sort(items),
        ] = tail
        else {
            return Err(crate::Error::internal(
                "projected RETURN ORDER BY did not lower directly to project/sort",
            ));
        };
        assert_eq!(expanded.items.len(), 1);
        assert!(matches!(
            items.as_slice(),
            [SortItem {
                expression: Expression::Variable(variable),
                ascending: true,
            }] if variable == "prop"
        ));
        Ok(())
    }

    #[test]
    fn standalone_updating_query_finishes_without_exposing_internal_bindings() -> crate::Result<()>
    {
        let bound = bind(
            parse("CREATE (n {value: 1})")?,
            &NameCatalog::default(),
            BindCapabilities {
                write: true,
                ..BindCapabilities::default()
            },
        )?;
        let physical = plan(bound)?;
        assert!(matches!(
            physical.operators.last(),
            Some(PhysicalOperator::Finish)
        ));
        Ok(())
    }

    #[test]
    fn updating_query_with_return_keeps_its_explicit_projection() -> crate::Result<()> {
        let bound = bind(
            parse("CREATE (n {value: 1}) RETURN n")?,
            &NameCatalog::default(),
            BindCapabilities {
                write: true,
                ..BindCapabilities::default()
            },
        )?;
        let physical = plan(bound)?;
        assert!(!matches!(
            physical.operators.last(),
            Some(PhysicalOperator::Finish)
        ));
        Ok(())
    }

    #[test]
    fn updating_query_return_with_limit_keeps_its_explicit_projection() -> crate::Result<()> {
        let bound = bind(
            parse("CREATE (n {value: 1}) RETURN n LIMIT 0")?,
            &NameCatalog::default(),
            BindCapabilities {
                write: true,
                ..BindCapabilities::default()
            },
        )?;
        let physical = plan(bound)?;
        assert!(matches!(
            physical.operators.last(),
            Some(PhysicalOperator::Limit(_))
        ));
        Ok(())
    }
}
