//! Deterministic statistics-backed logical rewrites and GPU-aware physical costing.

use std::collections::{BTreeMap, BTreeSet, HashSet};

use crate::{
    ScalarValue,
    execution::{BackendKind, metal_i64_sort_scratch_bytes},
    graph::{IndexCatalog, LayerMask, NameCatalog, OptimizerIndexStatistics, StatisticsSnapshot},
};

use super::{
    BinaryOperator, Direction, Expression, MapProjectionItem, NodePattern, PathSelector, Pattern,
    PhysicalOperator, PhysicalPlan, Projection, ProjectionItem, ResultValue, ScanAccessPath,
    SortItem, UnaryOperator, VectorAccessPath,
    value::{compare_predicate, equal as equal_values, truth as value_truth},
};

const JOIN_DP_LIMIT: usize = 10;
// Compact rows remain host-visible in the current physical executor. Account for the binding
// name, tagged value, inline-row bookkeeping, and allocator alignment rather than only an ID.
const ESTIMATED_BINDING_BYTES: u64 = 80;
const RUNTIME_REPLAN_THRESHOLD: f64 = 8.0;

/// Returns true only for an order-of-magnitude error large enough to repay one bounded restart.
/// Known-empty estimates are handled explicitly so zero never becomes an infinity/NaN accident.
#[must_use]
pub fn cardinality_misestimation(estimated: u64, actual: u64) -> bool {
    match (estimated, actual) {
        (0, 0) => false,
        (0, _) | (_, 0) => true,
        _ => {
            let larger = estimated.max(actual) as f64;
            let smaller = estimated.min(actual) as f64;
            larger / smaller >= RUNTIME_REPLAN_THRESHOLD
        }
    }
}

const CAP_SCAN: u64 = 1 << 0;
const CAP_FILTER: u64 = 1 << 1;
const CAP_EXPAND_OUT: u64 = 1 << 2;
const CAP_EXPAND_IN: u64 = 1 << 3;
const CAP_SORT_TOPK: u64 = 1 << 4;
const CAP_JOIN: u64 = 1 << 5;
const CAP_GROUP: u64 = 1 << 6;
const CAP_VECTOR_EXACT: u64 = 1 << 7;
const CAP_VECTOR_ANN: u64 = 1 << 8;
const CAP_TEMPORAL: u64 = 1 << 9;
const CAP_MULTIWAY_INTERSECTION: u64 = 1 << 10;

pub type RuntimeCardinalityFeedback = BTreeMap<[u8; 32], u64>;

/// Stable capability identity used by both physical costing and the plan-cache key.
#[must_use]
pub const fn backend_capability_bits(backend: BackendKind) -> u64 {
    match backend {
        BackendKind::Cpu => {
            CAP_SCAN
                | CAP_FILTER
                | CAP_EXPAND_OUT
                | CAP_EXPAND_IN
                | CAP_SORT_TOPK
                | CAP_JOIN
                | CAP_GROUP
                | CAP_VECTOR_EXACT
                | CAP_TEMPORAL
                | CAP_MULTIWAY_INTERSECTION
        }
        BackendKind::Metal | BackendKind::Cuda => {
            CAP_SCAN
                | CAP_FILTER
                | CAP_EXPAND_OUT
                | CAP_EXPAND_IN
                | CAP_SORT_TOPK
                | CAP_JOIN
                | CAP_GROUP
                | CAP_VECTOR_EXACT
                | CAP_VECTOR_ANN
                | CAP_TEMPORAL
                | CAP_MULTIWAY_INTERSECTION
        }
    }
}

/// Immutable inputs used to optimize one already-bound plan.
#[derive(Clone, Copy)]
pub struct OptimizerInput<'a> {
    pub statistics: &'a StatisticsSnapshot,
    pub catalog: &'a NameCatalog,
    pub indexes: Option<&'a IndexCatalog>,
    pub parameters: &'a BTreeMap<String, ResultValue>,
    pub backend: BackendKind,
    pub scratch_budget_bytes: usize,
    pub max_result_rows: usize,
    pub runtime_feedback: Option<&'a RuntimeCardinalityFeedback>,
    pub allow_runtime_checkpoint: bool,
}

/// Ephemeral diagnostics used for verification, profiling, admission, and cached-plan metadata.
#[derive(Clone, Debug, PartialEq)]
pub struct OptimizationProfile {
    pub statistics_revision: u64,
    pub schema_generation: [u8; 32],
    pub index_generation: [u8; 32],
    pub backend: BackendKind,
    pub backend_capabilities: u64,
    pub estimated_input_rows: f64,
    pub estimated_rows: f64,
    pub estimated_device_work: f64,
    pub estimated_persistent_bytes: u64,
    pub estimated_peak_scratch_bytes: u64,
    pub estimated_transfer_bytes: u64,
    pub estimated_index_work: f64,
    pub estimated_temporal_work: f64,
    pub estimated_ann_work: f64,
    pub estimated_adjacency_work: f64,
    pub estimated_materialization_bytes: u64,
    pub kernel_launches: u32,
    pub synchronization_points: u32,
    pub host_boundaries: u32,
    pub scratch_feasible: bool,
    /// Parameters whose *values* were folded into the plan, with the values they were folded from.
    ///
    /// Some rewrites resolve a parameter at planning time and bake the result into the operator
    /// tree — the dynamic node-property canonicalization below turns `n[$key]` into a static
    /// property access. The resulting plan is only valid for those exact values, but the plan cache
    /// keys parameters by shape rather than by value so that one plan can serve many bindings.
    /// Carrying the resolved bindings here lets the cache guard such plans without weakening the
    /// key for every other query.
    pub specialized_parameters: BTreeMap<String, ResultValue>,
}

#[derive(Clone, Copy, Debug, Default)]
struct CostVector {
    input_rows: f64,
    rows: f64,
    device_work: f64,
    persistent_bytes: u64,
    peak_scratch_bytes: u64,
    transfer_bytes: u64,
    index_work: f64,
    temporal_work: f64,
    ann_work: f64,
    adjacency_work: f64,
    materialization_bytes: u64,
    launches: u32,
    synchronizations: u32,
    host_boundaries: u32,
}

/// Optimizes every independent UNION branch using the same immutable statistics generation.
#[must_use]
pub fn optimize(
    mut plan: PhysicalPlan,
    input: OptimizerInput<'_>,
) -> (PhysicalPlan, OptimizationProfile) {
    if input.backend == BackendKind::Metal {
        canonicalize_metal_unique_entity_distinct(&mut plan);
        canonicalize_metal_relationship_type_predicate(&mut plan);
        canonicalize_metal_projection_boundaries(&mut plan);
        canonicalize_metal_static_entity_list_property(&mut plan);
        canonicalize_metal_static_union(&mut plan, input.max_result_rows);
        canonicalize_metal_static_distinct_top_one(&mut plan, input.max_result_rows);
    }
    let checkpoint = input.allow_runtime_checkpoint && plan.read_only && plan.unions.is_empty();
    let main = optimize_sequence(
        std::mem::take(&mut plan.operators),
        plan.read_layers,
        input,
        checkpoint,
    );
    plan.operators = main.operators;
    let mut total = main.cost;
    let mut specialized_parameters = main.specialized_parameters;
    for (_, branch) in &mut plan.unions {
        let mut optimized =
            optimize_sequence(std::mem::take(branch), plan.read_layers, input, false);
        *branch = std::mem::take(&mut optimized.operators);
        specialized_parameters.append(&mut optimized.specialized_parameters);
        total.input_rows = total
            .input_rows
            .saturating_add_f64(optimized.cost.input_rows);
        total.rows = total.rows.saturating_add_f64(optimized.cost.rows);
        total.device_work = total
            .device_work
            .saturating_add_f64(optimized.cost.device_work);
        total.peak_scratch_bytes = total
            .peak_scratch_bytes
            .max(optimized.cost.peak_scratch_bytes);
        total.persistent_bytes = total.persistent_bytes.max(optimized.cost.persistent_bytes);
        total.transfer_bytes = total
            .transfer_bytes
            .saturating_add(optimized.cost.transfer_bytes);
        total.index_work = total
            .index_work
            .saturating_add_f64(optimized.cost.index_work);
        total.temporal_work = total
            .temporal_work
            .saturating_add_f64(optimized.cost.temporal_work);
        total.ann_work = total.ann_work.saturating_add_f64(optimized.cost.ann_work);
        total.adjacency_work = total
            .adjacency_work
            .saturating_add_f64(optimized.cost.adjacency_work);
        total.materialization_bytes = total
            .materialization_bytes
            .saturating_add(optimized.cost.materialization_bytes);
        total.launches = total.launches.saturating_add(optimized.cost.launches);
        total.synchronizations = total
            .synchronizations
            .saturating_add(optimized.cost.synchronizations);
        total.host_boundaries = total
            .host_boundaries
            .saturating_add(optimized.cost.host_boundaries);
    }
    let profile = OptimizationProfile {
        statistics_revision: input.statistics.graph_revision,
        schema_generation: input.statistics.schema_generation,
        index_generation: input.indexes.map_or(
            input.statistics.index_generation,
            IndexCatalog::optimizer_generation,
        ),
        backend: input.backend,
        backend_capabilities: backend_capability_bits(input.backend),
        estimated_input_rows: total.input_rows,
        estimated_rows: total.rows,
        estimated_device_work: total.device_work,
        estimated_persistent_bytes: total.persistent_bytes,
        estimated_peak_scratch_bytes: total.peak_scratch_bytes,
        estimated_transfer_bytes: total.transfer_bytes,
        estimated_index_work: total.index_work,
        estimated_temporal_work: total.temporal_work,
        estimated_ann_work: total.ann_work,
        estimated_adjacency_work: total.adjacency_work,
        estimated_materialization_bytes: total.materialization_bytes,
        kernel_launches: total.launches,
        synchronization_points: total.synchronizations,
        host_boundaries: total.host_boundaries,
        scratch_feasible: total.peak_scratch_bytes <= input.scratch_budget_bytes as u64,
        specialized_parameters,
    };
    (plan, profile)
}

/// A non-optional node-only MATCH emits every canonical node at most once. If that node identity
/// is part of the first projection key, DISTINCT cannot remove a row, including when the planner
/// also materializes functionally dependent hidden ORDER BY properties. Removing only that
/// redundant flag exposes the existing typed row sorter without weakening scalar DISTINCT.
fn canonicalize_metal_unique_entity_distinct(plan: &mut PhysicalPlan) {
    if !plan.read_only {
        return;
    }
    remove_unique_entity_distinct(&mut plan.operators);
    for (_, branch) in &mut plan.unions {
        remove_unique_entity_distinct(branch);
    }
}

fn remove_unique_entity_distinct(operators: &mut [PhysicalOperator]) {
    let mut unique_node = None;
    for operator in operators {
        match operator {
            PhysicalOperator::ScanPattern {
                optional: false,
                pattern,
                ..
            } if unique_node.is_none() && pattern.steps.is_empty() => {
                let Some(variable) = pattern.start.variable.clone() else {
                    return;
                };
                unique_node = Some(variable);
            }
            PhysicalOperator::CardinalityCheckpoint { .. } | PhysicalOperator::Filter(_)
                if unique_node.is_some() => {}
            PhysicalOperator::Project { projection, .. } if unique_node.is_some() => {
                let Some(unique_node) = unique_node.as_deref() else {
                    return;
                };
                if projection.distinct
                    && projection.items.iter().any(|item| {
                        matches!(
                            &item.expression,
                            Expression::Variable(variable) if variable == unique_node
                        )
                    })
                {
                    projection.distinct = false;
                }
                return;
            }
            _ => return,
        }
    }
}

/// `MATCH (a)-[r]->(b) WHERE type(r) = 'X' [OR type(r) = 'Y' …]` binds exactly the same edges as
/// `MATCH (a)-[r:X|Y]->(b)`: constraining a bound relationship's type through a WHERE predicate is
/// the pattern's inline type constraint written a different way. The resident single-hop compilers
/// cover the inline-typed form but not the WHERE-predicate form, so on Metal this rewrite folds a
/// *pure* `type(r)` equality disjunction onto the step and drops the now-redundant `Filter`, letting
/// the proven typed-hop path answer natively instead of falling back to host. The rewrite is
/// semantics-preserving (CPU keeps the `Filter`, Metal folds it — both yield identical rows), and
/// Metal-only because that is where the coverage gap is.
fn canonicalize_metal_relationship_type_predicate(plan: &mut PhysicalPlan) {
    if !plan.read_only {
        return;
    }
    fold_relationship_type_predicate(&mut plan.operators);
    for (_, branch) in &mut plan.unions {
        fold_relationship_type_predicate(branch);
    }
}

fn fold_relationship_type_predicate(operators: &mut Vec<PhysicalOperator>) {
    let mut index = 0;
    while index + 1 < operators.len() {
        let folded = match (&operators[index], &operators[index + 1]) {
            (
                PhysicalOperator::ScanPattern {
                    optional: false,
                    pattern,
                    ..
                },
                PhysicalOperator::Filter(predicate),
            ) => relationship_type_disjunction_for_single_step(pattern, predicate),
            _ => None,
        };
        let Some(types) = folded else {
            index += 1;
            continue;
        };
        if let PhysicalOperator::ScanPattern { pattern, .. } = &mut operators[index] {
            pattern.steps[0].relationship.types = types;
        }
        operators.remove(index + 1);
        index += 1;
    }
}

/// The sorted, de-duplicated relationship types iff `pattern` is a fixed single hop binding a
/// relationship variable that has no inline type or property constraint, and `predicate` is
/// *entirely* a `type(<rel>)` equality disjunction. `None` (leave the plan unchanged) otherwise —
/// a conjunction with any other term, a property constraint, or a variable-length hop all decline,
/// so only the exact `type(r)`-only WHERE is ever folded.
fn relationship_type_disjunction_for_single_step(
    pattern: &Pattern,
    predicate: &Expression,
) -> Option<Vec<String>> {
    let [step] = pattern.steps.as_slice() else {
        return None;
    };
    let relationship = &step.relationship;
    let rel_var = relationship.variable.as_deref()?;
    if !relationship.types.is_empty()
        || relationship.variable_length
        || relationship.min_hops.is_some()
        || relationship.max_hops.is_some()
        || !relationship.properties.is_empty()
    {
        return None;
    }
    let mut names = Vec::new();
    if !collect_relationship_type_disjunction(predicate, rel_var, &mut names) || names.is_empty() {
        return None;
    }
    names.sort_unstable();
    names.dedup();
    Some(names)
}

/// True iff `expression` is entirely a disjunction of `type(<relationship>) = '<literal>'` equalities
/// (operands either order), collecting the literal type names. Any other node — a different function,
/// a non-literal operand, an AND, a comparison other than `=` — makes the whole thing decline.
fn collect_relationship_type_disjunction(
    expression: &Expression,
    relationship: &str,
    values: &mut Vec<String>,
) -> bool {
    if let Expression::Binary {
        left,
        operation: BinaryOperator::Or,
        right,
    } = expression
    {
        return collect_relationship_type_disjunction(left, relationship, values)
            && collect_relationship_type_disjunction(right, relationship, values);
    }
    let Expression::Binary {
        left,
        operation: BinaryOperator::Equal,
        right,
    } = expression
    else {
        return false;
    };
    let is_type_call = |candidate: &Expression| {
        matches!(
            candidate,
            Expression::Function { name, distinct: false, arguments }
                if matches!(name.as_slice(), [name] if name.eq_ignore_ascii_case("type"))
                    && matches!(arguments.as_slice(), [Expression::Variable(variable)] if variable == relationship)
        )
    };
    match (left.as_ref(), right.as_ref()) {
        (candidate, Expression::Literal(ScalarValue::String(value))) if is_type_call(candidate) => {
            values.push(value.to_string());
            true
        }
        (Expression::Literal(ScalarValue::String(value)), candidate) if is_type_call(candidate) => {
            values.push(value.to_string());
            true
        }
        _ => false,
    }
}

/// Removes a projection only when it repeats the complete, ordered output of the immediately
/// preceding scope-reset projection. The rule is Metal-only because its purpose is to expose a
/// plan already owned by a sealed resident compiler; ordinary CPU plans retain their source
/// boundaries for straightforward diagnostics.
fn canonicalize_metal_projection_boundaries(plan: &mut PhysicalPlan) {
    collapse_complete_identity_projection(&mut plan.operators);
    for (_, branch) in &mut plan.unions {
        collapse_complete_identity_projection(branch);
    }
}

fn collapse_complete_identity_projection(operators: &mut Vec<PhysicalOperator>) {
    let mut output = Vec::with_capacity(operators.len());
    for operator in operators.drain(..) {
        let redundant = match (output.last(), &operator) {
            (
                Some(PhysicalOperator::Project {
                    keep_scope: false,
                    projection: source,
                }),
                PhysicalOperator::Project {
                    keep_scope: false,
                    projection: identity,
                },
            ) => projection_output_names(source)
                .is_some_and(|names| projection_is_complete_identity(identity, &names)),
            _ => false,
        };
        if !redundant {
            output.push(operator);
        }
    }
    *operators = output;
}

/// Removes the exact entity-preserving list wrapper used by Graph6 `[4]`/`[8]` before native
/// admission. The source projection must construct precisely `[123, <one bound entity>]`; every
/// terminal item must then access a static property through `list[1]` and already carry a stable
/// alias or parsed source spelling. Replacing that expression by direct property access is ordinary
/// list-index substitution, while retaining that name metadata preserves the client-visible schema.
fn canonicalize_metal_static_entity_list_property(plan: &mut PhysicalPlan) {
    if !plan.read_only || plan.at_time.is_some() {
        return;
    }
    collapse_static_entity_list_property(&mut plan.operators);
    for (_, branch) in &mut plan.unions {
        collapse_static_entity_list_property(branch);
    }
}

fn collapse_static_entity_list_property(operators: &mut Vec<PhysicalOperator>) {
    let semantic = operators
        .iter()
        .enumerate()
        .filter_map(|(index, operator)| {
            (!matches!(operator, PhysicalOperator::CardinalityCheckpoint { .. }))
                .then_some((index, operator))
        })
        .collect::<Vec<_>>();
    let [
        (
            _,
            PhysicalOperator::ScanPattern {
                optional: false,
                pattern,
                ..
            },
        ),
        (
            wrapper_index,
            PhysicalOperator::Project {
                keep_scope: false,
                projection: wrapper,
            },
        ),
        (
            output_index,
            PhysicalOperator::Project {
                keep_scope: false,
                projection: output,
            },
        ),
    ] = semantic.as_slice()
    else {
        return;
    };
    if wrapper.distinct || output.distinct || output.items.is_empty() {
        return;
    }
    let [wrapper_item] = wrapper.items.as_slice() else {
        return;
    };
    let Some(wrapper_name) = wrapper_item.alias.as_deref() else {
        return;
    };
    let Expression::List(values) = &wrapper_item.expression else {
        return;
    };
    let [
        Expression::Literal(ScalarValue::Integer(123)),
        Expression::Variable(entity),
    ] = values.as_slice()
    else {
        return;
    };
    if !pattern_variables(pattern).contains(entity) {
        return;
    }

    let mut rewritten = output.clone();
    for item in &mut rewritten.items {
        if item.alias.is_none() && item.source_text.is_none() {
            // A synthesized unnamed item derives its client-visible name from the expression.
            // Rewriting it would therefore change the result schema even when the value is equal.
            return;
        }
        let Expression::Property(source, property) = &item.expression else {
            return;
        };
        let Expression::Index { expression, index } = source.as_ref() else {
            return;
        };
        if !matches!(expression.as_ref(), Expression::Variable(variable) if variable == wrapper_name)
            || !matches!(index.as_ref(), Expression::Literal(ScalarValue::Integer(1)))
        {
            return;
        }
        item.expression = Expression::Property(
            Box::new(Expression::Variable(entity.clone())),
            property.clone(),
        );
    }

    let output_index = *output_index;
    let wrapper_index = *wrapper_index;
    let PhysicalOperator::Project { projection, .. } = &mut operators[output_index] else {
        unreachable!("the sealed Graph6 output projection changed shape")
    };
    *projection = rewritten;
    operators.remove(wrapper_index);
}

fn projection_output_names(projection: &Projection) -> Option<Vec<String>> {
    if projection.items.is_empty()
        || projection
            .items
            .iter()
            .any(|item| matches!(item.expression, Expression::Star))
    {
        return None;
    }
    let mut names = Vec::with_capacity(projection.items.len());
    let mut unique = BTreeSet::new();
    for (index, item) in projection.items.iter().enumerate() {
        let name = item.column_name(index);
        if !unique.insert(name.clone()) {
            return None;
        }
        names.push(name);
    }
    Some(names)
}

fn projection_is_complete_identity(projection: &Projection, names: &[String]) -> bool {
    if projection.distinct {
        return false;
    }
    if matches!(projection.items.as_slice(), [item] if matches!(item.expression, Expression::Star))
    {
        return true;
    }
    projection.items.len() == names.len()
        && projection
            .items
            .iter()
            .zip(names)
            .enumerate()
            .all(|(index, (item, expected))| {
                item.column_name(index) == *expected
                    && matches!(
                        &item.expression,
                        Expression::Variable(variable) if variable == expected
                    )
            })
}

/// Converts a fully static, one-column UNION relation into the literal relation shape already
/// owned by the native materialized-key sorter. No branch is partially admitted: every row must
/// be an immutable homogeneous scalar literal, and DISTINCT is resolved only over that sealed
/// literal domain before the complete relation is sent through the selected backend.
fn canonicalize_metal_static_union(plan: &mut PhysicalPlan, max_result_rows: usize) {
    if !plan.read_only || plan.at_time.is_some() || plan.unions.is_empty() {
        return;
    }
    let Some((output_name, mut values)) = static_single_column_union_branch(&plan.operators) else {
        return;
    };
    for (all, branch) in &plan.unions {
        let Some((branch_name, branch_values)) = static_single_column_union_branch(branch) else {
            return;
        };
        if branch_name != output_name {
            return;
        }
        values.extend(branch_values);
        if !*all {
            deduplicate_static_union_values(&mut values);
        }
    }
    if values.is_empty()
        || values.len() > max_result_rows
        || !static_union_values_are_homogeneous(&values)
    {
        return;
    }

    let input_name = "\0union_value".to_owned();
    plan.operators = vec![
        PhysicalOperator::Unwind {
            expression: Expression::List(values.into_iter().map(Expression::Literal).collect()),
            variable: input_name.clone(),
        },
        PhysicalOperator::Project {
            keep_scope: false,
            projection: Projection {
                distinct: false,
                items: vec![ProjectionItem {
                    expression: Expression::Variable(input_name),
                    alias: Some(output_name.clone()),
                    source_text: None,
                }],
            },
        },
        // The CPU reference preserves branch order (and first occurrence for UNION DISTINCT).
        // A constant materialized key sends every immutable row through the native sorter while
        // its specified stable-tie contract preserves that exact order. Sorting by the union value
        // here would incorrectly turn [2, 1] into [1, 2].
        PhysicalOperator::Sort(vec![SortItem {
            expression: Expression::Literal(ScalarValue::Integer(0)),
            ascending: true,
        }]),
    ];
    plan.unions.clear();
}

fn static_single_column_union_branch(
    operators: &[PhysicalOperator],
) -> Option<(String, Vec<ScalarValue>)> {
    match operators {
        [
            PhysicalOperator::Project {
                keep_scope: false,
                projection,
            },
        ] if !projection.distinct && projection.items.len() == 1 => {
            let item = &projection.items[0];
            let Expression::Literal(value) = fold_expression(item.expression.clone()) else {
                return None;
            };
            static_union_value_supported(&value).then(|| (item.column_name(0), vec![value]))
        }
        [
            PhysicalOperator::Unwind {
                expression: Expression::List(values),
                variable,
            },
            PhysicalOperator::Project {
                keep_scope: false,
                projection,
            },
        ] if !projection.distinct && projection.items.len() == 1 => {
            let item = &projection.items[0];
            if !matches!(&item.expression, Expression::Variable(source) if source == variable) {
                return None;
            }
            let values = values
                .iter()
                .cloned()
                .map(fold_expression)
                .map(|expression| match expression {
                    Expression::Literal(value) if static_union_value_supported(&value) => {
                        Some(value)
                    }
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()?;
            Some((item.column_name(0), values))
        }
        _ => None,
    }
}

fn static_union_value_supported(value: &ScalarValue) -> bool {
    matches!(
        value,
        ScalarValue::Null
            | ScalarValue::Boolean(_)
            | ScalarValue::Integer(_)
            | ScalarValue::String(_)
    )
}

fn static_union_values_are_homogeneous(values: &[ScalarValue]) -> bool {
    let mut kind = None;
    for value in values {
        let next = match value {
            ScalarValue::Null => continue,
            ScalarValue::Boolean(_) => 0_u8,
            ScalarValue::Integer(_) => 1_u8,
            ScalarValue::String(_) => 2_u8,
            _ => return false,
        };
        if kind.replace(next).is_some_and(|kind| kind != next) {
            return false;
        }
    }
    true
}

fn deduplicate_static_union_values(values: &mut Vec<ScalarValue>) {
    let mut seen = HashSet::with_capacity(values.len());
    values.retain(|value| seen.insert(value.clone()));
}

/// Resolves the bounded literal-input DISTINCT in openCypher WithOrderBy1 scenario 44 before the
/// plan reaches the native materialized-key sorter. This is deliberately narrower than a generic
/// DISTINCT rewrite: the complete relation must be one non-empty, row-budgeted list of integer
/// literals; WITH, ORDER BY, and RETURN must all be the same single identity column; and LIMIT must
/// be exactly one. First-occurrence deduplication is semantic for DISTINCT, while the selected
/// ascending or descending minimum/maximum is still computed by the backend's receipted TopK.
fn canonicalize_metal_static_distinct_top_one(plan: &mut PhysicalPlan, max_result_rows: usize) {
    if !plan.read_only || plan.at_time.is_some() || !plan.unions.is_empty() {
        return;
    }
    let [
        PhysicalOperator::Unwind {
            expression: Expression::List(values),
            variable,
        },
        PhysicalOperator::Project {
            keep_scope: false,
            projection: distinct,
        },
        PhysicalOperator::Sort(sort_items),
        PhysicalOperator::Limit(Expression::Literal(ScalarValue::Integer(1))),
        PhysicalOperator::Project {
            keep_scope: false,
            projection: output,
        },
    ] = plan.operators.as_mut_slice()
    else {
        return;
    };
    let [sort] = sort_items.as_slice() else {
        return;
    };
    if values.is_empty()
        || values.len() > max_result_rows
        || !distinct.distinct
        || distinct.items.len() != 1
        || output.distinct
        || output.items.len() != 1
    {
        return;
    }
    let Expression::Variable(distinct_source) = &distinct.items[0].expression else {
        return;
    };
    let distinct_name = distinct.items[0].column_name(0);
    let Expression::Variable(sort_source) = &sort.expression else {
        return;
    };
    let Expression::Variable(output_source) = &output.items[0].expression else {
        return;
    };
    if distinct_source.as_str() != variable.as_str()
        || distinct_name.as_str() != variable.as_str()
        || sort_source.as_str() != distinct_name.as_str()
        || output_source.as_str() != distinct_name.as_str()
        || output.items[0].column_name(0).as_str() != distinct_name.as_str()
    {
        return;
    }

    let Some(mut scalars) = values
        .iter()
        .map(|value| match value {
            Expression::Literal(value @ ScalarValue::Integer(_)) => Some(value.clone()),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()
    else {
        return;
    };
    deduplicate_static_union_values(&mut scalars);
    *values = scalars.into_iter().map(Expression::Literal).collect();
    distinct.distinct = false;
}

struct OptimizedSequence {
    operators: Vec<PhysicalOperator>,
    cost: CostVector,
    specialized_parameters: BTreeMap<String, ResultValue>,
}

fn safe_rule_phase(operators: Vec<PhysicalOperator>) -> Vec<PhysicalOperator> {
    let folded = operators
        .into_iter()
        .filter(|operator| !matches!(operator, PhysicalOperator::CardinalityCheckpoint { .. }))
        .map(fold_operator_expressions)
        .collect::<Vec<_>>();
    let folded = remove_order_irrelevant_sorts(folded);
    prune_dead_scope_projections(folded)
}

fn fold_operator_expressions(operator: PhysicalOperator) -> PhysicalOperator {
    match operator {
        PhysicalOperator::Filter(expression) => {
            PhysicalOperator::Filter(fold_expression(expression))
        }
        PhysicalOperator::Unwind {
            expression,
            variable,
        } => PhysicalOperator::Unwind {
            expression: fold_expression(expression),
            variable,
        },
        PhysicalOperator::Project {
            keep_scope,
            mut projection,
        } => {
            for item in &mut projection.items {
                item.expression = fold_expression(item.expression.clone());
            }
            PhysicalOperator::Project {
                keep_scope,
                projection,
            }
        }
        PhysicalOperator::Sort(mut items) => {
            for item in &mut items {
                item.expression = fold_expression(item.expression.clone());
            }
            PhysicalOperator::Sort(items)
        }
        PhysicalOperator::Skip(expression) => PhysicalOperator::Skip(fold_expression(expression)),
        PhysicalOperator::Limit(expression) => PhysicalOperator::Limit(fold_expression(expression)),
        operator => operator,
    }
}

fn fold_expression(expression: Expression) -> Expression {
    match expression {
        Expression::Unary { operation, operand } => {
            let operand = fold_expression(*operand);
            if let Expression::Literal(value) = &operand {
                let folded = match (operation, value) {
                    (UnaryOperator::Not, ScalarValue::Boolean(value)) => {
                        Some(ScalarValue::Boolean(!value))
                    }
                    (UnaryOperator::Not, ScalarValue::Null) => Some(ScalarValue::Null),
                    (UnaryOperator::Positive, ScalarValue::Integer(value)) => {
                        Some(ScalarValue::Integer(*value))
                    }
                    (UnaryOperator::Positive, ScalarValue::Float(value)) => {
                        Some(ScalarValue::Float(*value))
                    }
                    (UnaryOperator::Negative, ScalarValue::Integer(value)) => {
                        value.checked_neg().map(ScalarValue::Integer)
                    }
                    (UnaryOperator::Negative, ScalarValue::Float(value)) => Some(
                        ScalarValue::Float(ordered_float::OrderedFloat(-value.into_inner())),
                    ),
                    _ => None,
                };
                if let Some(value) = folded {
                    return Expression::Literal(value);
                }
            }
            Expression::Unary {
                operation,
                operand: Box::new(operand),
            }
        }
        Expression::Binary {
            left,
            operation,
            right,
        } => {
            let left = fold_expression(*left);
            let right = fold_expression(*right);
            if let Some(value) = fold_literal_binary(&left, operation, &right) {
                return Expression::Literal(value);
            }
            Expression::Binary {
                left: Box::new(left),
                operation,
                right: Box::new(right),
            }
        }
        Expression::IsNull {
            expression,
            negated,
        } => {
            let expression = fold_expression(*expression);
            if let Expression::Literal(value) = &expression {
                return Expression::Literal(ScalarValue::Boolean(
                    matches!(value, ScalarValue::Null) != negated,
                ));
            }
            Expression::IsNull {
                expression: Box::new(expression),
                negated,
            }
        }
        Expression::List(values) => {
            Expression::List(values.into_iter().map(fold_expression).collect())
        }
        Expression::Map(values) => Expression::Map(
            values
                .into_iter()
                .map(|(name, value)| (name, fold_expression(value)))
                .collect(),
        ),
        expression => expression,
    }
}

fn fold_literal_binary(
    left: &Expression,
    operation: BinaryOperator,
    right: &Expression,
) -> Option<ScalarValue> {
    let (Expression::Literal(left), Expression::Literal(right)) = (left, right) else {
        return None;
    };
    let left = ResultValue::Scalar(left.clone());
    let right = ResultValue::Scalar(right.clone());
    let value = match operation {
        BinaryOperator::Equal => equal_values(&left, &right).ok()?.into_value(),
        BinaryOperator::NotEqual => equal_values(&left, &right).ok()?.not().into_value(),
        BinaryOperator::Less
        | BinaryOperator::LessOrEqual
        | BinaryOperator::Greater
        | BinaryOperator::GreaterOrEqual => compare_predicate(&left, &right, operation)
            .ok()?
            .into_value(),
        BinaryOperator::And => value_truth(&left)
            .ok()?
            .and(value_truth(&right).ok()?)
            .into_value(),
        BinaryOperator::Or => value_truth(&left)
            .ok()?
            .or(value_truth(&right).ok()?)
            .into_value(),
        BinaryOperator::Xor => value_truth(&left)
            .ok()?
            .xor(value_truth(&right).ok()?)
            .into_value(),
        _ => return None,
    };
    match value {
        ResultValue::Scalar(value) => Some(value),
        _ => None,
    }
}

fn remove_order_irrelevant_sorts(operators: Vec<PhysicalOperator>) -> Vec<PhysicalOperator> {
    let mut output = Vec::with_capacity(operators.len());
    let mut cursor = 0;
    while cursor < operators.len() {
        if matches!(operators.get(cursor), Some(PhysicalOperator::Sort(_)))
            && let Some(PhysicalOperator::Project { projection, .. }) = operators.get(cursor + 1)
            && projection_contains_aggregate(projection)
            && !projection_depends_on_input_order(projection)
        {
            cursor += 1;
            continue;
        }
        output.push(operators[cursor].clone());
        cursor += 1;
    }
    output
}

fn projection_contains_aggregate(projection: &Projection) -> bool {
    projection
        .items
        .iter()
        .any(|item| expression_contains_aggregate(&item.expression))
}

fn projection_depends_on_input_order(projection: &Projection) -> bool {
    projection
        .items
        .iter()
        .any(|item| expression_contains_function(&item.expression, "collect"))
}

fn expression_contains_aggregate(expression: &Expression) -> bool {
    const AGGREGATES: &[&str] = &[
        "count",
        "sum",
        "avg",
        "min",
        "max",
        "collect",
        "variance",
        "variancep",
        "stdev",
        "stdevp",
        "percentilecont",
        "percentiledisc",
    ];
    match expression {
        Expression::Function {
            name, arguments, ..
        } => {
            name.last().is_some_and(|name| {
                AGGREGATES
                    .iter()
                    .any(|aggregate| name.eq_ignore_ascii_case(aggregate))
            }) || arguments.iter().any(expression_contains_aggregate)
        }
        Expression::Unary { operand, .. }
        | Expression::Property(operand, _)
        | Expression::IsNull {
            expression: operand,
            ..
        } => expression_contains_aggregate(operand),
        Expression::Binary { left, right, .. } => {
            expression_contains_aggregate(left) || expression_contains_aggregate(right)
        }
        Expression::List(values) => values.iter().any(expression_contains_aggregate),
        Expression::Map(values) => values
            .iter()
            .any(|(_, value)| expression_contains_aggregate(value)),
        _ => false,
    }
}

fn expression_contains_function(expression: &Expression, wanted: &str) -> bool {
    match expression {
        Expression::Function {
            name, arguments, ..
        } => {
            name.last()
                .is_some_and(|name| name.eq_ignore_ascii_case(wanted))
                || arguments
                    .iter()
                    .any(|argument| expression_contains_function(argument, wanted))
        }
        Expression::Unary { operand, .. }
        | Expression::Property(operand, _)
        | Expression::IsNull {
            expression: operand,
            ..
        } => expression_contains_function(operand, wanted),
        Expression::Binary { left, right, .. } => {
            expression_contains_function(left, wanted)
                || expression_contains_function(right, wanted)
        }
        Expression::List(values) => values
            .iter()
            .any(|value| expression_contains_function(value, wanted)),
        Expression::Map(values) => values
            .iter()
            .any(|(_, value)| expression_contains_function(value, wanted)),
        _ => false,
    }
}

fn prune_dead_scope_projections(mut operators: Vec<PhysicalOperator>) -> Vec<PhysicalOperator> {
    let mut required = BTreeSet::new();
    let mut keep = vec![true; operators.len()];
    for position in (0..operators.len()).rev() {
        match &mut operators[position] {
            PhysicalOperator::Project {
                keep_scope: true,
                projection,
            } if !projection.distinct => {
                projection.items.retain(|item| {
                    let output = item.alias.as_ref().or_else(|| match &item.expression {
                        Expression::Variable(variable) => Some(variable),
                        _ => None,
                    });
                    let removable = output.is_some_and(|name| !required.contains(name))
                        && expression_is_total_for_dead_elimination(&item.expression);
                    if !removable {
                        required.extend(expression_variables(&item.expression));
                    }
                    !removable
                });
                if projection.items.is_empty() {
                    keep[position] = false;
                }
            }
            operator => required.extend(operator_referenced_variables(operator)),
        }
    }
    operators
        .into_iter()
        .zip(keep)
        .filter_map(|(operator, keep)| keep.then_some(operator))
        .collect()
}

fn expression_is_total_for_dead_elimination(expression: &Expression) -> bool {
    matches!(expression, Expression::Literal(_) | Expression::Variable(_))
}

fn operator_referenced_variables(operator: &PhysicalOperator) -> BTreeSet<String> {
    match operator {
        PhysicalOperator::Filter(expression)
        | PhysicalOperator::Unwind { expression, .. }
        | PhysicalOperator::Skip(expression)
        | PhysicalOperator::Limit(expression) => expression_variables(expression),
        PhysicalOperator::Project { projection, .. } => projection
            .items
            .iter()
            .flat_map(|item| expression_variables(&item.expression))
            .collect(),
        PhysicalOperator::Sort(items) | PhysicalOperator::TopK { items, .. } => items
            .iter()
            .flat_map(|item| expression_variables(&item.expression))
            .collect(),
        PhysicalOperator::ScanPattern { pattern, .. } => pattern_external_requirements(pattern),
        _ => BTreeSet::new(),
    }
}

fn optimize_sequence(
    mut operators: Vec<PhysicalOperator>,
    layers: LayerMask,
    input: OptimizerInput<'_>,
    allow_runtime_checkpoint: bool,
) -> OptimizedSequence {
    let mut specialized_parameters = BTreeMap::new();
    canonicalize_exact_static_dynamic_node_property(
        &mut operators,
        input.parameters,
        &mut specialized_parameters,
    );
    let operators = safe_rule_phase(operators);
    let operators = fuse_top_k(operators);
    let mut output = Vec::with_capacity(operators.len());
    let mut cursor = 0;
    let mut scope = BTreeSet::new();
    while cursor < operators.len() {
        let match_group = match operators.get(cursor) {
            Some(PhysicalOperator::ScanPattern {
                match_group,
                optional: false,
                ..
            }) => *match_group,
            _ => {
                let operator = operators[cursor].clone();
                update_scope(&operator, &mut scope);
                output.push(operator);
                cursor += 1;
                continue;
            }
        };
        let scan_start = cursor;
        while let Some(PhysicalOperator::ScanPattern {
            match_group: candidate_group,
            optional: false,
            ..
        }) = operators.get(cursor)
        {
            if *candidate_group != match_group {
                break;
            }
            cursor += 1;
        }

        let filter_start = cursor;
        while matches!(operators.get(cursor), Some(PhysicalOperator::Filter(_))) {
            cursor += 1;
        }
        let filter_conjuncts = operators[filter_start..cursor]
            .iter()
            .filter_map(|operator| match operator {
                PhysicalOperator::Filter(expression) => Some(split_conjunction(expression.clone())),
                _ => None,
            })
            .flatten()
            .collect::<Vec<_>>();
        let scans = operators[scan_start..filter_start]
            .iter()
            .filter_map(|operator| match operator {
                PhysicalOperator::ScanPattern { pattern, .. } => Some(pattern.clone()),
                _ => None,
            })
            .map(|pattern| optimize_pattern_anchor(pattern, &filter_conjuncts, layers, input))
            .collect::<Vec<_>>();
        let ordered = order_patterns(scans, &scope, layers, input);
        let mut filters_after = vec![Vec::<Expression>::new(); ordered.len()];
        let mut trailing_filters = Vec::new();
        for operator in &operators[filter_start..cursor] {
            let PhysicalOperator::Filter(expression) = operator else {
                continue;
            };
            for conjunct in split_conjunction(expression.clone()) {
                if !is_pushdown_safe(&conjunct) {
                    trailing_filters.push(conjunct);
                    continue;
                }
                let required = expression_variables(&conjunct);
                let mut available = scope.clone();
                let mut destination = None;
                for (position, pattern) in ordered.iter().enumerate() {
                    available.extend(pattern_variables(pattern));
                    if required.is_subset(&available) {
                        destination = Some(position);
                        break;
                    }
                }
                // Keep constants after the first scan. This preserves statement error timing while
                // still allowing them to prune every later join input.
                if let Some(position) = destination.or_else(|| ordered.is_empty().then_some(0))
                    && !ordered.is_empty()
                {
                    filters_after[position].push(conjunct);
                } else {
                    trailing_filters.push(conjunct);
                }
            }
        }
        for (position, pattern) in ordered.into_iter().enumerate() {
            let access =
                choose_access_path(&pattern, &filters_after[position], &scope, layers, input);
            // Through `access_estimated_rows`, because the chosen access path may already know far
            // more than the pattern does.
            //
            // `estimate_pattern` sees the PATTERN ONLY. When the equality predicate is written
            // inline — `(p:Product {name: $x})` — it is part of the pattern and selectivity applies.
            // When it is written as `WHERE p.name = $x` the predicate lives in a filter, the pattern
            // looks bare, and the estimate comes back as the whole label — even though
            // `choose_access_path` just lifted that same predicate into an `EqualityIndex` and
            // costed it at a row or two.
            //
            // The checkpoint then contradicted the access path it sits behind: measured on 200 000
            // nodes, `estimated_rows=200000 actual_rows=1`, which is a cardinality misestimation on
            // every execution, so every such query was replanned and re-optimized before it could
            // return. That is the whole 0.13 ms -> 69 ms gap between the two spellings of one query.
            // `access_estimated_rows` is the same reconciliation the cost model already applies, so
            // using it here makes the checkpoint agree with the plan instead of fighting it.
            let estimated_rows =
                access_estimated_rows(&access, estimate_pattern(&pattern, layers, input).max(0.0))
                    .ceil() as u64;
            let pattern_key = pattern_structural_key(&pattern);
            let can_checkpoint =
                allow_runtime_checkpoint && output.is_empty() && position == 0 && scope.is_empty();
            scope.extend(pattern_variables(&pattern));
            output.push(PhysicalOperator::ScanPattern {
                match_group,
                optional: false,
                pattern,
                access,
            });
            if can_checkpoint {
                output.push(PhysicalOperator::CardinalityCheckpoint {
                    pattern_key,
                    estimated_rows,
                });
            }
            output.extend(
                filters_after[position]
                    .drain(..)
                    .map(PhysicalOperator::Filter),
            );
        }
        output.extend(trailing_filters.into_iter().map(PhysicalOperator::Filter));
    }
    let output = lower_cyclic_multiway(output, input);
    let output = select_vector_access_paths(output, layers, input);
    let cost = estimate_sequence(&output, layers, input);
    OptimizedSequence {
        operators: output,
        cost,
        specialized_parameters,
    }
}

/// Canonicalizes the exact Graph7 node-property spelling already owned by the complete native
/// read and graph-free CREATE routes. A dynamic key is interchangeable with dot-property access
/// only when the indexed value is the node bound by the sole source operator and the complete key
/// recursively proves to one immutable STRING. Every wider expression or plan shape remains
/// untouched so it cannot acquire native admission through this normalization.
fn canonicalize_exact_static_dynamic_node_property(
    operators: &mut [PhysicalOperator],
    parameters: &BTreeMap<String, ResultValue>,
    specialized: &mut BTreeMap<String, ResultValue>,
) {
    let operator_indices = operators
        .iter()
        .enumerate()
        .filter_map(|(index, operator)| {
            (!matches!(operator, PhysicalOperator::CardinalityCheckpoint { .. })).then_some(index)
        })
        .collect::<Vec<_>>();
    let [source_index, projection_index] = operator_indices.as_slice() else {
        return;
    };
    let source = &operators[*source_index];
    let pattern = match source {
        PhysicalOperator::ScanPattern {
            optional: false,
            pattern,
            ..
        }
        | PhysicalOperator::CreatePattern(pattern) => pattern,
        _ => return,
    };
    if pattern.variable.is_some()
        || pattern.selector != PathSelector::All
        || pattern.mode != super::PathMode::DifferentRelationships
        || !pattern.steps.is_empty()
    {
        return;
    }
    let Some(variable) = pattern.start.variable.as_deref() else {
        return;
    };
    if variable.is_empty() {
        return;
    }
    let variable = variable.to_owned();
    let PhysicalOperator::Project {
        keep_scope: false,
        projection,
    } = &operators[*projection_index]
    else {
        return;
    };
    if projection.distinct {
        return;
    }
    let [item] = projection.items.as_slice() else {
        return;
    };
    let Expression::Index { expression, index } = &item.expression else {
        return;
    };
    if !matches!(expression.as_ref(), Expression::Variable(source) if source == &variable) {
        return;
    }
    let mut folded = BTreeMap::new();
    let Some(property) = immutable_dynamic_property_key(index, parameters, &mut folded) else {
        return;
    };
    let PhysicalOperator::Project { projection, .. } = &mut operators[*projection_index] else {
        unreachable!("the exact Graph7 projection changed shape")
    };
    projection.items[0].expression =
        Expression::Property(Box::new(Expression::Variable(variable)), property);
    // Only record the guard once the rewrite has actually committed, so a plan that was left
    // untouched never carries a constraint that would make it uncacheable.
    specialized.append(&mut folded);
}

fn immutable_dynamic_property_key(
    expression: &Expression,
    parameters: &BTreeMap<String, ResultValue>,
    folded: &mut BTreeMap<String, ResultValue>,
) -> Option<String> {
    match expression {
        Expression::Literal(ScalarValue::String(value)) => Some(value.to_string()),
        Expression::Parameter(name) => match parameters.get(name) {
            Some(value @ ResultValue::Scalar(ScalarValue::String(resolved))) => {
                let resolved = resolved.to_string();
                folded.insert(name.clone(), value.clone());
                Some(resolved)
            }
            _ => None,
        },
        Expression::Binary {
            left,
            operation: BinaryOperator::Add | BinaryOperator::Concat,
            right,
        } => {
            let mut value = immutable_dynamic_property_key(left, parameters, folded)?;
            let suffix = immutable_dynamic_property_key(right, parameters, folded)?;
            value.try_reserve_exact(suffix.len()).ok()?;
            value.push_str(&suffix);
            Some(value)
        }
        _ => None,
    }
}

fn select_vector_access_paths(
    mut operators: Vec<PhysicalOperator>,
    layers: LayerMask,
    input: OptimizerInput<'_>,
) -> Vec<PhysicalOperator> {
    let index_statistics = input
        .indexes
        .map(IndexCatalog::optimizer_statistics)
        .unwrap_or_default();
    let dimension = input
        .indexes
        .and_then(IndexCatalog::profile)
        .map_or(0_u64, |profile| u64::from(profile.dimension));
    let mut rows = 1.0_f64;
    let mut scope = BTreeSet::new();
    for operator in &mut operators {
        match operator {
            PhysicalOperator::ScanPattern {
                pattern, access, ..
            } => {
                let estimate =
                    access_estimated_rows(access, estimate_pattern(pattern, layers, input));
                let shared = pattern_variables(pattern)
                    .iter()
                    .any(|variable| scope.contains(variable));
                rows = if shared {
                    rows.min(estimate)
                } else {
                    bounded_product(rows, estimate, input.max_result_rows)
                };
                scope.extend(pattern_variables(pattern));
            }
            PhysicalOperator::VectorSearch { search, access } => {
                let statistics = index_statistics
                    .iter()
                    .find(|statistics| statistics.name == search.index);
                let total_rows = finite_rows(rows, input.max_result_rows);
                let filtered_rows = statistics.map_or(total_rows, |statistics| {
                    total_rows.min(statistics.vector_rows)
                });
                let query_expression = match &search.input {
                    super::SearchInput::Text(expression)
                    | super::SearchInput::Vector(expression) => expression,
                };
                let query_batch = if expression_variables(query_expression).is_empty() {
                    1
                } else {
                    let batches =
                        total_rows.saturating_add(filtered_rows.max(1) - 1) / filtered_rows.max(1);
                    u32::try_from(batches.max(1)).unwrap_or(u32::MAX)
                };
                let limit = constant_nonnegative_usize(&search.limit, input.parameters)
                    .unwrap_or(input.max_result_rows);
                *access = choose_vector_access(
                    statistics,
                    filtered_rows,
                    input.statistics.node_slot_count(),
                    query_batch,
                    u64::try_from(limit).unwrap_or(u64::MAX),
                    dimension,
                    input.backend,
                    input.scratch_budget_bytes,
                );
                rows = (filtered_rows.min(limit as u64)) as f64;
            }
            _ => update_scope(operator, &mut scope),
        }
    }
    operators
}

fn finite_rows(rows: f64, maximum: usize) -> u64 {
    if !rows.is_finite() || rows <= 0.0 {
        return 0;
    }
    rows.ceil().min(maximum as f64) as u64
}

fn constant_nonnegative_usize(
    expression: &Expression,
    parameters: &BTreeMap<String, ResultValue>,
) -> Option<usize> {
    let value = match expression {
        Expression::Literal(ScalarValue::Integer(value)) => *value,
        Expression::Parameter(name) => match parameters.get(name)? {
            ResultValue::Scalar(ScalarValue::Integer(value)) => *value,
            _ => return None,
        },
        _ => return None,
    };
    usize::try_from(value).ok()
}

fn choose_vector_access(
    statistics: Option<&OptimizerIndexStatistics>,
    filtered_rows: u64,
    resident_node_rows: u64,
    query_batch: u32,
    result_limit: u64,
    dimension: u64,
    backend: BackendKind,
    scratch_budget_bytes: usize,
) -> VectorAccessPath {
    let query_batch_u64 = u64::from(query_batch.max(1));
    let query_bytes = dimension.saturating_mul(6).saturating_mul(query_batch_u64);
    // Match the backend's hard vector-pipeline reservation. The vector column and candidate
    // masks are node-row-addressed, including tombstones, regardless of predicate selectivity.
    let pipeline_scratch = resident_node_rows
        .saturating_mul(2)
        .saturating_mul(256)
        .saturating_add(
            resident_node_rows
                .saturating_mul(2)
                .saturating_mul(dimension)
                .saturating_mul(2),
        )
        .saturating_add(
            result_limit
                .saturating_mul(query_batch_u64)
                .saturating_mul(32),
        )
        .saturating_add(query_bytes);
    let exact_scratch = pipeline_scratch;
    let exact = VectorAccessPath::Exact {
        filtered_rows,
        query_batch: query_batch.max(1),
        scratch_bytes: exact_scratch,
    };
    let Some(statistics) = statistics else {
        return exact;
    };
    let candidate_budget = statistics.ann_candidate_budget;
    let ann_scratch = pipeline_scratch.saturating_add(
        candidate_budget
            .saturating_mul(12)
            .saturating_mul(query_batch_u64),
    );
    let exact_work = filtered_rows
        .saturating_mul(dimension.max(1))
        .saturating_mul(query_batch_u64);
    let ann_work = candidate_budget
        .saturating_mul(dimension.max(1))
        .saturating_mul(query_batch_u64);
    let ann_capable = backend_capability_bits(backend) & CAP_VECTOR_ANN != 0;
    let sufficiently_selective = filtered_rows > candidate_budget.saturating_mul(4).max(1);
    if !ann_capable
        || candidate_budget == 0
        || result_limit > candidate_budget
        || !sufficiently_selective
        || ann_scratch > scratch_budget_bytes as u64
        || ann_work >= exact_work
    {
        return exact;
    }
    let Ok(candidate_budget) = u32::try_from(candidate_budget) else {
        return exact;
    };
    VectorAccessPath::IvfPq {
        filtered_rows,
        query_batch: query_batch.max(1),
        candidate_budget,
        scratch_bytes: ann_scratch,
    }
}

fn lower_cyclic_multiway(
    operators: Vec<PhysicalOperator>,
    input: OptimizerInput<'_>,
) -> Vec<PhysicalOperator> {
    if backend_capability_bits(input.backend) & CAP_MULTIWAY_INTERSECTION == 0 {
        return operators;
    }
    let mut output = Vec::with_capacity(operators.len());
    let mut cursor = 0;
    while cursor < operators.len() {
        // Optional scans cannot participate in the mandatory fixed-hop cycle lowering.  Treat
        // them like every other non-candidate operator so this pass always advances the cursor.
        let match_group = match &operators[cursor] {
            PhysicalOperator::ScanPattern {
                match_group,
                optional: false,
                ..
            } => *match_group,
            _ => {
                output.push(operators[cursor].clone());
                cursor += 1;
                continue;
            }
        };
        let start = cursor;
        let mut patterns = Vec::new();
        while cursor < operators.len() {
            match &operators[cursor] {
                PhysicalOperator::ScanPattern {
                    match_group: candidate_group,
                    optional: false,
                    pattern,
                    access,
                } if *candidate_group == match_group => {
                    patterns.push((pattern.clone(), access.clone()));
                    cursor += 1;
                }
                PhysicalOperator::CardinalityCheckpoint { .. } => cursor += 1,
                _ => break,
            }
        }
        if patterns.len() >= 3 && is_fixed_hop_cycle(&patterns) {
            output.push(PhysicalOperator::CyclicMultiwayJoin {
                match_group,
                patterns,
            });
        } else {
            output.extend_from_slice(&operators[start..cursor]);
        }
    }
    output
}

fn is_fixed_hop_cycle(patterns: &[(Pattern, ScanAccessPath)]) -> bool {
    let mut degree = BTreeMap::<&str, usize>::new();
    let mut adjacency = BTreeMap::<&str, BTreeSet<&str>>::new();
    for (pattern, _) in patterns {
        if pattern.variable.is_some()
            || pattern.selector != PathSelector::All
            || pattern.mode != super::PathMode::DifferentRelationships
            || pattern.steps.len() != 1
            || pattern.steps[0].relationship.variable_length
            || !pattern
                .start
                .properties
                .iter()
                .all(|(_, expression)| is_static_pattern_value(expression))
            || !pattern.steps[0]
                .node
                .properties
                .iter()
                .all(|(_, expression)| is_static_pattern_value(expression))
            || !pattern.steps[0]
                .relationship
                .properties
                .iter()
                .all(|(_, expression)| is_static_pattern_value(expression))
        {
            return false;
        }
        let (Some(start), Some(end)) = (
            pattern.start.variable.as_deref(),
            pattern.steps[0].node.variable.as_deref(),
        ) else {
            return false;
        };
        if start == end {
            return false;
        }
        *degree.entry(start).or_default() += 1;
        *degree.entry(end).or_default() += 1;
        adjacency.entry(start).or_default().insert(end);
        adjacency.entry(end).or_default().insert(start);
    }
    if degree.len() < 3 || degree.values().any(|degree| *degree < 2) {
        return false;
    }
    let Some(first) = degree.keys().next().copied() else {
        return false;
    };
    let mut reached = BTreeSet::from([first]);
    let mut pending = vec![first];
    while let Some(variable) = pending.pop() {
        for neighbor in adjacency.get(variable).into_iter().flatten() {
            if reached.insert(*neighbor) {
                pending.push(*neighbor);
            }
        }
    }
    reached.len() == degree.len()
}

fn is_static_pattern_value(expression: &Expression) -> bool {
    match expression {
        Expression::Literal(_) | Expression::Parameter(_) => true,
        Expression::List(values) => values.iter().all(is_static_pattern_value),
        Expression::Map(values) => values
            .iter()
            .all(|(_, value)| is_static_pattern_value(value)),
        _ => false,
    }
}

fn fuse_top_k(operators: Vec<PhysicalOperator>) -> Vec<PhysicalOperator> {
    let mut output = Vec::with_capacity(operators.len());
    let mut cursor = 0;
    while cursor < operators.len() {
        if let (
            Some(PhysicalOperator::Sort(items)),
            Some(PhysicalOperator::Limit(Expression::Literal(ScalarValue::Integer(limit)))),
        ) = (operators.get(cursor), operators.get(cursor + 1))
            && let Ok(limit) = usize::try_from(*limit)
        {
            output.push(PhysicalOperator::TopK {
                items: items.clone(),
                limit,
            });
            cursor += 2;
            continue;
        }
        output.push(operators[cursor].clone());
        cursor += 1;
    }
    output
}

fn pattern_structural_key(pattern: &Pattern) -> [u8; 32] {
    // The AST debug form is derived solely from ordered enum/field values and is identical for
    // equal bound plans in one binary. It is used only as an ephemeral deterministic tie key.
    *blake3::hash(format!("{pattern:?}").as_bytes()).as_bytes()
}

fn choose_access_path(
    pattern: &Pattern,
    filters: &[Expression],
    scope: &BTreeSet<String>,
    layers: LayerMask,
    input: OptimizerInput<'_>,
) -> ScanAccessPath {
    if pattern
        .start
        .variable
        .as_ref()
        .is_some_and(|variable| scope.contains(variable))
    {
        return ScanAccessPath::BoundVariable;
    }
    if let Some(variable) = pattern.start.variable.as_ref()
        && let Some(value) = filters
            .iter()
            .find_map(|filter| stable_id_lookup(filter, variable))
    {
        return ScanAccessPath::StableId {
            variable: variable.clone(),
            value,
        };
    }

    if let Some(indexes) = input.indexes {
        let mut indexed_values = BTreeMap::new();
        for (name, expression) in &pattern.start.properties {
            let Some(property) = input.catalog.property(name) else {
                continue;
            };
            let Some(value) = constant_scalar(expression, input.parameters) else {
                continue;
            };
            indexed_values.insert(property, (name.clone(), expression.clone(), value));
        }
        if let Some(variable) = pattern.start.variable.as_ref() {
            for filter in filters {
                let Some((candidate, name, BinaryOperator::Equal, expression)) =
                    simple_property_comparison(filter)
                else {
                    continue;
                };
                if candidate.as_str() != variable {
                    continue;
                }
                let Some(property) = input.catalog.property(&name) else {
                    continue;
                };
                let Some(value) = constant_scalar(&expression, input.parameters) else {
                    continue;
                };
                indexed_values
                    .entry(property)
                    .or_insert((name, expression, value));
            }
        }
        if !indexed_values.is_empty() {
            let values = indexed_values
                .iter()
                .map(|(property, (_, _, value))| (*property, value.clone()))
                .collect::<BTreeMap<_, _>>();
            let mut best = None;
            for label_name in &pattern.start.labels {
                let Some(label) = input.catalog.label(label_name) else {
                    continue;
                };
                if let Ok(Some(candidate)) = indexes.equality_candidate_estimate(label, &values)
                    && best
                        .as_ref()
                        .is_none_or(|(rows, name, _): &(u64, String, String)| {
                            (candidate.estimated_rows, candidate.name.as_str())
                                < (*rows, name.as_str())
                        })
                {
                    best = Some((candidate.estimated_rows, candidate.name, label_name.clone()));
                }
            }
            if let Some((estimated_rows, name, label)) = best {
                return ScanAccessPath::EqualityIndex {
                    name,
                    label,
                    values: indexed_values
                        .into_values()
                        .map(|(name, expression, _)| (name, expression))
                        .collect(),
                    estimated_rows,
                };
            }
        }
    }

    if let Some(variable) = pattern.start.variable.as_ref() {
        for filter in filters {
            let Some((candidate, property, operation, value)) = simple_property_comparison(filter)
            else {
                continue;
            };
            if candidate.as_str() != variable || constant_scalar(&value, input.parameters).is_none()
            {
                continue;
            }
            let estimated_rows = estimate_property_filter_rows(
                &pattern.start,
                &property,
                operation,
                &value,
                layers,
                input,
            );
            return ScanAccessPath::ResidentProperty {
                variable: variable.clone(),
                property,
                operation,
                value,
                estimated_rows,
            };
        }
    }
    pattern
        .start
        .labels
        .first()
        .map_or(ScanAccessPath::AllNodes, |label| ScanAccessPath::Label {
            label: label.clone(),
        })
}

fn stable_id_lookup(expression: &Expression, variable: &str) -> Option<Expression> {
    let Expression::Binary {
        left,
        operation: BinaryOperator::Equal,
        right,
    } = expression
    else {
        return None;
    };
    if is_id_of(left, variable) && expression_variables(right).is_empty() {
        return Some(right.as_ref().clone());
    }
    if is_id_of(right, variable) && expression_variables(left).is_empty() {
        return Some(left.as_ref().clone());
    }
    None
}

fn is_id_of(expression: &Expression, variable: &str) -> bool {
    matches!(
        expression,
        Expression::Function { name, arguments, .. }
            if name.len() == 1
                && name[0].eq_ignore_ascii_case("id")
                && matches!(arguments.as_slice(), [Expression::Variable(name)] if name == variable)
    )
}

fn simple_property_comparison(
    expression: &Expression,
) -> Option<(String, String, BinaryOperator, Expression)> {
    let Expression::Binary {
        left,
        operation,
        right,
    } = expression
    else {
        return None;
    };
    if !matches!(
        operation,
        BinaryOperator::Equal
            | BinaryOperator::Less
            | BinaryOperator::LessOrEqual
            | BinaryOperator::Greater
            | BinaryOperator::GreaterOrEqual
    ) {
        return None;
    }
    if let Expression::Property(source, property) = left.as_ref()
        && let Expression::Variable(variable) = source.as_ref()
        && expression_variables(right).is_empty()
    {
        return Some((
            variable.clone(),
            property.clone(),
            *operation,
            right.as_ref().clone(),
        ));
    }
    if let Expression::Property(source, property) = right.as_ref()
        && let Expression::Variable(variable) = source.as_ref()
        && expression_variables(left).is_empty()
    {
        return Some((
            variable.clone(),
            property.clone(),
            reverse_comparison(*operation),
            left.as_ref().clone(),
        ));
    }
    None
}

const fn reverse_comparison(operation: BinaryOperator) -> BinaryOperator {
    match operation {
        BinaryOperator::Less => BinaryOperator::Greater,
        BinaryOperator::LessOrEqual => BinaryOperator::GreaterOrEqual,
        BinaryOperator::Greater => BinaryOperator::Less,
        BinaryOperator::GreaterOrEqual => BinaryOperator::LessOrEqual,
        operation => operation,
    }
}

fn constant_scalar(
    expression: &Expression,
    parameters: &BTreeMap<String, ResultValue>,
) -> Option<ScalarValue> {
    match expression {
        Expression::Literal(value) => Some(value.clone()),
        Expression::Parameter(name) => match parameters.get(name)? {
            ResultValue::Scalar(value) => Some(value.clone()),
            _ => None,
        },
        _ => None,
    }
}

fn estimate_property_filter_rows(
    node: &NodePattern,
    property_name: &str,
    operation: BinaryOperator,
    value: &Expression,
    layers: LayerMask,
    input: OptimizerInput<'_>,
) -> u64 {
    let base = estimate_node(node, layers, input);
    let Some(property) = input.catalog.property(property_name) else {
        return 0;
    };
    let Some(stats) = input.statistics.node_property(property, layers) else {
        return base.ceil() as u64;
    };
    let Some(value) = constant_scalar(value, input.parameters) else {
        return base.ceil() as u64;
    };
    let selectivity = match (operation, value) {
        (BinaryOperator::Equal, _) => {
            stats.equality_selectivity(input.statistics.node_count(layers))
        }
        (operation, ScalarValue::Integer(value)) => stats
            .numeric_range_selectivity(
                input.statistics.node_count(layers),
                value as f64,
                matches!(
                    operation,
                    BinaryOperator::LessOrEqual | BinaryOperator::GreaterOrEqual
                ),
                matches!(
                    operation,
                    BinaryOperator::Less | BinaryOperator::LessOrEqual
                ),
            )
            .unwrap_or(1.0),
        (operation, ScalarValue::Float(value)) => stats
            .numeric_range_selectivity(
                input.statistics.node_count(layers),
                value.into_inner(),
                matches!(
                    operation,
                    BinaryOperator::LessOrEqual | BinaryOperator::GreaterOrEqual
                ),
                matches!(
                    operation,
                    BinaryOperator::Less | BinaryOperator::LessOrEqual
                ),
            )
            .unwrap_or(1.0),
        _ => 1.0,
    };
    (base * selectivity).max(0.0).ceil() as u64
}

fn order_patterns(
    patterns: Vec<Pattern>,
    initial_scope: &BTreeSet<String>,
    layers: LayerMask,
    input: OptimizerInput<'_>,
) -> Vec<Pattern> {
    if patterns.len() <= 1 {
        return patterns;
    }
    if patterns.len() <= JOIN_DP_LIMIT
        && let Some(order) = dynamic_program_order(&patterns, initial_scope, layers, input)
    {
        return order
            .into_iter()
            .map(|position| patterns[position].clone())
            .collect();
    }
    greedy_pattern_order(patterns, initial_scope, layers, input)
}

#[derive(Clone)]
struct JoinState {
    order: Vec<usize>,
    structural_order: Vec<[u8; 32]>,
    scope: BTreeSet<String>,
    rows: f64,
    work: f64,
    peak: u64,
}

fn dynamic_program_order(
    patterns: &[Pattern],
    initial_scope: &BTreeSet<String>,
    layers: LayerMask,
    input: OptimizerInput<'_>,
) -> Option<Vec<usize>> {
    let count = patterns.len();
    let estimates = patterns
        .iter()
        .map(|pattern| estimate_pattern(pattern, layers, input))
        .collect::<Vec<_>>();
    let variables = patterns.iter().map(pattern_variables).collect::<Vec<_>>();
    let requirements = patterns
        .iter()
        .map(pattern_external_requirements)
        .collect::<Vec<_>>();
    let structural_keys = patterns
        .iter()
        .map(pattern_structural_key)
        .collect::<Vec<_>>();
    let mut states = BTreeMap::<u16, JoinState>::new();
    for index in 0..count {
        if !requirements[index].is_subset(initial_scope) {
            continue;
        }
        let rows = estimates[index];
        let mut scope = initial_scope.clone();
        scope.extend(variables[index].iter().cloned());
        states.insert(
            1_u16 << index,
            JoinState {
                order: vec![index],
                structural_order: vec![structural_keys[index]],
                scope,
                rows,
                work: rows,
                peak: estimated_row_bytes(rows, variables[index].len()),
            },
        );
    }
    for _ in 1..count {
        let previous = states.clone();
        for (mask, state) in previous {
            let connected_remaining = (0..count).any(|index| {
                mask & (1_u16 << index) == 0
                    && variables[index]
                        .iter()
                        .any(|variable| state.scope.contains(variable))
            });
            for index in 0..count {
                let bit = 1_u16 << index;
                if mask & bit != 0 || !requirements[index].is_subset(&state.scope) {
                    continue;
                }
                let shared = variables[index]
                    .iter()
                    .any(|variable| state.scope.contains(variable));
                if connected_remaining && !shared {
                    continue;
                }
                let rows = if shared {
                    state.rows.min(estimates[index])
                } else {
                    bounded_product(state.rows, estimates[index], input.max_result_rows)
                };
                // The current executor applies each later pattern to every surviving prefix.
                // Counting the resulting prefixes makes a selective first anchor strictly
                // cheaper for an otherwise equivalent Cartesian ordering.
                let work = state.work.saturating_add_f64(rows);
                let mut scope = state.scope.clone();
                scope.extend(variables[index].iter().cloned());
                let peak = state.peak.max(estimated_row_bytes(rows, scope.len()));
                let mut order = state.order.clone();
                order.push(index);
                let mut structural_order = state.structural_order.clone();
                structural_order.push(structural_keys[index]);
                let candidate = JoinState {
                    order,
                    structural_order,
                    scope,
                    rows,
                    work,
                    peak,
                };
                let next_mask = mask | bit;
                if states
                    .get(&next_mask)
                    .is_none_or(|current| join_state_better(&candidate, current, input))
                {
                    states.insert(next_mask, candidate);
                }
            }
        }
    }
    states
        .get(&((1_u16 << count) - 1))
        .map(|state| state.order.clone())
}

fn join_state_better(
    candidate: &JoinState,
    current: &JoinState,
    input: OptimizerInput<'_>,
) -> bool {
    let candidate_feasible = candidate.peak <= input.scratch_budget_bytes as u64;
    let current_feasible = current.peak <= input.scratch_budget_bytes as u64;
    candidate_feasible
        .cmp(&current_feasible)
        .reverse()
        .then_with(|| candidate.work.total_cmp(&current.work))
        .then_with(|| candidate.peak.cmp(&current.peak))
        .then_with(|| candidate.structural_order.cmp(&current.structural_order))
        .then_with(|| candidate.order.cmp(&current.order))
        .is_lt()
}

fn greedy_pattern_order(
    mut patterns: Vec<Pattern>,
    initial_scope: &BTreeSet<String>,
    layers: LayerMask,
    input: OptimizerInput<'_>,
) -> Vec<Pattern> {
    let mut scope = initial_scope.clone();
    let mut output = Vec::with_capacity(patterns.len());
    while !patterns.is_empty() {
        let best = patterns
            .iter()
            .enumerate()
            .filter(|(_, pattern)| pattern_external_requirements(pattern).is_subset(&scope))
            .min_by(|(left_position, left), (right_position, right)| {
                let left_connected = pattern_variables(left)
                    .iter()
                    .any(|variable| scope.contains(variable));
                let right_connected = pattern_variables(right)
                    .iter()
                    .any(|variable| scope.contains(variable));
                right_connected
                    .cmp(&left_connected)
                    .then_with(|| {
                        estimate_pattern(left, layers, input)
                            .total_cmp(&estimate_pattern(right, layers, input))
                    })
                    .then_with(|| pattern_structural_key(left).cmp(&pattern_structural_key(right)))
                    .then_with(|| left_position.cmp(right_position))
            })
            .map(|(position, _)| position)
            .unwrap_or(0);
        let pattern = patterns.remove(best);
        scope.extend(pattern_variables(&pattern));
        output.push(pattern);
    }
    output
}

fn optimize_pattern_anchor(
    mut pattern: Pattern,
    filters: &[Expression],
    layers: LayerMask,
    input: OptimizerInput<'_>,
) -> Pattern {
    reorder_node_constraints(&mut pattern.start, layers, input);
    for step in &mut pattern.steps {
        reorder_node_constraints(&mut step.node, layers, input);
    }
    if !pattern_can_reverse(&pattern) {
        return pattern;
    }
    let Some(last) = pattern.steps.last().map(|step| &step.node) else {
        return pattern;
    };
    if last.variable.as_ref().is_some_and(|variable| {
        filters
            .iter()
            .any(|filter| stable_id_lookup(filter, variable).is_some())
    }) {
        return reverse_pattern(pattern);
    }
    let start_rows = estimate_node(&pattern.start, layers, input);
    let end_rows = estimate_node(last, layers, input);
    let forward = pattern_directional_work(&pattern, start_rows, layers, input, false);
    let reverse = pattern_directional_work(&pattern, end_rows, layers, input, true);
    if reverse < forward {
        reverse_pattern(pattern)
    } else {
        pattern
    }
}

fn pattern_directional_work(
    pattern: &Pattern,
    anchor_rows: f64,
    layers: LayerMask,
    input: OptimizerInput<'_>,
    reverse: bool,
) -> f64 {
    let mut work = anchor_rows;
    let steps: Box<dyn Iterator<Item = &super::PatternStep>> = if reverse {
        Box::new(pattern.steps.iter().rev())
    } else {
        Box::new(pattern.steps.iter())
    };
    for step in steps {
        let types = step
            .relationship
            .types
            .iter()
            .filter_map(|name| input.catalog.relationship_type(name))
            .collect::<Vec<_>>();
        let outgoing = match (step.relationship.direction, reverse) {
            (Direction::Outgoing, false) | (Direction::Incoming, true) => Some(true),
            (Direction::Incoming, false) | (Direction::Outgoing, true) => Some(false),
            (Direction::Undirected, _) => None,
        };
        let fanout = outgoing.map_or_else(
            || {
                input.statistics.directional_fanout(&types, layers, true)
                    + input.statistics.directional_fanout(&types, layers, false)
            },
            |outgoing| {
                input
                    .statistics
                    .directional_fanout(&types, layers, outgoing)
            },
        );
        work = work.saturating_add_f64(work * fanout.max(0.0));
    }
    work
}

fn reorder_node_constraints(node: &mut NodePattern, layers: LayerMask, input: OptimizerInput<'_>) {
    node.labels.sort_by(|left, right| {
        let left_count = input
            .catalog
            .label(left)
            .map_or(0, |label| input.statistics.label_count(label, layers));
        let right_count = input
            .catalog
            .label(right)
            .map_or(0, |label| input.statistics.label_count(label, layers));
        left_count.cmp(&right_count).then_with(|| left.cmp(right))
    });
    node.properties.sort_by(|(left, _), (right, _)| {
        let parent = input.statistics.node_count(layers);
        let left_selectivity = input
            .catalog
            .property(left)
            .and_then(|property| input.statistics.node_property(property, layers))
            .map_or(1.0, |stats| stats.equality_selectivity(parent));
        let right_selectivity = input
            .catalog
            .property(right)
            .and_then(|property| input.statistics.node_property(property, layers))
            .map_or(1.0, |stats| stats.equality_selectivity(parent));
        left_selectivity
            .total_cmp(&right_selectivity)
            .then_with(|| left.cmp(right))
    });
}

fn pattern_can_reverse(pattern: &Pattern) -> bool {
    pattern.variable.is_none()
        && pattern.selector == PathSelector::All
        && !pattern.steps.is_empty()
        && pattern.steps.iter().all(|step| {
            !step.relationship.variable_length
                && step
                    .relationship
                    .properties
                    .iter()
                    .all(|(_, expression)| is_constant_expression(expression))
                && step
                    .node
                    .properties
                    .iter()
                    .all(|(_, expression)| is_constant_expression(expression))
        })
        && pattern
            .start
            .properties
            .iter()
            .all(|(_, expression)| is_constant_expression(expression))
}

fn reverse_pattern(pattern: Pattern) -> Pattern {
    let start = pattern
        .steps
        .last()
        .map_or_else(|| pattern.start.clone(), |step| step.node.clone());
    let mut nodes = Vec::with_capacity(pattern.steps.len() + 1);
    nodes.push(pattern.start);
    nodes.extend(pattern.steps.iter().map(|step| step.node.clone()));
    let mut steps = Vec::with_capacity(pattern.steps.len());
    for index in (0..pattern.steps.len()).rev() {
        let mut relationship = pattern.steps[index].relationship.clone();
        relationship.direction = match relationship.direction {
            Direction::Outgoing => Direction::Incoming,
            Direction::Incoming => Direction::Outgoing,
            Direction::Undirected => Direction::Undirected,
        };
        steps.push(super::PatternStep {
            relationship,
            node: nodes[index].clone(),
        });
    }
    Pattern {
        variable: pattern.variable,
        selector: pattern.selector,
        mode: pattern.mode,
        start,
        steps,
    }
}

fn estimate_sequence(
    operators: &[PhysicalOperator],
    layers: LayerMask,
    input: OptimizerInput<'_>,
) -> CostVector {
    let mut cost = CostVector {
        input_rows: 1.0,
        rows: 1.0,
        persistent_bytes: input.statistics.resident_graph_bytes,
        ..CostVector::default()
    };
    let mut scope = BTreeSet::new();
    let backend_factor = match input.backend {
        BackendKind::Cpu => 1.0,
        BackendKind::Metal => 0.20,
        BackendKind::Cuda => 0.18,
    };
    for operator in operators {
        match operator {
            PhysicalOperator::ScanPattern {
                pattern, access, ..
            } => {
                let logical_estimate = estimate_pattern(pattern, layers, input);
                let estimate = access_estimated_rows(access, logical_estimate);
                let shared = pattern_variables(pattern)
                    .iter()
                    .any(|variable| scope.contains(variable));
                let previous = cost.rows;
                cost.rows = if shared {
                    cost.rows.min(estimate)
                } else {
                    bounded_product(cost.rows, estimate, input.max_result_rows)
                };
                cost.input_rows = cost.input_rows.saturating_add_f64(previous);
                cost.device_work = cost
                    .device_work
                    .saturating_add_f64((estimate + cost.rows) * backend_factor);
                let hops = pattern.steps.len() as f64;
                cost.adjacency_work = cost.adjacency_work.saturating_add_f64(cost.rows * hops);
                if matches!(access, ScanAccessPath::EqualityIndex { .. }) {
                    cost.index_work = cost
                        .index_work
                        .saturating_add_f64(estimate.max(1.0).log2() + cost.rows);
                } else if matches!(access, ScanAccessPath::StableId { .. }) {
                    cost.index_work = cost.index_work.saturating_add_f64(1.0);
                }
                cost.launches = cost.launches.saturating_add(1);
                if input.backend != BackendKind::Cpu {
                    // The currently selected operator ABI returns only a bounded selection
                    // vector. Full graph values remain resident; account for that vector.
                    cost.transfer_bytes = cost
                        .transfer_bytes
                        .saturating_add((cost.rows.ceil() as u64).saturating_mul(4));
                }
                scope.extend(pattern_variables(pattern));
            }
            PhysicalOperator::CardinalityCheckpoint { .. } => {}
            PhysicalOperator::CyclicMultiwayJoin { patterns, .. } => {
                let relation_rows = patterns
                    .iter()
                    .map(|(pattern, _)| estimate_pattern(pattern, layers, input).max(1.0))
                    .collect::<Vec<_>>();
                // For a simple cycle the fractional edge-cover bound assigns one half to every
                // binary relation. This is the admission bound for the generic-join lowering.
                let agm_rows = (relation_rows.iter().map(|rows| rows.ln()).sum::<f64>() * 0.5)
                    .exp()
                    .min(input.max_result_rows as f64);
                let prior = cost.rows;
                cost.rows = bounded_product(cost.rows, agm_rows, input.max_result_rows);
                cost.input_rows = cost.input_rows.saturating_add_f64(prior);
                let relation_work = relation_rows.iter().sum::<f64>();
                cost.device_work = cost
                    .device_work
                    .saturating_add_f64((relation_work + cost.rows) * backend_factor);
                cost.adjacency_work = cost.adjacency_work.saturating_add_f64(relation_work);
                cost.index_work = cost.index_work.saturating_add_f64(
                    patterns
                        .iter()
                        .filter(|(_, access)| {
                            matches!(
                                access,
                                ScanAccessPath::EqualityIndex { .. }
                                    | ScanAccessPath::StableId { .. }
                            )
                        })
                        .count() as f64,
                );
                cost.launches = cost.launches.saturating_add(1);
            }
            PhysicalOperator::Filter(_) => {
                cost.device_work = cost
                    .device_work
                    .saturating_add_f64(cost.rows * backend_factor);
                cost.launches = cost.launches.saturating_add(1);
            }
            PhysicalOperator::Sort(_) => {
                let sort_rows = cost.rows;
                let work = if input.backend == BackendKind::Metal {
                    sort_rows * 9.0
                } else {
                    sort_rows * sort_rows.max(2.0).log2()
                };
                cost.device_work = cost.device_work.saturating_add_f64(work * backend_factor);
                cost.launches =
                    cost.launches
                        .saturating_add(if input.backend == BackendKind::Metal {
                            27
                        } else {
                            1
                        });
                cost.synchronizations = cost.synchronizations.saturating_add(1);
                if input.backend == BackendKind::Metal {
                    cost.peak_scratch_bytes = cost
                        .peak_scratch_bytes
                        .max(estimated_metal_sort_scratch(sort_rows, sort_rows));
                }
                cost.materialization_bytes = cost
                    .materialization_bytes
                    .max(estimated_row_bytes(cost.rows, scope.len()));
            }
            PhysicalOperator::TopK { limit, .. } => {
                let kept = cost.rows.min(*limit as f64);
                let (work, launches) = if *limit == 0 {
                    (0.0, 0)
                } else if input.backend == BackendKind::Metal && *limit <= 256 {
                    let passes = estimated_metal_top_k_passes(cost.rows, *limit);
                    (cost.rows * 55.0 * 4.0 / 3.0, passes)
                } else if input.backend == BackendKind::Metal {
                    (cost.rows * 9.0, 27)
                } else {
                    (cost.rows * kept.max(2.0).log2(), 1)
                };
                cost.device_work = cost.device_work.saturating_add_f64(work * backend_factor);
                if input.backend == BackendKind::Metal {
                    cost.peak_scratch_bytes = cost
                        .peak_scratch_bytes
                        .max(estimated_metal_sort_scratch(cost.rows, kept));
                }
                cost.rows = kept;
                cost.launches = cost.launches.saturating_add(launches);
                cost.synchronizations = cost.synchronizations.saturating_add(1);
                cost.materialization_bytes = cost
                    .materialization_bytes
                    .max(estimated_row_bytes(kept, scope.len()));
            }
            PhysicalOperator::Limit(Expression::Literal(ScalarValue::Integer(limit))) => {
                if *limit >= 0 {
                    cost.rows = cost.rows.min(*limit as f64);
                }
            }
            PhysicalOperator::Skip(Expression::Literal(ScalarValue::Integer(skip))) => {
                if *skip >= 0 {
                    cost.rows = (cost.rows - *skip as f64).max(0.0);
                }
            }
            PhysicalOperator::TemporalHistory(_)
            | PhysicalOperator::TemporalWindow(_)
            | PhysicalOperator::TemporalHistoryWindow { .. } => {
                cost.launches = cost.launches.saturating_add(1);
                cost.synchronizations = cost.synchronizations.saturating_add(1);
                let temporal_rows = input
                    .statistics
                    .temporal_columns()
                    .iter()
                    .map(|column| column.sample_count)
                    .fold(0_u64, u64::saturating_add) as f64;
                cost.temporal_work = cost
                    .temporal_work
                    .saturating_add_f64(temporal_rows.max(cost.rows));
                cost.device_work = cost
                    .device_work
                    .saturating_add_f64(temporal_rows.max(cost.rows) * backend_factor);
            }
            PhysicalOperator::VectorSearch { search, access } => {
                cost.launches = cost.launches.saturating_add(1);
                cost.synchronizations = cost.synchronizations.saturating_add(1);
                let access_scratch = match access {
                    VectorAccessPath::Exact { scratch_bytes, .. }
                    | VectorAccessPath::IvfPq { scratch_bytes, .. } => *scratch_bytes,
                    VectorAccessPath::Unspecified => 0,
                };
                cost.peak_scratch_bytes = cost.peak_scratch_bytes.max(access_scratch);
                let candidate_budget = match access {
                    VectorAccessPath::IvfPq {
                        candidate_budget, ..
                    } => f64::from(*candidate_budget),
                    VectorAccessPath::Exact { filtered_rows, .. } => *filtered_rows as f64,
                    VectorAccessPath::Unspecified => input
                        .statistics
                        .online_indexes()
                        .iter()
                        .find(|index| index.name == search.index)
                        .map_or(cost.rows, |index| index.ann_candidate_budget as f64),
                };
                if access.uses_ivf_pq() {
                    cost.ann_work = cost.ann_work.saturating_add_f64(candidate_budget);
                }
                cost.device_work = cost
                    .device_work
                    .saturating_add_f64(candidate_budget * backend_factor);
            }
            PhysicalOperator::Unwind { .. } | PhysicalOperator::BuiltinProcedure(_) => {
                cost.launches = cost.launches.saturating_add(1);
                cost.synchronizations = cost.synchronizations.saturating_add(1);
                cost.device_work = cost
                    .device_work
                    .saturating_add_f64(cost.rows * backend_factor);
            }
            PhysicalOperator::Project { projection, .. } => {
                cost.device_work = cost.device_work.saturating_add_f64(
                    cost.rows * projection.items.len().max(1) as f64 * backend_factor,
                );
                cost.materialization_bytes = cost
                    .materialization_bytes
                    .max(estimated_row_bytes(cost.rows, projection.items.len()));
                cost.launches = cost.launches.saturating_add(1);
            }
            _ => {}
        }
        cost.peak_scratch_bytes = cost
            .peak_scratch_bytes
            .max(estimated_row_bytes(cost.rows, scope.len()));
        update_scope(operator, &mut scope);
    }
    let result_bytes = estimated_row_bytes(cost.rows, scope.len());
    cost.materialization_bytes = cost.materialization_bytes.max(result_bytes);
    if input.backend != BackendKind::Cpu {
        cost.transfer_bytes = cost.transfer_bytes.saturating_add(result_bytes);
        cost.host_boundaries = cost.host_boundaries.saturating_add(1);
    }
    cost
}

fn access_estimated_rows(access: &ScanAccessPath, fallback: f64) -> f64 {
    match access {
        ScanAccessPath::StableId { .. } => fallback.min(1.0),
        ScanAccessPath::EqualityIndex { estimated_rows, .. }
        | ScanAccessPath::ResidentProperty { estimated_rows, .. } => {
            fallback.min(*estimated_rows as f64)
        }
        _ => fallback,
    }
}

fn estimate_pattern(pattern: &Pattern, layers: LayerMask, input: OptimizerInput<'_>) -> f64 {
    if let Some(rows) = input
        .runtime_feedback
        .and_then(|feedback| feedback.get(&pattern_structural_key(pattern)))
    {
        return (*rows).min(input.max_result_rows as u64) as f64;
    }
    let mut rows = estimate_node(&pattern.start, layers, input);
    for step in &pattern.steps {
        let types = step
            .relationship
            .types
            .iter()
            .filter_map(|name| input.catalog.relationship_type(name))
            .collect::<Vec<_>>();
        let fanout = match step.relationship.direction {
            Direction::Outgoing => input.statistics.directional_fanout(&types, layers, true),
            Direction::Incoming => input.statistics.directional_fanout(&types, layers, false),
            Direction::Undirected => {
                input.statistics.directional_fanout(&types, layers, true)
                    + input.statistics.directional_fanout(&types, layers, false)
            }
        };
        let target = estimate_node(&step.node, layers, input);
        let total = input.statistics.node_count(layers) as f64;
        let target_fraction = if total == 0.0 { 0.0 } else { target / total };
        let (minimum, maximum) = if step.relationship.variable_length {
            (
                step.relationship.min_hops.unwrap_or(1),
                step.relationship.max_hops.unwrap_or(8).min(64),
            )
        } else {
            (1, 1)
        };
        let mut expansion = 0.0;
        let per_hop = fanout * target_fraction;
        for hops in minimum..=maximum {
            expansion += per_hop.powi(hops as i32);
        }
        rows *= expansion;
        if !rows.is_finite() {
            return input.max_result_rows as f64;
        }
    }
    rows.min(input.max_result_rows as f64)
}

fn estimate_node(node: &NodePattern, layers: LayerMask, input: OptimizerInput<'_>) -> f64 {
    let total = input.statistics.node_count(layers);
    if total == 0 {
        return 0.0;
    }
    let mut rows = total as f64;
    for label in &node.labels {
        let count = input
            .catalog
            .label(label)
            .map_or(0, |label| input.statistics.label_count(label, layers));
        rows = rows.min(count as f64);
    }
    for (name, _) in &node.properties {
        let selectivity = input
            .catalog
            .property(name)
            .and_then(|property| input.statistics.node_property(property, layers))
            .map_or(0.0, |stats| stats.equality_selectivity(total));
        rows *= selectivity;
    }
    rows.max(0.0)
}

fn bounded_product(left: f64, right: f64, maximum: usize) -> f64 {
    let product = left * right;
    if product.is_finite() {
        product.min(maximum as f64)
    } else {
        maximum as f64
    }
}

fn estimated_row_bytes(rows: f64, bindings: usize) -> u64 {
    let width = (bindings.max(1) as u64).saturating_mul(ESTIMATED_BINDING_BYTES);
    (rows.max(0.0).ceil() as u64).saturating_mul(width)
}

fn estimated_metal_sort_scratch(input_rows: f64, output_rows: f64) -> u64 {
    let input = input_rows.max(0.0).ceil() as usize;
    let output = output_rows.max(0.0).ceil() as usize;
    metal_i64_sort_scratch_bytes(input, output).map_or(u64::MAX, |bytes| bytes as u64)
}

fn estimated_metal_top_k_passes(rows: f64, limit: usize) -> u32 {
    if rows <= 0.0 || limit == 0 {
        return 0;
    }
    let mut candidates = rows.ceil() as u64;
    let limit = limit as u64;
    let mut passes = 0_u32;
    loop {
        passes = passes.saturating_add(1);
        let full_tiles = candidates / 1_024;
        let remainder = candidates % 1_024;
        candidates = full_tiles
            .saturating_mul(limit)
            .saturating_add(remainder.min(limit));
        if candidates <= limit {
            return passes;
        }
    }
}

fn split_conjunction(expression: Expression) -> Vec<Expression> {
    match expression {
        Expression::Binary {
            left,
            operation: BinaryOperator::And,
            right,
        } => {
            let mut output = split_conjunction(*left);
            output.extend(split_conjunction(*right));
            output
        }
        expression => vec![expression],
    }
}

fn pattern_variables(pattern: &Pattern) -> BTreeSet<String> {
    pattern
        .variable
        .iter()
        .chain(pattern.start.variable.iter())
        .chain(pattern.steps.iter().flat_map(|step| {
            step.relationship
                .variable
                .iter()
                .chain(step.node.variable.iter())
        }))
        .cloned()
        .collect()
}

fn pattern_external_requirements(pattern: &Pattern) -> BTreeSet<String> {
    let local = pattern_variables(pattern);
    pattern
        .start
        .properties
        .iter()
        .chain(pattern.steps.iter().flat_map(|step| {
            step.relationship
                .properties
                .iter()
                .chain(step.node.properties.iter())
        }))
        .flat_map(|(_, expression)| expression_variables(expression))
        .filter(|variable| !local.contains(variable))
        .collect()
}

pub(crate) fn expression_variables(expression: &Expression) -> BTreeSet<String> {
    let mut output = BTreeSet::new();
    collect_expression_variables(expression, &mut output, &BTreeSet::new());
    output
}

fn collect_expression_variables(
    expression: &Expression,
    output: &mut BTreeSet<String>,
    locals: &BTreeSet<String>,
) {
    match expression {
        Expression::Variable(variable) => {
            if !locals.contains(variable) {
                output.insert(variable.clone());
            }
        }
        Expression::Property(source, _)
        | Expression::Unary {
            operand: source, ..
        }
        | Expression::IsNull {
            expression: source, ..
        } => collect_expression_variables(source, output, locals),
        Expression::List(values)
        | Expression::Function {
            arguments: values, ..
        } => {
            for value in values {
                collect_expression_variables(value, output, locals);
            }
        }
        Expression::Map(values) => {
            for (_, value) in values {
                collect_expression_variables(value, output, locals);
            }
        }
        Expression::MapProjection { source, items } => {
            collect_expression_variables(source, output, locals);
            for item in items {
                match item {
                    MapProjectionItem::Variable(variable) if !locals.contains(variable) => {
                        output.insert(variable.clone());
                    }
                    MapProjectionItem::Entry(_, expression) => {
                        collect_expression_variables(expression, output, locals)
                    }
                    _ => {}
                }
            }
        }
        Expression::Case {
            operand,
            alternatives,
            default,
        } => {
            if let Some(operand) = operand {
                collect_expression_variables(operand, output, locals);
            }
            for alternative in alternatives {
                collect_expression_variables(&alternative.when, output, locals);
                collect_expression_variables(&alternative.then, output, locals);
            }
            if let Some(default) = default {
                collect_expression_variables(default, output, locals);
            }
        }
        Expression::ListComprehension {
            variable,
            list,
            predicate,
            projection,
        } => {
            collect_expression_variables(list, output, locals);
            let mut nested = locals.clone();
            nested.insert(variable.clone());
            if let Some(predicate) = predicate {
                collect_expression_variables(predicate, output, &nested);
            }
            if let Some(projection) = projection {
                collect_expression_variables(projection, output, &nested);
            }
        }
        Expression::Reduce {
            accumulator,
            initial,
            variable,
            list,
            expression,
        } => {
            collect_expression_variables(initial, output, locals);
            collect_expression_variables(list, output, locals);
            let mut nested = locals.clone();
            nested.insert(accumulator.clone());
            nested.insert(variable.clone());
            collect_expression_variables(expression, output, &nested);
        }
        Expression::ListPredicate {
            variable,
            list,
            predicate,
            ..
        } => {
            collect_expression_variables(list, output, locals);
            let mut nested = locals.clone();
            nested.insert(variable.clone());
            collect_expression_variables(predicate, output, &nested);
        }
        Expression::Binary { left, right, .. } => {
            collect_expression_variables(left, output, locals);
            collect_expression_variables(right, output, locals);
        }
        Expression::Index { expression, index } => {
            collect_expression_variables(expression, output, locals);
            collect_expression_variables(index, output, locals);
        }
        Expression::Slice {
            expression,
            start,
            end,
        } => {
            collect_expression_variables(expression, output, locals);
            if let Some(start) = start {
                collect_expression_variables(start, output, locals);
            }
            if let Some(end) = end {
                collect_expression_variables(end, output, locals);
            }
        }
        Expression::Literal(_)
        | Expression::Parameter(_)
        | Expression::ExistentialSubquery(_)
        | Expression::Star => {}
    }
}

fn is_pushdown_safe(expression: &Expression) -> bool {
    match expression {
        Expression::Literal(_) | Expression::Parameter(_) | Expression::Variable(_) => true,
        Expression::Property(source, _)
        | Expression::Unary {
            operand: source, ..
        }
        | Expression::IsNull {
            expression: source, ..
        } => is_pushdown_safe(source),
        Expression::List(values) => values.iter().all(is_pushdown_safe),
        Expression::Map(values) => values.iter().all(|(_, value)| is_pushdown_safe(value)),
        Expression::Binary { left, right, .. } => is_pushdown_safe(left) && is_pushdown_safe(right),
        Expression::Index { expression, index } => {
            is_pushdown_safe(expression) && is_pushdown_safe(index)
        }
        Expression::Slice {
            expression,
            start,
            end,
        } => {
            is_pushdown_safe(expression)
                && start.as_deref().is_none_or(is_pushdown_safe)
                && end.as_deref().is_none_or(is_pushdown_safe)
        }
        // Functions, comprehensions, reductions, CASE, and map projections are kept at their
        // original barrier until volatility and exception behavior are encoded by the binder.
        _ => false,
    }
}

fn is_constant_expression(expression: &Expression) -> bool {
    match expression {
        Expression::Literal(_) | Expression::Parameter(_) => true,
        Expression::List(values) => values.iter().all(is_constant_expression),
        Expression::Map(values) => values
            .iter()
            .all(|(_, value)| is_constant_expression(value)),
        Expression::Unary { operand, .. } => is_constant_expression(operand),
        Expression::Binary { left, right, .. } => {
            is_constant_expression(left) && is_constant_expression(right)
        }
        _ => false,
    }
}

fn update_scope(operator: &PhysicalOperator, scope: &mut BTreeSet<String>) {
    match operator {
        PhysicalOperator::ScanPattern { pattern, .. }
        | PhysicalOperator::CreatePattern(pattern)
        | PhysicalOperator::MergePattern { pattern, .. } => {
            scope.extend(pattern_variables(pattern));
        }
        PhysicalOperator::CyclicMultiwayJoin { patterns, .. } => {
            for (pattern, _) in patterns {
                scope.extend(pattern_variables(pattern));
            }
        }
        PhysicalOperator::Unwind { variable, .. } => {
            scope.insert(variable.clone());
        }
        PhysicalOperator::Project {
            keep_scope,
            projection,
        } => update_projection_scope(*keep_scope, projection, scope),
        _ => {}
    }
}

fn update_projection_scope(
    keep_scope: bool,
    projection: &Projection,
    scope: &mut BTreeSet<String>,
) {
    let mut projected = if keep_scope
        || projection
            .items
            .iter()
            .any(|item| matches!(item.expression, Expression::Star))
    {
        scope.clone()
    } else {
        BTreeSet::new()
    };
    for (position, item) in projection.items.iter().enumerate() {
        if matches!(item.expression, Expression::Star) {
            continue;
        }
        projected.insert(
            item.alias
                .clone()
                .unwrap_or_else(|| match &item.expression {
                    Expression::Variable(variable) => variable.clone(),
                    _ => format!("expression_{position}"),
                }),
        );
    }
    *scope = projected;
}

trait SaturatingFloat {
    fn saturating_add_f64(self, other: f64) -> f64;
}

impl SaturatingFloat for f64 {
    fn saturating_add_f64(self, other: f64) -> f64 {
        let sum = self + other;
        if sum.is_finite() { sum } else { f64::MAX }
    }
}

#[cfg(test)]
mod tests {
    use std::{collections::BTreeMap, sync::Arc, time::Instant};

    use ordered_float::OrderedFloat;
    use tokio_util::sync::CancellationToken;

    use crate::{
        Bookmark, EdgeId, Error, ErrorCode, Layer, NodeId, ProjectId, Result,
        cypher::{ExecutionContext, QueryEngine, ResultValue, bind, parse, plan},
        execution::{CpuBackend, ExecutionBackend, ResidentSortRequest},
        graph::{
            EdgeInput, GraphIndexDefinition, GraphIndexKind, GraphMutation, GraphStore,
            IndexCatalog, NodeInput,
        },
    };

    use super::*;

    fn optimized(source: &str, graph: &GraphStore) -> Result<(PhysicalPlan, OptimizationProfile)> {
        let query = parse(source)?;
        let bound = bind(
            query,
            graph.catalog(),
            super::super::BindCapabilities {
                write: true,
                schema: true,
                knowledge_write: true,
                workspace_write: true,
                require_native_execution: false,
            },
        )?;
        let physical = plan(bound)?;
        let statistics = StatisticsSnapshot::collect(graph);
        Ok(optimize(
            physical,
            OptimizerInput {
                statistics: &statistics,
                catalog: graph.catalog(),
                indexes: None,
                parameters: &BTreeMap::new(),
                backend: BackendKind::Metal,
                scratch_budget_bytes: 64 * 1024 * 1024,
                max_result_rows: 1_000_000,
                runtime_feedback: None,
                allow_runtime_checkpoint: true,
            },
        ))
    }

    fn optimized_cpu(source: &str, graph: &GraphStore) -> Result<PhysicalPlan> {
        let query = parse(source)?;
        let bound = bind(
            query,
            graph.catalog(),
            super::super::BindCapabilities::default(),
        )?;
        let statistics = StatisticsSnapshot::collect(graph);
        Ok(optimize(
            plan(bound)?,
            OptimizerInput {
                statistics: &statistics,
                catalog: graph.catalog(),
                indexes: None,
                parameters: &BTreeMap::new(),
                backend: BackendKind::Cpu,
                scratch_budget_bytes: usize::MAX,
                max_result_rows: 100_000,
                runtime_feedback: None,
                allow_runtime_checkpoint: true,
            },
        )
        .0)
    }

    #[test]
    fn selective_pattern_moves_before_cartesian_scan() -> Result<()> {
        let mut graph = GraphStore::default();
        let common = graph.catalog_mut().intern_label("Common")?;
        let rare = graph.catalog_mut().intern_label("Rare")?;
        for id in 1..=10 {
            graph.apply(GraphMutation::InsertNode(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: id,
                labels: if id == 1 {
                    vec![common, rare]
                } else {
                    vec![common]
                },
                properties: Vec::new(),
            }))?;
        }
        let (plan, _) = optimized("MATCH (a:Common), (b:Rare) RETURN a, b", &graph)?;
        let Some(PhysicalOperator::ScanPattern { pattern, .. }) = plan.operators.first() else {
            return Err(Error::internal(
                "first optimized operator is not a pattern scan",
            ));
        };
        assert_eq!(pattern.start.variable.as_deref(), Some("b"));
        Ok(())
    }

    #[test]
    fn scan_reordering_stays_inside_each_match_group() -> Result<()> {
        let mut graph = GraphStore::default();
        let common = graph.catalog_mut().intern_label("Common")?;
        let rare = graph.catalog_mut().intern_label("Rare")?;
        for id in 1..=10 {
            graph.apply(GraphMutation::InsertNode(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: id,
                labels: if id == 1 {
                    vec![common, rare]
                } else {
                    vec![common]
                },
                properties: Vec::new(),
            }))?;
        }

        let plan = optimized_cpu(
            "MATCH (a:Common), (b:Rare) MATCH (c:Common), (d:Rare) RETURN a, b, c, d",
            &graph,
        )?;
        let scans = plan
            .operators
            .iter()
            .filter_map(|operator| match operator {
                PhysicalOperator::ScanPattern {
                    match_group,
                    pattern,
                    ..
                } => Some((*match_group, pattern.start.variable.as_deref())),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(scans.len(), 4);
        assert_eq!(
            scans
                .iter()
                .map(|(_, variable)| *variable)
                .collect::<Vec<_>>(),
            vec![Some("b"), Some("a"), Some("d"), Some("c")]
        );
        assert_eq!(scans[0].0, scans[1].0);
        assert_eq!(scans[2].0, scans[3].0);
        assert_ne!(scans[1].0, scans[2].0);
        Ok(())
    }

    #[test]
    fn optional_pattern_does_not_stall_cyclic_lowering() -> Result<()> {
        let graph = GraphStore::default();
        let (plan, _) = optimized("OPTIONAL MATCH (n) RETURN n", &graph)?;
        assert!(plan.operators.iter().any(|operator| matches!(
            operator,
            PhysicalOperator::ScanPattern { optional: true, .. }
        )));
        Ok(())
    }

    #[test]
    fn predicates_move_only_after_their_variables_exist() -> Result<()> {
        let mut graph = GraphStore::default();
        graph.catalog_mut().intern_label("A")?;
        graph.catalog_mut().intern_label("B")?;
        graph.catalog_mut().intern_property("x")?;
        let (plan, _) = optimized(
            "MATCH (a:A), (b:B) WHERE a.x = 1 AND b.x = 2 RETURN a, b",
            &graph,
        )?;
        let positions = plan
            .operators
            .iter()
            .enumerate()
            .filter_map(|(index, operator)| {
                matches!(operator, PhysicalOperator::Filter(_)).then_some(index)
            })
            .collect::<Vec<_>>();
        assert_eq!(positions.len(), 2);
        assert!(positions[0] > 0);
        assert!(positions[1] > positions[0]);
        Ok(())
    }

    #[test]
    fn adjacent_literal_limit_becomes_top_k() -> Result<()> {
        let graph = GraphStore::default();
        let (plan, profile) =
            optimized("UNWIND [3, 1, 2] AS x RETURN x ORDER BY x LIMIT 2", &graph)?;
        assert!(
            plan.operators
                .iter()
                .any(|operator| matches!(operator, PhysicalOperator::TopK { limit: 2, .. }))
        );
        assert_eq!(profile.backend, BackendKind::Metal);
        Ok(())
    }

    #[test]
    fn with_orderby1_static_distinct_top_one_uses_the_native_sorter() -> Result<()> {
        let graph = GraphStore::default();
        let backend = CpuBackend::new(64 * 1024 * 1024, 0);
        for (source, expected, descending) in [
            (
                "UNWIND [0,2,1,2,0,1] AS x WITH DISTINCT x ORDER BY x ASC LIMIT 1 RETURN x",
                0,
                false,
            ),
            (
                "UNWIND [0,2,1,2,0,1] AS x WITH DISTINCT x ORDER BY x DESC LIMIT 1 RETURN x",
                2,
                true,
            ),
        ] {
            let raw = plan(bind(
                parse(source)?,
                graph.catalog(),
                super::super::BindCapabilities::default(),
            )?)?;
            assert!(raw.operators.iter().any(|operator| matches!(
                operator,
                PhysicalOperator::Project { projection, .. } if projection.distinct
            )));
            assert!(
                super::super::resident::compile_literal_sort(&raw, &BTreeMap::new(), 100)?
                    .is_none(),
                "unoptimized DISTINCT unexpectedly entered the native sorter: {source}"
            );

            let (metal, _) = optimized(source, &graph)?;
            assert!(metal.operators.iter().all(|operator| !matches!(
                operator,
                PhysicalOperator::Project { projection, .. } if projection.distinct
            )));
            let values = metal
                .operators
                .iter()
                .find_map(|operator| match operator {
                    PhysicalOperator::Unwind {
                        expression: Expression::List(values),
                        variable,
                    } if variable == "x" => Some(values),
                    _ => None,
                })
                .ok_or_else(|| Error::internal("static DISTINCT rewrite omitted UNWIND"))?;
            assert_eq!(
                values,
                &[
                    Expression::Literal(ScalarValue::Integer(0)),
                    Expression::Literal(ScalarValue::Integer(2)),
                    Expression::Literal(ScalarValue::Integer(1)),
                ],
                "query: {source}"
            );
            assert!(
                metal
                    .operators
                    .iter()
                    .any(|operator| matches!(operator, PhysicalOperator::TopK { limit: 1, .. }))
            );

            let compiled =
                super::super::resident::compile_literal_sort(&metal, &BTreeMap::new(), 100)?
                    .ok_or_else(|| {
                        Error::internal(format!(
                            "static DISTINCT did not enter the native sorter: {source}"
                        ))
                    })?;
            assert_eq!(compiled.output_names, ["x"]);
            assert_eq!(compiled.rows.len(), 3);
            assert_eq!(compiled.device_limit, Some(1));
            assert_eq!(compiled.keys.len(), 1);
            assert_eq!(compiled.keys[0].descending, descending);

            let sorted = backend.sort_rows(
                &ResidentSortRequest {
                    project: ProjectId(uuid::Uuid::nil()),
                    row_count: compiled.rows.len(),
                    keys: compiled.keys.clone(),
                    limit: compiled.device_limit,
                },
                &CancellationToken::new(),
            )?;
            let [position] = sorted.positions.as_slice() else {
                return Err(Error::internal(
                    "native DISTINCT TopK did not return exactly one position",
                ));
            };
            let position = usize::try_from(*position)
                .map_err(|_| Error::internal("native DISTINCT TopK position exceeds usize"))?;
            assert_eq!(
                compiled.rows[position].get("x"),
                Some(&ResultValue::Scalar(ScalarValue::Integer(expected))),
                "query: {source}"
            );

            let cpu = optimized_cpu(source, &graph)?;
            assert!(cpu.operators.iter().any(|operator| matches!(
                operator,
                PhysicalOperator::Project { projection, .. } if projection.distinct
            )));
            assert!(
                super::super::resident::compile_literal_sort(&cpu, &BTreeMap::new(), 100)?
                    .is_none(),
                "CPU semantic plan must retain its real DISTINCT: {source}"
            );
            let mut context = test_context(&graph);
            let output = QueryEngine.execute(source, &mut context)?;
            assert_eq!(
                output
                    .result
                    .batches
                    .iter()
                    .flat_map(|batch| &batch.columns)
                    .find(|column| column.name == "x")
                    .and_then(|column| column.values.first()),
                Some(&ResultValue::Scalar(ScalarValue::Integer(expected))),
                "query: {source}"
            );
        }
        Ok(())
    }

    #[test]
    fn with_orderby1_static_distinct_rewrite_keeps_neighbors_fail_closed() -> Result<()> {
        let graph = GraphStore::default();
        for source in [
            "UNWIND [0,2,1,2,0,1] AS x WITH DISTINCT x ORDER BY x LIMIT 2 RETURN x",
            "UNWIND [0,2,1,2,0,1] AS x WITH DISTINCT x ORDER BY x RETURN x",
            "UNWIND $values AS x WITH DISTINCT x ORDER BY x LIMIT 1 RETURN x",
            "UNWIND [0,1 + 1,1] AS x WITH DISTINCT x ORDER BY x LIMIT 1 RETURN x",
            "UNWIND [0,'zero',0] AS x WITH DISTINCT x ORDER BY x LIMIT 1 RETURN x",
            "UNWIND [0,2,1,2,0,1] AS x WITH DISTINCT x, x AS y ORDER BY x LIMIT 1 RETURN x",
            "UNWIND [0,2,1,2,0,1] AS x WITH DISTINCT x ORDER BY x LIMIT 1 RETURN x AS y",
        ] {
            let (metal, _) = optimized(source, &graph)?;
            assert!(
                metal.operators.iter().any(|operator| matches!(
                    operator,
                    PhysicalOperator::Project { projection, .. } if projection.distinct
                )),
                "neighboring shape was incorrectly rewritten: {source}"
            );
            assert!(
                super::super::resident::compile_literal_sort(&metal, &BTreeMap::new(), 100)?
                    .is_none(),
                "neighboring shape was incorrectly admitted: {source}"
            );
        }
        Ok(())
    }

    #[test]
    fn return_orderby2_unique_node_distinct_uses_the_native_row_sorter() -> Result<()> {
        let mut graph = GraphStore::default();
        let name = graph.catalog_mut().intern_property("name")?;
        let id = graph.catalog_mut().intern_property("id")?;
        for (node, text, number) in [(1, "A", 1), (2, "B", 10), (3, "C", 20)] {
            graph.apply(GraphMutation::InsertNode(NodeInput {
                id: NodeId(node),
                layer: Layer::Observed,
                revision: node,
                labels: Vec::new(),
                properties: vec![
                    (name, ScalarValue::String(Arc::from(text))),
                    (id, ScalarValue::Integer(number)),
                ],
            }))?;
        }

        for (source, output) in [
            ("MATCH (a) RETURN DISTINCT a ORDER BY a.name", "a"),
            ("MATCH (n) RETURN DISTINCT n ORDER BY n.id", "n"),
        ] {
            let raw = plan(bind(
                parse(source)?,
                graph.catalog(),
                super::super::BindCapabilities::default(),
            )?)?;
            assert!(
                super::super::resident_row::compile(
                    &raw,
                    ProjectId(uuid::Uuid::nil()),
                    Bookmark {
                        term: 1,
                        index: graph.revision(),
                    },
                    graph.catalog(),
                    &graph,
                    &BTreeMap::new(),
                    100,
                )?
                .is_none(),
                "query: {source}"
            );

            let (metal, _) = optimized(source, &graph)?;
            assert!(metal.operators.iter().all(|operator| {
                !matches!(
                    operator,
                    PhysicalOperator::Project { projection, .. } if projection.distinct
                )
            }));
            let compiled = super::super::resident_row::compile(
                &metal,
                ProjectId(uuid::Uuid::nil()),
                Bookmark {
                    term: 1,
                    index: graph.revision(),
                },
                graph.catalog(),
                &graph,
                &BTreeMap::new(),
                100,
            )?
            .ok_or_else(|| {
                Error::internal(format!(
                    "unique-node DISTINCT did not enter the native row sorter: {source}"
                ))
            })?;
            assert_eq!(compiled.outputs.len(), 1);
            assert_eq!(compiled.outputs[0].name, output);

            let cpu = optimized_cpu(source, &graph)?;
            assert!(cpu.operators.iter().any(|operator| matches!(
                operator,
                PhysicalOperator::Project { projection, .. } if projection.distinct
            )));
        }
        Ok(())
    }

    #[test]
    fn metal_static_union_tck_cases_use_the_native_literal_relation() -> Result<()> {
        let graph = GraphStore::default();
        let backend = CpuBackend::new(64 * 1024 * 1024, 0);
        let cases: &[(&str, &[i64])] = &[
            ("RETURN 1 AS x UNION RETURN 2 AS x", &[1, 2]),
            (
                "RETURN 2 AS x UNION RETURN 1 AS x UNION RETURN 2 AS x",
                &[2, 1],
            ),
            (
                "UNWIND [2, 1, 2, 3] AS x RETURN x UNION UNWIND [3, 4] AS x RETURN x",
                &[2, 1, 3, 4],
            ),
            ("RETURN 1 AS x UNION ALL RETURN 2 AS x", &[1, 2]),
            (
                "RETURN 2 AS x UNION ALL RETURN 1 AS x UNION ALL RETURN 2 AS x",
                &[2, 1, 2],
            ),
            (
                "UNWIND [2, 1, 2, 3] AS x RETURN x UNION ALL UNWIND [3, 4] AS x RETURN x",
                &[2, 1, 2, 3, 3, 4],
            ),
        ];
        for &(source, expected_values) in cases {
            let (metal, _) = optimized(source, &graph)?;
            assert!(metal.unions.is_empty(), "query: {source}");
            assert!(
                metal.operators.iter().any(|operator| matches!(
                    operator,
                    PhysicalOperator::Sort(items)
                        if matches!(items.as_slice(), [SortItem {
                            expression: Expression::Literal(ScalarValue::Integer(0)),
                            ascending: true,
                        }])
                )),
                "static UNION omitted its explicit stable-order key: {source}"
            );
            let compiled =
                super::super::resident::compile_literal_sort(&metal, &BTreeMap::new(), 100)?
                    .ok_or_else(|| {
                        Error::internal(format!(
                            "static UNION did not enter the native literal relation: {source}"
                        ))
                    })?;
            assert_eq!(compiled.output_names, ["x"], "query: {source}");
            assert!(!compiled.keys.is_empty(), "query: {source}");
            let sorted = backend.sort_rows(
                &ResidentSortRequest {
                    project: ProjectId(uuid::Uuid::nil()),
                    row_count: compiled.rows.len(),
                    keys: compiled.keys.clone(),
                    limit: compiled.device_limit,
                },
                &CancellationToken::new(),
            )?;
            assert!(
                sorted
                    .positions
                    .iter()
                    .enumerate()
                    .all(|(index, position)| usize::try_from(*position) == Ok(index)),
                "native stable sort changed UNION branch order: {source}"
            );
            let values = sorted
                .positions
                .iter()
                .map(|position| {
                    let position = usize::try_from(*position)
                        .map_err(|_| Error::internal("native UNION sort position exceeds usize"))?;
                    match compiled.rows[position].get("x") {
                        Some(ResultValue::Scalar(ScalarValue::Integer(value))) => Ok(*value),
                        _ => Err(Error::internal(format!(
                            "static UNION produced a non-integer row: {source}"
                        ))),
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            assert_eq!(values, expected_values, "query: {source}");

            let mut context = test_context(&graph);
            let cpu = QueryEngine.execute(source, &mut context)?;
            let cpu_values = cpu
                .result
                .batches
                .iter()
                .flat_map(|batch| batch.columns.iter())
                .filter(|column| column.name == "x")
                .flat_map(|column| column.values.iter())
                .map(|value| match value {
                    ResultValue::Scalar(ScalarValue::Integer(value)) => Ok(*value),
                    _ => Err(Error::internal(format!(
                        "CPU UNION produced a non-integer row: {source}"
                    ))),
                })
                .collect::<Result<Vec<_>>>()?;
            assert_eq!(cpu_values, expected_values, "CPU query: {source}");
        }

        let cpu = optimized_cpu("RETURN 1 AS x UNION RETURN 2 AS x", &graph)?;
        assert_eq!(cpu.unions.len(), 1);
        Ok(())
    }

    #[test]
    fn metal_static_union_keeps_dynamic_or_heterogeneous_relations_fail_closed() -> Result<()> {
        let graph = GraphStore::default();
        for source in [
            "MATCH (a:A) RETURN a AS x UNION MATCH (b:B) RETURN b AS x",
            "RETURN 1 AS x UNION RETURN 'one' AS x",
            "RETURN $value AS x UNION RETURN 2 AS x",
        ] {
            let (metal, _) = optimized(source, &graph)?;
            assert!(!metal.unions.is_empty(), "query: {source}");
            assert!(
                super::super::resident::compile_literal_sort(&metal, &BTreeMap::new(), 100)?
                    .is_none(),
                "query: {source}"
            );
        }
        Ok(())
    }

    #[test]
    fn with1_path_identity_boundary_exposes_existing_native_path_compiler() -> Result<()> {
        let graph = GraphStore::default();
        let source = "MATCH p = (a) WITH p RETURN p";
        let query = parse(source)?;
        let bound = bind(
            query,
            graph.catalog(),
            super::super::BindCapabilities::default(),
        )?;
        let unoptimized = plan(bound)?;
        assert!(
            super::super::resident_variable_path::compile(
                &unoptimized,
                ProjectId(uuid::Uuid::nil()),
                Bookmark { term: 1, index: 0 },
                graph.catalog(),
                &graph,
                100,
            )?
            .is_none()
        );

        let (metal, _) = optimized(source, &graph)?;
        assert_eq!(
            metal
                .operators
                .iter()
                .filter(|operator| matches!(operator, PhysicalOperator::Project { .. }))
                .count(),
            1
        );
        let compiled = super::super::resident_variable_path::compile(
            &metal,
            ProjectId(uuid::Uuid::nil()),
            Bookmark { term: 1, index: 0 },
            graph.catalog(),
            &graph,
            100,
        )?
        .ok_or_else(|| Error::internal("WITH path did not enter the native path compiler"))?;
        assert_eq!(compiled.outputs.len(), 1);
        assert_eq!(compiled.outputs[0].name(), "p");

        let cpu = optimized_cpu(source, &graph)?;
        assert_eq!(
            cpu.operators
                .iter()
                .filter(|operator| matches!(operator, PhysicalOperator::Project { .. }))
                .count(),
            2
        );
        Ok(())
    }

    #[test]
    fn metal_projection_boundary_never_drops_a_partial_result_projection() -> Result<()> {
        let graph = GraphStore::default();
        let (metal, _) = optimized("MATCH (a) WITH a, a.name AS name RETURN a", &graph)?;
        assert_eq!(
            metal
                .operators
                .iter()
                .filter(|operator| matches!(operator, PhysicalOperator::Project { .. }))
                .count(),
            2
        );
        Ok(())
    }

    #[test]
    fn metal_sort_cost_matches_runtime_scratch_and_bounded_passes() -> Result<()> {
        const ROWS: usize = 1_000_003;
        const LIMIT: usize = 128;
        assert_eq!(estimated_metal_top_k_passes(ROWS as f64, LIMIT), 5);
        assert_eq!(
            estimated_metal_sort_scratch(ROWS as f64, LIMIT as f64),
            crate::execution::metal_i64_sort_scratch_bytes(ROWS, LIMIT)? as u64
        );
        assert_eq!(estimated_metal_top_k_passes(1_025.0, 256), 2);
        assert_eq!(estimated_metal_top_k_passes(256.0, 256), 1);
        assert_eq!(estimated_metal_top_k_passes(0.0, 256), 0);
        Ok(())
    }

    #[test]
    fn boolean_folding_preserves_error_evaluation() {
        let division = Expression::Binary {
            left: Box::new(Expression::integer(1)),
            operation: BinaryOperator::Divide,
            right: Box::new(Expression::integer(0)),
        };
        let folded = fold_expression(Expression::Binary {
            left: Box::new(Expression::Literal(ScalarValue::Boolean(false))),
            operation: BinaryOperator::And,
            right: Box::new(division.clone()),
        });
        assert!(matches!(
            folded,
            Expression::Binary {
                operation: BinaryOperator::And,
                ..
            }
        ));
        assert!(matches!(
            fold_expression(Expression::Binary {
                left: Box::new(Expression::Literal(ScalarValue::Boolean(true))),
                operation: BinaryOperator::And,
                right: Box::new(Expression::Variable("x".to_owned())),
            }),
            Expression::Binary {
                operation: BinaryOperator::And,
                ..
            }
        ));
    }

    #[test]
    fn optimized_and_source_order_execution_are_equivalent() -> Result<()> {
        let mut graph = GraphStore::default();
        let a = graph.catalog_mut().intern_label("A")?;
        let b = graph.catalog_mut().intern_label("B")?;
        let rare = graph.catalog_mut().intern_label("Rare")?;
        let relationship = graph.catalog_mut().intern_relationship_type("R")?;
        let x = graph.catalog_mut().intern_property("x")?;
        for id in 1..=24_u64 {
            let mut labels = if id % 2 == 0 { vec![a] } else { vec![b] };
            if id == 23 {
                labels.push(rare);
            }
            graph.apply(GraphMutation::InsertNode(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: id,
                labels,
                properties: vec![(x, ScalarValue::Integer((id % 7) as i64))],
            }))?;
        }
        for (edge, source, target) in [(1, 2, 23), (2, 4, 23), (3, 6, 1), (4, 8, 3)] {
            graph.apply(GraphMutation::InsertEdge(EdgeInput {
                id: EdgeId(edge),
                source: NodeId(source),
                target: NodeId(target),
                relationship_type: relationship,
                layer: Layer::Observed,
                revision: 25 + edge,
                properties: Vec::new(),
            }))?;
        }

        for query in [
            "MATCH (a:A), (b:B) WHERE a.x >= 2 AND b.x <> 4 RETURN id(a) AS a, id(b) AS b ORDER BY a, b LIMIT 7",
            "MATCH (a:A)-[:R]->(b:Rare) RETURN id(a) AS a, id(b) AS b ORDER BY a, b",
            "MATCH (a:A), (a)-[:R]->(b) RETURN id(a) AS a, id(b) AS b ORDER BY a, b",
        ] {
            let mut optimized_context = test_context(&graph);
            let optimized = QueryEngine.execute(query, &mut optimized_context)?;
            let mut source_context = test_context(&graph);
            let source = QueryEngine.execute_unoptimized(query, &mut source_context)?;
            assert_eq!(optimized.result, source.result, "query: {query}");
            assert!(optimized.graph_mutations.is_empty(), "query: {query}");
            assert!(source.graph_mutations.is_empty(), "query: {query}");
            assert_eq!(
                optimized.temporal_mutations.len(),
                source.temporal_mutations.len(),
                "query: {query}"
            );
        }
        Ok(())
    }

    #[test]
    fn fixed_hop_cycle_uses_multiway_intersection_and_preserves_results() -> Result<()> {
        let mut graph = GraphStore::default();
        let relationship = graph.catalog_mut().intern_relationship_type("LINK")?;
        for id in 1..=4_u64 {
            graph.apply(GraphMutation::InsertNode(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: id,
                labels: Vec::new(),
                properties: Vec::new(),
            }))?;
        }
        for (id, source, target) in [(1, 1, 2), (2, 2, 3), (3, 3, 1), (4, 1, 4), (5, 4, 2)] {
            graph.apply(GraphMutation::InsertEdge(EdgeInput {
                id: EdgeId(id),
                source: NodeId(source),
                target: NodeId(target),
                relationship_type: relationship,
                layer: Layer::Observed,
                revision: 10 + id,
                properties: Vec::new(),
            }))?;
        }
        let source = "MATCH (a)-[:LINK]->(b), (b)-[:LINK]->(c), (c)-[:LINK]->(a) RETURN id(a) AS a, id(b) AS b, id(c) AS c ORDER BY a, b, c";
        let query = parse(source)?;
        let bound = bind(
            query,
            graph.catalog(),
            super::super::BindCapabilities::default(),
        )?;
        let statistics = StatisticsSnapshot::collect(&graph);
        let (physical, _) = optimize(
            plan(bound)?,
            OptimizerInput {
                statistics: &statistics,
                catalog: graph.catalog(),
                indexes: None,
                parameters: &BTreeMap::new(),
                backend: BackendKind::Cpu,
                scratch_budget_bytes: usize::MAX,
                max_result_rows: 100_000,
                runtime_feedback: None,
                allow_runtime_checkpoint: true,
            },
        );
        assert!(physical.operators.iter().any(|operator| matches!(
            operator,
            PhysicalOperator::CyclicMultiwayJoin { patterns, .. } if patterns.len() == 3
        )));

        let mut optimized_context = test_context(&graph);
        let optimized = QueryEngine.execute(source, &mut optimized_context)?;
        let mut source_context = test_context(&graph);
        let unoptimized = QueryEngine.execute_unoptimized(source, &mut source_context)?;
        assert_eq!(optimized.result, unoptimized.result);
        Ok(())
    }

    #[test]
    fn cyclic_lowering_never_merges_separate_match_groups() -> Result<()> {
        let mut graph = GraphStore::default();
        graph.catalog_mut().intern_relationship_type("LINK")?;
        let plan = optimized_cpu(
            "MATCH (a)-[:LINK]->(b) MATCH (b)-[:LINK]->(c) MATCH (c)-[:LINK]->(a) RETURN a, b, c",
            &graph,
        )?;

        assert!(
            !plan
                .operators
                .iter()
                .any(|operator| matches!(operator, PhysicalOperator::CyclicMultiwayJoin { .. }))
        );
        let groups = plan
            .operators
            .iter()
            .filter_map(|operator| match operator {
                PhysicalOperator::ScanPattern { match_group, .. } => Some(*match_group),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(groups.len(), 3);
        assert!(groups.windows(2).all(|pair| pair[0] != pair[1]));
        Ok(())
    }

    #[test]
    fn stale_zero_cardinality_replans_once_before_output() -> Result<()> {
        let mut graph = GraphStore::default();
        let label = graph.catalog_mut().intern_label("RuntimeEstimate")?;
        let stale = StatisticsSnapshot::collect(&graph);
        for id in 1..=32_u64 {
            graph.apply(GraphMutation::InsertNode(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: id,
                labels: vec![label],
                properties: Vec::new(),
            }))?;
        }
        let source = "MATCH (n:RuntimeEstimate) RETURN id(n) AS runtime_id ORDER BY runtime_id";
        let mut optimized_context = test_context(&graph);
        optimized_context.optimizer_statistics = Some(&stale);
        let optimized = QueryEngine.execute(source, &mut optimized_context)?;
        assert_eq!(optimized.runtime_replans, 1);
        let mut source_context = test_context(&graph);
        let unoptimized = QueryEngine.execute_unoptimized(source, &mut source_context)?;
        assert_eq!(optimized.result, unoptimized.result);
        Ok(())
    }

    /// The two spellings of one equality lookup must agree on how many rows they expect.
    ///
    /// `estimate_pattern` sees only the pattern, so `WHERE p.key = 'x'` — whose predicate lives in a
    /// filter — used to estimate the whole label while the access path beside it was an
    /// `EqualityIndex` costed at a row or two. The checkpoint then disagreed with its own plan, and
    /// execution replanned every time: measured `estimated_rows=200000 actual_rows=1` on a
    /// 200 000-node graph, which cost 69 ms against 0.13 ms for the inline spelling of the same
    /// question.
    #[test]
    fn the_where_spelling_estimates_the_same_rows_as_the_inline_spelling() -> Result<()> {
        let mut graph = GraphStore::default();
        let label = graph.catalog_mut().intern_label("IndexedEntity")?;
        let property = graph.catalog_mut().intern_property("key")?;
        for id in 1..=512u64 {
            graph.apply(GraphMutation::InsertNode(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: id,
                labels: vec![label],
                properties: vec![(
                    property,
                    ScalarValue::String(Arc::from(format!("k{id}").as_str())),
                )],
            }))?;
        }
        let mut indexes = IndexCatalog::default();
        indexes.create(
            &graph,
            GraphIndexDefinition {
                name: "entity_key".to_owned(),
                kind: GraphIndexKind::Equality,
                label,
                properties: vec![property],
                unique: false,
            },
        )?;
        let statistics = StatisticsSnapshot::collect_project(&graph, None, Some(&indexes));
        let checkpoint_estimate = |source: &str| -> Result<u64> {
            let bound = bind(
                parse(source)?,
                graph.catalog(),
                super::super::BindCapabilities::default(),
            )?;
            let (physical, _) = optimize(
                plan(bound)?,
                OptimizerInput {
                    statistics: &statistics,
                    catalog: graph.catalog(),
                    indexes: Some(&indexes),
                    parameters: &BTreeMap::new(),
                    backend: BackendKind::Cpu,
                    scratch_budget_bytes: usize::MAX,
                    max_result_rows: 100_000,
                    runtime_feedback: None,
                    allow_runtime_checkpoint: true,
                },
            );
            Ok(physical
                .operators
                .iter()
                .find_map(|operator| match operator {
                    PhysicalOperator::CardinalityCheckpoint { estimated_rows, .. } => {
                        Some(*estimated_rows)
                    }
                    _ => None,
                })
                .unwrap_or(u64::MAX))
        };
        let inline = checkpoint_estimate("MATCH (n:IndexedEntity {key: 'k7'}) RETURN id(n) AS i")?;
        let filtered =
            checkpoint_estimate("MATCH (n:IndexedEntity) WHERE n.key = 'k7' RETURN id(n) AS i")?;
        assert_eq!(
            inline, filtered,
            "the WHERE spelling estimated {filtered} rows against the inline spelling's {inline}; \
             a checkpoint that disagrees with its own EqualityIndex access replans every execution"
        );
        assert!(
            filtered < 512,
            "an indexed equality must not be estimated at the whole label ({filtered} of 512)"
        );
        Ok(())
    }

    #[test]
    fn equality_index_access_matches_canonical_scan() -> Result<()> {
        let mut graph = GraphStore::default();
        let label = graph.catalog_mut().intern_label("IndexedEntity")?;
        let property = graph.catalog_mut().intern_property("key")?;
        for (id, value) in [(1, "alpha"), (2, "beta"), (3, "alpha")] {
            graph.apply(GraphMutation::InsertNode(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: id,
                labels: vec![label],
                properties: vec![(property, ScalarValue::String(Arc::from(value)))],
            }))?;
        }
        let mut indexes = IndexCatalog::default();
        indexes.create(
            &graph,
            GraphIndexDefinition {
                name: "entity_key".to_owned(),
                kind: GraphIndexKind::Equality,
                label,
                properties: vec![property],
                unique: false,
            },
        )?;
        let source = "MATCH (n:IndexedEntity) WHERE n.key = 'alpha' RETURN id(n) AS indexed_id ORDER BY indexed_id";
        let bound = bind(
            parse(source)?,
            graph.catalog(),
            super::super::BindCapabilities::default(),
        )?;
        let statistics = StatisticsSnapshot::collect_project(&graph, None, Some(&indexes));
        let (physical, _) = optimize(
            plan(bound)?,
            OptimizerInput {
                statistics: &statistics,
                catalog: graph.catalog(),
                indexes: Some(&indexes),
                parameters: &BTreeMap::new(),
                backend: BackendKind::Cpu,
                scratch_budget_bytes: usize::MAX,
                max_result_rows: 100_000,
                runtime_feedback: None,
                allow_runtime_checkpoint: false,
            },
        );
        assert!(physical.operators.iter().any(|operator| matches!(
            operator,
            PhysicalOperator::ScanPattern {
                access: ScanAccessPath::EqualityIndex { name, .. },
                ..
            } if name == "entity_key"
        )));
        let mut indexed_context = test_context(&graph);
        indexed_context.scalar_indexes = Some(&indexes);
        indexed_context.optimizer_statistics = Some(&statistics);
        let accelerated = QueryEngine.execute(source, &mut indexed_context)?;
        let mut scan_context = test_context(&graph);
        let scanned = QueryEngine.execute_unoptimized(source, &mut scan_context)?;
        assert_eq!(accelerated.result, scanned.result);
        Ok(())
    }

    #[test]
    fn infeasible_resident_scratch_is_rejected_before_execution() -> Result<()> {
        let mut graph = GraphStore::default();
        let label = graph.catalog_mut().intern_label("N")?;
        for id in 1..=32_u64 {
            graph.apply(GraphMutation::InsertNode(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: id,
                labels: vec![label],
                properties: Vec::new(),
            }))?;
        }
        let backend = CpuBackend::new(1_024, 0);
        let mut context = test_context(&graph);
        context.backend = Some(&backend);
        let error = QueryEngine
            .execute("MATCH (a:N), (b:N) RETURN a, b", &mut context)
            .err()
            .ok_or_else(|| Error::internal("infeasible Cartesian query was admitted"))?;
        assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
        Ok(())
    }

    #[test]
    fn vector_access_is_population_backend_and_scratch_gated() {
        let statistics = OptimizerIndexStatistics {
            name: "semantic".to_owned(),
            kind: crate::graph::GraphIndexKind::Vector,
            label: crate::types::LabelId(1),
            properties: vec![crate::types::PropertyId(2)],
            distinct_keys: 0,
            total_postings: 0,
            largest_posting: 0,
            vector_rows: 1_000_000,
            ann_candidate_budget: 1_024,
            resident_bytes: 1,
        };
        assert!(matches!(
            choose_vector_access(
                Some(&statistics),
                1_000_000,
                1_000_000,
                8,
                100,
                384,
                BackendKind::Metal,
                usize::MAX,
            ),
            VectorAccessPath::IvfPq {
                filtered_rows: 1_000_000,
                query_batch: 8,
                candidate_budget: 1_024,
                ..
            }
        ));
        assert!(matches!(
            choose_vector_access(
                Some(&statistics),
                4_096,
                1_000_000,
                1,
                100,
                384,
                BackendKind::Metal,
                usize::MAX,
            ),
            VectorAccessPath::Exact { .. }
        ));
        assert!(matches!(
            choose_vector_access(
                Some(&statistics),
                1_000_000,
                1_000_000,
                1,
                100,
                384,
                BackendKind::Cpu,
                usize::MAX,
            ),
            VectorAccessPath::Exact { .. }
        ));
        assert!(matches!(
            choose_vector_access(
                Some(&statistics),
                1_000_000,
                1_000_000,
                1,
                100,
                384,
                BackendKind::Metal,
                1,
            ),
            VectorAccessPath::Exact { .. }
        ));
        assert!(matches!(
            choose_vector_access(
                Some(&statistics),
                1_000_000,
                1_000_000,
                1,
                1_025,
                384,
                BackendKind::Metal,
                usize::MAX,
            ),
            VectorAccessPath::Exact { .. }
        ));
    }

    #[test]
    fn statistical_aggregates_execute_with_stable_numeric_semantics() -> Result<()> {
        let graph = GraphStore::default();
        let mut context = test_context(&graph);
        let output = QueryEngine.execute(
            "UNWIND [1, 2, 3, 4] AS x RETURN sum(x) AS total, variance(x) AS sample, variancep(x) AS population, stdev(x) AS deviation, percentilecont(x, 0.5) AS median, percentiledisc(x, 0.5) AS discrete",
            &mut context,
        )?;
        let batch = output
            .result
            .batches
            .first()
            .ok_or_else(|| Error::internal("aggregate query returned no batch"))?;
        let value = |name: &str| {
            batch
                .columns
                .iter()
                .find(|column| column.name == name)
                .and_then(|column| column.values.first())
                .cloned()
                .ok_or_else(|| Error::internal(format!("aggregate column {name} is absent")))
        };
        assert_eq!(
            value("total")?,
            ResultValue::Scalar(ScalarValue::Integer(10))
        );
        assert_eq!(
            value("sample")?,
            ResultValue::Scalar(ScalarValue::Float(OrderedFloat(5.0 / 3.0)))
        );
        assert_eq!(
            value("population")?,
            ResultValue::Scalar(ScalarValue::Float(OrderedFloat(1.25)))
        );
        assert_eq!(
            value("median")?,
            ResultValue::Scalar(ScalarValue::Float(OrderedFloat(2.5)))
        );
        assert_eq!(
            value("discrete")?,
            ResultValue::Scalar(ScalarValue::Integer(2))
        );
        let ResultValue::Scalar(ScalarValue::Float(deviation)) = value("deviation")? else {
            return Err(Error::internal("stdev did not return a float"));
        };
        assert!((deviation.into_inner() - (5.0_f64 / 3.0).sqrt()).abs() < 1.0e-12);
        Ok(())
    }

    #[test]
    fn required_scalar_and_vector_functions_execute_semantically() -> Result<()> {
        let graph = GraphStore::default();
        let mut context = test_context(&graph);
        let output = QueryEngine.execute(
            "RETURN abs(-3) AS absolute, round(degrees(pi())) AS degrees, range(1, 5, 2) AS sequence, reverse('abc') AS reversed, vector.dot([1, 2], [3, 4]) AS dot, abs(null) AS missing",
            &mut context,
        )?;
        let batch = output
            .result
            .batches
            .first()
            .ok_or_else(|| Error::internal("scalar function query returned no batch"))?;
        let value = |name: &str| {
            batch
                .columns
                .iter()
                .find(|column| column.name == name)
                .and_then(|column| column.values.first())
                .cloned()
                .ok_or_else(|| Error::internal(format!("function column {name} is absent")))
        };
        assert_eq!(
            value("absolute")?,
            ResultValue::Scalar(ScalarValue::Integer(3))
        );
        assert_eq!(
            value("degrees")?,
            ResultValue::Scalar(ScalarValue::Float(OrderedFloat(180.0)))
        );
        assert_eq!(
            value("sequence")?,
            ResultValue::List(
                [1, 3, 5]
                    .into_iter()
                    .map(|value| ResultValue::Scalar(ScalarValue::Integer(value)))
                    .collect()
            )
        );
        assert_eq!(
            value("reversed")?,
            ResultValue::Scalar(ScalarValue::String(Arc::from("cba")))
        );
        assert_eq!(
            value("dot")?,
            ResultValue::Scalar(ScalarValue::Float(OrderedFloat(11.0)))
        );
        assert_eq!(value("missing")?, ResultValue::Scalar(ScalarValue::Null));
        Ok(())
    }

    #[test]
    fn temporal_constructors_and_iso_duration_are_deterministic() -> Result<()> {
        let graph = GraphStore::default();
        let mut context = test_context(&graph);
        let output = QueryEngine.execute(
            "RETURN date('1970-01-02') AS day, localtime('12:34:56.25') AS clock, datetime('2026-07-16T12:00:00+02:00') AS instant, duration('P1Y2M3DT4H5M6.25S') AS span",
            &mut context,
        )?;
        let batch = output
            .result
            .batches
            .first()
            .ok_or_else(|| Error::internal("temporal function query returned no batch"))?;
        let value = |name: &str| {
            batch
                .columns
                .iter()
                .find(|column| column.name == name)
                .and_then(|column| column.values.first())
                .cloned()
                .ok_or_else(|| Error::internal(format!("temporal column {name} is absent")))
        };
        assert_eq!(value("day")?, ResultValue::Scalar(ScalarValue::Date(1)));
        assert_eq!(
            value("clock")?,
            ResultValue::Scalar(ScalarValue::LocalTime(45_296_250_000_000))
        );
        assert_eq!(
            value("instant")?,
            ResultValue::Scalar(ScalarValue::ZonedDateTime {
                seconds: 1_784_196_000,
                nanos: 0,
                timezone: Arc::from("+02:00"),
            })
        );
        assert_eq!(
            value("span")?,
            ResultValue::Scalar(ScalarValue::Duration {
                months: 14,
                days: 3,
                seconds: 14_706,
                nanos: 250_000_000,
            })
        );
        Ok(())
    }

    fn test_context(graph: &GraphStore) -> ExecutionContext<'_> {
        ExecutionContext {
            project_id: ProjectId(uuid::Uuid::nil()),
            graph,
            binding_catalog: graph.catalog(),
            prior_graph_mutations: &[],
            temporal: None,
            prior_temporal_mutations: &[],
            vector_indexes: BTreeMap::new(),
            scalar_indexes: None,
            text_embedding: None,
            parameters: BTreeMap::new(),
            bookmark: Bookmark {
                term: 1,
                index: graph.revision(),
            },
            mutation_revision: graph.revision().saturating_add(1),
            resolved_time_nanos: 0,
            next_node_id: 100,
            next_edge_id: 100,
            predicate_versions: BTreeMap::new(),
            capabilities: super::super::BindCapabilities::default(),
            max_result_rows: 100_000,
            max_batch_rows: 4_096,
            optimizer_statistics: None,
            backend: None,
            cancellation: CancellationToken::new(),
            deadline: Some(Instant::now() + std::time::Duration::from_secs(5)),
            resolved_query_at_time_nanos: None,
        }
    }
}
