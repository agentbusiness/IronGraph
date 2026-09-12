//! Eligibility compiler for complete native typed-row plans.
//!
//! One request owns its complete row source (resident graph scan or typed literal `UNWIND`), all
//! row-local SSA expressions, every projection/scope boundary, one stable sort/pagination
//! boundary, and final entity/scalar publication. Unsupported operators return `None`; no prefix
//! is handed back to the generic executor.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use rand::Rng;

use crate::{
    Bookmark, DocumentItem, Error, ErrorCode, ProjectId, Result, ScalarValue,
    execution::{
        CompareOp, RESIDENT_NULLABLE_RELATION_MAX_LITERAL_STRING_BYTES,
        RESIDENT_NULLABLE_RELATION_MAX_PREDICATE_DEPTH,
        RESIDENT_NULLABLE_RELATION_MAX_PREDICATE_NODES, RESIDENT_ROW_PROGRAM_MAX_REGISTERS,
        ResidentDeleteCommand, ResidentDeleteContinuation, ResidentDeleteFingerprint,
        ResidentDeleteIntegerSource, ResidentDeleteOutput, ResidentDeleteOutputSource,
        ResidentDeletePostStage, ResidentDeleteRequest, ResidentDeleteSelectionFilter,
        ResidentDeleteTargetSelector, ResidentDirection, ResidentEntityBinding,
        ResidentExecutionId, ResidentExecutionObligation, ResidentI64Predicate,
        ResidentNodeBinding, ResidentNodePipelineRequest, ResidentNullableNodeDomain,
        ResidentNullableRelationBindingKind, ResidentNullableRelationCapacities,
        ResidentNullableRelationEntityLabelDomain, ResidentNullableRelationFilterPlacement,
        ResidentNullableRelationFilterStage, ResidentNullableRelationGeneration,
        ResidentNullableRelationMatchMode, ResidentNullableRelationOptionalGroup,
        ResidentNullableRelationOutputBinding, ResidentNullableRelationOutputSource,
        ResidentNullableRelationPredicate, ResidentNullableRelationPredicateProgram,
        ResidentNullableRelationPredicateValue, ResidentNullableRelationProgram,
        ResidentNullableRelationProjectionBinding, ResidentNullableRelationRequest,
        ResidentNullableRelationSlot, ResidentNullableRelationStage,
        ResidentNullableRelationTarget, ResidentNullableRelationshipDomain,
        ResidentNullableRelationshipEndpoint, ResidentObligationKind, ResidentObligationScope,
        ResidentPropertyFilterInstruction, ResidentPropertyFilterProgram, ResidentQuantifierBinary,
        ResidentQuantifierEntityKind, ResidentQuantifierEntityProperty,
        ResidentQuantifierEntityPropertyShape, ResidentQuantifierExpression,
        ResidentQuantifierFunction, ResidentQuantifierGeneration, ResidentQuantifierKind,
        ResidentQuantifierOutput, ResidentQuantifierProgram, ResidentQuantifierProgramRequest,
        ResidentQuantifierProjection, ResidentQuantifierSlot, ResidentQuantifierSource,
        ResidentQuantifierStage, ResidentQuantifierUnary, ResidentQuantifierValue,
        ResidentRowColumn, ResidentRowInstruction, ResidentRowOperation, ResidentRowProgram,
        ResidentRowProgramManifest, ResidentRowProgramRequest, ResidentRowSortKey,
        ResidentRowValueType, ResidentStringPredicateOperation, ResidentTemporalAccessor,
    },
    graph::{GraphStore, PropertyColumns, TypedColumn},
    types::{EntityKind, LabelId, RelationshipTypeId},
};

use super::{
    BinaryOperator, DependencyKind, Direction, Expression, ListPredicateKind, PathMode,
    PathSelector, Pattern, PhysicalOperator, PhysicalPlan, Projection, ProjectionItem, ResultValue,
    ScanAccessPath, UnaryOperator,
    expression::contains_aggregate,
    planner::MatchGroupId,
    resident::{compile_range_expression, resident_direct_literal_scalar_map},
};

const QUANTIFIER_ENTITY_LIST_MATERIALIZE_OBLIGATION: u64 = 0x5155_454e_544c_0001;
const QUANTIFIER_RANGE_SOURCE_OBLIGATION: u64 = 0x5155_5241_4e47_0001;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompiledResidentRowOutputSource {
    Node(ResidentNodeBinding),
    ProjectedRegister {
        column: usize,
        value_type: ResidentRowValueType,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompiledResidentRowOutput {
    pub name: String,
    pub source: CompiledResidentRowOutputSource,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompiledResidentRowPlan {
    pub request: ResidentRowProgramRequest,
    pub outputs: Vec<CompiledResidentRowOutput>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompiledResidentQuantifierPlan {
    pub request: ResidentQuantifierProgramRequest,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[allow(dead_code)] // ABI/compiler foundation; execution wiring is intentionally a later lane.
pub struct CompiledResidentNullableRelationOutput {
    pub name: String,
    pub source: ResidentNullableRelationOutputSource,
}

/// Compiler-only foundation for the staged nullable relation ABI. The executor intentionally does
/// not call this route until CPU and accelerator implementations own the complete command.
#[derive(Clone, Debug, PartialEq, Eq)]
#[allow(dead_code)] // ABI/compiler foundation; execution wiring is intentionally a later lane.
pub struct CompiledResidentNullableRelationPlan {
    pub request: ResidentNullableRelationRequest,
    pub outputs: Vec<CompiledResidentNullableRelationOutput>,
}

/// Client-side rendering applied only after one complete native command has returned a validated
/// canonical entity column. This is compiler metadata, not a backend ABI: the selected device
/// continues to publish the same bounded entity identity it already owns.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompiledResidentOutputRender {
    NonNullNodePath,
    NonNullNodeLabels,
    NullableNodeLabels,
    NonNullNodeHasProperty { key: String },
    NonNullNodeProperties,
    NullableNodeProperties,
    NonNullRelationshipProperties,
    NullableRelationshipProperties,
}

#[derive(Clone, Debug)]
pub struct CompiledResidentRenderedOutput {
    pub index: usize,
    pub name: String,
    pub render: CompiledResidentOutputRender,
}

#[derive(Clone, Debug)]
pub struct CompiledResidentRenderedPlan {
    pub plan: PhysicalPlan,
    pub outputs: Vec<CompiledResidentRenderedOutput>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum DirectEntityKind {
    Node,
    Relationship,
    Other,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DirectEntityBinding {
    kind: DirectEntityKind,
    nullable: bool,
    known_null: bool,
}

fn record_direct_entity(
    bindings: &mut BTreeMap<String, DirectEntityBinding>,
    variable: &Option<String>,
    binding: DirectEntityBinding,
) -> bool {
    let Some(variable) = variable else {
        return true;
    };
    match bindings.get(variable) {
        Some(existing) => *existing == binding,
        None => {
            bindings.insert(variable.clone(), binding);
            true
        }
    }
}

fn record_direct_pattern_entities(
    bindings: &mut BTreeMap<String, DirectEntityBinding>,
    pattern: &super::Pattern,
    optional: bool,
    catalog: &crate::graph::NameCatalog,
) -> bool {
    let catalog_proves_null = optional
        && pattern.steps.is_empty()
        && pattern
            .start
            .labels
            .iter()
            .any(|label| catalog.label(label).is_none());
    let existing_start = pattern
        .start
        .variable
        .as_ref()
        .and_then(|variable| bindings.get(variable))
        .copied();
    if existing_start.is_some_and(|binding| binding.kind != DirectEntityKind::Node) {
        return false;
    }
    let known_null =
        catalog_proves_null || existing_start.is_some_and(|binding| binding.known_null);
    let start_binding = existing_start.unwrap_or(DirectEntityBinding {
        kind: DirectEntityKind::Node,
        nullable: optional,
        known_null,
    });
    if !record_direct_entity(
        bindings,
        &pattern.variable,
        DirectEntityBinding {
            kind: DirectEntityKind::Other,
            nullable: optional,
            known_null,
        },
    ) || !record_direct_entity(bindings, &pattern.start.variable, start_binding)
    {
        return false;
    }
    pattern.steps.iter().all(|step| {
        record_direct_entity(
            bindings,
            &step.relationship.variable,
            DirectEntityBinding {
                kind: DirectEntityKind::Relationship,
                nullable: optional,
                known_null,
            },
        ) && record_direct_entity(
            bindings,
            &step.node.variable,
            DirectEntityBinding {
                kind: DirectEntityKind::Node,
                nullable: optional,
                known_null,
            },
        )
    })
}

fn static_label_mutation_target(
    operator: &PhysicalOperator,
    bindings: &BTreeMap<String, DirectEntityBinding>,
) -> Option<String> {
    let mut target = None::<String>;
    let mut accept = |variable: &str, labels: &[super::LabelName]| {
        let binding = bindings.get(variable)?;
        if binding.kind != DirectEntityKind::Node
            || labels.is_empty()
            || labels
                .iter()
                .any(|label| !matches!(label, super::LabelName::Static(_)))
            || target.as_ref().is_some_and(|existing| existing != variable)
        {
            return None;
        }
        target = Some(variable.to_owned());
        Some(())
    };
    match operator {
        PhysicalOperator::Set(items) => {
            for item in items {
                let super::SetItem::Labels { variable, labels } = item else {
                    return None;
                };
                accept(variable, labels)?;
            }
        }
        PhysicalOperator::Remove(items) => {
            for item in items {
                let super::RemoveItem::Labels { variable, labels } = item else {
                    return None;
                };
                accept(variable, labels)?;
            }
        }
        _ => return None,
    }
    target
}

fn known_null_noop_mutation_target(
    operator: &PhysicalOperator,
    bindings: &BTreeMap<String, DirectEntityBinding>,
) -> Option<String> {
    // Erasing SET is sound only when evaluating its RHS cannot fail, observe state, or introduce
    // a temporal write. REMOVE has no RHS. The surrounding exact-plan matcher separately proves
    // that this one nullable node binding is catalog-known null and that its row is still returned.
    let target = match operator {
        PhysicalOperator::Set(items) => match items.as_slice() {
            [
                super::SetItem::Property {
                    target,
                    value:
                        Expression::Literal(
                            ScalarValue::Null
                            | ScalarValue::Boolean(_)
                            | ScalarValue::Integer(_)
                            | ScalarValue::Float(_)
                            | ScalarValue::String(_)
                            | ScalarValue::Bytes(_)
                            | ScalarValue::Date(_)
                            | ScalarValue::LocalTime(_)
                            | ScalarValue::ZonedTime { .. }
                            | ScalarValue::LocalDateTime { .. }
                            | ScalarValue::ZonedDateTime { .. }
                            | ScalarValue::Duration { .. },
                        ),
                    event_time: None,
                },
            ] if !target.property.is_empty() => target.variable.clone(),
            [
                super::SetItem::MergeMap { variable, value }
                | super::SetItem::ReplaceMap { variable, value },
            ] if resident_direct_literal_scalar_map(value).is_some() => variable.clone(),
            _ => return static_label_mutation_target(operator, bindings),
        },
        PhysicalOperator::Remove(items) => match items.as_slice() {
            [super::RemoveItem::Property(target)] if !target.property.is_empty() => {
                target.variable.clone()
            }
            _ => return static_label_mutation_target(operator, bindings),
        },
        _ => return None,
    };
    let binding = bindings.get(&target)?;
    (binding.kind == DirectEntityKind::Node && binding.nullable && binding.known_null)
        .then_some(target)
}

fn compile_known_null_entity_mutation(
    plan: &PhysicalPlan,
    catalog: &crate::graph::NameCatalog,
) -> Option<CompiledResidentRenderedPlan> {
    if plan.read_only {
        return None;
    }
    let operators = plan
        .operators
        .iter()
        .enumerate()
        .filter(|(_, operator)| !matches!(operator, PhysicalOperator::CardinalityCheckpoint { .. }))
        .collect::<Vec<_>>();
    let [
        (
            _,
            PhysicalOperator::ScanPattern {
                optional: true,
                pattern,
                ..
            },
        ),
        (mutation_index, mutation @ (PhysicalOperator::Set(_) | PhysicalOperator::Remove(_))),
        (
            _,
            PhysicalOperator::Project {
                keep_scope: false,
                projection,
            },
        ),
    ] = operators.as_slice()
    else {
        return None;
    };
    if pattern.variable.is_some()
        || pattern.selector != PathSelector::All
        || pattern.mode != PathMode::DifferentRelationships
        || pattern.start.property_predicate_present
        || !pattern.start.properties.is_empty()
        || !pattern.steps.is_empty()
        || projection.distinct
    {
        return None;
    }
    let variable = pattern.start.variable.as_ref()?;
    if !pattern
        .start
        .labels
        .iter()
        .any(|label| catalog.label(label).is_none())
    {
        return None;
    }
    let bindings = BTreeMap::from([(
        variable.clone(),
        DirectEntityBinding {
            kind: DirectEntityKind::Node,
            nullable: true,
            known_null: true,
        },
    )]);
    if known_null_noop_mutation_target(mutation, &bindings).as_ref() != Some(variable) {
        return None;
    }
    let [item] = projection.items.as_slice() else {
        return None;
    };
    if !matches!(&item.expression, Expression::Variable(output) if output == variable) {
        return None;
    }

    let mut rendered = plan.clone();
    rendered.read_only = true;
    rendered.operators.remove(*mutation_index);
    Some(CompiledResidentRenderedPlan {
        plan: rendered,
        outputs: Vec::new(),
    })
}

/// Erases the exact `SET node += {}` identity. The empty literal has no evaluation, catalog, or
/// statistics effect, so the unchanged direct-node relation is the complete native command.
fn compile_empty_literal_merge_noop(
    plan: &PhysicalPlan,
    catalog: &crate::graph::NameCatalog,
) -> Option<CompiledResidentRenderedPlan> {
    if plan.read_only {
        return None;
    }
    let operators = plan
        .operators
        .iter()
        .enumerate()
        .filter(|(_, operator)| !matches!(operator, PhysicalOperator::CardinalityCheckpoint { .. }))
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
        (mutation_index, PhysicalOperator::Set(items)),
        (
            _,
            PhysicalOperator::Project {
                keep_scope: false,
                projection,
            },
        ),
    ] = operators.as_slice()
    else {
        return None;
    };
    if pattern.variable.is_some()
        || pattern.selector != PathSelector::All
        || pattern.mode != PathMode::DifferentRelationships
        || !pattern.steps.is_empty()
        || projection.distinct
    {
        return None;
    }
    let [super::SetItem::MergeMap { variable, value }] = items.as_slice() else {
        return None;
    };
    if !resident_direct_literal_scalar_map(value)?.is_empty()
        || pattern.start.variable.as_deref() != Some(variable)
    {
        return None;
    }
    let mut bindings = BTreeMap::new();
    if !record_direct_pattern_entities(&mut bindings, pattern, false, catalog)
        || bindings.get(variable).is_none_or(|binding| {
            binding.kind != DirectEntityKind::Node || binding.nullable || binding.known_null
        })
    {
        return None;
    }
    let [item] = projection.items.as_slice() else {
        return None;
    };
    if !matches!(&item.expression, Expression::Variable(output) if output == variable) {
        return None;
    }

    let mut rendered = plan.clone();
    rendered.read_only = true;
    rendered.operators.remove(*mutation_index);
    Some(CompiledResidentRenderedPlan {
        plan: rendered,
        outputs: Vec::new(),
    })
}

fn compile_list_zero_node_labels_render(
    plan: &PhysicalPlan,
    catalog: &crate::graph::NameCatalog,
) -> Option<CompiledResidentRenderedPlan> {
    if !plan.read_only {
        return None;
    }
    let operators = plan
        .operators
        .iter()
        .enumerate()
        .filter(|(_, operator)| !matches!(operator, PhysicalOperator::CardinalityCheckpoint { .. }))
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
            list_index,
            PhysicalOperator::Project {
                keep_scope: false,
                projection: list_projection,
            },
        ),
        (
            final_index,
            PhysicalOperator::Project {
                keep_scope: false,
                projection: final_projection,
            },
        ),
    ] = operators.as_slice()
    else {
        return None;
    };
    if pattern.variable.is_some()
        || pattern.selector != PathSelector::All
        || pattern.mode != PathMode::DifferentRelationships
        || pattern.start.property_predicate_present
        || !pattern.start.properties.is_empty()
        || !pattern.steps.is_empty()
        || list_projection.distinct
        || final_projection.distinct
    {
        return None;
    }
    let node = pattern.start.variable.as_ref()?;
    let [list_item] = list_projection.items.as_slice() else {
        return None;
    };
    let list_name = list_item.alias.as_ref()?;
    let Expression::List(elements) = &list_item.expression else {
        return None;
    };
    if !matches!(
        elements.as_slice(),
        [Expression::Variable(variable), Expression::Literal(ScalarValue::Integer(1))]
            if variable == node
    ) {
        return None;
    }
    let [item] = final_projection.items.as_slice() else {
        return None;
    };
    let Expression::Function {
        name,
        distinct: false,
        arguments,
    } = &item.expression
    else {
        return None;
    };
    if !matches!(name.as_slice(), [name] if name.eq_ignore_ascii_case("labels"))
        || !matches!(
            arguments.as_slice(),
            [Expression::Index { expression, index }]
                if matches!(expression.as_ref(), Expression::Variable(variable) if variable == list_name)
                    && matches!(index.as_ref(), Expression::Literal(ScalarValue::Integer(0)))
        )
    {
        return None;
    }
    let mut bindings = BTreeMap::new();
    if !record_direct_pattern_entities(&mut bindings, pattern, false, catalog)
        || bindings
            .get(node)
            .is_none_or(|binding| binding.kind != DirectEntityKind::Node || binding.nullable)
    {
        return None;
    }

    let output_name = item.column_name(0);
    let mut rendered = plan.clone();
    let PhysicalOperator::Project { projection, .. } = &mut rendered.operators[*list_index] else {
        unreachable!("the rendered list projection changed shape")
    };
    projection.items[0] = ProjectionItem {
        expression: Expression::Variable(node.clone()),
        alias: Some(list_name.clone()),
        source_text: None,
    };
    let PhysicalOperator::Project { projection, .. } = &mut rendered.operators[*final_index] else {
        unreachable!("the rendered labels projection changed shape")
    };
    projection.items[0] = ProjectionItem {
        expression: Expression::Variable(list_name.clone()),
        alias: Some(output_name.clone()),
        source_text: None,
    };
    Some(CompiledResidentRenderedPlan {
        plan: rendered,
        outputs: vec![CompiledResidentRenderedOutput {
            index: 0,
            name: output_name,
            render: CompiledResidentOutputRender::NonNullNodeLabels,
        }],
    })
}

/// Rewrites only the terminal `labels(node)` projection of the exact standalone node-MERGE
/// tranche. The mutation compiler and request validator independently prove the same one-command
/// shape; this metadata-only pass merely lets that command publish its existing non-null entity
/// output for canonical label rendering.
fn compile_exact_single_node_merge_labels_render(
    plan: &PhysicalPlan,
) -> Option<CompiledResidentRenderedPlan> {
    if plan.read_only {
        return None;
    }
    let operators = plan
        .operators
        .iter()
        .enumerate()
        .filter(|(_, operator)| !matches!(operator, PhysicalOperator::CardinalityCheckpoint { .. }))
        .collect::<Vec<_>>();
    let [
        (
            _,
            PhysicalOperator::MergePattern {
                pattern,
                on_create,
                on_match,
            },
        ),
        (
            projection_index,
            PhysicalOperator::Project {
                keep_scope: false,
                projection,
            },
        ),
    ] = operators.as_slice()
    else {
        return None;
    };
    if !on_create.is_empty()
        || !on_match.is_empty()
        || pattern.variable.is_some()
        || pattern.selector != PathSelector::All
        || pattern.mode != PathMode::DifferentRelationships
        || !pattern.steps.is_empty()
        || projection.distinct
    {
        return None;
    }
    let variable = pattern.start.variable.as_ref()?;
    if variable.is_empty() || pattern.start.properties.iter().any(|(name, value)| {
        name.is_empty()
            || !matches!(value, Expression::Literal(scalar) if !matches!(scalar, ScalarValue::Null))
    }) {
        return None;
    }
    let [item] = projection.items.as_slice() else {
        return None;
    };
    let Expression::Function {
        name,
        distinct: false,
        arguments,
    } = &item.expression
    else {
        return None;
    };
    if !matches!(name.as_slice(), [name] if name.eq_ignore_ascii_case("labels"))
        || !matches!(arguments.as_slice(), [Expression::Variable(output)] if output == variable)
    {
        return None;
    }

    let output_name = item.column_name(0);
    let mut rendered = plan.clone();
    let PhysicalOperator::Project { projection, .. } = &mut rendered.operators[*projection_index]
    else {
        unreachable!("the exact node-MERGE labels projection changed shape")
    };
    projection.items[0] = ProjectionItem {
        expression: Expression::Variable(variable.clone()),
        alias: Some(output_name.clone()),
        source_text: None,
    };
    Some(CompiledResidentRenderedPlan {
        plan: rendered,
        outputs: vec![CompiledResidentRenderedOutput {
            index: 0,
            name: output_name,
            render: CompiledResidentOutputRender::NonNullNodeLabels,
        }],
    })
}

/// Rewrites only Merge1 [13]'s named zero-hop path into the already-owned standalone node-MERGE
/// command. The resident mutation ABI continues to publish the one canonical created/matched node;
/// the executor wraps that node in a zero-relationship path at the client result boundary.
fn compile_exact_single_node_merge_path_render(
    plan: &PhysicalPlan,
) -> Option<CompiledResidentRenderedPlan> {
    if plan.read_only {
        return None;
    }
    let operators = plan
        .operators
        .iter()
        .enumerate()
        .filter(|(_, operator)| !matches!(operator, PhysicalOperator::CardinalityCheckpoint { .. }))
        .collect::<Vec<_>>();
    let [
        (
            merge_index,
            PhysicalOperator::MergePattern {
                pattern,
                on_create,
                on_match,
            },
        ),
        (
            projection_index,
            PhysicalOperator::Project {
                keep_scope: false,
                projection,
            },
        ),
    ] = operators.as_slice()
    else {
        return None;
    };
    if !on_create.is_empty()
        || !on_match.is_empty()
        || pattern.variable.as_deref() != Some("p")
        || pattern.selector != PathSelector::All
        || pattern.mode != PathMode::DifferentRelationships
        || pattern.start.variable.as_deref() != Some("a")
        || !pattern.start.labels.is_empty()
        || !pattern.start.property_predicate_present
        || !matches!(
            pattern.start.properties.as_slice(),
            [(name, Expression::Literal(ScalarValue::Integer(1)))] if name == "num"
        )
        || !pattern.steps.is_empty()
        || projection.distinct
    {
        return None;
    }
    let [item] = projection.items.as_slice() else {
        return None;
    };
    if item.alias.is_some()
        || item.source_text.as_deref() != Some("p")
        || !matches!(&item.expression, Expression::Variable(variable) if variable == "p")
    {
        return None;
    }

    let mut rendered = plan.clone();
    let PhysicalOperator::MergePattern { pattern, .. } = &mut rendered.operators[*merge_index]
    else {
        unreachable!("the exact node-only path MERGE changed shape")
    };
    pattern.variable = None;
    let PhysicalOperator::Project { projection, .. } = &mut rendered.operators[*projection_index]
    else {
        unreachable!("the exact node-only path projection changed shape")
    };
    projection.items[0] = ProjectionItem {
        expression: Expression::Variable("a".to_owned()),
        alias: Some("p".to_owned()),
        source_text: None,
    };
    Some(CompiledResidentRenderedPlan {
        plan: rendered,
        outputs: vec![CompiledResidentRenderedOutput {
            index: 0,
            name: "p".to_owned(),
            render: CompiledResidentOutputRender::NonNullNodePath,
        }],
    })
}

fn direct_entity_argument(
    arguments: &[Expression],
    bindings: &BTreeMap<String, DirectEntityBinding>,
    required_kind: Option<DirectEntityKind>,
) -> Option<(String, DirectEntityBinding)> {
    match arguments {
        [Expression::Variable(variable)] => {
            let binding = *bindings.get(variable)?;
            if required_kind.is_some_and(|kind| binding.kind != kind) {
                return None;
            }
            Some((variable.clone(), binding))
        }
        [Expression::Literal(ScalarValue::Null)] => bindings
            .iter()
            .find(|(_, binding)| {
                binding.known_null
                    && binding.kind != DirectEntityKind::Other
                    && required_kind.is_none_or(|kind| binding.kind == kind)
            })
            .map(|(variable, binding)| (variable.clone(), *binding)),
        _ => None,
    }
}

fn direct_metadata_render(
    expression: &Expression,
    bindings: &BTreeMap<String, DirectEntityBinding>,
    read_only: bool,
) -> Option<(String, CompiledResidentOutputRender)> {
    if let Expression::Function {
        name,
        distinct: false,
        arguments,
    } = expression
    {
        if matches!(name.as_slice(), [name] if name.eq_ignore_ascii_case("labels")) {
            let (variable, binding) =
                direct_entity_argument(arguments, bindings, Some(DirectEntityKind::Node))?;
            return Some((
                variable,
                if binding.nullable {
                    CompiledResidentOutputRender::NullableNodeLabels
                } else {
                    CompiledResidentOutputRender::NonNullNodeLabels
                },
            ));
        }
        if read_only && matches!(name.as_slice(), [name] if name.eq_ignore_ascii_case("properties"))
        {
            let (variable, binding) = direct_entity_argument(arguments, bindings, None)?;
            let render = match (binding.kind, binding.nullable) {
                (DirectEntityKind::Node, false) => {
                    CompiledResidentOutputRender::NonNullNodeProperties
                }
                (DirectEntityKind::Node, true) => {
                    CompiledResidentOutputRender::NullableNodeProperties
                }
                (DirectEntityKind::Relationship, false) => {
                    CompiledResidentOutputRender::NonNullRelationshipProperties
                }
                (DirectEntityKind::Relationship, true) => {
                    CompiledResidentOutputRender::NullableRelationshipProperties
                }
                (DirectEntityKind::Other, _) => return None,
            };
            return Some((variable, render));
        }
    }

    if !read_only {
        return None;
    }
    let Expression::Binary {
        left,
        operation: BinaryOperator::In,
        right,
    } = expression
    else {
        return None;
    };
    let Expression::Literal(ScalarValue::String(key)) = left.as_ref() else {
        return None;
    };
    let Expression::Function {
        name,
        distinct: false,
        arguments,
    } = right.as_ref()
    else {
        return None;
    };
    if !matches!(name.as_slice(), [name] if name.eq_ignore_ascii_case("keys")) {
        return None;
    }
    let (variable, binding) =
        direct_entity_argument(arguments, bindings, Some(DirectEntityKind::Node))?;
    if binding.nullable {
        return None;
    }
    Some((
        variable,
        CompiledResidentOutputRender::NonNullNodeHasProperty {
            key: key.to_string(),
        },
    ))
}

/// Recognizes only complete metadata projections that can reuse an existing native entity result,
/// plus exact statically-null optional entity-mutation no-ops. Unsupported expressions remain
/// with their complete-plan owner instead of acquiring a host expression tail after a resident
/// prefix.
pub fn compile_direct_node_labels_render(
    plan: &PhysicalPlan,
    catalog: &crate::graph::NameCatalog,
) -> Option<CompiledResidentRenderedPlan> {
    if plan.at_time.is_some() || !plan.unions.is_empty() {
        return None;
    }
    if let Some(compiled) = compile_known_null_entity_mutation(plan, catalog) {
        return Some(compiled);
    }
    if let Some(compiled) = compile_empty_literal_merge_noop(plan, catalog) {
        return Some(compiled);
    }
    if let Some(compiled) = compile_list_zero_node_labels_render(plan, catalog) {
        return Some(compiled);
    }
    if let Some(compiled) = compile_exact_single_node_merge_path_render(plan) {
        return Some(compiled);
    }
    if let Some(compiled) = compile_exact_single_node_merge_labels_render(plan) {
        return Some(compiled);
    }
    let final_index = plan.operators.iter().rposition(|operator| {
        !matches!(operator, PhysicalOperator::CardinalityCheckpoint { .. })
    })?;
    let PhysicalOperator::Project {
        keep_scope: false,
        projection,
    } = &plan.operators[final_index]
    else {
        return None;
    };
    if projection.distinct || projection.items.is_empty() {
        return None;
    }
    if plan.operators[final_index + 1..]
        .iter()
        .any(|operator| !matches!(operator, PhysicalOperator::CardinalityCheckpoint { .. }))
    {
        return None;
    }

    let mut bindings = BTreeMap::new();
    for operator in &plan.operators[..final_index] {
        let supported = match operator {
            PhysicalOperator::CardinalityCheckpoint { .. } => true,
            // A filter neither introduces nor changes entity bindings. The returned rendered
            // plan retains it, so only a complete native compiler that owns its predicate may
            // execute the metadata projection.
            PhysicalOperator::Filter(_) => true,
            PhysicalOperator::ScanPattern {
                optional, pattern, ..
            } => record_direct_pattern_entities(&mut bindings, pattern, *optional, catalog),
            PhysicalOperator::CreatePattern(pattern) => {
                record_direct_pattern_entities(&mut bindings, pattern, false, catalog)
            }
            operator @ (PhysicalOperator::Set(_) | PhysicalOperator::Remove(_)) => {
                static_label_mutation_target(operator, &bindings).is_some()
            }
            _ => false,
        };
        if !supported {
            return None;
        }
    }

    let mut rendered = plan.clone();
    let PhysicalOperator::Project {
        projection: rendered_projection,
        ..
    } = &mut rendered.operators[final_index]
    else {
        unreachable!("the rendered terminal projection changed shape")
    };
    let mut outputs = Vec::with_capacity(projection.items.len());
    for (index, item) in projection.items.iter().enumerate() {
        let (variable, render) =
            direct_metadata_render(&item.expression, &bindings, plan.read_only)?;
        let output_name = item.column_name(index);
        rendered_projection.items[index] = ProjectionItem {
            expression: Expression::Variable(variable),
            alias: Some(output_name.clone()),
            source_text: None,
        };
        outputs.push(CompiledResidentRenderedOutput {
            index,
            name: output_name,
            render,
        });
    }
    Some(CompiledResidentRenderedPlan {
        plan: rendered,
        outputs,
    })
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompiledResidentDeletePlan {
    pub request: ResidentDeleteRequest,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RowSymbol {
    Node(ResidentNodeBinding),
    Register(u16),
    DurationProperty {
        binding: ResidentEntityBinding,
        property: crate::types::PropertyId,
    },
}

/// Collapse one exact two-branch entity UNION into a single all-node scan with a native label OR.
/// UNION ALL is admitted only when the immutable generation proves the label domains disjoint.
fn canonicalize_exact_two_label_union(
    plan: &PhysicalPlan,
    catalog: &crate::graph::NameCatalog,
    graph: &GraphStore,
) -> Option<PhysicalPlan> {
    if !plan.read_only || plan.at_time.is_some() {
        return None;
    }
    let [(union_all, union_branch)] = plan.unions.as_slice() else {
        return None;
    };
    let branch_shape = |operators: &[PhysicalOperator]| {
        let semantic = operators
            .iter()
            .filter(|operator| !matches!(operator, PhysicalOperator::CardinalityCheckpoint { .. }))
            .collect::<Vec<_>>();
        let [
            PhysicalOperator::ScanPattern {
                optional: false,
                pattern,
                ..
            },
            PhysicalOperator::Project {
                keep_scope: false,
                projection,
            },
        ] = semantic.as_slice()
        else {
            return None;
        };
        let (Some(variable), [label], [item]) = (
            pattern.start.variable.as_deref(),
            pattern.start.labels.as_slice(),
            projection.items.as_slice(),
        ) else {
            return None;
        };
        if pattern.variable.is_some()
            || pattern.selector != PathSelector::All
            || pattern.mode != PathMode::DifferentRelationships
            || pattern.start.property_predicate_present
            || !pattern.start.properties.is_empty()
            || !pattern.steps.is_empty()
            || projection.distinct
            || !matches!(&item.expression, Expression::Variable(source) if source == variable)
        {
            return None;
        }
        Some((variable.to_owned(), label.to_owned(), item.column_name(0)))
    };
    let (base_variable, base_label, base_output) = branch_shape(&plan.operators)?;
    let (_, union_label, union_output) = branch_shape(union_branch)?;
    if base_output != union_output {
        return None;
    }

    if *union_all {
        let (Some(base_label_id), Some(union_label_id)) =
            (catalog.label(&base_label), catalog.label(&union_label))
        else {
            // An unresolved label denotes an empty branch, so the branches cannot overlap.
            return build_two_label_union_scan(plan, &base_variable, &base_label, &union_label);
        };
        if graph.scan_nodes(None, plan.read_layers).any(|node| {
            node.labels().contains(&base_label_id) && node.labels().contains(&union_label_id)
        }) {
            return None;
        }
    }
    build_two_label_union_scan(plan, &base_variable, &base_label, &union_label)
}

fn build_two_label_union_scan(
    plan: &PhysicalPlan,
    variable: &str,
    first_label: &str,
    second_label: &str,
) -> Option<PhysicalPlan> {
    let mut rewritten = plan.clone();
    rewritten.unions.clear();
    let scan_index = rewritten
        .operators
        .iter()
        .position(|operator| matches!(operator, PhysicalOperator::ScanPattern { .. }))?;
    let PhysicalOperator::ScanPattern {
        optional: false,
        pattern,
        access,
        ..
    } = rewritten.operators.get_mut(scan_index)?
    else {
        return None;
    };
    pattern.start.labels.clear();
    *access = ScanAccessPath::AllNodes;
    let label_filter = Expression::Binary {
        left: Box::new(Expression::entity_label_predicate(
            Expression::Variable(variable.to_owned()),
            vec![first_label.to_owned()],
        )),
        operation: BinaryOperator::Or,
        right: Box::new(Expression::entity_label_predicate(
            Expression::Variable(variable.to_owned()),
            vec![second_label.to_owned()],
        )),
    };
    rewritten.operators.insert(
        scan_index.saturating_add(1),
        PhysicalOperator::Filter(label_filter),
    );
    Some(rewritten)
}

/// Compile one complete native typed-row plan. Returning `None` means that the full plan is
/// outside this route; callers must not execute a resident prefix.
pub fn compile(
    plan: &PhysicalPlan,
    project: ProjectId,
    bookmark: Bookmark,
    catalog: &crate::graph::NameCatalog,
    graph: &GraphStore,
    parameters: &BTreeMap<String, ResultValue>,
    max_output_rows: usize,
) -> Result<Option<CompiledResidentRowPlan>> {
    if !plan.read_only || plan.at_time.is_some() || !plan.unions.is_empty() {
        return Ok(None);
    }
    if let Some(compiled) = compile_pure_zero_limit_window(plan, project, bookmark, graph)? {
        return Ok(Some(compiled));
    }

    let source_requires_filter = matches!(
        plan.operators.first(),
        Some(PhysicalOperator::ScanPattern {
            access: ScanAccessPath::ResidentProperty { .. },
            ..
        })
    );
    let mut builder = RowProgramBuilder::new(catalog, graph, parameters);
    let (mut input, mut cursor) = match plan.operators.first() {
        Some(PhysicalOperator::ScanPattern {
            optional: false,
            pattern,
            access,
            ..
        }) => {
            if !matches!(
                access,
                ScanAccessPath::Unspecified
                    | ScanAccessPath::AllNodes
                    | ScanAccessPath::Label { .. }
                    | ScanAccessPath::ResidentProperty { .. }
            ) || pattern.variable.is_some()
                || pattern.selector != PathSelector::All
                || pattern.mode != PathMode::DifferentRelationships
                || !pattern.steps.is_empty()
            {
                return Ok(None);
            }
            let Some(variable) = pattern.start.variable.as_deref() else {
                return Ok(None);
            };
            let Some(labels) = pattern
                .start
                .labels
                .iter()
                .map(|label| catalog.label(label))
                .collect::<Option<Vec<_>>>()
            else {
                return Ok(None);
            };
            let Some(property_filter) = row_pattern_property_filter(
                pattern,
                catalog,
                parameters,
                ResidentNodeBinding::Start,
            ) else {
                return Ok(None);
            };
            builder.scope.insert(
                variable.to_owned(),
                RowSymbol::Node(ResidentNodeBinding::Start),
            );
            let mut input = graph_input(project, plan, labels, graph.node_slot_count());
            if let Some(property_filter) = property_filter {
                input.property_filters.push(property_filter);
            }
            (input, 1)
        }
        Some(PhysicalOperator::Unwind {
            expression,
            variable,
        }) => {
            let Some(column) = typed_unwind_column(expression, parameters)? else {
                return Ok(None);
            };
            let rows = column.len();
            let value_type = column.value_type();
            let register = builder.push(value_type, ResidentRowOperation::InputColumn(column))?;
            builder
                .scope
                .insert(variable.clone(), RowSymbol::Register(register));
            (scalar_input(project, plan, rows), 1)
        }
        _ => return Ok(None),
    };

    while matches!(
        plan.operators.get(cursor),
        Some(PhysicalOperator::CardinalityCheckpoint { .. })
    ) {
        cursor += 1;
    }

    let mut proven_empty_filter = false;
    let lowered_input_filter =
        if let Some(PhysicalOperator::Filter(Expression::Literal(ScalarValue::Boolean(false)))) =
            plan.operators.get(cursor)
        {
            // A compile-time FALSE predicate is the strongest possible native selection proof.
            // Preserve it as a sealed zero window below; no source entity can become observable,
            // while the selected backend still executes the ordinary resident row command.
            proven_empty_filter = true;
            cursor += 1;
            true
        } else if let Some(PhysicalOperator::Filter(expression)) = plan.operators.get(cursor) {
            let Some(predicate) =
                row_integer_predicate(expression, &builder.scope, catalog, graph, parameters)
            else {
                return Ok(None);
            };
            input.predicates.push(predicate);
            cursor += 1;
            true
        } else {
            false
        };
    if source_requires_filter && !lowered_input_filter {
        return Ok(None);
    }

    let mut final_unsorted_projection = None;
    while let Some(PhysicalOperator::Project {
        keep_scope,
        projection,
    }) = plan.operators.get(cursor)
    {
        if projection.distinct {
            return Ok(None);
        }
        let Some(outputs) = builder.apply_projection(*keep_scope, projection) else {
            return Ok(None);
        };
        final_unsorted_projection = Some(outputs);
        cursor += 1;
    }
    if final_unsorted_projection.is_none() {
        return Ok(None);
    }

    let mut sort_keys = Vec::new();
    let mut offset = 0_usize;
    let mut limit = usize::MAX;
    let unsorted = !matches!(
        plan.operators.get(cursor),
        Some(PhysicalOperator::Sort(_) | PhysicalOperator::TopK { .. })
    );
    let final_projection = if unsorted {
        while let Some(operator) = plan.operators.get(cursor) {
            match operator {
                PhysicalOperator::Skip(expression) => {
                    let Some(value) = non_negative_count(expression, parameters) else {
                        return Ok(None);
                    };
                    let effective = value.min(limit);
                    offset = offset.checked_add(effective).ok_or_else(|| {
                        Error::new(
                            ErrorCode::ResultBudgetExceeded,
                            "resident row OFFSET exceeds the addressable row domain",
                        )
                    })?;
                    if limit != usize::MAX {
                        limit = limit.saturating_sub(value);
                    }
                }
                PhysicalOperator::Limit(expression) => {
                    let Some(value) = non_negative_count(expression, parameters) else {
                        return Ok(None);
                    };
                    limit = limit.min(value);
                }
                _ => return Ok(None),
            }
            cursor += 1;
        }
        final_unsorted_projection
    } else {
        let (sort_items, top_k) = match plan.operators.get(cursor) {
            Some(PhysicalOperator::Sort(items)) => (items.as_slice(), None),
            Some(PhysicalOperator::TopK { items, limit }) => (items.as_slice(), Some(*limit)),
            _ => return Ok(None),
        };
        if sort_items.is_empty() {
            return Ok(None);
        }
        cursor += 1;

        let sort_scope = builder.scope.clone();
        sort_keys.reserve(sort_items.len());
        for item in sort_items {
            let Some(RowSymbol::Register(register)) =
                builder.value(&item.expression, &sort_scope)?
            else {
                return Ok(None);
            };
            sort_keys.push(ResidentRowSortKey {
                register,
                descending: !item.ascending,
                // openCypher orders NULL after every non-null value ascending and before it
                // descending.
                nulls_first: !item.ascending,
            });
        }

        limit = top_k.unwrap_or(usize::MAX);
        let mut final_projection = None;
        while let Some(operator) = plan.operators.get(cursor) {
            match operator {
                PhysicalOperator::Project {
                    keep_scope,
                    projection,
                } => {
                    if projection.distinct {
                        return Ok(None);
                    }
                    // A post-sort expression may be evaluated for fewer rows than entered the
                    // sort; hoisting it into the pre-sort SSA frame could expose an error from a
                    // discarded row. This slice therefore admits only scope/alias projection
                    // after sorting.
                    let Some(outputs) = builder.apply_variable_projection(*keep_scope, projection)
                    else {
                        return Ok(None);
                    };
                    final_projection = Some(outputs);
                }
                PhysicalOperator::Skip(expression) => {
                    let Some(value) = non_negative_count(expression, parameters) else {
                        return Ok(None);
                    };
                    let effective = value.min(limit);
                    offset = offset.checked_add(effective).ok_or_else(|| {
                        Error::new(
                            ErrorCode::ResultBudgetExceeded,
                            "resident row OFFSET exceeds the addressable row domain",
                        )
                    })?;
                    if limit != usize::MAX {
                        limit = limit.saturating_sub(value);
                    }
                }
                PhysicalOperator::Limit(expression) => {
                    let Some(value) = non_negative_count(expression, parameters) else {
                        return Ok(None);
                    };
                    limit = limit.min(value);
                }
                _ => return Ok(None),
            }
            cursor += 1;
        }

        let Some(final_projection) = final_projection else {
            return Ok(None);
        };
        Some(final_projection)
    };

    let Some(final_projection) = final_projection else {
        return Ok(None);
    };
    if proven_empty_filter {
        limit = 0;
    }
    if final_projection.is_empty() {
        return Ok(None);
    }
    if unsorted
        && !proven_empty_filter
        && (builder.instructions.is_empty()
            || final_projection.iter().any(|(_, symbol)| {
                let RowSymbol::Register(register) = symbol else {
                    return true;
                };
                !builder.supports_unsorted_final_register(*register)
            }))
    {
        // The zero-key route owns complete fixed-width typed property projections and the exact
        // integer property-plus-constant composition proven below. Entity outputs, variable-width
        // values, and every other computed shape remain with their existing complete-plan owners.
        return Ok(None);
    }

    if proven_empty_filter && builder.instructions.is_empty() {
        // The typed-row ABI intentionally has no zero-register program. This unreachable Boolean
        // register supplies its sealed execution shape; LIMIT 0 and the FALSE proof above make it
        // impossible for the value to enter the published schema.
        builder.push(
            ResidentRowValueType::Boolean,
            ResidentRowOperation::BooleanConstant(false),
        )?;
    }

    let program = ResidentRowProgram {
        instructions: builder.instructions,
    };
    program.validate()?;

    let mut final_registers = Vec::new();
    let mut outputs = Vec::with_capacity(final_projection.len());
    for (name, symbol) in final_projection {
        let source = match symbol {
            RowSymbol::Node(binding) => CompiledResidentRowOutputSource::Node(binding),
            RowSymbol::Register(register) => {
                let value_type = program.register_type(register).ok_or_else(|| {
                    Error::internal("resident row final projection register disappeared")
                })?;
                let column = final_registers.len();
                final_registers.push(register);
                CompiledResidentRowOutputSource::ProjectedRegister { column, value_type }
            }
            RowSymbol::DurationProperty { .. } => return Ok(None),
        };
        outputs.push(CompiledResidentRowOutput { name, source });
    }

    let execution = fresh_execution_id();
    let manifest = ResidentRowProgramManifest::build(
        &program,
        &sort_keys,
        offset,
        limit,
        max_output_rows,
        &final_registers,
        1,
    )?;
    let request = ResidentRowProgramRequest {
        project,
        expected_bookmark: bookmark,
        expected_graph_revision: graph.revision(),
        expected_layout_version: graph.layout_version(),
        execution,
        input,
        program,
        manifest,
        sort_keys,
        offset,
        limit,
        max_output_rows,
        final_registers,
    };
    request.validate()?;
    Ok(Some(CompiledResidentRowPlan { request, outputs }))
}

/// Eliminates one exact, total read pipeline whose optimizer-owned `TopK(0)` proves that no
/// source row can be observed. Labels and property keys deliberately remain unresolved: on an
/// empty catalog those names are valid Cypher and the zero window makes their pure lookup and
/// ordering semantically dead. The selected backend still owns one sealed empty typed relation,
/// its expression receipt, and the empty sort/window receipt.
fn compile_pure_zero_limit_window(
    plan: &PhysicalPlan,
    project: ProjectId,
    bookmark: Bookmark,
    graph: &GraphStore,
) -> Result<Option<CompiledResidentRowPlan>> {
    let operators = plan
        .operators
        .iter()
        .filter(|operator| !matches!(operator, PhysicalOperator::CardinalityCheckpoint { .. }))
        .collect::<Vec<_>>();
    let [
        PhysicalOperator::ScanPattern {
            optional: false,
            pattern,
            access,
            ..
        },
        PhysicalOperator::Project {
            keep_scope: false,
            projection,
        },
        PhysicalOperator::TopK { items, limit: 0 },
    ] = operators.as_slice()
    else {
        return Ok(None);
    };
    if pattern.variable.is_some()
        || pattern.selector != PathSelector::All
        || pattern.mode != PathMode::DifferentRelationships
        || pattern.start.variable.is_none()
        || pattern.start.labels.len() != 1
        || pattern.start.property_predicate_present
        || !pattern.start.properties.is_empty()
        || !pattern.steps.is_empty()
        || !matches!(
            access,
            ScanAccessPath::Unspecified | ScanAccessPath::Label { .. }
        )
        || projection.distinct
    {
        return Ok(None);
    }
    let [item] = projection.items.as_slice() else {
        return Ok(None);
    };
    let Expression::Property(source, _property) = &item.expression else {
        return Ok(None);
    };
    let Expression::Variable(source_variable) = source.as_ref() else {
        return Ok(None);
    };
    if pattern.start.variable.as_deref() != Some(source_variable.as_str()) {
        return Ok(None);
    }
    let output_name = item.column_name(0);
    let [sort] = items.as_slice() else {
        return Ok(None);
    };
    let Expression::Variable(sort_variable) = &sort.expression else {
        return Ok(None);
    };
    if sort_variable != &output_name {
        return Ok(None);
    }

    // One zero-length immutable column is the physical empty relation. It never substitutes a
    // value: LIMIT 0 proves that neither the source property nor this placeholder can be observed.
    let program = ResidentRowProgram {
        instructions: vec![ResidentRowInstruction {
            output_type: ResidentRowValueType::Integer,
            operation: ResidentRowOperation::InputColumn(ResidentRowColumn::Integer {
                values: Vec::new(),
                validity: Vec::new(),
            }),
        }],
    };
    let sort_keys = vec![ResidentRowSortKey {
        register: 0,
        descending: !sort.ascending,
        nulls_first: !sort.ascending,
    }];
    let final_registers = vec![0];
    let manifest =
        ResidentRowProgramManifest::build(&program, &sort_keys, 0, 0, 0, &final_registers, 1)?;
    let request = ResidentRowProgramRequest {
        project,
        expected_bookmark: bookmark,
        expected_graph_revision: graph.revision(),
        expected_layout_version: graph.layout_version(),
        execution: fresh_execution_id(),
        input: scalar_input(project, plan, 0),
        program,
        manifest,
        sort_keys,
        offset: 0,
        limit: 0,
        max_output_rows: 0,
        final_registers,
    };
    request.validate()?;
    Ok(Some(CompiledResidentRowPlan {
        request,
        outputs: vec![CompiledResidentRowOutput {
            name: output_name,
            source: CompiledResidentRowOutputSource::ProjectedRegister {
                column: 0,
                value_type: ResidentRowValueType::Integer,
            },
        }],
    }))
}

/// Compile one complete graph-free multistage list/quantifier relation. This route accepts no
/// graph operator and returns `None` unless at least one Cypher list predicate is present. Every
/// projection, UNWIND, filter, grouping/count boundary, volatile `rand()` call, and final output
/// is retained in one immutable backend command.
#[allow(clippy::too_many_arguments)]
fn compile_entity_list_quantifier(
    plan: &PhysicalPlan,
    project: ProjectId,
    bookmark: Bookmark,
    catalog: &crate::graph::NameCatalog,
    graph: &GraphStore,
    parameters: &BTreeMap<String, ResultValue>,
    max_output_rows: usize,
) -> Result<Option<CompiledResidentQuantifierPlan>> {
    let operators = plan
        .operators
        .iter()
        .filter(|operator| {
            !matches!(
                operator,
                PhysicalOperator::CardinalityCheckpoint { .. } | PhysicalOperator::Finish
            )
        })
        .collect::<Vec<_>>();
    let [
        PhysicalOperator::ScanPattern {
            optional: false,
            pattern: path,
            ..
        },
        PhysicalOperator::Project {
            keep_scope: false,
            projection: source_projection,
        },
        remaining @ ..,
    ] = operators.as_slice()
    else {
        return Ok(None);
    };
    if path.steps.is_empty() {
        return Ok(None);
    }
    let Some(path_variable) = path.variable.as_deref() else {
        return Ok(None);
    };
    let Some((entity_kind, entity_name, count_name)) =
        quantifier_entity_source_projection(source_projection, path_variable)
    else {
        return Ok(None);
    };

    let execution = fresh_execution_id();
    let Some(path_request) = super::resident_variable_path::compile_request(
        plan,
        project,
        bookmark,
        catalog,
        graph,
        None,
        path,
        execution,
        max_output_rows,
    )?
    else {
        return Ok(None);
    };

    let mut builder = QuantifierProgramBuilder::new(parameters);
    let source_output = builder.seed_entity_source(&entity_name, count_name.as_deref())?;
    for operator in remaining {
        match operator {
            PhysicalOperator::Project {
                keep_scope,
                projection,
            } => {
                if projection.distinct || !builder.project(*keep_scope, projection)? {
                    return Ok(None);
                }
            }
            PhysicalOperator::Unwind {
                expression,
                variable,
            } => {
                if !builder.unwind(expression, variable)? {
                    return Ok(None);
                }
            }
            PhysicalOperator::Filter(expression) => {
                if !builder.filter(expression)? {
                    return Ok(None);
                }
            }
            _ => return Ok(None),
        }
    }
    if (!builder.saw_quantifier && !builder.saw_unwind) || builder.outputs.is_empty() {
        return Ok(None);
    }
    let slot_count = u16::try_from(builder.next_slot).map_err(|_| {
        Error::new(
            ErrorCode::GpuAdmissionFailure,
            "resident entity quantifier compiler exceeded the slot address space",
        )
    })?;
    let program = ResidentQuantifierProgram {
        slot_count,
        stages: builder.stages,
        outputs: builder.outputs,
    };
    let mut properties = Vec::new();
    for (key, property) in plan
        .dependencies
        .iter()
        .filter(|dependency| dependency.kind == DependencyKind::Property)
        .map(|dependency| (dependency.name.clone(), catalog.property(&dependency.name)))
        .collect::<BTreeMap<_, _>>()
    {
        let values = match (entity_kind, property) {
            (_, None) => Vec::new(),
            (EntityKind::Node, Some(property)) => graph
                .nodes()
                .filter_map(|node| node.property(property))
                .collect::<Vec<_>>(),
            (EntityKind::Relationship, Some(property)) => graph
                .edges()
                .filter_map(|relationship| relationship.property(property))
                .collect::<Vec<_>>(),
        };
        let Some(shape) = ResidentQuantifierEntityPropertyShape::from_values(values)? else {
            return Ok(None);
        };
        properties.push(ResidentQuantifierEntityProperty {
            key,
            property,
            shape,
        });
    }
    let source = ResidentQuantifierSource::VariablePathEntityList {
        path: path_request,
        output: source_output,
        entity_kind: match entity_kind {
            EntityKind::Node => ResidentQuantifierEntityKind::Node,
            EntityKind::Relationship => ResidentQuantifierEntityKind::Relationship,
        },
        skip: 1,
        properties,
        materialize_obligation: ResidentExecutionObligation {
            id: QUANTIFIER_ENTITY_LIST_MATERIALIZE_OBLIGATION,
            kind: ResidentObligationKind::Expression,
            scope: ResidentObligationScope::Expression(u16::MAX),
        },
    };
    let mut random = rand::rng();
    let request = ResidentQuantifierProgramRequest::build_with_source(
        ResidentQuantifierGeneration {
            project,
            bookmark,
            graph_revision: graph.revision(),
            layout_version: graph.layout_version(),
            catalog_generation: catalog.optimizer_generation(),
        },
        execution,
        source,
        program,
        max_output_rows,
        crate::execution::RESIDENT_QUANTIFIER_MAX_LITERAL_ITEMS,
        random.random(),
    )?;
    Ok(Some(CompiledResidentQuantifierPlan { request }))
}

pub fn compile_quantifier(
    plan: &PhysicalPlan,
    project: ProjectId,
    bookmark: Bookmark,
    catalog: &crate::graph::NameCatalog,
    graph: &GraphStore,
    parameters: &BTreeMap<String, ResultValue>,
    max_output_rows: usize,
) -> Result<Option<CompiledResidentQuantifierPlan>> {
    if !plan.read_only
        || plan.at_time.is_some()
        || !plan.unions.is_empty()
        || plan.operators.is_empty()
        || max_output_rows == 0
    {
        return Ok(None);
    }

    if let Some(compiled) = compile_entity_list_quantifier(
        plan,
        project,
        bookmark,
        catalog,
        graph,
        parameters,
        max_output_rows,
    )? {
        return Ok(Some(compiled));
    }

    let operators = plan
        .operators
        .iter()
        .filter(|operator| {
            !matches!(
                operator,
                PhysicalOperator::CardinalityCheckpoint { .. } | PhysicalOperator::Finish
            )
        })
        .collect::<Vec<_>>();
    let max_rows = max_output_rows;
    let mut builder = QuantifierProgramBuilder::new(parameters);
    let mut source = None;
    let mut operator_index = 0_usize;
    while let Some(operator) = operators.get(operator_index) {
        match operator {
            PhysicalOperator::Project {
                keep_scope,
                projection,
            } => {
                if let Some(next) = operators.get(operator_index + 1).copied()
                    && let PhysicalOperator::Unwind {
                        expression,
                        variable,
                    } = next
                    && builder.project_collect_unwind(
                        *keep_scope,
                        projection,
                        expression,
                        variable,
                    )?
                {
                    operator_index += 2;
                    continue;
                }
                if projection.distinct || !builder.project(*keep_scope, projection)? {
                    return Ok(None);
                }
            }
            PhysicalOperator::Unwind {
                expression,
                variable,
            } => {
                if source.is_none()
                    && let Some(range) =
                        builder.seed_range_source(expression, variable, max_rows)?
                {
                    source = Some(range);
                    operator_index += 1;
                    continue;
                }
                if !builder.unwind(expression, variable)? {
                    return Ok(None);
                }
            }
            PhysicalOperator::Filter(expression) => {
                if !builder.filter(expression)? {
                    return Ok(None);
                }
            }
            _ => return Ok(None),
        }
        operator_index += 1;
    }
    if (!builder.saw_quantifier
        && !builder.saw_unwind
        && !builder.saw_size_expression
        && !builder.saw_graph_free_value_function)
        || builder.outputs.is_empty()
    {
        return Ok(None);
    }
    let slot_count = u16::try_from(builder.next_slot).map_err(|_| {
        Error::new(
            ErrorCode::GpuAdmissionFailure,
            "resident quantifier compiler exceeded the slot address space",
        )
    })?;
    let program = ResidentQuantifierProgram {
        slot_count,
        stages: builder.stages,
        outputs: builder.outputs,
    };
    let mut random = rand::rng();
    let request = ResidentQuantifierProgramRequest::build_with_source(
        ResidentQuantifierGeneration {
            project,
            bookmark,
            graph_revision: graph.revision(),
            layout_version: graph.layout_version(),
            catalog_generation: catalog.optimizer_generation(),
        },
        fresh_execution_id(),
        source.unwrap_or(ResidentQuantifierSource::Unit),
        program,
        max_rows,
        crate::execution::RESIDENT_QUANTIFIER_MAX_LITERAL_ITEMS,
        random.random(),
    )?;
    Ok(Some(CompiledResidentQuantifierPlan { request }))
}

struct QuantifierProgramBuilder<'a> {
    parameters: &'a BTreeMap<String, ResultValue>,
    scope: BTreeMap<String, ResidentQuantifierSlot>,
    next_slot: usize,
    stages: Vec<ResidentQuantifierStage>,
    outputs: Vec<ResidentQuantifierOutput>,
    saw_quantifier: bool,
    saw_unwind: bool,
    /// True only when this complete program successfully lowered `size()`. Bare list values and
    /// list operators already have established routes; they must not select this quantifier route
    /// by themselves because doing so steals existing scalar/comparison execution plans.
    saw_size_expression: bool,
    /// True only for a graph-free value function whose complete typed value, allocation, and
    /// publication semantics are already owned by the quantifier ABI. This remains deliberately
    /// narrower than "saw any function": ordinary scalar functions must keep their established
    /// routes, and a newly parsed function may not select this command merely because it lowered.
    saw_graph_free_value_function: bool,
    non_null_slots: BTreeSet<ResidentQuantifierSlot>,
    has_unmaterialized_scope_value: bool,
}

impl<'a> QuantifierProgramBuilder<'a> {
    fn new(parameters: &'a BTreeMap<String, ResultValue>) -> Self {
        Self {
            parameters,
            scope: BTreeMap::new(),
            next_slot: 0,
            stages: Vec::new(),
            outputs: Vec::new(),
            saw_quantifier: false,
            saw_unwind: false,
            saw_size_expression: false,
            saw_graph_free_value_function: false,
            non_null_slots: BTreeSet::new(),
            has_unmaterialized_scope_value: false,
        }
    }

    fn slot(&mut self) -> Result<ResidentQuantifierSlot> {
        if self.next_slot >= crate::execution::RESIDENT_QUANTIFIER_MAX_SLOTS {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident quantifier compiler exceeded its bounded slot count",
            ));
        }
        let slot = ResidentQuantifierSlot(u16::try_from(self.next_slot).map_err(|_| {
            Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident quantifier slot is not addressable",
            )
        })?);
        self.next_slot += 1;
        Ok(slot)
    }

    fn seed_entity_source(
        &mut self,
        entity_name: &str,
        count_name: Option<&str>,
    ) -> Result<ResidentQuantifierSlot> {
        if entity_name.is_empty()
            || count_name.is_some_and(str::is_empty)
            || count_name == Some(entity_name)
            || !self.scope.is_empty()
            || !self.stages.is_empty()
        {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident entity quantifier source has invalid or duplicate bindings",
            ));
        }
        let source = self.slot()?;
        if let Some(count_name) = count_name {
            let grouped = self.slot()?;
            let count = self.slot()?;
            self.stages.push(ResidentQuantifierStage::GroupCount {
                groups: vec![ResidentQuantifierProjection {
                    output: grouped,
                    expression: ResidentQuantifierExpression::Slot(source),
                }],
                count_outputs: vec![count],
            });
            self.scope.insert(entity_name.to_owned(), grouped);
            self.scope.insert(count_name.to_owned(), count);
        } else {
            self.scope.insert(entity_name.to_owned(), source);
        }
        self.outputs.clear();
        Ok(source)
    }

    fn seed_range_source(
        &mut self,
        expression: &Expression,
        variable: &str,
        max_rows: usize,
    ) -> Result<Option<ResidentQuantifierSource>> {
        if !self.scope.is_empty() || !self.stages.is_empty() || !self.outputs.is_empty() {
            return Ok(None);
        }
        let Some(request) = compile_range_expression(expression, self.parameters, max_rows) else {
            return Ok(None);
        };
        request.validate_streaming()?;
        let output = self.slot()?;
        self.scope.insert(variable.to_owned(), output);
        self.non_null_slots.insert(output);
        self.outputs.clear();
        self.saw_unwind = true;
        Ok(Some(ResidentQuantifierSource::Range {
            request,
            output,
            obligation: ResidentExecutionObligation {
                id: QUANTIFIER_RANGE_SOURCE_OBLIGATION,
                kind: ResidentObligationKind::Expression,
                scope: ResidentObligationScope::Expression(u16::MAX - 2),
            },
        }))
    }

    fn project(&mut self, keep_scope: bool, projection: &Projection) -> Result<bool> {
        if self.has_unmaterialized_scope_value
            && projection
                .items
                .iter()
                .any(|item| matches!(item.expression, Expression::Star))
        {
            return Ok(false);
        }
        let incoming = self.scope.clone();
        let aggregate = projection
            .items
            .iter()
            .any(|item| contains_aggregate(&item.expression));
        let mut names = Vec::with_capacity(projection.items.len());
        let mut outputs = Vec::with_capacity(projection.items.len());
        if aggregate {
            if keep_scope {
                return Ok(false);
            }
            let mut groups = Vec::new();
            let mut count_outputs = Vec::new();
            for (index, item) in projection.items.iter().enumerate() {
                let name = item
                    .alias
                    .clone()
                    .unwrap_or_else(|| item.column_name(index));
                let output = self.slot()?;
                names.push((name, output));
                if quantifier_count_star(&item.expression)
                    || quantifier_count_non_null_slot(
                        &item.expression,
                        &incoming,
                        &self.non_null_slots,
                    )
                {
                    count_outputs.push(output);
                } else {
                    if contains_aggregate(&item.expression) {
                        return Ok(false);
                    }
                    let Some(expression) = self.expression(&item.expression, &incoming)? else {
                        return Ok(false);
                    };
                    groups.push(ResidentQuantifierProjection { output, expression });
                }
            }
            if count_outputs.is_empty() {
                return Ok(false);
            }
            self.stages.push(ResidentQuantifierStage::GroupCount {
                groups,
                count_outputs,
            });
        } else {
            let mut projected_names = BTreeSet::new();
            for (index, item) in projection.items.iter().enumerate() {
                if matches!(item.expression, Expression::Star) && item.alias.is_none() {
                    for (name, source) in &incoming {
                        if !projected_names.insert(name.clone()) {
                            return Ok(false);
                        }
                        let output = self.slot()?;
                        names.push((name.clone(), output));
                        outputs.push(ResidentQuantifierProjection {
                            output,
                            expression: ResidentQuantifierExpression::Slot(*source),
                        });
                    }
                    continue;
                }
                let Some(expression) = self.expression(&item.expression, &incoming)? else {
                    return Ok(false);
                };
                let output = self.slot()?;
                let name = item
                    .alias
                    .clone()
                    .unwrap_or_else(|| item.column_name(index));
                if !projected_names.insert(name.clone()) {
                    return Ok(false);
                }
                names.push((name, output));
                outputs.push(ResidentQuantifierProjection { output, expression });
            }
            if outputs.is_empty() {
                return Ok(false);
            }
            self.stages.push(ResidentQuantifierStage::Project {
                keep_scope,
                bindings: outputs,
            });
        }

        let mut next = keep_scope.then_some(incoming).unwrap_or_default();
        for (name, slot) in &names {
            next.insert(name.clone(), *slot);
        }
        self.scope = next;
        if !keep_scope {
            self.has_unmaterialized_scope_value = false;
        }
        self.outputs = names
            .into_iter()
            .map(|(name, source)| ResidentQuantifierOutput { name, source })
            .collect();
        Ok(true)
    }

    fn project_collect_unwind(
        &mut self,
        keep_scope: bool,
        projection: &Projection,
        unwind_expression: &Expression,
        unwind_variable: &str,
    ) -> Result<bool> {
        if keep_scope || projection.distinct || projection.items.is_empty() {
            return Ok(false);
        }
        let incoming = self.scope.clone();
        let mut collect = None;
        for (index, item) in projection.items.iter().enumerate() {
            let Expression::Function {
                name,
                distinct: false,
                arguments,
            } = &item.expression
            else {
                if contains_aggregate(&item.expression) {
                    return Ok(false);
                }
                continue;
            };
            if name.len() == 1 && name[0].eq_ignore_ascii_case("collect") {
                if collect.is_some() || arguments.len() != 1 {
                    return Ok(false);
                }
                collect = Some((index, item, &arguments[0]));
            } else if contains_aggregate(&item.expression) {
                return Ok(false);
            }
        }
        let Some((collect_index, collect_item, collect_argument)) = collect else {
            return Ok(false);
        };
        let collect_name = collect_item
            .alias
            .clone()
            .unwrap_or_else(|| collect_item.column_name(collect_index));
        if !matches!(unwind_expression, Expression::Variable(name) if name == &collect_name) {
            return Ok(false);
        }
        let Some(collect_expression) = self.expression(collect_argument, &incoming)? else {
            return Ok(false);
        };
        let ResidentQuantifierExpression::Slot(collected_slot) = collect_expression else {
            return Ok(false);
        };
        if !self.non_null_slots.contains(&collected_slot) {
            return Ok(false);
        }

        let mut bindings = Vec::with_capacity(projection.items.len());
        let mut names = Vec::with_capacity(projection.items.len());
        for (index, item) in projection.items.iter().enumerate() {
            if index == collect_index {
                continue;
            }
            let Some(expression) = self.expression(&item.expression, &incoming)? else {
                return Ok(false);
            };
            let output = self.slot()?;
            let name = item
                .alias
                .clone()
                .unwrap_or_else(|| item.column_name(index));
            names.push((name, output));
            bindings.push(ResidentQuantifierProjection { output, expression });
        }
        let output = self.slot()?;
        bindings.push(ResidentQuantifierProjection {
            output,
            expression: ResidentQuantifierExpression::Slot(collected_slot),
        });
        names.push((unwind_variable.to_owned(), output));
        self.stages.push(ResidentQuantifierStage::Project {
            keep_scope: false,
            bindings,
        });
        self.scope = names.into_iter().collect();
        self.non_null_slots.insert(output);
        self.outputs.clear();
        self.saw_unwind = true;
        self.has_unmaterialized_scope_value = collect_name != unwind_variable;
        Ok(true)
    }

    fn unwind(&mut self, expression: &Expression, variable: &str) -> Result<bool> {
        let scope = self.scope.clone();
        let Some(expression) = self.expression(expression, &scope)? else {
            return Ok(false);
        };
        // `split()` either yields NULL (and therefore no UNWIND rows) or a list containing only
        // concrete strings. Record that exact proof so a following `count(item)` may use the
        // existing row-count reduction without changing Cypher's null-skipping semantics. Do not
        // infer this for arbitrary lists or list-valued functions, which may contain NULL items.
        let emits_only_non_null_items = matches!(
            &expression,
            ResidentQuantifierExpression::Function {
                function: ResidentQuantifierFunction::Split,
                ..
            }
        );
        let output = self.slot()?;
        self.stages
            .push(ResidentQuantifierStage::Unwind { expression, output });
        self.scope.insert(variable.to_owned(), output);
        if emits_only_non_null_items {
            self.non_null_slots.insert(output);
        }
        self.outputs.clear();
        self.saw_unwind = true;
        Ok(true)
    }

    fn filter(&mut self, expression: &Expression) -> Result<bool> {
        let scope = self.scope.clone();
        let Some(predicate) = self.expression(expression, &scope)? else {
            return Ok(false);
        };
        self.stages
            .push(ResidentQuantifierStage::Filter { predicate });
        self.outputs.clear();
        Ok(true)
    }

    fn expression(
        &mut self,
        expression: &Expression,
        scope: &BTreeMap<String, ResidentQuantifierSlot>,
    ) -> Result<Option<ResidentQuantifierExpression>> {
        let compiled = match expression {
            Expression::Literal(value) => {
                let Some(value) = quantifier_scalar_value(value) else {
                    return Ok(None);
                };
                ResidentQuantifierExpression::Literal(value)
            }
            Expression::Parameter(name) => {
                let Some(value) = self.parameters.get(name).and_then(quantifier_result_value)
                else {
                    return Ok(None);
                };
                ResidentQuantifierExpression::Literal(value)
            }
            Expression::Variable(variable) => {
                let Some(slot) = scope.get(variable).copied() else {
                    return Ok(None);
                };
                ResidentQuantifierExpression::Slot(slot)
            }
            Expression::Property(source, key) => {
                let Some(source) = self.expression(source, scope)? else {
                    return Ok(None);
                };
                ResidentQuantifierExpression::Property {
                    source: Box::new(source),
                    key: key.clone(),
                }
            }
            Expression::List(values) => {
                let mut compiled = Vec::with_capacity(values.len());
                for value in values {
                    let Some(value) = self.expression(value, scope)? else {
                        return Ok(None);
                    };
                    compiled.push(value);
                }
                ResidentQuantifierExpression::List(compiled)
            }
            Expression::Map(entries) => {
                let mut compiled = Vec::with_capacity(entries.len());
                for (key, value) in entries {
                    let Some(value) = self.expression(value, scope)? else {
                        return Ok(None);
                    };
                    compiled.push((key.clone(), value));
                }
                compiled.sort_by(|left, right| left.0.cmp(&right.0));
                if compiled.windows(2).any(|pair| pair[0].0 == pair[1].0) {
                    return Ok(None);
                }
                ResidentQuantifierExpression::Map(compiled)
            }
            Expression::Case {
                operand,
                alternatives,
                default,
            } => {
                let operand = match operand {
                    Some(operand) => {
                        let Some(operand) = self.expression(operand, scope)? else {
                            return Ok(None);
                        };
                        Some(Box::new(operand))
                    }
                    None => None,
                };
                let mut compiled = Vec::with_capacity(alternatives.len());
                for alternative in alternatives {
                    let (Some(when), Some(then)) = (
                        self.expression(&alternative.when, scope)?,
                        self.expression(&alternative.then, scope)?,
                    ) else {
                        return Ok(None);
                    };
                    compiled.push((when, then));
                }
                let default = match default {
                    Some(default) => {
                        let Some(default) = self.expression(default, scope)? else {
                            return Ok(None);
                        };
                        Some(Box::new(default))
                    }
                    None => None,
                };
                ResidentQuantifierExpression::Case {
                    operand,
                    alternatives: compiled,
                    default,
                }
            }
            Expression::ListComprehension {
                variable,
                list,
                predicate,
                projection,
            } => {
                let Some(list) = self.expression(list, scope)? else {
                    return Ok(None);
                };
                let local = self.slot()?;
                let mut nested = scope.clone();
                nested.insert(variable.clone(), local);
                let predicate = match predicate {
                    Some(predicate) => {
                        let Some(predicate) = self.expression(predicate, &nested)? else {
                            return Ok(None);
                        };
                        Some(Box::new(predicate))
                    }
                    None => None,
                };
                let projection = match projection {
                    Some(projection) => {
                        let Some(projection) = self.expression(projection, &nested)? else {
                            return Ok(None);
                        };
                        Some(Box::new(projection))
                    }
                    None => None,
                };
                ResidentQuantifierExpression::ListComprehension {
                    variable: local,
                    list: Box::new(list),
                    predicate,
                    projection,
                }
            }
            Expression::ListPredicate {
                kind,
                variable,
                list,
                predicate,
            } => {
                let Some(list) = self.expression(list, scope)? else {
                    return Ok(None);
                };
                let local = self.slot()?;
                let mut nested = scope.clone();
                nested.insert(variable.clone(), local);
                let Some(predicate) = self.expression(predicate, &nested)? else {
                    return Ok(None);
                };
                self.saw_quantifier = true;
                ResidentQuantifierExpression::Predicate {
                    kind: match kind {
                        ListPredicateKind::All => ResidentQuantifierKind::All,
                        ListPredicateKind::Any => ResidentQuantifierKind::Any,
                        ListPredicateKind::None => ResidentQuantifierKind::None,
                        ListPredicateKind::Single => ResidentQuantifierKind::Single,
                    },
                    variable: local,
                    list: Box::new(list),
                    predicate: Box::new(predicate),
                }
            }
            Expression::Function {
                name,
                distinct: false,
                arguments,
            } => {
                let function = match name
                    .iter()
                    .map(String::as_str)
                    .collect::<Vec<_>>()
                    .as_slice()
                {
                    [name] if name.eq_ignore_ascii_case("rand") => ResidentQuantifierFunction::Rand,
                    [name] if name.eq_ignore_ascii_case("reverse") => {
                        ResidentQuantifierFunction::Reverse
                    }
                    [name] if name.eq_ignore_ascii_case("size") => ResidentQuantifierFunction::Size,
                    [name] if name.eq_ignore_ascii_case("coalesce") => {
                        ResidentQuantifierFunction::Coalesce
                    }
                    [name] if name.eq_ignore_ascii_case("split") => {
                        ResidentQuantifierFunction::Split
                    }
                    [name] if name.eq_ignore_ascii_case("toString") => {
                        ResidentQuantifierFunction::ToString
                    }
                    [name] if name.eq_ignore_ascii_case("substring") => {
                        ResidentQuantifierFunction::Substring
                    }
                    _ => return Ok(None),
                };
                let mut compiled = Vec::with_capacity(arguments.len());
                for argument in arguments {
                    let Some(argument) = self.expression(argument, scope)? else {
                        return Ok(None);
                    };
                    compiled.push(argument);
                }
                if function == ResidentQuantifierFunction::Size {
                    self.saw_size_expression = true;
                } else if function == ResidentQuantifierFunction::Reverse
                    && matches!(
                        compiled.as_slice(),
                        [ResidentQuantifierExpression::Literal(
                            ResidentQuantifierValue::Null | ResidentQuantifierValue::String(_)
                        )] | [ResidentQuantifierExpression::List(_)]
                    )
                {
                    self.saw_graph_free_value_function = true;
                } else if function == ResidentQuantifierFunction::Substring
                    && matches!(
                        compiled.as_slice(),
                        [
                            ResidentQuantifierExpression::Literal(
                                ResidentQuantifierValue::Null | ResidentQuantifierValue::String(_)
                            ),
                            ResidentQuantifierExpression::Literal(
                                ResidentQuantifierValue::Null | ResidentQuantifierValue::Integer(_)
                            ),
                        ] | [
                            ResidentQuantifierExpression::Literal(
                                ResidentQuantifierValue::Null | ResidentQuantifierValue::String(_)
                            ),
                            ResidentQuantifierExpression::Literal(
                                ResidentQuantifierValue::Null | ResidentQuantifierValue::Integer(_)
                            ),
                            ResidentQuantifierExpression::Literal(
                                ResidentQuantifierValue::Null | ResidentQuantifierValue::Integer(_)
                            ),
                        ]
                    )
                {
                    self.saw_graph_free_value_function = true;
                } else if function == ResidentQuantifierFunction::ToString
                    && matches!(
                        compiled.as_slice(),
                        [ResidentQuantifierExpression::Slot(slot)]
                            if !self.scope.values().any(|source| source == slot)
                    )
                {
                    // A slot outside the persistent stage scope is a bounded local binding of the
                    // list-comprehension/predicate currently being lowered. This admits the exact
                    // complete device expression without stealing ordinary top-level toString()
                    // scalars from their established typed-row route.
                    self.saw_graph_free_value_function = true;
                }
                ResidentQuantifierExpression::Function {
                    function,
                    arguments: compiled,
                }
            }
            Expression::Unary { operation, operand } => {
                let Some(operand) = self.expression(operand, scope)? else {
                    return Ok(None);
                };
                ResidentQuantifierExpression::Unary {
                    operation: match operation {
                        UnaryOperator::Not => ResidentQuantifierUnary::Not,
                        UnaryOperator::Positive => ResidentQuantifierUnary::Positive,
                        UnaryOperator::Negative => ResidentQuantifierUnary::Negative,
                    },
                    operand: Box::new(operand),
                }
            }
            Expression::Binary {
                left,
                operation,
                right,
            } => {
                let (Some(left), Some(right)) = (
                    self.expression(left, scope)?,
                    self.expression(right, scope)?,
                ) else {
                    return Ok(None);
                };
                if matches!(
                    operation,
                    BinaryOperator::StartsWith
                        | BinaryOperator::EndsWith
                        | BinaryOperator::Contains
                ) {
                    // A complete graph-free string-predicate expression is itself sufficient to
                    // select this command, even without UNWIND/list-predicate stages. Both
                    // operands and the three-valued result remain device-evaluated.
                    self.saw_graph_free_value_function = true;
                }
                let operation = match operation {
                    BinaryOperator::Or => ResidentQuantifierBinary::Or,
                    BinaryOperator::Xor => ResidentQuantifierBinary::Xor,
                    BinaryOperator::And => ResidentQuantifierBinary::And,
                    BinaryOperator::Equal => ResidentQuantifierBinary::Equal,
                    BinaryOperator::NotEqual => ResidentQuantifierBinary::NotEqual,
                    BinaryOperator::Less => ResidentQuantifierBinary::Less,
                    BinaryOperator::LessOrEqual => ResidentQuantifierBinary::LessOrEqual,
                    BinaryOperator::Greater => ResidentQuantifierBinary::Greater,
                    BinaryOperator::GreaterOrEqual => ResidentQuantifierBinary::GreaterOrEqual,
                    BinaryOperator::Concat | BinaryOperator::Add => ResidentQuantifierBinary::Add,
                    BinaryOperator::Subtract => ResidentQuantifierBinary::Subtract,
                    BinaryOperator::Multiply => ResidentQuantifierBinary::Multiply,
                    BinaryOperator::Divide => ResidentQuantifierBinary::Divide,
                    BinaryOperator::Modulo => ResidentQuantifierBinary::Modulo,
                    BinaryOperator::StartsWith => ResidentQuantifierBinary::StartsWith,
                    BinaryOperator::EndsWith => ResidentQuantifierBinary::EndsWith,
                    BinaryOperator::Contains => ResidentQuantifierBinary::Contains,
                    _ => return Ok(None),
                };
                ResidentQuantifierExpression::Binary {
                    left: Box::new(left),
                    operation,
                    right: Box::new(right),
                }
            }
            Expression::IsNull {
                expression,
                negated,
            } => {
                let Some(expression) = self.expression(expression, scope)? else {
                    return Ok(None);
                };
                ResidentQuantifierExpression::IsNull {
                    expression: Box::new(expression),
                    negated: *negated,
                }
            }
            Expression::MapProjection { .. }
            | Expression::Reduce { .. }
            | Expression::Function { distinct: true, .. }
            | Expression::Index { .. }
            | Expression::Slice { .. }
            | Expression::ExistentialSubquery(_)
            | Expression::Star => return Ok(None),
        };
        Ok(Some(compiled))
    }
}

fn quantifier_count_star(expression: &Expression) -> bool {
    matches!(
        expression,
        Expression::Function {
            name,
            distinct: false,
            arguments,
        } if name.len() == 1
            && name[0].eq_ignore_ascii_case("count")
            && matches!(arguments.as_slice(), [Expression::Star])
    )
}

fn quantifier_count_non_null_slot(
    expression: &Expression,
    scope: &BTreeMap<String, ResidentQuantifierSlot>,
    non_null_slots: &BTreeSet<ResidentQuantifierSlot>,
) -> bool {
    matches!(
        expression,
        Expression::Function {
            name,
            distinct: false,
            arguments,
        } if name.len() == 1
            && name[0].eq_ignore_ascii_case("count")
            && matches!(
                arguments.as_slice(),
                [Expression::Variable(variable)]
                    if scope
                        .get(variable)
                        .is_some_and(|slot| non_null_slots.contains(slot))
            )
    )
}

fn quantifier_tail_entity_list(expression: &Expression, path_variable: &str) -> Option<EntityKind> {
    let Expression::Function {
        name: tail_name,
        distinct: false,
        arguments: tail_arguments,
    } = expression
    else {
        return None;
    };
    if tail_name.len() != 1
        || !tail_name[0].eq_ignore_ascii_case("tail")
        || tail_arguments.len() != 1
    {
        return None;
    }
    let Expression::Function {
        name: entity_name,
        distinct: false,
        arguments: entity_arguments,
    } = &tail_arguments[0]
    else {
        return None;
    };
    if entity_name.len() != 1
        || !matches!(
            entity_arguments.as_slice(),
            [Expression::Variable(variable)] if variable == path_variable
        )
    {
        return None;
    }
    if entity_name[0].eq_ignore_ascii_case("nodes") {
        Some(EntityKind::Node)
    } else if entity_name[0].eq_ignore_ascii_case("relationships") {
        Some(EntityKind::Relationship)
    } else {
        None
    }
}

fn quantifier_entity_source_projection(
    projection: &Projection,
    path_variable: &str,
) -> Option<(EntityKind, String, Option<String>)> {
    if projection.distinct {
        return None;
    }
    let mut entity = None;
    let mut count = None;
    for (index, item) in projection.items.iter().enumerate() {
        if let Some(kind) = quantifier_tail_entity_list(&item.expression, path_variable) {
            if entity.is_some() {
                return None;
            }
            entity = Some((kind, item.column_name(index)));
        } else if quantifier_count_star(&item.expression) {
            if count.is_some() {
                return None;
            }
            count = Some(item.column_name(index));
        } else {
            return None;
        }
    }
    let (kind, name) = entity?;
    match (kind, count.as_ref()) {
        (EntityKind::Node, None) | (EntityKind::Relationship, Some(_)) => Some((kind, name, count)),
        _ => None,
    }
}

fn quantifier_scalar_value(value: &ScalarValue) -> Option<ResidentQuantifierValue> {
    match value {
        ScalarValue::Null => Some(ResidentQuantifierValue::Null),
        ScalarValue::Boolean(value) => Some(ResidentQuantifierValue::Boolean(*value)),
        ScalarValue::Integer(value) => Some(ResidentQuantifierValue::Integer(*value)),
        ScalarValue::Float(value) => Some(ResidentQuantifierValue::Float(value.0.to_bits())),
        ScalarValue::String(value) => Some(ResidentQuantifierValue::String(value.to_string())),
        ScalarValue::Bytes(_)
        | ScalarValue::Date(_)
        | ScalarValue::LocalTime(_)
        | ScalarValue::ZonedTime { .. }
        | ScalarValue::LocalDateTime { .. }
        | ScalarValue::ZonedDateTime { .. }
        | ScalarValue::Duration { .. }
        | ScalarValue::List(_)
        | ScalarValue::Map(_) => None,
    }
}

fn quantifier_result_value(value: &ResultValue) -> Option<ResidentQuantifierValue> {
    match value {
        ResultValue::Scalar(value) => quantifier_scalar_value(value),
        ResultValue::List(values) => values
            .iter()
            .map(quantifier_result_value)
            .collect::<Option<Vec<_>>>()
            .map(ResidentQuantifierValue::List),
        ResultValue::Map(entries) => entries
            .iter()
            .map(|(key, value)| Some((key.clone(), quantifier_result_value(value)?)))
            .collect::<Option<Vec<_>>>()
            .map(ResidentQuantifierValue::Map),
        ResultValue::Node(_)
        | ResultValue::Relationship(_)
        | ResultValue::Path { .. }
        | ResultValue::Vector(_) => None,
    }
}

/// Proves the exact heterogeneous-list identity used by Graph4 [5] without materializing an Any
/// list on the host or adding a second device ABI. `[r, 1][0]` is definitionally the already-live
/// relationship `r`; the rewritten physical plan therefore projects that relationship through the
/// WITH boundary and lets the ordinary native relationship-type output own publication. Every
/// other list shape, index, path shape, projection tail, and relationship domain remains with its
/// complete-plan owner.
fn exact_relationship_list_zero_type_plan(plan: &PhysicalPlan) -> Option<PhysicalPlan> {
    if !plan.read_only || plan.at_time.is_some() || !plan.unions.is_empty() {
        return None;
    }
    let operators = plan
        .operators
        .iter()
        .enumerate()
        .filter(|(_, operator)| !matches!(operator, PhysicalOperator::CardinalityCheckpoint { .. }))
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
            list_index,
            PhysicalOperator::Project {
                keep_scope: false,
                projection: list_projection,
            },
        ),
        (
            final_index,
            PhysicalOperator::Project {
                keep_scope: false,
                projection: final_projection,
            },
        ),
    ] = operators.as_slice()
    else {
        return None;
    };
    let [step] = pattern.steps.as_slice() else {
        return None;
    };
    if pattern.variable.is_some()
        || pattern.selector != PathSelector::All
        || pattern.mode != PathMode::DifferentRelationships
        || pattern
            .start
            .variable
            .as_ref()
            .is_none_or(|variable| variable.is_empty())
        || !pattern.start.labels.is_empty()
        || pattern.start.property_predicate_present
        || !pattern.start.properties.is_empty()
        || step
            .relationship
            .variable
            .as_ref()
            .is_none_or(|variable| variable.is_empty())
        || !step.relationship.types.is_empty()
        || step.relationship.direction != Direction::Outgoing
        || step.relationship.variable_length
        || step.relationship.min_hops.is_some()
        || step.relationship.max_hops.is_some()
        || !step.relationship.properties.is_empty()
        || step.node.variable.is_some()
        || !step.node.labels.is_empty()
        || step.node.property_predicate_present
        || !step.node.properties.is_empty()
        || list_projection.distinct
        || final_projection.distinct
    {
        return None;
    }
    let relationship = step.relationship.variable.as_ref()?;
    let [list_item] = list_projection.items.as_slice() else {
        return None;
    };
    let list_name = list_item.alias.as_ref()?;
    if list_name.is_empty()
        || !matches!(
            &list_item.expression,
            Expression::List(elements)
                if matches!(
                    elements.as_slice(),
                    [
                        Expression::Variable(variable),
                        Expression::Literal(ScalarValue::Integer(1)),
                    ] if variable == relationship
                )
        )
    {
        return None;
    }
    let [final_item] = final_projection.items.as_slice() else {
        return None;
    };
    if final_item.alias.is_some() {
        return None;
    }
    let Expression::Function {
        name,
        distinct: false,
        arguments,
    } = &final_item.expression
    else {
        return None;
    };
    if !matches!(name.as_slice(), [name] if name.eq_ignore_ascii_case("type"))
        || !matches!(
            arguments.as_slice(),
            [Expression::Index { expression, index }]
                if matches!(expression.as_ref(), Expression::Variable(variable) if variable == list_name)
                    && matches!(index.as_ref(), Expression::Literal(ScalarValue::Integer(0)))
        )
    {
        return None;
    }

    let output_name = final_item.column_name(0);
    let mut rewritten = plan.clone();
    let PhysicalOperator::Project { projection, .. } = &mut rewritten.operators[*list_index] else {
        unreachable!("the exact relationship-list projection changed shape")
    };
    projection.items[0] = ProjectionItem {
        expression: Expression::Variable(relationship.clone()),
        alias: Some(list_name.clone()),
        source_text: None,
    };
    let PhysicalOperator::Project { projection, .. } = &mut rewritten.operators[*final_index]
    else {
        unreachable!("the exact relationship-list type projection changed shape")
    };
    projection.items[0] = ProjectionItem {
        expression: Expression::Function {
            name: name.clone(),
            distinct: false,
            arguments: vec![Expression::Variable(list_name.clone())],
        },
        alias: Some(output_name),
        source_text: None,
    };
    Some(rewritten)
}

/// The nullable-relation predicate ABI already owns exact node-label membership, string
/// comparison, and three-valued Boolean composition. This rewrite recognizes only List12 [6]'s
/// complete catalog-bounded statement and spells its list membership in those existing terms:
///
/// `n.name IN [x IN labels(b) | toLower(x)]`
///
/// becomes one balanced disjunction of
///
/// `b:<label> AND n.name = '<lowercase label>'`
///
/// for every label in the fenced catalog generation. Nodes cannot carry a label outside that
/// generation, and the request separately seals the catalog hash, so this is a complete list
/// domain rather than a host-selected sample. In particular, an unlabeled `b` makes every label
/// leaf false, while a null/missing `n.name` remains null whenever `b` has a catalog label.
fn exact_catalog_label_comprehension_membership_plan(
    plan: &PhysicalPlan,
    catalog: &crate::graph::NameCatalog,
) -> Option<PhysicalPlan> {
    let operators = plan
        .operators
        .iter()
        .enumerate()
        .filter(|(_, operator)| !matches!(operator, PhysicalOperator::CardinalityCheckpoint { .. }))
        .collect::<Vec<_>>();
    let [
        (
            _,
            PhysicalOperator::ScanPattern {
                optional: false,
                pattern,
                access,
                ..
            },
        ),
        (filter_index, PhysicalOperator::Filter(expression)),
        (
            _,
            PhysicalOperator::Project {
                keep_scope: false,
                projection: final_projection,
            },
        ),
    ] = operators.as_slice()
    else {
        return None;
    };
    if !matches!(
        access,
        ScanAccessPath::Unspecified | ScanAccessPath::AllNodes
    ) || pattern.variable.is_some()
        || pattern.selector != PathSelector::All
        || pattern.mode != PathMode::DifferentRelationships
        || pattern.start.property_predicate_present
        || !pattern.start.labels.is_empty()
        || !pattern.start.properties.is_empty()
    {
        return None;
    }
    let [step] = pattern.steps.as_slice() else {
        return None;
    };
    if step.relationship.variable.is_some()
        || !step.relationship.types.is_empty()
        || step.relationship.direction != Direction::Outgoing
        || step.relationship.variable_length
        || step.relationship.min_hops.is_some()
        || step.relationship.max_hops.is_some()
        || !step.relationship.properties.is_empty()
        || step.node.property_predicate_present
        || !step.node.labels.is_empty()
        || !step.node.properties.is_empty()
    {
        return None;
    }
    let (Some(source_variable), Some(target_variable)) =
        (pattern.start.variable.as_ref(), step.node.variable.as_ref())
    else {
        return None;
    };
    if source_variable == target_variable {
        return None;
    }

    let [final_item] = final_projection.items.as_slice() else {
        return None;
    };
    if final_projection.distinct
        || final_item.alias.is_some()
        || !matches!(
            &final_item.expression,
            Expression::Variable(variable) if variable == target_variable
        )
    {
        return None;
    }

    let Expression::Binary {
        left,
        operation: BinaryOperator::In,
        right,
    } = expression
    else {
        return None;
    };
    let Expression::Property(property_source, property_name) = left.as_ref() else {
        return None;
    };
    if property_name != "name"
        || !matches!(
            property_source.as_ref(),
            Expression::Variable(variable) if variable == source_variable
        )
    {
        return None;
    }
    let Expression::ListComprehension {
        variable: element_variable,
        list,
        predicate: None,
        projection: Some(element_projection),
    } = right.as_ref()
    else {
        return None;
    };
    if element_variable == source_variable || element_variable == target_variable {
        return None;
    }
    let Expression::Function {
        name: labels_function,
        distinct: false,
        arguments: labels_arguments,
    } = list.as_ref()
    else {
        return None;
    };
    if !matches!(labels_function.as_slice(), [name] if name.eq_ignore_ascii_case("labels"))
        || !matches!(
            labels_arguments.as_slice(),
            [Expression::Variable(variable)] if variable == target_variable
        )
    {
        return None;
    }
    let Expression::Function {
        name: lower_function,
        distinct: false,
        arguments: lower_arguments,
    } = element_projection.as_ref()
    else {
        return None;
    };
    if !matches!(lower_function.as_slice(), [name] if name.eq_ignore_ascii_case("toLower"))
        || !matches!(
            lower_arguments.as_slice(),
            [Expression::Variable(variable)] if variable == element_variable
        )
    {
        return None;
    }

    let labels = catalog.labels().collect::<Vec<_>>();
    let label_count = labels.len();
    let (predicate_nodes, predicate_instructions, predicate_depth) = if label_count == 0 {
        (1_usize, 1_usize, 1_usize)
    } else {
        let predicate_nodes = label_count.checked_mul(4)?.checked_sub(1)?;
        // HasLabels is one Metal instruction, CompareString is two value loads plus comparison,
        // and each AND/OR is one instruction: 5N + (N - 1) = 6N - 1.
        let predicate_instructions = label_count.checked_mul(6)?.checked_sub(1)?;
        let or_depth = usize::BITS as usize - (label_count - 1).leading_zeros() as usize;
        (
            predicate_nodes,
            predicate_instructions,
            or_depth.checked_add(2)?,
        )
    };
    // The tree validator owns the first and third limits. Keep the encoded instruction image
    // under that same compact bound as an additional compiler proof; the encoder need not infer
    // a larger hidden capacity from a shallower tree.
    if predicate_nodes > RESIDENT_NULLABLE_RELATION_MAX_PREDICATE_NODES
        || predicate_instructions > RESIDENT_NULLABLE_RELATION_MAX_PREDICATE_NODES
        || predicate_depth > RESIDENT_NULLABLE_RELATION_MAX_PREDICATE_DEPTH
    {
        return None;
    }

    let mut literal_bytes = 0_usize;
    let mut leaves = Vec::with_capacity(label_count);
    for (_, label_name) in labels {
        let lowered = label_name.to_lowercase();
        literal_bytes = literal_bytes.checked_add(lowered.len())?;
        if lowered.len() > RESIDENT_NULLABLE_RELATION_MAX_LITERAL_STRING_BYTES
            || literal_bytes > RESIDENT_NULLABLE_RELATION_MAX_LITERAL_STRING_BYTES
        {
            return None;
        }
        leaves.push(Expression::Binary {
            left: Box::new(Expression::entity_label_predicate(
                Expression::Variable(target_variable.clone()),
                vec![label_name.to_owned()],
            )),
            operation: BinaryOperator::And,
            right: Box::new(Expression::Binary {
                left: left.clone(),
                operation: BinaryOperator::Equal,
                right: Box::new(Expression::string(lowered)),
            }),
        });
    }
    let lowered = if leaves.is_empty() {
        Expression::Literal(ScalarValue::Boolean(false))
    } else {
        while leaves.len() > 1 {
            let mut next = Vec::with_capacity(leaves.len().div_ceil(2));
            let mut operands = leaves.into_iter();
            while let Some(left) = operands.next() {
                next.push(match operands.next() {
                    Some(right) => Expression::Binary {
                        left: Box::new(left),
                        operation: BinaryOperator::Or,
                        right: Box::new(right),
                    },
                    None => left,
                });
            }
            leaves = next;
        }
        leaves.pop()?
    };

    let mut rewritten = plan.clone();
    rewritten.operators[*filter_index] = PhysicalOperator::Filter(lowered);
    Some(rewritten)
}

fn exact_relationship_type_operand(expression: &Expression, relationship: &str) -> bool {
    matches!(
        expression,
        Expression::Function {
            name,
            distinct: false,
            arguments,
        } if matches!(name.as_slice(), [name] if name.eq_ignore_ascii_case("type"))
            && matches!(arguments.as_slice(), [Expression::Variable(variable)] if variable == relationship)
    )
}

fn exact_relationship_type_equality(expression: &Expression, relationship: &str) -> Option<String> {
    let Expression::Binary {
        left,
        operation: BinaryOperator::Equal,
        right,
    } = expression
    else {
        return None;
    };
    match (left.as_ref(), right.as_ref()) {
        (typed, Expression::Literal(ScalarValue::String(name)))
            if exact_relationship_type_operand(typed, relationship) =>
        {
            Some(name.to_string())
        }
        (Expression::Literal(ScalarValue::String(name)), typed)
            if exact_relationship_type_operand(typed, relationship) =>
        {
            Some(name.to_string())
        }
        _ => None,
    }
}

fn exact_relationship_type_disjunction(
    expression: &Expression,
    relationship: &str,
    names: &mut Vec<String>,
) -> bool {
    if let Some(name) = exact_relationship_type_equality(expression, relationship) {
        names.push(name);
        return true;
    }
    let Expression::Binary {
        left,
        operation: BinaryOperator::Or,
        right,
    } = expression
    else {
        return false;
    };
    exact_relationship_type_disjunction(left, relationship, names)
        && exact_relationship_type_disjunction(right, relationship, names)
}

/// Narrows only the exact one-hop `type(r) = '<literal>'` TCK predicates to the relationship
/// domain already owned by the staged nullable-relation expansion. A one- or two-leaf OR is a
/// union of exact relationship types. Every conjunction, parameterized/dynamic name, pre-typed
/// relationship, wider path, OPTIONAL source, or non-terminal plan remains outside this adapter.
fn exact_relationship_type_filter_plan(plan: &PhysicalPlan) -> Option<PhysicalPlan> {
    let operators = plan
        .operators
        .iter()
        .enumerate()
        .filter(|(_, operator)| !matches!(operator, PhysicalOperator::CardinalityCheckpoint { .. }))
        .collect::<Vec<_>>();
    let [
        (
            scan_index,
            PhysicalOperator::ScanPattern {
                optional: false,
                pattern,
                ..
            },
        ),
        (filter_index, PhysicalOperator::Filter(expression)),
        (
            _,
            PhysicalOperator::Project {
                keep_scope: false,
                projection,
            },
        ),
    ] = operators.as_slice()
    else {
        return None;
    };
    let [step] = pattern.steps.as_slice() else {
        return None;
    };
    let relationship = step.relationship.variable.as_deref()?;
    if relationship.is_empty()
        || pattern.variable.is_some()
        || pattern.selector != PathSelector::All
        || pattern.mode != PathMode::DifferentRelationships
        || !step.relationship.types.is_empty()
        || step.relationship.variable_length
        || step.relationship.min_hops.is_some()
        || step.relationship.max_hops.is_some()
        || projection.distinct
    {
        return None;
    }
    let [output] = projection.items.as_slice() else {
        return None;
    };
    if output.alias.is_some()
        || !matches!(
            &output.expression,
            Expression::Variable(variable)
                if Some(variable) == pattern.start.variable.as_ref()
                    || variable == relationship
                    || Some(variable) == step.node.variable.as_ref()
        )
    {
        return None;
    }

    let mut names = Vec::with_capacity(2);
    if !exact_relationship_type_disjunction(expression, relationship, &mut names)
        || names.is_empty()
        || names.len() > 2
        || names.iter().any(String::is_empty)
    {
        return None;
    }
    names.sort();
    names.dedup();

    let mut rewritten = plan.clone();
    let PhysicalOperator::ScanPattern { pattern, .. } = &mut rewritten.operators[*scan_index]
    else {
        unreachable!("the exact relationship-type source changed shape")
    };
    pattern.steps[0].relationship.types = names;
    rewritten.operators.remove(*filter_index);
    Some(rewritten)
}

fn exact_scalar_alias_filter_operand(
    expression: &Expression,
    source: &str,
    property: &str,
    alias: &str,
) -> Option<(Expression, bool, bool)> {
    match expression {
        Expression::Variable(variable) if variable == alias => Some((
            Expression::Property(
                Box::new(Expression::Variable(source.to_owned())),
                property.to_owned(),
            ),
            true,
            true,
        )),
        Expression::Property(property_source, candidate)
            if candidate == property
                && matches!(property_source.as_ref(), Expression::Variable(variable) if variable == source) =>
        {
            Some((expression.clone(), true, false))
        }
        Expression::Literal(ScalarValue::String(_)) => Some((expression.clone(), false, false)),
        _ => None,
    }
}

fn exact_scalar_alias_filter(
    expression: &Expression,
    source: &str,
    property: &str,
    alias: &str,
) -> Option<(Expression, usize, bool)> {
    if let Expression::Binary {
        left,
        operation: BinaryOperator::Or,
        right,
    } = expression
    {
        let (left, left_leaves, left_alias) =
            exact_scalar_alias_filter(left, source, property, alias)?;
        let (right, right_leaves, right_alias) =
            exact_scalar_alias_filter(right, source, property, alias)?;
        return Some((
            Expression::Binary {
                left: Box::new(left),
                operation: BinaryOperator::Or,
                right: Box::new(right),
            },
            left_leaves.checked_add(right_leaves)?,
            left_alias || right_alias,
        ));
    }
    let Expression::Binary {
        left,
        operation: BinaryOperator::Equal,
        right,
    } = expression
    else {
        return None;
    };
    let (left, left_is_property, left_alias) =
        exact_scalar_alias_filter_operand(left, source, property, alias)?;
    let (right, right_is_property, right_alias) =
        exact_scalar_alias_filter_operand(right, source, property, alias)?;
    if left_is_property == right_is_property {
        return None;
    }
    Some((
        Expression::Binary {
            left: Box::new(left),
            operation: BinaryOperator::Equal,
            right: Box::new(right),
        },
        1,
        left_alias || right_alias,
    ))
}

/// Deforests the exact scalar alias boundary used by WithWhere7 [2]/[3]. The immutable property
/// expression is substituted only inside one or two string-equality leaves, and `RETURN *` is
/// made explicit as that same property under its alias. This preserves the WITH-visible schema
/// without extending the entity-only scope-projection ABI.
fn exact_scalar_with_where_alias_plan(plan: &PhysicalPlan) -> Option<PhysicalPlan> {
    let operators = plan
        .operators
        .iter()
        .enumerate()
        .filter(|(_, operator)| !matches!(operator, PhysicalOperator::CardinalityCheckpoint { .. }))
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
            materialize_index,
            PhysicalOperator::Project {
                keep_scope: true,
                projection: materialized,
            },
        ),
        (filter_index, PhysicalOperator::Filter(filter)),
        (
            visible_index,
            PhysicalOperator::Project {
                keep_scope: false,
                projection: visible,
            },
        ),
        tail @ ..,
    ] = operators.as_slice()
    else {
        return None;
    };
    let final_index = match tail {
        [] => None,
        [
            (
                final_index,
                PhysicalOperator::Project {
                    keep_scope: false,
                    projection: final_projection,
                },
            ),
        ] if !final_projection.distinct
            && matches!(
                final_projection.items.as_slice(),
                [item] if item.alias.is_none() && matches!(&item.expression, Expression::Star)
            ) =>
        {
            Some(*final_index)
        }
        _ => return None,
    };
    if pattern.variable.is_some()
        || pattern.selector != PathSelector::All
        || pattern.mode != PathMode::DifferentRelationships
        || !pattern.steps.is_empty()
        || pattern.start.property_predicate_present
        || !pattern.start.labels.is_empty()
        || !pattern.start.properties.is_empty()
        || materialized.distinct
        || visible.distinct
    {
        return None;
    }
    let source = pattern.start.variable.as_deref()?;
    let [materialized_item] = materialized.items.as_slice() else {
        return None;
    };
    let alias = materialized_item.alias.as_deref()?;
    let Expression::Property(property_source, property) = &materialized_item.expression else {
        return None;
    };
    if source.is_empty()
        || alias.is_empty()
        || source == alias
        || property.is_empty()
        || !matches!(property_source.as_ref(), Expression::Variable(variable) if variable == source)
    {
        return None;
    }
    let [visible_item] = visible.items.as_slice() else {
        return None;
    };
    if visible_item.alias.as_deref() != Some(alias)
        || visible_item.source_text.is_some()
        || !matches!(&visible_item.expression, Expression::Variable(variable) if variable == alias)
    {
        return None;
    }
    let (filter, leaves, saw_alias) = exact_scalar_alias_filter(filter, source, property, alias)?;
    if !(1..=2).contains(&leaves) || !saw_alias {
        return None;
    }

    let output = ProjectionItem {
        expression: materialized_item.expression.clone(),
        alias: Some(alias.to_owned()),
        source_text: None,
    };
    let mut rewritten = plan.clone();
    rewritten.operators[*materialize_index] = PhysicalOperator::Filter(filter);
    rewritten.operators[*filter_index] = PhysicalOperator::Project {
        keep_scope: false,
        projection: Projection {
            distinct: false,
            items: vec![output],
        },
    };
    if let Some(final_index) = final_index {
        rewritten.operators.remove(final_index);
    }
    rewritten.operators.remove(*visible_index);
    Some(rewritten)
}

/// `WITH DISTINCT a.p AS alias WHERE a.p = '<literal>' RETURN *` can produce only the one
/// non-null literal selected by its filter. Preserve the source entity through one bounded scope
/// projection, cap that identical-value relation at one row, and publish the property under the
/// original alias. This is deliberately limited to the exact WithWhere1 [2] shape.
fn exact_filtered_distinct_scalar_alias_plan(plan: &PhysicalPlan) -> Option<PhysicalPlan> {
    let operators = plan
        .operators
        .iter()
        .enumerate()
        .filter(|(_, operator)| !matches!(operator, PhysicalOperator::CardinalityCheckpoint { .. }))
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
            materialize_index,
            PhysicalOperator::Project {
                keep_scope: true,
                projection: materialized,
            },
        ),
        (filter_index, PhysicalOperator::Filter(filter)),
        (
            visible_index,
            PhysicalOperator::Project {
                keep_scope: false,
                projection: visible,
            },
        ),
        tail @ ..,
    ] = operators.as_slice()
    else {
        return None;
    };
    let final_index = match tail {
        [] => None,
        [
            (
                final_index,
                PhysicalOperator::Project {
                    keep_scope: false,
                    projection: final_projection,
                },
            ),
        ] if !final_projection.distinct
            && matches!(
                final_projection.items.as_slice(),
                [item] if item.alias.is_none() && matches!(&item.expression, Expression::Star)
            ) =>
        {
            Some(*final_index)
        }
        _ => return None,
    };
    if pattern.variable.is_some()
        || pattern.selector != PathSelector::All
        || pattern.mode != PathMode::DifferentRelationships
        || !pattern.steps.is_empty()
        || pattern.start.property_predicate_present
        || !pattern.start.labels.is_empty()
        || !pattern.start.properties.is_empty()
        || materialized.distinct
        || !visible.distinct
    {
        return None;
    }
    let source = pattern.start.variable.as_deref()?;
    let [materialized_item] = materialized.items.as_slice() else {
        return None;
    };
    let alias = materialized_item.alias.as_deref()?;
    let Expression::Property(property_source, property) = &materialized_item.expression else {
        return None;
    };
    if source.is_empty()
        || alias.is_empty()
        || source == alias
        || property.is_empty()
        || !matches!(property_source.as_ref(), Expression::Variable(variable) if variable == source)
    {
        return None;
    }
    let [visible_item] = visible.items.as_slice() else {
        return None;
    };
    if visible_item.alias.as_deref() != Some(alias)
        || visible_item.source_text.is_some()
        || !matches!(&visible_item.expression, Expression::Variable(variable) if variable == alias)
    {
        return None;
    }
    let (filter, leaves, saw_alias) = exact_scalar_alias_filter(filter, source, property, alias)?;
    if leaves != 1 || saw_alias {
        return None;
    }

    let mut rewritten = plan.clone();
    rewritten.operators[*materialize_index] = PhysicalOperator::Filter(filter);
    rewritten.operators[*filter_index] = PhysicalOperator::Project {
        keep_scope: false,
        projection: Projection {
            distinct: false,
            items: vec![ProjectionItem {
                expression: Expression::Variable(source.to_owned()),
                alias: Some(source.to_owned()),
                source_text: None,
            }],
        },
    };
    rewritten.operators[*visible_index] =
        PhysicalOperator::Limit(Expression::Literal(ScalarValue::Integer(1)));
    if let Some(final_index) = final_index {
        rewritten.operators.remove(final_index);
    }
    rewritten.operators.insert(
        visible_index.saturating_add(1),
        PhysicalOperator::Project {
            keep_scope: false,
            projection: Projection {
                distinct: false,
                items: vec![ProjectionItem {
                    expression: materialized_item.expression.clone(),
                    alias: Some(alias.to_owned()),
                    source_text: None,
                }],
            },
        },
    );
    Some(rewritten)
}

fn exact_forwarded_property_join_filter(
    expression: &Expression,
    source: &str,
    source_property: &str,
    alias: &str,
    target: &str,
) -> Option<(Expression, String)> {
    let Expression::Binary {
        left,
        operation: BinaryOperator::Equal,
        right,
    } = expression
    else {
        return None;
    };
    let forwarded = Expression::Property(
        Box::new(Expression::Variable(source.to_owned())),
        source_property.to_owned(),
    );
    match (left.as_ref(), right.as_ref()) {
        (Expression::Variable(variable), Expression::Property(property_source, property))
            if variable == alias
                && matches!(property_source.as_ref(), Expression::Variable(variable) if variable == target) =>
        {
            Some((
                Expression::Binary {
                    left: Box::new(forwarded),
                    operation: BinaryOperator::Equal,
                    right: right.clone(),
                },
                property.clone(),
            ))
        }
        (Expression::Property(property_source, property), Expression::Variable(variable))
            if variable == alias
                && matches!(property_source.as_ref(), Expression::Variable(variable) if variable == target) =>
        {
            Some((
                Expression::Binary {
                    left: left.clone(),
                    operation: BinaryOperator::Equal,
                    right: Box::new(forwarded),
                },
                property.clone(),
            ))
        }
        _ => None,
    }
}

/// Replaces one scalar-only WITH boundary by the source entity that owns that immutable property,
/// then substitutes the property into the one downstream equality. Cardinality and row order are
/// unchanged, so the optional exact LIMIT 1 remains attached to the existing entity ScopeProject.
/// The accepted domains are precisely With2 [1], With4 [2], and WithSkipLimit2 [2].
fn exact_forwarded_property_join_plan(plan: &PhysicalPlan) -> Option<PhysicalPlan> {
    let operators = plan
        .operators
        .iter()
        .enumerate()
        .filter(|(_, operator)| !matches!(operator, PhysicalOperator::CardinalityCheckpoint { .. }))
        .collect::<Vec<_>>();
    if !matches!(operators.len(), 5 | 6) {
        return None;
    }
    let (
        _,
        PhysicalOperator::ScanPattern {
            match_group: source_group,
            optional: false,
            pattern: source_pattern,
            ..
        },
    ) = operators[0]
    else {
        return None;
    };
    let (
        projection_index,
        PhysicalOperator::Project {
            keep_scope: false,
            projection,
        },
    ) = operators[1]
    else {
        return None;
    };
    let has_limit = operators.len() == 6;
    let target_position = if has_limit {
        if !matches!(
            operators[2].1,
            PhysicalOperator::Limit(Expression::Literal(ScalarValue::Integer(1)))
        ) {
            return None;
        }
        3
    } else {
        2
    };
    let (
        _,
        PhysicalOperator::ScanPattern {
            match_group: target_group,
            optional: false,
            pattern: target_pattern,
            ..
        },
    ) = operators[target_position]
    else {
        return None;
    };
    let (filter_index, PhysicalOperator::Filter(filter)) = operators[target_position + 1] else {
        return None;
    };
    let (
        _,
        PhysicalOperator::Project {
            keep_scope: false,
            projection: output,
        },
    ) = operators[target_position + 2]
    else {
        return None;
    };
    if source_group == target_group
        || source_pattern.variable.is_some()
        || source_pattern.selector != PathSelector::All
        || source_pattern.mode != PathMode::DifferentRelationships
        || !source_pattern.steps.is_empty()
        || source_pattern.start.property_predicate_present
        || source_pattern.start.labels.as_slice() != ["Begin"]
        || !source_pattern.start.properties.is_empty()
        || target_pattern.variable.is_some()
        || target_pattern.selector != PathSelector::All
        || target_pattern.mode != PathMode::DifferentRelationships
        || !target_pattern.steps.is_empty()
        || target_pattern.start.property_predicate_present
        || !target_pattern.start.properties.is_empty()
        || projection.distinct
        || output.distinct
    {
        return None;
    }
    let source = source_pattern.start.variable.as_deref()?;
    let target = target_pattern.start.variable.as_deref()?;
    if source.is_empty() || target.is_empty() || source == target {
        return None;
    }
    let [projection_item] = projection.items.as_slice() else {
        return None;
    };
    let alias = projection_item.alias.as_deref()?;
    let Expression::Property(property_source, source_property) = &projection_item.expression else {
        return None;
    };
    if alias.is_empty()
        || alias == source
        || alias == target
        || source_property != "num"
        || !matches!(property_source.as_ref(), Expression::Variable(variable) if variable == source)
    {
        return None;
    }
    let (filter, target_property) =
        exact_forwarded_property_join_filter(filter, source, source_property, alias, target)?;
    let exact_domain = if has_limit {
        target_pattern.start.labels.is_empty() && target_property == "id"
    } else {
        (target_pattern.start.labels.is_empty() && target_property == "id")
            || (target_pattern.start.labels.as_slice() == ["End"]
                && target_property == source_property.as_str())
    };
    if !exact_domain {
        return None;
    }
    let [output_item] = output.items.as_slice() else {
        return None;
    };
    if output_item.alias.is_some()
        || !matches!(&output_item.expression, Expression::Variable(variable) if variable == target)
    {
        return None;
    }

    let mut rewritten = plan.clone();
    rewritten.operators[projection_index] = PhysicalOperator::Project {
        keep_scope: false,
        projection: Projection {
            distinct: false,
            items: vec![ProjectionItem {
                expression: Expression::Variable(source.to_owned()),
                alias: Some(source.to_owned()),
                source_text: None,
            }],
        },
    };
    rewritten.operators[filter_index] = PhysicalOperator::Filter(filter);
    Some(rewritten)
}

fn exact_comparison1_to_integer_id_filter(
    expression: &Expression,
    node: &str,
    expected: &str,
) -> bool {
    let Expression::Binary {
        left,
        operation: BinaryOperator::Equal,
        right,
    } = expression
    else {
        return false;
    };
    let is_conversion = |expression: &Expression| {
        matches!(
            expression,
            Expression::Function {
                name,
                distinct: false,
                arguments,
            } if matches!(name.as_slice(), [name] if name.eq_ignore_ascii_case("toInteger"))
                && matches!(
                    arguments.as_slice(),
                    [Expression::Property(source, property)]
                        if property == "id"
                            && matches!(source.as_ref(), Expression::Variable(variable) if variable == node)
                )
        )
    };
    (is_conversion(left)
        && matches!(right.as_ref(), Expression::Variable(variable) if variable == expected))
        || (is_conversion(right)
            && matches!(left.as_ref(), Expression::Variable(variable) if variable == expected))
}

/// Deforests only the three heterogeneous literal-list seeds in Comparison1 [1]-[3]. The first
/// collected row is immutable and its zero index is therefore known before graph execution. An
/// INTEGER-only canonical `id` column makes `toInteger(n.id)` the identity; the two cross-type
/// expected values can never equal that INTEGER domain. The complete remaining scan/filter/entity
/// projection is still compiled by the nullable-relation backend.
fn exact_comparison1_collected_index_plan(
    plan: &PhysicalPlan,
    catalog: &crate::graph::NameCatalog,
    graph: &GraphStore,
) -> Option<PhysicalPlan> {
    let operators = plan
        .operators
        .iter()
        .enumerate()
        .filter(|(_, operator)| !matches!(operator, PhysicalOperator::CardinalityCheckpoint { .. }))
        .collect::<Vec<_>>();
    let [
        (
            _,
            PhysicalOperator::Project {
                keep_scope: false,
                projection: collected,
            },
        ),
        (
            _,
            PhysicalOperator::Unwind {
                expression: unwind_expression,
                variable: unwind_variable,
            },
        ),
        (
            _,
            PhysicalOperator::Project {
                keep_scope: false,
                projection: indexed,
            },
        ),
        (
            scan_index,
            scan @ PhysicalOperator::ScanPattern {
                optional: false,
                pattern,
                access,
                ..
            },
        ),
        (filter_index, PhysicalOperator::Filter(filter)),
        (
            final_index,
            final_projection @ PhysicalOperator::Project {
                keep_scope: false,
                projection: output,
            },
        ),
    ] = operators.as_slice()
    else {
        return None;
    };
    if collected.distinct || indexed.distinct || output.distinct {
        return None;
    }
    let [collected_item] = collected.items.as_slice() else {
        return None;
    };
    let collected_name = collected_item.alias.as_deref()?;
    let Expression::Function {
        name,
        distinct: false,
        arguments,
    } = &collected_item.expression
    else {
        return None;
    };
    let [Expression::List(list)] = arguments.as_slice() else {
        return None;
    };
    if collected_name.is_empty()
        || !matches!(name.as_slice(), [name] if name.eq_ignore_ascii_case("collect"))
        || !matches!(unwind_expression, Expression::Variable(variable) if variable == collected_name)
        || unwind_variable.is_empty()
        || unwind_variable == collected_name
    {
        return None;
    }
    let [indexed_item] = indexed.items.as_slice() else {
        return None;
    };
    let expected = indexed_item.alias.as_deref()?;
    if expected.is_empty()
        || expected == collected_name
        || expected == unwind_variable
        || !matches!(
            &indexed_item.expression,
            Expression::Index { expression, index }
                if matches!(expression.as_ref(), Expression::Variable(variable) if variable == unwind_variable)
                    && matches!(index.as_ref(), Expression::Literal(ScalarValue::Integer(0)))
        )
    {
        return None;
    }
    if pattern.variable.is_some()
        || pattern.selector != PathSelector::All
        || pattern.mode != PathMode::DifferentRelationships
        || pattern.start.property_predicate_present
        || !pattern.start.labels.is_empty()
        || !pattern.start.properties.is_empty()
        || !pattern.steps.is_empty()
        || !matches!(
            access,
            ScanAccessPath::Unspecified | ScanAccessPath::AllNodes
        )
    {
        return None;
    }
    let node = pattern.start.variable.as_deref()?;
    if node.is_empty()
        || node == collected_name
        || node == unwind_variable
        || node == expected
        || !exact_comparison1_to_integer_id_filter(filter, node, expected)
    {
        return None;
    }
    let [output_item] = output.items.as_slice() else {
        return None;
    };
    if output_item.alias.is_some()
        || output_item
            .source_text
            .as_deref()
            .is_some_and(|source_text| source_text != node)
        || !matches!(&output_item.expression, Expression::Variable(variable) if variable == node)
    {
        return None;
    }
    let property = catalog.property("id")?;
    if !graph.node_property_is_integer(property) {
        return None;
    }
    let rewritten_filter = match list.as_slice() {
        [
            Expression::Literal(ScalarValue::Integer(0)),
            Expression::Literal(ScalarValue::Float(value)),
        ] if value.0.to_bits() == 0.0_f64.to_bits() => Expression::Binary {
            left: Box::new(Expression::Property(
                Box::new(Expression::Variable(node.to_owned())),
                "id".to_owned(),
            )),
            operation: BinaryOperator::Equal,
            right: Box::new(Expression::Literal(ScalarValue::Integer(0))),
        },
        [
            Expression::Literal(ScalarValue::Float(value)),
            Expression::Literal(ScalarValue::Integer(0)),
        ] if value.0.to_bits() == 0.5_f64.to_bits() => {
            Expression::Literal(ScalarValue::Boolean(false))
        }
        [
            Expression::Literal(ScalarValue::String(value)),
            Expression::Literal(ScalarValue::Integer(0)),
        ] if value.as_ref() == "0" => Expression::Literal(ScalarValue::Boolean(false)),
        _ => return None,
    };

    let mut rewritten = plan.clone();
    rewritten.operators = vec![
        (*scan).clone(),
        PhysicalOperator::Filter(rewritten_filter),
        (*final_projection).clone(),
    ];
    debug_assert!(*scan_index < *filter_index && *filter_index < *final_index);
    Some(rewritten)
}

/// Compile a complete fixed-length nullable entity relation without wiring it into execution.
/// Returning `Some` proves that every physical operator was lowered; callers must still fail
/// closed until their selected backend implements `ResidentNullableRelationRequest` as one
/// generation-fenced command.
#[allow(dead_code)] // Kept closed until CPU and Metal implement the complete staged command.
pub fn compile_nullable_relation(
    plan: &PhysicalPlan,
    project: ProjectId,
    bookmark: Bookmark,
    catalog: &crate::graph::NameCatalog,
    graph: &GraphStore,
    max_output_rows: usize,
) -> Result<Option<CompiledResidentNullableRelationPlan>> {
    compile_nullable_relation_with_parameters(
        plan,
        project,
        bookmark,
        catalog,
        graph,
        &BTreeMap::new(),
        max_output_rows,
    )
}

pub fn compile_nullable_relation_with_parameters(
    plan: &PhysicalPlan,
    project: ProjectId,
    bookmark: Bookmark,
    catalog: &crate::graph::NameCatalog,
    graph: &GraphStore,
    parameters: &BTreeMap<String, ResultValue>,
    max_output_rows: usize,
) -> Result<Option<CompiledResidentNullableRelationPlan>> {
    let union_rewrite = canonicalize_exact_two_label_union(plan, catalog, graph);
    let plan = union_rewrite.as_ref().unwrap_or(plan);
    if !plan.read_only
        || plan.at_time.is_some()
        || !plan.unions.is_empty()
        || plan.operators.is_empty()
    {
        return Ok(None);
    }

    let collected_index_rewrite = exact_comparison1_collected_index_plan(plan, catalog, graph);
    let plan = collected_index_rewrite.as_ref().unwrap_or(plan);
    let scalar_alias_rewrite = exact_scalar_with_where_alias_plan(plan);
    let plan = scalar_alias_rewrite.as_ref().unwrap_or(plan);
    let distinct_scalar_alias_rewrite = exact_filtered_distinct_scalar_alias_plan(plan);
    let plan = distinct_scalar_alias_rewrite.as_ref().unwrap_or(plan);
    let forwarded_property_rewrite = exact_forwarded_property_join_plan(plan);
    let plan = forwarded_property_rewrite.as_ref().unwrap_or(plan);
    let relationship_type_rewrite = exact_relationship_type_filter_plan(plan);
    let plan = relationship_type_rewrite.as_ref().unwrap_or(plan);
    let label_membership_rewrite = exact_catalog_label_comprehension_membership_plan(plan, catalog);
    let plan = label_membership_rewrite.as_ref().unwrap_or(plan);
    let relationship_list_rewrite = exact_relationship_list_zero_type_plan(plan);
    let plan = relationship_list_rewrite.as_ref().unwrap_or(plan);

    let mut builder = NullableRelationBuilder::new(catalog, graph, parameters);
    let relationship_counts = plan.operators.iter().fold(
        BTreeMap::<MatchGroupId, usize>::new(),
        |mut counts, operator| {
            if let PhysicalOperator::ScanPattern {
                match_group,
                pattern,
                ..
            } = operator
            {
                *counts.entry(*match_group).or_default() += pattern.steps.len();
            }
            counts
        },
    );
    let mut cursor = 0_usize;
    let mut graph_stages = 0_usize;
    let mut outputs = None;
    while cursor < plan.operators.len() {
        match &plan.operators[cursor] {
            PhysicalOperator::CardinalityCheckpoint { .. } => {
                cursor += 1;
            }
            PhysicalOperator::ScanPattern {
                match_group,
                optional,
                pattern,
                access,
            } => {
                let first_stage = builder.stages.len();
                if outputs.is_some()
                    || !builder.scan_pattern(
                        *match_group,
                        relationship_counts.get(match_group).copied().unwrap_or(0) > 1,
                        *optional,
                        pattern,
                        access,
                    )?
                {
                    return Ok(None);
                }
                let Some(last_stage) = builder.stages.len().checked_sub(1) else {
                    return Err(Error::internal(
                        "resident nullable scan compiler emitted no graph stage",
                    ));
                };
                graph_stages = graph_stages.saturating_add(1);
                cursor += 1;
                if *optional {
                    while let Some(PhysicalOperator::Filter(expression)) =
                        plan.operators.get(cursor)
                    {
                        if !builder.optional_candidate_filter(
                            first_stage,
                            last_stage,
                            expression,
                        )? {
                            return Ok(None);
                        }
                        cursor += 1;
                    }
                }
            }
            PhysicalOperator::Filter(expression) => {
                if outputs.is_some() || !builder.relation_filter(expression)? {
                    return Ok(None);
                }
                cursor += 1;
            }
            PhysicalOperator::Project {
                keep_scope,
                projection,
            } => {
                let unwind_index = plan.operators[cursor + 1..]
                    .iter()
                    .position(|operator| {
                        !matches!(operator, PhysicalOperator::CardinalityCheckpoint { .. })
                    })
                    .map(|offset| cursor + 1 + offset);
                if outputs.is_none()
                    && let Some(unwind_index) = unwind_index
                    && let PhysicalOperator::Unwind {
                        expression,
                        variable,
                    } = &plan.operators[unwind_index]
                    && builder.project_collect_unwind(
                        *keep_scope,
                        projection,
                        expression,
                        variable,
                    )?
                {
                    cursor = unwind_index + 1;
                    continue;
                }
                if projection.distinct || outputs.is_some() {
                    return Ok(None);
                }
                if *keep_scope {
                    if !builder.keep_scope_projection(projection)? {
                        return Ok(None);
                    }
                    cursor += 1;
                    continue;
                }
                let terminal = plan.operators[cursor + 1..].iter().all(|operator| {
                    matches!(operator, PhysicalOperator::CardinalityCheckpoint { .. })
                });
                if terminal {
                    let Some(compiled_outputs) = builder.final_projection(projection)? else {
                        return Ok(None);
                    };
                    outputs = Some(compiled_outputs);
                } else if !builder.static_null_node_scope(*keep_scope, projection)?
                    && !builder.scope_projection(projection)?
                {
                    return Ok(None);
                }
                cursor += 1;
            }
            PhysicalOperator::Limit(expression) => {
                if outputs.is_some() || !builder.stable_scope_limit(expression)? {
                    return Ok(None);
                }
                cursor += 1;
            }
            _ => return Ok(None),
        }
    }
    let Some(outputs) = outputs else {
        return Ok(None);
    };
    if graph_stages == 0 {
        return Ok(None);
    }
    if builder.static_null_node_seeded && builder.statically_null_paths.is_empty() {
        // The literal-null node adapter exists solely to prove the exact named OPTIONAL path
        // erasure below. Never let it become a general scalar-row source.
        return Ok(None);
    }

    let program = ResidentNullableRelationProgram {
        layers: plan.read_layers,
        stages: builder.stages,
    };
    let predicate_program = builder.predicate_program;
    let visible_node_rows = graph
        .nodes()
        .filter(|node| plan.read_layers.contains_layer(node.layer()))
        .count();
    let visible_relationship_rows = graph
        .edges()
        .filter(|edge| plan.read_layers.contains_layer(edge.layer()))
        .count();
    let capacities = ResidentNullableRelationCapacities::derive(
        &program,
        graph.node_slot_count(),
        graph.edge_slot_count(),
        visible_node_rows,
        visible_relationship_rows,
        max_output_rows,
    )?;
    let generation = ResidentNullableRelationGeneration {
        project,
        bookmark,
        graph_revision: graph.revision(),
        layout_version: graph.layout_version(),
        catalog_generation: catalog.optimizer_generation(),
    };
    let request = ResidentNullableRelationRequest::build_with_predicates(
        generation,
        fresh_execution_id(),
        program,
        predicate_program,
        capacities,
        1,
    )?;
    Ok(Some(CompiledResidentNullableRelationPlan {
        request,
        outputs,
    }))
}

/// Typed normal form for a direct target produced by one nullable pattern.  A path expands to its
/// resident entity columns in lexical path order; it never becomes a host-materialized path or a
/// list of canonical IDs.  This compiler-only shape is deliberately separate from the v1 DELETE
/// ABI until one backend command can own nullable selection, every target, and typed publication.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum NullableDeleteTargetKind {
    Entity(EntityKind),
    Path,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct NullableDeleteOutput {
    name: String,
    kind: NullableDeleteTargetKind,
}

#[derive(Clone, Debug, PartialEq)]
struct NullableDeleteShape {
    pattern: Pattern,
    target_variable: String,
    target_kind: NullableDeleteTargetKind,
    target_bindings: Vec<ResidentEntityBinding>,
    detach: bool,
    output: Option<NullableDeleteOutput>,
}

/// Recognizes exactly one propertyless OPTIONAL entity/path source followed by one direct DELETE
/// target and either FINISH or a direct nullable target projection.  Checkpoints are planning
/// metadata and do not change this lexical contract.  Wider projections, multiple targets,
/// variable-length paths, predicates, or a second semantic operator fail before any request exists.
fn nullable_delete_shape(plan: &PhysicalPlan) -> Option<NullableDeleteShape> {
    let semantic = plan
        .operators
        .iter()
        .filter(|operator| !matches!(operator, PhysicalOperator::CardinalityCheckpoint { .. }))
        .collect::<Vec<_>>();
    let (scan, delete, trailing) = match semantic.as_slice() {
        [scan, delete] => (*scan, *delete, None),
        [scan, delete, trailing] => (*scan, *delete, Some(*trailing)),
        _ => return None,
    };
    let PhysicalOperator::ScanPattern {
        optional: true,
        pattern,
        ..
    } = scan
    else {
        return None;
    };
    if pattern.selector != PathSelector::All
        || pattern.mode != PathMode::DifferentRelationships
        || pattern.start.property_predicate_present
        || !pattern.start.properties.is_empty()
        || pattern.steps.len() > 1
        || pattern.steps.iter().any(|step| {
            step.relationship.variable_length
                || step.relationship.min_hops.is_some()
                || step.relationship.max_hops.is_some()
                || !step.relationship.properties.is_empty()
                || step.node.property_predicate_present
                || !step.node.properties.is_empty()
        })
    {
        return None;
    }
    let PhysicalOperator::Delete {
        detach,
        expressions,
    } = delete
    else {
        return None;
    };
    let [Expression::Variable(target_variable)] = expressions.as_slice() else {
        return None;
    };

    let (target_kind, target_bindings) =
        if pattern.variable.as_deref() == Some(target_variable.as_str()) {
            if pattern.steps.len() != 1 {
                return None;
            }
            (
                NullableDeleteTargetKind::Path,
                vec![
                    ResidentEntityBinding::Node(ResidentNodeBinding::Start),
                    ResidentEntityBinding::Relationship(0),
                    ResidentEntityBinding::Node(ResidentNodeBinding::End),
                ],
            )
        } else if pattern.steps.is_empty()
            && pattern.start.variable.as_deref() == Some(target_variable.as_str())
        {
            (
                NullableDeleteTargetKind::Entity(EntityKind::Node),
                vec![ResidentEntityBinding::Node(ResidentNodeBinding::Start)],
            )
        } else {
            let [step] = pattern.steps.as_slice() else {
                return None;
            };
            if pattern.variable.is_some()
                || step.relationship.variable.as_deref() != Some(target_variable.as_str())
            {
                return None;
            }
            (
                NullableDeleteTargetKind::Entity(EntityKind::Relationship),
                vec![ResidentEntityBinding::Relationship(0)],
            )
        };

    let output = match trailing {
        None | Some(PhysicalOperator::Finish) => None,
        Some(PhysicalOperator::Project {
            keep_scope: false,
            projection,
        }) if !projection.distinct => {
            let [item] = projection.items.as_slice() else {
                return None;
            };
            if !matches!(&item.expression, Expression::Variable(variable) if variable == target_variable)
            {
                return None;
            }
            Some(NullableDeleteOutput {
                name: item.column_name(0),
                kind: target_kind,
            })
        }
        _ => return None,
    };
    Some(NullableDeleteShape {
        pattern: pattern.clone(),
        target_variable: target_variable.clone(),
        target_kind,
        target_bindings,
        detach: *detach,
        output,
    })
}

/// The v1 resident DELETE command can consume a nullable node scan only when no nullable value is
/// published.  Optional relationship/path sources and nullable entity/path result columns need a
/// wider sealed command; keeping this predicate explicit prevents the normal form from becoming
/// accidental partial admission.
fn nullable_delete_shape_fits_v1(shape: &NullableDeleteShape) -> bool {
    // Both ordinary and DETACH node commands carry the same v1 Boolean lane. Reading it here is
    // part of the compiler fence even though either value is representable.
    let _detach = shape.detach;
    shape.pattern.steps.is_empty()
        && shape.pattern.start.variable.as_deref() == Some(shape.target_variable.as_str())
        && shape.target_kind == NullableDeleteTargetKind::Entity(EntityKind::Node)
        && matches!(
            shape.target_bindings.as_slice(),
            [ResidentEntityBinding::Node(ResidentNodeBinding::Start)]
        )
        && shape.output.is_none()
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct NullableDeleteEmptyTokenProof {
    missing_start_label: bool,
    missing_relationship_domain: bool,
    missing_end_label: bool,
}

impl NullableDeleteEmptyTokenProof {
    const fn is_empty(self) -> bool {
        self.missing_start_label || self.missing_relationship_domain || self.missing_end_label
    }
}

/// Removes catalog-absent names solely so the generic fixed-row compiler can lower the physical
/// shape, then records where an impossible generation-local sentinel must be restored in the
/// sealed request. Labels are conjunctive, while relationship types are disjunctive: a missing
/// type proves an empty relationship domain only when every requested type is absent.
fn prepare_nullable_delete_selection_pattern(
    pattern: &mut Pattern,
    catalog: &crate::graph::NameCatalog,
) -> NullableDeleteEmptyTokenProof {
    let mut proof = NullableDeleteEmptyTokenProof::default();
    pattern.start.labels.retain(|label| {
        let present = catalog.label(label).is_some();
        proof.missing_start_label |= !present;
        present
    });
    if let Some(step) = pattern.steps.first_mut() {
        let requested_relationship_types = !step.relationship.types.is_empty();
        step.relationship
            .types
            .retain(|name| catalog.relationship_type(name).is_some());
        proof.missing_relationship_domain =
            requested_relationship_types && step.relationship.types.is_empty();
        step.node.labels.retain(|label| {
            let present = catalog.label(label).is_some();
            proof.missing_end_label |= !present;
            present
        });
    }
    proof
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum DeleteCompileSymbol {
    Entity(ResidentEntityBinding),
    CollectedEntity(ResidentEntityBinding),
    /// Compiler-only expansion of one named fixed path into the entity columns already present in
    /// the resident selection relation. No path value or canonical ID list crosses the ABI.
    Path(Vec<ResidentEntityBinding>),
    CollectedPath(Vec<ResidentEntityBinding>),
    Map {
        key: String,
        value: Box<DeleteCompileSymbol>,
    },
    Integer(ResidentDeleteIntegerSource),
    RelationshipType(ResidentEntityBinding),
}

fn delete_pattern_bindings(
    pattern: &crate::cypher::Pattern,
) -> Option<(
    BTreeMap<String, ResidentEntityBinding>,
    BTreeMap<String, DeleteCompileSymbol>,
)> {
    let exact_variable_endpoint_path = matches!(pattern.steps.as_slice(), [step]
        if step.relationship.variable_length
            && matches!(step.relationship.min_hops, None | Some(1))
            && step.relationship.max_hops.is_none()
            && step.relationship.variable.is_none()
            && pattern.variable.is_none());
    if pattern.selector != PathSelector::All
        || pattern.mode != PathMode::DifferentRelationships
        || pattern.steps.len() > u16::MAX as usize
        || (!exact_variable_endpoint_path
            && pattern.steps.iter().any(|step| {
                step.relationship.variable_length
                    || step.relationship.min_hops.is_some()
                    || step.relationship.max_hops.is_some()
            }))
    {
        return None;
    }
    let mut bindings = BTreeMap::new();
    let mut path_bindings = vec![ResidentEntityBinding::Node(ResidentNodeBinding::Start)];
    if let Some(variable) = &pattern.start.variable {
        bindings.insert(
            variable.clone(),
            ResidentEntityBinding::Node(ResidentNodeBinding::Start),
        );
    }
    for (index, step) in pattern.steps.iter().enumerate() {
        let relationship = ResidentEntityBinding::Relationship(u16::try_from(index).ok()?);
        path_bindings.push(relationship);
        if let Some(variable) = &step.relationship.variable {
            bindings.insert(variable.clone(), relationship);
        }
        let node = if index + 1 == pattern.steps.len() {
            ResidentNodeBinding::End
        } else {
            ResidentNodeBinding::Intermediate(u16::try_from(index).ok()?)
        };
        path_bindings.push(ResidentEntityBinding::Node(node));
        if let Some(variable) = &step.node.variable {
            bindings.insert(variable.clone(), ResidentEntityBinding::Node(node));
        }
    }
    let mut scope = bindings
        .iter()
        .map(|(name, binding)| (name.clone(), DeleteCompileSymbol::Entity(*binding)))
        .collect::<BTreeMap<_, _>>();
    if let Some(variable) = &pattern.variable {
        if scope
            .insert(variable.clone(), DeleteCompileSymbol::Path(path_bindings))
            .is_some()
        {
            return None;
        }
    }
    Some((bindings, scope))
}

fn delete_integer_source(
    expression: &Expression,
    scope: &BTreeMap<String, DeleteCompileSymbol>,
    catalog: &crate::graph::NameCatalog,
    parameters: &BTreeMap<String, ResultValue>,
    allow_entity_property: bool,
) -> Result<Option<ResidentDeleteIntegerSource>> {
    match expression {
        Expression::Literal(ScalarValue::Integer(value)) => {
            Ok(Some(ResidentDeleteIntegerSource::Constant(*value)))
        }
        Expression::Parameter(name) => Ok(match parameters.get(name) {
            Some(ResultValue::Scalar(ScalarValue::Integer(value))) => {
                Some(ResidentDeleteIntegerSource::Constant(*value))
            }
            _ => None,
        }),
        Expression::Variable(variable) => Ok(match scope.get(variable) {
            Some(DeleteCompileSymbol::Integer(source)) => Some(*source),
            _ => None,
        }),
        Expression::Property(entity, property) if allow_entity_property => {
            let Expression::Variable(variable) = entity.as_ref() else {
                return Ok(None);
            };
            let Some(DeleteCompileSymbol::Entity(binding)) = scope.get(variable) else {
                return Ok(None);
            };
            Ok(Some(ResidentDeleteIntegerSource::Property {
                binding: *binding,
                property: catalog.property(property),
            }))
        }
        _ => Ok(None),
    }
}

fn delete_relationship_type_source(
    expression: &Expression,
    scope: &BTreeMap<String, DeleteCompileSymbol>,
) -> Option<ResidentEntityBinding> {
    let Expression::Function {
        name,
        distinct: false,
        arguments,
    } = expression
    else {
        return None;
    };
    if !matches!(name.as_slice(), [name] if name.eq_ignore_ascii_case("type")) {
        return None;
    }
    let [Expression::Variable(variable)] = arguments.as_slice() else {
        return None;
    };
    match scope.get(variable) {
        Some(DeleteCompileSymbol::Entity(binding @ ResidentEntityBinding::Relationship(_))) => {
            Some(*binding)
        }
        _ => None,
    }
}

fn resident_compare(operation: BinaryOperator) -> Option<CompareOp> {
    match operation {
        BinaryOperator::Equal => Some(CompareOp::Eq),
        BinaryOperator::NotEqual => Some(CompareOp::NotEq),
        BinaryOperator::Less => Some(CompareOp::Less),
        BinaryOperator::LessOrEqual => Some(CompareOp::LessOrEqual),
        BinaryOperator::Greater => Some(CompareOp::Greater),
        BinaryOperator::GreaterOrEqual => Some(CompareOp::GreaterOrEqual),
        _ => None,
    }
}

fn reverse_resident_compare(operation: CompareOp) -> CompareOp {
    match operation {
        CompareOp::Eq => CompareOp::Eq,
        CompareOp::NotEq => CompareOp::NotEq,
        CompareOp::Less => CompareOp::Greater,
        CompareOp::LessOrEqual => CompareOp::GreaterOrEqual,
        CompareOp::Greater => CompareOp::Less,
        CompareOp::GreaterOrEqual => CompareOp::LessOrEqual,
    }
}

fn integer_literal_or_parameter(
    expression: &Expression,
    parameters: &BTreeMap<String, ResultValue>,
) -> Option<i64> {
    match expression {
        Expression::Literal(ScalarValue::Integer(value)) => Some(*value),
        Expression::Parameter(name) => match parameters.get(name) {
            Some(ResultValue::Scalar(ScalarValue::Integer(value))) => Some(*value),
            _ => None,
        },
        _ => None,
    }
}

/// Lowers the inline property map on one single-node pattern into the same resident STRING
/// predicate program used by ordinary WHERE filters. Pattern-map entries are conjunctive; this
/// helper therefore emits their exact left-to-right SSA conjunction and rejects every non-STRING
/// value rather than allowing the typed-row route to ignore a source constraint.
fn row_pattern_property_filter(
    pattern: &Pattern,
    catalog: &crate::graph::NameCatalog,
    parameters: &BTreeMap<String, ResultValue>,
    binding: ResidentNodeBinding,
) -> Option<Option<ResidentPropertyFilterProgram>> {
    if pattern.start.properties.is_empty() {
        return (!pattern.start.property_predicate_present).then_some(None);
    }
    if !pattern.start.property_predicate_present
        || pattern.start.properties.len() > u16::MAX as usize
    {
        return None;
    }
    let mut instructions = Vec::with_capacity(
        pattern
            .start
            .properties
            .len()
            .saturating_mul(2)
            .saturating_sub(1),
    );
    let mut output = None::<u16>;
    for (name, expression) in &pattern.start.properties {
        let property = catalog.property(name)?;
        let operand = match expression {
            Expression::Literal(ScalarValue::String(value)) => value.as_bytes().to_vec(),
            Expression::Parameter(name) => match parameters.get(name) {
                Some(ResultValue::Scalar(ScalarValue::String(value))) => value.as_bytes().to_vec(),
                _ => return None,
            },
            _ => return None,
        };
        let leaf = u16::try_from(instructions.len()).ok()?;
        instructions.push(ResidentPropertyFilterInstruction::CompareString {
            binding,
            property,
            operation: ResidentStringPredicateOperation::Compare(CompareOp::Eq),
            operand: Some(operand),
        });
        output = Some(match output {
            None => leaf,
            Some(left) => {
                let combined = u16::try_from(instructions.len()).ok()?;
                instructions.push(ResidentPropertyFilterInstruction::And { left, right: leaf });
                combined
            }
        });
    }
    let program = ResidentPropertyFilterProgram {
        instructions,
        output: output?,
    };
    program.validate().ok()?;
    Some(Some(program))
}

fn row_integer_predicate(
    expression: &Expression,
    scope: &BTreeMap<String, RowSymbol>,
    catalog: &crate::graph::NameCatalog,
    graph: &GraphStore,
    parameters: &BTreeMap<String, ResultValue>,
) -> Option<ResidentI64Predicate> {
    let Expression::Binary {
        left,
        operation,
        right,
    } = expression
    else {
        return None;
    };
    let operation = resident_compare(*operation)?;
    let property = |expression: &Expression| {
        let Expression::Property(source, property) = expression else {
            return None;
        };
        let Expression::Variable(variable) = source.as_ref() else {
            return None;
        };
        let RowSymbol::Node(binding) = scope.get(variable).copied()? else {
            return None;
        };
        if binding != ResidentNodeBinding::Start {
            return None;
        }
        let property = catalog.property(property)?;
        graph
            .node_property_is_integer(property)
            .then_some((binding, property))
    };

    if let (Some((binding, property)), Some(operand)) = (
        property(left),
        integer_literal_or_parameter(right, parameters),
    ) {
        return Some(ResidentI64Predicate {
            binding,
            property,
            operation,
            operand,
        });
    }
    let (binding, property) = property(right)?;
    Some(ResidentI64Predicate {
        binding,
        property,
        operation: reverse_resident_compare(operation),
        operand: integer_literal_or_parameter(left, parameters)?,
    })
}

fn delete_relationship_selection_filter(
    expression: &Expression,
    scope: &BTreeMap<String, DeleteCompileSymbol>,
    catalog: &crate::graph::NameCatalog,
    parameters: &BTreeMap<String, ResultValue>,
) -> Option<(
    ResidentEntityBinding,
    Option<crate::types::PropertyId>,
    CompareOp,
    i64,
)> {
    let Expression::Binary {
        left,
        operation,
        right,
    } = expression
    else {
        return None;
    };
    let operation = resident_compare(*operation)?;
    let property = |expression: &Expression| {
        let Expression::Property(entity, property) = expression else {
            return None;
        };
        let Expression::Variable(variable) = entity.as_ref() else {
            return None;
        };
        let DeleteCompileSymbol::Entity(binding) = scope.get(variable)? else {
            return None;
        };
        matches!(*binding, ResidentEntityBinding::Relationship(_))
            .then_some((*binding, catalog.property(property)))
    };
    if let (Some((binding, property)), Some(operand)) = (
        property(left),
        integer_literal_or_parameter(right, parameters),
    ) {
        return Some((binding, property, operation, operand));
    }
    let (binding, property) = property(right)?;
    Some((
        binding,
        property,
        reverse_resident_compare(operation),
        integer_literal_or_parameter(left, parameters)?,
    ))
}

fn apply_delete_projection(
    projection: &Projection,
    keep_scope: bool,
    scope: &BTreeMap<String, DeleteCompileSymbol>,
    catalog: &crate::graph::NameCatalog,
    parameters: &BTreeMap<String, ResultValue>,
    allow_entity_property: bool,
) -> Result<Option<BTreeMap<String, DeleteCompileSymbol>>> {
    if projection.distinct {
        return Ok(None);
    }
    let mut next = if keep_scope {
        scope.clone()
    } else {
        BTreeMap::new()
    };
    for (index, item) in projection.items.iter().enumerate() {
        let symbol = delete_projection_symbol(
            &item.expression,
            scope,
            catalog,
            parameters,
            allow_entity_property,
        )?;
        let Some(symbol) = symbol else {
            return Ok(None);
        };
        next.insert(item.column_name(index), symbol);
    }
    Ok(Some(next))
}

/// Compiles only the representational wrappers used by the Delete5 entity/path scenarios. A
/// collected entity or fixed path denotes the complete stable prewrite row relation; a map is
/// compiler-only structure and never becomes a second resident value representation. All other
/// aggregate/expression shapes remain ineligible for strict native dispatch.
fn delete_projection_symbol(
    expression: &Expression,
    scope: &BTreeMap<String, DeleteCompileSymbol>,
    catalog: &crate::graph::NameCatalog,
    parameters: &BTreeMap<String, ResultValue>,
    allow_entity_property: bool,
) -> Result<Option<DeleteCompileSymbol>> {
    match expression {
        Expression::Variable(variable) => Ok(scope.get(variable).cloned()),
        Expression::Function {
            name,
            distinct: false,
            arguments,
        } if name.len() == 1
            && name[0].eq_ignore_ascii_case("collect")
            && matches!(arguments.as_slice(), [Expression::Variable(_)]) =>
        {
            let [Expression::Variable(variable)] = arguments.as_slice() else {
                return Ok(None);
            };
            Ok(match scope.get(variable) {
                Some(DeleteCompileSymbol::Entity(binding)) => {
                    Some(DeleteCompileSymbol::CollectedEntity(*binding))
                }
                Some(DeleteCompileSymbol::Path(bindings)) => {
                    Some(DeleteCompileSymbol::CollectedPath(bindings.clone()))
                }
                _ => None,
            })
        }
        Expression::Map(entries) if entries.len() == 1 => {
            let (key, value) = &entries[0];
            let Some(value) =
                delete_projection_symbol(value, scope, catalog, parameters, allow_entity_property)?
            else {
                return Ok(None);
            };
            if matches!(value, DeleteCompileSymbol::Integer(_)) {
                return Ok(None);
            }
            Ok(Some(DeleteCompileSymbol::Map {
                key: key.clone(),
                value: Box::new(value),
            }))
        }
        _ => Ok(delete_integer_source(
            expression,
            scope,
            catalog,
            parameters,
            allow_entity_property,
        )?
        .map(DeleteCompileSymbol::Integer)),
    }
}

fn delete_projection_introduces_target_wrapper(projection: &Projection) -> bool {
    projection.items.iter().any(|item| match &item.expression {
        Expression::Map(_) => true,
        Expression::Function {
            name,
            distinct: false,
            arguments,
        } => {
            name.len() == 1
                && name[0].eq_ignore_ascii_case("collect")
                && matches!(arguments.as_slice(), [Expression::Variable(_)])
        }
        _ => false,
    })
}

fn resolve_delete_target_symbol(
    expression: &Expression,
    scope: &BTreeMap<String, DeleteCompileSymbol>,
) -> Option<DeleteCompileSymbol> {
    match expression {
        Expression::Variable(variable) => scope.get(variable).cloned(),
        Expression::Property(source, requested_key) => {
            let DeleteCompileSymbol::Map { key, value } =
                resolve_delete_target_symbol(source, scope)?
            else {
                return None;
            };
            (key == *requested_key).then_some(*value)
        }
        _ => None,
    }
}

fn delete_targets(
    expression: &Expression,
    scope: &BTreeMap<String, DeleteCompileSymbol>,
    parameters: &BTreeMap<String, ResultValue>,
) -> Option<Vec<(ResidentEntityBinding, ResidentDeleteTargetSelector)>> {
    if let Expression::Index { expression, index } = expression {
        let bindings = match resolve_delete_target_symbol(expression, scope)? {
            DeleteCompileSymbol::CollectedEntity(binding) => vec![binding],
            DeleteCompileSymbol::CollectedPath(bindings) => bindings,
            _ => return None,
        };
        let index = u32::try_from(integer_literal_or_parameter(index, parameters)?).ok()?;
        return Some(
            bindings
                .into_iter()
                .map(|binding| {
                    (
                        binding,
                        ResidentDeleteTargetSelector::CollectedOrdinal { index },
                    )
                })
                .collect(),
        );
    }
    let bindings = match resolve_delete_target_symbol(expression, scope)? {
        DeleteCompileSymbol::Entity(binding) => vec![binding],
        DeleteCompileSymbol::Path(bindings) => bindings,
        _ => return None,
    };
    Some(
        bindings
            .into_iter()
            .map(|binding| (binding, ResidentDeleteTargetSelector::EachSelectedRow))
            .collect(),
    )
}

fn delete_modulo_filter(
    expression: &Expression,
    scope: &BTreeMap<String, DeleteCompileSymbol>,
    catalog: &crate::graph::NameCatalog,
    parameters: &BTreeMap<String, ResultValue>,
) -> Result<Option<(ResidentDeleteIntegerSource, i64, i64)>> {
    let Expression::Binary {
        left,
        operation: BinaryOperator::Equal,
        right,
    } = expression
    else {
        return Ok(None);
    };
    let Expression::Binary {
        left: source,
        operation: BinaryOperator::Modulo,
        right: divisor,
    } = left.as_ref()
    else {
        return Ok(None);
    };
    let Some(source) = delete_integer_source(source, scope, catalog, parameters, false)? else {
        return Ok(None);
    };
    let Some(divisor) = integer_literal_or_parameter(divisor, parameters) else {
        return Ok(None);
    };
    let Some(operand) = integer_literal_or_parameter(right, parameters) else {
        return Ok(None);
    };
    if divisor == 0 {
        return Ok(None);
    }
    Ok(Some((source, divisor, operand)))
}

fn delete_continuation_maximum_rows(
    stages: &[ResidentDeletePostStage],
    selected_rows: usize,
) -> usize {
    stages
        .iter()
        .fold(selected_rows, |rows, stage| match stage {
            ResidentDeletePostStage::Project { .. }
            | ResidentDeletePostStage::FilterIntegerModuloEquals { .. } => rows,
            ResidentDeletePostStage::Skip { rows: skip } => rows.saturating_sub(*skip),
            ResidentDeletePostStage::Limit { rows: limit } => rows.min(*limit),
            ResidentDeletePostStage::SumInteger { .. } | ResidentDeletePostStage::Count => 1,
        })
}

fn compile_delete_continuation(
    operators: &[PhysicalOperator],
    mut scope: BTreeMap<String, DeleteCompileSymbol>,
    catalog: &crate::graph::NameCatalog,
    parameters: &BTreeMap<String, ResultValue>,
) -> Result<Option<Option<(Vec<ResidentDeletePostStage>, ResidentDeleteOutput)>>> {
    if matches!(operators, [PhysicalOperator::Finish]) {
        return Ok(Some(None));
    }
    if operators.is_empty()
        || operators
            .iter()
            .any(|operator| matches!(operator, PhysicalOperator::Finish))
    {
        return Ok(None);
    }
    let mut stages = Vec::new();
    let mut output = None;
    for operator in operators {
        match operator {
            PhysicalOperator::CardinalityCheckpoint { .. } => {}
            PhysicalOperator::Project {
                keep_scope,
                projection,
            } if !projection.distinct && projection.items.len() == 1 => {
                let item = &projection.items[0];
                let name = item.column_name(0);
                let aggregate = match &item.expression {
                    Expression::Function {
                        name,
                        distinct: false,
                        arguments,
                    } if name.len() == 1
                        && name[0].eq_ignore_ascii_case("count")
                        && matches!(arguments.as_slice(), [Expression::Star]) =>
                    {
                        stages.push(ResidentDeletePostStage::Count);
                        Some(ResidentDeleteIntegerSource::Aggregate)
                    }
                    Expression::Function {
                        name,
                        distinct: false,
                        arguments,
                    } if name.len() == 1
                        && name[0].eq_ignore_ascii_case("sum")
                        && arguments.len() == 1 =>
                    {
                        let Some(source) = delete_integer_source(
                            &arguments[0],
                            &scope,
                            catalog,
                            parameters,
                            false,
                        )?
                        else {
                            return Ok(None);
                        };
                        stages.push(ResidentDeletePostStage::SumInteger { source });
                        Some(ResidentDeleteIntegerSource::Aggregate)
                    }
                    _ => None,
                };
                let (source, symbol) = if let Some(source) = aggregate {
                    (
                        ResidentDeleteOutputSource::Integer(source),
                        DeleteCompileSymbol::Integer(source),
                    )
                } else if let Some(binding) =
                    delete_relationship_type_source(&item.expression, &scope)
                {
                    let source = ResidentDeleteOutputSource::RelationshipType(binding);
                    stages.push(ResidentDeletePostStage::Project { source });
                    (source, DeleteCompileSymbol::RelationshipType(binding))
                } else {
                    let Some(source) = delete_integer_source(
                        &item.expression,
                        &scope,
                        catalog,
                        parameters,
                        false,
                    )?
                    else {
                        return Ok(None);
                    };
                    stages.push(ResidentDeletePostStage::Project {
                        source: ResidentDeleteOutputSource::Integer(source),
                    });
                    (
                        ResidentDeleteOutputSource::Integer(source),
                        DeleteCompileSymbol::Integer(source),
                    )
                };
                let mut next = if *keep_scope {
                    scope.clone()
                } else {
                    BTreeMap::new()
                };
                next.insert(name.clone(), symbol);
                scope = next;
                output = Some(ResidentDeleteOutput { name, source });
            }
            PhysicalOperator::Filter(expression) => {
                let Some((source, divisor, operand)) =
                    delete_modulo_filter(expression, &scope, catalog, parameters)?
                else {
                    return Ok(None);
                };
                stages.push(ResidentDeletePostStage::FilterIntegerModuloEquals {
                    source,
                    divisor,
                    operand,
                });
            }
            PhysicalOperator::Skip(expression) => {
                let Some(rows) = non_negative_count(expression, parameters) else {
                    return Ok(None);
                };
                stages.push(ResidentDeletePostStage::Skip { rows });
            }
            PhysicalOperator::Limit(expression) => {
                let Some(rows) = non_negative_count(expression, parameters) else {
                    return Ok(None);
                };
                stages.push(ResidentDeletePostStage::Limit { rows });
            }
            _ => return Ok(None),
        }
    }
    let Some(output) = output else {
        return Ok(None);
    };
    if stages.is_empty() {
        return Ok(None);
    }
    Ok(Some(Some((stages, output))))
}

/// Lowers the strict fixed-length DELETE tranche into one backend-neutral resident command.
/// Returning `None` means that some semantic operator escaped this exact command; strict native
/// execution must then fail closed rather than evaluating a host tail.
#[allow(clippy::too_many_arguments)]
pub fn compile_delete(
    plan: &PhysicalPlan,
    project: ProjectId,
    bookmark: Bookmark,
    catalog: &crate::graph::NameCatalog,
    graph: &GraphStore,
    parameters: &BTreeMap<String, ResultValue>,
    execution: ResidentExecutionId,
    max_output_rows: usize,
) -> Result<Option<CompiledResidentDeletePlan>> {
    if plan.read_only || plan.at_time.is_some() || !plan.unions.is_empty() {
        return Ok(None);
    }
    let nullable_shape = nullable_delete_shape(plan);
    let nullable_token_proof =
        nullable_shape
            .as_ref()
            .map_or_else(NullableDeleteEmptyTokenProof::default, |shape| {
                let mut pattern = shape.pattern.clone();
                prepare_nullable_delete_selection_pattern(&mut pattern, catalog)
            });
    let selection_is_statically_empty = nullable_shape.as_ref().is_some_and(|shape| {
        nullable_token_proof.is_empty()
            || (!shape.pattern.steps.is_empty() && graph.edge_count() == 0)
    });
    if let Some(shape) = &nullable_shape
        && !nullable_delete_shape_fits_v1(shape)
        && !matches!(
            (shape.target_kind, shape.output.as_ref()),
            (
                NullableDeleteTargetKind::Entity(EntityKind::Node | EntityKind::Relationship),
                Some(_)
            ) | (NullableDeleteTargetKind::Path, None)
        )
    {
        return Ok(None);
    }
    if nullable_shape
        .as_ref()
        .is_some_and(|shape| shape.output.is_some())
        && !selection_is_statically_empty
    {
        // Publishing an entity after DELETE would need a canonical prewrite entity snapshot. The
        // first typed continuation is deliberately narrower: it publishes only the proven-null
        // entity of a statically empty OPTIONAL pattern.
        return Ok(None);
    }
    let delete_index = plan
        .operators
        .iter()
        .position(|operator| matches!(operator, PhysicalOperator::Delete { .. }));
    let Some(delete_index) = delete_index else {
        return Ok(None);
    };
    if plan.operators[delete_index + 1..]
        .iter()
        .any(|operator| matches!(operator, PhysicalOperator::Delete { .. }))
    {
        return Ok(None);
    }
    let Some(PhysicalOperator::ScanPattern {
        optional: initial_optional,
        pattern,
        ..
    }) = plan.operators.first()
    else {
        return Ok(None);
    };
    let variable_path_pattern = matches!(pattern.steps.as_slice(), [step]
        if pattern.variable.is_none()
            && pattern.start.labels.is_empty()
            && pattern.start.properties.is_empty()
            && !pattern.start.property_predicate_present
            && step.relationship.variable_length
            && matches!(step.relationship.min_hops, None | Some(1))
            && step.relationship.max_hops.is_none()
            && step.relationship.direction == Direction::Undirected
            && step.relationship.variable.is_none()
            && step.relationship.types.is_empty()
            && step.relationship.properties.is_empty()
            && step.node.labels.is_empty()
            && step.node.properties.is_empty()
            && !step.node.property_predicate_present)
    .then_some(pattern.clone());
    if *initial_optional
        && nullable_shape.is_none()
        && (!pattern.start.properties.is_empty() || !pattern.steps.is_empty())
    {
        return Ok(None);
    }
    let Some((mut bindings, mut scope)) = delete_pattern_bindings(pattern) else {
        return Ok(None);
    };
    if bindings.is_empty() && scope.is_empty() {
        return Ok(None);
    }
    let mut selection_operators = Vec::new();
    let mut selection_filter_specs = Vec::new();
    let mut saw_target_wrapper_projection = false;
    for operator in &plan.operators[..delete_index] {
        match operator {
            PhysicalOperator::ScanPattern {
                match_group,
                optional,
                pattern,
                access,
            } => {
                if saw_target_wrapper_projection {
                    return Ok(None);
                }
                if *optional
                    && nullable_shape.is_none()
                    && (selection_operators.is_empty()
                        || !pattern.start.properties.is_empty()
                        || pattern.steps.len() != 1)
                {
                    return Ok(None);
                }
                if *optional && nullable_shape.is_none() {
                    let Some((pattern_bindings, pattern_scope)) = delete_pattern_bindings(pattern)
                    else {
                        return Ok(None);
                    };
                    for (name, binding) in pattern_bindings {
                        if bindings
                            .insert(name.clone(), binding)
                            .is_some_and(|existing| existing != binding)
                        {
                            return Ok(None);
                        }
                    }
                    for (name, symbol) in pattern_scope {
                        if scope
                            .insert(name.clone(), symbol.clone())
                            .is_some_and(|existing| existing != symbol)
                        {
                            return Ok(None);
                        }
                    }
                }
                let mut pattern = pattern.clone();
                if *optional && nullable_shape.is_some() {
                    let _ = prepare_nullable_delete_selection_pattern(&mut pattern, catalog);
                }
                // A named path is compiler-only structure expanded into the selector's existing
                // entity columns. The resident command therefore never materializes a path value.
                pattern.variable = None;
                if variable_path_pattern.is_some() {
                    let [step] = pattern.steps.as_mut_slice() else {
                        return Ok(None);
                    };
                    step.relationship.variable_length = false;
                    step.relationship.min_hops = None;
                    step.relationship.max_hops = None;
                }
                selection_operators.push(PhysicalOperator::ScanPattern {
                    match_group: *match_group,
                    // Whole-pattern OPTIONAL null extension is restored after generic fixed-row
                    // lowering so a one-hop source is not mistaken for a per-start optional join.
                    optional: *optional && nullable_shape.is_none(),
                    pattern,
                    access: access.clone(),
                });
            }
            PhysicalOperator::CardinalityCheckpoint { .. } => {
                selection_operators.push(operator.clone());
            }
            PhysicalOperator::Filter(expression) => {
                if saw_target_wrapper_projection {
                    return Ok(None);
                }
                if let Some(filter) =
                    delete_relationship_selection_filter(expression, &scope, catalog, parameters)
                {
                    selection_filter_specs.push(filter);
                } else {
                    selection_operators.push(operator.clone());
                }
            }
            PhysicalOperator::Project {
                keep_scope,
                projection,
            } => {
                if saw_target_wrapper_projection {
                    return Ok(None);
                }
                let introduces_target_wrapper =
                    delete_projection_introduces_target_wrapper(projection);
                if introduces_target_wrapper && (*keep_scope || projection.items.len() != 1) {
                    return Ok(None);
                }
                let Some(next) = apply_delete_projection(
                    projection,
                    *keep_scope,
                    &scope,
                    catalog,
                    parameters,
                    true,
                )?
                else {
                    return Ok(None);
                };
                scope = next;
                saw_target_wrapper_projection = introduces_target_wrapper;
            }
            _ => return Ok(None),
        }
    }
    let PhysicalOperator::Delete {
        detach,
        expressions,
    } = &plan.operators[delete_index]
    else {
        return Err(Error::internal("resident DELETE operator disappeared"));
    };
    if expressions.is_empty() {
        return Ok(None);
    }
    let mut command_targets = Vec::with_capacity(expressions.len());
    for expression in expressions {
        let Some(targets) = delete_targets(expression, &scope, parameters) else {
            return Ok(None);
        };
        command_targets.extend(targets);
    }
    let continuation_shape = if let Some(output) = nullable_shape
        .as_ref()
        .and_then(|shape| shape.output.as_ref())
    {
        let NullableDeleteTargetKind::Entity(kind) = output.kind else {
            return Ok(None);
        };
        let source = ResidentDeleteOutputSource::NullEntity(kind);
        Some((
            vec![ResidentDeletePostStage::Project { source }],
            ResidentDeleteOutput {
                name: output.name.clone(),
                source,
            },
        ))
    } else {
        let Some(continuation) = compile_delete_continuation(
            &plan.operators[delete_index + 1..],
            scope,
            catalog,
            parameters,
        )?
        else {
            return Ok(None);
        };
        continuation
    };

    // The existing graph-row compiler remains the one owner of MATCH/WHERE semantics. A path with
    // entirely anonymous components still needs one internal output symbol so the compiler keeps
    // its complete row source; the symbol never crosses the request boundary.
    if bindings.is_empty() {
        const HIDDEN_START: &str = "\0resident_delete_start";
        let Some(pattern) = selection_operators
            .iter_mut()
            .find_map(|operator| match operator {
                PhysicalOperator::ScanPattern { pattern, .. } => Some(pattern),
                _ => None,
            })
        else {
            return Ok(None);
        };
        if pattern.start.variable.is_some() {
            return Ok(None);
        }
        pattern.start.variable = Some(HIDDEN_START.to_owned());
        bindings.insert(
            HIDDEN_START.to_owned(),
            ResidentEntityBinding::Node(ResidentNodeBinding::Start),
        );
    }
    // This synthesized projection retains only entity bindings and is discarded before dispatch.
    selection_operators.push(PhysicalOperator::Project {
        keep_scope: false,
        projection: Projection {
            distinct: false,
            items: bindings
                .keys()
                .map(|variable| ProjectionItem {
                    expression: Expression::Variable(variable.clone()),
                    alias: Some(variable.clone()),
                    source_text: None,
                })
                .collect(),
        },
    });
    let mut selection_plan = plan.clone();
    selection_plan.read_only = true;
    selection_plan.operators = selection_operators;
    selection_plan.unions.clear();
    let variable_path_selection = if let Some(path) = variable_path_pattern.as_ref() {
        let Some(request) = super::resident_variable_path::compile_request(
            plan,
            project,
            bookmark,
            catalog,
            graph,
            None,
            path,
            execution,
            max_output_rows.max(1),
        )?
        else {
            return Ok(None);
        };
        Some(request)
    } else {
        None
    };
    let Some(mut selected) = super::resident::compile(
        &selection_plan,
        project,
        catalog,
        graph,
        None,
        parameters,
        bookmark.index,
        u32::MAX as usize,
    )?
    else {
        return Ok(None);
    };
    if selected.grouping.is_some() || selected.temporal.is_some() {
        return Ok(None);
    }
    if nullable_shape.is_some() {
        if nullable_token_proof.missing_start_label {
            selected
                .request
                .labels
                .push(LabelId(catalog.next_label_id()));
        }
        if let Some(expansion) = selected.request.expansion.as_mut() {
            if nullable_token_proof.missing_relationship_domain {
                expansion
                    .relationship_types
                    .push(RelationshipTypeId(catalog.next_relationship_type_id()));
            }
            if nullable_token_proof.missing_end_label {
                expansion.end_labels.push(LabelId(catalog.next_label_id()));
            }
        } else if nullable_token_proof.missing_relationship_domain
            || nullable_token_proof.missing_end_label
        {
            return Ok(None);
        }
    }
    let mut selection_capacity = if let Some(path) = &variable_path_selection {
        path.maximum_output_rows
    } else {
        selected
            .request
            .maximum_pipeline_rows(graph.node_slot_count(), graph.edge_slot_count())?
    };
    if nullable_shape.is_some() {
        selection_capacity = selection_capacity.max(1);
    }
    if selection_capacity > u32::MAX as usize {
        return Err(Error::new(
            ErrorCode::ResultBudgetExceeded,
            "resident DELETE selection exceeds its compact row ABI",
        ));
    }
    selected.request.orders.clear();
    selected.request.offset = 0;
    selected.request.limit = usize::MAX;
    selected.request.integer_projections.clear();
    selected.request.property_null_projections.clear();
    selected.request.max_output_rows = selection_capacity.max(1);
    selected.request.mutation = None;
    selected.request.initial_optional = nullable_shape.is_some();
    selected.request.validate_mutation()?;

    let mut next_obligation_id = 1_u64;
    let mut obligation = |kind, scope| -> Result<ResidentExecutionObligation> {
        let id = next_obligation_id;
        next_obligation_id = next_obligation_id.checked_add(1).ok_or_else(|| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "resident DELETE obligation ID space exhausted",
            )
        })?;
        Ok(ResidentExecutionObligation { id, kind, scope })
    };
    let selection_obligation = obligation(
        ResidentObligationKind::MutationSelect,
        ResidentObligationScope::Selection,
    )?;
    let selection_filters = selection_filter_specs
        .into_iter()
        .enumerate()
        .map(|(index, (binding, property, operation, operand))| {
            Ok(ResidentDeleteSelectionFilter {
                binding,
                property,
                operation,
                operand,
                obligation: obligation(
                    ResidentObligationKind::Filter,
                    ResidentObligationScope::Filter(u16::try_from(index).map_err(|_| {
                        Error::new(
                            ErrorCode::ResultBudgetExceeded,
                            "resident DELETE selection filter index exceeds u16",
                        )
                    })?),
                )?,
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let read_dependency_obligation = obligation(
        ResidentObligationKind::MutationReadSet,
        ResidentObligationScope::Selection,
    )?;
    let mut commands = Vec::with_capacity(command_targets.len());
    for (index, (target, selector)) in command_targets.into_iter().enumerate() {
        let scope =
            ResidentObligationScope::MutationCommand(u16::try_from(index).map_err(|_| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "resident DELETE command index exceeds u16",
                )
            })?);
        commands.push(ResidentDeleteCommand {
            target,
            selector,
            detach: *detach,
            rhs_obligation: obligation(ResidentObligationKind::MutationRhs, scope)?,
            effect_obligation: obligation(ResidentObligationKind::MutationEffect, scope)?,
        });
    }
    let continuation = continuation_shape
        .map(|(stages, output)| -> Result<ResidentDeleteContinuation> {
            let stage_obligations = stages
                .iter()
                .enumerate()
                .map(|(index, stage)| {
                    let kind = match stage {
                        ResidentDeletePostStage::Project { .. } => {
                            ResidentObligationKind::Expression
                        }
                        ResidentDeletePostStage::FilterIntegerModuloEquals { .. } => {
                            ResidentObligationKind::Filter
                        }
                        ResidentDeletePostStage::Skip { .. }
                        | ResidentDeletePostStage::Limit { .. } => ResidentObligationKind::Sort,
                        ResidentDeletePostStage::SumInteger { .. }
                        | ResidentDeletePostStage::Count => ResidentObligationKind::Aggregate,
                    };
                    obligation(
                        kind,
                        ResidentObligationScope::Expression(u16::try_from(index).map_err(
                            |_| {
                                Error::new(
                                    ErrorCode::ResultBudgetExceeded,
                                    "resident DELETE stage index exceeds u16",
                                )
                            },
                        )?),
                    )
                })
                .collect::<Result<Vec<_>>>()?;
            let final_relation_obligation = obligation(
                ResidentObligationKind::Sort,
                ResidentObligationScope::Expression(u16::try_from(stages.len()).map_err(|_| {
                    Error::new(
                        ErrorCode::ResultBudgetExceeded,
                        "resident DELETE final relation index exceeds u16",
                    )
                })?),
            )?;
            let maximum_output_rows =
                delete_continuation_maximum_rows(&stages, selection_capacity).min(max_output_rows);
            Ok(ResidentDeleteContinuation {
                stage_obligations,
                final_relation_obligation,
                stages,
                output,
                maximum_output_rows,
                maximum_output_cells: maximum_output_rows,
            })
        })
        .transpose()?;
    let maximum_read_dependencies = graph
        .node_slot_count()
        .checked_add(graph.edge_slot_count())
        .ok_or_else(|| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "resident DELETE read-dependency capacity overflow",
            )
        })?;
    let detach_capacity = if *detach { graph.edge_slot_count() } else { 0 };
    let maximum_delete_intents = selection_capacity
        .checked_mul(commands.len())
        .and_then(|capacity| capacity.checked_add(detach_capacity))
        .ok_or_else(|| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "resident DELETE intent capacity overflow",
            )
        })?
        .max(1);
    let mut request = ResidentDeleteRequest {
        project,
        expected_bookmark: bookmark,
        expected_graph_revision: graph.revision(),
        expected_layout_version: graph.layout_version(),
        execution,
        fingerprint: ResidentDeleteFingerprint([0; 32]),
        selection_obligation,
        read_dependency_obligation,
        selection: selected.request,
        variable_path_selection,
        selection_is_statically_empty,
        selection_filters,
        commands,
        continuation,
        maximum_delete_intents,
        maximum_read_dependencies,
    };
    request.seal()?;
    request.validate()?;
    Ok(Some(CompiledResidentDeletePlan { request }))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
struct NullableRelationBinding {
    slot: ResidentNullableRelationSlot,
    kind: ResidentNullableRelationBindingKind,
}

fn nullable_predicate_value_slots(
    value: &ResidentNullableRelationPredicateValue,
    slots: &mut BTreeSet<ResidentNullableRelationSlot>,
) {
    match value {
        ResidentNullableRelationPredicateValue::Binding { slot, .. }
        | ResidentNullableRelationPredicateValue::IntegerProperty { slot, .. }
        | ResidentNullableRelationPredicateValue::StringProperty { slot, .. } => {
            slots.insert(*slot);
        }
        ResidentNullableRelationPredicateValue::Null
        | ResidentNullableRelationPredicateValue::Boolean(_)
        | ResidentNullableRelationPredicateValue::Integer(_)
        | ResidentNullableRelationPredicateValue::String(_) => {}
    }
}

fn nullable_predicate_slots(
    predicate: &ResidentNullableRelationPredicate,
    slots: &mut BTreeSet<ResidentNullableRelationSlot>,
) {
    match predicate {
        ResidentNullableRelationPredicate::Constant(_) => {}
        ResidentNullableRelationPredicate::IsNull { value, .. } => {
            nullable_predicate_value_slots(value, slots);
        }
        ResidentNullableRelationPredicate::HasLabels { node, .. } => {
            slots.insert(*node);
        }
        ResidentNullableRelationPredicate::CompareInteger { left, right, .. }
        | ResidentNullableRelationPredicate::CompareString { left, right, .. } => {
            nullable_predicate_value_slots(left, slots);
            nullable_predicate_value_slots(right, slots);
        }
        ResidentNullableRelationPredicate::RelationshipEndpoint {
            relationship, node, ..
        } => {
            slots.insert(*relationship);
            slots.insert(*node);
        }
        ResidentNullableRelationPredicate::Not(operand) => {
            nullable_predicate_slots(operand, slots);
        }
        ResidentNullableRelationPredicate::And(left, right)
        | ResidentNullableRelationPredicate::Or(left, right) => {
            nullable_predicate_slots(left, slots);
            nullable_predicate_slots(right, slots);
        }
    }
}

#[allow(dead_code)]
struct NullableRelationBuilder<'a> {
    catalog: &'a crate::graph::NameCatalog,
    graph: &'a GraphStore,
    parameters: &'a BTreeMap<String, ResultValue>,
    node_properties: Option<PropertyColumns>,
    relationship_properties: Option<PropertyColumns>,
    scope: BTreeMap<String, NullableRelationBinding>,
    stages: Vec<ResidentNullableRelationStage>,
    predicate_program: ResidentNullableRelationPredicateProgram,
    uniqueness_group: Option<MatchGroupId>,
    uniqueness_relationships: BTreeSet<ResidentNullableRelationSlot>,
    fixed_path_lengths: BTreeMap<String, i64>,
    /// Named OPTIONAL paths whose required start binding is proven to be always null by the
    /// sealed staged program. The path value and every graph-list function over it are therefore
    /// exactly Cypher null without materializing a path on either backend.
    statically_null_paths: BTreeMap<String, NullableRelationBinding>,
    static_null_node_seeded: bool,
    unmaterialized_scope_values: BTreeSet<String>,
    next_slot: u16,
}

#[allow(dead_code)]
impl<'a> NullableRelationBuilder<'a> {
    fn new(
        catalog: &'a crate::graph::NameCatalog,
        graph: &'a GraphStore,
        parameters: &'a BTreeMap<String, ResultValue>,
    ) -> Self {
        Self {
            catalog,
            graph,
            parameters,
            node_properties: None,
            relationship_properties: None,
            scope: BTreeMap::new(),
            stages: Vec::new(),
            predicate_program: ResidentNullableRelationPredicateProgram::default(),
            uniqueness_group: None,
            uniqueness_relationships: BTreeSet::new(),
            fixed_path_lengths: BTreeMap::new(),
            statically_null_paths: BTreeMap::new(),
            static_null_node_seeded: false,
            unmaterialized_scope_values: BTreeSet::new(),
            next_slot: 0,
        }
    }

    fn allocate_slot(&mut self) -> Result<ResidentNullableRelationSlot> {
        if usize::from(self.next_slot) >= crate::execution::RESIDENT_NULLABLE_RELATION_MAX_BINDINGS
        {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident nullable relation compiler exhausted binding slots",
            ));
        }
        let slot = ResidentNullableRelationSlot(self.next_slot);
        self.next_slot = self.next_slot.checked_add(1).ok_or_else(|| {
            Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident nullable relation compiler binding-slot overflow",
            )
        })?;
        Ok(slot)
    }

    fn binding_is_statically_always_null(&self, binding: NullableRelationBinding) -> Result<bool> {
        if self.stages.is_empty() {
            return Ok(false);
        }
        let mut stages = self.stages.clone();
        stages.push(ResidentNullableRelationStage::FinalProject {
            bindings: vec![ResidentNullableRelationOutputBinding {
                name: "__static_null_proof".to_owned(),
                source: ResidentNullableRelationOutputSource::Entity {
                    slot: binding.slot,
                    kind: binding.kind,
                },
            }],
        });
        ResidentNullableRelationProgram {
            layers: crate::graph::LayerMask::ALL,
            stages,
        }
        .binding_is_statically_always_null(binding.slot, binding.kind)
    }

    fn binding_is_statically_never_null(&self, binding: NullableRelationBinding) -> Result<bool> {
        if self.stages.is_empty() {
            return Ok(false);
        }
        let mut stages = self.stages.clone();
        stages.push(ResidentNullableRelationStage::FinalProject {
            bindings: vec![ResidentNullableRelationOutputBinding {
                name: "__static_non_null_proof".to_owned(),
                source: ResidentNullableRelationOutputSource::Entity {
                    slot: binding.slot,
                    kind: binding.kind,
                },
            }],
        });
        ResidentNullableRelationProgram {
            layers: crate::graph::LayerMask::ALL,
            stages,
        }
        .binding_is_statically_never_null(binding.slot, binding.kind)
    }

    fn property_columns(
        &mut self,
        kind: ResidentNullableRelationBindingKind,
    ) -> Result<&PropertyColumns> {
        if self.node_properties.is_none() || self.relationship_properties.is_none() {
            let snapshot = self.graph.snapshot()?;
            if self.node_properties.is_none() {
                self.node_properties = Some(snapshot.node_properties);
            }
            if self.relationship_properties.is_none() {
                self.relationship_properties = Some(snapshot.edge_properties);
            }
        }
        match kind {
            ResidentNullableRelationBindingKind::Node => self.node_properties.as_ref(),
            ResidentNullableRelationBindingKind::Relationship => {
                self.relationship_properties.as_ref()
            }
        }
        .ok_or_else(|| Error::internal("resident nullable property columns vanished"))
    }

    fn string_property_maximum_bytes(
        properties: &PropertyColumns,
        property: crate::types::PropertyId,
    ) -> Result<u32> {
        let Some(TypedColumn::String { values, validity }) = properties.column(property) else {
            return Err(Error::internal(
                "resident nullable string projection lost its canonical string column",
            ));
        };
        let dictionary = properties.string_dictionary();
        let mut maximum = 0_usize;
        for row in 0..properties.rows() {
            if !validity.is_present(row) {
                continue;
            }
            let id = values.get(row).ok_or_else(|| {
                Error::new(
                    ErrorCode::CorruptStorage,
                    "resident nullable string projection column is shorter than its validity",
                )
            })?;
            let value = dictionary.resolve(*id).ok_or_else(|| {
                Error::new(
                    ErrorCode::CorruptStorage,
                    "resident nullable string projection references an unknown dictionary entry",
                )
            })?;
            maximum = maximum.max(value.len());
        }
        u32::try_from(maximum).map_err(|_| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "resident nullable string property width exceeds u32",
            )
        })
    }

    fn property_output_source(
        &mut self,
        binding: NullableRelationBinding,
        name: &str,
    ) -> Result<Option<ResidentNullableRelationOutputSource>> {
        let Some(property) = self.catalog.property(name) else {
            return Ok(Some(ResidentNullableRelationOutputSource::NullProperty {
                slot: binding.slot,
                kind: binding.kind,
            }));
        };
        let properties = self.property_columns(binding.kind)?;
        if properties.is_mixed(property) {
            return Ok(None);
        }
        let source = match properties.column(property) {
            Some(TypedColumn::Integer { .. }) => {
                ResidentNullableRelationOutputSource::IntegerProperty {
                    slot: binding.slot,
                    kind: binding.kind,
                    property,
                }
            }
            Some(TypedColumn::Float { .. }) => {
                ResidentNullableRelationOutputSource::FloatProperty {
                    slot: binding.slot,
                    kind: binding.kind,
                    property,
                }
            }
            Some(TypedColumn::String { .. }) => {
                ResidentNullableRelationOutputSource::StringProperty {
                    slot: binding.slot,
                    kind: binding.kind,
                    property,
                    maximum_bytes: Self::string_property_maximum_bytes(properties, property)?,
                }
            }
            None => ResidentNullableRelationOutputSource::NullProperty {
                slot: binding.slot,
                kind: binding.kind,
            },
            Some(_) => return Ok(None),
        };
        Ok(Some(source))
    }

    /// Compile the bounded nullable property-conversion tranche only when the argument is a
    /// direct property lookup and the immutable canonical tensor has the exact consumed shape.
    fn converted_property_output_source(
        &mut self,
        function: &str,
        argument: &Expression,
    ) -> Result<Option<ResidentNullableRelationOutputSource>> {
        let Expression::Property(source, property) = argument else {
            return Ok(None);
        };
        let Expression::Variable(variable) = source.as_ref() else {
            return Ok(None);
        };
        let Some(binding) = self.scope.get(variable).copied() else {
            return Ok(None);
        };
        let Some(source) = self.property_output_source(binding, property)? else {
            return Ok(None);
        };
        let converted = match (function, source) {
            (
                "tointeger",
                ResidentNullableRelationOutputSource::StringProperty {
                    slot,
                    kind,
                    property,
                    ..
                },
            ) => ResidentNullableRelationOutputSource::StringPropertyToInteger {
                slot,
                kind,
                property,
            },
            (
                "tofloat",
                ResidentNullableRelationOutputSource::IntegerProperty {
                    slot,
                    kind,
                    property,
                },
            ) => ResidentNullableRelationOutputSource::IntegerPropertyToFloat {
                slot,
                kind,
                property,
            },
            (
                "tostring",
                ResidentNullableRelationOutputSource::IntegerProperty {
                    slot,
                    kind,
                    property,
                },
            ) => ResidentNullableRelationOutputSource::IntegerPropertyToString {
                slot,
                kind,
                property,
            },
            (_, ResidentNullableRelationOutputSource::NullProperty { slot, kind }) => {
                ResidentNullableRelationOutputSource::NullProperty { slot, kind }
            }
            _ => return Ok(None),
        };
        Ok(Some(converted))
    }

    fn node_property_value(
        &mut self,
        node: ResidentNullableRelationSlot,
        name: &str,
    ) -> Result<Option<ResidentNullableRelationPredicateValue>> {
        let Some(property) = self.catalog.property(name) else {
            return Ok(Some(ResidentNullableRelationPredicateValue::Null));
        };
        let properties = self.property_columns(ResidentNullableRelationBindingKind::Node)?;
        if properties.is_mixed(property) {
            return Ok(None);
        }
        let value = match properties.column(property) {
            Some(TypedColumn::Integer { .. }) => {
                ResidentNullableRelationPredicateValue::IntegerProperty {
                    slot: node,
                    kind: ResidentNullableRelationBindingKind::Node,
                    property,
                }
            }
            Some(TypedColumn::String { .. }) => {
                ResidentNullableRelationPredicateValue::StringProperty {
                    slot: node,
                    kind: ResidentNullableRelationBindingKind::Node,
                    property,
                }
            }
            None => ResidentNullableRelationPredicateValue::Null,
            Some(_) => return Ok(None),
        };
        Ok(Some(value))
    }

    fn inline_property_equality(
        actual: ResidentNullableRelationPredicateValue,
        expected: &Expression,
    ) -> Option<ResidentNullableRelationPredicate> {
        if let Some(expected) = Self::inline_integer_literal(expected) {
            return match actual {
                actual @ (ResidentNullableRelationPredicateValue::IntegerProperty { .. }
                | ResidentNullableRelationPredicateValue::Null) => {
                    Some(ResidentNullableRelationPredicate::CompareInteger {
                        left: actual,
                        operation: CompareOp::Eq,
                        right: ResidentNullableRelationPredicateValue::Integer(expected),
                    })
                }
                actual @ ResidentNullableRelationPredicateValue::StringProperty { .. } => {
                    Self::false_or_null_for_present_property(actual)
                }
                _ => None,
            };
        }
        match expected {
            Expression::Literal(ScalarValue::String(expected)) => match actual {
                actual @ (ResidentNullableRelationPredicateValue::StringProperty { .. }
                | ResidentNullableRelationPredicateValue::Null) => {
                    Some(ResidentNullableRelationPredicate::CompareString {
                        left: actual,
                        operation: CompareOp::Eq,
                        right: ResidentNullableRelationPredicateValue::String(Arc::clone(expected)),
                    })
                }
                actual @ ResidentNullableRelationPredicateValue::IntegerProperty { .. } => {
                    Self::false_or_null_for_present_property(actual)
                }
                _ => None,
            },
            Expression::Literal(ScalarValue::Null) => {
                Some(ResidentNullableRelationPredicate::Constant(None))
            }
            _ => None,
        }
    }

    fn inline_node_property_filter(
        &mut self,
        stage: usize,
        node: ResidentNullableRelationSlot,
        properties: &[(String, Expression)],
    ) -> Result<bool> {
        let mut combined = None;
        for (name, expected) in properties {
            let Some(actual) = self.node_property_value(node, name)? else {
                return Ok(false);
            };
            let Some(predicate) = Self::inline_property_equality(actual, expected) else {
                return Ok(false);
            };
            combined = Some(match combined {
                None => predicate,
                Some(left) => {
                    ResidentNullableRelationPredicate::And(Box::new(left), Box::new(predicate))
                }
            });
        }
        let Some(predicate) = combined else {
            return Ok(true);
        };
        self.predicate_program
            .filters
            .push(ResidentNullableRelationFilterStage {
                placement: ResidentNullableRelationFilterPlacement::RelationAfter {
                    stage: u16::try_from(stage).map_err(|_| {
                        Error::new(
                            ErrorCode::GpuAdmissionFailure,
                            "resident nullable inline-property filter stage exceeds u16",
                        )
                    })?,
                },
                predicate,
            });
        Ok(true)
    }

    fn relationship_property_value(
        &mut self,
        relationship: ResidentNullableRelationSlot,
        name: &str,
    ) -> Result<Option<ResidentNullableRelationPredicateValue>> {
        let Some(property) = self.catalog.property(name) else {
            // A property token absent from the whole graph is NULL on every relationship.
            return Ok(Some(ResidentNullableRelationPredicateValue::Null));
        };
        let properties =
            self.property_columns(ResidentNullableRelationBindingKind::Relationship)?;
        if properties.is_mixed(property) {
            return Ok(None);
        }
        let value = match properties.column(property) {
            Some(TypedColumn::Integer { .. }) => {
                ResidentNullableRelationPredicateValue::IntegerProperty {
                    slot: relationship,
                    kind: ResidentNullableRelationBindingKind::Relationship,
                    property,
                }
            }
            Some(TypedColumn::String { .. }) => {
                ResidentNullableRelationPredicateValue::StringProperty {
                    slot: relationship,
                    kind: ResidentNullableRelationBindingKind::Relationship,
                    property,
                }
            }
            None => ResidentNullableRelationPredicateValue::Null,
            Some(_) => return Ok(None),
        };
        Ok(Some(value))
    }

    fn false_or_null_for_present_property(
        value: ResidentNullableRelationPredicateValue,
    ) -> Option<ResidentNullableRelationPredicate> {
        match value {
            value @ ResidentNullableRelationPredicateValue::IntegerProperty { .. } => {
                Some(ResidentNullableRelationPredicate::CompareInteger {
                    left: value.clone(),
                    operation: CompareOp::NotEq,
                    right: value,
                })
            }
            value @ ResidentNullableRelationPredicateValue::StringProperty { .. } => {
                Some(ResidentNullableRelationPredicate::CompareString {
                    left: value.clone(),
                    operation: CompareOp::NotEq,
                    right: value,
                })
            }
            ResidentNullableRelationPredicateValue::Null => {
                Some(ResidentNullableRelationPredicate::Constant(None))
            }
            _ => None,
        }
    }

    fn inline_integer_literal(expression: &Expression) -> Option<i64> {
        match expression {
            Expression::Literal(ScalarValue::Integer(value)) => Some(*value),
            Expression::Unary {
                operation: UnaryOperator::Positive,
                operand,
            } => Self::inline_integer_literal(operand),
            Expression::Unary {
                operation: UnaryOperator::Negative,
                operand,
            } => Self::inline_integer_literal(operand)?.checked_neg(),
            _ => None,
        }
    }

    fn inline_relationship_property_predicate(
        &mut self,
        relationship: ResidentNullableRelationSlot,
        properties: &[(String, Expression)],
    ) -> Result<Option<ResidentNullableRelationPredicate>> {
        let mut combined = None;
        for (name, expected) in properties {
            let Some(actual) = self.relationship_property_value(relationship, name)? else {
                return Ok(None);
            };
            let Some(predicate) = Self::inline_property_equality(actual, expected) else {
                return Ok(None);
            };
            combined = Some(match combined {
                None => predicate,
                Some(left) => {
                    ResidentNullableRelationPredicate::And(Box::new(left), Box::new(predicate))
                }
            });
        }
        Ok(combined)
    }

    fn scan_pattern(
        &mut self,
        match_group: MatchGroupId,
        retain_anonymous_relationships: bool,
        optional: bool,
        pattern: &crate::cypher::Pattern,
        access: &ScanAccessPath,
    ) -> Result<bool> {
        if !self.statically_null_paths.is_empty() {
            // The static-null path adapter owns one terminal OPTIONAL pattern only.
            return Ok(false);
        }
        if !matches!(
            access,
            ScanAccessPath::Unspecified
                | ScanAccessPath::BoundVariable
                | ScanAccessPath::AllNodes
                | ScanAccessPath::Label { .. }
        ) || pattern.selector != PathSelector::All
            || pattern.mode != PathMode::DifferentRelationships
        {
            return Ok(false);
        }
        let mut fixed_path = None;
        let mut statically_null_path = None;
        if let Some(variable) = pattern.variable.as_ref() {
            let Some(step) = pattern.steps.first() else {
                return Ok(false);
            };
            if pattern.steps.len() != 1
                || step.relationship.variable_length
                || step.relationship.min_hops.is_some()
                || step.relationship.max_hops.is_some()
                || self.scope.contains_key(variable)
                || self.fixed_path_lengths.contains_key(variable)
                || self.statically_null_paths.contains_key(variable)
                || self.unmaterialized_scope_values.contains(variable)
                || pattern.start.variable.as_ref() == Some(variable)
                || step.relationship.variable.as_ref() == Some(variable)
                || step.node.variable.as_ref() == Some(variable)
            {
                return Ok(false);
            }
            if optional {
                // Path1 [1] and Path2 [3] are the one safe named-OPTIONAL erasure: a literal-null
                // node seed is proven Always by the same sealed program validator, so the path
                // cannot exist and nodes(path)/relationships(path) are exactly null. Keep every
                // reachable, typed, labelled, property-constrained, reverse, undirected, or
                // multi-hop named OPTIONAL outside this narrow adapter.
                let Some(start_variable) = pattern.start.variable.as_ref() else {
                    return Ok(false);
                };
                let Some(start) = self.scope.get(start_variable).copied() else {
                    return Ok(false);
                };
                if start.kind != ResidentNullableRelationBindingKind::Node
                    || !self.binding_is_statically_always_null(start)?
                    || pattern.start.property_predicate_present
                    || !pattern.start.labels.is_empty()
                    || !pattern.start.properties.is_empty()
                    || step.relationship.variable.is_none()
                    || !step.relationship.types.is_empty()
                    || !step.relationship.properties.is_empty()
                    || step.relationship.direction != Direction::Outgoing
                    || step.node.variable.is_some()
                    || step.node.property_predicate_present
                    || !step.node.labels.is_empty()
                    || !step.node.properties.is_empty()
                {
                    return Ok(false);
                }
                statically_null_path = Some((variable.clone(), start));
            } else {
                fixed_path = Some((variable.clone(), 1_i64));
            }
        }
        match self.uniqueness_group {
            Some(current) if match_group < current => {
                return Err(Error::internal(
                    "resident nullable compiler observed a reopened MATCH group",
                ));
            }
            Some(current) if match_group > current => {
                self.uniqueness_group = Some(match_group);
                self.uniqueness_relationships.clear();
            }
            None => self.uniqueness_group = Some(match_group),
            Some(_) => {}
        }
        let mode = if optional {
            ResidentNullableRelationMatchMode::Optional
        } else {
            ResidentNullableRelationMatchMode::Mandatory
        };

        let Some(first_step) = pattern.steps.first() else {
            let Some(variable) = pattern.start.variable.as_ref() else {
                return Ok(false);
            };
            if self.scope.contains_key(variable) || optional && !pattern.start.properties.is_empty()
            {
                // Filtering an existing node binding is a distinct stage and remains fail-closed.
                return Ok(false);
            }
            let stage = self.stages.len();
            let output = self.allocate_slot()?;
            self.stages.push(ResidentNullableRelationStage::NodeScan {
                mode,
                output,
                labels: nullable_node_domain(&pattern.start.labels, self.catalog),
            });
            self.scope.insert(
                variable.clone(),
                NullableRelationBinding {
                    slot: output,
                    kind: ResidentNullableRelationBindingKind::Node,
                },
            );
            if pattern.start.property_predicate_present
                && !self.inline_node_property_filter(stage, output, &pattern.start.properties)?
            {
                return Ok(false);
            }
            return Ok(true);
        };

        let first_stage = self.stages.len();
        let start_binding = pattern
            .start
            .variable
            .as_ref()
            .and_then(|variable| self.scope.get(variable))
            .copied();
        let end_binding = first_step
            .node
            .variable
            .as_ref()
            .and_then(|variable| self.scope.get(variable))
            .copied();
        let existing_relationship_binding = (pattern.steps.len() == 1)
            .then(|| {
                first_step
                    .relationship
                    .variable
                    .as_ref()
                    .and_then(|variable| self.scope.get(variable))
                    .copied()
            })
            .flatten()
            .filter(|binding| binding.kind == ResidentNullableRelationBindingKind::Relationship);
        let (mut source, reverse_single_hop, first_source_labels) = if let Some(source) =
            start_binding
        {
            if source.kind != ResidentNullableRelationBindingKind::Node
                || pattern.start.property_predicate_present
            {
                // Property constraints on an already-bound anchor still need an explicit
                // candidate filter. Static labels are sealed directly into the first expansion.
                return Ok(false);
            }
            (
                source,
                false,
                nullable_node_domain(&pattern.start.labels, self.catalog),
            )
        } else if pattern.steps.len() == 1
            && let Some(source) = end_binding
        {
            if source.kind != ResidentNullableRelationBindingKind::Node {
                return Ok(false);
            }
            (
                source,
                true,
                nullable_node_domain(&first_step.node.labels, self.catalog),
            )
        } else if !optional {
            // A complete mandatory pattern starts from the implicit one-row unit relation. Seed
            // its syntactic start node as an ordinary resident scan, then let the expansions own
            // relationship identity, direction, and endpoint materialization.
            let slot = self.allocate_slot()?;
            let source = NullableRelationBinding {
                slot,
                kind: ResidentNullableRelationBindingKind::Node,
            };
            self.stages.push(ResidentNullableRelationStage::NodeScan {
                mode: ResidentNullableRelationMatchMode::Mandatory,
                output: slot,
                labels: nullable_node_domain(&pattern.start.labels, self.catalog),
            });
            if let Some(variable) = &pattern.start.variable {
                self.scope.insert(variable.clone(), source);
            }
            if !pattern.start.properties.is_empty() {
                if !self.inline_node_property_filter(
                    first_stage,
                    source.slot,
                    &pattern.start.properties,
                )? {
                    return Ok(false);
                }
            }
            (source, false, ResidentNullableNodeDomain::Any)
        } else if let Some(relationship) = existing_relationship_binding {
            if pattern.start.property_predicate_present {
                return Ok(false);
            }
            // The live relationship is the only anchor. Seed candidate start nodes from the
            // parent row and retain only the endpoint allowed by the pattern direction. The
            // enclosing atomic OPTIONAL group owns null extension if no complete pattern exists.
            let slot = self.allocate_slot()?;
            let source = NullableRelationBinding {
                slot,
                kind: ResidentNullableRelationBindingKind::Node,
            };
            self.stages.push(ResidentNullableRelationStage::NodeScan {
                mode: ResidentNullableRelationMatchMode::Optional,
                output: slot,
                labels: nullable_node_domain(&pattern.start.labels, self.catalog),
            });
            if let Some(variable) = &pattern.start.variable {
                self.scope.insert(variable.clone(), source);
            }
            self.predicate_program
                .filters
                .push(ResidentNullableRelationFilterStage {
                    placement: ResidentNullableRelationFilterPlacement::OptionalCandidates {
                        stage: u16::try_from(first_stage).map_err(|_| {
                            Error::new(
                                ErrorCode::GpuAdmissionFailure,
                                "resident nullable relationship seed stage exceeds u16",
                            )
                        })?,
                    },
                    predicate: ResidentNullableRelationPredicate::RelationshipEndpoint {
                        relationship: relationship.slot,
                        node: slot,
                        endpoint: match first_step.relationship.direction {
                            Direction::Outgoing => ResidentNullableRelationshipEndpoint::Source,
                            Direction::Incoming => ResidentNullableRelationshipEndpoint::Target,
                            Direction::Undirected => ResidentNullableRelationshipEndpoint::Either,
                        },
                    },
                });
            (source, false, ResidentNullableNodeDomain::Any)
        } else if pattern.variable.is_none()
            && pattern.steps.len() == 1
            && pattern.start.variable.is_none()
            && !pattern.start.property_predicate_present
            && pattern.start.labels.is_empty()
            && pattern.start.properties.is_empty()
            && first_step.relationship.variable.is_some()
            && !first_step.relationship.variable_length
            && first_step.relationship.min_hops.is_none()
            && first_step.relationship.max_hops.is_none()
            && first_step.relationship.types.is_empty()
            && first_step.relationship.properties.is_empty()
            && first_step.relationship.direction == Direction::Outgoing
            && first_step.node.variable.is_none()
            && !first_step.node.property_predicate_present
            && first_step.node.labels.is_empty()
            && first_step.node.properties.is_empty()
        {
            // Graph6 [6]/[7]: seed the exact anonymous one-hop OPTIONAL on-device. The node scan
            // and expansion form one atomic OPTIONAL group, so candidate start nodes never emit
            // independent null extensions; a graph with no complete relationship produces one
            // null-extended unit row, including when the graph itself has no nodes.
            let slot = self.allocate_slot()?;
            let source = NullableRelationBinding {
                slot,
                kind: ResidentNullableRelationBindingKind::Node,
            };
            self.stages.push(ResidentNullableRelationStage::NodeScan {
                mode: ResidentNullableRelationMatchMode::Optional,
                output: slot,
                labels: ResidentNullableNodeDomain::Any,
            });
            (source, false, ResidentNullableNodeDomain::Any)
        } else {
            return Ok(false);
        };

        let mut optional_relationship_predicates = Vec::new();
        for (step_index, step) in pattern.steps.iter().enumerate() {
            let variable_length = step.relationship.variable_length
                || step.relationship.min_hops.is_some()
                || step.relationship.max_hops.is_some();
            if variable_length
                && (!optional
                    || reverse_single_hop
                    || step.relationship.variable.is_some()
                    || !step.relationship.properties.is_empty()
                    || !self.binding_is_statically_always_null(source)?)
            {
                // A variable traversal is semantically irrelevant only when the exact preceding
                // staged relation proves its required source is always null. In that case every
                // hop range has the same OPTIONAL result: one retained parent with introduced
                // bindings null. All reachable variable traversals remain fail-closed here and
                // use the dedicated native variable-path route instead.
                return Ok(false);
            }
            if step.node.property_predicate_present || !step.node.properties.is_empty() {
                return Ok(false);
            }
            let (target_name, target_labels, direction) = if reverse_single_hop {
                (
                    pattern.start.variable.as_ref(),
                    &pattern.start.labels,
                    match step.relationship.direction {
                        Direction::Outgoing => ResidentDirection::Incoming,
                        Direction::Incoming => ResidentDirection::Outgoing,
                        Direction::Undirected => ResidentDirection::Undirected,
                    },
                )
            } else {
                (
                    step.node.variable.as_ref(),
                    &step.node.labels,
                    match step.relationship.direction {
                        Direction::Outgoing => ResidentDirection::Outgoing,
                        Direction::Incoming => ResidentDirection::Incoming,
                        Direction::Undirected => ResidentDirection::Undirected,
                    },
                )
            };
            let relationship = match step.relationship.variable.as_ref() {
                Some(variable) => match self.scope.get(variable).copied() {
                    Some(binding)
                        if binding.kind == ResidentNullableRelationBindingKind::Relationship =>
                    {
                        Some((None, binding.slot, false))
                    }
                    Some(_) => return Ok(false),
                    None => Some((Some(variable.clone()), self.allocate_slot()?, true)),
                },
                None if retain_anonymous_relationships
                    || !step.relationship.properties.is_empty() =>
                {
                    Some((None, self.allocate_slot()?, true))
                }
                None => None,
            };
            let (target, target_binding, introduced_target) = match target_name {
                Some(variable) => match self.scope.get(variable).copied() {
                    Some(binding) if binding.kind == ResidentNullableRelationBindingKind::Node => (
                        ResidentNullableRelationTarget::Existing(binding.slot),
                        binding,
                        None,
                    ),
                    Some(_) => return Ok(false),
                    None => {
                        let slot = self.allocate_slot()?;
                        let binding = NullableRelationBinding {
                            slot,
                            kind: ResidentNullableRelationBindingKind::Node,
                        };
                        (
                            ResidentNullableRelationTarget::Introduce(slot),
                            binding,
                            Some((variable.clone(), binding)),
                        )
                    }
                },
                None => {
                    let slot = self.allocate_slot()?;
                    let binding = NullableRelationBinding {
                        slot,
                        kind: ResidentNullableRelationBindingKind::Node,
                    };
                    (
                        ResidentNullableRelationTarget::Introduce(slot),
                        binding,
                        None,
                    )
                }
            };
            let stage = self.stages.len();
            let relationship_slot = relationship.as_ref().map(|(_, slot, _)| *slot);
            self.stages.push(ResidentNullableRelationStage::Expand {
                mode,
                uniqueness_group: match_group.get(),
                source: source.slot,
                source_labels: if step_index == 0 {
                    first_source_labels.clone()
                } else {
                    ResidentNullableNodeDomain::Any
                },
                relationship: relationship_slot,
                different_from: self.uniqueness_relationships.iter().copied().collect(),
                target,
                direction,
                relationship_types: nullable_relationship_domain(
                    &step.relationship.types,
                    self.catalog,
                ),
                target_labels: nullable_node_domain(target_labels, self.catalog),
            });
            if let Some((_, slot, _)) = relationship.as_ref() {
                self.uniqueness_relationships.insert(*slot);
            }
            if !step.relationship.properties.is_empty() {
                let Some(relationship) = relationship_slot else {
                    return Err(Error::internal(
                        "resident nullable relationship predicate omitted its relationship slot",
                    ));
                };
                let Some(predicate) = self.inline_relationship_property_predicate(
                    relationship,
                    &step.relationship.properties,
                )?
                else {
                    return Ok(false);
                };
                if optional {
                    optional_relationship_predicates.push((stage, predicate));
                } else {
                    self.predicate_program
                        .filters
                        .push(ResidentNullableRelationFilterStage {
                            placement: ResidentNullableRelationFilterPlacement::RelationAfter {
                                stage: u16::try_from(stage).map_err(|_| {
                                    Error::new(
                                        ErrorCode::GpuAdmissionFailure,
                                        "resident nullable relationship-property filter stage exceeds u16",
                                    )
                                })?,
                            },
                            predicate,
                        });
                }
            }
            if let Some((Some(variable), slot, true)) = relationship {
                self.scope.insert(
                    variable,
                    NullableRelationBinding {
                        slot,
                        kind: ResidentNullableRelationBindingKind::Relationship,
                    },
                );
            }
            if let Some((variable, binding)) = introduced_target {
                self.scope.insert(variable, binding);
            }
            source = target_binding;
            if reverse_single_hop && step_index != 0 {
                return Ok(false);
            }
        }
        let optional_group = if optional && self.stages.len().saturating_sub(first_stage) > 1 {
            let group =
                u16::try_from(self.predicate_program.optional_groups.len()).map_err(|_| {
                    Error::new(
                        ErrorCode::GpuAdmissionFailure,
                        "resident nullable OPTIONAL group index exceeds u16",
                    )
                })?;
            self.predicate_program
                .optional_groups
                .push(ResidentNullableRelationOptionalGroup {
                    first_stage: u16::try_from(first_stage).map_err(|_| {
                        Error::new(
                            ErrorCode::GpuAdmissionFailure,
                            "resident nullable OPTIONAL group start exceeds u16",
                        )
                    })?,
                    last_stage: u16::try_from(self.stages.len() - 1).map_err(|_| {
                        Error::new(
                            ErrorCode::GpuAdmissionFailure,
                            "resident nullable OPTIONAL group end exceeds u16",
                        )
                    })?,
                });
            Some(group)
        } else {
            None
        };
        for (stage, predicate) in optional_relationship_predicates {
            let placement = match optional_group {
                Some(group) => {
                    ResidentNullableRelationFilterPlacement::OptionalGroupCandidates { group }
                }
                None => ResidentNullableRelationFilterPlacement::OptionalCandidates {
                    stage: u16::try_from(stage).map_err(|_| {
                        Error::new(
                            ErrorCode::GpuAdmissionFailure,
                            "resident nullable relationship-property candidate stage exceeds u16",
                        )
                    })?,
                },
            };
            self.predicate_program
                .filters
                .push(ResidentNullableRelationFilterStage {
                    placement,
                    predicate,
                });
        }
        if let Some((variable, length)) = fixed_path {
            self.fixed_path_lengths.insert(variable.clone(), length);
            self.unmaterialized_scope_values.insert(variable);
        }
        if let Some((variable, binding)) = statically_null_path {
            self.statically_null_paths.insert(variable.clone(), binding);
            self.unmaterialized_scope_values.insert(variable);
        }
        Ok(true)
    }

    fn binding_node_domain(
        &self,
        binding: NullableRelationBinding,
    ) -> Option<ResidentNullableNodeDomain> {
        if binding.kind != ResidentNullableRelationBindingKind::Node {
            return None;
        }
        let mut slot = binding.slot;
        for _ in 0..=self.stages.len() {
            let mut projected_source = None;
            for stage in self.stages.iter().rev() {
                match stage {
                    ResidentNullableRelationStage::NodeScan { output, labels, .. }
                        if *output == slot =>
                    {
                        return Some(labels.clone());
                    }
                    ResidentNullableRelationStage::Expand {
                        target: ResidentNullableRelationTarget::Introduce(output),
                        target_labels,
                        ..
                    } if *output == slot => return Some(target_labels.clone()),
                    ResidentNullableRelationStage::ScopeProject { bindings } => {
                        if let Some(binding) =
                            bindings.iter().find(|binding| binding.output == slot)
                        {
                            projected_source = Some(binding.source);
                            break;
                        }
                    }
                    _ => {}
                }
            }
            slot = projected_source?;
        }
        None
    }

    fn binding_relationship_domain(
        &self,
        binding: NullableRelationBinding,
    ) -> Option<ResidentNullableRelationshipDomain> {
        if binding.kind != ResidentNullableRelationBindingKind::Relationship {
            return None;
        }
        let mut slot = binding.slot;
        for _ in 0..=self.stages.len() {
            let mut projected_source = None;
            for stage in self.stages.iter().rev() {
                match stage {
                    ResidentNullableRelationStage::Expand {
                        relationship: Some(output),
                        relationship_types,
                        ..
                    } if *output == slot => return Some(relationship_types.clone()),
                    ResidentNullableRelationStage::ScopeProject { bindings } => {
                        if let Some(binding) =
                            bindings.iter().find(|binding| binding.output == slot)
                        {
                            projected_source = Some(binding.source);
                            break;
                        }
                    }
                    _ => {}
                }
            }
            slot = projected_source?;
        }
        None
    }

    fn node_domain_contains(
        domain: &ResidentNullableNodeDomain,
        labels: &[crate::types::LabelId],
    ) -> bool {
        match domain {
            ResidentNullableNodeDomain::Any => true,
            ResidentNullableNodeDomain::Known(required) => {
                required.iter().all(|label| labels.contains(label))
            }
            ResidentNullableNodeDomain::KnownEmpty => false,
        }
    }

    fn mixed_node_property_is_string_in_domain(
        &self,
        binding: NullableRelationBinding,
        property: crate::types::PropertyId,
    ) -> bool {
        let Some(domain) = self.binding_node_domain(binding) else {
            return false;
        };
        self.graph
            .nodes()
            .filter(|node| Self::node_domain_contains(&domain, node.labels()))
            .all(|node| {
                node.property(property)
                    .is_none_or(|value| matches!(value, ScalarValue::String(_)))
            })
    }

    fn mixed_node_string_property_value(
        &mut self,
        expression: &Expression,
        require_string_domain: bool,
    ) -> Result<Option<ResidentNullableRelationPredicateValue>> {
        let Expression::Property(source, name) = expression else {
            return Ok(None);
        };
        let Expression::Variable(variable) = source.as_ref() else {
            return Ok(None);
        };
        let Some(binding) = self.scope.get(variable).copied() else {
            return Ok(None);
        };
        if binding.kind != ResidentNullableRelationBindingKind::Node {
            return Ok(None);
        }
        let Some(property) = self.catalog.property(name) else {
            return Ok(None);
        };
        if !self
            .property_columns(ResidentNullableRelationBindingKind::Node)?
            .is_mixed(property)
            || require_string_domain
                && !self.mixed_node_property_is_string_in_domain(binding, property)
        {
            return Ok(None);
        }
        Ok(Some(
            ResidentNullableRelationPredicateValue::StringProperty {
                slot: binding.slot,
                kind: binding.kind,
                property,
            },
        ))
    }

    fn predicate_comparison_value(
        &mut self,
        expression: &Expression,
    ) -> Result<Option<ResidentNullableRelationPredicateValue>> {
        if let Some(value) = self.predicate_value(expression)? {
            return Ok(Some(value));
        }
        self.mixed_node_string_property_value(expression, true)
    }

    fn mixed_non_null_guarded_string_comparison(
        &mut self,
        expression: &Expression,
    ) -> Result<Option<ResidentNullableRelationPredicate>> {
        let Expression::Binary {
            left: comparison,
            operation: BinaryOperator::Or,
            right: guard,
        } = expression
        else {
            return Ok(None);
        };
        let Expression::Binary {
            left,
            operation,
            right,
        } = comparison.as_ref()
        else {
            return Ok(None);
        };
        if !matches!(
            operation,
            BinaryOperator::Equal
                | BinaryOperator::NotEqual
                | BinaryOperator::Less
                | BinaryOperator::LessOrEqual
                | BinaryOperator::Greater
                | BinaryOperator::GreaterOrEqual
        ) {
            return Ok(None);
        }
        let Expression::IsNull {
            expression: guarded,
            negated: true,
        } = guard.as_ref()
        else {
            return Ok(None);
        };
        let guarded_is_comparison_property = (left.as_ref() == guarded.as_ref()
            && matches!(right.as_ref(), Expression::Literal(ScalarValue::String(_))))
            || (right.as_ref() == guarded.as_ref()
                && matches!(left.as_ref(), Expression::Literal(ScalarValue::String(_))));
        if !guarded_is_comparison_property {
            return Ok(None);
        }
        let Some(value) = self.mixed_node_string_property_value(guarded, false)? else {
            return Ok(None);
        };
        // For a non-null property the guard is true, so the OR is true regardless of the
        // comparison. For a null property the comparison is null and the guard is false, so the
        // exact result is null (not false). Preserve that distinction for the device receipt.
        Ok(Some(ResidentNullableRelationPredicate::Or(
            Box::new(ResidentNullableRelationPredicate::Constant(None)),
            Box::new(ResidentNullableRelationPredicate::IsNull {
                value,
                negated: true,
            }),
        )))
    }

    fn predicate_value(
        &mut self,
        expression: &Expression,
    ) -> Result<Option<ResidentNullableRelationPredicateValue>> {
        Ok(match expression {
            Expression::Literal(ScalarValue::Null) => {
                Some(ResidentNullableRelationPredicateValue::Null)
            }
            Expression::Literal(ScalarValue::Boolean(value)) => {
                Some(ResidentNullableRelationPredicateValue::Boolean(*value))
            }
            Expression::Literal(ScalarValue::Integer(value)) => {
                Some(ResidentNullableRelationPredicateValue::Integer(*value))
            }
            Expression::Literal(ScalarValue::String(value)) => Some(
                ResidentNullableRelationPredicateValue::String(Arc::clone(value)),
            ),
            Expression::Parameter(name) => match self.parameters.get(name) {
                Some(ResultValue::Scalar(ScalarValue::Null)) => {
                    Some(ResidentNullableRelationPredicateValue::Null)
                }
                Some(ResultValue::Scalar(ScalarValue::Boolean(value))) => {
                    Some(ResidentNullableRelationPredicateValue::Boolean(*value))
                }
                Some(ResultValue::Scalar(ScalarValue::Integer(value))) => {
                    Some(ResidentNullableRelationPredicateValue::Integer(*value))
                }
                Some(ResultValue::Scalar(ScalarValue::String(value))) => Some(
                    ResidentNullableRelationPredicateValue::String(Arc::clone(value)),
                ),
                _ => None,
            },
            Expression::Variable(variable) => {
                let Some(binding) = self.scope.get(variable).copied() else {
                    return Ok(None);
                };
                Some(ResidentNullableRelationPredicateValue::Binding {
                    slot: binding.slot,
                    kind: binding.kind,
                })
            }
            Expression::Property(source, name) => {
                let Expression::Variable(variable) = source.as_ref() else {
                    return Ok(None);
                };
                let Some(binding) = self.scope.get(variable).copied() else {
                    return Ok(None);
                };
                let Some(property) = self.catalog.property(name) else {
                    return Ok(Some(ResidentNullableRelationPredicateValue::Null));
                };
                match binding.kind {
                    ResidentNullableRelationBindingKind::Node => {
                        if self.graph.node_property_is_integer(property) {
                            Some(ResidentNullableRelationPredicateValue::IntegerProperty {
                                slot: binding.slot,
                                kind: binding.kind,
                                property,
                            })
                        } else if self.graph.node_property_is_string(property) {
                            Some(ResidentNullableRelationPredicateValue::StringProperty {
                                slot: binding.slot,
                                kind: binding.kind,
                                property,
                            })
                        } else {
                            return Ok(None);
                        }
                    }
                    ResidentNullableRelationBindingKind::Relationship => {
                        return self.relationship_property_value(binding.slot, name);
                    }
                }
            }
            Expression::Function {
                name,
                distinct: false,
                arguments,
            } if name.len() == 1
                && name[0].eq_ignore_ascii_case("length")
                && arguments.len() == 1 =>
            {
                let Expression::Variable(variable) = &arguments[0] else {
                    return Ok(None);
                };
                self.fixed_path_lengths
                    .get(variable)
                    .copied()
                    .map(ResidentNullableRelationPredicateValue::Integer)
            }
            _ => None,
        })
    }

    fn predicate(
        &mut self,
        expression: &Expression,
    ) -> Result<Option<ResidentNullableRelationPredicate>> {
        if let Some(predicate) = self.mixed_non_null_guarded_string_comparison(expression)? {
            return Ok(Some(predicate));
        }
        if let Some((source, names)) = expression.entity_label_predicate_parts() {
            let Expression::Variable(variable) = source else {
                return Ok(None);
            };
            let Some(binding) = self.scope.get(variable).copied() else {
                return Ok(None);
            };
            if binding.kind != ResidentNullableRelationBindingKind::Node {
                return Ok(None);
            }
            let labels = names
                .iter()
                .filter_map(|name| match name {
                    Expression::Literal(ScalarValue::String(name)) => Some(name.to_string()),
                    _ => None,
                })
                .collect::<Vec<_>>();
            if labels.len() != names.len() {
                return Ok(None);
            }
            return Ok(Some(ResidentNullableRelationPredicate::HasLabels {
                node: binding.slot,
                labels: nullable_node_domain(&labels, self.catalog),
            }));
        }
        Ok(match expression {
            Expression::Literal(ScalarValue::Boolean(value)) => {
                Some(ResidentNullableRelationPredicate::Constant(Some(*value)))
            }
            Expression::Literal(ScalarValue::Null) => {
                Some(ResidentNullableRelationPredicate::Constant(None))
            }
            Expression::IsNull {
                expression,
                negated,
            } => {
                let value = match self.predicate_value(expression)? {
                    Some(value) => Some(value),
                    None => self.mixed_node_string_property_value(expression, true)?,
                };
                value.map(|value| ResidentNullableRelationPredicate::IsNull {
                    value,
                    negated: *negated,
                })
            }
            Expression::Unary {
                operation: UnaryOperator::Not,
                operand,
            } => self
                .predicate(operand)?
                .map(|predicate| ResidentNullableRelationPredicate::Not(Box::new(predicate))),
            Expression::Binary {
                left,
                operation: BinaryOperator::And,
                right,
            } => match (self.predicate(left)?, self.predicate(right)?) {
                (Some(left), Some(right)) => Some(ResidentNullableRelationPredicate::And(
                    Box::new(left),
                    Box::new(right),
                )),
                _ => None,
            },
            Expression::Binary {
                left,
                operation: BinaryOperator::Or,
                right,
            } => match (self.predicate(left)?, self.predicate(right)?) {
                (Some(left), Some(right)) => Some(ResidentNullableRelationPredicate::Or(
                    Box::new(left),
                    Box::new(right),
                )),
                _ => None,
            },
            Expression::Binary {
                left,
                operation,
                right,
            } => {
                let operation = match operation {
                    BinaryOperator::Equal => Some(CompareOp::Eq),
                    BinaryOperator::NotEqual => Some(CompareOp::NotEq),
                    BinaryOperator::Less => Some(CompareOp::Less),
                    BinaryOperator::LessOrEqual => Some(CompareOp::LessOrEqual),
                    BinaryOperator::Greater => Some(CompareOp::Greater),
                    BinaryOperator::GreaterOrEqual => Some(CompareOp::GreaterOrEqual),
                    _ => None,
                };
                match (
                    operation,
                    self.predicate_comparison_value(left)?,
                    self.predicate_comparison_value(right)?,
                ) {
                    (Some(operation), Some(left), Some(right)) => {
                        let left_string = matches!(
                            &left,
                            ResidentNullableRelationPredicateValue::String(_)
                                | ResidentNullableRelationPredicateValue::StringProperty { .. }
                        );
                        let right_string = matches!(
                            &right,
                            ResidentNullableRelationPredicateValue::String(_)
                                | ResidentNullableRelationPredicateValue::StringProperty { .. }
                        );
                        let left_null =
                            matches!(&left, ResidentNullableRelationPredicateValue::Null);
                        let right_null =
                            matches!(&right, ResidentNullableRelationPredicateValue::Null);
                        if (left_string || right_string)
                            && (left_string || left_null)
                            && (right_string || right_null)
                        {
                            Some(ResidentNullableRelationPredicate::CompareString {
                                left,
                                operation,
                                right,
                            })
                        } else if !left_string && !right_string {
                            Some(ResidentNullableRelationPredicate::CompareInteger {
                                left,
                                operation,
                                right,
                            })
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            }
            _ => None,
        })
    }

    fn push_filter(
        &mut self,
        placement: ResidentNullableRelationFilterPlacement,
        expression: &Expression,
    ) -> Result<bool> {
        let Some(predicate) = self.predicate(expression)? else {
            return Ok(false);
        };
        self.predicate_program
            .filters
            .push(ResidentNullableRelationFilterStage {
                placement,
                predicate,
            });
        Ok(true)
    }

    fn optional_candidate_filter(
        &mut self,
        first_stage: usize,
        last_stage: usize,
        expression: &Expression,
    ) -> Result<bool> {
        if !self.statically_null_paths.is_empty() {
            return Ok(false);
        }
        let matching_group = self
            .predicate_program
            .optional_groups
            .iter()
            .position(|group| {
                usize::from(group.first_stage) == first_stage
                    && usize::from(group.last_stage) == last_stage
            });
        let placement = if let Some(group) = matching_group {
            ResidentNullableRelationFilterPlacement::OptionalGroupCandidates {
                group: u16::try_from(group).map_err(|_| {
                    Error::new(
                        ErrorCode::GpuAdmissionFailure,
                        "resident nullable OPTIONAL group filter index exceeds u16",
                    )
                })?,
            }
        } else {
            ResidentNullableRelationFilterPlacement::OptionalCandidates {
                stage: u16::try_from(last_stage).map_err(|_| {
                    Error::new(
                        ErrorCode::GpuAdmissionFailure,
                        "resident nullable candidate filter stage exceeds u16",
                    )
                })?,
            }
        };
        self.push_filter(placement, expression)
    }

    fn relation_filter(&mut self, expression: &Expression) -> Result<bool> {
        if !self.statically_null_paths.is_empty() {
            return Ok(false);
        }
        let Some(stage) = self.stages.len().checked_sub(1) else {
            return Ok(false);
        };
        let Some(predicate) = self.predicate(expression)? else {
            return Ok(false);
        };
        let stage = self.earliest_mandatory_filter_stage(&predicate, stage);
        self.predicate_program
            .filters
            .push(ResidentNullableRelationFilterStage {
                placement: ResidentNullableRelationFilterPlacement::RelationAfter {
                    stage: u16::try_from(stage).map_err(|_| {
                        Error::new(
                            ErrorCode::GpuAdmissionFailure,
                            "resident nullable relation filter stage exceeds u16",
                        )
                    })?,
                },
                predicate,
            });
        Ok(true)
    }

    /// Pushes an ordinary WHERE only through mandatory inner-join graph stages. Work then tracks
    /// the rows named by the predicate rather than unrelated later candidates; OPTIONAL and scope
    /// boundaries retain their original three-valued and cardinality semantics.
    fn earliest_mandatory_filter_stage(
        &self,
        predicate: &ResidentNullableRelationPredicate,
        final_stage: usize,
    ) -> usize {
        let mut required = BTreeSet::new();
        nullable_predicate_slots(predicate, &mut required);
        if required.is_empty() || !self.predicate_program.optional_groups.is_empty() {
            return final_stage;
        }
        let mut live = BTreeSet::new();
        for (stage_index, stage) in self.stages.iter().enumerate().take(final_stage + 1) {
            let mandatory = match stage {
                ResidentNullableRelationStage::NodeScan { mode, output, .. } => {
                    live.insert(*output);
                    *mode == ResidentNullableRelationMatchMode::Mandatory
                }
                ResidentNullableRelationStage::Expand {
                    mode,
                    relationship,
                    target,
                    ..
                } => {
                    if let Some(relationship) = relationship {
                        live.insert(*relationship);
                    }
                    if let ResidentNullableRelationTarget::Introduce(target) = target {
                        live.insert(*target);
                    }
                    *mode == ResidentNullableRelationMatchMode::Mandatory
                }
                ResidentNullableRelationStage::ScopeProject { bindings } => {
                    live = bindings.iter().map(|binding| binding.output).collect();
                    false
                }
                ResidentNullableRelationStage::FinalProject { .. } => false,
            };
            if required.is_subset(&live)
                && mandatory
                && self.stages[stage_index..=final_stage].iter().all(|stage| {
                    matches!(
                        stage,
                        ResidentNullableRelationStage::NodeScan {
                            mode: ResidentNullableRelationMatchMode::Mandatory,
                            ..
                        } | ResidentNullableRelationStage::Expand {
                            mode: ResidentNullableRelationMatchMode::Mandatory,
                            ..
                        }
                    )
                })
            {
                return stage_index;
            }
        }
        final_stage
    }

    /// Represent the exact leading `WITH null AS <node>` correlation seed as one device-native
    /// always-null node binding. An OPTIONAL scan over a `KnownEmpty` domain maps the unit
    /// relation to exactly one null row on CPU and Metal, so no host literal row or fallback is
    /// needed. This adapter is intentionally limited to the first, single-item scope boundary;
    /// the later OPTIONAL pattern must still consume the binding before the overall compiler can
    /// admit the query.
    fn static_null_node_scope(
        &mut self,
        keep_scope: bool,
        projection: &Projection,
    ) -> Result<bool> {
        if keep_scope
            || projection.distinct
            || projection.items.len() != 1
            || !self.stages.is_empty()
            || !self.scope.is_empty()
            || !self.fixed_path_lengths.is_empty()
            || !self.statically_null_paths.is_empty()
            || !self.unmaterialized_scope_values.is_empty()
        {
            return Ok(false);
        }
        let item = &projection.items[0];
        if !matches!(item.expression, Expression::Literal(ScalarValue::Null)) {
            return Ok(false);
        }
        let Some(variable) = item.alias.as_ref() else {
            return Ok(false);
        };
        if variable.is_empty() {
            return Ok(false);
        }
        let output = self.allocate_slot()?;
        self.stages.push(ResidentNullableRelationStage::NodeScan {
            mode: ResidentNullableRelationMatchMode::Optional,
            output,
            labels: ResidentNullableNodeDomain::KnownEmpty,
        });
        self.scope.insert(
            variable.clone(),
            NullableRelationBinding {
                slot: output,
                kind: ResidentNullableRelationBindingKind::Node,
            },
        );
        self.static_null_node_seeded = true;
        Ok(true)
    }

    fn keep_scope_projection(&mut self, projection: &Projection) -> Result<bool> {
        if !self.statically_null_paths.is_empty()
            || projection.distinct
            || !self.unmaterialized_scope_values.is_empty()
                && projection
                    .items
                    .iter()
                    .any(|item| matches!(item.expression, Expression::Star))
        {
            return Ok(false);
        }
        let mut aliases = Vec::new();
        for item in &projection.items {
            match &item.expression {
                Expression::Star if item.alias.is_none() => {}
                Expression::Variable(variable) => {
                    let Some(binding) = self.scope.get(variable).copied() else {
                        return Ok(false);
                    };
                    if let Some(alias) = &item.alias {
                        aliases.push((alias.clone(), binding));
                    }
                }
                _ => return Ok(false),
            }
        }
        for (alias, binding) in aliases {
            self.scope.insert(alias, binding);
        }
        Ok(true)
    }

    fn expanded_projection(
        &self,
        projection: &Projection,
    ) -> Option<Vec<(String, NullableRelationBinding)>> {
        if !self.unmaterialized_scope_values.is_empty()
            && projection
                .items
                .iter()
                .any(|item| matches!(item.expression, Expression::Star))
        {
            return None;
        }
        let mut expanded = Vec::new();
        let mut names = std::collections::BTreeSet::new();
        for (index, item) in projection.items.iter().enumerate() {
            match &item.expression {
                Expression::Star if item.alias.is_none() => {
                    for (name, binding) in &self.scope {
                        if !names.insert(name.clone()) {
                            return None;
                        }
                        expanded.push((name.clone(), *binding));
                    }
                }
                Expression::Variable(variable) => {
                    let binding = self.scope.get(variable).copied()?;
                    let name = item.column_name(index);
                    if !names.insert(name.clone()) {
                        return None;
                    }
                    expanded.push((name, binding));
                }
                Expression::Function {
                    name,
                    distinct: false,
                    arguments,
                } if name.len() == 1 && name[0].eq_ignore_ascii_case("coalesce") => {
                    let binding = self.known_null_node_coalesce(arguments)?;
                    let name = item.column_name(index);
                    if !names.insert(name.clone()) {
                        return None;
                    }
                    expanded.push((name, binding));
                }
                _ => return None,
            }
        }
        (!expanded.is_empty()).then_some(expanded)
    }

    /// A `KnownEmpty` node domain has no canonical row at this sealed catalog generation. A
    /// mandatory producer therefore eliminates the row, while an OPTIONAL producer can only
    /// publish the null sentinel. Consequently, on every relation row that can reach this WITH
    /// boundary, every such argument is exactly Cypher null and `coalesce` is also null. Reusing
    /// one argument slot preserves that value and lets the ordinary scope projection keep owning
    /// the alias; no host expression or fabricated result is introduced.
    fn known_null_node_coalesce(
        &self,
        arguments: &[Expression],
    ) -> Option<NullableRelationBinding> {
        let mut selected = None;
        for argument in arguments {
            let Expression::Variable(variable) = argument else {
                return None;
            };
            let binding = self.scope.get(variable).copied()?;
            if binding.kind != ResidentNullableRelationBindingKind::Node
                || !self
                    .binding_node_domain(binding)
                    .is_some_and(|domain| domain.is_known_empty())
            {
                return None;
            }
            selected.get_or_insert(binding);
        }
        selected
    }

    /// `type(null)` can reuse a relationship-type output only when the staged relationship domain
    /// itself proves that no canonical relationship can exist. A mandatory producer then emits no
    /// row, while an OPTIONAL producer publishes only the relationship null sentinel. Domains that
    /// are merely empty in the current data, dynamically typed, or target-empty stay fail-closed.
    fn known_null_relationship(&self) -> Option<NullableRelationBinding> {
        self.scope.values().copied().find(|binding| {
            binding.kind == ResidentNullableRelationBindingKind::Relationship
                && self
                    .binding_relationship_domain(*binding)
                    .is_some_and(|domain| domain.is_known_empty())
        })
    }

    fn statically_null_path_argument(
        &self,
        arguments: &[Expression],
    ) -> Option<NullableRelationBinding> {
        match arguments {
            [Expression::Variable(variable)] => self.statically_null_paths.get(variable).copied(),
            [Expression::Literal(ScalarValue::Null)] if self.statically_null_paths.len() == 1 => {
                self.statically_null_paths.values().next().copied()
            }
            _ => None,
        }
    }

    fn is_exact_statically_null_path_projection(&self, projection: &Projection) -> bool {
        let Some(path) = self.statically_null_paths.keys().next() else {
            return true;
        };
        if self.statically_null_paths.len() != 1 || projection.items.len() != 2 {
            return false;
        }
        let [path_item, null_item] = projection.items.as_slice() else {
            return false;
        };
        if path_item.alias.is_some() || null_item.alias.is_some() {
            return false;
        }
        let (
            Expression::Function {
                name: path_name,
                distinct: false,
                arguments: path_arguments,
            },
            Expression::Function {
                name: null_name,
                distinct: false,
                arguments: null_arguments,
            },
        ) = (&path_item.expression, &null_item.expression)
        else {
            return false;
        };
        path_name.len() == 1
            && null_name.len() == 1
            && path_name[0].eq_ignore_ascii_case(&null_name[0])
            && (path_name[0].eq_ignore_ascii_case("nodes")
                || path_name[0].eq_ignore_ascii_case("relationships"))
            && matches!(
                path_arguments.as_slice(),
                [Expression::Variable(variable)] if variable == path
            )
            && matches!(
                null_arguments.as_slice(),
                [Expression::Literal(ScalarValue::Null)]
            )
    }

    fn expanded_final_projection(
        &mut self,
        projection: &Projection,
    ) -> Result<Option<Vec<(String, ResidentNullableRelationOutputSource)>>> {
        if !self.is_exact_statically_null_path_projection(projection) {
            return Ok(None);
        }
        if !self.unmaterialized_scope_values.is_empty()
            && projection
                .items
                .iter()
                .any(|item| matches!(item.expression, Expression::Star))
        {
            return Ok(None);
        }
        let mut expanded = Vec::new();
        let mut names = BTreeSet::new();
        for (index, item) in projection.items.iter().enumerate() {
            if let Some((source, predicate_names)) = item.expression.entity_label_predicate_parts()
            {
                let Expression::Variable(variable) = source else {
                    return Ok(None);
                };
                let Some(binding) = self.scope.get(variable).copied() else {
                    return Ok(None);
                };
                let predicate_names = predicate_names
                    .iter()
                    .map(|name| match name {
                        Expression::Literal(ScalarValue::String(name)) => Ok(name.to_string()),
                        _ => Err(Error::internal(
                            "entity label output contains a non-literal name",
                        )),
                    })
                    .collect::<Result<Vec<_>>>()?;
                let domain = match binding.kind {
                    ResidentNullableRelationBindingKind::Node => {
                        ResidentNullableRelationEntityLabelDomain::Node(nullable_node_domain(
                            &predicate_names,
                            self.catalog,
                        ))
                    }
                    ResidentNullableRelationBindingKind::Relationship => {
                        ResidentNullableRelationEntityLabelDomain::Relationship(
                            nullable_relationship_predicate_domain(&predicate_names, self.catalog),
                        )
                    }
                };
                let name = item.column_name(index);
                if !names.insert(name.clone()) {
                    return Ok(None);
                }
                expanded.push((
                    name,
                    ResidentNullableRelationOutputSource::EntityLabelPredicate {
                        slot: binding.slot,
                        domain,
                    },
                ));
                continue;
            }
            match &item.expression {
                Expression::Star if item.alias.is_none() => {
                    for (name, binding) in &self.scope {
                        if !names.insert(name.clone()) {
                            return Ok(None);
                        }
                        expanded.push((
                            name.clone(),
                            ResidentNullableRelationOutputSource::Entity {
                                slot: binding.slot,
                                kind: binding.kind,
                            },
                        ));
                    }
                }
                Expression::Variable(variable) => {
                    let Some(binding) = self.scope.get(variable).copied() else {
                        return Ok(None);
                    };
                    let name = item.column_name(index);
                    if !names.insert(name.clone()) {
                        return Ok(None);
                    }
                    expanded.push((
                        name,
                        ResidentNullableRelationOutputSource::Entity {
                            slot: binding.slot,
                            kind: binding.kind,
                        },
                    ));
                }
                Expression::Property(source, property) => {
                    let Expression::Variable(variable) = source.as_ref() else {
                        return Ok(None);
                    };
                    let Some(binding) = self.scope.get(variable).copied() else {
                        return Ok(None);
                    };
                    let Some(source) = self.property_output_source(binding, property)? else {
                        return Ok(None);
                    };
                    let name = item.column_name(index);
                    if !names.insert(name.clone()) {
                        return Ok(None);
                    }
                    expanded.push((name, source));
                }
                Expression::Function {
                    name,
                    distinct: false,
                    arguments,
                } if name.len() == 1
                    && arguments.len() == 1
                    && (name[0].eq_ignore_ascii_case("tointeger")
                        || name[0].eq_ignore_ascii_case("tofloat")
                        || name[0].eq_ignore_ascii_case("tostring")) =>
                {
                    let function = name[0].to_ascii_lowercase();
                    let Some(source) =
                        self.converted_property_output_source(&function, &arguments[0])?
                    else {
                        return Ok(None);
                    };
                    let output_name = item.column_name(index);
                    if !names.insert(output_name.clone()) {
                        return Ok(None);
                    }
                    expanded.push((output_name, source));
                }
                Expression::Function {
                    name,
                    distinct: false,
                    arguments,
                } if name.len() == 1
                    && (name[0].eq_ignore_ascii_case("nodes")
                        || name[0].eq_ignore_ascii_case("relationships")) =>
                {
                    let Some(binding) = self.statically_null_path_argument(arguments) else {
                        return Ok(None);
                    };
                    let name = item.column_name(index);
                    if !names.insert(name.clone()) {
                        return Ok(None);
                    }
                    // Null is type-polymorphic in the client stream. Reuse the existing sealed
                    // always-null scalar source so both backends retain source-row alignment and
                    // publish no fabricated list payload.
                    expanded.push((
                        name,
                        ResidentNullableRelationOutputSource::NullProperty {
                            slot: binding.slot,
                            kind: binding.kind,
                        },
                    ));
                }
                Expression::Function {
                    name,
                    distinct: false,
                    arguments,
                } if name.len() == 1
                    && name[0].eq_ignore_ascii_case("type")
                    && arguments.len() == 1 =>
                {
                    let binding = match arguments.as_slice() {
                        [Expression::Variable(variable)] => {
                            let Some(binding) = self.scope.get(variable).copied() else {
                                return Ok(None);
                            };
                            binding
                        }
                        [Expression::Literal(ScalarValue::Null)] => {
                            let Some(binding) = self.known_null_relationship() else {
                                return Ok(None);
                            };
                            binding
                        }
                        _ => return Ok(None),
                    };
                    if binding.kind != ResidentNullableRelationBindingKind::Relationship {
                        return Ok(None);
                    }
                    let name = item.column_name(index);
                    if !names.insert(name.clone()) {
                        return Ok(None);
                    }
                    expanded.push((
                        name,
                        ResidentNullableRelationOutputSource::RelationshipType {
                            slot: binding.slot,
                        },
                    ));
                }
                _ => return Ok(None),
            }
        }
        Ok((!expanded.is_empty()).then_some(expanded))
    }

    fn scope_projection(&mut self, projection: &Projection) -> Result<bool> {
        let Some(expanded) = self.expanded_projection(projection) else {
            return Ok(false);
        };
        let mut bindings = Vec::with_capacity(expanded.len());
        let mut next_scope = BTreeMap::new();
        for (variable, source) in expanded {
            let output = self.allocate_slot()?;
            bindings.push(ResidentNullableRelationProjectionBinding {
                variable: variable.clone(),
                source: source.slot,
                output,
                row_limit: None,
            });
            next_scope.insert(
                variable,
                NullableRelationBinding {
                    slot: output,
                    kind: source.kind,
                },
            );
        }
        self.stages
            .push(ResidentNullableRelationStage::ScopeProject { bindings });
        self.scope = next_scope;
        self.fixed_path_lengths.clear();
        self.statically_null_paths.clear();
        self.unmaterialized_scope_values.clear();
        Ok(true)
    }

    /// Cancel one immediate `collect(entity) -> UNWIND` pair without ever constructing the list.
    ///
    /// `collect` preserves every non-null input entity (including duplicates) and `UNWIND`
    /// restores those members as rows. Therefore the pair is exactly a projection/rename when the
    /// collected binding is statically non-null. The list alias remains logically in scope after
    /// `UNWIND`; we deliberately remember that it was not materialized so any later direct or
    /// wildcard use fails closed instead of publishing a fabricated value.
    fn project_collect_unwind(
        &mut self,
        keep_scope: bool,
        projection: &Projection,
        unwind_expression: &Expression,
        unwind_variable: &str,
    ) -> Result<bool> {
        if keep_scope || projection.distinct || projection.items.is_empty() {
            return Ok(false);
        }

        let mut collect = None;
        let mut group_items = Vec::new();
        for (index, item) in projection.items.iter().enumerate() {
            let Expression::Function {
                name,
                distinct: false,
                arguments,
            } = &item.expression
            else {
                if contains_aggregate(&item.expression) {
                    return Ok(false);
                }
                group_items.push(item.clone());
                continue;
            };
            if name.len() == 1 && name[0].eq_ignore_ascii_case("collect") {
                if collect.is_some() || arguments.len() != 1 {
                    return Ok(false);
                }
                collect = Some((index, item, &arguments[0]));
            } else {
                if contains_aggregate(&item.expression) {
                    return Ok(false);
                }
                group_items.push(item.clone());
            }
        }

        let Some((collect_index, collect_item, collect_argument)) = collect else {
            return Ok(false);
        };
        let collect_name = collect_item
            .alias
            .clone()
            .unwrap_or_else(|| collect_item.column_name(collect_index));
        if !matches!(unwind_expression, Expression::Variable(name) if name == &collect_name) {
            return Ok(false);
        }
        let Expression::Variable(collected_name) = collect_argument else {
            return Ok(false);
        };
        let Some(collected) = self.scope.get(collected_name).copied() else {
            return Ok(false);
        };
        if !self.binding_is_statically_never_null(collected)? {
            return Ok(false);
        }

        let groups = if group_items.is_empty() {
            Vec::new()
        } else {
            let group_projection = Projection {
                distinct: false,
                items: group_items,
            };
            let Some(groups) = self.expanded_projection(&group_projection) else {
                return Ok(false);
            };
            groups
        };
        if groups.iter().any(|(name, _)| name == unwind_variable) {
            return Ok(false);
        }

        let mut bindings = Vec::with_capacity(groups.len() + 1);
        let mut next_scope = BTreeMap::new();
        for (name, source) in groups {
            let output = self.allocate_slot()?;
            bindings.push(ResidentNullableRelationProjectionBinding {
                variable: name.clone(),
                source: source.slot,
                output,
                row_limit: None,
            });
            next_scope.insert(
                name,
                NullableRelationBinding {
                    slot: output,
                    kind: source.kind,
                },
            );
        }
        let output = self.allocate_slot()?;
        bindings.push(ResidentNullableRelationProjectionBinding {
            variable: unwind_variable.to_owned(),
            source: collected.slot,
            output,
            row_limit: None,
        });
        next_scope.insert(
            unwind_variable.to_owned(),
            NullableRelationBinding {
                slot: output,
                kind: collected.kind,
            },
        );
        self.stages
            .push(ResidentNullableRelationStage::ScopeProject { bindings });
        self.scope = next_scope;
        self.fixed_path_lengths.clear();
        self.statically_null_paths.clear();
        self.unmaterialized_scope_values.clear();
        if collect_name != unwind_variable {
            self.unmaterialized_scope_values.insert(collect_name);
        }
        Ok(true)
    }

    fn stable_scope_limit(&mut self, expression: &Expression) -> Result<bool> {
        let Expression::Literal(ScalarValue::Integer(rows)) = expression else {
            return Ok(false);
        };
        let Ok(rows) = u64::try_from(*rows) else {
            return Ok(false);
        };
        let Some(ResidentNullableRelationStage::ScopeProject { bindings }) = self.stages.last_mut()
        else {
            return Ok(false);
        };
        if bindings.iter().any(|binding| binding.row_limit.is_some()) {
            return Ok(false);
        }
        for binding in bindings {
            binding.row_limit = Some(rows);
        }
        Ok(true)
    }

    fn final_projection(
        &mut self,
        projection: &Projection,
    ) -> Result<Option<Vec<CompiledResidentNullableRelationOutput>>> {
        let Some(expanded) = self.expanded_final_projection(projection)? else {
            return Ok(None);
        };
        let bindings = expanded
            .iter()
            .map(|(name, source)| ResidentNullableRelationOutputBinding {
                name: name.clone(),
                source: source.clone(),
            })
            .collect::<Vec<_>>();
        let outputs = expanded
            .into_iter()
            .map(|(name, source)| CompiledResidentNullableRelationOutput { name, source })
            .collect();
        self.stages
            .push(ResidentNullableRelationStage::FinalProject { bindings });
        Ok(Some(outputs))
    }
}

#[allow(dead_code)]
fn nullable_node_domain(
    labels: &[String],
    catalog: &crate::graph::NameCatalog,
) -> ResidentNullableNodeDomain {
    if labels.is_empty() {
        return ResidentNullableNodeDomain::Any;
    }
    let mut resolved = Vec::with_capacity(labels.len());
    for label in labels {
        let Some(label) = catalog.label(label) else {
            return ResidentNullableNodeDomain::KnownEmpty;
        };
        resolved.push(label);
    }
    resolved.sort_unstable();
    resolved.dedup();
    ResidentNullableNodeDomain::Known(resolved)
}

#[allow(dead_code)]
fn nullable_relationship_domain(
    relationship_types: &[String],
    catalog: &crate::graph::NameCatalog,
) -> ResidentNullableRelationshipDomain {
    if relationship_types.is_empty() {
        return ResidentNullableRelationshipDomain::Any;
    }
    let mut resolved = Vec::with_capacity(relationship_types.len());
    for relationship_type in relationship_types {
        let Some(relationship_type) = catalog.relationship_type(relationship_type) else {
            return ResidentNullableRelationshipDomain::KnownEmpty;
        };
        resolved.push(relationship_type);
    }
    resolved.sort_unstable();
    resolved.dedup();
    ResidentNullableRelationshipDomain::Known(resolved)
}

/// A relationship owns one exact type, whereas the colon expression is conjunctive. Repeated
/// equal names therefore collapse to one satisfiable type; any absent name or two distinct
/// resolved names make the predicate false for every non-null relationship.
fn nullable_relationship_predicate_domain(
    relationship_types: &[String],
    catalog: &crate::graph::NameCatalog,
) -> ResidentNullableRelationshipDomain {
    match nullable_relationship_domain(relationship_types, catalog) {
        ResidentNullableRelationshipDomain::Known(types) if types.len() == 1 => {
            ResidentNullableRelationshipDomain::Known(types)
        }
        ResidentNullableRelationshipDomain::Known(_)
        | ResidentNullableRelationshipDomain::KnownEmpty => {
            ResidentNullableRelationshipDomain::KnownEmpty
        }
        ResidentNullableRelationshipDomain::Any => {
            // Parser-produced colon expressions always contain at least one exact name.
            ResidentNullableRelationshipDomain::KnownEmpty
        }
    }
}

struct RowProgramBuilder<'a> {
    catalog: &'a crate::graph::NameCatalog,
    graph: &'a GraphStore,
    parameters: &'a BTreeMap<String, ResultValue>,
    scope: BTreeMap<String, RowSymbol>,
    instructions: Vec<ResidentRowInstruction>,
}

impl<'a> RowProgramBuilder<'a> {
    fn new(
        catalog: &'a crate::graph::NameCatalog,
        graph: &'a GraphStore,
        parameters: &'a BTreeMap<String, ResultValue>,
    ) -> Self {
        Self {
            catalog,
            graph,
            parameters,
            scope: BTreeMap::new(),
            instructions: Vec::new(),
        }
    }

    fn apply_projection(
        &mut self,
        keep_scope: bool,
        projection: &Projection,
    ) -> Option<Vec<(String, RowSymbol)>> {
        let incoming = self.scope.clone();
        let mut next = keep_scope.then_some(incoming.clone()).unwrap_or_default();
        let mut ordered = Vec::with_capacity(projection.items.len());
        for (index, item) in projection.items.iter().enumerate() {
            let value = self.value(&item.expression, &incoming).ok()??;
            let name = item
                .alias
                .clone()
                .unwrap_or_else(|| item.column_name(index));
            next.insert(name.clone(), value);
            ordered.push((name, value));
        }
        self.scope = next;
        Some(ordered)
    }

    fn apply_variable_projection(
        &mut self,
        keep_scope: bool,
        projection: &Projection,
    ) -> Option<Vec<(String, RowSymbol)>> {
        let incoming = self.scope.clone();
        let mut next = keep_scope.then_some(incoming.clone()).unwrap_or_default();
        let mut ordered = Vec::with_capacity(projection.items.len());
        for (index, item) in projection.items.iter().enumerate() {
            let Expression::Variable(variable) = &item.expression else {
                return None;
            };
            let value = *incoming.get(variable)?;
            let name = item
                .alias
                .clone()
                .unwrap_or_else(|| item.column_name(index));
            next.insert(name.clone(), value);
            ordered.push((name, value));
        }
        self.scope = next;
        Some(ordered)
    }

    fn value(
        &mut self,
        expression: &Expression,
        scope: &BTreeMap<String, RowSymbol>,
    ) -> Result<Option<RowSymbol>> {
        if let Expression::Variable(variable) = expression {
            return Ok(scope.get(variable).copied());
        }
        if let Expression::Property(source, name) = expression
            && let Expression::Variable(variable) = source.as_ref()
            && let Some(RowSymbol::Node(binding)) = scope.get(variable).copied()
            && let Some(property) = self.catalog.property(name)
            && self.graph.node_property_accepts(
                property,
                &ScalarValue::Duration {
                    months: 0,
                    days: 0,
                    seconds: 0,
                    nanos: 0,
                },
            ) == Some(true)
        {
            return Ok(Some(RowSymbol::DurationProperty {
                binding: ResidentEntityBinding::Node(binding),
                property,
            }));
        }
        self.expression(expression, scope)
            .map(|register| register.map(RowSymbol::Register))
    }

    fn expression(
        &mut self,
        expression: &Expression,
        scope: &BTreeMap<String, RowSymbol>,
    ) -> Result<Option<u16>> {
        if let Expression::Variable(variable) = expression {
            return Ok(match scope.get(variable) {
                Some(RowSymbol::Register(register)) => Some(*register),
                _ => None,
            });
        }
        let (output_type, operation) = match expression {
            Expression::Property(source, name) => {
                let Expression::Variable(variable) = source.as_ref() else {
                    return Ok(None);
                };
                if let Some(RowSymbol::Register(temporal)) = scope.get(variable).copied() {
                    let Some(accessor) = ResidentTemporalAccessor::from_property_name(name) else {
                        return Ok(None);
                    };
                    let Some(temporal_type) = self.register_type(temporal) else {
                        return Ok(None);
                    };
                    if !accessor.supports(temporal_type) {
                        return Ok(None);
                    }
                    let maximum_string_bytes =
                        self.temporal_accessor_string_capacity(temporal, temporal_type, accessor)?;
                    let named_zone_table =
                        self.temporal_accessor_named_zone_table(temporal, temporal_type)?;
                    return self
                        .push(
                            accessor.output_type(),
                            ResidentRowOperation::TemporalAccessor {
                                temporal,
                                accessor,
                                maximum_string_bytes,
                                named_zone_table,
                            },
                        )
                        .map(Some);
                }
                if let Some(RowSymbol::DurationProperty { binding, property }) =
                    scope.get(variable).copied()
                {
                    let Some(accessor) = ResidentTemporalAccessor::from_property_name(name) else {
                        return Ok(None);
                    };
                    if !accessor.is_duration() {
                        return Ok(None);
                    }
                    return self
                        .push(
                            ResidentRowValueType::Integer,
                            ResidentRowOperation::LoadDurationAccessor {
                                binding,
                                property,
                                accessor,
                            },
                        )
                        .map(Some);
                }
                let Some(RowSymbol::Node(binding)) = scope.get(variable).copied() else {
                    return Ok(None);
                };
                let Some(property) = self.catalog.property(name) else {
                    return Ok(None);
                };
                let binding = ResidentEntityBinding::Node(binding);
                if self.graph.node_property_is_boolean(property) {
                    (
                        ResidentRowValueType::Boolean,
                        ResidentRowOperation::LoadBooleanProperty { binding, property },
                    )
                } else if self.graph.node_property_is_integer(property) {
                    (
                        ResidentRowValueType::Integer,
                        ResidentRowOperation::LoadIntegerProperty { binding, property },
                    )
                } else if self.graph.node_property_is_float(property) {
                    (
                        ResidentRowValueType::Float,
                        ResidentRowOperation::LoadFloatProperty { binding, property },
                    )
                } else if self.graph.node_property_is_string(property) {
                    (
                        ResidentRowValueType::String,
                        ResidentRowOperation::LoadStringProperty {
                            binding,
                            property,
                            maximum_bytes: self.node_string_maximum_bytes(property)?,
                        },
                    )
                } else if self.graph.node_property_is_date(property) {
                    (
                        ResidentRowValueType::Date,
                        ResidentRowOperation::LoadDateProperty { binding, property },
                    )
                } else if self.graph.node_property_is_local_time(property) {
                    (
                        ResidentRowValueType::LocalTime,
                        ResidentRowOperation::LoadLocalTimeProperty { binding, property },
                    )
                } else if self.graph.node_property_is_zoned_time(property) {
                    (
                        ResidentRowValueType::ZonedTime,
                        ResidentRowOperation::LoadZonedTimeProperty { binding, property },
                    )
                } else if self.graph.node_property_is_local_datetime(property) {
                    (
                        ResidentRowValueType::LocalDateTime,
                        ResidentRowOperation::LoadLocalDateTimeProperty { binding, property },
                    )
                } else if self.graph.node_property_is_zoned_datetime(property) {
                    (
                        ResidentRowValueType::ZonedDateTime,
                        ResidentRowOperation::LoadZonedDateTimeProperty {
                            binding,
                            property,
                            maximum_timezone_bytes: self
                                .node_zoned_datetime_timezone_maximum_bytes(property)?,
                        },
                    )
                } else if let Some(maximum_elements) =
                    self.node_integer_list_maximum_elements(property)?
                {
                    (
                        ResidentRowValueType::List,
                        ResidentRowOperation::LoadListProperty {
                            binding,
                            property,
                            maximum_elements,
                        },
                    )
                } else {
                    return Ok(None);
                }
            }
            Expression::List(elements) => {
                let mut registers = Vec::with_capacity(elements.len());
                for element in elements {
                    let Some(register) = self.expression(element, scope)? else {
                        return Ok(None);
                    };
                    if self.register_type(register) != Some(ResidentRowValueType::Integer) {
                        return Ok(None);
                    }
                    registers.push(register);
                }
                (
                    ResidentRowValueType::List,
                    ResidentRowOperation::List {
                        elements: registers,
                    },
                )
            }
            Expression::Index { expression, index } => {
                let (Some(list), Some(index)) = (
                    self.expression(expression, scope)?,
                    self.expression(index, scope)?,
                ) else {
                    return Ok(None);
                };
                if self.register_type(list) != Some(ResidentRowValueType::List)
                    || self.register_type(index) != Some(ResidentRowValueType::Integer)
                {
                    return Ok(None);
                }
                (
                    ResidentRowValueType::Integer,
                    ResidentRowOperation::ListIndex { list, index },
                )
            }
            Expression::IsNull {
                expression,
                negated,
            } => {
                let value = match expression.as_ref() {
                    // This compiler's only entity symbol is produced by a mandatory node scan,
                    // so every row reaching the program contains a non-null canonical node.
                    Expression::Variable(variable)
                        if matches!(scope.get(variable), Some(RowSymbol::Node(_))) =>
                    {
                        *negated
                    }
                    // A property absent from the immutable binding catalog is null on every node
                    // in this graph revision. Do not extend this proof to a declared property:
                    // per-row validity must remain device-evaluated by a future general opcode.
                    Expression::Property(source, property)
                        if self.catalog.property(property).is_none()
                            && matches!(
                                source.as_ref(),
                                Expression::Variable(variable)
                                    if matches!(scope.get(variable), Some(RowSymbol::Node(_)))
                            ) =>
                    {
                        !*negated
                    }
                    _ => return Ok(None),
                };
                (
                    ResidentRowValueType::Boolean,
                    ResidentRowOperation::BooleanConstant(value),
                )
            }
            Expression::Literal(ScalarValue::Boolean(value)) => (
                ResidentRowValueType::Boolean,
                ResidentRowOperation::BooleanConstant(*value),
            ),
            Expression::Literal(ScalarValue::Integer(value)) => (
                ResidentRowValueType::Integer,
                ResidentRowOperation::IntegerConstant(*value),
            ),
            Expression::Literal(ScalarValue::Float(value)) => (
                ResidentRowValueType::Float,
                ResidentRowOperation::FloatConstant(value.0.to_bits()),
            ),
            Expression::Literal(ScalarValue::String(value)) => (
                ResidentRowValueType::String,
                ResidentRowOperation::StringConstant(value.to_string()),
            ),
            Expression::Parameter(name) => match self.parameters.get(name) {
                Some(ResultValue::Scalar(ScalarValue::Boolean(value))) => (
                    ResidentRowValueType::Boolean,
                    ResidentRowOperation::BooleanConstant(*value),
                ),
                Some(ResultValue::Scalar(ScalarValue::Integer(value))) => (
                    ResidentRowValueType::Integer,
                    ResidentRowOperation::IntegerConstant(*value),
                ),
                Some(ResultValue::Scalar(ScalarValue::Float(value))) => (
                    ResidentRowValueType::Float,
                    ResidentRowOperation::FloatConstant(value.0.to_bits()),
                ),
                Some(ResultValue::Scalar(ScalarValue::String(value))) => (
                    ResidentRowValueType::String,
                    ResidentRowOperation::StringConstant(value.to_string()),
                ),
                _ => return Ok(None),
            },
            Expression::Function {
                name,
                distinct: false,
                arguments,
            } if matches!(name.as_slice(), [name] if name.eq_ignore_ascii_case("coalesce"))
                && arguments.len() == 2 =>
            {
                let [
                    Expression::Property(left_source, _),
                    Expression::Property(right_source, _),
                ] = arguments.as_slice()
                else {
                    return Ok(None);
                };
                let (Expression::Variable(left_variable), Expression::Variable(right_variable)) =
                    (left_source.as_ref(), right_source.as_ref())
                else {
                    return Ok(None);
                };
                if left_variable != right_variable
                    || !matches!(scope.get(left_variable), Some(RowSymbol::Node(_)))
                {
                    return Ok(None);
                }
                let (Some(left), Some(right)) = (
                    self.expression(&arguments[0], scope)?,
                    self.expression(&arguments[1], scope)?,
                ) else {
                    return Ok(None);
                };
                if self.register_type(left) != Some(ResidentRowValueType::String)
                    || self.register_type(right) != Some(ResidentRowValueType::String)
                {
                    return Ok(None);
                }
                (
                    ResidentRowValueType::String,
                    ResidentRowOperation::StringCoalesce { left, right },
                )
            }
            Expression::Function {
                name,
                distinct: false,
                arguments,
            } if name.len() == 1
                && name[0].eq_ignore_ascii_case("toString")
                && arguments.len() == 1 =>
            {
                let Some(operand) = self.expression(&arguments[0], scope)? else {
                    return Ok(None);
                };
                if self.register_type(operand) != Some(ResidentRowValueType::Boolean) {
                    return Ok(None);
                }
                (
                    ResidentRowValueType::String,
                    ResidentRowOperation::BooleanToString { operand },
                )
            }
            Expression::Unary {
                operation: UnaryOperator::Not,
                operand,
            } => {
                let Some(operand) = self.expression(operand, scope)? else {
                    return Ok(None);
                };
                if self.register_type(operand) != Some(ResidentRowValueType::Boolean) {
                    return Ok(None);
                }
                (
                    ResidentRowValueType::Boolean,
                    ResidentRowOperation::BooleanNot { operand },
                )
            }
            Expression::Unary {
                operation: UnaryOperator::Positive,
                operand,
            } => return self.expression(operand, scope),
            Expression::Unary {
                operation: UnaryOperator::Negative,
                operand,
            } => {
                let Some(operand) = self.expression(operand, scope)? else {
                    return Ok(None);
                };
                let Some(
                    output_type @ (ResidentRowValueType::Integer | ResidentRowValueType::Float),
                ) = self.register_type(operand)
                else {
                    return Ok(None);
                };
                (output_type, ResidentRowOperation::NumericNegate { operand })
            }
            Expression::Binary {
                left,
                operation: BinaryOperator::And,
                right,
            } => {
                let (Some(left), Some(right)) = (
                    self.expression(left, scope)?,
                    self.expression(right, scope)?,
                ) else {
                    return Ok(None);
                };
                if self.register_type(left) != Some(ResidentRowValueType::Boolean)
                    || self.register_type(right) != Some(ResidentRowValueType::Boolean)
                {
                    return Ok(None);
                }
                (
                    ResidentRowValueType::Boolean,
                    ResidentRowOperation::BooleanAnd { left, right },
                )
            }
            Expression::Binary {
                left,
                operation:
                    operation @ (BinaryOperator::Add
                    | BinaryOperator::Subtract
                    | BinaryOperator::Multiply),
                right,
            } => {
                if *operation == BinaryOperator::Add {
                    let temporal_duration = if let Some(duration) =
                        constant_duration_components(right, self.parameters)?
                    {
                        Some((left.as_ref(), duration))
                    } else {
                        constant_duration_components(left, self.parameters)?
                            .map(|duration| (right.as_ref(), duration))
                    };
                    if let Some((temporal_expression, duration)) = temporal_duration {
                        let Some(temporal) = self.expression(temporal_expression, scope)? else {
                            return Ok(None);
                        };
                        let Some(output_type) = self.register_type(temporal) else {
                            return Ok(None);
                        };
                        if !matches!(
                            output_type,
                            ResidentRowValueType::Date
                                | ResidentRowValueType::LocalTime
                                | ResidentRowValueType::ZonedTime
                                | ResidentRowValueType::LocalDateTime
                                | ResidentRowValueType::ZonedDateTime
                        ) {
                            return Ok(None);
                        }
                        let (months, days, seconds, nanos) = duration;
                        return self
                            .push(
                                output_type,
                                ResidentRowOperation::TemporalAddDuration {
                                    temporal,
                                    months,
                                    days,
                                    seconds,
                                    nanos,
                                },
                            )
                            .map(Some);
                    }
                }
                let (Some(left), Some(right)) = (
                    self.expression(left, scope)?,
                    self.expression(right, scope)?,
                ) else {
                    return Ok(None);
                };
                let (Some(left_type), Some(right_type)) =
                    (self.register_type(left), self.register_type(right))
                else {
                    return Ok(None);
                };
                if *operation == BinaryOperator::Add
                    && left_type == ResidentRowValueType::String
                    && right_type == ResidentRowValueType::String
                {
                    return self
                        .push(
                            ResidentRowValueType::String,
                            ResidentRowOperation::StringConcat { left, right },
                        )
                        .map(Some);
                }
                if *operation == BinaryOperator::Add
                    && left_type == ResidentRowValueType::List
                    && right_type == ResidentRowValueType::List
                {
                    return self
                        .push(
                            ResidentRowValueType::List,
                            ResidentRowOperation::ListConcat { left, right },
                        )
                        .map(Some);
                }
                if !left_type.is_numeric() || !right_type.is_numeric() {
                    return Ok(None);
                }
                let output_type = if left_type == ResidentRowValueType::Float
                    || right_type == ResidentRowValueType::Float
                {
                    ResidentRowValueType::Float
                } else {
                    ResidentRowValueType::Integer
                };
                let operation = match operation {
                    BinaryOperator::Add => ResidentRowOperation::NumericAdd { left, right },
                    BinaryOperator::Subtract => {
                        ResidentRowOperation::NumericSubtract { left, right }
                    }
                    BinaryOperator::Multiply => {
                        ResidentRowOperation::NumericMultiply { left, right }
                    }
                    _ => unreachable!("matched numeric operation changed"),
                };
                (output_type, operation)
            }
            Expression::Binary {
                left,
                operation: BinaryOperator::Modulo,
                right,
            } => {
                let (Some(left), Some(right)) = (
                    self.expression(left, scope)?,
                    self.expression(right, scope)?,
                ) else {
                    return Ok(None);
                };
                if self.register_type(left) != Some(ResidentRowValueType::Integer)
                    || self.register_type(right) != Some(ResidentRowValueType::Integer)
                {
                    return Ok(None);
                }
                (
                    ResidentRowValueType::Integer,
                    ResidentRowOperation::IntegerModulo { left, right },
                )
            }
            _ => return Ok(None),
        };
        self.push(output_type, operation).map(Some)
    }

    fn push(
        &mut self,
        output_type: ResidentRowValueType,
        operation: ResidentRowOperation,
    ) -> Result<u16> {
        if self.instructions.len() >= RESIDENT_ROW_PROGRAM_MAX_REGISTERS {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident typed-row expression exceeds the register budget",
            ));
        }
        let register = u16::try_from(self.instructions.len()).map_err(|_| {
            Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident typed-row register is not addressable",
            )
        })?;
        self.instructions.push(ResidentRowInstruction {
            output_type,
            operation,
        });
        Ok(register)
    }

    fn register_type(&self, register: u16) -> Option<ResidentRowValueType> {
        self.instructions
            .get(register as usize)
            .map(|instruction| instruction.output_type)
    }

    fn supports_unsorted_final_register(&self, register: u16) -> bool {
        let Some(instruction) = self.instructions.get(usize::from(register)) else {
            return false;
        };
        match &instruction.operation {
            ResidentRowOperation::InputColumn(
                ResidentRowColumn::Boolean { .. }
                | ResidentRowColumn::Integer { .. }
                | ResidentRowColumn::Float { .. }
                | ResidentRowColumn::Date { .. }
                | ResidentRowColumn::LocalTime { .. }
                | ResidentRowColumn::ZonedTime { .. }
                | ResidentRowColumn::LocalDateTime { .. },
            )
            | ResidentRowOperation::BooleanConstant(_)
            | ResidentRowOperation::LoadBooleanProperty { .. }
            | ResidentRowOperation::LoadIntegerProperty { .. }
            | ResidentRowOperation::LoadFloatProperty { .. }
            | ResidentRowOperation::LoadDateProperty { .. }
            | ResidentRowOperation::LoadLocalTimeProperty { .. }
            | ResidentRowOperation::LoadZonedTimeProperty { .. }
            | ResidentRowOperation::LoadLocalDateTimeProperty { .. }
            | ResidentRowOperation::LoadZonedDateTimeProperty { .. }
            | ResidentRowOperation::LoadListProperty { .. }
            | ResidentRowOperation::TemporalAccessor { .. }
            | ResidentRowOperation::LoadDurationAccessor { .. } => true,
            ResidentRowOperation::BooleanToString { .. }
            | ResidentRowOperation::StringCoalesce { .. } => true,
            ResidentRowOperation::NumericAdd { left, right }
                if instruction.output_type == ResidentRowValueType::Integer =>
            {
                (self.is_integer_property_load(*left) && self.is_integer_constant(*right))
                    || (self.is_integer_constant(*left) && self.is_integer_property_load(*right))
            }
            ResidentRowOperation::List { elements }
                if instruction.output_type == ResidentRowValueType::List =>
            {
                elements.iter().all(|element| {
                    self.instructions
                        .get(usize::from(*element))
                        .is_some_and(|element| element.output_type == ResidentRowValueType::Integer)
                })
            }
            ResidentRowOperation::ListConcat { left, right }
                if instruction.output_type == ResidentRowValueType::List =>
            {
                self.is_list_register(*left) && self.is_list_register(*right)
            }
            _ => false,
        }
    }

    fn is_integer_property_load(&self, register: u16) -> bool {
        self.instructions
            .get(usize::from(register))
            .is_some_and(|instruction| {
                instruction.output_type == ResidentRowValueType::Integer
                    && matches!(
                        &instruction.operation,
                        ResidentRowOperation::LoadIntegerProperty { .. }
                    )
            })
    }

    fn is_integer_constant(&self, register: u16) -> bool {
        self.instructions
            .get(usize::from(register))
            .is_some_and(|instruction| {
                instruction.output_type == ResidentRowValueType::Integer
                    && matches!(
                        &instruction.operation,
                        ResidentRowOperation::IntegerConstant(_)
                    )
            })
    }

    fn is_list_register(&self, register: u16) -> bool {
        self.instructions
            .get(usize::from(register))
            .is_some_and(|instruction| instruction.output_type == ResidentRowValueType::List)
    }

    fn node_string_maximum_bytes(&self, property: crate::types::PropertyId) -> Result<u32> {
        let maximum = self
            .graph
            .nodes()
            .filter_map(|node| match node.property(property) {
                Some(ScalarValue::String(value)) => Some(value.len()),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        u32::try_from(maximum).map_err(|_| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "resident string property width exceeds u32",
            )
        })
    }

    fn node_zoned_datetime_timezone_maximum_bytes(
        &self,
        property: crate::types::PropertyId,
    ) -> Result<u32> {
        let maximum = self
            .graph
            .nodes()
            .filter_map(|node| match node.property(property) {
                Some(ScalarValue::ZonedDateTime { timezone, .. }) => Some(timezone.len()),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        u32::try_from(maximum).map_err(|_| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "resident zoned-datetime timezone width exceeds u32",
            )
        })
    }

    fn temporal_accessor_string_capacity(
        &self,
        temporal: u16,
        temporal_type: ResidentRowValueType,
        accessor: ResidentTemporalAccessor,
    ) -> Result<u32> {
        match accessor {
            ResidentTemporalAccessor::Timezone
                if temporal_type == ResidentRowValueType::ZonedDateTime =>
            {
                let Some(ResidentRowInstruction {
                    operation:
                        ResidentRowOperation::LoadZonedDateTimeProperty {
                            maximum_timezone_bytes,
                            ..
                        },
                    ..
                }) = self.instructions.get(usize::from(temporal))
                else {
                    return Err(Error::new(
                        ErrorCode::GpuAdmissionFailure,
                        "resident temporal accessor lost zoned-datetime source capacity",
                    ));
                };
                Ok(*maximum_timezone_bytes)
            }
            ResidentTemporalAccessor::Timezone | ResidentTemporalAccessor::Offset => Ok(9),
            _ => Ok(0),
        }
    }

    fn temporal_accessor_named_zone_table(
        &self,
        temporal: u16,
        temporal_type: ResidentRowValueType,
    ) -> Result<Arc<[u8]>> {
        if temporal_type != ResidentRowValueType::ZonedDateTime {
            return Ok(Arc::from([]));
        }
        let Some(ResidentRowInstruction {
            operation: ResidentRowOperation::LoadZonedDateTimeProperty { property, .. },
            ..
        }) = self.instructions.get(usize::from(temporal))
        else {
            return Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "resident temporal accessor requires a direct zoned-datetime property source",
            ));
        };
        let names = self
            .graph
            .nodes()
            .filter_map(|node| match node.property(*property) {
                Some(ScalarValue::ZonedDateTime { timezone, .. }) => Some(timezone.to_string()),
                _ => None,
            })
            .collect::<BTreeSet<_>>();
        #[cfg(all(feature = "accelerator", any(target_os = "macos", target_os = "ios")))]
        let bytes = crate::execution::resident_named_zone_table_bytes(&names)?;
        #[cfg(not(all(feature = "accelerator", any(target_os = "macos", target_os = "ios"))))]
        let bytes = {
            let _ = names;
            Vec::new()
        };
        Ok(Arc::from(bytes))
    }

    fn node_integer_list_maximum_elements(
        &self,
        property: crate::types::PropertyId,
    ) -> Result<Option<u32>> {
        if self.graph.node_property_accepts(
            property,
            &ScalarValue::List(crate::DocumentList::new(Vec::new())?),
        ) != Some(true)
        {
            return Ok(None);
        }
        let mut maximum = 0_usize;
        for node in self.graph.nodes() {
            let Some(value) = node.property(property) else {
                continue;
            };
            let ScalarValue::List(value) = value else {
                return Ok(None);
            };
            let items = value.items()?;
            if items
                .iter()
                .any(|item| !matches!(item, DocumentItem::Scalar(ScalarValue::Integer(_))))
            {
                return Ok(None);
            }
            maximum = maximum.max(items.len());
        }
        u32::try_from(maximum).map(Some).map_err(|_| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "resident integer-list property width exceeds u32",
            )
        })
    }
}

fn graph_input(
    project: ProjectId,
    plan: &PhysicalPlan,
    labels: Vec<crate::types::LabelId>,
    max_output_rows: usize,
) -> ResidentNodePipelineRequest {
    ResidentNodePipelineRequest {
        project,
        labels,
        layers: plan.read_layers,
        initial_optional: false,
        expansion: None,
        continuations: Vec::new(),
        correlated_optional: None,
        relationship_null_filter: None,
        predicates: Vec::new(),
        property_filters: Vec::new(),
        value_matrix: None,
        mutation: None,
        orders: Vec::new(),
        offset: 0,
        limit: usize::MAX,
        integer_projections: Vec::new(),
        property_null_projections: Vec::new(),
        max_output_rows,
    }
}

fn scalar_input(
    project: ProjectId,
    plan: &PhysicalPlan,
    rows: usize,
) -> ResidentNodePipelineRequest {
    graph_input(project, plan, Vec::new(), rows)
}

fn typed_unwind_column(
    expression: &Expression,
    parameters: &BTreeMap<String, ResultValue>,
) -> Result<Option<ResidentRowColumn>> {
    let Some(ResultValue::List(values)) = constant_result_value(expression, parameters)? else {
        return Ok(None);
    };
    let Some(values) = values
        .into_iter()
        .map(|value| match value {
            ResultValue::Scalar(value) => Some(value),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()
    else {
        return Ok(None);
    };

    let first_type = values.iter().find_map(|value| match value {
        ScalarValue::Null => None,
        ScalarValue::Boolean(_) => Some(ResidentRowValueType::Boolean),
        ScalarValue::Integer(_) => Some(ResidentRowValueType::Integer),
        ScalarValue::Float(_) => Some(ResidentRowValueType::Float),
        ScalarValue::String(_) => Some(ResidentRowValueType::String),
        ScalarValue::Date(_) => Some(ResidentRowValueType::Date),
        ScalarValue::LocalTime(_) => Some(ResidentRowValueType::LocalTime),
        ScalarValue::ZonedTime { .. } => Some(ResidentRowValueType::ZonedTime),
        ScalarValue::LocalDateTime { .. } => Some(ResidentRowValueType::LocalDateTime),
        ScalarValue::ZonedDateTime { .. } => Some(ResidentRowValueType::ZonedDateTime),
        _ => Some(ResidentRowValueType::Boolean), // rejected by the homogeneous check below
    });
    let value_type = first_type.unwrap_or(ResidentRowValueType::Integer);
    let homogeneous = values.iter().all(|value| {
        matches!(value, ScalarValue::Null)
            || matches!(
                (value_type, value),
                (ResidentRowValueType::Boolean, ScalarValue::Boolean(_))
                    | (ResidentRowValueType::Integer, ScalarValue::Integer(_))
                    | (ResidentRowValueType::Float, ScalarValue::Float(_))
                    | (ResidentRowValueType::String, ScalarValue::String(_))
                    | (ResidentRowValueType::Date, ScalarValue::Date(_))
                    | (ResidentRowValueType::LocalTime, ScalarValue::LocalTime(_))
                    | (
                        ResidentRowValueType::ZonedTime,
                        ScalarValue::ZonedTime { .. }
                    )
                    | (
                        ResidentRowValueType::LocalDateTime,
                        ScalarValue::LocalDateTime { .. }
                    )
                    | (
                        ResidentRowValueType::ZonedDateTime,
                        ScalarValue::ZonedDateTime { .. }
                    )
            )
    });
    if !homogeneous {
        return Ok(None);
    }

    let validity = values
        .iter()
        .map(|value| u8::from(!matches!(value, ScalarValue::Null)))
        .collect::<Vec<_>>();
    let column = match value_type {
        ResidentRowValueType::Boolean => Some(ResidentRowColumn::Boolean {
            values: values
                .iter()
                .map(|value| match value {
                    ScalarValue::Boolean(value) => u8::from(*value),
                    ScalarValue::Null => 0,
                    _ => unreachable!("homogeneous Boolean input changed"),
                })
                .collect(),
            validity,
        }),
        ResidentRowValueType::Integer => Some(ResidentRowColumn::Integer {
            values: values
                .iter()
                .map(|value| match value {
                    ScalarValue::Integer(value) => *value,
                    ScalarValue::Null => 0,
                    _ => unreachable!("homogeneous INTEGER input changed"),
                })
                .collect(),
            validity,
        }),
        ResidentRowValueType::Float => Some(ResidentRowColumn::Float {
            bits: values
                .iter()
                .map(|value| match value {
                    ScalarValue::Float(value) => value.0.to_bits(),
                    ScalarValue::Null => 0,
                    _ => unreachable!("homogeneous FLOAT input changed"),
                })
                .collect(),
            validity,
        }),
        ResidentRowValueType::String => {
            let mut offsets = Vec::with_capacity(values.len().saturating_add(1));
            let mut bytes = Vec::new();
            offsets.push(0_u32);
            for value in &values {
                match value {
                    ScalarValue::String(value) => bytes.extend_from_slice(value.as_bytes()),
                    ScalarValue::Null => {}
                    _ => unreachable!("homogeneous STRING input changed"),
                }
                offsets.push(u32::try_from(bytes.len()).map_err(|_| {
                    Error::new(
                        ErrorCode::ResultBudgetExceeded,
                        "resident STRING literal bytes exceed u32",
                    )
                })?);
            }
            Some(ResidentRowColumn::String {
                offsets,
                bytes,
                validity,
            })
        }
        ResidentRowValueType::Date => Some(ResidentRowColumn::Date {
            days: values
                .iter()
                .map(|value| match value {
                    ScalarValue::Date(value) => *value,
                    ScalarValue::Null => 0,
                    _ => unreachable!("homogeneous DATE input changed"),
                })
                .collect(),
            validity,
        }),
        ResidentRowValueType::LocalTime => Some(ResidentRowColumn::LocalTime {
            nanos: values
                .iter()
                .map(|value| match value {
                    ScalarValue::LocalTime(value) => *value,
                    ScalarValue::Null => 0,
                    _ => unreachable!("homogeneous LOCAL TIME input changed"),
                })
                .collect(),
            validity,
        }),
        ResidentRowValueType::ZonedTime => Some(ResidentRowColumn::ZonedTime {
            nanos: values
                .iter()
                .map(|value| match value {
                    ScalarValue::ZonedTime { nanos, .. } => *nanos,
                    ScalarValue::Null => 0,
                    _ => unreachable!("homogeneous ZONED TIME input changed"),
                })
                .collect(),
            offset_seconds: values
                .iter()
                .map(|value| match value {
                    ScalarValue::ZonedTime { offset_seconds, .. } => *offset_seconds,
                    ScalarValue::Null => 0,
                    _ => unreachable!("homogeneous ZONED TIME input changed"),
                })
                .collect(),
            validity,
        }),
        ResidentRowValueType::LocalDateTime => Some(ResidentRowColumn::LocalDateTime {
            seconds: values
                .iter()
                .map(|value| match value {
                    ScalarValue::LocalDateTime { seconds, .. } => *seconds,
                    ScalarValue::Null => 0,
                    _ => unreachable!("homogeneous LOCAL DATETIME input changed"),
                })
                .collect(),
            nanos: values
                .iter()
                .map(|value| match value {
                    ScalarValue::LocalDateTime { nanos, .. } => *nanos,
                    ScalarValue::Null => 0,
                    _ => unreachable!("homogeneous LOCAL DATETIME input changed"),
                })
                .collect(),
            validity,
        }),
        ResidentRowValueType::ZonedDateTime => {
            let mut timezone_offsets = Vec::with_capacity(values.len().saturating_add(1));
            let mut timezone_bytes = Vec::new();
            timezone_offsets.push(0_u32);
            for value in &values {
                match value {
                    ScalarValue::ZonedDateTime { timezone, .. } => {
                        timezone_bytes.extend_from_slice(timezone.as_bytes());
                    }
                    ScalarValue::Null => {}
                    _ => unreachable!("homogeneous ZONED DATETIME input changed"),
                }
                timezone_offsets.push(u32::try_from(timezone_bytes.len()).map_err(|_| {
                    Error::new(
                        ErrorCode::ResultBudgetExceeded,
                        "resident timezone literal bytes exceed u32",
                    )
                })?);
            }
            Some(ResidentRowColumn::ZonedDateTime {
                seconds: values
                    .iter()
                    .map(|value| match value {
                        ScalarValue::ZonedDateTime { seconds, .. } => *seconds,
                        ScalarValue::Null => 0,
                        _ => unreachable!("homogeneous ZONED DATETIME input changed"),
                    })
                    .collect(),
                nanos: values
                    .iter()
                    .map(|value| match value {
                        ScalarValue::ZonedDateTime { nanos, .. } => *nanos,
                        ScalarValue::Null => 0,
                        _ => unreachable!("homogeneous ZONED DATETIME input changed"),
                    })
                    .collect(),
                timezone_offsets,
                timezone_bytes,
                validity,
            })
        }
        ResidentRowValueType::List => None,
    };
    Ok(column)
}

fn constant_result_value(
    expression: &Expression,
    parameters: &BTreeMap<String, ResultValue>,
) -> Result<Option<ResultValue>> {
    match expression {
        Expression::Literal(value) => Ok(Some(ResultValue::Scalar(value.clone()))),
        Expression::Parameter(name) => Ok(parameters.get(name).cloned()),
        Expression::List(items) => {
            let mut values = Vec::with_capacity(items.len());
            for item in items {
                let Some(value) = constant_result_value(item, parameters)? else {
                    return Ok(None);
                };
                values.push(value);
            }
            Ok(Some(ResultValue::List(values)))
        }
        Expression::Map(items) => {
            let mut values = BTreeMap::new();
            for (name, expression) in items {
                let Some(value) = constant_result_value(expression, parameters)? else {
                    return Ok(None);
                };
                values.insert(name.clone(), value);
            }
            Ok(Some(ResultValue::Map(values)))
        }
        Expression::Function {
            name,
            distinct: false,
            arguments,
        } if arguments.len() == 1
            && matches!(
                name.as_slice(),
                [name]
                    if matches!(
                        name.as_str(),
                        "date" | "localtime" | "time" | "localdatetime" | "datetime"
                    )
            ) =>
        {
            let mut values = Vec::with_capacity(arguments.len());
            for argument in arguments {
                let Some(value) = constant_result_value(argument, parameters)? else {
                    return Ok(None);
                };
                values.push(value);
            }
            let [name] = name.as_slice() else {
                unreachable!("guarded temporal constructor name changed")
            };
            super::executor::evaluate_explicit_temporal_constructor(name, &values).map(Some)
        }
        Expression::Function {
            name,
            distinct: false,
            arguments,
        } if name.as_slice() == ["duration"] => {
            let mut values = Vec::with_capacity(arguments.len());
            for argument in arguments {
                let Some(value) = constant_result_value(argument, parameters)? else {
                    return Ok(None);
                };
                values.push(value);
            }
            super::executor::evaluate_explicit_duration_constructor(&values).map(Some)
        }
        _ => Ok(None),
    }
}

fn constant_duration_components(
    expression: &Expression,
    parameters: &BTreeMap<String, ResultValue>,
) -> Result<Option<(i64, i64, i64, i32)>> {
    Ok(match constant_result_value(expression, parameters)? {
        Some(ResultValue::Scalar(ScalarValue::Duration {
            months,
            days,
            seconds,
            nanos,
        })) => Some((months, days, seconds, nanos)),
        _ => None,
    })
}

fn non_negative_count(
    expression: &Expression,
    parameters: &BTreeMap<String, ResultValue>,
) -> Option<usize> {
    let value = match expression {
        Expression::Literal(ScalarValue::Integer(value)) => *value,
        Expression::Parameter(name) => match parameters.get(name) {
            Some(ResultValue::Scalar(ScalarValue::Integer(value))) => *value,
            _ => return None,
        },
        _ => return None,
    };
    usize::try_from(value).ok()
}

fn fresh_execution_id() -> ResidentExecutionId {
    let mut random = rand::rng();
    loop {
        let execution = ResidentExecutionId {
            high: random.random(),
            low: random.random(),
        };
        if execution.high != 0 || execution.low != 0 {
            return execution;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::{Duration, Instant};

    use tokio_util::sync::CancellationToken;

    use crate::{
        DocumentItem, DocumentList, EdgeId, Layer, NodeId,
        cypher::{
            BindCapabilities, ColumnType, ExecutionContext, ExecutionOutput, OptimizerInput,
            QueryEngine, ResultValue, bind, optimize, parse, plan,
        },
        execution::{
            BackendKind, CpuBackend, ExecutionBackend, ResidentDeviceCompletion,
            ResidentNullableNodeDomain, ResidentNullableRelationMatchMode,
            ResidentNullableRelationObligationKind, ResidentNullableRelationOutputColumn,
            ResidentNullableRelationStage, ResidentNullableRelationTarget,
            ResidentNullableRelationshipDomain, ResidentProjectImage,
        },
        graph::{EdgeInput, IndexCatalog, NodeInput, StatisticsSnapshot, TemporalStore},
    };

    use super::*;

    const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());

    fn physical(source: &str, graph: &GraphStore) -> Result<PhysicalPlan> {
        let query = parse(source)?;
        let bound = bind(query, graph.catalog(), BindCapabilities::default())?;
        plan(bound)
    }

    fn writable_physical(source: &str, graph: &GraphStore) -> Result<PhysicalPlan> {
        let query = parse(source)?;
        let bound = bind(
            query,
            graph.catalog(),
            BindCapabilities {
                write: true,
                ..BindCapabilities::default()
            },
        )?;
        plan(bound)
    }

    fn compile_query(source: &str, graph: &GraphStore) -> Result<Option<CompiledResidentRowPlan>> {
        let bookmark = Bookmark {
            term: 3,
            index: graph.revision(),
        };
        compile(
            &physical(source, graph)?,
            PROJECT,
            bookmark,
            graph.catalog(),
            graph,
            &BTreeMap::new(),
            64,
        )
    }

    fn compile_optimized_row_query(
        source: &str,
        graph: &GraphStore,
    ) -> Result<Option<CompiledResidentRowPlan>> {
        let physical = physical(source, graph)?;
        let statistics = StatisticsSnapshot::collect(graph);
        let parameters = BTreeMap::new();
        let (physical, _) = optimize(
            physical,
            OptimizerInput {
                statistics: &statistics,
                catalog: graph.catalog(),
                indexes: None,
                parameters: &parameters,
                backend: BackendKind::Metal,
                scratch_budget_bytes: 64 * 1024 * 1024,
                max_result_rows: 64,
                runtime_feedback: None,
                allow_runtime_checkpoint: true,
            },
        );
        let bookmark = Bookmark {
            term: 3,
            index: graph.revision(),
        };
        compile(
            &physical,
            PROJECT,
            bookmark,
            graph.catalog(),
            graph,
            &parameters,
            64,
        )
    }

    fn compile_nullable_query(
        source: &str,
        graph: &GraphStore,
        max_output_rows: usize,
    ) -> Result<Option<CompiledResidentNullableRelationPlan>> {
        compile_nullable_query_with_parameters(source, graph, &BTreeMap::new(), max_output_rows)
    }

    fn compile_nullable_query_with_parameters(
        source: &str,
        graph: &GraphStore,
        parameters: &BTreeMap<String, ResultValue>,
        max_output_rows: usize,
    ) -> Result<Option<CompiledResidentNullableRelationPlan>> {
        let physical = physical(source, graph)?;
        let statistics = StatisticsSnapshot::collect(graph);
        let (physical, _) = optimize(
            physical,
            OptimizerInput {
                statistics: &statistics,
                catalog: graph.catalog(),
                indexes: None,
                parameters,
                backend: BackendKind::Metal,
                scratch_budget_bytes: 64 * 1024 * 1024,
                max_result_rows: max_output_rows,
                runtime_feedback: None,
                allow_runtime_checkpoint: true,
            },
        );
        let bookmark = Bookmark {
            term: 3,
            index: graph.revision(),
        };
        compile_nullable_relation_with_parameters(
            &physical,
            PROJECT,
            bookmark,
            graph.catalog(),
            graph,
            parameters,
            max_output_rows,
        )
    }

    fn optional_fixture_graph() -> Result<GraphStore> {
        let mut graph = GraphStore::default();
        let single = graph.catalog_mut().intern_label("Single")?;
        let a = graph.catalog_mut().intern_label("A")?;
        let b = graph.catalog_mut().intern_label("B")?;
        let c = graph.catalog_mut().intern_label("C")?;
        let rel = graph.catalog_mut().intern_relationship_type("REL")?;
        for (id, labels) in [(1, vec![single]), (2, vec![a]), (3, vec![b]), (4, vec![c])] {
            graph.insert_node(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: id,
                labels,
                properties: Vec::new(),
            })?;
        }
        graph.insert_edge(EdgeInput {
            id: EdgeId(1),
            source: NodeId(1),
            target: NodeId(2),
            relationship_type: rel,
            layer: Layer::Observed,
            revision: 5,
            properties: Vec::new(),
        })?;
        graph.insert_edge(EdgeInput {
            id: EdgeId(2),
            source: NodeId(2),
            target: NodeId(4),
            relationship_type: rel,
            layer: Layer::Observed,
            revision: 6,
            properties: Vec::new(),
        })?;
        Ok(graph)
    }

    fn fixed_optional_predicate_fixture_graph() -> Result<GraphStore> {
        let mut graph = GraphStore::default();
        let single = graph.catalog_mut().intern_label("Single")?;
        let a = graph.catalog_mut().intern_label("A")?;
        let b = graph.catalog_mut().intern_label("B")?;
        let c = graph.catalog_mut().intern_label("C")?;
        let root = graph.catalog_mut().intern_label("Root")?;
        let text_node = graph.catalog_mut().intern_label("TextNode")?;
        let int_node = graph.catalog_mut().intern_label("IntNode")?;
        let x = graph.catalog_mut().intern_label("X")?;
        let y = graph.catalog_mut().intern_label("Y")?;
        let z = graph.catalog_mut().intern_label("Z")?;
        let t = graph.catalog_mut().intern_relationship_type("T")?;
        let knows = graph.catalog_mut().intern_relationship_type("KNOWS")?;
        let rel = graph.catalog_mut().intern_relationship_type("REL")?;
        let e1 = graph.catalog_mut().intern_relationship_type("E1")?;
        let e2 = graph.catalog_mut().intern_relationship_type("E2")?;
        let name = graph.catalog_mut().intern_property("name")?;
        let num = graph.catalog_mut().intern_property("num")?;
        let val = graph.catalog_mut().intern_property("val")?;
        let var = graph.catalog_mut().intern_property("var")?;

        for (id, labels, properties) in [
            (
                1,
                vec![single, a],
                vec![
                    (name, ScalarValue::String("A".into())),
                    (num, ScalarValue::Integer(1)),
                ],
            ),
            (
                2,
                vec![b],
                vec![
                    (name, ScalarValue::String("B".into())),
                    (num, ScalarValue::Integer(42)),
                ],
            ),
            (3, vec![c], vec![(name, ScalarValue::String("C".into()))]),
            (4, vec![x], vec![(val, ScalarValue::Integer(1))]),
            (5, vec![y], vec![(val, ScalarValue::Integer(2))]),
            (6, vec![z], vec![(val, ScalarValue::Integer(3))]),
            (7, vec![root], vec![(name, ScalarValue::String("x".into()))]),
            (
                8,
                vec![text_node],
                vec![(var, ScalarValue::String("text".into()))],
            ),
            (9, vec![int_node], vec![(var, ScalarValue::Integer(0))]),
        ] {
            graph.insert_node(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: id,
                labels,
                properties,
            })?;
        }

        for (id, source, target, relationship_type, properties) in [
            (1, 1, 2, t, vec![(name, ScalarValue::String("r1".into()))]),
            (2, 1, 2, knows, Vec::new()),
            (3, 2, 3, knows, Vec::new()),
            (4, 2, 3, rel, vec![(name, ScalarValue::String("r2".into()))]),
            (5, 4, 5, e1, Vec::new()),
            (6, 5, 6, e2, Vec::new()),
            (7, 7, 8, t, Vec::new()),
            (8, 7, 9, t, Vec::new()),
        ] {
            graph.insert_edge(EdgeInput {
                id: EdgeId(id),
                source: NodeId(source),
                target: NodeId(target),
                relationship_type,
                layer: Layer::Observed,
                revision: 9 + id,
                properties,
            })?;
        }
        Ok(graph)
    }

    fn relationship_property_fixture_graph() -> Result<GraphStore> {
        let mut graph = GraphStore::default();
        let a = graph.catalog_mut().intern_label("A")?;
        let b = graph.catalog_mut().intern_label("B")?;
        let relationship = graph.catalog_mut().intern_relationship_type("REL")?;
        let name = graph.catalog_mut().intern_property("name")?;
        let weight = graph.catalog_mut().intern_property("weight")?;
        graph.insert_node(NodeInput {
            id: NodeId(1),
            layer: Layer::Observed,
            revision: 1,
            labels: vec![a],
            properties: Vec::new(),
        })?;
        graph.insert_node(NodeInput {
            id: NodeId(2),
            layer: Layer::Observed,
            revision: 2,
            labels: vec![b],
            properties: Vec::new(),
        })?;
        graph.insert_edge(EdgeInput {
            id: EdgeId(1),
            source: NodeId(1),
            target: NodeId(2),
            relationship_type: relationship,
            layer: Layer::Observed,
            revision: 3,
            properties: vec![
                (name, ScalarValue::String("r".into())),
                (weight, ScalarValue::Integer(7)),
            ],
        })?;
        Ok(graph)
    }

    fn nullable_typed_projection_fixture_graph() -> Result<GraphStore> {
        let mut graph = GraphStore::default();
        let a = graph.catalog_mut().intern_relationship_type("A")?;
        let b = graph.catalog_mut().intern_relationship_type("B")?;
        let name = graph.catalog_mut().intern_property("name")?;
        let score = graph.catalog_mut().intern_property("score")?;
        let title = graph.catalog_mut().intern_property("title")?;
        let weight = graph.catalog_mut().intern_property("weight")?;
        let float = graph.catalog_mut().intern_property("float")?;
        let node_mixed = graph.catalog_mut().intern_property("node_mixed")?;
        let relationship_mixed = graph.catalog_mut().intern_property("relationship_mixed")?;
        graph.catalog_mut().intern_property("declared_null")?;

        graph.insert_node(NodeInput {
            id: NodeId(1),
            layer: Layer::Observed,
            revision: 1,
            labels: Vec::new(),
            properties: vec![
                (name, ScalarValue::String("a".into())),
                (score, ScalarValue::Integer(7)),
                (float, ScalarValue::Float(1.5.into())),
                (node_mixed, ScalarValue::Integer(1)),
            ],
        })?;
        graph.insert_node(NodeInput {
            id: NodeId(2),
            layer: Layer::Observed,
            revision: 2,
            labels: Vec::new(),
            properties: vec![
                (name, ScalarValue::String("wide-é".into())),
                (node_mixed, ScalarValue::String("mixed".into())),
            ],
        })?;
        graph.insert_edge(EdgeInput {
            id: EdgeId(1),
            source: NodeId(1),
            target: NodeId(2),
            relationship_type: a,
            layer: Layer::Observed,
            revision: 3,
            properties: vec![
                (title, ScalarValue::String("edge".into())),
                (weight, ScalarValue::Integer(11)),
                (float, ScalarValue::Float(2.5.into())),
                (relationship_mixed, ScalarValue::Integer(2)),
            ],
        })?;
        graph.insert_edge(EdgeInput {
            id: EdgeId(2),
            source: NodeId(2),
            target: NodeId(1),
            relationship_type: b,
            layer: Layer::Observed,
            revision: 4,
            properties: vec![
                (title, ScalarValue::String("relationship-δ".into())),
                (relationship_mixed, ScalarValue::String("mixed".into())),
            ],
        })?;
        Ok(graph)
    }

    fn nullable_stage_kinds(
        plan: &CompiledResidentNullableRelationPlan,
    ) -> Vec<ResidentNullableRelationObligationKind> {
        plan.request
            .obligations()
            .map(|obligation| obligation.kind)
            .collect()
    }

    fn nullable_filter_placement_counts(
        plan: &CompiledResidentNullableRelationPlan,
    ) -> (usize, usize, usize) {
        plan.request.predicate_program.filters.iter().fold(
            (0, 0, 0),
            |(relation, optional_stage, optional_group), filter| match filter.placement {
                ResidentNullableRelationFilterPlacement::RelationAfter { .. } => {
                    (relation + 1, optional_stage, optional_group)
                }
                ResidentNullableRelationFilterPlacement::OptionalCandidates { .. } => {
                    (relation, optional_stage + 1, optional_group)
                }
                ResidentNullableRelationFilterPlacement::OptionalGroupCandidates { .. } => {
                    (relation, optional_stage, optional_group + 1)
                }
            },
        )
    }

    #[test]
    fn two_label_entity_union_uses_one_complete_native_or_scan() -> Result<()> {
        let mut graph = GraphStore::default();
        let a = graph.catalog_mut().intern_label("A")?;
        let b = graph.catalog_mut().intern_label("B")?;
        for (id, labels) in [(1_u64, vec![a]), (2, vec![b])] {
            graph.insert_node(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: 1,
                labels,
                properties: Vec::new(),
            })?;
        }
        let distinct = "MATCH (a:A) RETURN a AS a UNION MATCH (b:B) RETURN b AS a";
        let all = "MATCH (a:A) RETURN a AS a UNION ALL MATCH (b:B) RETURN b AS a";
        for source in [distinct, all] {
            let compiled = compile_nullable_query(source, &graph, 64)?.ok_or_else(|| {
                Error::internal(format!("two-label UNION was not compiled: {source}"))
            })?;
            compiled.request.validate()?;
            assert!(matches!(
                compiled.outputs.as_slice(),
                [CompiledResidentNullableRelationOutput {
                    name,
                    source: ResidentNullableRelationOutputSource::Entity {
                        kind: ResidentNullableRelationBindingKind::Node,
                        ..
                    },
                }] if name == "a"
            ));
            assert_eq!(
                execute_nullable_relation_on_cpu(&graph, &compiled)?.row_count(),
                2
            );
        }

        graph.insert_node(NodeInput {
            id: NodeId(3),
            layer: Layer::Observed,
            revision: 2,
            labels: vec![a, b],
            properties: Vec::new(),
        })?;
        let distinct_overlap = compile_nullable_query(distinct, &graph, 64)?
            .ok_or_else(|| Error::internal("overlapping UNION DISTINCT was rejected"))?;
        assert_eq!(
            execute_nullable_relation_on_cpu(&graph, &distinct_overlap)?.row_count(),
            3
        );
        assert!(
            compile_nullable_query(all, &graph, 64)?.is_none(),
            "UNION ALL lost duplicate membership for an overlapping label domain"
        );
        Ok(())
    }

    #[test]
    fn fixed_one_hop_named_path_length_is_a_sealed_native_integer_filter() -> Result<()> {
        let graph = optional_fixture_graph()?;
        for (expected_length, expected_rows) in [(1_i64, 2_usize), (10_i64, 0_usize)] {
            let query = format!("MATCH p = (n)-->(x) WHERE length(p) = {expected_length} RETURN x");
            let compiled = compile_nullable_query(&query, &graph, 64)?.ok_or_else(|| {
                Error::internal("fixed one-hop path length did not lower to a nullable relation")
            })?;
            compiled.request.validate()?;
            assert!(matches!(
                compiled.request.program.stages.as_slice(),
                [
                    ResidentNullableRelationStage::NodeScan { .. },
                    ResidentNullableRelationStage::Expand { .. },
                    ResidentNullableRelationStage::FinalProject { .. },
                ]
            ));
            let [filter] = compiled.request.predicate_program.filters.as_slice() else {
                return Err(Error::internal(
                    "fixed path length did not emit exactly one native filter",
                ));
            };
            assert_eq!(
                filter.placement,
                ResidentNullableRelationFilterPlacement::RelationAfter { stage: 1 }
            );
            assert!(matches!(
                &filter.predicate,
                ResidentNullableRelationPredicate::CompareInteger {
                    left: ResidentNullableRelationPredicateValue::Integer(1),
                    operation: CompareOp::Eq,
                    right: ResidentNullableRelationPredicateValue::Integer(value),
                } if *value == expected_length
            ));

            let result = execute_nullable_relation_on_cpu(&graph, &compiled)?;
            assert_eq!(result.row_count(), expected_rows);
        }

        for unsupported in [
            "MATCH p = (n)-->(x) RETURN p",
            "MATCH p = (n)-->(x) RETURN *",
            "MATCH p = (n)-->(x)-->(y) WHERE length(p) = 2 RETURN y",
            "MATCH p = (n)-[*]->(x) WHERE length(p) = 1 RETURN x",
            "MATCH (n) OPTIONAL MATCH p = (n)-->(x) MATCH (y) WHERE length(p) = 1 RETURN x",
        ] {
            assert!(
                compile_nullable_query(unsupported, &graph, 64)?.is_none(),
                "unsupported named-path shape escaped fail-closed admission: {unsupported}"
            );
        }
        Ok(())
    }

    #[test]
    fn exact_first_nine_optional_ids_lower_to_complete_staged_relations() -> Result<()> {
        use ResidentNullableRelationObligationKind as Kind;

        let empty = GraphStore::default();
        let graph = optional_fixture_graph()?;
        let cases: [(usize, &str, &GraphStore, &[Kind]); 9] = [
            (
                370,
                "OPTIONAL MATCH (a) WITH a MATCH (a)-->(b) RETURN b",
                &empty,
                &[
                    Kind::OptionalNodeScan,
                    Kind::ScopeProjection,
                    Kind::MandatoryExpand,
                    Kind::FinalProjection,
                ],
            ),
            (
                371,
                "OPTIONAL MATCH (a:TheLabel) WITH a MATCH (a)-->(b) RETURN b",
                &graph,
                &[
                    Kind::OptionalNodeScan,
                    Kind::ScopeProjection,
                    Kind::MandatoryExpand,
                    Kind::FinalProjection,
                ],
            ),
            (
                511,
                "MATCH (n) OPTIONAL MATCH (n)-[:NOT_EXIST]->(x) RETURN n, x",
                &graph,
                &[
                    Kind::MandatoryNodeScan,
                    Kind::OptionalExpand,
                    Kind::FinalProjection,
                ],
            ),
            (
                519,
                "OPTIONAL MATCH (a) WITH a OPTIONAL MATCH (a)-->(b) RETURN b",
                &empty,
                &[
                    Kind::OptionalNodeScan,
                    Kind::ScopeProjection,
                    Kind::OptionalExpand,
                    Kind::FinalProjection,
                ],
            ),
            (
                530,
                "OPTIONAL MATCH (a:NotThere) OPTIONAL MATCH (b:NotThere) WITH a, b OPTIONAL MATCH (b)-[r:NOR_THIS]->(a) RETURN a, b, r",
                &graph,
                &[
                    Kind::OptionalNodeScan,
                    Kind::OptionalNodeScan,
                    Kind::ScopeProjection,
                    Kind::OptionalExpand,
                    Kind::FinalProjection,
                ],
            ),
            (
                536,
                "OPTIONAL MATCH (a:NotThere) WITH a MATCH (b:B) WITH a, b OPTIONAL MATCH (b)-[r:NOR_THIS]->(a) RETURN a, b, r",
                &graph,
                &[
                    Kind::OptionalNodeScan,
                    Kind::ScopeProjection,
                    Kind::MandatoryNodeScan,
                    Kind::ScopeProjection,
                    Kind::OptionalExpand,
                    Kind::FinalProjection,
                ],
            ),
            (
                537,
                "MATCH (n:Single) OPTIONAL MATCH (n)-[r]-(m:NonExistent) RETURN r",
                &graph,
                &[
                    Kind::MandatoryNodeScan,
                    Kind::OptionalExpand,
                    Kind::FinalProjection,
                ],
            ),
            (
                907,
                "OPTIONAL MATCH (a:Start) WITH a MATCH (a)-->(b) RETURN *",
                &empty,
                &[
                    Kind::OptionalNodeScan,
                    Kind::ScopeProjection,
                    Kind::MandatoryExpand,
                    Kind::FinalProjection,
                ],
            ),
            (
                908,
                "OPTIONAL MATCH (a:A) WITH a AS a MATCH (b:B) RETURN a, b",
                &graph,
                &[
                    Kind::OptionalNodeScan,
                    Kind::ScopeProjection,
                    Kind::MandatoryNodeScan,
                    Kind::FinalProjection,
                ],
            ),
        ];

        for (report_id, query, graph, expected) in cases {
            let compiled = compile_nullable_query(query, graph, 64)?.ok_or_else(|| {
                Error::internal(format!(
                    "OPTIONAL report {report_id} did not lower to the staged relation ABI"
                ))
            })?;
            compiled.request.validate()?;
            assert_eq!(
                nullable_stage_kinds(&compiled),
                expected,
                "wrong staged lowering for OPTIONAL report {report_id}"
            );
            assert_eq!(
                compiled.request.manifest.obligations.len(),
                compiled.request.program.stages.len()
            );
            assert_ne!(compiled.request.manifest.fingerprint.0, [0; 32]);
        }
        Ok(())
    }

    #[test]
    fn first_nine_domains_distinguish_known_empty_from_unsupported() -> Result<()> {
        let graph = optional_fixture_graph()?;

        let report_371 = compile_nullable_query(
            "OPTIONAL MATCH (a:TheLabel) WITH a MATCH (a)-->(b) RETURN b",
            &graph,
            64,
        )?
        .ok_or_else(|| Error::internal("report 371 did not compile"))?;
        assert!(matches!(
            &report_371.request.program.stages[0],
            ResidentNullableRelationStage::NodeScan {
                mode: ResidentNullableRelationMatchMode::Optional,
                labels: ResidentNullableNodeDomain::KnownEmpty,
                ..
            }
        ));
        assert_eq!(report_371.request.capacities.stage_candidate_rows[0], 0);
        assert_eq!(report_371.request.capacities.stage_output_rows[0], 1);

        let report_511 = compile_nullable_query(
            "MATCH (n) OPTIONAL MATCH (n)-[:NOT_EXIST]->(x) RETURN n, x",
            &graph,
            64,
        )?
        .ok_or_else(|| Error::internal("report 511 did not compile"))?;
        assert!(matches!(
            &report_511.request.program.stages[1],
            ResidentNullableRelationStage::Expand {
                mode: ResidentNullableRelationMatchMode::Optional,
                relationship_types: ResidentNullableRelationshipDomain::KnownEmpty,
                ..
            }
        ));
        assert_eq!(report_511.request.capacities.stage_candidate_rows[1], 0);
        assert_eq!(
            report_511.request.capacities.stage_output_rows[1],
            report_511.request.capacities.stage_input_rows[1]
        );

        let report_530 = compile_nullable_query(
            "OPTIONAL MATCH (a:NotThere) OPTIONAL MATCH (b:NotThere) WITH a, b OPTIONAL MATCH (b)-[r:NOR_THIS]->(a) RETURN a, b, r",
            &graph,
            64,
        )?
        .ok_or_else(|| Error::internal("report 530 did not compile"))?;
        assert!(matches!(
            &report_530.request.program.stages[0],
            ResidentNullableRelationStage::NodeScan {
                labels: ResidentNullableNodeDomain::KnownEmpty,
                ..
            }
        ));
        assert!(matches!(
            &report_530.request.program.stages[1],
            ResidentNullableRelationStage::NodeScan {
                labels: ResidentNullableNodeDomain::KnownEmpty,
                ..
            }
        ));
        assert!(matches!(
            &report_530.request.program.stages[3],
            ResidentNullableRelationStage::Expand {
                target: ResidentNullableRelationTarget::Existing(_),
                relationship_types: ResidentNullableRelationshipDomain::KnownEmpty,
                ..
            }
        ));

        let report_537 = compile_nullable_query(
            "MATCH (n:Single) OPTIONAL MATCH (n)-[r]-(m:NonExistent) RETURN r",
            &graph,
            64,
        )?
        .ok_or_else(|| Error::internal("report 537 did not compile"))?;
        assert!(matches!(
            &report_537.request.program.stages[1],
            ResidentNullableRelationStage::Expand {
                mode: ResidentNullableRelationMatchMode::Optional,
                target_labels: ResidentNullableNodeDomain::KnownEmpty,
                ..
            }
        ));
        Ok(())
    }

    #[test]
    fn graph4_null_type_reuses_only_a_known_empty_relationship_type_source() -> Result<()> {
        let mut graph = GraphStore::default();
        graph.insert_node(NodeInput {
            id: NodeId(1),
            layer: Layer::Observed,
            revision: 1,
            labels: Vec::new(),
            properties: Vec::new(),
        })?;
        graph.catalog_mut().intern_relationship_type("T")?;

        let compiled = compile_nullable_query(
            "MATCH (a) OPTIONAL MATCH (a)-[r:NOT_THERE]->() RETURN type(r), type(null)",
            &graph,
            64,
        )?
        .ok_or_else(|| Error::internal("Graph4 [3] did not compile as one nullable command"))?;
        compiled.request.validate()?;
        let relationship = compiled
            .request
            .program
            .stages
            .iter()
            .find_map(|stage| match stage {
                ResidentNullableRelationStage::Expand {
                    mode: ResidentNullableRelationMatchMode::Optional,
                    relationship: Some(relationship),
                    relationship_types: ResidentNullableRelationshipDomain::KnownEmpty,
                    ..
                } => Some(*relationship),
                _ => None,
            })
            .ok_or_else(|| {
                Error::internal("Graph4 [3] omitted its known-empty relationship proof")
            })?;
        let Some(ResidentNullableRelationStage::FinalProject { bindings }) =
            compiled.request.program.stages.last()
        else {
            return Err(Error::internal("Graph4 [3] omitted its final projection"));
        };
        assert_eq!(
            bindings
                .iter()
                .map(|binding| (binding.name.as_str(), binding.source.clone()))
                .collect::<Vec<_>>(),
            [
                (
                    "type(r)",
                    ResidentNullableRelationOutputSource::RelationshipType { slot: relationship },
                ),
                (
                    "type(null)",
                    ResidentNullableRelationOutputSource::RelationshipType { slot: relationship },
                ),
            ]
        );
        assert_eq!(compiled.outputs.len(), 2);

        for query in [
            "MATCH (a) OPTIONAL MATCH (a)-[r:T]->() RETURN type(null)",
            "MATCH (a) OPTIONAL MATCH (a)-[r]->() RETURN type(null)",
            "MATCH (a) OPTIONAL MATCH (a)-[r]->(:NOT_THERE) RETURN type(null)",
            "MATCH (a) OPTIONAL MATCH (a)-[r:NOT_THERE]->() RETURN DISTINCT type(null)",
            "MATCH (a) OPTIONAL MATCH (a)-[r:NOT_THERE]->() WITH [r, 1] AS list RETURN type(list[0])",
            "RETURN type(null)",
        ] {
            assert!(
                compile_nullable_query(query, &graph, 64)?.is_none(),
                "unsupported type(null) shape escaped fail-closed admission: {query}"
            );
        }
        Ok(())
    }

    #[test]
    fn graph4_mixed_relationship_type_keeps_token_and_null_sentinel_on_one_native_column()
    -> Result<()> {
        let mut graph = GraphStore::default();
        let relationship_type = graph.catalog_mut().intern_relationship_type("T")?;
        for id in 1..=2 {
            graph.insert_node(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: id,
                labels: Vec::new(),
                properties: Vec::new(),
            })?;
        }
        graph.insert_edge(EdgeInput {
            id: EdgeId(1),
            source: NodeId(1),
            target: NodeId(2),
            relationship_type,
            layer: Layer::Observed,
            revision: 3,
            properties: Vec::new(),
        })?;

        let query = "MATCH (a) OPTIONAL MATCH (a)-[r:T]->() RETURN type(r)";
        let compiled = compile_nullable_query(query, &graph, 64)?
            .ok_or_else(|| Error::internal("Graph4 [4] did not compile as one nullable command"))?;
        compiled.request.validate()?;
        let [
            ResidentNullableRelationStage::NodeScan { .. },
            ResidentNullableRelationStage::Expand {
                mode: ResidentNullableRelationMatchMode::Optional,
                relationship: Some(relationship),
                relationship_types: ResidentNullableRelationshipDomain::Known(types),
                ..
            },
            ResidentNullableRelationStage::FinalProject { bindings },
        ] = compiled.request.program.stages.as_slice()
        else {
            return Err(Error::internal(
                "Graph4 [4] changed its sealed optional-expansion shape",
            ));
        };
        assert_eq!(types, &[relationship_type]);
        assert!(matches!(
            bindings.as_slice(),
            [ResidentNullableRelationOutputBinding {
                name,
                source: ResidentNullableRelationOutputSource::RelationshipType { slot },
            }] if name == "type(r)" && slot == relationship
        ));
        assert_eq!(compiled.request.capacities.final_output_columns, 1);

        let result = execute_nullable_relation_on_cpu(&graph, &compiled)?;
        assert_eq!(result.row_count(), 2);
        assert!(matches!(
            result.columns(),
            [ResidentNullableRelationOutputColumn::RelationshipType {
                slot,
                source_rows,
                relationship_types,
            }] if slot == relationship
                && source_rows
                    == &[0, crate::execution::RESIDENT_NULLABLE_RELATION_NULL_ROW]
                && relationship_types
                    == &[relationship_type, crate::types::RelationshipTypeId(0)]
        ));

        for unsupported in [
            "MATCH (a) OPTIONAL MATCH (a)-[r:T]->() RETURN DISTINCT type(r)",
            "MATCH (a) OPTIONAL MATCH (a)-[r:T]->() RETURN type(r) ORDER BY type(r)",
            "MATCH (a) OPTIONAL MATCH (a)-[r:T]->() RETURN toString(type(r))",
            "MATCH (a) OPTIONAL MATCH (a)-[r:T]->() RETURN type(r), 1",
        ] {
            assert!(
                compile_nullable_query(unsupported, &graph, 64)?.is_none(),
                "unsupported neighboring type() shape escaped admission: {unsupported}"
            );
        }
        Ok(())
    }

    #[test]
    fn graph4_any_list_zero_type_reuses_the_projected_relationship_slot() -> Result<()> {
        let mut graph = GraphStore::default();
        let relationship_type = graph.catalog_mut().intern_relationship_type("T")?;
        for id in 1..=2 {
            graph.insert_node(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: id,
                labels: Vec::new(),
                properties: Vec::new(),
            })?;
        }
        graph.insert_edge(EdgeInput {
            id: EdgeId(1),
            source: NodeId(1),
            target: NodeId(2),
            relationship_type,
            layer: Layer::Observed,
            revision: 3,
            properties: Vec::new(),
        })?;

        let query = "MATCH (a)-[r]->() WITH [r, 1] AS list RETURN type(list[0])";
        let compiled = compile_nullable_query(query, &graph, 64)?
            .ok_or_else(|| Error::internal("Graph4 [5] did not compile as one nullable command"))?;
        compiled.request.validate()?;
        let [
            ResidentNullableRelationStage::NodeScan { .. },
            ResidentNullableRelationStage::Expand {
                mode: ResidentNullableRelationMatchMode::Mandatory,
                relationship: Some(relationship),
                relationship_types: ResidentNullableRelationshipDomain::Any,
                ..
            },
            ResidentNullableRelationStage::ScopeProject {
                bindings: scope_bindings,
            },
            ResidentNullableRelationStage::FinalProject {
                bindings: output_bindings,
            },
        ] = compiled.request.program.stages.as_slice()
        else {
            return Err(Error::internal(
                "Graph4 [5] changed its sealed relationship-list projection shape",
            ));
        };
        let [scope_binding] = scope_bindings.as_slice() else {
            return Err(Error::internal(
                "Graph4 [5] changed its one-column WITH boundary",
            ));
        };
        assert_eq!(scope_binding.variable, "list");
        assert_eq!(scope_binding.source, *relationship);
        assert_eq!(scope_binding.row_limit, None);
        let [output_binding] = output_bindings.as_slice() else {
            return Err(Error::internal(
                "Graph4 [5] changed its one-column final projection",
            ));
        };
        assert_eq!(output_binding.name, "type(list[0])");
        assert_eq!(
            output_binding.source,
            ResidentNullableRelationOutputSource::RelationshipType {
                slot: scope_binding.output,
            }
        );
        assert_eq!(compiled.outputs.len(), 1);
        assert_eq!(compiled.outputs[0].name, "type(list[0])");
        assert!(compiled.request.property_lanes().is_empty());

        let result = execute_nullable_relation_on_cpu(&graph, &compiled)?;
        assert_eq!(result.row_count(), 1);
        assert!(matches!(
            result.columns(),
            [ResidentNullableRelationOutputColumn::RelationshipType {
                slot,
                source_rows,
                relationship_types,
            }] if *slot == scope_binding.output
                && source_rows == &[0]
                && relationship_types == &[relationship_type]
        ));

        for unsupported in [
            "MATCH (a)-[r]->() WITH [1, r] AS list RETURN type(list[0])",
            "MATCH (a)-[r]->() WITH [r] AS list RETURN type(list[0])",
            "MATCH (a)-[r]->() WITH [r, 1, 2] AS list RETURN type(list[0])",
            "MATCH (a)-[r]->() WITH [r, 1] AS list RETURN type(list[1])",
            "MATCH (a)-[r]->() WITH [r, 1] AS list LIMIT 1 RETURN type(list[0])",
            "MATCH (a)-[r]->() WITH [r, 1] AS list RETURN type(list[0]) AS t",
            "MATCH (a)<-[r]-() WITH [r, 1] AS list RETURN type(list[0])",
            "MATCH (a)-[r:T]->() WITH [r, 1] AS list RETURN type(list[0])",
            "MATCH (a)-[r]->(b) WITH [r, 1] AS list RETURN type(list[0])",
        ] {
            assert!(
                compile_nullable_query(unsupported, &graph, 64)?.is_none(),
                "unsupported relationship-list type() shape escaped admission: {unsupported}"
            );
        }
        Ok(())
    }

    #[test]
    fn with_alias_and_return_star_are_explicit_slot_projections() -> Result<()> {
        let graph = optional_fixture_graph()?;
        let report_908 = compile_nullable_query(
            "OPTIONAL MATCH (a:A) WITH a AS a MATCH (b:B) RETURN a, b",
            &graph,
            64,
        )?
        .ok_or_else(|| Error::internal("report 908 did not compile"))?;
        let ResidentNullableRelationStage::ScopeProject { bindings } =
            &report_908.request.program.stages[1]
        else {
            return Err(Error::internal("report 908 omitted WITH projection"));
        };
        assert_eq!(bindings.len(), 1);
        assert_eq!(bindings[0].variable, "a");
        assert_ne!(bindings[0].source, bindings[0].output);
        assert_eq!(
            report_908
                .outputs
                .iter()
                .map(|output| output.name.as_str())
                .collect::<Vec<_>>(),
            ["a", "b"]
        );

        let empty = GraphStore::default();
        let report_907 = compile_nullable_query(
            "OPTIONAL MATCH (a:Start) WITH a MATCH (a)-->(b) RETURN *",
            &empty,
            64,
        )?
        .ok_or_else(|| Error::internal("report 907 did not compile"))?;
        assert_eq!(
            report_907
                .outputs
                .iter()
                .map(|output| output.name.as_str())
                .collect::<Vec<_>>(),
            ["a", "b"]
        );
        Ok(())
    }

    #[test]
    fn exact_six_case_scalar_scope_manifest_executes_and_near_misses_fail_closed() -> Result<()> {
        let mut graph = GraphStore::default();
        let begin = graph.catalog_mut().intern_label("Begin")?;
        let end = graph.catalog_mut().intern_label("End")?;
        let name2 = graph.catalog_mut().intern_property("name2")?;
        let num = graph.catalog_mut().intern_property("num")?;
        let id = graph.catalog_mut().intern_property("id")?;
        for (node_id, labels, properties) in [
            (
                1,
                Vec::new(),
                vec![(name2, ScalarValue::String("A".into()))],
            ),
            (
                2,
                Vec::new(),
                vec![(name2, ScalarValue::String("A".into()))],
            ),
            (
                3,
                Vec::new(),
                vec![(name2, ScalarValue::String("B".into()))],
            ),
            (
                4,
                Vec::new(),
                vec![(name2, ScalarValue::String("C".into()))],
            ),
            (5, vec![begin], vec![(num, ScalarValue::Integer(0))]),
            (6, vec![begin], vec![(num, ScalarValue::Integer(42))]),
            (
                7,
                vec![end],
                vec![
                    (num, ScalarValue::Integer(42)),
                    (id, ScalarValue::Integer(0)),
                ],
            ),
            (8, vec![end], vec![(num, ScalarValue::Integer(3))]),
        ] {
            graph.insert_node(NodeInput {
                id: NodeId(node_id),
                layer: Layer::Observed,
                revision: node_id,
                labels,
                properties,
            })?;
        }

        let manifest = [
            (
                908,
                "MATCH (a:Begin) WITH a.num AS property MATCH (b) WHERE b.id = property RETURN b",
                "b",
                1,
            ),
            (
                912,
                "MATCH (a:Begin) WITH a.num AS property MATCH (b:End) WHERE property = b.num RETURN b",
                "b",
                1,
            ),
            (
                1226,
                "MATCH (a:Begin) WITH a.num AS property LIMIT 1 MATCH (b) WHERE b.id = property RETURN b",
                "b",
                1,
            ),
            (
                1233,
                "MATCH (a) WITH DISTINCT a.name2 AS name WHERE a.name2 = 'B' RETURN *",
                "name",
                1,
            ),
            (
                1249,
                "MATCH (a) WITH a.name2 AS name WHERE name = 'B' RETURN *",
                "name",
                1,
            ),
            (
                1250,
                "MATCH (a) WITH a.name2 AS name WHERE name = 'B' OR a.name2 = 'C' RETURN *",
                "name",
                2,
            ),
        ];
        for (report_id, query, output, expected_rows) in manifest {
            let compiled = compile_nullable_query(query, &graph, 64)?.ok_or_else(|| {
                Error::internal(format!(
                    "scalar-scope report {report_id} did not enter the nullable route"
                ))
            })?;
            compiled.request.validate()?;
            assert_eq!(
                compiled
                    .outputs
                    .iter()
                    .map(|binding| binding.name.as_str())
                    .collect::<Vec<_>>(),
                [output],
                "report {report_id} changed its visible schema"
            );
            assert_eq!(
                execute_nullable_relation_on_cpu(&graph, &compiled)?.row_count(),
                expected_rows,
                "report {report_id} changed its exact CPU-reference cardinality"
            );
            assert_eq!(
                compiled.request.predicate_program.filters.len(),
                1,
                "report {report_id} did not seal its one complete predicate"
            );
        }

        for query in [
            "MATCH (a) WITH a.name2 AS name WHERE name = 'B' AND a.name2 = 'C' RETURN *",
            "MATCH (a) WITH a.name2 AS name WHERE name = 'A' OR name = 'B' OR name = 'C' RETURN *",
            "MATCH (a) WITH DISTINCT a.name2 AS name WHERE name = 'B' RETURN *",
            "MATCH (a) WITH DISTINCT a.name2 AS name WHERE a.name2 > 'B' RETURN *",
            "MATCH (a:Begin) WITH a.num + 1 AS property MATCH (b) WHERE b.id = property RETURN b",
            "MATCH (a:Begin) WITH a.num AS property, a MATCH (b) WHERE b.id = property RETURN b",
            "MATCH (a:Begin) WITH a.num AS property MATCH (b) WHERE b.id > property RETURN b",
            "MATCH (a:Begin) WITH a.num AS property MATCH (b:Other) WHERE b.id = property RETURN b",
            "MATCH (a:Begin) WITH a.num AS property LIMIT 2 MATCH (b) WHERE b.id = property RETURN b",
        ] {
            assert!(
                compile_nullable_query(query, &graph, 64)?.is_none(),
                "nearby scalar-scope shape escaped fail-closed admission: {query}"
            );
        }
        Ok(())
    }

    #[test]
    fn intermediate_relation_capacity_is_not_truncated_by_client_output_budget() -> Result<()> {
        let graph = optional_fixture_graph()?;
        let compiled = compile_nullable_query("MATCH (a) MATCH (b) RETURN a, b", &graph, 1)?
            .ok_or_else(|| Error::internal("two-scan relation did not compile"))?;
        assert_eq!(compiled.request.capacities.max_output_rows, 1);
        assert_eq!(compiled.request.capacities.stage_input_rows, [1, 4, 16]);
        assert_eq!(
            compiled.request.capacities.stage_candidate_rows,
            [4, 16, 16]
        );
        assert_eq!(compiled.request.capacities.stage_output_rows, [4, 16, 16]);
        Ok(())
    }

    #[test]
    fn match3_cyclic_node_property_returns_lower_to_typed_native_outputs() -> Result<()> {
        let graph = nullable_typed_projection_fixture_graph()?;
        let name = graph
            .catalog()
            .property("name")
            .ok_or_else(|| Error::internal("typed fixture omitted `name`"))?;
        let maximum_bytes = u32::try_from("wide-é".len()).expect("fixture string is bounded");

        for query in [
            "MATCH (a)-[:A]->()-[:B]->(a) RETURN a.name",
            "MATCH (a)-[:A]->(b), (b)-[:B]->(a) RETURN a.name",
        ] {
            let compiled = compile_nullable_query(query, &graph, 64)?.ok_or_else(|| {
                Error::internal(format!("cyclic property query did not compile: {query}"))
            })?;
            compiled.request.validate()?;
            let Some(ResidentNullableRelationStage::FinalProject { bindings }) =
                compiled.request.program.stages.last()
            else {
                return Err(Error::internal(
                    "cyclic property query omitted final projection",
                ));
            };
            let [binding] = bindings.as_slice() else {
                return Err(Error::internal(
                    "cyclic property query emitted the wrong output arity",
                ));
            };
            assert_eq!(binding.name, "a.name");
            assert!(matches!(
                binding.source,
                ResidentNullableRelationOutputSource::StringProperty {
                    kind: ResidentNullableRelationBindingKind::Node,
                    property,
                    maximum_bytes: width,
                    ..
                } if property == name && width == maximum_bytes
            ));
            assert_eq!(compiled.outputs.len(), 1);
            assert_eq!(compiled.outputs[0].name, binding.name);
            assert_eq!(compiled.outputs[0].source, binding.source);
        }
        Ok(())
    }

    #[test]
    fn final_projection_seals_entity_integer_string_and_absent_property_sources() -> Result<()> {
        let graph = nullable_typed_projection_fixture_graph()?;
        let score = graph
            .catalog()
            .property("score")
            .ok_or_else(|| Error::internal("typed fixture omitted `score`"))?;
        let name = graph
            .catalog()
            .property("name")
            .ok_or_else(|| Error::internal("typed fixture omitted `name`"))?;
        let weight = graph
            .catalog()
            .property("weight")
            .ok_or_else(|| Error::internal("typed fixture omitted `weight`"))?;
        let title = graph
            .catalog()
            .property("title")
            .ok_or_else(|| Error::internal("typed fixture omitted `title`"))?;
        let compiled = compile_nullable_query(
            "MATCH (n)-[r:A]->(m) RETURN n, n.score AS node_score, n.name AS node_name, n.not_declared AS missing_node, r, r.weight AS relationship_weight, r.title AS relationship_title, r.not_declared AS missing_relationship",
            &graph,
            64,
        )?
        .ok_or_else(|| Error::internal("typed node/relationship projection did not compile"))?;
        compiled.request.validate()?;
        let Some(ResidentNullableRelationStage::FinalProject { bindings }) =
            compiled.request.program.stages.last()
        else {
            return Err(Error::internal("typed projection omitted final stage"));
        };
        assert_eq!(bindings.len(), 8);

        let ResidentNullableRelationOutputSource::Entity {
            slot: node_slot,
            kind: ResidentNullableRelationBindingKind::Node,
        } = bindings[0].source
        else {
            return Err(Error::internal("node entity output has the wrong source"));
        };
        assert!(matches!(
            bindings[1].source,
            ResidentNullableRelationOutputSource::IntegerProperty {
                slot,
                kind: ResidentNullableRelationBindingKind::Node,
                property,
            } if slot == node_slot && property == score
        ));
        assert!(matches!(
            bindings[2].source,
            ResidentNullableRelationOutputSource::StringProperty {
                slot,
                kind: ResidentNullableRelationBindingKind::Node,
                property,
                maximum_bytes,
            } if slot == node_slot
                && property == name
                && maximum_bytes == u32::try_from("wide-é".len()).expect("bounded fixture")
        ));
        assert!(matches!(
            bindings[3].source,
            ResidentNullableRelationOutputSource::NullProperty {
                slot,
                kind: ResidentNullableRelationBindingKind::Node,
            } if slot == node_slot
        ));

        let ResidentNullableRelationOutputSource::Entity {
            slot: relationship_slot,
            kind: ResidentNullableRelationBindingKind::Relationship,
        } = bindings[4].source
        else {
            return Err(Error::internal(
                "relationship entity output has the wrong source",
            ));
        };
        assert!(matches!(
            bindings[5].source,
            ResidentNullableRelationOutputSource::IntegerProperty {
                slot,
                kind: ResidentNullableRelationBindingKind::Relationship,
                property,
            } if slot == relationship_slot && property == weight
        ));
        assert!(matches!(
            bindings[6].source,
            ResidentNullableRelationOutputSource::StringProperty {
                slot,
                kind: ResidentNullableRelationBindingKind::Relationship,
                property,
                maximum_bytes,
            } if slot == relationship_slot
                && property == title
                && maximum_bytes
                    == u32::try_from("relationship-δ".len()).expect("bounded fixture")
        ));
        assert!(matches!(
            bindings[7].source,
            ResidentNullableRelationOutputSource::NullProperty {
                slot,
                kind: ResidentNullableRelationBindingKind::Relationship,
            } if slot == relationship_slot
        ));
        assert_eq!(compiled.request.property_lanes().len(), 4);
        assert_eq!(
            compiled
                .outputs
                .iter()
                .map(|output| (&output.name, output.source.clone()))
                .collect::<Vec<_>>(),
            bindings
                .iter()
                .map(|binding| (&binding.name, binding.source.clone()))
                .collect::<Vec<_>>()
        );
        Ok(())
    }

    #[test]
    fn final_property_projection_distinguishes_absence_float_and_unsupported_shapes() -> Result<()>
    {
        let graph = nullable_typed_projection_fixture_graph()?;
        let absent = compile_nullable_query(
            "MATCH (n)-[r:A]->(m) RETURN n.weight AS node_cross_kind, n.declared_null AS node_declared_null, n.not_declared AS node_unknown, r.score AS relationship_cross_kind, r.declared_null AS relationship_declared_null, r.not_declared AS relationship_unknown",
            &graph,
            64,
        )?
        .ok_or_else(|| Error::internal("absent property projection did not compile"))?;
        absent.request.validate()?;
        let Some(ResidentNullableRelationStage::FinalProject { bindings }) =
            absent.request.program.stages.last()
        else {
            return Err(Error::internal(
                "absent property projection omitted final stage",
            ));
        };
        assert_eq!(bindings.len(), 6);
        assert!(bindings[..3].iter().all(|binding| matches!(
            binding.source,
            ResidentNullableRelationOutputSource::NullProperty {
                kind: ResidentNullableRelationBindingKind::Node,
                ..
            }
        )));
        assert!(bindings[3..].iter().all(|binding| matches!(
            binding.source,
            ResidentNullableRelationOutputSource::NullProperty {
                kind: ResidentNullableRelationBindingKind::Relationship,
                ..
            }
        )));
        assert!(absent.request.property_lanes().is_empty());

        let float = graph
            .catalog()
            .property("float")
            .ok_or_else(|| Error::internal("typed fixture omitted `float`"))?;
        for (query, kind) in [
            (
                "MATCH (n) RETURN n.float",
                ResidentNullableRelationBindingKind::Node,
            ),
            (
                "MATCH (n)-[r]->(m) RETURN r.float",
                ResidentNullableRelationBindingKind::Relationship,
            ),
        ] {
            let compiled = compile_nullable_query(query, &graph, 64)?.ok_or_else(|| {
                Error::internal(format!(
                    "float property projection did not compile: {query}"
                ))
            })?;
            assert!(matches!(
                compiled.outputs.as_slice(),
                [CompiledResidentNullableRelationOutput {
                    source: ResidentNullableRelationOutputSource::FloatProperty {
                        kind: output_kind,
                        property,
                        ..
                    },
                    ..
                }] if *output_kind == kind && *property == float
            ));
            assert_eq!(compiled.request.property_lanes().len(), 1);
        }

        for query in [
            "MATCH (n) RETURN n.node_mixed",
            "MATCH (n)-[r]->(m) RETURN r.relationship_mixed",
        ] {
            assert!(
                compile_nullable_query(query, &graph, 64)?.is_none(),
                "unsupported typed property projection escaped fail-closed admission: {query}"
            );
        }
        Ok(())
    }

    #[test]
    fn nullable_property_conversions_compile_exact_tck_shapes_and_preserve_nulls() -> Result<()> {
        let mut graph = GraphStore::default();
        let person = graph.catalog_mut().intern_label("Person")?;
        let movie = graph.catalog_mut().intern_label("Movie")?;
        let other = graph.catalog_mut().intern_label("Other")?;
        let name = graph.catalog_mut().intern_property("name")?;
        let rating = graph.catalog_mut().intern_property("rating")?;
        graph.insert_node(NodeInput {
            id: NodeId(1),
            layer: Layer::Observed,
            revision: 1,
            labels: vec![person, movie],
            properties: vec![
                (name, ScalarValue::String("42".into())),
                (rating, ScalarValue::Integer(4)),
            ],
        })?;
        graph.insert_node(NodeInput {
            id: NodeId(2),
            layer: Layer::Observed,
            revision: 2,
            labels: vec![other],
            properties: Vec::new(),
        })?;

        let integer = compile_nullable_query(
            "MATCH (p:Person { name:'42' }) WITH * MATCH (n) RETURN toInteger(n.name) AS name",
            &graph,
            64,
        )?
        .ok_or_else(|| Error::internal("TypeConversion2 [7] did not compile natively"))?;
        integer.request.validate()?;
        assert!(matches!(
            integer.outputs.as_slice(),
            [CompiledResidentNullableRelationOutput {
                source: ResidentNullableRelationOutputSource::StringPropertyToInteger {
                    kind: ResidentNullableRelationBindingKind::Node,
                    property,
                    ..
                },
                ..
            }] if *property == name
        ));
        assert_eq!(integer.request.property_lanes().len(), 1);
        let integer = execute_nullable_relation_on_cpu(&graph, &integer)?;
        assert!(matches!(
            integer.columns(),
            [ResidentNullableRelationOutputColumn::StringPropertyToInteger {
                source_rows,
                values,
                validity,
                ..
            }] if source_rows == &[0, 1]
                && values == &[42, 0]
                && validity == &[1, 0]
        ));

        let float = compile_nullable_query(
            "MATCH (m:Movie { rating:4 }) WITH * MATCH (n) RETURN toFloat(n.rating) AS float",
            &graph,
            64,
        )?
        .ok_or_else(|| Error::internal("TypeConversion3 [5] did not compile natively"))?;
        float.request.validate()?;
        assert!(matches!(
            float.outputs.as_slice(),
            [CompiledResidentNullableRelationOutput {
                source: ResidentNullableRelationOutputSource::IntegerPropertyToFloat {
                    kind: ResidentNullableRelationBindingKind::Node,
                    property,
                    ..
                },
                ..
            }] if *property == rating
        ));
        let float = execute_nullable_relation_on_cpu(&graph, &float)?;
        assert!(matches!(
            float.columns(),
            [ResidentNullableRelationOutputColumn::IntegerPropertyToFloat {
                source_rows,
                bits,
                validity,
                ..
            }] if source_rows == &[0, 1]
                && bits == &[4.0_f64.to_bits(), 0]
                && validity == &[1, 0]
        ));

        let string = compile_nullable_query(
            "MATCH (m:Movie { rating:4 }) WITH * MATCH (n) RETURN toString(n.rating) AS rating",
            &graph,
            64,
        )?
        .ok_or_else(|| Error::internal("TypeConversion4 [7] did not compile natively"))?;
        string.request.validate()?;
        assert!(matches!(
            string.outputs.as_slice(),
            [CompiledResidentNullableRelationOutput {
                source: ResidentNullableRelationOutputSource::IntegerPropertyToString {
                    kind: ResidentNullableRelationBindingKind::Node,
                    property,
                    ..
                },
                ..
            }] if *property == rating
        ));
        let string = execute_nullable_relation_on_cpu(&graph, &string)?;
        assert!(matches!(
            string.columns(),
            [ResidentNullableRelationOutputColumn::IntegerPropertyToString {
                source_rows,
                offsets,
                bytes,
                validity,
                ..
            }] if source_rows == &[0, 1]
                && offsets == &[0, 1, 1]
                && bytes == b"4"
                && validity == &[1, 0]
        ));

        for unsupported in [
            "MATCH (n) RETURN toInteger(n.rating)",
            "MATCH (n) RETURN toFloat(n.name)",
            "MATCH (n) RETURN toString(n.name)",
            "MATCH (n) RETURN toInteger(toString(n.rating))",
        ] {
            assert!(
                compile_nullable_query(unsupported, &graph, 64)?.is_none(),
                "unsupported nullable conversion escaped exact admission: {unsupported}"
            );
        }
        Ok(())
    }

    #[test]
    fn return2_property_projections_use_exact_nullable_native_sources_and_cpu_values() -> Result<()>
    {
        let mut missing_node_graph = GraphStore::default();
        let node_num = missing_node_graph.catalog_mut().intern_property("num")?;
        missing_node_graph.insert_node(NodeInput {
            id: NodeId(1),
            layer: Layer::Observed,
            revision: 1,
            labels: Vec::new(),
            properties: vec![(node_num, ScalarValue::Integer(1))],
        })?;
        assert!(missing_node_graph.catalog().property("name").is_none());

        let missing_node =
            compile_nullable_query("MATCH (a) RETURN a.name", &missing_node_graph, 64)?
                .ok_or_else(|| {
                    Error::internal("Return2 [3] did not enter the nullable native route")
                })?;
        missing_node.request.validate()?;
        assert_eq!(missing_node.outputs.len(), 1);
        assert_eq!(missing_node.outputs[0].name, "a.name");
        assert!(matches!(
            missing_node.outputs[0].source,
            ResidentNullableRelationOutputSource::NullProperty {
                kind: ResidentNullableRelationBindingKind::Node,
                ..
            }
        ));
        assert!(missing_node.request.property_lanes().is_empty());
        let missing_node_result =
            execute_nullable_relation_on_cpu(&missing_node_graph, &missing_node)?;
        assert_eq!(missing_node_result.row_count(), 1);
        assert!(matches!(
            missing_node_result.columns(),
            [ResidentNullableRelationOutputColumn::NullProperty {
                kind: ResidentNullableRelationBindingKind::Node,
                source_rows,
                ..
            }] if source_rows == &[0]
        ));

        let mut relationship_graph = GraphStore::default();
        let relationship_type = relationship_graph
            .catalog_mut()
            .intern_relationship_type("T")?;
        let relationship_num = relationship_graph.catalog_mut().intern_property("num")?;
        for id in 1..=2 {
            relationship_graph.insert_node(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: id,
                labels: Vec::new(),
                properties: Vec::new(),
            })?;
        }
        relationship_graph.insert_edge(EdgeInput {
            id: EdgeId(1),
            source: NodeId(1),
            target: NodeId(2),
            relationship_type,
            layer: Layer::Observed,
            revision: 3,
            properties: vec![(relationship_num, ScalarValue::Integer(1))],
        })?;

        let relationship =
            compile_nullable_query("MATCH ()-[r]->() RETURN r.num", &relationship_graph, 64)?
                .ok_or_else(|| {
                    Error::internal("Return2 [4] did not enter the nullable native route")
                })?;
        relationship.request.validate()?;
        assert_eq!(relationship.outputs.len(), 1);
        assert_eq!(relationship.outputs[0].name, "r.num");
        assert!(matches!(
            relationship.outputs[0].source,
            ResidentNullableRelationOutputSource::IntegerProperty {
                kind: ResidentNullableRelationBindingKind::Relationship,
                property,
                ..
            } if property == relationship_num
        ));
        assert_eq!(relationship.request.property_lanes().len(), 1);
        let relationship_result =
            execute_nullable_relation_on_cpu(&relationship_graph, &relationship)?;
        assert_eq!(relationship_result.row_count(), 1);
        assert!(matches!(
            relationship_result.columns(),
            [ResidentNullableRelationOutputColumn::IntegerProperty {
                kind: ResidentNullableRelationBindingKind::Relationship,
                property,
                source_rows,
                values,
                validity,
                ..
            }] if *property == relationship_num
                && source_rows == &[0]
                && values == &[1]
                && validity == &[1]
        ));

        let mut missing_relationship_graph = GraphStore::default();
        let relationship_type = missing_relationship_graph
            .catalog_mut()
            .intern_relationship_type("T")?;
        let relationship_name = missing_relationship_graph
            .catalog_mut()
            .intern_property("name")?;
        for id in 1..=2 {
            missing_relationship_graph.insert_node(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: id,
                labels: Vec::new(),
                properties: Vec::new(),
            })?;
        }
        missing_relationship_graph.insert_edge(EdgeInput {
            id: EdgeId(1),
            source: NodeId(1),
            target: NodeId(2),
            relationship_type,
            layer: Layer::Observed,
            revision: 3,
            properties: vec![(relationship_name, ScalarValue::Integer(1))],
        })?;
        assert!(
            missing_relationship_graph
                .catalog()
                .property("name2")
                .is_none()
        );

        let missing_relationship = compile_nullable_query(
            "MATCH ()-[r]->() RETURN r.name2",
            &missing_relationship_graph,
            64,
        )?
        .ok_or_else(|| Error::internal("Return2 [5] did not enter the nullable native route"))?;
        missing_relationship.request.validate()?;
        assert_eq!(missing_relationship.outputs.len(), 1);
        assert_eq!(missing_relationship.outputs[0].name, "r.name2");
        assert!(matches!(
            missing_relationship.outputs[0].source,
            ResidentNullableRelationOutputSource::NullProperty {
                kind: ResidentNullableRelationBindingKind::Relationship,
                ..
            }
        ));
        assert!(missing_relationship.request.property_lanes().is_empty());
        let missing_relationship_result =
            execute_nullable_relation_on_cpu(&missing_relationship_graph, &missing_relationship)?;
        assert_eq!(missing_relationship_result.row_count(), 1);
        assert!(matches!(
            missing_relationship_result.columns(),
            [ResidentNullableRelationOutputColumn::NullProperty {
                kind: ResidentNullableRelationBindingKind::Relationship,
                source_rows,
                ..
            }] if source_rows == &[0]
        ));
        Ok(())
    }

    #[test]
    fn nullable_compiler_fails_closed_on_unimplemented_shapes_and_legacy_execution_route()
    -> Result<()> {
        let graph = optional_fixture_graph()?;
        let supported = "MATCH (n) OPTIONAL MATCH (n)-[:NOT_EXIST]->(x) RETURN n, x";
        assert!(compile_nullable_query(supported, &graph, 64)?.is_some());
        assert!(
            compile_query(supported, &graph)?.is_none(),
            "legacy row execution route must remain closed until staged backends are wired"
        );
        let fixed_multi_hop = compile_nullable_query(
            "MATCH (a) OPTIONAL MATCH (a)-->(b)-->(c) RETURN c",
            &graph,
            64,
        )?
        .ok_or_else(|| Error::internal("fixed multi-hop OPTIONAL did not compile"))?;
        fixed_multi_hop.request.validate()?;
        assert_eq!(
            fixed_multi_hop
                .request
                .predicate_program
                .optional_groups
                .len(),
            1
        );
        let entity_equality = compile_nullable_query(
            "MATCH (a) OPTIONAL MATCH (a)-->(b) WHERE b = a RETURN b",
            &graph,
            64,
        )?
        .ok_or_else(|| Error::internal("OPTIONAL entity equality did not compile"))?;
        entity_equality.request.validate()?;
        assert_eq!(entity_equality.request.predicate_program.filters.len(), 1);

        for query in [
            "MATCH (a) OPTIONAL MATCH (a)-[*]->(b) RETURN b",
            "OPTIONAL MATCH (a {p: 1}) RETURN a",
            "OPTIONAL MATCH p = (a) RETURN a",
            "MATCH (a) OPTIONAL MATCH (a)-[r {p: [1]}]->(b) RETURN r",
        ] {
            assert!(
                compile_nullable_query(query, &graph, 64)?.is_none(),
                "unsupported nullable shape did not fail closed: {query}"
            );
        }
        Ok(())
    }

    #[test]
    fn graph6_unanchored_optional_relationship_properties_are_one_atomic_native_group() -> Result<()>
    {
        let mut graph = GraphStore::default();
        let relationship_type = graph.catalog_mut().intern_relationship_type("REL")?;
        let existing = graph.catalog_mut().intern_property("existing")?;
        // A property explicitly assigned null is retained in the catalog but has no canonical
        // payload lane, matching the Graph6 [6] fixture.
        let _missing = graph.catalog_mut().intern_property("missing")?;
        for id in 1..=2 {
            graph.insert_node(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: id,
                labels: Vec::new(),
                properties: Vec::new(),
            })?;
        }
        graph.insert_edge(EdgeInput {
            id: EdgeId(1),
            source: NodeId(1),
            target: NodeId(2),
            relationship_type,
            layer: Layer::Observed,
            revision: 3,
            properties: vec![(existing, ScalarValue::Integer(42))],
        })?;

        let compiled = compile_nullable_query(
            "OPTIONAL MATCH ()-[r]->() RETURN r.missing, r.missingToo, r.existing",
            &graph,
            64,
        )?
        .ok_or_else(|| Error::internal("Graph6 [6] did not enter the nullable native route"))?;
        compiled.request.validate()?;
        assert_eq!(compiled.request.predicate_program.optional_groups.len(), 1);
        let group = compiled.request.predicate_program.optional_groups[0];
        assert_eq!((group.first_stage, group.last_stage), (0, 1));
        assert!(matches!(
            compiled.request.program.stages.as_slice(),
            [
                ResidentNullableRelationStage::NodeScan {
                    mode: ResidentNullableRelationMatchMode::Optional,
                    labels: ResidentNullableNodeDomain::Any,
                    ..
                },
                ResidentNullableRelationStage::Expand {
                    mode: ResidentNullableRelationMatchMode::Optional,
                    direction: ResidentDirection::Outgoing,
                    ..
                },
                ResidentNullableRelationStage::FinalProject { .. }
            ]
        ));
        let result = execute_nullable_relation_on_cpu(&graph, &compiled)?;
        assert_eq!(result.row_count(), 1);
        assert!(matches!(
            result.columns(),
            [
                ResidentNullableRelationOutputColumn::NullProperty { source_rows: first, .. },
                ResidentNullableRelationOutputColumn::NullProperty { source_rows: second, .. },
                ResidentNullableRelationOutputColumn::IntegerProperty {
                    property,
                    source_rows,
                    values,
                    validity,
                    ..
                }
            ] if first == &[0]
                && second == &[0]
                && *property == existing
                && source_rows == &[0]
                && values == &[42]
                && validity == &[1]
        ));

        let empty = GraphStore::default();
        let empty_compiled =
            compile_nullable_query("OPTIONAL MATCH ()-[r]->() RETURN r.missing", &empty, 64)?
                .ok_or_else(|| {
                    Error::internal("Graph6 [7] did not enter the nullable native route")
                })?;
        let empty_result = execute_nullable_relation_on_cpu(&empty, &empty_compiled)?;
        assert_eq!(empty_result.row_count(), 1);
        assert!(matches!(
            empty_result.columns(),
            [ResidentNullableRelationOutputColumn::NullProperty { source_rows, .. }]
                if source_rows == &[crate::execution::RESIDENT_NULLABLE_RELATION_NULL_ROW]
        ));

        for query in [
            "OPTIONAL MATCH ()-[r:T]->() RETURN r",
            "OPTIONAL MATCH ()-[r]-() RETURN r",
            "OPTIONAL MATCH (a)-[r]->() RETURN r",
            "OPTIONAL MATCH ()-[r]->(b) RETURN r",
            "OPTIONAL MATCH ()-[r {p: 1}]->() RETURN r",
        ] {
            assert!(
                compile_nullable_query(query, &graph, 64)?.is_none(),
                "nearby unanchored OPTIONAL shape escaped the exact Graph6 adapter: {query}"
            );
        }
        Ok(())
    }

    #[test]
    fn path1_path2_statically_null_optional_paths_publish_only_aligned_nulls() -> Result<()> {
        let graph = optional_fixture_graph()?;
        for query in [
            "WITH null AS a OPTIONAL MATCH p = (a)-[r]->() RETURN nodes(p), nodes(null)",
            "WITH null AS a OPTIONAL MATCH p = (a)-[r]->() RETURN relationships(p), relationships(null)",
        ] {
            let compiled = compile_nullable_query(query, &graph, 64)?.ok_or_else(|| {
                Error::internal(format!("statically-null path did not compile: {query}"))
            })?;
            compiled.request.validate()?;
            assert!(
                compiled
                    .request
                    .predicate_program
                    .optional_groups
                    .is_empty()
            );
            assert!(matches!(
                compiled.request.program.stages.as_slice(),
                [
                    ResidentNullableRelationStage::NodeScan {
                        mode: ResidentNullableRelationMatchMode::Optional,
                        labels: ResidentNullableNodeDomain::KnownEmpty,
                        ..
                    },
                    ResidentNullableRelationStage::Expand {
                        mode: ResidentNullableRelationMatchMode::Optional,
                        direction: ResidentDirection::Outgoing,
                        ..
                    },
                    ResidentNullableRelationStage::FinalProject { .. }
                ]
            ));
            let result = execute_nullable_relation_on_cpu(&graph, &compiled)?;
            assert_eq!(result.row_count(), 1, "{query}");
            assert!(result.columns().iter().all(|column| matches!(
                column,
                ResidentNullableRelationOutputColumn::NullProperty { source_rows, .. }
                    if source_rows == &[crate::execution::RESIDENT_NULLABLE_RELATION_NULL_ROW]
            )));
        }

        for query in [
            "MATCH (a) OPTIONAL MATCH p = (a)-[r]->() RETURN nodes(p)",
            "WITH null AS a OPTIONAL MATCH p = (a)<-[r]-() RETURN nodes(p)",
            "WITH null AS a OPTIONAL MATCH p = (a)-[r:T]->() RETURN nodes(p)",
            "WITH null AS a OPTIONAL MATCH p = (a)-[r]->(b) RETURN nodes(p)",
            "WITH null AS a OPTIONAL MATCH p = (a)-[r*]->() RETURN nodes(p)",
            "WITH null AS a OPTIONAL MATCH p = (a)-[r]->() RETURN length(p)",
            "WITH null AS a OPTIONAL MATCH p = (a)-[r]->() RETURN p",
            "WITH null AS a OPTIONAL MATCH p = (a)-[r]->() RETURN nodes(p)",
            "WITH null AS a OPTIONAL MATCH p = (a)-[r]->() RETURN nodes(p) AS path_nodes, nodes(null)",
            "WITH null AS a OPTIONAL MATCH p = (a)-[r]->() RETURN nodes(p), a",
        ] {
            assert!(
                compile_nullable_query(query, &graph, 64)?.is_none(),
                "nearby named OPTIONAL path escaped static-null admission: {query}"
            );
        }
        Ok(())
    }

    #[test]
    fn match3_relationship_string_property_map_is_a_post_expand_predicate() -> Result<()> {
        let graph = relationship_property_fixture_graph()?;
        let property = graph
            .catalog()
            .property("name")
            .ok_or_else(|| Error::internal("relationship fixture omitted `name`"))?;
        let compiled =
            compile_nullable_query("MATCH (a)-[r {name: 'r'}]-(b) RETURN a, b", &graph, 64)?
                .ok_or_else(|| Error::internal("Match3 [5] relationship map did not compile"))?;
        compiled.request.validate()?;

        let (stage, relationship) = compiled
            .request
            .program
            .stages
            .iter()
            .enumerate()
            .find_map(|(stage, candidate)| match candidate {
                ResidentNullableRelationStage::Expand {
                    relationship: Some(relationship),
                    direction: ResidentDirection::Undirected,
                    ..
                } => Some((stage, *relationship)),
                _ => None,
            })
            .ok_or_else(|| Error::internal("Match3 [5] omitted its undirected relationship"))?;
        assert_eq!(
            compiled.request.predicate_program.filters,
            vec![ResidentNullableRelationFilterStage {
                placement: ResidentNullableRelationFilterPlacement::RelationAfter {
                    stage: u16::try_from(stage).expect("bounded resident stage"),
                },
                predicate: ResidentNullableRelationPredicate::CompareString {
                    left: ResidentNullableRelationPredicateValue::StringProperty {
                        slot: relationship,
                        kind: ResidentNullableRelationBindingKind::Relationship,
                        property,
                    },
                    operation: CompareOp::Eq,
                    right: ResidentNullableRelationPredicateValue::String("r".into()),
                },
            }]
        );

        let optional = compile_nullable_query(
            "MATCH (a:A) OPTIONAL MATCH (a)-[r:REL {name: 'r'}]->(b) RETURN a, r",
            &graph,
            64,
        )?
        .ok_or_else(|| Error::internal("optional relationship map did not compile"))?;
        let optional_stage = optional
            .request
            .program
            .stages
            .iter()
            .position(|stage| {
                matches!(
                    stage,
                    ResidentNullableRelationStage::Expand {
                        mode: ResidentNullableRelationMatchMode::Optional,
                        ..
                    }
                )
            })
            .ok_or_else(|| Error::internal("optional relationship map omitted its expansion"))?;
        assert!(matches!(
            optional.request.predicate_program.filters.as_slice(),
            [ResidentNullableRelationFilterStage {
                placement: ResidentNullableRelationFilterPlacement::OptionalCandidates { stage },
                predicate: ResidentNullableRelationPredicate::CompareString { .. },
            }] if usize::from(*stage) == optional_stage
        ));
        optional.request.validate()?;
        Ok(())
    }

    #[test]
    fn reversed_relationship_integer_property_map_keeps_the_relationship_slot() -> Result<()> {
        let graph = relationship_property_fixture_graph()?;
        let property = graph
            .catalog()
            .property("weight")
            .ok_or_else(|| Error::internal("relationship fixture omitted `weight`"))?;
        let compiled = compile_nullable_query(
            "MATCH (b:B) MATCH (a)-[r:REL {weight: 7}]->(b) RETURN r",
            &graph,
            64,
        )?
        .ok_or_else(|| Error::internal("reversed relationship property map did not compile"))?;
        compiled.request.validate()?;

        let (stage, relationship) = compiled
            .request
            .program
            .stages
            .iter()
            .enumerate()
            .find_map(|(stage, candidate)| match candidate {
                ResidentNullableRelationStage::Expand {
                    relationship: Some(relationship),
                    direction: ResidentDirection::Incoming,
                    ..
                } => Some((stage, *relationship)),
                _ => None,
            })
            .ok_or_else(|| Error::internal("bound target did not reverse the relationship scan"))?;
        assert_eq!(
            compiled.request.predicate_program.filters,
            vec![ResidentNullableRelationFilterStage {
                placement: ResidentNullableRelationFilterPlacement::RelationAfter {
                    stage: u16::try_from(stage).expect("bounded resident stage"),
                },
                predicate: ResidentNullableRelationPredicate::CompareInteger {
                    left: ResidentNullableRelationPredicateValue::IntegerProperty {
                        slot: relationship,
                        kind: ResidentNullableRelationBindingKind::Relationship,
                        property,
                    },
                    operation: CompareOp::Eq,
                    right: ResidentNullableRelationPredicateValue::Integer(7),
                },
            }]
        );
        Ok(())
    }

    #[test]
    fn anonymous_relationship_maps_retain_null_and_type_semantics() -> Result<()> {
        let graph = relationship_property_fixture_graph()?;
        let weight = graph
            .catalog()
            .property("weight")
            .ok_or_else(|| Error::internal("relationship fixture omitted `weight`"))?;

        let anonymous =
            compile_nullable_query("MATCH (a)-[:REL {name: 'r'}]->(b) RETURN a, b", &graph, 64)?
                .ok_or_else(|| {
                    Error::internal("anonymous relationship property map did not compile")
                })?;
        assert!(matches!(
            anonymous.request.program.stages.as_slice(),
            [
                ResidentNullableRelationStage::NodeScan { .. },
                ResidentNullableRelationStage::Expand {
                    relationship: Some(_),
                    ..
                },
                ResidentNullableRelationStage::FinalProject { .. }
            ]
        ));
        anonymous.request.validate()?;

        let mismatched =
            compile_nullable_query("MATCH (a)-[r:REL {weight: '7'}]->(b) RETURN r", &graph, 64)?
                .ok_or_else(|| {
                    Error::internal("cross-type relationship equality did not compile")
                })?;
        let [
            ResidentNullableRelationFilterStage {
                predicate:
                    ResidentNullableRelationPredicate::CompareInteger {
                        left,
                        operation: CompareOp::NotEq,
                        right,
                    },
                ..
            },
        ] = mismatched.request.predicate_program.filters.as_slice()
        else {
            return Err(Error::internal(
                "cross-type relationship equality lost its false-or-null predicate",
            ));
        };
        assert_eq!(left, right);
        assert!(matches!(
            left,
            ResidentNullableRelationPredicateValue::IntegerProperty {
                kind: ResidentNullableRelationBindingKind::Relationship,
                property,
                ..
            } if *property == weight
        ));
        mismatched.request.validate()?;

        let missing =
            compile_nullable_query("MATCH (a)-[r:REL {missing: 7}]->(b) RETURN r", &graph, 64)?
                .ok_or_else(|| {
                    Error::internal("missing relationship property did not compile as NULL")
                })?;
        assert!(matches!(
            missing.request.predicate_program.filters.as_slice(),
            [ResidentNullableRelationFilterStage {
                predicate: ResidentNullableRelationPredicate::CompareInteger {
                    left: ResidentNullableRelationPredicateValue::Null,
                    operation: CompareOp::Eq,
                    right: ResidentNullableRelationPredicateValue::Integer(7),
                },
                ..
            }]
        ));
        missing.request.validate()?;

        assert!(
            compile_nullable_query("MATCH (a)-[r:REL {weight: 7.5}]->(b) RETURN r", &graph, 64,)?
                .is_none(),
            "unsupported floating relationship map escaped fail-closed admission"
        );

        let mut mixed = relationship_property_fixture_graph()?;
        let relationship = mixed
            .catalog()
            .relationship_type("REL")
            .ok_or_else(|| Error::internal("relationship fixture omitted `REL`"))?;
        mixed.insert_node(NodeInput {
            id: NodeId(3),
            layer: Layer::Observed,
            revision: 4,
            labels: Vec::new(),
            properties: Vec::new(),
        })?;
        mixed.insert_edge(EdgeInput {
            id: EdgeId(2),
            source: NodeId(1),
            target: NodeId(3),
            relationship_type: relationship,
            layer: Layer::Observed,
            revision: 5,
            properties: vec![(weight, ScalarValue::String("seven".into()))],
        })?;
        assert!(mixed.snapshot()?.edge_properties.is_mixed(weight));
        assert!(
            compile_nullable_query("MATCH (a)-[r:REL {weight: 7}]->(b) RETURN r", &mixed, 64,)?
                .is_none(),
            "mixed relationship property escaped fail-closed admission"
        );
        Ok(())
    }

    #[test]
    fn nullable_compiler_declines_is_null_on_a_mixed_property_without_aborting_other_routes()
    -> Result<()> {
        let mut graph = GraphStore::default();
        let root = graph.catalog_mut().intern_label("Root")?;
        let child = graph.catalog_mut().intern_label("Child")?;
        let relationship = graph.catalog_mut().intern_relationship_type("R")?;
        let property = graph.catalog_mut().intern_property("var")?;
        graph.insert_node(NodeInput {
            id: NodeId(1),
            layer: Layer::Observed,
            revision: 1,
            labels: vec![root],
            properties: Vec::new(),
        })?;
        for (id, value) in [
            (2, ScalarValue::Integer(7)),
            (3, ScalarValue::String("z".into())),
        ] {
            graph.insert_node(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: id,
                labels: vec![child],
                properties: vec![(property, value)],
            })?;
        }
        for id in 2..=3 {
            graph.insert_edge(EdgeInput {
                id: EdgeId(id - 1),
                source: NodeId(1),
                target: NodeId(id),
                relationship_type: relationship,
                layer: Layer::Observed,
                revision: id + 2,
                properties: Vec::new(),
            })?;
        }
        assert!(graph.snapshot()?.node_properties.is_mixed(property));

        for query in [
            "MATCH (:Root)-->(i:Child) WHERE i.var IS NOT NULL AND i.var > 'x' RETURN i.var",
            "MATCH (:Root)-->(i:Child) WHERE i.var IS NULL OR i.var > 'x' RETURN i.var",
        ] {
            assert!(
                compile_nullable_query(query, &graph, 64)?.is_none(),
                "mixed-property predicate was incorrectly captured by the nullable route: {query}"
            );
        }
        Ok(())
    }

    #[test]
    fn quantifier_compiler_deforests_immediate_collect_unwind_of_range_rows() -> Result<()> {
        let graph = GraphStore::default();
        let parameters = BTreeMap::new();
        let physical = physical(
            "UNWIND range(1, 2) AS row WITH collect(row) AS rows UNWIND rows AS x RETURN x",
            &graph,
        )?;
        let statistics = StatisticsSnapshot::collect(&graph);
        let (physical, _) = optimize(
            physical,
            OptimizerInput {
                statistics: &statistics,
                catalog: graph.catalog(),
                indexes: None,
                parameters: &parameters,
                backend: BackendKind::Metal,
                scratch_budget_bytes: 64 * 1024 * 1024,
                max_result_rows: 64,
                runtime_feedback: None,
                allow_runtime_checkpoint: true,
            },
        );
        let compiled = compile_quantifier(
            &physical,
            PROJECT,
            Bookmark {
                term: 3,
                index: graph.revision(),
            },
            graph.catalog(),
            &graph,
            &parameters,
            64,
        )?;
        let compiled = compiled.unwrap_or_else(|| {
            panic!(
                "collect/UNWIND deforestation did not own the optimized plan: {:#?}",
                physical.operators
            )
        });
        assert!(matches!(
            compiled.request.source,
            ResidentQuantifierSource::Range { .. }
        ));
        assert_eq!(compiled.request.program.stages.len(), 2);
        assert_eq!(compiled.request.program.outputs[0].name, "x");
        Ok(())
    }

    #[test]
    fn nullable_relation_deforests_non_null_entity_collect_unwind_and_executes_on_cpu() -> Result<()>
    {
        let mut graph = GraphStore::default();
        let id = graph.catalog_mut().intern_property("id")?;
        for value in 1..=2 {
            graph.insert_node(NodeInput {
                id: NodeId(value),
                layer: Layer::Observed,
                revision: value,
                labels: Vec::new(),
                properties: vec![(id, ScalarValue::Integer(value as i64))],
            })?;
        }

        let query = "MATCH (row) \
                     WITH collect(row) AS rows \
                     UNWIND rows AS node \
                     RETURN node.id";
        let compiled = compile_nullable_query(query, &graph, 64)?.ok_or_else(|| {
            Error::internal("non-null entity collect/UNWIND did not remain a nullable relation")
        })?;
        compiled.request.validate()?;
        assert!(matches!(
            compiled.request.program.stages.as_slice(),
            [
                ResidentNullableRelationStage::NodeScan { .. },
                ResidentNullableRelationStage::ScopeProject { bindings },
                ResidentNullableRelationStage::FinalProject { .. },
            ] if bindings.len() == 1 && bindings[0].variable == "node"
        ));

        let result = execute_nullable_relation_on_cpu(&graph, &compiled)?;
        assert_eq!(result.row_count(), 2);
        let [
            ResidentNullableRelationOutputColumn::IntegerProperty {
                values, validity, ..
            },
        ] = result.columns()
        else {
            return Err(Error::internal(
                "entity collect/UNWIND returned the wrong native column",
            ));
        };
        assert_eq!(values, &[1, 2]);
        assert_eq!(validity, &[1, 1]);

        for unsupported in [
            "OPTIONAL MATCH (row) WITH collect(row) AS rows UNWIND rows AS node RETURN node",
            "MATCH (row) WITH collect(DISTINCT row) AS rows UNWIND rows AS node RETURN node",
            "MATCH (row) WITH collect(row) AS rows UNWIND rows AS node RETURN rows, node",
            "MATCH (row) WITH collect(row) AS rows UNWIND rows AS node RETURN *",
        ] {
            assert!(
                compile_nullable_query(unsupported, &graph, 64)?.is_none(),
                "unsafe entity collect/UNWIND rewrite escaped fail-closed admission: {unsupported}"
            );
        }
        Ok(())
    }

    #[test]
    fn nullable_relation_deforestation_preserves_group_keys_for_correlated_match() -> Result<()> {
        let mut graph = GraphStore::default();
        let s = graph.catalog_mut().intern_label("S")?;
        let e = graph.catalog_mut().intern_label("E")?;
        let x = graph.catalog_mut().intern_relationship_type("X")?;
        let y = graph.catalog_mut().intern_relationship_type("Y")?;
        for (id, labels) in [
            (NodeId(1), vec![s]),
            (NodeId(2), Vec::new()),
            (NodeId(3), vec![e]),
        ] {
            graph.insert_node(NodeInput {
                id,
                layer: Layer::Observed,
                revision: id.0,
                labels,
                properties: Vec::new(),
            })?;
        }
        for (id, source, relationship_type) in [(1, 1, x), (2, 1, y), (3, 2, y)] {
            graph.insert_edge(EdgeInput {
                id: EdgeId(id),
                source: NodeId(source),
                target: NodeId(3),
                relationship_type,
                layer: Layer::Observed,
                revision: 3 + id,
                properties: Vec::new(),
            })?;
        }

        let query = "MATCH (a:S)-[:X]->(b1) \
                     WITH a, collect(b1) AS bees \
                     UNWIND bees AS b2 \
                     MATCH (a)-[:Y]->(b2) \
                     RETURN a, b2";
        let compiled = compile_nullable_query(query, &graph, 64)?.ok_or_else(|| {
            Error::internal("grouped entity collect/UNWIND lost its correlated native MATCH")
        })?;
        compiled.request.validate()?;
        assert!(matches!(
            compiled.request.program.stages.as_slice(),
            [
                ResidentNullableRelationStage::NodeScan { .. },
                ResidentNullableRelationStage::Expand { .. },
                ResidentNullableRelationStage::ScopeProject { bindings },
                ResidentNullableRelationStage::Expand {
                    target: ResidentNullableRelationTarget::Existing(_),
                    ..
                },
                ResidentNullableRelationStage::FinalProject { .. },
            ] if bindings.len() == 2
                && bindings.iter().any(|binding| binding.variable == "a")
                && bindings.iter().any(|binding| binding.variable == "b2")
        ));

        let result = execute_nullable_relation_on_cpu(&graph, &compiled)?;
        assert_eq!(result.row_count(), 1);
        assert!(result.columns().iter().all(|column| matches!(
            column,
            ResidentNullableRelationOutputColumn::Entity { rows, .. }
                if rows != &[crate::execution::RESIDENT_NULLABLE_RELATION_NULL_ROW]
        )));
        Ok(())
    }

    #[test]
    fn entity_quantifier_source_projection_recognizes_only_the_exact_tail_path_shapes() {
        fn function(name: &str, arguments: Vec<Expression>) -> Expression {
            Expression::Function {
                name: vec![name.to_owned()],
                distinct: false,
                arguments,
            }
        }

        let list = |name: &str| ProjectionItem {
            expression: function(
                "tail",
                vec![function(name, vec![Expression::Variable("p".to_owned())])],
            ),
            alias: Some(name.to_owned()),
            source_text: None,
        };
        let count = ProjectionItem {
            expression: Expression::Function {
                name: vec!["count".to_owned()],
                distinct: false,
                arguments: vec![Expression::Star],
            },
            alias: Some("c".to_owned()),
            source_text: None,
        };

        assert_eq!(
            quantifier_entity_source_projection(
                &Projection {
                    distinct: false,
                    items: vec![list("nodes")],
                },
                "p",
            ),
            Some((EntityKind::Node, "nodes".to_owned(), None))
        );
        assert_eq!(
            quantifier_entity_source_projection(
                &Projection {
                    distinct: false,
                    items: vec![list("relationships"), count.clone()],
                },
                "p",
            ),
            Some((
                EntityKind::Relationship,
                "relationships".to_owned(),
                Some("c".to_owned()),
            ))
        );

        for projection in [
            Projection {
                distinct: false,
                items: vec![list("nodes"), count.clone()],
            },
            Projection {
                distinct: false,
                items: vec![list("relationships")],
            },
            Projection {
                distinct: true,
                items: vec![list("nodes")],
            },
        ] {
            assert!(quantifier_entity_source_projection(&projection, "p").is_none());
        }
    }

    fn graph_with_temporal_and_document_properties() -> Result<GraphStore> {
        let mut graph = GraphStore::default();
        let label_a = graph.catalog_mut().intern_label("A")?;
        let label_b = graph.catalog_mut().intern_label("B")?;
        let created = graph.catalog_mut().intern_property("created")?;
        let payload = graph.catalog_mut().intern_property("payload")?;
        graph.insert_node(NodeInput {
            id: NodeId(1),
            layer: Layer::Observed,
            revision: 1,
            labels: vec![label_a, label_b],
            properties: vec![
                (created, ScalarValue::Date(42)),
                (
                    payload,
                    ScalarValue::List(DocumentList::new(vec![DocumentItem::Scalar(
                        ScalarValue::Integer(7),
                    )])?),
                ),
            ],
        })?;
        graph.insert_node(NodeInput {
            id: NodeId(2),
            layer: Layer::Observed,
            revision: 2,
            labels: vec![label_a],
            properties: Vec::new(),
        })?;
        Ok(graph)
    }

    fn execution_context<'a>(
        graph: &'a GraphStore,
        backend: &'a dyn ExecutionBackend,
    ) -> ExecutionContext<'a> {
        ExecutionContext {
            project_id: PROJECT,
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
                term: 3,
                index: graph.revision(),
            },
            mutation_revision: graph.revision().saturating_add(1),
            resolved_time_nanos: 0,
            next_node_id: 100,
            next_edge_id: 100,
            predicate_versions: BTreeMap::new(),
            capabilities: BindCapabilities::default(),
            max_result_rows: 64,
            max_batch_rows: 2,
            optimizer_statistics: None,
            backend: Some(backend),
            cancellation: CancellationToken::new(),
            deadline: Some(Instant::now() + Duration::from_secs(5)),
            resolved_query_at_time_nanos: None,
        }
    }

    fn cpu_for(graph: &GraphStore) -> Result<CpuBackend> {
        let bookmark = Bookmark {
            term: 3,
            index: graph.revision(),
        };
        let image = ResidentProjectImage::build(
            PROJECT,
            bookmark,
            graph,
            &TemporalStore::default(),
            &IndexCatalog::default(),
        )?;
        let mut cpu = CpuBackend::new(64 * 1024 * 1024, 8 * 1024 * 1024);
        cpu.admit_project(image)?;
        Ok(cpu)
    }

    fn compile_delete_query(
        source: &str,
        graph: &GraphStore,
    ) -> Result<Option<CompiledResidentDeletePlan>> {
        let bookmark = Bookmark {
            term: 3,
            index: graph.revision(),
        };
        compile_delete(
            &writable_physical(source, graph)?,
            PROJECT,
            bookmark,
            graph.catalog(),
            graph,
            &BTreeMap::new(),
            ResidentExecutionId { high: 1, low: 2 },
            64,
        )
    }

    #[test]
    fn delete1_null_node_commands_keep_the_initial_optional_boundary_native() -> Result<()> {
        for (query, detach) in [
            ("OPTIONAL MATCH (n) DELETE n", false),
            ("OPTIONAL MATCH (n) DETACH DELETE n", true),
        ] {
            let graph = GraphStore::default();
            let compiled = compile_delete_query(query, &graph)?.ok_or_else(|| {
                Error::internal(format!("null-node DELETE did not compile: {query}"))
            })?;
            compiled.request.validate()?;
            assert!(compiled.request.selection.initial_optional);
            assert!(compiled.request.selection.expansion.is_none());
            assert!(compiled.request.selection.continuations.is_empty());
            assert!(compiled.request.selection_filters.is_empty());
            assert!(compiled.request.continuation.is_none());
            assert!(matches!(
                compiled.request.commands.as_slice(),
                [ResidentDeleteCommand {
                    target: ResidentEntityBinding::Node(ResidentNodeBinding::Start),
                    selector: ResidentDeleteTargetSelector::EachSelectedRow,
                    detach: command_detach,
                    ..
                }] if *command_detach == detach
            ));

            let cpu = cpu_for(&graph)?;
            let result =
                cpu.execute_delete_pipeline(&compiled.request, &CancellationToken::new())?;
            let validated = result.validate_for_publication(&compiled.request, BackendKind::Cpu)?;
            assert!(validated.intents().is_empty());
            assert!(validated.read_dependencies().is_empty());
            assert!(validated.final_relation().is_none());
        }

        let graph = GraphStore::default();
        for unsupported in [
            "OPTIONAL MATCH (n {p: 1}) DELETE n",
            "OPTIONAL MATCH (n)-[r]->() DELETE r",
        ] {
            assert!(
                compile_delete_query(unsupported, &graph)?.is_none(),
                "wider nullable DELETE escaped exact admission: {unsupported}"
            );
        }
        Ok(())
    }

    #[test]
    fn nullable_delete_node_relationship_and_path_share_one_sealed_command() -> Result<()> {
        struct Expected {
            query: &'static str,
            kind: NullableDeleteTargetKind,
            bindings: &'static [ResidentEntityBinding],
            detach: bool,
            output: Option<(&'static str, NullableDeleteTargetKind)>,
        }

        let cases = [
            Expected {
                query: "OPTIONAL MATCH (a:DoesNotExist) DELETE a RETURN a",
                kind: NullableDeleteTargetKind::Entity(EntityKind::Node),
                bindings: &[ResidentEntityBinding::Node(ResidentNodeBinding::Start)],
                detach: false,
                output: Some(("a", NullableDeleteTargetKind::Entity(EntityKind::Node))),
            },
            Expected {
                query: "OPTIONAL MATCH ()-[r:DoesNotExist]-() DELETE r RETURN r",
                kind: NullableDeleteTargetKind::Entity(EntityKind::Relationship),
                bindings: &[ResidentEntityBinding::Relationship(0)],
                detach: false,
                output: Some((
                    "r",
                    NullableDeleteTargetKind::Entity(EntityKind::Relationship),
                )),
            },
            Expected {
                query: "OPTIONAL MATCH p = ()-->() DETACH DELETE p",
                kind: NullableDeleteTargetKind::Path,
                bindings: &[
                    ResidentEntityBinding::Node(ResidentNodeBinding::Start),
                    ResidentEntityBinding::Relationship(0),
                    ResidentEntityBinding::Node(ResidentNodeBinding::End),
                ],
                detach: true,
                output: None,
            },
        ];

        let graph = GraphStore::default();
        for expected in cases {
            let physical = writable_physical(expected.query, &graph)?;
            let shape = nullable_delete_shape(&physical).ok_or_else(|| {
                Error::internal(format!(
                    "nullable DELETE did not normalize: {}",
                    expected.query
                ))
            })?;
            assert_eq!(shape.target_kind, expected.kind, "{}", expected.query);
            assert_eq!(
                shape.target_bindings, expected.bindings,
                "{}",
                expected.query
            );
            assert_eq!(shape.detach, expected.detach, "{}", expected.query);
            assert_eq!(
                shape
                    .output
                    .as_ref()
                    .map(|output| (output.name.as_str(), output.kind)),
                expected.output,
                "{}",
                expected.query
            );
            assert!(
                !nullable_delete_shape_fits_v1(&shape),
                "nullable DELETE escaped the v1 command fence: {}",
                expected.query
            );
            let compiled = compile_delete_query(expected.query, &graph)?.ok_or_else(|| {
                Error::internal(format!(
                    "nullable DELETE did not compile as one sealed command: {}",
                    expected.query
                ))
            })?;
            compiled.request.validate()?;
            assert!(
                compiled.request.selection.initial_optional,
                "{}",
                expected.query
            );
            assert!(
                compiled.request.selection_is_statically_empty,
                "{}",
                expected.query
            );
            assert_eq!(
                compiled
                    .request
                    .commands
                    .iter()
                    .map(|command| command.target)
                    .collect::<Vec<_>>(),
                expected.bindings,
                "{}",
                expected.query
            );
            assert!(
                compiled
                    .request
                    .commands
                    .iter()
                    .all(|command| command.detach == expected.detach
                        && command.selector == ResidentDeleteTargetSelector::EachSelectedRow),
                "{}",
                expected.query
            );
            match (expected.output, &compiled.request.continuation) {
                (Some((name, NullableDeleteTargetKind::Entity(kind))), Some(continuation)) => {
                    assert_eq!(continuation.output.name, name, "{}", expected.query);
                    assert_eq!(
                        continuation.output.source,
                        ResidentDeleteOutputSource::NullEntity(kind),
                        "{}",
                        expected.query
                    );
                    assert!(matches!(
                        continuation.stages.as_slice(),
                        [ResidentDeletePostStage::Project {
                            source: ResidentDeleteOutputSource::NullEntity(stage_kind),
                        }] if *stage_kind == kind
                    ));
                }
                (None, None) => {}
                _ => {
                    return Err(Error::internal(format!(
                        "nullable DELETE continuation differs from its exact shape: {}",
                        expected.query
                    )));
                }
            }
            let cpu = cpu_for(&graph)?;
            let validated = cpu
                .execute_delete_pipeline(&compiled.request, &CancellationToken::new())?
                .validate_for_publication(&compiled.request, BackendKind::Cpu)?;
            assert!(validated.intents().is_empty(), "{}", expected.query);
            assert!(
                validated.read_dependencies().is_empty(),
                "{}",
                expected.query
            );
            match expected.output {
                Some(_) => {
                    let relation = validated.final_relation().ok_or_else(|| {
                        Error::internal("nullable DELETE omitted its typed null relation")
                    })?;
                    assert_eq!(relation.row_count, 1, "{}", expected.query);
                    assert_eq!(relation.validity, [0], "{}", expected.query);
                }
                None => assert!(validated.final_relation().is_none(), "{}", expected.query),
            }
        }
        Ok(())
    }

    #[test]
    fn nullable_delete_normal_form_rejects_wider_or_untyped_tails() -> Result<()> {
        let graph = GraphStore::default();
        for query in [
            "OPTIONAL MATCH (a) DELETE a RETURN 1 AS x",
            "OPTIONAL MATCH (a), (b) DELETE a RETURN a",
            "OPTIONAL MATCH p = ()-[*]->() DETACH DELETE p",
            "OPTIONAL MATCH ()-[r]->() DELETE r, r",
            "OPTIONAL MATCH ()-[r]->() DELETE r RETURN r, 1 AS x",
        ] {
            assert!(
                nullable_delete_shape(&writable_physical(query, &graph)?).is_none(),
                "wider nullable DELETE entered the exact normal form: {query}"
            );
        }
        Ok(())
    }

    fn fixed_path_delete_graph() -> Result<GraphStore> {
        let mut graph = GraphStore::default();
        let x = graph.catalog_mut().intern_label("X")?;
        let r = graph.catalog_mut().intern_relationship_type("R")?;
        for (id, labels) in [
            (1, vec![x]),
            (2, Vec::new()),
            (3, Vec::new()),
            (4, Vec::new()),
        ] {
            graph.insert_node(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: id,
                labels,
                properties: Vec::new(),
            })?;
        }
        for (id, source, target) in [(1, 1, 2), (2, 2, 3), (3, 3, 4)] {
            graph.insert_edge(EdgeInput {
                id: EdgeId(id),
                source: NodeId(source),
                target: NodeId(target),
                relationship_type: r,
                layer: Layer::Observed,
                revision: 10 + id,
                properties: Vec::new(),
            })?;
        }
        Ok(graph)
    }

    #[test]
    fn nullable_delete_static_empty_proof_is_generation_fenced() -> Result<()> {
        let graph = fixed_path_delete_graph()?;

        let missing_node =
            compile_delete_query("OPTIONAL MATCH (a:DoesNotExist) DELETE a RETURN a", &graph)?
                .ok_or_else(|| Error::internal("missing-label nullable DELETE did not compile"))?;
        assert!(missing_node.request.selection_is_statically_empty);
        assert_eq!(
            missing_node.request.selection.labels,
            [LabelId(graph.catalog().next_label_id())]
        );

        let missing_relationship = compile_delete_query(
            "OPTIONAL MATCH ()-[r:DoesNotExist]-() DELETE r RETURN r",
            &graph,
        )?
        .ok_or_else(|| Error::internal("missing-type nullable DELETE did not compile"))?;
        assert!(missing_relationship.request.selection_is_statically_empty);
        assert_eq!(
            missing_relationship
                .request
                .selection
                .expansion
                .as_ref()
                .ok_or_else(|| Error::internal("nullable relationship expansion disappeared"))?
                .relationship_types,
            [RelationshipTypeId(
                graph.catalog().next_relationship_type_id()
            )]
        );

        assert!(
            compile_delete_query("OPTIONAL MATCH (a:X) DELETE a RETURN a", &graph)?.is_none(),
            "a potentially non-null entity escaped the typed-null output contract"
        );

        let path = compile_delete_query("OPTIONAL MATCH p = ()-->() DETACH DELETE p", &graph)?
            .ok_or_else(|| Error::internal("nonempty nullable path did not compile"))?;
        assert!(!path.request.selection_is_statically_empty);
        let mut forged = path.request;
        forged.selection_is_statically_empty = true;
        forged.seal()?;
        forged.validate()?;
        let cpu = cpu_for(&graph)?;
        let error = cpu
            .execute_delete_pipeline(&forged, &CancellationToken::new())
            .expect_err("forged static-empty proof selected real path rows");
        assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
        Ok(())
    }

    #[test]
    fn bound_optional_relationship_delete_is_one_complete_resident_command() -> Result<()> {
        let mut graph = GraphStore::default();
        graph.insert_node(NodeInput {
            id: NodeId(1),
            layer: Layer::Observed,
            revision: 1,
            labels: Vec::new(),
            properties: Vec::new(),
        })?;
        let compiled =
            compile_delete_query("MATCH (n) OPTIONAL MATCH (n)-[r]-() DELETE n, r", &graph)?
                .ok_or_else(|| Error::internal("bound OPTIONAL DELETE did not compile"))?;
        compiled.request.validate()?;
        assert!(!compiled.request.selection.initial_optional);
        assert!(
            compiled
                .request
                .selection
                .expansion
                .as_ref()
                .is_some_and(|expansion| expansion.optional
                    && expansion.direction == crate::execution::ResidentDirection::Undirected)
        );
        assert!(matches!(
            compiled.request.commands.as_slice(),
            [
                ResidentDeleteCommand {
                    target: ResidentEntityBinding::Node(ResidentNodeBinding::Start),
                    detach: false,
                    selector: ResidentDeleteTargetSelector::EachSelectedRow,
                    ..
                },
                ResidentDeleteCommand {
                    target: ResidentEntityBinding::Relationship(0),
                    detach: false,
                    selector: ResidentDeleteTargetSelector::EachSelectedRow,
                    ..
                }
            ]
        ));
        let validated = cpu_for(&graph)?
            .execute_delete_pipeline(&compiled.request, &CancellationToken::new())?
            .validate_for_publication(&compiled.request, BackendKind::Cpu)?;
        assert_eq!(validated.intents().len(), 1);
        assert_eq!(validated.intents()[0].target_kind, EntityKind::Node);
        assert!(validated.final_relation().is_none());
        Ok(())
    }

    #[test]
    fn deleted_relationship_type_is_captured_from_the_prewrite_generation() -> Result<()> {
        let mut graph = GraphStore::default();
        let relationship_type = graph.catalog_mut().intern_relationship_type("T")?;
        for id in 1..=2 {
            graph.insert_node(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: id,
                labels: Vec::new(),
                properties: Vec::new(),
            })?;
        }
        graph.insert_edge(EdgeInput {
            id: EdgeId(1),
            source: NodeId(1),
            target: NodeId(2),
            relationship_type,
            layer: Layer::Observed,
            revision: 3,
            properties: Vec::new(),
        })?;
        let compiled = compile_delete_query("MATCH ()-[r]->() DELETE r RETURN type(r)", &graph)?
            .ok_or_else(|| Error::internal("deleted relationship type did not compile"))?;
        compiled.request.validate()?;
        let relationship = ResidentEntityBinding::Relationship(0);
        let continuation = compiled
            .request
            .continuation
            .as_ref()
            .ok_or_else(|| Error::internal("relationship-type continuation disappeared"))?;
        assert_eq!(
            continuation.output.source,
            ResidentDeleteOutputSource::RelationshipType(relationship)
        );
        assert!(matches!(
            continuation.stages.as_slice(),
            [ResidentDeletePostStage::Project {
                source: ResidentDeleteOutputSource::RelationshipType(binding),
            }] if *binding == relationship
        ));
        let validated = cpu_for(&graph)?
            .execute_delete_pipeline(&compiled.request, &CancellationToken::new())?
            .validate_for_publication(&compiled.request, BackendKind::Cpu)?;
        let relation = validated
            .final_relation()
            .ok_or_else(|| Error::internal("relationship-type relation disappeared"))?;
        assert_eq!(relation.row_count, 1);
        assert_eq!(relation.values, [relationship_type.0 as i64]);
        assert_eq!(relation.validity, [1]);
        Ok(())
    }

    fn collected_path_delete_graph() -> Result<GraphStore> {
        let mut graph = GraphStore::default();
        let user = graph.catalog_mut().intern_label("User")?;
        let r = graph.catalog_mut().intern_relationship_type("R")?;
        for id in 1..=2 {
            graph.insert_node(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: id,
                labels: vec![user],
                properties: Vec::new(),
            })?;
        }
        for (id, source, target) in [(1, 1, 2), (2, 2, 1)] {
            graph.insert_edge(EdgeInput {
                id: EdgeId(id),
                source: NodeId(source),
                target: NodeId(target),
                relationship_type: r,
                layer: Layer::Observed,
                revision: 10 + id,
                properties: Vec::new(),
            })?;
        }
        Ok(graph)
    }

    #[test]
    fn delete3_fixed_path_flattens_to_seven_resident_entity_commands() -> Result<()> {
        let graph = fixed_path_delete_graph()?;
        let query = "MATCH p = (:X)-->()-->()-->() DETACH DELETE p";
        let compiled = compile_delete_query(query, &graph)?
            .ok_or_else(|| Error::internal("Delete3 fixed path did not compile"))?;
        let expected = [
            ResidentEntityBinding::Node(ResidentNodeBinding::Start),
            ResidentEntityBinding::Relationship(0),
            ResidentEntityBinding::Node(ResidentNodeBinding::Intermediate(0)),
            ResidentEntityBinding::Relationship(1),
            ResidentEntityBinding::Node(ResidentNodeBinding::Intermediate(1)),
            ResidentEntityBinding::Relationship(2),
            ResidentEntityBinding::Node(ResidentNodeBinding::End),
        ];
        assert_eq!(compiled.request.commands.len(), expected.len());
        for (command, target) in compiled.request.commands.iter().zip(expected) {
            assert_eq!(command.target, target);
            assert_eq!(
                command.selector,
                ResidentDeleteTargetSelector::EachSelectedRow
            );
            assert!(command.detach);
        }
        compiled.request.validate()?;

        let cpu = cpu_for(&graph)?;
        let validated = cpu
            .execute_delete_pipeline(&compiled.request, &CancellationToken::new())?
            .validate_for_publication(&compiled.request, BackendKind::Cpu)?;
        assert_eq!(validated.intents().len(), 7);
        assert_eq!(validated.read_dependencies().len(), 7);
        assert!(validated.final_relation().is_none());
        assert_eq!(
            validated
                .intents()
                .iter()
                .filter(|intent| intent.target_kind == EntityKind::Node)
                .count(),
            4
        );
        assert_eq!(
            validated
                .intents()
                .iter()
                .filter(|intent| intent.target_kind == EntityKind::Relationship)
                .count(),
            3
        );
        Ok(())
    }

    #[test]
    fn delete5_collected_paths_reuse_ordinal_selectors_for_every_path_entity() -> Result<()> {
        let graph = collected_path_delete_graph()?;
        let query = "MATCH p = (:User)-[r]->(:User) \
                     WITH {key: collect(p)} AS pathColls \
                     DELETE pathColls.key[0], pathColls.key[1]";
        let compiled = compile_delete_query(query, &graph)?
            .ok_or_else(|| Error::internal("Delete5 collected paths did not compile"))?;
        let path = [
            ResidentEntityBinding::Node(ResidentNodeBinding::Start),
            ResidentEntityBinding::Relationship(0),
            ResidentEntityBinding::Node(ResidentNodeBinding::End),
        ];
        assert_eq!(compiled.request.commands.len(), 6);
        for (ordinal, commands) in compiled.request.commands.chunks_exact(3).enumerate() {
            for (command, target) in commands.iter().zip(path) {
                assert_eq!(command.target, target);
                assert_eq!(
                    command.selector,
                    ResidentDeleteTargetSelector::CollectedOrdinal {
                        index: ordinal as u32,
                    }
                );
                assert!(!command.detach);
            }
        }
        compiled.request.validate()?;

        let cpu = cpu_for(&graph)?;
        let validated = cpu
            .execute_delete_pipeline(&compiled.request, &CancellationToken::new())?
            .validate_for_publication(&compiled.request, BackendKind::Cpu)?;
        assert_eq!(validated.intents().len(), 4);
        assert_eq!(validated.read_dependencies().len(), 4);
        assert!(validated.final_relation().is_none());
        assert_eq!(
            validated
                .intents()
                .iter()
                .filter(|intent| intent.target_kind == EntityKind::Node)
                .count(),
            2
        );
        assert_eq!(
            validated
                .intents()
                .iter()
                .filter(|intent| intent.target_kind == EntityKind::Relationship)
                .count(),
            2
        );
        Ok(())
    }

    fn execute_row_program_on_cpu(
        graph: &GraphStore,
        compiled: &CompiledResidentRowPlan,
    ) -> Result<crate::execution::ValidatedResidentRowProgramResult> {
        let image = ResidentProjectImage::build(
            PROJECT,
            compiled.request.expected_bookmark,
            graph,
            &TemporalStore::default(),
            &IndexCatalog::default(),
        )?;
        let mut cpu = CpuBackend::new(64 * 1024 * 1024, 8 * 1024 * 1024);
        cpu.admit_project(image)?;
        cpu.pin_project(PROJECT)?
            .execute_row_program(&compiled.request, &CancellationToken::new())?
            .validate(&compiled.request, BackendKind::Cpu)
    }

    #[test]
    fn constant_false_entity_return_with_skip_zero_is_one_sealed_empty_row_program() -> Result<()> {
        let mut graph = GraphStore::default();
        for id in 1..=2 {
            graph.insert_node(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: id,
                labels: Vec::new(),
                properties: Vec::new(),
            })?;
        }
        let compiled =
            compile_optimized_row_query("MATCH (n) WHERE 1 = 0 RETURN n SKIP 0", &graph)?
                .ok_or_else(|| Error::internal("constant-false SKIP 0 did not compile natively"))?;
        compiled.request.validate()?;
        assert_eq!(compiled.request.limit, 0);
        assert_eq!(compiled.request.offset, 0);
        assert!(compiled.request.final_registers.is_empty());
        assert!(matches!(
            compiled.outputs.as_slice(),
            [CompiledResidentRowOutput {
                source: CompiledResidentRowOutputSource::Node(ResidentNodeBinding::Start),
                ..
            }]
        ));
        let result = execute_row_program_on_cpu(&graph, &compiled)?.into_parts();
        assert!(result.source_positions.is_empty());
        assert!(result.projected_columns.is_empty());
        Ok(())
    }

    #[test]
    fn return_order_by4_2_keeps_pattern_string_filter_and_two_order_keys_native() -> Result<()> {
        let mut graph = GraphStore::default();
        let crew = graph.catalog_mut().intern_label("Crew")?;
        let name = graph.catalog_mut().intern_property("name")?;
        let rank = graph.catalog_mut().intern_property("rank")?;
        for (id, person, value) in [
            (1_u64, "Neo", 1_i64),
            (2, "Neo", 2),
            (3, "Neo", 3),
            (4, "Neo", 4),
            (5, "Neo", 5),
            (6, "Morpheus", 0),
        ] {
            graph.insert_node(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: id,
                labels: vec![crew],
                properties: vec![
                    (name, ScalarValue::String(person.into())),
                    (rank, ScalarValue::Integer(value)),
                ],
            })?;
        }
        let compiled = compile_optimized_row_query(
            "MATCH (c:Crew {name: 'Neo'}) WITH c, 0 AS relevance \
             RETURN c.rank AS rank ORDER BY relevance, c.rank",
            &graph,
        )?
        .ok_or_else(|| Error::internal("ReturnOrderBy4 [2] did not compile natively"))?;
        compiled.request.validate()?;
        assert!(matches!(
            compiled.request.input.property_filters.as_slice(),
            [ResidentPropertyFilterProgram { instructions, output: 0 }]
                if matches!(
                    instructions.as_slice(),
                    [ResidentPropertyFilterInstruction::CompareString {
                        binding: ResidentNodeBinding::Start,
                        property,
                        operation: ResidentStringPredicateOperation::Compare(CompareOp::Eq),
                        operand: Some(value),
                    }] if *property == name && value == b"Neo"
                )
        ));
        assert_eq!(
            compiled.request.sort_keys,
            [
                ResidentRowSortKey {
                    register: 0,
                    descending: false,
                    nulls_first: false,
                },
                ResidentRowSortKey {
                    register: 1,
                    descending: false,
                    nulls_first: false,
                },
            ]
        );
        let result = execute_row_program_on_cpu(&graph, &compiled)?.into_parts();
        assert!(matches!(
            result.projected_columns.as_slice(),
            [crate::execution::ResidentRowProjectedColumn {
                register: 1,
                column: ResidentRowColumn::Integer { values, validity },
            }] if values == &[1, 2, 3, 4, 5] && validity == &[1, 1, 1, 1, 1]
        ));
        Ok(())
    }

    fn execute_nullable_relation_on_cpu(
        graph: &GraphStore,
        compiled: &CompiledResidentNullableRelationPlan,
    ) -> Result<crate::execution::ValidatedResidentNullableRelationResult> {
        let image = ResidentProjectImage::build(
            PROJECT,
            compiled.request.generation.bookmark,
            graph,
            &TemporalStore::default(),
            &IndexCatalog::default(),
        )?;
        let mut cpu = CpuBackend::new(64 * 1024 * 1024, 8 * 1024 * 1024);
        cpu.admit_project(image)?;
        cpu.execute_nullable_relation(&compiled.request, &CancellationToken::new())?
            .validate(&compiled.request, ResidentDeviceCompletion::CpuReference)
    }

    #[test]
    fn mandatory_start_predicate_is_placed_before_unrelated_expansions() -> Result<()> {
        let mut graph = GraphStore::default();
        let property = graph.catalog_mut().intern_property("value")?;
        let parameters = BTreeMap::new();
        let mut builder = NullableRelationBuilder::new(graph.catalog(), &graph, &parameters);
        let start = ResidentNullableRelationSlot(0);
        let middle = ResidentNullableRelationSlot(1);
        let end = ResidentNullableRelationSlot(2);
        builder.stages = vec![
            ResidentNullableRelationStage::NodeScan {
                mode: ResidentNullableRelationMatchMode::Mandatory,
                output: start,
                labels: ResidentNullableNodeDomain::Any,
            },
            ResidentNullableRelationStage::Expand {
                mode: ResidentNullableRelationMatchMode::Mandatory,
                uniqueness_group: 0,
                source: start,
                source_labels: ResidentNullableNodeDomain::Any,
                relationship: None,
                different_from: Vec::new(),
                target: ResidentNullableRelationTarget::Introduce(middle),
                direction: ResidentDirection::Outgoing,
                relationship_types: ResidentNullableRelationshipDomain::Any,
                target_labels: ResidentNullableNodeDomain::Any,
            },
            ResidentNullableRelationStage::Expand {
                mode: ResidentNullableRelationMatchMode::Mandatory,
                uniqueness_group: 0,
                source: middle,
                source_labels: ResidentNullableNodeDomain::Any,
                relationship: None,
                different_from: Vec::new(),
                target: ResidentNullableRelationTarget::Introduce(end),
                direction: ResidentDirection::Outgoing,
                relationship_types: ResidentNullableRelationshipDomain::Any,
                target_labels: ResidentNullableNodeDomain::Any,
            },
        ];
        let predicate = ResidentNullableRelationPredicate::CompareInteger {
            left: ResidentNullableRelationPredicateValue::IntegerProperty {
                slot: start,
                kind: ResidentNullableRelationBindingKind::Node,
                property,
            },
            operation: CompareOp::Eq,
            right: ResidentNullableRelationPredicateValue::Integer(42),
        };
        assert_eq!(builder.earliest_mandatory_filter_stage(&predicate, 2), 0);

        let ResidentNullableRelationStage::Expand { mode, .. } = &mut builder.stages[1] else {
            return Err(Error::internal(
                "mandatory pushdown test lost its expansion",
            ));
        };
        *mode = ResidentNullableRelationMatchMode::Optional;
        assert_eq!(builder.earliest_mandatory_filter_stage(&predicate, 2), 2);
        Ok(())
    }

    fn execute_node_query(
        graph: &GraphStore,
        backend: &dyn ExecutionBackend,
        query: &str,
    ) -> Result<usize> {
        let output = QueryEngine.execute(query, &mut execution_context(graph, backend))?;
        let mut rows = 0_usize;
        for batch in &output.result.batches {
            rows = rows.saturating_add(batch.row_count);
            for column in &batch.columns {
                assert!(
                    column
                        .values
                        .iter()
                        .all(|value| { matches!(value, ResultValue::Node(_)) })
                );
            }
        }
        Ok(rows)
    }

    #[test]
    fn unsorted_entity_projections_decline_while_integer_lists_use_typed_rows() -> Result<()> {
        let empty = GraphStore::default();
        assert!(compile_query("MATCH (n) RETURN n", &empty)?.is_none());

        let graph = graph_with_temporal_and_document_properties()?;
        for query in [
            "MATCH (n) RETURN n",
            "MATCH (n:A:B) RETURN n",
            "MATCH (n) RETURN n AS node",
            "MATCH (n) RETURN n, n.created AS created",
        ] {
            assert!(
                compile_query(query, &graph)?.is_none(),
                "entity projection unexpectedly entered zero-key row route: {query}"
            );
        }
        let compiled = compile_query("MATCH (n) RETURN n.payload AS payload", &graph)?
            .ok_or_else(|| Error::internal("canonical integer LIST property lost its row route"))?;
        compiled.request.validate()?;
        assert!(matches!(
            compiled.request.program.instructions.as_slice(),
            [ResidentRowInstruction {
                output_type: ResidentRowValueType::List,
                operation: ResidentRowOperation::LoadListProperty { .. },
            }]
        ));
        assert!(matches!(
            compiled.outputs.as_slice(),
            [CompiledResidentRowOutput {
                name,
                source: CompiledResidentRowOutputSource::ProjectedRegister {
                    column: 0,
                    value_type: ResidentRowValueType::List,
                },
            }] if name == "payload"
        ));
        let result = execute_row_program_on_cpu(&graph, &compiled)?.into_parts();
        assert_eq!(result.rows.start_rows, [0, 1]);
        assert!(matches!(
            result.projected_columns.as_slice(),
            [projected] if matches!(
                &projected.column,
                ResidentRowColumn::List {
                    offsets,
                    values,
                    element_validity,
                    validity,
                } if offsets == &[0, 1, 1]
                    && values == &[7]
                    && element_validity == &[1]
                    && validity == &[1, 0]
            )
        ));
        Ok(())
    }

    #[test]
    fn cpu_entity_routes_still_execute_after_zero_key_compiler_declines_them() -> Result<()> {
        let empty = GraphStore::default();
        let empty_cpu = cpu_for(&empty)?;
        assert_eq!(
            execute_node_query(&empty, &empty_cpu, "MATCH (n) RETURN n")?,
            0
        );

        let graph = graph_with_temporal_and_document_properties()?;
        let cpu = cpu_for(&graph)?;
        for (query, expected_rows) in [
            ("MATCH (n) RETURN n", 2),
            ("MATCH (a:A:B) RETURN a", 1),
            ("MATCH (n) RETURN n AS node", 2),
        ] {
            assert_eq!(
                execute_node_query(&graph, &cpu, query)?,
                expected_rows,
                "{query}"
            );
        }
        Ok(())
    }

    #[test]
    fn return3_1_lowers_only_statically_proven_null_tests_to_boolean_constants() -> Result<()> {
        let mut graph = GraphStore::default();
        graph.insert_node(NodeInput {
            id: NodeId(1),
            layer: Layer::Observed,
            revision: 1,
            labels: Vec::new(),
            properties: Vec::new(),
        })?;

        let query = "MATCH (a) RETURN a.id IS NOT NULL AS a, a IS NOT NULL AS b";
        let compiled = compile_optimized_row_query(query, &graph)?
            .ok_or_else(|| Error::internal("Return3 [1] did not enter the native row route"))?;
        compiled.request.validate()?;
        assert!(compiled.request.sort_keys.is_empty());
        assert_eq!(compiled.request.final_registers, [0, 1]);
        assert_eq!(
            compiled.request.program.instructions,
            [
                ResidentRowInstruction {
                    output_type: ResidentRowValueType::Boolean,
                    operation: ResidentRowOperation::BooleanConstant(false),
                },
                ResidentRowInstruction {
                    output_type: ResidentRowValueType::Boolean,
                    operation: ResidentRowOperation::BooleanConstant(true),
                },
            ]
        );
        assert!(matches!(
            compiled.outputs.as_slice(),
            [
                CompiledResidentRowOutput {
                    name: first,
                    source: CompiledResidentRowOutputSource::ProjectedRegister {
                        column: 0,
                        value_type: ResidentRowValueType::Boolean,
                    },
                },
                CompiledResidentRowOutput {
                    name: second,
                    source: CompiledResidentRowOutputSource::ProjectedRegister {
                        column: 1,
                        value_type: ResidentRowValueType::Boolean,
                    },
                },
            ] if first == "a" && second == "b"
        ));

        let result = execute_row_program_on_cpu(&graph, &compiled)?.into_parts();
        assert_eq!(result.source_positions, [0]);
        assert_eq!(result.rows.start_rows, [0]);
        assert!(matches!(
            result.projected_columns.as_slice(),
            [first, second]
                if matches!(
                    &first.column,
                    ResidentRowColumn::Boolean { values, validity }
                        if values == &[0] && validity == &[1]
                ) && matches!(
                    &second.column,
                    ResidentRowColumn::Boolean { values, validity }
                        if values == &[1] && validity == &[1]
                )
        ));
        Ok(())
    }

    #[test]
    fn static_null_test_route_stays_fail_closed_for_per_row_or_nullable_inputs() -> Result<()> {
        let mut declared = GraphStore::default();
        let id = declared.catalog_mut().intern_property("id")?;
        declared.insert_node(NodeInput {
            id: NodeId(1),
            layer: Layer::Observed,
            revision: 1,
            labels: Vec::new(),
            properties: vec![(id, ScalarValue::Integer(7))],
        })?;
        assert!(
            compile_optimized_row_query(
                "MATCH (a) RETURN a.id IS NOT NULL AS present",
                &declared,
            )?
            .is_none(),
            "a declared property's per-row validity escaped the static proof"
        );

        let graph = GraphStore::default();
        for query in [
            "UNWIND [null] AS value RETURN value IS NOT NULL AS present",
            "OPTIONAL MATCH (a) RETURN a IS NOT NULL AS present",
        ] {
            assert!(
                compile_optimized_row_query(query, &graph)?.is_none(),
                "unsupported null-test input escaped fail-closed admission: {query}"
            );
        }
        Ok(())
    }

    #[test]
    fn return2_6_lowers_integer_property_literal_addition_and_executes_on_cpu_reference()
    -> Result<()> {
        let mut graph = GraphStore::default();
        let num = graph.catalog_mut().intern_property("num")?;
        graph.insert_node(NodeInput {
            id: NodeId(1),
            layer: Layer::Observed,
            revision: 1,
            labels: Vec::new(),
            properties: vec![(num, ScalarValue::Integer(1))],
        })?;

        let compiled = compile_optimized_row_query("MATCH (a) RETURN a.num + 1 AS foo", &graph)?
            .ok_or_else(|| Error::internal("Return2 [6] did not enter the native row route"))?;
        compiled.request.validate()?;
        assert!(compiled.request.sort_keys.is_empty());
        assert_eq!(compiled.request.final_registers, vec![2]);
        assert_eq!(compiled.request.program.instructions.len(), 3);
        assert!(matches!(
            &compiled.request.program.instructions[0],
            ResidentRowInstruction {
                output_type: ResidentRowValueType::Integer,
                operation: ResidentRowOperation::LoadIntegerProperty {
                    binding: ResidentEntityBinding::Node(ResidentNodeBinding::Start),
                    property,
                },
            } if *property == num
        ));
        assert!(matches!(
            &compiled.request.program.instructions[1],
            ResidentRowInstruction {
                output_type: ResidentRowValueType::Integer,
                operation: ResidentRowOperation::IntegerConstant(1),
            }
        ));
        assert!(matches!(
            &compiled.request.program.instructions[2],
            ResidentRowInstruction {
                output_type: ResidentRowValueType::Integer,
                operation: ResidentRowOperation::NumericAdd { left: 0, right: 1 },
            }
        ));
        assert!(matches!(
            compiled.outputs.as_slice(),
            [CompiledResidentRowOutput {
                name,
                source: CompiledResidentRowOutputSource::ProjectedRegister {
                    column: 0,
                    value_type: ResidentRowValueType::Integer,
                },
            }] if name == "foo"
        ));

        let result = execute_row_program_on_cpu(&graph, &compiled)?.into_parts();
        assert_eq!(result.source_positions, vec![0]);
        assert_eq!(result.projected_columns.len(), 1);
        assert_eq!(result.projected_columns[0].register, 2);
        assert!(matches!(
            &result.projected_columns[0].column,
            ResidentRowColumn::Integer { values, validity }
                if values == &[2] && validity == &[1]
        ));

        let reversed = compile_optimized_row_query("MATCH (a) RETURN 1 + a.num AS foo", &graph)?
            .ok_or_else(|| Error::internal("commuted integer property addition did not compile"))?;
        reversed.request.validate()?;
        assert!(matches!(
            &reversed.request.program.instructions[2].operation,
            ResidentRowOperation::NumericAdd { left: 0, right: 1 }
        ));
        assert!(matches!(
            &reversed.request.program.instructions[0].operation,
            ResidentRowOperation::IntegerConstant(1)
        ));
        assert!(matches!(
            &reversed.request.program.instructions[1].operation,
            ResidentRowOperation::LoadIntegerProperty { property, .. } if *property == num
        ));
        Ok(())
    }

    #[test]
    fn boolean_property_to_string_is_one_typed_row_conversion_with_exact_capacity() -> Result<()> {
        let mut graph = GraphStore::default();
        let movie = graph.catalog_mut().intern_label("Movie")?;
        let watched = graph.catalog_mut().intern_property("watched")?;
        let rating = graph.catalog_mut().intern_property("rating")?;
        let title = graph.catalog_mut().intern_property("title")?;
        for (id, properties) in [
            (
                1,
                vec![
                    (watched, ScalarValue::Boolean(true)),
                    (rating, ScalarValue::Integer(4)),
                    (title, ScalarValue::String("first".into())),
                ],
            ),
            (2, vec![(watched, ScalarValue::Boolean(false))]),
            (3, Vec::new()),
        ] {
            graph.insert_node(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: id,
                labels: vec![movie],
                properties,
            })?;
        }

        let compiled = compile_optimized_row_query(
            "MATCH (m:Movie) RETURN toString(m.watched) AS watched",
            &graph,
        )?
        .ok_or_else(|| {
            Error::internal("Boolean property toString did not enter the typed-row route")
        })?;
        compiled.request.validate()?;
        assert!(compiled.request.sort_keys.is_empty());
        assert_eq!(compiled.request.final_registers, [1]);
        assert_eq!(
            compiled.request.program.instructions,
            [
                ResidentRowInstruction {
                    output_type: ResidentRowValueType::Boolean,
                    operation: ResidentRowOperation::LoadBooleanProperty {
                        binding: ResidentEntityBinding::Node(ResidentNodeBinding::Start),
                        property: watched,
                    },
                },
                ResidentRowInstruction {
                    output_type: ResidentRowValueType::String,
                    operation: ResidentRowOperation::BooleanToString { operand: 0 },
                },
            ]
        );
        assert_eq!(
            compiled.request.program.string_register_capacity(1)?,
            Some(5)
        );
        assert!(matches!(
            compiled.outputs.as_slice(),
            [CompiledResidentRowOutput {
                name,
                source: CompiledResidentRowOutputSource::ProjectedRegister {
                    column: 0,
                    value_type: ResidentRowValueType::String,
                },
            }] if name == "watched"
        ));

        let result = execute_row_program_on_cpu(&graph, &compiled)?.into_parts();
        assert_eq!(result.source_positions, [0, 1, 2]);
        assert!(matches!(
            result.projected_columns.as_slice(),
            [projected] if projected.register == 1 && matches!(
                &projected.column,
                ResidentRowColumn::String { offsets, bytes, validity }
                    if offsets == &[0, 4, 9, 9]
                        && bytes == b"truefalse"
                        && validity == &[1, 1, 0]
            )
        ));

        let wrong_input = ResidentRowProgram {
            instructions: vec![
                ResidentRowInstruction {
                    output_type: ResidentRowValueType::Integer,
                    operation: ResidentRowOperation::IntegerConstant(1),
                },
                ResidentRowInstruction {
                    output_type: ResidentRowValueType::String,
                    operation: ResidentRowOperation::BooleanToString { operand: 0 },
                },
            ],
        };
        assert_eq!(
            wrong_input
                .validate()
                .expect_err("INTEGER toString was admitted")
                .code,
            ErrorCode::QueryType
        );
        let forward_input = ResidentRowProgram {
            instructions: vec![ResidentRowInstruction {
                output_type: ResidentRowValueType::String,
                operation: ResidentRowOperation::BooleanToString { operand: 0 },
            }],
        };
        assert_eq!(
            forward_input
                .validate()
                .expect_err("self-referential Boolean toString was admitted")
                .code,
            ErrorCode::QueryType
        );
        let mut wrong_output = compiled.request.program.clone();
        wrong_output.instructions[1].output_type = ResidentRowValueType::Boolean;
        assert_eq!(
            wrong_output
                .validate()
                .expect_err("Boolean toString with a Boolean output was admitted")
                .code,
            ErrorCode::QueryType
        );
        let mut fingerprint_tamper = compiled.request.clone();
        fingerprint_tamper.program.instructions[1].operation =
            ResidentRowOperation::StringConstant("true".to_owned());
        assert_eq!(
            fingerprint_tamper
                .validate()
                .expect_err("row-operation fingerprint tamper was admitted")
                .code,
            ErrorCode::GpuAdmissionFailure
        );

        for query in [
            "MATCH (m:Movie) RETURN toString(m.rating) AS value",
            "MATCH (m:Movie) RETURN toString(m.title) AS value",
        ] {
            assert!(
                compile_optimized_row_query(query, &graph)?.is_none(),
                "non-Boolean toString escaped the Boolean-only typed-row opcode: {query}"
            );
        }
        Ok(())
    }

    #[test]
    fn return_skip_limit2_5_uses_one_sealed_empty_window_without_catalog_resolution() -> Result<()>
    {
        let graph = GraphStore::default();
        let source = "MATCH (p:Person) RETURN p.name AS name ORDER BY p.name LIMIT 0";
        let compiled = compile_optimized_row_query(source, &graph)?.ok_or_else(|| {
            Error::internal("ReturnSkipLimit2 [5] did not enter the empty-window route")
        })?;
        compiled.request.validate()?;
        assert_eq!(compiled.request.scalar_input_rows()?, Some(0));
        assert!(!compiled.request.has_graph_input());
        assert_eq!(compiled.request.input.max_output_rows, 0);
        assert_eq!(compiled.request.offset, 0);
        assert_eq!(compiled.request.limit, 0);
        assert_eq!(compiled.request.max_output_rows, 0);
        assert_eq!(compiled.request.final_registers, [0]);
        assert!(matches!(
            compiled.request.program.instructions.as_slice(),
            [ResidentRowInstruction {
                output_type: ResidentRowValueType::Integer,
                operation: ResidentRowOperation::InputColumn(ResidentRowColumn::Integer {
                    values,
                    validity,
                }),
            }] if values.is_empty() && validity.is_empty()
        ));
        assert_eq!(
            compiled.request.sort_keys,
            [ResidentRowSortKey {
                register: 0,
                descending: false,
                nulls_first: false,
            }]
        );
        assert!(matches!(
            compiled.outputs.as_slice(),
            [CompiledResidentRowOutput {
                name,
                source: CompiledResidentRowOutputSource::ProjectedRegister {
                    column: 0,
                    value_type: ResidentRowValueType::Integer,
                },
            }] if name == "name"
        ));
        let result = execute_row_program_on_cpu(&graph, &compiled)?.into_parts();
        assert!(result.source_positions.is_empty());
        assert!(result.rows.start_rows.is_empty());
        assert!(matches!(
            result.projected_columns.as_slice(),
            [projected]
                if projected.register == 0
                    && matches!(
                        &projected.column,
                        ResidentRowColumn::Integer { values, validity }
                            if values.is_empty() && validity.is_empty()
                    )
        ));

        for query in [
            "MATCH (p:Person) RETURN p.name AS name ORDER BY p.name LIMIT 1",
            "MATCH (p:Person) WHERE p.name = 'x' RETURN p.name AS name ORDER BY p.name LIMIT 0",
            "MATCH (p:Person) RETURN 1 / 0 AS name ORDER BY name LIMIT 0",
            "MATCH (p:Person) RETURN rand() AS name ORDER BY name LIMIT 0",
            "MATCH (p:Person) RETURN p.name AS name ORDER BY p.name SKIP 1 LIMIT 0",
        ] {
            assert!(
                compile_optimized_row_query(query, &graph)?.is_none(),
                "unsupported zero-window shape escaped its exact proof: {query}"
            );
        }
        Ok(())
    }

    #[test]
    fn mathematical2_1_filters_then_adds_in_one_native_row_command() -> Result<()> {
        let mut graph = GraphStore::default();
        let id = graph.catalog_mut().intern_property("id")?;
        let version = graph.catalog_mut().intern_property("version")?;
        graph.insert_node(NodeInput {
            id: NodeId(1),
            layer: Layer::Observed,
            revision: 1,
            labels: Vec::new(),
            properties: vec![
                (id, ScalarValue::Integer(1_337)),
                (version, ScalarValue::Integer(99)),
            ],
        })?;

        let query = "MATCH (a) WHERE a.id = 1337 RETURN a.version + 5";
        let compiled = compile_optimized_row_query(query, &graph)?.ok_or_else(|| {
            Error::internal("Mathematical2 [1] did not enter the native row route")
        })?;
        compiled.request.validate()?;
        assert_eq!(
            compiled.request.input.predicates,
            vec![ResidentI64Predicate {
                binding: ResidentNodeBinding::Start,
                property: id,
                operation: CompareOp::Eq,
                operand: 1_337,
            }]
        );
        assert!(compiled.request.input.property_filters.is_empty());
        assert!(compiled.request.sort_keys.is_empty());
        assert_eq!(compiled.request.final_registers, vec![2]);
        assert_eq!(
            compiled.request.program.instructions,
            vec![
                ResidentRowInstruction {
                    output_type: ResidentRowValueType::Integer,
                    operation: ResidentRowOperation::LoadIntegerProperty {
                        binding: ResidentEntityBinding::Node(ResidentNodeBinding::Start),
                        property: version,
                    },
                },
                ResidentRowInstruction {
                    output_type: ResidentRowValueType::Integer,
                    operation: ResidentRowOperation::IntegerConstant(5),
                },
                ResidentRowInstruction {
                    output_type: ResidentRowValueType::Integer,
                    operation: ResidentRowOperation::NumericAdd { left: 0, right: 1 },
                },
            ]
        );
        assert!(matches!(
            compiled.outputs.as_slice(),
            [CompiledResidentRowOutput {
                name,
                source: CompiledResidentRowOutputSource::ProjectedRegister {
                    column: 0,
                    value_type: ResidentRowValueType::Integer,
                },
            }] if name == "a.version + 5"
        ));

        let result = execute_row_program_on_cpu(&graph, &compiled)?.into_parts();
        assert_eq!(result.source_positions, vec![0]);
        assert!(matches!(
            result.projected_columns.as_slice(),
            [projected]
                if projected.register == 2
                    && matches!(
                        &projected.column,
                        ResidentRowColumn::Integer { values, validity }
                            if values == &[104] && validity == &[1]
                    )
        ));

        let reversed = compile_optimized_row_query(
            "MATCH (a) WHERE 1000 < a.id RETURN a.version + 5",
            &graph,
        )?
        .ok_or_else(|| Error::internal("reversed integer filter did not enter the row route"))?;
        assert_eq!(
            reversed.request.input.predicates,
            vec![ResidentI64Predicate {
                binding: ResidentNodeBinding::Start,
                property: id,
                operation: CompareOp::Greater,
                operand: 1_000,
            }]
        );
        Ok(())
    }

    #[test]
    fn mathematical2_filter_keeps_zero_multi_null_and_overflow_semantics() -> Result<()> {
        let mut graph = GraphStore::default();
        let id = graph.catalog_mut().intern_property("id")?;
        let version = graph.catalog_mut().intern_property("version")?;
        for (node, properties) in [
            (
                1,
                vec![
                    (id, ScalarValue::Integer(1_337)),
                    (version, ScalarValue::Integer(99)),
                ],
            ),
            (
                2,
                vec![
                    (id, ScalarValue::Integer(7)),
                    // The predicate must execute before projection: this discarded row would
                    // overflow the addition if the compiler hoisted expression evaluation.
                    (version, ScalarValue::Integer(i64::MAX)),
                ],
            ),
            (
                3,
                vec![
                    (id, ScalarValue::Integer(1_337)),
                    (version, ScalarValue::Integer(100)),
                ],
            ),
            (4, vec![(id, ScalarValue::Integer(1_337))]),
        ] {
            graph.insert_node(NodeInput {
                id: NodeId(node),
                layer: Layer::Observed,
                revision: node,
                labels: Vec::new(),
                properties,
            })?;
        }

        let matched = compile_optimized_row_query(
            "MATCH (a) WHERE a.id = 1337 RETURN a.version + 5",
            &graph,
        )?
        .ok_or_else(|| Error::internal("multi-row Mathematical2 shape did not compile"))?;
        let matched = execute_row_program_on_cpu(&graph, &matched)?.into_parts();
        assert_eq!(matched.source_positions, vec![0, 1, 2]);
        assert_eq!(matched.rows.start_rows, vec![0, 2, 3]);
        assert!(matches!(
            matched.projected_columns.as_slice(),
            [projected]
                if matches!(
                    &projected.column,
                    ResidentRowColumn::Integer { values, validity }
                        if values == &[104, 105, 0] && validity == &[1, 1, 0]
                )
        ));

        let empty = compile_optimized_row_query(
            "MATCH (a) WHERE a.id = 9999 RETURN a.version + 5",
            &graph,
        )?
        .ok_or_else(|| Error::internal("zero-match Mathematical2 shape did not compile"))?;
        let empty = execute_row_program_on_cpu(&graph, &empty)?.into_parts();
        assert!(empty.source_positions.is_empty());
        assert!(matches!(
            empty.projected_columns.as_slice(),
            [projected]
                if matches!(
                    &projected.column,
                    ResidentRowColumn::Integer { values, validity }
                        if values.is_empty() && validity.is_empty()
                )
        ));

        let mut overflow_graph = GraphStore::default();
        let overflow_id = overflow_graph.catalog_mut().intern_property("id")?;
        let overflow_version = overflow_graph.catalog_mut().intern_property("version")?;
        overflow_graph.insert_node(NodeInput {
            id: NodeId(1),
            layer: Layer::Observed,
            revision: 1,
            labels: Vec::new(),
            properties: vec![
                (overflow_id, ScalarValue::Integer(1_337)),
                (overflow_version, ScalarValue::Integer(i64::MAX)),
            ],
        })?;
        let overflow = compile_optimized_row_query(
            "MATCH (a) WHERE a.id = 1337 RETURN a.version + 5",
            &overflow_graph,
        )?
        .ok_or_else(|| Error::internal("overflow Mathematical2 shape did not compile"))?;
        let error = execute_row_program_on_cpu(&overflow_graph, &overflow)
            .expect_err("integer overflow must fail the complete row command");
        assert_eq!(error.code, ErrorCode::QueryType);
        assert_eq!(error.message, "integer arithmetic overflow");
        Ok(())
    }

    #[test]
    fn mathematical2_filter_adapter_stays_fail_closed() -> Result<()> {
        let mut graph = GraphStore::default();
        let id = graph.catalog_mut().intern_property("id")?;
        let version = graph.catalog_mut().intern_property("version")?;
        graph.insert_node(NodeInput {
            id: NodeId(1),
            layer: Layer::Observed,
            revision: 1,
            labels: Vec::new(),
            properties: vec![
                (id, ScalarValue::Integer(1_337)),
                (version, ScalarValue::Integer(99)),
            ],
        })?;
        for query in [
            "MATCH (a) WHERE a.id = '1337' RETURN a.version + 5",
            "MATCH (a) WHERE a.id = 1337.0 RETURN a.version + 5",
            "MATCH (a) WHERE a.id + 0 = 1337 RETURN a.version + 5",
            "MATCH (a) WHERE a.id = 1337 AND a.version = 99 RETURN a.version + 5",
            "OPTIONAL MATCH (a) WHERE a.id = 1337 RETURN a.version + 5",
        ] {
            assert!(
                compile_optimized_row_query(query, &graph)?.is_none(),
                "unsupported filter shape escaped fail-closed admission: {query}"
            );
        }

        let mut string_graph = GraphStore::default();
        let string_id = string_graph.catalog_mut().intern_property("id")?;
        let string_version = string_graph.catalog_mut().intern_property("version")?;
        string_graph.insert_node(NodeInput {
            id: NodeId(1),
            layer: Layer::Observed,
            revision: 1,
            labels: Vec::new(),
            properties: vec![
                (string_id, ScalarValue::String(Arc::from("1337"))),
                (string_version, ScalarValue::Integer(99)),
            ],
        })?;
        assert!(
            compile_optimized_row_query(
                "MATCH (a) WHERE a.id = 1337 RETURN a.version + 5",
                &string_graph,
            )?
            .is_none(),
            "non-integer predicate property escaped fail-closed admission"
        );
        Ok(())
    }

    #[test]
    fn unsorted_integer_property_addition_stays_fail_closed_for_unsupported_shapes() -> Result<()> {
        let graph = nullable_typed_projection_fixture_graph()?;
        for query in [
            "MATCH (a) RETURN a.float + 1 AS foo",
            "MATCH (a) RETURN a.score + 1.0 AS foo",
            "MATCH (a) RETURN a.node_mixed + 1 AS foo",
            "MATCH (a) RETURN a.name + 'x' AS foo",
            "MATCH (a) RETURN a.score + a.score AS foo",
            "MATCH (a) RETURN a.score + 1 + 1 AS foo",
            "MATCH (a) RETURN a.score - 1 AS foo",
        ] {
            assert!(
                compile_optimized_row_query(query, &graph)?.is_none(),
                "unsupported unsorted computation escaped fail-closed admission: {query}"
            );
        }
        Ok(())
    }

    #[test]
    fn unsorted_route_requires_a_direct_supported_property_register() -> Result<()> {
        let graph = graph_with_temporal_and_document_properties()?;
        let compiled = compile_query("MATCH (n) RETURN n.created AS created", &graph)?
            .ok_or_else(|| Error::internal("direct temporal property projection was rejected"))?;
        assert!(compiled.request.sort_keys.is_empty());
        assert_eq!(compiled.request.offset, 0);
        assert_eq!(compiled.request.limit, usize::MAX);
        assert_eq!(compiled.request.program.instructions.len(), 1);
        assert_eq!(compiled.request.final_registers, vec![0]);
        assert!(matches!(
            compiled.request.program.instructions[0].operation,
            ResidentRowOperation::LoadDateProperty { .. }
        ));
        assert!(matches!(
            compiled.outputs.as_slice(),
            [CompiledResidentRowOutput {
                source: CompiledResidentRowOutputSource::ProjectedRegister {
                    column: 0,
                    value_type: ResidentRowValueType::Date,
                },
                ..
            }]
        ));

        assert!(
            compile_query(
                "MATCH (n) RETURN n.created + duration('P1D') AS created",
                &graph,
            )?
            .is_none(),
            "computed temporal expression unexpectedly widened the zero-key prerequisite"
        );
        Ok(())
    }

    #[test]
    fn unsorted_fixed_width_unwind_applies_skip_and_limit_natively() -> Result<()> {
        let graph = GraphStore::default();
        let compiled = compile_query(
            "UNWIND [10, 20, 30, 40] AS value RETURN value SKIP 1 LIMIT 2",
            &graph,
        )?
        .ok_or_else(|| Error::internal("unsorted fixed-width pagination was rejected"))?;
        compiled.request.validate()?;
        assert!(compiled.request.sort_keys.is_empty());
        assert_eq!(compiled.request.offset, 1);
        assert_eq!(compiled.request.limit, 2);
        assert_eq!(compiled.request.final_registers, vec![0]);
        assert!(matches!(
            &compiled.request.program.instructions[0].operation,
            ResidentRowOperation::InputColumn(ResidentRowColumn::Integer {
                values,
                validity,
            }) if values == &[10, 20, 30, 40] && validity == &[1, 1, 1, 1]
        ));

        let result = execute_row_program_on_cpu(&graph, &compiled)?.into_parts();
        assert_eq!(result.source_positions, vec![1, 2]);
        assert!(matches!(
            &result.projected_columns[0].column,
            ResidentRowColumn::Integer { values, validity }
                if values == &[20, 30] && validity == &[1, 1]
        ));
        Ok(())
    }

    #[test]
    fn sorted_property_route_keeps_its_existing_contract() -> Result<()> {
        let graph = graph_with_temporal_and_document_properties()?;
        let compiled = compile_query(
            "MATCH (n) WITH n.created AS created ORDER BY created LIMIT 1 RETURN created",
            &graph,
        )?
        .ok_or_else(|| Error::internal("existing sorted property route was rejected"))?;
        assert_eq!(compiled.request.sort_keys.len(), 1);
        assert_eq!(compiled.request.limit, 1);
        assert_eq!(compiled.request.final_registers, vec![0]);
        Ok(())
    }

    #[test]
    fn cpu_executes_unsorted_temporal_property_and_null_in_stable_scan_order() -> Result<()> {
        let graph = graph_with_temporal_and_document_properties()?;
        let bookmark = Bookmark {
            term: 3,
            index: graph.revision(),
        };
        let compiled = compile_query("MATCH (n) RETURN n.created AS created", &graph)?
            .ok_or_else(|| Error::internal("direct temporal property projection was rejected"))?;
        let image = ResidentProjectImage::build(
            PROJECT,
            bookmark,
            &graph,
            &TemporalStore::default(),
            &IndexCatalog::default(),
        )?;
        let mut cpu = CpuBackend::new(64 * 1024 * 1024, 8 * 1024 * 1024);
        cpu.admit_project(image)?;
        let pinned = cpu.pin_project(PROJECT)?;
        let raw = pinned.execute_row_program(&compiled.request, &CancellationToken::new())?;
        let parts = raw
            .validate(&compiled.request, BackendKind::Cpu)?
            .into_parts();
        assert_eq!(parts.source_positions, vec![0, 1]);
        assert_eq!(parts.projected_columns.len(), 1);
        assert_eq!(parts.projected_columns[0].register, 0);
        let ResidentRowColumn::Date { days, validity } = &parts.projected_columns[0].column else {
            return Err(Error::internal(
                "CPU unsorted temporal projection returned the wrong column type",
            ));
        };
        assert_eq!(days, &[42, 0]);
        assert_eq!(validity, &[1, 0]);
        assert!(parts.receipts.iter().all(|receipt| {
            receipt.completion == ResidentDeviceCompletion::CpuReference
                && receipt.input_cardinality == 2
                && receipt.output_cardinality == 2
        }));
        Ok(())
    }

    #[test]
    fn zero_key_manifest_remains_fenced_by_the_compiled_request() -> Result<()> {
        let graph = graph_with_temporal_and_document_properties()?;
        let compiled = compile_query("MATCH (n) RETURN n.created AS created", &graph)?
            .ok_or_else(|| Error::internal("direct temporal property projection was rejected"))?;
        compiled.request.validate()?;
        assert_eq!(
            compiled.request.obligations().count(),
            compiled.request.program.instructions.len() + 1
        );
        assert_ne!(
            compiled.request.manifest.fingerprint,
            crate::execution::ResidentRowManifestFingerprint([0; 32])
        );
        Ok(())
    }

    #[test]
    fn match7_first_eleven_exact_queries_seal_fixed_optional_boundaries() -> Result<()> {
        let graph = fixed_optional_predicate_fixture_graph()?;
        let cases = [
            ("Match7 [1]", "OPTIONAL MATCH (n) RETURN n", (0, 0, 0), 0),
            (
                "Match7 [2]",
                "MATCH (n) OPTIONAL MATCH (n)-[:NOT_EXIST]->(x) RETURN n, x",
                (0, 0, 0),
                0,
            ),
            (
                "Match7 [3]",
                "MATCH (a:A), (b:C) OPTIONAL MATCH (x)-->(b) RETURN x",
                (0, 0, 0),
                0,
            ),
            (
                "Match7 [4]",
                "MATCH (a1)-[r]->() WITH r, a1 LIMIT 1 OPTIONAL MATCH (a1)<-[r]-(b2) RETURN a1, r, b2",
                (0, 0, 0),
                0,
            ),
            (
                "Match7 [5]",
                "MATCH ()-[r]->() WITH r LIMIT 1 OPTIONAL MATCH (a2)-[r]->(b2) RETURN a2, r, b2",
                (0, 1, 0),
                1,
            ),
            (
                "Match7 [6]",
                "MATCH (a1)-[r]->() WITH r, a1 LIMIT 1 OPTIONAL MATCH (a1)-[r]->(b2) RETURN a1, r, b2",
                (0, 0, 0),
                0,
            ),
            (
                "Match7 [7]",
                "MATCH (a {name: 'A'}) OPTIONAL MATCH (a)-[:KNOWS]->()-[:KNOWS]->(foo) RETURN foo",
                (1, 0, 0),
                1,
            ),
            (
                "Match7 [8]",
                "MATCH (a:A), (c:C) OPTIONAL MATCH (a)-->(b)-->(c) RETURN b",
                (0, 0, 0),
                1,
            ),
            (
                "Match7 [9]",
                "MATCH (a:Single), (c:C) OPTIONAL MATCH (a)-->(b)-->(c) RETURN b",
                (0, 0, 0),
                1,
            ),
            (
                "Match7 [10]",
                "OPTIONAL MATCH (a) WITH a OPTIONAL MATCH (a)-->(b) RETURN b",
                (0, 0, 0),
                0,
            ),
            (
                "Match7 [11]",
                "MATCH (a)-[r {name: 'r1'}]-(b) OPTIONAL MATCH (b)-[r2]-(c) WHERE r <> r2 RETURN a, b, c",
                (1, 1, 0),
                0,
            ),
        ];

        for (scenario, query, expected_filters, expected_groups) in cases {
            let compiled = compile_nullable_query(query, &graph, 64)?.ok_or_else(|| {
                Error::internal(format!(
                    "{scenario} did not compile as a fixed nullable relation"
                ))
            })?;
            compiled.request.validate()?;
            assert_eq!(
                nullable_filter_placement_counts(&compiled),
                expected_filters,
                "{scenario} attached a predicate outside its Cypher boundary"
            );
            assert_eq!(
                compiled.request.predicate_program.optional_groups.len(),
                expected_groups,
                "{scenario} did not preserve atomic OPTIONAL null extension"
            );
        }
        Ok(())
    }

    #[test]
    fn match7_15_eliminates_only_a_variable_traversal_from_a_proven_null_source() -> Result<()> {
        let mut graph = optional_fixture_graph()?;
        let bar = graph.catalog_mut().intern_relationship_type("BAR")?;
        graph.insert_edge(EdgeInput {
            id: EdgeId(3),
            source: NodeId(3),
            target: NodeId(4),
            relationship_type: bar,
            layer: Layer::Observed,
            revision: 7,
            properties: Vec::new(),
        })?;
        let query = "MATCH (a:A) \
                     OPTIONAL MATCH (a)-[:FOO]->(b:B) \
                     OPTIONAL MATCH (b)<-[:BAR*]-(c:B) \
                     RETURN a, b, c";
        let compiled = compile_nullable_query(query, &graph, 64)?
            .ok_or_else(|| Error::internal("Match7 [15] did not compile natively"))?;
        compiled.request.validate()?;

        let stages = &compiled.request.program.stages;
        assert_eq!(stages.len(), 4);
        let b = match &stages[1] {
            ResidentNullableRelationStage::Expand {
                mode: ResidentNullableRelationMatchMode::Optional,
                relationship_types: ResidentNullableRelationshipDomain::KnownEmpty,
                target: ResidentNullableRelationTarget::Introduce(target),
                ..
            } => *target,
            stage => {
                return Err(Error::internal(format!(
                    "Match7 [15] did not prove its first endpoint null: {stage:?}"
                )));
            }
        };
        let c = match &stages[2] {
            ResidentNullableRelationStage::Expand {
                mode: ResidentNullableRelationMatchMode::Optional,
                source,
                relationship: None,
                direction: ResidentDirection::Incoming,
                relationship_types: ResidentNullableRelationshipDomain::Known(types),
                target: ResidentNullableRelationTarget::Introduce(target),
                ..
            } if *source == b && types == &vec![bar] => *target,
            stage => {
                return Err(Error::internal(format!(
                    "Match7 [15] did not retain the real BAR domain behind its null source: {stage:?}"
                )));
            }
        };
        assert!(matches!(
            &stages[3],
            ResidentNullableRelationStage::FinalProject { bindings }
                if bindings.len() == 3
                    && bindings[1].source
                        == ResidentNullableRelationOutputSource::Entity {
                            slot: b,
                            kind: ResidentNullableRelationBindingKind::Node,
                        }
                    && bindings[2].source
                        == ResidentNullableRelationOutputSource::Entity {
                            slot: c,
                            kind: ResidentNullableRelationBindingKind::Node,
                        }
        ));

        let result = execute_nullable_relation_on_cpu(&graph, &compiled)?;
        assert_eq!(result.row_count(), 1);
        let [
            ResidentNullableRelationOutputColumn::Entity { rows: a_rows, .. },
            ResidentNullableRelationOutputColumn::Entity { rows: b_rows, .. },
            ResidentNullableRelationOutputColumn::Entity { rows: c_rows, .. },
        ] = result.columns()
        else {
            return Err(Error::internal(
                "Match7 [15] returned the wrong native columns",
            ));
        };
        assert_ne!(
            a_rows,
            &[crate::execution::RESIDENT_NULLABLE_RELATION_NULL_ROW]
        );
        assert_eq!(
            b_rows,
            &[crate::execution::RESIDENT_NULLABLE_RELATION_NULL_ROW]
        );
        assert_eq!(
            c_rows,
            &[crate::execution::RESIDENT_NULLABLE_RELATION_NULL_ROW]
        );
        assert!(
            result
                .receipts()
                .iter()
                .all(|receipt| { receipt.completion == ResidentDeviceCompletion::CpuReference })
        );
        Ok(())
    }

    #[test]
    fn match7_22_known_null_coalesce_stays_inside_the_nullable_relation() -> Result<()> {
        let graph = fixed_optional_predicate_fixture_graph()?;
        let query = "MATCH (a:Single) \
                     OPTIONAL MATCH (a)-->(b:NonExistent) \
                     OPTIONAL MATCH (a)-->(c:NonExistent) \
                     WITH coalesce(b, c) AS x \
                     MATCH (x)-->(d) \
                     RETURN d";
        let compiled = compile_nullable_query(query, &graph, 64)?
            .ok_or_else(|| Error::internal("Match7 [22] did not compile natively"))?;
        compiled.request.validate()?;

        let stages = &compiled.request.program.stages;
        assert_eq!(stages.len(), 6);
        assert!(matches!(
            &stages[0],
            ResidentNullableRelationStage::NodeScan {
                mode: ResidentNullableRelationMatchMode::Mandatory,
                ..
            }
        ));
        let b = match &stages[1] {
            ResidentNullableRelationStage::Expand {
                mode: ResidentNullableRelationMatchMode::Optional,
                target: ResidentNullableRelationTarget::Introduce(target),
                target_labels: ResidentNullableNodeDomain::KnownEmpty,
                ..
            } => *target,
            stage => {
                return Err(Error::internal(format!(
                    "Match7 [22] emitted the wrong first OPTIONAL stage: {stage:?}"
                )));
            }
        };
        let c = match &stages[2] {
            ResidentNullableRelationStage::Expand {
                mode: ResidentNullableRelationMatchMode::Optional,
                target: ResidentNullableRelationTarget::Introduce(target),
                target_labels: ResidentNullableNodeDomain::KnownEmpty,
                ..
            } => *target,
            stage => {
                return Err(Error::internal(format!(
                    "Match7 [22] emitted the wrong second OPTIONAL stage: {stage:?}"
                )));
            }
        };
        let x = match &stages[3] {
            ResidentNullableRelationStage::ScopeProject { bindings }
                if bindings.len() == 1
                    && bindings[0].variable == "x"
                    && matches!(bindings[0].source, source if source == b || source == c) =>
            {
                bindings[0].output
            }
            stage => {
                return Err(Error::internal(format!(
                    "Match7 [22] did not preserve the proven-null coalesce scope: {stage:?}"
                )));
            }
        };
        let d = match &stages[4] {
            ResidentNullableRelationStage::Expand {
                mode: ResidentNullableRelationMatchMode::Mandatory,
                source,
                target: ResidentNullableRelationTarget::Introduce(target),
                ..
            } if *source == x => *target,
            stage => {
                return Err(Error::internal(format!(
                    "Match7 [22] did not correlate the mandatory MATCH to x: {stage:?}"
                )));
            }
        };
        assert!(matches!(
            &stages[5],
            ResidentNullableRelationStage::FinalProject { bindings }
                if bindings.len() == 1
                    && bindings[0].name == "d"
                    && bindings[0].source
                        == ResidentNullableRelationOutputSource::Entity {
                            slot: d,
                            kind: ResidentNullableRelationBindingKind::Node,
                        }
        ));

        let bookmark = compiled.request.generation.bookmark;
        let image = ResidentProjectImage::build(
            PROJECT,
            bookmark,
            &graph,
            &TemporalStore::default(),
            &IndexCatalog::default(),
        )?;
        let mut cpu = CpuBackend::new(64 * 1024 * 1024, 8 * 1024 * 1024);
        cpu.admit_project(image)?;
        let result = cpu
            .execute_nullable_relation(&compiled.request, &CancellationToken::new())?
            .validate(&compiled.request, ResidentDeviceCompletion::CpuReference)?;
        assert_eq!(result.row_count(), 0);
        assert_eq!(result.columns().len(), 1);

        let unsafe_query = "MATCH (a:Single) \
                            OPTIONAL MATCH (a)-->(b:NonExistent) \
                            OPTIONAL MATCH (a)-->(c:A) \
                            WITH coalesce(b, c) AS x \
                            MATCH (x)-->(d) \
                            RETURN d";
        assert!(
            compile_nullable_query(unsafe_query, &graph, 64)?.is_none(),
            "a maybe-non-null coalesce must remain outside this exact native subset"
        );
        Ok(())
    }

    #[test]
    fn match_where6_exact_queries_attach_where_to_optional_candidates() -> Result<()> {
        let graph = fixed_optional_predicate_fixture_graph()?;
        let cases = [
            (
                "MatchWhere6 [1]",
                "MATCH (a)-->(b) WHERE b:B OPTIONAL MATCH (a)-->(c) WHERE c:C RETURN a.name",
                (1, 1, 0),
                0,
            ),
            (
                "MatchWhere6 [2]",
                "MATCH (n:Single) OPTIONAL MATCH (n)-[r]-(m) WHERE m:NonExistent RETURN r",
                (0, 1, 0),
                0,
            ),
            (
                "MatchWhere6 [3]",
                "MATCH (n:Single) OPTIONAL MATCH (n)-[r]-(m) WHERE m.num = 42 RETURN m",
                (0, 1, 0),
                0,
            ),
            (
                "MatchWhere6 [4]",
                "MATCH (n)-->(x0) OPTIONAL MATCH (x0)-->(x1) WHERE x1.name = 'bar' RETURN x0.name",
                (0, 1, 0),
                0,
            ),
            (
                "MatchWhere6 [5]",
                "MATCH (a1)-[r]->() WITH r, a1 LIMIT 1 OPTIONAL MATCH (a2)<-[r]-(b2) WHERE a1 = a2 RETURN a1, r, b2, a2",
                (0, 1, 1),
                1,
            ),
            (
                "MatchWhere6 [6]",
                "MATCH (x:X) OPTIONAL MATCH (x)-[:E1]->(y:Y) WHERE x.val < y.val RETURN x, y",
                (0, 1, 0),
                0,
            ),
            (
                "MatchWhere6 [7]",
                "MATCH (x:X) OPTIONAL MATCH (x)-[:E1]->(y:Y)-[:E2]->(z:Z) WHERE x.val < z.val RETURN x, y, z",
                (0, 0, 1),
                1,
            ),
            (
                "MatchWhere6 [8]",
                "MATCH (x:X) OPTIONAL MATCH (x)-[:E1]->(y:Y) OPTIONAL MATCH (y)-[:E2]->(z:Z) WHERE x.val < z.val RETURN x, y, z",
                (0, 1, 0),
                0,
            ),
        ];

        for (scenario, query, expected_filters, expected_groups) in cases {
            let compiled = compile_nullable_query(query, &graph, 64)?.ok_or_else(|| {
                Error::internal(format!(
                    "{scenario} did not compile as a fixed nullable relation"
                ))
            })?;
            compiled.request.validate()?;
            assert_eq!(
                nullable_filter_placement_counts(&compiled),
                expected_filters,
                "{scenario} applied WHERE after null extension instead of to candidates"
            );
            assert_eq!(
                compiled.request.predicate_program.optional_groups.len(),
                expected_groups,
                "{scenario} emitted the wrong OPTIONAL group boundary"
            );
        }
        Ok(())
    }

    #[test]
    fn match_where5_mixed_string_ordering_lowers_with_exact_three_valued_logic() -> Result<()> {
        let graph = fixed_optional_predicate_fixture_graph()?;
        let var = graph
            .catalog()
            .property("var")
            .ok_or_else(|| Error::internal("fixed OPTIONAL fixture omitted `var`"))?;
        assert!(graph.snapshot()?.node_properties.is_mixed(var));

        for (scenario, query, expected) in [
            (
                "MatchWhere5 [1]",
                "MATCH (:Root {name: 'x'})-->(i:TextNode) WHERE i.var > 'te' RETURN i",
                1,
            ),
            (
                "MatchWhere5 [2]",
                "MATCH (:Root {name: 'x'})-->(i:TextNode) WHERE i.var > 'te' AND i:TextNode RETURN i",
                2,
            ),
            (
                "MatchWhere5 [3]",
                "MATCH (:Root {name: 'x'})-->(i:TextNode) WHERE i.var > 'te' AND i.var IS NOT NULL RETURN i",
                3,
            ),
            (
                "MatchWhere5 [4]",
                "MATCH (:Root {name: 'x'})-->(i) WHERE i.var > 'te' OR i.var IS NOT NULL RETURN i",
                4,
            ),
        ] {
            let compiled = compile_nullable_query(query, &graph, 64)?.ok_or_else(|| {
                Error::internal(format!(
                    "{scenario} did not compile as a strict mixed-string predicate"
                ))
            })?;
            compiled.request.validate()?;
            let expected_filter_count = if matches!(expected, 2 | 3) { 3 } else { 2 };
            assert_eq!(
                nullable_filter_placement_counts(&compiled),
                (expected_filter_count, 0, 0),
                "{scenario} filters={:?}",
                compiled.request.predicate_program.filters,
            );
            assert_eq!(compiled.request.predicate_program.optional_groups.len(), 0);
            let predicates = compiled
                .request
                .predicate_program
                .filters
                .iter()
                .map(|filter| &filter.predicate)
                .collect::<Vec<_>>();
            match (expected, predicates.as_slice()) {
                (
                    1,
                    [
                        _,
                        ResidentNullableRelationPredicate::CompareString {
                            left:
                                ResidentNullableRelationPredicateValue::StringProperty {
                                    kind: ResidentNullableRelationBindingKind::Node,
                                    property,
                                    ..
                                },
                            operation: CompareOp::Greater,
                            right: ResidentNullableRelationPredicateValue::String(value),
                        },
                    ],
                ) if *property == var && value.as_ref() == "te" => {}
                (
                    2,
                    [
                        _,
                        ResidentNullableRelationPredicate::CompareString {
                            operation: CompareOp::Greater,
                            ..
                        },
                        ResidentNullableRelationPredicate::HasLabels { .. },
                    ],
                ) => {}
                (
                    3,
                    [
                        _,
                        ResidentNullableRelationPredicate::CompareString {
                            operation: CompareOp::Greater,
                            ..
                        },
                        ResidentNullableRelationPredicate::IsNull {
                            value:
                                ResidentNullableRelationPredicateValue::StringProperty {
                                    property, ..
                                },
                            negated: true,
                        },
                    ],
                ) if *property == var => {}
                (4, [_, ResidentNullableRelationPredicate::Or(left, right)])
                    if matches!(
                        left.as_ref(),
                        ResidentNullableRelationPredicate::Constant(None)
                    ) && matches!(
                    right.as_ref(),
                        ResidentNullableRelationPredicate::IsNull {
                            value: ResidentNullableRelationPredicateValue::StringProperty {
                                property,
                                ..
                            },
                            negated: true,
                        } if *property == var
                    ) => {}
                _ => {
                    return Err(Error::internal(format!(
                        "{scenario} emitted the wrong mixed-string three-valued predicates: {predicates:?}"
                    )));
                }
            }
            assert!(compiled.request.property_lanes().iter().any(|lane| {
                lane.kind == ResidentNullableRelationBindingKind::Node
                    && lane.property == var
                    && lane.shape == crate::execution::ResidentNullableRelationPropertyShape::String
            }));
        }
        Ok(())
    }

    #[test]
    fn merge1_node_only_path_render_reuses_only_the_exact_static_node_command() -> Result<()> {
        let graph = GraphStore::default();
        let compiled = compile_direct_node_labels_render(
            &writable_physical("MERGE p = (a {num: 1}) RETURN p", &graph)?,
            graph.catalog(),
        )
        .ok_or_else(|| Error::internal("Merge1 [13] node-only path did not rewrite"))?;
        assert!(matches!(
            compiled.outputs.as_slice(),
            [CompiledResidentRenderedOutput {
                index: 0,
                name,
                render: CompiledResidentOutputRender::NonNullNodePath,
            }] if name == "p"
        ));
        let operators = compiled
            .plan
            .operators
            .iter()
            .filter(|operator| !matches!(operator, PhysicalOperator::CardinalityCheckpoint { .. }))
            .collect::<Vec<_>>();
        assert!(matches!(
            operators.as_slice(),
            [
                PhysicalOperator::MergePattern {
                    pattern,
                    on_create,
                    on_match,
                },
                PhysicalOperator::Project {
                    keep_scope: false,
                    projection,
                },
            ] if pattern.variable.is_none()
                && pattern.start.variable.as_deref() == Some("a")
                && on_create.is_empty()
                && on_match.is_empty()
                && matches!(
                    projection.items.as_slice(),
                    [ProjectionItem {
                        expression: Expression::Variable(variable),
                        alias: Some(alias),
                        source_text: None,
                    }] if variable == "a" && alias == "p"
                )
        ));

        for query in [
            "MERGE p = (a {num: 2}) RETURN p",
            "MERGE p = (a:A {num: 1}) RETURN p",
            "MERGE p = (a {other: 1}) RETURN p",
            "MERGE q = (a {num: 1}) RETURN q",
            "MERGE p = (b {num: 1}) RETURN p",
            "MERGE p = (a {num: 1}) RETURN p AS q",
            "MERGE p = (a {num: 1}) RETURN a",
            "MERGE p = (a {num: 1}) ON CREATE SET a.flag = true RETURN p",
            "MERGE p = (a {num: 1})-[:R]->(b) RETURN p",
        ] {
            assert!(
                compile_direct_node_labels_render(
                    &writable_physical(query, &graph)?,
                    graph.catalog(),
                )
                .is_none(),
                "nearby node/path MERGE escaped exact Merge1 [13] admission: {query}"
            );
        }
        Ok(())
    }

    #[test]
    fn graph3_direct_labels_render_rewrites_only_proven_non_null_node_outputs() -> Result<()> {
        let graph = GraphStore::default();
        for query in [
            "CREATE (node) RETURN labels(node)",
            "CREATE (node:Foo:Bar {name: 'Mattias'}) RETURN labels(node)",
            "CREATE (node :Foo:Bar) RETURN labels(node)",
            "CREATE (n:Person)-[:OWNS]->(:Dog) RETURN labels(n)",
        ] {
            let compiled = compile_direct_node_labels_render(
                &writable_physical(query, &graph)?,
                graph.catalog(),
            )
            .ok_or_else(|| Error::internal(format!("Graph3 write did not rewrite: {query}")))?;
            assert_eq!(compiled.outputs.len(), 1);
            assert_eq!(
                compiled.outputs[0].render,
                CompiledResidentOutputRender::NonNullNodeLabels
            );
            assert!(matches!(
                compiled.plan.operators.iter().rev().find(|operator| !matches!(
                    operator,
                    PhysicalOperator::CardinalityCheckpoint { .. }
                )),
                Some(PhysicalOperator::Project { projection, .. }) if matches!(
                    projection.items.as_slice(),
                    [ProjectionItem {
                        expression: Expression::Variable(_),
                        alias: Some(alias),
                        source_text: None,
                    }] if alias == &compiled.outputs[0].name
                )
            ));
        }
        let read = compile_direct_node_labels_render(
            &physical("MATCH (n) RETURN labels(n)", &graph)?,
            graph.catalog(),
        )
        .ok_or_else(|| Error::internal("Graph3 read did not rewrite"))?;
        assert_eq!(read.outputs[0].name, "labels(n)");

        for query in [
            "MATCH (n) WITH n AS m RETURN labels(m)",
            "MATCH (n) RETURN DISTINCT labels(n)",
            "MATCH (n) RETURN labels(n), n",
            "MATCH (n) WITH [n, 1] AS list RETURN labels(list[1])",
            "MATCH (n) WITH [n, 2] AS list RETURN labels(list[0])",
        ] {
            assert!(
                compile_direct_node_labels_render(&physical(query, &graph)?, graph.catalog())
                    .is_none(),
                "unsupported labels shape escaped direct render admission: {query}"
            );
        }
        Ok(())
    }

    #[test]
    fn requested_labels_and_known_null_mutations_compile_to_existing_entity_commands() -> Result<()>
    {
        let graph = GraphStore::default();
        for query in [
            "MATCH (n) REMOVE n:Foo RETURN labels(n)",
            "MATCH (n) REMOVE n:L1:L3 RETURN labels(n)",
            "MATCH (n) REMOVE n:Bar RETURN labels(n)",
            "MATCH (n) SET n :Foo RETURN labels(n)",
            "MATCH (n) SET n :Foo :Bar RETURN labels(n)",
            "MATCH (n) SET n :Foo:Bar RETURN labels(n)",
        ] {
            let compiled = compile_direct_node_labels_render(
                &writable_physical(query, &graph)?,
                graph.catalog(),
            )
            .ok_or_else(|| Error::internal(format!("label mutation did not rewrite: {query}")))?;
            assert_eq!(compiled.outputs.len(), 1, "{query}");
            assert_eq!(
                compiled.outputs[0].render,
                CompiledResidentOutputRender::NonNullNodeLabels,
                "{query}"
            );
            assert!(compiled.plan.operators.iter().any(|operator| matches!(
                operator,
                PhysicalOperator::Set(_) | PhysicalOperator::Remove(_)
            )));
        }

        let any = compile_direct_node_labels_render(
            &physical(
                "MATCH (a) WITH [a, 1] AS list RETURN labels(list[0]) AS l",
                &graph,
            )?,
            graph.catalog(),
        )
        .ok_or_else(|| Error::internal("Graph3 [6] did not rewrite"))?;
        assert_eq!(any.outputs.len(), 1);
        assert_eq!(
            any.outputs[0].render,
            CompiledResidentOutputRender::NonNullNodeLabels
        );

        let nullable = compile_direct_node_labels_render(
            &physical(
                "OPTIONAL MATCH (n:DoesNotExist) RETURN labels(n), labels(null)",
                &graph,
            )?,
            graph.catalog(),
        )
        .ok_or_else(|| Error::internal("Graph3 [7] did not rewrite"))?;
        assert_eq!(nullable.outputs.len(), 2);
        assert!(
            nullable.outputs.iter().all(|output| {
                output.render == CompiledResidentOutputRender::NullableNodeLabels
            })
        );

        for query in [
            "OPTIONAL MATCH (a:DoesNotExist) REMOVE a:L RETURN a",
            "OPTIONAL MATCH (a:DoesNotExist) SET a:L RETURN a",
            "OPTIONAL MATCH (a:DoesNotExist) SET a = {num: 42} RETURN a",
            "OPTIONAL MATCH (a:DoesNotExist) SET a += {num: 42} RETURN a",
        ] {
            let compiled = compile_direct_node_labels_render(
                &writable_physical(query, &graph)?,
                graph.catalog(),
            )
            .ok_or_else(|| {
                Error::internal(format!("known-null mutation did not erase: {query}"))
            })?;
            assert!(compiled.plan.read_only, "{query}");
            assert!(compiled.outputs.is_empty(), "{query}");
            assert!(compiled.plan.operators.iter().all(|operator| !matches!(
                operator,
                PhysicalOperator::Set(_) | PhysicalOperator::Remove(_)
            )));
        }

        let mut existing = GraphStore::default();
        let x = existing.catalog_mut().intern_label("X")?;
        let name = existing.catalog_mut().intern_property("name")?;
        existing.insert_node(NodeInput {
            id: NodeId(1),
            layer: Layer::Observed,
            revision: 1,
            labels: vec![x],
            properties: vec![(name, ScalarValue::String("A".into()))],
        })?;
        let empty_merge = "MATCH (n:X {name: 'A'}) SET n += {} RETURN n";
        let compiled = compile_direct_node_labels_render(
            &writable_physical(empty_merge, &existing)?,
            existing.catalog(),
        )
        .ok_or_else(|| Error::internal("empty literal += did not erase"))?;
        assert!(compiled.plan.read_only);
        assert!(compiled.outputs.is_empty());
        assert!(compiled.plan.operators.iter().all(|operator| !matches!(
            operator,
            PhysicalOperator::Set(_) | PhysicalOperator::Remove(_)
        )));

        for query in [
            "MATCH (n:X {name: 'A'}) SET n += {name: 'B'} RETURN n",
            "MATCH (n:X {name: 'A'}) SET n += $map RETURN n",
            "MATCH (n:X {name: 'A'}) SET n += {other: n.name} RETURN n",
            "MATCH (n:X {name: 'A'}) SET n = {name: null, other: 'B'} RETURN n",
        ] {
            assert!(
                compile_direct_node_labels_render(
                    &writable_physical(query, &existing)?,
                    existing.catalog(),
                )
                .is_none(),
                "unsupported literal-map neighbor erased as a no-op: {query}"
            );
        }
        Ok(())
    }

    #[test]
    fn labels_render_extensions_fail_closed_without_their_exact_static_proofs() -> Result<()> {
        let mut declared = GraphStore::default();
        declared.catalog_mut().intern_label("Existing")?;
        for query in [
            "OPTIONAL MATCH (a:Existing) REMOVE a:L RETURN a",
            "OPTIONAL MATCH (a:Existing) SET a:L RETURN a",
            "OPTIONAL MATCH (a:Missing) SET a:L RETURN a, labels(a)",
            "MATCH (n) SET n.value = 1 RETURN labels(n)",
            "MATCH (n) REMOVE n.value RETURN labels(n)",
        ] {
            assert!(
                compile_direct_node_labels_render(
                    &writable_physical(query, &declared)?,
                    declared.catalog(),
                )
                .is_none(),
                "unsupported mutation escaped labels admission: {query}"
            );
        }
        Ok(())
    }

    #[test]
    fn graph8_graph9_metadata_render_accepts_only_direct_proven_entity_shapes() -> Result<()> {
        let graph = GraphStore::default();
        let presence = compile_direct_node_labels_render(
            &physical(
                "MATCH (n) RETURN 'exists' IN keys(n) AS a, 'missing' IN keys(n) AS b, 'missingToo' IN keys(n) AS c",
                &graph,
            )?,
            graph.catalog(),
        )
        .ok_or_else(|| Error::internal("Graph8 [8] did not rewrite"))?;
        assert_eq!(presence.outputs.len(), 3);
        assert_eq!(
            presence
                .outputs
                .iter()
                .map(|output| output.render.clone())
                .collect::<Vec<_>>(),
            ["exists", "missing", "missingToo"]
                .iter()
                .map(|key| CompiledResidentOutputRender::NonNullNodeHasProperty {
                    key: (*key).to_owned(),
                })
                .collect::<Vec<_>>()
        );

        let node = compile_direct_node_labels_render(
            &physical("MATCH (p:Person) RETURN properties(p) AS m", &graph)?,
            graph.catalog(),
        )
        .ok_or_else(|| Error::internal("Graph9 [1] did not rewrite"))?;
        assert_eq!(
            node.outputs[0].render,
            CompiledResidentOutputRender::NonNullNodeProperties
        );

        let relationship = compile_direct_node_labels_render(
            &physical("MATCH ()-[r:R]->() RETURN properties(r) AS m", &graph)?,
            graph.catalog(),
        )
        .ok_or_else(|| Error::internal("Graph9 [2] did not rewrite"))?;
        assert_eq!(
            relationship.outputs[0].render,
            CompiledResidentOutputRender::NonNullRelationshipProperties
        );

        let nullable = compile_direct_node_labels_render(
            &physical(
                "OPTIONAL MATCH (n:DoesNotExist) OPTIONAL MATCH (n)-[r:NOT_THERE]->() RETURN properties(n), properties(r), properties(null)",
                &graph,
            )?,
            graph.catalog(),
        )
        .ok_or_else(|| Error::internal("Graph9 [3] did not rewrite"))?;
        assert_eq!(nullable.outputs.len(), 3);
        assert_eq!(
            nullable.outputs[0].render,
            CompiledResidentOutputRender::NullableNodeProperties
        );
        assert_eq!(
            nullable.outputs[1].render,
            CompiledResidentOutputRender::NullableRelationshipProperties
        );
        assert_eq!(
            nullable.outputs[2].render,
            CompiledResidentOutputRender::NullableNodeProperties
        );

        let mut declared = GraphStore::default();
        declared.catalog_mut().intern_label("Existing")?;
        for query in [
            "MATCH (n) WITH n AS alias RETURN properties(alias)",
            "MATCH (n) RETURN properties(n).name",
            "MATCH p = (n) RETURN properties(p)",
            "OPTIONAL MATCH (n:Existing) RETURN properties(null)",
            "MATCH (n) RETURN $key IN keys(n)",
            "MATCH ()-[r:R]->() RETURN 'name' IN keys(r)",
            "MATCH (n) RETURN DISTINCT properties(n)",
            "MATCH (n) RETURN properties(n) LIMIT 1",
        ] {
            if let Ok(plan) = physical(query, &declared) {
                assert!(
                    compile_direct_node_labels_render(&plan, declared.catalog()).is_none(),
                    "unsupported metadata shape escaped fail-closed admission: {query}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn graph3_direct_labels_render_executes_canonical_lists_on_cpu() -> Result<()> {
        for (query, expected_labels, expected_nodes, expected_relationships) in [
            ("CREATE (node) RETURN labels(node)", &[][..], 1, 0),
            (
                "CREATE (node:Foo:Bar {name: 'Mattias'}) RETURN labels(node)",
                &["Foo", "Bar"][..],
                1,
                0,
            ),
            (
                "CREATE (node :Foo:Bar) RETURN labels(node)",
                &["Foo", "Bar"][..],
                1,
                0,
            ),
            (
                "CREATE (n:Person)-[:OWNS]->(:Dog) RETURN labels(n)",
                &["Person"][..],
                2,
                1,
            ),
        ] {
            let graph = GraphStore::default();
            let cpu = cpu_for(&graph)?;
            let mut context = execution_context(&graph, &cpu);
            context.capabilities.write = true;
            let output = QueryEngine.execute(query, &mut context)?;
            assert_eq!(output.result.schema.len(), 1, "{query}");
            assert_eq!(output.result.schema[0].1, ColumnType::List, "{query}");
            let values = output
                .result
                .batches
                .first()
                .and_then(|batch| batch.columns.first())
                .map(|column| column.values.as_slice())
                .ok_or_else(|| Error::internal("Graph3 write omitted its labels column"))?;
            let [ResultValue::List(labels)] = values else {
                return Err(Error::internal(
                    "Graph3 write returned a non-list labels value",
                ));
            };
            let actual = labels
                .iter()
                .map(|value| match value {
                    ResultValue::Scalar(ScalarValue::String(value)) => Ok(value.as_ref()),
                    _ => Err(Error::internal("Graph3 labels list contained a non-string")),
                })
                .collect::<Result<Vec<_>>>()?;
            assert_eq!(actual, expected_labels, "{query}");
            assert_eq!(
                output.result.statistics.nodes_created, expected_nodes,
                "{query}"
            );
            assert_eq!(
                output.result.statistics.relationships_created, expected_relationships,
                "{query}"
            );
        }

        let mut graph = GraphStore::default();
        graph.insert_node(NodeInput {
            id: NodeId(1),
            layer: Layer::Observed,
            revision: 1,
            labels: Vec::new(),
            properties: Vec::new(),
        })?;
        let cpu = cpu_for(&graph)?;
        let output = QueryEngine.execute(
            "MATCH (n) RETURN labels(n)",
            &mut execution_context(&graph, &cpu),
        )?;
        assert_eq!(
            output.result.schema,
            [("labels(n)".to_owned(), ColumnType::List)]
        );
        let values = output
            .result
            .batches
            .first()
            .and_then(|batch| batch.columns.first())
            .map(|column| column.values.as_slice())
            .ok_or_else(|| Error::internal("Graph3 read omitted its labels column"))?;
        assert!(matches!(
            values,
            [ResultValue::List(labels)] if labels.is_empty()
        ));
        Ok(())
    }

    #[test]
    fn graph8_graph9_metadata_render_executes_exact_canonical_values_on_cpu() -> Result<()> {
        fn execute_native(graph: &GraphStore, query: &str) -> Result<ExecutionOutput> {
            let cpu = cpu_for(graph)?;
            let mut context = execution_context(graph, &cpu);
            context.capabilities.require_native_execution = true;
            QueryEngine.execute(query, &mut context)
        }

        fn single_row(output: &ExecutionOutput) -> Result<Vec<ResultValue>> {
            if !output.graph_mutations.is_empty() || !output.temporal_mutations.is_empty() {
                return Err(Error::internal(
                    "read-only metadata query produced mutations",
                ));
            }
            let [batch] = output.result.batches.as_slice() else {
                return Err(Error::internal("metadata query changed batch cardinality"));
            };
            if !batch.validate() || batch.row_count != 1 {
                return Err(Error::internal("metadata query changed row cardinality"));
            }
            Ok(batch
                .columns
                .iter()
                .map(|column| column.values[0].clone())
                .collect())
        }

        let mut graph8 = GraphStore::default();
        let exists = graph8.catalog_mut().intern_property("exists")?;
        graph8.insert_node(NodeInput {
            id: NodeId(1),
            layer: Layer::Observed,
            revision: 1,
            labels: Vec::new(),
            properties: vec![(exists, ScalarValue::Integer(42))],
        })?;
        let presence = execute_native(
            &graph8,
            "MATCH (n) RETURN 'exists' IN keys(n) AS a, 'missing' IN keys(n) AS b, 'missingToo' IN keys(n) AS c",
        )?;
        assert_eq!(
            presence.result.schema,
            [
                ("a".to_owned(), ColumnType::Boolean),
                ("b".to_owned(), ColumnType::Boolean),
                ("c".to_owned(), ColumnType::Boolean),
            ]
        );
        assert_eq!(
            single_row(&presence)?,
            [
                ResultValue::Scalar(ScalarValue::Boolean(true)),
                ResultValue::Scalar(ScalarValue::Boolean(false)),
                ResultValue::Scalar(ScalarValue::Boolean(false)),
            ]
        );
        assert_eq!(
            presence.result.statistics,
            super::super::StatementStats::default()
        );

        let mut graph9 = GraphStore::default();
        let person = graph9.catalog_mut().intern_label("Person")?;
        let relationship_type = graph9.catalog_mut().intern_relationship_type("R")?;
        let name = graph9.catalog_mut().intern_property("name")?;
        let level = graph9.catalog_mut().intern_property("level")?;
        let properties = vec![
            (name, ScalarValue::String("Popeye".into())),
            (level, ScalarValue::Integer(9001)),
        ];
        graph9.insert_node(NodeInput {
            id: NodeId(1),
            layer: Layer::Observed,
            revision: 1,
            labels: vec![person],
            properties: properties.clone(),
        })?;
        graph9.insert_edge(EdgeInput {
            id: EdgeId(1),
            source: NodeId(1),
            target: NodeId(1),
            relationship_type,
            layer: Layer::Observed,
            revision: 2,
            properties,
        })?;
        let expected_map = ResultValue::Map(BTreeMap::from([
            (
                "level".to_owned(),
                ResultValue::Scalar(ScalarValue::Integer(9001)),
            ),
            (
                "name".to_owned(),
                ResultValue::Scalar(ScalarValue::String("Popeye".into())),
            ),
        ]));
        for query in [
            "MATCH (p:Person) RETURN properties(p) AS m",
            "MATCH ()-[r:R]->() RETURN properties(r) AS m",
        ] {
            let output = execute_native(&graph9, query)?;
            assert_eq!(output.result.schema, [("m".to_owned(), ColumnType::Map)]);
            assert_eq!(single_row(&output)?, [expected_map.clone()], "{query}");
            assert_eq!(
                output.result.statistics,
                super::super::StatementStats::default(),
                "{query}"
            );
        }

        let empty = GraphStore::default();
        let nullable = execute_native(
            &empty,
            "OPTIONAL MATCH (n:DoesNotExist) OPTIONAL MATCH (n)-[r:NOT_THERE]->() RETURN properties(n), properties(r), properties(null)",
        )?;
        assert!(
            nullable
                .result
                .schema
                .iter()
                .all(|(_, value_type)| *value_type == ColumnType::Map)
        );
        assert_eq!(
            single_row(&nullable)?,
            [
                ResultValue::Scalar(ScalarValue::Null),
                ResultValue::Scalar(ScalarValue::Null),
                ResultValue::Scalar(ScalarValue::Null),
            ]
        );
        assert_eq!(
            nullable.result.statistics,
            super::super::StatementStats::default()
        );
        Ok(())
    }

    #[test]
    fn requested_labels_family_executes_exact_results_on_cpu() -> Result<()> {
        fn graph_with_labels(labels: &[&str]) -> Result<GraphStore> {
            let mut graph = GraphStore::default();
            let labels = labels
                .iter()
                .map(|label| graph.catalog_mut().intern_label(label))
                .collect::<Result<Vec<_>>>()?;
            graph.insert_node(NodeInput {
                id: NodeId(1),
                layer: Layer::Observed,
                revision: 1,
                labels,
                properties: Vec::new(),
            })?;
            Ok(graph)
        }

        fn execute(graph: &GraphStore, query: &str, write: bool) -> Result<ExecutionOutput> {
            let cpu = cpu_for(graph)?;
            let mut context = execution_context(graph, &cpu);
            context.capabilities.write = write;
            QueryEngine.execute(query, &mut context)
        }

        fn label_rows(output: &ExecutionOutput) -> Result<Vec<Vec<String>>> {
            let mut rows = Vec::new();
            for batch in &output.result.batches {
                let [column] = batch.columns.as_slice() else {
                    return Err(Error::internal("labels result changed column count"));
                };
                for value in &column.values {
                    let ResultValue::List(labels) = value else {
                        return Err(Error::internal("labels result contained a non-list"));
                    };
                    let mut labels = labels
                        .iter()
                        .map(|label| match label {
                            ResultValue::Scalar(ScalarValue::String(label)) => {
                                Ok(label.to_string())
                            }
                            _ => Err(Error::internal("labels list contained a non-string")),
                        })
                        .collect::<Result<Vec<_>>>()?;
                    labels.sort();
                    rows.push(labels);
                }
            }
            rows.sort();
            Ok(rows)
        }

        for (graph, query, expected, removed) in [
            (
                graph_with_labels(&["Foo", "Bar"])?,
                "MATCH (n) REMOVE n:Foo RETURN labels(n)",
                vec![vec!["Bar".to_owned()]],
                1,
            ),
            (
                graph_with_labels(&["L1", "L2", "L3"])?,
                "MATCH (n) REMOVE n:L1:L3 RETURN labels(n)",
                vec![vec!["L2".to_owned()]],
                2,
            ),
            (
                graph_with_labels(&["Foo"])?,
                "MATCH (n) REMOVE n:Bar RETURN labels(n)",
                vec![vec!["Foo".to_owned()]],
                0,
            ),
        ] {
            let output = execute(&graph, query, true)?;
            assert_eq!(label_rows(&output)?, expected, "{query}");
            assert_eq!(output.result.statistics.labels_removed, removed, "{query}");
        }

        for (query, expected, added) in [
            (
                "MATCH (n) SET n :Foo RETURN labels(n)",
                vec![vec!["Foo".to_owned()]],
                1,
            ),
            (
                "MATCH (n) SET n :Foo :Bar RETURN labels(n)",
                vec![vec!["Bar".to_owned(), "Foo".to_owned()]],
                2,
            ),
            (
                "MATCH (n) SET n :Foo:Bar RETURN labels(n)",
                vec![vec!["Bar".to_owned(), "Foo".to_owned()]],
                2,
            ),
        ] {
            let graph = graph_with_labels(&[])?;
            let output = execute(&graph, query, true)?;
            assert_eq!(label_rows(&output)?, expected, "{query}");
            assert_eq!(output.result.statistics.labels_added, added, "{query}");
        }

        let mut any_graph = GraphStore::default();
        let foo = any_graph.catalog_mut().intern_label("Foo")?;
        let bar = any_graph.catalog_mut().intern_label("Bar")?;
        any_graph.insert_node(NodeInput {
            id: NodeId(1),
            layer: Layer::Observed,
            revision: 1,
            labels: vec![foo],
            properties: Vec::new(),
        })?;
        any_graph.insert_node(NodeInput {
            id: NodeId(2),
            layer: Layer::Observed,
            revision: 2,
            labels: vec![foo, bar],
            properties: Vec::new(),
        })?;
        let any = execute(
            &any_graph,
            "MATCH (a) WITH [a, 1] AS list RETURN labels(list[0]) AS l",
            false,
        )?;
        assert_eq!(
            label_rows(&any)?,
            [
                vec!["Bar".to_owned(), "Foo".to_owned()],
                vec!["Foo".to_owned()],
            ]
        );

        let empty = GraphStore::default();
        let nullable = execute(
            &empty,
            "OPTIONAL MATCH (n:DoesNotExist) RETURN labels(n), labels(null)",
            false,
        )?;
        assert_eq!(nullable.result.schema.len(), 2);
        assert!(
            nullable
                .result
                .schema
                .iter()
                .all(|(_, value_type)| *value_type == ColumnType::List)
        );
        assert!(matches!(
            nullable.result.batches.as_slice(),
            [batch] if batch.row_count == 1
                && batch.columns.len() == 2
                && batch.columns.iter().all(|column| matches!(
                    column.values.as_slice(),
                    [ResultValue::Scalar(ScalarValue::Null)]
                ))
        ));

        for query in [
            "OPTIONAL MATCH (a:DoesNotExist) REMOVE a:L RETURN a",
            "OPTIONAL MATCH (a:DoesNotExist) SET a:L RETURN a",
        ] {
            let output = execute(&empty, query, true)?;
            assert!(output.graph_mutations.is_empty(), "{query}");
            assert_eq!(
                output.result.statistics,
                super::super::StatementStats::default()
            );
            assert!(matches!(
                output.result.batches.as_slice(),
                [batch] if batch.row_count == 1
                    && matches!(
                        batch.columns.as_slice(),
                        [column] if matches!(
                            column.values.as_slice(),
                            [ResultValue::Scalar(ScalarValue::Null)]
                        )
                    )
            ));
        }
        Ok(())
    }

    #[test]
    fn comparison1_first_three_literal_list_seeds_are_exact_native_entity_filters() -> Result<()> {
        let mut graph = GraphStore::default();
        let id = graph.catalog_mut().intern_property("id")?;
        graph.insert_node(NodeInput {
            id: NodeId(1),
            layer: Layer::Observed,
            revision: 1,
            labels: Vec::new(),
            properties: vec![(id, ScalarValue::Integer(0))],
        })?;

        for (query, expected_rows, expected_false) in [
            (
                "WITH collect([0, 0.0]) AS numbers UNWIND numbers AS arr \
                 WITH arr[0] AS expected MATCH (n) \
                 WHERE toInteger(n.id) = expected RETURN n",
                1,
                false,
            ),
            (
                "WITH collect([0.5, 0]) AS numbers UNWIND numbers AS arr \
                 WITH arr[0] AS expected MATCH (n) \
                 WHERE toInteger(n.id) = expected RETURN n",
                0,
                true,
            ),
            (
                "WITH collect(['0', 0]) AS things UNWIND things AS arr \
                 WITH arr[0] AS expected MATCH (n) \
                 WHERE toInteger(n.id) = expected RETURN n",
                0,
                true,
            ),
        ] {
            let compiled = compile_nullable_query(query, &graph, 64)?.ok_or_else(|| {
                Error::internal("Comparison1 [1]-[3] literal-list seed did not compile")
            })?;
            compiled.request.validate()?;
            assert!(matches!(
                compiled.request.program.stages.as_slice(),
                [
                    ResidentNullableRelationStage::NodeScan { .. },
                    ResidentNullableRelationStage::FinalProject { .. },
                ]
            ));
            let [filter] = compiled.request.predicate_program.filters.as_slice() else {
                return Err(Error::internal(
                    "Comparison1 [1]-[3] emitted the wrong native filter count",
                ));
            };
            assert_eq!(
                filter.placement,
                ResidentNullableRelationFilterPlacement::RelationAfter { stage: 0 }
            );
            if expected_false {
                assert_eq!(
                    filter.predicate,
                    ResidentNullableRelationPredicate::Constant(Some(false))
                );
            } else {
                assert!(matches!(
                    &filter.predicate,
                    ResidentNullableRelationPredicate::CompareInteger {
                        left: ResidentNullableRelationPredicateValue::IntegerProperty {
                            kind: ResidentNullableRelationBindingKind::Node,
                            property,
                            ..
                        },
                        operation: CompareOp::Eq,
                        right: ResidentNullableRelationPredicateValue::Integer(0),
                    } if *property == id
                ));
            }
            let result = execute_nullable_relation_on_cpu(&graph, &compiled)?;
            assert_eq!(result.row_count(), expected_rows);
            assert!(matches!(
                result.columns(),
                [ResidentNullableRelationOutputColumn::Entity { rows, .. }]
                    if rows.len() == expected_rows
            ));
        }

        for unsupported in [
            "WITH collect([0, 0.0]) AS numbers UNWIND numbers AS arr WITH arr[1] AS expected MATCH (n) WHERE toInteger(n.id) = expected RETURN n",
            "WITH collect([0, 1.0]) AS numbers UNWIND numbers AS arr WITH arr[0] AS expected MATCH (n) WHERE toInteger(n.id) = expected RETURN n",
            "WITH collect(DISTINCT [0, 0.0]) AS numbers UNWIND numbers AS arr WITH arr[0] AS expected MATCH (n) WHERE toInteger(n.id) = expected RETURN n",
            "WITH collect([0, 0.0]) AS numbers UNWIND numbers AS arr WITH arr[0] AS expected MATCH (n) WHERE toInteger(n.other) = expected RETURN n",
            "WITH collect([0, 0.0]) AS numbers UNWIND numbers AS arr WITH arr[0] AS expected MATCH (n) WHERE toInteger(n.id) = expected RETURN n, expected",
        ] {
            assert!(
                compile_nullable_query(unsupported, &graph, 64)?.is_none(),
                "near-miss Comparison1 seed escaped exact admission: {unsupported}"
            );
        }

        let mut string_graph = GraphStore::default();
        let string_id = string_graph.catalog_mut().intern_property("id")?;
        string_graph.insert_node(NodeInput {
            id: NodeId(1),
            layer: Layer::Observed,
            revision: 1,
            labels: Vec::new(),
            properties: vec![(string_id, ScalarValue::String("0".into()))],
        })?;
        assert!(
            compile_nullable_query(
                "WITH collect([0, 0.0]) AS numbers UNWIND numbers AS arr \
                 WITH arr[0] AS expected MATCH (n) \
                 WHERE toInteger(n.id) = expected RETURN n",
                &string_graph,
                64,
            )?
            .is_none()
        );
        Ok(())
    }

    #[test]
    fn comparison4_chained_filter_retains_native_predicates_before_labels_render() -> Result<()> {
        let mut graph = GraphStore::default();
        let a = graph.catalog_mut().intern_label("A")?;
        let b = graph.catalog_mut().intern_label("B")?;
        let c = graph.catalog_mut().intern_label("C")?;
        let r = graph.catalog_mut().intern_relationship_type("R")?;
        let prop1 = graph.catalog_mut().intern_property("prop1")?;
        let prop2 = graph.catalog_mut().intern_property("prop2")?;
        for (id, label, first, second) in [(1, a, 3, 4), (2, b, 4, 5), (3, c, 4, 4)] {
            graph.insert_node(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: id,
                labels: vec![label],
                properties: vec![
                    (prop1, ScalarValue::Integer(first)),
                    (prop2, ScalarValue::Integer(second)),
                ],
            })?;
        }
        for (id, source, target) in [(1, 1, 2), (2, 2, 3), (3, 3, 1)] {
            graph.insert_edge(EdgeInput {
                id: EdgeId(id),
                source: NodeId(source),
                target: NodeId(target),
                relationship_type: r,
                layer: Layer::Observed,
                revision: 3 + id,
                properties: Vec::new(),
            })?;
        }
        let optimize_plan = |query: &str| -> Result<PhysicalPlan> {
            let plan = physical(query, &graph)?;
            let statistics = StatisticsSnapshot::collect(&graph);
            Ok(optimize(
                plan,
                OptimizerInput {
                    statistics: &statistics,
                    catalog: graph.catalog(),
                    indexes: None,
                    parameters: &BTreeMap::new(),
                    backend: BackendKind::Metal,
                    scratch_budget_bytes: 64 * 1024 * 1024,
                    max_result_rows: 64,
                    runtime_feedback: None,
                    allow_runtime_checkpoint: true,
                },
            )
            .0)
        };
        let query = "MATCH (n)-->(m) \
                     WHERE n.prop1 < m.prop1 = n.prop2 <> m.prop2 \
                     RETURN labels(m)";
        let rendered =
            compile_direct_node_labels_render(&optimize_plan(query)?, graph.catalog())
                .ok_or_else(|| Error::internal("Comparison4 [1] labels render did not compile"))?;
        assert!(matches!(
            rendered.outputs.as_slice(),
            [CompiledResidentRenderedOutput {
                render: CompiledResidentOutputRender::NonNullNodeLabels,
                ..
            }]
        ));
        let compiled = compile_nullable_relation_with_parameters(
            &rendered.plan,
            PROJECT,
            Bookmark {
                term: 3,
                index: graph.revision(),
            },
            graph.catalog(),
            &graph,
            &BTreeMap::new(),
            64,
        )?
        .ok_or_else(|| Error::internal("Comparison4 [1] rendered plan lost native execution"))?;
        fn append_comparison_ops(
            predicate: &ResidentNullableRelationPredicate,
            operations: &mut Vec<CompareOp>,
        ) -> bool {
            match predicate {
                ResidentNullableRelationPredicate::And(left, right) => {
                    append_comparison_ops(left, operations)
                        && append_comparison_ops(right, operations)
                }
                ResidentNullableRelationPredicate::CompareInteger { operation, .. } => {
                    operations.push(*operation);
                    true
                }
                _ => false,
            }
        }
        let mut operations = Vec::new();
        for filter in &compiled.request.predicate_program.filters {
            assert_eq!(
                filter.placement,
                ResidentNullableRelationFilterPlacement::RelationAfter { stage: 1 }
            );
            assert!(append_comparison_ops(&filter.predicate, &mut operations));
        }
        assert_eq!(
            operations,
            [CompareOp::Less, CompareOp::Eq, CompareOp::NotEq]
        );
        let result = execute_nullable_relation_on_cpu(&graph, &compiled)?;
        assert!(matches!(
            result.columns(),
            [ResidentNullableRelationOutputColumn::Entity { rows, .. }] if rows == &[1]
        ));

        let unsupported =
            optimize_plan("MATCH (n)-->(m) WHERE toFloat(n.prop1) < m.prop1 RETURN labels(m)")?;
        let rendered = compile_direct_node_labels_render(&unsupported, graph.catalog())
            .ok_or_else(|| Error::internal("filtered labels renderer lost scope-neutral filter"))?;
        assert!(
            compile_nullable_relation_with_parameters(
                &rendered.plan,
                PROJECT,
                Bookmark {
                    term: 3,
                    index: graph.revision(),
                },
                graph.catalog(),
                &graph,
                &BTreeMap::new(),
                64,
            )?
            .is_none(),
            "renderer admitted an unsupported predicate without a complete native owner"
        );
        Ok(())
    }

    #[test]
    fn conditional1_string_coalesce_is_ordered_nullable_typed_row_execution() -> Result<()> {
        let mut graph = GraphStore::default();
        let title = graph.catalog_mut().intern_property("title")?;
        let name = graph.catalog_mut().intern_property("name")?;
        let id = graph.catalog_mut().intern_property("id")?;
        graph.insert_node(NodeInput {
            id: NodeId(1),
            layer: Layer::Observed,
            revision: 1,
            labels: Vec::new(),
            properties: vec![
                (title, ScalarValue::String("CEO".into())),
                (name, ScalarValue::String("Emil Eifrem".into())),
                (id, ScalarValue::Integer(1)),
            ],
        })?;
        graph.insert_node(NodeInput {
            id: NodeId(2),
            layer: Layer::Observed,
            revision: 2,
            labels: Vec::new(),
            properties: vec![
                (name, ScalarValue::String("Nobody".into())),
                (id, ScalarValue::Integer(2)),
            ],
        })?;
        let compiled =
            compile_optimized_row_query("MATCH (a) RETURN coalesce(a.title, a.name)", &graph)?
                .ok_or_else(|| {
                    Error::internal("Conditional1 [1] did not compile as a typed row")
                })?;
        compiled.request.validate()?;
        assert!(matches!(
            compiled.request.program.instructions.as_slice(),
            [
                ResidentRowInstruction {
                    output_type: ResidentRowValueType::String,
                    operation: ResidentRowOperation::LoadStringProperty {
                        property: first,
                        maximum_bytes: 3,
                        ..
                    },
                },
                ResidentRowInstruction {
                    output_type: ResidentRowValueType::String,
                    operation: ResidentRowOperation::LoadStringProperty {
                        property: second,
                        maximum_bytes: 11,
                        ..
                    },
                },
                ResidentRowInstruction {
                    output_type: ResidentRowValueType::String,
                    operation: ResidentRowOperation::StringCoalesce { left: 0, right: 1 },
                },
            ] if *first == title && *second == name
        ));
        assert_eq!(
            compiled.request.program.string_register_capacity(2)?,
            Some(11)
        );
        let result = execute_row_program_on_cpu(&graph, &compiled)?.into_parts();
        assert!(matches!(
            result.projected_columns.as_slice(),
            [projected] if projected.register == 2
                && matches!(
                    &projected.column,
                    ResidentRowColumn::String { offsets, bytes, validity }
                        if offsets == &[0, 3, 9]
                            && bytes == b"CEONobody"
                            && validity == &[1, 1]
                )
        ));

        let invalid = ResidentRowProgram {
            instructions: vec![
                ResidentRowInstruction {
                    output_type: ResidentRowValueType::Integer,
                    operation: ResidentRowOperation::IntegerConstant(1),
                },
                ResidentRowInstruction {
                    output_type: ResidentRowValueType::String,
                    operation: ResidentRowOperation::StringConstant("fallback".to_owned()),
                },
                ResidentRowInstruction {
                    output_type: ResidentRowValueType::String,
                    operation: ResidentRowOperation::StringCoalesce { left: 0, right: 1 },
                },
            ],
        };
        assert!(invalid.validate().is_err());
        for unsupported in [
            "MATCH (a) RETURN coalesce(a.title)",
            "MATCH (a) RETURN coalesce(a.title, a.name, a.name)",
            "MATCH (a) RETURN coalesce(a.title, 'fallback')",
            "MATCH (a) RETURN coalesce(coalesce(a.title, a.name), a.name)",
            "MATCH (a) RETURN coalesce(a.title, a.id)",
        ] {
            assert!(
                compile_optimized_row_query(unsupported, &graph)?.is_none(),
                "unsupported string coalesce escaped exact row admission: {unsupported}"
            );
        }
        Ok(())
    }
}
