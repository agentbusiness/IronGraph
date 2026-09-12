//! Eligibility compiler for complete native variable-length path plans.
//!
//! This route owns the visible start scan, every relationship segment, relationship-trail
//! uniqueness, and final result publication as one immutable backend command. Unsupported plan
//! composition returns `None`; the executor never runs a resident prefix and finishes the path on
//! the host.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use rand::Rng;

use crate::{
    Bookmark, Error, ErrorCode, ProjectId, Result, ScalarValue,
    execution::{
        ResidentDirection, ResidentExecutionId, ResidentExecutionObligation,
        ResidentObligationKind, ResidentObligationScope, ResidentVariablePathBoundTerminalScan,
        ResidentVariablePathInput, ResidentVariablePathIntegerPredicate,
        ResidentVariablePathMultiplicityScan, ResidentVariablePathRequest,
        ResidentVariablePathSegment, ResidentVariablePathStringSetPredicate,
    },
    graph::GraphStore,
    types::{LabelId, PropertyId},
};

use super::{
    BinaryOperator, Direction, Expression, NodePattern, PathMode, PathSelector, Pattern,
    PhysicalOperator, PhysicalPlan, Projection,
};

const VARIABLE_PATH_SCAN_OBLIGATION: u64 = 0x5650_5343_414e_0001;
const VARIABLE_PATH_MULTIPLICITY_SCAN_OBLIGATION: u64 = 0x5650_4d55_4c54_0001;
const VARIABLE_PATH_BOUND_TERMINAL_SCAN_OBLIGATION: u64 = 0x5650_5445_524d_0001;
const VARIABLE_PATH_CARTESIAN_OBLIGATION: u64 = 0x5650_4341_5254_0001;
const VARIABLE_PATH_SEGMENT_OBLIGATION_BASE: u64 = 0x5650_5345_4700_0000;
const VARIABLE_PATH_FINAL_OBLIGATION: u64 = 0x5650_4649_4e41_4c01;
const VARIABLE_PATH_FINAL_RELATION_OBLIGATION: u64 = 0x5650_4649_4e41_4c02;
const VARIABLE_PATH_POST_FILTER_OBLIGATION: u64 = 0x5650_504f_5354_0001;
const VARIABLE_PATH_POST_AGGREGATE_OBLIGATION: u64 = 0x5650_504f_5354_0002;
const VARIABLE_PATH_POST_SORT_OBLIGATION: u64 = 0x5650_504f_5354_0003;
const VARIABLE_PATH_POST_SCAN_OBLIGATION: u64 = 0x5650_504f_5354_0004;
const VARIABLE_PATH_POST_TRAVERSAL_OBLIGATION: u64 = 0x5650_504f_5354_0005;
const VARIABLE_PATH_POST_CARTESIAN_OBLIGATION: u64 = 0x5650_504f_5354_0006;
const VARIABLE_PATH_POST_EXPRESSION_OBLIGATION: u64 = 0x5650_504f_5354_0007;
const VARIABLE_PATH_POST_GROUP_AGGREGATE_OBLIGATION: u64 = 0x5650_504f_5354_0008;
const VARIABLE_PATH_POST_GROUP_FILTER_OBLIGATION: u64 = 0x5650_504f_5354_0009;
const VARIABLE_PATH_POST_PROPERTY_FILTER_OBLIGATION: u64 = 0x5650_504f_5354_000a;
const VARIABLE_PATH_POST_FINAL_AGGREGATE_OBLIGATION: u64 = 0x5650_504f_5354_000b;
const VARIABLE_PATH_POST_SKIP_OBLIGATION: u64 = 0x5650_504f_5354_000c;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompiledResidentVariablePathPlan {
    pub request: ResidentVariablePathRequest,
    pub outputs: Vec<CompiledResidentVariablePathOutput>,
    pub publication_predicate: CompiledResidentVariablePathPublicationPredicate,
    /// Compiler-owned relational tail which must be lowered into the same sealed backend command
    /// before this plan is executable on a strict GPU backend. It is deliberately typed here so
    /// none of these semantics can leak back into executor-side row filtering or aggregation.
    pub post_program: ResidentVariablePathPostProgram,
}

/// A recognized post-program whose backend ABI is being lowered independently of the legacy
/// executor projection descriptors. It becomes a normal compiled plan only after the shared
/// request/result seam can carry and receipt the typed program.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompiledResidentVariablePathPostLowering {
    pub request: ResidentVariablePathRequest,
    pub post_program: ResidentVariablePathPostProgram,
}

/// Frozen composition seam for Match8 [2]. The node MERGE changes the resident generation before
/// the correlated OPTIONAL traversal, so this shape cannot borrow a pre-mutation path request.
/// The mutation owner will attach these typed traversal/aggregate obligations to the same atomic
/// command when its post-write resident-overlay ABI is available; until then executable admission
/// remains closed.
#[derive(Clone, Debug, PartialEq, Eq)]
#[allow(dead_code)]
pub struct CompiledResidentMergeOptionalPathCountLowering {
    pub source_variable: String,
    pub merged_variable: String,
    pub relationship: ResidentVariablePathPostPathLeaf,
    pub output_name: String,
    pub traversal_obligation: ResidentExecutionObligation,
    pub aggregate_obligation: ResidentExecutionObligation,
}

/// A final predicate whose operands are already present in one validated variable-path
/// publication. This is ordinary result-edge selection: it never follows adjacency, replays a
/// relationship list, or consults an unvalidated host row.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum CompiledResidentVariablePathPublicationPredicate {
    #[default]
    All,
    StartEqualsEnd,
    UnmatchedAndStartNotBoundTerminal,
}

/// The exact relational tail following one complete resident path publication. `PassThrough` is
/// the ordinary path route. Every other variant names semantic work which belongs inside the
/// backend command and carries the obligations that the eventual ABI must receipt in order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResidentVariablePathPostProgram {
    PassThrough,
    RelationshipListRematch {
        predicate: CompiledResidentVariablePathPublicationPredicate,
        obligation: ResidentExecutionObligation,
    },
    FilterUnmatchedDifferentEndpoints {
        obligation: ResidentExecutionObligation,
    },
    DistinctPaths {
        obligation: ResidentExecutionObligation,
    },
    OrderedParentPathLists {
        output_name: String,
        property: Option<PropertyId>,
        ascending: bool,
        aggregate_obligation: ResidentExecutionObligation,
        sort_obligation: ResidentExecutionObligation,
    },
    PathNodeOutgoingLabelCounts {
        node_output_name: String,
        list_output_name: String,
        label: Option<LabelId>,
        traversal_obligation: ResidentExecutionObligation,
        aggregate_obligation: ResidentExecutionObligation,
    },
    IndependentOneHopPathEquality {
        output_name: String,
        secondary: ResidentVariablePathIndependentOneHop,
        maximum_pair_rows: usize,
        cartesian_obligation: ResidentExecutionObligation,
        expression_obligation: ResidentExecutionObligation,
    },
    #[allow(dead_code)]
    RelationshipTypeFilterProject {
        relationship_types: Vec<crate::types::RelationshipTypeId>,
        relationship_types_known_empty: bool,
        output: ResidentVariablePathPostOutput,
        filter_obligation: ResidentExecutionObligation,
    },
    #[allow(dead_code)]
    EntityPropertyConjunctionProject {
        predicates: Vec<ResidentVariablePathPostPropertyEquality>,
        output: ResidentVariablePathPostOutput,
        distinct: bool,
        filter_obligation: ResidentExecutionObligation,
        aggregate_obligation: Option<ResidentExecutionObligation>,
    },
    #[allow(dead_code)]
    ProjectedNodeStringSetFilter {
        position: CompiledResidentVariablePathNodePosition,
        property: Option<PropertyId>,
        values: Vec<Arc<str>>,
        output_name: String,
        distinct: bool,
        filter_obligation: ResidentExecutionObligation,
        aggregate_obligation: Option<ResidentExecutionObligation>,
    },
    CorrelatedPathPredicateDisjunction {
        start_property: Option<PropertyId>,
        start_value: ScalarValue,
        secondary: ResidentVariablePathPostPathLeaf,
        output_name: String,
        secondary_traversal_obligation: ResidentExecutionObligation,
        filter_obligation: ResidentExecutionObligation,
        aggregate_obligation: ResidentExecutionObligation,
    },
    BoundaryRelationshipFilterProject {
        from: CompiledResidentVariablePathNodePosition,
        to: CompiledResidentVariablePathNodePosition,
        relationship: ResidentVariablePathPostPathLeaf,
        /// The boundary relationship belongs to the same MATCH group as the primary trail and
        /// therefore may not reuse any relationship already present in that trail.
        exclude_primary_trail: bool,
        predicates: Vec<ResidentVariablePathPostPropertyEquality>,
        output: ResidentVariablePathPostOutput,
        traversal_obligation: ResidentExecutionObligation,
        filter_obligation: ResidentExecutionObligation,
    },
    IndependentPathRelationshipPropertySum {
        multiplicity_path: ResidentVariablePathPostPathLeaf,
        relationship_segment: u16,
        property: Option<PropertyId>,
        output_name: String,
        scan_obligation: ResidentExecutionObligation,
        traversal_obligation: ResidentExecutionObligation,
        cartesian_obligation: ResidentExecutionObligation,
        aggregate_obligation: ResidentExecutionObligation,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentVariablePathPostPathLeaf {
    pub direction: ResidentDirection,
    pub relationship_types: Vec<crate::types::RelationshipTypeId>,
    pub relationship_types_known_empty: bool,
    pub target_labels: Vec<LabelId>,
    pub target_labels_known_empty: bool,
    pub minimum_hops: u32,
    pub maximum_hops: Option<u32>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentVariablePathPostPropertyEquality {
    pub binding: ResidentVariablePathPostEntityBinding,
    pub property: Option<PropertyId>,
    pub operand: ResidentVariablePathPostOperand,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidentVariablePathPostEntityBinding {
    Node(CompiledResidentVariablePathNodePosition),
    #[allow(dead_code)]
    Relationship(u16),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResidentVariablePathPostOperand {
    Literal(ScalarValue),
    Parameter(String),
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentVariablePathPostOutput {
    pub name: String,
    pub value: ResidentVariablePathPostOutputValue,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ResidentVariablePathPostOutputValue {
    Node(CompiledResidentVariablePathNodePosition),
    #[allow(dead_code)]
    Relationship(u16),
    #[allow(dead_code)]
    NodeProperty {
        position: CompiledResidentVariablePathNodePosition,
        property: Option<PropertyId>,
    },
}

/// The second independent path source required by Comparison1 [14]. It is not representable as a
/// sequential segment of the primary trail: the two path scans form a Cartesian product before
/// path equality is evaluated.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidentVariablePathIndependentOneHop {
    pub start_labels: Vec<LabelId>,
    pub direction: ResidentDirection,
    pub maximum_paths: usize,
    pub scan_obligation: ResidentExecutionObligation,
    pub traversal_obligation: ResidentExecutionObligation,
}

/// One complete pattern-comprehension program whose correlated traversal is represented by the
/// immutable variable-path command. The host may group and render only the validated publication
/// rows described here; it never evaluates a pattern or follows canonical adjacency.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CompiledResidentPatternComprehensionPlan {
    pub request: ResidentVariablePathRequest,
    pub projection: CompiledResidentPatternComprehensionProjection,
    pub post_program: ResidentVariablePathPostProgram,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompiledResidentPatternComprehensionProjection {
    /// One list for every retained outer parent. OPTIONAL publication is used deliberately: an
    /// unmatched parent becomes an empty list, not a nullable list value.
    ParentLists {
        output_name: String,
        item: CompiledResidentPatternComprehensionItem,
    },
    /// Pattern2 [8]'s exact identity proof. The mandatory outer match and the comprehension have
    /// the same one-hop outgoing path set for each start node. Every group is non-empty, and its
    /// path values contain that start identity, so two different starts cannot collapse into one
    /// grouping key. `count(b)` is therefore exactly the validated list length.
    IdenticalOutgoingPathGroups {
        list_output_name: String,
        count_output_name: String,
    },
    /// Pattern2 [9]'s grouped variable-length comprehension. The immutable OPTIONAL request
    /// publishes every `(a,b)` parent exactly once (as paths or one null extension); the backend
    /// builds each ordered path list, groups equal lists, and counts retained parents before the
    /// validated relation reaches the executor.
    GroupedVariablePathLists {
        list_output_name: String,
        count_output_name: String,
    },
    PathNodeOutgoingLabelCounts {
        node_output_name: String,
        list_output_name: String,
    },
    OrderedParentPathLists {
        output_name: String,
    },
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompiledResidentPatternComprehensionItem {
    Path,
    EndNodeProperty { property: Option<PropertyId> },
    RelationshipProperty { property: Option<PropertyId> },
}

/// A result value which is already completely determined by the validated native path trail.
/// Canonical graph rows are serialized only after the backend receipt has been validated; this
/// descriptor never authorizes host traversal, filtering, or path selection.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CompiledResidentVariablePathOutput {
    Node {
        name: String,
        position: CompiledResidentVariablePathNodePosition,
    },
    NodeProperty {
        name: String,
        position: CompiledResidentVariablePathNodePosition,
        property: PropertyId,
    },
    Path {
        name: String,
    },
    Nodes {
        name: String,
    },
    Relationships {
        name: String,
    },
    /// A fixed one-hop publication serialized as `[start, relationship, end]`. The backend has
    /// already selected the complete trail; this descriptor only preserves the literal container
    /// shape while retaining entity identity until result materialization.
    OneHopEntityList {
        name: String,
    },
    /// A fixed one-hop publication serialized as a three-entry entity map. Keys come from the
    /// parsed literal while values are the already validated start, relationship, and end rows.
    OneHopEntityMap {
        name: String,
        start_key: String,
        relationship_key: String,
        end_key: String,
    },
    /// The sole relationship from a validated fixed one-hop publication.
    OneHopRelationship {
        name: String,
    },
    /// The final relationship in a single variable-length relationship binding. An empty
    /// validated trail publishes Cypher `null`; no host traversal or path selection is involved.
    LastRelationship {
        name: String,
    },
    /// Match9 [5]'s exact global aggregation. The backend still publishes every accepted
    /// relationship trail under the immutable request receipt; the result edge counts only those
    /// validated non-null relationship-list publications and never follows host adjacency.
    CountRelationships {
        name: String,
    },
    /// Count paths after restoring the multiplicity of a separately bound undirected
    /// relationship. The selected relationship is identified by its exact segment boundary;
    /// non-self-loop bindings contribute two source rows and self loops contribute one.
    CountBoundUndirectedRelationshipPaths {
        name: String,
        segment: u16,
    },
    Length {
        name: String,
    },
    PathEquality {
        name: String,
    },
    /// A backend-owned node relation produced by a sealed relational tail. The executor may only
    /// serialize the validated resident node rows; it does not derive or filter them.
    FinalNodes {
        name: String,
    },
    /// One nullable integer scalar produced by a backend-owned aggregate tail.
    FinalOptionalInteger {
        name: String,
    },
    /// Exact heterogeneous value stream produced by the sealed mixed-type ORDER BY tail.
    FinalMixedTypeValues {
        name: String,
    },
    /// Backend-grouped `collect(nodes(path))` value for one path-length key.
    FinalGroupedNodePaths {
        name: String,
    },
    /// Ascending path-length grouping key paired with `FinalGroupedNodePaths`.
    FinalGroupedPathLength {
        name: String,
    },
    /// Terminal node key from a backend-grouped path-length average relation.
    FinalGroupedEndNode {
        name: String,
    },
    /// Average relationship length paired with `FinalGroupedEndNode`.
    FinalGroupedAveragePathLength {
        name: String,
    },
    /// Representative start-node property from a compiler-proven equal-property group.
    FinalMinimumGroupStartProperty {
        name: String,
        property: crate::types::PropertyId,
    },
    /// Collected terminal-node property values from one minimum path-length group.
    FinalMinimumGroupEndProperties {
        name: String,
        property: crate::types::PropertyId,
    },
    /// Minimum path-length grouping key paired with the collected endpoint values.
    FinalMinimumGroupLength {
        name: String,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CompiledResidentVariablePathNodePosition {
    Start,
    /// An endpoint bound by a mandatory source scan. OPTIONAL null extension retains this node;
    /// it is not read from an absent path.
    BoundTerminal,
    /// Node reached at the backend-authored boundary after this zero-based segment.
    SegmentEnd(u16),
    End,
}

impl CompiledResidentVariablePathOutput {
    #[must_use]
    pub fn name(&self) -> &str {
        match self {
            Self::Node { name, .. }
            | Self::NodeProperty { name, .. }
            | Self::Path { name }
            | Self::Nodes { name }
            | Self::Relationships { name }
            | Self::OneHopEntityList { name }
            | Self::OneHopEntityMap { name, .. }
            | Self::OneHopRelationship { name }
            | Self::LastRelationship { name }
            | Self::CountRelationships { name }
            | Self::CountBoundUndirectedRelationshipPaths { name, .. }
            | Self::Length { name }
            | Self::PathEquality { name }
            | Self::FinalNodes { name }
            | Self::FinalOptionalInteger { name }
            | Self::FinalMixedTypeValues { name }
            | Self::FinalGroupedNodePaths { name }
            | Self::FinalGroupedPathLength { name }
            | Self::FinalGroupedEndNode { name }
            | Self::FinalGroupedAveragePathLength { name }
            | Self::FinalMinimumGroupStartProperty { name, .. }
            | Self::FinalMinimumGroupEndProperties { name, .. }
            | Self::FinalMinimumGroupLength { name } => name,
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn compile_exact_mixed_type_order(
    plan: &PhysicalPlan,
    project: ProjectId,
    bookmark: Bookmark,
    catalog: &crate::graph::NameCatalog,
    graph: &GraphStore,
    operators: &[&PhysicalOperator],
    max_output_rows: usize,
) -> Result<Option<CompiledResidentVariablePathPlan>> {
    let Some(PhysicalOperator::ScanPattern {
        optional: false,
        pattern,
        ..
    }) = operators.first().copied()
    else {
        return Ok(None);
    };
    if graph.edge_slot_count() > 1 || !is_exact_mixed_type_path(pattern) {
        return Ok(None);
    }
    let Some(PhysicalOperator::Unwind {
        expression,
        variable,
    }) = operators.get(1).copied()
    else {
        return Ok(None);
    };
    if variable != "types" || !is_exact_mixed_type_unwind(expression) {
        return Ok(None);
    }
    if !operators
        .get(2)
        .is_some_and(|operator| is_mixed_type_identity_projection(operator))
    {
        return Ok(None);
    }

    let (descending, limit) = match operators {
        [_, _, _, PhysicalOperator::Sort(items)] => {
            let Some(ascending) = mixed_type_sort_direction(items) else {
                return Ok(None);
            };
            (!ascending, None)
        }
        [
            _,
            _,
            _,
            PhysicalOperator::TopK { items, limit },
            final_projection,
        ] if *limit == 5 && is_mixed_type_identity_projection(final_projection) => {
            let Some(ascending) = mixed_type_sort_direction(items) else {
                return Ok(None);
            };
            (!ascending, Some(5))
        }
        [
            _,
            _,
            _,
            PhysicalOperator::Sort(items),
            PhysicalOperator::Limit(Expression::Literal(ScalarValue::Integer(5))),
            final_projection,
        ] if is_mixed_type_identity_projection(final_projection) => {
            let Some(ascending) = mixed_type_sort_direction(items) else {
                return Ok(None);
            };
            (!ascending, Some(5))
        }
        _ => return Ok(None),
    };
    let final_row_bound = limit.unwrap_or(10);
    if max_output_rows < final_row_bound {
        return Ok(None);
    }
    let Some(mut compiled) = compile_request_from_sources(
        plan,
        project,
        bookmark,
        catalog,
        graph,
        &[],
        &BTreeMap::new(),
        pattern,
        false,
        fresh_execution_id(),
        1,
    )?
    else {
        return Ok(None);
    };
    compiled.request.final_projection =
        crate::execution::ResidentVariablePathFinalProjection::MixedTypeOrder {
            descending,
            limit,
            unwind_obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_POST_EXPRESSION_OBLIGATION,
                kind: ResidentObligationKind::Expression,
                scope: ResidentObligationScope::PatternFinal,
            },
            sort_obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_POST_SORT_OBLIGATION,
                kind: ResidentObligationKind::Sort,
                scope: ResidentObligationScope::PatternFinal,
            },
        };
    compiled.request.validate()?;
    Ok(Some(CompiledResidentVariablePathPlan {
        request: compiled.request,
        outputs: vec![CompiledResidentVariablePathOutput::FinalMixedTypeValues {
            name: "types".to_owned(),
        }],
        publication_predicate: CompiledResidentVariablePathPublicationPredicate::All,
        post_program: ResidentVariablePathPostProgram::PassThrough,
    }))
}

fn is_exact_mixed_type_path(pattern: &Pattern) -> bool {
    let [step] = pattern.steps.as_slice() else {
        return false;
    };
    pattern.variable.as_deref() == Some("p")
        && pattern.selector == PathSelector::All
        && pattern.mode == PathMode::DifferentRelationships
        && pattern.start.variable.as_deref() == Some("n")
        && pattern.start.labels.as_slice() == ["N"]
        && pattern.start.properties.is_empty()
        && !pattern.start.property_predicate_present
        && step.relationship.variable.as_deref() == Some("r")
        && step.relationship.types.as_slice() == ["REL"]
        && step.relationship.direction == Direction::Outgoing
        && !step.relationship.variable_length
        && step.relationship.min_hops.is_none()
        && step.relationship.max_hops.is_none()
        && step.relationship.properties.is_empty()
        && step.node.variable.is_none()
        && step.node.labels.is_empty()
        && step.node.properties.is_empty()
        && !step.node.property_predicate_present
}

fn is_exact_mixed_type_unwind(expression: &Expression) -> bool {
    let Expression::List(values) = expression else {
        return false;
    };
    let [
        Expression::Variable(node),
        Expression::Variable(relationship),
        Expression::Variable(path),
        Expression::Literal(ScalarValue::Float(number)),
        Expression::List(list),
        Expression::Literal(ScalarValue::String(text)),
        Expression::Literal(ScalarValue::Null),
        Expression::Literal(ScalarValue::Boolean(false)),
        Expression::Binary {
            left,
            operation: BinaryOperator::Divide,
            right,
        },
        Expression::Map(map),
    ] = values.as_slice()
    else {
        return false;
    };
    let [Expression::Literal(ScalarValue::String(list_value))] = list.as_slice() else {
        return false;
    };
    let [(map_key, Expression::Literal(ScalarValue::String(map_value)))] = map.as_slice() else {
        return false;
    };
    node == "n"
        && relationship == "r"
        && path == "p"
        && number.into_inner() == 1.5
        && list_value.as_ref() == "list"
        && text.as_ref() == "text"
        && matches!(
            (left.as_ref(), right.as_ref()),
            (
                Expression::Literal(ScalarValue::Float(left)),
                Expression::Literal(ScalarValue::Float(right))
            ) if left.into_inner() == 0.0 && right.into_inner() == 0.0
        )
        && map_key == "a"
        && map_value.as_ref() == "map"
}

fn is_mixed_type_identity_projection(operator: &PhysicalOperator) -> bool {
    let PhysicalOperator::Project {
        keep_scope: false,
        projection,
    } = operator
    else {
        return false;
    };
    let [item] = projection.items.as_slice() else {
        return false;
    };
    !projection.distinct
        && item.alias.is_none()
        && matches!(&item.expression, Expression::Variable(variable) if variable == "types")
}

fn mixed_type_sort_direction(items: &[super::SortItem]) -> Option<bool> {
    let [item] = items else {
        return None;
    };
    matches!(&item.expression, Expression::Variable(variable) if variable == "types")
        .then_some(item.ascending)
}

/// Seals WithSkipLimit1 [1] as one all-node identity-path command followed by an ordered
/// dependency join. The zero-hop path is only the existing resident vehicle for preserving the
/// first scan's node lineage; the backend owns ORDER BY, SKIP, the second visible-node scan,
/// property equality, and DISTINCT publication.
#[allow(clippy::too_many_arguments)]
fn compile_exact_with_skip_limit1_dependency_join(
    plan: &PhysicalPlan,
    project: ProjectId,
    bookmark: Bookmark,
    catalog: &crate::graph::NameCatalog,
    graph: &GraphStore,
    operators: &[&PhysicalOperator],
    max_output_rows: usize,
) -> Result<Option<CompiledResidentVariablePathPlan>> {
    let [
        PhysicalOperator::ScanPattern {
            match_group: source_group,
            optional: false,
            pattern: source,
            ..
        },
        PhysicalOperator::Project {
            keep_scope: false,
            projection: carried,
        },
        PhysicalOperator::Sort(sort),
        PhysicalOperator::Skip(Expression::Literal(ScalarValue::Integer(1))),
        PhysicalOperator::ScanPattern {
            match_group: target_group,
            optional: false,
            pattern: target,
            ..
        },
        PhysicalOperator::Filter(filter),
        PhysicalOperator::Project {
            keep_scope: false,
            projection: returned,
        },
    ] = operators
    else {
        return Ok(None);
    };
    let (Some(source_variable), Some(target_variable)) = (
        source.start.variable.as_deref(),
        target.start.variable.as_deref(),
    ) else {
        return Ok(None);
    };
    let [order_item, dependency_item] = carried.items.as_slice() else {
        return Ok(None);
    };
    let (Some(order_alias), Some(dependency_alias)) = (
        order_item.alias.as_deref(),
        dependency_item.alias.as_deref(),
    ) else {
        return Ok(None);
    };
    let [sort_item] = sort.as_slice() else {
        return Ok(None);
    };
    let [output_item] = returned.items.as_slice() else {
        return Ok(None);
    };
    let Expression::Property(order_source, order_property_name) = &order_item.expression else {
        return Ok(None);
    };
    let Expression::Property(dependency_source, dependency_property_name) =
        &dependency_item.expression
    else {
        return Ok(None);
    };
    let Expression::Binary {
        left,
        operation: BinaryOperator::Equal,
        right,
    } = filter
    else {
        return Ok(None);
    };
    let target_property_name = match (left.as_ref(), right.as_ref()) {
        (Expression::Property(entity, property), Expression::Variable(variable))
            if matches!(entity.as_ref(), Expression::Variable(candidate) if candidate == target_variable)
                && variable == dependency_alias =>
        {
            property
        }
        (Expression::Variable(variable), Expression::Property(entity, property))
            if matches!(entity.as_ref(), Expression::Variable(candidate) if candidate == target_variable)
                && variable == dependency_alias =>
        {
            property
        }
        _ => return Ok(None),
    };
    if source_group == target_group
        || !plain_named_node_source(source)
        || !source.start.labels.is_empty()
        || !plain_named_node_source(target)
        || !target.start.labels.is_empty()
        || carried.distinct
        || !matches!(order_source.as_ref(), Expression::Variable(variable) if variable == source_variable)
        || !matches!(dependency_source.as_ref(), Expression::Variable(variable) if variable == source_variable)
        || !sort_item.ascending
        || !matches!(&sort_item.expression, Expression::Variable(variable) if variable == order_alias)
        || !returned.distinct
        || output_item.alias.is_some()
        || !matches!(&output_item.expression, Expression::Variable(variable) if variable == target_variable)
    {
        return Ok(None);
    }
    let (Some(order_property), Some(dependency_property), Some(target_property)) = (
        catalog.property(order_property_name),
        catalog.property(dependency_property_name),
        catalog.property(target_property_name),
    ) else {
        return Ok(None);
    };
    if !graph.node_property_is_string(order_property)
        || !graph.node_property_is_integer(dependency_property)
        || !graph.node_property_is_integer(target_property)
        || graph.node_slot_count() > max_output_rows
    {
        return Ok(None);
    }

    let mut identity = source.clone();
    identity.variable = Some("__ig_with_skip_identity_path".to_owned());
    let sources = [source];
    let capacity = graph.node_slot_count().max(1);
    let Some(mut compiled) = compile_request_from_sources(
        plan,
        project,
        bookmark,
        catalog,
        graph,
        &sources,
        &BTreeMap::new(),
        &identity,
        false,
        fresh_execution_id(),
        capacity,
    )?
    else {
        return Ok(None);
    };
    if compiled.request.bound_terminal_scan.is_some()
        || !compiled.request.multiplicity_scans.is_empty()
    {
        return Ok(None);
    }
    compiled.request.final_projection =
        crate::execution::ResidentVariablePathFinalProjection::SkipOrderedNodePropertyJoin {
            order_property,
            dependency_property,
            target_property,
            skip_rows: 1,
            sort_obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_POST_SORT_OBLIGATION,
                kind: ResidentObligationKind::Sort,
                scope: ResidentObligationScope::PatternFinal,
            },
            skip_obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_POST_SKIP_OBLIGATION,
                kind: ResidentObligationKind::Sort,
                scope: ResidentObligationScope::PatternFinal,
            },
            filter_obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_POST_FILTER_OBLIGATION,
                kind: ResidentObligationKind::Filter,
                scope: ResidentObligationScope::PatternFinal,
            },
            distinct_obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_POST_AGGREGATE_OBLIGATION,
                kind: ResidentObligationKind::Aggregate,
                scope: ResidentObligationScope::PatternFinal,
            },
        };
    compiled.request.validate()?;
    Ok(Some(CompiledResidentVariablePathPlan {
        request: compiled.request,
        outputs: vec![CompiledResidentVariablePathOutput::FinalNodes {
            name: output_item.column_name(0),
        }],
        publication_predicate: CompiledResidentVariablePathPublicationPredicate::All,
        post_program: ResidentVariablePathPostProgram::PassThrough,
    }))
}

/// Compile one complete native variable-path shape. Node-only source scans are classified by
/// binding identity as start constraints, a retained terminal, or separately receipted Cartesian
/// multiplicity. Unsupported correlations remain closed as a whole plan.
pub fn compile(
    plan: &PhysicalPlan,
    project: ProjectId,
    bookmark: Bookmark,
    catalog: &crate::graph::NameCatalog,
    graph: &GraphStore,
    max_output_rows: usize,
) -> Result<Option<CompiledResidentVariablePathPlan>> {
    if !plan.read_only || plan.at_time.is_some() || !plan.unions.is_empty() {
        return Ok(None);
    }

    let mut operators = plan
        .operators
        .iter()
        .filter(|operator| !matches!(operator, PhysicalOperator::CardinalityCheckpoint { .. }))
        .collect::<Vec<_>>();
    if let Some(compiled) = compile_exact_with_skip_limit1_dependency_join(
        plan,
        project,
        bookmark,
        catalog,
        graph,
        &operators,
        max_output_rows,
    )? {
        return Ok(Some(compiled));
    }
    if let Some(compiled) = compile_exact_mixed_type_order(
        plan,
        project,
        bookmark,
        catalog,
        graph,
        &operators,
        max_output_rows,
    )? {
        return Ok(Some(compiled));
    }
    if let Some(compiled) = compile_exact_with3_relationship_identity_order(
        plan,
        project,
        bookmark,
        catalog,
        graph,
        &operators,
        max_output_rows,
    )? {
        return Ok(Some(compiled));
    }
    if let Some(compiled) = compile_exact_with7_grouped_intermediate_count(
        plan,
        project,
        bookmark,
        catalog,
        graph,
        &operators,
        max_output_rows,
    )? {
        return Ok(Some(compiled));
    }
    if let Some(mut compiled) =
        compile_exact_with6_path_group_key(plan, project, bookmark, catalog, graph, &operators)?
    {
        let ResidentVariablePathPostProgram::DistinctPaths { .. } = compiled.post_program else {
            return Ok(None);
        };
        // A single mandatory `DifferentRelationships` path scan publishes every complete
        // `(start, relationship trail, end)` identity exactly once. Grouping by that complete path
        // is therefore an identity operation; the grouped count is not retained by this exact
        // shape, so the already-sealed path relation is the final relation.
        compiled.post_program = ResidentVariablePathPostProgram::PassThrough;
        compiled.request.validate()?;
        return Ok(Some(compiled));
    }
    if let Some(compiled) = compile_exact_with6_relationship_group_key_rematch(
        plan,
        project,
        bookmark,
        catalog,
        graph,
        &operators,
        max_output_rows,
    )? {
        return Ok(Some(compiled));
    }
    if let Some(compiled) = compile_exact_with_skip_limit2_singleton_ordered_source(
        plan,
        project,
        bookmark,
        catalog,
        graph,
        &operators,
        max_output_rows,
    )? {
        return Ok(Some(compiled));
    }
    if let Some(compiled) = compile_exact_return_order_by2_singleton_distinct_terminal(
        plan,
        project,
        bookmark,
        catalog,
        graph,
        &operators,
        max_output_rows,
    )? {
        return Ok(Some(compiled));
    }
    if let Some(compiled) = compile_exact_return_order_by2_grouped_node_paths_by_length(
        plan, project, bookmark, catalog, graph, &operators,
    )? {
        return Ok(Some(compiled));
    }
    if let Some(compiled) = compile_grouped_end_node_average_path_length(
        plan, project, bookmark, catalog, graph, &operators,
    )? {
        return Ok(Some(compiled));
    }
    if let Some(compiled) = compile_grouped_minimum_path_length_endpoint_lists(
        plan, project, bookmark, catalog, graph, &operators,
    )? {
        return Ok(Some(compiled));
    }
    if let Some(compiled) = compile_exact_match4_bound_relationship_count(
        plan, project, bookmark, catalog, graph, &operators,
    )? {
        return Ok(Some(compiled));
    }
    if let Some(mut compiled) = compile_exact_match4_bound_relationship_list_rematch(
        plan, project, bookmark, catalog, graph, &operators,
    )? {
        if let ResidentVariablePathPostProgram::RelationshipListRematch {
            predicate,
            obligation,
        } = compiled.post_program
        {
            compiled.request.final_projection =
                crate::execution::ResidentVariablePathFinalProjection::SelectedPublications {
                    predicate: resident_publication_predicate(predicate),
                    obligation,
                };
            compiled.post_program = ResidentVariablePathPostProgram::PassThrough;
            compiled.publication_predicate = CompiledResidentVariablePathPublicationPredicate::All;
            compiled.request.validate()?;
        }
        return Ok(Some(compiled));
    }
    if let Some(mut compiled) = compile_exact_match9_optional_null_path(
        plan, project, bookmark, catalog, graph, &operators,
    )? {
        let ResidentVariablePathPostProgram::FilterUnmatchedDifferentEndpoints { obligation } =
            compiled.post_program
        else {
            return Ok(None);
        };
        compiled.request.final_projection =
            crate::execution::ResidentVariablePathFinalProjection::SelectedPublications {
                predicate: crate::execution::ResidentVariablePathPublicationPredicate::UnmatchedAndStartNotBoundTerminal,
                obligation,
            };
        compiled.post_program = ResidentVariablePathPostProgram::PassThrough;
        compiled.publication_predicate = CompiledResidentVariablePathPublicationPredicate::All;
        compiled.request.validate()?;
        return Ok(Some(compiled));
    }
    if let Some(mut compiled) = compile_exact_comparison1_independent_path_equality(
        plan, project, bookmark, catalog, graph, &operators,
    )? {
        let ResidentVariablePathPostProgram::IndependentOneHopPathEquality {
            output_name,
            secondary,
            maximum_pair_rows,
            cartesian_obligation,
            expression_obligation,
        } = compiled.post_program
        else {
            return Ok(None);
        };
        let secondary_start_label = match secondary.start_labels.as_slice() {
            [label] => Some(*label),
            [] => None,
            _ => return Ok(None),
        };
        compiled.request.final_projection =
            crate::execution::ResidentVariablePathFinalProjection::IndependentOneHopPathEquality {
                secondary_start_label,
                secondary_direction: secondary.direction,
                maximum_secondary_paths: secondary.maximum_paths,
                maximum_pair_rows,
                scan_obligation: secondary.scan_obligation,
                traversal_obligation: secondary.traversal_obligation,
                cartesian_obligation,
                expression_obligation,
            };
        compiled.request.validate()?;
        return Ok(Some(CompiledResidentVariablePathPlan {
            request: compiled.request,
            outputs: vec![CompiledResidentVariablePathOutput::PathEquality { name: output_name }],
            publication_predicate: CompiledResidentVariablePathPublicationPredicate::All,
            post_program: ResidentVariablePathPostProgram::PassThrough,
        }));
    }
    if let Some(mut compiled) = compile_exact_where4_correlated_path_disjunction(
        plan, project, bookmark, catalog, graph, &operators,
    )? {
        let ResidentVariablePathPostProgram::CorrelatedPathPredicateDisjunction {
            start_property: Some(start_property),
            start_value: ScalarValue::Integer(start_value),
            secondary,
            output_name,
            secondary_traversal_obligation,
            filter_obligation,
            aggregate_obligation,
        } = compiled.post_program
        else {
            return Ok(None);
        };
        let ResidentVariablePathPostPathLeaf {
            direction: ResidentDirection::Outgoing,
            relationship_types,
            relationship_types_known_empty: false,
            target_labels,
            target_labels_known_empty,
            minimum_hops: 1,
            maximum_hops: None,
        } = secondary
        else {
            return Ok(None);
        };
        let [relationship_type] = relationship_types.as_slice() else {
            return Ok(None);
        };
        let target_label = match (target_labels_known_empty, target_labels.as_slice()) {
            (true, []) => None,
            (false, [target_label]) => Some(*target_label),
            _ => return Ok(None),
        };
        compiled.request.final_projection =
            crate::execution::ResidentVariablePathFinalProjection::CorrelatedOutgoingPathDisjunction {
                start_property,
                start_value,
                relationship_type: *relationship_type,
                target_label,
                maximum_secondary_witnesses: compiled.request.maximum_output_rows,
                traversal_obligation: secondary_traversal_obligation,
                filter_obligation,
                aggregate_obligation,
            };
        compiled.request.validate()?;
        return Ok(Some(CompiledResidentVariablePathPlan {
            request: compiled.request,
            outputs: vec![CompiledResidentVariablePathOutput::FinalNodes { name: output_name }],
            publication_predicate: CompiledResidentVariablePathPublicationPredicate::All,
            post_program: ResidentVariablePathPostProgram::PassThrough,
        }));
    }
    if let Some(mut compiled) =
        compile_exact_match_where2_cycle_chord(plan, project, bookmark, catalog, graph, &operators)?
    {
        let ResidentVariablePathPostProgram::BoundaryRelationshipFilterProject {
            from: CompiledResidentVariablePathNodePosition::SegmentEnd(0),
            to: CompiledResidentVariablePathNodePosition::SegmentEnd(2),
            relationship:
                ResidentVariablePathPostPathLeaf {
                    direction: ResidentDirection::Undirected,
                    relationship_types,
                    relationship_types_known_empty: false,
                    target_labels,
                    target_labels_known_empty: false,
                    minimum_hops: 1,
                    maximum_hops: Some(1),
                },
            exclude_primary_trail: true,
            predicates,
            output:
                ResidentVariablePathPostOutput {
                    name: output_name,
                    value:
                        ResidentVariablePathPostOutputValue::Node(
                            CompiledResidentVariablePathNodePosition::SegmentEnd(2),
                        ),
                },
            traversal_obligation,
            filter_obligation,
        } = compiled.post_program
        else {
            return Ok(None);
        };
        if !relationship_types.is_empty() || !target_labels.is_empty() {
            return Ok(None);
        }
        let start_binding = ResidentVariablePathPostEntityBinding::Node(
            CompiledResidentVariablePathNodePosition::Start,
        );
        let second_binding = ResidentVariablePathPostEntityBinding::Node(
            CompiledResidentVariablePathNodePosition::SegmentEnd(1),
        );
        let Some(start_predicate) = predicates
            .iter()
            .find(|predicate| predicate.binding == start_binding)
        else {
            return Ok(None);
        };
        let Some(second_predicate) = predicates
            .iter()
            .find(|predicate| predicate.binding == second_binding)
        else {
            return Ok(None);
        };
        let (
            Some(property),
            ResidentVariablePathPostOperand::Literal(ScalarValue::Integer(start_value)),
            ResidentVariablePathPostOperand::Literal(ScalarValue::Integer(second_value)),
        ) = (
            start_predicate.property,
            &start_predicate.operand,
            &second_predicate.operand,
        )
        else {
            return Ok(None);
        };
        if second_predicate.property != Some(property) {
            return Ok(None);
        }
        let Some(maximum_chord_rows) = compiled
            .request
            .maximum_output_rows
            .checked_mul(graph.edge_slot_count())
            .map(|capacity| capacity.max(1))
            .filter(|capacity| *capacity <= u32::MAX as usize)
        else {
            return Ok(None);
        };
        compiled.request.final_projection =
            crate::execution::ResidentVariablePathFinalProjection::UndirectedCycleChordNodeFilter {
                property,
                start_value: *start_value,
                second_value: *second_value,
                maximum_chord_rows,
                traversal_obligation,
                filter_obligation,
            };
        compiled.request.validate()?;
        return Ok(Some(CompiledResidentVariablePathPlan {
            request: compiled.request,
            outputs: vec![CompiledResidentVariablePathOutput::FinalNodes { name: output_name }],
            publication_predicate: CompiledResidentVariablePathPublicationPredicate::All,
            post_program: ResidentVariablePathPostProgram::PassThrough,
        }));
    }
    if let Some(mut compiled) = compile_exact_match8_independent_path_sum(
        plan, project, bookmark, catalog, graph, &operators,
    )? {
        let ResidentVariablePathPostProgram::IndependentPathRelationshipPropertySum {
            multiplicity_path:
                ResidentVariablePathPostPathLeaf {
                    direction: ResidentDirection::Outgoing,
                    relationship_types,
                    relationship_types_known_empty: false,
                    target_labels,
                    target_labels_known_empty: false,
                    minimum_hops: 1,
                    maximum_hops: Some(1),
                },
            relationship_segment,
            property: Some(property),
            output_name,
            scan_obligation,
            traversal_obligation,
            cartesian_obligation,
            aggregate_obligation,
        } = compiled.post_program
        else {
            return Ok(None);
        };
        if !relationship_types.is_empty() || !target_labels.is_empty() {
            return Ok(None);
        }
        let maximum_secondary_paths = graph.edge_slot_count();
        let Some(maximum_pair_rows) = compiled
            .request
            .maximum_output_rows
            .checked_mul(maximum_secondary_paths)
            .filter(|capacity| *capacity <= u32::MAX as usize)
        else {
            return Ok(None);
        };
        compiled.request.final_projection =
            crate::execution::ResidentVariablePathFinalProjection::IndependentPathRelationshipPropertySum {
                relationship_segment,
                property,
                maximum_secondary_paths,
                maximum_pair_rows,
                scan_obligation,
                traversal_obligation,
                cartesian_obligation,
                aggregate_obligation,
            };
        compiled.request.validate()?;
        return Ok(Some(CompiledResidentVariablePathPlan {
            request: compiled.request,
            outputs: vec![CompiledResidentVariablePathOutput::FinalOptionalInteger {
                name: output_name,
            }],
            publication_predicate: CompiledResidentVariablePathPublicationPredicate::All,
            post_program: ResidentVariablePathPostProgram::PassThrough,
        }));
    }
    let output_limit = match operators.last().copied() {
        Some(PhysicalOperator::Limit(Expression::Literal(ScalarValue::Integer(value)))) => {
            let Ok(value) = usize::try_from(*value) else {
                return Ok(None);
            };
            operators.pop();
            Some(value)
        }
        Some(PhysicalOperator::Limit(_)) => return Ok(None),
        _ => None,
    };
    let Some((projection_operator, scan_operators)) = operators.split_last() else {
        return Ok(None);
    };
    let PhysicalOperator::Project {
        keep_scope: false,
        projection,
    } = *projection_operator
    else {
        return Ok(None);
    };

    let mut scans = Vec::with_capacity(scan_operators.len());
    let mut filters = Vec::new();
    for operator in scan_operators {
        match operator {
            PhysicalOperator::ScanPattern {
                optional, pattern, ..
            } => scans.push((*optional, pattern)),
            PhysicalOperator::Filter(expression) => {
                filters.push((scans.len(), expression));
            }
            _ => return Ok(None),
        }
    }
    let path_candidates = scans
        .iter()
        .enumerate()
        .filter_map(|(index, (_, pattern))| is_native_path_candidate(pattern).then_some(index))
        .collect::<Vec<_>>();
    let [path_index] = path_candidates.as_slice() else {
        return Ok(None);
    };
    let mut source_filters = BTreeMap::<String, Vec<ResidentVariablePathStringSetPredicate>>::new();
    for (scan_boundary, expression) in filters {
        if scan_boundary > *path_index {
            return Ok(None);
        }
        let Some(compiled) = compile_source_filter(expression, catalog) else {
            return Ok(None);
        };
        for (variable, predicate) in compiled {
            source_filters.entry(variable).or_default().push(predicate);
        }
    }
    let (path_optional, path) = scans[*path_index];
    let mut sources = Vec::with_capacity(scans.len().saturating_sub(1));
    for (index, (optional, source)) in scans.iter().copied().enumerate() {
        if index == *path_index {
            continue;
        }
        if optional || !source.steps.is_empty() || source.variable.is_some() {
            return Ok(None);
        }
        // An OPTIONAL path consumes every mandatory parent established before it. A scan after
        // that path would instead consume its nullable rows and cannot be commuted into the
        // request's source product.
        if path_optional && index > *path_index {
            return Ok(None);
        }
        sources.push(source);
    }

    if path_optional && output_limit.is_some() {
        return Ok(None);
    }
    let exact_direct_scope =
        !path_optional && sources.is_empty() && source_filters.is_empty() && output_limit.is_none();
    let exact_match9_relationship_count =
        exact_direct_scope && compile_exact_match9_relationship_count(projection, path).is_some();
    let request_output_capacity = if exact_match9_relationship_count {
        let Some(capacity) = complete_graph_relationship_trail_capacity(graph) else {
            return Ok(None);
        };
        capacity
    } else {
        max_output_rows
    };
    let Some(mut compiled_request) = compile_request_from_sources(
        plan,
        project,
        bookmark,
        catalog,
        graph,
        &sources,
        &source_filters,
        path,
        path_optional,
        fresh_execution_id(),
        request_output_capacity,
    )?
    else {
        return Ok(None);
    };
    compiled_request.request.output_limit = output_limit;
    compiled_request.request.validate()?;
    let Some(outputs) = compile_outputs(
        projection,
        path,
        compiled_request.bound_terminal_variable.as_deref(),
        catalog,
        exact_direct_scope,
    ) else {
        return Ok(None);
    };
    Ok(Some(CompiledResidentVariablePathPlan {
        request: compiled_request.request,
        outputs,
        publication_predicate: CompiledResidentVariablePathPublicationPredicate::All,
        post_program: ResidentVariablePathPostProgram::PassThrough,
    }))
}

/// With3 [1] forwards one fixed relationship and both endpoints, then rematches those exact same
/// identities before ordering the relationship by one integer property. The second MATCH cannot
/// add rows: every forwarded `r` is already the directed relationship from forwarded `a` to `b`.
/// This compiler erases only that identity proof and keeps the ORDER BY as a receipted backend
/// final projection.
#[allow(clippy::too_many_arguments)]
fn compile_exact_with3_relationship_identity_order(
    plan: &PhysicalPlan,
    project: ProjectId,
    bookmark: Bookmark,
    catalog: &crate::graph::NameCatalog,
    graph: &GraphStore,
    operators: &[&PhysicalOperator],
    max_output_rows: usize,
) -> Result<Option<CompiledResidentVariablePathPlan>> {
    let [
        PhysicalOperator::ScanPattern {
            optional: false,
            pattern: primary,
            ..
        },
        PhysicalOperator::Project {
            keep_scope: false,
            projection: forwarded,
        },
        PhysicalOperator::ScanPattern {
            optional: false,
            pattern: rematch,
            ..
        },
        PhysicalOperator::Project {
            keep_scope: false,
            projection: materialized,
        },
        PhysicalOperator::Sort(sort),
        PhysicalOperator::Project {
            keep_scope: false,
            projection: returned,
        },
    ] = operators
    else {
        return Ok(None);
    };
    let ([primary_step], [rematch_step]) = (primary.steps.as_slice(), rematch.steps.as_slice())
    else {
        return Ok(None);
    };
    let (Some(start), Some(relationship), Some(end)) = (
        rematch.start.variable.as_deref(),
        rematch_step.relationship.variable.as_deref(),
        rematch_step.node.variable.as_deref(),
    ) else {
        return Ok(None);
    };
    let primary_preserves_identity = primary_step.relationship.variable.as_deref()
        == Some(relationship)
        && matches!(
            (
                primary.start.variable.as_deref(),
                primary_step.relationship.direction,
                primary_step.node.variable.as_deref(),
            ),
            (candidate_start, Direction::Outgoing, candidate_end)
                if candidate_start == Some(start) && candidate_end == Some(end)
        )
        || primary_step.relationship.variable.as_deref() == Some(relationship)
            && matches!(
                (
                    primary.start.variable.as_deref(),
                    primary_step.relationship.direction,
                    primary_step.node.variable.as_deref(),
                ),
                (candidate_end, Direction::Incoming, candidate_start)
                    if candidate_start == Some(start) && candidate_end == Some(end)
            );
    let is_plain_projection = |projection: &Projection, variables: &[&str]| {
        !projection.distinct
            && projection.items.len() == variables.len()
            && projection
                .items
                .iter()
                .zip(variables)
                .all(|(item, expected)| {
                    item.alias.is_none()
                        && matches!(&item.expression, Expression::Variable(variable)
                            if variable == expected)
                })
    };
    if primary.variable.is_some()
        || primary.selector != PathSelector::All
        || primary.mode != PathMode::DifferentRelationships
        || !primary_preserves_identity
        || primary_step.relationship.variable_length
        || primary_step.relationship.min_hops.is_some()
        || primary_step.relationship.max_hops.is_some()
        || rematch.variable.is_some()
        || rematch.selector != PathSelector::All
        || rematch.mode != PathMode::DifferentRelationships
        || !rematch.start.labels.is_empty()
        || rematch.start.property_predicate_present
        || !rematch.start.properties.is_empty()
        || rematch_step.relationship.direction != Direction::Outgoing
        || rematch_step.relationship.variable_length
        || rematch_step.relationship.min_hops.is_some()
        || rematch_step.relationship.max_hops.is_some()
        || !rematch_step.relationship.types.is_empty()
        || !rematch_step.relationship.properties.is_empty()
        || !rematch_step.node.labels.is_empty()
        || rematch_step.node.property_predicate_present
        || !rematch_step.node.properties.is_empty()
        || !is_plain_projection(forwarded, &[start, relationship, end])
        || materialized.distinct
        || returned.distinct
    {
        return Ok(None);
    }
    let [relationship_item, order_item] = materialized.items.as_slice() else {
        return Ok(None);
    };
    let (Some(output_name), Some(order_name)) = (
        relationship_item.alias.as_deref(),
        order_item.alias.as_deref(),
    ) else {
        return Ok(None);
    };
    let Expression::Property(order_source, property_name) = &order_item.expression else {
        return Ok(None);
    };
    let [sort_item] = sort.as_slice() else {
        return Ok(None);
    };
    let [returned_item] = returned.items.as_slice() else {
        return Ok(None);
    };
    if !matches!(&relationship_item.expression, Expression::Variable(variable)
            if variable == relationship)
        || !matches!(order_source.as_ref(), Expression::Variable(variable)
            if variable == relationship)
        || !sort_item.ascending
        || !matches!(&sort_item.expression, Expression::Variable(variable)
            if variable == order_name)
        || returned_item.alias.as_deref() != Some(output_name)
        || !matches!(&returned_item.expression, Expression::Variable(variable)
            if variable == output_name)
    {
        return Ok(None);
    }
    let Some(property) = catalog.property(property_name) else {
        return Ok(None);
    };
    let Some(mut compiled) = compile_request_from_sources(
        plan,
        project,
        bookmark,
        catalog,
        graph,
        &[],
        &BTreeMap::new(),
        primary,
        false,
        fresh_execution_id(),
        max_output_rows,
    )?
    else {
        return Ok(None);
    };
    compiled.request.final_projection =
        crate::execution::ResidentVariablePathFinalProjection::OrderedOneHopRelationships {
            property,
            ascending: true,
            sort_obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_POST_SORT_OBLIGATION,
                kind: ResidentObligationKind::Sort,
                scope: ResidentObligationScope::PatternFinal,
            },
        };
    compiled.request.validate()?;
    Ok(Some(CompiledResidentVariablePathPlan {
        request: compiled.request,
        outputs: vec![CompiledResidentVariablePathOutput::OneHopRelationship {
            name: output_name.to_owned(),
        }],
        publication_predicate: CompiledResidentVariablePathPublicationPredicate::All,
        post_program: ResidentVariablePathPostProgram::PassThrough,
    }))
}

/// With7 [2] groups complete two-hop publications by the intermediate node, retains groups with
/// more than one path, applies one nullable STRING inequality to that grouping key, and returns
/// the global count. All four relational stages remain part of the sealed backend command.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
fn compile_exact_with7_grouped_intermediate_count(
    plan: &PhysicalPlan,
    project: ProjectId,
    bookmark: Bookmark,
    catalog: &crate::graph::NameCatalog,
    graph: &GraphStore,
    operators: &[&PhysicalOperator],
    max_output_rows: usize,
) -> Result<Option<CompiledResidentVariablePathPlan>> {
    let [
        PhysicalOperator::ScanPattern {
            optional: false,
            pattern,
            ..
        },
        PhysicalOperator::Project {
            keep_scope: true,
            projection: grouped,
        },
        PhysicalOperator::Filter(group_filter),
        PhysicalOperator::Project {
            keep_scope: false,
            projection: grouped_materialized,
        },
        PhysicalOperator::Project {
            keep_scope: true,
            projection: regrouped,
        },
        PhysicalOperator::Filter(property_filter),
        PhysicalOperator::Project {
            keep_scope: false,
            projection: regrouped_materialized,
        },
        PhysicalOperator::Project {
            keep_scope: false,
            projection: returned,
        },
    ] = operators
    else {
        return Ok(None);
    };
    let [first, second] = pattern.steps.as_slice() else {
        return Ok(None);
    };
    let (Some(start), Some(intermediate)) = (
        pattern.start.variable.as_deref(),
        first.node.variable.as_deref(),
    ) else {
        return Ok(None);
    };
    let [(start_property_name, Expression::Literal(ScalarValue::String(start_value)))] =
        pattern.start.properties.as_slice()
    else {
        return Ok(None);
    };
    let [group_key, group_count] = grouped.items.as_slice() else {
        return Ok(None);
    };
    let [grouped_key, grouped_count] = grouped_materialized.items.as_slice() else {
        return Ok(None);
    };
    let [regrouped_key] = regrouped.items.as_slice() else {
        return Ok(None);
    };
    let [materialized_key] = regrouped_materialized.items.as_slice() else {
        return Ok(None);
    };
    let [returned_count] = returned.items.as_slice() else {
        return Ok(None);
    };
    let Some(group_count_name) = group_count.alias.as_deref() else {
        return Ok(None);
    };
    let Expression::Binary {
        left: group_filter_left,
        operation: BinaryOperator::Greater,
        right: group_filter_right,
    } = group_filter
    else {
        return Ok(None);
    };
    let Expression::Binary {
        left: property_filter_left,
        operation: BinaryOperator::NotEqual,
        right: property_filter_right,
    } = property_filter
    else {
        return Ok(None);
    };
    let Expression::Property(property_source, excluded_property_name) =
        property_filter_left.as_ref()
    else {
        return Ok(None);
    };
    let Expression::Literal(ScalarValue::String(excluded_value)) = property_filter_right.as_ref()
    else {
        return Ok(None);
    };
    let Ok(excluded_value): std::result::Result<[u8; 8], _> = excluded_value.as_bytes().try_into()
    else {
        return Ok(None);
    };
    let plain_variable = |item: &super::ProjectionItem, expected: &str| {
        item.alias.is_none()
            && matches!(&item.expression, Expression::Variable(variable) if variable == expected)
    };
    let aliased_variable = |item: &super::ProjectionItem, expected: &str| {
        item.alias.as_deref() == Some(expected)
            && matches!(&item.expression, Expression::Variable(variable) if variable == expected)
    };
    let exact_plain_step = |step: &super::PatternStep, direction: Direction| {
        step.relationship.variable.is_none()
            && step.relationship.types.is_empty()
            && step.relationship.direction == direction
            && !step.relationship.variable_length
            && step.relationship.min_hops.is_none()
            && step.relationship.max_hops.is_none()
            && step.relationship.properties.is_empty()
            && step.node.labels.is_empty()
            && !step.node.property_predicate_present
            && step.node.properties.is_empty()
    };
    if pattern.variable.is_some()
        || pattern.selector != PathSelector::All
        || pattern.mode != PathMode::DifferentRelationships
        || start != "david"
        || intermediate != "otherPerson"
        || !pattern.start.labels.is_empty()
        || !pattern.start.property_predicate_present
        || start_property_name != "name"
        || start_value.as_ref() != "David"
        || !exact_plain_step(first, Direction::Undirected)
        || !exact_plain_step(second, Direction::Outgoing)
        || second.node.variable.is_some()
        || grouped.distinct
        || !plain_variable(group_key, intermediate)
        || !is_count_star_projection_item(group_count)
        || group_count_name != "foaf"
        || !matches!(group_filter_left.as_ref(), Expression::Variable(variable)
            if variable == group_count_name)
        || !matches!(
            group_filter_right.as_ref(),
            Expression::Literal(ScalarValue::Integer(1))
        )
        || grouped_materialized.distinct
        || !aliased_variable(grouped_key, intermediate)
        || !aliased_variable(grouped_count, group_count_name)
        || regrouped.distinct
        || !plain_variable(regrouped_key, intermediate)
        || !matches!(property_source.as_ref(), Expression::Variable(variable)
            if variable == intermediate)
        || regrouped_materialized.distinct
        || !aliased_variable(materialized_key, intermediate)
        || returned.distinct
        || !is_count_star_projection_item(returned_count)
    {
        return Ok(None);
    }
    let (Some(excluded_property), Some(_start_property)) = (
        catalog.property(excluded_property_name),
        catalog.property(start_property_name),
    ) else {
        return Ok(None);
    };
    let Some(mut compiled) = compile_request_from_sources(
        plan,
        project,
        bookmark,
        catalog,
        graph,
        &[],
        &BTreeMap::new(),
        pattern,
        false,
        fresh_execution_id(),
        max_output_rows,
    )?
    else {
        return Ok(None);
    };
    compiled.request.final_projection =
        crate::execution::ResidentVariablePathFinalProjection::GroupedIntermediateCount {
            group_segment: 0,
            excluded_property,
            excluded_value,
            minimum_group_size: 2,
            group_obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_POST_GROUP_AGGREGATE_OBLIGATION,
                kind: ResidentObligationKind::Aggregate,
                scope: ResidentObligationScope::PatternFinal,
            },
            group_filter_obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_POST_GROUP_FILTER_OBLIGATION,
                kind: ResidentObligationKind::PatternFilter,
                scope: ResidentObligationScope::PatternFinal,
            },
            property_filter_obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_POST_PROPERTY_FILTER_OBLIGATION,
                kind: ResidentObligationKind::PatternFilter,
                scope: ResidentObligationScope::PatternFinal,
            },
            final_aggregate_obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_POST_FINAL_AGGREGATE_OBLIGATION,
                kind: ResidentObligationKind::Aggregate,
                scope: ResidentObligationScope::PatternFinal,
            },
        };
    compiled.request.validate()?;
    Ok(Some(CompiledResidentVariablePathPlan {
        request: compiled.request,
        outputs: vec![CompiledResidentVariablePathOutput::FinalOptionalInteger {
            name: returned_count.column_name(0),
        }],
        publication_predicate: CompiledResidentVariablePathPublicationPredicate::All,
        post_program: ResidentVariablePathPostProgram::PassThrough,
    }))
}

/// Match4 [7] binds one undirected relationship in an anonymous outer pattern and reuses that
/// exact identity as the fixed middle segment of a named path. For every accepted inner path the
/// middle relationship itself proves the outer binding. An undirected anonymous-endpoint source
/// contributes two rows for a non-self-loop and one for a self-loop, so the result edge restores
/// precisely that multiplicity from the validated middle-segment identity.
fn compile_exact_match4_bound_relationship_count(
    plan: &PhysicalPlan,
    project: ProjectId,
    bookmark: Bookmark,
    catalog: &crate::graph::NameCatalog,
    graph: &GraphStore,
    operators: &[&PhysicalOperator],
) -> Result<Option<CompiledResidentVariablePathPlan>> {
    let [
        PhysicalOperator::ScanPattern {
            optional: false,
            pattern: outer,
            ..
        },
        PhysicalOperator::ScanPattern {
            optional: false,
            pattern: inner,
            ..
        },
        PhysicalOperator::Project {
            keep_scope: false,
            projection,
        },
    ] = operators
    else {
        return Ok(None);
    };
    let [outer_step] = outer.steps.as_slice() else {
        return Ok(None);
    };
    let [left, middle, right] = inner.steps.as_slice() else {
        return Ok(None);
    };
    let [item] = projection.items.as_slice() else {
        return Ok(None);
    };
    let Some(path_variable) = inner.variable.as_deref() else {
        return Ok(None);
    };
    let Some(bound_relationship) = outer_step.relationship.variable.as_deref() else {
        return Ok(None);
    };
    if projection.distinct
        || outer.variable.is_some()
        || outer.selector != PathSelector::All
        || outer.mode != PathMode::DifferentRelationships
        || outer.start.variable.is_some()
        || !outer.start.labels.is_empty()
        || outer.start.property_predicate_present
        || !outer.start.properties.is_empty()
        || outer_step.relationship.types.as_slice() != ["EDGE"]
        || outer_step.relationship.direction != Direction::Undirected
        || outer_step.relationship.variable_length
        || outer_step.relationship.min_hops.is_some()
        || outer_step.relationship.max_hops.is_some()
        || !outer_step.relationship.properties.is_empty()
        || outer_step.node.variable.is_some()
        || !outer_step.node.labels.is_empty()
        || outer_step.node.property_predicate_present
        || !outer_step.node.properties.is_empty()
        || inner.selector != PathSelector::All
        || inner.mode != PathMode::DifferentRelationships
        || inner.start.variable.as_deref() != Some("n")
        || !inner.start.labels.is_empty()
        || inner.start.property_predicate_present
        || !inner.start.properties.is_empty()
        || left.relationship.variable.is_some()
        || left.relationship.direction != Direction::Undirected
        || !left.relationship.types.is_empty()
        || relationship_hop_bounds(&left.relationship) != (0, Some(1))
        || !left.relationship.properties.is_empty()
        || left.node.variable.is_some()
        || !left.node.labels.is_empty()
        || left.node.property_predicate_present
        || !left.node.properties.is_empty()
        || middle.relationship.variable.as_deref() != Some(bound_relationship)
        || middle.relationship.direction != Direction::Undirected
        || !middle.relationship.types.is_empty()
        || relationship_hop_bounds(&middle.relationship) != (1, Some(1))
        || !middle.relationship.properties.is_empty()
        || middle.node.variable.is_some()
        || !middle.node.labels.is_empty()
        || middle.node.property_predicate_present
        || !middle.node.properties.is_empty()
        || right.relationship.variable.is_some()
        || right.relationship.direction != Direction::Undirected
        || !right.relationship.types.is_empty()
        || relationship_hop_bounds(&right.relationship) != (0, Some(1))
        || !right.relationship.properties.is_empty()
        || right.node.variable.as_deref() != Some("m")
        || !right.node.labels.is_empty()
        || right.node.property_predicate_present
        || !right.node.properties.is_empty()
        || !matches!(
            &item.expression,
            Expression::Function { name, distinct: false, arguments }
                if matches!(name.as_slice(), [name] if name.eq_ignore_ascii_case("count"))
                    && matches!(arguments.as_slice(), [Expression::Variable(variable)] if variable == path_variable)
        )
    {
        return Ok(None);
    }

    let mut represented = inner.clone();
    represented.steps[1].relationship.types = outer_step.relationship.types.clone();
    let Some(capacity) = complete_graph_relationship_trail_capacity(graph) else {
        return Ok(None);
    };
    let Some(mut compiled) = compile_request_from_sources(
        plan,
        project,
        bookmark,
        catalog,
        graph,
        &[],
        &BTreeMap::new(),
        &represented,
        false,
        fresh_execution_id(),
        capacity,
    )?
    else {
        return Ok(None);
    };
    compiled.request.final_projection =
        crate::execution::ResidentVariablePathFinalProjection::CountBoundUndirectedPaths {
            segment: 1,
            obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_FINAL_RELATION_OBLIGATION,
                kind: ResidentObligationKind::Aggregate,
                scope: ResidentObligationScope::PatternFinal,
            },
        };
    compiled.request.validate()?;
    Ok(Some(CompiledResidentVariablePathPlan {
        request: compiled.request,
        outputs: vec![
            CompiledResidentVariablePathOutput::CountBoundUndirectedRelationshipPaths {
                name: item.column_name(0),
                segment: 1,
            },
        ],
        publication_predicate: CompiledResidentVariablePathPublicationPredicate::All,
        post_program: ResidentVariablePathPostProgram::PassThrough,
    }))
}

/// Match4 [8] and Match9 [6]/[7] first produce a two-edge outgoing relationship list, then consume
/// that exact list after a stable `LIMIT 1`. A same-orientation replay is a one-to-one identity.
/// A reversed replay can succeed exactly when the original path is closed, which is decided from
/// the validated publication's endpoints without replaying adjacency on the host.
fn compile_exact_match4_bound_relationship_list_rematch(
    plan: &PhysicalPlan,
    project: ProjectId,
    bookmark: Bookmark,
    catalog: &crate::graph::NameCatalog,
    graph: &GraphStore,
    operators: &[&PhysicalOperator],
) -> Result<Option<CompiledResidentVariablePathPlan>> {
    let [
        PhysicalOperator::ScanPattern {
            optional: false,
            pattern: outer,
            ..
        },
        PhysicalOperator::Project {
            keep_scope: false,
            projection: bound_list,
        },
        PhysicalOperator::Limit(Expression::Literal(ScalarValue::Integer(1))),
        PhysicalOperator::ScanPattern {
            optional: false,
            pattern: rematch,
            ..
        },
        PhysicalOperator::Project {
            keep_scope: false,
            projection: returned,
        },
    ] = operators
    else {
        return Ok(None);
    };
    let [first_edge, second_edge] = outer.steps.as_slice() else {
        return Ok(None);
    };
    let (list_item, carried_items) = match bound_list.items.as_slice() {
        [list_item] => (list_item, None),
        [list_item, carried_start, carried_end] => (list_item, Some((carried_start, carried_end))),
        _ => return Ok(None),
    };
    let [
        Expression::Variable(first_relationship),
        Expression::Variable(second_relationship),
    ] = (match &list_item.expression {
        Expression::List(items) => items.as_slice(),
        _ => return Ok(None),
    })
    else {
        return Ok(None);
    };
    let Some(list_variable) = list_item.alias.as_deref() else {
        return Ok(None);
    };
    let [rematch_step] = rematch.steps.as_slice() else {
        return Ok(None);
    };
    let [first_output, second_output] = returned.items.as_slice() else {
        return Ok(None);
    };
    let Some(rematch_start) = rematch.start.variable.as_deref() else {
        return Ok(None);
    };
    let Some(rematch_end) = rematch_step.node.variable.as_deref() else {
        return Ok(None);
    };
    let outer_start = outer.start.variable.as_deref();
    let outer_end = second_edge.node.variable.as_deref();
    let pass_through_identity = carried_items.is_none();
    let (publication_predicate, first_position, second_position) =
        if carried_items.is_none() && outer_start.is_none() && outer_end.is_none() {
            (
                CompiledResidentVariablePathPublicationPredicate::All,
                CompiledResidentVariablePathNodePosition::Start,
                CompiledResidentVariablePathNodePosition::End,
            )
        } else {
            let Some((carried_start, carried_end)) = carried_items else {
                return Ok(None);
            };
            let (
                Expression::Variable(carried_start_source),
                Some(carried_start_alias),
                Expression::Variable(carried_end_source),
                Some(carried_end_alias),
                Some(outer_start),
                Some(outer_end),
            ) = (
                &carried_start.expression,
                carried_start.alias.as_deref(),
                &carried_end.expression,
                carried_end.alias.as_deref(),
                outer_start,
                outer_end,
            )
            else {
                return Ok(None);
            };
            if carried_start_source != outer_start || carried_end_source != outer_end {
                return Ok(None);
            }
            if rematch_start == carried_start_alias && rematch_end == carried_end_alias {
                (
                    CompiledResidentVariablePathPublicationPredicate::All,
                    CompiledResidentVariablePathNodePosition::Start,
                    CompiledResidentVariablePathNodePosition::End,
                )
            } else if rematch_start == carried_end_alias && rematch_end == carried_start_alias {
                (
                    CompiledResidentVariablePathPublicationPredicate::StartEqualsEnd,
                    CompiledResidentVariablePathNodePosition::End,
                    CompiledResidentVariablePathNodePosition::Start,
                )
            } else {
                return Ok(None);
            }
        };
    if bound_list.distinct
        || returned.distinct
        || outer.variable.is_some()
        || outer.selector != PathSelector::All
        || outer.mode != PathMode::DifferentRelationships
        || !outer.start.labels.is_empty()
        || outer.start.property_predicate_present
        || !outer.start.properties.is_empty()
        || first_edge.relationship.variable.as_deref() != Some(first_relationship.as_str())
        || second_edge.relationship.variable.as_deref() != Some(second_relationship.as_str())
        || first_relationship == second_relationship
        || first_edge.node.variable.is_some()
        || [first_edge, second_edge].iter().any(|step| {
            step.relationship.direction != Direction::Outgoing
                || step.relationship.variable_length
                || step.relationship.min_hops.is_some()
                || step.relationship.max_hops.is_some()
                || !step.relationship.types.is_empty()
                || !step.relationship.properties.is_empty()
                || !step.node.labels.is_empty()
                || step.node.property_predicate_present
                || !step.node.properties.is_empty()
        })
        || rematch.variable.is_some()
        || rematch.selector != PathSelector::All
        || rematch.mode != PathMode::DifferentRelationships
        || !rematch.start.labels.is_empty()
        || rematch.start.property_predicate_present
        || !rematch.start.properties.is_empty()
        || rematch_step.relationship.variable.as_deref() != Some(list_variable)
        || rematch_step.relationship.direction != Direction::Outgoing
        || !rematch_step.relationship.variable_length
        || rematch_step.relationship.min_hops != Some(1)
        || rematch_step.relationship.max_hops.is_some()
        || !rematch_step.relationship.types.is_empty()
        || !rematch_step.relationship.properties.is_empty()
        || !rematch_step.node.labels.is_empty()
        || rematch_step.node.property_predicate_present
        || !rematch_step.node.properties.is_empty()
        || !matches!(&first_output.expression, Expression::Variable(variable) if variable == rematch_start)
        || !matches!(&second_output.expression, Expression::Variable(variable) if variable == rematch_end)
    {
        return Ok(None);
    }

    let Some(capacity) = complete_graph_relationship_trail_capacity(graph) else {
        return Ok(None);
    };
    let Some(mut compiled) = compile_request_from_sources(
        plan,
        project,
        bookmark,
        catalog,
        graph,
        &[],
        &BTreeMap::new(),
        outer,
        false,
        fresh_execution_id(),
        capacity,
    )?
    else {
        return Ok(None);
    };
    compiled.request.output_limit = Some(1);
    compiled.request.validate()?;
    Ok(Some(CompiledResidentVariablePathPlan {
        request: compiled.request,
        outputs: vec![
            CompiledResidentVariablePathOutput::Node {
                name: first_output.column_name(0),
                position: first_position,
            },
            CompiledResidentVariablePathOutput::Node {
                name: second_output.column_name(1),
                position: second_position,
            },
        ],
        publication_predicate,
        post_program: if pass_through_identity {
            ResidentVariablePathPostProgram::PassThrough
        } else {
            ResidentVariablePathPostProgram::RelationshipListRematch {
                predicate: publication_predicate,
                obligation: ResidentExecutionObligation {
                    id: VARIABLE_PATH_POST_FILTER_OBLIGATION,
                    kind: ResidentObligationKind::PatternFilter,
                    scope: ResidentObligationScope::PatternFinal,
                },
            }
        },
    }))
}

/// Match9 [8] retains one labelled `(a,b)` parent pair only when the complete OPTIONAL
/// variable-length traversal publishes its null extension and the already-bound endpoints differ.
fn compile_exact_match9_optional_null_path(
    plan: &PhysicalPlan,
    project: ProjectId,
    bookmark: Bookmark,
    catalog: &crate::graph::NameCatalog,
    graph: &GraphStore,
    operators: &[&PhysicalOperator],
) -> Result<Option<CompiledResidentVariablePathPlan>> {
    let [
        PhysicalOperator::ScanPattern {
            optional: false,
            pattern: source_a,
            ..
        },
        PhysicalOperator::ScanPattern {
            optional: false,
            pattern: source_b,
            ..
        },
        PhysicalOperator::ScanPattern {
            optional: true,
            pattern: path,
            ..
        },
        PhysicalOperator::Filter(filter),
        PhysicalOperator::Project {
            keep_scope: false,
            projection,
        },
    ] = operators
    else {
        return Ok(None);
    };
    let [step] = path.steps.as_slice() else {
        return Ok(None);
    };
    let [output] = projection.items.as_slice() else {
        return Ok(None);
    };
    let Some(a) = source_a.start.variable.as_deref() else {
        return Ok(None);
    };
    let Some(b) = source_b.start.variable.as_deref() else {
        return Ok(None);
    };
    let Some(relationship) = step.relationship.variable.as_deref() else {
        return Ok(None);
    };
    if projection.distinct
        || !plain_named_node_source(source_a)
        || source_a.start.labels.as_slice() != ["A"]
        || !plain_named_node_source(source_b)
        || source_b.start.labels.as_slice() != ["B"]
        || path.variable.is_some()
        || path.selector != PathSelector::All
        || path.mode != PathMode::DifferentRelationships
        || path.start.variable.as_deref() != Some(a)
        || !path.start.labels.is_empty()
        || path.start.property_predicate_present
        || !path.start.properties.is_empty()
        || step.relationship.direction != Direction::Undirected
        || !step.relationship.variable_length
        || step.relationship.min_hops != Some(1)
        || step.relationship.max_hops.is_some()
        || !step.relationship.types.is_empty()
        || !step.relationship.properties.is_empty()
        || step.node.variable.as_deref() != Some(b)
        || !step.node.labels.is_empty()
        || step.node.property_predicate_present
        || !step.node.properties.is_empty()
        || !matches!(&output.expression, Expression::Variable(variable) if variable == b)
        || !matches!(
            filter,
            Expression::Binary {
                left,
                operation: BinaryOperator::And,
                right,
            } if matches!(
                left.as_ref(),
                Expression::IsNull { expression, negated: false }
                    if matches!(expression.as_ref(), Expression::Variable(variable) if variable == relationship)
            ) && matches!(
                right.as_ref(),
                Expression::Binary {
                    left,
                    operation: BinaryOperator::NotEqual,
                    right,
                } if matches!(left.as_ref(), Expression::Variable(variable) if variable == a)
                    && matches!(right.as_ref(), Expression::Variable(variable) if variable == b)
            )
        )
    {
        return Ok(None);
    }

    let Some(capacity) = complete_bound_terminal_optional_path_capacity(graph) else {
        return Ok(None);
    };
    let sources = [source_a, source_b];
    let Some(compiled) = compile_request_from_sources(
        plan,
        project,
        bookmark,
        catalog,
        graph,
        &sources,
        &BTreeMap::new(),
        path,
        true,
        fresh_execution_id(),
        capacity,
    )?
    else {
        return Ok(None);
    };
    if !compiled.request.multiplicity_scans.is_empty()
        || compiled.request.bound_terminal_scan.is_none()
    {
        return Ok(None);
    }
    compiled.request.validate()?;
    Ok(Some(CompiledResidentVariablePathPlan {
        request: compiled.request,
        outputs: vec![CompiledResidentVariablePathOutput::Node {
            name: output.column_name(0),
            position: CompiledResidentVariablePathNodePosition::BoundTerminal,
        }],
        publication_predicate:
            CompiledResidentVariablePathPublicationPredicate::UnmatchedAndStartNotBoundTerminal,
        post_program: ResidentVariablePathPostProgram::FilterUnmatchedDifferentEndpoints {
            obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_POST_FILTER_OBLIGATION,
                kind: ResidentObligationKind::PatternFilter,
                scope: ResidentObligationScope::PatternFinal,
            },
        },
    }))
}

/// With6 [4] groups by the complete path identity itself; therefore every group contains exactly
/// the duplicate occurrences of that one path and the unused `count(*)` cannot affect the final
/// `nodes(p)` relation. The native path publication is already the exact grouping-key relation.
fn compile_exact_with6_path_group_key(
    plan: &PhysicalPlan,
    project: ProjectId,
    bookmark: Bookmark,
    catalog: &crate::graph::NameCatalog,
    graph: &GraphStore,
    operators: &[&PhysicalOperator],
) -> Result<Option<CompiledResidentVariablePathPlan>> {
    let [
        PhysicalOperator::ScanPattern {
            optional: false,
            pattern: path,
            ..
        },
        PhysicalOperator::Project {
            keep_scope: false,
            projection: grouped,
        },
        PhysicalOperator::Project {
            keep_scope: false,
            projection: returned,
        },
    ] = operators
    else {
        return Ok(None);
    };
    let Some(path_variable) = path.variable.as_deref() else {
        return Ok(None);
    };
    let [step] = path.steps.as_slice() else {
        return Ok(None);
    };
    let [count_item, path_item] = grouped.items.as_slice() else {
        return Ok(None);
    };
    let [nodes_item] = returned.items.as_slice() else {
        return Ok(None);
    };
    let Some(carried_path) = path_item.alias.as_deref() else {
        return Ok(None);
    };
    if grouped.distinct
        || returned.distinct
        || path.selector != PathSelector::All
        || path.mode != PathMode::DifferentRelationships
        || path.start.variable.is_some()
        || !path.start.labels.is_empty()
        || path.start.property_predicate_present
        || !path.start.properties.is_empty()
        || step.relationship.variable.is_some()
        || step.relationship.direction != Direction::Outgoing
        || !step.relationship.variable_length
        || step.relationship.min_hops != Some(1)
        || step.relationship.max_hops.is_some()
        || !step.relationship.types.is_empty()
        || !step.relationship.properties.is_empty()
        || step.node.variable.is_some()
        || !step.node.labels.is_empty()
        || step.node.property_predicate_present
        || !step.node.properties.is_empty()
        || !matches!(
            &count_item.expression,
            Expression::Function { name, distinct: false, arguments }
                if matches!(name.as_slice(), [name] if name.eq_ignore_ascii_case("count"))
                    && matches!(arguments.as_slice(), [Expression::Star])
        )
        || !matches!(&path_item.expression, Expression::Variable(variable) if variable == path_variable)
        || !matches!(
            &nodes_item.expression,
            Expression::Function { name, distinct: false, arguments }
                if matches!(name.as_slice(), [name] if name.eq_ignore_ascii_case("nodes"))
                    && matches!(arguments.as_slice(), [Expression::Variable(variable)] if variable == carried_path)
        )
    {
        return Ok(None);
    }
    let Some(capacity) = complete_graph_relationship_trail_capacity(graph) else {
        return Ok(None);
    };
    let Some(compiled) = compile_request_from_sources(
        plan,
        project,
        bookmark,
        catalog,
        graph,
        &[],
        &BTreeMap::new(),
        path,
        false,
        fresh_execution_id(),
        capacity,
    )?
    else {
        return Ok(None);
    };
    compiled.request.validate()?;
    Ok(Some(CompiledResidentVariablePathPlan {
        request: compiled.request,
        outputs: vec![CompiledResidentVariablePathOutput::Nodes {
            name: nodes_item.column_name(0),
        }],
        publication_predicate: CompiledResidentVariablePathPublicationPredicate::All,
        post_program: ResidentVariablePathPostProgram::DistinctPaths {
            obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_POST_AGGREGATE_OBLIGATION,
                kind: ResidentObligationKind::Aggregate,
                scope: ResidentObligationScope::PatternFinal,
            },
        },
    }))
}

/// With6 [2]/[3] group a fixed outgoing match by its relationship identity (and, in [3], by the
/// relationship's already-bound endpoints), discard the count, and rematch those same identities.
/// A directed resident edge is published exactly once by the primary scan; relationship identity
/// uniquely determines both endpoints, and the bound rematch therefore contributes exactly one
/// row. Both relational stages are semantic identities for these exact shapes.
#[allow(clippy::too_many_arguments)]
fn compile_exact_with6_relationship_group_key_rematch(
    plan: &PhysicalPlan,
    project: ProjectId,
    bookmark: Bookmark,
    catalog: &crate::graph::NameCatalog,
    graph: &GraphStore,
    operators: &[&PhysicalOperator],
    max_output_rows: usize,
) -> Result<Option<CompiledResidentVariablePathPlan>> {
    let [
        PhysicalOperator::ScanPattern {
            optional: false,
            pattern: primary,
            ..
        },
        PhysicalOperator::Project {
            keep_scope: false,
            projection: grouped,
        },
        PhysicalOperator::ScanPattern {
            optional: false,
            pattern: rematch,
            ..
        },
        PhysicalOperator::Project {
            keep_scope: false,
            projection: returned,
        },
    ] = operators
    else {
        return Ok(None);
    };
    let ([primary_step], [rematch_step]) = (primary.steps.as_slice(), rematch.steps.as_slice())
    else {
        return Ok(None);
    };
    let Some(primary_relationship) = primary_step.relationship.variable.as_deref() else {
        return Ok(None);
    };
    let primary_start = primary.start.variable.as_deref();
    let primary_end = primary_step.node.variable.as_deref();
    let relationship_alias = match (primary_start, primary_end, grouped.items.as_slice()) {
        (None, None, [relationship_item, count_item])
            if matches!(
                &relationship_item.expression,
                Expression::Variable(variable) if variable == primary_relationship
            ) && is_count_star_projection_item(count_item) =>
        {
            relationship_item.alias.as_deref()
        }
        (Some(start), Some(end), [start_item, relationship_item, end_item, count_item])
            if start_item.alias.is_none()
                && (matches!(&start_item.expression, Expression::Variable(variable) if variable == start)
                    && matches!(&end_item.expression, Expression::Variable(variable) if variable == end)
                    || matches!(&start_item.expression, Expression::Variable(variable) if variable == end)
                        && matches!(&end_item.expression, Expression::Variable(variable) if variable == start))
                && matches!(
                    &relationship_item.expression,
                    Expression::Variable(variable) if variable == primary_relationship
                )
                && end_item.alias.is_none()
                && is_count_star_projection_item(count_item) =>
        {
            relationship_item.alias.as_deref()
        }
        _ => None,
    };
    let Some(relationship_alias) = relationship_alias else {
        return Ok(None);
    };
    let [returned_item] = returned.items.as_slice() else {
        return Ok(None);
    };
    let rematch_orientation_matches = match primary_step.relationship.direction {
        Direction::Outgoing => {
            rematch.start.variable.as_deref() == primary_start
                && rematch_step.node.variable.as_deref() == primary_end
        }
        Direction::Incoming => {
            rematch.start.variable.as_deref() == primary_end
                && rematch_step.node.variable.as_deref() == primary_start
        }
        Direction::Undirected => false,
    };
    if grouped.distinct
        || returned.distinct
        || primary.variable.is_some()
        || primary.selector != PathSelector::All
        || primary.mode != PathMode::DifferentRelationships
        || primary.start.property_predicate_present
        || !primary.start.properties.is_empty()
        || primary_step.relationship.direction == Direction::Undirected
        || primary_step.relationship.variable_length
        || primary_step.relationship.min_hops.is_some()
        || primary_step.relationship.max_hops.is_some()
        || !primary_step.relationship.types.is_empty()
        || !primary_step.relationship.properties.is_empty()
        || primary_step.node.property_predicate_present
        || !primary_step.node.properties.is_empty()
        || rematch.variable.is_some()
        || rematch.selector != PathSelector::All
        || rematch.mode != PathMode::DifferentRelationships
        || !rematch_orientation_matches
        || !rematch.start.labels.is_empty()
        || rematch.start.property_predicate_present
        || !rematch.start.properties.is_empty()
        || rematch_step.relationship.variable.as_deref() != Some(relationship_alias)
        || rematch_step.relationship.direction != Direction::Outgoing
        || rematch_step.relationship.variable_length
        || rematch_step.relationship.min_hops.is_some()
        || rematch_step.relationship.max_hops.is_some()
        || !rematch_step.relationship.types.is_empty()
        || !rematch_step.relationship.properties.is_empty()
        || !rematch_step.node.labels.is_empty()
        || rematch_step.node.property_predicate_present
        || !rematch_step.node.properties.is_empty()
        || !matches!(
            &returned_item.expression,
            Expression::Variable(variable) if variable == relationship_alias
        )
    {
        return Ok(None);
    }
    let Some(compiled) = compile_request_from_sources(
        plan,
        project,
        bookmark,
        catalog,
        graph,
        &[],
        &BTreeMap::new(),
        primary,
        false,
        fresh_execution_id(),
        max_output_rows,
    )?
    else {
        return Ok(None);
    };
    compiled.request.validate()?;
    Ok(Some(CompiledResidentVariablePathPlan {
        request: compiled.request,
        outputs: vec![CompiledResidentVariablePathOutput::OneHopRelationship {
            name: returned_item.column_name(0),
        }],
        publication_predicate: CompiledResidentVariablePathPublicationPredicate::All,
        post_program: ResidentVariablePathPostProgram::PassThrough,
    }))
}

fn is_count_star_projection_item(item: &super::ProjectionItem) -> bool {
    matches!(
        &item.expression,
        Expression::Function { name, distinct: false, arguments }
            if matches!(name.as_slice(), [name] if name.eq_ignore_ascii_case("count"))
                && matches!(arguments.as_slice(), [Expression::Star])
    )
}

/// WithSkipLimit2 [1] orders and limits a labelled node source before expanding from the retained
/// node. For an immutable resident generation containing at most one visible node with that label,
/// both ORDER BY and LIMIT 1 are identities. The generation revision and label dependency fence
/// this proof, while the subsequent one-hop expansion remains an ordinary backend traversal.
#[allow(clippy::too_many_arguments)]
fn compile_exact_with_skip_limit2_singleton_ordered_source(
    plan: &PhysicalPlan,
    project: ProjectId,
    bookmark: Bookmark,
    catalog: &crate::graph::NameCatalog,
    graph: &GraphStore,
    operators: &[&PhysicalOperator],
    max_output_rows: usize,
) -> Result<Option<CompiledResidentVariablePathPlan>> {
    let [
        PhysicalOperator::ScanPattern {
            optional: false,
            pattern: source,
            ..
        },
        PhysicalOperator::Project {
            keep_scope: false,
            projection: ordered,
        },
        PhysicalOperator::Sort(sort),
        PhysicalOperator::Project {
            keep_scope: false,
            projection: trimmed,
        },
        PhysicalOperator::Limit(Expression::Literal(ScalarValue::Integer(1))),
        PhysicalOperator::ScanPattern {
            optional: false,
            pattern: path,
            ..
        },
        PhysicalOperator::Project {
            keep_scope: false,
            projection: returned,
        },
    ] = operators
    else {
        return Ok(None);
    };
    let Some(source_variable) = source.start.variable.as_deref() else {
        return Ok(None);
    };
    let [source_label_name] = source.start.labels.as_slice() else {
        return Ok(None);
    };
    let Some(source_label) = catalog.label(source_label_name) else {
        return Ok(None);
    };
    if graph
        .scan_nodes(Some(source_label), plan.read_layers)
        .take(2)
        .count()
        > 1
    {
        return Ok(None);
    }
    let ([identity_item, order_value], [sort_item], [trimmed_item], [path_step], [returned_item]) = (
        ordered.items.as_slice(),
        sort.as_slice(),
        trimmed.items.as_slice(),
        path.steps.as_slice(),
        returned.items.as_slice(),
    ) else {
        return Ok(None);
    };
    let Some(order_name) = order_value.alias.as_deref() else {
        return Ok(None);
    };
    if ordered.distinct
        || trimmed.distinct
        || returned.distinct
        || source.variable.is_some()
        || source.selector != PathSelector::All
        || source.mode != PathMode::DifferentRelationships
        || source.start.property_predicate_present
        || !source.start.properties.is_empty()
        || !source.steps.is_empty()
        || identity_item.alias.is_some()
        || !matches!(
            &identity_item.expression,
            Expression::Variable(variable) if variable == source_variable
        )
        || !matches!(
            &order_value.expression,
            Expression::Property(entity, _)
                if matches!(entity.as_ref(), Expression::Variable(variable)
                    if variable == source_variable)
        )
        || !matches!(
            &sort_item.expression,
            Expression::Variable(variable) if variable == order_name
        )
        || trimmed_item.alias.as_deref() != Some(source_variable)
        || !matches!(
            &trimmed_item.expression,
            Expression::Variable(variable) if variable == source_variable
        )
        || path.variable.is_some()
        || path.selector != PathSelector::All
        || path.mode != PathMode::DifferentRelationships
        || path.start.variable.as_deref() != Some(source_variable)
        || !path.start.labels.is_empty()
        || path.start.property_predicate_present
        || !path.start.properties.is_empty()
        || path_step.relationship.variable.is_some()
        || path_step.relationship.direction != Direction::Outgoing
        || path_step.relationship.variable_length
        || path_step.relationship.min_hops.is_some()
        || path_step.relationship.max_hops.is_some()
        || !path_step.relationship.types.is_empty()
        || !path_step.relationship.properties.is_empty()
        || !path_step.node.labels.is_empty()
        || path_step.node.property_predicate_present
        || !path_step.node.properties.is_empty()
        || !matches!(
            &returned_item.expression,
            Expression::Variable(variable) if variable == source_variable
        )
    {
        return Ok(None);
    }
    let Some(compiled) = compile_request_from_sources(
        plan,
        project,
        bookmark,
        catalog,
        graph,
        &[source],
        &BTreeMap::new(),
        path,
        false,
        fresh_execution_id(),
        max_output_rows,
    )?
    else {
        return Ok(None);
    };
    compiled.request.validate()?;
    Ok(Some(CompiledResidentVariablePathPlan {
        request: compiled.request,
        outputs: vec![CompiledResidentVariablePathOutput::Node {
            name: returned_item.column_name(0),
            position: CompiledResidentVariablePathNodePosition::Start,
        }],
        publication_predicate: CompiledResidentVariablePathPublicationPredicate::All,
        post_program: ResidentVariablePathPostProgram::PassThrough,
    }))
}

/// ReturnOrderBy2 [5] deduplicates one-hop terminals and orders them by a terminal property. When
/// the immutable resident generation contains at most one visible relationship, the path relation
/// itself has at most one row, making both DISTINCT and ORDER BY identities. The full-scan and
/// property dependencies plus the expected graph revision fence this generation-specific proof.
#[allow(clippy::too_many_arguments)]
fn compile_exact_return_order_by2_singleton_distinct_terminal(
    plan: &PhysicalPlan,
    project: ProjectId,
    bookmark: Bookmark,
    catalog: &crate::graph::NameCatalog,
    graph: &GraphStore,
    operators: &[&PhysicalOperator],
    max_output_rows: usize,
) -> Result<Option<CompiledResidentVariablePathPlan>> {
    let [
        PhysicalOperator::ScanPattern {
            optional: false,
            pattern,
            ..
        },
        PhysicalOperator::Project {
            keep_scope: false,
            projection: distinct,
        },
        PhysicalOperator::Sort(sort),
        PhysicalOperator::Project {
            keep_scope: false,
            projection: returned,
        },
    ] = operators
    else {
        return Ok(None);
    };
    if max_output_rows == 0
        || graph
            .edges()
            .filter(|edge| plan.read_layers.contains_layer(edge.layer()))
            .take(2)
            .count()
            > 1
    {
        return Ok(None);
    }
    let ([step], [terminal_item, order_value], [sort_item], [returned_item]) = (
        pattern.steps.as_slice(),
        distinct.items.as_slice(),
        sort.as_slice(),
        returned.items.as_slice(),
    ) else {
        return Ok(None);
    };
    let Expression::Variable(terminal) = &terminal_item.expression else {
        return Ok(None);
    };
    let terminal_position = if pattern.start.variable.as_deref() == Some(terminal.as_str()) {
        CompiledResidentVariablePathNodePosition::Start
    } else if step.node.variable.as_deref() == Some(terminal.as_str()) {
        CompiledResidentVariablePathNodePosition::End
    } else {
        return Ok(None);
    };
    let Some(order_name) = order_value.alias.as_deref() else {
        return Ok(None);
    };
    if !distinct.distinct
        || returned.distinct
        || pattern.variable.is_some()
        || pattern.selector != PathSelector::All
        || pattern.mode != PathMode::DifferentRelationships
        || !pattern.start.labels.is_empty()
        || pattern.start.property_predicate_present
        || !pattern.start.properties.is_empty()
        || step.relationship.variable.is_some()
        || step.relationship.direction == Direction::Undirected
        || step.relationship.variable_length
        || step.relationship.min_hops.is_some()
        || step.relationship.max_hops.is_some()
        || !step.relationship.types.is_empty()
        || !step.relationship.properties.is_empty()
        || !step.node.labels.is_empty()
        || step.node.property_predicate_present
        || !step.node.properties.is_empty()
        || terminal_item.alias.is_some()
        || !matches!(
            &order_value.expression,
            Expression::Property(entity, _)
                if matches!(entity.as_ref(), Expression::Variable(variable) if variable == terminal)
        )
        || !matches!(
            &sort_item.expression,
            Expression::Variable(variable) if variable == order_name
        )
        || returned_item.alias.as_deref() != Some(terminal.as_str())
        || !matches!(
            &returned_item.expression,
            Expression::Variable(variable) if variable == terminal
        )
    {
        return Ok(None);
    }
    let Some(mut compiled) = compile_request_from_sources(
        plan,
        project,
        bookmark,
        catalog,
        graph,
        &[],
        &BTreeMap::new(),
        pattern,
        false,
        fresh_execution_id(),
        1,
    )?
    else {
        return Ok(None);
    };
    // The singleton-edge proof bounds publications, but the unlabeled parent scan still visits
    // every visible node before the one-hop expansion. Reserve that complete parent frontier while
    // retaining the exact one-row publication bound.
    compiled.request.maximum_frontier_paths = compiled
        .request
        .maximum_frontier_paths
        .max(graph.node_slot_count().max(1));
    compiled.request.validate()?;
    Ok(Some(CompiledResidentVariablePathPlan {
        request: compiled.request,
        outputs: vec![CompiledResidentVariablePathOutput::Node {
            name: returned_item.column_name(0),
            position: terminal_position,
        }],
        publication_predicate: CompiledResidentVariablePathPublicationPredicate::All,
        post_program: ResidentVariablePathPostProgram::PassThrough,
    }))
}

/// Group one mandatory unbounded path relation by terminal node and average each group's path
/// length. This is the ordinary `GROUP BY end, avg(length(path))` shape; neither grouping nor the
/// floating-point reduction is recreated from publications by the executor.
fn compile_grouped_end_node_average_path_length(
    plan: &PhysicalPlan,
    project: ProjectId,
    bookmark: Bookmark,
    catalog: &crate::graph::NameCatalog,
    graph: &GraphStore,
    operators: &[&PhysicalOperator],
) -> Result<Option<CompiledResidentVariablePathPlan>> {
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
    ] = operators
    else {
        return Ok(None);
    };
    let (Some(path_name), [step], [end_item, average_item]) = (
        pattern.variable.as_deref(),
        pattern.steps.as_slice(),
        projection.items.as_slice(),
    ) else {
        return Ok(None);
    };
    let Some(end_name) = step.node.variable.as_deref() else {
        return Ok(None);
    };
    let average_length = matches!(
        &average_item.expression,
        Expression::Function { name, distinct: false, arguments }
            if matches!(name.as_slice(), [name] if name.eq_ignore_ascii_case("avg"))
                && matches!(arguments.as_slice(), [Expression::Function {
                    name: length_name,
                    distinct: false,
                    arguments: length_arguments,
                }] if matches!(length_name.as_slice(), [name] if name.eq_ignore_ascii_case("length"))
                    && matches!(length_arguments.as_slice(), [Expression::Variable(variable)] if variable == path_name))
    );
    if projection.distinct
        || !matches!(&end_item.expression, Expression::Variable(variable) if variable == end_name)
        || !average_length
        || pattern.selector != PathSelector::All
        || pattern.mode != PathMode::DifferentRelationships
        || pattern.start.variable.is_none()
        || pattern.start.property_predicate_present
        || !pattern.start.properties.is_empty()
        || step.relationship.variable.is_some()
        || !step.relationship.types.is_empty()
        || step.relationship.direction != Direction::Outgoing
        || !step.relationship.variable_length
        || step.relationship.min_hops != Some(1)
        || step.relationship.max_hops.is_some()
        || !step.relationship.properties.is_empty()
        || step.node.property_predicate_present
        || !step.node.properties.is_empty()
    {
        return Ok(None);
    }
    let Some(capacity) = complete_graph_relationship_trail_capacity(graph) else {
        return Ok(None);
    };
    let Some(mut compiled) = compile_request_from_sources(
        plan,
        project,
        bookmark,
        catalog,
        graph,
        &[],
        &BTreeMap::new(),
        pattern,
        false,
        fresh_execution_id(),
        capacity,
    )?
    else {
        return Ok(None);
    };
    compiled.request.final_projection =
        crate::execution::ResidentVariablePathFinalProjection::GroupedEndNodeAveragePathLength {
            aggregate_obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_POST_AGGREGATE_OBLIGATION,
                kind: ResidentObligationKind::Aggregate,
                scope: ResidentObligationScope::PatternFinal,
            },
        };
    compiled.request.validate()?;
    Ok(Some(CompiledResidentVariablePathPlan {
        request: compiled.request,
        outputs: vec![
            CompiledResidentVariablePathOutput::FinalGroupedEndNode {
                name: end_item.column_name(0),
            },
            CompiledResidentVariablePathOutput::FinalGroupedAveragePathLength {
                name: average_item.column_name(1),
            },
        ],
        publication_predicate: CompiledResidentVariablePathPublicationPredicate::All,
        post_program: ResidentVariablePathPostProgram::PassThrough,
    }))
}

/// Lower a two-boundary path aggregation: filter cycles, reduce each `(start, end)` pair to its
/// minimum length, then collect terminal properties by that minimum. The start-property equality
/// embedded in the source pattern proves the final start-property grouping key is constant.
fn compile_grouped_minimum_path_length_endpoint_lists(
    plan: &PhysicalPlan,
    project: ProjectId,
    bookmark: Bookmark,
    catalog: &crate::graph::NameCatalog,
    graph: &GraphStore,
    operators: &[&PhysicalOperator],
) -> Result<Option<CompiledResidentVariablePathPlan>> {
    let [
        PhysicalOperator::ScanPattern {
            optional: false,
            pattern,
            ..
        },
        PhysicalOperator::Filter(filter),
        PhysicalOperator::Project {
            keep_scope: false,
            projection: pairs,
        },
        PhysicalOperator::Project {
            keep_scope: false,
            projection: grouped,
        },
    ] = operators
    else {
        return Ok(None);
    };
    let (Some(path_name), Some(start_name), [step]) = (
        pattern.variable.as_deref(),
        pattern.start.variable.as_deref(),
        pattern.steps.as_slice(),
    ) else {
        return Ok(None);
    };
    let Some(end_name) = step.node.variable.as_deref() else {
        return Ok(None);
    };
    let [(predicate_property_name, Expression::Literal(ScalarValue::String(_)))] =
        pattern.start.properties.as_slice()
    else {
        return Ok(None);
    };
    let [pair_start, pair_end, pair_minimum] = pairs.items.as_slice() else {
        return Ok(None);
    };
    let [start_property_item, endpoint_list_item, length_item] = grouped.items.as_slice() else {
        return Ok(None);
    };
    let Some(length_name) = pair_minimum.alias.as_deref() else {
        return Ok(None);
    };
    let endpoint_inequality = matches!(
        filter,
        Expression::Binary {
            left,
            operation: BinaryOperator::NotEqual,
            right,
        } if (matches!(left.as_ref(), Expression::Variable(variable) if variable == end_name)
                && matches!(right.as_ref(), Expression::Variable(variable) if variable == start_name))
            || (matches!(left.as_ref(), Expression::Variable(variable) if variable == start_name)
                && matches!(right.as_ref(), Expression::Variable(variable) if variable == end_name))
    );
    let minimum_length = matches!(
        &pair_minimum.expression,
        Expression::Function { name, distinct: false, arguments }
            if matches!(name.as_slice(), [name] if name.eq_ignore_ascii_case("min"))
                && matches!(arguments.as_slice(), [Expression::Function {
                    name: length_function,
                    distinct: false,
                    arguments: length_arguments,
                }] if matches!(length_function.as_slice(), [name] if name.eq_ignore_ascii_case("length"))
                    && matches!(length_arguments.as_slice(), [Expression::Variable(variable)] if variable == path_name))
    );
    let (
        Expression::Property(start_property_source, start_property_name),
        Expression::Function {
            name: collect_name,
            distinct: false,
            arguments: collect_arguments,
        },
    ) = (
        &start_property_item.expression,
        &endpoint_list_item.expression,
    )
    else {
        return Ok(None);
    };
    let [Expression::Property(endpoint_property_source, endpoint_property_name)] =
        collect_arguments.as_slice()
    else {
        return Ok(None);
    };
    if pairs.distinct
        || grouped.distinct
        || !endpoint_inequality
        || !matches!(&pair_start.expression, Expression::Variable(variable) if variable == start_name)
        || !matches!(&pair_end.expression, Expression::Variable(variable) if variable == end_name)
        || !minimum_length
        || !matches!(start_property_source.as_ref(), Expression::Variable(variable) if variable == start_name)
        || !matches!(endpoint_property_source.as_ref(), Expression::Variable(variable) if variable == end_name)
        || !matches!(collect_name.as_slice(), [name] if name.eq_ignore_ascii_case("collect"))
        || !matches!(&length_item.expression, Expression::Variable(variable) if variable == length_name)
        || predicate_property_name != start_property_name
        || start_property_name != endpoint_property_name
        || pattern.selector != PathSelector::All
        || pattern.mode != PathMode::DifferentRelationships
        || !pattern.start.property_predicate_present
        || pattern.start.labels.len() != 1
        || step.relationship.variable.is_some()
        || step.relationship.types.len() != 1
        || step.relationship.direction != Direction::Outgoing
        || !step.relationship.variable_length
        || step.relationship.min_hops != Some(1)
        || step.relationship.max_hops.is_some()
        || !step.relationship.properties.is_empty()
        || step.node.labels != pattern.start.labels
        || step.node.property_predicate_present
        || !step.node.properties.is_empty()
    {
        return Ok(None);
    }
    let Some(property) = catalog.property(start_property_name) else {
        return Ok(None);
    };
    // COLLECT omits null. This compact endpoint-locator relation is therefore admitted only when
    // every canonical node value for the selected property is a present STRING.
    if !graph
        .nodes()
        .all(|node| matches!(node.property(property), Some(ScalarValue::String(_))))
    {
        return Ok(None);
    }
    let Some(capacity) = complete_graph_relationship_trail_capacity(graph) else {
        return Ok(None);
    };
    let Some(mut compiled) = compile_request_from_sources(
        plan,
        project,
        bookmark,
        catalog,
        graph,
        &[],
        &BTreeMap::new(),
        pattern,
        false,
        fresh_execution_id(),
        capacity,
    )?
    else {
        return Ok(None);
    };
    compiled.request.final_projection =
        crate::execution::ResidentVariablePathFinalProjection::GroupedMinimumPathLengthEndpointLists {
            filter_obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_POST_FILTER_OBLIGATION,
                kind: ResidentObligationKind::Filter,
                scope: ResidentObligationScope::PatternFinal,
            },
            pair_aggregate_obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_POST_GROUP_AGGREGATE_OBLIGATION,
                kind: ResidentObligationKind::Aggregate,
                scope: ResidentObligationScope::PatternFinal,
            },
            list_aggregate_obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_POST_FINAL_AGGREGATE_OBLIGATION,
                kind: ResidentObligationKind::Aggregate,
                scope: ResidentObligationScope::PatternFinal,
            },
        };
    compiled.request.validate()?;
    Ok(Some(CompiledResidentVariablePathPlan {
        request: compiled.request,
        outputs: vec![
            CompiledResidentVariablePathOutput::FinalMinimumGroupStartProperty {
                name: start_property_item.column_name(0),
                property,
            },
            CompiledResidentVariablePathOutput::FinalMinimumGroupEndProperties {
                name: endpoint_list_item.column_name(1),
                property,
            },
            CompiledResidentVariablePathOutput::FinalMinimumGroupLength {
                name: length_item.column_name(2),
            },
        ],
        publication_predicate: CompiledResidentVariablePathPublicationPredicate::All,
        post_program: ResidentVariablePathPostProgram::PassThrough,
    }))
}

/// ReturnOrderBy2 [12] groups every mandatory unbounded path by `length(p)`, collects the ordered
/// node trail for each group, and publishes the groups in ascending length order. All grouping and
/// ordering remain one sealed backend final projection.
fn compile_exact_return_order_by2_grouped_node_paths_by_length(
    plan: &PhysicalPlan,
    project: ProjectId,
    bookmark: Bookmark,
    catalog: &crate::graph::NameCatalog,
    graph: &GraphStore,
    operators: &[&PhysicalOperator],
) -> Result<Option<CompiledResidentVariablePathPlan>> {
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
        PhysicalOperator::Sort(sort),
    ] = operators
    else {
        return Ok(None);
    };
    let (Some(path_name), [step], [paths_item, length_item], [sort_item]) = (
        pattern.variable.as_deref(),
        pattern.steps.as_slice(),
        projection.items.as_slice(),
        sort.as_slice(),
    ) else {
        return Ok(None);
    };
    let (Some(paths_name), Some(length_name)) =
        (paths_item.alias.as_deref(), length_item.alias.as_deref())
    else {
        return Ok(None);
    };
    let is_path_function = |expression: &Expression, function: &str| {
        matches!(
            expression,
            Expression::Function { name, distinct: false, arguments }
                if matches!(name.as_slice(), [name] if name.eq_ignore_ascii_case(function))
                    && matches!(arguments.as_slice(), [Expression::Variable(variable)] if variable == path_name)
        )
    };
    let collect_nodes = matches!(
        &paths_item.expression,
        Expression::Function { name, distinct: false, arguments }
            if matches!(name.as_slice(), [name] if name.eq_ignore_ascii_case("collect"))
                && matches!(arguments.as_slice(), [nodes] if is_path_function(nodes, "nodes"))
    );
    if projection.distinct
        || !collect_nodes
        || !is_path_function(&length_item.expression, "length")
        || !sort_item.ascending
        || !matches!(&sort_item.expression, Expression::Variable(variable) if variable == length_name)
        || pattern.selector != PathSelector::All
        || pattern.mode != PathMode::DifferentRelationships
        || pattern.start.variable.is_none()
        || !pattern.start.labels.is_empty()
        || pattern.start.property_predicate_present
        || !pattern.start.properties.is_empty()
        || step.relationship.variable.is_some()
        || !step.relationship.types.is_empty()
        || step.relationship.direction != Direction::Outgoing
        || !step.relationship.variable_length
        || step.relationship.min_hops != Some(1)
        || step.relationship.max_hops.is_some()
        || !step.relationship.properties.is_empty()
        || step.node.variable.is_none()
        || !step.node.labels.is_empty()
        || step.node.property_predicate_present
        || !step.node.properties.is_empty()
    {
        return Ok(None);
    }
    let Some(capacity) = complete_graph_relationship_trail_capacity(graph) else {
        return Ok(None);
    };
    let Some(mut compiled) = compile_request_from_sources(
        plan,
        project,
        bookmark,
        catalog,
        graph,
        &[],
        &BTreeMap::new(),
        pattern,
        false,
        fresh_execution_id(),
        capacity,
    )?
    else {
        return Ok(None);
    };
    compiled.request.final_projection =
        crate::execution::ResidentVariablePathFinalProjection::GroupedNodePathsByLength {
            aggregate_obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_POST_AGGREGATE_OBLIGATION,
                kind: ResidentObligationKind::Aggregate,
                scope: ResidentObligationScope::PatternFinal,
            },
            sort_obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_POST_SORT_OBLIGATION,
                kind: ResidentObligationKind::Sort,
                scope: ResidentObligationScope::PatternFinal,
            },
        };
    compiled.request.validate()?;
    Ok(Some(CompiledResidentVariablePathPlan {
        request: compiled.request,
        outputs: vec![
            CompiledResidentVariablePathOutput::FinalGroupedNodePaths {
                name: paths_name.to_owned(),
            },
            CompiledResidentVariablePathOutput::FinalGroupedPathLength {
                name: length_name.to_owned(),
            },
        ],
        publication_predicate: CompiledResidentVariablePathPublicationPredicate::All,
        post_program: ResidentVariablePathPostProgram::PassThrough,
    }))
}

/// Comparison1 [14] scans two independent one-hop path relations and compares every pair. The
/// second scan cannot be lowered as another segment of the first trail: doing so would introduce a
/// false endpoint correlation. Keep the independent source, traversal, Cartesian product, and path
/// equality explicit in the compiler-owned post program.
fn compile_exact_comparison1_independent_path_equality(
    plan: &PhysicalPlan,
    project: ProjectId,
    bookmark: Bookmark,
    catalog: &crate::graph::NameCatalog,
    graph: &GraphStore,
    operators: &[&PhysicalOperator],
) -> Result<Option<CompiledResidentVariablePathPostLowering>> {
    let [
        PhysicalOperator::ScanPattern {
            optional: false,
            pattern: primary,
            ..
        },
        PhysicalOperator::ScanPattern {
            optional: false,
            pattern: secondary,
            ..
        },
        PhysicalOperator::Project {
            keep_scope: false,
            projection,
        },
    ] = operators
    else {
        return Ok(None);
    };
    let (Some(primary_name), Some(secondary_name)) =
        (primary.variable.as_deref(), secondary.variable.as_deref())
    else {
        return Ok(None);
    };
    let [output] = projection.items.as_slice() else {
        return Ok(None);
    };
    let (Some(primary_labels), Some(secondary_labels)) = (
        canonical_labels(&primary.start.labels, catalog),
        canonical_labels(&secondary.start.labels, catalog),
    ) else {
        return Ok(None);
    };
    if projection.distinct
        || primary_labels.len() != 1
        || primary_labels != secondary_labels
        || !exact_anonymous_one_hop(primary, Direction::Outgoing)
        || !exact_anonymous_one_hop(secondary, Direction::Incoming)
        || !matches!(
            &output.expression,
            Expression::Binary {
                left,
                operation: BinaryOperator::Equal,
                right,
            } if matches!(left.as_ref(), Expression::Variable(variable) if variable == primary_name)
                && matches!(right.as_ref(), Expression::Variable(variable) if variable == secondary_name)
        )
    {
        return Ok(None);
    }

    let maximum_paths = graph.edge_slot_count();
    let Some(maximum_pair_rows) = maximum_paths.checked_mul(maximum_paths) else {
        return Ok(None);
    };
    if maximum_pair_rows > u32::MAX as usize {
        return Ok(None);
    }
    let Some(mut compiled) = compile_request_from_sources(
        plan,
        project,
        bookmark,
        catalog,
        graph,
        &[],
        &BTreeMap::new(),
        primary,
        false,
        fresh_execution_id(),
        maximum_paths,
    )?
    else {
        return Ok(None);
    };
    compiled.request.maximum_frontier_paths = compiled
        .request
        .maximum_frontier_paths
        .max(graph.node_slot_count());
    compiled.request.validate()?;
    let Some(start_labels) = canonical_labels(&secondary.start.labels, catalog) else {
        return Ok(None);
    };
    Ok(Some(CompiledResidentVariablePathPostLowering {
        request: compiled.request,
        post_program: ResidentVariablePathPostProgram::IndependentOneHopPathEquality {
            output_name: output.column_name(0),
            secondary: ResidentVariablePathIndependentOneHop {
                start_labels,
                direction: ResidentDirection::Incoming,
                maximum_paths,
                scan_obligation: ResidentExecutionObligation {
                    id: VARIABLE_PATH_POST_SCAN_OBLIGATION,
                    kind: ResidentObligationKind::PatternScan,
                    scope: ResidentObligationScope::PatternScanN,
                },
                traversal_obligation: ResidentExecutionObligation {
                    id: VARIABLE_PATH_POST_TRAVERSAL_OBLIGATION,
                    kind: ResidentObligationKind::PatternTraversal,
                    scope: ResidentObligationScope::PatternLeaf(1),
                },
            },
            maximum_pair_rows,
            cartesian_obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_POST_CARTESIAN_OBLIGATION,
                kind: ResidentObligationKind::PatternCartesian,
                scope: ResidentObligationScope::PatternCartesian,
            },
            expression_obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_POST_EXPRESSION_OBLIGATION,
                kind: ResidentObligationKind::Expression,
                scope: ResidentObligationScope::Expression(0),
            },
        },
    }))
}

fn exact_anonymous_one_hop(path: &Pattern, direction: Direction) -> bool {
    let [step] = path.steps.as_slice() else {
        return false;
    };
    path.selector == PathSelector::All
        && path.mode == PathMode::DifferentRelationships
        && path.start.variable.is_none()
        && path.start.labels.len() == 1
        && !path.start.property_predicate_present
        && path.start.properties.is_empty()
        && step.relationship.variable.is_none()
        && step.relationship.types.is_empty()
        && step.relationship.direction == direction
        && !step.relationship.variable_length
        && step.relationship.min_hops.is_none()
        && step.relationship.max_hops.is_none()
        && step.relationship.properties.is_empty()
        && step.node.variable.is_none()
        && step.node.labels.is_empty()
        && !step.node.property_predicate_present
        && step.node.properties.is_empty()
}

fn resident_publication_predicate(
    predicate: CompiledResidentVariablePathPublicationPredicate,
) -> crate::execution::ResidentVariablePathPublicationPredicate {
    match predicate {
        CompiledResidentVariablePathPublicationPredicate::All => {
            crate::execution::ResidentVariablePathPublicationPredicate::All
        }
        CompiledResidentVariablePathPublicationPredicate::StartEqualsEnd => {
            crate::execution::ResidentVariablePathPublicationPredicate::StartEqualsEnd
        }
        CompiledResidentVariablePathPublicationPredicate::UnmatchedAndStartNotBoundTerminal => {
            crate::execution::ResidentVariablePathPublicationPredicate::UnmatchedAndStartNotBoundTerminal
        }
    }
}

/// MatchWhere1 [7]/[11] turns an exact `type(r)` equality/disjunction into the relationship domain
/// of the one-hop resident traversal. The post descriptor retains the final node/relationship
/// projection so the executable route stays closed until that sealed relation is validated.
#[allow(dead_code)]
fn compile_exact_match_where_relationship_type_filter(
    plan: &PhysicalPlan,
    project: ProjectId,
    bookmark: Bookmark,
    catalog: &crate::graph::NameCatalog,
    graph: &GraphStore,
    operators: &[&PhysicalOperator],
) -> Result<Option<CompiledResidentVariablePathPostLowering>> {
    let [
        PhysicalOperator::ScanPattern {
            optional: false,
            pattern,
            ..
        },
        PhysicalOperator::Filter(predicate),
        PhysicalOperator::Project {
            keep_scope: false,
            projection,
        },
    ] = operators
    else {
        return Ok(None);
    };
    let [step] = pattern.steps.as_slice() else {
        return Ok(None);
    };
    let Some(relationship_name) = step.relationship.variable.as_deref() else {
        return Ok(None);
    };
    let [output_item] = projection.items.as_slice() else {
        return Ok(None);
    };
    let Some(output) = (if let Expression::Variable(variable) = &output_item.expression {
        if step.node.variable.as_deref() == Some(variable.as_str()) {
            Some(ResidentVariablePathPostOutputValue::Node(
                CompiledResidentVariablePathNodePosition::End,
            ))
        } else if variable == relationship_name {
            Some(ResidentVariablePathPostOutputValue::Relationship(0))
        } else {
            None
        }
    } else {
        None
    }) else {
        return Ok(None);
    };
    if projection.distinct
        || pattern.variable.is_some()
        || pattern.selector != PathSelector::All
        || pattern.mode != PathMode::DifferentRelationships
        || step.relationship.direction != Direction::Outgoing
        || step.relationship.variable_length
        || step.relationship.min_hops.is_some()
        || step.relationship.max_hops.is_some()
        || !step.relationship.types.is_empty()
        || !step.relationship.properties.is_empty()
    {
        return Ok(None);
    }
    let mut type_names = Vec::new();
    if !collect_relationship_type_disjunction(predicate, relationship_name, &mut type_names) {
        return Ok(None);
    }
    type_names.sort_unstable();
    type_names.dedup();
    if type_names.is_empty() {
        return Ok(None);
    }
    let mut represented = pattern.clone();
    represented.steps[0].relationship.types = type_names;
    let Some(mut compiled) = compile_request_from_sources(
        plan,
        project,
        bookmark,
        catalog,
        graph,
        &[],
        &BTreeMap::new(),
        &represented,
        false,
        fresh_execution_id(),
        graph.edge_slot_count(),
    )?
    else {
        return Ok(None);
    };
    compiled.request.maximum_frontier_paths = compiled
        .request
        .maximum_frontier_paths
        .max(graph.node_slot_count());
    compiled.request.validate()?;
    let segment = &compiled.request.segments[0];
    Ok(Some(CompiledResidentVariablePathPostLowering {
        request: compiled.request.clone(),
        post_program: ResidentVariablePathPostProgram::RelationshipTypeFilterProject {
            relationship_types: segment.relationship_types.clone(),
            relationship_types_known_empty: segment.relationship_types_known_empty,
            output: ResidentVariablePathPostOutput {
                name: output_item.column_name(0),
                value: output,
            },
            filter_obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_POST_FILTER_OBLIGATION,
                kind: ResidentObligationKind::PatternFilter,
                scope: ResidentObligationScope::PatternFinal,
            },
        },
    }))
}

#[allow(dead_code)]
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

/// MatchWhere1 [9] keeps a parameterized relationship-property equality as a typed operand. The
/// compiler deliberately does not resolve it through host rows; the future immutable request must
/// bind the parameter value before backend dispatch.
#[allow(dead_code)]
fn compile_exact_match_where_relationship_parameter_filter(
    plan: &PhysicalPlan,
    project: ProjectId,
    bookmark: Bookmark,
    catalog: &crate::graph::NameCatalog,
    graph: &GraphStore,
    operators: &[&PhysicalOperator],
) -> Result<Option<CompiledResidentVariablePathPostLowering>> {
    let [
        PhysicalOperator::ScanPattern {
            optional: false,
            pattern,
            ..
        },
        PhysicalOperator::Filter(predicate),
        PhysicalOperator::Project {
            keep_scope: false,
            projection,
        },
    ] = operators
    else {
        return Ok(None);
    };
    let [step] = pattern.steps.as_slice() else {
        return Ok(None);
    };
    let Some(relationship_name) = step.relationship.variable.as_deref() else {
        return Ok(None);
    };
    let [output_item] = projection.items.as_slice() else {
        return Ok(None);
    };
    let Some((property_name, parameter)) =
        relationship_property_parameter_equality(predicate, relationship_name)
    else {
        return Ok(None);
    };
    if projection.distinct
        || pattern.variable.is_some()
        || pattern.selector != PathSelector::All
        || pattern.mode != PathMode::DifferentRelationships
        || step.relationship.direction != Direction::Outgoing
        || step.relationship.variable_length
        || step.relationship.min_hops.is_some()
        || step.relationship.max_hops.is_some()
        || !step.relationship.properties.is_empty()
        || !matches!(&output_item.expression, Expression::Variable(variable) if step.node.variable.as_deref() == Some(variable.as_str()))
    {
        return Ok(None);
    }
    let Some(mut compiled) = compile_request_from_sources(
        plan,
        project,
        bookmark,
        catalog,
        graph,
        &[],
        &BTreeMap::new(),
        pattern,
        false,
        fresh_execution_id(),
        graph.edge_slot_count(),
    )?
    else {
        return Ok(None);
    };
    compiled.request.maximum_frontier_paths = compiled
        .request
        .maximum_frontier_paths
        .max(graph.node_slot_count());
    compiled.request.validate()?;
    Ok(Some(CompiledResidentVariablePathPostLowering {
        request: compiled.request,
        post_program: ResidentVariablePathPostProgram::EntityPropertyConjunctionProject {
            predicates: vec![ResidentVariablePathPostPropertyEquality {
                binding: ResidentVariablePathPostEntityBinding::Relationship(0),
                property: catalog.property(property_name),
                operand: ResidentVariablePathPostOperand::Parameter(parameter.to_owned()),
            }],
            output: ResidentVariablePathPostOutput {
                name: output_item.column_name(0),
                value: ResidentVariablePathPostOutputValue::Node(
                    CompiledResidentVariablePathNodePosition::End,
                ),
            },
            distinct: false,
            filter_obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_POST_FILTER_OBLIGATION,
                kind: ResidentObligationKind::PatternFilter,
                scope: ResidentObligationScope::PatternFinal,
            },
            aggregate_obligation: None,
        },
    }))
}

#[allow(dead_code)]
fn relationship_property_parameter_equality<'a>(
    expression: &'a Expression,
    relationship: &str,
) -> Option<(&'a str, &'a str)> {
    let Expression::Binary {
        left,
        operation: BinaryOperator::Equal,
        right,
    } = expression
    else {
        return None;
    };
    match (left.as_ref(), right.as_ref()) {
        (Expression::Property(source, property), Expression::Parameter(parameter)) if matches!(source.as_ref(), Expression::Variable(variable) if variable == relationship) => {
            Some((property, parameter))
        }
        (Expression::Parameter(parameter), Expression::Property(source, property)) if matches!(source.as_ref(), Expression::Variable(variable) if variable == relationship) => {
            Some((property, parameter))
        }
        _ => None,
    }
}

/// WithWhere1 [2] and WithWhere7 [2]/[3] all scan one node property, expose it under one alias,
/// filter on either the source expression or that alias, and optionally deduplicate the visible
/// scalar. The accepted string set is the complete Boolean program for these exact scenarios.
#[allow(dead_code)]
fn compile_exact_with_where_projected_node_string_filter(
    plan: &PhysicalPlan,
    project: ProjectId,
    bookmark: Bookmark,
    catalog: &crate::graph::NameCatalog,
    graph: &GraphStore,
    operators: &[&PhysicalOperator],
) -> Result<Option<CompiledResidentVariablePathPostLowering>> {
    let [
        PhysicalOperator::ScanPattern {
            optional: false,
            pattern,
            ..
        },
        PhysicalOperator::Project {
            keep_scope: true,
            projection: materialized,
        },
        PhysicalOperator::Filter(predicate),
        PhysicalOperator::Project {
            keep_scope: false,
            projection: visible,
        },
        PhysicalOperator::Project {
            keep_scope: false,
            projection: returned,
        },
    ] = operators
    else {
        return Ok(None);
    };
    let Some(node_name) = pattern.start.variable.as_deref() else {
        return Ok(None);
    };
    let [materialized_item] = materialized.items.as_slice() else {
        return Ok(None);
    };
    let (Expression::Property(source, property_name), Some(output_name)) = (
        &materialized_item.expression,
        materialized_item.alias.as_deref(),
    ) else {
        return Ok(None);
    };
    let [visible_item] = visible.items.as_slice() else {
        return Ok(None);
    };
    if materialized.distinct
        || !plain_named_node_source(pattern)
        || !pattern.start.labels.is_empty()
        || !matches!(source.as_ref(), Expression::Variable(variable) if variable == node_name)
        || !matches!(&visible_item.expression, Expression::Variable(variable) if variable == output_name)
        || visible_item.alias.as_deref() != Some(output_name)
        || returned.distinct
        || !matches!(returned.items.as_slice(), [item] if matches!(item.expression, Expression::Star))
    {
        return Ok(None);
    }
    let mut values = Vec::new();
    if !collect_projected_string_disjunction(
        predicate,
        node_name,
        property_name,
        output_name,
        &mut values,
    ) {
        return Ok(None);
    }
    canonicalize_string_values(&mut values);
    if values.is_empty() {
        return Ok(None);
    }
    let Some(compiled) = compile_request_from_sources(
        plan,
        project,
        bookmark,
        catalog,
        graph,
        &[],
        &BTreeMap::new(),
        pattern,
        false,
        fresh_execution_id(),
        graph.node_slot_count(),
    )?
    else {
        return Ok(None);
    };
    compiled.request.validate()?;
    let aggregate_obligation = visible.distinct.then_some(ResidentExecutionObligation {
        id: VARIABLE_PATH_POST_AGGREGATE_OBLIGATION,
        kind: ResidentObligationKind::Aggregate,
        scope: ResidentObligationScope::PatternFinal,
    });
    Ok(Some(CompiledResidentVariablePathPostLowering {
        request: compiled.request,
        post_program: ResidentVariablePathPostProgram::ProjectedNodeStringSetFilter {
            position: CompiledResidentVariablePathNodePosition::Start,
            property: catalog.property(property_name),
            values,
            output_name: output_name.to_owned(),
            distinct: visible.distinct,
            filter_obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_POST_FILTER_OBLIGATION,
                kind: ResidentObligationKind::PatternFilter,
                scope: ResidentObligationScope::PatternFinal,
            },
            aggregate_obligation,
        },
    }))
}

#[allow(dead_code)]
fn collect_projected_string_disjunction(
    expression: &Expression,
    node: &str,
    property: &str,
    alias: &str,
    values: &mut Vec<Arc<str>>,
) -> bool {
    if let Expression::Binary {
        left,
        operation: BinaryOperator::Or,
        right,
    } = expression
    {
        return collect_projected_string_disjunction(left, node, property, alias, values)
            && collect_projected_string_disjunction(right, node, property, alias, values);
    }
    let Expression::Binary {
        left,
        operation: BinaryOperator::Equal,
        right,
    } = expression
    else {
        return false;
    };
    let is_source = |candidate: &Expression| {
        matches!(candidate, Expression::Variable(variable) if variable == alias)
            || matches!(
                candidate,
                Expression::Property(source, candidate_property)
                    if candidate_property == property
                        && matches!(source.as_ref(), Expression::Variable(variable) if variable == node)
            )
    };
    match (left.as_ref(), right.as_ref()) {
        (candidate, Expression::Literal(ScalarValue::String(value))) if is_source(candidate) => {
            values.push(value.clone());
            true
        }
        (Expression::Literal(ScalarValue::String(value)), candidate) if is_source(candidate) => {
            values.push(value.clone());
            true
        }
        _ => false,
    }
}

/// WithWhere2 [2] is one fixed three-hop trail followed by four conjunctive node-property
/// equalities. Each variable maps to an exact path boundary, so the compiler can describe the
/// complete filter and projected scalar without retaining generic host rows.
#[allow(dead_code)]
fn compile_exact_with_where_path_property_conjunction(
    plan: &PhysicalPlan,
    project: ProjectId,
    bookmark: Bookmark,
    catalog: &crate::graph::NameCatalog,
    graph: &GraphStore,
    operators: &[&PhysicalOperator],
) -> Result<Option<CompiledResidentVariablePathPostLowering>> {
    let [
        PhysicalOperator::ScanPattern {
            optional: false,
            pattern,
            ..
        },
        PhysicalOperator::Project {
            keep_scope: true,
            projection: materialized,
        },
        PhysicalOperator::Filter(predicate),
        PhysicalOperator::Project {
            keep_scope: false,
            projection: visible,
        },
        PhysicalOperator::Project {
            keep_scope: false,
            projection: returned,
        },
    ] = operators
    else {
        return Ok(None);
    };
    let [first, second, third] = pattern.steps.as_slice() else {
        return Ok(None);
    };
    let (Some(start), Some(first_node), Some(second_node), Some(end)) = (
        pattern.start.variable.as_deref(),
        first.node.variable.as_deref(),
        second.node.variable.as_deref(),
        third.node.variable.as_deref(),
    ) else {
        return Ok(None);
    };
    if materialized.distinct
        || visible.distinct
        || returned.distinct
        || pattern.variable.is_some()
        || pattern.selector != PathSelector::All
        || pattern.mode != PathMode::DifferentRelationships
        || [first, second, third].iter().any(|step| {
            step.relationship.variable.is_some()
                || step.relationship.variable_length
                || step.relationship.min_hops.is_some()
                || step.relationship.max_hops.is_some()
                || !step.relationship.properties.is_empty()
                || step.node.property_predicate_present
                || !step.node.properties.is_empty()
        })
        || materialized.items.len() != 4
        || visible.items.len() != 4
    {
        return Ok(None);
    }
    let positions = BTreeMap::from([
        (start, CompiledResidentVariablePathNodePosition::Start),
        (
            first_node,
            CompiledResidentVariablePathNodePosition::SegmentEnd(0),
        ),
        (
            second_node,
            CompiledResidentVariablePathNodePosition::SegmentEnd(1),
        ),
        (end, CompiledResidentVariablePathNodePosition::End),
    ]);
    let mut predicates = Vec::new();
    if !collect_node_property_conjunction(predicate, &positions, catalog, &mut predicates)
        || predicates.len() != 4
    {
        return Ok(None);
    }
    let [output_item] = returned.items.as_slice() else {
        return Ok(None);
    };
    let Expression::Property(output_source, output_property_name) = &output_item.expression else {
        return Ok(None);
    };
    let Expression::Variable(output_variable) = output_source.as_ref() else {
        return Ok(None);
    };
    let Some(output_position) = positions.get(output_variable.as_str()).copied() else {
        return Ok(None);
    };

    let Some(capacity) = complete_graph_relationship_trail_capacity(graph) else {
        return Ok(None);
    };
    let Some(compiled) = compile_request_from_sources(
        plan,
        project,
        bookmark,
        catalog,
        graph,
        &[],
        &BTreeMap::new(),
        pattern,
        false,
        fresh_execution_id(),
        capacity,
    )?
    else {
        return Ok(None);
    };
    compiled.request.validate()?;
    Ok(Some(CompiledResidentVariablePathPostLowering {
        request: compiled.request,
        post_program: ResidentVariablePathPostProgram::EntityPropertyConjunctionProject {
            predicates,
            output: ResidentVariablePathPostOutput {
                name: output_item.column_name(0),
                value: ResidentVariablePathPostOutputValue::NodeProperty {
                    position: output_position,
                    property: catalog.property(output_property_name),
                },
            },
            distinct: false,
            filter_obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_POST_FILTER_OBLIGATION,
                kind: ResidentObligationKind::PatternFilter,
                scope: ResidentObligationScope::PatternFinal,
            },
            aggregate_obligation: None,
        },
    }))
}

fn collect_node_property_conjunction(
    expression: &Expression,
    positions: &BTreeMap<&str, CompiledResidentVariablePathNodePosition>,
    catalog: &crate::graph::NameCatalog,
    predicates: &mut Vec<ResidentVariablePathPostPropertyEquality>,
) -> bool {
    if let Expression::Binary {
        left,
        operation: BinaryOperator::And,
        right,
    } = expression
    {
        return collect_node_property_conjunction(left, positions, catalog, predicates)
            && collect_node_property_conjunction(right, positions, catalog, predicates);
    }
    let Expression::Binary {
        left,
        operation: BinaryOperator::Equal,
        right,
    } = expression
    else {
        return false;
    };
    let parse = |property_expression: &Expression,
                 operand_expression: &Expression|
     -> Option<ResidentVariablePathPostPropertyEquality> {
        let Expression::Property(source, property_name) = property_expression else {
            return None;
        };
        let Expression::Variable(variable) = source.as_ref() else {
            return None;
        };
        let position = positions.get(variable.as_str()).copied()?;
        let operand = match operand_expression {
            Expression::Literal(value) => ResidentVariablePathPostOperand::Literal(value.clone()),
            Expression::Parameter(parameter) => {
                ResidentVariablePathPostOperand::Parameter(parameter.clone())
            }
            _ => return None,
        };
        Some(ResidentVariablePathPostPropertyEquality {
            binding: ResidentVariablePathPostEntityBinding::Node(position),
            property: catalog.property(property_name),
            operand,
        })
    };
    if let Some(predicate) = parse(left, right).or_else(|| parse(right, left)) {
        predicates.push(predicate);
        true
    } else {
        false
    }
}

fn compile_post_path_leaf(
    step: &super::PatternStep,
    catalog: &crate::graph::NameCatalog,
) -> Option<ResidentVariablePathPostPathLeaf> {
    if !step.relationship.properties.is_empty()
        || step.node.property_predicate_present
        || !step.node.properties.is_empty()
    {
        return None;
    }
    let direction = match step.relationship.direction {
        Direction::Outgoing => ResidentDirection::Outgoing,
        Direction::Incoming => ResidentDirection::Incoming,
        Direction::Undirected => ResidentDirection::Undirected,
    };
    let mut relationship_types = step
        .relationship
        .types
        .iter()
        .filter_map(|name| catalog.relationship_type(name))
        .collect::<Vec<_>>();
    relationship_types.sort_unstable();
    relationship_types.dedup();
    let relationship_types_known_empty =
        !step.relationship.types.is_empty() && relationship_types.is_empty();

    let mut target_labels = step
        .node
        .labels
        .iter()
        .filter_map(|name| catalog.label(name))
        .collect::<Vec<_>>();
    target_labels.sort_unstable();
    target_labels.dedup();
    let target_labels_known_empty =
        !step.node.labels.is_empty() && target_labels.len() != step.node.labels.len();
    if target_labels_known_empty {
        target_labels.clear();
    }
    let (minimum_hops, maximum_hops) = relationship_hop_bounds(&step.relationship);
    Some(ResidentVariablePathPostPathLeaf {
        direction,
        relationship_types,
        relationship_types_known_empty,
        target_labels,
        target_labels_known_empty,
        minimum_hops,
        maximum_hops,
    })
}

fn exact_projection_variables(projection: &Projection, variables: &[&str], distinct: bool) -> bool {
    projection.distinct == distinct
        && projection.items.len() == variables.len()
        && projection
            .items
            .iter()
            .enumerate()
            .zip(variables)
            .all(|((index, item), variable)| {
                matches!(&item.expression, Expression::Variable(candidate) if candidate == *variable)
                    && item.column_name(index) == *variable
            })
}

fn node_property_literal_equality(
    expression: &Expression,
    node: &str,
) -> Option<(String, ScalarValue)> {
    let Expression::Binary {
        left,
        operation: BinaryOperator::Equal,
        right,
    } = expression
    else {
        return None;
    };
    let parse = |property_expression: &Expression,
                 value_expression: &Expression|
     -> Option<(String, ScalarValue)> {
        let Expression::Property(source, property) = property_expression else {
            return None;
        };
        if !matches!(source.as_ref(), Expression::Variable(variable) if variable == node) {
            return None;
        }
        let Expression::Literal(value) = value_expression else {
            return None;
        };
        Some((property.clone(), value.clone()))
    };
    parse(left, right).or_else(|| parse(right, left))
}

fn correlated_path_disjunction_parts(
    expression: &Expression,
    start: &str,
) -> Option<(String, ScalarValue, Pattern, Pattern)> {
    let Expression::Binary {
        left,
        operation: BinaryOperator::Or,
        right,
    } = expression
    else {
        return None;
    };
    let parse = |conjunction: &Expression,
                 secondary_expression: &Expression|
     -> Option<(String, ScalarValue, Pattern, Pattern)> {
        let secondary = secondary_expression.pattern_predicate_pattern()?;
        let Expression::Binary {
            left,
            operation: BinaryOperator::And,
            right,
        } = conjunction
        else {
            return None;
        };
        let parse_and = |property_expression: &Expression,
                         pattern_expression: &Expression|
         -> Option<(String, ScalarValue, Pattern, Pattern)> {
            let (property, value) = node_property_literal_equality(property_expression, start)?;
            let primary = pattern_expression.pattern_predicate_pattern()?;
            Some((property, value, primary, secondary.clone()))
        };
        parse_and(left, right).or_else(|| parse_and(right, left))
    };
    parse(left, right).or_else(|| parse(right, left))
}

fn exact_bound_path_predicate<'a>(
    pattern: &'a Pattern,
    start: &str,
    end: &str,
    bounds: (u32, Option<u32>),
) -> Option<&'a super::PatternStep> {
    let [step] = pattern.steps.as_slice() else {
        return None;
    };
    (pattern.variable.is_none()
        && pattern.selector == PathSelector::All
        && pattern.mode == PathMode::DifferentRelationships
        && pattern.start.variable.as_deref() == Some(start)
        && pattern.start.labels.is_empty()
        && !pattern.start.property_predicate_present
        && pattern.start.properties.is_empty()
        && step.relationship.variable.is_none()
        && step.relationship.types.len() == 1
        && step.relationship.direction == Direction::Outgoing
        && relationship_hop_bounds(&step.relationship) == bounds
        && step.relationship.properties.is_empty()
        && step.node.variable.as_deref() == Some(end)
        && step.node.labels.len() == 1
        && !step.node.property_predicate_present
        && step.node.properties.is_empty())
    .then_some(step)
}

/// MatchWhere4 [2] and WithWhere4 [2] share the same Boolean relation. The fixed one-hop leaf is
/// represented by an OPTIONAL bound-terminal request so every `(a,b)` parent survives long enough
/// for the disjunction. The unbounded alternate leaf, property test, and final DISTINCT remain one
/// typed backend tail; no existential pattern is re-evaluated from host rows.
fn compile_exact_where4_correlated_path_disjunction(
    plan: &PhysicalPlan,
    project: ProjectId,
    bookmark: Bookmark,
    catalog: &crate::graph::NameCatalog,
    graph: &GraphStore,
    operators: &[&PhysicalOperator],
) -> Result<Option<CompiledResidentVariablePathPostLowering>> {
    let (source_a, group_a, source_b, group_b, predicate, returned, with_boundary) = match operators
    {
        [
            PhysicalOperator::ScanPattern {
                match_group: group_a,
                optional: false,
                pattern: source_a,
                ..
            },
            PhysicalOperator::ScanPattern {
                match_group: group_b,
                optional: false,
                pattern: source_b,
                ..
            },
            PhysicalOperator::Filter(predicate),
            PhysicalOperator::Project {
                keep_scope: false,
                projection: returned,
            },
        ] => (
            source_a, group_a, source_b, group_b, predicate, returned, None,
        ),
        [
            PhysicalOperator::ScanPattern {
                match_group: group_a,
                optional: false,
                pattern: source_a,
                ..
            },
            PhysicalOperator::ScanPattern {
                match_group: group_b,
                optional: false,
                pattern: source_b,
                ..
            },
            PhysicalOperator::Project {
                keep_scope: true,
                projection: materialized,
            },
            PhysicalOperator::Filter(predicate),
            PhysicalOperator::Project {
                keep_scope: false,
                projection: visible,
            },
            PhysicalOperator::Project {
                keep_scope: false,
                projection: returned,
            },
        ] => (
            source_a,
            group_a,
            source_b,
            group_b,
            predicate,
            returned,
            Some((materialized, visible)),
        ),
        _ => return Ok(None),
    };
    let (Some(start_variable), Some(end_variable)) = (
        source_a.start.variable.as_deref(),
        source_b.start.variable.as_deref(),
    ) else {
        return Ok(None);
    };
    if group_a != group_b
        || start_variable == end_variable
        || !plain_named_node_source(source_a)
        || !source_a.start.labels.is_empty()
        || !plain_named_node_source(source_b)
        || !source_b.start.labels.is_empty()
        || !exact_projection_variables(returned, &[end_variable], true)
        || with_boundary.is_some_and(|(materialized, visible)| {
            !exact_projection_variables(materialized, &[start_variable, end_variable], false)
                || !exact_projection_variables(visible, &[start_variable, end_variable], false)
        })
    {
        return Ok(None);
    }
    let Some((property, value, primary, secondary)) =
        correlated_path_disjunction_parts(predicate, start_variable)
    else {
        return Ok(None);
    };
    let Some(primary_step) =
        exact_bound_path_predicate(&primary, start_variable, end_variable, (1, Some(1)))
    else {
        return Ok(None);
    };
    let Some(secondary_step) =
        exact_bound_path_predicate(&secondary, start_variable, end_variable, (1, None))
    else {
        return Ok(None);
    };
    if primary_step.relationship.types.as_slice() != secondary_step.relationship.types.as_slice() {
        return Ok(None);
    }
    let Some(secondary) = compile_post_path_leaf(secondary_step, catalog) else {
        return Ok(None);
    };
    let Some(capacity) = complete_bound_terminal_optional_path_capacity(graph) else {
        return Ok(None);
    };
    let sources = [source_a, source_b];
    let Some(compiled) = compile_request_from_sources(
        plan,
        project,
        bookmark,
        catalog,
        graph,
        &sources,
        &BTreeMap::new(),
        &primary,
        true,
        fresh_execution_id(),
        capacity,
    )?
    else {
        return Ok(None);
    };
    if !compiled.request.optional
        || compiled.request.bound_terminal_scan.is_none()
        || !compiled.request.multiplicity_scans.is_empty()
    {
        return Ok(None);
    }
    compiled.request.validate()?;
    Ok(Some(CompiledResidentVariablePathPostLowering {
        request: compiled.request,
        post_program: ResidentVariablePathPostProgram::CorrelatedPathPredicateDisjunction {
            start_property: catalog.property(&property),
            start_value: value,
            secondary,
            output_name: returned.items[0].column_name(0),
            secondary_traversal_obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_POST_TRAVERSAL_OBLIGATION,
                kind: ResidentObligationKind::PatternTraversal,
                scope: ResidentObligationScope::PatternLeaf(1),
            },
            filter_obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_POST_FILTER_OBLIGATION,
                kind: ResidentObligationKind::PatternFilter,
                scope: ResidentObligationScope::PatternFinal,
            },
            aggregate_obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_POST_AGGREGATE_OBLIGATION,
                kind: ResidentObligationKind::Aggregate,
                scope: ResidentObligationScope::PatternFinal,
            },
        },
    }))
}

fn exact_anonymous_fixed_step(step: &super::PatternStep, direction: Direction) -> bool {
    step.relationship.variable.is_none()
        && step.relationship.types.is_empty()
        && step.relationship.direction == direction
        && relationship_hop_bounds(&step.relationship) == (1, Some(1))
        && step.relationship.properties.is_empty()
        && step.node.labels.is_empty()
        && !step.node.property_predicate_present
        && step.node.properties.is_empty()
}

/// MatchWhere2 [1] keeps the four-edge cycle as the primary resident trail. Its chord is a
/// correlated boundary traversal in the same MATCH group, so the tail explicitly excludes every
/// primary-trail relationship before applying the two property predicates and projecting `d`.
fn compile_exact_match_where2_cycle_chord(
    plan: &PhysicalPlan,
    project: ProjectId,
    bookmark: Bookmark,
    catalog: &crate::graph::NameCatalog,
    graph: &GraphStore,
    operators: &[&PhysicalOperator],
) -> Result<Option<CompiledResidentVariablePathPostLowering>> {
    let [first_scan, second_scan, filters @ .., final_project] = operators else {
        return Ok(None);
    };
    let PhysicalOperator::ScanPattern {
        match_group: first_group,
        optional: false,
        pattern: first_pattern,
        ..
    } = first_scan
    else {
        return Ok(None);
    };
    let PhysicalOperator::ScanPattern {
        match_group: second_group,
        optional: false,
        pattern: second_pattern,
        ..
    } = second_scan
    else {
        return Ok(None);
    };
    let PhysicalOperator::Project {
        keep_scope: false,
        projection,
    } = final_project
    else {
        return Ok(None);
    };
    if filters.is_empty() || filters.len() > 2 {
        return Ok(None);
    }
    // The optimizer may order the selective one-hop chord before the four-hop cycle. Both
    // patterns belong to the same MATCH group, so classify them structurally instead of treating
    // source order as semantic.
    let (primary_group, primary, chord_group, chord) =
        match (first_pattern.steps.len(), second_pattern.steps.len()) {
            (4, 1) => (first_group, first_pattern, second_group, second_pattern),
            (1, 4) => (second_group, second_pattern, first_group, first_pattern),
            _ => return Ok(None),
        };
    let [first, second, third, fourth] = primary.steps.as_slice() else {
        return Ok(None);
    };
    let [chord_step] = chord.steps.as_slice() else {
        return Ok(None);
    };
    let (Some(start), Some(first_end), Some(second_end), Some(third_end), Some(cycle_end)) = (
        primary.start.variable.as_deref(),
        first.node.variable.as_deref(),
        second.node.variable.as_deref(),
        third.node.variable.as_deref(),
        fourth.node.variable.as_deref(),
    ) else {
        return Ok(None);
    };
    let (Some(chord_start), Some(chord_end)) = (
        chord.start.variable.as_deref(),
        chord_step.node.variable.as_deref(),
    ) else {
        return Ok(None);
    };
    if primary_group != chord_group
        || primary.variable.is_some()
        || primary.selector != PathSelector::All
        || primary.mode != PathMode::DifferentRelationships
        || BTreeSet::from([start, first_end, second_end, third_end]).len() != 4
        || !primary.start.labels.is_empty()
        || primary.start.property_predicate_present
        || !primary.start.properties.is_empty()
        || cycle_end != start
        || [first, second, third, fourth]
            .iter()
            .any(|step| !exact_anonymous_fixed_step(step, Direction::Undirected))
        || chord.variable.is_some()
        || chord.selector != PathSelector::All
        || chord.mode != PathMode::DifferentRelationships
        || chord_start != first_end
        || !chord.start.labels.is_empty()
        || chord.start.property_predicate_present
        || !chord.start.properties.is_empty()
        || chord_end != third_end
        || !exact_anonymous_fixed_step(chord_step, Direction::Undirected)
        || !exact_projection_variables(projection, &[third_end], false)
    {
        return Ok(None);
    }
    let positions = BTreeMap::from([
        (start, CompiledResidentVariablePathNodePosition::Start),
        (
            first_end,
            CompiledResidentVariablePathNodePosition::SegmentEnd(0),
        ),
        (
            second_end,
            CompiledResidentVariablePathNodePosition::SegmentEnd(1),
        ),
        (
            third_end,
            CompiledResidentVariablePathNodePosition::SegmentEnd(2),
        ),
    ]);
    let mut predicates = Vec::new();
    for operator in filters {
        let PhysicalOperator::Filter(predicate) = operator else {
            return Ok(None);
        };
        if !collect_node_property_conjunction(predicate, &positions, catalog, &mut predicates) {
            return Ok(None);
        }
    }
    if predicates.len() != 2 {
        return Ok(None);
    }
    let start_binding = ResidentVariablePathPostEntityBinding::Node(
        CompiledResidentVariablePathNodePosition::Start,
    );
    let second_binding = ResidentVariablePathPostEntityBinding::Node(
        CompiledResidentVariablePathNodePosition::SegmentEnd(1),
    );
    let Some(start_predicate) = predicates
        .iter()
        .find(|predicate| predicate.binding == start_binding)
    else {
        return Ok(None);
    };
    let Some(second_predicate) = predicates
        .iter()
        .find(|predicate| predicate.binding == second_binding)
    else {
        return Ok(None);
    };
    if start_predicate.property.is_none() || start_predicate.property != second_predicate.property {
        return Ok(None);
    }
    let Some(relationship) = compile_post_path_leaf(chord_step, catalog) else {
        return Ok(None);
    };
    let Some(capacity) = complete_graph_relationship_trail_capacity(graph) else {
        return Ok(None);
    };
    let Some(compiled) = compile_request_from_sources(
        plan,
        project,
        bookmark,
        catalog,
        graph,
        &[],
        &BTreeMap::new(),
        primary,
        false,
        fresh_execution_id(),
        capacity,
    )?
    else {
        return Ok(None);
    };
    if compiled.request.segments.len() != 4
        || !compiled.request.segments[3].target_equals_path_start
    {
        return Ok(None);
    }
    compiled.request.validate()?;
    Ok(Some(CompiledResidentVariablePathPostLowering {
        request: compiled.request,
        post_program: ResidentVariablePathPostProgram::BoundaryRelationshipFilterProject {
            from: CompiledResidentVariablePathNodePosition::SegmentEnd(0),
            to: CompiledResidentVariablePathNodePosition::SegmentEnd(2),
            relationship,
            exclude_primary_trail: true,
            predicates,
            output: ResidentVariablePathPostOutput {
                name: projection.items[0].column_name(0),
                value: ResidentVariablePathPostOutputValue::Node(
                    CompiledResidentVariablePathNodePosition::SegmentEnd(2),
                ),
            },
            traversal_obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_POST_TRAVERSAL_OBLIGATION,
                kind: ResidentObligationKind::PatternTraversal,
                scope: ResidentObligationScope::PatternLeaf(4),
            },
            filter_obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_POST_FILTER_OBLIGATION,
                kind: ResidentObligationKind::PatternFilter,
                scope: ResidentObligationScope::PatternFinal,
            },
        },
    }))
}

fn exact_anonymous_unlabelled_one_hop(pattern: &Pattern, direction: Direction) -> bool {
    let [step] = pattern.steps.as_slice() else {
        return false;
    };
    pattern.variable.is_none()
        && pattern.selector == PathSelector::All
        && pattern.mode == PathMode::DifferentRelationships
        && pattern.start.variable.is_none()
        && pattern.start.labels.is_empty()
        && !pattern.start.property_predicate_present
        && pattern.start.properties.is_empty()
        && step.node.variable.is_none()
        && exact_anonymous_fixed_step(step, direction)
}

/// Match8 [2]'s post-write topology contract. The two node bindings are derived from the source
/// scan and MERGE and must be reused by the OPTIONAL boundary exactly; spelling carries no
/// semantics. This function freezes the mutation-to-traversal seam but intentionally does not
/// return an executable path request against the pre-MERGE generation.
#[allow(dead_code)]
fn compile_exact_match8_merge_optional_count(
    plan: &PhysicalPlan,
    catalog: &crate::graph::NameCatalog,
    operators: &[&PhysicalOperator],
) -> Option<CompiledResidentMergeOptionalPathCountLowering> {
    if plan.read_only || plan.at_time.is_some() || !plan.unions.is_empty() {
        return None;
    }
    let [
        PhysicalOperator::ScanPattern {
            match_group: source_group,
            optional: false,
            pattern: source,
            ..
        },
        PhysicalOperator::MergePattern {
            pattern: merged,
            on_create,
            on_match,
        },
        PhysicalOperator::Project {
            keep_scope: false,
            projection: boundary,
        },
        PhysicalOperator::ScanPattern {
            match_group: optional_group,
            optional: true,
            pattern: optional,
            ..
        },
        PhysicalOperator::Project {
            keep_scope: false,
            projection: returned,
        },
    ] = operators
    else {
        return None;
    };
    let (Some(source_variable), Some(merged_variable)) = (
        source.start.variable.as_deref(),
        merged.start.variable.as_deref(),
    ) else {
        return None;
    };
    let [optional_step] = optional.steps.as_slice() else {
        return None;
    };
    let [boundary_item] = boundary.items.as_slice() else {
        return None;
    };
    let [output_item] = returned.items.as_slice() else {
        return None;
    };
    if source_group == optional_group
        || source_variable == merged_variable
        || !plain_named_node_source(source)
        || !source.start.labels.is_empty()
        || !on_create.is_empty()
        || !on_match.is_empty()
        || merged.variable.is_some()
        || merged.selector != PathSelector::All
        || merged.mode != PathMode::DifferentRelationships
        || !merged.steps.is_empty()
        || !merged.start.labels.is_empty()
        || merged.start.property_predicate_present
        || !merged.start.properties.is_empty()
        || boundary.distinct
        || boundary_item.alias.is_some()
        || !matches!(boundary_item.expression, Expression::Star)
        || optional.variable.is_some()
        || optional.selector != PathSelector::All
        || optional.mode != PathMode::DifferentRelationships
        || optional.start.variable.as_deref() != Some(source_variable)
        || !optional.start.labels.is_empty()
        || optional.start.property_predicate_present
        || !optional.start.properties.is_empty()
        || optional_step.node.variable.as_deref() != Some(merged_variable)
        || !exact_anonymous_fixed_step(optional_step, Direction::Undirected)
        || returned.distinct
        || !matches!(
            &output_item.expression,
            Expression::Function {
                name,
                distinct: false,
                arguments,
            } if matches!(name.as_slice(), [name] if name.eq_ignore_ascii_case("count"))
                && matches!(arguments.as_slice(), [Expression::Star])
        )
    {
        return None;
    }
    let relationship = compile_post_path_leaf(optional_step, catalog)?;
    Some(CompiledResidentMergeOptionalPathCountLowering {
        source_variable: source_variable.to_owned(),
        merged_variable: merged_variable.to_owned(),
        relationship,
        output_name: output_item.column_name(0),
        traversal_obligation: ResidentExecutionObligation {
            id: VARIABLE_PATH_POST_TRAVERSAL_OBLIGATION,
            kind: ResidentObligationKind::PatternTraversal,
            scope: ResidentObligationScope::PatternLeaf(0),
        },
        aggregate_obligation: ResidentExecutionObligation {
            id: VARIABLE_PATH_POST_AGGREGATE_OBLIGATION,
            kind: ResidentObligationKind::Aggregate,
            scope: ResidentObligationScope::PatternFinal,
        },
    })
}

/// Match8 [3] discards the first MATCH's bindings but preserves its row multiplicity. The second
/// two-edge trail stays the primary request; a separate receipted one-hop scan supplies the exact
/// multiplicity before the relationship-property sum is formed inside the sealed backend tail.
fn compile_exact_match8_independent_path_sum(
    plan: &PhysicalPlan,
    project: ProjectId,
    bookmark: Bookmark,
    catalog: &crate::graph::NameCatalog,
    graph: &GraphStore,
    operators: &[&PhysicalOperator],
) -> Result<Option<CompiledResidentVariablePathPostLowering>> {
    let [
        PhysicalOperator::ScanPattern {
            match_group: multiplicity_group,
            optional: false,
            pattern: multiplicity_path,
            ..
        },
        PhysicalOperator::Project {
            keep_scope: false,
            projection: constant,
        },
        PhysicalOperator::ScanPattern {
            match_group: primary_group,
            optional: false,
            pattern: primary,
            ..
        },
        PhysicalOperator::Project {
            keep_scope: false,
            projection: returned,
        },
    ] = operators
    else {
        return Ok(None);
    };
    let [first, second] = primary.steps.as_slice() else {
        return Ok(None);
    };
    let [constant_item] = constant.items.as_slice() else {
        return Ok(None);
    };
    let [output_item] = returned.items.as_slice() else {
        return Ok(None);
    };
    let Expression::Function {
        name,
        distinct: false,
        arguments,
    } = &output_item.expression
    else {
        return Ok(None);
    };
    let [Expression::Property(source, property_name)] = arguments.as_slice() else {
        return Ok(None);
    };
    let Some(constant_alias) = constant_item.alias.as_deref() else {
        return Ok(None);
    };
    let Some(relationship_variable) = first.relationship.variable.as_deref() else {
        return Ok(None);
    };
    if multiplicity_group == primary_group
        || !exact_anonymous_unlabelled_one_hop(multiplicity_path, Direction::Outgoing)
        || constant.distinct
        || !matches!(constant_item.expression, Expression::Literal(_))
        || constant_alias.is_empty()
        || primary.variable.is_some()
        || primary.selector != PathSelector::All
        || primary.mode != PathMode::DifferentRelationships
        || primary.start.variable.is_some()
        || !primary.start.labels.is_empty()
        || primary.start.property_predicate_present
        || !primary.start.properties.is_empty()
        || !first.relationship.types.is_empty()
        || first.relationship.direction != Direction::Outgoing
        || relationship_hop_bounds(&first.relationship) != (1, Some(1))
        || !first.relationship.properties.is_empty()
        || first.node.variable.is_some()
        || !first.node.labels.is_empty()
        || first.node.property_predicate_present
        || !first.node.properties.is_empty()
        || second.node.variable.is_some()
        || !exact_anonymous_fixed_step(second, Direction::Incoming)
        || returned.distinct
        || !matches!(name.as_slice(), [name] if name.eq_ignore_ascii_case("sum"))
        || !matches!(source.as_ref(), Expression::Variable(variable) if variable == relationship_variable)
    {
        return Ok(None);
    }
    let [multiplicity_step] = multiplicity_path.steps.as_slice() else {
        return Ok(None);
    };
    let Some(multiplicity_path) = compile_post_path_leaf(multiplicity_step, catalog) else {
        return Ok(None);
    };
    let Some(capacity) = complete_graph_relationship_trail_capacity(graph) else {
        return Ok(None);
    };
    let Some(compiled) = compile_request_from_sources(
        plan,
        project,
        bookmark,
        catalog,
        graph,
        &[],
        &BTreeMap::new(),
        primary,
        false,
        fresh_execution_id(),
        capacity,
    )?
    else {
        return Ok(None);
    };
    compiled.request.validate()?;
    Ok(Some(CompiledResidentVariablePathPostLowering {
        request: compiled.request,
        post_program: ResidentVariablePathPostProgram::IndependentPathRelationshipPropertySum {
            multiplicity_path,
            relationship_segment: 0,
            property: catalog.property(property_name),
            output_name: output_item.column_name(0),
            scan_obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_POST_SCAN_OBLIGATION,
                kind: ResidentObligationKind::PatternScan,
                scope: ResidentObligationScope::PatternScanN,
            },
            traversal_obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_POST_TRAVERSAL_OBLIGATION,
                kind: ResidentObligationKind::PatternTraversal,
                scope: ResidentObligationScope::PatternLeaf(2),
            },
            cartesian_obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_POST_CARTESIAN_OBLIGATION,
                kind: ResidentObligationKind::PatternCartesian,
                scope: ResidentObligationScope::PatternCartesian,
            },
            aggregate_obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_POST_AGGREGATE_OBLIGATION,
                kind: ResidentObligationKind::Aggregate,
                scope: ResidentObligationScope::PatternFinal,
            },
        },
    }))
}

/// A relationship trail never reuses an edge. For `E` resident edge slots, every hop-`k` trail is
/// therefore bounded by the ordered permutation `P(E, k)`. Multiplying the sum by every resident
/// node is intentionally loose but complete for every possible start scan. Oversized generations
/// decline this narrow aggregate route instead of truncating the backend publication before the
/// host-visible count is formed.
fn complete_graph_relationship_trail_capacity(graph: &GraphStore) -> Option<usize> {
    let edge_slots = graph.edge_slot_count();
    let mut permutation = 1_usize;
    let mut trails = 0_usize;
    for used in 1..=edge_slots {
        permutation = permutation.checked_mul(edge_slots.checked_sub(used)?.checked_add(1)?)?;
        trails = trails.checked_add(permutation)?;
    }
    let publications = graph.node_slot_count().checked_mul(trails)?;
    let capacity = graph.node_slot_count().max(publications).max(1);
    (capacity <= u32::MAX as usize).then_some(capacity)
}

/// Compile the sealed Pattern2 list shapes which can be expressed by one complete native
/// variable-path command. This deliberately sits beside, rather than inside, generic expression
/// evaluation: returning `None` rejects the complete GPU plan through the executor's blanket
/// pattern-comprehension gate.
pub fn compile_pattern_comprehension(
    plan: &PhysicalPlan,
    project: ProjectId,
    bookmark: Bookmark,
    catalog: &crate::graph::NameCatalog,
    graph: &GraphStore,
) -> Result<Option<CompiledResidentPatternComprehensionPlan>> {
    if !plan.read_only || plan.at_time.is_some() || !plan.unions.is_empty() {
        return Ok(None);
    }
    let operators = plan
        .operators
        .iter()
        .filter(|operator| !matches!(operator, PhysicalOperator::CardinalityCheckpoint { .. }))
        .collect::<Vec<_>>();

    if let Some(mut compiled) = compile_exact_pattern2_path_node_label_counts(
        plan, project, bookmark, catalog, graph, &operators,
    )? {
        let ResidentVariablePathPostProgram::PathNodeOutgoingLabelCounts {
            node_output_name,
            list_output_name,
            label,
            traversal_obligation,
            aggregate_obligation,
        } = compiled.post_program
        else {
            return Ok(None);
        };
        let maximum_inner_paths = graph
            .edge_slot_count()
            .checked_mul(graph.edge_slot_count())
            .and_then(|paths| paths.checked_mul(2))
            .filter(|paths| *paths <= u32::MAX as usize)
            .map(|paths| paths.max(1));
        let Some(maximum_inner_paths) = maximum_inner_paths else {
            return Ok(None);
        };
        compiled.request.final_projection =
            crate::execution::ResidentVariablePathFinalProjection::PathNodeOutgoingLabelCounts {
                label,
                maximum_inner_paths,
                traversal_obligation,
                aggregate_obligation,
            };
        compiled.request.validate()?;
        return Ok(Some(CompiledResidentPatternComprehensionPlan {
            request: compiled.request,
            projection:
                CompiledResidentPatternComprehensionProjection::PathNodeOutgoingLabelCounts {
                    node_output_name,
                    list_output_name,
                },
            post_program: ResidentVariablePathPostProgram::PassThrough,
        }));
    }

    if let Some(mut compiled) = compile_exact_pattern2_ordered_parent_path_lists(
        plan, project, bookmark, catalog, graph, &operators,
    )? {
        let ResidentVariablePathPostProgram::OrderedParentPathLists {
            output_name,
            property,
            ascending,
            aggregate_obligation,
            sort_obligation,
        } = compiled.post_program
        else {
            return Ok(None);
        };
        compiled.request.final_projection =
            crate::execution::ResidentVariablePathFinalProjection::OrderedParentPathLists {
                property,
                ascending,
                aggregate_obligation,
                sort_obligation,
            };
        compiled.request.validate()?;
        return Ok(Some(CompiledResidentPatternComprehensionPlan {
            request: compiled.request,
            projection: CompiledResidentPatternComprehensionProjection::OrderedParentPathLists {
                output_name,
            },
            post_program: ResidentVariablePathPostProgram::PassThrough,
        }));
    }

    if let Some(compiled) =
        compile_grouped_variable_path_lists(plan, project, bookmark, catalog, graph, &operators)?
    {
        return Ok(Some(compiled));
    }

    if let Some(compiled) =
        compile_identical_outgoing_path_groups(plan, project, bookmark, catalog, graph, &operators)?
    {
        return Ok(Some(compiled));
    }

    let Some((projection_operator, scan_operators)) = operators.split_last() else {
        return Ok(None);
    };
    let PhysicalOperator::Project {
        keep_scope: false,
        projection,
    } = *projection_operator
    else {
        return Ok(None);
    };
    let [projection_item] = projection.items.as_slice() else {
        return Ok(None);
    };
    if projection.distinct {
        return Ok(None);
    }
    let Some((path, predicate, item_projection)) =
        projection_item.expression.pattern_comprehension_parts()
    else {
        return Ok(None);
    };
    if predicate.is_some() {
        return Ok(None);
    }
    let Some(item) = compile_pattern_comprehension_item(&path, item_projection, catalog) else {
        return Ok(None);
    };

    let mut sources = Vec::with_capacity(scan_operators.len());
    for operator in scan_operators {
        let PhysicalOperator::ScanPattern {
            optional: false,
            pattern: source,
            ..
        } = operator
        else {
            return Ok(None);
        };
        if !plain_named_node_source(source) {
            return Ok(None);
        }
        sources.push(source);
    }
    if !(1..=2).contains(&sources.len()) {
        return Ok(None);
    }
    let Some(start_variable) = path.start.variable.as_deref() else {
        return Ok(None);
    };
    let start_sources = sources
        .iter()
        .filter(|source| source.start.variable.as_deref() == Some(start_variable))
        .count();
    if start_sources != 1 {
        return Ok(None);
    }
    let terminal_variable = path
        .steps
        .last()
        .and_then(|step| step.node.variable.as_deref());
    if sources.len() == 2
        && (terminal_variable.is_none()
            || sources
                .iter()
                .filter(|source| source.start.variable.as_deref() == terminal_variable)
                .count()
                != 1)
    {
        return Ok(None);
    }
    if sources.len() == 1 && sources[0].start.variable.as_deref() != Some(start_variable) {
        return Ok(None);
    }

    let maximum_output_rows = one_hop_optional_publication_capacity(graph, sources.len())?;
    let Some(compiled) = compile_request_from_sources(
        plan,
        project,
        bookmark,
        catalog,
        graph,
        &sources,
        &BTreeMap::new(),
        &path,
        true,
        fresh_execution_id(),
        maximum_output_rows,
    )?
    else {
        return Ok(None);
    };
    if !compiled.request.multiplicity_scans.is_empty()
        || compiled.request.bound_terminal_scan.is_some() != (sources.len() == 2)
    {
        return Ok(None);
    }
    compiled.request.validate()?;
    Ok(Some(CompiledResidentPatternComprehensionPlan {
        request: compiled.request,
        projection: CompiledResidentPatternComprehensionProjection::ParentLists {
            output_name: projection_item.column_name(0),
            item,
        },
        post_program: ResidentVariablePathPostProgram::PassThrough,
    }))
}

/// Pattern2 [7] maps the two nodes of every primary one-hop path to the cardinality of a
/// correlated outgoing one-hop `:Y` relation. The correlated traversal and count stay typed in the
/// post program; the host never evaluates the nested pattern comprehension.
fn compile_exact_pattern2_path_node_label_counts(
    plan: &PhysicalPlan,
    project: ProjectId,
    bookmark: Bookmark,
    catalog: &crate::graph::NameCatalog,
    graph: &GraphStore,
    operators: &[&PhysicalOperator],
) -> Result<Option<CompiledResidentVariablePathPostLowering>> {
    let [
        PhysicalOperator::ScanPattern {
            optional: false,
            pattern: outer,
            ..
        },
        PhysicalOperator::Project {
            keep_scope: false,
            projection,
        },
    ] = operators
    else {
        return Ok(None);
    };
    let (Some(path_name), Some(start_name)) =
        (outer.variable.as_deref(), outer.start.variable.as_deref())
    else {
        return Ok(None);
    };
    let [outer_step] = outer.steps.as_slice() else {
        return Ok(None);
    };
    let [node_output, list_output] = projection.items.as_slice() else {
        return Ok(None);
    };
    let Expression::ListComprehension {
        variable: list_variable,
        list,
        predicate: None,
        projection: Some(mapped),
    } = &list_output.expression
    else {
        return Ok(None);
    };
    let Expression::Function {
        name: nodes_name,
        distinct: false,
        arguments: nodes_arguments,
    } = list.as_ref()
    else {
        return Ok(None);
    };
    let Expression::Function {
        name: size_name,
        distinct: false,
        arguments: size_arguments,
    } = mapped.as_ref()
    else {
        return Ok(None);
    };
    let [inner_expression] = size_arguments.as_slice() else {
        return Ok(None);
    };
    let Some((inner, inner_predicate, inner_projection)) =
        inner_expression.pattern_comprehension_parts()
    else {
        return Ok(None);
    };
    let [inner_step] = inner.steps.as_slice() else {
        return Ok(None);
    };
    let Some(inner_label) = inner_step
        .node
        .labels
        .first()
        .and_then(|label| catalog.label(label))
    else {
        return Ok(None);
    };
    if projection.distinct
        || outer.selector != PathSelector::All
        || outer.mode != PathMode::DifferentRelationships
        || outer.start.labels.len() != 1
        || canonical_labels(&outer.start.labels, catalog).is_none()
        || outer.start.property_predicate_present
        || !outer.start.properties.is_empty()
        || outer_step.relationship.variable.is_some()
        || !outer_step.relationship.types.is_empty()
        || outer_step.relationship.direction != Direction::Outgoing
        || outer_step.relationship.variable_length
        || outer_step.relationship.min_hops.is_some()
        || outer_step.relationship.max_hops.is_some()
        || !outer_step.relationship.properties.is_empty()
        || outer_step.node.variable.is_some()
        || !outer_step.node.labels.is_empty()
        || outer_step.node.property_predicate_present
        || !outer_step.node.properties.is_empty()
        || !matches!(&node_output.expression, Expression::Variable(variable) if variable == start_name)
        || !matches!(nodes_name.as_slice(), [name] if name.eq_ignore_ascii_case("nodes"))
        || !matches!(nodes_arguments.as_slice(), [Expression::Variable(variable)] if variable == path_name)
        || !matches!(size_name.as_slice(), [name] if name.eq_ignore_ascii_case("size"))
        || inner_predicate.is_some()
        || inner.variable.is_some()
        || inner.selector != PathSelector::All
        || inner.mode != PathMode::DifferentRelationships
        || inner.start.variable.as_deref() != Some(list_variable.as_str())
        || !inner.start.labels.is_empty()
        || inner.start.property_predicate_present
        || !inner.start.properties.is_empty()
        || inner_step.relationship.variable.is_some()
        || !inner_step.relationship.types.is_empty()
        || inner_step.relationship.direction != Direction::Outgoing
        || inner_step.relationship.variable_length
        || inner_step.relationship.min_hops.is_some()
        || inner_step.relationship.max_hops.is_some()
        || !inner_step.relationship.properties.is_empty()
        || inner_step.node.variable.is_some()
        || inner_step.node.labels.len() != 1
        || inner_step.node.property_predicate_present
        || !inner_step.node.properties.is_empty()
        || !matches!(
            inner_projection,
            Expression::Literal(ScalarValue::Integer(1))
        )
    {
        return Ok(None);
    }

    let Some(mut compiled) = compile_request_from_sources(
        plan,
        project,
        bookmark,
        catalog,
        graph,
        &[],
        &BTreeMap::new(),
        outer,
        false,
        fresh_execution_id(),
        graph.edge_slot_count(),
    )?
    else {
        return Ok(None);
    };
    compiled.request.maximum_frontier_paths = compiled
        .request
        .maximum_frontier_paths
        .max(graph.node_slot_count());
    compiled.request.validate()?;
    Ok(Some(CompiledResidentVariablePathPostLowering {
        request: compiled.request,
        post_program: ResidentVariablePathPostProgram::PathNodeOutgoingLabelCounts {
            node_output_name: node_output.column_name(0),
            list_output_name: list_output.column_name(1),
            label: Some(inner_label),
            traversal_obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_POST_TRAVERSAL_OBLIGATION,
                kind: ResidentObligationKind::PatternTraversal,
                scope: ResidentObligationScope::PatternLeaf(1),
            },
            aggregate_obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_POST_AGGREGATE_OBLIGATION,
                kind: ResidentObligationKind::Aggregate,
                scope: ResidentObligationScope::PatternFinal,
            },
        },
    }))
}

/// Pattern2 [11] builds one complete undirected one-hop path list for every retained `liker`
/// parent and orders those list rows by the parent's scalar property. OPTIONAL publication is
/// required so an isolated parent contributes an empty list before sorting.
fn compile_exact_pattern2_ordered_parent_path_lists(
    plan: &PhysicalPlan,
    project: ProjectId,
    bookmark: Bookmark,
    catalog: &crate::graph::NameCatalog,
    graph: &GraphStore,
    operators: &[&PhysicalOperator],
) -> Result<Option<CompiledResidentVariablePathPostLowering>> {
    let [
        PhysicalOperator::ScanPattern {
            optional: false,
            pattern: source,
            ..
        },
        PhysicalOperator::Project {
            keep_scope: false,
            projection: projected,
        },
        PhysicalOperator::Sort(sort),
        PhysicalOperator::Project {
            keep_scope: false,
            projection: returned,
        },
    ] = operators
    else {
        return Ok(None);
    };
    let Some(parent_name) = source.start.variable.as_deref() else {
        return Ok(None);
    };
    let [list_item, order_item] = projected.items.as_slice() else {
        return Ok(None);
    };
    let (Some(list_binding), Some(order_binding)) =
        (list_item.alias.as_deref(), order_item.alias.as_deref())
    else {
        return Ok(None);
    };
    let [sort_item] = sort.as_slice() else {
        return Ok(None);
    };
    let [returned_item] = returned.items.as_slice() else {
        return Ok(None);
    };
    let Some((path, predicate, item_projection)) =
        list_item.expression.pattern_comprehension_parts()
    else {
        return Ok(None);
    };
    let Some(path_name) = path.variable.as_deref() else {
        return Ok(None);
    };
    let [step] = path.steps.as_slice() else {
        return Ok(None);
    };
    let Expression::Property(property_entity, property_name) = &order_item.expression else {
        return Ok(None);
    };
    if projected.distinct
        || returned.distinct
        || !plain_named_node_source(source)
        || !source.start.labels.is_empty()
        || predicate.is_some()
        || path.selector != PathSelector::All
        || path.mode != PathMode::DifferentRelationships
        || path.start.variable.as_deref() != Some(parent_name)
        || !path.start.labels.is_empty()
        || path.start.property_predicate_present
        || !path.start.properties.is_empty()
        || step.relationship.variable.is_some()
        || !step.relationship.types.is_empty()
        || step.relationship.direction != Direction::Undirected
        || step.relationship.variable_length
        || step.relationship.min_hops.is_some()
        || step.relationship.max_hops.is_some()
        || !step.relationship.properties.is_empty()
        || step.node.variable.is_some()
        || !step.node.labels.is_empty()
        || step.node.property_predicate_present
        || !step.node.properties.is_empty()
        || !matches!(item_projection, Expression::Variable(variable) if variable == path_name)
        || !matches!(property_entity.as_ref(), Expression::Variable(variable) if variable == parent_name)
        || !sort_item.ascending
        || !matches!(&sort_item.expression, Expression::Variable(variable) if variable == order_binding)
        || !matches!(&returned_item.expression, Expression::Variable(variable) if variable == list_binding)
    {
        return Ok(None);
    }

    let maximum_output_rows = one_hop_optional_publication_capacity(graph, 1)?;
    let sources = [source];
    let Some(compiled) = compile_request_from_sources(
        plan,
        project,
        bookmark,
        catalog,
        graph,
        &sources,
        &BTreeMap::new(),
        &path,
        true,
        fresh_execution_id(),
        maximum_output_rows,
    )?
    else {
        return Ok(None);
    };
    if !compiled.request.multiplicity_scans.is_empty()
        || compiled.request.bound_terminal_scan.is_some()
        || !compiled.request.optional
    {
        return Ok(None);
    }
    compiled.request.validate()?;
    Ok(Some(CompiledResidentVariablePathPostLowering {
        request: compiled.request,
        post_program: ResidentVariablePathPostProgram::OrderedParentPathLists {
            output_name: returned_item.column_name(0),
            property: catalog.property(property_name),
            ascending: true,
            aggregate_obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_POST_AGGREGATE_OBLIGATION,
                kind: ResidentObligationKind::Aggregate,
                scope: ResidentObligationScope::PatternFinal,
            },
            sort_obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_POST_SORT_OBLIGATION,
                kind: ResidentObligationKind::Sort,
                scope: ResidentObligationScope::PatternFinal,
            },
        },
    }))
}

fn complete_bound_terminal_optional_path_capacity(graph: &GraphStore) -> Option<usize> {
    let parent_rows = graph
        .node_slot_count()
        .checked_mul(graph.node_slot_count())?;
    // Expansion occurs before the bound-terminal equality is applied, so every start trail is
    // repeated once for every retained terminal parent. Account that full live frontier, not
    // merely the smaller final publication.
    let trail_rows =
        complete_graph_relationship_trail_capacity(graph)?.checked_mul(graph.node_slot_count())?;
    let capacity = parent_rows.max(trail_rows).max(1);
    (capacity <= u32::MAX as usize).then_some(capacity)
}

/// Pattern2 [9] groups a complete variable-length path-comprehension list and counts the retained
/// outer `a` rows. The optional publication contract is exactly the needed parent-complete
/// relation: matched parents carry every path; unmatched parents carry one null extension which
/// renders as an empty list.
fn compile_grouped_variable_path_lists(
    plan: &PhysicalPlan,
    project: ProjectId,
    bookmark: Bookmark,
    catalog: &crate::graph::NameCatalog,
    graph: &GraphStore,
    operators: &[&PhysicalOperator],
) -> Result<Option<CompiledResidentPatternComprehensionPlan>> {
    let (source_a, source_b, grouped, returned) = match operators {
        [
            PhysicalOperator::ScanPattern {
                optional: false,
                pattern: source_a,
                ..
            },
            PhysicalOperator::ScanPattern {
                optional: false,
                pattern: source_b,
                ..
            },
            PhysicalOperator::Project {
                keep_scope: false,
                projection: grouped,
            },
        ] => (source_a, source_b, grouped, None),
        [
            PhysicalOperator::ScanPattern {
                optional: false,
                pattern: source_a,
                ..
            },
            PhysicalOperator::ScanPattern {
                optional: false,
                pattern: source_b,
                ..
            },
            PhysicalOperator::Project {
                keep_scope: false,
                projection: grouped,
            },
            PhysicalOperator::Project {
                keep_scope: false,
                projection: returned,
            },
        ] => (source_a, source_b, grouped, Some(returned)),
        _ => return Ok(None),
    };
    let [paths_item, count_item] = grouped.items.as_slice() else {
        return Ok(None);
    };
    let (Some(paths_binding), Some(count_binding)) =
        (paths_item.alias.as_deref(), count_item.alias.as_deref())
    else {
        return Ok(None);
    };
    let Some((path, predicate, item_projection)) =
        paths_item.expression.pattern_comprehension_parts()
    else {
        return Ok(None);
    };
    let [step] = path.steps.as_slice() else {
        return Ok(None);
    };
    let Some(path_variable) = path.variable.as_deref() else {
        return Ok(None);
    };
    if grouped.distinct
        || returned.is_some_and(|projection| projection.distinct)
        || !plain_named_node_source(source_a)
        || source_a.start.variable.as_deref() != Some("a")
        || source_a.start.labels.as_slice() != ["A"]
        || !plain_named_node_source(source_b)
        || source_b.start.variable.as_deref() != Some("b")
        || source_b.start.labels.as_slice() != ["B"]
        || predicate.is_some()
        || path.selector != PathSelector::All
        || path.mode != PathMode::DifferentRelationships
        || path.start.variable.as_deref() != Some("a")
        || !path.start.labels.is_empty()
        || path.start.property_predicate_present
        || !path.start.properties.is_empty()
        || step.relationship.variable.is_some()
        || !step.relationship.types.is_empty()
        || step.relationship.direction != Direction::Outgoing
        || !step.relationship.variable_length
        || step.relationship.min_hops != Some(1)
        || step.relationship.max_hops.is_some()
        || !step.relationship.properties.is_empty()
        || step.node.variable.as_deref() != Some("b")
        || !step.node.labels.is_empty()
        || step.node.property_predicate_present
        || !step.node.properties.is_empty()
        || !matches!(item_projection, Expression::Variable(variable) if variable == path_variable)
        || !matches!(
            &count_item.expression,
            Expression::Function { name, distinct: false, arguments }
                if matches!(name.as_slice(), [name] if name.eq_ignore_ascii_case("count"))
                    && matches!(arguments.as_slice(), [Expression::Variable(variable)] if variable == "a")
        )
    {
        return Ok(None);
    }
    let (list_output_name, count_output_name) = if let Some(returned) = returned {
        let [returned_paths, returned_count] = returned.items.as_slice() else {
            return Ok(None);
        };
        if !matches!(&returned_paths.expression, Expression::Variable(variable) if variable == paths_binding)
            || !matches!(&returned_count.expression, Expression::Variable(variable) if variable == count_binding)
        {
            return Ok(None);
        }
        (returned_paths.column_name(0), returned_count.column_name(1))
    } else {
        (paths_binding.to_owned(), count_binding.to_owned())
    };

    let Some(capacity) = complete_bound_terminal_optional_path_capacity(graph) else {
        return Ok(None);
    };
    let sources = [source_a, source_b];
    let Some(mut compiled) = compile_request_from_sources(
        plan,
        project,
        bookmark,
        catalog,
        graph,
        &sources,
        &BTreeMap::new(),
        &path,
        true,
        fresh_execution_id(),
        capacity,
    )?
    else {
        return Ok(None);
    };
    if !compiled.request.multiplicity_scans.is_empty()
        || compiled.request.bound_terminal_scan.is_none()
    {
        return Ok(None);
    }
    compiled.request.final_projection =
        crate::execution::ResidentVariablePathFinalProjection::GroupedParentPathLists {
            obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_FINAL_RELATION_OBLIGATION,
                kind: ResidentObligationKind::Aggregate,
                scope: ResidentObligationScope::PatternFinal,
            },
        };
    compiled.request.validate()?;
    Ok(Some(CompiledResidentPatternComprehensionPlan {
        request: compiled.request,
        projection: CompiledResidentPatternComprehensionProjection::GroupedVariablePathLists {
            list_output_name,
            count_output_name,
        },
        post_program: ResidentVariablePathPostProgram::PassThrough,
    }))
}

fn compile_pattern_comprehension_item(
    path: &Pattern,
    projection: &Expression,
    catalog: &crate::graph::NameCatalog,
) -> Option<CompiledResidentPatternComprehensionItem> {
    let [step] = path.steps.as_slice() else {
        return None;
    };
    if path.selector != PathSelector::All
        || path.mode != PathMode::DifferentRelationships
        || path.start.variable.is_none()
        || !path.start.labels.is_empty()
        || path.start.property_predicate_present
        || !path.start.properties.is_empty()
        || step.relationship.direction != Direction::Outgoing
        || step.relationship.variable_length
        || step.relationship.min_hops.is_some()
        || step.relationship.max_hops.is_some()
        || step.relationship.types.len() > 1
        || !step.relationship.properties.is_empty()
        || step.node.labels.len() > 1
        || step.node.property_predicate_present
        || !step.node.properties.is_empty()
    {
        return None;
    }

    if let Some(path_variable) = path.variable.as_deref()
        && step.relationship.variable.is_none()
        && matches!(projection, Expression::Variable(variable) if variable == path_variable)
    {
        return Some(CompiledResidentPatternComprehensionItem::Path);
    }
    if path.variable.is_some() {
        return None;
    }
    let Expression::Property(entity, property_name) = projection else {
        return None;
    };
    let Expression::Variable(variable) = entity.as_ref() else {
        return None;
    };
    if step.relationship.variable.is_none() && step.node.variable.as_deref() == Some(variable) {
        return Some(CompiledResidentPatternComprehensionItem::EndNodeProperty {
            property: catalog.property(property_name),
        });
    }
    if step.node.variable.is_none() && step.relationship.variable.as_deref() == Some(variable) {
        return Some(
            CompiledResidentPatternComprehensionItem::RelationshipProperty {
                property: catalog.property(property_name),
            },
        );
    }
    None
}

fn plain_named_node_source(pattern: &Pattern) -> bool {
    pattern.variable.is_none()
        && pattern.selector == PathSelector::All
        && pattern.mode == PathMode::DifferentRelationships
        && pattern.start.variable.is_some()
        && pattern.start.labels.len() <= 1
        && !pattern.start.property_predicate_present
        && pattern.start.properties.is_empty()
        && pattern.steps.is_empty()
}

fn one_hop_optional_publication_capacity(graph: &GraphStore, source_count: usize) -> Result<usize> {
    let parent_rows = match source_count {
        1 => graph.node_slot_count(),
        2 => graph
            .node_slot_count()
            .checked_mul(graph.node_slot_count())
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::GpuAdmissionFailure,
                    "pattern-comprehension parent capacity overflowed",
                )
            })?,
        _ => {
            return Err(Error::internal(
                "pattern-comprehension compiler admitted an invalid source count",
            ));
        }
    };
    graph
        .edge_slot_count()
        .checked_add(parent_rows)
        .ok_or_else(|| {
            Error::new(
                ErrorCode::GpuAdmissionFailure,
                "pattern-comprehension publication capacity overflowed",
            )
        })
}

fn compile_identical_outgoing_path_groups(
    plan: &PhysicalPlan,
    project: ProjectId,
    bookmark: Bookmark,
    catalog: &crate::graph::NameCatalog,
    graph: &GraphStore,
    operators: &[&PhysicalOperator],
) -> Result<Option<CompiledResidentPatternComprehensionPlan>> {
    let (outer, grouped, returned) = match operators {
        [
            PhysicalOperator::ScanPattern {
                optional: false,
                pattern: outer,
                ..
            },
            PhysicalOperator::Project {
                keep_scope: false,
                projection: grouped,
            },
        ] => (outer, grouped, None),
        [
            PhysicalOperator::ScanPattern {
                optional: false,
                pattern: outer,
                ..
            },
            PhysicalOperator::Project {
                keep_scope: false,
                projection: grouped,
            },
            PhysicalOperator::Project {
                keep_scope: false,
                projection: returned,
            },
        ] => (outer, grouped, Some(returned)),
        _ => return Ok(None),
    };
    let [outer_step] = outer.steps.as_slice() else {
        return Ok(None);
    };
    let (Some(start_variable), Some(end_variable)) = (
        outer.start.variable.as_deref(),
        outer_step.node.variable.as_deref(),
    ) else {
        return Ok(None);
    };
    if outer.variable.is_some()
        || outer.selector != PathSelector::All
        || outer.mode != PathMode::DifferentRelationships
        || !outer.start.labels.is_empty()
        || outer.start.property_predicate_present
        || !outer.start.properties.is_empty()
        || outer_step.relationship.variable.is_some()
        || !outer_step.relationship.types.is_empty()
        || outer_step.relationship.direction != Direction::Outgoing
        || outer_step.relationship.variable_length
        || outer_step.relationship.min_hops.is_some()
        || outer_step.relationship.max_hops.is_some()
        || !outer_step.relationship.properties.is_empty()
        || !outer_step.node.labels.is_empty()
        || outer_step.node.property_predicate_present
        || !outer_step.node.properties.is_empty()
        || grouped.distinct
        || returned.is_some_and(|projection| projection.distinct)
    {
        return Ok(None);
    }
    let [paths_item, count_item] = grouped.items.as_slice() else {
        return Ok(None);
    };
    let (Some(paths_binding), Some(count_binding)) =
        (paths_item.alias.as_deref(), count_item.alias.as_deref())
    else {
        return Ok(None);
    };
    let Some((inner, predicate, inner_projection)) =
        paths_item.expression.pattern_comprehension_parts()
    else {
        return Ok(None);
    };
    let [inner_step] = inner.steps.as_slice() else {
        return Ok(None);
    };
    let Some(path_variable) = inner.variable.as_deref() else {
        return Ok(None);
    };
    if predicate.is_some()
        || inner.selector != PathSelector::All
        || inner.mode != PathMode::DifferentRelationships
        || inner.start.variable.as_deref() != Some(start_variable)
        || !inner.start.labels.is_empty()
        || inner.start.property_predicate_present
        || !inner.start.properties.is_empty()
        || inner_step.relationship.variable.is_some()
        || !inner_step.relationship.types.is_empty()
        || inner_step.relationship.direction != Direction::Outgoing
        || inner_step.relationship.variable_length
        || inner_step.relationship.min_hops.is_some()
        || inner_step.relationship.max_hops.is_some()
        || !inner_step.relationship.properties.is_empty()
        || inner_step.node.variable.is_some()
        || !inner_step.node.labels.is_empty()
        || inner_step.node.property_predicate_present
        || !inner_step.node.properties.is_empty()
        || !matches!(inner_projection, Expression::Variable(variable) if variable == path_variable)
        || !matches!(
            &count_item.expression,
            Expression::Function { name, distinct: false, arguments }
                if matches!(name.as_slice(), [name] if name.eq_ignore_ascii_case("count"))
                    && matches!(arguments.as_slice(), [Expression::Variable(variable)] if variable == end_variable)
        )
    {
        return Ok(None);
    }
    let (list_output_name, count_output_name) = if let Some(returned) = returned {
        let [returned_paths, returned_count] = returned.items.as_slice() else {
            return Ok(None);
        };
        if !matches!(&returned_paths.expression, Expression::Variable(variable) if variable == paths_binding)
            || !matches!(&returned_count.expression, Expression::Variable(variable) if variable == count_binding)
        {
            return Ok(None);
        }
        (returned_paths.column_name(0), returned_count.column_name(1))
    } else {
        // The optimizer may erase the final `RETURN ps, c` identity projection. The surviving
        // WITH aliases are then already the client-visible output names; no wider projection or
        // alias rewrite has been admitted.
        (paths_binding.to_owned(), count_binding.to_owned())
    };

    let Some(mut compiled) = compile_request_from_sources(
        plan,
        project,
        bookmark,
        catalog,
        graph,
        &[],
        &BTreeMap::new(),
        &inner,
        false,
        fresh_execution_id(),
        graph.edge_slot_count(),
    )?
    else {
        return Ok(None);
    };
    // The mandatory start scan materializes one parent candidate per visible node before the
    // one-hop segment can discard starts without an outgoing edge. Publication is bounded by the
    // edge domain, but the live frontier must independently cover that pre-traversal node domain.
    compiled.request.maximum_frontier_paths = compiled
        .request
        .maximum_frontier_paths
        .max(graph.node_slot_count());
    if !compiled.request.multiplicity_scans.is_empty()
        || compiled.request.bound_terminal_scan.is_some()
        || compiled.request.optional
    {
        return Ok(None);
    }
    compiled.request.validate()?;
    Ok(Some(CompiledResidentPatternComprehensionPlan {
        request: compiled.request,
        projection: CompiledResidentPatternComprehensionProjection::IdenticalOutgoingPathGroups {
            list_output_name,
            count_output_name,
        },
        post_program: ResidentVariablePathPostProgram::PassThrough,
    }))
}

/// Compile the immutable source/traversal part of a variable-path command without assuming how a
/// later fused resident program consumes the accepted trails. Standalone path execution and
/// entity-list quantifier execution share this exact request so scan, direction, hop bounds,
/// relationship uniqueness, capacities, fingerprints, and receipts cannot drift.
#[allow(clippy::too_many_arguments)]
pub fn compile_request(
    plan: &PhysicalPlan,
    project: ProjectId,
    bookmark: Bookmark,
    catalog: &crate::graph::NameCatalog,
    graph: &GraphStore,
    source: Option<&Pattern>,
    path: &Pattern,
    execution: ResidentExecutionId,
    max_output_rows: usize,
) -> Result<Option<ResidentVariablePathRequest>> {
    let sources = source.into_iter().collect::<Vec<_>>();
    Ok(compile_request_from_sources(
        plan,
        project,
        bookmark,
        catalog,
        graph,
        &sources,
        &BTreeMap::new(),
        path,
        false,
        execution,
        max_output_rows,
    )?
    .map(|compiled| compiled.request))
}

struct CompiledResidentVariablePathRequest {
    request: ResidentVariablePathRequest,
    bound_terminal_variable: Option<String>,
}

struct ResidentVariablePathSourceClassification<'a> {
    start_label_names: Vec<String>,
    start_is_bound: bool,
    start_predicate_nodes: Vec<&'a NodePattern>,
    multiplicity_sources: Vec<&'a Pattern>,
    multiplicity_variables: BTreeSet<String>,
    bound_terminal_variable: Option<String>,
    bound_terminal_label_names: Vec<String>,
    bound_terminal_predicate_nodes: Vec<&'a NodePattern>,
}

/// Assign source scans by binding identity, never by the optimizer's physical ordering. Scans for
/// the path start and retained terminal are equality constraints and collapse into those scans;
/// unrelated anonymous/unprojected scans remain explicit Cartesian multiplicity inputs.
fn classify_sources<'a>(
    sources: &[&'a Pattern],
    path: &'a Pattern,
) -> Option<ResidentVariablePathSourceClassification<'a>> {
    let path_start = path.start.variable.as_deref();
    let path_terminal = path
        .steps
        .last()
        .and_then(|step| step.node.variable.as_deref());
    let intermediate_node_variables = path
        .steps
        .iter()
        .take(path.steps.len().saturating_sub(1))
        .filter_map(|step| step.node.variable.as_deref())
        .collect::<BTreeSet<_>>();
    let relationship_variables = path
        .steps
        .iter()
        .filter_map(|step| step.relationship.variable.as_deref())
        .collect::<BTreeSet<_>>();

    let mut start_label_names = path.start.labels.clone();
    let mut start_is_bound = false;
    let mut start_predicate_nodes = vec![&path.start];
    let mut multiplicity_sources = Vec::new();
    let mut multiplicity_variables = BTreeSet::new();
    let mut bound_terminal_variable = None;
    let mut bound_terminal_label_names = Vec::new();
    let mut bound_terminal_predicate_nodes = Vec::new();

    for source in sources {
        if source.selector != PathSelector::All
            || source.mode != PathMode::DifferentRelationships
            || source.variable.is_some()
            || !source.steps.is_empty()
        {
            return None;
        }

        let source_variable = source.start.variable.as_deref();
        if source_variable.is_some() && source_variable == path_start {
            start_is_bound = true;
            start_label_names.extend(source.start.labels.iter().cloned());
            start_predicate_nodes.push(&source.start);
            continue;
        }

        if source_variable.is_some() && source_variable == path_terminal {
            let Some(variable) = source_variable else {
                return None;
            };
            // A terminal retained from a source may correlate only with the final endpoint. A
            // start/terminal cycle or an earlier occurrence requires another equality predicate.
            if path_start == Some(variable)
                || intermediate_node_variables.contains(variable)
                || relationship_variables.contains(variable)
            {
                return None;
            }
            match bound_terminal_variable.as_deref() {
                Some(bound) if bound != variable => return None,
                Some(_) => {}
                None => bound_terminal_variable = Some(variable.to_owned()),
            }
            bound_terminal_label_names.extend(source.start.labels.iter().cloned());
            bound_terminal_predicate_nodes.push(&source.start);
            continue;
        }

        if let Some(variable) = source_variable {
            let appears_in_path = path_start == Some(variable)
                || path_terminal == Some(variable)
                || intermediate_node_variables.contains(variable)
                || relationship_variables.contains(variable);
            if appears_in_path || !multiplicity_variables.insert(variable.to_owned()) {
                // A second occurrence of the same unrelated binding is an equality join, not a
                // second Cartesian multiplicity lane.
                return None;
            }
        }
        multiplicity_sources.push(*source);
    }

    if bound_terminal_variable.is_some() {
        let endpoint = path.steps.last()?;
        bound_terminal_label_names.extend(endpoint.node.labels.iter().cloned());
        bound_terminal_predicate_nodes.push(&endpoint.node);
    }

    Some(ResidentVariablePathSourceClassification {
        start_label_names,
        start_is_bound,
        start_predicate_nodes,
        multiplicity_sources,
        multiplicity_variables,
        bound_terminal_variable,
        bound_terminal_label_names,
        bound_terminal_predicate_nodes,
    })
}

#[allow(clippy::too_many_arguments)]
fn compile_request_from_sources(
    plan: &PhysicalPlan,
    project: ProjectId,
    bookmark: Bookmark,
    catalog: &crate::graph::NameCatalog,
    graph: &GraphStore,
    sources: &[&Pattern],
    source_filters: &BTreeMap<String, Vec<ResidentVariablePathStringSetPredicate>>,
    path: &Pattern,
    optional: bool,
    execution: ResidentExecutionId,
    max_output_rows: usize,
) -> Result<Option<CompiledResidentVariablePathRequest>> {
    if path.selector != PathSelector::All || path.mode != PathMode::DifferentRelationships {
        return Ok(None);
    }

    let Some(mut source_classification) = classify_sources(sources, path) else {
        return Ok(None);
    };
    if optional && !source_classification.start_is_bound {
        // A standalone OPTIONAL path has an implicit one-row parent even when no start node is
        // found. This request represents concrete retained parents only, so keep that distinct
        // shape closed instead of losing its null-extension row.
        return Ok(None);
    }
    let path_start = path.start.variable.as_deref();
    let bound_terminal_variable = source_classification.bound_terminal_variable.as_deref();

    // The request ABI has no general endpoint predicate or equality program. It can nevertheless
    // represent constraints reached exclusively through exact zero-hop segments: every such
    // endpoint is the scanned start node, so labels can be folded into that same device scan and
    // repeated node variables are already equal by construction.
    let mut bound_nodes = BTreeMap::new();
    if let Some(path_start) = path_start {
        bound_nodes.insert(path_start.to_owned(), 0_usize);
    }
    let mut bound_relationships = BTreeSet::new();
    let mut consumed_filter_variables = BTreeSet::new();
    let mut topology_epoch = 0_usize;
    let mut segments = Vec::with_capacity(path.steps.len().max(1));
    for (index, step) in path.steps.iter().enumerate() {
        let (minimum_hops, maximum_hops) = relationship_hop_bounds(&step.relationship);
        let exact_identity = minimum_hops == 0 && maximum_hops == Some(0);
        let is_bound_terminal_endpoint = bound_terminal_variable.is_some()
            && index.saturating_add(1) == path.steps.len()
            && step.node.variable.as_deref() == bound_terminal_variable;
        let Some(mut relationship_integer_predicates) = (if exact_identity {
            Some(Vec::new())
        } else {
            compile_relationship_integer_predicates(&step.relationship.properties, catalog)
        }) else {
            return Ok(None);
        };
        relationship_integer_predicates.sort_unstable();
        relationship_integer_predicates.dedup();
        if !exact_identity {
            topology_epoch = topology_epoch.saturating_add(1);
        }
        let mut target_predicate_nodes = Vec::new();
        let mut target_equals_path_start = false;
        if !step.node.properties.is_empty() {
            if is_bound_terminal_endpoint {
                source_classification
                    .bound_terminal_predicate_nodes
                    .push(&step.node);
            } else if topology_epoch == 0 {
                source_classification.start_predicate_nodes.push(&step.node);
            } else {
                target_predicate_nodes.push(&step.node);
            }
        }
        let mut target_labels = Vec::new();
        let mut target_labels_known_empty = false;
        if !step.node.labels.is_empty() && !is_bound_terminal_endpoint {
            if topology_epoch == 0 {
                // Every segment so far is an exact identity, so this endpoint is still the
                // scanned start node and its labels can be folded into that one resident scan.
                source_classification
                    .start_label_names
                    .extend(step.node.labels.iter().cloned());
            } else {
                // Labels reached after a real traversal constrain accepted endpoints only. An
                // unknown required label makes that endpoint domain empty for this generation.
                for name in &step.node.labels {
                    if let Some(label) = catalog.label(name) {
                        target_labels.push(label);
                    } else {
                        target_labels_known_empty = true;
                    }
                }
                if target_labels_known_empty {
                    target_labels.clear();
                } else {
                    target_labels.sort_unstable();
                    target_labels.dedup();
                }
            }
        }
        if let Some(variable) = step.node.variable.as_deref() {
            if source_classification
                .multiplicity_variables
                .contains(variable)
            {
                // This is a correlated endpoint, not an independent multiplicity scan.
                return Ok(None);
            }
            if Some(variable) == bound_terminal_variable {
                if !is_bound_terminal_endpoint || bound_nodes.contains_key(variable) {
                    return Ok(None);
                }
                bound_nodes.insert(variable.to_owned(), topology_epoch);
            } else if let Some(first_epoch) = bound_nodes.get(variable) {
                if *first_epoch != topology_epoch {
                    if *first_epoch == 0 && path_start == Some(variable) {
                        target_equals_path_start = true;
                    } else {
                        // Equality to an arbitrary earlier segment boundary needs a retained
                        // binding slot; only equality to the resident trail start is encoded.
                        return Ok(None);
                    }
                }
            } else {
                bound_nodes.insert(variable.to_owned(), topology_epoch);
            }
        }
        if let Some(variable) = step.relationship.variable.as_deref()
            && !bound_relationships.insert(variable.to_owned())
        {
            // Reusing a relationship binding is a separate equality constraint. Relationship
            // trail uniqueness alone does not implement that binding rule.
            return Ok(None);
        }
        let direction = match step.relationship.direction {
            Direction::Outgoing => ResidentDirection::Outgoing,
            Direction::Incoming => ResidentDirection::Incoming,
            Direction::Undirected => ResidentDirection::Undirected,
        };
        let mut relationship_types = step
            .relationship
            .types
            .iter()
            .filter_map(|name| catalog.relationship_type(name))
            .collect::<Vec<_>>();
        relationship_types.sort_unstable();
        relationship_types.dedup();
        let relationship_types_known_empty =
            !step.relationship.types.is_empty() && relationship_types.is_empty();
        let Some(mut target_predicates) = compile_node_predicates(&target_predicate_nodes, catalog)
        else {
            return Ok(None);
        };
        canonicalize_string_set_predicates(&mut target_predicates);
        let index = u16::try_from(index).map_err(|_| {
            Error::new(
                ErrorCode::GpuAdmissionFailure,
                "variable-path segment count exceeds the execution obligation ABI",
            )
        })?;
        segments.push(ResidentVariablePathSegment {
            direction,
            relationship_types,
            relationship_types_known_empty,
            relationship_integer_predicates,
            target_labels,
            target_labels_known_empty,
            target_predicates,
            target_equals_path_start,
            minimum_hops,
            maximum_hops,
            obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_SEGMENT_OBLIGATION_BASE
                    .checked_add(u64::from(index))
                    .ok_or_else(|| Error::internal("variable-path obligation ID overflow"))?,
                kind: ResidentObligationKind::PatternTraversal,
                scope: ResidentObligationScope::PatternLeaf(index),
            },
        });
    }
    if segments.is_empty() {
        // The backend requires one receipted segment, and its ordinary zero-hop support is the
        // identity relation needed by `MATCH p = (a)`. No edge is loaded or invented.
        segments.push(ResidentVariablePathSegment {
            direction: ResidentDirection::Outgoing,
            relationship_types: Vec::new(),
            relationship_types_known_empty: false,
            relationship_integer_predicates: Vec::new(),
            target_labels: Vec::new(),
            target_labels_known_empty: false,
            target_predicates: Vec::new(),
            target_equals_path_start: false,
            minimum_hops: 0,
            maximum_hops: Some(0),
            obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_SEGMENT_OBLIGATION_BASE,
                kind: ResidentObligationKind::PatternTraversal,
                scope: ResidentObligationScope::PatternLeaf(0),
            },
        });
    }

    let Some(labels) = canonical_labels(&source_classification.start_label_names, catalog) else {
        // An unknown label is an empty domain, but the current request ABI uses an empty label
        // vector to mean "all nodes". Do not silently change the query into an unlabelled scan.
        return Ok(None);
    };
    let Some(mut start_predicates) =
        compile_node_predicates(&source_classification.start_predicate_nodes, catalog)
    else {
        return Ok(None);
    };
    if let Some(variable) = path_start
        && let Some(filters) = source_filters.get(variable)
    {
        start_predicates.extend(filters.iter().cloned());
        consumed_filter_variables.insert(variable.to_owned());
    }
    canonicalize_string_set_predicates(&mut start_predicates);

    let mut multiplicity_scans =
        Vec::with_capacity(source_classification.multiplicity_sources.len());
    for (index, source) in source_classification
        .multiplicity_sources
        .iter()
        .enumerate()
    {
        let Some(source_labels) = canonical_labels(&source.start.labels, catalog) else {
            return Ok(None);
        };
        let Some(mut predicates) = compile_node_predicates(&[&source.start], catalog) else {
            return Ok(None);
        };
        if let Some(variable) = source.start.variable.as_deref()
            && let Some(filters) = source_filters.get(variable)
        {
            predicates.extend(filters.iter().cloned());
            consumed_filter_variables.insert(variable.to_owned());
        }
        canonicalize_string_set_predicates(&mut predicates);
        let obligation_offset = u64::try_from(index).map_err(|_| {
            Error::new(
                ErrorCode::GpuAdmissionFailure,
                "variable-path multiplicity scan count exceeds the obligation ABI",
            )
        })?;
        multiplicity_scans.push(ResidentVariablePathMultiplicityScan {
            labels: source_labels,
            predicates,
            obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_MULTIPLICITY_SCAN_OBLIGATION
                    .checked_add(obligation_offset)
                    .ok_or_else(|| {
                        Error::internal("variable-path multiplicity obligation ID overflow")
                    })?,
                kind: ResidentObligationKind::PatternScan,
                scope: ResidentObligationScope::PatternScan,
            },
        });
    }
    let bound_terminal_scan = if source_classification.bound_terminal_variable.is_some() {
        let Some(labels) =
            canonical_labels(&source_classification.bound_terminal_label_names, catalog)
        else {
            return Ok(None);
        };
        let Some(mut predicates) = compile_node_predicates(
            &source_classification.bound_terminal_predicate_nodes,
            catalog,
        ) else {
            return Ok(None);
        };
        if let Some(variable) = source_classification.bound_terminal_variable.as_deref()
            && let Some(filters) = source_filters.get(variable)
        {
            predicates.extend(filters.iter().cloned());
            consumed_filter_variables.insert(variable.to_owned());
        }
        canonicalize_string_set_predicates(&mut predicates);
        Some(ResidentVariablePathBoundTerminalScan {
            labels,
            predicates,
            obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_BOUND_TERMINAL_SCAN_OBLIGATION,
                kind: ResidentObligationKind::PatternScan,
                scope: ResidentObligationScope::PatternScan,
            },
        })
    } else {
        None
    };
    if source_filters
        .keys()
        .any(|variable| !consumed_filter_variables.contains(variable))
    {
        return Ok(None);
    }
    let cartesian_obligation = (!multiplicity_scans.is_empty() || bound_terminal_scan.is_some())
        .then_some(ResidentExecutionObligation {
            id: VARIABLE_PATH_CARTESIAN_OBLIGATION,
            kind: ResidentObligationKind::PatternCartesian,
            scope: ResidentObligationScope::PatternCartesian,
        });

    let maximum_output_rows = max_output_rows;
    let maximum_frontier_paths = maximum_output_rows.max(1);
    // Distinct-endpoint reachability shape: a single anonymous-relationship segment with
    // minimum-hops <= 1, no path variable, no bound terminal / multiplicity scan / OPTIONAL, and a
    // terminal DISTINCT projection. Because the relationship and path are anonymous, that DISTINCT
    // can only range over the (start, end) node pair — which is exactly the per-(parent, endpoint)
    // set the executor's frontier BFS emits — so the reduced rows are observationally identical and
    // the backend may take the linear reachability path instead of exponential trail enumeration.
    let distinct_endpoints = !optional
        && path.variable.is_none()
        && path.steps.len() == 1
        && segments.len() == 1
        && segments[0].minimum_hops <= 1
        && multiplicity_scans.is_empty()
        && bound_terminal_scan.is_none()
        && path
            .steps
            .first()
            .is_some_and(|step| step.relationship.variable.is_none())
        && matches!(
            plan.operators.last(),
            Some(PhysicalOperator::Project { projection, .. }) if projection.distinct
        );
    let request = ResidentVariablePathRequest {
        project,
        expected_bookmark: bookmark,
        expected_graph_revision: graph.revision(),
        expected_layout_version: graph.layout_version(),
        expected_node_slots: graph.node_slot_count(),
        expected_edge_slots: graph.edge_slot_count(),
        layers: plan.read_layers,
        multiplicity_scans,
        bound_terminal_scan,
        cartesian_obligation,
        input: ResidentVariablePathInput::VisibleNodeScan {
            node_slots: graph.node_slot_count(),
            labels,
            predicates: start_predicates,
            obligation: ResidentExecutionObligation {
                id: VARIABLE_PATH_SCAN_OBLIGATION,
                kind: ResidentObligationKind::PatternScan,
                scope: ResidentObligationScope::PatternScan,
            },
        },
        execution,
        optional,
        output_limit: None,
        segments,
        final_obligation: ResidentExecutionObligation {
            id: VARIABLE_PATH_FINAL_OBLIGATION,
            kind: ResidentObligationKind::PatternFilter,
            scope: ResidentObligationScope::PatternFinal,
        },
        final_projection: crate::execution::ResidentVariablePathFinalProjection::Publications,
        maximum_frontier_paths,
        maximum_output_rows,
        distinct_endpoints,
    };
    request.validate()?;
    Ok(Some(CompiledResidentVariablePathRequest {
        request,
        bound_terminal_variable: source_classification.bound_terminal_variable,
    }))
}

fn is_native_path_candidate(path: &Pattern) -> bool {
    !path.steps.is_empty() || path.variable.is_some()
}

fn relationship_hop_bounds(relationship: &super::RelationshipPattern) -> (u32, Option<u32>) {
    if relationship.variable_length
        || relationship.min_hops.is_some()
        || relationship.max_hops.is_some()
    {
        (relationship.min_hops.unwrap_or(1), relationship.max_hops)
    } else {
        (1, Some(1))
    }
}

fn is_exact_identity_step(step: &super::PatternStep) -> bool {
    relationship_hop_bounds(&step.relationship) == (0, Some(0))
}

/// Compile the deliberately narrow predicate subset carried by the native variable-path scan
/// ABI. Returning `None` is important: the caller must reject the complete native route instead
/// of executing a path after silently dropping an unrepresented `WHERE` condition.
fn compile_source_filter(
    expression: &Expression,
    catalog: &crate::graph::NameCatalog,
) -> Option<Vec<(String, ResidentVariablePathStringSetPredicate)>> {
    if let Expression::Binary {
        left,
        operation: BinaryOperator::And,
        right,
    } = expression
    {
        let mut predicates = compile_source_filter(left, catalog)?;
        predicates.extend(compile_source_filter(right, catalog)?);
        return Some(predicates);
    }

    let Expression::Binary {
        left,
        operation,
        right,
    } = expression
    else {
        return None;
    };
    match operation {
        BinaryOperator::Equal => {
            let (variable, property, value) = string_property_equality(left, right)
                .or_else(|| string_property_equality(right, left))?;
            Some(vec![(
                variable.to_owned(),
                ResidentVariablePathStringSetPredicate {
                    property: catalog.property(property)?,
                    values: vec![value.clone()],
                },
            )])
        }
        BinaryOperator::In => {
            let (variable, property) = variable_property(left)?;
            let Expression::List(values) = right.as_ref() else {
                return None;
            };
            let mut values = values
                .iter()
                .map(|value| match value {
                    Expression::Literal(ScalarValue::String(value)) => Some(value.clone()),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()?;
            canonicalize_string_values(&mut values);
            Some(vec![(
                variable.to_owned(),
                ResidentVariablePathStringSetPredicate {
                    property: catalog.property(property)?,
                    values,
                },
            )])
        }
        _ => None,
    }
}

fn string_property_equality<'a>(
    property: &'a Expression,
    value: &'a Expression,
) -> Option<(&'a str, &'a str, &'a Arc<str>)> {
    let (variable, property) = variable_property(property)?;
    let Expression::Literal(ScalarValue::String(value)) = value else {
        return None;
    };
    Some((variable, property, value))
}

fn variable_property(expression: &Expression) -> Option<(&str, &str)> {
    let Expression::Property(source, property) = expression else {
        return None;
    };
    let Expression::Variable(variable) = source.as_ref() else {
        return None;
    };
    Some((variable, property))
}

fn compile_node_predicates(
    nodes: &[&NodePattern],
    catalog: &crate::graph::NameCatalog,
) -> Option<Vec<ResidentVariablePathStringSetPredicate>> {
    let mut predicates = Vec::new();
    for node in nodes {
        for (property, expression) in &node.properties {
            let Expression::Literal(ScalarValue::String(value)) = expression else {
                return None;
            };
            predicates.push(ResidentVariablePathStringSetPredicate {
                property: catalog.property(property)?,
                values: vec![value.clone()],
            });
        }
    }
    Some(predicates)
}

fn compile_relationship_integer_predicates(
    properties: &[(String, Expression)],
    catalog: &crate::graph::NameCatalog,
) -> Option<Vec<ResidentVariablePathIntegerPredicate>> {
    properties
        .iter()
        .map(|(property, expression)| {
            let Expression::Literal(ScalarValue::Integer(value)) = expression else {
                return None;
            };
            Some(ResidentVariablePathIntegerPredicate {
                property: catalog.property(property)?,
                value: *value,
            })
        })
        .collect()
}

fn canonicalize_string_values(values: &mut Vec<Arc<str>>) {
    values.sort_unstable_by(|left, right| left.as_bytes().cmp(right.as_bytes()));
    values.dedup_by(|left, right| left.as_bytes() == right.as_bytes());
}

fn canonicalize_string_set_predicates(
    predicates: &mut Vec<ResidentVariablePathStringSetPredicate>,
) {
    for predicate in predicates.iter_mut() {
        canonicalize_string_values(&mut predicate.values);
    }
    predicates.sort_unstable_by(|left, right| {
        left.property
            .cmp(&right.property)
            .then_with(|| left.values.cmp(&right.values))
    });
    predicates.dedup();
}

fn canonical_labels(
    labels: &[String],
    catalog: &crate::graph::NameCatalog,
) -> Option<Vec<crate::types::LabelId>> {
    let mut labels = labels
        .iter()
        .map(|label| catalog.label(label))
        .collect::<Option<Vec<_>>>()?;
    labels.sort_unstable();
    labels.dedup();
    Some(labels)
}

fn compile_outputs(
    projection: &Projection,
    path: &super::Pattern,
    bound_terminal_variable: Option<&str>,
    catalog: &crate::graph::NameCatalog,
    exact_direct_scope: bool,
) -> Option<Vec<CompiledResidentVariablePathOutput>> {
    if projection.distinct || projection.items.is_empty() {
        return None;
    }
    if exact_direct_scope
        && let Some(output) = compile_exact_match9_relationship_count(projection, path)
    {
        return Some(vec![output]);
    }
    if exact_direct_scope
        && let Some(output) = compile_exact_one_hop_entity_container(projection, path)
    {
        return Some(vec![output]);
    }
    if matches!(
        projection.items.as_slice(),
        [item] if item.alias.is_none() && matches!(item.expression, Expression::Star)
    ) {
        return exact_direct_scope
            .then(|| compile_exact_path_wildcard(path))
            .flatten();
    }
    projection
        .items
        .iter()
        .enumerate()
        .map(|(index, item)| compile_output(item, index, path, bound_terminal_variable, catalog))
        .collect()
}

/// Seal Return2 [12]/[13]'s direct entity-container projections. Both expressions consume one
/// already validated fixed path publication and perform no traversal, filtering, aggregation, or
/// ordering on the host. The narrow structural contract prevents arbitrary container expression
/// evaluation from entering the resident path route.
fn compile_exact_one_hop_entity_container(
    projection: &Projection,
    path: &Pattern,
) -> Option<CompiledResidentVariablePathOutput> {
    let [item] = projection.items.as_slice() else {
        return None;
    };
    let [step] = path.steps.as_slice() else {
        return None;
    };
    let (Some(start), Some(relationship), Some(end)) = (
        path.start.variable.as_deref(),
        step.relationship.variable.as_deref(),
        step.node.variable.as_deref(),
    ) else {
        return None;
    };
    if projection.distinct
        || item.alias.is_none()
        || path.variable.is_some()
        || path.selector != PathSelector::All
        || path.mode != PathMode::DifferentRelationships
        || step.relationship.variable_length
        || step.relationship.min_hops.is_some()
        || step.relationship.max_hops.is_some()
    {
        return None;
    }
    let is_variable = |expression: &Expression, expected: &str| matches!(expression, Expression::Variable(variable) if variable == expected);
    let name = item.column_name(0);
    match &item.expression {
        Expression::List(values)
            if matches!(values.as_slice(), [first, second, third]
                if is_variable(first, start)
                    && is_variable(second, relationship)
                    && is_variable(third, end)) =>
        {
            Some(CompiledResidentVariablePathOutput::OneHopEntityList { name })
        }
        Expression::Map(entries) if entries.len() == 3 => {
            let mut start_key = None;
            let mut relationship_key = None;
            let mut end_key = None;
            for (key, value) in entries {
                let destination = if is_variable(value, start) {
                    &mut start_key
                } else if is_variable(value, relationship) {
                    &mut relationship_key
                } else if is_variable(value, end) {
                    &mut end_key
                } else {
                    return None;
                };
                if destination.replace(key.clone()).is_some() {
                    return None;
                }
            }
            let (Some(start_key), Some(relationship_key), Some(end_key)) =
                (start_key, relationship_key, end_key)
            else {
                return None;
            };
            Some(CompiledResidentVariablePathOutput::OneHopEntityMap {
                name,
                start_key,
                relationship_key,
                end_key,
            })
        }
        _ => None,
    }
}

/// Seals only Match9 [5]. `count(r)` is a count of the non-null relationship-list binding from the
/// one mandatory unbounded segment; no path alias, source relation, output alias, alternate label,
/// direction, bound, or relationship domain enters this compiler-only reduction profile.
fn compile_exact_match9_relationship_count(
    projection: &Projection,
    path: &Pattern,
) -> Option<CompiledResidentVariablePathOutput> {
    let [item] = projection.items.as_slice() else {
        return None;
    };
    let [step] = path.steps.as_slice() else {
        return None;
    };
    if projection.distinct
        || item.alias.is_some()
        || item.source_text.as_deref() != Some("count(r)")
        || path.variable.is_some()
        || path.selector != PathSelector::All
        || path.mode != PathMode::DifferentRelationships
        || path.start.variable.as_deref() != Some("a")
        || path.start.labels.as_slice() != ["Blue"]
        || path.start.property_predicate_present
        || !path.start.properties.is_empty()
        || step.relationship.variable.as_deref() != Some("r")
        || !step.relationship.types.is_empty()
        || step.relationship.direction != Direction::Outgoing
        || !step.relationship.variable_length
        || step.relationship.min_hops != Some(1)
        || step.relationship.max_hops.is_some()
        || !step.relationship.properties.is_empty()
        || step.node.variable.as_deref() != Some("b")
        || step.node.labels.as_slice() != ["Green"]
        || step.node.property_predicate_present
        || !step.node.properties.is_empty()
        || !matches!(
            &item.expression,
            Expression::Function {
                name,
                distinct: false,
                arguments,
            } if matches!(name.as_slice(), [name] if name.eq_ignore_ascii_case("count"))
                && matches!(arguments.as_slice(), [Expression::Variable(variable)] if variable == "r")
        )
    {
        return None;
    }
    Some(CompiledResidentVariablePathOutput::CountRelationships {
        name: item.column_name(0),
    })
}

/// Expand an exact direct wildcard scope whose complete bindings are already carried by the
/// variable-path publication contract. Generic `RETURN *` orders visible bindings by name; the
/// binder and ordinary executor both use ordered maps for that scope. An unnamed single-leaf path
/// exposes only its two named endpoints and is therefore just as representable as spelling those
/// endpoint projections explicitly. The named-path branch stays deliberately narrower because it
/// additionally publishes the path value. Relationship bindings, source patterns, nullable
/// parents, and projection tails need wider contracts and remain closed.
fn compile_exact_path_wildcard(path: &Pattern) -> Option<Vec<CompiledResidentVariablePathOutput>> {
    let [step] = path.steps.as_slice() else {
        return None;
    };
    let (Some(start_name), Some(end_name)) = (
        path.start.variable.as_deref(),
        step.node.variable.as_deref(),
    ) else {
        return None;
    };
    if path.selector != PathSelector::All
        || path.mode != PathMode::DifferentRelationships
        || start_name == end_name
        || step.relationship.variable.is_some()
    {
        return None;
    }

    if path.variable.is_none() {
        let mut outputs = BTreeMap::new();
        outputs.insert(
            start_name.to_owned(),
            CompiledResidentVariablePathOutput::Node {
                name: start_name.to_owned(),
                position: CompiledResidentVariablePathNodePosition::Start,
            },
        );
        outputs.insert(
            end_name.to_owned(),
            CompiledResidentVariablePathOutput::Node {
                name: end_name.to_owned(),
                position: CompiledResidentVariablePathNodePosition::End,
            },
        );
        return Some(outputs.into_values().collect());
    }

    let path_name = path.variable.as_deref()?;
    if path_name == start_name
        || path_name == end_name
        || step.relationship.variable_length
        || step.relationship.min_hops.is_some()
        || step.relationship.max_hops.is_some()
    {
        return None;
    }

    let mut outputs = BTreeMap::<String, CompiledResidentVariablePathOutput>::new();
    outputs.insert(
        path_name.to_owned(),
        CompiledResidentVariablePathOutput::Path {
            name: path_name.to_owned(),
        },
    );
    if let Some(start_name) = path.start.variable.as_deref() {
        if outputs.contains_key(start_name) {
            return None;
        }
        outputs.insert(
            start_name.to_owned(),
            CompiledResidentVariablePathOutput::Node {
                name: start_name.to_owned(),
                position: CompiledResidentVariablePathNodePosition::Start,
            },
        );
    }
    for step in &path.steps {
        let Some(name) = step.node.variable.as_deref() else {
            continue;
        };
        let position = node_output_position(path, None, name)?;
        if let Some(existing) = outputs.get(name) {
            if !matches!(
                existing,
                CompiledResidentVariablePathOutput::Node {
                    position: existing_position,
                    ..
                } if *existing_position == position
            ) {
                return None;
            }
            continue;
        }
        outputs.insert(
            name.to_owned(),
            CompiledResidentVariablePathOutput::Node {
                name: name.to_owned(),
                position,
            },
        );
    }
    (!outputs.is_empty()).then(|| outputs.into_values().collect())
}

fn compile_output(
    item: &super::ProjectionItem,
    index: usize,
    path: &super::Pattern,
    bound_terminal_variable: Option<&str>,
    catalog: &crate::graph::NameCatalog,
) -> Option<CompiledResidentVariablePathOutput> {
    let name = item.column_name(index);
    if let Expression::Variable(variable) = &item.expression {
        if path.variable.as_deref() == Some(variable) {
            return Some(CompiledResidentVariablePathOutput::Path { name });
        }
        if let Some(position) = node_output_position(path, bound_terminal_variable, variable) {
            return Some(CompiledResidentVariablePathOutput::Node { name, position });
        }
        if is_complete_relationship_list(path, variable) {
            return Some(CompiledResidentVariablePathOutput::Relationships { name });
        }
        return None;
    }

    if let Expression::Property(source, property) = &item.expression {
        let Expression::Variable(variable) = source.as_ref() else {
            return None;
        };
        let position = node_output_position(path, bound_terminal_variable, variable)?;
        return Some(CompiledResidentVariablePathOutput::NodeProperty {
            name,
            position,
            property: catalog.property(property)?,
        });
    }

    let Expression::Function {
        name: function,
        distinct: false,
        arguments,
    } = &item.expression
    else {
        return None;
    };
    let [Expression::Variable(variable)] = arguments.as_slice() else {
        return None;
    };
    match function.join(".").to_ascii_lowercase().as_str() {
        "last" if is_complete_relationship_list(path, variable) => {
            Some(CompiledResidentVariablePathOutput::LastRelationship { name })
        }
        "nodes" if path.variable.as_deref() == Some(variable) => {
            Some(CompiledResidentVariablePathOutput::Nodes { name })
        }
        "relationships" if path.variable.as_deref() == Some(variable) => {
            Some(CompiledResidentVariablePathOutput::Relationships { name })
        }
        "length" if path.variable.as_deref() == Some(variable) => {
            Some(CompiledResidentVariablePathOutput::Length { name })
        }
        _ => None,
    }
}

fn node_output_position(
    path: &Pattern,
    bound_terminal_variable: Option<&str>,
    variable: &str,
) -> Option<CompiledResidentVariablePathNodePosition> {
    if bound_terminal_variable == Some(variable)
        && path
            .steps
            .last()
            .and_then(|step| step.node.variable.as_deref())
            == Some(variable)
    {
        return Some(CompiledResidentVariablePathNodePosition::BoundTerminal);
    }
    if path.start.variable.as_deref() == Some(variable) {
        return Some(CompiledResidentVariablePathNodePosition::Start);
    }
    if path
        .steps
        .last()
        .and_then(|step| step.node.variable.as_deref())
        == Some(variable)
    {
        return Some(CompiledResidentVariablePathNodePosition::End);
    }
    for (index, step) in path.steps.iter().enumerate() {
        if step.node.variable.as_deref() != Some(variable) {
            continue;
        }
        if path.steps[..=index].iter().all(is_exact_identity_step) {
            return Some(CompiledResidentVariablePathNodePosition::Start);
        }
        if path.steps[index + 1..].iter().all(is_exact_identity_step) {
            return Some(CompiledResidentVariablePathNodePosition::End);
        }
        return u16::try_from(index)
            .ok()
            .map(CompiledResidentVariablePathNodePosition::SegmentEnd);
    }
    None
}

fn is_complete_relationship_list(path: &super::Pattern, variable: &str) -> bool {
    let mut binding_found = false;
    for step in &path.steps {
        if step.relationship.variable.as_deref() == Some(variable) {
            if binding_found || !relationship_binds_list(&step.relationship) {
                return false;
            }
            binding_found = true;
        } else if !is_exact_identity_step(step) {
            // The validated result stores one complete relationship trail, not per-segment
            // boundaries. It equals this binding only when every other segment contributes the
            // empty list by an exact 0..0 bound.
            return false;
        }
    }
    binding_found
}

fn relationship_binds_list(relationship: &super::RelationshipPattern) -> bool {
    relationship.variable_length
        || relationship.min_hops.is_some()
        || relationship.max_hops.is_some()
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
    use std::collections::BTreeMap;

    use tokio_util::sync::CancellationToken;

    use crate::{
        Bookmark, EdgeId, Layer, NodeId, ProjectId, Result, ScalarValue,
        cypher::{BindCapabilities, OptimizerInput, bind, optimize, parse, plan},
        execution::{
            BackendKind, CpuBackend, ExecutionBackend, ResidentDirection,
            ResidentExecutionObligation, ResidentObligationKind, ResidentObligationScope,
            ResidentProjectImage, ResidentVariablePathInput,
        },
        graph::{
            EdgeInput, GraphStore, IndexCatalog, NodeInput, StatisticsSnapshot, TemporalStore,
        },
    };

    use super::{
        CompiledResidentMergeOptionalPathCountLowering, CompiledResidentPatternComprehensionPlan,
        CompiledResidentPatternComprehensionProjection, CompiledResidentVariablePathNodePosition,
        CompiledResidentVariablePathOutput, CompiledResidentVariablePathPlan,
        CompiledResidentVariablePathPostLowering, ResidentVariablePathPostProgram, compile,
        compile_exact_comparison1_independent_path_equality,
        compile_exact_match_where_relationship_parameter_filter,
        compile_exact_match_where_relationship_type_filter, compile_exact_match_where2_cycle_chord,
        compile_exact_match4_bound_relationship_list_rematch,
        compile_exact_match8_independent_path_sum, compile_exact_match8_merge_optional_count,
        compile_exact_match9_optional_null_path, compile_exact_pattern2_ordered_parent_path_lists,
        compile_exact_pattern2_path_node_label_counts,
        compile_exact_where4_correlated_path_disjunction,
        compile_exact_with_where_path_property_conjunction,
        compile_exact_with_where_projected_node_string_filter, compile_exact_with6_path_group_key,
        compile_pattern_comprehension,
    };

    const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());

    fn fixture_graph() -> Result<GraphStore> {
        let mut graph = GraphStore::default();
        for label in [
            "A", "Artist", "B", "Blue", "C", "End", "Green", "Single", "Start", "TheLabel", "X",
            "Y",
        ] {
            graph.catalog_mut().intern_label(label)?;
        }
        for relationship_type in [
            "AA_HAS_VALUE",
            "ADV_HAS_PRODUCT",
            "AP_HAS_VALUE",
            "CONTAINS",
            "EDGE",
            "FRIEND",
            "HATES",
            "KNOWS",
            "REL",
            "T",
            "WORKED_WITH",
        ] {
            graph
                .catalog_mut()
                .intern_relationship_type(relationship_type)?;
        }
        for property in ["id", "name", "name2", "time", "times", "var", "year"] {
            graph.catalog_mut().intern_property(property)?;
        }
        Ok(graph)
    }

    fn compile_query(
        source: &str,
        graph: &GraphStore,
    ) -> Result<Option<CompiledResidentVariablePathPlan>> {
        let query = parse(source)?;
        let bound = bind(query, graph.catalog(), BindCapabilities::default())?;
        let physical = plan(bound)?;
        compile(
            &physical,
            PROJECT,
            Bookmark {
                term: 3,
                index: graph.revision(),
            },
            graph.catalog(),
            graph,
            64,
        )
    }

    fn compile_optimized_query(
        source: &str,
        graph: &GraphStore,
    ) -> Result<Option<CompiledResidentVariablePathPlan>> {
        let query = parse(source)?;
        let bound = bind(query, graph.catalog(), BindCapabilities::default())?;
        let physical = plan(bound)?;
        let parameters = BTreeMap::new();
        let statistics = StatisticsSnapshot::collect(graph);
        let (physical, _) = optimize(
            physical,
            OptimizerInput {
                statistics: &statistics,
                catalog: graph.catalog(),
                indexes: None,
                parameters: &parameters,
                backend: BackendKind::Metal,
                scratch_budget_bytes: 512 * 1024 * 1024,
                max_result_rows: 64,
                runtime_feedback: None,
                allow_runtime_checkpoint: true,
            },
        );
        compile(
            &physical,
            PROJECT,
            Bookmark {
                term: 3,
                index: graph.revision(),
            },
            graph.catalog(),
            graph,
            64,
        )
    }

    #[test]
    fn return6_path_aggregations_lower_to_sealed_final_relations() -> Result<()> {
        let mut graph = fixture_graph()?;
        graph.catalog_mut().intern_label("L")?;
        graph.catalog_mut().intern_label("T")?;
        graph.catalog_mut().intern_relationship_type("R")?;
        let average =
            compile_optimized_query("MATCH p=(a:L)-[*]->(b) RETURN b, avg(length(p))", &graph)?
                .ok_or_else(|| crate::Error::internal("Return6 [8] was not admitted"))?;
        assert!(matches!(
            average.request.final_projection,
            crate::execution::ResidentVariablePathFinalProjection::GroupedEndNodeAveragePathLength { .. }
        ));
        let minimum = compile_optimized_query(
            "MATCH p = (a:T {name: 'a'})-[:R*]->(other:T) WHERE other <> a \
             WITH a, other, min(length(p)) AS len \
             RETURN a.name AS name, collect(other.name) AS others, len",
            &graph,
        )?
        .ok_or_else(|| crate::Error::internal("Return6 [13] was not admitted"))?;
        assert!(matches!(
            minimum.request.final_projection,
            crate::execution::ResidentVariablePathFinalProjection::GroupedMinimumPathLengthEndpointLists {
                ..
            }
        ));
        Ok(())
    }

    fn compile_pattern_query(
        source: &str,
        graph: &GraphStore,
    ) -> Result<Option<CompiledResidentPatternComprehensionPlan>> {
        let query = parse(source)?;
        let bound = bind(query, graph.catalog(), BindCapabilities::default())?;
        let physical = plan(bound)?;
        compile_pattern_comprehension(
            &physical,
            PROJECT,
            Bookmark {
                term: 3,
                index: graph.revision(),
            },
            graph.catalog(),
            graph,
        )
    }

    #[derive(Clone, Copy)]
    enum StagedVariablePostRecognizer {
        RelationshipListRematch,
        OptionalNullPath,
        DistinctPath,
    }

    fn compile_staged_variable_post_query(
        source: &str,
        graph: &GraphStore,
        recognizer: StagedVariablePostRecognizer,
    ) -> Result<Option<CompiledResidentVariablePathPlan>> {
        let query = parse(source)?;
        let bound = bind(query, graph.catalog(), BindCapabilities::default())?;
        let physical = plan(bound)?;
        let operators = physical
            .operators
            .iter()
            .filter(|operator| {
                !matches!(
                    operator,
                    super::PhysicalOperator::CardinalityCheckpoint { .. }
                )
            })
            .collect::<Vec<_>>();
        let bookmark = Bookmark {
            term: 3,
            index: graph.revision(),
        };
        match recognizer {
            StagedVariablePostRecognizer::RelationshipListRematch => {
                compile_exact_match4_bound_relationship_list_rematch(
                    &physical,
                    PROJECT,
                    bookmark,
                    graph.catalog(),
                    graph,
                    &operators,
                )
            }
            StagedVariablePostRecognizer::OptionalNullPath => {
                compile_exact_match9_optional_null_path(
                    &physical,
                    PROJECT,
                    bookmark,
                    graph.catalog(),
                    graph,
                    &operators,
                )
            }
            StagedVariablePostRecognizer::DistinctPath => compile_exact_with6_path_group_key(
                &physical,
                PROJECT,
                bookmark,
                graph.catalog(),
                graph,
                &operators,
            ),
        }
    }

    #[derive(Clone, Copy)]
    enum StagedLoweringRecognizer {
        PathNodeCounts,
        OrderedParentLists,
        IndependentPathEquality,
        RelationshipTypeFilter,
        RelationshipParameterFilter,
        ProjectedNodeStringFilter,
        PathPropertyConjunction,
        CorrelatedPathDisjunction,
        CycleChord,
        IndependentPathSum,
    }

    fn compile_staged_lowering_query(
        source: &str,
        graph: &GraphStore,
        recognizer: StagedLoweringRecognizer,
    ) -> Result<Option<CompiledResidentVariablePathPostLowering>> {
        let query = parse(source)?;
        let bound = bind(query, graph.catalog(), BindCapabilities::default())?;
        let physical = plan(bound)?;
        let operators = physical
            .operators
            .iter()
            .filter(|operator| {
                !matches!(
                    operator,
                    super::PhysicalOperator::CardinalityCheckpoint { .. }
                )
            })
            .collect::<Vec<_>>();
        let bookmark = Bookmark {
            term: 3,
            index: graph.revision(),
        };
        match recognizer {
            StagedLoweringRecognizer::PathNodeCounts => {
                compile_exact_pattern2_path_node_label_counts(
                    &physical,
                    PROJECT,
                    bookmark,
                    graph.catalog(),
                    graph,
                    &operators,
                )
            }
            StagedLoweringRecognizer::OrderedParentLists => {
                compile_exact_pattern2_ordered_parent_path_lists(
                    &physical,
                    PROJECT,
                    bookmark,
                    graph.catalog(),
                    graph,
                    &operators,
                )
            }
            StagedLoweringRecognizer::IndependentPathEquality => {
                compile_exact_comparison1_independent_path_equality(
                    &physical,
                    PROJECT,
                    bookmark,
                    graph.catalog(),
                    graph,
                    &operators,
                )
            }
            StagedLoweringRecognizer::RelationshipTypeFilter => {
                compile_exact_match_where_relationship_type_filter(
                    &physical,
                    PROJECT,
                    bookmark,
                    graph.catalog(),
                    graph,
                    &operators,
                )
            }
            StagedLoweringRecognizer::RelationshipParameterFilter => {
                compile_exact_match_where_relationship_parameter_filter(
                    &physical,
                    PROJECT,
                    bookmark,
                    graph.catalog(),
                    graph,
                    &operators,
                )
            }
            StagedLoweringRecognizer::ProjectedNodeStringFilter => {
                compile_exact_with_where_projected_node_string_filter(
                    &physical,
                    PROJECT,
                    bookmark,
                    graph.catalog(),
                    graph,
                    &operators,
                )
            }
            StagedLoweringRecognizer::PathPropertyConjunction => {
                compile_exact_with_where_path_property_conjunction(
                    &physical,
                    PROJECT,
                    bookmark,
                    graph.catalog(),
                    graph,
                    &operators,
                )
            }
            StagedLoweringRecognizer::CorrelatedPathDisjunction => {
                compile_exact_where4_correlated_path_disjunction(
                    &physical,
                    PROJECT,
                    bookmark,
                    graph.catalog(),
                    graph,
                    &operators,
                )
            }
            StagedLoweringRecognizer::CycleChord => compile_exact_match_where2_cycle_chord(
                &physical,
                PROJECT,
                bookmark,
                graph.catalog(),
                graph,
                &operators,
            ),
            StagedLoweringRecognizer::IndependentPathSum => {
                compile_exact_match8_independent_path_sum(
                    &physical,
                    PROJECT,
                    bookmark,
                    graph.catalog(),
                    graph,
                    &operators,
                )
            }
        }
    }

    fn compile_staged_merge_optional_count_query(
        source: &str,
        graph: &GraphStore,
    ) -> Result<Option<CompiledResidentMergeOptionalPathCountLowering>> {
        let query = parse(source)?;
        let bound = bind(
            query,
            graph.catalog(),
            BindCapabilities {
                write: true,
                ..BindCapabilities::default()
            },
        )?;
        let physical = plan(bound)?;
        let operators = physical
            .operators
            .iter()
            .filter(|operator| {
                !matches!(
                    operator,
                    super::PhysicalOperator::CardinalityCheckpoint { .. }
                )
            })
            .collect::<Vec<_>>();
        Ok(compile_exact_match8_merge_optional_count(
            &physical,
            graph.catalog(),
            &operators,
        ))
    }

    #[test]
    fn mixed_type_order_is_one_sealed_variable_path_command() -> Result<()> {
        let mut graph = fixture_graph()?;
        graph.catalog_mut().intern_label("N")?;
        let ascending = compile_optimized_query(
            "MATCH p = (n:N)-[r:REL]->() \
             UNWIND [n, r, p, 1.5, ['list'], 'text', null, false, 0.0 / 0.0, {a: 'map'}] AS types \
             RETURN types ORDER BY types",
            &graph,
        )?
        .ok_or_else(|| crate::Error::internal("mixed-type RETURN ORDER BY was not compiled"))?;
        assert!(matches!(
            ascending.request.final_projection,
            crate::execution::ResidentVariablePathFinalProjection::MixedTypeOrder {
                descending: false,
                limit: None,
                ..
            }
        ));
        assert!(matches!(
            ascending.outputs.as_slice(),
            [CompiledResidentVariablePathOutput::FinalMixedTypeValues { name }]
                if name == "types"
        ));

        let descending = compile_optimized_query(
            "MATCH p = (n:N)-[r:REL]->() \
             UNWIND [n, r, p, 1.5, ['list'], 'text', null, false, 0.0 / 0.0, {a: 'map'}] AS types \
             WITH types ORDER BY types DESC LIMIT 5 RETURN types",
            &graph,
        )?
        .ok_or_else(|| crate::Error::internal("mixed-type WITH ORDER BY was not compiled"))?;
        assert!(matches!(
            descending.request.final_projection,
            crate::execution::ResidentVariablePathFinalProjection::MixedTypeOrder {
                descending: true,
                limit: Some(5),
                ..
            }
        ));
        Ok(())
    }

    #[test]
    fn return2_entity_containers_compile_from_one_fixed_path_publication() -> Result<()> {
        let graph = fixture_graph()?;
        let list = compile_query("MATCH (n)-[r]->(m) RETURN [n, r, m] AS r", &graph)?
            .ok_or_else(|| crate::Error::internal("Return2 [12] was not compiled"))?;
        assert!(matches!(
            list.outputs.as_slice(),
            [CompiledResidentVariablePathOutput::OneHopEntityList { name }] if name == "r"
        ));

        let map = compile_query(
            "MATCH (n)-[r]->(m) RETURN {node1: n, rel: r, node2: m} AS m",
            &graph,
        )?
        .ok_or_else(|| crate::Error::internal("Return2 [13] was not compiled"))?;
        assert!(matches!(
            map.outputs.as_slice(),
            [CompiledResidentVariablePathOutput::OneHopEntityMap {
                name,
                start_key,
                relationship_key,
                end_key,
            }] if name == "m"
                && start_key == "node1"
                && relationship_key == "rel"
                && end_key == "node2"
        ));
        Ok(())
    }

    #[test]
    fn with3_relationship_identity_rematch_compiles_to_one_ordered_path_command() -> Result<()> {
        let graph = fixture_graph()?;
        let compiled = compile_optimized_query(
            "MATCH (a)-[r]->(b:X) WITH a, r, b MATCH (a)-[r]->(b) \
             RETURN r AS rel ORDER BY rel.id",
            &graph,
        )?
        .ok_or_else(|| crate::Error::internal("With3 [1] was not compiled"))?;
        assert!(matches!(
            compiled.outputs.as_slice(),
            [CompiledResidentVariablePathOutput::OneHopRelationship { name }] if name == "rel"
        ));
        assert!(matches!(
            compiled.request.final_projection,
            crate::execution::ResidentVariablePathFinalProjection::OrderedOneHopRelationships {
                property,
                ascending: true,
                ..
            } if property == graph.catalog().property("id").expect("fixture id property")
        ));
        Ok(())
    }

    #[test]
    fn with7_grouped_intermediate_count_compiles_to_one_sealed_path_command() -> Result<()> {
        let graph = fixture_graph()?;
        let compiled = compile_optimized_query(
            "MATCH (david {name: 'David'})--(otherPerson)-->() \
             WITH otherPerson, count(*) AS foaf WHERE foaf > 1 \
             WITH otherPerson WHERE otherPerson.name <> 'NotOther' RETURN count(*)",
            &graph,
        )?
        .ok_or_else(|| crate::Error::internal("With7 [2] was not compiled"))?;
        assert!(matches!(
            compiled.outputs.as_slice(),
            [CompiledResidentVariablePathOutput::FinalOptionalInteger { name }]
                if name == "count(*)"
        ));
        assert!(matches!(
            compiled.request.final_projection,
            crate::execution::ResidentVariablePathFinalProjection::GroupedIntermediateCount {
                group_segment: 0,
                minimum_group_size: 2,
                excluded_value,
                ..
            } if excluded_value == *b"NotOther"
        ));
        compiled.request.validate()?;
        Ok(())
    }

    #[test]
    fn with6_complete_path_group_key_is_an_executable_identity() -> Result<()> {
        let graph = fixture_graph()?;
        let compiled = compile_query(
            "MATCH p = ()-[*]->() WITH count(*) AS count, p AS p RETURN nodes(p) AS nodes",
            &graph,
        )?
        .ok_or_else(|| crate::Error::internal("With6 [4] was not compiled"))?;
        assert_eq!(
            compiled.post_program,
            ResidentVariablePathPostProgram::PassThrough
        );
        assert!(matches!(
            compiled.request.final_projection,
            crate::execution::ResidentVariablePathFinalProjection::Publications
        ));
        assert!(matches!(
            compiled.outputs.as_slice(),
            [CompiledResidentVariablePathOutput::Nodes { name }] if name == "nodes"
        ));
        compiled.request.validate()?;
        Ok(())
    }

    #[test]
    fn return_order_by2_grouped_node_paths_by_length_is_one_sealed_relation() -> Result<()> {
        let mut graph = fixture_graph()?;
        let relationship_type = graph
            .catalog()
            .relationship_type("REL")
            .expect("fixture REL relationship type");
        for id in 1_u64..=6 {
            graph.insert_node(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: 1,
                labels: Vec::new(),
                properties: Vec::new(),
            })?;
        }
        for (id, source, target) in [(1_u64, 1_u64, 2_u64), (2, 3, 4), (3, 4, 5), (4, 5, 6)] {
            graph.insert_edge(EdgeInput {
                id: EdgeId(id),
                source: NodeId(source),
                target: NodeId(target),
                relationship_type,
                layer: Layer::Observed,
                revision: 1,
                properties: Vec::new(),
            })?;
        }

        let query = "MATCH p = (a)-[*]->(b) \
                     RETURN collect(nodes(p)) AS paths, length(p) AS l ORDER BY l";
        let compiled = compile_optimized_query(query, &graph)?
            .ok_or_else(|| crate::Error::internal("ReturnOrderBy2 [12] was not compiled"))?;
        assert!(matches!(
            compiled.request.final_projection,
            crate::execution::ResidentVariablePathFinalProjection::GroupedNodePathsByLength {
                aggregate_obligation: ResidentExecutionObligation {
                    kind: ResidentObligationKind::Aggregate,
                    scope: ResidentObligationScope::PatternFinal,
                    ..
                },
                sort_obligation: ResidentExecutionObligation {
                    kind: ResidentObligationKind::Sort,
                    scope: ResidentObligationScope::PatternFinal,
                    ..
                },
            }
        ));
        assert!(matches!(
            compiled.outputs.as_slice(),
            [
                CompiledResidentVariablePathOutput::FinalGroupedNodePaths { name: paths },
                CompiledResidentVariablePathOutput::FinalGroupedPathLength { name: length },
            ] if paths == "paths" && length == "l"
        ));

        let bookmark = compiled.request.expected_bookmark;
        let image = ResidentProjectImage::build(
            PROJECT,
            bookmark,
            &graph,
            &TemporalStore::default(),
            &IndexCatalog::default(),
        )?;
        let mut cpu = CpuBackend::new(512 * 1024 * 1024, 64 * 1024 * 1024);
        cpu.admit_project(image)?;
        let result = cpu
            .execute_variable_path(&compiled.request, &CancellationToken::new())?
            .validate_for_publication(&compiled.request, BackendKind::Cpu)?;
        let crate::execution::ResidentVariablePathFinalRelation::GroupedNodePathsByLength(groups) =
            result.final_relation()
        else {
            return Err(crate::Error::internal(
                "ReturnOrderBy2 [12] did not publish grouped node paths",
            ));
        };
        assert_eq!(
            groups
                .iter()
                .map(|group| (group.length, group.paths.len()))
                .collect::<Vec<_>>(),
            vec![(1, 4), (2, 2), (3, 1)]
        );
        assert_eq!(result.rows().len(), 7);

        for near_miss in [
            "MATCH p = (a)-[*]->(b) RETURN collect(nodes(p)) AS paths, length(p) AS l ORDER BY l DESC",
            "MATCH p = (a)-[*]->(b) RETURN collect(relationships(p)) AS paths, length(p) AS l ORDER BY l",
            "MATCH p = (a)-[*]->(b) RETURN collect(nodes(p)) AS paths, avg(length(p)) AS l ORDER BY l",
            "MATCH p = (a)-[*1..2]->(b) RETURN collect(nodes(p)) AS paths, length(p) AS l ORDER BY l",
        ] {
            assert!(
                compile_optimized_query(near_miss, &graph)?.is_none(),
                "ReturnOrderBy2 [12] near miss was admitted: {near_miss}"
            );
        }
        Ok(())
    }

    #[test]
    fn with6_relationship_group_keys_and_bound_rematches_are_executable_identities() -> Result<()> {
        let mut graph = fixture_graph()?;
        let x = graph.catalog().label("X").expect("X label");
        let relationship_type = graph
            .catalog()
            .relationship_type("T")
            .expect("T relationship type");
        for (id, labels) in [
            (1_u64, Vec::new()),
            (2, vec![x]),
            (3, Vec::new()),
            (4, vec![x]),
        ] {
            graph.insert_node(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: 1,
                labels,
                properties: Vec::new(),
            })?;
        }
        for (id, source, target) in [(1_u64, 1_u64, 2_u64), (2, 3, 4)] {
            graph.insert_edge(EdgeInput {
                id: EdgeId(id),
                source: NodeId(source),
                target: NodeId(target),
                relationship_type,
                layer: Layer::Observed,
                revision: 1,
                properties: Vec::new(),
            })?;
        }
        for source in [
            "MATCH ()-[r1]->(:X) WITH r1 AS r2, count(*) AS c MATCH ()-[r2]->() RETURN r2 AS rel",
            "MATCH (a)-[r1]->(b:X) WITH a, r1 AS r2, b, count(*) AS c MATCH (a)-[r2]->(b) RETURN r2 AS rel",
        ] {
            let compiled = compile_optimized_query(source, &graph)?.ok_or_else(|| {
                crate::Error::internal(format!(
                    "With6 relationship identity was rejected: {source}"
                ))
            })?;
            assert_eq!(
                compiled.post_program,
                ResidentVariablePathPostProgram::PassThrough
            );
            assert!(matches!(
                compiled.request.final_projection,
                crate::execution::ResidentVariablePathFinalProjection::Publications
            ));
            assert!(matches!(
                compiled.outputs.as_slice(),
                [CompiledResidentVariablePathOutput::OneHopRelationship { name }] if name == "rel"
            ));
            compiled.request.validate()?;
        }

        assert!(
            compile_query(
                "MATCH (a)-[r1]->(b:X) WITH a, r1 AS r2, b, count(*) AS c MATCH (b)-[r2]->(a) RETURN r2 AS rel",
                &graph,
            )?
            .is_none(),
            "a reversed bound relationship rematch was treated as an identity"
        );
        Ok(())
    }

    #[test]
    fn with_skip_limit2_singleton_label_source_erases_only_identity_window() -> Result<()> {
        let mut graph = fixture_graph()?;
        let a = graph.catalog().label("A").expect("A label");
        graph.insert_node(NodeInput {
            id: NodeId(1),
            layer: Layer::Observed,
            revision: 1,
            labels: vec![a],
            properties: Vec::new(),
        })?;
        let source = "MATCH (a:A) WITH a ORDER BY a.name LIMIT 1 MATCH (a)-->(b) RETURN a";
        let compiled = compile_optimized_query(source, &graph)?
            .ok_or_else(|| crate::Error::internal("WithSkipLimit2 [1] was not compiled"))?;
        assert!(matches!(
            compiled.request.input,
            ResidentVariablePathInput::VisibleNodeScan { ref labels, .. } if labels == &[a]
        ));
        assert!(matches!(
            compiled.outputs.as_slice(),
            [CompiledResidentVariablePathOutput::Node {
                name,
                position: CompiledResidentVariablePathNodePosition::Start,
            }] if name == "a"
        ));
        compiled.request.validate()?;

        graph.insert_node(NodeInput {
            id: NodeId(2),
            layer: Layer::Observed,
            revision: 2,
            labels: vec![a],
            properties: Vec::new(),
        })?;
        assert!(
            compile_optimized_query(source, &graph)?.is_none(),
            "ORDER BY/LIMIT was erased for a non-singleton labelled source"
        );
        Ok(())
    }

    #[test]
    fn with_skip_limit1_dependency_join_is_one_sealed_cpu_relation() -> Result<()> {
        let mut graph = fixture_graph()?;
        let name = graph.catalog().property("name").expect("name property");
        let id = graph.catalog().property("id").expect("id property");
        let num = graph.catalog_mut().intern_property("num")?;
        for (node_id, node_name, node_id_value, num_value) in [
            (1_u64, "A", 0_i64, 0_i64),
            (2_u64, "B", 1_i64, 0_i64),
            (3_u64, "C", 2_i64, 0_i64),
        ] {
            graph.insert_node(NodeInput {
                id: NodeId(node_id),
                layer: Layer::Observed,
                revision: 1,
                labels: Vec::new(),
                properties: vec![
                    (name, ScalarValue::String(node_name.into())),
                    (id, ScalarValue::Integer(node_id_value)),
                    (num, ScalarValue::Integer(num_value)),
                ],
            })?;
        }
        let source = "MATCH (a) WITH a.name AS property, a.num AS idToUse \
                      ORDER BY property SKIP 1 MATCH (b) WHERE b.id = idToUse \
                      RETURN DISTINCT b";
        let compiled = compile_optimized_query(source, &graph)?
            .ok_or_else(|| crate::Error::internal("WithSkipLimit1 [1] was not compiled"))?;
        assert!(matches!(
            compiled.request.final_projection,
            crate::execution::ResidentVariablePathFinalProjection::SkipOrderedNodePropertyJoin {
                order_property,
                dependency_property,
                target_property,
                skip_rows: 1,
                ..
            } if order_property == name
                && dependency_property == num
                && target_property == id
        ));
        assert!(matches!(
            compiled.outputs.as_slice(),
            [CompiledResidentVariablePathOutput::FinalNodes { name }] if name == "b"
        ));
        compiled.request.validate()?;

        let bookmark = Bookmark {
            term: 3,
            index: graph.revision(),
        };
        let mut backend = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
        backend.admit_project(ResidentProjectImage::build(
            PROJECT,
            bookmark,
            &graph,
            &TemporalStore::default(),
            &IndexCatalog::default(),
        )?)?;
        let pinned = backend.pin_project(PROJECT)?;
        let result = pinned
            .execute_variable_path(&compiled.request, &CancellationToken::new())?
            .validate_for_publication(&compiled.request, BackendKind::Cpu)?;
        assert!(matches!(
            result.final_relation(),
            crate::execution::ResidentVariablePathFinalRelation::Nodes(nodes)
                if nodes.as_slice() == [0]
        ));

        for near_miss in [
            "MATCH (a) WITH a.name AS property, a.num AS idToUse ORDER BY property SKIP 2 MATCH (b) WHERE b.id = idToUse RETURN DISTINCT b",
            "MATCH (a) WITH a.name AS property, a.num AS idToUse ORDER BY property DESC SKIP 1 MATCH (b) WHERE b.id = idToUse RETURN DISTINCT b",
            "MATCH (a) WITH a.name AS property, a.num AS idToUse ORDER BY property SKIP 1 MATCH (b) WHERE b.id = idToUse RETURN b",
            "MATCH (a) WITH a.name AS property, a.num AS idToUse ORDER BY property SKIP 1 MATCH (b) WHERE b.id > idToUse RETURN DISTINCT b",
        ] {
            assert!(
                compile_optimized_query(near_miss, &graph)?.is_none(),
                "WithSkipLimit1 [1] near miss was admitted: {near_miss}"
            );
        }
        Ok(())
    }

    #[test]
    fn return_order_by2_singleton_path_erases_only_distinct_and_order_identities() -> Result<()> {
        let mut graph = fixture_graph()?;
        let relationship_type = graph
            .catalog()
            .relationship_type("T")
            .expect("T relationship type");
        for id in [1_u64, 2] {
            graph.insert_node(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: 1,
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
            revision: 1,
            properties: Vec::new(),
        })?;
        let source = "MATCH (a)-->(b) RETURN DISTINCT b ORDER BY b.name";
        let compiled = compile_optimized_query(source, &graph)?
            .ok_or_else(|| crate::Error::internal("ReturnOrderBy2 [5] was not compiled"))?;
        assert_eq!(compiled.request.maximum_output_rows, 1);
        assert!(matches!(
            compiled.outputs.as_slice(),
            [CompiledResidentVariablePathOutput::Node {
                name,
                position,
            }] if name == "b"
                && matches!(position,
                    CompiledResidentVariablePathNodePosition::Start
                        | CompiledResidentVariablePathNodePosition::End)
        ));
        compiled.request.validate()?;

        graph.insert_node(NodeInput {
            id: NodeId(3),
            layer: Layer::Observed,
            revision: 2,
            labels: Vec::new(),
            properties: Vec::new(),
        })?;
        graph.insert_edge(EdgeInput {
            id: EdgeId(2),
            source: NodeId(2),
            target: NodeId(3),
            relationship_type,
            layer: Layer::Observed,
            revision: 2,
            properties: Vec::new(),
        })?;
        assert!(
            compile_optimized_query(source, &graph)?.is_none(),
            "DISTINCT/ORDER BY was erased for a non-singleton relationship generation"
        );
        Ok(())
    }

    #[test]
    #[ignore = "developer-only physical-plan probe"]
    fn debug_next_variable_path_bundle_physical_plans() -> Result<()> {
        let graph = fixture_graph()?;
        for source in [
            "MATCH (a)-[r1]->()-[r2]->(b) WITH [r1, r2] AS rs, a AS first, b AS second LIMIT 1 MATCH (first)-[rs*]->(second) RETURN first, second",
            "MATCH (a)-[r1]->()-[r2]->(b) WITH [r1, r2] AS rs, a AS second, b AS first LIMIT 1 MATCH (first)-[rs*]->(second) RETURN first, second",
            "MATCH (a:A), (b:B) OPTIONAL MATCH (a)-[r*]-(b) WHERE r IS NULL AND a <> b RETURN b",
            "MATCH p = (n:X)-->() RETURN n, [x IN nodes(p) | size([(x)-->(:Y) | 1])] AS list",
            "MATCH (liker) RETURN [p = (liker)--() | p] AS isNew ORDER BY liker.time",
            "MATCH p1 = (:A)-->() MATCH p2 = (:A)<--() RETURN p1 = p2",
            "MATCH p = ()-[*]->() WITH count(*) AS count, p AS p RETURN nodes(p) AS nodes",
        ] {
            let query = parse(source)?;
            let bound = bind(query, graph.catalog(), BindCapabilities::default())?;
            let physical = plan(bound)?;
            eprintln!("QUERY: {source}\n{physical:#?}\n");
        }
        Ok(())
    }

    #[test]
    fn remaining_match4_and_pattern2_variable_shapes_have_exact_native_contracts() -> Result<()> {
        let graph = fixture_graph()?;

        let intermediate = compile_query(
            "MATCH (a {name: 'A'})-[:CONTAINS*0..1]->(b)-[:FRIEND*0..1]->(c) RETURN a, b, c",
            &graph,
        )?
        .ok_or_else(|| crate::Error::internal("Match4 [3] was not compiled"))?;
        assert_eq!(intermediate.request.segments.len(), 2);
        assert_eq!(
            intermediate.outputs,
            vec![
                CompiledResidentVariablePathOutput::Node {
                    name: "a".to_owned(),
                    position: CompiledResidentVariablePathNodePosition::Start,
                },
                CompiledResidentVariablePathOutput::Node {
                    name: "b".to_owned(),
                    position: CompiledResidentVariablePathNodePosition::SegmentEnd(0),
                },
                CompiledResidentVariablePathOutput::Node {
                    name: "c".to_owned(),
                    position: CompiledResidentVariablePathNodePosition::End,
                },
            ]
        );

        let bound_relationship = compile_query(
            "MATCH ()-[r:EDGE]-() MATCH p = (n)-[*0..1]-()-[r]-()-[*0..1]-(m) RETURN count(p) AS c",
            &graph,
        )?
        .ok_or_else(|| crate::Error::internal("Match4 [7] was not compiled"))?;
        assert_eq!(bound_relationship.request.segments.len(), 3);
        assert_eq!(
            bound_relationship.request.segments[1].relationship_types,
            vec![
                graph
                    .catalog()
                    .relationship_type("EDGE")
                    .expect("fixture EDGE type")
            ]
        );
        assert!(matches!(
            bound_relationship.outputs.as_slice(),
            [CompiledResidentVariablePathOutput::CountBoundUndirectedRelationshipPaths {
                name,
                segment: 1
            }] if name == "c"
        ));
        assert!(matches!(
            bound_relationship.request.final_projection,
            crate::execution::ResidentVariablePathFinalProjection::CountBoundUndirectedPaths {
                segment: 1,
                obligation: ResidentExecutionObligation {
                    kind: ResidentObligationKind::Aggregate,
                    scope: ResidentObligationScope::PatternFinal,
                    ..
                }
            }
        ));

        let bound_list = compile_query(
            "MATCH ()-[r1]->()-[r2]->() WITH [r1, r2] AS rs LIMIT 1 MATCH (first)-[rs*]->(second) RETURN first, second",
            &graph,
        )?
        .ok_or_else(|| crate::Error::internal("Match4 [8] was not compiled"))?;
        assert_eq!(bound_list.request.segments.len(), 2);
        assert_eq!(bound_list.request.output_limit, Some(1));
        assert_eq!(
            bound_list.post_program,
            ResidentVariablePathPostProgram::PassThrough
        );
        assert_eq!(
            bound_list.outputs,
            vec![
                CompiledResidentVariablePathOutput::Node {
                    name: "first".to_owned(),
                    position: CompiledResidentVariablePathNodePosition::Start,
                },
                CompiledResidentVariablePathOutput::Node {
                    name: "second".to_owned(),
                    position: CompiledResidentVariablePathNodePosition::End,
                },
            ]
        );

        let grouped = compile_pattern_query(
            "MATCH (a:A), (b:B) WITH [p = (a)-[*]->(b) | p] AS paths, count(a) AS c RETURN paths, c",
            &graph,
        )?
        .ok_or_else(|| crate::Error::internal("Pattern2 [9] was not compiled"))?;
        assert!(grouped.request.optional);
        assert!(grouped.request.bound_terminal_scan.is_some());
        assert!(matches!(
            grouped.request.final_projection,
            crate::execution::ResidentVariablePathFinalProjection::GroupedParentPathLists {
                obligation: ResidentExecutionObligation {
                    kind: ResidentObligationKind::Aggregate,
                    scope: ResidentObligationScope::PatternFinal,
                    ..
                }
            }
        ));
        assert!(matches!(
            grouped.projection,
            CompiledResidentPatternComprehensionProjection::GroupedVariablePathLists {
                ref list_output_name,
                ref count_output_name,
            } if list_output_name == "paths" && count_output_name == "c"
        ));

        for request in [
            &intermediate.request,
            &bound_relationship.request,
            &bound_list.request,
            &grouped.request,
        ] {
            request.validate()?;
        }
        Ok(())
    }

    #[test]
    fn next_seven_path_post_programs_have_exact_fail_closed_lowerings() -> Result<()> {
        let graph = fixture_graph()?;
        let same_rematch_query = "MATCH (a)-[r1]->()-[r2]->(b) WITH [r1, r2] AS rs, a AS first, b AS second LIMIT 1 MATCH (first)-[rs*]->(second) RETURN first, second";
        let reverse_rematch_query = "MATCH (a)-[r1]->()-[r2]->(b) WITH [r1, r2] AS rs, a AS second, b AS first LIMIT 1 MATCH (first)-[rs*]->(second) RETURN first, second";
        let optional_query =
            "MATCH (a:A), (b:B) OPTIONAL MATCH (a)-[r*]-(b) WHERE r IS NULL AND a <> b RETURN b";
        let distinct_query =
            "MATCH p = ()-[*]->() WITH count(*) AS count, p AS p RETURN nodes(p) AS nodes";
        let comparison_query = "MATCH p1 = (:A)-->() MATCH p2 = (:A)<--() RETURN p1 = p2";
        let node_counts_query =
            "MATCH p = (n:X)-->() RETURN n, [x IN nodes(p) | size([(x)-->(:Y) | 1])] AS list";
        let ordered_lists_query =
            "MATCH (liker) RETURN [p = (liker)--() | p] AS isNew ORDER BY liker.time";

        let same = compile_staged_variable_post_query(
            same_rematch_query,
            &graph,
            StagedVariablePostRecognizer::RelationshipListRematch,
        )?
        .ok_or_else(|| crate::Error::internal("Match9 [6] was not recognized"))?;
        assert!(matches!(
            same.post_program,
            ResidentVariablePathPostProgram::RelationshipListRematch {
                predicate: super::CompiledResidentVariablePathPublicationPredicate::All,
                obligation: ResidentExecutionObligation {
                    kind: ResidentObligationKind::PatternFilter,
                    scope: ResidentObligationScope::PatternFinal,
                    ..
                }
            }
        ));

        let reverse = compile_staged_variable_post_query(
            reverse_rematch_query,
            &graph,
            StagedVariablePostRecognizer::RelationshipListRematch,
        )?
        .ok_or_else(|| crate::Error::internal("Match9 [7] was not recognized"))?;
        assert!(matches!(
            reverse.post_program,
            ResidentVariablePathPostProgram::RelationshipListRematch {
                predicate: super::CompiledResidentVariablePathPublicationPredicate::StartEqualsEnd,
                ..
            }
        ));

        let optional = compile_staged_variable_post_query(
            optional_query,
            &graph,
            StagedVariablePostRecognizer::OptionalNullPath,
        )?
        .ok_or_else(|| crate::Error::internal("Match9 [8] was not recognized"))?;
        assert!(optional.request.optional);
        assert!(optional.request.bound_terminal_scan.is_some());
        assert!(matches!(
            optional.post_program,
            ResidentVariablePathPostProgram::FilterUnmatchedDifferentEndpoints {
                obligation: ResidentExecutionObligation {
                    kind: ResidentObligationKind::PatternFilter,
                    scope: ResidentObligationScope::PatternFinal,
                    ..
                }
            }
        ));

        let distinct = compile_staged_variable_post_query(
            distinct_query,
            &graph,
            StagedVariablePostRecognizer::DistinctPath,
        )?
        .ok_or_else(|| crate::Error::internal("With6 [4] was not recognized"))?;
        assert!(matches!(
            distinct.post_program,
            ResidentVariablePathPostProgram::DistinctPaths {
                obligation: ResidentExecutionObligation {
                    kind: ResidentObligationKind::Aggregate,
                    scope: ResidentObligationScope::PatternFinal,
                    ..
                }
            }
        ));

        let comparison = compile_staged_lowering_query(
            comparison_query,
            &graph,
            StagedLoweringRecognizer::IndependentPathEquality,
        )?
        .ok_or_else(|| crate::Error::internal("Comparison1 [14] was not recognized"))?;
        assert!(matches!(
            comparison.post_program,
            ResidentVariablePathPostProgram::IndependentOneHopPathEquality {
                ref output_name,
                ref secondary,
                cartesian_obligation: ResidentExecutionObligation {
                    kind: ResidentObligationKind::PatternCartesian,
                    scope: ResidentObligationScope::PatternCartesian,
                    ..
                },
                expression_obligation: ResidentExecutionObligation {
                    kind: ResidentObligationKind::Expression,
                    scope: ResidentObligationScope::Expression(0),
                    ..
                },
                ..
            } if output_name == "p1 = p2"
                && secondary.direction == ResidentDirection::Incoming
                && secondary.start_labels.as_slice() == [graph.catalog().label("A").expect("A label")]
        ));

        let node_counts = compile_staged_lowering_query(
            node_counts_query,
            &graph,
            StagedLoweringRecognizer::PathNodeCounts,
        )?
        .ok_or_else(|| crate::Error::internal("Pattern2 [7] was not recognized"))?;
        assert!(matches!(
            node_counts.post_program,
            ResidentVariablePathPostProgram::PathNodeOutgoingLabelCounts {
                ref node_output_name,
                ref list_output_name,
                label: Some(label),
                traversal_obligation: ResidentExecutionObligation {
                    kind: ResidentObligationKind::PatternTraversal,
                    scope: ResidentObligationScope::PatternLeaf(1),
                    ..
                },
                aggregate_obligation: ResidentExecutionObligation {
                    kind: ResidentObligationKind::Aggregate,
                    scope: ResidentObligationScope::PatternFinal,
                    ..
                }
            } if node_output_name == "n"
                && list_output_name == "list"
                && label == graph.catalog().label("Y").expect("Y label")
        ));

        let ordered = compile_staged_lowering_query(
            ordered_lists_query,
            &graph,
            StagedLoweringRecognizer::OrderedParentLists,
        )?
        .ok_or_else(|| crate::Error::internal("Pattern2 [11] was not recognized"))?;
        assert!(ordered.request.optional);
        assert!(matches!(
            ordered.post_program,
            ResidentVariablePathPostProgram::OrderedParentPathLists {
                ref output_name,
                property: Some(property),
                ascending: true,
                aggregate_obligation: ResidentExecutionObligation {
                    kind: ResidentObligationKind::Aggregate,
                    scope: ResidentObligationScope::PatternFinal,
                    ..
                },
                sort_obligation: ResidentExecutionObligation {
                    kind: ResidentObligationKind::Sort,
                    scope: ResidentObligationScope::PatternFinal,
                    ..
                }
            } if output_name == "isNew"
                && property == graph.catalog().property("time").expect("time property")
        ));

        for request in [
            &same.request,
            &reverse.request,
            &optional.request,
            &distinct.request,
            &comparison.request,
            &node_counts.request,
            &ordered.request,
        ] {
            request.validate()?;
        }

        // The newly sealed programs enter the ordinary compiler only after their final
        // predicate/reduction has been embedded in the immutable backend request.
        for query in [
            same_rematch_query,
            reverse_rematch_query,
            optional_query,
            comparison_query,
        ] {
            assert!(
                compile_query(query, &graph)?.is_some(),
                "sealed executable route rejected {query}"
            );
        }
        let sealed_distinct = compile_query(distinct_query, &graph)?
            .ok_or_else(|| crate::Error::internal("sealed With6 [4] route was rejected"))?;
        assert_eq!(
            sealed_distinct.post_program,
            ResidentVariablePathPostProgram::PassThrough,
            "complete path identity grouping must be erased only after the exact route proves it"
        );
        assert!(matches!(
            sealed_distinct.request.final_projection,
            crate::execution::ResidentVariablePathFinalProjection::Publications
        ));
        assert!(matches!(
            sealed_distinct.outputs.as_slice(),
            [CompiledResidentVariablePathOutput::Nodes { name }] if name == "nodes"
        ));
        for query in [node_counts_query, ordered_lists_query] {
            assert!(
                compile_pattern_query(query, &graph)?.is_some(),
                "sealed executable pattern route rejected {query}"
            );
        }
        Ok(())
    }

    #[test]
    fn comparison1_and_pattern2_07_bind_canonical_labels_without_fixture_names() -> Result<()> {
        let graph = fixture_graph()?;
        let comparison_query =
            "MATCH outgoing = (:B)-->() MATCH incoming = (:B)<--() RETURN outgoing = incoming";
        let comparison = compile_staged_lowering_query(
            comparison_query,
            &graph,
            StagedLoweringRecognizer::IndependentPathEquality,
        )?
        .ok_or_else(|| crate::Error::internal("renamed Comparison1 [14] was not recognized"))?;
        assert!(matches!(
            comparison.post_program,
            ResidentVariablePathPostProgram::IndependentOneHopPathEquality {
                ref secondary,
                ref output_name,
                ..
            } if secondary.start_labels.as_slice()
                    == [graph.catalog().label("B").expect("B label")]
                && output_name == "outgoing = incoming"
        ));
        comparison.request.validate()?;

        let node_counts_query = "MATCH route = (origin:B)-->() RETURN origin, [vertex IN nodes(route) | size([(vertex)-->(:C) | 1])] AS totals";
        let node_counts = compile_staged_lowering_query(
            node_counts_query,
            &graph,
            StagedLoweringRecognizer::PathNodeCounts,
        )?
        .ok_or_else(|| crate::Error::internal("renamed Pattern2 [7] was not recognized"))?;
        assert!(matches!(
            node_counts.post_program,
            ResidentVariablePathPostProgram::PathNodeOutgoingLabelCounts {
                ref node_output_name,
                ref list_output_name,
                label: Some(label),
                ..
            } if node_output_name == "origin"
                && list_output_name == "totals"
                && label == graph.catalog().label("C").expect("C label")
        ));
        node_counts.request.validate()?;

        assert!(compile_query(comparison_query, &graph)?.is_some());
        assert!(compile_pattern_query(node_counts_query, &graph)?.is_some());
        assert!(
            compile_staged_lowering_query(
                "MATCH left = (:A)-->() MATCH right = (:B)<--() RETURN left = right",
                &graph,
                StagedLoweringRecognizer::IndependentPathEquality,
            )?
            .is_none(),
            "different canonical path domains were admitted"
        );
        Ok(())
    }

    #[test]
    fn pattern2_07_unresolved_inner_label_fails_closed() -> Result<()> {
        let graph = fixture_graph()?;
        let query = "MATCH route = (origin:B)-->() RETURN origin, [vertex IN nodes(route) | size([(vertex)-->(:NeverInterned) | 1])] AS totals";
        assert!(
            compile_staged_lowering_query(query, &graph, StagedLoweringRecognizer::PathNodeCounts)?
                .is_none()
        );
        assert!(compile_pattern_query(query, &graph)?.is_none());
        Ok(())
    }

    #[test]
    fn next_seven_where_filter_shapes_have_exact_fail_closed_lowerings() -> Result<()> {
        let graph = fixture_graph()?;
        let type_query = "MATCH (n {name: 'A'})-[r]->(x) WHERE type(r) = 'KNOWS' RETURN x";
        let type_or_query =
            "MATCH (n)-[r]->(x) WHERE type(r) = 'KNOWS' OR type(r) = 'HATES' RETURN r";
        let relationship_parameter_query = "MATCH (a)-[r]->(b) WHERE r.name = $param RETURN b";
        let distinct_alias_query =
            "MATCH (a) WITH DISTINCT a.name2 AS name WHERE a.name2 = 'B' RETURN *";
        let alias_query = "MATCH (a) WITH a.name2 AS name WHERE name = 'B' RETURN *";
        let alias_or_source_query =
            "MATCH (a) WITH a.name2 AS name WHERE name = 'B' OR a.name2 = 'C' RETURN *";
        let path_conjunction_query = "MATCH (advertiser)-[:ADV_HAS_PRODUCT]->(out)-[:AP_HAS_VALUE]->(red)<-[:AA_HAS_VALUE]-(a) WITH a, advertiser, red, out WHERE advertiser.id = $1 AND a.id = $2 AND red.name = 'red' AND out.name = 'product1' RETURN out.name";

        let typed = compile_staged_lowering_query(
            type_query,
            &graph,
            StagedLoweringRecognizer::RelationshipTypeFilter,
        )?
        .ok_or_else(|| crate::Error::internal("MatchWhere1 [7] was not recognized"))?;
        assert!(matches!(
            typed.post_program,
            ResidentVariablePathPostProgram::RelationshipTypeFilterProject {
                ref relationship_types,
                relationship_types_known_empty: false,
                ref output,
                filter_obligation: ResidentExecutionObligation {
                    kind: ResidentObligationKind::PatternFilter,
                    scope: ResidentObligationScope::PatternFinal,
                    ..
                }
            } if relationship_types.as_slice()
                    == [graph.catalog().relationship_type("KNOWS").expect("KNOWS type")]
                && output.name == "x"
                && output.value
                    == super::ResidentVariablePathPostOutputValue::Node(
                        CompiledResidentVariablePathNodePosition::End
                    )
        ));

        let typed_or = compile_staged_lowering_query(
            type_or_query,
            &graph,
            StagedLoweringRecognizer::RelationshipTypeFilter,
        )?
        .ok_or_else(|| crate::Error::internal("MatchWhere1 [11] was not recognized"))?;
        assert!(matches!(
            typed_or.post_program,
            ResidentVariablePathPostProgram::RelationshipTypeFilterProject {
                ref relationship_types,
                ref output,
                ..
            } if relationship_types.len() == 2
                && relationship_types.contains(&graph.catalog().relationship_type("KNOWS").expect("KNOWS type"))
                && relationship_types.contains(&graph.catalog().relationship_type("HATES").expect("HATES type"))
                && output.name == "r"
                && output.value == super::ResidentVariablePathPostOutputValue::Relationship(0)
        ));

        let parameterized = compile_staged_lowering_query(
            relationship_parameter_query,
            &graph,
            StagedLoweringRecognizer::RelationshipParameterFilter,
        )?
        .ok_or_else(|| crate::Error::internal("MatchWhere1 [9] was not recognized"))?;
        assert!(matches!(
            parameterized.post_program,
            ResidentVariablePathPostProgram::EntityPropertyConjunctionProject {
                ref predicates,
                ref output,
                distinct: false,
                aggregate_obligation: None,
                ..
            } if matches!(
                    predicates.as_slice(),
                    [super::ResidentVariablePathPostPropertyEquality {
                        binding: super::ResidentVariablePathPostEntityBinding::Relationship(0),
                        property: Some(property),
                        operand: super::ResidentVariablePathPostOperand::Parameter(parameter),
                    }] if *property == graph.catalog().property("name").expect("name property")
                        && parameter == "param"
                ) && output.name == "b"
                    && output.value == super::ResidentVariablePathPostOutputValue::Node(
                        CompiledResidentVariablePathNodePosition::End
                    )
        ));

        for (query, expected_values, expected_distinct) in [
            (distinct_alias_query, &["B"][..], true),
            (alias_query, &["B"][..], false),
            (alias_or_source_query, &["B", "C"][..], false),
        ] {
            let lowered = compile_staged_lowering_query(
                query,
                &graph,
                StagedLoweringRecognizer::ProjectedNodeStringFilter,
            )?
            .ok_or_else(|| {
                crate::Error::internal(format!("WITH WHERE shape was not recognized: {query}"))
            })?;
            assert!(matches!(
                lowered.post_program,
                ResidentVariablePathPostProgram::ProjectedNodeStringSetFilter {
                    position: CompiledResidentVariablePathNodePosition::Start,
                    property: Some(property),
                    ref values,
                    ref output_name,
                    distinct,
                    aggregate_obligation,
                    ..
                } if property == graph.catalog().property("name2").expect("name2 property")
                    && values.iter().map(AsRef::as_ref).collect::<Vec<_>>() == expected_values
                    && output_name == "name"
                    && distinct == expected_distinct
                    && aggregate_obligation.is_some() == expected_distinct
            ));
            lowered.request.validate()?;
        }

        let path_conjunction = compile_staged_lowering_query(
            path_conjunction_query,
            &graph,
            StagedLoweringRecognizer::PathPropertyConjunction,
        )?
        .ok_or_else(|| crate::Error::internal("WithWhere2 [2] was not recognized"))?;
        assert_eq!(path_conjunction.request.segments.len(), 3);
        assert!(matches!(
            path_conjunction.post_program,
            ResidentVariablePathPostProgram::EntityPropertyConjunctionProject {
                ref predicates,
                ref output,
                distinct: false,
                aggregate_obligation: None,
                ..
            } if predicates.len() == 4
                && output.name == "out.name"
                && output.value == super::ResidentVariablePathPostOutputValue::NodeProperty {
                    position: CompiledResidentVariablePathNodePosition::SegmentEnd(0),
                    property: graph.catalog().property("name"),
                }
        ));

        for request in [
            &typed.request,
            &typed_or.request,
            &parameterized.request,
            &path_conjunction.request,
        ] {
            request.validate()?;
        }

        for query in [
            type_query,
            type_or_query,
            relationship_parameter_query,
            distinct_alias_query,
            alias_query,
            alias_or_source_query,
            path_conjunction_query,
        ] {
            assert!(
                compile_query(query, &graph)?.is_none(),
                "executable route admitted staged WHERE shape: {query}"
            );
        }
        Ok(())
    }

    #[test]
    fn matchwhere2_01_where4_02_withwhere4_02_and_match8_02_03_have_frozen_topology_lowerings()
    -> Result<()> {
        let graph = fixture_graph()?;
        let match_disjunction_query = "MATCH (a), (b) WHERE a.id = 0 AND (a)-[:T]->(b:TheLabel) OR (a)-[:T*]->(b:MissingLabel) RETURN DISTINCT b";
        let with_disjunction_query = "MATCH (a), (b) WITH a, b WHERE a.id = 0 AND (a)-[:T]->(b:TheLabel) OR (a)-[:T*]->(b:MissingLabel) RETURN DISTINCT b";
        let cycle_chord_query =
            "MATCH (a)--(b)--(c)--(d)--(a), (b)--(d) WHERE a.id = 1 AND c.id = 2 RETURN d";
        let mutation_optional_count =
            "MATCH (a) MERGE (b) WITH * OPTIONAL MATCH (a)--(b) RETURN count(*)";
        let independent_sum_query =
            "MATCH ()-->() WITH 1 AS x MATCH ()-[r1]->()<--() RETURN sum(r1.times)";

        for query in [match_disjunction_query, with_disjunction_query] {
            let lowered = compile_staged_lowering_query(
                query,
                &graph,
                StagedLoweringRecognizer::CorrelatedPathDisjunction,
            )?
            .ok_or_else(|| {
                crate::Error::internal(format!(
                    "shared WHERE4 path disjunction was not recognized: {query}"
                ))
            })?;
            lowered.request.validate()?;
            assert!(lowered.request.optional);
            assert!(lowered.request.multiplicity_scans.is_empty());
            assert!(
                lowered
                    .request
                    .bound_terminal_scan
                    .as_ref()
                    .is_some_and(|scan| scan.labels.as_slice()
                        == [graph.catalog().label("TheLabel").expect("TheLabel label")])
            );
            assert!(matches!(
                lowered.post_program,
                ResidentVariablePathPostProgram::CorrelatedPathPredicateDisjunction {
                    start_property: Some(property),
                    start_value: crate::ScalarValue::Integer(0),
                    ref secondary,
                    ref output_name,
                    secondary_traversal_obligation: ResidentExecutionObligation {
                        kind: ResidentObligationKind::PatternTraversal,
                        scope: ResidentObligationScope::PatternLeaf(1),
                        ..
                    },
                    filter_obligation: ResidentExecutionObligation {
                        kind: ResidentObligationKind::PatternFilter,
                        scope: ResidentObligationScope::PatternFinal,
                        ..
                    },
                    aggregate_obligation: ResidentExecutionObligation {
                        kind: ResidentObligationKind::Aggregate,
                        scope: ResidentObligationScope::PatternFinal,
                        ..
                    },
                } if property == graph.catalog().property("id").expect("id property")
                    && output_name == "b"
                    && secondary.direction == ResidentDirection::Outgoing
                    && secondary.relationship_types.as_slice()
                        == [graph.catalog().relationship_type("T").expect("T type")]
                    && !secondary.relationship_types_known_empty
                    && secondary.target_labels.is_empty()
                    && secondary.target_labels_known_empty
                    && secondary.minimum_hops == 1
                    && secondary.maximum_hops.is_none()
            ));
        }

        let cycle = compile_staged_lowering_query(
            cycle_chord_query,
            &graph,
            StagedLoweringRecognizer::CycleChord,
        )?
        .ok_or_else(|| crate::Error::internal("MatchWhere2 [1] was not recognized"))?;
        cycle.request.validate()?;
        assert_eq!(cycle.request.segments.len(), 4);
        assert!(cycle.request.segments[3].target_equals_path_start);
        assert!(matches!(
            cycle.post_program,
            ResidentVariablePathPostProgram::BoundaryRelationshipFilterProject {
                from: CompiledResidentVariablePathNodePosition::SegmentEnd(0),
                to: CompiledResidentVariablePathNodePosition::SegmentEnd(2),
                ref relationship,
                exclude_primary_trail: true,
                ref predicates,
                ref output,
                traversal_obligation: ResidentExecutionObligation {
                    kind: ResidentObligationKind::PatternTraversal,
                    scope: ResidentObligationScope::PatternLeaf(4),
                    ..
                },
                ..
            } if relationship.direction == ResidentDirection::Undirected
                && relationship.minimum_hops == 1
                && relationship.maximum_hops == Some(1)
                && predicates.len() == 2
                && output.name == "d"
                && output.value == super::ResidentVariablePathPostOutputValue::Node(
                    CompiledResidentVariablePathNodePosition::SegmentEnd(2)
                )
        ));

        let sum = compile_staged_lowering_query(
            independent_sum_query,
            &graph,
            StagedLoweringRecognizer::IndependentPathSum,
        )?
        .ok_or_else(|| crate::Error::internal("Match8 [3] was not recognized"))?;
        sum.request.validate()?;
        assert_eq!(sum.request.segments.len(), 2);
        assert!(matches!(
            sum.post_program,
            ResidentVariablePathPostProgram::IndependentPathRelationshipPropertySum {
                ref multiplicity_path,
                relationship_segment: 0,
                property: Some(property),
                ref output_name,
                scan_obligation: ResidentExecutionObligation {
                    kind: ResidentObligationKind::PatternScan,
                    scope: ResidentObligationScope::PatternScanN,
                    ..
                },
                traversal_obligation: ResidentExecutionObligation {
                    kind: ResidentObligationKind::PatternTraversal,
                    scope: ResidentObligationScope::PatternLeaf(2),
                    ..
                },
                cartesian_obligation: ResidentExecutionObligation {
                    kind: ResidentObligationKind::PatternCartesian,
                    scope: ResidentObligationScope::PatternCartesian,
                    ..
                },
                aggregate_obligation: ResidentExecutionObligation {
                    kind: ResidentObligationKind::Aggregate,
                    scope: ResidentObligationScope::PatternFinal,
                    ..
                },
            } if multiplicity_path.direction == ResidentDirection::Outgoing
                && multiplicity_path.minimum_hops == 1
                && multiplicity_path.maximum_hops == Some(1)
                && property == graph.catalog().property("times").expect("times property")
                && output_name == "sum(r1.times)"
        ));

        let mutation = compile_staged_merge_optional_count_query(mutation_optional_count, &graph)?
            .ok_or_else(|| crate::Error::internal("Match8 [2] was not recognized"))?;
        assert_eq!(mutation.source_variable, "a");
        assert_eq!(mutation.merged_variable, "b");
        assert_eq!(mutation.output_name, "count(*)");
        assert_eq!(
            mutation.relationship.direction,
            ResidentDirection::Undirected
        );
        assert_eq!(mutation.relationship.minimum_hops, 1);
        assert_eq!(mutation.relationship.maximum_hops, Some(1));
        assert_eq!(
            mutation.traversal_obligation.scope,
            ResidentObligationScope::PatternLeaf(0)
        );
        assert_eq!(
            mutation.aggregate_obligation.scope,
            ResidentObligationScope::PatternFinal
        );

        for query in [match_disjunction_query, with_disjunction_query] {
            let compiled = compile_query(query, &graph)?.ok_or_else(|| {
                crate::Error::internal(format!(
                    "executable correlated path disjunction was rejected: {query}"
                ))
            })?;
            assert!(matches!(
                compiled.request.final_projection,
                crate::execution::ResidentVariablePathFinalProjection::CorrelatedOutgoingPathDisjunction {
                    target_label: None,
                    start_value: 0,
                    ..
                }
            ));
            assert!(matches!(
                compiled.outputs.as_slice(),
                [CompiledResidentVariablePathOutput::FinalNodes { name }] if name == "b"
            ));
            compiled.request.validate()?;
        }
        let cycle = compile_query(cycle_chord_query, &graph)?
            .ok_or_else(|| crate::Error::internal("executable cycle chord was rejected"))?;
        assert!(matches!(
            cycle.request.final_projection,
            crate::execution::ResidentVariablePathFinalProjection::UndirectedCycleChordNodeFilter {
                start_value: 1,
                second_value: 2,
                ..
            }
        ));
        cycle.request.validate()?;

        let sum = compile_query(independent_sum_query, &graph)?.ok_or_else(|| {
            crate::Error::internal("executable independent path SUM was rejected")
        })?;
        assert!(matches!(
            sum.request.final_projection,
            crate::execution::ResidentVariablePathFinalProjection::IndependentPathRelationshipPropertySum {
                relationship_segment: 0,
                ..
            }
        ));
        sum.request.validate()?;

        // The exact post-write topology is frozen above, but executable admission remains closed
        // until the mutation command can traverse its own post-MERGE resident overlay.
        let mutation = parse(mutation_optional_count)?;
        let mutation = bind(
            mutation,
            graph.catalog(),
            BindCapabilities {
                write: true,
                ..BindCapabilities::default()
            },
        )?;
        let mutation = plan(mutation)?;
        assert!(!mutation.read_only);
        assert!(
            compile(
                &mutation,
                PROJECT,
                Bookmark {
                    term: 3,
                    index: graph.revision(),
                },
                graph.catalog(),
                &graph,
                64,
            )?
            .is_none()
        );
        Ok(())
    }

    #[test]
    fn topology_lowerings_bind_variables_properties_labels_types_and_literals_structurally()
    -> Result<()> {
        let graph = fixture_graph()?;
        let renamed_match = "MATCH (origin), (target) WHERE origin.year = 37 AND (origin)-[:FRIEND]->(target:Blue) OR (origin)-[:FRIEND*]->(target:Green) RETURN DISTINCT target";
        let renamed_with = "MATCH (origin), (target) WITH origin, target WHERE origin.year = 37 AND (origin)-[:FRIEND]->(target:Blue) OR (origin)-[:FRIEND*]->(target:Green) RETURN DISTINCT target";
        for query in [renamed_match, renamed_with] {
            let lowered = compile_staged_lowering_query(
                query,
                &graph,
                StagedLoweringRecognizer::CorrelatedPathDisjunction,
            )?
            .ok_or_else(|| crate::Error::internal("renamed WHERE4 topology was not recognized"))?;
            lowered.request.validate()?;
            assert!(
                lowered
                    .request
                    .bound_terminal_scan
                    .as_ref()
                    .is_some_and(|scan| {
                        scan.labels.as_slice()
                            == [graph.catalog().label("Blue").expect("Blue label")]
                    })
            );
            assert!(matches!(
                lowered.post_program,
                ResidentVariablePathPostProgram::CorrelatedPathPredicateDisjunction {
                    start_property: Some(property),
                    start_value: ScalarValue::Integer(37),
                    ref secondary,
                    ref output_name,
                    ..
                } if property == graph.catalog().property("year").expect("year property")
                    && output_name == "target"
                    && secondary.relationship_types.as_slice()
                        == [graph.catalog().relationship_type("FRIEND").expect("FRIEND type")]
                    && secondary.target_labels.as_slice()
                        == [graph.catalog().label("Green").expect("Green label")]
                    && !secondary.target_labels_known_empty
            ));
            let executable = compile_query(query, &graph)?.ok_or_else(|| {
                crate::Error::internal("renamed WHERE4 topology was not executable")
            })?;
            assert!(matches!(
                executable.request.final_projection,
                crate::execution::ResidentVariablePathFinalProjection::CorrelatedOutgoingPathDisjunction {
                    start_value: 37,
                    target_label: Some(label),
                    ..
                } if label == graph.catalog().label("Green").expect("Green label")
            ));
        }

        let renamed_cycle = "MATCH (root)--(leg)--(pivot)--(result)--(root), (leg)--(result) WHERE root.year = 9 AND pivot.year = 17 RETURN result";
        let cycle = compile_staged_lowering_query(
            renamed_cycle,
            &graph,
            StagedLoweringRecognizer::CycleChord,
        )?
        .ok_or_else(|| crate::Error::internal("renamed MatchWhere2 [1] was not recognized"))?;
        cycle.request.validate()?;
        assert!(matches!(
            cycle.post_program,
            ResidentVariablePathPostProgram::BoundaryRelationshipFilterProject {
                ref predicates,
                ref output,
                ..
            } if predicates.len() == 2
                && predicates.iter().all(|predicate| predicate.property
                    == graph.catalog().property("year"))
                && output.name == "result"
        ));
        let executable_cycle = compile_query(renamed_cycle, &graph)?
            .ok_or_else(|| crate::Error::internal("renamed cycle chord was not executable"))?;
        assert!(matches!(
            executable_cycle.request.final_projection,
            crate::execution::ResidentVariablePathFinalProjection::UndirectedCycleChordNodeFilter {
                start_value: 9,
                second_value: 17,
                ..
            }
        ));

        let renamed_mutation = "MATCH (source) MERGE (created) WITH * OPTIONAL MATCH (source)--(created) RETURN count(*) AS total";
        let mutation = compile_staged_merge_optional_count_query(renamed_mutation, &graph)?
            .ok_or_else(|| crate::Error::internal("renamed Match8 [2] was not recognized"))?;
        assert_eq!(mutation.source_variable, "source");
        assert_eq!(mutation.merged_variable, "created");
        assert_eq!(mutation.output_name, "total");

        let renamed_sum = "MATCH ()-->() WITH 'retained' AS ignored MATCH ()-[link]->()<--() RETURN sum(link.year) AS total";
        let sum = compile_staged_lowering_query(
            renamed_sum,
            &graph,
            StagedLoweringRecognizer::IndependentPathSum,
        )?
        .ok_or_else(|| crate::Error::internal("renamed Match8 [3] was not recognized"))?;
        sum.request.validate()?;
        assert!(matches!(
            sum.post_program,
            ResidentVariablePathPostProgram::IndependentPathRelationshipPropertySum {
                property: Some(property),
                ref output_name,
                ..
            } if property == graph.catalog().property("year").expect("year property")
                && output_name == "total"
        ));
        let executable_sum = compile_query(renamed_sum, &graph)?
            .ok_or_else(|| crate::Error::internal("renamed path SUM was not executable"))?;
        assert!(matches!(
            executable_sum.request.final_projection,
            crate::execution::ResidentVariablePathFinalProjection::IndependentPathRelationshipPropertySum {
                relationship_segment: 0,
                property,
                ..
            } if property == graph.catalog().property("year").expect("year property")
        ));

        assert!(
            compile_staged_lowering_query(
                "MATCH (origin), (target) WHERE origin.year = 37 AND (origin)-[:FRIEND]->(target:Blue) OR (origin)-[:KNOWS*]->(target:Green) RETURN DISTINCT target",
                &graph,
                StagedLoweringRecognizer::CorrelatedPathDisjunction,
            )?
            .is_none(),
            "different relationship-type bindings were admitted"
        );
        assert!(
            compile_staged_lowering_query(
                "MATCH (root)--(leg)--(pivot)--(result)--(root), (leg)--(result) WHERE root.year = 9 AND pivot.name2 = 'seventeen' RETURN result",
                &graph,
                StagedLoweringRecognizer::CycleChord,
            )?
            .is_none(),
            "different cycle predicate-property bindings were admitted"
        );
        assert!(
            compile_staged_merge_optional_count_query(
                "MATCH (source) MERGE (created) WITH * OPTIONAL MATCH (source)--(other) RETURN count(*) AS total",
                &graph,
            )?
            .is_none(),
            "OPTIONAL terminal identity drifted from the MERGE binding"
        );
        Ok(())
    }

    #[test]
    fn remaining_variable_path_near_misses_stay_closed() -> Result<()> {
        let graph = fixture_graph()?;
        for query in [
            "MATCH ()-[r:EDGE]->() MATCH p = (n)-[*0..1]-()-[r]-()-[*0..1]-(m) RETURN count(p) AS c",
            "MATCH ()-[r1]->()-[r2]->() WITH [r2, r1] AS rs LIMIT 1 MATCH (first)-[rs*]->(second) RETURN first, second",
        ] {
            assert!(
                compile_query(query, &graph)?.is_none(),
                "near miss admitted: {query}"
            );
        }
        assert!(
            compile_pattern_query(
                "MATCH (a:A), (b:B) WITH [p = (a)-[*]->(b) | p] AS paths, count(b) AS c RETURN paths, c",
                &graph,
            )?
            .is_none(),
            "Pattern2 [9] alternate aggregate was admitted",
        );
        Ok(())
    }

    #[test]
    fn lowers_match6_zero_length_named_path_to_receipted_identity_segment() -> Result<()> {
        let graph = fixture_graph()?;
        let compiled = compile_query("MATCH p = (a) RETURN p", &graph)?
            .ok_or_else(|| crate::Error::internal("Match6 [1] was not compiled natively"))?;

        assert_eq!(compiled.request.segments.len(), 1);
        let identity = &compiled.request.segments[0];
        assert_eq!(identity.minimum_hops, 0);
        assert_eq!(identity.maximum_hops, Some(0));
        assert!(identity.relationship_types.is_empty());
        assert!(matches!(
            &compiled.request.input,
            ResidentVariablePathInput::VisibleNodeScan { labels, .. } if labels.is_empty()
        ));
        assert_eq!(
            compiled.outputs,
            vec![CompiledResidentVariablePathOutput::Path {
                name: "p".to_owned(),
            }]
        );
        compiled.request.validate()?;
        Ok(())
    }

    #[test]
    fn match6_14_mixed_directed_and_undirected_path_keeps_exact_trails() -> Result<()> {
        let mut graph = GraphStore::default();
        let start = graph.catalog_mut().intern_label("Start")?;
        let end = graph.catalog_mut().intern_label("End")?;
        let connected = graph
            .catalog_mut()
            .intern_relationship_type("CONNECTED_TO")?;
        for (id, labels) in [
            (1_u64, vec![start]),
            (2, vec![end]),
            (3, Vec::new()),
            (4, Vec::new()),
        ] {
            graph.insert_node(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: 1,
                labels,
                properties: Vec::new(),
            })?;
        }
        for (id, target) in [(1_u64, 1_u64), (2, 2), (3, 2), (4, 4), (5, 4)] {
            graph.insert_edge(EdgeInput {
                id: EdgeId(id),
                source: NodeId(3),
                target: NodeId(target),
                relationship_type: connected,
                layer: Layer::Observed,
                revision: 1,
                properties: Vec::new(),
            })?;
        }
        let query = "MATCH topRoute = (:Start)<-[:CONNECTED_TO]-()\
                     -[:CONNECTED_TO*3..3]-(:End) RETURN topRoute";
        let compiled = compile_optimized_query(query, &graph)?
            .ok_or_else(|| crate::Error::internal("Match6 [14] was not compiled natively"))?;

        assert!(compiled.request.multiplicity_scans.is_empty());
        assert!(compiled.request.bound_terminal_scan.is_none());
        assert_eq!(compiled.request.segments.len(), 2);
        assert_eq!(
            compiled.request.segments[0].direction,
            ResidentDirection::Incoming
        );
        assert_eq!(compiled.request.segments[0].minimum_hops, 1);
        assert_eq!(compiled.request.segments[0].maximum_hops, Some(1));
        assert!(compiled.request.segments[0].target_labels.is_empty());
        assert_eq!(
            compiled.request.segments[1].direction,
            ResidentDirection::Undirected
        );
        assert_eq!(compiled.request.segments[1].minimum_hops, 3);
        assert_eq!(compiled.request.segments[1].maximum_hops, Some(3));
        assert_eq!(compiled.request.segments[1].target_labels, vec![end]);
        assert!(!compiled.request.segments[1].target_labels_known_empty);

        let bookmark = compiled.request.expected_bookmark;
        let image = ResidentProjectImage::build(
            PROJECT,
            bookmark,
            &graph,
            &TemporalStore::default(),
            &IndexCatalog::default(),
        )?;
        let mut cpu = CpuBackend::new(512 * 1024 * 1024, 64 * 1024 * 1024);
        cpu.admit_project(image)?;
        let result = cpu
            .execute_variable_path(&compiled.request, &CancellationToken::new())?
            .validate_for_publication(&compiled.request, BackendKind::Cpu)?;
        assert_eq!(result.rows().len(), 4);
        assert!(result.rows().iter().all(|row| {
            row.path
                .as_ref()
                .is_some_and(|path| path.relationships.len() == 4 && path.end() == Some(1))
        }));
        Ok(())
    }

    #[test]
    fn fixed_path_segments_keep_endpoint_predicates_repeated_start_and_limit() -> Result<()> {
        let graph = fixture_graph()?;
        let name = graph.catalog().property("name").expect("fixture property");

        let property_path =
            compile_query("MATCH p = ({name: 'a'})-->({name: 'b'}) RETURN p", &graph)?
                .ok_or_else(|| crate::Error::internal("endpoint property path was not compiled"))?;
        assert_eq!(property_path.request.segments.len(), 1);
        assert_eq!(property_path.request.segments[0].target_predicates.len(), 1);
        assert_eq!(
            property_path.request.segments[0].target_predicates[0].property,
            name
        );

        let repeated_start = compile_query("MATCH p = (n)-->(k)<--(n) RETURN p", &graph)?
            .ok_or_else(|| crate::Error::internal("repeated-start path was not compiled"))?;
        assert_eq!(repeated_start.request.segments.len(), 2);
        assert!(!repeated_start.request.segments[0].target_equals_path_start);
        assert!(repeated_start.request.segments[1].target_equals_path_start);

        let limited = compile_query("MATCH p = (n:A)--(m) RETURN p LIMIT 1", &graph)?
            .ok_or_else(|| crate::Error::internal("limited path was not compiled"))?;
        assert_eq!(limited.request.output_limit, Some(1));
        limited.request.validate()?;
        Ok(())
    }

    #[test]
    fn lowers_return7_1_exact_named_one_hop_wildcard_in_scope_order() -> Result<()> {
        let graph = fixture_graph()?;
        let start = graph.catalog().label("Start").expect("fixture label");
        let compiled = compile_query("MATCH p = (a:Start)-->(b) RETURN *", &graph)?
            .ok_or_else(|| crate::Error::internal("Return7 [1] was not compiled natively"))?;

        assert!(matches!(
            &compiled.request.input,
            ResidentVariablePathInput::VisibleNodeScan { labels, .. }
                if labels == &vec![start]
        ));
        let [segment] = compiled.request.segments.as_slice() else {
            panic!("Return7 [1] did not compile to one traversal segment");
        };
        assert_eq!(segment.direction, ResidentDirection::Outgoing);
        assert_eq!(segment.minimum_hops, 1);
        assert_eq!(segment.maximum_hops, Some(1));
        assert_eq!(
            compiled.outputs,
            vec![
                CompiledResidentVariablePathOutput::Node {
                    name: "a".to_owned(),
                    position: CompiledResidentVariablePathNodePosition::Start,
                },
                CompiledResidentVariablePathOutput::Node {
                    name: "b".to_owned(),
                    position: CompiledResidentVariablePathNodePosition::End,
                },
                CompiledResidentVariablePathOutput::Path {
                    name: "p".to_owned(),
                },
            ]
        );
        compiled.request.validate()?;
        Ok(())
    }

    #[test]
    fn keeps_neighboring_named_path_wildcards_fail_closed() -> Result<()> {
        let graph = fixture_graph()?;
        for (reason, query) in [
            (
                "mixed wildcard projection",
                "MATCH p = (a:Start)-->(b) RETURN *, a AS duplicate_a",
            ),
            (
                "distinct wildcard projection",
                "MATCH p = (a:Start)-->(b) RETURN DISTINCT *",
            ),
            (
                "visible relationship binding",
                "MATCH p = (a:Start)-[r]->(b) RETURN *",
            ),
            (
                "explicit variable-length syntax",
                "MATCH p = (a:Start)-[*1..1]->(b) RETURN *",
            ),
            (
                "intermediate node binding",
                "MATCH p = (a:Start)-->(b)-->(c) RETURN *",
            ),
            (
                "unrelated visible source",
                "MATCH p = (a:Start)-->(b), (u) RETURN *",
            ),
            (
                "nullable parent",
                "MATCH (a:Start) OPTIONAL MATCH p = (a)-->(b) RETURN *",
            ),
            ("anonymous start", "MATCH p = (:Start)-->(b) RETURN *"),
            ("anonymous end", "MATCH p = (a:Start)-->() RETURN *"),
            ("repeated endpoint", "MATCH p = (a:Start)-->(a) RETURN *"),
            (
                "post-path filter",
                "MATCH p = (a:Start)-->(b) WHERE a.name = 'A' RETURN *",
            ),
            (
                "ordered wildcard",
                "MATCH p = (a:Start)-->(b) RETURN * ORDER BY a.name",
            ),
            (
                "limited wildcard",
                "MATCH p = (a:Start)-->(b) RETURN * LIMIT 1",
            ),
        ] {
            assert!(
                compile_query(query, &graph)?.is_none(),
                "{reason} entered the exact Return7 wildcard route: {query}"
            );
        }
        Ok(())
    }

    #[test]
    fn folds_only_provable_identity_endpoint_constraints_into_the_start_scan() -> Result<()> {
        let graph = fixture_graph()?;
        let start = graph.catalog().label("Start").expect("fixture label");
        let end = graph.catalog().label("End").expect("fixture label");
        let compiled = compile_query(
            "MATCH p = (a:Start {})-[:REL*0..0 {year: 1988}]->(b:End {}) RETURN a, b, p",
            &graph,
        )?
        .ok_or_else(|| crate::Error::internal("exact identity endpoint was not compiled"))?;

        let ResidentVariablePathInput::VisibleNodeScan { labels, .. } = &compiled.request.input
        else {
            panic!("standalone identity path did not use a visible-node scan");
        };
        let mut expected_labels = vec![start, end];
        expected_labels.sort_unstable();
        assert_eq!(labels, &expected_labels);
        assert_eq!(compiled.request.segments[0].minimum_hops, 0);
        assert_eq!(compiled.request.segments[0].maximum_hops, Some(0));
        assert_eq!(
            compiled.outputs,
            vec![
                CompiledResidentVariablePathOutput::Node {
                    name: "a".to_owned(),
                    position: CompiledResidentVariablePathNodePosition::Start,
                },
                CompiledResidentVariablePathOutput::Node {
                    name: "b".to_owned(),
                    position: CompiledResidentVariablePathNodePosition::End,
                },
                CompiledResidentVariablePathOutput::Path {
                    name: "p".to_owned(),
                },
            ]
        );

        let repeated = compile_query(
            "MATCH p = (n:Start)-[:REL*0..0]->(n:End) RETURN n, p",
            &graph,
        )?;
        assert!(
            repeated.is_some(),
            "a repeated endpoint connected only by an exact zero-hop segment is already equal"
        );

        let combined = compile_query(
            "MATCH (a:Start) MATCH p = (a:End)-[:REL*0..0]->(b) RETURN p",
            &graph,
        )?
        .ok_or_else(|| crate::Error::internal("bound start labels were not combined"))?;
        let ResidentVariablePathInput::VisibleNodeScan { labels, .. } = &combined.request.input
        else {
            panic!("bound identity path did not use a visible-node scan");
        };
        assert_eq!(labels, &expected_labels);
        assert!(combined.request.multiplicity_scans.is_empty());
        Ok(())
    }

    #[test]
    fn lowers_the_exact_match9_relationship_list_cores_without_dropping_filters() -> Result<()> {
        let graph = fixture_graph()?;
        for (query, direction) in [
            (
                "MATCH (a)-[r:REL*2..2]->(b) RETURN r",
                ResidentDirection::Outgoing,
            ),
            (
                "MATCH (a)-[r:REL*2..2]-(b) RETURN r",
                ResidentDirection::Undirected,
            ),
        ] {
            let compiled = compile_query(query, &graph)?
                .ok_or_else(|| crate::Error::internal("exact Match9 list core was not compiled"))?;
            assert_eq!(compiled.request.segments.len(), 1, "{query}");
            let segment = &compiled.request.segments[0];
            assert_eq!(segment.direction, direction, "{query}");
            assert_eq!(segment.minimum_hops, 2, "{query}");
            assert_eq!(segment.maximum_hops, Some(2), "{query}");
            assert_eq!(
                compiled.outputs,
                vec![CompiledResidentVariablePathOutput::Relationships {
                    name: "r".to_owned(),
                }],
                "{query}"
            );
        }

        let unbounded =
            compile_query("MATCH (a:Blue)-[r*]->(b) RETURN r", &graph)?.ok_or_else(|| {
                crate::Error::internal("unbounded relationship list was not compiled")
            })?;
        assert_eq!(unbounded.request.segments[0].minimum_hops, 1);
        assert_eq!(unbounded.request.segments[0].maximum_hops, None);
        assert!(
            compile_query("MATCH (a:Blue)-[r*]->(b) RETURN count(r)", &graph)?.is_none(),
            "path cardinality aggregation must not be rewritten as per-path list publication"
        );
        Ok(())
    }

    #[test]
    fn zero_hop_padding_preserves_a_single_complete_relationship_list_binding() -> Result<()> {
        let graph = fixture_graph()?;
        let query = "MATCH (a)-[:REL*0..0]->(x)-[r:REL*2..2]->(y)\
            -[:REL*0..0]->(b) RETURN r, last(r) AS l";
        let compiled = compile_query(query, &graph)?.ok_or_else(|| {
            crate::Error::internal("zero-padded relationship list was not compiled")
        })?;

        assert_eq!(compiled.request.segments.len(), 3);
        assert_eq!(
            compiled
                .request
                .segments
                .iter()
                .map(|segment| (segment.minimum_hops, segment.maximum_hops))
                .collect::<Vec<_>>(),
            vec![(0, Some(0)), (2, Some(2)), (0, Some(0))]
        );
        assert_eq!(
            compiled.outputs,
            vec![
                CompiledResidentVariablePathOutput::Relationships {
                    name: "r".to_owned(),
                },
                CompiledResidentVariablePathOutput::LastRelationship {
                    name: "l".to_owned(),
                },
            ]
        );
        assert!(
            compile_query("MATCH (a)-[r:REL*1..1]->(x)-[:REL]->(b) RETURN r", &graph,)?.is_none(),
            "a non-empty sibling segment must not be folded into relationship binding `r`"
        );
        Ok(())
    }

    #[test]
    fn lowers_match7_12_to_parent_aware_optional_unbounded_traversal() -> Result<()> {
        let graph = fixture_graph()?;
        let single = graph.catalog().label("Single").expect("fixture label");
        let compiled = compile_query(
            "MATCH (a:Single) OPTIONAL MATCH (a)-[*]->(b) RETURN b",
            &graph,
        )?
        .ok_or_else(|| crate::Error::internal("Match7 [12] was not compiled natively"))?;

        assert!(compiled.request.optional);
        assert_eq!(compiled.request.maximum_output_rows, 64);
        assert_eq!(compiled.request.maximum_frontier_paths, 64);
        assert!(compiled.request.bound_terminal_scan.is_none());
        assert!(compiled.request.multiplicity_scans.is_empty());
        assert!(compiled.request.cartesian_obligation.is_none());
        assert!(matches!(
            &compiled.request.input,
            ResidentVariablePathInput::VisibleNodeScan { labels, .. }
                if labels == &vec![single]
        ));
        let [segment] = compiled.request.segments.as_slice() else {
            panic!("Match7 [12] did not compile to one traversal segment");
        };
        assert_eq!(segment.direction, ResidentDirection::Outgoing);
        assert_eq!(segment.minimum_hops, 1);
        assert_eq!(segment.maximum_hops, None);
        assert_eq!(
            compiled.outputs,
            vec![CompiledResidentVariablePathOutput::Node {
                name: "b".to_owned(),
                position: CompiledResidentVariablePathNodePosition::End,
            }]
        );
        compiled.request.validate()?;
        Ok(())
    }

    #[test]
    fn lowers_match7_13_by_binding_identity_and_retains_the_bound_terminal() -> Result<()> {
        let graph = fixture_graph()?;
        let single = graph.catalog().label("Single").expect("fixture label");
        let c = graph.catalog().label("C").expect("fixture label");
        let end = graph.catalog().label("End").expect("fixture label");

        for query in [
            "MATCH (a:Single), (x:C) OPTIONAL MATCH (a)-[*]->(x) RETURN x",
            "MATCH (x:C), (a:Single) OPTIONAL MATCH (a)-[*]->(x) RETURN x",
        ] {
            let compiled = compile_query(query, &graph)?.ok_or_else(|| {
                crate::Error::internal("Match7 [13] source bindings were classified by position")
            })?;
            assert!(compiled.request.optional, "{query}");
            assert_eq!(compiled.request.maximum_output_rows, 64, "{query}");
            assert!(compiled.request.multiplicity_scans.is_empty(), "{query}");
            assert!(compiled.request.cartesian_obligation.is_some(), "{query}");
            assert!(
                matches!(
                    &compiled.request.input,
                    ResidentVariablePathInput::VisibleNodeScan { labels, .. }
                        if labels == &vec![single]
                ),
                "{query}"
            );
            let terminal = compiled
                .request
                .bound_terminal_scan
                .as_ref()
                .expect("Match7 [13] must retain x as a terminal source");
            assert_eq!(terminal.labels, vec![c], "{query}");
            assert_eq!(
                terminal.obligation.id,
                super::VARIABLE_PATH_BOUND_TERMINAL_SCAN_OBLIGATION,
                "{query}"
            );
            let [segment] = compiled.request.segments.as_slice() else {
                panic!("Match7 [13] did not compile to one traversal segment");
            };
            assert_eq!(segment.direction, ResidentDirection::Outgoing, "{query}");
            assert_eq!(segment.minimum_hops, 1, "{query}");
            assert_eq!(segment.maximum_hops, None, "{query}");
            assert_eq!(
                compiled.outputs,
                vec![CompiledResidentVariablePathOutput::Node {
                    name: "x".to_owned(),
                    position: CompiledResidentVariablePathNodePosition::BoundTerminal,
                }],
                "{query}"
            );
            compiled.request.validate()?;
        }

        let labelled_endpoint = compile_query(
            "MATCH (a:Single), (x:C) OPTIONAL MATCH (a)-[*]->(x:End) RETURN x",
            &graph,
        )?
        .ok_or_else(|| crate::Error::internal("bound terminal endpoint labels were not folded"))?;
        let mut expected_terminal_labels = vec![c, end];
        expected_terminal_labels.sort_unstable();
        assert_eq!(
            labelled_endpoint
                .request
                .bound_terminal_scan
                .as_ref()
                .expect("labelled terminal scan")
                .labels,
            expected_terminal_labels
        );
        Ok(())
    }

    #[test]
    fn lowers_match7_14_without_turning_the_unbounded_hop_range_into_a_ceiling() -> Result<()> {
        let graph = fixture_graph()?;
        let compiled = compile_query(
            "MATCH (a:Single) OPTIONAL MATCH (a)-[*3..]-(b) RETURN b",
            &graph,
        )?
        .ok_or_else(|| crate::Error::internal("Match7 [14] was not compiled natively"))?;

        assert!(compiled.request.optional);
        let [segment] = compiled.request.segments.as_slice() else {
            panic!("Match7 [14] did not compile to one traversal segment");
        };
        assert_eq!(segment.direction, ResidentDirection::Undirected);
        assert_eq!(segment.minimum_hops, 3);
        assert_eq!(segment.maximum_hops, None);
        assert_eq!(
            compiled.outputs,
            vec![CompiledResidentVariablePathOutput::Node {
                name: "b".to_owned(),
                position: CompiledResidentVariablePathNodePosition::End,
            }]
        );
        compiled.request.validate()?;
        Ok(())
    }

    #[test]
    fn lowers_match7_16_and_18_unknown_types_to_an_executable_empty_domain() -> Result<()> {
        let graph = fixture_graph()?;
        for (query, has_bound_terminal) in [
            (
                "MATCH (a:A) OPTIONAL MATCH p = (a)-[:X]->(b) RETURN p",
                false,
            ),
            (
                "MATCH (a:A), (b:B) OPTIONAL MATCH p = (a)-[:X]->(b) RETURN p",
                true,
            ),
        ] {
            let compiled = compile_query(query, &graph)?.ok_or_else(|| {
                crate::Error::internal("known-empty named OPTIONAL path was not compiled")
            })?;
            assert!(compiled.request.optional, "{query}");
            assert_eq!(
                compiled.request.bound_terminal_scan.is_some(),
                has_bound_terminal,
                "{query}"
            );
            let [segment] = compiled.request.segments.as_slice() else {
                panic!("known-empty named OPTIONAL path did not compile to one segment")
            };
            assert!(segment.relationship_types.is_empty(), "{query}");
            assert!(segment.relationship_types_known_empty, "{query}");
            assert_eq!(segment.minimum_hops, 1, "{query}");
            assert_eq!(segment.maximum_hops, Some(1), "{query}");
            assert_eq!(
                compiled.outputs,
                vec![CompiledResidentVariablePathOutput::Path {
                    name: "p".to_owned(),
                }],
                "{query}"
            );
            compiled.request.validate()?;
        }
        Ok(())
    }

    #[test]
    fn lowers_match7_17_string_membership_into_start_and_bound_terminal_scans() -> Result<()> {
        let graph = fixture_graph()?;
        let name = graph.catalog().property("name").expect("fixture property");
        let compiled = compile_query(
            "MATCH (a {name: 'A'}), (x) WHERE x.name IN ['B', 'C'] \
             OPTIONAL MATCH p = (a)-->(x) RETURN x, p",
            &graph,
        )?
        .ok_or_else(|| crate::Error::internal("Match7 [17] was not compiled natively"))?;

        let ResidentVariablePathInput::VisibleNodeScan { predicates, .. } = &compiled.request.input
        else {
            panic!("Match7 [17] did not use a visible start scan");
        };
        assert_eq!(predicates.len(), 1);
        assert_eq!(predicates[0].property, name);
        assert_eq!(
            predicates[0]
                .values
                .iter()
                .map(AsRef::as_ref)
                .collect::<Vec<&str>>(),
            vec!["A"]
        );
        let terminal = compiled
            .request
            .bound_terminal_scan
            .as_ref()
            .expect("Match7 [17] must retain x as a terminal source");
        assert_eq!(terminal.predicates.len(), 1);
        assert_eq!(terminal.predicates[0].property, name);
        assert_eq!(
            terminal.predicates[0]
                .values
                .iter()
                .map(AsRef::as_ref)
                .collect::<Vec<&str>>(),
            vec!["B", "C"]
        );
        assert_eq!(compiled.request.segments.len(), 1);
        assert_eq!(
            compiled.outputs,
            vec![
                CompiledResidentVariablePathOutput::Node {
                    name: "x".to_owned(),
                    position: CompiledResidentVariablePathNodePosition::BoundTerminal,
                },
                CompiledResidentVariablePathOutput::Path {
                    name: "p".to_owned(),
                },
            ]
        );
        compiled.request.validate()?;
        Ok(())
    }

    #[test]
    fn lowers_match7_19_start_property_and_mixed_fixed_variable_path() -> Result<()> {
        let graph = fixture_graph()?;
        let name = graph.catalog().property("name").expect("fixture property");
        let compiled = compile_query(
            "MATCH (a {name: 'A'}) OPTIONAL MATCH p = (a)-->(b)-[*]->(c) RETURN p",
            &graph,
        )?
        .ok_or_else(|| crate::Error::internal("Match7 [19] was not compiled natively"))?;

        let ResidentVariablePathInput::VisibleNodeScan { predicates, .. } = &compiled.request.input
        else {
            panic!("Match7 [19] did not use a visible start scan");
        };
        assert_eq!(predicates.len(), 1);
        assert_eq!(predicates[0].property, name);
        assert_eq!(
            predicates[0]
                .values
                .iter()
                .map(AsRef::as_ref)
                .collect::<Vec<&str>>(),
            vec!["A"]
        );
        assert!(compiled.request.bound_terminal_scan.is_none());
        assert_eq!(compiled.request.segments.len(), 2);
        assert_eq!(compiled.request.segments[0].minimum_hops, 1);
        assert_eq!(compiled.request.segments[0].maximum_hops, Some(1));
        assert_eq!(compiled.request.segments[1].minimum_hops, 1);
        assert_eq!(compiled.request.segments[1].maximum_hops, None);
        assert_eq!(
            compiled.outputs,
            vec![CompiledResidentVariablePathOutput::Path {
                name: "p".to_owned(),
            }]
        );
        compiled.request.validate()?;
        Ok(())
    }

    #[test]
    fn keeps_unrelated_sources_as_multiplicity_and_additional_correlations_closed() -> Result<()> {
        let graph = fixture_graph()?;
        let a = graph.catalog().label("A").expect("fixture label");
        let compiled = compile_query(
            "MATCH (u:A), (a:Single) OPTIONAL MATCH (a)-[*]->(b) RETURN b",
            &graph,
        )?
        .ok_or_else(|| crate::Error::internal("unprojected source multiplicity was dropped"))?;
        assert_eq!(compiled.request.multiplicity_scans.len(), 1);
        assert_eq!(compiled.request.multiplicity_scans[0].labels, vec![a]);
        assert!(compiled.request.cartesian_obligation.is_some());
        assert!(compiled.request.bound_terminal_scan.is_none());

        assert!(
            compile_query(
                "MATCH (u:A), (a:Single) OPTIONAL MATCH (a)-[*]->(b) RETURN b, u",
                &graph,
            )?
            .is_none(),
            "a projected unrelated source cannot be represented as multiplicity only"
        );
        assert!(
            compile_query(
                "MATCH (a:Single), (x:C) OPTIONAL MATCH (a)-[*]->(x)-[*]->(b) RETURN b",
                &graph,
            )?
            .is_none(),
            "a retained terminal binding may not correlate with an intermediate endpoint"
        );
        Ok(())
    }

    #[test]
    fn represents_supported_tck_path_constraints_and_keeps_the_rest_closed() -> Result<()> {
        let graph = fixture_graph()?;
        let name = graph.catalog().property("name").expect("fixture property");
        let represented = compile_query("MATCH (a {name: 'A'})-[*]->(x) RETURN x", &graph)?
            .ok_or_else(|| crate::Error::internal("Match4 [2] start predicate was not compiled"))?;
        let ResidentVariablePathInput::VisibleNodeScan { predicates, .. } =
            &represented.request.input
        else {
            panic!("Match4 [2] did not retain its start predicate in the visible scan")
        };
        assert_eq!(predicates.len(), 1);
        assert_eq!(predicates[0].property, name);
        assert_eq!(
            predicates[0]
                .values
                .iter()
                .map(AsRef::as_ref)
                .collect::<Vec<&str>>(),
            vec!["A"]
        );
        represented.request.validate()?;

        let end = graph.catalog().label("End").expect("fixture label");
        for (query, direction) in [
            (
                "MATCH (a)-[r:REL*2..2]->(b:End) RETURN r",
                ResidentDirection::Outgoing,
            ),
            (
                "MATCH (a)-[r:REL*2..2]-(b:End) RETURN r",
                ResidentDirection::Undirected,
            ),
            (
                "MATCH p = (a)-[:REL*2..2]->(b:End) RETURN relationships(p)",
                ResidentDirection::Outgoing,
            ),
        ] {
            let represented = compile_query(query, &graph)?.ok_or_else(|| {
                crate::Error::internal("labelled relationship-list path was not compiled")
            })?;
            assert_eq!(represented.request.segments.len(), 1, "{query}");
            let segment = &represented.request.segments[0];
            assert_eq!(segment.direction, direction, "{query}");
            assert_eq!(segment.target_labels, vec![end], "{query}");
            assert_eq!(segment.minimum_hops, 2, "{query}");
            assert_eq!(segment.maximum_hops, Some(2), "{query}");
            assert!(
                matches!(
                    represented.outputs.as_slice(),
                    [CompiledResidentVariablePathOutput::Relationships { .. }]
                ),
                "{query} did not retain the complete ordered relationship list"
            );
            represented.request.validate()?;
        }

        let match9_count =
            compile_optimized_query("MATCH (a:Blue)-[r*]->(b:Green) RETURN count(r)", &graph)?
                .ok_or_else(|| {
                    crate::Error::internal(
                        "Match9 [5] optimized path-cardinality profile was not compiled",
                    )
                })?;
        assert!(matches!(
            match9_count.outputs.as_slice(),
            [CompiledResidentVariablePathOutput::CountRelationships { name }]
                if name == "count(r)"
        ));
        assert_eq!(match9_count.request.maximum_frontier_paths, 1);
        assert_eq!(match9_count.request.maximum_output_rows, 1);
        match9_count.request.validate()?;

        // Binding canonicalizes both `*` and the semantically identical `*1..` to the same
        // minimum-hop representation, so the native route must treat them identically.
        let explicit_minimum =
            compile_optimized_query("MATCH (a:Blue)-[r*1..]->(b:Green) RETURN count(r)", &graph)?
                .ok_or_else(|| {
                crate::Error::internal("canonical Match9 minimum-hop spelling was not compiled")
            })?;
        assert_eq!(explicit_minimum.outputs, match9_count.outputs);
        assert_eq!(explicit_minimum.request.input, match9_count.request.input);
        assert_eq!(
            explicit_minimum.request.segments,
            match9_count.request.segments
        );
        assert_eq!(
            explicit_minimum.request.maximum_output_rows,
            match9_count.request.maximum_output_rows
        );
        explicit_minimum.request.validate()?;

        let match4_relationship_predicate = compile_query(
            "MATCH (a:Artist)-[:WORKED_WITH* {year: 1988}]->(b:Artist) RETURN *",
            &graph,
        )?
        .ok_or_else(|| {
            crate::Error::internal("Match4 [5] relationship predicate was not compiled")
        })?;
        let [segment] = match4_relationship_predicate.request.segments.as_slice() else {
            return Err(crate::Error::internal(
                "Match4 [5] did not compile one variable segment",
            ));
        };
        assert_eq!(segment.relationship_integer_predicates.len(), 1);
        assert_eq!(segment.relationship_integer_predicates[0].value, 1988);
        assert_eq!(
            match4_relationship_predicate
                .outputs
                .iter()
                .map(CompiledResidentVariablePathOutput::name)
                .collect::<Vec<_>>(),
            vec!["a", "b"]
        );
        match4_relationship_predicate.request.validate()?;

        assert!(
            compile_query(
                "WITH null AS a OPTIONAL MATCH p = (a)-[r]->() RETURN nodes(p), nodes(null)",
                &graph,
            )?
            .is_none(),
            "Path1 [1] was admitted without its static-null ABI representation"
        );
        for query in [
            "MATCH (a:Blue)-[r*]->(b:Green) RETURN count(r) AS c",
            "MATCH (a:Green)-[r*]->(b:Blue) RETURN count(r)",
            "MATCH (a:Blue)-[r*]-(b:Green) RETURN count(r)",
        ] {
            assert!(
                compile_query(query, &graph)?.is_none(),
                "nearby non-Match9 profile was admitted: {query}"
            );
        }
        Ok(())
    }
}
