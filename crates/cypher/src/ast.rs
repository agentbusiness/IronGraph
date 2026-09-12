//! Lossless-enough semantic AST for Cypher clauses and required declarative extensions.

use std::sync::Arc;

use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

use crate::{Layer, ScalarValue, graph::LayerMask};

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Query {
    pub project: Option<String>,
    pub read_layers: LayerMask,
    pub write_layer: Layer,
    pub at_time: Option<Expression>,
    pub statement: Statement,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Statement {
    Query(QueryBody),
    CreateProject {
        name: String,
        if_not_exists: bool,
    },
    ShowProjects,
    ImportDataset {
        name: String,
    },
    ShowTopics,
    ShowQueues,
    ShowExchanges,
    ShowConsumerLag,
    CheckReadOnly,
    CreateTopic {
        name: String,
        partitions: u16,
        retention_days: Option<u32>,
    },
    AlterTopicRetention {
        name: String,
        retention_days: u32,
    },
    DropTopic {
        name: String,
    },
    ClearTopic {
        name: String,
    },
    CreateQueue {
        name: String,
        stream: bool,
        retention_days: Option<u32>,
    },
    AlterQueueRetention {
        name: String,
        retention_days: u32,
    },
    DropQueue {
        name: String,
    },
    PurgeQueue {
        name: String,
    },
    CreateExchange {
        name: String,
        kind: String,
    },
    DropExchange {
        name: String,
    },
    BindQueue {
        queue: String,
        exchange: String,
        routing_key: String,
    },
    UnbindQueue {
        queue: String,
        exchange: String,
        routing_key: String,
    },
    AlterProjectRename {
        name: String,
        new_name: String,
    },
    DropProject {
        name: String,
        if_exists: bool,
        cascade: bool,
    },
    CreateIndex(IndexDefinition),
    ShowIndexes,
    CreateConstraint(UniqueConstraintDefinition),
    ShowConstraints,
    RebuildIndex {
        name: String,
    },
    DropIndex {
        name: String,
        if_exists: bool,
    },
    DropConstraint {
        name: String,
        if_exists: bool,
    },
    DeclareTemporal(TemporalPropertyDeclaration),
    CreateRollup(RollupDefinition),
    CreateEmbedding(EmbeddingDefinition),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct QueryBody {
    pub clauses: Vec<Clause>,
    pub unions: Vec<UnionBranch>,
}

/// One parsed `EXISTS { ... }` expression.
///
/// The body remains an owned Cypher AST throughout parsing, binding, planning, and execution. It
/// is encoded only to fit inside the pre-existing expression envelope; execution decodes this
/// structure directly and never stringifies or reparses Cypher source.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ExistentialSubquery {
    pub body: QueryBody,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct UnionBranch {
    pub all: bool,
    pub body: Vec<Clause>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Clause {
    Match {
        optional: bool,
        patterns: Vec<Pattern>,
    },
    Where(Expression),
    Unwind {
        expression: Expression,
        variable: String,
    },
    /// GQL-aligned spelling of `UNWIND` introduced by Cypher 25.
    For {
        variable: String,
        expression: Expression,
    },
    /// Scope-preserving expression bindings (`WITH *, expression AS variable`).
    Let(Vec<LetItem>),
    /// Cypher 25 standalone row filter.
    Filter(Expression),
    Create(Vec<Pattern>),
    Merge {
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
    With(Projection),
    Return(Projection),
    OrderBy(Vec<SortItem>),
    Skip(Expression),
    Limit(Expression),
    History(HistoryClause),
    Window(WindowClause),
    Search(SearchClause),
    Call(CallClause),
    /// Executes the preceding pipeline for side effects and emits no columns or rows.
    Finish,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LetItem {
    pub variable: String,
    pub expression: Expression,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Pattern {
    pub variable: Option<String>,
    pub selector: PathSelector,
    pub mode: PathMode,
    pub start: NodePattern,
    pub steps: Vec<PatternStep>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum PathSelector {
    #[default]
    All,
    Any,
    AnyShortest,
    AllShortest,
    Shortest {
        count: u32,
        groups: bool,
    },
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum PathMode {
    #[default]
    DifferentRelationships,
    RepeatableElements,
    Acyclic,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct NodePattern {
    pub variable: Option<String>,
    pub labels: Vec<String>,
    /// Whether the source contained a property predicate, including an explicitly empty `{}`.
    ///
    /// This is syntax identity, not a value-derived shortcut: write-pattern binding must
    /// distinguish a reusable naked endpoint `(n)` from a redeclaration `(n {})` even though both
    /// carry zero property entries.
    pub property_predicate_present: bool,
    pub properties: Vec<(String, Expression)>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PatternStep {
    pub relationship: RelationshipPattern,
    pub node: NodePattern,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RelationshipPattern {
    pub variable: Option<String>,
    pub types: Vec<String>,
    pub direction: Direction,
    pub variable_length: bool,
    pub min_hops: Option<u32>,
    pub max_hops: Option<u32>,
    pub properties: Vec<(String, Expression)>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Direction {
    Outgoing,
    Incoming,
    Undirected,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SetItem {
    Property {
        target: PropertyAccess,
        value: Expression,
        event_time: Option<Expression>,
    },
    MergeMap {
        variable: String,
        value: Expression,
    },
    ReplaceMap {
        variable: String,
        value: Expression,
    },
    Labels {
        variable: String,
        labels: Vec<LabelName>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum RemoveItem {
    Property(PropertyAccess),
    Labels {
        variable: String,
        labels: Vec<LabelName>,
    },
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum LabelName {
    Static(String),
    Dynamic(Expression),
    DynamicAll(Expression),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PropertyAccess {
    pub variable: String,
    pub property: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Projection {
    pub distinct: bool,
    pub items: Vec<ProjectionItem>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ProjectionItem {
    pub expression: Expression,
    pub alias: Option<String>,
    /// Exact source spelling of a parsed RETURN/WITH expression, excluding its alias and the
    /// surrounding projection delimiter. Planner-synthesized items deliberately carry `None`.
    pub source_text: Option<Arc<str>>,
}

impl ProjectionItem {
    /// Client-visible projection column name.
    ///
    /// An explicit alias wins. Otherwise a parsed projection keeps its exact source spelling;
    /// synthesized projections fall back to the canonical AST renderer.
    #[must_use]
    pub fn column_name(&self, index: usize) -> String {
        self.alias
            .clone()
            .or_else(|| self.source_text.as_deref().map(str::to_owned))
            .unwrap_or_else(|| expression_display_name(&self.expression, index))
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SortItem {
    pub expression: Expression,
    pub ascending: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct HistoryClause {
    pub target: PropertyAccess,
    pub from: Expression,
    pub to: Expression,
    pub variable: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WindowClause {
    pub kind: WindowSyntax,
    pub width: Expression,
    pub every: Option<Expression>,
    pub event_expression: Expression,
    pub align: Option<Expression>,
    pub timezone: Option<String>,
    pub emit_empty: bool,
    pub variable: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WindowSyntax {
    Tumbling,
    Hopping,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SearchClause {
    pub variable: String,
    pub index: String,
    pub input: SearchInput,
    pub limit: Expression,
    pub score_variable: String,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum SearchInput {
    Text(Expression),
    Vector(Expression),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CallClause {
    pub name: Vec<String>,
    pub argument_mode: CallArgumentMode,
    pub arguments: Vec<Expression>,
    pub yield_mode: CallYieldMode,
    pub yields: Vec<ProjectionItem>,
    /// Whether this CALL is the sole clause in its query branch. Standalone calls expose all
    /// declared outputs when YIELD is omitted; in-query calls expose only explicit YIELD items.
    pub standalone: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CallArgumentMode {
    Explicit,
    Implicit,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum CallYieldMode {
    Omitted,
    Explicit,
    All,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Expression {
    Literal(ScalarValue),
    Parameter(String),
    Variable(String),
    Property(Box<Expression>, String),
    List(Vec<Expression>),
    Map(Vec<(String, Expression)>),
    MapProjection {
        source: Box<Expression>,
        items: Vec<MapProjectionItem>,
    },
    Case {
        operand: Option<Box<Expression>>,
        alternatives: Vec<CaseAlternative>,
        default: Option<Box<Expression>>,
    },
    ListComprehension {
        variable: String,
        list: Box<Expression>,
        predicate: Option<Box<Expression>>,
        projection: Option<Box<Expression>>,
    },
    Reduce {
        accumulator: String,
        initial: Box<Expression>,
        variable: String,
        list: Box<Expression>,
        expression: Box<Expression>,
    },
    ListPredicate {
        kind: ListPredicateKind,
        variable: String,
        list: Box<Expression>,
        predicate: Box<Expression>,
    },
    Function {
        name: Vec<String>,
        distinct: bool,
        arguments: Vec<Expression>,
    },
    /// One parsed correlated existential subquery. Keeping the typed body directly in the AST
    /// makes nesting linear in both memory and work; it is never exposed as a user-callable
    /// function and needs no encoded envelope or nesting bound.
    ExistentialSubquery(Box<ExistentialSubquery>),
    Unary {
        operation: UnaryOperator,
        operand: Box<Expression>,
    },
    Binary {
        left: Box<Expression>,
        operation: BinaryOperator,
        right: Box<Expression>,
    },
    IsNull {
        expression: Box<Expression>,
        negated: bool,
    },
    Index {
        expression: Box<Expression>,
        index: Box<Expression>,
    },
    Slice {
        expression: Box<Expression>,
        start: Option<Box<Expression>>,
        end: Option<Box<Expression>>,
    },
    Star,
}

/// Internal AST-intrinsic name for the polymorphic `value:Name[:Name...]` expression.
///
/// This contains a NUL so ordinary Cypher source cannot spell it as a function name. Keeping the
/// intrinsic in the existing function-shaped envelope avoids making every in-flight exhaustive
/// expression visitor uncompilable while still giving binder/executor/resident code one exact,
/// fail-closed shape to recognize.
pub const ENTITY_LABEL_PREDICATE_INTRINSIC: &str = "\0irongraph.entity_label_predicate";

/// Internal AST-intrinsic names for a parsed existential pattern predicate and its typed parts.
///
/// These names contain a NUL and therefore cannot be produced by Cypher source. The expression
/// envelope keeps the existing exhaustive expression visitors source-compatible while retaining
/// the complete pattern until the planner and native backends grow a dedicated predicate
/// operator. Consumers must decode the exact envelope instead of treating it as a host function.
pub const PATTERN_PREDICATE_INTRINSIC: &str = "\0irongraph.pattern_predicate";
/// Internal AST-intrinsic name for an openCypher pattern comprehension.
///
/// A pattern comprehension is not an existential predicate and is not an ordinary list
/// comprehension. Its graph pattern introduces a temporary graph-typed scope used by its optional
/// filter and required projection. Keeping a distinct unspellable envelope prevents generic host
/// function evaluation from accidentally claiming support for those semantics.
pub const PATTERN_COMPREHENSION_INTRINSIC: &str = "\0irongraph.pattern_comprehension";
const PATTERN_NODE_INTRINSIC: &str = "\0irongraph.pattern_node";
const PATTERN_RELATIONSHIP_INTRINSIC: &str = "\0irongraph.pattern_relationship";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct CaseAlternative {
    pub when: Expression,
    pub then: Expression,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum MapProjectionItem {
    AllProperties,
    Property(String),
    Variable(String),
    Entry(String, Expression),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ListPredicateKind {
    All,
    Any,
    None,
    Single,
}

impl Expression {
    #[must_use]
    pub fn integer(value: i64) -> Self {
        Self::Literal(ScalarValue::Integer(value))
    }

    #[must_use]
    pub fn float(value: f64) -> Self {
        Self::Literal(ScalarValue::Float(OrderedFloat(value)))
    }

    #[must_use]
    pub fn string(value: impl Into<Arc<str>>) -> Self {
        Self::Literal(ScalarValue::String(value.into()))
    }

    /// Build the polymorphic openCypher colon predicate without choosing an entity kind on the
    /// host. Runtime semantics are part of this intrinsic's contract:
    ///
    /// - NULL produces NULL;
    /// - a node is true only when it has every named label;
    /// - a relationship is true only when its one exact, case-sensitive type satisfies every
    ///   named type predicate; and
    /// - every other source kind raises the openCypher invalid-argument type error.
    pub fn entity_label_predicate(source: Self, names: Vec<String>) -> Self {
        let mut arguments = Vec::with_capacity(names.len().saturating_add(1));
        arguments.push(source);
        arguments.extend(names.into_iter().map(Self::string));
        Self::Function {
            name: vec![ENTITY_LABEL_PREDICATE_INTRINSIC.to_owned()],
            distinct: false,
            arguments,
        }
    }

    /// Decode only the exact parser-produced intrinsic. Consumers must explicitly implement its
    /// polymorphic semantics; treating it as `labels(source)` or as `type(source)` is incorrect.
    pub fn entity_label_predicate_parts(&self) -> Option<(&Self, &[Self])> {
        let Self::Function {
            name,
            distinct: false,
            arguments,
        } = self
        else {
            return None;
        };
        if name.as_slice() != [ENTITY_LABEL_PREDICATE_INTRINSIC] {
            return None;
        }
        let (source, names) = arguments.split_first()?;
        if names.is_empty()
            || names
                .iter()
                .any(|name| !matches!(name, Self::Literal(ScalarValue::String(_))))
        {
            return None;
        }
        Some((source, names))
    }

    /// Preserve an existential pattern predicate as one unspellable, lossless AST envelope.
    ///
    /// This is deliberately not a user-callable function. It is a compatibility bridge for the
    /// expression enum while every execution backend gains the same native pattern-predicate
    /// operator. The binder recognizes only envelopes produced by this constructor.
    pub fn pattern_predicate(pattern: Pattern) -> Self {
        Self::Function {
            name: vec![PATTERN_PREDICATE_INTRINSIC.to_owned()],
            distinct: false,
            arguments: encode_pattern(pattern),
        }
    }

    /// Decode only the exact parser-produced existential pattern-predicate envelope.
    ///
    /// Returning an owned `Pattern` is intentional: binding and planning may validate it without
    /// exposing the storage-shaped intrinsic arguments as a second public pattern API.
    pub fn pattern_predicate_pattern(&self) -> Option<Pattern> {
        let Self::Function {
            name,
            distinct: false,
            arguments,
        } = self
        else {
            return None;
        };
        if name.as_slice() != [PATTERN_PREDICATE_INTRINSIC] {
            return None;
        }
        decode_pattern(arguments)
    }

    /// Store one typed existential-subquery AST in the expression tree. Generic expression
    /// walkers treat this as an opaque lexical boundary, so aggregates and variables inside the
    /// body are never mistaken for members of the surrounding expression.
    pub fn existential_subquery(subquery: ExistentialSubquery) -> crate::Result<Self> {
        Ok(Self::ExistentialSubquery(Box::new(subquery)))
    }

    /// Borrow the parser-owned existential-subquery body.
    pub fn existential_subquery_parts(&self) -> Option<&ExistentialSubquery> {
        let Self::ExistentialSubquery(subquery) = self else {
            return None;
        };
        Some(subquery)
    }

    /// Preserve a pattern comprehension as a dedicated, lossless internal expression.
    ///
    /// The first five arguments are the canonical pattern envelope shared with pattern
    /// predicates. The sixth is a zero-or-one list carrying the optional pattern filter, and the
    /// seventh is the required projection. This shape is deliberately distinct from both the
    /// pattern-predicate intrinsic and `ListComprehension`.
    pub fn pattern_comprehension(
        pattern: Pattern,
        predicate: Option<Self>,
        projection: Self,
    ) -> Self {
        let mut arguments = encode_pattern(pattern);
        arguments.push(Self::List(predicate.into_iter().collect()));
        arguments.push(projection);
        Self::Function {
            name: vec![PATTERN_COMPREHENSION_INTRINSIC.to_owned()],
            distinct: false,
            arguments,
        }
    }

    /// Decode only the exact parser-produced pattern-comprehension envelope.
    pub fn pattern_comprehension_parts(&self) -> Option<(Pattern, Option<&Self>, &Self)> {
        let Self::Function {
            name,
            distinct: false,
            arguments,
        } = self
        else {
            return None;
        };
        if name.as_slice() != [PATTERN_COMPREHENSION_INTRINSIC] {
            return None;
        }
        let [pattern @ .., predicate, projection] = arguments.as_slice() else {
            return None;
        };
        if pattern.len() != 5 {
            return None;
        }
        let Expression::List(predicate) = predicate else {
            return None;
        };
        let predicate = match predicate.as_slice() {
            [] => None,
            [predicate] => Some(predicate),
            _ => return None,
        };
        Some((decode_pattern(pattern)?, predicate, projection))
    }
}

fn encode_pattern(pattern: Pattern) -> Vec<Expression> {
    let steps = pattern
        .steps
        .into_iter()
        .map(|step| {
            Expression::List(vec![
                encode_pattern_relationship(step.relationship),
                encode_pattern_node(step.node),
            ])
        })
        .collect();
    vec![
        encode_optional_pattern_name(pattern.variable),
        encode_path_selector(pattern.selector),
        Expression::integer(encode_path_mode(pattern.mode)),
        encode_pattern_node(pattern.start),
        Expression::List(steps),
    ]
}

fn decode_pattern(arguments: &[Expression]) -> Option<Pattern> {
    let [variable, selector, mode, start, steps] = arguments else {
        return None;
    };
    let Expression::List(steps) = steps else {
        return None;
    };
    let steps = steps
        .iter()
        .map(|step| {
            let Expression::List(parts) = step else {
                return None;
            };
            let [relationship, node] = parts.as_slice() else {
                return None;
            };
            Some(PatternStep {
                relationship: decode_pattern_relationship(relationship)?,
                node: decode_pattern_node(node)?,
            })
        })
        .collect::<Option<Vec<_>>>()?;
    Some(Pattern {
        variable: decode_optional_pattern_name(variable)?,
        selector: decode_path_selector(selector)?,
        mode: decode_path_mode(mode)?,
        start: decode_pattern_node(start)?,
        steps,
    })
}

fn encode_optional_pattern_name(name: Option<String>) -> Expression {
    name.map_or(Expression::Literal(ScalarValue::Null), Expression::string)
}

fn decode_optional_pattern_name(expression: &Expression) -> Option<Option<String>> {
    match expression {
        Expression::Literal(ScalarValue::Null) => Some(None),
        Expression::Literal(ScalarValue::String(name)) => Some(Some(name.to_string())),
        _ => None,
    }
}

fn encode_optional_pattern_bound(bound: Option<u32>) -> Expression {
    bound.map_or(Expression::Literal(ScalarValue::Null), |bound| {
        Expression::integer(i64::from(bound))
    })
}

fn decode_optional_pattern_bound(expression: &Expression) -> Option<Option<u32>> {
    match expression {
        Expression::Literal(ScalarValue::Null) => Some(None),
        Expression::Literal(ScalarValue::Integer(bound)) => Some(Some(u32::try_from(*bound).ok()?)),
        _ => None,
    }
}

fn encode_path_selector(selector: PathSelector) -> Expression {
    let (tag, count, groups) = match selector {
        PathSelector::All => (0, None, false),
        PathSelector::Any => (1, None, false),
        PathSelector::AnyShortest => (2, None, false),
        PathSelector::AllShortest => (3, None, false),
        PathSelector::Shortest { count, groups } => (4, Some(count), groups),
    };
    Expression::List(vec![
        Expression::integer(tag),
        encode_optional_pattern_bound(count),
        Expression::Literal(ScalarValue::Boolean(groups)),
    ])
}

fn decode_path_selector(expression: &Expression) -> Option<PathSelector> {
    let Expression::List(values) = expression else {
        return None;
    };
    let [
        Expression::Literal(ScalarValue::Integer(tag)),
        count,
        Expression::Literal(ScalarValue::Boolean(groups)),
    ] = values.as_slice()
    else {
        return None;
    };
    let count = decode_optional_pattern_bound(count)?;
    match (*tag, count, *groups) {
        (0, None, false) => Some(PathSelector::All),
        (1, None, false) => Some(PathSelector::Any),
        (2, None, false) => Some(PathSelector::AnyShortest),
        (3, None, false) => Some(PathSelector::AllShortest),
        (4, Some(count), groups) if count > 0 => Some(PathSelector::Shortest { count, groups }),
        _ => None,
    }
}

const fn encode_path_mode(mode: PathMode) -> i64 {
    match mode {
        PathMode::DifferentRelationships => 0,
        PathMode::RepeatableElements => 1,
        PathMode::Acyclic => 2,
    }
}

fn decode_path_mode(expression: &Expression) -> Option<PathMode> {
    match expression {
        Expression::Literal(ScalarValue::Integer(0)) => Some(PathMode::DifferentRelationships),
        Expression::Literal(ScalarValue::Integer(1)) => Some(PathMode::RepeatableElements),
        Expression::Literal(ScalarValue::Integer(2)) => Some(PathMode::Acyclic),
        _ => None,
    }
}

fn encode_pattern_node(node: NodePattern) -> Expression {
    Expression::Function {
        name: vec![PATTERN_NODE_INTRINSIC.to_owned()],
        distinct: false,
        arguments: vec![
            encode_optional_pattern_name(node.variable),
            Expression::List(node.labels.into_iter().map(Expression::string).collect()),
            Expression::Literal(ScalarValue::Boolean(node.property_predicate_present)),
            Expression::Map(node.properties),
        ],
    }
}

fn decode_pattern_node(expression: &Expression) -> Option<NodePattern> {
    let Expression::Function {
        name,
        distinct: false,
        arguments,
    } = expression
    else {
        return None;
    };
    if name.as_slice() != [PATTERN_NODE_INTRINSIC] {
        return None;
    }
    let [variable, labels, property_predicate_present, properties] = arguments.as_slice() else {
        return None;
    };
    let Expression::List(labels) = labels else {
        return None;
    };
    let labels = labels
        .iter()
        .map(|label| match label {
            Expression::Literal(ScalarValue::String(label)) => Some(label.to_string()),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    let Expression::Map(properties) = properties else {
        return None;
    };
    let Expression::Literal(ScalarValue::Boolean(property_predicate_present)) =
        property_predicate_present
    else {
        return None;
    };
    Some(NodePattern {
        variable: decode_optional_pattern_name(variable)?,
        labels,
        property_predicate_present: *property_predicate_present,
        properties: properties.clone(),
    })
}

fn encode_pattern_relationship(relationship: RelationshipPattern) -> Expression {
    Expression::Function {
        name: vec![PATTERN_RELATIONSHIP_INTRINSIC.to_owned()],
        distinct: false,
        arguments: vec![
            encode_optional_pattern_name(relationship.variable),
            Expression::List(
                relationship
                    .types
                    .into_iter()
                    .map(Expression::string)
                    .collect(),
            ),
            Expression::integer(match relationship.direction {
                Direction::Outgoing => 0,
                Direction::Incoming => 1,
                Direction::Undirected => 2,
            }),
            Expression::Literal(ScalarValue::Boolean(relationship.variable_length)),
            encode_optional_pattern_bound(relationship.min_hops),
            encode_optional_pattern_bound(relationship.max_hops),
            Expression::Map(relationship.properties),
        ],
    }
}

fn decode_pattern_relationship(expression: &Expression) -> Option<RelationshipPattern> {
    let Expression::Function {
        name,
        distinct: false,
        arguments,
    } = expression
    else {
        return None;
    };
    if name.as_slice() != [PATTERN_RELATIONSHIP_INTRINSIC] {
        return None;
    }
    let [
        variable,
        types,
        direction,
        variable_length,
        min_hops,
        max_hops,
        properties,
    ] = arguments.as_slice()
    else {
        return None;
    };
    let Expression::List(types) = types else {
        return None;
    };
    let types = types
        .iter()
        .map(|relationship_type| match relationship_type {
            Expression::Literal(ScalarValue::String(relationship_type)) => {
                Some(relationship_type.to_string())
            }
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    let direction = match direction {
        Expression::Literal(ScalarValue::Integer(0)) => Direction::Outgoing,
        Expression::Literal(ScalarValue::Integer(1)) => Direction::Incoming,
        Expression::Literal(ScalarValue::Integer(2)) => Direction::Undirected,
        _ => return None,
    };
    let Expression::Literal(ScalarValue::Boolean(variable_length)) = variable_length else {
        return None;
    };
    let Expression::Map(properties) = properties else {
        return None;
    };
    Some(RelationshipPattern {
        variable: decode_optional_pattern_name(variable)?,
        types,
        direction,
        variable_length: *variable_length,
        min_hops: decode_optional_pattern_bound(min_hops)?,
        max_hops: decode_optional_pattern_bound(max_hops)?,
        properties: properties.clone(),
    })
}

/// Canonical client-visible name for an unaliased projection expression.
///
/// Cypher exposes the expression text as the column name when `AS` is absent. Keeping this
/// renderer next to the AST ensures the generic executor and resident GPU pipelines cannot
/// silently disagree (for example, `count(*)` must not become merely `count`).
pub fn expression_display_name(expression: &Expression, index: usize) -> String {
    if let Some((source, names)) = expression.entity_label_predicate_parts() {
        let mut display = expression_display_name(source, index);
        for name in names {
            let Expression::Literal(ScalarValue::String(name)) = name else {
                continue;
            };
            display.push(':');
            display.push_str(name);
        }
        return display;
    }
    match expression {
        Expression::Literal(value) => scalar_display_name(value),
        Expression::Parameter(name) => format!("${name}"),
        Expression::Variable(name) => name.clone(),
        Expression::Property(source, property) => {
            format!("{}.{}", expression_display_name(source, index), property)
        }
        Expression::List(values) => format!(
            "[{}]",
            values
                .iter()
                .map(|value| expression_display_name(value, index))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Expression::Map(values) => format!(
            "{{{}}}",
            values
                .iter()
                .map(|(name, value)| format!("{name}: {}", expression_display_name(value, index)))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Expression::MapProjection { source, items } => format!(
            "{}{{{}}}",
            expression_display_name(source, index),
            items
                .iter()
                .map(|item| match item {
                    MapProjectionItem::AllProperties => ".*".to_owned(),
                    MapProjectionItem::Property(property) => format!(".{property}"),
                    MapProjectionItem::Variable(variable) => variable.clone(),
                    MapProjectionItem::Entry(name, value) => {
                        format!("{name}: {}", expression_display_name(value, index))
                    }
                })
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Expression::Case {
            operand,
            alternatives,
            default,
        } => {
            let mut parts = vec!["CASE".to_owned()];
            if let Some(operand) = operand {
                parts.push(expression_display_name(operand, index));
            }
            for alternative in alternatives {
                parts.push(format!(
                    "WHEN {} THEN {}",
                    expression_display_name(&alternative.when, index),
                    expression_display_name(&alternative.then, index)
                ));
            }
            if let Some(default) = default {
                parts.push(format!("ELSE {}", expression_display_name(default, index)));
            }
            parts.push("END".to_owned());
            parts.join(" ")
        }
        Expression::ListComprehension {
            variable,
            list,
            predicate,
            projection,
        } => {
            let mut body = format!("{variable} IN {}", expression_display_name(list, index));
            if let Some(predicate) = predicate {
                body.push_str(&format!(
                    " WHERE {}",
                    expression_display_name(predicate, index)
                ));
            }
            if let Some(projection) = projection {
                body.push_str(&format!(
                    " | {}",
                    expression_display_name(projection, index)
                ));
            }
            format!("[{body}]")
        }
        Expression::Reduce {
            accumulator,
            initial,
            variable,
            list,
            expression,
        } => format!(
            "reduce({accumulator} = {}, {variable} IN {} | {})",
            expression_display_name(initial, index),
            expression_display_name(list, index),
            expression_display_name(expression, index)
        ),
        Expression::ListPredicate {
            kind,
            variable,
            list,
            predicate,
        } => format!(
            "{}({variable} IN {} WHERE {})",
            match kind {
                ListPredicateKind::All => "all",
                ListPredicateKind::Any => "any",
                ListPredicateKind::None => "none",
                ListPredicateKind::Single => "single",
            },
            expression_display_name(list, index),
            expression_display_name(predicate, index)
        ),
        Expression::Function {
            name,
            distinct,
            arguments,
        } => format!(
            "{}({}{})",
            name.join("."),
            if *distinct { "DISTINCT " } else { "" },
            arguments
                .iter()
                .map(|argument| expression_display_name(argument, index))
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Expression::Unary { operation, operand } => format!(
            "{}{}",
            match operation {
                UnaryOperator::Not => "NOT ",
                UnaryOperator::Positive => "+",
                UnaryOperator::Negative => "-",
            },
            expression_display_name(operand, index)
        ),
        Expression::Binary {
            left,
            operation,
            right,
        } => format!(
            "{} {} {}",
            expression_display_name(left, index),
            match operation {
                BinaryOperator::Or => "OR",
                BinaryOperator::Xor => "XOR",
                BinaryOperator::And => "AND",
                BinaryOperator::Equal => "=",
                BinaryOperator::NotEqual => "<>",
                BinaryOperator::Less => "<",
                BinaryOperator::LessOrEqual => "<=",
                BinaryOperator::Greater => ">",
                BinaryOperator::GreaterOrEqual => ">=",
                BinaryOperator::In => "IN",
                BinaryOperator::StartsWith => "STARTS WITH",
                BinaryOperator::EndsWith => "ENDS WITH",
                BinaryOperator::Contains => "CONTAINS",
                BinaryOperator::RegexMatch => "=~",
                BinaryOperator::Concat => "+",
                BinaryOperator::Add => "+",
                BinaryOperator::Subtract => "-",
                BinaryOperator::Multiply => "*",
                BinaryOperator::Divide => "/",
                BinaryOperator::Modulo => "%",
                BinaryOperator::Power => "^",
            },
            expression_display_name(right, index)
        ),
        Expression::IsNull {
            expression,
            negated,
        } => format!(
            "{} IS {}NULL",
            expression_display_name(expression, index),
            if *negated { "NOT " } else { "" }
        ),
        Expression::Index {
            expression,
            index: key,
        } => format!(
            "{}[{}]",
            expression_display_name(expression, index),
            expression_display_name(key, index)
        ),
        Expression::Slice {
            expression,
            start,
            end,
        } => format!(
            "{}[{}..{}]",
            expression_display_name(expression, index),
            start
                .as_deref()
                .map_or_else(String::new, |value| expression_display_name(value, index)),
            end.as_deref()
                .map_or_else(String::new, |value| expression_display_name(value, index))
        ),
        Expression::ExistentialSubquery(_) => "EXISTS { ... }".to_owned(),
        Expression::Star => "*".to_owned(),
    }
}

fn scalar_display_name(value: &ScalarValue) -> String {
    match value {
        ScalarValue::Null => "null".to_owned(),
        ScalarValue::Boolean(value) => value.to_string(),
        ScalarValue::Integer(value) => value.to_string(),
        ScalarValue::Float(value) => value.to_string(),
        ScalarValue::String(value) => {
            format!("'{}'", value.replace('\\', "\\\\").replace('\'', "\\'"))
        }
        ScalarValue::Bytes(value) => format!("0x{}", hex::encode(value)),
        value => format!("{value:?}"),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum UnaryOperator {
    Not,
    Positive,
    Negative,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BinaryOperator {
    Or,
    Xor,
    And,
    Equal,
    NotEqual,
    Less,
    LessOrEqual,
    Greater,
    GreaterOrEqual,
    In,
    StartsWith,
    EndsWith,
    Contains,
    RegexMatch,
    Concat,
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    Power,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum IndexKind {
    Equality,
    Range,
    Text,
    Vector,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IndexDefinition {
    pub name: String,
    pub kind: IndexKind,
    pub variable: String,
    pub label: String,
    pub properties: Vec<String>,
}

/// Minimal enforced node-property uniqueness constraint required for deterministic knowledge
/// endpoint reuse. It is schema authority, not merely an index access-path declaration.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct UniqueConstraintDefinition {
    pub name: String,
    pub variable: String,
    pub label: String,
    pub property: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalTarget {
    Node,
    Relationship,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TemporalPropertyDeclaration {
    pub target: TemporalTarget,
    pub label_or_type: String,
    pub property: String,
    pub scalar_type: String,
    pub retention: Expression,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RollupDefinition {
    pub name: String,
    pub variable: String,
    pub label: String,
    pub property: String,
    pub window: WindowSyntax,
    pub width: Expression,
    pub every: Option<Expression>,
    pub align: Option<Expression>,
    pub timezone: Option<String>,
    pub aggregates: Vec<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EmbeddingDefinition {
    pub name: String,
    pub variable: String,
    pub label: String,
    pub source_property: String,
    pub target_property: String,
    pub model: String,
    pub similarity: String,
}
