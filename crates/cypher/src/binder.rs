//! Project/layer capability checks, symbol scopes, and conservative dependency discovery.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

use crate::{Error, ErrorCode, Layer, Result, ScalarValue, graph::NameCatalog};

use super::{
    ResultValue,
    ast::*,
    expression::{
        contains_aggregate, is_aggregate_function, reject_aggregate_in_row_context,
        validate_aggregate_nesting, validate_projection_aggregation,
    },
    procedure::{ProcedureCatalog, ResolvedProcedure, validate_call},
};

/// Non-user-selectable capabilities established below protocol adapters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BindCapabilities {
    pub write: bool,
    pub schema: bool,
    pub knowledge_write: bool,
    /// Permission to write the internal `Workspace` layer. Off for
    /// ordinary user queries; granted only to the trusted internal execution path.
    pub workspace_write: bool,
    /// Test/diagnostic execution contract: an active accelerator must own the complete query
    /// plan instead of falling through to the host executor.
    pub require_native_execution: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DependencyKind {
    FullNodeScan,
    FullRelationshipScan,
    Label,
    RelationshipType,
    Property,
    Index,
    Constraint,
    Temporal,
    ProjectCatalog,
}

/// Logical dependency later paired with the snapshot's conservative version stamp.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct DependencyStamp {
    pub kind: DependencyKind,
    pub name: String,
}

#[derive(Clone, Debug)]
pub struct BoundQuery {
    pub query: Query,
    pub read_only: bool,
    pub dependencies: Vec<DependencyStamp>,
}

pub fn bind(
    query: Query,
    catalog: &NameCatalog,
    capabilities: BindCapabilities,
) -> Result<BoundQuery> {
    bind_internal(query, catalog, capabilities, None, None)
}

/// Binds with ordinary query parameter values available for compile-time semantic checks.
pub fn bind_with_parameters(
    query: Query,
    catalog: &NameCatalog,
    capabilities: BindCapabilities,
    parameters: &BTreeMap<String, ResultValue>,
) -> Result<BoundQuery> {
    bind_internal(query, catalog, capabilities, None, Some(parameters))
}

/// Binds against an immutable execution-scoped procedure catalog and the parameter values whose
/// names/types are part of implicit CALL argument resolution.
pub fn bind_with_procedures(
    query: Query,
    catalog: &NameCatalog,
    capabilities: BindCapabilities,
    procedures: &ProcedureCatalog,
    parameters: &BTreeMap<String, ResultValue>,
) -> Result<BoundQuery> {
    bind_internal(
        query,
        catalog,
        capabilities,
        Some(procedures),
        Some(parameters),
    )
}

fn bind_internal<'a>(
    query: Query,
    catalog: &'a NameCatalog,
    capabilities: BindCapabilities,
    procedures: Option<&'a ProcedureCatalog>,
    parameters: Option<&'a BTreeMap<String, ResultValue>>,
) -> Result<BoundQuery> {
    Binder {
        catalog,
        capabilities,
        procedures,
        parameters,
        scope: BindingScope::new(),
        static_sources: StaticSourceScope::new(),
        static_lists: StaticListScope::new(),
        dependencies: BTreeSet::new(),
        read_only: true,
    }
    .bind(query)
}

/// Static role carried by a variable through clause boundaries.
///
/// Cypher permits node and relationship variables to be referenced by later patterns, but a name
/// cannot change from one graph-entity role to another. Paths are declarations rather than
/// reusable pattern endpoints, while ordinary values can never become pattern entities.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BindingKind {
    Node,
    Relationship,
    NodeList,
    RelationshipList,
    ScalarList,
    Path,
    /// The value came from a list whose element role is not statically known. Read patterns may
    /// resolve it to their required graph role at runtime; definitely scalar list elements use
    /// `NonEntity` and remain compile-time errors when reused as graph entities.
    Dynamic,
    /// A list element proven not to be a node or relationship. Its exact scalar/container kind
    /// can remain dynamic without allowing it to masquerade as a graph entity.
    NonEntity,
    Value,
}

/// A source category is retained only when the binder can prove it from syntax and existing
/// bindings. `Unknown` deliberately covers parameters, indexed values, heterogeneous branches,
/// and every expression whose runtime value could still be property-compatible.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StaticSourceShape {
    Unknown,
    Null,
    Scalar,
    List,
    Map,
    Node,
    Relationship,
    Path,
    Temporal,
    Duration,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StaticFunctionArgumentContract {
    Node,
    Relationship,
    PathLength,
    Size,
    PropertiesSource,
}

impl StaticSourceShape {
    const fn is_definitely_invalid_delete_target(self) -> bool {
        matches!(
            self,
            Self::Scalar | Self::List | Self::Map | Self::Temporal | Self::Duration
        )
    }

    const fn is_definitely_invalid_property_source(self) -> bool {
        matches!(self, Self::Scalar | Self::List)
    }

    const fn is_definitely_invalid_properties_argument(self) -> bool {
        matches!(
            self,
            Self::Scalar | Self::List | Self::Path | Self::Temporal | Self::Duration
        )
    }

    const fn violates_function_contract(self, contract: StaticFunctionArgumentContract) -> bool {
        match contract {
            StaticFunctionArgumentContract::Node => {
                !matches!(self, Self::Unknown | Self::Null | Self::Node)
            }
            StaticFunctionArgumentContract::Relationship => {
                !matches!(self, Self::Unknown | Self::Null | Self::Relationship)
            }
            // The existing runtime accepts length() for paths and the same scalar/container
            // values as size(). The graph-specific compile-time error is therefore limited to
            // values whose node/relationship role is already certain.
            StaticFunctionArgumentContract::PathLength => {
                matches!(self, Self::Node | Self::Relationship)
            }
            // `size()` is polymorphic over containers and strings, whose exact runtime shape is
            // intentionally not guessed here. A statically-known path is the one unambiguous
            // mismatch: paths use `length()` instead.
            StaticFunctionArgumentContract::Size => matches!(self, Self::Path),
            StaticFunctionArgumentContract::PropertiesSource => {
                self.is_definitely_invalid_properties_argument()
            }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PatternBindingMode {
    Match,
    Create,
    Merge,
}

impl PatternBindingMode {
    const fn is_write(self) -> bool {
        matches!(self, Self::Create | Self::Merge)
    }
}

/// Bindings owned by one syntactic graph pattern.
///
/// A write pattern may reference a node that existed when that pattern began, but only as an
/// unconstrained endpoint of a relationship declared by the pattern. A node first declared by the
/// pattern remains part of that pattern's creation declaration, including later occurrences of the
/// same variable. Relationship uniqueness is likewise local to one MATCH pattern: correlation in a
/// separate comma pattern or later MATCH clause remains legal.
struct PatternBindingOwner {
    entry_scope: BindingScope,
    has_relationship: bool,
    match_relationships: BTreeSet<String>,
}

impl PatternBindingOwner {
    fn new(scope: &BindingScope, pattern: &Pattern) -> Self {
        Self {
            entry_scope: scope.clone(),
            has_relationship: !pattern.steps.is_empty(),
            match_relationships: BTreeSet::new(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProjectionBoundary {
    With,
    Return,
}

type BindingScope = BTreeMap<String, BindingKind>;
type StaticSourceScope = BTreeMap<String, StaticSourceShape>;
type StaticListScope = BTreeMap<String, Vec<StaticExpressionKind>>;

struct Binder<'a> {
    catalog: &'a NameCatalog,
    capabilities: BindCapabilities,
    procedures: Option<&'a ProcedureCatalog>,
    parameters: Option<&'a BTreeMap<String, ResultValue>>,
    scope: BindingScope,
    static_sources: StaticSourceScope,
    /// Exact member kinds retained only for literal lists crossing an alias boundary.
    /// Runtime values and parameters never enter this map.
    static_lists: StaticListScope,
    dependencies: BTreeSet<DependencyStamp>,
    read_only: bool,
}

impl Binder<'_> {
    fn bind(mut self, query: Query) -> Result<BoundQuery> {
        self.validate_layers(&query)?;
        match &query.statement {
            Statement::Query(body) => {
                self.bind_clauses(&body.clauses)?;
                for branch in &body.unions {
                    self.scope.clear();
                    self.static_sources.clear();
                    self.static_lists.clear();
                    self.bind_clauses(&branch.body)?;
                }
            }
            Statement::ShowProjects => {
                self.dependency(DependencyKind::ProjectCatalog, "projects");
            }
            Statement::ShowIndexes => {
                self.dependency(DependencyKind::Index, "*");
            }
            Statement::ShowConstraints => {
                self.dependency(DependencyKind::Constraint, "*");
            }
            Statement::CheckReadOnly
            | Statement::ShowTopics
            | Statement::ShowQueues
            | Statement::ShowExchanges
            | Statement::ShowConsumerLag => {}
            Statement::ImportDataset { .. }
            | Statement::CreateProject { .. }
            | Statement::AlterProjectRename { .. }
            | Statement::DropProject { .. }
            | Statement::CreateIndex(_)
            | Statement::CreateConstraint(_)
            | Statement::RebuildIndex { .. }
            | Statement::DropIndex { .. }
            | Statement::DropConstraint { .. }
            | Statement::DeclareTemporal(_)
            | Statement::CreateRollup(_)
            | Statement::CreateEmbedding(_)
            | Statement::CreateTopic { .. }
            | Statement::AlterTopicRetention { .. }
            | Statement::DropTopic { .. }
            | Statement::ClearTopic { .. }
            | Statement::CreateQueue { .. }
            | Statement::AlterQueueRetention { .. }
            | Statement::DropQueue { .. }
            | Statement::PurgeQueue { .. }
            | Statement::CreateExchange { .. }
            | Statement::DropExchange { .. }
            | Statement::BindQueue { .. }
            | Statement::UnbindQueue { .. } => {
                self.require_schema()?;
                self.read_only = false;
            }
        }
        // The write layer must be readable only when the query actually writes to it (now that
        // clause binding has settled `read_only`). A read-only query may scope its reads to a single
        // layer — e.g. `USE LAYER KNOWLEDGE MATCH (n) RETURN n` — without tripping this, so the Graph
        // UI can view the Knowledge layer in isolation even though the default write layer is Observed.
        if !self.read_only && !query.read_layers.contains_layer(query.write_layer) {
            return Err(Error::new(
                ErrorCode::LayerNotAllowed,
                "write layer is not visible",
            ));
        }
        Ok(BoundQuery {
            query,
            read_only: self.read_only,
            dependencies: self.dependencies.into_iter().collect(),
        })
    }

    fn validate_layers(&self, query: &Query) -> Result<()> {
        match query.write_layer {
            Layer::Observed => {}
            Layer::Knowledge if !self.capabilities.knowledge_write => {
                return Err(Error::new(
                    ErrorCode::LayerNotAllowed,
                    "KNOWLEDGE write capability is required",
                ));
            }
            Layer::Knowledge => {}
            // Workspace is an internal working namespace, never writable by
            // ordinary user queries. It is excluded from AUTHORITY so it never leaks into
            // retrieval, and its writes require the internal-only `workspace_write` capability.
            Layer::Workspace if !self.capabilities.workspace_write => {
                return Err(Error::new(
                    ErrorCode::LayerNotAllowed,
                    "WORKSPACE write capability is required",
                ));
            }
            Layer::Workspace => {}
        }
        Ok(())
    }

    fn bind_clauses(&mut self, clauses: &[Clause]) -> Result<()> {
        for (clause_index, clause) in clauses.iter().enumerate() {
            match clause {
                Clause::Match { patterns, .. } => {
                    self.dependency(DependencyKind::FullNodeScan, "*");
                    for pattern in patterns {
                        self.bind_pattern(pattern, PatternBindingMode::Match)?;
                    }
                }
                Clause::Where(expression) => {
                    if !matches!(
                        clause_index
                            .checked_sub(1)
                            .and_then(|index| clauses.get(index)),
                        Some(Clause::With(_))
                    ) {
                        self.bind_predicate_expression(expression)?;
                    }
                }
                Clause::Unwind {
                    expression,
                    variable,
                } => {
                    self.bind_expression(expression)?;
                    let kind = self.list_element_binding_kind(expression);
                    self.scope.insert(variable.clone(), kind);
                    self.static_sources.remove(variable);
                    self.static_lists.remove(variable);
                }
                Clause::For {
                    variable,
                    expression,
                } => {
                    self.bind_expression(expression)?;
                    let kind = self.list_element_binding_kind(expression);
                    self.scope.insert(variable.clone(), kind);
                    self.static_sources.remove(variable);
                    self.static_lists.remove(variable);
                }
                Clause::Let(items) => {
                    for item in items {
                        if contains_aggregate(&item.expression) {
                            return Err(Error::new(
                                ErrorCode::QueryType,
                                "LET does not allow aggregate expressions",
                            ));
                        }
                        self.bind_expression(&item.expression)?;
                        let kind = self.expression_binding_kind(&item.expression);
                        let shape = self.static_source_shape(&item.expression);
                        let list = self.static_list_shape(&item.expression);
                        self.scope.insert(item.variable.clone(), kind);
                        self.set_static_source_shape(&item.variable, shape);
                        self.set_static_list_shape(&item.variable, list);
                    }
                }
                Clause::Filter(expression) => self.bind_predicate_expression(expression)?,
                Clause::Create(patterns) => {
                    self.require_write()?;
                    for pattern in patterns {
                        self.bind_pattern(pattern, PatternBindingMode::Create)?;
                    }
                    self.read_only = false;
                }
                Clause::Merge {
                    pattern,
                    on_create,
                    on_match,
                } => {
                    self.require_write()?;
                    self.bind_pattern(pattern, PatternBindingMode::Merge)?;
                    for item in on_create.iter().chain(on_match) {
                        self.bind_set(item)?;
                    }
                    self.read_only = false;
                }
                Clause::Set(items) => {
                    self.require_write()?;
                    for item in items {
                        self.bind_set(item)?;
                    }
                    self.read_only = false;
                }
                Clause::Remove(items) => {
                    self.require_write()?;
                    for item in items {
                        match item {
                            RemoveItem::Property(property) => {
                                self.require_variable(&property.variable)?;
                                self.dependency(DependencyKind::Property, &property.property);
                            }
                            RemoveItem::Labels { variable, labels } => {
                                self.require_variable(variable)?;
                                for label in labels {
                                    self.bind_label_name(label)?;
                                }
                            }
                        }
                    }
                    self.read_only = false;
                }
                Clause::Delete { expressions, .. } => {
                    self.require_write()?;
                    for expression in expressions {
                        self.bind_expression(expression)?;
                        self.validate_delete_target(expression)?;
                    }
                    self.read_only = false;
                }
                Clause::With(projection) => {
                    let projection_input_scope = self.scope.clone();
                    let projection_input_static_sources = self.static_sources.clone();
                    let projection_input_static_lists = self.static_lists.clone();
                    let defer_expression_alias_validation =
                        defers_with_alias_check_to_aggregate_order_validation(
                            projection,
                            clauses.get(clause_index + 1),
                        );
                    self.bind_projection(
                        projection,
                        ProjectionBoundary::With,
                        defer_expression_alias_validation,
                    )?;
                    let next = self.projection_scope(projection);
                    let next_static_sources = self.projection_static_source_scope(projection);
                    let next_static_lists = self.projection_static_list_scope(projection);
                    // In Cypher, WHERE syntactically following WITH filters the incoming rows
                    // after projection aliases are available but before the WITH scope is pruned.
                    // This permits `WITH c WHERE r IS NULL`: `c` is projected, while `r` is an
                    // incoming OPTIONAL MATCH binding used only by the filter.
                    if let Some(Clause::Where(expression)) = clauses.get(clause_index + 1) {
                        let mut filter_scope = self.scope.clone();
                        filter_scope.extend(next.iter().map(|(name, kind)| (name.clone(), *kind)));
                        let mut filter_static_sources = self.static_sources.clone();
                        filter_static_sources.extend(
                            next_static_sources
                                .iter()
                                .map(|(name, shape)| (name.clone(), *shape)),
                        );
                        self.scope = filter_scope;
                        self.static_sources = filter_static_sources;
                        let mut filter_static_lists = self.static_lists.clone();
                        filter_static_lists.extend(
                            next_static_lists
                                .iter()
                                .map(|(name, shape)| (name.clone(), shape.clone())),
                        );
                        self.static_lists = filter_static_lists;
                        self.bind_predicate_expression(expression)?;
                    }

                    // ORDER BY belongs to the WITH projection boundary. Its expressions may use
                    // projected aliases and, when planner validation permits it, bindings from the
                    // projection's input. Resolve both sets here, with aliases shadowing incoming
                    // names, and let the planner enforce DISTINCT/aggregation grouping rules.
                    if let Some(Clause::OrderBy(items)) = clauses.get(clause_index + 1) {
                        validate_order_by_aggregate_projection(items, projection)?;
                        let mut order_scope = projection_input_scope;
                        order_scope.extend(next.iter().map(|(name, kind)| (name.clone(), *kind)));
                        let mut order_static_sources = projection_input_static_sources;
                        order_static_sources.extend(
                            next_static_sources
                                .iter()
                                .map(|(name, shape)| (name.clone(), *shape)),
                        );
                        self.scope = order_scope;
                        self.static_sources = order_static_sources;
                        let mut order_static_lists = projection_input_static_lists;
                        order_static_lists.extend(
                            next_static_lists
                                .iter()
                                .map(|(name, shape)| (name.clone(), shape.clone())),
                        );
                        self.static_lists = order_static_lists;
                        for item in items {
                            self.bind_expression(&item.expression)?;
                        }
                    }
                    self.scope = next;
                    self.static_sources = next_static_sources;
                    self.static_lists = next_static_lists;
                }
                Clause::Return(projection) => {
                    self.bind_projection(projection, ProjectionBoundary::Return, false)?;
                    if self.capabilities.require_native_execution {
                        if clause_index == 2 {
                            let preceding_delete = clause_index
                                .checked_sub(1)
                                .and_then(|index| clauses.get(index));
                            let preceding_match = clause_index
                                .checked_sub(2)
                                .and_then(|index| clauses.get(index));
                            if let (
                                Some(Clause::Delete { expressions, .. }),
                                Some(Clause::Match {
                                    optional: false,
                                    patterns,
                                }),
                            ) = (preceding_delete, preceding_match)
                            {
                                self.validate_native_deleted_return_error_precedence(
                                    expressions,
                                    projection,
                                    patterns,
                                )?;
                            }
                        }
                        self.validate_native_percentile_range_error_precedence(
                            clauses, projection,
                        )?;
                    }
                    for item in &projection.items {
                        if let Some(alias) = &item.alias {
                            let kind = self.expression_binding_kind(&item.expression);
                            let shape = self.static_source_shape(&item.expression);
                            let list = self.static_list_shape(&item.expression);
                            self.scope.insert(alias.clone(), kind);
                            self.set_static_source_shape(alias, shape);
                            self.set_static_list_shape(alias, list);
                        }
                    }
                }
                Clause::OrderBy(items) => {
                    // An ORDER BY immediately following WITH was bound as part of that projection
                    // boundary, before its incoming scope was pruned.
                    if !matches!(
                        clause_index
                            .checked_sub(1)
                            .and_then(|index| clauses.get(index)),
                        Some(Clause::With(_))
                    ) {
                        for item in items {
                            self.bind_expression(&item.expression)?;
                        }
                    }
                }
                Clause::Skip(expression) => self.bind_row_count_expression(expression, "SKIP")?,
                Clause::Limit(expression) => self.bind_row_count_expression(expression, "LIMIT")?,
                Clause::History(history) => {
                    self.require_variable(&history.target.variable)?;
                    self.bind_expression(&history.from)?;
                    self.bind_expression(&history.to)?;
                    self.dependency(DependencyKind::Temporal, &history.target.property);
                    self.scope
                        .insert(history.variable.clone(), BindingKind::Value);
                    self.static_sources.remove(&history.variable);
                    self.static_lists.remove(&history.variable);
                }
                Clause::Window(window) => {
                    self.bind_expression(&window.width)?;
                    if let Some(every) = &window.every {
                        self.bind_expression(every)?;
                    }
                    self.bind_expression(&window.event_expression)?;
                    if let Some(align) = &window.align {
                        self.bind_expression(align)?;
                    }
                    self.scope
                        .insert(window.variable.clone(), BindingKind::Value);
                    self.static_sources.remove(&window.variable);
                    self.static_lists.remove(&window.variable);
                }
                Clause::Search(search) => {
                    if !self.scope.contains_key(&search.variable) {
                        let kind = match search.index.as_str() {
                            crate::graph::SEMANTIC_INDEX => BindingKind::Dynamic,
                            crate::graph::SEMANTIC_RELATIONSHIP_INDEX => BindingKind::Relationship,
                            _ => BindingKind::Node,
                        };
                        self.scope.insert(search.variable.clone(), kind);
                    }
                    match &search.input {
                        SearchInput::Text(value) | SearchInput::Vector(value) => {
                            self.bind_expression(value)?
                        }
                    }
                    self.bind_expression(&search.limit)?;
                    self.dependency(DependencyKind::Index, &search.index);
                    self.scope
                        .insert(search.score_variable.clone(), BindingKind::Value);
                    self.static_sources.remove(&search.score_variable);
                    self.static_lists.remove(&search.score_variable);
                }
                Clause::Call(call) => {
                    let resolved = validate_call(call, self.procedures, self.parameters)?;
                    if matches!(&resolved, ResolvedProcedure::Builtin(_)) {
                        self.dependency(DependencyKind::FullNodeScan, "*");
                        self.dependency(DependencyKind::FullRelationshipScan, "*");
                    }
                    for argument in &call.arguments {
                        self.bind_expression(argument)?;
                    }
                    for item in &call.yields {
                        if let Expression::Variable(variable) = &item.expression {
                            let alias = match &item.alias {
                                Some(alias) => alias,
                                None => variable,
                            };
                            if self.scope.contains_key(alias) {
                                return Err(Error::new(
                                    ErrorCode::QuerySyntax,
                                    format!(
                                        "VariableAlreadyBound: procedure YIELD alias `{alias}` is already bound"
                                    ),
                                ));
                            }
                            self.scope.insert(
                                alias.clone(),
                                procedure_output_binding_kind(&resolved, variable),
                            );
                            self.static_sources.remove(alias);
                            self.static_lists.remove(alias);
                        }
                    }
                }
                Clause::Finish => {
                    if clause_index + 1 != clauses.len() {
                        return Err(Error::new(
                            ErrorCode::QueryType,
                            "FINISH must be the final clause of a query branch",
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    /// Bind an existential pattern predicate without extending the surrounding scope.
    ///
    /// MATCH patterns declare missing variables; pattern predicates do not. Every named node or
    /// relationship in the predicate must already be bound with the matching graph role. Anonymous
    /// elements remain legal, and property expressions are resolved against the outer scope.
    fn bind_pattern_predicate(&mut self, pattern: &Pattern) -> Result<()> {
        if pattern.variable.is_some() {
            return Err(Error::new(
                ErrorCode::QuerySyntax,
                "UnexpectedSyntax: a named path declaration is not legal in a pattern predicate",
            ));
        }
        if pattern.steps.is_empty() {
            return Err(Error::new(
                ErrorCode::QuerySyntax,
                "InvalidArgumentType: a node value alone is not a pattern predicate",
            ));
        }
        self.dependency(DependencyKind::FullNodeScan, "*");
        self.dependency(DependencyKind::FullRelationshipScan, "*");
        self.bind_pattern_predicate_node(&pattern.start)?;
        for step in &pattern.steps {
            let required = if step.relationship.variable_length {
                BindingKind::RelationshipList
            } else {
                BindingKind::Relationship
            };
            if let Some(variable) = &step.relationship.variable {
                self.require_pattern_predicate_variable(variable, required)?;
            }
            for relationship_type in &step.relationship.types {
                self.dependency(DependencyKind::RelationshipType, relationship_type);
            }
            for (property, expression) in &step.relationship.properties {
                self.dependency(DependencyKind::Property, property);
                self.bind_expression(expression)?;
            }
            self.bind_pattern_predicate_node(&step.node)?;
        }
        Ok(())
    }

    /// Bind a pattern comprehension in a temporary lexical graph scope.
    ///
    /// Existing node/relationship variables correlate the pattern with the outer row. Missing
    /// graph variables and an optional named path are introduced only while binding the optional
    /// filter and required projection. Restoring both binding maps afterwards is what prevents a
    /// local `b`, `r`, or `p` from leaking into a sibling projection or later clause.
    fn bind_pattern_comprehension(
        &mut self,
        pattern: &Pattern,
        predicate: Option<&Expression>,
        projection: &Expression,
    ) -> Result<()> {
        if pattern.steps.is_empty() {
            return Err(Error::new(
                ErrorCode::QuerySyntax,
                "UnexpectedSyntax: a pattern comprehension requires a relationship pattern",
            ));
        }

        let outer_scope = self.scope.clone();
        let outer_static_sources = self.static_sources.clone();
        let outer_static_lists = self.static_lists.clone();
        let result = (|| {
            self.dependency(DependencyKind::FullNodeScan, "*");
            self.bind_pattern(pattern, PatternBindingMode::Match)?;
            if let Some(predicate) = predicate {
                reject_aggregate_in_iteration_scope(predicate, "pattern comprehension filter")?;
                self.bind_predicate_expression(predicate)?;
            }
            reject_aggregate_in_iteration_scope(projection, "pattern comprehension projection")?;
            self.bind_expression(projection)
        })();
        self.scope = outer_scope;
        self.static_sources = outer_static_sources;
        self.static_lists = outer_static_lists;
        result
    }

    fn bind_pattern_predicate_node(&mut self, node: &NodePattern) -> Result<()> {
        if let Some(variable) = &node.variable {
            self.require_pattern_predicate_variable(variable, BindingKind::Node)?;
        }
        for label in &node.labels {
            self.dependency(DependencyKind::Label, label);
            let _ = self.catalog.label(label);
        }
        for (property, expression) in &node.properties {
            self.dependency(DependencyKind::Property, property);
            self.bind_expression(expression)?;
        }
        Ok(())
    }

    fn require_pattern_predicate_variable(
        &self,
        variable: &str,
        required: BindingKind,
    ) -> Result<()> {
        match self.scope.get(variable).copied() {
            None => Err(Error::new(
                ErrorCode::QuerySyntax,
                format!("UndefinedVariable: variable `{variable}` is not defined"),
            )),
            Some(existing) if existing == required => Ok(()),
            Some(existing) => Err(Error::new(
                ErrorCode::QuerySyntax,
                format!(
                    "VariableTypeConflict: variable `{variable}` is bound as {} and cannot be used as {} in a pattern predicate",
                    binding_kind_name(existing),
                    binding_kind_name(required),
                ),
            )),
        }
    }

    fn bind_pattern(&mut self, pattern: &Pattern, mode: PatternBindingMode) -> Result<()> {
        if mode.is_write()
            && (pattern.selector != PathSelector::All
                || pattern.mode != PathMode::DifferentRelationships)
        {
            return Err(Error::new(
                ErrorCode::QueryType,
                "path selectors and match modes are legal only in MATCH",
            ));
        }
        let mut owner = PatternBindingOwner::new(&self.scope, pattern);
        self.bind_node_pattern(&pattern.start, mode, &owner)?;
        for step in &pattern.steps {
            if mode.is_write() {
                self.validate_write_relationship(&step.relationship, mode)?;
            }
            self.dependency(DependencyKind::FullRelationshipScan, "*");
            for relationship_type in &step.relationship.types {
                self.dependency(DependencyKind::RelationshipType, relationship_type);
            }
            if let Some(variable) = &step.relationship.variable {
                if mode.is_write() {
                    self.scope
                        .insert(variable.clone(), BindingKind::Relationship);
                    self.static_sources.remove(variable);
                    self.static_lists.remove(variable);
                } else {
                    if !owner.match_relationships.insert(variable.clone()) {
                        return Err(Error::new(
                            ErrorCode::QuerySyntax,
                            format!(
                                "RelationshipUniquenessViolation: relationship variable `{variable}` is used more than once in the same pattern"
                            ),
                        ));
                    }
                    let required = if step.relationship.variable_length {
                        BindingKind::RelationshipList
                    } else {
                        BindingKind::Relationship
                    };
                    self.bind_reusable_pattern_variable(variable, required, true)?;
                }
            }
            for (_, expression) in &step.relationship.properties {
                self.bind_expression(expression)?;
            }
            self.bind_node_pattern(&step.node, mode, &owner)?;
        }
        if let Some(variable) = &pattern.variable {
            self.bind_path_variable(variable)?;
        }
        Ok(())
    }

    fn validate_write_relationship(
        &self,
        relationship: &RelationshipPattern,
        mode: PatternBindingMode,
    ) -> Result<()> {
        if let Some(variable) = &relationship.variable
            && self.scope.contains_key(variable)
        {
            return Err(Error::new(
                ErrorCode::QuerySyntax,
                format!(
                    "VariableAlreadyBound: relationship variable `{variable}` is already bound"
                ),
            ));
        }
        if mode == PatternBindingMode::Create && relationship.direction == Direction::Undirected {
            return Err(Error::new(
                ErrorCode::QuerySyntax,
                "RequiresDirectedRelationship: CREATE relationships require exactly one direction",
            ));
        }
        if relationship.types.len() != 1 {
            return Err(Error::new(
                ErrorCode::QuerySyntax,
                "NoSingleRelationshipType: write relationships require exactly one type",
            ));
        }
        if relationship.variable_length {
            return Err(Error::new(
                ErrorCode::QuerySyntax,
                "CreatingVarLength: write relationships must have a fixed length of one",
            ));
        }
        Ok(())
    }

    fn bind_node_pattern(
        &mut self,
        node: &NodePattern,
        mode: PatternBindingMode,
        owner: &PatternBindingOwner,
    ) -> Result<()> {
        if let Some(variable) = &node.variable {
            self.bind_reusable_pattern_variable(variable, BindingKind::Node, !mode.is_write())?;
            if mode.is_write()
                && owner.entry_scope.get(variable) == Some(&BindingKind::Node)
                && (!owner.has_relationship
                    || !node.labels.is_empty()
                    || node.property_predicate_present)
            {
                return Err(Error::new(
                    ErrorCode::QuerySyntax,
                    format!(
                        "VariableAlreadyBound: node variable `{variable}` was bound before this write pattern and cannot be redeclared or constrained"
                    ),
                ));
            }
        }
        for label in &node.labels {
            self.dependency(DependencyKind::Label, label);
            if !mode.is_write() {
                let _ = self.catalog.label(label);
            }
        }
        for (property, expression) in &node.properties {
            self.dependency(DependencyKind::Property, property);
            self.bind_expression(expression)?;
        }
        Ok(())
    }

    /// Nodes and relationships may be constrained again by later patterns, but their graph role
    /// is fixed for the lifetime of the binding. This check runs while each pattern is walked, so
    /// it also covers conflicts inside one pattern and across comma-separated patterns.
    fn bind_reusable_pattern_variable(
        &mut self,
        variable: &str,
        required: BindingKind,
        allow_null_correlation: bool,
    ) -> Result<()> {
        match self.scope.get(variable).copied() {
            None => {
                self.scope.insert(variable.to_owned(), required);
                self.static_sources.remove(variable);
                self.static_lists.remove(variable);
                Ok(())
            }
            Some(existing) if existing == required => {
                self.static_sources.remove(variable);
                self.static_lists.remove(variable);
                Ok(())
            }
            // NULL is compatible with every nullable graph role. A read pattern may therefore
            // correlate a statically-null value as a node/relationship; this is what lets an
            // OPTIONAL MATCH retain the row and produce a null path instead of failing binding.
            Some(BindingKind::Value)
                if allow_null_correlation
                    && self.static_sources.get(variable) == Some(&StaticSourceShape::Null) =>
            {
                self.scope.insert(variable.to_owned(), required);
                self.static_sources.remove(variable);
                self.static_lists.remove(variable);
                Ok(())
            }
            // An element from a dynamically typed list can be a graph entity, but that fact is
            // knowable only for the current runtime row. Once a read pattern claims the role, the
            // ordinary role checks below prevent that same binding from changing role again.
            Some(BindingKind::Dynamic) if allow_null_correlation => {
                self.scope.insert(variable.to_owned(), required);
                self.static_sources.remove(variable);
                self.static_lists.remove(variable);
                Ok(())
            }
            Some(existing) => Err(Error::new(
                ErrorCode::QuerySyntax,
                format!(
                    "VariableTypeConflict: variable `{variable}` is bound as {} and cannot be used as {}",
                    binding_kind_name(existing),
                    binding_kind_name(required)
                ),
            )),
        }
    }

    /// A named path declares a new value for the whole pattern. Unlike node and relationship
    /// variables, an existing binding cannot be reused as the path declaration target.
    fn bind_path_variable(&mut self, variable: &str) -> Result<()> {
        if self.scope.contains_key(variable) {
            return Err(Error::new(
                ErrorCode::QuerySyntax,
                format!("VariableAlreadyBound: path variable `{variable}` is already bound"),
            ));
        }
        self.scope.insert(variable.to_owned(), BindingKind::Path);
        self.static_sources.remove(variable);
        self.static_lists.remove(variable);
        Ok(())
    }

    fn bind_set(&mut self, item: &SetItem) -> Result<()> {
        match item {
            SetItem::Property {
                target,
                value,
                event_time,
            } => {
                self.require_variable(&target.variable)?;
                self.dependency(DependencyKind::Property, &target.property);
                self.bind_expression(value)?;
                if self.capabilities.require_native_execution {
                    self.validate_native_property_value_type_precedence(value)?;
                }
                if let Some(event_time) = event_time {
                    self.bind_expression(event_time)?;
                }
            }
            SetItem::MergeMap { variable, value } | SetItem::ReplaceMap { variable, value } => {
                self.require_variable(variable)?;
                self.bind_expression(value)?;
            }
            SetItem::Labels { variable, labels } => {
                self.require_variable(variable)?;
                for label in labels {
                    self.bind_label_name(label)?;
                }
            }
        }
        Ok(())
    }

    fn bind_label_name(&mut self, label: &LabelName) -> Result<()> {
        match label {
            LabelName::Static(name) => self.dependency(DependencyKind::Label, name),
            LabelName::Dynamic(expression) | LabelName::DynamicAll(expression) => {
                self.bind_expression(expression)?;
                self.dependency(DependencyKind::Label, "*");
            }
        }
        Ok(())
    }

    fn bind_projection(
        &mut self,
        projection: &Projection,
        boundary: ProjectionBoundary,
        defer_expression_alias_validation: bool,
    ) -> Result<()> {
        if projection.items.is_empty() {
            return Err(Error::new(ErrorCode::QueryType, "projection is empty"));
        }

        if boundary == ProjectionBoundary::Return
            && self.scope.is_empty()
            && projection
                .items
                .iter()
                .any(|item| matches!(item.expression, Expression::Star))
        {
            return Err(Error::new(
                ErrorCode::QuerySyntax,
                "NoVariablesInScope: RETURN * requires at least one variable in scope",
            ));
        }

        let mut column_names = BTreeSet::new();
        for (index, item) in projection.items.iter().enumerate() {
            if boundary == ProjectionBoundary::With
                && !defer_expression_alias_validation
                && item.alias.is_none()
                && !matches!(item.expression, Expression::Variable(_) | Expression::Star)
            {
                return Err(Error::new(
                    ErrorCode::QuerySyntax,
                    format!(
                        "NoExpressionAlias: WITH expression `{}` must be explicitly aliased with AS",
                        item.column_name(index)
                    ),
                ));
            }

            if matches!(item.expression, Expression::Star) {
                for name in self.scope.keys() {
                    if !column_names.insert(name.clone()) {
                        return Err(duplicate_projection_column(name));
                    }
                }
            } else {
                let name = item.column_name(index);
                if !column_names.insert(name.clone()) {
                    return Err(duplicate_projection_column(&name));
                }
            }

            validate_aggregate_nesting_as_syntax(&item.expression)?;
            self.bind_expression(&item.expression)?;
        }
        validate_projection_aggregation(
            projection.items.iter().map(|item| &item.expression),
            self.scope.keys().map(String::as_str),
        )?;
        Ok(())
    }

    fn bind_row_count_expression(&mut self, expression: &Expression, clause: &str) -> Result<()> {
        if !row_count_expression_is_independent(expression, &BTreeSet::new()) {
            return Err(Error::new(
                ErrorCode::QuerySyntax,
                format!(
                    "NonConstantExpression: {clause} must not depend on row variables or aggregation"
                ),
            ));
        }
        self.bind_expression(expression)?;
        match classify_row_count_expression(expression) {
            RowCountClassification::KnownInteger(value) if value < 0 => Err(Error::new(
                ErrorCode::QuerySyntax,
                format!("NegativeIntegerArgument: {clause} must be non-negative"),
            )),
            RowCountClassification::NonInteger => Err(Error::new(
                ErrorCode::QuerySyntax,
                format!("InvalidArgumentType: {clause} requires an INTEGER"),
            )),
            RowCountClassification::KnownInteger(_)
            | RowCountClassification::Integer
            | RowCountClassification::Unknown => Ok(()),
        }
    }

    fn bind_predicate_expression(&mut self, expression: &Expression) -> Result<()> {
        reject_aggregate_in_row_context(expression, "predicate expressions")?;
        self.bind_expression(expression)?;
        if match expression {
            Expression::Variable(variable) => self.scope.get(variable).is_some_and(|kind| {
                !matches!(
                    *kind,
                    BindingKind::Value | BindingKind::Dynamic | BindingKind::NonEntity
                )
            }),
            _ => statically_non_boolean(expression),
        } {
            return Err(Error::new(
                ErrorCode::QuerySyntax,
                "InvalidArgumentType: predicate expression must produce BOOLEAN or NULL",
            ));
        }
        Ok(())
    }

    fn bind_expression(&mut self, expression: &Expression) -> Result<()> {
        if let Some(subquery) = expression.existential_subquery_parts() {
            return self.bind_existential_subquery(subquery);
        }
        if let Some((pattern, predicate, projection)) = expression.pattern_comprehension_parts() {
            return self.bind_pattern_comprehension(&pattern, predicate, projection);
        }
        if let Some(pattern) = expression.pattern_predicate_pattern() {
            return self.bind_pattern_predicate(&pattern);
        }
        if let Some((source, names)) = expression.entity_label_predicate_parts() {
            self.bind_expression(source)?;
            for name in names {
                let Expression::Literal(ScalarValue::String(name)) = name else {
                    return Err(Error::internal(
                        "entity label predicate contains a non-literal name",
                    ));
                };
                self.dependency(DependencyKind::Label, name);
                self.dependency(DependencyKind::RelationshipType, name);
            }
            return Ok(());
        }
        match expression {
            Expression::Literal(_)
            | Expression::Parameter(_)
            | Expression::ExistentialSubquery(_)
            | Expression::Star => {}
            Expression::Variable(variable) => self.require_variable(variable)?,
            Expression::Property(value, property) => {
                self.bind_expression(value)?;
                let source_shape = self.static_source_shape(value);
                if source_shape == StaticSourceShape::Path {
                    return Err(Error::new(
                        ErrorCode::QuerySyntax,
                        "InvalidArgumentType: property access is not defined for PATH values",
                    ));
                }
                if source_shape.is_definitely_invalid_property_source() {
                    return Err(Error::new(
                        ErrorCode::QueryType,
                        "InvalidArgumentType: property access requires a MAP, NODE, RELATIONSHIP, temporal value, or NULL",
                    ));
                }
                self.dependency(DependencyKind::Property, property);
            }
            Expression::List(values) => {
                for value in values {
                    self.bind_expression(value)?;
                }
            }
            Expression::Map(values) => {
                for (_, value) in values {
                    self.bind_expression(value)?;
                }
            }
            Expression::MapProjection { source, items } => {
                self.bind_expression(source)?;
                for item in items {
                    match item {
                        MapProjectionItem::AllProperties | MapProjectionItem::Property(_) => {}
                        MapProjectionItem::Variable(variable) => {
                            self.require_variable(variable)?;
                        }
                        MapProjectionItem::Entry(_, expression) => {
                            self.bind_expression(expression)?;
                        }
                    }
                }
            }
            Expression::Case {
                operand,
                alternatives,
                default,
            } => {
                if let Some(operand) = operand {
                    self.bind_expression(operand)?;
                }
                for alternative in alternatives {
                    self.bind_expression(&alternative.when)?;
                    self.bind_expression(&alternative.then)?;
                }
                if let Some(default) = default {
                    self.bind_expression(default)?;
                }
            }
            Expression::ListComprehension {
                variable,
                list,
                predicate,
                projection,
            } => {
                self.bind_expression(list)?;
                if self.capabilities.require_native_execution && predicate.is_none() {
                    self.validate_native_list_projection_type_precedence(
                        variable,
                        list,
                        projection.as_deref(),
                    )?;
                }
                let element_kind = self.list_element_binding_kind(list);
                let prior = self.scope.insert(variable.clone(), element_kind);
                let static_prior = self.static_sources.remove(variable);
                let static_list_prior = self.static_lists.remove(variable);
                if let Some(predicate) = predicate {
                    reject_aggregate_in_iteration_scope(predicate, "list comprehension predicate")?;
                    self.bind_expression(predicate)?;
                }
                if let Some(projection) = projection {
                    reject_aggregate_in_iteration_scope(
                        projection,
                        "list comprehension projection",
                    )?;
                    self.bind_expression(projection)?;
                }
                restore_binding(&mut self.scope, variable, prior);
                restore_static_source_shape(&mut self.static_sources, variable, static_prior);
                restore_static_list_shape(&mut self.static_lists, variable, static_list_prior);
            }
            Expression::Reduce {
                accumulator,
                initial,
                variable,
                list,
                expression,
            } => {
                self.bind_expression(initial)?;
                self.bind_expression(list)?;
                let accumulator_prior = self.scope.insert(accumulator.clone(), BindingKind::Value);
                let variable_prior = (variable != accumulator)
                    .then(|| self.scope.insert(variable.clone(), BindingKind::Value));
                let accumulator_static_prior = self.static_sources.remove(accumulator);
                let variable_static_prior =
                    (variable != accumulator).then(|| self.static_sources.remove(variable));
                let accumulator_static_list_prior = self.static_lists.remove(accumulator);
                let variable_static_list_prior =
                    (variable != accumulator).then(|| self.static_lists.remove(variable));
                reject_aggregate_in_iteration_scope(expression, "reduce expression")?;
                self.bind_expression(expression)?;
                if let Some(prior) = variable_prior {
                    restore_binding(&mut self.scope, variable, prior);
                }
                if let Some(prior) = variable_static_prior {
                    restore_static_source_shape(&mut self.static_sources, variable, prior);
                }
                if let Some(prior) = variable_static_list_prior {
                    restore_static_list_shape(&mut self.static_lists, variable, prior);
                }
                restore_binding(&mut self.scope, accumulator, accumulator_prior);
                restore_static_source_shape(
                    &mut self.static_sources,
                    accumulator,
                    accumulator_static_prior,
                );
                restore_static_list_shape(
                    &mut self.static_lists,
                    accumulator,
                    accumulator_static_list_prior,
                );
            }
            Expression::ListPredicate {
                variable,
                list,
                predicate,
                ..
            } => {
                self.bind_expression(list)?;
                let element_kind = self.list_element_binding_kind(list);
                let prior = self.scope.insert(variable.clone(), element_kind);
                let static_prior = self.static_sources.remove(variable);
                let static_list_prior = self.static_lists.remove(variable);
                reject_aggregate_in_iteration_scope(predicate, "list predicate")?;
                self.bind_expression(predicate)?;
                validate_static_list_predicate_types(variable, list, predicate)?;
                restore_binding(&mut self.scope, variable, prior);
                restore_static_source_shape(&mut self.static_sources, variable, static_prior);
                restore_static_list_shape(&mut self.static_lists, variable, static_list_prior);
            }
            Expression::Function {
                name, arguments, ..
            } => {
                let name = name.join(".").to_ascii_lowercase();
                if !is_supported_builtin_function(&name) {
                    return Err(Error::new(
                        ErrorCode::QuerySyntax,
                        format!("UnknownFunction: function `{name}` is not defined"),
                    ));
                }
                for argument in arguments {
                    self.bind_expression(argument)?;
                }
                validate_temporal_clock_call(&name, arguments)?;
                if self.capabilities.require_native_execution {
                    self.validate_native_indexed_function_type_precedence(&name, arguments)?;
                }
                self.validate_static_function_argument(&name, arguments)?;
            }
            Expression::Unary { operation, operand } => {
                self.bind_expression(operand)?;
                if *operation == UnaryOperator::Not && statically_non_boolean(operand) {
                    return Err(invalid_boolean_expression());
                }
            }
            Expression::Binary {
                left,
                operation,
                right,
            } => {
                self.bind_expression(left)?;
                self.bind_expression(right)?;
                if matches!(
                    operation,
                    BinaryOperator::And | BinaryOperator::Or | BinaryOperator::Xor
                ) && (statically_non_boolean(left) || statically_non_boolean(right))
                {
                    return Err(invalid_boolean_expression());
                }
                if *operation == BinaryOperator::In && statically_non_list(right) {
                    return Err(Error::new(
                        ErrorCode::QuerySyntax,
                        "InvalidArgumentType: IN right operand must be a LIST or NULL",
                    ));
                }
            }
            Expression::IsNull { expression, .. } => self.bind_expression(expression)?,
            Expression::Index { expression, index } => {
                self.bind_expression(expression)?;
                self.bind_expression(index)?;
            }
            Expression::Slice {
                expression,
                start,
                end,
            } => {
                self.bind_expression(expression)?;
                if let Some(start) = start {
                    self.bind_expression(start)?;
                }
                if let Some(end) = end {
                    self.bind_expression(end)?;
                }
            }
        }
        Ok(())
    }

    /// Bind a correlated read-only subquery in a lexical child scope.
    ///
    /// Outer graph/value bindings seed every branch. Bindings introduced by MATCH, WITH, or
    /// RETURN remain local, while dependency stamps discovered inside the body intentionally stay
    /// attached to the enclosing statement.
    fn bind_existential_subquery(&mut self, subquery: &ExistentialSubquery) -> Result<()> {
        stacker::maybe_grow(256 * 1024, 2 * 1024 * 1024, || {
            self.bind_existential_subquery_inner(subquery)
        })
    }

    fn bind_existential_subquery_inner(&mut self, subquery: &ExistentialSubquery) -> Result<()> {
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

        let outer_scope = self.scope.clone();
        let outer_static_sources = self.static_sources.clone();
        let outer_static_lists = self.static_lists.clone();
        let outer_read_only = self.read_only;
        let result = (|| {
            self.bind_clauses(&subquery.body.clauses)?;
            for branch in &subquery.body.unions {
                self.scope = outer_scope.clone();
                self.static_sources = outer_static_sources.clone();
                self.static_lists = outer_static_lists.clone();
                self.bind_clauses(&branch.body)?;
            }
            Ok(())
        })();
        self.scope = outer_scope;
        self.static_sources = outer_static_sources;
        self.static_lists = outer_static_lists;
        self.read_only = outer_read_only;
        result
    }

    fn require_variable(&self, variable: &str) -> Result<()> {
        if self.scope.contains_key(variable) {
            Ok(())
        } else {
            Err(Error::new(
                ErrorCode::QuerySyntax,
                format!("UndefinedVariable: variable `{variable}` is not defined"),
            ))
        }
    }

    /// Projection aliases preserve graph-entity roles and conservative list-element provenance.
    /// This is what lets `collect(node)` cross a WITH boundary without turning its later UNWIND
    /// element into an untyped scalar. Scalar lists and node/relationship lists stay distinct.
    fn expression_binding_kind(&self, expression: &Expression) -> BindingKind {
        match expression {
            Expression::Variable(variable) => self
                .scope
                .get(variable)
                .map_or(BindingKind::Value, |kind| *kind),
            Expression::Case {
                alternatives,
                default,
                ..
            } => {
                let mut kinds = alternatives
                    .iter()
                    .map(|alternative| self.expression_binding_kind(&alternative.then))
                    .chain(
                        default
                            .iter()
                            .map(|value| self.expression_binding_kind(value)),
                    );
                let Some(first) = kinds.next() else {
                    return BindingKind::Value;
                };
                if kinds.all(|kind| kind == first) {
                    first
                } else {
                    BindingKind::Value
                }
            }
            Expression::Function {
                name, arguments, ..
            } if name.join(".").eq_ignore_ascii_case("coalesce") => {
                let mut kinds = arguments
                    .iter()
                    .map(|argument| self.expression_binding_kind(argument));
                let Some(first) = kinds.next() else {
                    return BindingKind::Value;
                };
                if kinds.all(|kind| kind == first) {
                    first
                } else {
                    BindingKind::Value
                }
            }
            Expression::Function { name, .. }
                if matches!(
                    name.join(".").to_ascii_lowercase().as_str(),
                    "startnode" | "endnode"
                ) =>
            {
                BindingKind::Node
            }
            Expression::Function {
                name, arguments, ..
            } => match name.join(".").to_ascii_lowercase().as_str() {
                "nodes" => BindingKind::NodeList,
                "relationships" => BindingKind::RelationshipList,
                "collect" => arguments
                    .first()
                    .map_or(BindingKind::ScalarList, |argument| {
                        self.list_binding_kind_for_element(argument)
                    }),
                "range" | "keys" | "labels" | "split" | "tolist" => BindingKind::ScalarList,
                "tail" => arguments.first().map_or(BindingKind::Value, |argument| {
                    match self.expression_binding_kind(argument) {
                        BindingKind::NodeList => BindingKind::NodeList,
                        BindingKind::RelationshipList => BindingKind::RelationshipList,
                        BindingKind::ScalarList => BindingKind::ScalarList,
                        _ => BindingKind::Value,
                    }
                }),
                _ => BindingKind::Value,
            },
            Expression::List(items) => self.literal_list_binding_kind(items),
            Expression::Binary {
                left,
                operation,
                right,
            } if matches!(operation, BinaryOperator::Add | BinaryOperator::Concat) => {
                self.list_combination_binding_kind(left, *operation, right)
            }
            Expression::Index { expression, .. } => {
                match self.expression_binding_kind(expression) {
                    BindingKind::NodeList => BindingKind::Node,
                    BindingKind::RelationshipList => BindingKind::Relationship,
                    BindingKind::ScalarList => BindingKind::NonEntity,
                    _ => BindingKind::Dynamic,
                }
            }
            Expression::Slice { expression, .. } => {
                match self.expression_binding_kind(expression) {
                    BindingKind::NodeList => BindingKind::NodeList,
                    BindingKind::RelationshipList => BindingKind::RelationshipList,
                    BindingKind::ScalarList => BindingKind::ScalarList,
                    _ => BindingKind::Value,
                }
            }
            _ => BindingKind::Value,
        }
    }

    /// Preserve a list's element role only when the Cypher list operation proves every produced
    /// element has that same role. `+` may append or prepend one entity; `||` requires two lists.
    /// Mixed and unknown inputs deliberately collapse to `Value`, so indexing them becomes
    /// `Dynamic` and write-pattern validation continues to reject an unproven graph role.
    fn list_combination_binding_kind(
        &self,
        left: &Expression,
        operation: BinaryOperator,
        right: &Expression,
    ) -> BindingKind {
        let left_kind = self.expression_binding_kind(left);
        let right_kind = self.expression_binding_kind(right);

        match (left_kind, right_kind) {
            (BindingKind::NodeList, BindingKind::NodeList) => BindingKind::NodeList,
            (BindingKind::RelationshipList, BindingKind::RelationshipList) => {
                BindingKind::RelationshipList
            }
            (BindingKind::ScalarList, BindingKind::ScalarList) => BindingKind::ScalarList,
            (BindingKind::NodeList, BindingKind::Node)
            | (BindingKind::Node, BindingKind::NodeList)
                if operation == BinaryOperator::Add =>
            {
                BindingKind::NodeList
            }
            (BindingKind::RelationshipList, BindingKind::Relationship)
            | (BindingKind::Relationship, BindingKind::RelationshipList)
                if operation == BinaryOperator::Add =>
            {
                BindingKind::RelationshipList
            }
            _ if expression_is_empty_list(left) => match right_kind {
                BindingKind::NodeList | BindingKind::RelationshipList | BindingKind::ScalarList => {
                    right_kind
                }
                BindingKind::Node if operation == BinaryOperator::Add => BindingKind::NodeList,
                BindingKind::Relationship if operation == BinaryOperator::Add => {
                    BindingKind::RelationshipList
                }
                _ => BindingKind::Value,
            },
            _ if expression_is_empty_list(right) => match left_kind {
                BindingKind::NodeList | BindingKind::RelationshipList | BindingKind::ScalarList => {
                    left_kind
                }
                BindingKind::Node if operation == BinaryOperator::Add => BindingKind::NodeList,
                BindingKind::Relationship if operation == BinaryOperator::Add => {
                    BindingKind::RelationshipList
                }
                _ => BindingKind::Value,
            },
            _ => BindingKind::Value,
        }
    }

    /// Infer graph roles for values iterated from a list-producing expression.
    ///
    /// Known entity and scalar lists are exact. Unknown producers become `Dynamic`, allowing a
    /// read pattern to perform its normal runtime role check instead of rejecting a potentially
    /// legal query during binding.
    fn list_element_binding_kind(&self, expression: &Expression) -> BindingKind {
        match self.expression_binding_kind(expression) {
            BindingKind::NodeList => BindingKind::Node,
            BindingKind::RelationshipList => BindingKind::Relationship,
            BindingKind::ScalarList => BindingKind::NonEntity,
            _ => BindingKind::Dynamic,
        }
    }

    fn literal_list_binding_kind(&self, items: &[Expression]) -> BindingKind {
        let mut kinds = items
            .iter()
            .map(|item| self.list_binding_kind_for_element(item));
        let Some(first) = kinds.next() else {
            // An empty list produces no element whose role could be observed.
            return BindingKind::Value;
        };
        if kinds.all(|kind| kind == first) {
            first
        } else {
            // Heterogeneous or dynamically typed element lists are checked one row at runtime.
            BindingKind::Value
        }
    }

    fn list_binding_kind_for_element(&self, expression: &Expression) -> BindingKind {
        match self.expression_binding_kind(expression) {
            BindingKind::Node => BindingKind::NodeList,
            BindingKind::Relationship => BindingKind::RelationshipList,
            BindingKind::Dynamic => BindingKind::Value,
            BindingKind::NonEntity => BindingKind::ScalarList,
            BindingKind::Value
                if self.static_source_shape(expression) == StaticSourceShape::Unknown
                    && !matches!(
                        expression,
                        Expression::Property(_, _)
                            | Expression::Unary { .. }
                            | Expression::Binary { .. }
                            | Expression::IsNull { .. }
                            | Expression::ListPredicate { .. }
                            | Expression::MapProjection { .. }
                            | Expression::Slice { .. }
                    ) =>
            {
                // Parameters, indexed values, mixed CASE/coalesce expressions, and other truly
                // dynamic sources can still carry a graph entity supplied by the current row.
                BindingKind::Value
            }
            _ => BindingKind::ScalarList,
        }
    }

    fn static_source_shape(&self, expression: &Expression) -> StaticSourceShape {
        if expression.pattern_comprehension_parts().is_some() {
            return StaticSourceShape::List;
        }
        if expression.existential_subquery_parts().is_some() {
            return StaticSourceShape::Scalar;
        }
        match expression {
            Expression::Literal(value) => static_literal_source_shape(value),
            Expression::Parameter(_) | Expression::Star => StaticSourceShape::Unknown,
            Expression::Variable(variable) => match self.scope.get(variable) {
                Some(BindingKind::Node) => StaticSourceShape::Node,
                Some(BindingKind::Relationship) => StaticSourceShape::Relationship,
                Some(
                    BindingKind::NodeList | BindingKind::RelationshipList | BindingKind::ScalarList,
                ) => StaticSourceShape::List,
                Some(BindingKind::Path) => StaticSourceShape::Path,
                Some(BindingKind::Dynamic) => StaticSourceShape::Unknown,
                Some(BindingKind::NonEntity) => StaticSourceShape::Unknown,
                Some(BindingKind::Value) => self
                    .static_sources
                    .get(variable)
                    .copied()
                    .map_or(StaticSourceShape::Unknown, |shape| shape),
                None => StaticSourceShape::Unknown,
            },
            Expression::List(_) => StaticSourceShape::List,
            Expression::Map(_) | Expression::MapProjection { .. } => StaticSourceShape::Map,
            Expression::Case {
                alternatives,
                default,
                ..
            } => match default {
                Some(default) => uniform_static_source_shape(
                    alternatives
                        .iter()
                        .map(|alternative| self.static_source_shape(&alternative.then))
                        .chain(std::iter::once(self.static_source_shape(default))),
                ),
                None => StaticSourceShape::Unknown,
            },
            Expression::Function {
                name, arguments, ..
            } if name.join(".").eq_ignore_ascii_case("coalesce") => uniform_static_source_shape(
                arguments
                    .iter()
                    .map(|argument| self.static_source_shape(argument)),
            ),
            Expression::Function { name, .. } => static_function_source_shape(name),
            Expression::Unary { operand, .. } => match self.static_source_shape(operand) {
                StaticSourceShape::Null => StaticSourceShape::Null,
                StaticSourceShape::Scalar => StaticSourceShape::Scalar,
                _ => StaticSourceShape::Unknown,
            },
            Expression::IsNull { .. } => StaticSourceShape::Scalar,
            Expression::ExistentialSubquery(_) => StaticSourceShape::Scalar,
            Expression::Property(_, _)
            | Expression::ListComprehension { .. }
            | Expression::Reduce { .. }
            | Expression::ListPredicate { .. }
            | Expression::Binary { .. }
            | Expression::Index { .. }
            | Expression::Slice { .. } => StaticSourceShape::Unknown,
        }
    }

    /// DELETE accepts graph entities and paths, while NULL is a legal no-op. Property lookup,
    /// indexing, parameters, and mixed expressions can still resolve to an entity at runtime, so
    /// they remain dynamic. Binary operators, however, can produce only scalar/container,
    /// temporal, or NULL results and can never produce a node, relationship, or path.
    fn validate_delete_target(&self, expression: &Expression) -> Result<()> {
        if self
            .static_source_shape(expression)
            .is_definitely_invalid_delete_target()
            || matches!(expression, Expression::Binary { .. })
        {
            return Err(Error::new(
                ErrorCode::QuerySyntax,
                "InvalidArgumentType: DELETE expression must resolve to a node, relationship, or path",
            ));
        }
        Ok(())
    }

    /// Preserve the executor's deleted-entity error ahead of strict native-plan admission when
    /// the failing access is certain from a branch-local mandatory MATCH/DELETE/RETURN sequence.
    /// Provenance is deliberately local: the MATCH must begin the query branch, only
    /// direct node/relationship variables from that MATCH and DELETE participate, and only direct
    /// property reads or `labels(node)` are recognized. Optional matches, parameters, aliases
    /// crossing WITH, nested/guarded expressions, and all other runtime-dependent forms remain
    /// deferred. In particular, `type(deleted_relationship)` is legal and stays readable.
    fn validate_native_deleted_return_error_precedence(
        &self,
        delete_expressions: &[Expression],
        projection: &Projection,
        patterns: &[Pattern],
    ) -> Result<()> {
        let mut matched_entities = BTreeSet::new();
        for pattern in patterns {
            if let Some(variable) = &pattern.start.variable {
                matched_entities.insert(variable.as_str());
            }
            for step in &pattern.steps {
                if let Some(variable) = &step.relationship.variable {
                    matched_entities.insert(variable.as_str());
                }
                if let Some(variable) = &step.node.variable {
                    matched_entities.insert(variable.as_str());
                }
            }
        }

        let deleted = delete_expressions
            .iter()
            .filter_map(|expression| {
                let Expression::Variable(variable) = expression else {
                    return None;
                };
                if !matched_entities.contains(variable.as_str()) {
                    return None;
                }
                match self.scope.get(variable).copied() {
                    Some(kind @ (BindingKind::Node | BindingKind::Relationship)) => {
                        Some((variable.as_str(), kind))
                    }
                    _ => None,
                }
            })
            .collect::<BTreeMap<_, _>>();

        for item in &projection.items {
            match &item.expression {
                Expression::Property(source, _) => {
                    let Expression::Variable(variable) = source.as_ref() else {
                        continue;
                    };
                    let Some(kind) = deleted.get(variable.as_str()).copied() else {
                        continue;
                    };
                    let entity = match kind {
                        BindingKind::Node => "node",
                        BindingKind::Relationship => "relationship",
                        _ => continue,
                    };
                    return Err(deleted_entity_access_error(entity));
                }
                Expression::Function {
                    name,
                    distinct: false,
                    arguments,
                } if name.len() == 1 && name[0].eq_ignore_ascii_case("labels") => {
                    let [Expression::Variable(variable)] = arguments.as_slice() else {
                        continue;
                    };
                    if deleted.get(variable.as_str()) == Some(&BindingKind::Node) {
                        return Err(deleted_entity_access_error("node"));
                    }
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Preserve percentile range errors ahead of strict native admission only when one complete,
    /// bounded clause-local proof exists. The accepted pipeline proves that every surviving value
    /// of a direct `size()` alias is outside `[0, 1]`, forwards it unchanged, retains at least one
    /// row when input exists, and uses it directly as both percentile argument and grouping key.
    /// Every dynamic, compound, transformed, optional, or unbounded shape remains runtime-owned.
    fn validate_native_percentile_range_error_precedence(
        &self,
        clauses: &[Clause],
        projection: &Projection,
    ) -> Result<()> {
        let [
            Clause::Match {
                optional: false, ..
            },
            Clause::With(source_projection),
            Clause::Where(predicate),
            Clause::With(forward_projection),
            Clause::Limit(limit),
            Clause::Return(_),
        ] = clauses
        else {
            return Ok(());
        };
        if source_projection.items.len() != 2
            || forward_projection.items.len() != 1
            || projection.items.len() != 2
            || projection.distinct
            || static_integer_literal(limit).is_none_or(|limit| limit <= 0)
        {
            return Ok(());
        }

        let Some(constrained_variable) = static_out_of_percentile_range_variable(predicate) else {
            return Ok(());
        };
        let source_is_numeric_size = source_projection.items.iter().any(|item| {
            item.alias.as_deref() == Some(constrained_variable)
                && matches!(
                    &item.expression,
                    Expression::Function {
                        name,
                        distinct: false,
                        arguments,
                    } if name.len() == 1
                        && name[0].eq_ignore_ascii_case("size")
                        && arguments.len() == 1
                )
        });
        if source_projection.distinct || !source_is_numeric_size || forward_projection.distinct {
            return Ok(());
        }

        let [forwarded] = forward_projection.items.as_slice() else {
            return Ok(());
        };
        let Expression::Variable(forwarded_source) = &forwarded.expression else {
            return Ok(());
        };
        if forwarded_source != constrained_variable {
            return Ok(());
        }
        let forwarded_variable = forwarded.alias.as_deref().unwrap_or(forwarded_source);

        let direct_grouping_key = projection.items.iter().any(|item| {
            matches!(
                &item.expression,
                Expression::Variable(variable) if variable == forwarded_variable
            )
        });
        let invalid_percentile_call = projection.items.iter().any(|item| {
            let Expression::Function {
                name,
                distinct: false,
                arguments,
            } = &item.expression
            else {
                return false;
            };
            if !matches!(
                name.as_slice(),
                [name]
                    if name.eq_ignore_ascii_case("percentilecont")
                        || name.eq_ignore_ascii_case("percentiledisc")
            ) {
                return false;
            }
            let [value, Expression::Variable(percentile)] = arguments.as_slice() else {
                return false;
            };
            percentile == forwarded_variable
                && static_numeric_literal(value).is_some_and(f64::is_finite)
        });
        if direct_grouping_key && invalid_percentile_call {
            return Err(Error::new(
                ErrorCode::QueryType,
                "NumberOutOfRange: percentile must be finite and in 0..=1",
            ));
        }
        Ok(())
    }

    fn set_static_source_shape(&mut self, variable: &str, shape: StaticSourceShape) {
        if shape == StaticSourceShape::Unknown {
            self.static_sources.remove(variable);
        } else {
            self.static_sources.insert(variable.to_owned(), shape);
        }
    }

    fn validate_static_function_argument(
        &self,
        name: &str,
        arguments: &[Expression],
    ) -> Result<()> {
        let Some((contract, expected)) = static_function_argument_contract(name) else {
            return Ok(());
        };
        let [argument] = arguments else {
            // Arity validation is owned by the function implementation. Static kind checking is
            // useful only once this call has exactly one argument to classify.
            return Ok(());
        };
        if self
            .static_source_shape(argument)
            .violates_function_contract(contract)
        {
            return Err(Error::new(
                ErrorCode::QuerySyntax,
                format!("InvalidArgumentType: {name}() requires {expected}"),
            ));
        }
        Ok(())
    }

    /// The canonical property representation accepts only flat homogeneous scalar lists. In
    /// strict-native diagnostics, preserve the runtime property error when a literal list is
    /// already known to contain a collection; otherwise native-plan admission would hide it.
    /// Unknown values, parameters, top-level NULL, and scalar lists remain runtime-owned.
    fn validate_native_property_value_type_precedence(&self, value: &Expression) -> Result<()> {
        let Some(elements) = self.static_list_shape(value) else {
            return Ok(());
        };
        if elements
            .iter()
            .any(|kind| matches!(kind, StaticExpressionKind::List | StaticExpressionKind::Map))
        {
            return Err(Error::new(
                ErrorCode::QueryType,
                "InvalidPropertyType: property lists must be flat homogeneous lists of scalar values",
            ));
        }
        Ok(())
    }

    /// Resolve only a literal integer index into bounded list provenance retained across WITH.
    /// The normal binder deliberately keeps indexed heterogeneous values dynamic; this additional
    /// check exists solely so a strict native backend cannot replace the same runtime function
    /// error with `GpuAdmissionFailure`.
    fn validate_native_indexed_function_type_precedence(
        &self,
        name: &str,
        arguments: &[Expression],
    ) -> Result<()> {
        if !has_static_runtime_argument_contract(name) {
            return Ok(());
        }
        let [argument] = arguments else {
            return Ok(());
        };
        let Some(kind) = self.static_indexed_expression_kind(argument) else {
            return Ok(());
        };
        if let Some(message) = static_runtime_argument_error(name, kind) {
            return Err(Error::new(ErrorCode::QueryType, message));
        }
        Ok(())
    }

    fn static_indexed_expression_kind(
        &self,
        expression: &Expression,
    ) -> Option<StaticExpressionKind> {
        let Expression::Index { expression, index } = expression else {
            return None;
        };
        let elements = self.static_list_shape(expression)?;
        let index = static_integer_literal(index)?;
        let length = i64::try_from(elements.len()).ok()?;
        let index = if index < 0 {
            length.checked_add(index)?
        } else {
            index
        };
        if index < 0 || index >= length {
            return Some(StaticExpressionKind::Null);
        }
        usize::try_from(index)
            .ok()
            .and_then(|index| elements.get(index).copied())
    }

    fn static_list_shape(&self, expression: &Expression) -> Option<Vec<StaticExpressionKind>> {
        if !self.capabilities.require_native_execution {
            return None;
        }
        match expression {
            Expression::Variable(variable) => self.static_lists.get(variable).cloned(),
            Expression::List(elements) => {
                let mut scope = self.static_type_scope();
                elements
                    .iter()
                    .map(|element| conservative_static_expression_kind(element, &mut scope).ok())
                    .collect()
            }
            _ => None,
        }
    }

    fn set_static_list_shape(&mut self, variable: &str, shape: Option<Vec<StaticExpressionKind>>) {
        if let Some(shape) = shape {
            self.static_lists.insert(variable.to_owned(), shape);
        } else {
            self.static_lists.remove(variable);
        }
    }

    /// Strict native execution must not replace a deterministic runtime type error with a
    /// backend-admission error. A heterogeneous literal list cannot be represented by one static
    /// element kind, but an unfiltered comprehension invokes its direct projection once for every
    /// member. Validate those members independently when the projected function has an exact
    /// runtime type contract.
    ///
    /// This is deliberately an execution-admission check. Ordinary binding (and therefore the CPU
    /// reference path) continues to defer row-dependent errors until the expression is evaluated.
    /// Predicates, guarded/nested projections, non-literal lists, and unknown member kinds also
    /// remain deferred because their runtime reachability cannot be proved here.
    fn validate_native_list_projection_type_precedence(
        &self,
        variable: &str,
        list: &Expression,
        projection: Option<&Expression>,
    ) -> Result<()> {
        let (
            Expression::List(elements),
            Some(Expression::Function {
                name,
                distinct: false,
                arguments,
            }),
        ) = (list, projection)
        else {
            return Ok(());
        };
        let [Expression::Variable(argument)] = arguments.as_slice() else {
            return Ok(());
        };
        if argument != variable {
            return Ok(());
        }
        let function = name.join(".").to_ascii_lowercase();
        if !has_static_runtime_argument_contract(&function) {
            return Ok(());
        }

        let mut scope = self.static_type_scope();
        for element in elements {
            let kind = conservative_static_expression_kind(element, &mut scope)?;
            if let Some(message) = static_runtime_argument_error(&function, kind) {
                return Err(Error::new(ErrorCode::QueryType, message));
            }
        }
        Ok(())
    }

    fn static_type_scope(&self) -> StaticTypeScope {
        self.scope
            .iter()
            .map(|(variable, kind)| {
                let kind = match kind {
                    BindingKind::Node => StaticExpressionKind::Node,
                    BindingKind::Relationship => StaticExpressionKind::Relationship,
                    BindingKind::NodeList
                    | BindingKind::RelationshipList
                    | BindingKind::ScalarList => StaticExpressionKind::List,
                    BindingKind::Path => StaticExpressionKind::Path,
                    BindingKind::Dynamic | BindingKind::NonEntity => StaticExpressionKind::Unknown,
                    BindingKind::Value => self.static_sources.get(variable).map_or(
                        StaticExpressionKind::Unknown,
                        |shape| match shape {
                            StaticSourceShape::Unknown | StaticSourceShape::Scalar => {
                                StaticExpressionKind::Unknown
                            }
                            StaticSourceShape::Null => StaticExpressionKind::Null,
                            StaticSourceShape::List => StaticExpressionKind::List,
                            StaticSourceShape::Map => StaticExpressionKind::Map,
                            StaticSourceShape::Node => StaticExpressionKind::Node,
                            StaticSourceShape::Relationship => StaticExpressionKind::Relationship,
                            StaticSourceShape::Path => StaticExpressionKind::Path,
                            StaticSourceShape::Temporal => StaticExpressionKind::Temporal,
                            StaticSourceShape::Duration => StaticExpressionKind::Duration,
                        },
                    ),
                };
                (variable.clone(), kind)
            })
            .collect()
    }

    fn projection_scope(&self, projection: &Projection) -> BindingScope {
        let mut next = BindingScope::new();
        for item in &projection.items {
            if matches!(item.expression, Expression::Star) {
                next.extend(self.scope.iter().map(|(name, kind)| (name.clone(), *kind)));
            } else if let Some(alias) = &item.alias {
                next.insert(
                    alias.clone(),
                    self.expression_binding_kind(&item.expression),
                );
            } else if let Expression::Variable(variable) = &item.expression {
                if let Some(kind) = self.scope.get(variable) {
                    next.insert(variable.clone(), *kind);
                }
            }
        }
        next
    }

    fn projection_static_source_scope(&self, projection: &Projection) -> StaticSourceScope {
        let mut next = StaticSourceScope::new();
        for item in &projection.items {
            if matches!(item.expression, Expression::Star) {
                next.extend(
                    self.static_sources
                        .iter()
                        .map(|(name, shape)| (name.clone(), *shape)),
                );
            } else if let Some(alias) = &item.alias {
                let shape = self.static_source_shape(&item.expression);
                if shape != StaticSourceShape::Unknown {
                    next.insert(alias.clone(), shape);
                }
            } else if let Expression::Variable(variable) = &item.expression
                && let Some(shape) = self.static_sources.get(variable)
            {
                next.insert(variable.clone(), *shape);
            }
        }
        next
    }

    fn projection_static_list_scope(&self, projection: &Projection) -> StaticListScope {
        let mut next = StaticListScope::new();
        if !self.capabilities.require_native_execution {
            return next;
        }
        for item in &projection.items {
            if matches!(item.expression, Expression::Star) {
                next.extend(
                    self.static_lists
                        .iter()
                        .map(|(name, shape)| (name.clone(), shape.clone())),
                );
            } else if let Some(alias) = &item.alias {
                if let Some(shape) = self.static_list_shape(&item.expression) {
                    next.insert(alias.clone(), shape);
                }
            } else if let Expression::Variable(variable) = &item.expression
                && let Some(shape) = self.static_lists.get(variable)
            {
                next.insert(variable.clone(), shape.clone());
            }
        }
        next
    }

    fn require_write(&self) -> Result<()> {
        if self.capabilities.write {
            Ok(())
        } else {
            Err(Error::new(
                ErrorCode::AuthorizationDenied,
                "write capability is required",
            ))
        }
    }

    fn require_schema(&self) -> Result<()> {
        if self.capabilities.schema {
            Ok(())
        } else {
            Err(Error::new(
                ErrorCode::AuthorizationDenied,
                "schema capability is required",
            ))
        }
    }

    fn dependency(&mut self, kind: DependencyKind, name: &str) {
        self.dependencies.insert(DependencyStamp {
            kind,
            name: name.to_owned(),
        });
    }
}

fn procedure_output_binding_kind(procedure: &ResolvedProcedure<'_>, output: &str) -> BindingKind {
    match procedure {
        // Execution-scoped procedure signatures currently expose only declared scalar types.
        ResolvedProcedure::ExecutionScoped(_) => BindingKind::Value,
        ResolvedProcedure::Builtin(_) => match output.to_ascii_lowercase().as_str() {
            "node" | "predecessor" => BindingKind::Node,
            "relationship" => BindingKind::Relationship,
            "path" => BindingKind::Path,
            _ => BindingKind::Value,
        },
    }
}

fn static_literal_source_shape(value: &ScalarValue) -> StaticSourceShape {
    match value {
        ScalarValue::Null => StaticSourceShape::Null,
        ScalarValue::Boolean(_)
        | ScalarValue::Integer(_)
        | ScalarValue::Float(_)
        | ScalarValue::String(_)
        | ScalarValue::Bytes(_) => StaticSourceShape::Scalar,
        ScalarValue::Date(_)
        | ScalarValue::LocalTime(_)
        | ScalarValue::ZonedTime { .. }
        | ScalarValue::LocalDateTime { .. }
        | ScalarValue::ZonedDateTime { .. } => StaticSourceShape::Temporal,
        ScalarValue::Duration { .. } => StaticSourceShape::Duration,
        ScalarValue::List(_) => StaticSourceShape::List,
        ScalarValue::Map(_) => StaticSourceShape::Map,
    }
}

fn static_function_source_shape(name: &[String]) -> StaticSourceShape {
    if matches!(name, [intrinsic] if intrinsic == ENTITY_LABEL_PREDICATE_INTRINSIC || intrinsic == PATTERN_PREDICATE_INTRINSIC)
    {
        return StaticSourceShape::Scalar;
    }
    let name = name.join(".").to_ascii_lowercase();
    if is_temporal_clock_function(&name) {
        return StaticSourceShape::Temporal;
    }
    match name.as_str() {
        "properties" => StaticSourceShape::Map,
        "startnode" | "endnode" => StaticSourceShape::Node,
        "date"
        | "date.truncate"
        | "datetime"
        | "datetime.fromepoch"
        | "datetime.fromepochmillis"
        | "datetime.truncate"
        | "localdatetime"
        | "localdatetime.truncate"
        | "localtime"
        | "localtime.truncate"
        | "time"
        | "time.truncate" => StaticSourceShape::Temporal,
        "duration" | "duration.between" | "duration.indays" | "duration.inmonths"
        | "duration.inseconds" => StaticSourceShape::Duration,
        _ => StaticSourceShape::Unknown,
    }
}

fn static_function_argument_contract(
    name: &str,
) -> Option<(StaticFunctionArgumentContract, &'static str)> {
    match name {
        "labels" => Some((StaticFunctionArgumentContract::Node, "a NODE or NULL")),
        "type" => Some((
            StaticFunctionArgumentContract::Relationship,
            "a RELATIONSHIP or NULL",
        )),
        "length" => Some((
            StaticFunctionArgumentContract::PathLength,
            "a compatible scalar, container, PATH, or NULL",
        )),
        "size" => Some((
            StaticFunctionArgumentContract::Size,
            "a LIST, MAP, STRING, dynamically typed value, or NULL (use length() for PATH values)",
        )),
        "properties" => Some((
            StaticFunctionArgumentContract::PropertiesSource,
            "a MAP, NODE, RELATIONSHIP, or NULL",
        )),
        _ => None,
    }
}

fn duplicate_projection_column(name: &str) -> Error {
    Error::new(
        ErrorCode::QuerySyntax,
        format!("ColumnNameConflict: projection column `{name}` is specified more than once"),
    )
}

fn deleted_entity_access_error(entity: &'static str) -> Error {
    Error::new(
        ErrorCode::QueryType,
        format!("DeletedEntityAccess: {entity} was deleted in this query"),
    )
}

fn uniform_static_source_shape(
    mut shapes: impl Iterator<Item = StaticSourceShape>,
) -> StaticSourceShape {
    let Some(first) = shapes.next() else {
        return StaticSourceShape::Unknown;
    };
    if first == StaticSourceShape::Unknown || shapes.any(|shape| shape != first) {
        StaticSourceShape::Unknown
    } else {
        first
    }
}

const fn binding_kind_name(kind: BindingKind) -> &'static str {
    match kind {
        BindingKind::Node => "NODE",
        BindingKind::Relationship => "RELATIONSHIP",
        BindingKind::NodeList => "NODE LIST",
        BindingKind::RelationshipList => "RELATIONSHIP LIST",
        BindingKind::ScalarList => "VALUE LIST",
        BindingKind::Path => "PATH",
        BindingKind::Dynamic => "DYNAMIC VALUE",
        BindingKind::NonEntity => "VALUE",
        BindingKind::Value => "VALUE",
    }
}

fn restore_binding(scope: &mut BindingScope, variable: &str, prior: Option<BindingKind>) {
    if let Some(kind) = prior {
        scope.insert(variable.to_owned(), kind);
    } else {
        scope.remove(variable);
    }
}

fn expression_is_empty_list(expression: &Expression) -> bool {
    matches!(expression, Expression::List(items) if items.is_empty())
}

fn restore_static_source_shape(
    scope: &mut StaticSourceScope,
    variable: &str,
    prior: Option<StaticSourceShape>,
) {
    if let Some(shape) = prior {
        scope.insert(variable.to_owned(), shape);
    } else {
        scope.remove(variable);
    }
}

fn restore_static_list_shape(
    scope: &mut StaticListScope,
    variable: &str,
    prior: Option<Vec<StaticExpressionKind>>,
) {
    if let Some(shape) = prior {
        scope.insert(variable.to_owned(), shape);
    } else {
        scope.remove(variable);
    }
}

fn static_integer_literal(expression: &Expression) -> Option<i64> {
    match expression {
        Expression::Literal(ScalarValue::Integer(value)) => Some(*value),
        Expression::Unary {
            operation: UnaryOperator::Positive,
            operand,
        } => static_integer_literal(operand),
        Expression::Unary {
            operation: UnaryOperator::Negative,
            operand,
        } => static_integer_literal(operand)?.checked_neg(),
        _ => None,
    }
}

fn static_numeric_literal(expression: &Expression) -> Option<f64> {
    match expression {
        Expression::Literal(ScalarValue::Integer(value)) => Some(*value as f64),
        Expression::Literal(ScalarValue::Float(value)) => Some(value.0),
        Expression::Unary {
            operation: UnaryOperator::Positive,
            operand,
        } => static_numeric_literal(operand),
        Expression::Unary {
            operation: UnaryOperator::Negative,
            operand,
        } => Some(-static_numeric_literal(operand)?),
        _ => None,
    }
}

/// Returns a direct variable only when an atomic numeric comparison proves every surviving value
/// lies above the legal percentile interval. `size()` cannot be negative, so lower-bound failures
/// would instead prove an empty row set and must not be promoted. Equality, compound predicates,
/// parameters, and non-finite bounds intentionally carry no proof.
fn static_out_of_percentile_range_variable(expression: &Expression) -> Option<&str> {
    let Expression::Binary {
        left,
        operation,
        right,
    } = expression
    else {
        return None;
    };
    match (left.as_ref(), operation, right.as_ref()) {
        (Expression::Variable(variable), BinaryOperator::Greater, bound)
            if static_numeric_literal(bound)
                .is_some_and(|bound| bound.is_finite() && bound >= 1.0) =>
        {
            Some(variable)
        }
        (Expression::Variable(variable), BinaryOperator::GreaterOrEqual, bound)
            if static_numeric_literal(bound)
                .is_some_and(|bound| bound.is_finite() && bound > 1.0) =>
        {
            Some(variable)
        }
        (bound, BinaryOperator::Less, Expression::Variable(variable))
            if static_numeric_literal(bound)
                .is_some_and(|bound| bound.is_finite() && bound >= 1.0) =>
        {
            Some(variable)
        }
        (bound, BinaryOperator::LessOrEqual, Expression::Variable(variable))
            if static_numeric_literal(bound)
                .is_some_and(|bound| bound.is_finite() && bound > 1.0) =>
        {
            Some(variable)
        }
        _ => None,
    }
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

/// The shared expression walker is the structural detector for nested aggregates. The openCypher
/// TCK classifies that specific compile-time failure as syntax (`NestedAggregation`), so the binder
/// translates only this validator's error category while preserving its accepted expression set.
fn validate_aggregate_nesting_as_syntax(expression: &Expression) -> Result<()> {
    match validate_aggregate_nesting(expression) {
        Ok(()) => Ok(()),
        Err(error) if error.code == ErrorCode::QueryType => Err(Error::new(
            ErrorCode::QuerySyntax,
            "NestedAggregation: aggregate functions cannot be nested",
        )),
        Err(error) => Err(error),
    }
}

/// Aggregation consumes the rows entering a projection, whereas these expressions execute once
/// per element after their input list has already been produced. Aggregation is therefore legal
/// in an iteration construct's input (and in a reduce initializer), but not in its element-local
/// predicate, projection, or reduction expression.
fn reject_aggregate_in_iteration_scope(expression: &Expression, scope: &str) -> Result<()> {
    reject_aggregate_in_row_context(expression, &format!("a {scope}"))
}

/// Preserve the standard error precedence for an aggregate ORDER BY which is guaranteed to reject
/// an unaliased complex grouping expression as `AmbiguousAggregationExpression`. The projection
/// is still rejected during planning; deferral only prevents the broader `NoExpressionAlias`
/// check from hiding that more specific error. All ordinary WITH boundaries validate aliases
/// immediately.
fn defers_with_alias_check_to_aggregate_order_validation(
    projection: &Projection,
    next_clause: Option<&Clause>,
) -> bool {
    let Some(Clause::OrderBy(items)) = next_clause else {
        return false;
    };
    if !projection
        .items
        .iter()
        .any(|item| contains_aggregate(&item.expression))
    {
        return false;
    }

    items.iter().any(|item| {
        let Expression::Binary { left, right, .. } = &item.expression else {
            return false;
        };
        !contains_aggregate(left)
            && contains_aggregate(right)
            && projection.items.iter().any(|projected| {
                projected.alias.is_none()
                    && !contains_aggregate(&projected.expression)
                    && !matches!(
                        projected.expression,
                        Expression::Variable(_) | Expression::Property(_, _)
                    )
                    && projected.expression == **left
            })
    })
}

/// Once WITH has produced an aggregating table, ORDER BY may reuse aggregate expressions which
/// contributed to that table, but it cannot introduce a different aggregate over the input rows.
/// A non-aggregating WITH is deliberately excluded here so planner validation retains precedence
/// for the standard InvalidAggregation error.
fn validate_order_by_aggregate_projection(
    items: &[SortItem],
    projection: &Projection,
) -> Result<()> {
    let mut projected_aggregates = Vec::new();
    for item in &projection.items {
        collect_aggregate_expressions(&item.expression, &mut projected_aggregates);
    }
    if projected_aggregates.is_empty() {
        return Ok(());
    }

    for item in items {
        validate_aggregate_nesting_as_syntax(&item.expression)?;
        let mut order_aggregates = Vec::new();
        collect_aggregate_expressions(&item.expression, &mut order_aggregates);
        if order_aggregates.iter().any(|aggregate| {
            !projected_aggregates
                .iter()
                .any(|projected| *projected == *aggregate)
        }) {
            return Err(Error::new(
                ErrorCode::QuerySyntax,
                "UndefinedVariable: ORDER BY aggregate is not present in the aggregating projection",
            ));
        }
    }
    Ok(())
}

fn collect_aggregate_expressions<'a>(
    expression: &'a Expression,
    aggregates: &mut Vec<&'a Expression>,
) {
    if matches!(expression, Expression::Function { name, .. } if is_aggregate_function(name)) {
        aggregates.push(expression);
        return;
    }
    if !contains_aggregate(expression) {
        return;
    }

    match expression {
        Expression::Property(value, _)
        | Expression::Unary { operand: value, .. }
        | Expression::IsNull {
            expression: value, ..
        } => collect_aggregate_expressions(value, aggregates),
        Expression::List(values)
        | Expression::Function {
            arguments: values, ..
        } => {
            for value in values {
                collect_aggregate_expressions(value, aggregates);
            }
        }
        Expression::Map(values) => {
            for (_, value) in values {
                collect_aggregate_expressions(value, aggregates);
            }
        }
        Expression::MapProjection { source, items } => {
            collect_aggregate_expressions(source, aggregates);
            for item in items {
                if let MapProjectionItem::Entry(_, value) = item {
                    collect_aggregate_expressions(value, aggregates);
                }
            }
        }
        Expression::Case {
            operand,
            alternatives,
            default,
        } => {
            if let Some(operand) = operand {
                collect_aggregate_expressions(operand, aggregates);
            }
            for alternative in alternatives {
                collect_aggregate_expressions(&alternative.when, aggregates);
                collect_aggregate_expressions(&alternative.then, aggregates);
            }
            if let Some(default) = default {
                collect_aggregate_expressions(default, aggregates);
            }
        }
        Expression::ListComprehension {
            list,
            predicate,
            projection,
            ..
        } => {
            collect_aggregate_expressions(list, aggregates);
            if let Some(predicate) = predicate {
                collect_aggregate_expressions(predicate, aggregates);
            }
            if let Some(projection) = projection {
                collect_aggregate_expressions(projection, aggregates);
            }
        }
        Expression::Reduce {
            initial,
            list,
            expression,
            ..
        } => {
            collect_aggregate_expressions(initial, aggregates);
            collect_aggregate_expressions(list, aggregates);
            collect_aggregate_expressions(expression, aggregates);
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
            collect_aggregate_expressions(list, aggregates);
            collect_aggregate_expressions(predicate, aggregates);
        }
        Expression::Slice {
            expression,
            start,
            end,
        } => {
            collect_aggregate_expressions(expression, aggregates);
            if let Some(start) = start {
                collect_aggregate_expressions(start, aggregates);
            }
            if let Some(end) = end {
                collect_aggregate_expressions(end, aggregates);
            }
        }
        Expression::Literal(_)
        | Expression::Parameter(_)
        | Expression::Variable(_)
        | Expression::ExistentialSubquery(_)
        | Expression::Star => {}
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RowCountClassification {
    KnownInteger(i64),
    Integer,
    NonInteger,
    Unknown,
}

fn row_count_expression_is_independent(
    expression: &Expression,
    local_bindings: &BTreeSet<String>,
) -> bool {
    if matches!(expression, Expression::Function { name, .. } if is_aggregate_function(name)) {
        return false;
    }
    match expression {
        Expression::Literal(_) | Expression::Parameter(_) => true,
        Expression::Variable(variable) => local_bindings.contains(variable),
        Expression::Property(value, _) | Expression::Unary { operand: value, .. } => {
            row_count_expression_is_independent(value, local_bindings)
        }
        Expression::List(values)
        | Expression::Function {
            arguments: values, ..
        } => values
            .iter()
            .all(|value| row_count_expression_is_independent(value, local_bindings)),
        Expression::Map(values) => values
            .iter()
            .all(|(_, value)| row_count_expression_is_independent(value, local_bindings)),
        Expression::MapProjection { source, items } => {
            row_count_expression_is_independent(source, local_bindings)
                && items.iter().all(|item| match item {
                    MapProjectionItem::AllProperties | MapProjectionItem::Property(_) => true,
                    MapProjectionItem::Variable(variable) => local_bindings.contains(variable),
                    MapProjectionItem::Entry(_, value) => {
                        row_count_expression_is_independent(value, local_bindings)
                    }
                })
        }
        Expression::Case {
            operand,
            alternatives,
            default,
        } => {
            operand
                .as_deref()
                .is_none_or(|value| row_count_expression_is_independent(value, local_bindings))
                && alternatives.iter().all(|alternative| {
                    row_count_expression_is_independent(&alternative.when, local_bindings)
                        && row_count_expression_is_independent(&alternative.then, local_bindings)
                })
                && default
                    .as_deref()
                    .is_none_or(|value| row_count_expression_is_independent(value, local_bindings))
        }
        Expression::ListComprehension {
            variable,
            list,
            predicate,
            projection,
        } => {
            if !row_count_expression_is_independent(list, local_bindings) {
                return false;
            }
            let mut nested = local_bindings.clone();
            nested.insert(variable.clone());
            predicate
                .as_deref()
                .is_none_or(|value| row_count_expression_is_independent(value, &nested))
                && projection
                    .as_deref()
                    .is_none_or(|value| row_count_expression_is_independent(value, &nested))
        }
        Expression::Reduce {
            accumulator,
            initial,
            variable,
            list,
            expression,
        } => {
            if !row_count_expression_is_independent(initial, local_bindings)
                || !row_count_expression_is_independent(list, local_bindings)
            {
                return false;
            }
            let mut nested = local_bindings.clone();
            nested.insert(accumulator.clone());
            nested.insert(variable.clone());
            row_count_expression_is_independent(expression, &nested)
        }
        Expression::ListPredicate {
            variable,
            list,
            predicate,
            ..
        } => {
            if !row_count_expression_is_independent(list, local_bindings) {
                return false;
            }
            let mut nested = local_bindings.clone();
            nested.insert(variable.clone());
            row_count_expression_is_independent(predicate, &nested)
        }
        Expression::Binary { left, right, .. }
        | Expression::Index {
            expression: left,
            index: right,
        } => {
            row_count_expression_is_independent(left, local_bindings)
                && row_count_expression_is_independent(right, local_bindings)
        }
        Expression::IsNull { expression, .. } => {
            row_count_expression_is_independent(expression, local_bindings)
        }
        Expression::Slice {
            expression,
            start,
            end,
        } => {
            row_count_expression_is_independent(expression, local_bindings)
                && start
                    .as_deref()
                    .is_none_or(|value| row_count_expression_is_independent(value, local_bindings))
                && end
                    .as_deref()
                    .is_none_or(|value| row_count_expression_is_independent(value, local_bindings))
        }
        Expression::ExistentialSubquery(_) | Expression::Star => false,
    }
}

fn classify_row_count_expression(expression: &Expression) -> RowCountClassification {
    match expression {
        Expression::Literal(ScalarValue::Integer(value)) => {
            RowCountClassification::KnownInteger(*value)
        }
        Expression::Literal(_) => RowCountClassification::NonInteger,
        Expression::Parameter(_) => RowCountClassification::Unknown,
        Expression::Unary { operation, operand } => {
            let operand = classify_row_count_expression(operand);
            match operation {
                UnaryOperator::Positive => operand,
                UnaryOperator::Negative => match operand {
                    RowCountClassification::KnownInteger(value) => value.checked_neg().map_or(
                        RowCountClassification::Integer,
                        RowCountClassification::KnownInteger,
                    ),
                    RowCountClassification::Integer => RowCountClassification::Integer,
                    RowCountClassification::NonInteger => RowCountClassification::NonInteger,
                    RowCountClassification::Unknown => RowCountClassification::Unknown,
                },
                UnaryOperator::Not => RowCountClassification::NonInteger,
            }
        }
        Expression::Binary {
            left,
            operation,
            right,
        } => classify_row_count_binary(left, *operation, right),
        Expression::Function {
            name, arguments, ..
        } => classify_row_count_function(name, arguments),
        Expression::Case { .. }
        | Expression::Property(_, _)
        | Expression::Index { .. }
        | Expression::Reduce { .. } => RowCountClassification::Unknown,
        Expression::List(_)
        | Expression::Map(_)
        | Expression::MapProjection { .. }
        | Expression::ListComprehension { .. }
        | Expression::ListPredicate { .. }
        | Expression::IsNull { .. }
        | Expression::Slice { .. }
        | Expression::ExistentialSubquery(_)
        | Expression::Star => RowCountClassification::NonInteger,
        Expression::Variable(_) => RowCountClassification::Unknown,
    }
}

fn classify_row_count_binary(
    left: &Expression,
    operation: BinaryOperator,
    right: &Expression,
) -> RowCountClassification {
    let left = classify_row_count_expression(left);
    let right = classify_row_count_expression(right);
    match operation {
        BinaryOperator::Add
        | BinaryOperator::Subtract
        | BinaryOperator::Multiply
        | BinaryOperator::Divide
        | BinaryOperator::Modulo => classify_integer_row_count_arithmetic(left, operation, right),
        BinaryOperator::Power
        | BinaryOperator::Or
        | BinaryOperator::Xor
        | BinaryOperator::And
        | BinaryOperator::Equal
        | BinaryOperator::NotEqual
        | BinaryOperator::Less
        | BinaryOperator::LessOrEqual
        | BinaryOperator::Greater
        | BinaryOperator::GreaterOrEqual
        | BinaryOperator::In
        | BinaryOperator::StartsWith
        | BinaryOperator::EndsWith
        | BinaryOperator::Contains
        | BinaryOperator::RegexMatch
        | BinaryOperator::Concat => RowCountClassification::NonInteger,
    }
}

fn classify_integer_row_count_arithmetic(
    left: RowCountClassification,
    operation: BinaryOperator,
    right: RowCountClassification,
) -> RowCountClassification {
    match (left, right) {
        (
            RowCountClassification::KnownInteger(left),
            RowCountClassification::KnownInteger(right),
        ) => {
            let value = match operation {
                BinaryOperator::Add => left.checked_add(right),
                BinaryOperator::Subtract => left.checked_sub(right),
                BinaryOperator::Multiply => left.checked_mul(right),
                BinaryOperator::Divide => left.checked_div(right),
                BinaryOperator::Modulo => left.checked_rem(right),
                _ => None,
            };
            value.map_or(
                RowCountClassification::Integer,
                RowCountClassification::KnownInteger,
            )
        }
        (RowCountClassification::NonInteger, _) | (_, RowCountClassification::NonInteger) => {
            RowCountClassification::NonInteger
        }
        (RowCountClassification::Unknown, _) | (_, RowCountClassification::Unknown) => {
            RowCountClassification::Unknown
        }
        _ => RowCountClassification::Integer,
    }
}

fn classify_row_count_function(
    name: &[String],
    arguments: &[Expression],
) -> RowCountClassification {
    match name.join(".").to_ascii_lowercase().as_str() {
        "tointeger" | "size" | "length" | "id" | "revision" => RowCountClassification::Integer,
        "abs" => arguments
            .first()
            .map_or(
                RowCountClassification::Unknown,
                |argument| match classify_row_count_expression(argument) {
                    RowCountClassification::KnownInteger(value) => value.checked_abs().map_or(
                        RowCountClassification::Integer,
                        RowCountClassification::KnownInteger,
                    ),
                    classification => classification,
                },
            ),
        "sign" => arguments
            .first()
            .map_or(
                RowCountClassification::Unknown,
                |argument| match classify_row_count_expression(argument) {
                    RowCountClassification::KnownInteger(value) => {
                        RowCountClassification::KnownInteger(value.signum())
                    }
                    RowCountClassification::Integer => RowCountClassification::Integer,
                    RowCountClassification::NonInteger => RowCountClassification::Integer,
                    RowCountClassification::Unknown => RowCountClassification::Unknown,
                },
            ),
        "rand" | "ceil" | "floor" | "round" | "tofloat" => RowCountClassification::NonInteger,
        _ => RowCountClassification::Unknown,
    }
}

/// A literal container/number/string cannot become a Boolean at runtime. Rejecting it before
/// optimization preserves Cypher's compile-time `InvalidArgumentType` contract and prevents
/// Boolean identity rewrites from erasing the invalid operand.
fn statically_non_boolean(expression: &Expression) -> bool {
    match expression {
        Expression::Literal(ScalarValue::Boolean(_) | ScalarValue::Null) => false,
        Expression::Literal(_) | Expression::List(_) | Expression::Map(_) => true,
        _ => false,
    }
}

/// A literal scalar or map cannot become a LIST at runtime, while NULL remains a valid
/// three-valued `IN` right operand. Rejecting only the statically impossible cases during
/// binding gives both the CPU reference and every native backend the standard error phase.
fn statically_non_list(expression: &Expression) -> bool {
    match expression {
        Expression::Literal(ScalarValue::Null) | Expression::List(_) => false,
        Expression::Literal(_) | Expression::Map(_) => true,
        Expression::Unary {
            operation: UnaryOperator::Positive | UnaryOperator::Negative,
            operand,
        } => statically_non_list(operand),
        _ => false,
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StaticExpressionKind {
    Unknown,
    Null,
    Boolean,
    Number,
    String,
    Bytes,
    List,
    Map,
    Node,
    Relationship,
    Path,
    Temporal,
    Duration,
}

impl StaticExpressionKind {
    const fn is_definitely_non_boolean(self) -> bool {
        !matches!(self, Self::Unknown | Self::Null | Self::Boolean)
    }
}

type StaticTypeScope = BTreeMap<String, StaticExpressionKind>;

/// A quantified predicate evaluates once for every list element. A homogeneous literal list can
/// therefore establish a conservative local type for its iteration variable. Empty, mixed,
/// parameterized, and otherwise unresolved lists deliberately remain unknown.
fn validate_static_list_predicate_types(
    variable: &str,
    list: &Expression,
    predicate: &Expression,
) -> Result<()> {
    let mut scope = StaticTypeScope::new();
    let element_kind = homogeneous_literal_list_element_kind(list, &mut scope)?;
    scope.insert(variable.to_owned(), element_kind);
    let predicate_kind = conservative_static_expression_kind(predicate, &mut scope)?;
    if predicate_kind.is_definitely_non_boolean() {
        Err(Error::new(
            ErrorCode::QuerySyntax,
            "InvalidArgumentType: list predicate must produce BOOLEAN or NULL",
        ))
    } else {
        Ok(())
    }
}

fn homogeneous_literal_list_element_kind(
    expression: &Expression,
    scope: &mut StaticTypeScope,
) -> Result<StaticExpressionKind> {
    let Expression::List(values) = expression else {
        return Ok(StaticExpressionKind::Unknown);
    };
    let mut inferred = None;
    for value in values {
        let kind = conservative_static_expression_kind(value, scope)?;
        if kind == StaticExpressionKind::Unknown {
            return Ok(StaticExpressionKind::Unknown);
        }
        match inferred {
            None => inferred = Some(kind),
            Some(existing) if existing == kind => {}
            Some(_) => return Ok(StaticExpressionKind::Unknown),
        }
    }
    Ok(match inferred {
        Some(kind) => kind,
        None => StaticExpressionKind::Unknown,
    })
}

fn conservative_static_expression_kind(
    expression: &Expression,
    scope: &mut StaticTypeScope,
) -> Result<StaticExpressionKind> {
    if expression.pattern_comprehension_parts().is_some() {
        return Ok(StaticExpressionKind::List);
    }
    if expression.existential_subquery_parts().is_some() {
        return Ok(StaticExpressionKind::Boolean);
    }
    match expression {
        Expression::Literal(value) => Ok(static_literal_kind(value)),
        Expression::Parameter(_) | Expression::Star => Ok(StaticExpressionKind::Unknown),
        Expression::Variable(variable) => Ok(scope
            .get(variable)
            .copied()
            .map_or(StaticExpressionKind::Unknown, |kind| kind)),
        Expression::Property(value, _) => {
            let _ = conservative_static_expression_kind(value, scope)?;
            Ok(StaticExpressionKind::Unknown)
        }
        Expression::List(values) => {
            for value in values {
                let _ = conservative_static_expression_kind(value, scope)?;
            }
            Ok(StaticExpressionKind::List)
        }
        Expression::Map(values) => {
            for (_, value) in values {
                let _ = conservative_static_expression_kind(value, scope)?;
            }
            Ok(StaticExpressionKind::Map)
        }
        Expression::MapProjection { source, items } => {
            let _ = conservative_static_expression_kind(source, scope)?;
            for item in items {
                if let MapProjectionItem::Entry(_, value) = item {
                    let _ = conservative_static_expression_kind(value, scope)?;
                }
            }
            Ok(StaticExpressionKind::Map)
        }
        Expression::Case {
            operand,
            alternatives,
            default,
        } => {
            if let Some(operand) = operand {
                let _ = conservative_static_expression_kind(operand, scope)?;
            }
            for alternative in alternatives {
                let _ = conservative_static_expression_kind(&alternative.when, scope)?;
                let _ = conservative_static_expression_kind(&alternative.then, scope)?;
            }
            if let Some(default) = default {
                let _ = conservative_static_expression_kind(default, scope)?;
            }
            Ok(StaticExpressionKind::Unknown)
        }
        Expression::ListComprehension {
            variable,
            list,
            predicate,
            projection,
        } => {
            let _ = conservative_static_expression_kind(list, scope)?;
            let element_kind = homogeneous_literal_list_element_kind(list, scope)?;
            let prior = scope.insert(variable.clone(), element_kind);
            if let Some(predicate) = predicate {
                let kind = conservative_static_expression_kind(predicate, scope)?;
                require_static_boolean(kind)?;
            }
            if let Some(projection) = projection {
                let _ = conservative_static_expression_kind(projection, scope)?;
            }
            restore_static_type(scope, variable, prior);
            Ok(StaticExpressionKind::List)
        }
        Expression::Reduce {
            accumulator,
            initial,
            variable,
            list,
            expression,
        } => {
            let accumulator_kind = conservative_static_expression_kind(initial, scope)?;
            let _ = conservative_static_expression_kind(list, scope)?;
            let element_kind = homogeneous_literal_list_element_kind(list, scope)?;
            let accumulator_prior = scope.insert(accumulator.clone(), accumulator_kind);
            let variable_prior =
                (variable != accumulator).then(|| scope.insert(variable.clone(), element_kind));
            let result = conservative_static_expression_kind(expression, scope)?;
            if let Some(prior) = variable_prior {
                restore_static_type(scope, variable, prior);
            }
            restore_static_type(scope, accumulator, accumulator_prior);
            Ok(result)
        }
        Expression::ListPredicate {
            variable,
            list,
            predicate,
            ..
        } => {
            let _ = conservative_static_expression_kind(list, scope)?;
            let element_kind = homogeneous_literal_list_element_kind(list, scope)?;
            let prior = scope.insert(variable.clone(), element_kind);
            let predicate_kind = conservative_static_expression_kind(predicate, scope)?;
            restore_static_type(scope, variable, prior);
            require_static_boolean(predicate_kind)?;
            Ok(StaticExpressionKind::Boolean)
        }
        Expression::Function {
            name, arguments, ..
        } => {
            let mut argument_kinds = Vec::with_capacity(arguments.len());
            for argument in arguments {
                argument_kinds.push(conservative_static_expression_kind(argument, scope)?);
            }
            Ok(static_function_result_kind(name, &argument_kinds))
        }
        Expression::Unary { operation, operand } => {
            let operand = conservative_static_expression_kind(operand, scope)?;
            Ok(match operation {
                UnaryOperator::Not => {
                    require_static_boolean(operand)?;
                    StaticExpressionKind::Boolean
                }
                UnaryOperator::Positive | UnaryOperator::Negative => match operand {
                    StaticExpressionKind::Number => StaticExpressionKind::Number,
                    StaticExpressionKind::Null => StaticExpressionKind::Null,
                    _ => StaticExpressionKind::Unknown,
                },
            })
        }
        Expression::Binary {
            left,
            operation,
            right,
        } => {
            let left = conservative_static_expression_kind(left, scope)?;
            let right = conservative_static_expression_kind(right, scope)?;
            match operation {
                BinaryOperator::Modulo => static_modulo_result_kind(left, right),
                BinaryOperator::And | BinaryOperator::Or | BinaryOperator::Xor => {
                    require_static_boolean(left)?;
                    require_static_boolean(right)?;
                    Ok(StaticExpressionKind::Boolean)
                }
                BinaryOperator::Equal
                | BinaryOperator::NotEqual
                | BinaryOperator::Less
                | BinaryOperator::LessOrEqual
                | BinaryOperator::Greater
                | BinaryOperator::GreaterOrEqual
                | BinaryOperator::In
                | BinaryOperator::StartsWith
                | BinaryOperator::EndsWith
                | BinaryOperator::Contains
                | BinaryOperator::RegexMatch => Ok(StaticExpressionKind::Boolean),
                BinaryOperator::Add
                | BinaryOperator::Subtract
                | BinaryOperator::Multiply
                | BinaryOperator::Divide
                | BinaryOperator::Power
                | BinaryOperator::Concat => Ok(StaticExpressionKind::Unknown),
            }
        }
        Expression::IsNull { expression, .. } => {
            let _ = conservative_static_expression_kind(expression, scope)?;
            Ok(StaticExpressionKind::Boolean)
        }
        Expression::Index { expression, index } => {
            let _ = conservative_static_expression_kind(expression, scope)?;
            let _ = conservative_static_expression_kind(index, scope)?;
            Ok(StaticExpressionKind::Unknown)
        }
        Expression::Slice {
            expression,
            start,
            end,
        } => {
            let _ = conservative_static_expression_kind(expression, scope)?;
            if let Some(start) = start {
                let _ = conservative_static_expression_kind(start, scope)?;
            }
            if let Some(end) = end {
                let _ = conservative_static_expression_kind(end, scope)?;
            }
            Ok(StaticExpressionKind::Unknown)
        }
        Expression::ExistentialSubquery(_) => Ok(StaticExpressionKind::Boolean),
    }
}

fn static_literal_kind(value: &ScalarValue) -> StaticExpressionKind {
    match value {
        ScalarValue::Null => StaticExpressionKind::Null,
        ScalarValue::Boolean(_) => StaticExpressionKind::Boolean,
        ScalarValue::Integer(_) | ScalarValue::Float(_) => StaticExpressionKind::Number,
        ScalarValue::String(_) => StaticExpressionKind::String,
        ScalarValue::Bytes(_) => StaticExpressionKind::Bytes,
        ScalarValue::Date(_)
        | ScalarValue::LocalTime(_)
        | ScalarValue::ZonedTime { .. }
        | ScalarValue::LocalDateTime { .. }
        | ScalarValue::ZonedDateTime { .. } => StaticExpressionKind::Temporal,
        ScalarValue::Duration { .. } => StaticExpressionKind::Duration,
        ScalarValue::List(_) => StaticExpressionKind::List,
        ScalarValue::Map(_) => StaticExpressionKind::Map,
    }
}

fn static_function_result_kind(
    name: &[String],
    argument_kinds: &[StaticExpressionKind],
) -> StaticExpressionKind {
    if matches!(name, [intrinsic] if intrinsic == ENTITY_LABEL_PREDICATE_INTRINSIC || intrinsic == PATTERN_PREDICATE_INTRINSIC)
    {
        return StaticExpressionKind::Boolean;
    }
    let name = name.join(".").to_ascii_lowercase();
    if is_temporal_clock_function(&name) {
        if argument_kinds.contains(&StaticExpressionKind::Null) {
            return StaticExpressionKind::Null;
        }
        if argument_kinds.contains(&StaticExpressionKind::Unknown) {
            return StaticExpressionKind::Unknown;
        }
        return StaticExpressionKind::Temporal;
    }
    match name.as_str() {
        "toboolean" => StaticExpressionKind::Boolean,
        "abs" | "ceil" | "floor" | "length" | "round" | "sign" | "size" | "tofloat"
        | "tointeger" => StaticExpressionKind::Number,
        "lower" | "tolower" | "tostring" | "upper" => StaticExpressionKind::String,
        "collect" | "range" | "relationships" | "nodes" | "tail" => StaticExpressionKind::List,
        _ => StaticExpressionKind::Unknown,
    }
}

fn has_static_runtime_argument_contract(name: &str) -> bool {
    matches!(
        name,
        "toboolean" | "tointeger" | "tofloat" | "tostring" | "labels" | "type"
    )
}

/// Returns the exact runtime error for a statically incompatible function argument. Unknown
/// values are intentionally accepted: parameters, indexed values, and mixed dynamic expressions
/// must retain their normal runtime semantics.
fn static_runtime_argument_error(name: &str, kind: StaticExpressionKind) -> Option<&'static str> {
    if matches!(
        kind,
        StaticExpressionKind::Unknown | StaticExpressionKind::Null
    ) {
        return None;
    }
    let invalid = match name {
        "toboolean" => !matches!(
            kind,
            StaticExpressionKind::Boolean | StaticExpressionKind::String
        ),
        "tointeger" => !matches!(
            kind,
            StaticExpressionKind::Boolean
                | StaticExpressionKind::Number
                | StaticExpressionKind::String
        ),
        "tofloat" => !matches!(
            kind,
            StaticExpressionKind::Number | StaticExpressionKind::String
        ),
        "tostring" => !matches!(
            kind,
            StaticExpressionKind::Boolean
                | StaticExpressionKind::Number
                | StaticExpressionKind::String
                | StaticExpressionKind::Temporal
                | StaticExpressionKind::Duration
        ),
        "labels" => kind != StaticExpressionKind::Node,
        "type" => kind != StaticExpressionKind::Relationship,
        _ => return None,
    };
    invalid.then_some(match name {
        "toboolean" => "InvalidArgumentValue: toBoolean requires a BOOLEAN or STRING",
        "tointeger" => {
            "InvalidArgumentValue: toInteger requires a BOOLEAN, INTEGER, FLOAT, or STRING"
        }
        "tofloat" => "InvalidArgumentValue: toFloat requires an INTEGER, FLOAT, or STRING",
        "tostring" => "InvalidArgumentValue: toString received an unsupported value type",
        "labels" => "InvalidArgumentValue: labels requires a node",
        "type" => "InvalidArgumentValue: type requires a relationship",
        _ => unreachable!("contract presence was checked above"),
    })
}

fn static_modulo_result_kind(
    left: StaticExpressionKind,
    right: StaticExpressionKind,
) -> Result<StaticExpressionKind> {
    match (left, right) {
        (StaticExpressionKind::Null, _) | (_, StaticExpressionKind::Null) => {
            Ok(StaticExpressionKind::Null)
        }
        (StaticExpressionKind::Unknown, _) | (_, StaticExpressionKind::Unknown) => {
            Ok(StaticExpressionKind::Unknown)
        }
        (StaticExpressionKind::Number, StaticExpressionKind::Number) => {
            Ok(StaticExpressionKind::Number)
        }
        _ => Err(Error::new(
            ErrorCode::QuerySyntax,
            "InvalidArgumentType: modulo requires numeric operands",
        )),
    }
}

fn require_static_boolean(kind: StaticExpressionKind) -> Result<()> {
    if kind.is_definitely_non_boolean() {
        Err(Error::new(
            ErrorCode::QuerySyntax,
            "InvalidArgumentType: predicate expression must produce BOOLEAN or NULL",
        ))
    } else {
        Ok(())
    }
}

fn restore_static_type(
    scope: &mut StaticTypeScope,
    variable: &str,
    prior: Option<StaticExpressionKind>,
) {
    if let Some(kind) = prior {
        scope.insert(variable.to_owned(), kind);
    } else {
        scope.remove(variable);
    }
}

fn invalid_boolean_expression() -> Error {
    Error::new(
        ErrorCode::QuerySyntax,
        "InvalidArgumentType: logical expressions require BOOLEAN or NULL operands",
    )
}

const TEMPORAL_CLOCK_FUNCTIONS: &[&str] = &[
    "date.realtime",
    "date.statement",
    "date.transaction",
    "datetime.realtime",
    "datetime.statement",
    "datetime.transaction",
    "localdatetime.realtime",
    "localdatetime.statement",
    "localdatetime.transaction",
    "localtime.realtime",
    "localtime.statement",
    "localtime.transaction",
    "time.realtime",
    "time.statement",
    "time.transaction",
];

fn is_temporal_clock_function(name: &str) -> bool {
    TEMPORAL_CLOCK_FUNCTIONS.contains(&name)
}

fn is_supported_builtin_function(name: &str) -> bool {
    BUILTIN_FUNCTIONS.contains(&name) || is_temporal_clock_function(name)
}

/// Clock-qualified temporal functions accept no argument or one timezone string. NULL and every
/// unresolved expression remain legal because temporal functions propagate NULL and runtime
/// values may still be valid timezone strings.
fn validate_temporal_clock_call(name: &str, arguments: &[Expression]) -> Result<()> {
    if !is_temporal_clock_function(name) {
        return Ok(());
    }
    match arguments {
        [] => Ok(()),
        [argument] if statically_invalid_timezone_argument(argument) => Err(Error::new(
            ErrorCode::QuerySyntax,
            format!(
                "InvalidArgumentType: temporal clock function `{name}` requires a STRING timezone or NULL"
            ),
        )),
        [_] => Ok(()),
        _ => Err(Error::new(
            ErrorCode::QuerySyntax,
            format!(
                "InvalidArgumentCount: temporal clock function `{name}` accepts zero or one timezone argument"
            ),
        )),
    }
}

fn statically_invalid_timezone_argument(expression: &Expression) -> bool {
    match expression {
        Expression::Literal(ScalarValue::String(_) | ScalarValue::Null) => false,
        Expression::Literal(_) | Expression::List(_) | Expression::Map(_) => true,
        _ => false,
    }
}

const BUILTIN_FUNCTIONS: &[&str] = &[
    "abs",
    "acos",
    "asin",
    "atan",
    "atan2",
    "avg",
    "ceil",
    "coalesce",
    "collect",
    "cos",
    "cot",
    "count",
    "date",
    "date.truncate",
    "datetime",
    "datetime.fromepoch",
    "datetime.fromepochmillis",
    "datetime.truncate",
    "degrees",
    "duration",
    "duration.between",
    "duration.indays",
    "duration.inmonths",
    "duration.inseconds",
    "e",
    "endnode",
    "exp",
    "floor",
    "head",
    "id",
    "keys",
    "labels",
    "last",
    "left",
    "length",
    "isempty",
    "localdatetime",
    "localdatetime.truncate",
    "localtime",
    "localtime.truncate",
    "log",
    "log10",
    "lower",
    "ltrim",
    "max",
    "min",
    "nodes",
    "percentilecont",
    "percentiledisc",
    "pi",
    "power",
    "properties",
    "rand",
    "radians",
    "range",
    "relationships",
    "replace",
    "reverse",
    "revision",
    "right",
    "round",
    "rtrim",
    "sign",
    "sin",
    "size",
    "split",
    "sqrt",
    "startnode",
    "stdev",
    "stdevp",
    "substring",
    "sum",
    "tail",
    "tan",
    "time",
    "time.truncate",
    "toboolean",
    "tofloat",
    "tointeger",
    "tolist",
    "tolower",
    "tostring",
    "trim",
    "type",
    "upper",
    "variance",
    "variancep",
    "vector.cosine",
    "vector.distance",
    "vector.dot",
    "vector.normalize",
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cypher::parser::parse;

    fn bind_source(source: &str, require_native_execution: bool) -> Result<BoundQuery> {
        bind(
            parse(source)?,
            &NameCatalog::default(),
            BindCapabilities {
                require_native_execution,
                ..BindCapabilities::default()
            },
        )
    }

    fn bind_writable_source(source: &str, require_native_execution: bool) -> Result<BoundQuery> {
        bind(
            parse(source)?,
            &NameCatalog::default(),
            BindCapabilities {
                write: true,
                require_native_execution,
                ..BindCapabilities::default()
            },
        )
    }

    #[test]
    fn read_only_query_may_scope_reads_to_a_single_layer() -> Result<()> {
        // The Graph UI can view the Knowledge layer in isolation: a read-only query scopes its reads
        // without tripping the write-layer-visibility rule, even though the default write is Observed.
        let bound = bind_source("USE LAYER KNOWLEDGE MATCH (n) RETURN n", false)?;
        assert!(bound.read_only, "MATCH ... RETURN is read-only");
        assert!(bound.query.read_layers.contains_layer(Layer::Knowledge));
        assert!(!bound.query.read_layers.contains_layer(Layer::Observed));
        Ok(())
    }

    #[test]
    fn a_write_still_requires_its_write_layer_to_be_readable() {
        // Writing to the default Observed layer while only reading Knowledge is inconsistent: rejected.
        let error = bind_writable_source("USE LAYER KNOWLEDGE CREATE (n:Thing)", false)
            .expect_err("write to a layer outside the read mask must be rejected");
        assert_eq!(error.code, ErrorCode::LayerNotAllowed);
    }

    #[test]
    fn workspace_writes_require_the_workspace_capability() {
        // The internal Workspace layer is never writable by ordinary user
        // queries: without the internal-only capability, a WORKSPACE write is rejected.
        let error = bind_writable_source(
            "USE LAYER WORKSPACE WRITE LAYER WORKSPACE CREATE (n:Thing)",
            false,
        )
        .expect_err("workspace write without the capability must be rejected");
        assert_eq!(error.code, ErrorCode::LayerNotAllowed);
    }

    #[test]
    fn workspace_writes_are_allowed_with_the_capability() -> Result<()> {
        let bound = bind(
            parse("USE LAYER WORKSPACE WRITE LAYER WORKSPACE CREATE (n:Thing)")?,
            &NameCatalog::default(),
            BindCapabilities {
                write: true,
                workspace_write: true,
                ..BindCapabilities::default()
            },
        )?;
        assert_eq!(bound.query.write_layer, Layer::Workspace);
        Ok(())
    }

    fn assert_native_deleted_return_error(query: &str, entity: &str) -> Result<()> {
        let error = bind_writable_source(query, true)
            .expect_err("strict native binding lost a deterministic deleted-entity error");
        assert_eq!(error.code, ErrorCode::QueryType, "query: {query}");
        assert_eq!(
            error.message,
            format!("DeletedEntityAccess: {entity} was deleted in this query"),
            "query: {query}"
        );

        bind_writable_source(query, false).map_err(|error| {
            Error::internal(format!(
                "ordinary binding moved runtime deleted-entity error to compilation for `{query}`: {error}"
            ))
        })?;
        Ok(())
    }

    fn assert_native_percentile_range_error(query: &str) -> Result<()> {
        let error = bind_source(query, true)
            .expect_err("strict native binding lost a proven percentile range error");
        assert_eq!(error.code, ErrorCode::QueryType, "query: {query}");
        assert_eq!(
            error.message, "NumberOutOfRange: percentile must be finite and in 0..=1",
            "query: {query}"
        );

        bind_source(query, false).map_err(|error| {
            Error::internal(format!(
                "ordinary binding moved runtime percentile error to compilation for `{query}`: {error}"
            ))
        })?;
        Ok(())
    }

    fn conversion_cases() -> Vec<(String, &'static str)> {
        let mut cases = Vec::new();
        for invalid in ["[]", "{}", "1.0", "n", "r", "p"] {
            cases.push((
                format!(
                    "MATCH p = (n)-[r:T]->() RETURN [x IN [true, {invalid}] | toBoolean(x)] AS list"
                ),
                "InvalidArgumentValue: toBoolean requires a BOOLEAN or STRING",
            ));
        }
        for invalid in ["[]", "{}", "n", "r", "p"] {
            cases.push((
                format!(
                    "MATCH p = (n)-[r:T]->() RETURN [x IN [1, {invalid}] | toInteger(x)] AS list"
                ),
                "InvalidArgumentValue: toInteger requires a BOOLEAN, INTEGER, FLOAT, or STRING",
            ));
        }
        for invalid in ["true", "[]", "{}", "n", "r", "p"] {
            cases.push((
                format!(
                    "MATCH p = (n)-[r:T]->() RETURN [x IN [1.0, {invalid}] | toFloat(x)] AS list"
                ),
                "InvalidArgumentValue: toFloat requires an INTEGER, FLOAT, or STRING",
            ));
        }
        for invalid in ["[]", "{}", "n", "r", "p"] {
            cases.push((
                format!(
                    "MATCH p = (n)-[r:T]->() RETURN [x IN [1, '', {invalid}] | toString(x)] AS list"
                ),
                "InvalidArgumentValue: toString received an unsupported value type",
            ));
        }
        cases
    }

    #[test]
    fn native_list_conversion_preflight_preserves_all_22_runtime_type_errors() -> Result<()> {
        let cases = conversion_cases();
        assert_eq!(cases.len(), 22);
        for (query, expected) in cases {
            let error = bind_source(&query, true)
                .expect_err("strict native binding lost a deterministic conversion error");
            assert_eq!(error.code, ErrorCode::QueryType, "query: {query}");
            assert_eq!(error.message, expected, "query: {query}");
        }
        Ok(())
    }

    #[test]
    fn native_list_projection_preflight_preserves_graph_type_errors() -> Result<()> {
        for invalid in ["0", "1.0", "true", "''", "[]"] {
            let query =
                format!("MATCH p = (n)-[r:T]->() RETURN [x IN [r, {invalid}] | type(x)] AS list");
            let error = bind_source(&query, true)
                .expect_err("strict native binding lost a deterministic type() error");
            assert_eq!(error.code, ErrorCode::QueryType, "query: {query}");
            assert_eq!(
                error.message, "InvalidArgumentValue: type requires a relationship",
                "query: {query}"
            );
        }
        Ok(())
    }

    #[test]
    fn ordinary_binding_keeps_conversion_failures_on_the_cpu_runtime_path() -> Result<()> {
        for (query, _) in conversion_cases() {
            bind_source(&query, false).map_err(|error| {
                Error::internal(format!(
                    "ordinary binding eagerly rejected runtime conversion `{query}`: {error}"
                ))
            })?;
        }
        Ok(())
    }

    #[test]
    fn native_preflight_defers_unknown_filtered_and_guarded_values() -> Result<()> {
        for query in [
            "MATCH p = (n)-[r:T]->() RETURN [x IN [true, $value] | toBoolean(x)] AS list",
            "MATCH p = (n)-[r:T]->() RETURN [x IN $values | toInteger(x)] AS list",
            "MATCH p = (n)-[r:T]->() RETURN [x IN [true, []] WHERE false | toBoolean(x)] AS list",
            "MATCH p = (n)-[r:T]->() RETURN [x IN [true, []] | CASE WHEN x = [] THEN null ELSE toBoolean(x) END] AS list",
            "MATCH p = (n)-[r:T]->() WITH [true, []] AS values RETURN [x IN values | toBoolean(x)] AS list",
            "MATCH p = (n)-[r:T]->() RETURN [x IN [true, 'false', null] | toBoolean(x)] AS list",
            "MATCH p = (n)-[r:T]->() RETURN [x IN [true, 1, 1.0, '2', null] | toInteger(x)] AS list",
            "MATCH p = (n)-[r:T]->() RETURN [x IN [1, 1.0, '2', null] | toFloat(x)] AS list",
            "MATCH p = (n)-[r:T]->() RETURN [x IN [true, 1, 1.0, 'text', null] | toString(x)] AS list",
            "MATCH p = (n)-[r:T]->() RETURN [x IN [r, null] | type(x)] AS list",
            "MATCH p = (n)-[r:T]->() RETURN [x IN [n, null] | labels(x)] AS list",
        ] {
            bind_source(query, true).map_err(|error| {
                Error::internal(format!(
                    "strict native preflight rejected a deferred or valid expression `{query}`: {error}"
                ))
            })?;
        }
        Ok(())
    }

    #[test]
    fn native_deleted_node_property_preserves_return2_15_runtime_error() -> Result<()> {
        assert_native_deleted_return_error("MATCH (n) DELETE n RETURN n.num", "node")
    }

    #[test]
    fn native_deleted_node_labels_preserve_return2_16_runtime_error() -> Result<()> {
        assert_native_deleted_return_error("MATCH (n) DELETE n RETURN labels(n)", "node")
    }

    #[test]
    fn native_deleted_relationship_property_preserves_return2_17_runtime_error() -> Result<()> {
        assert_native_deleted_return_error("MATCH ()-[r]->() DELETE r RETURN r.num", "relationship")
    }

    #[test]
    fn native_deleted_entity_preflight_stays_clause_local_and_direct() -> Result<()> {
        for query in [
            "MATCH ()-[r]->() DELETE r RETURN type(r)",
            "MATCH (n) DELETE n RETURN $value",
            "MATCH (n) DELETE n RETURN n[$property]",
            "MATCH (n) DELETE n RETURN properties(n)",
            "MATCH (n) DELETE n RETURN CASE WHEN $read THEN n.num ELSE null END AS value",
            "MATCH (n) DELETE n WITH n AS kept RETURN kept.num",
            "MATCH (n) WHERE false DELETE n RETURN n.num",
            "OPTIONAL MATCH (n:Missing) DELETE n RETURN n.num",
            "UNWIND $values AS n MATCH (n) DELETE n RETURN n.num",
            "WITH $entity AS n DELETE n RETURN n.num",
            "WITH null AS n DELETE n RETURN n.num",
        ] {
            bind_writable_source(query, true).map_err(|error| {
                Error::internal(format!(
                    "strict native deleted-entity preflight rejected legal/deferred `{query}`: {error}"
                ))
            })?;
        }
        Ok(())
    }

    #[test]
    fn native_percentile_range_preserves_aggregation6_5_runtime_error() -> Result<()> {
        assert_native_percentile_range_error(
            "MATCH (n:S) WITH n, size([(n)-->() | 1]) AS deg WHERE deg > 2 WITH deg LIMIT 100 RETURN percentileDisc(0.90, deg), deg",
        )
    }

    #[test]
    fn native_percentile_range_uses_the_shared_continuous_error_contract() -> Result<()> {
        assert_native_percentile_range_error(
            "MATCH (n:S) WITH n, size([(n)-->() | 1]) AS deg WHERE 2 < deg WITH deg LIMIT 100 RETURN percentileCont(0.90, deg), deg",
        )
    }

    #[test]
    fn native_percentile_range_preflight_defers_unproven_dynamic_shapes() -> Result<()> {
        for query in [
            "OPTIONAL MATCH (n:S) WITH n, size([(n)-->() | 1]) AS deg WHERE deg > 2 WITH deg LIMIT 100 RETURN percentileDisc(0.90, deg), deg",
            "MATCH (n:S) WITH n, size([(n)-->() | 1]) AS deg WHERE deg > $minimum WITH deg LIMIT 100 RETURN percentileDisc(0.90, deg), deg",
            "MATCH (n:S) WITH n, size([(n)-->() | 1]) AS deg WHERE deg > 0 WITH deg LIMIT 100 RETURN percentileDisc(0.90, deg), deg",
            "MATCH (n:S) WITH n, size([(n)-->() | 1]) AS deg WHERE deg >= 1 WITH deg LIMIT 100 RETURN percentileDisc(0.90, deg), deg",
            "MATCH (n:S) WITH n, size([(n)-->() | 1]) AS deg WHERE deg < 0 WITH deg LIMIT 100 RETURN percentileDisc(0.90, deg), deg",
            "MATCH (n:S) WITH n, size([(n)-->() | 1]) AS deg WHERE deg > 2 AND $enabled WITH deg LIMIT 100 RETURN percentileDisc(0.90, deg), deg",
            "MATCH (n:S) WITH n, size([(n)-->() | 1]) AS deg WHERE deg > 2 WITH deg LIMIT 0 RETURN percentileDisc(0.90, deg), deg",
            "MATCH (n:S) WITH n, size([(n)-->() | 1]) AS deg WHERE deg > 2 WITH deg LIMIT $limit RETURN percentileDisc(0.90, deg), deg",
            "MATCH (n:S) WITH n, size([(n)-->() | 1]) AS deg WHERE deg > 2 WITH deg + 0 AS deg LIMIT 100 RETURN percentileDisc(0.90, deg), deg",
            "MATCH (n:S) WITH n, size([(n)-->() | 1]) AS deg WHERE deg > 2 WITH deg LIMIT 100 RETURN percentileDisc(deg, 0.90), deg",
            "MATCH (n:S) WITH n, size([(n)-->() | 1]) AS deg WHERE deg > 2 WITH deg LIMIT 100 RETURN percentileDisc(0.90, $percentile), deg",
            "MATCH (n:S) WITH n, $degree AS deg WHERE deg > 2 WITH deg LIMIT 100 RETURN percentileDisc(0.90, deg), deg",
        ] {
            bind_source(query, true).map_err(|error| {
                Error::internal(format!(
                    "strict native percentile preflight rejected deferred `{query}`: {error}"
                ))
            })?;
        }
        Ok(())
    }

    #[test]
    fn native_set_property_preflight_preserves_set1_10_runtime_type_error() -> Result<()> {
        let query = "CREATE (a) SET a.maplist = [{num: 1}]";
        let error = bind_writable_source(query, true)
            .expect_err("strict native binding lost Set1 [10]'s property type error");
        assert_eq!(error.code, ErrorCode::QueryType);
        assert_eq!(
            error.message,
            "InvalidPropertyType: property lists must be flat homogeneous lists of scalar values"
        );

        bind_writable_source(query, false).map_err(|error| {
            Error::internal(format!(
                "ordinary binding eagerly rejected Set1 [10]'s runtime error: {error}"
            ))
        })?;
        Ok(())
    }

    #[test]
    fn native_set_property_preflight_preserves_parameters_null_and_scalar_lists() -> Result<()> {
        for query in [
            "CREATE (a) SET a.value = null",
            "CREATE (a) SET a.value = $value",
            "CREATE (a) SET a.values = []",
            "CREATE (a) SET a.values = [1, 2, 3]",
            "CREATE (a) SET a.values = ['a', 'b']",
            "OPTIONAL MATCH (a:Missing) SET a.value = null RETURN a",
        ] {
            bind_writable_source(query, true).map_err(|error| {
                Error::internal(format!(
                    "strict native property preflight rejected legal/deferred `{query}`: {error}"
                ))
            })?;
        }
        Ok(())
    }

    #[test]
    fn native_index_provenance_preserves_graph3_9_runtime_type_error() -> Result<()> {
        let query = "MATCH (a) WITH [a, 1] AS list RETURN labels(list[1]) AS l";
        let error = bind_source(query, true)
            .expect_err("strict native binding lost Graph3 [9]'s labels() type error");
        assert_eq!(error.code, ErrorCode::QueryType);
        assert_eq!(
            error.message,
            "InvalidArgumentValue: labels requires a node"
        );

        bind_source(query, false).map_err(|error| {
            Error::internal(format!(
                "ordinary binding eagerly rejected Graph3 [9]'s runtime error: {error}"
            ))
        })?;
        Ok(())
    }

    #[test]
    fn indexed_list_provenance_survives_with_and_defers_dynamic_indexes() -> Result<()> {
        let forwarded =
            "MATCH (a) WITH [a, 1] AS list WITH list AS values RETURN labels(values[1]) AS l";
        let error = bind_source(forwarded, true)
            .expect_err("WITH forwarding discarded exact indexed-list provenance");
        assert_eq!(error.code, ErrorCode::QueryType);
        assert_eq!(
            error.message,
            "InvalidArgumentValue: labels requires a node"
        );

        for query in [
            "MATCH (a) WITH [a, 1] AS list RETURN labels(list[0]) AS l",
            "MATCH (a) WITH [a, 1] AS list RETURN labels(list[-2]) AS l",
            "MATCH (a) WITH [a, 1] AS list RETURN labels(list[2]) AS l",
            "MATCH (a) WITH [a, 1] AS list RETURN labels(list[$index]) AS l",
            "MATCH (a) WITH $values AS list RETURN labels(list[1]) AS l",
        ] {
            bind_source(query, true).map_err(|error| {
                Error::internal(format!(
                    "indexed-list provenance rejected legal/deferred `{query}`: {error}"
                ))
            })?;
        }
        Ok(())
    }
}
