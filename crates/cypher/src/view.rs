//! Borrowed canonical graph/temporal reads with sparse statement-local write overlays.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    EdgeId, Error, ErrorCode, Layer, NodeId, Result, ScalarValue,
    graph::{
        Csr, EdgeInput, EdgeView, GraphMutation, GraphStore, LayerMask, NodeInput, NodeView,
        TemporalSample, TemporalStore,
    },
    types::{EntityKind, LabelId, PropertyId, RelationshipTypeId},
};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScalarKind {
    Boolean,
    Integer,
    Float,
    String,
    Bytes,
    Date,
    LocalTime,
    ZonedTime,
    LocalDateTime,
    ZonedDateTime,
    Duration,
    List,
    Map,
    Mixed,
}

impl ScalarKind {
    fn of(value: &ScalarValue) -> Option<Self> {
        match value {
            ScalarValue::Null => None,
            ScalarValue::Boolean(_) => Some(Self::Boolean),
            ScalarValue::Integer(_) => Some(Self::Integer),
            ScalarValue::Float(_) => Some(Self::Float),
            ScalarValue::String(_) => Some(Self::String),
            ScalarValue::Bytes(_) => Some(Self::Bytes),
            ScalarValue::Date(_) => Some(Self::Date),
            ScalarValue::LocalTime(_) => Some(Self::LocalTime),
            ScalarValue::ZonedTime { .. } => Some(Self::ZonedTime),
            ScalarValue::LocalDateTime { .. } => Some(Self::LocalDateTime),
            ScalarValue::ZonedDateTime { .. } => Some(Self::ZonedDateTime),
            ScalarValue::Duration { .. } => Some(Self::Duration),
            ScalarValue::List(_) => Some(Self::List),
            ScalarValue::Map(_) => Some(Self::Map),
        }
    }
}

#[derive(Debug)]
struct CatalogOverlay<'a> {
    base: &'a crate::graph::NameCatalog,
    labels: BTreeMap<String, LabelId>,
    label_names: BTreeMap<LabelId, String>,
    properties: BTreeMap<String, PropertyId>,
    property_names: BTreeMap<PropertyId, String>,
    relationship_types: BTreeMap<String, RelationshipTypeId>,
    relationship_names: BTreeMap<RelationshipTypeId, String>,
    next_label: u64,
    next_property: u64,
    next_relationship_type: u64,
}

impl<'a> CatalogOverlay<'a> {
    fn new(base: &'a crate::graph::NameCatalog) -> Self {
        Self {
            base,
            labels: BTreeMap::new(),
            label_names: BTreeMap::new(),
            properties: BTreeMap::new(),
            property_names: BTreeMap::new(),
            relationship_types: BTreeMap::new(),
            relationship_names: BTreeMap::new(),
            next_label: base.next_label_id(),
            next_property: base.next_property_id(),
            next_relationship_type: base.next_relationship_type_id(),
        }
    }

    fn label(&self, name: &str) -> Option<LabelId> {
        self.labels
            .get(name)
            .copied()
            .or_else(|| self.base.label(name))
    }

    fn property(&self, name: &str) -> Option<PropertyId> {
        self.properties
            .get(name)
            .copied()
            .or_else(|| self.base.property(name))
    }

    fn relationship_type(&self, name: &str) -> Option<RelationshipTypeId> {
        self.relationship_types
            .get(name)
            .copied()
            .or_else(|| self.base.relationship_type(name))
    }

    fn label_name(&self, id: LabelId) -> Option<&str> {
        self.label_names
            .get(&id)
            .map(String::as_str)
            .or_else(|| self.base.label_name(id))
    }

    fn property_name(&self, id: PropertyId) -> Option<&str> {
        self.property_names
            .get(&id)
            .map(String::as_str)
            .or_else(|| self.base.property_name(id))
    }

    fn relationship_type_name(&self, id: RelationshipTypeId) -> Option<&str> {
        self.relationship_names
            .get(&id)
            .map(String::as_str)
            .or_else(|| self.base.relationship_type_name(id))
    }

    fn intern_label(&mut self, name: &str) -> Result<LabelId> {
        if let Some(id) = self.label(name) {
            return Ok(id);
        }
        let id = LabelId(self.next_label);
        self.next_label = self.next_label.checked_add(1).ok_or_else(|| {
            Error::new(ErrorCode::ResultBudgetExceeded, "label ID space exhausted")
        })?;
        self.labels.insert(name.to_owned(), id);
        self.label_names.insert(id, name.to_owned());
        Ok(id)
    }

    fn intern_property(&mut self, name: &str) -> Result<PropertyId> {
        if let Some(id) = self.property(name) {
            return Ok(id);
        }
        let id = PropertyId(self.next_property);
        self.next_property = self.next_property.checked_add(1).ok_or_else(|| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "property ID space exhausted",
            )
        })?;
        self.properties.insert(name.to_owned(), id);
        self.property_names.insert(id, name.to_owned());
        Ok(id)
    }

    fn intern_relationship_type(&mut self, name: &str) -> Result<RelationshipTypeId> {
        if let Some(id) = self.relationship_type(name) {
            return Ok(id);
        }
        let id = RelationshipTypeId(self.next_relationship_type);
        self.next_relationship_type =
            self.next_relationship_type.checked_add(1).ok_or_else(|| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "relationship type ID space exhausted",
                )
            })?;
        self.relationship_types.insert(name.to_owned(), id);
        self.relationship_names.insert(id, name.to_owned());
        Ok(id)
    }

    fn declare_label(&mut self, name: &str, id: LabelId) -> Result<()> {
        if self.label(name).is_some_and(|existing| existing != id)
            || self.label_name(id).is_some_and(|existing| existing != name)
        {
            return Err(Error::new(
                ErrorCode::TransactionConflict,
                "label name or ID is already assigned",
            ));
        }
        self.next_label = self.next_label.max(id.0.checked_add(1).ok_or_else(|| {
            Error::new(ErrorCode::ResultBudgetExceeded, "label ID space exhausted")
        })?);
        self.labels.insert(name.to_owned(), id);
        self.label_names.insert(id, name.to_owned());
        Ok(())
    }

    fn declare_property(&mut self, name: &str, id: PropertyId) -> Result<()> {
        if self.property(name).is_some_and(|existing| existing != id)
            || self
                .property_name(id)
                .is_some_and(|existing| existing != name)
        {
            return Err(Error::new(
                ErrorCode::TransactionConflict,
                "property name or ID is already assigned",
            ));
        }
        self.next_property = self.next_property.max(id.0.checked_add(1).ok_or_else(|| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "property ID space exhausted",
            )
        })?);
        self.properties.insert(name.to_owned(), id);
        self.property_names.insert(id, name.to_owned());
        Ok(())
    }

    fn declare_relationship_type(&mut self, name: &str, id: RelationshipTypeId) -> Result<()> {
        if self
            .relationship_type(name)
            .is_some_and(|existing| existing != id)
            || self
                .relationship_type_name(id)
                .is_some_and(|existing| existing != name)
        {
            return Err(Error::new(
                ErrorCode::TransactionConflict,
                "relationship type name or ID is already assigned",
            ));
        }
        self.next_relationship_type =
            self.next_relationship_type
                .max(id.0.checked_add(1).ok_or_else(|| {
                    Error::new(
                        ErrorCode::ResultBudgetExceeded,
                        "relationship type ID space exhausted",
                    )
                })?);
        self.relationship_types.insert(name.to_owned(), id);
        self.relationship_names.insert(id, name.to_owned());
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct OverlayNode {
    dense: u32,
    id: NodeId,
    layer: Layer,
    revision: u64,
    labels: Vec<LabelId>,
    properties: BTreeMap<PropertyId, ScalarValue>,
    deleted: bool,
}

#[derive(Clone, Debug)]
pub struct OverlayEdge {
    dense: u32,
    id: EdgeId,
    source: NodeId,
    target: NodeId,
    relationship_type: RelationshipTypeId,
    layer: Layer,
    revision: u64,
    properties: BTreeMap<PropertyId, ScalarValue>,
    deleted: bool,
}

#[derive(Clone, Copy, Debug)]
pub enum NodeReadView<'a> {
    Base(NodeView<'a>),
    Overlay(&'a OverlayNode),
}

impl<'a> NodeReadView<'a> {
    pub fn id(self) -> NodeId {
        match self {
            Self::Base(node) => node.id(),
            Self::Overlay(node) => node.id,
        }
    }

    pub fn dense(self) -> u32 {
        match self {
            Self::Base(node) => node.dense(),
            Self::Overlay(node) => node.dense,
        }
    }

    pub fn layer(self) -> Layer {
        match self {
            Self::Base(node) => node.layer(),
            Self::Overlay(node) => node.layer,
        }
    }

    pub fn revision(self) -> u64 {
        match self {
            Self::Base(node) => node.revision(),
            Self::Overlay(node) => node.revision,
        }
    }

    pub fn labels(self) -> &'a [LabelId] {
        match self {
            Self::Base(node) => node.labels(),
            Self::Overlay(node) => &node.labels,
        }
    }

    pub fn property(self, property: PropertyId) -> Option<ScalarValue> {
        match self {
            Self::Base(node) => node.property(property),
            Self::Overlay(node) => node.properties.get(&property).cloned(),
        }
    }

    pub fn properties(self) -> Vec<(PropertyId, ScalarValue)> {
        match self {
            Self::Base(node) => node.properties(),
            Self::Overlay(node) => node
                .properties
                .iter()
                .map(|(property, value)| (*property, value.clone()))
                .collect(),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum EdgeReadView<'a> {
    Base(EdgeView<'a>),
    Overlay(&'a OverlayEdge),
}

impl EdgeReadView<'_> {
    pub fn id(self) -> EdgeId {
        match self {
            Self::Base(edge) => edge.id(),
            Self::Overlay(edge) => edge.id,
        }
    }

    pub fn dense(self) -> u32 {
        match self {
            Self::Base(edge) => edge.dense(),
            Self::Overlay(edge) => edge.dense,
        }
    }

    pub fn source(self) -> NodeId {
        match self {
            Self::Base(edge) => edge.source(),
            Self::Overlay(edge) => edge.source,
        }
    }

    pub fn target(self) -> NodeId {
        match self {
            Self::Base(edge) => edge.target(),
            Self::Overlay(edge) => edge.target,
        }
    }

    pub fn relationship_type(self) -> RelationshipTypeId {
        match self {
            Self::Base(edge) => edge.relationship_type(),
            Self::Overlay(edge) => edge.relationship_type,
        }
    }

    pub fn layer(self) -> Layer {
        match self {
            Self::Base(edge) => edge.layer(),
            Self::Overlay(edge) => edge.layer,
        }
    }

    pub fn revision(self) -> u64 {
        match self {
            Self::Base(edge) => edge.revision(),
            Self::Overlay(edge) => edge.revision,
        }
    }

    pub fn property(self, property: PropertyId) -> Option<ScalarValue> {
        match self {
            Self::Base(edge) => edge.property(property),
            Self::Overlay(edge) => edge.properties.get(&property).cloned(),
        }
    }

    pub fn properties(self) -> Vec<(PropertyId, ScalarValue)> {
        match self {
            Self::Base(edge) => edge.properties(),
            Self::Overlay(edge) => edge
                .properties
                .iter()
                .map(|(property, value)| (*property, value.clone()))
                .collect(),
        }
    }
}

/// One statement reads canonical columns by reference and materializes only rows it changes.
pub struct GraphReadView<'a> {
    base: &'a GraphStore,
    catalog: CatalogOverlay<'a>,
    nodes: BTreeMap<u32, OverlayNode>,
    node_ids: BTreeMap<NodeId, u32>,
    edges: BTreeMap<u32, OverlayEdge>,
    edge_ids: BTreeMap<EdgeId, u32>,
    inserted_outgoing: BTreeMap<NodeId, BTreeSet<u32>>,
    inserted_incoming: BTreeMap<NodeId, BTreeSet<u32>>,
    node_property_types: BTreeMap<PropertyId, ScalarKind>,
    edge_property_types: BTreeMap<PropertyId, ScalarKind>,
    // O(1) counts of overlay-*inserted* (dense >= base slot count) nodes/edges. Maintained on
    // insert so the hot bulk-CREATE path never rescans the whole overlay to allocate the next dense
    // id — that scan made a single UNWIND ... CREATE of N rows O(N^2).
    inserted_node_total: usize,
    inserted_edge_total: usize,
    revision: u64,
}

impl<'a> GraphReadView<'a> {
    pub fn new(base: &'a GraphStore) -> Self {
        Self {
            base,
            catalog: CatalogOverlay::new(base.catalog()),
            nodes: BTreeMap::new(),
            node_ids: BTreeMap::new(),
            edges: BTreeMap::new(),
            edge_ids: BTreeMap::new(),
            inserted_outgoing: BTreeMap::new(),
            inserted_incoming: BTreeMap::new(),
            node_property_types: BTreeMap::new(),
            edge_property_types: BTreeMap::new(),
            inserted_node_total: 0,
            inserted_edge_total: 0,
            revision: base.revision(),
        }
    }

    pub fn label(&self, name: &str) -> Option<LabelId> {
        self.catalog.label(name)
    }

    pub fn property(&self, name: &str) -> Option<PropertyId> {
        self.catalog.property(name)
    }

    pub fn relationship_type(&self, name: &str) -> Option<RelationshipTypeId> {
        self.catalog.relationship_type(name)
    }

    pub fn revision(&self) -> u64 {
        self.revision
    }

    pub fn label_name(&self, id: LabelId) -> Option<&str> {
        self.catalog.label_name(id)
    }

    pub fn property_name(&self, id: PropertyId) -> Option<&str> {
        self.catalog.property_name(id)
    }

    pub fn relationship_type_name(&self, id: RelationshipTypeId) -> Option<&str> {
        self.catalog.relationship_type_name(id)
    }

    pub fn intern_label(&mut self, name: &str) -> Result<LabelId> {
        self.catalog.intern_label(name)
    }

    pub fn intern_property(&mut self, name: &str) -> Result<PropertyId> {
        self.catalog.intern_property(name)
    }

    pub fn intern_relationship_type(&mut self, name: &str) -> Result<RelationshipTypeId> {
        self.catalog.intern_relationship_type(name)
    }

    pub fn contains_node_id(&self, id: NodeId) -> bool {
        self.node_ids.contains_key(&id) || self.base.contains_node_id(id)
    }

    pub fn contains_edge_id(&self, id: EdgeId) -> bool {
        self.edge_ids.contains_key(&id) || self.base.contains_edge_id(id)
    }

    pub fn node(&self, id: NodeId) -> Option<NodeReadView<'_>> {
        if let Some(dense) = self.node_ids.get(&id) {
            return self.node_dense(*dense);
        }
        let base = self.base.node(id)?;
        self.node_dense(base.dense())
    }

    pub fn node_dense(&self, dense: u32) -> Option<NodeReadView<'_>> {
        if let Some(node) = self.nodes.get(&dense) {
            return (!node.deleted).then_some(NodeReadView::Overlay(node));
        }
        self.base.node_dense(dense).map(NodeReadView::Base)
    }

    pub fn node_dense_was_deleted(&self, dense: u32) -> bool {
        self.nodes.get(&dense).is_some_and(|node| node.deleted)
    }

    pub fn edge(&self, id: EdgeId) -> Option<EdgeReadView<'_>> {
        if let Some(dense) = self.edge_ids.get(&id) {
            return self.edge_dense(*dense);
        }
        let base = self.base.edge(id)?;
        self.edge_dense(base.dense())
    }

    pub fn edge_dense(&self, dense: u32) -> Option<EdgeReadView<'_>> {
        if let Some(edge) = self.edges.get(&dense) {
            if edge.deleted || self.node(edge.source).is_none() || self.node(edge.target).is_none()
            {
                return None;
            }
            return Some(EdgeReadView::Overlay(edge));
        }
        let edge = self.base.edge_dense(dense)?;
        if self.node(edge.source()).is_none() || self.node(edge.target()).is_none() {
            return None;
        }
        Some(EdgeReadView::Base(edge))
    }

    pub fn deleted_edge_relationship_type(&self, dense: u32) -> Option<RelationshipTypeId> {
        self.edges
            .get(&dense)
            .filter(|edge| edge.deleted)
            .map(|edge| edge.relationship_type)
    }

    pub fn node_count(&self) -> usize {
        let deleted_base = self
            .nodes
            .values()
            .filter(|node| node.dense < self.base.node_slot_count() as u32 && node.deleted)
            .count();
        let inserted = self
            .nodes
            .values()
            .filter(|node| node.dense >= self.base.node_slot_count() as u32 && !node.deleted)
            .count();
        self.base
            .node_count()
            .saturating_sub(deleted_base)
            .saturating_add(inserted)
    }

    pub fn edge_count(&self) -> usize {
        let deleted_base = self
            .edges
            .values()
            .filter(|edge| edge.dense < self.base.edge_slot_count() as u32 && edge.deleted)
            .count();
        let inserted = self
            .edges
            .values()
            .filter(|edge| edge.dense >= self.base.edge_slot_count() as u32 && !edge.deleted)
            .count();
        self.base
            .edge_count()
            .saturating_sub(deleted_base)
            .saturating_add(inserted)
    }

    /// Dense ids of every node the overlay has touched (inserted, property-updated, or relabeled).
    /// Edge writes do NOT touch `self.nodes`, so during a pure edge-load this is empty. Callers that
    /// consult the base-graph scalar index union these denses in so same-statement node inserts and
    /// property updates are never missed while still letting the index carry the base rows.
    pub fn overlay_modified_node_denses(&self) -> impl Iterator<Item = u32> + '_ {
        self.nodes.keys().copied()
    }

    /// True when the overlay has modified at least one node (insert/update/relabel). False when only
    /// edges have been written, in which case the base node index is still exact.
    pub fn has_overlay_node_modifications(&self) -> bool {
        !self.nodes.is_empty()
    }

    pub fn scan_node_denses(&self, label: Option<LabelId>, layers: LayerMask) -> Vec<u32> {
        let capacity = self
            .base
            .node_slot_count()
            .saturating_add(self.inserted_node_count());
        (0..capacity)
            .filter_map(|row| {
                let dense = u32::try_from(row).ok()?;
                let node = self.node_dense(dense)?;
                (layers.contains_layer(node.layer())
                    && label.is_none_or(|label| node.labels().contains(&label)))
                .then_some(dense)
            })
            .collect()
    }

    /// Direct scan that stops after `cap` matches. Apply every required label and the layer mask
    /// before counting a match, so pagination cannot stop on rows a later label check would reject.
    /// The sweep may inspect unrelated slots, but materializes at most `cap` matching node ids.
    pub fn scan_node_denses_bounded(
        &self,
        labels: &[LabelId],
        layers: LayerMask,
        cap: usize,
    ) -> Vec<u32> {
        let capacity = self
            .base
            .node_slot_count()
            .saturating_add(self.inserted_node_count());
        (0..capacity)
            .filter_map(|row| {
                let dense = u32::try_from(row).ok()?;
                let node = self.node_dense(dense)?;
                (layers.contains_layer(node.layer())
                    && labels.iter().all(|label| node.labels().contains(label)))
                .then_some(dense)
            })
            .take(cap)
            .collect()
    }

    pub fn expand_out(
        &self,
        node: NodeId,
        relationship_type: Option<RelationshipTypeId>,
        layers: LayerMask,
    ) -> Result<Vec<(EdgeReadView<'_>, NodeReadView<'_>)>> {
        self.expand(node, relationship_type, layers, true)
    }

    pub fn expand_in(
        &self,
        node: NodeId,
        relationship_type: Option<RelationshipTypeId>,
        layers: LayerMask,
    ) -> Result<Vec<(EdgeReadView<'_>, NodeReadView<'_>)>> {
        self.expand(node, relationship_type, layers, false)
    }

    fn expand(
        &self,
        node: NodeId,
        relationship_type: Option<RelationshipTypeId>,
        layers: LayerMask,
        outgoing: bool,
    ) -> Result<Vec<(EdgeReadView<'_>, NodeReadView<'_>)>> {
        let start = self
            .node(node)
            .ok_or_else(|| Error::new(ErrorCode::QueryType, "node does not exist"))?;
        if !layers.contains_layer(start.layer()) {
            return Ok(Vec::new());
        }
        let mut denses = Vec::new();
        let canonical = if outgoing {
            self.base.expand_out(node, None, LayerMask::ALL)
        } else {
            self.base.expand_in(node, None, LayerMask::ALL)
        };
        match canonical {
            Ok(canonical) => {
                denses.extend(canonical.into_iter().map(|(edge, _)| edge.dense()));
            }
            Err(error) if self.base.node(node).is_some() => return Err(error),
            Err(_) => {}
        }
        let inserted = if outgoing {
            self.inserted_outgoing.get(&node)
        } else {
            self.inserted_incoming.get(&node)
        };
        if let Some(inserted) = inserted {
            denses.extend(inserted.iter().copied());
        }
        denses.sort_unstable();
        denses.dedup();
        let mut result = Vec::new();
        for dense in denses {
            let Some(edge) = self.edge_dense(dense) else {
                continue;
            };
            if !layers.contains_layer(edge.layer())
                || relationship_type.is_some_and(|kind| edge.relationship_type() != kind)
            {
                continue;
            }
            let neighbor_id = if outgoing {
                edge.target()
            } else {
                edge.source()
            };
            let Some(neighbor) = self.node(neighbor_id) else {
                continue;
            };
            if layers.contains_layer(neighbor.layer()) {
                result.push((edge, neighbor));
            }
        }
        result.sort_by_key(|(edge, node)| (node.id(), edge.id()));
        Ok(result)
    }

    pub fn algorithm_adjacency(&self, layers: LayerMask) -> Result<(Csr, Csr, Vec<u32>, Vec<u32>)> {
        let node_capacity = self
            .base
            .node_slot_count()
            .saturating_add(self.inserted_node_count());
        let visible_nodes = (0..node_capacity)
            .filter_map(|row| {
                let dense = u32::try_from(row).ok()?;
                let node = self.node_dense(dense)?;
                layers.contains_layer(node.layer()).then_some(dense)
            })
            .collect::<Vec<_>>();
        // Dense IDs are already bounded integer indices. A tree map here made preparation
        // O(N log N), and the executor then rebuilt the same map a second time. This one compact
        // vector is produced with the filtered CSR and reused by argument resolution; unrelated
        // property payloads do not affect its shape or work.
        let mut ordinal_by_dense = vec![u32::MAX; node_capacity];
        for (ordinal, dense) in visible_nodes.iter().copied().enumerate() {
            ordinal_by_dense[dense as usize] = u32::try_from(ordinal).map_err(|_| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "algorithm node ordinal space exhausted",
                )
            })?;
        }
        let mut visible = Vec::new();
        let capacity = self
            .base
            .edge_slot_count()
            .saturating_add(self.inserted_edge_count());
        for row in 0..capacity {
            let dense = u32::try_from(row).map_err(|_| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "dense relationship ID space exhausted",
                )
            })?;
            let Some(edge) = self.edge_dense(dense) else {
                continue;
            };
            if !layers.contains_layer(edge.layer()) {
                continue;
            }
            let Some(source) = self
                .node(edge.source())
                .and_then(|node| ordinal_by_dense.get(node.dense() as usize).copied())
                .filter(|ordinal| *ordinal != u32::MAX)
            else {
                continue;
            };
            let Some(target) = self
                .node(edge.target())
                .and_then(|node| ordinal_by_dense.get(node.dense() as usize).copied())
                .filter(|ordinal| *ordinal != u32::MAX)
            else {
                continue;
            };
            visible.push((source, target, dense));
        }
        let outgoing = Csr::build(visible_nodes.len(), &visible)?;
        let incoming = Csr::build_transposed(visible_nodes.len(), &visible)?;
        Ok((outgoing, incoming, visible_nodes, ordinal_by_dense))
    }

    pub fn apply(&mut self, mutation: &GraphMutation) -> Result<()> {
        match mutation {
            GraphMutation::DeclareLabel { name, id } => self.catalog.declare_label(name, *id),
            GraphMutation::DeclareProperty { name, id } => self.catalog.declare_property(name, *id),
            GraphMutation::DeclareRelationshipType { name, id } => {
                self.catalog.declare_relationship_type(name, *id)
            }
            GraphMutation::InsertNode(input) => self.insert_node(input),
            GraphMutation::InsertEdge(input) => self.insert_edge(input),
            GraphMutation::SetNodeProperty {
                node,
                property,
                value,
                revision,
            } => self.set_node_property(*node, *property, value, *revision),
            GraphMutation::AddNodeLabels {
                node,
                labels,
                revision,
            } => self.update_node_labels(*node, labels, *revision, true),
            GraphMutation::RemoveNodeLabels {
                node,
                labels,
                revision,
            } => self.update_node_labels(*node, labels, *revision, false),
            GraphMutation::SetEdgeProperty {
                edge,
                property,
                value,
                revision,
            } => self.set_edge_property(*edge, *property, value, *revision),
            GraphMutation::DeleteNode {
                node,
                detach,
                revision,
            } => self.delete_node(*node, *detach, *revision),
            GraphMutation::DeleteEdge { edge, revision } => self.delete_edge(*edge, *revision),
        }
    }

    fn insert_node(&mut self, input: &NodeInput) -> Result<()> {
        self.ensure_forward_revision(input.revision)?;
        if self.contains_node_id(input.id) {
            return Err(Error::new(
                ErrorCode::TransactionConflict,
                "node ID already exists",
            ));
        }
        let mut labels = input.labels.clone();
        labels.sort_unstable();
        labels.dedup();
        if labels.iter().any(|label| self.label_name(*label).is_none()) {
            return Err(Error::invalid_data(
                "node input references undeclared labels",
            ));
        }
        let properties = normalized_properties(&input.properties)?;
        for (property, value) in &properties {
            if self.property_name(*property).is_none() {
                return Err(Error::invalid_data(
                    "node input references an undeclared property",
                ));
            }
            self.validate_node_property(*property, value)?;
        }
        let dense = u32::try_from(
            self.base
                .node_slot_count()
                .saturating_add(self.inserted_node_count()),
        )
        .map_err(|_| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "dense node ID space exhausted",
            )
        })?;
        self.nodes.insert(
            dense,
            OverlayNode {
                dense,
                id: input.id,
                layer: input.layer,
                revision: input.revision,
                labels,
                properties,
                deleted: false,
            },
        );
        self.node_ids.insert(input.id, dense);
        self.inserted_node_total += 1;
        self.revision = input.revision;
        Ok(())
    }

    fn insert_edge(&mut self, input: &EdgeInput) -> Result<()> {
        self.ensure_forward_revision(input.revision)?;
        if self.contains_edge_id(input.id) {
            return Err(Error::new(
                ErrorCode::TransactionConflict,
                "relationship ID already exists",
            ));
        }
        if self.node(input.source).is_none() || self.node(input.target).is_none() {
            return Err(Error::new(
                ErrorCode::QueryType,
                "relationship endpoint does not exist",
            ));
        }
        if self
            .relationship_type_name(input.relationship_type)
            .is_none()
        {
            return Err(Error::invalid_data(
                "relationship input references an undeclared type",
            ));
        }
        let properties = normalized_properties(&input.properties)?;
        for (property, value) in &properties {
            if self.property_name(*property).is_none() {
                return Err(Error::invalid_data(
                    "relationship input references an undeclared property",
                ));
            }
            self.validate_edge_property(*property, value)?;
        }
        let dense = u32::try_from(
            self.base
                .edge_slot_count()
                .saturating_add(self.inserted_edge_count()),
        )
        .map_err(|_| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "dense relationship ID space exhausted",
            )
        })?;
        self.edges.insert(
            dense,
            OverlayEdge {
                dense,
                id: input.id,
                source: input.source,
                target: input.target,
                relationship_type: input.relationship_type,
                layer: input.layer,
                revision: input.revision,
                properties,
                deleted: false,
            },
        );
        self.edge_ids.insert(input.id, dense);
        self.inserted_edge_total += 1;
        self.inserted_outgoing
            .entry(input.source)
            .or_default()
            .insert(dense);
        self.inserted_incoming
            .entry(input.target)
            .or_default()
            .insert(dense);
        self.revision = input.revision;
        Ok(())
    }

    fn set_node_property(
        &mut self,
        node: NodeId,
        property: PropertyId,
        value: &ScalarValue,
        revision: u64,
    ) -> Result<()> {
        self.ensure_forward_revision(revision)?;
        if self.property_name(property).is_none() {
            return Err(Error::invalid_data("node property ID is undeclared"));
        }
        self.validate_node_property(property, value)?;
        let dense = self
            .node(node)
            .ok_or_else(|| Error::new(ErrorCode::QueryType, "node does not exist"))?
            .dense();
        let row = self.materialize_node(dense)?;
        if matches!(value, ScalarValue::Null) {
            row.properties.remove(&property);
        } else {
            row.properties.insert(property, value.clone());
        }
        row.revision = revision;
        self.revision = revision;
        Ok(())
    }

    fn update_node_labels(
        &mut self,
        node: NodeId,
        labels: &[LabelId],
        revision: u64,
        add: bool,
    ) -> Result<()> {
        self.ensure_forward_revision(revision)?;
        if labels.iter().any(|label| self.label_name(*label).is_none()) {
            return Err(Error::invalid_data("node label ID is undeclared"));
        }
        let dense = self
            .node(node)
            .ok_or_else(|| Error::new(ErrorCode::QueryType, "node does not exist"))?
            .dense();
        let row = self.materialize_node(dense)?;
        if add {
            row.labels.extend(labels.iter().copied());
            row.labels.sort_unstable();
            row.labels.dedup();
        } else {
            let removed = labels.iter().copied().collect::<BTreeSet<_>>();
            row.labels.retain(|label| !removed.contains(label));
        }
        row.revision = revision;
        self.revision = revision;
        Ok(())
    }

    fn set_edge_property(
        &mut self,
        edge: EdgeId,
        property: PropertyId,
        value: &ScalarValue,
        revision: u64,
    ) -> Result<()> {
        self.ensure_forward_revision(revision)?;
        if self.property_name(property).is_none() {
            return Err(Error::invalid_data(
                "relationship property ID is undeclared",
            ));
        }
        self.validate_edge_property(property, value)?;
        let dense = self
            .edge(edge)
            .ok_or_else(|| Error::new(ErrorCode::QueryType, "relationship does not exist"))?
            .dense();
        let row = self.materialize_edge(dense)?;
        if matches!(value, ScalarValue::Null) {
            row.properties.remove(&property);
        } else {
            row.properties.insert(property, value.clone());
        }
        row.revision = revision;
        self.revision = revision;
        Ok(())
    }

    fn delete_edge(&mut self, edge: EdgeId, revision: u64) -> Result<()> {
        self.ensure_forward_revision(revision)?;
        let dense = self
            .edge(edge)
            .ok_or_else(|| Error::new(ErrorCode::QueryType, "relationship does not exist"))?
            .dense();
        let row = self.materialize_edge(dense)?;
        row.deleted = true;
        row.revision = revision;
        self.revision = revision;
        Ok(())
    }

    fn delete_node(&mut self, node: NodeId, detach: bool, revision: u64) -> Result<()> {
        self.ensure_forward_revision(revision)?;
        let dense = self
            .node(node)
            .ok_or_else(|| Error::new(ErrorCode::QueryType, "node does not exist"))?
            .dense();
        let mut incident = Vec::new();
        let capacity = self
            .base
            .edge_slot_count()
            .saturating_add(self.inserted_edge_count());
        for row in 0..capacity {
            let edge_dense = u32::try_from(row).map_err(|_| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "dense relationship ID space exhausted",
                )
            })?;
            if let Some(edge) = self.edge_dense(edge_dense)
                && (edge.source() == node || edge.target() == node)
            {
                incident.push(edge.id());
            }
        }
        if !detach && !incident.is_empty() {
            return Err(Error::new(
                ErrorCode::QueryType,
                "node still has relationships",
            ));
        }
        for edge in incident {
            self.delete_edge(edge, revision)?;
        }
        let row = self.materialize_node(dense)?;
        row.deleted = true;
        row.revision = revision;
        self.revision = revision;
        Ok(())
    }

    fn materialize_node(&mut self, dense: u32) -> Result<&mut OverlayNode> {
        if !self.nodes.contains_key(&dense) {
            let node = self
                .base
                .node_dense(dense)
                .ok_or_else(|| Error::new(ErrorCode::QueryType, "node does not exist"))?;
            let overlay = OverlayNode {
                dense,
                id: node.id(),
                layer: node.layer(),
                revision: node.revision(),
                labels: node.labels().to_vec(),
                properties: node.properties().into_iter().collect(),
                deleted: false,
            };
            self.node_ids.insert(overlay.id, dense);
            self.nodes.insert(dense, overlay);
        }
        self.nodes
            .get_mut(&dense)
            .ok_or_else(|| Error::internal("statement node overlay disappeared"))
    }

    fn materialize_edge(&mut self, dense: u32) -> Result<&mut OverlayEdge> {
        if !self.edges.contains_key(&dense) {
            let edge = self
                .base
                .edge_dense(dense)
                .ok_or_else(|| Error::new(ErrorCode::QueryType, "relationship does not exist"))?;
            let overlay = OverlayEdge {
                dense,
                id: edge.id(),
                source: edge.source(),
                target: edge.target(),
                relationship_type: edge.relationship_type(),
                layer: edge.layer(),
                revision: edge.revision(),
                properties: edge.properties().into_iter().collect(),
                deleted: false,
            };
            self.edge_ids.insert(overlay.id, dense);
            self.edges.insert(dense, overlay);
        }
        self.edges
            .get_mut(&dense)
            .ok_or_else(|| Error::internal("statement relationship overlay disappeared"))
    }

    fn validate_node_property(&mut self, property: PropertyId, value: &ScalarValue) -> Result<()> {
        self.base.validate_node_property_value(property, value)?;
        validate_overlay_property(&mut self.node_property_types, property, value)
    }

    fn validate_edge_property(&mut self, property: PropertyId, value: &ScalarValue) -> Result<()> {
        self.base.validate_edge_property_value(property, value)?;
        validate_overlay_property(&mut self.edge_property_types, property, value)
    }

    fn ensure_forward_revision(&self, revision: u64) -> Result<()> {
        if revision < self.revision {
            return Err(Error::invalid_data(
                "mutation revision moves graph state backwards",
            ));
        }
        Ok(())
    }

    fn inserted_node_count(&self) -> usize {
        self.inserted_node_total
    }

    fn inserted_edge_count(&self) -> usize {
        self.inserted_edge_total
    }

    #[cfg(test)]
    fn overlay_row_counts(&self) -> (usize, usize) {
        (self.nodes.len(), self.edges.len())
    }
}

fn normalized_properties(
    properties: &[(PropertyId, ScalarValue)],
) -> Result<BTreeMap<PropertyId, ScalarValue>> {
    let mut result = BTreeMap::new();
    let mut seen = BTreeSet::new();
    for (property, value) in properties {
        if !seen.insert(*property) {
            return Err(Error::invalid_data(
                "entity input contains a duplicate property",
            ));
        }
        if !matches!(value, ScalarValue::Null) {
            result.insert(*property, value.clone());
        }
    }
    Ok(result)
}

fn validate_overlay_property(
    types: &mut BTreeMap<PropertyId, ScalarKind>,
    property: PropertyId,
    value: &ScalarValue,
) -> Result<()> {
    let Some(kind) = ScalarKind::of(value) else {
        return Ok(());
    };
    match types.get_mut(&property) {
        Some(existing) if *existing != kind => *existing = ScalarKind::Mixed,
        Some(_) => {}
        None => {
            types.insert(property, kind);
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
pub struct SparseNodeState {
    pub labels: Vec<LabelId>,
    pub properties: BTreeMap<PropertyId, ScalarValue>,
}

/// Resolves only requested nodes after accumulated transaction mutations. Memory grows with the
/// touched mutation set rather than with the canonical project.
pub fn sparse_node_states_after_mutations(
    base: &GraphStore,
    prior: &[GraphMutation],
    current: &[GraphMutation],
    nodes: impl IntoIterator<Item = NodeId>,
) -> Result<BTreeMap<NodeId, SparseNodeState>> {
    let mut view = GraphReadView::new(base);
    for mutation in prior.iter().chain(current) {
        view.apply(mutation)?;
    }
    let mut states = BTreeMap::new();
    for id in nodes {
        let Some(node) = view.node(id) else {
            continue;
        };
        states.insert(
            id,
            SparseNodeState {
                labels: node.labels().to_vec(),
                properties: node.properties().into_iter().collect(),
            },
        );
    }
    Ok(states)
}

/// Temporal histories remain borrowed; only statement-local samples are retained in memory.
pub struct TemporalReadView<'a> {
    base: Option<&'a TemporalStore>,
    appended: Vec<(EntityKind, u64, TemporalSample)>,
}

impl<'a> TemporalReadView<'a> {
    pub fn new(base: Option<&'a TemporalStore>) -> Self {
        Self {
            base,
            appended: Vec::new(),
        }
    }

    pub fn resolve_target(
        &self,
        entity_kind: EntityKind,
        targets: &[u64],
        property: PropertyId,
    ) -> Result<Option<u64>> {
        self.base.map_or(Ok(None), |base| {
            base.resolve_target(entity_kind, targets, property)
        })
    }

    pub fn append(
        &mut self,
        entity_kind: EntityKind,
        target: u64,
        sample: TemporalSample,
        resolved_commit_time_nanos: i64,
    ) -> Result<()> {
        let base = self
            .base
            .ok_or_else(|| Error::new(ErrorCode::QueryType, "property is not declared temporal"))?;
        base.validate_append(entity_kind, target, &sample, resolved_commit_time_nanos)?;
        self.appended.push((entity_kind, target, sample));
        Ok(())
    }

    pub fn current(
        &self,
        entity_kind: EntityKind,
        target: u64,
        entity_id: u64,
        property: PropertyId,
    ) -> Option<TemporalSample> {
        self.base
            .and_then(|base| base.current(entity_kind, target, entity_id, property))
            .into_iter()
            .chain(
                self.appended
                    .iter()
                    .filter_map(|(kind, sample_target, sample)| {
                        (*kind == entity_kind
                            && *sample_target == target
                            && sample.entity_id == entity_id
                            && sample.property == property)
                            .then_some(sample.clone())
                    }),
            )
            .max_by_key(|sample| (sample.event_time_nanos, sample.sequence_index))
    }

    pub fn at_time(
        &self,
        entity_kind: EntityKind,
        target: u64,
        entity_id: u64,
        property: PropertyId,
        when_nanos: i64,
        bookmark_index: u64,
    ) -> Option<TemporalSample> {
        self.base
            .and_then(|base| {
                base.at_time(
                    entity_kind,
                    target,
                    entity_id,
                    property,
                    when_nanos,
                    bookmark_index,
                )
            })
            .into_iter()
            .chain(
                self.appended
                    .iter()
                    .filter_map(|(kind, sample_target, sample)| {
                        (*kind == entity_kind
                            && *sample_target == target
                            && sample.entity_id == entity_id
                            && sample.property == property
                            && sample.event_time_nanos <= when_nanos
                            && sample.sequence_index <= bookmark_index)
                            .then_some(sample.clone())
                    }),
            )
            .max_by_key(|sample| (sample.event_time_nanos, sample.sequence_index))
    }

    #[allow(clippy::too_many_arguments)]
    pub fn history(
        &self,
        entity_kind: EntityKind,
        target: u64,
        entity_id: u64,
        property: PropertyId,
        from_nanos: i64,
        to_nanos: i64,
        bookmark_index: u64,
    ) -> Result<Vec<TemporalSample>> {
        if from_nanos >= to_nanos {
            return Err(Error::new(
                ErrorCode::TemporalRange,
                "HISTORY range must be non-empty",
            ));
        }
        let base = self
            .base
            .ok_or_else(|| Error::new(ErrorCode::QueryType, "property is not declared temporal"))?;
        let mut samples = base.history(
            entity_kind,
            target,
            entity_id,
            property,
            from_nanos,
            to_nanos,
            bookmark_index,
        )?;
        samples.extend(
            self.appended
                .iter()
                .filter_map(|(kind, sample_target, sample)| {
                    (*kind == entity_kind
                        && *sample_target == target
                        && sample.entity_id == entity_id
                        && sample.property == property
                        && sample.event_time_nanos >= from_nanos
                        && sample.event_time_nanos < to_nanos
                        && sample.sequence_index <= bookmark_index)
                        .then_some(sample.clone())
                }),
        );
        samples.sort_by_key(|sample| (sample.event_time_nanos, sample.sequence_index));
        Ok(samples)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{TemporalDeclaration, TemporalType};

    fn graph_fixture() -> Result<(GraphStore, LabelId, PropertyId, RelationshipTypeId)> {
        let mut graph = GraphStore::default();
        let label = graph.catalog_mut().intern_label("Item")?;
        let property = graph.catalog_mut().intern_property("value")?;
        let relationship_type = graph.catalog_mut().intern_relationship_type("NEXT")?;
        graph.insert_node(NodeInput {
            id: NodeId(1),
            layer: Layer::Observed,
            revision: 1,
            labels: vec![label],
            properties: vec![(property, ScalarValue::Integer(10))],
        })?;
        graph.insert_node(NodeInput {
            id: NodeId(2),
            layer: Layer::Observed,
            revision: 1,
            labels: vec![label],
            properties: vec![(property, ScalarValue::Integer(20))],
        })?;
        graph.insert_edge(EdgeInput {
            id: EdgeId(1),
            source: NodeId(1),
            target: NodeId(2),
            relationship_type,
            layer: Layer::Observed,
            revision: 1,
            properties: Vec::new(),
        })?;
        Ok((graph, label, property, relationship_type))
    }

    #[test]
    fn read_only_view_borrows_canonical_rows_and_write_materializes_one_row() -> Result<()> {
        let (graph, label, property, _) = graph_fixture()?;
        let mut view = GraphReadView::new(&graph);

        assert_eq!(view.overlay_row_counts(), (0, 0));
        assert_eq!(
            view.scan_node_denses(Some(label), LayerMask::AUTHORITY),
            vec![0, 1]
        );
        assert_eq!(view.overlay_row_counts(), (0, 0));

        view.apply(&GraphMutation::SetNodeProperty {
            node: NodeId(1),
            property,
            value: ScalarValue::Integer(99),
            revision: 2,
        })?;
        assert_eq!(view.overlay_row_counts(), (1, 0));
        assert_eq!(
            view.node(NodeId(1))
                .and_then(|node| node.property(property)),
            Some(ScalarValue::Integer(99))
        );
        assert_eq!(
            graph
                .node(NodeId(1))
                .and_then(|node| node.property(property)),
            Some(ScalarValue::Integer(10))
        );
        assert_eq!(
            view.node(NodeId(2))
                .and_then(|node| node.property(property)),
            Some(ScalarValue::Integer(20))
        );
        Ok(())
    }

    #[test]
    fn algorithm_adjacency_dense_mapping_tracks_rows_not_dirty_property_bytes() -> Result<()> {
        let mut graph = GraphStore::default();
        let label = graph.catalog_mut().intern_label("Item")?;
        let body = graph.catalog_mut().intern_property("body")?;
        let relationship_type = graph.catalog_mut().intern_relationship_type("NEXT")?;
        let dirty = "x".repeat(128 * 1024);
        for row in 0..400_u64 {
            graph.insert_node(NodeInput {
                id: NodeId(row + 1),
                layer: Layer::Observed,
                revision: row + 1,
                labels: vec![label],
                properties: (row % 40 == 0)
                    .then(|| vec![(body, ScalarValue::String(dirty.clone().into()))])
                    .unwrap_or_default(),
            })?;
            if row > 0 {
                graph.insert_edge(EdgeInput {
                    id: EdgeId(row),
                    source: NodeId(row),
                    target: NodeId(row + 1),
                    relationship_type,
                    layer: Layer::Observed,
                    revision: row + 1,
                    properties: Vec::new(),
                })?;
            }
        }

        let view = GraphReadView::new(&graph);
        let (outgoing, incoming, nodes, ordinal_by_dense) =
            view.algorithm_adjacency(LayerMask::AUTHORITY)?;
        assert_eq!(nodes.len(), 400);
        assert_eq!(ordinal_by_dense.len(), 400);
        assert_eq!(ordinal_by_dense[0], 0);
        assert_eq!(ordinal_by_dense[399], 399);
        assert_eq!(outgoing.neighbors().len(), 399);
        assert_eq!(incoming.neighbors().len(), 399);
        Ok(())
    }

    #[test]
    fn null_property_writes_remove_values_from_the_statement_overlay() -> Result<()> {
        let (graph, label, property, relationship_type) = graph_fixture()?;
        let mut view = GraphReadView::new(&graph);

        view.apply(&GraphMutation::SetNodeProperty {
            node: NodeId(1),
            property,
            value: ScalarValue::Null,
            revision: 2,
        })?;
        assert!(
            view.node(NodeId(1))
                .and_then(|node| node.property(property))
                .is_none()
        );
        assert!(
            view.node(NodeId(1))
                .is_some_and(|node| node.properties().is_empty())
        );

        view.apply(&GraphMutation::SetEdgeProperty {
            edge: EdgeId(1),
            property,
            value: ScalarValue::Integer(7),
            revision: 3,
        })?;
        view.apply(&GraphMutation::SetEdgeProperty {
            edge: EdgeId(1),
            property,
            value: ScalarValue::Null,
            revision: 4,
        })?;
        assert!(
            view.edge(EdgeId(1))
                .and_then(|edge| edge.property(property))
                .is_none()
        );

        view.apply(&GraphMutation::InsertNode(NodeInput {
            id: NodeId(3),
            layer: Layer::Observed,
            revision: 5,
            labels: vec![label],
            properties: vec![(property, ScalarValue::Null)],
        }))?;
        assert!(
            view.node(NodeId(3))
                .is_some_and(|node| node.properties().is_empty())
        );
        assert_eq!(
            view.edge(EdgeId(1)).map(|edge| edge.relationship_type()),
            Some(relationship_type)
        );
        assert_eq!(
            graph
                .node(NodeId(1))
                .and_then(|node| node.property(property)),
            Some(ScalarValue::Integer(10))
        );
        Ok(())
    }

    #[test]
    fn sparse_overlay_preserves_read_your_writes_and_detach_visibility() -> Result<()> {
        let (graph, label, property, relationship_type) = graph_fixture()?;
        let mut view = GraphReadView::new(&graph);
        let added_property = view.intern_property("added")?;
        view.apply(&GraphMutation::InsertNode(NodeInput {
            id: NodeId(3),
            layer: Layer::Observed,
            revision: 2,
            labels: vec![label],
            properties: vec![
                (property, ScalarValue::Integer(30)),
                (added_property, ScalarValue::String("local".into())),
            ],
        }))?;
        view.apply(&GraphMutation::InsertEdge(EdgeInput {
            id: EdgeId(2),
            source: NodeId(2),
            target: NodeId(3),
            relationship_type,
            layer: Layer::Observed,
            revision: 2,
            properties: Vec::new(),
        }))?;
        assert_eq!(
            view.expand_out(NodeId(2), None, LayerMask::AUTHORITY)?
                .into_iter()
                .map(|(edge, node)| (edge.id(), node.id()))
                .collect::<Vec<_>>(),
            vec![(EdgeId(2), NodeId(3))]
        );

        view.apply(&GraphMutation::DeleteNode {
            node: NodeId(2),
            detach: true,
            revision: 3,
        })?;
        assert!(view.node(NodeId(2)).is_none());
        assert!(view.edge(EdgeId(1)).is_none());
        assert!(view.edge(EdgeId(2)).is_none());
        assert!(graph.node(NodeId(2)).is_some());
        assert!(graph.edge(EdgeId(1)).is_some());
        Ok(())
    }

    #[test]
    fn statement_overlay_accepts_heterogeneous_values_for_one_property_key() -> Result<()> {
        let (graph, label, property, relationship_type) = graph_fixture()?;
        let mut view = GraphReadView::new(&graph);
        view.apply(&GraphMutation::InsertNode(NodeInput {
            id: NodeId(3),
            layer: Layer::Observed,
            revision: 2,
            labels: vec![label],
            properties: vec![(property, ScalarValue::Integer(0))],
        }))?;
        view.apply(&GraphMutation::InsertNode(NodeInput {
            id: NodeId(4),
            layer: Layer::Observed,
            revision: 2,
            labels: vec![label],
            properties: vec![(property, ScalarValue::String("xx".into()))],
        }))?;
        view.apply(&GraphMutation::InsertNode(NodeInput {
            id: NodeId(5),
            layer: Layer::Observed,
            revision: 2,
            labels: vec![label],
            properties: Vec::new(),
        }))?;
        view.apply(&GraphMutation::InsertEdge(EdgeInput {
            id: EdgeId(2),
            source: NodeId(3),
            target: NodeId(4),
            relationship_type,
            layer: Layer::Observed,
            revision: 2,
            properties: vec![(property, ScalarValue::Integer(1))],
        }))?;
        view.apply(&GraphMutation::InsertEdge(EdgeInput {
            id: EdgeId(3),
            source: NodeId(4),
            target: NodeId(5),
            relationship_type,
            layer: Layer::Observed,
            revision: 2,
            properties: vec![(property, ScalarValue::String("edge".into()))],
        }))?;
        view.apply(&GraphMutation::SetNodeProperty {
            node: NodeId(1),
            property,
            value: ScalarValue::String("changed".into()),
            revision: 3,
        })?;

        assert_eq!(
            view.node_property_types.get(&property),
            Some(&ScalarKind::Mixed)
        );
        assert_eq!(
            view.edge_property_types.get(&property),
            Some(&ScalarKind::Mixed)
        );
        assert_eq!(
            view.node(NodeId(3))
                .and_then(|node| node.property(property)),
            Some(ScalarValue::Integer(0))
        );
        assert_eq!(
            view.node(NodeId(4))
                .and_then(|node| node.property(property)),
            Some(ScalarValue::String("xx".into()))
        );
        assert_eq!(
            view.node(NodeId(5))
                .and_then(|node| node.property(property)),
            None
        );
        assert_eq!(
            view.node(NodeId(1))
                .and_then(|node| node.property(property)),
            Some(ScalarValue::String("changed".into()))
        );
        assert_eq!(
            view.edge(EdgeId(2))
                .and_then(|edge| edge.property(property)),
            Some(ScalarValue::Integer(1))
        );
        assert_eq!(
            view.edge(EdgeId(3))
                .and_then(|edge| edge.property(property)),
            Some(ScalarValue::String("edge".into()))
        );
        Ok(())
    }

    #[test]
    fn temporal_overlay_reads_new_samples_without_cloning_or_mutating_base() -> Result<()> {
        let property = PropertyId(7);
        let mut temporal = TemporalStore::default();
        temporal.declare(
            TemporalDeclaration {
                entity_kind: EntityKind::Node,
                target: 4,
                property,
                value_type: TemporalType::Integer,
                retention_nanos: 1_000,
            },
            1_000,
        )?;
        temporal.append(
            EntityKind::Node,
            4,
            TemporalSample {
                entity_id: 9,
                property,
                event_time_nanos: 900,
                sequence_index: 1,
                value: ScalarValue::Integer(1),
            },
            1_000,
        )?;
        let mut view = TemporalReadView::new(Some(&temporal));
        view.append(
            EntityKind::Node,
            4,
            TemporalSample {
                entity_id: 9,
                property,
                event_time_nanos: 950,
                sequence_index: 2,
                value: ScalarValue::Integer(2),
            },
            1_000,
        )?;

        assert_eq!(
            view.current(EntityKind::Node, 4, 9, property)
                .map(|sample| sample.value),
            Some(ScalarValue::Integer(2))
        );
        assert_eq!(
            temporal
                .current(EntityKind::Node, 4, 9, property)
                .map(|sample| sample.value.clone()),
            Some(ScalarValue::Integer(1))
        );
        assert_eq!(
            view.history(EntityKind::Node, 4, 9, property, 800, 1_000, 2)?
                .len(),
            2
        );
        Ok(())
    }
}
