//! Canonical structure-of-arrays graph state and deterministic mutation application.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, OnceLock},
};

use bitvec::{order::Lsb0, vec::BitVec};
use serde::{Deserialize, Serialize};

use crate::{
    EdgeId, Error, ErrorCode, Layer, NodeId, Result, ScalarValue,
    types::{LabelId, PropertyId, RelationshipTypeId},
};

use super::{
    Adjacency, Csr, LayerMask, PackedLists, PropertyColumns, TypedColumn,
    persistent::{PagedVec, PersistentMap, stable_id_key, stable_id_row},
};

/// Canonical schema-name dictionaries. IDs are stable within a project.
#[derive(Clone, Debug, Default)]
pub struct NameCatalog {
    labels: SchemaNames,
    properties: SchemaNames,
    relationship_types: SchemaNames,
    next_label: u64,
    next_property: u64,
    next_relationship_type: u64,
    optimizer_generation_cache: OnceLock<[u8; 32]>,
}

/// Exact-name collision bucket with shared immutable string payloads.
type SchemaNameBucket = Arc<Vec<(Arc<str>, u64)>>;

/// Names and digest paths share their untouched allocations with pinned catalog generations.
#[derive(Clone, Debug, Default)]
struct SchemaNames {
    by_name: PersistentMap<SchemaNameBucket>,
    by_id: PersistentMap<Arc<str>>,
    digests: PersistentMap<[u8; 32]>,
}

impl SchemaNames {
    fn name_hash(name: &str) -> u128 {
        if name.len() < 16 {
            // Short names fit exactly in one radix key. The final length byte distinguishes
            // trailing zero bytes; collision buckets still validate the complete name.
            let mut key = [0_u8; 16];
            key[..name.len()].copy_from_slice(name.as_bytes());
            key[15] = name.len() as u8;
            return u128::from_be_bytes(key);
        }
        let hash = blake3::hash(name.as_bytes());
        let mut key = [0; 16];
        key.copy_from_slice(&hash.as_bytes()[..16]);
        u128::from_le_bytes(key)
    }

    fn get(&self, name: &str) -> Option<u64> {
        self.by_name
            .get(Self::name_hash(name))?
            .iter()
            .find_map(|(existing, id)| (existing.as_ref() == name).then_some(*id))
    }

    fn name(&self, id: u64) -> Option<&str> {
        self.by_id.get(stable_id_key(id)).map(AsRef::as_ref)
    }

    fn iter(&self) -> impl Iterator<Item = (u64, &str)> {
        self.by_id
            .iter()
            .map(|(key, name)| ((key >> 64) as u64, name.as_ref()))
    }

    fn declare(&mut self, next: &mut u64, name: String, id: u64, kind: &str) -> Result<()> {
        if self.get(&name).is_some_and(|existing| existing != id)
            || self.name(id).is_some_and(|existing| existing != name)
        {
            return Err(Error::new(
                ErrorCode::TransactionConflict,
                format!("{kind} name or ID is already assigned"),
            ));
        }
        let following = id.checked_add(1).ok_or_else(|| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                format!("{kind} ID space exhausted"),
            )
        })?;
        if self.name(id).is_none() {
            let name: Arc<str> = Arc::from(name);
            let key = Self::name_hash(&name);
            let mut bucket = self.by_name.get(key).cloned().unwrap_or_default();
            Arc::make_mut(&mut bucket).push((Arc::clone(&name), id));
            self.by_name.insert_cow(key, bucket);
            self.by_id.insert_cow(stable_id_key(id), Arc::clone(&name));
            self.update_digest(id, &name);
        }
        *next = (*next).max(following);
        Ok(())
    }

    fn digest_key(depth: u8, prefix: u64) -> u128 {
        (u128::from(depth) << 64) | u128::from(prefix)
    }

    fn update_digest(&mut self, id: u64, name: &str) {
        let mut leaf = blake3::Hasher::new();
        leaf.update(b"schema-name-leaf-v2\0");
        leaf.update(&id.to_le_bytes());
        leaf.update(&(name.len() as u64).to_le_bytes());
        leaf.update(name.as_bytes());
        self.digests
            .insert_cow(Self::digest_key(16, id), *leaf.finalize().as_bytes());
        // A fixed-depth Merkle tree makes insertion order irrelevant. Depth and child position
        // are committed; no commutative XOR accumulator or mutation-history hash is involved.
        for depth in (0_u8..16).rev() {
            let prefix = if depth == 0 {
                0
            } else {
                id >> (64 - u32::from(depth) * 4)
            };
            let mut branch = blake3::Hasher::new();
            branch.update(b"schema-name-branch-v2\0");
            branch.update(&[depth]);
            for child in 0..16 {
                let key = Self::digest_key(depth + 1, (prefix << 4) | child);
                branch.update(self.digests.get(key).unwrap_or(&[0; 32]));
            }
            self.digests.insert_cow(
                Self::digest_key(depth, prefix),
                *branch.finalize().as_bytes(),
            );
        }
    }

    fn root_digest(&self) -> &[u8; 32] {
        self.digests.get(Self::digest_key(0, 0)).unwrap_or(&[0; 32])
    }

    #[cfg(test)]
    fn detached_bytes_from(&self, previous: &Self) -> usize {
        self.by_name.detached_node_bytes_from(&previous.by_name)
            + self.by_id.detached_node_bytes_from(&previous.by_id)
            + self.digests.detached_node_bytes_from(&previous.digests)
    }
}

// Preserve the original nine serialized fields, their order and their map key/value types.
// Digest and lookup structures are derived on cold recovery and never enter the wire format.
#[derive(Serialize, Deserialize)]
struct NameCatalogWire {
    labels: BTreeMap<String, LabelId>,
    properties: BTreeMap<String, PropertyId>,
    relationship_types: BTreeMap<String, RelationshipTypeId>,
    label_names: BTreeMap<LabelId, String>,
    property_names: BTreeMap<PropertyId, String>,
    relationship_names: BTreeMap<RelationshipTypeId, String>,
    next_label: u64,
    next_property: u64,
    next_relationship_type: u64,
}

impl Serialize for NameCatalog {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("NameCatalog", 9)?;
        state.serialize_field(
            "labels",
            &self
                .labels()
                .map(|(id, name)| (name, id))
                .collect::<BTreeMap<_, _>>(),
        )?;
        state.serialize_field(
            "properties",
            &self
                .properties()
                .map(|(id, name)| (name, id))
                .collect::<BTreeMap<_, _>>(),
        )?;
        state.serialize_field(
            "relationship_types",
            &self
                .relationship_types()
                .map(|(id, name)| (name, id))
                .collect::<BTreeMap<_, _>>(),
        )?;
        state.serialize_field("label_names", &self.labels().collect::<BTreeMap<_, _>>())?;
        state.serialize_field(
            "property_names",
            &self.properties().collect::<BTreeMap<_, _>>(),
        )?;
        state.serialize_field(
            "relationship_names",
            &self.relationship_types().collect::<BTreeMap<_, _>>(),
        )?;
        state.serialize_field("next_label", &self.next_label)?;
        state.serialize_field("next_property", &self.next_property)?;
        state.serialize_field("next_relationship_type", &self.next_relationship_type)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for NameCatalog {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let wire = NameCatalogWire::deserialize(deserializer)?;
        if wire.labels.len() != wire.label_names.len()
            || wire.properties.len() != wire.property_names.len()
            || wire.relationship_types.len() != wire.relationship_names.len()
            || wire
                .label_names
                .iter()
                .any(|(id, name)| wire.labels.get(name) != Some(id))
            || wire
                .property_names
                .iter()
                .any(|(id, name)| wire.properties.get(name) != Some(id))
            || wire
                .relationship_names
                .iter()
                .any(|(id, name)| wire.relationship_types.get(name) != Some(id))
        {
            return Err(serde::de::Error::custom(
                "schema name dictionaries are inconsistent",
            ));
        }
        let mut catalog = Self::default();
        for (id, name) in wire.label_names {
            catalog
                .declare_label(name, id)
                .map_err(serde::de::Error::custom)?;
        }
        for (id, name) in wire.property_names {
            catalog
                .declare_property(name, id)
                .map_err(serde::de::Error::custom)?;
        }
        for (id, name) in wire.relationship_names {
            catalog
                .declare_relationship_type(name, id)
                .map_err(serde::de::Error::custom)?;
        }
        if wire.next_label < catalog.next_label
            || wire.next_property < catalog.next_property
            || wire.next_relationship_type < catalog.next_relationship_type
        {
            return Err(serde::de::Error::custom(
                "schema next ID precedes an assigned ID",
            ));
        }
        catalog.next_label = wire.next_label;
        catalog.next_property = wire.next_property;
        catalog.next_relationship_type = wire.next_relationship_type;
        Ok(catalog)
    }
}

impl NameCatalog {
    /// Stable schema generation used only by ephemeral plan/statistics caches.
    #[must_use]
    pub fn optimizer_generation(&self) -> [u8; 32] {
        *self.optimizer_generation_cache.get_or_init(|| {
            let mut hasher = blake3::Hasher::new();
            hasher.update(b"schema-generation-v2\0");
            hasher.update(self.labels.root_digest());
            hasher.update(self.properties.root_digest());
            hasher.update(self.relationship_types.root_digest());
            *hasher.finalize().as_bytes()
        })
    }

    #[must_use]
    pub const fn next_label_id(&self) -> u64 {
        self.next_label
    }

    #[must_use]
    pub const fn next_property_id(&self) -> u64 {
        self.next_property
    }

    #[must_use]
    pub const fn next_relationship_type_id(&self) -> u64 {
        self.next_relationship_type
    }

    pub fn declare_label(&mut self, name: String, id: LabelId) -> Result<()> {
        self.optimizer_generation_cache.take();
        self.labels
            .declare(&mut self.next_label, name, id.0, "label")
    }

    pub fn declare_property(&mut self, name: String, id: PropertyId) -> Result<()> {
        self.optimizer_generation_cache.take();
        self.properties
            .declare(&mut self.next_property, name, id.0, "property")
    }

    pub fn declare_relationship_type(
        &mut self,
        name: String,
        id: RelationshipTypeId,
    ) -> Result<()> {
        self.optimizer_generation_cache.take();
        self.relationship_types.declare(
            &mut self.next_relationship_type,
            name,
            id.0,
            "relationship type",
        )
    }

    pub fn intern_label(&mut self, name: &str) -> Result<LabelId> {
        if let Some(id) = self.label(name) {
            return Ok(id);
        }
        let id = LabelId(self.next_label);
        self.declare_label(name.to_owned(), id)?;
        Ok(id)
    }

    pub fn intern_property(&mut self, name: &str) -> Result<PropertyId> {
        if let Some(id) = self.property(name) {
            return Ok(id);
        }
        let id = PropertyId(self.next_property);
        self.declare_property(name.to_owned(), id)?;
        Ok(id)
    }

    pub fn intern_relationship_type(&mut self, name: &str) -> Result<RelationshipTypeId> {
        if let Some(id) = self.relationship_type(name) {
            return Ok(id);
        }
        let id = RelationshipTypeId(self.next_relationship_type);
        self.declare_relationship_type(name.to_owned(), id)?;
        Ok(id)
    }

    #[must_use]
    pub fn label(&self, name: &str) -> Option<LabelId> {
        self.labels.get(name).map(LabelId)
    }

    #[must_use]
    pub fn property(&self, name: &str) -> Option<PropertyId> {
        self.properties.get(name).map(PropertyId)
    }

    #[must_use]
    pub fn relationship_type(&self, name: &str) -> Option<RelationshipTypeId> {
        self.relationship_types.get(name).map(RelationshipTypeId)
    }

    #[must_use]
    pub fn label_name(&self, id: LabelId) -> Option<&str> {
        self.labels.name(id.0)
    }

    #[must_use]
    pub fn property_name(&self, id: PropertyId) -> Option<&str> {
        self.properties.name(id.0)
    }

    #[must_use]
    pub fn relationship_type_name(&self, id: RelationshipTypeId) -> Option<&str> {
        self.relationship_types.name(id.0)
    }

    /// Labels in stable numeric-ID order for schema completion and transport metadata.
    pub fn labels(&self) -> impl Iterator<Item = (LabelId, &str)> {
        self.labels.iter().map(|(id, name)| (LabelId(id), name))
    }

    /// Properties in stable numeric-ID order for schema completion and transport metadata.
    pub fn properties(&self) -> impl Iterator<Item = (PropertyId, &str)> {
        self.properties
            .iter()
            .map(|(id, name)| (PropertyId(id), name))
    }

    /// Relationship types in stable numeric-ID order for schema completion and metadata.
    pub fn relationship_types(&self) -> impl Iterator<Item = (RelationshipTypeId, &str)> {
        self.relationship_types
            .iter()
            .map(|(id, name)| (RelationshipTypeId(id), name))
    }
}

/// Fully resolved node insert used by the deterministic mutation path.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NodeInput {
    pub id: NodeId,
    pub layer: Layer,
    pub revision: u64,
    pub labels: Vec<LabelId>,
    pub properties: Vec<(PropertyId, ScalarValue)>,
}

/// Fully resolved relationship insert used by the deterministic mutation path.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct EdgeInput {
    pub id: EdgeId,
    pub source: NodeId,
    pub target: NodeId,
    pub relationship_type: RelationshipTypeId,
    pub layer: Layer,
    pub revision: u64,
    pub properties: Vec<(PropertyId, ScalarValue)>,
}

/// Backend-neutral resolved graph command suitable for direct canonical publication.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum GraphMutation {
    DeclareLabel {
        name: String,
        id: LabelId,
    },
    DeclareProperty {
        name: String,
        id: PropertyId,
    },
    DeclareRelationshipType {
        name: String,
        id: RelationshipTypeId,
    },
    InsertNode(NodeInput),
    InsertEdge(EdgeInput),
    SetNodeProperty {
        node: NodeId,
        property: PropertyId,
        value: ScalarValue,
        revision: u64,
    },
    AddNodeLabels {
        node: NodeId,
        labels: Vec<LabelId>,
        revision: u64,
    },
    RemoveNodeLabels {
        node: NodeId,
        labels: Vec<LabelId>,
        revision: u64,
    },
    SetEdgeProperty {
        edge: EdgeId,
        property: PropertyId,
        value: ScalarValue,
        revision: u64,
    },
    DeleteNode {
        node: NodeId,
        detach: bool,
        revision: u64,
    },
    DeleteEdge {
        edge: EdgeId,
        revision: u64,
    },
}

/// Borrowed node view over canonical columns.
#[derive(Clone, Copy, Debug)]
pub struct NodeView<'a> {
    graph: &'a GraphStore,
    dense: u32,
}

impl<'a> NodeView<'a> {
    #[must_use]
    pub fn id(self) -> NodeId {
        self.graph.node_ids[self.dense as usize]
    }

    #[must_use]
    pub fn dense(self) -> u32 {
        self.dense
    }

    #[must_use]
    pub fn layer(self) -> Layer {
        self.graph.node_layers[self.dense as usize]
    }

    #[must_use]
    pub fn revision(self) -> u64 {
        self.graph.node_revisions[self.dense as usize]
    }

    #[must_use]
    pub fn labels(self) -> &'a [LabelId] {
        self.graph.node_labels(self.dense)
    }

    #[must_use]
    pub fn property(self, property: PropertyId) -> Option<ScalarValue> {
        self.graph.node_properties.get(self.dense, property)
    }

    #[must_use]
    pub fn properties(self) -> Vec<(PropertyId, ScalarValue)> {
        self.graph
            .node_properties
            .property_ids()
            .filter_map(|property| self.property(property).map(|value| (property, value)))
            .collect()
    }
}

/// Borrowed relationship view over canonical columns.
#[derive(Clone, Copy, Debug)]
pub struct EdgeView<'a> {
    graph: &'a GraphStore,
    dense: u32,
}

impl EdgeView<'_> {
    #[must_use]
    pub fn id(self) -> EdgeId {
        self.graph.edge_ids[self.dense as usize]
    }

    #[must_use]
    pub fn dense(self) -> u32 {
        self.dense
    }

    #[must_use]
    pub fn source(self) -> NodeId {
        self.graph.node_ids[self.graph.edge_sources[self.dense as usize] as usize]
    }

    #[must_use]
    pub fn target(self) -> NodeId {
        self.graph.node_ids[self.graph.edge_targets[self.dense as usize] as usize]
    }

    #[must_use]
    pub fn relationship_type(self) -> RelationshipTypeId {
        self.graph.edge_types[self.dense as usize]
    }

    #[must_use]
    pub fn layer(self) -> Layer {
        self.graph.edge_layers[self.dense as usize]
    }

    #[must_use]
    pub fn revision(self) -> u64 {
        self.graph.edge_revisions[self.dense as usize]
    }

    #[must_use]
    pub fn property(self, property: PropertyId) -> Option<ScalarValue> {
        self.graph.edge_properties.get(self.dense, property)
    }

    #[must_use]
    pub fn properties(self) -> Vec<(PropertyId, ScalarValue)> {
        self.graph
            .edge_properties
            .property_ids()
            .filter_map(|property| self.property(property).map(|value| (property, value)))
            .collect()
    }
}

/// Immutable complete flat graph image used for CPU/GPU residency and checkpointing.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GraphSnapshot {
    pub revision: u64,
    pub node_ids: Vec<NodeId>,
    pub node_layers: Vec<Layer>,
    pub node_revisions: Vec<u64>,
    pub node_active: Vec<bool>,
    pub node_labels: PackedLists<LabelId>,
    pub node_properties: PropertyColumns,
    pub edge_ids: Vec<EdgeId>,
    pub edge_sources: Vec<u32>,
    pub edge_targets: Vec<u32>,
    pub edge_types: Vec<RelationshipTypeId>,
    pub edge_layers: Vec<Layer>,
    pub edge_revisions: Vec<u64>,
    pub edge_active: Vec<bool>,
    pub edge_properties: PropertyColumns,
    pub outgoing: Csr,
    pub incoming: Csr,
    /// Physical dense-row epoch. Kept as the trailing serialized field to avoid shifting the
    /// established shape; versioned database checkpoints use CBOR maps and default it to zero.
    #[serde(default)]
    pub layout_version: u64,
}

/// Complete immutable bulk graph columns rebound to one shared accelerator allocation set.
/// Lookup maps, tombstone journals and schema catalogs remain compact host control structures.
#[derive(Clone, Debug)]
pub struct GraphSharedBacking {
    pub revision: u64,
    pub layout_version: u64,
    pub node_ids: PagedVec<NodeId>,
    pub node_layers: PagedVec<Layer>,
    pub node_revisions: PagedVec<u64>,
    pub node_active: PagedVec<u8>,
    pub node_labels: PackedLists<LabelId>,
    pub node_properties: PropertyColumns,
    pub edge_ids: PagedVec<EdgeId>,
    pub edge_sources: PagedVec<u32>,
    pub edge_targets: PagedVec<u32>,
    pub edge_types: PagedVec<RelationshipTypeId>,
    pub edge_layers: PagedVec<Layer>,
    pub edge_revisions: PagedVec<u64>,
    pub edge_active: PagedVec<u8>,
    pub edge_properties: PropertyColumns,
    pub outgoing: Csr,
    pub incoming: Csr,
}

impl GraphSharedBacking {
    /// Reports whether a sparse device delta changes topology rather than merely carrying the
    /// endpoint rows of a relationship property/revision update.
    #[must_use]
    pub fn adjacency_delta_changes(&self, delta: &GraphDeviceDelta) -> bool {
        self.outgoing.offsets().len() != delta.node_capacity.saturating_add(1)
            || self.incoming.offsets().len() != delta.node_capacity.saturating_add(1)
            || delta
                .outgoing
                .iter()
                .any(|row| !csr_row_matches(&self.outgoing, row.dense, &row.neighbors, &row.edges))
            || delta
                .incoming
                .iter()
                .any(|row| !csr_row_matches(&self.incoming, row.dense, &row.neighbors, &row.edges))
    }

    /// Applies one committed sparse graph generation to a cloned shared base. The touched pages
    /// are ordinary staging memory until the accelerator publishes a replacement shared buffer;
    /// the previously published generation remains immutable throughout staging.
    #[cfg_attr(
        not(all(feature = "accelerator", any(target_os = "macos", target_os = "ios"))),
        allow(dead_code)
    )]
    pub fn apply_device_delta(&mut self, delta: &GraphDeviceDelta) -> Result<()> {
        self.apply_device_delta_inner(delta, true, true)
    }

    /// Applies canonical fixed/property columns while leaving changed adjacency payloads for a
    /// device-native publisher. Appended node rows still extend the base offset domains so every
    /// subsequent generation retains the correct row count. Changed topology is published as
    /// complete replacement rows over this immutable cold CSR; compaction may fold it later.
    pub fn apply_device_delta_columns(&mut self, delta: &GraphDeviceDelta) -> Result<()> {
        self.apply_device_delta_inner(delta, false, true)
    }

    /// Applies only changed canonical rows. The device publisher extends the immutable cold CSR
    /// offset domains and publishes replacement adjacency rows before exposing this generation.
    pub fn apply_device_delta_deferred_adjacency(
        &mut self,
        delta: &GraphDeviceDelta,
    ) -> Result<()> {
        self.apply_device_delta_inner(delta, false, false)
    }

    /// Applies canonical columns while deferring only appended empty CSR rows to the accelerator's
    /// already-admitted shared offset tails. This is a delta path, not a cold build: it is valid
    /// only when no adjacency replacement rows exist, and the unpublished caller must rebase both
    /// CSR offset domains before publication.
    pub fn apply_device_delta_deferred_empty_rows(
        &mut self,
        delta: &GraphDeviceDelta,
    ) -> Result<()> {
        if !delta.outgoing.is_empty() || !delta.incoming.is_empty() {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "deferred empty-row publication received adjacency replacements",
            ));
        }
        self.apply_device_delta_inner(delta, false, false)
    }

    fn apply_device_delta_inner(
        &mut self,
        delta: &GraphDeviceDelta,
        rebuild_adjacency: bool,
        extend_empty_rows: bool,
    ) -> Result<()> {
        if delta.revision < self.revision
            || delta.node_capacity < self.node_ids.len()
            || delta.edge_capacity < self.edge_ids.len()
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "graph delta moves a shared generation backwards",
            ));
        }

        let mut nodes = delta.nodes.iter().collect::<Vec<_>>();
        nodes.sort_unstable_by_key(|row| row.dense);
        for node in nodes {
            let row = node.dense as usize;
            if row > self.node_ids.len() || row >= delta.node_capacity {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "graph delta contains a non-contiguous node row",
                ));
            }
            if row == self.node_ids.len() {
                self.node_ids.push(node.id);
                self.node_layers.push(node.layer);
                self.node_revisions.push(node.revision);
                self.node_active.push(u8::from(node.active));
                self.node_labels.push(node.labels.iter().copied())?;
                self.node_properties.push_row(&node.properties)?;
            } else {
                if self.node_ids.get(row) != Some(&node.id) {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        "graph delta attempts to replace a stable node ID",
                    ));
                }
                replace_fixed_row(&mut self.node_layers, row, node.layer)?;
                replace_fixed_row(&mut self.node_revisions, row, node.revision)?;
                replace_fixed_row(&mut self.node_active, row, u8::from(node.active))?;
                if self.node_labels.get(node.dense) != Some(node.labels.as_slice()) {
                    self.node_labels
                        .replace(node.dense, node.labels.iter().copied())?;
                }
                replace_property_row(
                    &mut self.node_properties,
                    node.dense,
                    &node.properties,
                    node.properties_complete,
                )?;
            }
        }
        if self.node_ids.len() != delta.node_capacity
            || self.node_layers.len() != delta.node_capacity
            || self.node_revisions.len() != delta.node_capacity
            || self.node_active.len() != delta.node_capacity
            || self.node_labels.rows() != delta.node_capacity
            || self.node_properties.rows() != delta.node_capacity
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "graph delta omits an appended node row",
            ));
        }

        let mut edges = delta.edges.iter().collect::<Vec<_>>();
        edges.sort_unstable_by_key(|row| row.dense);
        for edge in edges {
            let row = edge.dense as usize;
            if row > self.edge_ids.len()
                || row >= delta.edge_capacity
                || edge.source as usize >= delta.node_capacity
                || edge.target as usize >= delta.node_capacity
            {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "graph delta contains an invalid relationship row",
                ));
            }
            if row == self.edge_ids.len() {
                self.edge_ids.push(edge.id);
                self.edge_sources.push(edge.source);
                self.edge_targets.push(edge.target);
                self.edge_types.push(edge.relationship_type);
                self.edge_layers.push(edge.layer);
                self.edge_revisions.push(edge.revision);
                self.edge_active.push(u8::from(edge.active));
                self.edge_properties.push_row(&edge.properties)?;
            } else {
                if self.edge_ids.get(row) != Some(&edge.id) {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        "graph delta attempts to replace a stable relationship ID",
                    ));
                }
                replace_fixed_row(&mut self.edge_sources, row, edge.source)?;
                replace_fixed_row(&mut self.edge_targets, row, edge.target)?;
                replace_fixed_row(&mut self.edge_types, row, edge.relationship_type)?;
                replace_fixed_row(&mut self.edge_layers, row, edge.layer)?;
                replace_fixed_row(&mut self.edge_revisions, row, edge.revision)?;
                replace_fixed_row(&mut self.edge_active, row, u8::from(edge.active))?;
                replace_property_row(
                    &mut self.edge_properties,
                    edge.dense,
                    &edge.properties,
                    edge.properties_complete,
                )?;
            }
        }
        if self.edge_ids.len() != delta.edge_capacity
            || self.edge_sources.len() != delta.edge_capacity
            || self.edge_targets.len() != delta.edge_capacity
            || self.edge_types.len() != delta.edge_capacity
            || self.edge_layers.len() != delta.edge_capacity
            || self.edge_revisions.len() != delta.edge_capacity
            || self.edge_active.len() != delta.edge_capacity
            || self.edge_properties.rows() != delta.edge_capacity
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "graph delta omits an appended relationship row",
            ));
        }

        // Appending nodes without adjacency only lengthens the offset arrays: the new rows are
        // empty, so the neighbor and edge arrays are unchanged. Falling through to the rebuild
        // below for this case turned a single edgeless node insert into a full re-materialization
        // of every edge in the project, sorted twice, on every commit.
        if extend_empty_rows
            && (delta.outgoing.is_empty() || !rebuild_adjacency)
            && self.outgoing.offsets().len() < delta.node_capacity.saturating_add(1)
        {
            self.outgoing.extend_rows(delta.node_capacity)?;
            self.incoming.extend_rows(delta.node_capacity)?;
        } else if rebuild_adjacency {
            let outgoing = delta
                .outgoing
                .iter()
                .filter(|row| {
                    !csr_row_matches(&self.outgoing, row.dense, &row.neighbors, &row.edges)
                })
                .map(|row| (row.dense, (row.neighbors.as_slice(), row.edges.as_slice())))
                .collect::<BTreeMap<_, _>>();
            let incoming = delta
                .incoming
                .iter()
                .filter(|row| {
                    !csr_row_matches(&self.incoming, row.dense, &row.neighbors, &row.edges)
                })
                .map(|row| (row.dense, (row.neighbors.as_slice(), row.edges.as_slice())))
                .collect::<BTreeMap<_, _>>();
            if !outgoing.is_empty()
                || self.outgoing.offsets().len() != delta.node_capacity.saturating_add(1)
            {
                self.outgoing = self.outgoing.replace_rows(delta.node_capacity, &outgoing)?;
            }
            if !incoming.is_empty()
                || self.incoming.offsets().len() != delta.node_capacity.saturating_add(1)
            {
                self.incoming = self.incoming.replace_rows(delta.node_capacity, &incoming)?;
            }
        }
        self.revision = delta.revision;
        Ok(())
    }
}

fn csr_row_matches(csr: &Csr, dense: u32, neighbors: &[u32], edges: &[u32]) -> bool {
    neighbors.len() == edges.len()
        && csr
            .row(dense)
            .is_some_and(|row| row.eq(neighbors.iter().copied().zip(edges.iter().copied())))
}

fn replace_fixed_row<T: Clone + PartialEq>(
    values: &mut PagedVec<T>,
    row: usize,
    value: T,
) -> Result<()> {
    if values.get(row) != Some(&value) {
        values
            .replace(row, value)
            .map_err(|message| Error::new(ErrorCode::CorruptStorage, message))?;
    }
    Ok(())
}

#[cfg_attr(
    not(all(feature = "accelerator", any(target_os = "macos", target_os = "ios"))),
    allow(dead_code)
)]
fn replace_property_row(
    columns: &mut PropertyColumns,
    row: u32,
    values: &[(PropertyId, ScalarValue)],
    complete: bool,
) -> Result<()> {
    if complete {
        let removed = columns
            .property_ids()
            .filter(|property| {
                !values.iter().any(|(candidate, _)| candidate == property)
                    && columns
                        .physical_column(*property)
                        .is_some_and(|column| column.validity().is_present(row as usize))
            })
            .collect::<Vec<_>>();
        for property in removed {
            columns.set(row, property, &ScalarValue::Null)?;
        }
    }
    for (property, value) in values {
        let unchanged = match (columns.column(*property), value) {
            (_, ScalarValue::Null) => columns
                .physical_column(*property)
                .is_none_or(|column| !column.validity().is_present(row as usize)),
            (Some(TypedColumn::String { values, validity }), ScalarValue::String(after)) => {
                validity.is_present(row as usize)
                    && values
                        .get(row as usize)
                        .and_then(|id| columns.string_dictionary().resolve(*id))
                        == Some(after.as_ref())
            }
            (Some(TypedColumn::Bytes { values, validity }), ScalarValue::Bytes(after)) => {
                validity.is_present(row as usize)
                    && values
                        .get(row as usize)
                        .is_some_and(|before| before.as_ref() == after.as_ref())
            }
            _ => match (columns.get(row, *property), value) {
                (None, ScalarValue::Null) => true,
                (Some(ScalarValue::Float(before)), ScalarValue::Float(after)) => {
                    before.0.to_bits() == after.0.to_bits()
                }
                (before, after) => before.as_ref() == Some(after),
            },
        };
        if !unchanged {
            columns.set(row, *property, value)?;
        }
    }
    Ok(())
}

/// Sparse, backend-neutral rows changed at one committed graph revision.
/// It is derived from canonical columns and is never persisted as a second graph.
#[derive(Clone, Debug, PartialEq)]
pub struct GraphDeviceDelta {
    pub revision: u64,
    pub node_capacity: usize,
    pub edge_capacity: usize,
    pub nodes: Vec<NodeDeviceDelta>,
    pub edges: Vec<EdgeDeviceDelta>,
    pub outgoing: Vec<AdjacencyRowDeviceDelta>,
    pub incoming: Vec<AdjacencyRowDeviceDelta>,
}

/// Stable identities touched by one committed graph revision. This is the compact identity-only
/// counterpart of `GraphDeviceDelta`, used by rebuildable local consumers that do not need a
/// second copy of canonical row values.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GraphChangeIds {
    pub nodes: Vec<NodeId>,
    pub edges: Vec<EdgeId>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct NodeDeviceDelta {
    pub dense: u32,
    pub id: NodeId,
    pub layer: Layer,
    pub revision: u64,
    pub active: bool,
    pub labels: Vec<LabelId>,
    /// False means only changed properties are supplied; Null explicitly removes a property.
    pub properties_complete: bool,
    pub properties: Vec<(PropertyId, ScalarValue)>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EdgeDeviceDelta {
    pub dense: u32,
    pub id: EdgeId,
    pub source: u32,
    pub target: u32,
    pub relationship_type: RelationshipTypeId,
    pub layer: Layer,
    pub revision: u64,
    pub active: bool,
    /// False means only changed properties are supplied; Null explicitly removes a property.
    pub properties_complete: bool,
    pub properties: Vec<(PropertyId, ScalarValue)>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AdjacencyRowDeviceDelta {
    pub dense: u32,
    pub neighbors: Vec<u32>,
    pub edges: Vec<u32>,
}

/// Stable-to-new-dense mapping emitted by deterministic physical compaction.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct CompactionMap {
    pub nodes: BTreeMap<NodeId, u32>,
    pub edges: BTreeMap<EdgeId, u32>,
}

impl GraphSnapshot {
    /// Tests physical row presence without decoding the property's value. This is the canonical
    /// source for native `keys(node)` and deliberately distinguishes an absent row cell from a
    /// present value of any supported scalar/document shape.
    #[must_use]
    pub fn node_property_is_present(&self, row: u32, property: PropertyId) -> bool {
        (row as usize) < self.node_ids.len()
            && self
                .node_properties
                .physical_column(property)
                .is_some_and(|column| column.validity().is_present(row as usize))
    }

    /// Relationship counterpart of [`Self::node_property_is_present`].
    #[must_use]
    pub fn edge_property_is_present(&self, row: u32, property: PropertyId) -> bool {
        (row as usize) < self.edge_ids.len()
            && self
                .edge_properties
                .physical_column(property)
                .is_some_and(|column| column.validity().is_present(row as usize))
    }

    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        let node_fixed = self.node_ids.len().saturating_mul(
            size_of::<NodeId>() + size_of::<Layer>() + size_of::<u64>() + size_of::<bool>(),
        );
        let edge_fixed = self.edge_ids.len().saturating_mul(
            size_of::<EdgeId>()
                + 2 * size_of::<u32>()
                + size_of::<RelationshipTypeId>()
                + size_of::<Layer>()
                + size_of::<u64>()
                + size_of::<bool>(),
        );
        let adjacency = (self.outgoing.offsets().len()
            + self.incoming.offsets().len()
            + self.outgoing.neighbors().len()
            + self.incoming.neighbors().len()
            + self.outgoing.edges().len()
            + self.incoming.edges().len())
        .saturating_mul(size_of::<u32>());
        node_fixed
            .saturating_add(edge_fixed)
            .saturating_add(adjacency)
            .saturating_add(self.node_labels.estimated_bytes())
            .saturating_add(self.node_properties.estimated_bytes())
            .saturating_add(self.edge_properties.estimated_bytes())
    }
}

/// Mutable canonical project graph. Dense ordinals change only during explicit compaction.
#[derive(Clone, Debug, Default, Serialize)]
pub struct GraphStore {
    revision: u64,
    catalog: Arc<NameCatalog>,
    node_lookup: PersistentMap<u32>,
    node_ids: PagedVec<NodeId>,
    node_layers: PagedVec<Layer>,
    node_revisions: PagedVec<u64>,
    node_labels: Arc<PackedLists<LabelId>>,
    #[serde(default)]
    node_label_deltas: PersistentMap<Arc<Vec<LabelId>>>,
    node_properties: PropertyColumns,
    node_deleted: Arc<BitVec<u64, Lsb0>>,
    #[serde(default)]
    node_tombstones: PersistentMap<()>,
    edge_lookup: PersistentMap<u32>,
    edge_ids: PagedVec<EdgeId>,
    edge_sources: PagedVec<u32>,
    edge_targets: PagedVec<u32>,
    edge_types: PagedVec<RelationshipTypeId>,
    edge_layers: PagedVec<Layer>,
    edge_revisions: PagedVec<u64>,
    edge_properties: PropertyColumns,
    edge_deleted: Arc<BitVec<u64, Lsb0>>,
    #[serde(default)]
    edge_tombstones: PersistentMap<()>,
    adjacency: Arc<Adjacency>,
    /// Persisted physical-layout epoch. This remains the final serialized field; versioned CBOR
    /// checkpoints default a missing legacy field to zero. Ordinary mutations never advance it.
    #[serde(default)]
    layout_version: u64,
    #[serde(skip)]
    live_counts: OnceLock<LiveCounts>,
    /// Per-label / per-layer live node counters, built at cold boundaries and updated only for
    /// affected labels thereafter. Immutable generations share all untouched radix branches.
    #[serde(skip)]
    label_counts: OnceLock<Arc<LabelCounts>>,
    /// Per-relationship-type / per-layer counters with the same incremental lifecycle.
    #[serde(skip)]
    edge_type_counts: OnceLock<Arc<RelationshipCounts>>,
    /// Ephemeral apply output used for proportional device publication; never checkpointed.
    #[serde(skip)]
    device_changes: Arc<GraphChangeJournal>,
}

impl<'de> Deserialize<'de> for GraphStore {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        #[derive(Deserialize)]
        struct StoredGraph {
            revision: u64,
            catalog: Arc<NameCatalog>,
            node_lookup: PersistentMap<u32>,
            node_ids: PagedVec<NodeId>,
            node_layers: PagedVec<Layer>,
            node_revisions: PagedVec<u64>,
            node_labels: Arc<PackedLists<LabelId>>,
            #[serde(default)]
            node_label_deltas: PersistentMap<Arc<Vec<LabelId>>>,
            node_properties: PropertyColumns,
            node_deleted: Arc<BitVec<u64, Lsb0>>,
            #[serde(default)]
            node_tombstones: PersistentMap<()>,
            edge_lookup: PersistentMap<u32>,
            edge_ids: PagedVec<EdgeId>,
            edge_sources: PagedVec<u32>,
            edge_targets: PagedVec<u32>,
            edge_types: PagedVec<RelationshipTypeId>,
            edge_layers: PagedVec<Layer>,
            edge_revisions: PagedVec<u64>,
            edge_properties: PropertyColumns,
            edge_deleted: Arc<BitVec<u64, Lsb0>>,
            #[serde(default)]
            edge_tombstones: PersistentMap<()>,
            adjacency: Arc<Adjacency>,
            #[serde(default)]
            layout_version: u64,
        }
        let stored = StoredGraph::deserialize(deserializer)?;
        let graph = Self {
            revision: stored.revision,
            catalog: stored.catalog,
            node_lookup: stored.node_lookup,
            node_ids: stored.node_ids,
            node_layers: stored.node_layers,
            node_revisions: stored.node_revisions,
            node_labels: stored.node_labels,
            node_label_deltas: stored.node_label_deltas,
            node_properties: stored.node_properties,
            node_deleted: stored.node_deleted,
            node_tombstones: stored.node_tombstones,
            edge_lookup: stored.edge_lookup,
            edge_ids: stored.edge_ids,
            edge_sources: stored.edge_sources,
            edge_targets: stored.edge_targets,
            edge_types: stored.edge_types,
            edge_layers: stored.edge_layers,
            edge_revisions: stored.edge_revisions,
            edge_properties: stored.edge_properties,
            edge_deleted: stored.edge_deleted,
            edge_tombstones: stored.edge_tombstones,
            adjacency: stored.adjacency,
            layout_version: stored.layout_version,
            live_counts: OnceLock::new(),
            label_counts: OnceLock::new(),
            edge_type_counts: OnceLock::new(),
            device_changes: Arc::default(),
        };
        // Count at the cold read boundary. The first post-recovery insertion/deletion then uses
        // the same constant-time counter as every subsequent surgical mutation.
        graph.live_counts();
        graph.label_counts();
        graph.edge_type_counts();
        Ok(graph)
    }
}

#[derive(Clone, Debug, Default)]
struct GraphChangeJournal {
    revision: u64,
    property_patches: bool,
    // None denotes an inserted full row; Some contains only properties changed in this batch.
    nodes: BTreeMap<u32, Option<std::collections::BTreeSet<PropertyId>>>,
    edges: BTreeMap<u32, Option<std::collections::BTreeSet<PropertyId>>>,
    outgoing: std::collections::BTreeSet<u32>,
    incoming: std::collections::BTreeSet<u32>,
}

#[derive(Clone, Copy, Debug)]
struct LiveCounts {
    nodes: usize,
    edges: usize,
}

/// Number of physical graph layers (`Observed`, `Knowledge`, `Workspace`). Kept in step with the
/// optimizer statistics snapshot, which counts the same layers.
const LAYER_COUNT: usize = Layer::ALL.len();

/// Per-label and total live node counts, split by physical layer. `nodes[i]` is the count of live
/// (non-tombstoned) nodes in layer `i`; `per_label[label][i]` is the count of live nodes carrying
/// `label` in layer `i`.
#[derive(Clone, Debug, Default)]
struct LabelCounts {
    nodes: [u64; LAYER_COUNT],
    per_label: PersistentMap<[u64; LAYER_COUNT]>,
}

/// Per-relationship-type and total live edge counts, split by physical layer. Mirrors `LabelCounts`
/// for edges: `edges[i]` counts live edges in layer `i`, `per_type[type][i]` counts live edges of
/// `type` in layer `i`. Answers `count(r)` / `count(r:TYPE)` in O(1) without a statistics snapshot.
#[derive(Clone, Debug, Default)]
struct RelationshipCounts {
    edges: [u64; LAYER_COUNT],
    per_type: PersistentMap<[u64; LAYER_COUNT]>,
}

fn adjust_count(count: &mut u64, add: bool) {
    *count = if add {
        count.saturating_add(1)
    } else {
        count.saturating_sub(1)
    };
}

fn adjust_layer_count(
    counts: &mut PersistentMap<[u64; LAYER_COUNT]>,
    key: u128,
    layer: usize,
    add: bool,
) {
    let mut value = counts.get(key).copied().unwrap_or_default();
    adjust_count(&mut value[layer], add);
    counts.insert_cow(key, value);
}

#[cfg(test)]
thread_local! {
    static COUNT_COLD_SCAN_ROWS: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

/// Folds a per-layer counter array down to the layers selected by `mask`.
fn count_layers(counts: &[u64; LAYER_COUNT], mask: LayerMask) -> u64 {
    Layer::ALL
        .into_iter()
        .enumerate()
        .filter(|(_, layer)| mask.contains_layer(*layer))
        .map(|(index, _)| counts[index])
        .fold(0_u64, u64::saturating_add)
}

impl GraphStore {
    /// Returns the O(1) physical dense-layout epoch used to fence resident row ordinals.
    #[must_use]
    pub const fn layout_version(&self) -> u64 {
        self.layout_version
    }

    /// Publishes one accelerator-owned immutable bulk generation as the canonical CPU-readable
    /// graph base. Sparse maps and later point mutations remain bounded COW overlays.
    pub fn rebind_shared(&mut self, backing: GraphSharedBacking) -> Result<()> {
        let node_count = backing.node_ids.len();
        let edge_count = backing.edge_ids.len();
        if backing.revision != self.revision
            || backing.layout_version != self.layout_version
            || backing.node_layers.len() != node_count
            || backing.node_revisions.len() != node_count
            || backing.node_active.len() != node_count
            || backing.node_labels.rows() != node_count
            || backing.node_properties.rows() != node_count
            || backing.edge_sources.len() != edge_count
            || backing.edge_targets.len() != edge_count
            || backing.edge_types.len() != edge_count
            || backing.edge_layers.len() != edge_count
            || backing.edge_revisions.len() != edge_count
            || backing.edge_active.len() != edge_count
            || backing.edge_properties.rows() != edge_count
            || backing.outgoing.offsets().len() != node_count.saturating_add(1)
            || backing.incoming.offsets().len() != node_count.saturating_add(1)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "shared graph backing does not match the published graph generation",
            ));
        }
        self.node_ids = backing.node_ids;
        self.node_layers = backing.node_layers;
        self.node_revisions = backing.node_revisions;
        self.node_labels = Arc::new(backing.node_labels);
        self.node_label_deltas = PersistentMap::default();
        self.node_properties = backing.node_properties;
        self.edge_ids = backing.edge_ids;
        self.edge_sources = backing.edge_sources;
        self.edge_targets = backing.edge_targets;
        self.edge_types = backing.edge_types;
        self.edge_layers = backing.edge_layers;
        self.edge_revisions = backing.edge_revisions;
        self.edge_properties = backing.edge_properties;
        // Keep the canonical adjacency that the mutation path already updated from the bounded
        // changed-row delta. Metal represents changed topology as replacement-row overlays over
        // its immutable cold CSR; `GraphSharedBacking` intentionally carries only that cold CSR.
        // Rebinding it here discarded valid incoming/outgoing deltas (and made a reversed edge
        // fail snapshot validation). Preserving this Arc is O(1), tracks only the rows touched by
        // the write, and never turns shared publication into an O(all edges) cold rebuild.
        let _resident_cold_adjacency = (backing.outgoing, backing.incoming);
        Ok(())
    }

    #[must_use]
    pub fn node_slot_count(&self) -> usize {
        self.node_ids.len()
    }

    #[must_use]
    pub fn edge_slot_count(&self) -> usize {
        self.edge_ids.len()
    }

    /// Validates a node-property value against the canonical typed column without mutating it.
    pub fn validate_node_property_value(
        &self,
        property: PropertyId,
        value: &ScalarValue,
    ) -> Result<()> {
        self.node_properties.validate_value(property, value)
    }

    /// Validates a relationship-property value against the canonical typed column without
    /// mutating it.
    pub fn validate_edge_property_value(
        &self,
        property: PropertyId,
        value: &ScalarValue,
    ) -> Result<()> {
        self.edge_properties.validate_value(property, value)
    }

    fn begin_device_change(&mut self, revision: u64) {
        if self.device_changes.revision != revision {
            if self.device_changes.property_patches {
                Arc::make_mut(&mut self.device_changes).revision = revision;
                return;
            }
            self.device_changes = Arc::new(GraphChangeJournal {
                revision,
                ..GraphChangeJournal::default()
            });
        }
    }

    /// Starts a new publication batch without changing canonical rows or their revision.
    /// Explicit transaction statements share a provisional revision but publish separate deltas.
    pub fn begin_device_delta_batch(&mut self) {
        self.device_changes = Arc::new(GraphChangeJournal {
            revision: self.revision,
            property_patches: true,
            ..GraphChangeJournal::default()
        });
    }

    fn record_node_change(&mut self, revision: u64, dense: u32) {
        self.begin_device_change(revision);
        Arc::make_mut(&mut self.device_changes)
            .nodes
            .entry(dense)
            .or_insert_with(|| Some(Default::default()));
    }

    fn record_edge_change(&mut self, revision: u64, dense: u32, source: u32, target: u32) {
        self.begin_device_change(revision);
        let changes = Arc::make_mut(&mut self.device_changes);
        changes
            .edges
            .entry(dense)
            .or_insert_with(|| Some(Default::default()));
        changes.outgoing.insert(source);
        changes.incoming.insert(target);
    }

    #[must_use]
    pub const fn revision(&self) -> u64 {
        self.revision
    }

    #[must_use]
    pub fn catalog(&self) -> &NameCatalog {
        &self.catalog
    }

    pub fn catalog_mut(&mut self) -> &mut NameCatalog {
        Arc::make_mut(&mut self.catalog)
    }

    #[must_use]
    pub fn node_count(&self) -> usize {
        self.live_counts().nodes
    }

    #[must_use]
    pub fn edge_count(&self) -> usize {
        self.live_counts().edges
    }

    fn live_counts(&self) -> LiveCounts {
        *self.live_counts.get_or_init(|| LiveCounts {
            nodes: (0..self.node_ids.len())
                .filter(|row| !self.node_is_deleted(*row))
                .count(),
            edges: (0..self.edge_ids.len())
                .filter(|row| !self.edge_is_deleted(*row))
                .count(),
        })
    }

    fn set_live_counts(&mut self, counts: LiveCounts) {
        self.live_counts = OnceLock::new();
        let _ = self.live_counts.set(counts);
    }

    /// Builds the cold label counters once. Ordinary mutations preserve this generation cache.
    fn label_counts(&self) -> Arc<LabelCounts> {
        Arc::clone(self.label_counts.get_or_init(|| {
            let mut counts = LabelCounts::default();
            for row in 0..self.node_ids.len() {
                #[cfg(test)]
                COUNT_COLD_SCAN_ROWS.with(|count| count.set(count.get() + 1));
                if self.node_is_deleted(row) {
                    continue;
                }
                let layer = self
                    .node_layers
                    .get(row)
                    .copied()
                    .unwrap_or(Layer::Observed);
                let index = layer as usize;
                if let Some(slot) = counts.nodes.get_mut(index) {
                    *slot = slot.saturating_add(1);
                }
                for label in self.node_labels(row as u32) {
                    adjust_layer_count(&mut counts.per_label, u128::from(label.0), index, true);
                }
            }
            Arc::new(counts)
        }))
    }

    fn adjust_label_counts(&mut self, layer: Layer, labels: &[LabelId], add: bool, node: bool) {
        self.label_counts();
        if let Some(counts) = self.label_counts.get_mut() {
            let counts = Arc::make_mut(counts);
            let index = layer as usize;
            if node {
                adjust_count(&mut counts.nodes[index], add);
            }
            for label in labels {
                adjust_layer_count(&mut counts.per_label, u128::from(label.0), index, add);
            }
        }
    }

    /// O(1) (amortized) count of live nodes carrying `label` within `layers`.
    #[must_use]
    pub fn label_node_count(&self, label: LabelId, layers: LayerMask) -> u64 {
        self.label_counts()
            .per_label
            .get(u128::from(label.0))
            .map_or(0, |counts| count_layers(counts, layers))
    }

    /// O(1) (amortized) count of live nodes within `layers`, ignoring labels.
    #[must_use]
    pub fn node_count_in_layers(&self, layers: LayerMask) -> u64 {
        count_layers(&self.label_counts().nodes, layers)
    }

    /// Shares exact current count generations for bounded optimizer-statistics publication.
    #[must_use]
    pub fn optimizer_count_generations(
        &self,
    ) -> (
        PersistentMap<[u64; LAYER_COUNT]>,
        PersistentMap<[u64; LAYER_COUNT]>,
    ) {
        (
            self.label_counts().per_label.clone(),
            self.edge_type_counts().per_type.clone(),
        )
    }

    /// Builds the cold relationship counters once; edits update only the affected type and layer.
    fn edge_type_counts(&self) -> Arc<RelationshipCounts> {
        Arc::clone(self.edge_type_counts.get_or_init(|| {
            let mut counts = RelationshipCounts::default();
            for row in 0..self.edge_ids.len() {
                #[cfg(test)]
                COUNT_COLD_SCAN_ROWS.with(|count| count.set(count.get() + 1));
                if self.edge_is_deleted(row) {
                    continue;
                }
                let layer = self
                    .edge_layers
                    .get(row)
                    .copied()
                    .unwrap_or(Layer::Observed);
                let index = layer as usize;
                if let Some(slot) = counts.edges.get_mut(index) {
                    *slot = slot.saturating_add(1);
                }
                if let Some(relationship_type) = self.edge_types.get(row) {
                    adjust_layer_count(
                        &mut counts.per_type,
                        u128::from(relationship_type.0),
                        index,
                        true,
                    );
                }
            }
            Arc::new(counts)
        }))
    }

    fn adjust_edge_type_counts(&mut self, layer: Layer, kind: RelationshipTypeId, add: bool) {
        self.edge_type_counts();
        if let Some(counts) = self.edge_type_counts.get_mut() {
            let counts = Arc::make_mut(counts);
            let index = layer as usize;
            adjust_count(&mut counts.edges[index], add);
            adjust_layer_count(&mut counts.per_type, u128::from(kind.0), index, add);
        }
    }

    /// O(1) (amortized) count of live edges of `relationship_type` within `layers`.
    #[must_use]
    pub fn relationship_count(
        &self,
        relationship_type: RelationshipTypeId,
        layers: LayerMask,
    ) -> u64 {
        self.edge_type_counts()
            .per_type
            .get(u128::from(relationship_type.0))
            .map_or(0, |counts| count_layers(counts, layers))
    }

    /// O(1) (amortized) count of live edges within `layers`, ignoring relationship type.
    #[must_use]
    pub fn edge_count_in_layers(&self, layers: LayerMask) -> u64 {
        count_layers(&self.edge_type_counts().edges, layers)
    }

    fn node_is_deleted(&self, row: usize) -> bool {
        self.node_deleted.get(row).is_some_and(|deleted| *deleted)
            || self.node_tombstones.contains_key(row as u128)
    }

    fn edge_is_deleted(&self, row: usize) -> bool {
        self.edge_deleted.get(row).is_some_and(|deleted| *deleted)
            || self.edge_tombstones.contains_key(row as u128)
    }

    fn node_labels(&self, dense: u32) -> &[LabelId] {
        self.node_label_deltas
            .get(u128::from(dense))
            .map(|labels| labels.as_slice())
            .or_else(|| self.node_labels.get(dense))
            .unwrap_or(&[])
    }

    /// Returns whether the canonical node property is backed by one integer column.
    #[must_use]
    pub fn node_property_is_integer(&self, property: PropertyId) -> bool {
        matches!(
            self.node_properties.column(property),
            Some(super::TypedColumn::Integer { .. })
        )
    }

    /// Returns whether the canonical node property is backed by one Boolean column.
    #[must_use]
    pub fn node_property_is_boolean(&self, property: PropertyId) -> bool {
        matches!(
            self.node_properties.column(property),
            Some(super::TypedColumn::Boolean { .. })
        )
    }

    /// Returns whether the canonical node property is backed by one UTF-8 string column.
    #[must_use]
    pub fn node_property_is_string(&self, property: PropertyId) -> bool {
        matches!(
            self.node_properties.column(property),
            Some(super::TypedColumn::String { .. })
        )
    }

    /// Returns whether the canonical node property is backed by one floating-point column.
    #[must_use]
    pub fn node_property_is_float(&self, property: PropertyId) -> bool {
        matches!(
            self.node_properties.column(property),
            Some(super::TypedColumn::Float { .. })
        )
    }

    /// Returns whether the canonical node property is backed by one date column.
    #[must_use]
    pub fn node_property_is_date(&self, property: PropertyId) -> bool {
        matches!(
            self.node_properties.column(property),
            Some(super::TypedColumn::Date { .. })
        )
    }

    /// Returns whether the canonical node property is backed by one local-time column.
    #[must_use]
    pub fn node_property_is_local_time(&self, property: PropertyId) -> bool {
        matches!(
            self.node_properties.column(property),
            Some(super::TypedColumn::LocalTime { .. })
        )
    }

    /// Returns whether the canonical node property is backed by one zoned-time column.
    #[must_use]
    pub fn node_property_is_zoned_time(&self, property: PropertyId) -> bool {
        matches!(
            self.node_properties.column(property),
            Some(super::TypedColumn::ZonedTime { .. })
        )
    }

    /// Returns whether the canonical node property is backed by one local-datetime column.
    #[must_use]
    pub fn node_property_is_local_datetime(&self, property: PropertyId) -> bool {
        matches!(
            self.node_properties.column(property),
            Some(super::TypedColumn::LocalDateTime { .. })
        )
    }

    /// Returns whether the canonical node property is backed by one zoned-datetime column.
    #[must_use]
    pub fn node_property_is_zoned_datetime(&self, property: PropertyId) -> bool {
        matches!(
            self.node_properties.column(property),
            Some(super::TypedColumn::ZonedDateTime { .. })
        )
    }

    /// Returns whether an existing project-local node property column accepts this exact typed
    /// value. `None` means no node value has established a type for the declared property yet.
    #[must_use]
    pub fn node_property_accepts(&self, property: PropertyId, value: &ScalarValue) -> Option<bool> {
        self.node_properties.accepts(property, value)
    }

    /// Builds only rows touched by `revision`, including tombstones and affected adjacency rows.
    /// Dense row ordinals are stable between explicit compactions.
    pub fn device_delta(&self, revision: u64) -> Result<GraphDeviceDelta> {
        let current = self.device_changes.revision == revision;
        let nodes = self
            .device_changes
            .nodes
            .iter()
            .filter(|_| current)
            .map(|(&dense, changed)| {
                let changed = changed
                    .as_ref()
                    .filter(|_| self.device_changes.property_patches);
                let row = dense as usize;
                Ok(NodeDeviceDelta {
                    dense,
                    id: self.node_ids[row],
                    layer: self.node_layers[row],
                    revision: self.node_revisions[row],
                    active: !self.node_is_deleted(row),
                    labels: self.node_labels(dense).to_vec(),
                    properties_complete: changed.is_none(),
                    properties: changed.map_or_else(
                        || {
                            self.node_properties
                                .property_ids()
                                .filter_map(|property| {
                                    self.node_properties
                                        .get(dense, property)
                                        .map(|value| (property, value))
                                })
                                .collect()
                        },
                        |properties| {
                            properties
                                .iter()
                                .map(|&property| {
                                    (
                                        property,
                                        self.node_properties
                                            .get(dense, property)
                                            .unwrap_or(ScalarValue::Null),
                                    )
                                })
                                .collect()
                        },
                    ),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        let edges = self
            .device_changes
            .edges
            .iter()
            .filter(|_| current)
            .map(|(&dense, changed)| {
                let changed = changed
                    .as_ref()
                    .filter(|_| self.device_changes.property_patches);
                let row = dense as usize;
                Ok(EdgeDeviceDelta {
                    dense,
                    id: self.edge_ids[row],
                    source: self.edge_sources[row],
                    target: self.edge_targets[row],
                    relationship_type: self.edge_types[row],
                    layer: self.edge_layers[row],
                    revision: self.edge_revisions[row],
                    active: !self.edge_is_deleted(row),
                    properties_complete: changed.is_none(),
                    properties: changed.map_or_else(
                        || {
                            self.edge_properties
                                .property_ids()
                                .filter_map(|property| {
                                    self.edge_properties
                                        .get(dense, property)
                                        .map(|value| (property, value))
                                })
                                .collect()
                        },
                        |properties| {
                            properties
                                .iter()
                                .map(|&property| {
                                    (
                                        property,
                                        self.edge_properties
                                            .get(dense, property)
                                            .unwrap_or(ScalarValue::Null),
                                    )
                                })
                                .collect()
                        },
                    ),
                })
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(GraphDeviceDelta {
            revision: self.revision,
            node_capacity: self.node_ids.len(),
            edge_capacity: self.edge_ids.len(),
            nodes,
            edges,
            outgoing: self
                .device_changes
                .outgoing
                .iter()
                .copied()
                .filter(|_| current)
                .map(|dense| self.device_adjacency_row(dense, true))
                .collect::<Result<_>>()?,
            incoming: self
                .device_changes
                .incoming
                .iter()
                .copied()
                .filter(|_| current)
                .map(|dense| self.device_adjacency_row(dense, false))
                .collect::<Result<_>>()?,
        })
    }

    /// Returns stable node and relationship identities touched by one committed graph revision.
    /// Tombstoned rows remain present so local derived views can remove their old representation.
    pub fn change_ids(&self, revision: u64) -> Result<GraphChangeIds> {
        if self.device_changes.revision != revision {
            return Ok(GraphChangeIds::default());
        }
        let nodes = self
            .device_changes
            .nodes
            .keys()
            .copied()
            .map(|dense| {
                self.node_ids.get(dense as usize).copied().ok_or_else(|| {
                    Error::new(
                        ErrorCode::CorruptStorage,
                        "graph change journal references a missing node row",
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let edges = self
            .device_changes
            .edges
            .keys()
            .copied()
            .map(|dense| {
                self.edge_ids.get(dense as usize).copied().ok_or_else(|| {
                    Error::new(
                        ErrorCode::CorruptStorage,
                        "graph change journal references a missing relationship row",
                    )
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(GraphChangeIds { nodes, edges })
    }

    /// Enumerates every currently live relationship touching one live node. This is used only by
    /// local derived views whose relationship text includes endpoint semantics.
    pub fn incident_edge_ids(&self, node: NodeId) -> Result<Vec<EdgeId>> {
        let dense = self.live_node_dense(node)?;
        let mut rows = Vec::new();
        let mut incoming = Vec::new();
        self.adjacency.expand_out(dense, &mut rows);
        self.adjacency.expand_in(dense, &mut incoming);
        rows.extend(incoming);
        rows.sort_unstable_by_key(|(_, edge)| *edge);
        rows.dedup_by_key(|(_, edge)| *edge);
        let mut ids = rows
            .into_iter()
            .filter_map(|(_, dense)| self.edge_dense(dense).map(|edge| edge.id()))
            .collect::<Vec<_>>();
        ids.sort_unstable();
        ids.dedup();
        Ok(ids)
    }

    fn device_adjacency_row(&self, dense: u32, outgoing: bool) -> Result<AdjacencyRowDeviceDelta> {
        let mut merged = Vec::new();
        if outgoing {
            self.adjacency.expand_out(dense, &mut merged);
        } else {
            self.adjacency.expand_in(dense, &mut merged);
        }
        merged.retain(|(neighbor, edge)| {
            self.edge_dense(*edge).is_some() && self.node_dense(*neighbor).is_some()
        });
        merged.sort_unstable();
        let (neighbors, edges) = merged.into_iter().unzip();
        Ok(AdjacencyRowDeviceDelta {
            dense,
            neighbors,
            edges,
        })
    }

    /// Iterates every live node in stable dense-row order without allocating an object graph.
    pub fn nodes(&self) -> impl Iterator<Item = NodeView<'_>> {
        (0..self.node_ids.len()).filter_map(|dense| {
            let dense = u32::try_from(dense).ok()?;
            self.node_dense(dense)
        })
    }

    /// Iterates every live relationship in stable dense-row order.
    pub fn edges(&self) -> impl Iterator<Item = EdgeView<'_>> {
        (0..self.edge_ids.len()).filter_map(|dense| {
            let dense = u32::try_from(dense).ok()?;
            self.edge_dense(dense)
        })
    }

    /// Produces an immutable compact view containing only the selected physical layers.
    pub fn filtered_clone(&self, layers: LayerMask) -> Result<Self> {
        let mut filtered = self.clone();
        for row in 0..filtered.node_ids.len() {
            if !layers.contains_layer(filtered.node_layers[row]) {
                filtered.node_tombstones.insert(row as u128, ());
            }
        }
        for row in 0..filtered.edge_ids.len() {
            if !layers.contains_layer(filtered.edge_layers[row])
                || filtered.node_is_deleted(filtered.edge_sources[row] as usize)
                || filtered.node_is_deleted(filtered.edge_targets[row] as usize)
            {
                filtered.edge_tombstones.insert(row as u128, ());
            }
        }
        filtered.compact()?;
        Ok(filtered)
    }

    /// Returns whether a stable node identity has ever been allocated, including tombstones.
    #[must_use]
    pub fn contains_node_id(&self, id: NodeId) -> bool {
        stable_id_row(&self.node_lookup, id.0).is_some()
    }

    /// Shares the immutable stable-ID lookup generation with a resident image. Sparse resident
    /// publication can then copy only the radix path for an appended ID instead of rebuilding a
    /// second node-count-sized map from the flat ID column.
    #[must_use]
    pub fn node_lookup_generation(&self) -> PersistentMap<u32> {
        self.node_lookup.clone()
    }

    /// Shares the immutable stable-relationship-ID lookup generation with a resident image. This
    /// is the relationship counterpart of `node_lookup_generation` and avoids rebuilding an
    /// edge-count-sized host map during backend admission.
    #[must_use]
    pub fn edge_lookup_generation(&self) -> PersistentMap<u32> {
        self.edge_lookup.clone()
    }

    /// Reports the canonical queryable graph bytes using the same accounting as a resident
    /// snapshot, without materializing that snapshot merely to collect benchmark metadata.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        let node_fixed = self.node_ids.len().saturating_mul(
            size_of::<NodeId>() + size_of::<Layer>() + size_of::<u64>() + size_of::<bool>(),
        );
        let edge_fixed = self.edge_ids.len().saturating_mul(
            size_of::<EdgeId>()
                + 2 * size_of::<u32>()
                + size_of::<RelationshipTypeId>()
                + size_of::<Layer>()
                + size_of::<u64>()
                + size_of::<bool>(),
        );
        let adjacency = (self.adjacency.outgoing().offsets().len()
            + self.adjacency.incoming().offsets().len()
            + self.adjacency.outgoing().neighbors().len()
            + self.adjacency.incoming().neighbors().len()
            + self.adjacency.outgoing().edges().len()
            + self.adjacency.incoming().edges().len())
        .saturating_mul(size_of::<u32>());
        node_fixed
            .saturating_add(edge_fixed)
            .saturating_add(adjacency)
            .saturating_add(self.node_labels.estimated_bytes())
            .saturating_add(self.node_properties.estimated_bytes())
            .saturating_add(self.edge_properties.estimated_bytes())
    }

    /// Returns whether a stable relationship identity has ever been allocated, including tombstones.
    #[must_use]
    pub fn contains_edge_id(&self, id: EdgeId) -> bool {
        stable_id_row(&self.edge_lookup, id.0).is_some()
    }

    pub fn apply(&mut self, mutation: GraphMutation) -> Result<()> {
        match mutation {
            GraphMutation::DeclareLabel { name, id } => self.catalog_mut().declare_label(name, id),
            GraphMutation::DeclareProperty { name, id } => {
                self.catalog_mut().declare_property(name, id)
            }
            GraphMutation::DeclareRelationshipType { name, id } => {
                self.catalog_mut().declare_relationship_type(name, id)
            }
            GraphMutation::InsertNode(input) => self.insert_node(input).map(|_| ()),
            GraphMutation::InsertEdge(input) => self.insert_edge(input).map(|_| ()),
            GraphMutation::SetNodeProperty {
                node,
                property,
                value,
                revision,
            } => self.set_node_property(node, property, value, revision),
            GraphMutation::AddNodeLabels {
                node,
                labels,
                revision,
            } => self.add_node_labels(node, labels, revision),
            GraphMutation::RemoveNodeLabels {
                node,
                labels,
                revision,
            } => self.remove_node_labels(node, labels, revision),
            GraphMutation::SetEdgeProperty {
                edge,
                property,
                value,
                revision,
            } => self.set_edge_property(edge, property, value, revision),
            GraphMutation::DeleteNode {
                node,
                detach,
                revision,
            } => self.delete_node(node, detach, revision),
            GraphMutation::DeleteEdge { edge, revision } => self.delete_edge(edge, revision),
        }
    }

    pub fn insert_node(&mut self, mut input: NodeInput) -> Result<u32> {
        let counts = self.live_counts();
        if stable_id_row(&self.node_lookup, input.id.0).is_some() {
            return Err(Error::new(
                ErrorCode::TransactionConflict,
                "node ID already exists",
            ));
        }
        if input.revision < self.revision {
            return Err(Error::invalid_data(
                "node revision moves graph state backwards",
            ));
        }
        input.labels.sort_unstable();
        input.labels.dedup();
        input.properties.sort_by_key(|(property, _)| *property);
        if input
            .properties
            .windows(2)
            .any(|pair| pair[0].0 == pair[1].0)
        {
            return Err(Error::invalid_data(
                "node input contains a duplicate property",
            ));
        }
        if input
            .labels
            .iter()
            .any(|label| self.catalog.label_name(*label).is_none())
            || input
                .properties
                .iter()
                .any(|(property, _)| self.catalog.property_name(*property).is_none())
        {
            return Err(Error::invalid_data(
                "node input references undeclared schema IDs",
            ));
        }
        let dense = u32::try_from(self.node_ids.len()).map_err(|_| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "dense node ID space exhausted",
            )
        })?;
        let revision = input.revision;
        self.node_properties.validate_push_row(&input.properties)?;
        // Every fallible shape/type check completes before canonical columns are extended.
        self.label_counts();
        self.node_properties.push_row(&input.properties)?;
        self.adjust_label_counts(input.layer, &input.labels, true, true);
        self.node_label_deltas
            .insert(u128::from(dense), Arc::new(input.labels));
        self.node_lookup.insert(stable_id_key(input.id.0), dense);
        self.node_ids.push(input.id);
        self.node_layers.push(input.layer);
        self.node_revisions.push(input.revision);
        self.revision = revision;
        self.set_live_counts(LiveCounts {
            nodes: counts.nodes + 1,
            edges: counts.edges,
        });
        self.record_node_change(revision, dense);
        Arc::make_mut(&mut self.device_changes)
            .nodes
            .insert(dense, None);
        Ok(dense)
    }

    pub fn insert_edge(&mut self, mut input: EdgeInput) -> Result<u32> {
        let counts = self.live_counts();
        if stable_id_row(&self.edge_lookup, input.id.0).is_some() {
            return Err(Error::new(
                ErrorCode::TransactionConflict,
                "relationship ID already exists",
            ));
        }
        if input.revision < self.revision {
            return Err(Error::invalid_data(
                "relationship revision moves graph state backwards",
            ));
        }
        let source = self.live_node_dense(input.source)?;
        let target = self.live_node_dense(input.target)?;
        input.properties.sort_by_key(|(property, _)| *property);
        if input
            .properties
            .windows(2)
            .any(|pair| pair[0].0 == pair[1].0)
        {
            return Err(Error::invalid_data(
                "relationship input contains a duplicate property",
            ));
        }
        if self
            .catalog
            .relationship_type_name(input.relationship_type)
            .is_none()
            || input
                .properties
                .iter()
                .any(|(property, _)| self.catalog.property_name(*property).is_none())
        {
            return Err(Error::invalid_data(
                "relationship input references undeclared schema IDs",
            ));
        }
        let dense = u32::try_from(self.edge_ids.len()).map_err(|_| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "dense relationship ID space exhausted",
            )
        })?;
        let revision = input.revision;
        self.edge_properties.validate_push_row(&input.properties)?;
        self.edge_type_counts();
        self.edge_properties.push_row(&input.properties)?;
        self.adjust_edge_type_counts(input.layer, input.relationship_type, true);
        self.edge_lookup.insert(stable_id_key(input.id.0), dense);
        self.edge_ids.push(input.id);
        self.edge_sources.push(source);
        self.edge_targets.push(target);
        self.edge_types.push(input.relationship_type);
        self.edge_layers.push(input.layer);
        self.edge_revisions.push(input.revision);
        Arc::make_mut(&mut self.adjacency).insert(source, target, dense);
        self.revision = revision;
        self.set_live_counts(LiveCounts {
            nodes: counts.nodes,
            edges: counts.edges + 1,
        });
        self.record_edge_change(revision, dense, source, target);
        Arc::make_mut(&mut self.device_changes)
            .edges
            .insert(dense, None);
        Ok(dense)
    }

    pub fn set_node_property(
        &mut self,
        node: NodeId,
        property: PropertyId,
        value: ScalarValue,
        revision: u64,
    ) -> Result<()> {
        self.ensure_forward_revision(revision)?;
        if self.catalog.property_name(property).is_none() {
            return Err(Error::invalid_data("node property ID is undeclared"));
        }
        let dense = self.live_node_dense(node)?;
        self.node_properties.set(dense, property, &value)?;
        self.node_revisions[dense as usize] = revision;
        self.revision = revision;
        self.record_node_change(revision, dense);
        if let Some(properties) = Arc::make_mut(&mut self.device_changes)
            .nodes
            .get_mut(&dense)
            .and_then(Option::as_mut)
        {
            properties.insert(property);
        }
        Ok(())
    }

    pub fn add_node_labels(
        &mut self,
        node: NodeId,
        mut labels: Vec<LabelId>,
        revision: u64,
    ) -> Result<()> {
        self.ensure_forward_revision(revision)?;
        labels.sort_unstable();
        labels.dedup();
        if labels
            .iter()
            .any(|label| self.catalog.label_name(*label).is_none())
        {
            return Err(Error::invalid_data("node label ID is undeclared"));
        }
        let dense = self.live_node_dense(node)?;
        let mut updated = self.node_labels(dense).to_vec();
        for label in labels {
            if let Err(position) = updated.binary_search(&label) {
                updated.insert(position, label);
                self.adjust_label_counts(self.node_layers[dense as usize], &[label], true, false);
            }
        }
        self.node_label_deltas
            .insert(u128::from(dense), Arc::new(updated));
        self.node_revisions[dense as usize] = revision;
        self.revision = revision;
        self.record_node_change(revision, dense);
        Ok(())
    }

    pub fn remove_node_labels(
        &mut self,
        node: NodeId,
        mut labels: Vec<LabelId>,
        revision: u64,
    ) -> Result<()> {
        self.ensure_forward_revision(revision)?;
        labels.sort_unstable();
        labels.dedup();
        if labels
            .iter()
            .any(|label| self.catalog.label_name(*label).is_none())
        {
            return Err(Error::invalid_data("node label ID is undeclared"));
        }
        let dense = self.live_node_dense(node)?;
        let current = self.node_labels(dense);
        let removed = current
            .iter()
            .copied()
            .filter(|label| labels.binary_search(label).is_ok())
            .collect::<Vec<_>>();
        let updated = current
            .iter()
            .copied()
            .filter(|label| labels.binary_search(label).is_err())
            .collect::<Vec<_>>();
        self.adjust_label_counts(self.node_layers[dense as usize], &removed, false, false);
        self.node_label_deltas
            .insert(u128::from(dense), Arc::new(updated));
        self.node_revisions[dense as usize] = revision;
        self.revision = revision;
        self.record_node_change(revision, dense);
        Ok(())
    }

    pub fn set_edge_property(
        &mut self,
        edge: EdgeId,
        property: PropertyId,
        value: ScalarValue,
        revision: u64,
    ) -> Result<()> {
        self.ensure_forward_revision(revision)?;
        if self.catalog.property_name(property).is_none() {
            return Err(Error::invalid_data(
                "relationship property ID is undeclared",
            ));
        }
        let dense = self.live_edge_dense(edge)?;
        self.edge_properties.set(dense, property, &value)?;
        self.edge_revisions[dense as usize] = revision;
        self.revision = revision;
        let row = dense as usize;
        let source = self.edge_sources[row];
        let target = self.edge_targets[row];
        self.record_edge_change(revision, dense, source, target);
        if let Some(properties) = Arc::make_mut(&mut self.device_changes)
            .edges
            .get_mut(&dense)
            .and_then(Option::as_mut)
        {
            properties.insert(property);
        }
        Ok(())
    }

    pub fn delete_edge(&mut self, edge: EdgeId, revision: u64) -> Result<()> {
        self.ensure_forward_revision(revision)?;
        let dense = self.live_edge_dense(edge)?;
        let counts = self.live_counts();
        let row = dense as usize;
        self.adjust_edge_type_counts(self.edge_layers[row], self.edge_types[row], false);
        self.edge_tombstones.insert(u128::from(dense), ());
        self.edge_revisions[row] = revision;
        Arc::make_mut(&mut self.adjacency).delete(
            self.edge_sources[row],
            self.edge_targets[row],
            dense,
        );
        self.revision = revision;
        self.set_live_counts(LiveCounts {
            nodes: counts.nodes,
            edges: counts.edges.saturating_sub(1),
        });
        let source = self.edge_sources[row];
        let target = self.edge_targets[row];
        self.record_edge_change(revision, dense, source, target);
        Ok(())
    }

    pub fn delete_node(&mut self, node: NodeId, detach: bool, revision: u64) -> Result<()> {
        self.ensure_forward_revision(revision)?;
        let dense = self.live_node_dense(node)?;
        let counts = self.live_counts();
        let mut adjacent = Vec::new();
        let mut incoming = Vec::new();
        self.adjacency.expand_out(dense, &mut adjacent);
        self.adjacency.expand_in(dense, &mut incoming);
        adjacent.extend(incoming);
        adjacent.sort_unstable_by_key(|(_, edge)| *edge);
        adjacent.dedup_by_key(|(_, edge)| *edge);
        let incident = adjacent
            .into_iter()
            .filter_map(|(_, edge)| self.edge_dense(edge).map(|edge| edge.id()))
            .collect::<Vec<_>>();
        if !detach && !incident.is_empty() {
            return Err(Error::new(
                ErrorCode::QueryType,
                "node still has relationships",
            ));
        }
        for edge in incident {
            self.delete_edge(edge, revision)?;
        }
        let labels = self.node_labels(dense).to_vec();
        self.adjust_label_counts(self.node_layers[dense as usize], &labels, false, true);
        self.node_tombstones.insert(u128::from(dense), ());
        self.node_revisions[dense as usize] = revision;
        self.revision = revision;
        self.set_live_counts(LiveCounts {
            nodes: counts.nodes.saturating_sub(1),
            edges: self.edge_count(),
        });
        self.record_node_change(revision, dense);
        Ok(())
    }

    #[must_use]
    pub fn node(&self, id: NodeId) -> Option<NodeView<'_>> {
        let dense = *stable_id_row(&self.node_lookup, id.0)?;
        (!self.node_is_deleted(dense as usize)).then_some(NodeView { graph: self, dense })
    }

    #[must_use]
    pub fn node_dense(&self, dense: u32) -> Option<NodeView<'_>> {
        if dense as usize >= self.node_ids.len() || self.node_is_deleted(dense as usize) {
            return None;
        }
        Some(NodeView { graph: self, dense })
    }

    #[must_use]
    pub fn edge(&self, id: EdgeId) -> Option<EdgeView<'_>> {
        let dense = *stable_id_row(&self.edge_lookup, id.0)?;
        self.edge_dense(dense)
    }

    #[must_use]
    pub fn edge_dense(&self, dense: u32) -> Option<EdgeView<'_>> {
        if dense as usize >= self.edge_ids.len() || self.edge_is_deleted(dense as usize) {
            return None;
        }
        Some(EdgeView { graph: self, dense })
    }

    pub fn scan_nodes(
        &self,
        label: Option<LabelId>,
        layers: LayerMask,
    ) -> impl Iterator<Item = NodeView<'_>> {
        let mut changed_labels = self.node_label_deltas.iter().peekable();
        let all_layers_visible = layers == LayerMask::ALL
            || self.label_counts.get().is_some_and(|counts| {
                Layer::ALL.iter().all(|layer| {
                    counts.nodes[*layer as usize] == 0 || layers.contains_layer(*layer)
                })
            });
        let mut row_layers = (!all_layers_visible).then(|| self.node_layers.iter());
        self.node_labels
            .iter()
            .chain(std::iter::repeat(&[][..]))
            .take(self.node_ids.len())
            .enumerate()
            .filter_map(move |(row, labels)| {
                let visible = match &mut row_layers {
                    Some(layers_iter) => layers.contains_layer(*layers_iter.next()?),
                    None => true,
                };
                let labels = if changed_labels
                    .peek()
                    .is_some_and(|(changed_row, _)| *changed_row == row as u128)
                {
                    changed_labels.next()?.1.as_slice()
                } else {
                    labels
                };
                if !visible || label.is_some_and(|label| !labels.contains(&label)) {
                    return None;
                }
                let dense = u32::try_from(row).ok()?;
                self.node_dense(dense)
            })
    }

    pub fn expand_out(
        &self,
        node: NodeId,
        relationship_type: Option<RelationshipTypeId>,
        layers: LayerMask,
    ) -> Result<Vec<(EdgeView<'_>, NodeView<'_>)>> {
        let source = self.live_node_dense(node)?;
        let mut merged = Vec::new();
        self.adjacency.expand_out(source, &mut merged);
        let mut result = Vec::new();
        for (target, edge_dense) in merged {
            let (Some(edge), Some(target)) = (self.edge_dense(edge_dense), self.node_dense(target))
            else {
                continue;
            };
            let source_visible = self
                .node_dense(source)
                .is_some_and(|source| layers.contains_layer(source.layer()));
            if source_visible
                && layers.contains_layer(edge.layer())
                && layers.contains_layer(target.layer())
                && relationship_type.is_none_or(|kind| kind == edge.relationship_type())
            {
                result.push((edge, target));
            }
        }
        result.sort_by_key(|(edge, target)| (target.id(), edge.id()));
        Ok(result)
    }

    /// Bounded one-hop expansion used by semantic-memory selection. The canonical adjacency row
    /// is merged in stable order and stops before allocating or returning more than `limit` rows.
    pub fn expand_out_bounded(
        &self,
        node: NodeId,
        relationship_type: Option<RelationshipTypeId>,
        layers: LayerMask,
        limit: usize,
    ) -> Result<Vec<(EdgeView<'_>, NodeView<'_>)>> {
        let source = self.live_node_dense(node)?;
        let mut merged = Vec::new();
        self.adjacency
            .expand_out_bounded(source, limit.saturating_mul(2).max(limit), &mut merged);
        let source_visible = self
            .node_dense(source)
            .is_some_and(|source| layers.contains_layer(source.layer()));
        let mut result = Vec::with_capacity(limit.min(merged.len()));
        if !source_visible {
            return Ok(result);
        }
        for (target, edge_dense) in merged {
            let (Some(edge), Some(target)) = (self.edge_dense(edge_dense), self.node_dense(target))
            else {
                continue;
            };
            if layers.contains_layer(edge.layer())
                && layers.contains_layer(target.layer())
                && relationship_type.is_none_or(|kind| kind == edge.relationship_type())
            {
                result.push((edge, target));
                if result.len() == limit {
                    break;
                }
            }
        }
        result.sort_by_key(|(edge, target)| (target.id(), edge.id()));
        Ok(result)
    }

    pub fn expand_in(
        &self,
        node: NodeId,
        relationship_type: Option<RelationshipTypeId>,
        layers: LayerMask,
    ) -> Result<Vec<(EdgeView<'_>, NodeView<'_>)>> {
        let target = self.live_node_dense(node)?;
        let mut merged = Vec::new();
        self.adjacency.expand_in(target, &mut merged);
        let mut result = Vec::new();
        for (source, edge_dense) in merged {
            let (Some(edge), Some(source)) = (self.edge_dense(edge_dense), self.node_dense(source))
            else {
                continue;
            };
            let target_visible = self
                .node_dense(target)
                .is_some_and(|target| layers.contains_layer(target.layer()));
            if target_visible
                && layers.contains_layer(edge.layer())
                && layers.contains_layer(source.layer())
                && relationship_type.is_none_or(|kind| kind == edge.relationship_type())
            {
                result.push((edge, source));
            }
        }
        result.sort_by_key(|(edge, source)| (source.id(), edge.id()));
        Ok(result)
    }

    /// Bounded incoming counterpart to `expand_out_bounded`.
    pub fn expand_in_bounded(
        &self,
        node: NodeId,
        relationship_type: Option<RelationshipTypeId>,
        layers: LayerMask,
        limit: usize,
    ) -> Result<Vec<(EdgeView<'_>, NodeView<'_>)>> {
        let target = self.live_node_dense(node)?;
        let mut merged = Vec::new();
        self.adjacency
            .expand_in_bounded(target, limit.saturating_mul(2).max(limit), &mut merged);
        let target_visible = self
            .node_dense(target)
            .is_some_and(|target| layers.contains_layer(target.layer()));
        let mut result = Vec::with_capacity(limit.min(merged.len()));
        if !target_visible {
            return Ok(result);
        }
        for (source, edge_dense) in merged {
            let (Some(edge), Some(source)) = (self.edge_dense(edge_dense), self.node_dense(source))
            else {
                continue;
            };
            if layers.contains_layer(edge.layer())
                && layers.contains_layer(source.layer())
                && relationship_type.is_none_or(|kind| kind == edge.relationship_type())
            {
                result.push((edge, source));
                if result.len() == limit {
                    break;
                }
            }
        }
        result.sort_by_key(|(edge, source)| (source.id(), edge.id()));
        Ok(result)
    }

    pub fn compact_adjacency(&mut self) -> Result<()> {
        let visible = self.visible_edge_triples();
        Arc::make_mut(&mut self.adjacency).compact(self.node_ids.len(), &visible)
    }

    /// Validates the complete canonical graph before a cold persistence or recovery boundary.
    ///
    /// This is deliberately not part of the incremental mutation path: it scans every canonical
    /// row and compares merged adjacency with the fixed endpoint columns. Snapshot construction and
    /// installation are already O(graph) cold paths, so they can afford to prove that a resident
    /// publication race or damaged file will not become the next recovery authority.
    pub fn validate_structure(&self) -> Result<()> {
        self.validate_fixed_column_cardinality()?;
        self.validate_node_identity()?;
        let node_count = self.node_ids.len();
        let edge_count = self.edge_ids.len();
        let mut edge_ids = BTreeSet::new();
        let mut expected_outgoing = vec![Vec::<(u32, u32)>::new(); node_count];
        let mut expected_incoming = vec![Vec::<(u32, u32)>::new(); node_count];
        for row in 0..edge_count {
            if self.edge_is_deleted(row) {
                continue;
            }
            let id = self.edge_ids[row];
            let source = self.edge_sources[row];
            let target = self.edge_targets[row];
            if source as usize >= node_count
                || target as usize >= node_count
                || self.node_is_deleted(source as usize)
                || self.node_is_deleted(target as usize)
            {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    format!(
                        "canonical relationship row {row} has an out-of-bounds endpoint: source={source} target={target} nodes={node_count}"
                    ),
                ));
            }
            if !edge_ids.insert(id)
                || self.edge_lookup.get(stable_id_key(id.0)) != Some(&(row as u32))
                || self
                    .catalog
                    .relationship_type_name(self.edge_types[row])
                    .is_none()
            {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    format!("canonical relationship row {row} has invalid identity or schema"),
                ));
            }
            expected_outgoing[source as usize].push((target, row as u32));
            expected_incoming[target as usize].push((source, row as u32));
        }

        let mut actual = Vec::new();
        for row in 0..node_count {
            expected_outgoing[row].sort_unstable();
            expected_incoming[row].sort_unstable();
            self.adjacency.expand_out(row as u32, &mut actual);
            if actual != expected_outgoing[row] {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    format!("canonical outgoing adjacency differs at node row {row}"),
                ));
            }
            self.adjacency.expand_in(row as u32, &mut actual);
            if actual != expected_incoming[row] {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    format!("canonical incoming adjacency differs at node row {row}"),
                ));
            }
        }
        Ok(())
    }

    /// Quarantines relationship rows that cannot name real canonical endpoints or schema, then
    /// compacts the surviving graph. This is recovery-only: corrupt relationships are discarded,
    /// while every valid node and relationship is retained and adjacency is rebuilt from the fixed
    /// columns. The caller must preserve the rejected snapshot as forensic evidence.
    pub fn quarantine_invalid_relationships(&mut self) -> Result<usize> {
        self.validate_fixed_column_cardinality()?;
        // Recovery is intentionally narrow. Node identity is the anchor for every surviving
        // datapoint, so node corruption fails the candidate snapshot instead of guessing which
        // node to retain. Only relationship rows and their rebuildable adjacency are quarantined.
        self.validate_node_identity()?;
        let node_count = self.node_ids.len();
        let mut seen = BTreeSet::new();
        let mut invalid = Vec::new();
        for row in 0..self.edge_ids.len() {
            if self.edge_is_deleted(row) {
                continue;
            }
            let id = self.edge_ids[row];
            let identity_valid =
                seen.insert(id) && self.edge_lookup.get(stable_id_key(id.0)) == Some(&(row as u32));
            let endpoints_valid = (self.edge_sources[row] as usize) < node_count
                && (self.edge_targets[row] as usize) < node_count
                && !self.node_is_deleted(self.edge_sources[row] as usize)
                && !self.node_is_deleted(self.edge_targets[row] as usize);
            let schema_valid = self
                .catalog
                .relationship_type_name(self.edge_types[row])
                .is_some();
            if !identity_valid || !endpoints_valid || !schema_valid {
                invalid.push(row);
            }
        }
        for row in &invalid {
            self.edge_tombstones.insert(*row as u128, ());
        }
        if invalid.is_empty() {
            self.compact_adjacency()?;
        } else {
            self.compact()?;
        }
        self.validate_structure()?;
        Ok(invalid.len())
    }

    fn validate_node_identity(&self) -> Result<()> {
        let mut node_ids = BTreeSet::new();
        for row in 0..self.node_ids.len() {
            if self.node_is_deleted(row) {
                continue;
            }
            let id = self.node_ids[row];
            if !node_ids.insert(id)
                || self.node_lookup.get(stable_id_key(id.0)) != Some(&(row as u32))
            {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "canonical node ID columns or lookup are inconsistent",
                ));
            }
        }
        Ok(())
    }

    fn validate_fixed_column_cardinality(&self) -> Result<()> {
        let node_count = self.node_ids.len();
        let edge_count = self.edge_ids.len();
        if self.node_ids.iter().count() != node_count
            || self.node_layers.len() != node_count
            || self.node_layers.iter().count() != node_count
            || self.node_revisions.len() != node_count
            || self.node_revisions.iter().count() != node_count
            || self.node_labels.rows() > node_count
            || self.node_properties.rows() != node_count
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "canonical node fixed columns have different cardinalities",
            ));
        }
        if self.edge_ids.iter().count() != edge_count
            || self.edge_sources.len() != edge_count
            || self.edge_sources.iter().count() != edge_count
            || self.edge_targets.len() != edge_count
            || self.edge_targets.iter().count() != edge_count
            || self.edge_types.len() != edge_count
            || self.edge_types.iter().count() != edge_count
            || self.edge_layers.len() != edge_count
            || self.edge_layers.iter().count() != edge_count
            || self.edge_revisions.len() != edge_count
            || self.edge_revisions.iter().count() != edge_count
            || self.edge_properties.rows() != edge_count
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "canonical relationship fixed columns have different cardinalities",
            ));
        }
        Ok(())
    }

    /// Reclaims tombstoned rows and remaps dense ordinals in stable-ID order.
    /// Replacement occurs only after the complete compacted image validates.
    pub fn compact(&mut self) -> Result<CompactionMap> {
        let layout_version = self.layout_version.checked_add(1).ok_or_else(|| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "graph dense-layout version exhausted",
            )
        })?;
        let mut compacted = Self {
            revision: self.revision,
            layout_version,
            catalog: self.catalog.clone(),
            ..Self::default()
        };
        let mut mapping = CompactionMap::default();
        let mut live_nodes: Vec<_> = (0..self.node_ids.len())
            .filter(|row| !self.node_is_deleted(*row))
            .collect();
        live_nodes.sort_unstable_by_key(|row| self.node_ids[*row]);
        for old in live_nodes {
            let dense = u32::try_from(compacted.node_ids.len()).map_err(|_| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "dense node ID space exhausted",
                )
            })?;
            let stable = self.node_ids[old];
            compacted.node_lookup.insert(stable_id_key(stable.0), dense);
            compacted.node_ids.push(stable);
            compacted.node_layers.push(self.node_layers[old]);
            compacted.node_revisions.push(self.node_revisions[old]);
            Arc::make_mut(&mut compacted.node_labels)
                .push(self.node_labels(old as u32).iter().copied())?;
            let properties: Vec<_> = self
                .node_properties
                .property_ids()
                .filter_map(|property| {
                    self.node_properties
                        .get(old as u32, property)
                        .map(|value| (property, value))
                })
                .collect();
            compacted.node_properties.push_row(&properties)?;
            mapping.nodes.insert(stable, dense);
        }
        let mut live_edges: Vec<_> = (0..self.edge_ids.len())
            .filter(|row| !self.edge_is_deleted(*row))
            .collect();
        live_edges.sort_unstable_by_key(|row| self.edge_ids[*row]);
        for old in live_edges {
            let source_id = self.node_ids[self.edge_sources[old] as usize];
            let target_id = self.node_ids[self.edge_targets[old] as usize];
            let (Some(source), Some(target)) =
                (mapping.nodes.get(&source_id), mapping.nodes.get(&target_id))
            else {
                continue;
            };
            let dense = u32::try_from(compacted.edge_ids.len()).map_err(|_| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "dense relationship ID space exhausted",
                )
            })?;
            let stable = self.edge_ids[old];
            compacted.edge_lookup.insert(stable_id_key(stable.0), dense);
            compacted.edge_ids.push(stable);
            compacted.edge_sources.push(*source);
            compacted.edge_targets.push(*target);
            compacted.edge_types.push(self.edge_types[old]);
            compacted.edge_layers.push(self.edge_layers[old]);
            compacted.edge_revisions.push(self.edge_revisions[old]);
            let properties: Vec<_> = self
                .edge_properties
                .property_ids()
                .filter_map(|property| {
                    self.edge_properties
                        .get(old as u32, property)
                        .map(|value| (property, value))
                })
                .collect();
            compacted.edge_properties.push_row(&properties)?;
            mapping.edges.insert(stable, dense);
        }
        compacted.compact_adjacency()?;
        compacted.set_live_counts(LiveCounts {
            nodes: compacted.node_ids.len(),
            edges: compacted.edge_ids.len(),
        });
        compacted.label_counts();
        compacted.edge_type_counts();
        *self = compacted;
        Ok(mapping)
    }

    pub fn snapshot(&self) -> Result<GraphSnapshot> {
        let visible = self.visible_edge_triples();
        let outgoing = Csr::build(self.node_ids.len(), &visible)?;
        let incoming = Csr::build_transposed(self.node_ids.len(), &visible)?;
        let mut node_labels = PackedLists::default();
        for row in 0..self.node_ids.len() {
            node_labels.push(self.node_labels(row as u32).iter().copied())?;
        }
        Ok(GraphSnapshot {
            revision: self.revision,
            layout_version: self.layout_version,
            node_ids: self.node_ids.to_vec(),
            node_layers: self.node_layers.to_vec(),
            node_revisions: self.node_revisions.to_vec(),
            node_active: (0..self.node_ids.len())
                .map(|row| !self.node_is_deleted(row))
                .collect(),
            node_labels,
            node_properties: self.node_properties.clone(),
            edge_ids: self.edge_ids.to_vec(),
            edge_sources: self.edge_sources.to_vec(),
            edge_targets: self.edge_targets.to_vec(),
            edge_types: self.edge_types.to_vec(),
            edge_layers: self.edge_layers.to_vec(),
            edge_revisions: self.edge_revisions.to_vec(),
            edge_active: (0..self.edge_ids.len())
                .map(|row| !self.edge_is_deleted(row))
                .collect(),
            edge_properties: self.edge_properties.clone(),
            outgoing,
            incoming,
        })
    }

    fn visible_edge_triples(&self) -> Vec<(u32, u32, u32)> {
        self.edge_sources
            .iter()
            .zip(self.edge_targets.iter())
            .enumerate()
            .filter_map(|(row, (&source, &target))| {
                if self.edge_is_deleted(row)
                    || self.node_is_deleted(source as usize)
                    || self.node_is_deleted(target as usize)
                {
                    return None;
                }
                Some((source, target, u32::try_from(row).ok()?))
            })
            .collect()
    }

    fn live_node_dense(&self, id: NodeId) -> Result<u32> {
        self.node(id)
            .map(NodeView::dense)
            .ok_or_else(|| Error::new(ErrorCode::QueryType, "node does not exist"))
    }

    fn live_edge_dense(&self, id: EdgeId) -> Result<u32> {
        self.edge(id)
            .map(EdgeView::dense)
            .ok_or_else(|| Error::new(ErrorCode::QueryType, "relationship does not exist"))
    }

    fn ensure_forward_revision(&self, revision: u64) -> Result<()> {
        if revision < self.revision {
            return Err(Error::invalid_data(
                "mutation revision moves graph state backwards",
            ));
        }
        Ok(())
    }

    #[cfg(test)]
    fn detached_value_page_bytes_from(&self, previous: &Self) -> usize {
        self.node_revisions
            .detached_page_bytes_from(&previous.node_revisions)
            .saturating_add(
                self.node_properties
                    .detached_page_bytes_from(&previous.node_properties),
            )
            .saturating_add(
                self.edge_revisions
                    .detached_page_bytes_from(&previous.edge_revisions),
            )
            .saturating_add(
                self.edge_properties
                    .detached_page_bytes_from(&previous.edge_properties),
            )
    }
}

#[cfg(test)]
mod cow_tests {
    use std::sync::Arc;

    use crate::{DocumentItem, DocumentList};
    use serde::Serialize;

    use super::*;

    #[test]
    fn schema_optimizer_generation_changes_only_with_declared_names() -> Result<()> {
        let mut catalog = NameCatalog::default();
        let empty = catalog.optimizer_generation();
        assert_eq!(catalog.optimizer_generation(), empty);
        let pinned = catalog.clone();
        catalog.intern_label("Document")?;
        assert_ne!(catalog.optimizer_generation(), empty);
        assert_eq!(pinned.optimizer_generation(), empty);
        let unchanged = catalog.clone();
        catalog.intern_label("Document")?;
        assert_eq!(
            catalog.optimizer_generation(),
            unchanged.optimizer_generation()
        );
        assert!(
            catalog
                .labels
                .digests
                .shared_with(&unchanged.labels.digests)
        );
        Ok(())
    }

    #[test]
    fn surgical_schema_names_share_dirty_payloads_and_bound_changed_paths() -> Result<()> {
        let mut measurements = Vec::new();
        for rows in [4_096_u64, 16_384] {
            let mut names = SchemaNames::default();
            let mut next = 0;
            for id in 0..rows {
                names.declare(
                    &mut next,
                    format!("{id:016x}:{}", "dirty-名-".repeat(128)),
                    id,
                    "fixture",
                )?;
            }
            let mut catalog = NameCatalog {
                labels: names.clone(),
                properties: names.clone(),
                relationship_types: names,
                next_label: next,
                next_property: next,
                next_relationship_type: next,
                optimizer_generation_cache: OnceLock::new(),
            };
            let pinned = catalog.clone();
            let old_generation = pinned.optimizer_generation();
            let id = 1_u64 << 60;
            let changed = format!("changed-{}", "long-語-".repeat(1_024));
            catalog.declare_label(changed.clone(), LabelId(id))?;
            catalog.declare_property(changed.clone(), PropertyId(id))?;
            catalog.declare_relationship_type(changed.clone(), RelationshipTypeId(id))?;
            let bytes = catalog.labels.detached_bytes_from(&pinned.labels)
                + catalog.properties.detached_bytes_from(&pinned.properties)
                + catalog
                    .relationship_types
                    .detached_bytes_from(&pinned.relationship_types);
            assert!(
                bytes < 256 * 1_024,
                "schema copied {bytes} metadata bytes for {rows} names"
            );
            measurements.push(bytes);
            assert_eq!(pinned.optimizer_generation(), old_generation);
            assert_ne!(catalog.optimizer_generation(), old_generation);
            assert_eq!(pinned.label_name(LabelId(id)), None);
            for (key, old_name) in pinned.labels.by_id.iter() {
                for current in [
                    &catalog.labels,
                    &catalog.properties,
                    &catalog.relationship_types,
                ] {
                    let current_name = current
                        .by_id
                        .get(key)
                        .ok_or_else(|| Error::internal("untouched schema name disappeared"))?;
                    assert!(
                        Arc::ptr_eq(old_name, current_name),
                        "untouched name payload was duplicated"
                    );
                }
            }
            let unchanged = catalog.clone();
            catalog.declare_property(changed.clone(), PropertyId(id))?;
            assert!(
                catalog
                    .properties
                    .by_id
                    .shared_with(&unchanged.properties.by_id)
            );
            assert!(
                catalog
                    .properties
                    .by_name
                    .shared_with(&unchanged.properties.by_name)
            );
            assert!(
                catalog
                    .properties
                    .digests
                    .shared_with(&unchanged.properties.digests)
            );
            assert!(
                catalog
                    .declare_property(changed, PropertyId(id + 1))
                    .is_err()
            );
            assert!(
                catalog
                    .declare_property("conflict".to_owned(), PropertyId(id))
                    .is_err()
            );
            assert!(
                catalog
                    .declare_property("overflow".to_owned(), PropertyId(u64::MAX))
                    .is_err()
            );
            assert_eq!(
                catalog.optimizer_generation(),
                unchanged.optimizer_generation()
            );
            assert_eq!(catalog.next_property_id(), id + 1);
        }
        eprintln!("surgical schema detached metadata bytes 4K/16K: {measurements:?}");
        assert!(measurements[1] <= measurements[0] + 32 * 1_024);
        Ok(())
    }

    #[test]
    fn surgical_label_overlay_keeps_large_neighbor_payload_shared() -> Result<()> {
        let mut graph = GraphStore::default();
        let mut labels = Vec::new();
        for id in 0..16_384 {
            labels.push(graph.catalog_mut().intern_label(&format!("label-{id}"))?);
        }
        graph.insert_node(NodeInput {
            id: NodeId(1),
            layer: Layer::Observed,
            revision: 1,
            labels: labels.clone(),
            properties: Vec::new(),
        })?;
        graph.insert_node(NodeInput {
            id: NodeId(2),
            layer: Layer::Observed,
            revision: 1,
            labels: vec![labels[0]],
            properties: Vec::new(),
        })?;
        let pinned = graph.clone();
        graph.add_node_labels(NodeId(2), vec![labels[1]], 2)?;
        let old = pinned
            .node_label_deltas
            .get(0)
            .ok_or_else(|| Error::internal("missing pinned labels"))?;
        let current = graph
            .node_label_deltas
            .get(0)
            .ok_or_else(|| Error::internal("missing neighbor labels"))?;
        assert!(Arc::ptr_eq(old, current));
        assert_eq!(current.len(), 16_384);
        assert_eq!(pinned.node_labels(1), &[labels[0]]);
        assert_eq!(graph.node_labels(1), &[labels[0], labels[1]]);
        assert!(
            graph
                .node_label_deltas
                .detached_node_bytes_from(&pinned.node_label_deltas)
                < 4_096
        );
        let mut old_wire = PersistentMap::default();
        for (key, value) in graph.node_label_deltas.iter() {
            old_wire.insert_cow(key, value.as_ref().clone());
        }
        let legacy = postcard::to_stdvec(&old_wire).map_err(|e| Error::internal(e.to_string()))?;
        let current = postcard::to_stdvec(&graph.node_label_deltas)
            .map_err(|e| Error::internal(e.to_string()))?;
        assert_eq!(legacy, current);
        let recovered: PersistentMap<Arc<Vec<LabelId>>> =
            postcard::from_bytes(&legacy).map_err(|e| Error::internal(e.to_string()))?;
        assert_eq!(recovered.get(0).map(|labels| labels.len()), Some(16_384));
        Ok(())
    }

    #[test]
    fn surgical_schema_merkle_is_order_independent_and_preserves_nine_field_wire() -> Result<()> {
        let mut first = NameCatalog::default();
        let mut second = NameCatalog::default();
        let entries = [
            (0, "zero"),
            (37, "unicode-名"),
            (1_u64 << 63, "sparse"),
            (u64::MAX - 1, "last"),
        ];
        for (id, name) in entries {
            first.declare_label(name.to_owned(), LabelId(id))?;
            first.declare_property(name.to_owned(), PropertyId(id))?;
            first.declare_relationship_type(name.to_owned(), RelationshipTypeId(id))?;
        }
        for (id, name) in entries.into_iter().rev() {
            second.declare_relationship_type(name.to_owned(), RelationshipTypeId(id))?;
            second.declare_property(name.to_owned(), PropertyId(id))?;
            second.declare_label(name.to_owned(), LabelId(id))?;
        }
        assert_eq!(first.optimizer_generation(), second.optimizer_generation());
        assert_eq!(
            first.labels().map(|(id, _)| id.0).collect::<Vec<_>>(),
            entries.map(|(id, _)| id)
        );
        let wire = NameCatalogWire {
            labels: first
                .labels()
                .map(|(id, name)| (name.to_owned(), id))
                .collect(),
            properties: first
                .properties()
                .map(|(id, name)| (name.to_owned(), id))
                .collect(),
            relationship_types: first
                .relationship_types()
                .map(|(id, name)| (name.to_owned(), id))
                .collect(),
            label_names: first
                .labels()
                .map(|(id, name)| (id, name.to_owned()))
                .collect(),
            property_names: first
                .properties()
                .map(|(id, name)| (id, name.to_owned()))
                .collect(),
            relationship_names: first
                .relationship_types()
                .map(|(id, name)| (id, name.to_owned()))
                .collect(),
            next_label: u64::MAX,
            next_property: u64::MAX,
            next_relationship_type: u64::MAX,
        };
        let old_bytes = postcard::to_stdvec(&wire).map_err(|e| Error::internal(e.to_string()))?;
        let new_bytes = postcard::to_stdvec(&first).map_err(|e| Error::internal(e.to_string()))?;
        assert_eq!(old_bytes, new_bytes);
        let recovered: NameCatalog =
            postcard::from_bytes(&old_bytes).map_err(|e| Error::internal(e.to_string()))?;
        assert_eq!(
            recovered.optimizer_generation(),
            first.optimizer_generation()
        );
        let mut old_cbor = Vec::new();
        let mut new_cbor = Vec::new();
        ciborium::ser::into_writer(&wire, &mut old_cbor)
            .map_err(|e| Error::internal(e.to_string()))?;
        ciborium::ser::into_writer(&first, &mut new_cbor)
            .map_err(|e| Error::internal(e.to_string()))?;
        assert_eq!(old_cbor, new_cbor);
        let recovered: NameCatalog = ciborium::de::from_reader(old_cbor.as_slice())
            .map_err(|e| Error::internal(e.to_string()))?;
        assert_eq!(
            recovered.optimizer_generation(),
            first.optimizer_generation()
        );
        let mut corrupt = wire;
        corrupt.labels.insert("forged".to_owned(), LabelId(0));
        let corrupt = postcard::to_stdvec(&corrupt).map_err(|e| Error::internal(e.to_string()))?;
        assert!(postcard::from_bytes::<NameCatalog>(&corrupt).is_err());
        let mut label_only = NameCatalog::default();
        let mut property_only = NameCatalog::default();
        label_only.intern_label("same")?;
        property_only.intern_property("same")?;
        assert_ne!(
            label_only.optimizer_generation(),
            property_only.optimizer_generation()
        );
        let mut collision_bucket = SchemaNames::default();
        collision_bucket.by_name.insert_cow(
            SchemaNames::name_hash("target"),
            Arc::new(vec![
                (Arc::from("different-name"), 7),
                (Arc::from("target"), 13),
            ]),
        );
        assert_eq!(collision_bucket.get("target"), Some(13));
        assert_eq!(collision_bucket.get("different-name"), None);
        Ok(())
    }

    /// Exact database-checkpoint CBOR shape from before the dense-layout epoch was introduced.
    #[derive(Serialize)]
    struct LegacyGraphSnapshot<'a> {
        revision: u64,
        node_ids: &'a [NodeId],
        node_layers: &'a [Layer],
        node_revisions: &'a [u64],
        node_active: &'a [bool],
        node_labels: &'a PackedLists<LabelId>,
        node_properties: &'a PropertyColumns,
        edge_ids: &'a [EdgeId],
        edge_sources: &'a [u32],
        edge_targets: &'a [u32],
        edge_types: &'a [RelationshipTypeId],
        edge_layers: &'a [Layer],
        edge_revisions: &'a [u64],
        edge_active: &'a [bool],
        edge_properties: &'a PropertyColumns,
        outgoing: &'a Csr,
        incoming: &'a Csr,
    }

    /// Exact database-checkpoint CBOR shape from before the dense-layout epoch was introduced.
    #[derive(Serialize)]
    struct LegacyGraphStore<'a> {
        revision: u64,
        catalog: &'a Arc<NameCatalog>,
        node_lookup: &'a PersistentMap<u32>,
        node_ids: &'a PagedVec<NodeId>,
        node_layers: &'a PagedVec<Layer>,
        node_revisions: &'a PagedVec<u64>,
        node_labels: &'a Arc<PackedLists<LabelId>>,
        node_label_deltas: &'a PersistentMap<Arc<Vec<LabelId>>>,
        node_properties: &'a PropertyColumns,
        node_deleted: &'a Arc<BitVec<u64, Lsb0>>,
        node_tombstones: &'a PersistentMap<()>,
        edge_lookup: &'a PersistentMap<u32>,
        edge_ids: &'a PagedVec<EdgeId>,
        edge_sources: &'a PagedVec<u32>,
        edge_targets: &'a PagedVec<u32>,
        edge_types: &'a PagedVec<RelationshipTypeId>,
        edge_layers: &'a PagedVec<Layer>,
        edge_revisions: &'a PagedVec<u64>,
        edge_properties: &'a PropertyColumns,
        edge_deleted: &'a Arc<BitVec<u64, Lsb0>>,
        edge_tombstones: &'a PersistentMap<()>,
        adjacency: &'a Arc<Adjacency>,
    }

    fn graph_with_integer_rows(rows: u64) -> Result<(GraphStore, PropertyId)> {
        let mut graph = GraphStore::default();
        let label = graph.catalog_mut().intern_label("Row")?;
        let property = graph.catalog_mut().intern_property("value")?;
        for id in 1..=rows {
            graph.insert_node(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: id,
                labels: vec![label],
                properties: vec![(property, ScalarValue::Integer(id as i64))],
            })?;
        }
        Ok((graph, property))
    }

    fn graph_with_one_relationship() -> Result<GraphStore> {
        let mut graph = GraphStore::default();
        let label = graph.catalog_mut().intern_label("Person")?;
        let relationship_type = graph.catalog_mut().intern_relationship_type("KNOWS")?;
        for (id, revision) in [(NodeId(1), 1), (NodeId(2), 2)] {
            graph.insert_node(NodeInput {
                id,
                layer: Layer::Observed,
                revision,
                labels: vec![label],
                properties: Vec::new(),
            })?;
        }
        graph.insert_edge(EdgeInput {
            id: EdgeId(10),
            source: NodeId(1),
            target: NodeId(2),
            relationship_type,
            layer: Layer::Observed,
            revision: 3,
            properties: Vec::new(),
        })?;
        graph.validate_structure()?;
        Ok(graph)
    }

    #[test]
    fn structural_validation_quarantines_only_invalid_relationships() -> Result<()> {
        let mut graph = graph_with_one_relationship()?;
        graph.edge_sources[0] = u32::MAX;

        let validation = graph.validate_structure().unwrap_err();
        assert_eq!(validation.code, ErrorCode::CorruptStorage);
        assert_eq!(graph.quarantine_invalid_relationships()?, 1);
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 0);
        assert!(graph.node(NodeId(1)).is_some());
        assert!(graph.node(NodeId(2)).is_some());
        graph.validate_structure()
    }

    #[test]
    fn structural_validation_rebuilds_missing_adjacency_without_data_loss() -> Result<()> {
        let mut graph = graph_with_one_relationship()?;
        graph.adjacency = Arc::new(Adjacency::default());

        let validation = graph.validate_structure().unwrap_err();
        assert_eq!(validation.code, ErrorCode::CorruptStorage);
        assert_eq!(graph.quarantine_invalid_relationships()?, 0);
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);
        assert!(graph.edge(EdgeId(10)).is_some());
        graph.validate_structure()
    }

    #[test]
    fn pre_layout_cbor_checkpoint_shapes_decode_with_zero_epoch() -> Result<()> {
        let (graph, property) = graph_with_integer_rows(3)?;
        let snapshot = graph.snapshot()?;
        let legacy_snapshot = LegacyGraphSnapshot {
            revision: snapshot.revision,
            node_ids: &snapshot.node_ids,
            node_layers: &snapshot.node_layers,
            node_revisions: &snapshot.node_revisions,
            node_active: &snapshot.node_active,
            node_labels: &snapshot.node_labels,
            node_properties: &snapshot.node_properties,
            edge_ids: &snapshot.edge_ids,
            edge_sources: &snapshot.edge_sources,
            edge_targets: &snapshot.edge_targets,
            edge_types: &snapshot.edge_types,
            edge_layers: &snapshot.edge_layers,
            edge_revisions: &snapshot.edge_revisions,
            edge_active: &snapshot.edge_active,
            edge_properties: &snapshot.edge_properties,
            outgoing: &snapshot.outgoing,
            incoming: &snapshot.incoming,
        };
        let mut legacy_snapshot_bytes = Vec::new();
        ciborium::ser::into_writer(&legacy_snapshot, &mut legacy_snapshot_bytes)
            .map_err(|error| Error::new(ErrorCode::CorruptStorage, error.to_string()))?;
        let decoded_snapshot: GraphSnapshot =
            ciborium::de::from_reader(legacy_snapshot_bytes.as_slice())
                .map_err(|error| Error::new(ErrorCode::CorruptStorage, error.to_string()))?;
        assert_eq!(decoded_snapshot.layout_version, 0);
        assert_eq!(decoded_snapshot.revision, snapshot.revision);
        assert_eq!(decoded_snapshot.node_ids, snapshot.node_ids);
        assert_eq!(
            decoded_snapshot.node_properties.get(1, property),
            Some(ScalarValue::Integer(2))
        );

        let legacy_store = LegacyGraphStore {
            revision: graph.revision,
            catalog: &graph.catalog,
            node_lookup: &graph.node_lookup,
            node_ids: &graph.node_ids,
            node_layers: &graph.node_layers,
            node_revisions: &graph.node_revisions,
            node_labels: &graph.node_labels,
            node_label_deltas: &graph.node_label_deltas,
            node_properties: &graph.node_properties,
            node_deleted: &graph.node_deleted,
            node_tombstones: &graph.node_tombstones,
            edge_lookup: &graph.edge_lookup,
            edge_ids: &graph.edge_ids,
            edge_sources: &graph.edge_sources,
            edge_targets: &graph.edge_targets,
            edge_types: &graph.edge_types,
            edge_layers: &graph.edge_layers,
            edge_revisions: &graph.edge_revisions,
            edge_properties: &graph.edge_properties,
            edge_deleted: &graph.edge_deleted,
            edge_tombstones: &graph.edge_tombstones,
            adjacency: &graph.adjacency,
        };
        let mut legacy_store_bytes = Vec::new();
        ciborium::ser::into_writer(&legacy_store, &mut legacy_store_bytes)
            .map_err(|error| Error::new(ErrorCode::CorruptStorage, error.to_string()))?;
        let mut decoded_store: GraphStore =
            ciborium::de::from_reader(legacy_store_bytes.as_slice())
                .map_err(|error| Error::new(ErrorCode::CorruptStorage, error.to_string()))?;
        assert_eq!(decoded_store.layout_version(), 0);
        assert_eq!(decoded_store.revision(), graph.revision());
        assert_eq!(
            decoded_store
                .node(NodeId(2))
                .and_then(|node| node.property(property)),
            Some(ScalarValue::Integer(2))
        );

        let revision = decoded_store.revision();
        decoded_store.compact()?;
        let mut current_bytes = Vec::new();
        ciborium::ser::into_writer(&decoded_store, &mut current_bytes)
            .map_err(|error| Error::new(ErrorCode::CorruptStorage, error.to_string()))?;
        let round_tripped: GraphStore = ciborium::de::from_reader(current_bytes.as_slice())
            .map_err(|error| Error::new(ErrorCode::CorruptStorage, error.to_string()))?;
        assert_eq!(round_tripped.revision(), revision);
        assert_eq!(round_tripped.layout_version(), 1);
        Ok(())
    }

    fn detached_bytes_after_point_update(rows: u64) -> Result<usize> {
        let (published, property) = graph_with_integer_rows(rows)?;
        let mut staged = published.clone();
        staged.set_node_property(NodeId(1_025), property, ScalarValue::Integer(-1), rows + 1)?;
        assert_eq!(
            published
                .node(NodeId(1_025))
                .and_then(|node| node.property(property)),
            Some(ScalarValue::Integer(1_025))
        );
        assert_eq!(
            staged
                .node(NodeId(1_025))
                .and_then(|node| node.property(property)),
            Some(ScalarValue::Integer(-1))
        );
        Ok(staged.detached_value_page_bytes_from(&published))
    }

    fn paged_values<T: Clone>(values: Vec<T>) -> PagedVec<T> {
        let mut paged = PagedVec::default();
        for value in values {
            paged.push(value);
        }
        paged
    }

    fn shared_backing(graph: &GraphStore) -> Result<GraphSharedBacking> {
        let snapshot = graph.snapshot()?;
        Ok(GraphSharedBacking {
            revision: snapshot.revision,
            layout_version: snapshot.layout_version,
            node_ids: paged_values(snapshot.node_ids),
            node_layers: paged_values(snapshot.node_layers),
            node_revisions: paged_values(snapshot.node_revisions),
            node_active: paged_values(snapshot.node_active.into_iter().map(u8::from).collect()),
            node_labels: snapshot.node_labels,
            node_properties: snapshot.node_properties,
            edge_ids: paged_values(snapshot.edge_ids),
            edge_sources: paged_values(snapshot.edge_sources),
            edge_targets: paged_values(snapshot.edge_targets),
            edge_types: paged_values(snapshot.edge_types),
            edge_layers: paged_values(snapshot.edge_layers),
            edge_revisions: paged_values(snapshot.edge_revisions),
            edge_active: paged_values(snapshot.edge_active.into_iter().map(u8::from).collect()),
            edge_properties: snapshot.edge_properties,
            outgoing: snapshot.outgoing,
            incoming: snapshot.incoming,
        })
    }

    #[test]
    fn canonical_scan_merges_shared_labels_sparse_changes_and_appended_rows() -> Result<()> {
        let (mut graph, _) = graph_with_integer_rows(32_768)?;
        let selected = graph.catalog_mut().intern_label("Selected")?;
        let body = graph.catalog_mut().intern_property("body")?;
        graph.set_node_property(
            NodeId(1),
            body,
            ScalarValue::Bytes(vec![0x5a; 256 * 1_024].into()),
            32_769,
        )?;
        graph.add_node_labels(NodeId(3), vec![selected], 32_769)?;
        graph.rebind_shared(shared_backing(&graph)?)?;
        let pinned = graph.clone();
        graph.add_node_labels(NodeId(7), vec![selected], 32_770)?;
        graph.delete_node(NodeId(3), true, 32_770)?;
        for (id, layer) in [(32_769, Layer::Observed), (32_770, Layer::Knowledge)] {
            graph.insert_node(NodeInput {
                id: NodeId(id),
                layer,
                revision: 32_770,
                labels: vec![selected],
                properties: Vec::new(),
            })?;
        }
        assert_eq!(
            graph
                .scan_nodes(Some(selected), LayerMask::OBSERVED)
                .map(|node| node.id())
                .collect::<Vec<_>>(),
            vec![NodeId(7), NodeId(32_769)]
        );
        assert_eq!(
            pinned
                .scan_nodes(Some(selected), LayerMask::ALL)
                .map(|node| node.id())
                .collect::<Vec<_>>(),
            vec![NodeId(3)]
        );
        assert_eq!(graph.scan_nodes(None, LayerMask::KNOWLEDGE).count(), 1);
        assert_eq!(graph.scan_nodes(None, LayerMask::ALL).count(), 32_769);
        graph.remove_node_labels(NodeId(7), vec![selected], 32_771)?;
        assert_eq!(
            graph
                .scan_nodes(Some(selected), LayerMask::OBSERVED)
                .map(|node| node.id())
                .collect::<Vec<_>>(),
            vec![NodeId(32_769)]
        );
        Ok(())
    }

    #[test]
    fn property_delta_preserves_unchanged_backing_fields_and_large_payloads() -> Result<()> {
        let mut graph = graph_with_one_relationship()?;
        let value = graph.catalog_mut().intern_property("value")?;
        let body = graph.catalog_mut().intern_property("body")?;
        let absent = graph.catalog_mut().intern_property("neighbor_only")?;
        let payload = ScalarValue::Bytes(vec![0x5a; 256 * 1_024].into());
        graph.set_node_property(NodeId(1), body, payload.clone(), 4)?;
        graph.set_edge_property(EdgeId(10), body, payload.clone(), 5)?;
        graph.set_node_property(NodeId(1), value, ScalarValue::Integer(0), 6)?;
        graph.set_edge_property(EdgeId(10), value, ScalarValue::Integer(0), 7)?;
        graph.set_node_property(NodeId(2), absent, ScalarValue::Integer(9), 7)?;
        let pinned = shared_backing(&graph)?;
        let mut backing = pinned.clone();

        for revision in 8..28 {
            graph.begin_device_delta_batch();
            graph.set_node_property(NodeId(1), value, ScalarValue::Integer(-1), revision)?;
            graph.set_edge_property(EdgeId(10), value, ScalarValue::Integer(-1), revision)?;
            let delta = graph.device_delta(revision)?;
            assert!(!delta.nodes[0].properties_complete);
            assert!(!delta.edges[0].properties_complete);
            assert_eq!(
                delta.nodes[0].properties,
                vec![(value, ScalarValue::Integer(-1))]
            );
            assert_eq!(
                delta.edges[0].properties,
                vec![(value, ScalarValue::Integer(-1))]
            );
            backing.apply_device_delta_columns(&delta)?;
            assert_eq!(
                backing
                    .node_layers
                    .detached_page_bytes_from(&pinned.node_layers),
                0
            );
            assert_eq!(
                backing
                    .node_active
                    .detached_page_bytes_from(&pinned.node_active),
                0
            );
            assert_eq!(
                backing
                    .edge_sources
                    .detached_page_bytes_from(&pinned.edge_sources),
                0
            );
            assert_eq!(
                backing
                    .edge_targets
                    .detached_page_bytes_from(&pinned.edge_targets),
                0
            );
            assert_eq!(
                backing
                    .edge_types
                    .detached_page_bytes_from(&pinned.edge_types),
                0
            );
            assert_eq!(
                backing
                    .edge_layers
                    .detached_page_bytes_from(&pinned.edge_layers),
                0
            );
            assert_eq!(
                backing
                    .edge_active
                    .detached_page_bytes_from(&pinned.edge_active),
                0
            );
            assert_eq!(
                backing.node_labels.get(0).unwrap().as_ptr(),
                pinned.node_labels.get(0).unwrap().as_ptr()
            );
            assert!(std::ptr::eq(
                backing.node_properties.physical_column(body).unwrap(),
                pinned.node_properties.physical_column(body).unwrap()
            ));
            assert!(std::ptr::eq(
                backing.node_properties.physical_column(absent).unwrap(),
                pinned.node_properties.physical_column(absent).unwrap()
            ));
            assert!(std::ptr::eq(
                backing.edge_properties.physical_column(body).unwrap(),
                pinned.edge_properties.physical_column(body).unwrap()
            ));
        }
        assert_eq!(
            backing.node_properties.get(0, value),
            Some(ScalarValue::Integer(-1))
        );
        assert_eq!(
            backing.edge_properties.get(0, value),
            Some(ScalarValue::Integer(-1))
        );
        assert_eq!(
            pinned.node_properties.get(0, value),
            Some(ScalarValue::Integer(0))
        );
        assert_eq!(
            pinned.edge_properties.get(0, value),
            Some(ScalarValue::Integer(0))
        );
        assert_eq!(backing.node_properties.get(0, body), Some(payload.clone()));
        graph.begin_device_delta_batch();
        graph.set_node_property(NodeId(1), value, ScalarValue::Null, 28)?;
        graph.set_edge_property(EdgeId(10), value, ScalarValue::Null, 28)?;
        let removed = graph.device_delta(28)?;
        assert_eq!(
            removed.nodes[0].properties,
            vec![(value, ScalarValue::Null)]
        );
        assert_eq!(
            removed.edges[0].properties,
            vec![(value, ScalarValue::Null)]
        );
        backing.apply_device_delta_columns(&removed)?;
        assert_eq!(backing.node_properties.get(0, value), None);
        assert_eq!(backing.edge_properties.get(0, value), None);
        graph.begin_device_delta_batch();
        graph.add_node_labels(NodeId(1), Vec::new(), 29)?;
        let metadata = graph.device_delta(29)?;
        assert!(metadata.nodes[0].properties.is_empty());
        assert!(!metadata.nodes[0].properties_complete);
        backing.apply_device_delta_columns(&metadata)?;
        assert_eq!(backing.node_properties.get(0, body), Some(payload.clone()));
        assert_eq!(backing.edge_properties.get(0, body), Some(payload));
        assert_eq!(
            pinned.node_properties.get(0, value),
            Some(ScalarValue::Integer(0))
        );
        graph.begin_device_delta_batch();
        graph.insert_node(NodeInput {
            id: NodeId(3),
            layer: Layer::Observed,
            revision: 30,
            labels: Vec::new(),
            properties: vec![(value, ScalarValue::Integer(1))],
        })?;
        graph.set_node_property(NodeId(3), value, ScalarValue::Integer(2), 30)?;
        let inserted = graph.device_delta(30)?;
        assert!(inserted.nodes[0].properties_complete);
        assert_eq!(
            inserted.nodes[0].properties,
            vec![(value, ScalarValue::Integer(2))]
        );
        backing.apply_device_delta_columns(&inserted)?;
        assert_eq!(
            backing.node_properties.get(2, value),
            Some(ScalarValue::Integer(2))
        );
        Ok(())
    }

    #[test]
    fn shared_rebind_rejects_a_different_dense_layout_generation() -> Result<()> {
        let (mut graph, _) = graph_with_integer_rows(3)?;
        let backing = shared_backing(&graph)?;
        let mut stale_layout = backing.clone();
        stale_layout.layout_version = stale_layout.layout_version.checked_add(1).unwrap();

        let error = graph.rebind_shared(stale_layout).unwrap_err();
        assert_eq!(error.code, ErrorCode::CorruptStorage);
        assert_eq!(graph.layout_version(), 0);
        assert_eq!(graph.node(NodeId(1)).map(NodeView::dense), Some(0));

        graph.rebind_shared(backing)?;
        assert_eq!(graph.layout_version(), 0);
        assert_eq!(graph.node(NodeId(1)).map(NodeView::dense), Some(0));
        Ok(())
    }

    #[test]
    fn shared_rebind_preserves_canonical_reversed_edge_adjacency() -> Result<()> {
        let mut graph = GraphStore::default();
        let label = graph.catalog_mut().intern_label("IcijNode")?;
        let relationship = graph.catalog_mut().intern_relationship_type("ICIJ_LINK")?;
        for (row, id) in [NodeId(165_428), NodeId(51_122)].into_iter().enumerate() {
            graph.insert_node(NodeInput {
                id,
                layer: Layer::Observed,
                revision: 1,
                labels: vec![label],
                properties: Vec::new(),
            })?;
            assert_eq!(graph.node(id).map(NodeView::dense), Some(row as u32));
        }
        graph.insert_edge(EdgeInput {
            id: EdgeId(1),
            source: NodeId(51_122),
            target: NodeId(165_428),
            relationship_type: relationship,
            layer: Layer::Observed,
            revision: 2,
            properties: Vec::new(),
        })?;
        graph.validate_structure()?;
        let delta_len = graph.adjacency.delta_len();
        assert_eq!(
            delta_len, 1,
            "fixture must exercise the bounded adjacency delta"
        );

        let snapshot = graph.snapshot()?;
        let empty_outgoing = Csr::build(snapshot.node_ids.len(), &[])?;
        let empty_incoming = Csr::build_transposed(snapshot.node_ids.len(), &[])?;
        graph.rebind_shared(GraphSharedBacking {
            revision: snapshot.revision,
            layout_version: snapshot.layout_version,
            node_ids: paged_values(snapshot.node_ids),
            node_layers: paged_values(snapshot.node_layers),
            node_revisions: paged_values(snapshot.node_revisions),
            node_active: paged_values(snapshot.node_active.into_iter().map(u8::from).collect()),
            node_labels: snapshot.node_labels,
            node_properties: snapshot.node_properties,
            edge_ids: paged_values(snapshot.edge_ids),
            edge_sources: paged_values(snapshot.edge_sources),
            edge_targets: paged_values(snapshot.edge_targets),
            edge_types: paged_values(snapshot.edge_types),
            edge_layers: paged_values(snapshot.edge_layers),
            edge_revisions: paged_values(snapshot.edge_revisions),
            edge_active: paged_values(snapshot.edge_active.into_iter().map(u8::from).collect()),
            edge_properties: snapshot.edge_properties,
            outgoing: empty_outgoing,
            incoming: empty_incoming,
        })?;

        assert_eq!(graph.adjacency.delta_len(), delta_len);
        graph.validate_structure()
    }

    #[test]
    fn point_mutation_detaches_bounded_pages_not_project_rows() -> Result<()> {
        let small = detached_bytes_after_point_update(4_096)?;
        let large = detached_bytes_after_point_update(32_768)?;
        assert!(
            small <= 3 * 16 * 1_024,
            "small mutation detached {small} bytes"
        );
        assert!(
            large <= 3 * 16 * 1_024,
            "large mutation detached {large} bytes"
        );
        // Both generations detach the same fixed set of pages. A partially filled validity page
        // copies only its live words, so its measured allocation may differ by at most one page.
        assert!(
            small.abs_diff(large) <= 16 * 1_024,
            "point-update copy grew beyond one fixed page: {small} vs {large} bytes"
        );
        Ok(())
    }

    #[test]
    fn change_ids_keep_detached_relationship_tombstones_for_local_derived_views() -> Result<()> {
        let mut graph = GraphStore::default();
        let label = graph.catalog_mut().intern_label("Person")?;
        let relationship_type = graph.catalog_mut().intern_relationship_type("KNOWS")?;
        for (id, revision) in [(NodeId(1), 1), (NodeId(2), 2)] {
            graph.apply(GraphMutation::InsertNode(NodeInput {
                id,
                layer: Layer::Observed,
                revision,
                labels: vec![label],
                properties: Vec::new(),
            }))?;
        }
        graph.apply(GraphMutation::InsertEdge(EdgeInput {
            id: EdgeId(7),
            source: NodeId(1),
            target: NodeId(2),
            relationship_type,
            layer: Layer::Observed,
            revision: 3,
            properties: Vec::new(),
        }))?;
        graph.apply(GraphMutation::DeleteNode {
            node: NodeId(1),
            detach: true,
            revision: 4,
        })?;

        assert_eq!(
            graph.change_ids(4)?,
            GraphChangeIds {
                nodes: vec![NodeId(1)],
                edges: vec![EdgeId(7)],
            }
        );
        assert!(graph.incident_edge_ids(NodeId(2))?.is_empty());
        Ok(())
    }

    #[test]
    fn surgical_statement_delta_does_not_republish_prior_same_revision_rows() -> Result<()> {
        for rows in [4_096_u64, 32_768] {
            let mut graph = GraphStore::default();
            let property = graph.catalog_mut().intern_property("value")?;
            for id in 1..=rows {
                graph.insert_node(NodeInput {
                    id: NodeId(id),
                    layer: Layer::Observed,
                    revision: 1,
                    labels: Vec::new(),
                    properties: vec![(property, ScalarValue::Integer(id as i64))],
                })?;
            }
            let pinned = graph.clone();
            assert!(Arc::ptr_eq(&graph.device_changes, &pinned.device_changes));
            graph.begin_device_delta_batch();
            assert_eq!(graph.revision(), pinned.revision());
            assert!(graph.device_delta(1)?.nodes.is_empty());
            graph.set_node_property(NodeId(17), property, ScalarValue::Integer(-1), 1)?;
            let delta = graph.device_delta(1)?;
            assert_eq!(delta.nodes.len(), 1);
            assert_eq!(delta.nodes[0].id, NodeId(17));
            assert_eq!(pinned.device_changes.nodes.len(), rows as usize);
            assert_eq!(
                pinned.node(NodeId(17)).unwrap().property(property),
                Some(ScalarValue::Integer(17))
            );
            assert_eq!(
                graph.node(NodeId(rows)).unwrap().property(property),
                Some(ScalarValue::Integer(rows as i64))
            );
        }
        Ok(())
    }

    #[test]
    fn surgical_counts_update_only_affected_entries_on_dirty_pinned_graphs() -> Result<()> {
        fn check(graph: &GraphStore) {
            for mask in [
                LayerMask::OBSERVED,
                LayerMask::KNOWLEDGE,
                LayerMask::WORKSPACE,
                LayerMask::ALL,
            ] {
                let mut labels = BTreeMap::<LabelId, u64>::new();
                let mut kinds = BTreeMap::<RelationshipTypeId, u64>::new();
                let mut nodes = 0;
                let mut edges = 0;
                for node in graph
                    .nodes()
                    .filter(|node| mask.contains_layer(node.layer()))
                {
                    nodes += 1;
                    for label in node.labels() {
                        *labels.entry(*label).or_default() += 1;
                    }
                }
                for edge in graph
                    .edges()
                    .filter(|edge| mask.contains_layer(edge.layer()))
                {
                    edges += 1;
                    *kinds.entry(edge.relationship_type()).or_default() += 1;
                }
                assert_eq!(graph.node_count_in_layers(mask), nodes);
                assert_eq!(graph.edge_count_in_layers(mask), edges);
                for (label, _) in graph.catalog.labels() {
                    assert_eq!(
                        graph.label_node_count(label, mask),
                        labels.get(&label).copied().unwrap_or(0)
                    );
                }
                for (kind, _) in graph.catalog.relationship_types() {
                    assert_eq!(
                        graph.relationship_count(kind, mask),
                        kinds.get(&kind).copied().unwrap_or(0)
                    );
                }
            }
        }
        let mut detached = Vec::new();
        for rows in [4_096_u64, 16_384] {
            let mut graph = GraphStore::default();
            let common = graph.catalog_mut().intern_label("Common")?;
            let marker = graph.catalog_mut().intern_label("Marker")?;
            let body = graph.catalog_mut().intern_property("body")?;
            let payload =
                ScalarValue::String(Arc::from("unrelated complete document ".repeat(192)));
            let mut labels = Vec::new();
            let mut kinds = Vec::new();
            for id in 1..=rows {
                let label = graph.catalog_mut().intern_label(&format!("Label{id}"))?;
                let kind = graph
                    .catalog_mut()
                    .intern_relationship_type(&format!("TYPE{id}"))?;
                labels.push(label);
                kinds.push(kind);
                graph.insert_node(NodeInput {
                    id: NodeId(id),
                    layer: Layer::ALL[(id as usize - 1) % LAYER_COUNT],
                    revision: 1,
                    labels: vec![common, label],
                    properties: vec![(body, payload.clone())],
                })?;
                if id > 1 {
                    graph.insert_edge(EdgeInput {
                        id: EdgeId(id - 1),
                        source: NodeId(id - 1),
                        target: NodeId(id),
                        relationship_type: kind,
                        layer: Layer::ALL[id as usize % LAYER_COUNT],
                        revision: 1,
                        properties: Vec::new(),
                    })?;
                }
            }
            graph.add_node_labels(NodeId(17), vec![marker], 2)?;
            graph.delete_edge(EdgeId(31), 2)?;
            graph.set_node_property(
                NodeId(9),
                body,
                ScalarValue::String(Arc::from("changed content ".repeat(300))),
                2,
            )?;
            check(&graph);
            let pinned = graph.clone();
            let old_labels = pinned.label_counts();
            let old_types = pinned.edge_type_counts();
            COUNT_COLD_SCAN_ROWS.with(|count| count.set(0));
            graph.add_node_labels(NodeId(10), vec![marker, marker], 3)?;
            graph.remove_node_labels(NodeId(11), vec![labels[10], labels[10], marker], 3)?;
            graph.remove_node_labels(NodeId(12), vec![common], 3)?;
            graph.remove_node_labels(NodeId(12), vec![common], 3)?;
            graph.insert_node(NodeInput {
                id: NodeId(rows + 1),
                layer: Layer::Knowledge,
                revision: 3,
                labels: vec![common, marker],
                properties: vec![(body, payload.clone())],
            })?;
            for (id, source, target) in [(rows + 1, rows, rows + 1), (rows + 2, 10, 10)] {
                graph.insert_edge(EdgeInput {
                    id: EdgeId(id),
                    source: NodeId(source),
                    target: NodeId(target),
                    relationship_type: kinds[0],
                    layer: Layer::Workspace,
                    revision: 3,
                    properties: Vec::new(),
                })?;
            }
            // A rejected non-detach delete must not adjust counts. The following detach sees
            // one self-loop twice in the adjacency union but decrements it exactly once.
            assert!(graph.delete_node(NodeId(10), false, 3).is_err());
            graph.delete_node(NodeId(10), true, 3)?;
            graph.delete_edge(EdgeId(rows + 1), 3)?;
            graph.delete_node(NodeId(rows + 1), false, 3)?;
            check(&graph);
            check(&pinned);
            COUNT_COLD_SCAN_ROWS.with(|count| {
                assert_eq!(
                    count.get(),
                    0,
                    "edit and following count reads scanned unrelated rows"
                )
            });
            assert_eq!(
                graph
                    .node(NodeId(rows))
                    .and_then(|node| node.property(body)),
                Some(payload)
            );
            assert!(Arc::ptr_eq(&old_labels, &pinned.label_counts()));
            assert!(Arc::ptr_eq(&old_types, &pinned.edge_type_counts()));
            let bytes = graph
                .label_counts()
                .per_label
                .detached_node_bytes_from(&old_labels.per_label)
                + graph
                    .edge_type_counts()
                    .per_type
                    .detached_node_bytes_from(&old_types.per_type);
            assert!(bytes < 128 * 1_024, "counter edits detached {bytes} bytes");
            detached.push(bytes);
        }
        assert!(
            detached[1] <= detached[0] * 2,
            "counter copying grew with unrelated graph size: {detached:?}"
        );
        eprintln!(
            "surgical counts: 4096/16384 rows detached {detached:?} radix bytes; zero cold rows scanned after edits"
        );
        Ok(())
    }

    #[test]
    fn surgical_counts_are_warm_after_recovery_and_explicit_compaction() -> Result<()> {
        let (mut graph, _) = graph_with_integer_rows(64)?;
        let label = graph.catalog_mut().intern_label("Counted")?;
        graph.add_node_labels(NodeId(1), vec![label], 65)?;
        let bytes =
            postcard::to_stdvec(&graph).map_err(|error| Error::internal(error.to_string()))?;
        let mut recovered: GraphStore =
            postcard::from_bytes(&bytes).map_err(|error| Error::internal(error.to_string()))?;
        for compact in [false, true] {
            if compact {
                recovered.compact()?;
            }
            assert!(recovered.label_counts.get().is_some());
            assert!(recovered.edge_type_counts.get().is_some());
            COUNT_COLD_SCAN_ROWS.with(|count| count.set(0));
            recovered.add_node_labels(NodeId(2), vec![label], 66)?;
            assert_eq!(recovered.label_node_count(label, LayerMask::ALL), 2);
            assert_eq!(recovered.node_count_in_layers(LayerMask::ALL), 64);
            assert_eq!(recovered.edge_count_in_layers(LayerMask::ALL), 0);
            COUNT_COLD_SCAN_ROWS.with(|count| assert_eq!(count.get(), 0));
        }
        assert_eq!(graph.label_node_count(label, LayerMask::ALL), 1);
        Ok(())
    }

    #[test]
    fn live_label_counters_track_every_node_and_label_mutation() -> Result<()> {
        let mut graph = GraphStore::default();
        let person = graph.catalog_mut().intern_label("Person")?;
        let admin = graph.catalog_mut().intern_label("Admin")?;

        // Two Observed nodes; one also in the Knowledge layer.
        graph.apply(GraphMutation::InsertNode(NodeInput {
            id: NodeId(1),
            layer: Layer::Observed,
            revision: 1,
            labels: vec![person],
            properties: Vec::new(),
        }))?;
        graph.apply(GraphMutation::InsertNode(NodeInput {
            id: NodeId(2),
            layer: Layer::Observed,
            revision: 2,
            labels: vec![person, admin],
            properties: Vec::new(),
        }))?;
        graph.apply(GraphMutation::InsertNode(NodeInput {
            id: NodeId(3),
            layer: Layer::Knowledge,
            revision: 3,
            labels: vec![person],
            properties: Vec::new(),
        }))?;

        assert_eq!(graph.node_count_in_layers(LayerMask::ALL), 3);
        assert_eq!(graph.label_node_count(person, LayerMask::ALL), 3);
        assert_eq!(graph.label_node_count(admin, LayerMask::ALL), 1);
        // Layer restriction is honored.
        assert_eq!(graph.label_node_count(person, LayerMask::OBSERVED), 2);
        assert_eq!(graph.label_node_count(person, LayerMask::KNOWLEDGE), 1);

        // Adding a label updates its counter.
        graph.apply(GraphMutation::AddNodeLabels {
            node: NodeId(1),
            labels: vec![admin],
            revision: 4,
        })?;
        assert_eq!(graph.label_node_count(admin, LayerMask::ALL), 2);

        // Removing a label updates its counter.
        graph.apply(GraphMutation::RemoveNodeLabels {
            node: NodeId(2),
            labels: vec![admin],
            revision: 5,
        })?;
        assert_eq!(graph.label_node_count(admin, LayerMask::ALL), 1);

        // Deleting a node drops it from both the total and its label counts.
        graph.apply(GraphMutation::DeleteNode {
            node: NodeId(3),
            detach: true,
            revision: 6,
        })?;
        assert_eq!(graph.node_count_in_layers(LayerMask::ALL), 2);
        assert_eq!(graph.label_node_count(person, LayerMask::ALL), 2);
        assert_eq!(graph.label_node_count(person, LayerMask::KNOWLEDGE), 0);
        Ok(())
    }

    #[test]
    fn cow_generation_round_trips_without_mutating_published_state() -> Result<()> {
        let (published, property) = graph_with_integer_rows(4_096)?;
        let mut staged = published.clone();
        staged.set_node_property(NodeId(2_048), property, ScalarValue::Integer(9), 4_097)?;
        let bytes = postcard::to_stdvec(&staged)
            .map_err(|error| Error::new(ErrorCode::CorruptStorage, error.to_string()))?;
        let recovered: GraphStore = postcard::from_bytes(&bytes)
            .map_err(|error| Error::new(ErrorCode::CorruptStorage, error.to_string()))?;
        assert_eq!(
            recovered
                .node(NodeId(2_048))
                .and_then(|node| node.property(property)),
            Some(ScalarValue::Integer(9))
        );
        assert_eq!(
            published
                .node(NodeId(2_048))
                .and_then(|node| node.property(property)),
            Some(ScalarValue::Integer(2_048))
        );
        Ok(())
    }

    #[test]
    fn surgical_structure_on_dirty_large_graphs_keeps_unrelated_payloads_and_adjacency()
    -> Result<()> {
        let mut copied = Vec::new();
        for rows in [4_096_u64, 32_768] {
            let mut graph = GraphStore::default();
            let label = graph.catalog_mut().intern_label("Vertex")?;
            let second_label = graph.catalog_mut().intern_label("Selected")?;
            let property = graph.catalog_mut().intern_property("body")?;
            let kind = graph.catalog_mut().intern_relationship_type("LINK")?;
            let body =
                ScalarValue::String(Arc::from("unrelated large source content ".repeat(128)));
            for id in 1..=rows {
                graph.insert_node(NodeInput {
                    id: NodeId(id),
                    layer: Layer::Observed,
                    revision: 1,
                    labels: vec![label],
                    properties: vec![(property, body.clone())],
                })?;
                if id > 1 {
                    graph.insert_edge(EdgeInput {
                        id: EdgeId(id - 1),
                        source: NodeId(id - 1),
                        target: NodeId(id),
                        relationship_type: kind,
                        layer: Layer::Observed,
                        revision: 1,
                        properties: Vec::new(),
                    })?;
                }
            }
            graph.compact_adjacency()?;
            graph.add_node_labels(NodeId(17), vec![second_label], 2)?;
            graph.delete_edge(EdgeId(31), 2)?;
            let pinned = graph.clone();
            let base = graph.adjacency.outgoing().offsets().as_ptr();
            graph.apply(GraphMutation::AddNodeLabels {
                node: NodeId(2),
                labels: vec![second_label],
                revision: 3,
            })?;
            graph.apply(GraphMutation::DeleteNode {
                node: NodeId(8),
                detach: true,
                revision: 3,
            })?;
            copied.push(graph.detached_value_page_bytes_from(&pinned));
            assert_eq!(graph.adjacency.outgoing().offsets().as_ptr(), base);
            assert!(pinned.node(NodeId(8)).is_some());
            assert!(graph.node(NodeId(8)).is_none());
            assert!(pinned.node(NodeId(2)).unwrap().labels().contains(&label));
            assert!(
                !pinned
                    .node(NodeId(2))
                    .unwrap()
                    .labels()
                    .contains(&second_label)
            );
            assert!(
                graph
                    .node(NodeId(2))
                    .unwrap()
                    .labels()
                    .contains(&second_label)
            );
            assert_eq!(
                graph.node(NodeId(rows)).unwrap().property(property),
                Some(body.clone())
            );
            assert_eq!(
                pinned
                    .expand_out(NodeId(7), Some(kind), LayerMask::ALL)?
                    .len(),
                1
            );
            assert!(
                graph
                    .expand_out(NodeId(7), Some(kind), LayerMask::ALL)?
                    .is_empty()
            );
        }
        assert!(copied.iter().all(|bytes| *bytes <= 48 * 1_024));
        assert!(
            copied[1] <= copied[0] + 16 * 1_024,
            "structural row copy grew with graph: {copied:?}"
        );
        Ok(())
    }

    #[test]
    fn surgical_adjacency_threshold_preserves_cold_rows_pinned_generation_and_restart() -> Result<()>
    {
        const ADJACENCY_SEAL_MIN_DELTAS: usize = 1_024;
        let mut graph = GraphStore::default();
        let label = graph.catalog_mut().intern_label("Vertex")?;
        let relationship_type = graph.catalog_mut().intern_relationship_type("LINK")?;
        for (id, revision) in [(NodeId(1), 1), (NodeId(2), 2)] {
            graph.apply(GraphMutation::InsertNode(NodeInput {
                id,
                layer: Layer::Observed,
                revision,
                labels: vec![label],
                properties: Vec::new(),
            }))?;
        }
        for edge in 1..ADJACENCY_SEAL_MIN_DELTAS as u64 {
            graph.apply(GraphMutation::InsertEdge(EdgeInput {
                id: EdgeId(edge),
                source: NodeId(1),
                target: NodeId(2),
                relationship_type,
                layer: Layer::Observed,
                revision: edge + 2,
                properties: Vec::new(),
            }))?;
        }
        assert_eq!(graph.adjacency.delta_len(), ADJACENCY_SEAL_MIN_DELTAS - 1);
        let pinned = graph.clone();
        let cold_offsets = graph.adjacency.outgoing().offsets().as_ptr();
        let final_edge = ADJACENCY_SEAL_MIN_DELTAS as u64;
        let final_revision = final_edge + 2;
        graph.apply(GraphMutation::InsertEdge(EdgeInput {
            id: EdgeId(final_edge),
            source: NodeId(1),
            target: NodeId(2),
            relationship_type,
            layer: Layer::Observed,
            revision: final_revision,
            properties: Vec::new(),
        }))?;

        assert_eq!(graph.adjacency.delta_len(), ADJACENCY_SEAL_MIN_DELTAS);
        assert_eq!(graph.adjacency.outgoing().offsets().as_ptr(), cold_offsets);
        assert_eq!(graph.edge_count(), ADJACENCY_SEAL_MIN_DELTAS);
        assert_eq!(pinned.edge_count(), ADJACENCY_SEAL_MIN_DELTAS - 1);
        assert!(pinned.edge(EdgeId(final_edge)).is_none());
        assert!(graph.edge(EdgeId(final_edge)).is_some());
        assert_eq!(
            pinned
                .expand_out(NodeId(1), Some(relationship_type), LayerMask::ALL)?
                .len(),
            ADJACENCY_SEAL_MIN_DELTAS - 1
        );
        assert_eq!(
            graph
                .expand_out(NodeId(1), Some(relationship_type), LayerMask::ALL)?
                .len(),
            ADJACENCY_SEAL_MIN_DELTAS
        );
        let device = graph.device_delta(final_revision)?;
        assert_eq!(device.edges.len(), 1);
        assert_eq!(device.outgoing.len(), 1);
        assert_eq!(device.incoming.len(), 1);

        let mut checkpoint = Vec::new();
        ciborium::ser::into_writer(&graph, &mut checkpoint)
            .map_err(|error| Error::internal(format!("test encoding failed: {error}")))?;
        let restored: GraphStore = ciborium::de::from_reader(checkpoint.as_slice())
            .map_err(|error| Error::internal(format!("test decoding failed: {error}")))?;
        assert_eq!(restored.adjacency.delta_len(), ADJACENCY_SEAL_MIN_DELTAS);
        assert!(
            restored.live_counts.get().is_some(),
            "recovery must build live counts before writes"
        );
        assert_eq!(
            restored
                .expand_out(NodeId(1), Some(relationship_type), LayerMask::ALL)?
                .len(),
            ADJACENCY_SEAL_MIN_DELTAS
        );
        Ok(())
    }

    #[test]
    fn surgical_document_rows_release_superseded_bytes_without_mutating_pinned_generation()
    -> Result<()> {
        fn document(value: char) -> Result<ScalarValue> {
            Ok(ScalarValue::List(DocumentList::new(vec![
                DocumentItem::Scalar(ScalarValue::String(Arc::from(
                    value.to_string().repeat(1_024),
                ))),
            ])?))
        }

        let mut graph = GraphStore::default();
        let label = graph.catalog_mut().intern_label("Document")?;
        let property = graph.catalog_mut().intern_property("payload")?;
        let original = document('a')?;
        let replacement = document('b')?;
        graph.apply(GraphMutation::InsertNode(NodeInput {
            id: NodeId(1),
            layer: Layer::Observed,
            revision: 1,
            labels: vec![label],
            properties: vec![(property, original.clone())],
        }))?;
        let pinned = graph.clone();

        for revision in 2..=256 {
            graph.apply(GraphMutation::SetNodeProperty {
                node: NodeId(1),
                property,
                value: replacement.clone(),
                revision,
            })?;
            let (allocated, live) = graph
                .node_properties
                .document_storage_bytes(property)
                .ok_or_else(|| Error::internal("document storage statistics are missing"))?;
            assert_eq!(
                allocated, live,
                "superseded row payload must be released immediately"
            );
        }
        assert_eq!(
            pinned
                .node(NodeId(1))
                .and_then(|node| node.property(property)),
            Some(original)
        );
        assert_eq!(
            graph
                .node(NodeId(1))
                .and_then(|node| node.property(property)),
            Some(replacement.clone())
        );

        let mut checkpoint = Vec::new();
        ciborium::ser::into_writer(&graph, &mut checkpoint)
            .map_err(|error| Error::internal(format!("test encoding failed: {error}")))?;
        let restored: GraphStore = ciborium::de::from_reader(checkpoint.as_slice())
            .map_err(|error| Error::internal(format!("test decoding failed: {error}")))?;
        let (allocated, live) = restored
            .node_properties
            .document_storage_bytes(property)
            .ok_or_else(|| Error::internal("restored document storage statistics are missing"))?;
        assert_eq!(allocated, live);
        assert_eq!(
            restored
                .node(NodeId(1))
                .and_then(|node| node.property(property)),
            Some(replacement)
        );
        Ok(())
    }
}
