//! Backend-neutral complete project image admitted as one revision-fenced unit.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{
    Bookmark, ProjectId, Result, ScalarValue,
    graph::{
        GraphDeviceDelta, GraphSharedBacking, GraphSnapshot, GraphStore, IndexCatalog,
        IndexDeviceImage, PersistentMap, ResolvedVectorMutation, SharedVectorBacking,
        TemporalCanonicalColumn, TemporalDeviceImage, TemporalSample, TemporalStore, stable_id_key,
    },
    types::EntityKind,
};

/// Complete queryable state for one project at one applied bookmark.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ResidentProjectImage {
    pub project: ProjectId,
    pub bookmark: Bookmark,
    pub graph: Arc<GraphSnapshot>,
    #[serde(skip)]
    pub node_id_rows: PersistentMap<u32>,
    #[serde(skip)]
    pub edge_id_rows: PersistentMap<u32>,
    pub temporal: TemporalDeviceImage,
    pub temporal_canonical: Vec<TemporalCanonicalColumn>,
    pub indexes: IndexDeviceImage,
}

/// Shared canonical allocations published by a unified-memory backend. It is ephemeral and never
/// serialized; checkpoints remain backend-neutral.
#[derive(Clone, Debug)]
pub struct ResidentSharedBacking {
    pub graph: GraphSharedBacking,
    pub temporal: Vec<TemporalCanonicalColumn>,
    pub vectors: Vec<SharedVectorBacking>,
}

/// One canonical temporal sample appended by a committed mutation.
#[derive(Clone, Debug, PartialEq)]
pub struct ResidentTemporalDelta {
    pub entity_kind: EntityKind,
    pub target: u64,
    pub sample: TemporalSample,
}

/// Bounded committed device update. Derived accelerators are invalidated and rebuilt separately;
/// canonical graph, temporal and vector values are never replaced wholesale on this path.
#[derive(Clone, Debug, PartialEq)]
pub struct ResidentProjectDelta {
    pub project: ProjectId,
    pub bookmark: Bookmark,
    pub graph: GraphDeviceDelta,
    pub temporal: Vec<ResidentTemporalDelta>,
    pub vectors: Vec<ResolvedVectorMutation>,
    pub invalidate_derived: bool,
}

impl ResidentProjectDelta {
    #[must_use]
    pub fn staging_bytes(&self) -> usize {
        let graph_rows = self
            .graph
            .nodes
            .iter()
            .map(|row| {
                size_of::<u32>()
                    + size_of::<u64>() * 2
                    + 2
                    + row.labels.len() * size_of::<u64>()
                    + row
                        .properties
                        .iter()
                        .map(|(_, value)| size_of::<u64>() + 1 + scalar_resident_bytes(value))
                        .fold(0_usize, usize::saturating_add)
            })
            .chain(self.graph.edges.iter().map(|row| {
                size_of::<u32>() * 3
                    + size_of::<u64>() * 3
                    + 2
                    + row
                        .properties
                        .iter()
                        .map(|(_, value)| size_of::<u64>() + 1 + scalar_resident_bytes(value))
                        .fold(0_usize, usize::saturating_add)
            }))
            .chain(
                self.graph
                    .outgoing
                    .iter()
                    .map(|row| (row.neighbors.len() + row.edges.len()) * size_of::<u32>()),
            )
            .chain(
                self.graph
                    .incoming
                    .iter()
                    .map(|row| (row.neighbors.len() + row.edges.len()) * size_of::<u32>()),
            )
            .fold(0_usize, usize::saturating_add);
        let temporal = self
            .temporal
            .iter()
            .map(|delta| 48_usize.saturating_add(scalar_resident_bytes(&delta.sample.value)))
            .fold(0_usize, usize::saturating_add);
        let vectors = self
            .vectors
            .iter()
            .map(|mutation| match mutation {
                ResolvedVectorMutation::Upsert { coordinates, .. } => {
                    coordinates.len().saturating_mul(size_of::<u16>()) + 32
                }
                ResolvedVectorMutation::Remove { .. } => 32,
            })
            .fold(0_usize, usize::saturating_add);
        graph_rows.saturating_add(temporal).saturating_add(vectors)
    }
}

fn scalar_resident_bytes(value: &ScalarValue) -> usize {
    match value {
        ScalarValue::Null => 0,
        ScalarValue::Boolean(_) => size_of::<u8>(),
        ScalarValue::Integer(_) | ScalarValue::Float(_) | ScalarValue::LocalTime(_) => {
            size_of::<u64>()
        }
        ScalarValue::String(value) => value.len(),
        ScalarValue::Bytes(value) => value.len(),
        ScalarValue::Date(_) => size_of::<i64>(),
        ScalarValue::ZonedTime { .. } => size_of::<i64>() + size_of::<i32>(),
        ScalarValue::LocalDateTime { .. } => size_of::<i64>() + size_of::<u32>(),
        ScalarValue::ZonedDateTime { timezone, .. } => {
            size_of::<i64>() + size_of::<u32>() + timezone.len()
        }
        ScalarValue::Duration { .. } => size_of::<i64>() * 3 + size_of::<i32>(),
        ScalarValue::List(value) => value.as_bytes().len(),
        ScalarValue::Map(value) => value.as_bytes().len(),
    }
}

impl ResidentProjectImage {
    /// Builds the exact image consumed by every execution backend.
    pub fn build(
        project: ProjectId,
        bookmark: Bookmark,
        graph: &GraphStore,
        temporal: &TemporalStore,
        indexes: &IndexCatalog,
    ) -> Result<Self> {
        let node_id_rows = graph.node_lookup_generation();
        let edge_id_rows = graph.edge_lookup_generation();
        Ok(Self {
            project,
            bookmark,
            graph: Arc::new(graph.snapshot()?),
            node_id_rows,
            edge_id_rows,
            temporal: temporal.device_image()?,
            temporal_canonical: temporal.canonical_columns()?,
            indexes: indexes.device_image()?,
        })
    }

    /// Compatibility constructor for graph-only operator tests.
    #[must_use]
    pub fn graph_only(graph: Arc<GraphSnapshot>) -> Self {
        let mut node_id_rows = PersistentMap::default();
        let mut edge_id_rows = PersistentMap::default();
        for (row, id) in graph.node_ids.iter().enumerate() {
            if let Ok(row) = u32::try_from(row) {
                node_id_rows.insert(stable_id_key(id.0), row);
            }
        }
        for (row, id) in graph.edge_ids.iter().enumerate() {
            if let Ok(row) = u32::try_from(row) {
                edge_id_rows.insert(stable_id_key(id.0), row);
            }
        }
        Self {
            project: ProjectId(Uuid::nil()),
            bookmark: Bookmark {
                term: 0,
                index: graph.revision,
            },
            graph,
            node_id_rows,
            edge_id_rows,
            temporal: TemporalDeviceImage::default(),
            temporal_canonical: Vec::new(),
            indexes: IndexDeviceImage::default(),
        }
    }

    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.graph
            .resident_bytes()
            .saturating_add(self.temporal.resident_bytes())
            .saturating_add(self.indexes.resident_bytes())
    }

    /// Bytes this image holds only until a backend has taken what it needs from it.
    ///
    /// The canonical temporal columns exist so a backend can rebind unified memory, and every
    /// backend releases them during intake — so they are correctly absent from `resident_bytes`.
    /// They are nonetheless live while admission reserves, so a reservation that covers only the
    /// resident total under-books the transient peak by the size of a second copy of the project's
    /// temporal data. Admission adds this so the governor books what is actually held.
    #[must_use]
    pub fn staging_only_bytes(&self) -> usize {
        self.temporal_canonical
            .iter()
            .map(TemporalCanonicalColumn::staging_bytes)
            .fold(0_usize, usize::saturating_add)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        Bookmark, Layer, NodeId, ProjectId,
        graph::{
            GraphMutation, GraphStore, IndexCatalog, NodeInput, TemporalDeclaration,
            TemporalSample, TemporalStore, TemporalType,
        },
        types::{EntityKind, ScalarValue},
    };

    #[test]
    fn a_temporal_project_reports_the_staging_copy_admission_must_also_reserve() -> Result<()> {
        // The canonical temporal columns are built with the image and released by whichever
        // backend takes it, so they are deliberately not resident. They are still held while
        // admission reserves, and a reservation covering only the resident total under-books the
        // real peak by the size of a second copy of the project's temporal data.
        let mut graph = GraphStore::default();
        let label = graph.catalog_mut().intern_label("Sensor")?;
        let property = graph.catalog_mut().intern_property("reading")?;
        graph.apply(GraphMutation::InsertNode(NodeInput {
            id: NodeId(1),
            layer: Layer::Observed,
            revision: 1,
            labels: vec![label],
            properties: Vec::new(),
        }))?;

        let mut temporal = TemporalStore::default();
        temporal.declare(
            TemporalDeclaration {
                entity_kind: EntityKind::Node,
                target: u64::from(label.0),
                property,
                value_type: TemporalType::Integer,
                retention_nanos: i64::MAX,
            },
            0,
        )?;
        for index in 0..256_u64 {
            temporal.append(
                EntityKind::Node,
                u64::from(label.0),
                TemporalSample {
                    entity_id: 1,
                    property,
                    event_time_nanos: i64::try_from(index).unwrap_or(0),
                    sequence_index: index.saturating_add(1),
                    value: ScalarValue::Integer(i64::try_from(index).unwrap_or(0)),
                },
                i64::try_from(index).unwrap_or(0),
            )?;
        }

        let image = ResidentProjectImage::build(
            ProjectId(Uuid::nil()),
            Bookmark { term: 1, index: 1 },
            &graph,
            &temporal,
            &IndexCatalog::default(),
        )?;
        assert!(
            image.staging_only_bytes() > 0,
            "a project with temporal samples carries a canonical staging copy"
        );

        // A graph-only image has nothing staged, so admission is unchanged for the common case.
        let plain = ResidentProjectImage::build(
            ProjectId(Uuid::nil()),
            Bookmark { term: 1, index: 1 },
            &graph,
            &TemporalStore::default(),
            &IndexCatalog::default(),
        )?;
        assert_eq!(plain.staging_only_bytes(), 0);
        Ok(())
    }
}
