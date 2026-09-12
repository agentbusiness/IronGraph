//! Bounded, rebuildable statistics for logical cardinality and physical device planning.

use std::collections::{BTreeMap, BTreeSet};

use crate::{
    Layer, ScalarValue,
    types::{LabelId, PropertyId, RelationshipTypeId},
};

use super::{
    GraphStore, IndexCatalog, LayerMask, OptimizerIndexStatistics,
    TemporalOptimizerColumnStatistics, TemporalStore,
};

const LAYER_COUNT: usize = Layer::ALL.len();
const DISTINCT_SKETCH_SIZE: usize = 256;
const NUMERIC_SAMPLE_SIZE: usize = 512;
const HISTOGRAM_BUCKETS: usize = 32;

/// Deterministic bounded equi-depth histogram. Bounds retain enough shape to interpolate within
/// a bucket instead of treating every sampled value as its bucket maximum.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct NumericHistogram {
    pub lower_bounds: Vec<f64>,
    pub upper_bounds: Vec<f64>,
    pub cumulative: Vec<u32>,
    pub distinct_per_bucket: Vec<u32>,
    pub sampled: u32,
}

impl NumericHistogram {
    fn from_sorted_sample(sample: &[f64]) -> Self {
        if sample.is_empty() {
            return Self::default();
        }
        let bucket_count = sample.len().min(HISTOGRAM_BUCKETS);
        let mut lower_bounds = Vec::with_capacity(bucket_count);
        let mut upper_bounds = Vec::with_capacity(bucket_count);
        let mut cumulative = Vec::with_capacity(bucket_count);
        let mut distinct_per_bucket = Vec::with_capacity(bucket_count);
        let mut start = 0;
        for bucket in 1..=bucket_count {
            let end = bucket.saturating_mul(sample.len()).div_ceil(bucket_count);
            let values = &sample[start..end];
            lower_bounds.push(values[0]);
            upper_bounds.push(sample[end.saturating_sub(1)]);
            cumulative.push(end as u32);
            distinct_per_bucket.push(
                values
                    .windows(2)
                    .filter(|pair| pair[0].total_cmp(&pair[1]).is_ne())
                    .count()
                    .saturating_add(1) as u32,
            );
            start = end;
        }
        Self {
            lower_bounds,
            upper_bounds,
            cumulative,
            distinct_per_bucket,
            sampled: sample.len() as u32,
        }
    }

    fn range_fraction(&self, operand: f64, include_equal: bool, less_than: bool) -> Option<f64> {
        if self.sampled == 0
            || self.upper_bounds.is_empty()
            || self.lower_bounds.len() != self.upper_bounds.len()
        {
            return None;
        }
        // A greater-than predicate needs the opposite CDF boundary before taking the complement:
        // `x > v = 1 - P(x <= v)` and `x >= v = 1 - P(x < v)`.
        let cdf_includes_equal = if less_than {
            include_equal
        } else {
            !include_equal
        };
        let mut below = 0_f64;
        for (position, (&lower, &upper)) in
            self.lower_bounds.iter().zip(&self.upper_bounds).enumerate()
        {
            let prior = if position == 0 {
                0
            } else {
                self.cumulative[position - 1]
            };
            let count = self.cumulative[position].saturating_sub(prior);
            if operand < lower || (operand == lower && !cdf_includes_equal) {
                below = f64::from(prior);
                break;
            }
            if operand > upper || (operand == upper && cdf_includes_equal) {
                below = f64::from(self.cumulative[position]);
                continue;
            }
            if lower.total_cmp(&upper).is_eq() {
                below = f64::from(prior)
                    + if cdf_includes_equal && operand == lower {
                        f64::from(count)
                    } else {
                        0.0
                    };
            } else {
                let width_fraction = ((operand - lower) / (upper - lower)).clamp(0.0, 1.0);
                let equality_mass = if cdf_includes_equal {
                    1.0 / f64::from(
                        self.distinct_per_bucket
                            .get(position)
                            .copied()
                            .unwrap_or(1)
                            .max(1),
                    )
                } else {
                    0.0
                };
                below =
                    f64::from(prior) + f64::from(count) * (width_fraction + equality_mass).min(1.0);
            }
            break;
        }
        let matching = if less_than {
            below
        } else {
            f64::from(self.sampled) - below
        };
        Some((matching / f64::from(self.sampled)).clamp(0.0, 1.0))
    }
}

/// Bounded scalar-column summary. It is derived from canonical graph columns and is never
/// checkpointed.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PropertyStatistics {
    pub present: u64,
    pub null_or_missing: u64,
    pub distinct: u64,
    pub minimum: Option<f64>,
    pub maximum: Option<f64>,
    pub histogram: NumericHistogram,
}

impl PropertyStatistics {
    /// Estimates equality selectivity, including the fraction of rows where the value is present.
    #[must_use]
    pub fn equality_selectivity(&self, parent_rows: u64) -> f64 {
        if parent_rows == 0 || self.present == 0 || self.distinct == 0 {
            return 0.0;
        }
        (self.present as f64 / parent_rows as f64) / self.distinct as f64
    }

    /// Estimates a numeric comparison using the deterministic bounded sample. Non-numeric
    /// columns deliberately return no estimate rather than inventing a fixed selectivity.
    #[must_use]
    pub fn numeric_range_selectivity(
        &self,
        parent_rows: u64,
        operand: f64,
        include_equal: bool,
        less_than: bool,
    ) -> Option<f64> {
        if parent_rows == 0 || self.present == 0 || !operand.is_finite() {
            return None;
        }
        let present_fraction = self.present as f64 / parent_rows as f64;
        self.histogram
            .range_fraction(operand, include_equal, less_than)
            .map(|fraction| present_fraction * fraction)
    }
}

/// Directional relationship fanout. Endpoint cardinalities use deterministic bounded sketches.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct FanoutStatistics {
    pub edges: u64,
    pub distinct_sources: u64,
    pub distinct_targets: u64,
    pub mean_outgoing_active: f64,
    pub mean_incoming_active: f64,
}

/// Immutable statistics generation tied to one canonical graph revision.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StatisticsSnapshot {
    pub graph_revision: u64,
    pub schema_generation: [u8; 32],
    pub index_generation: [u8; 32],
    pub resident_graph_bytes: u64,
    pub average_node_row_bytes: u64,
    pub average_relationship_row_bytes: u64,
    pub backend_capabilities: u64,
    node_slot_count: u64,
    node_counts: [u64; LAYER_COUNT],
    edge_counts: [u64; LAYER_COUNT],
    label_counts: BTreeMap<LabelId, [u64; LAYER_COUNT]>,
    relationship_counts: BTreeMap<RelationshipTypeId, [u64; LAYER_COUNT]>,
    node_properties: BTreeMap<PropertyId, [PropertyStatistics; LAYER_COUNT]>,
    relationship_properties: BTreeMap<PropertyId, [PropertyStatistics; LAYER_COUNT]>,
    fanout: BTreeMap<RelationshipTypeId, [FanoutStatistics; LAYER_COUNT]>,
    online_indexes: Vec<OptimizerIndexStatistics>,
    temporal: Vec<TemporalOptimizerColumnStatistics>,
}

impl StatisticsSnapshot {
    /// Rebuilds a deterministic bounded summary from the flat canonical graph.
    #[must_use]
    pub fn collect(graph: &GraphStore) -> Self {
        Self::collect_project(graph, None, None)
    }

    /// Rebuilds the full project statistics generation, including derived accelerator lifecycle
    /// and temporal segment summaries when those stores are available.
    #[must_use]
    pub fn collect_project(
        graph: &GraphStore,
        temporal: Option<&TemporalStore>,
        indexes: Option<&IndexCatalog>,
    ) -> Self {
        let mut node_counts = [0_u64; LAYER_COUNT];
        let mut edge_counts = [0_u64; LAYER_COUNT];
        let mut label_counts = BTreeMap::<LabelId, [u64; LAYER_COUNT]>::new();
        let mut relationship_counts = BTreeMap::<RelationshipTypeId, [u64; LAYER_COUNT]>::new();
        let mut fanout = BTreeMap::<RelationshipTypeId, [FanoutAccumulator; LAYER_COUNT]>::new();
        let mut node_accumulators =
            BTreeMap::<PropertyId, [PropertyAccumulator; LAYER_COUNT]>::new();
        let mut relationship_accumulators =
            BTreeMap::<PropertyId, [PropertyAccumulator; LAYER_COUNT]>::new();
        for (property, _) in graph.catalog().properties() {
            node_accumulators.entry(property).or_default();
            relationship_accumulators.entry(property).or_default();
        }
        let mut node_resident_bytes = 0_u64;
        let mut relationship_resident_bytes = 0_u64;

        for node in graph.nodes() {
            let layer = layer_index(node.layer());
            node_counts[layer] = node_counts[layer].saturating_add(1);
            node_resident_bytes = node_resident_bytes
                .saturating_add(17)
                .saturating_add((node.labels().len() as u64).saturating_mul(8));
            for label in node.labels() {
                let counts = label_counts.entry(*label).or_default();
                counts[layer] = counts[layer].saturating_add(1);
            }
            for (property, value) in node.properties() {
                node_resident_bytes = node_resident_bytes
                    .saturating_add(9)
                    .saturating_add(scalar_resident_bytes(&value));
                node_accumulators.entry(property).or_default()[layer].insert(
                    node.id().0,
                    property,
                    node.layer(),
                    &value,
                );
            }
        }

        for relationship in graph.edges() {
            let layer = layer_index(relationship.layer());
            edge_counts[layer] = edge_counts[layer].saturating_add(1);
            relationship_resident_bytes = relationship_resident_bytes.saturating_add(41);
            let counts = relationship_counts
                .entry(relationship.relationship_type())
                .or_default();
            counts[layer] = counts[layer].saturating_add(1);
            fanout.entry(relationship.relationship_type()).or_default()[layer]
                .insert(relationship.source().0, relationship.target().0);
            for (property, value) in relationship.properties() {
                relationship_resident_bytes = relationship_resident_bytes
                    .saturating_add(9)
                    .saturating_add(scalar_resident_bytes(&value));
                relationship_accumulators.entry(property).or_default()[layer].insert(
                    relationship.id().0,
                    property,
                    relationship.layer(),
                    &value,
                );
            }
        }

        let online_indexes = indexes.map_or_else(Vec::new, IndexCatalog::optimizer_statistics);
        let index_generation = indexes.map_or([0_u8; 32], IndexCatalog::optimizer_generation);
        let temporal = temporal.map_or_else(Vec::new, TemporalStore::optimizer_statistics);
        let index_bytes = online_indexes
            .iter()
            .map(|index| index.resident_bytes)
            .fold(0_u64, u64::saturating_add);
        let temporal_bytes = temporal
            .iter()
            .map(|column| column.resident_bytes)
            .fold(0_u64, u64::saturating_add);

        Self {
            graph_revision: graph.revision(),
            schema_generation: graph.catalog().optimizer_generation(),
            index_generation,
            resident_graph_bytes: node_resident_bytes
                .saturating_add(relationship_resident_bytes)
                .saturating_add(index_bytes)
                .saturating_add(temporal_bytes),
            average_node_row_bytes: average_width(node_resident_bytes, graph.node_count()),
            average_relationship_row_bytes: average_width(
                relationship_resident_bytes,
                graph.edge_count(),
            ),
            backend_capabilities: 0,
            node_slot_count: u64::try_from(graph.node_slot_count()).unwrap_or(u64::MAX),
            node_counts,
            edge_counts,
            label_counts,
            relationship_counts,
            node_properties: finalize_properties(node_accumulators, &node_counts),
            relationship_properties: finalize_properties(relationship_accumulators, &edge_counts),
            fanout: fanout
                .into_iter()
                .map(|(kind, layers)| (kind, layers.map(FanoutAccumulator::finish)))
                .collect(),
            online_indexes,
            temporal,
        }
    }

    /// Binds one statistics snapshot to the physical capability set used for this plan. The
    /// clone remains ephemeral and never changes the cached project generation.
    #[must_use]
    pub fn with_backend_capabilities(mut self, capabilities: u64) -> Self {
        self.backend_capabilities = capabilities;
        self
    }

    #[must_use]
    pub fn node_count(&self, layers: LayerMask) -> u64 {
        count_layers(&self.node_counts, layers)
    }

    /// Physical node-row capacity, including tombstoned rows that remain addressable in the
    /// resident flat image until compaction.
    #[must_use]
    pub const fn node_slot_count(&self) -> u64 {
        self.node_slot_count
    }

    #[must_use]
    pub fn edge_count(&self, layers: LayerMask) -> u64 {
        count_layers(&self.edge_counts, layers)
    }

    #[must_use]
    pub fn label_count(&self, label: LabelId, layers: LayerMask) -> u64 {
        self.label_counts
            .get(&label)
            .map_or(0, |counts| count_layers(counts, layers))
    }

    #[must_use]
    pub fn relationship_count(
        &self,
        relationship_type: RelationshipTypeId,
        layers: LayerMask,
    ) -> u64 {
        self.relationship_counts
            .get(&relationship_type)
            .map_or(0, |counts| count_layers(counts, layers))
    }

    #[must_use]
    pub fn node_property(
        &self,
        property: PropertyId,
        layers: LayerMask,
    ) -> Option<PropertyStatistics> {
        merge_property_layers(self.node_properties.get(&property)?, layers)
    }

    #[must_use]
    pub fn relationship_property(
        &self,
        property: PropertyId,
        layers: LayerMask,
    ) -> Option<PropertyStatistics> {
        merge_property_layers(self.relationship_properties.get(&property)?, layers)
    }

    /// Mean directional fanout for a type selection. A known-empty selection remains zero.
    #[must_use]
    pub fn mean_fanout(&self, relationship_types: &[RelationshipTypeId], layers: LayerMask) -> f64 {
        let nodes = self.node_count(layers);
        if nodes == 0 {
            return 0.0;
        }
        let edges = if relationship_types.is_empty() {
            self.edge_count(layers)
        } else {
            relationship_types
                .iter()
                .map(|kind| self.relationship_count(*kind, layers))
                .fold(0_u64, u64::saturating_add)
        };
        edges as f64 / nodes as f64
    }

    /// Direction-sensitive mean over active endpoints. Empty/unknown selections safely fall back
    /// to the all-node mean used by the original cardinality formula.
    #[must_use]
    pub fn directional_fanout(
        &self,
        relationship_types: &[RelationshipTypeId],
        layers: LayerMask,
        outgoing: bool,
    ) -> f64 {
        if relationship_types.is_empty() {
            return self.mean_fanout(relationship_types, layers);
        }
        let mut edges = 0_u64;
        let mut endpoints = 0_u64;
        for kind in relationship_types {
            let Some(stats) = self.fanout.get(kind) else {
                continue;
            };
            for (position, layer) in Layer::ALL.into_iter().enumerate() {
                if !layers.contains_layer(layer) {
                    continue;
                }
                edges = edges.saturating_add(stats[position].edges);
                endpoints = endpoints.saturating_add(if outgoing {
                    stats[position].distinct_sources
                } else {
                    stats[position].distinct_targets
                });
            }
        }
        if endpoints == 0 {
            self.mean_fanout(relationship_types, layers)
        } else {
            edges as f64 / endpoints as f64
        }
    }

    #[must_use]
    pub fn online_indexes(&self) -> &[OptimizerIndexStatistics] {
        &self.online_indexes
    }

    #[must_use]
    pub fn temporal_columns(&self) -> &[TemporalOptimizerColumnStatistics] {
        &self.temporal
    }
}

#[derive(Clone, Debug, Default)]
struct PropertyAccumulator {
    present: u64,
    distinct_hashes: BTreeSet<u64>,
    numeric_sample: BTreeMap<u64, f64>,
    minimum: Option<f64>,
    maximum: Option<f64>,
}

impl PropertyAccumulator {
    fn insert(&mut self, entity: u64, property: PropertyId, layer: Layer, value: &ScalarValue) {
        if matches!(value, ScalarValue::Null) {
            return;
        }
        self.present = self.present.saturating_add(1);
        self.distinct_hashes.insert(scalar_hash(value));
        if self.distinct_hashes.len() > DISTINCT_SKETCH_SIZE
            && let Some(largest) = self.distinct_hashes.last().copied()
        {
            self.distinct_hashes.remove(&largest);
        }
        let Some(number) = numeric_value(value) else {
            return;
        };
        self.minimum = Some(self.minimum.map_or(number, |current| current.min(number)));
        self.maximum = Some(self.maximum.map_or(number, |current| current.max(number)));
        let key = sample_hash(entity, property, layer);
        self.numeric_sample.insert(key, number);
        if self.numeric_sample.len() > NUMERIC_SAMPLE_SIZE
            && let Some(largest) = self.numeric_sample.last_key_value().map(|(key, _)| *key)
        {
            self.numeric_sample.remove(&largest);
        }
    }

    fn finish(self, parent_rows: u64) -> PropertyStatistics {
        let distinct = approximate_distinct(self.present, &self.distinct_hashes);
        let mut numeric_sample = self.numeric_sample.into_values().collect::<Vec<_>>();
        numeric_sample.sort_by(f64::total_cmp);
        PropertyStatistics {
            present: self.present,
            null_or_missing: parent_rows.saturating_sub(self.present),
            distinct,
            minimum: self.minimum,
            maximum: self.maximum,
            histogram: NumericHistogram::from_sorted_sample(&numeric_sample),
        }
    }
}

fn finalize_properties(
    source: BTreeMap<PropertyId, [PropertyAccumulator; LAYER_COUNT]>,
    parent_rows: &[u64; LAYER_COUNT],
) -> BTreeMap<PropertyId, [PropertyStatistics; LAYER_COUNT]> {
    source
        .into_iter()
        .map(|(property, layers)| {
            let [observed, knowledge, workspace] = layers;
            (
                property,
                [
                    observed.finish(parent_rows[0]),
                    knowledge.finish(parent_rows[1]),
                    workspace.finish(parent_rows[2]),
                ],
            )
        })
        .collect()
}

fn merge_property_layers(
    layers: &[PropertyStatistics; LAYER_COUNT],
    mask: LayerMask,
) -> Option<PropertyStatistics> {
    let selected = Layer::ALL
        .into_iter()
        .enumerate()
        .filter(|(_, layer)| mask.contains_layer(*layer))
        .map(|(index, _)| &layers[index])
        .collect::<Vec<_>>();
    if selected.is_empty() {
        return None;
    }
    let mut numeric_sample = Vec::new();
    for stats in &selected {
        let mut previous = 0_u32;
        for (upper, cumulative) in stats
            .histogram
            .upper_bounds
            .iter()
            .zip(&stats.histogram.cumulative)
        {
            let count = cumulative.saturating_sub(previous) as usize;
            numeric_sample.extend(std::iter::repeat_n(*upper, count));
            previous = *cumulative;
        }
    }
    numeric_sample.sort_by(f64::total_cmp);
    if numeric_sample.len() > NUMERIC_SAMPLE_SIZE {
        let length = numeric_sample.len();
        numeric_sample = (0..NUMERIC_SAMPLE_SIZE)
            .map(|position| numeric_sample[position.saturating_mul(length) / NUMERIC_SAMPLE_SIZE])
            .collect();
    }
    Some(PropertyStatistics {
        present: selected
            .iter()
            .map(|stats| stats.present)
            .fold(0_u64, u64::saturating_add),
        null_or_missing: selected
            .iter()
            .map(|stats| stats.null_or_missing)
            .fold(0_u64, u64::saturating_add),
        // Distinct sketches are intentionally not persisted in the public summary. Summing is a
        // conservative upper bound across physical layers and cannot underestimate join fanout.
        distinct: selected
            .iter()
            .map(|stats| stats.distinct)
            .fold(0_u64, u64::saturating_add)
            .max(1),
        minimum: selected
            .iter()
            .filter_map(|stats| stats.minimum)
            .reduce(f64::min),
        maximum: selected
            .iter()
            .filter_map(|stats| stats.maximum)
            .reduce(f64::max),
        histogram: NumericHistogram::from_sorted_sample(&numeric_sample),
    })
}

#[derive(Clone, Debug, Default)]
struct FanoutAccumulator {
    edges: u64,
    sources: BTreeSet<u64>,
    targets: BTreeSet<u64>,
}

impl FanoutAccumulator {
    fn insert(&mut self, source: u64, target: u64) {
        self.edges = self.edges.saturating_add(1);
        insert_bottom_hash(&mut self.sources, endpoint_hash(source));
        insert_bottom_hash(&mut self.targets, endpoint_hash(target));
    }

    fn finish(self) -> FanoutStatistics {
        let distinct_sources = approximate_distinct(self.edges, &self.sources);
        let distinct_targets = approximate_distinct(self.edges, &self.targets);
        FanoutStatistics {
            edges: self.edges,
            distinct_sources,
            distinct_targets,
            mean_outgoing_active: if distinct_sources == 0 {
                0.0
            } else {
                self.edges as f64 / distinct_sources as f64
            },
            mean_incoming_active: if distinct_targets == 0 {
                0.0
            } else {
                self.edges as f64 / distinct_targets as f64
            },
        }
    }
}

fn insert_bottom_hash(sketch: &mut BTreeSet<u64>, hash: u64) {
    sketch.insert(hash);
    if sketch.len() > DISTINCT_SKETCH_SIZE
        && let Some(largest) = sketch.last().copied()
    {
        sketch.remove(&largest);
    }
}

fn endpoint_hash(id: u64) -> u64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"endpoint\0");
    hasher.update(&id.to_le_bytes());
    first_u64(hasher.finalize().as_bytes())
}

fn average_width(bytes: u64, rows: usize) -> u64 {
    if rows == 0 {
        0
    } else {
        bytes.div_ceil(rows as u64)
    }
}

fn scalar_resident_bytes(value: &ScalarValue) -> u64 {
    match value {
        ScalarValue::Null => 0,
        ScalarValue::Boolean(_) => 1,
        ScalarValue::Date(_) => 8,
        ScalarValue::Integer(_) | ScalarValue::Float(_) | ScalarValue::LocalTime(_) => 8,
        ScalarValue::ZonedTime { .. } | ScalarValue::LocalDateTime { .. } => 12,
        ScalarValue::ZonedDateTime { timezone, .. } => 12_u64.saturating_add(timezone.len() as u64),
        ScalarValue::Duration { .. } => 28,
        ScalarValue::String(value) => 4_u64.saturating_add(value.len() as u64),
        ScalarValue::Bytes(value) => 4_u64.saturating_add(value.len() as u64),
        ScalarValue::List(value) => 4_u64.saturating_add(value.as_bytes().len() as u64),
        ScalarValue::Map(value) => 4_u64.saturating_add(value.as_bytes().len() as u64),
    }
}

fn approximate_distinct(present: u64, hashes: &BTreeSet<u64>) -> u64 {
    if present == 0 || hashes.is_empty() {
        return 0;
    }
    if hashes.len() < DISTINCT_SKETCH_SIZE || present <= DISTINCT_SKETCH_SIZE as u64 {
        return hashes.len() as u64;
    }
    let threshold = hashes.last().copied().unwrap_or(u64::MAX) as f64 / u64::MAX as f64;
    if threshold <= f64::EPSILON {
        return present;
    }
    (((DISTINCT_SKETCH_SIZE - 1) as f64 / threshold).round() as u64)
        .clamp(hashes.len() as u64, present)
}

fn count_layers(counts: &[u64; LAYER_COUNT], mask: LayerMask) -> u64 {
    Layer::ALL
        .into_iter()
        .enumerate()
        .filter(|(_, layer)| mask.contains_layer(*layer))
        .map(|(index, _)| counts[index])
        .fold(0_u64, u64::saturating_add)
}

const fn layer_index(layer: Layer) -> usize {
    layer as usize
}

fn numeric_value(value: &ScalarValue) -> Option<f64> {
    match value {
        ScalarValue::Integer(value) => Some(*value as f64),
        ScalarValue::Float(value) if value.is_finite() => Some(value.into_inner()),
        ScalarValue::Date(value) => Some(*value as f64),
        ScalarValue::LocalTime(value) => Some(*value as f64),
        _ => None,
    }
}

fn sample_hash(entity: u64, property: PropertyId, layer: Layer) -> u64 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(&entity.to_le_bytes());
    hasher.update(&property.0.to_le_bytes());
    hasher.update(&[layer as u8]);
    first_u64(hasher.finalize().as_bytes())
}

fn scalar_hash(value: &ScalarValue) -> u64 {
    let mut hasher = blake3::Hasher::new();
    match value {
        ScalarValue::Null => hasher.update(&[0]),
        ScalarValue::Boolean(value) => hasher.update(&[1, u8::from(*value)]),
        ScalarValue::Integer(value) => {
            hasher.update(&[2]);
            hasher.update(&value.to_le_bytes())
        }
        ScalarValue::Float(value) => {
            hasher.update(&[3]);
            hasher.update(&value.into_inner().to_bits().to_le_bytes())
        }
        ScalarValue::String(value) => {
            hasher.update(&[4]);
            hasher.update(value.as_bytes())
        }
        ScalarValue::Bytes(value) => {
            hasher.update(&[5]);
            hasher.update(value)
        }
        ScalarValue::Date(value) => {
            hasher.update(&[6]);
            hasher.update(&value.to_le_bytes())
        }
        ScalarValue::LocalTime(value) => {
            hasher.update(&[7]);
            hasher.update(&value.to_le_bytes())
        }
        ScalarValue::ZonedTime {
            nanos,
            offset_seconds,
        } => {
            hasher.update(&[8]);
            hasher.update(&nanos.to_le_bytes());
            hasher.update(&offset_seconds.to_le_bytes())
        }
        ScalarValue::LocalDateTime { seconds, nanos } => {
            hasher.update(&[9]);
            hasher.update(&seconds.to_le_bytes());
            hasher.update(&nanos.to_le_bytes())
        }
        ScalarValue::ZonedDateTime {
            seconds,
            nanos,
            timezone,
        } => {
            hasher.update(&[10]);
            hasher.update(&seconds.to_le_bytes());
            hasher.update(&nanos.to_le_bytes());
            hasher.update(timezone.as_bytes())
        }
        ScalarValue::Duration {
            months,
            days,
            seconds,
            nanos,
        } => {
            hasher.update(&[11]);
            hasher.update(&months.to_le_bytes());
            hasher.update(&days.to_le_bytes());
            hasher.update(&seconds.to_le_bytes());
            hasher.update(&nanos.to_le_bytes())
        }
        ScalarValue::List(value) => {
            hasher.update(&[12]);
            hasher.update(value.as_bytes())
        }
        ScalarValue::Map(value) => {
            hasher.update(&[13]);
            hasher.update(value.as_bytes())
        }
    };
    first_u64(hasher.finalize().as_bytes())
}

fn first_u64(hash: &[u8; 32]) -> u64 {
    u64::from_le_bytes([
        hash[0], hash[1], hash[2], hash[3], hash[4], hash[5], hash[6], hash[7],
    ])
}

#[cfg(test)]
mod tests {
    use ordered_float::OrderedFloat;

    use crate::{GraphMutation, Layer, NodeId, NodeInput, ScalarValue};

    use super::*;

    #[test]
    fn empty_statistics_preserve_known_zero() {
        let statistics = StatisticsSnapshot::collect(&GraphStore::default());
        assert_eq!(statistics.node_slot_count(), 0);
        assert_eq!(statistics.node_count(LayerMask::AUTHORITY), 0);
        assert_eq!(statistics.edge_count(LayerMask::ALL), 0);
        assert_eq!(statistics.mean_fanout(&[], LayerMask::ALL), 0.0);
    }

    #[test]
    fn date_statistics_charge_the_canonical_i64_width() {
        assert_eq!(scalar_resident_bytes(&ScalarValue::Date(i64::MIN)), 8);
        assert_eq!(scalar_resident_bytes(&ScalarValue::Date(i64::MAX)), 8);
    }

    #[test]
    fn statistics_are_layer_scoped_and_deterministic() -> crate::Result<()> {
        let mut graph = GraphStore::default();
        let label = graph.catalog_mut().intern_label("Metric")?;
        let property = graph.catalog_mut().intern_property("value")?;
        for (id, layer, value) in [
            (1, Layer::Observed, 1.0),
            (2, Layer::Observed, 1.0),
            (3, Layer::Knowledge, 9.0),
        ] {
            graph.apply(GraphMutation::InsertNode(NodeInput {
                id: NodeId(id),
                layer,
                revision: id,
                labels: vec![label],
                properties: vec![(property, ScalarValue::Float(OrderedFloat(value)))],
            }))?;
        }
        let first = StatisticsSnapshot::collect(&graph);
        let second = StatisticsSnapshot::collect(&graph);
        assert_eq!(first, second);
        assert_eq!(first.node_slot_count(), 3);
        assert_eq!(first.label_count(label, LayerMask::OBSERVED), 2);
        assert_eq!(first.label_count(label, LayerMask::KNOWLEDGE), 1);
        let property_stats = first
            .node_property(property, LayerMask::AUTHORITY)
            .ok_or_else(|| crate::Error::internal("inserted property is absent from statistics"))?;
        assert_eq!(property_stats.present, 3);
        assert_eq!(property_stats.minimum, Some(1.0));
        assert_eq!(property_stats.maximum, Some(9.0));
        assert!(property_stats.distinct >= 2);
        Ok(())
    }

    #[test]
    fn equi_depth_histogram_tracks_skew_and_strict_greater_boundaries() -> crate::Result<()> {
        let skewed = [
            std::iter::repeat_n(0.0, 100).collect::<Vec<_>>(),
            std::iter::repeat_n(1_000.0, 100).collect::<Vec<_>>(),
        ]
        .concat();
        let histogram = NumericHistogram::from_sorted_sample(&skewed);
        let below = histogram
            .range_fraction(10.0, false, true)
            .ok_or_else(|| crate::Error::internal("histogram sample is unexpectedly empty"))?;
        assert!((0.45..=0.55).contains(&below));

        let duplicates = NumericHistogram::from_sorted_sample(&[1.0, 1.0, 2.0, 2.0]);
        assert_eq!(duplicates.range_fraction(1.0, false, false), Some(0.5));
        assert_eq!(duplicates.range_fraction(1.0, true, false), Some(1.0));
        Ok(())
    }
}
