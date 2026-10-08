//! Rebuildable equality, range, text, exact-vector, and deterministic IVF-PQ indexes.

use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

use half::{bf16, f16};
use ordered_float::OrderedFloat;
use roaring::RoaringBitmap;
use serde::{Deserialize, Serialize};
use tokio_util::sync::CancellationToken;
use unicode_normalization::UnicodeNormalization;

use crate::{
    Error, ErrorCode, Result, ScalarValue,
    types::{LabelId, PropertyId},
};

use super::{GraphMutation, GraphStore, NodeView, persistent::PagedVec};

#[path = "concurrent_index.rs"]
mod concurrent_index;
use concurrent_index::{ConcurrentMap, EntryMap, Postings, ProfileCell};

/// Total-order key for indexable scalar values.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum IndexKey {
    Boolean(bool),
    Integer(i64),
    Float(OrderedFloat<f64>),
    String(String),
    Bytes(Vec<u8>),
    Date(i64),
    LocalTime(i64),
    ZonedTime(i64, i32),
    LocalDateTime(i64, u32),
    ZonedDateTime(i64, u32, String),
    Duration(i64, i64, i64, i32),
    Composite(Vec<IndexKey>),
}

impl TryFrom<&ScalarValue> for IndexKey {
    type Error = Error;

    fn try_from(value: &ScalarValue) -> Result<Self> {
        match value {
            ScalarValue::Null => Err(Error::new(ErrorCode::QueryType, "NULL is not indexed")),
            ScalarValue::Boolean(value) => Ok(Self::Boolean(*value)),
            ScalarValue::Integer(value) => Ok(Self::Integer(*value)),
            ScalarValue::Float(value) => Ok(Self::Float(*value)),
            ScalarValue::String(value) => Ok(Self::String(value.to_string())),
            ScalarValue::Bytes(value) => Ok(Self::Bytes(value.to_vec())),
            ScalarValue::Date(value) => Ok(Self::Date(*value)),
            ScalarValue::LocalTime(value) => Ok(Self::LocalTime(*value)),
            ScalarValue::ZonedTime {
                nanos,
                offset_seconds,
            } => Ok(Self::ZonedTime(*nanos, *offset_seconds)),
            ScalarValue::LocalDateTime { seconds, nanos } => {
                Ok(Self::LocalDateTime(*seconds, *nanos))
            }
            ScalarValue::ZonedDateTime {
                seconds,
                nanos,
                timezone,
            } => Ok(Self::ZonedDateTime(*seconds, *nanos, timezone.to_string())),
            ScalarValue::Duration {
                months,
                days,
                seconds,
                nanos,
            } => Ok(Self::Duration(*months, *days, *seconds, *nanos)),
            ScalarValue::List(_) | ScalarValue::Map(_) => Err(Error::new(
                ErrorCode::QueryType,
                "document properties require an explicitly declared document-path index",
            )),
        }
    }
}

/// Equality memberships over stable dense row ordinals.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EqualityIndex {
    postings: Postings<IndexKey>,
}
impl EqualityIndex {
    pub fn insert(&self, value: &ScalarValue, row: u32) -> Result<()> {
        if !matches!(value, ScalarValue::Null) {
            self.insert_key(IndexKey::try_from(value)?, row);
        }
        Ok(())
    }
    pub fn insert_key(&self, key: IndexKey, row: u32) {
        self.postings.insert(key, row);
    }
    pub fn remove(&self, value: &ScalarValue, row: u32) -> Result<()> {
        if !matches!(value, ScalarValue::Null) {
            self.remove_key(&IndexKey::try_from(value)?, row);
        }
        Ok(())
    }
    pub fn remove_key(&self, key: &IndexKey, row: u32) {
        self.postings.remove(key, row);
    }
    pub fn get(&self, key: &IndexKey) -> Option<RoaringBitmap> {
        self.postings.rows(key)
    }
    fn get_bounded(&self, key: &IndexKey, limit: usize) -> Vec<u32> {
        self.postings.bounded(key, limit)
    }
    fn materialized(&self) -> BTreeMap<IndexKey, RoaringBitmap> {
        self.postings.materialized()
    }
}
/// Concurrent range memberships; query unions are ephemeral result memory.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RangeIndex {
    postings: Postings<IndexKey>,
}
impl RangeIndex {
    pub fn insert(&self, value: &ScalarValue, row: u32) -> Result<()> {
        if !matches!(value, ScalarValue::Null) {
            self.insert_key(IndexKey::try_from(value)?, row);
        }
        Ok(())
    }
    pub fn insert_key(&self, key: IndexKey, row: u32) {
        self.postings.insert(key, row);
    }
    pub fn remove_key(&self, key: &IndexKey, row: u32) {
        self.postings.remove(key, row);
    }
    fn exact_bounded(&self, key: &IndexKey, limit: usize) -> Vec<u32> {
        self.postings.bounded(key, limit)
    }
    pub fn between(
        &self,
        lower: Option<(&IndexKey, bool)>,
        upper: Option<(&IndexKey, bool)>,
    ) -> RoaringBitmap {
        let mut rows = RoaringBitmap::new();
        for (key, posting) in self.materialized() {
            if lower.is_none_or(|(bound, inclusive)| {
                if inclusive {
                    &key >= bound
                } else {
                    &key > bound
                }
            }) && upper.is_none_or(|(bound, inclusive)| {
                if inclusive {
                    &key <= bound
                } else {
                    &key < bound
                }
            }) {
                rows |= posting;
            }
        }
        rows
    }
    fn materialized(&self) -> BTreeMap<IndexKey, RoaringBitmap> {
        self.postings.materialized()
    }
}
/// Token membership is updated per changed owner, without copying unrelated postings.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TextIndex {
    postings: Postings<String>,
    rows: ConcurrentMap<u32, Arc<BTreeSet<String>>>,
}
impl TextIndex {
    pub fn upsert(&self, row: u32, text: &str) {
        self.remove(row);
        let terms = tokenize(text);
        for term in &terms {
            self.postings.insert(term.clone(), row);
        }
        self.rows.insert(row, Arc::new(terms));
    }
    pub fn remove(&self, row: u32) {
        if let Some(terms) = self.rows.remove(&row) {
            for term in terms.iter() {
                self.postings.remove(term, row);
            }
        }
    }
    pub fn search(&self, query: &str) -> RoaringBitmap {
        let terms = tokenize(query);
        let mut terms = terms.iter();
        let Some(first) = terms.next() else {
            return RoaringBitmap::new();
        };
        let Some(mut result) = self.postings.rows(first) else {
            return RoaringBitmap::new();
        };
        for term in terms {
            let Some(rows) = self.postings.rows(term) else {
                return RoaringBitmap::new();
            };
            result &= rows;
        }
        result
    }
    fn bounded_ranked_search(&self, query: &str, limit: usize) -> Vec<(u32, u16)> {
        let mut scores = BTreeMap::<u32, u16>::new();
        for term in tokenize(query).into_iter().take(32) {
            for row in self.postings.bounded(&term, limit.saturating_mul(4)) {
                *scores.entry(row).or_default() += 1;
            }
        }
        let mut scores: Vec<_> = scores.into_iter().collect();
        scores.sort_by_key(|(row, score)| (std::cmp::Reverse(*score), *row));
        scores.truncate(limit);
        scores
    }
    fn materialized_postings(&self) -> BTreeMap<String, RoaringBitmap> {
        self.postings.materialized()
    }
}

fn tokenize(text: &str) -> BTreeSet<String> {
    text.nfkc()
        .flat_map(char::to_lowercase)
        .collect::<String>()
        .split(|character: char| !character.is_alphanumeric())
        .filter(|term| !term.is_empty())
        .map(ToOwned::to_owned)
        .collect()
}

/// Vector comparison semantics fixed by the project embedding profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Similarity {
    Cosine,
    Dot,
    Euclidean,
}

/// Exact reranked vector result, stable-ID ordered on score ties.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct VectorHit {
    pub entity_id: u64,
    pub score: f32,
}

/// Canonical per-owner quantized vectors. Clones share the same allocations.
#[derive(Clone)]
pub struct VectorIndex {
    dimension: usize,
    similarity: Similarity,
    dtype: EmbeddingDType,
    store: Arc<VectorStore>,
}
#[derive(Default)]
struct VectorStore {
    slots: crate::concurrent::Segments<VectorSlot>,
    free: crate::concurrent::FreeSlots,
    sequence: std::sync::atomic::AtomicU64,
    rows: ConcurrentMap<u64, usize>,
    dirty: ConcurrentMap<usize, u64>,
    graph: arc_swap::ArcSwapOption<GraphStore>,
    live: std::sync::atomic::AtomicUsize,
}
#[derive(Default)]
struct VectorSlot {
    entity_id: std::sync::atomic::AtomicU64,
    deleted_revision: std::sync::atomic::AtomicU64,
    payload: arc_swap::ArcSwapOption<VectorPayload>,
}
#[derive(Serialize, Deserialize)]
struct VectorPayload {
    entity_id: u64,
    stamp: u64,
    #[serde(with = "atomic_vector_revision")]
    revision: std::sync::atomic::AtomicU64,
    coordinates: VectorCoordinates,
}
mod atomic_vector_revision {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::sync::atomic::{AtomicU64, Ordering};
    pub fn serialize<S: Serializer>(value: &AtomicU64, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u64(value.load(Ordering::Acquire))
    }
    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<AtomicU64, D::Error> {
        Ok(AtomicU64::new(u64::deserialize(d)?))
    }
}
#[derive(Serialize, Deserialize)]
enum VectorCoordinates {
    Quantized(Box<[u16]>),
    PropertyOwner {
        property: PropertyId,
        kind: crate::types::EntityKind,
    },
}
struct CoordinateView<'a> {
    quantized: Option<&'a [u16]>,
    numeric: Option<(irongraph_types::DocumentList, f32, EmbeddingDType)>,
}
impl CoordinateView<'_> {
    fn iter(&self) -> impl Iterator<Item = u16> + '_ {
        self.quantized
            .into_iter()
            .flat_map(|values| values.iter().copied())
            .chain(self.numeric.iter().flat_map(|(list, norm, dtype)| {
                list.numeric_values()
                    .into_iter()
                    .flatten()
                    .map(move |value| encode_coordinate(*dtype, value as f32 / *norm))
            }))
    }
}

impl std::fmt::Debug for VectorIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VectorIndex")
            .field("dimension", &self.dimension)
            .field("rows", &self.row_count())
            .finish()
    }
}
#[derive(Serialize, Deserialize)]
struct VectorWire {
    dimension: usize,
    similarity: Similarity,
    dtype: EmbeddingDType,
    rows: Vec<(u64, u64, Option<Arc<VectorPayload>>)>,
    dirty: ConcurrentMap<usize, u64>,
}
impl Serialize for VectorIndex {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::{SerializeSeq, SerializeStruct};
        struct Rows<'a>(&'a VectorIndex);
        impl Serialize for Rows<'_> {
            fn serialize<S: serde::Serializer>(
                &self,
                s: S,
            ) -> std::result::Result<S::Ok, S::Error> {
                let count = self.0.row_count();
                let mut seq = s.serialize_seq(Some(count))?;
                for row in 0..count {
                    if let Some(slot) = self.0.store.slots.get(row) {
                        seq.serialize_element(&(
                            slot.entity_id.load(std::sync::atomic::Ordering::Acquire),
                            self.0.row_version(row).unwrap_or(0),
                            slot.payload.load_full(),
                        ))?;
                    }
                }
                seq.end()
            }
        }
        let mut value = s.serialize_struct("VectorIndex", 5)?;
        value.serialize_field("dimension", &self.dimension)?;
        value.serialize_field("similarity", &self.similarity)?;
        value.serialize_field("dtype", &self.dtype)?;
        value.serialize_field("rows", &Rows(self))?;
        value.serialize_field("dirty", &self.store.dirty)?;
        value.end()
    }
}
impl<'de> Deserialize<'de> for VectorIndex {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let wire = VectorWire::deserialize(d)?;
        let result = Self::new_with_dtype(wire.dimension, wire.similarity, wire.dtype)
            .map_err(serde::de::Error::custom)?;
        for (entity, revision, payload) in wire.rows {
            if result.store.rows.contains_key(&entity) {
                return Err(serde::de::Error::custom("duplicate canonical vector owner"));
            }
            if let Some(payload) = &payload {
                if let VectorCoordinates::Quantized(coordinates) = &payload.coordinates {
                    result
                        .validate_coordinates(coordinates)
                        .map_err(serde::de::Error::custom)?;
                }
            }
            let active = payload.is_some();
            let row = result
                .store
                .slots
                .push(VectorSlot {
                    entity_id: std::sync::atomic::AtomicU64::new(entity),
                    deleted_revision: std::sync::atomic::AtomicU64::new(revision),
                    payload: arc_swap::ArcSwapOption::from(payload),
                })
                .map_err(serde::de::Error::custom)?;
            result
                .store
                .free
                .ensure(row as u32)
                .map_err(serde::de::Error::custom)?;
            result
                .store
                .sequence
                .fetch_max(revision, std::sync::atomic::Ordering::Relaxed);
            if active {
                result.store.rows.insert(entity, row);
                result
                    .store
                    .live
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            } else {
                result.store.free.release(row as u32);
            }
        }
        for (row, revision) in wire.dirty.iter() {
            if row >= result.row_count() {
                return Err(serde::de::Error::custom(
                    "vector delta references a missing owner slot",
                ));
            }
            result.store.dirty.insert(row, revision);
        }
        Ok(result)
    }
}

#[derive(Clone, Debug)]
pub struct SharedVectorBacking {
    pub property: PropertyId,
    pub dimension: usize,
    pub entity_ids: PagedVec<u64>,
    pub values: PagedVec<u16>,
    pub versions: PagedVec<u64>,
    pub active: PagedVec<bool>,
}

/// Canonical FP16/BF16 vector matrix admitted as one complete device-resident column.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct VectorDeviceImage {
    pub property: PropertyId,
    pub dimension: usize,
    pub similarity: Similarity,
    pub dtype: EmbeddingDType,
    pub entity_ids: Vec<u64>,
    pub values: Vec<u16>,
    pub versions: Vec<u64>,
    pub active: Vec<u8>,
}

impl VectorDeviceImage {
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.entity_ids
            .len()
            .saturating_mul(size_of::<u64>())
            .saturating_add(self.values.len().saturating_mul(size_of::<u16>()))
            .saturating_add(self.versions.len().saturating_mul(size_of::<u64>()))
            .saturating_add(self.active.len())
    }
}

/// Flattened immutable IVF-PQ pages. Canonical vectors remain in `VectorDeviceImage`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnnDeviceImage {
    pub config: IvfPqConfig,
    pub dimension: usize,
    pub similarity: Similarity,
    pub coarse_values: Vec<f32>,
    pub coarse_count: usize,
    pub codebook_subspace_offsets: Vec<u32>,
    pub codebook_vector_offsets: Vec<u32>,
    pub codebook_values: Vec<f32>,
    pub rows: Vec<u32>,
    pub codes: Vec<u8>,
    pub list_offsets: Vec<u32>,
    pub list_positions: Vec<u32>,
    pub built_versions: Vec<u64>,
    /// Stable identity of the complete validated derived page generation.
    pub build_generation: [u8; 32],
}

impl AnnDeviceImage {
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.coarse_values
            .len()
            .saturating_mul(size_of::<f32>())
            .saturating_add(
                self.codebook_subspace_offsets
                    .len()
                    .saturating_mul(size_of::<u32>()),
            )
            .saturating_add(
                self.codebook_vector_offsets
                    .len()
                    .saturating_mul(size_of::<u32>()),
            )
            .saturating_add(self.codebook_values.len().saturating_mul(size_of::<f32>()))
            .saturating_add(self.rows.len().saturating_mul(size_of::<u32>()))
            .saturating_add(self.codes.len())
            .saturating_add(self.list_offsets.len().saturating_mul(size_of::<u32>()))
            .saturating_add(self.list_positions.len().saturating_mul(size_of::<u32>()))
            .saturating_add(self.built_versions.len().saturating_mul(size_of::<u64>()))
    }
}

/// Byte-comparable canonical scalar keys plus flattened postings.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PostingDeviceImage {
    pub key_offsets: Vec<u32>,
    pub key_bytes: Vec<u8>,
    pub posting_offsets: Vec<u32>,
    pub rows: Vec<u32>,
}

impl PostingDeviceImage {
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.key_offsets
            .len()
            .saturating_mul(size_of::<u32>())
            .saturating_add(self.key_bytes.len())
            .saturating_add(self.posting_offsets.len().saturating_mul(size_of::<u32>()))
            .saturating_add(self.rows.len().saturating_mul(size_of::<u32>()))
    }
}

/// Normalized term dictionary plus flattened text postings.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TextDeviceImage {
    pub term_offsets: Vec<u32>,
    pub term_bytes: Vec<u8>,
    pub posting_offsets: Vec<u32>,
    pub rows: Vec<u32>,
}

impl TextDeviceImage {
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.term_offsets
            .len()
            .saturating_mul(size_of::<u32>())
            .saturating_add(self.term_bytes.len())
            .saturating_add(self.posting_offsets.len().saturating_mul(size_of::<u32>()))
            .saturating_add(self.rows.len().saturating_mul(size_of::<u32>()))
    }
}

/// One ONLINE rebuildable index in backend-neutral device columns.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum DerivedIndexDeviceImage {
    Equality {
        name: String,
        postings: PostingDeviceImage,
    },
    Range {
        name: String,
        postings: PostingDeviceImage,
    },
    Text {
        name: String,
        postings: TextDeviceImage,
    },
    Vector {
        name: String,
        property: PropertyId,
        approximate: Option<AnnDeviceImage>,
    },
}

impl DerivedIndexDeviceImage {
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        match self {
            Self::Equality { postings, .. } | Self::Range { postings, .. } => {
                postings.resident_bytes()
            }
            Self::Text { postings, .. } => postings.resident_bytes(),
            Self::Vector { approximate, .. } => approximate
                .as_ref()
                .map_or(0, AnnDeviceImage::resident_bytes),
        }
    }
}

/// Complete canonical-vector and ONLINE-derived-index image for one project.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct IndexDeviceImage {
    pub profile: Option<EmbeddingProfile>,
    pub vectors: Vec<VectorDeviceImage>,
    pub indexes: Vec<DerivedIndexDeviceImage>,
}

impl IndexDeviceImage {
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.profile
            .as_ref()
            .map_or(0, |_| size_of::<EmbeddingProfile>())
            .saturating_add(
                self.vectors
                    .iter()
                    .map(VectorDeviceImage::resident_bytes)
                    .chain(
                        self.indexes
                            .iter()
                            .map(DerivedIndexDeviceImage::resident_bytes),
                    )
                    .fold(0_usize, usize::saturating_add),
            )
    }
}

impl VectorIndex {
    pub fn new(dimension: usize, similarity: Similarity) -> Result<Self> {
        Self::new_with_dtype(dimension, similarity, EmbeddingDType::F16)
    }
    pub fn new_with_dtype(
        dimension: usize,
        similarity: Similarity,
        dtype: EmbeddingDType,
    ) -> Result<Self> {
        if dimension == 0 {
            return Err(Error::new(
                ErrorCode::EmbeddingProfileMismatch,
                "vector dimension must be positive",
            ));
        }
        Ok(Self {
            dimension,
            similarity,
            dtype,
            store: Arc::new(VectorStore::default()),
        })
    }
    fn validate_coordinates(&self, coordinates: &[u16]) -> Result<()> {
        if coordinates.len() != self.dimension
            || coordinates
                .iter()
                .any(|bits| !decode_coordinate(self.dtype, *bits).is_finite())
        {
            return Err(Error::new(
                ErrorCode::EmbeddingProfileMismatch,
                "quantized vector is incompatible with the embedding profile",
            ));
        }
        Ok(())
    }
    pub fn upsert(&self, entity_id: u64, vector: &[f32], revision: u64) -> Result<()> {
        self.upsert_quantized(entity_id, &self.quantize(vector)?, revision)
    }
    pub fn upsert_quantized(
        &self,
        entity_id: u64,
        coordinates: &[u16],
        revision: u64,
    ) -> Result<()> {
        self.validate_coordinates(coordinates)?;
        self.publish_payload(
            entity_id,
            VectorCoordinates::Quantized(coordinates.into()),
            revision,
        )
    }
    fn publish_payload(
        &self,
        entity_id: u64,
        coordinates: VectorCoordinates,
        revision: u64,
    ) -> Result<()> {
        let payload = Arc::new(VectorPayload {
            entity_id,
            stamp: self
                .store
                .sequence
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                + 1,
            revision: std::sync::atomic::AtomicU64::new(revision),
            coordinates,
        });
        let stamp = payload.stamp;
        if let Some(row) = self.store.rows.get(&entity_id) {
            let slot = self
                .store
                .slots
                .get(row)
                .ok_or_else(|| Error::internal("vector owner slot is missing"))?;
            if slot.payload.swap(Some(payload)).is_none() {
                self.store
                    .live
                    .fetch_add(1, std::sync::atomic::Ordering::Release);
            }
            self.store.dirty.insert(row, stamp);
        } else {
            let row = if let Some(row) = self.store.free.pop() {
                let row = row as usize;
                let slot = self
                    .store
                    .slots
                    .get(row)
                    .ok_or_else(|| Error::internal("free vector slot is missing"))?;
                slot.entity_id
                    .store(entity_id, std::sync::atomic::Ordering::Release);
                slot.payload.store(Some(payload));
                row
            } else {
                let row = self.store.slots.push(VectorSlot {
                    entity_id: std::sync::atomic::AtomicU64::new(entity_id),
                    deleted_revision: std::sync::atomic::AtomicU64::new(stamp),
                    payload: arc_swap::ArcSwapOption::from(Some(payload)),
                })?;
                let row_id = u32::try_from(row).map_err(|_| {
                    Error::new(ErrorCode::ResultBudgetExceeded, "vector rows exceed u32")
                })?;
                self.store.free.ensure(row_id)?;
                row
            };
            self.store.rows.insert(entity_id, row);
            self.store.dirty.insert(row, stamp);
            self.store
                .live
                .fetch_add(1, std::sync::atomic::Ordering::Release);
        }
        Ok(())
    }
    /// Binds the same canonical graph allocation; no graph or vector payload is copied.
    pub fn bind_canonical_graph(&self, graph: &GraphStore) {
        self.store.graph.store(Some(Arc::new(graph.clone())));
    }
    pub fn upsert_property_owner(
        &self,
        graph: &GraphStore,
        kind: crate::types::EntityKind,
        property: PropertyId,
        entity_id: u64,
        revision: u64,
    ) -> Result<()> {
        let value = match kind {
            crate::types::EntityKind::Node => graph
                .node(crate::NodeId(entity_id))
                .and_then(|node| node.property(property)),
            crate::types::EntityKind::Relationship => graph
                .edge(crate::EdgeId(entity_id))
                .and_then(|edge| edge.property(property)),
        };
        let Some(ScalarValue::List(list)) = value else {
            return Err(Error::new(
                ErrorCode::EmbeddingProfileMismatch,
                "vector property owner has no numeric list",
            ));
        };
        self.numeric_norm(&list)?;
        if self.store.graph.load().is_none() {
            self.bind_canonical_graph(graph);
        }
        self.publish_payload(
            entity_id,
            VectorCoordinates::PropertyOwner { property, kind },
            revision,
        )
    }
    fn numeric_norm(&self, list: &irongraph_types::DocumentList) -> Result<f32> {
        let values = list.numeric_values()?;
        if values.len() != self.dimension {
            return Err(Error::new(
                ErrorCode::EmbeddingProfileMismatch,
                "canonical vector property has wrong dimension",
            ));
        }
        let mut squared = 0.0_f32;
        for value in values {
            let value = value as f32;
            if !value.is_finite() {
                return Err(Error::new(
                    ErrorCode::EmbeddingProfileMismatch,
                    "canonical vector property coordinate is not finite",
                ));
            }
            squared += value * value;
        }
        let norm = if self.similarity == Similarity::Cosine {
            squared.sqrt()
        } else {
            1.0
        };
        if !norm.is_finite() || norm <= f32::EPSILON {
            return Err(Error::new(
                ErrorCode::EmbeddingProfileMismatch,
                "canonical vector property has invalid norm",
            ));
        }
        Ok(norm)
    }
    fn payload_coordinates<'a>(
        &self,
        entity_id: u64,
        payload: &'a VectorPayload,
    ) -> Result<Option<CoordinateView<'a>>> {
        match &payload.coordinates {
            VectorCoordinates::Quantized(coordinates) => Ok(Some(CoordinateView {
                quantized: Some(coordinates),
                numeric: None,
            })),
            VectorCoordinates::PropertyOwner { property, kind } => {
                let graph = self.store.graph.load();
                let Some(graph) = graph.as_ref() else {
                    return Ok(None);
                };
                let value = match kind {
                    crate::types::EntityKind::Node => graph
                        .node(crate::NodeId(entity_id))
                        .and_then(|node| node.property(*property)),
                    crate::types::EntityKind::Relationship => graph
                        .edge(crate::EdgeId(entity_id))
                        .and_then(|edge| edge.property(*property)),
                };
                let Some(ScalarValue::List(list)) = value else {
                    return Ok(None);
                };
                let Ok(norm) = self.numeric_norm(&list) else {
                    return Ok(None);
                };
                Ok(Some(CoordinateView {
                    quantized: None,
                    numeric: Some((list, norm, self.dtype)),
                }))
            }
        }
    }
    pub fn remove(&self, entity_id: u64, _revision: u64) {
        if let Some(row) = self.store.rows.remove(&entity_id) {
            if let Some(slot) = self.store.slots.get(row) {
                let stamp = self
                    .store
                    .sequence
                    .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                    + 1;
                slot.deleted_revision
                    .store(stamp, std::sync::atomic::Ordering::Release);
                if slot.payload.swap(None).is_some() {
                    self.store
                        .live
                        .fetch_sub(1, std::sync::atomic::Ordering::Release);
                }
                self.store.dirty.insert(row, stamp);
                self.store.free.release(row as u32);
            }
        }
    }
    fn retain_entities(&self, retained: &BTreeSet<u64>) -> Result<BTreeMap<usize, u32>> {
        let mut remap = BTreeMap::new();
        for row in 0..self.row_count() {
            if let Some(entity) = self.row_entity(row) {
                if retained.contains(&entity) && self.row_active(row) {
                    remap.insert(
                        row,
                        u32::try_from(row).map_err(|_| {
                            Error::new(ErrorCode::ResultBudgetExceeded, "vector row exceeds u32")
                        })?,
                    );
                } else {
                    self.remove(entity, self.row_version(row).unwrap_or(0));
                }
            }
        }
        Ok(remap)
    }
    pub fn exact_search(&self, query: &[f32], limit: usize) -> Result<Vec<VectorHit>> {
        self.exact_search_where(query, limit, |_| true)
    }
    pub fn exact_search_where(
        &self,
        query: &[f32],
        limit: usize,
        visible: impl Fn(u64) -> bool,
    ) -> Result<Vec<VectorHit>> {
        let query = self.prepare(query)?;
        if limit == 0 {
            return Ok(Vec::new());
        }
        let mut hits = Vec::with_capacity(limit.min(self.len()));
        for row in 0..self.row_count() {
            let Some(slot) = self.store.slots.get(row) else {
                continue;
            };
            let payload = slot.payload.load();
            let Some(payload) = payload.as_ref() else {
                continue;
            };
            if !visible(payload.entity_id) {
                continue;
            }
            let Some(coordinates) = self.payload_coordinates(payload.entity_id, payload)? else {
                continue;
            };
            let score = match self.similarity {
                Similarity::Euclidean => query
                    .iter()
                    .zip(coordinates.iter())
                    .map(|(left, bits)| {
                        let delta = left - decode_coordinate(self.dtype, bits);
                        delta * delta
                    })
                    .sum::<f32>()
                    .sqrt(),
                Similarity::Dot | Similarity::Cosine => query
                    .iter()
                    .zip(coordinates.iter())
                    .map(|(left, bits)| left * decode_coordinate(self.dtype, bits))
                    .sum(),
            };
            let hit = VectorHit {
                entity_id: payload.entity_id,
                score,
            };
            let at = hits.partition_point(|current: &VectorHit| {
                let order = if self.similarity == Similarity::Euclidean {
                    current.score.total_cmp(&hit.score)
                } else {
                    hit.score.total_cmp(&current.score)
                };
                order
                    .then_with(|| current.entity_id.cmp(&hit.entity_id))
                    .is_lt()
            });
            if at < limit {
                hits.insert(at, hit);
                if hits.len() > limit {
                    hits.pop();
                }
            }
        }
        sort_hits(&mut hits, self.similarity);
        hits.truncate(limit);
        Ok(hits)
    }
    fn acknowledge_ann(&self, ann: &IvfPqIndex) {
        for (row, revision) in self.store.dirty.iter() {
            let built = ann
                .rows
                .binary_search(&(row as u32))
                .ok()
                .and_then(|at| ann.built_versions.get(at))
                .copied();
            if built == Some(revision)
                || (!self.row_active(row) && self.row_version(row) == Some(revision))
            {
                self.store.dirty.remove_matching(&row, &revision);
            }
        }
    }
    pub fn vector_for(&self, entity_id: u64) -> Option<Vec<f32>> {
        let slot = self.store.slots.get(self.store.rows.get(&entity_id)?)?;
        let payload = slot.payload.load();
        let payload = payload.as_ref()?;
        (payload.entity_id == entity_id)
            .then(|| self.decode_payload(payload))
            .flatten()
    }
    /// Current owner revision without reading or decoding vector coordinates.
    pub fn row_revision(&self, entity_id: u64) -> Option<u64> {
        let slot = self.store.slots.get(self.store.rows.get(&entity_id)?)?;
        let payload = slot.payload.load();
        let payload = payload.as_ref()?;
        (payload.entity_id == entity_id)
            .then(|| payload.revision.load(std::sync::atomic::Ordering::Acquire))
    }
    /// Advance freshness after a metadata-only owner mutation, without replacing coordinates.
    /// A vector older than the previous owner revision remains stale.
    pub fn advance_row_revision(
        &self,
        entity_id: u64,
        expected_old_revision: u64,
        new_revision: u64,
    ) -> bool {
        use std::sync::atomic::Ordering;
        let Some(row) = self.store.rows.get(&entity_id) else {
            return false;
        };
        let Some(slot) = self.store.slots.get(row) else {
            return false;
        };
        let guard = slot.payload.load();
        let Some(payload) = guard.as_ref() else {
            return false;
        };
        if payload.entity_id != entity_id || new_revision < expected_old_revision {
            return false;
        }
        let previous = payload.revision.load(Ordering::Acquire);
        if previous < expected_old_revision || previous > new_revision {
            return false;
        }
        if payload
            .revision
            .compare_exchange(previous, new_revision, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            return false;
        }
        let current = slot.payload.load();
        current
            .as_ref()
            .is_some_and(|current| Arc::ptr_eq(current, payload))
            && self.store.rows.get(&entity_id) == Some(row)
    }
    fn prepare(&self, vector: &[f32]) -> Result<Vec<f32>> {
        if vector.len() != self.dimension || vector.iter().any(|value| !value.is_finite()) {
            return Err(Error::new(
                ErrorCode::EmbeddingProfileMismatch,
                "vector dimension or coordinate is incompatible with the embedding profile",
            ));
        }
        let mut result = vector.to_vec();
        if self.similarity == Similarity::Cosine {
            let norm = result.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm <= f32::EPSILON {
                return Err(Error::new(
                    ErrorCode::EmbeddingProfileMismatch,
                    "cosine vector has zero norm",
                ));
            }
            for value in &mut result {
                *value /= norm;
            }
        }
        Ok(result)
    }
    pub fn quantize(&self, vector: &[f32]) -> Result<Vec<u16>> {
        Ok(self
            .prepare(vector)?
            .into_iter()
            .map(|v| encode_coordinate(self.dtype, v))
            .collect())
    }
    fn vector(&self, row: usize) -> Option<Vec<f32>> {
        let slot = self.store.slots.get(row)?;
        let payload = slot.payload.load();
        self.decode_payload(payload.as_ref()?)
    }
    fn decode_payload(&self, payload: &VectorPayload) -> Option<Vec<f32>> {
        let coordinates = self
            .payload_coordinates(payload.entity_id, payload)
            .ok()??;
        Some(
            coordinates
                .iter()
                .map(|v| decode_coordinate(self.dtype, v))
                .collect(),
        )
    }
    fn row_active(&self, row: usize) -> bool {
        self.store
            .slots
            .get(row)
            .is_some_and(|slot| slot.payload.load().is_some())
    }
    fn row_entity(&self, row: usize) -> Option<u64> {
        self.store
            .slots
            .get(row)
            .map(|slot| slot.entity_id.load(std::sync::atomic::Ordering::Acquire))
    }
    fn row_version(&self, row: usize) -> Option<u64> {
        self.store.slots.get(row).map(|slot| {
            slot.payload.load().as_ref().map_or_else(
                || {
                    slot.deleted_revision
                        .load(std::sync::atomic::Ordering::Acquire)
                },
                |payload| payload.stamp,
            )
        })
    }
    pub const fn dimension(&self) -> usize {
        self.dimension
    }
    pub const fn similarity(&self) -> Similarity {
        self.similarity
    }
    pub const fn dtype(&self) -> EmbeddingDType {
        self.dtype
    }
    pub fn len(&self) -> usize {
        self.store.live.load(std::sync::atomic::Ordering::Acquire)
    }
    pub fn row_count(&self) -> usize {
        self.store.slots.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn device_image(&self, property: PropertyId) -> Result<VectorDeviceImage> {
        let mut entity_ids = Vec::new();
        let mut values = Vec::new();
        let mut versions = Vec::new();
        let mut active = Vec::new();
        for row in 0..self.row_count() {
            let slot = self
                .store
                .slots
                .get(row)
                .ok_or_else(|| Error::internal("missing vector slot"))?;
            let payload = slot.payload.load();
            entity_ids.push(slot.entity_id.load(std::sync::atomic::Ordering::Acquire));
            versions.push(self.row_version(row).unwrap_or(0));
            active.push(u8::from(payload.is_some()));
            if let Some(coordinates) = payload.as_ref().and_then(|payload| {
                self.payload_coordinates(payload.entity_id, payload)
                    .ok()
                    .flatten()
            }) {
                values.extend(coordinates.iter());
            } else {
                values.resize(values.len() + self.dimension, 0);
            }
        }
        Ok(VectorDeviceImage {
            property,
            dimension: self.dimension,
            similarity: self.similarity,
            dtype: self.dtype,
            entity_ids,
            values,
            versions,
            active,
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IvfPqConfig {
    pub size_class_version: u16,
    pub coarse_centroids: usize,
    pub subquantizers: usize,
    pub bits_per_code: u8,
    pub probes: usize,
    pub candidate_budget: usize,
    pub iterations: usize,
    pub seed: u64,
}

/// Version of the deterministic row-count-to-IVF-PQ parameter table.
pub const IVF_PQ_SIZE_CLASS_VERSION: u16 = 1;
/// Reserved derived vector columns. These are not properties on canonical graph records.
pub const SEMANTIC_NODE_PROPERTY: PropertyId = PropertyId(u64::MAX - 1);
pub const SEMANTIC_RELATIONSHIP_PROPERTY: PropertyId = PropertyId(u64::MAX - 2);
pub const SEMANTIC_NODE_INDEX: &str = "semantic_nodes";
pub const SEMANTIC_RELATIONSHIP_INDEX: &str = "semantic_relationships";
pub const SEMANTIC_INDEX: &str = "graph_semantic";
const MAX_IVF_PQ_TRAINING_ROWS: usize = 131_072;
/// Fixed source-row batch used by device rebuild executors.
pub const IVF_PQ_BUILD_BATCH_ROWS: usize = 16_384;
/// Hard upper bound for one device-resident assignment distance tile.
pub const IVF_PQ_ASSIGNMENT_TILE_BYTES: usize = 256 * 1024 * 1024;

/// Deterministic memory contract checked before an IVF-PQ rebuild starts. All values are bytes or
/// row counts and are independent of backend pointer widths and local device count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IvfPqBuildPlan {
    pub active_rows: u64,
    pub dimension: usize,
    pub training_rows: usize,
    pub batch_rows: usize,
    pub assignment_tile_bytes: usize,
    pub derived_bytes: u64,
    pub peak_scratch_bytes: u64,
}

/// Backend assignment primitive used by deterministic IVF-PQ training and streamed encoding.
/// Implementations return the lowest centroid index on an exact-distance tie.
pub trait IvfPqBuildKernel {
    fn assign(
        &mut self,
        vectors: &[f32],
        row_count: usize,
        dimension: usize,
        centroids: &[f32],
        centroid_count: usize,
    ) -> Result<Vec<u32>>;
}

struct CpuIvfPqBuildKernel;

impl IvfPqBuildKernel for CpuIvfPqBuildKernel {
    fn assign(
        &mut self,
        vectors: &[f32],
        row_count: usize,
        dimension: usize,
        centroids: &[f32],
        centroid_count: usize,
    ) -> Result<Vec<u32>> {
        validate_assignment_shape(vectors, row_count, dimension, centroids, centroid_count)?;
        vectors
            .chunks_exact(dimension)
            .map(|vector| {
                centroids
                    .chunks_exact(dimension)
                    .enumerate()
                    .min_by(|(left_index, left), (right_index, right)| {
                        total_f32(l2_squared(vector, left), l2_squared(vector, right))
                            .then_with(|| left_index.cmp(right_index))
                    })
                    .map(|(index, _)| index)
                    .ok_or_else(|| Error::new(ErrorCode::IndexUnavailable, "centroid set is empty"))
                    .and_then(|index| {
                        u32::try_from(index).map_err(|_| {
                            Error::new(ErrorCode::IndexUnavailable, "centroid index exceeds u32")
                        })
                    })
            })
            .collect()
    }
}

impl IvfPqBuildPlan {
    pub fn for_shape(active_rows: u64, dimension: usize, config: IvfPqConfig) -> Result<Self> {
        if active_rows == 0 || active_rows > u64::from(u32::MAX) || dimension == 0 {
            return Err(Error::new(
                ErrorCode::IndexUnavailable,
                "IVF-PQ build shape is empty or exceeds dense-row addressing",
            ));
        }
        let training_rows = usize::try_from(active_rows.min(MAX_IVF_PQ_TRAINING_ROWS as u64))
            .map_err(|_| Error::new(ErrorCode::IndexUnavailable, "training rows exceed usize"))?;
        if config.size_class_version != IVF_PQ_SIZE_CLASS_VERSION
            || config.coarse_centroids == 0
            || config.coarse_centroids > training_rows
            || config.subquantizers == 0
            || config.subquantizers > dimension
            || config.bits_per_code == 0
            || config.bits_per_code > 8
            || config.probes == 0
            || config.probes > config.coarse_centroids
            || config.candidate_budget == 0
            || config.iterations == 0
        {
            return Err(Error::new(
                ErrorCode::IndexUnavailable,
                "IVF-PQ build configuration does not fit its deterministic training sample",
            ));
        }
        let codebook_centroids = (1_u64 << config.bits_per_code).min(training_rows as u64);
        let assignment_tile_bytes = (training_rows as u64)
            .checked_mul((config.coarse_centroids as u64).max(codebook_centroids))
            .and_then(|distances| distances.checked_mul(4))
            .map(|bytes| bytes.min(IVF_PQ_ASSIGNMENT_TILE_BYTES as u64).max(4))
            .and_then(|bytes| usize::try_from(bytes).ok())
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::IndexUnavailable,
                    "IVF assignment tile bytes overflow",
                )
            })?;
        let row_bytes = 4_u64
            .checked_add(config.subquantizers as u64)
            .and_then(|bytes| bytes.checked_add(4))
            .and_then(|bytes| bytes.checked_add(8))
            .ok_or_else(|| Error::new(ErrorCode::IndexUnavailable, "IVF row bytes overflow"))?;
        let page_bytes = active_rows.checked_mul(row_bytes).ok_or_else(|| {
            Error::new(
                ErrorCode::IndexUnavailable,
                "IVF derived byte estimate overflow",
            )
        })?;
        let coarse_bytes = (config.coarse_centroids as u64)
            .checked_mul(dimension as u64)
            .and_then(|values| values.checked_mul(4))
            .ok_or_else(|| Error::new(ErrorCode::IndexUnavailable, "coarse bytes overflow"))?;
        let codebook_bytes = codebook_centroids
            .checked_mul(dimension as u64)
            .and_then(|values| values.checked_mul(4))
            .ok_or_else(|| Error::new(ErrorCode::IndexUnavailable, "codebook bytes overflow"))?;
        let offsets_bytes = (config.coarse_centroids as u64)
            .checked_add(1)
            .and_then(|values| values.checked_mul(4))
            .ok_or_else(|| Error::new(ErrorCode::IndexUnavailable, "offset bytes overflow"))?;
        let codebook_offsets_bytes = codebook_centroids
            .checked_mul(config.subquantizers as u64)
            .and_then(|values| values.checked_add(1))
            .and_then(|values| values.checked_mul(4))
            .and_then(|bytes| {
                (config.subquantizers as u64)
                    .checked_add(1)
                    .and_then(|values| values.checked_mul(4))
                    .and_then(|subspace_bytes| bytes.checked_add(subspace_bytes))
            })
            .ok_or_else(|| Error::new(ErrorCode::IndexUnavailable, "codebook offsets overflow"))?;
        let derived_bytes = page_bytes
            .checked_add(coarse_bytes)
            .and_then(|bytes| bytes.checked_add(codebook_bytes))
            .and_then(|bytes| bytes.checked_add(offsets_bytes))
            .and_then(|bytes| bytes.checked_add(codebook_offsets_bytes))
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::IndexUnavailable,
                    "IVF derived byte estimate overflow",
                )
            })?;

        let training_matrix = (training_rows as u64)
            .checked_mul(dimension as u64)
            .and_then(|values| values.checked_mul(4))
            .ok_or_else(|| Error::new(ErrorCode::IndexUnavailable, "training bytes overflow"))?;
        let coarse_accumulators = (config.coarse_centroids as u64)
            .checked_mul(dimension as u64)
            .and_then(|values| values.checked_mul(8))
            .ok_or_else(|| Error::new(ErrorCode::IndexUnavailable, "training sums overflow"))?;
        let widest_subspace = dimension.div_ceil(config.subquantizers);
        let subspace_training = (training_rows as u64)
            .checked_mul(widest_subspace as u64)
            .and_then(|values| values.checked_mul(4))
            .ok_or_else(|| Error::new(ErrorCode::IndexUnavailable, "PQ training bytes overflow"))?;
        let batch_rows = usize::try_from(active_rows.min(IVF_PQ_BUILD_BATCH_ROWS as u64))
            .map_err(|_| Error::new(ErrorCode::IndexUnavailable, "batch rows exceed usize"))?;
        let batch_vectors = (batch_rows as u64)
            .checked_mul(dimension as u64)
            .and_then(|values| values.checked_mul(4))
            .ok_or_else(|| Error::new(ErrorCode::IndexUnavailable, "build batch bytes overflow"))?;
        let batch_residuals = (batch_rows as u64)
            .checked_mul(widest_subspace as u64)
            .and_then(|values| values.checked_mul(4))
            .ok_or_else(|| Error::new(ErrorCode::IndexUnavailable, "build batch bytes overflow"))?;
        let batch_metadata = (batch_rows as u64)
            .checked_mul(config.subquantizers as u64 + 16)
            .ok_or_else(|| Error::new(ErrorCode::IndexUnavailable, "build batch bytes overflow"))?;
        let batch_bytes = batch_vectors
            .checked_add(batch_residuals)
            .and_then(|bytes| bytes.checked_add(batch_metadata))
            .ok_or_else(|| Error::new(ErrorCode::IndexUnavailable, "build batch bytes overflow"))?;
        let peak_scratch_bytes = training_matrix
            // Unified-memory backends can temporarily hold both the deterministic host sample
            // and its device assignment tensor. Discrete backends use the same conservative gate.
            .checked_add(training_matrix)
            .and_then(|bytes| bytes.checked_add(coarse_bytes))
            .and_then(|bytes| bytes.checked_add(coarse_bytes))
            .and_then(|bytes| bytes.checked_add(codebook_bytes))
            .and_then(|bytes| bytes.checked_add(coarse_accumulators))
            .and_then(|bytes| bytes.checked_add(subspace_training))
            .and_then(|bytes| bytes.checked_add(batch_bytes))
            .and_then(|bytes| bytes.checked_add(assignment_tile_bytes as u64))
            .ok_or_else(|| {
                Error::new(ErrorCode::IndexUnavailable, "IVF scratch estimate overflow")
            })?;
        Ok(Self {
            active_rows,
            dimension,
            training_rows,
            batch_rows,
            assignment_tile_bytes,
            derived_bytes,
            peak_scratch_bytes,
        })
    }
}

impl Default for IvfPqConfig {
    fn default() -> Self {
        Self {
            size_class_version: IVF_PQ_SIZE_CLASS_VERSION,
            coarse_centroids: 64,
            subquantizers: 8,
            bits_per_code: 8,
            probes: 8,
            candidate_budget: 256,
            iterations: 16,
            seed: 0x4947_4956_4650_5101,
        }
    }
}

/// Immutable IVF-PQ accelerator. Canonical vectors remain in `VectorIndex` for reranking.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IvfPqIndex {
    config: IvfPqConfig,
    dimension: usize,
    similarity: Similarity,
    coarse: Vec<Vec<f32>>,
    codebooks: Vec<Vec<Vec<f32>>>,
    rows: Vec<u32>,
    codes: Vec<u8>,
    list_offsets: Vec<u32>,
    list_positions: Vec<u32>,
    built_versions: Vec<u64>,
    #[serde(default)]
    build_generation: [u8; 32],
}

impl IvfPqIndex {
    #[must_use]
    pub const fn config(&self) -> IvfPqConfig {
        self.config
    }

    #[must_use]
    pub const fn build_generation(&self) -> [u8; 32] {
        self.build_generation
    }

    fn remap_rows(&self, rows: &BTreeMap<usize, u32>) -> Result<Option<Self>> {
        let mut remapped_rows = Vec::new();
        let mut remapped_codes = Vec::new();
        let mut remapped_versions = Vec::new();
        let mut positions = BTreeMap::<u32, u32>::new();
        for (position, row) in self.rows.iter().copied().enumerate() {
            let Some(row) = rows.get(&(row as usize)).copied() else {
                continue;
            };
            let next_position = u32::try_from(remapped_rows.len()).map_err(|_| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "filtered ANN position exceeds device index width",
                )
            })?;
            let position_u32 = u32::try_from(position).map_err(|_| {
                Error::new(
                    ErrorCode::CorruptStorage,
                    "ANN position exceeds its device index width",
                )
            })?;
            let code_start = position
                .checked_mul(self.config.subquantizers)
                .ok_or_else(|| Error::new(ErrorCode::CorruptStorage, "ANN code offset overflow"))?;
            let code_end = code_start
                .checked_add(self.config.subquantizers)
                .ok_or_else(|| Error::new(ErrorCode::CorruptStorage, "ANN code offset overflow"))?;
            remapped_codes.extend_from_slice(self.codes.get(code_start..code_end).ok_or_else(
                || Error::new(ErrorCode::CorruptStorage, "ANN code row is truncated"),
            )?);
            remapped_rows.push(row);
            remapped_versions.push(*self.built_versions.get(position).ok_or_else(|| {
                Error::new(ErrorCode::CorruptStorage, "ANN row has no built revision")
            })?);
            positions.insert(position_u32, next_position);
        }
        if remapped_rows.is_empty() {
            return Ok(None);
        }
        let mut list_offsets = Vec::with_capacity(self.coarse.len().saturating_add(1));
        let mut list_positions = Vec::with_capacity(remapped_rows.len());
        list_offsets.push(0);
        for list in 0..self.coarse.len() {
            let bounds = self.list_offsets.get(list..=list + 1).ok_or_else(|| {
                Error::new(ErrorCode::CorruptStorage, "ANN list offsets are truncated")
            })?;
            for position in self
                .list_positions
                .get(bounds[0] as usize..bounds[1] as usize)
                .ok_or_else(|| {
                    Error::new(ErrorCode::CorruptStorage, "ANN list posting is truncated")
                })?
            {
                if let Some(position) = positions.get(position) {
                    list_positions.push(*position);
                }
            }
            list_offsets.push(u32::try_from(list_positions.len()).map_err(|_| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "filtered ANN postings exceed device index width",
                )
            })?);
        }
        let mut remapped = Self {
            config: self.config,
            dimension: self.dimension,
            similarity: self.similarity,
            coarse: self.coarse.clone(),
            codebooks: self.codebooks.clone(),
            rows: remapped_rows,
            codes: remapped_codes,
            list_offsets,
            list_positions,
            built_versions: remapped_versions,
            build_generation: [0_u8; 32],
        };
        remapped.refresh_generation()?;
        Ok(Some(remapped))
    }

    fn device_image(&self) -> Result<AnnDeviceImage> {
        let mut coarse_values = Vec::new();
        for centroid in &self.coarse {
            if centroid.len() != self.dimension {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "IVF centroid has an invalid dimension",
                ));
            }
            coarse_values.extend_from_slice(centroid);
        }

        let mut codebook_subspace_offsets = Vec::with_capacity(self.codebooks.len() + 1);
        let mut codebook_vector_offsets = vec![0_u32];
        let mut codebook_values = Vec::new();
        codebook_subspace_offsets.push(0);
        let mut centroid_count = 0_usize;
        for (subspace, book) in self.codebooks.iter().enumerate() {
            let (start, end) = subspace_bounds(self.dimension, self.config.subquantizers, subspace);
            let expected_dimension = end.saturating_sub(start);
            for centroid in book {
                if centroid.len() != expected_dimension {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        "PQ centroid has an invalid subspace dimension",
                    ));
                }
                codebook_values.extend_from_slice(centroid);
                codebook_vector_offsets.push(u32_len(codebook_values.len(), "PQ codebook")?);
                centroid_count = centroid_count.checked_add(1).ok_or_else(|| {
                    Error::new(
                        ErrorCode::ResultBudgetExceeded,
                        "PQ centroid count overflow",
                    )
                })?;
            }
            codebook_subspace_offsets.push(u32_len(centroid_count, "PQ centroids")?);
        }

        let rows = self.rows.clone();
        if self.codes.len()
            != rows
                .len()
                .checked_mul(self.config.subquantizers)
                .ok_or_else(|| {
                    Error::new(ErrorCode::ResultBudgetExceeded, "PQ code shape overflow")
                })?
            || self.list_offsets.len() != self.coarse.len().saturating_add(1)
            || self.list_offsets.last().copied().unwrap_or(0) as usize != self.list_positions.len()
            || self
                .list_positions
                .iter()
                .any(|position| *position as usize >= rows.len())
            || self.built_versions.len() != rows.len()
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "IVF-PQ pages are structurally inconsistent",
            ));
        }
        Ok(AnnDeviceImage {
            config: self.config,
            dimension: self.dimension,
            similarity: self.similarity,
            coarse_values,
            coarse_count: self.coarse.len(),
            codebook_subspace_offsets,
            codebook_vector_offsets,
            codebook_values,
            rows,
            codes: self.codes.clone(),
            list_offsets: self.list_offsets.clone(),
            list_positions: self.list_positions.clone(),
            built_versions: self.built_versions.clone(),
            build_generation: self.build_generation,
        })
    }

    pub fn build(source: &VectorIndex, config: IvfPqConfig) -> Result<Self> {
        Self::build_with_kernel(source, config, &mut CpuIvfPqBuildKernel)
    }

    pub fn build_cancellable(
        source: &VectorIndex,
        config: IvfPqConfig,
        cancellation: &CancellationToken,
    ) -> Result<Self> {
        Self::build_with_kernel_cancellable(source, config, &mut CpuIvfPqBuildKernel, cancellation)
    }

    pub fn build_with_kernel(
        source: &VectorIndex,
        config: IvfPqConfig,
        kernel: &mut dyn IvfPqBuildKernel,
    ) -> Result<Self> {
        Self::build_with_kernel_cancellable(source, config, kernel, &CancellationToken::new())
    }

    pub fn build_with_kernel_cancellable(
        source: &VectorIndex,
        config: IvfPqConfig,
        kernel: &mut dyn IvfPqBuildKernel,
        cancellation: &CancellationToken,
    ) -> Result<Self> {
        check_ivf_build_cancelled(cancellation)?;
        validate_ivf_config(source, config)?;
        let active_rows = source.len();
        if active_rows == 0 {
            return Err(Error::new(
                ErrorCode::IndexUnavailable,
                "cannot build IVF-PQ on an empty vector matrix",
            ));
        }
        let plan = IvfPqBuildPlan::for_shape(active_rows as u64, source.dimension, config)?;
        let coarse_count = config.coarse_centroids.min(active_rows);
        let training_count = plan.training_rows;
        if training_count < coarse_count {
            return Err(Error::new(
                ErrorCode::IndexUnavailable,
                "IVF-PQ training sample is smaller than its coarse-centroid count",
            ));
        }
        let training = deterministic_training_sample(source, active_rows, training_count)?;
        check_ivf_build_cancelled(cancellation)?;
        if training.len() != training_count {
            return Err(Error::internal(
                "vector matrix is structurally inconsistent",
            ));
        }
        let coarse = kmeans(
            &training,
            coarse_count,
            config.iterations,
            config.seed,
            kernel,
            cancellation,
        )?;
        let training_assignments = assign_nested(kernel, &training, &coarse)?;
        let mut codebooks = Vec::with_capacity(config.subquantizers);
        let code_count = (1_usize << config.bits_per_code).min(training.len());
        for subspace in 0..config.subquantizers {
            check_ivf_build_cancelled(cancellation)?;
            let (start, end) = subspace_bounds(source.dimension, config.subquantizers, subspace);
            // Only one subspace of residual training data exists at a time. Peak training scratch
            // is therefore independent of the full source-row count and bounded by the frozen
            // sample size plus one subspace.
            let sub_training = training
                .iter()
                .zip(&training_assignments)
                .map(|(vector, assignment)| {
                    vector[start..end]
                        .iter()
                        .zip(&coarse[*assignment as usize][start..end])
                        .map(|(value, centroid)| value - centroid)
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            codebooks.push(kmeans(
                &sub_training,
                code_count,
                config.iterations,
                config.seed ^ (subspace as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15),
                kernel,
                cancellation,
            )?);
        }
        let encoded = stream_encode_rows(
            source,
            &coarse,
            &codebooks,
            config,
            plan,
            kernel,
            cancellation,
        )?;
        let mut index = Self {
            config,
            dimension: source.dimension,
            similarity: source.similarity,
            coarse,
            codebooks,
            rows: encoded.rows,
            codes: encoded.codes,
            list_offsets: encoded.list_offsets,
            list_positions: encoded.list_positions,
            built_versions: encoded.built_versions,
            build_generation: [0_u8; 32],
        };
        check_ivf_build_cancelled(cancellation)?;
        index.refresh_generation()?;
        Ok(index)
    }

    fn refresh_generation(&mut self) -> Result<()> {
        self.build_generation = [0_u8; 32];
        let encoded = postcard::to_stdvec(self).map_err(|error| {
            Error::new(
                ErrorCode::IndexUnavailable,
                format!("IVF-PQ generation encoding failed: {error}"),
            )
        })?;
        self.build_generation = *blake3::hash(&encoded).as_bytes();
        Ok(())
    }

    pub fn search(
        &self,
        source: &VectorIndex,
        query: &[f32],
        limit: usize,
    ) -> Result<Vec<VectorHit>> {
        if source.dimension != self.dimension || source.similarity != self.similarity {
            return Err(Error::new(
                ErrorCode::EmbeddingProfileMismatch,
                "ANN profile does not match vector matrix",
            ));
        }
        let query = source.prepare(query)?;
        let mut coarse_order: Vec<_> = self
            .coarse
            .iter()
            .enumerate()
            .map(|(index, centroid)| (l2_squared(&query, centroid), index))
            .collect();
        coarse_order
            .sort_by(|left, right| total_f32(left.0, right.0).then_with(|| left.1.cmp(&right.1)));
        let mut approximate = Vec::new();
        for (probe, (_, list_index)) in coarse_order.into_iter().enumerate() {
            if probe >= self.config.probes && approximate.len() >= limit {
                break;
            }
            let Some(bounds) = self.list_offsets.get(list_index..=list_index + 1) else {
                continue;
            };
            let residual = subtract(&query, &self.coarse[list_index])?;
            for position in &self.list_positions[bounds[0] as usize..bounds[1] as usize] {
                if self
                    .rows
                    .get(*position as usize)
                    .is_none_or(|row| !source.row_active(*row as usize))
                {
                    continue;
                }
                let mut distance = 0.0_f32;
                for subspace in 0..self.config.subquantizers {
                    let code_offset = (*position as usize)
                        .checked_mul(self.config.subquantizers)
                        .and_then(|offset| offset.checked_add(subspace))
                        .ok_or_else(|| Error::internal("PQ code offset overflow"))?;
                    let Some(code) = self.codes.get(code_offset).copied() else {
                        continue;
                    };
                    let Some(centroid) = self
                        .codebooks
                        .get(subspace)
                        .and_then(|book| book.get(code as usize))
                    else {
                        continue;
                    };
                    let (start, end) =
                        subspace_bounds(self.dimension, self.config.subquantizers, subspace);
                    distance += l2_squared(&residual[start..end], centroid);
                }
                approximate.push((distance, *position as usize));
            }
        }
        approximate
            .sort_by(|left, right| total_f32(left.0, right.0).then_with(|| left.1.cmp(&right.1)));
        approximate.truncate(self.config.candidate_budget.max(limit));
        let mut candidate_rows = BTreeSet::new();
        for (_, position) in approximate {
            if let Some(row) = self.rows.get(position).copied() {
                candidate_rows.insert(row as usize);
            }
        }
        for (row, revision) in source.store.dirty.iter() {
            let built_revision = self
                .rows
                .binary_search(&(row as u32))
                .ok()
                .and_then(|position| self.built_versions.get(position))
                .copied();
            if built_revision != Some(revision) {
                candidate_rows.insert(row);
            }
        }
        let mut hits = Vec::new();
        for row in candidate_rows {
            if !source.row_active(row) {
                continue;
            }
            let (Some(entity_id), Some(vector)) = (source.row_entity(row), source.vector(row))
            else {
                continue;
            };
            hits.push(VectorHit {
                entity_id,
                score: public_score(self.similarity, &query, &vector),
            });
        }
        sort_hits(&mut hits, self.similarity);
        hits.truncate(limit);
        Ok(hits)
    }
}

struct StreamEncodedIvfPq {
    rows: Vec<u32>,
    codes: Vec<u8>,
    list_offsets: Vec<u32>,
    list_positions: Vec<u32>,
    built_versions: Vec<u64>,
}

fn deterministic_training_sample(
    source: &VectorIndex,
    active_rows: usize,
    training_count: usize,
) -> Result<Vec<Vec<f32>>> {
    if training_count == 0 || training_count > active_rows {
        return Err(Error::new(
            ErrorCode::IndexUnavailable,
            "IVF-PQ training sample shape is invalid",
        ));
    }
    let targets = (0..training_count)
        .map(|sample| sample.saturating_mul(active_rows) / training_count)
        .collect::<Vec<_>>();
    let mut training = Vec::with_capacity(training_count);
    let mut active_position = 0_usize;
    let mut target_position = 0_usize;
    for row in 0..source.row_count() {
        let active = source.row_active(row);
        if !active {
            continue;
        }
        while targets.get(target_position).copied() == Some(active_position) {
            training.push(source.vector(row).ok_or_else(|| {
                Error::new(
                    ErrorCode::CorruptStorage,
                    "canonical vector matrix is structurally inconsistent",
                )
            })?);
            target_position += 1;
        }
        active_position += 1;
    }
    if target_position != training_count {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "canonical vector activity and training sample differ",
        ));
    }
    Ok(training)
}

fn stream_encode_rows(
    source: &VectorIndex,
    coarse: &[Vec<f32>],
    codebooks: &[Vec<Vec<f32>>],
    config: IvfPqConfig,
    plan: IvfPqBuildPlan,
    kernel: &mut dyn IvfPqBuildKernel,
    cancellation: &CancellationToken,
) -> Result<StreamEncodedIvfPq> {
    let active_rows = source.len();
    let code_capacity = active_rows
        .checked_mul(config.subquantizers)
        .ok_or_else(|| Error::new(ErrorCode::IndexUnavailable, "PQ code capacity overflow"))?;
    let mut rows = Vec::with_capacity(active_rows);
    let mut codes = Vec::with_capacity(code_capacity);
    let mut built_versions = Vec::with_capacity(active_rows);
    let mut list_counts = vec![0_u32; coarse.len()];
    let coarse_flat = flatten_centroids(coarse);
    let codebook_flats = codebooks
        .iter()
        .map(|book| flatten_centroids(book))
        .collect::<Vec<_>>();
    let mut batch_rows = Vec::with_capacity(plan.batch_rows);
    for row in 0..source.row_count() {
        let active = source.row_active(row);
        if row.is_multiple_of(plan.batch_rows.max(1)) {
            check_ivf_build_cancelled(cancellation)?;
        }
        if !active {
            continue;
        }
        batch_rows.push(row);
        if batch_rows.len() == plan.batch_rows {
            encode_ivf_pq_batch(
                source,
                &batch_rows,
                coarse,
                codebooks,
                &coarse_flat,
                &codebook_flats,
                config,
                kernel,
                &mut rows,
                &mut codes,
                &mut built_versions,
                &mut list_counts,
                cancellation,
            )?;
            batch_rows.clear();
        }
    }
    if !batch_rows.is_empty() {
        encode_ivf_pq_batch(
            source,
            &batch_rows,
            coarse,
            codebooks,
            &coarse_flat,
            &codebook_flats,
            config,
            kernel,
            &mut rows,
            &mut codes,
            &mut built_versions,
            &mut list_counts,
            cancellation,
        )?;
    }

    let mut list_offsets = Vec::with_capacity(coarse.len().saturating_add(1));
    list_offsets.push(0);
    for count in list_counts {
        let next = list_offsets
            .last()
            .copied()
            .unwrap_or(0_u32)
            .checked_add(count)
            .ok_or_else(|| Error::new(ErrorCode::IndexUnavailable, "IVF postings exceed u32"))?;
        list_offsets.push(next);
    }
    let posting_count = list_offsets.last().copied().unwrap_or(0) as usize;
    if posting_count != rows.len() {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "IVF list counts differ from encoded rows",
        ));
    }
    let mut cursors = list_offsets[..coarse.len()].to_vec();
    let mut list_positions = vec![0_u32; posting_count];
    for (batch_number, batch) in rows.chunks(plan.batch_rows).enumerate() {
        check_ivf_build_cancelled(cancellation)?;
        let vectors = decode_vector_rows(source, batch.iter().map(|row| *row as usize))?;
        let assignments = assign_matrix(
            kernel,
            &vectors,
            batch.len(),
            source.dimension,
            &coarse_flat,
            coarse.len(),
        )?;
        let base_position = batch_number.saturating_mul(plan.batch_rows);
        for (batch_position, assignment) in assignments.into_iter().enumerate() {
            let cursor = cursors.get_mut(assignment as usize).ok_or_else(|| {
                Error::new(ErrorCode::CorruptStorage, "IVF assignment is out of bounds")
            })?;
            let output = list_positions.get_mut(*cursor as usize).ok_or_else(|| {
                Error::new(
                    ErrorCode::CorruptStorage,
                    "IVF posting cursor is out of bounds",
                )
            })?;
            *output = u32_len(base_position.saturating_add(batch_position), "IVF position")?;
            *cursor = cursor
                .checked_add(1)
                .ok_or_else(|| Error::new(ErrorCode::IndexUnavailable, "IVF cursor exceeds u32"))?;
        }
    }
    if cursors
        .iter()
        .zip(&list_offsets[1..])
        .any(|(cursor, end)| cursor != end)
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "IVF list publication did not fill every posting exactly once",
        ));
    }
    Ok(StreamEncodedIvfPq {
        rows,
        codes,
        list_offsets,
        list_positions,
        built_versions,
    })
}

#[allow(clippy::too_many_arguments)]
fn encode_ivf_pq_batch(
    source: &VectorIndex,
    batch_rows: &[usize],
    coarse: &[Vec<f32>],
    codebooks: &[Vec<Vec<f32>>],
    coarse_flat: &[f32],
    codebook_flats: &[Vec<f32>],
    config: IvfPqConfig,
    kernel: &mut dyn IvfPqBuildKernel,
    rows: &mut Vec<u32>,
    codes: &mut Vec<u8>,
    built_versions: &mut Vec<u64>,
    list_counts: &mut [u32],
    cancellation: &CancellationToken,
) -> Result<()> {
    check_ivf_build_cancelled(cancellation)?;
    let vectors = decode_vector_rows(source, batch_rows.iter().copied())?;
    let assignments = assign_matrix(
        kernel,
        &vectors,
        batch_rows.len(),
        source.dimension,
        coarse_flat,
        coarse.len(),
    )?;
    let mut batch_codes = vec![0_u8; batch_rows.len().saturating_mul(config.subquantizers)];
    for subspace in 0..config.subquantizers {
        check_ivf_build_cancelled(cancellation)?;
        let (start, end) = subspace_bounds(source.dimension, config.subquantizers, subspace);
        let width = end.saturating_sub(start);
        let mut residuals = Vec::with_capacity(batch_rows.len().saturating_mul(width));
        for (vector, assignment) in vectors
            .chunks_exact(source.dimension)
            .zip(assignments.iter().copied())
        {
            residuals.extend(
                vector[start..end]
                    .iter()
                    .zip(&coarse[assignment as usize][start..end])
                    .map(|(value, centroid)| value - centroid),
            );
        }
        let subspace_codes = assign_matrix(
            kernel,
            &residuals,
            batch_rows.len(),
            width,
            &codebook_flats[subspace],
            codebooks[subspace].len(),
        )?;
        for (row, code) in subspace_codes.into_iter().enumerate() {
            batch_codes[row * config.subquantizers + subspace] = u8::try_from(code)
                .map_err(|_| Error::new(ErrorCode::IndexUnavailable, "PQ code exceeds one byte"))?;
        }
    }
    for (row, assignment) in batch_rows.iter().copied().zip(&assignments) {
        list_counts[*assignment as usize] = list_counts[*assignment as usize]
            .checked_add(1)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::IndexUnavailable,
                    "IVF list row count exceeds u32",
                )
            })?;
        rows.push(u32_len(row, "IVF source row")?);
        built_versions.push(source.row_version(row).ok_or_else(|| {
            Error::new(
                ErrorCode::CorruptStorage,
                "canonical vector revisions are structurally inconsistent",
            )
        })?);
    }
    codes.extend(batch_codes);
    Ok(())
}

fn decode_vector_rows(
    source: &VectorIndex,
    rows: impl ExactSizeIterator<Item = usize>,
) -> Result<Vec<f32>> {
    let mut vectors = Vec::with_capacity(rows.len().saturating_mul(source.dimension));
    for row in rows {
        vectors.extend(source.vector(row).ok_or_else(|| {
            Error::new(
                ErrorCode::CorruptStorage,
                "canonical vector matrix is structurally inconsistent",
            )
        })?);
    }
    Ok(vectors)
}

fn assign_flat(
    kernel: &mut dyn IvfPqBuildKernel,
    vectors: &[f32],
    row_count: usize,
    dimension: usize,
    centroids: &[Vec<f32>],
) -> Result<Vec<u32>> {
    let flat = flatten_centroids(centroids);
    assign_matrix(
        kernel,
        vectors,
        row_count,
        dimension,
        &flat,
        centroids.len(),
    )
}

fn assign_matrix(
    kernel: &mut dyn IvfPqBuildKernel,
    vectors: &[f32],
    row_count: usize,
    dimension: usize,
    centroids: &[f32],
    centroid_count: usize,
) -> Result<Vec<u32>> {
    validate_assignment_shape(vectors, row_count, dimension, centroids, centroid_count)?;
    let assignments = kernel.assign(vectors, row_count, dimension, centroids, centroid_count)?;
    if assignments.len() != row_count
        || assignments
            .iter()
            .any(|assignment| *assignment as usize >= centroid_count)
    {
        return Err(Error::new(
            ErrorCode::IndexUnavailable,
            "IVF-PQ build kernel returned an invalid assignment vector",
        ));
    }
    Ok(assignments)
}

fn flatten_centroids(centroids: &[Vec<f32>]) -> Vec<f32> {
    centroids
        .iter()
        .flat_map(|centroid| centroid.iter().copied())
        .collect()
}

fn assign_nested(
    kernel: &mut dyn IvfPqBuildKernel,
    vectors: &[Vec<f32>],
    centroids: &[Vec<f32>],
) -> Result<Vec<u32>> {
    let dimension = vectors.first().map_or(0, Vec::len);
    let flat = vectors
        .iter()
        .flat_map(|vector| vector.iter().copied())
        .collect::<Vec<_>>();
    assign_flat(kernel, &flat, vectors.len(), dimension, centroids)
}

fn validate_assignment_shape(
    vectors: &[f32],
    row_count: usize,
    dimension: usize,
    centroids: &[f32],
    centroid_count: usize,
) -> Result<()> {
    let vector_values = row_count.checked_mul(dimension).ok_or_else(|| {
        Error::new(
            ErrorCode::IndexUnavailable,
            "assignment matrix shape overflow",
        )
    })?;
    let centroid_values = centroid_count.checked_mul(dimension).ok_or_else(|| {
        Error::new(
            ErrorCode::IndexUnavailable,
            "centroid matrix shape overflow",
        )
    })?;
    if row_count == 0
        || dimension == 0
        || centroid_count == 0
        || vectors.len() != vector_values
        || centroids.len() != centroid_values
        || vectors
            .iter()
            .chain(centroids)
            .any(|value| !value.is_finite())
    {
        return Err(Error::new(
            ErrorCode::IndexUnavailable,
            "IVF-PQ assignment matrices are invalid",
        ));
    }
    Ok(())
}

/// Durable scalar/vector index family declared through Cypher.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GraphIndexKind {
    Equality,
    Range,
    Text,
    Vector,
}

/// Minimal durable ordinary-index definition. Physical postings and ANN pages are derived.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GraphIndexDefinition {
    pub name: String,
    pub kind: GraphIndexKind,
    pub label: LabelId,
    pub properties: Vec<PropertyId>,
    /// Enforced node-property uniqueness constraint backed by this equality posting. This is
    /// canonical schema authority; ordinary equality indexes always keep this false.
    #[serde(default)]
    pub unique: bool,
}

/// Local lifecycle of one rebuildable physical index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DerivedIndexState {
    Populating,
    Online,
    Failed,
    Dropping,
}

/// SHOW INDEXES row produced from the durable definition and local derived state.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexStatus {
    pub name: String,
    pub kind: GraphIndexKind,
    pub state: DerivedIndexState,
    pub diagnostic: Option<String>,
}

/// Bounded planner-facing summary of one ONLINE derived index. This is ephemeral cost data, not
/// part of the durable index definition or checkpoint format.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OptimizerIndexStatistics {
    pub name: String,
    pub kind: GraphIndexKind,
    pub label: LabelId,
    pub properties: Vec<PropertyId>,
    pub distinct_keys: u64,
    pub total_postings: u64,
    pub largest_posting: u64,
    pub vector_rows: u64,
    pub ann_candidate_budget: u64,
    pub resident_bytes: u64,
}

/// Exact local posting estimate selected by stable cardinality/name ordering.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ScalarIndexCandidate {
    pub name: String,
    pub estimated_rows: u64,
}

/// Coordinate representation selected once by the project embedding profile.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmbeddingDType {
    F16,
    Bf16,
}

fn encode_coordinate(dtype: EmbeddingDType, value: f32) -> u16 {
    match dtype {
        EmbeddingDType::F16 => f16::from_f32(value).to_bits(),
        EmbeddingDType::Bf16 => bf16::from_f32(value).to_bits(),
    }
}

fn decode_coordinate(dtype: EmbeddingDType, bits: u16) -> f32 {
    match dtype {
        EmbeddingDType::F16 => f16::from_bits(bits).to_f32(),
        EmbeddingDType::Bf16 => bf16::from_bits(bits).to_f32(),
    }
}

/// One immutable local embedding profile saved once per project.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingProfile {
    pub model_hash: [u8; 32],
    pub tokenizer_hash: [u8; 32],
    pub dimension: u32,
    pub dtype: EmbeddingDType,
    pub normalized: bool,
    pub similarity: Similarity,
    pub profile_hash: [u8; 32],
}

impl EmbeddingProfile {
    pub fn new(
        model_hash: [u8; 32],
        tokenizer_hash: [u8; 32],
        dimension: u32,
        dtype: EmbeddingDType,
        normalized: bool,
        similarity: Similarity,
    ) -> Result<Self> {
        if dimension == 0 {
            return Err(Error::new(
                ErrorCode::EmbeddingProfileMismatch,
                "embedding dimension must be positive",
            ));
        }
        let encoded = postcard::to_stdvec(&(
            model_hash,
            tokenizer_hash,
            dimension,
            dtype,
            normalized,
            similarity,
        ))
        .map_err(|error| Error::internal(format!("embedding profile encoding failed: {error}")))?;
        Ok(Self {
            model_hash,
            tokenizer_hash,
            dimension,
            dtype,
            normalized,
            similarity,
            profile_hash: *blake3::hash(&encoded).as_bytes(),
        })
    }

    pub fn validate(&self) -> Result<()> {
        let expected = Self::new(
            self.model_hash,
            self.tokenizer_hash,
            self.dimension,
            self.dtype,
            self.normalized,
            self.similarity,
        )?;
        if expected.profile_hash != self.profile_hash {
            return Err(Error::new(
                ErrorCode::EmbeddingProfileMismatch,
                "embedding profile hash is invalid",
            ));
        }
        Ok(())
    }

    pub fn quantize(&self, vector: &[f32]) -> Result<Vec<u16>> {
        self.validate()?;
        VectorIndex::new_with_dtype(self.dimension as usize, self.similarity, self.dtype)?
            .quantize(vector)
    }
}

/// Durable mapping from one scalar text source to one canonical vector column.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EmbeddingIndexDefinition {
    pub name: String,
    pub label: LabelId,
    pub source_property: PropertyId,
    pub target_property: PropertyId,
    pub model: String,
}

/// Vector coordinates resolved once by the sequencer and published byte-for-byte.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ResolvedVectorMutation {
    Upsert {
        property: PropertyId,
        entity_id: u64,
        coordinates: Vec<u16>,
        revision: u64,
    },
    Remove {
        property: PropertyId,
        entity_id: u64,
        revision: u64,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
enum RuntimeIndex {
    Equality(EqualityIndex),
    Range(RangeIndex),
    Text(TextIndex),
    Vector {
        property: PropertyId,
        approximate: Option<Arc<IvfPqIndex>>,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct IndexEntry {
    definition: GraphIndexDefinition,
    state: DerivedIndexState,
    diagnostic: Option<String>,
    runtime: Option<Arc<RuntimeIndex>>,
}

/// Project-scoped durable definitions plus rebuildable physical index state and canonical vectors.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct IndexCatalog {
    entries: EntryMap,
    embeddings: ConcurrentMap<String, EmbeddingIndexDefinition>,
    profile: ProfileCell,
    /// Vectors are canonical embedding rows; ANN structures in `entries` are derived.
    vector_columns: ConcurrentMap<PropertyId, Arc<VectorIndex>>,
    /// Allocated source slots covered by the last successful automatic-index build. Deleted
    /// owners keep their slots, so live ANN row counts cannot measure subsequent source growth.
    #[serde(default)]
    semantic_built_slots: ConcurrentMap<PropertyId, usize>,
    #[serde(skip)]
    optimizer_generation_cache: std::sync::Arc<std::sync::atomic::AtomicU64>,
}

impl IndexCatalog {
    /// Simulates only keys affected by this batch. Canonical rows and postings remain untouched.
    pub fn validate_graph_mutations(
        &self,
        graph: &GraphStore,
        mutations: &[GraphMutation],
    ) -> Result<()> {
        let constraints: Vec<_> = self.constraint_definitions().collect();
        if constraints.is_empty() {
            return Ok(());
        }
        struct Keys {
            active: bool,
            labels: BTreeSet<LabelId>,
            properties: BTreeMap<PropertyId, ScalarValue>,
        }
        impl Keys {
            fn key(&self, definition: &GraphIndexDefinition) -> Result<Option<IndexKey>> {
                if !self.active || !self.labels.contains(&definition.label) {
                    return Ok(None);
                }
                let Some(value) = self.properties.get(&definition.properties[0]) else {
                    return Ok(None);
                };
                if matches!(value, ScalarValue::Null) {
                    return Ok(None);
                }
                IndexKey::try_from(value).map(Some)
            }
        }
        let properties: BTreeSet<_> = constraints
            .iter()
            .flat_map(|definition| definition.properties.iter().copied())
            .collect();
        let mut pending = BTreeMap::<crate::NodeId, Keys>::new();
        let mut claims = BTreeMap::<(String, IndexKey), crate::NodeId>::new();
        for mutation in mutations {
            let owner = match mutation {
                GraphMutation::InsertNode(input) => input.id,
                GraphMutation::SetNodeProperty { node, .. }
                | GraphMutation::AddNodeLabels { node, .. }
                | GraphMutation::RemoveNodeLabels { node, .. }
                | GraphMutation::DeleteNode { node, .. } => *node,
                _ => continue,
            };
            pending.entry(owner).or_insert_with(|| {
                if let Some(node) = graph.node(owner) {
                    Keys {
                        active: true,
                        labels: node.labels().iter().copied().collect(),
                        properties: properties
                            .iter()
                            .filter_map(|property| {
                                node.property(*property).map(|value| (*property, value))
                            })
                            .collect(),
                    }
                } else {
                    Keys {
                        active: false,
                        labels: BTreeSet::new(),
                        properties: BTreeMap::new(),
                    }
                }
            });
            let state = pending
                .get_mut(&owner)
                .ok_or_else(|| Error::internal("validation owner disappeared"))?;
            for definition in &constraints {
                if let Some(key) = state.key(definition)? {
                    claims.remove(&(definition.name.clone(), key));
                }
            }
            match mutation {
                GraphMutation::InsertNode(input) => {
                    state.active = true;
                    state.labels = input.labels.iter().copied().collect();
                    state.properties = input
                        .properties
                        .iter()
                        .filter(|(property, _)| properties.contains(property))
                        .cloned()
                        .collect();
                }
                GraphMutation::SetNodeProperty {
                    property, value, ..
                } => {
                    if properties.contains(property) {
                        state.properties.insert(*property, value.clone());
                    }
                }
                GraphMutation::AddNodeLabels { labels, .. } => {
                    state.labels.extend(labels.iter().copied())
                }
                GraphMutation::RemoveNodeLabels { labels, .. } => {
                    state.labels.retain(|label| !labels.contains(label))
                }
                GraphMutation::DeleteNode { .. } => state.active = false,
                _ => {}
            }
            for definition in &constraints {
                let key = pending
                    .get(&owner)
                    .ok_or_else(|| Error::internal("validation owner disappeared"))?
                    .key(definition)?;
                let Some(key) = key else {
                    continue;
                };
                if claims
                    .get(&(definition.name.clone(), key.clone()))
                    .is_some_and(|other| *other != owner)
                {
                    return Err(Error::new(
                        ErrorCode::TransactionConflict,
                        "node property uniqueness constraint was violated",
                    ));
                }
                let entry = self
                    .entries
                    .get(&definition.name)
                    .ok_or_else(|| Error::internal("constraint disappeared during validation"))?;
                let Some(RuntimeIndex::Equality(index)) = entry.runtime.as_deref() else {
                    return Err(Error::new(
                        ErrorCode::IndexUnavailable,
                        "constraint has no equality runtime",
                    ));
                };
                if let Some(rows) = index.get(&key) {
                    for row in rows {
                        let Some(node) = graph.node_dense(row) else {
                            continue;
                        };
                        if node.id() == owner {
                            continue;
                        }
                        let conflict = if let Some(changed) = pending.get(&node.id()) {
                            changed.key(definition)?.as_ref() == Some(&key)
                        } else {
                            node.labels().contains(&definition.label)
                                && composite_key(node, &definition.properties)?.as_ref()
                                    == Some(&key)
                        };
                        if conflict {
                            return Err(Error::new(
                                ErrorCode::TransactionConflict,
                                "node property uniqueness constraint was violated",
                            ));
                        }
                    }
                }
                claims.insert((definition.name.clone(), key), owner);
            }
        }
        Ok(())
    }

    /// Read-only pre-WAL validation: no canonical definition or vector is changed here.
    pub fn validate_activate_profile(&self, profile: &EmbeddingProfile) -> Result<()> {
        profile.validate()?;
        if self
            .profile
            .get()
            .is_some_and(|current| current.as_ref() != profile)
            && !self.embedding_profile_is_mutable()
        {
            return Err(Error::new(
                ErrorCode::EmbeddingProfileImmutable,
                "project embedding profile is fixed by vector data or an index",
            ));
        }
        Ok(())
    }

    pub fn validate_initialize_semantic(&self, profile: &EmbeddingProfile) -> Result<()> {
        self.validate_activate_profile(profile)?;
        for (name, property) in [
            (SEMANTIC_NODE_INDEX, SEMANTIC_NODE_PROPERTY),
            (SEMANTIC_RELATIONSHIP_INDEX, SEMANTIC_RELATIONSHIP_PROPERTY),
        ] {
            if self
                .entries
                .get(name)
                .is_some_and(|entry| entry.definition.properties != [property])
            {
                return Err(Error::new(
                    ErrorCode::QueryType,
                    "automatic semantic index name is reserved",
                ));
            }
        }
        Ok(())
    }

    pub fn validate_create(
        &self,
        graph: &GraphStore,
        definition: &GraphIndexDefinition,
    ) -> Result<()> {
        if matches!(
            definition.name.as_str(),
            SEMANTIC_INDEX | SEMANTIC_NODE_INDEX | SEMANTIC_RELATIONSHIP_INDEX
        ) {
            return Err(Error::new(
                ErrorCode::QueryType,
                "automatic semantic index name is reserved",
            ));
        }
        validate_definition(graph, definition)?;
        if self.contains(&definition.name) {
            return Err(Error::new(
                ErrorCode::TransactionConflict,
                "index already exists",
            ));
        }
        if definition.kind == GraphIndexKind::Vector {
            let profile = self.profile.get().ok_or_else(|| {
                Error::new(
                    ErrorCode::EmbeddingProfileMismatch,
                    "vector index requires a project embedding profile",
                )
            })?;
            profile.validate()?;
            self.ensure_vector_property_available(definition.properties[0])?;
            let validator = VectorIndex::new_with_dtype(
                profile.dimension as usize,
                profile.similarity,
                profile.dtype,
            )?;
            for node in graph
                .nodes()
                .filter(|node| node.labels().contains(&definition.label))
            {
                match node.property(definition.properties[0]) {
                    Some(ScalarValue::List(list)) => {
                        validator.numeric_norm(&list)?;
                    }
                    None | Some(ScalarValue::Null) => {}
                    Some(_) => {
                        return Err(Error::new(
                            ErrorCode::EmbeddingProfileMismatch,
                            "vector property requires a numeric list",
                        ));
                    }
                }
            }
        }
        if definition.unique {
            let mut keys = BTreeSet::new();
            for node in graph
                .nodes()
                .filter(|node| node.labels().contains(&definition.label))
            {
                if let Some(key) = composite_key(node, &definition.properties)? {
                    if !keys.insert(key) {
                        return Err(Error::new(
                            ErrorCode::TransactionConflict,
                            "node property uniqueness constraint was violated",
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    pub fn validate_drop_index(&self, name: &str) -> Result<()> {
        if matches!(
            name,
            SEMANTIC_INDEX | SEMANTIC_NODE_INDEX | SEMANTIC_RELATIONSHIP_INDEX
        ) {
            return Err(Error::new(
                ErrorCode::QueryType,
                "automatic semantic indexes cannot be dropped",
            ));
        }
        let entry = self
            .entries
            .get(name)
            .ok_or_else(|| Error::new(ErrorCode::IndexUnavailable, "index does not exist"))?;
        if entry.definition.unique {
            return Err(Error::new(
                ErrorCode::QueryType,
                "remove an enforced constraint with DROP CONSTRAINT",
            ));
        }
        Ok(())
    }

    pub fn validate_drop_constraint(&self, name: &str) -> Result<()> {
        if self
            .entries
            .get(name)
            .is_none_or(|entry| !entry.definition.unique)
        {
            return Err(Error::new(
                ErrorCode::IndexUnavailable,
                "uniqueness constraint does not exist",
            ));
        }
        Ok(())
    }

    pub fn validate_create_embedding(
        &self,
        graph: &GraphStore,
        definition: &EmbeddingIndexDefinition,
        profile: &EmbeddingProfile,
        rows: &[(u64, Vec<u16>, u64)],
    ) -> Result<()> {
        self.validate_activate_profile(profile)?;
        let ordinary = GraphIndexDefinition {
            name: definition.name.clone(),
            kind: GraphIndexKind::Vector,
            label: definition.label,
            properties: vec![definition.target_property],
            unique: false,
        };
        if matches!(
            definition.name.as_str(),
            SEMANTIC_INDEX | SEMANTIC_NODE_INDEX | SEMANTIC_RELATIONSHIP_INDEX
        ) {
            return Err(Error::new(
                ErrorCode::QueryType,
                "automatic semantic index name is reserved",
            ));
        }
        validate_definition(graph, &ordinary)?;
        if self.contains(&definition.name) {
            return Err(Error::new(
                ErrorCode::TransactionConflict,
                "index already exists",
            ));
        }
        if definition.model.to_ascii_lowercase() != "default"
            || graph
                .catalog()
                .property_name(definition.source_property)
                .is_none()
        {
            return Err(Error::new(
                ErrorCode::EmbeddingProfileMismatch,
                "embedding declaration does not match the active model or schema",
            ));
        }
        self.ensure_vector_property_available(definition.target_property)?;
        let validator = VectorIndex::new_with_dtype(
            profile.dimension as usize,
            profile.similarity,
            profile.dtype,
        )?;
        for (owner, coordinates, _) in rows {
            validator.validate_coordinates(coordinates)?;
            let node = graph.node(crate::NodeId(*owner)).ok_or_else(|| {
                Error::new(ErrorCode::QueryType, "embedding owner does not exist")
            })?;
            if !node.labels().contains(&definition.label) {
                return Err(Error::new(
                    ErrorCode::QueryType,
                    "embedding owner does not match its declared label",
                ));
            }
        }
        Ok(())
    }

    pub fn validate_vector_mutations(&self, mutations: &[ResolvedVectorMutation]) -> Result<()> {
        self.validate_vector_mutations_with_planned_columns(mutations, None, &[])
    }

    /// Validates definitions introduced by this WAL batch without installing them first.
    pub fn validate_vector_mutations_with_planned_columns(
        &self,
        mutations: &[ResolvedVectorMutation],
        planned_profile: Option<&EmbeddingProfile>,
        planned_columns: &[PropertyId],
    ) -> Result<()> {
        if let Some(profile) = planned_profile {
            self.validate_activate_profile(profile)?;
        }
        for mutation in mutations {
            let property = match mutation {
                ResolvedVectorMutation::Upsert { property, .. }
                | ResolvedVectorMutation::Remove { property, .. } => property,
            };
            if let Some(column) = self.vector_columns.get(property) {
                if let ResolvedVectorMutation::Upsert { coordinates, .. } = mutation {
                    column.validate_coordinates(coordinates)?;
                }
            } else {
                let profile = planned_profile
                    .filter(|_| planned_columns.contains(property))
                    .ok_or_else(|| {
                        Error::new(
                            ErrorCode::EmbeddingProfileMismatch,
                            "resolved vector targets an undeclared vector property",
                        )
                    })?;
                if let ResolvedVectorMutation::Upsert { coordinates, .. } = mutation {
                    if coordinates.len() != profile.dimension as usize
                        || coordinates
                            .iter()
                            .any(|bits| !decode_coordinate(profile.dtype, *bits).is_finite())
                    {
                        return Err(Error::new(
                            ErrorCode::EmbeddingProfileMismatch,
                            "planned vector is incompatible with the embedding profile",
                        ));
                    }
                }
            }
        }
        Ok(())
    }

    pub fn validate_create_embedding_with_planned_schema(
        &self,
        graph: &GraphStore,
        definition: &EmbeddingIndexDefinition,
        profile: &EmbeddingProfile,
        rows: &[(u64, Vec<u16>, u64)],
        planned: &[GraphMutation],
    ) -> Result<()> {
        self.validate_activate_profile(profile)?;
        if definition.name.is_empty()
            || definition.name.len() > 255
            || matches!(
                definition.name.as_str(),
                SEMANTIC_INDEX | SEMANTIC_NODE_INDEX | SEMANTIC_RELATIONSHIP_INDEX
            )
        {
            return Err(Error::new(
                ErrorCode::QueryType,
                "embedding index name is invalid or reserved",
            ));
        }
        if self.contains(&definition.name) {
            return Err(Error::new(
                ErrorCode::TransactionConflict,
                "index already exists",
            ));
        }
        if !definition.model.eq_ignore_ascii_case("default") {
            return Err(Error::new(
                ErrorCode::EmbeddingProfileMismatch,
                "only the active embedding model is supported",
            ));
        }
        let label_exists=graph.catalog().label_name(definition.label).is_some() || planned.iter().any(|mutation|matches!(mutation,GraphMutation::DeclareLabel{id,..} if *id==definition.label));
        let property_exists = |property: PropertyId| {
            graph.catalog().property_name(property).is_some() || planned.iter().any(|mutation|matches!(mutation,GraphMutation::DeclareProperty{id,..} if *id==property))
        };
        if !label_exists
            || !property_exists(definition.source_property)
            || !property_exists(definition.target_property)
        {
            return Err(Error::new(
                ErrorCode::QueryType,
                "embedding definition references undeclared schema IDs",
            ));
        }
        self.ensure_vector_property_available(definition.target_property)?;
        for (owner, coordinates, _) in rows {
            if coordinates.len() != profile.dimension as usize
                || coordinates
                    .iter()
                    .any(|bits| !decode_coordinate(profile.dtype, *bits).is_finite())
            {
                return Err(Error::new(
                    ErrorCode::EmbeddingProfileMismatch,
                    "planned embedding row has invalid coordinates",
                ));
            }
            let matches=graph.node(crate::NodeId(*owner)).is_some_and(|node|node.labels().contains(&definition.label)) || planned.iter().any(|mutation|matches!(mutation,GraphMutation::InsertNode(node) if node.id.0==*owner && node.labels.contains(&definition.label)));
            if !matches {
                return Err(Error::new(
                    ErrorCode::QueryType,
                    "planned embedding owner does not match its declared label",
                ));
            }
        }
        Ok(())
    }

    pub fn apply_vector_mutations(&self, mutations: &[ResolvedVectorMutation]) -> Result<()> {
        self.validate_vector_mutations(mutations)?;
        for mutation in mutations {
            self.apply_vector_mutation(mutation)?;
        }
        Ok(())
    }

    /// Installs the two automatically maintained graph-content indexes. Their vector slots are
    /// derived storage addresses and never become properties or entities in the user's graph.
    pub fn initialize_semantic(&self, profile: EmbeddingProfile) -> Result<()> {
        self.validate_initialize_semantic(&profile)?;
        self.activate_profile(profile.clone())?;
        for (name, property) in [
            (SEMANTIC_NODE_INDEX, SEMANTIC_NODE_PROPERTY),
            (SEMANTIC_RELATIONSHIP_INDEX, SEMANTIC_RELATIONSHIP_PROPERTY),
        ] {
            if let Some(entry) = self.entries.get(name) {
                if entry.definition.properties != [property] {
                    return Err(Error::new(
                        ErrorCode::QueryType,
                        "automatic semantic index name is reserved",
                    ));
                }
                continue;
            }
            self.vector_columns.insert(
                property,
                Arc::new(VectorIndex::new_with_dtype(
                    profile.dimension as usize,
                    profile.similarity,
                    profile.dtype,
                )?),
            );
            self.entries.insert(
                name.to_owned(),
                Arc::new(IndexEntry {
                    definition: GraphIndexDefinition {
                        name: name.to_owned(),
                        kind: GraphIndexKind::Vector,
                        label: LabelId(u64::MAX),
                        properties: vec![property],
                        unique: false,
                    },
                    state: DerivedIndexState::Populating,
                    diagnostic: None,
                    runtime: None,
                }),
            );
        }
        self.invalidate_optimizer_generation();
        Ok(())
    }

    /// A cold ANN build is amortized across growth; ordinary edits stay in the exact delta.
    #[must_use]
    pub fn semantic_rebuild_needed(&self) -> bool {
        [SEMANTIC_NODE_INDEX, SEMANTIC_RELATIONSHIP_INDEX]
            .iter()
            .any(|name| {
                self.vector_search_source(name).is_some_and(|(vectors, _)| {
                    let property = if *name == SEMANTIC_NODE_INDEX {
                        SEMANTIC_NODE_PROPERTY
                    } else {
                        SEMANTIC_RELATIONSHIP_PROPERTY
                    };
                    let built_slots = self.semantic_built_slots.get(&property).unwrap_or(0);
                    vectors.row_count() >= 1_024
                        && vectors.row_count() >= built_slots.saturating_mul(2).max(1)
                })
            })
    }

    fn invalidate_optimizer_generation(&self) {
        self.optimizer_generation_cache
            .fetch_add(1, std::sync::atomic::Ordering::Release);
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
            && self.embeddings.is_empty()
            && self.profile.is_none()
            && self.vector_columns.is_empty()
    }

    /// Exports indexes exclusively for the preserved inactive graph implementation.
    pub fn inactive_device_image(
        &self,
        _graph: &crate::legacy::GraphStore,
    ) -> Result<IndexDeviceImage> {
        let vectors = self
            .vector_columns
            .iter()
            .map(|(property, column)| column.device_image(property))
            .collect::<Result<Vec<_>>>()?;
        let mut indexes = Vec::new();
        for (name, entry) in &self.entries {
            if entry.state != DerivedIndexState::Online {
                continue;
            }
            let runtime = entry.runtime.as_deref().ok_or_else(|| {
                Error::new(
                    ErrorCode::CorruptStorage,
                    "ONLINE index has no physical runtime",
                )
            })?;
            indexes.push(match runtime {
                RuntimeIndex::Equality(index) => DerivedIndexDeviceImage::Equality {
                    name: name.clone(),
                    postings: posting_device_image(&index.materialized())?,
                },
                RuntimeIndex::Range(index) => DerivedIndexDeviceImage::Range {
                    name: name.clone(),
                    postings: posting_device_image(&index.materialized())?,
                },
                RuntimeIndex::Text(index) => DerivedIndexDeviceImage::Text {
                    name: name.clone(),
                    postings: text_device_image(&index.materialized_postings())?,
                },
                RuntimeIndex::Vector {
                    property,
                    approximate,
                } => DerivedIndexDeviceImage::Vector {
                    name: name.clone(),
                    property: *property,
                    approximate: approximate
                        .as_ref()
                        .map(|index| index.device_image())
                        .transpose()?,
                },
            });
        }
        Ok(IndexDeviceImage {
            profile: self.profile.get().map(|profile| (*profile).clone()),
            vectors,
            indexes,
        })
    }

    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    #[must_use]
    pub fn profile(&self) -> Option<Arc<EmbeddingProfile>> {
        self.profile.get()
    }

    /// Whether replacing the project profile would preserve every canonical vector and vector
    /// index. Scalar, range, and text indexes do not depend on an embedding profile.
    #[must_use]
    pub fn embedding_profile_is_mutable(&self) -> bool {
        self.embeddings.is_empty()
            && self.vector_columns.is_empty()
            && self
                .entries
                .values()
                .all(|entry| entry.definition.kind != GraphIndexKind::Vector)
    }

    pub fn activate_profile(&self, profile: EmbeddingProfile) -> Result<()> {
        profile.validate()?;
        if let Some(current) = self.profile.get() {
            if current.as_ref() != &profile {
                if !self.embedding_profile_is_mutable() {
                    return Err(Error::new(
                        ErrorCode::EmbeddingProfileImmutable,
                        "project embedding profile is fixed by vector data or an index",
                    ));
                }
                self.invalidate_optimizer_generation();
                self.profile.set(profile);
            }
            return Ok(());
        }
        self.invalidate_optimizer_generation();
        self.profile.set(profile);
        Ok(())
    }

    pub fn create(&self, graph: &GraphStore, definition: GraphIndexDefinition) -> Result<()> {
        self.create_with_vector_population(graph, definition, true)
    }

    /// Applies the durable definition while deferring vector population to the selected local
    /// execution backend. Scalar families still build immediately because they have no separate
    /// accelerator builder and may enforce constraints during semantic preflight.
    pub fn create_deferred(
        &self,
        graph: &GraphStore,
        definition: GraphIndexDefinition,
    ) -> Result<()> {
        self.create_with_vector_population(graph, definition, false)
    }

    fn create_with_vector_population(
        &self,
        graph: &GraphStore,
        definition: GraphIndexDefinition,
        populate_vector: bool,
    ) -> Result<()> {
        self.validate_create(graph, &definition)?;
        if matches!(
            definition.name.as_str(),
            SEMANTIC_INDEX | SEMANTIC_NODE_INDEX | SEMANTIC_RELATIONSHIP_INDEX
        ) {
            return Err(Error::new(
                ErrorCode::QueryType,
                "automatic semantic index name is reserved",
            ));
        }
        validate_definition(graph, &definition)?;
        let is_vector = definition.kind == GraphIndexKind::Vector;
        if self.entries.contains_key(&definition.name) {
            return Err(Error::new(
                ErrorCode::TransactionConflict,
                "index already exists",
            ));
        }
        self.invalidate_optimizer_generation();
        if is_vector {
            let profile = self.profile.get().ok_or_else(|| {
                Error::new(
                    ErrorCode::EmbeddingProfileMismatch,
                    "vector index requires a project embedding profile",
                )
            })?;
            profile.validate()?;
            let property = *definition.properties.first().ok_or_else(|| {
                Error::new(ErrorCode::QueryType, "vector index requires one property")
            })?;
            self.ensure_vector_property_available(property)?;
            let matrix = self.vector_columns.get_or_insert(
                property,
                Arc::new(VectorIndex::new_with_dtype(
                    profile.dimension as usize,
                    profile.similarity,
                    profile.dtype,
                )?),
            );
            matrix.bind_canonical_graph(graph);
            for node in graph
                .nodes()
                .filter(|node| node.labels().contains(&definition.label))
            {
                if matches!(node.property(property), Some(ScalarValue::List(_))) {
                    matrix.upsert_property_owner(
                        graph,
                        crate::types::EntityKind::Node,
                        property,
                        node.id().0,
                        node.revision(),
                    )?;
                }
            }
        }
        let name = definition.name.clone();
        if definition.unique {
            let runtime = self.build_runtime(graph, &definition)?;
            validate_unique_runtime(&runtime)?;
            self.entries.insert(
                name,
                Arc::new(IndexEntry {
                    definition,
                    state: DerivedIndexState::Online,
                    diagnostic: None,
                    runtime: Some(Arc::new(runtime)),
                }),
            );
            return Ok(());
        }
        self.entries.insert(
            name.clone(),
            Arc::new(IndexEntry {
                definition,
                state: DerivedIndexState::Populating,
                diagnostic: None,
                runtime: None,
            }),
        );
        if is_vector && !populate_vector {
            Ok(())
        } else {
            self.rebuild(graph, &name)
        }
    }

    /// Creates an embedding declaration from already resolved canonical vectors.
    pub fn create_embedding(
        &self,
        graph: &GraphStore,
        definition: EmbeddingIndexDefinition,
        profile: EmbeddingProfile,
        rows: Vec<(u64, Vec<u16>, u64)>,
    ) -> Result<()> {
        self.create_embedding_with_vector_population(graph, definition, profile, rows, true)
    }

    pub fn create_embedding_deferred(
        &self,
        graph: &GraphStore,
        definition: EmbeddingIndexDefinition,
        profile: EmbeddingProfile,
        rows: Vec<(u64, Vec<u16>, u64)>,
    ) -> Result<()> {
        self.create_embedding_with_vector_population(graph, definition, profile, rows, false)
    }

    fn create_embedding_with_vector_population(
        &self,
        graph: &GraphStore,
        definition: EmbeddingIndexDefinition,
        profile: EmbeddingProfile,
        rows: Vec<(u64, Vec<u16>, u64)>,
        populate_vector: bool,
    ) -> Result<()> {
        self.validate_create_embedding(graph, &definition, &profile, &rows)?;
        if matches!(
            definition.name.as_str(),
            SEMANTIC_INDEX | SEMANTIC_NODE_INDEX | SEMANTIC_RELATIONSHIP_INDEX
        ) {
            return Err(Error::new(
                ErrorCode::QueryType,
                "automatic semantic index name is reserved",
            ));
        }
        self.activate_profile(profile.clone())?;
        if self.entries.contains_key(&definition.name) {
            return Err(Error::new(
                ErrorCode::TransactionConflict,
                "index already exists",
            ));
        }
        self.invalidate_optimizer_generation();
        if definition.model.to_ascii_lowercase() != "default" {
            return Err(Error::new(
                ErrorCode::EmbeddingProfileMismatch,
                "only the active project embedding model is supported",
            ));
        }
        if graph.catalog().label_name(definition.label).is_none()
            || graph
                .catalog()
                .property_name(definition.source_property)
                .is_none()
            || graph
                .catalog()
                .property_name(definition.target_property)
                .is_none()
        {
            return Err(Error::new(
                ErrorCode::QueryType,
                "embedding definition references unknown schema IDs",
            ));
        }
        self.ensure_vector_property_available(definition.target_property)?;
        let matrix = self.vector_columns.get_or_insert(
            definition.target_property,
            Arc::new(VectorIndex::new_with_dtype(
                profile.dimension as usize,
                profile.similarity,
                profile.dtype,
            )?),
        );
        let matrix = matrix.as_ref();
        for (entity_id, coordinates, revision) in rows {
            let node = graph.node(crate::NodeId(entity_id)).ok_or_else(|| {
                Error::new(
                    ErrorCode::QueryType,
                    "embedding row references a missing node",
                )
            })?;
            if !node.labels().contains(&definition.label) {
                return Err(Error::new(
                    ErrorCode::QueryType,
                    "embedding row does not match the declared label",
                ));
            }
            matrix.upsert_quantized(entity_id, &coordinates, revision)?;
        }
        let name = definition.name.clone();
        let ordinary = GraphIndexDefinition {
            name: name.clone(),
            kind: GraphIndexKind::Vector,
            label: definition.label,
            properties: vec![definition.target_property],
            unique: false,
        };
        self.embeddings.insert(name.clone(), definition);
        self.entries.insert(
            name.clone(),
            Arc::new(IndexEntry {
                definition: ordinary,
                state: DerivedIndexState::Populating,
                diagnostic: None,
                runtime: None,
            }),
        );
        if populate_vector {
            self.rebuild(graph, &name)
        } else {
            Ok(())
        }
    }

    fn ensure_vector_property_available(&self, property: PropertyId) -> Result<()> {
        if self.entries.values().any(|entry| {
            entry.definition.kind == GraphIndexKind::Vector
                && entry.definition.properties.first().copied() == Some(property)
        }) {
            return Err(Error::new(
                ErrorCode::TransactionConflict,
                "a vector index already owns this canonical vector property",
            ));
        }
        Ok(())
    }

    /// Rebuilds one local derived index. Canonical data remains valid if population fails.
    pub fn rebuild(&self, graph: &GraphStore, name: &str) -> Result<()> {
        let definition = self
            .entries
            .get(name)
            .map(|entry| entry.definition.clone())
            .ok_or_else(|| Error::new(ErrorCode::IndexUnavailable, "index does not exist"))?;
        self.invalidate_optimizer_generation();
        if definition.unique {
            let runtime = self.build_runtime(graph, &definition)?;
            validate_unique_runtime(&runtime)?;
            let mut entry = self
                .entries
                .edit(name)
                .ok_or_else(|| Error::internal("constraint disappeared during rebuild"))?;
            let entry = &mut *entry;
            entry.runtime = Some(Arc::new(runtime));
            entry.state = DerivedIndexState::Online;
            entry.diagnostic = None;
            return Ok(());
        }
        if let Some(mut entry) = self.entries.edit(name) {
            let entry = &mut *entry;
            entry.state = DerivedIndexState::Populating;
            entry.diagnostic = None;
            entry.runtime = None;
        }
        let result: Result<()> = match self.build_runtime(graph, &definition) {
            Ok(runtime) => {
                if let Some(property) = definition.properties.first().copied().filter(|property| {
                    matches!(
                        *property,
                        SEMANTIC_NODE_PROPERTY | SEMANTIC_RELATIONSHIP_PROPERTY
                    )
                }) {
                    if let Some(source) = self.vector_columns.get(&property) {
                        self.semantic_built_slots
                            .insert(property, source.row_count());
                    }
                }
                let mut entry = self
                    .entries
                    .edit(name)
                    .ok_or_else(|| Error::internal("index disappeared during population"))?;
                let entry = &mut *entry;
                entry.runtime = Some(Arc::new(runtime));
                entry.state = DerivedIndexState::Online;
                Ok(())
            }
            Err(error) => {
                let mut entry = self.entries.edit(name).ok_or_else(|| {
                    Error::internal("index disappeared while recording population failure")
                })?;
                let entry = &mut *entry;
                entry.state = DerivedIndexState::Failed;
                entry.diagnostic = Some(error.to_string());
                Ok(())
            }
        };
        result?;
        self.acknowledge_published_ann(name);
        Ok(())
    }

    fn acknowledge_published_ann(&self, name: &str) {
        if let Some((source, Some(ann))) = self.vector_search_source(name) {
            source.acknowledge_ann(&ann);
        }
    }

    /// Marks a vector rebuild request without executing a host builder during semantic
    /// validation. An existing validated runtime stays ONLINE until its replacement is ready.
    pub fn rebuild_deferred(&self, graph: &GraphStore, name: &str) -> Result<()> {
        if name == SEMANTIC_INDEX {
            self.rebuild_deferred(graph, SEMANTIC_NODE_INDEX)?;
            return self.rebuild_deferred(graph, SEMANTIC_RELATIONSHIP_INDEX);
        }
        let entry = self
            .entries
            .get(name)
            .ok_or_else(|| Error::new(ErrorCode::IndexUnavailable, "index does not exist"))?;
        if entry.definition.kind != GraphIndexKind::Vector {
            return self.rebuild(graph, name);
        }
        self.invalidate_optimizer_generation();
        let mut entry = self
            .entries
            .edit(name)
            .ok_or_else(|| Error::internal("vector index disappeared during rebuild request"))?;
        let entry = &mut *entry;
        // Readers keep using the canonical vectors while this derived builder is pending.
        entry.state = DerivedIndexState::Populating;
        entry.diagnostic = None;
        Ok(())
    }

    /// Retries failed derived vector builds when an execution backend is (re)admitted.
    pub fn retry_failed_vectors(&self) {
        for mut entry in self
            .entries
            .keys()
            .filter_map(|key| self.entries.edit(&key))
        {
            if entry.definition.kind == GraphIndexKind::Vector
                && entry.state == DerivedIndexState::Failed
            {
                let entry = &mut *entry;
                entry.state = DerivedIndexState::Populating;
                entry.diagnostic = None;
            }
        }
        self.invalidate_optimizer_generation();
    }

    /// Builds every local vector generation through the selected backend and atomically replaces
    /// each runtime after cancellation and structural checks.
    pub fn rebuild_vectors_with(
        &self,
        mut builder: impl FnMut(&VectorIndex, IvfPqConfig) -> Result<IvfPqIndex>,
    ) -> Result<()> {
        let names = self
            .entries
            .iter()
            .filter_map(|(name, entry)| {
                (entry.definition.kind == GraphIndexKind::Vector
                    && entry.state == DerivedIndexState::Populating)
                    .then_some(name.clone())
            })
            .collect::<Vec<_>>();
        if !names.is_empty() {
            self.invalidate_optimizer_generation();
        }
        for name in names {
            let definition = self
                .entries
                .get(&name)
                .map(|entry| entry.definition.clone())
                .ok_or_else(|| Error::internal("vector definition disappeared before build"))?;
            let property = *definition.properties.first().ok_or_else(|| {
                Error::new(
                    ErrorCode::CorruptStorage,
                    "vector index has no source property",
                )
            })?;
            let source = self.vector_columns.get(&property).ok_or_else(|| {
                Error::new(
                    ErrorCode::CorruptStorage,
                    "vector index has no canonical vector column",
                )
            })?;
            let candidate = if source.is_empty() {
                Ok(None)
            } else {
                builder(&source, ivf_config(&source)).and_then(|index| {
                    if index.build_generation() == [0_u8; 32] {
                        return Err(Error::new(
                            ErrorCode::IndexUnavailable,
                            "IVF-PQ builder returned an unvalidated generation",
                        ));
                    }
                    Ok(Some(Arc::new(index)))
                })
            };
            match candidate {
                Ok(approximate) => {
                    let replacement = RuntimeIndex::Vector {
                        property,
                        approximate,
                    };
                    let mut entry = self.entries.edit(&name).ok_or_else(|| {
                        Error::internal("vector index disappeared before atomic publication")
                    })?;
                    let entry = &mut *entry;
                    entry.runtime = Some(Arc::new(replacement));
                    entry.state = DerivedIndexState::Online;
                    entry.diagnostic = None;
                    if matches!(
                        property,
                        SEMANTIC_NODE_PROPERTY | SEMANTIC_RELATIONSHIP_PROPERTY
                    ) {
                        self.semantic_built_slots
                            .insert(property, source.row_count());
                    }
                }
                Err(error) => {
                    let mut entry = self.entries.edit(&name).ok_or_else(|| {
                        Error::internal("vector index disappeared while recording build failure")
                    })?;
                    let entry = &mut *entry;
                    if entry.runtime.is_none() {
                        entry.state = DerivedIndexState::Failed;
                    } else {
                        // Failed rebuilds never evict the previously validated generation.
                        entry.state = DerivedIndexState::Online;
                    }
                    entry.diagnostic = Some(error.to_string());
                }
            }
            self.acknowledge_published_ann(&name);
        }
        Ok(())
    }

    pub fn rebuild_all(&self, graph: &GraphStore) -> Result<()> {
        let names = self.entries.keys().collect::<Vec<_>>();
        for name in names {
            self.rebuild(graph, &name)?;
        }
        Ok(())
    }

    /// Rebuilds every derived family after a layer-filtered graph is compacted. Vector columns are
    /// first reduced to the stable entities retained by the compacted graph.
    pub fn rebuild_row_indexes(&self, graph: &GraphStore) -> Result<()> {
        self.invalidate_optimizer_generation();
        let retained = graph
            .nodes()
            .map(|node| node.id().0)
            .collect::<BTreeSet<_>>();
        let mut vector_row_remaps = BTreeMap::new();
        let retained_relationships = graph
            .edges()
            .map(|edge| edge.id().0)
            .collect::<BTreeSet<_>>();
        for (property, column) in &self.vector_columns {
            let owners = if property == SEMANTIC_RELATIONSHIP_PROPERTY {
                &retained_relationships
            } else {
                &retained
            };
            let remap = column.retain_entities(owners)?;

            vector_row_remaps.insert(property, remap);
        }
        let names = self.entries.keys().collect::<Vec<_>>();
        for name in names {
            let is_vector = self
                .entries
                .get(&name)
                .is_some_and(|entry| entry.definition.kind == GraphIndexKind::Vector);
            if !is_vector {
                self.rebuild(graph, &name)?;
                continue;
            }

            let mut entry = self
                .entries
                .edit(&name)
                .ok_or_else(|| Error::internal("vector index disappeared during filtering"))?;
            let entry = &mut *entry;
            let property = entry
                .definition
                .properties
                .first()
                .copied()
                .ok_or_else(|| {
                    Error::new(ErrorCode::CorruptStorage, "vector index has no property")
                })?;
            if !self.vector_columns.contains_key(&property) {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "vector index has no canonical vector column",
                ));
            }
            let remap = vector_row_remaps.get(&property).ok_or_else(|| {
                Error::new(
                    ErrorCode::CorruptStorage,
                    "vector index row remap is unavailable",
                )
            })?;
            let approximate = match entry.runtime.as_deref() {
                Some(RuntimeIndex::Vector {
                    property: runtime_property,
                    approximate,
                }) if *runtime_property == property => approximate
                    .as_ref()
                    .map(|index| index.remap_rows(remap))
                    .transpose()?
                    .flatten()
                    .map(Arc::new),
                Some(RuntimeIndex::Vector { .. }) => {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        "vector runtime property does not match its definition",
                    ));
                }
                Some(_) => {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        "vector definition has a non-vector runtime",
                    ));
                }
                None => None,
            };
            entry.runtime = Some(Arc::new(RuntimeIndex::Vector {
                property,
                approximate,
            }));
            entry.state = DerivedIndexState::Online;
            entry.diagnostic = None;
        }
        Ok(())
    }

    pub fn drop_index(&self, name: &str) -> Result<()> {
        if matches!(
            name,
            SEMANTIC_INDEX | SEMANTIC_NODE_INDEX | SEMANTIC_RELATIONSHIP_INDEX
        ) {
            return Err(Error::new(
                ErrorCode::QueryType,
                "automatic semantic indexes are maintained with graph content and cannot be dropped",
            ));
        }
        if self
            .entries
            .get(name)
            .is_some_and(|entry| entry.definition.unique)
        {
            return Err(Error::new(
                ErrorCode::QueryType,
                "an enforced uniqueness constraint must be removed with DROP CONSTRAINT",
            ));
        }
        self.remove_definition(name)
    }

    pub fn drop_constraint(&self, name: &str) -> Result<()> {
        if self
            .entries
            .get(name)
            .is_none_or(|entry| !entry.definition.unique)
        {
            return Err(Error::new(
                ErrorCode::IndexUnavailable,
                "uniqueness constraint does not exist",
            ));
        }
        self.remove_definition(name)
    }

    fn remove_definition(&self, name: &str) -> Result<()> {
        self.invalidate_optimizer_generation();
        let Some(_entry) = self.entries.remove(name) else {
            return Err(Error::new(
                ErrorCode::IndexUnavailable,
                "index does not exist",
            ));
        };
        self.embeddings.remove(name);
        let used_targets = self
            .entries
            .values()
            .filter_map(|entry| {
                (entry.definition.kind == GraphIndexKind::Vector)
                    .then(|| entry.definition.properties.first().copied())
                    .flatten()
            })
            .collect::<BTreeSet<_>>();
        self.vector_columns
            .retain(|property, _| used_targets.contains(property));
        Ok(())
    }

    #[must_use]
    pub fn has_unique_constraint(&self, label: LabelId, property: PropertyId) -> bool {
        self.entries.values().any(|entry| {
            entry.state == DerivedIndexState::Online
                && entry.definition.unique
                && entry.definition.label == label
                && entry.definition.properties.as_slice() == [property]
        })
    }

    pub fn constraint_definitions(&self) -> impl Iterator<Item = GraphIndexDefinition> {
        self.entries
            .values()
            .filter(|entry| entry.definition.unique)
            .map(|entry| entry.definition.clone())
    }

    pub fn statuses(&self) -> impl Iterator<Item = IndexStatus> + '_ {
        let combined = self
            .entries
            .get(SEMANTIC_NODE_INDEX)
            .zip(self.entries.get(SEMANTIC_RELATIONSHIP_INDEX))
            .map(|(nodes, edges)| IndexStatus {
                name: SEMANTIC_INDEX.to_owned(),
                kind: GraphIndexKind::Vector,
                state: if nodes.state == DerivedIndexState::Failed
                    || edges.state == DerivedIndexState::Failed
                {
                    DerivedIndexState::Failed
                } else if nodes.state == DerivedIndexState::Online
                    && edges.state == DerivedIndexState::Online
                {
                    DerivedIndexState::Online
                } else {
                    DerivedIndexState::Populating
                },
                diagnostic: nodes
                    .diagnostic
                    .clone()
                    .or_else(|| edges.diagnostic.clone()),
            });
        self.entries
            .values()
            .map(|entry| IndexStatus {
                name: entry.definition.name.clone(),
                kind: entry.definition.kind,
                state: entry.state,
                diagnostic: entry.diagnostic.clone(),
            })
            .chain(combined)
    }

    /// Stable generation of the schema and local lifecycle facts that can change physical access
    /// path legality. Canonical posting contents are intentionally excluded: data churn may make
    /// an estimate stale, but cannot make an ONLINE access path semantically invalid.
    #[must_use]
    pub fn optimizer_generation(&self) -> [u8; 32] {
        {
            let mut hasher = blake3::Hasher::new();
            hasher.update(b"index-generation-v1\0");
            for (name, entry) in &self.entries {
                hash_bytes(&mut hasher, name.as_bytes());
                hasher.update(&[graph_index_kind_byte(entry.definition.kind)]);
                hasher.update(&[u8::from(entry.definition.unique)]);
                hasher.update(&entry.definition.label.0.to_le_bytes());
                hasher.update(&(entry.definition.properties.len() as u64).to_le_bytes());
                for property in &entry.definition.properties {
                    hasher.update(&property.0.to_le_bytes());
                }
                hasher.update(&[derived_index_state_byte(entry.state)]);
            }
            for (name, definition) in &self.embeddings {
                hash_bytes(&mut hasher, name.as_bytes());
                hasher.update(&definition.label.0.to_le_bytes());
                hasher.update(&definition.source_property.0.to_le_bytes());
                hasher.update(&definition.target_property.0.to_le_bytes());
                hash_bytes(&mut hasher, definition.model.as_bytes());
            }
            if let Some(profile) = self.profile.get() {
                hasher.update(&profile.profile_hash);
            }
            *hasher.finalize().as_bytes()
        }
    }

    /// Collects bounded planner statistics without exposing or persisting derived pages.
    #[must_use]
    pub fn optimizer_statistics(&self) -> Vec<OptimizerIndexStatistics> {
        self.entries
            .iter()
            .filter_map(|(name, entry)| {
                if entry.state != DerivedIndexState::Online {
                    return None;
                }
                let runtime = entry.runtime.as_deref()?;
                let (
                    distinct_keys,
                    total_postings,
                    largest_posting,
                    vector_rows,
                    candidate_budget,
                    resident_bytes,
                ) = match runtime {
                    RuntimeIndex::Equality(index) => {
                        let shape = scalar_posting_shape(&index.postings);
                        (shape.0, shape.1, shape.2, 0, 0, shape.3)
                    }
                    RuntimeIndex::Range(index) => {
                        let shape = scalar_posting_shape(&index.postings);
                        (shape.0, shape.1, shape.2, 0, 0, shape.3)
                    }
                    RuntimeIndex::Text(index) => {
                        let shape = text_posting_shape(index);
                        (shape.0, shape.1, shape.2, 0, 0, shape.3)
                    }
                    RuntimeIndex::Vector {
                        property,
                        approximate,
                    } => {
                        let rows = self
                            .vector_columns
                            .get(property)
                            .map_or(0_u64, |vectors| vectors.len() as u64);
                        let candidate_budget = approximate
                            .as_ref()
                            .map_or(0_u64, |ann| ann.config.candidate_budget as u64);
                        let dimension = self
                            .profile
                            .get()
                            .map_or(0_u64, |profile| u64::from(profile.dimension));
                        let bytes = rows.saturating_mul(
                            8_u64
                                .saturating_add(8)
                                .saturating_add(1)
                                .saturating_add(dimension.saturating_mul(2)),
                        );
                        (0, 0, 0, rows, candidate_budget, bytes)
                    }
                };
                Some(OptimizerIndexStatistics {
                    name: name.clone(),
                    kind: entry.definition.kind,
                    label: entry.definition.label,
                    properties: entry.definition.properties.clone(),
                    distinct_keys,
                    total_postings,
                    largest_posting,
                    vector_rows,
                    ann_candidate_budget: candidate_budget,
                    resident_bytes,
                })
            })
            .collect()
    }

    pub fn definitions(&self) -> impl Iterator<Item = GraphIndexDefinition> {
        self.entries.values().map(|entry| entry.definition.clone())
    }

    /// Selects an exact ONLINE equality posting by count and then stable index name without
    /// materializing the posting itself.
    pub fn equality_candidate_estimate(
        &self,
        label: LabelId,
        values: &BTreeMap<PropertyId, ScalarValue>,
    ) -> Result<Option<ScalarIndexCandidate>> {
        let mut best: Option<ScalarIndexCandidate> = None;
        for (name, entry) in &self.entries {
            if entry.state != DerivedIndexState::Online
                || entry.definition.label != label
                || !matches!(
                    entry.definition.kind,
                    GraphIndexKind::Equality | GraphIndexKind::Range
                )
                || entry
                    .definition
                    .properties
                    .iter()
                    .any(|property| !values.contains_key(property))
            {
                continue;
            }
            let Some(key) = scalar_lookup_key(&entry.definition.properties, values)? else {
                let candidate = ScalarIndexCandidate {
                    name: name.clone(),
                    estimated_rows: 0,
                };
                if best.as_ref().is_none_or(|current| {
                    (candidate.estimated_rows, candidate.name.as_str())
                        < (current.estimated_rows, current.name.as_str())
                }) {
                    best = Some(candidate);
                }
                continue;
            };
            let runtime = entry.runtime.as_deref().ok_or_else(|| {
                Error::new(
                    ErrorCode::CorruptStorage,
                    "ONLINE scalar index has no physical runtime",
                )
            })?;
            let estimated_rows = match runtime {
                RuntimeIndex::Equality(index) => index.postings.count(&key),
                RuntimeIndex::Range(index) => index.postings.count(&key),
                RuntimeIndex::Text(_) | RuntimeIndex::Vector { .. } => {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        "scalar index definition and runtime kind differ",
                    ));
                }
            };
            let candidate = ScalarIndexCandidate {
                name: name.clone(),
                estimated_rows,
            };
            if best.as_ref().is_none_or(|current| {
                (candidate.estimated_rows, candidate.name.as_str())
                    < (current.estimated_rows, current.name.as_str())
            }) {
                best = Some(candidate);
            }
        }
        Ok(best)
    }

    /// Reads one named equality/range posting selected by a cached physical plan. `None` means
    /// the derived path is no longer legal and the caller must use the canonical scan fallback.
    pub fn equality_candidates_named(
        &self,
        name: &str,
        label: LabelId,
        values: &BTreeMap<PropertyId, ScalarValue>,
    ) -> Result<Option<Vec<u32>>> {
        let Some(entry) = self.entries.get(name) else {
            return Ok(None);
        };
        if entry.state != DerivedIndexState::Online
            || entry.definition.label != label
            || !matches!(
                entry.definition.kind,
                GraphIndexKind::Equality | GraphIndexKind::Range
            )
            || entry
                .definition
                .properties
                .iter()
                .any(|property| !values.contains_key(property))
        {
            return Ok(None);
        }
        let Some(key) = scalar_lookup_key(&entry.definition.properties, values)? else {
            return Ok(Some(Vec::new()));
        };
        let runtime = entry.runtime.as_deref().ok_or_else(|| {
            Error::new(
                ErrorCode::CorruptStorage,
                "ONLINE scalar index has no physical runtime",
            )
        })?;
        let rows = match runtime {
            RuntimeIndex::Equality(index) => index
                .get(&key)
                .map(|rows| rows.iter().collect())
                .unwrap_or_default(),
            RuntimeIndex::Range(index) => index
                .between(Some((&key, true)), Some((&key, true)))
                .iter()
                .collect(),
            RuntimeIndex::Text(_) | RuntimeIndex::Vector { .. } => {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "scalar index definition and runtime kind differ",
                ));
            }
        };
        Ok(Some(rows))
    }

    /// Bounded equivalent of `equality_candidates_named` for retrieval seeding. The posting is
    /// never expanded into an unbounded row vector.
    pub fn equality_candidates_named_bounded(
        &self,
        name: &str,
        label: LabelId,
        values: &BTreeMap<PropertyId, ScalarValue>,
        limit: usize,
    ) -> Result<Option<Vec<u32>>> {
        if limit == 0 {
            return Ok(Some(Vec::new()));
        }
        let Some(entry) = self.entries.get(name) else {
            return Ok(None);
        };
        if entry.state != DerivedIndexState::Online
            || entry.definition.label != label
            || !matches!(
                entry.definition.kind,
                GraphIndexKind::Equality | GraphIndexKind::Range
            )
            || entry
                .definition
                .properties
                .iter()
                .any(|property| !values.contains_key(property))
        {
            return Ok(None);
        }
        let Some(key) = scalar_lookup_key(&entry.definition.properties, values)? else {
            return Ok(Some(Vec::new()));
        };
        let runtime = entry.runtime.as_deref().ok_or_else(|| {
            Error::new(
                ErrorCode::CorruptStorage,
                "ONLINE scalar index has no physical runtime",
            )
        })?;
        let rows = match runtime {
            RuntimeIndex::Equality(index) => index.get_bounded(&key, limit),
            RuntimeIndex::Range(index) => index.exact_bounded(&key, limit),
            RuntimeIndex::Text(_) | RuntimeIndex::Vector { .. } => {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "scalar index definition and runtime kind differ",
                ));
            }
        };
        Ok(Some(rows))
    }

    /// Searches all ONLINE declared text indexes with bounded work and merges them by matched
    /// term count, then dense row. This is an index seed, never a semantic evidence score.
    pub fn bounded_text_candidates(&self, query: &str, limit: usize) -> Result<Vec<u32>> {
        if limit == 0 || query.trim().is_empty() {
            return Ok(Vec::new());
        }
        let mut scores = BTreeMap::<u32, u32>::new();
        for entry in self.entries.values().take(64) {
            if entry.state != DerivedIndexState::Online
                || entry.definition.kind != GraphIndexKind::Text
            {
                continue;
            }
            let Some(runtime) = entry.runtime.as_deref() else {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "ONLINE text index has no physical runtime",
                ));
            };
            let RuntimeIndex::Text(index) = runtime else {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "text index definition and runtime kind differ",
                ));
            };
            for (row, score) in index.bounded_ranked_search(query, limit.saturating_mul(2)) {
                scores
                    .entry(row)
                    .and_modify(|total| *total = total.saturating_add(u32::from(score)))
                    .or_insert(u32::from(score));
            }
        }
        let mut ranked = scores.into_iter().collect::<Vec<_>>();
        ranked.sort_unstable_by(|left, right| {
            right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0))
        });
        ranked.truncate(limit);
        Ok(ranked.into_iter().map(|(row, _)| row).collect())
    }

    /// Returns the smallest exact ONLINE scalar posting compatible with the supplied label and
    /// equality values. Selection is deterministic by posting size then index name.
    pub fn equality_candidates(
        &self,
        label: LabelId,
        values: &BTreeMap<PropertyId, ScalarValue>,
    ) -> Result<Option<Vec<u32>>> {
        let mut best: Option<(usize, String, Vec<u32>)> = None;
        for (name, entry) in &self.entries {
            if entry.state != DerivedIndexState::Online
                || entry.definition.label != label
                || !matches!(
                    entry.definition.kind,
                    GraphIndexKind::Equality | GraphIndexKind::Range
                )
                || entry
                    .definition
                    .properties
                    .iter()
                    .any(|property| !values.contains_key(property))
            {
                continue;
            }
            let mut keys = Vec::with_capacity(entry.definition.properties.len());
            let mut contains_null = false;
            for property in &entry.definition.properties {
                let value = values
                    .get(property)
                    .ok_or_else(|| Error::internal("index value disappeared"))?;
                if matches!(value, ScalarValue::Null) {
                    contains_null = true;
                    break;
                }
                keys.push(IndexKey::try_from(value)?);
            }
            let key = match keys.as_slice() {
                [value] => value.clone(),
                _ => IndexKey::Composite(keys),
            };
            let rows = if contains_null {
                Vec::new()
            } else {
                let runtime = entry.runtime.as_deref().ok_or_else(|| {
                    Error::new(
                        ErrorCode::CorruptStorage,
                        "ONLINE scalar index has no physical runtime",
                    )
                })?;
                match runtime {
                    RuntimeIndex::Equality(index) => index
                        .get(&key)
                        .map(|rows| rows.iter().collect())
                        .unwrap_or_default(),
                    RuntimeIndex::Range(index) => index
                        .between(Some((&key, true)), Some((&key, true)))
                        .iter()
                        .collect(),
                    RuntimeIndex::Text(_) | RuntimeIndex::Vector { .. } => {
                        return Err(Error::new(
                            ErrorCode::CorruptStorage,
                            "scalar index definition and runtime kind differ",
                        ));
                    }
                }
            };
            let candidate = (rows.len(), name, rows);
            if best.as_ref().is_none_or(|current| {
                (candidate.0, candidate.1.as_str()) < (current.0, current.1.as_str())
            }) {
                best = Some(candidate);
            }
        }
        Ok(best.map(|(_, _, rows)| rows))
    }

    pub fn embedding_definitions(&self) -> impl Iterator<Item = EmbeddingIndexDefinition> {
        self.embeddings.values()
    }

    #[must_use]
    pub fn embedding_definition(&self, name: &str) -> Option<EmbeddingIndexDefinition> {
        self.embeddings.get(name)
    }

    /// Returns the project's single canonical embedding column when one is declared.
    pub fn canonical_embedding_column(&self) -> Result<Option<(PropertyId, Arc<VectorIndex>)>> {
        let mut targets = self
            .embeddings
            .values()
            .map(|definition| definition.target_property)
            .collect::<BTreeSet<_>>()
            .into_iter();
        let Some(property) = targets.next() else {
            return Ok(None);
        };
        if targets.next().is_some() {
            return Err(Error::new(
                ErrorCode::EmbeddingProfileMismatch,
                "project has more than one canonical embedding target",
            ));
        }
        let column = self.vector_columns.get(&property).ok_or_else(|| {
            Error::new(
                ErrorCode::CorruptStorage,
                "embedding definition has no canonical vector column",
            )
        })?;
        Ok(Some((property, column)))
    }

    /// Removes old postings before a canonical graph mutation changes or deletes a row.
    pub fn before_graph_apply(&self, graph: &GraphStore, mutation: &GraphMutation) -> Result<()> {
        match mutation {
            GraphMutation::SetNodeProperty { node, .. }
            | GraphMutation::AddNodeLabels { node, .. }
            | GraphMutation::RemoveNodeLabels { node, .. } => {
                if let Some(view) = graph.node(*node) {
                    self.remove_node(view, false)?;
                }
            }
            GraphMutation::DeleteNode {
                node,
                detach,
                revision,
            } => {
                if let Some(view) = graph.node(*node) {
                    self.remove_node(view, true)?;
                }
                if let Some(column) = self.vector_columns.get(&crate::SEMANTIC_NODE_PROPERTY) {
                    column.remove(node.0, *revision);
                }
                if *detach {
                    if let Some(column) = self
                        .vector_columns
                        .get(&crate::SEMANTIC_RELATIONSHIP_PROPERTY)
                    {
                        for edge in graph.incident_edge_ids(*node)? {
                            column.remove(edge.0, *revision);
                        }
                    }
                }
            }
            GraphMutation::DeleteEdge { edge, revision } => {
                if let Some(column) = self
                    .vector_columns
                    .get(&crate::SEMANTIC_RELATIONSHIP_PROPERTY)
                {
                    column.remove(edge.0, *revision);
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Adds current postings after a canonical graph mutation creates or updates a row.
    pub fn after_graph_apply(&self, graph: &GraphStore, mutation: &GraphMutation) -> Result<()> {
        match mutation {
            GraphMutation::InsertNode(input) => {
                if let Some(view) = graph.node(input.id) {
                    self.insert_node(view)?;
                }
            }
            GraphMutation::SetNodeProperty { node, .. }
            | GraphMutation::AddNodeLabels { node, .. }
            | GraphMutation::RemoveNodeLabels { node, .. } => {
                if let Some(view) = graph.node(*node) {
                    self.insert_node(view)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Binds recovered property-owner references to the one canonical graph.
    pub fn bind_catalog_graph(&self, graph: &GraphStore) {
        for matrix in self.vector_columns.values() {
            matrix.bind_canonical_graph(graph);
        }
    }

    /// Publishes owner references for user vector properties; generated embeddings own quantized cells.
    pub fn apply_vector_mutation_from_graph(
        &self,
        graph: &GraphStore,
        mutation: &ResolvedVectorMutation,
    ) -> Result<()> {
        if let ResolvedVectorMutation::Upsert {
            property,
            entity_id,
            revision,
            ..
        } = mutation
        {
            let kind = self.entries.values().find_map(|entry| {
                (entry.definition.kind == GraphIndexKind::Vector
                    && entry.definition.properties.first() == Some(property)
                    && !self.embeddings.contains_key(&entry.definition.name)
                    && !matches!(
                        entry.definition.name.as_str(),
                        SEMANTIC_INDEX | SEMANTIC_NODE_INDEX | SEMANTIC_RELATIONSHIP_INDEX
                    ))
                .then_some(crate::types::EntityKind::Node)
            });
            if let Some(kind) = kind {
                let matrix = self.vector_columns.get(property).ok_or_else(|| {
                    Error::new(
                        ErrorCode::EmbeddingProfileMismatch,
                        "vector property is undeclared",
                    )
                })?;
                return matrix.upsert_property_owner(graph, kind, *property, *entity_id, *revision);
            }
        }
        self.apply_vector_mutation(mutation)
    }

    pub fn apply_vector_mutation(&self, mutation: &ResolvedVectorMutation) -> Result<()> {
        let (property, entity_id, revision) = match mutation {
            ResolvedVectorMutation::Upsert {
                property,
                entity_id,
                revision,
                ..
            }
            | ResolvedVectorMutation::Remove {
                property,
                entity_id,
                revision,
            } => (*property, *entity_id, *revision),
        };
        let matrix = self.vector_columns.get(&property).ok_or_else(|| {
            Error::new(
                ErrorCode::EmbeddingProfileMismatch,
                "resolved vector targets an undeclared vector property",
            )
        })?;
        let matrix = matrix.as_ref();
        match mutation {
            ResolvedVectorMutation::Upsert { coordinates, .. } => {
                matrix.upsert_quantized(entity_id, coordinates, revision)?;
            }
            ResolvedVectorMutation::Remove { .. } => matrix.remove(entity_id, revision),
        }
        Ok(())
    }

    /// Returns an ONLINE vector matrix and its immutable ANN base for native SEARCH.
    #[must_use]
    pub fn vector_search_source(
        &self,
        name: &str,
    ) -> Option<(Arc<VectorIndex>, Option<Arc<IvfPqIndex>>)> {
        let entry = self.entries.get(name)?;
        if entry.definition.kind != GraphIndexKind::Vector {
            return None;
        }
        let property = *entry.definition.properties.first()?;
        let source = self.vector_columns.get(&property)?;
        let Some(RuntimeIndex::Vector {
            property,
            approximate,
        }) = entry.runtime.as_deref()
        else {
            return Some((source, None));
        };
        Some((self.vector_columns.get(property)?, approximate.clone()))
    }

    fn build_runtime(
        &self,
        graph: &GraphStore,
        definition: &GraphIndexDefinition,
    ) -> Result<RuntimeIndex> {
        match definition.kind {
            GraphIndexKind::Equality => {
                let index = EqualityIndex::default();
                for node in graph
                    .nodes()
                    .filter(|node| node.labels().contains(&definition.label))
                {
                    if let Some(key) = composite_key(node, &definition.properties)? {
                        index.insert_key(key, node.dense());
                    }
                }
                Ok(RuntimeIndex::Equality(index))
            }
            GraphIndexKind::Range => {
                let index = RangeIndex::default();
                for node in graph
                    .nodes()
                    .filter(|node| node.labels().contains(&definition.label))
                {
                    if let Some(key) = composite_key(node, &definition.properties)? {
                        index.insert_key(key, node.dense());
                    }
                }
                Ok(RuntimeIndex::Range(index))
            }
            GraphIndexKind::Text => {
                let property = definition.properties[0];
                let index = TextIndex::default();
                for node in graph
                    .nodes()
                    .filter(|node| node.labels().contains(&definition.label))
                {
                    match node.property(property) {
                        Some(ScalarValue::String(value)) => index.upsert(node.dense(), &value),
                        Some(ScalarValue::Null) | None => {}
                        Some(_) => {
                            return Err(Error::new(
                                ErrorCode::QueryType,
                                "text index encountered a non-string value",
                            ));
                        }
                    }
                }
                Ok(RuntimeIndex::Text(index))
            }
            GraphIndexKind::Vector => {
                let property = definition.properties[0];
                let matrix = self.vector_columns.get(&property).ok_or_else(|| {
                    Error::new(
                        ErrorCode::EmbeddingProfileMismatch,
                        "vector property has no canonical matrix",
                    )
                })?;
                let approximate = if matrix.is_empty() {
                    None
                } else {
                    Some(Arc::new(IvfPqIndex::build(&matrix, ivf_config(&matrix))?))
                };
                Ok(RuntimeIndex::Vector {
                    property,
                    approximate,
                })
            }
        }
    }

    fn remove_node(&self, node: NodeView<'_>, remove_vectors: bool) -> Result<()> {
        for entry in self.entries.values() {
            if entry.state != DerivedIndexState::Online
                || !node.labels().contains(&entry.definition.label)
            {
                continue;
            }
            let Some(runtime) = entry.runtime.as_ref() else {
                continue;
            };
            if matches!(runtime.as_ref(), RuntimeIndex::Vector { .. }) {
                continue;
            }
            match runtime.as_ref() {
                RuntimeIndex::Equality(index) => {
                    if let Some(key) = composite_key(node, &entry.definition.properties)? {
                        index.remove_key(&key, node.dense());
                    }
                }
                RuntimeIndex::Range(index) => {
                    if let Some(key) = composite_key(node, &entry.definition.properties)? {
                        index.remove_key(&key, node.dense());
                    }
                }
                RuntimeIndex::Text(index) => index.remove(node.dense()),
                RuntimeIndex::Vector { .. } => {}
            }
        }
        if remove_vectors {
            for definition in self.embeddings.values() {
                if node.labels().contains(&definition.label) {
                    if let Some(matrix) = self.vector_columns.get(&definition.target_property) {
                        matrix.remove(node.id().0, node.revision());
                    }
                }
            }
        }
        Ok(())
    }

    fn insert_node(&self, node: NodeView<'_>) -> Result<()> {
        for entry in self.entries.values() {
            if entry.state != DerivedIndexState::Online
                || !node.labels().contains(&entry.definition.label)
            {
                continue;
            }
            let Some(runtime) = entry.runtime.as_ref() else {
                continue;
            };
            if matches!(runtime.as_ref(), RuntimeIndex::Vector { .. }) {
                continue;
            }
            match runtime.as_ref() {
                RuntimeIndex::Equality(index) => {
                    if let Some(key) = composite_key(node, &entry.definition.properties)? {
                        if entry.definition.unique
                            && index
                                .get(&key)
                                .is_some_and(|rows| rows.iter().any(|row| row != node.dense()))
                        {
                            return Err(Error::new(
                                ErrorCode::TransactionConflict,
                                "node property uniqueness constraint was violated",
                            ));
                        }
                        index.insert_key(key, node.dense());
                    }
                }
                RuntimeIndex::Range(index) => {
                    if let Some(key) = composite_key(node, &entry.definition.properties)? {
                        index.insert_key(key, node.dense());
                    }
                }
                RuntimeIndex::Text(index) => {
                    let property = entry.definition.properties[0];
                    if let Some(ScalarValue::String(value)) = node.property(property) {
                        index.upsert(node.dense(), &value);
                    }
                }
                RuntimeIndex::Vector { .. } => {}
            }
        }
        Ok(())
    }
}

fn hash_bytes(hasher: &mut blake3::Hasher, bytes: &[u8]) {
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
}

const fn graph_index_kind_byte(kind: GraphIndexKind) -> u8 {
    match kind {
        GraphIndexKind::Equality => 0,
        GraphIndexKind::Range => 1,
        GraphIndexKind::Text => 2,
        GraphIndexKind::Vector => 3,
    }
}

const fn derived_index_state_byte(state: DerivedIndexState) -> u8 {
    match state {
        DerivedIndexState::Populating => 0,
        DerivedIndexState::Online => 1,
        DerivedIndexState::Failed => 2,
        DerivedIndexState::Dropping => 3,
    }
}

fn scalar_lookup_key(
    properties: &[PropertyId],
    values: &BTreeMap<PropertyId, ScalarValue>,
) -> Result<Option<IndexKey>> {
    let mut keys = Vec::with_capacity(properties.len());
    for property in properties {
        let value = values
            .get(property)
            .ok_or_else(|| Error::internal("index value disappeared"))?;
        if matches!(value, ScalarValue::Null) {
            return Ok(None);
        }
        keys.push(IndexKey::try_from(value)?);
    }
    Ok(Some(match keys.as_slice() {
        [value] => value.clone(),
        _ => IndexKey::Composite(keys),
    }))
}

fn scalar_posting_shape(postings: &Postings<IndexKey>) -> (u64, u64, u64, u64) {
    let values = postings.materialized();
    let total = values.values().map(RoaringBitmap::len).sum();
    let largest = values.values().map(RoaringBitmap::len).max().unwrap_or(0);
    let bytes = values
        .iter()
        .map(|(key, rows)| index_key_estimated_bytes(key) + rows.len() * 4)
        .sum();
    (values.len() as u64, total, largest, bytes)
}
fn text_posting_shape(index: &TextIndex) -> (u64, u64, u64, u64) {
    let values = index.materialized_postings();
    let total = values.values().map(RoaringBitmap::len).sum();
    let largest = values.values().map(RoaringBitmap::len).max().unwrap_or(0);
    let bytes = values
        .iter()
        .map(|(key, rows)| key.len() as u64 + rows.len() * 4)
        .sum();
    (values.len() as u64, total, largest, bytes)
}

fn index_key_estimated_bytes(key: &IndexKey) -> u64 {
    match key {
        IndexKey::Boolean(_) => 2,
        IndexKey::Integer(_) | IndexKey::Float(_) | IndexKey::LocalTime(_) => 9,
        IndexKey::Date(_) => 9,
        IndexKey::ZonedTime(_, _) => 13,
        IndexKey::LocalDateTime(_, _) => 13,
        IndexKey::ZonedDateTime(_, _, timezone) => 13_u64.saturating_add(timezone.len() as u64),
        IndexKey::Duration(_, _, _, _) => 29,
        IndexKey::String(value) => 9_u64.saturating_add(value.len() as u64),
        IndexKey::Bytes(value) => 9_u64.saturating_add(value.len() as u64),
        IndexKey::Composite(values) => values.iter().fold(9_u64, |bytes, value| {
            bytes.saturating_add(index_key_estimated_bytes(value))
        }),
    }
}

fn u32_len(value: usize, field: &str) -> Result<u32> {
    u32::try_from(value).map_err(|_| {
        Error::new(
            ErrorCode::ResultBudgetExceeded,
            format!("{field} exceeds the device image limit"),
        )
    })
}

fn posting_device_image(
    postings: &BTreeMap<IndexKey, RoaringBitmap>,
) -> Result<PostingDeviceImage> {
    let mut key_offsets = Vec::with_capacity(postings.len() + 1);
    let mut key_bytes = Vec::new();
    let mut posting_offsets = Vec::with_capacity(postings.len() + 1);
    let mut rows = Vec::new();
    key_offsets.push(0);
    posting_offsets.push(0);
    for (key, bitmap) in postings {
        let encoded = postcard::to_stdvec(key)
            .map_err(|error| Error::new(ErrorCode::CorruptStorage, error.to_string()))?;
        key_bytes.extend_from_slice(&encoded);
        key_offsets.push(u32_len(key_bytes.len(), "index key bytes")?);
        rows.extend(bitmap.iter());
        posting_offsets.push(u32_len(rows.len(), "index postings")?);
    }
    Ok(PostingDeviceImage {
        key_offsets,
        key_bytes,
        posting_offsets,
        rows,
    })
}

fn text_device_image(postings: &BTreeMap<String, RoaringBitmap>) -> Result<TextDeviceImage> {
    let mut term_offsets = Vec::with_capacity(postings.len() + 1);
    let mut term_bytes = Vec::new();
    let mut posting_offsets = Vec::with_capacity(postings.len() + 1);
    let mut rows = Vec::new();
    term_offsets.push(0);
    posting_offsets.push(0);
    for (term, bitmap) in postings {
        term_bytes.extend_from_slice(term.as_bytes());
        term_offsets.push(u32_len(term_bytes.len(), "text-index terms")?);
        rows.extend(bitmap.iter());
        posting_offsets.push(u32_len(rows.len(), "text-index postings")?);
    }
    Ok(TextDeviceImage {
        term_offsets,
        term_bytes,
        posting_offsets,
        rows,
    })
}

fn validate_definition(graph: &GraphStore, definition: &GraphIndexDefinition) -> Result<()> {
    if definition.name.is_empty() || definition.name.len() > 255 {
        return Err(Error::new(
            ErrorCode::QueryType,
            "index name must contain 1..=255 bytes",
        ));
    }
    if graph.catalog().label_name(definition.label).is_none()
        || definition.properties.is_empty()
        || definition
            .properties
            .iter()
            .any(|property| graph.catalog().property_name(*property).is_none())
    {
        return Err(Error::new(
            ErrorCode::QueryType,
            "index definition references unknown or empty schema targets",
        ));
    }
    if matches!(
        definition.kind,
        GraphIndexKind::Text | GraphIndexKind::Vector
    ) && definition.properties.len() != 1
    {
        return Err(Error::new(
            ErrorCode::QueryType,
            "text and vector indexes require exactly one property",
        ));
    }
    if definition.unique
        && (definition.kind != GraphIndexKind::Equality || definition.properties.len() != 1)
    {
        return Err(Error::new(
            ErrorCode::QueryType,
            "a uniqueness constraint requires exactly one equality-indexed property",
        ));
    }
    Ok(())
}

fn validate_unique_runtime(runtime: &RuntimeIndex) -> Result<()> {
    let RuntimeIndex::Equality(index) = runtime else {
        return Err(Error::new(
            ErrorCode::QueryType,
            "uniqueness constraints require an equality runtime",
        ));
    };
    if index.materialized().values().any(|rows| rows.len() > 1) {
        return Err(Error::new(
            ErrorCode::TransactionConflict,
            "existing data violates the requested uniqueness constraint",
        ));
    }
    Ok(())
}

fn composite_key(node: NodeView<'_>, properties: &[PropertyId]) -> Result<Option<IndexKey>> {
    let mut values = Vec::with_capacity(properties.len());
    for property in properties {
        let Some(value) = node.property(*property) else {
            return Ok(None);
        };
        if matches!(value, ScalarValue::Null) {
            return Ok(None);
        }
        values.push(IndexKey::try_from(&value)?);
    }
    Ok(match values.as_slice() {
        [value] => Some(value.clone()),
        _ => Some(IndexKey::Composite(values)),
    })
}

fn ivf_config(matrix: &VectorIndex) -> IvfPqConfig {
    ivf_config_for_shape(matrix.len(), matrix.dimension())
}

fn ivf_config_for_shape(row_count: usize, dimension: usize) -> IvfPqConfig {
    let mut config = IvfPqConfig::default();
    let rows = row_count.max(1);
    let (coarse_centroids, probes, candidate_budget, target_subquantizers) = match rows {
        0..=9_999 => (64, 8, 256, 8),
        10_000..=99_999 => (256, 12, 512, 12),
        100_000..=999_999 => (1_024, 24, 1_024, 16),
        1_000_000..=9_999_999 => (4_096, 48, 2_048, 32),
        10_000_000..=99_999_999 => (16_384, 96, 4_096, 48),
        _ => (u16::MAX as usize, 128, 8_192, 48),
    };
    config.coarse_centroids = coarse_centroids.min(rows).max(1);
    config.probes = probes.min(config.coarse_centroids).max(1);
    config.candidate_budget = candidate_budget.max(256);
    config.subquantizers = [64, 48, 32, 24, 16, 12, 8, 4, 2, 1]
        .into_iter()
        .filter(|candidate| *candidate <= target_subquantizers)
        .find(|candidate| *candidate <= dimension && dimension.is_multiple_of(*candidate))
        .unwrap_or(1);
    config
}

fn validate_ivf_config(source: &VectorIndex, config: IvfPqConfig) -> Result<()> {
    if config.size_class_version != IVF_PQ_SIZE_CLASS_VERSION
        || config.coarse_centroids == 0
        || config.coarse_centroids > u16::MAX as usize
        || config.subquantizers == 0
        || config.subquantizers > source.dimension
        || config.bits_per_code == 0
        || config.bits_per_code > 8
        || config.probes == 0
        || config.candidate_budget == 0
        || config.iterations == 0
    {
        return Err(Error::new(
            ErrorCode::IndexUnavailable,
            "invalid IVF-PQ configuration",
        ));
    }
    Ok(())
}

fn kmeans(
    data: &[Vec<f32>],
    count: usize,
    iterations: usize,
    seed: u64,
    kernel: &mut dyn IvfPqBuildKernel,
    cancellation: &CancellationToken,
) -> Result<Vec<Vec<f32>>> {
    let Some(first) = data.first() else {
        return Err(Error::new(
            ErrorCode::IndexUnavailable,
            "k-means training set is empty",
        ));
    };
    let dimension = first.len();
    if dimension == 0
        || data.iter().any(|row| row.len() != dimension)
        || count == 0
        || count > data.len()
    {
        return Err(Error::new(
            ErrorCode::IndexUnavailable,
            "k-means training dimensions are invalid",
        ));
    }
    let offset = (seed as usize) % data.len();
    let mut centers = (0..count)
        .map(|center| {
            let rank = center.saturating_mul(data.len()) / count;
            data[(offset + rank) % data.len()].clone()
        })
        .collect::<Vec<_>>();
    let mut assignment = vec![u32::MAX; data.len()];
    for _ in 0..iterations {
        check_ivf_build_cancelled(cancellation)?;
        let next = assign_nested(kernel, data, &centers)?;
        let changed = assignment != next;
        assignment = next;
        let mut sums = vec![vec![0.0_f64; dimension]; count];
        let mut counts = vec![0_u64; count];
        for (vector, cluster) in data.iter().zip(&assignment) {
            let cluster = *cluster as usize;
            counts[cluster] = counts[cluster].saturating_add(1);
            for (sum, value) in sums[cluster].iter_mut().zip(vector) {
                *sum += f64::from(*value);
            }
        }
        for cluster in 0..count {
            if counts[cluster] == 0 {
                continue;
            }
            for coordinate in 0..dimension {
                centers[cluster][coordinate] =
                    (sums[cluster][coordinate] / counts[cluster] as f64) as f32;
            }
        }
        if !changed {
            break;
        }
    }
    Ok(centers)
}

fn check_ivf_build_cancelled(cancellation: &CancellationToken) -> Result<()> {
    if cancellation.is_cancelled() {
        return Err(Error::new(
            ErrorCode::Cancelled,
            "IVF-PQ build was cancelled",
        ));
    }
    Ok(())
}

fn subspace_bounds(dimension: usize, subquantizers: usize, subspace: usize) -> (usize, usize) {
    let start = dimension.saturating_mul(subspace) / subquantizers;
    let end = dimension.saturating_mul(subspace.saturating_add(1)) / subquantizers;
    (start, end)
}

fn subtract(left: &[f32], right: &[f32]) -> Result<Vec<f32>> {
    if left.len() != right.len() {
        return Err(Error::new(
            ErrorCode::EmbeddingProfileMismatch,
            "vector dimensions differ",
        ));
    }
    Ok(left
        .iter()
        .zip(right)
        .map(|(left, right)| left - right)
        .collect())
}

fn l2_squared(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(left, right)| {
            let delta = left - right;
            delta * delta
        })
        .sum()
}

fn public_score(similarity: Similarity, left: &[f32], right: &[f32]) -> f32 {
    match similarity {
        Similarity::Euclidean => l2_squared(left, right).sqrt(),
        Similarity::Dot | Similarity::Cosine => left
            .iter()
            .zip(right)
            .map(|(left, right)| left * right)
            .sum(),
    }
}

fn sort_hits(hits: &mut [VectorHit], similarity: Similarity) {
    hits.sort_by(|left, right| {
        let score_order = match similarity {
            Similarity::Euclidean => total_f32(left.score, right.score),
            Similarity::Dot | Similarity::Cosine => total_f32(right.score, left.score),
        };
        score_order.then_with(|| left.entity_id.cmp(&right.entity_id))
    });
}

fn total_f32(left: f32, right: f32) -> Ordering {
    left.total_cmp(&right)
}

#[cfg(test)]
mod concurrent_tests {
    use super::*;
    use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};

    fn profile() -> Result<EmbeddingProfile> {
        EmbeddingProfile::new(
            [1; 32],
            [2; 32],
            4,
            EmbeddingDType::F16,
            false,
            Similarity::Dot,
        )
    }
    #[test]
    fn metadata_revision_advances_without_coordinates_or_ann_stamp_replacement() -> Result<()> {
        let source = VectorIndex::new(4, Similarity::Dot)?;
        source.upsert(1, &[1.0; 4], 10)?;
        let row = source.store.rows.get(&1).expect("owner");
        let before = source
            .store
            .slots
            .get(row)
            .expect("slot")
            .payload
            .load_full()
            .expect("payload");
        let dirty = source.store.dirty.get(&row);
        assert!(source.advance_row_revision(1, 10, 11));
        let after = source
            .store
            .slots
            .get(row)
            .expect("slot")
            .payload
            .load_full()
            .expect("payload");
        assert!(Arc::ptr_eq(&before, &after));
        assert_eq!(before.stamp, after.stamp);
        assert_eq!(source.store.dirty.get(&row), dirty);
        assert_eq!(source.row_revision(1), Some(11));
        assert!(!source.advance_row_revision(1, 12, 13));
        assert_eq!(source.row_revision(1), Some(11));
        source.remove(1, 14);
        source.upsert(2, &[2.0; 4], 1)?;
        assert!(!source.advance_row_revision(1, 11, 15));
        assert_eq!(source.row_revision(2), Some(1));
        Ok(())
    }
    fn fixture() -> Result<(GraphStore, LabelId, PropertyId, PropertyId)> {
        let graph = GraphStore::default();
        let label = graph.catalog().intern_label("Record")?;
        let body = graph.catalog().intern_property("body")?;
        let vector = graph.catalog().intern_property("vector")?;
        for id in 1..=4 {
            graph.insert_node(crate::NodeInput {
                id: crate::NodeId(id),
                layer: crate::Layer::Knowledge,
                revision: 1,
                labels: vec![label],
                properties: vec![(body, ScalarValue::String("complete document".into()))],
            })?;
        }
        Ok((graph, label, body, vector))
    }
    #[test]
    fn manual_vectors_serialize_owner_references_and_rebind_without_payload_copy()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let (graph, label, _, property) = fixture()?;
        let numeric = |value: f64| {
            irongraph_types::DocumentList::new(vec![
                irongraph_types::DocumentItem::Scalar(
                    ScalarValue::Float(value.into())
                );
                4
            ])
            .map(ScalarValue::List)
        };
        graph.set_node_property(crate::NodeId(1), property, numeric(1.0)?, 2)?;
        let catalog = IndexCatalog::default();
        catalog.activate_profile(profile()?)?;
        catalog.create(
            &graph,
            GraphIndexDefinition {
                name: "manual".into(),
                kind: GraphIndexKind::Vector,
                label,
                properties: vec![property],
                unique: false,
            },
        )?;
        let (source, _) = catalog
            .vector_search_source("manual")
            .ok_or("missing source")?;
        let payload = source
            .store
            .slots
            .get(0)
            .ok_or("missing slot")?
            .payload
            .load_full()
            .ok_or("missing payload")?;
        assert!(matches!(
            payload.coordinates,
            VectorCoordinates::PropertyOwner { .. }
        ));
        let bytes = postcard::to_stdvec(source.as_ref())?;
        assert!(
            bytes.len() < 37,
            "index checkpoint stores coordinate bytes: {}",
            bytes.len()
        );
        let restored: VectorIndex = postcard::from_bytes(&bytes)?;
        restored.bind_canonical_graph(&graph);
        assert_eq!(restored.vector_for(1), Some(vec![1.0; 4]));
        graph.set_node_property(crate::NodeId(1), property, numeric(2.0)?, 3)?;
        assert_eq!(restored.vector_for(1), Some(vec![2.0; 4]));
        assert_eq!(source.vector_for(1), Some(vec![2.0; 4]));
        Ok(())
    }

    #[test]
    fn invalid_final_unique_write_is_pure_and_released_keys_can_be_reused() -> Result<()> {
        let (graph, label, property, _) = fixture()?;
        for owner in 1..=4 {
            graph.set_node_property(
                crate::NodeId(owner),
                property,
                ScalarValue::Integer(owner as i64),
                2,
            )?;
        }
        let catalog = IndexCatalog::default();
        catalog.create(
            &graph,
            GraphIndexDefinition {
                name: "unique".into(),
                kind: GraphIndexKind::Equality,
                label,
                properties: vec![property],
                unique: true,
            },
        )?;
        let write = |owner, value| GraphMutation::SetNodeProperty {
            node: crate::NodeId(owner),
            property,
            value: ScalarValue::Integer(value),
            revision: 3,
        };
        assert!(
            catalog
                .validate_graph_mutations(&graph, &[write(1, 10), write(2, 10)])
                .is_err()
        );
        assert_eq!(
            graph
                .node(crate::NodeId(1))
                .and_then(|node| node.property(property)),
            Some(ScalarValue::Integer(1))
        );
        assert_eq!(
            graph
                .node(crate::NodeId(2))
                .and_then(|node| node.property(property)),
            Some(ScalarValue::Integer(2))
        );
        catalog.validate_graph_mutations(&graph, &[write(1, 10), write(2, 1)])?;
        Ok(())
    }

    #[test]
    fn lifecycle_edit_never_resurrects_dropped_definition() -> Result<()> {
        let (graph, label, body, _) = fixture()?;
        let catalog = IndexCatalog::default();
        catalog.create(
            &graph,
            GraphIndexDefinition {
                name: "text".into(),
                kind: GraphIndexKind::Text,
                label,
                properties: vec![body],
                unique: false,
            },
        )?;
        let mut edit = catalog
            .entries
            .edit("text")
            .ok_or_else(|| Error::internal("missing definition"))?;
        catalog.entries.remove("text");
        edit.state = DerivedIndexState::Failed;
        drop(edit);
        assert!(!catalog.contains("text"));
        Ok(())
    }

    #[test]
    fn scalar_text_clone_shares_canonical_memberships() -> Result<()> {
        let equality = EqualityIndex::default();
        let alias = equality.clone();
        equality.insert(&ScalarValue::Integer(7), 3)?;
        assert_eq!(
            alias.get(&IndexKey::Integer(7)).map(|rows| rows.len()),
            Some(1)
        );
        alias.remove_key(&IndexKey::Integer(7), 3);
        assert!(equality.get(&IndexKey::Integer(7)).is_none());
        let range = RangeIndex::default();
        range.insert(&ScalarValue::Integer(5), 2)?;
        range.insert(&ScalarValue::Integer(9), 4)?;
        assert_eq!(
            range
                .between(Some((&IndexKey::Integer(5), false)), None)
                .iter()
                .collect::<Vec<_>>(),
            vec![4]
        );
        let text = TextIndex::default();
        text.upsert(2, "complete long owner text");
        let alias = text.clone();
        alias.upsert(2, "changed long owner text");
        assert!(text.search("complete").is_empty());
        assert_eq!(
            text.search("changed owner").iter().collect::<Vec<_>>(),
            vec![2]
        );
        Ok(())
    }
    #[test]
    fn vector_slot_churn_keeps_metadata_bounded_and_owner_guards_valid() -> Result<()> {
        let source = VectorIndex::new(4, Similarity::Dot)?;
        for owner in 0..16 {
            source.upsert(owner, &[1.0; 4], 1)?;
        }
        let held = source
            .store
            .slots
            .get(0)
            .ok_or_else(|| Error::internal("missing slot"))?
            .payload
            .load_full()
            .ok_or_else(|| Error::internal("missing payload"))?;
        for step in 0..4096_u64 {
            source.remove(step, 1);
            source.upsert(step + 16, &[2.0; 4], 1)?;
            assert_eq!(source.row_count(), 16);
            assert_eq!(source.store.rows.len(), 16);
            assert!(source.store.dirty.len() <= 16);
            assert_eq!(source.vector_for(step), None);
        }
        assert_eq!(held.entity_id, 0);
        assert_eq!(source.decode_payload(&held), Some(vec![1.0; 4]));
        assert_eq!(source.row_revision(4111), Some(1));
        Ok(())
    }

    #[test]
    fn readers_and_vector_writers_progress_with_held_old_payload() -> Result<()> {
        let vector = VectorIndex::new(384, Similarity::Dot)?;
        vector.upsert(1, &vec![1.0; 384], 1)?;
        let slot = vector
            .store
            .slots
            .get(0)
            .ok_or_else(|| Error::internal("missing test slot"))?;
        let held = slot
            .payload
            .load_full()
            .ok_or_else(|| Error::internal("missing test payload"))?;
        let retired = Arc::downgrade(&held);
        let complete = AtomicBool::new(false);
        std::thread::scope(|scope| -> Result<()> {
            let writer = scope.spawn(|| -> Result<()> {
                for revision in 2..=2000 {
                    vector.upsert(1, &vec![revision as f32 / 2000.0; 384], revision)?;
                }
                complete.store(true, AtomicOrdering::Release);
                Ok(())
            });
            while !complete.load(AtomicOrdering::Acquire) {
                let coordinates = vector
                    .vector_for(1)
                    .ok_or_else(|| Error::internal("live owner disappeared"))?;
                assert_eq!(coordinates.len(), 384);
                assert!(coordinates.iter().all(|v| v.is_finite()));
                assert!(coordinates.windows(2).all(|pair| pair[0] == pair[1]));
            }
            writer
                .join()
                .map_err(|_| Error::internal("test writer panicked"))??;
            Ok(())
        })?;
        assert_eq!(vector.row_count(), 1);
        assert_eq!(vector.store.dirty.len(), 1);
        assert!(
            matches!(&held.coordinates,VectorCoordinates::Quantized(values) if values.len()==384)
        );
        drop(held);
        assert!(
            retired.upgrade().is_none(),
            "retired payload was retained without a reader"
        );
        Ok(())
    }
    #[test]
    fn scalar_text_reads_remain_structurally_valid_during_writes() -> Result<()> {
        let equality = EqualityIndex::default();
        let text = TextIndex::default();
        let done = AtomicBool::new(false);
        std::thread::scope(|scope| -> Result<()> {
            let writer = scope.spawn(|| {
                for _ in 0..3000 {
                    equality.insert_key(IndexKey::Integer(1), 8);
                    text.upsert(8, "old text");
                    equality.remove_key(&IndexKey::Integer(1), 8);
                    text.upsert(8, "new text");
                }
                done.store(true, AtomicOrdering::Release);
            });
            while !done.load(AtomicOrdering::Acquire) {
                if let Some(rows) = equality.get(&IndexKey::Integer(1)) {
                    assert!(rows.iter().all(|row| row == 8));
                }
                assert!(text.search("text").iter().all(|row| row == 8));
            }
            writer
                .join()
                .map_err(|_| Error::internal("test writer panicked"))?;
            Ok(())
        })
    }
    #[test]
    fn invalid_final_vector_mutation_leaves_every_payload_unchanged() -> Result<()> {
        let catalog = IndexCatalog::default();
        catalog.initialize_semantic(profile()?)?;
        let initial = ResolvedVectorMutation::Upsert {
            property: SEMANTIC_NODE_PROPERTY,
            entity_id: 1,
            coordinates: vec![f16::from_f32(1.0).to_bits(); 4],
            revision: 1,
        };
        catalog.apply_vector_mutation(&initial)?;
        let alias = catalog.clone();
        let invalid = [
            ResolvedVectorMutation::Upsert {
                property: SEMANTIC_NODE_PROPERTY,
                entity_id: 1,
                coordinates: vec![f16::from_f32(2.0).to_bits(); 4],
                revision: 2,
            },
            ResolvedVectorMutation::Upsert {
                property: SEMANTIC_NODE_PROPERTY,
                entity_id: 2,
                coordinates: vec![0; 3],
                revision: 2,
            },
        ];
        assert!(catalog.validate_vector_mutations(&invalid).is_err());
        assert!(catalog.apply_vector_mutations(&invalid).is_err());
        let (source, _) = alias
            .vector_search_source(SEMANTIC_NODE_INDEX)
            .ok_or_else(|| Error::internal("missing source"))?;
        assert_eq!(source.vector_for(1), Some(vec![1.0; 4]));
        assert!(source.vector_for(2).is_none());
        assert!(Arc::ptr_eq(
            &source,
            &catalog
                .vector_search_source(SEMANTIC_NODE_INDEX)
                .ok_or_else(|| Error::internal("missing source"))?
                .0
        ));
        Ok(())
    }
    #[test]
    fn invalid_embedding_declaration_does_not_publish_profile_or_owners() -> Result<()> {
        let (graph, label, body, vector) = fixture()?;
        let catalog = IndexCatalog::default();
        let definition = EmbeddingIndexDefinition {
            name: "documents".into(),
            label,
            source_property: body,
            target_property: vector,
            model: "default".into(),
        };
        let rows = vec![(1, vec![0; 4], 1), (2, vec![0; 3], 1)];
        assert!(
            catalog
                .create_embedding_deferred(&graph, definition, profile()?, rows)
                .is_err()
        );
        assert!(catalog.is_empty());
        assert!(catalog.profile().is_none());
        Ok(())
    }
    #[test]
    fn catalog_postcard_roundtrip_preserves_scalar_text_vectors_and_deltas()
    -> std::result::Result<(), Box<dyn std::error::Error>> {
        let (graph, label, body, _) = fixture()?;
        let catalog = IndexCatalog::default();
        for (name, kind) in [
            ("equality", GraphIndexKind::Equality),
            ("range", GraphIndexKind::Range),
            ("text", GraphIndexKind::Text),
        ] {
            catalog.create(
                &graph,
                GraphIndexDefinition {
                    name: name.into(),
                    kind,
                    label,
                    properties: vec![body],
                    unique: false,
                },
            )?;
        }
        catalog.initialize_semantic(profile()?)?;
        catalog.apply_vector_mutation(&ResolvedVectorMutation::Upsert {
            property: SEMANTIC_NODE_PROPERTY,
            entity_id: 1,
            coordinates: vec![f16::from_f32(1.0).to_bits(); 4],
            revision: 2,
        })?;
        let bytes = postcard::to_stdvec(&catalog)?;
        let restored: IndexCatalog = postcard::from_bytes(&bytes)?;
        assert_eq!(restored.definitions().count(), 5);
        let values = BTreeMap::from([(body, ScalarValue::String("complete document".into()))]);
        assert_eq!(
            restored
                .equality_candidates_named("equality", label, &values)?
                .map(|rows| rows.len()),
            Some(4)
        );
        assert_eq!(restored.bounded_text_candidates("document", 10)?.len(), 4);
        let (source, _) = restored
            .vector_search_source(SEMANTIC_NODE_INDEX)
            .ok_or("missing source")?;
        assert_eq!(source.vector_for(1), Some(vec![1.0; 4]));
        assert_eq!(source.store.dirty.len(), 1);
        Ok(())
    }
    #[test]
    fn ann_cold_build_and_single_row_delta_do_not_gate_search() -> Result<()> {
        let source = VectorIndex::new(4, Similarity::Dot)?;
        for owner in 0..4096 {
            source.upsert(owner, &[0.1, 0.2, (owner % 7) as f32 / 10.0, 0.3], 1)?;
        }
        let config = IvfPqConfig {
            coarse_centroids: 4,
            subquantizers: 2,
            bits_per_code: 2,
            probes: 2,
            candidate_budget: 32,
            iterations: 2,
            ..IvfPqConfig::default()
        };
        let ann = IvfPqIndex::build(&source, config)?;
        source.acknowledge_ann(&ann);
        assert_eq!(source.store.dirty.len(), 0);
        source.upsert(100, &[0.0, 0.0, 100.0, 0.0], 2)?;
        assert_eq!(source.store.dirty.len(), 1);
        assert_eq!(
            ann.search(&source, &[0.0, 0.0, 1.0, 0.0], 1)?
                .first()
                .map(|hit| hit.entity_id),
            Some(100)
        );
        assert_eq!(source.row_count(), 4096);
        source.remove(100, 3);
        assert!(
            ann.search(&source, &[0.0, 0.0, 1.0, 0.0], 32)?
                .iter()
                .all(|hit| hit.entity_id != 100)
        );
        source.upsert(10_000, &[0.0, 0.0, 200.0, 0.0], 1)?;
        assert_eq!(source.row_count(), 4096);
        assert_eq!(
            ann.search(&source, &[0.0, 0.0, 1.0, 0.0], 1)?
                .first()
                .map(|hit| hit.entity_id),
            Some(10_000)
        );
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        assert!(IvfPqIndex::build_cancellable(&source, config, &cancellation).is_err());
        Ok(())
    }
    #[test]
    fn bounded_text_delta_replaces_only_changed_large_owner() -> Result<()> {
        let text = TextIndex::default();
        let body = "unchanged token ".repeat(4096);
        for row in 0..4096 {
            text.upsert(row, &body);
        }
        let held = text
            .rows
            .get(&99)
            .ok_or_else(|| Error::internal("missing owner terms"))?;
        let retired = Arc::downgrade(&held);
        text.upsert(99, "replacement token");
        drop(held);
        // Papaya may defer retired map values to its bounded reclamation batch.
        for row in 5000..6000 {
            text.upsert(row, "pressure");
            text.remove(row);
        }
        assert_eq!(
            text.search("replacement").iter().collect::<Vec<_>>(),
            vec![99]
        );
        assert!(!text.search("unchanged").contains(99));
        assert_eq!(text.search("unchanged").len(), 4095);
        assert!(text.bounded_ranked_search("token", 10).len() <= 10);
        assert!(retired.strong_count() <= 1);
        Ok(())
    }
}
