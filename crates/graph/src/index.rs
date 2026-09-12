//! Rebuildable equality, range, text, exact-vector, and deterministic IVF-PQ indexes.

use std::{
    cmp::Ordering,
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, OnceLock},
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

use super::{
    GraphMutation, GraphStore, NodeView,
    persistent::{PagedVec, PersistentMap},
};

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

/// Equality postings over dense row ordinals.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct EqualityIndex {
    postings: Arc<BTreeMap<IndexKey, RoaringBitmap>>,
    #[serde(default)]
    deltas: PersistentMap<Vec<PostingDelta>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PostingDelta {
    key: IndexKey,
    changes: PersistentMap<bool>,
}

fn index_key_hash(key: &IndexKey) -> u128 {
    fn write_key(hasher: &mut blake3::Hasher, key: &IndexKey) {
        match key {
            IndexKey::Boolean(value) => {
                hasher.update(&[0, u8::from(*value)]);
            }
            IndexKey::Integer(value) => {
                hasher.update(&[1]);
                hasher.update(&value.to_le_bytes());
            }
            IndexKey::Float(value) => {
                hasher.update(&[2]);
                hasher.update(&value.to_bits().to_le_bytes());
            }
            IndexKey::String(value) => {
                hasher.update(&[3]);
                hasher.update(value.as_bytes());
            }
            IndexKey::Bytes(value) => {
                hasher.update(&[4]);
                hasher.update(value);
            }
            IndexKey::Date(value) => {
                hasher.update(&[5]);
                hasher.update(&value.to_le_bytes());
            }
            IndexKey::LocalTime(value) => {
                hasher.update(&[6]);
                hasher.update(&value.to_le_bytes());
            }
            IndexKey::ZonedTime(nanos, offset) => {
                hasher.update(&[7]);
                hasher.update(&nanos.to_le_bytes());
                hasher.update(&offset.to_le_bytes());
            }
            IndexKey::LocalDateTime(seconds, nanos) => {
                hasher.update(&[8]);
                hasher.update(&seconds.to_le_bytes());
                hasher.update(&nanos.to_le_bytes());
            }
            IndexKey::ZonedDateTime(seconds, nanos, timezone) => {
                hasher.update(&[9]);
                hasher.update(&seconds.to_le_bytes());
                hasher.update(&nanos.to_le_bytes());
                hasher.update(timezone.as_bytes());
            }
            IndexKey::Duration(months, days, seconds, nanos) => {
                hasher.update(&[10]);
                hasher.update(&months.to_le_bytes());
                hasher.update(&days.to_le_bytes());
                hasher.update(&seconds.to_le_bytes());
                hasher.update(&nanos.to_le_bytes());
            }
            IndexKey::Composite(values) => {
                hasher.update(&[11]);
                for value in values {
                    write_key(hasher, value);
                }
            }
        }
    }
    let mut hasher = blake3::Hasher::new();
    write_key(&mut hasher, key);
    let bytes = hasher.finalize();
    let mut prefix = [0_u8; 16];
    prefix.copy_from_slice(&bytes.as_bytes()[..16]);
    u128::from_le_bytes(prefix)
}

fn append_posting_delta(
    deltas: &mut PersistentMap<Vec<PostingDelta>>,
    key: IndexKey,
    row: u32,
    present: bool,
) {
    let hash = index_key_hash(&key);
    let mut bucket = deltas.get(hash).cloned().unwrap_or_default();
    if let Some(delta) = bucket.iter_mut().find(|delta| delta.key == key) {
        delta.changes.insert(u128::from(row), present);
    } else {
        let mut changes = PersistentMap::default();
        changes.insert(u128::from(row), present);
        bucket.push(PostingDelta { key, changes });
    }
    deltas.insert(hash, bucket);
}

fn materialize_postings(
    base: &BTreeMap<IndexKey, RoaringBitmap>,
    deltas: &PersistentMap<Vec<PostingDelta>>,
) -> BTreeMap<IndexKey, RoaringBitmap> {
    let mut postings = base.clone();
    for (_, bucket) in deltas.iter() {
        for delta in bucket {
            let rows = postings.entry(delta.key.clone()).or_default();
            for (row, present) in delta.changes.iter() {
                let Ok(row) = u32::try_from(row) else {
                    continue;
                };
                if *present {
                    rows.insert(row);
                } else {
                    rows.remove(row);
                }
            }
            if rows.is_empty() {
                postings.remove(&delta.key);
            }
        }
    }
    postings
}

impl EqualityIndex {
    pub fn insert(&mut self, value: &ScalarValue, row: u32) -> Result<()> {
        if !matches!(value, ScalarValue::Null) {
            self.insert_key(IndexKey::try_from(value)?, row);
        }
        Ok(())
    }

    pub fn insert_key(&mut self, key: IndexKey, row: u32) {
        append_posting_delta(&mut self.deltas, key, row, true);
    }

    pub fn remove(&mut self, value: &ScalarValue, row: u32) -> Result<()> {
        if matches!(value, ScalarValue::Null) {
            return Ok(());
        }
        self.remove_key(&IndexKey::try_from(value)?, row);
        Ok(())
    }

    pub fn remove_key(&mut self, key: &IndexKey, row: u32) {
        append_posting_delta(&mut self.deltas, key.clone(), row, false);
    }

    #[must_use]
    pub fn get(&self, key: &IndexKey) -> Option<RoaringBitmap> {
        let mut rows = self.postings.get(key).cloned().unwrap_or_default();
        if let Some(bucket) = self.deltas.get(index_key_hash(key))
            && let Some(delta) = bucket.iter().find(|delta| &delta.key == key)
        {
            apply_changes(&mut rows, &delta.changes);
        }
        (!rows.is_empty()).then_some(rows)
    }

    fn get_bounded(&self, key: &IndexKey, limit: usize) -> Vec<u32> {
        let changes = self
            .deltas
            .get(index_key_hash(key))
            .and_then(|bucket| bucket.iter().find(|delta| &delta.key == key))
            .map(|delta| &delta.changes);
        bounded_bitmap_with_changes(self.postings.get(key), changes, limit)
    }

    fn seal(&mut self) {
        self.postings = Arc::new(materialize_postings(&self.postings, &self.deltas));
        self.deltas = PersistentMap::default();
    }

    fn materialized(&self) -> BTreeMap<IndexKey, RoaringBitmap> {
        materialize_postings(&self.postings, &self.deltas)
    }

    #[cfg(test)]
    fn detached_storage_bytes_from(&self, previous: &Self) -> usize {
        let empty = PersistentMap::default();
        let mut bytes = self.deltas.detached_node_bytes_from(&previous.deltas);
        for (hash, bucket) in self.deltas.iter() {
            let old_bucket = previous.deltas.get(hash);
            for delta in bucket {
                let old = old_bucket
                    .and_then(|bucket| bucket.iter().find(|old| old.key == delta.key))
                    .map_or(&empty, |old| &old.changes);
                bytes = bytes.saturating_add(delta.changes.detached_node_bytes_from(old));
            }
        }
        bytes
    }
}

/// Sorted scalar range postings with inclusive/exclusive endpoints.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RangeIndex {
    postings: Arc<BTreeMap<IndexKey, RoaringBitmap>>,
    #[serde(default)]
    deltas: PersistentMap<Vec<PostingDelta>>,
}

impl RangeIndex {
    pub fn insert(&mut self, value: &ScalarValue, row: u32) -> Result<()> {
        if !matches!(value, ScalarValue::Null) {
            self.insert_key(IndexKey::try_from(value)?, row);
        }
        Ok(())
    }

    pub fn insert_key(&mut self, key: IndexKey, row: u32) {
        append_posting_delta(&mut self.deltas, key, row, true);
    }

    pub fn remove_key(&mut self, key: &IndexKey, row: u32) {
        append_posting_delta(&mut self.deltas, key.clone(), row, false);
    }

    fn exact_bounded(&self, key: &IndexKey, limit: usize) -> Vec<u32> {
        let changes = self
            .deltas
            .get(index_key_hash(key))
            .and_then(|bucket| bucket.iter().find(|delta| &delta.key == key))
            .map(|delta| &delta.changes);
        bounded_bitmap_with_changes(self.postings.get(key), changes, limit)
    }

    pub fn between(
        &self,
        lower: Option<(&IndexKey, bool)>,
        upper: Option<(&IndexKey, bool)>,
    ) -> RoaringBitmap {
        fn in_bounds(
            key: &IndexKey,
            lower: Option<(&IndexKey, bool)>,
            upper: Option<(&IndexKey, bool)>,
        ) -> bool {
            let above_lower = lower.is_none_or(
                |(bound, inclusive)| {
                    if inclusive { key >= bound } else { key > bound }
                },
            );
            let below_upper = upper.is_none_or(
                |(bound, inclusive)| {
                    if inclusive { key <= bound } else { key < bound }
                },
            );
            above_lower && below_upper
        }

        let mut result = RoaringBitmap::new();
        for (key, base_rows) in self.postings.iter() {
            if in_bounds(key, lower, upper) {
                let mut rows = base_rows.clone();
                apply_posting_delta(&mut rows, &self.deltas, key);
                result |= rows;
            }
        }
        for (_, bucket) in self.deltas.iter() {
            for delta in bucket {
                if !self.postings.contains_key(&delta.key) && in_bounds(&delta.key, lower, upper) {
                    let mut rows = RoaringBitmap::new();
                    apply_changes(&mut rows, &delta.changes);
                    result |= rows;
                }
            }
        }
        result
    }

    fn seal(&mut self) {
        self.postings = Arc::new(materialize_postings(&self.postings, &self.deltas));
        self.deltas = PersistentMap::default();
    }

    fn materialized(&self) -> BTreeMap<IndexKey, RoaringBitmap> {
        materialize_postings(&self.postings, &self.deltas)
    }
}

fn apply_changes(rows: &mut RoaringBitmap, changes: &PersistentMap<bool>) {
    for (row, present) in changes.iter() {
        let Ok(row) = u32::try_from(row) else {
            continue;
        };
        if *present {
            rows.insert(row);
        } else {
            rows.remove(row);
        }
    }
}

fn bounded_bitmap_with_changes(
    base: Option<&RoaringBitmap>,
    changes: Option<&PersistentMap<bool>>,
    limit: usize,
) -> Vec<u32> {
    let mut base = base.into_iter().flat_map(RoaringBitmap::iter).peekable();
    let mut changes = changes
        .into_iter()
        .flat_map(PersistentMap::iter)
        .filter_map(|(row, present)| u32::try_from(row).ok().map(|row| (row, *present)))
        .peekable();
    let mut rows = Vec::with_capacity(limit.min(4096));
    while rows.len() < limit {
        let next = match (base.peek().copied(), changes.peek().copied()) {
            (Some(base_row), Some((changed_row, _))) if base_row < changed_row => {
                base.next();
                Some(base_row)
            }
            (Some(base_row), Some((changed_row, present))) if changed_row < base_row => {
                changes.next();
                present.then_some(changed_row)
            }
            (Some(base_row), Some((_, present))) => {
                base.next();
                changes.next();
                present.then_some(base_row)
            }
            (Some(base_row), None) => {
                base.next();
                Some(base_row)
            }
            (None, Some((changed_row, present))) => {
                changes.next();
                present.then_some(changed_row)
            }
            (None, None) => break,
        };
        if let Some(row) = next {
            rows.push(row);
        }
    }
    rows
}

fn apply_posting_delta(
    rows: &mut RoaringBitmap,
    deltas: &PersistentMap<Vec<PostingDelta>>,
    key: &IndexKey,
) {
    if let Some(bucket) = deltas.get(index_key_hash(key))
        && let Some(delta) = bucket.iter().find(|delta| &delta.key == key)
    {
        apply_changes(rows, &delta.changes);
    }
}

/// Declared document text index using normalized token postings.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct TextIndex {
    postings: Arc<BTreeMap<String, RoaringBitmap>>,
    rows: Arc<BTreeMap<u32, BTreeSet<String>>>,
    #[serde(default)]
    posting_deltas: PersistentMap<Vec<TextPostingDelta>>,
    #[serde(default)]
    row_overrides: PersistentMap<Option<BTreeSet<String>>>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct TextPostingDelta {
    term: String,
    changes: PersistentMap<bool>,
}

fn text_term_hash(term: &str) -> u128 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"text-posting\0");
    hasher.update(term.as_bytes());
    let digest = hasher.finalize();
    let mut prefix = [0_u8; 16];
    prefix.copy_from_slice(&digest.as_bytes()[..16]);
    u128::from_le_bytes(prefix)
}

fn append_text_delta(
    deltas: &mut PersistentMap<Vec<TextPostingDelta>>,
    term: String,
    row: u32,
    present: bool,
) {
    let hash = text_term_hash(&term);
    let mut bucket = deltas.get(hash).cloned().unwrap_or_default();
    if let Some(delta) = bucket.iter_mut().find(|delta| delta.term == term) {
        delta.changes.insert(u128::from(row), present);
    } else {
        let mut changes = PersistentMap::default();
        changes.insert(u128::from(row), present);
        bucket.push(TextPostingDelta { term, changes });
    }
    deltas.insert(hash, bucket);
}

impl TextIndex {
    pub fn upsert(&mut self, row: u32, text: &str) {
        self.remove(row);
        let terms = tokenize(text);
        for term in &terms {
            append_text_delta(&mut self.posting_deltas, term.clone(), row, true);
        }
        self.row_overrides.insert(u128::from(row), Some(terms));
    }

    pub fn remove(&mut self, row: u32) {
        let terms = match self.row_overrides.get(u128::from(row)) {
            Some(terms) => terms.clone(),
            None => self.rows.get(&row).cloned(),
        };
        let Some(terms) = terms else {
            return;
        };
        for term in terms {
            append_text_delta(&mut self.posting_deltas, term, row, false);
        }
        self.row_overrides.insert(u128::from(row), None);
    }

    /// AND search: every normalized query term must be present.
    #[must_use]
    pub fn search(&self, query: &str) -> RoaringBitmap {
        let terms = tokenize(query);
        let mut iter = terms.iter();
        let Some(first) = iter.next() else {
            return RoaringBitmap::new();
        };
        let Some(mut result) = self.posting(first) else {
            return RoaringBitmap::new();
        };
        for term in iter {
            let Some(rows) = self.posting(term) else {
                return RoaringBitmap::new();
            };
            result &= &rows;
        }
        result
    }

    /// Returns a deterministic bounded OR-ranked seed set. Work is capped by query terms and
    /// posting rows so semantic retrieval never materializes a project-sized posting union.
    fn bounded_ranked_search(&self, query: &str, limit: usize) -> Vec<(u32, u16)> {
        if limit == 0 {
            return Vec::new();
        }
        let row_budget = limit.saturating_mul(4).max(limit);
        let mut scores = BTreeMap::<u32, u16>::new();
        for term in tokenize(query).into_iter().take(32) {
            for row in self.bounded_posting_rows(&term, row_budget) {
                scores
                    .entry(row)
                    .and_modify(|score| *score = score.saturating_add(1))
                    .or_insert(1);
            }
        }
        let mut ranked = scores.into_iter().collect::<Vec<_>>();
        ranked.sort_unstable_by(|left, right| {
            right.1.cmp(&left.1).then_with(|| left.0.cmp(&right.0))
        });
        ranked.truncate(limit);
        ranked
    }

    fn bounded_posting_rows(&self, term: &str, limit: usize) -> Vec<u32> {
        let changes = self
            .posting_deltas
            .get(text_term_hash(term))
            .and_then(|bucket| bucket.iter().find(|delta| delta.term == term))
            .map(|delta| &delta.changes);
        bounded_bitmap_with_changes(self.postings.get(term), changes, limit)
    }

    fn posting(&self, term: &str) -> Option<RoaringBitmap> {
        let mut rows = self.postings.get(term).cloned().unwrap_or_default();
        if let Some(bucket) = self.posting_deltas.get(text_term_hash(term))
            && let Some(delta) = bucket.iter().find(|delta| delta.term == term)
        {
            apply_changes(&mut rows, &delta.changes);
        }
        (!rows.is_empty()).then_some(rows)
    }

    fn materialized_postings(&self) -> BTreeMap<String, RoaringBitmap> {
        let mut postings = self.postings.as_ref().clone();
        for (_, bucket) in self.posting_deltas.iter() {
            for delta in bucket {
                let rows = postings.entry(delta.term.clone()).or_default();
                apply_changes(rows, &delta.changes);
                if rows.is_empty() {
                    postings.remove(&delta.term);
                }
            }
        }
        postings
    }

    fn materialized_rows(&self) -> BTreeMap<u32, BTreeSet<String>> {
        let mut rows = self.rows.as_ref().clone();
        for (row, terms) in self.row_overrides.iter() {
            let Ok(row) = u32::try_from(row) else {
                continue;
            };
            match terms {
                Some(terms) => {
                    rows.insert(row, terms.clone());
                }
                None => {
                    rows.remove(&row);
                }
            }
        }
        rows
    }

    fn seal(&mut self) {
        self.postings = Arc::new(self.materialized_postings());
        self.rows = Arc::new(self.materialized_rows());
        self.posting_deltas = PersistentMap::default();
        self.row_overrides = PersistentMap::default();
    }

    #[cfg(test)]
    fn detached_storage_bytes_from(&self, previous: &Self) -> usize {
        let empty = PersistentMap::default();
        let mut bytes = self
            .posting_deltas
            .detached_node_bytes_from(&previous.posting_deltas)
            .saturating_add(
                self.row_overrides
                    .detached_node_bytes_from(&previous.row_overrides),
            );
        for (hash, bucket) in self.posting_deltas.iter() {
            let old_bucket = previous.posting_deltas.get(hash);
            for delta in bucket {
                let old = old_bucket
                    .and_then(|bucket| bucket.iter().find(|old| old.term == delta.term))
                    .map_or(&empty, |old| &old.changes);
                bytes = bytes.saturating_add(delta.changes.detached_node_bytes_from(old));
            }
        }
        bytes
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

/// Contiguous vector matrix with stable identities and per-row revisions.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct VectorIndex {
    dimension: usize,
    similarity: Similarity,
    dtype: EmbeddingDType,
    entity_ids: PagedVec<u64>,
    /// Canonical IEEE 754 half/bfloat16 bit patterns, quantized by the sequencer.
    values: PagedVec<u16>,
    versions: PagedVec<u64>,
    active: PagedVec<bool>,
    rows: PersistentMap<usize>,
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
    /// Recall measured against deterministic exact top-k queries before publication.
    pub recall_basis_points: u16,
    pub validation_queries: u16,
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
    fn rebind_shared(&mut self, backing: SharedVectorBacking) -> Result<()> {
        if self.dimension != backing.dimension
            || self.entity_ids.len() != backing.entity_ids.len()
            || self.values.len() != backing.values.len()
            || self.versions.len() != backing.versions.len()
            || self.active.len() != backing.active.len()
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "shared vector backing does not match the canonical vector column",
            ));
        }
        self.entity_ids = backing.entity_ids;
        self.values = backing.values;
        self.versions = backing.versions;
        self.active = backing.active;
        Ok(())
    }

    fn device_image(&self, property: PropertyId) -> Result<VectorDeviceImage> {
        if self.values.len()
            != self
                .entity_ids
                .len()
                .checked_mul(self.dimension)
                .ok_or_else(|| Error::new(ErrorCode::CorruptStorage, "vector shape overflow"))?
            || self.versions.len() != self.entity_ids.len()
            || self.active.len() != self.entity_ids.len()
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "canonical vector matrix is structurally inconsistent",
            ));
        }
        Ok(VectorDeviceImage {
            property,
            dimension: self.dimension,
            similarity: self.similarity,
            dtype: self.dtype,
            entity_ids: self.entity_ids.to_vec(),
            values: self.values.to_vec(),
            versions: self.versions.to_vec(),
            active: self.active.iter().map(|active| u8::from(*active)).collect(),
        })
    }

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
            entity_ids: PagedVec::default(),
            values: PagedVec::default(),
            versions: PagedVec::default(),
            active: PagedVec::default(),
            rows: PersistentMap::default(),
        })
    }

    pub fn upsert(&mut self, entity_id: u64, vector: &[f32], revision: u64) -> Result<()> {
        let vector = self.quantize(vector)?;
        self.upsert_quantized(entity_id, &vector, revision)
    }

    pub fn upsert_quantized(
        &mut self,
        entity_id: u64,
        coordinates: &[u16],
        revision: u64,
    ) -> Result<()> {
        if coordinates.len() != self.dimension
            || coordinates
                .iter()
                .map(|bits| decode_coordinate(self.dtype, *bits))
                .any(|value| !value.is_finite())
        {
            return Err(Error::new(
                ErrorCode::EmbeddingProfileMismatch,
                "quantized vector is incompatible with the embedding profile",
            ));
        }
        if let Some(row) = self.rows.get(u128::from(entity_id)).copied() {
            let start = row.checked_mul(self.dimension).ok_or_else(|| {
                Error::new(ErrorCode::ResultBudgetExceeded, "vector offset overflow")
            })?;
            let end = start.checked_add(self.dimension).ok_or_else(|| {
                Error::new(ErrorCode::ResultBudgetExceeded, "vector offset overflow")
            })?;
            if end > self.values.len() {
                return Err(Error::internal("vector row points outside its matrix"));
            }
            for (offset, coordinate) in coordinates.iter().copied().enumerate() {
                let target = self
                    .values
                    .get_mut(start + offset)
                    .ok_or_else(|| Error::internal("vector row points outside its matrix"))?;
                *target = coordinate;
            }
            if let Some(version) = self.versions.get_mut(row) {
                *version = revision;
            }
            if let Some(active) = self.active.get_mut(row) {
                *active = true;
            }
            return Ok(());
        }
        let row = self.entity_ids.len();
        self.entity_ids.push(entity_id);
        for coordinate in coordinates {
            self.values.push(*coordinate);
        }
        self.versions.push(revision);
        self.active.push(true);
        self.rows.insert(u128::from(entity_id), row);
        Ok(())
    }

    pub fn remove(&mut self, entity_id: u64, revision: u64) {
        if let Some(row) = self.rows.get(u128::from(entity_id)).copied() {
            if let Some(active) = self.active.get_mut(row) {
                *active = false;
            }
            if let Some(version) = self.versions.get_mut(row) {
                *version = revision;
            }
        }
    }

    fn retain_entities(&mut self, retained: &BTreeSet<u64>) -> Result<BTreeMap<usize, u32>> {
        let mut filtered = Self::new_with_dtype(self.dimension, self.similarity, self.dtype)?;
        let mut remap = BTreeMap::new();
        for row in 0..self.entity_ids.len() {
            let Some(entity) = self.entity_ids.get(row).copied() else {
                continue;
            };
            if !self.active.get(row).copied().unwrap_or(false) || !retained.contains(&entity) {
                continue;
            }
            let start = row.checked_mul(self.dimension).ok_or_else(|| {
                Error::new(ErrorCode::CorruptStorage, "vector row offset overflow")
            })?;
            let mut coordinates = Vec::with_capacity(self.dimension);
            for offset in 0..self.dimension {
                coordinates.push(*self.values.get(start + offset).ok_or_else(|| {
                    Error::new(
                        ErrorCode::CorruptStorage,
                        "vector row points outside its canonical matrix",
                    )
                })?);
            }
            let revision = self.versions.get(row).copied().ok_or_else(|| {
                Error::new(ErrorCode::CorruptStorage, "vector row has no revision")
            })?;
            let next_row = u32::try_from(filtered.entity_ids.len()).map_err(|_| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "filtered vector row exceeds device index width",
                )
            })?;
            filtered.upsert_quantized(entity, &coordinates, revision)?;
            remap.insert(row, next_row);
        }
        *self = filtered;
        Ok(remap)
    }

    pub fn exact_search(&self, query: &[f32], limit: usize) -> Result<Vec<VectorHit>> {
        let query = self.prepare(query)?;
        let mut hits = Vec::with_capacity(self.active.iter().filter(|active| **active).count());
        for row in 0..self.entity_ids.len() {
            if !self.active.get(row).copied().unwrap_or(false) {
                continue;
            }
            let Some(vector) = self.vector(row) else {
                continue;
            };
            hits.push(VectorHit {
                entity_id: self.entity_ids.get(row).copied().unwrap_or_default(),
                score: public_score(self.similarity, &query, &vector),
            });
        }
        sort_hits(&mut hits, self.similarity);
        hits.truncate(limit);
        Ok(hits)
    }

    /// Decodes one active canonical row for model-memory projection.
    #[must_use]
    pub fn vector_for(&self, entity_id: u64) -> Option<Vec<f32>> {
        let row = self.rows.get(u128::from(entity_id)).copied()?;
        self.active
            .get(row)
            .copied()
            .unwrap_or(false)
            .then(|| self.vector(row))?
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
            let norm = result.iter().map(|value| value * value).sum::<f32>().sqrt();
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
            .map(|value| encode_coordinate(self.dtype, value))
            .collect())
    }

    fn vector(&self, row: usize) -> Option<Vec<f32>> {
        let start = row.checked_mul(self.dimension)?;
        let end = start.checked_add(self.dimension)?;
        (start..end)
            .map(|index| {
                self.values
                    .get(index)
                    .copied()
                    .map(|bits| decode_coordinate(self.dtype, bits))
            })
            .collect()
    }

    #[must_use]
    pub const fn dimension(&self) -> usize {
        self.dimension
    }

    #[must_use]
    pub const fn similarity(&self) -> Similarity {
        self.similarity
    }

    #[must_use]
    pub const fn dtype(&self) -> EmbeddingDType {
        self.dtype
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.active.iter().filter(|active| **active).count()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[cfg(test)]
    fn detached_storage_bytes_from(&self, previous: &Self) -> usize {
        self.entity_ids
            .detached_page_bytes_from(&previous.entity_ids)
            .saturating_add(self.values.detached_page_bytes_from(&previous.values))
            .saturating_add(self.versions.detached_page_bytes_from(&previous.versions))
            .saturating_add(self.active.detached_page_bytes_from(&previous.active))
            .saturating_add(self.rows.detached_node_bytes_from(&previous.rows))
    }
}

/// Frozen deterministic IVF-PQ build/search parameters.
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
/// Minimum measured validation recall required before an IVF-PQ build may be planned.
pub const IVF_PQ_MIN_RECALL_BASIS_POINTS: u16 = 9_000;
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
    recall_basis_points: u16,
    #[serde(default)]
    validation_queries: u16,
    #[serde(default)]
    build_generation: [u8; 32],
}

impl IvfPqIndex {
    #[must_use]
    pub const fn config(&self) -> IvfPqConfig {
        self.config
    }

    #[must_use]
    pub const fn recall_basis_points(&self) -> u16 {
        self.recall_basis_points
    }

    #[must_use]
    pub const fn validation_queries(&self) -> u16 {
        self.validation_queries
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
            // Filtering changes the validation population. The remapped pages remain a correct
            // candidate accelerator with exact delta/reranking, but cannot be optimizer-selected
            // as ANN until the filtered population is independently recall-validated.
            recall_basis_points: 0,
            validation_queries: 0,
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
            recall_basis_points: self.recall_basis_points,
            validation_queries: self.validation_queries,
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
        let active_rows = source.active.iter().filter(|active| **active).count();
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
            recall_basis_points: 0,
            validation_queries: 0,
            build_generation: [0_u8; 32],
        };
        index.validate_recall(source, cancellation)?;
        index.refresh_generation()?;
        Ok(index)
    }

    fn validate_recall(
        &mut self,
        source: &VectorIndex,
        cancellation: &CancellationToken,
    ) -> Result<()> {
        const MAX_VALIDATION_QUERIES: usize = 64;
        const VALIDATION_TOP_K: usize = 10;
        let active = source
            .active
            .iter()
            .copied()
            .enumerate()
            .filter_map(|(row, active)| active.then_some(row))
            .collect::<Vec<_>>();
        let query_count = active.len().min(MAX_VALIDATION_QUERIES);
        if query_count == 0 {
            return Err(Error::new(
                ErrorCode::IndexUnavailable,
                "IVF-PQ recall validation has no active queries",
            ));
        }
        let mut expected = 0_u64;
        let mut matched = 0_u64;
        for ordinal in 0..query_count {
            check_ivf_build_cancelled(cancellation)?;
            let sampled = ordinal.saturating_mul(active.len()) / query_count;
            let row = active[sampled.min(active.len() - 1)];
            let query = source.vector(row).ok_or_else(|| {
                Error::new(
                    ErrorCode::CorruptStorage,
                    "IVF-PQ validation row has no canonical vector",
                )
            })?;
            let k = VALIDATION_TOP_K.min(active.len());
            let exact = source.exact_search(&query, k)?;
            let approximate = self.search(source, &query, k)?;
            let approximate_ids = approximate
                .iter()
                .map(|hit| hit.entity_id)
                .collect::<BTreeSet<_>>();
            expected = expected.saturating_add(exact.len() as u64);
            matched = matched.saturating_add(
                exact
                    .iter()
                    .filter(|hit| approximate_ids.contains(&hit.entity_id))
                    .count() as u64,
            );
        }
        let recall = if expected == 0 {
            10_000_u64
        } else {
            matched.saturating_mul(10_000) / expected
        };
        self.recall_basis_points = u16::try_from(recall.min(10_000)).map_err(|_| {
            Error::internal("IVF-PQ recall basis points exceed their bounded representation")
        })?;
        self.validation_queries = u16::try_from(query_count)
            .map_err(|_| Error::internal("IVF-PQ validation query count exceeds u16"))?;
        if self.recall_basis_points < IVF_PQ_MIN_RECALL_BASIS_POINTS {
            return Err(Error::new(
                ErrorCode::IndexUnavailable,
                format!(
                    "IVF-PQ recall {}bp is below the {}bp publication floor",
                    self.recall_basis_points, IVF_PQ_MIN_RECALL_BASIS_POINTS
                ),
            ));
        }
        Ok(())
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
        for (_, list_index) in coarse_order
            .into_iter()
            .take(self.config.probes.min(self.coarse.len()))
        {
            let Some(bounds) = self.list_offsets.get(list_index..=list_index + 1) else {
                continue;
            };
            let residual = subtract(&query, &self.coarse[list_index])?;
            for position in &self.list_positions[bounds[0] as usize..bounds[1] as usize] {
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
        let mut built_position = 0_usize;
        for row in 0..source.entity_ids.len() {
            while self
                .rows
                .get(built_position)
                .is_some_and(|built| (*built as usize) < row)
            {
                built_position += 1;
            }
            let built_revision = self
                .rows
                .get(built_position)
                .filter(|built| (**built as usize) == row)
                .and_then(|_| self.built_versions.get(built_position))
                .copied();
            if built_revision != source.versions.get(row).copied() {
                candidate_rows.insert(row);
            }
        }
        let mut hits = Vec::new();
        for row in candidate_rows {
            if !source.active.get(row).copied().unwrap_or(false) {
                continue;
            }
            let (Some(entity_id), Some(vector)) = (source.entity_ids.get(row), source.vector(row))
            else {
                continue;
            };
            hits.push(VectorHit {
                entity_id: *entity_id,
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
    for (row, active) in source.active.iter().copied().enumerate() {
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
    for (row, active) in source.active.iter().copied().enumerate() {
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
        built_versions.push(source.versions.get(row).copied().ok_or_else(|| {
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
    /// Recall measured by the last validated build. Zero means no publishable ANN evidence.
    pub ann_recall_basis_points: u16,
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
        approximate: Option<IvfPqIndex>,
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
    entries: BTreeMap<String, Arc<IndexEntry>>,
    embeddings: BTreeMap<String, EmbeddingIndexDefinition>,
    profile: Option<EmbeddingProfile>,
    /// Vectors are canonical embedding rows; ANN structures in `entries` are derived.
    vector_columns: BTreeMap<PropertyId, Arc<VectorIndex>>,
    #[serde(skip)]
    optimizer_generation_cache: OnceLock<[u8; 32]>,
}

impl IndexCatalog {
    fn invalidate_optimizer_generation(&mut self) {
        self.optimizer_generation_cache.take();
    }

    pub fn rebind_shared_vectors(&mut self, backings: Vec<SharedVectorBacking>) -> Result<()> {
        if backings.len() != self.vector_columns.len() {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "shared vector backing set is incomplete",
            ));
        }
        for backing in backings {
            let column = self
                .vector_columns
                .get_mut(&backing.property)
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::CorruptStorage,
                        "shared vector backing targets an unknown property",
                    )
                })?;
            Arc::make_mut(column).rebind_shared(backing)?;
        }
        Ok(())
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
            && self.embeddings.is_empty()
            && self.profile.is_none()
            && self.vector_columns.is_empty()
    }

    /// Produces the complete canonical-vector and ONLINE-derived-index device image.
    /// Failed or populating indexes remain query-ineligible and are intentionally absent.
    pub fn device_image(&self) -> Result<IndexDeviceImage> {
        let vectors = self
            .vector_columns
            .iter()
            .map(|(property, column)| column.device_image(*property))
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
                        .map(IvfPqIndex::device_image)
                        .transpose()?,
                },
            });
        }
        Ok(IndexDeviceImage {
            profile: self.profile.clone(),
            vectors,
            indexes,
        })
    }

    #[must_use]
    pub fn contains(&self, name: &str) -> bool {
        self.entries.contains_key(name)
    }

    #[must_use]
    pub fn profile(&self) -> Option<&EmbeddingProfile> {
        self.profile.as_ref()
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

    pub fn activate_profile(&mut self, profile: EmbeddingProfile) -> Result<()> {
        profile.validate()?;
        if let Some(current) = &self.profile {
            if current != &profile {
                if !self.embedding_profile_is_mutable() {
                    return Err(Error::new(
                        ErrorCode::EmbeddingProfileImmutable,
                        "project embedding profile is fixed by vector data or an index",
                    ));
                }
                self.invalidate_optimizer_generation();
                self.profile = Some(profile);
            }
            return Ok(());
        }
        self.invalidate_optimizer_generation();
        self.profile = Some(profile);
        Ok(())
    }

    pub fn create(&mut self, graph: &GraphStore, definition: GraphIndexDefinition) -> Result<()> {
        self.create_with_vector_population(graph, definition, true)
    }

    /// Applies the durable definition while deferring vector population to the selected local
    /// execution backend. Scalar families still build immediately because they have no separate
    /// accelerator builder and may enforce constraints during semantic preflight.
    pub fn create_deferred(
        &mut self,
        graph: &GraphStore,
        definition: GraphIndexDefinition,
    ) -> Result<()> {
        self.create_with_vector_population(graph, definition, false)
    }

    fn create_with_vector_population(
        &mut self,
        graph: &GraphStore,
        definition: GraphIndexDefinition,
        populate_vector: bool,
    ) -> Result<()> {
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
            let profile = self.profile.as_ref().ok_or_else(|| {
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
            self.vector_columns
                .entry(property)
                .or_insert(Arc::new(VectorIndex::new_with_dtype(
                    profile.dimension as usize,
                    profile.similarity,
                    profile.dtype,
                )?));
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
        &mut self,
        graph: &GraphStore,
        definition: EmbeddingIndexDefinition,
        profile: EmbeddingProfile,
        rows: Vec<(u64, Vec<u16>, u64)>,
    ) -> Result<()> {
        self.create_embedding_with_vector_population(graph, definition, profile, rows, true)
    }

    pub fn create_embedding_deferred(
        &mut self,
        graph: &GraphStore,
        definition: EmbeddingIndexDefinition,
        profile: EmbeddingProfile,
        rows: Vec<(u64, Vec<u16>, u64)>,
    ) -> Result<()> {
        self.create_embedding_with_vector_population(graph, definition, profile, rows, false)
    }

    fn create_embedding_with_vector_population(
        &mut self,
        graph: &GraphStore,
        definition: EmbeddingIndexDefinition,
        profile: EmbeddingProfile,
        rows: Vec<(u64, Vec<u16>, u64)>,
        populate_vector: bool,
    ) -> Result<()> {
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
        let matrix = self
            .vector_columns
            .entry(definition.target_property)
            .or_insert(Arc::new(VectorIndex::new_with_dtype(
                profile.dimension as usize,
                profile.similarity,
                profile.dtype,
            )?));
        let matrix = Arc::make_mut(matrix);
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
    pub fn rebuild(&mut self, graph: &GraphStore, name: &str) -> Result<()> {
        let definition = self
            .entries
            .get(name)
            .map(|entry| entry.definition.clone())
            .ok_or_else(|| Error::new(ErrorCode::IndexUnavailable, "index does not exist"))?;
        self.invalidate_optimizer_generation();
        if definition.unique {
            let runtime = self.build_runtime(graph, &definition)?;
            validate_unique_runtime(&runtime)?;
            let entry = self
                .entries
                .get_mut(name)
                .ok_or_else(|| Error::internal("constraint disappeared during rebuild"))?;
            let entry = Arc::make_mut(entry);
            entry.runtime = Some(Arc::new(runtime));
            entry.state = DerivedIndexState::Online;
            entry.diagnostic = None;
            return Ok(());
        }
        if let Some(entry) = self.entries.get_mut(name) {
            let entry = Arc::make_mut(entry);
            entry.state = DerivedIndexState::Populating;
            entry.diagnostic = None;
            entry.runtime = None;
        }
        match self.build_runtime(graph, &definition) {
            Ok(runtime) => {
                let entry = self
                    .entries
                    .get_mut(name)
                    .ok_or_else(|| Error::internal("index disappeared during population"))?;
                let entry = Arc::make_mut(entry);
                entry.runtime = Some(Arc::new(runtime));
                entry.state = DerivedIndexState::Online;
                Ok(())
            }
            Err(error) => {
                let entry = self.entries.get_mut(name).ok_or_else(|| {
                    Error::internal("index disappeared while recording population failure")
                })?;
                let entry = Arc::make_mut(entry);
                entry.state = DerivedIndexState::Failed;
                entry.diagnostic = Some(error.to_string());
                Ok(())
            }
        }
    }

    /// Marks a vector rebuild request without executing a host builder during semantic
    /// validation. An existing validated runtime stays ONLINE until its replacement is ready.
    pub fn rebuild_deferred(&mut self, graph: &GraphStore, name: &str) -> Result<()> {
        let entry = self
            .entries
            .get(name)
            .ok_or_else(|| Error::new(ErrorCode::IndexUnavailable, "index does not exist"))?;
        if entry.definition.kind != GraphIndexKind::Vector {
            return self.rebuild(graph, name);
        }
        self.invalidate_optimizer_generation();
        let entry = self
            .entries
            .get_mut(name)
            .ok_or_else(|| Error::internal("vector index disappeared during rebuild request"))?;
        let entry = Arc::make_mut(entry);
        // The old runtime, when present, remains owned by this staged catalog until the fully
        // validated replacement is swapped below. No reader can observe this staged state.
        entry.state = DerivedIndexState::Populating;
        entry.diagnostic = None;
        Ok(())
    }

    /// Builds every local vector generation through the selected backend and atomically replaces
    /// each runtime only after cancellation, structural checks, and the recall floor pass.
    pub fn rebuild_vectors_with(
        &mut self,
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
            let source = self.vector_columns.get(&property).cloned().ok_or_else(|| {
                Error::new(
                    ErrorCode::CorruptStorage,
                    "vector index has no canonical vector column",
                )
            })?;
            let candidate = if source.is_empty() {
                Ok(None)
            } else {
                builder(&source, ivf_config(&source)).and_then(|index| {
                    if index.recall_basis_points() < IVF_PQ_MIN_RECALL_BASIS_POINTS
                        || index.validation_queries() == 0
                        || index.build_generation() == [0_u8; 32]
                    {
                        return Err(Error::new(
                            ErrorCode::IndexUnavailable,
                            "IVF-PQ builder returned an unvalidated generation",
                        ));
                    }
                    Ok(Some(index))
                })
            };
            match candidate {
                Ok(approximate) => {
                    let replacement = RuntimeIndex::Vector {
                        property,
                        approximate,
                    };
                    let entry = self.entries.get_mut(&name).ok_or_else(|| {
                        Error::internal("vector index disappeared before atomic publication")
                    })?;
                    let entry = Arc::make_mut(entry);
                    entry.runtime = Some(Arc::new(replacement));
                    entry.state = DerivedIndexState::Online;
                    entry.diagnostic = None;
                }
                Err(error) => {
                    let entry = self.entries.get_mut(&name).ok_or_else(|| {
                        Error::internal("vector index disappeared while recording build failure")
                    })?;
                    let entry = Arc::make_mut(entry);
                    if entry.runtime.is_none() {
                        entry.state = DerivedIndexState::Failed;
                    } else {
                        // Failed rebuilds never evict the previously validated generation.
                        entry.state = DerivedIndexState::Online;
                    }
                    entry.diagnostic = Some(error.to_string());
                }
            }
        }
        Ok(())
    }

    pub fn rebuild_all(&mut self, graph: &GraphStore) -> Result<()> {
        let names = self.entries.keys().cloned().collect::<Vec<_>>();
        for name in names {
            self.rebuild(graph, &name)?;
        }
        Ok(())
    }

    /// Rebuilds every derived family after a layer-filtered graph is compacted. Vector columns are
    /// first reduced to the stable entities retained by the compacted graph.
    pub fn rebuild_row_indexes(&mut self, graph: &GraphStore) -> Result<()> {
        self.invalidate_optimizer_generation();
        let retained = graph
            .nodes()
            .map(|node| node.id().0)
            .collect::<BTreeSet<_>>();
        let mut vector_row_remaps = BTreeMap::new();
        for (property, column) in &mut self.vector_columns {
            vector_row_remaps.insert(*property, Arc::make_mut(column).retain_entities(&retained)?);
        }
        let names = self.entries.keys().cloned().collect::<Vec<_>>();
        for name in names {
            let is_vector = self
                .entries
                .get(&name)
                .is_some_and(|entry| entry.definition.kind == GraphIndexKind::Vector);
            if !is_vector {
                self.rebuild(graph, &name)?;
                continue;
            }

            let entry = self
                .entries
                .get_mut(&name)
                .ok_or_else(|| Error::internal("vector index disappeared during filtering"))?;
            let entry = Arc::make_mut(entry);
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
                    .flatten(),
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

    pub fn drop_index(&mut self, name: &str) -> Result<()> {
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

    pub fn drop_constraint(&mut self, name: &str) -> Result<()> {
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

    fn remove_definition(&mut self, name: &str) -> Result<()> {
        self.invalidate_optimizer_generation();
        let Some(mut entry) = self.entries.remove(name) else {
            return Err(Error::new(
                ErrorCode::IndexUnavailable,
                "index does not exist",
            ));
        };
        let entry = Arc::make_mut(&mut entry);
        entry.state = DerivedIndexState::Dropping;
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

    pub fn constraint_definitions(&self) -> impl Iterator<Item = &GraphIndexDefinition> {
        self.entries
            .values()
            .filter(|entry| entry.definition.unique)
            .map(|entry| &entry.definition)
    }

    pub fn statuses(&self) -> impl Iterator<Item = IndexStatus> + '_ {
        self.entries.values().map(|entry| IndexStatus {
            name: entry.definition.name.clone(),
            kind: entry.definition.kind,
            state: entry.state,
            diagnostic: entry.diagnostic.clone(),
        })
    }

    /// Stable generation of the schema and local lifecycle facts that can change physical access
    /// path legality. Canonical posting contents are intentionally excluded: data churn may make
    /// an estimate stale, but cannot make an ONLINE access path semantically invalid.
    #[must_use]
    pub fn optimizer_generation(&self) -> [u8; 32] {
        *self.optimizer_generation_cache.get_or_init(|| {
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
            if let Some(profile) = &self.profile {
                hasher.update(&profile.profile_hash);
            }
            *hasher.finalize().as_bytes()
        })
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
                    recall_basis_points,
                    resident_bytes,
                ) = match runtime {
                    RuntimeIndex::Equality(index) => {
                        let shape = scalar_posting_shape(&index.postings, &index.deltas);
                        (shape.0, shape.1, shape.2, 0, 0, 0, shape.3)
                    }
                    RuntimeIndex::Range(index) => {
                        let shape = scalar_posting_shape(&index.postings, &index.deltas);
                        (shape.0, shape.1, shape.2, 0, 0, 0, shape.3)
                    }
                    RuntimeIndex::Text(index) => {
                        let shape = text_posting_shape(index);
                        (shape.0, shape.1, shape.2, 0, 0, 0, shape.3)
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
                        let recall_basis_points = approximate
                            .as_ref()
                            .map_or(0_u16, |ann| ann.recall_basis_points());
                        let dimension = self
                            .profile
                            .as_ref()
                            .map_or(0_u64, |profile| u64::from(profile.dimension));
                        let bytes = rows.saturating_mul(
                            8_u64
                                .saturating_add(8)
                                .saturating_add(1)
                                .saturating_add(dimension.saturating_mul(2)),
                        );
                        (0, 0, 0, rows, candidate_budget, recall_basis_points, bytes)
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
                    ann_recall_basis_points: recall_basis_points,
                    resident_bytes,
                })
            })
            .collect()
    }

    pub fn definitions(&self) -> impl Iterator<Item = &GraphIndexDefinition> {
        self.entries.values().map(|entry| &entry.definition)
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
                RuntimeIndex::Equality(index) => posting_cardinality(
                    index.postings.get(&key),
                    posting_changes(&index.deltas, &key),
                ),
                RuntimeIndex::Range(index) => posting_cardinality(
                    index.postings.get(&key),
                    posting_changes(&index.deltas, &key),
                ),
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
        let mut best: Option<(usize, &str, Vec<u32>)> = None;
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
            let candidate = (rows.len(), name.as_str(), rows);
            if best
                .as_ref()
                .is_none_or(|current| (candidate.0, candidate.1) < (current.0, current.1))
            {
                best = Some(candidate);
            }
        }
        Ok(best.map(|(_, _, rows)| rows))
    }

    pub fn embedding_definitions(&self) -> impl Iterator<Item = &EmbeddingIndexDefinition> {
        self.embeddings.values()
    }

    #[must_use]
    pub fn embedding_definition(&self, name: &str) -> Option<&EmbeddingIndexDefinition> {
        self.embeddings.get(name)
    }

    /// Returns the project's single canonical embedding column when one is declared.
    pub fn canonical_embedding_column(&self) -> Result<Option<(PropertyId, &VectorIndex)>> {
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
    pub fn before_graph_apply(
        &mut self,
        graph: &GraphStore,
        mutation: &GraphMutation,
    ) -> Result<()> {
        match mutation {
            GraphMutation::SetNodeProperty { node, .. }
            | GraphMutation::AddNodeLabels { node, .. }
            | GraphMutation::RemoveNodeLabels { node, .. } => {
                if let Some(view) = graph.node(*node) {
                    self.remove_node(view, false)?;
                }
            }
            GraphMutation::DeleteNode { node, .. } => {
                if let Some(view) = graph.node(*node) {
                    self.remove_node(view, true)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// Adds current postings after a canonical graph mutation creates or updates a row.
    pub fn after_graph_apply(
        &mut self,
        graph: &GraphStore,
        mutation: &GraphMutation,
    ) -> Result<()> {
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

    pub fn apply_vector_mutation(&mut self, mutation: &ResolvedVectorMutation) -> Result<()> {
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
        let matrix = self.vector_columns.get_mut(&property).ok_or_else(|| {
            Error::new(
                ErrorCode::EmbeddingProfileMismatch,
                "resolved vector targets an undeclared vector property",
            )
        })?;
        let matrix = Arc::make_mut(matrix);
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
    pub fn vector_search_source(&self, name: &str) -> Option<(&VectorIndex, Option<&IvfPqIndex>)> {
        let entry = self.entries.get(name)?;
        if entry.state != DerivedIndexState::Online {
            return None;
        }
        let RuntimeIndex::Vector {
            property,
            approximate,
        } = entry.runtime.as_deref()?
        else {
            return None;
        };
        Some((self.vector_columns.get(property)?, approximate.as_ref()))
    }

    fn build_runtime(
        &self,
        graph: &GraphStore,
        definition: &GraphIndexDefinition,
    ) -> Result<RuntimeIndex> {
        match definition.kind {
            GraphIndexKind::Equality => {
                let mut index = EqualityIndex::default();
                for node in graph
                    .nodes()
                    .filter(|node| node.labels().contains(&definition.label))
                {
                    if let Some(key) = composite_key(node, &definition.properties)? {
                        index.insert_key(key, node.dense());
                    }
                }
                index.seal();
                Ok(RuntimeIndex::Equality(index))
            }
            GraphIndexKind::Range => {
                let mut index = RangeIndex::default();
                for node in graph
                    .nodes()
                    .filter(|node| node.labels().contains(&definition.label))
                {
                    if let Some(key) = composite_key(node, &definition.properties)? {
                        index.insert_key(key, node.dense());
                    }
                }
                index.seal();
                Ok(RuntimeIndex::Range(index))
            }
            GraphIndexKind::Text => {
                let property = definition.properties[0];
                let mut index = TextIndex::default();
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
                index.seal();
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
                    Some(IvfPqIndex::build(matrix, ivf_config(matrix))?)
                };
                Ok(RuntimeIndex::Vector {
                    property,
                    approximate,
                })
            }
        }
    }

    fn remove_node(&mut self, node: NodeView<'_>, remove_vectors: bool) -> Result<()> {
        for entry in self.entries.values_mut() {
            if entry.state != DerivedIndexState::Online
                || !node.labels().contains(&entry.definition.label)
            {
                continue;
            }
            let entry = Arc::make_mut(entry);
            let Some(runtime) = entry.runtime.as_mut() else {
                continue;
            };
            if matches!(runtime.as_ref(), RuntimeIndex::Vector { .. }) {
                continue;
            }
            match Arc::make_mut(runtime) {
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
                    if let Some(matrix) = self.vector_columns.get_mut(&definition.target_property) {
                        Arc::make_mut(matrix).remove(node.id().0, node.revision());
                    }
                }
            }
        }
        Ok(())
    }

    fn insert_node(&mut self, node: NodeView<'_>) -> Result<()> {
        for entry in self.entries.values_mut() {
            if entry.state != DerivedIndexState::Online
                || !node.labels().contains(&entry.definition.label)
            {
                continue;
            }
            let entry = Arc::make_mut(entry);
            let Some(runtime) = entry.runtime.as_mut() else {
                continue;
            };
            if matches!(runtime.as_ref(), RuntimeIndex::Vector { .. }) {
                continue;
            }
            match Arc::make_mut(runtime) {
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

fn posting_changes<'a>(
    deltas: &'a PersistentMap<Vec<PostingDelta>>,
    key: &IndexKey,
) -> Option<&'a PersistentMap<bool>> {
    deltas
        .get(index_key_hash(key))?
        .iter()
        .find(|delta| &delta.key == key)
        .map(|delta| &delta.changes)
}

fn posting_cardinality(base: Option<&RoaringBitmap>, changes: Option<&PersistentMap<bool>>) -> u64 {
    let mut count = base.map_or(0, RoaringBitmap::len);
    if let Some(changes) = changes {
        for (row, present) in changes.iter() {
            let Ok(row) = u32::try_from(row) else {
                continue;
            };
            let was_present = base.is_some_and(|rows| rows.contains(row));
            match (*present, was_present) {
                (true, false) => count = count.saturating_add(1),
                (false, true) => count = count.saturating_sub(1),
                _ => {}
            }
        }
    }
    count
}

fn scalar_posting_shape(
    base: &BTreeMap<IndexKey, RoaringBitmap>,
    deltas: &PersistentMap<Vec<PostingDelta>>,
) -> (u64, u64, u64, u64) {
    let mut keys = BTreeSet::<&IndexKey>::new();
    keys.extend(base.keys());
    for (_, bucket) in deltas.iter() {
        keys.extend(bucket.iter().map(|delta| &delta.key));
    }
    let mut distinct = 0_u64;
    let mut total = 0_u64;
    let mut largest = 0_u64;
    let mut bytes = 0_u64;
    for key in keys {
        let rows = posting_cardinality(base.get(key), posting_changes(deltas, key));
        if rows == 0 {
            continue;
        }
        distinct = distinct.saturating_add(1);
        total = total.saturating_add(rows);
        largest = largest.max(rows);
        bytes = bytes
            .saturating_add(index_key_estimated_bytes(key))
            .saturating_add(rows.saturating_mul(4));
    }
    (distinct, total, largest, bytes)
}

fn text_posting_shape(index: &TextIndex) -> (u64, u64, u64, u64) {
    // Statistics need only cardinalities. Applying sparse presence overrides arithmetically avoids
    // cloning every bitmap and never scales with unrelated posting rows; work is one pass over the
    // term dictionary plus the changed row keys.
    let mut terms = BTreeSet::<&str>::new();
    terms.extend(index.postings.keys().map(String::as_str));
    for (_, bucket) in index.posting_deltas.iter() {
        terms.extend(bucket.iter().map(|delta| delta.term.as_str()));
    }
    let mut distinct = 0_u64;
    let mut total = 0_u64;
    let mut largest = 0_u64;
    let mut bytes = 0_u64;
    for term in terms {
        let changes = index
            .posting_deltas
            .get(text_term_hash(term))
            .and_then(|bucket| bucket.iter().find(|delta| delta.term == term))
            .map(|delta| &delta.changes);
        let rows = posting_cardinality(index.postings.get(term), changes);
        if rows == 0 {
            continue;
        }
        distinct = distinct.saturating_add(1);
        total = total.saturating_add(rows);
        largest = largest.max(rows);
        bytes = bytes
            .saturating_add(term.len() as u64)
            .saturating_add(rows.saturating_mul(4));
    }
    (distinct, total, largest, bytes)
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
mod tests {
    use std::sync::Arc;

    use super::*;
    use crate::{Layer, NodeId, NodeInput};

    #[test]
    fn scalar_index_key_preserves_wide_date_days() -> Result<()> {
        let wide_days = 365_242_499_634_i64;
        assert_eq!(
            IndexKey::try_from(&ScalarValue::Date(wide_days))?,
            IndexKey::Date(wide_days)
        );
        Ok(())
    }

    fn indexed_graph() -> Result<(GraphStore, LabelId, PropertyId, PropertyId)> {
        let mut graph = GraphStore::default();
        let label = graph.catalog_mut().intern_label("Document")?;
        let source = graph.catalog_mut().intern_property("text")?;
        let target = graph.catalog_mut().intern_property("embedding")?;
        graph.insert_node(NodeInput {
            id: NodeId(7),
            layer: Layer::Observed,
            revision: 1,
            labels: vec![label],
            properties: vec![(source, ScalarValue::String(Arc::from("first")))],
        })?;
        Ok((graph, label, source, target))
    }

    #[test]
    fn index_optimizer_generation_is_cached_and_invalidated_by_definition_change() -> Result<()> {
        let (graph, label, source, _) = indexed_graph()?;
        let mut catalog = IndexCatalog::default();
        let empty = catalog.optimizer_generation();
        assert!(catalog.optimizer_generation_cache.get().is_some());
        assert_eq!(catalog.optimizer_generation(), empty);
        catalog.create(
            &graph,
            GraphIndexDefinition {
                name: "document_text".to_owned(),
                kind: GraphIndexKind::Equality,
                label,
                properties: vec![source],
                unique: false,
            },
        )?;
        assert!(catalog.optimizer_generation_cache.get().is_none());
        assert_ne!(catalog.optimizer_generation(), empty);
        assert!(catalog.optimizer_generation_cache.get().is_some());
        Ok(())
    }

    fn equality_with_rows(rows: u32) -> EqualityIndex {
        EqualityIndex {
            postings: Arc::new(BTreeMap::from([(
                IndexKey::String("shared".to_owned()),
                (0..rows).collect(),
            )])),
            deltas: PersistentMap::default(),
        }
    }

    fn text_with_rows(rows: u32) -> TextIndex {
        let terms = BTreeSet::from(["common".to_owned()]);
        TextIndex {
            postings: Arc::new(BTreeMap::from([("common".to_owned(), (0..rows).collect())])),
            rows: Arc::new((0..rows).map(|row| (row, terms.clone())).collect()),
            posting_deltas: PersistentMap::default(),
            row_overrides: PersistentMap::default(),
        }
    }

    #[test]
    fn optimizer_statistics_delta_shape_avoids_posting_materialization() {
        let small = text_with_rows(40);
        let large = text_with_rows(400_000);
        let mut small_next = small.clone();
        let mut large_next = large.clone();
        small_next.upsert(17, "replacement");
        large_next.upsert(17, "replacement");

        assert!(Arc::ptr_eq(&small.postings, &small_next.postings));
        assert!(Arc::ptr_eq(&large.postings, &large_next.postings));
        assert_eq!(text_posting_shape(&small_next), (2, 40, 39, 177));
        assert_eq!(
            text_posting_shape(&large_next),
            (2, 400_000, 399_999, 1_600_017)
        );
        assert_eq!(
            small_next.detached_storage_bytes_from(&small),
            large_next.detached_storage_bytes_from(&large),
            "one changed row must detach the same bounded delta shape regardless of base postings"
        );
    }

    fn vector_with_rows(rows: u64) -> Result<VectorIndex> {
        let mut index = VectorIndex::new(16, Similarity::Dot)?;
        let coordinates = vec![f16::from_f32(1.0).to_bits(); 16];
        for entity in 0..rows {
            index.upsert_quantized(entity, &coordinates, 1)?;
        }
        Ok(index)
    }

    #[test]
    fn ivf_pq_parameters_and_bounded_scratch_scale_to_large_shape() -> Result<()> {
        let small = ivf_config_for_shape(8_000, 384);
        assert_eq!(small.coarse_centroids, 64);
        assert_eq!(small.subquantizers, 8);
        assert_eq!(small.candidate_budget, 256);
        let small_plan = IvfPqBuildPlan::for_shape(8_000, 384, small)?;
        assert!(small_plan.assignment_tile_bytes < IVF_PQ_ASSIGNMENT_TILE_BYTES);

        let large_rows = 100_000_000;
        let large = ivf_config_for_shape(large_rows, 384);
        assert_eq!(large.coarse_centroids, u16::MAX as usize);
        assert_eq!(large.subquantizers, 48);
        assert_eq!(large.probes, 128);
        assert_eq!(large.candidate_budget, 8_192);
        let plan = IvfPqBuildPlan::for_shape(large_rows as u64, 384, large)?;
        assert_eq!(plan.training_rows, MAX_IVF_PQ_TRAINING_ROWS);
        assert_eq!(plan.batch_rows, IVF_PQ_BUILD_BATCH_ROWS);
        assert_eq!(plan.assignment_tile_bytes, IVF_PQ_ASSIGNMENT_TILE_BYTES);
        assert!((6_000_000_000..7_000_000_000).contains(&plan.derived_bytes));
        assert!(plan.peak_scratch_bytes < 1_200_000_000);
        let mut incompatible = large;
        incompatible.size_class_version = 0;
        assert!(IvfPqBuildPlan::for_shape(large_rows as u64, 384, incompatible).is_err());
        Ok(())
    }

    #[test]
    fn source_encoding_is_streamed_in_fixed_bounded_batches() -> Result<()> {
        struct RecordingKernel {
            row_counts: Vec<usize>,
        }

        impl IvfPqBuildKernel for RecordingKernel {
            fn assign(
                &mut self,
                vectors: &[f32],
                row_count: usize,
                dimension: usize,
                centroids: &[f32],
                centroid_count: usize,
            ) -> Result<Vec<u32>> {
                self.row_counts.push(row_count);
                CpuIvfPqBuildKernel.assign(vectors, row_count, dimension, centroids, centroid_count)
            }
        }

        let row_count = IVF_PQ_BUILD_BATCH_ROWS + 17;
        let mut source = VectorIndex::new(4, Similarity::Dot)?;
        for row in 0..row_count {
            let value = (row % 97) as f32 / 97.0;
            source.upsert(row as u64, &[value, 1.0 - value, value * value, 0.5], 1)?;
        }
        let config = IvfPqConfig {
            size_class_version: IVF_PQ_SIZE_CLASS_VERSION,
            coarse_centroids: 2,
            subquantizers: 2,
            bits_per_code: 2,
            probes: 1,
            candidate_budget: 64,
            iterations: 1,
            seed: 7,
        };
        let plan = IvfPqBuildPlan::for_shape(row_count as u64, 4, config)?;
        let coarse = vec![vec![0.0, 1.0, 0.0, 0.5], vec![1.0, 0.0, 1.0, 0.5]];
        let codebooks = vec![
            vec![
                vec![-0.5, -0.5],
                vec![-0.1, 0.1],
                vec![0.1, -0.1],
                vec![0.5, 0.5],
            ],
            vec![
                vec![-0.5, 0.0],
                vec![-0.1, 0.0],
                vec![0.1, 0.0],
                vec![0.5, 0.0],
            ],
        ];
        let mut recording = RecordingKernel {
            row_counts: Vec::new(),
        };
        let encoded = stream_encode_rows(
            &source,
            &coarse,
            &codebooks,
            config,
            plan,
            &mut recording,
            &CancellationToken::new(),
        )?;
        assert_eq!(encoded.rows.len(), row_count);
        assert_eq!(encoded.codes.len(), row_count * config.subquantizers);
        assert_eq!(encoded.list_offsets.last().copied(), Some(row_count as u32));
        assert!(
            recording
                .row_counts
                .iter()
                .all(|rows| *rows <= IVF_PQ_BUILD_BATCH_ROWS)
        );
        assert!(recording.row_counts.contains(&IVF_PQ_BUILD_BATCH_ROWS));
        assert!(recording.row_counts.contains(&17));
        Ok(())
    }

    #[test]
    fn filtered_vector_rows_remap_existing_ann_without_retraining() -> Result<()> {
        let mut source = VectorIndex::new(4, Similarity::Dot)?;
        for entity in 0_u64..64 {
            let value = entity as f32 / 64.0;
            source.upsert(
                entity,
                &[value, 1.0 - value, value * value, (entity % 7) as f32],
                entity + 1,
            )?;
        }
        let config = IvfPqConfig {
            size_class_version: IVF_PQ_SIZE_CLASS_VERSION,
            coarse_centroids: 4,
            subquantizers: 2,
            bits_per_code: 3,
            probes: 4,
            candidate_budget: 64,
            iterations: 2,
            seed: 19,
        };
        let approximate = IvfPqIndex::build(&source, config)?;
        let retained = (0_u64..64)
            .filter(|entity| entity % 3 == 1)
            .collect::<BTreeSet<_>>();
        let remap = source.retain_entities(&retained)?;
        let filtered = approximate
            .remap_rows(&remap)?
            .ok_or_else(|| Error::internal("non-empty ANN filter returned no index"))?;
        let query = [0.4, 0.6, 0.16, 3.0];
        assert_eq!(
            filtered.search(&source, &query, 10)?,
            source.exact_search(&query, 10)?
        );
        assert_eq!(filtered.rows.len(), retained.len());
        filtered.device_image()?;

        let empty = BTreeSet::new();
        let empty_remap = source.retain_entities(&empty)?;
        assert!(filtered.remap_rows(&empty_remap)?.is_none());
        Ok(())
    }

    #[test]
    fn index_point_updates_detach_only_bounded_storage_pages() -> Result<()> {
        let equality_key = IndexKey::String("shared".to_owned());
        let small_equality = equality_with_rows(4_096);
        let large_equality = equality_with_rows(32_768);
        let mut small_equality_next = small_equality.clone();
        let mut large_equality_next = large_equality.clone();
        small_equality_next.remove_key(&equality_key, 17);
        large_equality_next.remove_key(&equality_key, 17);
        assert!(Arc::ptr_eq(
            &small_equality.postings,
            &small_equality_next.postings
        ));
        assert!(Arc::ptr_eq(
            &large_equality.postings,
            &large_equality_next.postings
        ));
        assert!(
            small_equality
                .get(&equality_key)
                .is_some_and(|rows| rows.contains(17))
        );
        assert!(
            small_equality_next
                .get(&equality_key)
                .is_some_and(|rows| !rows.contains(17))
        );
        let small_scalar_bytes = small_equality_next.detached_storage_bytes_from(&small_equality);
        let large_scalar_bytes = large_equality_next.detached_storage_bytes_from(&large_equality);
        assert_eq!(small_scalar_bytes, large_scalar_bytes);
        assert!(large_scalar_bytes <= 32 * 1024);

        let small_text = text_with_rows(4_096);
        let large_text = text_with_rows(32_768);
        let mut small_text_next = small_text.clone();
        let mut large_text_next = large_text.clone();
        small_text_next.upsert(17, "replacement");
        large_text_next.upsert(17, "replacement");
        assert!(Arc::ptr_eq(&small_text.postings, &small_text_next.postings));
        assert!(Arc::ptr_eq(&small_text.rows, &small_text_next.rows));
        assert!(Arc::ptr_eq(&large_text.postings, &large_text_next.postings));
        assert!(Arc::ptr_eq(&large_text.rows, &large_text_next.rows));
        assert!(small_text.search("common").contains(17));
        assert!(!small_text_next.search("common").contains(17));
        assert!(small_text_next.search("replacement").contains(17));
        let small_text_bytes = small_text_next.detached_storage_bytes_from(&small_text);
        let large_text_bytes = large_text_next.detached_storage_bytes_from(&large_text);
        assert_eq!(small_text_bytes, large_text_bytes);
        assert!(large_text_bytes <= 96 * 1024);

        let small_vector = vector_with_rows(4_096)?;
        let large_vector = vector_with_rows(8_192)?;
        let mut small_vector_next = small_vector.clone();
        let mut large_vector_next = large_vector.clone();
        let replacement = vec![f16::from_f32(2.0).to_bits(); 16];
        small_vector_next.upsert_quantized(17, &replacement, 2)?;
        large_vector_next.upsert_quantized(17, &replacement, 2)?;
        assert_eq!(
            small_vector
                .vector_for(17)
                .and_then(|row| row.first().copied()),
            Some(1.0)
        );
        assert_eq!(
            small_vector_next
                .vector_for(17)
                .and_then(|row| row.first().copied()),
            Some(2.0)
        );
        let small_vector_bytes = small_vector_next.detached_storage_bytes_from(&small_vector);
        let large_vector_bytes = large_vector_next.detached_storage_bytes_from(&large_vector);
        assert_eq!(small_vector_bytes, large_vector_bytes);
        assert!(large_vector_bytes <= 32 * 1024);
        Ok(())
    }

    #[test]
    fn posting_overrides_do_not_retain_write_history() -> Result<()> {
        let key = IndexKey::String("value".to_owned());
        let mut equality = EqualityIndex::default();
        for revision in 0..10_000 {
            if revision % 2 == 0 {
                equality.insert_key(key.clone(), 7);
            } else {
                equality.remove_key(&key, 7);
            }
        }
        assert!(equality.get(&key).is_none());
        let equality_bytes = postcard::to_stdvec(&equality)
            .map_err(|error| Error::new(ErrorCode::CorruptStorage, error.to_string()))?;
        assert!(equality_bytes.len() < 1_024);

        let mut text = TextIndex::default();
        for revision in 0..10_000 {
            text.upsert(7, if revision % 2 == 0 { "alpha" } else { "beta" });
        }
        assert!(!text.search("alpha").contains(7));
        assert!(text.search("beta").contains(7));
        let text_bytes = postcard::to_stdvec(&text)
            .map_err(|error| Error::new(ErrorCode::CorruptStorage, error.to_string()))?;
        assert!(text_bytes.len() < 2_048);
        Ok(())
    }

    #[test]
    fn quantized_vector_mutation_replays_identical_bits() -> Result<()> {
        let (graph, label, source, target) = indexed_graph()?;
        let profile = EmbeddingProfile::new(
            [11; 32],
            [17; 32],
            4,
            EmbeddingDType::F16,
            true,
            Similarity::Cosine,
        )?;
        let initial = profile.quantize(&[1.0, 0.0, 0.0, 0.0])?;
        let definition = EmbeddingIndexDefinition {
            name: "semantic".to_owned(),
            label,
            source_property: source,
            target_property: target,
            model: "default".to_owned(),
        };
        let mut first = IndexCatalog::default();
        first.create_embedding(
            &graph,
            definition.clone(),
            profile.clone(),
            vec![(7, initial.clone(), 1)],
        )?;
        let mut second = IndexCatalog::default();
        second.create_embedding(&graph, definition, profile.clone(), vec![(7, initial, 1)])?;

        let bits = profile.quantize(&[0.25, -0.5, 0.75, 1.0])?;
        let mutation = ResolvedVectorMutation::Upsert {
            property: target,
            entity_id: 7,
            coordinates: bits.clone(),
            revision: 41,
        };
        let encoded = postcard::to_stdvec(&mutation)
            .map_err(|error| Error::internal(format!("test encoding failed: {error}")))?;
        let replayed: ResolvedVectorMutation = postcard::from_bytes(&encoded)
            .map_err(|error| Error::internal(format!("test decoding failed: {error}")))?;
        let ResolvedVectorMutation::Upsert { coordinates, .. } = &replayed else {
            return Err(Error::internal("upsert changed kind during replay"));
        };
        assert_eq!(coordinates, &bits);
        first.apply_vector_mutation(&mutation)?;
        second.apply_vector_mutation(&replayed)?;

        let first_bytes = postcard::to_stdvec(&first)
            .map_err(|error| Error::internal(format!("test encoding failed: {error}")))?;
        let second_bytes = postcard::to_stdvec(&second)
            .map_err(|error| Error::internal(format!("test encoding failed: {error}")))?;
        assert_eq!(first_bytes, second_bytes);
        assert_eq!(
            first
                .vector_search_source("semantic")
                .ok_or_else(|| Error::internal("first vector index is offline"))?
                .0
                .exact_search(&[0.25, -0.5, 0.75, 1.0], 1)?,
            second
                .vector_search_source("semantic")
                .ok_or_else(|| Error::internal("second vector index is offline"))?
                .0
                .exact_search(&[0.25, -0.5, 0.75, 1.0], 1)?,
        );
        Ok(())
    }

    #[test]
    fn ivf_build_is_cancellable_recall_gated_and_searches_exact_delta() -> Result<()> {
        let mut source = VectorIndex::new(2, Similarity::Euclidean)?;
        for (entity, value) in [(1_u64, 10.0_f32), (2, 20.0), (3, 30.0)] {
            source.upsert_quantized(
                entity,
                &[
                    f16::from_f32(value).to_bits(),
                    f16::from_f32(value).to_bits(),
                ],
                1,
            )?;
        }
        let cancellation = CancellationToken::new();
        cancellation.cancel();
        let cancelled = IvfPqIndex::build_cancellable(&source, ivf_config(&source), &cancellation)
            .expect_err("cancelled IVF-PQ build completed");
        assert_eq!(cancelled.code, ErrorCode::Cancelled);

        let index = IvfPqIndex::build(&source, ivf_config(&source))?;
        assert!(index.recall_basis_points() >= IVF_PQ_MIN_RECALL_BASIS_POINTS);
        assert!(index.validation_queries() > 0);
        assert_ne!(index.build_generation(), [0_u8; 32]);

        let zero = [f16::from_f32(0.0).to_bits(); 2];
        source.upsert_quantized(99, &zero, 2)?;
        let hits = index.search(&source, &[0.0, 0.0], 1)?;
        assert_eq!(hits.first().map(|hit| hit.entity_id), Some(99));
        assert_eq!(hits.first().map(|hit| hit.score), Some(0.0));
        Ok(())
    }

    #[test]
    fn failed_vector_rebuild_keeps_previous_validated_generation_online() -> Result<()> {
        let (graph, label, source, target) = indexed_graph()?;
        let profile = EmbeddingProfile::new(
            [11; 32],
            [17; 32],
            4,
            EmbeddingDType::F16,
            true,
            Similarity::Cosine,
        )?;
        let mut indexes = IndexCatalog::default();
        indexes.create_embedding(
            &graph,
            EmbeddingIndexDefinition {
                name: "semantic".to_owned(),
                label,
                source_property: source,
                target_property: target,
                model: "default".to_owned(),
            },
            profile.clone(),
            vec![(7, profile.quantize(&[1.0, 0.0, 0.0, 0.0])?, 1)],
        )?;
        let before = indexes
            .vector_search_source("semantic")
            .and_then(|(_, approximate)| approximate)
            .map(IvfPqIndex::build_generation)
            .ok_or_else(|| Error::internal("initial ANN generation is unavailable"))?;
        indexes.rebuild_deferred(&graph, "semantic")?;
        indexes.rebuild_vectors_with(|_, _| {
            Err(Error::new(
                ErrorCode::GpuAdmissionFailure,
                "injected local builder failure",
            ))
        })?;
        let after = indexes
            .vector_search_source("semantic")
            .and_then(|(_, approximate)| approximate)
            .map(IvfPqIndex::build_generation)
            .ok_or_else(|| Error::internal("failed rebuild evicted the valid ANN generation"))?;
        assert_eq!(after, before);
        assert_eq!(
            indexes.statuses().next().map(|status| status.state),
            Some(DerivedIndexState::Online)
        );
        Ok(())
    }

    #[test]
    fn profile_replacement_ignores_scalar_indexes_but_not_vector_state() -> Result<()> {
        let (graph, label, source, target) = indexed_graph()?;
        let first = EmbeddingProfile::new(
            [1; 32],
            [2; 32],
            4,
            EmbeddingDType::F16,
            true,
            Similarity::Cosine,
        )?;
        let second = EmbeddingProfile::new(
            [3; 32],
            [4; 32],
            4,
            EmbeddingDType::F16,
            true,
            Similarity::Cosine,
        )?;
        let mut catalog = IndexCatalog::default();
        catalog.activate_profile(first.clone())?;
        catalog.create(
            &graph,
            GraphIndexDefinition {
                name: "ordinary".to_owned(),
                kind: GraphIndexKind::Equality,
                label,
                properties: vec![source],
                unique: false,
            },
        )?;
        assert!(catalog.embedding_profile_is_mutable());
        catalog.activate_profile(second.clone())?;
        assert_eq!(catalog.profile(), Some(&second));

        catalog.create(
            &graph,
            GraphIndexDefinition {
                name: "vectors".to_owned(),
                kind: GraphIndexKind::Vector,
                label,
                properties: vec![target],
                unique: false,
            },
        )?;
        assert!(
            catalog
                .create(
                    &graph,
                    GraphIndexDefinition {
                        name: "duplicate_vectors".to_owned(),
                        kind: GraphIndexKind::Vector,
                        label,
                        properties: vec![target],
                        unique: false,
                    },
                )
                .is_err()
        );
        assert!(!catalog.embedding_profile_is_mutable());
        let error = catalog
            .activate_profile(first)
            .expect_err("vector state must freeze the embedding profile");
        assert_eq!(error.code, ErrorCode::EmbeddingProfileImmutable);
        Ok(())
    }

    #[test]
    fn uniqueness_constraint_rejects_existing_and_new_duplicate_values() -> Result<()> {
        let (graph, label, property, _) = indexed_graph()?;
        let definition = GraphIndexDefinition {
            name: "document_text_unique".to_owned(),
            kind: GraphIndexKind::Equality,
            label,
            properties: vec![property],
            unique: true,
        };

        let mut duplicated = graph.clone();
        duplicated.apply(GraphMutation::InsertNode(NodeInput {
            id: NodeId(8),
            layer: Layer::Observed,
            revision: 2,
            labels: vec![label],
            properties: vec![(property, ScalarValue::String(Arc::from("first")))],
        }))?;
        let mut rejected = IndexCatalog::default();
        let error = rejected
            .create(&duplicated, definition.clone())
            .expect_err("constraint accepted duplicate existing values");
        assert_eq!(error.code, ErrorCode::TransactionConflict);

        let mut graph = graph;
        let mut catalog = IndexCatalog::default();
        catalog.create(&graph, definition)?;
        assert!(catalog.has_unique_constraint(label, property));
        let mutation = GraphMutation::InsertNode(NodeInput {
            id: NodeId(8),
            layer: Layer::Observed,
            revision: 2,
            labels: vec![label],
            properties: vec![(property, ScalarValue::String(Arc::from("first")))],
        });
        catalog.before_graph_apply(&graph, &mutation)?;
        graph.apply(mutation.clone())?;
        let error = catalog
            .after_graph_apply(&graph, &mutation)
            .expect_err("constraint accepted a duplicate mutation");
        assert_eq!(error.code, ErrorCode::TransactionConflict);
        assert!(catalog.drop_index("document_text_unique").is_err());
        catalog.drop_constraint("document_text_unique")?;
        Ok(())
    }

    #[test]
    fn index_catalog_lifecycle_survives_checkpoint_roundtrip() -> Result<()> {
        let (mut graph, label, source, _) = indexed_graph()?;
        let mut indexes = IndexCatalog::default();
        indexes.create(
            &graph,
            GraphIndexDefinition {
                name: "text_lookup".to_owned(),
                kind: GraphIndexKind::Text,
                label,
                properties: vec![source],
                unique: false,
            },
        )?;
        assert_eq!(
            indexes.statuses().next().map(|status| status.state),
            Some(DerivedIndexState::Online)
        );

        let checkpoint = postcard::to_stdvec(&indexes)
            .map_err(|error| Error::internal(format!("test encoding failed: {error}")))?;
        let mut restored: IndexCatalog = postcard::from_bytes(&checkpoint)
            .map_err(|error| Error::internal(format!("test decoding failed: {error}")))?;
        let mutation = GraphMutation::SetNodeProperty {
            node: NodeId(7),
            property: source,
            value: ScalarValue::String(Arc::from("updated graph memory")),
            revision: 2,
        };
        restored.before_graph_apply(&graph, &mutation)?;
        graph.apply(mutation.clone())?;
        restored.after_graph_apply(&graph, &mutation)?;
        restored.rebuild(&graph, "text_lookup")?;
        assert_eq!(
            restored.statuses().next().map(|status| status.state),
            Some(DerivedIndexState::Online)
        );
        restored.drop_index("text_lookup")?;
        assert!(restored.statuses().next().is_none());
        Ok(())
    }

    #[test]
    fn catalog_selects_exact_online_scalar_posting() -> Result<()> {
        let (graph, label, source, _) = indexed_graph()?;
        let mut indexes = IndexCatalog::default();
        indexes.create(
            &graph,
            GraphIndexDefinition {
                name: "document_text".to_owned(),
                kind: GraphIndexKind::Equality,
                label,
                properties: vec![source],
                unique: false,
            },
        )?;
        let rows = indexes
            .equality_candidates(
                label,
                &BTreeMap::from([(source, ScalarValue::String(Arc::from("first")))]),
            )?
            .ok_or_else(|| Error::internal("ONLINE equality posting was not selected"))?;
        assert_eq!(rows, vec![0]);
        assert_eq!(
            indexes.equality_candidates(
                label,
                &BTreeMap::from([(source, ScalarValue::String(Arc::from("missing")))])
            )?,
            Some(Vec::new())
        );
        Ok(())
    }
}
