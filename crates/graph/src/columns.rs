//! Typed structure-of-arrays columns and compact validity bitmaps.

use std::{
    collections::{BTreeMap, BTreeSet},
    sync::{Arc, OnceLock},
};

use ordered_float::OrderedFloat;
use serde::{Deserialize, Deserializer, Serialize, Serializer, ser::SerializeSeq};

use crate::{DocumentItem, Error, ErrorCode, Result, ScalarValue, types::PropertyId};

use super::{
    persistent::{ArenaSpan, PagedArena, PagedVec, PersistentMap},
    shared::SharedFlat,
};

use crate::document::{DocumentRoot, validate_canonical};

pub const DOCUMENT_SEAL_MIN_GARBAGE_BYTES: usize = 64 * 1_024;
const DOCUMENT_SEAL_LIVE_FRACTION: usize = 4;

const MIXED_BOOLEAN: u8 = 1;
const MIXED_INTEGER: u8 = 2;
const MIXED_FLOAT: u8 = 3;
const MIXED_STRING: u8 = 4;
pub const MIXED_STRING_TAG: u8 = MIXED_STRING;
const MIXED_BYTES: u8 = 5;
const MIXED_DATE: u8 = 6;
const MIXED_LOCAL_TIME: u8 = 7;
const MIXED_ZONED_TIME: u8 = 8;
const MIXED_LOCAL_DATETIME: u8 = 9;
const MIXED_ZONED_DATETIME: u8 = 10;
const MIXED_DURATION: u8 = 11;
const MIXED_LIST: u8 = 12;
const MIXED_MAP: u8 = 13;
const MIXED_FULL_VALIDITY_WORD_MARKER: u64 = 0x4d49_5845_445f_434f;

fn replace_paged<T: Clone>(values: &mut PagedVec<T>, row: usize, value: T) -> Result<()> {
    values
        .replace(row, value)
        .map_err(|message| Error::new(ErrorCode::CorruptStorage, message))
}

/// A compact null bitmap. A set bit means the corresponding value is present.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Validity {
    words: PagedVec<u64>,
    len: usize,
}

impl Validity {
    #[cfg_attr(
        not(all(feature = "accelerator", any(target_os = "macos", target_os = "ios"))),
        allow(dead_code)
    )]
    pub fn rebase_shared_words(&mut self, words: SharedFlat<u64>) -> Result<()> {
        let expected = self.len.div_ceil(u64::BITS as usize);
        if words.len() != expected {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "shared validity word count does not match its row count",
            ));
        }
        self.words
            .rebase_shared(words)
            .map_err(|message| Error::new(ErrorCode::CorruptStorage, message))
    }

    #[cfg_attr(
        not(all(feature = "accelerator", any(target_os = "macos", target_os = "ios"))),
        allow(dead_code)
    )]
    pub fn words(&self) -> impl Iterator<Item = u64> + '_ {
        self.words.iter().copied()
    }

    pub fn push(&mut self, present: bool) -> Result<()> {
        let bit = self.len % u64::BITS as usize;
        if bit == 0 {
            self.words.push(u64::from(present) << bit);
        } else {
            let word = self.words.len().checked_sub(1).ok_or_else(|| {
                Error::new(
                    ErrorCode::CorruptStorage,
                    "validity bitmap is missing its active word",
                )
            })?;
            let current = self.words.get(word).copied().ok_or_else(|| {
                Error::new(
                    ErrorCode::CorruptStorage,
                    "validity bitmap is missing its active word",
                )
            })?;
            if present {
                replace_paged(&mut self.words, word, current | (1_u64 << bit))?;
            }
        }
        self.len += 1;
        Ok(())
    }

    pub fn set(&mut self, row: usize, present: bool) -> Result<()> {
        if row >= self.len {
            return Err(Error::invalid_data("column row is out of bounds"));
        }
        let word = row / u64::BITS as usize;
        let mask = 1_u64 << (row % u64::BITS as usize);
        let current = self.words.get(word).copied().ok_or_else(|| {
            Error::new(
                ErrorCode::CorruptStorage,
                "validity bitmap row has no backing word",
            )
        })?;
        let updated = if present {
            current | mask
        } else {
            current & !mask
        };
        replace_paged(&mut self.words, word, updated)
    }

    #[must_use]
    pub fn is_present(&self, row: usize) -> bool {
        (row < self.len)
            && self
                .words
                .get(row / u64::BITS as usize)
                .is_some_and(|word| word & (1_u64 << (row % u64::BITS as usize)) != 0)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Byte-per-row mask used directly by accelerator kernels during project admission.
    #[must_use]
    pub fn to_byte_mask(&self) -> Vec<u8> {
        (0..self.len)
            .map(|row| u8::from(self.is_present(row)))
            .collect()
    }

    fn validate(&self) -> Result<()> {
        let expected_words = self.len.div_ceil(u64::BITS as usize);
        if self.words.len() != expected_words {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "validity word count does not match its row count",
            ));
        }
        let trailing_bits = self.len % u64::BITS as usize;
        if trailing_bits != 0
            && self
                .words
                .get(expected_words.saturating_sub(1))
                .is_some_and(|word| *word >> trailing_bits != 0)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "validity contains non-canonical trailing bits",
            ));
        }
        Ok(())
    }

    fn mark_mixed_wire(&mut self) -> Result<()> {
        self.validate()?;
        let expected_words = self.len.div_ceil(u64::BITS as usize);
        let trailing_bits = self.len % u64::BITS as usize;
        if trailing_bits == 0 {
            self.words.push(MIXED_FULL_VALIDITY_WORD_MARKER);
        } else {
            let word = expected_words.saturating_sub(1);
            let current = self.words.get(word).copied().ok_or_else(|| {
                Error::new(
                    ErrorCode::CorruptStorage,
                    "mixed validity marker has no backing word",
                )
            })?;
            replace_paged(&mut self.words, word, current | (1_u64 << 63))?;
        }
        Ok(())
    }

    fn take_mixed_wire_marker(&mut self) -> Result<bool> {
        let expected_words = self.len.div_ceil(u64::BITS as usize);
        let trailing_bits = self.len % u64::BITS as usize;
        if trailing_bits == 0 {
            if self.words.len() != expected_words.saturating_add(1)
                || self.words.get(expected_words).copied() != Some(MIXED_FULL_VALIDITY_WORD_MARKER)
            {
                return Ok(false);
            }
            let retained = self
                .words
                .iter()
                .take(expected_words)
                .copied()
                .collect::<Vec<_>>();
            self.words = PagedVec::default();
            for word in retained {
                self.words.push(word);
            }
            return Ok(true);
        }
        if self.words.len() != expected_words {
            return Ok(false);
        }
        let word = expected_words.saturating_sub(1);
        let current = self.words.get(word).copied().ok_or_else(|| {
            Error::new(
                ErrorCode::CorruptStorage,
                "mixed validity marker has no backing word",
            )
        })?;
        if current & (1_u64 << 63) == 0 {
            return Ok(false);
        }
        replace_paged(&mut self.words, word, current & !(1_u64 << 63))?;
        Ok(true)
    }

    #[cfg(test)]
    fn detached_page_bytes_from(&self, previous: &Self) -> usize {
        self.words.detached_page_bytes_from(&previous.words)
    }
}

/// Append-only dictionary used by repeated strings and schema names. UTF-8 bytes use the same
/// flat offsets/value backing on CPU and Metal; the lookup contains only compact numeric IDs.
#[derive(Clone, Debug, Default)]
pub struct Dictionary {
    values: PackedLists<u8>,
    lookup: PersistentMap<Vec<u32>>,
}

/// Existing checkpoint shape. Serialization deliberately remains a string sequence plus the
/// historical lookup fields even though runtime lookup no longer duplicates string allocations.
#[derive(Serialize, Deserialize)]
struct DictionaryWire {
    values: Vec<Arc<str>>,
    lookup: Arc<BTreeMap<Arc<str>, u32>>,
    #[serde(default)]
    lookup_deltas: PersistentMap<Vec<(Arc<str>, u32)>>,
}

impl Dictionary {
    pub fn intern(&mut self, value: &str) -> Result<u32> {
        if let Some(id) = self.id(value) {
            return Ok(id);
        }
        let id = u32::try_from(self.values.rows()).map_err(|_| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "string dictionary exhausted",
            )
        })?;
        self.values.push(value.as_bytes().iter().copied())?;
        let hash = dictionary_key_hash(value);
        let mut bucket = self.lookup.get(hash).cloned().unwrap_or_default();
        bucket.push(id);
        self.lookup.insert(hash, bucket);
        Ok(id)
    }

    #[must_use]
    fn id(&self, value: &str) -> Option<u32> {
        self.lookup
            .get(dictionary_key_hash(value))
            .and_then(|bucket| bucket.iter().find(|id| self.resolve(**id) == Some(value)))
            .copied()
    }

    #[must_use]
    pub fn resolve(&self, id: u32) -> Option<&str> {
        std::str::from_utf8(self.values.get(id)?).ok()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.values.rows()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.values.rows() == 0
    }

    /// Stable dictionary entries in numeric-ID order.
    pub fn values(&self) -> impl Iterator<Item = &str> {
        (0..self.values.rows()).filter_map(|id| self.resolve(id as u32))
    }

    #[cfg_attr(
        not(all(feature = "accelerator", any(target_os = "macos", target_os = "ios"))),
        allow(dead_code)
    )]
    pub fn offsets(&self) -> &[u32] {
        self.values.offsets()
    }

    #[cfg_attr(
        not(all(feature = "accelerator", any(target_os = "macos", target_os = "ios"))),
        allow(dead_code)
    )]
    pub fn bytes(&self) -> &[u8] {
        self.values.values()
    }

    #[must_use]
    pub fn estimated_bytes(&self) -> usize {
        self.values.estimated_bytes()
    }

    #[cfg(test)]
    fn detached_storage_bytes_from(&self, previous: &Self) -> usize {
        self.values
            .estimated_bytes()
            .saturating_sub(previous.values.estimated_bytes())
            .saturating_add(self.lookup.detached_node_bytes_from(&previous.lookup))
    }

    #[cfg_attr(
        not(all(feature = "accelerator", any(target_os = "macos", target_os = "ios"))),
        allow(dead_code)
    )]
    pub fn rebase_shared(
        &mut self,
        offsets: SharedFlat<u32>,
        values: SharedFlat<u8>,
    ) -> Result<()> {
        self.values.rebase_shared(offsets, values)
    }
}

impl Serialize for Dictionary {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let values = self.values().map(Arc::<str>::from).collect::<Vec<_>>();
        let lookup = values
            .iter()
            .enumerate()
            .map(|(id, value)| (Arc::clone(value), id as u32))
            .collect::<BTreeMap<_, _>>();
        DictionaryWire {
            values,
            lookup: Arc::new(lookup),
            lookup_deltas: PersistentMap::default(),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for Dictionary {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = DictionaryWire::deserialize(deserializer)?;
        let mut dictionary = Self::default();
        for value in wire.values {
            dictionary
                .intern(&value)
                .map_err(serde::de::Error::custom)?;
        }
        Ok(dictionary)
    }
}

fn dictionary_key_hash(value: &str) -> u128 {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"dictionary\0");
    hasher.update(value.as_bytes());
    let digest = hasher.finalize();
    let mut prefix = [0_u8; 16];
    prefix.copy_from_slice(&digest.as_bytes()[..16]);
    u128::from_le_bytes(prefix)
}

/// Offset/child storage for compact variable-width row lists.
#[derive(Debug)]
pub struct PackedLists<T> {
    /// Immutable accelerator-owned base. Mutations never rewrite it; they append into `arena`
    /// and replace only one bounded row descriptor in `overrides`.
    base: Option<SharedPackedBase<T>>,
    spans: PagedVec<ArenaSpan>,
    arena: PagedArena<T>,
    overrides: PersistentMap<ArenaSpan>,
    live_values: usize,
    flat: OnceLock<FlatLists<T>>,
}

#[derive(Clone, Debug)]
struct SharedPackedBase<T> {
    offsets: SharedFlat<u32>,
    values: SharedFlat<T>,
}

#[derive(Debug)]
struct FlatLists<T> {
    offsets: Vec<u32>,
    values: Vec<T>,
}

#[derive(Serialize, Deserialize)]
struct PackedListsWire<T> {
    offsets: Vec<u32>,
    values: Vec<T>,
}

impl<T> Default for PackedLists<T> {
    fn default() -> Self {
        Self {
            base: None,
            spans: PagedVec::default(),
            arena: PagedArena::default(),
            overrides: PersistentMap::default(),
            live_values: 0,
            flat: OnceLock::new(),
        }
    }
}

impl<T: Clone> Clone for PackedLists<T> {
    fn clone(&self) -> Self {
        Self {
            base: self.base.clone(),
            spans: self.spans.clone(),
            arena: self.arena.clone(),
            overrides: self.overrides.clone(),
            live_values: self.live_values,
            flat: OnceLock::new(),
        }
    }
}

impl<T: Clone + PartialEq> PartialEq for PackedLists<T> {
    fn eq(&self, other: &Self) -> bool {
        self.rows() == other.rows()
            && (0..self.rows()).all(|row| self.get(row as u32) == other.get(row as u32))
    }
}

impl<T: Clone + Eq> Eq for PackedLists<T> {}

impl<T: Clone> PackedLists<T> {
    /// Rebinds the complete immutable row/value base to accelerator-owned flat allocations.
    /// Checkpoint serialization remains the same offsets-plus-values wire format.
    #[cfg_attr(
        not(all(feature = "accelerator", any(target_os = "macos", target_os = "ios"))),
        allow(dead_code)
    )]
    pub fn rebase_shared(&mut self, offsets: SharedFlat<u32>, values: SharedFlat<T>) -> Result<()> {
        if offsets.as_slice().first().copied() != Some(0)
            || offsets
                .as_slice()
                .last()
                .copied()
                .map(|offset| offset as usize)
                != Some(values.len())
            || offsets.as_slice().windows(2).any(|pair| pair[0] > pair[1])
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "shared packed-list offsets are invalid",
            ));
        }
        self.base = Some(SharedPackedBase { offsets, values });
        self.spans = PagedVec::default();
        self.arena = PagedArena::default();
        self.overrides = PersistentMap::default();
        self.live_values = self.base.as_ref().map_or(0, |base| base.values.len());
        self.flat = OnceLock::new();
        Ok(())
    }

    /// Seals appended rows into longer views of the same shared offset/value allocations. The
    /// admitted base prefix is immutable, so only the boundary and appended offsets require new
    /// structural validation; work tracks appended rows rather than all unrelated labels.
    pub fn rebase_shared_extension(
        &mut self,
        offsets: SharedFlat<u32>,
        values: SharedFlat<T>,
    ) -> Result<()> {
        let Some(base) = self.base.as_ref() else {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "shared packed-list extension has no base prefix",
            ));
        };
        let previous_offset_len = base.offsets.len();
        let previous_value_len = base.values.len();
        let offset_slice = offsets.as_slice();
        if previous_offset_len == 0
            || offset_slice.len() <= previous_offset_len
            || offset_slice.len() != self.rows().saturating_add(1)
            || values.len() != self.live_values
            || !base.offsets.is_prefix_of(&offsets)
            || !base.values.is_prefix_of(&values)
            || offset_slice.get(previous_offset_len - 1).copied()
                != u32::try_from(previous_value_len).ok()
            || offset_slice[previous_offset_len - 1..]
                .windows(2)
                .any(|pair| pair[0] > pair[1])
            || offset_slice.last().copied().map(|offset| offset as usize) != Some(values.len())
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "shared packed-list extension is invalid",
            ));
        }
        self.base = Some(SharedPackedBase { offsets, values });
        self.spans = PagedVec::default();
        self.arena = PagedArena::default();
        self.overrides = PersistentMap::default();
        self.live_values = self.base.as_ref().map_or(0, |base| base.values.len());
        self.flat = OnceLock::new();
        Ok(())
    }

    fn base_rows(&self) -> usize {
        self.base
            .as_ref()
            .and_then(|base| base.offsets.len().checked_sub(1))
            .unwrap_or(0)
    }

    fn shared_base_is_pristine(&self) -> bool {
        self.base.is_some() && self.spans.is_empty() && self.overrides.len() == 0
    }

    pub fn validate_push_len(&self, additional: usize) -> Result<()> {
        let end = self
            .live_values
            .checked_add(additional)
            .ok_or_else(|| Error::new(ErrorCode::ResultBudgetExceeded, "packed list exhausted"))?;
        u32::try_from(end)
            .map(|_| ())
            .map_err(|_| Error::new(ErrorCode::ResultBudgetExceeded, "packed list exhausted"))
    }

    pub fn push<I>(&mut self, values: I) -> Result<()>
    where
        I: IntoIterator<Item = T>,
    {
        let values = values.into_iter().collect::<Vec<_>>();
        self.validate_push_len(values.len())?;
        let len = values.len();
        let span = self
            .arena
            .append(values)
            .map_err(|message| Error::new(ErrorCode::ResultBudgetExceeded, message))?;
        self.spans.push(span);
        self.live_values += len;
        self.flat = OnceLock::new();
        Ok(())
    }

    /// Replaces one variable-width row while preserving every row ordinal.
    /// Capacity and encoded-offset bounds are admitted before either vector is modified.
    pub fn replace<I>(&mut self, row: u32, values: I) -> Result<()>
    where
        I: IntoIterator<Item = T>,
    {
        let row = row as usize;
        let base_rows = self.base_rows();
        let previous = if row < base_rows {
            self.overrides
                .get(row as u128)
                .copied()
                .map(|span| span.len as usize)
                .or_else(|| {
                    let base = self.base.as_ref()?;
                    let offsets = base.offsets.as_slice();
                    Some((offsets[row + 1] - offsets[row]) as usize)
                })
                .ok_or_else(|| Error::invalid_data("packed-list row is out of bounds"))?
        } else {
            self.spans
                .get(row - base_rows)
                .map(|span| span.len as usize)
                .ok_or_else(|| Error::invalid_data("packed-list row is out of bounds"))?
        };
        let replacement = values.into_iter().collect::<Vec<_>>();
        let replacement_len = replacement.len();
        let final_len = self
            .live_values
            .checked_sub(previous)
            .and_then(|length| length.checked_add(replacement_len))
            .ok_or_else(|| Error::new(ErrorCode::ResultBudgetExceeded, "packed list exhausted"))?;
        u32::try_from(final_len)
            .map_err(|_| Error::new(ErrorCode::ResultBudgetExceeded, "packed list exhausted"))?;
        let span = self
            .arena
            .append(replacement)
            .map_err(|message| Error::new(ErrorCode::ResultBudgetExceeded, message))?;
        if row < base_rows {
            self.overrides.insert(row as u128, span);
        } else {
            self.spans[row - base_rows] = span;
        }
        self.live_values = final_len;
        self.flat = OnceLock::new();
        Ok(())
    }

    #[must_use]
    pub fn get(&self, row: u32) -> Option<&[T]> {
        let row = row as usize;
        let base_rows = self.base_rows();
        if row < base_rows {
            if let Some(span) = self.overrides.get(row as u128).copied() {
                return self.arena.get(span);
            }
            let base = self.base.as_ref()?;
            let offsets = base.offsets.as_slice();
            let start = offsets[row] as usize;
            let end = offsets[row + 1] as usize;
            return base.values.as_slice().get(start..end);
        }
        self.arena.get(*self.spans.get(row - base_rows)?)
    }

    #[must_use]
    pub fn rows(&self) -> usize {
        self.base_rows().saturating_add(self.spans.len())
    }

    #[must_use]
    pub fn offsets(&self) -> &[u32] {
        if self.shared_base_is_pristine()
            && let Some(base) = self.base.as_ref()
        {
            return base.offsets.as_slice();
        }
        &self.flattened().offsets
    }

    #[must_use]
    pub fn values(&self) -> &[T] {
        if self.shared_base_is_pristine()
            && let Some(base) = self.base.as_ref()
        {
            return base.values.as_slice();
        }
        &self.flattened().values
    }

    #[must_use]
    pub fn estimated_bytes(&self) -> usize {
        let base = self.base.as_ref().map_or(0, |base| {
            base.offsets
                .len()
                .saturating_mul(size_of::<u32>())
                .saturating_add(base.values.len().saturating_mul(size_of::<T>()))
        });
        base.saturating_add(
            self.spans
                .len()
                .saturating_mul(size_of::<ArenaSpan>())
                .saturating_add(self.arena.estimated_bytes()),
        )
    }

    #[must_use]
    fn garbage_bytes(&self) -> usize {
        let delta_live = self
            .spans
            .iter()
            .map(|span| span.len as usize)
            .chain(self.overrides.iter().map(|(_, span)| span.len as usize))
            .fold(0_usize, usize::saturating_add);
        self.arena
            .elements()
            .saturating_sub(delta_live)
            .saturating_mul(size_of::<T>())
    }

    #[must_use]
    fn needs_seal(&self) -> bool {
        let garbage = self.garbage_bytes();
        let live_threshold = self
            .live_values
            .saturating_mul(size_of::<T>())
            .div_ceil(DOCUMENT_SEAL_LIVE_FRACTION);
        garbage >= DOCUMENT_SEAL_MIN_GARBAGE_BYTES && garbage >= live_threshold
    }

    fn seal(&mut self, validity: &Validity) -> Result<()> {
        if validity.len() != self.rows() {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "document validity and packed rows differ",
            ));
        }
        let mut sealed = Self::default();
        for row in 0..self.rows() {
            if validity.is_present(row) {
                let values = self.get(row as u32).ok_or_else(|| {
                    Error::new(ErrorCode::CorruptStorage, "document row span is invalid")
                })?;
                sealed.push(values.iter().cloned())?;
            } else {
                sealed.push(std::iter::empty())?;
            }
        }
        *self = sealed;
        Ok(())
    }

    fn flattened(&self) -> &FlatLists<T> {
        self.flat.get_or_init(|| {
            let mut offsets = Vec::with_capacity(self.rows().saturating_add(1));
            let mut values = Vec::with_capacity(self.live_values);
            offsets.push(0);
            for row in 0..self.rows() {
                if let Some(row_values) = self.get(row as u32) {
                    values.extend_from_slice(row_values);
                }
                offsets.push(values.len() as u32);
            }
            FlatLists { offsets, values }
        })
    }
}

impl<T: Clone + Serialize> Serialize for PackedLists<T> {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        PackedListsWire {
            offsets: self.offsets().to_vec(),
            values: self.values().to_vec(),
        }
        .serialize(serializer)
    }
}

impl<'de, T> Deserialize<'de> for PackedLists<T>
where
    T: Clone + Deserialize<'de>,
{
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let wire = PackedListsWire::<T>::deserialize(deserializer)?;
        if wire.offsets.first().copied() != Some(0)
            || wire.offsets.last().copied().map(|value| value as usize) != Some(wire.values.len())
            || wire.offsets.windows(2).any(|pair| pair[0] > pair[1])
        {
            return Err(serde::de::Error::custom("packed-list offsets are invalid"));
        }
        let mut lists = Self::default();
        for range in wire.offsets.windows(2) {
            lists
                .push(
                    wire.values[range[0] as usize..range[1] as usize]
                        .iter()
                        .cloned(),
                )
                .map_err(serde::de::Error::custom)?;
        }
        Ok(lists)
    }
}

/// One flat byte range per row with checkpoint-compatible sequence serialization.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ByteValues(PackedLists<u8>);

impl ByteValues {
    fn validate_push_len(&self, additional: usize) -> Result<()> {
        self.0.validate_push_len(additional)
    }

    fn push(&mut self, value: &[u8]) -> Result<()> {
        self.0.push(value.iter().copied())
    }

    fn replace(&mut self, row: u32, value: &[u8]) -> Result<()> {
        self.0.replace(row, value.iter().copied())
    }

    #[must_use]
    pub fn get(&self, row: usize) -> Option<&[u8]> {
        self.0.get(row as u32)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.0.rows()
    }

    pub fn iter(&self) -> impl Iterator<Item = &[u8]> {
        (0..self.len()).filter_map(|row| self.get(row))
    }

    #[cfg_attr(
        not(all(feature = "accelerator", any(target_os = "macos", target_os = "ios"))),
        allow(dead_code)
    )]
    pub fn offsets(&self) -> &[u32] {
        self.0.offsets()
    }

    #[cfg_attr(
        not(all(feature = "accelerator", any(target_os = "macos", target_os = "ios"))),
        allow(dead_code)
    )]
    pub fn values(&self) -> &[u8] {
        self.0.values()
    }

    #[cfg_attr(
        not(all(feature = "accelerator", any(target_os = "macos", target_os = "ios"))),
        allow(dead_code)
    )]
    pub fn rebase_shared(
        &mut self,
        offsets: SharedFlat<u32>,
        values: SharedFlat<u8>,
    ) -> Result<()> {
        self.0.rebase_shared(offsets, values)
    }

    #[must_use]
    fn estimated_bytes(&self) -> usize {
        self.0.estimated_bytes()
    }

    fn needs_seal(&self) -> bool {
        self.0.needs_seal()
    }

    fn seal(&mut self) -> Result<()> {
        let mut sealed = Self::default();
        for row in 0..self.len() {
            let value = self.get(row).ok_or_else(|| {
                Error::new(ErrorCode::CorruptStorage, "byte column row span is invalid")
            })?;
            sealed.push(value)?;
        }
        *self = sealed;
        Ok(())
    }

    #[cfg(test)]
    fn storage_bytes(&self) -> (usize, usize) {
        (
            self.0.arena.estimated_bytes(),
            self.0.live_values.saturating_mul(size_of::<u8>()),
        )
    }

    #[cfg(test)]
    fn detached_storage_bytes_from(&self, previous: &Self) -> usize {
        self.0
            .spans
            .detached_page_bytes_from(&previous.0.spans)
            .saturating_add(
                self.0
                    .arena
                    .estimated_bytes()
                    .saturating_sub(previous.0.arena.estimated_bytes()),
            )
            .saturating_add(
                self.0
                    .overrides
                    .detached_node_bytes_from(&previous.0.overrides),
            )
    }
}

impl Serialize for ByteValues {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut sequence = serializer.serialize_seq(Some(self.len()))?;
        for value in self.iter() {
            sequence.serialize_element(value)?;
        }
        sequence.end()
    }
}

impl<'de> Deserialize<'de> for ByteValues {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let rows = Vec::<Vec<u8>>::deserialize(deserializer)?;
        let mut values = Self::default();
        for row in rows {
            values.push(&row).map_err(serde::de::Error::custom)?;
        }
        Ok(values)
    }
}

fn mixed_corrupt(message: &'static str) -> Error {
    Error::new(ErrorCode::CorruptStorage, message)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PropertyListScalarKind {
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
}

fn invalid_property_type(message: &'static str) -> Error {
    Error::new(
        ErrorCode::QueryType,
        format!("InvalidPropertyType: {message}"),
    )
}

fn property_list_scalar_kind(value: &ScalarValue) -> Result<PropertyListScalarKind> {
    match value {
        ScalarValue::Boolean(_) => Ok(PropertyListScalarKind::Boolean),
        ScalarValue::Integer(_) => Ok(PropertyListScalarKind::Integer),
        ScalarValue::Float(_) => Ok(PropertyListScalarKind::Float),
        ScalarValue::String(_) => Ok(PropertyListScalarKind::String),
        ScalarValue::Bytes(_) => Ok(PropertyListScalarKind::Bytes),
        ScalarValue::Date(_) => Ok(PropertyListScalarKind::Date),
        ScalarValue::LocalTime(_) => Ok(PropertyListScalarKind::LocalTime),
        ScalarValue::ZonedTime { .. } => Ok(PropertyListScalarKind::ZonedTime),
        ScalarValue::LocalDateTime { .. } => Ok(PropertyListScalarKind::LocalDateTime),
        ScalarValue::ZonedDateTime { .. } => Ok(PropertyListScalarKind::ZonedDateTime),
        ScalarValue::Duration { .. } => Ok(PropertyListScalarKind::Duration),
        ScalarValue::Null => Err(invalid_property_type("property lists cannot contain NULL")),
        ScalarValue::List(_) | ScalarValue::Map(_) => Err(invalid_property_type(
            "property lists cannot contain nested collections",
        )),
    }
}

fn validate_property_value_shape(value: &ScalarValue) -> Result<()> {
    match value {
        ScalarValue::List(value) => {
            let mut element_kind = None;
            for item in value.items()? {
                let DocumentItem::Scalar(item) = item else {
                    return Err(invalid_property_type(
                        "property lists must be flat homogeneous lists of scalar values",
                    ));
                };
                let kind = property_list_scalar_kind(&item)?;
                if element_kind.is_some_and(|existing| existing != kind) {
                    return Err(invalid_property_type(
                        "property lists must contain values of one scalar type",
                    ));
                }
                element_kind = Some(kind);
            }
            Ok(())
        }
        _ => Ok(()),
    }
}

fn mixed_payload(value: &ScalarValue) -> Result<Option<Vec<u8>>> {
    let mut payload = Vec::new();
    match value {
        ScalarValue::Null => return Ok(None),
        ScalarValue::Boolean(value) => {
            payload.push(MIXED_BOOLEAN);
            payload.push(u8::from(*value));
        }
        ScalarValue::Integer(value) => {
            payload.push(MIXED_INTEGER);
            payload.extend_from_slice(&value.to_le_bytes());
        }
        ScalarValue::Float(value) => {
            payload.push(MIXED_FLOAT);
            payload.extend_from_slice(&value.into_inner().to_bits().to_le_bytes());
        }
        ScalarValue::String(value) => {
            payload.push(MIXED_STRING);
            payload.extend_from_slice(value.as_bytes());
        }
        ScalarValue::Bytes(value) => {
            payload.push(MIXED_BYTES);
            payload.extend_from_slice(value);
        }
        ScalarValue::Date(value) => {
            payload.push(MIXED_DATE);
            payload.extend_from_slice(&value.to_le_bytes());
        }
        ScalarValue::LocalTime(value) => {
            payload.push(MIXED_LOCAL_TIME);
            payload.extend_from_slice(&value.to_le_bytes());
        }
        ScalarValue::ZonedTime {
            nanos,
            offset_seconds,
        } => {
            payload.push(MIXED_ZONED_TIME);
            payload.extend_from_slice(&nanos.to_le_bytes());
            payload.extend_from_slice(&offset_seconds.to_le_bytes());
        }
        ScalarValue::LocalDateTime { seconds, nanos } => {
            payload.push(MIXED_LOCAL_DATETIME);
            payload.extend_from_slice(&seconds.to_le_bytes());
            payload.extend_from_slice(&nanos.to_le_bytes());
        }
        ScalarValue::ZonedDateTime {
            seconds,
            nanos,
            timezone,
        } => {
            payload.push(MIXED_ZONED_DATETIME);
            payload.extend_from_slice(&seconds.to_le_bytes());
            payload.extend_from_slice(&nanos.to_le_bytes());
            payload.extend_from_slice(timezone.as_bytes());
        }
        ScalarValue::Duration {
            months,
            days,
            seconds,
            nanos,
        } => {
            payload.push(MIXED_DURATION);
            payload.extend_from_slice(&months.to_le_bytes());
            payload.extend_from_slice(&days.to_le_bytes());
            payload.extend_from_slice(&seconds.to_le_bytes());
            payload.extend_from_slice(&nanos.to_le_bytes());
        }
        ScalarValue::List(value) => {
            validate_canonical(value.as_bytes(), DocumentRoot::List)?;
            payload.push(MIXED_LIST);
            payload.extend_from_slice(value.as_bytes());
        }
        ScalarValue::Map(value) => {
            validate_canonical(value.as_bytes(), DocumentRoot::Map)?;
            payload.push(MIXED_MAP);
            payload.extend_from_slice(value.as_bytes());
        }
    }
    Ok(Some(payload))
}

fn mixed_exact<const N: usize>(payload: &[u8], message: &'static str) -> Result<[u8; N]> {
    payload.try_into().map_err(|_| mixed_corrupt(message))
}

fn mixed_value(payload: &[u8]) -> Result<ScalarValue> {
    let (tag, payload) = payload
        .split_first()
        .ok_or_else(|| mixed_corrupt("mixed property payload is empty"))?;
    match *tag {
        MIXED_BOOLEAN => match payload {
            [0] => Ok(ScalarValue::Boolean(false)),
            [1] => Ok(ScalarValue::Boolean(true)),
            _ => Err(mixed_corrupt("mixed Boolean payload is invalid")),
        },
        MIXED_INTEGER => Ok(ScalarValue::Integer(i64::from_le_bytes(mixed_exact(
            payload,
            "mixed integer payload is invalid",
        )?))),
        MIXED_FLOAT => Ok(ScalarValue::Float(OrderedFloat(f64::from_bits(
            u64::from_le_bytes(mixed_exact(payload, "mixed float payload is invalid")?),
        )))),
        MIXED_STRING => Ok(ScalarValue::String(Arc::from(
            std::str::from_utf8(payload)
                .map_err(|_| mixed_corrupt("mixed string payload is not UTF-8"))?,
        ))),
        MIXED_BYTES => Ok(ScalarValue::Bytes(Arc::from(payload))),
        MIXED_DATE => Ok(ScalarValue::Date(i64::from_le_bytes(mixed_exact(
            payload,
            "mixed date payload is invalid",
        )?))),
        MIXED_LOCAL_TIME => Ok(ScalarValue::LocalTime(i64::from_le_bytes(mixed_exact(
            payload,
            "mixed local-time payload is invalid",
        )?))),
        MIXED_ZONED_TIME => {
            if payload.len() != 12 {
                return Err(mixed_corrupt("mixed zoned-time payload is invalid"));
            }
            Ok(ScalarValue::ZonedTime {
                nanos: i64::from_le_bytes(mixed_exact(
                    &payload[..8],
                    "mixed zoned-time nanos are invalid",
                )?),
                offset_seconds: i32::from_le_bytes(mixed_exact(
                    &payload[8..],
                    "mixed zoned-time offset is invalid",
                )?),
            })
        }
        MIXED_LOCAL_DATETIME => {
            if payload.len() != 12 {
                return Err(mixed_corrupt("mixed local-datetime payload is invalid"));
            }
            Ok(ScalarValue::LocalDateTime {
                seconds: i64::from_le_bytes(mixed_exact(
                    &payload[..8],
                    "mixed local-datetime seconds are invalid",
                )?),
                nanos: u32::from_le_bytes(mixed_exact(
                    &payload[8..],
                    "mixed local-datetime nanos are invalid",
                )?),
            })
        }
        MIXED_ZONED_DATETIME => {
            if payload.len() < 12 {
                return Err(mixed_corrupt("mixed zoned-datetime payload is invalid"));
            }
            Ok(ScalarValue::ZonedDateTime {
                seconds: i64::from_le_bytes(mixed_exact(
                    &payload[..8],
                    "mixed zoned-datetime seconds are invalid",
                )?),
                nanos: u32::from_le_bytes(mixed_exact(
                    &payload[8..12],
                    "mixed zoned-datetime nanos are invalid",
                )?),
                timezone: Arc::from(
                    std::str::from_utf8(&payload[12..])
                        .map_err(|_| mixed_corrupt("mixed zoned-datetime timezone is not UTF-8"))?,
                ),
            })
        }
        MIXED_DURATION => {
            if payload.len() != 28 {
                return Err(mixed_corrupt("mixed duration payload is invalid"));
            }
            Ok(ScalarValue::Duration {
                months: i64::from_le_bytes(mixed_exact(
                    &payload[..8],
                    "mixed duration months are invalid",
                )?),
                days: i64::from_le_bytes(mixed_exact(
                    &payload[8..16],
                    "mixed duration days are invalid",
                )?),
                seconds: i64::from_le_bytes(mixed_exact(
                    &payload[16..24],
                    "mixed duration seconds are invalid",
                )?),
                nanos: i32::from_le_bytes(mixed_exact(
                    &payload[24..],
                    "mixed duration nanos are invalid",
                )?),
            })
        }
        MIXED_LIST => {
            validate_canonical(payload, DocumentRoot::List)?;
            Ok(ScalarValue::List(crate::DocumentList::from_canonical(
                Arc::from(payload),
            )))
        }
        MIXED_MAP => {
            validate_canonical(payload, DocumentRoot::Map)?;
            Ok(ScalarValue::Map(crate::DocumentMap::from_canonical(
                Arc::from(payload),
            )))
        }
        _ => Err(mixed_corrupt("mixed property tag is invalid")),
    }
}

fn validate_column_len(actual: usize, expected: usize, message: &'static str) -> Result<()> {
    if actual != expected {
        return Err(Error::new(ErrorCode::CorruptStorage, message));
    }
    Ok(())
}

/// A physically typed property column. Nulls live only in `validity`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum TypedColumn {
    Boolean {
        values: PagedVec<bool>,
        validity: Validity,
    },
    Integer {
        values: PagedVec<i64>,
        validity: Validity,
    },
    Float {
        values: PagedVec<OrderedFloat<f64>>,
        validity: Validity,
    },
    String {
        values: PagedVec<u32>,
        validity: Validity,
    },
    Bytes {
        values: ByteValues,
        validity: Validity,
    },
    Date {
        values: PagedVec<i64>,
        validity: Validity,
    },
    LocalTime {
        values: PagedVec<i64>,
        validity: Validity,
    },
    ZonedTime {
        nanos: PagedVec<i64>,
        offsets: PagedVec<i32>,
        validity: Validity,
    },
    LocalDateTime {
        seconds: PagedVec<i64>,
        nanos: PagedVec<u32>,
        validity: Validity,
    },
    ZonedDateTime {
        seconds: PagedVec<i64>,
        nanos: PagedVec<u32>,
        timezones: PagedVec<u32>,
        validity: Validity,
    },
    Duration {
        months: PagedVec<i64>,
        days: PagedVec<i64>,
        seconds: PagedVec<i64>,
        nanos: PagedVec<i32>,
        validity: Validity,
    },
    /// One canonical flat byte range per row; nested objects are not retained in host columns.
    List {
        values: PackedLists<u8>,
        validity: Validity,
    },
    /// One canonical flat byte range per row; lexical map-key order is encoded in each range.
    Map {
        values: PackedLists<u8>,
        validity: Validity,
    },
}

impl TypedColumn {
    pub fn accepts(&self, value: &ScalarValue) -> bool {
        matches!(value, ScalarValue::Null)
            || matches!(
                (self, value),
                (Self::Boolean { .. }, ScalarValue::Boolean(_))
                    | (Self::Integer { .. }, ScalarValue::Integer(_))
                    | (Self::Float { .. }, ScalarValue::Float(_))
                    | (Self::String { .. }, ScalarValue::String(_))
                    | (Self::Bytes { .. }, ScalarValue::Bytes(_))
                    | (Self::Date { .. }, ScalarValue::Date(_))
                    | (Self::LocalTime { .. }, ScalarValue::LocalTime(_))
                    | (Self::ZonedTime { .. }, ScalarValue::ZonedTime { .. })
                    | (
                        Self::LocalDateTime { .. },
                        ScalarValue::LocalDateTime { .. }
                    )
                    | (
                        Self::ZonedDateTime { .. },
                        ScalarValue::ZonedDateTime { .. }
                    )
                    | (Self::Duration { .. }, ScalarValue::Duration { .. })
                    | (Self::List { .. }, ScalarValue::List(_))
                    | (Self::Map { .. }, ScalarValue::Map(_))
            )
    }

    fn validate(&self, strings: &Dictionary) -> Result<()> {
        self.validity().validate()?;
        let rows = self.len();
        match self {
            Self::Boolean { values, .. } => validate_column_len(
                values.len(),
                rows,
                "Boolean values and validity have different row counts",
            )?,
            Self::Integer { values, .. } | Self::LocalTime { values, .. } => validate_column_len(
                values.len(),
                rows,
                "typed values and validity have different row counts",
            )?,
            Self::Float { values, .. } => validate_column_len(
                values.len(),
                rows,
                "float values and validity have different row counts",
            )?,
            Self::String { values, .. } => validate_column_len(
                values.len(),
                rows,
                "string values and validity have different row counts",
            )?,
            Self::Date { values, .. } => validate_column_len(
                values.len(),
                rows,
                "date values and validity have different row counts",
            )?,
            Self::Bytes { values, .. } => validate_column_len(
                values.len(),
                rows,
                "byte values and validity have different row counts",
            )?,
            Self::ZonedTime { nanos, offsets, .. } => {
                validate_column_len(
                    nanos.len(),
                    rows,
                    "zoned-time nanos and validity have different row counts",
                )?;
                validate_column_len(
                    offsets.len(),
                    rows,
                    "zoned-time offsets and validity have different row counts",
                )?;
            }
            Self::LocalDateTime { seconds, nanos, .. } => {
                validate_column_len(
                    seconds.len(),
                    rows,
                    "local-datetime seconds and validity have different row counts",
                )?;
                validate_column_len(
                    nanos.len(),
                    rows,
                    "local-datetime nanos and validity have different row counts",
                )?;
            }
            Self::ZonedDateTime {
                seconds,
                nanos,
                timezones,
                ..
            } => {
                validate_column_len(
                    seconds.len(),
                    rows,
                    "zoned-datetime seconds and validity have different row counts",
                )?;
                validate_column_len(
                    nanos.len(),
                    rows,
                    "zoned-datetime nanos and validity have different row counts",
                )?;
                validate_column_len(
                    timezones.len(),
                    rows,
                    "zoned-datetime timezones and validity have different row counts",
                )?;
            }
            Self::Duration {
                months,
                days,
                seconds,
                nanos,
                ..
            } => {
                validate_column_len(
                    months.len(),
                    rows,
                    "duration months and validity have different row counts",
                )?;
                validate_column_len(
                    days.len(),
                    rows,
                    "duration days and validity have different row counts",
                )?;
                validate_column_len(
                    seconds.len(),
                    rows,
                    "duration seconds and validity have different row counts",
                )?;
                validate_column_len(
                    nanos.len(),
                    rows,
                    "duration nanos and validity have different row counts",
                )?;
            }
            Self::List { values, .. } | Self::Map { values, .. } => validate_column_len(
                values.rows(),
                rows,
                "document values and validity have different row counts",
            )?,
        }
        match self {
            Self::String { values, validity } => {
                for row in 0..rows {
                    if validity.is_present(row)
                        && values
                            .get(row)
                            .and_then(|id| strings.resolve(*id))
                            .is_none()
                    {
                        return Err(Error::new(
                            ErrorCode::CorruptStorage,
                            "string property references an unknown dictionary entry",
                        ));
                    }
                }
            }
            Self::ZonedDateTime {
                timezones,
                validity,
                ..
            } => {
                for row in 0..rows {
                    if validity.is_present(row)
                        && timezones
                            .get(row)
                            .and_then(|id| strings.resolve(*id))
                            .is_none()
                    {
                        return Err(Error::new(
                            ErrorCode::CorruptStorage,
                            "zoned-datetime property references an unknown timezone",
                        ));
                    }
                }
            }
            Self::List { values, validity } => {
                for row in 0..rows {
                    let value = values.get(row as u32).ok_or_else(|| {
                        Error::new(ErrorCode::CorruptStorage, "list property row is absent")
                    })?;
                    if validity.is_present(row) {
                        validate_canonical(value, DocumentRoot::List)?;
                    } else if !value.is_empty() {
                        return Err(Error::new(
                            ErrorCode::CorruptStorage,
                            "NULL list property retains non-canonical bytes",
                        ));
                    }
                }
            }
            Self::Map { values, validity } => {
                for row in 0..rows {
                    let value = values.get(row as u32).ok_or_else(|| {
                        Error::new(ErrorCode::CorruptStorage, "map property row is absent")
                    })?;
                    if validity.is_present(row) {
                        validate_canonical(value, DocumentRoot::Map)?;
                    } else if !value.is_empty() {
                        return Err(Error::new(
                            ErrorCode::CorruptStorage,
                            "NULL map property retains non-canonical bytes",
                        ));
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn into_mixed(&self, strings: &Dictionary) -> Result<Self> {
        self.validate(strings)?;
        let mut values = ByteValues::default();
        let mut validity = Validity::default();
        for row in 0..self.len() {
            let value = self.get(row, strings).ok_or_else(|| {
                Error::new(
                    ErrorCode::CorruptStorage,
                    "typed property row cannot be materialized during mixed promotion",
                )
            })?;
            match mixed_payload(&value)? {
                Some(payload) => {
                    values.push(&payload)?;
                    validity.push(true)?;
                }
                None => {
                    values.push(&[])?;
                    validity.push(false)?;
                }
            }
        }
        Ok(Self::Bytes { values, validity })
    }

    fn mixed_validate(&self) -> Result<()> {
        let Self::Bytes { values, validity } = self else {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "mixed property is not backed by a tagged byte column",
            ));
        };
        validity.validate()?;
        validate_column_len(
            values.len(),
            validity.len(),
            "mixed values and validity have different row counts",
        )?;
        for row in 0..validity.len() {
            let payload = values.get(row).ok_or_else(|| {
                Error::new(ErrorCode::CorruptStorage, "mixed property row is absent")
            })?;
            if validity.is_present(row) {
                if matches!(mixed_value(payload)?, ScalarValue::Null) {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        "mixed property stores NULL as a present value",
                    ));
                }
            } else if !payload.is_empty() {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "NULL mixed property retains non-canonical bytes",
                ));
            }
        }
        Ok(())
    }

    fn mixed_validate_push(&self, value: &ScalarValue) -> Result<()> {
        let Self::Bytes { values, .. } = self else {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "mixed property is not backed by a tagged byte column",
            ));
        };
        let additional = mixed_payload(value)?.map_or(0, |payload| payload.len());
        values.validate_push_len(additional)
    }

    fn mixed_push(&mut self, value: &ScalarValue) -> Result<()> {
        let payload = mixed_payload(value)?;
        let Self::Bytes { values, validity } = self else {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "mixed property is not backed by a tagged byte column",
            ));
        };
        values.validate_push_len(payload.as_ref().map_or(0, Vec::len))?;
        match payload {
            Some(payload) => {
                values.push(&payload)?;
                validity.push(true)?;
            }
            None => {
                values.push(&[])?;
                validity.push(false)?;
            }
        }
        Ok(())
    }

    fn mixed_set(&mut self, row: usize, value: &ScalarValue) -> Result<()> {
        let payload = mixed_payload(value)?;
        let Self::Bytes { values, validity } = self else {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "mixed property is not backed by a tagged byte column",
            ));
        };
        if row >= validity.len() || row >= values.len() {
            return Err(Error::invalid_data("mixed property row is out of bounds"));
        }
        match payload {
            Some(payload) => {
                values.replace(row as u32, &payload)?;
                validity.set(row, true)?;
            }
            None => {
                values.replace(row as u32, &[])?;
                validity.set(row, false)?;
            }
        }
        Ok(())
    }

    fn mixed_get(&self, row: usize) -> Option<Result<ScalarValue>> {
        let Self::Bytes { values, validity } = self else {
            return Some(Err(Error::new(
                ErrorCode::CorruptStorage,
                "mixed property is not backed by a tagged byte column",
            )));
        };
        if row >= validity.len() || row >= values.len() {
            return None;
        }
        if !validity.is_present(row) {
            return Some(Ok(ScalarValue::Null));
        }
        Some(mixed_value(values.get(row)?))
    }

    fn with_value(
        value: &ScalarValue,
        preceding_nulls: usize,
        strings: &mut Dictionary,
    ) -> Result<Self> {
        let mut column = match value {
            ScalarValue::Null => {
                return Err(Error::new(
                    ErrorCode::QueryType,
                    "a property column type cannot be inferred from NULL",
                ));
            }
            ScalarValue::Boolean(_) => Self::Boolean {
                values: PagedVec::default(),
                validity: Validity::default(),
            },
            ScalarValue::Integer(_) => Self::Integer {
                values: PagedVec::default(),
                validity: Validity::default(),
            },
            ScalarValue::Float(_) => Self::Float {
                values: PagedVec::default(),
                validity: Validity::default(),
            },
            ScalarValue::String(_) => Self::String {
                values: PagedVec::default(),
                validity: Validity::default(),
            },
            ScalarValue::Bytes(_) => Self::Bytes {
                values: ByteValues::default(),
                validity: Validity::default(),
            },
            ScalarValue::Date(_) => Self::Date {
                values: PagedVec::default(),
                validity: Validity::default(),
            },
            ScalarValue::LocalTime(_) => Self::LocalTime {
                values: PagedVec::default(),
                validity: Validity::default(),
            },
            ScalarValue::ZonedTime { .. } => Self::ZonedTime {
                nanos: PagedVec::default(),
                offsets: PagedVec::default(),
                validity: Validity::default(),
            },
            ScalarValue::LocalDateTime { .. } => Self::LocalDateTime {
                seconds: PagedVec::default(),
                nanos: PagedVec::default(),
                validity: Validity::default(),
            },
            ScalarValue::ZonedDateTime { .. } => Self::ZonedDateTime {
                seconds: PagedVec::default(),
                nanos: PagedVec::default(),
                timezones: PagedVec::default(),
                validity: Validity::default(),
            },
            ScalarValue::Duration { .. } => Self::Duration {
                months: PagedVec::default(),
                days: PagedVec::default(),
                seconds: PagedVec::default(),
                nanos: PagedVec::default(),
                validity: Validity::default(),
            },
            ScalarValue::List(_) => Self::List {
                values: PackedLists::default(),
                validity: Validity::default(),
            },
            ScalarValue::Map(_) => Self::Map {
                values: PackedLists::default(),
                validity: Validity::default(),
            },
        };
        for _ in 0..preceding_nulls {
            column.push_null()?;
        }
        column.push(value, strings)?;
        Ok(column)
    }

    pub fn push_null(&mut self) -> Result<()> {
        match self {
            Self::Boolean { values, validity } => {
                values.push(false);
                validity.push(false)?;
            }
            Self::Integer { values, validity } => {
                values.push(0);
                validity.push(false)?;
            }
            Self::Float { values, validity } => {
                values.push(OrderedFloat(0.0));
                validity.push(false)?;
            }
            Self::String { values, validity } => {
                values.push(0);
                validity.push(false)?;
            }
            Self::Bytes { values, validity } => {
                values.push(&[])?;
                validity.push(false)?;
            }
            Self::Date { values, validity } => {
                values.push(0);
                validity.push(false)?;
            }
            Self::LocalTime { values, validity } => {
                values.push(0);
                validity.push(false)?;
            }
            Self::ZonedTime {
                nanos,
                offsets,
                validity,
            } => {
                nanos.push(0);
                offsets.push(0);
                validity.push(false)?;
            }
            Self::LocalDateTime {
                seconds,
                nanos,
                validity,
            } => {
                seconds.push(0);
                nanos.push(0);
                validity.push(false)?;
            }
            Self::ZonedDateTime {
                seconds,
                nanos,
                timezones,
                validity,
            } => {
                seconds.push(0);
                nanos.push(0);
                timezones.push(0);
                validity.push(false)?;
            }
            Self::Duration {
                months,
                days,
                seconds,
                nanos,
                validity,
            } => {
                months.push(0);
                days.push(0);
                seconds.push(0);
                nanos.push(0);
                validity.push(false)?;
            }
            Self::List { values, validity } | Self::Map { values, validity } => {
                values.push(std::iter::empty())?;
                validity.push(false)?;
            }
        }
        Ok(())
    }

    pub fn push(&mut self, value: &ScalarValue, strings: &mut Dictionary) -> Result<()> {
        if matches!(value, ScalarValue::Null) {
            return self.push_null();
        }
        match (self, value) {
            (Self::Boolean { values, validity }, ScalarValue::Boolean(value)) => {
                values.push(*value);
                validity.push(true)?;
            }
            (Self::Integer { values, validity }, ScalarValue::Integer(value)) => {
                values.push(*value);
                validity.push(true)?;
            }
            (Self::Float { values, validity }, ScalarValue::Float(value)) => {
                values.push(*value);
                validity.push(true)?;
            }
            (Self::String { values, validity }, ScalarValue::String(value)) => {
                values.push(strings.intern(value)?);
                validity.push(true)?;
            }
            (Self::Bytes { values, validity }, ScalarValue::Bytes(value)) => {
                values.push(value)?;
                validity.push(true)?;
            }
            (Self::Date { values, validity }, ScalarValue::Date(value)) => {
                values.push(*value);
                validity.push(true)?;
            }
            (Self::LocalTime { values, validity }, ScalarValue::LocalTime(value)) => {
                values.push(*value);
                validity.push(true)?;
            }
            (
                Self::ZonedTime {
                    nanos,
                    offsets,
                    validity,
                },
                ScalarValue::ZonedTime {
                    nanos: n,
                    offset_seconds,
                },
            ) => {
                nanos.push(*n);
                offsets.push(*offset_seconds);
                validity.push(true)?;
            }
            (
                Self::LocalDateTime {
                    seconds,
                    nanos,
                    validity,
                },
                ScalarValue::LocalDateTime {
                    seconds: s,
                    nanos: n,
                },
            ) => {
                seconds.push(*s);
                nanos.push(*n);
                validity.push(true)?;
            }
            (
                Self::ZonedDateTime {
                    seconds,
                    nanos,
                    timezones,
                    validity,
                },
                ScalarValue::ZonedDateTime {
                    seconds: s,
                    nanos: n,
                    timezone,
                },
            ) => {
                seconds.push(*s);
                nanos.push(*n);
                timezones.push(strings.intern(timezone)?);
                validity.push(true)?;
            }
            (
                Self::Duration {
                    months,
                    days,
                    seconds,
                    nanos,
                    validity,
                },
                ScalarValue::Duration {
                    months: m,
                    days: d,
                    seconds: s,
                    nanos: n,
                },
            ) => {
                months.push(*m);
                days.push(*d);
                seconds.push(*s);
                nanos.push(*n);
                validity.push(true)?;
            }
            (Self::List { values, validity }, ScalarValue::List(value)) => {
                values.push(value.as_bytes().iter().copied())?;
                validity.push(true)?;
            }
            (Self::Map { values, validity }, ScalarValue::Map(value)) => {
                values.push(value.as_bytes().iter().copied())?;
                validity.push(true)?;
            }
            _ => {
                return Err(Error::new(
                    ErrorCode::QueryType,
                    "property value does not match column type",
                ));
            }
        }
        Ok(())
    }

    pub fn set(&mut self, row: usize, value: &ScalarValue, strings: &mut Dictionary) -> Result<()> {
        if row >= self.len() {
            return Err(Error::invalid_data("column row is out of bounds"));
        }
        if matches!(value, ScalarValue::Null) {
            match self {
                Self::List { values, .. } | Self::Map { values, .. } => {
                    values.replace(row as u32, std::iter::empty())?;
                }
                _ => {}
            }
            self.validity_mut().set(row, false)?;
            return Ok(());
        }
        match (self, value) {
            (Self::Boolean { values, validity }, ScalarValue::Boolean(value)) => {
                replace_paged(values, row, *value)?;
                validity.set(row, true)?;
            }
            (Self::Integer { values, validity }, ScalarValue::Integer(value)) => {
                replace_paged(values, row, *value)?;
                validity.set(row, true)?;
            }
            (Self::Float { values, validity }, ScalarValue::Float(value)) => {
                replace_paged(values, row, *value)?;
                validity.set(row, true)?;
            }
            (Self::String { values, validity }, ScalarValue::String(value)) => {
                replace_paged(values, row, strings.intern(value)?)?;
                validity.set(row, true)?;
            }
            (Self::Bytes { values, validity }, ScalarValue::Bytes(value)) => {
                values.replace(row as u32, value)?;
                validity.set(row, true)?;
            }
            (Self::Date { values, validity }, ScalarValue::Date(value)) => {
                replace_paged(values, row, *value)?;
                validity.set(row, true)?;
            }
            (Self::LocalTime { values, validity }, ScalarValue::LocalTime(value)) => {
                replace_paged(values, row, *value)?;
                validity.set(row, true)?;
            }
            (
                Self::ZonedTime {
                    nanos,
                    offsets,
                    validity,
                },
                ScalarValue::ZonedTime {
                    nanos: n,
                    offset_seconds,
                },
            ) => {
                replace_paged(nanos, row, *n)?;
                replace_paged(offsets, row, *offset_seconds)?;
                validity.set(row, true)?;
            }
            (
                Self::LocalDateTime {
                    seconds,
                    nanos,
                    validity,
                },
                ScalarValue::LocalDateTime {
                    seconds: s,
                    nanos: n,
                },
            ) => {
                replace_paged(seconds, row, *s)?;
                replace_paged(nanos, row, *n)?;
                validity.set(row, true)?;
            }
            (
                Self::ZonedDateTime {
                    seconds,
                    nanos,
                    timezones,
                    validity,
                },
                ScalarValue::ZonedDateTime {
                    seconds: s,
                    nanos: n,
                    timezone,
                },
            ) => {
                replace_paged(seconds, row, *s)?;
                replace_paged(nanos, row, *n)?;
                replace_paged(timezones, row, strings.intern(timezone)?)?;
                validity.set(row, true)?;
            }
            (
                Self::Duration {
                    months,
                    days,
                    seconds,
                    nanos,
                    validity,
                },
                ScalarValue::Duration {
                    months: m,
                    days: d,
                    seconds: s,
                    nanos: n,
                },
            ) => {
                replace_paged(months, row, *m)?;
                replace_paged(days, row, *d)?;
                replace_paged(seconds, row, *s)?;
                replace_paged(nanos, row, *n)?;
                validity.set(row, true)?;
            }
            (Self::List { values, validity }, ScalarValue::List(value)) => {
                values.replace(row as u32, value.as_bytes().iter().copied())?;
                validity.set(row, true)?;
            }
            (Self::Map { values, validity }, ScalarValue::Map(value)) => {
                values.replace(row as u32, value.as_bytes().iter().copied())?;
                validity.set(row, true)?;
            }
            _ => {
                return Err(Error::new(
                    ErrorCode::QueryType,
                    "property value does not match column type",
                ));
            }
        }
        Ok(())
    }

    #[must_use]
    pub fn get(&self, row: usize, strings: &Dictionary) -> Option<ScalarValue> {
        if !self.validity().is_present(row) {
            return Some(ScalarValue::Null);
        }
        match self {
            Self::Boolean { values, .. } => values.get(row).copied().map(ScalarValue::Boolean),
            Self::Integer { values, .. } => values.get(row).copied().map(ScalarValue::Integer),
            Self::Float { values, .. } => values.get(row).copied().map(ScalarValue::Float),
            Self::String { values, .. } => values
                .get(row)
                .and_then(|id| strings.resolve(*id))
                .map(|s| ScalarValue::String(Arc::from(s))),
            Self::Bytes { values, .. } => values
                .get(row)
                .map(|value| ScalarValue::Bytes(Arc::from(value))),
            Self::Date { values, .. } => values.get(row).copied().map(ScalarValue::Date),
            Self::LocalTime { values, .. } => values.get(row).copied().map(ScalarValue::LocalTime),
            Self::ZonedTime { nanos, offsets, .. } => Some(ScalarValue::ZonedTime {
                nanos: *nanos.get(row)?,
                offset_seconds: *offsets.get(row)?,
            }),
            Self::LocalDateTime { seconds, nanos, .. } => Some(ScalarValue::LocalDateTime {
                seconds: *seconds.get(row)?,
                nanos: *nanos.get(row)?,
            }),
            Self::ZonedDateTime {
                seconds,
                nanos,
                timezones,
                ..
            } => Some(ScalarValue::ZonedDateTime {
                seconds: *seconds.get(row)?,
                nanos: *nanos.get(row)?,
                timezone: Arc::from(strings.resolve(*timezones.get(row)?)?),
            }),
            Self::Duration {
                months,
                days,
                seconds,
                nanos,
                ..
            } => Some(ScalarValue::Duration {
                months: *months.get(row)?,
                days: *days.get(row)?,
                seconds: *seconds.get(row)?,
                nanos: *nanos.get(row)?,
            }),
            Self::List { values, .. } => values
                .get(row as u32)
                .map(|value| ScalarValue::List(crate::DocumentList::from_canonical(value.into()))),
            Self::Map { values, .. } => values
                .get(row as u32)
                .map(|value| ScalarValue::Map(crate::DocumentMap::from_canonical(value.into()))),
        }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.validity().len()
    }

    #[must_use]
    pub fn validity(&self) -> &Validity {
        match self {
            Self::Boolean { validity, .. }
            | Self::Integer { validity, .. }
            | Self::Float { validity, .. }
            | Self::String { validity, .. }
            | Self::Bytes { validity, .. }
            | Self::Date { validity, .. }
            | Self::LocalTime { validity, .. }
            | Self::ZonedTime { validity, .. }
            | Self::LocalDateTime { validity, .. }
            | Self::ZonedDateTime { validity, .. }
            | Self::Duration { validity, .. }
            | Self::List { validity, .. }
            | Self::Map { validity, .. } => validity,
        }
    }

    pub fn validity_mut(&mut self) -> &mut Validity {
        match self {
            Self::Boolean { validity, .. }
            | Self::Integer { validity, .. }
            | Self::Float { validity, .. }
            | Self::String { validity, .. }
            | Self::Bytes { validity, .. }
            | Self::Date { validity, .. }
            | Self::LocalTime { validity, .. }
            | Self::ZonedTime { validity, .. }
            | Self::LocalDateTime { validity, .. }
            | Self::ZonedDateTime { validity, .. }
            | Self::Duration { validity, .. }
            | Self::List { validity, .. }
            | Self::Map { validity, .. } => validity,
        }
    }

    #[must_use]
    fn estimated_bytes(&self) -> usize {
        let validity = self.validity().len().div_ceil(8);
        let values = match self {
            Self::Boolean { values, .. } => values.len().saturating_mul(size_of::<bool>()),
            Self::Integer { values, .. } | Self::LocalTime { values, .. } => {
                values.len().saturating_mul(size_of::<i64>())
            }
            Self::Float { values, .. } => values.len().saturating_mul(size_of::<f64>()),
            Self::String { values, .. } => values.len().saturating_mul(size_of::<u32>()),
            Self::Bytes { values, .. } => values.estimated_bytes(),
            Self::Date { values, .. } => values.len().saturating_mul(size_of::<i64>()),
            Self::ZonedTime { nanos, offsets, .. } => nanos
                .len()
                .saturating_mul(size_of::<i64>())
                .saturating_add(offsets.len().saturating_mul(size_of::<i32>())),
            Self::LocalDateTime { seconds, nanos, .. } => seconds
                .len()
                .saturating_mul(size_of::<i64>())
                .saturating_add(nanos.len().saturating_mul(size_of::<u32>())),
            Self::ZonedDateTime {
                seconds,
                nanos,
                timezones,
                ..
            } => seconds
                .len()
                .saturating_mul(size_of::<i64>())
                .saturating_add(nanos.len().saturating_mul(size_of::<u32>()))
                .saturating_add(timezones.len().saturating_mul(size_of::<u32>())),
            Self::Duration {
                months,
                days,
                seconds,
                nanos,
                ..
            } => months
                .len()
                .saturating_mul(size_of::<i64>())
                .saturating_add(days.len().saturating_mul(size_of::<i64>()))
                .saturating_add(seconds.len().saturating_mul(size_of::<i64>()))
                .saturating_add(nanos.len().saturating_mul(size_of::<i32>())),
            Self::List { values, .. } | Self::Map { values, .. } => values.estimated_bytes(),
        };
        validity.saturating_add(values)
    }

    #[must_use]
    fn document_needs_seal(&self) -> bool {
        match self {
            Self::List { values, .. } | Self::Map { values, .. } => values.needs_seal(),
            _ => false,
        }
    }

    fn seal_document(&mut self) -> Result<()> {
        match self {
            Self::List { values, validity } | Self::Map { values, validity } => {
                values.seal(validity)
            }
            _ => Ok(()),
        }
    }

    #[cfg(test)]
    fn document_storage_bytes(&self) -> Option<(usize, usize)> {
        match self {
            Self::List { values, .. } | Self::Map { values, .. } => Some((
                values.arena.estimated_bytes(),
                values.live_values.saturating_mul(size_of::<u8>()),
            )),
            _ => None,
        }
    }

    #[cfg(test)]
    fn detached_page_bytes_from(&self, previous: &Self) -> usize {
        match (self, previous) {
            (
                Self::Integer { values, validity },
                Self::Integer {
                    values: old_values,
                    validity: old_validity,
                },
            )
            | (
                Self::LocalTime { values, validity },
                Self::LocalTime {
                    values: old_values,
                    validity: old_validity,
                },
            ) => values
                .detached_page_bytes_from(old_values)
                .saturating_add(validity.detached_page_bytes_from(old_validity)),
            _ => 0,
        }
    }
}

/// Schema-keyed property columns with one shared string dictionary.
///
/// Homogeneous properties remain physically typed. A property is marked in `mixed` only after a
/// conflicting non-null kind is observed; its physical `Bytes` column then contains one compact
/// tagged payload per present row. Keeping the marker separate preserves the established typed
/// column ABI while making unsupported accelerator admission fail instead of misreading mixed
/// payloads as ordinary byte properties.
#[derive(Clone, Debug, Default)]
pub struct PropertyColumns {
    rows: usize,
    columns: BTreeMap<PropertyId, Arc<TypedColumn>>,
    strings: Arc<Dictionary>,
    mixed: BTreeSet<PropertyId>,
}

#[derive(Serialize, Deserialize)]
struct PropertyColumnsWire {
    rows: usize,
    columns: BTreeMap<PropertyId, Arc<TypedColumn>>,
    strings: Arc<Dictionary>,
}

impl Serialize for PropertyColumns {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut columns = self.columns.clone();
        for property in &self.mixed {
            let column = columns.get_mut(property).ok_or_else(|| {
                serde::ser::Error::custom("mixed property marker has no physical column")
            })?;
            let column = Arc::make_mut(column);
            if !matches!(column, TypedColumn::Bytes { .. }) {
                return Err(serde::ser::Error::custom(
                    "mixed property is not backed by a tagged byte column",
                ));
            }
            column
                .validity_mut()
                .mark_mixed_wire()
                .map_err(serde::ser::Error::custom)?;
        }
        PropertyColumnsWire {
            rows: self.rows,
            columns,
            strings: Arc::clone(&self.strings),
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PropertyColumns {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let mut wire = PropertyColumnsWire::deserialize(deserializer)?;
        let mut mixed = BTreeSet::new();
        for (property, column) in &mut wire.columns {
            if matches!(column.as_ref(), TypedColumn::Bytes { .. })
                && Arc::make_mut(column)
                    .validity_mut()
                    .take_mixed_wire_marker()
                    .map_err(serde::de::Error::custom)?
            {
                mixed.insert(*property);
            }
        }
        let columns = Self {
            rows: wire.rows,
            columns: wire.columns,
            strings: wire.strings,
            mixed,
        };
        columns.validate().map_err(serde::de::Error::custom)?;
        Ok(columns)
    }
}

impl PropertyColumns {
    pub fn push_row(&mut self, values: &[(PropertyId, ScalarValue)]) -> Result<()> {
        self.validate_push_row(values)?;
        let row = self.rows;
        let row_values = values
            .iter()
            .map(|(property, value)| (*property, value))
            .collect::<BTreeMap<_, _>>();
        let mut promotions = BTreeMap::new();
        for (property, value) in &row_values {
            let Some(column) = self.columns.get(property) else {
                continue;
            };
            if !self.mixed.contains(property) && !column.accepts(value) {
                let mut promoted = column.into_mixed(&self.strings)?;
                promoted.mixed_push(value)?;
                promotions.insert(*property, Arc::new(promoted));
            }
        }

        for (property, column) in &mut self.columns {
            if promotions.contains_key(property) {
                continue;
            }
            let value = row_values
                .get(property)
                .copied()
                .unwrap_or(&ScalarValue::Null);
            if self.mixed.contains(property) {
                Arc::make_mut(column).mixed_push(value)?;
            } else {
                Arc::make_mut(column).push(value, Arc::make_mut(&mut self.strings))?;
            }
        }
        for (property, column) in promotions {
            self.columns.insert(property, column);
            self.mixed.insert(property);
        }
        for (property, value) in row_values {
            if !self.columns.contains_key(&property) && !matches!(value, ScalarValue::Null) {
                self.columns.insert(
                    property,
                    Arc::new(TypedColumn::with_value(
                        value,
                        row,
                        Arc::make_mut(&mut self.strings),
                    )?),
                );
            }
        }
        self.rows += 1;
        Ok(())
    }

    pub fn set(&mut self, row: u32, property: PropertyId, value: &ScalarValue) -> Result<()> {
        let row = row as usize;
        if row >= self.rows {
            return Err(Error::invalid_data("property row is out of bounds"));
        }
        self.validate_value(property, value)?;
        if self.mixed.contains(&property) {
            let column = self.columns.get_mut(&property).ok_or_else(|| {
                Error::new(
                    ErrorCode::CorruptStorage,
                    "mixed property marker has no physical column",
                )
            })?;
            return Arc::make_mut(column).mixed_set(row, value);
        }
        if let Some(column) = self.columns.get_mut(&property) {
            if column.accepts(value) {
                Arc::make_mut(column).set(row, value, Arc::make_mut(&mut self.strings))
            } else {
                let mut promoted = column.into_mixed(&self.strings)?;
                promoted.mixed_set(row, value)?;
                *column = Arc::new(promoted);
                self.mixed.insert(property);
                Ok(())
            }
        } else if matches!(value, ScalarValue::Null) {
            Ok(())
        } else {
            let mut column = TypedColumn::with_value(value, row, Arc::make_mut(&mut self.strings))?;
            for _ in (row + 1)..self.rows {
                column.push_null()?;
            }
            self.columns.insert(property, Arc::new(column));
            Ok(())
        }
    }

    pub fn validate_push_row(&self, values: &[(PropertyId, ScalarValue)]) -> Result<()> {
        self.rows.checked_add(1).ok_or_else(|| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "property row count exhausted",
            )
        })?;
        if values
            .iter()
            .map(|(property, _)| *property)
            .collect::<BTreeSet<_>>()
            .len()
            != values.len()
        {
            return Err(Error::invalid_data(
                "property row contains a duplicate property",
            ));
        }
        let mut new_strings = BTreeMap::<&str, ()>::new();
        for (property, value) in values {
            validate_property_value_shape(value)?;
            if self.mixed.contains(property) {
                self.columns
                    .get(property)
                    .ok_or_else(|| {
                        Error::new(
                            ErrorCode::CorruptStorage,
                            "mixed property marker has no physical column",
                        )
                    })?
                    .mixed_validate_push(value)?;
            }
            let needs_dictionary = self
                .columns
                .get(property)
                .is_none_or(|column| column.accepts(value))
                && !self.mixed.contains(property);
            if needs_dictionary {
                if let ScalarValue::String(value) = value {
                    new_strings.insert(value, ());
                } else if let ScalarValue::ZonedDateTime { timezone, .. } = value {
                    new_strings.insert(timezone, ());
                }
            } else {
                let _ = mixed_payload(value)?;
            }
        }
        let missing = new_strings
            .keys()
            .filter(|value| self.strings.id(value).is_none())
            .count();
        let final_len = self
            .strings
            .values
            .rows()
            .checked_add(missing)
            .ok_or_else(|| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "string dictionary exhausted",
                )
            })?;
        if final_len > u32::MAX as usize {
            return Err(Error::new(
                ErrorCode::ResultBudgetExceeded,
                "string dictionary exhausted",
            ));
        }
        Ok(())
    }

    pub fn validate_value(&self, property: PropertyId, value: &ScalarValue) -> Result<()> {
        validate_property_value_shape(value)?;
        if self.mixed.contains(&property) {
            let column = self.columns.get(&property).ok_or_else(|| {
                Error::new(
                    ErrorCode::CorruptStorage,
                    "mixed property marker has no physical column",
                )
            })?;
            if !matches!(column.as_ref(), TypedColumn::Bytes { .. }) {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "mixed property is not backed by a tagged byte column",
                ));
            }
            let _ = mixed_payload(value)?;
            return Ok(());
        }
        let existing_accepts = self
            .columns
            .get(&property)
            .is_none_or(|column| column.accepts(value));
        if !existing_accepts {
            let _ = mixed_payload(value)?;
            return Ok(());
        }
        if let ScalarValue::String(value) = value
            && self.strings.id(value).is_none()
            && self.strings.values.rows() >= u32::MAX as usize
        {
            return Err(Error::new(
                ErrorCode::ResultBudgetExceeded,
                "string dictionary exhausted",
            ));
        }
        if let ScalarValue::ZonedDateTime { timezone, .. } = value
            && self.strings.id(timezone).is_none()
            && self.strings.values.rows() >= u32::MAX as usize
        {
            return Err(Error::new(
                ErrorCode::ResultBudgetExceeded,
                "string dictionary exhausted",
            ));
        }
        Ok(())
    }

    #[must_use]
    pub fn get(&self, row: u32, property: PropertyId) -> Option<ScalarValue> {
        let column = self.columns.get(&property)?;
        let value = if self.mixed.contains(&property) {
            column.mixed_get(row as usize)?.ok()?
        } else {
            column.get(row as usize, &self.strings)?
        };
        (!matches!(value, ScalarValue::Null)).then_some(value)
    }

    /// Returns a homogeneous typed column. Mixed properties deliberately return `None`, so a
    /// backend that only understands homogeneous columns cannot silently reinterpret tagged
    /// payloads as an ordinary byte property.
    #[must_use]
    pub fn column(&self, property: PropertyId) -> Option<&TypedColumn> {
        if self.mixed.contains(&property) {
            return None;
        }
        self.columns.get(&property).map(AsRef::as_ref)
    }

    /// Returns the physical backing column even when its semantic representation is tagged
    /// mixed storage. Device uploaders must pair this with [`Self::is_mixed`] and may not treat
    /// the returned `Bytes` payload as an ordinary byte-valued property.
    #[must_use]
    pub fn physical_column(&self, property: PropertyId) -> Option<&TypedColumn> {
        self.columns.get(&property).map(AsRef::as_ref)
    }

    #[must_use]
    pub fn contains(&self, property: PropertyId) -> bool {
        self.columns.contains_key(&property)
    }

    #[must_use]
    pub fn is_mixed(&self, property: PropertyId) -> bool {
        self.mixed.contains(&property)
    }

    #[must_use]
    pub fn accepts(&self, property: PropertyId, value: &ScalarValue) -> Option<bool> {
        self.columns
            .get(&property)
            .map(|column| self.mixed.contains(&property) || column.accepts(value))
    }

    #[must_use]
    pub fn rows(&self) -> usize {
        self.rows
    }

    #[must_use]
    pub fn property_ids(&self) -> impl Iterator<Item = PropertyId> + '_ {
        self.columns.keys().copied()
    }

    /// Raw physical columns in stable property-ID order. A mixed semantic column is physically a
    /// tagged `Bytes` column and is identified by [`Self::is_mixed`].
    pub fn columns(&self) -> impl Iterator<Item = (PropertyId, &TypedColumn)> {
        self.columns
            .iter()
            .map(|(property, column)| (*property, column.as_ref()))
    }

    /// Mutable counterpart to [`Self::physical_column`], used only to publish/rebase canonical
    /// storage into a device-addressable allocation without changing its semantic type.
    pub fn physical_column_mut(&mut self, property: PropertyId) -> Option<&mut TypedColumn> {
        self.columns.get_mut(&property).map(Arc::make_mut)
    }

    #[must_use]
    pub fn string_dictionary(&self) -> &Dictionary {
        &self.strings
    }

    #[cfg_attr(
        not(all(feature = "accelerator", any(target_os = "macos", target_os = "ios"))),
        allow(dead_code)
    )]
    pub fn string_dictionary_mut(&mut self) -> &mut Dictionary {
        Arc::make_mut(&mut self.strings)
    }

    #[must_use]
    pub fn estimated_bytes(&self) -> usize {
        self.columns
            .values()
            .map(|column| column.estimated_bytes())
            .sum::<usize>()
            .saturating_add(self.strings.estimated_bytes())
            .saturating_add(self.mixed.len().saturating_mul(size_of::<PropertyId>()))
    }

    pub fn seal_documents_if_needed(&mut self) -> Result<bool> {
        let mut sealed = false;
        for (property, column) in &mut self.columns {
            let mixed = self.mixed.contains(property);
            let needs_seal = if mixed {
                matches!(column.as_ref(), TypedColumn::Bytes { values, .. } if values.needs_seal())
            } else {
                column.document_needs_seal()
            };
            if needs_seal {
                let column = Arc::make_mut(column);
                if mixed {
                    let TypedColumn::Bytes { values, .. } = column else {
                        return Err(Error::new(
                            ErrorCode::CorruptStorage,
                            "mixed property is not backed by a tagged byte column",
                        ));
                    };
                    values.seal()?;
                } else {
                    column.seal_document()?;
                }
                sealed = true;
            }
        }
        Ok(sealed)
    }

    #[cfg(test)]
    pub fn document_storage_bytes(&self, property: PropertyId) -> Option<(usize, usize)> {
        let column = self.columns.get(&property)?;
        if self.mixed.contains(&property) {
            let TypedColumn::Bytes { values, .. } = column.as_ref() else {
                return None;
            };
            Some(values.storage_bytes())
        } else {
            column.document_storage_bytes()
        }
    }

    #[cfg(test)]
    pub fn detached_page_bytes_from(&self, previous: &Self) -> usize {
        let column_bytes = self
            .columns
            .iter()
            .filter_map(|(property, column)| {
                let old = previous.columns.get(property)?;
                (!Arc::ptr_eq(column, old)).then(|| {
                    if self.mixed.contains(property) && previous.mixed.contains(property) {
                        match (column.as_ref(), old.as_ref()) {
                            (
                                TypedColumn::Bytes { values, validity },
                                TypedColumn::Bytes {
                                    values: old_values,
                                    validity: old_validity,
                                },
                            ) => values
                                .detached_storage_bytes_from(old_values)
                                .saturating_add(validity.detached_page_bytes_from(old_validity)),
                            _ => column.estimated_bytes(),
                        }
                    } else {
                        column.detached_page_bytes_from(old)
                    }
                })
            })
            .fold(0_usize, usize::saturating_add);
        let dictionary_bytes = (!Arc::ptr_eq(&self.strings, &previous.strings))
            .then(|| self.strings.detached_storage_bytes_from(&previous.strings))
            .unwrap_or(0);
        column_bytes.saturating_add(dictionary_bytes)
    }

    fn validate(&self) -> Result<()> {
        for property in &self.mixed {
            if !self.columns.contains_key(property) {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "mixed property marker has no physical column",
                ));
            }
        }
        for (property, column) in &self.columns {
            column.validate(&self.strings)?;
            if column.len() != self.rows {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "property column row count does not match the graph",
                ));
            }
            if self.mixed.contains(property) {
                column.mixed_validate()?;
            }
        }
        Ok(())
    }
}

#[cfg(test)]
mod cow_tests {
    use super::*;
    use crate::{DocumentItem, DocumentList};

    #[test]
    fn packed_list_shared_extension_seals_only_the_same_allocation_tail() -> Result<()> {
        let old_rows = 100_000_usize;
        let appended = 256_usize;
        let offsets: Arc<[u32]> = (0..=old_rows + appended)
            .map(|row| u32::try_from(row).expect("test row fits u32"))
            .collect::<Vec<_>>()
            .into();
        let values: Arc<[u64]> = (0..old_rows + appended)
            .map(|row| row as u64)
            .collect::<Vec<_>>()
            .into();
        let full_offsets = SharedFlat::from_arc_slice(offsets).expect("shared offsets");
        let full_values = SharedFlat::from_arc_slice(values).expect("shared values");
        let mut lists = PackedLists::default();
        lists.rebase_shared(
            full_offsets
                .slice(0, old_rows + 1)
                .expect("shared offset prefix"),
            full_values.slice(0, old_rows).expect("shared value prefix"),
        )?;
        for row in old_rows..old_rows + appended {
            lists.push(std::iter::once(row as u64))?;
        }

        lists.rebase_shared_extension(full_offsets, full_values)?;

        assert_eq!(lists.rows(), old_rows + appended);
        assert_eq!(lists.get(0), Some([0_u64].as_slice()));
        assert_eq!(
            lists.get((old_rows + appended - 1) as u32),
            Some([(old_rows + appended - 1) as u64].as_slice())
        );
        Ok(())
    }

    fn heterogeneous_values() -> Result<Vec<ScalarValue>> {
        Ok(vec![
            ScalarValue::Integer(0),
            ScalarValue::String(Arc::from("xx")),
            ScalarValue::Boolean(true),
            ScalarValue::Float(OrderedFloat(1.25)),
            ScalarValue::Bytes(Arc::from([1_u8, 2, 3])),
            ScalarValue::Date(19_000),
            ScalarValue::LocalTime(42),
            ScalarValue::ZonedTime {
                nanos: 43,
                offset_seconds: 3_600,
            },
            ScalarValue::LocalDateTime {
                seconds: 44,
                nanos: 45,
            },
            ScalarValue::ZonedDateTime {
                seconds: 46,
                nanos: 47,
                timezone: Arc::from("Europe/London"),
            },
            ScalarValue::Duration {
                months: 1,
                days: 2,
                seconds: 3,
                nanos: 4,
            },
            ScalarValue::List(DocumentList::new(vec![DocumentItem::Scalar(
                ScalarValue::Integer(5),
            )])?),
        ])
    }

    #[test]
    fn absent_property_cells_are_not_materialized_as_null_properties() -> Result<()> {
        let property = PropertyId(7);
        let mut columns = PropertyColumns::default();
        columns.push_row(&[(property, ScalarValue::Integer(42))])?;
        columns.push_row(&[])?;

        assert_eq!(columns.get(0, property), Some(ScalarValue::Integer(42)));
        assert_eq!(columns.get(1, property), None);
        Ok(())
    }

    #[test]
    fn conflicting_property_kinds_promote_once_and_preserve_every_value() -> Result<()> {
        let property = PropertyId(7);
        let values = heterogeneous_values()?;
        let mut columns = PropertyColumns::default();
        for value in &values {
            columns.push_row(&[(property, value.clone())])?;
        }
        columns.push_row(&[])?;
        columns.push_row(&[(property, ScalarValue::Null)])?;

        assert!(columns.is_mixed(property));
        assert!(columns.column(property).is_none());
        assert!(matches!(
            columns
                .columns()
                .find(|(candidate, _)| *candidate == property)
                .map(|(_, column)| column),
            Some(TypedColumn::Bytes { .. })
        ));
        for (row, expected) in values.into_iter().enumerate() {
            assert_eq!(columns.get(row as u32, property), Some(expected));
        }
        assert_eq!(columns.get((columns.rows() - 2) as u32, property), None);
        assert_eq!(columns.get((columns.rows() - 1) as u32, property), None);
        columns.validate()?;
        Ok(())
    }

    #[test]
    fn set_promotes_homogeneous_column_without_mutating_pinned_generation() -> Result<()> {
        let property = PropertyId(9);
        let mut published = PropertyColumns::default();
        published.push_row(&[(property, ScalarValue::Integer(10))])?;
        published.push_row(&[(property, ScalarValue::Integer(20))])?;
        assert!(matches!(
            published.column(property),
            Some(TypedColumn::Integer { .. })
        ));

        let mut staged = published.clone();
        staged.set(1, property, &ScalarValue::String(Arc::from("twenty")))?;

        assert!(!published.is_mixed(property));
        assert_eq!(published.get(0, property), Some(ScalarValue::Integer(10)));
        assert_eq!(published.get(1, property), Some(ScalarValue::Integer(20)));
        assert!(staged.is_mixed(property));
        assert_eq!(staged.get(0, property), Some(ScalarValue::Integer(10)));
        assert_eq!(
            staged.get(1, property),
            Some(ScalarValue::String(Arc::from("twenty")))
        );
        Ok(())
    }

    #[test]
    fn mixed_column_checkpoint_round_trip_is_deterministic_and_validated() -> Result<()> {
        let property = PropertyId(11);
        let mut columns = PropertyColumns::default();
        columns.push_row(&[(property, ScalarValue::Integer(0))])?;
        columns.push_row(&[(property, ScalarValue::String(Arc::from("xx")))])?;
        columns.push_row(&[])?;

        let first = postcard::to_stdvec(&columns)
            .map_err(|error| Error::new(ErrorCode::CorruptStorage, error.to_string()))?;
        let second = postcard::to_stdvec(&columns)
            .map_err(|error| Error::new(ErrorCode::CorruptStorage, error.to_string()))?;
        assert_eq!(first, second);
        let restored: PropertyColumns = postcard::from_bytes(&first)
            .map_err(|error| Error::new(ErrorCode::CorruptStorage, error.to_string()))?;
        assert!(restored.is_mixed(property));
        assert_eq!(restored.get(0, property), Some(ScalarValue::Integer(0)));
        assert_eq!(
            restored.get(1, property),
            Some(ScalarValue::String(Arc::from("xx")))
        );
        assert_eq!(restored.get(2, property), None);
        assert_eq!(restored.estimated_bytes(), columns.estimated_bytes());

        let mut malformed = columns.clone();
        let column = Arc::make_mut(
            malformed
                .columns
                .get_mut(&property)
                .ok_or_else(|| Error::internal("mixed test column disappeared"))?,
        );
        let TypedColumn::Bytes { values, .. } = column else {
            return Err(Error::internal("mixed test column is not physical bytes"));
        };
        values.replace(0, &[u8::MAX])?;
        let encoded = postcard::to_stdvec(&malformed)
            .map_err(|error| Error::new(ErrorCode::CorruptStorage, error.to_string()))?;
        assert!(postcard::from_bytes::<PropertyColumns>(&encoded).is_err());
        Ok(())
    }

    #[test]
    fn homogeneous_checkpoint_without_mixed_marker_remains_readable() -> Result<()> {
        #[derive(Serialize)]
        struct LegacyPropertyColumns {
            rows: usize,
            columns: BTreeMap<PropertyId, Arc<TypedColumn>>,
            strings: Arc<Dictionary>,
        }

        let property = PropertyId(12);
        let mut columns = PropertyColumns::default();
        columns.push_row(&[(property, ScalarValue::Integer(7))])?;
        let legacy = LegacyPropertyColumns {
            rows: columns.rows,
            columns: columns.columns.clone(),
            strings: Arc::clone(&columns.strings),
        };
        let bytes = postcard::to_stdvec(&legacy)
            .map_err(|error| Error::new(ErrorCode::CorruptStorage, error.to_string()))?;
        let current = postcard::to_stdvec(&columns)
            .map_err(|error| Error::new(ErrorCode::CorruptStorage, error.to_string()))?;
        assert_eq!(current, bytes);
        let restored: PropertyColumns = postcard::from_bytes(&bytes)
            .map_err(|error| Error::new(ErrorCode::CorruptStorage, error.to_string()))?;
        assert_eq!(restored.get(0, property), Some(ScalarValue::Integer(7)));
        assert!(!restored.is_mixed(property));
        Ok(())
    }

    #[test]
    fn mixed_checkpoint_marker_handles_full_validity_words() -> Result<()> {
        let property = PropertyId(14);
        let mut columns = PropertyColumns::default();
        for value in 0..64 {
            columns.push_row(&[(property, ScalarValue::Integer(value))])?;
        }
        columns.set(63, property, &ScalarValue::String(Arc::from("sixty-four")))?;
        let bytes = postcard::to_stdvec(&columns)
            .map_err(|error| Error::new(ErrorCode::CorruptStorage, error.to_string()))?;
        let restored: PropertyColumns = postcard::from_bytes(&bytes)
            .map_err(|error| Error::new(ErrorCode::CorruptStorage, error.to_string()))?;
        assert!(restored.is_mixed(property));
        assert_eq!(restored.get(0, property), Some(ScalarValue::Integer(0)));
        assert_eq!(
            restored.get(63, property),
            Some(ScalarValue::String(Arc::from("sixty-four")))
        );
        restored.validate()?;
        Ok(())
    }

    #[test]
    fn mixed_point_update_detaches_storage_and_keeps_pinned_bytes_unchanged() -> Result<()> {
        let property = PropertyId(13);
        let mut published = PropertyColumns::default();
        published.push_row(&[(property, ScalarValue::Integer(1))])?;
        published.push_row(&[(property, ScalarValue::String(Arc::from("two")))])?;
        let mut staged = published.clone();
        staged.set(0, property, &ScalarValue::Boolean(false))?;

        assert_eq!(published.get(0, property), Some(ScalarValue::Integer(1)));
        assert_eq!(staged.get(0, property), Some(ScalarValue::Boolean(false)));
        let detached = staged.detached_page_bytes_from(&published);
        assert!(detached > 0);
        assert!(
            detached <= 128 * 1_024,
            "mixed point update detached {detached} bytes"
        );
        Ok(())
    }

    #[test]
    fn malformed_validity_storage_returns_corruption_instead_of_panicking() -> Result<()> {
        let mut validity = Validity {
            words: PagedVec::default(),
            len: 1,
        };
        let error = validity
            .set(0, true)
            .err()
            .ok_or_else(|| Error::internal("malformed validity storage was accepted"))?;
        assert_eq!(error.code, ErrorCode::CorruptStorage);
        Ok(())
    }

    #[test]
    fn malformed_typed_column_storage_returns_corruption_instead_of_panicking() -> Result<()> {
        let mut validity = Validity::default();
        validity.push(true)?;
        let mut column = TypedColumn::Integer {
            values: PagedVec::default(),
            validity,
        };
        let mut strings = Dictionary::default();
        let error = column
            .set(0, &ScalarValue::Integer(7), &mut strings)
            .err()
            .ok_or_else(|| Error::internal("malformed typed column storage was accepted"))?;
        assert_eq!(error.code, ErrorCode::CorruptStorage);
        Ok(())
    }

    #[test]
    fn empty_packed_lists_expose_canonical_empty_flat_views() {
        let lists = PackedLists::<u8>::default();
        assert_eq!(lists.offsets(), &[0]);
        assert!(lists.values().is_empty());
    }

    fn dictionary_with_values(count: usize) -> Result<Dictionary> {
        let mut dictionary = Dictionary::default();
        for index in 0..count {
            dictionary.intern(&format!("value-{index}"))?;
        }
        Ok(dictionary)
    }

    #[test]
    fn dictionary_insert_preserves_published_generation_with_bounded_copy() -> Result<()> {
        for count in [4_096, 8_192] {
            let published = dictionary_with_values(count)?;
            let mut staged = published.clone();
            let id = staged.intern("new-value")?;
            assert_eq!(id as usize, count);
            assert!(published.id("new-value").is_none());
            assert_eq!(staged.id("new-value"), Some(id));
            let detached = staged.detached_storage_bytes_from(&published);
            assert!(
                detached <= 32 * 1_024,
                "dictionary insert detached {detached} bytes"
            );
        }
        Ok(())
    }

    #[test]
    fn dictionary_deltas_survive_checkpoint_round_trip() -> Result<()> {
        let mut dictionary = Dictionary::default();
        let first = dictionary.intern("first")?;
        let second = dictionary.intern("second")?;
        let bytes = postcard::to_stdvec(&dictionary)
            .map_err(|error| Error::new(ErrorCode::CorruptStorage, error.to_string()))?;
        let mut restored: Dictionary = postcard::from_bytes(&bytes)
            .map_err(|error| Error::new(ErrorCode::CorruptStorage, error.to_string()))?;
        assert_eq!(restored.resolve(first), Some("first"));
        assert_eq!(restored.resolve(second), Some("second"));
        assert_eq!(restored.intern("first")?, first);
        assert_eq!(restored.intern("third")?, 2);
        Ok(())
    }
}
