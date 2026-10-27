//! Concurrent index cells. Clones share identity; they never fork index contents.

use arc_swap::{ArcSwap, ArcSwapOption};
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{
    borrow::Borrow,
    hash::Hash,
    ops::{Deref, DerefMut},
    sync::Arc,
};

pub(super) struct ConcurrentMap<K, V>(Arc<papaya::HashMap<K, V>>);
impl<K: Hash + Eq, V> std::fmt::Debug for ConcurrentMap<K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConcurrentMap")
            .field("len", &self.0.len())
            .finish()
    }
}

impl<K, V> Clone for ConcurrentMap<K, V> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<K: Hash + Eq, V> Default for ConcurrentMap<K, V> {
    fn default() -> Self {
        Self(Arc::new(papaya::HashMap::new()))
    }
}
impl<K: Hash + Eq + Clone, V: Clone> ConcurrentMap<K, V> {
    pub fn get<Q: Hash + Eq + ?Sized>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
    {
        self.0.pin().get(key).cloned()
    }
    pub fn insert(&self, key: K, value: V) {
        self.0.pin().insert(key, value);
    }
    pub fn swap_value(&self, key: K, value: V) -> Option<V> {
        self.0.pin().insert(key, value).cloned()
    }
    pub fn remove_if_value(&self, key: &K, predicate: impl Fn(&V) -> bool) -> Option<V> {
        self.0
            .pin()
            .remove_if(key, |_, value| predicate(value))
            .ok()
            .flatten()
            .map(|(_, value)| value.clone())
    }
    pub fn remove<Q: Hash + Eq + ?Sized>(&self, key: &Q) -> Option<V>
    where
        K: Borrow<Q>,
    {
        self.0.pin().remove(key).cloned()
    }
    pub fn remove_matching(&self, key: &K, value: &V)
    where
        V: PartialEq,
    {
        let _ = self.0.pin().remove_if(key, |_, current| current == value);
    }
    fn remove_when(&self, key: &K, predicate: impl Fn(&V) -> bool) {
        let _ = self.0.pin().remove_if(key, |_, current| predicate(current));
    }
    pub fn contains_key<Q: Hash + Eq + ?Sized>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
    {
        self.0.pin().contains_key(key)
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn iter(&self) -> std::vec::IntoIter<(K, V)> {
        self.0
            .pin()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect::<Vec<_>>()
            .into_iter()
    }
    pub fn values(&self) -> impl Iterator<Item = V> {
        self.iter().map(|(_, v)| v)
    }
    pub fn keys(&self) -> impl Iterator<Item = K> {
        self.iter().map(|(k, _)| k)
    }
    pub fn retain(&self, mut f: impl FnMut(&K, &V) -> bool) {
        for (k, v) in self.iter() {
            if !f(&k, &v) {
                self.remove(&k);
            }
        }
    }
    pub fn get_or_insert(&self, key: K, value: V) -> V {
        self.0.pin().get_or_insert(key, value).clone()
    }
}
impl<K: Hash + Eq + Clone, V: Clone> IntoIterator for &ConcurrentMap<K, V> {
    type Item = (K, V);
    type IntoIter = std::vec::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
impl<K: Hash + Eq + Clone + Serialize, V: Clone + Serialize> Serialize for ConcurrentMap<K, V> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let guard = self.0.pin();
        let mut map = s.serialize_map(Some(guard.len()))?;
        for (k, v) in guard.iter() {
            map.serialize_entry(k, v)?;
        }
        map.end()
    }
}
impl<'de, K: Hash + Eq + Clone + Deserialize<'de>, V: Clone + Deserialize<'de>> Deserialize<'de>
    for ConcurrentMap<K, V>
{
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct Visitor<K, V>(std::marker::PhantomData<(K, V)>);
        impl<'de, K: Hash + Eq + Clone + Deserialize<'de>, V: Clone + Deserialize<'de>>
            serde::de::Visitor<'de> for Visitor<K, V>
        {
            type Value = ConcurrentMap<K, V>;
            fn expecting(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                f.write_str("canonical index map")
            }
            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                mut map: M,
            ) -> Result<Self::Value, M::Error> {
                let result = ConcurrentMap::default();
                while let Some((k, v)) = map.next_entry()? {
                    result.insert(k, v);
                }
                Ok(result)
            }
        }
        d.deserialize_map(Visitor(std::marker::PhantomData))
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub(super) struct EntryMap(ConcurrentMap<String, Arc<super::IndexEntry>>);
impl EntryMap {
    pub fn get(&self, key: &str) -> Option<Arc<super::IndexEntry>> {
        self.0.get(key)
    }
    pub fn insert(&self, key: String, value: Arc<super::IndexEntry>) {
        self.0.insert(key, value);
    }
    pub fn remove(&self, key: &str) -> Option<Arc<super::IndexEntry>> {
        self.0.remove(key)
    }
    pub fn contains_key(&self, key: &str) -> bool {
        self.0.contains_key(key)
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn iter(&self) -> std::vec::IntoIter<(String, Arc<super::IndexEntry>)> {
        self.0.iter()
    }
    pub fn values(&self) -> impl Iterator<Item = Arc<super::IndexEntry>> {
        self.0.values()
    }
    pub fn keys(&self) -> impl Iterator<Item = String> {
        self.0.keys()
    }
    pub fn edit(&self, key: &str) -> Option<EntryEdit<'_>> {
        self.get(key).map(|value| EntryEdit {
            map: self,
            key: key.to_owned(),
            value: (*value).clone(),
            original: value,
        })
    }
}
impl IntoIterator for &EntryMap {
    type Item = (String, Arc<super::IndexEntry>);
    type IntoIter = std::vec::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
/// Only schema/lifecycle metadata is prepared here. Runtime payloads remain the same canonical cells.
pub(super) struct EntryEdit<'a> {
    map: &'a EntryMap,
    key: String,
    value: super::IndexEntry,
    original: Arc<super::IndexEntry>,
}
impl Deref for EntryEdit<'_> {
    type Target = super::IndexEntry;
    fn deref(&self) -> &Self::Target {
        &self.value
    }
}
impl DerefMut for EntryEdit<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.value
    }
}
impl Drop for EntryEdit<'_> {
    fn drop(&mut self) {
        let replacement = Arc::new(self.value.clone());
        self.map.0.0.pin().update(self.key.clone(), |current| {
            if Arc::ptr_eq(current, &self.original) {
                replacement.clone()
            } else {
                current.clone()
            }
        });
    }
}

#[derive(Clone, Debug, Default)]
pub(super) struct ProfileCell(Arc<ArcSwapOption<super::EmbeddingProfile>>);
impl ProfileCell {
    pub fn get(&self) -> Option<Arc<super::EmbeddingProfile>> {
        self.0.load_full()
    }
    pub fn set(&self, value: super::EmbeddingProfile) {
        self.0.store(Some(Arc::new(value)));
    }
    pub fn is_none(&self) -> bool {
        self.0.load().is_none()
    }
}
impl Serialize for ProfileCell {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        self.get().serialize(s)
    }
}
impl<'de> Deserialize<'de> for ProfileCell {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let value = Option::<super::EmbeddingProfile>::deserialize(d)?;
        Ok(Self(Arc::new(ArcSwapOption::from(value.map(Arc::new)))))
    }
}

/// One atomically published policy and reconciliation stamp, shared by catalogue handles.
#[derive(Clone, Debug, Default)]
pub(super) struct RelationshipScopeCell(Arc<ArcSwap<super::RelationshipSemanticState>>);
impl RelationshipScopeCell {
    pub fn get(&self) -> Arc<super::RelationshipSemanticState> {
        self.0.load_full()
    }
    pub fn restore(&self, value: super::RelationshipSemanticState) {
        self.0.store(Arc::new(value));
    }
    pub fn set(&self, scope: super::RelationshipSemanticScope) -> crate::Result<bool> {
        loop {
            let previous = self.get();
            if previous.scope == scope {
                return Ok(false);
            }
            let revision = previous.revision.checked_add(1).ok_or_else(|| {
                crate::Error::invalid_data("relationship semantic scope revision overflow")
            })?;
            let next = Arc::new(super::RelationshipSemanticState {
                scope: scope.clone(),
                revision,
                reconciled_revision: previous.reconciled_revision,
            });
            if Arc::ptr_eq(&previous, &self.0.compare_and_swap(&previous, next)) {
                return Ok(true);
            }
        }
    }
    pub fn complete(&self, expected_revision: u64) -> bool {
        loop {
            let previous = self.get();
            if previous.revision != expected_revision {
                return false;
            }
            if previous.reconciled_revision == expected_revision {
                return true;
            }
            let next = Arc::new(super::RelationshipSemanticState {
                scope: previous.scope.clone(),
                revision: previous.revision,
                reconciled_revision: expected_revision,
            });
            if Arc::ptr_eq(&previous, &self.0.compare_and_swap(&previous, next)) {
                return true;
            }
        }
    }
}
impl Serialize for RelationshipScopeCell {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.get().serialize(serializer)
    }
}
impl<'de> Deserialize<'de> for RelationshipScopeCell {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let value = super::RelationshipSemanticState::deserialize(deserializer)?;
        Ok(Self(Arc::new(ArcSwap::from_pointee(value))))
    }
}

/// Canonical membership lives in individual map entries, never copied posting bitmaps.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Postings<K: Hash + Eq + Clone>(ConcurrentMap<K, Arc<PostingRows>>);
#[derive(Debug, Default, Serialize, Deserialize)]
struct PostingRows {
    values: PostingValues,
    #[serde(skip)]
    writers: std::sync::atomic::AtomicUsize,
}
// Most equality keys (sids, paths, identifiers) own one row. A hash table per such key costs
// several KiB before storing a single posting. Keep that row inline and allocate a shared
// table only for genuinely multi-row keys. Readers never acquire the mutation mutex.
#[derive(Debug, Default)]
struct PostingValues {
    inline: std::sync::atomic::AtomicU64,
    spill: std::sync::OnceLock<ConcurrentMap<u32, ()>>,
    writer: std::sync::Mutex<()>,
}
const SPILLED: u64 = u64::MAX;
impl PostingValues {
    fn insert(&self, row: u32, (): ()) {
        use std::sync::atomic::Ordering;
        let current = self.inline.load(Ordering::Acquire);
        let value = u64::from(row) + 1;
        match current {
            0 => self.inline.store(value, Ordering::Release),
            SPILLED => self
                .spill
                .get_or_init(ConcurrentMap::default)
                .insert(row, ()),
            current if current == value => {}
            current => {
                let spill = self.spill.get_or_init(ConcurrentMap::default);
                spill.insert((current - 1) as u32, ());
                spill.insert(row, ());
                self.inline.store(SPILLED, Ordering::Release);
            }
        }
    }
    fn remove(&self, row: &u32) {
        use std::sync::atomic::Ordering;
        match self.inline.load(Ordering::Acquire) {
            SPILLED => {
                if let Some(spill) = self.spill.get() {
                    spill.remove(row);
                }
            }
            value if value == u64::from(*row) + 1 => self.inline.store(0, Ordering::Release),
            _ => {}
        }
    }
    fn len(&self) -> usize {
        match self.inline.load(std::sync::atomic::Ordering::Acquire) {
            0 => 0,
            SPILLED => self.spill.get().map_or(0, ConcurrentMap::len),
            _ => 1,
        }
    }
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn bounded(&self, limit: usize) -> Vec<u32> {
        if limit == 0 {
            return Vec::new();
        }
        match self.inline.load(std::sync::atomic::Ordering::Acquire) {
            0 => Vec::new(),
            SPILLED => self.spill.get().map_or_else(Vec::new, |spill| {
                spill.0.pin().keys().take(limit).copied().collect()
            }),
            value => vec![(value - 1) as u32],
        }
    }
    fn rows(&self) -> roaring::RoaringBitmap {
        match self.inline.load(std::sync::atomic::Ordering::Acquire) {
            0 => roaring::RoaringBitmap::new(),
            SPILLED => self
                .spill
                .get()
                .map_or_else(roaring::RoaringBitmap::new, |spill| {
                    spill.0.pin().keys().copied().collect()
                }),
            value => std::iter::once((value - 1) as u32).collect(),
        }
    }
}
impl Serialize for PostingValues {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        // Preserve the original map wire shape, including old checkpoints and postcard data.
        match self.inline.load(std::sync::atomic::Ordering::Acquire) {
            SPILLED => {
                if let Some(spill) = self.spill.get() {
                    spill.serialize(serializer)
                } else {
                    serializer.serialize_map(Some(0))?.end()
                }
            }
            value => {
                let mut map = serializer.serialize_map(Some(usize::from(value != 0)))?;
                if value != 0 {
                    map.serialize_entry(&((value - 1) as u32), &())?;
                }
                map.end()
            }
        }
    }
}
impl<'de> Deserialize<'de> for PostingValues {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = PostingValues;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("a posting row map")
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> Result<Self::Value, A::Error> {
                let result = PostingValues::default();
                while let Some((row, value)) = map.next_entry::<u32, ()>()? {
                    result.insert(row, value);
                }
                Ok(result)
            }
        }
        deserializer.deserialize_map(Visitor)
    }
}
const CLOSED: usize = usize::MAX / 2 + 1;
impl<K: Hash + Eq + Clone> Default for Postings<K> {
    fn default() -> Self {
        Self(ConcurrentMap::default())
    }
}
impl<K: Hash + Eq + Clone> Postings<K> {
    fn edit(&self, key: K, row: u32, insert: bool) {
        use std::sync::atomic::Ordering;
        loop {
            let rows = match self.0.get(&key) {
                Some(rows) => rows,
                None if insert => self
                    .0
                    .get_or_insert(key.clone(), Arc::new(PostingRows::default())),
                None => return,
            };
            if rows
                .writers
                .fetch_update(Ordering::AcqRel, Ordering::Acquire, |writers| {
                    (writers < CLOSED).then_some(writers + 1)
                })
                .is_err()
            {
                continue;
            }
            let _writer = rows
                .values
                .writer
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if insert {
                rows.values.insert(row, ());
            } else {
                rows.values.remove(&row);
            }
            if rows.writers.fetch_sub(1, Ordering::AcqRel) == 1
                && rows.values.is_empty()
                && rows
                    .writers
                    .compare_exchange(0, CLOSED, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
            {
                self.0
                    .remove_when(&key, |current| Arc::ptr_eq(current, &rows));
            }
            return;
        }
    }
    pub fn insert(&self, key: K, row: u32) {
        self.edit(key, row, true);
    }
    pub fn remove(&self, key: &K, row: u32) {
        self.edit(key.clone(), row, false);
    }
    pub fn rows(&self, key: &K) -> Option<roaring::RoaringBitmap> {
        let rows = self.0.get(key)?.values.rows();
        (!rows.is_empty()).then_some(rows)
    }
    pub fn bounded(&self, key: &K, limit: usize) -> Vec<u32> {
        let Some(rows) = self.0.get(key) else {
            return Vec::new();
        };
        let mut result = rows.values.bounded(limit);
        result.sort_unstable();
        result
    }
    pub fn count(&self, key: &K) -> u64 {
        self.0.get(key).map_or(0, |rows| rows.values.len() as u64)
    }
    pub fn materialized(&self) -> std::collections::BTreeMap<K, roaring::RoaringBitmap>
    where
        K: Ord,
    {
        self.0
            .keys()
            .filter_map(|key| self.rows(&key).map(|rows| (key, rows)))
            .collect()
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod posting_tests {
    use super::*;

    #[test]
    fn unique_dirty_keys_do_not_allocate_posting_tables() {
        let postings = Postings::default();
        for row in 0..20_000_u32 {
            postings.insert(format!("code:{row}:{}", "x".repeat(700)), row);
        }
        assert_eq!(postings.0.len(), 20_000);
        for (_, rows) in &postings.0 {
            assert_eq!(rows.values.len(), 1);
            assert!(rows.values.spill.get().is_none());
        }
        let key = format!("code:0:{}", "x".repeat(700));
        postings.insert(key.clone(), u32::MAX);
        assert_eq!(postings.rows(&key).unwrap().len(), 2);
        assert_eq!(postings.bounded(&key, 0), Vec::<u32>::new());
        postings.remove(&key, 0);
        assert_eq!(postings.bounded(&key, 1), vec![u32::MAX]);
        postings.remove(&key, u32::MAX);
        assert_eq!(postings.count(&key), 0);
        assert!(!postings.0.contains_key(&key));
    }

    #[test]
    fn posting_wire_shape_reads_old_maps_and_roundtrips_promoted_rows() -> crate::Result<()> {
        #[derive(Serialize)]
        struct OldRows {
            values: std::collections::BTreeMap<u32, ()>,
        }
        for values in [vec![], vec![0], vec![u32::MAX], vec![0, 7, u32::MAX]] {
            let old = OldRows {
                values: values.iter().map(|row| (*row, ())).collect(),
            };
            let mut encoded = Vec::new();
            ciborium::ser::into_writer(&old, &mut encoded)
                .map_err(|e| crate::Error::internal(e.to_string()))?;
            let rows: PostingRows = ciborium::de::from_reader(encoded.as_slice())
                .map_err(|e| crate::Error::internal(e.to_string()))?;
            assert_eq!(rows.values.rows().iter().collect::<Vec<_>>(), values);
            let encoded =
                postcard::to_stdvec(&rows).map_err(|e| crate::Error::internal(e.to_string()))?;
            let recovered: PostingRows = postcard::from_bytes(&encoded)
                .map_err(|e| crate::Error::internal(e.to_string()))?;
            assert_eq!(recovered.values.rows(), rows.values.rows());
        }
        Ok(())
    }

    #[test]
    fn concurrent_posting_insert_remove_keeps_completed_rows() {
        let postings = Postings::default();
        std::thread::scope(|scope| {
            for worker in 0..8_u32 {
                let postings = &postings;
                scope.spawn(move || {
                    for iteration in 0..1000_u32 {
                        let row = worker * 1000 + iteration;
                        postings.insert(1, row);
                        assert!(postings.rows(&1).is_some_and(|rows| rows.contains(row)));
                        postings.remove(&1, row);
                    }
                    postings.insert(1, 10_000 + worker);
                });
            }
        });
        assert_eq!(postings.rows(&1).unwrap(), (10_000..10_008).collect());
    }
}
