//! Concurrent index cells. Clones share identity; they never fork index contents.

use arc_swap::ArcSwapOption;
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

/// Canonical membership lives in individual map entries, never copied posting bitmaps.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub(super) struct Postings<K: Hash + Eq + Clone>(ConcurrentMap<K, Arc<PostingRows>>);
#[derive(Debug, Default, Serialize, Deserialize)]
struct PostingRows {
    values: ConcurrentMap<u32, ()>,
    #[serde(skip)]
    writers: std::sync::atomic::AtomicUsize,
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
        let rows: roaring::RoaringBitmap = self.0.get(key)?.values.keys().collect();
        (!rows.is_empty()).then_some(rows)
    }
    pub fn bounded(&self, key: &K, limit: usize) -> Vec<u32> {
        let Some(rows) = self.0.get(key) else {
            return Vec::new();
        };
        let mut result: Vec<_> = rows.values.0.pin().keys().take(limit).copied().collect();
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
