//! Canonical broker cells. A clone shares identity; a reader guards only the row it borrows.
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::{
    borrow::Borrow,
    hash::Hash,
    ops::{Deref, DerefMut},
    sync::{
        Arc,
        atomic::{AtomicU64, AtomicUsize, Ordering},
    },
};

macro_rules! counter {
    ($name:ident,$atomic:ident,$value:ty) => {
        #[derive(Clone, Default)]
        pub(super) struct $name(Arc<$atomic>);
        impl $name {
            pub fn new(value: $value) -> Self {
                Self(Arc::new($atomic::new(value)))
            }
            pub fn get(&self) -> $value {
                self.0.load(Ordering::Acquire)
            }
            pub fn set(&self, value: $value) {
                self.0.store(value, Ordering::Release)
            }
        }
        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                self.get().fmt(f)
            }
        }
        impl Serialize for $name {
            fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
                self.get().serialize(s)
            }
        }
        impl<'de> Deserialize<'de> for $name {
            fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
                Ok(Self::new(<$value>::deserialize(d)?))
            }
        }
    };
}
counter!(Counter, AtomicU64, u64);
counter!(Position, AtomicUsize, usize);

#[derive(Clone, Debug, Default)]
pub(crate) struct CanonicalSet<T: Hash + Eq + Clone>(CanonicalMap<T, ()>);
impl<T: Hash + Eq + Clone> CanonicalSet<T> {
    pub fn insert(&self, value: T) -> bool {
        self.0.insert(value, ()).is_none()
    }
    pub fn contains(&self, value: &T) -> bool {
        self.0.contains_key(value)
    }
    pub fn remove(&self, value: &T) -> bool {
        self.0.remove(value).is_some()
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
    pub fn iter(&self) -> impl Iterator<Item = T> {
        self.0.keys()
    }
    pub fn retain(&self, mut keep: impl FnMut(&T) -> bool) {
        self.0.retain(|key, _| keep(key));
    }
}
impl<T: Hash + Eq + Clone + Serialize> Serialize for CanonicalSet<T> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let mut sequence = s.serialize_seq(Some(self.len()))?;
        for key in self.iter() {
            sequence.serialize_element(&key)?;
        }
        sequence.end()
    }
}
impl<'de, T: Hash + Eq + Clone + Deserialize<'de>> Deserialize<'de> for CanonicalSet<T> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let values = Vec::<T>::deserialize(d)?;
        let result = Self(CanonicalMap::default());
        for value in values {
            result.insert(value);
        }
        Ok(result)
    }
}

pub(crate) struct CanonicalMap<K, V>(Arc<papaya::HashMap<K, Arc<V>>>);
impl<K, V> Clone for CanonicalMap<K, V> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<K: Hash + Eq, V> Default for CanonicalMap<K, V> {
    fn default() -> Self {
        Self(Arc::new(papaya::HashMap::new()))
    }
}
impl<K: Hash + Eq, V> std::fmt::Debug for CanonicalMap<K, V> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CanonicalMap")
            .field("len", &self.0.len())
            .finish()
    }
}
impl<K: Hash + Eq + Clone, V: Clone> CanonicalMap<K, V> {
    pub fn get<Q: Hash + Eq + ?Sized>(&self, key: &Q) -> Option<Arc<V>>
    where
        K: Borrow<Q>,
    {
        self.0.pin().get(key).cloned()
    }
    pub fn contains_key<Q: Hash + Eq + ?Sized>(&self, key: &Q) -> bool
    where
        K: Borrow<Q>,
    {
        self.0.pin().contains_key(key)
    }
    pub fn insert(&self, key: K, value: V) -> Option<Arc<V>> {
        self.0.pin().insert(key, Arc::new(value)).cloned()
    }
    pub fn remove<Q: Hash + Eq + ?Sized>(&self, key: &Q) -> Option<Arc<V>>
    where
        K: Borrow<Q>,
    {
        self.0.pin().remove(key).cloned()
    }
    pub fn len(&self) -> usize {
        self.0.len()
    }
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
    pub fn iter(&self) -> std::vec::IntoIter<(K, Arc<V>)> {
        self.0
            .pin()
            .iter()
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect::<Vec<_>>()
            .into_iter()
    }
    pub fn values(&self) -> impl Iterator<Item = Arc<V>> {
        self.iter().map(|(_, v)| v)
    }
    pub fn keys(&self) -> impl Iterator<Item = K> {
        self.iter().map(|(k, _)| k)
    }
    pub fn retain(&self, mut keep: impl FnMut(&K, &V) -> bool) {
        for (key, value) in self.iter() {
            if !keep(&key, &value) {
                self.remove(&key);
            }
        }
    }
    pub fn clear(&self) {
        self.0.pin().clear();
    }
    pub fn get_mut(&self, key: &K) -> Option<Edit<'_, K, V>> {
        let original = self.get(key)?;
        Some(Edit {
            map: self,
            key: key.clone(),
            value: Some((*original).clone()),
            original,
        })
    }
    pub fn entry(&self, key: K) -> Entry<'_, K, V> {
        Entry {
            map: self,
            key,
            modify: None,
        }
    }
    pub fn last_key_value(&self) -> Option<(K, Arc<V>)>
    where
        K: Ord,
    {
        self.iter().max_by(|a, b| a.0.cmp(&b.0))
    }
    #[cfg(test)]
    pub fn shared_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}
impl<K: Hash + Eq + Clone, V: Clone> IntoIterator for &CanonicalMap<K, V> {
    type Item = (K, Arc<V>);
    type IntoIter = std::vec::IntoIter<Self::Item>;
    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}
pub(crate) struct Edit<'a, K: Hash + Eq + Clone, V: Clone> {
    map: &'a CanonicalMap<K, V>,
    key: K,
    value: Option<V>,
    original: Arc<V>,
}
impl<K: Hash + Eq + Clone, V: Clone> Deref for Edit<'_, K, V> {
    type Target = V;
    fn deref(&self) -> &V {
        self.value.as_ref().expect("broker edit owns its value")
    }
}
impl<K: Hash + Eq + Clone, V: Clone> DerefMut for Edit<'_, K, V> {
    fn deref_mut(&mut self) -> &mut V {
        self.value.as_mut().expect("broker edit owns its value")
    }
}
impl<K: Hash + Eq + Clone, V: Clone> Drop for Edit<'_, K, V> {
    fn drop(&mut self) {
        let Some(value) = self.value.take() else {
            return;
        };
        let value = Arc::new(value);
        self.map.0.pin().update(self.key.clone(), |current| {
            if Arc::ptr_eq(current, &self.original) {
                value.clone()
            } else {
                current.clone()
            }
        });
    }
}
pub(crate) struct Entry<'a, K: Hash + Eq + Clone, V: Clone> {
    map: &'a CanonicalMap<K, V>,
    key: K,
    modify: Option<Box<dyn FnOnce(&mut V) + 'a>>,
}
impl<'a, K: Hash + Eq + Clone, V: Clone> Entry<'a, K, V> {
    pub fn and_modify(mut self, f: impl FnOnce(&mut V) + 'a) -> Self {
        self.modify = Some(Box::new(f));
        self
    }
    pub fn or_insert(self, value: V) -> Edit<'a, K, V> {
        self.or_insert_with(|| value)
    }
    pub fn or_default(self) -> Edit<'a, K, V>
    where
        V: Default,
    {
        self.or_insert_with(V::default)
    }
    pub fn or_insert_with(self, f: impl FnOnce() -> V) -> Edit<'a, K, V> {
        let original = self
            .map
            .0
            .pin()
            .get_or_insert_with(self.key.clone(), || Arc::new(f()))
            .clone();
        let mut value = (*original).clone();
        if let Some(modify) = self.modify {
            modify(&mut value);
        }
        Edit {
            map: self.map,
            key: self.key,
            value: Some(value),
            original,
        }
    }
}
impl<K: Hash + Eq + Clone + Serialize, V: Clone + Serialize> Serialize for CanonicalMap<K, V> {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let guard = self.0.pin();
        let mut map = s.serialize_map(Some(guard.len()))?;
        for (key, value) in guard.iter() {
            map.serialize_entry(key, value.as_ref())?;
        }
        map.end()
    }
}
impl<'de, K: Hash + Eq + Clone + Deserialize<'de>, V: Clone + Deserialize<'de>> Deserialize<'de>
    for CanonicalMap<K, V>
{
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        struct Visitor<K, V>(std::marker::PhantomData<(K, V)>);
        impl<'de, K: Hash + Eq + Clone + Deserialize<'de>, V: Clone + Deserialize<'de>>
            serde::de::Visitor<'de> for Visitor<K, V>
        {
            type Value = CanonicalMap<K, V>;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("canonical broker map")
            }
            fn visit_map<M: serde::de::MapAccess<'de>>(
                self,
                mut m: M,
            ) -> Result<Self::Value, M::Error> {
                let values = CanonicalMap::default();
                while let Some((key, value)) = m.next_entry()? {
                    if values.insert(key, value).is_some() {
                        return Err(serde::de::Error::custom("duplicate broker key"));
                    }
                }
                Ok(values)
            }
        }
        d.deserialize_map(Visitor(std::marker::PhantomData))
    }
}
