//! Small transparent copy-on-write owner used by immutable published state across crates.

use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
};

use serde::{Deserialize, Deserializer, Serialize, Serializer};

/// An `Arc` whose serialized form is exactly the owned value.
///
/// Cloning a database snapshot shares the value. The first mutation detaches only that value,
/// which keeps unrelated canonical state out of a mutation's staging working set.
#[derive(Debug, Default)]
pub struct CowArc<T>(Arc<T>);

impl<T> CowArc<T> {
    #[must_use]
    pub fn new(value: T) -> Self {
        Self(Arc::new(value))
    }

    #[must_use]
    #[doc(hidden)]
    pub fn shared_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
}

impl<T> Clone for CowArc<T> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl<T> From<T> for CowArc<T> {
    fn from(value: T) -> Self {
        Self::new(value)
    }
}

impl<T> Deref for CowArc<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<T: Clone> DerefMut for CowArc<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Arc::make_mut(&mut self.0)
    }
}

impl<T: Serialize> Serialize for CowArc<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for CowArc<T> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        T::deserialize(deserializer).map(Self::new)
    }
}

#[cfg(test)]
mod tests {
    use super::CowArc;

    #[test]
    fn clone_shares_until_mutated() {
        let first = CowArc::new(vec![1_u64, 2]);
        let mut second = first.clone();
        assert!(first.shared_with(&second));
        second.push(3);
        assert!(!first.shared_with(&second));
        assert_eq!(&*first, &[1, 2]);
        assert_eq!(&*second, &[1, 2, 3]);
    }
}
