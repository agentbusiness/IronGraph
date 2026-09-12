//! Bounded-path copy-on-write containers for published graph generations.

use std::{
    fmt,
    ops::{Index, IndexMut},
    sync::Arc,
};

use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{MapAccess, Visitor},
    ser::{SerializeMap, SerializeSeq},
};

use super::shared::SharedFlat;

const PAGE_BYTES: usize = 16 * 1024;
const MAX_PAGE_ELEMENTS: usize = 4_096;
const PAGES_PER_LEAF: usize = 256;
const RADIX_BITS: u32 = 4;
const RADIX_FANOUT: usize = 1 << RADIX_BITS;
const RADIX_LEVELS: u32 = 128 / RADIX_BITS;
const RADIX_LEAF_CAPACITY: usize = 8;

fn page_capacity<T>() -> usize {
    let width = size_of::<T>().max(1);
    (PAGE_BYTES / width).clamp(1, MAX_PAGE_ELEMENTS)
}

#[derive(Clone)]
enum ValuePage<T> {
    Owned(Vec<T>),
    Shared(SharedFlat<T>),
}

impl<T> ValuePage<T> {
    fn len(&self) -> usize {
        match self {
            Self::Owned(values) => values.len(),
            Self::Shared(values) => values.len(),
        }
    }

    fn as_slice(&self) -> &[T] {
        match self {
            Self::Owned(values) => values,
            Self::Shared(values) => values.as_slice(),
        }
    }

    fn get_mut(&mut self, index: usize) -> Option<&mut T>
    where
        T: Clone,
    {
        if let Self::Shared(values) = self {
            *self = Self::Owned(values.as_slice().to_vec());
        }
        match self {
            Self::Owned(values) => values.get_mut(index),
            Self::Shared(_) => unreachable!("shared page was detached"),
        }
    }

    fn push(&mut self, value: T)
    where
        T: Clone,
    {
        match self {
            Self::Owned(values) => values.push(value),
            Self::Shared(values) => {
                let mut detached = values.as_slice().to_vec();
                detached.push(value);
                *self = Self::Owned(detached);
            }
        }
    }

    fn index_mut(&mut self, index: usize) -> &mut T
    where
        T: Clone,
    {
        loop {
            match self {
                Self::Owned(values) => return &mut values[index],
                Self::Shared(values) => *self = Self::Owned(values.as_slice().to_vec()),
            }
        }
    }
}

#[derive(Clone)]
struct PageLeaf<T> {
    pages: Vec<Arc<ValuePage<T>>>,
    len: usize,
}

#[derive(Clone)]
struct PageDirectory<T> {
    // Retired leading leaves become `None` instead of shifting the complete directory. This
    // releases their value pages in bounded chunks while preserving stable locations for every
    // remaining leaf and every concurrently pinned generation.
    leaves: Vec<Option<Arc<PageLeaf<T>>>>,
    first_leaf: usize,
    len: usize,
}

/// A flat logical vector backed by a two-level immutable page directory.
///
/// Cloning is one `Arc` increment. A point write copies one 16-KiB page, one bounded leaf
/// directory, and the small root directory. Serialization remains a flat sequence.
pub struct PagedVec<T> {
    root: Arc<PageDirectory<T>>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ArenaSpan {
    pub page: u32,
    pub start: u32,
    pub len: u32,
}

/// Append-only typed arena. Small values share 16-KiB pages and every span remains contiguous;
/// replacing a row therefore appends one range and retargets one paged descriptor.
#[derive(Clone, Debug)]
pub struct PagedArena<T> {
    pages: PagedVec<Arc<Vec<T>>>,
    elements: usize,
}

impl<T> Default for PagedArena<T> {
    fn default() -> Self {
        Self {
            pages: PagedVec::default(),
            elements: 0,
        }
    }
}

impl<T> PagedArena<T> {
    pub fn append<I>(&mut self, values: I) -> Result<ArenaSpan, &'static str>
    where
        I: IntoIterator<Item = T>,
        T: Clone,
    {
        let values = values.into_iter().collect::<Vec<_>>();
        if values.is_empty() {
            return Ok(ArenaSpan::default());
        }
        let len = u32::try_from(values.len()).map_err(|_| "arena value range exceeds u32")?;
        let final_elements = self
            .elements
            .checked_add(len as usize)
            .ok_or("arena element count overflow")?;
        let capacity = page_capacity::<T>();
        let page_index = self.pages.len().saturating_sub(1);
        let reuse = self
            .pages
            .get(page_index)
            .is_some_and(|page| page.capacity().saturating_sub(page.len()) >= values.len());
        if reuse {
            let page = self
                .pages
                .get_mut(page_index)
                .ok_or("arena page directory is inconsistent")?;
            let page = Arc::make_mut(page);
            let start = u32::try_from(page.len()).map_err(|_| "arena page offset exceeds u32")?;
            page.extend(values);
            self.elements = final_elements;
            return Ok(ArenaSpan {
                page: u32::try_from(page_index).map_err(|_| "arena page count exceeds u32")?,
                start,
                len,
            });
        }
        let mut page = Vec::with_capacity(capacity.max(values.len()));
        page.extend(values);
        let page_index =
            u32::try_from(self.pages.len()).map_err(|_| "arena page count exceeds u32")?;
        self.pages.push(Arc::new(page));
        self.elements = final_elements;
        Ok(ArenaSpan {
            page: page_index,
            start: 0,
            len,
        })
    }

    #[must_use]
    pub fn get(&self, span: ArenaSpan) -> Option<&[T]> {
        if span.len == 0 {
            return Some(&[]);
        }
        let page = self.pages.get(span.page as usize)?;
        let start = span.start as usize;
        let end = start.checked_add(span.len as usize)?;
        page.get(start..end)
    }

    #[must_use]
    pub fn estimated_bytes(&self) -> usize {
        self.elements.saturating_mul(size_of::<T>())
    }

    #[must_use]
    pub const fn elements(&self) -> usize {
        self.elements
    }
}

impl<T> Default for PagedVec<T> {
    fn default() -> Self {
        Self {
            root: Arc::new(PageDirectory {
                leaves: Vec::new(),
                first_leaf: 0,
                len: 0,
            }),
        }
    }
}

impl<T> Clone for PagedVec<T> {
    fn clone(&self) -> Self {
        Self {
            root: Arc::clone(&self.root),
        }
    }
}

impl<T: fmt::Debug> fmt::Debug for PagedVec<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_list().entries(self.iter()).finish()
    }
}

impl<T> PagedVec<T> {
    /// Rebuilds only the bounded page directory over one immutable shared flat allocation. Value
    /// bytes remain owned by the allocation and are not copied. A later point mutation detaches
    /// exactly the touched page into ordinary host memory.
    pub fn from_shared(values: SharedFlat<T>) -> Result<Self, &'static str>
    where
        T: Clone,
    {
        let capacity = page_capacity::<T>();
        let mut leaves = Vec::<Option<Arc<PageLeaf<T>>>>::new();
        let mut offset = 0_usize;
        while offset < values.len() {
            let leaf_index = (offset / capacity) / PAGES_PER_LEAF;
            if leaves.len() == leaf_index {
                leaves.push(Some(Arc::new(PageLeaf {
                    pages: Vec::new(),
                    len: 0,
                })));
            }
            let page_len = capacity.min(values.len() - offset);
            let page = ValuePage::Shared(values.slice(offset, page_len)?);
            let leaf = leaves
                .get_mut(leaf_index)
                .and_then(Option::as_mut)
                .ok_or("shared page directory is inconsistent")?;
            let leaf = Arc::make_mut(leaf);
            leaf.pages.push(Arc::new(page));
            leaf.len += page_len;
            offset += page_len;
        }
        Ok(Self {
            root: Arc::new(PageDirectory {
                leaves,
                first_leaf: 0,
                len: values.len(),
            }),
        })
    }

    #[cfg_attr(
        not(all(feature = "accelerator", any(target_os = "macos", target_os = "ios"))),
        allow(dead_code)
    )]
    pub fn rebase_shared(&mut self, values: SharedFlat<T>) -> Result<(), &'static str>
    where
        T: Clone,
    {
        *self = Self::from_shared(values)?;
        Ok(())
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.root.len
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.root.len == 0
    }

    fn location(index: usize) -> (usize, usize, usize) {
        let page_capacity = page_capacity::<T>();
        let page = index / page_capacity;
        (
            page / PAGES_PER_LEAF,
            page % PAGES_PER_LEAF,
            index % page_capacity,
        )
    }

    #[must_use]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.root.len {
            return None;
        }
        let (leaf, page, offset) = Self::location(index);
        self.root
            .leaves
            .get(self.root.first_leaf.checked_add(leaf)?)?
            .as_ref()?
            .pages
            .get(page)?
            .as_slice()
            .get(offset)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T>
    where
        T: Clone,
    {
        if index >= self.root.len {
            return None;
        }
        let (leaf_index, page_index, offset) = Self::location(index);
        let directory = Arc::make_mut(&mut self.root);
        let physical_leaf = directory.first_leaf.checked_add(leaf_index)?;
        let leaf = Arc::make_mut(directory.leaves.get_mut(physical_leaf)?.as_mut()?);
        Arc::make_mut(leaf.pages.get_mut(page_index)?).get_mut(offset)
    }

    pub fn replace(&mut self, index: usize, value: T) -> Result<(), &'static str>
    where
        T: Clone,
    {
        let slot = self
            .get_mut(index)
            .ok_or("paged vector index is out of bounds")?;
        *slot = value;
        Ok(())
    }

    pub fn push(&mut self, value: T)
    where
        T: Clone,
    {
        let capacity = page_capacity::<T>();
        let directory = Arc::make_mut(&mut self.root);
        let needs_leaf = directory
            .leaves
            .last()
            .and_then(Option::as_ref)
            .is_none_or(|leaf| {
                leaf.pages.len() == PAGES_PER_LEAF
                    && leaf.pages.last().is_some_and(|page| page.len() == capacity)
            });
        if needs_leaf {
            directory.leaves.push(Some(Arc::new(PageLeaf {
                pages: Vec::new(),
                len: 0,
            })));
        }
        let leaf = loop {
            if let Some(leaf) = directory.leaves.last_mut().and_then(Option::as_mut) {
                break Arc::make_mut(leaf);
            }
            directory.leaves.push(Some(Arc::new(PageLeaf {
                pages: Vec::new(),
                len: 0,
            })));
        };
        if leaf.pages.last().is_none_or(|page| page.len() == capacity) {
            leaf.pages
                .push(Arc::new(ValuePage::Owned(Vec::with_capacity(capacity))));
        }
        let page_index = leaf.pages.len() - 1;
        Arc::make_mut(&mut leaf.pages[page_index]).push(value);
        leaf.len += 1;
        directory.len += 1;
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> + DoubleEndedIterator {
        self.root
            .leaves
            .get(self.root.first_leaf..)
            .into_iter()
            .flatten()
            .filter_map(Option::as_ref)
            .flat_map(|leaf| leaf.pages.iter())
            .flat_map(|page| page.as_slice().iter())
    }

    /// Releases complete leading leaf allocations covered by `elements` and returns the exact
    /// number of removed values. At most the small root directory and the removed leaf slots are
    /// detached; remaining value pages are never copied or shifted.
    pub fn discard_prefix_leaves(&mut self, elements: usize) -> usize
    where
        T: Clone,
    {
        let mut removable = 0_usize;
        let mut leaves = 0_usize;
        for leaf in self
            .root
            .leaves
            .get(self.root.first_leaf..)
            .into_iter()
            .flatten()
            .filter_map(Option::as_ref)
        {
            let Some(next) = removable.checked_add(leaf.len) else {
                break;
            };
            if next > elements {
                break;
            }
            removable = next;
            leaves += 1;
        }
        if leaves == 0 {
            return 0;
        }
        let directory = Arc::make_mut(&mut self.root);
        let end = directory.first_leaf.saturating_add(leaves);
        for leaf in &mut directory.leaves[directory.first_leaf..end] {
            *leaf = None;
        }
        directory.first_leaf = end;
        directory.len = directory.len.saturating_sub(removable);
        if directory.len == 0 {
            directory.leaves.clear();
            directory.first_leaf = 0;
        }
        removable
    }

    #[must_use]
    pub fn first_leaf_len(&self) -> usize {
        self.root
            .leaves
            .get(self.root.first_leaf)
            .and_then(Option::as_ref)
            .map_or(0, |leaf| leaf.len)
    }

    #[must_use]
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.iter().cloned().collect()
    }

    /// Actual value-page bytes no longer shared with `previous` at the same logical positions.
    /// This inspects allocation identity and is used by scale tests rather than an estimated
    /// mutation counter.
    #[doc(hidden)]
    pub fn detached_page_bytes_from(&self, previous: &Self) -> usize {
        self.root
            .leaves
            .iter()
            .enumerate()
            .filter_map(|(leaf_index, leaf)| leaf.as_ref().map(|leaf| (leaf_index, leaf)))
            .flat_map(|(leaf_index, leaf)| {
                leaf.pages
                    .iter()
                    .enumerate()
                    .map(move |(page_index, page)| {
                        let shared = previous
                            .root
                            .leaves
                            .get(leaf_index)
                            .and_then(Option::as_ref)
                            .and_then(|leaf| leaf.pages.get(page_index))
                            .is_some_and(|old| Arc::ptr_eq(old, page));
                        (!shared)
                            .then_some(page.len().saturating_mul(size_of::<T>()))
                            .unwrap_or(0)
                    })
            })
            .sum()
    }

    #[cfg(test)]
    pub fn shared_value_bytes(&self) -> usize {
        self.root
            .leaves
            .iter()
            .filter_map(Option::as_ref)
            .flat_map(|leaf| leaf.pages.iter())
            .filter_map(|page| match page.as_ref() {
                ValuePage::Shared(values) => Some(values.len().saturating_mul(size_of::<T>())),
                ValuePage::Owned(_) => None,
            })
            .sum()
    }
}

impl<T> Index<usize> for PagedVec<T> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        let (leaf, page, offset) = Self::location(index);
        let physical_leaf = self.root.first_leaf + leaf;
        let pages = self.root.leaves[physical_leaf]
            .as_ref()
            .map_or(&[][..], |leaf| leaf.pages.as_slice());
        &pages[page].as_slice()[offset]
    }
}

impl<T: Clone> IndexMut<usize> for PagedVec<T> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        let (leaf, page, offset) = Self::location(index);
        let directory = Arc::make_mut(&mut self.root);
        let physical_leaf = directory.first_leaf + leaf;
        let leaf = directory.leaves[physical_leaf].get_or_insert_with(|| {
            Arc::new(PageLeaf {
                pages: Vec::new(),
                len: 0,
            })
        });
        let leaf = Arc::make_mut(leaf);
        Arc::make_mut(&mut leaf.pages[page]).index_mut(offset)
    }
}

impl<T: PartialEq> PartialEq for PagedVec<T> {
    fn eq(&self, other: &Self) -> bool {
        self.len() == other.len() && self.iter().eq(other.iter())
    }
}

impl<T: Eq> Eq for PagedVec<T> {}

impl<T: Serialize> Serialize for PagedVec<T> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
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

impl<'de, T> Deserialize<'de> for PagedVec<T>
where
    T: Deserialize<'de> + Clone,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct PagedVecVisitor<T>(std::marker::PhantomData<T>);

        impl<'de, T> Visitor<'de> for PagedVecVisitor<T>
        where
            T: Deserialize<'de> + Clone,
        {
            type Value = PagedVec<T>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a sequence of paged values")
            }

            fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
            where
                A: serde::de::SeqAccess<'de>,
            {
                let mut paged = PagedVec::default();
                while let Some(value) = sequence.next_element::<T>()? {
                    paged.push(value);
                }
                Ok(paged)
            }
        }

        deserializer.deserialize_seq(PagedVecVisitor(std::marker::PhantomData))
    }
}

#[derive(Clone)]
enum RadixNode<V> {
    Branch(Box<[Option<Arc<RadixNode<V>>>; RADIX_FANOUT]>),
    Leaf(Vec<(u128, V)>),
}

/// Persistent fixed-depth radix map. A write copies at most 32 small branch paths regardless of
/// project size; stable numeric keys make traversal and serialization deterministic.
pub struct PersistentMap<V> {
    root: Option<Arc<RadixNode<V>>>,
    len: usize,
}

/// Places a 64-bit stable graph identity in the high half of the fixed 128-bit radix key. Older
/// snapshots stored the identity in the low half; [`stable_id_row`] retains that fallback while
/// new writes avoid sixteen guaranteed leading-zero branch levels.
#[must_use]
pub const fn stable_id_key(id: u64) -> u128 {
    (id as u128) << 64
}

/// Resolves both the current high-half encoding and the legacy raw encoding.
#[must_use]
pub fn stable_id_row(map: &PersistentMap<u32>, id: u64) -> Option<&u32> {
    map.get(stable_id_key(id))
        .or_else(|| map.get(u128::from(id)))
}

impl<V> Default for PersistentMap<V> {
    fn default() -> Self {
        Self { root: None, len: 0 }
    }
}

impl<V> Clone for PersistentMap<V> {
    fn clone(&self) -> Self {
        Self {
            root: self.root.clone(),
            len: self.len,
        }
    }
}

impl<V: fmt::Debug> fmt::Debug for PersistentMap<V> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_map().entries(self.iter()).finish()
    }
}

impl<V> PersistentMap<V> {
    #[doc(hidden)]
    pub fn shared_with(&self, other: &Self) -> bool {
        match (&self.root, &other.root) {
            (Some(first), Some(second)) => Arc::ptr_eq(first, second),
            (None, None) => true,
            _ => false,
        }
    }

    #[must_use]
    pub const fn len(&self) -> usize {
        self.len
    }

    #[must_use]
    pub fn get(&self, key: u128) -> Option<&V> {
        let mut node = self.root.as_deref()?;
        for depth in 0..=RADIX_LEVELS {
            match node {
                RadixNode::Branch(children) => {
                    if depth == RADIX_LEVELS {
                        return None;
                    }
                    let shift = 128 - RADIX_BITS * (depth + 1);
                    let slot = ((key >> shift) & ((1 << RADIX_BITS) - 1)) as usize;
                    node = children[slot].as_deref()?;
                }
                RadixNode::Leaf(entries) => {
                    return entries
                        .binary_search_by_key(&key, |(stored, _)| *stored)
                        .ok()
                        .and_then(|index| entries.get(index))
                        .map(|(_, value)| value);
                }
            }
        }
        None
    }

    #[must_use]
    pub fn contains_key(&self, key: u128) -> bool {
        self.get(key).is_some()
    }

    pub fn insert(&mut self, key: u128, value: V) -> Option<V>
    where
        V: Clone,
    {
        let previous = self.get(key).cloned();
        self.root = Some(insert_radix(self.root.as_ref(), 0, key, value));
        if previous.is_none() {
            self.len += 1;
        }
        previous
    }

    /// Inserts into an unpublished COW generation without allocating an immutable intermediate
    /// root for every row in a batch. The first touched path detaches from any pinned generation;
    /// later inserts mutate only branches already unique to this map.
    pub fn insert_cow(&mut self, key: u128, value: V) -> Option<V>
    where
        V: Clone,
    {
        let previous = self.get(key).cloned();
        match self.root.as_mut() {
            Some(root) => insert_radix_cow(root, 0, key, value),
            None => {
                self.root = Some(Arc::new(RadixNode::Leaf(vec![(key, value)])));
            }
        }
        if previous.is_none() {
            self.len += 1;
        }
        previous
    }

    pub fn get_mut(&mut self, key: u128) -> Option<&mut V>
    where
        V: Clone,
    {
        get_radix_mut(Arc::make_mut(self.root.as_mut()?), 0, key)
    }

    pub fn remove(&mut self, key: u128) -> Option<V>
    where
        V: Clone,
    {
        let (root, removed) = remove_radix(self.root.as_ref(), 0, key);
        if removed.is_some() {
            self.root = root;
            self.len = self.len.saturating_sub(1);
        }
        removed
    }

    pub fn iter(&self) -> PersistentMapIter<'_, V> {
        let mut stack = Vec::new();
        if let Some(root) = self.root.as_deref() {
            stack.push(root);
        }
        PersistentMapIter { stack, leaf: None }
    }

    /// Bytes occupied by radix nodes and branch tables in this generation that are not shared
    /// with `previous`. Allocation identity, rather than an operation counter, determines sharing.
    #[cfg(test)]
    pub fn detached_node_bytes_from(&self, previous: &Self) -> usize {
        fn detached<V>(
            current: Option<&Arc<RadixNode<V>>>,
            previous: Option<&Arc<RadixNode<V>>>,
        ) -> usize {
            let Some(current) = current else {
                return 0;
            };
            if previous.is_some_and(|previous| Arc::ptr_eq(current, previous)) {
                return 0;
            }
            let allocation = size_of::<RadixNode<V>>()
                .saturating_add(2_usize.saturating_mul(size_of::<usize>()));
            match (current.as_ref(), previous.map(AsRef::as_ref)) {
                (RadixNode::Branch(children), Some(RadixNode::Branch(old_children))) => allocation
                    .saturating_add(size_of_val(children.as_ref()))
                    .saturating_add(
                        children
                            .iter()
                            .zip(old_children.iter())
                            .map(|(child, old_child)| detached(child.as_ref(), old_child.as_ref()))
                            .fold(0_usize, usize::saturating_add),
                    ),
                (RadixNode::Branch(children), _) => allocation
                    .saturating_add(size_of_val(children.as_ref()))
                    .saturating_add(
                        children
                            .iter()
                            .map(|child| detached(child.as_ref(), None))
                            .fold(0_usize, usize::saturating_add),
                    ),
                (RadixNode::Leaf(entries), _) => allocation
                    .saturating_add(entries.capacity().saturating_mul(size_of::<(u128, V)>())),
            }
        }

        detached(self.root.as_ref(), previous.root.as_ref())
    }
}

fn get_radix_mut<V: Clone>(node: &mut RadixNode<V>, depth: u32, key: u128) -> Option<&mut V> {
    match node {
        RadixNode::Branch(children) if depth < RADIX_LEVELS => {
            let shift = 128 - RADIX_BITS * (depth + 1);
            let slot = ((key >> shift) & ((1 << RADIX_BITS) - 1)) as usize;
            get_radix_mut(Arc::make_mut(children[slot].as_mut()?), depth + 1, key)
        }
        RadixNode::Leaf(entries) => entries
            .binary_search_by_key(&key, |(stored, _)| *stored)
            .ok()
            .and_then(|index| entries.get_mut(index))
            .map(|(_, value)| value),
        RadixNode::Branch(_) => None,
    }
}

fn remove_radix<V: Clone>(
    current: Option<&Arc<RadixNode<V>>>,
    depth: u32,
    key: u128,
) -> (Option<Arc<RadixNode<V>>>, Option<V>) {
    match current.map(AsRef::as_ref) {
        Some(RadixNode::Leaf(current_entries)) => {
            let Ok(index) = current_entries.binary_search_by_key(&key, |(stored, _)| *stored)
            else {
                return (current.cloned(), None);
            };
            let mut entries = current_entries.clone();
            let (_, removed) = entries.remove(index);
            let root = (!entries.is_empty()).then(|| Arc::new(RadixNode::Leaf(entries)));
            (root, Some(removed))
        }
        Some(RadixNode::Branch(current_children)) if depth < RADIX_LEVELS => {
            let shift = 128 - RADIX_BITS * (depth + 1);
            let slot = ((key >> shift) & ((1 << RADIX_BITS) - 1)) as usize;
            let (replacement, removed) =
                remove_radix(current_children[slot].as_ref(), depth + 1, key);
            if removed.is_none() {
                return (current.cloned(), None);
            }
            let mut children = current_children.clone();
            children[slot] = replacement;
            let root = children
                .iter()
                .any(Option::is_some)
                .then(|| Arc::new(RadixNode::Branch(children)));
            (root, removed)
        }
        Some(RadixNode::Branch(_)) | None => (current.cloned(), None),
    }
}

fn insert_radix<V: Clone>(
    current: Option<&Arc<RadixNode<V>>>,
    depth: u32,
    key: u128,
    value: V,
) -> Arc<RadixNode<V>> {
    match current.map(AsRef::as_ref) {
        Some(RadixNode::Branch(current_children)) if depth < RADIX_LEVELS => {
            let mut children = current_children.clone();
            let shift = 128 - RADIX_BITS * (depth + 1);
            let slot = ((key >> shift) & ((1 << RADIX_BITS) - 1)) as usize;
            children[slot] = Some(insert_radix(children[slot].as_ref(), depth + 1, key, value));
            Arc::new(RadixNode::Branch(children))
        }
        Some(RadixNode::Leaf(current_entries)) => {
            let mut entries = current_entries.clone();
            match entries.binary_search_by_key(&key, |(stored, _)| *stored) {
                Ok(index) => entries[index].1 = value,
                Err(index) => entries.insert(index, (key, value)),
            }
            if entries.len() <= RADIX_LEAF_CAPACITY || depth == RADIX_LEVELS {
                return Arc::new(RadixNode::Leaf(entries));
            }
            let mut children: Box<[Option<Arc<RadixNode<V>>>; RADIX_FANOUT]> =
                Box::new(std::array::from_fn(|_| None));
            let shift = 128 - RADIX_BITS * (depth + 1);
            for (entry_key, entry_value) in entries {
                let slot = ((entry_key >> shift) & ((1 << RADIX_BITS) - 1)) as usize;
                children[slot] = Some(insert_radix(
                    children[slot].as_ref(),
                    depth + 1,
                    entry_key,
                    entry_value,
                ));
            }
            Arc::new(RadixNode::Branch(children))
        }
        Some(RadixNode::Branch(_)) | None => Arc::new(RadixNode::Leaf(vec![(key, value)])),
    }
}

fn insert_radix_cow<V: Clone>(current: &mut Arc<RadixNode<V>>, depth: u32, key: u128, value: V) {
    let node = Arc::make_mut(current);
    match node {
        RadixNode::Branch(children) if depth < RADIX_LEVELS => {
            let shift = 128 - RADIX_BITS * (depth + 1);
            let slot = ((key >> shift) & ((1 << RADIX_BITS) - 1)) as usize;
            match children[slot].as_mut() {
                Some(child) => insert_radix_cow(child, depth + 1, key, value),
                None => {
                    children[slot] = Some(Arc::new(RadixNode::Leaf(vec![(key, value)])));
                }
            }
        }
        RadixNode::Leaf(entries) => {
            match entries.binary_search_by_key(&key, |(stored, _)| *stored) {
                Ok(index) => entries[index].1 = value,
                Err(index) => entries.insert(index, (key, value)),
            }
            if entries.len() <= RADIX_LEAF_CAPACITY || depth == RADIX_LEVELS {
                return;
            }
            let split = std::mem::take(entries);
            let mut children: Box<[Option<Arc<RadixNode<V>>>; RADIX_FANOUT]> =
                Box::new(std::array::from_fn(|_| None));
            let shift = 128 - RADIX_BITS * (depth + 1);
            for (entry_key, entry_value) in split {
                let slot = ((entry_key >> shift) & ((1 << RADIX_BITS) - 1)) as usize;
                match children[slot].as_mut() {
                    Some(child) => {
                        insert_radix_cow(child, depth + 1, entry_key, entry_value);
                    }
                    None => {
                        children[slot] =
                            Some(Arc::new(RadixNode::Leaf(vec![(entry_key, entry_value)])));
                    }
                }
            }
            *node = RadixNode::Branch(children);
        }
        RadixNode::Branch(_) => unreachable!("radix branch exceeded its fixed depth"),
    }
}

pub struct PersistentMapIter<'a, V> {
    stack: Vec<&'a RadixNode<V>>,
    leaf: Option<std::slice::Iter<'a, (u128, V)>>,
}

impl<'a, V> Iterator for PersistentMapIter<'a, V> {
    type Item = (u128, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(entries) = &mut self.leaf
                && let Some((key, value)) = entries.next()
            {
                return Some((*key, value));
            }
            self.leaf = None;
            let node = self.stack.pop()?;
            match node {
                RadixNode::Leaf(entries) => self.leaf = Some(entries.iter()),
                RadixNode::Branch(children) => {
                    for child in children.iter().rev().flatten() {
                        self.stack.push(child);
                    }
                }
            }
        }
    }
}

impl<V: Serialize> Serialize for PersistentMap<V> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(self.len))?;
        for (key, value) in self.iter() {
            map.serialize_entry(&key, value)?;
        }
        map.end()
    }
}

impl<'de, V> Deserialize<'de> for PersistentMap<V>
where
    V: Deserialize<'de> + Clone,
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct PersistentMapVisitor<V>(std::marker::PhantomData<V>);

        impl<'de, V> Visitor<'de> for PersistentMapVisitor<V>
        where
            V: Deserialize<'de> + Clone,
        {
            type Value = PersistentMap<V>;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a persistent numeric-key map")
            }

            fn visit_map<A>(self, mut entries: A) -> Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut map = PersistentMap::default();
                while let Some((key, value)) = entries.next_entry::<u128, V>()? {
                    if map.insert(key, value).is_some() {
                        return Err(serde::de::Error::custom(
                            "persistent map contains a duplicate key",
                        ));
                    }
                }
                Ok(map)
            }
        }

        deserializer.deserialize_map(PersistentMapVisitor(std::marker::PhantomData))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::shared::SharedFlat;

    use super::{
        PAGE_BYTES, PAGES_PER_LEAF, PagedVec, PersistentMap, page_capacity, stable_id_key,
        stable_id_row,
    };

    #[test]
    fn point_update_detaches_one_value_page() {
        let mut original = PagedVec::default();
        for value in 0_u64..100_000 {
            original.push(value);
        }
        let mut updated = original.clone();
        updated[50_000] = 7;
        assert_eq!(original[50_000], 50_000);
        assert_eq!(updated[50_000], 7);
        assert!(updated.detached_page_bytes_from(&original) <= PAGE_BYTES);
    }

    #[test]
    fn out_of_bounds_point_replacement_is_rejected_without_mutation() {
        let mut values = PagedVec::default();
        values.push(11_u64);
        assert_eq!(
            values.replace(1, 22),
            Err("paged vector index is out of bounds")
        );
        assert_eq!(values.get(0), Some(&11));
        assert_eq!(values.len(), 1);
    }

    #[test]
    fn shared_flat_pages_are_zero_copy_and_point_updates_detach_one_page() {
        let values: Arc<[u64]> = (0_u64..100_000).collect::<Vec<_>>().into();
        let flat = SharedFlat::from_arc_slice(values.clone()).expect("shared view");
        let mut paged = PagedVec::from_shared(flat).expect("shared pages");
        assert_eq!(paged.shared_value_bytes(), values.len() * size_of::<u64>());
        assert_eq!(paged.get(50_000), Some(&50_000));
        assert!(std::ptr::eq(
            paged.get(50_000).expect("paged value"),
            &values[50_000]
        ));

        let pinned = paged.clone();
        paged[50_000] = 7;
        assert_eq!(pinned[50_000], 50_000);
        assert_eq!(paged[50_000], 7);
        assert_eq!(
            paged.shared_value_bytes(),
            values.len() * size_of::<u64>() - page_capacity::<u64>() * size_of::<u64>()
        );
        assert!(paged.detached_page_bytes_from(&pinned) <= PAGE_BYTES);
    }

    #[test]
    fn append_to_partial_shared_page_detaches_and_preserves_pinned_generation() {
        let values: Arc<[u64]> = vec![11, 22, 33].into();
        let flat = SharedFlat::from_arc_slice(values.clone()).expect("shared view");
        let mut paged = PagedVec::from_shared(flat).expect("shared pages");
        let pinned = paged.clone();

        paged.push(44);

        assert_eq!(pinned.to_vec(), vec![11, 22, 33]);
        assert_eq!(paged.to_vec(), vec![11, 22, 33, 44]);
        assert_eq!(pinned.shared_value_bytes(), 3 * size_of::<u64>());
        assert_eq!(paged.shared_value_bytes(), 0);
        assert!(paged.detached_page_bytes_from(&pinned) <= PAGE_BYTES);
    }

    #[test]
    fn complete_prefix_leaves_retire_without_copying_the_remaining_generation() {
        let leaf_elements = page_capacity::<u64>() * PAGES_PER_LEAF;
        let mut current = PagedVec::default();
        for value in 0..leaf_elements + 17 {
            current.push(value as u64);
        }
        let pinned = current.clone();
        assert_eq!(current.discard_prefix_leaves(leaf_elements - 1), 0);
        assert_eq!(current.discard_prefix_leaves(leaf_elements), leaf_elements);
        assert_eq!(current.len(), 17);
        assert_eq!(current.get(0), Some(&(leaf_elements as u64)));
        assert_eq!(current.get(16), Some(&(leaf_elements as u64 + 16)));
        assert_eq!(pinned.len(), leaf_elements + 17);
        assert_eq!(pinned.get(0), Some(&0));
        assert_eq!(pinned.get(leaf_elements), Some(&(leaf_elements as u64)));
        assert_eq!(current.detached_page_bytes_from(&pinned), 0);
    }

    #[test]
    fn radix_map_preserves_old_generation() {
        let mut first = PersistentMap::default();
        first.insert(u128::MAX, 1_u32);
        let mut second = first.clone();
        second.insert(7, 2);
        second.insert(u128::MAX, 3);
        assert_eq!(first.get(u128::MAX), Some(&1));
        assert_eq!(first.get(7), None);
        assert_eq!(second.get(u128::MAX), Some(&3));
        assert_eq!(second.get(7), Some(&2));
    }

    #[test]
    fn radix_cow_batch_preserves_pinned_generation_and_matches_persistent_insert() {
        let mut base = PersistentMap::default();
        for key in 0_u128..10_000 {
            base.insert(key.saturating_mul(97), key as u32);
        }
        let pinned = base.clone();
        let mut cow = base.clone();
        let mut persistent = base;
        for key in 10_000_u128..12_000 {
            let key = key.saturating_mul(97);
            assert_eq!(cow.insert_cow(key, key as u32), None);
            assert_eq!(persistent.insert(key, key as u32), None);
        }
        assert_eq!(cow.len(), persistent.len());
        assert!(cow.iter().eq(persistent.iter()));
        assert_eq!(pinned.len(), 10_000);
        assert_eq!(pinned.get(11_000_u128 * 97), None);
    }

    #[test]
    fn stable_id_lookup_reads_legacy_and_high_half_generations() {
        let mut legacy = PersistentMap::default();
        legacy.insert(u128::from(42_u64), 7_u32);
        assert_eq!(stable_id_row(&legacy, 42), Some(&7));

        let mut current = legacy.clone();
        current.insert(stable_id_key(43), 8);
        assert_eq!(stable_id_row(&current, 42), Some(&7));
        assert_eq!(stable_id_row(&current, 43), Some(&8));
        assert_eq!(stable_id_key(43), u128::from(43_u64) << 64);
    }

    #[test]
    fn radix_point_mutation_and_removal_preserve_published_generations() {
        let mut original = PersistentMap::default();
        for key in 1_u128..10_000 {
            original.insert(key, key as u64);
        }
        let mut updated = original.clone();
        *updated.get_mut(5_000).expect("key must exist") = 7;
        assert_eq!(updated.remove(9_000), Some(9_000));
        assert_eq!(original.get(5_000), Some(&5_000));
        assert_eq!(original.get(9_000), Some(&9_000));
        assert_eq!(updated.get(5_000), Some(&7));
        assert_eq!(updated.get(9_000), None);
        assert_eq!(updated.len(), original.len() - 1);
        assert!(updated.detached_node_bytes_from(&original) < 64 * 1024);
    }
}
