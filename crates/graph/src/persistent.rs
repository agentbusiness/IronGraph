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
            Self::Owned(values) => {
                if values.len() == values.capacity() && values.len() > page_capacity::<T>() / 2 {
                    values.reserve_exact(page_capacity::<T>().saturating_sub(values.len()));
                }
                values.push(value);
            }
            Self::Shared(values) => {
                let mut detached = if values.len() > page_capacity::<T>() / 2 {
                    let mut detached = Vec::with_capacity(page_capacity::<T>());
                    detached.extend_from_slice(values.as_slice());
                    detached
                } else {
                    values.as_slice().to_vec()
                };
                detached.push(value);
                *self = Self::Owned(detached);
            }
        }
    }
}

#[derive(Clone)]
struct PageLeaf<T> {
    pages: Vec<Option<Arc<ValuePage<T>>>>,
}

#[derive(Clone)]
struct PageDirectory<T> {
    leaves: PersistentMap<Arc<PageLeaf<T>>>,
    first_page: usize,
    len: usize,
    default: Option<Arc<T>>,
}

impl<T> PageDirectory<T> {
    fn page(&self, page: usize) -> Option<&Arc<ValuePage<T>>> {
        self.leaves
            .get(((page / PAGES_PER_LEAF) as u128).reverse_bits())?
            .pages
            .get(page % PAGES_PER_LEAF)?
            .as_ref()
    }
}

impl<T: Clone> PageDirectory<T> {
    #[allow(clippy::expect_used)] // The missing leaf is inserted immediately before this lookup.
    fn page_mut(&mut self, page: usize) -> &mut Option<Arc<ValuePage<T>>> {
        let key = ((page / PAGES_PER_LEAF) as u128).reverse_bits();
        if !self.leaves.contains_key(key) {
            self.leaves
                .insert_cow(key, Arc::new(PageLeaf { pages: Vec::new() }));
        }
        let leaf = Arc::make_mut(self.leaves.get_mut(key).expect("page leaf was inserted"));
        let slot = page % PAGES_PER_LEAF;
        if leaf.pages.len() <= slot {
            leaf.pages.resize_with(slot + 1, || None);
        }
        &mut leaf.pages[slot]
    }
}

/// A flat logical vector backed by an immutable radix page directory.
///
/// Cloning is one `Arc` increment. A point write copies one 16-KiB page, at most 256 page
/// references, and a bounded radix path. Sparse default rows have no allocated page.
/// Serialization remains a flat sequence.
pub struct PagedVec<T> {
    root: Arc<PageDirectory<T>>,
}

impl<T> Default for PagedVec<T> {
    fn default() -> Self {
        Self {
            root: Arc::new(PageDirectory {
                leaves: PersistentMap::default(),
                first_page: 0,
                len: 0,
                default: None,
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
        let mut directory = PageDirectory {
            leaves: PersistentMap::default(),
            first_page: 0,
            len: values.len(),
            default: None,
        };
        let mut offset = 0_usize;
        while offset < values.len() {
            let page_len = capacity.min(values.len() - offset);
            let page = ValuePage::Shared(values.slice(offset, page_len)?);
            *directory.page_mut(offset / capacity) = Some(Arc::new(page));
            offset += page_len;
        }
        Ok(Self {
            root: Arc::new(directory),
        })
    }

    /// Creates a logical repeated vector without allocating or visiting its rows. Only pages
    /// subsequently changed become materialized; checkpoints still serialize a flat sequence.
    pub fn repeat(value: T, len: usize) -> Self {
        Self {
            root: Arc::new(PageDirectory {
                leaves: PersistentMap::default(),
                first_page: 0,
                len,
                default: Some(Arc::new(value)),
            }),
        }
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

    #[must_use]
    pub fn get(&self, index: usize) -> Option<&T> {
        if index >= self.root.len {
            return None;
        }
        let capacity = page_capacity::<T>();
        let page = self.root.first_page.checked_add(index / capacity)?;
        self.root
            .page(page)
            .and_then(|page| page.as_slice().get(index % capacity))
            .or(self.root.default.as_deref())
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T>
    where
        T: Clone,
    {
        if index >= self.root.len {
            return None;
        }
        let capacity = page_capacity::<T>();
        let directory = Arc::make_mut(&mut self.root);
        let page_index = directory.first_page.checked_add(index / capacity)?;
        if directory.page(page_index).is_none() {
            let value = directory.default.as_deref()?.clone();
            let len = capacity.min(directory.len - index / capacity * capacity);
            *directory.page_mut(page_index) = Some(Arc::new(ValuePage::Owned(vec![value; len])));
        }
        Arc::make_mut(directory.page_mut(page_index).as_mut()?).get_mut(index % capacity)
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
        let page_index = directory.first_page + directory.len / capacity;
        let offset = directory.len % capacity;
        let default = directory.default.clone();
        let page = directory.page_mut(page_index).get_or_insert_with(|| {
            let mut values = Vec::with_capacity(capacity);
            if offset != 0
                && let Some(default) = default.as_deref()
            {
                values.resize(offset, default.clone());
            }
            Arc::new(ValuePage::Owned(values))
        });
        Arc::make_mut(page).push(value);
        directory.len += 1;
    }

    pub fn iter(&self) -> impl Iterator<Item = &T> + DoubleEndedIterator {
        let capacity = page_capacity::<T>();
        (0..self.len().div_ceil(capacity)).flat_map(move |page| {
            let len = capacity.min(self.len() - page * capacity);
            let values = self.root.page(self.root.first_page + page);
            let allocated = values.map_or(&[][..], |values| &values.as_slice()[..len]);
            allocated.iter().chain(
                self.root
                    .default
                    .as_deref()
                    .into_iter()
                    .flat_map(move |value| std::iter::repeat_n(value, len - allocated.len())),
            )
        })
    }

    /// Releases complete leading leaf allocations covered by `elements` and returns the exact
    /// number of removed values. At most the small root directory and the removed leaf slots are
    /// detached; remaining value pages are never copied or shifted.
    pub fn discard_prefix_leaves(&mut self, elements: usize) -> usize
    where
        T: Clone,
    {
        let capacity = page_capacity::<T>();
        let leaf_capacity = capacity * PAGES_PER_LEAF;
        let removable = if elements >= self.len() {
            self.len()
        } else {
            elements / leaf_capacity * leaf_capacity
        };
        if removable == 0 {
            return 0;
        }
        let directory = Arc::make_mut(&mut self.root);
        let page_count = removable.div_ceil(capacity);
        for leaf in directory.first_page / PAGES_PER_LEAF
            ..(directory.first_page + page_count).div_ceil(PAGES_PER_LEAF)
        {
            directory.leaves.remove((leaf as u128).reverse_bits());
        }
        directory.first_page += page_count;
        directory.len = directory.len.saturating_sub(removable);
        if directory.len == 0 {
            directory.leaves = PersistentMap::default();
            directory.first_page = 0;
        }
        removable
    }

    #[must_use]
    pub fn first_leaf_len(&self) -> usize {
        self.len().min(page_capacity::<T>() * PAGES_PER_LEAF)
    }

    #[must_use]
    pub fn to_vec(&self) -> Vec<T>
    where
        T: Clone,
    {
        self.iter().cloned().collect()
    }

    /// Inspects materialized value allocations, excluding unmaterialized default rows. This
    /// instrumentation walks allocated pages; it must not be used to plan a point mutation.
    #[doc(hidden)]
    pub fn allocated_value_bytes(&self) -> usize {
        self.root
            .leaves
            .iter()
            .flat_map(|(_, leaf)| leaf.pages.iter().flatten())
            .map(|page| match page.as_ref() {
                ValuePage::Owned(values) => values.capacity().saturating_mul(size_of::<T>()),
                ValuePage::Shared(values) => values.len().saturating_mul(size_of::<T>()),
            })
            .sum::<usize>()
            .saturating_add(usize::from(self.root.default.is_some()) * size_of::<T>())
    }

    /// Actual value-page bytes no longer shared with `previous` at the same logical positions.
    /// This inspects allocation identity and is used by scale tests rather than an estimated
    /// mutation counter.
    #[doc(hidden)]
    pub fn detached_page_bytes_from(&self, previous: &Self) -> usize {
        self.root
            .leaves
            .iter()
            .flat_map(|(key, leaf)| {
                let first_page = key.reverse_bits() as usize * PAGES_PER_LEAF;
                leaf.pages
                    .iter()
                    .enumerate()
                    .filter_map(move |(slot, page)| {
                        page.as_ref().map(|page| {
                            let shared = previous
                                .root
                                .page(first_page + slot)
                                .is_some_and(|old| Arc::ptr_eq(old, page));
                            if shared {
                                0
                            } else {
                                page.len().saturating_mul(size_of::<T>())
                            }
                        })
                    })
            })
            .sum()
    }

    #[cfg(test)]
    pub fn shared_value_bytes(&self) -> usize {
        self.root
            .leaves
            .iter()
            .flat_map(|(_, leaf)| leaf.pages.iter().flatten())
            .filter_map(|page| match page.as_ref() {
                ValuePage::Shared(values) => Some(values.len().saturating_mul(size_of::<T>())),
                ValuePage::Owned(_) => None,
            })
            .sum()
    }
}

impl<T> Index<usize> for PagedVec<T> {
    type Output = T;

    #[allow(clippy::expect_used)] // Index follows the standard slice out-of-bounds contract.
    fn index(&self, index: usize) -> &Self::Output {
        self.get(index).expect("paged vector index out of bounds")
    }
}

impl<T: Clone> IndexMut<usize> for PagedVec<T> {
    #[allow(clippy::expect_used)] // IndexMut follows the standard slice out-of-bounds contract.
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        self.get_mut(index)
            .expect("paged vector index out of bounds")
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
    root_depth: u32,
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
        Self {
            root: None,
            root_depth: 0,
            len: 0,
        }
    }
}

impl<V> Clone for PersistentMap<V> {
    fn clone(&self) -> Self {
        Self {
            root: self.root.clone(),
            root_depth: self.root_depth,
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
        if key.leading_zeros() / RADIX_BITS < self.root_depth {
            return None;
        }
        for depth in self.root_depth..=RADIX_LEVELS {
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

    /// Changed radix allocations, identified by prefix and nibble depth. Shared subtrees stop
    /// immediately; leaf splits include every newly born sibling, not only the inserted key.
    /// Counts cover node/branch/entry storage and exclude separately owned value payloads.
    #[must_use]
    pub fn changed_path_nodes(&self, previous: &Self) -> Vec<(u128, u8, usize, usize)> {
        fn bytes<V>(node: Option<&Arc<RadixNode<V>>>) -> usize {
            let Some(node) = node else {
                return 0;
            };
            let allocation = size_of::<RadixNode<V>>() + 2 * size_of::<usize>();
            allocation.saturating_add(match node.as_ref() {
                RadixNode::Branch(children) => size_of_val(children.as_ref()),
                RadixNode::Leaf(entries) => {
                    entries.capacity().saturating_mul(size_of::<(u128, V)>())
                }
            })
        }
        fn visit<V>(
            current: Option<&Arc<RadixNode<V>>>,
            previous: Option<&Arc<RadixNode<V>>>,
            current_depth: u32,
            previous_depth: u32,
            prefix: u128,
            depth: u32,
            changed: &mut Vec<(u128, u8, usize, usize)>,
        ) {
            if current.is_none() && previous.is_none()
                || current
                    .zip(previous)
                    .is_some_and(|(current, previous)| Arc::ptr_eq(current, previous))
            {
                return;
            }
            let old_bytes = if depth >= previous_depth {
                bytes(previous)
            } else {
                0
            };
            let new_bytes = if depth >= current_depth {
                bytes(current)
            } else {
                0
            };
            if old_bytes != 0 || new_bytes != 0 {
                changed.push((prefix, depth as u8, old_bytes, new_bytes));
            }
            if depth == RADIX_LEVELS {
                return;
            }
            let current_children = match current.map(AsRef::as_ref) {
                Some(RadixNode::Branch(children)) => Some(children),
                _ => None,
            };
            let previous_children = match previous.map(AsRef::as_ref) {
                Some(RadixNode::Branch(children)) => Some(children),
                _ => None,
            };
            if current_children.is_none()
                && previous_children.is_none()
                && depth >= current_depth
                && depth >= previous_depth
            {
                return;
            }
            let shift = 128 - RADIX_BITS * (depth + 1);
            for slot in 0..RADIX_FANOUT {
                visit(
                    if depth < current_depth {
                        (slot == 0).then_some(current).flatten()
                    } else {
                        current_children.and_then(|children| children[slot].as_ref())
                    },
                    if depth < previous_depth {
                        (slot == 0).then_some(previous).flatten()
                    } else {
                        previous_children.and_then(|children| children[slot].as_ref())
                    },
                    current_depth.max(depth + 1),
                    previous_depth.max(depth + 1),
                    prefix | ((slot as u128) << shift),
                    depth + 1,
                    changed,
                );
            }
        }
        let mut changed = Vec::new();
        visit(
            self.root.as_ref(),
            previous.root.as_ref(),
            self.root_depth,
            previous.root_depth,
            0,
            self.root_depth.min(previous.root_depth),
            &mut changed,
        );
        changed
    }

    pub fn insert(&mut self, key: u128, value: V) -> Option<V>
    where
        V: Clone,
    {
        self.insert_cow(key, value)
    }

    /// Inserts into an unpublished COW generation without allocating an immutable intermediate
    /// root for every row in a batch. The first touched path detaches from any pinned generation;
    /// later inserts mutate only branches already unique to this map.
    pub fn insert_cow(&mut self, key: u128, value: V) -> Option<V>
    where
        V: Clone,
    {
        // Omit leading zero levels without changing keys or their serialized order.
        // A larger key promotes only the root; already published subtrees remain shared.
        let key_depth = (key.leading_zeros() / RADIX_BITS).min(RADIX_LEVELS - 1);
        if let Some(root) = &mut self.root {
            if matches!(root.as_ref(), RadixNode::Leaf(_)) {
                self.root_depth = self.root_depth.min(key_depth);
            } else {
                while self.root_depth > key_depth {
                    let mut children = Box::new(std::array::from_fn(|_| None));
                    children[0] = Some(Arc::clone(root));
                    *root = Arc::new(RadixNode::Branch(children));
                    self.root_depth -= 1;
                }
            }
        } else {
            self.root_depth = key_depth;
        }
        let previous = match self.root.as_mut() {
            Some(root) => insert_radix_cow(root, self.root_depth, key, value),
            None => {
                self.root = Some(Arc::new(RadixNode::Leaf(vec![(key, value)])));
                None
            }
        };
        if previous.is_none() {
            self.len += 1;
        }
        previous
    }

    pub fn get_mut(&mut self, key: u128) -> Option<&mut V>
    where
        V: Clone,
    {
        if key.leading_zeros() / RADIX_BITS < self.root_depth {
            return None;
        }
        get_radix_mut(Arc::make_mut(self.root.as_mut()?), self.root_depth, key)
    }

    pub fn remove(&mut self, key: u128) -> Option<V>
    where
        V: Clone,
    {
        if key.leading_zeros() / RADIX_BITS < self.root_depth {
            return None;
        }
        let (root, removed) = remove_radix(self.root.as_ref(), self.root_depth, key);
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
        // Test oracle follows allocation identity even when root promotion moves a shared
        // subtree to a different depth. Production accounting remains a bounded path walk.
        fn collect<V>(node: &Arc<RadixNode<V>>, known: &mut std::collections::HashSet<usize>) {
            known.insert(Arc::as_ptr(node) as usize);
            if let RadixNode::Branch(children) = node.as_ref() {
                for child in children.iter().flatten() {
                    collect(child, known);
                }
            }
        }
        fn detached<V>(
            node: &Arc<RadixNode<V>>,
            known: &std::collections::HashSet<usize>,
        ) -> usize {
            if known.contains(&(Arc::as_ptr(node) as usize)) {
                return 0;
            }
            let allocation = size_of::<RadixNode<V>>() + 2 * size_of::<usize>();
            allocation.saturating_add(match node.as_ref() {
                RadixNode::Branch(children) => size_of_val(children.as_ref()).saturating_add(
                    children
                        .iter()
                        .flatten()
                        .map(|child| detached(child, known))
                        .fold(0_usize, usize::saturating_add),
                ),
                RadixNode::Leaf(entries) => {
                    entries.capacity().saturating_mul(size_of::<(u128, V)>())
                }
            })
        }
        let mut known = std::collections::HashSet::new();
        if let Some(root) = &previous.root {
            collect(root, &mut known);
        }
        self.root.as_ref().map_or(0, |root| detached(root, &known))
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

fn insert_radix_cow<V: Clone>(
    current: &mut Arc<RadixNode<V>>,
    depth: u32,
    key: u128,
    value: V,
) -> Option<V> {
    let node = Arc::make_mut(current);
    match node {
        RadixNode::Branch(children) if depth < RADIX_LEVELS => {
            let shift = 128 - RADIX_BITS * (depth + 1);
            let slot = ((key >> shift) & ((1 << RADIX_BITS) - 1)) as usize;
            match children[slot].as_mut() {
                Some(child) => insert_radix_cow(child, depth + 1, key, value),
                None => {
                    children[slot] = Some(Arc::new(RadixNode::Leaf(vec![(key, value)])));
                    None
                }
            }
        }
        RadixNode::Leaf(entries) => {
            let previous = match entries.binary_search_by_key(&key, |(stored, _)| *stored) {
                Ok(index) => Some(std::mem::replace(&mut entries[index].1, value)),
                Err(index) => {
                    entries.insert(index, (key, value));
                    None
                }
            };
            if entries.len() <= RADIX_LEAF_CAPACITY || depth == RADIX_LEVELS {
                return previous;
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
            previous
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
    fn surgical_radix_changes_include_split_siblings_and_skip_shared_subtrees() {
        let mut original = PersistentMap::default();
        for slot in 0..8_u128 {
            original.insert_cow(slot << 124, slot as u64);
        }
        let mut split = original.clone();
        split.insert_cow(8_u128 << 124, 8);
        let changes = split.changed_path_nodes(&original);
        assert_eq!(changes.len(), 10);
        assert!(changes[0].2 > 0 && changes[0].3 > 0);
        for slot in 0..9_u128 {
            assert!(
                changes
                    .iter()
                    .any(|(prefix, depth, old, new)| *prefix == slot << 124
                        && *depth == 1
                        && *old == 0
                        && *new > 0)
            );
        }
        assert_eq!(
            changes.iter().map(|entry| entry.3).sum::<usize>(),
            split.detached_node_bytes_from(&original)
        );
        assert!(split.changed_path_nodes(&split.clone()).is_empty());
        for rows in [4_096_u64, 32_768] {
            let mut before = PersistentMap::default();
            for id in 0..rows {
                before.insert_cow(stable_id_key(id.wrapping_mul(0x9e37_79b9_7f4a_7c15)), id);
            }
            let mut after = before.clone();
            let key = stable_id_key(17_u64.wrapping_mul(0x9e37_79b9_7f4a_7c15));
            after.insert_cow(key, u64::MAX);
            let changes = after.changed_path_nodes(&before);
            assert!(changes.len() <= 33);
            assert_eq!(
                changes.iter().map(|entry| entry.3).sum::<usize>(),
                after.detached_node_bytes_from(&before)
            );
            assert!(changes.iter().all(|entry| entry.2 != 0 && entry.3 != 0));
            assert_eq!(before.get(key), Some(&17));
            assert_eq!(after.get(key), Some(&u64::MAX));
        }
    }

    #[test]
    fn paged_iteration_preserves_sparse_defaults_and_both_ends() {
        let capacity = page_capacity::<u64>();
        let mut values = PagedVec::repeat(17_u64, capacity * 3 + 11);
        let mut expected = vec![17_u64; values.len()];
        for (row, value) in [
            (0, 31),
            (capacity - 1, 37),
            (capacity * 2 + 3, 41),
            (capacity * 3 + 10, 43),
        ] {
            values[row] = value;
            expected[row] = value;
        }
        assert_eq!(values.iter().copied().collect::<Vec<_>>(), expected);
        assert_eq!(
            values.iter().rev().copied().collect::<Vec<_>>(),
            expected.iter().rev().copied().collect::<Vec<_>>()
        );
        let mut actual = values.iter();
        let mut expected = expected.iter();
        while let Some(front) = expected.next() {
            assert_eq!(actual.next(), Some(front));
            assert_eq!(actual.next_back(), expected.next_back());
        }
        assert!(actual.next().is_none());
        assert!(actual.next_back().is_none());
    }

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
    fn point_update_keeps_untouched_leaf_directories_at_large_sizes() {
        for rows in [64_000, 2_000_000] {
            let values: Arc<[u64]> = vec![37_u64; rows].into();
            let flat = SharedFlat::from_arc_slice(values).expect("shared fixture");
            let original = PagedVec::from_shared(flat).expect("paged fixture");
            let mut updated = original.clone();
            updated[rows / 2] = 41;
            assert_eq!(original[rows / 2], 37);
            assert_eq!(updated.detached_page_bytes_from(&original), PAGE_BYTES);
            let changed = updated
                .root
                .leaves
                .iter()
                .filter(|(key, leaf)| {
                    !original
                        .root
                        .leaves
                        .get(*key)
                        .is_some_and(|old| Arc::ptr_eq(old, leaf))
                })
                .collect::<Vec<_>>();
            assert_eq!(changed.len(), 1);
            assert!(changed[0].1.pages.len() <= PAGES_PER_LEAF);
            assert!(
                updated
                    .root
                    .leaves
                    .detached_node_bytes_from(&original.root.leaves)
                    <= 16 * 1024
            );
        }
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
    fn append_to_mostly_full_shared_page_keeps_one_page_of_capacity() {
        let len = page_capacity::<u64>() / 2 + 17;
        let values = (0..len as u64).collect::<Vec<_>>();
        let shared = SharedFlat::from_arc_slice(values.clone().into()).expect("shared fixture");
        let mut current = PagedVec::from_shared(shared).expect("paged fixture");
        let pinned = current.clone();
        current.push(7919);
        assert_eq!(pinned.to_vec(), values);
        assert_eq!(current.get(len), Some(&7919));
        assert_eq!(current.allocated_value_bytes(), PAGE_BYTES);
        let retained = current.clone();
        current.push(7920);
        assert_eq!(retained.get(len + 1), None);
        assert_eq!(current.get(len + 1), Some(&7920));
        assert_eq!(current.allocated_value_bytes(), PAGE_BYTES);
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
    fn radix_batch_preserves_pinned_generation_and_matches_ordered_map() {
        let mut base = PersistentMap::default();
        let mut ordered = std::collections::BTreeMap::new();
        for key in 0_u128..10_000 {
            base.insert(key.saturating_mul(97), key as u32);
            ordered.insert(key.saturating_mul(97), key as u32);
        }
        let pinned = base.clone();
        let mut cow = base.clone();
        let mut persistent = base;
        for key in 10_000_u128..12_000 {
            let key = key.saturating_mul(97);
            assert_eq!(cow.insert_cow(key, key as u32), None);
            assert_eq!(persistent.insert(key, key as u32), None);
            ordered.insert(key, key as u32);
        }
        assert_eq!(cow.len(), persistent.len());
        assert!(cow.iter().eq(persistent.iter()));
        assert!(
            persistent
                .iter()
                .eq(ordered.iter().map(|(key, value)| (*key, value)))
        );
        assert_eq!(pinned.len(), 10_000);
        assert_eq!(pinned.get(11_000_u128 * 97), None);
    }

    #[test]
    fn radix_replacement_moves_unique_values_and_preserves_pinned_values() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        struct Value {
            body: Arc<str>,
            clones: Arc<AtomicUsize>,
        }
        impl Clone for Value {
            fn clone(&self) -> Self {
                self.clones.fetch_add(1, Ordering::Relaxed);
                Self {
                    body: self.body.clone(),
                    clones: self.clones.clone(),
                }
            }
        }
        let clones = Arc::new(AtomicUsize::new(0));
        let first: Arc<str> = "old ".repeat(65_536).into();
        let second: Arc<str> = "new ".repeat(65_536).into();
        let value = |body: Arc<str>| Value {
            body,
            clones: clones.clone(),
        };
        let mut map = PersistentMap::default();
        assert!(map.insert(7, value(first.clone())).is_none());
        let previous = map
            .insert(7, value(second.clone()))
            .expect("previous value");
        assert!(Arc::ptr_eq(&previous.body, &first));
        assert_eq!(
            clones.load(Ordering::Relaxed),
            0,
            "unique replacements must move the old value"
        );
        let pinned = map.clone();
        let previous = map.insert(7, value(first.clone())).expect("previous value");
        assert!(Arc::ptr_eq(&previous.body, &second));
        assert!(Arc::ptr_eq(
            &pinned.get(7).expect("pinned value").body,
            &second
        ));
        assert!(Arc::ptr_eq(
            &map.get(7).expect("current value").body,
            &first
        ));
        assert_eq!(
            clones.load(Ordering::Relaxed),
            1,
            "detach the retained leaf once"
        );
    }

    #[test]
    fn ordinary_radix_inserts_reuse_unique_paths_and_keep_large_pinned_values() {
        let payload: Arc<str> = Arc::from("x".repeat(256 * 1024));
        let mut current = PersistentMap::default();
        for id in 0..65_536_u64 {
            current.insert(stable_id_key(id), Arc::clone(&payload));
        }
        let pinned = current.clone();
        let changed: Arc<str> = Arc::from("changed");
        current.insert(stable_id_key(17), Arc::clone(&changed));
        // The first larger ID promotes the root once; subsequent batch inserts reuse it.
        assert_eq!(
            current.insert(stable_id_key(65_536), Arc::clone(&changed)),
            None
        );
        let root = Arc::as_ptr(current.root.as_ref().expect("radix root"));
        for id in 65_537..66_560_u64 {
            assert_eq!(
                current.insert(stable_id_key(id), Arc::clone(&changed)),
                None
            );
            assert_eq!(
                Arc::as_ptr(current.root.as_ref().expect("radix root")),
                root
            );
        }
        assert!(Arc::ptr_eq(
            pinned.get(stable_id_key(17)).expect("pinned row"),
            &payload
        ));
        assert_eq!(pinned.len(), 65_536);
        assert_eq!(pinned.get(stable_id_key(65_536)), None);
        assert_eq!(current.len(), 66_560);
        assert!(current.detached_node_bytes_from(&pinned) < 1024 * 1024);
    }

    #[test]
    fn leading_zero_levels_are_skipped_without_changing_keys_or_pinned_generations() {
        let mut original = PersistentMap::default();
        for id in 0..4096_u64 {
            original.insert_cow(stable_id_key(id), id);
        }
        assert_eq!(original.root_depth, 13);
        let mut updated = original.clone();
        updated.insert_cow(stable_id_key(4096), 4096);
        assert_eq!(updated.root_depth, 12);
        assert_eq!(original.get(stable_id_key(4096)), None);
        assert_eq!(updated.get(stable_id_key(4096)), Some(&4096));
        let changes = updated.changed_path_nodes(&original);
        assert_eq!(
            changes.len(),
            2,
            "promotion shares the complete older subtree"
        );
        assert!(changes.iter().all(|entry| entry.2 == 0 && entry.3 > 0));
        assert_eq!(
            changes.iter().map(|entry| entry.3).sum::<usize>(),
            updated.detached_node_bytes_from(&original)
        );
        assert_eq!(updated.get(1_u128 << 127), None);
        assert_eq!(updated.get_mut(1_u128 << 127), None);
        assert_eq!(updated.remove(1_u128 << 127), None);
        *updated.get_mut(stable_id_key(17)).expect("stored key") = 99;
        assert_eq!(updated.remove(stable_id_key(18)), Some(18));
        assert_eq!(original.get(stable_id_key(17)), Some(&17));
        assert_eq!(original.get(stable_id_key(18)), Some(&18));
        let mut bytes = Vec::new();
        ciborium::into_writer(&updated, &mut bytes).expect("serialize map");
        let decoded: PersistentMap<u64> =
            ciborium::from_reader(bytes.as_slice()).expect("decode map");
        assert_eq!(
            updated.iter().collect::<Vec<_>>(),
            decoded.iter().collect::<Vec<_>>()
        );
        let mut encoded_again = Vec::new();
        ciborium::into_writer(&decoded, &mut encoded_again).expect("serialize decoded map");
        assert_eq!(bytes, encoded_again);
    }

    #[test]
    fn root_promotions_account_for_new_allocations_only_across_key_shapes() {
        let mut map = PersistentMap::default();
        let mut oracle = std::collections::BTreeMap::new();
        for key in 0..256_u128 {
            map.insert_cow(key, key);
            oracle.insert(key, key);
        }
        for key in [4096, 1_u128 << 64, 1_u128 << 124, u128::MAX, 17, 0] {
            let previous = map.clone();
            map.insert_cow(key, key + u128::from(key != u128::MAX));
            oracle.insert(key, key + u128::from(key != u128::MAX));
            assert_eq!(
                map.iter()
                    .map(|(key, value)| (key, *value))
                    .collect::<Vec<_>>(),
                oracle
                    .iter()
                    .map(|(key, value)| (*key, *value))
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                map.changed_path_nodes(&previous)
                    .iter()
                    .map(|entry| entry.3)
                    .sum::<usize>(),
                map.detached_node_bytes_from(&previous)
            );
        }
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
