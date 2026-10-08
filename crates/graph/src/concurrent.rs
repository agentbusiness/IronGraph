//! One canonical CPU graph, with stable atomic columns and exact borrowed-value reclamation.
//! Writers are ordered by the database sequencer. Readers never acquire the writer permit.

use std::{
    collections::{BTreeMap, BTreeSet},
    fmt,
    ops::Deref,
    sync::{
        Arc, Mutex, OnceLock,
        atomic::{AtomicBool, AtomicU8, AtomicU32, AtomicU64, AtomicUsize, Ordering},
    },
};

use arc_swap::ArcSwapOption;
use papaya::HashMap;
use serde::{
    Deserialize, Serialize,
    de::{DeserializeSeed, MapAccess, SeqAccess, Visitor},
    ser::{SerializeSeq, SerializeStruct},
};

use crate::{
    CompactionMap, EdgeId, EdgeInput, Error, ErrorCode, GraphMutation, LabelId, Layer, LayerMask,
    NodeId, NodeInput, PersistentMap, PropertyId, RelationshipTypeId, Result, ScalarValue,
};

const SEGMENT: usize = 256;
const NONE: u32 = u32::MAX;
const NO_EDGE: u64 = u64::MAX;
const LAYERS: usize = 3;

/// Fixed-radix directory: readers use `OnceLock::get`, never `get_or_init`.
/// Directory metadata survives until project deletion; payload cells release removed values.
struct Directory<T> {
    children: OnceLock<Box<[OnceLock<Directory<T>>; SEGMENT]>>,
    page: OnceLock<Box<[OnceLock<T>; SEGMENT]>>,
}

#[derive(Default)]
struct AllocationAccount {
    bytes: AtomicUsize,
    owner: Option<Arc<AtomicUsize>>,
}
impl AllocationAccount {
    fn add(&self, bytes: usize) {
        self.bytes.fetch_add(bytes, Ordering::Relaxed);
        if let Some(owner) = &self.owner {
            owner.fetch_add(bytes, Ordering::Relaxed);
        }
    }
}
impl Drop for AllocationAccount {
    fn drop(&mut self) {
        if let Some(owner) = &self.owner {
            owner.fetch_sub(*self.bytes.get_mut(), Ordering::Relaxed);
        }
    }
}

impl<T> Default for Directory<T> {
    fn default() -> Self {
        Self {
            children: OnceLock::new(),
            page: OnceLock::new(),
        }
    }
}

impl<T> Directory<T> {
    #[inline(always)]
    fn borrow_page(&self, page: usize) -> Option<&[OnceLock<T>; SEGMENT]> {
        let mut branch = self;
        for shift in [16, 8, 0] {
            branch = branch.children.get()?[(page >> shift) & 255].get()?;
        }
        branch.page.get().map(Box::as_ref)
    }

    fn page(&self, page: usize, account: &AllocationAccount) -> &[OnceLock<T>; SEGMENT] {
        let mut branch = self;
        for shift in [16, 8, 0] {
            let children = branch.children.get_or_init(|| {
                let values = Box::new(std::array::from_fn(|_| OnceLock::new()));
                account.add(size_of::<[OnceLock<Directory<T>>; SEGMENT]>());
                values
            });
            let slot = &children[(page >> shift) & 255];
            branch = slot.get_or_init(Self::default);
        }
        branch.page.get_or_init(|| {
            let values = Box::new(std::array::from_fn(|_| OnceLock::new()));
            account.add(size_of::<[OnceLock<T>; SEGMENT]>());
            values
        })
    }
}

#[derive(Default)]
pub(crate) struct Segments<T> {
    directory: Directory<T>,
    len: AtomicUsize,
    allocations: AllocationAccount,
}

impl<T> Segments<T> {
    pub(crate) fn len(&self) -> usize {
        self.len.load(Ordering::Acquire)
    }
    fn allocated_bytes(&self) -> usize {
        size_of::<Self>() + self.allocations.bytes.load(Ordering::Relaxed)
    }
    fn with_accounting(owner: Arc<AtomicUsize>) -> Self {
        Self {
            directory: Directory::default(),
            len: AtomicUsize::new(0),
            allocations: AllocationAccount {
                bytes: AtomicUsize::new(0),
                owner: Some(owner),
            },
        }
    }
    fn get_or_insert_with(&self, row: usize, make: impl FnOnce() -> T) -> Result<&T> {
        if row >= NONE as usize {
            return Err(budget("canonical row space exhausted"));
        }
        let page = self.directory.page(row / SEGMENT, &self.allocations);
        let value = page[row % SEGMENT].get_or_init(make);
        self.len.fetch_max(row + 1, Ordering::Release);
        Ok(value)
    }
    #[inline(always)]
    pub(crate) fn get(&self, row: usize) -> Option<&T> {
        if row >= self.len() {
            return None;
        }
        self.directory.borrow_page(row / SEGMENT)?[row % SEGMENT].get()
    }
    pub(crate) fn push(&self, value: T) -> Result<usize> {
        let row = self.len();
        if row >= NONE as usize {
            return Err(budget("canonical row space exhausted"));
        }
        let page = self.directory.page(row / SEGMENT, &self.allocations);
        page[row % SEGMENT]
            .set(value)
            .map_err(|_| Error::internal("unordered canonical writers"))?;
        self.len.store(row + 1, Ordering::Release);
        Ok(row)
    }
}

fn budget(message: &'static str) -> Error {
    Error::new(ErrorCode::ResultBudgetExceeded, message)
}
fn missing(message: &'static str) -> Error {
    Error::new(ErrorCode::QueryType, message)
}
#[inline(always)]
fn load(column: &Segments<AtomicU64>, row: u32) -> u64 {
    column
        .get(row as usize)
        .map_or(0, |cell| cell.load(Ordering::Acquire))
}
fn layer(value: u64) -> Layer {
    match value {
        1 => Layer::Knowledge,
        2 => Layer::Workspace,
        _ => Layer::Observed,
    }
}

#[derive(Default)]
struct Names {
    by_name: HashMap<Arc<str>, u64>,
    by_id: HashMap<u64, Arc<str>>,
    next: AtomicU64,
    digests: HashMap<u128, [u8; 32]>,
}

impl Names {
    fn id(&self, name: &str) -> Option<u64> {
        self.by_name.pin().get(name).copied()
    }
    fn name(&self, id: u64) -> Option<Arc<str>> {
        self.by_id.pin().get(&id).cloned()
    }
    fn declare(&self, name: String, id: u64) -> Result<()> {
        let next = id
            .checked_add(1)
            .ok_or_else(|| budget("schema ID space exhausted"))?;
        if self.id(&name).is_some_and(|old| old != id)
            || self.name(id).is_some_and(|old| old.as_ref() != name)
        {
            return Err(Error::new(
                ErrorCode::TransactionConflict,
                "schema name or ID already assigned",
            ));
        }
        if self.name(id).is_none() {
            let name: Arc<str> = Arc::from(name);
            self.by_id.pin().insert(id, Arc::clone(&name));
            self.by_name.pin().insert(Arc::clone(&name), id);
            self.update_digest(id, &name);
        }
        self.next.fetch_max(next, Ordering::AcqRel);
        Ok(())
    }
    fn update_digest(&self, id: u64, name: &str) {
        let pin = self.digests.pin();
        let key = |depth: u8, prefix: u64| (u128::from(depth) << 64) | u128::from(prefix);
        let mut leaf = blake3::Hasher::new();
        leaf.update(b"schema-name-leaf-v2\0");
        leaf.update(&id.to_le_bytes());
        leaf.update(&(name.len() as u64).to_le_bytes());
        leaf.update(name.as_bytes());
        pin.insert(key(16, id), *leaf.finalize().as_bytes());
        for depth in (0_u8..16).rev() {
            let prefix = if depth == 0 {
                0
            } else {
                id >> (64 - u32::from(depth) * 4)
            };
            let mut branch = blake3::Hasher::new();
            branch.update(b"schema-name-branch-v2\0");
            branch.update(&[depth]);
            for child in 0..16 {
                branch.update(
                    pin.get(&key(depth + 1, (prefix << 4) | child))
                        .unwrap_or(&[0; 32]),
                );
            }
            pin.insert(key(depth, prefix), *branch.finalize().as_bytes());
        }
    }
    fn digest(&self) -> [u8; 32] {
        self.digests.pin().get(&0).copied().unwrap_or([0; 32])
    }
    fn intern(&self, name: &str) -> Result<u64> {
        if let Some(id) = self.id(name) {
            return Ok(id);
        }
        let id = self.next.load(Ordering::Acquire);
        self.declare(name.to_owned(), id)?;
        Ok(id)
    }
    fn entries(&self) -> Vec<(u64, Arc<str>)> {
        let mut entries = self
            .by_id
            .pin()
            .iter()
            .map(|(id, name)| (*id, Arc::clone(name)))
            .collect::<Vec<_>>();
        entries.sort_by_key(|(id, _)| *id);
        entries
    }
}

#[derive(Default)]
struct CatalogInner {
    labels: Names,
    properties: Names,
    relationships: Names,
    generation: ArcSwapOption<[u8; 32]>,
}

/// Clones are handles to the same mutable schema catalog, never historical generations.
#[derive(Clone, Default)]
pub struct NameCatalog(Arc<CatalogInner>);

impl fmt::Debug for NameCatalog {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("NameCatalog")
            .field("labels", &self.0.labels.next.load(Ordering::Acquire))
            .field(
                "properties",
                &self.0.properties.next.load(Ordering::Acquire),
            )
            .finish()
    }
}

impl NameCatalog {
    pub fn next_label_id(&self) -> u64 {
        self.0.labels.next.load(Ordering::Acquire)
    }
    pub fn next_property_id(&self) -> u64 {
        self.0.properties.next.load(Ordering::Acquire)
    }
    pub fn next_relationship_type_id(&self) -> u64 {
        self.0.relationships.next.load(Ordering::Acquire)
    }
    pub fn declare_label(&self, name: String, id: LabelId) -> Result<()> {
        self.0.labels.declare(name, id.0)?;
        self.refresh_generation();
        Ok(())
    }
    pub fn declare_property(&self, name: String, id: PropertyId) -> Result<()> {
        self.0.properties.declare(name, id.0)?;
        self.refresh_generation();
        Ok(())
    }
    pub fn declare_relationship_type(&self, name: String, id: RelationshipTypeId) -> Result<()> {
        self.0.relationships.declare(name, id.0)?;
        self.refresh_generation();
        Ok(())
    }
    pub fn intern_label(&self, name: &str) -> Result<LabelId> {
        let id = self.0.labels.intern(name)?;
        self.refresh_generation();
        Ok(LabelId(id))
    }
    pub fn intern_property(&self, name: &str) -> Result<PropertyId> {
        let id = self.0.properties.intern(name)?;
        self.refresh_generation();
        Ok(PropertyId(id))
    }
    pub fn intern_relationship_type(&self, name: &str) -> Result<RelationshipTypeId> {
        let id = self.0.relationships.intern(name)?;
        self.refresh_generation();
        Ok(RelationshipTypeId(id))
    }
    pub fn label(&self, name: &str) -> Option<LabelId> {
        self.0.labels.id(name).map(LabelId)
    }
    pub fn property(&self, name: &str) -> Option<PropertyId> {
        self.0.properties.id(name).map(PropertyId)
    }
    pub fn relationship_type(&self, name: &str) -> Option<RelationshipTypeId> {
        self.0.relationships.id(name).map(RelationshipTypeId)
    }
    pub fn label_name(&self, id: LabelId) -> Option<Arc<str>> {
        self.0.labels.name(id.0)
    }
    pub fn property_name(&self, id: PropertyId) -> Option<Arc<str>> {
        self.0.properties.name(id.0)
    }
    pub fn relationship_type_name(&self, id: RelationshipTypeId) -> Option<Arc<str>> {
        self.0.relationships.name(id.0)
    }
    pub fn labels(&self) -> impl Iterator<Item = (LabelId, Arc<str>)> {
        self.0
            .labels
            .entries()
            .into_iter()
            .map(|(id, name)| (LabelId(id), name))
    }
    pub fn properties(&self) -> impl Iterator<Item = (PropertyId, Arc<str>)> {
        self.0
            .properties
            .entries()
            .into_iter()
            .map(|(id, name)| (PropertyId(id), name))
    }
    pub fn relationship_types(&self) -> impl Iterator<Item = (RelationshipTypeId, Arc<str>)> {
        self.0
            .relationships
            .entries()
            .into_iter()
            .map(|(id, name)| (RelationshipTypeId(id), name))
    }
    pub fn optimizer_generation(&self) -> [u8; 32] {
        if let Some(generation) = self.0.generation.load().as_ref() {
            return **generation;
        }
        self.compute_generation()
    }
    fn refresh_generation(&self) {
        let generation = self.compute_generation();
        if self
            .0
            .generation
            .load()
            .as_ref()
            .is_none_or(|current| **current != generation)
        {
            self.0.generation.store(Some(Arc::new(generation)));
        }
    }
    fn compute_generation(&self) -> [u8; 32] {
        let mut hash = blake3::Hasher::new();
        hash.update(b"canonical-cpu-schema-v1");
        for (kind, names) in [
            (0_u8, &self.0.labels),
            (1, &self.0.properties),
            (2, &self.0.relationships),
        ] {
            hash.update(&[kind]);
            hash.update(&names.digest());
        }
        *hash.finalize().as_bytes()
    }
    fn import(&self, other: &Self) -> Result<()> {
        for (id, name) in other.labels() {
            self.declare_label(name.to_string(), id)?;
        }
        for (id, name) in other.properties() {
            self.declare_property(name.to_string(), id)?;
        }
        for (id, name) in other.relationship_types() {
            self.declare_relationship_type(name.to_string(), id)?;
        }
        self.0
            .labels
            .next
            .fetch_max(other.next_label_id(), Ordering::AcqRel);
        self.0
            .properties
            .next
            .fetch_max(other.next_property_id(), Ordering::AcqRel);
        self.0
            .relationships
            .next
            .fetch_max(other.next_relationship_type_id(), Ordering::AcqRel);
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
struct CatalogWire {
    labels: Vec<(LabelId, Arc<str>)>,
    properties: Vec<(PropertyId, Arc<str>)>,
    relationships: Vec<(RelationshipTypeId, Arc<str>)>,
    next: [u64; 3],
}
impl Serialize for NameCatalog {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        CatalogWire {
            labels: self.labels().collect(),
            properties: self.properties().collect(),
            relationships: self.relationship_types().collect(),
            next: [
                self.next_label_id(),
                self.next_property_id(),
                self.next_relationship_type_id(),
            ],
        }
        .serialize(serializer)
    }
}
impl<'de> Deserialize<'de> for NameCatalog {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        let wire = CatalogWire::deserialize(d)?;
        let catalog = Self::default();
        for (id, name) in wire.labels {
            catalog
                .declare_label(name.to_string(), id)
                .map_err(serde::de::Error::custom)?;
        }
        for (id, name) in wire.properties {
            catalog
                .declare_property(name.to_string(), id)
                .map_err(serde::de::Error::custom)?;
        }
        for (id, name) in wire.relationships {
            catalog
                .declare_relationship_type(name.to_string(), id)
                .map_err(serde::de::Error::custom)?;
        }
        if wire.next[0] < catalog.next_label_id()
            || wire.next[1] < catalog.next_property_id()
            || wire.next[2] < catalog.next_relationship_type_id()
        {
            return Err(serde::de::Error::custom(
                "schema next ID precedes an assigned ID",
            ));
        }
        catalog.0.labels.next.store(wire.next[0], Ordering::Release);
        catalog
            .0
            .properties
            .next
            .store(wire.next[1], Ordering::Release);
        catalog
            .0
            .relationships
            .next
            .store(wire.next[2], Ordering::Release);
        Ok(catalog)
    }
}

/// Small label sets are carried inline; larger allocations survive while a reader owns the handle.
#[derive(Clone, Debug, Default)]
pub struct LabelsHandle(LabelsValue);
#[derive(Clone, Debug, Default)]
enum LabelsValue {
    #[default]
    Empty,
    Single([LabelId; 1]),
    Multiple(Arc<LabelsAllocation>),
}
impl Deref for LabelsHandle {
    type Target = [LabelId];
    fn deref(&self) -> &[LabelId] {
        match &self.0 {
            LabelsValue::Empty => &[],
            LabelsValue::Single(value) => value,
            LabelsValue::Multiple(value) => &value.values,
        }
    }
}

#[derive(Default)]
struct LabelsCell {
    // Zero means empty; id+1 means one label; MAX selects the out-of-line set.
    // The two forms never hold duplicate active labels.
    inline: AtomicU64,
    multiple: ArcSwapOption<LabelsAllocation>,
}
impl LabelsCell {
    fn new(values: Vec<LabelId>, accounting: &Arc<AtomicUsize>) -> Self {
        let cell = Self::default();
        cell.store(values, accounting);
        cell
    }

    fn store(&self, values: Vec<LabelId>, accounting: &Arc<AtomicUsize>) {
        let inline = match values.as_slice() {
            [] => 0,
            [id] if id.0 < u64::MAX - 1 => id.0 + 1,
            _ => {
                self.multiple
                    .store(Some(LabelsAllocation::new(values, accounting)));
                self.inline.store(u64::MAX, Ordering::Release);
                return;
            }
        };
        self.inline.store(inline, Ordering::Release);
        self.multiple.store(None);
    }

    #[inline]
    fn contains_all(&self, labels: &[LabelId]) -> bool {
        match self.inline.load(Ordering::Acquire) {
            u64::MAX => {
                let allocation = self.multiple.load();
                allocation
                    .as_deref()
                    .is_some_and(|value| labels.iter().all(|label| value.values.contains(label)))
            }
            inline => labels
                .iter()
                .all(|label| inline != 0 && label.0 == inline - 1),
        }
    }

    fn handle(&self) -> LabelsHandle {
        LabelsHandle(match self.inline.load(Ordering::Acquire) {
            0 => LabelsValue::Empty,
            u64::MAX => self
                .multiple
                .load_full()
                .map_or(LabelsValue::Empty, LabelsValue::Multiple),
            inline => LabelsValue::Single([LabelId(inline - 1)]),
        })
    }
}

#[derive(Debug)]
struct LabelsAllocation {
    values: Vec<LabelId>,
    accounting: Arc<AtomicUsize>,
    bytes: usize,
}
impl LabelsAllocation {
    fn new(values: Vec<LabelId>, accounting: &Arc<AtomicUsize>) -> Arc<Self> {
        let bytes = size_of::<Self>() + values.capacity() * size_of::<LabelId>();
        accounting.fetch_add(bytes, Ordering::AcqRel);
        Arc::new(Self {
            values,
            accounting: Arc::clone(accounting),
            bytes,
        })
    }
}
impl Drop for LabelsAllocation {
    fn drop(&mut self) {
        self.accounting.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

struct ValueAllocation {
    value: ScalarValue,
    accounting: Arc<AtomicUsize>,
    bytes: usize,
}
impl ValueAllocation {
    fn new(value: ScalarValue, accounting: &Arc<AtomicUsize>) -> Arc<Self> {
        let bytes = size_of::<Self>() + value_bytes(&value) - size_of::<ScalarValue>();
        accounting.fetch_add(bytes, Ordering::AcqRel);
        Arc::new(Self {
            value,
            accounting: Arc::clone(accounting),
            bytes,
        })
    }
}
impl Drop for ValueAllocation {
    fn drop(&mut self) {
        self.accounting.fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

const VARIABLE_CELL: u8 = u8::MAX;

#[derive(Default)]
struct PropertyCell {
    mode: AtomicU8,
    bits: AtomicU64,
    payload: ArcSwapOption<ValueAllocation>,
}
impl PropertyCell {
    fn guarded_value(&self) -> Option<ScalarValue> {
        self.payload
            .load()
            .as_ref()
            .map(|value| value.value.clone())
    }
    fn get(&self) -> Option<ScalarValue> {
        let mode = self.mode.load(Ordering::Acquire);
        if mode == VARIABLE_CELL {
            return self.guarded_value();
        }
        if mode == 0 {
            return None;
        }
        let bits = self.bits.load(Ordering::Acquire);
        if self.mode.load(Ordering::Acquire) != mode {
            return self.guarded_value();
        }
        match mode {
            1 => Some(ScalarValue::Boolean(bits != 0)),
            2 => Some(ScalarValue::Integer(i64::from_ne_bytes(bits.to_ne_bytes()))),
            3 => Some(ScalarValue::Float(f64::from_bits(bits).into())),
            _ => None,
        }
    }
    #[inline(always)]
    fn get_integer(&self) -> Option<i64> {
        let mode = self.mode.load(Ordering::Acquire);
        if mode == 2 {
            let bits = self.bits.load(Ordering::Acquire);
            if self.mode.load(Ordering::Acquire) == mode {
                return Some(i64::from_ne_bytes(bits.to_ne_bytes()));
            }
        } else if mode != VARIABLE_CELL {
            return None;
        }
        match self.payload.load().as_deref().map(|value| &value.value) {
            Some(ScalarValue::Integer(value)) => Some(*value),
            _ => None,
        }
    }
    fn kind(&self) -> usize {
        let mode = self.mode.load(Ordering::Acquire);
        if mode == VARIABLE_CELL {
            self.payload
                .load()
                .as_deref()
                .map_or(0, |value| value_kind(&value.value))
        } else {
            usize::from(mode)
        }
    }
    /// All publications hold the owner writer permit. A primitive kind is stable until promotion.
    /// Acquire of cleared bits synchronizes the earlier Variable-mode publication, so a reader
    /// detects promotion and performs one guarded read of the actual new value, without a retry.
    fn set(&self, value: ScalarValue, accounting: &Arc<AtomicUsize>) {
        let mode = self.mode.load(Ordering::Acquire);
        let kind = value_kind(&value);
        if (1..=3).contains(&kind) && (mode == 0 || usize::from(mode) == kind) {
            let bits = match value {
                ScalarValue::Boolean(value) => u64::from(value),
                ScalarValue::Integer(value) => u64::from_ne_bytes(value.to_ne_bytes()),
                ScalarValue::Float(value) => value.0.to_bits(),
                _ => return,
            };
            self.bits.store(bits, Ordering::Release);
            if mode == 0 {
                self.mode.store(kind as u8, Ordering::Release);
            }
        } else {
            self.payload
                .store((kind != 0).then(|| ValueAllocation::new(value, accounting)));
            if mode != VARIABLE_CELL {
                self.mode.store(VARIABLE_CELL, Ordering::Release);
                self.bits.store(0, Ordering::Release);
            }
        }
    }
    /// Only full entity deletion resets mode; active is false before this owner-claimed operation.
    /// Public reads carry the owner's unique reuse stamp and reject the old identity afterward.
    fn reset(&self) {
        self.payload.store(None);
        self.mode.store(0, Ordering::Release);
        self.bits.store(0, Ordering::Release);
    }
}

pub struct ConcurrentPropertyColumn {
    cells: Segments<PropertyCell>,
    kinds: [AtomicU64; 14],
    accounting: Arc<AtomicUsize>,
}
impl Default for ConcurrentPropertyColumn {
    fn default() -> Self {
        Self::new(Arc::new(AtomicUsize::new(0)))
    }
}
impl Drop for ConcurrentPropertyColumn {
    fn drop(&mut self) {
        self.accounting
            .fetch_sub(size_of::<Self>(), Ordering::Relaxed);
    }
}

impl ConcurrentPropertyColumn {
    fn new(accounting: Arc<AtomicUsize>) -> Self {
        accounting.fetch_add(size_of::<Self>(), Ordering::Relaxed);
        Self {
            cells: Segments::with_accounting(Arc::clone(&accounting)),
            kinds: std::array::from_fn(|_| AtomicU64::new(0)),
            accounting,
        }
    }
    pub fn kind_mask(&self) -> u16 {
        self.kinds
            .iter()
            .enumerate()
            .fold(0_u16, |mask, (kind, count)| {
                if count.load(Ordering::Acquire) == 0 {
                    mask
                } else {
                    mask | (1_u16 << kind)
                }
            })
    }
    fn get(&self, row: u32) -> Option<ScalarValue> {
        self.cells.get(row as usize)?.get()
    }
    #[inline(always)]
    fn get_integer(&self, row: u32) -> Option<i64> {
        self.cells.get(row as usize)?.get_integer()
    }
    pub fn is_present(&self, row: u32) -> bool {
        self.cells
            .get(row as usize)
            .is_some_and(|cell| cell.kind() != 0)
    }
    pub fn value_count(&self) -> u64 {
        self.kinds
            .iter()
            .map(|count| count.load(Ordering::Acquire))
            .sum()
    }
    fn is_kind(&self, kind: usize) -> bool {
        let count = self.kinds[kind].load(Ordering::Acquire);
        count > 0
            && self
                .kinds
                .iter()
                .enumerate()
                .all(|(index, value)| index == kind || value.load(Ordering::Acquire) == 0)
    }
    fn remove(&self, row: u32) {
        if let Some(cell) = self.cells.get(row as usize) {
            let kind = cell.kind();
            cell.set(ScalarValue::Null, &self.accounting);
            if kind != 0 {
                self.kinds[kind].fetch_sub(1, Ordering::AcqRel);
            }
        }
    }
    fn reset(&self, row: u32) {
        if let Some(cell) = self.cells.get(row as usize) {
            let kind = cell.kind();
            cell.reset();
            if kind != 0 {
                self.kinds[kind].fetch_sub(1, Ordering::AcqRel);
            }
        }
    }
}

fn value_kind(value: &ScalarValue) -> usize {
    match value {
        ScalarValue::Null => 0,
        ScalarValue::Boolean(_) => 1,
        ScalarValue::Integer(_) => 2,
        ScalarValue::Float(_) => 3,
        ScalarValue::String(_) => 4,
        ScalarValue::Bytes(_) => 5,
        ScalarValue::Date(_) => 6,
        ScalarValue::LocalTime(_) => 7,
        ScalarValue::ZonedTime { .. } => 8,
        ScalarValue::LocalDateTime { .. } => 9,
        ScalarValue::ZonedDateTime { .. } => 10,
        ScalarValue::Duration { .. } => 11,
        ScalarValue::List(_) => 12,
        ScalarValue::Map(_) => 13,
    }
}
#[derive(Default)]
struct Properties {
    columns: HashMap<PropertyId, Arc<ConcurrentPropertyColumn>>,
    bytes: Arc<AtomicUsize>,
}

impl Properties {
    fn column(&self, property: PropertyId, create: bool) -> Option<Arc<ConcurrentPropertyColumn>> {
        let pin = self.columns.pin();
        if create {
            Some(Arc::clone(pin.get_or_insert_with(property, || {
                Arc::new(ConcurrentPropertyColumn::new(Arc::clone(&self.bytes)))
            })))
        } else {
            pin.get(&property).cloned()
        }
    }
    fn get(&self, row: u32, property: PropertyId) -> Option<ScalarValue> {
        let column = self.column(property, false)?;
        column.get(row)
    }
    fn set(&self, row: u32, property: PropertyId, value: ScalarValue) -> Result<()> {
        let Some(column) = self.column(property, !matches!(value, ScalarValue::Null)) else {
            return Ok(());
        };
        if matches!(value, ScalarValue::Null) {
            column.remove(row);
            return Ok(());
        }
        let cell = column
            .cells
            .get_or_insert_with(row as usize, PropertyCell::default)?;
        let kind = value_kind(&value);
        let previous = cell.kind();
        cell.set(value, &self.bytes);
        if previous != kind {
            if previous != 0 {
                column.kinds[previous].fetch_sub(1, Ordering::AcqRel);
            }
            column.kinds[kind].fetch_add(1, Ordering::AcqRel);
        }
        Ok(())
    }
    fn values(&self, row: u32) -> Vec<(PropertyId, ScalarValue)> {
        let mut values = self
            .columns
            .pin()
            .iter()
            .filter_map(|(property, _)| self.get(row, *property).map(|value| (*property, value)))
            .collect::<Vec<_>>();
        values.sort_by_key(|(property, _)| *property);
        values
    }
    fn clear(&self, row: u32) {
        for (_, column) in self.columns.pin().iter() {
            column.reset(row);
        }
    }
    fn accepts(&self, property: PropertyId, value: &ScalarValue) -> Option<bool> {
        self.column(property, false)
            .map(|_| !matches!(value, ScalarValue::Null))
    }
    fn has_type(&self, property: PropertyId, kind: usize) -> bool {
        self.column(property, false)
            .is_some_and(|column| column.is_kind(kind))
    }
}

#[derive(Default)]
struct Counts {
    layers: [AtomicU64; LAYERS],
    kinds: HashMap<u64, Arc<[AtomicU64; LAYERS]>>,
}
/// Shared direct access to canonical counters; cloning does not freeze their values.
#[derive(Clone, Default)]
pub struct CanonicalCounts(Arc<Counts>);
impl fmt::Debug for CanonicalCounts {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_map().entries(self.entries()).finish()
    }
}
impl CanonicalCounts {
    pub fn len(&self) -> usize {
        self.0.kinds.len()
    }
    pub fn is_empty(&self) -> bool {
        self.0.kinds.is_empty()
    }
    pub fn get(&self, key: u128) -> Option<[u64; LAYERS]> {
        let key = u64::try_from(key).ok()?;
        self.0
            .kinds
            .pin()
            .get(&key)
            .map(|counts| std::array::from_fn(|i| counts[i].load(Ordering::Acquire)))
    }
    pub fn shared_with(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
    }
    pub fn entries(&self) -> Vec<(u128, [u64; LAYERS])> {
        let mut entries = self
            .0
            .kinds
            .pin()
            .iter()
            .map(|(kind, counts)| {
                (
                    u128::from(*kind),
                    std::array::from_fn(|i| counts[i].load(Ordering::Acquire)),
                )
            })
            .collect::<Vec<_>>();
        entries.sort_by_key(|(kind, _)| *kind);
        entries
    }
}
impl Counts {
    fn adjust(&self, layer: Layer, kind: Option<u64>, add: bool) {
        let cell = &self.layers[layer as usize];
        if kind.is_none() {
            if add {
                cell.fetch_add(1, Ordering::AcqRel);
            } else {
                cell.fetch_sub(1, Ordering::AcqRel);
            }
        }
        if let Some(kind) = kind {
            let pin = self.kinds.pin();
            let cells = pin.get_or_insert_with(kind, || {
                Arc::new(std::array::from_fn(|_| AtomicU64::new(0)))
            });
            if add {
                cells[layer as usize].fetch_add(1, Ordering::AcqRel);
            } else {
                cells[layer as usize].fetch_sub(1, Ordering::AcqRel);
            }
        }
    }
    fn sum(&self, mask: LayerMask, kind: Option<u64>) -> u64 {
        let pin = self.kinds.pin();
        let values = match kind {
            Some(kind) => match pin.get(&kind) {
                Some(values) => values.as_ref(),
                None => return 0,
            },
            None => &self.layers,
        };
        Layer::ALL
            .into_iter()
            .filter(|layer| mask.contains_layer(*layer))
            .map(|layer| values[layer as usize].load(Ordering::Acquire))
            .sum()
    }
}

/// Free ordinal metadata is mutated only by the externally ordered writer.
pub(crate) struct FreeSlots {
    head: AtomicU32,
    links: Segments<AtomicU32>,
    available: AtomicUsize,
}
impl Default for FreeSlots {
    fn default() -> Self {
        Self {
            head: AtomicU32::new(NONE),
            links: Segments::default(),
            available: AtomicUsize::new(0),
        }
    }
}
impl FreeSlots {
    pub(crate) fn ensure(&self, row: u32) -> Result<()> {
        if self.links.get(row as usize).is_none() {
            self.links.push(AtomicU32::new(NONE))?;
        }
        Ok(())
    }
    pub(crate) fn pop(&self) -> Option<u32> {
        let row = self.head.load(Ordering::Acquire);
        if row == NONE {
            return None;
        }
        let next = self.links.get(row as usize)?.load(Ordering::Acquire);
        self.head.store(next, Ordering::Release);
        self.available.fetch_sub(1, Ordering::AcqRel);
        Some(row)
    }
    pub(crate) fn release(&self, row: u32) {
        if let Some(cell) = self.links.get(row as usize) {
            cell.store(self.head.load(Ordering::Acquire), Ordering::Release);
            self.head.store(row, Ordering::Release);
            self.available.fetch_add(1, Ordering::AcqRel);
        }
    }
}
fn publish_atomic(column: &Segments<AtomicU64>, row: u32, value: u64) -> Result<()> {
    if let Some(cell) = column.get(row as usize) {
        cell.store(value, Ordering::Release);
    } else {
        column.push(AtomicU64::new(value))?;
    }
    Ok(())
}
fn publish_active(column: &Segments<AtomicBool>, row: u32, value: bool) -> Result<()> {
    if let Some(cell) = column.get(row as usize) {
        cell.store(value, Ordering::Release);
    } else {
        column.push(AtomicBool::new(value))?;
    }
    Ok(())
}
fn next_stamp(counter: &AtomicU64) -> Result<u64> {
    counter
        .fetch_update(Ordering::AcqRel, Ordering::Acquire, |v| v.checked_add(1))
        .map(|v| v + 1)
        .map_err(|_| budget("allocation identity stamp exhausted"))
}
#[derive(Default)]
struct GraphInner {
    catalog: NameCatalog,
    revision: AtomicU64,
    layout: AtomicU64,
    next_node_id: AtomicU64,
    next_edge_id: AtomicU64,
    node_lookup: HashMap<u64, u32>,
    node_ids: Segments<AtomicU64>,
    node_layers: Segments<AtomicU64>,
    node_revisions: Segments<AtomicU64>,
    node_active: Segments<AtomicBool>,
    node_stamps: Segments<AtomicU64>,
    node_writers: Segments<Mutex<()>>,
    node_clock: AtomicU64,
    node_free: FreeSlots,
    labels: Segments<LabelsCell>,
    label_bytes: Arc<AtomicUsize>,
    outgoing: Segments<AtomicU64>,
    incoming: Segments<AtomicU64>,
    node_properties: Properties,
    edge_lookup: HashMap<u64, u32>,
    edge_ids: Segments<AtomicU64>,
    edge_sources: Segments<AtomicU64>,
    edge_targets: Segments<AtomicU64>,
    edge_types: Segments<AtomicU64>,
    edge_layers: Segments<AtomicU64>,
    edge_revisions: Segments<AtomicU64>,
    edge_active: Segments<AtomicBool>,
    edge_stamps: Segments<AtomicU64>,
    edge_writers: Segments<Mutex<()>>,
    edge_clock: AtomicU64,
    recovery_rejected: AtomicUsize,
    edge_free: FreeSlots,
    next_out: Segments<AtomicU64>,
    next_in: Segments<AtomicU64>,
    edge_properties: Properties,
    node_counts: Arc<Counts>,
    edge_counts: Arc<Counts>,
}

/// Clone means shared ownership of this exact store. It never constructs a working copy.
#[derive(Clone, Default)]
pub struct GraphStore(Arc<GraphInner>);
impl fmt::Debug for GraphStore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GraphStore")
            .field("nodes", &self.node_count())
            .field("edges", &self.edge_count())
            .field("revision", &self.revision())
            .finish()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct NodeView<'a> {
    graph: &'a GraphStore,
    dense: u32,
    id: NodeId,
    layer: Layer,
    stamp: u64,
    active: &'a AtomicBool,
    stamp_cell: &'a AtomicU64,
}
/// A handle to an existing canonical column, tied to its graph owner. It stores no row data.
pub struct NodePropertyReader<'a> {
    graph: &'a GraphStore,
    column: Arc<ConcurrentPropertyColumn>,
}
impl NodePropertyReader<'_> {
    /// Reads one primitive value while rejecting a deleted/recycled view or another graph owner.
    #[inline(always)]
    pub fn get_integer(&self, node: NodeView<'_>) -> Option<i64> {
        if !Arc::ptr_eq(&self.graph.0, &node.graph.0) || !node.current() {
            return None;
        }
        let value = self.column.get_integer(node.dense);
        node.current().then_some(value).flatten()
    }

    /// Selects labels and reads this column under one before/after owner validation.
    /// The outer option rejects a nonmatching or recycled owner; the inner option is its value.
    #[inline(always)]
    pub fn get_integer_with_labels(
        &self,
        node: NodeView<'_>,
        labels: &[LabelId],
    ) -> Option<Option<i64>> {
        if !Arc::ptr_eq(&self.graph.0, &node.graph.0) || !node.current() {
            return None;
        }
        if !labels.is_empty()
            && !self
                .graph
                .0
                .labels
                .get(node.dense as usize)
                .is_some_and(|cell| cell.contains_all(labels))
        {
            return None;
        }
        let value = self.column.get_integer(node.dense);
        node.current().then_some(value)
    }

    /// Visits conservative candidate ordinals directly in the existing property pages.
    /// No values or entity identities escape: callers must validate the current owner and
    /// evaluate their original predicate. A missing/noninteger value remains a candidate,
    /// including a cell concurrently promoted from INTEGER to an equal FLOAT.
    pub fn visit_integer_candidates(
        &self,
        end: usize,
        operand: i64,
        checkpoint: impl FnMut() -> Result<()>,
        visit: impl FnMut(u32) -> Result<()>,
    ) -> Result<()> {
        self.visit_integer_candidates_range(0, end, operand, checkpoint, visit)
    }

    /// Borrows disjoint ordinal ranges of the same column for bounded parallel scans.
    pub fn visit_integer_candidates_range(
        &self,
        start: usize,
        end: usize,
        operand: i64,
        mut checkpoint: impl FnMut() -> Result<()>,
        mut visit: impl FnMut(u32) -> Result<()>,
    ) -> Result<()> {
        let end = end.min(NONE as usize);
        for base in (start / SEGMENT * SEGMENT..end).step_by(SEGMENT) {
            checkpoint()?;
            let page = self.column.cells.directory.borrow_page(base / SEGMENT);
            let count = (end - base).min(SEGMENT);
            for offset in start.saturating_sub(base)..count {
                let value = page
                    .and_then(|page| page[offset].get())
                    .and_then(PropertyCell::get_integer);
                if value.is_none_or(|value| value == operand) {
                    visit((base + offset) as u32)?;
                }
            }
        }
        checkpoint()
    }
}
impl NodeView<'_> {
    fn current(self) -> bool {
        self.active.load(Ordering::Acquire) && self.stamp_cell.load(Ordering::Acquire) == self.stamp
    }
    pub fn id(self) -> NodeId {
        self.id
    }
    pub fn dense(self) -> u32 {
        self.dense
    }
    pub fn layer(self) -> Layer {
        self.layer
    }
    pub fn revision(self) -> u64 {
        let value = load(&self.graph.0.node_revisions, self.dense);
        if self.current() { value } else { 0 }
    }
    pub fn labels(self) -> LabelsHandle {
        let value = self.graph.labels(self.dense);
        if self.current() {
            value
        } else {
            LabelsHandle::default()
        }
    }
    /// Tests labels under one short allocation guard, without creating an owned labels handle.
    #[inline(always)]
    pub fn has_labels(self, labels: &[LabelId]) -> bool {
        if !self.current() {
            return false;
        }
        if labels.is_empty() {
            return true;
        }
        let Some(cell) = self.graph.0.labels.get(self.dense as usize) else {
            return false;
        };
        let selected = cell.contains_all(labels);
        selected && self.current()
    }
    pub fn property(self, property: PropertyId) -> Option<ScalarValue> {
        if !self.current() {
            return None;
        }
        let value = self.graph.0.node_properties.get(self.dense, property);
        self.current().then_some(value).flatten()
    }
    pub fn properties(self) -> Vec<(PropertyId, ScalarValue)> {
        if !self.current() {
            return Vec::new();
        }
        let value = self.graph.0.node_properties.values(self.dense);
        if self.current() { value } else { Vec::new() }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct EdgeView<'a> {
    graph: &'a GraphStore,
    dense: u32,
    id: EdgeId,
    source: NodeId,
    target: NodeId,
    kind: RelationshipTypeId,
    layer: Layer,
    stamp: u64,
    active: &'a AtomicBool,
    stamp_cell: &'a AtomicU64,
}
impl EdgeView<'_> {
    fn current(self) -> bool {
        self.active.load(Ordering::Acquire) && self.stamp_cell.load(Ordering::Acquire) == self.stamp
    }
    pub fn id(self) -> EdgeId {
        self.id
    }
    pub fn dense(self) -> u32 {
        self.dense
    }
    pub fn source(self) -> NodeId {
        self.source
    }
    pub fn target(self) -> NodeId {
        self.target
    }
    pub fn relationship_type(self) -> RelationshipTypeId {
        self.kind
    }
    pub fn layer(self) -> Layer {
        self.layer
    }
    pub fn revision(self) -> u64 {
        let value = load(&self.graph.0.edge_revisions, self.dense);
        if self.current() { value } else { 0 }
    }
    pub fn property(self, property: PropertyId) -> Option<ScalarValue> {
        if !self.current() {
            return None;
        }
        let value = self.graph.0.edge_properties.get(self.dense, property);
        self.current().then_some(value).flatten()
    }
    pub fn properties(self) -> Vec<(PropertyId, ScalarValue)> {
        if !self.current() {
            return Vec::new();
        }
        let value = self.graph.0.edge_properties.values(self.dense);
        if self.current() { value } else { Vec::new() }
    }
}

impl GraphStore {
    fn claim_node(&self, node: NodeView<'_>) -> Result<std::sync::MutexGuard<'_, ()>> {
        let permit = self
            .0
            .node_writers
            .get(node.dense as usize)
            .ok_or_else(|| Error::internal("node writer permit absent"))?;
        let guard = permit
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !node.current() {
            return Err(missing("node no longer exists"));
        }
        Ok(guard)
    }
    fn claim_edge(&self, edge: EdgeView<'_>) -> Result<std::sync::MutexGuard<'_, ()>> {
        let permit = self
            .0
            .edge_writers
            .get(edge.dense as usize)
            .ok_or_else(|| Error::internal("relationship writer permit absent"))?;
        let guard = permit
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if !edge.current() {
            return Err(missing("relationship no longer exists"));
        }
        Ok(guard)
    }
    pub fn catalog(&self) -> &NameCatalog {
        &self.0.catalog
    }
    pub fn catalog_mut(&self) -> &NameCatalog {
        self.catalog()
    }
    pub fn revision(&self) -> u64 {
        self.0.revision.load(Ordering::Acquire)
    }
    pub fn layout_version(&self) -> u64 {
        self.0.layout.load(Ordering::Acquire)
    }
    /// Reserve an identity before a statement plans its insertion. Aborted statements may
    /// leave gaps; readers and the ordered mutation publisher do not acquire a lock here.
    pub fn reserve_node_id(&self, minimum: u64) -> Result<NodeId> {
        self.0
            .next_node_id
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                next.max(minimum).checked_add(1)
            })
            .map(|next| NodeId(next.max(minimum)))
            .map_err(|_| budget("node ID space exhausted"))
    }
    pub fn reserve_edge_id(&self, minimum: u64) -> Result<EdgeId> {
        self.0
            .next_edge_id
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |next| {
                next.max(minimum).checked_add(1)
            })
            .map(|next| EdgeId(next.max(minimum)))
            .map_err(|_| budget("relationship ID space exhausted"))
    }
    pub fn node_slot_count(&self) -> usize {
        self.0.node_ids.len()
    }
    pub fn edge_slot_count(&self) -> usize {
        self.0.edge_ids.len()
    }
    pub fn node_count(&self) -> usize {
        self.0.node_counts.sum(LayerMask::ALL, None) as usize
    }
    pub fn edge_count(&self) -> usize {
        self.0.edge_counts.sum(LayerMask::ALL, None) as usize
    }
    pub fn node_count_in_layers(&self, mask: LayerMask) -> u64 {
        self.0.node_counts.sum(mask, None)
    }
    pub fn edge_count_in_layers(&self, mask: LayerMask) -> u64 {
        self.0.edge_counts.sum(mask, None)
    }
    pub fn label_node_count(&self, label: LabelId, mask: LayerMask) -> u64 {
        self.0.node_counts.sum(mask, Some(label.0))
    }
    pub fn relationship_count(&self, kind: RelationshipTypeId, mask: LayerMask) -> u64 {
        self.0.edge_counts.sum(mask, Some(kind.0))
    }
    pub fn optimizer_count_generations(&self) -> (CanonicalCounts, CanonicalCounts) {
        (
            CanonicalCounts(Arc::clone(&self.0.node_counts)),
            CanonicalCounts(Arc::clone(&self.0.edge_counts)),
        )
    }
    pub fn contains_node_id(&self, id: NodeId) -> bool {
        self.0.node_lookup.pin().contains_key(&id.0)
    }
    pub fn contains_edge_id(&self, id: EdgeId) -> bool {
        self.0.edge_lookup.pin().contains_key(&id.0)
    }
    fn labels(&self, row: u32) -> LabelsHandle {
        self.0
            .labels
            .get(row as usize)
            .map_or_else(LabelsHandle::default, LabelsCell::handle)
    }
    fn node_active(&self, row: u32) -> bool {
        self.0
            .node_active
            .get(row as usize)
            .is_some_and(|active| active.load(Ordering::Acquire))
    }
    fn edge_active(&self, row: u32) -> bool {
        self.0
            .edge_active
            .get(row as usize)
            .is_some_and(|active| active.load(Ordering::Acquire))
    }
    #[inline(always)]
    pub fn node_dense(&self, dense: u32) -> Option<NodeView<'_>> {
        let stamp_cell = self.0.node_stamps.get(dense as usize)?;
        let active = self.0.node_active.get(dense as usize)?;
        let stamp = stamp_cell.load(Ordering::Acquire);
        if !active.load(Ordering::Acquire) {
            return None;
        }
        let view = NodeView {
            graph: self,
            dense,
            id: NodeId(load(&self.0.node_ids, dense)),
            layer: layer(load(&self.0.node_layers, dense)),
            stamp,
            active,
            stamp_cell,
        };
        view.current().then_some(view)
    }
    pub fn edge_dense(&self, dense: u32) -> Option<EdgeView<'_>> {
        let stamp_cell = self.0.edge_stamps.get(dense as usize)?;
        let active = self.0.edge_active.get(dense as usize)?;
        let stamp = stamp_cell.load(Ordering::Acquire);
        if !active.load(Ordering::Acquire) {
            return None;
        }
        let source = self.node_dense(load(&self.0.edge_sources, dense) as u32)?;
        let target = self.node_dense(load(&self.0.edge_targets, dense) as u32)?;
        let view = EdgeView {
            graph: self,
            dense,
            id: EdgeId(load(&self.0.edge_ids, dense)),
            source: source.id(),
            target: target.id(),
            kind: RelationshipTypeId(load(&self.0.edge_types, dense)),
            layer: layer(load(&self.0.edge_layers, dense)),
            stamp,
            active,
            stamp_cell,
        };
        (view.current() && source.current() && target.current()).then_some(view)
    }
    pub fn node(&self, id: NodeId) -> Option<NodeView<'_>> {
        let row = *self.0.node_lookup.pin().get(&id.0)?;
        self.node_dense(row).filter(|view| view.id() == id)
    }
    pub fn edge(&self, id: EdgeId) -> Option<EdgeView<'_>> {
        let row = *self.0.edge_lookup.pin().get(&id.0)?;
        self.edge_dense(row).filter(|view| view.id() == id)
    }
    pub fn nodes(&self) -> impl Iterator<Item = NodeView<'_>> {
        (0..self.node_slot_count()).filter_map(|row| self.node_dense(row as u32))
    }
    pub fn edges(&self) -> impl Iterator<Item = EdgeView<'_>> {
        (0..self.edge_slot_count()).filter_map(|row| self.edge_dense(row as u32))
    }
    pub fn scan_nodes(
        &self,
        label: Option<LabelId>,
        mask: LayerMask,
    ) -> impl Iterator<Item = NodeView<'_>> {
        self.nodes().filter(move |node| {
            mask.contains_layer(node.layer())
                && label.is_none_or(|label| node.labels().contains(&label))
        })
    }
    fn forward(&self, revision: u64) -> Result<()> {
        if revision < self.revision() {
            Err(Error::invalid_data(
                "mutation revision moves graph backwards",
            ))
        } else {
            Ok(())
        }
    }
    fn validate_properties(&self, values: &[(PropertyId, ScalarValue)]) -> Result<()> {
        let mut seen = BTreeSet::new();
        for (property, value) in values {
            super::columns::validate_property_value_shape(value)?;
            if !seen.insert(*property) {
                return Err(Error::invalid_data("duplicate property"));
            }
            if self.catalog().property_name(*property).is_none() {
                return Err(Error::invalid_data("undeclared property ID"));
            }
        }
        Ok(())
    }
    pub fn validate_node_property_value(
        &self,
        property: PropertyId,
        value: &ScalarValue,
    ) -> Result<()> {
        super::columns::validate_property_value_shape(value)?;
        if self.catalog().property_name(property).is_none() {
            Err(Error::invalid_data("undeclared property ID"))
        } else {
            Ok(())
        }
    }
    pub fn validate_edge_property_value(
        &self,
        property: PropertyId,
        value: &ScalarValue,
    ) -> Result<()> {
        self.validate_node_property_value(property, value)
    }
    pub fn insert_node(&self, mut input: NodeInput) -> Result<u32> {
        self.forward(input.revision)?;
        if self.contains_node_id(input.id) {
            return Err(Error::new(
                ErrorCode::TransactionConflict,
                "node ID already allocated",
            ));
        }
        input.labels.sort_unstable();
        input.labels.dedup();
        if input
            .labels
            .iter()
            .any(|id| self.catalog().label_name(*id).is_none())
        {
            return Err(Error::invalid_data("undeclared label ID"));
        }
        self.validate_properties(&input.properties)?;
        self.append_node(input, true)
    }
    fn append_node(&self, input: NodeInput, active: bool) -> Result<u32> {
        self.0
            .next_node_id
            .fetch_max(input.id.0.saturating_add(1), Ordering::Relaxed);
        let recycled = active.then(|| self.0.node_free.pop()).flatten();
        let row = recycled.unwrap_or(
            u32::try_from(self.node_slot_count())
                .map_err(|_| budget("node row space exhausted"))?,
        );
        if row == NONE {
            return Err(budget("node row space exhausted"));
        }
        self.0.node_free.ensure(row)?;
        let permit = self
            .0
            .node_writers
            .get_or_insert_with(row as usize, || Mutex::new(()))?;
        let _owner = permit
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        publish_atomic(&self.0.node_stamps, row, next_stamp(&self.0.node_clock)?)?;
        publish_atomic(&self.0.node_layers, row, input.layer as u64)?;
        publish_atomic(&self.0.node_revisions, row, input.revision)?;
        let labels = if active {
            input.labels.clone()
        } else {
            Vec::new()
        };
        if let Some(cell) = self.0.labels.get(row as usize) {
            cell.store(labels, &self.0.label_bytes);
        } else {
            self.0
                .labels
                .push(LabelsCell::new(labels, &self.0.label_bytes))?;
        }
        publish_atomic(&self.0.outgoing, row, NO_EDGE)?;
        publish_atomic(&self.0.incoming, row, NO_EDGE)?;
        for (property, value) in input.properties {
            if active {
                self.0.node_properties.set(row, property, value)?;
            }
        }
        publish_atomic(&self.0.node_ids, row, input.id.0)?;
        publish_active(&self.0.node_active, row, active)?;
        if active {
            self.0.node_lookup.pin().insert(input.id.0, row);
            self.0.node_counts.adjust(input.layer, None, true);
            for label in input.labels {
                self.0.node_counts.adjust(input.layer, Some(label.0), true);
            }
        }
        self.0.revision.fetch_max(input.revision, Ordering::AcqRel);
        Ok(row)
    }
    pub fn insert_edge(&self, input: EdgeInput) -> Result<u32> {
        self.forward(input.revision)?;
        if self.contains_edge_id(input.id) {
            return Err(Error::new(
                ErrorCode::TransactionConflict,
                "relationship ID already allocated",
            ));
        }
        let source = self
            .node(input.source)
            .ok_or_else(|| missing("source node does not exist"))?
            .dense();
        let target = self
            .node(input.target)
            .ok_or_else(|| missing("target node does not exist"))?
            .dense();
        if self
            .catalog()
            .relationship_type_name(input.relationship_type)
            .is_none()
        {
            return Err(Error::invalid_data("undeclared relationship type"));
        }
        self.validate_properties(&input.properties)?;
        self.append_edge(input, source, target, true)
    }
    fn append_edge(&self, input: EdgeInput, source: u32, target: u32, active: bool) -> Result<u32> {
        if input.id.0 == NO_EDGE {
            return Err(budget(
                "relationship identity reserved for adjacency terminator",
            ));
        }
        self.0
            .next_edge_id
            .fetch_max(input.id.0 + 1, Ordering::Relaxed);
        // Writers claim endpoints in dense order, then the relationship. Readers
        // never take these lifecycle permits. Deletion uses node -> edge too.
        let first = active
            .then(|| self.node_dense(source.min(target)))
            .flatten();
        let second = (active && source != target)
            .then(|| self.node_dense(source.max(target)))
            .flatten();
        if active && (first.is_none() || (source != target && second.is_none())) {
            return Err(missing("relationship endpoint does not exist"));
        }
        let _first_owner = first.map(|node| self.claim_node(node)).transpose()?;
        let _second_owner = second.map(|node| self.claim_node(node)).transpose()?;
        if active
            && (self
                .node_dense(source)
                .is_none_or(|node| node.id() != input.source)
                || self
                    .node_dense(target)
                    .is_none_or(|node| node.id() != input.target))
        {
            return Err(missing("relationship endpoint identity changed"));
        }
        let recycled = active.then(|| self.0.edge_free.pop()).flatten();
        let row = recycled.unwrap_or(
            u32::try_from(self.edge_slot_count())
                .map_err(|_| budget("relationship row space exhausted"))?,
        );
        if row == NONE {
            return Err(budget("relationship row space exhausted"));
        }
        self.0.edge_free.ensure(row)?;
        let permit = self
            .0
            .edge_writers
            .get_or_insert_with(row as usize, || Mutex::new(()))?;
        let _owner = permit
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        publish_atomic(&self.0.edge_stamps, row, next_stamp(&self.0.edge_clock)?)?;
        publish_atomic(&self.0.edge_sources, row, u64::from(source))?;
        publish_atomic(&self.0.edge_targets, row, u64::from(target))?;
        publish_atomic(&self.0.edge_types, row, input.relationship_type.0)?;
        publish_atomic(&self.0.edge_layers, row, input.layer as u64)?;
        publish_atomic(&self.0.edge_revisions, row, input.revision)?;
        let outgoing = self.0.outgoing.get(source as usize);
        let incoming = self.0.incoming.get(target as usize);
        publish_atomic(
            &self.0.next_out,
            row,
            if active {
                outgoing.map_or(NO_EDGE, |head| head.load(Ordering::Acquire))
            } else {
                NO_EDGE
            },
        )?;
        publish_atomic(
            &self.0.next_in,
            row,
            if active {
                incoming.map_or(NO_EDGE, |head| head.load(Ordering::Acquire))
            } else {
                NO_EDGE
            },
        )?;
        for (property, value) in input.properties {
            if active {
                self.0.edge_properties.set(row, property, value)?;
            }
        }
        publish_atomic(&self.0.edge_ids, row, input.id.0)?;
        publish_active(&self.0.edge_active, row, active)?;
        if active {
            self.0.edge_lookup.pin().insert(input.id.0, row);
            if let Some(head) = outgoing {
                head.store(u64::from(row), Ordering::Release);
            }
            if let Some(head) = incoming {
                head.store(u64::from(row), Ordering::Release);
            }
            self.0.edge_counts.adjust(input.layer, None, true);
            self.0
                .edge_counts
                .adjust(input.layer, Some(input.relationship_type.0), true);
        }
        self.0.revision.fetch_max(input.revision, Ordering::AcqRel);
        Ok(row)
    }
    fn unlink_edge(&self, edge_row: u32, source: u32, target: u32) {
        for (heads, links, owner) in [
            (&self.0.outgoing, &self.0.next_out, source),
            (&self.0.incoming, &self.0.next_in, target),
        ] {
            let Some(head) = heads.get(owner as usize) else {
                continue;
            };
            let mut previous = head;
            let mut token = head.load(Ordering::Acquire);
            for _ in 0..self.edge_slot_count() {
                if token == NO_EDGE {
                    break;
                }
                let Ok(row) = u32::try_from(token) else {
                    break;
                };
                let Some(link) = links.get(row as usize) else {
                    break;
                };
                let next = link.load(Ordering::Acquire);
                if row == edge_row {
                    previous.store(next, Ordering::Release);
                    break;
                }
                previous = link;
                token = next;
            }
        }
    }
    pub fn set_node_property(
        &self,
        node: NodeId,
        property: PropertyId,
        value: ScalarValue,
        revision: u64,
    ) -> Result<()> {
        self.forward(revision)?;
        self.validate_node_property_value(property, &value)?;
        let view = self
            .node(node)
            .ok_or_else(|| missing("node does not exist"))?;
        self.set_node_property_view(view, property, value, revision)
    }
    fn set_node_property_view(
        &self,
        view: NodeView<'_>,
        property: PropertyId,
        value: ScalarValue,
        revision: u64,
    ) -> Result<()> {
        let _owner = self.claim_node(view)?;
        self.forward(revision)?;
        let row = view.dense();
        self.0.node_properties.set(row, property, value)?;
        if let Some(cell) = self.0.node_revisions.get(row as usize) {
            cell.store(revision, Ordering::Release);
        }
        self.0.revision.fetch_max(revision, Ordering::AcqRel);
        Ok(())
    }
    pub fn set_edge_property(
        &self,
        edge: EdgeId,
        property: PropertyId,
        value: ScalarValue,
        revision: u64,
    ) -> Result<()> {
        self.forward(revision)?;
        self.validate_edge_property_value(property, &value)?;
        let view = self
            .edge(edge)
            .ok_or_else(|| missing("relationship does not exist"))?;
        let _owner = self.claim_edge(view)?;
        self.forward(revision)?;
        let row = view.dense();
        self.0.edge_properties.set(row, property, value)?;
        if let Some(cell) = self.0.edge_revisions.get(row as usize) {
            cell.store(revision, Ordering::Release);
        }
        self.0.revision.fetch_max(revision, Ordering::AcqRel);
        Ok(())
    }
    pub fn add_node_labels(&self, node: NodeId, labels: Vec<LabelId>, revision: u64) -> Result<()> {
        self.change_labels(node, labels, revision, true)
    }
    pub fn remove_node_labels(
        &self,
        node: NodeId,
        labels: Vec<LabelId>,
        revision: u64,
    ) -> Result<()> {
        self.change_labels(node, labels, revision, false)
    }
    fn change_labels(
        &self,
        id: NodeId,
        labels: Vec<LabelId>,
        revision: u64,
        add: bool,
    ) -> Result<()> {
        self.forward(revision)?;
        if labels
            .iter()
            .any(|id| self.catalog().label_name(*id).is_none())
        {
            return Err(Error::invalid_data("undeclared label ID"));
        }
        let node = self
            .node(id)
            .ok_or_else(|| missing("node does not exist"))?;
        let _owner = self.claim_node(node)?;
        self.forward(revision)?;
        let old = node.labels();
        let mut next = old.to_vec();
        if add {
            next.extend(labels);
            next.sort_unstable();
            next.dedup();
        } else {
            next.retain(|label| !labels.contains(label));
        }
        for label in old.iter().filter(|label| !next.contains(label)) {
            self.0
                .node_counts
                .adjust(node.layer(), Some(label.0), false);
        }
        for label in next.iter().filter(|label| !old.contains(label)) {
            self.0.node_counts.adjust(node.layer(), Some(label.0), true);
        }
        if let Some(cell) = self.0.labels.get(node.dense as usize) {
            cell.store(next, &self.0.label_bytes);
        }
        if let Some(cell) = self.0.node_revisions.get(node.dense as usize) {
            cell.store(revision, Ordering::Release);
        }
        self.0.revision.fetch_max(revision, Ordering::AcqRel);
        Ok(())
    }
    pub fn delete_edge(&self, id: EdgeId, revision: u64) -> Result<()> {
        self.forward(revision)?;
        let edge = self
            .edge(id)
            .ok_or_else(|| missing("relationship does not exist"))?;
        let _owner = self.claim_edge(edge)?;
        self.forward(revision)?;
        let row = edge.dense;
        let layer = edge.layer();
        let kind = edge.relationship_type();
        if let Some(active) = self.0.edge_active.get(row as usize) {
            active.store(false, Ordering::Release);
        }
        self.unlink_edge(
            row,
            load(&self.0.edge_sources, row) as u32,
            load(&self.0.edge_targets, row) as u32,
        );
        self.0.edge_lookup.pin().remove(&id.0);
        self.0.edge_properties.clear(row);
        self.0.edge_counts.adjust(layer, None, false);
        self.0.edge_counts.adjust(layer, Some(kind.0), false);
        if let Some(cell) = self.0.edge_revisions.get(row as usize) {
            cell.store(revision, Ordering::Release);
        }
        self.0.revision.fetch_max(revision, Ordering::AcqRel);
        self.0.edge_free.release(row);
        Ok(())
    }
    pub fn delete_node(&self, id: NodeId, detach: bool, revision: u64) -> Result<()> {
        self.forward(revision)?;
        let node = self
            .node(id)
            .ok_or_else(|| missing("node does not exist"))?;
        let _owner = self.claim_node(node)?;
        self.forward(revision)?;
        let edges = self.incident_edge_ids(id)?;
        if !detach && !edges.is_empty() {
            return Err(missing("node still has relationships"));
        }
        for edge in edges {
            self.delete_edge(edge, revision)?;
        }
        let labels = node.labels();
        let row = node.dense;
        let layer = node.layer();
        if let Some(active) = self.0.node_active.get(row as usize) {
            active.store(false, Ordering::Release);
        }
        self.0.node_lookup.pin().remove(&id.0);
        self.0.node_properties.clear(row);
        if let Some(cell) = self.0.labels.get(row as usize) {
            cell.store(Vec::new(), &self.0.label_bytes);
        }
        self.0.node_counts.adjust(layer, None, false);
        for label in labels.iter() {
            self.0.node_counts.adjust(layer, Some(label.0), false);
        }
        if let Some(cell) = self.0.node_revisions.get(row as usize) {
            cell.store(revision, Ordering::Release);
        }
        self.0.revision.fetch_max(revision, Ordering::AcqRel);
        self.0.node_free.release(row);
        Ok(())
    }
    pub fn apply(&self, mutation: GraphMutation) -> Result<()> {
        match mutation {
            GraphMutation::DeclareLabel { name, id } => self.catalog().declare_label(name, id),
            GraphMutation::DeclareProperty { name, id } => {
                self.catalog().declare_property(name, id)
            }
            GraphMutation::DeclareRelationshipType { name, id } => {
                self.catalog().declare_relationship_type(name, id)
            }
            GraphMutation::InsertNode(input) => self.insert_node(input).map(|_| ()),
            GraphMutation::InsertEdge(input) => self.insert_edge(input).map(|_| ()),
            GraphMutation::SetNodeProperty {
                node,
                property,
                value,
                revision,
            } => self.set_node_property(node, property, value, revision),
            GraphMutation::SetEdgeProperty {
                edge,
                property,
                value,
                revision,
            } => self.set_edge_property(edge, property, value, revision),
            GraphMutation::AddNodeLabels {
                node,
                labels,
                revision,
            } => self.add_node_labels(node, labels, revision),
            GraphMutation::RemoveNodeLabels {
                node,
                labels,
                revision,
            } => self.remove_node_labels(node, labels, revision),
            GraphMutation::DeleteNode {
                node,
                detach,
                revision,
            } => self.delete_node(node, detach, revision),
            GraphMutation::DeleteEdge { edge, revision } => self.delete_edge(edge, revision),
        }
    }
    /// Validates sequential batch effects using only changed IDs/schema and owner-local edges.
    /// The database must keep its writer reservation from this check through ordered apply.
    pub fn validate_mutations(&self, mutations: &[GraphMutation]) -> Result<()> {
        self.validate_mutations_inner(mutations, None)
    }

    /// Ordered database admission uses its assigned commit revision, matching publication.
    /// Planning revisions are provisional; no mutation data is copied or rewritten here.
    pub fn validate_mutations_at_revision(
        &self,
        mutations: &[GraphMutation],
        revision: u64,
    ) -> Result<()> {
        self.validate_mutations_inner(mutations, Some(revision))
    }

    fn validate_mutations_inner(
        &self,
        mutations: &[GraphMutation],
        ordered_revision: Option<u64>,
    ) -> Result<()> {
        #[derive(Clone, Copy)]
        struct EdgeState {
            active: bool,
            source: NodeId,
            target: NodeId,
        }
        let mut nodes = BTreeMap::<NodeId, bool>::new();
        let mut edges = BTreeMap::<EdgeId, EdgeState>::new();
        let mut schemas = BTreeMap::<(u8, u64), String>::new();
        let mut names = BTreeMap::<(u8, String), u64>::new();
        let mut revision = self.revision();
        let mut new_nodes = 0_usize;
        let mut new_edges = 0_usize;
        let node_live = |id: NodeId, nodes: &BTreeMap<NodeId, bool>| {
            nodes
                .get(&id)
                .copied()
                .unwrap_or_else(|| self.node(id).is_some())
        };
        let schema_exists = |kind: u8, id: u64, schemas: &BTreeMap<(u8, u64), String>| {
            schemas.contains_key(&(kind, id))
                || match kind {
                    0 => self.catalog().label_name(LabelId(id)).is_some(),
                    1 => self.catalog().property_name(PropertyId(id)).is_some(),
                    _ => self
                        .catalog()
                        .relationship_type_name(RelationshipTypeId(id))
                        .is_some(),
                }
        };
        let properties_valid = |properties: &[(PropertyId, ScalarValue)],
                                schemas: &BTreeMap<(u8, u64), String>|
         -> Result<()> {
            let mut seen = BTreeSet::new();
            for (property, value) in properties {
                if !seen.insert(*property) {
                    return Err(Error::invalid_data("duplicate property"));
                }
                if !schema_exists(1, property.0, schemas) {
                    return Err(Error::invalid_data("undeclared property ID"));
                }
                super::columns::validate_property_value_shape(value)?;
            }
            Ok(())
        };
        for mutation in mutations {
            let declaration = match mutation {
                GraphMutation::DeclareLabel { name, id } => Some((0, id.0, name)),
                GraphMutation::DeclareProperty { name, id } => Some((1, id.0, name)),
                GraphMutation::DeclareRelationshipType { name, id } => Some((2, id.0, name)),
                _ => None,
            };
            if let Some((kind, id, name)) = declaration {
                id.checked_add(1)
                    .ok_or_else(|| budget("schema ID space exhausted"))?;
                let old_name = match kind {
                    0 => self.catalog().label_name(LabelId(id)),
                    1 => self.catalog().property_name(PropertyId(id)),
                    _ => self
                        .catalog()
                        .relationship_type_name(RelationshipTypeId(id)),
                };
                let old_id = match kind {
                    0 => self.catalog().label(name).map(|id| id.0),
                    1 => self.catalog().property(name).map(|id| id.0),
                    _ => self.catalog().relationship_type(name).map(|id| id.0),
                };
                if old_name.is_some_and(|old| old.as_ref() != name)
                    || old_id.is_some_and(|old| old != id)
                    || schemas.get(&(kind, id)).is_some_and(|old| old != name)
                    || names
                        .get(&(kind, name.clone()))
                        .is_some_and(|old| *old != id)
                {
                    return Err(Error::new(
                        ErrorCode::TransactionConflict,
                        "schema name or ID already assigned",
                    ));
                }
                schemas.insert((kind, id), name.clone());
                names.insert((kind, name.clone()), id);
                continue;
            }
            let next_revision = ordered_revision.unwrap_or(match mutation {
                GraphMutation::InsertNode(input) => input.revision,
                GraphMutation::InsertEdge(input) => input.revision,
                GraphMutation::SetNodeProperty { revision, .. }
                | GraphMutation::SetEdgeProperty { revision, .. }
                | GraphMutation::AddNodeLabels { revision, .. }
                | GraphMutation::RemoveNodeLabels { revision, .. }
                | GraphMutation::DeleteNode { revision, .. }
                | GraphMutation::DeleteEdge { revision, .. } => *revision,
                _ => revision,
            });
            if next_revision < revision {
                return Err(Error::invalid_data(
                    "mutation revision moves graph backwards",
                ));
            }
            revision = next_revision;
            match mutation {
                GraphMutation::InsertNode(input) => {
                    if nodes.contains_key(&input.id) || self.contains_node_id(input.id) {
                        return Err(Error::new(
                            ErrorCode::TransactionConflict,
                            "node ID already allocated",
                        ));
                    }
                    if input
                        .labels
                        .iter()
                        .any(|id| !schema_exists(0, id.0, &schemas))
                    {
                        return Err(Error::invalid_data("undeclared label ID"));
                    }
                    properties_valid(&input.properties, &schemas)?;
                    nodes.insert(input.id, true);
                    new_nodes += 1;
                }
                GraphMutation::InsertEdge(input) => {
                    if input.id.0 == NO_EDGE {
                        return Err(budget(
                            "relationship identity reserved for adjacency terminator",
                        ));
                    }
                    if edges.contains_key(&input.id) || self.contains_edge_id(input.id) {
                        return Err(Error::new(
                            ErrorCode::TransactionConflict,
                            "relationship ID already allocated",
                        ));
                    }
                    if !node_live(input.source, &nodes) || !node_live(input.target, &nodes) {
                        return Err(missing("relationship endpoint does not exist"));
                    }
                    if !schema_exists(2, input.relationship_type.0, &schemas) {
                        return Err(Error::invalid_data("undeclared relationship type"));
                    }
                    properties_valid(&input.properties, &schemas)?;
                    edges.insert(
                        input.id,
                        EdgeState {
                            active: true,
                            source: input.source,
                            target: input.target,
                        },
                    );
                    new_edges += 1;
                }
                GraphMutation::SetNodeProperty {
                    node,
                    property,
                    value,
                    ..
                } => {
                    if !node_live(*node, &nodes) {
                        return Err(missing("node does not exist"));
                    }
                    properties_valid(&[(*property, value.clone())], &schemas)?;
                }
                GraphMutation::AddNodeLabels { node, labels, .. }
                | GraphMutation::RemoveNodeLabels { node, labels, .. } => {
                    if !node_live(*node, &nodes) {
                        return Err(missing("node does not exist"));
                    }
                    if labels.iter().any(|id| !schema_exists(0, id.0, &schemas)) {
                        return Err(Error::invalid_data("undeclared label ID"));
                    }
                }
                GraphMutation::SetEdgeProperty {
                    edge,
                    property,
                    value,
                    ..
                } => {
                    let live = edges.get(edge).is_some_and(|state| state.active)
                        || (!edges.contains_key(edge) && self.edge(*edge).is_some());
                    if !live {
                        return Err(missing("relationship does not exist"));
                    }
                    properties_valid(&[(*property, value.clone())], &schemas)?;
                }
                GraphMutation::DeleteEdge { edge, .. } => {
                    let state = edges
                        .get(edge)
                        .copied()
                        .or_else(|| {
                            self.edge(*edge).map(|view| EdgeState {
                                active: true,
                                source: view.source(),
                                target: view.target(),
                            })
                        })
                        .ok_or_else(|| missing("relationship does not exist"))?;
                    if !state.active {
                        return Err(missing("relationship does not exist"));
                    }
                    edges.insert(
                        *edge,
                        EdgeState {
                            active: false,
                            ..state
                        },
                    );
                }
                GraphMutation::DeleteNode { node, detach, .. } => {
                    if !node_live(*node, &nodes) {
                        return Err(missing("node does not exist"));
                    }
                    let mut incident = BTreeSet::new();
                    if self.node(*node).is_some() {
                        for edge in self.incident_edge_ids(*node)? {
                            if !edges.get(&edge).is_some_and(|state| !state.active) {
                                incident.insert(edge);
                            }
                        }
                    }
                    incident.extend(
                        edges
                            .iter()
                            .filter(|(_, state)| {
                                state.active && (state.source == *node || state.target == *node)
                            })
                            .map(|(id, _)| *id),
                    );
                    if !detach && !incident.is_empty() {
                        return Err(missing("node still has relationships"));
                    }
                    for edge in incident {
                        let state = edges
                            .get(&edge)
                            .copied()
                            .or_else(|| {
                                self.edge(edge).map(|view| EdgeState {
                                    active: true,
                                    source: view.source(),
                                    target: view.target(),
                                })
                            })
                            .ok_or_else(|| {
                                Error::invalid_data("incident edge disappeared during validation")
                            })?;
                        edges.insert(
                            edge,
                            EdgeState {
                                active: false,
                                ..state
                            },
                        );
                    }
                    nodes.insert(*node, false);
                }
                _ => {}
            }
        }
        if self
            .node_slot_count()
            .saturating_sub(self.0.node_free.available.load(Ordering::Acquire))
            .checked_add(new_nodes)
            .is_none_or(|count| count >= NONE as usize)
            || self
                .edge_slot_count()
                .saturating_sub(self.0.edge_free.available.load(Ordering::Acquire))
                .checked_add(new_edges)
                .is_none_or(|count| count >= NONE as usize)
        {
            return Err(budget("canonical row space exhausted"));
        }
        if self
            .0
            .node_clock
            .load(Ordering::Acquire)
            .checked_add(new_nodes as u64)
            .is_none()
            || self
                .0
                .edge_clock
                .load(Ordering::Acquire)
                .checked_add(new_edges as u64)
                .is_none()
        {
            return Err(budget("allocation identity stamp exhausted"));
        }
        Ok(())
    }
    fn expand(
        &self,
        id: NodeId,
        kind: Option<RelationshipTypeId>,
        mask: LayerMask,
        limit: usize,
        outgoing: bool,
    ) -> Result<Vec<(EdgeView<'_>, NodeView<'_>)>> {
        let node = self
            .node(id)
            .ok_or_else(|| missing("node does not exist"))?;
        let mut result = Vec::new();
        self.walk_adjacency(node, kind, mask, limit, outgoing, |edge, other| {
            result.push((edge, other))
        });
        result.sort_by_key(|(edge, node)| (node.id(), edge.id()));
        Ok(result)
    }

    /// Appends live canonical ordinal pairs to reusable query scratch, without a row allocation.
    pub fn append_neighbor_denses(
        &self,
        node: NodeView<'_>,
        outgoing: bool,
        mask: LayerMask,
        output: &mut Vec<(u32, u32)>,
    ) -> bool {
        self.visit_neighbor_denses(node, outgoing, mask, |neighbor, edge| {
            output.push((neighbor, edge));
        })
    }

    /// Visits live canonical ordinal pairs without retaining an adjacency row.
    pub fn visit_neighbor_denses(
        &self,
        node: NodeView<'_>,
        outgoing: bool,
        mask: LayerMask,
        visit: impl FnMut(u32, u32),
    ) -> bool {
        if !Arc::ptr_eq(&self.0, &node.graph.0) || !node.current() {
            return false;
        }
        let heads = if outgoing {
            &self.0.outgoing
        } else {
            &self.0.incoming
        };
        let token = heads
            .get(node.dense as usize)
            .map_or(NO_EDGE, |head| head.load(Ordering::Acquire));
        self.walk_neighbor_denses_from(node, outgoing, mask, token, visit);
        true
    }

    /// A captured head may name a recycled slot. Allocation stamps and live owner checks
    /// validate every emitted pair; strictly descending stamps bound traversal through cycles.
    fn walk_neighbor_denses_from(
        &self,
        node: NodeView<'_>,
        outgoing: bool,
        mask: LayerMask,
        token: u64,
        mut visit: impl FnMut(u32, u32),
    ) {
        self.walk_neighbor_denses_until(
            node,
            outgoing,
            mask,
            token,
            |neighbor, edge, _, _, _, layer| {
                if mask.contains_layer(layer) && mask.contains_layer(neighbor.layer()) {
                    visit(neighbor.dense(), edge);
                }
                true
            },
        );
    }

    /// Visits the same canonical adjacency, stopping immediately when the visitor fails.
    pub fn try_visit_neighbors<'a>(
        &'a self,
        node: NodeView<'a>,
        outgoing: bool,
        mask: LayerMask,
        mut check: impl FnMut() -> Result<()>,
        mut visit: impl FnMut(EdgeView<'a>, NodeView<'a>) -> Result<()>,
    ) -> Result<bool> {
        if !Arc::ptr_eq(&self.0, &node.graph.0) || !node.current() {
            return Ok(false);
        }
        let heads = if outgoing {
            &self.0.outgoing
        } else {
            &self.0.incoming
        };
        let token = heads
            .get(node.dense as usize)
            .map_or(NO_EDGE, |head| head.load(Ordering::Acquire));
        let mut failure = None;
        let mut visited = 0_usize;
        self.walk_neighbor_denses_until(
            node,
            outgoing,
            mask,
            token,
            |neighbor, dense, stamp, active, stamp_cell, layer| {
                visited += 1;
                if visited.is_multiple_of(64)
                    && let Err(error) = check()
                {
                    failure = Some(error);
                    return false;
                }
                if !mask.contains_layer(layer) || !mask.contains_layer(neighbor.layer()) {
                    return true;
                }
                let edge = EdgeView {
                    graph: self,
                    dense,
                    id: EdgeId(load(&self.0.edge_ids, dense)),
                    source: if outgoing { node.id() } else { neighbor.id() },
                    target: if outgoing { neighbor.id() } else { node.id() },
                    kind: RelationshipTypeId(load(&self.0.edge_types, dense)),
                    layer,
                    stamp,
                    active,
                    stamp_cell,
                };
                if !edge.current() || !node.current() || !neighbor.current() {
                    return false;
                }
                match visit(edge, neighbor) {
                    Ok(()) => true,
                    Err(error) => {
                        failure = Some(error);
                        false
                    }
                }
            },
        );
        failure.map_or(Ok(true), Err)
    }

    fn walk_neighbor_denses_until<'a>(
        &'a self,
        node: NodeView<'a>,
        outgoing: bool,
        mask: LayerMask,
        mut token: u64,
        mut visit: impl FnMut(NodeView<'a>, u32, u64, &'a AtomicBool, &'a AtomicU64, Layer) -> bool,
    ) {
        if !mask.contains_layer(node.layer()) {
            return;
        }
        let links = if outgoing {
            &self.0.next_out
        } else {
            &self.0.next_in
        };
        let mut previous_stamp = u64::MAX;
        // Pages have stable addresses. Borrow only those addresses; reload every field
        // and lifetime fence even when the next relationship occupies the same page.
        let mut pages = None;
        let mut borrowed_page = usize::MAX;
        while token != NO_EDGE && node.current() {
            let Ok(dense) = u32::try_from(token) else {
                break;
            };
            let page_index = dense as usize / SEGMENT;
            if borrowed_page != page_index {
                pages = (|| {
                    Some((
                        self.0.edge_stamps.directory.borrow_page(page_index)?,
                        self.0.edge_active.directory.borrow_page(page_index)?,
                        self.0.edge_sources.directory.borrow_page(page_index)?,
                        self.0.edge_targets.directory.borrow_page(page_index)?,
                        self.0.edge_layers.directory.borrow_page(page_index)?,
                        links.directory.borrow_page(page_index)?,
                    ))
                })();
                borrowed_page = page_index;
            }
            let Some((stamps, actives, sources, targets, layers, nexts)) = pages else {
                break;
            };
            let offset = dense as usize % SEGMENT;
            let (
                Some(stamp_cell),
                Some(active),
                Some(source),
                Some(target),
                Some(edge_layer),
                Some(next),
            ) = (
                stamps[offset].get(),
                actives[offset].get(),
                sources[offset].get(),
                targets[offset].get(),
                layers[offset].get(),
                nexts[offset].get(),
            )
            else {
                break;
            };
            let stamp = stamp_cell.load(Ordering::Acquire);
            if stamp >= previous_stamp || !active.load(Ordering::Acquire) {
                break;
            }
            let Ok(source) = u32::try_from(source.load(Ordering::Acquire)) else {
                break;
            };
            let Ok(target) = u32::try_from(target.load(Ordering::Acquire)) else {
                break;
            };
            let (owner, other_dense) = if outgoing {
                (source, target)
            } else {
                (target, source)
            };
            if owner != node.dense() {
                break;
            }
            let Some(other) = self.node_dense(other_dense) else {
                break;
            };
            let next = next.load(Ordering::Acquire);
            let edge_layer = layer(edge_layer.load(Ordering::Acquire));
            if !node.current()
                || !other.current()
                || !active.load(Ordering::Acquire)
                || stamp_cell.load(Ordering::Acquire) != stamp
            {
                break;
            }
            if !visit(other, dense, stamp, active, stamp_cell, edge_layer) {
                break;
            }
            previous_stamp = stamp;
            token = next;
        }
    }

    fn walk_adjacency<'a>(
        &'a self,
        node: NodeView<'_>,
        kind: Option<RelationshipTypeId>,
        mask: LayerMask,
        limit: usize,
        outgoing: bool,
        mut visit: impl FnMut(EdgeView<'a>, NodeView<'a>),
    ) {
        if !mask.contains_layer(node.layer()) || limit == 0 {
            return;
        }
        let id = node.id();
        let heads = if outgoing {
            &self.0.outgoing
        } else {
            &self.0.incoming
        };
        let links = if outgoing {
            &self.0.next_out
        } else {
            &self.0.next_in
        };
        let mut token = heads
            .get(node.dense as usize)
            .map_or(NO_EDGE, |head| head.load(Ordering::Acquire));
        let mut visited = 0;
        let mut previous_stamp = u64::MAX;
        while token != NO_EDGE && node.current() {
            let Ok(dense) = u32::try_from(token) else {
                break;
            };
            let Some(edge) = self.edge_dense(dense) else {
                break;
            };
            if edge.stamp >= previous_stamp
                || (if outgoing {
                    edge.source()
                } else {
                    edge.target()
                }) != id
            {
                break;
            }
            let next = links
                .get(edge.dense as usize)
                .map_or(NO_EDGE, |link| link.load(Ordering::Acquire));
            if !edge.current() {
                break;
            }
            let other_id = if outgoing {
                edge.target()
            } else {
                edge.source()
            };
            let other_dense = if outgoing {
                load(&self.0.edge_targets, edge.dense)
            } else {
                load(&self.0.edge_sources, edge.dense)
            } as u32;
            if let Some(other) = self.node_dense(other_dense) {
                // Recycled edge/endpoint slots must still match the identities captured above.
                if !edge.current() || other.id() != other_id {
                    break;
                }
                if mask.contains_layer(edge.layer())
                    && mask.contains_layer(other.layer())
                    && kind.is_none_or(|kind| kind == edge.relationship_type())
                {
                    visit(edge, other);
                    visited += 1;
                    if visited == limit {
                        break;
                    }
                }
            }
            previous_stamp = edge.stamp;
            token = next;
        }
    }
    pub fn expand_out(
        &self,
        node: NodeId,
        kind: Option<RelationshipTypeId>,
        mask: LayerMask,
    ) -> Result<Vec<(EdgeView<'_>, NodeView<'_>)>> {
        self.expand(node, kind, mask, usize::MAX, true)
    }
    pub fn expand_in(
        &self,
        node: NodeId,
        kind: Option<RelationshipTypeId>,
        mask: LayerMask,
    ) -> Result<Vec<(EdgeView<'_>, NodeView<'_>)>> {
        self.expand(node, kind, mask, usize::MAX, false)
    }
    pub fn expand_out_bounded(
        &self,
        node: NodeId,
        kind: Option<RelationshipTypeId>,
        mask: LayerMask,
        limit: usize,
    ) -> Result<Vec<(EdgeView<'_>, NodeView<'_>)>> {
        self.expand(node, kind, mask, limit, true)
    }
    pub fn expand_in_bounded(
        &self,
        node: NodeId,
        kind: Option<RelationshipTypeId>,
        mask: LayerMask,
        limit: usize,
    ) -> Result<Vec<(EdgeView<'_>, NodeView<'_>)>> {
        self.expand(node, kind, mask, limit, false)
    }
    pub fn incident_edge_ids(&self, node: NodeId) -> Result<Vec<EdgeId>> {
        let mut edges = self
            .expand_out(node, None, LayerMask::ALL)?
            .into_iter()
            .chain(self.expand_in(node, None, LayerMask::ALL)?)
            .map(|(edge, _)| edge.id())
            .collect::<Vec<_>>();
        edges.sort_unstable();
        edges.dedup();
        Ok(edges)
    }
    pub fn compact_adjacency(&self) -> Result<()> {
        // Deletion unlinks each relationship before releasing its ordinal; no historical chain remains.
        for node in self.nodes() {
            self.expand_out(node.id(), None, LayerMask::ALL)?;
            self.expand_in(node.id(), None, LayerMask::ALL)?;
        }
        Ok(())
    }
    pub fn compact(&self) -> Result<CompactionMap> {
        self.compact_adjacency()?;
        Ok(CompactionMap {
            nodes: self.nodes().map(|node| (node.id(), node.dense())).collect(),
            edges: self.edges().map(|edge| (edge.id(), edge.dense())).collect(),
        })
    }
    fn recycle_imported_tombstones(&self) {
        for row in 0..self.node_slot_count() {
            if !self.node_active(row as u32) {
                self.0
                    .node_lookup
                    .pin()
                    .remove(&load(&self.0.node_ids, row as u32));
                self.0.node_free.release(row as u32);
            }
        }
        for row in 0..self.edge_slot_count() {
            if !self.edge_active(row as u32) {
                self.0
                    .edge_lookup
                    .pin()
                    .remove(&load(&self.0.edge_ids, row as u32));
                self.0.edge_free.release(row as u32);
            }
        }
    }
    pub fn validate_structure(&self) -> Result<()> {
        let nodes = self.node_slot_count();
        let edges = self.edge_slot_count();
        if [
            self.0.node_layers.len(),
            self.0.node_revisions.len(),
            self.0.labels.len(),
            self.0.node_active.len(),
            self.0.node_stamps.len(),
            self.0.incoming.len(),
            self.0.outgoing.len(),
        ]
        .iter()
        .any(|len| *len != nodes)
            || [
                self.0.edge_sources.len(),
                self.0.edge_targets.len(),
                self.0.edge_types.len(),
                self.0.edge_layers.len(),
                self.0.edge_revisions.len(),
                self.0.edge_active.len(),
                self.0.edge_stamps.len(),
                self.0.next_out.len(),
                self.0.next_in.len(),
            ]
            .iter()
            .any(|len| *len != edges)
        {
            return Err(Error::invalid_data("canonical column cardinalities differ"));
        }
        for node in self.nodes() {
            if self.0.node_lookup.pin().get(&node.id().0) != Some(&node.dense)
                || node
                    .labels()
                    .iter()
                    .any(|label| self.catalog().label_name(*label).is_none())
            {
                return Err(Error::invalid_data(
                    "canonical node identity/schema invalid",
                ));
            }
        }
        for row in 0..edges {
            if self.edge_active(row as u32)
                && (self.edge_dense(row as u32).is_none()
                    || self
                        .catalog()
                        .relationship_type_name(RelationshipTypeId(load(
                            &self.0.edge_types,
                            row as u32,
                        )))
                        .is_none())
            {
                return Err(Error::invalid_data(
                    "canonical relationship endpoint/schema invalid",
                ));
            }
        }
        Ok(())
    }
    pub fn quarantine_invalid_relationships(&self) -> Result<usize> {
        let mut rejected = self.0.recovery_rejected.swap(0, Ordering::AcqRel);
        for row in 0..self.edge_slot_count() {
            if self.edge_active(row as u32) && self.edge_dense(row as u32).is_none() {
                if let Some(cell) = self.0.edge_active.get(row) {
                    cell.store(false, Ordering::Release);
                }
                let id = EdgeId(load(&self.0.edge_ids, row as u32));
                self.unlink_edge(
                    row as u32,
                    load(&self.0.edge_sources, row as u32) as u32,
                    load(&self.0.edge_targets, row as u32) as u32,
                );
                self.0.edge_lookup.pin().remove(&id.0);
                self.0.edge_properties.clear(row as u32);
                self.0.edge_free.release(row as u32);
                let layer = layer(load(&self.0.edge_layers, row as u32));
                self.0.edge_counts.adjust(layer, None, false);
                self.0
                    .edge_counts
                    .adjust(layer, Some(load(&self.0.edge_types, row as u32)), false);
                rejected += 1;
            }
        }
        self.compact_adjacency()?;
        Ok(rejected)
    }
    pub fn resident_bytes(&self) -> usize {
        [
            &self.0.node_ids,
            &self.0.node_layers,
            &self.0.node_revisions,
            &self.0.edge_ids,
            &self.0.edge_sources,
            &self.0.edge_targets,
            &self.0.edge_types,
            &self.0.edge_layers,
            &self.0.edge_revisions,
        ]
        .into_iter()
        .map(Segments::allocated_bytes)
        .sum::<usize>()
            + self.0.node_stamps.allocated_bytes()
            + self.0.node_writers.allocated_bytes()
            + self.0.edge_writers.allocated_bytes()
            + self.0.node_free.links.allocated_bytes()
            + self.0.edge_stamps.allocated_bytes()
            + self.0.edge_free.links.allocated_bytes()
            + self.0.node_active.allocated_bytes()
            + self.0.edge_active.allocated_bytes()
            + self.0.labels.allocated_bytes()
            + self.0.outgoing.allocated_bytes()
            + self.0.incoming.allocated_bytes()
            + self.0.next_out.allocated_bytes()
            + self.0.next_in.allocated_bytes()
            + self.0.label_bytes.load(Ordering::Acquire)
            + self.0.node_properties.bytes.load(Ordering::Acquire)
            + self.0.edge_properties.bytes.load(Ordering::Acquire)
    }
    pub fn node_property_accepts(&self, property: PropertyId, value: &ScalarValue) -> Option<bool> {
        self.0.node_properties.accepts(property, value)
    }
    pub fn node_property_column(&self, p: PropertyId) -> Option<Arc<ConcurrentPropertyColumn>> {
        self.0.node_properties.column(p, false)
    }
    pub fn node_property_reader(&self, p: PropertyId) -> Option<NodePropertyReader<'_>> {
        self.node_property_column(p)
            .map(|column| NodePropertyReader {
                graph: self,
                column,
            })
    }
    pub fn edge_property_column(&self, p: PropertyId) -> Option<Arc<ConcurrentPropertyColumn>> {
        self.0.edge_properties.column(p, false)
    }
    pub fn node_property_is_integer(&self, p: PropertyId) -> bool {
        self.0.node_properties.has_type(p, 2)
    }
    pub fn node_property_is_boolean(&self, p: PropertyId) -> bool {
        self.0.node_properties.has_type(p, 1)
    }
    pub fn node_property_is_string(&self, p: PropertyId) -> bool {
        self.0.node_properties.has_type(p, 4)
    }
    pub fn node_property_is_float(&self, p: PropertyId) -> bool {
        self.0.node_properties.has_type(p, 3)
    }
    pub fn node_property_is_date(&self, p: PropertyId) -> bool {
        self.0.node_properties.has_type(p, 6)
    }
    pub fn node_property_is_local_time(&self, p: PropertyId) -> bool {
        self.0.node_properties.has_type(p, 7)
    }
    pub fn node_property_is_zoned_time(&self, p: PropertyId) -> bool {
        self.0.node_properties.has_type(p, 8)
    }
    pub fn node_property_is_local_datetime(&self, p: PropertyId) -> bool {
        self.0.node_properties.has_type(p, 9)
    }
    pub fn node_property_is_zoned_datetime(&self, p: PropertyId) -> bool {
        self.0.node_properties.has_type(p, 10)
    }
}

fn value_bytes(value: &ScalarValue) -> usize {
    size_of::<ScalarValue>()
        + match value {
            ScalarValue::String(value) => value.len(),
            ScalarValue::Bytes(value) => value.len(),
            ScalarValue::List(value) => value.as_bytes().len(),
            ScalarValue::Map(value) => value.as_bytes().len(),
            ScalarValue::ZonedDateTime { timezone, .. } => timezone.len(),
            _ => 0,
        }
}

#[derive(Serialize, Deserialize)]
struct NodeWire {
    input: StoredNodeInput,
    active: bool,
}
#[derive(Serialize, Deserialize)]
struct EdgeWire {
    input: StoredEdgeInput,
    source: u32,
    target: u32,
    active: bool,
}
#[derive(Serialize, Deserialize)]
struct StoredNodeInput {
    id: NodeId,
    layer: Layer,
    revision: u64,
    labels: Vec<LabelId>,
    #[serde(with = "wire_properties")]
    properties: Vec<(PropertyId, ScalarValue)>,
}
#[derive(Serialize, Deserialize)]
struct StoredEdgeInput {
    id: EdgeId,
    source: NodeId,
    target: NodeId,
    relationship_type: RelationshipTypeId,
    layer: Layer,
    revision: u64,
    #[serde(with = "wire_properties")]
    properties: Vec<(PropertyId, ScalarValue)>,
}
impl From<StoredNodeInput> for NodeInput {
    fn from(value: StoredNodeInput) -> Self {
        Self {
            id: value.id,
            layer: value.layer,
            revision: value.revision,
            labels: value.labels,
            properties: value.properties,
        }
    }
}
impl From<StoredEdgeInput> for EdgeInput {
    fn from(value: StoredEdgeInput) -> Self {
        Self {
            id: value.id,
            source: value.source,
            target: value.target,
            relationship_type: value.relationship_type,
            layer: value.layer,
            revision: value.revision,
            properties: value.properties,
        }
    }
}

mod wire_properties {
    use super::*;
    struct ValueRef<'a>(&'a ScalarValue);
    impl Serialize for ValueRef<'_> {
        fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
            let bytes = super::super::columns::mixed_payload(self.0)
                .map_err(serde::ser::Error::custom)?
                .unwrap_or_default();
            s.serialize_bytes(&bytes)
        }
    }
    struct OwnedValue(ScalarValue);
    impl<'de> Deserialize<'de> for OwnedValue {
        fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
            struct BytesVisitor;
            impl<'de> Visitor<'de> for BytesVisitor {
                type Value = OwnedValue;
                fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                    f.write_str("canonical scalar bytes")
                }
                fn visit_bytes<E: serde::de::Error>(
                    self,
                    bytes: &[u8],
                ) -> std::result::Result<OwnedValue, E> {
                    if bytes.is_empty() {
                        return Ok(OwnedValue(ScalarValue::Null));
                    }
                    super::super::columns::mixed_value(bytes)
                        .map(OwnedValue)
                        .map_err(E::custom)
                }
                fn visit_byte_buf<E: serde::de::Error>(
                    self,
                    bytes: Vec<u8>,
                ) -> std::result::Result<OwnedValue, E> {
                    self.visit_bytes(&bytes)
                }
            }
            d.deserialize_byte_buf(BytesVisitor)
        }
    }
    pub fn serialize<S: serde::Serializer>(
        values: &[(PropertyId, ScalarValue)],
        s: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        let mut seq = s.serialize_seq(Some(values.len()))?;
        for (property, value) in values {
            seq.serialize_element(&(*property, ValueRef(value)))?;
        }
        seq.end()
    }
    pub fn deserialize<'de, D: serde::Deserializer<'de>>(
        d: D,
    ) -> std::result::Result<Vec<(PropertyId, ScalarValue)>, D::Error> {
        Ok(Vec::<(PropertyId, OwnedValue)>::deserialize(d)?
            .into_iter()
            .map(|(property, value)| (property, value.0))
            .collect())
    }
}
struct Rows<'a> {
    graph: &'a GraphStore,
    nodes: bool,
}
impl Serialize for Rows<'_> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        let count = if self.nodes {
            self.graph.node_slot_count()
        } else {
            self.graph.edge_slot_count()
        };
        let mut seq = s.serialize_seq(Some(count))?;
        for row in 0..count {
            let row = row as u32;
            if self.nodes {
                seq.serialize_element(&NodeWire {
                    input: StoredNodeInput {
                        id: NodeId(load(&self.graph.0.node_ids, row)),
                        layer: layer(load(&self.graph.0.node_layers, row)),
                        revision: load(&self.graph.0.node_revisions, row),
                        labels: self.graph.labels(row).to_vec(),
                        properties: self.graph.0.node_properties.values(row),
                    },
                    active: self.graph.node_active(row),
                })?;
            } else {
                let source = load(&self.graph.0.edge_sources, row) as u32;
                let target = load(&self.graph.0.edge_targets, row) as u32;
                seq.serialize_element(&EdgeWire {
                    input: StoredEdgeInput {
                        id: EdgeId(load(&self.graph.0.edge_ids, row)),
                        source: NodeId(load(&self.graph.0.node_ids, source)),
                        target: NodeId(load(&self.graph.0.node_ids, target)),
                        relationship_type: RelationshipTypeId(load(&self.graph.0.edge_types, row)),
                        layer: layer(load(&self.graph.0.edge_layers, row)),
                        revision: load(&self.graph.0.edge_revisions, row),
                        properties: self.graph.0.edge_properties.values(row),
                    },
                    source,
                    target,
                    active: self.graph.edge_active(row),
                })?;
            }
        }
        seq.end()
    }
}
impl Serialize for GraphStore {
    /// Caller pauses mutation apply at the committed WAL bookmark before encoding.
    fn serialize<S: serde::Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        let mut state = s.serialize_struct("CanonicalCpuGraph", 6)?;
        state.serialize_field("format", &1_u32)?;
        state.serialize_field("catalog", self.catalog())?;
        state.serialize_field(
            "nodes",
            &Rows {
                graph: self,
                nodes: true,
            },
        )?;
        state.serialize_field(
            "edges",
            &Rows {
                graph: self,
                nodes: false,
            },
        )?;
        state.serialize_field("revision", &self.revision())?;
        state.serialize_field("layout_version", &self.layout_version())?;
        state.end()
    }
}
struct RowsSeed<'a> {
    graph: &'a GraphStore,
    nodes: bool,
}
impl<'de> DeserializeSeed<'de> for RowsSeed<'_> {
    type Value = ();
    fn deserialize<D: serde::Deserializer<'de>>(self, d: D) -> std::result::Result<(), D::Error> {
        d.deserialize_seq(self)
    }
}
impl<'de> Visitor<'de> for RowsSeed<'_> {
    type Value = ();
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("canonical rows")
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> std::result::Result<(), A::Error> {
        if self.nodes {
            while let Some(wire) = seq.next_element::<NodeWire>()? {
                if self.graph.contains_node_id(wire.input.id) {
                    return Err(serde::de::Error::custom("duplicate canonical node ID"));
                }
                self.graph
                    .append_node(wire.input.into(), wire.active)
                    .map_err(serde::de::Error::custom)?;
            }
        } else {
            while let Some(wire) = seq.next_element::<EdgeWire>()? {
                if self.graph.contains_edge_id(wire.input.id) {
                    return Err(serde::de::Error::custom(
                        "duplicate canonical relationship ID",
                    ));
                }
                let active = wire.active
                    && self.graph.node_dense(wire.source).is_some()
                    && self.graph.node_dense(wire.target).is_some();
                if wire.active && !active {
                    self.graph
                        .0
                        .recovery_rejected
                        .fetch_add(1, Ordering::AcqRel);
                }
                self.graph
                    .append_edge(wire.input.into(), wire.source, wire.target, active)
                    .map_err(serde::de::Error::custom)?;
            }
        }
        Ok(())
    }
}

struct AtomicColumnSeed<'a> {
    column: &'a Segments<AtomicU64>,
    layers: bool,
}
impl<'de> DeserializeSeed<'de> for AtomicColumnSeed<'_> {
    type Value = ();
    fn deserialize<D: serde::Deserializer<'de>>(self, d: D) -> std::result::Result<(), D::Error> {
        d.deserialize_seq(self)
    }
}
impl<'de> Visitor<'de> for AtomicColumnSeed<'_> {
    type Value = ();
    fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("canonical primitive column")
    }
    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> std::result::Result<(), A::Error> {
        if self.layers {
            while let Some(value) = seq.next_element::<Layer>()? {
                self.column
                    .push(AtomicU64::new(value as u64))
                    .map_err(serde::de::Error::custom)?;
            }
        } else {
            while let Some(value) = seq.next_element::<u64>()? {
                self.column
                    .push(AtomicU64::new(value))
                    .map_err(serde::de::Error::custom)?;
            }
        }
        Ok(())
    }
}

impl GraphStore {
    fn import_properties(&self, old: super::PropertyColumns, nodes: bool) -> Result<()> {
        let rows = if nodes {
            self.node_slot_count()
        } else {
            self.edge_slot_count()
        };
        if old.rows() != rows {
            return Err(Error::invalid_data("legacy property row count differs"));
        }
        let properties = if nodes {
            &self.0.node_properties
        } else {
            &self.0.edge_properties
        };
        // The legacy dictionary follows its columns on disk, so this one legacy field must be
        // decoded before conversion. It is dropped here; no LegacyGraphStore is constructed.
        for property in old.property_ids() {
            for row in 0..rows {
                if let Some(value) = old.get(row as u32, property) {
                    properties.set(row as u32, property, value)?;
                }
            }
        }
        Ok(())
    }
    fn finish_legacy_import(&self) -> Result<()> {
        let nodes = self.node_slot_count();
        let edges = self.edge_slot_count();
        for row in 0..nodes {
            self.0.next_node_id.fetch_max(
                load(&self.0.node_ids, row as u32).saturating_add(1),
                Ordering::Relaxed,
            );
            self.0
                .node_writers
                .get_or_insert_with(row, || Mutex::new(()))?;
            self.0.node_free.ensure(row as u32)?;
            if self.0.node_stamps.get(row).is_none() {
                publish_atomic(
                    &self.0.node_stamps,
                    row as u32,
                    next_stamp(&self.0.node_clock)?,
                )?;
            }
        }
        for row in 0..edges {
            self.0.next_edge_id.fetch_max(
                load(&self.0.edge_ids, row as u32).saturating_add(1),
                Ordering::Relaxed,
            );
            self.0
                .edge_writers
                .get_or_insert_with(row, || Mutex::new(()))?;
            self.0.edge_free.ensure(row as u32)?;
            if self.0.edge_stamps.get(row).is_none() {
                publish_atomic(
                    &self.0.edge_stamps,
                    row as u32,
                    next_stamp(&self.0.edge_clock)?,
                )?;
            }
        }
        if self.0.node_layers.len() != nodes
            || self.0.node_revisions.len() != nodes
            || self.0.edge_sources.len() != edges
            || self.0.edge_targets.len() != edges
            || self.0.edge_types.len() != edges
            || self.0.edge_layers.len() != edges
            || self.0.edge_revisions.len() != edges
        {
            return Err(Error::invalid_data(
                "legacy graph column cardinalities differ",
            ));
        }
        for row in 0..nodes {
            if self.node_active(row as u32) {
                let layer = layer(load(&self.0.node_layers, row as u32));
                self.0.node_counts.adjust(layer, None, true);
                for label in self.labels(row as u32).iter() {
                    self.0.node_counts.adjust(layer, Some(label.0), true);
                }
            } else {
                self.0.node_properties.clear(row as u32);
                if let Some(cell) = self.0.labels.get(row) {
                    cell.store(Vec::new(), &self.0.label_bytes);
                }
            }
        }
        for row in 0..edges {
            let source = load(&self.0.edge_sources, row as u32) as u32;
            let target = load(&self.0.edge_targets, row as u32) as u32;
            if self.edge_active(row as u32)
                && (!self.node_active(source) || !self.node_active(target))
            {
                if let Some(cell) = self.0.edge_active.get(row) {
                    cell.store(false, Ordering::Release);
                }
                self.0.edge_properties.clear(row as u32);
                self.0.recovery_rejected.fetch_add(1, Ordering::AcqRel);
            }
            if self.edge_active(row as u32) {
                let outgoing = self
                    .0
                    .outgoing
                    .get(source as usize)
                    .ok_or_else(|| Error::invalid_data("legacy edge source missing"))?;
                let incoming = self
                    .0
                    .incoming
                    .get(target as usize)
                    .ok_or_else(|| Error::invalid_data("legacy edge target missing"))?;
                if let Some(link) = self.0.next_out.get(row) {
                    link.store(outgoing.load(Ordering::Acquire), Ordering::Release);
                }
                if let Some(link) = self.0.next_in.get(row) {
                    link.store(incoming.load(Ordering::Acquire), Ordering::Release);
                }
                outgoing.store(row as u64, Ordering::Release);
                incoming.store(row as u64, Ordering::Release);
                let layer = layer(load(&self.0.edge_layers, row as u32));
                self.0.edge_counts.adjust(layer, None, true);
                self.0
                    .edge_counts
                    .adjust(layer, Some(load(&self.0.edge_types, row as u32)), true);
            } else {
                self.0.edge_properties.clear(row as u32);
            }
        }
        self.recycle_imported_tombstones();
        self.validate_structure()
    }
}

impl<'de> Deserialize<'de> for GraphStore {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct GraphVisitor;
        impl<'de> Visitor<'de> for GraphVisitor {
            type Value = GraphStore;
            fn expecting(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
                f.write_str("canonical CPU graph format 1")
            }
            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> std::result::Result<GraphStore, A::Error> {
                let graph = GraphStore::default();
                if seq.next_element::<u32>()? != Some(1) {
                    return Err(serde::de::Error::custom(
                        "unsupported canonical graph format",
                    ));
                }
                let catalog = seq
                    .next_element::<NameCatalog>()?
                    .ok_or_else(|| serde::de::Error::custom("missing catalog"))?;
                graph
                    .catalog()
                    .import(&catalog)
                    .map_err(serde::de::Error::custom)?;
                seq.next_element_seed(RowsSeed {
                    graph: &graph,
                    nodes: true,
                })?
                .ok_or_else(|| serde::de::Error::custom("missing node rows"))?;
                seq.next_element_seed(RowsSeed {
                    graph: &graph,
                    nodes: false,
                })?
                .ok_or_else(|| serde::de::Error::custom("missing edge rows"))?;
                let revision = seq
                    .next_element::<u64>()?
                    .ok_or_else(|| serde::de::Error::custom("missing graph revision"))?;
                if revision < graph.revision() {
                    return Err(serde::de::Error::custom(
                        "graph revision precedes row revision",
                    ));
                }
                graph.0.revision.store(revision, Ordering::Release);
                graph.0.layout.store(
                    seq.next_element::<u64>()?
                        .ok_or_else(|| serde::de::Error::custom("missing layout epoch"))?,
                    Ordering::Release,
                );
                graph.recycle_imported_tombstones();
                graph
                    .validate_structure()
                    .map_err(serde::de::Error::custom)?;
                Ok(graph)
            }
            fn visit_map<A: MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<GraphStore, A::Error> {
                let graph = GraphStore::default();
                let mut fields = BTreeSet::new();
                let mut revision = None;
                while let Some(key) = map.next_key::<String>()? {
                    if !fields.insert(key.clone()) {
                        return Err(serde::de::Error::custom("duplicate graph field"));
                    }
                    match key.as_str() {
                        "format" => {
                            if map.next_value::<u32>()? != 1 {
                                return Err(serde::de::Error::custom(
                                    "unsupported canonical graph format",
                                ));
                            }
                        }
                        "catalog" => {
                            if fields.contains("format") {
                                let catalog = map.next_value::<NameCatalog>()?;
                                graph
                                    .catalog()
                                    .import(&catalog)
                                    .map_err(serde::de::Error::custom)?;
                            } else {
                                let catalog =
                                    map.next_value::<super::store::LegacyNameCatalog>()?;
                                for (id, name) in catalog.labels() {
                                    graph
                                        .catalog()
                                        .declare_label(name.to_owned(), id)
                                        .map_err(serde::de::Error::custom)?;
                                }
                                for (id, name) in catalog.properties() {
                                    graph
                                        .catalog()
                                        .declare_property(name.to_owned(), id)
                                        .map_err(serde::de::Error::custom)?;
                                }
                                for (id, name) in catalog.relationship_types() {
                                    graph
                                        .catalog()
                                        .declare_relationship_type(name.to_owned(), id)
                                        .map_err(serde::de::Error::custom)?;
                                }
                                graph
                                    .catalog()
                                    .0
                                    .labels
                                    .next
                                    .fetch_max(catalog.next_label_id(), Ordering::AcqRel);
                                graph
                                    .catalog()
                                    .0
                                    .properties
                                    .next
                                    .fetch_max(catalog.next_property_id(), Ordering::AcqRel);
                                graph.catalog().0.relationships.next.fetch_max(
                                    catalog.next_relationship_type_id(),
                                    Ordering::AcqRel,
                                );
                            }
                        }
                        "nodes" => {
                            map.next_value_seed(RowsSeed {
                                graph: &graph,
                                nodes: true,
                            })?;
                        }
                        "edges" => {
                            if !fields.contains("nodes") {
                                return Err(serde::de::Error::custom(
                                    "canonical edges must follow node columns",
                                ));
                            }
                            map.next_value_seed(RowsSeed {
                                graph: &graph,
                                nodes: false,
                            })?;
                        }
                        "revision" => {
                            revision = Some(map.next_value::<u64>()?);
                        }
                        "layout_version" => {
                            graph.0.layout.store(map.next_value()?, Ordering::Release);
                        }
                        "node_ids" => {
                            map.next_value_seed(AtomicColumnSeed {
                                column: &graph.0.node_ids,
                                layers: false,
                            })?;
                            for row in 0..graph.node_slot_count() {
                                let id = load(&graph.0.node_ids, row as u32);
                                if graph.0.node_lookup.pin().insert(id, row as u32).is_some() {
                                    return Err(serde::de::Error::custom(
                                        "duplicate legacy node ID",
                                    ));
                                }
                                graph
                                    .0
                                    .node_active
                                    .push(AtomicBool::new(true))
                                    .map_err(serde::de::Error::custom)?;
                                graph
                                    .0
                                    .labels
                                    .push(LabelsCell::default())
                                    .map_err(serde::de::Error::custom)?;
                                graph
                                    .0
                                    .outgoing
                                    .push(AtomicU64::new(NO_EDGE))
                                    .map_err(serde::de::Error::custom)?;
                                graph
                                    .0
                                    .incoming
                                    .push(AtomicU64::new(NO_EDGE))
                                    .map_err(serde::de::Error::custom)?;
                            }
                        }
                        "node_layers" => {
                            map.next_value_seed(AtomicColumnSeed {
                                column: &graph.0.node_layers,
                                layers: true,
                            })?;
                        }
                        "node_revisions" => {
                            map.next_value_seed(AtomicColumnSeed {
                                column: &graph.0.node_revisions,
                                layers: false,
                            })?;
                        }
                        "node_labels" => {
                            let labels = map.next_value::<super::PackedLists<LabelId>>()?;
                            if labels.rows() > graph.node_slot_count() {
                                return Err(serde::de::Error::custom(
                                    "legacy labels exceed node rows",
                                ));
                            }
                            for row in 0..labels.rows() {
                                if let Some(cell) = graph.0.labels.get(row) {
                                    cell.store(
                                        labels.get(row as u32).unwrap_or(&[]).to_vec(),
                                        &graph.0.label_bytes,
                                    );
                                }
                            }
                        }
                        "node_label_deltas" => {
                            let labels = map.next_value::<PersistentMap<Arc<Vec<LabelId>>>>()?;
                            for (row, values) in labels.iter() {
                                let cell = graph
                                    .0
                                    .labels
                                    .get(usize::try_from(row).map_err(serde::de::Error::custom)?)
                                    .ok_or_else(|| {
                                        serde::de::Error::custom("legacy label delta row missing")
                                    })?;
                                cell.store(values.as_ref().clone(), &graph.0.label_bytes);
                            }
                        }
                        "node_properties" => {
                            graph
                                .import_properties(map.next_value()?, true)
                                .map_err(serde::de::Error::custom)?;
                        }
                        "node_deleted" => {
                            let deleted =
                                map.next_value::<bitvec::vec::BitVec<u64, bitvec::order::Lsb0>>()?;
                            for row in deleted.iter_ones() {
                                if let Some(cell) = graph.0.node_active.get(row) {
                                    cell.store(false, Ordering::Release);
                                }
                            }
                        }
                        "node_tombstones" => {
                            let deleted = map.next_value::<PersistentMap<()>>()?;
                            for (row, _) in deleted.iter() {
                                let cell = graph
                                    .0
                                    .node_active
                                    .get(usize::try_from(row).map_err(serde::de::Error::custom)?)
                                    .ok_or_else(|| {
                                        serde::de::Error::custom("legacy tombstone row missing")
                                    })?;
                                cell.store(false, Ordering::Release);
                            }
                        }
                        "edge_ids" => {
                            map.next_value_seed(AtomicColumnSeed {
                                column: &graph.0.edge_ids,
                                layers: false,
                            })?;
                            for row in 0..graph.edge_slot_count() {
                                let id = load(&graph.0.edge_ids, row as u32);
                                if graph.0.edge_lookup.pin().insert(id, row as u32).is_some() {
                                    return Err(serde::de::Error::custom(
                                        "duplicate legacy edge ID",
                                    ));
                                }
                                graph
                                    .0
                                    .edge_active
                                    .push(AtomicBool::new(true))
                                    .map_err(serde::de::Error::custom)?;
                                graph
                                    .0
                                    .next_out
                                    .push(AtomicU64::new(NO_EDGE))
                                    .map_err(serde::de::Error::custom)?;
                                graph
                                    .0
                                    .next_in
                                    .push(AtomicU64::new(NO_EDGE))
                                    .map_err(serde::de::Error::custom)?;
                            }
                        }
                        "edge_sources" => {
                            map.next_value_seed(AtomicColumnSeed {
                                column: &graph.0.edge_sources,
                                layers: false,
                            })?;
                        }
                        "edge_targets" => {
                            map.next_value_seed(AtomicColumnSeed {
                                column: &graph.0.edge_targets,
                                layers: false,
                            })?;
                        }
                        "edge_types" => {
                            map.next_value_seed(AtomicColumnSeed {
                                column: &graph.0.edge_types,
                                layers: false,
                            })?;
                        }
                        "edge_layers" => {
                            map.next_value_seed(AtomicColumnSeed {
                                column: &graph.0.edge_layers,
                                layers: true,
                            })?;
                        }
                        "edge_revisions" => {
                            map.next_value_seed(AtomicColumnSeed {
                                column: &graph.0.edge_revisions,
                                layers: false,
                            })?;
                        }
                        "edge_properties" => {
                            graph
                                .import_properties(map.next_value()?, false)
                                .map_err(serde::de::Error::custom)?;
                        }
                        "edge_deleted" => {
                            let deleted =
                                map.next_value::<bitvec::vec::BitVec<u64, bitvec::order::Lsb0>>()?;
                            for row in deleted.iter_ones() {
                                if let Some(cell) = graph.0.edge_active.get(row) {
                                    cell.store(false, Ordering::Release);
                                }
                            }
                        }
                        "edge_tombstones" => {
                            let deleted = map.next_value::<PersistentMap<()>>()?;
                            for (row, _) in deleted.iter() {
                                let cell = graph
                                    .0
                                    .edge_active
                                    .get(usize::try_from(row).map_err(serde::de::Error::custom)?)
                                    .ok_or_else(|| {
                                        serde::de::Error::custom(
                                            "legacy edge tombstone row missing",
                                        )
                                    })?;
                                cell.store(false, Ordering::Release);
                            }
                        }
                        _ => {
                            let _: serde::de::IgnoredAny = map.next_value()?;
                        }
                    }
                }
                if !fields.contains("format") {
                    if [
                        "catalog",
                        "node_ids",
                        "node_layers",
                        "node_revisions",
                        "edge_ids",
                        "edge_sources",
                        "edge_targets",
                        "edge_types",
                        "edge_layers",
                        "edge_revisions",
                        "revision",
                    ]
                    .iter()
                    .any(|name| !fields.contains(*name))
                    {
                        return Err(serde::de::Error::custom("missing legacy graph field"));
                    }
                    graph
                        .finish_legacy_import()
                        .map_err(serde::de::Error::custom)?;
                } else if ["format", "catalog", "nodes", "edges", "revision"]
                    .iter()
                    .any(|name| !fields.contains(*name))
                {
                    return Err(serde::de::Error::custom("missing canonical graph field"));
                }
                let revision =
                    revision.ok_or_else(|| serde::de::Error::custom("missing revision"))?;
                if revision < graph.revision() {
                    return Err(serde::de::Error::custom(
                        "graph revision precedes row revision",
                    ));
                }
                graph.0.revision.store(revision, Ordering::Release);
                if fields.contains("format") {
                    graph.recycle_imported_tombstones();
                }
                graph
                    .validate_structure()
                    .map_err(serde::de::Error::custom)?;
                Ok(graph)
            }
        }
        d.deserialize_struct(
            "CanonicalCpuGraph",
            &[
                "format",
                "catalog",
                "nodes",
                "edges",
                "revision",
                "layout_version",
            ],
            GraphVisitor,
        )
    }
}

#[cfg(test)]
#[path = "concurrent_adjacency_tests.rs"]
mod concurrent_adjacency_tests;

#[cfg(test)]
#[path = "concurrent_property_tests.rs"]
mod concurrent_property_tests;

#[cfg(test)]
mod tests {
    use super::*;
    use std::{
        sync::{Barrier, mpsc},
        thread,
        time::Duration,
    };

    fn fixture() -> Result<(GraphStore, LabelId, PropertyId, RelationshipTypeId)> {
        let graph = GraphStore::default();
        let label = graph.catalog().intern_label("Record")?;
        let property = graph.catalog().intern_property("body")?;
        let kind = graph.catalog().intern_relationship_type("LINK")?;
        for id in 1..=2 {
            graph.insert_node(NodeInput {
                id: NodeId(id),
                layer: Layer::Knowledge,
                revision: 1,
                labels: vec![label],
                properties: vec![(property, ScalarValue::Integer(id as i64))],
            })?;
        }
        graph.insert_edge(EdgeInput {
            id: EdgeId(1),
            source: NodeId(1),
            target: NodeId(2),
            relationship_type: kind,
            layer: Layer::Knowledge,
            revision: 1,
            properties: Vec::new(),
        })?;
        Ok((graph, label, property, kind))
    }

    #[test]
    fn parallel_identity_reservations_share_cursors_without_graph_rows() -> Result<()> {
        let graph = GraphStore::default();
        let barrier = Barrier::new(16);
        let identities = thread::scope(|scope| -> Result<Vec<(NodeId, EdgeId)>> {
            let readers = (0..16)
                .map(|_| {
                    let graph = &graph;
                    let barrier = &barrier;
                    scope.spawn(move || -> Result<Vec<(NodeId, EdgeId)>> {
                        barrier.wait();
                        (0..256)
                            .map(|_| Ok((graph.reserve_node_id(100)?, graph.reserve_edge_id(200)?)))
                            .collect()
                    })
                })
                .collect::<Vec<_>>();
            let mut identities = Vec::new();
            for reader in readers {
                identities.extend(
                    reader
                        .join()
                        .map_err(|_| Error::internal("identity reservation panicked"))??,
                );
            }
            Ok(identities)
        })?;
        assert_eq!(
            identities
                .iter()
                .map(|(node, _)| *node)
                .collect::<BTreeSet<_>>()
                .len(),
            4096
        );
        assert_eq!(
            identities
                .iter()
                .map(|(_, edge)| *edge)
                .collect::<BTreeSet<_>>()
                .len(),
            4096
        );
        assert_eq!(graph.node_slot_count(), 0);
        assert_eq!(graph.edge_slot_count(), 0);
        assert_eq!(graph.reserve_node_id(10_000)?, NodeId(10_000));
        assert_eq!(graph.clone().reserve_node_id(1)?, NodeId(10_001));
        assert_eq!(graph.reserve_edge_id(20_000)?, EdgeId(20_000));
        assert_eq!(graph.clone().reserve_edge_id(1)?, EdgeId(20_001));
        assert!(graph.reserve_node_id(u64::MAX).is_err());
        assert!(graph.reserve_edge_id(u64::MAX).is_err());
        Ok(())
    }

    #[test]
    fn restored_identity_cursors_cover_inserted_and_deleted_rows() -> Result<()> {
        let (graph, _, _, _) = fixture()?;
        graph.delete_edge(EdgeId(1), 2)?;
        graph.delete_node(NodeId(2), false, 2)?;
        let mut encoded = Vec::new();
        ciborium::into_writer(&graph, &mut encoded)
            .map_err(|error| Error::internal(error.to_string()))?;
        let restored: GraphStore = ciborium::from_reader(encoded.as_slice())
            .map_err(|error| Error::internal(error.to_string()))?;
        assert_eq!(restored.reserve_node_id(1)?, NodeId(3));
        assert_eq!(restored.reserve_edge_id(1)?, EdgeId(2));
        assert_eq!(restored.node_count(), 1);
        assert_eq!(restored.edge_count(), 0);
        Ok(())
    }

    #[test]
    fn recycled_ordinals_never_change_a_paused_readers_identity() -> Result<()> {
        let (graph, label, property, kind) = fixture()?;
        let old_node = graph
            .node(NodeId(1))
            .ok_or_else(|| missing("fixture node"))?;
        let old_edge = graph
            .edge(EdgeId(1))
            .ok_or_else(|| missing("fixture relationship"))?;
        let barrier = Barrier::new(2);
        thread::scope(|scope| -> Result<()> {
            let reader = scope.spawn(|| {
                barrier.wait();
                barrier.wait();
                assert_eq!(old_node.id(), NodeId(1));
                assert!(old_node.property(property).is_none());
                assert!(old_node.labels().is_empty());
                assert_eq!(old_edge.id(), EdgeId(1));
                assert_eq!(old_edge.source(), NodeId(1));
                assert!(old_edge.property(property).is_none());
            });
            barrier.wait();
            graph.delete_node(NodeId(1), true, 2)?;
            let row = graph.insert_node(NodeInput {
                id: NodeId(3),
                layer: Layer::Knowledge,
                revision: 3,
                labels: vec![label],
                properties: vec![(property, ScalarValue::String("new owner".into()))],
            })?;
            assert_eq!(row, old_node.dense());
            let edge_row = graph.insert_edge(EdgeInput {
                id: EdgeId(2),
                source: NodeId(3),
                target: NodeId(2),
                relationship_type: kind,
                layer: Layer::Knowledge,
                revision: 3,
                properties: vec![(property, ScalarValue::Integer(99))],
            })?;
            assert_eq!(edge_row, old_edge.dense());
            barrier.wait();
            reader
                .join()
                .map_err(|_| Error::internal("paused identity reader panicked"))?;
            Ok(())
        })?;
        assert_eq!(graph.expand_out(NodeId(3), None, LayerMask::ALL)?.len(), 1);
        assert!(
            graph
                .expand_out(NodeId(2), None, LayerMask::ALL)?
                .is_empty()
        );
        assert_eq!(graph.node_slot_count(), 2);
        assert_eq!(graph.edge_slot_count(), 1);
        graph.validate_structure()
    }

    #[test]
    fn cached_column_reader_rejects_recycled_views_and_other_owners() -> Result<()> {
        let (graph, label, property, _) = fixture()?;
        graph.set_node_property(NodeId(1), property, ScalarValue::Integer(7), 2)?;
        let reader = graph
            .node_property_reader(property)
            .ok_or_else(|| Error::internal("missing canonical reader"))?;
        let old = graph
            .node(NodeId(1))
            .ok_or_else(|| Error::internal("missing node"))?;
        assert_eq!(reader.get_integer(old), Some(7));
        assert_eq!(reader.get_integer_with_labels(old, &[label]), Some(Some(7)));
        assert_eq!(
            reader.get_integer_with_labels(old, &[LabelId(u64::MAX)]),
            None
        );
        graph.set_node_property(NodeId(1), property, ScalarValue::Integer(8), 3)?;
        assert_eq!(reader.get_integer(old), Some(8));
        assert_eq!(reader.get_integer_with_labels(old, &[label]), Some(Some(8)));
        let (other, _, _, _) = fixture()?;
        assert_eq!(
            other
                .node(NodeId(1))
                .and_then(|node| reader.get_integer(node)),
            None
        );
        assert_eq!(
            other
                .node(NodeId(1))
                .and_then(|node| reader.get_integer_with_labels(node, &[label])),
            None
        );
        graph.delete_node(NodeId(1), true, 4)?;
        graph.insert_node(NodeInput {
            id: NodeId(3),
            layer: Layer::Observed,
            revision: 5,
            labels: vec![label],
            properties: vec![(property, ScalarValue::Integer(99))],
        })?;
        assert_eq!(reader.get_integer(old), None);
        assert_eq!(reader.get_integer_with_labels(old, &[label]), None);
        assert_eq!(
            graph
                .node(NodeId(3))
                .and_then(|node| reader.get_integer(node)),
            Some(99)
        );
        let current = graph
            .node(NodeId(3))
            .ok_or_else(|| missing("recycled node"))?;
        assert_eq!(
            reader.get_integer_with_labels(current, &[label]),
            Some(Some(99))
        );
        graph.set_node_property(NodeId(3), property, ScalarValue::Null, 6)?;
        assert_eq!(
            reader.get_integer_with_labels(current, &[label]),
            Some(None)
        );
        graph.remove_node_labels(NodeId(3), vec![label], 7)?;
        assert_eq!(reader.get_integer_with_labels(current, &[label]), None);
        assert_eq!(reader.get_integer_with_labels(current, &[]), Some(None));
        let second = graph.catalog().intern_label("Reader")?;
        graph.add_node_labels(NodeId(3), vec![label, second], 8)?;
        assert_eq!(
            reader.get_integer_with_labels(current, &[label, second]),
            Some(None)
        );
        Ok(())
    }

    #[test]
    fn million_insert_delete_cycles_bound_canonical_slots_and_payloads() -> Result<()> {
        let (graph, label, property, kind) = fixture()?;
        graph.delete_node(NodeId(1), true, 2)?;
        let baseline = graph.resident_bytes();
        let body = "x".repeat(2048);
        for index in 0..1_000_000_u64 {
            let id = index + 3;
            let revision = index + 3;
            graph.insert_node(NodeInput {
                id: NodeId(id),
                layer: Layer::Knowledge,
                revision,
                labels: vec![label],
                properties: vec![(
                    property,
                    ScalarValue::String(format!("{index}:{body}").into()),
                )],
            })?;
            graph.insert_edge(EdgeInput {
                id: EdgeId(id),
                source: NodeId(id),
                target: NodeId(2),
                relationship_type: kind,
                layer: Layer::Knowledge,
                revision,
                properties: Vec::new(),
            })?;
            graph.delete_node(NodeId(id), true, revision)?;
        }
        assert_eq!(graph.node_count(), 1);
        assert_eq!(graph.edge_count(), 0);
        assert_eq!(graph.node_slot_count(), 2);
        assert_eq!(graph.edge_slot_count(), 1);
        assert_eq!(graph.0.node_lookup.len(), 1);
        assert_eq!(graph.0.edge_lookup.len(), 0);
        assert_eq!(graph.resident_bytes(), baseline);
        graph.validate_structure()
    }

    #[test]
    fn deleted_checkpoint_slots_are_recycled_after_recovery() -> Result<()> {
        let (graph, label, property, _) = fixture()?;
        graph.delete_node(NodeId(1), true, 2)?;
        let bytes =
            postcard::to_allocvec(&graph).map_err(|e| Error::invalid_data(e.to_string()))?;
        let restored: GraphStore =
            postcard::from_bytes(&bytes).map_err(|e| Error::invalid_data(e.to_string()))?;
        let row = restored.insert_node(NodeInput {
            id: NodeId(3),
            layer: Layer::Knowledge,
            revision: 3,
            labels: vec![label],
            properties: vec![(property, ScalarValue::Integer(3))],
        })?;
        assert_eq!(row, 0);
        assert_eq!(restored.node_slot_count(), 2);
        restored.validate_structure()
    }

    #[test]
    fn readers_finish_while_writer_is_paused_between_records() -> Result<()> {
        let (graph, label, property, _) = fixture()?;
        let written = Arc::new(Barrier::new(2));
        let release = Arc::new(Barrier::new(2));
        let writer_graph = graph.clone();
        let writer_written = Arc::clone(&written);
        let writer_release = Arc::clone(&release);
        let writer = thread::spawn(move || -> Result<()> {
            writer_graph.set_node_property(NodeId(1), property, ScalarValue::Integer(100), 2)?;
            writer_written.wait();
            writer_release.wait();
            writer_graph.set_node_property(NodeId(2), property, ScalarValue::Integer(200), 2)
        });
        written.wait();
        let (send, receive) = mpsc::channel();
        let reader_graph = graph.clone();
        let reader = thread::spawn(move || -> Result<()> {
            let first = reader_graph
                .node(NodeId(1))
                .ok_or_else(|| missing("first node absent"))?
                .property(property);
            let second = reader_graph
                .node(NodeId(2))
                .ok_or_else(|| missing("second node absent"))?
                .property(property);
            let scanned = reader_graph.scan_nodes(Some(label), LayerMask::ALL).count();
            let expanded = reader_graph
                .expand_out(NodeId(1), None, LayerMask::ALL)?
                .len();
            send.send((first, second, scanned, expanded))
                .map_err(|error| Error::internal(error.to_string()))?;
            Ok(())
        });
        let result = receive.recv_timeout(Duration::from_secs(2));
        release.wait();
        writer
            .join()
            .map_err(|_| Error::internal("writer panicked"))??;
        reader
            .join()
            .map_err(|_| Error::internal("reader panicked"))??;
        let (first, second, scanned, expanded) = result.map_err(|error| {
            Error::internal(format!("reader waited for paused writer: {error}"))
        })?;
        assert_eq!(first, Some(ScalarValue::Integer(100)));
        assert_eq!(second, Some(ScalarValue::Integer(2)));
        assert_eq!(scanned, 2);
        assert_eq!(expanded, 1);
        Ok(())
    }

    #[test]
    fn replaced_large_payload_reclaims_every_unborrowed_intermediate() -> Result<()> {
        let (graph, _, property, _) = fixture()?;
        let metadata_bytes = graph.0.node_properties.bytes.load(Ordering::Acquire);
        let body = "a".repeat(1024 * 1024);
        graph.set_node_property(NodeId(1), property, ScalarValue::String(body.into()), 2)?;
        let column = graph
            .node_property_column(property)
            .ok_or_else(|| Error::internal("property column absent"))?;
        let cell = column
            .cells
            .get(0)
            .ok_or_else(|| Error::internal("property cell absent"))?;
        let borrowed = cell
            .payload
            .load_full()
            .ok_or_else(|| Error::internal("property payload absent"))?;
        let old_bytes = borrowed.bytes;
        for revision in 3..=130 {
            graph.set_node_property(
                NodeId(1),
                property,
                ScalarValue::String("b".repeat(1024 * 1024).into()),
                revision,
            )?;
            let retained = graph.0.node_properties.bytes.load(Ordering::Acquire);
            assert!(
                retained <= metadata_bytes + old_bytes * 2 + size_of::<ValueAllocation>(),
                "retained {retained} exceeds one borrowed and one current payload"
            );
        }
        let before = graph.resident_bytes();
        drop(borrowed);
        assert_eq!(before - graph.resident_bytes(), old_bytes);
        graph.delete_node(NodeId(1), true, 131)?;
        assert_eq!(
            graph.0.node_properties.bytes.load(Ordering::Acquire),
            metadata_bytes
        );
        Ok(())
    }

    #[test]
    fn labels_reclaim_only_after_actual_borrow_ends() -> Result<()> {
        let (graph, label, _, _) = fixture()?;
        let other = graph.catalog().intern_label("Borrowed")?;
        graph.add_node_labels(NodeId(1), vec![other], 2)?;
        let borrowed = graph
            .node(NodeId(1))
            .ok_or_else(|| missing("node missing"))?
            .labels();
        graph.remove_node_labels(NodeId(1), vec![label, other], 3)?;
        assert_eq!(&*borrowed, &[label, other]);
        let before = graph.resident_bytes();
        drop(borrowed);
        assert!(graph.resident_bytes() < before);
        Ok(())
    }

    #[test]
    fn shared_store_clone_is_one_mutable_owner() -> Result<()> {
        let (graph, _, property, _) = fixture()?;
        let second = graph.clone();
        graph.set_node_property(NodeId(1), property, ScalarValue::Integer(42), 2)?;
        assert!(Arc::ptr_eq(&graph.0, &second.0));
        assert_eq!(
            second
                .node(NodeId(1))
                .and_then(|node| node.property(property)),
            Some(ScalarValue::Integer(42))
        );
        Ok(())
    }

    #[test]
    fn invalid_last_mutation_is_rejected_before_first_publication() -> Result<()> {
        let (graph, _, property, _) = fixture()?;
        let mutations = vec![
            GraphMutation::SetNodeProperty {
                node: NodeId(1),
                property,
                value: ScalarValue::Integer(100),
                revision: 2,
            },
            GraphMutation::SetNodeProperty {
                node: NodeId(999),
                property,
                value: ScalarValue::Integer(200),
                revision: 2,
            },
        ];
        assert!(graph.validate_mutations(&mutations).is_err());
        assert_eq!(graph.revision(), 1);
        assert_eq!(
            graph
                .node(NodeId(1))
                .and_then(|node| node.property(property)),
            Some(ScalarValue::Integer(1))
        );
        Ok(())
    }

    #[test]
    fn batch_validation_tracks_new_schema_endpoints_and_detach() -> Result<()> {
        let graph = GraphStore::default();
        let mutations = vec![
            GraphMutation::DeclareLabel {
                name: "Record".to_owned(),
                id: LabelId(0),
            },
            GraphMutation::DeclareRelationshipType {
                name: "LINK".to_owned(),
                id: RelationshipTypeId(0),
            },
            GraphMutation::InsertNode(NodeInput {
                id: NodeId(1),
                layer: Layer::Knowledge,
                revision: 1,
                labels: vec![LabelId(0)],
                properties: Vec::new(),
            }),
            GraphMutation::InsertEdge(EdgeInput {
                id: EdgeId(1),
                source: NodeId(1),
                target: NodeId(1),
                relationship_type: RelationshipTypeId(0),
                layer: Layer::Knowledge,
                revision: 1,
                properties: Vec::new(),
            }),
            GraphMutation::DeleteNode {
                node: NodeId(1),
                detach: true,
                revision: 1,
            },
        ];
        graph.validate_mutations(&mutations)?;
        for mutation in mutations {
            graph.apply(mutation)?;
        }
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
        graph.validate_structure()
    }

    #[test]
    fn malformed_checkpoint_relationship_is_quarantined_without_losing_nodes() -> Result<()> {
        let (graph, _, _, _) = fixture()?;
        let mut bytes = Vec::new();
        ciborium::ser::into_writer(&graph, &mut bytes)
            .map_err(|e| Error::internal(e.to_string()))?;
        let mut wire: ciborium::Value = ciborium::de::from_reader(bytes.as_slice())
            .map_err(|e| Error::internal(e.to_string()))?;
        let ciborium::Value::Map(fields) = &mut wire else {
            return Err(Error::internal("canonical graph is a map"));
        };
        let edges = fields
            .iter_mut()
            .find(|(key, _)| key.as_text() == Some("edges"))
            .ok_or_else(|| Error::internal("edges field absent"))?;
        let ciborium::Value::Array(rows) = &mut edges.1 else {
            return Err(Error::internal("edges are rows"));
        };
        let ciborium::Value::Map(row) = &mut rows[0] else {
            return Err(Error::internal("edge row is a map"));
        };
        row.iter_mut()
            .find(|(key, _)| key.as_text() == Some("source"))
            .ok_or_else(|| Error::internal("source field absent"))?
            .1 = ciborium::Value::Integer(u32::MAX.into());
        bytes.clear();
        ciborium::ser::into_writer(&wire, &mut bytes)
            .map_err(|e| Error::internal(e.to_string()))?;
        let reopened: GraphStore = ciborium::de::from_reader(bytes.as_slice())
            .map_err(|e| Error::internal(e.to_string()))?;
        assert_eq!(reopened.node_count(), 2);
        assert_eq!(reopened.edge_count(), 0);
        assert_eq!(reopened.quarantine_invalid_relationships()?, 1);
        assert_eq!(reopened.quarantine_invalid_relationships()?, 0);
        reopened.validate_structure()
    }

    #[test]
    fn canonical_cbor_and_postcard_stream_round_trip() -> Result<()> {
        let (graph, _, property, _) = fixture()?;
        graph.set_node_property(NodeId(1), property, ScalarValue::String("body".into()), 2)?;
        let mut bytes = Vec::new();
        ciborium::ser::into_writer(&graph, &mut bytes)
            .map_err(|e| Error::internal(e.to_string()))?;
        let reopened: GraphStore = ciborium::de::from_reader(bytes.as_slice())
            .map_err(|e| Error::internal(e.to_string()))?;
        assert_eq!(
            reopened
                .node(NodeId(1))
                .and_then(|node| node.property(property)),
            Some(ScalarValue::String("body".into()))
        );
        let bytes = postcard::to_stdvec(&graph).map_err(|e| Error::internal(e.to_string()))?;
        let reopened: GraphStore =
            postcard::from_bytes(&bytes).map_err(|e| Error::internal(e.to_string()))?;
        assert_eq!(reopened.node_count(), 2);
        assert_eq!(reopened.edge_count(), 1);
        reopened.validate_structure()
    }

    #[test]
    fn previous_cbor_checkpoint_imports_without_legacy_graph_owner() -> Result<()> {
        let mut old = super::super::store::LegacyGraphStore::default();
        let label = old.catalog_mut().intern_label("Record")?;
        let property = old.catalog_mut().intern_property("body")?;
        let kind = old.catalog_mut().intern_relationship_type("LINK")?;
        for id in 1..=3 {
            old.insert_node(NodeInput {
                id: NodeId(id),
                layer: Layer::Knowledge,
                revision: 1,
                labels: vec![label],
                properties: vec![(property, ScalarValue::String(format!("body-{id}").into()))],
            })?;
        }
        old.insert_edge(EdgeInput {
            id: EdgeId(1),
            source: NodeId(1),
            target: NodeId(2),
            relationship_type: kind,
            layer: Layer::Knowledge,
            revision: 1,
            properties: Vec::new(),
        })?;
        old.delete_node(NodeId(3), true, 2)?;
        let mut bytes = Vec::new();
        ciborium::ser::into_writer(&old, &mut bytes).map_err(|e| Error::internal(e.to_string()))?;
        let reopened: GraphStore = ciborium::de::from_reader(bytes.as_slice())
            .map_err(|e| Error::internal(e.to_string()))?;
        assert_eq!(reopened.node_count(), 2);
        assert_eq!(reopened.edge_count(), 1);
        assert!(!reopened.contains_node_id(NodeId(3)));
        assert!(reopened.node(NodeId(3)).is_none());
        assert_eq!(
            reopened
                .node(NodeId(1))
                .and_then(|node| node.property(property)),
            Some(ScalarValue::String("body-1".into()))
        );
        assert_eq!(
            reopened.expand_out(NodeId(1), None, LayerMask::ALL)?.len(),
            1
        );
        reopened.validate_structure()
    }
}
