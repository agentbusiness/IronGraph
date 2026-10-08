//! Canonical temporal samples with individually guarded publication and reclamation.
use super::*;
use arc_swap::ArcSwap;
use ordered_float::OrderedFloat;
use papaya::{Guard, HashMap};
use serde::{
    Deserialize, Serialize,
    ser::{SerializeMap, SerializeSeq, SerializeStruct},
};
use std::{
    collections::BTreeMap,
    sync::{
        Arc, Mutex,
        atomic::{AtomicU64, Ordering},
    },
};

type Key = (u8, u64, PropertyId);

#[derive(Debug)]
struct Column {
    declaration: ArcSwap<TemporalDeclaration>,
    rows: HashMap<u64, TemporalSample>,
    entities: HashMap<u64, Arc<HashMap<u64, ()>>>,
    current: HashMap<u64, u64>,
    expiry: Mutex<BTreeMap<(i64, u64), ()>>,
    rollups: HashMap<String, ()>,
    next_row: AtomicU64,
}

impl Column {
    fn new(declaration: TemporalDeclaration) -> Self {
        Self {
            declaration: ArcSwap::from_pointee(declaration),
            rows: HashMap::new(),
            entities: HashMap::new(),
            current: HashMap::new(),
            expiry: Mutex::new(BTreeMap::new()),
            rollups: HashMap::new(),
            next_row: AtomicU64::new(0),
        }
    }
    fn sample(&self, row: u64) -> Option<TemporalSample> {
        self.rows.pin().get(&row).cloned()
    }
    fn keys(&self) -> Vec<u64> {
        let mut keys = self
            .rows
            .pin()
            .iter()
            .map(|(row, _)| *row)
            .collect::<Vec<_>>();
        keys.sort_unstable();
        keys
    }
    fn entity_rows(&self, entity: u64) -> Vec<u64> {
        let index = self.entities.pin().get(&entity).cloned();
        index
            .map(|index| index.pin().iter().map(|(row, ())| *row).collect())
            .unwrap_or_default()
    }
    fn samples(&self) -> impl Iterator<Item = TemporalSample> + '_ {
        self.keys().into_iter().filter_map(|row| self.sample(row))
    }
    fn insert(&self, sample: TemporalSample) -> Result<()> {
        let row = self
            .next_row
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |row| {
                row.checked_add(1)
            })
            .map_err(|_| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "temporal row identity exhausted",
                )
            })?;
        let entity = sample.entity_id;
        let time = sample.event_time_nanos;
        let newest = self
            .current
            .pin()
            .get(&entity)
            .copied()
            .and_then(|row| self.sample(row))
            .is_none_or(|previous| {
                (time, sample.sequence_index) > (previous.event_time_nanos, previous.sequence_index)
            });
        self.rows.pin().insert(row, sample);
        let index = {
            let entities = self.entities.pin();
            if let Some(index) = entities.get(&entity) {
                Arc::clone(index)
            } else {
                let index = Arc::new(HashMap::new());
                entities.insert(entity, Arc::clone(&index));
                index
            }
        };
        index.pin().insert(row, ());
        self.expiry
            .lock()
            .map_err(|_| Error::internal("temporal expiry writer poisoned"))?
            .insert((time, row), ());
        if newest {
            self.current.pin().insert(entity, row);
        }
        Ok(())
    }
    fn rebuild_current(&self, entity: u64) {
        let newest = self
            .entity_rows(entity)
            .into_iter()
            .filter_map(|row| {
                self.sample(row)
                    .map(|sample| (sample.event_time_nanos, sample.sequence_index, row))
            })
            .max();
        if let Some((_, _, row)) = newest {
            self.current.pin().insert(entity, row);
        } else {
            self.current.pin().remove(&entity);
            self.entities.pin().remove(&entity);
        }
    }
}

#[derive(Debug)]
struct Bucket {
    value: ArcSwap<RollupBucket>,
    numbers: Mutex<BTreeMap<OrderedFloat<f64>, u64>>,
    source_rows: AtomicU64,
}

#[derive(Debug)]
struct Rollup {
    definition: TemporalRollupDefinition,
    buckets: HashMap<(u64, i64), Arc<Bucket>>,
}

impl Rollup {
    fn apply(&self, sample: &TemporalSample, remove: bool) -> Result<()> {
        let number = match sample.value {
            ScalarValue::Integer(value) => Some(value as f64),
            ScalarValue::Float(value) => Some(value.into_inner()),
            _ => None,
        };
        let end = sample.event_time_nanos.checked_add(1).ok_or_else(|| {
            Error::new(
                ErrorCode::TemporalRange,
                "sample timestamp cannot form a range",
            )
        })?;
        for (start, end) in
            TemporalStore::window_boundaries(sample.event_time_nanos, end, &self.definition.window)?
        {
            let key = (sample.entity_id, start);
            let bucket = {
                let buckets = self.buckets.pin();
                if let Some(bucket) = buckets.get(&key) {
                    Arc::clone(bucket)
                } else if remove {
                    continue;
                } else {
                    let bucket = Arc::new(Bucket {
                        value: ArcSwap::from_pointee(RollupBucket::empty(start, end)),
                        numbers: Mutex::new(BTreeMap::new()),
                        source_rows: AtomicU64::new(0),
                    });
                    buckets.insert(key, Arc::clone(&bucket));
                    bucket
                }
            };
            let mut numbers = bucket
                .numbers
                .lock()
                .map_err(|_| Error::internal("rollup writer poisoned"))?;
            let source_rows = if remove {
                bucket
                    .source_rows
                    .fetch_sub(1, Ordering::Relaxed)
                    .saturating_sub(1)
            } else {
                bucket
                    .source_rows
                    .fetch_add(1, Ordering::Relaxed)
                    .saturating_add(1)
            };
            if let Some(number) = number {
                let numeric = OrderedFloat(number);
                if remove {
                    if let Some(count) = numbers.get_mut(&numeric) {
                        *count -= 1;
                        if *count == 0 {
                            numbers.remove(&numeric);
                        }
                    }
                } else {
                    *numbers.entry(numeric).or_default() += 1;
                }
            }
            let previous = bucket.value.load();
            let count = if number.is_none() {
                previous.count
            } else if remove {
                previous.count.saturating_sub(1)
            } else {
                previous.count.saturating_add(1)
            };
            let sum = if count == 0 {
                None
            } else {
                Some(
                    previous.sum.unwrap_or(0.0)
                        + number.map_or(0.0, |number| if remove { -number } else { number }),
                )
            };
            let next = RollupBucket {
                start_nanos: start,
                end_nanos: end,
                count,
                sum,
                min: numbers
                    .first_key_value()
                    .map(|(value, _)| value.into_inner()),
                max: numbers
                    .last_key_value()
                    .map(|(value, _)| value.into_inner()),
                avg: sum.map(|sum| sum / count as f64),
            };
            drop(previous);
            bucket.value.store(Arc::new(next));
            if source_rows == 0 {
                self.buckets.pin().remove(&key);
            }
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
struct State {
    columns: HashMap<Key, Arc<Column>>,
    rollups: HashMap<String, Arc<Rollup>>,
    writer: Mutex<()>,
}

/// Cloning shares the same canonical store; mutation never detaches a graph generation.
#[derive(Clone, Debug, Default)]
pub struct TemporalStore {
    state: Arc<State>,
}

impl TemporalStore {
    fn key(kind: EntityKind, target: u64, property: PropertyId) -> Key {
        (kind as u8, target, property)
    }
    fn column(&self, key: Key) -> Option<Arc<Column>> {
        self.state.columns.pin().get(&key).cloned()
    }
    fn columns(&self) -> Vec<(Key, Arc<Column>)> {
        let mut columns = self
            .state
            .columns
            .pin()
            .iter()
            .map(|(key, column)| (*key, Arc::clone(column)))
            .collect::<Vec<_>>();
        columns.sort_by_key(|(key, _)| *key);
        columns
    }
    pub fn is_empty(&self) -> bool {
        self.state.columns.is_empty() && self.state.rollups.is_empty()
    }
    pub fn is_declared(&self, kind: EntityKind, target: u64, property: PropertyId) -> bool {
        self.state
            .columns
            .pin()
            .contains_key(&Self::key(kind, target, property))
    }
    pub fn declares_property(&self, property: PropertyId) -> bool {
        self.state
            .columns
            .pin()
            .iter()
            .any(|(key, _)| key.2 == property)
    }
    pub fn declared_type(
        &self,
        kind: EntityKind,
        target: u64,
        property: PropertyId,
    ) -> Option<TemporalType> {
        self.column(Self::key(kind, target, property))
            .map(|column| column.declaration.load().value_type)
    }
    pub fn resolve_target(
        &self,
        kind: EntityKind,
        targets: &[u64],
        property: PropertyId,
    ) -> Result<Option<u64>> {
        let mut matches = targets
            .iter()
            .copied()
            .filter(|target| self.is_declared(kind, *target, property));
        let first = matches.next();
        if matches.next().is_some() {
            return Err(Error::new(
                ErrorCode::QueryType,
                "temporal property matches more than one entity target",
            ));
        }
        Ok(first)
    }
    /// Pure pre-WAL validation; never publishes schema or expires samples.
    pub fn validate_declare(&self, declaration: &TemporalDeclaration) -> Result<()> {
        if declaration.retention_nanos <= 0 {
            return Err(Error::new(
                ErrorCode::TemporalRange,
                "retention must be positive",
            ));
        }
        let key = Self::key(
            declaration.entity_kind,
            declaration.target,
            declaration.property,
        );
        if let Some(column) = self.column(key) {
            if column.declaration.load().value_type != declaration.value_type
                && !column.rows.is_empty()
            {
                return Err(Error::new(
                    ErrorCode::QueryType,
                    "a populated temporal property cannot change scalar type",
                ));
            }
        }
        Ok(())
    }
    pub fn declare(&self, declaration: TemporalDeclaration, now: i64) -> Result<()> {
        let _writer = self
            .state
            .writer
            .lock()
            .map_err(|_| Error::internal("temporal writer poisoned"))?;
        self.validate_declare(&declaration)?;
        let key = Self::key(
            declaration.entity_kind,
            declaration.target,
            declaration.property,
        );
        if let Some(column) = self.column(key) {
            column.declaration.store(Arc::new(declaration));
        } else {
            self.state
                .columns
                .pin()
                .insert(key, Arc::new(Column::new(declaration)));
        }
        self.expire(key, now)
    }
    pub fn validate_append(
        &self,
        kind: EntityKind,
        target: u64,
        sample: &TemporalSample,
        now: i64,
    ) -> Result<()> {
        let column = self
            .column(Self::key(kind, target, sample.property))
            .ok_or_else(|| Error::new(ErrorCode::QueryType, "property is not declared temporal"))?;
        let declaration = column.declaration.load();
        if sample.value.is_document() {
            return Err(Error::new(
                ErrorCode::QueryType,
                "temporal properties accept scalar values only; LIST and MAP are non-temporal",
            ));
        }
        if !declaration.value_type.accepts(&sample.value) {
            return Err(Error::new(
                ErrorCode::QueryType,
                "temporal sample has the wrong scalar type",
            ));
        }
        if sample.event_time_nanos < now.saturating_sub(declaration.retention_nanos) {
            return Err(Error::new(
                ErrorCode::RetentionExpired,
                "temporal sample is older than the canonical retention horizon",
            ));
        }
        // Rollup boundaries must be valid before publishing the canonical sample.
        for name in column
            .rollups
            .pin()
            .iter()
            .map(|(name, ())| name.clone())
            .collect::<Vec<_>>()
        {
            if let Some(rollup) = self.state.rollups.pin().get(&name) {
                let end = sample.event_time_nanos.checked_add(1).ok_or_else(|| {
                    Error::new(
                        ErrorCode::TemporalRange,
                        "sample timestamp cannot form a range",
                    )
                })?;
                Self::window_boundaries(sample.event_time_nanos, end, &rollup.definition.window)?;
            }
        }
        Ok(())
    }
    pub fn append(
        &self,
        kind: EntityKind,
        target: u64,
        sample: TemporalSample,
        now: i64,
    ) -> Result<()> {
        let _writer = self
            .state
            .writer
            .lock()
            .map_err(|_| Error::internal("temporal writer poisoned"))?;
        self.validate_append(kind, target, &sample, now)?;
        let key = Self::key(kind, target, sample.property);
        let column = self
            .column(key)
            .ok_or_else(|| Error::internal("temporal column disappeared"))?;
        // A single borrowed sample supplies derived aggregates, then moves into its canonical row.
        self.apply_rollups(&column, &sample, false)?;
        column.insert(sample)?;
        self.expire(key, now)
    }
    fn apply_rollups(&self, column: &Column, sample: &TemporalSample, remove: bool) -> Result<()> {
        let names = column
            .rollups
            .pin()
            .iter()
            .map(|(name, ())| name.clone())
            .collect::<Vec<_>>();
        for name in names {
            let rollup = self.state.rollups.pin().get(&name).cloned();
            if let Some(rollup) = rollup {
                rollup.apply(sample, remove)?;
            }
        }
        Ok(())
    }
    fn expire(&self, key: Key, now: i64) -> Result<()> {
        let column = self
            .column(key)
            .ok_or_else(|| Error::internal("temporal column disappeared"))?;
        let oldest = now.saturating_sub(column.declaration.load().retention_nanos);
        let mut affected_entities = std::collections::BTreeSet::new();
        loop {
            let expired = {
                let mut expiry = column
                    .expiry
                    .lock()
                    .map_err(|_| Error::internal("temporal expiry poisoned"))?;
                match expiry.first_key_value() {
                    Some((key, ())) if key.0 < oldest => {
                        let key = *key;
                        expiry.remove(&key);
                        Some(key.1)
                    }
                    _ => None,
                }
            };
            let Some(row) = expired else {
                break;
            };
            let sample = column.sample(row);
            if let Some(sample) = sample {
                self.apply_rollups(&column, &sample, true)?;
                column.rows.pin().remove(&row);
                if let Some(index) = column.entities.pin().get(&sample.entity_id) {
                    index.pin().remove(&row);
                }
                if column.current.pin().get(&sample.entity_id) == Some(&row) {
                    affected_entities.insert(sample.entity_id);
                }
            }
        }
        for entity in affected_entities {
            column.rebuild_current(entity);
        }
        // Do not leave large retired scalar payloads in a thread-local batch until later work.
        // Every returned read value owns only its currently borrowed scalar allocation.
        column.rows.guard().flush();
        Ok(())
    }
    pub fn current(
        &self,
        kind: EntityKind,
        target: u64,
        entity: u64,
        property: PropertyId,
    ) -> Option<TemporalSample> {
        let column = self.column(Self::key(kind, target, property))?;
        let row = column.current.pin().get(&entity).copied()?;
        column.sample(row)
    }
    pub fn at_time(
        &self,
        kind: EntityKind,
        target: u64,
        entity: u64,
        property: PropertyId,
        when: i64,
        bookmark: u64,
    ) -> Option<TemporalSample> {
        let column = self.column(Self::key(kind, target, property))?;
        column
            .entity_rows(entity)
            .into_iter()
            .filter_map(|row| column.sample(row))
            .filter(|sample| sample.event_time_nanos <= when && sample.sequence_index <= bookmark)
            .max_by_key(|sample| (sample.event_time_nanos, sample.sequence_index))
    }
    pub fn history(
        &self,
        kind: EntityKind,
        target: u64,
        entity: u64,
        property: PropertyId,
        from: i64,
        to: i64,
        bookmark: u64,
    ) -> Result<Vec<TemporalSample>> {
        if from >= to {
            return Err(Error::new(
                ErrorCode::TemporalRange,
                "HISTORY range must be non-empty",
            ));
        }
        let column = self
            .column(Self::key(kind, target, property))
            .ok_or_else(|| Error::new(ErrorCode::QueryType, "property is not declared temporal"))?;
        let mut samples = column
            .entity_rows(entity)
            .into_iter()
            .filter_map(|row| column.sample(row))
            .filter(|sample| {
                sample.event_time_nanos >= from
                    && sample.event_time_nanos < to
                    && sample.sequence_index <= bookmark
            })
            .collect::<Vec<_>>();
        samples.sort_by_key(|sample| (sample.event_time_nanos, sample.sequence_index));
        Ok(samples)
    }
    pub fn compact(&self, now: i64) -> Result<()> {
        let _writer = self
            .state
            .writer
            .lock()
            .map_err(|_| Error::internal("temporal writer poisoned"))?;
        for (key, _) in self.columns() {
            self.expire(key, now)?;
        }
        Ok(())
    }
    /// Pure pre-WAL validation; never registers or fills a derived rollup.
    pub fn validate_create_rollup(&self, definition: &TemporalRollupDefinition) -> Result<()> {
        if definition.name.is_empty()
            || definition.name.len() > 255
            || definition.aggregates.is_empty()
        {
            return Err(Error::new(
                ErrorCode::QueryType,
                "rollup name and aggregate set must be nonempty",
            ));
        }
        definition.window.validate()?;
        let key = Self::key(
            definition.entity_kind,
            definition.target,
            definition.property,
        );
        let column = self.column(key).ok_or_else(|| {
            Error::new(
                ErrorCode::QueryType,
                "rollup property is not declared temporal for its target",
            )
        })?;
        if self.state.rollups.pin().contains_key(&definition.name) {
            return Err(Error::new(
                ErrorCode::TransactionConflict,
                "rollup already exists",
            ));
        }
        for sample in column.samples() {
            let end = sample.event_time_nanos.checked_add(1).ok_or_else(|| {
                Error::new(
                    ErrorCode::TemporalRange,
                    "sample timestamp cannot form a range",
                )
            })?;
            Self::window_boundaries(sample.event_time_nanos, end, &definition.window)?;
        }
        Ok(())
    }
    pub fn create_rollup(&self, mut definition: TemporalRollupDefinition) -> Result<()> {
        let _writer = self
            .state
            .writer
            .lock()
            .map_err(|_| Error::internal("temporal writer poisoned"))?;
        self.validate_create_rollup(&definition)?;
        let key = Self::key(
            definition.entity_kind,
            definition.target,
            definition.property,
        );
        let column = self
            .column(key)
            .ok_or_else(|| Error::internal("rollup column disappeared"))?;
        definition.window.aggregates = definition.aggregates;
        let name = definition.name.clone();
        let rollup = Arc::new(Rollup {
            definition,
            buckets: HashMap::new(),
        });
        for sample in column.samples() {
            rollup.apply(&sample, false)?;
        }
        self.state.rollups.pin().insert(name.clone(), rollup);
        column.rollups.pin().insert(name, ());
        Ok(())
    }
    pub fn rebuild_rollup(&self, name: &str) -> Result<()> {
        let _writer = self
            .state
            .writer
            .lock()
            .map_err(|_| Error::internal("temporal writer poisoned"))?;
        let rollup = self
            .state
            .rollups
            .pin()
            .get(name)
            .cloned()
            .ok_or_else(|| Error::new(ErrorCode::IndexUnavailable, "rollup does not exist"))?;
        // Rebuild each bucket in place; there is no retained second rollup generation.
        for key in rollup
            .buckets
            .pin()
            .iter()
            .map(|(key, _)| *key)
            .collect::<Vec<_>>()
        {
            rollup.buckets.pin().remove(&key);
        }
        let d = &rollup.definition;
        if let Some(column) = self.column(Self::key(d.entity_kind, d.target, d.property)) {
            for sample in column.samples() {
                rollup.apply(&sample, false)?;
            }
        }
        Ok(())
    }
    pub fn rollup_buckets(
        &self,
        name: &str,
        entity: u64,
        from: i64,
        to: i64,
    ) -> Result<Vec<RollupBucket>> {
        if from >= to {
            return Err(Error::new(
                ErrorCode::TemporalRange,
                "rollup range must be non-empty",
            ));
        }
        let rollup = self
            .state
            .rollups
            .pin()
            .get(name)
            .cloned()
            .ok_or_else(|| Error::new(ErrorCode::IndexUnavailable, "rollup does not exist"))?;
        let keys = rollup
            .buckets
            .pin()
            .iter()
            .filter_map(|(key, _)| (key.0 == entity).then_some(*key))
            .collect::<Vec<_>>();
        let mut buckets = keys
            .into_iter()
            .filter_map(|key| {
                let bucket = rollup.buckets.pin().get(&key).cloned()?;
                let value = bucket.value.load();
                (value.end_nanos > from && value.start_nanos < to).then(|| (**value).clone())
            })
            .collect::<Vec<_>>();
        buckets.sort_by_key(|bucket| bucket.start_nanos);
        Ok(buckets)
    }
    pub fn rollup_definitions(&self) -> impl Iterator<Item = TemporalRollupDefinition> {
        let mut definitions = self
            .state
            .rollups
            .pin()
            .iter()
            .map(|(_, rollup)| rollup.definition.clone())
            .collect::<Vec<_>>();
        definitions.sort_by(|a, b| a.name.cmp(&b.name));
        definitions.into_iter()
    }
    pub fn compatible_rollup(
        &self,
        kind: EntityKind,
        target: u64,
        property: PropertyId,
        window: &WindowSpec,
        aggregates: AggregateSet,
    ) -> Option<TemporalRollupDefinition> {
        self.rollup_definitions().find(|d| {
            d.entity_kind == kind
                && d.target == target
                && d.property == property
                && d.window.kind == window.kind
                && d.window.align_nanos == window.align_nanos
                && d.window.timezone == window.timezone
                && d.aggregates.contains(aggregates)
        })
    }
    pub fn window(
        samples: &[TemporalSample],
        from: i64,
        to: i64,
        spec: &WindowSpec,
    ) -> Result<Vec<RollupBucket>> {
        InactiveTemporalStore::window(samples, from, to, spec)
    }
    pub fn window_boundaries(from: i64, to: i64, spec: &WindowSpec) -> Result<Vec<(i64, i64)>> {
        InactiveTemporalStore::window_boundaries(from, to, spec)
    }
    pub fn optimizer_statistics(&self) -> Vec<TemporalOptimizerColumnStatistics> {
        self.columns()
            .into_iter()
            .map(|(_, column)| {
                let declaration = column.declaration.load();
                let mut minimum = None;
                let mut maximum = None;
                for sample in column.samples() {
                    minimum = Some(minimum.map_or(sample.event_time_nanos, |min: i64| {
                        min.min(sample.event_time_nanos)
                    }));
                    maximum = Some(maximum.map_or(sample.event_time_nanos, |max: i64| {
                        max.max(sample.event_time_nanos)
                    }));
                }
                let count = column.rows.len() as u64;
                TemporalOptimizerColumnStatistics {
                    entity_kind: declaration.entity_kind,
                    target: declaration.target,
                    property: declaration.property,
                    segment_count: u64::from(count != 0),
                    sample_count: count,
                    minimum_event_time_nanos: minimum,
                    maximum_event_time_nanos: maximum,
                    compatible_rollups: column.rollups.len() as u64,
                    resident_bytes: count.saturating_mul(48),
                }
            })
            .collect()
    }
}

// Preserve the existing snapshot field layout while streaming canonical rows only once. Current
// pointers and derived buckets are rebuilt on recovery and encoded as empty maps.
struct Samples<'a>(&'a Column);
impl Serialize for Samples<'_> {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        let keys = self.0.keys();
        let mut sequence = serializer.serialize_seq(Some(keys.len()))?;
        // Snapshot serialization is sequenced with mutations by the database snapshot writer.
        for row in keys {
            let sample = self
                .0
                .sample(row)
                .ok_or_else(|| serde::ser::Error::custom("temporal row removed during snapshot"))?;
            sequence.serialize_element(&sample)?;
        }
        sequence.end()
    }
}
struct PersistColumn<'a>(&'a Column);
impl Serialize for PersistColumn<'_> {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        let mut state = serializer.serialize_struct("PersistedTemporalColumn", 3)?;
        state.serialize_field("base", &Samples(self.0))?;
        state.serialize_field("delta", &Vec::<TemporalSample>::new())?;
        state.serialize_field("current", &BTreeMap::<u128, TemporalSample>::new())?;
        state.end()
    }
}
struct PersistColumns<'a>(&'a TemporalStore);
impl Serialize for PersistColumns<'_> {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        let columns = self.0.columns();
        let mut map = serializer.serialize_map(Some(columns.len()))?;
        for (key, column) in columns {
            map.serialize_entry(&key, &PersistColumn(&column))?;
        }
        map.end()
    }
}
struct PersistDeclarations<'a>(&'a TemporalStore);
impl Serialize for PersistDeclarations<'_> {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        let columns = self.0.columns();
        let mut map = serializer.serialize_map(Some(columns.len()))?;
        for (key, column) in columns {
            map.serialize_entry(&key, &**column.declaration.load())?;
        }
        map.end()
    }
}
#[derive(Serialize)]
struct PersistRollup<'a> {
    definition: &'a TemporalRollupDefinition,
    buckets: BTreeMap<(u64, i64), RollupBucket>,
}
struct PersistRollups<'a>(&'a TemporalStore);
impl Serialize for PersistRollups<'_> {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        let definitions = self.0.rollup_definitions().collect::<Vec<_>>();
        let mut map = serializer.serialize_map(Some(definitions.len()))?;
        for d in definitions {
            map.serialize_entry(
                &d.name,
                &PersistRollup {
                    definition: &d,
                    buckets: BTreeMap::new(),
                },
            )?;
        }
        map.end()
    }
}
impl Serialize for TemporalStore {
    fn serialize<S: serde::Serializer>(
        &self,
        serializer: S,
    ) -> std::result::Result<S::Ok, S::Error> {
        let _writer = self
            .state
            .writer
            .lock()
            .map_err(|_| serde::ser::Error::custom("temporal writer poisoned"))?;
        let mut state = serializer.serialize_struct("TemporalStore", 3)?;
        state.serialize_field("declarations", &PersistDeclarations(self))?;
        state.serialize_field("columns", &PersistColumns(self))?;
        state.serialize_field("rollups", &PersistRollups(self))?;
        state.end()
    }
}
#[derive(Deserialize)]
struct DecodedColumn {
    base: Vec<TemporalSample>,
    delta: Vec<TemporalSample>,
    current: BTreeMap<u128, TemporalSample>,
}
#[derive(Deserialize)]
struct DecodedRollup {
    definition: TemporalRollupDefinition,
    buckets: BTreeMap<(u64, i64), RollupBucket>,
}
#[derive(Deserialize)]
struct DecodedStore {
    declarations: BTreeMap<Key, TemporalDeclaration>,
    columns: BTreeMap<Key, DecodedColumn>,
    rollups: BTreeMap<String, DecodedRollup>,
}
impl<'de> Deserialize<'de> for TemporalStore {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        let decoded = DecodedStore::deserialize(deserializer)?;
        if decoded.declarations.len() != decoded.columns.len() {
            return Err(serde::de::Error::custom(
                "temporal declarations and columns differ",
            ));
        }
        let store = Self::default();
        for (key, column) in decoded.columns {
            let d = decoded
                .declarations
                .get(&key)
                .ok_or_else(|| serde::de::Error::custom("temporal declaration missing"))?;
            if key != Self::key(d.entity_kind, d.target, d.property) || d.retention_nanos <= 0 {
                return Err(serde::de::Error::custom("invalid temporal declaration"));
            }
            let canonical = Arc::new(Column::new(d.clone()));
            for sample in column.base.into_iter().chain(column.delta) {
                if sample.property != d.property || !d.value_type.accepts(&sample.value) {
                    return Err(serde::de::Error::custom("invalid temporal sample"));
                }
                canonical.insert(sample).map_err(serde::de::Error::custom)?;
            }
            drop(column.current);
            store.state.columns.pin().insert(key, canonical);
        }
        for (name, rollup) in decoded.rollups {
            if name != rollup.definition.name
                || rollup.definition.window.aggregates != rollup.definition.aggregates
            {
                return Err(serde::de::Error::custom(
                    "invalid temporal rollup declaration",
                ));
            }
            drop(rollup.buckets);
            store
                .create_rollup(rollup.definition)
                .map_err(serde::de::Error::custom)?;
        }
        Ok(store)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{sync::mpsc, time::Duration};

    fn declaration(
        property: PropertyId,
        value_type: TemporalType,
        retention: i64,
    ) -> TemporalDeclaration {
        TemporalDeclaration {
            entity_kind: EntityKind::Node,
            target: 7,
            property,
            value_type,
            retention_nanos: retention,
        }
    }
    fn sample(
        property: PropertyId,
        entity: u64,
        time: i64,
        sequence: u64,
        value: ScalarValue,
    ) -> TemporalSample {
        TemporalSample {
            entity_id: entity,
            property,
            event_time_nanos: time,
            sequence_index: sequence,
            value,
        }
    }
    fn numeric(property: PropertyId, time: i64, value: i64) -> TemporalSample {
        sample(property, 9, time, time as u64, ScalarValue::Integer(value))
    }
    fn rollup(property: PropertyId) -> TemporalRollupDefinition {
        TemporalRollupDefinition {
            name: "hourly".into(),
            entity_kind: EntityKind::Node,
            target: 7,
            property,
            window: WindowSpec::tumbling(100),
            aggregates: AggregateSet::all(),
        }
    }

    #[test]
    fn temporal_reads_complete_while_writer_is_paused() -> Result<()> {
        let store = TemporalStore::default();
        let property = PropertyId(2);
        store.declare(declaration(property, TemporalType::Integer, 1000), 0)?;
        store.append(EntityKind::Node, 7, numeric(property, 10, 42), 10)?;
        store.create_rollup(rollup(property))?;
        let writer_store = store.clone();
        let reader_store = store.clone();
        let (locked_tx, locked_rx) = mpsc::channel();
        let (release_tx, release_rx) = mpsc::channel();
        let writer = std::thread::spawn(move || -> Result<()> {
            let _writer = writer_store
                .state
                .writer
                .lock()
                .map_err(|_| Error::internal("writer poisoned"))?;
            locked_tx
                .send(())
                .map_err(|_| Error::internal("pause channel closed"))?;
            release_rx
                .recv()
                .map_err(|_| Error::internal("resume channel closed"))?;
            Ok(())
        });
        locked_rx
            .recv_timeout(Duration::from_secs(2))
            .map_err(|_| Error::internal("writer failed to pause"))?;
        let (done_tx, done_rx) = mpsc::channel();
        let reader = std::thread::spawn(move || -> Result<()> {
            assert_eq!(
                reader_store
                    .current(EntityKind::Node, 7, 9, property)
                    .map(|s| s.value),
                Some(ScalarValue::Integer(42))
            );
            assert!(
                reader_store
                    .at_time(EntityKind::Node, 7, 9, property, 10, 10)
                    .is_some()
            );
            let history = reader_store.history(EntityKind::Node, 7, 9, property, 0, 100, 10)?;
            assert_eq!(history.len(), 1);
            assert_eq!(
                TemporalStore::window(&history, 0, 100, &WindowSpec::tumbling(100))?[0].sum,
                Some(42.0)
            );
            assert_eq!(
                reader_store.rollup_buckets("hourly", 9, 0, 100)?[0].sum,
                Some(42.0)
            );
            assert_eq!(reader_store.optimizer_statistics()[0].sample_count, 1);
            done_tx
                .send(())
                .map_err(|_| Error::internal("read completion channel closed"))?;
            Ok(())
        });
        let completion = done_rx.recv_timeout(Duration::from_secs(2));
        release_tx
            .send(())
            .map_err(|_| Error::internal("writer resume failed"))?;
        writer
            .join()
            .map_err(|_| Error::internal("writer panicked"))??;
        reader
            .join()
            .map_err(|_| Error::internal("reader panicked"))??;
        assert!(completion.is_ok(), "reads waited for paused writer");
        Ok(())
    }

    #[test]
    fn temporal_retention_reclaims_large_payloads_without_detaching_shared_store() -> Result<()> {
        let store = TemporalStore::default();
        let property = PropertyId(4);
        store.declare(declaration(property, TemporalType::String, 4), 0)?;
        let reader = store.clone();
        let mut allocations = Vec::new();
        for tick in 0..2048_i64 {
            let text: Arc<str> = Arc::from(format!("{tick}:{}", "x".repeat(32768)));
            allocations.push(Arc::downgrade(&text));
            store.append(
                EntityKind::Node,
                7,
                sample(property, 9, tick, tick as u64, ScalarValue::String(text)),
                tick,
            )?;
        }
        assert_eq!(
            reader
                .current(EntityKind::Node, 7, 9, property)
                .map(|s| s.event_time_nanos),
            Some(2047)
        );
        let column = store
            .column(TemporalStore::key(EntityKind::Node, 7, property))
            .ok_or_else(|| Error::internal("column absent"))?;
        assert_eq!(column.rows.len(), 5);
        assert_eq!(column.entity_rows(9).len(), 5);
        assert_eq!(
            column
                .expiry
                .lock()
                .map_err(|_| Error::internal("expiry poisoned"))?
                .len(),
            5
        );
        assert!(
            allocations
                .iter()
                .filter(|weak| weak.upgrade().is_some())
                .count()
                <= 16,
            "expired payloads accumulate beyond live rows and bounded reclamation batch"
        );
        assert!(Arc::ptr_eq(&store.state, &reader.state));
        Ok(())
    }

    #[test]
    fn temporal_null_only_rollup_bucket_survives_until_its_last_source_row_expires() -> Result<()> {
        let store = TemporalStore::default();
        let property = PropertyId(8);
        store.declare(declaration(property, TemporalType::Integer, 100), 0)?;
        store.create_rollup(rollup(property))?;
        let mut null = numeric(property, 90, 0);
        null.value = ScalarValue::Null;
        store.append(EntityKind::Node, 7, null, 90)?;
        store.append(EntityKind::Node, 7, numeric(property, 30, 3), 90)?;
        store.compact(150)?;
        let bucket = store.rollup_buckets("hourly", 9, 0, 100)?.remove(0);
        assert_eq!(
            (bucket.count, bucket.sum, bucket.min, bucket.max),
            (0, None, None, None)
        );
        store.compact(191)?;
        assert!(store.rollup_buckets("hourly", 9, 0, 100)?.is_empty());
        Ok(())
    }

    #[test]
    fn temporal_late_samples_retention_and_restart_preserve_rollups_and_history() -> Result<()> {
        let store = TemporalStore::default();
        let property = PropertyId(8);
        store.declare(declaration(property, TemporalType::Integer, 100), 0)?;
        store.create_rollup(rollup(property))?;
        for (time, value) in [(30, 3), (10, 1), (20, 2), (90, 9)] {
            store.append(EntityKind::Node, 7, numeric(property, time, value), 90)?;
        }
        assert_eq!(
            store
                .current(EntityKind::Node, 7, 9, property)
                .map(|s| s.event_time_nanos),
            Some(90)
        );
        assert_eq!(
            store
                .at_time(EntityKind::Node, 7, 9, property, 25, 20)
                .map(|s| s.value),
            Some(ScalarValue::Integer(2))
        );
        assert_eq!(
            store.rollup_buckets("hourly", 9, 0, 100)?[0].sum,
            Some(15.0)
        );
        store.compact(125)?;
        let bucket = store.rollup_buckets("hourly", 9, 0, 100)?.remove(0);
        assert_eq!(
            (bucket.count, bucket.sum, bucket.min, bucket.max),
            (2, Some(12.0), Some(3.0), Some(9.0))
        );
        let mut wire = Vec::new();
        ciborium::ser::into_writer(&store, &mut wire)
            .map_err(|e| Error::internal(e.to_string()))?;
        let recovered: TemporalStore = ciborium::de::from_reader(wire.as_slice())
            .map_err(|e| Error::internal(e.to_string()))?;
        assert_eq!(
            recovered.history(EntityKind::Node, 7, 9, property, 0, 100, u64::MAX)?,
            store.history(EntityKind::Node, 7, 9, property, 0, 100, u64::MAX)?
        );
        assert_eq!(
            recovered.rollup_buckets("hourly", 9, 0, 100)?,
            store.rollup_buckets("hourly", 9, 0, 100)?
        );
        let mut legacy: InactiveTemporalStore = ciborium::de::from_reader(wire.as_slice())
            .map_err(|e| Error::internal(e.to_string()))?;
        let mut legacy_wire = Vec::new();
        ciborium::ser::into_writer(&legacy, &mut legacy_wire)
            .map_err(|e| Error::internal(e.to_string()))?;
        let compatible: TemporalStore = ciborium::de::from_reader(legacy_wire.as_slice())
            .map_err(|e| Error::internal(e.to_string()))?;
        assert_eq!(
            compatible.current(EntityKind::Node, 7, 9, property),
            store.current(EntityKind::Node, 7, 9, property)
        );
        let declaration = Arc::get_mut(&mut legacy.declarations)
            .ok_or_else(|| Error::internal("legacy corruption fixture unexpectedly shared"))?
            .get_mut(&TemporalStore::key(EntityKind::Node, 7, property))
            .ok_or_else(|| Error::internal("test declaration absent"))?;
        declaration.retention_nanos = 0;
        let mut corrupt = Vec::new();
        ciborium::ser::into_writer(&legacy, &mut corrupt)
            .map_err(|e| Error::internal(e.to_string()))?;
        assert!(ciborium::de::from_reader::<TemporalStore, _>(corrupt.as_slice()).is_err());
        Ok(())
    }

    #[test]
    fn temporal_point_append_touches_only_its_column_and_affected_buckets() -> Result<()> {
        let store = TemporalStore::default();
        let property = PropertyId(3);
        let other = PropertyId(5);
        store.declare(declaration(property, TemporalType::Integer, i64::MAX), 0)?;
        store.declare(declaration(other, TemporalType::String, i64::MAX), 0)?;
        store.create_rollup(rollup(property))?;
        for entity in 0..4096_u64 {
            store.append(
                EntityKind::Node,
                7,
                sample(
                    other,
                    entity,
                    1,
                    1,
                    ScalarValue::String(Arc::from(format!("{entity}:{}", "x".repeat(4096)))),
                ),
                1,
            )?;
        }
        let unrelated = store
            .column(TemporalStore::key(EntityKind::Node, 7, other))
            .ok_or_else(|| Error::internal("column absent"))?;
        let guard = unrelated.rows.pin();
        let before = guard.get(&0).ok_or_else(|| Error::internal("row absent"))?;
        for tick in 0..256 {
            store.append(EntityKind::Node, 7, numeric(property, tick, 1), tick)?;
        }
        let same = guard.get(&0).ok_or_else(|| Error::internal("row absent"))?;
        assert!(std::ptr::eq(before, same));
        assert_eq!(unrelated.rows.len(), 4096);
        assert_eq!(
            store
                .rollup_buckets("hourly", 9, 0, 300)?
                .iter()
                .map(|b| b.count)
                .sum::<u64>(),
            256
        );
        Ok(())
    }

    #[test]
    fn temporal_concurrent_append_retention_never_returns_torn_samples() -> Result<()> {
        let store = TemporalStore::default();
        let property = PropertyId(6);
        store.declare(declaration(property, TemporalType::String, 32), 0)?;
        let writer_store = store.clone();
        let writer = std::thread::spawn(move || -> Result<()> {
            for tick in 0..2000_i64 {
                writer_store.append(
                    EntityKind::Node,
                    7,
                    sample(
                        property,
                        9,
                        tick,
                        tick as u64,
                        ScalarValue::String(Arc::from(format!("{tick}:{}", "x".repeat(4096)))),
                    ),
                    tick,
                )?;
            }
            Ok(())
        });
        let mut reads = 0;
        while !writer.is_finished() {
            for sample in store.history(EntityKind::Node, 7, 9, property, 0, i64::MAX, u64::MAX)? {
                let ScalarValue::String(text) = sample.value else {
                    return Err(Error::internal("torn sample type"));
                };
                assert!(text.starts_with(&format!("{}:", sample.event_time_nanos)));
                assert_eq!(sample.event_time_nanos as u64, sample.sequence_index);
                assert_eq!(sample.entity_id, 9);
                assert_eq!(sample.property, property);
            }
            reads += 1;
        }
        writer
            .join()
            .map_err(|_| Error::internal("writer panicked"))??;
        assert!(reads > 0);
        assert_eq!(
            store
                .current(EntityKind::Node, 7, 9, property)
                .map(|s| s.event_time_nanos),
            Some(1999)
        );
        Ok(())
    }
}
