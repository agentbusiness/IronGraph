//! Scalar event-time histories, point projection, windows, and rebuildable rollups.

use std::{collections::BTreeMap, sync::Arc};

use bitflags::bitflags;
use chrono::{Datelike, LocalResult, NaiveDate, TimeZone, Timelike, Utc};
use chrono_tz::Tz;
use serde::{Deserialize, Serialize};

use crate::{
    Error, ErrorCode, Result, ScalarValue,
    types::{EntityKind, PropertyId},
};

use super::{
    PropertyColumns,
    persistent::{PagedVec, PersistentMap},
};

/// Declared scalar type of a temporal property.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TemporalType {
    Boolean,
    Integer,
    Float,
    String,
    Date,
    LocalTime,
    ZonedTime,
    LocalDateTime,
    ZonedDateTime,
    Duration,
}

impl TemporalType {
    #[must_use]
    pub fn accepts(self, value: &ScalarValue) -> bool {
        matches!(value, ScalarValue::Null)
            || matches!(
                (self, value),
                (Self::Boolean, ScalarValue::Boolean(_))
                    | (Self::Integer, ScalarValue::Integer(_))
                    | (Self::Float, ScalarValue::Float(_))
                    | (Self::String, ScalarValue::String(_))
                    | (Self::Date, ScalarValue::Date(_))
                    | (Self::LocalTime, ScalarValue::LocalTime(_))
                    | (Self::ZonedTime, ScalarValue::ZonedTime { .. })
                    | (Self::LocalDateTime, ScalarValue::LocalDateTime { .. })
                    | (Self::ZonedDateTime, ScalarValue::ZonedDateTime { .. })
                    | (Self::Duration, ScalarValue::Duration { .. })
            )
    }
}

impl TryFrom<&str> for TemporalType {
    type Error = Error;

    fn try_from(value: &str) -> Result<Self> {
        match value.to_ascii_uppercase().as_str() {
            "BOOLEAN" => Ok(Self::Boolean),
            "INTEGER" => Ok(Self::Integer),
            "FLOAT" => Ok(Self::Float),
            "STRING" => Ok(Self::String),
            "DATE" => Ok(Self::Date),
            "LOCAL TIME" => Ok(Self::LocalTime),
            "ZONED TIME" => Ok(Self::ZonedTime),
            "LOCAL DATETIME" => Ok(Self::LocalDateTime),
            "ZONED DATETIME" => Ok(Self::ZonedDateTime),
            "DURATION" => Ok(Self::Duration),
            _ => Err(Error::new(
                ErrorCode::QueryType,
                "temporal declaration requires an allowed scalar type",
            )),
        }
    }
}

impl TryFrom<String> for TemporalType {
    type Error = Error;

    fn try_from(value: String) -> Result<Self> {
        Self::try_from(value.as_str())
    }
}

/// Target-qualified temporal schema declaration.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalDeclaration {
    pub entity_kind: EntityKind,
    /// Stable label ID for nodes or relationship-type ID for relationships.
    pub target: u64,
    pub property: PropertyId,
    pub value_type: TemporalType,
    /// Canonical retention horizon in nanoseconds.
    pub retention_nanos: i64,
}

/// Canonical scalar sample, ordered by event time then publication sequence.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalSample {
    pub entity_id: u64,
    pub property: PropertyId,
    pub event_time_nanos: i64,
    pub sequence_index: u64,
    pub value: ScalarValue,
}

/// Bounded planner-facing summary for one target-qualified temporal column.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TemporalOptimizerColumnStatistics {
    pub entity_kind: EntityKind,
    pub target: u64,
    pub property: PropertyId,
    pub segment_count: u64,
    pub sample_count: u64,
    pub minimum_event_time_nanos: Option<i64>,
    pub maximum_event_time_nanos: Option<i64>,
    pub compatible_rollups: u64,
    pub resident_bytes: u64,
}

/// Typed, allocation-free temporal value columns used to build one resident device image.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalDeviceValues {
    Boolean(Vec<u8>),
    Integer(Vec<i64>),
    /// Exact IEEE-754 bits; Metal kernels preserve F64 semantics through the canonical encoding.
    FloatBits(Vec<u64>),
    String {
        offsets: Vec<u32>,
        bytes: Vec<u8>,
    },
    Date(Vec<i64>),
    LocalTime(Vec<i64>),
    ZonedTime {
        nanos: Vec<i64>,
        offsets: Vec<i32>,
    },
    LocalDateTime {
        seconds: Vec<i64>,
        nanos: Vec<u32>,
    },
    ZonedDateTime {
        seconds: Vec<i64>,
        nanos: Vec<u32>,
        timezone_offsets: Vec<u32>,
        timezone_bytes: Vec<u8>,
    },
    Duration {
        months: Vec<i64>,
        days: Vec<i64>,
        seconds: Vec<i64>,
        nanos: Vec<i32>,
    },
}

impl TemporalDeviceValues {
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        match self {
            Self::Boolean(values) => values.len(),
            Self::Integer(values) | Self::LocalTime(values) => {
                values.len().saturating_mul(size_of::<i64>())
            }
            Self::FloatBits(values) => values.len().saturating_mul(size_of::<u64>()),
            Self::String { offsets, bytes } => offsets
                .len()
                .saturating_mul(size_of::<u32>())
                .saturating_add(bytes.len()),
            Self::Date(values) => values.len().saturating_mul(size_of::<i64>()),
            Self::ZonedTime { nanos, offsets } => nanos
                .len()
                .saturating_mul(size_of::<i64>())
                .saturating_add(offsets.len().saturating_mul(size_of::<i32>())),
            Self::LocalDateTime { seconds, nanos } => seconds
                .len()
                .saturating_mul(size_of::<i64>())
                .saturating_add(nanos.len().saturating_mul(size_of::<u32>())),
            Self::ZonedDateTime {
                seconds,
                nanos,
                timezone_offsets,
                timezone_bytes,
            } => seconds
                .len()
                .saturating_mul(size_of::<i64>())
                .saturating_add(nanos.len().saturating_mul(size_of::<u32>()))
                .saturating_add(timezone_offsets.len().saturating_mul(size_of::<u32>()))
                .saturating_add(timezone_bytes.len()),
            Self::Duration {
                months,
                days,
                seconds,
                nanos,
            } => months
                .len()
                .saturating_mul(size_of::<i64>())
                .saturating_add(days.len().saturating_mul(size_of::<i64>()))
                .saturating_add(seconds.len().saturating_mul(size_of::<i64>()))
                .saturating_add(nanos.len().saturating_mul(size_of::<i32>())),
        }
    }
}

/// One entity-kind-separated temporal column in deterministic device row order.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalDeviceColumn {
    pub entity_kind: EntityKind,
    pub target: u64,
    pub property: PropertyId,
    pub value_type: TemporalType,
    pub entity_ids: Vec<u64>,
    pub event_times_nanos: Vec<i64>,
    pub sequence_indexes: Vec<u64>,
    pub validity: Vec<u8>,
    pub values: TemporalDeviceValues,
    pub current_entity_ids: Vec<u64>,
    pub current_rows: Vec<u32>,
}

impl TemporalDeviceColumn {
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.entity_ids
            .len()
            .saturating_mul(size_of::<u64>())
            .saturating_add(
                self.event_times_nanos
                    .len()
                    .saturating_mul(size_of::<i64>()),
            )
            .saturating_add(self.sequence_indexes.len().saturating_mul(size_of::<u64>()))
            .saturating_add(self.validity.len())
            .saturating_add(self.values.resident_bytes())
            .saturating_add(
                self.current_entity_ids
                    .len()
                    .saturating_mul(size_of::<u64>()),
            )
            .saturating_add(self.current_rows.len().saturating_mul(size_of::<u32>()))
    }
}

/// Materialized rollup columns admitted with the canonical histories they accelerate.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RollupDeviceColumn {
    pub name: String,
    pub entity_ids: Vec<u64>,
    pub starts_nanos: Vec<i64>,
    pub ends_nanos: Vec<i64>,
    pub counts: Vec<u64>,
    pub sums: Vec<f64>,
    pub minimums: Vec<f64>,
    pub maximums: Vec<f64>,
    pub averages: Vec<f64>,
    pub aggregate_validity: Vec<u8>,
}

impl RollupDeviceColumn {
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        let rows = self.entity_ids.len();
        rows.saturating_mul(
            size_of::<u64>()
                + 2 * size_of::<i64>()
                + size_of::<u64>()
                + 4 * size_of::<f64>()
                + size_of::<u8>(),
        )
    }
}

/// Complete retained temporal and rollup image for one project.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct TemporalDeviceImage {
    pub columns: Vec<TemporalDeviceColumn>,
    pub rollups: Vec<RollupDeviceColumn>,
}

impl TemporalDeviceImage {
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.columns
            .iter()
            .map(TemporalDeviceColumn::resident_bytes)
            .chain(self.rollups.iter().map(RollupDeviceColumn::resident_bytes))
            .fold(0_usize, usize::saturating_add)
    }
}

#[derive(Clone, Debug, Default)]
struct TemporalColumn {
    property: PropertyId,
    entity_ids: PagedVec<u64>,
    event_times_nanos: PagedVec<i64>,
    sequence_indexes: PagedVec<u64>,
    values: PropertyColumns,
    current: PersistentMap<usize>,
}

impl TemporalColumn {
    fn with_property(property: PropertyId) -> Self {
        Self {
            property,
            ..Self::default()
        }
    }

    fn len(&self) -> usize {
        self.entity_ids.len()
    }

    fn is_empty(&self) -> bool {
        self.entity_ids.is_empty()
    }

    fn sample(&self, row: usize) -> Option<TemporalSample> {
        Some(TemporalSample {
            entity_id: *self.entity_ids.get(row)?,
            property: self.property,
            event_time_nanos: *self.event_times_nanos.get(row)?,
            sequence_index: *self.sequence_indexes.get(row)?,
            value: self
                .values
                .get(u32::try_from(row).ok()?, self.property)
                .unwrap_or(ScalarValue::Null),
        })
    }

    fn samples(&self) -> impl Iterator<Item = TemporalSample> + '_ {
        (0..self.len()).filter_map(|row| self.sample(row))
    }

    fn push(&mut self, sample: &TemporalSample) -> Result<usize> {
        if sample.property != self.property {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "temporal sample targets the wrong flat column",
            ));
        }
        let row = self.len();
        self.values
            .push_row(&[(self.property, sample.value.clone())])?;
        self.entity_ids.push(sample.entity_id);
        self.event_times_nanos.push(sample.event_time_nanos);
        self.sequence_indexes.push(sample.sequence_index);
        Ok(row)
    }

    fn replace_samples(&mut self, samples: impl IntoIterator<Item = TemporalSample>) -> Result<()> {
        let property = self.property;
        *self = Self::with_property(property);
        for sample in samples {
            self.push(&sample)?;
        }
        rebuild_current(self);
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
struct PersistedTemporalColumn {
    base: Arc<Vec<TemporalSample>>,
    delta: PagedVec<TemporalSample>,
    current: PersistentMap<TemporalSample>,
}

impl Serialize for TemporalColumn {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        let mut current = PersistentMap::default();
        for (entity, row) in self.current.iter() {
            if let Some(sample) = self.sample(*row) {
                current.insert(entity, sample);
            }
        }
        PersistedTemporalColumn {
            base: Arc::new(self.samples().collect()),
            delta: PagedVec::default(),
            current,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for TemporalColumn {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let persisted = PersistedTemporalColumn::deserialize(deserializer)?;
        let property = persisted
            .base
            .first()
            .or_else(|| persisted.delta.get(0))
            .map(|sample| sample.property)
            .unwrap_or_default();
        let mut column = Self::with_property(property);
        for sample in persisted.base.iter().chain(persisted.delta.iter()) {
            column.push(sample).map_err(serde::de::Error::custom)?;
        }
        rebuild_current(&mut column);
        Ok(column)
    }
}

/// Backend-neutral flat canonical temporal column. A Metal backend rebinds these fields to the
/// exact shared buffers used by its kernels; declarations and rollups remain compact control and
/// rebuildable derived state in `TemporalStore`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TemporalCanonicalColumn {
    pub entity_kind: EntityKind,
    pub target: u64,
    pub property: PropertyId,
    pub value_type: TemporalType,
    pub entity_ids: PagedVec<u64>,
    pub event_times_nanos: PagedVec<i64>,
    pub sequence_indexes: PagedVec<u64>,
    pub values: PropertyColumns,
}

impl TemporalCanonicalColumn {
    /// Bytes this staging-only canonical view holds while a project is being admitted.
    ///
    /// The canonical columns are built alongside the device image and released as soon as a
    /// backend has taken what it needs, so they are deliberately absent from
    /// `ResidentProjectImage::resident_bytes`. They are still live at the moment admission
    /// reserves, though, so the reservation has to cover them or the transient peak exceeds what
    /// the governor believes it admitted. The fixed-width columns are measured exactly; the value
    /// column reports its own footprint the same way the device columns do.
    #[must_use]
    pub fn staging_bytes(&self) -> usize {
        self.entity_ids
            .len()
            .saturating_mul(size_of::<u64>())
            .saturating_add(
                self.event_times_nanos
                    .len()
                    .saturating_mul(size_of::<i64>()),
            )
            .saturating_add(self.sequence_indexes.len().saturating_mul(size_of::<u64>()))
            .saturating_add(self.values.estimated_bytes())
    }

    fn from_column(
        entity_kind: EntityKind,
        target: u64,
        value_type: TemporalType,
        column: &TemporalColumn,
    ) -> Result<Self> {
        let mut samples = column.samples().collect::<Vec<_>>();
        samples.sort_by_key(|sample| {
            (
                sample.entity_id,
                sample.event_time_nanos,
                sample.sequence_index,
            )
        });
        let mut flat = TemporalColumn::with_property(column.property);
        for sample in samples {
            flat.push(&sample)?;
        }
        Ok(Self {
            entity_kind,
            target,
            property: column.property,
            value_type,
            entity_ids: flat.entity_ids,
            event_times_nanos: flat.event_times_nanos,
            sequence_indexes: flat.sequence_indexes,
            values: flat.values,
        })
    }

    #[cfg_attr(
        not(all(feature = "accelerator", any(target_os = "macos", target_os = "ios"))),
        allow(dead_code)
    )]
    pub fn len(&self) -> usize {
        self.entity_ids.len()
    }

    #[cfg_attr(
        not(all(feature = "accelerator", any(target_os = "macos", target_os = "ios"))),
        allow(dead_code)
    )]
    pub fn push_and_sort(&mut self, sample: &TemporalSample) -> Result<()> {
        self.extend_and_sort(std::slice::from_ref(sample))
    }

    /// Applies one committed batch with a single base scan and a single sort. The delta path is
    /// intentionally column-batched: applying `k` samples must never reconstruct the `n` existing
    /// rows `k` times. The caller groups target-qualified samples before invoking this method.
    #[cfg_attr(
        not(all(feature = "accelerator", any(target_os = "macos", target_os = "ios"))),
        allow(dead_code)
    )]
    pub fn extend_and_sort(&mut self, additions: &[TemporalSample]) -> Result<()> {
        if additions.is_empty() {
            return Ok(());
        }
        if additions
            .iter()
            .any(|sample| sample.property != self.property)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "temporal delta targets the wrong shared column",
            ));
        }
        let mut samples = (0..self.len())
            .map(|row| {
                Ok(TemporalSample {
                    entity_id: *self
                        .entity_ids
                        .get(row)
                        .ok_or_else(|| Error::internal("shared temporal entity row disappeared"))?,
                    property: self.property,
                    event_time_nanos: *self.event_times_nanos.get(row).ok_or_else(|| {
                        Error::internal("shared temporal event-time row disappeared")
                    })?,
                    sequence_index: *self.sequence_indexes.get(row).ok_or_else(|| {
                        Error::internal("shared temporal sequence row disappeared")
                    })?,
                    value: self
                        .values
                        .get(
                            u32::try_from(row).map_err(|_| {
                                Error::new(
                                    ErrorCode::ResultBudgetExceeded,
                                    "temporal row exceeds u32",
                                )
                            })?,
                            self.property,
                        )
                        .unwrap_or(ScalarValue::Null),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        samples.extend(additions.iter().cloned());
        samples.sort_by_key(|sample| {
            (
                sample.entity_id,
                sample.event_time_nanos,
                sample.sequence_index,
            )
        });
        let mut flat = TemporalColumn::with_property(self.property);
        for sample in samples {
            flat.push(&sample)?;
        }
        self.entity_ids = flat.entity_ids;
        self.event_times_nanos = flat.event_times_nanos;
        self.sequence_indexes = flat.sequence_indexes;
        self.values = flat.values;
        Ok(())
    }
}

bitflags! {
    /// Aggregates materialized for a rollup bucket.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
    pub struct AggregateSet: u8 {
        const AVG = 1 << 0;
        const MIN = 1 << 1;
        const MAX = 1 << 2;
        const COUNT = 1 << 3;
        const SUM = 1 << 4;
        const ALL = Self::AVG.bits() | Self::MIN.bits() | Self::MAX.bits()
            | Self::COUNT.bits() | Self::SUM.bits();
    }
}

impl Serialize for AggregateSet {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_u8(self.bits())
    }
}

impl<'de> Deserialize<'de> for AggregateSet {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let bits = u8::deserialize(deserializer)?;
        Self::from_bits(bits)
            .ok_or_else(|| serde::de::Error::custom("invalid temporal aggregate bitset"))
    }
}

/// Fixed elapsed-time or calendar-aligned window shape.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum WindowKind {
    Fixed { width_nanos: i64, every_nanos: i64 },
    Calendar { months: u32, days: u32 },
}

/// Deterministic window specification. Ranges and buckets are half-open.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WindowSpec {
    pub kind: WindowKind,
    pub align_nanos: i64,
    pub timezone: Option<String>,
    pub emit_empty: bool,
    pub aggregates: AggregateSet,
}

impl WindowSpec {
    #[must_use]
    pub fn tumbling(width_nanos: i64) -> Self {
        Self {
            kind: WindowKind::Fixed {
                width_nanos,
                every_nanos: width_nanos,
            },
            align_nanos: 0,
            timezone: None,
            emit_empty: false,
            aggregates: AggregateSet::ALL,
        }
    }

    #[must_use]
    pub fn hopping(width_nanos: i64, every_nanos: i64) -> Self {
        Self {
            kind: WindowKind::Fixed {
                width_nanos,
                every_nanos,
            },
            align_nanos: 0,
            timezone: None,
            emit_empty: false,
            aggregates: AggregateSet::ALL,
        }
    }

    fn validate(&self) -> Result<()> {
        match self.kind {
            WindowKind::Fixed {
                width_nanos,
                every_nanos,
            } => {
                if width_nanos <= 0 || every_nanos <= 0 || every_nanos > width_nanos {
                    return Err(Error::new(
                        ErrorCode::TemporalRange,
                        "window width and EVERY must be positive and EVERY cannot exceed width",
                    ));
                }
                if let Some(zone) = self.timezone.as_deref() {
                    zone.parse::<Tz>().map_err(|_| {
                        Error::new(ErrorCode::TemporalRange, "invalid IANA time zone")
                    })?;
                }
            }
            WindowKind::Calendar { months, days } => {
                if (months == 0 && days == 0) || (months != 0 && days != 0) {
                    return Err(Error::new(
                        ErrorCode::TemporalRange,
                        "calendar window requires exactly one positive month or day component",
                    ));
                }
                let Some(zone) = self.timezone.as_deref() else {
                    return Err(Error::new(
                        ErrorCode::TemporalRange,
                        "calendar windows require an explicit IANA time zone",
                    ));
                };
                zone.parse::<Tz>()
                    .map_err(|_| Error::new(ErrorCode::TemporalRange, "invalid IANA time zone"))?;
            }
        }
        Ok(())
    }
}

/// One aggregate bucket. Null numeric values do not contribute.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RollupBucket {
    pub start_nanos: i64,
    pub end_nanos: i64,
    pub count: u64,
    pub sum: Option<f64>,
    pub min: Option<f64>,
    pub max: Option<f64>,
    pub avg: Option<f64>,
}

/// Durable definition of a rebuildable temporal accelerator.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TemporalRollupDefinition {
    pub name: String,
    pub entity_kind: EntityKind,
    /// Stable label ID for nodes or relationship-type ID for relationships.
    pub target: u64,
    pub property: PropertyId,
    pub window: WindowSpec,
    pub aggregates: AggregateSet,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct MaterializedRollup {
    definition: TemporalRollupDefinition,
    /// Entity ID and bucket start identify a bucket. End is determined by the definition.
    buckets: Arc<BTreeMap<(u64, i64), RollupBucket>>,
    /// Current bucket replacements since the last rebuild. Rollups are derived, so this bounded
    /// persistent overlay is deliberately rebuilt rather than checkpointed.
    #[serde(default, skip)]
    bucket_overrides: PersistentMap<RollupBucket>,
}

impl MaterializedRollup {
    #[must_use]
    fn bucket(&self, entity_id: u64, start_nanos: i64) -> Option<&RollupBucket> {
        self.bucket_overrides
            .get(rollup_bucket_key(entity_id, start_nanos))
            .or_else(|| self.buckets.get(&(entity_id, start_nanos)))
    }

    fn insert_bucket(&mut self, entity_id: u64, bucket: RollupBucket) {
        self.bucket_overrides
            .insert(rollup_bucket_key(entity_id, bucket.start_nanos), bucket);
    }

    fn merged_buckets(&self) -> BTreeMap<(u64, i64), &RollupBucket> {
        let mut merged = self
            .buckets
            .iter()
            .map(|(key, bucket)| (*key, bucket))
            .collect::<BTreeMap<_, _>>();
        for (key, bucket) in self.bucket_overrides.iter() {
            merged.insert(rollup_bucket_parts(key), bucket);
        }
        merged
    }
}

impl RollupBucket {
    fn empty(start_nanos: i64, end_nanos: i64) -> Self {
        Self {
            start_nanos,
            end_nanos,
            count: 0,
            sum: None,
            min: None,
            max: None,
            avg: None,
        }
    }

    fn add(&mut self, value: &ScalarValue) {
        let number = match value {
            ScalarValue::Integer(value) => Some(*value as f64),
            ScalarValue::Float(value) => Some(value.into_inner()),
            _ => None,
        };
        let Some(number) = number else { return };
        self.count = self.count.saturating_add(1);
        self.sum = Some(self.sum.unwrap_or(0.0) + number);
        self.min = Some(self.min.map_or(number, |old| old.min(number)));
        self.max = Some(self.max.map_or(number, |old| old.max(number)));
        self.avg = self.sum.map(|sum| sum / self.count as f64);
    }
}

/// Separate node/relationship temporal families sharing one implementation.
#[derive(Clone, Debug, Default, Serialize)]
pub struct TemporalStore {
    declarations: Arc<BTreeMap<(u8, u64, PropertyId), TemporalDeclaration>>,
    columns: BTreeMap<(u8, u64, PropertyId), Arc<TemporalColumn>>,
    rollups: BTreeMap<String, Arc<MaterializedRollup>>,
}

#[derive(Deserialize)]
struct PersistedTemporalStore {
    declarations: Arc<BTreeMap<(u8, u64, PropertyId), TemporalDeclaration>>,
    columns: BTreeMap<(u8, u64, PropertyId), Arc<TemporalColumn>>,
    rollups: BTreeMap<String, Arc<MaterializedRollup>>,
}

impl<'de> Deserialize<'de> for TemporalStore {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let persisted = PersistedTemporalStore::deserialize(deserializer)?;
        let mut store = Self {
            declarations: persisted.declarations,
            columns: persisted.columns,
            rollups: persisted.rollups,
        };
        store
            .validate_and_rebuild_derived()
            .map_err(serde::de::Error::custom)?;
        Ok(store)
    }
}

impl TemporalStore {
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.declarations.is_empty() && self.columns.is_empty() && self.rollups.is_empty()
    }

    pub fn canonical_columns(&self) -> Result<Vec<TemporalCanonicalColumn>> {
        self.columns
            .iter()
            .map(|((kind, target, property), column)| {
                let entity_kind = match *kind {
                    value if value == EntityKind::Node as u8 => EntityKind::Node,
                    value if value == EntityKind::Relationship as u8 => EntityKind::Relationship,
                    _ => {
                        return Err(Error::new(
                            ErrorCode::CorruptStorage,
                            "temporal column has an invalid entity kind",
                        ));
                    }
                };
                if *property != column.property {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        "temporal column property differs from its key",
                    ));
                }
                let declaration = self
                    .declarations
                    .get(&(*kind, *target, *property))
                    .ok_or_else(|| {
                        Error::new(
                            ErrorCode::CorruptStorage,
                            "temporal column has no declaration",
                        )
                    })?;
                TemporalCanonicalColumn::from_column(
                    entity_kind,
                    *target,
                    declaration.value_type,
                    column,
                )
            })
            .collect()
    }

    pub fn rebind_shared(&mut self, columns: Vec<TemporalCanonicalColumn>) -> Result<()> {
        if columns.len() != self.columns.len() {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "shared temporal backing set is incomplete",
            ));
        }
        for backing in columns {
            let key = (backing.entity_kind as u8, backing.target, backing.property);
            let row_count = backing.entity_ids.len();
            if self
                .declarations
                .get(&key)
                .is_none_or(|declaration| declaration.value_type != backing.value_type)
                || row_count != backing.event_times_nanos.len()
                || row_count != backing.sequence_indexes.len()
                || row_count != backing.values.rows()
            {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "shared temporal column lengths differ",
                ));
            }
            let column = self.columns.get_mut(&key).ok_or_else(|| {
                Error::new(
                    ErrorCode::CorruptStorage,
                    "shared temporal backing targets an unknown column",
                )
            })?;
            let column = Arc::make_mut(column);
            column.property = backing.property;
            column.entity_ids = backing.entity_ids;
            column.event_times_nanos = backing.event_times_nanos;
            column.sequence_indexes = backing.sequence_indexes;
            column.values = backing.values;
            rebuild_current(column);
        }
        Ok(())
    }

    /// Returns rebuildable cost summaries without materializing a device image or bucket map.
    #[must_use]
    pub fn optimizer_statistics(&self) -> Vec<TemporalOptimizerColumnStatistics> {
        self.columns
            .iter()
            .filter_map(|(key, column)| {
                let declaration = self.declarations.get(key)?;
                let minimum = column.samples().map(|sample| sample.event_time_nanos).min();
                let maximum = column.samples().map(|sample| sample.event_time_nanos).max();
                let sample_count = column.len();
                let segment_count = usize::from(!column.is_empty());
                let compatible_rollups = self
                    .rollups
                    .values()
                    .filter(|rollup| {
                        rollup.definition.entity_kind == declaration.entity_kind
                            && rollup.definition.target == declaration.target
                            && rollup.definition.property == declaration.property
                    })
                    .count();
                // Entity, property, event time, publication order, validity, and a conservative
                // tagged scalar width. Cost estimates may overstate; admission never understates.
                let resident_bytes = (sample_count as u64).saturating_mul(48);
                Some(TemporalOptimizerColumnStatistics {
                    entity_kind: declaration.entity_kind,
                    target: declaration.target,
                    property: declaration.property,
                    segment_count: segment_count as u64,
                    sample_count: sample_count as u64,
                    minimum_event_time_nanos: minimum,
                    maximum_event_time_nanos: maximum,
                    compatible_rollups: compatible_rollups as u64,
                    resident_bytes,
                })
            })
            .collect()
    }

    /// Builds the exact typed image admitted by CPU and accelerator backends.
    pub fn device_image(&self) -> Result<TemporalDeviceImage> {
        let mut columns = Vec::with_capacity(self.columns.len());
        for (key, column) in &self.columns {
            let declaration = self.declarations.get(key).ok_or_else(|| {
                Error::new(
                    ErrorCode::CorruptStorage,
                    "temporal column has no target-qualified declaration",
                )
            })?;
            let mut samples = column.samples().collect::<Vec<_>>();
            samples.sort_by_key(|sample| {
                (
                    sample.entity_id,
                    sample.event_time_nanos,
                    sample.sequence_index,
                )
            });
            let (validity, values) =
                encode_temporal_device_values(declaration.value_type, &samples)?;
            let mut positions = BTreeMap::new();
            for (row, sample) in samples.iter().enumerate() {
                positions.insert(
                    (
                        sample.entity_id,
                        sample.event_time_nanos,
                        sample.sequence_index,
                    ),
                    u32::try_from(row).map_err(|_| {
                        Error::new(
                            ErrorCode::ResultBudgetExceeded,
                            "temporal device row exceeds u32",
                        )
                    })?,
                );
            }
            let current_count = column.current.iter().count();
            let mut current_entity_ids = Vec::with_capacity(current_count);
            let mut current_rows = Vec::with_capacity(current_count);
            for (entity, current_row) in column.current.iter() {
                let entity = u64::try_from(entity).map_err(|_| {
                    Error::new(
                        ErrorCode::CorruptStorage,
                        "temporal current entity exceeds u64",
                    )
                })?;
                let current = column.sample(*current_row).ok_or_else(|| {
                    Error::new(
                        ErrorCode::CorruptStorage,
                        "temporal current row is outside canonical history",
                    )
                })?;
                let row = positions
                    .get(&(
                        current.entity_id,
                        current.event_time_nanos,
                        current.sequence_index,
                    ))
                    .copied()
                    .ok_or_else(|| {
                        Error::new(
                            ErrorCode::CorruptStorage,
                            "temporal current row is absent from canonical history",
                        )
                    })?;
                current_entity_ids.push(entity);
                current_rows.push(row);
            }
            columns.push(TemporalDeviceColumn {
                entity_kind: declaration.entity_kind,
                target: declaration.target,
                property: declaration.property,
                value_type: declaration.value_type,
                entity_ids: samples.iter().map(|sample| sample.entity_id).collect(),
                event_times_nanos: samples
                    .iter()
                    .map(|sample| sample.event_time_nanos)
                    .collect(),
                sequence_indexes: samples.iter().map(|sample| sample.sequence_index).collect(),
                validity,
                values,
                current_entity_ids,
                current_rows,
            });
        }

        let mut rollups = Vec::with_capacity(self.rollups.len());
        for (name, rollup) in &self.rollups {
            let buckets = rollup.merged_buckets();
            let mut resident = RollupDeviceColumn {
                name: name.clone(),
                entity_ids: Vec::with_capacity(buckets.len()),
                starts_nanos: Vec::with_capacity(buckets.len()),
                ends_nanos: Vec::with_capacity(buckets.len()),
                counts: Vec::with_capacity(buckets.len()),
                sums: Vec::with_capacity(buckets.len()),
                minimums: Vec::with_capacity(buckets.len()),
                maximums: Vec::with_capacity(buckets.len()),
                averages: Vec::with_capacity(buckets.len()),
                aggregate_validity: Vec::with_capacity(buckets.len()),
            };
            for ((entity, _), bucket) in buckets {
                resident.entity_ids.push(entity);
                resident.starts_nanos.push(bucket.start_nanos);
                resident.ends_nanos.push(bucket.end_nanos);
                resident.counts.push(bucket.count);
                resident.sums.push(bucket.sum.unwrap_or(0.0));
                resident.minimums.push(bucket.min.unwrap_or(0.0));
                resident.maximums.push(bucket.max.unwrap_or(0.0));
                resident.averages.push(bucket.avg.unwrap_or(0.0));
                resident.aggregate_validity.push(
                    u8::from(bucket.sum.is_some())
                        | (u8::from(bucket.min.is_some()) << 1)
                        | (u8::from(bucket.max.is_some()) << 2)
                        | (u8::from(bucket.avg.is_some()) << 3),
                );
            }
            rollups.push(resident);
        }
        Ok(TemporalDeviceImage { columns, rollups })
    }

    #[must_use]
    pub fn is_declared(&self, entity_kind: EntityKind, target: u64, property: PropertyId) -> bool {
        self.declarations
            .contains_key(&(entity_kind as u8, target, property))
    }

    /// Whether any declaration anywhere covers this property, for any entity kind and target.
    ///
    /// A caller that only needs to know "is this property remembered" cannot resolve a target
    /// first: it may be looking at an expression whose binding has not been narrowed to one label.
    /// This is the conservative question, and answering it costs one scan of the declaration keys.
    #[must_use]
    pub fn declares_property(&self, property: PropertyId) -> bool {
        self.declarations
            .keys()
            .any(|(_, _, declared)| *declared == property)
    }

    #[must_use]
    pub fn declared_type(
        &self,
        entity_kind: EntityKind,
        target: u64,
        property: PropertyId,
    ) -> Option<TemporalType> {
        self.declarations
            .get(&(entity_kind as u8, target, property))
            .map(|declaration| declaration.value_type)
    }

    /// Resolves a temporal declaration for one entity. Multiple matching node labels are
    /// deliberately rejected because they would make the physical column and retention rule
    /// ambiguous.
    pub fn resolve_target(
        &self,
        entity_kind: EntityKind,
        targets: &[u64],
        property: PropertyId,
    ) -> Result<Option<u64>> {
        let mut matches = targets
            .iter()
            .copied()
            .filter(|target| self.is_declared(entity_kind, *target, property));
        let first = matches.next();
        if matches.next().is_some() {
            return Err(Error::new(
                ErrorCode::QueryType,
                "temporal property matches more than one entity target",
            ));
        }
        Ok(first)
    }

    /// Applies a target-qualified declaration and any retention shortening at one resolved
    /// commit boundary.
    pub fn declare(
        &mut self,
        declaration: TemporalDeclaration,
        resolved_commit_time_nanos: i64,
    ) -> Result<()> {
        if declaration.retention_nanos <= 0 {
            return Err(Error::new(
                ErrorCode::TemporalRange,
                "retention must be positive",
            ));
        }
        let key = (
            declaration.entity_kind as u8,
            declaration.target,
            declaration.property,
        );
        if let Some(previous) = self.declarations.get(&key)
            && previous.value_type != declaration.value_type
            && self
                .columns
                .get(&key)
                .is_some_and(|column| !column.is_empty())
        {
            return Err(Error::new(
                ErrorCode::QueryType,
                "a populated temporal property cannot change scalar type",
            ));
        }
        Arc::make_mut(&mut self.declarations).insert(key, declaration);
        self.columns
            .entry(key)
            .or_insert_with(|| Arc::new(TemporalColumn::with_property(key.2)));
        self.expire_key(key, resolved_commit_time_nanos)?;
        self.rebuild_rollups_for_key(key)?;
        Ok(())
    }

    pub fn append(
        &mut self,
        entity_kind: EntityKind,
        target: u64,
        sample: TemporalSample,
        resolved_commit_time_nanos: i64,
    ) -> Result<()> {
        self.validate_append(entity_kind, target, &sample, resolved_commit_time_nanos)?;
        let key = (entity_kind as u8, target, sample.property);
        let column = self
            .columns
            .get_mut(&key)
            .ok_or_else(|| Error::internal("temporal declaration has no corresponding column"))?;
        let column = Arc::make_mut(column);
        let replace_current = column
            .current
            .get(u128::from(sample.entity_id))
            .and_then(|row| column.sample(*row))
            .is_none_or(|current| {
                (sample.event_time_nanos, sample.sequence_index)
                    > (current.event_time_nanos, current.sequence_index)
            });
        let row = column.push(&sample)?;
        if replace_current {
            column.current.insert(u128::from(sample.entity_id), row);
        }
        self.update_rollups_for_sample(key, &sample)?;
        Ok(())
    }

    pub fn validate_append(
        &self,
        entity_kind: EntityKind,
        target: u64,
        sample: &TemporalSample,
        resolved_commit_time_nanos: i64,
    ) -> Result<()> {
        let key = (entity_kind as u8, target, sample.property);
        let declaration = self
            .declarations
            .get(&key)
            .ok_or_else(|| Error::new(ErrorCode::QueryType, "property is not declared temporal"))?;
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
        let oldest = resolved_commit_time_nanos.saturating_sub(declaration.retention_nanos);
        if sample.event_time_nanos < oldest {
            return Err(Error::new(
                ErrorCode::RetentionExpired,
                "temporal sample is older than the canonical retention horizon",
            ));
        }
        Ok(())
    }

    #[must_use]
    pub fn current(
        &self,
        entity_kind: EntityKind,
        target: u64,
        entity_id: u64,
        property: PropertyId,
    ) -> Option<TemporalSample> {
        let column = self.columns.get(&(entity_kind as u8, target, property))?;
        let row = *column.current.get(u128::from(entity_id))?;
        column.sample(row)
    }

    pub fn at_time(
        &self,
        entity_kind: EntityKind,
        target: u64,
        entity_id: u64,
        property: PropertyId,
        when_nanos: i64,
        bookmark_index: u64,
    ) -> Option<TemporalSample> {
        let column = self.columns.get(&(entity_kind as u8, target, property))?;
        column
            .samples()
            .filter(|sample| {
                sample.entity_id == entity_id
                    && sample.event_time_nanos <= when_nanos
                    && sample.sequence_index <= bookmark_index
            })
            .max_by_key(|sample| (sample.event_time_nanos, sample.sequence_index))
    }

    pub fn history(
        &self,
        entity_kind: EntityKind,
        target: u64,
        entity_id: u64,
        property: PropertyId,
        from_nanos: i64,
        to_nanos: i64,
        bookmark_index: u64,
    ) -> Result<Vec<TemporalSample>> {
        if from_nanos >= to_nanos {
            return Err(Error::new(
                ErrorCode::TemporalRange,
                "HISTORY range must be non-empty",
            ));
        }
        let Some(column) = self.columns.get(&(entity_kind as u8, target, property)) else {
            return Err(Error::new(
                ErrorCode::QueryType,
                "property is not declared temporal",
            ));
        };
        let mut result: Vec<_> = column
            .samples()
            .filter(|sample| {
                sample.entity_id == entity_id
                    && sample.event_time_nanos >= from_nanos
                    && sample.event_time_nanos < to_nanos
                    && sample.sequence_index <= bookmark_index
            })
            .collect();
        result.sort_by_key(|sample| (sample.event_time_nanos, sample.sequence_index));
        Ok(result)
    }

    pub fn compact(&mut self, resolved_commit_time_nanos: i64) -> Result<()> {
        let keys = self.columns.keys().copied().collect::<Vec<_>>();
        for key in &keys {
            let Some(column) = self.columns.get_mut(key) else {
                continue;
            };
            let column = Arc::make_mut(column);
            let mut samples = column.samples().collect::<Vec<_>>();
            samples.sort_by_key(|sample| {
                (
                    sample.entity_id,
                    sample.event_time_nanos,
                    sample.sequence_index,
                )
            });
            if let Some(declaration) = self.declarations.get(key) {
                let oldest = resolved_commit_time_nanos.saturating_sub(declaration.retention_nanos);
                samples.retain(|sample| sample.event_time_nanos >= oldest);
            }
            column.replace_samples(samples)?;
        }
        for key in keys {
            self.rebuild_rollups_for_key(key)?;
        }
        Ok(())
    }

    /// Creates and fully populates one materialized rollup from canonical samples.
    pub fn create_rollup(&mut self, mut definition: TemporalRollupDefinition) -> Result<()> {
        if definition.name.is_empty() || definition.name.len() > 255 {
            return Err(Error::new(
                ErrorCode::QueryType,
                "rollup name must contain 1..=255 bytes",
            ));
        }
        definition.window.validate()?;
        if definition.aggregates.is_empty() {
            return Err(Error::new(
                ErrorCode::QueryType,
                "rollup requires at least one aggregate",
            ));
        }
        let key = (
            definition.entity_kind as u8,
            definition.target,
            definition.property,
        );
        if !self.declarations.contains_key(&key) {
            return Err(Error::new(
                ErrorCode::QueryType,
                "rollup property is not declared temporal for its target",
            ));
        }
        definition.window.aggregates = definition.aggregates;
        if self.rollups.contains_key(&definition.name) {
            return Err(Error::new(
                ErrorCode::TransactionConflict,
                "rollup already exists",
            ));
        }
        let name = definition.name.clone();
        self.rollups.insert(
            name.clone(),
            Arc::new(MaterializedRollup {
                definition,
                buckets: Arc::new(BTreeMap::new()),
                bucket_overrides: PersistentMap::default(),
            }),
        );
        self.rebuild_rollup(&name)
    }

    /// Rebuilds a materialized rollup deterministically from canonical base and delta samples.
    pub fn rebuild_rollup(&mut self, name: &str) -> Result<()> {
        let definition = self
            .rollups
            .get(name)
            .map(|rollup| rollup.definition.clone())
            .ok_or_else(|| Error::new(ErrorCode::IndexUnavailable, "rollup does not exist"))?;
        let key = (
            definition.entity_kind as u8,
            definition.target,
            definition.property,
        );
        let samples = self
            .columns
            .get(&key)
            .map(|column| column.samples().collect::<Vec<_>>())
            .unwrap_or_default();
        let mut buckets = BTreeMap::new();
        for sample in &samples {
            add_sample_to_buckets(&definition.window, &mut buckets, sample)?;
        }
        let rollup = self
            .rollups
            .get_mut(name)
            .ok_or_else(|| Error::internal("rollup disappeared during rebuild"))?;
        let rollup = Arc::make_mut(rollup);
        rollup.buckets = Arc::new(buckets);
        rollup.bucket_overrides = PersistentMap::default();
        Ok(())
    }

    /// Returns deterministic materialized buckets for one entity and half-open range.
    pub fn rollup_buckets(
        &self,
        name: &str,
        entity_id: u64,
        from_nanos: i64,
        to_nanos: i64,
    ) -> Result<Vec<RollupBucket>> {
        if from_nanos >= to_nanos {
            return Err(Error::new(
                ErrorCode::TemporalRange,
                "rollup range must be non-empty",
            ));
        }
        let rollup = self
            .rollups
            .get(name)
            .ok_or_else(|| Error::new(ErrorCode::IndexUnavailable, "rollup does not exist"))?;
        let mut buckets = rollup
            .buckets
            .range((entity_id, i64::MIN)..=(entity_id, i64::MAX))
            .filter_map(|(_, bucket)| {
                (bucket.end_nanos > from_nanos && bucket.start_nanos < to_nanos)
                    .then_some((bucket.start_nanos, bucket.clone()))
            })
            .collect::<BTreeMap<_, _>>();
        for (key, bucket) in rollup.bucket_overrides.iter() {
            let (entity, _) = rollup_bucket_parts(key);
            if entity == entity_id && bucket.end_nanos > from_nanos && bucket.start_nanos < to_nanos
            {
                buckets.insert(bucket.start_nanos, bucket.clone());
            }
        }
        Ok(buckets.into_values().collect())
    }

    /// Definitions are durable schema; buckets are derived and intentionally not exposed here.
    pub fn rollup_definitions(&self) -> impl Iterator<Item = &TemporalRollupDefinition> {
        self.rollups.values().map(|rollup| &rollup.definition)
    }

    /// Finds a rollup that is exactly compatible with a requested target, window, and aggregate
    /// set. Stable name order makes selection deterministic.
    #[must_use]
    pub fn compatible_rollup(
        &self,
        entity_kind: EntityKind,
        target: u64,
        property: PropertyId,
        window: &WindowSpec,
        aggregates: AggregateSet,
    ) -> Option<&TemporalRollupDefinition> {
        self.rollups.values().find_map(|rollup| {
            let definition = &rollup.definition;
            (definition.entity_kind == entity_kind
                && definition.target == target
                && definition.property == property
                && definition.window.kind == window.kind
                && definition.window.align_nanos == window.align_nanos
                && definition.window.timezone == window.timezone
                && definition.aggregates.contains(aggregates))
            .then_some(definition)
        })
    }

    pub fn window(
        samples: &[TemporalSample],
        from_nanos: i64,
        to_nanos: i64,
        spec: &WindowSpec,
    ) -> Result<Vec<RollupBucket>> {
        if from_nanos >= to_nanos {
            return Err(Error::new(
                ErrorCode::TemporalRange,
                "window range must be non-empty",
            ));
        }
        spec.validate()?;
        let boundaries = match spec.kind {
            WindowKind::Fixed {
                width_nanos,
                every_nanos,
            } => fixed_boundaries(
                from_nanos,
                to_nanos,
                spec.align_nanos,
                width_nanos,
                every_nanos,
            )?,
            WindowKind::Calendar { months, days } => calendar_boundaries(
                from_nanos,
                to_nanos,
                spec.align_nanos,
                months,
                days,
                spec.timezone.as_deref().ok_or_else(|| {
                    Error::new(ErrorCode::TemporalRange, "calendar TIME ZONE is missing")
                })?,
            )?,
        };
        let mut buckets: Vec<_> = boundaries
            .into_iter()
            .map(|(start, end)| RollupBucket::empty(start, end))
            .collect();
        for sample in samples {
            if sample.event_time_nanos < from_nanos || sample.event_time_nanos >= to_nanos {
                continue;
            }
            for bucket in &mut buckets {
                if sample.event_time_nanos >= bucket.start_nanos
                    && sample.event_time_nanos < bucket.end_nanos
                {
                    bucket.add(&sample.value);
                }
            }
        }
        if !spec.emit_empty {
            buckets.retain(|bucket| bucket.count != 0);
        }
        Ok(buckets)
    }

    /// Returns all deterministic half-open boundaries intersecting a requested range.
    pub fn window_boundaries(
        from_nanos: i64,
        to_nanos: i64,
        spec: &WindowSpec,
    ) -> Result<Vec<(i64, i64)>> {
        if from_nanos >= to_nanos {
            return Err(Error::new(
                ErrorCode::TemporalRange,
                "window range must be non-empty",
            ));
        }
        spec.validate()?;
        match spec.kind {
            WindowKind::Fixed {
                width_nanos,
                every_nanos,
            } => fixed_boundaries(
                from_nanos,
                to_nanos,
                spec.align_nanos,
                width_nanos,
                every_nanos,
            ),
            WindowKind::Calendar { months, days } => calendar_boundaries(
                from_nanos,
                to_nanos,
                spec.align_nanos,
                months,
                days,
                spec.timezone.as_deref().ok_or_else(|| {
                    Error::new(ErrorCode::TemporalRange, "calendar TIME ZONE is missing")
                })?,
            ),
        }
    }

    fn expire_key(
        &mut self,
        key: (u8, u64, PropertyId),
        resolved_commit_time_nanos: i64,
    ) -> Result<()> {
        let declaration = self
            .declarations
            .get(&key)
            .ok_or_else(|| Error::internal("temporal declaration disappeared"))?;
        let oldest = resolved_commit_time_nanos.saturating_sub(declaration.retention_nanos);
        let column = self
            .columns
            .get_mut(&key)
            .ok_or_else(|| Error::internal("temporal declaration has no corresponding column"))?;
        let column = Arc::make_mut(column);
        let retained = column
            .samples()
            .filter(|sample| sample.event_time_nanos >= oldest)
            .collect::<Vec<_>>();
        column.replace_samples(retained)
    }

    fn update_rollups_for_sample(
        &mut self,
        key: (u8, u64, PropertyId),
        sample: &TemporalSample,
    ) -> Result<()> {
        let names = self
            .rollups
            .iter()
            .filter_map(|(name, rollup)| {
                let definition = &rollup.definition;
                ((
                    definition.entity_kind as u8,
                    definition.target,
                    definition.property,
                ) == key)
                    .then_some(name.clone())
            })
            .collect::<Vec<_>>();
        for name in names {
            let rollup = self
                .rollups
                .get_mut(&name)
                .ok_or_else(|| Error::internal("rollup disappeared during incremental apply"))?;
            let rollup = Arc::make_mut(rollup);
            add_sample_to_rollup(rollup, sample)?;
        }
        Ok(())
    }

    fn rebuild_rollups_for_key(&mut self, key: (u8, u64, PropertyId)) -> Result<()> {
        let names = self
            .rollups
            .iter()
            .filter_map(|(name, rollup)| {
                let definition = &rollup.definition;
                ((
                    definition.entity_kind as u8,
                    definition.target,
                    definition.property,
                ) == key)
                    .then_some(name.clone())
            })
            .collect::<Vec<_>>();
        for name in names {
            self.rebuild_rollup(&name)?;
        }
        Ok(())
    }

    fn validate_and_rebuild_derived(&mut self) -> Result<()> {
        if self.declarations.len() != self.columns.len() {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "temporal declarations and canonical columns differ",
            ));
        }
        for (key, declaration) in self.declarations.iter() {
            let expected = (
                declaration.entity_kind as u8,
                declaration.target,
                declaration.property,
            );
            if *key != expected || declaration.retention_nanos <= 0 {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "temporal declaration key or retention is invalid",
                ));
            }
            let column = self.columns.get_mut(key).ok_or_else(|| {
                Error::new(
                    ErrorCode::CorruptStorage,
                    "temporal declaration has no canonical column",
                )
            })?;
            let column = Arc::make_mut(column);
            column.property = declaration.property;
            let row_count = column.entity_ids.len();
            if row_count != column.event_times_nanos.len()
                || row_count != column.sequence_indexes.len()
                || row_count != column.values.rows()
            {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "flat temporal column lengths differ",
                ));
            }
            for sample in column.samples() {
                if sample.property != declaration.property
                    || !declaration.value_type.accepts(&sample.value)
                {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        "temporal sample property or scalar type is invalid",
                    ));
                }
            }
            rebuild_current(column);
        }
        for (name, rollup) in &mut self.rollups {
            let rollup = Arc::make_mut(rollup);
            let definition = &rollup.definition;
            definition.window.validate().map_err(|_| {
                Error::new(
                    ErrorCode::CorruptStorage,
                    "temporal rollup window is invalid",
                )
            })?;
            let key = (
                definition.entity_kind as u8,
                definition.target,
                definition.property,
            );
            if name != &definition.name
                || definition.aggregates.is_empty()
                || definition.window.aggregates != definition.aggregates
                || !self.declarations.contains_key(&key)
            {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "temporal rollup definition is inconsistent",
                ));
            }
            rollup.buckets = Arc::new(BTreeMap::new());
            rollup.bucket_overrides = PersistentMap::default();
        }
        let names = self.rollups.keys().cloned().collect::<Vec<_>>();
        for name in names {
            self.rebuild_rollup(&name)?;
        }
        Ok(())
    }
}

fn encode_temporal_device_values(
    value_type: TemporalType,
    samples: &[TemporalSample],
) -> Result<(Vec<u8>, TemporalDeviceValues)> {
    let validity = samples
        .iter()
        .map(|sample| u8::from(!matches!(&sample.value, ScalarValue::Null)))
        .collect::<Vec<_>>();
    let mismatch = || {
        Error::new(
            ErrorCode::CorruptStorage,
            "temporal sample value differs from its declared scalar type",
        )
    };
    let values = match value_type {
        TemporalType::Boolean => TemporalDeviceValues::Boolean(
            samples
                .iter()
                .map(|sample| match &sample.value {
                    ScalarValue::Null => Ok(0),
                    ScalarValue::Boolean(value) => Ok(u8::from(*value)),
                    _ => Err(mismatch()),
                })
                .collect::<Result<Vec<_>>>()?,
        ),
        TemporalType::Integer => TemporalDeviceValues::Integer(
            samples
                .iter()
                .map(|sample| match &sample.value {
                    ScalarValue::Null => Ok(0),
                    ScalarValue::Integer(value) => Ok(*value),
                    _ => Err(mismatch()),
                })
                .collect::<Result<Vec<_>>>()?,
        ),
        TemporalType::Float => TemporalDeviceValues::FloatBits(
            samples
                .iter()
                .map(|sample| match &sample.value {
                    ScalarValue::Null => Ok(0),
                    ScalarValue::Float(value) => Ok(value.0.to_bits()),
                    _ => Err(mismatch()),
                })
                .collect::<Result<Vec<_>>>()?,
        ),
        TemporalType::String => {
            let mut offsets = Vec::with_capacity(samples.len().saturating_add(1));
            let mut bytes = Vec::new();
            offsets.push(0);
            for sample in samples {
                match &sample.value {
                    ScalarValue::Null => {}
                    ScalarValue::String(value) => bytes.extend_from_slice(value.as_bytes()),
                    _ => return Err(mismatch()),
                }
                offsets.push(u32::try_from(bytes.len()).map_err(|_| {
                    Error::new(
                        ErrorCode::ResultBudgetExceeded,
                        "temporal string bytes exceed u32",
                    )
                })?);
            }
            TemporalDeviceValues::String { offsets, bytes }
        }
        TemporalType::Date => TemporalDeviceValues::Date(
            samples
                .iter()
                .map(|sample| match &sample.value {
                    ScalarValue::Null => Ok(0),
                    ScalarValue::Date(value) => Ok(*value),
                    _ => Err(mismatch()),
                })
                .collect::<Result<Vec<_>>>()?,
        ),
        TemporalType::LocalTime => TemporalDeviceValues::LocalTime(
            samples
                .iter()
                .map(|sample| match &sample.value {
                    ScalarValue::Null => Ok(0),
                    ScalarValue::LocalTime(value) => Ok(*value),
                    _ => Err(mismatch()),
                })
                .collect::<Result<Vec<_>>>()?,
        ),
        TemporalType::ZonedTime => {
            let mut nanos = Vec::with_capacity(samples.len());
            let mut offsets = Vec::with_capacity(samples.len());
            for sample in samples {
                match &sample.value {
                    ScalarValue::Null => {
                        nanos.push(0);
                        offsets.push(0);
                    }
                    ScalarValue::ZonedTime {
                        nanos: value,
                        offset_seconds,
                    } => {
                        nanos.push(*value);
                        offsets.push(*offset_seconds);
                    }
                    _ => return Err(mismatch()),
                }
            }
            TemporalDeviceValues::ZonedTime { nanos, offsets }
        }
        TemporalType::LocalDateTime => {
            let mut seconds = Vec::with_capacity(samples.len());
            let mut nanos = Vec::with_capacity(samples.len());
            for sample in samples {
                match &sample.value {
                    ScalarValue::Null => {
                        seconds.push(0);
                        nanos.push(0);
                    }
                    ScalarValue::LocalDateTime {
                        seconds: value,
                        nanos: fraction,
                    } => {
                        seconds.push(*value);
                        nanos.push(*fraction);
                    }
                    _ => return Err(mismatch()),
                }
            }
            TemporalDeviceValues::LocalDateTime { seconds, nanos }
        }
        TemporalType::ZonedDateTime => {
            let mut seconds = Vec::with_capacity(samples.len());
            let mut nanos = Vec::with_capacity(samples.len());
            let mut timezone_offsets = Vec::with_capacity(samples.len().saturating_add(1));
            let mut timezone_bytes = Vec::new();
            timezone_offsets.push(0);
            for sample in samples {
                match &sample.value {
                    ScalarValue::Null => {
                        seconds.push(0);
                        nanos.push(0);
                    }
                    ScalarValue::ZonedDateTime {
                        seconds: value,
                        nanos: fraction,
                        timezone,
                    } => {
                        seconds.push(*value);
                        nanos.push(*fraction);
                        timezone_bytes.extend_from_slice(timezone.as_bytes());
                    }
                    _ => return Err(mismatch()),
                }
                timezone_offsets.push(u32::try_from(timezone_bytes.len()).map_err(|_| {
                    Error::new(
                        ErrorCode::ResultBudgetExceeded,
                        "temporal timezone bytes exceed u32",
                    )
                })?);
            }
            TemporalDeviceValues::ZonedDateTime {
                seconds,
                nanos,
                timezone_offsets,
                timezone_bytes,
            }
        }
        TemporalType::Duration => {
            let mut months = Vec::with_capacity(samples.len());
            let mut days = Vec::with_capacity(samples.len());
            let mut seconds = Vec::with_capacity(samples.len());
            let mut nanos = Vec::with_capacity(samples.len());
            for sample in samples {
                match &sample.value {
                    ScalarValue::Null => {
                        months.push(0);
                        days.push(0);
                        seconds.push(0);
                        nanos.push(0);
                    }
                    ScalarValue::Duration {
                        months: m,
                        days: d,
                        seconds: s,
                        nanos: n,
                    } => {
                        months.push(*m);
                        days.push(*d);
                        seconds.push(*s);
                        nanos.push(*n);
                    }
                    _ => return Err(mismatch()),
                }
            }
            TemporalDeviceValues::Duration {
                months,
                days,
                seconds,
                nanos,
            }
        }
    };
    Ok((validity, values))
}

fn rebuild_current(column: &mut TemporalColumn) {
    column.current = PersistentMap::default();
    for row in 0..column.len() {
        let Some(sample) = column.sample(row) else {
            continue;
        };
        let replace = column
            .current
            .get(u128::from(sample.entity_id))
            .is_none_or(|current_row| {
                column.sample(*current_row).is_none_or(|current| {
                    (sample.event_time_nanos, sample.sequence_index)
                        > (current.event_time_nanos, current.sequence_index)
                })
            });
        if replace {
            column.current.insert(u128::from(sample.entity_id), row);
        }
    }
}

fn rollup_bucket_key(entity_id: u64, start_nanos: i64) -> u128 {
    (u128::from(entity_id) << 64) | u128::from((start_nanos as u64) ^ (1_u64 << 63))
}

fn rollup_bucket_parts(key: u128) -> (u64, i64) {
    let entity_id = (key >> 64) as u64;
    let start_nanos = ((key as u64) ^ (1_u64 << 63)) as i64;
    (entity_id, start_nanos)
}

fn add_sample_to_rollup(rollup: &mut MaterializedRollup, sample: &TemporalSample) -> Result<()> {
    let end = sample.event_time_nanos.checked_add(1).ok_or_else(|| {
        Error::new(
            ErrorCode::TemporalRange,
            "sample timestamp cannot form a range",
        )
    })?;
    for (start, bucket_end) in
        TemporalStore::window_boundaries(sample.event_time_nanos, end, &rollup.definition.window)?
    {
        let mut bucket = rollup
            .bucket(sample.entity_id, start)
            .cloned()
            .unwrap_or_else(|| RollupBucket::empty(start, bucket_end));
        if bucket.end_nanos != bucket_end {
            return Err(Error::internal(
                "rollup boundary changed for an existing bucket",
            ));
        }
        bucket.add(&sample.value);
        rollup.insert_bucket(sample.entity_id, bucket);
    }
    Ok(())
}

fn add_sample_to_buckets(
    spec: &WindowSpec,
    buckets: &mut BTreeMap<(u64, i64), RollupBucket>,
    sample: &TemporalSample,
) -> Result<()> {
    let end = sample.event_time_nanos.checked_add(1).ok_or_else(|| {
        Error::new(
            ErrorCode::TemporalRange,
            "sample timestamp cannot form a range",
        )
    })?;
    for (start, bucket_end) in TemporalStore::window_boundaries(sample.event_time_nanos, end, spec)?
    {
        let bucket = buckets
            .entry((sample.entity_id, start))
            .or_insert_with(|| RollupBucket::empty(start, bucket_end));
        if bucket.end_nanos != bucket_end {
            return Err(Error::internal(
                "rollup boundary changed for an existing bucket",
            ));
        }
        bucket.add(&sample.value);
    }
    Ok(())
}

fn floor_div(value: i64, divisor: i64) -> i64 {
    let quotient = value / divisor;
    let remainder = value % divisor;
    if remainder != 0 && (remainder < 0) != (divisor < 0) {
        quotient - 1
    } else {
        quotient
    }
}

fn fixed_boundaries(
    from: i64,
    to: i64,
    align: i64,
    width: i64,
    every: i64,
) -> Result<Vec<(i64, i64)>> {
    let earliest = from.saturating_sub(width).saturating_add(1);
    let mut start = align
        .saturating_add(floor_div(earliest.saturating_sub(align), every).saturating_mul(every));
    let mut result = Vec::new();
    while start < to {
        let end = start
            .checked_add(width)
            .ok_or_else(|| Error::new(ErrorCode::TemporalRange, "window boundary overflow"))?;
        if end > from {
            result.push((start, end));
            if result.len() > 1_000_000 {
                return Err(Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "window produced too many buckets",
                ));
            }
        }
        start = start
            .checked_add(every)
            .ok_or_else(|| Error::new(ErrorCode::TemporalRange, "window boundary overflow"))?;
    }
    Ok(result)
}

fn calendar_boundaries(
    from: i64,
    to: i64,
    align: i64,
    months: u32,
    days: u32,
    timezone: &str,
) -> Result<Vec<(i64, i64)>> {
    let zone = timezone
        .parse::<Tz>()
        .map_err(|_| Error::new(ErrorCode::TemporalRange, "invalid IANA time zone"))?;
    let mut cursor = nanos_to_local(align, zone)?;
    let from_local = nanos_to_local(from, zone)?;
    while cursor > from_local {
        cursor = shift_calendar(cursor, zone, -(months as i32), -(days as i64))?;
    }
    loop {
        let next = shift_calendar(cursor, zone, months as i32, days as i64)?;
        if next > from_local {
            break;
        }
        cursor = next;
    }
    let mut result = Vec::new();
    while datetime_to_nanos(cursor.with_timezone(&Utc))? < to {
        let next = shift_calendar(cursor, zone, months as i32, days as i64)?;
        let start = datetime_to_nanos(cursor.with_timezone(&Utc))?;
        let end = datetime_to_nanos(next.with_timezone(&Utc))?;
        if end > from {
            result.push((start, end));
            if result.len() > 1_000_000 {
                return Err(Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "window produced too many buckets",
                ));
            }
        }
        cursor = next;
    }
    Ok(result)
}

fn nanos_to_local(nanos: i64, zone: Tz) -> Result<chrono::DateTime<Tz>> {
    let seconds = floor_div(nanos, 1_000_000_000);
    let sub = nanos.saturating_sub(seconds.saturating_mul(1_000_000_000));
    let sub = u32::try_from(sub).map_err(|_| {
        Error::new(
            ErrorCode::TemporalRange,
            "timestamp nanoseconds are invalid",
        )
    })?;
    let utc = chrono::DateTime::<Utc>::from_timestamp(seconds, sub)
        .ok_or_else(|| Error::new(ErrorCode::TemporalRange, "timestamp is out of range"))?;
    Ok(utc.with_timezone(&zone))
}

fn datetime_to_nanos(datetime: chrono::DateTime<Utc>) -> Result<i64> {
    datetime
        .timestamp()
        .checked_mul(1_000_000_000)
        .and_then(|value| value.checked_add(i64::from(datetime.timestamp_subsec_nanos())))
        .ok_or_else(|| Error::new(ErrorCode::TemporalRange, "timestamp nanoseconds overflow"))
}

fn shift_calendar(
    datetime: chrono::DateTime<Tz>,
    zone: Tz,
    month_delta: i32,
    day_delta: i64,
) -> Result<chrono::DateTime<Tz>> {
    let naive = datetime.naive_local();
    let total_month = naive
        .year()
        .saturating_mul(12)
        .saturating_add(i32::try_from(naive.month0()).unwrap_or(0))
        .saturating_add(month_delta);
    let year = floor_div(i64::from(total_month), 12) as i32;
    let month0 = total_month.rem_euclid(12) as u32;
    let last_day = days_in_month(year, month0 + 1)?;
    let day = naive.day().min(last_day);
    let date = NaiveDate::from_ymd_opt(year, month0 + 1, day)
        .ok_or_else(|| Error::new(ErrorCode::TemporalRange, "calendar date is out of range"))?
        .checked_add_signed(chrono::Duration::days(day_delta))
        .ok_or_else(|| Error::new(ErrorCode::TemporalRange, "calendar date overflow"))?;
    let shifted = date
        .and_hms_nano_opt(
            naive.hour(),
            naive.minute(),
            naive.second(),
            naive.nanosecond(),
        )
        .ok_or_else(|| Error::new(ErrorCode::TemporalRange, "calendar time is invalid"))?;
    match zone.from_local_datetime(&shifted) {
        LocalResult::Single(value) => Ok(value),
        LocalResult::Ambiguous(first, _) => Ok(first),
        LocalResult::None => {
            let mut probe = shifted;
            for _ in 0..180 {
                probe = probe
                    .checked_add_signed(chrono::Duration::minutes(1))
                    .ok_or_else(|| {
                        Error::new(ErrorCode::TemporalRange, "DST gap adjustment overflow")
                    })?;
                if let LocalResult::Single(value) | LocalResult::Ambiguous(value, _) =
                    zone.from_local_datetime(&probe)
                {
                    return Ok(value);
                }
            }
            Err(Error::new(
                ErrorCode::TemporalRange,
                "unable to resolve local time across DST gap",
            ))
        }
    }
}

fn days_in_month(year: i32, month: u32) -> Result<u32> {
    let (next_year, next_month) = if month == 12 {
        (year.saturating_add(1), 1)
    } else {
        (year, month + 1)
    };
    let next = NaiveDate::from_ymd_opt(next_year, next_month, 1)
        .ok_or_else(|| Error::new(ErrorCode::TemporalRange, "calendar month is out of range"))?;
    let last = next
        .pred_opt()
        .ok_or_else(|| Error::new(ErrorCode::TemporalRange, "calendar month underflow"))?;
    Ok(last.day())
}

#[cfg(test)]
mod tests {
    use ordered_float::OrderedFloat;

    use super::*;

    fn sample(property: PropertyId, time: i64, revision: u64, value: f64) -> TemporalSample {
        TemporalSample {
            entity_id: 9,
            property,
            event_time_nanos: time,
            sequence_index: revision,
            value: ScalarValue::Float(OrderedFloat(value)),
        }
    }

    #[test]
    fn temporal_device_date_column_preserves_wide_epoch_days() -> Result<()> {
        let wide_days = 365_242_499_634_i64;
        let samples = [TemporalSample {
            entity_id: 9,
            property: PropertyId(3),
            event_time_nanos: 1,
            sequence_index: 1,
            value: ScalarValue::Date(wide_days),
        }];
        let (validity, values) = encode_temporal_device_values(TemporalType::Date, &samples)?;
        assert_eq!(validity, vec![1]);
        assert_eq!(values, TemporalDeviceValues::Date(vec![wide_days]));
        assert_eq!(values.resident_bytes(), size_of::<i64>());
        Ok(())
    }

    #[test]
    fn late_rollup_repair_matches_rebuild_and_restart() -> Result<()> {
        let property = PropertyId(3);
        let mut store = TemporalStore::default();
        store.declare(
            TemporalDeclaration {
                entity_kind: EntityKind::Node,
                target: 12,
                property,
                value_type: TemporalType::Float,
                retention_nanos: 100_000,
            },
            1_000,
        )?;
        store.create_rollup(TemporalRollupDefinition {
            name: "ten_nanos".to_owned(),
            entity_kind: EntityKind::Node,
            target: 12,
            property,
            window: WindowSpec::tumbling(10),
            aggregates: AggregateSet::ALL,
        })?;
        store.append(EntityKind::Node, 12, sample(property, 12, 1, 2.0), 1_000)?;
        store.append(EntityKind::Node, 12, sample(property, 28, 2, 8.0), 1_000)?;
        store.append(EntityKind::Node, 12, sample(property, 16, 3, 4.0), 1_000)?;
        let incrementally_repaired = store.rollup_buckets("ten_nanos", 9, 0, 40)?;

        let mut rebuilt = store.clone();
        rebuilt.rebuild_rollup("ten_nanos")?;
        assert_eq!(
            incrementally_repaired,
            rebuilt.rollup_buckets("ten_nanos", 9, 0, 40)?
        );
        assert_eq!(incrementally_repaired[0].count, 2);
        assert_eq!(incrementally_repaired[0].sum, Some(6.0));

        // Database checkpoints use canonical CBOR. Postcard cannot deserialize the internally
        // tagged scalar-value representation and is deliberately not a database-state codec.
        let mut persisted_with_stale_derived_state = store.clone();
        persisted_with_stale_derived_state
            .columns
            .values_mut()
            .for_each(|column| Arc::make_mut(column).current = PersistentMap::default());
        persisted_with_stale_derived_state
            .rollups
            .values_mut()
            .for_each(|rollup| {
                let rollup = Arc::make_mut(rollup);
                rollup.buckets = Arc::new(BTreeMap::new());
                rollup.bucket_overrides = PersistentMap::default();
            });
        let mut checkpoint = Vec::new();
        ciborium::ser::into_writer(&persisted_with_stale_derived_state, &mut checkpoint)
            .map_err(|error| Error::internal(format!("test encoding failed: {error}")))?;
        let restored: TemporalStore = ciborium::de::from_reader(checkpoint.as_slice())
            .map_err(|error| Error::internal(format!("test decoding failed: {error}")))?;
        assert_eq!(
            restored.rollup_buckets("ten_nanos", 9, 0, 40)?,
            incrementally_repaired
        );
        assert_eq!(
            restored
                .current(EntityKind::Node, 12, 9, property)
                .map(|sample| sample.sequence_index),
            Some(2)
        );
        Ok(())
    }

    #[test]
    fn target_qualified_columns_never_alias() -> Result<()> {
        let property = PropertyId(5);
        let mut store = TemporalStore::default();
        for target in [21, 22] {
            store.declare(
                TemporalDeclaration {
                    entity_kind: EntityKind::Node,
                    target,
                    property,
                    value_type: TemporalType::Float,
                    retention_nanos: 1_000,
                },
                100,
            )?;
        }
        store.append(EntityKind::Node, 21, sample(property, 20, 1, 1.0), 100)?;
        store.append(EntityKind::Node, 22, sample(property, 30, 2, 2.0), 100)?;
        assert_eq!(
            store
                .current(EntityKind::Node, 21, 9, property)
                .map(|value| value.value.clone()),
            Some(ScalarValue::Float(OrderedFloat(1.0)))
        );
        assert_eq!(
            store
                .current(EntityKind::Node, 22, 9, property)
                .map(|value| value.value.clone()),
            Some(ScalarValue::Float(OrderedFloat(2.0)))
        );
        Ok(())
    }

    #[test]
    fn restart_rejects_inconsistent_canonical_temporal_state() -> Result<()> {
        let property = PropertyId(7);
        let mut store = TemporalStore::default();
        store.declare(
            TemporalDeclaration {
                entity_kind: EntityKind::Node,
                target: 31,
                property,
                value_type: TemporalType::Float,
                retention_nanos: 1_000,
            },
            100,
        )?;
        let declaration = Arc::make_mut(&mut store.declarations)
            .remove(&(EntityKind::Node as u8, 31, property))
            .ok_or_else(|| Error::internal("test declaration is missing"))?;
        Arc::make_mut(&mut store.declarations)
            .insert((EntityKind::Node as u8, 32, property), declaration);

        let mut checkpoint = Vec::new();
        ciborium::ser::into_writer(&store, &mut checkpoint)
            .map_err(|error| Error::internal(format!("test encoding failed: {error}")))?;
        assert!(ciborium::de::from_reader::<TemporalStore, _>(checkpoint.as_slice()).is_err());
        Ok(())
    }

    #[test]
    fn temporal_and_rollup_point_updates_detach_bounded_paths() -> Result<()> {
        fn measure(rows: u64) -> Result<(usize, usize)> {
            let property = PropertyId(11);
            let target = 41;
            let mut store = TemporalStore::default();
            store.declare(
                TemporalDeclaration {
                    entity_kind: EntityKind::Node,
                    target,
                    property,
                    value_type: TemporalType::Float,
                    retention_nanos: i64::MAX,
                },
                0,
            )?;
            for entity_id in 0..rows {
                store.append(
                    EntityKind::Node,
                    target,
                    TemporalSample {
                        entity_id,
                        property,
                        event_time_nanos: entity_id as i64,
                        sequence_index: entity_id + 1,
                        value: ScalarValue::Float(OrderedFloat(entity_id as f64)),
                    },
                    rows as i64,
                )?;
            }
            store.create_rollup(TemporalRollupDefinition {
                name: "point".to_owned(),
                entity_kind: EntityKind::Node,
                target,
                property,
                window: WindowSpec::tumbling(1),
                aggregates: AggregateSet::ALL,
            })?;

            let published = store.clone();
            store.append(
                EntityKind::Node,
                target,
                TemporalSample {
                    entity_id: rows,
                    property,
                    event_time_nanos: rows as i64,
                    sequence_index: rows + 1,
                    value: ScalarValue::Float(OrderedFloat(1.0)),
                },
                rows as i64,
            )?;

            let key = (EntityKind::Node as u8, target, property);
            let before_column = published
                .columns
                .get(&key)
                .ok_or_else(|| Error::internal("published temporal column is missing"))?;
            let after_column = store
                .columns
                .get(&key)
                .ok_or_else(|| Error::internal("updated temporal column is missing"))?;
            assert!(Arc::ptr_eq(&published.declarations, &store.declarations));
            assert!(
                published
                    .current(EntityKind::Node, target, rows, property)
                    .is_none()
            );
            assert!(
                store
                    .current(EntityKind::Node, target, rows, property)
                    .is_some()
            );
            let column_detached = after_column
                .entity_ids
                .detached_page_bytes_from(&before_column.entity_ids)
                .saturating_add(
                    after_column
                        .event_times_nanos
                        .detached_page_bytes_from(&before_column.event_times_nanos),
                )
                .saturating_add(
                    after_column
                        .sequence_indexes
                        .detached_page_bytes_from(&before_column.sequence_indexes),
                )
                .saturating_add(
                    after_column
                        .values
                        .detached_page_bytes_from(&before_column.values),
                )
                .saturating_add(
                    after_column
                        .current
                        .detached_node_bytes_from(&before_column.current),
                );

            let before_rollup = published
                .rollups
                .get("point")
                .ok_or_else(|| Error::internal("published temporal rollup is missing"))?;
            let after_rollup = store
                .rollups
                .get("point")
                .ok_or_else(|| Error::internal("updated temporal rollup is missing"))?;
            assert!(Arc::ptr_eq(&before_rollup.buckets, &after_rollup.buckets));
            assert!(
                published
                    .rollup_buckets("point", rows, rows as i64, rows as i64 + 1)?
                    .is_empty()
            );
            assert_eq!(
                store
                    .rollup_buckets("point", rows, rows as i64, rows as i64 + 1)?
                    .len(),
                1
            );
            let rollup_detached = after_rollup
                .bucket_overrides
                .detached_node_bytes_from(&before_rollup.bucket_overrides);
            Ok((column_detached, rollup_detached))
        }

        let small = measure(4_096)?;
        let large = measure(32_768)?;
        assert!(small.0 <= 128 * 1_024 && large.0 <= 128 * 1_024);
        assert!(small.1 <= 8 * 1_024 && large.1 <= 8 * 1_024);
        Ok(())
    }
}
