use std::{
    collections::{BTreeMap, BTreeSet, HashMap},
    fmt,
    sync::Arc,
};

use rayon::prelude::*;
use serde::{
    Deserialize, Deserializer, Serialize, Serializer,
    de::{MapAccess, SeqAccess, Visitor},
    ser::{SerializeMap, SerializeSeq},
};
use uuid::Uuid;

use crate::storage::ConnectionId;
use crate::{
    Bookmark, CommitAcknowledgement, Error, ProjectId, Result,
    engine::ApplicationWait,
    graph::{PagedVec, PersistentMap, stable_id_key},
    storage::{
        CowArc, SegmentDescriptor, SegmentFamily, SegmentRecord, SegmentRecordLocation,
        SegmentStore,
    },
    types::{MessageId, StreamId},
};

const BROKER_PAYLOAD_SEGMENT_KIND: u16 = 1;
const BROKER_RAW_AMQP_PAYLOAD_SEGMENT_KIND: u16 = 2;
const BROKER_RAW_KAFKA_PAYLOAD_SEGMENT_KIND: u16 = 3;
const BROKER_PAYLOAD_RECORD_V2_MAGIC: &[u8; 4] = b"IGP2";
const MAX_BROKER_BATCH_BYTES: usize = 128 * 1024 * 1024;
const MAX_BROKER_DELIVERY_RESULT_BYTES: usize = 16 * 1024 * 1024;
// Queue, exchange, and routing names are each bounded to 255 bytes. The remainder covers
// delivery offsets/tags, serialized field names, collection delimiters, and enum envelopes.
const BROKER_DELIVERY_ENVELOPE_BYTES: usize = 1_024;
pub(crate) const DELIVERY_LEASE_MILLIS: i64 = 120_000;

/// Broker payload origin retained once with the immutable payload.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case", deny_unknown_fields)]
pub enum IngressMetadata {
    Kafka {
        create_time_ms: Option<i64>,
        key: Option<Vec<u8>>,
        headers: BTreeMap<String, Vec<u8>>,
        #[serde(default)]
        value_is_null: bool,
    },
    Amqp {
        exchange: String,
        routing_key: String,
        properties: BTreeMap<String, Vec<u8>>,
        headers: BTreeMap<String, Vec<u8>>,
        death_count: u32,
    },
}

/// Bounds bytes that become visible on a protocol response but may be stored only once per raw
/// segment. The fixed and per-field allowances deliberately overestimate Postcard/AMQP/Kafka
/// framing, so admission and retention cannot be defeated by large keys or field names.
fn ingress_accounted_bytes(ingress: &IngressMetadata) -> u64 {
    const ENVELOPE_BYTES: u64 = 64;
    const FIELD_ENVELOPE_BYTES: u64 = 16;
    let map_bytes = |values: &BTreeMap<String, Vec<u8>>| {
        values.iter().fold(0_u64, |total, (key, value)| {
            total
                .saturating_add(FIELD_ENVELOPE_BYTES)
                .saturating_add(key.len() as u64)
                .saturating_add(value.len() as u64)
        })
    };
    match ingress {
        IngressMetadata::Kafka { key, headers, .. } => ENVELOPE_BYTES
            .saturating_add(key.as_ref().map_or(0, |key| key.len() as u64))
            .saturating_add(map_bytes(headers)),
        IngressMetadata::Amqp {
            exchange,
            routing_key,
            properties,
            headers,
            ..
        } => ENVELOPE_BYTES
            .saturating_add(exchange.len() as u64)
            .saturating_add(routing_key.len() as u64)
            .saturating_add(map_bytes(properties))
            .saturating_add(map_bytes(headers)),
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PayloadRecord {
    pub id: MessageId,
    pub resolved_time_ms: i64,
    pub ingress: IngressMetadata,
    /// Immutable payload segment shared by canonical state clones and fan-out references.
    pub payload: Arc<[u8]>,
    pub checksum: [u8; 32],
}

/// Strict finite-read bounds for an in-process stream caller.
pub struct PartitionRead<'a> {
    pub project: ProjectId,
    pub topic: &'a str,
    pub partition: i32,
    pub offset: u64,
    pub maximum_bytes: usize,
    pub maximum_records: usize,
}

/// Canonical payload metadata. The opaque bytes live only in the referenced immutable segment;
/// protocol reads materialize one bounded payload batch into transient `PayloadRecord` values.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StoredPayloadRecord {
    id: MessageId,
    resolved_time_ms: i64,
    segment: u64,
    location: SegmentRecordLocation,
    /// Exact immutable bytes occupied by this payload record, including ingress metadata and
    /// segment framing. Retention limits account for physical storage rather than body bytes.
    retained_bytes: u64,
    /// Protocol-visible body bytes used only for response budgeting.
    payload_bytes: u64,
    checksum: [u8; 32],
    kafka_time_ms: Option<i64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct BrokerPayloadSegment {
    id: u64,
    descriptor: SegmentDescriptor,
    live_records: u32,
    #[serde(default)]
    shared_amqp_ingress: Option<IngressMetadata>,
    #[serde(default)]
    shared_kafka_ingress: Option<IngressMetadata>,
    /// Conservative protocol/accounting charge for an ingress envelope stored once here rather
    /// than repeated inside every raw payload record. Older snapshots recompute a zero default.
    #[serde(default)]
    shared_ingress_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct PayloadSegmentRecord {
    project: ProjectId,
    id: MessageId,
    resolved_time_ms: i64,
    ingress: IngressMetadata,
    payload: Vec<u8>,
    checksum: [u8; 32],
}

#[cfg(test)]
#[derive(Serialize)]
struct PayloadSegmentRecordRef<'a> {
    project: ProjectId,
    id: MessageId,
    resolved_time_ms: i64,
    ingress: &'a IngressMetadata,
    payload: &'a [u8],
    checksum: [u8; 32],
}

#[derive(Serialize)]
enum CompactIngressMetadataRef<'a> {
    Kafka {
        create_time_ms: Option<i64>,
        key: &'a Option<Vec<u8>>,
        headers: &'a BTreeMap<String, Vec<u8>>,
        value_is_null: bool,
    },
    Amqp {
        exchange: &'a str,
        routing_key: &'a str,
        properties: &'a BTreeMap<String, Vec<u8>>,
        headers: &'a BTreeMap<String, Vec<u8>>,
        death_count: u32,
    },
}

#[derive(Deserialize)]
enum CompactIngressMetadata {
    Kafka {
        create_time_ms: Option<i64>,
        key: Option<Vec<u8>>,
        headers: BTreeMap<String, Vec<u8>>,
        value_is_null: bool,
    },
    Amqp {
        exchange: String,
        routing_key: String,
        properties: BTreeMap<String, Vec<u8>>,
        headers: BTreeMap<String, Vec<u8>>,
        death_count: u32,
    },
}

#[derive(Serialize)]
struct CompactPayloadSegmentRecordRef<'a> {
    project: ProjectId,
    id: MessageId,
    resolved_time_ms: i64,
    ingress: CompactIngressMetadataRef<'a>,
    payload: &'a [u8],
    checksum: [u8; 32],
}

#[derive(Deserialize)]
struct CompactPayloadSegmentRecord {
    project: ProjectId,
    id: MessageId,
    resolved_time_ms: i64,
    ingress: CompactIngressMetadata,
    payload: Vec<u8>,
    checksum: [u8; 32],
}

fn compact_ingress(ingress: &IngressMetadata) -> CompactIngressMetadataRef<'_> {
    match ingress {
        IngressMetadata::Kafka {
            create_time_ms,
            key,
            headers,
            value_is_null,
        } => CompactIngressMetadataRef::Kafka {
            create_time_ms: *create_time_ms,
            key,
            headers,
            value_is_null: *value_is_null,
        },
        IngressMetadata::Amqp {
            exchange,
            routing_key,
            properties,
            headers,
            death_count,
        } => CompactIngressMetadataRef::Amqp {
            exchange,
            routing_key,
            properties,
            headers,
            death_count: *death_count,
        },
    }
}

fn expand_compact_ingress(ingress: CompactIngressMetadata) -> IngressMetadata {
    match ingress {
        CompactIngressMetadata::Kafka {
            create_time_ms,
            key,
            headers,
            value_is_null,
        } => IngressMetadata::Kafka {
            create_time_ms,
            key,
            headers,
            value_is_null,
        },
        CompactIngressMetadata::Amqp {
            exchange,
            routing_key,
            properties,
            headers,
            death_count,
        } => IngressMetadata::Amqp {
            exchange,
            routing_key,
            properties,
            headers,
            death_count,
        },
    }
}

struct PendingPayloadRecord<'a> {
    id: MessageId,
    ingress: &'a IngressMetadata,
    payload: &'a [u8],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum QueueKind {
    Classic,
    Stream,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AmqpExchangeKind {
    Direct,
    Fanout,
    Topic,
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RetentionPolicy {
    pub max_age_ms: Option<u64>,
    pub max_bytes: Option<u64>,
}

impl RetentionPolicy {
    pub fn validate(self) -> Result<Self> {
        if self.max_age_ms == Some(0) || self.max_bytes == Some(0) {
            return Err(Error::invalid_data(
                "retention bounds must be positive when present",
            ));
        }
        if self
            .max_age_ms
            .is_some_and(|maximum| maximum > i64::MAX as u64)
        {
            return Err(Error::invalid_data(
                "retention age exceeds the supported timestamp range",
            ));
        }
        Ok(self)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Topic {
    partitions: Vec<StreamId>,
    retention: RetentionPolicy,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Exchange {
    kind: AmqpExchangeKind,
    durable: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Queue {
    stream: StreamId,
    kind: QueueKind,
    durable: bool,
    dead_letter_exchange: Option<String>,
    dead_letter_routing_key: Option<String>,
    retention: RetentionPolicy,
    /// Connection that exclusively owns this queue. No other connection may declare, consume from,
    /// bind or delete it, and it disappears when that connection does.
    #[serde(default)]
    exclusive_owner: Option<Uuid>,
    /// Deleted once its last consumer goes away, per AMQP 0-9-1 1.7.2.1.
    #[serde(default)]
    auto_delete: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Binding {
    exchange: String,
    queue: String,
    routing_key: String,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
enum DeliveryState {
    Ready,
    Unacked {
        owner: Uuid,
        consumer: u64,
        tag: u64,
    },
    Acknowledged,
    DeadLettered,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct StreamReference {
    offset: u64,
    message: MessageId,
    delivery: DeliveryState,
    redelivered: bool,
    death_count: u32,
    current_exchange: Option<String>,
    current_routing_key: Option<String>,
}

struct PendingDelivery {
    offset: u64,
    stored: StoredPayloadRecord,
    redelivered: bool,
    death_count: u32,
    exchange: Option<String>,
    routing_key: Option<String>,
}

#[derive(Clone, Debug, Default)]
struct MessageTable {
    first_id: u64,
    values: PagedVec<Option<StoredPayloadRecord>>,
    live: usize,
}

fn broker_map_id(key: u128) -> Option<u64> {
    if key > 0 && key <= u128::from(u64::MAX) {
        return Some(key as u64);
    }
    (key as u64 == 0)
        .then_some((key >> 64) as u64)
        .filter(|id| *id != 0)
}

fn broker_map_is_valid<V>(map: &PersistentMap<V>) -> bool {
    let mut ids = BTreeSet::new();
    map.iter()
        .all(|(key, _)| broker_map_id(key).is_some_and(|id| ids.insert(id)))
}

impl Serialize for MessageTable {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut map = serializer.serialize_map(Some(self.live))?;
        for (id, record) in self.iter() {
            map.serialize_entry(&stable_id_key(id.0), record)?;
        }
        map.end()
    }
}

impl<'de> Deserialize<'de> for MessageTable {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct MessageTableVisitor;

        impl<'de> Visitor<'de> for MessageTableVisitor {
            type Value = MessageTable;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a broker message numeric-key map")
            }

            fn visit_map<A>(self, mut entries: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: MapAccess<'de>,
            {
                let mut ordered = BTreeMap::<u64, StoredPayloadRecord>::new();
                while let Some((key, value)) = entries.next_entry::<u128, StoredPayloadRecord>()? {
                    let id = broker_map_id(key).ok_or_else(|| {
                        serde::de::Error::custom(
                            "broker message table key exceeds the canonical ID domain",
                        )
                    })?;
                    if ordered.insert(id, value).is_some() {
                        return Err(serde::de::Error::custom(
                            "broker message table repeats a logical message ID",
                        ));
                    }
                }
                let Some((&first_id, _)) = ordered.first_key_value() else {
                    return Ok(MessageTable::default());
                };
                let last_id = *ordered
                    .last_key_value()
                    .map(|(id, _)| id)
                    .ok_or_else(|| serde::de::Error::custom("broker message table disappeared"))?;
                let span = last_id
                    .checked_sub(first_id)
                    .and_then(|span| span.checked_add(1))
                    .and_then(|span| usize::try_from(span).ok())
                    .ok_or_else(|| {
                        serde::de::Error::custom("broker message table span exceeds host capacity")
                    })?;
                let live = ordered.len();
                let mut values = PagedVec::default();
                for offset in 0..span {
                    let id = first_id + offset as u64;
                    values.push(ordered.remove(&id));
                }
                Ok(MessageTable {
                    first_id,
                    values,
                    live,
                })
            }
        }

        deserializer.deserialize_map(MessageTableVisitor)
    }
}

impl MessageTable {
    #[cfg(test)]
    fn shared_with(&self, other: &Self) -> bool {
        self.first_id == other.first_id
            && self.live == other.live
            && self.values.detached_page_bytes_from(&other.values) == 0
    }

    fn get(&self, id: &MessageId) -> Option<&StoredPayloadRecord> {
        let index = usize::try_from(id.0.checked_sub(self.first_id)?).ok()?;
        self.values.get(index)?.as_ref()
    }

    fn insert(
        &mut self,
        id: MessageId,
        record: StoredPayloadRecord,
    ) -> Result<Option<StoredPayloadRecord>> {
        if self.values.is_empty() {
            self.first_id = id.0;
        }
        let expected = self
            .first_id
            .checked_add(self.values.len() as u64)
            .ok_or_else(|| Error::internal("broker message table ID span exhausted"))?;
        if id.0 != expected {
            return Err(Error::internal(
                "broker message table append is not monotonic",
            ));
        }
        self.values.push(Some(record));
        self.live = self
            .live
            .checked_add(1)
            .ok_or_else(|| Error::internal("broker message table live count exhausted"))?;
        Ok(None)
    }

    fn remove(&mut self, id: &MessageId) -> Option<StoredPayloadRecord> {
        let index = usize::try_from(id.0.checked_sub(self.first_id)?).ok()?;
        let removed = self.values.get_mut(index)?.take()?;
        self.live = self.live.saturating_sub(1);
        loop {
            let first_leaf = self.values.first_leaf_len();
            if first_leaf == 0 || self.values.iter().take(first_leaf).any(Option::is_some) {
                break;
            }
            let discarded = self.values.discard_prefix_leaves(first_leaf);
            self.first_id = self.first_id.saturating_add(discarded as u64);
        }
        Some(removed)
    }

    fn iter(&self) -> impl Iterator<Item = (MessageId, &StoredPayloadRecord)> {
        self.values
            .iter()
            .enumerate()
            .filter_map(|(index, record)| {
                record
                    .as_ref()
                    .map(|record| (MessageId(self.first_id + index as u64), record))
            })
    }

    fn keys(&self) -> impl Iterator<Item = MessageId> + '_ {
        self.iter().map(|(id, _)| id)
    }

    fn last_key_value(&self) -> Option<(MessageId, &StoredPayloadRecord)> {
        self.iter().last()
    }

    const fn len(&self) -> usize {
        self.live
    }
}

#[derive(Clone, Debug, Default, Serialize)]
struct PayloadSegmentTable(PersistentMap<Arc<BrokerPayloadSegment>>);

impl<'de> Deserialize<'de> for PayloadSegmentTable {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let map = PersistentMap::deserialize(deserializer)?;
        if !broker_map_is_valid(&map) {
            return Err(serde::de::Error::custom(
                "broker payload segment key exceeds the canonical ID domain",
            ));
        }
        Ok(Self(map))
    }
}

impl PayloadSegmentTable {
    fn get(&self, id: u64) -> Option<&Arc<BrokerPayloadSegment>> {
        self.0
            .get(stable_id_key(id))
            .or_else(|| self.0.get(u128::from(id)))
    }

    fn get_mut(&mut self, id: u64) -> Option<&mut Arc<BrokerPayloadSegment>> {
        let key = if self.0.contains_key(stable_id_key(id)) {
            stable_id_key(id)
        } else {
            u128::from(id)
        };
        self.0.get_mut(key)
    }

    fn insert(
        &mut self,
        id: u64,
        segment: Arc<BrokerPayloadSegment>,
    ) -> Option<Arc<BrokerPayloadSegment>> {
        self.0.insert_cow(stable_id_key(id), segment)
    }

    fn remove(&mut self, id: u64) -> Option<Arc<BrokerPayloadSegment>> {
        self.0
            .remove(stable_id_key(id))
            .or_else(|| self.0.remove(u128::from(id)))
    }

    fn iter(&self) -> impl Iterator<Item = (u64, &Arc<BrokerPayloadSegment>)> {
        self.0
            .iter()
            .filter_map(|(id, segment)| broker_map_id(id).map(|id| (id, segment)))
    }

    const fn len(&self) -> usize {
        self.0.len()
    }
}

#[derive(Clone, Debug, Default, Serialize)]
struct StreamTable(PersistentMap<Arc<StreamState>>);

impl<'de> Deserialize<'de> for StreamTable {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let map = PersistentMap::deserialize(deserializer)?;
        if !broker_map_is_valid(&map) {
            return Err(serde::de::Error::custom(
                "broker stream table key exceeds the canonical ID domain",
            ));
        }
        Ok(Self(map))
    }
}

impl StreamTable {
    #[cfg(test)]
    fn shared_with(&self, other: &Self) -> bool {
        self.0.shared_with(&other.0)
    }

    fn get(&self, id: &StreamId) -> Option<&Arc<StreamState>> {
        self.0
            .get(stable_id_key(id.0))
            .or_else(|| self.0.get(u128::from(id.0)))
    }

    fn get_mut(&mut self, id: &StreamId) -> Option<&mut Arc<StreamState>> {
        let key = if self.0.contains_key(stable_id_key(id.0)) {
            stable_id_key(id.0)
        } else {
            u128::from(id.0)
        };
        self.0.get_mut(key)
    }

    fn insert(&mut self, id: StreamId, state: Arc<StreamState>) -> Option<Arc<StreamState>> {
        self.0.insert_cow(stable_id_key(id.0), state)
    }

    fn remove(&mut self, id: &StreamId) -> Option<Arc<StreamState>> {
        self.0
            .remove(stable_id_key(id.0))
            .or_else(|| self.0.remove(u128::from(id.0)))
    }

    fn contains_key(&self, id: &StreamId) -> bool {
        self.0.contains_key(stable_id_key(id.0)) || self.0.contains_key(u128::from(id.0))
    }

    fn iter(&self) -> impl Iterator<Item = (StreamId, &Arc<StreamState>)> {
        self.0
            .iter()
            .filter_map(|(id, state)| broker_map_id(id).map(|id| (StreamId(id), state)))
    }

    fn last_key_value(&self) -> Option<(StreamId, &Arc<StreamState>)> {
        self.iter().last()
    }

    fn retain(&mut self, mut keep: impl FnMut(&StreamId, &Arc<StreamState>) -> bool) {
        let remove = self
            .iter()
            .filter_map(|(id, state)| (!keep(&id, state)).then_some(id))
            .collect::<Vec<_>>();
        for id in remove {
            let _ = self.remove(&id);
        }
    }

    const fn len(&self) -> usize {
        self.0.len()
    }
}

#[derive(Clone, Default)]
struct StreamRecords {
    values: PagedVec<StreamReference>,
    head: usize,
}

impl StreamRecords {
    #[cfg(test)]
    fn detached_page_bytes_from(&self, previous: &Self) -> usize {
        self.values.detached_page_bytes_from(&previous.values)
    }

    fn len(&self) -> usize {
        self.values.len().saturating_sub(self.head)
    }

    fn iter(&self) -> StreamRecordsIter<'_> {
        StreamRecordsIter {
            records: self,
            front: self.head,
            back: self.values.len(),
        }
    }

    fn front(&self) -> Option<&StreamReference> {
        self.values.get(self.head)
    }

    fn push_back(&mut self, reference: StreamReference) {
        self.values.push(reference);
    }

    fn pop_front(&mut self) -> Option<StreamReference> {
        let value = self.values.get(self.head)?.clone();
        self.head = self.head.saturating_add(1);
        let first_leaf = self.values.first_leaf_len();
        if first_leaf != 0 && self.head >= first_leaf {
            let removed = self.values.discard_prefix_leaves(self.head);
            self.head = self.head.saturating_sub(removed);
        }
        Some(value)
    }

    fn get_by_offset(&self, base_offset: u64, offset: u64) -> Option<&StreamReference> {
        let relative = usize::try_from(offset.checked_sub(base_offset)?).ok()?;
        self.values.get(self.head.checked_add(relative)?)
    }

    fn get_mut_by_offset(&mut self, base_offset: u64, offset: u64) -> Option<&mut StreamReference> {
        let relative = usize::try_from(offset.checked_sub(base_offset)?).ok()?;
        self.values.get_mut(self.head.checked_add(relative)?)
    }

    /// Mutable access by position from the live front, for callers that walk the whole stream
    /// without knowing its base offset.
    fn get_mut_relative(&mut self, relative: usize) -> Option<&mut StreamReference> {
        self.values.get_mut(self.head.checked_add(relative)?)
    }
}

impl fmt::Debug for StreamRecords {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_list().entries(self.iter()).finish()
    }
}

struct StreamRecordsIter<'a> {
    records: &'a StreamRecords,
    front: usize,
    back: usize,
}

impl<'a> Iterator for StreamRecordsIter<'a> {
    type Item = &'a StreamReference;

    fn next(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        let value = self.records.values.get(self.front);
        self.front = self.front.saturating_add(1);
        value
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.back.saturating_sub(self.front);
        (remaining, Some(remaining))
    }
}

impl DoubleEndedIterator for StreamRecordsIter<'_> {
    fn next_back(&mut self) -> Option<Self::Item> {
        if self.front >= self.back {
            return None;
        }
        self.back = self.back.saturating_sub(1);
        self.records.values.get(self.back)
    }
}

impl ExactSizeIterator for StreamRecordsIter<'_> {}

impl<'a> IntoIterator for &'a StreamRecords {
    type Item = &'a StreamReference;
    type IntoIter = StreamRecordsIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl Serialize for StreamRecords {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let mut sequence = serializer.serialize_seq(Some(self.len()))?;
        for reference in self.iter() {
            sequence.serialize_element(reference)?;
        }
        sequence.end()
    }
}

impl<'de> Deserialize<'de> for StreamRecords {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        struct StreamRecordsVisitor;

        impl<'de> Visitor<'de> for StreamRecordsVisitor {
            type Value = StreamRecords;

            fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
                formatter.write_str("a sequence of broker stream references")
            }

            fn visit_seq<A>(self, mut sequence: A) -> std::result::Result<Self::Value, A::Error>
            where
                A: SeqAccess<'de>,
            {
                let mut records = StreamRecords::default();
                while let Some(reference) = sequence.next_element::<StreamReference>()? {
                    records.push_back(reference);
                }
                Ok(records)
            }
        }

        deserializer.deserialize_seq(StreamRecordsVisitor)
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
struct StreamState {
    base_offset: u64,
    next_offset: u64,
    retained_bytes: u64,
    records: StreamRecords,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GroupState {
    generation: i32,
    leader: String,
    members: BTreeMap<String, GroupMember>,
    protocol: String,
    assignments: BTreeMap<String, Vec<u8>>,
    offsets: BTreeMap<(String, i32), u64>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct GroupMember {
    protocols: BTreeMap<String, Vec<u8>>,
    session_timeout_ms: u32,
    last_heartbeat_ms: i64,
}

/// One decoded Kafka record in an atomic partition append.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct KafkaBatchRecord {
    pub create_time_ms: Option<i64>,
    pub key: Option<Vec<u8>>,
    pub headers: BTreeMap<String, Vec<u8>>,
    pub payload: Vec<u8>,
    /// Kafka distinguishes a null record value from a present zero-length value.
    #[serde(default)]
    pub value_is_null: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AmqpBatchRecord {
    pub exchange: String,
    pub routing_key: String,
    pub mandatory: bool,
    pub properties: BTreeMap<String, Vec<u8>>,
    pub headers: BTreeMap<String, Vec<u8>>,
    pub payload: Vec<u8>,
}

/// Canonical mutation submitted through the same ordered local path as graph writes.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
pub enum BrokerCommand {
    CreateTopic {
        project: ProjectId,
        name: String,
        partitions: u16,
        retention: RetentionPolicy,
    },
    SetTopicRetention {
        project: ProjectId,
        name: String,
        retention: RetentionPolicy,
    },
    DeleteTopic {
        project: ProjectId,
        name: String,
    },
    ClearTopic {
        project: ProjectId,
        name: String,
    },
    CreateExchange {
        project: ProjectId,
        name: String,
        kind: AmqpExchangeKind,
        durable: bool,
        passive: bool,
    },
    CreateQueue {
        project: ProjectId,
        name: String,
        kind: QueueKind,
        durable: bool,
        passive: bool,
        dead_letter_exchange: Option<String>,
        dead_letter_routing_key: Option<String>,
        retention: RetentionPolicy,
        /// Connection claiming exclusive ownership, from AMQP `queue.declare` `exclusive`.
        exclusive_owner: Option<Uuid>,
        auto_delete: bool,
    },
    SetQueueRetention {
        project: ProjectId,
        name: String,
        retention: RetentionPolicy,
    },
    /// Attaches a consumer to a queue. `basic.consume` submits this so that auto-delete, exclusive
    /// ownership and `queue.delete` if-unused can be decided from canonical state.
    RegisterConsumer {
        project: ProjectId,
        queue: String,
        owner: Uuid,
        consumer: u64,
    },
    /// `basic.cancel`. Drops the queue when it was auto-delete and this was its last consumer.
    UnregisterConsumer {
        project: ProjectId,
        queue: String,
        owner: Uuid,
        consumer: u64,
    },
    /// A connection went away. Removes every consumer it held, deletes the exclusive queues it
    /// owned, and collects any auto-delete queue those consumers were the last on.
    ReleaseConnection {
        project: ProjectId,
        owner: Uuid,
    },
    BindQueue {
        project: ProjectId,
        exchange: String,
        queue: String,
        routing_key: String,
    },
    UnbindQueue {
        project: ProjectId,
        exchange: String,
        queue: String,
        routing_key: String,
    },
    /// Discards every ready message on a queue. Messages already handed to a consumer and not yet
    /// settled are left alone, matching `queue.purge` in AMQP 0-9-1 section 1.7.2.3.
    PurgeQueue {
        project: ProjectId,
        name: String,
    },
    DeleteQueue {
        project: ProjectId,
        name: String,
        /// AMQP `if-unused`: refuse while any consumer is still registered on the queue.
        if_unused: bool,
        if_empty: bool,
    },
    DeleteExchange {
        project: ProjectId,
        name: String,
        /// AMQP `if-unused`: refuse while any binding still names this exchange.
        if_unused: bool,
    },
    PublishKafkaBatch {
        project: ProjectId,
        topic: String,
        partition: i32,
        resolved_time_ms: i64,
        records: Vec<KafkaBatchRecord>,
    },
    PublishAmqp {
        project: ProjectId,
        exchange: String,
        routing_key: String,
        mandatory: bool,
        resolved_time_ms: i64,
        properties: BTreeMap<String, Vec<u8>>,
        headers: BTreeMap<String, Vec<u8>>,
        payload: Vec<u8>,
    },
    /// Publishes a wire-order group assembled from one AMQP connection read. The delta path
    /// materializes exactly these routed messages into one immutable payload segment; unrelated
    /// queues and payloads are neither scanned nor rebuilt.
    PublishAmqpBatch {
        project: ProjectId,
        resolved_time_ms: i64,
        records: Vec<AmqpBatchRecord>,
    },
    /// Compact form for the common case where a socket read contains publishes with one AMQP
    /// envelope. The delta path routes once and stores the envelope once; work tracks only the
    /// bodies in this batch and never scans unrelated queues or messages.
    PublishAmqpUniformBatch {
        project: ProjectId,
        resolved_time_ms: i64,
        exchange: String,
        routing_key: String,
        mandatory: bool,
        properties: BTreeMap<String, Vec<u8>>,
        headers: BTreeMap<String, Vec<u8>>,
        payloads: Vec<Vec<u8>>,
    },
    CommitOffset {
        project: ProjectId,
        group: String,
        topic: String,
        partition: i32,
        offset: u64,
        generation: Option<i32>,
        member: Option<String>,
    },
    JoinGroup {
        project: ProjectId,
        group: String,
        member: String,
        session_timeout_ms: u32,
        resolved_time_ms: i64,
        protocols: BTreeMap<String, Vec<u8>>,
    },
    SyncGroup {
        project: ProjectId,
        group: String,
        generation: i32,
        assignments: BTreeMap<String, Vec<u8>>,
    },
    LeaveGroup {
        project: ProjectId,
        group: String,
        member: String,
    },
    HeartbeatGroup {
        project: ProjectId,
        group: String,
        generation: i32,
        member: String,
        resolved_time_ms: i64,
    },
    Ack {
        project: ProjectId,
        queue: String,
        owner: Uuid,
        consumer: u64,
        delivery_tag: u64,
        multiple: bool,
    },
    Nack {
        project: ProjectId,
        queue: String,
        owner: Uuid,
        consumer: u64,
        delivery_tag: u64,
        multiple: bool,
        requeue: bool,
    },
    DeliverQueue {
        project: ProjectId,
        queue: String,
        offset: StreamOffset,
        maximum: u32,
        owner: Uuid,
        consumer: u64,
        automatic_ack: bool,
        resolved_time_ms: i64,
    },
    RenewDeliveryLease {
        project: ProjectId,
        owner: Uuid,
        resolved_time_ms: i64,
    },
    ReleaseDeliveryLease {
        project: ProjectId,
        owner: Uuid,
    },
    Retain {
        project: ProjectId,
        resolved_time_ms: i64,
    },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum BrokerReply {
    Declared,
    TopicAlreadyExists,
    Deleted,
    Unbound,
    /// Ready messages discarded by `queue.purge` or still present when `queue.delete` succeeded.
    /// AMQP carries this as a `long` (u32); the engine saturates rather than wrapping.
    MessagesDiscarded {
        message_count: u32,
    },
    Published {
        message: MessageId,
        offsets: Vec<(StreamId, u64)>,
    },
    KafkaBatchPublished {
        first_offset: u64,
        record_count: u32,
    },
    AmqpBatchPublished {
        offsets: Vec<Vec<(StreamId, u64)>>,
    },
    AmqpUniformBatchPublished {
        record_count: u32,
        routed: bool,
    },
    Group {
        generation: i32,
        leader: String,
        members: Vec<String>,
        protocol: String,
        metadata: BTreeMap<String, Vec<u8>>,
    },
    OffsetCommitted,
    Acknowledged,
    Deliveries(Vec<Delivery>),
    Retained {
        removed_references: u64,
        removed_payloads: u64,
    },
}

/// Result of one coordinator submission.
#[derive(Clone, Debug)]
pub struct BrokerCommit {
    pub bookmark: Bookmark,
    pub reply: BrokerReply,
    pub application: ApplicationWait,
}

/// Protocol-facing adapter for the standalone ordered write path.
pub trait BrokerCoordinator: Send + Sync {
    fn submit(&self, command: BrokerCommand, wait: CommitAcknowledgement) -> Result<BrokerCommit>;
    fn submit_from(
        &self,
        connection: ConnectionId,
        command: BrokerCommand,
        wait: CommitAcknowledgement,
    ) -> Result<BrokerCommit> {
        let _ = connection;
        self.submit(command, wait)
    }
    fn submit_with_timeout(
        &self,
        command: BrokerCommand,
        wait: CommitAcknowledgement,
        _timeout_millis: u32,
    ) -> Result<BrokerCommit> {
        self.submit(command, wait)
    }
    fn submit_with_timeout_from(
        &self,
        connection: ConnectionId,
        command: BrokerCommand,
        wait: CommitAcknowledgement,
        timeout_millis: u32,
    ) -> Result<BrokerCommit> {
        let _ = connection;
        self.submit_with_timeout(command, wait, timeout_millis)
    }
    fn snapshot(&self) -> Result<BrokerStateMachine>;
    fn reclaim_payload_storage(&self) -> Result<u64> {
        Ok(0)
    }
    fn subscribe_changes(&self) -> tokio::sync::watch::Receiver<u64> {
        tokio::sync::watch::channel(0).1
    }
    fn projects_with_state(&self) -> Result<BTreeSet<ProjectId>> {
        Ok(self.snapshot()?.projects_with_state())
    }
    fn topic_metadata(&self, project: ProjectId) -> Result<Vec<(String, usize)>> {
        Ok(self.snapshot()?.topic_metadata(project))
    }
    fn committed_offset(
        &self,
        project: ProjectId,
        group: &str,
        topic: &str,
        partition: i32,
    ) -> Result<Option<u64>> {
        Ok(self
            .snapshot()?
            .committed_offset(project, group, topic, partition))
    }
    fn group_leader(
        &self,
        project: ProjectId,
        group: &str,
        generation: i32,
    ) -> Result<Option<String>> {
        Ok(self
            .snapshot()?
            .group_leader(project, group, generation)
            .map(str::to_owned))
    }
    fn group_assignment(
        &self,
        project: ProjectId,
        group: &str,
        generation: i32,
        member: &str,
    ) -> Result<Option<Vec<u8>>> {
        Ok(self
            .snapshot()?
            .group_assignment(project, group, generation, member))
    }
    fn fetch_partition(
        &self,
        project: ProjectId,
        topic: &str,
        partition: i32,
        offset: u64,
        maximum_bytes: usize,
    ) -> Result<Vec<(u64, Arc<PayloadRecord>)>>;
    fn list_offset(
        &self,
        project: ProjectId,
        topic: &str,
        partition: i32,
        timestamp: i64,
    ) -> Result<Option<(u64, i64)>>;
    fn queue_info(&self, project: ProjectId, name: &str) -> Result<Option<QueueInfo>>;
    fn fetch_stream_queue(
        &self,
        project: ProjectId,
        queue: &str,
        offset: StreamOffset,
        maximum: usize,
        consumer: u64,
        automatic_ack: bool,
    ) -> Result<Vec<Delivery>>;
}

/// Per-transport view that binds every mutation to one stable connection identity without
/// changing broker semantics or adding protocol-visible state.
pub(crate) struct ConnectionBrokerCoordinator {
    inner: Arc<dyn BrokerCoordinator>,
    connection: ConnectionId,
}

impl ConnectionBrokerCoordinator {
    pub(crate) fn new(inner: Arc<dyn BrokerCoordinator>) -> Self {
        Self {
            inner,
            connection: ConnectionId::new(),
        }
    }
}

impl BrokerCoordinator for ConnectionBrokerCoordinator {
    fn submit(&self, command: BrokerCommand, wait: CommitAcknowledgement) -> Result<BrokerCommit> {
        self.inner.submit_from(self.connection, command, wait)
    }

    fn submit_with_timeout(
        &self,
        command: BrokerCommand,
        wait: CommitAcknowledgement,
        timeout_millis: u32,
    ) -> Result<BrokerCommit> {
        self.inner
            .submit_with_timeout_from(self.connection, command, wait, timeout_millis)
    }

    fn snapshot(&self) -> Result<BrokerStateMachine> {
        self.inner.snapshot()
    }

    fn reclaim_payload_storage(&self) -> Result<u64> {
        self.inner.reclaim_payload_storage()
    }

    fn subscribe_changes(&self) -> tokio::sync::watch::Receiver<u64> {
        self.inner.subscribe_changes()
    }

    fn projects_with_state(&self) -> Result<BTreeSet<ProjectId>> {
        self.inner.projects_with_state()
    }

    fn topic_metadata(&self, project: ProjectId) -> Result<Vec<(String, usize)>> {
        self.inner.topic_metadata(project)
    }

    fn committed_offset(
        &self,
        project: ProjectId,
        group: &str,
        topic: &str,
        partition: i32,
    ) -> Result<Option<u64>> {
        self.inner
            .committed_offset(project, group, topic, partition)
    }

    fn group_leader(
        &self,
        project: ProjectId,
        group: &str,
        generation: i32,
    ) -> Result<Option<String>> {
        self.inner.group_leader(project, group, generation)
    }

    fn group_assignment(
        &self,
        project: ProjectId,
        group: &str,
        generation: i32,
        member: &str,
    ) -> Result<Option<Vec<u8>>> {
        self.inner
            .group_assignment(project, group, generation, member)
    }

    fn fetch_partition(
        &self,
        project: ProjectId,
        topic: &str,
        partition: i32,
        offset: u64,
        maximum_bytes: usize,
    ) -> Result<Vec<(u64, Arc<PayloadRecord>)>> {
        self.inner
            .fetch_partition(project, topic, partition, offset, maximum_bytes)
    }

    fn list_offset(
        &self,
        project: ProjectId,
        topic: &str,
        partition: i32,
        timestamp: i64,
    ) -> Result<Option<(u64, i64)>> {
        self.inner.list_offset(project, topic, partition, timestamp)
    }

    fn queue_info(&self, project: ProjectId, name: &str) -> Result<Option<QueueInfo>> {
        self.inner.queue_info(project, name)
    }

    fn fetch_stream_queue(
        &self,
        project: ProjectId,
        queue: &str,
        offset: StreamOffset,
        maximum: usize,
        consumer: u64,
        automatic_ack: bool,
    ) -> Result<Vec<Delivery>> {
        self.inner
            .fetch_stream_queue(project, queue, offset, maximum, consumer, automatic_ack)
    }
}

/// Deterministic canonical broker state applied from committed log entries.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct BrokerStateMachine {
    next_message_id: u64,
    next_stream_id: u64,
    next_payload_segment_id: u64,
    payloads: MessageTable,
    payload_segments: PayloadSegmentTable,
    topics: CowArc<BTreeMap<(ProjectId, String), Topic>>,
    exchanges: CowArc<BTreeMap<(ProjectId, String), Exchange>>,
    queues: CowArc<BTreeMap<(ProjectId, String), Queue>>,
    bindings: CowArc<Vec<((ProjectId, String), Binding)>>,
    streams: StreamTable,
    groups: CowArc<BTreeMap<(ProjectId, String), Arc<GroupState>>>,
    #[serde(default)]
    delivery_leases: CowArc<BTreeMap<(ProjectId, Uuid), i64>>,
    /// Consumers currently attached to each queue, carrying the connection that owns them.
    /// Registration has to be canonical rather than connection-local: `queue.delete` if-unused,
    /// auto-delete queues and exclusive queues are all defined in terms of who is consuming, and
    /// no connection can observe another connection's consumers.
    #[serde(default)]
    queue_consumers: CowArc<BTreeMap<(ProjectId, String), BTreeSet<(Uuid, u64)>>>,
}

impl BrokerStateMachine {
    /// Projects with canonical broker state. The maintenance worker uses this derived set for retention;
    /// project identifiers are not duplicated in a maintenance registry.
    #[must_use]
    pub fn projects_with_state(&self) -> BTreeSet<ProjectId> {
        let mut projects = BTreeSet::new();
        projects.extend(self.topics.keys().map(|(project, _)| *project));
        projects.extend(self.exchanges.keys().map(|(project, _)| *project));
        projects.extend(self.queues.keys().map(|(project, _)| *project));
        projects.extend(self.groups.keys().map(|(project, _)| *project));
        projects.extend(self.delivery_leases.keys().map(|(project, _)| *project));
        projects.extend(
            self.payload_segments
                .iter()
                .filter_map(|(_, segment)| segment.descriptor.project_id),
        );
        projects
    }

    /// Returns whether the project owns no canonical broker declarations or cursors.
    #[must_use]
    pub(crate) fn project_is_empty(&self, project: ProjectId) -> bool {
        !self
            .topics
            .keys()
            .any(|(candidate, _)| *candidate == project)
            && !self
                .exchanges
                .keys()
                .any(|(candidate, _)| *candidate == project)
            && !self
                .queues
                .keys()
                .any(|(candidate, _)| *candidate == project)
            && !self
                .bindings
                .iter()
                .any(|((candidate, _), _)| *candidate == project)
            && !self
                .groups
                .keys()
                .any(|(candidate, _)| *candidate == project)
            && !self
                .delivery_leases
                .keys()
                .any(|(candidate, _)| *candidate == project)
    }

    /// Removes all canonical broker state owned by one dropped project. Content-addressed
    /// payload files may remain unreferenced until storage compaction.
    pub(crate) fn drop_project(&mut self, project: ProjectId) -> Result<()> {
        let mut removed_streams = BTreeSet::new();
        self.topics.retain(|(candidate, _), topic| {
            if *candidate == project {
                removed_streams.extend(topic.partitions.iter().copied());
                false
            } else {
                true
            }
        });
        self.queues.retain(|(candidate, _), queue| {
            if *candidate == project {
                removed_streams.insert(queue.stream);
                false
            } else {
                true
            }
        });
        self.exchanges
            .retain(|(candidate, _), _| *candidate != project);
        self.bindings
            .retain(|((candidate, _), _)| *candidate != project);
        self.groups
            .retain(|(candidate, _), _| *candidate != project);
        self.delivery_leases
            .retain(|(candidate, _), _| *candidate != project);
        let removed_payloads = removed_streams
            .iter()
            .filter_map(|stream| self.streams.get(stream))
            .flat_map(|stream| stream.records.iter().map(|record| record.message))
            .collect::<BTreeSet<_>>();
        self.streams
            .retain(|stream, _| !removed_streams.contains(stream));
        for message in removed_payloads {
            self.remove_payload(message)?;
        }
        self.validate_state()
    }

    pub(crate) fn validate_state(&self) -> Result<()> {
        if self
            .payloads
            .last_key_value()
            .is_some_and(|(id, _)| id.0 > self.next_message_id)
            || self
                .streams
                .last_key_value()
                .is_some_and(|(id, _)| id.0 > self.next_stream_id)
            || self
                .payload_segments
                .iter()
                .last()
                .is_some_and(|(id, _)| id > self.next_payload_segment_id)
        {
            return Err(Error::new(
                crate::ErrorCode::CorruptStorage,
                "broker allocator cursor precedes canonical records",
            ));
        }
        let mut stream_projects = BTreeMap::<StreamId, ProjectId>::new();
        for ((project, _), topic) in &*self.topics {
            if topic.partitions.is_empty()
                || topic.partitions.len() > 4_096
                || topic.retention.validate().is_err()
            {
                return Err(Error::new(
                    crate::ErrorCode::CorruptStorage,
                    "broker topic partition set is invalid",
                ));
            }
            for stream in &topic.partitions {
                if stream_projects.insert(*stream, *project).is_some() {
                    return Err(Error::new(
                        crate::ErrorCode::CorruptStorage,
                        "broker stream is assigned more than once",
                    ));
                }
            }
        }
        for ((project, _), queue) in &*self.queues {
            if queue.retention.validate().is_err() {
                return Err(Error::new(
                    crate::ErrorCode::CorruptStorage,
                    "broker queue retention policy is invalid",
                ));
            }
            if stream_projects.insert(queue.stream, *project).is_some() {
                return Err(Error::new(
                    crate::ErrorCode::CorruptStorage,
                    "broker queue stream is assigned more than once",
                ));
            }
        }
        if stream_projects.len() != self.streams.len()
            || stream_projects
                .keys()
                .any(|stream| !self.streams.contains_key(stream))
        {
            return Err(Error::new(
                crate::ErrorCode::CorruptStorage,
                "broker stream declaration set is incomplete",
            ));
        }

        let mut live_payloads = BTreeSet::new();
        let mut live_segment_records = BTreeMap::<u64, u32>::new();
        let mut live_delivery_leases = BTreeSet::new();
        for (stream_id, stream) in self.streams.iter() {
            if stream.base_offset > stream.next_offset {
                return Err(Error::new(
                    crate::ErrorCode::CorruptStorage,
                    "broker stream cursor moved backwards",
                ));
            }
            let mut expected = stream.base_offset;
            let mut retained_bytes = 0_u64;
            let project = stream_projects.get(&stream_id).ok_or_else(|| {
                Error::new(
                    crate::ErrorCode::CorruptStorage,
                    "broker stream has no project owner",
                )
            })?;
            for reference in &stream.records {
                if reference.offset != expected {
                    return Err(Error::new(
                        crate::ErrorCode::CorruptStorage,
                        "broker stream offsets are not consecutive",
                    ));
                }
                expected = expected.checked_add(1).ok_or_else(|| {
                    Error::new(
                        crate::ErrorCode::CorruptStorage,
                        "broker stream offset overflow",
                    )
                })?;
                let payload = self.payloads.get(&reference.message).ok_or_else(|| {
                    Error::new(
                        crate::ErrorCode::CorruptStorage,
                        "broker reference has no payload",
                    )
                })?;
                let segment = self.payload_segments.get(payload.segment).ok_or_else(|| {
                    Error::new(
                        crate::ErrorCode::CorruptStorage,
                        "broker payload segment is missing",
                    )
                })?;
                if segment.descriptor.family != SegmentFamily::BrokerPayload
                    || segment.descriptor.project_id != Some(*project)
                    || payload.payload_bytes > 256 * 1024 * 1024
                    || payload.retained_bytes < payload.payload_bytes
                    || payload.retained_bytes != payload.location.bytes
                    || payload
                        .location
                        .offset
                        .checked_add(payload.location.bytes)
                        .is_none_or(|end| end > segment.descriptor.bytes)
                {
                    return Err(Error::new(
                        crate::ErrorCode::CorruptStorage,
                        "broker payload descriptor has invalid scope or size",
                    ));
                }
                if live_payloads.insert(reference.message) {
                    let count = live_segment_records.entry(payload.segment).or_default();
                    *count = count.checked_add(1).ok_or_else(|| {
                        Error::new(
                            crate::ErrorCode::CorruptStorage,
                            "broker payload segment live count overflow",
                        )
                    })?;
                }
                retained_bytes = retained_bytes
                    .checked_add(self.accounted_payload_bytes(payload)?)
                    .ok_or_else(|| {
                        Error::new(
                            crate::ErrorCode::CorruptStorage,
                            "broker retained byte accounting overflow",
                        )
                    })?;
                if let DeliveryState::Unacked { owner, .. } = reference.delivery {
                    if owner.is_nil() {
                        return Err(Error::new(
                            crate::ErrorCode::CorruptStorage,
                            "broker delivery lease owner is invalid",
                        ));
                    }
                    live_delivery_leases.insert((*project, owner));
                }
            }
            if expected != stream.next_offset || retained_bytes != stream.retained_bytes {
                return Err(Error::new(
                    crate::ErrorCode::CorruptStorage,
                    "broker stream cursor or retained byte accounting is invalid",
                ));
            }
        }
        if live_payloads.len() != self.payloads.len()
            || self.payloads.keys().any(|id| !live_payloads.contains(&id))
            || self.payloads.iter().any(|(id, payload)| id != payload.id)
        {
            return Err(Error::new(
                crate::ErrorCode::CorruptStorage,
                "broker payload set differs from live stream references",
            ));
        }
        if live_segment_records.len() != self.payload_segments.len()
            || self.payload_segments.iter().any(|(id, segment)| {
                segment.id != id
                    || segment.live_records == 0
                    || live_segment_records.get(&id).copied() != Some(segment.live_records)
            })
        {
            return Err(Error::new(
                crate::ErrorCode::CorruptStorage,
                "broker payload segment table differs from live messages",
            ));
        }
        if self
            .delivery_leases
            .keys()
            .copied()
            .collect::<BTreeSet<_>>()
            != live_delivery_leases
        {
            return Err(Error::new(
                crate::ErrorCode::CorruptStorage,
                "broker delivery lease set differs from unacknowledged deliveries",
            ));
        }
        for ((project, exchange), binding) in &*self.bindings {
            if binding.exchange != *exchange
                || !self.exchanges.contains_key(&(*project, exchange.clone()))
                || !self.queues.contains_key(&(*project, binding.queue.clone()))
            {
                return Err(Error::new(
                    crate::ErrorCode::CorruptStorage,
                    "broker binding endpoint is missing or cross-project",
                ));
            }
        }
        for ((project, _), group) in &*self.groups {
            if group.generation < 0
                || (!group.members.is_empty()
                    && (!group.members.contains_key(&group.leader)
                        || select_group_protocol(&group.members).map_err(|_| {
                            Error::new(
                                crate::ErrorCode::CorruptStorage,
                                "broker consumer group has no common protocol",
                            )
                        })? != group.protocol))
                || group
                    .assignments
                    .keys()
                    .any(|member| !group.members.contains_key(member))
            {
                return Err(Error::new(
                    crate::ErrorCode::CorruptStorage,
                    "broker consumer group state is invalid",
                ));
            }
            for ((topic, partition), offset) in &group.offsets {
                let stream = self.partition_stream(*project, topic, *partition)?;
                let state = self.streams.get(&stream).ok_or_else(|| {
                    Error::new(
                        crate::ErrorCode::CorruptStorage,
                        "broker committed offset stream is missing",
                    )
                })?;
                if *offset > state.next_offset {
                    return Err(Error::new(
                        crate::ErrorCode::CorruptStorage,
                        "broker committed offset exceeds the log end",
                    ));
                }
            }
        }
        Ok(())
    }

    /// Performs every command-dependent semantic check without cloning canonical streams or
    /// payload segments. Committed application may therefore mutate in place under the database
    /// publication lock; any later failure is an invariant/storage failure, not a client error.
    pub fn validate_command(&self, command: &BrokerCommand) -> Result<()> {
        match command {
            BrokerCommand::CreateTopic {
                project,
                name,
                partitions,
                retention,
            } => {
                valid_name(name)?;
                retention.validate()?;
                if *partitions == 0 || *partitions > 4_096 {
                    return Err(Error::invalid_data(
                        "topic partition count must be 1..=4096",
                    ));
                }
                if let Some(existing) = self.topics.get(&(*project, name.clone())) {
                    if existing.partitions.len() != *partitions as usize {
                        return Err(Error::invalid_data("topic partition count is immutable"));
                    }
                } else {
                    self.next_stream_id
                        .checked_add(u64::from(*partitions))
                        .ok_or_else(|| Error::internal("stream ID exhausted"))?;
                }
            }
            BrokerCommand::SetTopicRetention {
                project,
                name,
                retention,
            } => {
                retention.validate()?;
                if !self.topics.contains_key(&(*project, name.clone())) {
                    return Err(Error::invalid_data("topic does not exist"));
                }
            }
            BrokerCommand::DeleteTopic { project, name }
            | BrokerCommand::ClearTopic { project, name } => {
                if !self.topics.contains_key(&(*project, name.clone())) {
                    return Err(Error::invalid_data("topic does not exist"));
                }
            }
            BrokerCommand::CreateExchange {
                project,
                name,
                kind,
                durable,
                passive,
            } => {
                valid_name(name)?;
                if let Some(existing) = self.exchanges.get(&(*project, name.clone())) {
                    if existing.kind != *kind || existing.durable != *durable {
                        return Err(Error::invalid_data(
                            "exchange redeclaration is incompatible",
                        ));
                    }
                } else if *passive {
                    return Err(Error::invalid_data("passive exchange does not exist"));
                }
            }
            BrokerCommand::CreateQueue {
                project,
                name,
                kind,
                durable,
                passive,
                dead_letter_exchange,
                retention,
                exclusive_owner,
                ..
            } => {
                valid_name(name)?;
                retention.validate()?;
                if let Some(existing) = self.queues.get(&(*project, name.clone())) {
                    if existing.kind != *kind || existing.durable != *durable {
                        return Err(Error::invalid_data("queue redeclaration is incompatible"));
                    }
                    // An exclusive queue is locked to its declaring connection, so a redeclaration
                    // from anywhere else is refused even when every other property agrees.
                    if existing.exclusive_owner.is_some()
                        && existing.exclusive_owner != *exclusive_owner
                    {
                        return Err(Error::new(
                            crate::ErrorCode::AuthorizationDenied,
                            "queue is exclusive to another connection",
                        ));
                    }
                } else {
                    if *passive {
                        return Err(Error::invalid_data("passive queue does not exist"));
                    }
                    self.next_stream_id
                        .checked_add(1)
                        .ok_or_else(|| Error::internal("stream ID exhausted"))?;
                }
                if let Some(exchange) = dead_letter_exchange
                    && !exchange.is_empty()
                    && !self.exchanges.contains_key(&(*project, exchange.clone()))
                {
                    return Err(Error::invalid_data("dead-letter exchange does not exist"));
                }
            }
            BrokerCommand::SetQueueRetention {
                project,
                name,
                retention,
            } => {
                retention.validate()?;
                if !self.queues.contains_key(&(*project, name.clone())) {
                    return Err(Error::invalid_data("queue does not exist"));
                }
            }
            BrokerCommand::BindQueue {
                project,
                exchange,
                queue,
                routing_key,
            } => {
                valid_routing_key(routing_key)?;
                if !self.exchanges.contains_key(&(*project, exchange.clone()))
                    || !self.queues.contains_key(&(*project, queue.clone()))
                {
                    return Err(Error::invalid_data("binding endpoint does not exist"));
                }
            }
            BrokerCommand::UnbindQueue {
                project,
                exchange,
                queue,
                routing_key,
            } => {
                valid_routing_key(routing_key)?;
                if !self.binding_exists(*project, exchange, queue, routing_key) {
                    return Err(Error::invalid_data("binding does not exist"));
                }
            }
            BrokerCommand::PurgeQueue { project, name } => {
                if !self.queues.contains_key(&(*project, name.clone())) {
                    return Err(Error::invalid_data("queue does not exist"));
                }
            }
            BrokerCommand::RegisterConsumer {
                project,
                queue,
                owner,
                ..
            } => {
                let declared = self
                    .queues
                    .get(&(*project, queue.clone()))
                    .ok_or_else(|| Error::invalid_data("queue does not exist"))?;
                if let Some(exclusive) = declared.exclusive_owner
                    && exclusive != *owner
                {
                    return Err(Error::new(
                        crate::ErrorCode::AuthorizationDenied,
                        "queue is exclusive to another connection",
                    ));
                }
            }
            // Both are idempotent teardown: cancelling an unknown consumer or releasing a
            // connection that registered nothing has to succeed, or a client that disconnects
            // mid-cancel would wedge the queue it was consuming.
            BrokerCommand::UnregisterConsumer { .. } | BrokerCommand::ReleaseConnection { .. } => {}
            BrokerCommand::DeleteQueue {
                project,
                name,
                if_unused,
                if_empty,
            } => {
                self.queue_deletion_counts(*project, name, *if_unused, *if_empty)?;
            }
            BrokerCommand::DeleteExchange {
                project,
                name,
                if_unused,
            } => {
                if !self.exchanges.contains_key(&(*project, name.clone())) {
                    return Err(Error::invalid_data("exchange does not exist"));
                }
                if *if_unused && self.exchange_has_bindings(*project, name) {
                    return Err(Error::invalid_data("exchange still has bindings"));
                }
            }
            BrokerCommand::PublishKafkaBatch {
                project,
                topic,
                partition,
                records,
                ..
            } => {
                validate_kafka_batch(records)?;
                let stream = self.partition_stream(*project, topic, *partition)?;
                self.validate_stream_appends(&[stream], records.len() as u64)?;
            }
            BrokerCommand::PublishAmqp {
                project,
                exchange,
                routing_key,
                payload,
                ..
            } => {
                validate_payload(payload)?;
                let targets = self.amqp_targets(*project, exchange, routing_key)?;
                self.validate_stream_appends(&targets.into_iter().collect::<Vec<_>>(), 1)?;
            }
            BrokerCommand::PublishAmqpBatch {
                project, records, ..
            } => {
                validate_amqp_batch(records)?;
                let mut counts = BTreeMap::<StreamId, u64>::new();
                for record in records {
                    for stream in
                        self.amqp_targets(*project, &record.exchange, &record.routing_key)?
                    {
                        *counts.entry(stream).or_default() += 1;
                    }
                }
                for (stream, count) in counts {
                    self.validate_stream_appends(&[stream], count)?;
                }
            }
            BrokerCommand::PublishAmqpUniformBatch {
                project,
                exchange,
                routing_key,
                properties,
                headers,
                payloads,
                ..
            } => {
                validate_amqp_uniform_batch(properties, headers, payloads)?;
                let targets = self.amqp_targets(*project, exchange, routing_key)?;
                self.validate_stream_appends(
                    &targets.into_iter().collect::<Vec<_>>(),
                    payloads.len() as u64,
                )?;
            }
            BrokerCommand::CommitOffset {
                project,
                group,
                topic,
                partition,
                offset,
                generation,
                member,
            } => {
                let stream = self.partition_stream(*project, topic, *partition)?;
                if *offset
                    > self
                        .streams
                        .get(&stream)
                        .ok_or_else(|| Error::internal("topic partition stream is missing"))?
                        .next_offset
                {
                    return Err(Error::invalid_data(
                        "committed offset exceeds the partition log end",
                    ));
                }
                let state = self
                    .groups
                    .get(&(*project, group.clone()))
                    .ok_or_else(|| Error::invalid_data("consumer group does not exist"))?;
                match (generation, member) {
                    (Some(generation), Some(member))
                        if *generation == state.generation
                            && state.members.contains_key(member) => {}
                    (None, None) => {}
                    _ => {
                        return Err(Error::new(
                            crate::ErrorCode::TransactionConflict,
                            "consumer group generation or member changed",
                        ));
                    }
                }
            }
            BrokerCommand::JoinGroup {
                project,
                group,
                member,
                session_timeout_ms,
                protocols,
                resolved_time_ms,
            } => {
                valid_name(group)?;
                valid_name(member)?;
                validate_group_member(*session_timeout_ms, protocols)?;
                if let Some(state) = self.groups.get(&(*project, group.clone())) {
                    let mut candidate = state.members.clone();
                    candidate.retain(|_, member| {
                        resolved_time_ms.saturating_sub(member.last_heartbeat_ms)
                            < i64::from(member.session_timeout_ms)
                    });
                    candidate.insert(
                        member.clone(),
                        GroupMember {
                            protocols: protocols.clone(),
                            session_timeout_ms: *session_timeout_ms,
                            last_heartbeat_ms: 0,
                        },
                    );
                    select_group_protocol(&candidate)?;
                    if !state.members.contains_key(member)
                        || state.members.get(member).is_some_and(|existing| {
                            existing.protocols != *protocols
                                || existing.session_timeout_ms != *session_timeout_ms
                        })
                    {
                        state
                            .generation
                            .checked_add(1)
                            .ok_or_else(|| Error::internal("group generation exhausted"))?;
                    }
                }
            }
            BrokerCommand::SyncGroup {
                project,
                group,
                generation,
                assignments,
            } => {
                let state = self
                    .groups
                    .get(&(*project, group.clone()))
                    .ok_or_else(|| Error::invalid_data("consumer group does not exist"))?;
                if state.generation != *generation {
                    return Err(Error::new(
                        crate::ErrorCode::TransactionConflict,
                        "consumer group generation changed",
                    ));
                }
                if !assignments.is_empty()
                    && (assignments.len() != state.members.len()
                        || assignments
                            .keys()
                            .any(|assigned| !state.members.contains_key(assigned)))
                {
                    return Err(Error::invalid_data(
                        "group sync must install one assignment per member",
                    ));
                }
            }
            BrokerCommand::LeaveGroup { project, group, .. } => {
                if !self.groups.contains_key(&(*project, group.clone())) {
                    return Err(Error::invalid_data("consumer group does not exist"));
                }
            }
            BrokerCommand::HeartbeatGroup {
                project,
                group,
                generation,
                member,
                resolved_time_ms,
            } => {
                let state = self
                    .groups
                    .get(&(*project, group.clone()))
                    .ok_or_else(|| Error::invalid_data("consumer group does not exist"))?;
                let expired_member_exists = state.members.values().any(|member| {
                    resolved_time_ms.saturating_sub(member.last_heartbeat_ms)
                        >= i64::from(member.session_timeout_ms)
                });
                if expired_member_exists
                    || state.generation != *generation
                    || !state.members.get(member).is_some_and(|member| {
                        resolved_time_ms.saturating_sub(member.last_heartbeat_ms)
                            < i64::from(member.session_timeout_ms)
                    })
                {
                    return Err(Error::new(
                        crate::ErrorCode::TransactionConflict,
                        "consumer group generation or member changed",
                    ));
                }
            }
            BrokerCommand::Ack {
                project,
                queue,
                owner,
                consumer,
                delivery_tag,
                multiple,
            } => self.validate_settlement(
                *project,
                queue,
                *owner,
                *consumer,
                *delivery_tag,
                *multiple,
            )?,
            BrokerCommand::Nack {
                project,
                queue,
                owner,
                consumer,
                delivery_tag,
                multiple,
                requeue,
            } => {
                self.validate_settlement(
                    *project,
                    queue,
                    *owner,
                    *consumer,
                    *delivery_tag,
                    *multiple,
                )?;
                if !requeue {
                    self.validate_dead_letter_appends(
                        *project,
                        queue,
                        *owner,
                        *consumer,
                        *delivery_tag,
                        *multiple,
                    )?;
                }
            }
            BrokerCommand::DeliverQueue {
                project,
                queue,
                offset,
                maximum,
                owner,
                ..
            } => {
                if owner.is_nil() {
                    return Err(Error::invalid_data("delivery lease owner is invalid"));
                }
                if *maximum == 0 || *maximum > 65_536 {
                    return Err(Error::invalid_data("delivery maximum must be 1..=65536"));
                }
                let queue = self
                    .queues
                    .get(&(*project, queue.clone()))
                    .ok_or_else(|| Error::invalid_data("queue does not exist"))?;
                let stream = self
                    .streams
                    .get(&queue.stream)
                    .ok_or_else(|| Error::internal("queue stream is missing"))?;
                let start = self.resolve_offset(stream, *offset)?;
                let mut response_bytes = 0_usize;
                let mut records = 0_u32;
                for reference in stream
                    .records
                    .iter()
                    .filter(|record| record.offset >= start)
                {
                    if records >= *maximum {
                        break;
                    }
                    if queue.kind == QueueKind::Classic
                        && reference.delivery != DeliveryState::Ready
                    {
                        continue;
                    }
                    let payload = self
                        .payloads
                        .get(&reference.message)
                        .ok_or_else(|| Error::internal("broker payload is missing"))?;
                    let next_response_bytes = response_bytes
                        .checked_add(
                            usize::try_from(self.accounted_payload_bytes(payload)?).map_err(
                                |_| Error::internal("broker delivery length exceeds this platform"),
                            )?,
                        )
                        .and_then(|bytes| bytes.checked_add(BROKER_DELIVERY_ENVELOPE_BYTES))
                        .ok_or_else(|| Error::internal("delivery response size overflow"))?;
                    if next_response_bytes > MAX_BROKER_DELIVERY_RESULT_BYTES {
                        if records == 0 {
                            return Err(Error::new(
                                crate::ErrorCode::ResultBudgetExceeded,
                                "delivery response exceeds the durable result bound",
                            ));
                        }
                        break;
                    }
                    response_bytes = next_response_bytes;
                    records += 1;
                }
            }
            BrokerCommand::RenewDeliveryLease { owner, .. }
            | BrokerCommand::ReleaseDeliveryLease { owner, .. } => {
                if owner.is_nil() {
                    return Err(Error::invalid_data("delivery lease owner is invalid"));
                }
            }
            BrokerCommand::Retain { project, .. } => {
                for stream in self
                    .topics
                    .iter()
                    .filter(|((id, _), _)| id == project)
                    .flat_map(|(_, topic)| &topic.partitions)
                    .chain(
                        self.queues
                            .iter()
                            .filter(|((id, _), _)| id == project)
                            .map(|(_, queue)| &queue.stream),
                    )
                {
                    if !self.streams.contains_key(stream) {
                        return Err(Error::internal("retention stream is missing"));
                    }
                }
            }
        }
        Ok(())
    }

    fn validate_stream_appends(&self, streams: &[StreamId], count: u64) -> Result<()> {
        self.next_message_id
            .checked_add(count)
            .ok_or_else(|| Error::internal("message ID exhausted"))?;
        for stream in streams {
            self.streams
                .get(stream)
                .ok_or_else(|| Error::internal("broker stream does not exist"))?
                .next_offset
                .checked_add(count)
                .ok_or_else(|| Error::internal("broker offset exhausted"))?;
        }
        Ok(())
    }

    fn amqp_targets(
        &self,
        project: ProjectId,
        exchange: &str,
        routing_key: &str,
    ) -> Result<BTreeSet<StreamId>> {
        if exchange.is_empty() {
            return Ok(self
                .queues
                .get(&(project, routing_key.to_owned()))
                .map(|queue| BTreeSet::from([queue.stream]))
                .unwrap_or_default());
        }
        let declaration = self
            .exchanges
            .get(&(project, exchange.to_owned()))
            .ok_or_else(|| Error::invalid_data("exchange does not exist"))?;
        Ok(self
            .bindings
            .iter()
            .filter(|((id, bound_exchange), binding)| {
                *id == project
                    && bound_exchange == exchange
                    && route_matches(declaration.kind, &binding.routing_key, routing_key)
            })
            .filter_map(|(_, binding)| {
                self.queues
                    .get(&(project, binding.queue.clone()))
                    .map(|queue| queue.stream)
            })
            .collect())
    }

    fn validate_settlement(
        &self,
        project: ProjectId,
        queue: &str,
        owner: Uuid,
        consumer: u64,
        delivery_tag: u64,
        multiple: bool,
    ) -> Result<()> {
        let queue = self
            .queues
            .get(&(project, queue.to_owned()))
            .ok_or_else(|| Error::invalid_data("queue does not exist"))?;
        if queue.kind != QueueKind::Classic {
            return Err(Error::invalid_data(
                "stream queues use offsets, not destructive acknowledgements",
            ));
        }
        let stream = self
            .streams
            .get(&queue.stream)
            .ok_or_else(|| Error::internal("queue stream is missing"))?;
        if !stream.records.iter().any(|reference| {
            matches!(
                reference.delivery,
                DeliveryState::Unacked {
                    owner: candidate,
                    consumer: candidate_consumer,
                    tag,
                }
                    if candidate == owner && candidate_consumer == consumer
                        && if multiple { tag <= delivery_tag } else { tag == delivery_tag }
            )
        }) {
            return Err(Error::new(
                crate::ErrorCode::ProtocolViolation,
                "delivery tag is unknown",
            ));
        }
        Ok(())
    }

    fn validate_dead_letter_appends(
        &self,
        project: ProjectId,
        queue: &str,
        owner: Uuid,
        consumer: u64,
        delivery_tag: u64,
        multiple: bool,
    ) -> Result<()> {
        let queue_record = self
            .queues
            .get(&(project, queue.to_owned()))
            .ok_or_else(|| Error::invalid_data("queue does not exist"))?;
        let Some(exchange) = queue_record.dead_letter_exchange.as_deref() else {
            return Ok(());
        };
        let routing = queue_record
            .dead_letter_routing_key
            .as_deref()
            .unwrap_or_default();
        let targets = self.amqp_targets(project, exchange, routing)?;
        let source = self
            .streams
            .get(&queue_record.stream)
            .ok_or_else(|| Error::internal("queue stream is missing"))?;
        let appends = source
            .records
            .iter()
            .filter(|reference| {
                matches!(
                    reference.delivery,
                    DeliveryState::Unacked {
                        owner: candidate,
                        consumer: candidate_consumer,
                        tag,
                    }
                        if candidate == owner && candidate_consumer == consumer
                            && if multiple { tag <= delivery_tag } else { tag == delivery_tag }
                )
            })
            .count() as u64;
        for target in targets {
            self.streams
                .get(&target)
                .ok_or_else(|| Error::internal("dead-letter stream is missing"))?
                .next_offset
                .checked_add(appends)
                .ok_or_else(|| Error::internal("dead-letter offset exhausted"))?;
        }
        Ok(())
    }

    pub fn apply(
        &mut self,
        command: BrokerCommand,
        segments: &SegmentStore,
    ) -> Result<BrokerReply> {
        match command {
            BrokerCommand::CreateTopic {
                project,
                name,
                partitions,
                retention,
            } => {
                valid_name(&name)?;
                retention.validate()?;
                if partitions == 0 || partitions > 4_096 {
                    return Err(Error::invalid_data(
                        "topic partition count must be 1..=4096",
                    ));
                }
                let key = (project, name);
                if let Some(existing) = self.topics.get(&key) {
                    if existing.partitions.len() != partitions as usize {
                        return Err(Error::invalid_data("topic partition count is immutable"));
                    }
                    return Ok(BrokerReply::TopicAlreadyExists);
                }
                let mut streams = Vec::with_capacity(partitions as usize);
                for _ in 0..partitions {
                    let stream = self.allocate_stream()?;
                    streams.push(stream);
                }
                self.topics.insert(
                    key,
                    Topic {
                        partitions: streams,
                        retention,
                    },
                );
                Ok(BrokerReply::Declared)
            }
            BrokerCommand::SetTopicRetention {
                project,
                name,
                retention,
            } => {
                retention.validate()?;
                let topic = self
                    .topics
                    .get_mut(&(project, name))
                    .ok_or_else(|| Error::invalid_data("topic does not exist"))?;
                topic.retention = retention;
                Ok(BrokerReply::Declared)
            }
            BrokerCommand::DeleteTopic { project, name } => {
                let topic = self
                    .topics
                    .remove(&(project, name.clone()))
                    .ok_or_else(|| {
                        Error::new(crate::ErrorCode::InvalidData, "topic does not exist")
                    })?;
                let removed_payloads = topic
                    .partitions
                    .iter()
                    .filter_map(|stream| self.streams.get(stream))
                    .flat_map(|stream| stream.records.iter().map(|record| record.message))
                    .collect::<BTreeSet<_>>();
                for stream in topic.partitions {
                    self.streams.remove(&stream);
                }
                for ((group_project, _), group) in &mut *self.groups {
                    let group = Arc::make_mut(group);
                    if *group_project == project {
                        group.offsets.retain(|(topic, _), _| topic != &name);
                        group.assignments.clear();
                    }
                }
                self.reclaim_payload_candidates(project, &removed_payloads)?;
                Ok(BrokerReply::Deleted)
            }
            BrokerCommand::ClearTopic { project, name } => {
                let partitions = self
                    .topics
                    .get(&(project, name))
                    .ok_or_else(|| Error::invalid_data("topic does not exist"))?
                    .partitions
                    .clone();
                let mut removed = BTreeSet::new();
                let mut count = 0_u64;
                for stream_id in partitions {
                    let stream = self
                        .streams
                        .get_mut(&stream_id)
                        .ok_or_else(|| Error::internal("topic partition stream is missing"))?;
                    let stream = Arc::make_mut(stream);
                    count = count.saturating_add(stream.records.len() as u64);
                    removed.extend(stream.records.iter().map(|record| record.message));
                    stream.records = StreamRecords::default();
                    stream.base_offset = stream.next_offset;
                    stream.retained_bytes = 0;
                }
                self.reclaim_payload_candidates(project, &removed)?;
                Ok(BrokerReply::MessagesDiscarded {
                    message_count: u32::try_from(count).unwrap_or(u32::MAX),
                })
            }
            BrokerCommand::CreateExchange {
                project,
                name,
                kind,
                durable,
                passive,
            } => {
                valid_name(&name)?;
                let key = (project, name);
                if let Some(existing) = self.exchanges.get(&key) {
                    if existing.kind != kind || existing.durable != durable {
                        return Err(Error::invalid_data(
                            "exchange redeclaration is incompatible",
                        ));
                    }
                    return Ok(BrokerReply::Declared);
                }
                if passive {
                    return Err(Error::invalid_data("passive exchange does not exist"));
                }
                self.exchanges.insert(key, Exchange { kind, durable });
                Ok(BrokerReply::Declared)
            }
            BrokerCommand::CreateQueue {
                project,
                name,
                kind,
                durable,
                passive,
                dead_letter_exchange,
                dead_letter_routing_key,
                retention,
                exclusive_owner,
                auto_delete,
            } => {
                valid_name(&name)?;
                retention.validate()?;
                let key = (project, name);
                if let Some(existing) = self.queues.get(&key) {
                    if existing.kind != kind || existing.durable != durable {
                        return Err(Error::invalid_data("queue redeclaration is incompatible"));
                    }
                    if existing.exclusive_owner.is_some()
                        && existing.exclusive_owner != exclusive_owner
                    {
                        return Err(Error::new(
                            crate::ErrorCode::AuthorizationDenied,
                            "queue is exclusive to another connection",
                        ));
                    }
                    return Ok(BrokerReply::Declared);
                }
                if passive {
                    return Err(Error::invalid_data("passive queue does not exist"));
                }
                if let Some(exchange) = &dead_letter_exchange {
                    if !exchange.is_empty()
                        && !self.exchanges.contains_key(&(project, exchange.clone()))
                    {
                        return Err(Error::invalid_data("dead-letter exchange does not exist"));
                    }
                }
                let stream = self.allocate_stream()?;
                self.queues.insert(
                    key,
                    Queue {
                        stream,
                        kind,
                        durable,
                        dead_letter_exchange,
                        dead_letter_routing_key,
                        retention,
                        exclusive_owner,
                        auto_delete,
                    },
                );
                Ok(BrokerReply::Declared)
            }
            BrokerCommand::SetQueueRetention {
                project,
                name,
                retention,
            } => {
                retention.validate()?;
                let queue = self
                    .queues
                    .get_mut(&(project, name))
                    .ok_or_else(|| Error::invalid_data("queue does not exist"))?;
                queue.retention = retention;
                Ok(BrokerReply::Declared)
            }
            BrokerCommand::RegisterConsumer {
                project,
                queue,
                owner,
                consumer,
            } => {
                let declared = self
                    .queues
                    .get(&(project, queue.clone()))
                    .ok_or_else(|| Error::invalid_data("queue does not exist"))?;
                if let Some(exclusive) = declared.exclusive_owner
                    && exclusive != owner
                {
                    return Err(Error::new(
                        crate::ErrorCode::AuthorizationDenied,
                        "queue is exclusive to another connection",
                    ));
                }
                self.queue_consumers
                    .entry((project, queue))
                    .or_default()
                    .insert((owner, consumer));
                Ok(BrokerReply::Declared)
            }
            BrokerCommand::UnregisterConsumer {
                project,
                queue,
                owner,
                consumer,
            } => {
                let key = (project, queue);
                let emptied = match self.queue_consumers.get_mut(&key) {
                    Some(consumers) => {
                        consumers.remove(&(owner, consumer));
                        consumers.is_empty()
                    }
                    None => false,
                };
                if emptied {
                    self.queue_consumers.remove(&key);
                }
                // An auto-delete queue only goes away once its LAST consumer leaves, which is the
                // difference between a shared work queue and a per-client reply queue.
                if emptied && self.queue_is_auto_delete(key.0, &key.1) {
                    self.drop_queue(key.0, &key.1)?;
                }
                Ok(BrokerReply::Declared)
            }
            BrokerCommand::ReleaseConnection { project, owner } => {
                let touched = self
                    .queue_consumers
                    .iter()
                    .filter(|((candidate, _), consumers)| {
                        *candidate == project
                            && consumers.iter().any(|(holder, _)| *holder == owner)
                    })
                    .map(|(key, _)| key.1.clone())
                    .collect::<Vec<_>>();
                let mut collect = BTreeSet::new();
                for queue in touched {
                    let key = (project, queue.clone());
                    let Some(consumers) = self.queue_consumers.get_mut(&key) else {
                        continue;
                    };
                    consumers.retain(|(holder, _)| *holder != owner);
                    if consumers.is_empty() {
                        self.queue_consumers.remove(&key);
                        if self.queue_is_auto_delete(project, &queue) {
                            collect.insert(queue);
                        }
                    }
                }
                // Exclusive queues die with their connection whether or not anything consumed them,
                // which is what makes `queue.declare(queue='', exclusive=True)` self-cleaning.
                collect.extend(
                    self.queues
                        .iter()
                        .filter(|((candidate, _), queue)| {
                            *candidate == project && queue.exclusive_owner == Some(owner)
                        })
                        .map(|(key, _)| key.1.clone()),
                );
                for queue in collect {
                    self.drop_queue(project, &queue)?;
                }
                Ok(BrokerReply::Deleted)
            }
            BrokerCommand::BindQueue {
                project,
                exchange,
                queue,
                routing_key,
            } => {
                valid_routing_key(&routing_key)?;
                if !self.exchanges.contains_key(&(project, exchange.clone()))
                    || !self.queues.contains_key(&(project, queue.clone()))
                {
                    return Err(Error::invalid_data("binding endpoint does not exist"));
                }
                let key = (project, exchange.clone());
                let binding = Binding {
                    exchange,
                    queue,
                    routing_key,
                };
                if !self.bindings.iter().any(|item| {
                    item.0 == key
                        && item.1.queue == binding.queue
                        && item.1.routing_key == binding.routing_key
                }) {
                    self.bindings.push((key, binding));
                }
                Ok(BrokerReply::Declared)
            }
            BrokerCommand::UnbindQueue {
                project,
                exchange,
                queue,
                routing_key,
            } => {
                valid_routing_key(&routing_key)?;
                if !self.binding_exists(project, &exchange, &queue, &routing_key) {
                    return Err(Error::invalid_data("binding does not exist"));
                }
                self.bindings.retain(|(key, binding)| {
                    !(key.0 == project
                        && binding.exchange == exchange
                        && binding.queue == queue
                        && binding.routing_key == routing_key)
                });
                Ok(BrokerReply::Unbound)
            }
            BrokerCommand::PurgeQueue { project, name } => {
                let stream = self
                    .queues
                    .get(&(project, name.clone()))
                    .ok_or_else(|| Error::invalid_data("queue does not exist"))?
                    .stream;
                let discarded = self.discard_ready_records(project, stream)?;
                Ok(BrokerReply::MessagesDiscarded {
                    message_count: discarded,
                })
            }
            BrokerCommand::DeleteQueue {
                project,
                name,
                if_unused,
                if_empty,
            } => {
                let ready = self.queue_deletion_counts(project, &name, if_unused, if_empty)?;
                self.drop_queue(project, &name)?;
                Ok(BrokerReply::MessagesDiscarded {
                    message_count: ready,
                })
            }
            BrokerCommand::DeleteExchange {
                project,
                name,
                if_unused,
            } => {
                if !self.exchanges.contains_key(&(project, name.clone())) {
                    return Err(Error::invalid_data("exchange does not exist"));
                }
                if if_unused && self.exchange_has_bindings(project, &name) {
                    return Err(Error::invalid_data("exchange still has bindings"));
                }
                self.exchanges.remove(&(project, name.clone()));
                self.bindings
                    .retain(|(key, binding)| !(key.0 == project && binding.exchange == name));
                Ok(BrokerReply::Deleted)
            }
            BrokerCommand::PublishKafkaBatch {
                project,
                topic,
                partition,
                resolved_time_ms,
                records,
            } => self.publish_kafka_batch(
                project,
                topic,
                partition,
                resolved_time_ms,
                records,
                segments,
            ),
            BrokerCommand::PublishAmqp {
                project,
                exchange,
                routing_key,
                mandatory,
                resolved_time_ms,
                properties,
                headers,
                payload,
            } => self.publish_amqp(
                project,
                exchange,
                routing_key,
                mandatory,
                resolved_time_ms,
                properties,
                headers,
                payload,
                segments,
            ),
            BrokerCommand::PublishAmqpBatch {
                project,
                resolved_time_ms,
                records,
            } => self.publish_amqp_batch(project, resolved_time_ms, records, segments),
            BrokerCommand::PublishAmqpUniformBatch {
                project,
                resolved_time_ms,
                exchange,
                routing_key,
                mandatory,
                properties,
                headers,
                payloads,
            } => self.publish_amqp_uniform_batch(
                project,
                resolved_time_ms,
                exchange,
                routing_key,
                mandatory,
                properties,
                headers,
                payloads,
                segments,
            ),
            BrokerCommand::CommitOffset {
                project,
                group,
                topic,
                partition,
                offset,
                generation,
                member,
            } => {
                self.partition_stream(project, &topic, partition)?;
                let state = self
                    .groups
                    .get_mut(&(project, group))
                    .ok_or_else(|| Error::invalid_data("consumer group does not exist"))?;
                let state = Arc::make_mut(state);
                match (generation, member) {
                    (Some(generation), Some(member))
                        if generation == state.generation
                            && state.members.contains_key(&member) => {}
                    (None, None) => {}
                    _ => {
                        return Err(Error::new(
                            crate::ErrorCode::TransactionConflict,
                            "consumer group generation or member changed",
                        ));
                    }
                }
                state.offsets.insert((topic, partition), offset);
                Ok(BrokerReply::OffsetCommitted)
            }
            BrokerCommand::JoinGroup {
                project,
                group,
                member,
                session_timeout_ms,
                resolved_time_ms,
                protocols,
            } => {
                valid_name(&group)?;
                valid_name(&member)?;
                validate_group_member(session_timeout_ms, &protocols)?;
                self.expire_group_members(project, resolved_time_ms)?;
                let state = self.groups.entry((project, group)).or_insert_with(|| {
                    Arc::new(GroupState {
                        generation: 0,
                        leader: member.clone(),
                        members: BTreeMap::new(),
                        protocol: String::new(),
                        assignments: BTreeMap::new(),
                        offsets: BTreeMap::new(),
                    })
                });
                let state = Arc::make_mut(state);
                let replacement = GroupMember {
                    protocols,
                    session_timeout_ms,
                    last_heartbeat_ms: resolved_time_ms,
                };
                let changed = state.members.get(&member).is_none_or(|existing| {
                    existing.protocols != replacement.protocols
                        || existing.session_timeout_ms != replacement.session_timeout_ms
                });
                state.members.insert(member, replacement);
                let selected_protocol = select_group_protocol(&state.members)?;
                if changed || state.protocol != selected_protocol {
                    state.protocol = selected_protocol;
                    state.generation = state
                        .generation
                        .checked_add(1)
                        .ok_or_else(|| Error::internal("group generation exhausted"))?;
                    state.assignments.clear();
                }
                Ok(group_reply(state))
            }
            BrokerCommand::SyncGroup {
                project,
                group,
                generation,
                assignments,
            } => {
                let group_name = group;
                let state = self
                    .groups
                    .get_mut(&(project, group_name.clone()))
                    .ok_or_else(|| Error::invalid_data("consumer group does not exist"))?;
                let state = Arc::make_mut(state);
                if state.generation != generation {
                    return Err(Error::new(
                        crate::ErrorCode::TransactionConflict,
                        "consumer group generation changed",
                    ));
                }
                if !assignments.is_empty() {
                    if assignments.len() != state.members.len()
                        || assignments
                            .keys()
                            .any(|assigned| !state.members.contains_key(assigned))
                    {
                        return Err(Error::invalid_data(
                            "group sync must install one assignment per member",
                        ));
                    }
                    state.assignments = assignments;
                }
                // The leader proposed an assignment from partition counts alone. Replace it with
                // one weighted by real lag, which is the whole point of balancing here rather than
                // in the client: only the broker can see the partitions a consumer does not own.
                // Every publication applies this deterministically over canonical state, so
                // they all install the same bytes.
                let group = Arc::clone(
                    self.groups
                        .get(&(project, group_name.clone()))
                        .ok_or_else(|| Error::internal("group disappeared during sync"))?,
                );
                if let Some(balanced) = self.rebalanced_assignment(project, &group) {
                    let state = self
                        .groups
                        .get_mut(&(project, group_name))
                        .ok_or_else(|| Error::internal("group disappeared during sync"))?;
                    let state = Arc::make_mut(state);
                    state.assignments = balanced;
                    return Ok(group_reply(state));
                }
                let state = self
                    .groups
                    .get(&(project, group_name))
                    .ok_or_else(|| Error::internal("group disappeared during sync"))?;
                Ok(group_reply(state))
            }
            BrokerCommand::LeaveGroup {
                project,
                group,
                member,
            } => {
                let state = self
                    .groups
                    .get_mut(&(project, group))
                    .ok_or_else(|| Error::invalid_data("consumer group does not exist"))?;
                let state = Arc::make_mut(state);
                if state.members.remove(&member).is_some() {
                    state.assignments.clear();
                    state.generation = state
                        .generation
                        .checked_add(1)
                        .ok_or_else(|| Error::internal("group generation exhausted"))?;
                    if state.leader == member {
                        state.leader = state.members.keys().next().cloned().unwrap_or_default();
                    }
                }
                Ok(group_reply(state))
            }
            BrokerCommand::HeartbeatGroup {
                project,
                group,
                generation,
                member,
                resolved_time_ms,
            } => {
                self.expire_group_members(project, resolved_time_ms)?;
                let state = self
                    .groups
                    .get_mut(&(project, group))
                    .ok_or_else(|| Error::invalid_data("consumer group does not exist"))?;
                let state = Arc::make_mut(state);
                if state.generation != generation {
                    return Err(Error::new(
                        crate::ErrorCode::TransactionConflict,
                        "consumer group generation changed",
                    ));
                }
                let member = state.members.get_mut(&member).ok_or_else(|| {
                    Error::new(
                        crate::ErrorCode::TransactionConflict,
                        "consumer group member is unknown",
                    )
                })?;
                member.last_heartbeat_ms = resolved_time_ms;
                Ok(BrokerReply::Acknowledged)
            }
            BrokerCommand::Ack {
                project,
                queue,
                owner,
                consumer,
                delivery_tag,
                multiple,
            } => {
                self.settle(
                    project,
                    &queue,
                    owner,
                    consumer,
                    delivery_tag,
                    multiple,
                    true,
                    false,
                )?;
                Ok(BrokerReply::Acknowledged)
            }
            BrokerCommand::Nack {
                project,
                queue,
                owner,
                consumer,
                delivery_tag,
                multiple,
                requeue,
            } => {
                self.settle(
                    project,
                    &queue,
                    owner,
                    consumer,
                    delivery_tag,
                    multiple,
                    false,
                    requeue,
                )?;
                Ok(BrokerReply::Acknowledged)
            }
            BrokerCommand::DeliverQueue {
                project,
                queue,
                offset,
                maximum,
                owner,
                consumer,
                automatic_ack,
                resolved_time_ms,
            } => Ok(BrokerReply::Deliveries(self.fetch_queue(
                project,
                &queue,
                offset,
                maximum as usize,
                owner,
                consumer,
                automatic_ack,
                resolved_time_ms,
                segments,
            )?)),
            BrokerCommand::RenewDeliveryLease {
                project,
                owner,
                resolved_time_ms,
            } => {
                self.renew_delivery_lease(project, owner, resolved_time_ms)?;
                Ok(BrokerReply::Acknowledged)
            }
            BrokerCommand::ReleaseDeliveryLease { project, owner } => {
                self.release_delivery_lease(project, owner);
                Ok(BrokerReply::Acknowledged)
            }
            BrokerCommand::Retain {
                project,
                resolved_time_ms,
            } => {
                self.expire_group_members(project, resolved_time_ms)?;
                self.expire_delivery_leases(project, resolved_time_ms);
                let (removed_references, candidates) = self.retain(project, resolved_time_ms)?;
                let before = self.payloads.len();
                self.reclaim_payload_candidates(project, &candidates)?;
                Ok(BrokerReply::Retained {
                    removed_references,
                    removed_payloads: (before - self.payloads.len()) as u64,
                })
            }
        }
    }

    pub fn fetch_partition(
        &self,
        project: ProjectId,
        topic: &str,
        partition: i32,
        offset: u64,
        maximum_bytes: usize,
        segments: &SegmentStore,
    ) -> Result<Vec<(u64, Arc<PayloadRecord>)>> {
        let stream = self.partition_stream(project, topic, partition)?;
        self.fetch_stream(stream, offset, maximum_bytes, false, segments)
    }

    pub(crate) fn fetch_partition_bounded(
        &self,
        read: &PartitionRead<'_>,
        segments: &SegmentStore,
    ) -> Result<(u64, Vec<(u64, Arc<PayloadRecord>)>)> {
        let stream = self.partition_stream(read.project, read.topic, read.partition)?;
        let high_watermark = self
            .streams
            .get(&stream)
            .ok_or_else(|| Error::internal("partition stream is absent"))?
            .next_offset;
        let pending = self.fetch_stream_records_with_bounds(
            stream,
            read.offset,
            read.maximum_bytes,
            false,
            Some(read.maximum_records),
        )?;
        let loaded = load_payloads(
            &self.payload_segments,
            pending.iter().map(|(_, stored)| stored.clone()),
            segments,
        )?;
        let records = pending
            .into_iter()
            .map(|(offset, stored)| {
                Ok((
                    offset,
                    loaded.get(&stored.id).cloned().ok_or_else(|| {
                        Error::internal("broker payload batch omitted a requested ID")
                    })?,
                ))
            })
            .collect::<Result<_>>()?;
        Ok((high_watermark, records))
    }

    pub(crate) fn partition_fetch_segment_descriptors(
        &self,
        project: ProjectId,
        topic: &str,
        partition: i32,
        offset: u64,
        maximum_bytes: usize,
    ) -> Result<Vec<SegmentDescriptor>> {
        let stream = self.partition_stream(project, topic, partition)?;
        let pending = self.fetch_stream_records(stream, offset, maximum_bytes, false)?;
        unique_payload_descriptors(
            &self.payload_segments,
            pending.into_iter().map(|(_, stored)| stored),
        )
    }

    pub(crate) fn stream_queue_fetch_segment_descriptors(
        &self,
        project: ProjectId,
        queue: &str,
        offset: StreamOffset,
        maximum: usize,
    ) -> Result<Vec<SegmentDescriptor>> {
        let pending = self.stream_queue_records(project, queue, offset, maximum)?;
        unique_payload_descriptors(
            &self.payload_segments,
            pending.into_iter().map(|record| record.stored),
        )
    }

    fn stream_queue_records(
        &self,
        project: ProjectId,
        queue: &str,
        offset: StreamOffset,
        maximum: usize,
    ) -> Result<Vec<PendingDelivery>> {
        let queue_record = self
            .queues
            .get(&(project, queue.to_owned()))
            .ok_or_else(|| Error::invalid_data("queue does not exist"))?;
        if queue_record.kind != QueueKind::Stream {
            return Err(Error::invalid_data(
                "read-only stream fetch is valid only for stream queues",
            ));
        }
        let stream = self
            .streams
            .get(&queue_record.stream)
            .ok_or_else(|| Error::internal("queue stream is missing"))?;
        let start = self.resolve_offset(stream, offset)?;
        let mut visible_bytes = 0_usize;
        let mut pending = Vec::new();
        for reference in stream
            .records
            .iter()
            .filter(|reference| reference.offset >= start)
            .take(maximum)
        {
            let stored = self
                .payloads
                .get(&reference.message)
                .cloned()
                .ok_or_else(|| Error::internal("broker payload is missing"))?;
            let next_visible_bytes = visible_bytes
                .checked_add(
                    usize::try_from(self.accounted_payload_bytes(&stored)?).map_err(|_| {
                        Error::internal("broker delivery length exceeds this platform")
                    })?,
                )
                .and_then(|bytes| bytes.checked_add(BROKER_DELIVERY_ENVELOPE_BYTES))
                .ok_or_else(|| Error::internal("broker delivery result length overflow"))?;
            if next_visible_bytes > MAX_BROKER_DELIVERY_RESULT_BYTES {
                if pending.is_empty() {
                    return Err(Error::new(
                        crate::ErrorCode::ResultBudgetExceeded,
                        "delivery response exceeds the result bound",
                    ));
                }
                break;
            }
            visible_bytes = next_visible_bytes;
            pending.push(PendingDelivery {
                offset: reference.offset,
                stored,
                redelivered: reference.redelivered,
                death_count: reference.death_count,
                exchange: reference.current_exchange.clone(),
                routing_key: reference.current_routing_key.clone(),
            });
        }
        Ok(pending)
    }

    pub fn fetch_queue(
        &mut self,
        project: ProjectId,
        queue: &str,
        offset: StreamOffset,
        maximum: usize,
        owner: Uuid,
        consumer: u64,
        automatic_ack: bool,
        resolved_time_ms: i64,
        segments: &SegmentStore,
    ) -> Result<Vec<Delivery>> {
        let queue_record = self
            .queues
            .get(&(project, queue.to_owned()))
            .cloned()
            .ok_or_else(|| Error::invalid_data("queue does not exist"))?;
        let stream = self
            .streams
            .get(&queue_record.stream)
            .ok_or_else(|| Error::internal("queue stream is missing"))?;
        let start = self.resolve_offset(stream, offset)?;
        let mut pending = Vec::new();
        let mut selected = BTreeMap::new();
        let mut visible_bytes = 0_usize;
        for reference in stream
            .records
            .iter()
            .filter(|record| record.offset >= start)
        {
            if pending.len() >= maximum {
                break;
            }
            if queue_record.kind == QueueKind::Classic && reference.delivery != DeliveryState::Ready
            {
                continue;
            }
            let stored = self
                .payloads
                .get(&reference.message)
                .cloned()
                .ok_or_else(|| Error::internal("broker payload is missing"))?;
            let next_visible_bytes = visible_bytes
                .checked_add(
                    usize::try_from(self.accounted_payload_bytes(&stored)?).map_err(|_| {
                        Error::internal("broker delivery length exceeds this platform")
                    })?,
                )
                .and_then(|bytes| bytes.checked_add(BROKER_DELIVERY_ENVELOPE_BYTES))
                .ok_or_else(|| Error::internal("broker delivery result length overflow"))?;
            if next_visible_bytes > MAX_BROKER_DELIVERY_RESULT_BYTES {
                if pending.is_empty() {
                    return Err(Error::new(
                        crate::ErrorCode::ResultBudgetExceeded,
                        "delivery response exceeds the result bound",
                    ));
                }
                break;
            }
            visible_bytes = next_visible_bytes;
            let tag = reference
                .offset
                .checked_add(1)
                .ok_or_else(|| Error::internal("delivery tag exhausted"))?;
            selected.insert(reference.offset, tag);
            pending.push((
                reference.offset,
                stored,
                reference.redelivered,
                reference.death_count,
                reference.current_exchange.clone(),
                reference.current_routing_key.clone(),
            ));
        }
        let loaded = load_payloads(
            &self.payload_segments,
            pending.iter().map(|(_, stored, ..)| stored.clone()),
            segments,
        )?;
        let mut deliveries = Vec::with_capacity(pending.len());
        for (offset, stored, redelivered, death_count, exchange, routing_key) in pending {
            let payload = loaded
                .get(&stored.id)
                .cloned()
                .ok_or_else(|| Error::internal("broker payload batch omitted a requested ID"))?;
            let (exchange, routing_key) = delivery_route(&payload, exchange, routing_key);
            deliveries.push(Delivery {
                queue: queue.to_owned(),
                offset,
                delivery_tag: offset
                    .checked_add(1)
                    .ok_or_else(|| Error::internal("delivery tag exhausted"))?,
                redelivered,
                death_count,
                exchange,
                routing_key,
                payload,
            });
        }
        // Payload reads and every fallible result construction completed before changing classic
        // delivery state, so a missing/corrupt segment cannot partially deliver a batch.
        if queue_record.kind == QueueKind::Classic {
            let lease_deadline = if automatic_ack {
                None
            } else {
                Some(
                    resolved_time_ms
                        .checked_add(DELIVERY_LEASE_MILLIS)
                        .ok_or_else(|| Error::internal("delivery lease deadline overflow"))?,
                )
            };
            let stream = self
                .streams
                .get_mut(&queue_record.stream)
                .ok_or_else(|| Error::internal("queue stream disappeared"))?;
            let stream = Arc::make_mut(stream);
            let base_offset = stream.base_offset;
            for (offset, tag) in &selected {
                let reference = stream
                    .records
                    .get_mut_by_offset(base_offset, *offset)
                    .ok_or_else(|| Error::internal("selected delivery offset disappeared"))?;
                reference.delivery = if automatic_ack {
                    DeliveryState::Acknowledged
                } else {
                    DeliveryState::Unacked {
                        owner,
                        consumer,
                        tag: *tag,
                    }
                };
            }
            if let Some(lease_deadline) = lease_deadline.filter(|_| !selected.is_empty()) {
                self.delivery_leases
                    .entry((project, owner))
                    .and_modify(|deadline| *deadline = (*deadline).max(lease_deadline))
                    .or_insert(lease_deadline);
            }
        }
        Ok(deliveries)
    }

    #[must_use]
    pub fn topic_metadata(&self, project: ProjectId) -> Vec<(String, usize)> {
        self.topics
            .iter()
            .filter(|((id, _), _)| *id == project)
            .map(|((_, name), topic)| (name.clone(), topic.partitions.len()))
            .collect()
    }

    #[must_use]
    pub fn topic_metrics(&self, project: ProjectId) -> Vec<TopicMetrics> {
        let mut metrics = self
            .topics
            .iter()
            .filter(|((candidate, _), _)| *candidate == project)
            .flat_map(|((_, name), topic)| {
                topic
                    .partitions
                    .iter()
                    .enumerate()
                    .filter_map(|(partition, stream_id)| {
                        let stream = self.streams.get(stream_id)?;
                        Some(TopicMetrics {
                            name: name.clone(),
                            partition: u16::try_from(partition).ok()?,
                            base_offset: stream.base_offset,
                            next_offset: stream.next_offset,
                            record_count: stream.records.len() as u64,
                            retained_bytes: stream.retained_bytes,
                            retention_ms: topic.retention.max_age_ms,
                        })
                    })
            })
            .collect::<Vec<_>>();
        metrics.sort_by(|left, right| {
            left.name
                .cmp(&right.name)
                .then(left.partition.cmp(&right.partition))
        });
        metrics
    }

    #[must_use]
    pub fn queue_metrics(&self, project: ProjectId) -> Vec<QueueMetrics> {
        let mut metrics = self
            .queues
            .iter()
            .filter(|((candidate, _), _)| *candidate == project)
            .filter_map(|((_, name), queue)| {
                let info = self.queue_info(project, name)?;
                let stream = self.streams.get(&queue.stream)?;
                Some(QueueMetrics {
                    name: name.clone(),
                    kind: queue.kind,
                    message_count: info.message_count,
                    available_count: info.available_count,
                    retained_bytes: stream.retained_bytes,
                    retention_ms: queue.retention.max_age_ms,
                })
            })
            .collect::<Vec<_>>();
        metrics.sort_by(|left, right| left.name.cmp(&right.name));
        metrics
    }

    #[must_use]
    pub fn exchange_metrics(&self, project: ProjectId) -> Vec<ExchangeMetrics> {
        let mut metrics = self
            .exchanges
            .iter()
            .filter(|((candidate, _), _)| *candidate == project)
            .map(|((_, name), exchange)| ExchangeMetrics {
                name: name.clone(),
                kind: exchange.kind,
                durable: exchange.durable,
                binding_count: self
                    .bindings
                    .iter()
                    .filter(|((candidate, _), binding)| {
                        *candidate == project && binding.exchange == *name
                    })
                    .count() as u64,
            })
            .collect::<Vec<_>>();
        metrics.sort_by(|left, right| left.name.cmp(&right.name));
        metrics
    }

    #[must_use]
    pub fn consumer_lag_metrics(&self, project: ProjectId) -> Vec<ConsumerLagMetrics> {
        let mut metrics = Vec::new();
        for ((candidate, group), state) in &*self.groups {
            if *candidate != project {
                continue;
            }
            for ((topic, partition), committed_offset) in &state.offsets {
                let Ok(stream_id) = self.partition_stream(project, topic, *partition) else {
                    continue;
                };
                let Some(stream) = self.streams.get(&stream_id) else {
                    continue;
                };
                metrics.push(ConsumerLagMetrics {
                    group: group.clone(),
                    topic: topic.clone(),
                    partition: *partition,
                    committed_offset: *committed_offset,
                    next_offset: stream.next_offset,
                    lag: stream.next_offset.saturating_sub(*committed_offset),
                });
            }
        }
        metrics.sort_by(|left, right| {
            left.group
                .cmp(&right.group)
                .then(left.topic.cmp(&right.topic))
                .then(left.partition.cmp(&right.partition))
        });
        metrics
    }

    pub fn committed_offset(
        &self,
        project: ProjectId,
        group: &str,
        topic: &str,
        partition: i32,
    ) -> Option<u64> {
        self.groups
            .get(&(project, group.to_owned()))
            .and_then(|state| state.offsets.get(&(topic.to_owned(), partition)).copied())
    }

    pub fn list_offset(
        &self,
        project: ProjectId,
        topic: &str,
        partition: i32,
        timestamp: i64,
    ) -> Result<Option<(u64, i64)>> {
        let stream_id = self.partition_stream(project, topic, partition)?;
        let stream = self
            .streams
            .get(&stream_id)
            .ok_or_else(|| Error::internal("topic partition stream is missing"))?;
        if timestamp == -2 {
            let record_time = stream
                .records
                .front()
                .and_then(|reference| self.payloads.get(&reference.message))
                .and_then(|payload| payload.kafka_time_ms)
                .unwrap_or(-1);
            return Ok(Some((stream.base_offset, record_time)));
        }
        if timestamp == -1 {
            return Ok(Some((stream.next_offset, -1)));
        }
        if timestamp < 0 {
            return Ok(None);
        }
        Ok(stream.records.iter().find_map(|reference| {
            self.payloads.get(&reference.message).and_then(|payload| {
                payload
                    .kafka_time_ms
                    .filter(|record_time| *record_time >= timestamp)
                    .map(|record_time| (reference.offset, record_time))
            })
        }))
    }

    /// Immutable payload segments referenced by canonical message IDs, in descriptor order.
    pub(crate) fn payload_segments(&self) -> Vec<SegmentDescriptor> {
        let mut descriptors = self
            .payload_segments
            .iter()
            .map(|(_, segment)| segment.descriptor.clone())
            .collect::<Vec<_>>();
        descriptors.sort();
        descriptors.dedup();
        descriptors
    }

    #[must_use]
    pub(crate) const fn message_cursor(&self) -> u64 {
        self.next_message_id
    }

    #[must_use]
    pub(crate) fn newest_payload_segment_after(
        &self,
        previous_cursor: u64,
    ) -> Option<SegmentDescriptor> {
        (self.next_message_id > previous_cursor)
            .then(|| self.payloads.get(&MessageId(self.next_message_id)))
            .flatten()
            .and_then(|payload| self.payload_segments.get(payload.segment))
            .map(|segment| segment.descriptor.clone())
    }

    pub(crate) fn validate_payload_segments(&self, segments: &SegmentStore) -> Result<()> {
        let mut grouped = BTreeMap::<u64, Vec<StoredPayloadRecord>>::new();
        for (id, payload) in self.payloads.iter() {
            let segment = self.payload_segments.get(payload.segment).ok_or_else(|| {
                Error::new(
                    crate::ErrorCode::CorruptStorage,
                    "broker payload segment table is incomplete",
                )
            })?;
            if id != payload.id
                || payload.payload_bytes > 256 * 1024 * 1024
                || payload.retained_bytes < payload.payload_bytes
                || payload.retained_bytes != payload.location.bytes
                || payload
                    .location
                    .offset
                    .checked_add(payload.location.bytes)
                    .is_none_or(|end| end > segment.descriptor.bytes)
            {
                return Err(Error::new(
                    crate::ErrorCode::CorruptStorage,
                    "broker payload reference is invalid",
                ));
            }
            grouped
                .entry(payload.segment)
                .or_default()
                .push(payload.clone());
        }
        for (segment_id, mut expected) in grouped {
            let segment = self.payload_segments.get(segment_id).ok_or_else(|| {
                Error::new(
                    crate::ErrorCode::CorruptStorage,
                    "broker payload segment table is incomplete",
                )
            })?;
            if usize::try_from(segment.live_records).ok() != Some(expected.len()) {
                return Err(Error::new(
                    crate::ErrorCode::CorruptStorage,
                    "broker payload segment live count differs from canonical messages",
                ));
            }
            segments.validate(&segment.descriptor)?;
            expected.sort_by_key(|payload| payload.location.offset);
            let mut previous_end = 0_u64;
            let mut previous_location = None;
            for stored in expected {
                let aliases_previous = previous_location == Some(stored.location);
                if stored.location.offset < previous_end && !aliases_previous {
                    return Err(Error::new(
                        crate::ErrorCode::CorruptStorage,
                        "broker payload record extents overlap",
                    ));
                }
                previous_end = stored
                    .location
                    .offset
                    .checked_add(stored.location.bytes)
                    .ok_or_else(|| {
                        Error::new(
                            crate::ErrorCode::CorruptStorage,
                            "broker payload record extent overflows",
                        )
                    })?;
                previous_location = Some(stored.location);
                let record = segments.read_record_at(&segment.descriptor, stored.location)?;
                let payload = materialize_stored_payload(&record, &stored, segment)?;
                validate_loaded_payload(&stored, &segment.descriptor, &payload)?;
            }
        }
        Ok(())
    }

    /// Materializes a publish payload into its deterministic immutable segment before the
    /// ordered append. This is derived storage, not a second WAL; an uncommitted orphan is
    /// harmless and later content-address reuse is exact. Pre-staging prevents an oversized
    /// segment or local disk failure from first appearing after a client mutation has committed.
    #[cfg(test)]
    pub(crate) fn prestage_command_payload(
        &self,
        command: &BrokerCommand,
        segments: &SegmentStore,
    ) -> Result<Vec<SegmentDescriptor>> {
        let mut staged = Vec::new();
        match command {
            BrokerCommand::PublishKafkaBatch {
                project,
                topic,
                partition,
                resolved_time_ms,
                records,
            } => {
                self.partition_stream(*project, topic, *partition)?;
                validate_kafka_batch(records)?;
                let mut ingress = Vec::with_capacity(records.len());
                for record in records {
                    ingress.push(IngressMetadata::Kafka {
                        create_time_ms: record.create_time_ms,
                        key: record.key.clone(),
                        headers: record.headers.clone(),
                        value_is_null: record.value_is_null,
                    });
                }
                let mut pending = Vec::with_capacity(records.len());
                for (index, (record, ingress)) in records.iter().zip(&ingress).enumerate() {
                    let increment = u64::try_from(index)
                        .map_err(|_| Error::internal("Kafka batch index exceeds u64"))?
                        .checked_add(1)
                        .ok_or_else(|| Error::internal("message ID exhausted"))?;
                    let id = MessageId(
                        self.next_message_id
                            .checked_add(increment)
                            .ok_or_else(|| Error::internal("message ID exhausted"))?,
                    );
                    pending.push(PendingPayloadRecord {
                        id,
                        ingress,
                        payload: &record.payload,
                    });
                }
                staged.push(if kafka_batch_can_share_ingress(records) {
                    write_raw_kafka_payload_segment_batch(*project, records, segments)?.0
                } else {
                    write_payload_segment_batch(*project, *resolved_time_ms, &pending, segments)?.0
                });
            }
            BrokerCommand::PublishAmqp {
                project,
                exchange,
                routing_key,
                resolved_time_ms,
                properties,
                headers,
                payload,
                ..
            } if !self
                .amqp_targets(*project, exchange, routing_key)?
                .is_empty() =>
            {
                let id = MessageId(
                    self.next_message_id
                        .checked_add(1)
                        .ok_or_else(|| Error::internal("message ID exhausted"))?,
                );
                let ingress = IngressMetadata::Amqp {
                    exchange: exchange.clone(),
                    routing_key: routing_key.clone(),
                    properties: properties.clone(),
                    headers: headers.clone(),
                    death_count: 0,
                };
                staged.push(
                    write_raw_amqp_payload_segment_batch(
                        *project,
                        &[PendingPayloadRecord {
                            id,
                            ingress: &ingress,
                            payload,
                        }],
                        segments,
                    )?
                    .0,
                );
            }
            BrokerCommand::PublishAmqpBatch {
                project,
                resolved_time_ms,
                records,
            } => {
                validate_amqp_batch(records)?;
                let mut routed = Vec::with_capacity(records.len());
                for record in records {
                    let targets =
                        self.amqp_targets(*project, &record.exchange, &record.routing_key)?;
                    if !targets.is_empty() {
                        routed.push((record, targets));
                    }
                }
                if !routed.is_empty() {
                    let ingress = routed
                        .iter()
                        .map(|(record, _)| IngressMetadata::Amqp {
                            exchange: record.exchange.clone(),
                            routing_key: record.routing_key.clone(),
                            properties: record.properties.clone(),
                            headers: record.headers.clone(),
                            death_count: 0,
                        })
                        .collect::<Vec<_>>();
                    let pending = routed
                        .iter()
                        .zip(&ingress)
                        .enumerate()
                        .map(|(index, ((record, _), ingress))| {
                            let increment = u64::try_from(index)
                                .map_err(|_| Error::internal("AMQP batch index exceeds u64"))?
                                .checked_add(1)
                                .ok_or_else(|| Error::internal("message ID exhausted"))?;
                            Ok(PendingPayloadRecord {
                                id: MessageId(
                                    self.next_message_id
                                        .checked_add(increment)
                                        .ok_or_else(|| Error::internal("message ID exhausted"))?,
                                ),
                                ingress,
                                payload: &record.payload,
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    let descriptor = if ingress
                        .first()
                        .is_some_and(|first| ingress.iter().skip(1).all(|value| value == first))
                    {
                        write_raw_amqp_payload_segment_batch(*project, &pending, segments)?.0
                    } else {
                        write_payload_segment_batch(
                            *project,
                            *resolved_time_ms,
                            &pending,
                            segments,
                        )?
                        .0
                    };
                    staged.push(descriptor);
                }
            }
            BrokerCommand::PublishAmqpUniformBatch {
                project,
                exchange,
                routing_key,
                properties,
                headers,
                payloads,
                ..
            } => {
                validate_amqp_uniform_batch(properties, headers, payloads)?;
                if !self
                    .amqp_targets(*project, exchange, routing_key)?
                    .is_empty()
                {
                    let ingress = IngressMetadata::Amqp {
                        exchange: exchange.clone(),
                        routing_key: routing_key.clone(),
                        properties: properties.clone(),
                        headers: headers.clone(),
                        death_count: 0,
                    };
                    let pending = payloads
                        .iter()
                        .enumerate()
                        .map(|(index, payload)| {
                            let increment = u64::try_from(index)
                                .map_err(|_| Error::internal("AMQP batch index exceeds u64"))?
                                .checked_add(1)
                                .ok_or_else(|| Error::internal("message ID exhausted"))?;
                            Ok(PendingPayloadRecord {
                                id: MessageId(
                                    self.next_message_id
                                        .checked_add(increment)
                                        .ok_or_else(|| Error::internal("message ID exhausted"))?,
                                ),
                                ingress: &ingress,
                                payload,
                            })
                        })
                        .collect::<Result<Vec<_>>>()?;
                    staged.push(
                        write_raw_amqp_payload_segment_batch(*project, &pending, segments)?.0,
                    );
                }
            }
            _ => {}
        }
        Ok(staged)
    }

    #[must_use]
    pub fn queue_info(&self, project: ProjectId, name: &str) -> Option<QueueInfo> {
        let queue = self.queues.get(&(project, name.to_owned()))?;
        let stream = self.streams.get(&queue.stream)?;
        Some(QueueInfo {
            kind: queue.kind,
            message_count: stream
                .records
                .iter()
                .filter(|record| {
                    !matches!(
                        record.delivery,
                        DeliveryState::Acknowledged | DeliveryState::DeadLettered
                    )
                })
                .count() as u64,
            available_count: stream
                .records
                .iter()
                .filter(|record| {
                    queue.kind == QueueKind::Stream || record.delivery == DeliveryState::Ready
                })
                .count() as u64,
        })
    }

    pub fn read_stream_queue(
        &self,
        project: ProjectId,
        queue: &str,
        offset: StreamOffset,
        maximum: usize,
        _consumer: u64,
        _automatic_ack: bool,
        segments: &SegmentStore,
    ) -> Result<Vec<Delivery>> {
        let pending = self.stream_queue_records(project, queue, offset, maximum)?;
        let loaded = load_payloads(
            &self.payload_segments,
            pending.iter().map(|record| record.stored.clone()),
            segments,
        )?;
        pending
            .into_iter()
            .map(|record| {
                let payload = loaded.get(&record.stored.id).cloned().ok_or_else(|| {
                    Error::internal("broker payload batch omitted a requested ID")
                })?;
                let (exchange, routing_key) =
                    delivery_route(&payload, record.exchange, record.routing_key);
                Ok(Delivery {
                    queue: queue.to_owned(),
                    offset: record.offset,
                    delivery_tag: record.offset.saturating_add(1),
                    redelivered: record.redelivered,
                    death_count: record.death_count,
                    exchange,
                    routing_key,
                    payload,
                })
            })
            .collect()
    }

    #[must_use]
    pub fn group_member(
        &self,
        project: ProjectId,
        group: &str,
        generation: i32,
        member: &str,
    ) -> bool {
        self.groups
            .get(&(project, group.to_owned()))
            .is_some_and(|state| {
                state.generation == generation && state.members.contains_key(member)
            })
    }

    #[must_use]
    pub fn group_assignment(
        &self,
        project: ProjectId,
        group: &str,
        generation: i32,
        member: &str,
    ) -> Option<Vec<u8>> {
        self.groups
            .get(&(project, group.to_owned()))
            .filter(|state| state.generation == generation && state.members.contains_key(member))
            .and_then(|state| state.assignments.get(member).cloned())
    }

    #[must_use]
    pub fn group_leader(&self, project: ProjectId, group: &str, generation: i32) -> Option<&str> {
        self.groups
            .get(&(project, group.to_owned()))
            .filter(|state| state.generation == generation)
            .map(|state| state.leader.as_str())
    }

    fn publish_kafka_batch(
        &mut self,
        project: ProjectId,
        topic: String,
        partition: i32,
        resolved_time_ms: i64,
        records: Vec<KafkaBatchRecord>,
        segments: &SegmentStore,
    ) -> Result<BrokerReply> {
        validate_kafka_batch(&records)?;
        let stream = self.partition_stream(project, &topic, partition)?;
        let record_count = u32::try_from(records.len())
            .map_err(|_| Error::invalid_data("Kafka partition batch exceeds u32 records"))?;
        let first_offset = self
            .streams
            .get(&stream)
            .ok_or_else(|| Error::internal("Kafka partition stream is missing"))?
            .next_offset;
        self.validate_stream_appends(&[stream], records.len() as u64)?;
        if kafka_batch_can_share_ingress(&records) {
            return self.publish_raw_kafka_batch(
                project,
                stream,
                first_offset,
                record_count,
                resolved_time_ms,
                records,
                segments,
            );
        }
        let mut ingress = Vec::with_capacity(records.len());
        for record in &records {
            ingress.push(IngressMetadata::Kafka {
                create_time_ms: record.create_time_ms,
                key: record.key.clone(),
                headers: record.headers.clone(),
                value_is_null: record.value_is_null,
            });
        }
        let pending = records
            .iter()
            .zip(&ingress)
            .enumerate()
            .map(|(index, (record, ingress))| {
                let increment = u64::try_from(index)
                    .map_err(|_| Error::internal("Kafka batch index exceeds u64"))?
                    .checked_add(1)
                    .ok_or_else(|| Error::internal("message ID exhausted"))?;
                Ok(PendingPayloadRecord {
                    id: MessageId(
                        self.next_message_id
                            .checked_add(increment)
                            .ok_or_else(|| Error::internal("message ID exhausted"))?,
                    ),
                    ingress,
                    payload: &record.payload,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let (descriptor, locations) =
            write_payload_segment_batch(project, resolved_time_ms, &pending, segments)?;
        let message_ids = pending.iter().map(|pending| pending.id).collect::<Vec<_>>();
        drop(pending);
        let segment = self.allocate_payload_segment(descriptor, record_count, None)?;
        let mut messages = Vec::with_capacity(records.len());
        for (((record, ingress), id), location) in records
            .into_iter()
            .zip(ingress)
            .zip(message_ids)
            .zip(locations)
        {
            let message = self.allocate_message_in_segment(
                id,
                resolved_time_ms,
                &ingress,
                record.payload,
                segment,
                location,
            )?;
            messages.push(message);
        }
        let appended_first = self.append_initial_references_batch(stream, &messages)?;
        if appended_first != first_offset {
            return Err(Error::internal(
                "Kafka atomic batch offset changed during publication",
            ));
        }
        Ok(BrokerReply::KafkaBatchPublished {
            first_offset,
            record_count,
        })
    }

    fn publish_raw_kafka_batch(
        &mut self,
        project: ProjectId,
        stream: StreamId,
        first_offset: u64,
        record_count: u32,
        resolved_time_ms: i64,
        records: Vec<KafkaBatchRecord>,
        segments: &SegmentStore,
    ) -> Result<BrokerReply> {
        let (descriptor, locations, checksums) =
            write_raw_kafka_payload_segment_batch(project, &records, segments)?;
        let shared_ingress = IngressMetadata::Kafka {
            create_time_ms: None,
            key: None,
            headers: BTreeMap::new(),
            value_is_null: false,
        };
        let segment = self.allocate_payload_segment_with_shared_ingress(
            descriptor,
            record_count,
            None,
            Some(shared_ingress),
        )?;
        let mut messages = Vec::with_capacity(records.len());
        for ((record, location), checksum) in records.into_iter().zip(locations).zip(checksums) {
            let id = MessageId(
                self.next_message_id
                    .checked_add(1)
                    .ok_or_else(|| Error::internal("message ID exhausted"))?,
            );
            let create_time_ms = record.create_time_ms.ok_or_else(|| {
                Error::internal("raw Kafka record lost its validated create timestamp")
            })?;
            let message = self.allocate_stored_message_in_segment(
                id,
                resolved_time_ms,
                record.payload.len(),
                checksum,
                Some(create_time_ms),
                segment,
                location,
            )?;
            messages.push(message);
        }
        let appended_first = self.append_initial_references_batch(stream, &messages)?;
        if appended_first != first_offset {
            return Err(Error::internal(
                "Kafka atomic batch offset changed during raw publication",
            ));
        }
        Ok(BrokerReply::KafkaBatchPublished {
            first_offset,
            record_count,
        })
    }

    fn publish_amqp(
        &mut self,
        project: ProjectId,
        exchange: String,
        routing_key: String,
        mandatory: bool,
        resolved_time_ms: i64,
        properties: BTreeMap<String, Vec<u8>>,
        headers: BTreeMap<String, Vec<u8>>,
        payload: Vec<u8>,
        segments: &SegmentStore,
    ) -> Result<BrokerReply> {
        let targets = if exchange.is_empty() {
            self.queues
                .get(&(project, routing_key.clone()))
                .map(|queue| BTreeSet::from([queue.stream]))
                .unwrap_or_default()
        } else {
            let declaration = self
                .exchanges
                .get(&(project, exchange.clone()))
                .ok_or_else(|| Error::invalid_data("exchange does not exist"))?;
            self.bindings
                .iter()
                .filter(|((binding_project, binding_exchange), binding)| {
                    *binding_project == project
                        && *binding_exchange == exchange
                        && route_matches(declaration.kind, &binding.routing_key, &routing_key)
                })
                .filter_map(|(_, binding)| {
                    self.queues
                        .get(&(project, binding.queue.clone()))
                        .map(|queue| queue.stream)
                })
                .collect::<BTreeSet<_>>()
        };
        if targets.is_empty() {
            let _ = mandatory;
            return Ok(BrokerReply::Published {
                message: MessageId(0),
                offsets: Vec::new(),
            });
        }
        self.validate_stream_appends(&targets.iter().copied().collect::<Vec<_>>(), 1)?;
        let ingress = IngressMetadata::Amqp {
            exchange: exchange.clone(),
            routing_key: routing_key.clone(),
            properties,
            headers,
            death_count: 0,
        };
        let id = MessageId(
            self.next_message_id
                .checked_add(1)
                .ok_or_else(|| Error::internal("message ID exhausted"))?,
        );
        let pending = [PendingPayloadRecord {
            id,
            ingress: &ingress,
            payload: &payload,
        }];
        let (descriptor, mut locations) =
            write_raw_amqp_payload_segment_batch(project, &pending, segments)?;
        let location = locations
            .pop()
            .ok_or_else(|| Error::internal("broker payload segment omitted its record extent"))?;
        let segment = self.allocate_payload_segment(descriptor, 1, Some(ingress.clone()))?;
        let message = self.allocate_message_in_segment(
            id,
            resolved_time_ms,
            &ingress,
            payload,
            segment,
            location,
        )?;
        let mut offsets = Vec::with_capacity(targets.len());
        for stream in targets {
            offsets.push((
                stream,
                self.append_reference_with_route(
                    stream,
                    message,
                    Some(exchange.clone()),
                    Some(routing_key.clone()),
                    0,
                )?,
            ));
        }
        Ok(BrokerReply::Published { message, offsets })
    }

    #[allow(clippy::too_many_arguments)]
    fn publish_amqp_uniform_batch(
        &mut self,
        project: ProjectId,
        resolved_time_ms: i64,
        exchange: String,
        routing_key: String,
        _mandatory: bool,
        properties: BTreeMap<String, Vec<u8>>,
        headers: BTreeMap<String, Vec<u8>>,
        payloads: Vec<Vec<u8>>,
        segments: &SegmentStore,
    ) -> Result<BrokerReply> {
        validate_amqp_uniform_batch(&properties, &headers, &payloads)?;
        let targets = self.amqp_targets(project, &exchange, &routing_key)?;
        self.validate_stream_appends(
            &targets.iter().copied().collect::<Vec<_>>(),
            payloads.len() as u64,
        )?;
        let outcome_count = payloads.len();
        if targets.is_empty() {
            return Ok(BrokerReply::AmqpUniformBatchPublished {
                record_count: u32::try_from(outcome_count)
                    .map_err(|_| Error::internal("AMQP batch record count exceeds u32"))?,
                routed: false,
            });
        }

        let ingress = IngressMetadata::Amqp {
            exchange: exchange.clone(),
            routing_key: routing_key.clone(),
            properties,
            headers,
            death_count: 0,
        };
        let pending = payloads
            .iter()
            .enumerate()
            .map(|(index, payload)| {
                let increment = u64::try_from(index)
                    .map_err(|_| Error::internal("AMQP batch index exceeds u64"))?
                    .checked_add(1)
                    .ok_or_else(|| Error::internal("message ID exhausted"))?;
                Ok(PendingPayloadRecord {
                    id: MessageId(
                        self.next_message_id
                            .checked_add(increment)
                            .ok_or_else(|| Error::internal("message ID exhausted"))?,
                    ),
                    ingress: &ingress,
                    payload,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let (descriptor, locations) =
            write_raw_amqp_payload_segment_batch(project, &pending, segments)?;
        let message_ids = pending.iter().map(|pending| pending.id).collect::<Vec<_>>();
        drop(pending);
        let live_records = u32::try_from(outcome_count)
            .map_err(|_| Error::internal("AMQP batch record count exceeds u32"))?;
        let segment =
            self.allocate_payload_segment(descriptor, live_records, Some(ingress.clone()))?;
        let mut messages = Vec::with_capacity(outcome_count);
        for ((payload, id), location) in payloads.into_iter().zip(message_ids).zip(locations) {
            let message = self.allocate_message_in_segment(
                id,
                resolved_time_ms,
                &ingress,
                payload,
                segment,
                location,
            )?;
            messages.push(message);
        }
        for stream in targets {
            let _ = self.append_initial_references_batch(stream, &messages)?;
        }
        Ok(BrokerReply::AmqpUniformBatchPublished {
            record_count: live_records,
            routed: true,
        })
    }

    fn publish_amqp_batch(
        &mut self,
        project: ProjectId,
        resolved_time_ms: i64,
        records: Vec<AmqpBatchRecord>,
        segments: &SegmentStore,
    ) -> Result<BrokerReply> {
        validate_amqp_batch(&records)?;
        let outcome_count = records.len();
        let mut stream_counts = BTreeMap::<StreamId, u64>::new();
        let mut routed = Vec::with_capacity(records.len());
        for (index, record) in records.into_iter().enumerate() {
            let targets = self.amqp_targets(project, &record.exchange, &record.routing_key)?;
            for stream in &targets {
                *stream_counts.entry(*stream).or_default() += 1;
            }
            if !targets.is_empty() {
                routed.push((index, record, targets));
            }
        }
        for (stream, count) in stream_counts {
            self.validate_stream_appends(&[stream], count)?;
        }
        let mut offsets = vec![Vec::new(); outcome_count];
        if routed.is_empty() {
            return Ok(BrokerReply::AmqpBatchPublished { offsets });
        }

        let ingress = routed
            .iter()
            .map(|(_, record, _)| IngressMetadata::Amqp {
                exchange: record.exchange.clone(),
                routing_key: record.routing_key.clone(),
                properties: record.properties.clone(),
                headers: record.headers.clone(),
                death_count: 0,
            })
            .collect::<Vec<_>>();
        let pending = routed
            .iter()
            .zip(&ingress)
            .enumerate()
            .map(|(index, ((_, record, _), ingress))| {
                let increment = u64::try_from(index)
                    .map_err(|_| Error::internal("AMQP batch index exceeds u64"))?
                    .checked_add(1)
                    .ok_or_else(|| Error::internal("message ID exhausted"))?;
                Ok(PendingPayloadRecord {
                    id: MessageId(
                        self.next_message_id
                            .checked_add(increment)
                            .ok_or_else(|| Error::internal("message ID exhausted"))?,
                    ),
                    ingress,
                    payload: &record.payload,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let shared_amqp_ingress = ingress
            .first()
            .filter(|first| ingress.iter().skip(1).all(|candidate| candidate == *first));
        let (descriptor, locations) = if shared_amqp_ingress.is_some() {
            write_raw_amqp_payload_segment_batch(project, &pending, segments)?
        } else {
            write_payload_segment_batch(project, resolved_time_ms, &pending, segments)?
        };
        let message_ids = pending.iter().map(|pending| pending.id).collect::<Vec<_>>();
        drop(pending);
        let live_records = u32::try_from(routed.len())
            .map_err(|_| Error::internal("AMQP batch record count exceeds u32"))?;
        let segment =
            self.allocate_payload_segment(descriptor, live_records, shared_amqp_ingress.cloned())?;
        for ((((original_index, record, targets), ingress), id), location) in routed
            .into_iter()
            .zip(ingress)
            .zip(message_ids)
            .zip(locations)
        {
            let exchange = record.exchange.clone();
            let routing_key = record.routing_key.clone();
            let message = self.allocate_message_in_segment(
                id,
                resolved_time_ms,
                &ingress,
                record.payload,
                segment,
                location,
            )?;
            let record_offsets = &mut offsets[original_index];
            record_offsets.reserve(targets.len());
            for stream in targets {
                record_offsets.push((
                    stream,
                    self.append_reference_with_route(
                        stream,
                        message,
                        Some(exchange.clone()),
                        Some(routing_key.clone()),
                        0,
                    )?,
                ));
            }
        }
        Ok(BrokerReply::AmqpBatchPublished { offsets })
    }

    fn allocate_message_in_segment(
        &mut self,
        id: MessageId,
        timestamp_ms: i64,
        ingress: &IngressMetadata,
        payload: Vec<u8>,
        segment: u64,
        location: SegmentRecordLocation,
    ) -> Result<MessageId> {
        let checksum = *blake3::hash(&payload).as_bytes();
        let kafka_time_ms = match ingress {
            IngressMetadata::Kafka { create_time_ms, .. } => {
                Some(create_time_ms.unwrap_or(timestamp_ms))
            }
            IngressMetadata::Amqp { .. } => None,
        };
        self.allocate_stored_message_in_segment(
            id,
            timestamp_ms,
            payload.len(),
            checksum,
            kafka_time_ms,
            segment,
            location,
        )
    }

    fn allocate_stored_message_in_segment(
        &mut self,
        id: MessageId,
        timestamp_ms: i64,
        payload_len: usize,
        checksum: [u8; 32],
        kafka_time_ms: Option<i64>,
        segment: u64,
        location: SegmentRecordLocation,
    ) -> Result<MessageId> {
        if payload_len > 256 * 1024 * 1024 {
            return Err(Error::new(
                crate::ErrorCode::Backpressure,
                "broker payload exceeds 256 MiB",
            ));
        }
        let next_message_id = self
            .next_message_id
            .checked_add(1)
            .ok_or_else(|| Error::internal("message ID exhausted"))?;
        if id != MessageId(next_message_id) {
            return Err(Error::internal(
                "prepared broker message ID differs from the allocator cursor",
            ));
        }
        let payload_bytes = u64::try_from(payload_len)
            .map_err(|_| Error::internal("broker payload length exceeds u64"))?;
        let descriptor = &self
            .payload_segments
            .get(segment)
            .ok_or_else(|| Error::internal("broker payload segment is missing"))?
            .descriptor;
        if location.bytes < payload_bytes
            || location
                .offset
                .checked_add(location.bytes)
                .is_none_or(|end| end > descriptor.bytes)
        {
            return Err(Error::internal(
                "broker payload record extent differs from its segment",
            ));
        }
        self.next_message_id = next_message_id;
        if self
            .payloads
            .insert(
                id,
                StoredPayloadRecord {
                    id,
                    resolved_time_ms: timestamp_ms,
                    segment,
                    location,
                    retained_bytes: location.bytes,
                    payload_bytes,
                    checksum,
                    kafka_time_ms,
                },
            )?
            .is_some()
        {
            return Err(Error::internal("broker message ID was reused"));
        }
        Ok(id)
    }

    fn allocate_payload_segment(
        &mut self,
        descriptor: SegmentDescriptor,
        live_records: u32,
        shared_amqp_ingress: Option<IngressMetadata>,
    ) -> Result<u64> {
        self.allocate_payload_segment_with_shared_ingress(
            descriptor,
            live_records,
            shared_amqp_ingress,
            None,
        )
    }

    fn allocate_payload_segment_with_shared_ingress(
        &mut self,
        descriptor: SegmentDescriptor,
        live_records: u32,
        shared_amqp_ingress: Option<IngressMetadata>,
        shared_kafka_ingress: Option<IngressMetadata>,
    ) -> Result<u64> {
        if live_records == 0 || descriptor.family != SegmentFamily::BrokerPayload {
            return Err(Error::internal(
                "broker payload segment metadata is invalid",
            ));
        }
        let shared_ingress_bytes = shared_amqp_ingress
            .as_ref()
            .or(shared_kafka_ingress.as_ref())
            .map_or(0, ingress_accounted_bytes);
        self.next_payload_segment_id = self
            .next_payload_segment_id
            .checked_add(1)
            .ok_or_else(|| Error::internal("broker payload segment ID exhausted"))?;
        let id = self.next_payload_segment_id;
        if self
            .payload_segments
            .insert(
                id,
                Arc::new(BrokerPayloadSegment {
                    id,
                    descriptor,
                    live_records,
                    shared_amqp_ingress,
                    shared_kafka_ingress,
                    shared_ingress_bytes,
                }),
            )
            .is_some()
        {
            return Err(Error::internal("broker payload segment ID was reused"));
        }
        Ok(id)
    }

    fn accounted_payload_bytes(&self, payload: &StoredPayloadRecord) -> Result<u64> {
        let segment = self.payload_segments.get(payload.segment).ok_or_else(|| {
            Error::new(
                crate::ErrorCode::CorruptStorage,
                "broker payload segment table is incomplete",
            )
        })?;
        let shared_ingress_bytes = if segment.shared_ingress_bytes != 0 {
            segment.shared_ingress_bytes
        } else {
            // Compatibility with snapshots written before the cached charge existed.
            segment
                .shared_amqp_ingress
                .as_ref()
                .or(segment.shared_kafka_ingress.as_ref())
                .map_or(0, ingress_accounted_bytes)
        };
        payload
            .retained_bytes
            .checked_add(shared_ingress_bytes)
            .ok_or_else(|| Error::internal("broker accounted byte length overflow"))
    }

    fn allocate_stream(&mut self) -> Result<StreamId> {
        self.next_stream_id = self
            .next_stream_id
            .checked_add(1)
            .ok_or_else(|| Error::internal("stream ID exhausted"))?;
        let id = StreamId(self.next_stream_id);
        self.streams.insert(id, Arc::new(StreamState::default()));
        Ok(id)
    }

    #[cfg(test)]
    fn append_reference(&mut self, stream: StreamId, message: MessageId) -> Result<u64> {
        self.append_reference_with_route(stream, message, None, None, 0)
    }

    fn append_initial_references_batch(
        &mut self,
        stream: StreamId,
        messages: &[MessageId],
    ) -> Result<u64> {
        let common_segment = messages
            .first()
            .and_then(|message| self.payloads.get(message))
            .map(|payload| payload.segment);
        let common_shared_ingress_bytes = common_segment
            .and_then(|segment| self.payload_segments.get(segment))
            .map(|segment| {
                if segment.shared_ingress_bytes != 0 {
                    segment.shared_ingress_bytes
                } else {
                    segment
                        .shared_amqp_ingress
                        .as_ref()
                        .or(segment.shared_kafka_ingress.as_ref())
                        .map_or(0, ingress_accounted_bytes)
                }
            })
            .unwrap_or(0);
        let retained_bytes = messages.iter().try_fold(0_u64, |total, message| {
            let payload = self
                .payloads
                .get(message)
                .ok_or_else(|| Error::internal("broker batch payload is missing"))?;
            let bytes = if Some(payload.segment) == common_segment {
                payload
                    .retained_bytes
                    .checked_add(common_shared_ingress_bytes)
                    .ok_or_else(|| Error::internal("broker accounted byte length overflow"))?
            } else {
                self.accounted_payload_bytes(payload)?
            };
            total
                .checked_add(bytes)
                .ok_or_else(|| Error::internal("broker retained byte accounting overflow"))
        })?;
        let count = u64::try_from(messages.len())
            .map_err(|_| Error::internal("broker batch length exceeds u64"))?;
        let state = self
            .streams
            .get_mut(&stream)
            .ok_or_else(|| Error::internal("broker stream does not exist"))?;
        let state = Arc::make_mut(state);
        let first_offset = state.next_offset;
        let next_offset = first_offset
            .checked_add(count)
            .ok_or_else(|| Error::internal("broker offset exhausted"))?;
        let next_retained_bytes = state
            .retained_bytes
            .checked_add(retained_bytes)
            .ok_or_else(|| Error::internal("broker retained byte accounting overflow"))?;
        for (index, message) in messages.iter().copied().enumerate() {
            state.records.push_back(StreamReference {
                offset: first_offset + index as u64,
                message,
                delivery: DeliveryState::Ready,
                redelivered: false,
                death_count: 0,
                // The immutable payload segment owns the shared original route. Overrides are
                // stored only after rerouting/dead-lettering, avoiding two strings per publish.
                current_exchange: None,
                current_routing_key: None,
            });
        }
        state.next_offset = next_offset;
        state.retained_bytes = next_retained_bytes;
        Ok(first_offset)
    }

    fn append_reference_with_route(
        &mut self,
        stream: StreamId,
        message: MessageId,
        current_exchange: Option<String>,
        current_routing_key: Option<String>,
        death_count: u32,
    ) -> Result<u64> {
        let retained_bytes = self
            .payloads
            .get(&message)
            .map(|record| self.accounted_payload_bytes(record))
            .transpose()?
            .unwrap_or(0);
        let state = self
            .streams
            .get_mut(&stream)
            .ok_or_else(|| Error::internal("broker stream does not exist"))?;
        let state = Arc::make_mut(state);
        let offset = state.next_offset;
        state.next_offset = state
            .next_offset
            .checked_add(1)
            .ok_or_else(|| Error::internal("broker offset exhausted"))?;
        state.retained_bytes = state
            .retained_bytes
            .checked_add(retained_bytes)
            .ok_or_else(|| Error::internal("broker retained byte accounting overflow"))?;
        state.records.push_back(StreamReference {
            offset,
            message,
            delivery: DeliveryState::Ready,
            redelivered: false,
            death_count,
            current_exchange,
            current_routing_key,
        });
        Ok(offset)
    }

    fn partition_stream(
        &self,
        project: ProjectId,
        topic: &str,
        partition: i32,
    ) -> Result<StreamId> {
        if partition < 0 {
            return Err(Error::invalid_data("partition must be non-negative"));
        }
        self.topics
            .get(&(project, topic.to_owned()))
            .and_then(|topic| topic.partitions.get(partition as usize).copied())
            .ok_or_else(|| Error::invalid_data("topic partition does not exist"))
    }

    fn fetch_stream(
        &self,
        stream: StreamId,
        offset: u64,
        maximum_bytes: usize,
        include_settled: bool,
        segments: &SegmentStore,
    ) -> Result<Vec<(u64, Arc<PayloadRecord>)>> {
        let pending = self.fetch_stream_records(stream, offset, maximum_bytes, include_settled)?;
        let loaded = load_payloads(
            &self.payload_segments,
            pending.iter().map(|(_, stored)| stored.clone()),
            segments,
        )?;
        pending
            .into_iter()
            .map(|(offset, stored)| {
                Ok((
                    offset,
                    loaded.get(&stored.id).cloned().ok_or_else(|| {
                        Error::internal("broker payload batch omitted a requested ID")
                    })?,
                ))
            })
            .collect()
    }

    fn fetch_stream_records(
        &self,
        stream: StreamId,
        offset: u64,
        maximum_bytes: usize,
        include_settled: bool,
    ) -> Result<Vec<(u64, StoredPayloadRecord)>> {
        self.fetch_stream_records_with_bounds(stream, offset, maximum_bytes, include_settled, None)
    }

    fn fetch_stream_records_with_bounds(
        &self,
        stream: StreamId,
        offset: u64,
        maximum_bytes: usize,
        include_settled: bool,
        maximum_records: Option<usize>,
    ) -> Result<Vec<(u64, StoredPayloadRecord)>> {
        let state = self
            .streams
            .get(&stream)
            .ok_or_else(|| Error::invalid_data("stream does not exist"))?;
        if offset < state.base_offset {
            return Err(Error::new(
                crate::ErrorCode::RetentionExpired,
                "requested offset was retained",
            ));
        }
        let mut bytes = 0usize;
        let mut pending = Vec::new();
        for reference in state
            .records
            .iter()
            .filter(|record| record.offset >= offset)
        {
            if maximum_records.is_some_and(|maximum| pending.len() >= maximum) {
                break;
            }
            if !include_settled
                && matches!(
                    reference.delivery,
                    DeliveryState::Acknowledged | DeliveryState::DeadLettered
                )
            {
                continue;
            }
            let stored = self
                .payloads
                .get(&reference.message)
                .ok_or_else(|| Error::internal("broker payload is missing"))?;
            let visible_bytes = usize::try_from(self.accounted_payload_bytes(stored)?)
                .map_err(|_| Error::internal("broker record length exceeds this platform"))?;
            if bytes.saturating_add(visible_bytes) > maximum_bytes
                && (!pending.is_empty() || maximum_records.is_some())
            {
                if pending.is_empty() {
                    return Err(Error::new(
                        crate::ErrorCode::ResultBudgetExceeded,
                        "first stream record exceeds the explicit byte bound",
                    ));
                }
                break;
            }
            bytes = bytes.saturating_add(visible_bytes);
            pending.push((reference.offset, stored.clone()));
        }
        Ok(pending)
    }

    fn resolve_offset(&self, stream: &StreamState, offset: StreamOffset) -> Result<u64> {
        Ok(match offset {
            StreamOffset::First => stream.base_offset,
            StreamOffset::Last => stream.next_offset.saturating_sub(1),
            StreamOffset::Absolute(value) => {
                if value < stream.base_offset {
                    return Err(Error::new(
                        crate::ErrorCode::RetentionExpired,
                        "stream offset was retained",
                    ));
                }
                value
            }
            StreamOffset::Timestamp(timestamp) => stream
                .records
                .iter()
                .find(|reference| {
                    self.payloads
                        .get(&reference.message)
                        .is_some_and(|payload| payload.resolved_time_ms >= timestamp)
                })
                .map_or(stream.next_offset, |reference| reference.offset),
            StreamOffset::Next => stream.next_offset,
        })
    }

    fn settle(
        &mut self,
        project: ProjectId,
        queue: &str,
        owner: Uuid,
        consumer: u64,
        delivery_tag: u64,
        multiple: bool,
        ack: bool,
        requeue: bool,
    ) -> Result<()> {
        let queue_record = self
            .queues
            .get(&(project, queue.to_owned()))
            .cloned()
            .ok_or_else(|| Error::invalid_data("queue does not exist"))?;
        if queue_record.kind != QueueKind::Classic {
            return Err(Error::invalid_data(
                "stream queues use offsets, not destructive acknowledgements",
            ));
        }
        let mut dead_letters = Vec::new();
        let matched = {
            let stream = self
                .streams
                .get_mut(&queue_record.stream)
                .ok_or_else(|| Error::internal("queue stream is missing"))?;
            let stream = Arc::make_mut(stream);
            let offsets = if multiple {
                stream
                    .records
                    .iter()
                    .take_while(|reference| reference.offset.saturating_add(1) <= delivery_tag)
                    .filter_map(|reference| {
                        matches!(
                            reference.delivery,
                            DeliveryState::Unacked {
                                owner: candidate,
                                consumer: candidate_consumer,
                                ..
                            } if candidate == owner && candidate_consumer == consumer
                        )
                        .then_some(reference.offset)
                    })
                    .collect::<Vec<_>>()
            } else {
                delivery_tag
                    .checked_sub(1)
                    .and_then(|offset| {
                        stream
                            .records
                            .get_by_offset(stream.base_offset, offset)
                            .filter(|reference| {
                                matches!(
                                    reference.delivery,
                                    DeliveryState::Unacked {
                                        owner: candidate,
                                        consumer: candidate_consumer,
                                        tag,
                                    } if candidate == owner
                                        && candidate_consumer == consumer
                                        && tag == delivery_tag
                                )
                            })
                            .map(|_| vec![offset])
                    })
                    .unwrap_or_default()
            };
            let base_offset = stream.base_offset;
            for offset in &offsets {
                let reference = stream
                    .records
                    .get_mut_by_offset(base_offset, *offset)
                    .ok_or_else(|| Error::internal("settled delivery offset disappeared"))?;
                reference.delivery = if ack {
                    DeliveryState::Acknowledged
                } else if requeue {
                    reference.redelivered = true;
                    DeliveryState::Ready
                } else {
                    dead_letters.push((reference.message, reference.death_count.saturating_add(1)));
                    DeliveryState::DeadLettered
                };
            }
            !offsets.is_empty()
        };
        if !matched {
            return Err(Error::new(
                crate::ErrorCode::ProtocolViolation,
                "delivery tag is unknown",
            ));
        }
        self.remove_delivery_lease_if_idle(project, owner);
        if !dead_letters.is_empty() {
            let Some(exchange) = queue_record.dead_letter_exchange else {
                return Ok(());
            };
            let routing = queue_record.dead_letter_routing_key.unwrap_or_default();
            let targets = if exchange.is_empty() {
                self.queues
                    .get(&(project, routing.clone()))
                    .map(|queue| BTreeSet::from([queue.stream]))
                    .unwrap_or_default()
            } else {
                let declaration = self
                    .exchanges
                    .get(&(project, exchange.clone()))
                    .ok_or_else(|| Error::internal("configured dead-letter exchange is missing"))?;
                self.bindings
                    .iter()
                    .filter(|((binding_project, binding_exchange), binding)| {
                        *binding_project == project
                            && *binding_exchange == exchange
                            && route_matches(declaration.kind, &binding.routing_key, &routing)
                    })
                    .filter_map(|(_, binding)| {
                        self.queues
                            .get(&(project, binding.queue.clone()))
                            .map(|queue| queue.stream)
                    })
                    .collect::<BTreeSet<_>>()
            };
            for (message, death_count) in dead_letters {
                for target in &targets {
                    let _ = self.append_reference_with_route(
                        *target,
                        message,
                        Some(exchange.clone()),
                        Some(routing.clone()),
                        death_count,
                    )?;
                }
            }
        }
        Ok(())
    }

    fn renew_delivery_lease(
        &mut self,
        project: ProjectId,
        owner: Uuid,
        resolved_time_ms: i64,
    ) -> Result<()> {
        if !self.owner_has_unacked(project, owner) {
            self.delivery_leases.remove(&(project, owner));
            return Ok(());
        }
        let deadline = resolved_time_ms
            .checked_add(DELIVERY_LEASE_MILLIS)
            .ok_or_else(|| Error::internal("delivery lease deadline overflow"))?;
        self.delivery_leases
            .entry((project, owner))
            .and_modify(|current| *current = (*current).max(deadline))
            .or_insert(deadline);
        Ok(())
    }

    fn release_delivery_lease(&mut self, project: ProjectId, owner: Uuid) {
        let streams = self
            .queues
            .iter()
            .filter_map(|((candidate, _), queue)| {
                (*candidate == project && queue.kind == QueueKind::Classic).then_some(queue.stream)
            })
            .collect::<Vec<_>>();
        for stream in streams {
            let Some(stream) = self.streams.get_mut(&stream) else {
                continue;
            };
            let stream = Arc::make_mut(stream);
            let offsets = stream
                .records
                .iter()
                .filter_map(|reference| {
                    matches!(
                        reference.delivery,
                        DeliveryState::Unacked {
                            owner: candidate,
                            ..
                        } if candidate == owner
                    )
                    .then_some(reference.offset)
                })
                .collect::<Vec<_>>();
            let base_offset = stream.base_offset;
            for offset in offsets {
                if let Some(reference) = stream.records.get_mut_by_offset(base_offset, offset) {
                    reference.delivery = DeliveryState::Ready;
                    reference.redelivered = true;
                }
            }
        }
        self.delivery_leases.remove(&(project, owner));
    }

    fn expire_delivery_leases(&mut self, project: ProjectId, resolved_time_ms: i64) {
        let expired = self
            .delivery_leases
            .iter()
            .filter_map(|((candidate, owner), deadline)| {
                (*candidate == project && *deadline <= resolved_time_ms).then_some(*owner)
            })
            .collect::<Vec<_>>();
        for owner in expired {
            self.release_delivery_lease(project, owner);
        }
    }

    fn remove_delivery_lease_if_idle(&mut self, project: ProjectId, owner: Uuid) {
        if !self.owner_has_unacked(project, owner) {
            self.delivery_leases.remove(&(project, owner));
        }
    }

    fn owner_has_unacked(&self, project: ProjectId, owner: Uuid) -> bool {
        self.queues
            .iter()
            .filter(|((candidate, _), queue)| {
                *candidate == project && queue.kind == QueueKind::Classic
            })
            .filter_map(|(_, queue)| self.streams.get(&queue.stream))
            .flat_map(|stream| &stream.records)
            .any(|reference| {
                matches!(
                    reference.delivery,
                    DeliveryState::Unacked {
                        owner: candidate,
                        ..
                    } if candidate == owner
                )
            })
    }

    fn retain(
        &mut self,
        project: ProjectId,
        resolved_time_ms: i64,
    ) -> Result<(u64, BTreeSet<MessageId>)> {
        let mut policies = HashMap::new();
        for ((id, _), topic) in &*self.topics {
            if *id == project {
                for stream in &topic.partitions {
                    policies.insert(*stream, topic.retention);
                }
            }
        }
        for ((id, _), queue) in &*self.queues {
            if *id == project {
                policies.insert(queue.stream, queue.retention);
            }
        }
        let mut plan = Vec::new();
        for (stream_id, policy) in policies {
            let state = self
                .streams
                .get(&stream_id)
                .ok_or_else(|| Error::internal("retention stream is missing"))?;
            let mut retained_bytes = state.retained_bytes;
            let mut remove = 0_usize;
            for front in &state.records {
                // An active classic delivery remains canonical until its owner settles it or the
                // durable lease expires and maintenance requeues it. Retention cannot erase
                // the reference while the delivery lease still names it.
                if matches!(front.delivery, DeliveryState::Unacked { .. }) {
                    break;
                }
                let payload = self
                    .payloads
                    .get(&front.message)
                    .ok_or_else(|| Error::internal("retention payload is missing"))?;
                let too_old = policy.max_age_ms.is_some_and(|age| {
                    i64::try_from(age).is_ok_and(|age| {
                        resolved_time_ms.saturating_sub(payload.resolved_time_ms) >= age
                    })
                });
                let too_large = policy
                    .max_bytes
                    .is_some_and(|maximum| retained_bytes > maximum);
                let settled_classic = matches!(
                    front.delivery,
                    DeliveryState::Acknowledged | DeliveryState::DeadLettered
                );
                if !(too_old || too_large || settled_classic) {
                    break;
                }
                retained_bytes =
                    retained_bytes.saturating_sub(self.accounted_payload_bytes(payload)?);
                remove = remove.saturating_add(1);
            }
            if remove != 0 {
                plan.push((stream_id, remove, state.retained_bytes - retained_bytes));
            }
        }

        let mut removed = 0u64;
        let mut candidates = BTreeSet::new();
        for (stream_id, count, removed_bytes) in plan {
            let state = self
                .streams
                .get_mut(&stream_id)
                .ok_or_else(|| Error::internal("retention stream is missing"))?;
            let state = Arc::make_mut(state);
            for _ in 0..count {
                let reference = state
                    .records
                    .pop_front()
                    .ok_or_else(|| Error::internal("retention plan exceeds stream length"))?;
                state.base_offset = reference.offset.saturating_add(1);
                candidates.insert(reference.message);
                removed = removed.saturating_add(1);
            }
            state.retained_bytes = state.retained_bytes.saturating_sub(removed_bytes);
        }
        Ok((removed, candidates))
    }

    fn expire_group_members(&mut self, project: ProjectId, resolved_time_ms: i64) -> Result<()> {
        let changed = self
            .groups
            .iter()
            .filter(|((candidate, _), _)| *candidate == project)
            .filter_map(|(key, state)| {
                state
                    .members
                    .values()
                    .any(|member| {
                        resolved_time_ms.saturating_sub(member.last_heartbeat_ms)
                            >= i64::from(member.session_timeout_ms)
                    })
                    .then_some(key.clone())
            })
            .collect::<Vec<_>>();
        for key in changed {
            let state = self
                .groups
                .get_mut(&key)
                .ok_or_else(|| Error::internal("expiring consumer group disappeared"))?;
            let state = Arc::make_mut(state);
            state.members.retain(|_, member| {
                resolved_time_ms.saturating_sub(member.last_heartbeat_ms)
                    < i64::from(member.session_timeout_ms)
            });
            state.generation = state
                .generation
                .checked_add(1)
                .ok_or_else(|| Error::internal("group generation exhausted"))?;
            state.assignments.clear();
            if !state.members.contains_key(&state.leader) {
                state.leader = state.members.keys().next().cloned().unwrap_or_default();
            }
            state.protocol = if state.members.is_empty() {
                String::new()
            } else {
                select_group_protocol(&state.members)?
            };
        }
        Ok(())
    }

    /// Records a group has yet to consume, per subscribed partition. This is the load signal the
    /// assignor balances on, and it exists only here: a consumer can measure lag for partitions it
    /// already owns, never for the ones it would need in order to decide it is overloaded.
    fn subscribed_partition_lag(
        &self,
        project: ProjectId,
        group: &GroupState,
        topics: &BTreeSet<String>,
    ) -> BTreeMap<(String, i32), u64> {
        let mut lag = BTreeMap::new();
        for topic in topics {
            let Some(declared) = self.topics.get(&(project, topic.clone())) else {
                continue;
            };
            for (index, stream) in declared.partitions.iter().enumerate() {
                let Ok(partition) = i32::try_from(index) else {
                    continue;
                };
                let Some(state) = self.streams.get(stream) else {
                    continue;
                };
                // An uncommitted partition is weighed from its base rather than from zero, so a
                // long-retained topic does not look infinitely heavy the first time it is read.
                let committed = group
                    .offsets
                    .get(&(topic.clone(), partition))
                    .copied()
                    .unwrap_or(state.base_offset);
                lag.insert(
                    (topic.clone(), partition),
                    state.next_offset.saturating_sub(committed),
                );
            }
        }
        lag
    }

    /// Assignment computed by the broker from live lag, replacing whatever the group leader
    /// proposed. Returns `None` when no member's metadata could be decoded, which leaves the
    /// client-side assignor in charge rather than emptying the group.
    fn rebalanced_assignment(
        &self,
        project: ProjectId,
        group: &GroupState,
    ) -> Option<BTreeMap<String, Vec<u8>>> {
        // Every member has to decode, not merely some. Balancing over a subset would install an
        // assignment map that omits the members whose metadata was unreadable, and a member absent
        // from the map receives no partitions at all — it would sit idle indefinitely rather than
        // fall back. One unreadable member sends the whole group back to the client assignor.
        let mut subscriptions = BTreeMap::new();
        for (member, record) in &group.members {
            let topics = record
                .protocols
                .get(&group.protocol)
                .and_then(|metadata| decode_subscribed_topics(metadata))?;
            subscriptions.insert(member.clone(), topics);
        }
        if subscriptions.is_empty() {
            return None;
        }
        let topics = subscriptions
            .values()
            .flatten()
            .cloned()
            .collect::<BTreeSet<_>>();
        let lag = self.subscribed_partition_lag(project, group, &topics);
        Some(
            balance_partitions_by_lag(&subscriptions, &lag)
                .into_iter()
                .map(|(member, partitions)| (member, encode_partition_assignment(&partitions)))
                .collect(),
        )
    }

    fn queue_is_auto_delete(&self, project: ProjectId, queue: &str) -> bool {
        self.queues
            .get(&(project, queue.to_owned()))
            .is_some_and(|declared| declared.auto_delete)
    }

    /// Removes a queue together with everything that references it: its stream, its bindings, its
    /// consumer registrations, and the payloads it was the last holder of.
    fn drop_queue(&mut self, project: ProjectId, name: &str) -> Result<()> {
        let key = (project, name.to_owned());
        let Some(queue) = self.queues.remove(&key) else {
            return Ok(());
        };
        self.queue_consumers.remove(&key);
        self.bindings
            .retain(|(binding_key, binding)| !(binding_key.0 == project && binding.queue == name));
        let removed_payloads = self
            .streams
            .get(&queue.stream)
            .map(|state| state.records.iter().map(|record| record.message).collect())
            .unwrap_or_else(BTreeSet::new);
        self.streams.remove(&queue.stream);
        self.reclaim_payload_candidates(project, &removed_payloads)
    }

    fn binding_exists(
        &self,
        project: ProjectId,
        exchange: &str,
        queue: &str,
        routing_key: &str,
    ) -> bool {
        self.bindings.iter().any(|(key, binding)| {
            key.0 == project
                && binding.exchange == exchange
                && binding.queue == queue
                && binding.routing_key == routing_key
        })
    }

    fn exchange_has_bindings(&self, project: ProjectId, exchange: &str) -> bool {
        self.bindings
            .iter()
            .any(|(key, binding)| key.0 == project && binding.exchange == exchange)
    }

    /// Ready and unsettled counts for a queue, after enforcing the `queue.delete` preconditions.
    /// Shared by validation and apply so a command that validates cannot then fail while applying.
    fn queue_deletion_counts(
        &self,
        project: ProjectId,
        name: &str,
        if_unused: bool,
        if_empty: bool,
    ) -> Result<u32> {
        let queue = self
            .queues
            .get(&(project, name.to_owned()))
            .ok_or_else(|| Error::invalid_data("queue does not exist"))?;
        let ready = self.streams.get(&queue.stream).map_or(0, |state| {
            state
                .records
                .iter()
                .filter(|record| matches!(record.delivery, DeliveryState::Ready))
                .fold(0u32, |count, _| count.saturating_add(1))
        });
        if if_unused
            && self
                .queue_consumers
                .get(&(project, name.to_owned()))
                .is_some_and(|consumers| !consumers.is_empty())
        {
            return Err(Error::invalid_data("queue is still in use"));
        }
        if if_empty && ready != 0 {
            return Err(Error::invalid_data("queue is not empty"));
        }
        Ok(ready)
    }

    /// Settles every ready record on a stream so it stops being deliverable and becomes eligible
    /// for the ordinary retention trim. Records are only ever removed from the front of a stream,
    /// so a purge cannot physically drop a ready record that sits behind an unsettled one; marking
    /// it settled is what makes the purge observable, and reclamation follows on the next trim.
    fn discard_ready_records(&mut self, project: ProjectId, stream: StreamId) -> Result<u32> {
        let _ = project;
        let Some(state) = self.streams.get_mut(&stream) else {
            return Ok(0);
        };
        let state = Arc::make_mut(state);
        let mut discarded = 0u32;
        for index in 0..state.records.len() {
            let Some(record) = state.records.get_mut_relative(index) else {
                break;
            };
            if matches!(record.delivery, DeliveryState::Ready) {
                record.delivery = DeliveryState::Acknowledged;
                discarded = discarded.saturating_add(1);
            }
        }
        Ok(discarded)
    }

    fn reclaim_payload_candidates(
        &mut self,
        project: ProjectId,
        candidates: &BTreeSet<MessageId>,
    ) -> Result<()> {
        if candidates.is_empty() {
            return Ok(());
        }
        let project_streams = self
            .topics
            .iter()
            .filter(|((candidate, _), _)| *candidate == project)
            .flat_map(|(_, topic)| topic.partitions.iter().copied())
            .chain(
                self.queues
                    .iter()
                    .filter(|((candidate, _), _)| *candidate == project)
                    .map(|(_, queue)| queue.stream),
            )
            .collect::<BTreeSet<_>>();
        let still_live = project_streams
            .iter()
            .filter_map(|stream| self.streams.get(stream))
            .flat_map(|stream| stream.records.iter().map(|record| record.message))
            .filter(|message| candidates.contains(message))
            .collect::<BTreeSet<_>>();
        for message in candidates.difference(&still_live) {
            self.remove_payload(*message)?;
        }
        Ok(())
    }

    fn remove_payload(&mut self, message: MessageId) -> Result<()> {
        let Some(payload) = self.payloads.remove(&message) else {
            return Ok(());
        };
        let remove_segment = {
            let segment = self
                .payload_segments
                .get_mut(payload.segment)
                .ok_or_else(|| Error::internal("broker payload segment is missing"))?;
            let segment = Arc::make_mut(segment);
            segment.live_records = segment
                .live_records
                .checked_sub(1)
                .ok_or_else(|| Error::internal("broker payload segment live count underflow"))?;
            segment.live_records == 0
        };
        if remove_segment {
            self.payload_segments
                .remove(payload.segment)
                .ok_or_else(|| Error::internal("broker payload segment disappeared"))?;
        }
        Ok(())
    }
}

fn write_payload_segment_batch(
    project: ProjectId,
    resolved_time_ms: i64,
    pending: &[PendingPayloadRecord<'_>],
    segments: &SegmentStore,
) -> Result<(SegmentDescriptor, Vec<SegmentRecordLocation>)> {
    if pending.is_empty() {
        return Err(Error::invalid_data("broker payload segment batch is empty"));
    }
    // This is the publish delta path, not a cold rebuild: encode only this command's records and
    // publish their derived segment without an acknowledgement-path fsync. The complete payload
    // remains in the eventual-durability WAL and can recreate a crash-lost segment; snapshot
    // attachment makes the segment durable before compacting that WAL suffix.
    // Records are independent. Encode the bounded command batch across the CPU pool before the
    // ordered writer frames it; this keeps filesystem publication serial and deterministic while
    // removing CBOR, payload hashing, and allocation from the single apply core. The temporary
    // vector is bounded by MAX_BROKER_BATCH_BYTES and consumed entry-by-entry without cloning.
    let mut encoded = pending
        .par_iter()
        .map(|pending| {
            encode_payload_segment_record(
                project,
                pending.id,
                resolved_time_ms,
                pending.ingress,
                pending.payload,
            )
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .map(Some)
        .collect::<Vec<_>>();
    segments.write_replayable_immutable_streaming_indexed(
        SegmentFamily::BrokerPayload,
        Some(project),
        encoded.len(),
        |index| {
            encoded
                .get_mut(index)
                .and_then(Option::take)
                .ok_or_else(|| Error::internal("encoded broker payload batch index disappeared"))
        },
    )
}

fn write_raw_amqp_payload_segment_batch(
    project: ProjectId,
    pending: &[PendingPayloadRecord<'_>],
    segments: &SegmentStore,
) -> Result<(SegmentDescriptor, Vec<SegmentRecordLocation>)> {
    if pending.is_empty()
        || !pending
            .iter()
            .all(|record| matches!(record.ingress, IngressMetadata::Amqp { .. }))
    {
        return Err(Error::invalid_data(
            "raw AMQP payload segment batch is empty or heterogeneous",
        ));
    }
    segments.write_replayable_immutable_streaming_indexed(
        SegmentFamily::BrokerPayload,
        Some(project),
        pending.len(),
        |index| {
            let payload = pending
                .get(index)
                .ok_or_else(|| Error::internal("raw AMQP payload batch index disappeared"))?
                .payload;
            Ok(SegmentRecord {
                kind: BROKER_RAW_AMQP_PAYLOAD_SEGMENT_KIND,
                payload: payload.to_vec(),
            })
        },
    )
}

fn write_raw_kafka_payload_segment_batch(
    project: ProjectId,
    records: &[KafkaBatchRecord],
    segments: &SegmentStore,
) -> Result<(SegmentDescriptor, Vec<SegmentRecordLocation>, Vec<[u8; 32]>)> {
    if !kafka_batch_can_share_ingress(records) {
        return Err(Error::invalid_data(
            "raw Kafka payload segment batch is empty or carries per-record envelope data",
        ));
    }
    let first_payload = records
        .first()
        .ok_or_else(|| Error::internal("validated Kafka batch disappeared"))?
        .payload
        .as_slice();
    if records
        .iter()
        .skip(1)
        .all(|record| record.payload.as_slice() == first_payload)
    {
        let (descriptor, unique_locations) = segments
            .write_replayable_immutable_streaming_indexed(
                SegmentFamily::BrokerPayload,
                Some(project),
                1,
                |_| {
                    Ok(SegmentRecord {
                        kind: BROKER_RAW_KAFKA_PAYLOAD_SEGMENT_KIND,
                        payload: first_payload.to_vec(),
                    })
                },
            )?;
        let location = *unique_locations
            .first()
            .ok_or_else(|| Error::internal("interned Kafka payload location disappeared"))?;
        let checksum = *blake3::hash(first_payload).as_bytes();
        return Ok((
            descriptor,
            vec![location; records.len()],
            vec![checksum; records.len()],
        ));
    }
    let mut unique_payloads = Vec::<&[u8]>::new();
    let mut unique_index = HashMap::<&[u8], usize>::with_capacity(records.len());
    let mut payload_indices = Vec::with_capacity(records.len());
    for record in records {
        let next = unique_payloads.len();
        let payload = record.payload.as_slice();
        let index = *unique_index.entry(payload).or_insert_with(|| {
            unique_payloads.push(payload);
            next
        });
        payload_indices.push(index);
    }
    let (descriptor, unique_locations) = segments.write_replayable_immutable_streaming_indexed(
        SegmentFamily::BrokerPayload,
        Some(project),
        unique_payloads.len(),
        |index| {
            let payload = unique_payloads
                .get(index)
                .ok_or_else(|| Error::internal("unique Kafka payload index disappeared"))?;
            Ok(SegmentRecord {
                kind: BROKER_RAW_KAFKA_PAYLOAD_SEGMENT_KIND,
                payload: payload.to_vec(),
            })
        },
    )?;
    let unique_checksums = unique_payloads
        .iter()
        .map(|payload| *blake3::hash(payload).as_bytes())
        .collect::<Vec<_>>();
    let mut locations = Vec::with_capacity(payload_indices.len());
    let mut checksums = Vec::with_capacity(payload_indices.len());
    for index in payload_indices {
        locations.push(
            unique_locations
                .get(index)
                .copied()
                .ok_or_else(|| Error::internal("unique Kafka payload location disappeared"))?,
        );
        checksums.push(
            *unique_checksums
                .get(index)
                .ok_or_else(|| Error::internal("unique Kafka payload checksum disappeared"))?,
        );
    }
    Ok((descriptor, locations, checksums))
}

fn encode_payload_segment_record(
    project: ProjectId,
    id: MessageId,
    resolved_time_ms: i64,
    ingress: &IngressMetadata,
    payload: &[u8],
) -> Result<SegmentRecord> {
    let checksum = *blake3::hash(payload).as_bytes();
    let mut segment_payload = Vec::with_capacity(
        BROKER_PAYLOAD_RECORD_V2_MAGIC
            .len()
            .saturating_add(payload.len())
            .saturating_add(128),
    );
    segment_payload.extend_from_slice(BROKER_PAYLOAD_RECORD_V2_MAGIC);
    segment_payload = postcard::to_extend(
        &CompactPayloadSegmentRecordRef {
            project,
            id,
            resolved_time_ms,
            ingress: compact_ingress(ingress),
            payload,
            checksum,
        },
        segment_payload,
    )
    .map_err(|error| Error::invalid_data(format!("broker payload encoding failed: {error}")))?;
    Ok(SegmentRecord {
        kind: BROKER_PAYLOAD_SEGMENT_KIND,
        payload: segment_payload,
    })
}

fn decode_payload_segment_record(encoded: &[u8]) -> Result<PayloadSegmentRecord> {
    if let Some(payload) = encoded.strip_prefix(BROKER_PAYLOAD_RECORD_V2_MAGIC) {
        let compact: CompactPayloadSegmentRecord =
            postcard::from_bytes(payload).map_err(|error| {
                Error::new(
                    crate::ErrorCode::CorruptStorage,
                    format!("broker payload segment v2 is invalid: {error}"),
                )
            })?;
        Ok(PayloadSegmentRecord {
            project: compact.project,
            id: compact.id,
            resolved_time_ms: compact.resolved_time_ms,
            ingress: expand_compact_ingress(compact.ingress),
            payload: compact.payload,
            checksum: compact.checksum,
        })
    } else {
        // Snapshot compatibility for segments published before the compact record format.
        ciborium::de::from_reader(encoded).map_err(|error| {
            Error::new(
                crate::ErrorCode::CorruptStorage,
                format!("legacy broker payload segment is invalid: {error}"),
            )
        })
    }
}

fn load_payloads(
    payload_segments: &PayloadSegmentTable,
    stored: impl IntoIterator<Item = StoredPayloadRecord>,
    segments: &SegmentStore,
) -> Result<BTreeMap<MessageId, Arc<PayloadRecord>>> {
    let mut requested = Vec::new();
    let mut expected = BTreeSet::new();
    for stored in stored {
        if !expected.insert(stored.id) {
            return Err(Error::internal(
                "broker payload request repeats a canonical message ID",
            ));
        }
        requested.push(stored);
    }
    requested.sort_by_key(|stored| (stored.segment, stored.location.offset));
    let mut loaded = BTreeMap::new();
    for stored in requested {
        let segment = payload_segments.get(stored.segment).ok_or_else(|| {
            Error::new(
                crate::ErrorCode::CorruptStorage,
                "broker payload segment table is incomplete",
            )
        })?;
        let record = segments.read_record_at(&segment.descriptor, stored.location)?;
        let payload = materialize_stored_payload(&record, &stored, segment)?;
        validate_loaded_payload(&stored, &segment.descriptor, &payload)?;
        if loaded
            .insert(
                payload.id,
                Arc::new(PayloadRecord {
                    id: payload.id,
                    resolved_time_ms: payload.resolved_time_ms,
                    ingress: payload.ingress,
                    payload: Arc::from(payload.payload),
                    checksum: payload.checksum,
                }),
            )
            .is_some()
        {
            return Err(Error::new(
                crate::ErrorCode::CorruptStorage,
                "broker payload segment repeats a message ID",
            ));
        }
    }
    if loaded.len() != expected.len() || expected.iter().any(|id| !loaded.contains_key(id)) {
        return Err(Error::new(
            crate::ErrorCode::CorruptStorage,
            "broker payload segment omitted a canonical message ID",
        ));
    }
    Ok(loaded)
}

fn materialize_stored_payload(
    record: &SegmentRecord,
    stored: &StoredPayloadRecord,
    segment: &BrokerPayloadSegment,
) -> Result<PayloadSegmentRecord> {
    match record.kind {
        BROKER_PAYLOAD_SEGMENT_KIND => decode_payload_segment_record(&record.payload),
        BROKER_RAW_AMQP_PAYLOAD_SEGMENT_KIND => {
            let ingress = segment.shared_amqp_ingress.clone().ok_or_else(|| {
                Error::new(
                    crate::ErrorCode::CorruptStorage,
                    "raw AMQP payload segment is missing its shared ingress metadata",
                )
            })?;
            if !matches!(ingress, IngressMetadata::Amqp { .. }) {
                return Err(Error::new(
                    crate::ErrorCode::CorruptStorage,
                    "raw AMQP payload segment carries non-AMQP ingress metadata",
                ));
            }
            Ok(PayloadSegmentRecord {
                project: segment.descriptor.project_id.ok_or_else(|| {
                    Error::new(
                        crate::ErrorCode::CorruptStorage,
                        "raw AMQP payload segment has no project",
                    )
                })?,
                id: stored.id,
                resolved_time_ms: stored.resolved_time_ms,
                ingress,
                payload: record.payload.clone(),
                checksum: stored.checksum,
            })
        }
        BROKER_RAW_KAFKA_PAYLOAD_SEGMENT_KIND => {
            let ingress = segment.shared_kafka_ingress.as_ref().ok_or_else(|| {
                Error::new(
                    crate::ErrorCode::CorruptStorage,
                    "raw Kafka payload segment is missing its shared ingress metadata",
                )
            })?;
            let IngressMetadata::Kafka {
                key,
                headers,
                value_is_null,
                ..
            } = ingress
            else {
                return Err(Error::new(
                    crate::ErrorCode::CorruptStorage,
                    "raw Kafka payload segment carries non-Kafka ingress metadata",
                ));
            };
            if key.is_some() || !headers.is_empty() || *value_is_null {
                return Err(Error::new(
                    crate::ErrorCode::CorruptStorage,
                    "raw Kafka payload segment carries a non-uniform envelope",
                ));
            }
            Ok(PayloadSegmentRecord {
                project: segment.descriptor.project_id.ok_or_else(|| {
                    Error::new(
                        crate::ErrorCode::CorruptStorage,
                        "raw Kafka payload segment has no project",
                    )
                })?,
                id: stored.id,
                resolved_time_ms: stored.resolved_time_ms,
                ingress: IngressMetadata::Kafka {
                    create_time_ms: stored.kafka_time_ms,
                    key: None,
                    headers: BTreeMap::new(),
                    value_is_null: false,
                },
                payload: record.payload.clone(),
                checksum: stored.checksum,
            })
        }
        _ => Err(Error::new(
            crate::ErrorCode::CorruptStorage,
            "broker payload segment has an unknown record kind",
        )),
    }
}

fn unique_payload_descriptors(
    payload_segments: &PayloadSegmentTable,
    stored: impl IntoIterator<Item = StoredPayloadRecord>,
) -> Result<Vec<SegmentDescriptor>> {
    stored
        .into_iter()
        .map(|stored| {
            payload_segments
                .get(stored.segment)
                .map(|segment| segment.descriptor.clone())
                .ok_or_else(|| Error::internal("broker payload segment table is incomplete"))
        })
        .collect::<Result<BTreeSet<_>>>()
        .map(BTreeSet::into_iter)
        .map(Iterator::collect)
}

fn validate_loaded_payload(
    stored: &StoredPayloadRecord,
    descriptor: &SegmentDescriptor,
    payload: &PayloadSegmentRecord,
) -> Result<()> {
    let kafka_time_ms = match &payload.ingress {
        IngressMetadata::Kafka { create_time_ms, .. } => {
            Some(create_time_ms.unwrap_or(payload.resolved_time_ms))
        }
        IngressMetadata::Amqp { .. } => None,
    };
    if descriptor.family != SegmentFamily::BrokerPayload
        || descriptor.project_id != Some(payload.project)
        || payload.id != stored.id
        || payload.resolved_time_ms != stored.resolved_time_ms
        || payload.payload.len() as u64 != stored.payload_bytes
        || stored.location.bytes != stored.retained_bytes
        || payload.checksum != stored.checksum
        || kafka_time_ms != stored.kafka_time_ms
        || *blake3::hash(&payload.payload).as_bytes() != payload.checksum
    {
        return Err(Error::new(
            crate::ErrorCode::CorruptStorage,
            "broker payload segment differs from its canonical reference",
        ));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub enum StreamOffset {
    First,
    Last,
    Absolute(u64),
    Timestamp(i64),
    Next,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Delivery {
    pub queue: String,
    pub offset: u64,
    pub delivery_tag: u64,
    pub redelivered: bool,
    pub death_count: u32,
    pub exchange: String,
    pub routing_key: String,
    pub payload: Arc<PayloadRecord>,
}

#[derive(Clone, Copy, Debug)]
pub struct QueueInfo {
    pub kind: QueueKind,
    pub message_count: u64,
    pub available_count: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TopicMetrics {
    pub name: String,
    pub partition: u16,
    pub base_offset: u64,
    pub next_offset: u64,
    pub record_count: u64,
    pub retained_bytes: u64,
    pub retention_ms: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QueueMetrics {
    pub name: String,
    pub kind: QueueKind,
    pub message_count: u64,
    pub available_count: u64,
    pub retained_bytes: u64,
    pub retention_ms: Option<u64>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExchangeMetrics {
    pub name: String,
    pub kind: AmqpExchangeKind,
    pub durable: bool,
    pub binding_count: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ConsumerLagMetrics {
    pub group: String,
    pub topic: String,
    pub partition: i32,
    pub committed_offset: u64,
    pub next_offset: u64,
    pub lag: u64,
}

fn route_matches(kind: AmqpExchangeKind, binding: &str, routing: &str) -> bool {
    match kind {
        AmqpExchangeKind::Fanout => true,
        AmqpExchangeKind::Direct => binding == routing,
        AmqpExchangeKind::Topic => topic_match(binding, routing),
    }
}

fn topic_match(binding: &str, routing: &str) -> bool {
    // AMQP 0-9-1 topic matching, evaluated as a bounded dynamic program rather than by
    // backtracking. The recursive formulation retries every split point for each `#`, which costs
    // `C(segments + hashes, hashes)` attempts and does not terminate for bindings that the 255-byte
    // name limit still admits. Routing runs inside the deterministic local apply path, and the
    // publish path reaches it while the global apply mutex is held, so an unbounded match stalls
    // every unrelated graph write rather than merely slowing one request.
    //
    // `reachable[index]` records whether the pattern consumed so far can leave the routing key
    // positioned at segment `index`. Each pattern segment rewrites that row exactly once, so the
    // total work is bounded by `pattern segments * routing segments` for every possible input.
    let pattern = binding.split('.').collect::<Vec<_>>();
    let value = routing.split('.').collect::<Vec<_>>();
    let mut reachable = vec![false; value.len().saturating_add(1)];
    let Some(origin) = reachable.first_mut() else {
        return false;
    };
    *origin = true;
    let mut next = vec![false; reachable.len()];
    for segment in pattern {
        if segment == "#" {
            // `#` consumes zero or more segments, so every position at or after a reachable
            // position becomes reachable. One prefix scan replaces the backtracking entirely.
            let mut carried = false;
            for (entry, reached) in next.iter_mut().zip(reachable.iter()) {
                carried |= *reached;
                *entry = carried;
            }
        } else {
            // `*` consumes exactly one segment; a literal consumes one matching segment.
            for entry in &mut next {
                *entry = false;
            }
            for (index, reached) in reachable.iter().enumerate() {
                if !*reached {
                    continue;
                }
                let Some(actual) = value.get(index) else {
                    continue;
                };
                if segment != "*" && segment != *actual {
                    continue;
                }
                if let Some(entry) = next.get_mut(index.saturating_add(1)) {
                    *entry = true;
                }
            }
        }
        std::mem::swap(&mut reachable, &mut next);
    }
    // The binding matches only when it consumed the routing key exactly.
    reachable.last().copied().unwrap_or(false)
}

fn valid_name(name: &str) -> Result<()> {
    if name.is_empty()
        || name.len() > 255
        || name
            .bytes()
            .any(|byte| byte == 0 || byte.is_ascii_control())
    {
        return Err(Error::invalid_data(
            "broker name is empty, oversized, or contains control bytes",
        ));
    }
    Ok(())
}

fn validate_payload(payload: &[u8]) -> Result<()> {
    if payload.len() > 256 * 1024 * 1024 {
        return Err(Error::new(
            crate::ErrorCode::Backpressure,
            "broker payload exceeds 256 MiB",
        ));
    }
    Ok(())
}

fn delivery_route(
    payload: &PayloadRecord,
    exchange_override: Option<String>,
    routing_key_override: Option<String>,
) -> (String, String) {
    let (original_exchange, original_routing_key) = match &payload.ingress {
        IngressMetadata::Amqp {
            exchange,
            routing_key,
            ..
        } => (exchange.as_str(), routing_key.as_str()),
        IngressMetadata::Kafka { .. } => ("", ""),
    };
    (
        exchange_override.unwrap_or_else(|| original_exchange.to_owned()),
        routing_key_override.unwrap_or_else(|| original_routing_key.to_owned()),
    )
}

fn kafka_batch_can_share_ingress(records: &[KafkaBatchRecord]) -> bool {
    !records.is_empty()
        && records.iter().all(|record| {
            record.create_time_ms.is_some()
                && record.key.is_none()
                && record.headers.is_empty()
                && !record.value_is_null
        })
}

fn validate_kafka_batch(records: &[KafkaBatchRecord]) -> Result<()> {
    if records.is_empty() || records.len() > 1_000_000 {
        return Err(Error::invalid_data(
            "Kafka partition batch must contain 1..=1000000 records",
        ));
    }
    let mut encoded_bytes = 0_usize;
    for record in records {
        validate_payload(&record.payload)?;
        if record.headers.iter().any(|(key, value)| {
            key.is_empty() || key.len() > 32_767 || value.len() > MAX_BROKER_BATCH_BYTES
        }) {
            return Err(Error::invalid_data("Kafka record header is invalid"));
        }
        encoded_bytes = encoded_bytes
            .checked_add(record.payload.len())
            .and_then(|bytes| bytes.checked_add(record.key.as_ref().map_or(0, Vec::len)))
            .and_then(|bytes| {
                record
                    .headers
                    .iter()
                    .try_fold(bytes, |total, (key, value)| {
                        total.checked_add(key.len())?.checked_add(value.len())
                    })
            })
            .ok_or_else(|| {
                Error::new(crate::ErrorCode::Backpressure, "Kafka batch is too large")
            })?;
        if encoded_bytes > MAX_BROKER_BATCH_BYTES {
            return Err(Error::new(
                crate::ErrorCode::Backpressure,
                "Kafka partition batch exceeds the request frame bound",
            ));
        }
    }
    Ok(())
}

fn validate_amqp_batch(records: &[AmqpBatchRecord]) -> Result<()> {
    if records.is_empty() || records.len() > 1_000_000 {
        return Err(Error::invalid_data(
            "AMQP publish batch must contain 1..=1000000 records",
        ));
    }
    let mut encoded_bytes = 0_usize;
    for record in records {
        validate_payload(&record.payload)?;
        encoded_bytes = encoded_bytes
            .checked_add(record.payload.len())
            .and_then(|bytes| {
                record
                    .properties
                    .iter()
                    .chain(&record.headers)
                    .try_fold(bytes, |total, (key, value)| {
                        total.checked_add(key.len())?.checked_add(value.len())
                    })
            })
            .ok_or_else(|| Error::new(crate::ErrorCode::Backpressure, "AMQP batch is too large"))?;
        if encoded_bytes > MAX_BROKER_BATCH_BYTES {
            return Err(Error::new(
                crate::ErrorCode::Backpressure,
                "AMQP publish batch exceeds the request frame bound",
            ));
        }
    }
    Ok(())
}

fn validate_amqp_uniform_batch(
    properties: &BTreeMap<String, Vec<u8>>,
    headers: &BTreeMap<String, Vec<u8>>,
    payloads: &[Vec<u8>],
) -> Result<()> {
    if payloads.is_empty() || payloads.len() > 1_000_000 {
        return Err(Error::invalid_data(
            "AMQP publish batch must contain 1..=1000000 records",
        ));
    }
    let envelope_bytes = properties
        .iter()
        .chain(headers)
        .try_fold(0_usize, |total, (key, value)| {
            total.checked_add(key.len())?.checked_add(value.len())
        })
        .ok_or_else(|| Error::new(crate::ErrorCode::Backpressure, "AMQP batch is too large"))?;
    let mut encoded_bytes = envelope_bytes;
    for payload in payloads {
        validate_payload(payload)?;
        encoded_bytes = encoded_bytes
            .checked_add(payload.len())
            .ok_or_else(|| Error::new(crate::ErrorCode::Backpressure, "AMQP batch is too large"))?;
        if encoded_bytes > MAX_BROKER_BATCH_BYTES {
            return Err(Error::new(
                crate::ErrorCode::Backpressure,
                "AMQP publish batch exceeds the request frame bound",
            ));
        }
    }
    Ok(())
}

fn validate_group_member(
    session_timeout_ms: u32,
    protocols: &BTreeMap<String, Vec<u8>>,
) -> Result<()> {
    if !(1_000..=86_400_000).contains(&session_timeout_ms) {
        return Err(Error::invalid_data(
            "Kafka group session timeout must be 1000..=86400000 milliseconds",
        ));
    }
    if protocols.is_empty() || protocols.len() > 128 {
        return Err(Error::invalid_data(
            "Kafka group member protocol set is empty or excessive",
        ));
    }
    for (protocol, metadata) in protocols {
        valid_name(protocol)?;
        if metadata.len() > 16 * 1024 * 1024 {
            return Err(Error::new(
                crate::ErrorCode::Backpressure,
                "Kafka group member metadata exceeds 16 MiB",
            ));
        }
    }
    Ok(())
}

fn select_group_protocol(members: &BTreeMap<String, GroupMember>) -> Result<String> {
    let mut members = members.values();
    let first = members
        .next()
        .ok_or_else(|| Error::invalid_data("consumer group has no members"))?;
    let mut common = first.protocols.keys().cloned().collect::<BTreeSet<_>>();
    for member in members {
        common.retain(|protocol| member.protocols.contains_key(protocol));
    }
    common
        .into_iter()
        .next()
        .ok_or_else(|| Error::invalid_data("consumer group has no common assignment protocol"))
}

fn valid_routing_key(key: &str) -> Result<()> {
    if key.len() > 255 || key.bytes().any(|byte| byte == 0 || byte.is_ascii_control()) {
        return Err(Error::invalid_data(
            "broker routing key is oversized or contains control bytes",
        ));
    }
    Ok(())
}

/// Decodes a `ConsumerProtocolSubscription`, the blob a Kafka consumer sends as its join metadata.
///
/// ```text
/// version int16, topics [string], user_data nullable-bytes
/// ```
///
/// Anything the broker cannot parse yields no subscription rather than an error: an unrecognised
/// assignor version has to fall back to the client's own assignment, not break the group.
fn decode_subscribed_topics(metadata: &[u8]) -> Option<Vec<String>> {
    fn take<'a>(metadata: &'a [u8], cursor: &mut usize, count: usize) -> Option<&'a [u8]> {
        let end = cursor.checked_add(count)?;
        let slice = metadata.get(*cursor..end)?;
        *cursor = end;
        Some(slice)
    }

    let mut cursor = 0usize;
    let _version = i16::from_be_bytes(take(metadata, &mut cursor, 2)?.try_into().ok()?);
    let count = i32::from_be_bytes(take(metadata, &mut cursor, 4)?.try_into().ok()?);
    let count = usize::try_from(count).ok()?;
    // A topic name is at least a two-byte length, so a count larger than the remaining bytes can
    // only be a lie. Checking it here keeps a malformed blob from reserving an absurd vector.
    if count.saturating_mul(2) > metadata.len().saturating_sub(cursor) {
        return None;
    }
    let mut topics = Vec::with_capacity(count);
    for _ in 0..count {
        let length = i16::from_be_bytes(take(metadata, &mut cursor, 2)?.try_into().ok()?);
        let length = usize::try_from(length).ok()?;
        topics.push(String::from_utf8(take(metadata, &mut cursor, length)?.to_vec()).ok()?);
    }
    Some(topics)
}

/// Encodes a `ConsumerProtocolAssignment`, which is what a consumer parses out of its SyncGroup
/// response to learn which partitions it owns.
///
/// ```text
/// version int16, assignment [topic string, partitions [int32]], user_data nullable-bytes
/// ```
fn encode_partition_assignment(assignment: &BTreeMap<String, Vec<i32>>) -> Vec<u8> {
    let mut encoded = Vec::new();
    encoded.extend_from_slice(&0_i16.to_be_bytes());
    encoded.extend_from_slice(&(i32::try_from(assignment.len()).unwrap_or(0)).to_be_bytes());
    for (topic, partitions) in assignment {
        let name = topic.as_bytes();
        encoded.extend_from_slice(&(i16::try_from(name.len()).unwrap_or(0)).to_be_bytes());
        encoded.extend_from_slice(name);
        encoded.extend_from_slice(&(i32::try_from(partitions.len()).unwrap_or(0)).to_be_bytes());
        for partition in partitions {
            encoded.extend_from_slice(&partition.to_be_bytes());
        }
    }
    // user_data: null.
    encoded.extend_from_slice(&(-1_i32).to_be_bytes());
    encoded
}

/// Greedy longest-processing-time assignment: partitions are placed heaviest first, each onto the
/// member currently holding the least work.
///
/// This is the difference from a stock Kafka range or round-robin assignor, which balance by
/// partition *count* and therefore happily give one consumer four idle partitions and another one
/// partition holding a million-record backlog. Balancing by measured lag is only possible on the
/// broker, because a client cannot see any partition it does not already own. Greedy LPT is within
/// 4/3 of optimal, which is far closer than counting partitions ever gets.
///
/// A single member takes everything it subscribes to, so "one consumer goes through all of it" and
/// "many consumers share the load" are the same code path with a different member count.
fn balance_partitions_by_lag(
    subscriptions: &BTreeMap<String, Vec<String>>,
    lag: &BTreeMap<(String, i32), u64>,
) -> BTreeMap<String, BTreeMap<String, Vec<i32>>> {
    let mut assignment: BTreeMap<String, BTreeMap<String, Vec<i32>>> = subscriptions
        .keys()
        .map(|member| (member.clone(), BTreeMap::new()))
        .collect();
    let mut load: BTreeMap<String, u64> = subscriptions
        .keys()
        .map(|member| (member.clone(), 0))
        .collect();

    // Heaviest first, with the partition key as a tiebreak so repeated evaluation of the same
    // canonical command reaches a byte-identical assignment.
    let mut ordered = lag.iter().collect::<Vec<_>>();
    ordered.sort_by(|left, right| right.1.cmp(left.1).then_with(|| left.0.cmp(right.0)));

    for ((topic, partition), weight) in ordered {
        let chosen = subscriptions
            .iter()
            .filter(|(_, topics)| topics.iter().any(|subscribed| subscribed == topic))
            .min_by(|left, right| {
                let left_load = load.get(left.0).copied().unwrap_or(0);
                let right_load = load.get(right.0).copied().unwrap_or(0);
                left_load.cmp(&right_load).then_with(|| left.0.cmp(right.0))
            })
            .map(|(member, _)| member.clone());
        let Some(member) = chosen else {
            // Nobody subscribes to this topic. Leaving it unassigned is correct; inventing an
            // owner would hand records to a consumer that never asked for them.
            continue;
        };
        load.entry(member.clone())
            .and_modify(|total| *total = total.saturating_add(*weight))
            .or_insert(*weight);
        assignment
            .entry(member)
            .or_default()
            .entry(topic.clone())
            .or_default()
            .push(*partition);
    }
    assignment
}

fn group_reply(state: &GroupState) -> BrokerReply {
    let metadata = state
        .members
        .iter()
        .map(|(member, record)| {
            (
                member.clone(),
                record
                    .protocols
                    .get(&state.protocol)
                    .cloned()
                    .unwrap_or_default(),
            )
        })
        .collect();
    BrokerReply::Group {
        generation: state.generation,
        leader: state.leader.clone(),
        members: state.members.keys().cloned().collect(),
        protocol: state.protocol.clone(),
        metadata,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delivery_route_uses_shared_ingress_until_a_reference_overrides_it() {
        let payload = PayloadRecord {
            id: MessageId(1),
            resolved_time_ms: 0,
            ingress: IngressMetadata::Amqp {
                exchange: "original".to_owned(),
                routing_key: "created".to_owned(),
                properties: BTreeMap::new(),
                headers: BTreeMap::new(),
                death_count: 0,
            },
            payload: Arc::<[u8]>::from([]),
            checksum: [0; 32],
        };
        assert_eq!(
            delivery_route(&payload, None, None),
            ("original".to_owned(), "created".to_owned())
        );
        assert_eq!(
            delivery_route(
                &payload,
                Some("dead-letter".to_owned()),
                Some("failed".to_owned())
            ),
            ("dead-letter".to_owned(), "failed".to_owned())
        );
    }

    #[test]
    fn uniform_amqp_command_serializes_one_envelope_for_unrelated_batch_sizes() -> Result<()> {
        fn encoded_len(command: &BrokerCommand) -> Result<usize> {
            let mut encoded = Vec::new();
            ciborium::ser::into_writer(command, &mut encoded)
                .map_err(|error| Error::internal(format!("encode broker command: {error}")))?;
            Ok(encoded.len())
        }

        let make_uniform = |count| BrokerCommand::PublishAmqpUniformBatch {
            project: ProjectId(Uuid::from_bytes([7; 16])),
            resolved_time_ms: 1,
            exchange: "events".to_owned(),
            routing_key: "created".to_owned(),
            mandatory: false,
            properties: BTreeMap::from([("content_type".to_owned(), b"application/json".to_vec())]),
            headers: BTreeMap::from([("source".to_owned(), b"shape-test".to_vec())]),
            payloads: (0..count).map(|_| vec![3; 64]).collect(),
        };
        let make_heterogeneous = |count| BrokerCommand::PublishAmqpBatch {
            project: ProjectId(Uuid::from_bytes([7; 16])),
            resolved_time_ms: 1,
            records: (0..count)
                .map(|_| AmqpBatchRecord {
                    exchange: "events".to_owned(),
                    routing_key: "created".to_owned(),
                    mandatory: false,
                    properties: BTreeMap::from([(
                        "content_type".to_owned(),
                        b"application/json".to_vec(),
                    )]),
                    headers: BTreeMap::from([("source".to_owned(), b"shape-test".to_vec())]),
                    payload: vec![3; 64],
                })
                .collect(),
        };

        let compact_40 = encoded_len(&make_uniform(40))?;
        let compact_400 = encoded_len(&make_uniform(400))?;
        let repeated_400 = encoded_len(&make_heterogeneous(400))?;
        assert!(
            compact_400 > compact_40 * 8,
            "work must track the rows involved"
        );
        assert!(
            compact_400 * 2 < repeated_400,
            "the common envelope must not be serialized once per message"
        );
        Ok(())
    }

    #[test]
    fn uniform_amqp_reply_does_not_materialize_per_message_offsets() -> Result<()> {
        let compact = BrokerReply::AmqpUniformBatchPublished {
            record_count: 400,
            routed: true,
        };
        let detailed = BrokerReply::AmqpBatchPublished {
            offsets: (0..400).map(|offset| vec![(StreamId(9), offset)]).collect(),
        };
        let mut compact_bytes = Vec::new();
        let mut detailed_bytes = Vec::new();
        ciborium::ser::into_writer(&compact, &mut compact_bytes)
            .map_err(|error| Error::internal(format!("encode compact broker reply: {error}")))?;
        ciborium::ser::into_writer(&detailed, &mut detailed_bytes)
            .map_err(|error| Error::internal(format!("encode detailed broker reply: {error}")))?;
        assert!(compact_bytes.len() < 64);
        assert!(detailed_bytes.len() > compact_bytes.len() * 20);
        Ok(())
    }

    #[test]
    fn compact_payload_record_round_trips_and_legacy_cbor_still_loads() -> Result<()> {
        let project = ProjectId::random();
        let id = MessageId(41);
        let ingress = IngressMetadata::Amqp {
            exchange: "events".to_owned(),
            routing_key: "created".to_owned(),
            properties: BTreeMap::new(),
            headers: BTreeMap::new(),
            death_count: 0,
        };
        let body = b"compact body";
        let checksum = *blake3::hash(body).as_bytes();
        let encoded = encode_payload_segment_record(project, id, 7, &ingress, body)?;
        assert!(encoded.payload.starts_with(BROKER_PAYLOAD_RECORD_V2_MAGIC));
        let decoded = decode_payload_segment_record(&encoded.payload)?;
        assert_eq!(decoded.project, project);
        assert_eq!(decoded.id, id);
        assert_eq!(decoded.resolved_time_ms, 7);
        assert_eq!(decoded.payload, body);
        assert_eq!(decoded.checksum, checksum);
        assert!(matches!(
            decoded.ingress,
            IngressMetadata::Amqp {
                ref exchange,
                ref routing_key,
                death_count: 0,
                ..
            } if exchange == "events" && routing_key == "created"
        ));

        let mut legacy = Vec::new();
        ciborium::ser::into_writer(
            &PayloadSegmentRecordRef {
                project,
                id,
                resolved_time_ms: 7,
                ingress: &ingress,
                payload: body,
                checksum,
            },
            &mut legacy,
        )
        .map_err(|error| Error::invalid_data(error.to_string()))?;
        let decoded_legacy = decode_payload_segment_record(&legacy)?;
        assert_eq!(decoded_legacy.project, project);
        assert_eq!(decoded_legacy.id, id);
        assert_eq!(decoded_legacy.payload, body);
        Ok(())
    }

    fn subscription(topics: &[&str]) -> Vec<u8> {
        let mut encoded = Vec::new();
        encoded.extend_from_slice(&0_i16.to_be_bytes());
        encoded.extend_from_slice(&(i32::try_from(topics.len()).unwrap_or(0)).to_be_bytes());
        for topic in topics {
            encoded.extend_from_slice(&(i16::try_from(topic.len()).unwrap_or(0)).to_be_bytes());
            encoded.extend_from_slice(topic.as_bytes());
        }
        encoded.extend_from_slice(&(-1_i32).to_be_bytes());
        encoded
    }

    /// Decodes what `encode_partition_assignment` produced, so the round trip is checked against a
    /// reader written from the Kafka format rather than against the encoder itself.
    fn decode_assignment(bytes: &[u8]) -> BTreeMap<String, Vec<i32>> {
        let mut cursor = 2usize;
        let mut read = |count: usize| {
            let slice = bytes[cursor..cursor + count].to_vec();
            cursor += count;
            slice
        };
        let topic_count = i32::from_be_bytes(read(4).try_into().unwrap_or([0; 4]));
        let mut assignment = BTreeMap::new();
        for _ in 0..topic_count {
            let name_length = i16::from_be_bytes(read(2).try_into().unwrap_or([0; 2]));
            let name = String::from_utf8(read(usize::try_from(name_length).unwrap_or(0)))
                .unwrap_or_default();
            let partition_count = i32::from_be_bytes(read(4).try_into().unwrap_or([0; 4]));
            let mut partitions = Vec::new();
            for _ in 0..partition_count {
                partitions.push(i32::from_be_bytes(read(4).try_into().unwrap_or([0; 4])));
            }
            assignment.insert(name, partitions);
        }
        assignment
    }

    #[test]
    fn a_subscription_blob_round_trips_and_a_malformed_one_is_declined() {
        assert_eq!(
            decode_subscribed_topics(&subscription(&["events", "audit"])),
            Some(vec!["events".to_owned(), "audit".to_owned()])
        );
        assert_eq!(decode_subscribed_topics(&subscription(&[])), Some(vec![]));
        // A count far larger than the bytes that follow must be declined rather than reserving
        // for it: this blob is client-supplied and reaches the broker before any authorization.
        let mut lying = Vec::new();
        lying.extend_from_slice(&0_i16.to_be_bytes());
        lying.extend_from_slice(&1_000_000_i32.to_be_bytes());
        assert_eq!(decode_subscribed_topics(&lying), None);
        assert_eq!(decode_subscribed_topics(&[]), None);
        assert_eq!(decode_subscribed_topics(&[0, 0, 0]), None);
    }

    #[test]
    fn a_lone_consumer_is_assigned_every_partition_it_subscribes_to() {
        // "can be one consumer going through all" — the same code path as the shared case, just
        // with one member.
        let subscriptions = BTreeMap::from([("solo".to_owned(), vec!["events".to_owned()])]);
        let lag = BTreeMap::from([
            (("events".to_owned(), 0), 10),
            (("events".to_owned(), 1), 20),
            (("events".to_owned(), 2), 30),
        ]);
        let assignment = balance_partitions_by_lag(&subscriptions, &lag);
        assert_eq!(
            assignment["solo"]["events"],
            vec![2, 1, 0],
            "a single member must own every partition of its subscription"
        );
    }

    #[test]
    fn partitions_are_balanced_by_lag_rather_than_by_count() {
        // The case a stock range or round-robin assignor gets wrong. Four partitions, two members.
        // Counting partitions gives each member two and leaves one holding 900 of the 960 backlog.
        // Weighting by lag puts the single heavy partition alone against the three light ones.
        let subscriptions = BTreeMap::from([
            ("fast".to_owned(), vec!["events".to_owned()]),
            ("slow".to_owned(), vec!["events".to_owned()]),
        ]);
        let lag = BTreeMap::from([
            (("events".to_owned(), 0), 900),
            (("events".to_owned(), 1), 20),
            (("events".to_owned(), 2), 20),
            (("events".to_owned(), 3), 20),
        ]);
        let assignment = balance_partitions_by_lag(&subscriptions, &lag);

        let heavy_owner = assignment
            .iter()
            .find(|(_, topics)| topics.get("events").is_some_and(|list| list.contains(&0)))
            .map(|(member, _)| member.clone())
            .expect("the heavy partition must be assigned");
        let heavy = &assignment[&heavy_owner]["events"];
        assert_eq!(
            heavy,
            &vec![0],
            "the member holding a 900-record backlog must not also be given the light partitions"
        );

        let total: usize = assignment
            .values()
            .filter_map(|topics| topics.get("events"))
            .map(Vec::len)
            .sum();
        assert_eq!(total, 4, "every partition must be assigned exactly once");
    }

    #[test]
    fn every_member_that_is_balanced_receives_an_entry() {
        // The failure this guards is silent: a member missing from the assignment map is not told
        // it owns nothing, it is simply never told anything, and consumes no records for the rest
        // of the generation. So balancing must produce one entry per member even when a member
        // ends up with no partitions, which is what happens whenever members outnumber partitions.
        let subscriptions = BTreeMap::from([
            ("a".to_owned(), vec!["events".to_owned()]),
            ("b".to_owned(), vec!["events".to_owned()]),
            ("c".to_owned(), vec!["events".to_owned()]),
        ]);
        let lag = BTreeMap::from([(("events".to_owned(), 0), 7)]);
        let assignment = balance_partitions_by_lag(&subscriptions, &lag);
        assert_eq!(
            assignment.keys().collect::<Vec<_>>(),
            vec!["a", "b", "c"],
            "every member must appear, including the ones that end up idle"
        );
        let assigned: usize = assignment.values().filter_map(|t| t.get("events")).count();
        assert_eq!(assigned, 1, "the one partition goes to exactly one member");
    }

    #[test]
    fn a_partition_nobody_subscribes_to_is_left_unassigned() {
        let subscriptions = BTreeMap::from([("only-events".to_owned(), vec!["events".to_owned()])]);
        let lag = BTreeMap::from([
            (("events".to_owned(), 0), 5),
            (("audit".to_owned(), 0), 500),
        ]);
        let assignment = balance_partitions_by_lag(&subscriptions, &lag);
        assert_eq!(assignment["only-events"]["events"], vec![0]);
        assert!(
            !assignment["only-events"].contains_key("audit"),
            "a member must never be handed a topic it did not subscribe to, however heavy it is"
        );
    }

    #[test]
    fn balancing_is_deterministic_for_repeated_canonical_evaluation() {
        // Re-evaluating the same SyncGroup must reach byte-identical assignments, so ties resolve
        // on the key rather than on map iteration order.
        let subscriptions = BTreeMap::from([
            ("a".to_owned(), vec!["events".to_owned()]),
            ("b".to_owned(), vec!["events".to_owned()]),
        ]);
        let lag = BTreeMap::from([
            (("events".to_owned(), 0), 100),
            (("events".to_owned(), 1), 100),
            (("events".to_owned(), 2), 100),
            (("events".to_owned(), 3), 100),
        ]);
        let first = balance_partitions_by_lag(&subscriptions, &lag);
        for _ in 0..16 {
            assert_eq!(balance_partitions_by_lag(&subscriptions, &lag), first);
        }
        // Equal weights must also split evenly rather than piling onto one member.
        assert_eq!(first["a"]["events"].len(), 2);
        assert_eq!(first["b"]["events"].len(), 2);
    }

    #[test]
    fn an_encoded_assignment_is_readable_as_a_kafka_consumer_assignment() {
        let assignment = BTreeMap::from([
            ("events".to_owned(), vec![0, 3, 7]),
            ("audit".to_owned(), vec![1]),
        ]);
        let decoded = decode_assignment(&encode_partition_assignment(&assignment));
        assert_eq!(decoded, assignment);
        assert_eq!(
            decode_assignment(&encode_partition_assignment(&BTreeMap::new())),
            BTreeMap::new()
        );
    }

    #[test]
    fn topic_match_follows_amqp_wildcard_semantics() {
        assert!(topic_match("a.b.c", "a.b.c"));
        assert!(!topic_match("a.b.c", "a.b.d"));
        assert!(!topic_match("a.b.c", "a.b"));
        assert!(!topic_match("a.b", "a.b.c"));
        // `*` consumes exactly one segment.
        assert!(topic_match("a.*.c", "a.b.c"));
        assert!(!topic_match("a.*.c", "a.b.b.c"));
        assert!(topic_match("*", "a"));
        assert!(!topic_match("*", "a.b"));
        // `#` consumes zero or more segments.
        assert!(topic_match("a.#.c", "a.c"));
        assert!(topic_match("a.#.c", "a.b.c"));
        assert!(topic_match("a.#.c", "a.b.b.b.c"));
        assert!(topic_match("a.#", "a"));
        assert!(!topic_match("a.#", "b.a"));
        assert!(topic_match("#", "a.b.c"));
        assert!(topic_match("#", ""));
        assert!(topic_match("#.#", "a.b"));
        assert!(topic_match("#.b.#", "a.b.c"));
        assert!(!topic_match("#.z", "a.a.a"));
        // Literal segments still have to match exactly, wildcards included.
        assert!(topic_match("a.*.#", "a.b"));
        assert!(!topic_match("a.*.#", "a"));
    }

    #[test]
    fn topic_match_is_bounded_for_adversarial_wildcard_patterns() {
        // The previous recursive matcher retried every split point for each `#`, so this input
        // cost `C(128 + 127, 127)` attempts and never returned — while holding the apply mutex.
        // Both operands sit exactly at the 255-byte limit that `valid_name`/`valid_routing_key`
        // already admit, so this is reachable input, not a synthetic worst case.
        let binding = vec!["#"; 127]
            .into_iter()
            .chain(std::iter::once("z"))
            .collect::<Vec<_>>()
            .join(".");
        let routing = vec!["a"; 128].join(".");
        assert_eq!(binding.len(), 255);
        assert_eq!(routing.len(), 255);
        let started = std::time::Instant::now();
        assert!(!topic_match(&binding, &routing));
        assert!(
            started.elapsed() < std::time::Duration::from_secs(1),
            "topic matching must stay bounded for adversarial wildcard bindings"
        );
    }

    #[test]
    fn fanout_binding_accepts_empty_routing_key() -> Result<()> {
        let project = ProjectId::random();
        let mut broker = BrokerStateMachine::default();
        let directory = tempfile::tempdir()?;
        let segments = SegmentStore::open(directory.path(), 1024 * 1024)?;
        broker.apply(
            BrokerCommand::CreateExchange {
                project,
                name: "events".to_owned(),
                kind: AmqpExchangeKind::Fanout,
                durable: true,
                passive: false,
            },
            &segments,
        )?;
        broker.apply(
            BrokerCommand::CreateQueue {
                project,
                name: "workers".to_owned(),
                kind: QueueKind::Classic,
                durable: true,
                passive: false,
                dead_letter_exchange: None,
                dead_letter_routing_key: None,
                retention: RetentionPolicy {
                    max_age_ms: None,
                    max_bytes: None,
                },
                exclusive_owner: None,
                auto_delete: false,
            },
            &segments,
        )?;
        let reply = broker.apply(
            BrokerCommand::BindQueue {
                project,
                exchange: "events".to_owned(),
                queue: "workers".to_owned(),
                routing_key: String::new(),
            },
            &segments,
        )?;
        assert!(matches!(reply, BrokerReply::Declared));
        Ok(())
    }

    #[test]
    fn dropping_project_removes_only_its_broker_state() -> Result<()> {
        let removed = ProjectId::random();
        let retained = ProjectId::random();
        let directory = tempfile::tempdir()?;
        let segments = SegmentStore::open(directory.path(), 1024 * 1024)?;
        let mut broker = BrokerStateMachine::default();
        for (project, name) in [(removed, "removed"), (retained, "retained")] {
            broker.apply(
                BrokerCommand::CreateTopic {
                    project,
                    name: name.to_owned(),
                    partitions: 2,
                    retention: RetentionPolicy {
                        max_age_ms: None,
                        max_bytes: None,
                    },
                },
                &segments,
            )?;
        }
        assert!(!broker.project_is_empty(removed));
        assert!(!broker.project_is_empty(retained));
        broker.drop_project(removed)?;
        assert!(broker.project_is_empty(removed));
        assert!(!broker.project_is_empty(retained));
        assert_eq!(
            broker.list_offset(retained, "retained", 0, -2)?,
            Some((0, -1))
        );
        broker.validate_state()
    }

    #[test]
    fn clear_topic_preserves_partitions_and_monotonic_offsets() -> Result<()> {
        let project = ProjectId::random();
        let directory = tempfile::tempdir()?;
        let segments = SegmentStore::open(directory.path(), 2 * 1024 * 1024)?;
        let mut broker = BrokerStateMachine::default();
        broker.apply(
            BrokerCommand::CreateTopic {
                project,
                name: "events".to_owned(),
                partitions: 1,
                retention: RetentionPolicy {
                    max_age_ms: None,
                    max_bytes: None,
                },
            },
            &segments,
        )?;
        broker.apply(
            BrokerCommand::SetTopicRetention {
                project,
                name: "events".to_owned(),
                retention: RetentionPolicy {
                    max_age_ms: Some(7 * 86_400_000),
                    max_bytes: None,
                },
            },
            &segments,
        )?;
        broker.apply(
            BrokerCommand::PublishKafkaBatch {
                project,
                topic: "events".to_owned(),
                partition: 0,
                resolved_time_ms: 11,
                records: vec![KafkaBatchRecord {
                    create_time_ms: Some(11),
                    key: None,
                    headers: BTreeMap::new(),
                    payload: b"before-clear".to_vec(),
                    value_is_null: false,
                }],
            },
            &segments,
        )?;
        let reply = broker.apply(
            BrokerCommand::ClearTopic {
                project,
                name: "events".to_owned(),
            },
            &segments,
        )?;
        assert!(matches!(
            reply,
            BrokerReply::MessagesDiscarded { message_count: 1 }
        ));
        assert_eq!(
            broker.topic_metrics(project),
            vec![TopicMetrics {
                name: "events".to_owned(),
                partition: 0,
                base_offset: 1,
                next_offset: 1,
                record_count: 0,
                retained_bytes: 0,
                retention_ms: Some(7 * 86_400_000),
            }]
        );
        let published = broker.apply(
            BrokerCommand::PublishKafkaBatch {
                project,
                topic: "events".to_owned(),
                partition: 0,
                resolved_time_ms: 12,
                records: vec![KafkaBatchRecord {
                    create_time_ms: Some(12),
                    key: None,
                    headers: BTreeMap::new(),
                    payload: b"after-clear".to_vec(),
                    value_is_null: false,
                }],
            },
            &segments,
        )?;
        assert!(matches!(
            published,
            BrokerReply::KafkaBatchPublished {
                first_offset: 1,
                record_count: 1
            }
        ));
        Ok(())
    }

    #[test]
    fn payload_bytes_live_in_immutable_segments_and_reopen_by_message_id() -> Result<()> {
        let project = ProjectId::random();
        let directory = tempfile::tempdir()?;
        let segments = SegmentStore::open(directory.path(), 2 * 1024 * 1024)?;
        let mut broker = BrokerStateMachine::default();
        broker.apply(
            BrokerCommand::CreateTopic {
                project,
                name: "events".to_owned(),
                partitions: 1,
                retention: RetentionPolicy {
                    max_age_ms: None,
                    max_bytes: None,
                },
            },
            &segments,
        )?;
        let payload = vec![0x5a; 256 * 1024];
        let published = broker.apply(
            BrokerCommand::PublishKafkaBatch {
                project,
                topic: "events".to_owned(),
                partition: 0,
                resolved_time_ms: 11,
                records: vec![KafkaBatchRecord {
                    create_time_ms: Some(7),
                    key: Some(b"key".to_vec()),
                    headers: BTreeMap::new(),
                    payload: payload.clone(),
                    value_is_null: false,
                }],
            },
            &segments,
        )?;
        assert!(matches!(
            published,
            BrokerReply::KafkaBatchPublished {
                first_offset: 0,
                record_count: 1
            }
        ));
        assert_eq!(broker.payload_segments().len(), 1);

        let encoded =
            postcard::to_stdvec(&broker).map_err(|error| Error::invalid_data(error.to_string()))?;
        assert!(encoded.len() < payload.len() / 8);
        let reopened: BrokerStateMachine = postcard::from_bytes(&encoded)
            .map_err(|error| Error::new(crate::ErrorCode::CorruptStorage, error.to_string()))?;
        reopened.validate_state()?;
        reopened.validate_payload_segments(&segments)?;
        let fetched = reopened.fetch_partition(project, "events", 0, 0, 512 * 1024, &segments)?;
        assert_eq!(fetched.len(), 1);
        assert_eq!(fetched[0].1.payload.as_ref(), payload.as_slice());
        assert_eq!(fetched[0].1.resolved_time_ms, 11);
        Ok(())
    }

    #[test]
    fn kafka_partition_batch_is_one_atomic_offset_allocation() -> Result<()> {
        let project = ProjectId::random();
        let directory = tempfile::tempdir()?;
        let segments = SegmentStore::open(directory.path(), 4 * 1024 * 1024)?;
        let mut broker = BrokerStateMachine::default();
        broker.apply(
            BrokerCommand::CreateTopic {
                project,
                name: "events".to_owned(),
                partitions: 1,
                retention: RetentionPolicy {
                    max_age_ms: None,
                    max_bytes: None,
                },
            },
            &segments,
        )?;
        let records = [
            b"first".as_slice(),
            b"second".as_slice(),
            b"third".as_slice(),
        ]
        .into_iter()
        .map(|payload| KafkaBatchRecord {
            create_time_ms: Some(7),
            key: None,
            headers: BTreeMap::new(),
            payload: payload.to_vec(),
            value_is_null: false,
        })
        .collect();
        let reply = broker.apply(
            BrokerCommand::PublishKafkaBatch {
                project,
                topic: "events".to_owned(),
                partition: 0,
                resolved_time_ms: 9,
                records,
            },
            &segments,
        )?;
        assert!(matches!(
            reply,
            BrokerReply::KafkaBatchPublished {
                first_offset: 0,
                record_count: 3
            }
        ));
        let fetched = broker.fetch_partition(project, "events", 0, 0, 1024, &segments)?;
        assert_eq!(
            fetched
                .iter()
                .map(|(offset, _)| *offset)
                .collect::<Vec<_>>(),
            vec![0, 1, 2]
        );
        assert_eq!(fetched[1].1.payload.as_ref(), b"second");
        assert!(matches!(
            fetched[1].1.ingress,
            IngressMetadata::Kafka {
                create_time_ms: Some(7),
                key: None,
                ref headers,
                value_is_null: false,
            } if headers.is_empty()
        ));
        let descriptors = broker.payload_segments();
        assert_eq!(descriptors.len(), 1);
        let segment_records = segments.read(&descriptors[0])?;
        assert_eq!(segment_records.len(), 3);
        assert!(
            segment_records
                .iter()
                .all(|record| record.kind == BROKER_RAW_KAFKA_PAYLOAD_SEGMENT_KIND)
        );
        let encoded =
            postcard::to_stdvec(&broker).map_err(|error| Error::invalid_data(error.to_string()))?;
        assert_eq!(
            encoded
                .windows(descriptors[0].file_name.len())
                .filter(|window| *window == descriptors[0].file_name.as_bytes())
                .count(),
            1,
            "a batched descriptor must be canonicalized once, not copied per message"
        );

        let before =
            postcard::to_stdvec(&broker).map_err(|error| Error::invalid_data(error.to_string()))?;
        let failed = broker.apply(
            BrokerCommand::PublishKafkaBatch {
                project,
                topic: "events".to_owned(),
                partition: 0,
                resolved_time_ms: 10,
                records: Vec::new(),
            },
            &segments,
        );
        assert!(failed.is_err());
        let after =
            postcard::to_stdvec(&broker).map_err(|error| Error::invalid_data(error.to_string()))?;
        assert_eq!(before, after);
        Ok(())
    }

    #[test]
    fn kafka_duplicate_payloads_share_an_extent_without_merging_logical_records() -> Result<()> {
        let project = ProjectId::random();
        let directory = tempfile::tempdir()?;
        let segments = SegmentStore::open(directory.path(), 4 * 1024 * 1024)?;
        let mut broker = BrokerStateMachine::default();
        broker.apply(
            BrokerCommand::CreateTopic {
                project,
                name: "events".to_owned(),
                partitions: 1,
                retention: RetentionPolicy {
                    max_age_ms: None,
                    max_bytes: None,
                },
            },
            &segments,
        )?;
        broker.apply(
            BrokerCommand::PublishKafkaBatch {
                project,
                topic: "events".to_owned(),
                partition: 0,
                resolved_time_ms: 1,
                records: vec![KafkaBatchRecord {
                    create_time_ms: Some(1),
                    key: Some(b"dirty".to_vec()),
                    headers: BTreeMap::new(),
                    payload: vec![0xaa; 2 * 1024 * 1024],
                    value_is_null: false,
                }],
            },
            &segments,
        )?;
        broker.apply(
            BrokerCommand::PublishKafkaBatch {
                project,
                topic: "events".to_owned(),
                partition: 0,
                resolved_time_ms: 2,
                records: (0..400)
                    .map(|index| KafkaBatchRecord {
                        create_time_ms: Some(10 + index),
                        key: None,
                        headers: BTreeMap::new(),
                        payload: vec![0x5a; 64],
                        value_is_null: false,
                    })
                    .collect(),
            },
            &segments,
        )?;

        let duplicate_extents = broker
            .payloads
            .iter()
            .skip(1)
            .map(|(_, stored)| (stored.segment, stored.location))
            .collect::<Vec<_>>();
        assert_eq!(duplicate_extents.len(), 400);
        assert!(
            duplicate_extents
                .iter()
                .all(|extent| *extent == duplicate_extents[0])
        );
        assert_eq!(broker.payloads.len(), 401);
        assert_eq!(
            broker
                .streams
                .iter()
                .next()
                .map(|(_, stream)| stream.records.len()),
            Some(401)
        );
        broker.validate_payload_segments(&segments)?;
        let fetched = broker.fetch_partition(project, "events", 0, 1, 1024 * 1024, &segments)?;
        assert_eq!(fetched.len(), 400);
        assert!(
            fetched
                .iter()
                .all(|(_, payload)| payload.payload.as_ref() == [0x5a; 64])
        );
        assert!(matches!(
            fetched.last().map(|(_, payload)| &payload.ingress),
            Some(IngressMetadata::Kafka {
                create_time_ms: Some(409),
                ..
            })
        ));
        Ok(())
    }

    #[test]
    fn kafka_batch_prestage_publishes_once_and_apply_reuses_the_segment() -> Result<()> {
        let project = ProjectId::random();
        let directory = tempfile::tempdir()?;
        let segments = SegmentStore::open(directory.path(), 4 * 1024 * 1024)?;
        let mut broker = BrokerStateMachine::default();
        broker.apply(
            BrokerCommand::CreateTopic {
                project,
                name: "events".to_owned(),
                partitions: 1,
                retention: RetentionPolicy {
                    max_age_ms: None,
                    max_bytes: None,
                },
            },
            &segments,
        )?;
        let command = BrokerCommand::PublishKafkaBatch {
            project,
            topic: "events".to_owned(),
            partition: 0,
            resolved_time_ms: 17,
            records: (0..32)
                .map(|value| KafkaBatchRecord {
                    create_time_ms: Some(17),
                    key: None,
                    headers: BTreeMap::new(),
                    payload: vec![value; 1_024],
                    value_is_null: false,
                })
                .collect(),
        };
        let _ = broker.prestage_command_payload(&command, &segments)?;
        let before = segments.immutable_file_names()?;
        assert_eq!(before.len(), 1);
        broker.apply(command, &segments)?;
        assert_eq!(segments.immutable_file_names()?, before);
        assert_eq!(broker.payload_segments().len(), 1);
        assert_eq!(segments.read(&broker.payload_segments()[0])?.len(), 32);
        Ok(())
    }

    #[test]
    fn amqp_batch_delta_writes_one_segment_and_reuses_prestaged_bytes() -> Result<()> {
        let project = ProjectId::random();
        let directory = tempfile::tempdir()?;
        let segments = SegmentStore::open(directory.path(), 4 * 1024 * 1024)?;
        let mut broker = BrokerStateMachine::default();
        broker.apply(
            BrokerCommand::CreateExchange {
                project,
                name: "events".to_owned(),
                kind: AmqpExchangeKind::Direct,
                durable: true,
                passive: false,
            },
            &segments,
        )?;
        broker.apply(
            BrokerCommand::CreateQueue {
                project,
                name: "events.queue".to_owned(),
                kind: QueueKind::Classic,
                durable: true,
                passive: false,
                dead_letter_exchange: None,
                dead_letter_routing_key: None,
                retention: RetentionPolicy {
                    max_age_ms: None,
                    max_bytes: None,
                },
                exclusive_owner: None,
                auto_delete: false,
            },
            &segments,
        )?;
        broker.apply(
            BrokerCommand::BindQueue {
                project,
                exchange: "events".to_owned(),
                queue: "events.queue".to_owned(),
                routing_key: "created".to_owned(),
            },
            &segments,
        )?;
        let command = BrokerCommand::PublishAmqpBatch {
            project,
            resolved_time_ms: 23,
            records: (0..256)
                .map(|value| AmqpBatchRecord {
                    exchange: "events".to_owned(),
                    routing_key: "created".to_owned(),
                    mandatory: false,
                    properties: BTreeMap::new(),
                    headers: BTreeMap::new(),
                    payload: vec![value as u8; 64],
                })
                .collect(),
        };
        let _ = broker.prestage_command_payload(&command, &segments)?;
        let before = segments.immutable_file_names()?;
        assert_eq!(before.len(), 1);
        let reply = broker.apply(command, &segments)?;
        let BrokerReply::AmqpBatchPublished { offsets } = reply else {
            return Err(Error::internal("AMQP batch returned the wrong reply"));
        };
        assert_eq!(offsets.len(), 256);
        assert!(offsets.iter().all(|record| record.len() == 1));
        assert_eq!(segments.immutable_file_names()?, before);
        assert_eq!(broker.payload_segments().len(), 1);
        assert_eq!(segments.read(&broker.payload_segments()[0])?.len(), 256);
        assert_eq!(
            broker
                .queue_info(project, "events.queue")
                .map(|info| info.message_count),
            Some(256)
        );
        Ok(())
    }

    #[test]
    fn broker_clone_shares_canonical_families_and_detaches_only_mutated_roots() -> Result<()> {
        let project = ProjectId::random();
        let directory = tempfile::tempdir()?;
        let segments = SegmentStore::open(directory.path(), 1024 * 1024)?;
        let mut broker = BrokerStateMachine::default();
        broker.apply(
            BrokerCommand::CreateTopic {
                project,
                name: "events".to_owned(),
                partitions: 1,
                retention: RetentionPolicy {
                    max_age_ms: None,
                    max_bytes: None,
                },
            },
            &segments,
        )?;
        let published = broker.clone();
        assert!(published.payloads.shared_with(&broker.payloads));
        assert!(published.streams.shared_with(&broker.streams));
        assert!(published.topics.shared_with(&broker.topics));
        assert!(published.groups.shared_with(&broker.groups));

        broker.apply(
            BrokerCommand::PublishKafkaBatch {
                project,
                topic: "events".to_owned(),
                partition: 0,
                resolved_time_ms: 1,
                records: vec![KafkaBatchRecord {
                    create_time_ms: None,
                    key: None,
                    headers: BTreeMap::new(),
                    payload: b"value".to_vec(),
                    value_is_null: false,
                }],
            },
            &segments,
        )?;
        assert!(!published.payloads.shared_with(&broker.payloads));
        assert!(!published.streams.shared_with(&broker.streams));
        assert!(published.topics.shared_with(&broker.topics));
        assert!(published.groups.shared_with(&broker.groups));
        assert_eq!(
            published.list_offset(project, "events", 0, -1)?,
            Some((0, -1))
        );
        assert_eq!(broker.list_offset(project, "events", 0, -1)?, Some((1, -1)));

        let before_maintenance = broker.clone();
        broker.apply(
            BrokerCommand::Retain {
                project,
                resolved_time_ms: 2,
            },
            &segments,
        )?;
        assert!(before_maintenance.payloads.shared_with(&broker.payloads));
        assert!(before_maintenance.streams.shared_with(&broker.streams));
        assert!(before_maintenance.groups.shared_with(&broker.groups));
        Ok(())
    }

    #[test]
    fn single_delivery_and_ack_detach_only_the_target_reference_page() -> Result<()> {
        let project = ProjectId::random();
        let directory = tempfile::tempdir()?;
        let segments = SegmentStore::open(directory.path(), 2 * 1024 * 1024)?;
        let mut broker = BrokerStateMachine::default();
        broker.apply(
            BrokerCommand::CreateQueue {
                project,
                name: "jobs".to_owned(),
                kind: QueueKind::Classic,
                durable: true,
                passive: false,
                dead_letter_exchange: None,
                dead_letter_routing_key: None,
                retention: RetentionPolicy {
                    max_age_ms: None,
                    max_bytes: None,
                },
                exclusive_owner: None,
                auto_delete: false,
            },
            &segments,
        )?;
        let published = broker.apply(
            BrokerCommand::PublishAmqp {
                project,
                exchange: String::new(),
                routing_key: "jobs".to_owned(),
                mandatory: true,
                resolved_time_ms: 1,
                properties: BTreeMap::new(),
                headers: BTreeMap::new(),
                payload: b"shared".to_vec(),
            },
            &segments,
        )?;
        let BrokerReply::Published { message, .. } = published else {
            return Err(Error::internal("publish returned the wrong broker reply"));
        };
        let stream = broker
            .queues
            .get(&(project, "jobs".to_owned()))
            .ok_or_else(|| Error::internal("queue disappeared"))?
            .stream;
        for _ in 0..50_000 {
            let _ = broker.append_reference(stream, message)?;
        }

        let before_delivery = broker.clone();
        let owner = Uuid::new_v4();
        let BrokerReply::Deliveries(deliveries) = broker.apply(
            BrokerCommand::DeliverQueue {
                project,
                queue: "jobs".to_owned(),
                offset: StreamOffset::First,
                maximum: 1,
                owner,
                consumer: 7,
                automatic_ack: false,
                resolved_time_ms: 2,
            },
            &segments,
        )?
        else {
            return Err(Error::internal("delivery returned the wrong broker reply"));
        };
        let before_records = &before_delivery
            .streams
            .get(&stream)
            .ok_or_else(|| Error::internal("published stream disappeared"))?
            .records;
        let delivered_records = &broker
            .streams
            .get(&stream)
            .ok_or_else(|| Error::internal("delivered stream disappeared"))?
            .records;
        assert!(delivered_records.detached_page_bytes_from(before_records) <= 16 * 1024);

        let before_ack = broker.clone();
        broker.apply(
            BrokerCommand::Ack {
                project,
                queue: "jobs".to_owned(),
                owner,
                consumer: 7,
                delivery_tag: deliveries[0].delivery_tag,
                multiple: false,
            },
            &segments,
        )?;
        let before_ack_records = &before_ack
            .streams
            .get(&stream)
            .ok_or_else(|| Error::internal("pre-ack stream disappeared"))?
            .records;
        let acked_records = &broker
            .streams
            .get(&stream)
            .ok_or_else(|| Error::internal("acked stream disappeared"))?
            .records;
        assert!(acked_records.detached_page_bytes_from(before_ack_records) <= 16 * 1024);
        Ok(())
    }

    #[test]
    fn kafka_group_heartbeats_and_session_expiry_are_canonical() -> Result<()> {
        let project = ProjectId::random();
        let directory = tempfile::tempdir()?;
        let segments = SegmentStore::open(directory.path(), 1024 * 1024)?;
        let mut broker = BrokerStateMachine::default();
        let protocols = BTreeMap::from([("range".to_owned(), b"metadata".to_vec())]);
        let first = broker.apply(
            BrokerCommand::JoinGroup {
                project,
                group: "workers".to_owned(),
                member: "one".to_owned(),
                session_timeout_ms: 1_000,
                resolved_time_ms: 1_000,
                protocols: protocols.clone(),
            },
            &segments,
        )?;
        let BrokerReply::Group { generation, .. } = first else {
            return Err(Error::internal("join returned the wrong broker reply"));
        };
        broker.apply(
            BrokerCommand::JoinGroup {
                project,
                group: "workers".to_owned(),
                member: "two".to_owned(),
                session_timeout_ms: 1_000,
                resolved_time_ms: 1_500,
                protocols,
            },
            &segments,
        )?;
        let current_generation = generation + 1;
        broker.apply(
            BrokerCommand::HeartbeatGroup {
                project,
                group: "workers".to_owned(),
                generation: current_generation,
                member: "one".to_owned(),
                resolved_time_ms: 1_999,
            },
            &segments,
        )?;
        broker.apply(
            BrokerCommand::Retain {
                project,
                resolved_time_ms: 2_600,
            },
            &segments,
        )?;
        assert!(!broker.group_member(project, "workers", current_generation, "one"));
        assert!(broker.group_member(project, "workers", current_generation + 1, "one"));
        assert!(!broker.group_member(project, "workers", current_generation + 1, "two"));
        broker.validate_state()
    }

    #[test]
    fn amqp_requeue_and_dead_letter_move_only_delivery_references() -> Result<()> {
        let project = ProjectId::random();
        let directory = tempfile::tempdir()?;
        let segments = SegmentStore::open(directory.path(), 2 * 1024 * 1024)?;
        let mut broker = BrokerStateMachine::default();
        let owner = Uuid::new_v4();
        broker.apply(
            BrokerCommand::CreateExchange {
                project,
                name: "dead".to_owned(),
                kind: AmqpExchangeKind::Direct,
                durable: true,
                passive: false,
            },
            &segments,
        )?;
        for (name, dead_letter_exchange, dead_letter_routing_key) in [
            ("work", Some("dead".to_owned()), Some("rejected".to_owned())),
            ("failed", None, None),
        ] {
            broker.apply(
                BrokerCommand::CreateQueue {
                    project,
                    name: name.to_owned(),
                    kind: QueueKind::Classic,
                    durable: true,
                    passive: false,
                    dead_letter_exchange,
                    dead_letter_routing_key,
                    retention: RetentionPolicy {
                        max_age_ms: None,
                        max_bytes: None,
                    },
                    exclusive_owner: None,
                    auto_delete: false,
                },
                &segments,
            )?;
        }
        broker.apply(
            BrokerCommand::BindQueue {
                project,
                exchange: "dead".to_owned(),
                queue: "failed".to_owned(),
                routing_key: "rejected".to_owned(),
            },
            &segments,
        )?;
        broker.apply(
            BrokerCommand::PublishAmqp {
                project,
                exchange: String::new(),
                routing_key: "work".to_owned(),
                mandatory: true,
                resolved_time_ms: 1,
                properties: BTreeMap::new(),
                headers: BTreeMap::new(),
                payload: b"job".to_vec(),
            },
            &segments,
        )?;
        let BrokerReply::Deliveries(first) = broker.apply(
            BrokerCommand::DeliverQueue {
                project,
                queue: "work".to_owned(),
                offset: StreamOffset::First,
                maximum: 1,
                owner,
                consumer: 7,
                automatic_ack: false,
                resolved_time_ms: 2,
            },
            &segments,
        )?
        else {
            return Err(Error::internal("classic delivery returned the wrong reply"));
        };
        broker.apply(
            BrokerCommand::Nack {
                project,
                queue: "work".to_owned(),
                owner,
                consumer: 7,
                delivery_tag: first[0].delivery_tag,
                multiple: false,
                requeue: true,
            },
            &segments,
        )?;
        let BrokerReply::Deliveries(redelivery) = broker.apply(
            BrokerCommand::DeliverQueue {
                project,
                queue: "work".to_owned(),
                offset: StreamOffset::First,
                maximum: 1,
                owner,
                consumer: 8,
                automatic_ack: false,
                resolved_time_ms: 3,
            },
            &segments,
        )?
        else {
            return Err(Error::internal(
                "classic redelivery returned the wrong reply",
            ));
        };
        assert!(redelivery[0].redelivered);
        broker.apply(
            BrokerCommand::Nack {
                project,
                queue: "work".to_owned(),
                owner,
                consumer: 8,
                delivery_tag: redelivery[0].delivery_tag,
                multiple: false,
                requeue: false,
            },
            &segments,
        )?;
        let BrokerReply::Deliveries(dead) = broker.apply(
            BrokerCommand::DeliverQueue {
                project,
                queue: "failed".to_owned(),
                offset: StreamOffset::First,
                maximum: 1,
                owner,
                consumer: 9,
                automatic_ack: true,
                resolved_time_ms: 4,
            },
            &segments,
        )?
        else {
            return Err(Error::internal(
                "dead-letter delivery returned the wrong reply",
            ));
        };
        assert_eq!(dead.len(), 1);
        assert_eq!(dead[0].death_count, 1);
        assert_eq!(dead[0].payload.id, redelivery[0].payload.id);
        assert_eq!(dead[0].payload.payload.as_ref(), b"job");
        broker.validate_state()
    }

    #[test]
    fn amqp_delivery_owner_lease_fences_settlement_and_requeues_after_expiry() -> Result<()> {
        let project = ProjectId::random();
        let directory = tempfile::tempdir()?;
        let segments = SegmentStore::open(directory.path(), 2 * 1024 * 1024)?;
        let mut broker = BrokerStateMachine::default();
        broker.apply(
            BrokerCommand::CreateQueue {
                project,
                name: "jobs".to_owned(),
                kind: QueueKind::Classic,
                durable: true,
                passive: false,
                dead_letter_exchange: None,
                dead_letter_routing_key: None,
                retention: RetentionPolicy {
                    max_age_ms: None,
                    max_bytes: None,
                },
                exclusive_owner: None,
                auto_delete: false,
            },
            &segments,
        )?;
        broker.apply(
            BrokerCommand::PublishAmqp {
                project,
                exchange: String::new(),
                routing_key: "jobs".to_owned(),
                mandatory: true,
                resolved_time_ms: 1,
                properties: BTreeMap::new(),
                headers: BTreeMap::new(),
                payload: b"lease".to_vec(),
            },
            &segments,
        )?;

        let owner = Uuid::new_v4();
        let intruder = Uuid::new_v4();
        let BrokerReply::Deliveries(first) = broker.apply(
            BrokerCommand::DeliverQueue {
                project,
                queue: "jobs".to_owned(),
                offset: StreamOffset::First,
                maximum: 1,
                owner,
                consumer: 7,
                automatic_ack: false,
                resolved_time_ms: 10,
            },
            &segments,
        )?
        else {
            return Err(Error::internal("classic delivery returned the wrong reply"));
        };
        let tag = first[0].delivery_tag;
        let error = broker
            .validate_command(&BrokerCommand::Ack {
                project,
                queue: "jobs".to_owned(),
                owner: intruder,
                consumer: 7,
                delivery_tag: tag,
                multiple: false,
            })
            .expect_err("another connection must not settle the delivery");
        assert_eq!(error.code, crate::ErrorCode::ProtocolViolation);

        broker.apply(
            BrokerCommand::RenewDeliveryLease {
                project,
                owner,
                resolved_time_ms: 100,
            },
            &segments,
        )?;
        broker.apply(
            BrokerCommand::Retain {
                project,
                resolved_time_ms: 100 + DELIVERY_LEASE_MILLIS - 1,
            },
            &segments,
        )?;
        let BrokerReply::Deliveries(blocked) = broker.apply(
            BrokerCommand::DeliverQueue {
                project,
                queue: "jobs".to_owned(),
                offset: StreamOffset::First,
                maximum: 1,
                owner: intruder,
                consumer: 8,
                automatic_ack: false,
                resolved_time_ms: 100 + DELIVERY_LEASE_MILLIS - 1,
            },
            &segments,
        )?
        else {
            return Err(Error::internal("classic delivery returned the wrong reply"));
        };
        assert!(blocked.is_empty());

        broker.apply(
            BrokerCommand::Retain {
                project,
                resolved_time_ms: 100 + DELIVERY_LEASE_MILLIS,
            },
            &segments,
        )?;
        let BrokerReply::Deliveries(redelivery) = broker.apply(
            BrokerCommand::DeliverQueue {
                project,
                queue: "jobs".to_owned(),
                offset: StreamOffset::First,
                maximum: 1,
                owner: intruder,
                consumer: 8,
                automatic_ack: false,
                resolved_time_ms: 101 + DELIVERY_LEASE_MILLIS,
            },
            &segments,
        )?
        else {
            return Err(Error::internal("classic delivery returned the wrong reply"));
        };
        assert_eq!(redelivery.len(), 1);
        assert!(redelivery[0].redelivered);

        broker.apply(
            BrokerCommand::ReleaseDeliveryLease {
                project,
                owner: intruder,
            },
            &segments,
        )?;
        let BrokerReply::Deliveries(final_delivery) = broker.apply(
            BrokerCommand::DeliverQueue {
                project,
                queue: "jobs".to_owned(),
                offset: StreamOffset::First,
                maximum: 1,
                owner,
                consumer: 9,
                automatic_ack: true,
                resolved_time_ms: i64::MAX,
            },
            &segments,
        )?
        else {
            return Err(Error::internal("classic delivery returned the wrong reply"));
        };
        assert_eq!(final_delivery.len(), 1);
        assert!(final_delivery[0].redelivered);
        broker.validate_state()
    }

    #[test]
    fn retention_cannot_remove_an_unacknowledged_classic_delivery() -> Result<()> {
        let project = ProjectId::random();
        let directory = tempfile::tempdir()?;
        let segments = SegmentStore::open(directory.path(), 2 * 1024 * 1024)?;
        let mut broker = BrokerStateMachine::default();
        broker.apply(
            BrokerCommand::CreateQueue {
                project,
                name: "jobs".to_owned(),
                kind: QueueKind::Classic,
                durable: true,
                passive: false,
                dead_letter_exchange: None,
                dead_letter_routing_key: None,
                retention: RetentionPolicy {
                    max_age_ms: None,
                    max_bytes: Some(1),
                },
                exclusive_owner: None,
                auto_delete: false,
            },
            &segments,
        )?;
        broker.apply(
            BrokerCommand::PublishAmqp {
                project,
                exchange: String::new(),
                routing_key: "jobs".to_owned(),
                mandatory: true,
                resolved_time_ms: 1,
                properties: BTreeMap::new(),
                headers: BTreeMap::new(),
                payload: b"leased".to_vec(),
            },
            &segments,
        )?;
        let owner = Uuid::new_v4();
        let BrokerReply::Deliveries(deliveries) = broker.apply(
            BrokerCommand::DeliverQueue {
                project,
                queue: "jobs".to_owned(),
                offset: StreamOffset::First,
                maximum: 1,
                owner,
                consumer: 7,
                automatic_ack: false,
                resolved_time_ms: 10,
            },
            &segments,
        )?
        else {
            return Err(Error::internal("classic delivery returned the wrong reply"));
        };
        assert_eq!(deliveries.len(), 1);

        let retained = broker.apply(
            BrokerCommand::Retain {
                project,
                resolved_time_ms: 10 + DELIVERY_LEASE_MILLIS - 1,
            },
            &segments,
        )?;
        assert!(matches!(
            retained,
            BrokerReply::Retained {
                removed_references: 0,
                removed_payloads: 0
            }
        ));
        broker.validate_state()?;

        let expired = broker.apply(
            BrokerCommand::Retain {
                project,
                resolved_time_ms: 10 + DELIVERY_LEASE_MILLIS,
            },
            &segments,
        )?;
        assert!(matches!(
            expired,
            BrokerReply::Retained {
                removed_references: 1,
                removed_payloads: 1
            }
        ));
        broker.validate_state()
    }

    #[test]
    fn retention_rejects_an_age_outside_the_signed_timestamp_domain() {
        let error = RetentionPolicy {
            max_age_ms: Some(i64::MAX as u64 + 1),
            max_bytes: None,
        }
        .validate()
        .expect_err("an age that cannot be compared to i64 timestamps must be rejected");
        assert_eq!(error.code, crate::ErrorCode::InvalidData);
    }

    #[test]
    fn byte_retention_counts_ingress_metadata_and_segment_framing() -> Result<()> {
        let project = ProjectId::random();
        let directory = tempfile::tempdir()?;
        let segments = SegmentStore::open(directory.path(), 2 * 1024 * 1024)?;
        let mut broker = BrokerStateMachine::default();
        broker.apply(
            BrokerCommand::CreateQueue {
                project,
                name: "stream".to_owned(),
                kind: QueueKind::Stream,
                durable: true,
                passive: false,
                dead_letter_exchange: None,
                dead_letter_routing_key: None,
                retention: RetentionPolicy {
                    max_age_ms: None,
                    max_bytes: Some(256),
                },
                exclusive_owner: None,
                auto_delete: false,
            },
            &segments,
        )?;
        broker.apply(
            BrokerCommand::PublishAmqp {
                project,
                exchange: String::new(),
                routing_key: "stream".to_owned(),
                mandatory: true,
                resolved_time_ms: 1,
                properties: BTreeMap::from([("content_type".to_owned(), vec![b'p'; 2_048])]),
                headers: BTreeMap::from([("trace".to_owned(), vec![b'h'; 2_048])]),
                payload: Vec::new(),
            },
            &segments,
        )?;
        let retained = broker.apply(
            BrokerCommand::Retain {
                project,
                resolved_time_ms: 1,
            },
            &segments,
        )?;
        assert!(matches!(
            retained,
            BrokerReply::Retained {
                removed_references: 1,
                removed_payloads: 1
            }
        ));
        broker.validate_state()
    }

    #[test]
    fn delivery_result_budget_counts_protocol_visible_metadata() -> Result<()> {
        let project = ProjectId::random();
        let directory = tempfile::tempdir()?;
        let segments = SegmentStore::open(directory.path(), 32 * 1024 * 1024)?;
        let mut broker = BrokerStateMachine::default();
        broker.apply(
            BrokerCommand::CreateQueue {
                project,
                name: "jobs".to_owned(),
                kind: QueueKind::Classic,
                durable: true,
                passive: false,
                dead_letter_exchange: None,
                dead_letter_routing_key: None,
                retention: RetentionPolicy {
                    max_age_ms: None,
                    max_bytes: None,
                },
                exclusive_owner: None,
                auto_delete: false,
            },
            &segments,
        )?;
        broker.apply(
            BrokerCommand::PublishAmqp {
                project,
                exchange: String::new(),
                routing_key: "jobs".to_owned(),
                mandatory: true,
                resolved_time_ms: 1,
                properties: BTreeMap::from([(
                    "metadata".to_owned(),
                    vec![0_u8; MAX_BROKER_DELIVERY_RESULT_BYTES + 1],
                )]),
                headers: BTreeMap::new(),
                payload: Vec::new(),
            },
            &segments,
        )?;
        let error = broker
            .validate_command(&BrokerCommand::DeliverQueue {
                project,
                queue: "jobs".to_owned(),
                offset: StreamOffset::First,
                maximum: 1,
                owner: Uuid::new_v4(),
                consumer: 1,
                automatic_ack: true,
                resolved_time_ms: 2,
            })
            .expect_err("metadata must count toward the durable delivery result bound");
        assert_eq!(error.code, crate::ErrorCode::ResultBudgetExceeded);
        Ok(())
    }

    #[test]
    fn kafka_fetch_byte_limit_counts_keys_headers_and_record_overhead() -> Result<()> {
        let project = ProjectId::random();
        let directory = tempfile::tempdir()?;
        let segments = SegmentStore::open(directory.path(), 2 * 1024 * 1024)?;
        let mut broker = BrokerStateMachine::default();
        broker.apply(
            BrokerCommand::CreateTopic {
                project,
                name: "events".to_owned(),
                partitions: 1,
                retention: RetentionPolicy {
                    max_age_ms: None,
                    max_bytes: None,
                },
            },
            &segments,
        )?;
        broker.apply(
            BrokerCommand::PublishKafkaBatch {
                project,
                topic: "events".to_owned(),
                partition: 0,
                resolved_time_ms: 1,
                records: (0..2)
                    .map(|_| KafkaBatchRecord {
                        create_time_ms: None,
                        key: Some(vec![b'k'; 1_024]),
                        headers: BTreeMap::from([("header".to_owned(), vec![b'h'; 2_048])]),
                        payload: Vec::new(),
                        value_is_null: false,
                    })
                    .collect(),
            },
            &segments,
        )?;
        let one_record_bytes = usize::try_from(
            broker
                .payloads
                .get(&MessageId(1))
                .ok_or_else(|| Error::internal("published payload is missing"))?
                .retained_bytes,
        )
        .map_err(|_| Error::internal("record length exceeds this platform"))?;
        let fetched =
            broker.fetch_partition(project, "events", 0, 0, one_record_bytes, &segments)?;
        assert_eq!(fetched.len(), 1);
        Ok(())
    }

    #[test]
    fn kafka_list_offsets_uses_effective_record_create_time_and_reports_no_match() -> Result<()> {
        let project = ProjectId::random();
        let directory = tempfile::tempdir()?;
        let segments = SegmentStore::open(directory.path(), 2 * 1024 * 1024)?;
        let mut broker = BrokerStateMachine::default();
        broker.apply(
            BrokerCommand::CreateTopic {
                project,
                name: "events".to_owned(),
                partitions: 1,
                retention: RetentionPolicy {
                    max_age_ms: None,
                    max_bytes: None,
                },
            },
            &segments,
        )?;
        broker.apply(
            BrokerCommand::PublishKafkaBatch {
                project,
                topic: "events".to_owned(),
                partition: 0,
                resolved_time_ms: 1_000,
                records: [100, 300]
                    .into_iter()
                    .map(|create_time_ms| KafkaBatchRecord {
                        create_time_ms: Some(create_time_ms),
                        key: None,
                        headers: BTreeMap::new(),
                        payload: b"value".to_vec(),
                        value_is_null: false,
                    })
                    .collect(),
            },
            &segments,
        )?;
        assert_eq!(
            broker.list_offset(project, "events", 0, -2)?,
            Some((0, 100))
        );
        assert_eq!(
            broker.list_offset(project, "events", 0, 200)?,
            Some((1, 300))
        );
        assert_eq!(broker.list_offset(project, "events", 0, 400)?, None);
        assert_eq!(broker.list_offset(project, "events", 0, -1)?, Some((2, -1)));
        Ok(())
    }

    #[test]
    fn amqp_stream_offsets_and_byte_retention_are_non_destructive() -> Result<()> {
        let project = ProjectId::random();
        let directory = tempfile::tempdir()?;
        let segments = SegmentStore::open(directory.path(), 2 * 1024 * 1024)?;
        let mut broker = BrokerStateMachine::default();
        broker.apply(
            BrokerCommand::CreateQueue {
                project,
                name: "stream".to_owned(),
                kind: QueueKind::Stream,
                durable: true,
                passive: false,
                dead_letter_exchange: None,
                dead_letter_routing_key: None,
                retention: RetentionPolicy {
                    max_age_ms: None,
                    max_bytes: None,
                },
                exclusive_owner: None,
                auto_delete: false,
            },
            &segments,
        )?;
        for (time, payload) in [(10, b"aaa".as_slice()), (20, b"bbb".as_slice())] {
            broker.apply(
                BrokerCommand::PublishAmqp {
                    project,
                    exchange: String::new(),
                    routing_key: "stream".to_owned(),
                    mandatory: true,
                    resolved_time_ms: time,
                    properties: BTreeMap::new(),
                    headers: BTreeMap::new(),
                    payload: payload.to_vec(),
                },
                &segments,
            )?;
        }
        let single_record = broker
            .payloads
            .get(&MessageId(2))
            .ok_or_else(|| Error::internal("published payload is missing"))?;
        let single_record_bytes = broker.accounted_payload_bytes(single_record)?;
        broker
            .queues
            .get_mut(&(project, "stream".to_owned()))
            .ok_or_else(|| Error::internal("stream queue is missing"))?
            .retention
            .max_bytes = Some(single_record_bytes);
        let first = broker.read_stream_queue(
            project,
            "stream",
            StreamOffset::First,
            8,
            1,
            false,
            &segments,
        )?;
        assert_eq!(first.len(), 2);
        let again = broker.read_stream_queue(
            project,
            "stream",
            StreamOffset::First,
            8,
            1,
            false,
            &segments,
        )?;
        assert_eq!(again.len(), 2);
        broker.apply(
            BrokerCommand::Retain {
                project,
                resolved_time_ms: 21,
            },
            &segments,
        )?;
        let retained = broker.read_stream_queue(
            project,
            "stream",
            StreamOffset::First,
            8,
            1,
            false,
            &segments,
        )?;
        assert_eq!(retained.len(), 1);
        assert_eq!(retained[0].offset, 1);
        assert_eq!(retained[0].payload.payload.as_ref(), b"bbb");
        broker.validate_state()
    }
}
