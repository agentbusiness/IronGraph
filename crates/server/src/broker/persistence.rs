//! Incremental durable records for the existing canonical broker cells.
//!
//! Capture is local to the ordered application thread. The observers retain only changed rows
//! and identities of their existing child maps; no broker image or second live store is built.

use std::{
    any::{Any, TypeId},
    cell::RefCell,
    collections::HashMap,
    io::Cursor,
    marker::PhantomData,
    rc::Rc,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use serde::{Serialize, de::DeserializeOwned};

use super::*;
use crate::{ErrorCode, storage::RecordOp};

type Name = (ProjectId, String);

#[derive(Clone)]
enum Kind {
    Payload,
    Segment,
    Topic,
    Exchange,
    Queue,
    Binding,
    Stream,
    Group,
    Lease,
    Consumers,
    References(u64),
    Members(Name),
    Assignments(Name),
    Offsets(Name),
    ConsumerRows(Name),
}

enum Event {
    Row {
        map: usize,
        key: Box<dyn Any>,
        value: Option<Box<dyn Any>>,
        observed: bool,
    },
    Clear(usize),
}

struct Capture {
    token: u64,
    owner: usize,
    maps: HashMap<usize, Kind>,
    events: Vec<Event>,
}

thread_local! {
    static CAPTURE: RefCell<Option<Capture>> = const { RefCell::new(None) };
}
static NEXT_CAPTURE: AtomicU64 = AtomicU64::new(1);

/// Thread-bound RAII scope for one ordered broker mutation. Dropping an unsuccessful capture
/// clears every observer, including when validation or application returns an error.
pub struct BrokerPersistenceSelection {
    token: u64,
    _thread: PhantomData<Rc<()>>,
}

impl Drop for BrokerPersistenceSelection {
    fn drop(&mut self) {
        CAPTURE.with(|slot| {
            let mut slot = slot.borrow_mut();
            if slot
                .as_ref()
                .is_some_and(|capture| capture.token == self.token)
            {
                *slot = None;
            }
        });
    }
}

pub(in crate::broker) fn changed<K: 'static, V: 'static>(
    map: usize,
    key: K,
    value: Option<Arc<V>>,
) {
    CAPTURE.with(|slot| {
        if let Some(capture) = slot.borrow_mut().as_mut() {
            capture.events.push(Event::Row {
                map,
                key: Box::new(key),
                value: value.map(|value| Box::new(value) as Box<dyn Any>),
                observed: false,
            });
        }
    });
}

pub(in crate::broker) fn observe<K: 'static, V: 'static>(map: usize, key: K, value: Arc<V>) {
    // These rows own child maps whose identities are needed to name changed nested entries.
    // Scalar row reads and stream-reference scans never generate persistence observations.
    let value_type = TypeId::of::<V>();
    if value_type != TypeId::of::<StreamState>()
        && value_type != TypeId::of::<GroupState>()
        && value_type != TypeId::of::<CanonicalSet<(Uuid, u64)>>()
    {
        return;
    }
    CAPTURE.with(|slot| {
        if let Some(capture) = slot.borrow_mut().as_mut() {
            capture.events.push(Event::Row {
                map,
                key: Box::new(key),
                value: Some(Box::new(value)),
                observed: true,
            });
        }
    });
}

pub(in crate::broker) fn cleared(map: usize) {
    CAPTURE.with(|slot| {
        if let Some(capture) = slot.borrow_mut().as_mut() {
            capture.events.push(Event::Clear(map));
        }
    });
}

#[derive(Serialize, Deserialize)]
struct StreamMetadata {
    base_offset: u64,
    next_offset: u64,
    retained_bytes: u64,
    head: usize,
    tail: usize,
}

#[derive(Serialize, Deserialize)]
struct GroupMetadata {
    generation: i32,
    leader: String,
    protocol: String,
}

impl BrokerStateMachine {
    pub fn begin_persistence(&self) -> Result<BrokerPersistenceSelection> {
        let token = NEXT_CAPTURE.fetch_add(1, Ordering::Relaxed);
        if token == 0 {
            return Err(Error::internal(
                "broker persistence capture identity exhausted",
            ));
        }
        let maps = HashMap::from([
            (self.payloads.0.identity(), Kind::Payload),
            (self.payload_segments.0.identity(), Kind::Segment),
            (self.topics.identity(), Kind::Topic),
            (self.exchanges.identity(), Kind::Exchange),
            (self.queues.identity(), Kind::Queue),
            (self.bindings.0.identity(), Kind::Binding),
            (self.streams.0.identity(), Kind::Stream),
            (self.groups.identity(), Kind::Group),
            (self.delivery_leases.identity(), Kind::Lease),
            (self.queue_consumers.identity(), Kind::Consumers),
        ]);
        CAPTURE.with(|slot| {
            let mut slot = slot.borrow_mut();
            if slot.is_some() {
                return Err(Error::internal(
                    "broker persistence capture is already active",
                ));
            }
            *slot = Some(Capture {
                token,
                owner: self.topics.identity(),
                maps,
                events: Vec::new(),
            });
            Ok(BrokerPersistenceSelection {
                token,
                _thread: PhantomData,
            })
        })
    }

    pub fn persistence_changes(
        &self,
        selection: &BrokerPersistenceSelection,
    ) -> Result<Vec<RecordOp>> {
        let mut capture = CAPTURE.with(|slot| {
            let mut slot = slot.borrow_mut();
            match slot.as_ref() {
                Some(capture)
                    if capture.token == selection.token
                        && capture.owner == self.topics.identity() =>
                {
                    slot.take()
                        .ok_or_else(|| Error::internal("validated broker capture disappeared"))
                }
                _ => Err(Error::internal(
                    "broker persistence capture belongs to another scope",
                )),
            }
        })?;
        // Discover child identities before converting events: parent edits publish after their
        // nested edits, while both refer to the same existing child maps.
        for event in &capture.events {
            let Event::Row {
                map,
                key,
                value: Some(value),
                ..
            } = event
            else {
                continue;
            };
            match capture.maps.get(map) {
                Some(Kind::Stream) => {
                    let id = typed::<u64>(key.as_ref())?;
                    let stream = row::<StreamState>(value.as_ref())?;
                    capture
                        .maps
                        .insert(stream.records.values.identity(), Kind::References(*id));
                }
                Some(Kind::Group) => {
                    let name = typed::<Name>(key.as_ref())?.clone();
                    let group = row::<GroupState>(value.as_ref())?;
                    capture
                        .maps
                        .insert(group.members.identity(), Kind::Members(name.clone()));
                    capture.maps.insert(
                        group.assignments.identity(),
                        Kind::Assignments(name.clone()),
                    );
                    capture
                        .maps
                        .insert(group.offsets.identity(), Kind::Offsets(name));
                }
                Some(Kind::Consumers) => {
                    let name = typed::<Name>(key.as_ref())?.clone();
                    let consumers = row::<CanonicalSet<(Uuid, u64)>>(value.as_ref())?;
                    capture
                        .maps
                        .insert(consumers.identity(), Kind::ConsumerRows(name));
                }
                _ => {}
            }
        }
        let mut operations = Vec::with_capacity(capture.events.len() + 1);
        for event in capture.events {
            match event {
                Event::Row {
                    map,
                    key,
                    value,
                    observed: false,
                } => {
                    let kind = capture.maps.get(&map).ok_or_else(|| {
                        Error::internal("changed broker collection has no durable owner")
                    })?;
                    record_change(kind, &*key, value.as_deref(), &mut operations)?;
                }
                Event::Clear(map) => {
                    let kind = capture.maps.get(&map).ok_or_else(|| {
                        Error::internal("cleared broker collection has no durable owner")
                    })?;
                    operations.push(RecordOp::DeletePrefix {
                        prefix: collection_prefix(kind)?,
                    });
                }
                Event::Row { observed: true, .. } => {}
            }
        }
        operations.push(RecordOp::Put {
            key: b"b/z".to_vec(),
            value: encode(&(
                self.next_message_id.get(),
                self.next_stream_id.get(),
                self.next_payload_segment_id.get(),
            ))?,
        });
        Ok(operations)
    }

    pub fn restore_persistence_record(&self, key: &[u8], value: &[u8]) -> Result<()> {
        if key == b"b/z" {
            let (message, stream, segment): (u64, u64, u64) = decode(value)?;
            self.next_message_id.set(message);
            self.next_stream_id.set(stream);
            self.next_payload_segment_id.set(segment);
            return Ok(());
        }
        if let Some(suffix) = key.strip_prefix(b"b/t/") {
            self.topics
                .insert(decode::<Name>(suffix)?, decode::<Topic>(value)?);
        } else if let Some(suffix) = key.strip_prefix(b"b/e/") {
            self.exchanges
                .insert(decode::<Name>(suffix)?, decode::<Exchange>(value)?);
        } else if let Some(suffix) = key.strip_prefix(b"b/q/") {
            self.queues
                .insert(decode::<Name>(suffix)?, decode::<Queue>(value)?);
        } else if let Some(suffix) = key.strip_prefix(b"b/b/") {
            self.bindings.0.insert(
                decode::<(ProjectId, String, String, String)>(suffix)?,
                decode::<Binding>(value)?,
            );
        } else if let Some(suffix) = key.strip_prefix(b"b/p/") {
            let id = integer(suffix)?;
            let payload: StoredPayloadRecord = decode(value)?;
            if payload.id.0 != id {
                return Err(corrupt(
                    "broker payload identity differs from its record key",
                ));
            }
            self.payloads.0.insert(id, payload);
        } else if let Some(suffix) = key.strip_prefix(b"b/v/") {
            let id = integer(suffix)?;
            let segment: BrokerPayloadSegment = decode(value)?;
            if segment.id != id {
                return Err(corrupt(
                    "broker segment identity differs from its record key",
                ));
            }
            self.payload_segments.0.insert(id, segment);
        } else if let Some(suffix) = key.strip_prefix(b"b/l/") {
            self.delivery_leases
                .insert(decode::<(ProjectId, Uuid)>(suffix)?, decode::<i64>(value)?);
        } else if let Some(suffix) = key.strip_prefix(b"b/s/") {
            self.restore_stream_record(suffix, value)?;
        } else if let Some(suffix) = key.strip_prefix(b"b/g/") {
            self.restore_group_record(suffix, value)?;
        } else if let Some(suffix) = key.strip_prefix(b"b/c/") {
            let (name, suffix) = take_name(suffix)?;
            let consumers = self.queue_consumers.entry(name).or_default();
            if suffix == b"/m" {
                decode::<()>(value)?;
            } else if let Some(suffix) = suffix.strip_prefix(b"/r/") {
                consumers.insert(decode::<(Uuid, u64)>(suffix)?);
                decode::<()>(value)?;
            } else {
                return Err(corrupt("unknown broker consumer record"));
            }
        } else {
            return Err(corrupt("unknown broker persistence record"));
        }
        Ok(())
    }

    fn restore_stream_record(&self, suffix: &[u8], value: &[u8]) -> Result<()> {
        if suffix.len() < 10 {
            return Err(corrupt("truncated broker stream key"));
        }
        let id = integer(&suffix[..8])?;
        let state = self.streams.0.entry(id).or_default();
        if &suffix[8..] == b"/m" {
            let meta: StreamMetadata = decode(value)?;
            if meta.head > meta.tail || meta.base_offset > meta.next_offset {
                return Err(corrupt("broker stream metadata has reversed bounds"));
            }
            state.base_offset.set(meta.base_offset);
            state.next_offset.set(meta.next_offset);
            state.retained_bytes.set(meta.retained_bytes);
            state.records.head.set(meta.head);
            state.records.tail.set(meta.tail);
        } else if let Some(position) = suffix[8..].strip_prefix(b"/r/") {
            let position = usize::try_from(integer(position)?)
                .map_err(|_| corrupt("broker stream position exceeds this platform"))?;
            state
                .records
                .values
                .insert(position, decode::<StreamReference>(value)?);
        } else {
            return Err(corrupt("unknown broker stream record"));
        }
        Ok(())
    }

    fn restore_group_record(&self, suffix: &[u8], value: &[u8]) -> Result<()> {
        let (name, suffix) = take_name(suffix)?;
        let mut group = self.groups.entry(name).or_insert_with(|| GroupState {
            generation: 0,
            leader: String::new(),
            protocol: String::new(),
            members: CanonicalMap::default(),
            assignments: CanonicalMap::default(),
            offsets: CanonicalMap::default(),
        });
        if suffix == b"/m" {
            let meta: GroupMetadata = decode(value)?;
            group.generation = meta.generation;
            group.leader = meta.leader;
            group.protocol = meta.protocol;
        } else if let Some(suffix) = suffix.strip_prefix(b"/u/") {
            group
                .members
                .insert(decode::<String>(suffix)?, decode::<GroupMember>(value)?);
        } else if let Some(suffix) = suffix.strip_prefix(b"/a/") {
            group
                .assignments
                .insert(decode::<String>(suffix)?, decode::<Vec<u8>>(value)?);
        } else if let Some(suffix) = suffix.strip_prefix(b"/o/") {
            group
                .offsets
                .insert(decode::<(String, i32)>(suffix)?, decode::<u64>(value)?);
        } else {
            return Err(corrupt("unknown broker group record"));
        }
        Ok(())
    }

    pub fn finish_persistence_restore(&self) -> Result<()> {
        self.validate_state()
    }
}

fn typed<T: 'static>(value: &dyn Any) -> Result<&T> {
    value
        .downcast_ref()
        .ok_or_else(|| Error::internal("broker persistence observer type mismatch"))
}
fn row<T: 'static>(value: &dyn Any) -> Result<&T> {
    Ok(typed::<Arc<T>>(value)?.as_ref())
}
fn encode(value: &impl Serialize) -> Result<Vec<u8>> {
    let mut bytes = Vec::new();
    ciborium::ser::into_writer(value, &mut bytes)
        .map_err(|error| Error::internal(format!("broker record encoding failed: {error}")))?;
    Ok(bytes)
}
fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    let mut cursor = Cursor::new(bytes);
    let value = ciborium::de::from_reader(&mut cursor)
        .map_err(|error| corrupt(format!("broker record is invalid: {error}")))?;
    if cursor.position() != bytes.len() as u64 {
        return Err(corrupt("broker record has trailing bytes"));
    }
    Ok(value)
}
fn corrupt(message: impl Into<String>) -> Error {
    Error::new(ErrorCode::CorruptStorage, message.into())
}
fn integer(bytes: &[u8]) -> Result<u64> {
    Ok(u64::from_be_bytes(bytes.try_into().map_err(|_| {
        corrupt("broker numeric record key is not eight bytes")
    })?))
}
fn take_name(bytes: &[u8]) -> Result<(Name, &[u8])> {
    let mut cursor = Cursor::new(bytes);
    let name = ciborium::de::from_reader(&mut cursor)
        .map_err(|error| corrupt(format!("broker name key is invalid: {error}")))?;
    let consumed = usize::try_from(cursor.position())
        .map_err(|_| corrupt("broker name key length overflow"))?;
    Ok((name, &bytes[consumed..]))
}
fn joined(prefix: &[u8], suffix: &[u8]) -> Vec<u8> {
    let mut key = Vec::with_capacity(prefix.len() + suffix.len());
    key.extend_from_slice(prefix);
    key.extend_from_slice(suffix);
    key
}
fn name_prefix(prefix: &[u8], name: &Name) -> Result<Vec<u8>> {
    Ok(joined(prefix, &encode(name)?))
}
fn stream_prefix(id: u64) -> Vec<u8> {
    joined(b"b/s/", &id.to_be_bytes())
}
fn collection_prefix(kind: &Kind) -> Result<Vec<u8>> {
    Ok(match kind {
        Kind::Payload => b"b/p/".to_vec(),
        Kind::Segment => b"b/v/".to_vec(),
        Kind::Topic => b"b/t/".to_vec(),
        Kind::Exchange => b"b/e/".to_vec(),
        Kind::Queue => b"b/q/".to_vec(),
        Kind::Binding => b"b/b/".to_vec(),
        Kind::Stream => b"b/s/".to_vec(),
        Kind::Group => b"b/g/".to_vec(),
        Kind::Lease => b"b/l/".to_vec(),
        Kind::Consumers => b"b/c/".to_vec(),
        Kind::References(id) => joined(&stream_prefix(*id), b"/r/"),
        Kind::Members(name) => joined(&name_prefix(b"b/g/", name)?, b"/u/"),
        Kind::Assignments(name) => joined(&name_prefix(b"b/g/", name)?, b"/a/"),
        Kind::Offsets(name) => joined(&name_prefix(b"b/g/", name)?, b"/o/"),
        Kind::ConsumerRows(name) => joined(&name_prefix(b"b/c/", name)?, b"/r/"),
    })
}
fn ordinary<K: Serialize + 'static, V: Serialize + 'static>(
    kind: &Kind,
    key: &dyn Any,
    value: Option<&dyn Any>,
    operations: &mut Vec<RecordOp>,
) -> Result<()> {
    let key = joined(&collection_prefix(kind)?, &encode(typed::<K>(key)?)?);
    operations.push(match value {
        Some(value) => RecordOp::Put {
            key,
            value: encode(row::<V>(value)?)?,
        },
        None => RecordOp::Delete { key },
    });
    Ok(())
}
fn numeric<V: Serialize + 'static>(
    kind: &Kind,
    id: u64,
    value: Option<&dyn Any>,
    operations: &mut Vec<RecordOp>,
) -> Result<()> {
    let key = joined(&collection_prefix(kind)?, &id.to_be_bytes());
    operations.push(match value {
        Some(value) => RecordOp::Put {
            key,
            value: encode(row::<V>(value)?)?,
        },
        None => RecordOp::Delete { key },
    });
    Ok(())
}
fn record_change(
    kind: &Kind,
    key: &dyn Any,
    value: Option<&dyn Any>,
    operations: &mut Vec<RecordOp>,
) -> Result<()> {
    match kind {
        Kind::Payload => {
            numeric::<StoredPayloadRecord>(kind, *typed::<u64>(key)?, value, operations)
        }
        Kind::Segment => {
            numeric::<BrokerPayloadSegment>(kind, *typed::<u64>(key)?, value, operations)
        }
        Kind::Topic => ordinary::<Name, Topic>(kind, key, value, operations),
        Kind::Exchange => ordinary::<Name, Exchange>(kind, key, value, operations),
        Kind::Queue => ordinary::<Name, Queue>(kind, key, value, operations),
        Kind::Binding => {
            ordinary::<(ProjectId, String, String, String), Binding>(kind, key, value, operations)
        }
        Kind::Lease => ordinary::<(ProjectId, Uuid), i64>(kind, key, value, operations),
        Kind::Members(_) => ordinary::<String, GroupMember>(kind, key, value, operations),
        Kind::Assignments(_) => ordinary::<String, Vec<u8>>(kind, key, value, operations),
        Kind::Offsets(_) => ordinary::<(String, i32), u64>(kind, key, value, operations),
        Kind::ConsumerRows(_) => ordinary::<(Uuid, u64), ()>(kind, key, value, operations),
        Kind::References(_) => {
            numeric::<StreamReference>(kind, *typed::<usize>(key)? as u64, value, operations)
        }
        Kind::Stream => {
            let prefix = stream_prefix(*typed::<u64>(key)?);
            operations.push(match value {
                Some(value) => {
                    let stream = row::<StreamState>(value)?;
                    RecordOp::Put {
                        key: joined(&prefix, b"/m"),
                        value: encode(&StreamMetadata {
                            base_offset: stream.base_offset.get(),
                            next_offset: stream.next_offset.get(),
                            retained_bytes: stream.retained_bytes.get(),
                            head: stream.records.head.get(),
                            tail: stream.records.tail.get(),
                        })?,
                    }
                }
                None => RecordOp::DeletePrefix { prefix },
            });
            Ok(())
        }
        Kind::Group => {
            let prefix = name_prefix(b"b/g/", typed::<Name>(key)?)?;
            operations.push(match value {
                Some(value) => {
                    let group = row::<GroupState>(value)?;
                    RecordOp::Put {
                        key: joined(&prefix, b"/m"),
                        value: encode(&GroupMetadata {
                            generation: group.generation,
                            leader: group.leader.clone(),
                            protocol: group.protocol.clone(),
                        })?,
                    }
                }
                None => RecordOp::DeletePrefix { prefix },
            });
            Ok(())
        }
        Kind::Consumers => {
            let prefix = name_prefix(b"b/c/", typed::<Name>(key)?)?;
            operations.push(match value {
                Some(_) => RecordOp::Put {
                    key: joined(&prefix, b"/m"),
                    value: encode(&())?,
                },
                None => RecordOp::DeletePrefix { prefix },
            });
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn retain_forever() -> RetentionPolicy {
        RetentionPolicy {
            max_age_ms: None,
            max_bytes: None,
        }
    }

    fn save(
        broker: &BrokerStateMachine,
        segments: &SegmentStore,
        command: BrokerCommand,
        records: &mut BTreeMap<Vec<u8>, Vec<u8>>,
    ) -> Result<BrokerReply> {
        let scope = broker.begin_persistence()?;
        let reply = broker.apply(command, segments)?;
        for operation in broker.persistence_changes(&scope)? {
            match operation {
                RecordOp::Put { key, value } => {
                    records.insert(key, value);
                }
                RecordOp::Delete { key } => {
                    records.remove(&key);
                }
                RecordOp::DeletePrefix { prefix } => {
                    records.retain(|key, _| !key.starts_with(&prefix));
                }
            }
        }
        Ok(reply)
    }

    fn load(records: &BTreeMap<Vec<u8>, Vec<u8>>) -> Result<BrokerStateMachine> {
        let restored = BrokerStateMachine::default();
        for (key, value) in records {
            restored.restore_persistence_record(key, value)?;
        }
        restored.finish_persistence_restore()?;
        Ok(restored)
    }

    fn kafka(project: ProjectId, payloads: Vec<Vec<u8>>) -> BrokerCommand {
        BrokerCommand::PublishKafkaBatch {
            project,
            topic: "events".to_owned(),
            partition: 0,
            resolved_time_ms: 1,
            records: payloads
                .into_iter()
                .map(|payload| KafkaBatchRecord {
                    create_time_ms: Some(1),
                    key: None,
                    headers: BTreeMap::new(),
                    payload,
                    value_is_null: false,
                })
                .collect(),
        }
    }

    #[test]
    fn broker_records_restore_streams_groups_queues_and_unacked_deliveries() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let segments = SegmentStore::open(directory.path(), usize::MAX)?;
        let broker = BrokerStateMachine::default();
        let project = ProjectId::random();
        let owner = Uuid::new_v4();
        let mut records = BTreeMap::new();
        save(
            &broker,
            &segments,
            BrokerCommand::CreateTopic {
                project,
                name: "events".to_owned(),
                partitions: 1,
                retention: retain_forever(),
            },
            &mut records,
        )?;
        save(
            &broker,
            &segments,
            kafka(project, vec![b"first".to_vec(), b"second".to_vec()]),
            &mut records,
        )?;
        save(
            &broker,
            &segments,
            BrokerCommand::JoinGroup {
                project,
                group: "workers".to_owned(),
                member: "member".to_owned(),
                session_timeout_ms: 30_000,
                resolved_time_ms: 1,
                protocols: BTreeMap::from([("range".to_owned(), vec![1, 2, 3])]),
            },
            &mut records,
        )?;
        save(
            &broker,
            &segments,
            BrokerCommand::CommitOffset {
                project,
                group: "workers".to_owned(),
                topic: "events".to_owned(),
                partition: 0,
                offset: 1,
                generation: None,
                member: None,
            },
            &mut records,
        )?;
        save(
            &broker,
            &segments,
            BrokerCommand::CreateExchange {
                project,
                name: "fan".to_owned(),
                kind: AmqpExchangeKind::Fanout,
                durable: true,
                passive: false,
            },
            &mut records,
        )?;
        save(
            &broker,
            &segments,
            BrokerCommand::CreateQueue {
                project,
                name: "jobs".to_owned(),
                kind: QueueKind::Classic,
                durable: true,
                passive: false,
                dead_letter_exchange: None,
                dead_letter_routing_key: None,
                retention: retain_forever(),
                exclusive_owner: None,
                auto_delete: false,
            },
            &mut records,
        )?;
        save(
            &broker,
            &segments,
            BrokerCommand::BindQueue {
                project,
                exchange: "fan".to_owned(),
                queue: "jobs".to_owned(),
                routing_key: String::new(),
            },
            &mut records,
        )?;
        save(
            &broker,
            &segments,
            BrokerCommand::RegisterConsumer {
                project,
                queue: "jobs".to_owned(),
                owner,
                consumer: 7,
            },
            &mut records,
        )?;
        save(
            &broker,
            &segments,
            BrokerCommand::PublishAmqp {
                project,
                exchange: "fan".to_owned(),
                routing_key: String::new(),
                mandatory: false,
                resolved_time_ms: 2,
                properties: BTreeMap::new(),
                headers: BTreeMap::new(),
                payload: b"job".to_vec(),
            },
            &mut records,
        )?;
        save(
            &broker,
            &segments,
            BrokerCommand::DeliverQueue {
                project,
                queue: "jobs".to_owned(),
                offset: StreamOffset::First,
                maximum: 1,
                owner,
                consumer: 7,
                automatic_ack: false,
                resolved_time_ms: 3,
            },
            &mut records,
        )?;

        let restored = load(&records)?;
        assert_eq!(
            restored.topic_metrics(project),
            broker.topic_metrics(project)
        );
        assert_eq!(
            restored.queue_metrics(project),
            broker.queue_metrics(project)
        );
        assert_eq!(
            restored.exchange_metrics(project),
            broker.exchange_metrics(project)
        );
        assert_eq!(
            restored.committed_offset(project, "workers", "events", 0),
            Some(1)
        );
        assert_eq!(
            restored.group_member(project, "workers", 1, "member"),
            broker.group_member(project, "workers", 1, "member")
        );
        assert_eq!(
            restored
                .delivery_leases
                .get(&(project, owner))
                .map(|value| *value),
            broker
                .delivery_leases
                .get(&(project, owner))
                .map(|value| *value)
        );
        assert_eq!(
            restored
                .queue_consumers
                .get(&(project, "jobs".to_owned()))
                .map(|set| set.len()),
            Some(1)
        );
        let stream = restored
            .queues
            .get(&(project, "jobs".to_owned()))
            .ok_or_else(|| Error::internal("restored queue missing"))?
            .stream;
        assert!(
            matches!(restored.streams.get(&stream).and_then(|state| state.records.front()).map(|reference| reference.delivery), Some(DeliveryState::Unacked { owner: candidate, consumer: 7, .. }) if candidate == owner)
        );

        save(
            &broker,
            &segments,
            BrokerCommand::ClearTopic {
                project,
                name: "events".to_owned(),
            },
            &mut records,
        )?;
        save(
            &broker,
            &segments,
            BrokerCommand::DeleteQueue {
                project,
                name: "jobs".to_owned(),
                if_unused: false,
                if_empty: false,
            },
            &mut records,
        )?;
        let restored = load(&records)?;
        assert_eq!(
            restored.topic_metrics(project),
            broker.topic_metrics(project)
        );
        assert!(restored.queue_info(project, "jobs").is_none());
        assert_eq!(
            restored.exchange_metrics(project),
            broker.exchange_metrics(project)
        );
        Ok(())
    }

    #[test]
    fn one_append_does_not_serialize_unrelated_large_payloads_or_stream_history() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let segments = SegmentStore::open(directory.path(), usize::MAX)?;
        let broker = BrokerStateMachine::default();
        let project = ProjectId::random();
        let mut records = BTreeMap::new();
        save(
            &broker,
            &segments,
            BrokerCommand::CreateTopic {
                project,
                name: "events".to_owned(),
                partitions: 1,
                retention: retain_forever(),
            },
            &mut records,
        )?;
        save(
            &broker,
            &segments,
            kafka(project, vec![vec![9; 128 * 1024]; 64]),
            &mut records,
        )?;
        let scope = broker.begin_persistence()?;
        broker.apply(kafka(project, vec![vec![1]]), &segments)?;
        let changes = broker.persistence_changes(&scope)?;
        assert!(
            changes.len() <= 8,
            "append captured {} records",
            changes.len()
        );
        let encoded_bytes: usize = changes
            .iter()
            .map(|change| match change {
                RecordOp::Put { key, value } => key.len() + value.len(),
                RecordOp::Delete { key } => key.len(),
                RecordOp::DeletePrefix { prefix } => prefix.len(),
            })
            .sum();
        assert!(
            encoded_bytes < 4096,
            "one append encoded {encoded_bytes} metadata bytes"
        );
        Ok(())
    }

    #[test]
    fn failed_and_dropped_scopes_do_not_leave_capture_active() -> Result<()> {
        let broker = BrokerStateMachine::default();
        let scope = broker.begin_persistence()?;
        assert!(broker.begin_persistence().is_err());
        drop(scope);
        let scope = broker.begin_persistence()?;
        assert_eq!(broker.persistence_changes(&scope)?.len(), 1);
        let next = broker.begin_persistence()?;
        // Dropping a consumed previous guard must not remove a newer scope.
        drop(scope);
        assert!(broker.begin_persistence().is_err());
        drop(next);
        Ok(())
    }
}
