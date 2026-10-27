use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    sync::{Arc, Weak},
};

use irongraph_types::{Bookmark, Error, ErrorCode, ProjectId, Result};
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::records::RecordReader;
use super::{RecordOp, RecordStore};

const PREFIX: &[u8] = b"\xffpayload/";

/// Domain ownership of an immutable payload collection.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[repr(u8)]
pub enum SegmentFamily {
    Nodes = 0,
    Edges = 1,
    NodeTemporal = 2,
    EdgeTemporal = 3,
    Embeddings = 4,
    BrokerPayload = 5,
    BrokerReferences = 6,
    BrokerCoordination = 7,
    Security = 8,
    Catalog = 9,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentRecord {
    pub kind: u16,
    pub payload: Vec<u8>,
}

/// Stable content identity. `file_name` identifies redb records; it is never an OS file path.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SegmentDescriptor {
    pub family: SegmentFamily,
    pub project_id: Option<ProjectId>,
    pub file_name: String,
    pub bytes: u64,
    pub checksum: [u8; 32],
}

/// Stable logical record extent, independent of redb page placement.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentRecordLocation {
    pub offset: u64,
    pub bytes: u64,
}

#[derive(Serialize, Deserialize)]
struct Manifest {
    descriptor: SegmentDescriptor,
    records: u64,
}

#[derive(Debug)]
struct StagedPayload {
    descriptor: SegmentDescriptor,
    records: Vec<(SegmentRecordLocation, Vec<u8>)>,
    claims: Mutex<Claims>,
}

#[derive(Debug)]
struct Claims {
    unbound: usize,
    latest_bound: Option<Bookmark>,
}

#[derive(Debug, Default)]
struct PendingPayloads {
    payloads: BTreeMap<String, Arc<StagedPayload>>,
    // Exactly one entry per bound descriptor, replaced when its latest publication advances.
    publications: BTreeSet<(u64, String)>,
}

/// Payload persistence adapter sharing the graph's transactional file. New bytes stay in
/// in-flight staging until their enclosing ordered record batch is durably committed.
#[derive(Clone, Debug)]
pub struct SegmentStore {
    records: RecordStore,
    pending: Arc<Mutex<PendingPayloads>>,
    pins: Arc<Mutex<BTreeMap<String, Weak<PinnedPayload>>>>,
}

#[derive(Debug)]
struct PinnedPayload {
    descriptor: SegmentDescriptor,
    staged: Option<Arc<StagedPayload>>,
    reader: Option<RecordReader>,
    registry: Weak<Mutex<BTreeMap<String, Weak<PinnedPayload>>>>,
}

impl Drop for PinnedPayload {
    fn drop(&mut self) {
        if let Some(registry) = self.registry.upgrade() {
            let mut pins = registry.lock();
            if pins
                .get(&self.descriptor.file_name)
                .is_some_and(|weak| std::ptr::eq(weak.as_ptr(), self))
            {
                pins.remove(&self.descriptor.file_name);
            }
        }
    }
}

/// A short-lived broker read guard. It protects only bytes currently borrowed by a reader;
/// it exposes no historical query, export or user savepoint API.
#[derive(Debug)]
pub struct SegmentPin {
    _store: SegmentStore,
    _payload: Arc<PinnedPayload>,
}

impl SegmentStore {
    /// Standalone fixture opener. Production shares its RecordStore with `with_records`.
    /// The unused size argument imposes no quota.
    pub fn open(root: impl AsRef<Path>, _max_record_bytes: usize) -> Result<Self> {
        Ok(Self::with_records(RecordStore::open(
            root.as_ref().join("database.redb"),
            Uuid::nil(),
        )?))
    }

    #[must_use]
    pub fn with_records(records: RecordStore) -> Self {
        Self {
            records,
            pending: Arc::new(Mutex::new(PendingPayloads::default())),
            pins: Arc::new(Mutex::new(BTreeMap::new())),
        }
    }

    pub fn pin(&self, descriptor: &SegmentDescriptor) -> Result<SegmentPin> {
        validate_descriptor(descriptor)?;
        let existing = self
            .pins
            .lock()
            .get(&descriptor.file_name)
            .and_then(Weak::upgrade);
        if let Some(payload) = existing {
            if payload.descriptor != *descriptor {
                return Err(corrupt("pinned payload identity differs"));
            }
            return Ok(SegmentPin {
                _store: self.clone(),
                _payload: payload,
            });
        }
        let staged = self
            .pending
            .lock()
            .payloads
            .get(&descriptor.file_name)
            .cloned();
        if staged
            .as_ref()
            .is_some_and(|payload| payload.descriptor != *descriptor)
        {
            return Err(corrupt("staged payload identity differs"));
        }
        let reader = if staged.is_some() {
            None
        } else {
            Some(self.records.reader()?)
        };
        if let Some(reader) = &reader {
            manifest_from_reader(reader, descriptor)?;
        }
        let payload = Arc::new(PinnedPayload {
            descriptor: descriptor.clone(),
            staged,
            reader,
            registry: Arc::downgrade(&self.pins),
        });
        let mut pins = self.pins.lock();
        let payload = match pins.get(&descriptor.file_name).and_then(Weak::upgrade) {
            Some(existing) => existing,
            None => {
                pins.insert(descriptor.file_name.clone(), Arc::downgrade(&payload));
                payload
            }
        };
        Ok(SegmentPin {
            _store: self.clone(),
            _payload: payload,
        })
    }

    pub fn write_immutable(
        &self,
        family: SegmentFamily,
        project_id: Option<ProjectId>,
        records: &[SegmentRecord],
    ) -> Result<SegmentDescriptor> {
        self.write_immutable_streaming(family, project_id, records.len(), |index| {
            records
                .get(index)
                .cloned()
                .ok_or_else(|| Error::internal("payload record disappeared"))
        })
    }

    pub fn write_immutable_streaming(
        &self,
        family: SegmentFamily,
        project_id: Option<ProjectId>,
        count: usize,
        record_at: impl FnMut(usize) -> Result<SegmentRecord>,
    ) -> Result<SegmentDescriptor> {
        self.write_immutable_streaming_indexed(family, project_id, count, record_at)
            .map(|(descriptor, _)| descriptor)
    }

    /// Encodes each input once, without filesystem operations or durability waits.
    pub fn write_immutable_streaming_indexed(
        &self,
        family: SegmentFamily,
        project_id: Option<ProjectId>,
        count: usize,
        mut record_at: impl FnMut(usize) -> Result<SegmentRecord>,
    ) -> Result<(SegmentDescriptor, Vec<SegmentRecordLocation>)> {
        if family_requires_project(family) != project_id.is_some() {
            return Err(Error::invalid_data(
                "payload project scope does not match its family",
            ));
        }
        let mut hash = payload_hasher(family, project_id);
        let mut bytes = 0_u64;
        let mut records = Vec::with_capacity(count);
        let mut locations = Vec::with_capacity(count);
        for index in 0..count {
            let encoded = postcard::to_stdvec(&record_at(index)?)
                .map_err(|error| Error::invalid_data(format!("payload encoding: {error}")))?;
            let length = u64::try_from(encoded.len())
                .map_err(|_| Error::invalid_data("payload length overflow"))?;
            let location = SegmentRecordLocation {
                offset: bytes,
                bytes: length,
            };
            bytes = bytes
                .checked_add(length)
                .and_then(|n| n.checked_add(8))
                .ok_or_else(|| Error::invalid_data("payload extent overflow"))?;
            hash.update(&length.to_le_bytes());
            hash.update(&encoded);
            locations.push(location);
            records.push((location, encoded));
        }
        let checksum = *hash.finalize().as_bytes();
        let descriptor = SegmentDescriptor {
            family,
            project_id,
            file_name: hex::encode(checksum),
            bytes,
            checksum,
        };
        let staged = Arc::new(StagedPayload {
            descriptor: descriptor.clone(),
            records,
            claims: Mutex::new(Claims {
                unbound: 1,
                latest_bound: None,
            }),
        });
        let mut pending = self.pending.lock();
        if let Some(existing) = pending.payloads.get(&descriptor.file_name) {
            if existing.descriptor != descriptor {
                return Err(corrupt("payload content identity collision"));
            }
            let mut claims = existing.claims.lock();
            claims.unbound = claims
                .unbound
                .checked_add(1)
                .ok_or_else(|| Error::internal("pending payload claim overflow"))?;
        } else {
            pending
                .payloads
                .insert(descriptor.file_name.clone(), staged);
        }
        Ok((descriptor, locations))
    }

    /// Operations join the same RecordBatch as referencing broker metadata.
    pub fn pending_operations(
        &self,
        descriptors: &[SegmentDescriptor],
        publication: Bookmark,
    ) -> Result<Vec<RecordOp>> {
        if publication.term == 0 || publication.index == 0 {
            return Err(Error::invalid_data("payload publication must be nonzero"));
        }
        let mut operations = Vec::new();
        let mut seen = BTreeSet::new();
        let mut captured = Vec::new();
        for descriptor in descriptors {
            validate_descriptor(descriptor)?;
            if !seen.insert(&descriptor.file_name) {
                continue;
            }
            let staged = self
                .pending
                .lock()
                .payloads
                .get(&descriptor.file_name)
                .cloned();
            let Some(staged) = staged else {
                continue;
            };
            if staged.descriptor != *descriptor {
                return Err(corrupt("staged payload identity differs"));
            }
            if staged.claims.lock().unbound == 0 {
                return Err(Error::internal("payload publication has no unbound claim"));
            }
            let manifest = Manifest {
                descriptor: descriptor.clone(),
                records: staged.records.len() as u64,
            };
            for (location, value) in &staged.records {
                operations.push(RecordOp::Put {
                    key: record_key(descriptor, location.offset),
                    value: value.clone(),
                });
            }
            operations.push(RecordOp::Put {
                key: manifest_key(descriptor),
                value: postcard::to_stdvec(&manifest)
                    .map_err(|error| Error::invalid_data(error.to_string()))?,
            });
            captured.push((descriptor, staged));
        }
        // Bind only after every descriptor and serialization has succeeded. One watermark
        // covers all bound claims for identical bytes; a later publication retains those bytes
        // until it too is durable, without retaining a publication history.
        for (descriptor, staged) in captured {
            let mut pending = self.pending.lock();
            if !pending
                .payloads
                .get(&descriptor.file_name)
                .is_some_and(|current| Arc::ptr_eq(current, &staged))
            {
                return Err(Error::internal("staged payload changed before publication"));
            }
            let mut claims = staged.claims.lock();
            claims.unbound = claims
                .unbound
                .checked_sub(1)
                .ok_or_else(|| Error::internal("payload publication has no unbound claim"))?;
            if claims
                .latest_bound
                .is_none_or(|latest| latest.index < publication.index)
            {
                if let Some(old) = claims.latest_bound {
                    pending
                        .publications
                        .remove(&(old.index, descriptor.file_name.clone()));
                }
                pending
                    .publications
                    .insert((publication.index, descriptor.file_name.clone()));
                claims.latest_bound = Some(publication);
            }
        }
        Ok(operations)
    }

    /// Retires covered staging even when coalescing removed its manifest operation.
    pub fn release_committed_through(&self, bookmark: Bookmark) -> Result<()> {
        loop {
            let retired = {
                let mut pending = self.pending.lock();
                if pending
                    .publications
                    .first()
                    .is_none_or(|(index, _)| *index > bookmark.index)
                {
                    break;
                }
                let Some((_, name)) = pending.publications.pop_first() else {
                    return Err(Error::internal(
                        "payload publication index changed under lock",
                    ));
                };
                let staged = pending
                    .payloads
                    .get(&name)
                    .ok_or_else(|| Error::internal("indexed staged payload is absent"))?;
                let mut claims = staged.claims.lock();
                claims.latest_bound = None;
                let remove = claims.unbound == 0;
                drop(claims);
                if remove {
                    pending.payloads.remove(&name)
                } else {
                    None
                }
            };
            // Each lock acquisition touches one descriptor. Free its potentially large byte
            // allocation only after unlocking, before considering the next covered descriptor.
            drop(retired);
        }
        Ok(())
    }

    pub fn discard_staged(&self, descriptor: &SegmentDescriptor) {
        let retired = release_claim(&mut self.pending.lock(), &descriptor.file_name);
        drop(retired);
    }

    /// Deletions must commit together with the metadata retiring those payloads, never ahead of it.
    pub fn reclamation_operations(
        &self,
        candidates: &[SegmentDescriptor],
        live: &BTreeSet<String>,
    ) -> Result<Vec<RecordOp>> {
        let mut seen = BTreeSet::new();
        let mut operations = Vec::new();
        for descriptor in candidates {
            validate_descriptor(descriptor)?;
            if !live.contains(&descriptor.file_name) && seen.insert(&descriptor.file_name) {
                operations.push(RecordOp::DeletePrefix {
                    prefix: payload_prefix(descriptor),
                });
            }
        }
        Ok(operations)
    }

    pub fn immutable_file_names(&self) -> Result<Vec<String>> {
        let mut names = BTreeSet::new();
        let mut previous = None::<String>;
        loop {
            let next = {
                let pending = self.pending.lock();
                match &previous {
                    Some(previous) => pending
                        .payloads
                        .range((
                            std::ops::Bound::Excluded(previous.clone()),
                            std::ops::Bound::Unbounded,
                        ))
                        .next(),
                    None => pending.payloads.first_key_value(),
                }
                .map(|(name, _)| name.clone())
            };
            let Some(name) = next else {
                break;
            };
            previous = Some(name.clone());
            names.insert(name);
        }
        self.records.scan_prefix(PREFIX, |key, _| {
            if is_manifest_key(key) {
                names.insert(hex::encode(&key[PREFIX.len()..PREFIX.len() + 32]));
            }
            Ok(())
        })?;
        Ok(names.into_iter().collect())
    }

    pub fn read(&self, descriptor: &SegmentDescriptor) -> Result<Vec<SegmentRecord>> {
        let mut records = Vec::new();
        self.try_for_each_record(descriptor, |record| {
            records.push(record);
            Ok(())
        })?;
        Ok(records)
    }

    pub fn try_for_each_record(
        &self,
        descriptor: &SegmentDescriptor,
        mut visitor: impl FnMut(SegmentRecord) -> Result<()>,
    ) -> Result<()> {
        validate_descriptor(descriptor)?;
        let staged = self
            .pending
            .lock()
            .payloads
            .get(&descriptor.file_name)
            .cloned();
        if let Some(staged) = staged {
            if staged.descriptor != *descriptor {
                return Err(corrupt("staged payload identity differs"));
            }
            for (_, encoded) in &staged.records {
                visitor(decode_record(encoded)?)?;
            }
        } else {
            let manifest = self.manifest(descriptor)?;
            let mut count = 0_u64;
            self.records
                .scan_prefix(&record_prefix(descriptor), |_, encoded| {
                    count += 1;
                    visitor(decode_record(encoded)?)
                })?;
            if count != manifest.records {
                return Err(corrupt("payload record count mismatch"));
            }
        }
        Ok(())
    }

    pub fn read_record_at(
        &self,
        descriptor: &SegmentDescriptor,
        location: SegmentRecordLocation,
    ) -> Result<SegmentRecord> {
        validate_descriptor(descriptor)?;
        if location
            .offset
            .checked_add(location.bytes)
            .is_none_or(|end| end > descriptor.bytes)
        {
            return Err(corrupt("payload record extent exceeds its descriptor"));
        }
        let pinned = self
            .pins
            .lock()
            .get(&descriptor.file_name)
            .and_then(Weak::upgrade);
        let staged = pinned
            .as_ref()
            .and_then(|pin| pin.staged.clone())
            .or_else(|| {
                self.pending
                    .lock()
                    .payloads
                    .get(&descriptor.file_name)
                    .cloned()
            });
        if let Some(staged) = staged {
            if staged.descriptor != *descriptor {
                return Err(corrupt("staged payload identity differs"));
            }
            let index = staged
                .records
                .binary_search_by_key(&location.offset, |(extent, _)| extent.offset)
                .map_err(|_| corrupt("payload record offset is absent"))?;
            let (extent, encoded) = &staged.records[index];
            if *extent != location {
                return Err(corrupt("payload record length differs"));
            }
            return decode_record(encoded);
        }
        let key = record_key(descriptor, location.offset);
        let encoded = match pinned.as_ref().and_then(|pin| pin.reader.as_ref()) {
            Some(reader) => reader.get(&key)?,
            None => self.records.get(&key)?,
        }
        .ok_or_else(|| {
            Error::new(
                ErrorCode::RetentionExpired,
                "requested payload record was reclaimed",
            )
        })?;
        if encoded.len() as u64 != location.bytes {
            return Err(corrupt("payload record length differs"));
        }
        decode_record(&encoded)
    }

    pub fn validate(&self, descriptor: &SegmentDescriptor) -> Result<()> {
        validate_descriptor(descriptor)?;
        let staged = self
            .pending
            .lock()
            .payloads
            .get(&descriptor.file_name)
            .cloned();
        if let Some(staged) = staged {
            if staged.descriptor != *descriptor {
                return Err(corrupt("staged payload identity differs"));
            }
            return Ok(());
        }
        let manifest = self.manifest(descriptor)?;
        let mut hash = payload_hasher(descriptor.family, descriptor.project_id);
        let mut bytes = 0_u64;
        let mut count = 0_u64;
        let prefix = record_prefix(descriptor);
        self.records.scan_prefix(&prefix, |key, value| {
            let offset = key
                .get(prefix.len()..)
                .and_then(|b| <[u8; 8]>::try_from(b).ok())
                .map(u64::from_be_bytes)
                .ok_or_else(|| corrupt("payload key offset is invalid"))?;
            if offset != bytes {
                return Err(corrupt("payload record offsets are discontinuous"));
            }
            let length = value.len() as u64;
            bytes = bytes
                .checked_add(length)
                .and_then(|n| n.checked_add(8))
                .ok_or_else(|| corrupt("payload extent overflow"))?;
            hash.update(&length.to_le_bytes());
            hash.update(value);
            count += 1;
            Ok(())
        })?;
        if bytes != descriptor.bytes
            || count != manifest.records
            || hash.finalize().as_bytes() != &descriptor.checksum
        {
            return Err(corrupt("payload checksum, length or record count mismatch"));
        }
        Ok(())
    }

    fn manifest(&self, descriptor: &SegmentDescriptor) -> Result<Manifest> {
        let encoded = self
            .records
            .get(&manifest_key(descriptor))?
            .ok_or_else(|| corrupt("payload descriptor is absent"))?;
        let manifest: Manifest = postcard::from_bytes(&encoded)
            .map_err(|error| corrupt_owned(format!("payload manifest: {error}")))?;
        if manifest.descriptor != *descriptor {
            return Err(corrupt("payload manifest identity differs"));
        }
        Ok(manifest)
    }
}

fn decode_record(bytes: &[u8]) -> Result<SegmentRecord> {
    postcard::from_bytes(bytes).map_err(|error| corrupt_owned(format!("payload record: {error}")))
}
fn manifest_from_reader(reader: &RecordReader, descriptor: &SegmentDescriptor) -> Result<Manifest> {
    let encoded = reader
        .get(&manifest_key(descriptor))?
        .ok_or_else(|| corrupt("payload descriptor is absent"))?;
    let manifest: Manifest = postcard::from_bytes(&encoded)
        .map_err(|error| corrupt_owned(format!("payload manifest: {error}")))?;
    if manifest.descriptor != *descriptor {
        return Err(corrupt("payload manifest identity differs"));
    }
    Ok(manifest)
}
fn release_claim(pending: &mut PendingPayloads, name: &str) -> Option<Arc<StagedPayload>> {
    if let Some(staged) = pending.payloads.get(name) {
        let mut claims = staged.claims.lock();
        claims.unbound = claims.unbound.saturating_sub(1);
        let remove = claims.unbound == 0 && claims.latest_bound.is_none();
        drop(claims);
        if remove {
            return pending.payloads.remove(name);
        }
    }
    None
}
fn payload_prefix(descriptor: &SegmentDescriptor) -> Vec<u8> {
    let mut key = PREFIX.to_vec();
    key.extend_from_slice(&descriptor.checksum);
    key
}
fn manifest_key(descriptor: &SegmentDescriptor) -> Vec<u8> {
    let mut key = payload_prefix(descriptor);
    key.push(b'm');
    key
}
fn record_prefix(descriptor: &SegmentDescriptor) -> Vec<u8> {
    let mut key = payload_prefix(descriptor);
    key.push(b'r');
    key
}
fn record_key(descriptor: &SegmentDescriptor, offset: u64) -> Vec<u8> {
    let mut key = record_prefix(descriptor);
    key.extend_from_slice(&offset.to_be_bytes());
    key
}
fn is_manifest_key(key: &[u8]) -> bool {
    key.starts_with(PREFIX) && key.len() == PREFIX.len() + 33 && key.last() == Some(&b'm')
}
fn family_requires_project(family: SegmentFamily) -> bool {
    !matches!(family, SegmentFamily::Security | SegmentFamily::Catalog)
}
fn validate_descriptor(descriptor: &SegmentDescriptor) -> Result<()> {
    if descriptor.file_name != hex::encode(descriptor.checksum)
        || family_requires_project(descriptor.family) != descriptor.project_id.is_some()
    {
        return Err(corrupt("payload content identity is invalid"));
    }
    Ok(())
}
fn payload_hasher(family: SegmentFamily, project: Option<ProjectId>) -> blake3::Hasher {
    let mut hash = blake3::Hasher::new();
    hash.update(b"IronGraph-payload-1");
    hash.update(&[family as u8]);
    if let Some(project) = project {
        hash.update(project.0.as_bytes());
    }
    hash
}
fn corrupt(message: &'static str) -> Error {
    Error::new(ErrorCode::CorruptStorage, message)
}
fn corrupt_owned(message: String) -> Error {
    Error::new(ErrorCode::CorruptStorage, message)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::RecordBatch;
    use irongraph_types::Bookmark;

    #[test]
    fn staged_payload_is_readable_before_io_and_restores_from_same_database() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("database.redb");
        let identity = Uuid::new_v4();
        let project = ProjectId::random();
        let payload = vec![7; 1024 * 1024 + 13];
        let descriptor;
        let locations;
        {
            let records = RecordStore::open(&path, identity)?;
            let store = SegmentStore::with_records(records.clone());
            (descriptor, locations) = store.write_immutable_streaming_indexed(
                SegmentFamily::BrokerPayload,
                Some(project),
                2,
                |i| {
                    Ok(SegmentRecord {
                        kind: i as u16,
                        payload: payload.clone(),
                    })
                },
            )?;
            assert_eq!(
                store.read_record_at(&descriptor, locations[1])?.payload,
                payload
            );
            assert_eq!(
                records.next_after(PREFIX, None)?,
                None,
                "staging must not write the durable database"
            );
            let batch = RecordBatch {
                first_index: 1,
                bookmark: Bookmark { term: 1, index: 1 },
                operations: store.pending_operations(
                    std::slice::from_ref(&descriptor),
                    Bookmark { term: 1, index: 1 },
                )?,
            };
            records.commit(std::slice::from_ref(&batch))?;
            store.release_committed_through(batch.bookmark)?;
            assert!(store.pending.lock().payloads.is_empty());
            store.validate(&descriptor)?;
            assert_eq!(
                store.read_record_at(&descriptor, locations[0])?.payload,
                payload
            );
            assert_eq!(std::fs::read_dir(directory.path())?.count(), 1);
        }
        let records = RecordStore::open(&path, identity)?;
        let store = SegmentStore::with_records(records.clone());
        store.validate(&descriptor)?;
        assert_eq!(store.read(&descriptor)?.len(), 2);
        assert_eq!(
            store.read_record_at(&descriptor, locations[1])?.payload,
            payload
        );
        let ops =
            store.reclamation_operations(std::slice::from_ref(&descriptor), &BTreeSet::new())?;
        records.commit(&[RecordBatch {
            first_index: 2,
            bookmark: Bookmark { term: 1, index: 2 },
            operations: ops,
        }])?;
        assert_eq!(
            store
                .read_record_at(&descriptor, locations[0])
                .err()
                .map(|e| e.code),
            Some(ErrorCode::RetentionExpired)
        );
        Ok(())
    }

    #[test]
    fn content_identity_includes_domain_scope_and_rejects_invalid_extent() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let store = SegmentStore::open(directory.path(), 1)?;
        let project = ProjectId::random();
        let input = [SegmentRecord {
            kind: 1,
            payload: vec![3; 4096],
        }];
        let a = store.write_immutable(SegmentFamily::BrokerPayload, Some(project), &input)?;
        let b = store.write_immutable(SegmentFamily::Nodes, Some(project), &input)?;
        assert_ne!(a.checksum, b.checksum);
        assert!(
            store
                .read_record_at(
                    &a,
                    SegmentRecordLocation {
                        offset: 1,
                        bytes: 1
                    }
                )
                .is_err()
        );
        assert!(
            store
                .reclamation_operations(
                    std::slice::from_ref(&a),
                    &BTreeSet::from([a.file_name.clone()])
                )?
                .is_empty()
        );
        Ok(())
    }

    #[test]
    fn borrowed_payload_survives_retirement_and_releases_page_pin_after_read() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let records = RecordStore::open(directory.path().join("database.redb"), Uuid::new_v4())?;
        let store = SegmentStore::with_records(records.clone());
        let (descriptor, locations) = store.write_immutable_streaming_indexed(
            SegmentFamily::BrokerPayload,
            Some(ProjectId::random()),
            1,
            |_| {
                Ok(SegmentRecord {
                    kind: 1,
                    payload: vec![5; 1024 * 1024],
                })
            },
        )?;
        let batch = RecordBatch {
            first_index: 1,
            bookmark: Bookmark { term: 1, index: 1 },
            operations: store.pending_operations(
                std::slice::from_ref(&descriptor),
                Bookmark { term: 1, index: 1 },
            )?,
        };
        records.commit(std::slice::from_ref(&batch))?;
        store.release_committed_through(batch.bookmark)?;
        let pin = store.pin(&descriptor)?;
        records.commit(&[RecordBatch {
            first_index: 2,
            bookmark: Bookmark { term: 1, index: 2 },
            operations: store
                .reclamation_operations(std::slice::from_ref(&descriptor), &BTreeSet::new())?,
        }])?;
        assert_eq!(
            store.read_record_at(&descriptor, locations[0])?.payload,
            vec![5; 1024 * 1024]
        );
        drop(pin);
        assert!(store.pins.lock().is_empty());
        assert_eq!(
            store
                .read_record_at(&descriptor, locations[0])
                .err()
                .map(|e| e.code),
            Some(ErrorCode::RetentionExpired)
        );
        Ok(())
    }

    #[test]
    fn coalesced_retirement_releases_covered_claims_but_preserves_later_and_unbound() -> Result<()>
    {
        let directory = tempfile::tempdir()?;
        let records = RecordStore::open(directory.path().join("database.redb"), Uuid::new_v4())?;
        let store = SegmentStore::with_records(records.clone());
        let project = Some(ProjectId::random());
        let input = [SegmentRecord {
            kind: 1,
            payload: vec![9; 4096],
        }];
        let descriptor = store.write_immutable(SegmentFamily::BrokerPayload, project, &input)?;
        let first = Bookmark { term: 1, index: 1 };
        let later = Bookmark { term: 1, index: 3 };
        store.pending_operations(std::slice::from_ref(&descriptor), first)?;
        // The runtime coalesces publication 1's payload away when publication 2 deletes it.
        records.commit(&[RecordBatch {
            first_index: 1,
            bookmark: Bookmark { term: 1, index: 2 },
            operations: store
                .reclamation_operations(std::slice::from_ref(&descriptor), &BTreeSet::new())?,
        }])?;
        // A reintroduction binds the same bytes before the older durable completion arrives.
        store.write_immutable(SegmentFamily::BrokerPayload, project, &input)?;
        let later_operations =
            store.pending_operations(std::slice::from_ref(&descriptor), later)?;
        store.release_committed_through(Bookmark { term: 1, index: 2 })?;
        assert_eq!(store.read(&descriptor)?, input);
        assert_eq!(store.pending.lock().payloads.len(), 1);
        // An unbound claim must also survive completion of all bound publications.
        store.write_immutable(SegmentFamily::BrokerPayload, project, &input)?;
        records.commit(&[RecordBatch {
            first_index: 3,
            bookmark: later,
            operations: later_operations,
        }])?;
        store.release_committed_through(later)?;
        assert_eq!(store.pending.lock().payloads.len(), 1);
        store.release_committed_through(later)?; // idempotent completion
        assert_eq!(store.pending.lock().payloads.len(), 1);
        store.discard_staged(&descriptor);
        assert!(store.pending.lock().payloads.is_empty());
        store.validate(&descriptor)?;

        // If there is no reintroduction, a coalesced delete releases every bound claim.
        let gone = store.write_immutable(SegmentFamily::Nodes, project, &input)?;
        store.pending_operations(std::slice::from_ref(&gone), Bookmark { term: 1, index: 4 })?;
        records.commit(&[RecordBatch {
            first_index: 4,
            bookmark: Bookmark { term: 1, index: 10 },
            operations: store
                .reclamation_operations(std::slice::from_ref(&gone), &BTreeSet::new())?,
        }])?;
        store.release_committed_through(Bookmark { term: 1, index: 10 })?;
        assert!(store.pending.lock().payloads.is_empty());
        assert!(records.get(&manifest_key(&gone))?.is_none());
        Ok(())
    }

    #[test]
    fn indexed_release_races_reintroduction_without_retaining_publication_history() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let records = RecordStore::open(directory.path().join("database.redb"), Uuid::new_v4())?;
        let store = SegmentStore::with_records(records.clone());
        let project = Some(ProjectId::random());
        let mut descriptors = Vec::new();
        for number in 0u32..128 {
            let input = [SegmentRecord {
                kind: 1,
                payload: number.to_be_bytes().repeat(1024),
            }];
            let descriptor =
                store.write_immutable(SegmentFamily::BrokerPayload, project, &input)?;
            store.pending_operations(
                std::slice::from_ref(&descriptor),
                Bookmark { term: 1, index: 1 },
            )?;
            descriptors.push((descriptor, input));
        }
        records.commit(&[RecordBatch {
            first_index: 1,
            bookmark: Bookmark { term: 1, index: 2 },
            operations: vec![],
        }])?;
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let releaser = store.clone();
        let release_barrier = barrier.clone();
        let release = std::thread::spawn(move || {
            release_barrier.wait();
            releaser.release_committed_through(Bookmark { term: 1, index: 2 })
        });
        barrier.wait();
        for (descriptor, input) in &descriptors {
            assert_eq!(
                store.write_immutable(SegmentFamily::BrokerPayload, project, input)?,
                *descriptor
            );
            store.pending_operations(
                std::slice::from_ref(descriptor),
                Bookmark { term: 1, index: 3 },
            )?;
        }
        release.join().expect("release thread")?;
        {
            let pending = store.pending.lock();
            assert_eq!(pending.payloads.len(), 128);
            assert_eq!(pending.publications.len(), 128);
            assert!(pending.publications.iter().all(|(index, _)| *index == 3));
        }
        let (descriptor, input) = &descriptors[0];
        for index in 4..=32 {
            store.write_immutable(SegmentFamily::BrokerPayload, project, input)?;
            store.pending_operations(
                std::slice::from_ref(descriptor),
                Bookmark { term: 1, index },
            )?;
            assert_eq!(
                store.pending.lock().publications.len(),
                128,
                "rebind replaces index entry"
            );
        }
        store.release_committed_through(Bookmark { term: 1, index: 31 })?;
        assert_eq!(store.pending.lock().payloads.len(), 1);
        assert_eq!(store.read(descriptor)?, *input);
        store.release_committed_through(Bookmark { term: 1, index: 32 })?;
        let pending = store.pending.lock();
        assert!(pending.payloads.is_empty());
        assert!(pending.publications.is_empty());
        Ok(())
    }

    #[test]
    fn old_commit_cannot_release_reintroduced_inflight_payload() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let records = RecordStore::open(directory.path().join("database.redb"), Uuid::new_v4())?;
        let store = SegmentStore::with_records(records.clone());
        let project = ProjectId::random();
        let input = [SegmentRecord {
            kind: 1,
            payload: vec![8; 4096],
        }];
        let descriptor =
            store.write_immutable(SegmentFamily::BrokerPayload, Some(project), &input)?;
        let first = RecordBatch {
            first_index: 1,
            bookmark: Bookmark { term: 1, index: 1 },
            operations: store.pending_operations(
                std::slice::from_ref(&descriptor),
                Bookmark { term: 1, index: 1 },
            )?,
        };
        let removal = RecordBatch {
            first_index: 2,
            bookmark: Bookmark { term: 1, index: 2 },
            operations: store
                .reclamation_operations(std::slice::from_ref(&descriptor), &BTreeSet::new())?,
        };
        assert_eq!(
            store.write_immutable(SegmentFamily::BrokerPayload, Some(project), &input)?,
            descriptor
        );
        let committed = [first, removal];
        records.commit(&committed)?;
        store.release_committed_through(Bookmark { term: 1, index: 2 })?;
        assert_eq!(store.pending.lock().payloads.len(), 1);
        let third = RecordBatch {
            first_index: 3,
            bookmark: Bookmark { term: 1, index: 3 },
            operations: store.pending_operations(
                std::slice::from_ref(&descriptor),
                Bookmark { term: 1, index: 3 },
            )?,
        };
        assert!(!third.operations.is_empty());
        records.commit(std::slice::from_ref(&third))?;
        store.release_committed_through(third.bookmark)?;
        assert!(store.pending.lock().payloads.is_empty());
        store.validate(&descriptor)?;
        assert_eq!(store.read(&descriptor)?, input);
        Ok(())
    }
}
