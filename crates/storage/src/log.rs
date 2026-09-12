use std::{
    collections::{BTreeMap, BTreeSet},
    io::Write,
    path::Path,
    sync::{Arc, OnceLock},
};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use irongraph_types::{Bookmark, Error, ErrorCode, ProjectId, Result};

use super::{Durability, FramedFile};

const LOG_MAGIC: [u8; 8] = *b"IGWAL001";

/// Canonical WAL entry category.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[repr(u8)]
pub enum MutationKind {
    Graph = 0,
    Security = 2,
    Policy = 3,
    Broker = 4,
    Project = 5,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
struct MutationBody {
    term: u64,
    index: u64,
    kind: MutationKind,
    project_id: Option<ProjectId>,
    request_id: Option<Uuid>,
    commit_time_millis: i64,
    payload: Vec<u8>,
}

/// Resolved, backend-neutral deterministic mutation stored in the standalone WAL.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MutationEntry {
    body: Arc<MutationBody>,
    checksum: [u8; 32],
    #[serde(skip)]
    checksum_verified: OnceLock<()>,
}

impl MutationEntry {
    pub fn new(
        term: u64,
        index: u64,
        kind: MutationKind,
        project_id: Option<ProjectId>,
        request_id: Option<Uuid>,
        commit_time_millis: i64,
        payload: Vec<u8>,
    ) -> Result<Self> {
        if term == 0 || index == 0 || payload.is_empty() {
            return Err(Error::invalid_data(
                "mutation term, index, and payload must be non-zero",
            ));
        }
        if matches!(
            kind,
            MutationKind::Graph | MutationKind::Broker | MutationKind::Project
        ) && project_id.is_none()
        {
            return Err(Error::invalid_data(
                "project-scoped mutation is missing project ID",
            ));
        }
        if matches!(kind, MutationKind::Security | MutationKind::Policy) && project_id.is_some() {
            return Err(Error::invalid_data(
                "store-wide mutation must not carry a project ID",
            ));
        }
        let body = MutationBody {
            term,
            index,
            kind,
            project_id,
            request_id,
            commit_time_millis,
            payload,
        };
        let checksum = body_checksum(&body)?;
        let checksum_verified = OnceLock::new();
        let _ = checksum_verified.set(());
        Ok(Self {
            body: Arc::new(body),
            checksum,
            checksum_verified,
        })
    }

    #[must_use]
    pub fn bookmark(&self) -> Bookmark {
        Bookmark {
            term: self.body.term,
            index: self.body.index,
        }
    }

    #[must_use]
    pub fn term(&self) -> u64 {
        self.body.term
    }

    #[must_use]
    pub fn index(&self) -> u64 {
        self.body.index
    }

    #[must_use]
    pub fn kind(&self) -> MutationKind {
        self.body.kind
    }

    #[must_use]
    pub fn project_id(&self) -> Option<ProjectId> {
        self.body.project_id
    }

    #[must_use]
    pub fn request_id(&self) -> Option<Uuid> {
        self.body.request_id
    }

    #[must_use]
    pub fn commit_time_millis(&self) -> i64 {
        self.body.commit_time_millis
    }

    #[must_use]
    pub fn payload(&self) -> &[u8] {
        &self.body.payload
    }

    #[must_use]
    pub const fn checksum(&self) -> [u8; 32] {
        self.checksum
    }

    pub fn verify(&self) -> Result<()> {
        if self.checksum_verified.get().is_some() {
            return Ok(());
        }
        let calculated = body_checksum(&self.body)?;
        if calculated != self.checksum {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "mutation checksum mismatch",
            ));
        }
        let _ = self.checksum_verified.set(());
        Ok(())
    }

    fn encode(&self) -> Result<Vec<u8>> {
        postcard::to_stdvec(self)
            .map_err(|error| Error::new(ErrorCode::InvalidData, error.to_string()))
    }

    fn decode(bytes: &[u8]) -> Result<Self> {
        let entry: Self = postcard::from_bytes(bytes)
            .map_err(|error| Error::new(ErrorCode::CorruptStorage, error.to_string()))?;
        entry.verify()?;
        Ok(entry)
    }
}

/// Durable ordered mutation log. A compacted bookmark is stored as the first frame so replacing
/// the retained suffix and its base is one atomic file operation.
pub struct DurableLog {
    frames: FramedFile,
    compacted_through: Bookmark,
    entries: Vec<MutationEntry>,
    request_index: BTreeMap<Uuid, u64>,
}

impl DurableLog {
    pub fn open(
        path: impl AsRef<Path>,
        max_record_bytes: usize,
        compacted_through: Bookmark,
    ) -> Result<Self> {
        let mut frames = FramedFile::open(path, LOG_MAGIC, max_record_bytes)?;
        let mut entries = Vec::with_capacity(frames.len());
        let mut request_index = BTreeMap::new();
        let mut persisted_base = None;
        let mut expected = compacted_through
            .index
            .checked_add(1)
            .ok_or_else(|| Error::new(ErrorCode::CorruptStorage, "compacted log index overflow"))?;
        let mut previous_term = compacted_through.term;
        frames.try_for_each(|record| {
            if entries.is_empty() && persisted_base.is_none() && record.payload.first() == Some(&0)
            {
                let bookmark: Bookmark = postcard::from_bytes(&record.payload[1..])
                    .map_err(|error| Error::new(ErrorCode::CorruptStorage, error.to_string()))?;
                if bookmark.index < compacted_through.index {
                    return Err(Error::new(
                        ErrorCode::CorruptStorage,
                        "WAL compacted bookmark precedes the recovered snapshot",
                    ));
                }
                persisted_base = Some(bookmark);
                expected = bookmark.index.checked_add(1).ok_or_else(|| {
                    Error::new(ErrorCode::CorruptStorage, "compacted log index overflow")
                })?;
                previous_term = bookmark.term;
                return Ok(());
            }
            let entry = MutationEntry::decode(&record.payload)?;
            if entry.index() != expected {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "mutation log contains a gap or duplicate index",
                ));
            }
            if entry.term() < previous_term {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "mutation log term moved backwards",
                ));
            }
            if let Some(request_id) = entry.request_id()
                && request_index.insert(request_id, entry.index()).is_some()
            {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "mutation log contains a duplicate request ID",
                ));
            }
            previous_term = entry.term();
            expected = expected.checked_add(1).ok_or_else(|| {
                Error::new(ErrorCode::CorruptStorage, "mutation log index overflow")
            })?;
            entries.push(entry);
            Ok(())
        })?;
        let compacted_through = persisted_base.unwrap_or(compacted_through);
        Ok(Self {
            frames,
            compacted_through,
            entries,
            request_index,
        })
    }

    #[must_use]
    pub const fn compacted_through(&self) -> Bookmark {
        self.compacted_through
    }

    #[must_use]
    pub fn last_bookmark(&self) -> Bookmark {
        self.entries
            .last()
            .map_or(self.compacted_through, MutationEntry::bookmark)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn append(&mut self, entry: MutationEntry) -> Result<Bookmark> {
        self.append_with_durability(entry, Durability::Sync)
    }

    /// Appends to the operating-system write cache without forcing a durability barrier.
    ///
    /// The standalone eventual-durability writer uses this only after canonical publication. It
    /// preserves the exact framing and in-memory index of [`Self::append`]; a later [`Self::sync`]
    /// establishes the durable prefix.
    pub fn append_buffered(&mut self, entry: MutationEntry) -> Result<Bookmark> {
        self.append_with_durability(entry, Durability::Buffered)
    }

    fn append_with_durability(
        &mut self,
        entry: MutationEntry,
        durability: Durability,
    ) -> Result<Bookmark> {
        entry.verify()?;
        let last = self.last_bookmark();
        let expected = last
            .index
            .checked_add(1)
            .ok_or_else(|| Error::new(ErrorCode::CorruptStorage, "mutation index overflow"))?;
        if entry.index() != expected {
            return Err(Error::invalid_data(
                "mutation index is not the next log index",
            ));
        }
        if entry.term() < last.term {
            return Err(Error::invalid_data(
                "mutation term is older than the log tail",
            ));
        }
        if let Some(request_id) = entry.request_id()
            && self.request_index.contains_key(&request_id)
        {
            return Err(Error::invalid_data(
                "request ID already exists in mutation log",
            ));
        }
        let encoded = entry.encode()?;
        self.frames.append(&encoded, durability)?;
        if let Some(request_id) = entry.request_id() {
            self.request_index.insert(request_id, entry.index());
        }
        let bookmark = entry.bookmark();
        self.entries.push(entry);
        Ok(bookmark)
    }

    /// Appends an ordered group with one durability barrier. Validation and serialization finish
    /// before the file is touched, and the in-memory log is published only after the complete
    /// framed batch is durable.
    pub fn append_batch(&mut self, entries: Vec<MutationEntry>) -> Result<Bookmark> {
        self.append_batch_with_durability(entries, Durability::Sync)
    }

    /// Appends an ordered group without an immediate fsync. Validation and serialization still
    /// finish before the file is touched and the group is published atomically to the log's
    /// in-memory index; [`Self::sync`] later establishes its durable boundary.
    pub fn append_batch_buffered(&mut self, entries: Vec<MutationEntry>) -> Result<Bookmark> {
        self.append_batch_with_durability(entries, Durability::Buffered)
    }

    fn append_batch_with_durability(
        &mut self,
        entries: Vec<MutationEntry>,
        durability: Durability,
    ) -> Result<Bookmark> {
        if entries.is_empty() {
            return Err(Error::invalid_data("mutation batch is empty"));
        }
        let mut expected = self
            .last_bookmark()
            .index
            .checked_add(1)
            .ok_or_else(|| Error::new(ErrorCode::CorruptStorage, "mutation index overflow"))?;
        let mut previous_term = self.last_bookmark().term;
        let mut request_ids = BTreeSet::new();
        let mut encoded = Vec::with_capacity(entries.len());
        for entry in &entries {
            entry.verify()?;
            if entry.index() != expected {
                return Err(Error::invalid_data(
                    "mutation batch is not the next contiguous log suffix",
                ));
            }
            if entry.term() < previous_term {
                return Err(Error::invalid_data("mutation batch term moves backwards"));
            }
            if let Some(request_id) = entry.request_id()
                && (self.request_index.contains_key(&request_id) || !request_ids.insert(request_id))
            {
                return Err(Error::invalid_data(
                    "request ID already exists in mutation log or batch",
                ));
            }
            encoded.push(entry.encode()?);
            previous_term = entry.term();
            expected = expected
                .checked_add(1)
                .ok_or_else(|| Error::new(ErrorCode::CorruptStorage, "mutation index overflow"))?;
        }

        self.frames.append_batch(&encoded, durability)?;
        for entry in &entries {
            if let Some(request_id) = entry.request_id() {
                self.request_index.insert(request_id, entry.index());
            }
        }
        let bookmark = entries
            .last()
            .map(MutationEntry::bookmark)
            .ok_or_else(|| Error::internal("validated mutation batch disappeared"))?;
        self.entries.extend(entries);
        Ok(bookmark)
    }

    pub fn sync(&self) -> Result<()> {
        self.frames.sync()
    }

    #[must_use]
    pub fn lookup(&self, index: u64) -> Option<&MutationEntry> {
        if index <= self.compacted_through.index {
            return None;
        }
        let relative = index
            .checked_sub(self.compacted_through.index)?
            .checked_sub(1)?;
        usize::try_from(relative)
            .ok()
            .and_then(|offset| self.entries.get(offset))
    }

    #[must_use]
    pub fn term_at(&self, index: u64) -> Option<u64> {
        if index == self.compacted_through.index {
            return Some(self.compacted_through.term);
        }
        self.lookup(index).map(MutationEntry::term)
    }

    #[must_use]
    pub fn request_bookmark(&self, request_id: Uuid) -> Option<Bookmark> {
        self.request_index
            .get(&request_id)
            .and_then(|index| self.lookup(*index))
            .map(MutationEntry::bookmark)
    }

    #[must_use]
    pub fn replay_from(&self, first_index: u64) -> Vec<MutationEntry> {
        self.entries
            .iter()
            .filter(|entry| entry.index() >= first_index)
            .cloned()
            .collect()
    }

    pub fn truncate_uncommitted(&mut self, committed_index: u64) -> Result<()> {
        if committed_index < self.compacted_through.index {
            return Err(Error::invalid_data(
                "cannot truncate before compacted checkpoint",
            ));
        }
        if committed_index >= self.last_bookmark().index {
            return Ok(());
        }
        self.entries
            .retain(|entry| entry.index() <= committed_index);
        self.rewrite_entries()
    }

    /// Repair an empty compaction marker that was published without its corresponding snapshot.
    ///
    /// No retained entries may exist: lowering a real suffix base would invent a gap. This is only
    /// for recovery from a caller that compacted an empty WAL after snapshot creation was skipped.
    pub fn repair_empty_compacted_base(&mut self, recovered: Bookmark) -> Result<()> {
        if !self.entries.is_empty() || recovered.index > self.compacted_through.index {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "cannot repair a non-empty or older standalone WAL base",
            ));
        }
        self.compacted_through = recovered;
        self.rewrite_entries()
    }

    /// Realign the log beneath a restored snapshot that is newer than the retained tail.
    ///
    /// Every retained entry must sit at or below the snapshot bookmark: each one is then part of
    /// the restored state, so discarding them and adopting the snapshot as the compacted base
    /// loses nothing. An entry beyond the snapshot is refused — dropping it would lose a write.
    pub fn discard_snapshot_covered_entries(&mut self, recovered: Bookmark) -> Result<()> {
        if self.last_bookmark().index > recovered.index {
            return Err(Error::invalid_data(
                "cannot discard log entries beyond the restored snapshot",
            ));
        }
        if recovered.index < self.compacted_through.index {
            return Err(Error::invalid_data(
                "cannot discard behind the compacted checkpoint",
            ));
        }
        self.entries.clear();
        self.compacted_through = recovered;
        self.rewrite_entries()
    }

    pub fn compact_prefix(&mut self, included: Bookmark) -> Result<()> {
        if included.index < self.compacted_through.index
            || included.index > self.last_bookmark().index
        {
            return Err(Error::invalid_data(
                "checkpoint bookmark is outside mutation log",
            ));
        }
        let actual_term = self.term_at(included.index).ok_or_else(|| {
            Error::new(
                ErrorCode::CorruptStorage,
                "checkpoint term is absent from mutation log",
            )
        })?;
        if actual_term != included.term {
            return Err(Error::invalid_data(
                "checkpoint term does not match mutation log",
            ));
        }
        self.entries.retain(|entry| entry.index() > included.index);
        self.compacted_through = included;
        self.rewrite_entries()
    }

    fn rewrite_entries(&mut self) -> Result<()> {
        let marker = encode_compaction_marker(self.compacted_through)?;
        self.frames.rewrite_streaming(
            std::iter::once(Ok(marker)).chain(self.entries.iter().map(MutationEntry::encode)),
        )?;
        self.request_index.clear();
        for entry in &self.entries {
            if let Some(request_id) = entry.request_id() {
                self.request_index.insert(request_id, entry.index());
            }
        }
        Ok(())
    }
}

fn encode_compaction_marker(bookmark: Bookmark) -> Result<Vec<u8>> {
    let mut encoded = vec![0];
    encoded.extend(
        postcard::to_stdvec(&bookmark)
            .map_err(|error| Error::new(ErrorCode::InvalidData, error.to_string()))?,
    );
    Ok(encoded)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(index: u64) -> Result<MutationEntry> {
        MutationEntry::new(
            1,
            index,
            MutationKind::Project,
            Some(ProjectId::random()),
            Some(Uuid::new_v4()),
            1,
            vec![1],
        )
    }

    #[test]
    fn mutation_entry_clones_share_payload_and_preserve_wire_encoding() -> Result<()> {
        #[derive(Serialize)]
        struct OwnedEntry<'a> {
            body: &'a MutationBody,
            checksum: [u8; 32],
        }

        let entry = MutationEntry::new(
            7,
            9,
            MutationKind::Graph,
            Some(ProjectId::random()),
            Some(Uuid::new_v4()),
            42,
            vec![3; 1024 * 1024],
        )?;
        let cloned = entry.clone();
        assert!(Arc::ptr_eq(&entry.body, &cloned.body));
        assert_eq!(
            entry.encode()?,
            postcard::to_stdvec(&OwnedEntry {
                body: &entry.body,
                checksum: entry.checksum,
            })
            .map_err(|error| Error::new(ErrorCode::InvalidData, error.to_string()))?
        );
        Ok(())
    }

    #[test]
    fn durable_batch_is_contiguous_atomic_and_reopens_exactly_once() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("batched.wal");
        let mut log = DurableLog::open(&path, 1024, Bookmark::default())?;
        let batch = (1..=4).map(entry).collect::<Result<Vec<_>>>()?;
        assert_eq!(log.append_batch(batch)?, Bookmark { term: 1, index: 4 });
        assert_eq!(log.len(), 4);

        let invalid = vec![entry(5)?, entry(7)?];
        assert!(log.append_batch(invalid).is_err());
        assert_eq!(log.len(), 4);
        drop(log);

        let reopened = DurableLog::open(&path, 1024, Bookmark::default())?;
        assert_eq!(reopened.len(), 4);
        assert_eq!(reopened.last_bookmark(), Bookmark { term: 1, index: 4 });
        Ok(())
    }

    #[test]
    fn snapshot_covered_discard_realigns_the_base_and_refuses_to_drop_newer_entries() -> Result<()>
    {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("standalone.wal");
        let mut log = DurableLog::open(&path, 1024, Bookmark::default())?;
        for index in 1..=3 {
            log.append(entry(index)?)?;
        }

        // Entry 3 is beyond the snapshot: discarding it would lose a write.
        assert!(
            log.discard_snapshot_covered_entries(Bookmark { term: 1, index: 2 })
                .is_err()
        );
        assert_eq!(log.len(), 3);

        log.discard_snapshot_covered_entries(Bookmark { term: 1, index: 5 })?;
        assert_eq!(log.compacted_through(), Bookmark { term: 1, index: 5 });
        assert!(log.is_empty());
        // The realigned base accepts the snapshot's successor and survives reopen.
        log.append(entry(6)?)?;
        drop(log);

        let reopened = DurableLog::open(&path, 1024, Bookmark::default())?;
        assert_eq!(reopened.compacted_through(), Bookmark { term: 1, index: 5 });
        assert_eq!(reopened.last_bookmark(), Bookmark { term: 1, index: 6 });
        Ok(())
    }

    #[test]
    fn compacted_bookmark_and_suffix_reopen_atomically() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("standalone.wal");
        let mut log = DurableLog::open(&path, 1024, Bookmark::default())?;
        for index in 1..=4 {
            log.append(entry(index)?)?;
        }
        log.compact_prefix(Bookmark { term: 1, index: 3 })?;
        drop(log);

        let reopened = DurableLog::open(&path, 1024, Bookmark::default())?;
        assert_eq!(reopened.compacted_through(), Bookmark { term: 1, index: 3 });
        assert_eq!(reopened.last_bookmark(), Bookmark { term: 1, index: 4 });
        assert_eq!(reopened.replay_from(4).len(), 1);
        Ok(())
    }
}

fn body_checksum(body: &MutationBody) -> Result<[u8; 32]> {
    struct HashWriter(blake3::Hasher);

    impl Write for HashWriter {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            self.0.update(bytes);
            Ok(bytes.len())
        }

        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    let writer = postcard::to_io(body, HashWriter(blake3::Hasher::new()))
        .map_err(|error| Error::new(ErrorCode::InvalidData, error.to_string()))?;
    Ok(*writer.0.finalize().as_bytes())
}
