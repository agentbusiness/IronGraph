use std::{
    collections::BTreeMap,
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex},
};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use irongraph_types::{Error, ErrorCode, ProjectId, Result};

use super::{
    Durability, FramedFile,
    fs::{sync_durable, sync_parent},
};

const SEGMENT_MAGIC: [u8; 8] = *b"IGSEG001";

/// Canonical segment family. Entity kind and value type live in the enclosing family.
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

/// One typed record in an immutable segment.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentRecord {
    pub kind: u16,
    pub payload: Vec<u8>,
}

/// Content-addressed immutable segment descriptor saved in a checkpoint.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SegmentDescriptor {
    pub family: SegmentFamily,
    pub project_id: Option<ProjectId>,
    pub file_name: String,
    pub bytes: u64,
    pub checksum: [u8; 32],
}

/// Stable byte extent of one record inside an immutable segment.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SegmentRecordLocation {
    pub offset: u64,
    pub bytes: u64,
}

/// Immutable content-addressed segment store.
#[derive(Clone, Debug)]
pub struct SegmentStore {
    directory: PathBuf,
    max_record_bytes: usize,
    runtime: Arc<SegmentRuntime>,
}

#[derive(Debug, Default)]
struct SegmentRuntime {
    /// Serializes immutable publication/import and physical reclamation. Readers use pins, so
    /// the lock is held only while namespace membership can change.
    files: Mutex<()>,
    pins: Mutex<BTreeMap<String, usize>>,
}

/// Ephemeral read/checkpoint pin for one immutable segment. Physical reclamation synchronizes
/// with this guard, so callers may release the database publication lock before doing file IO.
#[derive(Debug)]
pub struct SegmentPin {
    store: SegmentStore,
    descriptor: SegmentDescriptor,
}

impl SegmentPin {
    pub fn link_raw_to(&self, destination: &Path) -> Result<()> {
        let _files = self
            .store
            .runtime
            .files
            .lock()
            .map_err(|_| Error::internal("segment file registry is poisoned"))?;
        let source = self.store.path_for(&self.descriptor)?;
        if fs::metadata(&source)?.len() != self.descriptor.bytes {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "pinned segment length changed before snapshot linking",
            ));
        }
        // Broker payload segments may be published without an acknowledgement-path fsync because
        // their complete bytes remain replayable from the eventual-durability WAL. A snapshot is
        // the boundary that can retire that WAL suffix, so make the source inode durable here
        // before linking it into the snapshot attachment directory.
        sync_durable(&File::open(&source)?)?;
        fs::hard_link(source, destination)?;
        sync_parent(destination)
    }
}

impl Drop for SegmentPin {
    fn drop(&mut self) {
        let Ok(mut pins) = self.store.runtime.pins.lock() else {
            return;
        };
        let Some(count) = pins.get_mut(&self.descriptor.file_name) else {
            return;
        };
        *count = count.saturating_sub(1);
        if *count == 0 {
            pins.remove(&self.descriptor.file_name);
        }
    }
}

impl SegmentStore {
    pub fn open(root: impl AsRef<Path>, max_record_bytes: usize) -> Result<Self> {
        if max_record_bytes == 0 {
            return Err(Error::invalid_data("segment record limit must be non-zero"));
        }
        let directory = root.as_ref().join("segments");
        fs::create_dir_all(&directory)?;
        let mut removed_ephemeral = false;
        for entry in fs::read_dir(&directory)? {
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                continue;
            }
            let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
                continue;
            };
            let abandoned_temporary = name.starts_with('.') && name.ends_with(".tmp");
            if abandoned_temporary {
                fs::remove_file(entry.path())?;
                removed_ephemeral = true;
            }
        }
        if removed_ephemeral {
            sync_directory_path(&directory)?;
        }
        Ok(Self {
            directory,
            max_record_bytes,
            runtime: Arc::new(SegmentRuntime::default()),
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
                .ok_or_else(|| Error::internal("immutable segment record index disappeared"))
        })
    }

    /// Publishes a deterministic record batch while retaining and encoding one record at a time.
    pub fn write_immutable_streaming(
        &self,
        family: SegmentFamily,
        project_id: Option<ProjectId>,
        record_count: usize,
        record_at: impl FnMut(usize) -> Result<SegmentRecord>,
    ) -> Result<SegmentDescriptor> {
        self.write_immutable_streaming_indexed(family, project_id, record_count, record_at)
            .map(|(descriptor, _)| descriptor)
    }

    /// Encodes and durably writes each immutable record exactly once while collecting stable
    /// byte extents for direct record reads. The content address is calculated from the exact
    /// framing bytes as they are appended; no completed-file re-read is required.
    pub fn write_immutable_streaming_indexed(
        &self,
        family: SegmentFamily,
        project_id: Option<ProjectId>,
        record_count: usize,
        record_at: impl FnMut(usize) -> Result<SegmentRecord>,
    ) -> Result<(SegmentDescriptor, Vec<SegmentRecordLocation>)> {
        self.write_immutable_streaming_indexed_with_durability(
            family,
            project_id,
            record_count,
            Durability::Sync,
            record_at,
        )
    }

    /// Atomically publishes a replayable derived segment without forcing storage durability.
    ///
    /// This is intentionally narrower than [`Self::write_immutable_streaming_indexed`]: callers
    /// must retain a durable source (the WAL) from which a crash-lost segment can be rebuilt. A
    /// later snapshot attachment calls [`SegmentPin::link_raw_to`], which establishes durability
    /// before that WAL suffix may be compacted.
    pub fn write_replayable_immutable_streaming_indexed(
        &self,
        family: SegmentFamily,
        project_id: Option<ProjectId>,
        record_count: usize,
        record_at: impl FnMut(usize) -> Result<SegmentRecord>,
    ) -> Result<(SegmentDescriptor, Vec<SegmentRecordLocation>)> {
        self.write_immutable_streaming_indexed_with_durability(
            family,
            project_id,
            record_count,
            Durability::Buffered,
            record_at,
        )
    }

    fn write_immutable_streaming_indexed_with_durability(
        &self,
        family: SegmentFamily,
        project_id: Option<ProjectId>,
        record_count: usize,
        durability: Durability,
        mut record_at: impl FnMut(usize) -> Result<SegmentRecord>,
    ) -> Result<(SegmentDescriptor, Vec<SegmentRecordLocation>)> {
        if family_requires_project(family) != project_id.is_some() {
            return Err(Error::invalid_data(
                "segment project scope does not match its family",
            ));
        }

        let temporary = self
            .directory
            .join(format!(".{}.segment.tmp", Uuid::new_v4()));
        let result = (|| -> Result<(SegmentDescriptor, Vec<SegmentRecordLocation>)> {
            let mut writer =
                FramedFile::create_staged(&temporary, SEGMENT_MAGIC, self.max_record_bytes)?;
            let mut bytes = 0_u64;
            let mut hasher = blake3::Hasher::new();
            let mut locations = Vec::with_capacity(record_count);
            const WRITE_BATCH_RECORDS: usize = 8_192;
            const WRITE_BATCH_BYTES: usize = 1024 * 1024;
            let mut encoded_batch = Vec::with_capacity(record_count.min(WRITE_BATCH_RECORDS));
            let mut encoded_batch_bytes = 0_usize;
            for index in 0..record_count {
                let record = record_at(index)?;
                let encoded = postcard::to_stdvec(&record)
                    .map_err(|error| Error::new(ErrorCode::InvalidData, error.to_string()))?;
                if encoded.len() > self.max_record_bytes {
                    return Err(Error::invalid_data(
                        "segment record exceeds configured limit",
                    ));
                }
                FramedFile::update_content_hash(SEGMENT_MAGIC, &encoded, &mut hasher)?;
                encoded_batch_bytes = encoded_batch_bytes
                    .checked_add(encoded.len())
                    .ok_or_else(|| Error::internal("immutable segment batch length overflow"))?;
                encoded_batch.push(encoded);
                if encoded_batch.len() == WRITE_BATCH_RECORDS
                    || encoded_batch_bytes >= WRITE_BATCH_BYTES
                    || index + 1 == record_count
                {
                    let offsets = writer.append_batch(&encoded_batch, Durability::Buffered)?;
                    for (offset, encoded) in offsets.into_iter().zip(&encoded_batch) {
                        let record_bytes = FramedFile::physical_record_bytes(encoded.len())?;
                        bytes = bytes
                            .checked_add(record_bytes)
                            .ok_or_else(|| Error::internal("immutable segment length overflow"))?;
                        locations.push(SegmentRecordLocation {
                            offset,
                            bytes: record_bytes,
                        });
                    }
                    encoded_batch.clear();
                    encoded_batch_bytes = 0;
                }
            }
            if durability == Durability::Sync {
                writer.sync()?;
            }
            let checksum = *hasher.finalize().as_bytes();
            let file_name = format!("{}.segment", hex::encode(checksum));
            let descriptor = SegmentDescriptor {
                family,
                project_id,
                file_name: file_name.clone(),
                bytes,
                checksum,
            };
            let _files = self
                .runtime
                .files
                .lock()
                .map_err(|_| Error::internal("segment file registry is poisoned"))?;
            let final_path = self.directory.join(&file_name);
            if final_path.exists() {
                if self.validate(&descriptor).is_ok() {
                    fs::remove_file(&temporary)?;
                    return Ok((descriptor, locations));
                }
                fs::remove_file(&final_path)?;
                sync_directory_path(&self.directory)?;
            }
            match fs::hard_link(&temporary, &final_path) {
                Ok(()) => {
                    fs::remove_file(&temporary)?;
                    if durability == Durability::Sync {
                        sync_parent(&final_path)?;
                    }
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                    fs::remove_file(&temporary)?;
                    self.validate(&descriptor)?;
                }
                Err(error) => return Err(error.into()),
            }
            Ok((descriptor, locations))
        })();
        if result.is_err() {
            let _ignored = fs::remove_file(&temporary);
        }
        result
    }

    /// Pins an immutable segment against local physical reclamation. Content verification is
    /// deliberately deferred to the subsequent read/copy so the database publication lock is
    /// held only for a bounded metadata check.
    pub fn pin(&self, descriptor: &SegmentDescriptor) -> Result<SegmentPin> {
        let _files = self
            .runtime
            .files
            .lock()
            .map_err(|_| Error::internal("segment file registry is poisoned"))?;
        let mut pins = self
            .runtime
            .pins
            .lock()
            .map_err(|_| Error::internal("segment pin registry is poisoned"))?;
        validate_file_name(&descriptor.file_name)?;
        if descriptor.file_name != format!("{}.segment", hex::encode(descriptor.checksum))
            || family_requires_project(descriptor.family) != descriptor.project_id.is_some()
            || fs::metadata(self.path_for(descriptor)?)?.len() != descriptor.bytes
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "segment pin metadata differs from its descriptor",
            ));
        }
        *pins.entry(descriptor.file_name.clone()).or_default() += 1;
        Ok(SegmentPin {
            store: self.clone(),
            descriptor: descriptor.clone(),
        })
    }

    /// Reclaims exact immutable files no longer referenced by canonical state. A pinned reader
    /// or checkpoint makes the candidate a harmless no-op; the next maintenance pass retries.
    pub fn reclaim_unreferenced(
        &self,
        candidates: &[SegmentDescriptor],
        live_file_names: &std::collections::BTreeSet<String>,
    ) -> Result<Vec<SegmentDescriptor>> {
        let _files = self
            .runtime
            .files
            .lock()
            .map_err(|_| Error::internal("segment file registry is poisoned"))?;
        let pins = self
            .runtime
            .pins
            .lock()
            .map_err(|_| Error::internal("segment pin registry is poisoned"))?;
        let mut reclaimed = Vec::new();
        for descriptor in candidates {
            validate_file_name(&descriptor.file_name)?;
            if live_file_names.contains(&descriptor.file_name)
                || pins.get(&descriptor.file_name).copied().unwrap_or(0) != 0
            {
                continue;
            }
            let path = self.path_for(descriptor)?;
            match fs::remove_file(&path) {
                Ok(()) => reclaimed.push(descriptor.clone()),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    reclaimed.push(descriptor.clone());
                }
                Err(error) => return Err(error.into()),
            }
        }
        if !reclaimed.is_empty() {
            sync_directory_path(&self.directory)?;
        }
        Ok(reclaimed)
    }

    /// Discovers crash-leftover immutable files and reclaims a bounded batch that is neither
    /// named by canonical state nor pinned by an in-flight reader/reservation. Discovery and
    /// unlink run under the same namespace lock as publication/import, so a file cannot become
    /// visible between the liveness decision and removal. Files whose names are not exact
    /// content addresses are deliberately left untouched for operator inspection.
    pub fn reclaim_discovered_unreferenced(
        &self,
        live_file_names: &std::collections::BTreeSet<String>,
        maximum_files: usize,
    ) -> Result<Vec<String>> {
        if maximum_files == 0 {
            return Err(Error::invalid_data(
                "orphan segment reclamation bound must be non-zero",
            ));
        }
        let _files = self
            .runtime
            .files
            .lock()
            .map_err(|_| Error::internal("segment file registry is poisoned"))?;
        let pins = self
            .runtime
            .pins
            .lock()
            .map_err(|_| Error::internal("segment pin registry is poisoned"))?;
        let mut candidates = Vec::new();
        for entry in fs::read_dir(&self.directory)? {
            let entry = entry?;
            let file_type = entry.file_type()?;
            if !file_type.is_file() || file_type.is_symlink() {
                continue;
            }
            let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
                continue;
            };
            if validate_file_name(&name).is_err()
                || live_file_names.contains(&name)
                || pins.get(&name).copied().unwrap_or(0) != 0
            {
                continue;
            }
            candidates.push(name);
        }
        candidates.sort_unstable();
        candidates.truncate(maximum_files);

        let mut reclaimed = Vec::with_capacity(candidates.len());
        for name in candidates {
            match fs::remove_file(self.directory.join(&name)) {
                Ok(()) => reclaimed.push(name),
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                    // No other publication/reclamation path can run under `_files`; a missing
                    // file can only be a harmless external deletion and is already reclaimed.
                    reclaimed.push(name);
                }
                Err(error) => return Err(error.into()),
            }
        }
        if !reclaimed.is_empty() {
            sync_directory_path(&self.directory)?;
        }
        Ok(reclaimed)
    }

    /// Lists immutable content-addressed files without assigning a semantic segment family.
    #[doc(hidden)]
    pub fn immutable_file_names(&self) -> Result<Vec<String>> {
        let mut names = Vec::new();
        for entry in fs::read_dir(&self.directory)? {
            let entry = entry?;
            if !entry.file_type()?.is_file() {
                continue;
            }
            let Some(name) = entry.file_name().to_str().map(str::to_owned) else {
                continue;
            };
            if validate_file_name(&name).is_ok() {
                names.push(name);
            }
        }
        names.sort();
        Ok(names)
    }

    pub fn read(&self, descriptor: &SegmentDescriptor) -> Result<Vec<SegmentRecord>> {
        let mut records = Vec::new();
        self.try_for_each_record(descriptor, |record| {
            records.push(record);
            Ok(())
        })?;
        Ok(records)
    }

    /// Verifies and decodes one immutable record at a time without retaining the complete
    /// segment in memory.
    pub fn try_for_each_record(
        &self,
        descriptor: &SegmentDescriptor,
        mut visitor: impl FnMut(SegmentRecord) -> Result<()>,
    ) -> Result<()> {
        self.validate(descriptor)?;
        let path = self.path_for(descriptor)?;
        let mut framed = FramedFile::open_strict(path, SEGMENT_MAGIC, self.max_record_bytes)?;
        framed.try_for_each(|record| {
            let decoded = postcard::from_bytes(&record.payload)
                .map_err(|error| Error::new(ErrorCode::CorruptStorage, error.to_string()))?;
            visitor(decoded)
        })
    }

    /// Reads one canonically indexed record without hashing or scanning the rest of its segment.
    /// Immutable files are fully verified at publication/import/recovery; the selected frame is
    /// independently protected by its header and payload CRC on every direct read.
    pub fn read_record_at(
        &self,
        descriptor: &SegmentDescriptor,
        location: SegmentRecordLocation,
    ) -> Result<SegmentRecord> {
        validate_file_name(&descriptor.file_name)?;
        if descriptor.file_name != format!("{}.segment", hex::encode(descriptor.checksum))
            || family_requires_project(descriptor.family) != descriptor.project_id.is_some()
            || fs::metadata(self.path_for(descriptor)?)?.len() != descriptor.bytes
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "indexed segment metadata differs from its descriptor",
            ));
        }
        let framed = FramedFile::read_record_at(
            self.path_for(descriptor)?,
            SEGMENT_MAGIC,
            self.max_record_bytes,
            location.offset,
            location.bytes,
        )?;
        postcard::from_bytes(&framed.payload)
            .map_err(|error| Error::new(ErrorCode::CorruptStorage, error.to_string()))
    }

    pub fn validate(&self, descriptor: &SegmentDescriptor) -> Result<()> {
        validate_file_name(&descriptor.file_name)?;
        let expected_name = format!("{}.segment", hex::encode(descriptor.checksum));
        if descriptor.file_name != expected_name {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "segment file name does not match its content address",
            ));
        }
        if family_requires_project(descriptor.family) != descriptor.project_id.is_some() {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "segment project scope does not match family",
            ));
        }
        let path = self.directory.join(&descriptor.file_name);
        let (bytes, checksum) = hash_file(&path)?;
        if bytes != descriptor.bytes || checksum != descriptor.checksum {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "checkpoint segment size or checksum mismatch",
            ));
        }
        Ok(())
    }

    /// Streams one verified raw segment in bounded chunks without loading it twice or projecting
    /// its typed records. The raw framing is the content-addressed snapshot attachment unit.
    pub fn for_each_raw_chunk(
        &self,
        descriptor: &SegmentDescriptor,
        chunk_bytes: usize,
        mut consume: impl FnMut(u32, u32, Vec<u8>) -> Result<()>,
    ) -> Result<()> {
        if chunk_bytes == 0 {
            return Err(Error::invalid_data("segment chunk size must be non-zero"));
        }
        self.validate(descriptor)?;
        let count_u64 = descriptor.bytes.div_ceil(chunk_bytes as u64);
        let count = u32::try_from(count_u64).map_err(|_| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "segment requires too many snapshot chunks",
            )
        })?;
        let mut file = File::open(self.path_for(descriptor)?)?;
        let mut buffer = vec![0_u8; chunk_bytes];
        for index in 0..count {
            let mut filled = 0_usize;
            while filled < buffer.len() {
                let read = file.read(&mut buffer[filled..])?;
                if read == 0 {
                    break;
                }
                filled = filled
                    .checked_add(read)
                    .ok_or_else(|| Error::internal("segment chunk length overflow"))?;
            }
            if filled == 0 {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "segment ended before its declared byte length",
                ));
            }
            consume(index, count, buffer[..filled].to_vec())?;
        }
        let mut trailing = [0_u8; 1];
        if file.read(&mut trailing)? != 0 {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "segment exceeds its declared byte length",
            ));
        }
        Ok(())
    }

    pub fn import_raw_from(
        &self,
        descriptor: &SegmentDescriptor,
        reader: &mut impl Read,
        chunk_bytes: usize,
    ) -> Result<()> {
        if chunk_bytes == 0 {
            return Err(Error::invalid_data("segment import chunk must be non-zero"));
        }
        validate_file_name(&descriptor.file_name)?;
        if descriptor.file_name != format!("{}.segment", hex::encode(descriptor.checksum))
            || family_requires_project(descriptor.family) != descriptor.project_id.is_some()
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "imported segment descriptor is invalid",
            ));
        }
        let _files = self
            .runtime
            .files
            .lock()
            .map_err(|_| Error::internal("segment file registry is poisoned"))?;
        let destination = self.path_for(descriptor)?;
        if destination.exists() {
            if self.validate(descriptor).is_ok() {
                let mut discard = vec![0_u8; chunk_bytes];
                let mut remaining = descriptor.bytes;
                while remaining > 0 {
                    let requested = usize::try_from(remaining.min(discard.len() as u64))
                        .map_err(|_| Error::internal("segment import length overflow"))?;
                    reader.read_exact(&mut discard[..requested])?;
                    remaining -= requested as u64;
                }
                return Ok(());
            }
            fs::remove_file(&destination)?;
            sync_directory_path(&self.directory)?;
        }
        let temporary = self
            .directory
            .join(format!(".{}.segment.tmp", Uuid::new_v4()));
        let result = (|| -> Result<()> {
            let mut output = OpenOptions::new()
                .write(true)
                .create_new(true)
                .open(&temporary)?;
            let mut hasher = blake3::Hasher::new();
            let mut remaining = descriptor.bytes;
            let mut buffer = vec![0_u8; chunk_bytes];
            while remaining > 0 {
                let requested = usize::try_from(remaining.min(buffer.len() as u64))
                    .map_err(|_| Error::internal("segment import length overflow"))?;
                reader.read_exact(&mut buffer[..requested])?;
                output.write_all(&buffer[..requested])?;
                hasher.update(&buffer[..requested]);
                remaining -= requested as u64;
            }
            if *hasher.finalize().as_bytes() != descriptor.checksum {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "imported segment checksum mismatch",
                ));
            }
            sync_durable(&output)?;
            fs::rename(&temporary, &destination)?;
            sync_parent(&destination)?;
            Ok(())
        })();
        if result.is_err() {
            let _ignored = fs::remove_file(&temporary);
        }
        result
    }

    fn path_for(&self, descriptor: &SegmentDescriptor) -> Result<PathBuf> {
        validate_file_name(&descriptor.file_name)?;
        Ok(self.directory.join(&descriptor.file_name))
    }
}

fn family_requires_project(family: SegmentFamily) -> bool {
    !matches!(family, SegmentFamily::Security | SegmentFamily::Catalog)
}

fn validate_file_name(file_name: &str) -> Result<()> {
    let expected_len = 64_usize
        .checked_add(".segment".len())
        .ok_or_else(|| Error::internal("segment name length overflow"))?;
    if file_name.len() != expected_len
        || !file_name.ends_with(".segment")
        || !file_name[..64].bytes().all(|byte| byte.is_ascii_hexdigit())
    {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "invalid segment file name",
        ));
    }
    Ok(())
}

fn hash_file(path: &Path) -> Result<(u64, [u8; 32])> {
    let mut file = File::open(path)?;
    let bytes = file.metadata()?.len();
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0_u8; 16 * 1024];
    loop {
        let read = file.read(&mut buffer)?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok((bytes, *hasher.finalize().as_bytes()))
}

#[cfg(unix)]
fn sync_directory_path(path: &Path) -> Result<()> {
    sync_durable(&File::open(path)?)
}

#[cfg(not(unix))]
fn sync_directory_path(_path: &Path) -> Result<()> {
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::{cell::Cell, collections::BTreeSet};

    use tempfile::tempdir;

    use super::*;

    #[test]
    fn read_pin_fences_physical_reclamation_until_reader_release() -> Result<()> {
        let directory = tempdir()?;
        let store = SegmentStore::open(directory.path(), 1024)?;
        let project = ProjectId::random();
        let descriptor = store.write_immutable(
            SegmentFamily::BrokerPayload,
            Some(project),
            &[SegmentRecord {
                kind: 1,
                payload: b"payload".to_vec(),
            }],
        )?;
        let pin = store.pin(&descriptor)?;
        assert!(
            store
                .reclaim_unreferenced(std::slice::from_ref(&descriptor), &BTreeSet::new())?
                .is_empty()
        );
        store.validate(&descriptor)?;

        drop(pin);
        assert_eq!(
            store.reclaim_unreferenced(std::slice::from_ref(&descriptor), &BTreeSet::new())?,
            vec![descriptor.clone()]
        );
        assert!(store.validate(&descriptor).is_err());
        Ok(())
    }

    #[test]
    fn streaming_publication_encodes_each_record_once_and_returns_direct_extents() -> Result<()> {
        let directory = tempdir()?;
        let store = SegmentStore::open(directory.path(), 1024)?;
        let project = ProjectId::random();
        let calls = Cell::new(0_usize);
        let (descriptor, locations) = store.write_immutable_streaming_indexed(
            SegmentFamily::BrokerPayload,
            Some(project),
            3,
            |index| {
                calls.set(calls.get() + 1);
                Ok(SegmentRecord {
                    kind: 1,
                    payload: vec![index as u8; index + 1],
                })
            },
        )?;
        assert_eq!(calls.get(), 3);
        assert_eq!(locations.len(), 3);
        assert_eq!(locations[0].offset, 0);
        assert_eq!(locations[2].offset + locations[2].bytes, descriptor.bytes);
        for (index, location) in locations.into_iter().enumerate() {
            assert_eq!(
                store.read_record_at(&descriptor, location)?,
                SegmentRecord {
                    kind: 1,
                    payload: vec![index as u8; index + 1],
                }
            );
        }
        Ok(())
    }

    #[test]
    fn replayable_publication_becomes_snapshot_attachment_without_reencoding() -> Result<()> {
        let directory = tempdir()?;
        let store = SegmentStore::open(directory.path(), 1024)?;
        let project = ProjectId::random();
        let calls = Cell::new(0_usize);
        let (descriptor, locations) = store.write_replayable_immutable_streaming_indexed(
            SegmentFamily::BrokerPayload,
            Some(project),
            1,
            |_| {
                calls.set(calls.get() + 1);
                Ok(SegmentRecord {
                    kind: 1,
                    payload: b"eventual payload".to_vec(),
                })
            },
        )?;
        assert_eq!(calls.get(), 1, "the publish delta encodes its record once");
        assert_eq!(
            store.read_record_at(&descriptor, locations[0])?.payload,
            b"eventual payload"
        );

        let attachment_directory = directory.path().join("snapshot-attachments");
        fs::create_dir(&attachment_directory)?;
        let attachment = attachment_directory.join("payload.segment");
        store.pin(&descriptor)?.link_raw_to(&attachment)?;
        assert_eq!(fs::metadata(attachment)?.len(), descriptor.bytes);
        assert_eq!(
            calls.get(),
            1,
            "snapshot attachment reuses the segment bytes"
        );
        Ok(())
    }

    #[test]
    fn live_file_name_prevents_cross_family_content_reclamation() -> Result<()> {
        let directory = tempdir()?;
        let store = SegmentStore::open(directory.path(), 1024)?;
        let project = ProjectId::random();
        let descriptor = store.write_immutable(
            SegmentFamily::BrokerPayload,
            Some(project),
            &[SegmentRecord {
                kind: 1,
                payload: b"shared".to_vec(),
            }],
        )?;
        let live = BTreeSet::from([descriptor.file_name.clone()]);
        assert!(
            store
                .reclaim_unreferenced(std::slice::from_ref(&descriptor), &live)?
                .is_empty()
        );
        store.validate(&descriptor)
    }

    #[test]
    fn orphan_discovery_is_bounded_and_respects_live_and_inflight_pins() -> Result<()> {
        let directory = tempdir()?;
        let store = SegmentStore::open(directory.path(), 1024)?;
        let project = ProjectId::random();
        let live = store.write_immutable(
            SegmentFamily::BrokerPayload,
            Some(project),
            &[SegmentRecord {
                kind: 1,
                payload: b"live".to_vec(),
            }],
        )?;
        let pinned = store.write_immutable(
            SegmentFamily::BrokerPayload,
            Some(project),
            &[SegmentRecord {
                kind: 1,
                payload: b"pinned".to_vec(),
            }],
        )?;
        let orphan = store.write_immutable(
            SegmentFamily::BrokerPayload,
            Some(project),
            &[SegmentRecord {
                kind: 1,
                payload: b"orphan".to_vec(),
            }],
        )?;
        let pin = store.pin(&pinned)?;
        let unknown = store.directory.join("not-a-content-address.segment");
        fs::write(&unknown, b"ambiguous")?;

        let live_names = BTreeSet::from([live.file_name.clone()]);
        assert_eq!(
            store.reclaim_discovered_unreferenced(&live_names, 1)?,
            vec![orphan.file_name.clone()]
        );
        store.validate(&live)?;
        store.validate(&pinned)?;
        assert!(unknown.exists());

        drop(pin);
        assert_eq!(
            store.reclaim_discovered_unreferenced(&live_names, 1)?,
            vec![pinned.file_name.clone()]
        );
        assert_eq!(store.immutable_file_names()?, vec![live.file_name]);
        assert!(unknown.exists());
        Ok(())
    }

    #[test]
    fn reopen_removes_abandoned_segment_temporaries() -> Result<()> {
        let directory = tempdir()?;
        let segments = directory.path().join("segments");
        fs::create_dir_all(&segments)?;
        let temporary = segments.join(".abandoned.segment.00000000.tmp");
        fs::write(&temporary, b"partial")?;

        let _store = SegmentStore::open(directory.path(), 1024)?;
        assert!(!temporary.exists());
        Ok(())
    }

    #[test]
    fn deterministic_publication_repairs_a_corrupt_existing_content_address() -> Result<()> {
        let directory = tempdir()?;
        let store = SegmentStore::open(directory.path(), 1024)?;
        let project = ProjectId::random();
        let records = [SegmentRecord {
            kind: 1,
            payload: b"recoverable".to_vec(),
        }];
        let descriptor =
            store.write_immutable(SegmentFamily::BrokerPayload, Some(project), &records)?;
        let path = store.path_for(&descriptor)?;
        let mut bytes = fs::read(&path)?;
        let final_index = bytes.len() - 1;
        bytes[final_index] ^= 0xff;
        fs::write(&path, bytes)?;
        assert!(store.validate(&descriptor).is_err());

        let repaired =
            store.write_immutable(SegmentFamily::BrokerPayload, Some(project), &records)?;
        assert_eq!(repaired, descriptor);
        store.validate(&descriptor)?;
        assert_eq!(store.read(&descriptor)?, records);
        Ok(())
    }

    #[test]
    fn verified_raw_import_replaces_a_corrupt_existing_segment() -> Result<()> {
        let directory = tempdir()?;
        let store = SegmentStore::open(directory.path(), 1024)?;
        let project = ProjectId::random();
        let descriptor = store.write_immutable(
            SegmentFamily::BrokerPayload,
            Some(project),
            &[SegmentRecord {
                kind: 1,
                payload: b"checkpoint source".to_vec(),
            }],
        )?;
        let path = store.path_for(&descriptor)?;
        let canonical = fs::read(&path)?;
        let mut corrupt = canonical.clone();
        let middle = corrupt.len() / 2;
        corrupt[middle] ^= 0x55;
        fs::write(&path, corrupt)?;

        store.import_raw_from(&descriptor, &mut canonical.as_slice(), 7)?;
        store.validate(&descriptor)?;
        assert_eq!(fs::read(path)?, canonical);
        Ok(())
    }
}
