use std::{
    fs,
    path::{Path, PathBuf},
};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use irongraph_types::{Bookmark, Error, ErrorCode, Result};

use super::{SegmentDescriptor, SegmentStore, atomic_write, read_bounded};

const CURRENT_POINTER: &str = "CURRENT";
const PREVIOUS_POINTER: &str = "PREVIOUS";
const MANIFEST_FORMAT: u16 = 2;
const MAX_POINTER_BYTES: usize = 256;

/// Content hash of a complete checkpoint manifest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ManifestId(pub [u8; 32]);

impl std::fmt::Display for ManifestId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", hex::encode(self.0))
    }
}

/// Complete canonical checkpoint state at one applied local write position.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointManifest {
    pub format_version: u16,
    pub store_id: Uuid,
    pub included: Bookmark,
    pub segments: Vec<SegmentDescriptor>,
}

impl CheckpointManifest {
    #[must_use]
    pub fn new(store_id: Uuid, included: Bookmark, mut segments: Vec<SegmentDescriptor>) -> Self {
        segments.sort();
        Self {
            format_version: MANIFEST_FORMAT,
            store_id,
            included,
            segments,
        }
    }

    pub fn validate_shape(&self) -> Result<()> {
        if self.format_version != MANIFEST_FORMAT {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "unsupported checkpoint format version",
            ));
        }
        if self.included.index == 0 || self.included.term == 0 {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "checkpoint bookmark must be non-zero",
            ));
        }
        if self.segments.windows(2).any(|pair| pair[0] >= pair[1]) {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "checkpoint segments are duplicated or not canonically ordered",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct ManifestEnvelope {
    payload: Vec<u8>,
    checksum: [u8; 32],
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
struct PointerEnvelope {
    id: ManifestId,
    checksum: [u8; 32],
}

/// Which durable pointer supplied recovery state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckpointSource {
    Current,
    Previous,
}

/// Verified checkpoint selected during startup.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RecoveredCheckpoint {
    pub id: ManifestId,
    pub manifest: CheckpointManifest,
    pub source: CheckpointSource,
}

/// Atomic current/previous checkpoint publisher with complete segment verification.
pub struct CheckpointStore {
    directory: PathBuf,
    manifests: PathBuf,
    segment_store: SegmentStore,
    max_manifest_bytes: usize,
}

impl CheckpointStore {
    pub fn open(
        root: impl AsRef<Path>,
        max_manifest_bytes: usize,
        max_segment_record_bytes: usize,
    ) -> Result<Self> {
        if max_manifest_bytes == 0 {
            return Err(Error::invalid_data(
                "checkpoint manifest limit must be non-zero",
            ));
        }
        let directory = root.as_ref().join("checkpoint");
        let manifests = directory.join("manifests");
        fs::create_dir_all(&manifests)?;
        let segment_store = SegmentStore::open(root, max_segment_record_bytes)?;
        Ok(Self {
            directory,
            manifests,
            segment_store,
            max_manifest_bytes,
        })
    }

    #[must_use]
    pub const fn segment_store(&self) -> &SegmentStore {
        &self.segment_store
    }

    pub fn publish(&self, manifest: &CheckpointManifest) -> Result<ManifestId> {
        manifest.validate_shape()?;
        for segment in &manifest.segments {
            self.segment_store.validate(segment)?;
        }
        let payload = postcard::to_stdvec(manifest)
            .map_err(|error| Error::new(ErrorCode::InvalidData, error.to_string()))?;
        if payload.len() > self.max_manifest_bytes {
            return Err(Error::invalid_data(
                "checkpoint manifest exceeds configured limit",
            ));
        }
        let checksum = *blake3::hash(&payload).as_bytes();
        let id = ManifestId(checksum);
        let envelope = ManifestEnvelope { payload, checksum };
        let encoded = postcard::to_stdvec(&envelope)
            .map_err(|error| Error::new(ErrorCode::InvalidData, error.to_string()))?;
        let manifest_path = self.manifest_path(id);
        if manifest_path.exists() {
            let existing = self.read_manifest(id)?;
            if &existing != manifest {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "checkpoint content-address collision",
                ));
            }
        } else {
            atomic_write(&manifest_path, &encoded, false)?;
        }

        let current = self.read_pointer(CURRENT_POINTER)?;
        if current != Some(id) {
            if let Some(previous_id) = current {
                self.write_pointer(PREVIOUS_POINTER, previous_id)?;
            }
            self.write_pointer(CURRENT_POINTER, id)?;
        }
        Ok(id)
    }

    pub fn recover(&self) -> Result<Option<RecoveredCheckpoint>> {
        let current_result = self.try_recover_pointer(CURRENT_POINTER, CheckpointSource::Current);
        if let Ok(Some(recovered)) = current_result {
            return Ok(Some(recovered));
        }
        let previous_result =
            self.try_recover_pointer(PREVIOUS_POINTER, CheckpointSource::Previous);
        if let Ok(Some(recovered)) = previous_result {
            return Ok(Some(recovered));
        }

        let has_current = self.directory.join(CURRENT_POINTER).exists();
        let has_previous = self.directory.join(PREVIOUS_POINTER).exists();
        if !has_current && !has_previous {
            return Ok(None);
        }
        match (current_result, previous_result) {
            (Err(error), _) | (_, Err(error)) => Err(error),
            _ => Err(Error::new(
                ErrorCode::CorruptStorage,
                "checkpoint pointers do not reference a recoverable manifest",
            )),
        }
    }

    pub fn read_manifest(&self, id: ManifestId) -> Result<CheckpointManifest> {
        let bytes = read_bounded(
            &self.manifest_path(id),
            self.max_manifest_bytes
                .checked_add(128)
                .ok_or_else(|| Error::internal("manifest envelope limit overflow"))?,
        )?;
        let envelope: ManifestEnvelope = postcard::from_bytes(&bytes)
            .map_err(|error| Error::new(ErrorCode::CorruptStorage, error.to_string()))?;
        let checksum = *blake3::hash(&envelope.payload).as_bytes();
        if checksum != envelope.checksum || checksum != id.0 {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "checkpoint manifest checksum mismatch",
            ));
        }
        let manifest: CheckpointManifest = postcard::from_bytes(&envelope.payload)
            .map_err(|error| Error::new(ErrorCode::CorruptStorage, error.to_string()))?;
        manifest.validate_shape()?;
        Ok(manifest)
    }

    fn try_recover_pointer(
        &self,
        name: &str,
        source: CheckpointSource,
    ) -> Result<Option<RecoveredCheckpoint>> {
        let Some(id) = self.read_pointer(name)? else {
            return Ok(None);
        };
        let manifest = self.read_manifest(id)?;
        for segment in &manifest.segments {
            self.segment_store.validate(segment)?;
        }
        Ok(Some(RecoveredCheckpoint {
            id,
            manifest,
            source,
        }))
    }

    fn write_pointer(&self, name: &str, id: ManifestId) -> Result<()> {
        let checksum = pointer_checksum(id);
        let encoded = postcard::to_stdvec(&PointerEnvelope { id, checksum })
            .map_err(|error| Error::new(ErrorCode::InvalidData, error.to_string()))?;
        atomic_write(&self.directory.join(name), &encoded, false)
    }

    fn read_pointer(&self, name: &str) -> Result<Option<ManifestId>> {
        let path = self.directory.join(name);
        if !path.exists() {
            return Ok(None);
        }
        let bytes = read_bounded(&path, MAX_POINTER_BYTES)?;
        let pointer: PointerEnvelope = postcard::from_bytes(&bytes)
            .map_err(|error| Error::new(ErrorCode::CorruptStorage, error.to_string()))?;
        if pointer.checksum != pointer_checksum(pointer.id) {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "checkpoint pointer checksum mismatch",
            ));
        }
        Ok(Some(pointer.id))
    }

    fn manifest_path(&self, id: ManifestId) -> PathBuf {
        self.manifests.join(format!("{id}.manifest"))
    }
}

fn pointer_checksum(id: ManifestId) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"irongraph-checkpoint-pointer-v1");
    hasher.update(&id.0);
    *hasher.finalize().as_bytes()
}
