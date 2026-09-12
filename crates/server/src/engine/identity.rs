use std::path::Path;

use ed25519_dalek::pkcs8::EncodePrivateKey as _;
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand::Rng;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use uuid::Uuid;
use zeroize::{Zeroize, Zeroizing};

use crate::{
    Error, ErrorCode, Result,
    storage::{atomic_create, read_bounded},
};

const IDENTITY_FORMAT: u16 = 2;
const MAX_IDENTITY_BYTES: usize = 1024;

/// Immutable local store identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StoreId(pub Uuid);

impl StoreId {
    #[must_use]
    pub fn random() -> Self {
        Self(Uuid::new_v4())
    }
}

impl std::fmt::Display for StoreId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

/// Stable database-node identity, unrelated to graph node IDs.
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct ProcessId(pub Uuid);

impl ProcessId {
    #[must_use]
    pub fn random() -> Self {
        Self(Uuid::new_v4())
    }
}

impl std::fmt::Display for ProcessId {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(formatter)
    }
}

/// SHA-256 fingerprint anchoring the local store identity.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StoreFingerprint(pub [u8; 32]);

/// Public process identity used by the standalone write path and remote credential issuer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NodeIdentityPublic {
    pub store_id: StoreId,
    pub node_id: ProcessId,
    pub signing_key: [u8; 32],
    pub store_fingerprint: StoreFingerprint,
}

impl NodeIdentityPublic {
    pub fn validate(&self) -> Result<()> {
        if self.store_id.0.is_nil()
            || self.node_id.0.is_nil()
            || self.signing_key == [0_u8; 32]
            || self.store_fingerprint.0 == [0_u8; 32]
        {
            return Err(Error::new(
                ErrorCode::AuthenticationFailed,
                "invalid node public identity",
            ));
        }
        Ok(())
    }
}

#[derive(Serialize, Deserialize)]
struct IdentityDisk {
    format_version: u16,
    store_id: StoreId,
    node_id: ProcessId,
    signing_secret: [u8; 32],
    store_fingerprint: StoreFingerprint,
}

impl Drop for IdentityDisk {
    fn drop(&mut self) {
        self.signing_secret.zeroize();
    }
}

#[derive(Serialize, Deserialize)]
struct IdentityEnvelope {
    payload: Vec<u8>,
    checksum: [u8; 32],
}

impl Drop for IdentityEnvelope {
    fn drop(&mut self) {
        self.payload.zeroize();
    }
}

/// Long-term Ed25519 identity for the local store and credential issuer.
pub struct NodeIdentity {
    store_id: StoreId,
    node_id: ProcessId,
    signing: SigningKey,
    fingerprint: StoreFingerprint,
}

impl NodeIdentity {
    /// Creates the first node and the immutable genesis fingerprint.
    #[must_use]
    pub fn generate_genesis() -> Self {
        let store_id = StoreId::random();
        let node_id = ProcessId::random();
        let signing = generate_signing_key();
        let fingerprint = genesis_fingerprint(store_id, signing.verifying_key().to_bytes());
        Self {
            store_id,
            node_id,
            signing,
            fingerprint,
        }
    }

    #[must_use]
    pub fn public(&self) -> NodeIdentityPublic {
        NodeIdentityPublic {
            store_id: self.store_id,
            node_id: self.node_id,
            signing_key: self.signing.verifying_key().to_bytes(),
            store_fingerprint: self.fingerprint,
        }
    }

    #[must_use]
    pub const fn store_id(&self) -> StoreId {
        self.store_id
    }

    #[must_use]
    pub const fn node_id(&self) -> ProcessId {
        self.node_id
    }

    #[must_use]
    pub const fn store_fingerprint(&self) -> StoreFingerprint {
        self.fingerprint
    }

    #[must_use]
    pub fn is_genesis_identity(&self) -> bool {
        self.fingerprint
            == genesis_fingerprint(self.store_id, self.signing.verifying_key().to_bytes())
    }

    #[must_use]
    pub fn sign(&self, bytes: &[u8]) -> [u8; 64] {
        self.signing.sign(bytes).to_bytes()
    }

    pub fn verify(public_key: [u8; 32], bytes: &[u8], signature: &[u8]) -> Result<()> {
        let key = VerifyingKey::from_bytes(&public_key).map_err(|_| {
            Error::new(
                ErrorCode::AuthenticationFailed,
                "invalid Ed25519 public key",
            )
        })?;
        let signature = Signature::from_slice(signature).map_err(|_| {
            Error::new(
                ErrorCode::AuthenticationFailed,
                "invalid Ed25519 signature encoding",
            )
        })?;
        key.verify(bytes, &signature).map_err(|_| {
            Error::new(
                ErrorCode::AuthenticationFailed,
                "signature verification failed",
            )
        })
    }

    pub(crate) fn signing_key_pkcs8_der(&self) -> Result<Zeroizing<Vec<u8>>> {
        let document = self.signing.to_pkcs8_der().map_err(|error| {
            Error::new(
                ErrorCode::AuthenticationFailed,
                format!("cannot encode node signing identity: {error}"),
            )
        })?;
        Ok(Zeroizing::new(document.as_bytes().to_vec()))
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        let encoded = self.encode()?;
        if atomic_create(path, &encoded, true)? {
            return Ok(());
        }
        let existing = Self::load(path)?;
        if existing.public() == self.public() {
            Ok(())
        } else {
            Err(Error::new(
                ErrorCode::AuthenticationFailed,
                "refusing to replace an existing node identity",
            ))
        }
    }

    /// Loads the durable genesis identity or creates it exactly once without replacement races.
    pub fn load_or_generate_genesis(path: impl AsRef<Path>) -> Result<(Self, bool)> {
        let path = path.as_ref();
        match std::fs::metadata(path) {
            Ok(_) => return Self::load(path).map(|identity| (identity, false)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }

        let candidate = Self::generate_genesis();
        let encoded = candidate.encode()?;
        if atomic_create(path, &encoded, true)? {
            Ok((candidate, true))
        } else {
            Self::load(path).map(|identity| (identity, false))
        }
    }

    fn encode(&self) -> Result<Zeroizing<Vec<u8>>> {
        let disk = IdentityDisk {
            format_version: IDENTITY_FORMAT,
            store_id: self.store_id,
            node_id: self.node_id,
            signing_secret: self.signing.to_bytes(),
            store_fingerprint: self.fingerprint,
        };
        let payload = postcard::to_stdvec(&disk)
            .map_err(|error| Error::new(ErrorCode::InvalidData, error.to_string()))?;
        let checksum = identity_checksum(&payload);
        let envelope = postcard::to_stdvec(&IdentityEnvelope { payload, checksum })
            .map_err(|error| Error::new(ErrorCode::InvalidData, error.to_string()))?;
        if envelope.len() > MAX_IDENTITY_BYTES {
            return Err(Error::internal("encoded node identity exceeds its bound"));
        }
        Ok(Zeroizing::new(envelope))
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        reject_unsafe_identity_file(path)?;
        let bytes = Zeroizing::new(read_bounded(path, MAX_IDENTITY_BYTES)?);
        let envelope: IdentityEnvelope = postcard::from_bytes(&bytes)
            .map_err(|error| Error::new(ErrorCode::CorruptStorage, error.to_string()))?;
        if identity_checksum(&envelope.payload) != envelope.checksum {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "node identity checksum mismatch",
            ));
        }
        let (format_version, _) = postcard::take_from_bytes::<u16>(&envelope.payload)
            .map_err(|error| Error::new(ErrorCode::CorruptStorage, error.to_string()))?;
        if format_version != IDENTITY_FORMAT {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "unsupported node identity format",
            ));
        }
        let disk: IdentityDisk = postcard::from_bytes(&envelope.payload)
            .map_err(|error| Error::new(ErrorCode::CorruptStorage, error.to_string()))?;
        if disk.store_id.0.is_nil()
            || disk.node_id.0.is_nil()
            || disk.signing_secret == [0_u8; 32]
            || disk.store_fingerprint.0 == [0_u8; 32]
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "node identity contains invalid key material",
            ));
        }
        let signing = SigningKey::from_bytes(&disk.signing_secret);
        Ok(Self {
            store_id: disk.store_id,
            node_id: disk.node_id,
            signing,
            fingerprint: disk.store_fingerprint,
        })
    }
}

#[cfg(unix)]
fn reject_unsafe_identity_file(path: &Path) -> Result<()> {
    use std::os::unix::fs::MetadataExt as _;

    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.file_type().is_file() || metadata.mode() & 0o077 != 0 {
        return Err(Error::new(
            ErrorCode::AuthenticationFailed,
            "node identity must be a regular file inaccessible to group and other users",
        ));
    }
    Ok(())
}

#[cfg(not(unix))]
fn reject_unsafe_identity_file(path: &Path) -> Result<()> {
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.file_type().is_file() {
        return Err(Error::new(
            ErrorCode::AuthenticationFailed,
            "node identity must be a regular file",
        ));
    }
    Ok(())
}

fn generate_signing_key() -> SigningKey {
    let mut signing_secret = Zeroizing::new([0_u8; 32]);
    let mut rng = rand::rng();
    while signing_secret.iter().all(|byte| *byte == 0) {
        rng.fill(&mut *signing_secret);
    }
    SigningKey::from_bytes(&signing_secret)
}

fn genesis_fingerprint(store_id: StoreId, initial_key: [u8; 32]) -> StoreFingerprint {
    let mut hasher = Sha256::new();
    hasher.update(b"irongraph-genesis-v1");
    hasher.update(store_id.0.as_bytes());
    hasher.update(initial_key);
    StoreFingerprint(hasher.finalize().into())
}

fn identity_checksum(payload: &[u8]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"irongraph-node-identity-v1");
    hasher.update(payload);
    *hasher.finalize().as_bytes()
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::NodeIdentity;

    #[test]
    fn durable_identity_roundtrips_without_replacement() -> crate::Result<()> {
        let directory = tempdir().map_err(crate::Error::from)?;
        let path = directory.path().join("identity.bin");
        let identity = NodeIdentity::generate_genesis();
        identity.save(&path)?;
        identity.save(&path)?;
        let restored = NodeIdentity::load(&path)?;
        assert_eq!(restored.public(), identity.public());
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn identity_loader_rejects_group_or_world_access() -> crate::Result<()> {
        use std::os::unix::fs::PermissionsExt as _;

        let directory = tempdir().map_err(crate::Error::from)?;
        let path = directory.path().join("identity.bin");
        NodeIdentity::generate_genesis().save(&path)?;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o640))?;
        let error = NodeIdentity::load(&path)
            .err()
            .ok_or_else(|| crate::Error::internal("unsafe identity file was accepted"))?;
        assert_eq!(error.code, crate::ErrorCode::AuthenticationFailed);
        Ok(())
    }
}
