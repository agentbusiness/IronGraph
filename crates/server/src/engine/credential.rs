use std::collections::BTreeSet;

use crate::broker::concurrent::{CanonicalMap, CanonicalSet};

use bitflags::bitflags;
use serde::{Deserialize, Serialize};
use sha2::{Digest as _, Sha256};
use x509_parser::parse_x509_certificate;

use crate::{Error, ErrorCode, ProjectId, Result, types::CredentialId};

const MAX_CREDENTIAL_RECORDS: usize = 65_536;
const MAX_CERTIFICATE_DER_BYTES: usize = 1024 * 1024;

/// SHA-256 fingerprint of the complete DER SubjectPublicKeyInfo carried by a client certificate.
/// Certificate rotation may therefore reissue an equivalent certificate without changing the
/// credential, while a key rotation requires a new durable write record.
pub fn certificate_public_key_fingerprint(certificate_der: &[u8]) -> Result<[u8; 32]> {
    let certificate = parse_certificate(certificate_der)?;
    Ok(Sha256::digest(certificate.tbs_certificate.subject_pki.raw).into())
}

/// Extracts the raw Ed25519 subject key used to bind a node TLS certificate to its committed
/// long-term signing identity.
pub fn certificate_ed25519_identity_key(certificate_der: &[u8]) -> Result<[u8; 32]> {
    let certificate = parse_certificate(certificate_der)?;
    let public = &certificate.tbs_certificate.subject_pki;
    if public.algorithm.algorithm.to_id_string() != "1.3.101.112"
        || public.subject_public_key.unused_bits != 0
    {
        return Err(Error::new(
            ErrorCode::AuthenticationFailed,
            "node certificate does not carry an Ed25519 identity key",
        ));
    }
    public
        .subject_public_key
        .data
        .as_ref()
        .try_into()
        .map_err(|_| {
            Error::new(
                ErrorCode::AuthenticationFailed,
                "node certificate Ed25519 key has an invalid length",
            )
        })
}

fn parse_certificate(
    certificate_der: &[u8],
) -> Result<x509_parser::certificate::X509Certificate<'_>> {
    if certificate_der.is_empty() || certificate_der.len() > MAX_CERTIFICATE_DER_BYTES {
        return Err(Error::new(
            ErrorCode::AuthenticationFailed,
            "certificate is empty or oversized",
        ));
    }
    let (remaining, certificate) = parse_x509_certificate(certificate_der).map_err(|_| {
        Error::new(
            ErrorCode::AuthenticationFailed,
            "certificate DER is invalid",
        )
    })?;
    if !remaining.is_empty() {
        return Err(Error::new(
            ErrorCode::AuthenticationFailed,
            "certificate contains trailing data",
        ));
    }
    Ok(certificate)
}

bitflags! {
    /// Remote protocol listener scope encoded in one compact bit mask.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct ProtocolScope: u8 {
        const BOLT = 0b0001;
        const QUERY_HTTP = 0b0010;
        const KAFKA = 0b0100;
        const AMQP = 0b1000;
    }
}

bitflags! {
    /// Operations granted to an mTLS service credential.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct OperationScope: u8 {
        const READ = 0b0001;
        const WRITE = 0b0010;
        const SCHEMA = 0b0100;
        const BROKER = 0b1000;
    }
}

bitflags! {
    /// Publicly addressable graph layers.
    #[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
    pub struct LayerScope: u8 {
        const OBSERVED = 0b01;
        const KNOWLEDGE = 0b10;
        const WORKSPACE = 0b100;
    }
}

/// Compact durable certificate-fingerprint authorization.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CredentialRecord {
    pub credential_id: CredentialId,
    pub certificate_fingerprint: [u8; 32],
    pub project_id: ProjectId,
    pub protocols: ProtocolScope,
    pub operations: OperationScope,
    pub layers: LayerScope,
    pub expires_at_millis: i64,
    pub revoked_at_index: Option<u64>,
}

impl CredentialRecord {
    pub fn validate(&self) -> Result<()> {
        if self.credential_id.0 == 0
            || self.certificate_fingerprint == [0_u8; 32]
            || self.protocols.is_empty()
            || self.operations.is_empty()
            || self.layers.is_empty()
            || self.expires_at_millis <= 0
            || self.revoked_at_index == Some(0)
            || self.protocols.bits() & !ProtocolScope::all().bits() != 0
            || self.operations.bits() & !OperationScope::all().bits() != 0
            || self.layers.bits() & !LayerScope::all().bits() != 0
        {
            return Err(Error::invalid_data("invalid client credential record"));
        }
        if self.operations.contains(OperationScope::BROKER)
            && !self
                .protocols
                .intersects(ProtocolScope::KAFKA | ProtocolScope::AMQP)
        {
            return Err(Error::invalid_data(
                "broker operation requires a broker protocol",
            ));
        }
        Ok(())
    }
}

/// Successful authorization result supplied to protocol-independent project/layer checks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct AuthorizedCredential {
    pub credential_id: CredentialId,
    pub project_id: ProjectId,
}

/// Deterministic durable mTLS service-credential registry.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct CredentialRegistry {
    by_fingerprint: CanonicalMap<[u8; 32], CredentialRecord>,
    ids: CanonicalSet<CredentialId>,
}
impl PartialEq for CredentialRegistry {
    fn eq(&self, other: &Self) -> bool {
        self.by_fingerprint.len() == other.by_fingerprint.len()
            && self.by_fingerprint.iter().all(|(key, value)| {
                other
                    .by_fingerprint
                    .get(&key)
                    .is_some_and(|other| other == value)
            })
            && self.ids.iter().collect::<BTreeSet<_>>() == other.ids.iter().collect::<BTreeSet<_>>()
    }
}
impl Eq for CredentialRegistry {}

impl CredentialRegistry {
    pub fn authenticate(
        &self,
        fingerprint: [u8; 32],
        protocol: ProtocolScope,
        now_millis: i64,
    ) -> Result<CredentialRecord> {
        if protocol.bits().count_ones() != 1 {
            return Err(Error::new(
                ErrorCode::AuthorizationDenied,
                "authentication request must name one protocol",
            ));
        }
        let record = self.by_fingerprint.get(&fingerprint).ok_or_else(|| {
            Error::new(
                ErrorCode::AuthenticationFailed,
                "unknown client certificate",
            )
        })?;
        if record.expires_at_millis <= now_millis {
            return Err(Error::new(
                ErrorCode::AuthenticationFailed,
                "client certificate expired",
            ));
        }
        if record.revoked_at_index.is_some() {
            return Err(Error::new(
                ErrorCode::AuthenticationFailed,
                "client certificate revoked",
            ));
        }
        if !record.protocols.contains(protocol) {
            return Err(Error::new(
                ErrorCode::AuthorizationDenied,
                "credential does not allow this protocol",
            ));
        }
        Ok((*record).clone())
    }

    pub fn validate_register(&self, record: &CredentialRecord, now_millis: i64) -> Result<()> {
        record.validate()?;
        if record.expires_at_millis <= now_millis || record.revoked_at_index.is_some() {
            return Err(Error::invalid_data("new credential is expired or revoked"));
        }
        if self
            .by_fingerprint
            .contains_key(&record.certificate_fingerprint)
            || self.ids.contains(&record.credential_id)
        {
            return Err(Error::invalid_data(
                "credential ID or fingerprint already exists",
            ));
        }
        if self.by_fingerprint.len() >= MAX_CREDENTIAL_RECORDS {
            return Err(Error::retryable(
                ErrorCode::Backpressure,
                "client credential retention limit reached",
                None,
            ));
        }
        Ok(())
    }

    pub fn register(&self, record: CredentialRecord, now_millis: i64) -> Result<()> {
        self.validate_register(&record, now_millis)?;
        self.ids.insert(record.credential_id);
        self.by_fingerprint
            .insert(record.certificate_fingerprint, record);
        Ok(())
    }

    pub fn validate_rotate(
        &self,
        old_fingerprint: [u8; 32],
        replacement: &CredentialRecord,
        rotation_index: u64,
        now_millis: i64,
    ) -> Result<()> {
        if rotation_index == 0 {
            return Err(Error::invalid_data(
                "credential rotation index must be non-zero",
            ));
        }
        let old = self
            .by_fingerprint
            .get(&old_fingerprint)
            .ok_or_else(|| Error::new(ErrorCode::AuthenticationFailed, "unknown credential"))?;
        if old.revoked_at_index.is_some() || old.expires_at_millis <= now_millis {
            return Err(Error::new(
                ErrorCode::AuthenticationFailed,
                "credential is already expired or revoked",
            ));
        }
        if replacement.project_id != old.project_id {
            return Err(Error::new(
                ErrorCode::AuthorizationDenied,
                "credential rotation cannot change immutable project",
            ));
        }
        replacement.validate()?;
        if replacement.expires_at_millis <= now_millis
            || replacement.revoked_at_index.is_some()
            || self
                .by_fingerprint
                .contains_key(&replacement.certificate_fingerprint)
            || self.ids.contains(&replacement.credential_id)
        {
            return Err(Error::invalid_data("invalid replacement credential"));
        }
        if self.by_fingerprint.len() >= MAX_CREDENTIAL_RECORDS {
            return Err(Error::retryable(
                ErrorCode::Backpressure,
                "client credential retention limit reached",
                None,
            ));
        }

        Ok(())
    }

    /// Publishes a replacement and revokes the previous fingerprint in writer order.
    pub fn rotate(
        &self,
        old_fingerprint: [u8; 32],
        replacement: CredentialRecord,
        rotation_index: u64,
        now_millis: i64,
    ) -> Result<()> {
        self.validate_rotate(old_fingerprint, &replacement, rotation_index, now_millis)?;
        let mut old = self
            .by_fingerprint
            .get_mut(&old_fingerprint)
            .ok_or_else(|| Error::new(ErrorCode::AuthenticationFailed, "unknown credential"))?;
        self.ids.insert(replacement.credential_id);
        self.by_fingerprint
            .insert(replacement.certificate_fingerprint, replacement);
        old.revoked_at_index = Some(rotation_index);
        Ok(())
    }

    pub fn validate_revoke(&self, fingerprint: [u8; 32], index: u64) -> Result<()> {
        if index == 0 {
            return Err(Error::invalid_data(
                "credential revocation index must be non-zero",
            ));
        }
        let record = self
            .by_fingerprint
            .get(&fingerprint)
            .ok_or_else(|| Error::new(ErrorCode::AuthenticationFailed, "unknown credential"))?;
        if record.revoked_at_index.is_some() {
            return Err(Error::new(
                ErrorCode::AuthenticationFailed,
                "credential is revoked",
            ));
        }
        Ok(())
    }
    pub fn revoke(&self, fingerprint: [u8; 32], index: u64) -> Result<()> {
        self.validate_revoke(fingerprint, index)?;
        let mut record = self
            .by_fingerprint
            .get_mut(&fingerprint)
            .ok_or_else(|| Error::new(ErrorCode::AuthenticationFailed, "unknown credential"))?;
        record.revoked_at_index = Some(index);
        Ok(())
    }

    pub fn authorize(
        &self,
        fingerprint: [u8; 32],
        project_id: ProjectId,
        protocol: ProtocolScope,
        operation: OperationScope,
        layers: LayerScope,
        now_millis: i64,
    ) -> Result<AuthorizedCredential> {
        if protocol.bits().count_ones() != 1
            || operation.bits().count_ones() != 1
            || layers.is_empty()
        {
            return Err(Error::new(
                ErrorCode::AuthorizationDenied,
                "authorization request must name one protocol and operation",
            ));
        }
        let broker_protocol = protocol.intersects(ProtocolScope::KAFKA | ProtocolScope::AMQP);
        if broker_protocol != operation.contains(OperationScope::BROKER) {
            return Err(Error::new(
                ErrorCode::AuthorizationDenied,
                "operation is incompatible with the selected protocol",
            ));
        }
        let record = self.authenticate(fingerprint, protocol, now_millis)?;
        if record.project_id != project_id
            || !record.operations.contains(operation)
            || !record.layers.contains(layers)
        {
            return Err(Error::new(
                ErrorCode::AuthorizationDenied,
                "credential is out of scope",
            ));
        }
        Ok(AuthorizedCredential {
            credential_id: record.credential_id,
            project_id: record.project_id,
        })
    }

    pub fn cleanup_expired(&self, now_millis: i64) {
        let expired = self
            .by_fingerprint
            .iter()
            .filter_map(|(fingerprint, record)| {
                (record.expires_at_millis <= now_millis)
                    .then_some((fingerprint, record.credential_id))
            })
            .collect::<Vec<_>>();
        for (fingerprint, id) in expired {
            self.by_fingerprint.remove(&fingerprint);
            self.ids.remove(&id);
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.by_fingerprint.len() > MAX_CREDENTIAL_RECORDS
            || self.ids.len() != self.by_fingerprint.len()
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "invalid client credential registry bounds",
            ));
        }
        let mut expected_ids = BTreeSet::new();
        for (fingerprint, record) in &self.by_fingerprint {
            record.validate()?;
            if fingerprint != record.certificate_fingerprint
                || !expected_ids.insert(record.credential_id)
            {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "invalid persisted client credential registry",
                ));
            }
        }
        if expected_ids != self.ids.iter().collect::<BTreeSet<_>>() {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "client credential ID index mismatch",
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use rcgen::{CertificateParams, KeyPair};

    use super::{certificate_ed25519_identity_key, certificate_public_key_fingerprint};

    fn record(id: u64, project: crate::ProjectId) -> super::CredentialRecord {
        super::CredentialRecord {
            credential_id: crate::types::CredentialId(id),
            certificate_fingerprint: [id as u8; 32],
            project_id: project,
            protocols: super::ProtocolScope::QUERY_HTTP,
            operations: super::OperationScope::READ,
            layers: super::LayerScope::KNOWLEDGE,
            expires_at_millis: 10_000,
            revoked_at_index: None,
        }
    }
    #[test]
    fn canonical_credentials_and_checkpoint_preserve_authorization() -> crate::Result<()> {
        let registry = super::CredentialRegistry::default();
        let project = crate::ProjectId::random();
        registry.register(record(1, project), 1)?;
        let reader = registry.clone();
        registry.rotate([1; 32], record(2, project), 2, 1)?;
        assert!(
            reader
                .authenticate([1; 32], super::ProtocolScope::QUERY_HTTP, 1)
                .is_err()
        );
        assert!(
            reader
                .authorize(
                    [2; 32],
                    project,
                    super::ProtocolScope::QUERY_HTTP,
                    super::OperationScope::READ,
                    super::LayerScope::KNOWLEDGE,
                    1
                )
                .is_ok()
        );
        assert!(
            reader
                .authorize(
                    [2; 32],
                    crate::ProjectId::random(),
                    super::ProtocolScope::QUERY_HTTP,
                    super::OperationScope::READ,
                    super::LayerScope::KNOWLEDGE,
                    1
                )
                .is_err()
        );
        let bytes =
            postcard::to_allocvec(&registry).map_err(|e| crate::Error::internal(e.to_string()))?;
        let restored: super::CredentialRegistry =
            postcard::from_bytes(&bytes).map_err(|e| crate::Error::internal(e.to_string()))?;
        restored.validate()?;
        assert_eq!(registry, restored);
        registry.cleanup_expired(10_000);
        assert!(
            reader
                .authenticate([2; 32], super::ProtocolScope::QUERY_HTTP, 1)
                .is_err()
        );
        Ok(())
    }
    #[test]
    fn credential_checkpoint_preserves_existing_wire_shape()
    -> Result<(), Box<dyn std::error::Error>> {
        #[derive(serde::Serialize, serde::Deserialize)]
        struct WireRegistry {
            by_fingerprint: std::collections::BTreeMap<[u8; 32], super::CredentialRecord>,
            ids: std::collections::BTreeSet<crate::types::CredentialId>,
        }
        let credential = record(1, crate::ProjectId::random());
        let wire = WireRegistry {
            by_fingerprint: [(credential.certificate_fingerprint, credential.clone())].into(),
            ids: [credential.credential_id].into(),
        };
        let registry: super::CredentialRegistry =
            postcard::from_bytes(&postcard::to_allocvec(&wire)?)?;
        registry.validate()?;
        assert_eq!(
            registry.authenticate([1; 32], super::ProtocolScope::QUERY_HTTP, 1)?,
            credential
        );
        let restored: WireRegistry = postcard::from_bytes(&postcard::to_allocvec(&registry)?)?;
        assert_eq!(restored.by_fingerprint, wire.by_fingerprint);
        assert_eq!(restored.ids, wire.ids);
        Ok(())
    }
    #[test]
    fn invalid_credential_preflight_does_not_mutate_canonical_registry() -> crate::Result<()> {
        let registry = super::CredentialRegistry::default();
        let project = crate::ProjectId::random();
        registry.register(record(1, project), 1)?;
        let before =
            postcard::to_allocvec(&registry).map_err(|e| crate::Error::internal(e.to_string()))?;
        let invalid = record(2, crate::ProjectId::random());
        assert!(registry.validate_rotate([1; 32], &invalid, 2, 1).is_err());
        assert!(registry.rotate([1; 32], invalid, 2, 1).is_err());
        assert!(registry.validate_register(&record(1, project), 1).is_err());
        assert!(registry.validate_revoke([1; 32], 0).is_err());
        assert_eq!(
            before,
            postcard::to_allocvec(&registry).map_err(|e| crate::Error::internal(e.to_string()))?
        );
        Ok(())
    }
    #[test]
    fn authentication_finishes_while_credential_writer_is_paused()
    -> Result<(), Box<dyn std::error::Error>> {
        let registry = super::CredentialRegistry::default();
        registry.register(record(1, crate::ProjectId::random()), 1)?;
        let reader = registry.clone();
        let mut pending = registry
            .by_fingerprint
            .get_mut(&[1; 32])
            .ok_or("credential missing")?;
        pending.revoked_at_index = Some(2);
        let (send, receive) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            send.send(
                reader
                    .authenticate([1; 32], super::ProtocolScope::QUERY_HTTP, 1)
                    .is_ok(),
            )
            .unwrap();
        });
        let result = receive.recv_timeout(std::time::Duration::from_secs(2));
        drop(pending);
        worker
            .join()
            .map_err(|_| "authentication reader panicked")?;
        assert!(result?);
        assert!(
            registry
                .authenticate([1; 32], super::ProtocolScope::QUERY_HTTP, 1)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn fingerprint_is_bound_to_subject_public_key_not_certificate_bytes()
    -> Result<(), Box<dyn std::error::Error>> {
        let key = KeyPair::generate()?;
        let first = CertificateParams::new(vec!["first.example".to_owned()])?.self_signed(&key)?;
        let second =
            CertificateParams::new(vec!["second.example".to_owned()])?.self_signed(&key)?;
        let other_key = KeyPair::generate()?;
        let other =
            CertificateParams::new(vec!["first.example".to_owned()])?.self_signed(&other_key)?;

        let fingerprint = certificate_public_key_fingerprint(first.der())?;
        assert_eq!(
            fingerprint,
            certificate_public_key_fingerprint(second.der())?
        );
        assert_ne!(
            fingerprint,
            certificate_public_key_fingerprint(other.der())?
        );
        assert!(certificate_public_key_fingerprint(b"not a certificate").is_err());
        Ok(())
    }

    #[test]
    fn node_identity_extraction_accepts_only_the_exact_ed25519_subject_key()
    -> Result<(), Box<dyn std::error::Error>> {
        let key = KeyPair::generate_for(&rcgen::PKCS_ED25519)?;
        let expected: [u8; 32] = key.public_key_raw().try_into()?;
        let certificate =
            CertificateParams::new(vec!["node.internal".to_owned()])?.self_signed(&key)?;
        assert_eq!(
            certificate_ed25519_identity_key(certificate.der())?,
            expected
        );

        let other = KeyPair::generate()?;
        let other_certificate =
            CertificateParams::new(vec!["node.internal".to_owned()])?.self_signed(&other)?;
        assert!(certificate_ed25519_identity_key(other_certificate.der()).is_err());
        Ok(())
    }
}
