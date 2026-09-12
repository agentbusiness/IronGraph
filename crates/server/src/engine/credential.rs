use std::collections::{BTreeMap, BTreeSet};

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
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct CredentialRegistry {
    by_fingerprint: BTreeMap<[u8; 32], CredentialRecord>,
    ids: BTreeSet<CredentialId>,
}

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
        Ok(record.clone())
    }

    pub fn register(&mut self, record: CredentialRecord, now_millis: i64) -> Result<()> {
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
        self.ids.insert(record.credential_id);
        self.by_fingerprint
            .insert(record.certificate_fingerprint, record);
        Ok(())
    }

    /// Atomically registers a replacement and revokes the old fingerprint at one log index.
    pub fn rotate(
        &mut self,
        old_fingerprint: [u8; 32],
        replacement: CredentialRecord,
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

        let replacement_id = replacement.credential_id;
        let replacement_fingerprint = replacement.certificate_fingerprint;
        self.ids.insert(replacement_id);
        self.by_fingerprint
            .insert(replacement_fingerprint, replacement);
        if let Some(old_mut) = self.by_fingerprint.get_mut(&old_fingerprint) {
            old_mut.revoked_at_index = Some(rotation_index);
        } else {
            self.by_fingerprint.remove(&replacement_fingerprint);
            self.ids.remove(&replacement_id);
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "credential disappeared during rotation",
            ));
        }
        Ok(())
    }

    pub fn revoke(&mut self, fingerprint: [u8; 32], index: u64) -> Result<()> {
        if index == 0 {
            return Err(Error::invalid_data(
                "credential revocation index must be non-zero",
            ));
        }
        let record = self
            .by_fingerprint
            .get_mut(&fingerprint)
            .ok_or_else(|| Error::new(ErrorCode::AuthenticationFailed, "unknown credential"))?;
        if record.revoked_at_index.is_some() {
            return Err(Error::new(
                ErrorCode::AuthenticationFailed,
                "credential is revoked",
            ));
        }
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

    pub fn cleanup_expired(&mut self, now_millis: i64) {
        let expired = self
            .by_fingerprint
            .iter()
            .filter_map(|(fingerprint, record)| {
                (record.expires_at_millis <= now_millis)
                    .then_some((*fingerprint, record.credential_id))
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
            if fingerprint != &record.certificate_fingerprint
                || !expected_ids.insert(record.credential_id)
            {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "invalid persisted client credential registry",
                ));
            }
        }
        if expected_ids != self.ids {
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
