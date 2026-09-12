use std::{
    io::{BufReader, Cursor},
    path::Path,
    sync::Arc,
};

use rustls::{
    ServerConfig,
    pki_types::{CertificateDer, PrivateKeyDer},
    server::WebPkiClientVerifier,
};
// The remote Query, Kafka, AMQP, and Bolt listeners share this strict client-certificate verifier.

use crate::{Error, ErrorCode, Result};

const MAX_PEM_BYTES: u64 = 16 * 1024 * 1024;

pub(super) fn load_remote_service_tls(
    certificate_path: &Path,
    private_key_path: &Path,
    trust_path: &Path,
    alpn_protocols: Vec<Vec<u8>>,
) -> Result<Arc<ServerConfig>> {
    reject_unsafe_private_key(private_key_path)?;
    if alpn_protocols.len() > 8
        || alpn_protocols
            .iter()
            .any(|protocol| protocol.is_empty() || protocol.len() > 255)
    {
        return Err(Error::invalid_data("remote service ALPN list is invalid"));
    }
    let certificate_chain = parse_certificates(
        &read_regular_file(certificate_path)?,
        "remote service certificate",
    )?;
    let private_key = parse_private_key(&read_regular_file(private_key_path)?)?;
    let trust_anchors =
        parse_certificates(&read_regular_file(trust_path)?, "remote client trust store")?;
    let mut roots = rustls::RootCertStore::empty();
    for certificate in trust_anchors {
        roots.add(certificate).map_err(|error| {
            Error::new(
                ErrorCode::AuthenticationFailed,
                format!("invalid remote client trust anchor: {error}"),
            )
        })?;
    }
    let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
    let verifier = WebPkiClientVerifier::builder_with_provider(roots.into(), Arc::clone(&provider))
        .build()
        .map_err(tls_configuration_error)?;
    let mut server = ServerConfig::builder_with_provider(provider)
        .with_protocol_versions(&[&rustls::version::TLS13])
        .map_err(tls_configuration_error)?
        .with_client_cert_verifier(verifier)
        .with_single_cert(certificate_chain, private_key)
        .map_err(tls_configuration_error)?;
    server.alpn_protocols = alpn_protocols;
    server.max_early_data_size = 0;
    Ok(Arc::new(server))
}

fn read_regular_file(path: &Path) -> Result<Vec<u8>> {
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.file_type().is_file() || metadata.len() == 0 || metadata.len() > MAX_PEM_BYTES {
        return Err(Error::new(
            ErrorCode::AuthenticationFailed,
            format!(
                "TLS material is not a bounded regular file: {}",
                path.display()
            ),
        ));
    }
    std::fs::read(path).map_err(Error::from)
}

fn parse_certificates(bytes: &[u8], description: &str) -> Result<Vec<CertificateDer<'static>>> {
    let mut reader = BufReader::new(Cursor::new(bytes));
    let mut certificates = Vec::new();
    loop {
        let item = rustls_pemfile::read_one(&mut reader).map_err(|error| {
            Error::new(
                ErrorCode::AuthenticationFailed,
                format!("cannot parse {description}: {error}"),
            )
        })?;
        match item {
            Some(rustls_pemfile::Item::X509Certificate(certificate)) => {
                certificates.push(certificate);
            }
            Some(_) => {
                return Err(Error::new(
                    ErrorCode::AuthenticationFailed,
                    format!("{description} contains a non-certificate PEM section"),
                ));
            }
            None => break,
        }
    }
    if certificates.is_empty() {
        return Err(Error::new(
            ErrorCode::AuthenticationFailed,
            format!("{description} contains no certificates"),
        ));
    }
    Ok(certificates)
}

fn parse_private_key(bytes: &[u8]) -> Result<PrivateKeyDer<'static>> {
    let mut reader = BufReader::new(Cursor::new(bytes));
    let mut key = None;
    loop {
        let item = rustls_pemfile::read_one(&mut reader).map_err(|error| {
            Error::new(
                ErrorCode::AuthenticationFailed,
                format!("cannot parse node private key: {error}"),
            )
        })?;
        let parsed = match item {
            Some(rustls_pemfile::Item::Pkcs1Key(value)) => value.into(),
            Some(rustls_pemfile::Item::Pkcs8Key(value)) => value.into(),
            Some(rustls_pemfile::Item::Sec1Key(value)) => value.into(),
            Some(_) => {
                return Err(Error::new(
                    ErrorCode::AuthenticationFailed,
                    "node private-key file contains a non-key PEM section",
                ));
            }
            None => break,
        };
        if key.replace(parsed).is_some() {
            return Err(Error::new(
                ErrorCode::AuthenticationFailed,
                "node private-key file contains more than one key",
            ));
        }
    }
    key.ok_or_else(|| {
        Error::new(
            ErrorCode::AuthenticationFailed,
            "node private-key file contains no supported key",
        )
    })
}

#[cfg(unix)]
fn reject_unsafe_private_key(path: &Path) -> Result<()> {
    use std::os::unix::fs::MetadataExt as _;

    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.file_type().is_file() || metadata.mode() & 0o077 != 0 {
        return Err(Error::new(
            ErrorCode::AuthenticationFailed,
            "node private key must be a regular file inaccessible to group and other users",
        ));
    }
    Ok(())
}

#[cfg(not(unix))]
fn reject_unsafe_private_key(path: &Path) -> Result<()> {
    let metadata = std::fs::symlink_metadata(path)?;
    if !metadata.file_type().is_file() {
        return Err(Error::new(
            ErrorCode::AuthenticationFailed,
            "node private key must be a regular file",
        ));
    }
    Ok(())
}

fn tls_configuration_error(error: impl std::fmt::Display) -> Error {
    Error::new(
        ErrorCode::AuthenticationFailed,
        format!("invalid node TLS configuration: {error}"),
    )
}

// These exercise the shared remote-service mTLS loader.
