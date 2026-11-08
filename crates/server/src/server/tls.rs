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
#[cfg(test)]
mod tests {
    use std::io::{Read as _, Write as _};

    use rcgen::{BasicConstraints, CertificateParams, CertifiedIssuer, IsCa, KeyPair};
    use rustls::{ClientConfig, ClientConnection, ServerConnection, pki_types::ServerName};

    use super::*;

    type TestResult<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

    fn certificate_authority() -> TestResult<CertifiedIssuer<'static, KeyPair>> {
        let mut params = CertificateParams::new(Vec::<String>::new())?;
        params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
        Ok(CertifiedIssuer::self_signed(params, KeyPair::generate()?)?)
    }

    fn client_config(
        trusted: &CertifiedIssuer<'_, KeyPair>,
        identity: Option<&CertifiedIssuer<'_, KeyPair>>,
        alpn: &[u8],
    ) -> TestResult<Arc<ClientConfig>> {
        let mut roots = rustls::RootCertStore::empty();
        roots.add(trusted.der().clone())?;
        let builder = ClientConfig::builder_with_provider(Arc::new(
            rustls::crypto::aws_lc_rs::default_provider(),
        ))
        .with_protocol_versions(&[&rustls::version::TLS13])?
        .with_root_certificates(roots);
        let mut client = if let Some(issuer) = identity {
            let key = KeyPair::generate()?;
            let cert =
                CertificateParams::new(vec!["client".to_owned()])?.signed_by(&key, issuer)?;
            builder.with_client_auth_cert(
                vec![cert.der().clone()],
                rustls::pki_types::PrivatePkcs8KeyDer::from(key.serialize_der()).into(),
            )?
        } else {
            builder.with_no_client_auth()
        };
        client.alpn_protocols = if alpn.is_empty() {
            Vec::new()
        } else {
            vec![alpn.to_vec()]
        };
        Ok(Arc::new(client))
    }

    fn server_config(
        issuer: &CertifiedIssuer<'_, KeyPair>,
        alpn: &[u8],
    ) -> TestResult<Arc<ServerConfig>> {
        let directory = tempfile::tempdir()?;
        let key = KeyPair::generate()?;
        let cert = CertificateParams::new(vec!["localhost".to_owned()])?.signed_by(&key, issuer)?;
        let certificate_path = directory.path().join("server.pem");
        let key_path = directory.path().join("key.pem");
        let trust_path = directory.path().join("ca.pem");
        std::fs::write(&certificate_path, cert.pem())?;
        std::fs::write(&key_path, key.serialize_pem())?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            std::fs::set_permissions(&key_path, std::fs::Permissions::from_mode(0o600))?;
        }
        std::fs::write(&trust_path, issuer.pem())?;
        Ok(load_remote_service_tls(
            &certificate_path,
            &key_path,
            &trust_path,
            if alpn.is_empty() {
                Vec::new()
            } else {
                vec![alpn.to_vec()]
            },
        )?)
    }

    fn transfer(client: &mut ClientConnection, server: &mut ServerConnection) -> TestResult {
        for _ in 0..16 {
            let mut to_server = Vec::new();
            client.write_tls(&mut to_server)?;
            let mut input = to_server.as_slice();
            while !input.is_empty() {
                server.read_tls(&mut input)?;
                server.process_new_packets()?;
            }
            let mut to_client = Vec::new();
            server.write_tls(&mut to_client)?;
            let mut input = to_client.as_slice();
            while !input.is_empty() {
                client.read_tls(&mut input)?;
                client.process_new_packets()?;
            }
            if !client.is_handshaking()
                && !server.is_handshaking()
                && !client.wants_write()
                && !server.wants_write()
            {
                return Ok(());
            }
        }
        Err("TLS fixture did not finish exchanging records".into())
    }

    #[test]
    fn mtls_preserves_payloads_and_alpn_for_every_remote_protocol() -> TestResult {
        let issuer = certificate_authority()?;
        let payload = "query rows: null, -41, 3.14159, 東京, café\n".repeat(512);
        // HTTP negotiates ALPN; Bolt, Kafka and AMQP use no ALPN.
        for alpn in [b"h2".as_slice(), b"http/1.1", b""] {
            let mut server = ServerConnection::new(server_config(&issuer, alpn)?)?;
            let mut client = ClientConnection::new(
                client_config(&issuer, Some(&issuer), alpn)?,
                ServerName::try_from("localhost")?,
            )?;
            transfer(&mut client, &mut server)?;
            assert_eq!(
                server.protocol_version(),
                Some(rustls::ProtocolVersion::TLSv1_3)
            );
            let expected_alpn = (!alpn.is_empty()).then_some(alpn);
            assert_eq!(client.alpn_protocol(), expected_alpn);
            assert_eq!(server.alpn_protocol(), expected_alpn);
            client.writer().write_all(payload.as_bytes())?;
            transfer(&mut client, &mut server)?;
            let mut received = vec![0; payload.len()];
            server.reader().read_exact(&mut received)?;
            assert_eq!(received, payload.as_bytes());
            server.writer().write_all(&received)?;
            transfer(&mut client, &mut server)?;
            client.reader().read_exact(&mut received)?;
            assert_eq!(received, payload.as_bytes());
        }
        Ok(())
    }

    #[test]
    fn mtls_rejects_missing_untrusted_clients_and_untrusted_or_misnamed_servers() -> TestResult {
        let trusted = certificate_authority()?;
        let untrusted = certificate_authority()?;
        for (roots, identity, hostname) in [
            (&trusted, None, "localhost"),
            (&trusted, Some(&untrusted), "localhost"),
            (&untrusted, Some(&trusted), "localhost"),
            (&trusted, Some(&trusted), "different.invalid"),
        ] {
            let mut server = ServerConnection::new(server_config(&trusted, b"http/1.1")?)?;
            let mut client = ClientConnection::new(
                client_config(roots, identity, b"http/1.1")?,
                ServerName::try_from(hostname)?,
            )?;
            assert!(transfer(&mut client, &mut server).is_err(), "{hostname}");
        }
        Ok(())
    }
}
