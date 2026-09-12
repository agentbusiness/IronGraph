use std::{
    collections::BTreeMap,
    env,
    fs::{self, File, OpenOptions},
    io::{BufRead, BufReader, Read, Seek, SeekFrom, Write},
    net::{TcpStream, ToSocketAddrs},
    path::{Path, PathBuf},
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt as _, PermissionsExt as _};

use rustls::{ClientConfig, ClientConnection, RootCertStore, StreamOwned};
use sha2::{Digest as _, Sha256};
use url::Url;

use crate::{
    Error, ErrorCode, Result,
    graph::{EmbeddingDType, EmbeddingProfile, Similarity},
};

use super::{EmbeddingModelArtifacts, embedding::encoder_hash};

pub const DEFAULT_EMBEDDING_REVISION: &str = "113abe4acafa848e77ead9c0623205e511932348";
pub const DEFAULT_EMBEDDING_MODEL_FILE: &str = "model.safetensors";
pub const DEFAULT_EMBEDDING_MODEL_BYTES: u64 = 2_471_644_736;
pub const DEFAULT_EMBEDDING_MODEL_SHA256: [u8; 32] = [
    0x45, 0xf8, 0x44, 0x06, 0x82, 0xa8, 0x9a, 0xc5, 0x77, 0xcc, 0x8d, 0x53, 0xb1, 0xbb, 0x34, 0x58,
    0x04, 0x77, 0x2a, 0xdb, 0x7b, 0x34, 0xe0, 0x57, 0x35, 0x62, 0xe2, 0xfc, 0xa4, 0xe6, 0x2b, 0x0d,
];
pub const DEFAULT_EMBEDDING_CONFIG_FILE: &str = "config.json";
pub const DEFAULT_EMBEDDING_CONFIG_BYTES: u64 = 1_158;
pub const DEFAULT_EMBEDDING_CONFIG_SHA256: [u8; 32] = [
    0xdd, 0xf5, 0x69, 0x70, 0x53, 0xb3, 0x46, 0x11, 0xda, 0x16, 0xea, 0x2d, 0xd5, 0x65, 0xf8, 0x6d,
    0x43, 0x41, 0xf1, 0x4a, 0x05, 0xfc, 0x46, 0x0b, 0x54, 0xd2, 0x0e, 0x50, 0xa1, 0x0b, 0x98, 0x95,
];
pub const DEFAULT_EMBEDDING_TOKENIZER_FILE: &str = "tokenizer.json";
pub const DEFAULT_EMBEDDING_TOKENIZER_BYTES: u64 = 9_085_657;
pub const DEFAULT_EMBEDDING_TOKENIZER_SHA256: [u8; 32] = [
    0x79, 0xe3, 0xe5, 0x22, 0x63, 0x5f, 0x31, 0x71, 0x30, 0x09, 0x13, 0xbb, 0x42, 0x14, 0x64, 0xa8,
    0x7d, 0xe6, 0x22, 0x21, 0x82, 0xa0, 0x57, 0x0b, 0x9b, 0x2c, 0xcb, 0xa2, 0xa9, 0x64, 0xb2, 0xb4,
];
const DEFAULT_EMBEDDING_MODEL_URL: &str = "https://huggingface.co/nvidia/llama-nemotron-embed-1b-v2/resolve/113abe4acafa848e77ead9c0623205e511932348/model.safetensors?download=true";
const DEFAULT_EMBEDDING_CONFIG_URL: &str = "https://huggingface.co/nvidia/llama-nemotron-embed-1b-v2/resolve/113abe4acafa848e77ead9c0623205e511932348/config.json?download=true";
const DEFAULT_EMBEDDING_TOKENIZER_URL: &str = "https://huggingface.co/nvidia/llama-nemotron-embed-1b-v2/resolve/113abe4acafa848e77ead9c0623205e511932348/tokenizer.json?download=true";
const MAX_REDIRECTS: usize = 8;
const MAX_HEADER_BYTES: usize = 64 * 1024;
const NETWORK_TIMEOUT: Duration = Duration::from_secs(60);

/// Thread-safe count of bytes written toward the current artifact. The installer increments it as
/// the body streams to disk; the server reads it to render live download progress against the
/// artifact's known exact size. Cloning shares the same counter.
#[derive(Clone, Debug, Default)]
pub struct ProgressSink {
    written: Arc<AtomicU64>,
}

impl ProgressSink {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    fn add(&self, bytes: u64) {
        self.written.fetch_add(bytes, Ordering::Relaxed);
    }

    /// Reset the counter to an absolute value (used when a resume seeds already-present bytes or a
    /// restart discards them).
    fn set(&self, bytes: u64) {
        self.written.store(bytes, Ordering::Relaxed);
    }

    /// Bytes written toward the current artifact so far. Consumed by the server's readiness
    /// progress reporting; `allow(dead_code)` covers the interval before that reader is wired.
    #[must_use]
    #[allow(dead_code)]
    pub fn written(&self) -> u64 {
        self.written.load(Ordering::Relaxed)
    }
}

#[derive(Clone, Debug)]
struct PinnedArtifact {
    url: &'static str,
    file_name: &'static str,
    exact_bytes: u64,
    sha256: [u8; 32],
}

const DEFAULT_EMBEDDING_MODEL_ARTIFACT: PinnedArtifact = PinnedArtifact {
    url: DEFAULT_EMBEDDING_MODEL_URL,
    file_name: DEFAULT_EMBEDDING_MODEL_FILE,
    exact_bytes: DEFAULT_EMBEDDING_MODEL_BYTES,
    sha256: DEFAULT_EMBEDDING_MODEL_SHA256,
};

const DEFAULT_EMBEDDING_CONFIG_ARTIFACT: PinnedArtifact = PinnedArtifact {
    url: DEFAULT_EMBEDDING_CONFIG_URL,
    file_name: DEFAULT_EMBEDDING_CONFIG_FILE,
    exact_bytes: DEFAULT_EMBEDDING_CONFIG_BYTES,
    sha256: DEFAULT_EMBEDDING_CONFIG_SHA256,
};

const DEFAULT_EMBEDDING_TOKENIZER_ARTIFACT: PinnedArtifact = PinnedArtifact {
    url: DEFAULT_EMBEDDING_TOKENIZER_URL,
    file_name: DEFAULT_EMBEDDING_TOKENIZER_FILE,
    exact_bytes: DEFAULT_EMBEDDING_TOKENIZER_BYTES,
    sha256: DEFAULT_EMBEDDING_TOKENIZER_SHA256,
};

/// Returns the exact pinned first embedding encoder, installing every required file when absent.
pub fn ensure_default_embedding_model() -> Result<EmbeddingModelArtifacts> {
    let directory = private_artifact_root()?
        .join(".irongraph")
        .join("models")
        .join("nvidia-llama-nemotron-embed-1b-v2")
        .join(DEFAULT_EMBEDDING_REVISION);
    let model_safetensors = install_pinned(&directory, &DEFAULT_EMBEDDING_MODEL_ARTIFACT)?;
    let model_config_json = install_pinned(&directory, &DEFAULT_EMBEDDING_CONFIG_ARTIFACT)?;
    let tokenizer_json = install_pinned(&directory, &DEFAULT_EMBEDDING_TOKENIZER_ARTIFACT)?;
    let profile = EmbeddingProfile::new(
        DEFAULT_EMBEDDING_MODEL_SHA256,
        encoder_hash(
            DEFAULT_EMBEDDING_TOKENIZER_SHA256,
            DEFAULT_EMBEDDING_CONFIG_SHA256,
        ),
        384,
        EmbeddingDType::F16,
        true,
        Similarity::Cosine,
    )?;
    Ok(EmbeddingModelArtifacts {
        model_safetensors,
        tokenizer_json,
        model_config_json,
        model_sha256: DEFAULT_EMBEDDING_MODEL_SHA256,
        model_exact_bytes: DEFAULT_EMBEDDING_MODEL_BYTES,
        tokenizer_sha256: DEFAULT_EMBEDDING_TOKENIZER_SHA256,
        tokenizer_exact_bytes: DEFAULT_EMBEDDING_TOKENIZER_BYTES,
        config_sha256: DEFAULT_EMBEDDING_CONFIG_SHA256,
        config_exact_bytes: DEFAULT_EMBEDDING_CONFIG_BYTES,
        profile,
        maximum_input_tokens: 8_192,
    })
}

fn private_artifact_root() -> Result<PathBuf> {
    env::var_os("HOME")
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
        .ok_or_else(|| embedding_unavailable("the current user has no absolute home directory"))
}

fn install_pinned(directory: &Path, artifact: &PinnedArtifact) -> Result<PathBuf> {
    install_pinned_tracked(directory, artifact, &ProgressSink::new())
}

fn install_pinned_tracked(
    directory: &Path,
    artifact: &PinnedArtifact,
    sink: &ProgressSink,
) -> Result<PathBuf> {
    ensure_private_hierarchy(directory)?;
    let lock_path = directory.join(".install.lock");
    let lock = open_private_rw(&lock_path, false)?;
    lock.lock()
        .map_err(|error| embedding_io("acquiring the model installation lock", error))?;

    let final_path = directory.join(artifact.file_name);
    if final_path.exists() {
        if verify_exact(&final_path, artifact).is_ok() {
            set_private_file_mode(&final_path)?;
            return Ok(final_path);
        }
        quarantine_invalid(&final_path)?;
    }

    let partial_path = directory.join(format!("{}.part", artifact.file_name));
    let mut partial = open_private_rw(&partial_path, false)?;
    let mut existing = partial
        .metadata()
        .map_err(|error| embedding_io("reading partial model metadata", error))?
        .len();
    if existing > artifact.exact_bytes {
        partial
            .set_len(0)
            .map_err(|error| embedding_io("discarding an oversized partial model", error))?;
        existing = 0;
    }

    let (mut digest_state, digested_bytes) = hash_prefix(&mut partial, existing)?;
    if digested_bytes != existing {
        return Err(embedding_unavailable(
            "the partial model changed while its resume digest was prepared",
        ));
    }
    // Progress reflects verified prefix bytes already on disk, then rises as the body streams.
    sink.set(existing);
    if existing < artifact.exact_bytes {
        partial
            .seek(SeekFrom::Start(existing))
            .map_err(|error| embedding_io("seeking the partial model", error))?;
        download_from(
            artifact,
            existing,
            &mut partial,
            &mut digest_state,
            &partial_path,
            sink,
        )?;
    }
    let mut length = partial
        .metadata()
        .map_err(|error| embedding_io("checking downloaded model size", error))?
        .len();
    let mut actual: [u8; 32] = digest_state.clone().finalize().into();
    if length != artifact.exact_bytes || actual != artifact.sha256 {
        partial
            .set_len(0)
            .and_then(|()| partial.seek(SeekFrom::Start(0)).map(|_| ()))
            .map_err(|error| embedding_io("replacing an unverified partial model", error))?;
        digest_state = Sha256::new();
        sink.set(0);
        download_from(
            artifact,
            0,
            &mut partial,
            &mut digest_state,
            &partial_path,
            sink,
        )?;
        length = partial
            .metadata()
            .map_err(|error| embedding_io("checking replacement model size", error))?
            .len();
        actual = digest_state.finalize().into();
        if length != artifact.exact_bytes || actual != artifact.sha256 {
            return Err(embedding_unavailable(
                "downloaded model failed its pinned size or SHA-256 verification",
            ));
        }
    }
    partial
        .sync_all()
        .map_err(|error| embedding_io("syncing the downloaded model", error))?;
    set_private_file_mode(&partial_path)?;
    drop(partial);
    fs::rename(&partial_path, &final_path)
        .map_err(|error| embedding_io("atomically installing the verified model", error))?;
    sync_directory(directory)?;
    verify_exact(&final_path, artifact)?;
    Ok(final_path)
}

fn download_from(
    artifact: &PinnedArtifact,
    existing: u64,
    destination: &mut File,
    hasher: &mut Sha256,
    partial_path: &Path,
    sink: &ProgressSink,
) -> Result<()> {
    let mut url = Url::parse(artifact.url)
        .map_err(|error| embedding_unavailable(format!("pinned model URL is invalid: {error}")))?;
    let mut response = None;
    for _ in 0..=MAX_REDIRECTS {
        require_https(&url)?;
        let current = open_https(&url, existing)?;
        if matches!(current.status, 301 | 302 | 303 | 307 | 308) {
            let location = current
                .header("location")
                .ok_or_else(|| embedding_unavailable("model redirect has no Location header"))?;
            url = url.join(location).map_err(|error| {
                embedding_unavailable(format!("model redirect URL is invalid: {error}"))
            })?;
            require_https(&url)?;
            continue;
        }
        response = Some(current);
        break;
    }
    let mut response = response.ok_or_else(|| embedding_unavailable("too many model redirects"))?;
    require_https(&response.url)?;

    let mut write_offset = existing;
    match (existing, response.status) {
        (0, 200 | 206) => {
            if response.status == 206 {
                validate_content_range(response.header("content-range"), 0, artifact.exact_bytes)?;
            }
        }
        (_, 206) => validate_content_range(
            response.header("content-range"),
            existing,
            artifact.exact_bytes,
        )?,
        (_, 200) => {
            destination
                .set_len(0)
                .and_then(|()| destination.seek(SeekFrom::Start(0)).map(|_| ()))
                .map_err(|error| embedding_io("restarting a non-range model response", error))?;
            *hasher = Sha256::new();
            sink.set(0);
            write_offset = 0;
        }
        (_, 416) if existing == artifact.exact_bytes => return Ok(()),
        (_, status) => {
            return Err(embedding_unavailable(format!(
                "model server returned HTTP status {status}"
            )));
        }
    }
    if response
        .header("content-encoding")
        .is_some_and(|value| !value.eq_ignore_ascii_case("identity"))
    {
        return Err(embedding_unavailable(
            "model server applied an unsupported content encoding",
        ));
    }
    let remaining = artifact
        .exact_bytes
        .checked_sub(write_offset)
        .ok_or_else(|| embedding_unavailable("model resume offset exceeds the pinned size"))?;
    if let Some(length) = response.content_length()? {
        if length > remaining {
            return Err(embedding_unavailable(
                "model response exceeds the pinned remaining size",
            ));
        }
    }
    let written = response.copy_body(destination, hasher, remaining, sink)?;
    if written != remaining {
        return Err(embedding_unavailable(format!(
            "model transfer ended after {written} bytes with {remaining} required"
        )));
    }
    if destination
        .metadata()
        .map_err(|error| embedding_io("checking partial model after transfer", error))?
        .len()
        != artifact.exact_bytes
    {
        return Err(embedding_unavailable(format!(
            "partial model length changed unexpectedly: {}",
            partial_path.display()
        )));
    }
    Ok(())
}

struct HttpsResponse {
    status: u16,
    headers: BTreeMap<String, String>,
    reader: BufReader<StreamOwned<ClientConnection, TcpStream>>,
    url: Url,
}

impl HttpsResponse {
    fn header(&self, name: &str) -> Option<&str> {
        self.headers.get(name).map(String::as_str)
    }

    fn content_length(&self) -> Result<Option<u64>> {
        self.header("content-length")
            .map(|value| {
                value.parse::<u64>().map_err(|_| {
                    embedding_unavailable("model response has an invalid Content-Length")
                })
            })
            .transpose()
    }

    fn copy_body(
        &mut self,
        destination: &mut File,
        hasher: &mut Sha256,
        maximum: u64,
        sink: &ProgressSink,
    ) -> Result<u64> {
        if self
            .header("transfer-encoding")
            .is_some_and(|value| value.eq_ignore_ascii_case("chunked"))
        {
            copy_chunked(&mut self.reader, destination, hasher, maximum, sink)
        } else if let Some(length) = self.content_length()? {
            if length > maximum {
                return Err(embedding_unavailable("model body exceeds its pinned bound"));
            }
            copy_exact(&mut self.reader, destination, hasher, length, sink)
        } else {
            copy_to_eof(&mut self.reader, destination, hasher, maximum, sink)
        }
    }
}

fn open_https(url: &Url, start: u64) -> Result<HttpsResponse> {
    require_https(url)?;
    let host = url
        .host_str()
        .ok_or_else(|| embedding_unavailable("model URL has no host"))?;
    let port = url.port_or_known_default().unwrap_or(443);
    let addresses = (host, port)
        .to_socket_addrs()
        .map_err(|error| embedding_io("resolving the model host", error))?;
    let mut socket = None;
    for address in addresses {
        if let Ok(candidate) = TcpStream::connect_timeout(&address, NETWORK_TIMEOUT) {
            socket = Some(candidate);
            break;
        }
    }
    let socket = socket.ok_or_else(|| embedding_unavailable("the model host is unavailable"))?;
    socket
        .set_read_timeout(Some(NETWORK_TIMEOUT))
        .and_then(|()| socket.set_write_timeout(Some(NETWORK_TIMEOUT)))
        .map_err(|error| embedding_io("setting model transfer timeouts", error))?;

    let roots = static_roots();
    let config = ClientConfig::builder()
        .with_root_certificates(roots)
        .with_no_client_auth();
    let server_name = rustls::pki_types::ServerName::try_from(host.to_owned())
        .map_err(|_| embedding_unavailable("model URL host is not a valid TLS server name"))?;
    let connection =
        ClientConnection::new(std::sync::Arc::new(config), server_name).map_err(|error| {
            embedding_unavailable(format!("creating model TLS session failed: {error}"))
        })?;
    let mut stream = StreamOwned::new(connection, socket);
    let mut target = url.path().to_owned();
    if target.is_empty() {
        target.push('/');
    }
    if let Some(query) = url.query() {
        target.push('?');
        target.push_str(query);
    }
    let host_header = if port == 443 {
        host.to_owned()
    } else {
        format!("{host}:{port}")
    };
    write!(
        stream,
        "GET {target} HTTP/1.1\r\nHost: {host_header}\r\nUser-Agent: IronGraph/0.1\r\nAccept: application/octet-stream\r\nAccept-Encoding: identity\r\nRange: bytes={start}-\r\nConnection: close\r\n\r\n"
    )
    .and_then(|()| stream.flush())
    .map_err(|error| embedding_io("sending the model HTTPS request", error))?;
    let mut reader = BufReader::new(stream);
    let (status, headers) = read_response_head(&mut reader)?;
    Ok(HttpsResponse {
        status,
        headers,
        reader,
        url: url.clone(),
    })
}

fn static_roots() -> RootCertStore {
    let mut roots = RootCertStore::empty();
    roots.extend(webpki_roots::TLS_SERVER_ROOTS.iter().cloned());
    roots
}

fn read_response_head<R: BufRead>(reader: &mut R) -> Result<(u16, BTreeMap<String, String>)> {
    let mut consumed = 0_usize;
    let status_line = read_header_line(reader, &mut consumed)?;
    let mut fields = status_line.split_whitespace();
    let version = fields.next().unwrap_or_default();
    let status = fields
        .next()
        .and_then(|value| value.parse::<u16>().ok())
        .ok_or_else(|| {
            embedding_unavailable("model server returned an invalid HTTP status line")
        })?;
    if !matches!(version, "HTTP/1.0" | "HTTP/1.1") {
        return Err(embedding_unavailable(
            "model server returned an unsupported HTTP version",
        ));
    }
    let mut headers = BTreeMap::new();
    loop {
        let line = read_header_line(reader, &mut consumed)?;
        if line.is_empty() {
            break;
        }
        if line.starts_with([' ', '\t']) {
            return Err(embedding_unavailable(
                "folded HTTP headers are not accepted",
            ));
        }
        let (name, value) = line
            .split_once(':')
            .ok_or_else(|| embedding_unavailable("model response contains a malformed header"))?;
        if name.is_empty()
            || !name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || byte == b'-')
        {
            return Err(embedding_unavailable(
                "model response header name is invalid",
            ));
        }
        let name = name.to_ascii_lowercase();
        let value = value.trim().to_owned();
        if let Some(previous) = headers.insert(name.clone(), value.clone())
            && (name == "content-length" || name == "content-range")
            && previous != value
        {
            return Err(embedding_unavailable(
                "model response contains conflicting length headers",
            ));
        }
    }
    Ok((status, headers))
}

fn read_header_line<R: BufRead>(reader: &mut R, consumed: &mut usize) -> Result<String> {
    let mut line = Vec::new();
    let count = reader
        .read_until(b'\n', &mut line)
        .map_err(|error| embedding_io("reading model response headers", error))?;
    *consumed = consumed.saturating_add(count);
    if count == 0 || *consumed > MAX_HEADER_BYTES || !line.ends_with(b"\n") {
        return Err(embedding_unavailable(
            "model response headers are truncated or oversized",
        ));
    }
    if line.ends_with(b"\n") {
        line.pop();
    }
    if line.ends_with(b"\r") {
        line.pop();
    }
    String::from_utf8(line)
        .map_err(|_| embedding_unavailable("model response headers are not valid ASCII/UTF-8"))
}

fn copy_exact<R: Read>(
    reader: &mut R,
    destination: &mut File,
    hasher: &mut Sha256,
    length: u64,
    sink: &ProgressSink,
) -> Result<u64> {
    let mut remaining = length;
    let mut copied = 0_u64;
    let mut buffer = [0_u8; 1024 * 1024];
    while remaining > 0 {
        let limit = usize::try_from(remaining.min(buffer.len() as u64))
            .map_err(|_| embedding_unavailable("model body length exceeds this platform"))?;
        let read = reader
            .read(&mut buffer[..limit])
            .map_err(|error| embedding_io("reading the model body", error))?;
        if read == 0 {
            return Err(embedding_unavailable("model response body ended early"));
        }
        destination
            .write_all(&buffer[..read])
            .map_err(|error| embedding_io("writing the partial model", error))?;
        hasher.update(&buffer[..read]);
        let read = u64::try_from(read)
            .map_err(|_| embedding_unavailable("model read length exceeds u64"))?;
        copied = copied
            .checked_add(read)
            .ok_or_else(|| embedding_unavailable("model transfer length overflowed"))?;
        sink.add(read);
        remaining -= read;
    }
    Ok(copied)
}

fn copy_to_eof<R: Read>(
    reader: &mut R,
    destination: &mut File,
    hasher: &mut Sha256,
    maximum: u64,
    sink: &ProgressSink,
) -> Result<u64> {
    let mut copied = 0_u64;
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let read = reader
            .read(&mut buffer)
            .map_err(|error| embedding_io("reading the model body", error))?;
        if read == 0 {
            break;
        }
        let read_u64 = u64::try_from(read)
            .map_err(|_| embedding_unavailable("model read length exceeds u64"))?;
        copied = copied
            .checked_add(read_u64)
            .filter(|total| *total <= maximum)
            .ok_or_else(|| embedding_unavailable("model response exceeds its pinned size"))?;
        destination
            .write_all(&buffer[..read])
            .map_err(|error| embedding_io("writing the partial model", error))?;
        hasher.update(&buffer[..read]);
        sink.add(read_u64);
    }
    Ok(copied)
}

fn copy_chunked<R: BufRead>(
    reader: &mut R,
    destination: &mut File,
    hasher: &mut Sha256,
    maximum: u64,
    sink: &ProgressSink,
) -> Result<u64> {
    let mut copied = 0_u64;
    loop {
        let mut consumed = 0;
        let line = read_header_line(reader, &mut consumed)?;
        let size = line
            .split(';')
            .next()
            .and_then(|value| u64::from_str_radix(value.trim(), 16).ok())
            .ok_or_else(|| embedding_unavailable("model response has an invalid chunk size"))?;
        if size == 0 {
            loop {
                let trailer = read_header_line(reader, &mut consumed)?;
                if trailer.is_empty() {
                    return Ok(copied);
                }
            }
        }
        copied = copied
            .checked_add(size)
            .filter(|value| *value <= maximum)
            .ok_or_else(|| embedding_unavailable("chunked model body exceeds its pinned size"))?;
        let _ = copy_exact(reader, destination, hasher, size, sink)?;
        let mut ending = [0_u8; 2];
        reader
            .read_exact(&mut ending)
            .map_err(|error| embedding_io("reading model chunk terminator", error))?;
        if ending != *b"\r\n" {
            return Err(embedding_unavailable(
                "model response has an invalid chunk terminator",
            ));
        }
    }
}

fn validate_content_range(value: Option<&str>, start: u64, total: u64) -> Result<()> {
    let value =
        value.ok_or_else(|| embedding_unavailable("range response has no Content-Range"))?;
    let value = value
        .strip_prefix("bytes ")
        .ok_or_else(|| embedding_unavailable("model Content-Range unit is invalid"))?;
    let (range, declared_total) = value
        .split_once('/')
        .ok_or_else(|| embedding_unavailable("model Content-Range is invalid"))?;
    let (declared_start, declared_end) = range
        .split_once('-')
        .ok_or_else(|| embedding_unavailable("model Content-Range is invalid"))?;
    let declared_start = declared_start
        .parse::<u64>()
        .map_err(|_| embedding_unavailable("model Content-Range start is invalid"))?;
    let declared_end = declared_end
        .parse::<u64>()
        .map_err(|_| embedding_unavailable("model Content-Range end is invalid"))?;
    let declared_total = declared_total
        .parse::<u64>()
        .map_err(|_| embedding_unavailable("model Content-Range total is invalid"))?;
    if declared_start != start || declared_total != total || total == 0 || declared_end != total - 1
    {
        return Err(embedding_unavailable(
            "model Content-Range differs from the pinned object",
        ));
    }
    Ok(())
}

fn hash_prefix(file: &mut File, length: u64) -> Result<(Sha256, u64)> {
    file.seek(SeekFrom::Start(0))
        .map_err(|error| embedding_io("seeking the partial model for hashing", error))?;
    let mut hasher = Sha256::new();
    let mut consumed = 0_u64;
    let mut buffer = [0_u8; 1024 * 1024];
    while consumed < length {
        let limit = usize::try_from((length - consumed).min(buffer.len() as u64))
            .map_err(|_| embedding_unavailable("partial model size exceeds this platform"))?;
        let read = file
            .read(&mut buffer[..limit])
            .map_err(|error| embedding_io("hashing the partial model", error))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
        consumed += u64::try_from(read)
            .map_err(|_| embedding_unavailable("partial model read length exceeds u64"))?;
    }
    Ok((hasher, consumed))
}

/// Where the record of a completed verification lives, beside the artifact it describes.
fn verified_marker(path: &Path) -> PathBuf {
    path.with_extension("verified")
}

fn hex_digest(digest: [u8; 32]) -> String {
    use std::fmt::Write as _;
    digest
        .iter()
        .fold(String::with_capacity(64), |mut out, byte| {
            let _ = write!(out, "{byte:02x}");
            out
        })
}

/// Verify an installed artifact, hashing it only the first time.
///
/// The size check stays on every path — it is a stat, and a truncated file must never be trusted.
/// Hashing the encoder is expensive, so a successful verification is cached with file identity.
/// The digest need not be recomputed on every start to re-establish something proven when the
/// bytes arrived. It runs once, and
/// the digest that was checked is recorded next to the file; a record naming a different digest, or
/// no record at all, hashes again.
///
/// This is not a defence against someone who can write to the embedding directory — they could write
/// the marker too. It is the same guarantee as before against corruption and against the wrong file
/// being served, which is what the check was for.
fn verify_exact(path: &Path, artifact: &PinnedArtifact) -> Result<()> {
    let recorded = std::fs::read_to_string(verified_marker(path)).ok();
    if recorded.as_deref().map(str::trim) == Some(&hex_digest(artifact.sha256)) {
        let length = std::fs::metadata(path)
            .map_err(|error| embedding_io("reading installed model metadata", error))?
            .len();
        if length == artifact.exact_bytes {
            return Ok(());
        }
    }
    verify_exact_by_hash(path, artifact)?;
    let _ = std::fs::write(verified_marker(path), hex_digest(artifact.sha256));
    Ok(())
}

fn verify_exact_by_hash(path: &Path, artifact: &PinnedArtifact) -> Result<()> {
    let mut file = open_private_rw(path, false)?;
    let length = file
        .metadata()
        .map_err(|error| embedding_io("reading installed model metadata", error))?
        .len();
    if length != artifact.exact_bytes {
        return Err(embedding_unavailable(
            "installed model size differs from the pinned artifact",
        ));
    }
    let (hasher, consumed) = hash_prefix(&mut file, length)?;
    if consumed != length {
        return Err(embedding_unavailable(
            "installed model changed while it was verified",
        ));
    }
    let actual: [u8; 32] = hasher.finalize().into();
    if actual != artifact.sha256 {
        return Err(embedding_unavailable(
            "installed model SHA-256 differs from the pinned artifact",
        ));
    }
    Ok(())
}

fn ensure_private_hierarchy(directory: &Path) -> Result<()> {
    let mut current = PathBuf::new();
    let mut private = false;
    for component in directory.components() {
        current.push(component.as_os_str());
        if current == Path::new("/") {
            continue;
        }
        if component.as_os_str() == ".irongraph" {
            private = true;
        }
        if !private {
            continue;
        }
        match fs::symlink_metadata(&current) {
            Ok(metadata) => {
                if metadata.file_type().is_symlink() || !metadata.is_dir() {
                    return Err(embedding_unavailable(format!(
                        "model directory component is not a real directory: {}",
                        current.display()
                    )));
                }
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
                fs::create_dir(&current)
                    .map_err(|error| embedding_io("creating the private model directory", error))?;
            }
            Err(error) => return Err(embedding_io("checking the private model directory", error)),
        }
        set_private_directory_mode(&current)?;
    }
    Ok(())
}

fn open_private_rw(path: &Path, truncate: bool) -> Result<File> {
    if let Ok(metadata) = fs::symlink_metadata(path)
        && (metadata.file_type().is_symlink() || !metadata.is_file())
    {
        return Err(embedding_unavailable(format!(
            "model installation path is not a regular file: {}",
            path.display()
        )));
    }
    let mut options = OpenOptions::new();
    options
        .read(true)
        .write(true)
        .create(true)
        .truncate(truncate);
    #[cfg(unix)]
    options
        .mode(0o600)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
    let file = options
        .open(path)
        .map_err(|error| embedding_io("opening the private model artifact", error))?;
    set_private_file_mode(path)?;
    Ok(file)
}

fn quarantine_invalid(path: &Path) -> Result<()> {
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| embedding_unavailable("model artifact has no valid file name"))?;
    let quarantine = path.with_file_name(format!("{file_name}.invalid"));
    if quarantine.exists() {
        let metadata = fs::symlink_metadata(&quarantine)
            .map_err(|error| embedding_io("checking quarantined model", error))?;
        if metadata.file_type().is_symlink() || !metadata.is_file() {
            return Err(embedding_unavailable(
                "model quarantine path is not a regular file",
            ));
        }
        fs::remove_file(&quarantine)
            .map_err(|error| embedding_io("replacing quarantined model", error))?;
    }
    fs::rename(path, quarantine)
        .map_err(|error| embedding_io("quarantining an invalid model", error))
}

#[cfg(unix)]
fn set_private_directory_mode(path: &Path) -> Result<()> {
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))
        .map_err(|error| embedding_io("setting private model directory permissions", error))
}

#[cfg(not(unix))]
fn set_private_directory_mode(_: &Path) -> Result<()> {
    Err(embedding_unavailable(
        "private model installation is unavailable on this platform",
    ))
}

#[cfg(unix)]
fn set_private_file_mode(path: &Path) -> Result<()> {
    fs::set_permissions(path, fs::Permissions::from_mode(0o600))
        .map_err(|error| embedding_io("setting private model file permissions", error))
}

#[cfg(not(unix))]
fn set_private_file_mode(_: &Path) -> Result<()> {
    Err(embedding_unavailable(
        "private model installation is unavailable on this platform",
    ))
}

fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| embedding_io("syncing the model directory", error))
}

fn require_https(url: &Url) -> Result<()> {
    if url.scheme() != "https" || url.username() != "" || url.password().is_some() {
        return Err(embedding_unavailable(
            "model acquisition permits HTTPS URLs without embedded credentials only",
        ));
    }
    Ok(())
}

fn embedding_io(context: &str, error: std::io::Error) -> Error {
    embedding_unavailable(format!("{context} failed: {error}"))
}

fn embedding_unavailable(message: impl Into<String>) -> Error {
    Error::new(ErrorCode::EmbeddingUnavailable, message.into())
}

#[cfg(test)]
mod tests {
    use super::{
        DEFAULT_EMBEDDING_CONFIG_BYTES, DEFAULT_EMBEDDING_MODEL_BYTES,
        DEFAULT_EMBEDDING_TOKENIZER_BYTES, PinnedArtifact, ProgressSink, copy_exact,
        ensure_default_embedding_model, hash_prefix, open_private_rw, validate_content_range,
        verify_exact,
    };
    use sha2::{Digest as _, Sha256};
    use std::io::{Read as _, Seek as _, SeekFrom, Write as _};

    #[test]
    fn pinned_verification_rejects_size_and_digest_mismatch() -> crate::Result<()> {
        let directory = tempfile::tempdir().map_err(crate::Error::from)?;
        let directory = std::fs::canonicalize(directory.path()).map_err(crate::Error::from)?;
        let path = directory.join("tiny.bin");
        let bytes = b"tiny verified artifact";
        std::fs::write(&path, bytes).map_err(crate::Error::from)?;
        let spec = PinnedArtifact {
            url: "https://example.invalid/tiny.bin",
            file_name: "tiny.bin",
            exact_bytes: bytes.len() as u64,
            sha256: Sha256::digest(bytes).into(),
        };
        verify_exact(&path, &spec)?;
        let mut wrong = spec.clone();
        wrong.sha256[0] ^= 1;
        assert!(verify_exact(&path, &wrong).is_err());
        wrong = spec;
        wrong.exact_bytes += 1;
        assert!(verify_exact(&path, &wrong).is_err());
        Ok(())
    }

    #[test]
    fn copy_exact_reports_written_progress_and_hashes() -> crate::Result<()> {
        let body = vec![0x5a_u8; 3_500_000]; // spans multiple 1 MiB buffers
        let mut destination = tempfile::tempfile().map_err(crate::Error::from)?;
        let mut hasher = Sha256::new();
        let sink = ProgressSink::new();
        let mut reader = body.as_slice();
        let copied = copy_exact(
            &mut reader,
            &mut destination,
            &mut hasher,
            body.len() as u64,
            &sink,
        )?;
        assert_eq!(copied, body.len() as u64);
        assert_eq!(sink.written(), body.len() as u64);
        let expected: [u8; 32] = Sha256::digest(&body).into();
        assert_eq!(<[u8; 32]>::from(hasher.finalize()), expected);
        destination
            .seek(SeekFrom::Start(0))
            .map_err(crate::Error::from)?;
        let mut written = Vec::new();
        destination
            .read_to_end(&mut written)
            .map_err(crate::Error::from)?;
        assert_eq!(written, body);
        Ok(())
    }

    #[test]
    fn resume_hash_and_content_range_are_exact() -> crate::Result<()> {
        let mut file = tempfile::tempfile().map_err(crate::Error::from)?;
        file.write_all(b"prefix").map_err(crate::Error::from)?;
        let (hash, length) = hash_prefix(&mut file, 6)?;
        assert_eq!(length, 6);
        let expected: [u8; 32] = Sha256::digest(b"prefix").into();
        assert_eq!(<[u8; 32]>::from(hash.finalize()), expected);
        validate_content_range(Some("bytes 6-9/10"), 6, 10)?;
        assert!(validate_content_range(Some("bytes 5-9/10"), 6, 10).is_err());
        assert!(validate_content_range(Some("bytes 6-8/10"), 6, 10).is_err());
        Ok(())
    }

    #[test]
    fn partial_reopen_preserves_resumable_bytes() -> crate::Result<()> {
        let directory = tempfile::tempdir().map_err(crate::Error::from)?;
        let directory = std::fs::canonicalize(directory.path()).map_err(crate::Error::from)?;
        let path = directory.join("model.part");
        {
            let mut partial = open_private_rw(&path, false)?;
            partial
                .write_all(b"resumable-prefix")
                .map_err(crate::Error::from)?;
            partial.sync_all().map_err(crate::Error::from)?;
        }
        let partial = open_private_rw(&path, false)?;
        assert_eq!(
            partial.metadata().map_err(crate::Error::from)?.len(),
            b"resumable-prefix".len() as u64
        );
        Ok(())
    }

    #[test]
    #[ignore = "verifies and, when needed, acquires the full pinned embedding installation"]
    fn pinned_embedding_installation_contains_every_exact_file() -> crate::Result<()> {
        let artifacts = ensure_default_embedding_model()?;
        assert_eq!(
            std::fs::metadata(&artifacts.model_safetensors)
                .map_err(crate::Error::from)?
                .len(),
            DEFAULT_EMBEDDING_MODEL_BYTES
        );
        assert_eq!(
            std::fs::metadata(&artifacts.model_config_json)
                .map_err(crate::Error::from)?
                .len(),
            DEFAULT_EMBEDDING_CONFIG_BYTES
        );
        assert_eq!(
            std::fs::metadata(&artifacts.tokenizer_json)
                .map_err(crate::Error::from)?
                .len(),
            DEFAULT_EMBEDDING_TOKENIZER_BYTES
        );
        Ok(())
    }
}
