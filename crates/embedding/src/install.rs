use super::EmbeddingModelArtifacts;
use crate::{
    Error, ErrorCode, Result,
    graph::{EmbeddingDType, EmbeddingProfile, Similarity},
};
use rustls::{ClientConfig, ClientConnection, RootCertStore, StreamOwned};
#[cfg(unix)]
use std::os::unix::fs::{OpenOptionsExt as _, PermissionsExt as _};
use std::{
    collections::BTreeMap,
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
use url::Url;

pub const DEFAULT_EMBEDDING_MODEL_ID: &str = "hotchpotch/bekko-embedding-v1-a8m";
pub const DEFAULT_EMBEDDING_REVISION: &str = "c721113d59a1d91b447450324f51c4b3332c924a";
pub const DEFAULT_EMBEDDING_MODEL_FILE: &str = "model.safetensors";
pub const DEFAULT_EMBEDDING_CONFIG_FILE: &str = "config.json";
pub const DEFAULT_EMBEDDING_TOKENIZER_FILE: &str = "tokenizer.json";
const MODEL_BASE: &str = "https://huggingface.co/hotchpotch/bekko-embedding-v1-a8m/resolve/c721113d59a1d91b447450324f51c4b3332c924a/";
const MAX_REDIRECTS: usize = 8;
const MAX_HEADER_BYTES: usize = 64 * 1024;
const NETWORK_TIMEOUT: Duration = Duration::from_secs(60);

/// Download progress is reported from HTTP metadata, without checksumming model files.
#[derive(Clone, Debug, Default)]
pub struct ProgressSink {
    written: Arc<AtomicU64>,
    total: Arc<AtomicU64>,
    cancellation: tokio_util::sync::CancellationToken,
}
impl ProgressSink {
    fn check(&self) -> Result<()> {
        if self.cancellation.is_cancelled() {
            return Err(Error::new(
                ErrorCode::Cancelled,
                "model installation was cancelled",
            ));
        }
        Ok(())
    }
    fn add(&self, bytes: u64) {
        self.written.fetch_add(bytes, Ordering::Relaxed);
    }
    fn set(&self, bytes: u64) {
        self.written.store(bytes, Ordering::Relaxed);
    }
    #[must_use]
    pub fn written(&self) -> u64 {
        self.written.load(Ordering::Relaxed)
    }
}

#[derive(Clone, Debug, Default)]
pub struct ModelInstallProgress {
    model: ProgressSink,
    config: ProgressSink,
    tokenizer: ProgressSink,
}
impl ModelInstallProgress {
    pub fn cancel(&self) {
        self.model.cancellation.cancel();
        self.config.cancellation.cancel();
        self.tokenizer.cancellation.cancel();
    }
    #[must_use]
    pub fn written(&self) -> u64 {
        self.model
            .written()
            .saturating_add(self.config.written())
            .saturating_add(self.tokenizer.written())
    }
    /// Zero until every file's size is known; cached files contribute their local size.
    #[must_use]
    pub fn total_bytes(&self) -> u64 {
        let totals = [
            self.model.total.load(Ordering::Relaxed),
            self.config.total.load(Ordering::Relaxed),
            self.tokenizer.total.load(Ordering::Relaxed),
        ];
        if totals.contains(&0) {
            return 0;
        }
        totals.into_iter().fold(0, u64::saturating_add)
    }
}

/// Identifies the encoder recipe, not a checksum of downloaded content.
pub(super) fn default_profile() -> Result<EmbeddingProfile> {
    let model = format!("{DEFAULT_EMBEDDING_MODEL_ID}@{DEFAULT_EMBEDDING_REVISION}");
    EmbeddingProfile::new(
        *blake3::hash(model.as_bytes()).as_bytes(),
        *blake3::hash(b"bekko-a8m-raw-text-masked-mean-normalized384-v1").as_bytes(),
        384,
        EmbeddingDType::F16,
        true,
        Similarity::Cosine,
    )
}

pub fn ensure_default_embedding_model() -> Result<EmbeddingModelArtifacts> {
    ensure_default_embedding_model_with_progress(&ModelInstallProgress::default())
}
pub fn ensure_default_embedding_model_with_progress(
    progress: &ModelInstallProgress,
) -> Result<EmbeddingModelArtifacts> {
    ensure_default_embedding_model_with_storage(&crate::EmbeddingStorage::default(), progress)
}

pub fn ensure_default_embedding_model_with_storage(
    storage: &crate::EmbeddingStorage,
    progress: &ModelInstallProgress,
) -> Result<EmbeddingModelArtifacts> {
    let root = storage.model_directory()?;
    let family = root.join("hotchpotch-bekko-embedding-v1-a8m");
    ensure_private_hierarchy(&root)?;
    ensure_private_hierarchy(&family)?;
    install_in(&family.join(DEFAULT_EMBEDDING_REVISION), progress)
}
fn install_in(
    directory: &Path,
    progress: &ModelInstallProgress,
) -> Result<EmbeddingModelArtifacts> {
    Ok(EmbeddingModelArtifacts {
        model_safetensors: install_tracked(
            directory,
            DEFAULT_EMBEDDING_MODEL_FILE,
            &progress.model,
        )?,
        model_config_json: install_tracked(
            directory,
            DEFAULT_EMBEDDING_CONFIG_FILE,
            &progress.config,
        )?,
        tokenizer_json: install_tracked(
            directory,
            DEFAULT_EMBEDDING_TOKENIZER_FILE,
            &progress.tokenizer,
        )?,
        profile: default_profile()?,
        maximum_input_tokens: 8_192,
    })
}
fn install_tracked(directory: &Path, file_name: &str, sink: &ProgressSink) -> Result<PathBuf> {
    sink.check()?;
    ensure_private_hierarchy(directory)?;
    let lock = open_private_rw(&directory.join(".install.lock"), false)?;
    loop {
        sink.check()?;
        match lock.try_lock() {
            Ok(()) => break,
            Err(std::fs::TryLockError::WouldBlock) => std::thread::sleep(Duration::from_millis(10)),
            Err(std::fs::TryLockError::Error(error)) => {
                return Err(embedding_io("acquiring the model installation lock", error));
            }
        }
    }
    let final_path = directory.join(file_name);
    match fs::symlink_metadata(&final_path) {
        Ok(metadata) => {
            if !metadata.is_file() || metadata.file_type().is_symlink() || metadata.len() == 0 {
                return Err(embedding_unavailable(format!(
                    "cached model artifact is not a nonempty regular file: {}",
                    final_path.display()
                )));
            }
            set_private_file_mode(&final_path)?;
            sink.set(metadata.len());
            sink.total.store(metadata.len(), Ordering::Relaxed);
            return Ok(final_path);
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
        Err(error) => return Err(embedding_io("reading cached model metadata", error)),
    }
    let partial_path = directory.join(format!("{file_name}.part"));
    let mut partial = open_private_rw(&partial_path, false)?;
    let existing = partial
        .metadata()
        .map_err(|error| embedding_io("reading partial model metadata", error))?
        .len();
    partial
        .seek(SeekFrom::Start(existing))
        .map_err(|error| embedding_io("seeking the partial model", error))?;
    sink.set(existing);
    download_from(
        &format!("{MODEL_BASE}{file_name}?download=true"),
        existing,
        &mut partial,
        sink,
    )?;
    sink.check()?;
    let length = partial
        .metadata()
        .map_err(|error| embedding_io("reading downloaded model metadata", error))?
        .len();
    if length == 0 {
        return Err(embedding_unavailable("downloaded model artifact is empty"));
    }
    partial
        .sync_all()
        .map_err(|error| embedding_io("syncing the downloaded model", error))?;
    drop(partial);
    fs::rename(&partial_path, &final_path)
        .map_err(|error| embedding_io("installing the downloaded model", error))?;
    sync_directory(directory)?;
    sink.total.store(length, Ordering::Relaxed);
    Ok(final_path)
}

fn download_from(
    address: &str,
    existing: u64,
    destination: &mut File,
    sink: &ProgressSink,
) -> Result<()> {
    let mut url = Url::parse(address)
        .map_err(|error| embedding_unavailable(format!("model URL is invalid: {error}")))?;
    let mut response = None;
    for _ in 0..=MAX_REDIRECTS {
        sink.check()?;
        require_https(&url)?;
        let current = open_https(&url, existing)?;
        if matches!(current.status, 301 | 302 | 303 | 307 | 308) {
            let location = current
                .header("location")
                .ok_or_else(|| embedding_unavailable("model redirect has no Location header"))?;
            url = url.join(location).map_err(|error| {
                embedding_unavailable(format!("model redirect URL is invalid: {error}"))
            })?;
            continue;
        }
        response = Some(current);
        break;
    }
    let mut response = response.ok_or_else(|| embedding_unavailable("too many model redirects"))?;
    require_https(&response.url)?;
    if response
        .header("content-encoding")
        .is_some_and(|value| !value.eq_ignore_ascii_case("identity"))
    {
        return Err(embedding_unavailable(
            "model server applied an unsupported content encoding",
        ));
    }
    let (offset, expected) = match response.status {
        200 => {
            destination
                .set_len(0)
                .and_then(|()| destination.seek(SeekFrom::Start(0)).map(|_| ()))
                .map_err(|error| embedding_io("restarting a non-range model response", error))?;
            sink.set(0);
            (0, response.content_length()?)
        }
        206 => {
            let total = validate_content_range(response.header("content-range"), existing)?;
            (existing, Some(total - existing))
        }
        416 if existing > 0
            && response.header("content-range") == Some(format!("bytes */{existing}").as_str()) =>
        {
            sink.total.store(existing, Ordering::Relaxed);
            return Ok(());
        }
        status => {
            return Err(embedding_unavailable(format!(
                "model server returned HTTP status {status}"
            )));
        }
    };
    if let Some(length) = expected {
        let total = offset
            .checked_add(length)
            .ok_or_else(|| embedding_unavailable("model transfer length overflowed"))?;
        sink.total.store(total, Ordering::Relaxed);
        if response
            .content_length()?
            .is_some_and(|declared| declared != length)
        {
            return Err(embedding_unavailable(
                "model Content-Length differs from its Content-Range",
            ));
        }
    }
    let written = response.copy_body(destination, expected.unwrap_or(u64::MAX), sink)?;
    if expected.is_some_and(|length| length != written) {
        return Err(embedding_unavailable(
            "model response body ended before the declared length",
        ));
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
        maximum: u64,
        sink: &ProgressSink,
    ) -> Result<u64> {
        if self
            .header("transfer-encoding")
            .is_some_and(|value| value.eq_ignore_ascii_case("chunked"))
        {
            copy_chunked(&mut self.reader, destination, maximum, sink)
        } else if let Some(length) = self.content_length()? {
            if length > maximum {
                return Err(embedding_unavailable(
                    "model body exceeds its declared length",
                ));
            }
            copy_exact(&mut self.reader, destination, length, sink)
        } else {
            copy_to_eof(&mut self.reader, destination, maximum, sink)
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
    length: u64,
    sink: &ProgressSink,
) -> Result<u64> {
    let mut remaining = length;
    let mut copied = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    while remaining > 0 {
        sink.check()?;
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
    maximum: u64,
    sink: &ProgressSink,
) -> Result<u64> {
    let mut copied = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        sink.check()?;
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
            .ok_or_else(|| embedding_unavailable("model response exceeds its declared length"))?;
        destination
            .write_all(&buffer[..read])
            .map_err(|error| embedding_io("writing the partial model", error))?;
        sink.add(read_u64);
    }
    Ok(copied)
}

fn copy_chunked<R: BufRead>(
    reader: &mut R,
    destination: &mut File,
    maximum: u64,
    sink: &ProgressSink,
) -> Result<u64> {
    let mut copied = 0_u64;
    loop {
        sink.check()?;
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
            .ok_or_else(|| {
                embedding_unavailable("chunked model body exceeds its declared length")
            })?;
        let _ = copy_exact(reader, destination, size, sink)?;
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

fn validate_content_range(value: Option<&str>, start: u64) -> Result<u64> {
    let value = value
        .and_then(|value| value.strip_prefix("bytes "))
        .ok_or_else(|| embedding_unavailable("range response has no valid Content-Range"))?;
    let (range, total) = value
        .split_once('/')
        .ok_or_else(|| embedding_unavailable("model Content-Range is invalid"))?;
    let (first, last) = range
        .split_once('-')
        .ok_or_else(|| embedding_unavailable("model Content-Range is invalid"))?;
    let parse = |text: &str| {
        text.parse::<u64>()
            .map_err(|_| embedding_unavailable("model Content-Range number is invalid"))
    };
    let (first, last, total) = (parse(first)?, parse(last)?, parse(total)?);
    if first != start || first > last || total == 0 || last != total - 1 {
        return Err(embedding_unavailable(
            "model Content-Range does not cover the requested remainder",
        ));
    }
    Ok(total)
}

fn ensure_private_hierarchy(directory: &Path) -> Result<()> {
    let mut current = PathBuf::new();
    let mut private = false;
    for component in directory.components() {
        current.push(component.as_os_str());
        if current == Path::new("/") {
            continue;
        }
        // Existing ancestors belong to the host. Newly created app directories, the
        // final directory, and the default .irongraph hierarchy are private to IronGraph.
        if component.as_os_str() == ".irongraph" || current == directory || !current.exists() {
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
                match fs::create_dir(&current) {
                    Ok(()) => {}
                    Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => {
                        let metadata = fs::symlink_metadata(&current)
                            .map_err(|error| embedding_io("checking model directory", error))?;
                        if !metadata.is_dir() || metadata.file_type().is_symlink() {
                            return Err(embedding_unavailable(
                                "model directory is not a real directory",
                            ));
                        }
                    }
                    Err(error) => {
                        return Err(embedding_io("creating the private model directory", error));
                    }
                }
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

#[cfg(unix)]
fn set_private_directory_mode(path: &Path) -> Result<()> {
    fs::set_permissions(path, fs::Permissions::from_mode(0o700))
        .map_err(|error| embedding_io("setting private model directory permissions", error))
}

#[cfg(not(unix))]
fn set_private_directory_mode(_: &Path) -> Result<()> {
    // The model cache inherits the current user's home-directory access controls.
    Ok(())
}

#[cfg(unix)]
fn set_private_file_mode(path: &Path) -> Result<()> {
    fs::set_permissions(path, fs::Permissions::from_mode(0o600))
        .map_err(|error| embedding_io("setting private model file permissions", error))
}

#[cfg(not(unix))]
fn set_private_file_mode(_: &Path) -> Result<()> {
    Ok(())
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> Result<()> {
    File::open(path)
        .and_then(|directory| directory.sync_all())
        .map_err(|error| embedding_io("syncing the model directory", error))
}

#[cfg(not(unix))]
fn sync_directory(_: &Path) -> Result<()> {
    Ok(())
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
    use super::*;
    use std::io::Cursor;

    #[test]
    fn application_paths_install_and_reuse_models_outside_the_shared_home() -> Result<()> {
        let directory = tempfile::tempdir()?;
        for app in ["app-a", "app-b"] {
            let storage = crate::EmbeddingStorage {
                model_dir: Some(directory.path().join(app).join("models")),
                cache_dir: Some(directory.path().join(app).join("cache")),
            };
            let model_directory = storage
                .model_directory()?
                .join("hotchpotch-bekko-embedding-v1-a8m")
                .join(DEFAULT_EMBEDDING_REVISION);
            ensure_private_hierarchy(&model_directory)?;
            for file in [
                DEFAULT_EMBEDDING_MODEL_FILE,
                DEFAULT_EMBEDDING_CONFIG_FILE,
                DEFAULT_EMBEDDING_TOKENIZER_FILE,
            ] {
                fs::write(model_directory.join(file), app.as_bytes())?;
            }
            let artifacts = ensure_default_embedding_model_with_storage(
                &storage,
                &ModelInstallProgress::default(),
            )?;
            assert_eq!(
                artifacts.model_safetensors,
                model_directory.join(DEFAULT_EMBEDDING_MODEL_FILE)
            );
            assert_eq!(fs::read(&artifacts.model_safetensors)?, app.as_bytes());
            assert!(!storage.cache_directory()?.exists());
            #[cfg(unix)]
            assert_eq!(
                fs::metadata(&model_directory)?.permissions().mode() & 0o777,
                0o700
            );
        }
        assert!(!directory.path().join(".irongraph").exists());
        Ok(())
    }

    #[test]
    fn cached_files_are_reused_without_network_or_checksum_sidecars() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let progress = ModelInstallProgress::default();
        for name in [
            DEFAULT_EMBEDDING_MODEL_FILE,
            DEFAULT_EMBEDDING_CONFIG_FILE,
            DEFAULT_EMBEDDING_TOKENIZER_FILE,
        ] {
            fs::write(directory.path().join(name), b"cached artifact")?;
        }
        let artifacts = install_in(directory.path(), &progress)?;
        assert_eq!(artifacts.profile, default_profile()?);
        assert_eq!(progress.written(), 45);
        assert_eq!(progress.total_bytes(), 45);
        assert!(!directory.path().join("model.verified").exists());
        Ok(())
    }

    #[test]
    fn overall_progress_stays_indeterminate_while_any_size_is_unknown() {
        let progress = ModelInstallProgress::default();
        progress.model.total.store(100, Ordering::Relaxed);
        progress.model.set(100);
        progress.config.total.store(10, Ordering::Relaxed);
        progress.config.set(10);
        progress.tokenizer.set(50);
        assert_eq!(progress.written(), 160);
        assert_eq!(progress.total_bytes(), 0);
        progress.tokenizer.total.store(80, Ordering::Relaxed);
        assert_eq!(progress.total_bytes(), 190);
    }

    #[test]
    fn truncated_and_cancelled_transfers_do_not_report_success() -> Result<()> {
        let mut output = tempfile::tempfile()?;
        let sink = ProgressSink::default();
        let mut input = b"short".as_slice();
        assert!(copy_exact(&mut input, &mut output, 10, &sink).is_err());
        assert_eq!(output.metadata()?.len(), 5);
        sink.cancellation.cancel();
        let mut input = b"unused".as_slice();
        assert_eq!(
            copy_exact(&mut input, &mut output, 6, &sink)
                .unwrap_err()
                .code,
            ErrorCode::Cancelled
        );
        assert_eq!(output.metadata()?.len(), 5);
        Ok(())
    }

    #[test]
    fn chunked_download_and_progress_preserve_every_byte() -> Result<()> {
        let mut input = Cursor::new(b"3\r\nabc\r\n2\r\nde\r\n0\r\n\r\n");
        let mut output = tempfile::tempfile()?;
        let sink = ProgressSink::default();
        assert_eq!(copy_chunked(&mut input, &mut output, u64::MAX, &sink)?, 5);
        assert_eq!(sink.written(), 5);
        output.rewind()?;
        let mut bytes = Vec::new();
        output.read_to_end(&mut bytes)?;
        assert_eq!(bytes, b"abcde");
        let mut bad = Cursor::new(b"3\r\nab");
        assert!(copy_chunked(&mut bad, &mut output, u64::MAX, &sink).is_err());
        Ok(())
    }

    #[test]
    fn resume_range_requires_the_complete_requested_remainder() -> Result<()> {
        assert_eq!(validate_content_range(Some("bytes 6-9/10"), 6)?, 10);
        for value in [
            "bytes 5-9/10",
            "bytes 6-8/10",
            "bytes 10-9/10",
            "bytes 6-9/0",
            "items 6-9/10",
        ] {
            assert!(validate_content_range(Some(value), 6).is_err(), "{value}");
        }
        assert!(validate_content_range(None, 0).is_err());
        Ok(())
    }

    #[test]
    fn cancelled_installer_releases_a_waiting_lock_and_keeps_partial_data() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let lock = open_private_rw(&directory.path().join(".install.lock"), false)?;
        lock.lock()?;
        fs::write(directory.path().join("config.json.part"), b"prefix")?;
        let sink = ProgressSink::default();
        let worker_sink = sink.clone();
        let path = directory.path().to_owned();
        let (send, receive) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            send.send(install_tracked(
                &path,
                DEFAULT_EMBEDDING_CONFIG_FILE,
                &worker_sink,
            ))
            .unwrap();
        });
        sink.cancellation.cancel();
        let result = receive.recv_timeout(Duration::from_secs(2)).unwrap();
        drop(lock);
        worker.join().unwrap();
        assert_eq!(result.unwrap_err().code, ErrorCode::Cancelled);
        assert_eq!(
            fs::read(directory.path().join("config.json.part"))?,
            b"prefix"
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn installation_rejects_symlinked_files_and_directories() -> Result<()> {
        use std::os::unix::fs::symlink;
        let directory = tempfile::tempdir()?;
        let target = directory.path().join("target");
        fs::write(&target, b"weights")?;
        symlink(&target, directory.path().join(DEFAULT_EMBEDDING_MODEL_FILE))?;
        assert!(
            install_tracked(
                directory.path(),
                DEFAULT_EMBEDDING_MODEL_FILE,
                &ProgressSink::default()
            )
            .is_err()
        );
        assert_eq!(fs::read(&target)?, b"weights");
        let linked = directory.path().join(".irongraph");
        symlink(directory.path(), &linked)?;
        assert!(ensure_private_hierarchy(&linked).is_err());
        let custom = directory.path().join("app-models");
        symlink(directory.path(), &custom)?;
        let storage = crate::EmbeddingStorage {
            model_dir: Some(custom),
            cache_dir: Some(directory.path().join("cache")),
        };
        assert!(ensure_default_embedding_model_with_storage(
            &storage, &ModelInstallProgress::default(),
        ).is_err());
        Ok(())
    }

    #[test]
    fn resumable_file_reopen_does_not_truncate() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("weights.part");
        open_private_rw(&path, false)?.write_all(b"prefix")?;
        assert_eq!(open_private_rw(&path, false)?.metadata()?.len(), 6);
        Ok(())
    }

    #[test]
    #[ignore = "downloads and loads the local Bekko model"]
    fn default_embedding_installation_contains_all_files() -> Result<()> {
        let artifacts = ensure_default_embedding_model()?;
        for path in [
            artifacts.model_safetensors,
            artifacts.model_config_json,
            artifacts.tokenizer_json,
        ] {
            assert!(fs::metadata(path)?.len() > 0);
        }
        Ok(())
    }
}
