use std::{
    fs::{self, File, OpenOptions},
    io::{self, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    pin::Pin,
    task::{Context, Poll},
};

use tokio::io::{AsyncRead, AsyncSeek, AsyncWrite, ReadBuf};
use uuid::Uuid;

use crate::{Error, ErrorCode, Result};

const RECEIVE_CHUNK_BYTES: u64 = 8 * 1024 * 1024;

/// One independently checksummed immutable file carried by a state-machine snapshot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SnapshotAttachment {
    pub(crate) path: PathBuf,
    pub(crate) bytes: u64,
    pub(crate) checksum: [u8; 32],
}

impl SnapshotAttachment {
    pub(crate) fn new(path: PathBuf, bytes: u64, checksum: [u8; 32]) -> Result<Self> {
        let metadata = fs::symlink_metadata(&path)?;
        if !metadata.is_file()
            || metadata.file_type().is_symlink()
            || bytes == 0
            || metadata.len() != bytes
            || checksum == [0_u8; 32]
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "snapshot attachment metadata is invalid",
            ));
        }
        Ok(Self {
            path,
            bytes,
            checksum,
        })
    }
}

/// Bookmark-exact canonical state plus immutable byte attachments. The state file contains only
/// canonical coordination/graph state and verified manifests; broker and retained authority
/// bytes remain separate files throughout construction, transfer, recovery, and installation.
#[derive(Clone, Debug)]
pub struct BackendSnapshot {
    pub(crate) state_path: PathBuf,
    pub(crate) state_bytes: u64,
    pub(crate) attachments: Vec<SnapshotAttachment>,
}

impl BackendSnapshot {
    pub fn state_only(state_path: PathBuf, state_bytes: u64) -> Result<Self> {
        Self::new(state_path, state_bytes, Vec::new())
    }

    #[must_use]
    pub fn state_path(&self) -> &Path {
        &self.state_path
    }

    pub(crate) fn new(
        state_path: PathBuf,
        state_bytes: u64,
        mut attachments: Vec<SnapshotAttachment>,
    ) -> Result<Self> {
        let metadata = fs::symlink_metadata(&state_path)?;
        if !metadata.is_file()
            || metadata.file_type().is_symlink()
            || state_bytes == 0
            || metadata.len() != state_bytes
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "snapshot canonical state metadata is invalid",
            ));
        }
        attachments.sort_by_key(|attachment| attachment.checksum);
        for adjacent in attachments.windows(2) {
            if adjacent[0].checksum == adjacent[1].checksum
                && (adjacent[0].bytes != adjacent[1].bytes || adjacent[0].path != adjacent[1].path)
            {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "snapshot attachment content address is ambiguous",
                ));
            }
        }
        attachments.dedup_by_key(|attachment| attachment.checksum);
        Ok(Self {
            state_path,
            state_bytes,
            attachments,
        })
    }
}

#[derive(Debug)]
enum SnapshotDataMode {
    Read {
        parts: Vec<SnapshotPart>,
        position: u64,
        bytes: u64,
    },
    Receive {
        directory: PathBuf,
        position: u64,
        bytes: u64,
        chunk_bytes: u64,
    },
}

#[derive(Debug)]
struct SnapshotPart {
    file: File,
    start: u64,
    bytes: u64,
}

/// Snapshot data backed by a logical concatenation of immutable files. Published snapshots keep
/// large payload segments as independent content-addressed files while exposing one seekable byte
/// stream. Recovery stores that stream in bounded files rather than one monolithic inode.
#[derive(Debug)]
pub struct SnapshotData {
    mode: SnapshotDataMode,
    pending_seek: Option<io::Result<u64>>,
}

impl SnapshotData {
    pub(crate) fn from_parts(paths: Vec<PathBuf>) -> Result<Self> {
        let mut files = Vec::with_capacity(paths.len());
        for path in paths {
            let metadata = fs::symlink_metadata(&path)?;
            if !metadata.is_file() || metadata.file_type().is_symlink() {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "snapshot transfer part is not a regular file",
                ));
            }
            files.push(File::open(path)?);
        }
        Self::from_files(files)
    }

    pub(crate) fn from_open_file_and_paths(first: File, paths: Vec<PathBuf>) -> Result<Self> {
        let mut files = Vec::with_capacity(paths.len().saturating_add(1));
        files.push(first);
        for path in paths {
            let metadata = fs::symlink_metadata(&path)?;
            if !metadata.is_file() || metadata.file_type().is_symlink() {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "snapshot transfer part is not a regular file",
                ));
            }
            files.push(File::open(path)?);
        }
        Self::from_files(files)
    }

    fn from_files(files: Vec<File>) -> Result<Self> {
        if files.is_empty() {
            return Err(Error::invalid_data("snapshot has no transfer parts"));
        }
        let mut parts = Vec::with_capacity(files.len());
        let mut start = 0_u64;
        for mut file in files {
            let metadata = file.metadata()?;
            if !metadata.is_file() {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "snapshot transfer part is not a regular file",
                ));
            }
            let bytes = metadata.len();
            if bytes == 0 {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "snapshot transfer part is empty",
                ));
            }
            file.seek(SeekFrom::Start(0))?;
            parts.push(SnapshotPart { file, start, bytes });
            start = start.checked_add(bytes).ok_or_else(|| {
                Error::new(ErrorCode::ResultBudgetExceeded, "snapshot size overflow")
            })?;
        }
        Ok(Self {
            mode: SnapshotDataMode::Read {
                parts,
                position: 0,
                bytes: start,
            },
            pending_seek: None,
        })
    }

    pub(crate) fn receiving(parent: &Path) -> Result<Self> {
        Self::receiving_with_chunk_bytes(parent, RECEIVE_CHUNK_BYTES)
    }

    fn receiving_with_chunk_bytes(parent: &Path, chunk_bytes: u64) -> Result<Self> {
        if chunk_bytes == 0 {
            return Err(Error::invalid_data(
                "snapshot receive chunk size must be non-zero",
            ));
        }
        let metadata = fs::symlink_metadata(parent)?;
        if !metadata.is_dir() || metadata.file_type().is_symlink() {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "snapshot receive parent is not a real directory",
            ));
        }
        let directory = parent.join(format!(".receive.{}.snapshot.parts", Uuid::new_v4()));
        fs::create_dir(&directory)?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt as _;
            fs::set_permissions(&directory, fs::Permissions::from_mode(0o700))?;
        }
        crate::storage::sync_durable(&File::open(parent)?)?;
        Ok(Self {
            mode: SnapshotDataMode::Receive {
                directory,
                position: 0,
                bytes: 0,
                chunk_bytes,
            },
            pending_seek: None,
        })
    }

    #[must_use]
    pub(crate) fn len(&self) -> u64 {
        match &self.mode {
            SnapshotDataMode::Read { bytes, .. } | SnapshotDataMode::Receive { bytes, .. } => {
                *bytes
            }
        }
    }

    /// Flushes every chunk this snapshot owns through the platform durability barrier.
    pub(crate) fn sync_durable_chunks(&self) -> Result<()> {
        let SnapshotDataMode::Receive {
            directory,
            bytes,
            chunk_bytes,
            ..
        } = &self.mode
        else {
            return Ok(());
        };
        let count = bytes.div_ceil(*chunk_bytes);
        for index in 0..count {
            crate::storage::sync_durable(&File::open(receive_chunk_path(directory, index))?)?;
        }
        crate::storage::sync_durable(&File::open(directory)?)?;
        Ok(())
    }

    fn position(&self) -> u64 {
        match &self.mode {
            SnapshotDataMode::Read { position, .. }
            | SnapshotDataMode::Receive { position, .. } => *position,
        }
    }

    fn set_position(&mut self, position: u64) {
        match &mut self.mode {
            SnapshotDataMode::Read {
                position: current, ..
            }
            | SnapshotDataMode::Receive {
                position: current, ..
            } => *current = position,
        }
    }

    fn checked_seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        let current = i128::from(self.position());
        let end = i128::from(self.len());
        let next = match position {
            SeekFrom::Start(value) => i128::from(value),
            SeekFrom::Current(value) => current + i128::from(value),
            SeekFrom::End(value) => end + i128::from(value),
        };
        if next < 0 || next > i128::from(u64::MAX) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "snapshot seek is outside the u64 domain",
            ));
        }
        let next = next as u64;
        self.set_position(next);
        Ok(next)
    }
}

impl Read for SnapshotData {
    fn read(&mut self, output: &mut [u8]) -> io::Result<usize> {
        if output.is_empty() {
            return Ok(0);
        }
        match &mut self.mode {
            SnapshotDataMode::Read {
                parts,
                position,
                bytes,
            } => {
                if *position >= *bytes {
                    return Ok(0);
                }
                let index = parts
                    .partition_point(|part| part.start.saturating_add(part.bytes) <= *position);
                let part = parts.get_mut(index).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::UnexpectedEof, "snapshot part is absent")
                })?;
                let within = position.saturating_sub(part.start);
                let available = part.bytes.saturating_sub(within);
                let requested = usize::try_from(available.min(output.len() as u64))
                    .map_err(|_| io::Error::other("snapshot read length overflow"))?;
                part.file.seek(SeekFrom::Start(within))?;
                let read = part.file.read(&mut output[..requested])?;
                *position = position
                    .checked_add(read as u64)
                    .ok_or_else(|| io::Error::other("snapshot position overflow"))?;
                Ok(read)
            }
            SnapshotDataMode::Receive {
                directory,
                position,
                bytes,
                chunk_bytes,
            } => {
                if *position >= *bytes {
                    return Ok(0);
                }
                let index = *position / *chunk_bytes;
                let within = *position % *chunk_bytes;
                let available = (*chunk_bytes - within).min(*bytes - *position);
                let requested = usize::try_from(available.min(output.len() as u64))
                    .map_err(|_| io::Error::other("snapshot receive read length overflow"))?;
                let mut file = File::open(receive_chunk_path(directory, index))?;
                file.seek(SeekFrom::Start(within))?;
                let read = file.read(&mut output[..requested])?;
                *position = position
                    .checked_add(read as u64)
                    .ok_or_else(|| io::Error::other("snapshot receive position overflow"))?;
                Ok(read)
            }
        }
    }
}

impl Write for SnapshotData {
    fn write(&mut self, input: &[u8]) -> io::Result<usize> {
        if input.is_empty() {
            return Ok(0);
        }
        let SnapshotDataMode::Receive {
            directory,
            position,
            bytes,
            chunk_bytes,
        } = &mut self.mode
        else {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "published snapshot data is read-only",
            ));
        };
        if *position > *bytes {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "snapshot receive cannot create a sparse logical gap",
            ));
        }
        let index = *position / *chunk_bytes;
        let within = *position % *chunk_bytes;
        let available = *chunk_bytes - within;
        let requested = usize::try_from(available.min(input.len() as u64))
            .map_err(|_| io::Error::other("snapshot receive write length overflow"))?;
        let path = receive_chunk_path(directory, index);
        let mut options = OpenOptions::new();
        options.read(true).write(true).create(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt as _;
            options.mode(0o600);
        }
        let mut file = options.open(path)?;
        file.seek(SeekFrom::Start(within))?;
        file.write_all(&input[..requested])?;
        *position = position
            .checked_add(requested as u64)
            .ok_or_else(|| io::Error::other("snapshot receive position overflow"))?;
        *bytes = (*bytes).max(*position);
        Ok(requested)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.sync_durable_chunks().map_err(io::Error::other)
    }
}

impl Seek for SnapshotData {
    fn seek(&mut self, position: SeekFrom) -> io::Result<u64> {
        self.checked_seek(position)
    }
}

impl AsyncRead for SnapshotData {
    fn poll_read(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<io::Result<()>> {
        let output = buffer.initialize_unfilled();
        match Read::read(&mut *self, output) {
            Ok(read) => {
                buffer.advance(read);
                Poll::Ready(Ok(()))
            }
            Err(error) => Poll::Ready(Err(error)),
        }
    }
}

impl AsyncWrite for SnapshotData {
    fn poll_write(
        mut self: Pin<&mut Self>,
        _cx: &mut Context<'_>,
        input: &[u8],
    ) -> Poll<io::Result<usize>> {
        Poll::Ready(Write::write(&mut *self, input))
    }

    fn poll_flush(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(Write::flush(&mut *self))
    }

    fn poll_shutdown(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<()>> {
        Poll::Ready(self.sync_durable_chunks().map_err(io::Error::other))
    }
}

impl AsyncSeek for SnapshotData {
    fn start_seek(mut self: Pin<&mut Self>, position: SeekFrom) -> io::Result<()> {
        let result = self.checked_seek(position);
        self.pending_seek = Some(result);
        Ok(())
    }

    fn poll_complete(mut self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<io::Result<u64>> {
        Poll::Ready(
            self.pending_seek
                .take()
                .unwrap_or_else(|| Ok(self.position())),
        )
    }
}

impl Drop for SnapshotData {
    fn drop(&mut self) {
        if let SnapshotDataMode::Receive { directory, .. } = &self.mode {
            let _ignored = fs::remove_dir_all(directory);
        }
    }
}

fn receive_chunk_path(directory: &Path, index: u64) -> PathBuf {
    directory.join(format!("chunk-{index:016x}.bin"))
}

#[cfg(test)]
pub(crate) fn checked_snapshot_stream_bytes(
    metadata_bytes: u64,
    manifest_bytes: u64,
    attachments: impl IntoIterator<Item = u64>,
) -> Result<u64> {
    attachments.into_iter().try_fold(
        metadata_bytes
            .checked_add(manifest_bytes)
            .ok_or_else(|| Error::new(ErrorCode::ResultBudgetExceeded, "snapshot size overflow"))?,
        |total, bytes| {
            total.checked_add(bytes).ok_or_else(|| {
                Error::new(ErrorCode::ResultBudgetExceeded, "snapshot size overflow")
            })
        },
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn logical_sender_and_chunked_receiver_support_seek_and_rewrite() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let first = directory.path().join("first");
        let second = directory.path().join("second");
        fs::write(&first, b"abc")?;
        fs::write(&second, b"defgh")?;
        let mut sender = SnapshotData::from_parts(vec![first, second])?;
        let mut received = SnapshotData::receiving_with_chunk_bytes(directory.path(), 3)?;
        io::copy(&mut sender, &mut received)?;
        assert_eq!(received.len(), 8);

        received.seek(SeekFrom::Start(2))?;
        received.write_all(b"XY")?;
        received.seek(SeekFrom::Start(0))?;
        let mut bytes = Vec::new();
        received.read_to_end(&mut bytes)?;
        assert_eq!(bytes, b"abXYefgh");
        Ok(())
    }

    #[test]
    fn stream_capacity_arithmetic_has_no_sixty_four_gibibyte_ceiling() -> Result<()> {
        let sixty_four_gib = 64_u64 * 1024 * 1024 * 1024;
        assert_eq!(
            checked_snapshot_stream_bytes(1024, 2048, [sixty_four_gib, 17])?,
            sixty_four_gib + 3089
        );
        assert!(checked_snapshot_stream_bytes(u64::MAX, 1, []).is_err());
        Ok(())
    }
}
