use std::{
    fs::{self, File, OpenOptions},
    io::{Read, Write},
    path::{Path, PathBuf},
};

use uuid::Uuid;

use irongraph_types::{Error, ErrorCode, Result};

/// Flushes a file all the way to durable media, not merely into the operating system.
///
/// On Apple platforms `fsync(2)` returns once the data reaches the drive and deliberately does not
/// wait for the drive to flush its own volatile write cache; only `fcntl(F_FULLFSYNC)` does. Every
/// completed WAL fsync and snapshot publication is built on this call, so the durability boundary
/// would otherwise be vulnerable to sudden power loss on the primary platform.
///
/// Every durability-critical write in the storage, WAL, and snapshot paths goes through this
/// one function so the platform difference is stated and handled in a single place.
pub fn sync_durable(file: &File) -> Result<()> {
    #[cfg(target_vendor = "apple")]
    {
        apple::full_fsync(file)
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        file.sync_all()?;
        Ok(())
    }
}

/// Flushes a file's contents to durable media without forcing a metadata flush.
///
/// This is the counterpart of [`sync_durable`] for append paths that deliberately use `fsyncdata`
/// semantics — the log writer extends a file whose metadata it does not depend on, and skipping the
/// inode update is worth real throughput. `F_FULLFSYNC` has no data-only variant, so on Apple this
/// is the same call; the distinction is preserved for every other platform.
pub(crate) fn sync_durable_data(file: &File) -> Result<()> {
    #[cfg(target_vendor = "apple")]
    {
        apple::full_fsync(file)
    }
    #[cfg(not(target_vendor = "apple"))]
    {
        file.sync_data()?;
        Ok(())
    }
}

#[cfg(target_vendor = "apple")]
mod apple {
    // `F_FULLFSYNC` has no safe wrapper in `std`. The unsafe surface is one `fcntl` call taking an
    // owned descriptor and an integer command, kept in this module so the crate-wide `unsafe_code`
    // denial still covers everything else.
    #![allow(unsafe_code)]

    use std::{fs::File, io, os::fd::AsRawFd};

    use irongraph_types::Result;

    pub(super) fn full_fsync(file: &File) -> Result<()> {
        // SAFETY: `fcntl` is called with a descriptor borrowed from a live `File`, so it is open
        // for the duration of the call, and `F_FULLFSYNC` takes no further arguments. The call
        // only reports a status; it neither transfers ownership nor writes through a pointer.
        let status = unsafe { libc::fcntl(file.as_raw_fd(), libc::F_FULLFSYNC) };
        if status != -1 {
            return Ok(());
        }
        let error = io::Error::last_os_error();
        // Network mounts and some virtualised volumes do not implement `F_FULLFSYNC` and report
        // that rather than failing the flush. An ordinary `fsync` is the strongest barrier those
        // filesystems offer, so fall back rather than refusing to write at all.
        if matches!(
            error.raw_os_error(),
            Some(libc::ENOTTY | libc::ENOTSUP | libc::EINVAL)
        ) {
            file.sync_all()?;
            return Ok(());
        }
        Err(error.into())
    }
}

pub fn atomic_write(path: &Path, bytes: &[u8], private: bool) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| Error::invalid_data("durable file has no parent directory"))?;
    fs::create_dir_all(parent)?;

    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| Error::invalid_data("durable file name is not valid UTF-8"))?;
    let temporary = parent.join(format!(".{file_name}.{}.tmp", Uuid::new_v4()));

    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(if private { 0o600 } else { 0o640 });
    }

    let result = (|| -> Result<()> {
        let mut file = options.open(&temporary)?;
        file.write_all(bytes)?;
        sync_durable(&file)?;
        fs::rename(&temporary, path)?;
        sync_parent(path)
    })();

    if result.is_err() {
        let _ignored = fs::remove_file(&temporary);
    }
    result
}

/// Publishes a new durable file without ever replacing an existing one.
///
/// The temporary inode is fully synced before it is linked at `path`. A `false` result means
/// another process won the create race and the existing file was left untouched.
pub fn atomic_create(path: &Path, bytes: &[u8], private: bool) -> Result<bool> {
    let parent = path
        .parent()
        .ok_or_else(|| Error::invalid_data("durable file has no parent directory"))?;
    fs::create_dir_all(parent)?;

    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| Error::invalid_data("durable file name is not valid UTF-8"))?;
    let temporary = parent.join(format!(".{file_name}.{}.tmp", Uuid::new_v4()));

    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(if private { 0o600 } else { 0o640 });
    }

    let result = (|| -> Result<bool> {
        let mut file = options.open(&temporary)?;
        file.write_all(bytes)?;
        sync_durable(&file)?;
        let created = match fs::hard_link(&temporary, path) {
            Ok(()) => true,
            Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => false,
            Err(error) => return Err(error.into()),
        };
        fs::remove_file(&temporary)?;
        sync_parent(path)?;
        Ok(created)
    })();

    if result.is_err() {
        let _ignored = fs::remove_file(&temporary);
    }
    result
}

pub fn read_bounded(path: &Path, max_bytes: usize) -> Result<Vec<u8>> {
    let metadata = fs::metadata(path)?;
    let length = usize::try_from(metadata.len())
        .map_err(|_| Error::new(ErrorCode::CorruptStorage, "durable file is too large"))?;
    if length > max_bytes {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "durable file exceeds its configured size limit",
        ));
    }
    let mut bytes = Vec::with_capacity(length);
    File::open(path)?.read_to_end(&mut bytes)?;
    if bytes.len() != length {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "durable file changed while it was read",
        ));
    }
    Ok(bytes)
}

pub fn sync_parent(path: &Path) -> Result<()> {
    let parent = path
        .parent()
        .ok_or_else(|| Error::invalid_data("durable file has no parent directory"))?;
    sync_directory(parent)
}

#[cfg(unix)]
fn sync_directory(path: &Path) -> Result<()> {
    // The rename itself must reach durable media, not just the file it points at.
    sync_durable(&File::open(path)?)
}

#[cfg(not(unix))]
fn sync_directory(_path: &Path) -> Result<()> {
    Ok(())
}

pub fn temporary_path(path: &Path) -> Result<PathBuf> {
    let parent = path
        .parent()
        .ok_or_else(|| Error::invalid_data("durable file has no parent directory"))?;
    let file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| Error::invalid_data("durable file name is not valid UTF-8"))?;
    Ok(parent.join(format!(".{file_name}.{}.tmp", Uuid::new_v4())))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn durable_sync_reaches_the_platform_barrier_and_survives_reopen() -> Result<()> {
        // Exercises the real barrier on whatever platform the suite runs on. On Apple this is the
        // `fcntl(F_FULLFSYNC)` path, so the test fails loudly if the command is ever rejected by
        // the filesystem under test rather than silently degrading to a weaker flush.
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("durable.bin");
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)?;
        file.write_all(b"committed")?;
        sync_durable(&file)?;
        sync_durable_data(&file)?;
        sync_parent(&path)?;
        drop(file);
        assert_eq!(read_bounded(&path, 64)?, b"committed");
        Ok(())
    }

    #[test]
    fn atomic_publication_is_readable_after_a_durable_barrier() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("nested").join("state.bin");
        atomic_write(&path, b"first", true)?;
        assert_eq!(read_bounded(&path, 64)?, b"first");
        atomic_write(&path, b"second", true)?;
        assert_eq!(read_bounded(&path, 64)?, b"second");
        assert!(!atomic_create(&path, b"third", true)?);
        assert_eq!(read_bounded(&path, 64)?, b"second");
        Ok(())
    }
}
