//! Descriptor-anchored, bounded loading of immutable encoder artifacts.

use std::{
    collections::BTreeMap,
    fs::File,
    io::{Read, Seek, SeekFrom},
    ops::Range,
    path::Path,
};

use candle_core::{DType, Device, Error, Result, Shape, Tensor, safetensors::Load};
use candle_nn::{VarBuilder, var_builder::SimpleBackend};
use memmap2::Mmap;
use safetensors::{Dtype as SafeDType, SafeTensors, tensor::TensorView};
use sha2::{Digest as _, Sha256};

/// Opens one regular artifact without following any path component and requires its exact size.
///
/// The returned descriptor is the one the caller must parse; no pathname re-open is necessary
/// after validation.
///
/// # Errors
///
/// Returns an error when the path cannot be opened safely or the file is not a regular artifact
/// with exactly `exact_bytes` bytes.
pub fn open_regular_exact(path: &Path, exact_bytes: u64) -> Result<File> {
    let file = open_regular(path)?;
    let length = bounded_length(&file, path, exact_bytes, false)?;
    if length != exact_bytes {
        return Err(artifact_error(
            path,
            "artifact length differs from its pinned size",
        ));
    }
    Ok(file)
}

/// Computes SHA-256 from a no-follow descriptor while enforcing one exact artifact size.
///
/// # Errors
///
/// Returns an error when the artifact cannot be opened, read, or remains inconsistent with the
/// requested exact size while it is hashed.
pub fn sha256_exact(path: &Path, exact_bytes: u64) -> Result<[u8; 32]> {
    let mut file = open_regular_exact(path, exact_bytes)?;
    sha256_descriptor(&mut file, path, exact_bytes)
}

/// Opens and authenticates a pinned SHA-256 artifact through one no-follow descriptor, then
/// rewinds that same descriptor for the caller's parser.
///
/// # Errors
///
/// Returns an error when safe opening, exact-size verification, hashing, digest comparison, or
/// rewinding fails.
/// Open an artifact, hashing it only the first time it is seen.
///
/// Hashing several gigabytes is not free, and doing it on every
/// start pays that price to re-learn something already established. The check that matters happens
/// once, when the bytes arrive from the network; after that the file has not left the disk.
///
/// A sidecar records which digest was verified. Both the size and that record must match before the
/// hash is skipped, so a file replaced by one of a different length, or a sidecar that names a
/// different digest, still forces a full verification. Anything unexpected — no sidecar, wrong
/// contents, unreadable — falls through to hashing, which is the safe direction to fail in.
///
/// This is not defence against an attacker who can write to the artifact directory: they could write
/// the sidecar too. It defends against corruption and against the wrong file being served, which is
/// what the original check was for, and it is the same guarantee the download already gives.
/// Map an artifact into memory, hashing it only the first time it is seen.
///
/// The same guarantee as [`open_verified_once`], handed back as bytes rather than a descriptor.
/// Mapping avoids copying a large tensor artifact into a fresh buffer before parsing. The kernel
/// pages mapped bytes in on demand and keeps them cached across repeated starts.
///
/// # Errors
///
/// Returns an error when the artifact cannot be opened, is the wrong size, fails verification, or
/// cannot be mapped.
/// Map a file this process owns, read-only, without verifying it.
///
/// For caches and other data we wrote ourselves, where there is no pinned digest to check because
/// the file is not an artifact anyone published. The unsafety is the same as every other mapping —
/// truncation underneath us — and lives behind the same boundary rather than being re-argued at the
/// call site.
///
/// # Errors
///
/// Returns an error when the file cannot be opened or mapped.
pub fn map_owned_file(path: &Path) -> Result<Mmap> {
    // Opened plainly, not with `O_NOFOLLOW`.
    //
    // That hardening exists for artifacts fetched from elsewhere, where a symlinked component would
    // be an attack. This is a file this process wrote, at a path this process built, and refusing
    // to follow a symlink here fails on ordinary setups — a home directory behind a link, or a
    // temporary directory under macOS's symlinked `/var` — by silently declining to use the cache.
    let file =
        File::open(path).map_err(|error| artifact_error(path, &format!("cannot open: {error}")))?;
    ffi::map_readonly(&file).map_err(|error| artifact_error(path, &format!("cannot map: {error}")))
}

/// Opens, verifies once, and maps a pinned artifact read-only.
///
/// # Errors
///
/// Returns an error when the artifact cannot be opened, verified, or mapped.
pub fn map_verified_once(path: &Path, exact_bytes: u64, expected_sha256: [u8; 32]) -> Result<Mmap> {
    let file = open_verified_once(path, exact_bytes, expected_sha256)?;
    ffi::map_readonly(&file).map_err(|error| artifact_error(path, &format!("cannot map: {error}")))
}

/// Opens a pinned artifact, using its verified marker when available.
///
/// # Errors
///
/// Returns an error when the artifact cannot be opened or fails size or digest verification.
pub fn open_verified_once(
    path: &Path,
    exact_bytes: u64,
    expected_sha256: [u8; 32],
) -> Result<File> {
    let marker = path.with_extension("verified");
    let recorded = std::fs::read_to_string(&marker).ok();
    if recorded.as_deref().map(str::trim) == Some(&hex(expected_sha256)) {
        // Size is checked by the open itself, so a truncated or swollen file never gets here.
        return open_regular_exact(path, exact_bytes);
    }
    let file = open_verified_sha256_exact(path, exact_bytes, expected_sha256)?;
    // Best effort: a marker that cannot be written costs the next start ten seconds, nothing more.
    let _ = std::fs::write(&marker, hex(expected_sha256));
    Ok(file)
}

fn hex(digest: [u8; 32]) -> String {
    use std::fmt::Write;
    digest
        .iter()
        .fold(String::with_capacity(64), |mut out, byte| {
            let _ = write!(out, "{byte:02x}");
            out
        })
}

/// Opens a regular file and verifies its exact length and SHA-256 digest.
///
/// # Errors
///
/// Returns an error when the file cannot be opened or its length or digest differs.
pub fn open_verified_sha256_exact(
    path: &Path,
    exact_bytes: u64,
    expected_sha256: [u8; 32],
) -> Result<File> {
    let mut file = open_regular_exact(path, exact_bytes)?;
    let actual = sha256_descriptor(&mut file, path, exact_bytes)?;
    if actual != expected_sha256 {
        return Err(artifact_error(path, "artifact SHA-256 mismatch"));
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|error| Error::from(error).with_path(path))?;
    Ok(file)
}

fn sha256_descriptor(file: &mut File, path: &Path, exact_bytes: u64) -> Result<[u8; 32]> {
    let mut hasher = Sha256::new();
    let mut consumed = 0_u64;
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| Error::from(error).with_path(path))?;
        if read == 0 {
            break;
        }
        consumed = consumed
            .checked_add(
                u64::try_from(read)
                    .map_err(|_| artifact_error(path, "artifact read length exceeds u64"))?,
            )
            .ok_or_else(|| artifact_error(path, "artifact length overflows"))?;
        if consumed > exact_bytes {
            return Err(artifact_error(path, "artifact exceeds its pinned size"));
        }
        hasher.update(&buffer[..read]);
    }
    if consumed != exact_bytes
        || file
            .metadata()
            .map_err(|error| Error::from(error).with_path(path))?
            .len()
            != exact_bytes
    {
        return Err(artifact_error(
            path,
            "artifact changed length while it was hashed",
        ));
    }
    Ok(hasher.finalize().into())
}

/// Reads one regular file through a descriptor opened without following any symbolic link.
///
/// The size is checked before allocation and again after the descriptor is consumed. The path is
/// never reopened between inspection and reading.
///
/// # Errors
///
/// Returns an error when safe opening or reading fails, the size exceeds `maximum_bytes`, or the
/// descriptor changes length during the read.
pub fn read_bounded(path: &Path, maximum_bytes: usize) -> Result<Vec<u8>> {
    let mut file = open_regular(path)?;
    let maximum_u64 = u64::try_from(maximum_bytes)
        .map_err(|_| artifact_error(path, "artifact byte limit exceeds u64"))?;
    let length = bounded_length(&file, path, maximum_u64, false)?;
    let capacity = usize::try_from(length).map_err(|_| {
        artifact_error(
            path,
            "artifact length exceeds this platform's address space",
        )
    })?;
    let mut bytes = Vec::with_capacity(capacity);
    let read_limit = maximum_bytes
        .checked_add(1)
        .ok_or_else(|| artifact_error(path, "artifact byte limit overflows"))?;
    let read_limit = u64::try_from(read_limit)
        .map_err(|_| artifact_error(path, "artifact read limit exceeds u64"))?;
    file.by_ref()
        .take(read_limit)
        .read_to_end(&mut bytes)
        .map_err(|error| Error::from(error).with_path(path))?;
    if bytes.len() != capacity {
        return Err(artifact_error(
            path,
            "artifact changed length while it was being read",
        ));
    }
    let final_length = file
        .metadata()
        .map_err(|error| Error::from(error).with_path(path))?
        .len();
    if final_length != length {
        return Err(artifact_error(
            path,
            "artifact changed length while it was being read",
        ));
    }
    Ok(bytes)
}

/// Reads and authenticates a bounded artifact through one descriptor.
///
/// # Errors
///
/// Returns an error when bounded reading fails or the artifact digest differs from
/// `expected_hash`.
pub fn read_verified_bounded(
    path: &Path,
    expected_hash: [u8; 32],
    maximum_bytes: usize,
) -> Result<Vec<u8>> {
    let bytes = read_bounded(path, maximum_bytes)?;
    if blake3::hash(&bytes).as_bytes() != &expected_hash {
        return Err(artifact_error(path, "artifact digest mismatch"));
    }
    Ok(bytes)
}

/// Hashes one bounded regular artifact without following symbolic links or reopening the path.
///
/// # Errors
///
/// Returns an error when safe opening or reading fails, the artifact exceeds `maximum_bytes`, or
/// the descriptor changes length while it is hashed.
pub fn hash_bounded(path: &Path, maximum_bytes: u64) -> Result<[u8; 32]> {
    let mut file = open_regular(path)?;
    let length = bounded_length(&file, path, maximum_bytes, false)?;
    let mut hasher = blake3::Hasher::new();
    let mut consumed = 0_u64;
    let mut buffer = vec![0_u8; 1024 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| Error::from(error).with_path(path))?;
        if read == 0 {
            break;
        }
        consumed = consumed
            .checked_add(
                u64::try_from(read)
                    .map_err(|_| artifact_error(path, "artifact read length exceeds u64"))?,
            )
            .ok_or_else(|| artifact_error(path, "artifact length overflows"))?;
        if consumed > maximum_bytes {
            return Err(artifact_error(path, "artifact exceeds its byte limit"));
        }
        hasher.update(&buffer[..read]);
    }
    if consumed != length
        || file
            .metadata()
            .map_err(|error| Error::from(error).with_path(path))?
            .len()
            != length
    {
        return Err(artifact_error(
            path,
            "artifact changed length while it was being hashed",
        ));
    }
    Ok(*hasher.finalize().as_bytes())
}

/// Builds a lazy safetensors source from the exact descriptor that was size-checked and hashed.
///
/// The artifact must be a non-writable regular file. The mapping and descriptor remain owned by
/// the returned backend, eliminating path replacement between verification and model loading.
/// Filesystem write bits are an activation-lifecycle guard, not a hard seal against an
/// administrator or another process controlling the same local account; those actors remain
/// inside the trusted local artifact boundary.
///
/// # Errors
///
/// Returns an error when the artifact is unsafe, mutable, oversized, unreadable, has the wrong
/// digest, or cannot be mapped and parsed as safetensors on `device`.
pub fn mmap_verified_safetensors(
    path: &Path,
    expected_hash: [u8; 32],
    maximum_bytes: u64,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    let mut file = open_regular(path)?;
    let length = bounded_length(&file, path, maximum_bytes, true)?;
    if length == 0 {
        return Err(artifact_error(path, "safetensors artifact is empty"));
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|error| Error::from(error).with_path(path))?;
    let mapping = ffi::map(&file, path)?;
    if u64::try_from(mapping.len())
        .map_err(|_| artifact_error(path, "mapped artifact length exceeds u64"))?
        != length
    {
        return Err(artifact_error(
            path,
            "artifact changed length while it was being mapped",
        ));
    }
    if blake3::hash(&mapping).as_bytes() != &expected_hash {
        return Err(artifact_error(path, "artifact digest mismatch"));
    }
    verified_safetensors_builder(mapping, path, dtype, device)
}

/// Builds a lazy safetensors source from one exact-size, SHA-256-pinned descriptor.
///
/// This variant is for externally published artifacts whose immutable identity is specified as
/// SHA-256. It accepts the private `0600` installation mode used by the model installer, hashes
/// and maps the same no-follow descriptor, and never reopens the pathname between authentication
/// and parsing.
///
/// # Errors
///
/// Returns an error when safe opening, exact-size or SHA-256 verification, mapping, or
/// safetensors parsing fails.
pub fn mmap_verified_safetensors_sha256_exact(
    path: &Path,
    exact_bytes: u64,
    expected_sha256: [u8; 32],
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    // Hashed the first time it is seen, and not again — the same contract as
    // [`open_verified_once`]. This is the retrieval encoder, and re-reading its weights to
    // re-establish a digest proven at download cost seconds on every single start.
    let mut file = open_regular_exact(path, exact_bytes)?;
    let marker = path.with_extension("verified");
    let recorded = std::fs::read_to_string(&marker).ok();
    if recorded.as_deref().map(str::trim) != Some(&hex(expected_sha256)) {
        let actual = sha256_descriptor(&mut file, path, exact_bytes)?;
        if actual != expected_sha256 {
            return Err(artifact_error(path, "artifact SHA-256 mismatch"));
        }
        let _ = std::fs::write(&marker, hex(expected_sha256));
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|error| Error::from(error).with_path(path))?;
    let mapping = ffi::map(&file, path)?;
    if u64::try_from(mapping.len())
        .map_err(|_| artifact_error(path, "mapped artifact length exceeds u64"))?
        != exact_bytes
    {
        return Err(artifact_error(
            path,
            "artifact changed length while it was being mapped",
        ));
    }
    verified_safetensors_builder(mapping, path, dtype, device)
}

fn verified_safetensors_builder(
    mapping: Mmap,
    path: &Path,
    dtype: DType,
    device: &Device,
) -> Result<VarBuilder<'static>> {
    let parsed =
        SafeTensors::deserialize(&mapping).map_err(|error| Error::from(error).with_path(path))?;
    let mut index = BTreeMap::new();
    let base = mapping.as_ptr() as usize;
    for (name, view) in parsed.tensors() {
        let start = (view.data().as_ptr() as usize)
            .checked_sub(base)
            .ok_or_else(|| artifact_error(path, "tensor data precedes its mapping"))?;
        let end = start
            .checked_add(view.data().len())
            .filter(|end| *end <= mapping.len())
            .ok_or_else(|| artifact_error(path, "tensor data exceeds its mapping"))?;
        if index
            .insert(
                name,
                TensorIndex {
                    dtype: view.dtype(),
                    shape: view.shape().to_vec(),
                    bytes: start..end,
                },
            )
            .is_some()
        {
            return Err(artifact_error(path, "safetensors field name is duplicated"));
        }
    }
    drop(parsed);
    let backend = VerifiedSafetensors { mapping, index };
    Ok(VarBuilder::from_backend(
        Box::new(backend),
        dtype,
        device.clone(),
    ))
}

struct VerifiedSafetensors {
    mapping: Mmap,
    index: BTreeMap<String, TensorIndex>,
}

struct TensorIndex {
    dtype: SafeDType,
    shape: Vec<usize>,
    bytes: Range<usize>,
}

impl VerifiedSafetensors {
    fn load(&self, name: &str, device: &Device) -> Result<Tensor> {
        let index = self
            .index
            .get(name)
            .ok_or_else(|| Error::Msg(format!("safetensors field is absent: {name}")))?;
        let bytes = self
            .mapping
            .get(index.bytes.clone())
            .ok_or_else(|| Error::Msg(format!("safetensors field range is invalid: {name}")))?;
        TensorView::new(index.dtype, index.shape.clone(), bytes)?.load(device)
    }
}

impl SimpleBackend for VerifiedSafetensors {
    fn get(
        &self,
        expected_shape: Shape,
        name: &str,
        _: candle_nn::Init,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let tensor = self.load(name, device)?.to_dtype(dtype)?;
        if tensor.shape() != &expected_shape {
            return Err(Error::Msg(format!(
                "shape mismatch for safetensors field {name}: expected {expected_shape:?}, got {:?}",
                tensor.shape()
            )));
        }
        Ok(tensor)
    }

    fn get_unchecked(&self, name: &str, dtype: DType, device: &Device) -> Result<Tensor> {
        self.load(name, device)?.to_dtype(dtype)
    }

    fn contains_tensor(&self, name: &str) -> bool {
        self.index.contains_key(name)
    }
}

fn bounded_length(file: &File, path: &Path, maximum_bytes: u64, immutable: bool) -> Result<u64> {
    let metadata = file
        .metadata()
        .map_err(|error| Error::from(error).with_path(path))?;
    if !metadata.is_file() {
        return Err(artifact_error(path, "artifact is not a regular file"));
    }
    if metadata.len() > maximum_bytes {
        return Err(artifact_error(path, "artifact exceeds its byte limit"));
    }
    if immutable && !ffi::is_immutable_mode(&metadata) {
        return Err(artifact_error(
            path,
            "mapped artifact must have every filesystem write bit removed",
        ));
    }
    Ok(metadata.len())
}

fn open_regular(path: &Path) -> Result<File> {
    ffi::open_nofollow(path).map_err(|error| Error::from(error).with_path(path))
}

fn artifact_error(path: &Path, message: &str) -> Error {
    Error::Msg(format!("{message}: {}", path.display()))
}

/// File-descriptor and mapping safety boundary. Every component is opened relative to an already
/// opened directory with `O_NOFOLLOW`; no checked path component is later reopened by name.
mod ffi {
    #![allow(unsafe_code)]

    use std::{
        ffi::CString,
        fs::{File, Metadata},
        io,
        path::{Component, Path},
    };

    use memmap2::{Mmap, MmapOptions};

    /// Map a whole file read-only.
    ///
    /// Unsafe because the mapping is invalidated if the file is truncated underneath it. These are
    /// our own artifacts, in a directory we own, already opened and checked — the same assumption
    /// the read path was making, expressed once here instead of at every call site.
    pub fn map_readonly(file: &File) -> io::Result<Mmap> {
        unsafe { MmapOptions::new().map(file) }
    }

    #[cfg(unix)]
    use std::os::{
        fd::{AsRawFd, FromRawFd},
        unix::{ffi::OsStrExt, fs::MetadataExt},
    };

    #[cfg(unix)]
    pub fn open_nofollow(path: &Path) -> io::Result<File> {
        let absolute = path.is_absolute();
        let mut names = Vec::new();
        for component in path.components() {
            match component {
                Component::RootDir | Component::CurDir => {}
                Component::Normal(name) => names.push(name),
                Component::ParentDir | Component::Prefix(_) => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "artifact path contains a forbidden component",
                    ));
                }
            }
        }
        let (file_name, directories) = names.split_last().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "artifact path has no file name",
            )
        })?;
        let start = if absolute { "/" } else { "." };
        let mut directory = open_component(
            libc::AT_FDCWD,
            CString::new(start).map_err(invalid_name)?.as_c_str(),
            libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        )?;
        for name in directories {
            let name = CString::new(name.as_bytes()).map_err(invalid_name)?;
            directory = open_component(
                directory.as_raw_fd(),
                &name,
                libc::O_RDONLY | libc::O_DIRECTORY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
            )?;
        }
        let file_name = CString::new(file_name.as_bytes()).map_err(invalid_name)?;
        open_component(
            directory.as_raw_fd(),
            &file_name,
            libc::O_RDONLY | libc::O_CLOEXEC | libc::O_NOFOLLOW,
        )
    }

    #[cfg(not(unix))]
    pub fn open_nofollow(_: &Path) -> io::Result<File> {
        Err(io::Error::new(
            io::ErrorKind::Unsupported,
            "secure artifact traversal is unavailable on this platform",
        ))
    }

    #[cfg(unix)]
    fn open_component(directory: i32, name: &std::ffi::CStr, flags: i32) -> io::Result<File> {
        // SAFETY: `name` is a valid NUL-terminated C string, flags do not request creation, and a
        // successful descriptor is immediately transferred into sole `File` ownership.
        let descriptor = unsafe { libc::openat(directory, name.as_ptr(), flags) };
        if descriptor < 0 {
            Err(io::Error::last_os_error())
        } else {
            // SAFETY: `descriptor` is newly returned by `openat` and has no other owner.
            Ok(unsafe { File::from_raw_fd(descriptor) })
        }
    }

    #[cfg(unix)]
    fn invalid_name(error: std::ffi::NulError) -> io::Error {
        io::Error::new(io::ErrorKind::InvalidInput, error)
    }

    pub fn map(file: &File, path: &Path) -> candle_core::Result<Mmap> {
        // SAFETY: the caller has opened a non-writable regular file by descriptor, keeps that
        // descriptor alive through mapping, and the returned `Mmap` owns the mapped lifetime.
        unsafe { MmapOptions::new().map(file) }
            .map_err(|error| candle_core::Error::from(error).with_path(path))
    }

    #[cfg(unix)]
    pub fn is_immutable_mode(metadata: &Metadata) -> bool {
        metadata.mode() & 0o222 == 0
    }

    #[cfg(not(unix))]
    pub fn is_immutable_mode(_: &Metadata) -> bool {
        false
    }
}

#[cfg(test)]
mod tests {
    use std::{fs, io::Read as _, path::Path};

    use candle_core::{DType, Device};
    use safetensors::{Dtype as SafeDType, tensor::TensorView};
    use sha2::{Digest as _, Sha256};
    use tempfile::tempdir;

    use super::{
        hash_bounded, mmap_verified_safetensors, mmap_verified_safetensors_sha256_exact,
        open_verified_sha256_exact, read_bounded, read_verified_bounded, sha256_exact,
    };

    #[test]
    fn bounded_reads_and_hashes_use_the_opened_regular_file() -> candle_core::Result<()> {
        let directory = tempdir().map_err(candle_core::Error::from)?;
        let root = fs::canonicalize(directory.path()).map_err(candle_core::Error::from)?;
        let path = root.join("artifact.json");
        fs::write(&path, b"{\"format\":1}").map_err(candle_core::Error::from)?;
        let expected = *blake3::hash(b"{\"format\":1}").as_bytes();
        assert_eq!(read_bounded(&path, 64)?, b"{\"format\":1}");
        assert_eq!(
            read_verified_bounded(&path, expected, 64)?,
            b"{\"format\":1}"
        );
        assert_eq!(hash_bounded(&path, 64)?, expected);
        assert!(read_bounded(&path, 4).is_err());
        Ok(())
    }

    #[test]
    fn exact_sha256_verification_returns_the_authenticated_descriptor() -> candle_core::Result<()> {
        let directory = tempdir().map_err(candle_core::Error::from)?;
        let root = fs::canonicalize(directory.path()).map_err(candle_core::Error::from)?;
        let path = root.join("encoder.bin");
        let bytes = b"pinned encoder bytes";
        fs::write(&path, bytes).map_err(candle_core::Error::from)?;
        let expected: [u8; 32] = Sha256::digest(bytes).into();
        assert_eq!(sha256_exact(&path, bytes.len() as u64)?, expected);
        let mut descriptor = open_verified_sha256_exact(&path, bytes.len() as u64, expected)?;
        let mut authenticated = Vec::new();
        descriptor
            .read_to_end(&mut authenticated)
            .map_err(candle_core::Error::from)?;
        assert_eq!(authenticated, bytes);
        assert!(open_verified_sha256_exact(&path, bytes.len() as u64, [0; 32]).is_err());
        assert!(sha256_exact(&path, bytes.len() as u64 + 1).is_err());
        Ok(())
    }

    #[test]
    fn exact_sha256_safetensors_accepts_private_installer_mode() -> candle_core::Result<()> {
        let directory = tempdir().map_err(candle_core::Error::from)?;
        let root = fs::canonicalize(directory.path()).map_err(candle_core::Error::from)?;
        let path = root.join("weights.safetensors");
        let bytes = scalar_safetensors(11.0)?;
        fs::write(&path, &bytes).map_err(candle_core::Error::from)?;
        let expected: [u8; 32] = Sha256::digest(&bytes).into();
        let builder = mmap_verified_safetensors_sha256_exact(
            &path,
            bytes.len() as u64,
            expected,
            DType::F32,
            &Device::Cpu,
        )?;
        assert_eq!(builder.get(1, "value")?.to_vec1::<f32>()?, vec![11.0]);
        assert!(
            mmap_verified_safetensors_sha256_exact(
                &path,
                bytes.len() as u64,
                [0; 32],
                DType::F32,
                &Device::Cpu,
            )
            .is_err()
        );
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn every_symbolic_link_component_is_rejected() -> candle_core::Result<()> {
        use std::os::unix::fs::symlink;

        let directory = tempdir().map_err(candle_core::Error::from)?;
        let root = fs::canonicalize(directory.path()).map_err(candle_core::Error::from)?;
        let real = root.join("real");
        fs::create_dir(&real).map_err(candle_core::Error::from)?;
        fs::write(real.join("artifact"), b"sealed").map_err(candle_core::Error::from)?;
        let linked = root.join("linked");
        symlink(Path::new("real"), &linked).map_err(candle_core::Error::from)?;
        assert!(read_bounded(&linked.join("artifact"), 64).is_err());
        assert!(read_bounded(&real.join("../real/artifact"), 64).is_err());
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn mapped_descriptor_survives_atomic_path_replacement() -> candle_core::Result<()> {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempdir().map_err(candle_core::Error::from)?;
        let root = fs::canonicalize(directory.path()).map_err(candle_core::Error::from)?;
        let path = root.join("weights.safetensors");
        let original = scalar_safetensors(7.0)?;
        let replacement = scalar_safetensors(99.0)?;
        fs::write(&path, &original).map_err(candle_core::Error::from)?;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o444))
            .map_err(candle_core::Error::from)?;
        let builder = mmap_verified_safetensors(
            &path,
            *blake3::hash(&original).as_bytes(),
            1024 * 1024,
            DType::F32,
            &Device::Cpu,
        )?;

        fs::rename(&path, root.join("old.safetensors")).map_err(candle_core::Error::from)?;
        fs::write(&path, replacement).map_err(candle_core::Error::from)?;
        let value = builder.get(1, "value")?.to_vec1::<f32>()?;
        assert_eq!(value, vec![7.0]);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn truncated_or_writable_mappings_fail_closed() -> candle_core::Result<()> {
        use std::os::unix::fs::PermissionsExt;

        let directory = tempdir().map_err(candle_core::Error::from)?;
        let root = fs::canonicalize(directory.path()).map_err(candle_core::Error::from)?;
        let path = root.join("weights.safetensors");
        let complete = scalar_safetensors(3.0)?;
        fs::write(&path, &complete).map_err(candle_core::Error::from)?;
        assert!(
            mmap_verified_safetensors(
                &path,
                *blake3::hash(&complete).as_bytes(),
                1024 * 1024,
                DType::F32,
                &Device::Cpu,
            )
            .is_err()
        );
        let truncated_length = complete.len().saturating_sub(1);
        fs::write(&path, &complete[..truncated_length]).map_err(candle_core::Error::from)?;
        fs::set_permissions(&path, fs::Permissions::from_mode(0o444))
            .map_err(candle_core::Error::from)?;
        assert!(
            mmap_verified_safetensors(
                &path,
                *blake3::hash(&complete).as_bytes(),
                1024 * 1024,
                DType::F32,
                &Device::Cpu,
            )
            .is_err()
        );
        Ok(())
    }

    fn scalar_safetensors(value: f32) -> candle_core::Result<Vec<u8>> {
        let bytes = value.to_le_bytes();
        let view = TensorView::new(SafeDType::F32, vec![1], &bytes)
            .map_err(|error| candle_core::Error::Msg(error.to_string()))?;
        safetensors::tensor::serialize([("value", view)], None)
            .map_err(|error| candle_core::Error::Msg(error.to_string()))
    }
}
