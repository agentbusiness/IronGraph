use std::{
    fs::{self, File, OpenOptions},
    io::{BufWriter, Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
};

use crc32fast::Hasher;

use irongraph_types::{Error, ErrorCode, Result};

use super::fs::{sync_durable, sync_durable_data, sync_parent, temporary_path};

const FORMAT_VERSION: u16 = 1;
const HEADER_BYTES: usize = 32;
const BATCH_WRITE_BUFFER_BYTES: usize = 1024 * 1024;

/// Whether an append must reach durable media before returning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Durability {
    Buffered,
    Sync,
}

/// A recovered framed payload and its byte offset in the file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FramedRecord {
    pub offset: u64,
    pub payload: Vec<u8>,
}

/// Result of opening and crash-recovering a framed file.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RecoveryReport {
    pub record_count: usize,
    pub truncated_tail_bytes: u64,
}

#[derive(Clone, Copy, Debug)]
struct RecordLocation {
    offset: u64,
    payload_len: u64,
}

/// Crash-safe, length-delimited append file with checksummed headers and payloads.
pub struct FramedFile {
    path: PathBuf,
    file: File,
    magic: [u8; 8],
    max_payload_bytes: usize,
    records: Vec<RecordLocation>,
    recovery: RecoveryReport,
}

impl FramedFile {
    /// Exact durable bytes occupied by one encoded record, including its checksummed frame
    /// header. Capacity governors must use this rather than payload or logical object size.
    pub(crate) fn physical_record_bytes(payload_bytes: usize) -> Result<u64> {
        u64::try_from(HEADER_BYTES)
            .map_err(|_| Error::internal("framed header length does not fit u64"))?
            .checked_add(
                u64::try_from(payload_bytes)
                    .map_err(|_| Error::invalid_data("payload length does not fit u64"))?,
            )
            .ok_or_else(|| Error::internal("framed physical record length overflow"))
    }

    /// Extends a content-address hash with the exact durable framing bytes for one payload.
    /// Immutable segment publication uses this to discover an already-published batch before
    /// opening a temporary file, avoiding a second write and durability barrier on the leader.
    pub(crate) fn update_content_hash(
        magic: [u8; 8],
        payload: &[u8],
        hasher: &mut blake3::Hasher,
    ) -> Result<u64> {
        let header = encode_header(magic, payload)?;
        hasher.update(&header);
        hasher.update(payload);
        Self::physical_record_bytes(payload.len())
    }

    pub fn open(path: impl AsRef<Path>, magic: [u8; 8], max_payload_bytes: usize) -> Result<Self> {
        if max_payload_bytes == 0 {
            return Err(Error::invalid_data("framed payload limit must be non-zero"));
        }
        let path = path.as_ref().to_path_buf();
        let parent = path
            .parent()
            .ok_or_else(|| Error::invalid_data("framed file has no parent directory"))?;
        fs::create_dir_all(parent)?;
        let existed = path.exists();
        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(&path)?;
        if !existed {
            sync_durable(&file)?;
            sync_parent(&path)?;
        }
        let (records, truncated_tail_bytes) =
            scan_and_recover(&mut file, magic, max_payload_bytes, true)?;
        Ok(Self {
            path,
            file,
            magic,
            max_payload_bytes,
            recovery: RecoveryReport {
                record_count: records.len(),
                truncated_tail_bytes,
            },
            records,
        })
    }

    /// Opens an immutable framed file without repairing a partial tail. Immutable segments are
    /// atomically published as complete files, so truncation is corruption rather than a WAL
    /// recovery boundary.
    pub(crate) fn open_strict(
        path: impl AsRef<Path>,
        magic: [u8; 8],
        max_payload_bytes: usize,
    ) -> Result<Self> {
        if max_payload_bytes == 0 {
            return Err(Error::invalid_data("framed payload limit must be non-zero"));
        }
        let path = path.as_ref().to_path_buf();
        let mut file = OpenOptions::new().read(true).write(true).open(&path)?;
        let (records, truncated_tail_bytes) =
            scan_and_recover(&mut file, magic, max_payload_bytes, false)?;
        Ok(Self {
            path,
            file,
            magic,
            max_payload_bytes,
            recovery: RecoveryReport {
                record_count: records.len(),
                truncated_tail_bytes,
            },
            records,
        })
    }

    /// Creates a private unpublished framed file. The caller must sync its contents and publish
    /// it atomically; no durability barrier is spent on the empty temporary inode.
    pub(crate) fn create_staged(
        path: impl AsRef<Path>,
        magic: [u8; 8],
        max_payload_bytes: usize,
    ) -> Result<Self> {
        if max_payload_bytes == 0 {
            return Err(Error::invalid_data("framed payload limit must be non-zero"));
        }
        let path = path.as_ref().to_path_buf();
        let parent = path
            .parent()
            .ok_or_else(|| Error::invalid_data("framed file has no parent directory"))?;
        fs::create_dir_all(parent)?;
        let file = OpenOptions::new()
            .read(true)
            .write(true)
            .create_new(true)
            .open(&path)?;
        Ok(Self {
            path,
            file,
            magic,
            max_payload_bytes,
            recovery: RecoveryReport::default(),
            records: Vec::new(),
        })
    }

    /// Reads one known immutable record extent without scanning unrelated records. The caller
    /// obtains the extent from canonical segment metadata; this method still validates the frame
    /// header, declared length, and payload CRC before returning bytes.
    pub(crate) fn read_record_at(
        path: impl AsRef<Path>,
        magic: [u8; 8],
        max_payload_bytes: usize,
        offset: u64,
        expected_physical_bytes: u64,
    ) -> Result<FramedRecord> {
        let mut file = File::open(path)?;
        let file_bytes = file.metadata()?.len();
        let header_bytes = u64::try_from(HEADER_BYTES)
            .map_err(|_| Error::internal("framed header length does not fit u64"))?;
        if expected_physical_bytes < header_bytes
            || offset
                .checked_add(expected_physical_bytes)
                .is_none_or(|end| end > file_bytes)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "framed record extent exceeds the immutable file",
            ));
        }
        file.seek(SeekFrom::Start(offset))?;
        let mut header = [0_u8; HEADER_BYTES];
        file.read_exact(&mut header)?;
        let payload_len = decode_header(&header, magic, max_payload_bytes)?;
        if header_bytes
            .checked_add(payload_len)
            .is_none_or(|bytes| bytes != expected_physical_bytes)
        {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "framed record length differs from its canonical extent",
            ));
        }
        let payload_len = usize::try_from(payload_len).map_err(|_| {
            Error::new(
                ErrorCode::CorruptStorage,
                "framed record length does not fit memory",
            )
        })?;
        let mut payload = vec![0_u8; payload_len];
        file.read_exact(&mut payload)?;
        let expected_crc = u32::from_le_bytes(copy_array::<4>(&header[20..24])?);
        if crc32(&payload) != expected_crc {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "framed payload checksum mismatch",
            ));
        }
        Ok(FramedRecord { offset, payload })
    }

    #[must_use]
    pub const fn recovery_report(&self) -> RecoveryReport {
        self.recovery
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    pub fn byte_len(&self) -> Result<u64> {
        Ok(self.file.metadata()?.len())
    }

    pub fn append(&mut self, payload: &[u8], durability: Durability) -> Result<u64> {
        if payload.len() > self.max_payload_bytes {
            return Err(Error::invalid_data(
                "framed payload exceeds configured limit",
            ));
        }
        let offset = self.file.seek(SeekFrom::End(0))?;
        let header = encode_header(self.magic, payload)?;
        if let Err(error) = self
            .file
            .write_all(&header)
            .and_then(|()| self.file.write_all(payload))
        {
            self.rollback_partial_append(offset)?;
            return Err(error.into());
        }
        if durability == Durability::Sync
            && let Err(error) = sync_durable_data(&self.file)
        {
            self.rollback_partial_append(offset)?;
            return Err(error);
        }
        self.records.push(RecordLocation {
            offset,
            payload_len: u64::try_from(payload.len())
                .map_err(|_| Error::invalid_data("payload length does not fit u64"))?,
        });
        Ok(offset)
    }

    /// Appends a prevalidated group with one durability barrier. No in-memory record becomes
    /// visible unless every framed payload is written and the requested sync succeeds.
    pub fn append_batch(
        &mut self,
        payloads: &[Vec<u8>],
        durability: Durability,
    ) -> Result<Vec<u64>> {
        if payloads.is_empty() {
            return Err(Error::invalid_data("framed append batch is empty"));
        }
        if payloads
            .iter()
            .any(|payload| payload.len() > self.max_payload_bytes)
        {
            return Err(Error::invalid_data(
                "framed payload exceeds configured limit",
            ));
        }

        let framed_bytes = payloads.iter().try_fold(0_usize, |total, payload| {
            total
                .checked_add(HEADER_BYTES)
                .and_then(|total| total.checked_add(payload.len()))
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::ResultBudgetExceeded,
                        "framed batch size overflow",
                    )
                })
        })?;

        let initial_offset = self.file.seek(SeekFrom::End(0))?;
        let mut offset = initial_offset;
        let mut offsets = Vec::with_capacity(payloads.len());
        let mut locations = Vec::with_capacity(payloads.len());
        let write = (|| -> Result<()> {
            {
                let mut writer = BufWriter::with_capacity(
                    framed_bytes.clamp(HEADER_BYTES, BATCH_WRITE_BUFFER_BYTES),
                    &mut self.file,
                );
                for payload in payloads {
                    let header = encode_header(self.magic, payload)?;
                    writer.write_all(&header)?;
                    writer.write_all(payload)?;
                    offsets.push(offset);
                    let payload_len = u64::try_from(payload.len())
                        .map_err(|_| Error::invalid_data("payload length does not fit u64"))?;
                    locations.push(RecordLocation {
                        offset,
                        payload_len,
                    });
                    offset = offset
                        .checked_add(Self::physical_record_bytes(payload.len())?)
                        .ok_or_else(|| Error::internal("framed batch offset overflow"))?;
                }
                writer.flush()?;
            }
            if durability == Durability::Sync {
                sync_durable_data(&self.file)?;
            }
            Ok(())
        })();
        if let Err(error) = write {
            self.rollback_partial_append(initial_offset)?;
            return Err(error);
        }
        self.records.extend(locations);
        Ok(offsets)
    }

    pub fn sync(&self) -> Result<()> {
        sync_durable_data(&self.file)
    }

    pub fn read_all(&mut self) -> Result<Vec<FramedRecord>> {
        let mut result = Vec::with_capacity(self.records.len());
        self.try_for_each(|record| {
            result.push(record);
            Ok(())
        })?;
        Ok(result)
    }

    /// Reads and validates one recovered payload at a time without materializing the full file.
    pub fn try_for_each(
        &mut self,
        mut visitor: impl FnMut(FramedRecord) -> Result<()>,
    ) -> Result<()> {
        for index in 0..self.records.len() {
            let location = self.records[index];
            self.file.seek(SeekFrom::Start(
                location.offset.checked_add(32).ok_or_else(|| {
                    Error::new(ErrorCode::CorruptStorage, "record offset overflow")
                })?,
            ))?;
            let length = usize::try_from(location.payload_len).map_err(|_| {
                Error::new(
                    ErrorCode::CorruptStorage,
                    "record length does not fit memory",
                )
            })?;
            let mut payload = vec![0_u8; length];
            self.file.read_exact(&mut payload)?;
            visitor(FramedRecord {
                offset: location.offset,
                payload,
            })?;
        }
        Ok(())
    }

    pub fn rewrite(&mut self, payloads: &[Vec<u8>]) -> Result<()> {
        self.rewrite_streaming(payloads.iter().map(|payload| Ok(payload.as_slice())))
    }

    /// Atomically replaces the file while holding only one encoded payload at a time.
    pub fn rewrite_streaming<I, B>(&mut self, payloads: I) -> Result<()>
    where
        I: IntoIterator<Item = Result<B>>,
        B: AsRef<[u8]>,
    {
        let temporary = temporary_path(&self.path)?;
        let result = (|| -> Result<()> {
            let mut replacement = OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&temporary)?;
            for payload in payloads {
                let payload = payload?;
                let payload = payload.as_ref();
                if payload.len() > self.max_payload_bytes {
                    return Err(Error::invalid_data(
                        "rewritten payload exceeds configured limit",
                    ));
                }
                let header = encode_header(self.magic, payload)?;
                replacement.write_all(&header)?;
                replacement.write_all(payload)?;
            }
            sync_durable(&replacement)?;
            let (records, truncated_tail_bytes) =
                scan_and_recover(&mut replacement, self.magic, self.max_payload_bytes, false)?;
            fs::rename(&temporary, &self.path)?;
            self.file = replacement;
            self.records = records;
            self.recovery = RecoveryReport {
                record_count: self.records.len(),
                truncated_tail_bytes,
            };
            sync_parent(&self.path)?;
            Ok(())
        })();
        if result.is_err() {
            let _ignored = fs::remove_file(&temporary);
        }
        result
    }

    fn rollback_partial_append(&mut self, offset: u64) -> Result<()> {
        self.file.set_len(offset)?;
        sync_durable_data(&self.file)?;
        self.file.seek(SeekFrom::Start(offset))?;
        Ok(())
    }
}

fn scan_and_recover(
    file: &mut File,
    magic: [u8; 8],
    max_payload_bytes: usize,
    recover_tail: bool,
) -> Result<(Vec<RecordLocation>, u64)> {
    let file_len = file.metadata()?.len();
    let header_len = u64::try_from(HEADER_BYTES)
        .map_err(|_| Error::internal("framed header length does not fit u64"))?;
    let mut offset = 0_u64;
    let mut records = Vec::new();
    while offset < file_len {
        let remaining = file_len.saturating_sub(offset);
        if remaining < header_len {
            return recover_partial_tail(file, offset, file_len, recover_tail, records);
        }
        file.seek(SeekFrom::Start(offset))?;
        let mut header = [0_u8; HEADER_BYTES];
        file.read_exact(&mut header)?;
        let payload_len = decode_header(&header, magic, max_payload_bytes)?;
        let end = offset
            .checked_add(header_len)
            .and_then(|value| value.checked_add(payload_len))
            .ok_or_else(|| {
                Error::new(ErrorCode::CorruptStorage, "framed record offset overflow")
            })?;
        if end > file_len {
            return recover_partial_tail(file, offset, file_len, recover_tail, records);
        }
        let length = usize::try_from(payload_len)
            .map_err(|_| Error::new(ErrorCode::CorruptStorage, "payload length is too large"))?;
        let mut payload = vec![0_u8; length];
        file.read_exact(&mut payload)?;
        let expected_crc = u32::from_le_bytes(copy_array::<4>(&header[20..24])?);
        if crc32(&payload) != expected_crc {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "framed payload checksum mismatch",
            ));
        }
        records.push(RecordLocation {
            offset,
            payload_len,
        });
        offset = end;
    }
    Ok((records, 0))
}

fn recover_partial_tail(
    file: &File,
    valid_len: u64,
    original_len: u64,
    recover_tail: bool,
    records: Vec<RecordLocation>,
) -> Result<(Vec<RecordLocation>, u64)> {
    if !recover_tail {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "framed file has a truncated tail",
        ));
    }
    file.set_len(valid_len)?;
    sync_durable_data(file)?;
    Ok((records, original_len.saturating_sub(valid_len)))
}

fn encode_header(magic: [u8; 8], payload: &[u8]) -> Result<[u8; HEADER_BYTES]> {
    let payload_len = u64::try_from(payload.len())
        .map_err(|_| Error::invalid_data("payload length does not fit u64"))?;
    let mut header = [0_u8; HEADER_BYTES];
    header[0..8].copy_from_slice(&magic);
    header[8..10].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
    header[12..20].copy_from_slice(&payload_len.to_le_bytes());
    header[20..24].copy_from_slice(&crc32(payload).to_le_bytes());
    let header_crc = crc32(&header[0..24]);
    header[24..28].copy_from_slice(&header_crc.to_le_bytes());
    Ok(header)
}

fn decode_header(header: &[u8; HEADER_BYTES], magic: [u8; 8], max: usize) -> Result<u64> {
    if header[0..8] != magic {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "framed file magic mismatch",
        ));
    }
    let version = u16::from_le_bytes(copy_array::<2>(&header[8..10])?);
    if version != FORMAT_VERSION {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "unsupported framed file version",
        ));
    }
    if header[10..12] != [0_u8; 2] || header[28..32] != [0_u8; 4] {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "framed header reserved bytes are non-zero",
        ));
    }
    let expected_header_crc = u32::from_le_bytes(copy_array::<4>(&header[24..28])?);
    if crc32(&header[0..24]) != expected_header_crc {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "framed header checksum mismatch",
        ));
    }
    let payload_len = u64::from_le_bytes(copy_array::<8>(&header[12..20])?);
    if payload_len > u64::try_from(max).map_err(|_| Error::internal("frame limit overflow"))? {
        return Err(Error::new(
            ErrorCode::CorruptStorage,
            "framed payload length exceeds configured limit",
        ));
    }
    Ok(payload_len)
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut hasher = Hasher::new();
    hasher.update(bytes);
    hasher.finalize()
}

fn copy_array<const N: usize>(bytes: &[u8]) -> Result<[u8; N]> {
    bytes
        .try_into()
        .map_err(|_| Error::new(ErrorCode::CorruptStorage, "invalid framed integer width"))
}

#[cfg(test)]
mod tests {
    use tempfile::tempdir;

    use super::*;

    #[test]
    fn batch_append_has_one_publication_boundary_and_prevalidates_limits()
    -> Result<(), Box<dyn std::error::Error>> {
        let directory = tempdir()?;
        let path = directory.path().join("batch.log");
        let mut file = FramedFile::open(&path, *b"BATCH001", 8)?;
        let payloads = vec![b"one".to_vec(), b"two".to_vec()];
        let offsets = file.append_batch(&payloads, Durability::Sync)?;
        assert_eq!(offsets, vec![0, 35]);
        assert_eq!(file.len(), 2);
        let durable_len = file.byte_len()?;

        let invalid = vec![b"ok".to_vec(), vec![0_u8; 9]];
        assert!(file.append_batch(&invalid, Durability::Sync).is_err());
        assert_eq!(file.len(), 2);
        assert_eq!(file.byte_len()?, durable_len);
        drop(file);

        let mut reopened = FramedFile::open(&path, *b"BATCH001", 8)?;
        assert_eq!(
            reopened
                .read_all()?
                .into_iter()
                .map(|record| record.payload)
                .collect::<Vec<_>>(),
            payloads
        );
        Ok(())
    }
}
