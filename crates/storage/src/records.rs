use std::{ops::Bound, path::Path, sync::Arc};

use irongraph_types::{Bookmark, Error, ErrorCode, Result};
use redb::{Database, ReadableDatabase, ReadableTable, TableDefinition, TableHandle};
use uuid::Uuid;

const METADATA: TableDefinition<&[u8], &[u8]> = TableDefinition::new("metadata");
const RECORDS: TableDefinition<&[u8], &[u8]> = TableDefinition::new("records");
const CHUNKS: TableDefinition<(&[u8], u64), &[u8]> = TableDefinition::new("chunks");
const FORMAT: &[u8] = b"IronGraph-records-1";
const CHUNK_BYTES: usize = 64 * 1024;
const HEADER_BYTES: usize = 48;
const DELETE_BATCH: usize = 128;
const CACHE_BYTES: usize = 16 * 1024 * 1024;

/// One incremental change to the current durable state. Keys are opaque application identities.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RecordOp {
    Put { key: Vec<u8>, value: Vec<u8> },
    Delete { key: Vec<u8> },
    DeletePrefix { prefix: Vec<u8> },
}

/// Net changes across an inclusive canonical publication range. No mutation history is retained.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct RecordBatch {
    pub first_index: u64,
    pub bookmark: Bookmark,
    pub operations: Vec<RecordOp>,
}

/// Synchronous storage used exclusively by blocking startup and dedicated persistence workers.
/// All application records, large-value chunks, identity, and the committed bookmark occupy the
/// same database file and participate in the same atomic transaction.
#[derive(Clone)]
pub struct RecordStore {
    database: Arc<Database>,
}

pub(crate) struct RecordReader {
    transaction: redb::ReadTransaction,
}

impl std::fmt::Debug for RecordReader {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RecordReader")
            .finish_non_exhaustive()
    }
}

impl RecordReader {
    pub(crate) fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        let records = self
            .transaction
            .open_table(RECORDS)
            .map_err(storage_error)?;
        let chunks = self.transaction.open_table(CHUNKS).map_err(storage_error)?;
        records
            .get(key)
            .map_err(storage_error)?
            .map(|value| read_record(&chunks, key, value.value()))
            .transpose()
    }

    pub(crate) fn scan_prefix(
        &self,
        prefix: &[u8],
        mut visit: impl FnMut(&[u8], &[u8]) -> Result<()>,
    ) -> Result<()> {
        let records = self
            .transaction
            .open_table(RECORDS)
            .map_err(storage_error)?;
        let chunks = self.transaction.open_table(CHUNKS).map_err(storage_error)?;
        for entry in records.range(prefix..).map_err(storage_error)? {
            let (key, value) = entry.map_err(storage_error)?;
            if !key.value().starts_with(prefix) {
                break;
            }
            let decoded = read_record(&chunks, key.value(), value.value())?;
            visit(key.value(), &decoded)?;
        }
        Ok(())
    }
}

impl std::fmt::Debug for RecordStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RecordStore")
            .finish_non_exhaustive()
    }
}

impl RecordStore {
    pub fn open(path: impl AsRef<Path>, store_id: Uuid) -> Result<Self> {
        let absolute_path = std::path::absolute(path.as_ref())?;
        let path = absolute_path.as_path();
        if let Some(parent) = path.parent().filter(|p| !p.as_os_str().is_empty()) {
            std::fs::create_dir_all(parent)?;
        }
        let database = Database::builder()
            .set_cache_size(CACHE_BYTES)
            .create(path)
            .map_err(storage_error)?;
        let mut transaction = database.begin_write().map_err(storage_error)?;
        transaction.set_quick_repair(true);
        let existing_tables = transaction
            .list_tables()
            .map_err(storage_error)?
            .map(|table| table.name().to_owned())
            .collect::<std::collections::BTreeSet<_>>();
        {
            let mut metadata = transaction.open_table(METADATA).map_err(storage_error)?;
            let existing_format = metadata
                .get(b"format".as_slice())
                .map_err(storage_error)?
                .map(|v| v.value().to_vec());
            match existing_format {
                Some(format) if format == FORMAT => {
                    if !["metadata", "records", "chunks"]
                        .iter()
                        .all(|name| existing_tables.contains(*name))
                    {
                        return Err(corrupt("record store table is absent"));
                    }
                    let identity = metadata
                        .get(b"identity".as_slice())
                        .map_err(storage_error)?;
                    if identity.as_ref().map(|v| v.value()) != Some(store_id.as_bytes().as_slice())
                    {
                        return Err(Error::new(
                            ErrorCode::CorruptStorage,
                            "record store identity mismatch",
                        ));
                    }
                    let bookmark = metadata
                        .get(b"bookmark".as_slice())
                        .map_err(storage_error)?;
                    decode_bookmark(
                        bookmark
                            .as_ref()
                            .ok_or_else(|| corrupt("missing durable bookmark"))?
                            .value(),
                    )?;
                }
                Some(_) => return Err(corrupt("unsupported record store format")),
                None => {
                    if !existing_tables.is_empty() {
                        return Err(corrupt("record store format is absent"));
                    }
                    metadata
                        .insert(b"format".as_slice(), FORMAT)
                        .map_err(storage_error)?;
                    metadata
                        .insert(b"identity".as_slice(), store_id.as_bytes().as_slice())
                        .map_err(storage_error)?;
                    metadata
                        .insert(
                            b"bookmark".as_slice(),
                            encode_bookmark(Bookmark::default()).as_slice(),
                        )
                        .map_err(storage_error)?;
                }
            }
        }
        transaction.open_table(RECORDS).map_err(storage_error)?;
        transaction.open_table(CHUNKS).map_err(storage_error)?;
        transaction.commit().map_err(storage_error)?;
        // File-content durability alone does not publish a newly created directory entry.
        super::fs::sync_parent(path)?;
        Ok(Self {
            database: Arc::new(database),
        })
    }

    pub fn bookmark(&self) -> Result<Bookmark> {
        let transaction = self.database.begin_read().map_err(storage_error)?;
        let metadata = transaction.open_table(METADATA).map_err(storage_error)?;
        let value = metadata
            .get(b"bookmark".as_slice())
            .map_err(storage_error)?
            .ok_or_else(|| corrupt("missing durable bookmark"))?;
        decode_bookmark(value.value())
    }

    /// Commits the next contiguous publication prefix. Either every operation and its bookmark
    /// becomes durable, or the previous committed state remains authoritative after recovery.
    pub fn commit(&self, batches: &[RecordBatch]) -> Result<()> {
        if batches.is_empty() {
            return Ok(());
        }
        let mut transaction = self.database.begin_write().map_err(storage_error)?;
        transaction.set_quick_repair(true);
        {
            let mut metadata = transaction.open_table(METADATA).map_err(storage_error)?;
            let mut bookmark = {
                let value = metadata
                    .get(b"bookmark".as_slice())
                    .map_err(storage_error)?
                    .ok_or_else(|| corrupt("missing durable bookmark"))?;
                decode_bookmark(value.value())?
            };
            let mut records = transaction.open_table(RECORDS).map_err(storage_error)?;
            let mut chunks = transaction.open_table(CHUNKS).map_err(storage_error)?;
            for batch in batches {
                if batch.bookmark.term == 0
                    || batch.first_index
                        != bookmark
                            .index
                            .checked_add(1)
                            .ok_or_else(|| corrupt("durable position overflow"))?
                    || batch.bookmark.term < bookmark.term
                    || batch.bookmark.index < batch.first_index
                {
                    return Err(corrupt("record batch does not continue the durable prefix"));
                }
                for operation in &batch.operations {
                    match operation {
                        RecordOp::Put { key, value } => {
                            delete_record(&mut records, &mut chunks, key)?;
                            let chunk_count = if value.len() <= CHUNK_BYTES {
                                0
                            } else {
                                value.len().div_ceil(CHUNK_BYTES)
                            };
                            let length = u64::try_from(value.len())
                                .map_err(|_| Error::invalid_data("record length overflow"))?;
                            let count = u64::try_from(chunk_count)
                                .map_err(|_| Error::invalid_data("record chunk count overflow"))?;
                            let mut header = Vec::with_capacity(
                                HEADER_BYTES + if chunk_count == 0 { value.len() } else { 0 },
                            );
                            header.extend_from_slice(&length.to_le_bytes());
                            header.extend_from_slice(&count.to_le_bytes());
                            header.extend_from_slice(blake3::hash(value).as_bytes());
                            if chunk_count == 0 {
                                header.extend_from_slice(value);
                            } else {
                                for (index, bytes) in value.chunks(CHUNK_BYTES).enumerate() {
                                    chunks
                                        .insert((key.as_slice(), index as u64), bytes)
                                        .map_err(storage_error)?;
                                }
                            }
                            records
                                .insert(key.as_slice(), header.as_slice())
                                .map_err(storage_error)?;
                        }
                        RecordOp::Delete { key } => delete_record(&mut records, &mut chunks, key)?,
                        RecordOp::DeletePrefix { prefix } => loop {
                            let keys = records
                                .range(prefix.as_slice()..)
                                .map_err(storage_error)?
                                .take_while(|entry| {
                                    entry
                                        .as_ref()
                                        .map_or(true, |(k, _)| k.value().starts_with(prefix))
                                })
                                .take(DELETE_BATCH)
                                .map(|entry| {
                                    entry
                                        .map(|(k, _)| k.value().to_vec())
                                        .map_err(storage_error)
                                })
                                .collect::<Result<Vec<_>>>()?;
                            if keys.is_empty() {
                                break;
                            }
                            for key in keys {
                                delete_record(&mut records, &mut chunks, &key)?;
                            }
                        },
                    }
                }
                bookmark = batch.bookmark;
            }
            metadata
                .insert(b"bookmark".as_slice(), encode_bookmark(bookmark).as_slice())
                .map_err(storage_error)?;
        }
        transaction.commit().map_err(storage_error)
    }

    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.reader()?.get(key)
    }

    pub(crate) fn reader(&self) -> Result<RecordReader> {
        Ok(RecordReader {
            transaction: self.database.begin_read().map_err(storage_error)?,
        })
    }

    /// Atomically retires derived work only if its stored revision still matches. The canonical
    /// durability bookmark is unchanged; a newer owner revision cannot be removed by stale work.
    /// Like all RecordStore methods, this is called only on blocking workers.
    pub fn remove_if_value(&self, key: &[u8], expected: &[u8]) -> Result<bool> {
        let mut transaction = self.database.begin_write().map_err(storage_error)?;
        transaction.set_quick_repair(true);
        {
            let mut records = transaction.open_table(RECORDS).map_err(storage_error)?;
            let mut chunks = transaction.open_table(CHUNKS).map_err(storage_error)?;
            let matches = records
                .get(key)
                .map_err(storage_error)?
                .map(|value| read_record(&chunks, key, value.value()))
                .transpose()?
                .is_some_and(|value| value == expected);
            if !matches {
                return Ok(false);
            }
            delete_record(&mut records, &mut chunks, key)?;
        }
        transaction.commit().map_err(storage_error)?;
        Ok(true)
    }

    /// Visits one logical record at a time without collecting a second copy of the database.
    /// The callback must not recursively write to this store or retain borrowed values.
    pub fn scan_prefix(
        &self,
        prefix: &[u8],
        visit: impl FnMut(&[u8], &[u8]) -> Result<()>,
    ) -> Result<()> {
        self.reader()?.scan_prefix(prefix, visit)
    }

    pub fn next_after(
        &self,
        prefix: &[u8],
        after: Option<&[u8]>,
    ) -> Result<Option<(Vec<u8>, Vec<u8>)>> {
        let transaction = self.database.begin_read().map_err(storage_error)?;
        let records = transaction.open_table(RECORDS).map_err(storage_error)?;
        let chunks = transaction.open_table(CHUNKS).map_err(storage_error)?;
        let start = after
            .filter(|last| *last >= prefix)
            .map_or(Bound::Included(prefix), Bound::Excluded);
        let mut range = records
            .range::<&[u8]>((start, Bound::Unbounded))
            .map_err(storage_error)?;
        let Some(entry) = range.next() else {
            return Ok(None);
        };
        let (key, value) = entry.map_err(storage_error)?;
        if !key.value().starts_with(prefix) {
            return Ok(None);
        }
        Ok(Some((
            key.value().to_vec(),
            read_record(&chunks, key.value(), value.value())?,
        )))
    }
}

fn delete_record(
    records: &mut redb::Table<&[u8], &[u8]>,
    chunks: &mut redb::Table<(&[u8], u64), &[u8]>,
    key: &[u8],
) -> Result<()> {
    if let Some(value) = records.remove(key).map_err(storage_error)? {
        let (_, count, _) = decode_header(value.value())?;
        for index in 0..count {
            if chunks
                .remove((key, index))
                .map_err(storage_error)?
                .is_none()
            {
                return Err(corrupt("record chunk is absent during replacement"));
            }
        }
    }
    Ok(())
}

fn read_record(
    chunks: &impl ReadableTable<(&'static [u8], u64), &'static [u8]>,
    key: &[u8],
    encoded: &[u8],
) -> Result<Vec<u8>> {
    let (length, count, checksum) = decode_header(encoded)?;
    let length =
        usize::try_from(length).map_err(|_| corrupt("record length exceeds process range"))?;
    let expected_count = if length <= CHUNK_BYTES {
        0
    } else {
        length.div_ceil(CHUNK_BYTES) as u64
    };
    if count != expected_count {
        return Err(corrupt("record chunk count disagrees with length"));
    }
    let mut value = Vec::new();
    value
        .try_reserve_exact(length)
        .map_err(|_| Error::internal("record allocation failed"))?;
    if count == 0 {
        if encoded.len()
            != HEADER_BYTES
                .checked_add(length)
                .ok_or_else(|| corrupt("record extent overflow"))?
        {
            return Err(corrupt("inline record length mismatch"));
        }
        value.extend_from_slice(&encoded[HEADER_BYTES..]);
    } else {
        if encoded.len() != HEADER_BYTES {
            return Err(corrupt("chunked record header length mismatch"));
        }
        for index in 0..count {
            let chunk = chunks
                .get((key, index))
                .map_err(storage_error)?
                .ok_or_else(|| corrupt("record chunk is absent"))?;
            let expected = (length - value.len()).min(CHUNK_BYTES);
            if chunk.value().len() != expected {
                return Err(corrupt("record chunk length mismatch"));
            }
            value.extend_from_slice(chunk.value());
        }
    }
    if blake3::hash(&value).as_bytes() != checksum {
        return Err(corrupt("record checksum mismatch"));
    }
    Ok(value)
}

fn decode_header(bytes: &[u8]) -> Result<(u64, u64, &[u8])> {
    if bytes.len() < HEADER_BYTES {
        return Err(corrupt("record header is truncated"));
    }
    let length = u64::from_le_bytes(
        bytes[..8]
            .try_into()
            .map_err(|_| corrupt("record length is truncated"))?,
    );
    let count = u64::from_le_bytes(
        bytes[8..16]
            .try_into()
            .map_err(|_| corrupt("record count is truncated"))?,
    );
    let expected_count = if length <= CHUNK_BYTES as u64 {
        0
    } else {
        length.div_ceil(CHUNK_BYTES as u64)
    };
    if count != expected_count {
        return Err(corrupt("record chunk count disagrees with length"));
    }
    if count != 0 && bytes.len() != HEADER_BYTES {
        return Err(corrupt("chunked record header length mismatch"));
    }
    if count == 0
        && bytes.len() as u64
            != (HEADER_BYTES as u64)
                .checked_add(length)
                .ok_or_else(|| corrupt("record extent overflow"))?
    {
        return Err(corrupt("inline record length mismatch"));
    }
    Ok((length, count, &bytes[16..48]))
}
fn encode_bookmark(bookmark: Bookmark) -> [u8; 16] {
    let mut bytes = [0; 16];
    bytes[..8].copy_from_slice(&bookmark.term.to_le_bytes());
    bytes[8..].copy_from_slice(&bookmark.index.to_le_bytes());
    bytes
}
fn decode_bookmark(bytes: &[u8]) -> Result<Bookmark> {
    if bytes.len() != 16 {
        return Err(corrupt("durable bookmark length mismatch"));
    }
    let term = u64::from_le_bytes(
        bytes[..8]
            .try_into()
            .map_err(|_| corrupt("durable term is truncated"))?,
    );
    let index = u64::from_le_bytes(
        bytes[8..]
            .try_into()
            .map_err(|_| corrupt("durable index is truncated"))?,
    );
    if (term == 0) != (index == 0) {
        return Err(corrupt("durable bookmark is invalid"));
    }
    Ok(Bookmark { term, index })
}
fn corrupt(message: &'static str) -> Error {
    Error::new(ErrorCode::CorruptStorage, message)
}
fn storage_error(error: impl Into<redb::Error>) -> Error {
    let error = error.into();
    let code = match error {
        redb::Error::Corrupted(_) | redb::Error::UpgradeRequired(_) => ErrorCode::CorruptStorage,
        redb::Error::ValueTooLarge(_) => ErrorCode::InvalidData,
        _ => ErrorCode::Io,
    };
    Error::new(code, format!("record storage: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn batch(index: u64, operations: Vec<RecordOp>) -> RecordBatch {
        RecordBatch {
            first_index: index,
            bookmark: Bookmark { term: 1, index },
            operations,
        }
    }
    fn put(key: &[u8], value: &[u8]) -> RecordOp {
        RecordOp::Put {
            key: key.to_vec(),
            value: value.to_vec(),
        }
    }

    #[test]
    fn records_chunk_roundtrip_update_delete_and_prefix_seek() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("database.redb");
        let identity = Uuid::new_v4();
        let large = vec![7; CHUNK_BYTES * 3 + 19];
        {
            let store = RecordStore::open(&path, identity)?;
            store.commit(&[batch(
                1,
                vec![
                    put(b"a/1", &large),
                    put(b"a/2", b"two"),
                    put(b"b/1", b"three"),
                ],
            )])?;
            assert_eq!(store.get(b"a/1")?, Some(large.clone()));
            assert_eq!(
                store.next_after(b"a/", None)?,
                Some((b"a/1".to_vec(), large.clone()))
            );
            assert_eq!(
                store.next_after(b"a/", Some(b"a/1"))?,
                Some((b"a/2".to_vec(), b"two".to_vec()))
            );
            assert_eq!(store.next_after(b"a/", Some(b"a/2"))?, None);
            store.commit(&[batch(2, vec![put(b"a/1", b"tiny")])])?;
            let read = store.database.begin_read().map_err(storage_error)?;
            let chunks = read.open_table(CHUNKS).map_err(storage_error)?;
            assert_eq!(
                chunks.iter().map_err(storage_error)?.count(),
                0,
                "replacing a large value must remove its chunks"
            );
        }
        let store = RecordStore::open(&path, identity)?;
        assert_eq!(store.bookmark()?, Bookmark { term: 1, index: 2 });
        assert_eq!(store.get(b"a/1")?, Some(b"tiny".to_vec()));
        store.commit(&[batch(
            3,
            vec![RecordOp::DeletePrefix {
                prefix: b"a/".to_vec(),
            }],
        )])?;
        assert_eq!(store.next_after(b"a/", None)?, None);
        assert_eq!(store.get(b"b/1")?, Some(b"three".to_vec()));
        let mut visited = Vec::new();
        store.scan_prefix(b"", |key, value| {
            visited.push((key.to_vec(), value.to_vec()));
            Ok(())
        })?;
        assert_eq!(visited, vec![(b"b/1".to_vec(), b"three".to_vec())]);
        assert_eq!(
            std::fs::read_dir(directory.path())?.count(),
            1,
            "the store must create only its database file"
        );
        Ok(())
    }

    #[test]
    fn invalid_later_batch_aborts_all_record_changes_and_bookmark() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let store = RecordStore::open(directory.path().join("database.redb"), Uuid::new_v4())?;
        assert!(
            store
                .commit(&[
                    batch(1, vec![put(b"first", b"one")]),
                    batch(3, vec![put(b"second", b"two")])
                ])
                .is_err()
        );
        assert_eq!(store.bookmark()?, Bookmark::default());
        assert_eq!(store.get(b"first")?, None);
        assert_eq!(store.get(b"second")?, None);
        store.commit(&[
            batch(1, vec![put(b"first", b"one")]),
            batch(2, vec![put(b"second", b"two")]),
        ])?;
        assert_eq!(store.bookmark()?, Bookmark { term: 1, index: 2 });
        Ok(())
    }

    #[test]
    fn coalesced_spans_are_contiguous_atomic_and_apply_prefixes_before_points() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("database.redb");
        let identity = Uuid::new_v4();
        let store = RecordStore::open(&path, identity)?;
        store.commit(&[batch(
            1,
            vec![put(b"p/old", b"old"), put(b"other", b"keep")],
        )])?;
        let span = RecordBatch {
            first_index: 2,
            bookmark: Bookmark { term: 1, index: 20 },
            operations: vec![
                RecordOp::DeletePrefix {
                    prefix: b"p/".to_vec(),
                },
                put(b"p/new", b"latest"),
            ],
        };
        let mut gap = batch(22, vec![put(b"bad", b"value")]);
        assert!(store.commit(&[span.clone(), gap.clone()]).is_err());
        assert_eq!(store.bookmark()?.index, 1);
        assert_eq!(store.get(b"p/old")?, Some(b"old".to_vec()));
        assert_eq!(store.get(b"p/new")?, None);
        gap.first_index = 21;
        gap.bookmark.index = 20; // reversed range
        assert!(store.commit(&[span.clone(), gap]).is_err());
        assert_eq!(store.bookmark()?.index, 1);
        store.commit(&[
            span,
            RecordBatch {
                first_index: 21,
                bookmark: Bookmark { term: 2, index: 50 },
                operations: vec![],
            },
        ])?;
        drop(store);
        let store = RecordStore::open(path, identity)?;
        assert_eq!(store.bookmark()?, Bookmark { term: 2, index: 50 });
        assert_eq!(store.get(b"p/old")?, None);
        assert_eq!(store.get(b"p/new")?, Some(b"latest".to_vec()));
        assert_eq!(store.get(b"other")?, Some(b"keep".to_vec()));
        assert_eq!(store.get(b"bad")?, None);
        assert!(store.commit(&[batch(50, vec![])]).is_err());
        Ok(())
    }

    #[test]
    fn prefix_deletion_crosses_internal_batches_and_preserves_adjacent_namespaces() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let store = RecordStore::open(directory.path().join("database.redb"), Uuid::new_v4())?;
        let mut operations = vec![
            put(b"", b"empty"),
            put(b"aa", b"before"),
            put(b"c", b"after"),
        ];
        for index in 0..DELETE_BATCH * 3 + 1 {
            operations.push(put(
                format!("b/{index:04}").as_bytes(),
                &vec![4; CHUNK_BYTES + 1],
            ));
        }
        store.commit(&[batch(1, operations)])?;
        assert!(
            store
                .next_after(b"b/", Some(b"a"))?
                .is_some_and(|(key, _)| key == b"b/0000")
        );
        store.commit(&[batch(
            2,
            vec![RecordOp::DeletePrefix {
                prefix: b"b/".to_vec(),
            }],
        )])?;
        assert_eq!(store.next_after(b"b/", None)?, None);
        assert_eq!(store.get(b"")?, Some(b"empty".to_vec()));
        assert_eq!(store.get(b"aa")?, Some(b"before".to_vec()));
        assert_eq!(store.get(b"c")?, Some(b"after".to_vec()));
        let read = store.database.begin_read().map_err(storage_error)?;
        let chunks = read.open_table(CHUNKS).map_err(storage_error)?;
        assert_eq!(chunks.iter().map_err(storage_error)?.count(), 0);
        Ok(())
    }

    #[test]
    fn identity_is_verified_and_value_corruption_is_detected() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("database.redb");
        let identity = Uuid::new_v4();
        {
            let store = RecordStore::open(&path, identity)?;
            store.commit(&[batch(1, vec![put(b"key", b"value")])])?;
            let transaction = store.database.begin_write().map_err(storage_error)?;
            {
                let mut records = transaction.open_table(RECORDS).map_err(storage_error)?;
                records
                    .insert(b"key".as_slice(), b"invalid".as_slice())
                    .map_err(storage_error)?;
            }
            transaction.commit().map_err(storage_error)?;
            assert_eq!(
                store.get(b"key").err().map(|e| e.code),
                Some(ErrorCode::CorruptStorage)
            );
        }
        assert_eq!(
            RecordStore::open(&path, Uuid::new_v4())
                .err()
                .map(|e| e.code),
            Some(ErrorCode::CorruptStorage)
        );
        Ok(())
    }

    #[test]
    fn stale_derived_completion_cannot_delete_newer_work() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let store = RecordStore::open(directory.path().join("database.redb"), Uuid::new_v4())?;
        store.commit(&[batch(1, vec![put(b"e/owner", b"revision1")])])?;
        store.commit(&[batch(2, vec![put(b"e/owner", b"revision2")])])?;
        assert!(!store.remove_if_value(b"e/owner", b"revision1")?);
        assert_eq!(store.get(b"e/owner")?, Some(b"revision2".to_vec()));
        assert!(store.remove_if_value(b"e/owner", b"revision2")?);
        assert_eq!(store.get(b"e/owner")?, None);
        assert_eq!(store.bookmark()?, Bookmark { term: 1, index: 2 });
        assert!(!store.remove_if_value(b"e/owner", b"revision2")?);
        Ok(())
    }

    #[test]
    fn abrupt_process_death_keeps_durable_prefix_and_discards_uncommitted_pages() -> Result<()> {
        use std::io::BufRead;
        let directory = tempfile::tempdir()?;
        let path = directory.path().join("database.redb");
        let identity = Uuid::new_v4();
        let mut child = std::process::Command::new(std::env::current_exe()?)
            .args(["--exact", "records::tests::crash_child", "--nocapture"])
            .env("IRONGRAPH_STORAGE_CRASH_PATH", &path)
            .env("IRONGRAPH_STORAGE_CRASH_ID", identity.to_string())
            .stdout(std::process::Stdio::piped())
            .spawn()?;
        let stdout = child
            .stdout
            .take()
            .ok_or_else(|| Error::internal("child stdout missing"))?;
        let (send, receive) = std::sync::mpsc::channel();
        std::thread::spawn(move || {
            for line in std::io::BufReader::new(stdout).lines() {
                if let Ok(line) = line
                    && line.contains("DURABLE_PREFIX_READY")
                {
                    let _ = send.send(());
                    break;
                }
            }
        });
        let ready = receive.recv_timeout(std::time::Duration::from_secs(20));
        let _ = child.kill();
        child.wait()?;
        ready.map_err(|_| Error::internal("crash child did not reach durable prefix"))?;
        let store = RecordStore::open(path, identity)?;
        assert_eq!(store.bookmark()?, Bookmark { term: 1, index: 1 });
        assert_eq!(store.get(b"key")?, Some(vec![9; CHUNK_BYTES * 2 + 3]));
        assert_eq!(store.get(b"incomplete")?, None);
        Ok(())
    }

    #[test]
    fn crash_child() -> Result<()> {
        use std::io::Write;
        let Some(path) = std::env::var_os("IRONGRAPH_STORAGE_CRASH_PATH") else {
            return Ok(());
        };
        let identity = Uuid::parse_str(
            &std::env::var("IRONGRAPH_STORAGE_CRASH_ID")
                .map_err(|e| Error::internal(e.to_string()))?,
        )
        .map_err(|e| Error::internal(e.to_string()))?;
        let store = RecordStore::open(path, identity)?;
        store.commit(&[batch(1, vec![put(b"key", &vec![9; CHUNK_BYTES * 2 + 3])])])?;
        let transaction = store.database.begin_write().map_err(storage_error)?;
        let mut records = transaction.open_table(RECORDS).map_err(storage_error)?;
        records
            .insert(b"incomplete".as_slice(), b"uncommitted".as_slice())
            .map_err(storage_error)?;
        println!("DURABLE_PREFIX_READY");
        std::io::stdout().flush()?;
        loop {
            std::thread::park();
        }
    }
}
