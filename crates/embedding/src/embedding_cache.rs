//! Embeddings kept on disk, so the same text is never encoded twice.
//!
//! A pinned encoder is a pure function: the same text, under the same profile, always produces the
//! same vector. Recomputing one is therefore never necessary — only expensive. Indexing a graph of
//! a few thousand nodes cost more than a minute of forward passes, every start, for vectors that
//! had already been computed and thrown away; at a million nodes it would never finish.
//!
//! Content-addressed rather than element-addressed. The key is the hash of the text, so an edit
//! misses and re-encodes, an unchanged node hits, and two nodes that happen to say the same thing
//! share one entry. Nothing here needs to know about graphs, projects or revisions — which is what
//! lets it sit under the encoder and serve every caller rather than just the index builder.
//!
//! The profile hash is part of the path, so changing encoder cannot serve vectors from the old one.

use std::{
    collections::HashMap,
    fs::OpenOptions,
    io::{Read, Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::Mutex,
};

use sha2::{Digest as _, Sha256};

use crate::{Error, ErrorCode, Result};

/// One record: the key that identifies the text, then the vector itself.
///
/// Fixed width so the file is a flat array. A partial trailing record — a crash mid-append — is
/// discarded on load rather than treated as data.
const KEY_BYTES: usize = 32;

/// Bump only when the semantic embedding recipe changes without changing the public profile.
/// Execution backends do not belong in this key: the Core ML program and Candle path implement the
/// same pinned bidirectional/masked-average/Matryoshka recipe, and are numerically cross-validated.
/// Keying on the backend would discard correct persisted vectors whenever execution is accelerated.
/// Earlier caches did need invalidating after fixes to bidirectional attention and batched pooling,
/// because those changed the actual recipe rather than merely how it was executed.
const ENCODING_RECIPE_REVISION: &[u8] =
    b"llama-nemotron-bidirectional-masked-average-matryoshka384-v2";

/// Vectors the file may hold before the oldest are dropped.
///
/// Nothing here knows which graph elements are alive, and it deliberately does not: keys are
/// content, so a vector is useful to anyone who encodes that text again, in any project. Without a
/// bound the file grows forever, because an edited node leaves its old vector behind and nothing
/// ever comes back for it.
///
/// Insertion order is the eviction order. It is not a claim about which vectors matter — it is the
/// one ordering the file already has, and it costs a rewrite rather than an index. At this size the
/// file is about a gigabyte and a half, which is a reasonable ceiling for a cache that exists to
/// avoid hours of encoding.
const MAX_CACHED_VECTORS: usize = 1_000_000;

/// Vectors on disk beside the model weights, keyed by what was encoded.
///
/// The file is mapped rather than read. Holding a million 384-float vectors in a map would be about
/// a gigabyte and a half of resident memory for data that is already sitting on disk in exactly the
/// layout it is wanted in; mapped, the process holds a key and an offset per entry — around forty
/// megabytes at that size — and the kernel pages the vectors in as they are touched and evicts them
/// under pressure. Vectors appended during this run retain only their file offsets; reads use the
/// kernel page cache until the next open maps the complete file. This keeps a long-lived ingest
/// from duplicating up to 1.5 GB of already-durable vector values in a HashMap.
pub(super) struct EmbeddingCache {
    path: PathBuf,
    width: usize,
    /// The file as it stood when opened, plus where each key's vector begins in it.
    mapped: Option<memmap2::Mmap>,
    offsets: HashMap<[u8; KEY_BYTES], usize>,
    /// Written during this run, not yet in `mapped`; values are absolute vector byte offsets.
    fresh_offsets: Mutex<HashMap<[u8; KEY_BYTES], usize>>,
    /// Serializes complete append batches so records from concurrent encoders cannot interleave.
    append: Mutex<()>,
}

impl EmbeddingCache {
    /// Open the cache for one encoder profile, reading whatever it already holds.
    ///
    /// A cache that cannot be opened or parsed is not an error: the encoder still works, it is
    /// merely slower, and refusing to start because a cache file is unreadable would trade a
    /// performance feature for availability.
    pub(super) fn open(profile_hash: &str, width: usize) -> Option<Self> {
        let home = std::env::var("HOME").ok()?;
        let directory = PathBuf::from(home).join(".irongraph").join("embeddings");
        Self::open_in(&directory, profile_hash, width)
    }

    /// Open a cache in a named directory.
    ///
    /// Separate from [`Self::open`] so the location is an argument rather than an environment
    /// variable — which is what lets a test exercise the real mapped read path without mutating the
    /// process environment out from under every other test in the binary.
    pub(super) fn open_in(directory: &Path, profile_hash: &str, width: usize) -> Option<Self> {
        std::fs::create_dir_all(directory).ok()?;
        let path = directory.join(format!("{profile_hash}.vectors"));
        Self::compact_if_oversized(&path, width);
        Self::truncate_partial_tail(&path, width);
        let (mapped, offsets) = Self::map_existing(&path, width);
        Some(Self {
            path,
            width,
            mapped,
            offsets,
            fresh_offsets: Mutex::new(HashMap::new()),
            append: Mutex::new(()),
        })
    }

    /// Remove a crash-partial tail before another append. Otherwise every later complete record
    /// remains shifted behind the partial bytes and is invisible on reopen.
    fn truncate_partial_tail(path: &Path, width: usize) {
        let record = KEY_BYTES + width * 4;
        let Ok(file) = OpenOptions::new().write(true).open(path) else {
            return;
        };
        let Ok(metadata) = file.metadata() else {
            return;
        };
        let complete = metadata.len() / record as u64 * record as u64;
        if complete != metadata.len() {
            let _ = file.set_len(complete);
        }
    }

    /// Map the file and record where each key's vector starts.
    ///
    /// One pass over the keys, which is the only part that has to be resident. A trailing partial
    /// record — a crash mid-append — is ignored rather than treated as data.
    fn map_existing(
        path: &Path,
        width: usize,
    ) -> (Option<memmap2::Mmap>, HashMap<[u8; KEY_BYTES], usize>) {
        let Ok(mapped) = irongraph_artifact_loader::map_owned_file(path) else {
            return (None, HashMap::new());
        };
        let record = KEY_BYTES + width * 4;
        let mut offsets = HashMap::new();
        let mut at = 0usize;
        while at + record <= mapped.len() {
            let mut key = [0_u8; KEY_BYTES];
            key.copy_from_slice(&mapped[at..at + KEY_BYTES]);
            offsets.insert(key, at + KEY_BYTES);
            at += record;
        }
        (Some(mapped), offsets)
    }

    /// Drop the oldest records when the file has outgrown its bound.
    ///
    /// Done at open, when nothing is mapped and no reader can be part-way through a record. The
    /// rewrite goes to a sibling file and is renamed over the original, so an interrupted
    /// compaction leaves the previous cache intact rather than a truncated one.
    ///
    /// Failure is silent by design: a cache that cannot be compacted is a cache that is too big,
    /// which is slow, and refusing to start over it would be worse.
    fn compact_if_oversized(path: &Path, width: usize) {
        let record = KEY_BYTES + width * 4;
        let Ok(metadata) = std::fs::metadata(path) else {
            return;
        };
        let held = metadata.len() as usize / record;
        if held <= MAX_CACHED_VECTORS {
            return;
        }
        let Ok(bytes) = std::fs::read(path) else {
            return;
        };
        // Keep the newest, which are at the end: the file is append-ordered.
        let keep_from = (held - MAX_CACHED_VECTORS) * record;
        let end = held * record;
        let temporary = path.with_extension("compacting");
        if std::fs::write(&temporary, &bytes[keep_from..end]).is_ok() {
            let _ = std::fs::rename(&temporary, path);
        } else {
            let _ = std::fs::remove_file(&temporary);
        }
    }

    /// The key for one text. Includes the width so a profile change cannot alias.
    pub(super) fn key(text: &str, width: usize) -> [u8; KEY_BYTES] {
        let mut hasher = Sha256::new();
        hasher.update(ENCODING_RECIPE_REVISION);
        hasher.update(width.to_le_bytes());
        hasher.update(text.as_bytes());
        hasher.finalize().into()
    }

    pub(super) fn get(&self, key: &[u8; KEY_BYTES]) -> Option<Vec<f32>> {
        if let (Some(mapped), Some(offset)) = (self.mapped.as_ref(), self.offsets.get(key)) {
            let end = offset + self.width * 4;
            if end <= mapped.len() {
                return Some(
                    mapped[*offset..end]
                        .chunks_exact(4)
                        .map(|bytes| f32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
                        .collect(),
                );
            }
        }
        let offset = *self.fresh_offsets.lock().ok()?.get(key)?;
        let mut file = OpenOptions::new().read(true).open(&self.path).ok()?;
        file.seek(SeekFrom::Start(offset as u64)).ok()?;
        let mut bytes = vec![0_u8; self.width * 4];
        file.read_exact(&mut bytes).ok()?;
        Some(
            bytes
                .chunks_exact(4)
                .map(|value| f32::from_le_bytes([value[0], value[1], value[2], value[3]]))
                .collect(),
        )
    }

    /// Record newly computed vectors on disk and retain only their offsets.
    ///
    /// Appended in one write rather than one per vector, and a failed write is dropped: the vectors
    /// are already correct in memory, and the only cost of losing them is encoding them again.
    pub(super) fn put(&self, batch: &[([u8; KEY_BYTES], Vec<f32>)]) -> Result<()> {
        if batch.is_empty() {
            return Ok(());
        }
        let _append = self.append.lock().map_err(|_| {
            Error::new(
                ErrorCode::Internal,
                "the embedding cache append lock was poisoned",
            )
        })?;
        let Ok(mut file) = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
        else {
            return Ok(());
        };
        let start = match file.metadata() {
            Ok(metadata) => metadata.len() as usize,
            Err(_) => return Ok(()),
        };
        let record = KEY_BYTES + self.width * 4;
        let valid = batch
            .iter()
            .filter(|(_, vector)| vector.len() == self.width)
            .count();
        let mut encoded = Vec::with_capacity(valid.saturating_mul(record));
        let mut appended = Vec::with_capacity(valid);
        for (key, vector) in batch {
            if vector.len() != self.width {
                continue;
            }
            let offset = start + encoded.len() + KEY_BYTES;
            encoded.extend_from_slice(key);
            for value in vector {
                encoded.extend_from_slice(&value.to_le_bytes());
            }
            appended.push((*key, offset));
        }
        if encoded.is_empty() || file.write_all(&encoded).is_err() || file.flush().is_err() {
            return Ok(());
        }
        let mut fresh = self.fresh_offsets.lock().map_err(|_| {
            Error::new(
                ErrorCode::Internal,
                "the embedding cache offset lock was poisoned",
            )
        })?;
        fresh.extend(appended);
        Ok(())
    }

    #[cfg(test)]
    fn fresh_resident_value_bytes(&self) -> usize {
        self.fresh_offsets.lock().map_or(usize::MAX, |offsets| {
            offsets.len() * (KEY_BYTES + size_of::<usize>())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::EmbeddingCache;

    /// The key must change with the text and with the width, and not otherwise.
    #[test]
    fn keys_identify_text_and_width() {
        let a = EmbeddingCache::key("hello", 384);
        assert_eq!(a, EmbeddingCache::key("hello", 384));
        assert_ne!(a, EmbeddingCache::key("hello ", 384));
        assert_ne!(a, EmbeddingCache::key("hello", 512));
    }

    /// Written, then read back through the mapping on the next open — the path that matters.
    #[test]
    fn vectors_survive_a_reopen_through_the_mapping() {
        let home = tempfile::tempdir().expect("temp directory");
        let profile = "test-profile-roundtrip";

        let written = {
            let cache = EmbeddingCache::open_in(home.path(), profile, 4).expect("cache opens");
            let key = EmbeddingCache::key("a message", 4);
            cache
                .put(&[(key, vec![0.5, -0.25, 0.125, 1.0])])
                .expect("put succeeds");
            // Before any reopen it comes from the just-appended file offset.
            assert_eq!(cache.get(&key), Some(vec![0.5, -0.25, 0.125, 1.0]));
            key
        };

        let reopened = EmbeddingCache::open_in(home.path(), profile, 4).expect("cache reopens");
        assert_eq!(
            reopened.get(&written),
            Some(vec![0.5, -0.25, 0.125, 1.0]),
            "the mapped read must return exactly what was appended"
        );
        assert_eq!(
            reopened.get(&EmbeddingCache::key("never written", 4)),
            None,
            "an unknown key must miss rather than return a neighbour's vector"
        );
    }

    /// The file is bounded: past the ceiling, the oldest records go and the newest survive.
    #[test]
    fn embedding_cache_residency_is_bounded() {
        let home = tempfile::tempdir().expect("temp directory");
        let profile = "test-profile-compaction";
        let width = 2;
        // Write more than the bound allows, in order, so "oldest" is unambiguous.
        {
            let cache = EmbeddingCache::open_in(home.path(), profile, width).expect("opens");
            for start in (0..super::MAX_CACHED_VECTORS + 8).step_by(4_096) {
                let end = (start + 4_096).min(super::MAX_CACHED_VECTORS + 8);
                let records = (start..end)
                    .map(|index| {
                        (
                            EmbeddingCache::key(&format!("text {index}"), width),
                            vec![index as f32, 0.0],
                        )
                    })
                    .collect::<Vec<_>>();
                cache.put(&records).expect("put");
            }
            assert!(
                cache.fresh_resident_value_bytes()
                    <= (super::MAX_CACHED_VECTORS + 8) * (super::KEY_BYTES + size_of::<usize>()),
                "new vectors retain keys and offsets, never a second copy of vector coordinates"
            );
            let newest =
                EmbeddingCache::key(&format!("text {}", super::MAX_CACHED_VECTORS + 7), width);
            assert_eq!(
                cache.get(&newest),
                Some(vec![(super::MAX_CACHED_VECTORS + 7) as f32, 0.0]),
                "an offset-backed vector is readable before reopen"
            );
        }

        let reopened = EmbeddingCache::open_in(home.path(), profile, width).expect("reopens");
        let oldest = EmbeddingCache::key("text 0", width);
        let newest = EmbeddingCache::key(&format!("text {}", super::MAX_CACHED_VECTORS + 7), width);
        assert_eq!(reopened.get(&oldest), None, "the oldest record was dropped");
        assert!(
            reopened.get(&newest).is_some(),
            "the newest record survived compaction"
        );
    }

    /// A truncated trailing record is a crash mid-append, not data.
    #[test]
    fn a_partial_trailing_record_is_ignored() {
        let home = tempfile::tempdir().expect("temp directory");
        let profile = "test-profile-partial";
        let key = EmbeddingCache::key("complete", 4);
        {
            let cache = EmbeddingCache::open_in(home.path(), profile, 4).expect("cache opens");
            cache.put(&[(key, vec![1.0, 2.0, 3.0, 4.0])]).expect("put");
        }
        let path = home.path().join(format!("{profile}.vectors"));
        // Append half a record, as an interrupted write would leave behind.
        let mut file = std::fs::OpenOptions::new()
            .append(true)
            .open(&path)
            .expect("open for append");
        std::io::Write::write_all(&mut file, &[7_u8; 20]).expect("partial append");
        drop(file);

        let reopened = EmbeddingCache::open_in(home.path(), profile, 4).expect("cache reopens");
        assert_eq!(reopened.get(&key), Some(vec![1.0, 2.0, 3.0, 4.0]));
    }
}
