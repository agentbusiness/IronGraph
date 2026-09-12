use std::{
    path::PathBuf,
    sync::atomic::{AtomicBool, Ordering},
};

#[cfg(feature = "accelerator")]
use std::{borrow::Cow, io::Read as _};

#[cfg(feature = "accelerator")]
use parking_lot::{Mutex, RwLock};

use crate::{
    Error, ErrorCode, Result,
    execution::TextEmbedding,
    gpu::{BackendKind, DeviceMemoryGovernor, ResolvedComputeDevice},
    graph::{EmbeddingDType, EmbeddingProfile, Similarity},
};

#[cfg(feature = "accelerator")]
use crate::gpu::SharedMemoryClass;

use super::EmbeddingDevice;
#[cfg(feature = "accelerator")]
use super::{
    device,
    encoder::{BidirectionalEncoder, BidirectionalEncoderConfig},
};

const MAX_EMBEDDING_TEXT_BYTES: usize = 4 * 1024 * 1024;
#[cfg(feature = "accelerator")]
pub(super) const PINNED_HIDDEN_SIZE: usize = 2_048;
/// Texts per forward pass while indexing. Large enough to amortise the call, small enough that
/// the padded rectangle stays modest when one text in the batch is long.
#[cfg(feature = "accelerator")]
const EMBEDDING_BATCH: usize = 16;

/// Padded tokens a single batch may carry.
///
/// The rectangle is what the device actually processes, so this — not the row count — is what keeps
/// one long document from dragging its whole batch with it. It also decides how much of the long
/// tail gets batched, and bigger is not better. At 8,192 tokens a representative 200-document
/// rebuild held roughly 20 GiB of transient encoder state and spent 228 seconds waiting on one
/// Metal command buffer. A 4,096-token rectangle was still able to peak above 7 GiB and leave the
/// real 645-message cold rebuild waiting indefinitely in Metal. Keep each submission to 1,024
/// padded tokens and at most sixteen rows. This is also small enough for an interactive query to
/// get the encoder lock between background batches without a multi-second head-of-line stall.
#[cfg(feature = "accelerator")]
const EMBEDDING_BATCH_TOKENS: usize = 1_024;

// Shared rolling windows use the same bound on every embedding execution backend.
#[cfg(feature = "accelerator")]
const COREML_INPUT_TOKENS: usize = 512;

pub(super) const PINNED_OUTPUT_SIZE: usize = 384;
pub(super) const PINNED_MAXIMUM_INPUT_TOKENS: usize = 8_192;
#[cfg(feature = "accelerator")]
const QUERY_PREFIX: &str = "query: ";
#[cfg(feature = "accelerator")]
const PASSAGE_PREFIX: &str = "passage: ";
#[cfg(feature = "accelerator")]
const PRESERVED_INPUT_HEAD_TOKENS: usize = 16;

#[cfg(feature = "accelerator")]
fn embedding_error(error: candle_core::Error) -> Error {
    Error::new(
        ErrorCode::EmbeddingUnavailable,
        format!("local text embedding failed: {error}"),
    )
}

/// Immutable exact files for the project-compatible local text encoder.
#[derive(Clone, Debug)]
pub struct EmbeddingModelArtifacts {
    pub model_safetensors: PathBuf,
    pub tokenizer_json: PathBuf,
    pub model_config_json: PathBuf,
    pub model_sha256: [u8; 32],
    pub model_exact_bytes: u64,
    pub tokenizer_sha256: [u8; 32],
    pub tokenizer_exact_bytes: u64,
    pub config_sha256: [u8; 32],
    pub config_exact_bytes: u64,
    pub profile: EmbeddingProfile,
    pub maximum_input_tokens: usize,
}

/// Hash-verified local encoder used only by the sequencer to resolve canonical vectors.
pub struct LocalEmbeddingModel {
    artifacts: EmbeddingModelArtifacts,
    resolved_device: ResolvedComputeDevice,
    warmed: AtomicBool,
    #[cfg(feature = "accelerator")]
    inner: Mutex<LoadedEmbeddingModel>,
    #[cfg(feature = "accelerator")]
    governor: RwLock<Option<DeviceMemoryGovernor>>,
    /// Vectors already computed for this profile, kept on disk between runs. `None` when the cache
    /// could not be opened, which costs speed and nothing else.
    #[cfg(feature = "accelerator")]
    cache: Option<super::embedding_cache::EmbeddingCache>,
    /// Recently encoded query windows, in memory only.
    ///
    /// Separate from `cache` on purpose: that one is the durable passage store keyed for indexing,
    /// while this exists because one question is embedded repeatedly — the repeated semantic query encodes
    /// the current question every turn, and a re-asked or retried question encodes it again. A
    /// query embed measured 26-53 ms, all of it on the request path.
    #[cfg(feature = "accelerator")]
    query_cache: Mutex<QueryEmbeddingCache>,
}

/// Bounded most-recent-wins cache for query vectors.
///
/// Insertion-ordered rather than true LRU: queries arrive in bursts around one conversation, so
/// evicting the oldest insertion is the same decision an LRU would make almost every time, without
/// carrying recency bookkeeping on every hit.
#[cfg(feature = "accelerator")]
#[derive(Default)]
struct QueryEmbeddingCache {
    entries: std::collections::HashMap<String, Vec<f32>>,
    order: std::collections::VecDeque<String>,
}

#[cfg(feature = "accelerator")]
impl QueryEmbeddingCache {
    /// Enough to cover a conversation's worth of questions and their retries. At 384 f32 per
    /// entry this is about 1.5 MB when full.
    const CAPACITY: usize = 1_024;

    fn get(&self, text: &str) -> Option<Vec<f32>> {
        self.entries.get(text).cloned()
    }

    fn insert(&mut self, text: &str, vector: &[f32]) {
        if self.entries.contains_key(text) {
            return;
        }
        while self.order.len() >= Self::CAPACITY {
            if let Some(evicted) = self.order.pop_front() {
                self.entries.remove(&evicted);
            }
        }
        self.entries.insert(text.to_owned(), vector.to_vec());
        self.order.push_back(text.to_owned());
    }
}

#[cfg(feature = "accelerator")]
struct LoadedEmbeddingModel {
    model: BidirectionalEncoder,
    tokenizer: tokenizers::Tokenizer,
    device: candle_core::Device,
    maximum_input_tokens: usize,
    #[cfg(target_vendor = "apple")]
    coreml_buckets: Vec<CoreMlEmbeddingProgram>,
}

#[cfg(all(feature = "accelerator", target_vendor = "apple"))]
struct CoreMlEmbeddingProgram {
    model: coreml_native::Model,
    resident_bytes: usize,
    fixed_batch: usize,
    fixed_tokens: usize,
    /// The compiled artifact backing `model`. Retained so it can be released when it
    /// is still Core ML's temporary output rather than a cached copy.
    compiled_path: PathBuf,
    /// True when `compiled_path` is in `$TMPDIR` and this process must delete it.
    owns_temp: bool,
}

#[cfg(all(feature = "accelerator", target_vendor = "apple"))]
impl Drop for CoreMlEmbeddingProgram {
    fn drop(&mut self) {
        // Core ML hands the caller ownership of its compile output. When we could not
        // move it into the cache, releasing it here is what stops `$TMPDIR` growing by
        // ~2.3GB per program on every process start.
        if self.owns_temp
            && self.compiled_path.starts_with(std::env::temp_dir())
            && std::fs::remove_dir_all(&self.compiled_path).is_ok()
        {
            tracing::debug!(
                path = %self.compiled_path.display(),
                "released temporary Core ML compile artifact"
            );
        }
    }
}

#[cfg(feature = "accelerator")]
struct BoundedEmbeddingInput<'a> {
    ids: Cow<'a, [u32]>,
    attention: Cow<'a, [u32]>,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum EmbeddingPurpose {
    Query,
    Passage,
}

impl EmbeddingPurpose {
    #[cfg(feature = "accelerator")]
    const fn prefix(self) -> &'static str {
        match self {
            Self::Query => QUERY_PREFIX,
            Self::Passage => PASSAGE_PREFIX,
        }
    }
}

impl LocalEmbeddingModel {
    pub fn load(artifacts: EmbeddingModelArtifacts, device: EmbeddingDevice) -> Result<Self> {
        validate_artifact_contract(&artifacts)?;

        #[cfg(feature = "accelerator")]
        {
            let selected = device::select_device(device)?;
            let device = selected.device;
            let dtype = selected.dtype;
            let resolved_device = selected.identity;
            let config_bytes = read_verified_sha256_exact(
                &artifacts.model_config_json,
                artifacts.config_exact_bytes,
                artifacts.config_sha256,
            )?;
            let config: BidirectionalEncoderConfig = serde_json::from_slice(&config_bytes)
                .map_err(|error| {
                    Error::new(
                        ErrorCode::EmbeddingUnavailable,
                        format!("embedding model config is invalid: {error}"),
                    )
                })?;
            config.validate_pinned(artifacts.maximum_input_tokens)?;
            let tokenizer_bytes = read_verified_sha256_exact(
                &artifacts.tokenizer_json,
                artifacts.tokenizer_exact_bytes,
                artifacts.tokenizer_sha256,
            )?;
            let tokenizer =
                tokenizers::Tokenizer::from_bytes(&tokenizer_bytes).map_err(|error| {
                    Error::new(
                        ErrorCode::EmbeddingUnavailable,
                        format!("embedding tokenizer is invalid: {error}"),
                    )
                })?;
            if tokenizer.get_vocab_size(true) != 128_256 {
                return Err(Error::new(
                    ErrorCode::EmbeddingProfileMismatch,
                    "embedding tokenizer vocabulary differs from the pinned encoder",
                ));
            }
            let builder = irongraph_artifact_loader::mmap_verified_safetensors_sha256_exact(
                &artifacts.model_safetensors,
                artifacts.model_exact_bytes,
                artifacts.model_sha256,
                dtype,
                &device,
            )
            .map_err(embedding_error)?;
            let embedding_model =
                BidirectionalEncoder::load(builder, &config, artifacts.maximum_input_tokens)
                    .map_err(embedding_error)?;
            #[cfg(target_vendor = "apple")]
            let coreml_buckets = if resolved_device.backend == BackendKind::Metal {
                // Reclaim artifacts stranded by runs that were killed before Drop could
                // fire. An hour is comfortably longer than a cold compile, so a
                // concurrently starting process cannot lose its in-flight output.
                sweep_stale_coreml_artifacts(std::time::Duration::from_secs(60 * 60));
                [
                    ("model-64.mlpackage", 64),
                    ("model-512.mlpackage", COREML_INPUT_TOKENS),
                ]
                .into_iter()
                .filter_map(|(name, tokens)| {
                    load_coreml_fixed(&artifacts.model_safetensors, name, 1, tokens)
                })
                .collect()
            } else {
                Vec::new()
            };
            let maximum_input_tokens = artifacts.maximum_input_tokens;
            // Hex so it can name a file, and part of the path so a different encoder can never be
            // served vectors produced by this one.
            let artifacts_profile_hash = artifacts
                .profile
                .profile_hash
                .iter()
                .map(|byte| format!("{byte:02x}"))
                .collect::<String>();
            Ok(Self {
                artifacts,
                resolved_device,
                warmed: AtomicBool::new(false),
                inner: Mutex::new(LoadedEmbeddingModel {
                    model: embedding_model,
                    tokenizer,
                    device,
                    maximum_input_tokens,
                    #[cfg(target_vendor = "apple")]
                    coreml_buckets,
                }),
                governor: RwLock::new(None),
                cache: super::embedding_cache::EmbeddingCache::open(
                    &artifacts_profile_hash,
                    PINNED_OUTPUT_SIZE,
                ),
                query_cache: Mutex::new(QueryEmbeddingCache::default()),
            })
        }
        #[cfg(not(feature = "accelerator"))]
        {
            let _ = device;
            Err(Error::new(
                ErrorCode::EmbeddingUnavailable,
                "binary was built without the local embedding encoder",
            ))
        }
    }

    #[must_use]
    pub fn profile(&self) -> &EmbeddingProfile {
        &self.artifacts.profile
    }

    /// Encodes native SEARCH/retrieval text with the pinned query prefix.
    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_for(EmbeddingPurpose::Query, text)
    }

    /// Encodes indexed graph/document values with the pinned passage prefix.
    pub fn embed_passage(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_for(EmbeddingPurpose::Passage, text)
    }

    /// Embed many passages in as few forward passes as the batch size allows.
    ///
    /// Same vectors as calling [`Self::embed_passage`] on each, at a fraction of the cost: a
    /// forward pass has a fixed overhead that one short sequence barely uses, and indexing a graph
    /// makes thousands of them.
    pub fn embed_passages(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embed_passages_cancellable(texts, &|| false)
    }

    /// Background semantic-index construction can yield between bounded encoder forwards when an
    /// interactive query arrives. Holding the one local encoder lock for a whole mailbox rebuild
    /// made queries wait tens of seconds behind idle maintenance.
    pub fn embed_passages_cancellable(
        &self,
        texts: &[String],
        is_cancelled: &dyn Fn() -> bool,
    ) -> Result<Vec<Vec<f32>>> {
        #[cfg(not(feature = "accelerator"))]
        let _ = is_cancelled;
        for text in texts {
            if text.is_empty() || text.len() > MAX_EMBEDDING_TEXT_BYTES {
                return Err(Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "embedding text is empty or exceeds its byte budget",
                ));
            }
        }
        #[cfg(feature = "accelerator")]
        {
            ensure_embedding_not_cancelled(is_cancelled)?;
            // Already known vectors never reach the encoder.
            //
            // A pinned encoder is a pure function of its input, so a text encoded once never needs
            // encoding again. On a graph that mostly did not change since the last start — which is
            // every graph, most of the time — this is the difference between minutes of forward
            // passes and reading a file.
            let mut out: Vec<Option<Vec<f32>>> = Vec::with_capacity(texts.len());
            // Multiple graph elements may have byte-identical semantic prose (document templates are
            // the common case). The on-disk cache cannot help until this call publishes its newly
            // computed rows, so deduplicate misses inside the call as well: one encoder forward per
            // distinct text, then fan that vector back out to every original row.
            let mut work_texts = Vec::<String>::new();
            let mut work_by_text = std::collections::HashMap::<String, usize>::new();
            let mut missing = Vec::<(usize, usize)>::new();
            for (index, text) in texts.iter().enumerate() {
                let cached = self.cache.as_ref().and_then(|cache| {
                    cache.get(&super::embedding_cache::EmbeddingCache::key(
                        text,
                        PINNED_OUTPUT_SIZE,
                    ))
                });
                if cached.is_none() {
                    let work_index = match work_by_text.get(text) {
                        Some(index) => *index,
                        None => {
                            let index = work_texts.len();
                            work_texts.push(text.clone());
                            work_by_text.insert(text.clone(), index);
                            index
                        }
                    };
                    missing.push((index, work_index));
                }
                out.push(cached);
            }
            if missing.is_empty() {
                // Per-call instrumentation, so it lives at debug: at INFO it printed several lines
                // a second into the operator's console during ordinary retrieval. A call that was
                // handed nothing to encode says nothing at all — `requested=0 cache_hits=0
                // unique_misses=0` is a line about no work having happened.
                if !texts.is_empty() {
                    tracing::debug!(
                        requested = texts.len(),
                        cache_hits = texts.len(),
                        unique_misses = 0,
                        "embedding cache lookup completed"
                    );
                }
                return out
                    .into_iter()
                    .map(|vector| {
                        vector.ok_or_else(|| {
                            Error::new(
                                ErrorCode::EmbeddingUnavailable,
                                "an embedding row went missing",
                            )
                        })
                    })
                    .collect();
            }
            let outer = out;
            // Same call, the branch with work to do. `missing` is non-empty here, so there is
            // always something to report; only the level was wrong.
            tracing::debug!(
                requested = texts.len(),
                cache_hits = texts.len().saturating_sub(missing.len()),
                unique_misses = work_texts.len(),
                "embedding cache lookup completed"
            );

            let governor = self.governor.read().clone();

            // Tokenise once, then group by length.
            //
            // A batch is padded to its longest row, so mixing a two-token name with an
            // eight-thousand-token document makes the model chew a rectangle that is almost
            // entirely padding — slower than doing them one at a time. Sorted by length, each
            // batch is nearly square, and batches are closed on total padded area rather than on a
            // fixed count so a few long rows never build a large one.
            let mut rows: Vec<(usize, (Vec<u32>, Vec<u32>))> = {
                let loaded = self.inner.lock();
                let mut rows = Vec::with_capacity(work_texts.len());
                for (index, text) in work_texts.iter().enumerate() {
                    ensure_embedding_not_cancelled(is_cancelled)?;
                    rows.push((
                        index,
                        tokenize_for(&loaded, EmbeddingPurpose::Passage, text)?,
                    ));
                }
                rows
            };
            rows.sort_by_key(|(_, (ids, _))| ids.len());

            // Batch row indices address the deduplicated work set, not the original input. Using
            // `texts.len()` left an unfilled tail whenever repeated document templates collapsed to
            // fewer work rows, and the rebuild failed after minutes of successful encoding with
            // "an embedding row went missing".
            let mut out: Vec<Option<Vec<f32>>> = (0..work_texts.len()).map(|_| None).collect();
            let mut batch: Vec<(usize, (Vec<u32>, Vec<u32>))> = Vec::new();
            let mut widest = 0usize;
            for row in rows {
                let next_widest = widest.max(row.1.0.len());
                let would_be = next_widest.saturating_mul(batch.len() + 1);
                if !batch.is_empty()
                    && (would_be > EMBEDDING_BATCH_TOKENS || batch.len() >= EMBEDDING_BATCH)
                {
                    let completed = batch.iter().map(|(index, _)| *index).collect::<Vec<_>>();
                    flush_embedding_batch(
                        &self.inner,
                        &mut batch,
                        governor.as_ref(),
                        &mut out,
                        is_cancelled,
                    )?;
                    cache_completed_embedding_rows(
                        self.cache.as_ref(),
                        &work_texts,
                        &completed,
                        &out,
                    )?;
                    widest = 0;
                }
                widest = widest.max(row.1.0.len());
                batch.push(row);
            }
            let completed = batch.iter().map(|(index, _)| *index).collect::<Vec<_>>();
            flush_embedding_batch(
                &self.inner,
                &mut batch,
                governor.as_ref(),
                &mut out,
                is_cancelled,
            )?;
            cache_completed_embedding_rows(self.cache.as_ref(), &work_texts, &completed, &out)?;

            let computed: Vec<Vec<f32>> = out
                .into_iter()
                .map(|vector| {
                    vector.ok_or_else(|| {
                        Error::new(
                            ErrorCode::EmbeddingUnavailable,
                            "an embedding row went missing",
                        )
                    })
                })
                .collect::<Result<_>>()?;

            let mut merged = outer;
            for (original_index, work_index) in missing {
                merged[original_index] = Some(computed[work_index].clone());
            }
            merged
                .into_iter()
                .map(|vector| {
                    vector.ok_or_else(|| {
                        Error::new(
                            ErrorCode::EmbeddingUnavailable,
                            "an embedding row went missing",
                        )
                    })
                })
                .collect()
        }
        #[cfg(not(feature = "accelerator"))]
        {
            Err(Error::new(
                ErrorCode::EmbeddingUnavailable,
                "binary was built without the local embedding encoder",
            ))
        }
    }

    /// Executes the real tokenizer, bidirectional forward, pooling, truncation, and device-to-host
    /// path before model readiness is published.
    pub fn warm_up(&self) -> Result<()> {
        if self.warmed.load(Ordering::Acquire) {
            return Ok(());
        }
        let started = std::time::Instant::now();
        let vector = self.embed_query("warmup")?;
        if vector.len() != PINNED_OUTPUT_SIZE {
            return Err(Error::new(
                ErrorCode::EmbeddingUnavailable,
                "embedding warmup returned an invalid vector width",
            ));
        }
        #[cfg(all(feature = "accelerator", target_vendor = "apple"))]
        {
            let loaded = self.inner.lock();
            if let Some(coreml) = loaded
                .coreml_buckets
                .iter()
                .find(|program| program.fixed_tokens == COREML_INPUT_TOKENS)
            {
                let warm_text = "retrieval encoder fused execution warmup ".repeat(32);
                let row = tokenize_for(&loaded, EmbeddingPurpose::Passage, &warm_text)?;
                let rows = [row];
                let vectors = embed_coreml_fixed(coreml, &rows)?;
                if vectors.len() != 1 || vectors[0].len() != PINNED_OUTPUT_SIZE {
                    return Err(Error::new(
                        ErrorCode::EmbeddingUnavailable,
                        "Core ML embedding warmup returned an invalid vector shape",
                    ));
                }
            }
        }
        self.warmed.store(true, Ordering::Release);
        tracing::info!(
            elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0,
            "local retrieval encoder warmed"
        );
        Ok(())
    }

    #[must_use]
    pub fn is_warm(&self) -> bool {
        self.warmed.load(Ordering::Acquire)
    }

    fn embed_for(&self, purpose: EmbeddingPurpose, text: &str) -> Result<Vec<f32>> {
        if text.is_empty() || text.len() > MAX_EMBEDDING_TEXT_BYTES {
            return Err(Error::new(
                ErrorCode::ResultBudgetExceeded,
                "embedding text is empty or exceeds its byte budget",
            ));
        }
        #[cfg(feature = "accelerator")]
        {
            // Only queries are cached. A passage is encoded once during indexing and then lives in
            // the durable cache, so an in-memory copy would only evict the queries this is for.
            if purpose == EmbeddingPurpose::Query
                && let Some(hit) = self.query_cache.lock().get(text)
            {
                return Ok(hit);
            }
            let governor = self.governor.read().clone();
            let vector = embed_loaded(&self.inner.lock(), purpose, text, governor.as_ref())?;
            if purpose == EmbeddingPurpose::Query {
                self.query_cache.lock().insert(text, &vector);
            }
            Ok(vector)
        }
        #[cfg(not(feature = "accelerator"))]
        {
            let _ = purpose;
            Err(Error::new(
                ErrorCode::EmbeddingUnavailable,
                "binary was built without the local embedding encoder",
            ))
        }
    }

    /// Actual encoder device selected after construction.
    #[must_use]
    pub const fn execution_device(&self) -> ResolvedComputeDevice {
        self.resolved_device
    }

    #[must_use]
    pub const fn backend_kind(&self) -> BackendKind {
        self.resolved_device.backend
    }

    #[cfg(feature = "accelerator")]
    pub fn candle_device(&self) -> candle_core::Device {
        self.inner.lock().device.clone()
    }

    /// Bytes measured from the persistent encoder tensors resident on the selected device.
    #[must_use]
    pub fn resident_weight_bytes(&self) -> usize {
        #[cfg(feature = "accelerator")]
        {
            let loaded = self.inner.lock();
            let candle = loaded.model.resident_bytes();
            #[cfg(target_vendor = "apple")]
            {
                candle.saturating_add(
                    loaded
                        .coreml_buckets
                        .iter()
                        .map(|program| program.resident_bytes)
                        .fold(0usize, usize::saturating_add),
                )
            }
            #[cfg(not(target_vendor = "apple"))]
            {
                candle
            }
        }
        #[cfg(not(feature = "accelerator"))]
        {
            0
        }
    }

    pub fn bind_memory_governor(&self, governor: DeviceMemoryGovernor) -> Result<()> {
        #[cfg(feature = "accelerator")]
        {
            let mut current = self.governor.write();
            if current
                .as_ref()
                .is_some_and(|bound| !bound.shares_budget_with(&governor))
            {
                return Err(Error::new(
                    ErrorCode::GpuAdmissionFailure,
                    "embedding encoder is already bound to another memory governor",
                ));
            }
            *current = Some(governor);
            Ok(())
        }
        #[cfg(not(feature = "accelerator"))]
        {
            let _ = governor;
            Err(Error::new(
                ErrorCode::EmbeddingUnavailable,
                "binary was built without the local embedding encoder",
            ))
        }
    }
}

/// Publish each completed device batch immediately. A cold graph build is restartable derived
/// work, but successfully encoded content should not be thrown away when a later Metal submission
/// fails, the process is stopped, or the canonical revision changes before index publication.
#[cfg(feature = "accelerator")]
fn cache_completed_embedding_rows(
    cache: Option<&super::embedding_cache::EmbeddingCache>,
    texts: &[String],
    completed: &[usize],
    vectors: &[Option<Vec<f32>>],
) -> Result<()> {
    let Some(cache) = cache else {
        return Ok(());
    };
    let records = completed
        .iter()
        .filter_map(|index| {
            Some((
                super::embedding_cache::EmbeddingCache::key(texts.get(*index)?, PINNED_OUTPUT_SIZE),
                vectors.get(*index)?.as_ref()?.clone(),
            ))
        })
        .collect::<Vec<_>>();
    cache.put(&records)
}

impl TextEmbedding for LocalEmbeddingModel {
    fn profile(&self) -> &EmbeddingProfile {
        self.profile()
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embed_passages(texts)
    }

    fn index_windows(&self, text: &str) -> Result<Vec<String>> {
        #[cfg(feature = "accelerator")]
        {
            let loaded = self.inner.lock();
            rolling_embedding_windows(&loaded, EmbeddingPurpose::Passage, text)
        }
        #[cfg(not(feature = "accelerator"))]
        {
            Ok(vec![text.to_owned()])
        }
    }

    fn query_windows(&self, text: &str) -> Result<Vec<String>> {
        #[cfg(feature = "accelerator")]
        {
            let loaded = self.inner.lock();
            rolling_embedding_windows(&loaded, EmbeddingPurpose::Query, text)
        }
        #[cfg(not(feature = "accelerator"))]
        {
            Ok(vec![text.to_owned()])
        }
    }

    fn embed_batch_cancellable(
        &self,
        texts: &[String],
        is_cancelled: &dyn Fn() -> bool,
    ) -> Result<Vec<Vec<f32>>> {
        self.embed_passages_cancellable(texts, is_cancelled)
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_passage(text)
    }

    fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_query(text)
    }
}

/// Split one canonical value into overlapping accelerated-index windows. The strings are derived
/// encoder inputs only: the semantic row key retains the original graph owner and its window
/// number, and model-visible memory is materialised from that complete owner.
#[cfg(feature = "accelerator")]
fn rolling_embedding_windows(
    loaded: &LoadedEmbeddingModel,
    purpose: EmbeddingPurpose,
    text: &str,
) -> Result<Vec<String>> {
    const WINDOW_CONTENT_TOKENS: usize = 400;
    const WINDOW_OVERLAP_TOKENS: usize = 64;

    if text.is_empty() {
        return Ok(Vec::new());
    }
    let prefix = purpose.prefix();
    let mut prefixed = String::with_capacity(prefix.len().saturating_add(text.len()));
    prefixed.push_str(prefix);
    prefixed.push_str(text);
    let complete = loaded.tokenizer.encode(prefixed, true).map_err(|error| {
        Error::new(
            ErrorCode::EmbeddingUnavailable,
            format!("embedding tokenization failed: {error}"),
        )
    })?;
    if complete.len() <= COREML_INPUT_TOKENS {
        return Ok(vec![text.to_owned()]);
    }

    // Offsets let the derived windows retain byte-exact substrings of the canonical value rather
    // than tokenizer-decoded approximations. The final fit check includes the required `passage: `
    // prefix and special tokens, so every returned row is guaranteed to use the 512-token path.
    let encoding = loaded.tokenizer.encode(text, false).map_err(|error| {
        Error::new(
            ErrorCode::EmbeddingUnavailable,
            format!("embedding tokenization failed: {error}"),
        )
    })?;
    let offsets = encoding.get_offsets();
    if offsets.is_empty() {
        return Ok(vec![text.to_owned()]);
    }
    let mut windows = Vec::new();
    let mut start_token = 0usize;
    while start_token < offsets.len() {
        let mut end_token = (start_token + WINDOW_CONTENT_TOKENS).min(offsets.len());
        let start_byte = if start_token == 0 {
            0
        } else {
            offsets[start_token].0.min(text.len())
        };
        let mut accepted = None;
        while end_token > start_token {
            let end_byte = if end_token == offsets.len() {
                text.len()
            } else {
                offsets[end_token].0.min(text.len())
            };
            if start_byte < end_byte
                && text.is_char_boundary(start_byte)
                && text.is_char_boundary(end_byte)
            {
                let candidate = &text[start_byte..end_byte];
                let mut input = String::with_capacity(prefix.len().saturating_add(candidate.len()));
                input.push_str(prefix);
                input.push_str(candidate);
                let tokens = loaded.tokenizer.encode(input, true).map_err(|error| {
                    Error::new(
                        ErrorCode::EmbeddingUnavailable,
                        format!("embedding tokenization failed: {error}"),
                    )
                })?;
                if tokens.len() <= COREML_INPUT_TOKENS {
                    accepted = Some((end_token, candidate.to_owned()));
                    break;
                }
            }
            end_token -= 1;
        }
        let Some((end_token, window)) = accepted else {
            return Err(Error::new(
                ErrorCode::ResultBudgetExceeded,
                "embedding tokenizer could not produce a bounded rolling window",
            ));
        };
        windows.push(window);
        if end_token == offsets.len() {
            break;
        }
        let next = end_token.saturating_sub(WINDOW_OVERLAP_TOKENS);
        start_token = next.max(start_token + 1);
    }
    Ok(windows)
}

fn validate_artifact_contract(artifacts: &EmbeddingModelArtifacts) -> Result<()> {
    artifacts.profile.validate()?;
    if artifacts.model_exact_bytes == 0
        || artifacts.tokenizer_exact_bytes == 0
        || artifacts.config_exact_bytes == 0
        || artifacts.maximum_input_tokens != PINNED_MAXIMUM_INPUT_TOKENS
        || artifacts.profile.model_hash != artifacts.model_sha256
        || encoder_hash(artifacts.tokenizer_sha256, artifacts.config_sha256)
            != artifacts.profile.tokenizer_hash
        || artifacts.profile.dimension as usize != PINNED_OUTPUT_SIZE
        || artifacts.profile.dtype != EmbeddingDType::F16
        || !artifacts.profile.normalized
        || artifacts.profile.similarity != Similarity::Cosine
    {
        return Err(Error::new(
            ErrorCode::EmbeddingProfileMismatch,
            "embedding artifacts differ from the pinned 384-coordinate FP16 cosine profile",
        ));
    }
    #[cfg(feature = "accelerator")]
    if artifacts.model_sha256 != super::install::DEFAULT_EMBEDDING_MODEL_SHA256
        || artifacts.model_exact_bytes != super::install::DEFAULT_EMBEDDING_MODEL_BYTES
        || artifacts.tokenizer_sha256 != super::install::DEFAULT_EMBEDDING_TOKENIZER_SHA256
        || artifacts.tokenizer_exact_bytes != super::install::DEFAULT_EMBEDDING_TOKENIZER_BYTES
        || artifacts.config_sha256 != super::install::DEFAULT_EMBEDDING_CONFIG_SHA256
        || artifacts.config_exact_bytes != super::install::DEFAULT_EMBEDDING_CONFIG_BYTES
    {
        return Err(Error::new(
            ErrorCode::EmbeddingProfileMismatch,
            "embedding files differ from the pinned encoder revision",
        ));
    }
    Ok(())
}

#[cfg(feature = "accelerator")]
fn read_verified_sha256_exact(
    path: &std::path::Path,
    bytes: u64,
    sha256: [u8; 32],
) -> Result<Vec<u8>> {
    let mut file = irongraph_artifact_loader::open_verified_once(path, bytes, sha256)
        .map_err(embedding_error)?;
    let capacity = usize::try_from(bytes).map_err(|_| {
        Error::new(
            ErrorCode::EmbeddingUnavailable,
            "embedding artifact exceeds this platform's address space",
        )
    })?;
    let mut output = Vec::with_capacity(capacity);
    file.read_to_end(&mut output)
        .map_err(|error| Error::new(ErrorCode::EmbeddingUnavailable, error.to_string()))?;
    if output.len() != capacity {
        return Err(Error::new(
            ErrorCode::EmbeddingUnavailable,
            "embedding artifact changed while it was read",
        ));
    }
    Ok(output)
}

/// Load a locally derived fixed 512-token Core ML program when present. The verified
/// safetensors remain the authority and Candle remains the complete 8,192-token fallback; failure
/// to compile this acceleration artifact therefore costs speed, never availability or semantics.
#[cfg(all(feature = "accelerator", target_vendor = "apple"))]
fn load_coreml_fixed(
    safetensors: &std::path::Path,
    package_name: &str,
    fixed_batch: usize,
    fixed_tokens: usize,
) -> Option<CoreMlEmbeddingProgram> {
    let package = safetensors.parent()?.join(package_name);
    if !package.is_dir() {
        return None;
    }
    let started = std::time::Instant::now();

    // Compile once, then reuse. `MLModel.compileModelAtURL:` writes into `$TMPDIR` and
    // transfers ownership to the caller; the upstream crate documents that it must be
    // copied somewhere permanent. Discarding that path leaked ~2.3GB per program per
    // process start, and because each compile produced a *fresh* path the Neural Engine
    // saw a new bundle identity every time and could never reuse its own cache, so
    // `com.apple.e5rt.e5bundlecache` grew unbounded alongside it.
    //
    // Both buckets contain `Data/com.apple.CoreML/model.mlmodel`, so both compile to
    // `model.mlmodelc` — the destination must be keyed per bucket or they collide.
    // That collision is precisely why Core ML was appending `_<UUID>` disambiguators.
    let cached = coreml_cache_destination(&package);
    let (compiled_path, owns_temp) = match cached {
        Some(destination) if destination.is_dir() => {
            tracing::debug!(path = %destination.display(), "reusing cached Core ML program");
            (destination, false)
        }
        destination => match coreml_native::compile_model(&package) {
            Ok(compiled) => destination
                .and_then(|destination| persist_compiled_model(&compiled, &destination))
                .map_or((compiled, true), |persisted| (persisted, false)),
            Err(error) => {
                tracing::warn!(%error, "fused Core ML retrieval encoder unavailable; using Candle");
                return None;
            }
        },
    };

    match coreml_native::Model::load(&compiled_path, coreml_native::ComputeUnits::All) {
        Ok(model) => {
            tracing::info!(
                fixed_batch,
                fixed_tokens,
                elapsed_ms = started.elapsed().as_secs_f64() * 1_000.0,
                "fused Core ML retrieval encoder loaded"
            );
            Some(CoreMlEmbeddingProgram {
                model,
                // Measure what is actually resident: the compiled program, not the
                // source package it was built from.
                resident_bytes: directory_file_bytes(&compiled_path),
                fixed_batch,
                fixed_tokens,
                compiled_path,
                owns_temp,
            })
        }
        Err(error) => {
            // The load failed, so nothing else will free this. Do it here rather than
            // stranding it — this path is why the leak survived even failed startups.
            if owns_temp && compiled_path.starts_with(std::env::temp_dir()) {
                let _ = std::fs::remove_dir_all(&compiled_path);
            }
            tracing::warn!(%error, "fused Core ML retrieval encoder unavailable; using Candle");
            None
        }
    }
}

/// Where the compiled form of `package` is cached, keyed by the package name so the
/// two fixed-shape buckets cannot overwrite one another.
#[cfg(all(feature = "accelerator", target_vendor = "apple"))]
fn coreml_cache_destination(package: &std::path::Path) -> Option<PathBuf> {
    let parent = package.parent()?;
    let stem = package.file_stem()?.to_str()?;
    if !is_writable_dir(parent) {
        return None;
    }
    Some(parent.join("compiled").join(format!("{stem}.mlmodelc")))
}

#[cfg(all(feature = "accelerator", target_vendor = "apple"))]
fn is_writable_dir(dir: &std::path::Path) -> bool {
    if !dir.is_dir() {
        return false;
    }
    let probe = dir.join(format!(".ig-write-probe-{}", std::process::id()));
    match std::fs::File::create(&probe) {
        Ok(_) => {
            let _ = std::fs::remove_file(&probe);
            true
        }
        Err(_) => false,
    }
}

/// Move `compiled` to `destination`, returning the final path on success.
///
/// Staged through a unique sibling and renamed into place, so two processes racing to
/// warm the cache can never observe a half-written program.
#[cfg(all(feature = "accelerator", target_vendor = "apple"))]
fn persist_compiled_model(
    compiled: &std::path::Path,
    destination: &std::path::Path,
) -> Option<PathBuf> {
    let parent = destination.parent()?;
    std::fs::create_dir_all(parent).ok()?;

    let staging = parent.join(format!(".staging-{}.mlmodelc", std::process::id()));
    let _ = std::fs::remove_dir_all(&staging);

    if std::fs::rename(compiled, &staging).is_err() {
        // Different volume, or otherwise unrenameable: fall back to copying.
        if copy_dir_recursive(compiled, &staging).is_err() {
            let _ = std::fs::remove_dir_all(&staging);
            return None;
        }
        let _ = std::fs::remove_dir_all(compiled);
    }

    match std::fs::rename(&staging, destination) {
        Ok(()) => {
            tracing::info!(path = %destination.display(), "cached compiled Core ML program");
            Some(destination.to_path_buf())
        }
        Err(_) if destination.is_dir() => {
            // Another process won the race; adopt theirs and discard ours.
            let _ = std::fs::remove_dir_all(&staging);
            Some(destination.to_path_buf())
        }
        Err(_) => {
            let _ = std::fs::remove_dir_all(&staging);
            None
        }
    }
}

#[cfg(all(feature = "accelerator", target_vendor = "apple"))]
fn copy_dir_recursive(from: &std::path::Path, to: &std::path::Path) -> std::io::Result<()> {
    std::fs::create_dir_all(to)?;
    for entry in std::fs::read_dir(from)? {
        let entry = entry?;
        let target = to.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir_recursive(&entry.path(), &target)?;
        } else {
            std::fs::copy(entry.path(), &target)?;
        }
    }
    Ok(())
}

/// Remove Core ML compile artifacts stranded in `$TMPDIR` by earlier runs.
///
/// `CoreMlEmbeddingProgram::drop` releases the artifact it owns, but a process killed
/// with `SIGKILL` never runs `Drop`. This is the guarantee behind that mechanism — the
/// the same pairing used for other atomic artifact cleanup. Without
/// it, every crash strands multiple gigabytes permanently.
///
/// Only directories that are genuinely Core ML output and older than `max_age` are
/// removed, so a concurrently starting process cannot lose its in-flight compile.
#[cfg(all(feature = "accelerator", target_vendor = "apple"))]
pub fn sweep_stale_coreml_artifacts(max_age: std::time::Duration) {
    let temp_dir = std::env::temp_dir();
    let Ok(entries) = std::fs::read_dir(&temp_dir) else {
        return;
    };
    let now = std::time::SystemTime::now();
    let mut removed = 0usize;
    let mut reclaimed = 0usize;

    for entry in entries.flatten() {
        let path = entry.path();
        let is_candidate = path.is_dir()
            && path
                .file_name()
                .and_then(|name| name.to_str())
                .is_some_and(|name| name.starts_with("model") && name.ends_with(".mlmodelc"));
        if !is_candidate {
            continue;
        }
        // Confirm it is really a compiled Core ML bundle before removing anything.
        if !path.join("coremldata.bin").exists() && !path.join("model.mil").exists() {
            continue;
        }
        let old_enough = entry
            .metadata()
            .and_then(|metadata| metadata.modified())
            .ok()
            .and_then(|modified| now.duration_since(modified).ok())
            .is_some_and(|age| age >= max_age);
        if !old_enough {
            continue;
        }
        let bytes = directory_file_bytes(&path);
        if std::fs::remove_dir_all(&path).is_ok() {
            removed += 1;
            reclaimed = reclaimed.saturating_add(bytes);
        }
    }

    if removed > 0 {
        tracing::info!(
            removed,
            reclaimed_mb = reclaimed / (1024 * 1024),
            "swept stale Core ML compile artifacts"
        );
    }
}

#[cfg(all(feature = "accelerator", target_vendor = "apple"))]
fn directory_file_bytes(path: &std::path::Path) -> usize {
    let Ok(entries) = std::fs::read_dir(path) else {
        return 0;
    };
    entries
        .filter_map(std::result::Result::ok)
        .map(|entry| {
            entry.metadata().map_or(0, |metadata| {
                if metadata.is_dir() {
                    directory_file_bytes(&entry.path())
                } else {
                    usize::try_from(metadata.len()).unwrap_or(usize::MAX)
                }
            })
        })
        .fold(0usize, usize::saturating_add)
}

#[cfg(feature = "accelerator")]
fn embed_loaded(
    loaded: &LoadedEmbeddingModel,
    purpose: EmbeddingPurpose,
    text: &str,
    governor: Option<&DeviceMemoryGovernor>,
) -> Result<Vec<f32>> {
    let mut input = String::with_capacity(purpose.prefix().len().saturating_add(text.len()));
    input.push_str(purpose.prefix());
    input.push_str(text);
    let encoding = loaded.tokenizer.encode(input, true).map_err(|error| {
        Error::new(
            ErrorCode::EmbeddingUnavailable,
            format!("embedding tokenization failed: {error}"),
        )
    })?;
    let bounded = bound_embedding_input(
        encoding.get_ids(),
        encoding.get_attention_mask(),
        loaded.maximum_input_tokens,
    )?;
    let ids = bounded.ids.as_ref();
    let attention = bounded.attention.as_ref();
    if attention.iter().all(|value| *value == 0) {
        return Err(Error::new(
            ErrorCode::ResultBudgetExceeded,
            "embedding token sequence has an empty attention mask",
        ));
    }
    #[cfg(target_vendor = "apple")]
    if let Some(coreml) = loaded
        .coreml_buckets
        .iter()
        .find(|program| ids.len() <= program.fixed_tokens)
    {
        let rows = [(ids.to_vec(), attention.to_vec())];
        return embed_coreml_fixed(coreml, &rows).and_then(|mut vectors| {
            vectors.pop().ok_or_else(|| {
                Error::new(
                    ErrorCode::EmbeddingUnavailable,
                    "Core ML embedding returned no query vector",
                )
            })
        });
    }
    let _activation = governor
        .map(|governor| {
            embedding_activation_bytes(ids.len(), &loaded.device)
                .and_then(|bytes| governor.reserve_shared(SharedMemoryClass::EncoderState, bytes))
        })
        .transpose()?;
    // One Candle Metal device is not safe to drive from two threads: its buffer pools and its
    // `MTLResidencySet` are mutated without a covering lock, and every readback sweeps and frees
    // pooled buffers that another thread's in-flight command buffer may still reference. The
    // encoder mutex is released between batches so queries can preempt, and the semantic index
    // runs its own tensor work on this same device, so the mutex alone does not make the device
    // single-threaded. See `irongraph_gpu::metal_gate`.
    let _gate = irongraph_gpu::lock_metal_device(&loaded.device);
    let input = candle_core::Tensor::from_slice(ids, (1, ids.len()), &loaded.device)
        .map_err(embedding_error)?;
    let hidden = loaded.model.forward(&input).map_err(embedding_error)?;
    pool_truncate_normalize(&hidden, attention, PINNED_OUTPUT_SIZE)
}

/// Tokenise one text for the encoder, bounded to its input limit.
#[cfg(feature = "accelerator")]
fn tokenize_for(
    loaded: &LoadedEmbeddingModel,
    purpose: EmbeddingPurpose,
    text: &str,
) -> Result<(Vec<u32>, Vec<u32>)> {
    let mut input = String::with_capacity(purpose.prefix().len().saturating_add(text.len()));
    input.push_str(purpose.prefix());
    input.push_str(text);
    let encoding = loaded.tokenizer.encode(input, true).map_err(|error| {
        Error::new(
            ErrorCode::EmbeddingUnavailable,
            format!("embedding tokenization failed: {error}"),
        )
    })?;
    let bounded = bound_embedding_input(
        encoding.get_ids(),
        encoding.get_attention_mask(),
        loaded.maximum_input_tokens,
    )?;
    if bounded.attention.iter().all(|value| *value == 0) {
        return Err(Error::new(
            ErrorCode::ResultBudgetExceeded,
            "embedding token sequence has an empty attention mask",
        ));
    }
    Ok((bounded.ids.to_vec(), bounded.attention.to_vec()))
}

/// Run one gathered batch and scatter its vectors back to their original positions.
#[cfg(feature = "accelerator")]
fn flush_embedding_batch(
    inner: &Mutex<LoadedEmbeddingModel>,
    batch: &mut Vec<(usize, (Vec<u32>, Vec<u32>))>,
    governor: Option<&DeviceMemoryGovernor>,
    out: &mut [Option<Vec<f32>>],
    is_cancelled: &dyn Fn() -> bool,
) -> Result<()> {
    if batch.is_empty() {
        return Ok(());
    }
    ensure_embedding_not_cancelled(is_cancelled)?;
    let rows: Vec<(Vec<u32>, Vec<u32>)> = batch.iter().map(|(_, row)| row.clone()).collect();
    // Retain the mutex only for one bounded device forward. This lets a query run between
    // maintenance batches on the shared local encoder.
    let vectors = {
        let loaded = inner.lock();
        ensure_embedding_not_cancelled(is_cancelled)?;
        embed_batch_loaded(&loaded, &rows, governor)?
    };
    for ((index, _), vector) in batch.drain(..).zip(vectors) {
        out[index] = Some(vector);
    }
    Ok(())
}

#[cfg(feature = "accelerator")]
fn ensure_embedding_not_cancelled(is_cancelled: &dyn Fn() -> bool) -> Result<()> {
    if is_cancelled() {
        return Err(Error::new(
            ErrorCode::Cancelled,
            "embedding operation was cancelled for interactive work",
        ));
    }
    Ok(())
}

/// Embed many texts in one forward pass.
///
/// The encoder was called once per text: tokenise, one row of shape `(1, len)` through the model,
/// pool, repeat. A forward pass has a large fixed cost that a single short sequence barely uses, so
/// a thousand of them cost a thousand times that overhead — two minutes to index a small mailbox,
/// and growing with it.
///
/// Batched, the same work is a handful of passes. Sequences are padded to the longest in the batch
/// and each row is pooled against its own attention mask, so padding contributes nothing to any
/// vector: the results are identical to the one-at-a-time path, which the tests assert directly.
#[cfg(feature = "accelerator")]
fn embed_batch_loaded(
    loaded: &LoadedEmbeddingModel,
    rows: &[(Vec<u32>, Vec<u32>)],
    governor: Option<&DeviceMemoryGovernor>,
) -> Result<Vec<Vec<f32>>> {
    if rows.is_empty() {
        return Ok(Vec::new());
    }
    #[cfg(target_vendor = "apple")]
    if let Some(widest) = rows.iter().map(|(ids, _)| ids.len()).max()
        && let Some(coreml) = loaded
            .coreml_buckets
            .iter()
            .find(|program| widest <= program.fixed_tokens)
        && rows
            .iter()
            .all(|(ids, attention)| ids.len() == attention.len() && !ids.is_empty())
    {
        return embed_coreml_fixed(coreml, rows);
    }
    let width = rows
        .iter()
        .map(|(ids, _)| ids.len())
        .max()
        .unwrap_or_default();
    let batch = rows.len();
    // Reserved for the padded rectangle, which is what the device actually holds.
    let _activation = governor
        .map(|governor| {
            embedding_activation_bytes(width.saturating_mul(batch), &loaded.device)
                .and_then(|bytes| governor.reserve_shared(SharedMemoryClass::EncoderState, bytes))
        })
        .transpose()?;

    let mut flat_ids = Vec::with_capacity(batch * width);
    for (ids, _) in rows {
        flat_ids.extend_from_slice(ids);
        // Padding is masked out below, so the value only has to be a token the model can embed.
        flat_ids.resize(flat_ids.len() + (width - ids.len()), 0);
    }
    // One Candle Metal device is not safe to drive from two threads: its buffer pools and its
    // `MTLResidencySet` are mutated without a covering lock, and every readback sweeps and frees
    // pooled buffers that another thread's in-flight command buffer may still reference. The
    // encoder mutex is released between batches so queries can preempt, and the semantic index
    // runs its own tensor work on this same device, so the mutex alone does not make the device
    // single-threaded. See `irongraph_gpu::metal_gate`.
    let _gate = irongraph_gpu::lock_metal_device(&loaded.device);
    let input = candle_core::Tensor::from_slice(&flat_ids, (batch, width), &loaded.device)
        .map_err(embedding_error)?;
    let hidden = loaded.model.forward(&input).map_err(embedding_error)?;

    pool_batch_truncate_normalize(&hidden, rows, PINNED_OUTPUT_SIZE)
}

/// Execute a fused fixed-shape program in its native batch size. A partial final batch repeats its
/// last real input in the unused slots; those outputs are discarded. Repetition keeps every model
/// row valid without inventing an empty attention mask or changing a real row's vector.
#[cfg(all(feature = "accelerator", target_vendor = "apple"))]
fn embed_coreml_fixed(
    program: &CoreMlEmbeddingProgram,
    rows: &[(Vec<u32>, Vec<u32>)],
) -> Result<Vec<Vec<f32>>> {
    use coreml_native::{AsMultiArray, BorrowedTensor};

    let mut output = Vec::with_capacity(rows.len());
    if program.fixed_batch == 0 {
        return Err(Error::new(
            ErrorCode::EmbeddingUnavailable,
            "Core ML embedding program has an empty batch",
        ));
    }
    for chunk in rows.chunks(program.fixed_batch) {
        let mut padded_ids = vec![0_i32; program.fixed_batch * program.fixed_tokens];
        let mut padded_attention = vec![0_i32; program.fixed_batch * program.fixed_tokens];
        for slot in 0..program.fixed_batch {
            let (ids, attention) = &chunk[slot.min(chunk.len() - 1)];
            if ids.len() != attention.len() || ids.is_empty() || ids.len() > program.fixed_tokens {
                return Err(Error::new(
                    ErrorCode::EmbeddingUnavailable,
                    "embedding row does not fit the fixed Core ML input",
                ));
            }
            let start = slot * program.fixed_tokens;
            for (target, value) in padded_ids[start..].iter_mut().zip(ids) {
                *target = i32::try_from(*value).map_err(|_| {
                    Error::new(
                        ErrorCode::EmbeddingUnavailable,
                        "embedding token exceeds Core ML i32",
                    )
                })?;
            }
            for (target, value) in padded_attention[start..].iter_mut().zip(attention) {
                *target = i32::try_from(*value).map_err(|_| {
                    Error::new(
                        ErrorCode::EmbeddingUnavailable,
                        "embedding attention value exceeds Core ML i32",
                    )
                })?;
            }
        }
        let shape = [program.fixed_batch, program.fixed_tokens];
        let input_ids = BorrowedTensor::from_i32(&padded_ids, &shape)
            .map_err(|error| Error::new(ErrorCode::EmbeddingUnavailable, error.to_string()))?;
        let attention_mask = BorrowedTensor::from_i32(&padded_attention, &shape)
            .map_err(|error| Error::new(ErrorCode::EmbeddingUnavailable, error.to_string()))?;
        let inputs: [(&str, &dyn AsMultiArray); 2] = [
            ("input_ids", &input_ids),
            ("attention_mask", &attention_mask),
        ];
        let prediction = program
            .model
            .predict(&inputs)
            .map_err(|error| Error::new(ErrorCode::EmbeddingUnavailable, error.to_string()))?;
        let (vectors, output_shape) = prediction
            .get_f32("embeddings")
            .map_err(|error| Error::new(ErrorCode::EmbeddingUnavailable, error.to_string()))?;
        let expected = program.fixed_batch.saturating_mul(PINNED_OUTPUT_SIZE);
        if vectors.len() != expected
            || (output_shape.as_slice() != [program.fixed_batch, PINNED_OUTPUT_SIZE]
                && !(program.fixed_batch == 1 && output_shape.as_slice() == [PINNED_OUTPUT_SIZE]))
        {
            return Err(Error::new(
                ErrorCode::EmbeddingUnavailable,
                format!("Core ML embedding returned unexpected shape {output_shape:?}"),
            ));
        }
        for vector in vectors.chunks_exact(PINNED_OUTPUT_SIZE).take(chunk.len()) {
            let mut vector = vector.to_vec();
            normalize(&mut vector)?;
            output.push(vector);
        }
    }
    Ok(output)
}

/// Pool a padded encoder batch with one device reduction and one device-to-host transfer.
///
/// The previous implementation reused the single-row helper once per text. Although the encoder
/// itself was batched, every helper invocation submitted a separate mask/reduction command graph
/// and then synchronously copied one vector to the host. A 64-row batch therefore paid 64 Metal
/// synchronization boundaries. Keeping the padded mask rectangular lets the device pool all rows
/// together and makes the readback cost once per encoder batch instead.
#[cfg(feature = "accelerator")]
fn pool_batch_truncate_normalize(
    hidden: &candle_core::Tensor,
    rows: &[(Vec<u32>, Vec<u32>)],
    output_width: usize,
) -> Result<Vec<Vec<f32>>> {
    use candle_core::{DType, Tensor};

    let (batch, sequence_length, hidden_width) = hidden.dims3().map_err(embedding_error)?;
    if batch != rows.len()
        || hidden_width != PINNED_HIDDEN_SIZE
        || output_width == 0
        || output_width > hidden_width
    {
        return Err(Error::new(
            ErrorCode::EmbeddingUnavailable,
            "embedding encoder returned an invalid batched hidden-state shape",
        ));
    }
    let mut flat_mask = Vec::with_capacity(batch.saturating_mul(sequence_length));
    let mut inverse_counts = Vec::with_capacity(batch);
    for (ids, attention) in rows {
        if ids.len() != attention.len() || ids.len() > sequence_length {
            return Err(Error::new(
                ErrorCode::EmbeddingUnavailable,
                "embedding tokenizer produced an invalid batched attention mask",
            ));
        }
        let count = attention.iter().try_fold(0_u32, |sum, value| {
            sum.checked_add(*value).ok_or_else(|| {
                Error::new(
                    ErrorCode::EmbeddingUnavailable,
                    "attention count overflowed",
                )
            })
        })?;
        if count == 0 {
            return Err(Error::new(
                ErrorCode::EmbeddingUnavailable,
                "embedding tokenizer produced an empty attention mask",
            ));
        }
        flat_mask.extend_from_slice(attention);
        flat_mask.resize(flat_mask.len() + sequence_length - attention.len(), 0);
        inverse_counts.push(1.0_f32 / count as f32);
    }
    let mask = Tensor::from_slice(&flat_mask, (batch, sequence_length, 1), hidden.device())
        .and_then(|value| value.to_dtype(DType::F32))
        .map_err(embedding_error)?;
    let inverse_counts = Tensor::from_slice(&inverse_counts, (batch, 1), hidden.device())
        .map_err(embedding_error)?;
    let pooled = hidden
        .to_dtype(DType::F32)
        .and_then(|value| value.broadcast_mul(&mask))
        .and_then(|value| value.sum(1))
        .and_then(|value| value.broadcast_mul(&inverse_counts))
        .map_err(embedding_error)?;
    let mut vectors = pooled.to_vec2::<f32>().map_err(embedding_error)?;
    for vector in &mut vectors {
        normalize(vector)?;
        vector.truncate(output_width);
        normalize(vector)?;
        if vector.len() != output_width || vector.iter().any(|coordinate| !coordinate.is_finite()) {
            return Err(Error::new(
                ErrorCode::EmbeddingUnavailable,
                "embedding encoder returned non-finite or incomplete coordinates",
            ));
        }
    }
    Ok(vectors)
}

#[cfg(feature = "accelerator")]
fn bound_embedding_input<'a>(
    ids: &'a [u32],
    attention: &'a [u32],
    maximum_input_tokens: usize,
) -> Result<BoundedEmbeddingInput<'a>> {
    if ids.is_empty() || attention.len() != ids.len() || maximum_input_tokens == 0 {
        return Err(Error::new(
            ErrorCode::ResultBudgetExceeded,
            "embedding token sequence is empty or malformed",
        ));
    }
    if ids.len() <= maximum_input_tokens {
        return Ok(BoundedEmbeddingInput {
            ids: Cow::Borrowed(ids),
            attention: Cow::Borrowed(attention),
        });
    }

    // The pinned tokenizer places special tokens and the exact query/passage prefix at the head.
    // Keep that small header and spend the remaining budget on the newest semantic content.
    let head_tokens = PRESERVED_INPUT_HEAD_TOKENS.min(maximum_input_tokens.saturating_sub(1));
    let tail_tokens = maximum_input_tokens - head_tokens;
    let tail_start = ids.len() - tail_tokens;
    let mut bounded_ids = Vec::with_capacity(maximum_input_tokens);
    bounded_ids.extend_from_slice(&ids[..head_tokens]);
    bounded_ids.extend_from_slice(&ids[tail_start..]);
    let mut bounded_attention = Vec::with_capacity(maximum_input_tokens);
    bounded_attention.extend_from_slice(&attention[..head_tokens]);
    bounded_attention.extend_from_slice(&attention[tail_start..]);

    Ok(BoundedEmbeddingInput {
        ids: Cow::Owned(bounded_ids),
        attention: Cow::Owned(bounded_attention),
    })
}

#[cfg(feature = "accelerator")]
fn embedding_activation_bytes(tokens: usize, device: &candle_core::Device) -> Result<usize> {
    let hidden = tokens
        .checked_mul(PINNED_HIDDEN_SIZE)
        .and_then(|values| values.checked_mul(12))
        .ok_or_else(|| {
            Error::new(
                ErrorCode::GpuAdmissionFailure,
                "embedding hidden-state admission estimate overflowed",
            )
        })?;
    let feed_forward = tokens
        .checked_mul(8_192)
        .and_then(|values| values.checked_mul(6))
        .ok_or_else(|| {
            Error::new(
                ErrorCode::GpuAdmissionFailure,
                "embedding feed-forward admission estimate overflowed",
            )
        })?;
    let attention = if device.is_cpu() || device.is_metal() {
        tokens
            .checked_mul(PINNED_HIDDEN_SIZE + 1_024)
            .and_then(|values| values.checked_mul(6))
    } else {
        32_usize
            .checked_mul(tokens)
            .and_then(|values| values.checked_mul(tokens))
            .and_then(|values| values.checked_mul(std::mem::size_of::<f32>()))
            .and_then(|scores| {
                tokens
                    .checked_mul(PINNED_HIDDEN_SIZE + 1_024)
                    .and_then(|values| values.checked_mul(6))
                    .and_then(|values| values.checked_add(scores))
            })
    }
    .ok_or_else(|| {
        Error::new(
            ErrorCode::GpuAdmissionFailure,
            "embedding attention admission estimate overflowed",
        )
    })?;
    hidden
        .checked_add(feed_forward)
        .and_then(|bytes| bytes.checked_add(attention))
        .and_then(|bytes| bytes.checked_add(8 * 1024 * 1024))
        .ok_or_else(|| {
            Error::new(
                ErrorCode::GpuAdmissionFailure,
                "embedding activation admission estimate overflowed",
            )
        })
}

#[cfg(feature = "accelerator")]
fn pool_truncate_normalize(
    hidden: &candle_core::Tensor,
    attention: &[u32],
    output_width: usize,
) -> Result<Vec<f32>> {
    use candle_core::{DType, Tensor};

    let (batch, sequence_length, hidden_width) = hidden.dims3().map_err(embedding_error)?;
    if batch != 1
        || sequence_length != attention.len()
        || hidden_width != PINNED_HIDDEN_SIZE
        || output_width == 0
        || output_width > hidden_width
    {
        return Err(Error::new(
            ErrorCode::EmbeddingUnavailable,
            "embedding encoder returned an invalid hidden-state shape",
        ));
    }
    let count = attention.iter().try_fold(0_u32, |sum, value| {
        sum.checked_add(*value).ok_or_else(|| {
            Error::new(
                ErrorCode::EmbeddingUnavailable,
                "attention count overflowed",
            )
        })
    })?;
    if count == 0 {
        return Err(Error::new(
            ErrorCode::EmbeddingUnavailable,
            "embedding tokenizer produced an empty attention mask",
        ));
    }
    let mask = Tensor::from_slice(attention, (1, sequence_length, 1), hidden.device())
        .and_then(|value| value.to_dtype(DType::F32))
        .map_err(embedding_error)?;
    let pooled = hidden
        .to_dtype(DType::F32)
        .and_then(|value| value.broadcast_mul(&mask))
        .and_then(|value| value.sum(1))
        .and_then(|value| value.affine(1.0 / f64::from(count), 0.0))
        .map_err(embedding_error)?;
    let mut rows = pooled.to_vec2::<f32>().map_err(embedding_error)?;
    let mut vector = rows.pop().ok_or_else(|| {
        Error::new(
            ErrorCode::EmbeddingUnavailable,
            "embedding encoder returned no pooled vector",
        )
    })?;
    normalize(&mut vector)?;
    vector.truncate(output_width);
    normalize(&mut vector)?;
    if vector.len() != output_width || vector.iter().any(|coordinate| !coordinate.is_finite()) {
        return Err(Error::new(
            ErrorCode::EmbeddingUnavailable,
            "embedding encoder returned non-finite or incomplete coordinates",
        ));
    }
    Ok(vector)
}

#[cfg(feature = "accelerator")]
fn normalize(vector: &mut [f32]) -> Result<()> {
    let norm = vector
        .iter()
        .fold(0.0_f64, |sum, value| {
            let coordinate = f64::from(*value);
            sum + coordinate * coordinate
        })
        .sqrt();
    if !norm.is_finite() || norm <= f64::EPSILON {
        return Err(Error::new(
            ErrorCode::EmbeddingUnavailable,
            "embedding vector has zero or invalid norm",
        ));
    }
    for coordinate in vector {
        *coordinate = (f64::from(*coordinate) / norm) as f32;
    }
    Ok(())
}

pub(super) fn encoder_hash(tokenizer_hash: [u8; 32], config_hash: [u8; 32]) -> [u8; 32] {
    let mut hasher = blake3::Hasher::new();
    hasher.update(b"irongraph-embedding-encoder-v1");
    hasher.update(&tokenizer_hash);
    hasher.update(&config_hash);
    *hasher.finalize().as_bytes()
}

#[cfg(all(test, feature = "accelerator"))]
mod tests {
    use super::*;

    #[test]
    fn pooling_uses_only_visible_tokens_then_applies_matryoshka_normalization() -> crate::Result<()>
    {
        let mut values = vec![0.0_f32; 3 * PINNED_HIDDEN_SIZE];
        values[0] = 3.0;
        values[1] = 4.0;
        values[PINNED_HIDDEN_SIZE] = 3.0;
        values[PINNED_HIDDEN_SIZE + 1] = 4.0;
        values[2 * PINNED_HIDDEN_SIZE] = 1_000.0;
        let hidden = candle_core::Tensor::from_vec(
            values,
            (1, 3, PINNED_HIDDEN_SIZE),
            &candle_core::Device::Cpu,
        )
        .map_err(embedding_error)?;
        let vector = pool_truncate_normalize(&hidden, &[1, 1, 0], PINNED_OUTPUT_SIZE)?;
        assert_eq!(vector.len(), PINNED_OUTPUT_SIZE);
        assert!((vector[0] - 0.6).abs() < 1e-6);
        assert!((vector[1] - 0.8).abs() < 1e-6);
        let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn batched_pooling_matches_independent_rows_with_padding() -> crate::Result<()> {
        let sequence = 3;
        let mut values = vec![0.0_f32; 2 * sequence * PINNED_HIDDEN_SIZE];
        for row in 0..2 {
            for token in 0..sequence {
                for coordinate in 0..PINNED_OUTPUT_SIZE {
                    values[(row * sequence + token) * PINNED_HIDDEN_SIZE + coordinate] =
                        (row * 17 + token * 5 + coordinate + 1) as f32;
                }
            }
        }
        let hidden = candle_core::Tensor::from_vec(
            values,
            (2, sequence, PINNED_HIDDEN_SIZE),
            &candle_core::Device::Cpu,
        )
        .map_err(embedding_error)?;
        let rows = vec![(vec![1, 2], vec![1, 1]), (vec![3, 4, 5], vec![1, 0, 1])];
        let batched = pool_batch_truncate_normalize(&hidden, &rows, PINNED_OUTPUT_SIZE)?;
        for (row, (_, attention)) in rows.iter().enumerate() {
            let independent = pool_truncate_normalize(
                &hidden
                    .narrow(0, row, 1)
                    .and_then(|value| value.narrow(1, 0, attention.len()))
                    .map_err(embedding_error)?,
                attention,
                PINNED_OUTPUT_SIZE,
            )?;
            assert_eq!(batched[row].len(), independent.len());
            assert!(
                batched[row]
                    .iter()
                    .zip(independent)
                    .all(|(left, right)| (left - right).abs() < 1e-6)
            );
        }
        Ok(())
    }

    #[test]
    fn query_and_passage_prefixes_are_exact_and_distinct() {
        assert_eq!(EmbeddingPurpose::Query.prefix(), "query: ");
        assert_eq!(EmbeddingPurpose::Passage.prefix(), "passage: ");
    }

    #[test]
    fn embedding_input_at_or_below_limit_is_preserved_without_copying() -> crate::Result<()> {
        let ids = [128_000, 2_000, 25, 220, 91, 128_001];
        let attention = [1, 1, 1, 1, 1, 1];
        let bounded = bound_embedding_input(&ids, &attention, ids.len())?;

        assert!(matches!(bounded.ids, Cow::Borrowed(_)));
        assert!(matches!(bounded.attention, Cow::Borrowed(_)));
        assert_eq!(bounded.ids.as_ref(), ids);
        assert_eq!(bounded.attention.as_ref(), attention);
        Ok(())
    }

    #[test]
    fn oversized_embedding_input_preserves_exact_head_tail_shape_and_mask() -> crate::Result<()> {
        const LIMIT: usize = PRESERVED_INPUT_HEAD_TOKENS + 4;
        let ids = (10_000..10_040).collect::<Vec<_>>();
        let attention = (0..ids.len())
            .map(|index| u32::from(index % 3 != 0))
            .collect::<Vec<_>>();
        let bounded = bound_embedding_input(&ids, &attention, LIMIT)?;

        assert_eq!(bounded.ids.len(), LIMIT);
        assert_eq!(bounded.attention.len(), LIMIT);
        assert_eq!(
            bounded.ids.as_ref(),
            [
                ids[..PRESERVED_INPUT_HEAD_TOKENS].as_ref(),
                ids[36..].as_ref()
            ]
            .concat()
        );
        assert_eq!(
            bounded.attention.as_ref(),
            [
                attention[..PRESERVED_INPUT_HEAD_TOKENS].as_ref(),
                attention[36..].as_ref(),
            ]
            .concat()
        );
        Ok(())
    }

    #[test]
    #[ignore = "loads and executes the full pinned embedding encoder on the selected acceptance device"]
    fn installed_embedding_model_executes_real_forward() -> crate::Result<()> {
        let artifacts = super::super::install::ensure_default_embedding_model()?;
        let device = match std::env::var("IRONGRAPH_REAL_EMBEDDING_DEVICE").as_deref() {
            Ok("cpu") => EmbeddingDevice::Cpu,
            Ok("metal") => EmbeddingDevice::Metal(0),
            Ok("cuda") => EmbeddingDevice::Cuda(0),
            _ => EmbeddingDevice::Auto,
        };
        let model = LocalEmbeddingModel::load(artifacts, device)?;
        let query = model.embed_query("durable graph memory")?;
        let passage = model.embed_passage("durable graph memory")?;
        assert_eq!(query.len(), PINNED_OUTPUT_SIZE);
        assert_eq!(passage.len(), PINNED_OUTPUT_SIZE);
        assert_ne!(query, passage);
        Ok(())
    }
}

#[cfg(all(test, feature = "accelerator"))]
mod throughput {
    /// What the encoder actually achieves, batched against one at a time.
    ///
    /// Ignored by default: it loads the real encoder and takes tens of seconds. Run it with
    /// `cargo test --release -- --ignored encoder_throughput --nocapture` when changing the
    /// batching, because the only honest way to know whether a change helped is to measure it on
    /// the device rather than reason about the shape of the loop.
    #[test]
    #[ignore = "loads the real encoder"]
    fn encoder_throughput() {
        use super::{LocalEmbeddingModel, PINNED_OUTPUT_SIZE};

        let Ok(artifacts) = crate::ensure_default_embedding_model() else {
            eprintln!("encoder is not installed; nothing to measure");
            return;
        };
        let device = match std::env::var("IRONGRAPH_REAL_EMBEDDING_DEVICE").as_deref() {
            Ok("cpu") => crate::EmbeddingDevice::Cpu,
            Ok("metal") => crate::EmbeddingDevice::Metal(0),
            _ => crate::EmbeddingDevice::Auto,
        };
        let model = LocalEmbeddingModel::load(artifacts, device).expect("encoder loads");
        model.warm_up().expect("encoder warms");
        let rows = std::env::var("IRONGRAPH_EMBED_BENCH_ROWS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(256);
        let words = std::env::var("IRONGRAPH_EMBED_BENCH_WORDS")
            .ok()
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(24);
        let nonce = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("clock")
            .as_nanos();

        // Texts shaped like graph rows: a sentence naming a thing and describing it.
        let texts: Vec<String> = (0..rows)
            .map(|index| {
                format!(
                    "Benchmark {nonce} row {index}. {}",
                    "The record contains factual account details, dates, entities, values, and follow-up context. "
                        .repeat(words.div_ceil(12))
                )
            })
            .collect();

        let started = std::time::Instant::now();
        let one_at_a_time: Vec<_> = texts
            .iter()
            .take(4)
            .map(|text| model.embed_passage(text).expect("single embed"))
            .collect();
        let single = started.elapsed();

        let started = std::time::Instant::now();
        let batched = model.embed_passages(&texts).expect("batched embed");
        let batch = started.elapsed();

        assert_eq!(one_at_a_time.len(), rows.min(4));
        assert_eq!(batched.len(), texts.len());
        assert!(batched.iter().all(|row| row.len() == PINNED_OUTPUT_SIZE));

        let per_single = single.as_secs_f64() / one_at_a_time.len().max(1) as f64;
        let per_batched = batch.as_secs_f64() / texts.len() as f64;
        eprintln!(
            "device={:?} rows={rows} words={words} single: {per_single:.4}s/text   \
             batched: {per_batched:.4}s/text   speedup: {:.1}x",
            model.execution_device(),
            per_single / per_batched.max(f64::EPSILON)
        );

        #[cfg(target_vendor = "apple")]
        if std::env::var_os("IRONGRAPH_EMBED_COMPARE_CANDLE").is_some() {
            let candle_texts = texts
                .iter()
                .enumerate()
                .map(|(index, text)| format!("Candle comparison {nonce}-{index}: {text}"))
                .collect::<Vec<_>>();
            let coreml_buckets = {
                let mut loaded = model.inner.lock();
                std::mem::take(&mut loaded.coreml_buckets)
            };
            let started = std::time::Instant::now();
            let candle = model
                .embed_passages(&candle_texts)
                .expect("Candle batched embed");
            let candle_elapsed = started.elapsed();
            {
                let mut loaded = model.inner.lock();
                loaded.coreml_buckets = coreml_buckets;
            }
            assert_eq!(candle.len(), candle_texts.len());
            eprintln!(
                "Candle comparison: {:.4}s/text total={:.3}s",
                candle_elapsed.as_secs_f64() / candle_texts.len().max(1) as f64,
                candle_elapsed.as_secs_f64()
            );
        }

        #[cfg(target_vendor = "apple")]
        if std::env::var_os("IRONGRAPH_EMBED_COMPARE_B8").is_some() {
            let b8_texts = texts
                .iter()
                .enumerate()
                .map(|(index, text)| format!("Core ML batch-eight {nonce}-{index}: {text}"))
                .collect::<Vec<_>>();
            let defaults = {
                let mut loaded = model.inner.lock();
                std::mem::take(&mut loaded.coreml_buckets)
            };
            let b8 = super::load_coreml_fixed(
                &model.artifacts.model_safetensors,
                "model-512-b8.mlpackage",
                8,
                512,
            );
            model.inner.lock().coreml_buckets = b8.into_iter().collect();
            let started = std::time::Instant::now();
            let vectors = model.embed_passages(&b8_texts).expect("Core ML b8 embed");
            let elapsed = started.elapsed();
            model.inner.lock().coreml_buckets = defaults;
            assert_eq!(vectors.len(), b8_texts.len());
            eprintln!(
                "Core ML b8 comparison: {:.4}s/text total={:.3}s",
                elapsed.as_secs_f64() / b8_texts.len().max(1) as f64,
                elapsed.as_secs_f64()
            );
        }

        #[cfg(target_vendor = "apple")]
        if std::env::var_os("IRONGRAPH_EMBED_COMPARE_64").is_some() {
            let (token_rows, baseline) = {
                let loaded = model.inner.lock();
                let token_rows = texts
                    .iter()
                    .map(|text| {
                        super::tokenize_for(&loaded, super::EmbeddingPurpose::Passage, text)
                            .expect("tokenize comparison row")
                    })
                    .collect::<Vec<_>>();
                assert!(token_rows.iter().all(|(ids, _)| ids.len() <= 64));
                let baseline = super::embed_coreml_fixed(
                    loaded
                        .coreml_buckets
                        .iter()
                        .find(|program| program.fixed_tokens == 512)
                        .expect("512-token program"),
                    &token_rows,
                )
                .expect("512-token baseline");
                (token_rows, baseline)
            };
            let started = std::time::Instant::now();
            let vectors = {
                let loaded = model.inner.lock();
                super::embed_coreml_fixed(
                    loaded
                        .coreml_buckets
                        .iter()
                        .find(|program| program.fixed_tokens == 64)
                        .expect("64-token program"),
                    &token_rows,
                )
                .expect("64-token bucket")
            };
            let elapsed = started.elapsed();
            assert_eq!(vectors.len(), texts.len());
            for (expected, actual) in baseline.iter().zip(&vectors) {
                let cosine = expected
                    .iter()
                    .zip(actual)
                    .map(|(left, right)| left * right)
                    .sum::<f32>();
                assert!(cosine > 0.999, "bucket changed embedding: {cosine}");
            }
            eprintln!(
                "Core ML 64-token comparison: {:.4}s/text total={:.3}s",
                elapsed.as_secs_f64() / texts.len().max(1) as f64,
                elapsed.as_secs_f64()
            );
        }
    }
}
