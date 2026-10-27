use std::{
    path::PathBuf,
    sync::atomic::{AtomicBool, Ordering},
};

#[cfg(feature = "accelerator")]
use std::borrow::Cow;

#[cfg(feature = "accelerator")]
use parking_lot::{Mutex, RwLock};

use crate::{
    Error, ErrorCode, Result,
    execution::TextEmbedding,
    gpu::{BackendKind, DeviceMemoryGovernor, ResolvedComputeDevice},
    graph::EmbeddingProfile,
};

#[cfg(feature = "accelerator")]
use crate::gpu::SharedMemoryClass;

use super::EmbeddingDevice;
#[cfg(feature = "accelerator")]
use super::{
    device,
    encoder::{Encoder, EncoderConfig},
};

#[cfg(feature = "accelerator")]
pub(super) const EMBEDDING_HIDDEN_SIZE: usize = 384;
/// Texts per forward pass while indexing. Large enough to amortise the call, small enough that
/// the padded rectangle stays modest when one text in the batch is long.
#[cfg(feature = "accelerator")]
const EMBEDDING_BATCH: usize = 16;

/// Maximum padded token rectangle per submission; queries can interleave between batches.
#[cfg(feature = "accelerator")]
const EMBEDDING_BATCH_TOKENS: usize = 1_024;

// Rolling windows use the same bound on every embedding execution backend.
#[cfg(feature = "accelerator")]
const EMBEDDING_WINDOW_TOKENS: usize = 512;

pub(super) const EMBEDDING_OUTPUT_SIZE: usize = 384;
pub(super) const EMBEDDING_MAXIMUM_INPUT_TOKENS: usize = 8_192;
#[cfg(feature = "accelerator")]
const PRESERVED_INPUT_HEAD_TOKENS: usize = 1;

#[cfg(feature = "accelerator")]
fn embedding_error(error: candle_core::Error) -> Error {
    Error::new(
        ErrorCode::EmbeddingUnavailable,
        format!("local text embedding failed: {error}"),
    )
}

/// Locally cached files for the one supported text encoder.
#[derive(Clone, Debug)]
pub struct EmbeddingModelArtifacts {
    pub model_safetensors: PathBuf,
    pub tokenizer_json: PathBuf,
    pub model_config_json: PathBuf,
    pub profile: EmbeddingProfile,
    pub maximum_input_tokens: usize,
}

/// Local text encoder shared by interactive retrieval and background embedding jobs.
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
    /// cache hit avoids inference on the request path.
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
    model: Encoder,
    tokenizer: tokenizers::Tokenizer,
    device: candle_core::Device,
    maximum_input_tokens: usize,
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

impl LocalEmbeddingModel {
    pub fn load(artifacts: EmbeddingModelArtifacts, device: EmbeddingDevice) -> Result<Self> {
        Self::load_with_storage(artifacts, device, &crate::EmbeddingStorage::default())
    }

    pub fn load_with_storage(
        artifacts: EmbeddingModelArtifacts,
        device: EmbeddingDevice,
        storage: &crate::EmbeddingStorage,
    ) -> Result<Self> {
        validate_artifact_contract(&artifacts)?;
        let cache_directory = storage.cache_directory()?;

        #[cfg(feature = "accelerator")]
        {
            let config_bytes = std::fs::read(&artifacts.model_config_json).map_err(|error| {
                Error::new(
                    ErrorCode::EmbeddingUnavailable,
                    format!("reading embedding configuration failed: {error}"),
                )
            })?;
            let config: EncoderConfig = serde_json::from_slice(&config_bytes).map_err(|error| {
                Error::new(
                    ErrorCode::EmbeddingUnavailable,
                    format!("embedding model config is invalid: {error}"),
                )
            })?;
            config.validate(artifacts.maximum_input_tokens)?;
            let tokenizer_bytes = std::fs::read(&artifacts.tokenizer_json).map_err(|error| {
                Error::new(
                    ErrorCode::EmbeddingUnavailable,
                    format!("reading embedding tokenizer failed: {error}"),
                )
            })?;
            let mut tokenizer =
                tokenizers::Tokenizer::from_bytes(&tokenizer_bytes).map_err(|error| {
                    Error::new(
                        ErrorCode::EmbeddingUnavailable,
                        format!("embedding tokenizer is invalid: {error}"),
                    )
                })?;
            if tokenizer.get_vocab_size(true) != config.vocabulary_size() {
                return Err(Error::new(
                    ErrorCode::EmbeddingProfileMismatch,
                    "embedding tokenizer vocabulary does not match Bekko a8m",
                ));
            }
            tokenizer.with_padding(None);
            tokenizer.with_truncation(None).map_err(|error| {
                Error::new(
                    ErrorCode::EmbeddingUnavailable,
                    format!("configuring embedding tokenizer failed: {error}"),
                )
            })?;
            let selected = device::select_device(device)?;
            let selected_gpu = selected.identity.backend != BackendKind::Cpu;
            let loaded = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                load_on_device(&artifacts, &config, tokenizer.clone(), selected)
            }))
            .unwrap_or_else(|_| {
                Err(Error::new(
                    ErrorCode::EmbeddingUnavailable,
                    "embedding device initialization panicked",
                ))
            });
            let (loaded, resolved_device) = match loaded {
                Ok(loaded) => loaded,
                Err(error) if device == EmbeddingDevice::Auto && selected_gpu => {
                    tracing::warn!(%error, "GPU embedding initialization failed; using CPU");
                    load_on_device(
                        &artifacts,
                        &config,
                        tokenizer,
                        device::select_device(EmbeddingDevice::Cpu)?,
                    )?
                }
                Err(error) => return Err(error),
            };
            tracing::info!(model = super::install::DEFAULT_EMBEDDING_MODEL_ID, backend = ?resolved_device.backend,
                ordinal = resolved_device.ordinal, "local embedding encoder selected");
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
                inner: Mutex::new(loaded),
                governor: RwLock::new(None),
                cache: super::embedding_cache::EmbeddingCache::open_in(
                    &cache_directory,
                    &artifacts_profile_hash,
                    EMBEDDING_OUTPUT_SIZE,
                ),
                query_cache: Mutex::new(QueryEmbeddingCache::default()),
            })
        }
        #[cfg(not(feature = "accelerator"))]
        {
            let _ = (device, cache_directory);
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

    /// Encodes raw query text for semantic retrieval.
    pub fn embed_query(&self, text: &str) -> Result<Vec<f32>> {
        self.embed_for(EmbeddingPurpose::Query, text)
    }

    /// Encodes raw graph/document text for indexing.
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
            if text.is_empty() {
                return Err(Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "embedding text is empty",
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
                        EMBEDDING_OUTPUT_SIZE,
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
                    rows.push((index, tokenize_for(&loaded, text)?));
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
        if vector.len() != EMBEDDING_OUTPUT_SIZE {
            return Err(Error::new(
                ErrorCode::EmbeddingUnavailable,
                "embedding warmup returned an invalid vector width",
            ));
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
        if text.is_empty() {
            return Err(Error::new(
                ErrorCode::ResultBudgetExceeded,
                "embedding text is empty",
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
            let vector = embed_loaded(&self.inner.lock(), text, governor.as_ref())?;
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
            self.inner.lock().model.resident_bytes()
        }
        #[cfg(not(feature = "accelerator"))]
        {
            0
        }
    }

    pub fn bind_memory_governor(&self, governor: DeviceMemoryGovernor) -> Result<()> {
        #[cfg(feature = "accelerator")]
        {
            // Embedding admission is independent of graph execution. In particular, a CPU graph
            // process never constructs a Metal graph backend to establish the encoder's limit.
            #[cfg(any(target_os = "macos", target_os = "ios"))]
            if self.resolved_device.backend == BackendKind::Metal {
                let mut governor = governor;
                governor.set_host_available_probe(std::sync::Arc::new(
                    crate::gpu::host_available_bytes,
                ));
                let device = self.candle_device();
                let metal = device.as_metal_device().map_err(|error| {
                    Error::new(ErrorCode::EmbeddingUnavailable, error.to_string())
                })?;
                let recommended = metal.metal_device().recommended_max_working_set_size();
                governor.cap_limit(recommended)?;
                return self.install_memory_governor(governor);
            }
            self.install_memory_governor(governor)
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

    #[cfg(feature = "accelerator")]
    fn install_memory_governor(&self, governor: DeviceMemoryGovernor) -> Result<()> {
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
                super::embedding_cache::EmbeddingCache::key(
                    texts.get(*index)?,
                    EMBEDDING_OUTPUT_SIZE,
                ),
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

    fn supports_prefix_dimensions(&self, dimensions: usize) -> bool {
        self.artifacts.profile.dimension == EMBEDDING_OUTPUT_SIZE as u32
            && super::install::default_profile()
                .is_ok_and(|profile| profile.profile_hash == self.artifacts.profile.profile_hash)
            && matches!(dimensions, 64 | 128 | 256)
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.embed_passages(texts)
    }

    fn index_windows(&self, text: &str) -> Result<Vec<String>> {
        #[cfg(feature = "accelerator")]
        {
            let loaded = self.inner.lock();
            rolling_embedding_windows(&loaded, text)
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
            rolling_embedding_windows(&loaded, text)
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
fn rolling_embedding_windows(loaded: &LoadedEmbeddingModel, text: &str) -> Result<Vec<String>> {
    const WINDOW_CONTENT_TOKENS: usize = 400;
    const WINDOW_OVERLAP_TOKENS: usize = 64;

    if text.is_empty() {
        return Ok(Vec::new());
    }
    let complete = loaded.tokenizer.encode(text, true).map_err(|error| {
        Error::new(
            ErrorCode::EmbeddingUnavailable,
            format!("embedding tokenization failed: {error}"),
        )
    })?;
    if complete.len() <= EMBEDDING_WINDOW_TOKENS {
        return Ok(vec![text.to_owned()]);
    }

    // Byte offsets preserve source text exactly. The final fit check includes special tokens.
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
                let tokens = loaded.tokenizer.encode(candidate, true).map_err(|error| {
                    Error::new(
                        ErrorCode::EmbeddingUnavailable,
                        format!("embedding tokenization failed: {error}"),
                    )
                })?;
                if tokens.len() <= EMBEDDING_WINDOW_TOKENS {
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
    if artifacts.maximum_input_tokens != EMBEDDING_MAXIMUM_INPUT_TOKENS
        || artifacts.profile != super::install::default_profile()?
    {
        return Err(Error::new(
            ErrorCode::EmbeddingProfileMismatch,
            "embedding profile is not the Bekko a8m 384-coordinate cosine profile",
        ));
    }
    Ok(())
}

#[cfg(feature = "accelerator")]
fn load_on_device(
    artifacts: &EmbeddingModelArtifacts,
    config: &EncoderConfig,
    tokenizer: tokenizers::Tokenizer,
    selected: crate::gpu::ResolvedCandleDevice,
) -> Result<(LoadedEmbeddingModel, ResolvedComputeDevice)> {
    let _gate = irongraph_gpu::lock_metal_device(&selected.device);
    let builder = irongraph_artifact_loader::mmap_safetensors(
        &artifacts.model_safetensors,
        selected.dtype,
        &selected.device,
    )
    .map_err(embedding_error)?;
    let model = Encoder::load(builder, config).map_err(embedding_error)?;
    // Resolve kernel/dtype availability before advertising a usable GPU. Read back to synchronize.
    let input = tokenizer
        .encode("embedding device warmup", true)
        .map_err(|error| Error::new(ErrorCode::EmbeddingUnavailable, error.to_string()))?;
    let ids = candle_core::Tensor::from_slice(input.get_ids(), (1, input.len()), &selected.device)
        .map_err(embedding_error)?;
    let attention = ids.ones_like().map_err(embedding_error)?;
    let hidden = model.forward(&ids, &attention).map_err(embedding_error)?;
    pool_truncate_normalize(&hidden, input.get_attention_mask(), EMBEDDING_OUTPUT_SIZE)?;
    Ok((
        LoadedEmbeddingModel {
            model,
            tokenizer,
            device: selected.device,
            maximum_input_tokens: artifacts.maximum_input_tokens,
        },
        selected.identity,
    ))
}

#[cfg(feature = "accelerator")]
fn embed_loaded(
    loaded: &LoadedEmbeddingModel,
    text: &str,
    governor: Option<&DeviceMemoryGovernor>,
) -> Result<Vec<f32>> {
    let encoding = loaded.tokenizer.encode(text, true).map_err(|error| {
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
    let _activation = governor
        .map(|governor| {
            embedding_activation_bytes(1, ids.len(), &loaded.device)
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
    let mask = candle_core::Tensor::from_slice(attention, (1, ids.len()), &loaded.device)
        .map_err(embedding_error)?;
    let hidden = loaded
        .model
        .forward(&input, &mask)
        .map_err(embedding_error)?;
    pool_truncate_normalize(&hidden, attention, EMBEDDING_OUTPUT_SIZE)
}

/// Tokenise one text for the encoder, bounded to its input limit.
#[cfg(feature = "accelerator")]
fn tokenize_for(loaded: &LoadedEmbeddingModel, text: &str) -> Result<(Vec<u32>, Vec<u32>)> {
    let encoding = loaded.tokenizer.encode(text, true).map_err(|error| {
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
    let width = rows
        .iter()
        .map(|(ids, _)| ids.len())
        .max()
        .unwrap_or_default();
    let batch = rows.len();
    // Reserved for the padded rectangle, which is what the device actually holds.
    let _activation = governor
        .map(|governor| {
            embedding_activation_bytes(batch, width, &loaded.device)
                .and_then(|bytes| governor.reserve_shared(SharedMemoryClass::EncoderState, bytes))
        })
        .transpose()?;

    let mut flat_ids = Vec::with_capacity(batch * width);
    let mut flat_attention = Vec::with_capacity(batch * width);
    for (ids, attention) in rows {
        flat_attention.extend_from_slice(attention);
        flat_attention.resize(flat_attention.len() + width - attention.len(), 0);
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
    let mask = candle_core::Tensor::from_slice(&flat_attention, (batch, width), &loaded.device)
        .map_err(embedding_error)?;
    let hidden = loaded
        .model
        .forward(&input, &mask)
        .map_err(embedding_error)?;

    pool_batch_truncate_normalize(&hidden, rows, EMBEDDING_OUTPUT_SIZE)
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
    if batch != rows.len() || hidden_width != EMBEDDING_HIDDEN_SIZE || output_width != hidden_width
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

    // Preserve the leading special token and the newest semantic content for direct calls.
    // Automatic indexing uses overlapping windows and retains the complete source.
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
fn embedding_activation_bytes(
    batch: usize,
    sequence: usize,
    device: &candle_core::Device,
) -> Result<usize> {
    let bytes_per_value = if device.is_cpu() { 4usize } else { 2 };
    let estimate = || -> Option<usize> {
        let tokens = batch.checked_mul(sequence)?;
        let hidden = tokens
            .checked_mul(EMBEDDING_HIDDEN_SIZE)?
            .checked_mul(12)?
            .checked_mul(bytes_per_value)?;
        let mlp = tokens
            .checked_mul(1152)?
            .checked_mul(6)?
            .checked_mul(bytes_per_value)?;
        // QK scores and FP32 softmax coexist; local masks also scale with padded sequence area.
        let scores = batch
            .checked_mul(6)?
            .checked_mul(sequence)?
            .checked_mul(sequence)?
            .checked_mul(12)?;
        let masks = batch
            .checked_mul(sequence)?
            .checked_mul(sequence)?
            .checked_mul(8)?;
        hidden
            .checked_add(mlp)?
            .checked_add(scores)?
            .checked_add(masks)?
            .checked_add(8 * 1024 * 1024)
    };
    estimate().ok_or_else(|| {
        Error::new(
            ErrorCode::GpuAdmissionFailure,
            "embedding activation estimate overflowed",
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
        || hidden_width != EMBEDDING_HIDDEN_SIZE
        || output_width != hidden_width
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

#[cfg(all(test, feature = "accelerator"))]
mod tests {
    use super::*;

    #[test]
    #[ignore = "copies and loads the installed real encoder in two application directories"]
    fn application_paths_real_encoder_uses_selected_model_and_cache() -> crate::Result<()> {
        let root = tempfile::tempdir()?;
        let source = crate::EmbeddingStorage::default()
            .model_directory()?
            .join("hotchpotch-bekko-embedding-v1-a8m")
            .join(crate::DEFAULT_EMBEDDING_REVISION);
        let mut first = None;
        for app in ["app-a", "app-b"] {
            let storage = crate::EmbeddingStorage {
                model_dir: Some(root.path().join(app).join("models")),
                cache_dir: Some(root.path().join(app).join("cache")),
            };
            let model_directory = storage
                .model_directory()?
                .join("hotchpotch-bekko-embedding-v1-a8m")
                .join(crate::DEFAULT_EMBEDDING_REVISION);
            std::fs::create_dir_all(&model_directory)?;
            for file in [
                crate::DEFAULT_EMBEDDING_MODEL_FILE,
                crate::DEFAULT_EMBEDDING_CONFIG_FILE,
                crate::DEFAULT_EMBEDDING_TOKENIZER_FILE,
            ] {
                std::fs::copy(source.join(file), model_directory.join(file))?;
            }
            let artifacts = crate::ensure_default_embedding_model_with_storage(
                &storage,
                &crate::ModelInstallProgress::default(),
            )?;
            assert!(
                artifacts
                    .model_safetensors
                    .starts_with(storage.model_directory()?)
            );
            let encoder = LocalEmbeddingModel::load_with_storage(
                artifacts.clone(),
                EmbeddingDevice::Cpu,
                &storage,
            )?;
            encoder.warm_up()?;
            let texts = vec!["Application-owned embedding storage preserves complete searchable document content.".to_owned()];
            let vector = encoder.embed_passages(&texts)?.remove(0);
            assert_eq!(vector.len(), EMBEDDING_OUTPUT_SIZE);
            assert!(vector.iter().all(|value| value.is_finite()));
            assert!(encoder.cache.is_some(), "selected cache opens");
            let cached = std::fs::read_dir(storage.cache_directory()?)?
                .collect::<std::io::Result<Vec<_>>>()?;
            assert_eq!(cached.len(), 1);
            assert_eq!(
                cached[0]
                    .path()
                    .extension()
                    .and_then(|value| value.to_str()),
                Some("vectors")
            );
            assert!(cached[0].metadata()?.len() > 0);
            if let Some(first) = &first {
                assert_eq!(&vector, first);
            } else {
                first = Some(vector.clone());
            }
            drop(encoder);
            let reopened =
                LocalEmbeddingModel::load_with_storage(artifacts, EmbeddingDevice::Cpu, &storage)?;
            assert_eq!(reopened.embed_passages(&texts)?.remove(0), vector);
        }
        Ok(())
    }

    #[test]
    fn pooling_uses_only_visible_tokens_then_normalizes() -> crate::Result<()> {
        let mut values = vec![0.0_f32; 3 * EMBEDDING_HIDDEN_SIZE];
        values[0] = 3.0;
        values[1] = 4.0;
        values[EMBEDDING_HIDDEN_SIZE] = 3.0;
        values[EMBEDDING_HIDDEN_SIZE + 1] = 4.0;
        values[2 * EMBEDDING_HIDDEN_SIZE] = 1_000.0;
        let hidden = candle_core::Tensor::from_vec(
            values,
            (1, 3, EMBEDDING_HIDDEN_SIZE),
            &candle_core::Device::Cpu,
        )
        .map_err(embedding_error)?;
        let vector = pool_truncate_normalize(&hidden, &[1, 1, 0], EMBEDDING_OUTPUT_SIZE)?;
        assert_eq!(vector.len(), EMBEDDING_OUTPUT_SIZE);
        assert!((vector[0] - 0.6).abs() < 1e-6);
        assert!((vector[1] - 0.8).abs() < 1e-6);
        let norm = vector.iter().map(|value| value * value).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        Ok(())
    }

    #[test]
    fn batched_pooling_matches_independent_rows_with_padding() -> crate::Result<()> {
        let sequence = 3;
        let mut values = vec![0.0_f32; 2 * sequence * EMBEDDING_HIDDEN_SIZE];
        for row in 0..2 {
            for token in 0..sequence {
                for coordinate in 0..EMBEDDING_OUTPUT_SIZE {
                    values[(row * sequence + token) * EMBEDDING_HIDDEN_SIZE + coordinate] =
                        (row * 17 + token * 5 + coordinate + 1) as f32;
                }
            }
        }
        let hidden = candle_core::Tensor::from_vec(
            values,
            (2, sequence, EMBEDDING_HIDDEN_SIZE),
            &candle_core::Device::Cpu,
        )
        .map_err(embedding_error)?;
        let rows = vec![(vec![1, 2], vec![1, 1]), (vec![3, 4, 5], vec![1, 0, 1])];
        let batched = pool_batch_truncate_normalize(&hidden, &rows, EMBEDDING_OUTPUT_SIZE)?;
        for (row, (_, attention)) in rows.iter().enumerate() {
            let independent = pool_truncate_normalize(
                &hidden
                    .narrow(0, row, 1)
                    .and_then(|value| value.narrow(1, 0, attention.len()))
                    .map_err(embedding_error)?,
                attention,
                EMBEDDING_OUTPUT_SIZE,
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
        assert_eq!(query.len(), EMBEDDING_OUTPUT_SIZE);
        assert_eq!(passage.len(), EMBEDDING_OUTPUT_SIZE);
        assert_eq!(query, passage);
        Ok(())
    }

    #[test]
    #[ignore = "downloads and executes Bekko on the selected real device"]
    fn bekko_reference_vectors_and_batch_padding_match() -> crate::Result<()> {
        let artifacts = super::super::install::ensure_default_embedding_model()?;
        let requested = match std::env::var("IRONGRAPH_REAL_EMBEDDING_DEVICE").as_deref() {
            Ok("cpu") => EmbeddingDevice::Cpu,
            Ok("metal") => EmbeddingDevice::Metal(0),
            Ok("cuda") => EmbeddingDevice::Cuda(0),
            _ => EmbeddingDevice::Auto,
        };
        let mut model = LocalEmbeddingModel::load(artifacts, requested)?;
        // Every device must execute inference; a persistent CPU cache must not mask GPU defects.
        model.cache = None;
        model.warm_up()?;
        assert!(model.is_warm());
        match requested {
            EmbeddingDevice::Cpu => assert_eq!(model.backend_kind(), BackendKind::Cpu),
            EmbeddingDevice::Metal(_) => assert_eq!(model.backend_kind(), BackendKind::Metal),
            EmbeddingDevice::Cuda(_) => assert_eq!(model.backend_kind(), BackendKind::Cuda),
            EmbeddingDevice::Auto => {
                #[cfg(any(target_os = "macos", target_os = "ios"))]
                assert_eq!(model.backend_kind(), BackendKind::Metal);
            }
        }
        let fixture: serde_json::Value =
            serde_json::from_str(include_str!("../tests/fixtures/bekko_reference.json"))
                .map_err(|error| Error::new(ErrorCode::Internal, error.to_string()))?;
        assert_eq!(
            fixture["model"],
            super::super::install::DEFAULT_EMBEDDING_MODEL_ID
        );
        assert_eq!(
            fixture["revision"],
            super::super::install::DEFAULT_EMBEDDING_REVISION
        );
        let texts: Vec<String> = serde_json::from_value(fixture["texts"].clone())
            .map_err(|error| Error::new(ErrorCode::Internal, error.to_string()))?;
        let expected: Vec<Vec<f32>> = serde_json::from_value(fixture["vectors"].clone())
            .map_err(|error| Error::new(ErrorCode::Internal, error.to_string()))?;
        let actual = model.embed_passages(&texts)?;
        assert_eq!(actual.len(), texts.len());
        for ((text, vector), reference) in texts.iter().zip(&actual).zip(&expected) {
            assert_eq!(vector.len(), EMBEDDING_OUTPUT_SIZE);
            assert!(vector.iter().all(|coordinate| coordinate.is_finite()));
            let norm = vector
                .iter()
                .map(|coordinate| coordinate * coordinate)
                .sum::<f32>();
            assert!((norm - 1.).abs() < 1e-5);
            let cosine = vector
                .iter()
                .zip(reference)
                .map(|(left, right)| left * right)
                .sum::<f32>();
            assert!(cosine > 0.999, "reference cosine {cosine}: {text}");
            let single = model.embed_passage(text)?;
            let cosine = vector
                .iter()
                .zip(&single)
                .map(|(left, right)| left * right)
                .sum::<f32>();
            assert!(cosine > 0.9999, "single/batch cosine {cosine}: {text}");
            assert_eq!(single, model.embed_query(text)?);
        }
        let repeated = vec![texts[0].clone(), texts[1].clone(), texts[0].clone()];
        let deduplicated = model.embed_passages(&repeated)?;
        assert_eq!(deduplicated[0], deduplicated[2]);
        let uneven = model.embed_passages(&[texts[0].clone(), texts[4].clone()])?;
        for (vector, reference) in uneven.iter().zip([&expected[0], &expected[4]]) {
            let cosine = vector
                .iter()
                .zip(reference)
                .map(|(left, right)| left * right)
                .sum::<f32>();
            assert!(cosine > 0.999, "heavily padded batch cosine {cosine}");
        }
        assert_eq!(
            model
                .embed_passages_cancellable(&repeated, &|| true)
                .unwrap_err()
                .code,
            ErrorCode::Cancelled
        );
        let source = "Árvíztűrő tükörfúrógép: complete graph document context. ".repeat(2000);
        let windows = model.index_windows(&source)?;
        assert!(windows.len() > 1);
        let loaded = model.inner.lock();
        for window in &windows {
            assert!(source.contains(window));
            assert!(
                loaded
                    .tokenizer
                    .encode(window.as_str(), true)
                    .unwrap()
                    .len()
                    <= EMBEDDING_WINDOW_TOKENS
            );
        }
        assert!(source.starts_with(&windows[0]));
        assert!(source.ends_with(windows.last().unwrap()));
        drop(loaded);
        assert_eq!(model.embed_passages(&windows[..2])?.len(), 2);
        eprintln!(
            "BEKKO_REAL_DEVICE_PASSED backend={:?} vectors={} dimension={} resident_bytes={}",
            model.backend_kind(),
            actual.len(),
            EMBEDDING_OUTPUT_SIZE,
            model.resident_weight_bytes()
        );
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
        use super::{EMBEDDING_OUTPUT_SIZE, LocalEmbeddingModel};

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
        assert!(batched.iter().all(|row| row.len() == EMBEDDING_OUTPUT_SIZE));

        let per_single = single.as_secs_f64() / one_at_a_time.len().max(1) as f64;
        let per_batched = batch.as_secs_f64() / texts.len() as f64;
        eprintln!(
            "device={:?} rows={rows} words={words} single: {per_single:.4}s/text   \
             batched: {per_batched:.4}s/text   speedup: {:.1}x",
            model.execution_device(),
            per_single / per_batched.max(f64::EPSILON)
        );
    }
}
