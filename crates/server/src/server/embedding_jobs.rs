//! Disk-backed owner catch-up for asynchronous local embedding inference.
use crate::{
    Error, ErrorCode, ProjectId, Result,
    execution::TextEmbedding,
    graph::ResolvedVectorMutation,
    types::{EntityKind, PropertyId},
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use tokio::sync::{Mutex, Notify};
use tokio_util::sync::CancellationToken;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct EmbeddingOwner {
    pub project: ProjectId,
    pub kind: EntityKind,
    pub entity_id: u64,
}
impl Ord for EmbeddingOwner {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.project, self.kind as u8, self.entity_id).cmp(&(
            other.project,
            other.kind as u8,
            other.entity_id,
        ))
    }
}
impl PartialOrd for EmbeddingOwner {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct EmbeddingJob {
    pub owner: EmbeddingOwner,
    pub revision: u64,
}

/// Worker-only canonical input; the durable cursor retains no source text or graph snapshots.
pub(super) struct SemanticTextChunk {
    pub property: PropertyId,
    pub text: Option<Arc<str>>,
}
pub(super) struct EmbeddingWork {
    pub encoder: Arc<dyn TextEmbedding>,
    pub chunks: Vec<SemanticTextChunk>,
    pub is_current: Option<Arc<dyn Fn() -> bool + Send + Sync>>,
}

/// A cursor owns only its iteration position and canonical shared handles, never all owner text.
pub(super) type EmbeddingSeed = Box<dyn Iterator<Item = Result<EmbeddingJob>> + Send>;
pub(super) type InitializeProject =
    dyn Fn(ProjectId, &CancellationToken) -> Result<Option<EmbeddingSeed>> + Send + Sync;
pub(super) type LoadOwner =
    dyn Fn(EmbeddingJob, &CancellationToken) -> Result<Option<EmbeddingWork>> + Send + Sync;
pub(super) type PublishVectors = dyn Fn(EmbeddingJob, Vec<ResolvedVectorMutation>, &CancellationToken) -> Result<bool>
    + Send
    + Sync;
pub(super) type ReportFailure = dyn Fn(EmbeddingJob, &Error) + Send + Sync;

pub(super) struct EmbeddingCallbacks {
    /// Return None for an absent/stale owner. Acquire canonical values here, in the CPU worker.
    pub load: Arc<LoadOwner>,
    /// Must check the canonical owner revision and publish under the same mutation boundary.
    /// Returning false means stale work was discarded; a check followed by an unfenced commit
    /// does not satisfy this callback's contract.
    pub publish: Arc<PublishVectors>,
    pub failed: Arc<ReportFailure>,
}

/// Pulls one owner from canonical recovery or durable pending-owner records.
pub(super) type NextCommittedOwner =
    dyn Fn(&CancellationToken) -> Result<Option<EmbeddingJob>> + Send + Sync;

pub(super) struct DurableEmbeddingWorker {
    cancellation: CancellationToken,
    wake: Arc<Notify>,
    wake_pending: Arc<AtomicBool>,
    worker: Mutex<Option<tokio::task::JoinHandle<()>>>,
    idle: Arc<AtomicBool>,
}

impl Drop for DurableEmbeddingWorker {
    fn drop(&mut self) {
        self.cancellation.cancel();
    }
}

impl DurableEmbeddingWorker {
    pub fn start(next: Arc<NextCommittedOwner>, callbacks: EmbeddingCallbacks) -> Result<Self> {
        let runtime = tokio::runtime::Handle::try_current()
            .map_err(|_| Error::internal("embedding worker requires an async runtime"))?;
        let cancellation = CancellationToken::new();
        let cancel = cancellation.clone();
        let callbacks = Arc::new(callbacks);
        let idle = Arc::new(AtomicBool::new(false));
        let worker_idle = Arc::clone(&idle);
        let wake = Arc::new(Notify::new());
        let worker_wake = Arc::clone(&wake);
        let wake_pending = Arc::new(AtomicBool::new(false));
        let worker_wake_pending = Arc::clone(&wake_pending);
        let worker = runtime.spawn(async move {
            while !cancel.is_cancelled() {
                // Register before reading durable work so a commit racing the empty check
                // cannot be lost between that check and the idle wait.
                let notified = worker_wake.notified();
                tokio::pin!(notified);
                notified.as_mut().enable();
                worker_idle.store(false, Ordering::Release);
                let source = Arc::clone(&next);
                let token = cancel.clone();
                let next = tokio::task::spawn_blocking(move || source(&token)).await;
                match next {
                    Ok(Ok(Some(job))) => {
                        if !process_job(job, Arc::clone(&callbacks), 16, cancel.clone()).await {
                            break;
                        }
                    }
                    Ok(Ok(None)) => {
                        // A durable insertion may sort before the current cursor while
                        // inference runs. Finish this pass, then revisit its beginning.
                        if worker_wake_pending.swap(false, Ordering::AcqRel) {
                            continue;
                        }
                        worker_idle.store(true, Ordering::Release);
                        tokio::select! {
                            _ = cancel.cancelled() => break,
                            _ = notified => {}
                        }
                    }
                    Ok(Err(error)) => {
                        tracing::error!(code = ?error.code, message = %error.message, "embedding owner catch-up failed");
                        tokio::select! {
                            _ = cancel.cancelled() => break,
                            _ = tokio::time::sleep(std::time::Duration::from_millis(100)) => {}
                        }
                    }
                    Err(error) => {
                        tracing::error!(%error, "embedding owner reader worker terminated");
                        break;
                    }
                }
            }
        });
        Ok(Self {
            cancellation,
            wake,
            wake_pending,
            worker: Mutex::new(Some(worker)),
            idle,
        })
    }

    /// Call only after pending owner records become durable.
    pub fn wake(&self) {
        self.wake_pending.store(true, Ordering::Release);
        self.wake.notify_one();
    }

    pub async fn shutdown(&self) -> Result<()> {
        self.idle.store(false, Ordering::Release);
        self.cancellation.cancel();
        if let Some(worker) = self.worker.lock().await.take() {
            worker
                .await
                .map_err(|error| Error::internal(format!("embedding worker failed: {error}")))?;
        }
        Ok(())
    }

    #[cfg(test)]
    pub fn is_idle(&self) -> bool {
        self.idle.load(Ordering::Acquire)
    }
}

async fn process_job(
    job: EmbeddingJob,
    callbacks: Arc<EmbeddingCallbacks>,
    batch_size: usize,
    cancellation: CancellationToken,
) -> bool {
    let mut delay_ms = 10_u64;
    loop {
        if cancellation.is_cancelled() {
            return true;
        }
        let cb = Arc::clone(&callbacks);
        let cancel = cancellation.clone();
        let execution = tokio::task::spawn_blocking(move || {
            let result = (|| {
                let Some(work) = (cb.load)(job, &cancel)? else {
                    return Ok(false);
                };
                let obsolete = || {
                    cancel.is_cancelled()
                        || work.is_current.as_ref().is_some_and(|current| !current())
                };
                let vectors = encode(&work, job, batch_size, &obsolete)?;
                if obsolete() {
                    return Err(cancelled());
                }
                (cb.publish)(job, vectors, &cancel)
            })();
            if let Err(error) = &result
                && error.code != ErrorCode::Cancelled
                && !cancel.is_cancelled()
            {
                (cb.failed)(job, error);
            }
            result
        })
        .await;
        match execution {
            Err(error) => {
                tracing::error!(%error, "embedding CPU worker terminated unexpectedly");
                cancellation.cancel();
                return false;
            }
            Ok(Ok(_)) => return true,
            Ok(Err(error)) if error.code == ErrorCode::Cancelled => return true,
            Ok(Err(_)) => {
                tokio::select! {
                    _ = cancellation.cancelled() => return true,
                    _ = tokio::time::sleep(std::time::Duration::from_millis(delay_ms)) => {},
                }
                delay_ms = delay_ms.saturating_mul(2).min(1000);
            }
        }
    }
}

fn cancelled() -> Error {
    Error::new(ErrorCode::Cancelled, "embedding work was cancelled")
}

fn encode(
    work: &EmbeddingWork,
    job: EmbeddingJob,
    batch_size: usize,
    is_cancelled: &dyn Fn() -> bool,
) -> Result<Vec<ResolvedVectorMutation>> {
    if batch_size == 0 {
        return Err(Error::internal(
            "embedding inference batch must be nonempty",
        ));
    }
    let profile = work.encoder.profile();
    profile.validate()?;
    let dimension = usize::try_from(profile.dimension)
        .map_err(|_| Error::internal("embedding dimension exceeds process range"))?;
    let quantizer =
        crate::graph::VectorIndex::new_with_dtype(dimension, profile.similarity, profile.dtype)?;
    let mut mutations = Vec::with_capacity(work.chunks.len());
    for chunk in &work.chunks {
        if is_cancelled() {
            return Err(cancelled());
        }
        let Some(text) = chunk.text.as_ref() else {
            mutations.push(ResolvedVectorMutation::Remove {
                property: chunk.property,
                entity_id: job.owner.entity_id,
                revision: job.revision,
            });
            continue;
        };
        let mut windows = work.encoder.index_windows(text)?;
        if windows.is_empty() {
            windows.push(text.to_string());
        }
        // Keep each window's meaning. These vectors belong to the original owner/property;
        // they neither create graph records nor require a second summary inference pass.
        // Only one owner's derived output is retained until its revision-fenced publication.
        let mut coordinates = Vec::with_capacity(windows.len());
        for batch in windows.chunks(batch_size) {
            if is_cancelled() {
                return Err(cancelled());
            }
            let encoded = work.encoder.embed_batch_cancellable(batch, is_cancelled)?;
            if encoded.len() != batch.len()
                || encoded.iter().any(|vector| vector.len() != dimension)
            {
                return Err(Error::new(
                    ErrorCode::EmbeddingProfileMismatch,
                    "embedding output shape differs from active profile",
                ));
            }
            for mut vector in encoded {
                if is_cancelled() {
                    return Err(cancelled());
                }
                if vector.iter().any(|value| !value.is_finite()) {
                    return Err(Error::new(
                        ErrorCode::EmbeddingProfileMismatch,
                        "embedding output contains a non-finite coordinate",
                    ));
                }
                if profile.normalized {
                    let norm = vector
                        .iter()
                        .map(|value| f64::from(*value).powi(2))
                        .sum::<f64>()
                        .sqrt();
                    if norm == 0.0 {
                        return Err(Error::new(
                            ErrorCode::EmbeddingProfileMismatch,
                            "normalized embedding output has zero norm",
                        ));
                    }
                    for value in &mut vector {
                        *value = (f64::from(*value) / norm) as f32;
                    }
                }
                coordinates.push(quantizer.quantize(&vector)?);
            }
        }
        // A one-window owner also uses the span format. Besides retaining its exact vector,
        // this marks generated data as using span semantics, so recovery can distinguish it
        // from older averaged vectors without re-encoding current owners on every restart.
        mutations.push(ResolvedVectorMutation::UpsertSpans {
            property: chunk.property,
            entity_id: job.owner.entity_id,
            coordinates,
            revision: job.revision,
        });
    }
    Ok(mutations)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{EmbeddingDType, EmbeddingProfile, Similarity};
    use std::sync::atomic::{AtomicBool, AtomicUsize};

    struct Encoder {
        profile: EmbeddingProfile,
        entered: AtomicBool,
        release: AtomicBool,
        calls: AtomicUsize,
        largest_batch: AtomicUsize,
    }
    impl Encoder {
        fn new(block: bool) -> Result<Self> {
            Ok(Self {
                profile: EmbeddingProfile::new(
                    [1; 32],
                    [2; 32],
                    2,
                    EmbeddingDType::F16,
                    false,
                    Similarity::Dot,
                )?,
                entered: AtomicBool::new(false),
                release: AtomicBool::new(!block),
                calls: AtomicUsize::new(0),
                largest_batch: AtomicUsize::new(0),
            })
        }
    }
    impl TextEmbedding for Encoder {
        fn profile(&self) -> &EmbeddingProfile {
            &self.profile
        }
        fn embed(&self, text: &str) -> Result<Vec<f32>> {
            Ok(if text == "head" {
                vec![1.0, 0.0]
            } else {
                vec![0.0, 1.0]
            })
        }
        fn index_windows(&self, text: &str) -> Result<Vec<String>> {
            Ok(text.split('|').map(str::to_owned).collect())
        }
        fn embed_batch_cancellable(
            &self,
            texts: &[String],
            cancel: &dyn Fn() -> bool,
        ) -> Result<Vec<Vec<f32>>> {
            self.entered.store(true, Ordering::Release);
            while !self.release.load(Ordering::Acquire) {
                if cancel() {
                    return Err(cancelled());
                }
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            if cancel() {
                return Err(cancelled());
            }
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.largest_batch.fetch_max(texts.len(), Ordering::Relaxed);
            texts.iter().map(|text| self.embed(text)).collect()
        }
    }
    fn job(project: ProjectId, id: u64, revision: u64) -> EmbeddingJob {
        EmbeddingJob {
            owner: EmbeddingOwner {
                project,
                kind: EntityKind::Node,
                entity_id: id,
            },
            revision,
        }
    }
    async fn wait_until(predicate: impl Fn() -> bool) -> Result<()> {
        tokio::time::timeout(std::time::Duration::from_secs(3), async {
            while !predicate() {
                tokio::time::sleep(std::time::Duration::from_millis(1)).await;
            }
        })
        .await
        .map_err(|_| Error::internal("test worker progress timed out"))
    }

    #[test]
    fn encoding_covers_complete_windows_in_bounded_batches_and_emits_removals() -> Result<()> {
        let encoder = Arc::new(Encoder::new(false)?);
        let work = EmbeddingWork {
            is_current: None,
            encoder: encoder.clone(),
            chunks: vec![
                SemanticTextChunk {
                    property: PropertyId(1),
                    text: Some(Arc::from("head|tail|tail|tail")),
                },
                SemanticTextChunk {
                    property: PropertyId(2),
                    text: None,
                },
            ],
        };
        let owner = job(ProjectId::random(), 7, 11);
        let output = encode(&work, owner, 2, &|| false)?;
        assert_eq!(
            output,
            vec![
                ResolvedVectorMutation::UpsertSpans {
                    property: PropertyId(1),
                    entity_id: 7,
                    coordinates: vec![
                        encoder.profile.quantize(&[1.0, 0.0])?,
                        encoder.profile.quantize(&[0.0, 1.0])?,
                        encoder.profile.quantize(&[0.0, 1.0])?,
                        encoder.profile.quantize(&[0.0, 1.0])?,
                    ],
                    revision: 11
                },
                ResolvedVectorMutation::Remove {
                    property: PropertyId(2),
                    entity_id: 7,
                    revision: 11
                }
            ]
        );
        assert_eq!(encoder.calls.load(Ordering::Relaxed), 2);
        assert_eq!(encoder.largest_batch.load(Ordering::Relaxed), 2);
        assert!(encode(&work, owner, 2, &|| true).is_err());
        Ok(())
    }

    #[test]
    fn retrieval_span_single_window_preserves_owner_revision_without_summary_inference()
    -> Result<()> {
        let encoder = Arc::new(Encoder::new(false)?);
        let work = EmbeddingWork {
            encoder: encoder.clone(),
            chunks: vec![SemanticTextChunk {
                property: PropertyId(9),
                text: Some(Arc::from("head")),
            }],
            is_current: None,
        };
        let output = encode(&work, job(ProjectId::random(), 42, 77), 16, &|| false)?;
        assert_eq!(
            output,
            vec![ResolvedVectorMutation::UpsertSpans {
                property: PropertyId(9),
                entity_id: 42,
                coordinates: vec![encoder.profile.quantize(&[1.0, 0.0])?],
                revision: 77,
            }]
        );
        assert_eq!(encoder.calls.load(Ordering::Relaxed), 1);
        assert_eq!(encoder.largest_batch.load(Ordering::Relaxed), 1);
        Ok(())
    }

    #[test]
    fn retrieval_span_large_owner_keeps_distinct_topics_and_bounded_inference() -> Result<()> {
        let encoder = Arc::new(Encoder::new(false)?);
        let text = std::iter::once("head".to_owned())
            .chain((0..1_000).map(|_| "tail".repeat(1_024)))
            .collect::<Vec<_>>()
            .join("|");
        let source: Arc<str> = text.into();
        let original = Arc::clone(&source);
        let work = EmbeddingWork {
            encoder: encoder.clone(),
            chunks: vec![SemanticTextChunk {
                property: PropertyId(1),
                text: Some(source),
            }],
            is_current: None,
        };
        let output = encode(&work, job(ProjectId::random(), 7, 11), 16, &|| false)?;
        let [
            ResolvedVectorMutation::UpsertSpans {
                coordinates,
                entity_id,
                revision,
                ..
            },
        ] = output.as_slice()
        else {
            return Err(Error::internal("long owner did not publish span vectors"));
        };
        assert_eq!((*entity_id, *revision), (7, 11));
        assert_eq!(coordinates.len(), 1_001);
        assert_eq!(coordinates[0], encoder.profile.quantize(&[1.0, 0.0])?);
        assert!(
            coordinates[1..]
                .iter()
                .all(|vector| *vector == coordinates[1])
        );
        assert_eq!(
            encoder.calls.load(Ordering::Relaxed),
            1_001_usize.div_ceil(16)
        );
        assert_eq!(encoder.largest_batch.load(Ordering::Relaxed), 16);
        assert_eq!(work.chunks[0].text.as_deref(), Some(original.as_ref()));
        Ok(())
    }

    #[test]
    fn retrieval_span_cancellation_between_batches_discards_partial_owner() -> Result<()> {
        let encoder = Arc::new(Encoder::new(false)?);
        let work = EmbeddingWork {
            encoder: encoder.clone(),
            chunks: vec![SemanticTextChunk {
                property: PropertyId(1),
                text: Some(Arc::from("head|tail|tail|tail")),
            }],
            is_current: None,
        };
        let error = encode(&work, job(ProjectId::random(), 7, 11), 2, &|| {
            encoder.calls.load(Ordering::Relaxed) >= 1
        })
        .err()
        .ok_or_else(|| Error::internal("cancelled owner returned partial vectors"))?;
        assert_eq!(error.code, ErrorCode::Cancelled);
        assert_eq!(encoder.calls.load(Ordering::Relaxed), 1);
        let error = encode(&work, job(ProjectId::random(), 7, 11), 0, &|| false)
            .err()
            .ok_or_else(|| Error::internal("zero inference batch was accepted"))?;
        assert!(error.message.contains("batch must be nonempty"));
        Ok(())
    }

    struct InvalidEncoder {
        profile: EmbeddingProfile,
        vector: Vec<f32>,
    }

    impl TextEmbedding for InvalidEncoder {
        fn profile(&self) -> &EmbeddingProfile {
            &self.profile
        }

        fn embed(&self, _text: &str) -> Result<Vec<f32>> {
            Ok(self.vector.clone())
        }
    }

    struct MissingBatchEncoder(EmbeddingProfile);

    impl TextEmbedding for MissingBatchEncoder {
        fn profile(&self) -> &EmbeddingProfile {
            &self.0
        }

        fn embed(&self, _text: &str) -> Result<Vec<f32>> {
            Ok(vec![1.0, 0.0])
        }

        fn index_windows(&self, _text: &str) -> Result<Vec<String>> {
            Ok(vec!["head".to_owned(), "tail".to_owned()])
        }

        fn embed_batch(&self, _texts: &[String]) -> Result<Vec<Vec<f32>>> {
            Ok(vec![vec![1.0, 0.0]])
        }
    }

    #[test]
    fn retrieval_span_rejects_an_encoder_that_drops_a_window() -> Result<()> {
        let work = EmbeddingWork {
            encoder: Arc::new(MissingBatchEncoder(Encoder::new(false)?.profile)),
            chunks: vec![SemanticTextChunk {
                property: PropertyId(1),
                text: Some(Arc::from("head|tail")),
            }],
            is_current: None,
        };
        let error = encode(&work, job(ProjectId::random(), 7, 11), 16, &|| false)
            .err()
            .ok_or_else(|| Error::internal("encoder dropped a window without an error"))?;
        assert_eq!(error.code, ErrorCode::EmbeddingProfileMismatch);
        Ok(())
    }

    #[test]
    fn retrieval_span_rejects_malformed_vectors_and_normalizes_each_window() -> Result<()> {
        let profile = EmbeddingProfile::new(
            [1; 32],
            [2; 32],
            2,
            EmbeddingDType::F16,
            true,
            Similarity::Dot,
        )?;
        for vector in [
            vec![1.0],
            vec![f32::NAN, 0.0],
            vec![f32::INFINITY, 1.0],
            vec![0.0, 0.0],
        ] {
            let work = EmbeddingWork {
                encoder: Arc::new(InvalidEncoder {
                    profile: profile.clone(),
                    vector,
                }),
                chunks: vec![SemanticTextChunk {
                    property: PropertyId(1),
                    text: Some(Arc::from("text")),
                }],
                is_current: None,
            };
            let error = encode(&work, job(ProjectId::random(), 7, 11), 16, &|| false)
                .err()
                .ok_or_else(|| Error::internal("malformed encoder output was accepted"))?;
            assert_eq!(error.code, ErrorCode::EmbeddingProfileMismatch);
        }
        let work = EmbeddingWork {
            encoder: Arc::new(InvalidEncoder {
                profile: profile.clone(),
                vector: vec![3.0, 4.0],
            }),
            chunks: vec![SemanticTextChunk {
                property: PropertyId(1),
                text: Some(Arc::from("text")),
            }],
            is_current: None,
        };
        let output = encode(&work, job(ProjectId::random(), 7, 11), 16, &|| false)?;
        assert_eq!(
            output,
            vec![ResolvedVectorMutation::UpsertSpans {
                property: PropertyId(1),
                entity_id: 7,
                coordinates: vec![profile.quantize(&[0.6, 0.8])?],
                revision: 11,
            }]
        );
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn retrieval_span_stale_owner_never_publishes_partial_windows() -> Result<()> {
        let encoder = Arc::new(Encoder::new(true)?);
        let current = Arc::new(AtomicBool::new(true));
        let loaded_encoder = Arc::clone(&encoder);
        let loaded_current = Arc::clone(&current);
        let published = Arc::new(AtomicUsize::new(0));
        let publish_count = Arc::clone(&published);
        let failures = Arc::new(AtomicUsize::new(0));
        let failure_count = Arc::clone(&failures);
        let callbacks = Arc::new(EmbeddingCallbacks {
            load: Arc::new(move |_, _| {
                let current = Arc::clone(&loaded_current);
                Ok(Some(EmbeddingWork {
                    encoder: loaded_encoder.clone(),
                    chunks: vec![SemanticTextChunk {
                        property: PropertyId(1),
                        text: Some(Arc::from("head|tail")),
                    }],
                    is_current: Some(Arc::new(move || current.load(Ordering::Acquire))),
                }))
            }),
            publish: Arc::new(move |_, _, _| {
                publish_count.fetch_add(1, Ordering::Relaxed);
                Ok(true)
            }),
            failed: Arc::new(move |_, _| {
                failure_count.fetch_add(1, Ordering::Relaxed);
            }),
        });
        let task = tokio::spawn(process_job(
            job(ProjectId::random(), 7, 11),
            callbacks,
            1,
            CancellationToken::new(),
        ));
        wait_until(|| encoder.entered.load(Ordering::Acquire)).await?;
        current.store(false, Ordering::Release);
        encoder.release.store(true, Ordering::Release);
        assert!(
            task.await
                .map_err(|error| Error::internal(format!("test worker failed: {error}")))?
        );
        assert_eq!(published.load(Ordering::Relaxed), 0);
        assert_eq!(failures.load(Ordering::Relaxed), 0);
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn retrieval_span_failed_publication_reloads_and_retries_whole_owner() -> Result<()> {
        let encoder = Arc::new(Encoder::new(false)?);
        let loaded_encoder = Arc::clone(&encoder);
        let loads = Arc::new(AtomicUsize::new(0));
        let load_count = Arc::clone(&loads);
        let published = Arc::new(AtomicUsize::new(0));
        let publish_count = Arc::clone(&published);
        let failures = Arc::new(AtomicUsize::new(0));
        let failure_count = Arc::clone(&failures);
        let callbacks = Arc::new(EmbeddingCallbacks {
            load: Arc::new(move |_, _| {
                load_count.fetch_add(1, Ordering::Relaxed);
                Ok(Some(EmbeddingWork {
                    encoder: loaded_encoder.clone(),
                    chunks: vec![SemanticTextChunk {
                        property: PropertyId(1),
                        text: Some(Arc::from("head|tail")),
                    }],
                    is_current: None,
                }))
            }),
            publish: Arc::new(move |_, output, _| {
                let [ResolvedVectorMutation::UpsertSpans { coordinates, .. }] = output.as_slice()
                else {
                    return Err(Error::internal("retry did not publish all owner spans"));
                };
                assert_eq!(coordinates.len(), 2);
                if publish_count.fetch_add(1, Ordering::Relaxed) == 0 {
                    Err(Error::internal("transient publish failure"))
                } else {
                    Ok(true)
                }
            }),
            failed: Arc::new(move |_, _| {
                failure_count.fetch_add(1, Ordering::Relaxed);
            }),
        });
        assert!(
            process_job(
                job(ProjectId::random(), 7, 11),
                callbacks,
                1,
                CancellationToken::new(),
            )
            .await
        );
        assert_eq!(loads.load(Ordering::Relaxed), 2);
        assert_eq!(published.load(Ordering::Relaxed), 2);
        assert_eq!(failures.load(Ordering::Relaxed), 1);
        assert_eq!(encoder.calls.load(Ordering::Relaxed), 4);
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn durable_cursor_idle_does_not_poll_and_commit_racing_empty_read_wakes() -> Result<()> {
        let reads = Arc::new(AtomicUsize::new(0));
        let entered = Arc::new(AtomicBool::new(false));
        let release = Arc::new(AtomicBool::new(false));
        let pending = Arc::new(AtomicBool::new(false));
        let loaded = Arc::new(AtomicUsize::new(0));
        let (source_reads, source_entered, source_release, source_pending) = (
            reads.clone(),
            entered.clone(),
            release.clone(),
            pending.clone(),
        );
        let load_count = loaded.clone();
        let project = ProjectId::random();
        let worker = DurableEmbeddingWorker::start(
            Arc::new(move |cancel| {
                if source_reads.fetch_add(1, Ordering::AcqRel) == 0 {
                    source_entered.store(true, Ordering::Release);
                    while !source_release.load(Ordering::Acquire) && !cancel.is_cancelled() {
                        std::thread::sleep(std::time::Duration::from_millis(1));
                    }
                    // Return the empty result observed before the racing commit.
                    return Ok(None);
                }
                Ok(source_pending
                    .swap(false, Ordering::AcqRel)
                    .then_some(job(project, 7, 1)))
            }),
            EmbeddingCallbacks {
                load: Arc::new(move |_, _| {
                    load_count.fetch_add(1, Ordering::AcqRel);
                    Ok(None)
                }),
                publish: Arc::new(|_, _, _| panic!("absent owner cannot publish")),
                failed: Arc::new(|_, error| panic!("unexpected failure: {error}")),
            },
        )?;
        wait_until(|| entered.load(Ordering::Acquire)).await?;
        pending.store(true, Ordering::Release);
        worker.wake();
        release.store(true, Ordering::Release);
        wait_until(|| loaded.load(Ordering::Acquire) == 1 && worker.is_idle()).await?;
        let idle_reads = reads.load(Ordering::Acquire);
        tokio::time::sleep(std::time::Duration::from_millis(120)).await;
        assert_eq!(reads.load(Ordering::Acquire), idle_reads);
        tokio::time::timeout(std::time::Duration::from_secs(2), worker.shutdown())
            .await
            .map_err(|_| Error::internal("idle embedding worker did not cancel"))??;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn durable_cursor_commit_during_inference_revisits_completed_pass() -> Result<()> {
        let reads = Arc::new(AtomicUsize::new(0));
        let loaded = Arc::new(AtomicUsize::new(0));
        let release = Arc::new(AtomicBool::new(false));
        let source_reads = reads.clone();
        let load_count = loaded.clone();
        let load_release = release.clone();
        let project = ProjectId::random();
        let worker = DurableEmbeddingWorker::start(
            Arc::new(move |_| {
                // The second owner sorts before the cursor, so it appears only after
                // the empty result resets the source's scan position.
                let pass = source_reads.fetch_add(1, Ordering::AcqRel);
                Ok(matches!(pass, 0 | 2).then_some(job(project, pass as u64 + 1, 1)))
            }),
            EmbeddingCallbacks {
                load: Arc::new(move |_, cancel| {
                    if load_count.fetch_add(1, Ordering::AcqRel) == 0 {
                        while !load_release.load(Ordering::Acquire) && !cancel.is_cancelled() {
                            std::thread::sleep(std::time::Duration::from_millis(1));
                        }
                    }
                    Ok(None)
                }),
                publish: Arc::new(|_, _, _| panic!("absent owner cannot publish")),
                failed: Arc::new(|_, error| panic!("unexpected failure: {error}")),
            },
        )?;
        wait_until(|| loaded.load(Ordering::Acquire) == 1).await?;
        worker.wake();
        release.store(true, Ordering::Release);
        wait_until(|| loaded.load(Ordering::Acquire) == 2 && worker.is_idle()).await?;
        worker.shutdown().await?;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn durable_cursor_reads_one_owner_while_inference_is_paused_and_shutdown_cancels()
    -> Result<()> {
        let encoder = Arc::new(Encoder::new(true)?);
        let source_reads = Arc::new(AtomicUsize::new(0));
        let source = source_reads.clone();
        let project = ProjectId::random();
        let load_encoder = encoder.clone();
        let published = Arc::new(AtomicUsize::new(0));
        let publish_count = published.clone();
        let worker = DurableEmbeddingWorker::start(
            Arc::new(move |_| {
                let index = source.fetch_add(1, Ordering::Relaxed);
                Ok(Some(job(project, index as u64 + 1, 1)))
            }),
            EmbeddingCallbacks {
                load: Arc::new(move |_, _| {
                    Ok(Some(EmbeddingWork {
                        encoder: load_encoder.clone(),
                        chunks: vec![SemanticTextChunk {
                            property: PropertyId(1),
                            text: Some(Arc::from("head|tail")),
                        }],
                        is_current: None,
                    }))
                }),
                publish: Arc::new(move |_, _, _| {
                    publish_count.fetch_add(1, Ordering::Relaxed);
                    Ok(true)
                }),
                failed: Arc::new(|_, error| panic!("unexpected embedding failure: {error}")),
            },
        )?;
        wait_until(|| encoder.entered.load(Ordering::Acquire)).await?;
        assert_eq!(source_reads.load(Ordering::Relaxed), 1);
        assert_eq!(published.load(Ordering::Relaxed), 0);
        tokio::time::timeout(std::time::Duration::from_secs(2), worker.shutdown())
            .await
            .map_err(|_| Error::internal("paused inference did not cancel"))??;
        assert_eq!(published.load(Ordering::Relaxed), 0);
        assert_eq!(source_reads.load(Ordering::Relaxed), 1);
        Ok(())
    }
}
