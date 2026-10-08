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
use tokio::sync::Mutex;
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

/// Pulls one owner from canonical recovery or the committed WAL. No producer-side work queue.
pub(super) type NextCommittedOwner =
    dyn Fn(&CancellationToken) -> Result<Option<EmbeddingJob>> + Send + Sync;

pub(super) struct DurableEmbeddingWorker {
    cancellation: CancellationToken,
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
        let worker = runtime.spawn(async move {
            while !cancel.is_cancelled() {
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
                        worker_idle.store(true, Ordering::Release);
                        tokio::select! {
                            _ = cancel.cancelled() => break,
                            _ = tokio::time::sleep(std::time::Duration::from_millis(10)) => {}
                        }
                    }
                    Ok(Err(error)) => {
                        tracing::error!(code = ?error.code, message = %error.message, "embedding WAL catch-up failed");
                        tokio::select! {
                            _ = cancel.cancelled() => break,
                            _ = tokio::time::sleep(std::time::Duration::from_millis(100)) => {}
                        }
                    }
                    Err(error) => {
                        tracing::error!(%error, "embedding WAL reader worker terminated");
                        break;
                    }
                }
            }
        });
        Ok(Self {
            cancellation,
            worker: Mutex::new(Some(worker)),
            idle,
        })
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
    work.encoder.profile().validate()?;
    let dimension = usize::try_from(work.encoder.profile().dimension)
        .map_err(|_| Error::internal("embedding dimension exceeds process range"))?;
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
        let count = windows.len();
        let mut aggregate = vec![0_f64; dimension];
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
            for vector in encoded {
                for (sum, value) in aggregate.iter_mut().zip(vector) {
                    *sum += f64::from(value);
                }
            }
        }
        let mut vector = aggregate
            .into_iter()
            .map(|sum| (sum / count as f64) as f32)
            .collect::<Vec<_>>();
        if work.encoder.profile().normalized {
            let norm = vector
                .iter()
                .map(|value| f64::from(*value).powi(2))
                .sum::<f64>()
                .sqrt();
            if norm > 0.0 {
                for value in &mut vector {
                    *value = (f64::from(*value) / norm) as f32;
                }
            }
        }
        mutations.push(ResolvedVectorMutation::Upsert {
            property: chunk.property,
            entity_id: job.owner.entity_id,
            coordinates: work.encoder.profile().quantize(&vector)?,
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
                ResolvedVectorMutation::Upsert {
                    property: PropertyId(1),
                    entity_id: 7,
                    coordinates: encoder.profile.quantize(&[0.25, 0.75])?,
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
