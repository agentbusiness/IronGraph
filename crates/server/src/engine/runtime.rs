use std::{
    collections::VecDeque,
    path::Path,
    sync::Arc,
    sync::mpsc::{self, RecvTimeoutError, SyncSender, TrySendError},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use crate::storage::{
    AdmissionClass, AdmissionController, AdmissionLimits, AdmissionPermit, ConnectionId,
    DurableLog, MutationEntry,
};
use crate::{Bookmark, Error, ErrorCode, Result};

use super::{
    ApplicationWait, CommandReservation, CommandReservationOutcome, CommittedWrite,
    MutationStateBackend, ProcessId, StandaloneNode, WriteCommand, WriteRequest, WriteResponse,
    WriteStorageLimits,
};

const MAX_CLIENT_WAIT_MILLIS: u64 = 10 * 60 * 1_000;
// Keep the no-delay single-write path, but do not impose an artificial throughput ceiling when
// thousands of already-admitted independent writes arrive together. This bound remains finite,
// every queued payload already owns an admission permit, and a graph backend may end a batch
// earlier when its COW overlay reaches capacity.
const MAX_WRITE_BATCH: usize = 8_192;
const MAX_PENDING_WAL_BATCHES: usize = 8;
const WAL_SYNC_INTERVAL: Duration = Duration::from_millis(10);

struct QueuedWrite {
    command: Option<WriteCommand>,
    deadline: ClientDeadline,
    admission: AdmissionPermit,
    command_prepared: bool,
    reply: tokio::sync::oneshot::Sender<Result<CommittedWrite>>,
}

enum WriterMessage {
    Write(QueuedWrite),
    #[cfg(test)]
    Flush(tokio::sync::oneshot::Sender<Result<()>>),
    Shutdown(tokio::sync::oneshot::Sender<Result<()>>),
}

enum PersistenceMessage {
    Append(Vec<MutationEntry>),
    Flush(mpsc::SyncSender<Result<()>>),
    Shutdown(mpsc::SyncSender<Result<()>>),
}

#[cfg(test)]
#[derive(Default)]
struct EventualDurabilityTestHook {
    block_append: std::sync::atomic::AtomicBool,
    append_reached: std::sync::atomic::AtomicBool,
}

struct PreparedWrite {
    mutation: MutationEntry,
    reservation: CommandReservation,
    request_id: Option<uuid::Uuid>,
    admission: AdmissionPermit,
    reply: tokio::sync::oneshot::Sender<Result<CommittedWrite>>,
}

fn standalone_writer_loop(
    receiver: mpsc::Receiver<WriterMessage>,
    persistence: SyncSender<PersistenceMessage>,
    backend: Arc<dyn MutationStateBackend>,
    gate: Arc<parking_lot::Mutex<()>>,
    runtime: tokio::runtime::Handle,
) {
    let mut pending = VecDeque::new();
    let mut deferred_control = None;
    loop {
        if pending.is_empty() {
            let received = if let Some(message) = deferred_control.take() {
                Ok(message)
            } else {
                receiver.recv().map_err(|_| RecvTimeoutError::Disconnected)
            };
            match received {
                Ok(WriterMessage::Write(write)) => pending.push_back(write),
                #[cfg(test)]
                Ok(WriterMessage::Flush(reply)) => {
                    let (flush, completed) = mpsc::sync_channel(1);
                    let result = persistence
                        .send(PersistenceMessage::Flush(flush))
                        .map_err(|_| Error::internal("WAL persistence worker is unavailable"))
                        .and_then(|()| {
                            completed.recv().map_err(|_| {
                                Error::internal("WAL persistence flush response was lost")
                            })?
                        });
                    let _ = reply.send(result);
                    continue;
                }
                Ok(WriterMessage::Shutdown(reply)) => {
                    let (shutdown, completed) = mpsc::sync_channel(1);
                    let result = persistence
                        .send(PersistenceMessage::Shutdown(shutdown))
                        .map_err(|_| Error::internal("WAL persistence worker is unavailable"))
                        .and_then(|()| {
                            completed.recv().map_err(|_| {
                                Error::internal("WAL persistence shutdown response was lost")
                            })?
                        });
                    let _ = reply.send(result);
                    return;
                }
                Err(RecvTimeoutError::Timeout) => continue,
                Err(RecvTimeoutError::Disconnected) => {
                    let (shutdown, completed) = mpsc::sync_channel(1);
                    let _ = persistence.send(PersistenceMessage::Shutdown(shutdown));
                    let _ = completed.recv();
                    return;
                }
            }
        }
        while pending.len() < MAX_WRITE_BATCH {
            match receiver.try_recv() {
                Ok(WriterMessage::Write(item)) => pending.push_back(item),
                Ok(control) => {
                    deferred_control = Some(control);
                    break;
                }
                Err(mpsc::TryRecvError::Empty) => break,
                Err(mpsc::TryRecvError::Disconnected) => break,
            }
        }
        match process_writer_batch(
            &receiver,
            &mut deferred_control,
            &mut pending,
            &backend,
            &gate,
            &runtime,
        ) {
            Ok(appended) if appended.is_empty() => {}
            Ok(appended) => {
                if persistence
                    .send(PersistenceMessage::Append(appended))
                    .is_err()
                {
                    tracing::error!(
                        "standalone WAL persistence stopped; the writer thread is shutting down \
                         and further writes will be rejected"
                    );
                    return;
                }
            }
            Err(error) => {
                tracing::error!(
                    code = ?error.code,
                    message = %error.message,
                    "standalone writer batch failed; the writer thread is shutting down and \
                     further writes will be rejected"
                );
                while let Some(write) = pending.pop_front() {
                    let _ = write.reply.send(Err(copy_error(&error)));
                }
                return;
            }
        }
    }
}

fn standalone_persistence_loop(
    receiver: mpsc::Receiver<PersistenceMessage>,
    wal: Arc<parking_lot::Mutex<DurableLog>>,
    #[cfg(test)] test_hook: Arc<EventualDurabilityTestHook>,
) {
    let mut unsynced = false;
    let mut last_sync = Instant::now();
    loop {
        let received = if unsynced {
            receiver.recv_timeout(WAL_SYNC_INTERVAL.saturating_sub(last_sync.elapsed()))
        } else {
            receiver.recv().map_err(|_| RecvTimeoutError::Disconnected)
        };
        match received {
            Ok(PersistenceMessage::Append(mut entries)) => {
                #[cfg(test)]
                {
                    use std::sync::atomic::Ordering;

                    test_hook.append_reached.store(true, Ordering::Release);
                    while test_hook.block_append.load(Ordering::Acquire) {
                        std::thread::yield_now();
                    }
                }
                let result = if entries.len() == 1 {
                    wal.lock()
                        .append_buffered(entries.pop().expect("one buffered mutation"))
                } else {
                    wal.lock().append_batch_buffered(entries)
                };
                if let Err(error) = result {
                    tracing::error!(
                        code = ?error.code,
                        message = %error.message,
                        "standalone WAL append failed; the persistence thread is shutting down \
                         and WAL flushes will be rejected"
                    );
                    return;
                }
                unsynced = true;
            }
            Ok(PersistenceMessage::Flush(reply)) => {
                let result = wal.lock().sync();
                if result.is_ok() {
                    unsynced = false;
                    last_sync = Instant::now();
                }
                let _ = reply.send(result);
            }
            Ok(PersistenceMessage::Shutdown(reply)) => {
                let result = wal.lock().sync();
                let _ = reply.send(result);
                return;
            }
            Err(RecvTimeoutError::Timeout) => {
                if let Err(error) = wal.lock().sync() {
                    tracing::error!(
                        code = ?error.code,
                        message = %error.message,
                        "standalone WAL sync failed; the persistence thread is shutting down \
                         and WAL flushes will be rejected"
                    );
                    return;
                }
                unsynced = false;
                last_sync = Instant::now();
            }
            Err(RecvTimeoutError::Disconnected) => {
                if unsynced {
                    let _ = wal.lock().sync();
                }
                return;
            }
        }
    }
}

fn process_writer_batch(
    receiver: &mpsc::Receiver<WriterMessage>,
    deferred_control: &mut Option<WriterMessage>,
    pending: &mut VecDeque<QueuedWrite>,
    backend: &Arc<dyn MutationStateBackend>,
    gate: &Arc<parking_lot::Mutex<()>>,
    runtime: &tokio::runtime::Handle,
) -> Result<Vec<MutationEntry>> {
    let Some(first_deadline) = pending.front().map(|item| item.deadline) else {
        return Ok(Vec::new());
    };
    let Ok(remaining) = first_deadline.remaining() else {
        if let Some(item) = pending.pop_front() {
            let _ = item.reply.send(Err(client_deadline_exceeded(
                "standalone write expired in the writer queue",
            )));
        }
        return Ok(Vec::new());
    };
    let Some(_batch_gate) = gate.try_lock_for(remaining) else {
        if let Some(item) = pending.pop_front() {
            let _ = item.reply.send(Err(client_deadline_exceeded(
                "standalone write admission timed out",
            )));
        }
        return Ok(Vec::new());
    };

    let applied_at = runtime.block_on(backend.applied_bookmark());
    let term = applied_at.term.max(1);
    let mut next_index = match applied_at.index.checked_add(1) {
        Some(index) => index,
        None => {
            if let Some(item) = pending.pop_front() {
                let _ = item
                    .reply
                    .send(Err(Error::internal("standalone mutation index exhausted")));
            }
            return Ok(Vec::new());
        }
    };
    let mut prepared = Vec::with_capacity(pending.len().min(MAX_WRITE_BATCH));
    while prepared.len() < MAX_WRITE_BATCH {
        let Some(mut item) = pending.pop_front() else {
            break;
        };
        if item.deadline.remaining().is_err() {
            let _ = item.reply.send(Err(client_deadline_exceeded(
                "standalone write expired in the writer queue",
            )));
            continue;
        }
        let command = match item.command.take() {
            Some(command) if item.command_prepared => command,
            Some(command) => {
                let commit_time_millis = match SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .ok()
                    .and_then(|duration| i64::try_from(duration.as_millis()).ok())
                {
                    Some(value) => value,
                    None => {
                        let _ = item
                            .reply
                            .send(Err(Error::internal("commit time exceeds i64 milliseconds")));
                        continue;
                    }
                };
                match runtime.block_on(backend.prepare_command(command, commit_time_millis)) {
                    Ok(command) => command,
                    Err(error) => {
                        let _ = item.reply.send(Err(error));
                        continue;
                    }
                }
            }
            None => {
                let _ = item
                    .reply
                    .send(Err(Error::internal("queued write lost its command")));
                continue;
            }
        };
        let position = Bookmark {
            term,
            index: next_index,
        };
        let reservation = match runtime.block_on(backend.reserve_command(&command, position)) {
            Ok(reservation) => reservation,
            Err(error) if error.code == ErrorCode::WriteAdmissionFull && !prepared.is_empty() => {
                item.command = Some(command);
                item.command_prepared = true;
                pending.push_front(item);
                break;
            }
            Err(error) => {
                let _ = item.reply.send(Err(error));
                continue;
            }
        };
        let request_id = command.request_id;
        let mutation = match MutationEntry::new(
            term,
            next_index,
            command.kind,
            command.project_id,
            command.request_id,
            command.commit_time_millis,
            command.payload,
        ) {
            Ok(mutation) => mutation,
            Err(error) => {
                let completion = runtime.block_on(backend.complete_command_reservation(
                    reservation,
                    CommandReservationOutcome::Rejected,
                ));
                let _ = item.reply.send(completion.and(Err(error)));
                continue;
            }
        };
        let pipelined = reservation.may_release_after_append();
        prepared.push(PreparedWrite {
            mutation,
            reservation,
            request_id,
            admission: item.admission,
            reply: item.reply,
        });
        next_index = match next_index.checked_add(1) {
            Some(index) => index,
            None => break,
        };
        if !pipelined {
            break;
        }
        // Preparation/reservation is often long enough for concurrent callers to arrive. Pull
        // them into this group before deciding the batch is complete; no artificial timer is
        // added to the single-write latency path.
        while pending.len() + prepared.len() < MAX_WRITE_BATCH {
            match receiver.try_recv() {
                Ok(WriterMessage::Write(item)) => pending.push_back(item),
                Ok(control) => {
                    *deferred_control = Some(control);
                    break;
                }
                Err(mpsc::TryRecvError::Empty | mpsc::TryRecvError::Disconnected) => break,
            }
        }
    }
    if prepared.is_empty() {
        return Ok(Vec::new());
    }

    // Canonical publication is the commit boundary. Replies are delivered immediately afterward;
    // WAL serialization, write, and fsync are deliberately outside the request's latency path.
    // The delta path moves each already-encoded mutation exactly once into one buffered append and
    // never rebuilds or rescans unrelated graph state.
    let mut appended = Vec::with_capacity(prepared.len());
    let mut fatal = None;
    for write in prepared {
        let PreparedWrite {
            mutation,
            reservation,
            request_id,
            admission,
            reply,
        } = write;
        let result = if let Some(error) = fatal.as_ref() {
            let _ = runtime.block_on(
                backend
                    .complete_command_reservation(reservation, CommandReservationOutcome::Rejected),
            );
            Err(copy_error(error))
        } else {
            match runtime.block_on(backend.apply_mutation(&mutation)) {
                Ok(applied) => {
                    let bookmark = mutation.bookmark();
                    appended.push(mutation);
                    runtime
                        .block_on(backend.complete_command_reservation(
                            reservation,
                            CommandReservationOutcome::Applied,
                        ))
                        .map(|()| CommittedWrite {
                            response: WriteResponse {
                                bookmark,
                                request_id,
                                payload: applied.response,
                                duplicate: applied.duplicate,
                            },
                            application: ApplicationWait::Complete,
                        })
                }
                Err(error) => {
                    let _ = runtime.block_on(backend.complete_command_reservation(
                        reservation,
                        CommandReservationOutcome::Rejected,
                    ));
                    fatal = Some(copy_error(&error));
                    Err(error)
                }
            }
        };
        drop(admission);
        let _ = reply.send(result);
    }

    if appended.is_empty() {
        return Ok(Vec::new());
    }
    Ok(appended)
}

fn copy_error(error: &Error) -> Error {
    Error {
        code: error.code,
        message: error.message.clone(),
        retryable: error.retryable,
        retry_after_ms: error.retry_after_ms,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

    use async_trait::async_trait;

    use crate::ProjectId;

    use super::*;
    use crate::engine::{
        BackendSnapshot, ExecutionClass, MutationApplyResult, NodeIdentity, WriteStorageLimits,
    };
    use crate::storage::MutationKind;

    struct BatchedBackend {
        applied: AtomicU64,
        reservations: AtomicUsize,
        first_apply_reservations: AtomicUsize,
    }

    #[async_trait]
    impl MutationStateBackend for BatchedBackend {
        async fn prepare_command(
            &self,
            mut command: WriteCommand,
            sequencer_time_millis: i64,
        ) -> Result<WriteCommand> {
            if self.reservations.load(Ordering::Acquire) == 0 {
                std::thread::sleep(Duration::from_millis(10));
            }
            command.commit_time_millis = sequencer_time_millis;
            Ok(command)
        }

        async fn reserve_command(
            &self,
            _command: &WriteCommand,
            position: Bookmark,
        ) -> Result<CommandReservation> {
            self.reservations.fetch_add(1, Ordering::AcqRel);
            CommandReservation::pipelined(position)
        }

        async fn apply_mutation(&self, mutation: &MutationEntry) -> Result<MutationApplyResult> {
            let previous = self.applied.fetch_add(1, Ordering::AcqRel);
            if mutation.index() != previous + 1 {
                return Err(Error::internal("test backend observed an apply gap"));
            }
            if mutation.index() == 1 {
                self.first_apply_reservations
                    .store(self.reservations.load(Ordering::Acquire), Ordering::Release);
            }
            Ok(MutationApplyResult::default())
        }

        async fn applied_bookmark(&self) -> Bookmark {
            Bookmark {
                term: 1,
                index: self.applied.load(Ordering::Acquire),
            }
        }

        async fn build_snapshot(
            &self,
            _bookmark: Bookmark,
            _destination: &Path,
        ) -> Result<BackendSnapshot> {
            Err(Error::internal("snapshot is outside this test"))
        }

        async fn install_snapshot(
            &self,
            _bookmark: Bookmark,
            _snapshot: &BackendSnapshot,
        ) -> Result<()> {
            Err(Error::internal("snapshot is outside this test"))
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn non_blocking_durability_paths_batch_independent_writes() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let identity = NodeIdentity::generate_genesis().public();
        let backend = Arc::new(BatchedBackend {
            applied: AtomicU64::new(0),
            reservations: AtomicUsize::new(0),
            first_apply_reservations: AtomicUsize::new(0),
        });
        let runtime = WriteRuntime::start(
            directory.path(),
            StandaloneNode {
                identity,
                execution_class: ExecutionClass::Cpu,
            },
            backend.clone(),
            WriteStorageLimits {
                max_log_record_bytes: 1024 * 1024,
                max_log_entries_per_read: 1024,
                max_snapshot_bytes: 1024 * 1024,
            },
        )
        .await?;
        let project = ProjectId::random();
        let request = |payload: u8| WriteRequest {
            command: WriteCommand {
                kind: MutationKind::Graph,
                project_id: Some(project),
                request_id: Some(uuid::Uuid::new_v4()),
                commit_time_millis: 0,
                payload: vec![payload],
            },
            timeout_millis: 5_000,
            connection_id: ConnectionId::new(),
            admission_class: AdmissionClass::Client,
            transaction_fence: None,
        };

        let runtime_work_progressed = Arc::new(AtomicBool::new(false));
        let progress = Arc::clone(&runtime_work_progressed);
        let unrelated = tokio::spawn(async move {
            tokio::task::yield_now().await;
            progress.store(true, Ordering::Release);
        });
        let (first, second) = tokio::join!(runtime.write(request(1)), runtime.write(request(2)));
        unrelated
            .await
            .map_err(|error| Error::internal(error.to_string()))?;
        assert_eq!(first?.response.bookmark.index, 1);
        assert_eq!(second?.response.bookmark.index, 2);
        assert!(runtime_work_progressed.load(Ordering::Acquire));
        assert_eq!(backend.first_apply_reservations.load(Ordering::Acquire), 2);
        assert_eq!(backend.reservations.load(Ordering::Acquire), 2);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn eventual_durability_ack_precedes_wal_and_shutdown_recovers_flushed_prefix()
    -> Result<()> {
        const WRITES: usize = 32;
        let directory = tempfile::tempdir()?;
        let identity = NodeIdentity::generate_genesis().public();
        let backend = Arc::new(BatchedBackend {
            applied: AtomicU64::new(0),
            reservations: AtomicUsize::new(0),
            first_apply_reservations: AtomicUsize::new(0),
        });
        let limits = WriteStorageLimits {
            max_log_record_bytes: 1024 * 1024,
            max_log_entries_per_read: 1024,
            max_snapshot_bytes: 1024 * 1024,
        };
        let runtime = WriteRuntime::start(
            directory.path(),
            StandaloneNode {
                identity,
                execution_class: ExecutionClass::Cpu,
            },
            backend.clone(),
            limits,
        )
        .await?;
        let project = ProjectId::random();
        let request = |payload: u8| WriteRequest {
            command: WriteCommand {
                kind: MutationKind::Graph,
                project_id: Some(project),
                request_id: Some(uuid::Uuid::new_v4()),
                commit_time_millis: 0,
                payload: vec![payload],
            },
            timeout_millis: 5_000,
            connection_id: ConnectionId::new(),
            admission_class: AdmissionClass::Client,
            transaction_fence: None,
        };

        runtime
            .eventual_durability_test_hook
            .block_append
            .store(true, Ordering::Release);
        let first = runtime.write(request(0)).await?;
        assert_eq!(first.response.bookmark.index, 1);
        assert_eq!(backend.applied.load(Ordering::Acquire), 1);
        tokio::time::timeout(Duration::from_secs(1), async {
            while !runtime
                .eventual_durability_test_hook
                .append_reached
                .load(Ordering::Acquire)
            {
                tokio::task::yield_now().await;
            }
        })
        .await
        .map_err(|_| Error::internal("writer did not reach the deferred WAL append"))?;
        assert_eq!(runtime.standalone_wal.lock().len(), 0);
        let second = runtime.write(request(1)).await?;
        assert_eq!(second.response.bookmark.index, 2);
        assert_eq!(backend.applied.load(Ordering::Acquire), 2);
        assert_eq!(runtime.standalone_wal.lock().len(), 0);

        runtime
            .eventual_durability_test_hook
            .block_append
            .store(false, Ordering::Release);
        runtime.flush_wal().await?;
        for offset in 2..WRITES {
            let committed = runtime.write(request(offset as u8)).await?;
            assert_eq!(committed.response.bookmark.index, offset as u64 + 1);
        }
        runtime.shutdown().await?;
        drop(runtime);

        let recovered_backend = Arc::new(BatchedBackend {
            applied: AtomicU64::new(0),
            reservations: AtomicUsize::new(0),
            first_apply_reservations: AtomicUsize::new(0),
        });
        let recovered_runtime = WriteRuntime::start(
            directory.path(),
            StandaloneNode {
                identity,
                execution_class: ExecutionClass::Cpu,
            },
            recovered_backend.clone(),
            limits,
        )
        .await?;
        let recovered = recovered_runtime.replay_standalone_wal().await?;
        assert_eq!(recovered.index, WRITES as u64);
        assert_eq!(
            recovered_backend.applied.load(Ordering::Acquire),
            WRITES as u64
        );
        recovered_runtime.shutdown().await?;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn replay_discards_a_wal_suffix_covered_by_a_newer_restored_snapshot() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let identity = NodeIdentity::generate_genesis().public();
        let limits = WriteStorageLimits {
            max_log_record_bytes: 1024 * 1024,
            max_log_entries_per_read: 1024,
            max_snapshot_bytes: 1024 * 1024,
        };
        let backend = Arc::new(BatchedBackend {
            applied: AtomicU64::new(0),
            reservations: AtomicUsize::new(0),
            first_apply_reservations: AtomicUsize::new(0),
        });
        let runtime = WriteRuntime::start(
            directory.path(),
            StandaloneNode {
                identity,
                execution_class: ExecutionClass::Cpu,
            },
            backend.clone(),
            limits,
        )
        .await?;
        let project = ProjectId::random();
        let request = |payload: u8| WriteRequest {
            command: WriteCommand {
                kind: MutationKind::Graph,
                project_id: Some(project),
                request_id: Some(uuid::Uuid::new_v4()),
                commit_time_millis: 0,
                payload: vec![payload],
            },
            timeout_millis: 5_000,
            connection_id: ConnectionId::new(),
            admission_class: AdmissionClass::Client,
            transaction_fence: None,
        };
        for payload in 0..3 {
            runtime.write(request(payload)).await?;
        }
        runtime.shutdown().await?;
        drop(runtime);

        // A restored snapshot is newer than the WAL tail when the previous process lost its WAL
        // appender and kept snapshotting published writes. Replay must realign the log to the
        // snapshot instead of refusing to start or leaving a colliding tail behind.
        let recovered_backend = Arc::new(BatchedBackend {
            applied: AtomicU64::new(5),
            reservations: AtomicUsize::new(0),
            first_apply_reservations: AtomicUsize::new(0),
        });
        let recovered_runtime = WriteRuntime::start(
            directory.path(),
            StandaloneNode {
                identity,
                execution_class: ExecutionClass::Cpu,
            },
            recovered_backend.clone(),
            limits,
        )
        .await?;
        let recovered = recovered_runtime.replay_standalone_wal().await?;
        assert_eq!(recovered, Bookmark { term: 1, index: 5 });
        assert_eq!(recovered_backend.applied.load(Ordering::Acquire), 5);

        let committed = recovered_runtime.write(request(9)).await?;
        assert_eq!(committed.response.bookmark.index, 6);
        recovered_runtime.shutdown().await?;
        drop(recovered_runtime);

        let wal = DurableLog::open(
            directory.path().join("standalone.wal"),
            limits.max_log_record_bytes,
            Bookmark::default(),
        )?;
        assert_eq!(wal.compacted_through(), Bookmark { term: 1, index: 5 });
        assert_eq!(wal.last_bookmark(), Bookmark { term: 1, index: 6 });
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn non_blocking_durability_paths_batch_beyond_old_sixty_four_write_ceiling() -> Result<()>
    {
        const WRITES: usize = 128;
        let directory = tempfile::tempdir()?;
        let identity = NodeIdentity::generate_genesis().public();
        let backend = Arc::new(BatchedBackend {
            applied: AtomicU64::new(0),
            reservations: AtomicUsize::new(0),
            first_apply_reservations: AtomicUsize::new(0),
        });
        let runtime = WriteRuntime::start(
            directory.path(),
            StandaloneNode {
                identity,
                execution_class: ExecutionClass::Cpu,
            },
            backend.clone(),
            WriteStorageLimits {
                max_log_record_bytes: 1024 * 1024,
                max_log_entries_per_read: 1024,
                max_snapshot_bytes: 1024 * 1024,
            },
        )
        .await?;
        let project = ProjectId::random();
        let requests = (0..WRITES).map(|offset| {
            runtime.write(WriteRequest {
                command: WriteCommand {
                    kind: MutationKind::Graph,
                    project_id: Some(project),
                    request_id: Some(uuid::Uuid::new_v4()),
                    commit_time_millis: 0,
                    payload: vec![offset as u8],
                },
                timeout_millis: 5_000,
                connection_id: ConnectionId::new(),
                admission_class: AdmissionClass::Client,
                transaction_fence: None,
            })
        });
        let committed = futures::future::join_all(requests).await;
        for (offset, result) in committed.into_iter().enumerate() {
            assert_eq!(result?.response.bookmark.index, offset as u64 + 1);
        }
        assert_eq!(backend.reservations.load(Ordering::Acquire), WRITES);
        assert_eq!(
            backend.first_apply_reservations.load(Ordering::Acquire),
            WRITES,
            "the coordinator split at the obsolete 64-write ceiling"
        );
        Ok(())
    }
}

/// Exact term/index reserved by the local sequencer for a generated control write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SequencerPosition {
    pub term: u64,
    pub index: u64,
}

/// Running standalone durable write path.
pub struct WriteRuntime {
    node_id: ProcessId,
    node: StandaloneNode,
    backend: Arc<dyn MutationStateBackend>,
    sequencer_admission: AdmissionController,
    // One dedicated writer thread owns ordering, canonical publication, and background WAL work.
    // This gate excludes snapshot recovery/compaction from an active publish/append boundary.
    standalone_gate: Arc<parking_lot::Mutex<()>>,
    standalone_wal: Arc<parking_lot::Mutex<DurableLog>>,
    persistence_tx: SyncSender<PersistenceMessage>,
    writer_tx: SyncSender<WriterMessage>,
    #[cfg(test)]
    eventual_durability_test_hook: Arc<EventualDurabilityTestHook>,
}

impl WriteRuntime {
    /// Standalone construction. Writes publish
    /// directly to the backend; the dedicated writer persists their ordered WAL suffix afterward.
    pub async fn start(
        root: impl AsRef<Path>,
        node: StandaloneNode,
        backend: Arc<dyn MutationStateBackend>,
        storage_limits: WriteStorageLimits,
    ) -> Result<Self> {
        let node_id = node.identity.node_id;
        node.validate_for(node_id)?;
        let storage_root = root.as_ref().to_owned();
        let maximum_log_record_bytes = storage_limits.max_log_record_bytes;
        let standalone_wal = tokio::task::spawn_blocking(move || {
            std::fs::create_dir_all(&storage_root)?;
            DurableLog::open(
                storage_root.join("standalone.wal"),
                maximum_log_record_bytes,
                Bookmark::default(),
            )
        })
        .await
        .map_err(|error| Error::internal(format!("WAL startup task failed: {error}")))??;
        let sequencer_bytes = storage_limits.max_log_record_bytes.saturating_mul(64);
        let sequencer_control_bytes = (sequencer_bytes / 16).max(1);
        let standalone_gate = Arc::new(parking_lot::Mutex::new(()));
        let standalone_wal = Arc::new(parking_lot::Mutex::new(standalone_wal));
        let (writer_tx, writer_rx) = mpsc::sync_channel(MAX_WRITE_BATCH);
        let (persistence_tx, persistence_rx) = mpsc::sync_channel(MAX_PENDING_WAL_BATCHES);
        let writer_backend = Arc::clone(&backend);
        let writer_gate = Arc::clone(&standalone_gate);
        let writer_wal = Arc::clone(&standalone_wal);
        let runtime = tokio::runtime::Handle::current();
        #[cfg(test)]
        let eventual_durability_test_hook = Arc::new(EventualDurabilityTestHook::default());
        #[cfg(test)]
        let writer_test_hook = Arc::clone(&eventual_durability_test_hook);
        std::thread::Builder::new()
            .name("irongraph-wal-persistence".to_owned())
            .spawn(move || {
                standalone_persistence_loop(
                    persistence_rx,
                    writer_wal,
                    #[cfg(test)]
                    writer_test_hook,
                );
            })
            .map_err(|error| {
                Error::internal(format!("failed to start WAL persistence: {error}"))
            })?;
        let writer_persistence = persistence_tx.clone();
        std::thread::Builder::new()
            .name("irongraph-wal-writer".to_owned())
            .spawn(move || {
                standalone_writer_loop(
                    writer_rx,
                    writer_persistence,
                    writer_backend,
                    writer_gate,
                    runtime,
                );
            })
            .map_err(|error| Error::internal(format!("failed to start WAL writer: {error}")))?;
        Ok(Self {
            node_id,
            node,
            backend,
            sequencer_admission: AdmissionController::new(AdmissionLimits {
                max_requests: 1_024,
                max_encoded_bytes: sequencer_bytes,
                reserved_control_requests: 64,
                reserved_control_bytes: sequencer_control_bytes,
                max_requests_per_connection: 64,
                max_encoded_bytes_per_connection: storage_limits
                    .max_log_record_bytes
                    .saturating_mul(16)
                    .min(sequencer_bytes),
                retry_after_ms: 25,
            })?,
            standalone_gate,
            standalone_wal,
            persistence_tx,
            writer_tx,
            #[cfg(test)]
            eventual_durability_test_hook,
        })
    }

    #[must_use]
    pub const fn node_id(&self) -> ProcessId {
        self.node_id
    }

    #[must_use]
    pub const fn node(&self) -> &StandaloneNode {
        &self.node
    }

    /// Accepts a mutation and preserves the original client connection for admission fairness.
    pub async fn write(&self, request: WriteRequest) -> Result<CommittedWrite> {
        validate_write_request(&request)?;
        let deadline = ClientDeadline::new(request.timeout_millis)?;

        // The dedicated writer owns ordering, apply, and subsequent WAL batching/fsync. Submitting
        // here is non-blocking; acknowledgement follows canonical publication, not filesystem I/O.
        let admission = self.admit_at_sequencer(&request, deadline)?;
        let (reply, response) = tokio::sync::oneshot::channel();
        let queued = QueuedWrite {
            command: Some(request.command),
            deadline,
            admission,
            command_prepared: false,
            reply,
        };
        match self.writer_tx.try_send(WriterMessage::Write(queued)) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                return Err(Error::retryable(
                    ErrorCode::WriteAdmissionFull,
                    "standalone writer queue is full",
                    Some(1),
                ));
            }
            Err(TrySendError::Disconnected(_)) => {
                return Err(Error::internal("standalone writer is unavailable"));
            }
        }
        tokio::time::timeout(deadline.remaining()?, response)
            .await
            .map_err(|_| client_deadline_exceeded("standalone write completion timed out"))?
            .map_err(|_| Error::internal("standalone writer stopped before replying"))?
    }

    /// Replays every WAL mutation above the backend's restored snapshot bookmark.
    ///
    /// This must run after snapshot installation and before listeners start accepting work.
    pub async fn replay_standalone_wal(&self) -> Result<Bookmark> {
        let _gate = self.standalone_gate.lock();
        let applied = self.backend.applied_bookmark().await;
        let (wal_base, wal_tail, wal_is_empty) = {
            let wal = self.standalone_wal.lock();
            (wal.compacted_through(), wal.last_bookmark(), wal.is_empty())
        };
        if wal_base.index > applied.index && wal_is_empty {
            // A previous snapshot loop could report success while snapshotting was skipped for
            // live broker segments, then compact this empty WAL beyond the actual snapshot. Those
            // un-snapshotted records cannot be recovered; restore the honest snapshot boundary so
            // the database can start and sources can idempotently refill their derived state.
            tracing::warn!(
                snapshot_index = applied.index,
                invalid_wal_base = wal_base.index,
                "repairing a standalone WAL base published without a snapshot"
            );
            self.standalone_wal
                .lock()
                .repair_empty_compacted_base(applied)?;
            return Ok(applied);
        }
        if wal_tail.index < applied.index {
            // A previous process that lost its WAL appender kept publishing writes and snapshots,
            // so the restored snapshot is ahead of the durable log tail. Every retained WAL entry
            // is included in that snapshot; realign by treating the log as compacted through it.
            tracing::warn!(
                snapshot_index = applied.index,
                wal_tail = wal_tail.index,
                "restored snapshot is newer than the standalone WAL; discarding the covered entries"
            );
            self.standalone_wal
                .lock()
                .discard_snapshot_covered_entries(applied)?;
            return Ok(applied);
        }
        let first = applied
            .index
            .checked_add(1)
            .ok_or_else(|| Error::new(ErrorCode::CorruptStorage, "applied bookmark overflow"))?;
        let entries = self.standalone_wal.lock().replay_from(first);
        let mut recovered = applied;
        for entry in entries {
            if entry.index() != recovered.index.saturating_add(1) {
                return Err(Error::new(
                    ErrorCode::CorruptStorage,
                    "standalone WAL does not continue the restored snapshot",
                ));
            }
            if let Err(error) = self.backend.apply_mutation(&entry).await {
                // Older standalone builds prepared concurrent graph writes outside the write gate
                // and could fsync a transaction that the state-dependent preflight then rejected.
                // Such an entry was never visible or acknowledged, and no later index could be
                // appended because the backend bookmark did not advance. Remove that rejected
                // suffix so startup can recover the last actually committed prefix.
                if error.code == ErrorCode::TransactionConflict {
                    tracing::warn!(
                        index = entry.index(),
                        "discarding an uncommitted standalone WAL tail rejected during recovery"
                    );
                    self.standalone_wal
                        .lock()
                        .truncate_uncommitted(recovered.index)?;
                    return Ok(recovered);
                }
                return Err(error);
            }
            recovered = entry.bookmark();
        }
        if recovered != wal_tail {
            return Err(Error::new(
                ErrorCode::CorruptStorage,
                "standalone WAL tail was not fully replayed",
            ));
        }
        Ok(recovered)
    }

    /// Drops WAL records included in an atomically published snapshot.
    pub fn compact_wal_through(&self, included: Bookmark) -> Result<()> {
        let _gate = self.standalone_gate.lock();
        let (flush, completed) = mpsc::sync_channel(1);
        self.persistence_tx
            .send(PersistenceMessage::Flush(flush))
            .map_err(|_| Error::internal("WAL persistence worker is unavailable"))?;
        completed
            .recv()
            .map_err(|_| Error::internal("WAL persistence flush response was lost"))??;
        let mut wal = self.standalone_wal.lock();
        wal.compact_prefix(included)
    }

    fn admit_at_sequencer(
        &self,
        request: &WriteRequest,
        deadline: ClientDeadline,
    ) -> Result<crate::storage::AdmissionPermit> {
        let encoded_bytes = request
            .command
            .payload
            .len()
            .checked_add(128)
            .ok_or_else(|| Error::invalid_data("sequencer admission size overflow"))?;
        self.sequencer_admission.try_admit(
            request.connection_id,
            request.admission_class,
            encoded_bytes,
            std::time::Instant::now() + deadline.remaining()?,
        )
    }

    #[cfg(test)]
    async fn flush_wal(&self) -> Result<()> {
        let (reply, response) = tokio::sync::oneshot::channel();
        self.writer_tx
            .try_send(WriterMessage::Flush(reply))
            .map_err(|error| Error::internal(format!("failed to request WAL flush: {error}")))?;
        response
            .await
            .map_err(|_| Error::internal("standalone writer stopped before WAL flush completed"))?
    }

    /// Drains every earlier write message and establishes a final WAL durability boundary before
    /// the dedicated persistence thread exits.
    pub async fn shutdown(&self) -> Result<()> {
        let (reply, response) = tokio::sync::oneshot::channel();
        let sender = self.writer_tx.clone();
        tokio::task::spawn_blocking(move || sender.send(WriterMessage::Shutdown(reply)))
            .await
            .map_err(|error| Error::internal(format!("WAL shutdown send task failed: {error}")))?
            .map_err(|_| Error::internal("standalone writer is unavailable during shutdown"))?;
        response
            .await
            .map_err(|_| Error::internal("standalone writer stopped before shutdown completed"))?
    }
}

fn validate_write_request(request: &WriteRequest) -> Result<()> {
    request.command.validate()?;
    request_timeout(request.timeout_millis)?;
    if !request.connection_id.is_valid() {
        return Err(Error::invalid_data(
            "write request is missing its origin connection identity",
        ));
    }
    if let Some(fence) = request.transaction_fence
        && (fence.sequencer.0.is_nil() || fence.term == 0 || fence.snapshot.term > fence.term)
    {
        return Err(Error::invalid_data("explicit transaction fence is invalid"));
    }
    Ok(())
}

fn request_timeout(millis: u64) -> Result<Duration> {
    if millis == 0 || millis > MAX_CLIENT_WAIT_MILLIS {
        return Err(Error::invalid_data(
            "client consistency timeout is out of range",
        ));
    }
    Ok(Duration::from_millis(millis))
}

#[derive(Clone, Copy)]
struct ClientDeadline {
    at: tokio::time::Instant,
}

impl ClientDeadline {
    fn new(timeout_millis: u64) -> Result<Self> {
        Ok(Self {
            at: tokio::time::Instant::now() + request_timeout(timeout_millis)?,
        })
    }

    fn remaining(self) -> Result<Duration> {
        self.at
            .checked_duration_since(tokio::time::Instant::now())
            .filter(|remaining| !remaining.is_zero())
            .ok_or_else(|| client_deadline_exceeded("client deadline exhausted"))
    }
}

fn client_deadline_exceeded(message: &'static str) -> Error {
    Error::retryable(ErrorCode::DeadlineExceeded, message, None)
}
