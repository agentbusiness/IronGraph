use std::{
    collections::{BTreeMap, BTreeSet, LinkedList, VecDeque},
    sync::Arc,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use crate::storage::{
    AdmissionClass, AdmissionController, AdmissionPermit, ConnectionId, MutationEntry, RecordBatch,
    RecordOp,
};
use crate::{Bookmark, Error, ErrorCode, Result};

use super::{
    ApplicationWait, CommandReservation, CommandReservationOutcome, CommittedWrite,
    MutationStateBackend, ProcessId, StandaloneNode, WriteCommand, WriteRequest, WriteResponse,
};

// Keep the no-delay single-write path, but do not impose an artificial throughput ceiling when
// thousands of already-admitted independent writes arrive together. This bound remains finite,
// every queued payload already owns an admission permit, and a graph backend may end a batch
// earlier when its ordered mutation overlay reaches capacity.
const MAX_WRITE_BATCH: usize = 8_192;
const PERSISTENCE_GROUP_INTERVAL: Duration = Duration::from_secs(1);
const PERSISTENCE_BATCH_TARGET_BYTES: usize = 4 * 1024 * 1024;

struct QueuedWrite {
    command: Option<WriteCommand>,
    deadline: ClientDeadline,
    admission: AdmissionPermit,
    command_prepared: bool,
    reply: tokio::sync::oneshot::Sender<Result<CommittedWrite>>,
}

enum WriterMessage {
    Write(QueuedWrite),
    Flush(tokio::sync::oneshot::Sender<Result<()>>),
    Shutdown(tokio::sync::oneshot::Sender<Result<()>>),
}

type ApplicationFailure = Arc<parking_lot::Mutex<Option<Error>>>;

#[derive(Default)]
struct PersistenceRecords {
    batches: Vec<RecordBatch>,
}

#[derive(Default)]
struct DirtyRecords {
    first_index: u64,
    bookmark: Bookmark,
    points: BTreeMap<Vec<u8>, Option<Vec<u8>>>,
    prefixes: BTreeSet<Vec<u8>>,
}

impl DirtyRecords {
    fn absorb(&mut self, batch: RecordBatch) {
        if self.first_index == 0 {
            self.first_index = batch.first_index;
        }
        self.bookmark = batch.bookmark;
        for operation in batch.operations {
            match operation {
                RecordOp::Put { key, value } => {
                    self.points.insert(key, Some(value));
                }
                RecordOp::Delete { key } => {
                    self.points.insert(key, None);
                }
                RecordOp::DeletePrefix { prefix } => {
                    let removed = self
                        .points
                        .range(prefix.clone()..)
                        .take_while(|(key, _)| key.starts_with(&prefix))
                        .map(|(key, _)| key.clone())
                        .collect::<Vec<_>>();
                    for key in removed {
                        self.points.remove(&key);
                    }
                    if !self
                        .prefixes
                        .range(..=prefix.clone())
                        .next_back()
                        .is_some_and(|previous| prefix.starts_with(previous))
                    {
                        let removed = self
                            .prefixes
                            .range(prefix.clone()..)
                            .take_while(|previous| previous.starts_with(&prefix))
                            .cloned()
                            .collect::<Vec<_>>();
                        for previous in removed {
                            self.prefixes.remove(&previous);
                        }
                        self.prefixes.insert(prefix);
                    }
                }
            }
        }
    }

    fn into_batch(self) -> RecordBatch {
        RecordBatch {
            first_index: self.first_index,
            bookmark: self.bookmark,
            operations: self
                .prefixes
                .into_iter()
                .map(|prefix| RecordOp::DeletePrefix { prefix })
                .chain(self.points.into_iter().map(|(key, value)| match value {
                    Some(value) => RecordOp::Put { key, value },
                    None => RecordOp::Delete { key },
                }))
                .collect(),
        }
    }
}

type DurabilityReply = tokio::sync::oneshot::Sender<Result<()>>;
struct PersistenceState {
    pending: Option<DirtyRecords>,
    deadline: Option<Instant>,
    published: u64,
    durable: u64,
    waiters: LinkedList<(u64, DurabilityReply)>,
    force: bool,
    closing: bool,
    error: Option<Error>,
    publication_parked: bool,
}

struct PersistenceMailbox {
    state: parking_lot::Mutex<PersistenceState>,
    wake: parking_lot::Condvar,
    #[cfg(test)]
    publication_pause: parking_lot::Mutex<Option<Arc<PublicationPause>>>,
    #[cfg(test)]
    parked: tokio::sync::Notify,
    #[cfg(test)]
    published: tokio::sync::Notify,
}

#[cfg(test)]
#[derive(Default)]
struct PublicationPause {
    entered: tokio::sync::Notify,
    released: parking_lot::Mutex<bool>,
    resume: parking_lot::Condvar,
}

#[cfg(test)]
impl PublicationPause {
    fn release(&self) {
        *self.released.lock() = true;
        self.resume.notify_all();
    }
}

impl PersistenceMailbox {
    fn new(bookmark: Bookmark) -> Self {
        Self {
            state: parking_lot::Mutex::new(PersistenceState {
                pending: None,
                deadline: None,
                published: bookmark.index,
                durable: bookmark.index,
                waiters: LinkedList::new(),
                force: false,
                closing: false,
                error: None,
                publication_parked: false,
            }),
            wake: parking_lot::Condvar::new(),
            #[cfg(test)]
            publication_pause: parking_lot::Mutex::new(None),
            #[cfg(test)]
            parked: tokio::sync::Notify::new(),
            #[cfg(test)]
            published: tokio::sync::Notify::new(),
        }
    }

    fn publish(&self, records: PersistenceRecords) -> Result<()> {
        if records.batches.is_empty() {
            return Ok(());
        }
        let published = self.state.lock().published;
        let mut previous = published;
        for batch in &records.batches {
            if previous.checked_add(1) != Some(batch.first_index)
                || batch.bookmark.index < batch.first_index
            {
                return Err(Error::internal(
                    "noncontiguous current-record publication span",
                ));
            }
            previous = batch.bookmark.index;
        }
        // Only the ordered RAM writer merges. Swap roots under the mailbox lock, then perform
        // all key traversal, allocation and obsolete-value destruction outside the lock.
        let mut dirty = self.state.lock().pending.take().unwrap_or_default();
        #[cfg(test)]
        if let Some(pause) = self.publication_pause.lock().take() {
            pause.entered.notify_one();
            let mut released = pause.released.lock();
            while !*released {
                pause.resume.wait(&mut released);
            }
        }
        for batch in records.batches {
            dirty.absorb(batch);
        }
        let published = dirty.bookmark.index;
        let mut state = self.state.lock();
        state.pending = Some(dirty);
        state.published = published;
        let notify = state.error.is_none()
            && (state.deadline.is_none()
                || state.publication_parked
                || state.force
                || state
                    .deadline
                    .is_some_and(|deadline| Instant::now() >= deadline));
        if state.error.is_none() {
            state
                .deadline
                .get_or_insert_with(|| Instant::now() + PERSISTENCE_GROUP_INTERVAL);
        }
        drop(state);
        #[cfg(test)]
        self.published.notify_one();
        if notify {
            self.wake.notify_one();
        }
        Ok(())
    }

    fn barrier(&self, reply: Option<DurabilityReply>, closing: bool) {
        let mut state = self.state.lock();
        let target = state.published;
        if let Some(reply) = reply {
            if target <= state.durable {
                let _ = reply.send(Ok(()));
            } else {
                state.waiters.push_back((target, reply));
            }
        }
        state.closing |= closing;
        let wake = closing || target > state.durable;
        state.force |= wake; // A barrier that needs durability explicitly retries IO failures.
        drop(state);
        if wake {
            self.wake.notify_one();
        }
    }
}

struct PreparedWrite {
    mutation: MutationEntry,
    reservation: CommandReservation,
    request_id: Option<uuid::Uuid>,
    admission: AdmissionPermit,
    reply: tokio::sync::oneshot::Sender<Result<CommittedWrite>>,
}

fn standalone_writer_loop(
    mut receiver: tokio::sync::mpsc::Receiver<WriterMessage>,
    persistence: Arc<PersistenceMailbox>,
    backend: Arc<dyn MutationStateBackend>,
    gate: Arc<parking_lot::Mutex<()>>,
    runtime: tokio::runtime::Handle,
    failure: ApplicationFailure,
) {
    let mut pending = VecDeque::new();
    let mut deferred_control = None;
    loop {
        if pending.is_empty() {
            let received = if let Some(message) = deferred_control.take() {
                Some(message)
            } else {
                receiver.blocking_recv()
            };
            match received {
                Some(WriterMessage::Write(write)) => pending.push_back(write),
                Some(WriterMessage::Flush(reply)) => {
                    persistence.barrier(Some(reply), false);
                    continue;
                }
                Some(WriterMessage::Shutdown(reply)) => {
                    persistence.barrier(Some(reply), true);
                    return;
                }
                None => {
                    persistence.barrier(None, true);
                    return;
                }
            }
        }
        while pending.len() < MAX_WRITE_BATCH && deferred_control.is_none() {
            match receiver.try_recv() {
                Ok(WriterMessage::Write(item)) => pending.push_back(item),
                Ok(control) => {
                    deferred_control = Some(control);
                    break;
                }
                Err(
                    tokio::sync::mpsc::error::TryRecvError::Empty
                    | tokio::sync::mpsc::error::TryRecvError::Disconnected,
                ) => break,
            }
        }
        match process_writer_batch(
            &mut receiver,
            &mut deferred_control,
            &mut pending,
            &backend,
            &gate,
            &runtime,
            &failure,
        ) {
            Ok(records) => {
                if let Err(error) = persistence.publish(records) {
                    *failure.lock() = Some(error);
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
    mailbox: Arc<PersistenceMailbox>,
    backend: Arc<dyn MutationStateBackend>,
) {
    let mut retained: Option<DirtyRecords> = None;
    loop {
        let pending = {
            let mut state = mailbox.state.lock();
            loop {
                if state.closing && state.pending.is_none() && retained.is_none() {
                    return;
                }
                let retry_allowed = state.error.is_none() || state.force;
                let due = state.force
                    || state
                        .deadline
                        .is_some_and(|deadline| Instant::now() >= deadline);
                if retry_allowed && due && (state.pending.is_some() || retained.is_some()) {
                    state.force = false;
                    state.deadline = None;
                    break state.pending.take();
                }
                if state.error.is_none()
                    && state.pending.is_some()
                    && let Some(deadline) = state.deadline
                {
                    mailbox.wake.wait_for(
                        &mut state,
                        deadline.saturating_duration_since(Instant::now()),
                    );
                } else {
                    // A producer may temporarily own the pending root while merging outside
                    // this lock. Arm the wake handshake before atomically unlocking to park.
                    state.publication_parked = state.error.is_none() && state.pending.is_none();
                    #[cfg(test)]
                    if state.publication_parked {
                        mailbox.parked.notify_one();
                    }
                    mailbox.wake.wait(&mut state);
                    state.publication_parked = false;
                }
            }
        };
        let mut dirty = retained.take().unwrap_or_default();
        if let Some(pending) = pending {
            dirty.absorb(pending.into_batch());
        }
        let batch = dirty.into_batch();
        let bookmark = batch.bookmark;
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            backend.persist(std::slice::from_ref(&batch))
        }))
        .unwrap_or_else(|_| Err(Error::internal("persistence backend panicked")));
        if let Err(error) = &result {
            tracing::error!(code = ?error.code, message = %error.message,
                "background persistence failed; current records retained for explicit flush or close retry; RAM publication continues");
        }
        let mut state = mailbox.state.lock();
        match result {
            Ok(()) => {
                state.durable = bookmark.index;
                state.error = None;
                let waiters = std::mem::take(&mut state.waiters);
                let durable = state.durable;
                drop(state);
                let mut remaining = LinkedList::new();
                for (target, reply) in waiters {
                    if target <= durable {
                        let _ = reply.send(Ok(()));
                    } else {
                        remaining.push_back((target, reply));
                    }
                }
                let mut state = mailbox.state.lock();
                state.waiters.append(&mut remaining);
                if state.closing || !state.waiters.is_empty() {
                    state.force = true;
                }
            }
            Err(error) => {
                state.error = Some(copy_error(&error));
                state.force = false;
                state.deadline = None;
                let waiters = std::mem::take(&mut state.waiters);
                let closing = state.closing;
                drop(state);
                let mut dirty = DirtyRecords::default();
                dirty.absorb(batch);
                retained = Some(dirty);
                for (_, reply) in waiters {
                    let _ = reply.send(Err(copy_error(&error)));
                }
                if closing {
                    return;
                }
            }
        }
    }
}

fn process_writer_batch(
    receiver: &mut tokio::sync::mpsc::Receiver<WriterMessage>,
    deferred_control: &mut Option<WriterMessage>,
    pending: &mut VecDeque<QueuedWrite>,
    backend: &Arc<dyn MutationStateBackend>,
    gate: &Arc<parking_lot::Mutex<()>>,
    runtime: &tokio::runtime::Handle,
    failure: &ApplicationFailure,
) -> Result<PersistenceRecords> {
    if let Some(error) = failure.lock().as_ref() {
        return Err(copy_error(error));
    }
    let Some(first_deadline) = pending.front().map(|item| item.deadline) else {
        return Ok(PersistenceRecords::default());
    };
    let Ok(remaining) = first_deadline.remaining() else {
        if let Some(item) = pending.pop_front() {
            let _ = item.reply.send(Err(client_deadline_exceeded(
                "standalone write expired in the writer queue",
            )));
        }
        return Ok(PersistenceRecords::default());
    };
    let batch_gate = match remaining {
        Some(remaining) => gate.try_lock_for(remaining),
        None => Some(gate.lock()),
    };
    let Some(_batch_gate) = batch_gate else {
        if let Some(item) = pending.pop_front() {
            let _ = item.reply.send(Err(client_deadline_exceeded(
                "standalone write admission timed out",
            )));
        }
        return Ok(PersistenceRecords::default());
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
            return Ok(PersistenceRecords::default());
        }
    };
    let mut prepared = Vec::with_capacity(pending.len().min(MAX_WRITE_BATCH));
    let mut prepared_bytes = 0_usize;
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
        let pipelined = reservation.may_pipeline();
        prepared_bytes = prepared_bytes.saturating_add(mutation.payload().len());
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
        if !pipelined || prepared_bytes >= PERSISTENCE_BATCH_TARGET_BYTES {
            break;
        }
        // Preparation/reservation is often long enough for concurrent callers to arrive. Pull
        // them into this group before deciding the batch is complete; no artificial timer is
        // added to the single-write latency path.
        while pending.len() + prepared.len() < MAX_WRITE_BATCH && deferred_control.is_none() {
            match receiver.try_recv() {
                Ok(WriterMessage::Write(item)) => pending.push_back(item),
                Ok(control) => {
                    *deferred_control = Some(control);
                    break;
                }
                Err(
                    tokio::sync::mpsc::error::TryRecvError::Empty
                    | tokio::sync::mpsc::error::TryRecvError::Disconnected,
                ) => break,
            }
        }
    }
    if prepared.is_empty() {
        return Ok(PersistenceRecords::default());
    }

    // Canonical publication is the commit boundary. Replies are delivered immediately afterward;
    // Record commits and filesystem synchronization run only on the persistence thread. Capture
    // each mutation's records at publication, never by rescanning a later live graph view.
    let mut records = PersistenceRecords {
        batches: Vec::with_capacity(prepared.len()),
    };
    let mut fatal = None;
    for write in prepared {
        let PreparedWrite {
            mutation,
            reservation,
            request_id,
            admission,
            reply,
        } = write;
        let failed = failure.lock().as_ref().map(copy_error);
        let result = if let Some(error) = fatal.as_ref().or(failed.as_ref()) {
            let _ = runtime.block_on(
                backend
                    .complete_command_reservation(reservation, CommandReservationOutcome::Rejected),
            );
            Err(copy_error(error))
        } else {
            match runtime.block_on(backend.apply_mutation(&mutation)) {
                Ok(applied) => {
                    let bookmark = mutation.bookmark();
                    {
                        let Some(batch) = applied.persistence.filter(|batch| {
                            batch.bookmark == bookmark && batch.first_index == bookmark.index
                        }) else {
                            let error = Error::internal(
                                "applied mutation omitted its current-record span or returned a mismatched span",
                            );
                            *failure.lock() = Some(copy_error(&error));
                            let _ = runtime.block_on(backend.complete_command_reservation(
                                reservation,
                                CommandReservationOutcome::Applied,
                            ));
                            fatal = Some(copy_error(&error));
                            drop(admission);
                            let _ = reply.send(Err(error));
                            continue;
                        };
                        records.batches.push(batch);
                    }
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
                    *failure.lock() = Some(copy_error(&error));
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

    Ok(records)
}

fn copy_error(error: &Error) -> Error {
    Error {
        code: error.code,
        message: error.message.clone(),
        retryable: error.retryable,
        retry_after_ms: error.retry_after_ms,
    }
}

/// Exact term/index reserved by the local sequencer for a generated control write.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SequencerPosition {
    pub term: u64,
    pub index: u64,
}

/// Running standalone ordered publication and asynchronous record persistence.
pub struct WriteRuntime {
    node_id: ProcessId,
    node: StandaloneNode,
    sequencer_admission: AdmissionController,
    failure: ApplicationFailure,
    persistence: Arc<PersistenceMailbox>,
    writer_tx: tokio::sync::mpsc::Sender<WriterMessage>,
    workers: parking_lot::Mutex<Option<(std::thread::JoinHandle<()>, std::thread::JoinHandle<()>)>>,
}

impl WriteRuntime {
    /// Start dedicated ordered application and persistence threads. The backend must already be
    /// restored before this call, so the first allocated position follows its durable state.
    pub async fn start(
        node: StandaloneNode,
        backend: Arc<dyn MutationStateBackend>,
    ) -> Result<Self> {
        let node_id = node.identity.node_id;
        node.validate_for(node_id)?;
        let gate = Arc::new(parking_lot::Mutex::new(()));
        let failure = Arc::new(parking_lot::Mutex::new(None));
        let (writer_tx, writer_rx) = tokio::sync::mpsc::channel(MAX_WRITE_BATCH);
        let persistence = Arc::new(PersistenceMailbox::new(backend.applied_bookmark().await));
        let persistence_backend = Arc::clone(&backend);
        let persistence_mailbox = Arc::clone(&persistence);
        let persistence_worker = std::thread::Builder::new()
            .name("irongraph-persistence".to_owned())
            .spawn(move || standalone_persistence_loop(persistence_mailbox, persistence_backend))
            .map_err(|error| Error::internal(format!("failed to start persistence: {error}")))?;
        let writer_persistence = Arc::clone(&persistence);
        let writer_failure = Arc::clone(&failure);
        let runtime = tokio::runtime::Handle::current();
        let writer_worker = match std::thread::Builder::new()
            .name("irongraph-writer".to_owned())
            .spawn(move || {
                standalone_writer_loop(
                    writer_rx,
                    writer_persistence,
                    backend,
                    gate,
                    runtime,
                    writer_failure,
                )
            }) {
            Ok(worker) => worker,
            Err(error) => {
                persistence.barrier(None, true);
                let _ = persistence_worker.join();
                return Err(Error::internal(format!("failed to start writer: {error}")));
            }
        };
        Ok(Self {
            node_id,
            node,
            sequencer_admission: AdmissionController::unrestricted(),
            failure,
            persistence,
            writer_tx,
            workers: parking_lot::Mutex::new(Some((writer_worker, persistence_worker))),
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
        if let Some(error) = self.failure.lock().as_ref() {
            return Err(copy_error(error));
        }
        validate_write_request(&request)?;
        let deadline = ClientDeadline::new(request.timeout_millis)?;

        // The dedicated writer owns ordering and apply; record commits run on its sibling thread. Submitting
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
        self.writer_tx
            .send(WriterMessage::Write(queued))
            .await
            .map_err(|_| Error::internal("standalone writer is unavailable"))?;
        let remaining = deadline.remaining().map_err(|_| {
            Error::new(
                ErrorCode::DeadlineExceeded,
                "write was dispatched; its outcome is uncertain; do not retry automatically",
            )
        })?;
        let completed = match remaining {
            Some(remaining) => tokio::time::timeout(remaining, response)
                .await.map_err(|_| Error::new(ErrorCode::DeadlineExceeded,
                    "write completion timed out after dispatch; its outcome is uncertain; do not retry automatically"))?,
            None => response.await,
        };
        completed
            .map_err(|_| Error::internal("writer stopped after dispatch; write outcome is uncertain; do not retry automatically"))?
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
        self.sequencer_admission.try_admit_until(
            request.connection_id,
            request.admission_class,
            encoded_bytes,
            deadline
                .remaining()?
                .map(|remaining| std::time::Instant::now() + remaining),
        )
    }

    /// Await durability of all earlier queued writes while later RAM publications continue.
    /// After an IO failure this explicitly retries retained current records.
    pub async fn flush_persistence(&self) -> Result<()> {
        let (reply, response) = tokio::sync::oneshot::channel();
        self.writer_tx
            .send(WriterMessage::Flush(reply))
            .await
            .map_err(|error| {
                Error::internal(format!("failed to request persistence flush: {error}"))
            })?;
        response.await.map_err(|_| {
            Error::internal("standalone writer stopped before persistence flush completed")
        })?
    }

    /// Drains every earlier write message and establishes a final record durability boundary before
    /// the dedicated persistence thread exits.
    pub async fn shutdown(&self) -> Result<()> {
        let Some(workers) = self.workers.lock().take() else {
            return Ok(());
        };
        let (reply, response) = tokio::sync::oneshot::channel();
        let sender = self.writer_tx.clone();
        let sent = sender
            .send(WriterMessage::Shutdown(reply))
            .await
            .map_err(|_| Error::internal("standalone writer is unavailable during shutdown"));
        let completed = response
            .await
            .map_err(|_| Error::internal("standalone writer stopped before shutdown completed"));
        let persistence = Arc::clone(&self.persistence);
        let joined = tokio::task::spawn_blocking(move || join_workers(workers, &persistence))
            .await
            .map_err(|error| {
                Error::internal(format!("persistence worker join task failed: {error}"))
            })?;
        sent?;
        completed??;
        joined
    }
}

fn join_workers(
    workers: (std::thread::JoinHandle<()>, std::thread::JoinHandle<()>),
    persistence: &PersistenceMailbox,
) -> Result<()> {
    let writer = workers.0.join();
    // A panicked writer cannot send the normal shutdown message. Ensure its sibling is also
    // stopped before joining. Closing the mailbox is idempotent if its worker already exited.
    persistence.barrier(None, true);
    let persistence = workers.1.join();
    writer.map_err(|_| Error::internal("ordered writer panicked"))?;
    persistence.map_err(|_| Error::internal("persistence worker panicked"))?;
    Ok(())
}

impl Drop for WriteRuntime {
    fn drop(&mut self) {
        if let Some(workers) = self.workers.get_mut().take() {
            let writer = self.writer_tx.clone();
            let persistence = Arc::clone(&self.persistence);
            // Explicit shutdown is the observable durability boundary. Implicit destruction must
            // not block a Tokio worker on disk synchronization or worker joins.
            let _ = std::thread::Builder::new()
                .name("irongraph-persistence-close".to_owned())
                .spawn(move || {
                    let (reply, _response) = tokio::sync::oneshot::channel();
                    let _ = writer.blocking_send(WriterMessage::Shutdown(reply));
                    let _ = join_workers(workers, &persistence);
                });
        }
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

fn request_timeout(millis: u64) -> Result<Option<Duration>> {
    Ok((millis != 0).then(|| Duration::from_millis(millis)))
}

#[derive(Clone, Copy)]
struct ClientDeadline {
    at: Option<tokio::time::Instant>,
}

impl ClientDeadline {
    fn new(timeout_millis: u64) -> Result<Self> {
        Ok(Self {
            at: request_timeout(timeout_millis)?
                .map(|duration| {
                    tokio::time::Instant::now()
                        .checked_add(duration)
                        .ok_or_else(|| Error::invalid_data("client deadline exceeds clock range"))
                })
                .transpose()?,
        })
    }

    fn remaining(self) -> Result<Option<Duration>> {
        let Some(at) = self.at else {
            return Ok(None);
        };
        at.checked_duration_since(tokio::time::Instant::now())
            .filter(|remaining| !remaining.is_zero())
            .map(Some)
            .ok_or_else(|| client_deadline_exceeded("client deadline exhausted"))
    }
}

fn client_deadline_exceeded(message: &'static str) -> Error {
    Error::retryable(ErrorCode::DeadlineExceeded, message, None)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

    use async_trait::async_trait;

    use super::*;
    use crate::engine::{ExecutionClass, MutationApplyResult, NodeIdentity};
    use crate::storage::MutationKind;

    #[derive(Default)]
    struct TestBackend {
        applied: AtomicU64,
        durable: AtomicU64,
        reservations: AtomicUsize,
        first_apply_reservations: AtomicUsize,
        persistence_calls: AtomicUsize,
        persistence_attempts: AtomicUsize,
        last_value: AtomicU64,
        reached: AtomicBool,
        entered: tokio::sync::Notify,
        committed: tokio::sync::Notify,
        fail: AtomicBool,
        paused: parking_lot::Mutex<bool>,
        resume: parking_lot::Condvar,
    }

    impl TestBackend {
        fn release(&self) {
            *self.paused.lock() = false;
            self.resume.notify_all();
        }
    }

    struct ReleaseOnDrop(Arc<TestBackend>);
    impl Drop for ReleaseOnDrop {
        fn drop(&mut self) {
            self.0.release();
        }
    }

    struct ResumePublication(Arc<PublicationPause>);
    impl Drop for ResumePublication {
        fn drop(&mut self) {
            self.0.release();
        }
    }

    #[async_trait]
    impl MutationStateBackend for TestBackend {
        async fn prepare_command(
            &self,
            mut command: WriteCommand,
            sequencer_time_millis: i64,
        ) -> Result<WriteCommand> {
            // Let independent submissions accumulate before the first publication.
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
            if previous == 0 {
                self.first_apply_reservations
                    .store(self.reservations.load(Ordering::Acquire), Ordering::Release);
            }
            Ok(MutationApplyResult {
                persistence: Some(RecordBatch {
                    first_index: mutation.index(),
                    bookmark: mutation.bookmark(),
                    operations: vec![RecordOp::Put {
                        key: b"current".to_vec(),
                        value: mutation.index().to_be_bytes().to_vec(),
                    }],
                }),
                ..MutationApplyResult::default()
            })
        }

        async fn applied_bookmark(&self) -> Bookmark {
            Bookmark {
                term: 1,
                index: self.applied.load(Ordering::Acquire),
            }
        }

        fn persist(&self, batches: &[RecordBatch]) -> Result<()> {
            self.persistence_attempts.fetch_add(1, Ordering::AcqRel);
            if tokio::runtime::Handle::try_current().is_ok() {
                return Err(Error::internal(
                    "persistence ran on an async runtime thread",
                ));
            }
            self.reached.store(true, Ordering::Release);
            self.entered.notify_one();
            let mut paused = self.paused.lock();
            while *paused {
                self.resume.wait(&mut paused);
            }
            if self.fail.load(Ordering::Acquire) {
                return Err(Error::internal("injected persistence failure"));
            }
            let mut expected = self.durable.load(Ordering::Acquire) + 1;
            for batch in batches {
                if batch.first_index != expected {
                    return Err(Error::internal(
                        "persistence did not receive an ordered prefix",
                    ));
                }
                expected = batch.bookmark.index + 1;
            }
            self.durable.store(expected - 1, Ordering::Release);
            for batch in batches {
                for operation in &batch.operations {
                    if let RecordOp::Put { key, value } = operation
                        && key == b"current"
                    {
                        self.last_value.store(
                            u64::from_be_bytes(
                                value
                                    .as_slice()
                                    .try_into()
                                    .map_err(|_| Error::internal("invalid current value"))?,
                            ),
                            Ordering::Release,
                        );
                    }
                }
            }
            self.persistence_calls.fetch_add(1, Ordering::AcqRel);
            self.committed.notify_one();
            Ok(())
        }
    }

    fn request() -> WriteRequest {
        WriteRequest {
            command: WriteCommand {
                kind: MutationKind::Graph,
                project_id: Some(crate::ProjectId::random()),
                request_id: Some(uuid::Uuid::new_v4()),
                commit_time_millis: 0,
                payload: vec![1],
            },
            timeout_millis: 5_000,
            connection_id: ConnectionId::new(),
            admission_class: AdmissionClass::Client,
            transaction_fence: None,
        }
    }

    async fn start(backend: Arc<TestBackend>) -> Result<WriteRuntime> {
        WriteRuntime::start(
            StandaloneNode {
                identity: NodeIdentity::generate_genesis().public(),
                execution_class: ExecutionClass::Cpu,
            },
            backend,
        )
        .await
    }

    async fn await_reached(backend: &TestBackend) -> Result<()> {
        if backend.reached.load(Ordering::Acquire) {
            return Ok(());
        }
        tokio::time::timeout(Duration::from_secs(2), backend.entered.notified())
            .await
            .map_err(|_| Error::internal("persistence worker never reached the test gate"))
    }

    async fn await_record_publication(runtime: &WriteRuntime, target: u64) -> Result<()> {
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                // Register before checking the mailbox so publication cannot fall between
                // checking the target and waiting for its notification.
                let published = runtime.persistence.published.notified();
                tokio::pin!(published);
                published.as_mut().enable();
                if runtime.persistence.state.lock().published >= target {
                    return;
                }
                published.await;
            }
        })
        .await
        .map_err(|_| Error::internal("writer never published its current persistence records"))
    }

    #[tokio::test(flavor = "current_thread")]
    async fn acknowledgements_and_async_progress_do_not_wait_for_blocked_persistence() -> Result<()>
    {
        let backend = Arc::new(TestBackend::default());
        *backend.paused.lock() = true;
        let _release = ReleaseOnDrop(Arc::clone(&backend));
        let runtime = Arc::new(start(Arc::clone(&backend)).await?);
        assert_eq!(runtime.write(request()).await?.response.bookmark.index, 1);
        await_reached(&backend).await?;
        // The same single async worker can publish another graph mutation while disk is blocked.
        assert_eq!(runtime.write(request()).await?.response.bookmark.index, 2);
        let flush_runtime = Arc::clone(&runtime);
        let flush = tokio::spawn(async move { flush_runtime.flush_persistence().await });
        let progress = tokio::spawn(async {
            tokio::task::yield_now().await;
            7
        });
        assert_eq!(
            progress
                .await
                .map_err(|error| Error::internal(error.to_string()))?,
            7
        );
        assert!(!flush.is_finished());
        assert_eq!(backend.durable.load(Ordering::Acquire), 0);
        assert_eq!(
            runtime
                .persistence
                .state
                .lock()
                .pending
                .as_ref()
                .map(|dirty| dirty.points.len()),
            Some(1)
        );
        backend.release();
        flush
            .await
            .map_err(|error| Error::internal(error.to_string()))??;
        assert_eq!(backend.durable.load(Ordering::Acquire), 2);
        runtime.shutdown().await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn ordered_batches_preserve_all_writes_and_shutdown_drains_them() -> Result<()> {
        const WRITES: usize = 257;
        let backend = Arc::new(TestBackend::default());
        let runtime = Arc::new(start(Arc::clone(&backend)).await?);
        let mut tasks = Vec::new();
        for _ in 0..WRITES {
            let runtime = Arc::clone(&runtime);
            tasks.push(tokio::spawn(async move { runtime.write(request()).await }));
        }
        let mut positions = std::collections::BTreeSet::new();
        for task in tasks {
            positions.insert(
                task.await
                    .map_err(|error| Error::internal(error.to_string()))??
                    .response
                    .bookmark
                    .index,
            );
        }
        assert_eq!(positions.len(), WRITES);
        assert_eq!(positions.last(), Some(&(WRITES as u64)));
        runtime.shutdown().await?;
        assert_eq!(backend.durable.load(Ordering::Acquire), WRITES as u64);
        assert!(backend.first_apply_reservations.load(Ordering::Acquire) > 1);
        assert!(backend.persistence_calls.load(Ordering::Acquire) < WRITES);
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn persistence_failure_retains_changes_and_explicit_flush_retries() -> Result<()> {
        let backend = Arc::new(TestBackend::default());
        backend.fail.store(true, Ordering::Release);
        let runtime = start(Arc::clone(&backend)).await?;
        runtime.write(request()).await?;
        assert!(runtime.flush_persistence().await.is_err());
        runtime.write(request()).await?;
        assert_eq!(backend.applied.load(Ordering::Acquire), 2);
        assert_eq!(backend.durable.load(Ordering::Acquire), 0);
        assert!(runtime.persistence.state.lock().deadline.is_none());
        tokio::time::sleep(PERSISTENCE_GROUP_INTERVAL + Duration::from_millis(25)).await;
        assert_eq!(backend.persistence_attempts.load(Ordering::Acquire), 1);
        backend.fail.store(false, Ordering::Release);
        runtime.flush_persistence().await?;
        assert_eq!(backend.durable.load(Ordering::Acquire), 2);
        assert_eq!(backend.last_value.load(Ordering::Acquire), 2);
        runtime.shutdown().await?;
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn first_write_follows_the_restored_backend_position() -> Result<()> {
        let backend = Arc::new(TestBackend::default());
        backend.applied.store(41, Ordering::Release);
        backend.durable.store(41, Ordering::Release);
        let runtime = start(Arc::clone(&backend)).await?;
        assert_eq!(runtime.write(request()).await?.response.bookmark.index, 42);
        runtime.flush_persistence().await?;
        assert_eq!(backend.durable.load(Ordering::Acquire), 42);
        runtime.shutdown().await
    }

    #[tokio::test(flavor = "current_thread")]
    async fn oversized_writes_and_thousands_of_publications_ignore_blocked_disk_and_flush()
    -> Result<()> {
        let backend = Arc::new(TestBackend::default());
        *backend.paused.lock() = true;
        let _release = ReleaseOnDrop(Arc::clone(&backend));
        let runtime = Arc::new(start(Arc::clone(&backend)).await?);
        let mut large = request();
        large.command.payload = vec![5; PERSISTENCE_BATCH_TARGET_BYTES * 2 + 1];
        assert_eq!(runtime.write(large).await?.response.bookmark.index, 1);
        await_reached(&backend).await?;
        let next_runtime = Arc::clone(&runtime);
        let next = tokio::spawn(async move { next_runtime.write(request()).await });
        let flush_runtime = Arc::clone(&runtime);
        let flush = tokio::spawn(async move { flush_runtime.flush_persistence().await });
        assert_eq!(
            next.await
                .map_err(|error| Error::internal(error.to_string()))??
                .response
                .bookmark
                .index,
            2
        );
        for expected in 3..=5_123 {
            assert_eq!(
                runtime.write(request()).await?.response.bookmark.index,
                expected
            );
        }
        {
            let state = runtime.persistence.state.lock();
            let dirty = state.pending.as_ref().expect("pending current records");
            assert_eq!(dirty.points.len(), 1);
            assert_eq!(dirty.bookmark.index, 5_123);
            assert_eq!(
                dirty.points.get(b"current".as_slice()),
                Some(&Some(5_123_u64.to_be_bytes().to_vec()))
            );
        }
        assert!(!flush.is_finished());
        assert_eq!(backend.durable.load(Ordering::Acquire), 0);
        backend.release();
        flush
            .await
            .map_err(|error| Error::internal(error.to_string()))??;
        runtime.shutdown().await?;
        assert_eq!(backend.durable.load(Ordering::Acquire), 5_123);
        assert_eq!(backend.last_value.load(Ordering::Acquire), 5_123);
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn publisher_root_restore_wakes_worker_parked_during_merge_without_another_event()
    -> Result<()> {
        let mailbox = Arc::new(PersistenceMailbox::new(Bookmark { term: 1, index: 0 }));
        let batch = |index| PersistenceRecords {
            batches: vec![RecordBatch {
                first_index: index,
                bookmark: Bookmark { term: 1, index },
                operations: vec![RecordOp::Put {
                    key: b"current".to_vec(),
                    value: index.to_be_bytes().to_vec(),
                }],
            }],
        };
        mailbox.publish(batch(1))?;
        let pause = Arc::new(PublicationPause::default());
        *mailbox.publication_pause.lock() = Some(Arc::clone(&pause));
        let publisher_mailbox = Arc::clone(&mailbox);
        let publisher = std::thread::spawn(move || publisher_mailbox.publish(batch(2)));
        let swapped = tokio::time::timeout(Duration::from_secs(2), pause.entered.notified()).await;
        // Keep a future deadline while the pending root is absent. The actor must park rather
        // than poll until the publisher restores that root and explicitly wakes its wait state.
        mailbox.state.lock().deadline = Some(Instant::now() + PERSISTENCE_GROUP_INTERVAL);
        let backend = Arc::new(TestBackend::default());
        let persistence_backend: Arc<dyn MutationStateBackend> = backend.clone();
        let worker_mailbox = Arc::clone(&mailbox);
        let worker = std::thread::spawn(move || {
            standalone_persistence_loop(worker_mailbox, persistence_backend)
        });
        let parked = tokio::time::timeout(Duration::from_secs(2), mailbox.parked.notified()).await;
        let parked_before_deadline = {
            let state = mailbox.state.lock();
            state.publication_parked
                && state.pending.is_none()
                && state
                    .deadline
                    .is_some_and(|deadline| deadline > Instant::now())
        };
        pause.release();
        let publication = tokio::task::spawn_blocking(move || publisher.join()).await;
        // No additional publication, flush, close or wake notification precedes this commit.
        let committed =
            tokio::time::timeout(Duration::from_secs(2), backend.committed.notified()).await;
        let durable = backend.durable.load(Ordering::Acquire);
        mailbox.barrier(None, true);
        tokio::task::spawn_blocking(move || worker.join())
            .await
            .map_err(|error| Error::internal(error.to_string()))?
            .map_err(|_| Error::internal("test persistence worker panicked"))?;
        swapped
            .map_err(|_| Error::internal("publisher did not pause after swapping pending root"))?;
        parked.map_err(|_| Error::internal("actor did not park while pending root was absent"))?;
        assert!(parked_before_deadline);
        publication
            .map_err(|error| Error::internal(error.to_string()))?
            .map_err(|_| Error::internal("test publisher panicked"))??;
        committed.map_err(|_| {
            Error::internal("restored dirty root missed its automatic commit deadline")
        })?;
        assert_eq!(durable, 2);
        assert_eq!(backend.last_value.load(Ordering::Acquire), 2);
        Ok(())
    }

    #[tokio::test(flavor = "current_thread")]
    async fn idle_actor_does_no_work_and_first_dirty_publication_arms_one_deadline() -> Result<()> {
        let backend = Arc::new(TestBackend::default());
        let runtime = start(Arc::clone(&backend)).await?;
        tokio::time::sleep(PERSISTENCE_GROUP_INTERVAL + Duration::from_millis(25)).await;
        assert_eq!(backend.persistence_calls.load(Ordering::Acquire), 0);
        assert!(runtime.persistence.state.lock().deadline.is_none());
        runtime.flush_persistence().await?;
        assert!(!runtime.persistence.state.lock().force);
        let pause = Arc::new(PublicationPause::default());
        let _resume = ResumePublication(Arc::clone(&pause));
        *runtime.persistence.publication_pause.lock() = Some(Arc::clone(&pause));
        // The RAM acknowledgement is deliberately independent of record coalescing. Hold
        // that phase to prove the test must synchronize on mailbox publication, not the ACK.
        let first = runtime.write(request()).await?;
        let entered = tokio::time::timeout(Duration::from_secs(2), pause.entered.notified()).await;
        let before_publication = {
            let state = runtime.persistence.state.lock();
            (state.published, state.deadline)
        };
        pause.release();
        entered.map_err(|_| Error::internal("writer did not reach the record-publication gate"))?;
        assert_eq!(backend.applied.load(Ordering::Acquire), 1);
        assert_eq!(before_publication, (0, None));
        await_record_publication(&runtime, first.response.bookmark.index).await?;
        let first_deadline = runtime
            .persistence
            .state
            .lock()
            .deadline
            .expect("dirty deadline");
        let second = runtime.write(request()).await?;
        await_record_publication(&runtime, second.response.bookmark.index).await?;
        assert_eq!(
            runtime.persistence.state.lock().deadline,
            Some(first_deadline)
        );
        assert_eq!(backend.persistence_calls.load(Ordering::Acquire), 0);
        await_reached(&backend).await?;
        runtime.flush_persistence().await?;
        assert_eq!(backend.persistence_calls.load(Ordering::Acquire), 1);
        assert!(runtime.persistence.state.lock().deadline.is_none());
        runtime.shutdown().await
    }

    #[test]
    fn prefix_coalescing_preserves_unrelated_large_pending_allocations_and_rejects_gaps()
    -> Result<()> {
        let mailbox = PersistenceMailbox::new(Bookmark { term: 1, index: 0 });
        mailbox.publish(PersistenceRecords {
            batches: vec![RecordBatch {
                first_index: 1,
                bookmark: Bookmark { term: 1, index: 1 },
                operations: (0..64)
                    .map(|index| RecordOp::Put {
                        key: format!("unrelated/{index}").into_bytes(),
                        value: vec![42; 128 * 1024],
                    })
                    .chain([RecordOp::Put {
                        key: b"target/a".to_vec(),
                        value: vec![1],
                    }])
                    .collect(),
            }],
        })?;
        let original = mailbox
            .state
            .lock()
            .pending
            .as_ref()
            .expect("pending")
            .points
            .get(b"unrelated/0".as_slice())
            .and_then(Option::as_ref)
            .expect("large value")
            .as_ptr();
        mailbox.publish(PersistenceRecords {
            batches: vec![RecordBatch {
                first_index: 2,
                bookmark: Bookmark { term: 1, index: 2 },
                operations: vec![RecordOp::DeletePrefix {
                    prefix: b"target/".to_vec(),
                }],
            }],
        })?;
        {
            let state = mailbox.state.lock();
            let dirty = state.pending.as_ref().expect("pending");
            assert_eq!(dirty.points.len(), 64);
            assert_eq!(
                dirty
                    .points
                    .get(b"unrelated/0".as_slice())
                    .and_then(Option::as_ref)
                    .expect("large value")
                    .as_ptr(),
                original
            );
        }
        assert!(
            mailbox
                .publish(PersistenceRecords {
                    batches: vec![RecordBatch {
                        first_index: 4,
                        bookmark: Bookmark { term: 1, index: 4 },
                        operations: Vec::new(),
                    }]
                })
                .is_err()
        );
        assert_eq!(mailbox.state.lock().published, 2);
        Ok(())
    }

    #[test]
    fn coalescing_matches_ordered_prefix_deletions_and_later_point_changes() {
        fn apply(rows: &mut BTreeMap<Vec<u8>, Vec<u8>>, operation: RecordOp) {
            match operation {
                RecordOp::Put { key, value } => {
                    rows.insert(key, value);
                }
                RecordOp::Delete { key } => {
                    rows.remove(&key);
                }
                RecordOp::DeletePrefix { prefix } => {
                    rows.retain(|key, _| !key.starts_with(&prefix));
                }
            }
        }
        let initial = BTreeMap::from([(b"a/old".to_vec(), vec![1]), (b"b/old".to_vec(), vec![2])]);
        let mut expected = initial.clone();
        let mut dirty = DirtyRecords::default();
        for index in 1_u64..=512 {
            let key =
                format!("{}/{}", if index % 2 == 0 { "a" } else { "b" }, index % 11).into_bytes();
            let operation = match index % 7 {
                0 => RecordOp::DeletePrefix {
                    prefix: b"a/".to_vec(),
                },
                1 => RecordOp::DeletePrefix {
                    prefix: b"b/1".to_vec(),
                },
                2 => RecordOp::Delete { key },
                _ => RecordOp::Put {
                    key,
                    value: index.to_be_bytes().to_vec(),
                },
            };
            apply(&mut expected, operation.clone());
            dirty.absorb(RecordBatch {
                first_index: index,
                bookmark: Bookmark { term: 1, index },
                operations: vec![operation],
            });
        }
        let batch = dirty.into_batch();
        assert_eq!(batch.first_index, 1);
        assert_eq!(batch.bookmark.index, 512);
        assert!(batch.operations.len() < 30);
        let mut actual = initial;
        for operation in batch.operations {
            apply(&mut actual, operation);
        }
        assert_eq!(actual, expected);
    }

    #[tokio::test(flavor = "current_thread")]
    async fn implicit_drop_does_not_join_blocked_disk_work_on_an_async_worker() -> Result<()> {
        let backend = Arc::new(TestBackend::default());
        *backend.paused.lock() = true;
        let _release = ReleaseOnDrop(Arc::clone(&backend));
        let runtime = start(Arc::clone(&backend)).await?;
        runtime.write(request()).await?;
        await_reached(&backend).await?;
        while tokio::task::coop::has_budget_remaining() {
            tokio::task::consume_budget().await;
        }
        drop(runtime);
        let progress = tokio::spawn(async { 1 });
        assert_eq!(
            progress
                .await
                .map_err(|error| Error::internal(error.to_string()))?,
            1
        );
        backend.release();
        tokio::time::timeout(Duration::from_secs(2), async {
            while backend.durable.load(Ordering::Acquire) != 1 {
                tokio::task::yield_now().await;
            }
        })
        .await
        .map_err(|_| Error::internal("implicit background close did not drain persistence"))?;
        Ok(())
    }
}
