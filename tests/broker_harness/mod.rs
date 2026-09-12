// Test-only target. See the note in any `tests/*.rs` file.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]
#![allow(dead_code)]

//! Shared fixture for driving the Stream and Queue wire surfaces with real clients.
//!
//! Both surfaces sit on one standalone broker state machine, so both interop suites need the same
//! thing: a coordinator that applies commands, and a listener on a port the test can discover.
//! Neither suite mocks the protocol — they run the shipping listener and let an off-the-shelf
//! client library negotiate against it, which is the only way to find out what a real client
//! actually does rather than what the code appears to allow.

use std::{
    net::SocketAddr,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use irongraph::{
    Bookmark, CommitAcknowledgement, ProjectId, Result,
    broker::{
        BrokerCommand, BrokerCommit, BrokerCoordinator, BrokerReply, BrokerStateMachine, Delivery,
        PayloadRecord, QueueInfo, StreamOffset,
    },
    engine::ApplicationWait,
    storage::SegmentStore,
};
use parking_lot::Mutex;

/// Single-node coordinator that applies commands directly.
///
/// The protocol surface is under test, so this supplies the local coordinator while leaving the
/// state machine, segment storage, and every protocol path exactly as they ship.
pub struct LocalCoordinator {
    broker: Mutex<BrokerStateMachine>,
    segments: SegmentStore,
    _directory: tempfile::TempDir,
    index: AtomicU64,
    changes: tokio::sync::watch::Sender<u64>,
}

impl LocalCoordinator {
    pub fn new() -> Result<Self> {
        let directory = tempfile::tempdir()?;
        let segments = SegmentStore::open(directory.path(), 64 * 1024 * 1024)?;
        let (changes, _) = tokio::sync::watch::channel(0);
        Ok(Self {
            broker: Mutex::new(BrokerStateMachine::default()),
            segments,
            _directory: directory,
            index: AtomicU64::new(0),
            changes,
        })
    }
}

impl BrokerCoordinator for LocalCoordinator {
    fn submit(&self, command: BrokerCommand, _wait: CommitAcknowledgement) -> Result<BrokerCommit> {
        let reply = self.broker.lock().apply(command, &self.segments)?;
        let index = self.index.fetch_add(1, Ordering::Relaxed).saturating_add(1);
        // Consumers wait on this to avoid polling; without it a fetch only advances on its timer.
        let _ignored = self.changes.send(index);
        Ok(BrokerCommit {
            bookmark: Bookmark { term: 1, index },
            reply,
            application: ApplicationWait::Complete,
        })
    }

    fn snapshot(&self) -> Result<BrokerStateMachine> {
        Ok(self.broker.lock().clone())
    }

    fn subscribe_changes(&self) -> tokio::sync::watch::Receiver<u64> {
        self.changes.subscribe()
    }

    fn fetch_partition(
        &self,
        project: ProjectId,
        topic: &str,
        partition: i32,
        offset: u64,
        maximum_bytes: usize,
    ) -> Result<Vec<(u64, Arc<PayloadRecord>)>> {
        self.broker.lock().fetch_partition(
            project,
            topic,
            partition,
            offset,
            maximum_bytes,
            &self.segments,
        )
    }

    fn list_offset(
        &self,
        project: ProjectId,
        topic: &str,
        partition: i32,
        timestamp: i64,
    ) -> Result<Option<(u64, i64)>> {
        self.broker
            .lock()
            .list_offset(project, topic, partition, timestamp)
    }

    fn fetch_stream_queue(
        &self,
        project: ProjectId,
        queue: &str,
        offset: StreamOffset,
        maximum: usize,
        consumer: u64,
        automatic_ack: bool,
    ) -> Result<Vec<Delivery>> {
        self.broker.lock().fetch_queue(
            project,
            queue,
            offset,
            maximum,
            uuid::Uuid::nil(),
            consumer,
            automatic_ack,
            0,
            &self.segments,
        )
    }

    fn queue_info(&self, project: ProjectId, name: &str) -> Result<Option<QueueInfo>> {
        Ok(self.broker.lock().queue_info(project, name))
    }
}

/// Reserves a loopback port and releases it so a listener can claim it.
///
/// The listeners bind internally, so a test cannot ask them which ephemeral port they took. The
/// window between releasing and rebinding is small and the port is not reused by the OS
/// immediately, which is sufficient for a test fixture.
pub async fn reserve_loopback_port() -> Result<SocketAddr> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    drop(listener);
    Ok(address)
}

/// Waits until a listener accepts connections, so a client does not race server startup.
pub async fn wait_until_listening(address: SocketAddr) -> Result<()> {
    for _ in 0..200 {
        if tokio::net::TcpStream::connect(address).await.is_ok() {
            return Ok(());
        }
        tokio::time::sleep(std::time::Duration::from_millis(25)).await;
    }
    Err(irongraph::Error::internal(format!(
        "listener at {address} never accepted a connection"
    )))
}

/// Applies a command outside any protocol path, for fixtures the client is not testing.
pub fn apply(coordinator: &Arc<LocalCoordinator>, command: BrokerCommand) -> Result<BrokerReply> {
    Ok(coordinator
        .submit(command, CommitAcknowledgement::Published)?
        .reply)
}
