use std::{
    collections::{BTreeMap, BTreeSet},
    net::SocketAddr,
    sync::Arc,
    time::{Duration, Instant},
};

use bytes::{Bytes, BytesMut};
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::{sync::Semaphore, task::JoinSet};
use uuid::Uuid;

use crate::{CommitAcknowledgement, Error, ErrorCode, ProjectId, Result};

use super::engine::{
    AmqpBatchRecord, AmqpExchangeKind, BrokerCommand, BrokerCoordinator, BrokerReply,
    ConnectionBrokerCoordinator, Delivery, IngressMetadata, QueueKind, RetentionPolicy,
    StreamOffset,
};
use super::memory::{BrokerMemoryReservation, process_broker_memory};

const PROTOCOL_HEADER: &[u8; 8] = b"AMQP\0\0\x09\x01";
const FRAME_END: u8 = 0xce;
const FRAME_METHOD: u8 = 1;
const FRAME_HEADER: u8 = 2;
const FRAME_BODY: u8 = 3;
const FRAME_HEARTBEAT: u8 = 8;
const SERVER_CHANNEL_MAX: u16 = 2_047;
const SERVER_FRAME_MAX: u32 = 131_072;
const SERVER_HEARTBEAT: u16 = 30;
const MIN_FRAME_MAX: u32 = 4_096;
const FRAME_OVERHEAD: usize = 8;
// The durable delivery result is capped at 16 MiB and its canonical CBOR record can encode an
// arbitrary body octet in two bytes. Seven MiB leaves two MiB after that worst-case expansion for
// the bounded 128 KiB content header, ingress fields, segment framing, and delivery envelope. An
// accepted message is therefore deliverable through both classic and stream queue paths.
const MAX_MESSAGE_BYTES: usize = 7 * 1024 * 1024;
// During local apply the body remains in the command while canonical CBOR may expand each
// octet to two bytes and the immutable segment writer temporarily holds a same-sized postcard
// frame. One input body + both two-body workspaces is five body-sized allocations; mandatory
// return keeps one further body copy. The fixed reserve covers the bounded content header and
// its canonical clones/encoding.
const PUBLISH_STORAGE_BODY_COPIES: usize = 5;
const PUBLISH_STORAGE_FIXED_RESERVE: usize = 2 * 1024 * 1024;
const PUBLISH_STORAGE_MIN_METADATA_RESERVE: usize = 4 * 1024;
const MAX_BUFFERED_BYTES: usize = MAX_MESSAGE_BYTES + (SERVER_FRAME_MAX as usize * 2);
const CONNECTION_BUFFER_RESERVE: usize = SERVER_FRAME_MAX as usize * 3;
const MAX_AMQP_PUBLISH_BATCH: usize = 8_192;
const PUBLISH_FLUSH_INTERVAL: Duration = Duration::from_micros(250);
// Direct storage reads transiently hold the framed postcard bytes and its decoded canonical
// record (each up to twice the body size). This four-body reserve covers that peak; the fixed
// allowance covers decoded ingress metadata and the AMQP delivery envelope.
const EGRESS_DELIVERY_RESERVE: usize = MAX_MESSAGE_BYTES * 4 + PUBLISH_STORAGE_FIXED_RESERVE;
const MAX_TABLE_BYTES: usize = 1024 * 1024;
const MAX_TABLE_ENTRIES: usize = 1_024;
const MAX_FIELD_DEPTH: usize = 8;
const MAX_CHANNELS: usize = SERVER_CHANNEL_MAX as usize;
const MAX_CONSUMERS_PER_CHANNEL: usize = 1_024;
const MAX_CONSUMERS_PER_CONNECTION: usize = 4_096;
const MAX_DELIVERIES_PER_PUMP: u32 = 16_384;
const MAX_DELIVERY_RESULT_BYTES: usize = 16 * 1024 * 1024;

/// Broker-memory reservation covering one pump round of `deliveries` messages.
///
/// Small groups retain the prior worst-case per-message charge. Larger groups clamp at four copies
/// of the engine's 16 MiB result bound because validation now stops before crossing that bound;
/// charging 7 MiB for every tiny record made safe batches artificially shrink to 32 messages.
fn delivery_reserve_bytes(deliveries: u32) -> usize {
    MAX_MESSAGE_BYTES
        .saturating_mul(deliveries.max(1) as usize)
        .min(MAX_DELIVERY_RESULT_BYTES)
        .saturating_mul(4)
        .saturating_add(PUBLISH_STORAGE_FIXED_RESERVE)
}

/// Largest pump batch the process broker budget could ever admit, even when otherwise idle.
///
/// Starting the reservation above this would fail on every attempt and only cost the shrink loop
/// extra iterations, so the credit is clamped here before the first try.
fn admissible_delivery_batch(credit: u32) -> u32 {
    let budget = process_broker_memory().maximum_bytes();
    let mut batch = credit.clamp(1, MAX_DELIVERIES_PER_PUMP);
    while batch > 1 && delivery_reserve_bytes(batch) > budget {
        batch /= 2;
    }
    batch
}
const PUMP_INTERVAL: Duration = Duration::from_millis(10);
const MAX_IDLE_POLL_INTERVAL: Duration = Duration::from_millis(250);
const DELIVERY_LEASE_RENEW_INTERVAL: Duration = Duration::from_secs(30);
const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(10);
const MAX_CONNECTIONS: usize = 512;

type StoredFieldMap = BTreeMap<String, Vec<u8>>;
type StoredBasicProperties = (StoredFieldMap, StoredFieldMap);

const CLASS_CONNECTION: u16 = 10;
const CLASS_CHANNEL: u16 = 20;
const CLASS_EXCHANGE: u16 = 40;
const CLASS_QUEUE: u16 = 50;
const CLASS_BASIC: u16 = 60;
const CLASS_CONFIRM: u16 = 85;

const REPLY_CONTENT_TOO_LARGE: u16 = 311;
const REPLY_NO_ROUTE: u16 = 312;
const REPLY_CONNECTION_FORCED: u16 = 320;
const REPLY_INVALID_PATH: u16 = 402;
const REPLY_ACCESS_REFUSED: u16 = 403;
const REPLY_NOT_FOUND: u16 = 404;
const REPLY_PRECONDITION_FAILED: u16 = 406;
const REPLY_FRAME_ERROR: u16 = 501;
const REPLY_SYNTAX_ERROR: u16 = 502;
const REPLY_COMMAND_INVALID: u16 = 503;
const REPLY_CHANNEL_ERROR: u16 = 504;
const REPLY_UNEXPECTED_FRAME: u16 = 505;
const REPLY_RESOURCE_ERROR: u16 = 506;
const REPLY_NOT_ALLOWED: u16 = 530;
const REPLY_NOT_IMPLEMENTED: u16 = 540;
const REPLY_INTERNAL_ERROR: u16 = 541;

/// AMQP 0-9-1 listener over the standalone broker state machine.
pub struct QueueServer {
    address: SocketAddr,
    project: ProjectId,
    coordinator: Arc<dyn BrokerCoordinator>,
}

impl QueueServer {
    /// Creates a project-scoped AMQP listener.
    #[must_use]
    pub fn new(
        address: SocketAddr,
        project: ProjectId,
        coordinator: Arc<dyn BrokerCoordinator>,
    ) -> Self {
        Self {
            address,
            project,
            coordinator,
        }
    }

    /// Runs the loopback listener. Remote callers must enter through an authenticated TLS
    /// acceptor and call [`Self::serve_authenticated_transport`].
    ///
    /// # Errors
    ///
    /// Returns an error when the address is not loopback, binding fails, or listener I/O fails.
    pub async fn run(self, shutdown: tokio_util::sync::CancellationToken) -> Result<()> {
        if !self.address.ip().is_loopback() {
            return Err(Error::new(
                ErrorCode::AuthenticationFailed,
                "plain AMQP listener is restricted to loopback; remote AMQP requires mTLS",
            ));
        }
        let listener = tokio::net::TcpListener::bind(self.address).await?;
        let permits = Arc::new(Semaphore::new(MAX_CONNECTIONS));
        let mut connections = JoinSet::new();
        loop {
            tokio::select! {
                () = shutdown.cancelled() => break,
                accepted = listener.accept() => {
                    let (stream, peer) = accepted?;
                    if !peer.ip().is_loopback() {
                        continue;
                    }
                    let Ok(permit) = Arc::clone(&permits).try_acquire_owned() else {
                        drop(stream);
                        continue;
                    };
                    let coordinator = Arc::clone(&self.coordinator);
                    let project = self.project;
                    let connection_shutdown = shutdown.child_token();
                    let connection = Box::pin(serve(
                        stream,
                        project,
                        coordinator,
                        connection_shutdown,
                        TransportTrust::Loopback,
                    ));
                    connections.spawn(async move {
                        let _permit = permit;
                        let _ = connection.await;
                    });
                }
                _ = connections.join_next(), if !connections.is_empty() => {}
            }
        }
        connections.abort_all();
        while connections.join_next().await.is_some() {}
        Ok(())
    }

    /// Serves one transport after its peer certificate has already been authenticated and
    /// authorized for this immutable project by the owning TLS acceptor.
    ///
    /// # Errors
    ///
    /// Returns an error for protocol violations, transport I/O failures, or broker failures.
    pub async fn serve_authenticated_transport<T>(
        transport: T,
        project: ProjectId,
        coordinator: Arc<dyn BrokerCoordinator>,
        shutdown: tokio_util::sync::CancellationToken,
    ) -> Result<()>
    where
        T: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        serve(
            transport,
            project,
            coordinator,
            shutdown,
            TransportTrust::PeerAuthenticated,
        )
        .await
    }
}

#[derive(Debug)]
struct Frame {
    kind: u8,
    channel: u16,
    payload: Bytes,
}

fn is_basic_publish_method(frame: &Frame) -> bool {
    frame.payload.len() >= 4
        && u16::from_be_bytes([frame.payload[0], frame.payload[1]]) == CLASS_BASIC
        && u16::from_be_bytes([frame.payload[2], frame.payload[3]]) == 40
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ConnectionPhase {
    AwaitStartOk,
    AwaitTuneOk,
    AwaitOpen,
    Open,
}

/// How the peer on this transport was established, which decides the SASL mechanisms offered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TransportTrust {
    /// The plaintext listener, which `QueueServer::run` restricts to loopback. Reaching it already
    /// implies local access, so `PLAIN` grants no authority a caller did not already have — and
    /// refusing it meant a default-configured client could not connect at all.
    Loopback,
    /// A TLS transport whose peer certificate was validated and authorized for this project before
    /// the connection reached the protocol. Only `EXTERNAL` is meaningful there: the identity is
    /// the certificate, and accepting a password would let a weaker credential override it.
    PeerAuthenticated,
}

struct ConnectionState {
    phase: ConnectionPhase,
    trust: TransportTrust,
    project: ProjectId,
    channels: BTreeMap<u16, ChannelState>,
    closing_channels: BTreeSet<u16>,
    negotiated_channel_max: u16,
    negotiated_frame_max: u32,
    heartbeat: u16,
    next_consumer_id: u64,
    consumer_count: usize,
    client_name: Option<String>,
    buffered_publish_bytes: usize,
    ready_publishes: Vec<ReadyPublish>,
    delivery_owner: Uuid,
    delivery_lease_renew_after: Instant,
    delivery_lease_touched: bool,
    /// Set once this connection owns canonical state that outlives a single method: a consumer
    /// registration, or an exclusive or auto-delete queue. Teardown only pays for a
    /// `ReleaseConnection` round trip when there is something to release.
    connection_state_registered: bool,
}

impl ConnectionState {
    fn new(project: ProjectId, trust: TransportTrust) -> Self {
        Self {
            phase: ConnectionPhase::AwaitStartOk,
            trust,
            project,
            channels: BTreeMap::new(),
            closing_channels: BTreeSet::new(),
            negotiated_channel_max: SERVER_CHANNEL_MAX,
            negotiated_frame_max: SERVER_FRAME_MAX,
            heartbeat: SERVER_HEARTBEAT,
            next_consumer_id: 0,
            consumer_count: 0,
            client_name: None,
            buffered_publish_bytes: 0,
            ready_publishes: Vec::with_capacity(256),
            delivery_owner: Uuid::new_v4(),
            delivery_lease_renew_after: Instant::now() + DELIVERY_LEASE_RENEW_INTERVAL,
            delivery_lease_touched: false,
            connection_state_registered: false,
        }
    }

    fn allocate_consumer(&mut self) -> std::result::Result<u64, ProtocolError> {
        self.next_consumer_id = self.next_consumer_id.checked_add(1).ok_or_else(|| {
            ProtocolError::connection(
                REPLY_RESOURCE_ERROR,
                "consumer identifier space exhausted",
                CLASS_BASIC,
                20,
            )
        })?;
        Ok(self.next_consumer_id)
    }
}

struct ChannelState {
    flow_active: bool,
    prefetch_count: u16,
    prefetch_global: bool,
    consumers: BTreeMap<String, ConsumerState>,
    outstanding: BTreeMap<u64, OutstandingDelivery>,
    next_delivery_tag: u64,
    confirms: bool,
    next_publish_sequence: u64,
    pending_publish: Option<PendingPublish>,
    pump_cursor: usize,
    last_declared_queue: Option<String>,
}

impl ChannelState {
    const fn new() -> Self {
        Self {
            flow_active: true,
            prefetch_count: 0,
            prefetch_global: false,
            consumers: BTreeMap::new(),
            outstanding: BTreeMap::new(),
            next_delivery_tag: 0,
            confirms: false,
            next_publish_sequence: 0,
            pending_publish: None,
            pump_cursor: 0,
            last_declared_queue: None,
        }
    }

    fn allocate_delivery_tag(&mut self) -> std::result::Result<u64, ProtocolError> {
        self.next_delivery_tag = self.next_delivery_tag.checked_add(1).ok_or_else(|| {
            ProtocolError::channel(
                REPLY_RESOURCE_ERROR,
                "delivery tag space exhausted",
                CLASS_BASIC,
                60,
            )
        })?;
        Ok(self.next_delivery_tag)
    }

    fn allocate_publish_sequence(&mut self) -> std::result::Result<u64, ProtocolError> {
        self.next_publish_sequence =
            self.next_publish_sequence.checked_add(1).ok_or_else(|| {
                ProtocolError::channel(
                    REPLY_RESOURCE_ERROR,
                    "publisher sequence space exhausted",
                    CLASS_BASIC,
                    40,
                )
            })?;
        Ok(self.next_publish_sequence)
    }
}

struct ConsumerState {
    id: u64,
    queue: String,
    automatic_ack: bool,
    queue_kind: QueueKind,
    next_offset: StreamOffset,
    outstanding: u32,
    poll_after: Instant,
    idle_poll_interval: Duration,
}

struct OutstandingDelivery {
    consumer_id: u64,
    consumer_tag: Option<String>,
    queue: String,
    queue_kind: QueueKind,
    engine_delivery_tag: u64,
    stream_offset: u64,
}

struct PendingPublish {
    exchange: String,
    routing_key: String,
    mandatory: bool,
    sequence: u64,
    expected_body: Option<u64>,
    properties: Option<BasicProperties>,
    body: Vec<u8>,
    memory: BrokerMemoryReservation,
}

struct ReadyPublish {
    channel: u16,
    sequence: u64,
    confirms: bool,
    returned_properties: Option<BasicProperties>,
    returned_body: Option<Vec<u8>>,
    record: AmqpBatchRecord,
    memory: BrokerMemoryReservation,
}

struct PublishCompletion {
    channel: u16,
    sequence: u64,
    confirms: bool,
    returned_exchange: Option<String>,
    returned_routing_key: Option<String>,
    returned_properties: Option<BasicProperties>,
    returned_body: Option<Vec<u8>>,
}

#[derive(Debug)]
struct ProtocolError {
    code: u16,
    text: String,
    class_id: u16,
    method_id: u16,
    connection: bool,
    channel: u16,
}

impl ProtocolError {
    fn connection(code: u16, text: impl Into<String>, class_id: u16, method_id: u16) -> Self {
        Self {
            code,
            text: bounded_reply_text(text.into()),
            class_id,
            method_id,
            connection: true,
            channel: 0,
        }
    }

    fn channel(code: u16, text: impl Into<String>, class_id: u16, method_id: u16) -> Self {
        Self {
            code,
            text: bounded_reply_text(text.into()),
            class_id,
            method_id,
            connection: false,
            channel: 0,
        }
    }

    fn from_engine(error: Error, class_id: u16, method_id: u16) -> Self {
        if matches!(error.code, ErrorCode::DeadlineExceeded) {
            return Self::connection(
                REPLY_CONNECTION_FORCED,
                "broker outcome is unknown after the commit deadline",
                class_id,
                method_id,
            );
        }
        let code = match error.code {
            ErrorCode::AuthenticationFailed | ErrorCode::AuthorizationDenied => {
                REPLY_ACCESS_REFUSED
            }
            ErrorCode::Backpressure | ErrorCode::WriteAdmissionFull => REPLY_RESOURCE_ERROR,
            ErrorCode::ProtocolViolation | ErrorCode::InvalidData | ErrorCode::RetentionExpired => {
                REPLY_PRECONDITION_FAILED
            }
            ErrorCode::ProjectNotFound | ErrorCode::ProjectFenced => REPLY_NOT_FOUND,
            _ => REPLY_INTERNAL_ERROR,
        };
        Self::channel(code, error.message.into_owned(), class_id, method_id)
    }

    const fn on_channel(mut self, channel: u16) -> Self {
        if channel == 0 {
            self.connection = true;
            self.channel = 0;
        } else if !self.connection {
            self.channel = channel;
        }
        self
    }
}

fn bounded_reply_text(mut text: String) -> String {
    if text.len() > u8::MAX as usize {
        let mut boundary = u8::MAX as usize;
        while boundary > 0 && !text.is_char_boundary(boundary) {
            boundary -= 1;
        }
        text.truncate(boundary);
    }
    text
}

#[allow(clippy::too_many_lines)]
async fn serve<T>(
    transport: T,
    project: ProjectId,
    coordinator: Arc<dyn BrokerCoordinator>,
    shutdown: tokio_util::sync::CancellationToken,
    trust: TransportTrust,
) -> Result<()>
where
    T: AsyncRead + AsyncWrite + Unpin + Send + 'static,
{
    let coordinator: Arc<dyn BrokerCoordinator> =
        Arc::new(ConnectionBrokerCoordinator::new(coordinator));
    let (mut reader, writer) = tokio::io::split(transport);
    let mut writer = tokio::io::BufWriter::with_capacity(1024 * 1024, writer);
    let mut protocol = [0u8; PROTOCOL_HEADER.len()];
    match tokio::time::timeout(HANDSHAKE_TIMEOUT, reader.read_exact(&mut protocol)).await {
        Ok(Ok(_)) => {}
        Ok(Err(error)) => return Err(error.into()),
        Err(_) => {
            return Err(Error::new(
                ErrorCode::DeadlineExceeded,
                "AMQP protocol header timed out",
            ));
        }
    }
    if &protocol != PROTOCOL_HEADER {
        writer.write_all(PROTOCOL_HEADER).await?;
        writer.flush().await?;
        return Err(Error::new(
            ErrorCode::ProtocolViolation,
            "unsupported AMQP protocol header",
        ));
    }

    let mut state = ConnectionState::new(project, trust);
    let mut started = Writer::new();
    started.u16(CLASS_CONNECTION);
    started.u16(10);
    started.u8(0);
    started.u8(9);
    started.field_table(&server_properties())?;
    // Offering only `EXTERNAL` made the default configuration of every AMQP client fail at the
    // handshake. Both are advertised; `PLAIN` is honoured only on a transport where it grants no
    // authority the caller did not already hold.
    started.long_bytes(match trust {
        TransportTrust::Loopback => b"PLAIN EXTERNAL".as_slice(),
        TransportTrust::PeerAuthenticated => b"EXTERNAL".as_slice(),
    })?;
    started.long_bytes(b"en_US")?;
    write_frame(
        &mut writer,
        FRAME_METHOD,
        0,
        started.as_slice(),
        SERVER_FRAME_MAX,
    )
    .await?;

    let _connection_memory = process_broker_memory()
        .reserve(CONNECTION_BUFFER_RESERVE)
        .await?;
    let handshake_deadline = tokio::time::sleep(HANDSHAKE_TIMEOUT);
    tokio::pin!(handshake_deadline);
    let mut input = BytesMut::with_capacity(SERVER_FRAME_MAX as usize * 2);
    // One socket read can now feed thousands of small frames into one canonical publish batch.
    // This heap buffer is already covered by the per-connection reservation and removes the
    // former 16 KiB batching ceiling without growing an async task's stack.
    let mut scratch = vec![0u8; SERVER_FRAME_MAX as usize * 2];
    let mut heartbeat_tick = tokio::time::interval(Duration::from_secs(1));
    heartbeat_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut pump_tick = tokio::time::interval(PUMP_INTERVAL);
    pump_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut publish_flush_tick = tokio::time::interval(PUBLISH_FLUSH_INTERVAL);
    publish_flush_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    publish_flush_tick.tick().await;
    let mut last_received = Instant::now();
    let mut last_sent = Instant::now();
    let outcome = 'connection: loop {
        tokio::select! {
            () = &mut handshake_deadline, if state.phase != ConnectionPhase::Open => {
                let _ = send_connection_close(
                    &mut writer,
                    state.negotiated_frame_max,
                    REPLY_CONNECTION_FORCED,
                    "connection handshake timed out",
                    CLASS_CONNECTION,
                    0,
                ).await;
                break Ok(());
            }
            () = shutdown.cancelled() => {
                let _ = send_connection_close(
                    &mut writer,
                    state.negotiated_frame_max,
                    REPLY_CONNECTION_FORCED,
                    "server is shutting down",
                    0,
                    0,
                ).await;
                break Ok(());
            }
            read = reader.read(&mut scratch) => {
                match read {
                    Ok(0) => break Ok(()),
                    Ok(count) => {
                        if input.len().saturating_add(count) > MAX_BUFFERED_BYTES {
                            let error = ProtocolError::connection(
                                REPLY_RESOURCE_ERROR,
                                "connection input buffer limit exceeded",
                                0,
                                0,
                            );
                            let _ = send_protocol_error(&mut writer, 0, state.negotiated_frame_max, &error).await;
                            break Ok(());
                        }
                        input.extend_from_slice(&scratch[..count]);
                        last_received = Instant::now();
                        loop {
                            let frame = match try_decode_frame(&mut input, state.negotiated_frame_max) {
                                Ok(Some(frame)) => frame,
                                Ok(None) => break,
                                Err(error) => {
                                    let _ = send_protocol_error(&mut writer, 0, state.negotiated_frame_max, &error).await;
                                    break 'connection Err(Error::new(ErrorCode::ProtocolViolation, error.text));
                                }
                            };
                            if !state.ready_publishes.is_empty()
                                && frame.kind == FRAME_METHOD
                                && !is_basic_publish_method(&frame)
                            {
                                match flush_ready_publishes(&mut state, &coordinator, &mut writer).await {
                                    Ok(HandleOutcome::Continue) => last_sent = Instant::now(),
                                    Ok(_) => {}
                                    Err(error) => {
                                        let error_channel = error_channel(&error);
                                        let _ = send_protocol_error(
                                            &mut writer,
                                            error_channel,
                                            state.negotiated_frame_max,
                                            &error,
                                        ).await;
                                        break 'connection Ok(());
                                    }
                                }
                            }
                            match handle_frame(
                                frame,
                                &mut state,
                                &coordinator,
                                &mut writer,
                            ).await {
                                Ok(HandleOutcome::Continue) => last_sent = Instant::now(),
                                Ok(HandleOutcome::NoResponse) => {}
                                Ok(HandleOutcome::Close) => break 'connection Ok(()),
                                Err(error) => {
                                    let error_channel = error_channel(&error);
                                    let _ = send_protocol_error(
                                        &mut writer,
                                        error_channel,
                                        state.negotiated_frame_max,
                                        &error,
                                    ).await;
                                    if error.connection {
                                        break 'connection Ok(());
                                    }
                                    cleanup_channel(
                                        error_channel,
                                        &mut state,
                                        &coordinator,
                                    ).await;
                                    state.channels.remove(&error_channel);
                                    state.closing_channels.insert(error_channel);
                                    last_sent = Instant::now();
                                }
                            }
                            if state.ready_publishes.len() >= MAX_AMQP_PUBLISH_BATCH {
                                match flush_ready_publishes(&mut state, &coordinator, &mut writer).await {
                                    Ok(HandleOutcome::Continue) => last_sent = Instant::now(),
                                    Ok(_) => {}
                                    Err(error) => {
                                        let error_channel = error_channel(&error);
                                        let _ = send_protocol_error(
                                            &mut writer,
                                            error_channel,
                                            state.negotiated_frame_max,
                                            &error,
                                        ).await;
                                        break 'connection Ok(());
                                    }
                                }
                            }
                        }
                        if state.ready_publishes.len() == 1 {
                            match flush_ready_publishes(&mut state, &coordinator, &mut writer).await {
                                Ok(HandleOutcome::Continue) => last_sent = Instant::now(),
                                Ok(_) => {}
                                Err(error) => {
                                    let error_channel = error_channel(&error);
                                    let _ = send_protocol_error(
                                        &mut writer,
                                        error_channel,
                                        state.negotiated_frame_max,
                                        &error,
                                    ).await;
                                    break 'connection Ok(());
                                }
                            }
                        }
                    }
                    Err(error) => break Err(error.into()),
                }
            }
            _ = publish_flush_tick.tick() => {
                if !state.ready_publishes.is_empty() {
                    match flush_ready_publishes(&mut state, &coordinator, &mut writer).await {
                        Ok(HandleOutcome::Continue) => last_sent = Instant::now(),
                        Ok(_) => {}
                        Err(error) => {
                            let error_channel = error_channel(&error);
                            let _ = send_protocol_error(
                                &mut writer,
                                error_channel,
                                state.negotiated_frame_max,
                                &error,
                            ).await;
                            break Ok(());
                        }
                    }
                }
            }
            _ = heartbeat_tick.tick(), if state.phase == ConnectionPhase::Open => {
                if state.heartbeat != 0 {
                    let heartbeat = Duration::from_secs(u64::from(state.heartbeat));
                    if last_received.elapsed() >= heartbeat.saturating_mul(2) {
                        let _ = send_connection_close(
                            &mut writer,
                            state.negotiated_frame_max,
                            REPLY_CONNECTION_FORCED,
                            "heartbeat timeout",
                            CLASS_CONNECTION,
                            0,
                        ).await;
                        break Ok(());
                    }
                    if last_sent.elapsed() >= heartbeat / 2 {
                        write_frame(
                            &mut writer,
                            FRAME_HEARTBEAT,
                            0,
                            &[],
                            state.negotiated_frame_max,
                        ).await?;
                        last_sent = Instant::now();
                    }
                }
            }
            _ = pump_tick.tick(), if state.phase == ConnectionPhase::Open => {
                if let Err(error) = renew_delivery_lease_if_due(&mut state, &coordinator).await {
                    let _ = send_protocol_error(
                        &mut writer,
                        0,
                        state.negotiated_frame_max,
                        &error,
                    ).await;
                    break Ok(());
                }
                match pump_consumers(&mut state, &coordinator, &mut writer).await {
                    Ok(sent) => {
                        if sent {
                            last_sent = Instant::now();
                        }
                    }
                    Err(error) => {
                        let channel = error_channel(&error);
                        let _ = send_protocol_error(
                            &mut writer,
                            channel,
                            state.negotiated_frame_max,
                            &error,
                        ).await;
                        if error.connection {
                            break Ok(());
                        }
                        cleanup_channel(channel, &mut state, &coordinator).await;
                        state.channels.remove(&channel);
                        state.closing_channels.insert(channel);
                    }
                }
                writer.flush().await?;
            }
        }
    };

    cleanup_connection(&mut state, &coordinator).await;
    outcome
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum HandleOutcome {
    Continue,
    NoResponse,
    Close,
}

const fn error_channel(error: &ProtocolError) -> u16 {
    if error.connection { 0 } else { error.channel }
}

fn try_decode_frame(
    input: &mut BytesMut,
    frame_max: u32,
) -> std::result::Result<Option<Frame>, ProtocolError> {
    if input.len() < 7 {
        return Ok(None);
    }
    let kind = input[0];
    let channel = u16::from_be_bytes([input[1], input[2]]);
    let size = u32::from_be_bytes([input[3], input[4], input[5], input[6]]);
    let total = (size as usize)
        .checked_add(FRAME_OVERHEAD)
        .ok_or_else(|| ProtocolError::connection(REPLY_FRAME_ERROR, "frame size overflow", 0, 0))?;
    if total > frame_max as usize {
        return Err(ProtocolError::connection(
            REPLY_FRAME_ERROR,
            "frame exceeds negotiated frame-max",
            0,
            0,
        ));
    }
    if input.len() < total {
        return Ok(None);
    }
    if input[total - 1] != FRAME_END {
        return Err(ProtocolError::connection(
            REPLY_FRAME_ERROR,
            "frame-end marker is invalid",
            0,
            0,
        ));
    }
    if !matches!(
        kind,
        FRAME_METHOD | FRAME_HEADER | FRAME_BODY | FRAME_HEARTBEAT
    ) {
        return Err(ProtocolError::connection(
            REPLY_FRAME_ERROR,
            "frame type is not supported",
            0,
            0,
        ));
    }
    if kind == FRAME_HEARTBEAT && (channel != 0 || size != 0) {
        return Err(ProtocolError::connection(
            REPLY_FRAME_ERROR,
            "heartbeat frame must be empty on channel zero",
            CLASS_CONNECTION,
            0,
        ));
    }
    // Split ownership from the receive buffer and retain a slice of that allocation. This avoids
    // copying every method/header/body frame before the protocol handler consumes it.
    let frame = input.split_to(total).freeze();
    let payload = frame.slice(7..7 + size as usize);
    Ok(Some(Frame {
        kind,
        channel,
        payload,
    }))
}

async fn write_frame<W: AsyncWrite + Unpin>(
    writer: &mut W,
    kind: u8,
    channel: u16,
    payload: &[u8],
    frame_max: u32,
) -> Result<()> {
    write_frame_buffered(writer, kind, channel, payload, frame_max).await?;
    writer.flush().await?;
    Ok(())
}

async fn write_frame_buffered<W: AsyncWrite + Unpin>(
    writer: &mut W,
    kind: u8,
    channel: u16,
    payload: &[u8],
    frame_max: u32,
) -> Result<()> {
    let total = payload
        .len()
        .checked_add(FRAME_OVERHEAD)
        .ok_or_else(|| Error::internal("AMQP frame size overflow"))?;
    if total > frame_max as usize {
        return Err(Error::new(
            ErrorCode::ProtocolViolation,
            "outbound AMQP frame exceeds negotiated frame-max",
        ));
    }
    let payload_len = u32::try_from(payload.len())
        .map_err(|_| Error::internal("AMQP payload length exceeds u32"))?;
    writer.write_u8(kind).await?;
    writer.write_u16(channel).await?;
    writer.write_u32(payload_len).await?;
    writer.write_all(payload).await?;
    writer.write_u8(FRAME_END).await?;
    Ok(())
}

async fn write_method<W: AsyncWrite + Unpin>(
    writer: &mut W,
    channel: u16,
    method: &Writer,
    frame_max: u32,
) -> Result<()> {
    write_frame(writer, FRAME_METHOD, channel, method.as_slice(), frame_max).await
}

async fn write_method_buffered<W: AsyncWrite + Unpin>(
    writer: &mut W,
    channel: u16,
    method: &Writer,
    frame_max: u32,
) -> Result<()> {
    write_frame_buffered(writer, FRAME_METHOD, channel, method.as_slice(), frame_max).await
}

async fn send_protocol_error<W: AsyncWrite + Unpin>(
    writer: &mut W,
    channel: u16,
    frame_max: u32,
    error: &ProtocolError,
) -> Result<()> {
    if error.connection || channel == 0 {
        send_connection_close(
            writer,
            frame_max,
            error.code,
            &error.text,
            error.class_id,
            error.method_id,
        )
        .await
    } else {
        let mut method = Writer::method(CLASS_CHANNEL, 40);
        method.u16(error.code);
        method.short_string(&error.text)?;
        method.u16(error.class_id);
        method.u16(error.method_id);
        write_method(writer, channel, &method, frame_max).await
    }
}

async fn send_connection_close<W: AsyncWrite + Unpin>(
    writer: &mut W,
    frame_max: u32,
    code: u16,
    text: &str,
    class_id: u16,
    method_id: u16,
) -> Result<()> {
    let mut method = Writer::method(CLASS_CONNECTION, 50);
    method.u16(code);
    method.short_string(text)?;
    method.u16(class_id);
    method.u16(method_id);
    write_method(writer, 0, &method, frame_max).await
}

#[derive(Clone, Debug)]
enum FieldValue {
    Boolean(bool),
    I8(i8),
    U8(u8),
    I16(i16),
    U16(u16),
    I32(i32),
    U32(u32),
    I64(i64),
    F32(f32),
    F64(f64),
    Decimal { scale: u8, value: u32 },
    LongString(Vec<u8>),
    Array(Vec<Self>),
    Timestamp(u64),
    Table(BTreeMap<String, Self>),
    Void,
    Bytes(Vec<u8>),
}

impl FieldValue {
    fn as_utf8(&self) -> Option<&str> {
        match self {
            Self::LongString(value) => std::str::from_utf8(value).ok(),
            _ => None,
        }
    }

    fn as_u64(&self) -> Option<u64> {
        match self {
            Self::U8(value) => Some(u64::from(*value)),
            Self::U16(value) => Some(u64::from(*value)),
            Self::U32(value) => Some(u64::from(*value)),
            Self::I8(value) => u64::try_from(*value).ok(),
            Self::I16(value) => u64::try_from(*value).ok(),
            Self::I32(value) => u64::try_from(*value).ok(),
            Self::I64(value) => u64::try_from(*value).ok(),
            Self::Timestamp(value) => Some(*value),
            _ => None,
        }
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    const fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, position: 0 }
    }

    const fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.position)
    }

    fn take(&mut self, length: usize) -> std::result::Result<&'a [u8], ProtocolError> {
        let end = self.position.checked_add(length).ok_or_else(|| {
            ProtocolError::connection(REPLY_FRAME_ERROR, "AMQP field length overflow", 0, 0)
        })?;
        if end > self.bytes.len() {
            return Err(ProtocolError::connection(
                REPLY_FRAME_ERROR,
                "AMQP frame ended inside a field",
                0,
                0,
            ));
        }
        let value = &self.bytes[self.position..end];
        self.position = end;
        Ok(value)
    }

    fn u8(&mut self) -> std::result::Result<u8, ProtocolError> {
        Ok(self.take(1)?[0])
    }

    fn i8(&mut self) -> std::result::Result<i8, ProtocolError> {
        Ok(i8::from_be_bytes([self.u8()?]))
    }

    fn u16(&mut self) -> std::result::Result<u16, ProtocolError> {
        let bytes = self.take(2)?;
        Ok(u16::from_be_bytes([bytes[0], bytes[1]]))
    }

    fn i16(&mut self) -> std::result::Result<i16, ProtocolError> {
        let bytes = self.take(2)?;
        Ok(i16::from_be_bytes([bytes[0], bytes[1]]))
    }

    fn u32(&mut self) -> std::result::Result<u32, ProtocolError> {
        let bytes = self.take(4)?;
        Ok(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn i32(&mut self) -> std::result::Result<i32, ProtocolError> {
        let bytes = self.take(4)?;
        Ok(i32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn u64(&mut self) -> std::result::Result<u64, ProtocolError> {
        let bytes = self.take(8)?;
        Ok(u64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn i64(&mut self) -> std::result::Result<i64, ProtocolError> {
        let bytes = self.take(8)?;
        Ok(i64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn f32(&mut self) -> std::result::Result<f32, ProtocolError> {
        Ok(f32::from_bits(self.u32()?))
    }

    fn f64(&mut self) -> std::result::Result<f64, ProtocolError> {
        Ok(f64::from_bits(self.u64()?))
    }

    fn short_bytes(&mut self) -> std::result::Result<&'a [u8], ProtocolError> {
        let length = usize::from(self.u8()?);
        self.take(length)
    }

    fn short_string(&mut self) -> std::result::Result<String, ProtocolError> {
        let bytes = self.short_bytes()?;
        let value = std::str::from_utf8(bytes).map_err(|_| {
            ProtocolError::channel(REPLY_SYNTAX_ERROR, "AMQP short string is not UTF-8", 0, 0)
        })?;
        if value
            .bytes()
            .any(|byte| byte == 0 || byte.is_ascii_control())
        {
            return Err(ProtocolError::channel(
                REPLY_SYNTAX_ERROR,
                "AMQP short string contains a control byte",
                0,
                0,
            ));
        }
        Ok(value.to_owned())
    }

    fn long_bytes(&mut self) -> std::result::Result<Vec<u8>, ProtocolError> {
        let length = self.u32()? as usize;
        if length > MAX_TABLE_BYTES {
            return Err(ProtocolError::connection(
                REPLY_RESOURCE_ERROR,
                "AMQP long string exceeds the field limit",
                0,
                0,
            ));
        }
        Ok(self.take(length)?.to_vec())
    }

    fn field_table(
        &mut self,
        depth: usize,
    ) -> std::result::Result<BTreeMap<String, FieldValue>, ProtocolError> {
        if depth > MAX_FIELD_DEPTH {
            return Err(ProtocolError::connection(
                REPLY_RESOURCE_ERROR,
                "AMQP field table nesting is too deep",
                0,
                0,
            ));
        }
        let length = self.u32()? as usize;
        if length > MAX_TABLE_BYTES {
            return Err(ProtocolError::connection(
                REPLY_RESOURCE_ERROR,
                "AMQP field table exceeds the size limit",
                0,
                0,
            ));
        }
        let bytes = self.take(length)?;
        let mut table_reader = Self::new(bytes);
        let mut table = BTreeMap::new();
        while table_reader.remaining() != 0 {
            if table.len() >= MAX_TABLE_ENTRIES {
                return Err(ProtocolError::connection(
                    REPLY_RESOURCE_ERROR,
                    "AMQP field table has too many entries",
                    0,
                    0,
                ));
            }
            let key = table_reader.short_string()?;
            let value = table_reader.field_value(depth + 1)?;
            if table.insert(key, value).is_some() {
                return Err(ProtocolError::connection(
                    REPLY_SYNTAX_ERROR,
                    "AMQP field table contains a duplicate key",
                    0,
                    0,
                ));
            }
        }
        Ok(table)
    }

    fn field_array(&mut self, depth: usize) -> std::result::Result<Vec<FieldValue>, ProtocolError> {
        if depth > MAX_FIELD_DEPTH {
            return Err(ProtocolError::connection(
                REPLY_RESOURCE_ERROR,
                "AMQP field array nesting is too deep",
                0,
                0,
            ));
        }
        let length = self.u32()? as usize;
        if length > MAX_TABLE_BYTES {
            return Err(ProtocolError::connection(
                REPLY_RESOURCE_ERROR,
                "AMQP field array exceeds the size limit",
                0,
                0,
            ));
        }
        let bytes = self.take(length)?;
        let mut array_reader = Self::new(bytes);
        let mut values = Vec::new();
        while array_reader.remaining() != 0 {
            if values.len() >= MAX_TABLE_ENTRIES {
                return Err(ProtocolError::connection(
                    REPLY_RESOURCE_ERROR,
                    "AMQP field array has too many entries",
                    0,
                    0,
                ));
            }
            values.push(array_reader.field_value(depth + 1)?);
        }
        Ok(values)
    }

    fn field_value(&mut self, depth: usize) -> std::result::Result<FieldValue, ProtocolError> {
        let kind = self.u8()?;
        match kind {
            b't' => Ok(FieldValue::Boolean(self.u8()? != 0)),
            b'b' => Ok(FieldValue::I8(self.i8()?)),
            b'B' => Ok(FieldValue::U8(self.u8()?)),
            b's' => Ok(FieldValue::I16(self.i16()?)),
            b'u' => Ok(FieldValue::U16(self.u16()?)),
            b'I' => Ok(FieldValue::I32(self.i32()?)),
            b'i' => Ok(FieldValue::U32(self.u32()?)),
            b'l' => Ok(FieldValue::I64(self.i64()?)),
            b'f' => Ok(FieldValue::F32(self.f32()?)),
            b'd' => Ok(FieldValue::F64(self.f64()?)),
            b'D' => Ok(FieldValue::Decimal {
                scale: self.u8()?,
                value: self.u32()?,
            }),
            b'S' => Ok(FieldValue::LongString(self.long_bytes()?)),
            b'A' => Ok(FieldValue::Array(self.field_array(depth)?)),
            b'T' => Ok(FieldValue::Timestamp(self.u64()?)),
            b'F' => Ok(FieldValue::Table(self.field_table(depth)?)),
            b'V' => Ok(FieldValue::Void),
            b'x' => Ok(FieldValue::Bytes(self.long_bytes()?)),
            _ => Err(ProtocolError::connection(
                REPLY_SYNTAX_ERROR,
                format!("unsupported AMQP field type 0x{kind:02x}"),
                0,
                0,
            )),
        }
    }

    fn finish(&self) -> std::result::Result<(), ProtocolError> {
        if self.remaining() == 0 {
            Ok(())
        } else {
            Err(ProtocolError::connection(
                REPLY_FRAME_ERROR,
                "AMQP method has trailing bytes",
                0,
                0,
            ))
        }
    }
}

#[derive(Clone, Debug, Default)]
struct Writer {
    bytes: Vec<u8>,
}

impl Writer {
    fn new() -> Self {
        Self::default()
    }

    fn method(class_id: u16, method_id: u16) -> Self {
        let mut writer = Self::new();
        writer.u16(class_id);
        writer.u16(method_id);
        writer
    }

    fn as_slice(&self) -> &[u8] {
        &self.bytes
    }

    fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }

    fn i8(&mut self, value: i8) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn i16(&mut self, value: i16) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn i32(&mut self, value: i32) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn i64(&mut self, value: i64) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }

    fn f32(&mut self, value: f32) {
        self.u32(value.to_bits());
    }

    fn f64(&mut self, value: f64) {
        self.u64(value.to_bits());
    }

    fn raw(&mut self, value: &[u8]) {
        self.bytes.extend_from_slice(value);
    }

    fn short_string(&mut self, value: &str) -> Result<()> {
        if value.len() > u8::MAX as usize || value.as_bytes().contains(&0) {
            return Err(Error::new(
                ErrorCode::ProtocolViolation,
                "AMQP short string is oversized or contains NUL",
            ));
        }
        let length = u8::try_from(value.len())
            .map_err(|_| Error::internal("AMQP short string length exceeded u8"))?;
        self.u8(length);
        self.raw(value.as_bytes());
        Ok(())
    }

    fn long_bytes(&mut self, value: &[u8]) -> Result<()> {
        let length = u32::try_from(value.len())
            .map_err(|_| Error::internal("AMQP long string exceeds u32"))?;
        self.u32(length);
        self.raw(value);
        Ok(())
    }

    fn field_table(&mut self, table: &BTreeMap<String, FieldValue>) -> Result<()> {
        self.field_table_at(table, 0)
    }

    fn field_table_at(&mut self, table: &BTreeMap<String, FieldValue>, depth: usize) -> Result<()> {
        if depth > MAX_FIELD_DEPTH {
            return Err(Error::new(
                ErrorCode::ResultBudgetExceeded,
                "AMQP field nesting is too deep",
            ));
        }
        if table.len() > MAX_TABLE_ENTRIES {
            return Err(Error::new(
                ErrorCode::ResultBudgetExceeded,
                "AMQP table has too many entries",
            ));
        }
        let mut fields = Self::new();
        for (key, value) in table {
            fields.short_string(key)?;
            fields.field_value(value, depth + 1)?;
        }
        if fields.bytes.len() > MAX_TABLE_BYTES {
            return Err(Error::new(
                ErrorCode::ResultBudgetExceeded,
                "AMQP table exceeds the size limit",
            ));
        }
        self.long_bytes(fields.as_slice())
    }

    fn field_array(&mut self, values: &[FieldValue], depth: usize) -> Result<()> {
        let mut body = Self::new();
        for value in values {
            body.field_value(value, depth + 1)?;
        }
        self.long_bytes(body.as_slice())
    }

    fn field_value(&mut self, value: &FieldValue, depth: usize) -> Result<()> {
        if depth > MAX_FIELD_DEPTH {
            return Err(Error::new(
                ErrorCode::ResultBudgetExceeded,
                "AMQP field nesting is too deep",
            ));
        }
        match value {
            FieldValue::Boolean(value) => {
                self.u8(b't');
                self.u8(u8::from(*value));
            }
            FieldValue::I8(value) => {
                self.u8(b'b');
                self.i8(*value);
            }
            FieldValue::U8(value) => {
                self.u8(b'B');
                self.u8(*value);
            }
            FieldValue::I16(value) => {
                self.u8(b's');
                self.i16(*value);
            }
            FieldValue::U16(value) => {
                self.u8(b'u');
                self.u16(*value);
            }
            FieldValue::I32(value) => {
                self.u8(b'I');
                self.i32(*value);
            }
            FieldValue::U32(value) => {
                self.u8(b'i');
                self.u32(*value);
            }
            FieldValue::I64(value) => {
                self.u8(b'l');
                self.i64(*value);
            }
            FieldValue::F32(value) => {
                self.u8(b'f');
                self.f32(*value);
            }
            FieldValue::F64(value) => {
                self.u8(b'd');
                self.f64(*value);
            }
            FieldValue::Decimal { scale, value } => {
                self.u8(b'D');
                self.u8(*scale);
                self.u32(*value);
            }
            FieldValue::LongString(value) => {
                self.u8(b'S');
                self.long_bytes(value)?;
            }
            FieldValue::Array(value) => {
                self.u8(b'A');
                self.field_array(value, depth)?;
            }
            FieldValue::Timestamp(value) => {
                self.u8(b'T');
                self.u64(*value);
            }
            FieldValue::Table(value) => {
                self.u8(b'F');
                self.field_table_at(value, depth)?;
            }
            FieldValue::Void => self.u8(b'V'),
            FieldValue::Bytes(value) => {
                self.u8(b'x');
                self.long_bytes(value)?;
            }
        }
        Ok(())
    }
}

fn server_properties() -> BTreeMap<String, FieldValue> {
    BTreeMap::from([
        (
            "product".to_owned(),
            FieldValue::LongString(b"IronGraph Queue".to_vec()),
        ),
        (
            "version".to_owned(),
            FieldValue::LongString(env!("CARGO_PKG_VERSION").as_bytes().to_vec()),
        ),
        (
            "platform".to_owned(),
            FieldValue::LongString(b"Rust".to_vec()),
        ),
        (
            "capabilities".to_owned(),
            FieldValue::Table(BTreeMap::from([
                ("publisher_confirms".to_owned(), FieldValue::Boolean(true)),
                ("basic.nack".to_owned(), FieldValue::Boolean(true)),
                (
                    "consumer_cancel_notify".to_owned(),
                    FieldValue::Boolean(true),
                ),
                ("connection.blocked".to_owned(), FieldValue::Boolean(false)),
            ])),
        ),
    ])
}

#[derive(Clone, Debug, Default)]
struct BasicProperties {
    content_type: Option<String>,
    content_encoding: Option<String>,
    headers: BTreeMap<String, FieldValue>,
    delivery_mode: Option<u8>,
    priority: Option<u8>,
    correlation_id: Option<String>,
    reply_to: Option<String>,
    expiration: Option<String>,
    message_id: Option<String>,
    timestamp: Option<u64>,
    message_type: Option<String>,
    user_id: Option<String>,
    app_id: Option<String>,
}

impl BasicProperties {
    fn parse(reader: &mut Reader<'_>) -> std::result::Result<Self, ProtocolError> {
        let flags = reader.u16()?;
        if flags & 0x0001 != 0 {
            return Err(ProtocolError::connection(
                REPLY_NOT_IMPLEMENTED,
                "extended AMQP content property flags are not supported",
                CLASS_BASIC,
                0,
            ));
        }
        if flags & 0x0006 != 0 {
            return Err(ProtocolError::connection(
                REPLY_SYNTAX_ERROR,
                "reserved AMQP basic property is set",
                CLASS_BASIC,
                0,
            ));
        }
        let mut properties = Self::default();
        if flags & 0x8000 != 0 {
            properties.content_type = Some(reader.short_string()?);
        }
        if flags & 0x4000 != 0 {
            properties.content_encoding = Some(reader.short_string()?);
        }
        if flags & 0x2000 != 0 {
            properties.headers = reader.field_table(0)?;
        }
        if flags & 0x1000 != 0 {
            let value = reader.u8()?;
            if !matches!(value, 1 | 2) {
                return Err(ProtocolError::channel(
                    REPLY_PRECONDITION_FAILED,
                    "delivery-mode must be 1 or 2",
                    CLASS_BASIC,
                    40,
                ));
            }
            properties.delivery_mode = Some(value);
        }
        if flags & 0x0800 != 0 {
            properties.priority = Some(reader.u8()?);
        }
        if flags & 0x0400 != 0 {
            properties.correlation_id = Some(reader.short_string()?);
        }
        if flags & 0x0200 != 0 {
            properties.reply_to = Some(reader.short_string()?);
        }
        if flags & 0x0100 != 0 {
            properties.expiration = Some(reader.short_string()?);
        }
        if flags & 0x0080 != 0 {
            properties.message_id = Some(reader.short_string()?);
        }
        if flags & 0x0040 != 0 {
            properties.timestamp = Some(reader.u64()?);
        }
        if flags & 0x0020 != 0 {
            properties.message_type = Some(reader.short_string()?);
        }
        if flags & 0x0010 != 0 {
            properties.user_id = Some(reader.short_string()?);
        }
        if flags & 0x0008 != 0 {
            properties.app_id = Some(reader.short_string()?);
        }
        reader.finish()?;
        Ok(properties)
    }

    fn encode(&self, writer: &mut Writer) -> Result<()> {
        let mut flags = 0u16;
        flags |= u16::from(self.content_type.is_some()) << 15;
        flags |= u16::from(self.content_encoding.is_some()) << 14;
        flags |= u16::from(!self.headers.is_empty()) << 13;
        flags |= u16::from(self.delivery_mode.is_some()) << 12;
        flags |= u16::from(self.priority.is_some()) << 11;
        flags |= u16::from(self.correlation_id.is_some()) << 10;
        flags |= u16::from(self.reply_to.is_some()) << 9;
        flags |= u16::from(self.expiration.is_some()) << 8;
        flags |= u16::from(self.message_id.is_some()) << 7;
        flags |= u16::from(self.timestamp.is_some()) << 6;
        flags |= u16::from(self.message_type.is_some()) << 5;
        flags |= u16::from(self.user_id.is_some()) << 4;
        flags |= u16::from(self.app_id.is_some()) << 3;
        writer.u16(flags);
        if let Some(value) = &self.content_type {
            writer.short_string(value)?;
        }
        if let Some(value) = &self.content_encoding {
            writer.short_string(value)?;
        }
        if !self.headers.is_empty() {
            writer.field_table(&self.headers)?;
        }
        if let Some(value) = self.delivery_mode {
            writer.u8(value);
        }
        if let Some(value) = self.priority {
            writer.u8(value);
        }
        if let Some(value) = &self.correlation_id {
            writer.short_string(value)?;
        }
        if let Some(value) = &self.reply_to {
            writer.short_string(value)?;
        }
        if let Some(value) = &self.expiration {
            writer.short_string(value)?;
        }
        if let Some(value) = &self.message_id {
            writer.short_string(value)?;
        }
        if let Some(value) = self.timestamp {
            writer.u64(value);
        }
        if let Some(value) = &self.message_type {
            writer.short_string(value)?;
        }
        if let Some(value) = &self.user_id {
            writer.short_string(value)?;
        }
        if let Some(value) = &self.app_id {
            writer.short_string(value)?;
        }
        Ok(())
    }

    fn into_storage(self) -> Result<StoredBasicProperties> {
        let mut properties = BTreeMap::new();
        insert_stored_string(&mut properties, "content_type", self.content_type)?;
        insert_stored_string(&mut properties, "content_encoding", self.content_encoding)?;
        insert_stored_u8(&mut properties, "delivery_mode", self.delivery_mode)?;
        insert_stored_u8(&mut properties, "priority", self.priority)?;
        insert_stored_string(&mut properties, "correlation_id", self.correlation_id)?;
        insert_stored_string(&mut properties, "reply_to", self.reply_to)?;
        insert_stored_string(&mut properties, "expiration", self.expiration)?;
        insert_stored_string(&mut properties, "message_id", self.message_id)?;
        if let Some(value) = self.timestamp {
            properties.insert(
                "timestamp".to_owned(),
                encode_stored(&FieldValue::Timestamp(value))?,
            );
        }
        insert_stored_string(&mut properties, "type", self.message_type)?;
        insert_stored_string(&mut properties, "user_id", self.user_id)?;
        insert_stored_string(&mut properties, "app_id", self.app_id)?;
        let mut headers = BTreeMap::new();
        for (key, value) in self.headers {
            headers.insert(key, encode_stored(&value)?);
        }
        Ok((properties, headers))
    }

    fn from_storage(
        properties: &BTreeMap<String, Vec<u8>>,
        headers: &BTreeMap<String, Vec<u8>>,
    ) -> Self {
        let timestamp = properties
            .get("timestamp")
            .and_then(|bytes| decode_stored(bytes).ok())
            .and_then(|value| match value {
                FieldValue::Timestamp(value) => Some(value),
                _ => None,
            });
        let mut output = Self {
            content_type: stored_string(properties.get("content_type")),
            content_encoding: stored_string(properties.get("content_encoding")),
            delivery_mode: stored_u8(properties.get("delivery_mode")),
            priority: stored_u8(properties.get("priority")),
            correlation_id: stored_string(properties.get("correlation_id")),
            reply_to: stored_string(properties.get("reply_to")),
            expiration: stored_string(properties.get("expiration")),
            message_id: stored_string(properties.get("message_id")),
            timestamp,
            message_type: stored_string(properties.get("type")),
            user_id: stored_string(properties.get("user_id")),
            app_id: stored_string(properties.get("app_id")),
            headers: BTreeMap::new(),
        };
        for (key, value) in headers {
            if let Ok(value) = decode_stored(value) {
                output.headers.insert(key.clone(), value);
            }
        }
        output
    }
}

fn insert_stored_string(
    target: &mut BTreeMap<String, Vec<u8>>,
    key: &str,
    value: Option<String>,
) -> Result<()> {
    if let Some(value) = value {
        target.insert(
            key.to_owned(),
            encode_stored(&FieldValue::LongString(value.into_bytes()))?,
        );
    }
    Ok(())
}

fn insert_stored_u8(
    target: &mut BTreeMap<String, Vec<u8>>,
    key: &str,
    value: Option<u8>,
) -> Result<()> {
    if let Some(value) = value {
        target.insert(key.to_owned(), encode_stored(&FieldValue::U8(value))?);
    }
    Ok(())
}

fn encode_stored(value: &FieldValue) -> Result<Vec<u8>> {
    let mut writer = Writer::new();
    writer.field_value(value, 0)?;
    Ok(writer.bytes)
}

fn decode_stored(bytes: &[u8]) -> std::result::Result<FieldValue, ProtocolError> {
    let mut reader = Reader::new(bytes);
    let value = reader.field_value(0)?;
    reader.finish()?;
    Ok(value)
}

fn stored_string(value: Option<&Vec<u8>>) -> Option<String> {
    value
        .and_then(|bytes| decode_stored(bytes).ok())
        .and_then(|value| match value {
            FieldValue::LongString(bytes) => String::from_utf8(bytes).ok(),
            _ => None,
        })
}

fn stored_u8(value: Option<&Vec<u8>>) -> Option<u8> {
    value
        .and_then(|bytes| decode_stored(bytes).ok())
        .and_then(|value| match value {
            FieldValue::U8(value) => Some(value),
            _ => None,
        })
}

async fn handle_frame<W: AsyncWrite + Unpin>(
    frame: Frame,
    state: &mut ConnectionState,
    coordinator: &Arc<dyn BrokerCoordinator>,
    writer: &mut W,
) -> std::result::Result<HandleOutcome, ProtocolError> {
    let channel = frame.channel;
    let result = match frame.kind {
        FRAME_HEARTBEAT => {
            if state.phase == ConnectionPhase::Open {
                Ok(HandleOutcome::NoResponse)
            } else {
                Err(ProtocolError::connection(
                    REPLY_UNEXPECTED_FRAME,
                    "heartbeat received before connection.open",
                    CLASS_CONNECTION,
                    0,
                ))
            }
        }
        FRAME_METHOD => {
            let mut reader = Reader::new(&frame.payload);
            let class_id = reader.u16()?;
            let method_id = reader.u16()?;
            handle_method(
                channel,
                class_id,
                method_id,
                &mut reader,
                state,
                coordinator,
                writer,
            )
            .await
        }
        FRAME_HEADER => {
            handle_content_header(channel, &frame.payload, state, coordinator, writer).await
        }
        FRAME_BODY => {
            handle_content_body(channel, &frame.payload, state, coordinator, writer).await
        }
        _ => Err(ProtocolError::connection(
            REPLY_FRAME_ERROR,
            "unsupported AMQP frame type",
            0,
            0,
        )),
    };
    result.map_err(|error| error.on_channel(channel))
}

#[allow(clippy::too_many_arguments)]
async fn handle_method<W: AsyncWrite + Unpin>(
    channel: u16,
    class_id: u16,
    method_id: u16,
    reader: &mut Reader<'_>,
    state: &mut ConnectionState,
    coordinator: &Arc<dyn BrokerCoordinator>,
    writer: &mut W,
) -> std::result::Result<HandleOutcome, ProtocolError> {
    if state.phase != ConnectionPhase::Open {
        return handle_handshake(channel, class_id, method_id, reader, state, writer).await;
    }
    if channel == 0 {
        if class_id != CLASS_CONNECTION {
            return Err(ProtocolError::connection(
                REPLY_COMMAND_INVALID,
                "only connection-class methods are valid on channel zero",
                class_id,
                method_id,
            ));
        }
        return handle_connection_method(method_id, reader, state, writer).await;
    }
    if channel > state.negotiated_channel_max {
        return Err(ProtocolError::connection(
            REPLY_CHANNEL_ERROR,
            "channel exceeds negotiated channel-max",
            class_id,
            method_id,
        ));
    }
    if class_id == CLASS_CHANNEL && method_id == 41 && state.closing_channels.remove(&channel) {
        reader.finish()?;
        return Ok(HandleOutcome::NoResponse);
    }
    if state.closing_channels.contains(&channel) {
        return Err(ProtocolError::connection(
            REPLY_CHANNEL_ERROR,
            "method received while channel close is in progress",
            class_id,
            method_id,
        ));
    }
    if class_id == CLASS_CHANNEL && method_id == 10 {
        return channel_open(channel, reader, state, writer).await;
    }
    if !state.channels.contains_key(&channel) {
        return Err(ProtocolError::connection(
            REPLY_CHANNEL_ERROR,
            "method received on a channel that is not open",
            class_id,
            method_id,
        ));
    }
    if state
        .channels
        .get(&channel)
        .is_some_and(|channel_state| channel_state.pending_publish.is_some())
    {
        return Err(ProtocolError::channel(
            REPLY_UNEXPECTED_FRAME,
            "method received before publish content completed",
            class_id,
            method_id,
        ));
    }

    match (class_id, method_id) {
        (CLASS_CHANNEL, 20) => channel_flow(channel, reader, state, writer).await,
        (CLASS_CHANNEL, 40) => channel_close(channel, reader, state, coordinator, writer).await,
        (CLASS_CHANNEL, 41) => {
            reader.finish()?;
            state.channels.remove(&channel);
            Ok(HandleOutcome::Continue)
        }
        (CLASS_EXCHANGE, 10) => exchange_declare(channel, reader, state, coordinator, writer).await,
        (CLASS_EXCHANGE, 20) => exchange_delete(channel, reader, state, coordinator, writer).await,
        (CLASS_QUEUE, 10) => queue_declare(channel, reader, state, coordinator, writer).await,
        (CLASS_QUEUE, 20) => queue_bind(channel, reader, state, coordinator, writer).await,
        (CLASS_QUEUE, 30) => queue_purge(channel, reader, state, coordinator, writer).await,
        (CLASS_QUEUE, 40) => queue_delete(channel, reader, state, coordinator, writer).await,
        (CLASS_QUEUE, 50) => queue_unbind(channel, reader, state, coordinator, writer).await,
        (CLASS_BASIC, 10) => basic_qos(channel, reader, state, writer).await,
        (CLASS_BASIC, 20) => basic_consume(channel, reader, state, coordinator, writer).await,
        (CLASS_BASIC, 30) => basic_cancel(channel, reader, state, coordinator, writer).await,
        (CLASS_BASIC, 40) => basic_publish(channel, reader, state),
        (CLASS_BASIC, 70) => basic_get(channel, reader, state, coordinator, writer).await,
        (CLASS_BASIC, 80) => basic_ack(channel, reader, state, coordinator).await,
        (CLASS_BASIC, 90) => basic_reject(channel, reader, state, coordinator).await,
        (CLASS_BASIC, 100) => basic_recover_async(channel, reader, state, coordinator).await,
        (CLASS_BASIC, 110) => basic_recover(channel, reader, state, coordinator, writer).await,
        (CLASS_BASIC, 120) => basic_nack(channel, reader, state, coordinator).await,
        (CLASS_CONFIRM, 10) => confirm_select(channel, reader, state, writer).await,
        _ => Err(ProtocolError::channel(
            REPLY_NOT_IMPLEMENTED,
            format!("AMQP method {class_id}.{method_id} is not implemented"),
            class_id,
            method_id,
        )),
    }
}

#[allow(clippy::too_many_lines)]
async fn handle_handshake<W: AsyncWrite + Unpin>(
    channel: u16,
    class_id: u16,
    method_id: u16,
    reader: &mut Reader<'_>,
    state: &mut ConnectionState,
    writer: &mut W,
) -> std::result::Result<HandleOutcome, ProtocolError> {
    if channel != 0 || class_id != CLASS_CONNECTION {
        return Err(ProtocolError::connection(
            REPLY_UNEXPECTED_FRAME,
            "connection handshake method must use channel zero",
            class_id,
            method_id,
        ));
    }
    match (state.phase, method_id) {
        (ConnectionPhase::AwaitStartOk, 11) => {
            let properties = reader.field_table(0)?;
            state.client_name = properties
                .get("connection_name")
                .and_then(FieldValue::as_utf8)
                .map(ToOwned::to_owned);
            let mechanism = reader.short_string()?;
            let response = reader.long_bytes()?;
            let locale = reader.short_string()?;
            reader.finish()?;
            match (mechanism.as_str(), state.trust) {
                ("EXTERNAL", _) => {}
                ("PLAIN", TransportTrust::Loopback) => {
                    // RFC 4616: authzid NUL authcid NUL password. The credentials carry no
                    // authority here — the listener is already loopback-only and this product has
                    // no database login — but a malformed blob still indicates a confused client,
                    // so the shape is checked rather than ignored.
                    if response.split(|byte| *byte == 0).count() != 3 {
                        return Err(ProtocolError::connection(
                            REPLY_ACCESS_REFUSED,
                            "PLAIN response must be authzid, authcid, and password",
                            CLASS_CONNECTION,
                            11,
                        ));
                    }
                }
                _ => {
                    return Err(ProtocolError::connection(
                        REPLY_ACCESS_REFUSED,
                        "this transport accepts only the EXTERNAL mechanism",
                        CLASS_CONNECTION,
                        11,
                    ));
                }
            }
            if locale != "en_US" {
                return Err(ProtocolError::connection(
                    REPLY_NOT_ALLOWED,
                    "only the en_US AMQP locale is supported",
                    CLASS_CONNECTION,
                    11,
                ));
            }
            let mut tune = Writer::method(CLASS_CONNECTION, 30);
            tune.u16(SERVER_CHANNEL_MAX);
            tune.u32(SERVER_FRAME_MAX);
            tune.u16(SERVER_HEARTBEAT);
            write_method(writer, 0, &tune, SERVER_FRAME_MAX)
                .await
                .map_err(|error| {
                    ProtocolError::connection(REPLY_INTERNAL_ERROR, error.to_string(), 10, 30)
                })?;
            state.phase = ConnectionPhase::AwaitTuneOk;
            Ok(HandleOutcome::Continue)
        }
        (ConnectionPhase::AwaitTuneOk, 31) => {
            let channel_max = reader.u16()?;
            let frame_max = reader.u32()?;
            let heartbeat = reader.u16()?;
            reader.finish()?;
            state.negotiated_channel_max = if channel_max == 0 {
                SERVER_CHANNEL_MAX
            } else if channel_max <= SERVER_CHANNEL_MAX {
                channel_max
            } else {
                return Err(ProtocolError::connection(
                    REPLY_NOT_ALLOWED,
                    "client channel-max exceeds server offer",
                    CLASS_CONNECTION,
                    31,
                ));
            };
            state.negotiated_frame_max = if frame_max == 0 {
                SERVER_FRAME_MAX
            } else if (MIN_FRAME_MAX..=SERVER_FRAME_MAX).contains(&frame_max) {
                frame_max
            } else {
                return Err(ProtocolError::connection(
                    REPLY_NOT_ALLOWED,
                    "client frame-max is outside the server offer",
                    CLASS_CONNECTION,
                    31,
                ));
            };
            if heartbeat > SERVER_HEARTBEAT {
                return Err(ProtocolError::connection(
                    REPLY_NOT_ALLOWED,
                    "client heartbeat exceeds server offer",
                    CLASS_CONNECTION,
                    31,
                ));
            }
            state.heartbeat = heartbeat;
            state.phase = ConnectionPhase::AwaitOpen;
            Ok(HandleOutcome::NoResponse)
        }
        (ConnectionPhase::AwaitOpen, 40) => {
            let virtual_host = reader.short_string()?;
            let _reserved = reader.short_string()?;
            let flags = reader.u8()?;
            reader.finish()?;
            if flags & !0x01 != 0 {
                return Err(ProtocolError::connection(
                    REPLY_SYNTAX_ERROR,
                    "connection.open reserved bits are set",
                    CLASS_CONNECTION,
                    40,
                ));
            }
            let project_path = format!("/{}", state.project);
            if virtual_host != "/"
                && virtual_host != project_path
                && virtual_host != state.project.to_string()
            {
                return Err(ProtocolError::connection(
                    REPLY_INVALID_PATH,
                    "virtual host does not map to the authenticated project",
                    CLASS_CONNECTION,
                    40,
                ));
            }
            let mut open_ok = Writer::method(CLASS_CONNECTION, 41);
            open_ok.short_string("").map_err(|error| {
                ProtocolError::connection(REPLY_INTERNAL_ERROR, error.to_string(), 10, 41)
            })?;
            write_method(writer, 0, &open_ok, state.negotiated_frame_max)
                .await
                .map_err(|error| {
                    ProtocolError::connection(REPLY_INTERNAL_ERROR, error.to_string(), 10, 41)
                })?;
            state.phase = ConnectionPhase::Open;
            Ok(HandleOutcome::Continue)
        }
        _ => Err(ProtocolError::connection(
            REPLY_UNEXPECTED_FRAME,
            "unexpected AMQP connection handshake method",
            class_id,
            method_id,
        )),
    }
}

async fn handle_connection_method<W: AsyncWrite + Unpin>(
    method_id: u16,
    reader: &mut Reader<'_>,
    state: &ConnectionState,
    writer: &mut W,
) -> std::result::Result<HandleOutcome, ProtocolError> {
    match method_id {
        50 => {
            let _code = reader.u16()?;
            let _text = reader.short_string()?;
            let _class = reader.u16()?;
            let _method = reader.u16()?;
            reader.finish()?;
            let close_ok = Writer::method(CLASS_CONNECTION, 51);
            write_method(writer, 0, &close_ok, state.negotiated_frame_max)
                .await
                .map_err(|error| {
                    ProtocolError::connection(REPLY_INTERNAL_ERROR, error.to_string(), 10, 51)
                })?;
            Ok(HandleOutcome::Close)
        }
        51 => {
            reader.finish()?;
            Ok(HandleOutcome::Close)
        }
        _ => Err(ProtocolError::connection(
            REPLY_COMMAND_INVALID,
            "unsupported connection method",
            CLASS_CONNECTION,
            method_id,
        )),
    }
}

async fn channel_open<W: AsyncWrite + Unpin>(
    channel: u16,
    reader: &mut Reader<'_>,
    state: &mut ConnectionState,
    writer: &mut W,
) -> std::result::Result<HandleOutcome, ProtocolError> {
    let reserved = reader.short_string()?;
    reader.finish()?;
    if !reserved.is_empty() {
        return Err(ProtocolError::channel(
            REPLY_SYNTAX_ERROR,
            "channel.open reserved field must be empty",
            CLASS_CHANNEL,
            10,
        ));
    }
    if state.channels.contains_key(&channel) {
        return Err(ProtocolError::channel(
            REPLY_CHANNEL_ERROR,
            "channel is already open",
            CLASS_CHANNEL,
            10,
        ));
    }
    if state.channels.len() >= MAX_CHANNELS {
        return Err(ProtocolError::connection(
            REPLY_RESOURCE_ERROR,
            "connection channel limit reached",
            CLASS_CHANNEL,
            10,
        ));
    }
    state.channels.insert(channel, ChannelState::new());
    let mut open_ok = Writer::method(CLASS_CHANNEL, 11);
    open_ok.long_bytes(&[]).map_err(|error| {
        ProtocolError::connection(REPLY_INTERNAL_ERROR, error.to_string(), CLASS_CHANNEL, 11)
    })?;
    write_method(writer, channel, &open_ok, state.negotiated_frame_max)
        .await
        .map_err(|error| {
            ProtocolError::connection(REPLY_INTERNAL_ERROR, error.to_string(), 20, 11)
        })?;
    Ok(HandleOutcome::Continue)
}

async fn channel_flow<W: AsyncWrite + Unpin>(
    channel: u16,
    reader: &mut Reader<'_>,
    state: &mut ConnectionState,
    writer: &mut W,
) -> std::result::Result<HandleOutcome, ProtocolError> {
    let flags = reader.u8()?;
    reader.finish()?;
    if flags & !0x01 != 0 {
        return Err(ProtocolError::channel(
            REPLY_SYNTAX_ERROR,
            "channel.flow reserved bits are set",
            CLASS_CHANNEL,
            20,
        ));
    }
    let active = flags & 0x01 != 0;
    let channel_state = state.channels.get_mut(&channel).ok_or_else(|| {
        ProtocolError::connection(REPLY_CHANNEL_ERROR, "channel is not open", 20, 20)
    })?;
    channel_state.flow_active = active;
    let mut flow_ok = Writer::method(CLASS_CHANNEL, 21);
    flow_ok.u8(u8::from(active));
    write_method(writer, channel, &flow_ok, state.negotiated_frame_max)
        .await
        .map_err(|error| {
            ProtocolError::connection(REPLY_INTERNAL_ERROR, error.to_string(), 20, 21)
        })?;
    Ok(HandleOutcome::Continue)
}

async fn channel_close<W: AsyncWrite + Unpin>(
    channel: u16,
    reader: &mut Reader<'_>,
    state: &mut ConnectionState,
    coordinator: &Arc<dyn BrokerCoordinator>,
    writer: &mut W,
) -> std::result::Result<HandleOutcome, ProtocolError> {
    let _code = reader.u16()?;
    let _text = reader.short_string()?;
    let _class = reader.u16()?;
    let _method = reader.u16()?;
    reader.finish()?;
    cleanup_channel(channel, state, coordinator).await;
    state.channels.remove(&channel);
    let close_ok = Writer::method(CLASS_CHANNEL, 41);
    write_method(writer, channel, &close_ok, state.negotiated_frame_max)
        .await
        .map_err(|error| {
            ProtocolError::connection(REPLY_INTERNAL_ERROR, error.to_string(), 20, 41)
        })?;
    Ok(HandleOutcome::Continue)
}

async fn exchange_declare<W: AsyncWrite + Unpin>(
    channel: u16,
    reader: &mut Reader<'_>,
    state: &ConnectionState,
    coordinator: &Arc<dyn BrokerCoordinator>,
    writer: &mut W,
) -> std::result::Result<HandleOutcome, ProtocolError> {
    let ticket = reader.u16()?;
    let exchange = reader.short_string()?;
    let exchange_type = reader.short_string()?;
    let flags = reader.u8()?;
    let arguments = reader.field_table(0)?;
    reader.finish()?;
    if ticket != 0 || flags & !0x1f != 0 {
        return Err(ProtocolError::channel(
            REPLY_SYNTAX_ERROR,
            "exchange.declare has nonzero reserved fields",
            CLASS_EXCHANGE,
            10,
        ));
    }
    if exchange.is_empty() {
        return Err(ProtocolError::channel(
            REPLY_ACCESS_REFUSED,
            "the default exchange cannot be declared",
            CLASS_EXCHANGE,
            10,
        ));
    }
    if flags & 0x04 != 0 || flags & 0x08 != 0 {
        return Err(ProtocolError::channel(
            REPLY_NOT_IMPLEMENTED,
            "auto-delete and internal exchanges are not supported",
            CLASS_EXCHANGE,
            10,
        ));
    }
    if !arguments.is_empty() {
        return Err(ProtocolError::channel(
            REPLY_PRECONDITION_FAILED,
            "exchange arguments are not supported",
            CLASS_EXCHANGE,
            10,
        ));
    }
    let kind = match exchange_type.as_str() {
        "direct" => AmqpExchangeKind::Direct,
        "fanout" => AmqpExchangeKind::Fanout,
        "topic" => AmqpExchangeKind::Topic,
        _ => {
            return Err(ProtocolError::channel(
                REPLY_COMMAND_INVALID,
                "exchange type must be direct, fanout, or topic",
                CLASS_EXCHANGE,
                10,
            ));
        }
    };
    let command = BrokerCommand::CreateExchange {
        project: state.project,
        name: exchange,
        kind,
        durable: flags & 0x02 != 0,
        passive: flags & 0x01 != 0,
    };
    let reply = coordinator_submit(
        Arc::clone(coordinator),
        command,
        CommitAcknowledgement::Published,
    )
    .await
    .map_err(|error| ProtocolError::from_engine(error, CLASS_EXCHANGE, 10))?;
    expect_declared(&reply, CLASS_EXCHANGE, 10)?;
    if flags & 0x10 != 0 {
        return Ok(HandleOutcome::NoResponse);
    }
    let declared = Writer::method(CLASS_EXCHANGE, 11);
    write_method(writer, channel, &declared, state.negotiated_frame_max)
        .await
        .map_err(|error| {
            ProtocolError::connection(REPLY_INTERNAL_ERROR, error.to_string(), 40, 11)
        })?;
    Ok(HandleOutcome::Continue)
}

#[allow(clippy::too_many_lines)]
async fn queue_declare<W: AsyncWrite + Unpin>(
    channel: u16,
    reader: &mut Reader<'_>,
    state: &mut ConnectionState,
    coordinator: &Arc<dyn BrokerCoordinator>,
    writer: &mut W,
) -> std::result::Result<HandleOutcome, ProtocolError> {
    let ticket = reader.u16()?;
    let requested_name = reader.short_string()?;
    let flags = reader.u8()?;
    let arguments = reader.field_table(0)?;
    reader.finish()?;
    if ticket != 0 || flags & !0x1f != 0 {
        return Err(ProtocolError::channel(
            REPLY_SYNTAX_ERROR,
            "queue.declare has nonzero reserved fields",
            CLASS_QUEUE,
            10,
        ));
    }
    let exclusive = flags & 0x04 != 0;
    let auto_delete = flags & 0x08 != 0;
    if requested_name.is_empty() && flags & 0x01 != 0 {
        return Err(ProtocolError::channel(
            REPLY_PRECONDITION_FAILED,
            "passive queue declaration requires a queue name",
            CLASS_QUEUE,
            10,
        ));
    }
    let name = if requested_name.is_empty() {
        format!("amq.gen-{}", ulid::Ulid::new())
    } else {
        requested_name
    };
    let queue_arguments = parse_queue_arguments(&arguments)?;
    let durable = flags & 0x02 != 0;
    if queue_arguments.kind == QueueKind::Stream && !durable {
        return Err(ProtocolError::channel(
            REPLY_PRECONDITION_FAILED,
            "stream queues must be durable",
            CLASS_QUEUE,
            10,
        ));
    }
    let command = BrokerCommand::CreateQueue {
        project: state.project,
        name: name.clone(),
        kind: queue_arguments.kind,
        durable,
        passive: flags & 0x01 != 0,
        dead_letter_exchange: queue_arguments.dead_letter_exchange,
        dead_letter_routing_key: queue_arguments.dead_letter_routing_key,
        retention: queue_arguments.retention,
        // The delivery owner is this connection's stable canonical identity, so it is what an
        // exclusive claim is recorded against and what teardown later matches on.
        exclusive_owner: exclusive.then_some(state.delivery_owner),
        auto_delete,
    };
    let reply = coordinator_submit(
        Arc::clone(coordinator),
        command,
        CommitAcknowledgement::Published,
    )
    .await
    .map_err(|error| ProtocolError::from_engine(error, CLASS_QUEUE, 10))?;
    expect_declared(&reply, CLASS_QUEUE, 10)?;
    if exclusive || auto_delete {
        state.connection_state_registered = true;
    }
    state
        .channels
        .get_mut(&channel)
        .ok_or_else(|| {
            ProtocolError::connection(REPLY_CHANNEL_ERROR, "channel is not open", 50, 10)
        })?
        .last_declared_queue = Some(name.clone());
    if flags & 0x10 != 0 {
        return Ok(HandleOutcome::NoResponse);
    }
    let info = coordinator_queue_info(Arc::clone(coordinator), state.project, name.clone())
        .await
        .map_err(|error| ProtocolError::from_engine(error, CLASS_QUEUE, 10))?
        .ok_or_else(|| {
            ProtocolError::channel(
                REPLY_NOT_FOUND,
                "declared queue is not visible at the applied snapshot",
                CLASS_QUEUE,
                10,
            )
        })?;
    let consumers = state
        .channels
        .values()
        .flat_map(|channel_state| channel_state.consumers.values())
        .filter(|consumer| consumer.queue == name)
        .count();
    let mut declared = Writer::method(CLASS_QUEUE, 11);
    declared.short_string(&name).map_err(|error| {
        ProtocolError::connection(REPLY_INTERNAL_ERROR, error.to_string(), CLASS_QUEUE, 11)
    })?;
    declared.u32(u32::try_from(info.available_count).unwrap_or(u32::MAX));
    declared.u32(u32::try_from(consumers).unwrap_or(u32::MAX));
    write_method(writer, channel, &declared, state.negotiated_frame_max)
        .await
        .map_err(|error| {
            ProtocolError::connection(REPLY_INTERNAL_ERROR, error.to_string(), 50, 11)
        })?;
    Ok(HandleOutcome::Continue)
}

async fn queue_bind<W: AsyncWrite + Unpin>(
    channel: u16,
    reader: &mut Reader<'_>,
    state: &ConnectionState,
    coordinator: &Arc<dyn BrokerCoordinator>,
    writer: &mut W,
) -> std::result::Result<HandleOutcome, ProtocolError> {
    let ticket = reader.u16()?;
    let queue = resolve_queue_name(channel, reader.short_string()?, state, CLASS_QUEUE, 20)?;
    let exchange = reader.short_string()?;
    let routing_key = reader.short_string()?;
    let flags = reader.u8()?;
    let arguments = reader.field_table(0)?;
    reader.finish()?;
    if ticket != 0 || flags & !0x01 != 0 {
        return Err(ProtocolError::channel(
            REPLY_SYNTAX_ERROR,
            "queue.bind has nonzero reserved fields",
            CLASS_QUEUE,
            20,
        ));
    }
    if exchange.is_empty() {
        return Err(ProtocolError::channel(
            REPLY_ACCESS_REFUSED,
            "the default exchange cannot be bound",
            CLASS_QUEUE,
            20,
        ));
    }
    if !arguments.is_empty() {
        return Err(ProtocolError::channel(
            REPLY_PRECONDITION_FAILED,
            "binding arguments are not supported",
            CLASS_QUEUE,
            20,
        ));
    }
    let reply = coordinator_submit(
        Arc::clone(coordinator),
        BrokerCommand::BindQueue {
            project: state.project,
            exchange,
            queue,
            routing_key,
        },
        CommitAcknowledgement::Published,
    )
    .await
    .map_err(|error| ProtocolError::from_engine(error, CLASS_QUEUE, 20))?;
    expect_declared(&reply, CLASS_QUEUE, 20)?;
    if flags & 0x01 != 0 {
        return Ok(HandleOutcome::NoResponse);
    }
    let bound = Writer::method(CLASS_QUEUE, 21);
    write_method(writer, channel, &bound, state.negotiated_frame_max)
        .await
        .map_err(|error| {
            ProtocolError::connection(REPLY_INTERNAL_ERROR, error.to_string(), 50, 21)
        })?;
    Ok(HandleOutcome::Continue)
}

fn resolve_queue_name(
    channel: u16,
    name: String,
    state: &ConnectionState,
    class_id: u16,
    method_id: u16,
) -> std::result::Result<String, ProtocolError> {
    if !name.is_empty() {
        return Ok(name);
    }
    state
        .channels
        .get(&channel)
        .and_then(|channel_state| channel_state.last_declared_queue.clone())
        .ok_or_else(|| {
            ProtocolError::channel(
                REPLY_NOT_FOUND,
                "empty queue name has no preceding queue declaration on this channel",
                class_id,
                method_id,
            )
        })
}

struct QueueArguments {
    kind: QueueKind,
    dead_letter_exchange: Option<String>,
    dead_letter_routing_key: Option<String>,
    retention: RetentionPolicy,
}

fn parse_queue_arguments(
    arguments: &BTreeMap<String, FieldValue>,
) -> std::result::Result<QueueArguments, ProtocolError> {
    let mut kind = QueueKind::Classic;
    let mut dead_letter_exchange = None;
    let mut dead_letter_routing_key = None;
    let mut max_age_ms = None;
    let mut max_bytes = None;
    for (key, value) in arguments {
        match key.as_str() {
            "x-queue-type" => {
                kind = match value.as_utf8() {
                    Some("classic") => QueueKind::Classic,
                    Some("stream") => QueueKind::Stream,
                    _ => {
                        return Err(ProtocolError::channel(
                            REPLY_PRECONDITION_FAILED,
                            "x-queue-type must be classic or stream",
                            CLASS_QUEUE,
                            10,
                        ));
                    }
                };
            }
            "x-dead-letter-exchange" => {
                dead_letter_exchange = Some(required_argument_string(key, value)?);
            }
            "x-dead-letter-routing-key" => {
                dead_letter_routing_key = Some(required_argument_string(key, value)?);
            }
            "x-max-age" => {
                let value = required_argument_string(key, value)?;
                max_age_ms = Some(parse_retention_duration(&value)?);
            }
            "x-message-ttl" => {
                max_age_ms = Some(required_positive_integer(key, value)?);
            }
            "x-max-length-bytes" => {
                max_bytes = Some(required_positive_integer(key, value)?);
            }
            "x-overflow" if value.as_utf8() == Some("drop-head") => {}
            _ => {
                return Err(ProtocolError::channel(
                    REPLY_PRECONDITION_FAILED,
                    format!("unsupported queue argument {key}"),
                    CLASS_QUEUE,
                    10,
                ));
            }
        }
    }
    Ok(QueueArguments {
        kind,
        dead_letter_exchange,
        dead_letter_routing_key,
        retention: RetentionPolicy {
            max_age_ms,
            max_bytes,
        },
    })
}

fn required_argument_string(
    key: &str,
    value: &FieldValue,
) -> std::result::Result<String, ProtocolError> {
    value.as_utf8().map(ToOwned::to_owned).ok_or_else(|| {
        ProtocolError::channel(
            REPLY_PRECONDITION_FAILED,
            format!("queue argument {key} must be a string"),
            CLASS_QUEUE,
            10,
        )
    })
}

fn required_positive_integer(
    key: &str,
    value: &FieldValue,
) -> std::result::Result<u64, ProtocolError> {
    value.as_u64().filter(|value| *value != 0).ok_or_else(|| {
        ProtocolError::channel(
            REPLY_PRECONDITION_FAILED,
            format!("queue argument {key} must be a positive integer"),
            CLASS_QUEUE,
            10,
        )
    })
}

fn parse_retention_duration(value: &str) -> std::result::Result<u64, ProtocolError> {
    let split = value
        .find(|character: char| !character.is_ascii_digit())
        .unwrap_or(value.len());
    let (number, unit) = value.split_at(split);
    let number = number
        .parse::<u64>()
        .ok()
        .filter(|value| *value != 0)
        .ok_or_else(|| {
            ProtocolError::channel(
                REPLY_PRECONDITION_FAILED,
                "x-max-age must start with a positive integer",
                CLASS_QUEUE,
                10,
            )
        })?;
    let multiplier = match unit {
        "ms" => 1,
        "s" => 1_000,
        "m" => 60_000,
        "h" => 3_600_000,
        "D" | "d" => 86_400_000,
        "W" | "w" => 604_800_000,
        _ => {
            return Err(ProtocolError::channel(
                REPLY_PRECONDITION_FAILED,
                "x-max-age unit must be ms, s, m, h, D, or W",
                CLASS_QUEUE,
                10,
            ));
        }
    };
    number.checked_mul(multiplier).ok_or_else(|| {
        ProtocolError::channel(
            REPLY_PRECONDITION_FAILED,
            "x-max-age exceeds u64 milliseconds",
            CLASS_QUEUE,
            10,
        )
    })
}

/// `queue.unbind` (50.50). Unlike every other queue method it carries no `no-wait` bit, so the
/// unbind-ok is unconditional.
async fn queue_unbind<W: AsyncWrite + Unpin>(
    channel: u16,
    reader: &mut Reader<'_>,
    state: &ConnectionState,
    coordinator: &Arc<dyn BrokerCoordinator>,
    writer: &mut W,
) -> std::result::Result<HandleOutcome, ProtocolError> {
    let ticket = reader.u16()?;
    let queue = resolve_queue_name(channel, reader.short_string()?, state, CLASS_QUEUE, 50)?;
    let exchange = reader.short_string()?;
    let routing_key = reader.short_string()?;
    let arguments = reader.field_table(0)?;
    reader.finish()?;
    if ticket != 0 {
        return Err(ProtocolError::channel(
            REPLY_SYNTAX_ERROR,
            "queue.unbind has nonzero reserved fields",
            CLASS_QUEUE,
            50,
        ));
    }
    if exchange.is_empty() {
        return Err(ProtocolError::channel(
            REPLY_ACCESS_REFUSED,
            "the default exchange cannot be unbound",
            CLASS_QUEUE,
            50,
        ));
    }
    if !arguments.is_empty() {
        return Err(ProtocolError::channel(
            REPLY_PRECONDITION_FAILED,
            "binding arguments are not supported",
            CLASS_QUEUE,
            50,
        ));
    }
    let reply = coordinator_submit(
        Arc::clone(coordinator),
        BrokerCommand::UnbindQueue {
            project: state.project,
            exchange,
            queue,
            routing_key,
        },
        CommitAcknowledgement::Published,
    )
    .await
    .map_err(|error| ProtocolError::from_engine(error, CLASS_QUEUE, 50))?;
    if !matches!(reply, BrokerReply::Unbound) {
        return Err(ProtocolError::channel(
            REPLY_INTERNAL_ERROR,
            "broker returned an invalid unbind reply",
            CLASS_QUEUE,
            50,
        ));
    }
    let unbound = Writer::method(CLASS_QUEUE, 51);
    write_method(writer, channel, &unbound, state.negotiated_frame_max)
        .await
        .map_err(|error| {
            ProtocolError::connection(REPLY_INTERNAL_ERROR, error.to_string(), CLASS_QUEUE, 51)
        })?;
    Ok(HandleOutcome::Continue)
}

/// `queue.purge` (50.30) and `queue.delete` (50.40). Both reply with the number of ready messages
/// the call discarded, so they share the submission and response shape.
async fn queue_purge<W: AsyncWrite + Unpin>(
    channel: u16,
    reader: &mut Reader<'_>,
    state: &ConnectionState,
    coordinator: &Arc<dyn BrokerCoordinator>,
    writer: &mut W,
) -> std::result::Result<HandleOutcome, ProtocolError> {
    let ticket = reader.u16()?;
    let queue = resolve_queue_name(channel, reader.short_string()?, state, CLASS_QUEUE, 30)?;
    let flags = reader.u8()?;
    reader.finish()?;
    if ticket != 0 || flags & !0x01 != 0 {
        return Err(ProtocolError::channel(
            REPLY_SYNTAX_ERROR,
            "queue.purge has nonzero reserved fields",
            CLASS_QUEUE,
            30,
        ));
    }
    let reply = coordinator_submit(
        Arc::clone(coordinator),
        BrokerCommand::PurgeQueue {
            project: state.project,
            name: queue,
        },
        CommitAcknowledgement::Published,
    )
    .await
    .map_err(|error| ProtocolError::from_engine(error, CLASS_QUEUE, 30))?;
    let discarded = expect_discarded(&reply, CLASS_QUEUE, 30)?;
    if flags & 0x01 != 0 {
        return Ok(HandleOutcome::NoResponse);
    }
    let mut purged = Writer::method(CLASS_QUEUE, 31);
    purged.u32(discarded);
    write_method(writer, channel, &purged, state.negotiated_frame_max)
        .await
        .map_err(|error| {
            ProtocolError::connection(REPLY_INTERNAL_ERROR, error.to_string(), CLASS_QUEUE, 31)
        })?;
    Ok(HandleOutcome::Continue)
}

async fn queue_delete<W: AsyncWrite + Unpin>(
    channel: u16,
    reader: &mut Reader<'_>,
    state: &ConnectionState,
    coordinator: &Arc<dyn BrokerCoordinator>,
    writer: &mut W,
) -> std::result::Result<HandleOutcome, ProtocolError> {
    let ticket = reader.u16()?;
    let queue = resolve_queue_name(channel, reader.short_string()?, state, CLASS_QUEUE, 40)?;
    let flags = reader.u8()?;
    reader.finish()?;
    if ticket != 0 || flags & !0x07 != 0 {
        return Err(ProtocolError::channel(
            REPLY_SYNTAX_ERROR,
            "queue.delete has nonzero reserved fields",
            CLASS_QUEUE,
            40,
        ));
    }
    let reply = coordinator_submit(
        Arc::clone(coordinator),
        BrokerCommand::DeleteQueue {
            project: state.project,
            name: queue,
            if_unused: flags & 0x01 != 0,
            if_empty: flags & 0x02 != 0,
        },
        CommitAcknowledgement::Published,
    )
    .await
    .map_err(|error| ProtocolError::from_engine(error, CLASS_QUEUE, 40))?;
    let discarded = expect_discarded(&reply, CLASS_QUEUE, 40)?;
    if flags & 0x04 != 0 {
        return Ok(HandleOutcome::NoResponse);
    }
    let mut deleted = Writer::method(CLASS_QUEUE, 41);
    deleted.u32(discarded);
    write_method(writer, channel, &deleted, state.negotiated_frame_max)
        .await
        .map_err(|error| {
            ProtocolError::connection(REPLY_INTERNAL_ERROR, error.to_string(), CLASS_QUEUE, 41)
        })?;
    Ok(HandleOutcome::Continue)
}

/// `exchange.delete` (40.20).
async fn exchange_delete<W: AsyncWrite + Unpin>(
    channel: u16,
    reader: &mut Reader<'_>,
    state: &ConnectionState,
    coordinator: &Arc<dyn BrokerCoordinator>,
    writer: &mut W,
) -> std::result::Result<HandleOutcome, ProtocolError> {
    let ticket = reader.u16()?;
    let exchange = reader.short_string()?;
    let flags = reader.u8()?;
    reader.finish()?;
    if ticket != 0 || flags & !0x03 != 0 {
        return Err(ProtocolError::channel(
            REPLY_SYNTAX_ERROR,
            "exchange.delete has nonzero reserved fields",
            CLASS_EXCHANGE,
            20,
        ));
    }
    if exchange.is_empty() {
        return Err(ProtocolError::channel(
            REPLY_ACCESS_REFUSED,
            "the default exchange cannot be deleted",
            CLASS_EXCHANGE,
            20,
        ));
    }
    let reply = coordinator_submit(
        Arc::clone(coordinator),
        BrokerCommand::DeleteExchange {
            project: state.project,
            name: exchange,
            if_unused: flags & 0x01 != 0,
        },
        CommitAcknowledgement::Published,
    )
    .await
    .map_err(|error| ProtocolError::from_engine(error, CLASS_EXCHANGE, 20))?;
    if !matches!(reply, BrokerReply::Deleted) {
        return Err(ProtocolError::channel(
            REPLY_INTERNAL_ERROR,
            "broker returned an invalid delete reply",
            CLASS_EXCHANGE,
            20,
        ));
    }
    if flags & 0x02 != 0 {
        return Ok(HandleOutcome::NoResponse);
    }
    let deleted = Writer::method(CLASS_EXCHANGE, 21);
    write_method(writer, channel, &deleted, state.negotiated_frame_max)
        .await
        .map_err(|error| {
            ProtocolError::connection(REPLY_INTERNAL_ERROR, error.to_string(), CLASS_EXCHANGE, 21)
        })?;
    Ok(HandleOutcome::Continue)
}

fn expect_discarded(
    reply: &BrokerReply,
    class_id: u16,
    method_id: u16,
) -> std::result::Result<u32, ProtocolError> {
    match reply {
        BrokerReply::MessagesDiscarded { message_count } => Ok(*message_count),
        _ => Err(ProtocolError::channel(
            REPLY_INTERNAL_ERROR,
            "broker returned an invalid message-count reply",
            class_id,
            method_id,
        )),
    }
}

fn expect_declared(
    reply: &BrokerReply,
    class_id: u16,
    method_id: u16,
) -> std::result::Result<(), ProtocolError> {
    if matches!(reply, BrokerReply::Declared) {
        Ok(())
    } else {
        Err(ProtocolError::channel(
            REPLY_INTERNAL_ERROR,
            "broker returned an invalid declaration reply",
            class_id,
            method_id,
        ))
    }
}

async fn coordinator_submit(
    coordinator: Arc<dyn BrokerCoordinator>,
    command: BrokerCommand,
    wait: CommitAcknowledgement,
) -> Result<BrokerReply> {
    coordinator_submit_inner(coordinator, command, wait, None).await
}

async fn coordinator_submit_reserved(
    coordinator: Arc<dyn BrokerCoordinator>,
    command: BrokerCommand,
    wait: CommitAcknowledgement,
    memory: &BrokerMemoryReservation,
) -> Result<BrokerReply> {
    coordinator_submit_inner(coordinator, command, wait, Some(memory)).await
}

async fn coordinator_submit_inner(
    coordinator: Arc<dyn BrokerCoordinator>,
    command: BrokerCommand,
    wait: CommitAcknowledgement,
    memory: Option<&BrokerMemoryReservation>,
) -> Result<BrokerReply> {
    let work = move || coordinator.submit(command, wait).map(|commit| commit.reply);
    if let Some(memory) = memory {
        memory
            .spawn_blocking(work)
            .await
            .map_err(|error| Error::internal(format!("broker coordinator task failed: {error}")))?
    } else {
        tokio::task::spawn_blocking(work)
            .await
            .map_err(|error| Error::internal(format!("broker coordinator task failed: {error}")))?
    }
}

async fn coordinator_queue_info(
    coordinator: Arc<dyn BrokerCoordinator>,
    project: ProjectId,
    queue: String,
) -> Result<Option<super::engine::QueueInfo>> {
    tokio::task::spawn_blocking(move || coordinator.queue_info(project, &queue))
        .await
        .map_err(|error| Error::internal(format!("broker queue-info task failed: {error}")))?
}

#[allow(clippy::too_many_arguments)]
async fn coordinator_fetch_stream_queue(
    coordinator: Arc<dyn BrokerCoordinator>,
    project: ProjectId,
    queue: String,
    offset: StreamOffset,
    maximum: usize,
    consumer: u64,
    automatic_ack: bool,
    memory: &BrokerMemoryReservation,
) -> Result<Vec<Delivery>> {
    memory
        .spawn_blocking(move || {
            coordinator.fetch_stream_queue(
                project,
                &queue,
                offset,
                maximum,
                consumer,
                automatic_ack,
            )
        })
        .await
        .map_err(|error| Error::internal(format!("broker stream-fetch task failed: {error}")))?
}

async fn basic_qos<W: AsyncWrite + Unpin>(
    channel: u16,
    reader: &mut Reader<'_>,
    state: &mut ConnectionState,
    writer: &mut W,
) -> std::result::Result<HandleOutcome, ProtocolError> {
    let prefetch_size = reader.u32()?;
    let prefetch_count = reader.u16()?;
    let flags = reader.u8()?;
    reader.finish()?;
    if flags & !0x01 != 0 {
        return Err(ProtocolError::channel(
            REPLY_SYNTAX_ERROR,
            "basic.qos reserved bits are set",
            CLASS_BASIC,
            10,
        ));
    }
    if prefetch_size != 0 {
        return Err(ProtocolError::channel(
            REPLY_NOT_IMPLEMENTED,
            "byte-based prefetch is not supported; use prefetch-count",
            CLASS_BASIC,
            10,
        ));
    }
    let channel_state = state.channels.get_mut(&channel).ok_or_else(|| {
        ProtocolError::connection(REPLY_CHANNEL_ERROR, "channel is not open", 60, 10)
    })?;
    channel_state.prefetch_count = prefetch_count;
    channel_state.prefetch_global = flags & 0x01 != 0;
    let qos_ok = Writer::method(CLASS_BASIC, 11);
    write_method(writer, channel, &qos_ok, state.negotiated_frame_max)
        .await
        .map_err(|error| {
            ProtocolError::connection(REPLY_INTERNAL_ERROR, error.to_string(), 60, 11)
        })?;
    Ok(HandleOutcome::Continue)
}

async fn basic_consume<W: AsyncWrite + Unpin>(
    channel: u16,
    reader: &mut Reader<'_>,
    state: &mut ConnectionState,
    coordinator: &Arc<dyn BrokerCoordinator>,
    writer: &mut W,
) -> std::result::Result<HandleOutcome, ProtocolError> {
    let ticket = reader.u16()?;
    let queue = resolve_queue_name(channel, reader.short_string()?, state, CLASS_BASIC, 20)?;
    let requested_tag = reader.short_string()?;
    let flags = reader.u8()?;
    let arguments = reader.field_table(0)?;
    reader.finish()?;
    if ticket != 0 || flags & !0x0f != 0 {
        return Err(ProtocolError::channel(
            REPLY_SYNTAX_ERROR,
            "basic.consume has nonzero reserved fields",
            CLASS_BASIC,
            20,
        ));
    }
    if flags & 0x01 != 0 {
        return Err(ProtocolError::channel(
            REPLY_NOT_IMPLEMENTED,
            "no-local consumers are not supported",
            CLASS_BASIC,
            20,
        ));
    }
    if flags & 0x04 != 0 {
        return Err(ProtocolError::channel(
            REPLY_NOT_IMPLEMENTED,
            "exclusive consumers are not supported",
            CLASS_BASIC,
            20,
        ));
    }
    if state.consumer_count >= MAX_CONSUMERS_PER_CONNECTION {
        return Err(ProtocolError::connection(
            REPLY_RESOURCE_ERROR,
            "connection consumer limit reached",
            CLASS_BASIC,
            20,
        ));
    }
    let info = coordinator_queue_info(Arc::clone(coordinator), state.project, queue.clone())
        .await
        .map_err(|error| ProtocolError::from_engine(error, CLASS_BASIC, 20))?
        .ok_or_else(|| {
            ProtocolError::channel(REPLY_NOT_FOUND, "queue does not exist", CLASS_BASIC, 20)
        })?;
    let offset = parse_consumer_arguments(info.kind, &arguments)?;
    let consumer_id = state.allocate_consumer()?;
    // Registration is canonical, so an exclusive queue can refuse a consumer from another
    // connection and an auto-delete queue can tell when its last one has gone.
    coordinator_submit(
        Arc::clone(coordinator),
        BrokerCommand::RegisterConsumer {
            project: state.project,
            queue: queue.clone(),
            owner: state.delivery_owner,
            consumer: consumer_id,
        },
        CommitAcknowledgement::Published,
    )
    .await
    .map_err(|error| ProtocolError::from_engine(error, CLASS_BASIC, 20))?;
    state.connection_state_registered = true;
    let tag = if requested_tag.is_empty() {
        format!("ctag-{consumer_id}")
    } else {
        requested_tag
    };
    let channel_state = state.channels.get_mut(&channel).ok_or_else(|| {
        ProtocolError::connection(REPLY_CHANNEL_ERROR, "channel is not open", 60, 20)
    })?;
    if channel_state.consumers.len() >= MAX_CONSUMERS_PER_CHANNEL {
        return Err(ProtocolError::channel(
            REPLY_RESOURCE_ERROR,
            "channel consumer limit reached",
            CLASS_BASIC,
            20,
        ));
    }
    if channel_state.consumers.contains_key(&tag) {
        return Err(ProtocolError::channel(
            REPLY_NOT_ALLOWED,
            "consumer tag is already in use on this channel",
            CLASS_BASIC,
            20,
        ));
    }
    channel_state.consumers.insert(
        tag.clone(),
        ConsumerState {
            id: consumer_id,
            queue,
            automatic_ack: flags & 0x02 != 0,
            queue_kind: info.kind,
            next_offset: offset,
            outstanding: 0,
            poll_after: Instant::now(),
            idle_poll_interval: PUMP_INTERVAL,
        },
    );
    state.consumer_count += 1;
    if flags & 0x08 != 0 {
        return Ok(HandleOutcome::NoResponse);
    }
    let mut consume_ok = Writer::method(CLASS_BASIC, 21);
    consume_ok.short_string(&tag).map_err(|error| {
        ProtocolError::connection(REPLY_INTERNAL_ERROR, error.to_string(), CLASS_BASIC, 21)
    })?;
    write_method(writer, channel, &consume_ok, state.negotiated_frame_max)
        .await
        .map_err(|error| {
            ProtocolError::connection(REPLY_INTERNAL_ERROR, error.to_string(), 60, 21)
        })?;
    Ok(HandleOutcome::Continue)
}

fn parse_consumer_arguments(
    queue_kind: QueueKind,
    arguments: &BTreeMap<String, FieldValue>,
) -> std::result::Result<StreamOffset, ProtocolError> {
    if queue_kind == QueueKind::Classic {
        if arguments.is_empty() {
            return Ok(StreamOffset::First);
        }
        return Err(ProtocolError::channel(
            REPLY_PRECONDITION_FAILED,
            "classic consumers do not accept stream arguments",
            CLASS_BASIC,
            20,
        ));
    }
    if arguments.keys().any(|key| key != "x-stream-offset") {
        return Err(ProtocolError::channel(
            REPLY_PRECONDITION_FAILED,
            "unsupported stream consumer argument",
            CLASS_BASIC,
            20,
        ));
    }
    let Some(value) = arguments.get("x-stream-offset") else {
        return Ok(StreamOffset::Next);
    };
    match value {
        FieldValue::LongString(value) => match std::str::from_utf8(value).ok() {
            Some("first") => Ok(StreamOffset::First),
            Some("last") => Ok(StreamOffset::Last),
            Some("next") => Ok(StreamOffset::Next),
            _ => Err(ProtocolError::channel(
                REPLY_PRECONDITION_FAILED,
                "x-stream-offset string must be first, last, or next",
                CLASS_BASIC,
                20,
            )),
        },
        FieldValue::Timestamp(seconds) => {
            let milliseconds = seconds.checked_mul(1_000).ok_or_else(|| {
                ProtocolError::channel(
                    REPLY_PRECONDITION_FAILED,
                    "x-stream-offset timestamp is too large",
                    CLASS_BASIC,
                    20,
                )
            })?;
            Ok(StreamOffset::Timestamp(
                i64::try_from(milliseconds).map_err(|_| {
                    ProtocolError::channel(
                        REPLY_PRECONDITION_FAILED,
                        "x-stream-offset timestamp exceeds i64",
                        CLASS_BASIC,
                        20,
                    )
                })?,
            ))
        }
        _ => value.as_u64().map(StreamOffset::Absolute).ok_or_else(|| {
            ProtocolError::channel(
                REPLY_PRECONDITION_FAILED,
                "x-stream-offset must be first, last, next, an offset, or a timestamp",
                CLASS_BASIC,
                20,
            )
        }),
    }
}

async fn basic_cancel<W: AsyncWrite + Unpin>(
    channel: u16,
    reader: &mut Reader<'_>,
    state: &mut ConnectionState,
    coordinator: &Arc<dyn BrokerCoordinator>,
    writer: &mut W,
) -> std::result::Result<HandleOutcome, ProtocolError> {
    let tag = reader.short_string()?;
    let flags = reader.u8()?;
    reader.finish()?;
    if flags & !0x01 != 0 {
        return Err(ProtocolError::channel(
            REPLY_SYNTAX_ERROR,
            "basic.cancel reserved bits are set",
            CLASS_BASIC,
            30,
        ));
    }
    let consumer = state
        .channels
        .get_mut(&channel)
        .and_then(|channel_state| channel_state.consumers.remove(&tag))
        .ok_or_else(|| {
            ProtocolError::channel(
                REPLY_NOT_FOUND,
                "consumer tag does not exist",
                CLASS_BASIC,
                30,
            )
        })?;
    state.consumer_count = state.consumer_count.saturating_sub(1);
    let tags = state
        .channels
        .get(&channel)
        .map(|channel_state| {
            channel_state
                .outstanding
                .iter()
                .filter_map(|(tag, delivery)| (delivery.consumer_id == consumer.id).then_some(*tag))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    settle_tags(channel, &tags, false, true, state, coordinator).await?;
    // Unregistering last is what collects an auto-delete queue whose final consumer this was.
    coordinator_submit(
        Arc::clone(coordinator),
        BrokerCommand::UnregisterConsumer {
            project: state.project,
            queue: consumer.queue.clone(),
            owner: state.delivery_owner,
            consumer: consumer.id,
        },
        CommitAcknowledgement::Published,
    )
    .await
    .map_err(|error| ProtocolError::from_engine(error, CLASS_BASIC, 30))?;
    if flags & 0x01 != 0 {
        return Ok(HandleOutcome::NoResponse);
    }
    let mut cancel_ok = Writer::method(CLASS_BASIC, 31);
    cancel_ok.short_string(&tag).map_err(|error| {
        ProtocolError::connection(REPLY_INTERNAL_ERROR, error.to_string(), CLASS_BASIC, 31)
    })?;
    write_method(writer, channel, &cancel_ok, state.negotiated_frame_max)
        .await
        .map_err(|error| {
            ProtocolError::connection(REPLY_INTERNAL_ERROR, error.to_string(), 60, 31)
        })?;
    Ok(HandleOutcome::Continue)
}

fn basic_publish(
    channel: u16,
    reader: &mut Reader<'_>,
    state: &mut ConnectionState,
) -> std::result::Result<HandleOutcome, ProtocolError> {
    let ticket = reader.u16()?;
    let exchange = reader.short_string()?;
    let routing_key = reader.short_string()?;
    let flags = reader.u8()?;
    reader.finish()?;
    if ticket != 0 || flags & !0x03 != 0 {
        return Err(ProtocolError::channel(
            REPLY_SYNTAX_ERROR,
            "basic.publish has nonzero reserved fields",
            CLASS_BASIC,
            40,
        ));
    }
    if flags & 0x02 != 0 {
        return Err(ProtocolError::channel(
            REPLY_NOT_IMPLEMENTED,
            "immediate publishing is not supported by AMQP 0-9-1",
            CLASS_BASIC,
            40,
        ));
    }
    let channel_state = state.channels.get_mut(&channel).ok_or_else(|| {
        ProtocolError::connection(REPLY_CHANNEL_ERROR, "channel is not open", 60, 40)
    })?;
    if !channel_state.flow_active {
        return Err(ProtocolError::channel(
            REPLY_RESOURCE_ERROR,
            "channel flow is inactive",
            CLASS_BASIC,
            40,
        ));
    }
    let sequence = if channel_state.confirms {
        channel_state.allocate_publish_sequence()?
    } else {
        0
    };
    channel_state.pending_publish = Some(PendingPublish {
        exchange,
        routing_key,
        mandatory: flags & 0x01 != 0,
        sequence,
        expected_body: None,
        properties: None,
        body: Vec::new(),
        memory: BrokerMemoryReservation::empty(process_broker_memory()),
    });
    Ok(HandleOutcome::NoResponse)
}

fn validate_publish_body_size(body_size: u64) -> std::result::Result<(), ProtocolError> {
    if body_size > MAX_MESSAGE_BYTES as u64 {
        return Err(ProtocolError::channel(
            REPLY_CONTENT_TOO_LARGE,
            "published message exceeds the message size limit",
            CLASS_BASIC,
            40,
        ));
    }
    Ok(())
}

async fn handle_content_header<W: AsyncWrite + Unpin>(
    channel: u16,
    payload: &[u8],
    state: &mut ConnectionState,
    _coordinator: &Arc<dyn BrokerCoordinator>,
    _writer: &mut W,
) -> std::result::Result<HandleOutcome, ProtocolError> {
    if state.phase != ConnectionPhase::Open || channel == 0 {
        return Err(ProtocolError::connection(
            REPLY_UNEXPECTED_FRAME,
            "content header received outside an open channel",
            CLASS_BASIC,
            40,
        ));
    }
    let mut reader = Reader::new(payload);
    let class_id = reader.u16()?;
    let weight = reader.u16()?;
    let body_size = reader.u64()?;
    if class_id != CLASS_BASIC || weight != 0 {
        return Err(ProtocolError::channel(
            REPLY_UNEXPECTED_FRAME,
            "publish content header has an invalid class or weight",
            CLASS_BASIC,
            40,
        ));
    }
    validate_publish_body_size(body_size)?;
    let properties = BasicProperties::parse(&mut reader)?;
    if properties.user_id.is_some() {
        return Err(ProtocolError::channel(
            REPLY_ACCESS_REFUSED,
            "user-id cannot be asserted without a transport identity mapping",
            CLASS_BASIC,
            40,
        ));
    }
    if properties.expiration.is_some() {
        return Err(ProtocolError::channel(
            REPLY_NOT_IMPLEMENTED,
            "per-message expiration is not supported; declare queue retention instead",
            CLASS_BASIC,
            40,
        ));
    }
    let complete = {
        let channel_state = state.channels.get_mut(&channel).ok_or_else(|| {
            ProtocolError::connection(REPLY_CHANNEL_ERROR, "channel is not open", 60, 40)
        })?;
        let pending = channel_state.pending_publish.as_mut().ok_or_else(|| {
            ProtocolError::channel(
                REPLY_UNEXPECTED_FRAME,
                "content header has no preceding basic.publish",
                CLASS_BASIC,
                40,
            )
        })?;
        if pending.expected_body.is_some() {
            return Err(ProtocolError::channel(
                REPLY_UNEXPECTED_FRAME,
                "duplicate publish content header",
                CLASS_BASIC,
                40,
            ));
        }
        let body_bytes = usize::try_from(body_size).map_err(|_| {
            ProtocolError::channel(
                REPLY_RESOURCE_ERROR,
                "published message size exceeds this platform",
                CLASS_BASIC,
                40,
            )
        })?;
        let body_copies = PUBLISH_STORAGE_BODY_COPIES + usize::from(pending.mandatory);
        let mut encoded_properties = Writer::new();
        properties
            .encode(&mut encoded_properties)
            .map_err(|error| {
                ProtocolError::channel(REPLY_RESOURCE_ERROR, error.to_string(), CLASS_BASIC, 40)
            })?;
        // Metadata is bounded by its actual decoded wire size. Charging every small publish the
        // former 2 MiB worst case made 64-byte messages exhaust the governor before a useful batch
        // could form. Four copies cover the decoded tree, storage maps, canonical encoding, and a
        // mandatory return; the small floor covers map nodes and scalar fields.
        let metadata_reserve = encoded_properties
            .as_slice()
            .len()
            .checked_mul(4)
            .and_then(|bytes| bytes.checked_add(PUBLISH_STORAGE_MIN_METADATA_RESERVE))
            .ok_or_else(|| {
                ProtocolError::channel(
                    REPLY_RESOURCE_ERROR,
                    "published message metadata workspace size overflow",
                    CLASS_BASIC,
                    40,
                )
            })?;
        let reservation_bytes = body_bytes
            .checked_mul(body_copies)
            .and_then(|bytes| bytes.checked_add(metadata_reserve))
            .ok_or_else(|| {
                ProtocolError::channel(
                    REPLY_RESOURCE_ERROR,
                    "published message workspace size overflow",
                    CLASS_BASIC,
                    40,
                )
            })?;
        pending
            .memory
            .try_grow_to(reservation_bytes)
            .map_err(|error| {
                ProtocolError::channel(REPLY_RESOURCE_ERROR, error.to_string(), CLASS_BASIC, 40)
            })?;
        pending.expected_body = Some(body_size);
        pending.properties = Some(properties);
        body_size == 0
    };
    if complete {
        complete_publish(channel, state)
    } else {
        Ok(HandleOutcome::NoResponse)
    }
}

async fn handle_content_body<W: AsyncWrite + Unpin>(
    channel: u16,
    payload: &[u8],
    state: &mut ConnectionState,
    _coordinator: &Arc<dyn BrokerCoordinator>,
    _writer: &mut W,
) -> std::result::Result<HandleOutcome, ProtocolError> {
    if state.phase != ConnectionPhase::Open || channel == 0 {
        return Err(ProtocolError::connection(
            REPLY_UNEXPECTED_FRAME,
            "content body received outside an open channel",
            CLASS_BASIC,
            40,
        ));
    }
    if payload.is_empty() {
        return Err(ProtocolError::channel(
            REPLY_UNEXPECTED_FRAME,
            "empty content body frame cannot advance a publish",
            CLASS_BASIC,
            40,
        ));
    }
    let connection_bytes = state
        .buffered_publish_bytes
        .checked_add(payload.len())
        .ok_or_else(|| {
            ProtocolError::connection(
                REPLY_RESOURCE_ERROR,
                "connection publish buffer size overflow",
                CLASS_BASIC,
                40,
            )
        })?;
    if connection_bytes > MAX_MESSAGE_BYTES {
        return Err(ProtocolError::connection(
            REPLY_RESOURCE_ERROR,
            "connection publish buffer limit exceeded",
            CLASS_BASIC,
            40,
        ));
    }
    let complete = {
        let channel_state = state.channels.get_mut(&channel).ok_or_else(|| {
            ProtocolError::connection(REPLY_CHANNEL_ERROR, "channel is not open", 60, 40)
        })?;
        let pending = channel_state.pending_publish.as_mut().ok_or_else(|| {
            ProtocolError::channel(
                REPLY_UNEXPECTED_FRAME,
                "content body has no preceding basic.publish",
                CLASS_BASIC,
                40,
            )
        })?;
        let expected = pending.expected_body.ok_or_else(|| {
            ProtocolError::channel(
                REPLY_UNEXPECTED_FRAME,
                "content body arrived before the content header",
                CLASS_BASIC,
                40,
            )
        })?;
        let received = pending
            .body
            .len()
            .checked_add(payload.len())
            .ok_or_else(|| {
                ProtocolError::channel(
                    REPLY_CONTENT_TOO_LARGE,
                    "published message size overflow",
                    CLASS_BASIC,
                    40,
                )
            })?;
        if received as u64 > expected || received > MAX_MESSAGE_BYTES {
            return Err(ProtocolError::channel(
                REPLY_UNEXPECTED_FRAME,
                "content body exceeds declared body size",
                CLASS_BASIC,
                40,
            ));
        }
        let expected_capacity = usize::try_from(expected).map_err(|_| {
            ProtocolError::channel(
                REPLY_RESOURCE_ERROR,
                "published message size exceeds this platform",
                CLASS_BASIC,
                40,
            )
        })?;
        let target_capacity = if received > pending.body.capacity() {
            pending
                .body
                .capacity()
                .max(SERVER_FRAME_MAX as usize)
                .saturating_mul(2)
                .max(received)
                .min(expected_capacity)
        } else {
            pending.body.capacity()
        };
        if target_capacity > pending.body.capacity() {
            pending
                .body
                .try_reserve_exact(target_capacity.saturating_sub(pending.body.len()))
                .map_err(|_| {
                    ProtocolError::channel(
                        REPLY_RESOURCE_ERROR,
                        "published message allocation failed",
                        CLASS_BASIC,
                        40,
                    )
                })?;
        }
        pending.body.extend_from_slice(payload);
        received as u64 == expected
    };
    state.buffered_publish_bytes = connection_bytes;
    if complete {
        complete_publish(channel, state)
    } else {
        Ok(HandleOutcome::NoResponse)
    }
}

fn complete_publish(
    channel: u16,
    state: &mut ConnectionState,
) -> std::result::Result<HandleOutcome, ProtocolError> {
    let (mut pending, confirms) = {
        let channel_state = state.channels.get_mut(&channel).ok_or_else(|| {
            ProtocolError::connection(REPLY_CHANNEL_ERROR, "channel is not open", 60, 40)
        })?;
        let pending = channel_state.pending_publish.take().ok_or_else(|| {
            ProtocolError::channel(
                REPLY_UNEXPECTED_FRAME,
                "publish state disappeared",
                CLASS_BASIC,
                40,
            )
        })?;
        (pending, channel_state.confirms)
    };
    state.buffered_publish_bytes = state
        .buffered_publish_bytes
        .saturating_sub(pending.body.len());
    let properties = pending.properties.ok_or_else(|| {
        ProtocolError::channel(
            REPLY_UNEXPECTED_FRAME,
            "publish content properties are missing",
            CLASS_BASIC,
            40,
        )
    })?;
    let returned_properties = pending.mandatory.then(|| properties.clone());
    let (stored_properties, stored_headers) = properties.into_storage().map_err(|error| {
        ProtocolError::channel(REPLY_INTERNAL_ERROR, error.to_string(), CLASS_BASIC, 40)
    })?;
    let returned_body = pending.mandatory.then(|| pending.body.clone());
    let record = AmqpBatchRecord {
        exchange: pending.exchange.clone(),
        routing_key: pending.routing_key.clone(),
        mandatory: pending.mandatory,
        properties: stored_properties,
        headers: stored_headers,
        payload: std::mem::take(&mut pending.body),
    };
    state.ready_publishes.push(ReadyPublish {
        channel,
        sequence: pending.sequence,
        confirms,
        returned_properties,
        returned_body,
        record,
        memory: pending.memory,
    });
    Ok(HandleOutcome::NoResponse)
}

async fn flush_ready_publishes<W: AsyncWrite + Unpin>(
    state: &mut ConnectionState,
    coordinator: &Arc<dyn BrokerCoordinator>,
    writer: &mut W,
) -> std::result::Result<HandleOutcome, ProtocolError> {
    if state.ready_publishes.is_empty() {
        return Ok(HandleOutcome::NoResponse);
    }
    let ready = std::mem::take(&mut state.ready_publishes);
    let mut records = Vec::with_capacity(ready.len());
    let mut completions = Vec::with_capacity(ready.len());
    let mut reservations = Vec::with_capacity(ready.len());
    for ready in ready {
        let returned_exchange = ready
            .record
            .mandatory
            .then(|| ready.record.exchange.clone());
        let returned_routing_key = ready
            .record
            .mandatory
            .then(|| ready.record.routing_key.clone());
        records.push(ready.record);
        completions.push(PublishCompletion {
            channel: ready.channel,
            sequence: ready.sequence,
            confirms: ready.confirms,
            returned_exchange,
            returned_routing_key,
            returned_properties: ready.returned_properties,
            returned_body: ready.returned_body,
        });
        reservations.push(ready.memory);
    }
    let uniform = records.first().is_some_and(|first| {
        records.iter().skip(1).all(|record| {
            record.exchange == first.exchange
                && record.routing_key == first.routing_key
                && record.mandatory == first.mandatory
                && record.properties == first.properties
                && record.headers == first.headers
        })
    });
    let command = if uniform {
        let mut records = records.into_iter();
        let first = records.next().ok_or_else(|| {
            ProtocolError::connection(
                REPLY_INTERNAL_ERROR,
                "empty AMQP publish batch",
                CLASS_BASIC,
                40,
            )
        })?;
        let mut payloads = Vec::with_capacity(completions.len());
        payloads.push(first.payload);
        payloads.extend(records.map(|record| record.payload));
        BrokerCommand::PublishAmqpUniformBatch {
            project: state.project,
            resolved_time_ms: 0,
            exchange: first.exchange,
            routing_key: first.routing_key,
            mandatory: first.mandatory,
            properties: first.properties,
            headers: first.headers,
            payloads,
        }
    } else {
        BrokerCommand::PublishAmqpBatch {
            project: state.project,
            resolved_time_ms: 0,
            records,
        }
    };
    let coordinator = Arc::clone(coordinator);
    let submitted = tokio::task::spawn_blocking(move || {
        let _reservations = reservations;
        coordinator
            .submit(command, CommitAcknowledgement::Published)
            .map(|commit| commit.reply)
    })
    .await
    .map_err(|error| {
        ProtocolError::connection(
            REPLY_INTERNAL_ERROR,
            format!("broker coordinator task failed: {error}"),
            CLASS_BASIC,
            40,
        )
    })?;
    let (offsets, uniform_routed) = match submitted {
        Ok(BrokerReply::AmqpBatchPublished { offsets }) if offsets.len() == completions.len() => {
            (Some(offsets), None)
        }
        Ok(BrokerReply::AmqpUniformBatchPublished {
            record_count,
            routed,
        }) if uniform && usize::try_from(record_count).ok() == Some(completions.len()) => {
            (None, Some(routed))
        }
        Ok(_) => {
            return Err(ProtocolError::channel(
                REPLY_INTERNAL_ERROR,
                "broker returned an invalid AMQP batch reply",
                CLASS_BASIC,
                40,
            ));
        }
        Err(error)
            if completions.iter().any(|completion| completion.confirms)
                && publish_failure_is_known(&error) =>
        {
            let mut last_by_channel = BTreeMap::<u16, u64>::new();
            for completion in &completions {
                if completion.confirms {
                    last_by_channel.insert(completion.channel, completion.sequence);
                }
            }
            for (channel, sequence) in last_by_channel {
                let mut nack = Writer::method(CLASS_BASIC, 120);
                nack.u64(sequence);
                nack.u8(1);
                write_method(writer, channel, &nack, state.negotiated_frame_max)
                    .await
                    .map_err(|write_error| {
                        ProtocolError::connection(
                            REPLY_INTERNAL_ERROR,
                            write_error.to_string(),
                            CLASS_BASIC,
                            120,
                        )
                    })?;
            }
            return Ok(HandleOutcome::Continue);
        }
        Err(error) => return Err(ProtocolError::from_engine(error, CLASS_BASIC, 40)),
    };

    let mut last_confirm_by_channel = BTreeMap::<u16, u64>::new();
    let mut wrote_response = false;
    for (index, completion) in completions.into_iter().enumerate() {
        let routed = uniform_routed.unwrap_or_else(|| {
            offsets
                .as_ref()
                .and_then(|offsets| offsets.get(index))
                .is_some_and(|record_offsets| !record_offsets.is_empty())
        });
        if completion.returned_exchange.is_some() && !routed {
            let returned_exchange = completion.returned_exchange.as_deref().unwrap_or_default();
            let returned_routing_key = completion
                .returned_routing_key
                .as_deref()
                .unwrap_or_default();
            let mut returned = Writer::method(CLASS_BASIC, 50);
            returned.u16(REPLY_NO_ROUTE);
            returned.short_string("NO_ROUTE").map_err(|error| {
                ProtocolError::connection(REPLY_INTERNAL_ERROR, error.to_string(), CLASS_BASIC, 50)
            })?;
            returned.short_string(returned_exchange).map_err(|error| {
                ProtocolError::connection(REPLY_INTERNAL_ERROR, error.to_string(), CLASS_BASIC, 50)
            })?;
            returned
                .short_string(returned_routing_key)
                .map_err(|error| {
                    ProtocolError::connection(
                        REPLY_INTERNAL_ERROR,
                        error.to_string(),
                        CLASS_BASIC,
                        50,
                    )
                })?;
            write_method(
                writer,
                completion.channel,
                &returned,
                state.negotiated_frame_max,
            )
            .await
            .map_err(|error| {
                ProtocolError::connection(REPLY_INTERNAL_ERROR, error.to_string(), 60, 50)
            })?;
            write_content(
                writer,
                completion.channel,
                completion.returned_properties.as_ref().ok_or_else(|| {
                    ProtocolError::connection(
                        REPLY_INTERNAL_ERROR,
                        "mandatory publish lost its return properties",
                        CLASS_BASIC,
                        50,
                    )
                })?,
                completion.returned_body.as_deref().unwrap_or_default(),
                state.negotiated_frame_max,
            )
            .await?;
            wrote_response = true;
        }
        if completion.confirms {
            last_confirm_by_channel.insert(completion.channel, completion.sequence);
        }
    }
    for (channel, sequence) in last_confirm_by_channel {
        let mut ack = Writer::method(CLASS_BASIC, 80);
        ack.u64(sequence);
        ack.u8(1);
        write_method(writer, channel, &ack, state.negotiated_frame_max)
            .await
            .map_err(|error| {
                ProtocolError::connection(REPLY_INTERNAL_ERROR, error.to_string(), 60, 80)
            })?;
        wrote_response = true;
    }
    Ok(if wrote_response {
        HandleOutcome::Continue
    } else {
        HandleOutcome::NoResponse
    })
}

const fn publish_failure_is_known(error: &Error) -> bool {
    !matches!(error.code, ErrorCode::DeadlineExceeded)
}

async fn write_content<W: AsyncWrite + Unpin>(
    writer: &mut W,
    channel: u16,
    properties: &BasicProperties,
    body: &[u8],
    frame_max: u32,
) -> std::result::Result<(), ProtocolError> {
    write_content_inner(writer, channel, properties, body, frame_max, false).await
}

async fn write_content_buffered<W: AsyncWrite + Unpin>(
    writer: &mut W,
    channel: u16,
    properties: &BasicProperties,
    body: &[u8],
    frame_max: u32,
) -> std::result::Result<(), ProtocolError> {
    write_content_inner(writer, channel, properties, body, frame_max, true).await
}

async fn write_content_inner<W: AsyncWrite + Unpin>(
    writer: &mut W,
    channel: u16,
    properties: &BasicProperties,
    body: &[u8],
    frame_max: u32,
    buffered: bool,
) -> std::result::Result<(), ProtocolError> {
    let mut header = Writer::new();
    header.u16(CLASS_BASIC);
    header.u16(0);
    header.u64(body.len() as u64);
    properties.encode(&mut header).map_err(|error| {
        ProtocolError::connection(REPLY_INTERNAL_ERROR, error.to_string(), CLASS_BASIC, 0)
    })?;
    let header_result = if buffered {
        write_frame_buffered(writer, FRAME_HEADER, channel, header.as_slice(), frame_max).await
    } else {
        write_frame(writer, FRAME_HEADER, channel, header.as_slice(), frame_max).await
    };
    header_result.map_err(|error| {
        ProtocolError::connection(REPLY_INTERNAL_ERROR, error.to_string(), CLASS_BASIC, 0)
    })?;
    let maximum_body = (frame_max as usize).saturating_sub(FRAME_OVERHEAD);
    if maximum_body == 0 && !body.is_empty() {
        return Err(ProtocolError::connection(
            REPLY_FRAME_ERROR,
            "negotiated frame-max cannot carry a body frame",
            CLASS_BASIC,
            0,
        ));
    }
    for chunk in body.chunks(maximum_body.max(1)) {
        let body_result = if buffered {
            write_frame_buffered(writer, FRAME_BODY, channel, chunk, frame_max).await
        } else {
            write_frame(writer, FRAME_BODY, channel, chunk, frame_max).await
        };
        body_result.map_err(|error| {
            ProtocolError::connection(REPLY_INTERNAL_ERROR, error.to_string(), CLASS_BASIC, 0)
        })?;
    }
    Ok(())
}

#[allow(clippy::too_many_lines)]
async fn basic_get<W: AsyncWrite + Unpin>(
    channel: u16,
    reader: &mut Reader<'_>,
    state: &mut ConnectionState,
    coordinator: &Arc<dyn BrokerCoordinator>,
    writer: &mut W,
) -> std::result::Result<HandleOutcome, ProtocolError> {
    let ticket = reader.u16()?;
    let queue = resolve_queue_name(channel, reader.short_string()?, state, CLASS_BASIC, 70)?;
    let flags = reader.u8()?;
    reader.finish()?;
    if ticket != 0 || flags & !0x01 != 0 {
        return Err(ProtocolError::channel(
            REPLY_SYNTAX_ERROR,
            "basic.get has nonzero reserved fields",
            CLASS_BASIC,
            70,
        ));
    }
    let automatic_ack = flags & 0x01 != 0;
    let info = coordinator_queue_info(Arc::clone(coordinator), state.project, queue.clone())
        .await
        .map_err(|error| ProtocolError::from_engine(error, CLASS_BASIC, 70))?
        .ok_or_else(|| {
            ProtocolError::channel(REPLY_NOT_FOUND, "queue does not exist", CLASS_BASIC, 70)
        })?;
    if info.kind == QueueKind::Stream {
        return Err(ProtocolError::channel(
            REPLY_NOT_IMPLEMENTED,
            "basic.get is not stateful enough for a stream queue; use basic.consume",
            CLASS_BASIC,
            70,
        ));
    }
    if info.available_count == 0 {
        let mut empty = Writer::method(CLASS_BASIC, 72);
        empty.short_string("").map_err(|error| {
            ProtocolError::connection(REPLY_INTERNAL_ERROR, error.to_string(), CLASS_BASIC, 72)
        })?;
        write_method(writer, channel, &empty, state.negotiated_frame_max)
            .await
            .map_err(|error| {
                ProtocolError::connection(REPLY_INTERNAL_ERROR, error.to_string(), 60, 72)
            })?;
        return Ok(HandleOutcome::Continue);
    }
    let consumer_id = state.allocate_consumer()?;
    if !automatic_ack {
        state.delivery_lease_touched = true;
    }
    let _delivery_memory = process_broker_memory()
        .try_reserve(EGRESS_DELIVERY_RESERVE)
        .map_err(|error| {
            ProtocolError::channel(REPLY_RESOURCE_ERROR, error.to_string(), CLASS_BASIC, 70)
        })?;
    let reply = coordinator_submit_reserved(
        Arc::clone(coordinator),
        BrokerCommand::DeliverQueue {
            project: state.project,
            queue: queue.clone(),
            offset: StreamOffset::First,
            maximum: 1,
            owner: state.delivery_owner,
            consumer: consumer_id,
            automatic_ack,
            resolved_time_ms: 0,
        },
        CommitAcknowledgement::Published,
        &_delivery_memory,
    )
    .await
    .map_err(|error| ProtocolError::from_engine(error, CLASS_BASIC, 70))?;
    let BrokerReply::Deliveries(mut deliveries) = reply else {
        return Err(ProtocolError::channel(
            REPLY_INTERNAL_ERROR,
            "broker returned an invalid delivery reply",
            CLASS_BASIC,
            70,
        ));
    };
    let Some(delivery) = deliveries.pop() else {
        let mut empty = Writer::method(CLASS_BASIC, 72);
        empty.short_string("").map_err(|error| {
            ProtocolError::connection(REPLY_INTERNAL_ERROR, error.to_string(), CLASS_BASIC, 72)
        })?;
        write_method(writer, channel, &empty, state.negotiated_frame_max)
            .await
            .map_err(|error| {
                ProtocolError::connection(REPLY_INTERNAL_ERROR, error.to_string(), 60, 72)
            })?;
        return Ok(HandleOutcome::Continue);
    };
    let app_tag = {
        let channel_state = state.channels.get_mut(&channel).ok_or_else(|| {
            ProtocolError::connection(REPLY_CHANNEL_ERROR, "channel is not open", 60, 70)
        })?;
        let tag = channel_state.allocate_delivery_tag()?;
        if !automatic_ack {
            channel_state.outstanding.insert(
                tag,
                OutstandingDelivery {
                    consumer_id,
                    consumer_tag: None,
                    queue: queue.clone(),
                    queue_kind: QueueKind::Classic,
                    engine_delivery_tag: delivery.delivery_tag,
                    stream_offset: delivery.offset,
                },
            );
        }
        tag
    };
    let remaining = coordinator_queue_info(Arc::clone(coordinator), state.project, queue.clone())
        .await
        .ok()
        .flatten()
        .map_or(0, |queue_info| {
            u32::try_from(queue_info.available_count).unwrap_or(u32::MAX)
        });
    let mut get_ok = Writer::method(CLASS_BASIC, 71);
    get_ok.u64(app_tag);
    get_ok.u8(u8::from(delivery.redelivered));
    get_ok.short_string(&delivery.exchange).map_err(|error| {
        ProtocolError::connection(REPLY_INTERNAL_ERROR, error.to_string(), CLASS_BASIC, 71)
    })?;
    get_ok
        .short_string(&delivery.routing_key)
        .map_err(|error| {
            ProtocolError::connection(REPLY_INTERNAL_ERROR, error.to_string(), CLASS_BASIC, 71)
        })?;
    get_ok.u32(remaining);
    write_method(writer, channel, &get_ok, state.negotiated_frame_max)
        .await
        .map_err(|error| {
            ProtocolError::connection(REPLY_INTERNAL_ERROR, error.to_string(), 60, 71)
        })?;
    write_delivery_content(
        writer,
        channel,
        &delivery,
        QueueKind::Classic,
        state.negotiated_frame_max,
    )
    .await?;
    Ok(HandleOutcome::Continue)
}

async fn basic_ack(
    channel: u16,
    reader: &mut Reader<'_>,
    state: &mut ConnectionState,
    coordinator: &Arc<dyn BrokerCoordinator>,
) -> std::result::Result<HandleOutcome, ProtocolError> {
    let delivery_tag = reader.u64()?;
    let flags = reader.u8()?;
    reader.finish()?;
    if flags & !0x01 != 0 {
        return Err(ProtocolError::channel(
            REPLY_SYNTAX_ERROR,
            "basic.ack reserved bits are set",
            CLASS_BASIC,
            80,
        ));
    }
    let tags = select_delivery_tags(channel, delivery_tag, flags & 0x01 != 0, state, 80)?;
    settle_tags(channel, &tags, true, false, state, coordinator).await?;
    Ok(HandleOutcome::NoResponse)
}

async fn basic_reject(
    channel: u16,
    reader: &mut Reader<'_>,
    state: &mut ConnectionState,
    coordinator: &Arc<dyn BrokerCoordinator>,
) -> std::result::Result<HandleOutcome, ProtocolError> {
    let delivery_tag = reader.u64()?;
    let flags = reader.u8()?;
    reader.finish()?;
    if flags & !0x01 != 0 {
        return Err(ProtocolError::channel(
            REPLY_SYNTAX_ERROR,
            "basic.reject reserved bits are set",
            CLASS_BASIC,
            90,
        ));
    }
    let tags = select_delivery_tags(channel, delivery_tag, false, state, 90)?;
    settle_tags(channel, &tags, false, flags & 0x01 != 0, state, coordinator).await?;
    Ok(HandleOutcome::NoResponse)
}

async fn basic_nack(
    channel: u16,
    reader: &mut Reader<'_>,
    state: &mut ConnectionState,
    coordinator: &Arc<dyn BrokerCoordinator>,
) -> std::result::Result<HandleOutcome, ProtocolError> {
    let delivery_tag = reader.u64()?;
    let flags = reader.u8()?;
    reader.finish()?;
    if flags & !0x03 != 0 {
        return Err(ProtocolError::channel(
            REPLY_SYNTAX_ERROR,
            "basic.nack reserved bits are set",
            CLASS_BASIC,
            120,
        ));
    }
    let tags = select_delivery_tags(channel, delivery_tag, flags & 0x01 != 0, state, 120)?;
    settle_tags(channel, &tags, false, flags & 0x02 != 0, state, coordinator).await?;
    Ok(HandleOutcome::NoResponse)
}

async fn basic_recover_async(
    channel: u16,
    reader: &mut Reader<'_>,
    state: &mut ConnectionState,
    coordinator: &Arc<dyn BrokerCoordinator>,
) -> std::result::Result<HandleOutcome, ProtocolError> {
    let flags = reader.u8()?;
    reader.finish()?;
    if flags != 0x01 {
        return Err(ProtocolError::channel(
            REPLY_NOT_IMPLEMENTED,
            "basic.recover-async requires requeue=true",
            CLASS_BASIC,
            100,
        ));
    }
    let tags = all_delivery_tags(channel, state);
    settle_tags(channel, &tags, false, true, state, coordinator).await?;
    Ok(HandleOutcome::NoResponse)
}

async fn basic_recover<W: AsyncWrite + Unpin>(
    channel: u16,
    reader: &mut Reader<'_>,
    state: &mut ConnectionState,
    coordinator: &Arc<dyn BrokerCoordinator>,
    writer: &mut W,
) -> std::result::Result<HandleOutcome, ProtocolError> {
    let flags = reader.u8()?;
    reader.finish()?;
    if flags != 0x01 {
        return Err(ProtocolError::channel(
            REPLY_NOT_IMPLEMENTED,
            "basic.recover requires requeue=true",
            CLASS_BASIC,
            110,
        ));
    }
    let tags = all_delivery_tags(channel, state);
    settle_tags(channel, &tags, false, true, state, coordinator).await?;
    let recover_ok = Writer::method(CLASS_BASIC, 111);
    write_method(writer, channel, &recover_ok, state.negotiated_frame_max)
        .await
        .map_err(|error| {
            ProtocolError::connection(REPLY_INTERNAL_ERROR, error.to_string(), 60, 111)
        })?;
    Ok(HandleOutcome::Continue)
}

async fn confirm_select<W: AsyncWrite + Unpin>(
    channel: u16,
    reader: &mut Reader<'_>,
    state: &mut ConnectionState,
    writer: &mut W,
) -> std::result::Result<HandleOutcome, ProtocolError> {
    let flags = reader.u8()?;
    reader.finish()?;
    if flags & !0x01 != 0 {
        return Err(ProtocolError::channel(
            REPLY_SYNTAX_ERROR,
            "confirm.select reserved bits are set",
            CLASS_CONFIRM,
            10,
        ));
    }
    let channel_state = state.channels.get_mut(&channel).ok_or_else(|| {
        ProtocolError::connection(REPLY_CHANNEL_ERROR, "channel is not open", 85, 10)
    })?;
    if !channel_state.confirms {
        channel_state.confirms = true;
        channel_state.next_publish_sequence = 0;
    }
    if flags & 0x01 != 0 {
        return Ok(HandleOutcome::NoResponse);
    }
    let select_ok = Writer::method(CLASS_CONFIRM, 11);
    write_method(writer, channel, &select_ok, state.negotiated_frame_max)
        .await
        .map_err(|error| {
            ProtocolError::connection(REPLY_INTERNAL_ERROR, error.to_string(), 85, 11)
        })?;
    Ok(HandleOutcome::Continue)
}

fn select_delivery_tags(
    channel: u16,
    delivery_tag: u64,
    multiple: bool,
    state: &ConnectionState,
    method_id: u16,
) -> std::result::Result<Vec<u64>, ProtocolError> {
    let channel_state = state.channels.get(&channel).ok_or_else(|| {
        ProtocolError::connection(REPLY_CHANNEL_ERROR, "channel is not open", 60, method_id)
    })?;
    let tags = if multiple {
        if delivery_tag == 0 {
            channel_state.outstanding.keys().copied().collect()
        } else {
            channel_state
                .outstanding
                .range(..=delivery_tag)
                .map(|(tag, _)| *tag)
                .collect()
        }
    } else if channel_state.outstanding.contains_key(&delivery_tag) {
        vec![delivery_tag]
    } else {
        Vec::new()
    };
    if tags.is_empty() && !(multiple && delivery_tag == 0) {
        return Err(ProtocolError::channel(
            REPLY_PRECONDITION_FAILED,
            "delivery tag is unknown",
            CLASS_BASIC,
            method_id,
        ));
    }
    Ok(tags)
}

fn all_delivery_tags(channel: u16, state: &ConnectionState) -> Vec<u64> {
    state
        .channels
        .get(&channel)
        .map(|channel_state| channel_state.outstanding.keys().copied().collect())
        .unwrap_or_default()
}

/// Decides which classic-queue deliveries can be settled by one ordered prefix command.
///
/// `basic.ack`/`basic.nack` with `multiple` set covers every unacknowledged delivery up to a tag,
/// and the state machine already implements exactly that as a prefix scan. Issuing one command per
/// tag instead costs a full ordered write each, so acknowledging a thousand messages — the
/// ordinary high-throughput idiom — meant a thousand serialized commits while the connection
/// could not even answer heartbeats.
///
/// The prefix form is only equivalent when the tags being settled are *all* of that consumer's
/// outstanding deliveries at or below the highest one: the state machine settles by offset, so a
/// gap would silently settle a delivery the client did not name. This returns the highest engine
/// tag per `(queue, consumer)` group that satisfies that condition, and everything else falls back
/// to one command per delivery.
fn prefix_settleable_groups(
    channel_state: &ChannelState,
    tags: &[u64],
) -> BTreeMap<(String, u64), u64> {
    let mut groups: BTreeMap<(String, u64), (u64, usize)> = BTreeMap::new();
    for tag in tags {
        let Some(delivery) = channel_state.outstanding.get(tag) else {
            continue;
        };
        if delivery.queue_kind != QueueKind::Classic {
            continue;
        }
        let key = (delivery.queue.clone(), delivery.consumer_id);
        let entry = groups.entry(key).or_insert((0, 0));
        entry.0 = entry.0.max(delivery.engine_delivery_tag);
        entry.1 = entry.1.saturating_add(1);
    }
    groups
        .into_iter()
        .filter(|((queue, consumer), (highest, selected))| {
            // Every outstanding delivery this prefix would reach must be one we were asked to
            // settle; otherwise the prefix would swallow a delivery the client still owns.
            let covered = channel_state
                .outstanding
                .values()
                .filter(|delivery| {
                    delivery.queue_kind == QueueKind::Classic
                        && delivery.queue == *queue
                        && delivery.consumer_id == *consumer
                        && delivery.engine_delivery_tag <= *highest
                })
                .count();
            covered == *selected
        })
        .map(|(key, (highest, _))| (key, highest))
        .collect()
}

async fn settle_tags(
    channel: u16,
    tags: &[u64],
    acknowledge: bool,
    requeue: bool,
    state: &mut ConnectionState,
    coordinator: &Arc<dyn BrokerCoordinator>,
) -> std::result::Result<(), ProtocolError> {
    let prefixes = state
        .channels
        .get(&channel)
        .map(|channel_state| prefix_settleable_groups(channel_state, tags))
        .unwrap_or_default();
    let mut settled_prefixes: BTreeSet<(String, u64)> = BTreeSet::new();
    for app_tag in tags {
        let delivery = state
            .channels
            .get(&channel)
            .and_then(|channel_state| channel_state.outstanding.get(app_tag))
            .ok_or_else(|| {
                ProtocolError::channel(
                    REPLY_PRECONDITION_FAILED,
                    "delivery tag disappeared during settlement",
                    CLASS_BASIC,
                    if acknowledge { 80 } else { 120 },
                )
            })?;
        let queue = delivery.queue.clone();
        let consumer_id = delivery.consumer_id;
        let engine_delivery_tag = delivery.engine_delivery_tag;
        let queue_kind = delivery.queue_kind;
        let stream_offset = delivery.stream_offset;
        let consumer_tag = delivery.consumer_tag.clone();
        // A group that can be settled as a prefix is committed once, on its highest tag; the
        // remaining tags in that group are already covered and must not be submitted again.
        let group = (queue.clone(), consumer_id);
        let prefix = prefixes.get(&group).copied();
        let submit = queue_kind == QueueKind::Classic
            && prefix.is_none_or(|highest| {
                engine_delivery_tag == highest && settled_prefixes.insert(group.clone())
            });
        if submit {
            let multiple = prefix.is_some();
            let command = if acknowledge {
                BrokerCommand::Ack {
                    project: state.project,
                    queue,
                    owner: state.delivery_owner,
                    consumer: consumer_id,
                    delivery_tag: engine_delivery_tag,
                    multiple,
                }
            } else {
                BrokerCommand::Nack {
                    project: state.project,
                    queue,
                    owner: state.delivery_owner,
                    consumer: consumer_id,
                    delivery_tag: engine_delivery_tag,
                    multiple,
                    requeue,
                }
            };
            let reply = coordinator_submit(
                Arc::clone(coordinator),
                command,
                CommitAcknowledgement::Published,
            )
            .await
            .map_err(|error| {
                ProtocolError::from_engine(error, CLASS_BASIC, if acknowledge { 80 } else { 120 })
            })?;
            if !matches!(reply, BrokerReply::Acknowledged) {
                return Err(ProtocolError::channel(
                    REPLY_INTERNAL_ERROR,
                    "broker returned an invalid settlement reply",
                    CLASS_BASIC,
                    if acknowledge { 80 } else { 120 },
                ));
            }
        }
        let channel_state = state.channels.get_mut(&channel).ok_or_else(|| {
            ProtocolError::connection(REPLY_CHANNEL_ERROR, "channel is not open", 60, 80)
        })?;
        channel_state.outstanding.remove(app_tag);
        if let Some(consumer) = consumer_tag.and_then(|tag| channel_state.consumers.get_mut(&tag)) {
            consumer.outstanding = consumer.outstanding.saturating_sub(1);
            if queue_kind == QueueKind::Stream && !acknowledge && requeue {
                let reset = match consumer.next_offset {
                    StreamOffset::Absolute(current) => current.min(stream_offset),
                    _ => stream_offset,
                };
                consumer.next_offset = StreamOffset::Absolute(reset);
            }
        }
    }
    Ok(())
}

struct PumpRequest {
    channel: u16,
    consumer_tag: String,
    consumer_id: u64,
    queue: String,
    queue_kind: QueueKind,
    offset: StreamOffset,
    automatic_ack: bool,
    /// Deliveries this consumer may receive in one round, from its prefetch credit.
    maximum: u32,
}

#[allow(clippy::too_many_lines)]
async fn pump_consumers<W: AsyncWrite + Unpin>(
    state: &mut ConnectionState,
    coordinator: &Arc<dyn BrokerCoordinator>,
    writer: &mut W,
) -> std::result::Result<bool, ProtocolError> {
    let mut requests = Vec::new();
    for (channel, channel_state) in &mut state.channels {
        if !channel_state.flow_active {
            continue;
        }
        let global_outstanding = u32::try_from(channel_state.outstanding.len()).unwrap_or(u32::MAX);
        let mut global_available =
            if channel_state.prefetch_global && channel_state.prefetch_count != 0 {
                u32::from(channel_state.prefetch_count).saturating_sub(global_outstanding)
            } else {
                u32::MAX
            };
        let mut consumer_tags = channel_state.consumers.keys().cloned().collect::<Vec<_>>();
        if !consumer_tags.is_empty() {
            let rotation = channel_state.pump_cursor % consumer_tags.len();
            consumer_tags.rotate_left(rotation);
            channel_state.pump_cursor = channel_state.pump_cursor.wrapping_add(1);
        }
        let now = Instant::now();
        for consumer_tag in consumer_tags {
            let Some(consumer) = channel_state.consumers.get(&consumer_tag) else {
                continue;
            };
            if now < consumer.poll_after {
                continue;
            }
            let maximum = if consumer.automatic_ack || channel_state.prefetch_count == 0 {
                MAX_DELIVERIES_PER_PUMP
            } else if channel_state.prefetch_global {
                global_available.min(MAX_DELIVERIES_PER_PUMP)
            } else {
                u32::from(channel_state.prefetch_count)
                    .saturating_sub(consumer.outstanding)
                    .min(MAX_DELIVERIES_PER_PUMP)
            };
            if maximum == 0 {
                continue;
            }
            if channel_state.prefetch_global && !consumer.automatic_ack {
                global_available = global_available.saturating_sub(maximum);
            }
            requests.push(PumpRequest {
                channel: *channel,
                consumer_tag,
                consumer_id: consumer.id,
                queue: consumer.queue.clone(),
                queue_kind: consumer.queue_kind,
                offset: consumer.next_offset,
                automatic_ack: consumer.automatic_ack,
                maximum,
            });
        }
    }

    let mut sent = false;
    for request in requests {
        let Some(info) = coordinator_queue_info(
            Arc::clone(coordinator),
            state.project,
            request.queue.clone(),
        )
        .await
        .map_err(|error| {
            ProtocolError::from_engine(error, CLASS_BASIC, 60).on_channel(request.channel)
        })?
        else {
            return Err(ProtocolError::channel(
                REPLY_NOT_FOUND,
                "consumer queue was deleted",
                CLASS_BASIC,
                60,
            )
            .on_channel(request.channel));
        };
        if info.available_count == 0 {
            record_consumer_poll(state, &request, false);
            continue;
        }
        // Every delivery costs one ordered write, so the pump asks for as many as the
        // consumer's prefetch credit allows instead of one at a time. The reservation has to cover
        // the whole batch, so it shrinks until it fits rather than refusing outright: under memory
        // pressure this degrades to exactly the previous single-delivery behaviour.
        let mut admitted = admissible_delivery_batch(request.maximum);
        let reservation = loop {
            let bytes = delivery_reserve_bytes(admitted);
            match process_broker_memory().try_reserve(bytes) {
                Ok(reservation) => break Some(reservation),
                Err(_) if admitted > 1 => admitted /= 2,
                Err(_) => break None,
            }
        };
        let Some(_delivery_memory) = reservation else {
            // Consumer delivery is pull-driven by the pump. Leaving the message untouched and
            // retrying on a later tick is AMQP backpressure and cannot lose or duplicate a
            // delivery lease.
            record_consumer_poll(state, &request, false);
            continue;
        };
        let deliveries = if request.queue_kind == QueueKind::Classic {
            if !request.automatic_ack {
                state.delivery_lease_touched = true;
            }
            let reply = coordinator_submit_reserved(
                Arc::clone(coordinator),
                BrokerCommand::DeliverQueue {
                    project: state.project,
                    queue: request.queue.clone(),
                    offset: request.offset,
                    maximum: admitted,
                    owner: state.delivery_owner,
                    consumer: request.consumer_id,
                    automatic_ack: request.automatic_ack,
                    resolved_time_ms: 0,
                },
                CommitAcknowledgement::Published,
                &_delivery_memory,
            )
            .await
            .map_err(|error| {
                ProtocolError::from_engine(error, CLASS_BASIC, 60).on_channel(request.channel)
            })?;
            match reply {
                BrokerReply::Deliveries(deliveries) => deliveries,
                _ => {
                    return Err(ProtocolError::channel(
                        REPLY_INTERNAL_ERROR,
                        "broker returned an invalid delivery reply",
                        CLASS_BASIC,
                        60,
                    )
                    .on_channel(request.channel));
                }
            }
        } else {
            coordinator_fetch_stream_queue(
                Arc::clone(coordinator),
                state.project,
                request.queue.clone(),
                request.offset,
                admitted as usize,
                request.consumer_id,
                request.automatic_ack,
                &_delivery_memory,
            )
            .await
            .map_err(|error| {
                ProtocolError::from_engine(error, CLASS_BASIC, 60).on_channel(request.channel)
            })?
        };
        record_consumer_poll(state, &request, !deliveries.is_empty());
        for delivery in deliveries {
            let app_tag = {
                let channel_state = state.channels.get_mut(&request.channel).ok_or_else(|| {
                    ProtocolError::connection(
                        REPLY_CHANNEL_ERROR,
                        "consumer channel disappeared",
                        CLASS_BASIC,
                        60,
                    )
                })?;
                let consumer = channel_state
                    .consumers
                    .get_mut(&request.consumer_tag)
                    .filter(|consumer| consumer.id == request.consumer_id)
                    .ok_or_else(|| {
                        ProtocolError::channel(
                            REPLY_NOT_FOUND,
                            "consumer disappeared during delivery",
                            CLASS_BASIC,
                            60,
                        )
                    })?;
                if request.queue_kind == QueueKind::Stream {
                    consumer.next_offset = StreamOffset::Absolute(
                        delivery.offset.checked_add(1).ok_or_else(|| {
                            ProtocolError::channel(
                                REPLY_RESOURCE_ERROR,
                                "stream offset exhausted",
                                CLASS_BASIC,
                                60,
                            )
                        })?,
                    );
                }
                let tag = channel_state.allocate_delivery_tag()?;
                if !request.automatic_ack {
                    channel_state.outstanding.insert(
                        tag,
                        OutstandingDelivery {
                            consumer_id: request.consumer_id,
                            consumer_tag: Some(request.consumer_tag.clone()),
                            queue: request.queue.clone(),
                            queue_kind: request.queue_kind,
                            engine_delivery_tag: delivery.delivery_tag,
                            stream_offset: delivery.offset,
                        },
                    );
                    if let Some(consumer) = channel_state.consumers.get_mut(&request.consumer_tag) {
                        consumer.outstanding = consumer.outstanding.saturating_add(1);
                    }
                }
                tag
            };
            let mut deliver = Writer::method(CLASS_BASIC, 60);
            deliver
                .short_string(&request.consumer_tag)
                .map_err(|error| {
                    ProtocolError::connection(REPLY_INTERNAL_ERROR, error.to_string(), 60, 60)
                })?;
            deliver.u64(app_tag);
            deliver.u8(u8::from(delivery.redelivered));
            deliver.short_string(&delivery.exchange).map_err(|error| {
                ProtocolError::connection(REPLY_INTERNAL_ERROR, error.to_string(), 60, 60)
            })?;
            deliver
                .short_string(&delivery.routing_key)
                .map_err(|error| {
                    ProtocolError::connection(REPLY_INTERNAL_ERROR, error.to_string(), 60, 60)
                })?;
            write_method_buffered(
                writer,
                request.channel,
                &deliver,
                state.negotiated_frame_max,
            )
            .await
            .map_err(|error| {
                ProtocolError::connection(REPLY_INTERNAL_ERROR, error.to_string(), 60, 60)
            })?;
            write_delivery_content(
                writer,
                request.channel,
                &delivery,
                request.queue_kind,
                state.negotiated_frame_max,
            )
            .await?;
            sent = true;
        }
    }
    Ok(sent)
}

fn record_consumer_poll(state: &mut ConnectionState, request: &PumpRequest, delivered: bool) {
    let Some(consumer) = state
        .channels
        .get_mut(&request.channel)
        .and_then(|channel_state| channel_state.consumers.get_mut(&request.consumer_tag))
        .filter(|consumer| consumer.id == request.consumer_id)
    else {
        return;
    };
    if delivered {
        consumer.idle_poll_interval = PUMP_INTERVAL;
    } else {
        consumer.idle_poll_interval = consumer
            .idle_poll_interval
            .saturating_mul(2)
            .min(MAX_IDLE_POLL_INTERVAL);
    }
    consumer.poll_after = Instant::now() + consumer.idle_poll_interval;
}

async fn write_delivery_content<W: AsyncWrite + Unpin>(
    writer: &mut W,
    channel: u16,
    delivery: &Delivery,
    queue_kind: QueueKind,
    frame_max: u32,
) -> std::result::Result<(), ProtocolError> {
    let mut properties = match &delivery.payload.ingress {
        IngressMetadata::Amqp {
            properties,
            headers,
            ..
        } => BasicProperties::from_storage(properties, headers),
        IngressMetadata::Kafka { .. } => BasicProperties::default(),
    };
    if delivery.death_count != 0 {
        let death = BTreeMap::from([
            (
                "count".to_owned(),
                FieldValue::I64(i64::from(delivery.death_count)),
            ),
            (
                "reason".to_owned(),
                FieldValue::LongString(b"rejected".to_vec()),
            ),
            (
                "queue".to_owned(),
                FieldValue::LongString(delivery.queue.as_bytes().to_vec()),
            ),
            (
                "exchange".to_owned(),
                FieldValue::LongString(delivery.exchange.as_bytes().to_vec()),
            ),
            (
                "routing-keys".to_owned(),
                FieldValue::Array(vec![FieldValue::LongString(
                    delivery.routing_key.as_bytes().to_vec(),
                )]),
            ),
        ]);
        properties.headers.insert(
            "x-death".to_owned(),
            FieldValue::Array(vec![FieldValue::Table(death)]),
        );
    }
    if queue_kind == QueueKind::Stream {
        let offset = i64::try_from(delivery.offset).map_err(|_| {
            ProtocolError::channel(
                REPLY_RESOURCE_ERROR,
                "stream offset exceeds AMQP signed long-long",
                CLASS_BASIC,
                60,
            )
        })?;
        properties
            .headers
            .insert("x-stream-offset".to_owned(), FieldValue::I64(offset));
    }
    write_content_buffered(
        writer,
        channel,
        &properties,
        &delivery.payload.payload,
        frame_max,
    )
    .await
}

async fn cleanup_channel(
    channel: u16,
    state: &mut ConnectionState,
    coordinator: &Arc<dyn BrokerCoordinator>,
) {
    if let Some(pending_bytes) = state
        .channels
        .get(&channel)
        .and_then(|channel_state| channel_state.pending_publish.as_ref())
        .map(|pending| pending.body.len())
    {
        state.buffered_publish_bytes = state.buffered_publish_bytes.saturating_sub(pending_bytes);
    }
    let tags = all_delivery_tags(channel, state);
    let _ = settle_tags(channel, &tags, false, true, state, coordinator).await;
    if let Some(channel_state) = state.channels.get(&channel) {
        state.consumer_count = state
            .consumer_count
            .saturating_sub(channel_state.consumers.len());
    }
}

fn has_classic_outstanding(state: &ConnectionState) -> bool {
    state.channels.values().any(|channel| {
        channel
            .outstanding
            .values()
            .any(|delivery| delivery.queue_kind == QueueKind::Classic)
    })
}

async fn renew_delivery_lease_if_due(
    state: &mut ConnectionState,
    coordinator: &Arc<dyn BrokerCoordinator>,
) -> std::result::Result<(), ProtocolError> {
    let now = Instant::now();
    if now < state.delivery_lease_renew_after {
        return Ok(());
    }
    state.delivery_lease_renew_after = now + DELIVERY_LEASE_RENEW_INTERVAL;
    if !has_classic_outstanding(state) {
        return Ok(());
    }
    let reply = coordinator_submit(
        Arc::clone(coordinator),
        BrokerCommand::RenewDeliveryLease {
            project: state.project,
            owner: state.delivery_owner,
            resolved_time_ms: 0,
        },
        CommitAcknowledgement::Published,
    )
    .await
    .map_err(|error| {
        ProtocolError::connection(
            REPLY_CONNECTION_FORCED,
            format!("delivery lease renewal failed: {}", error.message),
            CLASS_BASIC,
            60,
        )
    })?;
    if !matches!(reply, BrokerReply::Acknowledged) {
        return Err(ProtocolError::connection(
            REPLY_INTERNAL_ERROR,
            "broker returned an invalid delivery lease renewal reply",
            CLASS_BASIC,
            60,
        ));
    }
    Ok(())
}

async fn cleanup_connection(state: &mut ConnectionState, coordinator: &Arc<dyn BrokerCoordinator>) {
    let release_delivery_lease = state.delivery_lease_touched;
    let release_connection = state.connection_state_registered;
    let channels = state.channels.keys().copied().collect::<Vec<_>>();
    for channel in channels {
        cleanup_channel(channel, state, coordinator).await;
        state.channels.remove(&channel);
    }
    if release_connection {
        // Drops this connection's consumer registrations, its exclusive queues, and any
        // auto-delete queue it was the last consumer of. Best effort: the connection is already
        // gone, and an unreachable coordinator will collect the same state through lease expiry.
        let _ = coordinator_submit(
            Arc::clone(coordinator),
            BrokerCommand::ReleaseConnection {
                project: state.project,
                owner: state.delivery_owner,
            },
            CommitAcknowledgement::Published,
        )
        .await;
    }
    if release_delivery_lease {
        let _ = coordinator_submit(
            Arc::clone(coordinator),
            BrokerCommand::ReleaseDeliveryLease {
                project: state.project,
                owner: state.delivery_owner,
            },
            CommitAcknowledgement::Published,
        )
        .await;
    }
}

#[cfg(test)]
mod tests {
    use std::{
        process::Command,
        sync::atomic::{AtomicU64, Ordering},
    };

    use parking_lot::Mutex;

    use super::*;

    fn outstanding_delivery(
        queue: &str,
        consumer_id: u64,
        engine_delivery_tag: u64,
    ) -> OutstandingDelivery {
        OutstandingDelivery {
            consumer_id,
            consumer_tag: Some("tag".to_owned()),
            queue: queue.to_owned(),
            queue_kind: QueueKind::Classic,
            engine_delivery_tag,
            stream_offset: engine_delivery_tag.saturating_sub(1),
        }
    }

    fn channel_with(deliveries: &[(u64, &str, u64, u64)]) -> ChannelState {
        let mut channel = ChannelState::new();
        for (app_tag, queue, consumer, engine_tag) in deliveries {
            channel.outstanding.insert(
                *app_tag,
                outstanding_delivery(queue, *consumer, *engine_tag),
            );
        }
        channel
    }

    #[test]
    fn delivery_batches_are_charged_in_proportion_and_clamped_to_the_budget() {
        // A batch of one must cost exactly what the previous single-delivery reserve cost, so
        // batching cannot quietly loosen the process budget.
        assert_eq!(delivery_reserve_bytes(1), EGRESS_DELIVERY_RESERVE);
        assert_eq!(
            delivery_reserve_bytes(4),
            MAX_DELIVERY_RESULT_BYTES * 4 + PUBLISH_STORAGE_FIXED_RESERVE
        );
        // The credit is clamped to something the budget could actually admit, and never below one.
        let budget = process_broker_memory().maximum_bytes();
        for credit in [0_u32, 1, 2, 7, 64, u32::MAX] {
            let batch = admissible_delivery_batch(credit);
            assert!(batch >= 1);
            assert!(batch <= MAX_DELIVERIES_PER_PUMP);
            assert!(batch <= credit.max(1));
            assert!(batch == 1 || delivery_reserve_bytes(batch) <= budget);
        }
    }

    #[test]
    fn multiple_settlement_collapses_a_complete_prefix_into_one_ordered_command() {
        // Four deliveries from one consumer, all acknowledged together: the state machine settles
        // by offset prefix, so this is one command rather than four ordered writes.
        let channel = channel_with(&[
            (1, "orders", 7, 11),
            (2, "orders", 7, 12),
            (3, "orders", 7, 13),
            (4, "orders", 7, 14),
        ]);
        let groups = prefix_settleable_groups(&channel, &[1, 2, 3, 4]);
        assert_eq!(groups.get(&("orders".to_owned(), 7)), Some(&14));
        assert_eq!(groups.len(), 1);
    }

    #[test]
    fn multiple_settlement_falls_back_when_the_prefix_would_swallow_a_live_delivery() {
        // Tag 2 is still outstanding but was not acknowledged. A prefix up to engine tag 13 would
        // settle it anyway, so this group must be settled one delivery at a time instead.
        let channel = channel_with(&[
            (1, "orders", 7, 11),
            (2, "orders", 7, 12),
            (3, "orders", 7, 13),
        ]);
        assert!(prefix_settleable_groups(&channel, &[1, 3]).is_empty());
        // Acknowledging a true prefix of the same channel is still collapsed.
        assert_eq!(
            prefix_settleable_groups(&channel, &[1, 2]).get(&("orders".to_owned(), 7)),
            Some(&12)
        );
    }

    #[test]
    fn multiple_settlement_groups_each_queue_and_consumer_independently() {
        // One channel can hold deliveries from several consumers and queues. Each is its own
        // prefix domain in the state machine, so each gets its own command, and a queue with a
        // gap does not prevent the others from collapsing.
        let channel = channel_with(&[
            (1, "orders", 7, 11),
            (2, "orders", 7, 12),
            (3, "audit", 9, 4),
            (4, "audit", 9, 5),
            (5, "audit", 9, 6),
        ]);
        let groups = prefix_settleable_groups(&channel, &[1, 2, 3, 5]);
        assert_eq!(groups.get(&("orders".to_owned(), 7)), Some(&12));
        assert_eq!(groups.get(&("audit".to_owned(), 9)), None);

        let all = prefix_settleable_groups(&channel, &[1, 2, 3, 4, 5]);
        assert_eq!(all.get(&("orders".to_owned(), 7)), Some(&12));
        assert_eq!(all.get(&("audit".to_owned(), 9)), Some(&6));
    }

    #[test]
    fn stream_queue_deliveries_are_never_prefix_settled() {
        // Stream queues carry no durable delivery lease, so they issue no settlement command at
        // all and must never be folded into a classic-queue prefix.
        let mut channel = ChannelState::new();
        let mut delivery = outstanding_delivery("events", 3, 1);
        delivery.queue_kind = QueueKind::Stream;
        channel.outstanding.insert(1, delivery);
        assert!(prefix_settleable_groups(&channel, &[1]).is_empty());
    }

    struct TestCoordinator {
        broker: Mutex<super::super::engine::BrokerStateMachine>,
        segments: crate::storage::SegmentStore,
        _directory: tempfile::TempDir,
        index: AtomicU64,
    }

    impl TestCoordinator {
        fn new() -> Result<Self> {
            let directory = tempfile::tempdir()?;
            let segments = crate::storage::SegmentStore::open(directory.path(), 64 * 1024 * 1024)?;
            Ok(Self {
                broker: Mutex::new(super::super::engine::BrokerStateMachine::default()),
                segments,
                _directory: directory,
                index: AtomicU64::new(0),
            })
        }
    }

    impl BrokerCoordinator for TestCoordinator {
        fn submit(
            &self,
            command: BrokerCommand,
            _wait: CommitAcknowledgement,
        ) -> Result<super::super::engine::BrokerCommit> {
            let reply = self.broker.lock().apply(command, &self.segments)?;
            let index = self.index.fetch_add(1, Ordering::Relaxed).saturating_add(1);
            Ok(super::super::engine::BrokerCommit {
                bookmark: crate::Bookmark { term: 1, index },
                reply,
                application: crate::engine::ApplicationWait::Complete,
            })
        }

        fn snapshot(&self) -> Result<super::super::engine::BrokerStateMachine> {
            Ok(self.broker.lock().clone())
        }

        fn fetch_partition(
            &self,
            project: ProjectId,
            topic: &str,
            partition: i32,
            offset: u64,
            maximum_bytes: usize,
        ) -> Result<Vec<(u64, Arc<super::super::engine::PayloadRecord>)>> {
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

        fn queue_info(
            &self,
            project: ProjectId,
            name: &str,
        ) -> Result<Option<super::super::engine::QueueInfo>> {
            Ok(self.broker.lock().queue_info(project, name))
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
            self.broker.lock().read_stream_queue(
                project,
                queue,
                offset,
                maximum,
                consumer,
                automatic_ack,
                &self.segments,
            )
        }
    }

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    enum BlockingOperation {
        Submit,
        StreamFetch,
    }

    struct BlockingCoordinator {
        operation: BlockingOperation,
        entered: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
        release: Mutex<std::sync::mpsc::Receiver<()>>,
    }

    impl BlockingCoordinator {
        fn block(&self, operation: BlockingOperation) -> Result<()> {
            if self.operation != operation {
                return Ok(());
            }
            if let Some(entered) = self.entered.lock().take() {
                let _ = entered.send(());
            }
            self.release
                .lock()
                .recv()
                .map_err(|_| Error::internal("AMQP blocking test release channel closed"))
        }
    }

    impl BrokerCoordinator for BlockingCoordinator {
        fn submit(
            &self,
            _command: BrokerCommand,
            _wait: CommitAcknowledgement,
        ) -> Result<super::super::engine::BrokerCommit> {
            self.block(BlockingOperation::Submit)?;
            Ok(super::super::engine::BrokerCommit {
                bookmark: crate::Bookmark { term: 1, index: 1 },
                reply: BrokerReply::Acknowledged,
                application: crate::engine::ApplicationWait::Complete,
            })
        }

        fn snapshot(&self) -> Result<super::super::engine::BrokerStateMachine> {
            Ok(super::super::engine::BrokerStateMachine::default())
        }

        fn fetch_partition(
            &self,
            _project: ProjectId,
            _topic: &str,
            _partition: i32,
            _offset: u64,
            _maximum_bytes: usize,
        ) -> Result<Vec<(u64, Arc<super::super::engine::PayloadRecord>)>> {
            Ok(Vec::new())
        }

        fn list_offset(
            &self,
            _project: ProjectId,
            _topic: &str,
            _partition: i32,
            _timestamp: i64,
        ) -> Result<Option<(u64, i64)>> {
            Ok(Some((0, -1)))
        }

        fn queue_info(
            &self,
            _project: ProjectId,
            _name: &str,
        ) -> Result<Option<super::super::engine::QueueInfo>> {
            Ok(None)
        }

        fn fetch_stream_queue(
            &self,
            _project: ProjectId,
            _queue: &str,
            _offset: StreamOffset,
            _maximum: usize,
            _consumer: u64,
            _automatic_ack: bool,
        ) -> Result<Vec<Delivery>> {
            self.block(BlockingOperation::StreamFetch)?;
            Ok(Vec::new())
        }
    }

    fn blocking_coordinator(
        operation: BlockingOperation,
    ) -> (
        Arc<dyn BrokerCoordinator>,
        tokio::sync::oneshot::Receiver<()>,
        std::sync::mpsc::Sender<()>,
    ) {
        let (entered_sender, entered_receiver) = tokio::sync::oneshot::channel();
        let (release_sender, release_receiver) = std::sync::mpsc::channel();
        (
            Arc::new(BlockingCoordinator {
                operation,
                entered: Mutex::new(Some(entered_sender)),
                release: Mutex::new(release_receiver),
            }),
            entered_receiver,
            release_sender,
        )
    }

    async fn wait_for_full_governor(
        governor: &super::super::memory::BrokerMemoryGovernor,
        bytes: usize,
    ) -> Result<()> {
        tokio::time::timeout(Duration::from_secs(2), async {
            while governor.available_bytes() != bytes {
                tokio::task::yield_now().await;
            }
        })
        .await
        .map_err(|_| Error::internal("AMQP blocking operation did not release memory"))
    }

    #[tokio::test]
    async fn aborted_submit_waiter_holds_memory_until_blocking_work_finishes() -> Result<()> {
        const QUANTUM: usize = 64 * 1024;
        let governor = Arc::new(super::super::memory::BrokerMemoryGovernor::new(
            4 * QUANTUM,
        )?);
        let reservation = governor.reserve(QUANTUM).await?;
        let (coordinator, entered, release) = blocking_coordinator(BlockingOperation::Submit);
        let waiter = tokio::spawn(async move {
            coordinator_submit_reserved(
                coordinator,
                BrokerCommand::Retain {
                    project: ProjectId::random(),
                    resolved_time_ms: 0,
                },
                CommitAcknowledgement::Published,
                &reservation,
            )
            .await
        });
        tokio::time::timeout(Duration::from_secs(2), entered)
            .await
            .map_err(|_| Error::internal("AMQP blocking submit did not start"))?
            .map_err(|_| Error::internal("AMQP blocking submit entry channel closed"))?;
        assert_eq!(governor.available_bytes(), 3 * QUANTUM);
        waiter.abort();
        assert!(
            waiter
                .await
                .expect_err("AMQP submit waiter must be cancelled")
                .is_cancelled()
        );
        assert_eq!(governor.available_bytes(), 3 * QUANTUM);
        release
            .send(())
            .map_err(|_| Error::internal("AMQP blocking submit already exited"))?;
        wait_for_full_governor(&governor, 4 * QUANTUM).await
    }

    #[tokio::test]
    async fn aborted_stream_fetch_waiter_holds_memory_until_blocking_work_finishes() -> Result<()> {
        const QUANTUM: usize = 64 * 1024;
        let governor = Arc::new(super::super::memory::BrokerMemoryGovernor::new(
            4 * QUANTUM,
        )?);
        let reservation = governor.reserve(QUANTUM).await?;
        let (coordinator, entered, release) = blocking_coordinator(BlockingOperation::StreamFetch);
        let waiter = tokio::spawn(async move {
            coordinator_fetch_stream_queue(
                coordinator,
                ProjectId::random(),
                "events".to_owned(),
                StreamOffset::First,
                1,
                1,
                true,
                &reservation,
            )
            .await
        });
        tokio::time::timeout(Duration::from_secs(2), entered)
            .await
            .map_err(|_| Error::internal("AMQP blocking stream fetch did not start"))?
            .map_err(|_| Error::internal("AMQP blocking stream fetch entry channel closed"))?;
        assert_eq!(governor.available_bytes(), 3 * QUANTUM);
        waiter.abort();
        assert!(
            waiter
                .await
                .expect_err("AMQP stream-fetch waiter must be cancelled")
                .is_cancelled()
        );
        assert_eq!(governor.available_bytes(), 3 * QUANTUM);
        release
            .send(())
            .map_err(|_| Error::internal("AMQP blocking stream fetch already exited"))?;
        wait_for_full_governor(&governor, 4 * QUANTUM).await
    }

    fn protocol_error(error: ProtocolError) -> Error {
        Error::new(ErrorCode::ProtocolViolation, error.text)
    }

    async fn open_test_peer(
        project: ProjectId,
        coordinator: Arc<dyn BrokerCoordinator>,
        heartbeat: u16,
    ) -> Result<(
        tokio::io::DuplexStream,
        tokio_util::sync::CancellationToken,
        tokio::task::JoinHandle<Result<()>>,
    )> {
        let shutdown = tokio_util::sync::CancellationToken::new();
        let (server_io, mut client_io) = tokio::io::duplex(2 * 1024 * 1024);
        let server = tokio::spawn(QueueServer::serve_authenticated_transport(
            server_io,
            project,
            coordinator,
            shutdown.child_token(),
        ));
        client_io.write_all(PROTOCOL_HEADER).await?;
        assert_method(
            &read_test_frame(&mut client_io).await?,
            CLASS_CONNECTION,
            10,
        )?;
        let mut start_ok = Writer::method(CLASS_CONNECTION, 11);
        start_ok.field_table(&BTreeMap::new())?;
        start_ok.short_string("EXTERNAL")?;
        start_ok.long_bytes(&[])?;
        start_ok.short_string("en_US")?;
        write_frame(
            &mut client_io,
            FRAME_METHOD,
            0,
            start_ok.as_slice(),
            SERVER_FRAME_MAX,
        )
        .await?;
        assert_method(
            &read_test_frame(&mut client_io).await?,
            CLASS_CONNECTION,
            30,
        )?;
        let mut tune_ok = Writer::method(CLASS_CONNECTION, 31);
        tune_ok.u16(32);
        tune_ok.u32(SERVER_FRAME_MAX);
        tune_ok.u16(heartbeat);
        write_frame(
            &mut client_io,
            FRAME_METHOD,
            0,
            tune_ok.as_slice(),
            SERVER_FRAME_MAX,
        )
        .await?;
        let mut open = Writer::method(CLASS_CONNECTION, 40);
        open.short_string("/")?;
        open.short_string("")?;
        open.u8(0);
        write_frame(
            &mut client_io,
            FRAME_METHOD,
            0,
            open.as_slice(),
            SERVER_FRAME_MAX,
        )
        .await?;
        assert_method(
            &read_test_frame(&mut client_io).await?,
            CLASS_CONNECTION,
            41,
        )?;
        let mut channel_open = Writer::method(CLASS_CHANNEL, 10);
        channel_open.short_string("")?;
        write_frame(
            &mut client_io,
            FRAME_METHOD,
            1,
            channel_open.as_slice(),
            SERVER_FRAME_MAX,
        )
        .await?;
        assert_method(&read_test_frame(&mut client_io).await?, CLASS_CHANNEL, 11)?;
        Ok((client_io, shutdown, server))
    }

    async fn send_test_method(
        peer: &mut tokio::io::DuplexStream,
        channel: u16,
        method: &Writer,
    ) -> Result<()> {
        write_frame(
            peer,
            FRAME_METHOD,
            channel,
            method.as_slice(),
            SERVER_FRAME_MAX,
        )
        .await
    }

    async fn send_test_publish(
        peer: &mut tokio::io::DuplexStream,
        exchange: &str,
        routing_key: &str,
        mandatory: bool,
        body: &[u8],
    ) -> Result<()> {
        let mut publish = Writer::method(CLASS_BASIC, 40);
        publish.u16(0);
        publish.short_string(exchange)?;
        publish.short_string(routing_key)?;
        publish.u8(u8::from(mandatory));
        send_test_method(peer, 1, &publish).await?;
        let mut header = Writer::new();
        header.u16(CLASS_BASIC);
        header.u16(0);
        header.u64(body.len() as u64);
        BasicProperties {
            delivery_mode: Some(2),
            ..BasicProperties::default()
        }
        .encode(&mut header)?;
        write_frame(peer, FRAME_HEADER, 1, header.as_slice(), SERVER_FRAME_MAX).await?;
        write_frame(peer, FRAME_BODY, 1, body, SERVER_FRAME_MAX).await
    }

    async fn read_test_delivery(
        peer: &mut tokio::io::DuplexStream,
    ) -> Result<(u64, bool, BasicProperties, Vec<u8>)> {
        let method = read_test_frame(peer).await?;
        assert_method(&method, CLASS_BASIC, 60)?;
        let mut reader = Reader::new(&method.payload);
        let _class = reader.u16().map_err(protocol_error)?;
        let _method = reader.u16().map_err(protocol_error)?;
        let _consumer_tag = reader.short_string().map_err(protocol_error)?;
        let delivery_tag = reader.u64().map_err(protocol_error)?;
        let redelivered = reader.u8().map_err(protocol_error)? != 0;
        let _exchange = reader.short_string().map_err(protocol_error)?;
        let _routing_key = reader.short_string().map_err(protocol_error)?;
        reader.finish().map_err(protocol_error)?;
        let header = read_test_frame(peer).await?;
        if header.kind != FRAME_HEADER {
            return Err(Error::new(
                ErrorCode::ProtocolViolation,
                "AMQP test delivery omitted its content header",
            ));
        }
        let mut reader = Reader::new(&header.payload);
        if reader.u16().map_err(protocol_error)? != CLASS_BASIC
            || reader.u16().map_err(protocol_error)? != 0
        {
            return Err(Error::new(
                ErrorCode::ProtocolViolation,
                "AMQP test delivery content header is invalid",
            ));
        }
        let body_size = usize::try_from(reader.u64().map_err(protocol_error)?)
            .map_err(|_| Error::internal("AMQP test body exceeds usize"))?;
        let properties = BasicProperties::parse(&mut reader).map_err(protocol_error)?;
        let mut body = Vec::with_capacity(body_size);
        while body.len() < body_size {
            let frame = read_test_frame(peer).await?;
            if frame.kind != FRAME_BODY {
                return Err(Error::new(
                    ErrorCode::ProtocolViolation,
                    "AMQP test delivery omitted a content body frame",
                ));
            }
            body.extend_from_slice(&frame.payload);
        }
        if body.len() != body_size {
            return Err(Error::new(
                ErrorCode::ProtocolViolation,
                "AMQP test delivery body size changed",
            ));
        }
        Ok((delivery_tag, redelivered, properties, body))
    }

    async fn read_test_get(
        peer: &mut tokio::io::DuplexStream,
    ) -> Result<(u64, bool, BasicProperties, Vec<u8>)> {
        let method = read_test_frame(peer).await?;
        assert_method(&method, CLASS_BASIC, 71)?;
        let mut reader = Reader::new(&method.payload);
        let _class = reader.u16().map_err(protocol_error)?;
        let _method = reader.u16().map_err(protocol_error)?;
        let delivery_tag = reader.u64().map_err(protocol_error)?;
        let redelivered = reader.u8().map_err(protocol_error)? != 0;
        let _exchange = reader.short_string().map_err(protocol_error)?;
        let _routing_key = reader.short_string().map_err(protocol_error)?;
        let _remaining = reader.u32().map_err(protocol_error)?;
        reader.finish().map_err(protocol_error)?;
        let header = read_test_frame(peer).await?;
        if header.kind != FRAME_HEADER {
            return Err(Error::new(
                ErrorCode::ProtocolViolation,
                "AMQP test get omitted its content header",
            ));
        }
        let mut reader = Reader::new(&header.payload);
        if reader.u16().map_err(protocol_error)? != CLASS_BASIC
            || reader.u16().map_err(protocol_error)? != 0
        {
            return Err(Error::new(
                ErrorCode::ProtocolViolation,
                "AMQP test get content header is invalid",
            ));
        }
        let body_size = usize::try_from(reader.u64().map_err(protocol_error)?)
            .map_err(|_| Error::internal("AMQP test get body exceeds usize"))?;
        let properties = BasicProperties::parse(&mut reader).map_err(protocol_error)?;
        let mut body = Vec::with_capacity(body_size);
        while body.len() < body_size {
            let frame = read_test_frame(peer).await?;
            if frame.kind != FRAME_BODY {
                return Err(Error::new(
                    ErrorCode::ProtocolViolation,
                    "AMQP test get omitted a content body frame",
                ));
            }
            body.extend_from_slice(&frame.payload);
        }
        Ok((delivery_tag, redelivered, properties, body))
    }

    #[test]
    fn field_table_round_trips_nested_values() -> Result<()> {
        let table = BTreeMap::from([
            ("enabled".to_owned(), FieldValue::Boolean(true)),
            (
                "nested".to_owned(),
                FieldValue::Table(BTreeMap::from([("offset".to_owned(), FieldValue::I64(42))])),
            ),
            (
                "items".to_owned(),
                FieldValue::Array(vec![FieldValue::LongString(b"first".to_vec())]),
            ),
        ]);
        let mut writer = Writer::new();
        writer.field_table(&table)?;
        let mut reader = Reader::new(writer.as_slice());
        let decoded = reader.field_table(0).map_err(protocol_error)?;
        reader.finish().map_err(protocol_error)?;
        assert!(matches!(
            decoded.get("enabled"),
            Some(FieldValue::Boolean(true))
        ));
        assert!(matches!(
            decoded.get("nested"),
            Some(FieldValue::Table(nested))
                if matches!(nested.get("offset"), Some(FieldValue::I64(42)))
        ));
        assert!(matches!(
            decoded.get("items"),
            Some(FieldValue::Array(items))
                if matches!(items.as_slice(), [FieldValue::LongString(value)] if value == b"first")
        ));
        Ok(())
    }

    #[test]
    fn frame_decoder_waits_for_complete_frame_and_checks_terminator() -> Result<()> {
        let payload = [0, 60, 0, 10];
        let mut encoded = Vec::new();
        encoded.push(FRAME_METHOD);
        encoded.extend_from_slice(&1u16.to_be_bytes());
        encoded.extend_from_slice(&4u32.to_be_bytes());
        encoded.extend_from_slice(&payload);
        encoded.push(FRAME_END);
        let mut input = BytesMut::from(&encoded[..5]);
        assert!(
            try_decode_frame(&mut input, SERVER_FRAME_MAX)
                .map_err(protocol_error)?
                .is_none()
        );
        input.extend_from_slice(&encoded[5..]);
        let Some(frame) = try_decode_frame(&mut input, SERVER_FRAME_MAX).map_err(protocol_error)?
        else {
            return Err(Error::internal("complete AMQP frame was not decoded"));
        };
        assert_eq!(frame.kind, FRAME_METHOD);
        assert_eq!(frame.channel, 1);
        assert_eq!(frame.payload.as_ref(), payload);
        assert!(input.is_empty());

        let final_index = encoded.len().saturating_sub(1);
        encoded[final_index] = 0;
        let mut invalid = BytesMut::from(encoded.as_slice());
        assert!(try_decode_frame(&mut invalid, SERVER_FRAME_MAX).is_err());
        Ok(())
    }

    #[test]
    fn stream_offset_and_retention_arguments_are_strict() -> Result<()> {
        let timestamp = BTreeMap::from([(
            "x-stream-offset".to_owned(),
            FieldValue::Timestamp(1_700_000_000),
        )]);
        assert!(matches!(
            parse_consumer_arguments(QueueKind::Stream, &timestamp).map_err(protocol_error)?,
            StreamOffset::Timestamp(1_700_000_000_000)
        ));
        let first = BTreeMap::from([(
            "x-stream-offset".to_owned(),
            FieldValue::LongString(b"first".to_vec()),
        )]);
        assert!(matches!(
            parse_consumer_arguments(QueueKind::Stream, &first).map_err(protocol_error)?,
            StreamOffset::First
        ));
        assert_eq!(
            parse_retention_duration("7D").map_err(protocol_error)?,
            604_800_000
        );
        assert!(parse_retention_duration("0s").is_err());
        assert!(parse_retention_duration("4years").is_err());
        Ok(())
    }

    #[test]
    fn basic_properties_survive_canonical_storage_encoding() -> Result<()> {
        let original = BasicProperties {
            content_type: Some("application/json".to_owned()),
            delivery_mode: Some(2),
            correlation_id: Some("request-42".to_owned()),
            timestamp: Some(1_700_000_000),
            headers: BTreeMap::from([(
                "trace".to_owned(),
                FieldValue::LongString(b"abc".to_vec()),
            )]),
            ..BasicProperties::default()
        };
        let (properties, headers) = original.into_storage()?;
        let decoded = BasicProperties::from_storage(&properties, &headers);
        assert_eq!(decoded.content_type.as_deref(), Some("application/json"));
        assert_eq!(decoded.delivery_mode, Some(2));
        assert_eq!(decoded.correlation_id.as_deref(), Some("request-42"));
        assert_eq!(decoded.timestamp, Some(1_700_000_000));
        assert!(matches!(
            decoded.headers.get("trace"),
            Some(FieldValue::LongString(value)) if value == b"abc"
        ));
        Ok(())
    }

    #[test]
    fn admitted_maximum_publish_is_deliverable_from_classic_and_stream_queues() -> Result<()> {
        assert!(validate_publish_body_size(MAX_MESSAGE_BYTES as u64).is_ok());
        let oversized = validate_publish_body_size(MAX_MESSAGE_BYTES as u64 + 1)
            .expect_err("AMQP body above the deliverable limit must be rejected");
        assert_eq!(oversized.code, REPLY_CONTENT_TOO_LARGE);

        let project = ProjectId::random();
        let coordinator = TestCoordinator::new()?;
        let retention = RetentionPolicy {
            max_age_ms: None,
            max_bytes: None,
        };
        let _ = coordinator.submit(
            BrokerCommand::CreateExchange {
                project,
                name: "events".to_owned(),
                kind: AmqpExchangeKind::Direct,
                durable: true,
                passive: false,
            },
            CommitAcknowledgement::Published,
        )?;
        for (queue, kind) in [
            ("classic", QueueKind::Classic),
            ("stream", QueueKind::Stream),
        ] {
            let _ = coordinator.submit(
                BrokerCommand::CreateQueue {
                    project,
                    name: queue.to_owned(),
                    kind,
                    durable: true,
                    passive: false,
                    dead_letter_exchange: None,
                    dead_letter_routing_key: None,
                    retention,
                    exclusive_owner: None,
                    auto_delete: false,
                },
                CommitAcknowledgement::Published,
            )?;
            let _ = coordinator.submit(
                BrokerCommand::BindQueue {
                    project,
                    exchange: "events".to_owned(),
                    queue: queue.to_owned(),
                    routing_key: "all".to_owned(),
                },
                CommitAcknowledgement::Published,
            )?;
        }

        // An AMQP content header is bounded by the negotiated 128 KiB frame. This near-bound
        // canonical header plus the maximum admitted body exercises the complete stored-record
        // accounting rather than validating the body in isolation.
        let headers = BTreeMap::from([(
            "semantic".to_owned(),
            vec![0x5a; SERVER_FRAME_MAX as usize - 4 * 1024],
        )]);
        let published = coordinator.submit(
            BrokerCommand::PublishAmqp {
                project,
                exchange: "events".to_owned(),
                routing_key: "all".to_owned(),
                mandatory: true,
                resolved_time_ms: 1,
                properties: BTreeMap::new(),
                headers,
                payload: vec![0xa5; MAX_MESSAGE_BYTES],
            },
            CommitAcknowledgement::Published,
        )?;
        assert!(matches!(published.reply, BrokerReply::Published { .. }));

        let classic = coordinator.submit(
            BrokerCommand::DeliverQueue {
                project,
                queue: "classic".to_owned(),
                offset: StreamOffset::First,
                maximum: 1,
                owner: Uuid::new_v4(),
                consumer: 1,
                automatic_ack: true,
                resolved_time_ms: 2,
            },
            CommitAcknowledgement::Published,
        )?;
        let BrokerReply::Deliveries(classic) = classic.reply else {
            return Err(Error::internal("classic queue did not return deliveries"));
        };
        assert_eq!(classic.len(), 1);
        assert_eq!(classic[0].payload.payload.len(), MAX_MESSAGE_BYTES);

        let stream =
            coordinator.fetch_stream_queue(project, "stream", StreamOffset::First, 1, 2, true)?;
        assert_eq!(stream.len(), 1);
        assert_eq!(stream[0].payload.payload.len(), MAX_MESSAGE_BYTES);
        Ok(())
    }

    #[tokio::test]
    #[allow(clippy::too_many_lines)]
    async fn wire_peer_exercises_delivery_recovery_dead_letter_and_stream_methods() -> Result<()> {
        tokio::time::timeout(Duration::from_secs(10), async {
            let project = ProjectId::random();
            let coordinator: Arc<dyn BrokerCoordinator> = Arc::new(TestCoordinator::new()?);
            let (mut peer, shutdown, server) = open_test_peer(project, coordinator, 0).await?;

            write_frame(&mut peer, FRAME_HEARTBEAT, 0, &[], SERVER_FRAME_MAX).await?;

            for active in [false, true] {
                let mut flow = Writer::method(CLASS_CHANNEL, 20);
                flow.u8(u8::from(active));
                send_test_method(&mut peer, 1, &flow).await?;
                let response = read_test_frame(&mut peer).await?;
                assert_method(&response, CLASS_CHANNEL, 21)?;
                let mut reader = Reader::new(&response.payload);
                let _class = reader.u16().map_err(protocol_error)?;
                let _method = reader.u16().map_err(protocol_error)?;
                assert_eq!(reader.u8().map_err(protocol_error)?, u8::from(active));
            }

            let mut confirm = Writer::method(CLASS_CONFIRM, 10);
            confirm.u8(0);
            send_test_method(&mut peer, 1, &confirm).await?;
            assert_method(&read_test_frame(&mut peer).await?, CLASS_CONFIRM, 11)?;

            for (name, kind) in [("events", "direct"), ("dead", "direct")] {
                let mut exchange = Writer::method(CLASS_EXCHANGE, 10);
                exchange.u16(0);
                exchange.short_string(name)?;
                exchange.short_string(kind)?;
                exchange.u8(0x02);
                exchange.field_table(&BTreeMap::new())?;
                send_test_method(&mut peer, 1, &exchange).await?;
                assert_method(&read_test_frame(&mut peer).await?, CLASS_EXCHANGE, 11)?;
            }

            let declarations = [
                ("work", BTreeMap::new()),
                (
                    "source",
                    BTreeMap::from([
                        (
                            "x-dead-letter-exchange".to_owned(),
                            FieldValue::LongString(b"dead".to_vec()),
                        ),
                        (
                            "x-dead-letter-routing-key".to_owned(),
                            FieldValue::LongString(b"rejected".to_vec()),
                        ),
                    ]),
                ),
                ("failed", BTreeMap::new()),
                (
                    "stream",
                    BTreeMap::from([
                        (
                            "x-queue-type".to_owned(),
                            FieldValue::LongString(b"stream".to_vec()),
                        ),
                        ("x-max-length-bytes".to_owned(), FieldValue::I64(1024)),
                        (
                            "x-overflow".to_owned(),
                            FieldValue::LongString(b"drop-head".to_vec()),
                        ),
                    ]),
                ),
            ];
            for (name, arguments) in declarations {
                let mut queue = Writer::method(CLASS_QUEUE, 10);
                queue.u16(0);
                queue.short_string(name)?;
                queue.u8(0x02);
                queue.field_table(&arguments)?;
                send_test_method(&mut peer, 1, &queue).await?;
                assert_method(&read_test_frame(&mut peer).await?, CLASS_QUEUE, 11)?;
            }

            let mut bind = Writer::method(CLASS_QUEUE, 20);
            bind.u16(0);
            bind.short_string("failed")?;
            bind.short_string("dead")?;
            bind.short_string("rejected")?;
            bind.u8(0);
            bind.field_table(&BTreeMap::new())?;
            send_test_method(&mut peer, 1, &bind).await?;
            assert_method(&read_test_frame(&mut peer).await?, CLASS_QUEUE, 21)?;

            send_test_publish(&mut peer, "", "missing", true, b"return-me").await?;
            assert_method(&read_test_frame(&mut peer).await?, CLASS_BASIC, 50)?;
            let returned_header = read_test_frame(&mut peer).await?;
            assert_eq!(returned_header.kind, FRAME_HEADER);
            let returned_body = read_test_frame(&mut peer).await?;
            assert_eq!(returned_body.kind, FRAME_BODY);
            assert_eq!(returned_body.payload.as_ref(), b"return-me");
            assert_method(&read_test_frame(&mut peer).await?, CLASS_BASIC, 80)?;

            let mut qos = Writer::method(CLASS_BASIC, 10);
            qos.u32(0);
            qos.u16(1);
            qos.u8(0);
            send_test_method(&mut peer, 1, &qos).await?;
            assert_method(&read_test_frame(&mut peer).await?, CLASS_BASIC, 11)?;

            let mut consume = Writer::method(CLASS_BASIC, 20);
            consume.u16(0);
            consume.short_string("work")?;
            consume.short_string("worker")?;
            consume.u8(0);
            consume.field_table(&BTreeMap::new())?;
            send_test_method(&mut peer, 1, &consume).await?;
            assert_method(&read_test_frame(&mut peer).await?, CLASS_BASIC, 21)?;

            send_test_publish(&mut peer, "", "work", false, b"requeue").await?;
            assert_method(&read_test_frame(&mut peer).await?, CLASS_BASIC, 80)?;
            let (tag, redelivered, _, body) = read_test_delivery(&mut peer).await?;
            assert!(!redelivered);
            assert_eq!(body, b"requeue");
            let mut nack = Writer::method(CLASS_BASIC, 120);
            nack.u64(tag);
            nack.u8(0x02);
            send_test_method(&mut peer, 1, &nack).await?;
            let (tag, redelivered, _, body) = read_test_delivery(&mut peer).await?;
            assert!(redelivered);
            assert_eq!(body, b"requeue");
            let mut ack = Writer::method(CLASS_BASIC, 80);
            ack.u64(tag);
            ack.u8(0);
            send_test_method(&mut peer, 1, &ack).await?;

            send_test_publish(&mut peer, "", "work", false, b"recover").await?;
            assert_method(&read_test_frame(&mut peer).await?, CLASS_BASIC, 80)?;
            let (_tag, _, _, body) = read_test_delivery(&mut peer).await?;
            assert_eq!(body, b"recover");
            let mut recover_async = Writer::method(CLASS_BASIC, 100);
            recover_async.u8(1);
            send_test_method(&mut peer, 1, &recover_async).await?;
            let (_tag, redelivered, _, _) = read_test_delivery(&mut peer).await?;
            assert!(redelivered);
            let mut recover = Writer::method(CLASS_BASIC, 110);
            recover.u8(1);
            send_test_method(&mut peer, 1, &recover).await?;
            assert_method(&read_test_frame(&mut peer).await?, CLASS_BASIC, 111)?;
            let (tag, redelivered, _, _) = read_test_delivery(&mut peer).await?;
            assert!(redelivered);
            let mut ack = Writer::method(CLASS_BASIC, 80);
            ack.u64(tag);
            ack.u8(0);
            send_test_method(&mut peer, 1, &ack).await?;

            send_test_publish(&mut peer, "", "source", false, b"dead-letter").await?;
            assert_method(&read_test_frame(&mut peer).await?, CLASS_BASIC, 80)?;
            let mut get = Writer::method(CLASS_BASIC, 70);
            get.u16(0);
            get.short_string("source")?;
            get.u8(0);
            send_test_method(&mut peer, 1, &get).await?;
            let (tag, _, _, body) = read_test_get(&mut peer).await?;
            assert_eq!(body, b"dead-letter");
            let mut reject = Writer::method(CLASS_BASIC, 90);
            reject.u64(tag);
            reject.u8(0);
            send_test_method(&mut peer, 1, &reject).await?;
            let mut get = Writer::method(CLASS_BASIC, 70);
            get.u16(0);
            get.short_string("failed")?;
            get.u8(1);
            send_test_method(&mut peer, 1, &get).await?;
            let (_, _, properties, body) = read_test_get(&mut peer).await?;
            assert_eq!(body, b"dead-letter");
            assert!(matches!(
                properties.headers.get("x-death"),
                Some(FieldValue::Array(deaths)) if !deaths.is_empty()
            ));

            send_test_publish(&mut peer, "", "stream", false, b"stream-value").await?;
            assert_method(&read_test_frame(&mut peer).await?, CLASS_BASIC, 80)?;
            let mut stream_consume = Writer::method(CLASS_BASIC, 20);
            stream_consume.u16(0);
            stream_consume.short_string("stream")?;
            stream_consume.short_string("streamer")?;
            stream_consume.u8(0);
            stream_consume.field_table(&BTreeMap::from([(
                "x-stream-offset".to_owned(),
                FieldValue::LongString(b"first".to_vec()),
            )]))?;
            send_test_method(&mut peer, 1, &stream_consume).await?;
            assert_method(&read_test_frame(&mut peer).await?, CLASS_BASIC, 21)?;
            let (tag, _, properties, body) = read_test_delivery(&mut peer).await?;
            assert_eq!(body, b"stream-value");
            assert!(matches!(
                properties.headers.get("x-stream-offset"),
                Some(FieldValue::I64(0))
            ));
            let mut ack = Writer::method(CLASS_BASIC, 80);
            ack.u64(tag);
            ack.u8(0);
            send_test_method(&mut peer, 1, &ack).await?;

            for consumer in ["worker", "streamer"] {
                let mut cancel = Writer::method(CLASS_BASIC, 30);
                cancel.short_string(consumer)?;
                cancel.u8(0);
                send_test_method(&mut peer, 1, &cancel).await?;
                assert_method(&read_test_frame(&mut peer).await?, CLASS_BASIC, 31)?;
            }

            let mut close_channel = Writer::method(CLASS_CHANNEL, 40);
            close_channel.u16(200);
            close_channel.short_string("done")?;
            close_channel.u16(0);
            close_channel.u16(0);
            send_test_method(&mut peer, 1, &close_channel).await?;
            assert_method(&read_test_frame(&mut peer).await?, CLASS_CHANNEL, 41)?;

            let mut close = Writer::method(CLASS_CONNECTION, 50);
            close.u16(200);
            close.short_string("done")?;
            close.u16(0);
            close.u16(0);
            send_test_method(&mut peer, 0, &close).await?;
            assert_method(&read_test_frame(&mut peer).await?, CLASS_CONNECTION, 51)?;
            shutdown.cancel();
            server
                .await
                .map_err(|error| Error::internal(format!("AMQP test server failed: {error}")))??;
            Ok(())
        })
        .await
        .map_err(|_| Error::new(ErrorCode::DeadlineExceeded, "AMQP method matrix timed out"))?
    }

    #[tokio::test]
    async fn negotiated_heartbeat_closes_an_idle_peer() -> Result<()> {
        tokio::time::timeout(Duration::from_secs(5), async {
            let project = ProjectId::random();
            let coordinator: Arc<dyn BrokerCoordinator> = Arc::new(TestCoordinator::new()?);
            let (mut peer, shutdown, server) = open_test_peer(project, coordinator, 1).await?;
            loop {
                let frame = read_test_frame(&mut peer).await?;
                if frame.kind == FRAME_HEARTBEAT {
                    continue;
                }
                assert_method(&frame, CLASS_CONNECTION, 50)?;
                break;
            }
            shutdown.cancel();
            server
                .await
                .map_err(|error| Error::internal(format!("AMQP test server failed: {error}")))??;
            Ok(())
        })
        .await
        .map_err(|_| Error::new(ErrorCode::DeadlineExceeded, "AMQP heartbeat test timed out"))?
    }

    #[tokio::test]
    #[allow(clippy::too_many_lines)]
    async fn authenticated_transport_negotiates_and_moves_a_confirmed_message() -> Result<()> {
        tokio::time::timeout(Duration::from_secs(5), async {
            let project = ProjectId::random();
            let coordinator: Arc<dyn BrokerCoordinator> = Arc::new(TestCoordinator::new()?);
            let shutdown = tokio_util::sync::CancellationToken::new();
            let (server_io, mut client_io) = tokio::io::duplex(1024 * 1024);
            let server_shutdown = shutdown.child_token();
            let server = tokio::spawn(QueueServer::serve_authenticated_transport(
                server_io,
                project,
                coordinator,
                server_shutdown,
            ));

            client_io.write_all(PROTOCOL_HEADER).await?;
            assert_method(
                &read_test_frame(&mut client_io).await?,
                CLASS_CONNECTION,
                10,
            )?;

            let mut start_ok = Writer::method(CLASS_CONNECTION, 11);
            start_ok.field_table(&BTreeMap::new())?;
            start_ok.short_string("EXTERNAL")?;
            start_ok.long_bytes(&[])?;
            start_ok.short_string("en_US")?;
            write_frame(
                &mut client_io,
                FRAME_METHOD,
                0,
                start_ok.as_slice(),
                SERVER_FRAME_MAX,
            )
            .await?;
            assert_method(
                &read_test_frame(&mut client_io).await?,
                CLASS_CONNECTION,
                30,
            )?;

            let mut tune_ok = Writer::method(CLASS_CONNECTION, 31);
            tune_ok.u16(32);
            tune_ok.u32(SERVER_FRAME_MAX);
            tune_ok.u16(0);
            write_frame(
                &mut client_io,
                FRAME_METHOD,
                0,
                tune_ok.as_slice(),
                SERVER_FRAME_MAX,
            )
            .await?;
            let mut open = Writer::method(CLASS_CONNECTION, 40);
            open.short_string("/")?;
            open.short_string("")?;
            open.u8(0);
            write_frame(
                &mut client_io,
                FRAME_METHOD,
                0,
                open.as_slice(),
                SERVER_FRAME_MAX,
            )
            .await?;
            assert_method(
                &read_test_frame(&mut client_io).await?,
                CLASS_CONNECTION,
                41,
            )?;

            let mut channel_open = Writer::method(CLASS_CHANNEL, 10);
            channel_open.short_string("")?;
            write_frame(
                &mut client_io,
                FRAME_METHOD,
                1,
                channel_open.as_slice(),
                SERVER_FRAME_MAX,
            )
            .await?;
            assert_method(&read_test_frame(&mut client_io).await?, CLASS_CHANNEL, 11)?;

            let mut exchange = Writer::method(CLASS_EXCHANGE, 10);
            exchange.u16(0);
            exchange.short_string("events")?;
            exchange.short_string("fanout")?;
            exchange.u8(0x02);
            exchange.field_table(&BTreeMap::new())?;
            write_frame(
                &mut client_io,
                FRAME_METHOD,
                1,
                exchange.as_slice(),
                SERVER_FRAME_MAX,
            )
            .await?;
            assert_method(&read_test_frame(&mut client_io).await?, CLASS_EXCHANGE, 11)?;

            let mut queue = Writer::method(CLASS_QUEUE, 10);
            queue.u16(0);
            queue.short_string("work")?;
            queue.u8(0x02);
            queue.field_table(&BTreeMap::new())?;
            write_frame(
                &mut client_io,
                FRAME_METHOD,
                1,
                queue.as_slice(),
                SERVER_FRAME_MAX,
            )
            .await?;
            assert_method(&read_test_frame(&mut client_io).await?, CLASS_QUEUE, 11)?;

            let mut bind = Writer::method(CLASS_QUEUE, 20);
            bind.u16(0);
            bind.short_string("work")?;
            bind.short_string("events")?;
            bind.short_string("")?;
            bind.u8(0);
            bind.field_table(&BTreeMap::new())?;
            write_frame(
                &mut client_io,
                FRAME_METHOD,
                1,
                bind.as_slice(),
                SERVER_FRAME_MAX,
            )
            .await?;
            assert_method(&read_test_frame(&mut client_io).await?, CLASS_QUEUE, 21)?;

            let mut confirm = Writer::method(CLASS_CONFIRM, 10);
            confirm.u8(0);
            write_frame(
                &mut client_io,
                FRAME_METHOD,
                1,
                confirm.as_slice(),
                SERVER_FRAME_MAX,
            )
            .await?;
            assert_method(&read_test_frame(&mut client_io).await?, CLASS_CONFIRM, 11)?;

            let mut publish = Writer::method(CLASS_BASIC, 40);
            publish.u16(0);
            publish.short_string("events")?;
            publish.short_string("")?;
            publish.u8(0);
            write_frame(
                &mut client_io,
                FRAME_METHOD,
                1,
                publish.as_slice(),
                SERVER_FRAME_MAX,
            )
            .await?;
            let body = b"durable message";
            let mut header = Writer::new();
            header.u16(CLASS_BASIC);
            header.u16(0);
            header.u64(body.len() as u64);
            BasicProperties {
                delivery_mode: Some(2),
                ..BasicProperties::default()
            }
            .encode(&mut header)?;
            write_frame(
                &mut client_io,
                FRAME_HEADER,
                1,
                header.as_slice(),
                SERVER_FRAME_MAX,
            )
            .await?;
            write_frame(&mut client_io, FRAME_BODY, 1, body, SERVER_FRAME_MAX).await?;
            assert_method(&read_test_frame(&mut client_io).await?, CLASS_BASIC, 80)?;

            let mut get = Writer::method(CLASS_BASIC, 70);
            get.u16(0);
            get.short_string("work")?;
            get.u8(1);
            write_frame(
                &mut client_io,
                FRAME_METHOD,
                1,
                get.as_slice(),
                SERVER_FRAME_MAX,
            )
            .await?;
            assert_method(&read_test_frame(&mut client_io).await?, CLASS_BASIC, 71)?;
            let content_header = read_test_frame(&mut client_io).await?;
            assert_eq!(content_header.kind, FRAME_HEADER);
            let content_body = read_test_frame(&mut client_io).await?;
            assert_eq!(content_body.kind, FRAME_BODY);
            assert_eq!(content_body.payload.as_ref(), body);

            let mut close = Writer::method(CLASS_CONNECTION, 50);
            close.u16(200);
            close.short_string("bye")?;
            close.u16(0);
            close.u16(0);
            write_frame(
                &mut client_io,
                FRAME_METHOD,
                0,
                close.as_slice(),
                SERVER_FRAME_MAX,
            )
            .await?;
            assert_method(
                &read_test_frame(&mut client_io).await?,
                CLASS_CONNECTION,
                51,
            )?;
            shutdown.cancel();
            server
                .await
                .map_err(|error| Error::internal(format!("AMQP test server failed: {error}")))??;
            Ok(())
        })
        .await
        .map_err(|_| {
            Error::new(
                ErrorCode::DeadlineExceeded,
                "AMQP integration test timed out",
            )
        })?
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "manual gate: requires pika and loopback socket permission"]
    async fn pika_external_client_confirms_requeues_and_acknowledges() -> Result<()> {
        tokio::time::timeout(Duration::from_secs(30), async {
            let project = ProjectId::random();
            let coordinator: Arc<dyn BrokerCoordinator> = Arc::new(TestCoordinator::new()?);
            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
            let address = listener.local_addr()?;
            let shutdown = tokio_util::sync::CancellationToken::new();
            let server_shutdown = shutdown.clone();
            let server = tokio::spawn(async move {
                loop {
                    tokio::select! {
                        () = server_shutdown.cancelled() => return Ok::<_, Error>(()),
                        accepted = listener.accept() => {
                            let (stream, _) = accepted?;
                            let coordinator = Arc::clone(&coordinator);
                            let connection_shutdown = server_shutdown.child_token();
                            tokio::spawn(async move {
                                if let Err(error) = serve(
                                    stream,
                                    project,
                                    coordinator,
                                    connection_shutdown,
                                    TransportTrust::Loopback,
                                )
                                .await
                                {
                                    eprintln!("AMQP interoperability peer failed: {error}");
                                }
                            });
                        }
                    }
                }
            });

            let endpoint = address.to_string();
            let python_path = std::env::var("IG_PIKA_PYTHONPATH")
                .unwrap_or_else(|_| "/private/tmp/irongraph-python-clients".to_owned());
            let driver = tokio::task::spawn_blocking(move || {
                Command::new("python3")
                    .env("PYTHONPATH", python_path)
                    .env("IG_AMQP_ENDPOINT", endpoint)
                    .arg("-c")
                    .arg(
                        r#"
import os
import pika

host, port = os.environ["IG_AMQP_ENDPOINT"].rsplit(":", 1)
connection = pika.BlockingConnection(pika.ConnectionParameters(
    host=host,
    port=int(port),
    virtual_host="/",
    credentials=pika.credentials.ExternalCredentials(),
    heartbeat=0,
    socket_timeout=10,
    blocked_connection_timeout=10,
))
channel = connection.channel()
channel.exchange_declare(exchange="events", exchange_type="direct", durable=True)
channel.queue_declare(queue="work", durable=True)
channel.queue_bind(queue="work", exchange="events", routing_key="work")
channel.confirm_delivery()
channel.basic_publish(
    exchange="events",
    routing_key="work",
    body=b"external-client",
    mandatory=True,
    properties=pika.BasicProperties(
        content_type="application/octet-stream",
        headers={"source": "pika"},
        delivery_mode=2,
    ),
)
method, properties, body = channel.basic_get(queue="work", auto_ack=False)
assert body == b"external-client"
assert properties.headers == {"source": "pika"}
channel.basic_nack(delivery_tag=method.delivery_tag, requeue=True)
method, properties, body = channel.basic_get(queue="work", auto_ack=False)
assert method.redelivered is True
assert body == b"external-client"
channel.basic_ack(delivery_tag=method.delivery_tag)
assert channel.basic_get(queue="work", auto_ack=True) == (None, None, None)
channel.close()
connection.close()
print(f"pika={pika.__version__}")
"#,
                    )
                    .output()
            })
            .await
            .map_err(|error| Error::internal(format!("Pika task failed: {error}")))??;
            shutdown.cancel();
            server
                .await
                .map_err(|error| Error::internal(format!("AMQP test server failed: {error}")))??;
            if !driver.status.success() {
                return Err(Error::new(
                    ErrorCode::ProtocolViolation,
                    format!(
                        "Pika interoperability failed\nstdout:\n{}\nstderr:\n{}",
                        String::from_utf8_lossy(&driver.stdout),
                        String::from_utf8_lossy(&driver.stderr),
                    ),
                ));
            }
            eprintln!("{}", String::from_utf8_lossy(&driver.stdout));
            Ok(())
        })
        .await
        .map_err(|_| {
            Error::new(
                ErrorCode::DeadlineExceeded,
                "Pika interoperability test timed out",
            )
        })?
    }

    async fn read_test_frame<T: AsyncRead + Unpin>(reader: &mut T) -> Result<Frame> {
        let kind = reader.read_u8().await?;
        let channel = reader.read_u16().await?;
        let size = reader.read_u32().await? as usize;
        if size > SERVER_FRAME_MAX as usize {
            return Err(Error::new(
                ErrorCode::ProtocolViolation,
                "test frame is oversized",
            ));
        }
        let mut payload = vec![0; size];
        reader.read_exact(&mut payload).await?;
        if reader.read_u8().await? != FRAME_END {
            return Err(Error::new(
                ErrorCode::ProtocolViolation,
                "test frame end is invalid",
            ));
        }
        Ok(Frame {
            kind,
            channel,
            payload: payload.into(),
        })
    }

    fn assert_method(frame: &Frame, class_id: u16, method_id: u16) -> Result<()> {
        if frame.kind != FRAME_METHOD {
            return Err(Error::new(
                ErrorCode::ProtocolViolation,
                "expected a method frame",
            ));
        }
        let mut reader = Reader::new(&frame.payload);
        let actual_class = reader.u16().map_err(protocol_error)?;
        let actual_method = reader.u16().map_err(protocol_error)?;
        if actual_class != class_id || actual_method != method_id {
            return Err(Error::new(
                ErrorCode::ProtocolViolation,
                format!(
                    "expected AMQP method {class_id}.{method_id}, got {actual_class}.{actual_method}"
                ),
            ));
        }
        Ok(())
    }
}
