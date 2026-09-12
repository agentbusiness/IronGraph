use std::{
    collections::BTreeMap,
    net::SocketAddr,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
};

use parking_lot::Mutex;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::{
    sync::{Semaphore, mpsc},
    task::{JoinHandle, JoinSet},
};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::storage::ConnectionId;
use crate::{Bookmark, CommitAcknowledgement, Error, ProjectId, Result};

use super::{
    QueryExecutor, QueryLimits, QueryRequest, QueryStreamEvent, QueryTransaction, TypedValue,
};

const BOLT_MAGIC: u32 = 0x6060_B017;
const BOLT_V5_0: u32 = 5;

/// Server agent reported in the `HELLO` response.
///
/// Drivers treat this field as the identity of the *wire dialect*, not of the product: they parse
/// it to gate optional Bolt features, and several close the connection unless it names an agent
/// they recognise. It therefore states the Bolt dialect implemented here — 5.0, the only version
/// this listener negotiates — rather than the product name.
const BOLT_SERVER_AGENT: &str = "Neo4j/5.26.0";

/// Signature of the `ROUTE` request introduced for the `neo4j://` scheme.
const BOLT_ROUTE_SIGNATURE: u8 = 0x66;

/// Lifetime advertised for the routing table this server returns, in seconds.
///
/// Membership here is fixed for the lifetime of a connection, so the value only controls how often
/// a driver re-asks. Five minutes keeps that traffic negligible without pinning a stale table.
const BOLT_ROUTING_TABLE_TTL_SECONDS: i64 = 300;
const MAX_MESSAGE_BYTES: usize = 16 * 1024 * 1024;
const MAX_PIPELINED_MESSAGES: usize = 64;
const MAX_PIPELINED_BYTES: usize = 32 * 1024 * 1024;
const MAX_VALUE_DEPTH: usize = 64;
const MAX_COLLECTION_ITEMS: usize = 1_000_000;
const CURSOR_EVENT_CAPACITY: usize = 2;
const MAX_CONNECTIONS: usize = 512;

/// Complete PackStream value representation used by the Bolt state machine.
#[derive(Clone, Debug, PartialEq)]
pub enum PackStreamValue {
    Null,
    Boolean(bool),
    Integer(i64),
    Float(f64),
    Bytes(Vec<u8>),
    String(String),
    List(Vec<PackStreamValue>),
    Map(BTreeMap<String, PackStreamValue>),
    Structure {
        signature: u8,
        fields: Vec<PackStreamValue>,
    },
}

/// Bolt listener. Each accepted connection has an isolated protocol state machine.
pub struct BoltServer {
    address: SocketAddr,
    executor: Arc<dyn QueryExecutor>,
}

impl BoltServer {
    #[must_use]
    pub fn new(address: SocketAddr, executor: Arc<dyn QueryExecutor>) -> Self {
        Self { address, executor }
    }

    pub async fn run(self, shutdown: tokio_util::sync::CancellationToken) -> Result<()> {
        if !self.address.ip().is_loopback() {
            return Err(Error::new(
                crate::ErrorCode::AuthenticationFailed,
                "plain Bolt listener is restricted to loopback; remote Bolt requires mTLS",
            ));
        }
        let listener = tokio::net::TcpListener::bind(self.address).await?;
        let permits = Arc::new(Semaphore::new(MAX_CONNECTIONS));
        let mut connections = JoinSet::new();
        loop {
            tokio::select! {
                _ = shutdown.cancelled() => break,
                accepted = listener.accept() => {
                    let (stream, _) = accepted?;
                    let Ok(permit) = Arc::clone(&permits).try_acquire_owned() else {
                        drop(stream);
                        continue;
                    };
                    let executor = Arc::clone(&self.executor);
                    connections.spawn(async move {
                        let _permit = permit;
                        let _ = BoltSession::new(executor).serve(stream).await;
                    });
                }
                _ = connections.join_next(), if !connections.is_empty() => {}
            }
        }
        connections.abort_all();
        while connections.join_next().await.is_some() {}
        Ok(())
    }

    /// Serves one connection after the owning TLS acceptor has authenticated it and supplied a
    /// credential-scoped query executor.
    pub async fn serve_authenticated_transport<S>(
        transport: S,
        executor: Arc<dyn QueryExecutor>,
    ) -> Result<()>
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        BoltSession::new(executor).serve(transport).await
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SessionState {
    Authentication,
    Ready,
    Streaming,
    Transaction,
    Failed,
    Closed,
}

/// Bolt 5.0 session with autocommit and explicit transaction state.
pub struct BoltSession {
    executor: Arc<dyn QueryExecutor>,
    state: SessionState,
    transaction_project: Option<ProjectId>,
    cursor: Option<BoltCursor>,
    transaction: Option<Box<dyn QueryTransaction>>,
    stream_return_state: SessionState,
    interrupt: Arc<InterruptRegistry>,
    connection_id: ConnectionId,
}

#[derive(Default)]
struct InterruptRegistry {
    active: Mutex<Option<CancellationToken>>,
    reset_pending: AtomicBool,
}

impl InterruptRegistry {
    fn activate(&self, cancellation: CancellationToken) {
        if self.reset_pending.load(Ordering::Acquire) {
            cancellation.cancel();
        }
        *self.active.lock() = Some(cancellation);
    }

    fn clear(&self) {
        self.active.lock().take();
    }

    fn signal_reset(&self) {
        self.reset_pending.store(true, Ordering::Release);
        if let Some(cancellation) = self.active.lock().as_ref() {
            cancellation.cancel();
        }
    }

    fn acknowledge_reset(&self) {
        self.reset_pending.store(false, Ordering::Release);
    }

    fn cancel_active(&self) {
        if let Some(cancellation) = self.active.lock().as_ref() {
            cancellation.cancel();
        }
    }
}

enum IncomingMessage {
    Message {
        value: PackStreamValue,
        encoded_bytes: usize,
    },
    Error(Error),
    End,
}

impl BoltSession {
    #[must_use]
    pub fn new(executor: Arc<dyn QueryExecutor>) -> Self {
        Self {
            executor,
            state: SessionState::Authentication,
            transaction_project: None,
            cursor: None,
            transaction: None,
            stream_return_state: SessionState::Ready,
            interrupt: Arc::new(InterruptRegistry::default()),
            connection_id: ConnectionId::new(),
        }
    }

    pub async fn serve<S>(&mut self, mut stream: S) -> Result<()>
    where
        S: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        if !negotiate(&mut stream).await? {
            return Ok(());
        }
        let (reader, mut writer) = tokio::io::split(stream);
        let (incoming_sender, mut incoming) = mpsc::unbounded_channel();
        let queued_messages = Arc::new(AtomicUsize::new(0));
        let queued_bytes = Arc::new(AtomicUsize::new(0));
        let reader_task = tokio::spawn(read_messages(
            reader,
            incoming_sender,
            Arc::clone(&self.interrupt),
            Arc::clone(&queued_messages),
            Arc::clone(&queued_bytes),
        ));
        while self.state != SessionState::Closed {
            let value = match incoming.recv().await {
                Some(IncomingMessage::Message {
                    value,
                    encoded_bytes,
                }) => {
                    queued_messages.fetch_sub(1, Ordering::AcqRel);
                    queued_bytes.fetch_sub(encoded_bytes, Ordering::AcqRel);
                    value
                }
                Some(IncomingMessage::End) | None => {
                    self.goodbye(Vec::new()).await?;
                    break;
                }
                Some(IncomingMessage::Error(error)) => {
                    let _ = write_failure(&mut writer, &error).await;
                    let _ = self.goodbye(Vec::new()).await;
                    reader_task.abort();
                    return Err(error);
                }
            };
            let state_before = self.state;
            if let Err(error) = self.dispatch(value, &mut writer).await {
                if state_before == SessionState::Authentication {
                    write_failure(&mut writer, &error).await?;
                    let _ = self.goodbye(Vec::new()).await;
                    reader_task.abort();
                    return Err(error);
                }
                self.state = SessionState::Failed;
                write_failure(&mut writer, &error).await?;
            }
        }
        self.interrupt.cancel_active();
        reader_task.abort();
        let _ = reader_task.await;
        Ok(())
    }

    async fn dispatch<S>(&mut self, value: PackStreamValue, stream: &mut S) -> Result<()>
    where
        S: AsyncWrite + Unpin,
    {
        let PackStreamValue::Structure { signature, fields } = value else {
            return Err(Error::new(
                crate::ErrorCode::ProtocolViolation,
                "Bolt message is not a structure",
            ));
        };
        if self.state == SessionState::Failed && signature != 0x0F && signature != 0x02 {
            return write_ignored(stream).await;
        }
        match signature {
            0x01 => self.hello(fields, stream).await,
            0x02 => self.goodbye(fields).await,
            0x10 => self.run_query(fields, stream).await,
            0x3F => self.pull(fields, stream).await,
            0x2F => self.discard(fields, stream).await,
            0x11 => self.begin(fields, stream).await,
            0x12 => self.commit(fields, stream).await,
            0x13 => self.rollback(fields, stream).await,
            0x0F => self.reset(fields, stream).await,
            BOLT_ROUTE_SIGNATURE => self.route(fields, stream).await,
            _ => Err(Error::new(
                crate::ErrorCode::ProtocolViolation,
                format!("unsupported Bolt message signature 0x{signature:02x}"),
            )),
        }
    }

    async fn hello<S>(&mut self, fields: Vec<PackStreamValue>, stream: &mut S) -> Result<()>
    where
        S: AsyncWrite + Unpin,
    {
        if self.state != SessionState::Authentication || fields.len() != 1 {
            return Err(protocol_state("HELLO", self.state));
        }
        let metadata = as_map(&fields[0])?;
        reject_unknown_keys(
            metadata,
            &["user_agent", "scheme", "routing", "patch_bolt"],
            "HELLO",
        )?;
        let user_agent = metadata
            .get("user_agent")
            .and_then(as_string_ref)
            .ok_or_else(|| {
                Error::new(
                    crate::ErrorCode::ProtocolViolation,
                    "HELLO requires user_agent",
                )
            })?;
        if user_agent.len() > 1_024 {
            return Err(Error::new(
                crate::ErrorCode::ProtocolViolation,
                "user_agent is oversized",
            ));
        }
        if let Some(value) = metadata.get("scheme") {
            let Some(scheme) = as_string_ref(value) else {
                return Err(Error::new(
                    crate::ErrorCode::ProtocolViolation,
                    "Bolt HELLO scheme must be a string",
                ));
            };
            if scheme != "none" {
                return Err(Error::new(
                    crate::ErrorCode::AuthenticationFailed,
                    "Bolt authentication is transport-bound; use the none scheme",
                ));
            }
        }
        if metadata.contains_key("principal") || metadata.contains_key("credentials") {
            return Err(Error::new(
                crate::ErrorCode::AuthenticationFailed,
                "Bolt HELLO must not carry reusable credentials",
            ));
        }
        // A `neo4j://` URI always sends a routing context, and refusing it here failed the
        // handshake before the driver could ask for a routing table at all — so every driver using
        // that scheme, which is the documented default in their own quickstarts, could not connect.
        // The context is accepted and answered by `ROUTE` below with a single-server table.
        if let Some(routing) = metadata.get("routing")
            && !matches!(routing, PackStreamValue::Null)
        {
            let PackStreamValue::Map(_) = routing else {
                return Err(Error::new(
                    crate::ErrorCode::ProtocolViolation,
                    "Bolt HELLO routing context must be a map",
                ));
            };
        }
        self.state = SessionState::Ready;
        let mut success = BTreeMap::new();
        // The `server` field is a compatibility contract, not a product name: drivers parse it to
        // decide which Bolt features to enable, and several refuse the connection outright unless
        // it names an agent they recognise. Advertising the product here meant no official driver
        // would talk to this endpoint at all. The agent describes the wire dialect implemented; the
        // product identifies itself through its own surfaces.
        success.insert(
            "server".to_owned(),
            PackStreamValue::String(BOLT_SERVER_AGENT.to_owned()),
        );
        success.insert(
            "connection_id".to_owned(),
            PackStreamValue::String(Uuid::new_v4().to_string()),
        );
        write_success(stream, success).await
    }

    /// Answers `ROUTE` with the single-server routing table this deployment presents.
    ///
    /// A driver opened with the `neo4j://` scheme will not run a query until it has a routing
    /// table, so without this message the scheme fails after a successful handshake. Every node
    /// here holds the complete graph and any node accepts a write, so the table names one server —
    /// the address the client already reached — in all three roles. The address is taken from the
    /// routing context the driver supplies rather than guessed, because this session does not know
    /// which of the listener's addresses the client used.
    async fn route<S>(&mut self, fields: Vec<PackStreamValue>, stream: &mut S) -> Result<()>
    where
        S: AsyncWrite + Unpin,
    {
        if !matches!(self.state, SessionState::Ready) || fields.len() != 3 {
            return Err(protocol_state("ROUTE", self.state));
        }
        let routing = as_map(&fields[0])?;
        let database = match fields.get(2).map(as_map).transpose()?.and_then(|extra| {
            extra
                .get("db")
                .and_then(as_string_ref)
                .map(std::borrow::ToOwned::to_owned)
        }) {
            Some(database) => PackStreamValue::String(database),
            None => PackStreamValue::Null,
        };
        let Some(address) = routing.get("address").and_then(as_string_ref) else {
            return Err(Error::new(
                crate::ErrorCode::ProtocolViolation,
                "Bolt ROUTE requires an address in the routing context",
            ));
        };
        let addresses = PackStreamValue::List(vec![PackStreamValue::String(address.to_owned())]);
        let server = |role: &str| {
            PackStreamValue::Map(BTreeMap::from([
                ("addresses".to_owned(), addresses.clone()),
                ("role".to_owned(), PackStreamValue::String(role.to_owned())),
            ]))
        };
        let table = PackStreamValue::Map(BTreeMap::from([
            (
                "ttl".to_owned(),
                PackStreamValue::Integer(BOLT_ROUTING_TABLE_TTL_SECONDS),
            ),
            ("db".to_owned(), database),
            (
                "servers".to_owned(),
                PackStreamValue::List(vec![server("WRITE"), server("READ"), server("ROUTE")]),
            ),
        ]));
        write_success(stream, BTreeMap::from([("rt".to_owned(), table)])).await
    }

    async fn run_query<S>(&mut self, fields: Vec<PackStreamValue>, stream: &mut S) -> Result<()>
    where
        S: AsyncWrite + Unpin,
    {
        if !matches!(self.state, SessionState::Ready | SessionState::Transaction)
            || fields.len() != 3
        {
            return Err(protocol_state("RUN", self.state));
        }
        let query = as_string_ref(&fields[0])
            .ok_or_else(|| {
                Error::new(
                    crate::ErrorCode::ProtocolViolation,
                    "RUN query must be a string",
                )
            })?
            .to_owned();
        let parameters = pack_map_to_json(as_map(&fields[1])?)?;
        let extra = as_map(&fields[2])?;
        reject_unknown_keys(
            extra,
            &[
                "bookmarks",
                "db",
                "mode",
                "tx_timeout",
                "tx_metadata",
                "imp_user",
                "notifications",
            ],
            "RUN",
        )?;
        let supplied_project = project_from_extra(self.executor.as_ref(), extra)?;
        let project = if self.state == SessionState::Transaction {
            if supplied_project
                .zip(self.transaction_project)
                .is_some_and(|(requested, transaction)| requested != transaction)
            {
                return Err(Error::new(
                    crate::ErrorCode::AuthorizationDenied,
                    "Bolt RUN cannot switch the project of an explicit transaction",
                ));
            }
            supplied_project.or(self.transaction_project)
        } else {
            supplied_project
        };
        let bookmark = bookmark_from_extra(extra)?;
        if self.state == SessionState::Transaction && bookmark.is_some() {
            return Err(Error::new(
                crate::ErrorCode::ProtocolViolation,
                "bookmarks belong on BEGIN, not RUN inside an explicit transaction",
            ));
        }
        let cancellation = CancellationToken::new();
        let request = QueryRequest {
            request_id: Uuid::new_v4(),
            project_id: project,
            query,
            parameters,
            consistency: CommitAcknowledgement::Published,
            bookmark,
            limits: QueryLimits::default(),
            cancellation,
            deadline: Some(std::time::Instant::now() + std::time::Duration::from_secs(120)),
            connection_id: self.connection_id,
        };
        request.validate()?;
        let mut cursor = if self.state == SessionState::Transaction {
            let transaction = self.transaction.take().ok_or_else(|| {
                Error::internal("Bolt transaction state has no query transaction")
            })?;
            self.stream_return_state = SessionState::Transaction;
            BoltCursor::transaction(transaction, request)
        } else {
            self.stream_return_state = SessionState::Ready;
            BoltCursor::autocommit(Arc::clone(&self.executor), request)
        };
        self.interrupt.activate(cursor.cancellation.clone());
        let result = cursor.await_schema().await;
        let fields = match result {
            Ok(fields) => fields,
            Err(error) => {
                let completion = cursor.finish(true).await?;
                self.interrupt.clear();
                self.transaction = completion.transaction;
                return Err(completion.result.err().unwrap_or(error));
            }
        };
        self.cursor = Some(cursor);
        self.state = SessionState::Streaming;
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "fields".to_owned(),
            PackStreamValue::List(fields.into_iter().map(PackStreamValue::String).collect()),
        );
        metadata.insert("t_first".to_owned(), PackStreamValue::Integer(0));
        if self.stream_return_state == SessionState::Transaction {
            metadata.insert("qid".to_owned(), PackStreamValue::Integer(0));
        }
        write_success(stream, metadata).await
    }

    async fn pull<S>(&mut self, fields: Vec<PackStreamValue>, stream: &mut S) -> Result<()>
    where
        S: AsyncWrite + Unpin,
    {
        if self.state != SessionState::Streaming || fields.len() != 1 {
            return Err(protocol_state("PULL", self.state));
        }
        let metadata = as_map(&fields[0])?;
        reject_unknown_keys(metadata, &["n", "qid"], "PULL")?;
        let count = fetch_count(
            metadata,
            "PULL",
            self.stream_return_state == SessionState::Transaction,
        )?;
        let mut cursor = self
            .cursor
            .take()
            .ok_or_else(|| Error::internal("Bolt streaming state has no cursor"))?;
        let mut exhausted = false;
        for _ in 0..count {
            match cursor.next_record().await {
                Ok(Some(record)) => {
                    write_message(
                        stream,
                        &PackStreamValue::Structure {
                            signature: 0x71,
                            fields: vec![PackStreamValue::List(record)],
                        },
                    )
                    .await?;
                }
                Ok(None) => {
                    exhausted = true;
                    break;
                }
                Err(error) => {
                    let completion = cursor.finish(true).await?;
                    self.interrupt.clear();
                    self.transaction = completion.transaction;
                    return Err(error);
                }
            }
        }
        if !exhausted && count != usize::MAX {
            match cursor.has_more().await {
                Ok(has_more) => exhausted = !has_more,
                Err(error) => {
                    let completion = cursor.finish(true).await?;
                    self.interrupt.clear();
                    self.transaction = completion.transaction;
                    return Err(error);
                }
            }
        }
        if exhausted {
            let completion = cursor.finish(false).await?;
            self.interrupt.clear();
            self.transaction = completion.transaction;
            completion.result?;
            self.state = self.stream_return_state;
            let mut summary = completion.summary.ok_or_else(|| {
                Error::new(
                    crate::ErrorCode::ProtocolViolation,
                    "query cursor closed without a terminal summary",
                )
            })?;
            summary.insert("has_more".to_owned(), PackStreamValue::Boolean(false));
            write_success(stream, summary).await
        } else {
            self.cursor = Some(cursor);
            write_success(
                stream,
                BTreeMap::from([("has_more".to_owned(), PackStreamValue::Boolean(true))]),
            )
            .await
        }
    }

    async fn discard<S>(&mut self, fields: Vec<PackStreamValue>, stream: &mut S) -> Result<()>
    where
        S: AsyncWrite + Unpin,
    {
        if self.state != SessionState::Streaming || fields.len() != 1 {
            return Err(protocol_state("DISCARD", self.state));
        }
        let metadata = as_map(&fields[0])?;
        reject_unknown_keys(metadata, &["n", "qid"], "DISCARD")?;
        let count = fetch_count(
            metadata,
            "DISCARD",
            self.stream_return_state == SessionState::Transaction,
        )?;
        let mut cursor = self
            .cursor
            .take()
            .ok_or_else(|| Error::internal("Bolt streaming state has no cursor"))?;
        let mut exhausted = false;
        for _ in 0..count {
            match cursor.next_record().await {
                Ok(Some(_)) => {}
                Ok(None) => {
                    exhausted = true;
                    break;
                }
                Err(error) => {
                    let completion = cursor.finish(true).await?;
                    self.interrupt.clear();
                    self.transaction = completion.transaction;
                    return Err(error);
                }
            }
        }
        if !exhausted && count != usize::MAX {
            match cursor.has_more().await {
                Ok(has_more) => exhausted = !has_more,
                Err(error) => {
                    let completion = cursor.finish(true).await?;
                    self.interrupt.clear();
                    self.transaction = completion.transaction;
                    return Err(error);
                }
            }
        }
        if exhausted {
            let completion = cursor.finish(false).await?;
            self.interrupt.clear();
            self.transaction = completion.transaction;
            completion.result?;
            self.state = self.stream_return_state;
            let mut summary = completion.summary.ok_or_else(|| {
                Error::new(
                    crate::ErrorCode::ProtocolViolation,
                    "query cursor closed without a terminal summary",
                )
            })?;
            summary.insert("has_more".to_owned(), PackStreamValue::Boolean(false));
            write_success(stream, summary).await
        } else {
            self.cursor = Some(cursor);
            write_success(
                stream,
                BTreeMap::from([("has_more".to_owned(), PackStreamValue::Boolean(true))]),
            )
            .await
        }
    }

    async fn begin<S>(&mut self, fields: Vec<PackStreamValue>, stream: &mut S) -> Result<()>
    where
        S: AsyncWrite + Unpin,
    {
        if self.state != SessionState::Ready || fields.len() != 1 {
            return Err(protocol_state("BEGIN", self.state));
        }
        let extra = as_map(&fields[0])?;
        reject_unknown_keys(
            extra,
            &[
                "bookmarks",
                "db",
                "mode",
                "tx_timeout",
                "tx_metadata",
                "imp_user",
                "notifications",
            ],
            "BEGIN",
        )?;
        let project = project_from_extra(self.executor.as_ref(), extra)?;
        let bookmark = bookmark_from_extra(extra)?;
        self.transaction = Some(self.executor.begin_on_connection(
            self.connection_id,
            project,
            bookmark,
            CommitAcknowledgement::Published,
        )?);
        self.transaction_project = project;
        self.state = SessionState::Transaction;
        write_success(stream, BTreeMap::new()).await
    }

    async fn commit<S>(&mut self, fields: Vec<PackStreamValue>, stream: &mut S) -> Result<()>
    where
        S: AsyncWrite + Unpin,
    {
        if self.state != SessionState::Transaction || !fields.is_empty() {
            return Err(protocol_state("COMMIT", self.state));
        }
        let transaction = self
            .transaction
            .take()
            .ok_or_else(|| Error::internal("Bolt transaction state has no query transaction"))?;
        let bookmark = tokio::task::spawn_blocking(move || transaction.commit())
            .await
            .map_err(|error| {
                Error::internal(format!("transaction worker terminated: {error}"))
            })??;
        self.transaction_project = None;
        self.state = SessionState::Ready;
        write_success(
            stream,
            BTreeMap::from([(
                "bookmark".to_owned(),
                PackStreamValue::String(format!("ig:{}:{}", bookmark.term, bookmark.index)),
            )]),
        )
        .await
    }

    async fn rollback<S>(&mut self, fields: Vec<PackStreamValue>, stream: &mut S) -> Result<()>
    where
        S: AsyncWrite + Unpin,
    {
        if self.state != SessionState::Transaction || !fields.is_empty() {
            return Err(protocol_state("ROLLBACK", self.state));
        }
        let transaction = self
            .transaction
            .take()
            .ok_or_else(|| Error::internal("Bolt transaction state has no query transaction"))?;
        tokio::task::spawn_blocking(move || transaction.rollback())
            .await
            .map_err(|error| {
                Error::internal(format!("transaction worker terminated: {error}"))
            })??;
        self.transaction_project = None;
        self.state = SessionState::Ready;
        write_success(stream, BTreeMap::new()).await
    }

    async fn reset<S>(&mut self, fields: Vec<PackStreamValue>, stream: &mut S) -> Result<()>
    where
        S: AsyncWrite + Unpin,
    {
        if !fields.is_empty()
            || matches!(
                self.state,
                SessionState::Authentication | SessionState::Closed
            )
        {
            return Err(Error::new(
                crate::ErrorCode::ProtocolViolation,
                "RESET is unavailable before HELLO and takes no fields",
            ));
        }
        if let Some(cursor) = self.cursor.take() {
            let completion = cursor.finish(true).await?;
            self.interrupt.clear();
            self.transaction = completion.transaction;
        }
        if let Some(transaction) = self.transaction.take() {
            tokio::task::spawn_blocking(move || transaction.rollback())
                .await
                .map_err(|error| {
                    Error::internal(format!("transaction worker terminated: {error}"))
                })??;
        }
        self.transaction_project = None;
        self.state = SessionState::Ready;
        self.interrupt.acknowledge_reset();
        write_success(stream, BTreeMap::new()).await
    }

    async fn goodbye(&mut self, fields: Vec<PackStreamValue>) -> Result<()> {
        if !fields.is_empty() {
            return Err(Error::new(
                crate::ErrorCode::ProtocolViolation,
                "GOODBYE takes no fields",
            ));
        }
        if let Some(cursor) = self.cursor.take() {
            let completion = cursor.finish(true).await?;
            self.interrupt.clear();
            self.transaction = completion.transaction;
        }
        if let Some(transaction) = self.transaction.take() {
            tokio::task::spawn_blocking(move || transaction.rollback())
                .await
                .map_err(|error| {
                    Error::internal(format!("transaction worker terminated: {error}"))
                })??;
        }
        self.transaction_project = None;
        self.interrupt.acknowledge_reset();
        self.state = SessionState::Closed;
        Ok(())
    }
}

struct CursorBatch {
    row_count: usize,
    next_row: usize,
    columns: Vec<super::BatchColumn>,
}

struct WorkerCompletion {
    transaction: Option<Box<dyn QueryTransaction>>,
    result: Result<()>,
}

struct CursorCompletion {
    transaction: Option<Box<dyn QueryTransaction>>,
    result: Result<()>,
    summary: Option<BTreeMap<String, PackStreamValue>>,
}

struct BoltCursor {
    request_id: Uuid,
    receiver: mpsc::Receiver<QueryStreamEvent>,
    worker: Option<JoinHandle<WorkerCompletion>>,
    cancellation: CancellationToken,
    batch: Option<CursorBatch>,
    pending_record: Option<Vec<PackStreamValue>>,
    field_count: Option<usize>,
    expected_sequence: u64,
    summary: Option<BTreeMap<String, PackStreamValue>>,
    terminal: bool,
}

impl BoltCursor {
    fn autocommit(executor: Arc<dyn QueryExecutor>, request: QueryRequest) -> Self {
        let request_id = request.request_id;
        let cancellation = request.cancellation.clone();
        let (sender, receiver) = mpsc::channel(CURSOR_EVENT_CAPACITY);
        let worker = tokio::task::spawn_blocking(move || {
            let result = executor.execute(request, &mut |event| {
                sender.blocking_send(event).map_err(|_| {
                    Error::new(crate::ErrorCode::Cancelled, "Bolt result cursor was closed")
                })
            });
            WorkerCompletion {
                transaction: None,
                result,
            }
        });
        Self::new(request_id, receiver, worker, cancellation)
    }

    fn transaction(mut transaction: Box<dyn QueryTransaction>, request: QueryRequest) -> Self {
        let request_id = request.request_id;
        let cancellation = request.cancellation.clone();
        let (sender, receiver) = mpsc::channel(CURSOR_EVENT_CAPACITY);
        let worker = tokio::task::spawn_blocking(move || {
            let result = transaction.run(request, &mut |event| {
                sender.blocking_send(event).map_err(|_| {
                    Error::new(crate::ErrorCode::Cancelled, "Bolt result cursor was closed")
                })
            });
            WorkerCompletion {
                transaction: Some(transaction),
                result,
            }
        });
        Self::new(request_id, receiver, worker, cancellation)
    }

    fn new(
        request_id: Uuid,
        receiver: mpsc::Receiver<QueryStreamEvent>,
        worker: JoinHandle<WorkerCompletion>,
        cancellation: CancellationToken,
    ) -> Self {
        Self {
            request_id,
            receiver,
            worker: Some(worker),
            cancellation,
            batch: None,
            pending_record: None,
            field_count: None,
            expected_sequence: 0,
            summary: None,
            terminal: false,
        }
    }

    async fn await_schema(&mut self) -> Result<Vec<String>> {
        loop {
            let event = self.receiver.recv().await.ok_or_else(|| {
                Error::new(
                    crate::ErrorCode::ProtocolViolation,
                    "query cursor closed before schema",
                )
            })?;
            match event {
                QueryStreamEvent::Catalog { .. } => {}
                QueryStreamEvent::Schema {
                    request_id,
                    columns,
                } => {
                    self.verify_request(request_id)?;
                    if self.field_count.is_some() {
                        return Err(Error::new(
                            crate::ErrorCode::ProtocolViolation,
                            "query cursor emitted duplicate schema",
                        ));
                    }
                    self.field_count = Some(columns.len());
                    return Ok(columns.into_iter().map(|column| column.name).collect());
                }
                QueryStreamEvent::Error {
                    request_id,
                    code,
                    message,
                    retryable,
                    retry_after_ms,
                } => {
                    self.verify_request(request_id)?;
                    return Err(Error {
                        code,
                        message: message.into(),
                        retryable,
                        retry_after_ms,
                    });
                }
                QueryStreamEvent::Batch { .. } | QueryStreamEvent::Summary { .. } => {
                    return Err(Error::new(
                        crate::ErrorCode::ProtocolViolation,
                        "query cursor emitted data before schema",
                    ));
                }
            }
        }
    }

    async fn next_record(&mut self) -> Result<Option<Vec<PackStreamValue>>> {
        if let Some(record) = self.pending_record.take() {
            return Ok(Some(record));
        }
        loop {
            if let Some(batch) = &mut self.batch {
                if batch.next_row < batch.row_count {
                    let row = batch.next_row;
                    batch.next_row += 1;
                    return batch
                        .columns
                        .iter()
                        .map(|column| {
                            let value = column.values.get(row).ok_or_else(|| {
                                Error::new(
                                    crate::ErrorCode::ProtocolViolation,
                                    "query batch column is shorter than its row count",
                                )
                            })?;
                            typed_to_pack(value)
                        })
                        .collect::<Result<Vec<_>>>()
                        .map(Some);
                }
                self.batch = None;
            }
            if self.terminal {
                return Ok(None);
            }
            let Some(event) = self.receiver.recv().await else {
                self.terminal = true;
                return Ok(None);
            };
            match event {
                QueryStreamEvent::Catalog { .. } => {}
                QueryStreamEvent::Schema { request_id, .. } => {
                    self.verify_request(request_id)?;
                    return Err(Error::new(
                        crate::ErrorCode::ProtocolViolation,
                        "query cursor emitted duplicate schema",
                    ));
                }
                QueryStreamEvent::Batch {
                    request_id,
                    sequence,
                    row_count,
                    columns,
                } => {
                    self.verify_request(request_id)?;
                    if sequence != self.expected_sequence {
                        return Err(Error::new(
                            crate::ErrorCode::ProtocolViolation,
                            "query batch sequence is discontinuous",
                        ));
                    }
                    self.expected_sequence = self.expected_sequence.saturating_add(1);
                    let row_count = usize::try_from(row_count)
                        .map_err(|_| Error::invalid_data("row count does not fit memory"))?;
                    if columns.len() != self.field_count.unwrap_or_default()
                        || columns
                            .iter()
                            .any(|column| column.values.len() != row_count)
                    {
                        return Err(Error::new(
                            crate::ErrorCode::ProtocolViolation,
                            "query batch does not match its schema or row count",
                        ));
                    }
                    self.batch = Some(CursorBatch {
                        row_count,
                        next_row: 0,
                        columns,
                    });
                }
                QueryStreamEvent::Summary {
                    request_id,
                    bookmark,
                    statistics,
                    truncated,
                    ..
                } => {
                    self.verify_request(request_id)?;
                    self.summary = Some(BTreeMap::from([
                        (
                            "bookmark".to_owned(),
                            PackStreamValue::String(format!(
                                "ig:{}:{}",
                                bookmark.term, bookmark.index
                            )),
                        ),
                        (
                            "t_last".to_owned(),
                            PackStreamValue::Integer(statistics.elapsed_ms as i64),
                        ),
                        ("truncated".to_owned(), PackStreamValue::Boolean(truncated)),
                    ]));
                    self.terminal = true;
                }
                QueryStreamEvent::Error {
                    request_id,
                    code,
                    message,
                    retryable,
                    retry_after_ms,
                } => {
                    self.verify_request(request_id)?;
                    self.terminal = true;
                    return Err(Error {
                        code,
                        message: message.into(),
                        retryable,
                        retry_after_ms,
                    });
                }
            }
        }
    }

    async fn has_more(&mut self) -> Result<bool> {
        match self.next_record().await? {
            Some(record) => {
                self.pending_record = Some(record);
                Ok(true)
            }
            None => Ok(false),
        }
    }

    fn verify_request(&self, request_id: Uuid) -> Result<()> {
        if request_id == self.request_id {
            Ok(())
        } else {
            Err(Error::new(
                crate::ErrorCode::ProtocolViolation,
                "query cursor emitted an event for another request",
            ))
        }
    }

    async fn finish(mut self, cancel: bool) -> Result<CursorCompletion> {
        if cancel {
            self.cancellation.cancel();
            self.receiver.close();
        }
        let worker = self
            .worker
            .take()
            .ok_or_else(|| Error::internal("query cursor worker is missing"))?;
        let completion = worker
            .await
            .map_err(|error| Error::internal(format!("query worker terminated: {error}")))?;
        Ok(CursorCompletion {
            transaction: completion.transaction,
            result: completion.result,
            summary: self.summary.take(),
        })
    }
}

impl Drop for BoltCursor {
    fn drop(&mut self) {
        self.cancellation.cancel();
        self.receiver.close();
        if let Some(worker) = self.worker.take() {
            worker.abort();
        }
    }
}

fn fetch_count(
    metadata: &BTreeMap<String, PackStreamValue>,
    message: &str,
    explicit_transaction: bool,
) -> Result<usize> {
    let qid = metadata.get("qid").map(as_i64).transpose()?;
    if !explicit_transaction && qid.is_some() {
        return Err(Error::new(
            crate::ErrorCode::ProtocolViolation,
            "qid is valid only for an explicit transaction result",
        ));
    }
    if qid.is_some_and(|qid| !matches!(qid, -1 | 0)) {
        return Err(Error::new(
            crate::ErrorCode::ProtocolViolation,
            "only the current Bolt result cursor may be consumed",
        ));
    }
    let maximum = metadata
        .get("n")
        .ok_or_else(|| {
            Error::new(
                crate::ErrorCode::ProtocolViolation,
                format!("{message} requires n"),
            )
        })
        .and_then(as_i64)?;
    if maximum == 0 || maximum < -1 {
        return Err(Error::new(
            crate::ErrorCode::ProtocolViolation,
            format!("{message} n must be -1 or positive"),
        ));
    }
    if maximum == -1 {
        Ok(usize::MAX)
    } else {
        usize::try_from(maximum)
            .map_err(|_| Error::invalid_data(format!("{message} n does not fit this platform")))
    }
}

async fn negotiate<S>(stream: &mut S) -> Result<bool>
where
    S: AsyncRead + AsyncWrite + Unpin,
{
    let mut handshake = [0u8; 20];
    match stream.read_exact(&mut handshake).await {
        Ok(_) => {}
        Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(false),
        Err(error) => return Err(error.into()),
    }
    if u32::from_be_bytes(
        handshake[0..4]
            .try_into()
            .map_err(|_| Error::internal("handshake slice"))?,
    ) != BOLT_MAGIC
    {
        return Err(Error::new(
            crate::ErrorCode::ProtocolViolation,
            "invalid Bolt magic",
        ));
    }
    let supported = handshake[4..]
        .chunks_exact(4)
        .map(|chunk| u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .any(proposal_includes_v5_0);
    let selected = if supported {
        BOLT_V5_0.to_be_bytes()
    } else {
        [0, 0, 0, 0]
    };
    stream.write_all(&selected).await?;
    stream.flush().await?;
    Ok(supported)
}

async fn read_messages<R>(
    mut reader: R,
    sender: mpsc::UnboundedSender<IncomingMessage>,
    interrupt: Arc<InterruptRegistry>,
    queued_messages: Arc<AtomicUsize>,
    queued_bytes: Arc<AtomicUsize>,
) where
    R: AsyncRead + Unpin,
{
    loop {
        let encoded = match read_chunked_message(&mut reader).await {
            Ok(Some(encoded)) => encoded,
            Ok(None) => {
                interrupt.cancel_active();
                let _ = sender.send(IncomingMessage::End);
                return;
            }
            Err(error) => {
                interrupt.cancel_active();
                let _ = sender.send(IncomingMessage::Error(error));
                return;
            }
        };
        let encoded_bytes = encoded.len();
        let value = match decode_packstream(&encoded) {
            Ok(value) => value,
            Err(error) => {
                interrupt.cancel_active();
                let _ = sender.send(IncomingMessage::Error(error));
                return;
            }
        };
        if matches!(
            value,
            PackStreamValue::Structure {
                signature: 0x0F,
                ..
            }
        ) {
            // RESET is a jump-ahead interrupt signal. Decoding runs independently of query
            // execution so a pipelined RESET can cancel a worker blocked before its first row.
            interrupt.signal_reset();
        }
        let message_count = queued_messages.fetch_add(1, Ordering::AcqRel) + 1;
        let byte_count = queued_bytes.fetch_add(encoded_bytes, Ordering::AcqRel) + encoded_bytes;
        if message_count > MAX_PIPELINED_MESSAGES || byte_count > MAX_PIPELINED_BYTES {
            queued_messages.fetch_sub(1, Ordering::AcqRel);
            queued_bytes.fetch_sub(encoded_bytes, Ordering::AcqRel);
            interrupt.cancel_active();
            let _ = sender.send(IncomingMessage::Error(Error::retryable(
                crate::ErrorCode::Backpressure,
                "Bolt pipeline exceeds its bounded admission capacity",
                Some(25),
            )));
            return;
        }
        if sender
            .send(IncomingMessage::Message {
                value,
                encoded_bytes,
            })
            .is_err()
        {
            interrupt.cancel_active();
            return;
        }
    }
}

fn proposal_includes_v5_0(proposal: u32) -> bool {
    let major = proposal & 0xFF;
    let minor = (proposal >> 8) & 0xFF;
    let range = (proposal >> 16) & 0xFF;
    let reserved = proposal >> 24;
    reserved == 0 && major == 5 && range <= minor && minor.saturating_sub(range) == 0
}

async fn read_chunked_message<S>(stream: &mut S) -> Result<Option<Vec<u8>>>
where
    S: AsyncRead + Unpin,
{
    let mut message = Vec::new();
    loop {
        let mut size = [0u8; 2];
        match stream.read_exact(&mut size).await {
            Ok(_) => {}
            Err(error)
                if error.kind() == std::io::ErrorKind::UnexpectedEof && message.is_empty() =>
            {
                return Ok(None);
            }
            Err(error) => return Err(error.into()),
        }
        let length = u16::from_be_bytes(size) as usize;
        if length == 0 {
            if message.is_empty() {
                continue;
            }
            return Ok(Some(message));
        }
        if message.len().saturating_add(length) > MAX_MESSAGE_BYTES {
            return Err(Error::new(
                crate::ErrorCode::ProtocolViolation,
                "Bolt message is oversized",
            ));
        }
        let start = message.len();
        message.resize(start + length, 0);
        stream.read_exact(&mut message[start..]).await?;
    }
}

async fn write_message<S>(stream: &mut S, value: &PackStreamValue) -> Result<()>
where
    S: AsyncWrite + Unpin,
{
    let mut payload = Vec::new();
    encode_packstream(value, &mut payload)?;
    for chunk in payload.chunks(u16::MAX as usize) {
        stream
            .write_all(&(chunk.len() as u16).to_be_bytes())
            .await?;
        stream.write_all(chunk).await?;
    }
    stream.write_all(&[0, 0]).await?;
    stream.flush().await?;
    Ok(())
}

async fn write_success<S>(stream: &mut S, metadata: BTreeMap<String, PackStreamValue>) -> Result<()>
where
    S: AsyncWrite + Unpin,
{
    write_message(
        stream,
        &PackStreamValue::Structure {
            signature: 0x70,
            fields: vec![PackStreamValue::Map(metadata)],
        },
    )
    .await
}

async fn write_failure<S>(stream: &mut S, error: &Error) -> Result<()>
where
    S: AsyncWrite + Unpin,
{
    let mut metadata = BTreeMap::new();
    metadata.insert(
        "code".to_owned(),
        PackStreamValue::String(bolt_error_code(error).to_owned()),
    );
    metadata.insert(
        "message".to_owned(),
        PackStreamValue::String(error.message.to_string()),
    );
    write_message(
        stream,
        &PackStreamValue::Structure {
            signature: 0x7F,
            fields: vec![PackStreamValue::Map(metadata)],
        },
    )
    .await
}

fn bolt_error_code(error: &Error) -> &'static str {
    use crate::ErrorCode;

    match error.code {
        ErrorCode::AuthenticationFailed => "Neo.ClientError.Security.Unauthorized",
        ErrorCode::AuthorizationDenied | ErrorCode::LayerNotAllowed => {
            "Neo.ClientError.Security.Forbidden"
        }
        ErrorCode::ProjectNotFound | ErrorCode::ProjectFenced => {
            "Neo.ClientError.Database.DatabaseNotFound"
        }
        ErrorCode::QuerySyntax => "Neo.ClientError.Statement.SyntaxError",
        ErrorCode::QueryType
        | ErrorCode::TemporalRange
        | ErrorCode::EmbeddingProfileMismatch
        | ErrorCode::EmbeddingProfileImmutable => "Neo.ClientError.Statement.TypeError",
        ErrorCode::InvalidData | ErrorCode::ProtocolViolation => "Neo.ClientError.Request.Invalid",
        ErrorCode::TransactionExpired => "Neo.ClientError.Transaction.TransactionTimedOut",
        ErrorCode::TransactionConflict | ErrorCode::TransactionSequencerChanged => {
            "Neo.TransientError.Transaction.DeadlockDetected"
        }
        ErrorCode::StaleLocalRead
        | ErrorCode::WriteAdmissionFull
        | ErrorCode::Backpressure
        | ErrorCode::DeadlineExceeded
        | ErrorCode::GpuAdmissionFailure
        | ErrorCode::IndexUnavailable
        | ErrorCode::EmbeddingUnavailable => "Neo.TransientError.General.DatabaseUnavailable",
        ErrorCode::Cancelled => "Neo.ClientError.Transaction.Terminated",
        ErrorCode::ResultBudgetExceeded | ErrorCode::RetentionExpired => {
            "Neo.ClientError.Statement.ExecutionFailed"
        }
        ErrorCode::CorruptStorage | ErrorCode::Io | ErrorCode::Internal => {
            "Neo.DatabaseError.General.UnknownError"
        }
    }
}

async fn write_ignored<S>(stream: &mut S) -> Result<()>
where
    S: AsyncWrite + Unpin,
{
    write_message(
        stream,
        &PackStreamValue::Structure {
            signature: 0x7E,
            fields: Vec::new(),
        },
    )
    .await
}

fn decode_packstream(input: &[u8]) -> Result<PackStreamValue> {
    let mut decoder = Decoder {
        input,
        offset: 0,
        items: 0,
    };
    let value = decoder.value(0)?;
    if decoder.offset != input.len() {
        return Err(Error::new(
            crate::ErrorCode::ProtocolViolation,
            "trailing PackStream bytes",
        ));
    }
    Ok(value)
}

struct Decoder<'a> {
    input: &'a [u8],
    offset: usize,
    items: usize,
}

impl Decoder<'_> {
    fn value(&mut self, depth: usize) -> Result<PackStreamValue> {
        if depth > MAX_VALUE_DEPTH || self.items >= MAX_COLLECTION_ITEMS {
            return Err(Error::new(
                crate::ErrorCode::ProtocolViolation,
                "PackStream nesting/item limit exceeded",
            ));
        }
        self.items += 1;
        let marker = self.byte()?;
        match marker {
            0x00..=0x7F => Ok(PackStreamValue::Integer(marker as i64)),
            0xF0..=0xFF => Ok(PackStreamValue::Integer(i8::from_be_bytes([marker]) as i64)),
            0x80..=0x8F => self.string((marker & 0x0F) as usize),
            0x90..=0x9F => self.list((marker & 0x0F) as usize, depth),
            0xA0..=0xAF => self.map((marker & 0x0F) as usize, depth),
            0xB0..=0xBF => self.structure((marker & 0x0F) as usize, depth),
            0xC0 => Ok(PackStreamValue::Null),
            0xC1 => Ok(PackStreamValue::Float(f64::from_bits(self.u64()?))),
            0xC2 => Ok(PackStreamValue::Boolean(false)),
            0xC3 => Ok(PackStreamValue::Boolean(true)),
            0xC8 => Ok(PackStreamValue::Integer(self.byte()? as i8 as i64)),
            0xC9 => Ok(PackStreamValue::Integer(self.u16()? as i16 as i64)),
            0xCA => Ok(PackStreamValue::Integer(self.u32()? as i32 as i64)),
            0xCB => Ok(PackStreamValue::Integer(self.u64()? as i64)),
            0xCC => {
                let len = self.byte()? as usize;
                self.bytes(len)
            }
            0xCD => {
                let len = self.u16()? as usize;
                self.bytes(len)
            }
            0xCE => {
                let len = self.length32()?;
                self.bytes(len)
            }
            0xD0 => {
                let len = self.byte()? as usize;
                self.string(len)
            }
            0xD1 => {
                let len = self.u16()? as usize;
                self.string(len)
            }
            0xD2 => {
                let len = self.length32()?;
                self.string(len)
            }
            0xD4 => {
                let len = self.byte()? as usize;
                self.list(len, depth)
            }
            0xD5 => {
                let len = self.u16()? as usize;
                self.list(len, depth)
            }
            0xD6 => {
                let len = self.length32()?;
                self.list(len, depth)
            }
            0xD8 => {
                let len = self.byte()? as usize;
                self.map(len, depth)
            }
            0xD9 => {
                let len = self.u16()? as usize;
                self.map(len, depth)
            }
            0xDA => {
                let len = self.length32()?;
                self.map(len, depth)
            }
            _ => Err(Error::new(
                crate::ErrorCode::ProtocolViolation,
                format!("unknown PackStream marker 0x{marker:02x}"),
            )),
        }
    }

    fn byte(&mut self) -> Result<u8> {
        let value = self.input.get(self.offset).copied().ok_or_else(|| {
            Error::new(
                crate::ErrorCode::ProtocolViolation,
                "truncated PackStream value",
            )
        })?;
        self.offset += 1;
        Ok(value)
    }

    fn take(&mut self, length: usize) -> Result<&[u8]> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or_else(|| Error::invalid_data("PackStream length overflow"))?;
        let value = self.input.get(self.offset..end).ok_or_else(|| {
            Error::new(
                crate::ErrorCode::ProtocolViolation,
                "truncated PackStream payload",
            )
        })?;
        self.offset = end;
        Ok(value)
    }

    fn u16(&mut self) -> Result<u16> {
        let bytes: [u8; 2] = self
            .take(2)?
            .try_into()
            .map_err(|_| Error::internal("u16 slice"))?;
        Ok(u16::from_be_bytes(bytes))
    }

    fn u32(&mut self) -> Result<u32> {
        let bytes: [u8; 4] = self
            .take(4)?
            .try_into()
            .map_err(|_| Error::internal("u32 slice"))?;
        Ok(u32::from_be_bytes(bytes))
    }

    fn u64(&mut self) -> Result<u64> {
        let bytes: [u8; 8] = self
            .take(8)?
            .try_into()
            .map_err(|_| Error::internal("u64 slice"))?;
        Ok(u64::from_be_bytes(bytes))
    }

    fn length32(&mut self) -> Result<usize> {
        usize::try_from(self.u32()?)
            .map_err(|_| Error::invalid_data("PackStream length does not fit usize"))
    }

    fn bytes(&mut self, length: usize) -> Result<PackStreamValue> {
        Ok(PackStreamValue::Bytes(self.take(length)?.to_vec()))
    }

    fn string(&mut self, length: usize) -> Result<PackStreamValue> {
        let text = std::str::from_utf8(self.take(length)?).map_err(|_| {
            Error::new(
                crate::ErrorCode::ProtocolViolation,
                "PackStream string is not UTF-8",
            )
        })?;
        Ok(PackStreamValue::String(text.to_owned()))
    }

    fn list(&mut self, length: usize, depth: usize) -> Result<PackStreamValue> {
        if self.items.saturating_add(length) > MAX_COLLECTION_ITEMS {
            return Err(Error::new(
                crate::ErrorCode::ProtocolViolation,
                "PackStream list is oversized",
            ));
        }
        let mut values = Vec::with_capacity(length);
        for _ in 0..length {
            values.push(self.value(depth + 1)?);
        }
        Ok(PackStreamValue::List(values))
    }

    fn map(&mut self, length: usize, depth: usize) -> Result<PackStreamValue> {
        if self.items.saturating_add(length.saturating_mul(2)) > MAX_COLLECTION_ITEMS {
            return Err(Error::new(
                crate::ErrorCode::ProtocolViolation,
                "PackStream map is oversized",
            ));
        }
        let mut values = BTreeMap::new();
        for _ in 0..length {
            let PackStreamValue::String(key) = self.value(depth + 1)? else {
                return Err(Error::new(
                    crate::ErrorCode::ProtocolViolation,
                    "PackStream map key is not a string",
                ));
            };
            if values.insert(key, self.value(depth + 1)?).is_some() {
                return Err(Error::new(
                    crate::ErrorCode::ProtocolViolation,
                    "duplicate PackStream map key",
                ));
            }
        }
        Ok(PackStreamValue::Map(values))
    }

    fn structure(&mut self, length: usize, depth: usize) -> Result<PackStreamValue> {
        let signature = self.byte()?;
        let mut fields = Vec::with_capacity(length);
        for _ in 0..length {
            fields.push(self.value(depth + 1)?);
        }
        Ok(PackStreamValue::Structure { signature, fields })
    }
}

fn encode_packstream(value: &PackStreamValue, output: &mut Vec<u8>) -> Result<()> {
    match value {
        PackStreamValue::Null => output.push(0xC0),
        PackStreamValue::Boolean(false) => output.push(0xC2),
        PackStreamValue::Boolean(true) => output.push(0xC3),
        PackStreamValue::Integer(value) if (0..=127).contains(value) => output.push(*value as u8),
        PackStreamValue::Integer(value) if (-16..=-1).contains(value) => {
            output.push(*value as i8 as u8)
        }
        PackStreamValue::Integer(value) if i8::try_from(*value).is_ok() => {
            output.extend_from_slice(&[0xC8, *value as i8 as u8]);
        }
        PackStreamValue::Integer(value) if i16::try_from(*value).is_ok() => {
            output.push(0xC9);
            output.extend_from_slice(&(*value as i16).to_be_bytes());
        }
        PackStreamValue::Integer(value) if i32::try_from(*value).is_ok() => {
            output.push(0xCA);
            output.extend_from_slice(&(*value as i32).to_be_bytes());
        }
        PackStreamValue::Integer(value) => {
            output.push(0xCB);
            output.extend_from_slice(&value.to_be_bytes());
        }
        PackStreamValue::Float(value) => {
            output.push(0xC1);
            output.extend_from_slice(&value.to_bits().to_be_bytes());
        }
        PackStreamValue::Bytes(bytes) => {
            encode_length(bytes.len(), 0xCC, 0xCD, 0xCE, output)?;
            output.extend_from_slice(bytes);
        }
        PackStreamValue::String(value) => {
            encode_tiny_or_length(value.len(), 0x80, 0xD0, 0xD1, 0xD2, output)?;
            output.extend_from_slice(value.as_bytes());
        }
        PackStreamValue::List(values) => {
            encode_tiny_or_length(values.len(), 0x90, 0xD4, 0xD5, 0xD6, output)?;
            for value in values {
                encode_packstream(value, output)?;
            }
        }
        PackStreamValue::Map(values) => {
            encode_tiny_or_length(values.len(), 0xA0, 0xD8, 0xD9, 0xDA, output)?;
            for (key, value) in values {
                encode_packstream(&PackStreamValue::String(key.clone()), output)?;
                encode_packstream(value, output)?;
            }
        }
        PackStreamValue::Structure { signature, fields } => {
            if fields.len() > 15 {
                return Err(Error::new(
                    crate::ErrorCode::ProtocolViolation,
                    "PackStream structure has more than 15 fields",
                ));
            }
            output.push(0xB0 | fields.len() as u8);
            output.push(*signature);
            for field in fields {
                encode_packstream(field, output)?;
            }
        }
    }
    Ok(())
}

fn encode_tiny_or_length(
    length: usize,
    tiny: u8,
    marker8: u8,
    marker16: u8,
    marker32: u8,
    output: &mut Vec<u8>,
) -> Result<()> {
    if length <= 15 {
        output.push(tiny | length as u8);
        Ok(())
    } else {
        encode_length(length, marker8, marker16, marker32, output)
    }
}

fn encode_length(
    length: usize,
    marker8: u8,
    marker16: u8,
    marker32: u8,
    output: &mut Vec<u8>,
) -> Result<()> {
    if let Ok(value) = u8::try_from(length) {
        output.extend_from_slice(&[marker8, value]);
    } else if let Ok(value) = u16::try_from(length) {
        output.push(marker16);
        output.extend_from_slice(&value.to_be_bytes());
    } else {
        let value = u32::try_from(length).map_err(|_| {
            Error::new(
                crate::ErrorCode::ProtocolViolation,
                "PackStream value is too large",
            )
        })?;
        output.push(marker32);
        output.extend_from_slice(&value.to_be_bytes());
    }
    Ok(())
}

fn typed_to_pack(value: &TypedValue) -> Result<PackStreamValue> {
    Ok(match value {
        TypedValue::Null => PackStreamValue::Null,
        TypedValue::Boolean(value) => PackStreamValue::Boolean(*value),
        TypedValue::Integer(value) => {
            PackStreamValue::Integer(value.parse().map_err(|_| {
                Error::internal("query result contains an invalid canonical integer")
            })?)
        }
        TypedValue::Float(value) => PackStreamValue::Float(*value),
        TypedValue::String(value) => PackStreamValue::String(value.clone()),
        TypedValue::Bytes(value) => PackStreamValue::Bytes(value.clone()),
        TypedValue::Date(days) => PackStreamValue::Structure {
            signature: 0x44,
            fields: vec![PackStreamValue::Integer(*days)],
        },
        TypedValue::Time {
            nanos,
            offset_seconds,
        } => {
            validate_result_time(*nanos, *offset_seconds)?;
            offset_seconds.map_or_else(
                || PackStreamValue::Structure {
                    signature: 0x74,
                    fields: vec![PackStreamValue::Integer(*nanos)],
                },
                |offset| PackStreamValue::Structure {
                    signature: 0x54,
                    fields: vec![
                        PackStreamValue::Integer(*nanos),
                        PackStreamValue::Integer(i64::from(offset)),
                    ],
                },
            )
        }
        TypedValue::DateTime {
            seconds,
            nanos,
            timezone,
        } => {
            if *nanos > 999_999_999 {
                return Err(Error::internal(
                    "query result datetime nanoseconds exceed one second",
                ));
            }
            match timezone {
                None => PackStreamValue::Structure {
                    signature: 0x64,
                    fields: vec![
                        PackStreamValue::Integer(*seconds),
                        PackStreamValue::Integer(i64::from(*nanos)),
                    ],
                },
                Some(timezone) => {
                    if let Some(offset) = fixed_offset_seconds(timezone) {
                        PackStreamValue::Structure {
                            signature: 0x49,
                            fields: vec![
                                PackStreamValue::Integer(*seconds),
                                PackStreamValue::Integer(i64::from(*nanos)),
                                PackStreamValue::Integer(i64::from(offset)),
                            ],
                        }
                    } else {
                        timezone.parse::<chrono_tz::Tz>().map_err(|_| {
                            Error::internal("query result contains an invalid timezone")
                        })?;
                        PackStreamValue::Structure {
                            signature: 0x69,
                            fields: vec![
                                PackStreamValue::Integer(*seconds),
                                PackStreamValue::Integer(i64::from(*nanos)),
                                PackStreamValue::String(timezone.clone()),
                            ],
                        }
                    }
                }
            }
        }
        TypedValue::Duration {
            months,
            days,
            seconds,
            nanos,
        } => {
            if nanos.unsigned_abs() > 999_999_999 {
                return Err(Error::internal(
                    "query result duration nanoseconds exceed one second",
                ));
            }
            PackStreamValue::Structure {
                signature: 0x45,
                fields: vec![
                    PackStreamValue::Integer(*months),
                    PackStreamValue::Integer(*days),
                    PackStreamValue::Integer(*seconds),
                    PackStreamValue::Integer(i64::from(*nanos)),
                ],
            }
        }
        TypedValue::Vector(values) => PackStreamValue::List(
            values
                .iter()
                .map(|value| PackStreamValue::Float(f64::from(*value)))
                .collect(),
        ),
        TypedValue::Node(node) => node_to_pack(node)?,
        TypedValue::Relationship(edge) => relationship_to_pack(edge)?,
        TypedValue::Path(path) => path_to_pack(path)?,
        TypedValue::List(values) => PackStreamValue::List(
            values
                .iter()
                .map(typed_to_pack)
                .collect::<Result<Vec<_>>>()?,
        ),
        TypedValue::Map(values) => PackStreamValue::Map(
            values
                .iter()
                .map(|(key, value)| Ok((key.clone(), typed_to_pack(value)?)))
                .collect::<Result<BTreeMap<_, _>>>()?,
        ),
    })
}

fn validate_result_time(nanos: i64, offset_seconds: Option<i32>) -> Result<()> {
    if !(0..86_400_000_000_000).contains(&nanos) {
        return Err(Error::internal("query result time is outside one day"));
    }
    if offset_seconds.is_some_and(|offset| i64::from(offset).abs() >= 86_400) {
        return Err(Error::internal(
            "query result timezone offset is outside one day",
        ));
    }
    Ok(())
}

fn fixed_offset_seconds(timezone: &str) -> Option<i32> {
    let (sign, value) = match timezone.as_bytes().first().copied()? {
        b'+' => (1_i32, &timezone[1..]),
        b'-' => (-1_i32, &timezone[1..]),
        _ => return None,
    };
    let mut parts = value.split(':');
    let hours = parts.next()?.parse::<i32>().ok()?;
    let minutes = parts.next()?.parse::<i32>().ok()?;
    let seconds = parts
        .next()
        .map(str::parse::<i32>)
        .transpose()
        .ok()?
        .unwrap_or(0);
    if parts.next().is_some() || hours > 23 || minutes > 59 || seconds > 59 || value.len() < 5 {
        return None;
    }
    hours
        .checked_mul(3_600)?
        .checked_add(minutes.checked_mul(60)?)?
        .checked_add(seconds)?
        .checked_mul(sign)
}

fn stable_numeric_id(value: &str) -> Result<i64> {
    value
        .parse::<u64>()
        .map(|value| value as i64)
        .map_err(|_| Error::internal("query result contains an invalid entity identifier"))
}

fn node_to_pack(node: &super::ResultNode) -> Result<PackStreamValue> {
    Ok(PackStreamValue::Structure {
        signature: 0x4E,
        fields: vec![
            PackStreamValue::Integer(stable_numeric_id(&node.id)?),
            PackStreamValue::List(
                node.labels
                    .iter()
                    .cloned()
                    .map(PackStreamValue::String)
                    .collect(),
            ),
            PackStreamValue::Map(
                node.properties
                    .iter()
                    .map(|(key, value)| Ok((key.clone(), typed_to_pack(value)?)))
                    .collect::<Result<BTreeMap<_, _>>>()?,
            ),
            PackStreamValue::String(node.id.clone()),
        ],
    })
}

fn relationship_to_pack(edge: &super::RelationshipValue) -> Result<PackStreamValue> {
    Ok(PackStreamValue::Structure {
        signature: 0x52,
        fields: vec![
            PackStreamValue::Integer(stable_numeric_id(&edge.id)?),
            PackStreamValue::Integer(stable_numeric_id(&edge.source)?),
            PackStreamValue::Integer(stable_numeric_id(&edge.target)?),
            PackStreamValue::String(edge.relationship_type.clone()),
            PackStreamValue::Map(
                edge.properties
                    .iter()
                    .map(|(key, value)| Ok((key.clone(), typed_to_pack(value)?)))
                    .collect::<Result<BTreeMap<_, _>>>()?,
            ),
            PackStreamValue::String(edge.id.clone()),
            PackStreamValue::String(edge.source.clone()),
            PackStreamValue::String(edge.target.clone()),
        ],
    })
}

fn unbound_relationship_to_pack(edge: &super::RelationshipValue) -> Result<PackStreamValue> {
    Ok(PackStreamValue::Structure {
        signature: 0x72,
        fields: vec![
            PackStreamValue::Integer(stable_numeric_id(&edge.id)?),
            PackStreamValue::String(edge.relationship_type.clone()),
            PackStreamValue::Map(
                edge.properties
                    .iter()
                    .map(|(key, value)| Ok((key.clone(), typed_to_pack(value)?)))
                    .collect::<Result<BTreeMap<_, _>>>()?,
            ),
            PackStreamValue::String(edge.id.clone()),
        ],
    })
}

fn path_to_pack(path: &super::PathValue) -> Result<PackStreamValue> {
    if path.nodes.len() != path.relationships.len().saturating_add(1) {
        return Err(Error::internal("query result contains a malformed path"));
    }
    let mut node_indexes = BTreeMap::<&str, usize>::new();
    let mut nodes = Vec::new();
    for node in &path.nodes {
        if !node_indexes.contains_key(node.id.as_str()) {
            let index = nodes.len();
            node_indexes.insert(node.id.as_str(), index);
            nodes.push(node_to_pack(node)?);
        }
    }
    let mut relationship_indexes = BTreeMap::<&str, usize>::new();
    let mut relationships = Vec::new();
    let mut sequence = Vec::with_capacity(path.relationships.len().saturating_mul(2));
    for (index, relationship) in path.relationships.iter().enumerate() {
        let relationship_index =
            if let Some(existing) = relationship_indexes.get(relationship.id.as_str()) {
                *existing
            } else {
                let next = relationships.len();
                relationship_indexes.insert(relationship.id.as_str(), next);
                relationships.push(unbound_relationship_to_pack(relationship)?);
                next
            };
        let left = &path.nodes[index].id;
        let right = &path.nodes[index + 1].id;
        let forward = relationship.source == *left && relationship.target == *right;
        let reverse = relationship.source == *right && relationship.target == *left;
        if !forward && !reverse {
            return Err(Error::internal(
                "query result path relationship does not connect adjacent nodes",
            ));
        }
        let relationship_reference = i64::try_from(relationship_index.saturating_add(1))
            .map_err(|_| Error::internal("query result path exceeds Bolt index range"))?;
        sequence.push(PackStreamValue::Integer(if forward {
            relationship_reference
        } else {
            -relationship_reference
        }));
        let node_reference = node_indexes
            .get(right.as_str())
            .copied()
            .ok_or_else(|| Error::internal("query result path node is missing"))?;
        let node_reference = i64::try_from(node_reference)
            .map_err(|_| Error::internal("query result path exceeds Bolt index range"))?;
        sequence.push(PackStreamValue::Integer(node_reference));
    }
    Ok(PackStreamValue::Structure {
        signature: 0x50,
        fields: vec![
            PackStreamValue::List(nodes),
            PackStreamValue::List(relationships),
            PackStreamValue::List(sequence),
        ],
    })
}

fn as_map(value: &PackStreamValue) -> Result<&BTreeMap<String, PackStreamValue>> {
    if let PackStreamValue::Map(value) = value {
        Ok(value)
    } else {
        Err(Error::new(
            crate::ErrorCode::ProtocolViolation,
            "expected PackStream map",
        ))
    }
}

fn reject_unknown_keys(
    values: &BTreeMap<String, PackStreamValue>,
    allowed: &[&str],
    message: &str,
) -> Result<()> {
    if let Some(key) = values.keys().find(|key| !allowed.contains(&key.as_str())) {
        return Err(Error::new(
            crate::ErrorCode::ProtocolViolation,
            format!("{message} field {key:?} is not supported by Bolt 5.0"),
        ));
    }
    Ok(())
}

fn as_string_ref(value: &PackStreamValue) -> Option<&str> {
    if let PackStreamValue::String(value) = value {
        Some(value)
    } else {
        None
    }
}

fn as_i64(value: &PackStreamValue) -> Result<i64> {
    if let PackStreamValue::Integer(value) = value {
        Ok(*value)
    } else {
        Err(Error::new(
            crate::ErrorCode::ProtocolViolation,
            "expected PackStream integer",
        ))
    }
}

fn pack_map_to_json(
    values: &BTreeMap<String, PackStreamValue>,
) -> Result<BTreeMap<String, serde_json::Value>> {
    values
        .iter()
        .map(|(key, value)| Ok((key.clone(), pack_to_json(value)?)))
        .collect()
}

fn pack_to_json(value: &PackStreamValue) -> Result<serde_json::Value> {
    match value {
        PackStreamValue::Null => Ok(serde_json::Value::Null),
        PackStreamValue::Boolean(value) => Ok((*value).into()),
        PackStreamValue::Integer(value) => Ok((*value).into()),
        PackStreamValue::Float(value) => serde_json::Number::from_f64(*value)
            .map(serde_json::Value::Number)
            .ok_or_else(|| Error::invalid_data("non-finite query parameter")),
        PackStreamValue::String(value) => Ok(value.clone().into()),
        PackStreamValue::Bytes(value) => Ok(serde_json::json!({
            "$irongraph_type": "bytes",
            "value": value,
        })),
        PackStreamValue::List(values) => values
            .iter()
            .map(pack_to_json)
            .collect::<Result<Vec<_>>>()
            .map(serde_json::Value::Array),
        PackStreamValue::Map(values) => values
            .iter()
            .map(|(key, value)| Ok((key.clone(), pack_to_json(value)?)))
            .collect::<Result<serde_json::Map<_, _>>>()
            .map(serde_json::Value::Object),
        PackStreamValue::Structure { signature, fields } => {
            temporal_parameter_to_json(*signature, fields)
        }
    }
}

fn temporal_parameter_to_json(
    signature: u8,
    fields: &[PackStreamValue],
) -> Result<serde_json::Value> {
    match (signature, fields) {
        (0x44, [days]) => Ok(serde_json::json!({
            "$irongraph_type": "date",
            "days": parameter_i64(days, "date days")?,
        })),
        (0x74, [nanos]) => Ok(serde_json::json!({
            "$irongraph_type": "local_time",
            "nanos": parameter_time_nanos(nanos, "local time nanos")?,
        })),
        (0x54, [nanos, offset]) => Ok(serde_json::json!({
            "$irongraph_type": "zoned_time",
            "nanos": parameter_time_nanos(nanos, "time nanos")?,
            "offset_seconds": parameter_offset(offset, "time offset")?,
        })),
        (0x64, [seconds, nanos]) => Ok(serde_json::json!({
            "$irongraph_type": "local_datetime",
            "seconds": parameter_i64(seconds, "datetime seconds")?,
            "nanos": parameter_fraction_nanos(nanos, "datetime nanos")?,
        })),
        (0x49, [seconds, nanos, offset]) => Ok(serde_json::json!({
            "$irongraph_type": "zoned_datetime",
            "seconds": parameter_i64(seconds, "datetime seconds")?,
            "nanos": parameter_fraction_nanos(nanos, "datetime nanos")?,
            "timezone": format_offset(parameter_offset(offset, "datetime offset")?)?,
        })),
        (0x69, [seconds, nanos, PackStreamValue::String(timezone)]) => {
            if timezone.is_empty()
                || timezone.len() > 255
                || timezone.parse::<chrono_tz::Tz>().is_err()
            {
                return Err(Error::new(
                    crate::ErrorCode::ProtocolViolation,
                    "datetime timezone is not a supported IANA identifier",
                ));
            }
            Ok(serde_json::json!({
                "$irongraph_type": "zoned_datetime",
                "seconds": parameter_i64(seconds, "datetime seconds")?,
                "nanos": parameter_fraction_nanos(nanos, "datetime nanos")?,
                "timezone": timezone,
            }))
        }
        (0x45, [months, days, seconds, nanos]) => Ok(serde_json::json!({
            "$irongraph_type": "duration",
            "months": parameter_i64(months, "duration months")?,
            "days": parameter_i64(days, "duration days")?,
            "seconds": parameter_i64(seconds, "duration seconds")?,
            "nanos": parameter_duration_nanos(nanos)?,
        })),
        _ => Err(Error::new(
            crate::ErrorCode::ProtocolViolation,
            format!("PackStream structure 0x{signature:02x} is not a query parameter value"),
        )),
    }
}

fn parameter_i64(value: &PackStreamValue, field: &str) -> Result<i64> {
    match value {
        PackStreamValue::Integer(value) => Ok(*value),
        _ => Err(Error::new(
            crate::ErrorCode::ProtocolViolation,
            format!("{field} must be an integer"),
        )),
    }
}

fn parameter_i32(value: &PackStreamValue, field: &str) -> Result<i32> {
    i32::try_from(parameter_i64(value, field)?).map_err(|_| {
        Error::new(
            crate::ErrorCode::ProtocolViolation,
            format!("{field} exceeds 32-bit range"),
        )
    })
}

fn parameter_u32(value: &PackStreamValue, field: &str) -> Result<u32> {
    u32::try_from(parameter_i64(value, field)?).map_err(|_| {
        Error::new(
            crate::ErrorCode::ProtocolViolation,
            format!("{field} is negative or exceeds 32-bit range"),
        )
    })
}

fn parameter_time_nanos(value: &PackStreamValue, field: &str) -> Result<i64> {
    let nanos = parameter_i64(value, field)?;
    if !(0..86_400_000_000_000).contains(&nanos) {
        return Err(Error::new(
            crate::ErrorCode::ProtocolViolation,
            format!("{field} is outside one day"),
        ));
    }
    Ok(nanos)
}

fn parameter_fraction_nanos(value: &PackStreamValue, field: &str) -> Result<u32> {
    let nanos = parameter_u32(value, field)?;
    if nanos > 999_999_999 {
        return Err(Error::new(
            crate::ErrorCode::ProtocolViolation,
            format!("{field} exceeds one second"),
        ));
    }
    Ok(nanos)
}

fn parameter_offset(value: &PackStreamValue, field: &str) -> Result<i32> {
    let offset = parameter_i32(value, field)?;
    if i64::from(offset).abs() >= 86_400 {
        return Err(Error::new(
            crate::ErrorCode::ProtocolViolation,
            format!("{field} is outside one day"),
        ));
    }
    Ok(offset)
}

fn parameter_duration_nanos(value: &PackStreamValue) -> Result<i32> {
    let nanos = parameter_i32(value, "duration nanos")?;
    if nanos.unsigned_abs() > 999_999_999 {
        return Err(Error::new(
            crate::ErrorCode::ProtocolViolation,
            "duration nanos exceeds one second",
        ));
    }
    Ok(nanos)
}

fn format_offset(offset_seconds: i32) -> Result<String> {
    let absolute = i64::from(offset_seconds).abs();
    if absolute >= 24 * 60 * 60 {
        return Err(Error::new(
            crate::ErrorCode::ProtocolViolation,
            "datetime offset is outside one day",
        ));
    }
    let sign = if offset_seconds < 0 { '-' } else { '+' };
    let hours = absolute / 3_600;
    let minutes = absolute % 3_600 / 60;
    let seconds = absolute % 60;
    if seconds == 0 {
        Ok(format!("{sign}{hours:02}:{minutes:02}"))
    } else {
        Ok(format!("{sign}{hours:02}:{minutes:02}:{seconds:02}"))
    }
}

fn project_from_extra(
    executor: &dyn QueryExecutor,
    extra: &BTreeMap<String, PackStreamValue>,
) -> Result<Option<ProjectId>> {
    let Some(value) = extra.get("db") else {
        return Ok(None);
    };
    let text = as_string_ref(value)
        .ok_or_else(|| Error::new(crate::ErrorCode::ProtocolViolation, "db must be a string"))?;
    executor.resolve_project(text).map(Some)
}

fn bookmark_from_extra(extra: &BTreeMap<String, PackStreamValue>) -> Result<Option<Bookmark>> {
    let Some(value) = extra.get("bookmarks") else {
        return Ok(None);
    };
    let PackStreamValue::List(values) = value else {
        return Err(Error::new(
            crate::ErrorCode::ProtocolViolation,
            "bookmarks must be a list of strings",
        ));
    };
    let mut selected: Option<Bookmark> = None;
    for value in values {
        let text = as_string_ref(value).ok_or_else(|| {
            Error::new(
                crate::ErrorCode::ProtocolViolation,
                "bookmark is not a string",
            )
        })?;
        let bookmark = parse_bookmark(text)?;
        if selected
            .is_some_and(|current| current.index == bookmark.index && current.term != bookmark.term)
        {
            return Err(Error::new(
                crate::ErrorCode::ProtocolViolation,
                "bookmarks conflict at one local write index",
            ));
        }
        if selected.is_none_or(|current| bookmark.index > current.index) {
            selected = Some(bookmark);
        }
    }
    Ok(selected)
}

fn parse_bookmark(text: &str) -> Result<Bookmark> {
    let mut parts = text.split(':');
    if parts.next() != Some("ig") {
        return Err(Error::new(
            crate::ErrorCode::ProtocolViolation,
            "bookmark prefix is invalid",
        ));
    }
    let term = parts
        .next()
        .and_then(|value| value.parse().ok())
        .ok_or_else(|| {
            Error::new(
                crate::ErrorCode::ProtocolViolation,
                "bookmark term is invalid",
            )
        })?;
    let index = parts
        .next()
        .and_then(|value| value.parse().ok())
        .ok_or_else(|| {
            Error::new(
                crate::ErrorCode::ProtocolViolation,
                "bookmark index is invalid",
            )
        })?;
    if parts.next().is_some() {
        return Err(Error::new(
            crate::ErrorCode::ProtocolViolation,
            "bookmark has extra components",
        ));
    }
    Ok(Bookmark { term, index })
}

fn protocol_state(message: &str, state: SessionState) -> Error {
    Error::new(
        crate::ErrorCode::ProtocolViolation,
        format!("{message} is not valid in Bolt state {state:?}"),
    )
}

#[cfg(test)]
mod tests {
    use super::{
        BOLT_MAGIC, BOLT_ROUTE_SIGNATURE, BOLT_SERVER_AGENT, BoltSession, PackStreamValue,
        as_string_ref, bookmark_from_extra, decode_packstream, encode_packstream, fetch_count,
        path_to_pack, proposal_includes_v5_0, read_chunked_message, typed_to_pack, write_message,
    };
    use crate::{
        Bookmark, CommitAcknowledgement, Error, ErrorCode, ProjectId, Result,
        protocol::{
            BatchColumn, PathValue, QueryColumn, QueryExecutor, QueryRequest, QueryStatistics,
            QueryStreamEvent, QueryTransaction, RelationshipValue, ResultNode, TypedValue,
        },
    };
    use std::{
        collections::BTreeMap,
        sync::{
            Arc,
            atomic::{AtomicBool, AtomicUsize, Ordering},
        },
    };
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use uuid::Uuid;

    async fn negotiate_and_hello(client: &mut tokio::io::DuplexStream) -> Result<()> {
        let mut handshake = Vec::from(BOLT_MAGIC.to_be_bytes());
        handshake.extend_from_slice(&0x0004_0405_u32.to_be_bytes());
        handshake.extend_from_slice(&[0; 12]);
        client.write_all(&handshake).await?;
        let mut selected = [0_u8; 4];
        client.read_exact(&mut selected).await?;
        if selected != 5_u32.to_be_bytes() {
            return Err(Error::new(
                ErrorCode::ProtocolViolation,
                "server did not select Bolt 5.0",
            ));
        }
        write_message(
            client,
            &PackStreamValue::Structure {
                signature: 0x01,
                fields: vec![PackStreamValue::Map(BTreeMap::from([(
                    "user_agent".to_owned(),
                    PackStreamValue::String("irongraph-test".to_owned()),
                )]))],
            },
        )
        .await?;
        assert_signature(&read_message(client).await?, 0x70)
    }

    fn response_metadata(
        value: &PackStreamValue,
        signature: u8,
    ) -> Result<&BTreeMap<String, PackStreamValue>> {
        match value {
            PackStreamValue::Structure {
                signature: actual,
                fields,
            } if *actual == signature => match fields.as_slice() {
                [PackStreamValue::Map(metadata)] => Ok(metadata),
                _ => Err(Error::new(
                    ErrorCode::ProtocolViolation,
                    "Bolt response has invalid metadata fields",
                )),
            },
            _ => Err(Error::new(
                ErrorCode::ProtocolViolation,
                format!("expected Bolt signature 0x{signature:02x}"),
            )),
        }
    }

    #[test]
    fn packstream_round_trip() {
        let value = PackStreamValue::Structure {
            signature: 0x10,
            fields: vec![
                PackStreamValue::String("MATCH (n) RETURN n".to_owned()),
                PackStreamValue::Map(BTreeMap::from([(
                    "limit".to_owned(),
                    PackStreamValue::Integer(42),
                )])),
            ],
        };
        let mut encoded = Vec::new();
        assert!(encode_packstream(&value, &mut encoded).is_ok());
        assert_eq!(decode_packstream(&encoded).ok(), Some(value));
    }

    #[test]
    fn bolt_date_preserves_wide_epoch_days() {
        let wide_days = 365_242_499_634_i64;
        assert!(matches!(
            typed_to_pack(&TypedValue::Date(wide_days)),
            Ok(PackStreamValue::Structure {
                signature: 0x44,
                fields,
            }) if fields == vec![PackStreamValue::Integer(wide_days)]
        ));
    }

    #[test]
    fn temporal_and_path_values_use_native_bolt_structures() {
        assert!(proposal_includes_v5_0(0x0000_0005));
        assert!(proposal_includes_v5_0(0x0004_0405));
        assert!(!proposal_includes_v5_0(0x0000_0105));
        assert!(!proposal_includes_v5_0(0x0000_0004));

        assert!(matches!(
            typed_to_pack(&TypedValue::Time {
                nanos: 42,
                offset_seconds: None,
            }),
            Ok(PackStreamValue::Structure {
                signature: 0x74,
                fields,
            }) if fields.len() == 1
        ));
        assert!(matches!(
            typed_to_pack(&TypedValue::DateTime {
                seconds: 42,
                nanos: 7,
                timezone: None,
            }),
            Ok(PackStreamValue::Structure {
                signature: 0x64,
                fields,
            }) if fields.len() == 2
        ));

        let path = PathValue {
            nodes: vec![
                ResultNode {
                    id: "1".to_owned(),
                    labels: vec!["A".to_owned()],
                    properties: BTreeMap::new(),
                },
                ResultNode {
                    id: "2".to_owned(),
                    labels: vec!["B".to_owned()],
                    properties: BTreeMap::new(),
                },
            ],
            relationships: vec![RelationshipValue {
                id: "3".to_owned(),
                source: "1".to_owned(),
                target: "2".to_owned(),
                relationship_type: "LINK".to_owned(),
                properties: BTreeMap::new(),
            }],
        };
        assert!(matches!(
            path_to_pack(&path),
            Ok(PackStreamValue::Structure {
                signature: 0x50,
                fields,
            }) if fields.len() == 3
        ));
    }

    #[test]
    fn bolt_5_0_range_negotiation_and_fetch_metadata_are_strict() -> Result<()> {
        assert!(proposal_includes_v5_0(0x0000_0005));
        assert!(proposal_includes_v5_0(0x0004_0405));
        assert!(!proposal_includes_v5_0(0x0000_0105));
        assert!(!proposal_includes_v5_0(0x0003_0405));
        assert!(!proposal_includes_v5_0(0x0104_0405));

        assert!(fetch_count(&BTreeMap::new(), "PULL", false).is_err());
        assert!(
            fetch_count(
                &BTreeMap::from([("n".to_owned(), PackStreamValue::Integer(0))]),
                "PULL",
                false,
            )
            .is_err()
        );
        assert!(
            fetch_count(
                &BTreeMap::from([
                    ("n".to_owned(), PackStreamValue::Integer(1)),
                    ("qid".to_owned(), PackStreamValue::Integer(0)),
                ]),
                "PULL",
                false,
            )
            .is_err()
        );
        assert_eq!(
            fetch_count(
                &BTreeMap::from([
                    ("n".to_owned(), PackStreamValue::Integer(-1)),
                    ("qid".to_owned(), PackStreamValue::Integer(0)),
                ]),
                "PULL",
                true,
            )?,
            usize::MAX
        );
        assert!(
            bookmark_from_extra(&BTreeMap::from([(
                "bookmarks".to_owned(),
                PackStreamValue::List(vec![
                    PackStreamValue::String("ig:1:9".to_owned()),
                    PackStreamValue::String("ig:2:9".to_owned()),
                ]),
            )]))
            .is_err()
        );
        Ok(())
    }

    #[test]
    fn bolt_5_temporal_and_entity_shapes_reject_invalid_internal_values() -> Result<()> {
        assert!(matches!(
            typed_to_pack(&TypedValue::DateTime {
                seconds: 1,
                nanos: 2,
                timezone: Some("+01:30".to_owned()),
            })?,
            PackStreamValue::Structure {
                signature: 0x49,
                fields,
            } if fields.len() == 3
        ));
        assert!(matches!(
            typed_to_pack(&TypedValue::DateTime {
                seconds: 1,
                nanos: 2,
                timezone: Some("Europe/London".to_owned()),
            })?,
            PackStreamValue::Structure {
                signature: 0x69,
                fields,
            } if fields.len() == 3
        ));
        assert!(typed_to_pack(&TypedValue::Integer("not-an-integer".to_owned())).is_err());
        assert!(
            typed_to_pack(&TypedValue::Time {
                nanos: 86_400_000_000_000,
                offset_seconds: None,
            })
            .is_err()
        );
        assert!(
            typed_to_pack(&TypedValue::DateTime {
                seconds: 1,
                nanos: 2,
                timezone: Some("not/a-zone".to_owned()),
            })
            .is_err()
        );

        let node = typed_to_pack(&TypedValue::Node(ResultNode {
            id: "11".to_owned(),
            labels: vec!["Item".to_owned()],
            properties: BTreeMap::new(),
        }))?;
        assert!(matches!(
            node,
            PackStreamValue::Structure {
                signature: 0x4E,
                fields,
            } if fields.len() == 4
        ));
        let relationship = typed_to_pack(&TypedValue::Relationship(RelationshipValue {
            id: "12".to_owned(),
            source: "11".to_owned(),
            target: "13".to_owned(),
            relationship_type: "LINK".to_owned(),
            properties: BTreeMap::new(),
        }))?;
        assert!(matches!(
            relationship,
            PackStreamValue::Structure {
                signature: 0x52,
                fields,
            } if fields.len() == 8
        ));
        Ok(())
    }

    struct StreamingExecutor {
        emitted_batches: Arc<AtomicUsize>,
        cursor_closed: Arc<AtomicBool>,
    }

    struct CancellingExecutor {
        cancelled: Arc<AtomicBool>,
    }

    impl QueryExecutor for CancellingExecutor {
        fn execute(
            &self,
            request: QueryRequest,
            _emit: &mut dyn FnMut(QueryStreamEvent) -> Result<()>,
        ) -> Result<()> {
            while !request.cancellation.is_cancelled() {
                std::thread::sleep(std::time::Duration::from_millis(1));
            }
            self.cancelled.store(true, Ordering::SeqCst);
            Err(Error::new(ErrorCode::Cancelled, "query was reset"))
        }

        fn begin(
            &self,
            _project: Option<ProjectId>,
            _bookmark: Option<Bookmark>,
            _consistency: CommitAcknowledgement,
        ) -> Result<Box<dyn QueryTransaction>> {
            Err(Error::new(
                ErrorCode::ProtocolViolation,
                "transactions are not used by this test",
            ))
        }
    }

    struct TransactionExecutor {
        project: ProjectId,
        begun: Arc<parking_lot::Mutex<Option<(Option<ProjectId>, Option<Bookmark>)>>>,
    }

    impl QueryExecutor for TransactionExecutor {
        fn resolve_project(&self, selector: &str) -> Result<ProjectId> {
            if selector == "analytics" {
                Ok(self.project)
            } else {
                Err(Error::new(
                    ErrorCode::ProjectNotFound,
                    "test project does not exist",
                ))
            }
        }

        fn execute(
            &self,
            _request: QueryRequest,
            _emit: &mut dyn FnMut(QueryStreamEvent) -> Result<()>,
        ) -> Result<()> {
            Err(Error::new(
                ErrorCode::ProtocolViolation,
                "autocommit is not used by this test",
            ))
        }

        fn begin(
            &self,
            project: Option<ProjectId>,
            bookmark: Option<Bookmark>,
            consistency: CommitAcknowledgement,
        ) -> Result<Box<dyn QueryTransaction>> {
            if consistency != CommitAcknowledgement::Published {
                return Err(Error::internal("unexpected transaction consistency"));
            }
            *self.begun.lock() = Some((project, bookmark));
            Ok(Box::new(TestTransaction {
                project: self.project,
            }))
        }
    }

    struct TestTransaction {
        project: ProjectId,
    }

    impl QueryTransaction for TestTransaction {
        fn run(
            &mut self,
            request: QueryRequest,
            emit: &mut dyn FnMut(QueryStreamEvent) -> Result<()>,
        ) -> Result<()> {
            if request.project_id != Some(self.project) {
                return Err(Error::internal("transaction project binding changed"));
            }
            emit(QueryStreamEvent::Schema {
                request_id: request.request_id,
                columns: vec![QueryColumn {
                    name: "value".to_owned(),
                    value_type: "INTEGER".to_owned(),
                    nullable: false,
                }],
            })?;
            emit(QueryStreamEvent::Batch {
                request_id: request.request_id,
                sequence: 0,
                row_count: 2,
                columns: vec![BatchColumn {
                    name: "value".to_owned(),
                    value_type: "INTEGER".to_owned(),
                    values: vec![
                        TypedValue::Integer("1".to_owned()),
                        TypedValue::Integer("2".to_owned()),
                    ],
                }],
            })?;
            emit(QueryStreamEvent::Summary {
                request_id: request.request_id,
                bookmark: Bookmark { term: 4, index: 12 },
                statistics: QueryStatistics {
                    rows: 2,
                    ..QueryStatistics::default()
                },
                truncated: false,
                truncation_reason: None,
            })
        }

        fn commit(self: Box<Self>) -> Result<Bookmark> {
            Ok(Bookmark { term: 5, index: 20 })
        }

        fn rollback(self: Box<Self>) -> Result<()> {
            Ok(())
        }
    }

    impl QueryExecutor for StreamingExecutor {
        fn execute(
            &self,
            request: QueryRequest,
            emit: &mut dyn FnMut(QueryStreamEvent) -> Result<()>,
        ) -> Result<()> {
            emit(QueryStreamEvent::Schema {
                request_id: request.request_id,
                columns: vec![QueryColumn {
                    name: "value".to_owned(),
                    value_type: "INTEGER".to_owned(),
                    nullable: false,
                }],
            })?;
            for sequence in 0..16_u64 {
                let result = emit(QueryStreamEvent::Batch {
                    request_id: request.request_id,
                    sequence,
                    row_count: 1,
                    columns: vec![BatchColumn {
                        name: "value".to_owned(),
                        value_type: "INTEGER".to_owned(),
                        values: vec![TypedValue::Integer(sequence.to_string())],
                    }],
                });
                if let Err(error) = result {
                    self.cursor_closed.store(true, Ordering::SeqCst);
                    return Err(error);
                }
                self.emitted_batches.fetch_add(1, Ordering::SeqCst);
            }
            emit(QueryStreamEvent::Summary {
                request_id: request.request_id,
                bookmark: Bookmark { term: 1, index: 9 },
                statistics: QueryStatistics {
                    rows: 16,
                    ..QueryStatistics::default()
                },
                truncated: false,
                truncation_reason: None,
            })
        }

        fn begin(
            &self,
            _project: Option<ProjectId>,
            _bookmark: Option<Bookmark>,
            _consistency: CommitAcknowledgement,
        ) -> Result<Box<dyn QueryTransaction>> {
            Err(Error::new(
                ErrorCode::ProtocolViolation,
                "transactions are not used by this test",
            ))
        }
    }

    async fn read_message(stream: &mut tokio::io::DuplexStream) -> Result<PackStreamValue> {
        let bytes = read_chunked_message(stream)
            .await?
            .ok_or_else(|| Error::new(ErrorCode::ProtocolViolation, "Bolt stream closed"))?;
        decode_packstream(&bytes)
    }

    fn assert_signature(value: &PackStreamValue, expected: u8) -> Result<()> {
        if matches!(value, PackStreamValue::Structure { signature, .. } if *signature == expected) {
            Ok(())
        } else {
            Err(Error::new(
                ErrorCode::ProtocolViolation,
                format!("expected Bolt signature 0x{expected:02x}"),
            ))
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pull_is_bounded_and_reset_cancels_the_query_worker() -> Result<()> {
        let emitted_batches = Arc::new(AtomicUsize::new(0));
        let cursor_closed = Arc::new(AtomicBool::new(false));
        let executor: Arc<dyn QueryExecutor> = Arc::new(StreamingExecutor {
            emitted_batches: Arc::clone(&emitted_batches),
            cursor_closed: Arc::clone(&cursor_closed),
        });
        let (mut client, server) = tokio::io::duplex(64 * 1024);
        let serving = tokio::spawn(async move { BoltSession::new(executor).serve(server).await });

        negotiate_and_hello(&mut client).await?;

        write_message(
            &mut client,
            &PackStreamValue::Structure {
                signature: 0x10,
                fields: vec![
                    PackStreamValue::String("RETURN 1".to_owned()),
                    PackStreamValue::Map(BTreeMap::new()),
                    PackStreamValue::Map(BTreeMap::new()),
                ],
            },
        )
        .await?;
        assert_signature(&read_message(&mut client).await?, 0x70)?;
        tokio::task::yield_now().await;
        assert!(emitted_batches.load(Ordering::SeqCst) <= 2);

        write_message(
            &mut client,
            &PackStreamValue::Structure {
                signature: 0x3F,
                fields: vec![PackStreamValue::Map(BTreeMap::from([(
                    "n".to_owned(),
                    PackStreamValue::Integer(1),
                )]))],
            },
        )
        .await?;
        assert_signature(&read_message(&mut client).await?, 0x71)?;
        assert_signature(&read_message(&mut client).await?, 0x70)?;
        // Two events may be buffered while the worker is blocked, and `has_more` deliberately
        // peeks one row so an exact-size pull can report the terminal summary accurately.
        assert!(emitted_batches.load(Ordering::SeqCst) <= 4);

        write_message(
            &mut client,
            &PackStreamValue::Structure {
                signature: 0x0F,
                fields: Vec::new(),
            },
        )
        .await?;
        assert_signature(&read_message(&mut client).await?, 0x70)?;
        assert!(cursor_closed.load(Ordering::SeqCst));

        write_message(
            &mut client,
            &PackStreamValue::Structure {
                signature: 0x02,
                fields: Vec::new(),
            },
        )
        .await?;
        drop(client);
        serving
            .await
            .map_err(|error| Error::internal(format!("Bolt server task failed: {error}")))??;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn pipelined_reset_interrupts_a_query_before_schema() -> Result<()> {
        let cancelled = Arc::new(AtomicBool::new(false));
        let executor: Arc<dyn QueryExecutor> = Arc::new(CancellingExecutor {
            cancelled: Arc::clone(&cancelled),
        });
        let (mut client, server) = tokio::io::duplex(64 * 1024);
        let serving = tokio::spawn(async move { BoltSession::new(executor).serve(server).await });
        negotiate_and_hello(&mut client).await?;

        write_message(
            &mut client,
            &PackStreamValue::Structure {
                signature: 0x10,
                fields: vec![
                    PackStreamValue::String("RETURN 1".to_owned()),
                    PackStreamValue::Map(BTreeMap::new()),
                    PackStreamValue::Map(BTreeMap::new()),
                ],
            },
        )
        .await?;
        write_message(
            &mut client,
            &PackStreamValue::Structure {
                signature: 0x0F,
                fields: Vec::new(),
            },
        )
        .await?;
        let failure =
            tokio::time::timeout(std::time::Duration::from_secs(2), read_message(&mut client))
                .await
                .map_err(|_| {
                    Error::internal("pipelined RESET did not interrupt query execution")
                })??;
        assert_signature(&failure, 0x7F)?;
        let reset =
            tokio::time::timeout(std::time::Duration::from_secs(2), read_message(&mut client))
                .await
                .map_err(|_| Error::internal("RESET acknowledgement timed out"))??;
        assert_signature(&reset, 0x70)?;
        assert!(cancelled.load(Ordering::SeqCst));

        write_message(
            &mut client,
            &PackStreamValue::Structure {
                signature: 0x02,
                fields: Vec::new(),
            },
        )
        .await?;
        drop(client);
        serving
            .await
            .map_err(|error| Error::internal(format!("Bolt server task failed: {error}")))??;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn explicit_transaction_uses_project_bookmark_qid_and_stream_counts() -> Result<()> {
        let project = ProjectId(Uuid::new_v4());
        let begun = Arc::new(parking_lot::Mutex::new(None));
        let executor: Arc<dyn QueryExecutor> = Arc::new(TransactionExecutor {
            project,
            begun: Arc::clone(&begun),
        });
        let (mut client, server) = tokio::io::duplex(64 * 1024);
        let serving = tokio::spawn(async move { BoltSession::new(executor).serve(server).await });
        negotiate_and_hello(&mut client).await?;

        write_message(
            &mut client,
            &PackStreamValue::Structure {
                signature: 0x11,
                fields: vec![PackStreamValue::Map(BTreeMap::from([
                    (
                        "db".to_owned(),
                        PackStreamValue::String("analytics".to_owned()),
                    ),
                    (
                        "bookmarks".to_owned(),
                        PackStreamValue::List(vec![
                            PackStreamValue::String("ig:2:4".to_owned()),
                            PackStreamValue::String("ig:3:7".to_owned()),
                        ]),
                    ),
                ]))],
            },
        )
        .await?;
        assert_signature(&read_message(&mut client).await?, 0x70)?;

        write_message(
            &mut client,
            &PackStreamValue::Structure {
                signature: 0x10,
                fields: vec![
                    PackStreamValue::String("UNWIND [1, 2] AS value RETURN value".to_owned()),
                    PackStreamValue::Map(BTreeMap::new()),
                    PackStreamValue::Map(BTreeMap::new()),
                ],
            },
        )
        .await?;
        let run = read_message(&mut client).await?;
        let metadata = response_metadata(&run, 0x70)?;
        assert_eq!(metadata.get("qid"), Some(&PackStreamValue::Integer(0)));

        write_message(
            &mut client,
            &PackStreamValue::Structure {
                signature: 0x3F,
                fields: vec![PackStreamValue::Map(BTreeMap::from([
                    ("n".to_owned(), PackStreamValue::Integer(1)),
                    ("qid".to_owned(), PackStreamValue::Integer(-1)),
                ]))],
            },
        )
        .await?;
        assert_signature(&read_message(&mut client).await?, 0x71)?;
        let partial = read_message(&mut client).await?;
        assert_eq!(
            response_metadata(&partial, 0x70)?.get("has_more"),
            Some(&PackStreamValue::Boolean(true))
        );

        write_message(
            &mut client,
            &PackStreamValue::Structure {
                signature: 0x3F,
                fields: vec![PackStreamValue::Map(BTreeMap::from([
                    ("n".to_owned(), PackStreamValue::Integer(-1)),
                    ("qid".to_owned(), PackStreamValue::Integer(0)),
                ]))],
            },
        )
        .await?;
        assert_signature(&read_message(&mut client).await?, 0x71)?;
        let summary = read_message(&mut client).await?;
        let metadata = response_metadata(&summary, 0x70)?;
        assert_eq!(
            metadata.get("has_more"),
            Some(&PackStreamValue::Boolean(false))
        );
        assert_eq!(
            metadata.get("bookmark"),
            Some(&PackStreamValue::String("ig:4:12".to_owned()))
        );

        write_message(
            &mut client,
            &PackStreamValue::Structure {
                signature: 0x12,
                fields: Vec::new(),
            },
        )
        .await?;
        let committed = read_message(&mut client).await?;
        assert_eq!(
            response_metadata(&committed, 0x70)?.get("bookmark"),
            Some(&PackStreamValue::String("ig:5:20".to_owned()))
        );
        assert_eq!(
            begun.lock().clone(),
            Some((Some(project), Some(Bookmark { term: 3, index: 7 })))
        );

        write_message(
            &mut client,
            &PackStreamValue::Structure {
                signature: 0x02,
                fields: Vec::new(),
            },
        )
        .await?;
        drop(client);
        serving
            .await
            .map_err(|error| Error::internal(format!("Bolt server task failed: {error}")))??;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn bolt_5_0_rejects_logon_and_recovers_only_with_reset() -> Result<()> {
        let executor: Arc<dyn QueryExecutor> = Arc::new(StreamingExecutor {
            emitted_batches: Arc::new(AtomicUsize::new(0)),
            cursor_closed: Arc::new(AtomicBool::new(false)),
        });
        let (mut client, server) = tokio::io::duplex(64 * 1024);
        let serving = tokio::spawn(async move { BoltSession::new(executor).serve(server).await });
        negotiate_and_hello(&mut client).await?;
        write_message(
            &mut client,
            &PackStreamValue::Structure {
                signature: 0x6A,
                fields: vec![PackStreamValue::Map(BTreeMap::new())],
            },
        )
        .await?;
        assert_signature(&read_message(&mut client).await?, 0x7F)?;
        write_message(
            &mut client,
            &PackStreamValue::Structure {
                signature: 0x6B,
                fields: Vec::new(),
            },
        )
        .await?;
        assert_signature(&read_message(&mut client).await?, 0x7E)?;
        write_message(
            &mut client,
            &PackStreamValue::Structure {
                signature: 0x0F,
                fields: Vec::new(),
            },
        )
        .await?;
        assert_signature(&read_message(&mut client).await?, 0x70)?;
        write_message(
            &mut client,
            &PackStreamValue::Structure {
                signature: 0x02,
                fields: Vec::new(),
            },
        )
        .await?;
        drop(client);
        serving
            .await
            .map_err(|error| Error::internal(format!("Bolt server task failed: {error}")))??;
        Ok(())
    }

    #[tokio::test]
    async fn a_routing_driver_completes_the_handshake_and_receives_a_routing_table() -> Result<()> {
        // Everything here is what an official driver sends by default and what it refuses to
        // continue without. Each was previously rejected: the routing context failed `HELLO`
        // outright, so no `neo4j://` URI could connect; the agent named the product, which drivers
        // parse to gate features and some refuse outright; and `ROUTE` did not exist, so even an
        // accepted routing context left the driver with no table and no way to run a query.
        let executor: Arc<dyn QueryExecutor> = Arc::new(TransactionExecutor {
            project: ProjectId(Uuid::from_u128(9)),
            begun: Arc::new(parking_lot::Mutex::new(None)),
        });
        let (mut client, server) = tokio::io::duplex(64 * 1024);
        let serving = tokio::spawn(async move { BoltSession::new(executor).serve(server).await });

        let mut handshake = Vec::from(BOLT_MAGIC.to_be_bytes());
        handshake.extend_from_slice(&0x0004_0405_u32.to_be_bytes());
        handshake.extend_from_slice(&[0; 12]);
        client.write_all(&handshake).await?;
        let mut selected = [0_u8; 4];
        client.read_exact(&mut selected).await?;
        assert_eq!(selected, 5_u32.to_be_bytes());

        write_message(
            &mut client,
            &PackStreamValue::Structure {
                signature: 0x01,
                fields: vec![PackStreamValue::Map(BTreeMap::from([
                    (
                        "user_agent".to_owned(),
                        PackStreamValue::String("neo4j-python/5.28.0".to_owned()),
                    ),
                    (
                        "scheme".to_owned(),
                        PackStreamValue::String("none".to_owned()),
                    ),
                    (
                        "routing".to_owned(),
                        PackStreamValue::Map(BTreeMap::from([(
                            "address".to_owned(),
                            PackStreamValue::String("127.0.0.1:18485".to_owned()),
                        )])),
                    ),
                ]))],
            },
        )
        .await?;
        let hello = read_message(&mut client).await?;
        let metadata = response_metadata(&hello, 0x70)?;
        let agent = metadata.get("server").and_then(as_string_ref);
        assert_eq!(agent, Some(BOLT_SERVER_AGENT));
        assert!(
            agent.is_some_and(|agent| agent.starts_with("Neo4j/")),
            "drivers gate on the agent prefix and refuse anything they do not recognise"
        );

        write_message(
            &mut client,
            &PackStreamValue::Structure {
                signature: BOLT_ROUTE_SIGNATURE,
                fields: vec![
                    PackStreamValue::Map(BTreeMap::from([(
                        "address".to_owned(),
                        PackStreamValue::String("127.0.0.1:18485".to_owned()),
                    )])),
                    PackStreamValue::List(Vec::new()),
                    PackStreamValue::Map(BTreeMap::new()),
                ],
            },
        )
        .await?;
        let routed = read_message(&mut client).await?;
        let routed = response_metadata(&routed, 0x70)?;
        let Some(PackStreamValue::Map(table)) = routed.get("rt") else {
            return Err(Error::internal("ROUTE did not return a routing table"));
        };
        assert!(matches!(
            table.get("ttl"),
            Some(PackStreamValue::Integer(_))
        ));
        let Some(PackStreamValue::List(servers)) = table.get("servers") else {
            return Err(Error::internal("routing table has no servers"));
        };
        let mut roles = servers
            .iter()
            .filter_map(|server| match server {
                PackStreamValue::Map(entry) => entry.get("role").and_then(as_string_ref),
                _ => None,
            })
            .collect::<Vec<_>>();
        roles.sort_unstable();
        assert_eq!(roles, vec!["READ", "ROUTE", "WRITE"]);

        // A read-mode transaction is what `session.execute_read` opens; rejecting `mode` made the
        // most common driver API unusable.
        write_message(
            &mut client,
            &PackStreamValue::Structure {
                signature: 0x11,
                fields: vec![PackStreamValue::Map(BTreeMap::from([
                    ("mode".to_owned(), PackStreamValue::String("r".to_owned())),
                    ("tx_timeout".to_owned(), PackStreamValue::Integer(30_000)),
                    (
                        "tx_metadata".to_owned(),
                        PackStreamValue::Map(BTreeMap::new()),
                    ),
                ]))],
            },
        )
        .await?;
        assert_signature(&read_message(&mut client).await?, 0x70)?;

        write_message(
            &mut client,
            &PackStreamValue::Structure {
                signature: 0x02,
                fields: Vec::new(),
            },
        )
        .await?;
        drop(client);
        serving
            .await
            .map_err(|error| Error::internal(format!("Bolt server task failed: {error}")))??;
        Ok(())
    }

    #[tokio::test]
    async fn unsupported_bolt_minor_is_not_advertised() -> Result<()> {
        let executor: Arc<dyn QueryExecutor> = Arc::new(StreamingExecutor {
            emitted_batches: Arc::new(AtomicUsize::new(0)),
            cursor_closed: Arc::new(AtomicBool::new(false)),
        });
        let (mut client, server) = tokio::io::duplex(1024);
        let serving = tokio::spawn(async move { BoltSession::new(executor).serve(server).await });
        let mut handshake = Vec::from(BOLT_MAGIC.to_be_bytes());
        handshake.extend_from_slice(&0x0000_0105_u32.to_be_bytes());
        handshake.extend_from_slice(&[0; 12]);
        client.write_all(&handshake).await?;
        let mut selected = [1_u8; 4];
        client.read_exact(&mut selected).await?;
        assert_eq!(selected, [0_u8; 4]);
        drop(client);
        serving
            .await
            .map_err(|error| Error::internal(format!("Bolt server task failed: {error}")))??;
        Ok(())
    }
}
