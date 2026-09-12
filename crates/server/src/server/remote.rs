use std::{
    future::Future,
    net::SocketAddr,
    pin::Pin,
    sync::Arc,
    task::{Context, Poll},
    time::Duration,
};

use axum::{
    Json, Router,
    body::Body,
    extract::{ConnectInfo, DefaultBodyLimit, State},
    http::{HeaderValue, header},
    response::Response,
    routing::post,
};
use tokio::{
    io::{AsyncRead, AsyncWrite, ReadBuf},
    net::{TcpListener, TcpStream},
    sync::{OwnedSemaphorePermit, Semaphore},
    task::JoinSet,
};
use tokio_rustls::{TlsAcceptor, server::TlsStream};
use tokio_util::sync::CancellationToken;
use tower::limit::ConcurrencyLimitLayer;

use crate::storage::ConnectionId;
use crate::{
    CommitAcknowledgement, Error, ErrorCode, Layer, ProjectId, Result,
    broker::{
        BrokerCommand, BrokerCommit, BrokerCoordinator, BrokerStateMachine, Delivery,
        PayloadRecord, QueueInfo, QueueServer, StreamOffset, StreamServer,
    },
    cypher::{Clause, Statement, parse},
    engine::{LayerScope, OperationScope, ProtocolScope, certificate_public_key_fingerprint},
    protocol::{
        BoltServer, QueryExecutor, QueryIngressAdmission, QueryRequest, QueryStreamEvent,
        QueryTransaction, ndjson_channel,
    },
};

use super::Database;

#[derive(Clone, Copy, Debug)]
pub(super) struct RemotePeer {
    pub _address: SocketAddr,
    pub certificate_fingerprint: [u8; 32],
    pub project: ProjectId,
    pub connection_id: ConnectionId,
}

pub(super) struct RemoteIo {
    stream: Pin<Box<TlsStream<TcpStream>>>,
    _permit: OwnedSemaphorePermit,
}

impl AsyncRead for RemoteIo {
    fn poll_read(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffer: &mut ReadBuf<'_>,
    ) -> Poll<std::io::Result<()>> {
        self.stream.as_mut().poll_read(context, buffer)
    }
}

impl AsyncWrite for RemoteIo {
    fn poll_write(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
        buffer: &[u8],
    ) -> Poll<std::io::Result<usize>> {
        self.stream.as_mut().poll_write(context, buffer)
    }

    fn poll_flush(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<std::io::Result<()>> {
        self.stream.as_mut().poll_flush(context)
    }

    fn poll_shutdown(
        mut self: Pin<&mut Self>,
        context: &mut Context<'_>,
    ) -> Poll<std::io::Result<()>> {
        self.stream.as_mut().poll_shutdown(context)
    }
}

pub(super) struct AuthenticatedTlsListener {
    listener: TcpListener,
    acceptor: TlsAcceptor,
    database: Database,
    protocol: ProtocolScope,
    handshake_timeout: Duration,
    permits: Arc<Semaphore>,
    handshakes: JoinSet<Result<(RemoteIo, RemotePeer)>>,
}

impl AuthenticatedTlsListener {
    pub async fn bind(
        address: SocketAddr,
        tls: Arc<rustls::ServerConfig>,
        database: Database,
        protocol: ProtocolScope,
        handshake_timeout: Duration,
        maximum_connections: usize,
    ) -> Result<Self> {
        if handshake_timeout.is_zero() || maximum_connections == 0 {
            return Err(Error::invalid_data("invalid remote listener limits"));
        }
        Ok(Self {
            listener: TcpListener::bind(address).await?,
            acceptor: TlsAcceptor::from(tls),
            database,
            protocol,
            handshake_timeout,
            permits: Arc::new(Semaphore::new(maximum_connections)),
            handshakes: JoinSet::new(),
        })
    }

    async fn accept_authenticated(&mut self) -> (RemoteIo, RemotePeer) {
        loop {
            tokio::select! {
                accepted = self.listener.accept() => {
                    match accepted {
                        Ok((stream, address)) => {
                            let Ok(permit) = Arc::clone(&self.permits).try_acquire_owned() else {
                                drop(stream);
                                continue;
                            };
                            if stream.set_nodelay(true).is_err() {
                                continue;
                            }
                            let acceptor = self.acceptor.clone();
                            let database = self.database.clone();
                            let protocol = self.protocol;
                            let timeout = self.handshake_timeout;
                            self.handshakes.spawn(async move {
                                let stream = tokio::time::timeout(timeout, acceptor.accept(stream))
                                    .await
                                    .map_err(|_| Error::retryable(ErrorCode::DeadlineExceeded, "remote TLS handshake timed out", None))?
                                    .map_err(|error| Error::new(ErrorCode::AuthenticationFailed, error.to_string()))?;
                                let certificate = stream
                                    .get_ref()
                                    .1
                                    .peer_certificates()
                                    .and_then(|chain| chain.first())
                                    .ok_or_else(|| Error::new(ErrorCode::AuthenticationFailed, "remote client sent no certificate"))?;
                                let fingerprint = certificate_public_key_fingerprint(certificate.as_ref())?;
                                let credential = database.authenticate_client(fingerprint, protocol)?;
                                Ok((
                                    RemoteIo {
                                        stream: Box::pin(stream),
                                        _permit: permit,
                                    },
                                    RemotePeer {
                                        _address: address,
                                        certificate_fingerprint: fingerprint,
                                        project: credential.project_id,
                                        connection_id: ConnectionId::new(),
                                    },
                                ))
                            });
                        }
                        Err(error) => {
                            tracing::warn!(%error, "remote listener accept failed");
                            tokio::time::sleep(Duration::from_millis(100)).await;
                        }
                    }
                }
                completed = self.handshakes.join_next(), if !self.handshakes.is_empty() => {
                    match completed {
                        Some(Ok(Ok(connection))) => return connection,
                        Some(Ok(Err(error))) => tracing::warn!(code = ?error.code, message = %error.message, "remote authentication rejected"),
                        Some(Err(error)) => tracing::error!(%error, "remote handshake task failed"),
                        None => {}
                    }
                }
            }
        }
    }
}

impl axum::serve::Listener for AuthenticatedTlsListener {
    type Io = RemoteIo;
    type Addr = RemotePeer;

    fn accept(&mut self) -> impl Future<Output = (Self::Io, Self::Addr)> + Send {
        self.accept_authenticated()
    }

    fn local_addr(&self) -> std::io::Result<Self::Addr> {
        Ok(RemotePeer {
            _address: self.listener.local_addr()?,
            certificate_fingerprint: [0_u8; 32],
            project: ProjectId(uuid::Uuid::nil()),
            connection_id: ConnectionId::new(),
        })
    }
}

impl<'a>
    axum::extract::connect_info::Connected<
        axum::serve::IncomingStream<'a, AuthenticatedTlsListener>,
    > for RemotePeer
{
    fn connect_info(stream: axum::serve::IncomingStream<'a, AuthenticatedTlsListener>) -> Self {
        *stream.remote_addr()
    }
}

#[derive(Clone)]
struct RemoteQueryState {
    database: Database,
    query_admission: Arc<QueryIngressAdmission>,
}

pub(super) async fn run_remote_query(
    listener: AuthenticatedTlsListener,
    database: Database,
    shutdown: CancellationToken,
) -> Result<()> {
    let app = remote_query_application(database)?;
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<RemotePeer>(),
    )
    .with_graceful_shutdown(shutdown.cancelled_owned())
    .await
    .map_err(Error::from)
}

fn remote_query_application(database: Database) -> Result<Router> {
    Ok(Router::new()
        .route("/api/query", post(remote_query))
        .layer(DefaultBodyLimit::max(24 * 1024 * 1024))
        .layer(ConcurrencyLimitLayer::new(256))
        .with_state(RemoteQueryState {
            database,
            query_admission: Arc::new(QueryIngressAdmission::new(256, 256 * 1024 * 1024)?),
        }))
}

async fn remote_query(
    ConnectInfo(peer): ConnectInfo<RemotePeer>,
    State(state): State<RemoteQueryState>,
    Json(mut request): Json<QueryRequest>,
) -> Response {
    request.cancellation = CancellationToken::new();
    request.deadline = Some(std::time::Instant::now() + Duration::from_secs(120));
    request.connection_id = peer.connection_id;
    let ingress = match state.query_admission.try_admit(&request) {
        Ok(permit) => permit,
        Err(error) => {
            let (sender, body) = ndjson_channel::<QueryStreamEvent>(1);
            let _ = sender
                .send(QueryStreamEvent::Error {
                    request_id: request.request_id,
                    code: error.code,
                    message: error.message.to_string(),
                    retryable: error.retryable,
                    retry_after_ms: error.retry_after_ms,
                })
                .await;
            return ndjson_response(body.into_body());
        }
    };
    let cancellation = request.cancellation.clone();
    let request_id = request.request_id;
    let executor = match ScopedQueryExecutor::new(
        state.database,
        peer.certificate_fingerprint,
        ProtocolScope::QUERY_HTTP,
        peer.project,
    ) {
        Ok(executor) => executor,
        Err(error) => {
            let (sender, body) = ndjson_channel::<QueryStreamEvent>(1);
            let _ = sender
                .send(QueryStreamEvent::Error {
                    request_id,
                    code: error.code,
                    message: error.message.to_string(),
                    retryable: error.retryable,
                    retry_after_ms: error.retry_after_ms,
                })
                .await;
            return ndjson_response(body.into_body());
        }
    };
    let (sender, body) = ndjson_channel::<QueryStreamEvent>(8);
    let close_monitor = sender.clone();
    tokio::spawn(async move {
        close_monitor.closed().await;
        cancellation.cancel();
    });
    tokio::task::spawn_blocking(move || {
        let _ingress = ingress;
        if let Err(error) = executor.execute(request, &mut |event| sender.blocking_send(event)) {
            let _ = sender.blocking_send(QueryStreamEvent::Error {
                request_id,
                code: error.code,
                message: error.message.to_string(),
                retryable: error.retryable,
                retry_after_ms: error.retry_after_ms,
            });
        }
    });
    ndjson_response(body.into_body())
}

fn ndjson_response(body: Body) -> Response {
    let mut response = Response::new(body);
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/x-ndjson; charset=utf-8"),
    );
    response
}

#[derive(Clone)]
struct ScopedQueryExecutor {
    database: Database,
    fingerprint: [u8; 32],
    protocol: ProtocolScope,
    project: ProjectId,
}

impl ScopedQueryExecutor {
    fn new(
        database: Database,
        fingerprint: [u8; 32],
        protocol: ProtocolScope,
        project: ProjectId,
    ) -> Result<Self> {
        let credential = database.authenticate_client(fingerprint, protocol)?;
        if credential.project_id != project {
            return Err(Error::new(
                ErrorCode::AuthenticationFailed,
                "remote credential project changed during connection setup",
            ));
        }
        Ok(Self {
            database,
            fingerprint,
            protocol,
            project,
        })
    }

    fn scope(&self, mut request: QueryRequest) -> Result<QueryRequest> {
        if request
            .project_id
            .is_some_and(|project| project != self.project)
        {
            return Err(Error::new(
                ErrorCode::AuthorizationDenied,
                "remote query selected another project",
            ));
        }
        let parsed = parse(&request.query)?;
        let selected_project = parsed
            .project
            .as_deref()
            .map(|name| self.database.resolve_project_name(name))
            .transpose()?;
        validate_remote_project(selected_project, self.project)?;
        let operations = required_query_operations(&parsed.statement)?;
        let layers = required_query_layers(&parsed)?;
        for operation in operations {
            self.database.authorize_client(
                self.fingerprint,
                self.project,
                self.protocol,
                operation,
                layers,
            )?;
        }
        request.project_id = Some(self.project);
        Ok(request)
    }

    fn reauthenticate(&self) -> Result<()> {
        let credential = self
            .database
            .authenticate_client(self.fingerprint, self.protocol)?;
        if credential.project_id != self.project {
            return Err(Error::new(
                ErrorCode::AuthenticationFailed,
                "remote credential project changed",
            ));
        }
        Ok(())
    }
}

fn validate_remote_project(selected: Option<ProjectId>, allowed: ProjectId) -> Result<()> {
    if selected.is_some_and(|project| project != allowed) {
        return Err(Error::new(
            ErrorCode::AuthorizationDenied,
            "remote query selected another project",
        ));
    }
    Ok(())
}

fn required_query_layers(query: &crate::cypher::Query) -> Result<LayerScope> {
    let mut layers = LayerScope::empty();
    if query.read_layers.contains_layer(Layer::Observed) {
        layers |= LayerScope::OBSERVED;
    }
    if query.read_layers.contains_layer(Layer::Knowledge) {
        layers |= LayerScope::KNOWLEDGE;
    }
    if query.read_layers.contains_layer(Layer::Workspace) {
        layers |= LayerScope::WORKSPACE;
    }
    if layers.is_empty() {
        return Err(Error::new(
            ErrorCode::LayerNotAllowed,
            "remote query requested an unavailable layer",
        ));
    }
    Ok(layers)
}

impl QueryExecutor for ScopedQueryExecutor {
    fn resolve_project(&self, selector: &str) -> Result<ProjectId> {
        let project = self.database.resolve_project(selector)?;
        if project != self.project {
            return Err(Error::new(
                ErrorCode::AuthorizationDenied,
                "remote Bolt connection selected another project",
            ));
        }
        Ok(project)
    }

    fn execute(
        &self,
        request: QueryRequest,
        emit: &mut dyn FnMut(QueryStreamEvent) -> Result<()>,
    ) -> Result<()> {
        self.database.execute(self.scope(request)?, emit)
    }

    fn begin(
        &self,
        project: Option<ProjectId>,
        bookmark: Option<crate::Bookmark>,
        consistency: CommitAcknowledgement,
    ) -> Result<Box<dyn QueryTransaction>> {
        if project.is_some_and(|requested| requested != self.project) {
            return Err(Error::new(
                ErrorCode::AuthorizationDenied,
                "remote transaction selected another project",
            ));
        }
        self.reauthenticate()?;
        Ok(Box::new(ScopedQueryTransaction {
            inner: self
                .database
                .begin(Some(self.project), bookmark, consistency)?,
            scope: self.clone(),
        }))
    }

    fn begin_on_connection(
        &self,
        connection: ConnectionId,
        project: Option<ProjectId>,
        bookmark: Option<crate::Bookmark>,
        consistency: CommitAcknowledgement,
    ) -> Result<Box<dyn QueryTransaction>> {
        if project.is_some_and(|requested| requested != self.project) {
            return Err(Error::new(
                ErrorCode::AuthorizationDenied,
                "remote transaction selected another project",
            ));
        }
        self.reauthenticate()?;
        Ok(Box::new(ScopedQueryTransaction {
            inner: self.database.begin_on_connection(
                connection,
                Some(self.project),
                bookmark,
                consistency,
            )?,
            scope: self.clone(),
        }))
    }
}

struct ScopedQueryTransaction {
    inner: Box<dyn QueryTransaction>,
    scope: ScopedQueryExecutor,
}

impl QueryTransaction for ScopedQueryTransaction {
    fn run(
        &mut self,
        request: QueryRequest,
        emit: &mut dyn FnMut(QueryStreamEvent) -> Result<()>,
    ) -> Result<()> {
        self.inner.run(self.scope.scope(request)?, emit)
    }

    fn commit(self: Box<Self>) -> Result<crate::Bookmark> {
        self.scope.reauthenticate()?;
        self.inner.commit()
    }

    fn rollback(self: Box<Self>) -> Result<()> {
        self.inner.rollback()
    }
}

fn required_query_operations(statement: &Statement) -> Result<Vec<OperationScope>> {
    match statement {
        Statement::ImportDataset { .. }
        | Statement::CreateProject { .. }
        | Statement::AlterProjectRename { .. }
        | Statement::DropProject { .. }
        | Statement::ShowProjects => Err(Error::new(
            ErrorCode::AuthorizationDenied,
            "remote credentials cannot administer the project catalog",
        )),
        Statement::CreateIndex(_)
        | Statement::CreateConstraint(_)
        | Statement::RebuildIndex { .. }
        | Statement::DropIndex { .. }
        | Statement::DropConstraint { .. }
        | Statement::DeclareTemporal(_)
        | Statement::CreateRollup(_)
        | Statement::CreateEmbedding(_) => Ok(vec![OperationScope::SCHEMA]),
        Statement::CheckReadOnly
        | Statement::ShowIndexes
        | Statement::ShowConstraints
        | Statement::ShowTopics
        | Statement::ShowQueues
        | Statement::ShowExchanges
        | Statement::ShowConsumerLag => Ok(vec![OperationScope::READ]),
        Statement::CreateTopic { .. }
        | Statement::AlterTopicRetention { .. }
        | Statement::DropTopic { .. }
        | Statement::ClearTopic { .. }
        | Statement::CreateQueue { .. }
        | Statement::AlterQueueRetention { .. }
        | Statement::DropQueue { .. }
        | Statement::PurgeQueue { .. }
        | Statement::CreateExchange { .. }
        | Statement::DropExchange { .. }
        | Statement::BindQueue { .. }
        | Statement::UnbindQueue { .. } => Ok(vec![OperationScope::BROKER]),
        Statement::Query(body) => {
            let writes = body
                .clauses
                .iter()
                .chain(body.unions.iter().flat_map(|branch| branch.body.iter()))
                .any(|clause| {
                    matches!(
                        clause,
                        Clause::Create(_)
                            | Clause::Merge { .. }
                            | Clause::Set(_)
                            | Clause::Remove(_)
                            | Clause::Delete { .. }
                    )
                });
            let reads_existing = body
                .clauses
                .iter()
                .chain(body.unions.iter().flat_map(|branch| branch.body.iter()))
                .any(|clause| {
                    matches!(
                        clause,
                        Clause::Match { .. }
                            | Clause::Merge { .. }
                            | Clause::History(_)
                            | Clause::Search(_)
                            | Clause::Call(_)
                    )
                });
            Ok(if writes && reads_existing {
                vec![OperationScope::READ, OperationScope::WRITE]
            } else if writes {
                vec![OperationScope::WRITE]
            } else {
                vec![OperationScope::READ]
            })
        }
    }
}

#[derive(Clone)]
struct ScopedBrokerCoordinator {
    database: Database,
    fingerprint: [u8; 32],
    protocol: ProtocolScope,
    project: ProjectId,
}

impl ScopedBrokerCoordinator {
    fn authorize(&self, project: ProjectId) -> Result<()> {
        if project != self.project {
            return Err(Error::new(
                ErrorCode::AuthorizationDenied,
                "broker operation crossed its credential project",
            ));
        }
        self.database.authorize_client(
            self.fingerprint,
            self.project,
            self.protocol,
            OperationScope::BROKER,
            LayerScope::OBSERVED,
        )?;
        Ok(())
    }
}

impl BrokerCoordinator for ScopedBrokerCoordinator {
    fn subscribe_changes(&self) -> tokio::sync::watch::Receiver<u64> {
        self.database.subscribe_changes()
    }

    fn submit(&self, command: BrokerCommand, wait: CommitAcknowledgement) -> Result<BrokerCommit> {
        self.authorize(broker_command_project(&command))?;
        self.database.submit(command, wait)
    }

    fn submit_with_timeout(
        &self,
        command: BrokerCommand,
        wait: CommitAcknowledgement,
        timeout_millis: u32,
    ) -> Result<BrokerCommit> {
        self.authorize(broker_command_project(&command))?;
        self.database
            .submit_with_timeout(command, wait, timeout_millis)
    }

    fn submit_from(
        &self,
        connection: ConnectionId,
        command: BrokerCommand,
        wait: CommitAcknowledgement,
    ) -> Result<BrokerCommit> {
        self.authorize(broker_command_project(&command))?;
        self.database.submit_from(connection, command, wait)
    }

    fn submit_with_timeout_from(
        &self,
        connection: ConnectionId,
        command: BrokerCommand,
        wait: CommitAcknowledgement,
        timeout_millis: u32,
    ) -> Result<BrokerCommit> {
        self.authorize(broker_command_project(&command))?;
        self.database
            .submit_with_timeout_from(connection, command, wait, timeout_millis)
    }

    fn snapshot(&self) -> Result<BrokerStateMachine> {
        self.authorize(self.project)?;
        self.database.snapshot()
    }

    fn fetch_partition(
        &self,
        project: ProjectId,
        topic: &str,
        partition: i32,
        offset: u64,
        maximum_bytes: usize,
    ) -> Result<Vec<(u64, Arc<PayloadRecord>)>> {
        self.authorize(project)?;
        self.database
            .fetch_partition(project, topic, partition, offset, maximum_bytes)
    }

    fn list_offset(
        &self,
        project: ProjectId,
        topic: &str,
        partition: i32,
        timestamp: i64,
    ) -> Result<Option<(u64, i64)>> {
        self.authorize(project)?;
        self.database
            .list_offset(project, topic, partition, timestamp)
    }

    fn queue_info(&self, project: ProjectId, name: &str) -> Result<Option<QueueInfo>> {
        self.authorize(project)?;
        self.database.queue_info(project, name)
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
        self.authorize(project)?;
        self.database
            .fetch_stream_queue(project, queue, offset, maximum, consumer, automatic_ack)
    }
}

fn broker_command_project(command: &BrokerCommand) -> ProjectId {
    match command {
        BrokerCommand::CreateTopic { project, .. }
        | BrokerCommand::SetTopicRetention { project, .. }
        | BrokerCommand::DeleteTopic { project, .. }
        | BrokerCommand::ClearTopic { project, .. }
        | BrokerCommand::CreateExchange { project, .. }
        | BrokerCommand::CreateQueue { project, .. }
        | BrokerCommand::SetQueueRetention { project, .. }
        | BrokerCommand::BindQueue { project, .. }
        | BrokerCommand::UnbindQueue { project, .. }
        | BrokerCommand::PurgeQueue { project, .. }
        | BrokerCommand::DeleteQueue { project, .. }
        | BrokerCommand::DeleteExchange { project, .. }
        | BrokerCommand::RegisterConsumer { project, .. }
        | BrokerCommand::UnregisterConsumer { project, .. }
        | BrokerCommand::ReleaseConnection { project, .. }
        | BrokerCommand::PublishKafkaBatch { project, .. }
        | BrokerCommand::PublishAmqp { project, .. }
        | BrokerCommand::PublishAmqpBatch { project, .. }
        | BrokerCommand::PublishAmqpUniformBatch { project, .. }
        | BrokerCommand::CommitOffset { project, .. }
        | BrokerCommand::JoinGroup { project, .. }
        | BrokerCommand::SyncGroup { project, .. }
        | BrokerCommand::LeaveGroup { project, .. }
        | BrokerCommand::HeartbeatGroup { project, .. }
        | BrokerCommand::Ack { project, .. }
        | BrokerCommand::Nack { project, .. }
        | BrokerCommand::DeliverQueue { project, .. }
        | BrokerCommand::RenewDeliveryLease { project, .. }
        | BrokerCommand::ReleaseDeliveryLease { project, .. }
        | BrokerCommand::Retain { project, .. } => *project,
    }
}

pub(super) async fn run_remote_bolt(
    mut listener: AuthenticatedTlsListener,
    database: Database,
    shutdown: CancellationToken,
) -> Result<()> {
    let mut connections = JoinSet::new();
    loop {
        tokio::select! {
            () = shutdown.cancelled() => break,
            accepted = listener.accept_authenticated() => {
                let (stream, peer) = accepted;
                let executor = match ScopedQueryExecutor::new(
                    database.clone(),
                    peer.certificate_fingerprint,
                    ProtocolScope::BOLT,
                    peer.project,
                ) {
                    Ok(executor) => Arc::new(executor),
                    Err(error) => {
                        tracing::warn!(code = ?error.code, message = %error.message, "remote Bolt credential changed before session start");
                        continue;
                    }
                };
                connections.spawn(BoltServer::serve_authenticated_transport(stream, executor));
            }
            completed = connections.join_next(), if !connections.is_empty() => log_connection_result("Bolt", completed),
        }
    }
    connections.abort_all();
    while connections.join_next().await.is_some() {}
    Ok(())
}

pub(super) async fn run_remote_kafka(
    mut listener: AuthenticatedTlsListener,
    database: Database,
    advertised_host: String,
    advertised_port: u16,
    shutdown: CancellationToken,
) -> Result<()> {
    let mut connections = JoinSet::new();
    loop {
        tokio::select! {
            () = shutdown.cancelled() => break,
            accepted = listener.accept_authenticated() => {
                let (stream, peer) = accepted;
                let coordinator: Arc<dyn BrokerCoordinator> = Arc::new(ScopedBrokerCoordinator {
                    database: database.clone(),
                    fingerprint: peer.certificate_fingerprint,
                    protocol: ProtocolScope::KAFKA,
                    project: peer.project,
                });
                connections.spawn(StreamServer::serve_authenticated_transport(
                    stream,
                    peer.project,
                    coordinator,
                    advertised_host.clone(),
                    advertised_port,
                ));
            }
            completed = connections.join_next(), if !connections.is_empty() => log_connection_result("Kafka", completed),
        }
    }
    connections.abort_all();
    while connections.join_next().await.is_some() {}
    Ok(())
}

pub(super) async fn run_remote_amqp(
    mut listener: AuthenticatedTlsListener,
    database: Database,
    shutdown: CancellationToken,
) -> Result<()> {
    let mut connections = JoinSet::new();
    loop {
        tokio::select! {
            () = shutdown.cancelled() => break,
            accepted = listener.accept_authenticated() => {
                let (stream, peer) = accepted;
                let coordinator: Arc<dyn BrokerCoordinator> = Arc::new(ScopedBrokerCoordinator {
                    database: database.clone(),
                    fingerprint: peer.certificate_fingerprint,
                    protocol: ProtocolScope::AMQP,
                    project: peer.project,
                });
                connections.spawn(QueueServer::serve_authenticated_transport(
                    stream,
                    peer.project,
                    coordinator,
                    shutdown.child_token(),
                ));
            }
            completed = connections.join_next(), if !connections.is_empty() => log_connection_result("AMQP", completed),
        }
    }
    connections.abort_all();
    while connections.join_next().await.is_some() {}
    Ok(())
}

fn log_connection_result(
    protocol: &'static str,
    completed: Option<std::result::Result<Result<()>, tokio::task::JoinError>>,
) {
    match completed {
        Some(Ok(Ok(()))) => {}
        Some(Ok(Err(error))) => {
            tracing::warn!(protocol, code = ?error.code, message = %error.message, "remote protocol connection closed")
        }
        Some(Err(error)) => tracing::error!(protocol, %error, "remote protocol task failed"),
        None => {}
    }
}

#[cfg(test)]
mod tests {
    use axum::http::{Request, StatusCode};
    use rcgen::{CertificateParams, KeyPair};
    use rustls::{ServerConfig, pki_types::PrivatePkcs8KeyDer};
    use tempfile::TempDir;
    use tokio::io::AsyncWriteExt as _;
    use tower::ServiceExt as _;

    use super::*;

    fn test_database() -> Result<(TempDir, Database)> {
        let directory = tempfile::tempdir()?;
        let identity = crate::engine::NodeIdentity::generate_genesis();
        let database = Database::open_backend(
            directory.path(),
            1024 * 1024,
            Duration::from_secs(1),
            identity.public(),
        )?;
        Ok((directory, database))
    }

    fn test_server_tls() -> Result<Arc<ServerConfig>> {
        let key = KeyPair::generate().map_err(|error| Error::internal(error.to_string()))?;
        let certificate = CertificateParams::new(vec!["localhost".to_owned()])
            .map_err(|error| Error::internal(error.to_string()))?
            .self_signed(&key)
            .map_err(|error| Error::internal(error.to_string()))?;
        let private_key = PrivatePkcs8KeyDer::from(key.serialize_der());
        let server = ServerConfig::builder()
            .with_no_client_auth()
            .with_single_cert(vec![certificate.der().clone()], private_key.into())
            .map_err(|error| Error::internal(error.to_string()))?;
        Ok(Arc::new(server))
    }

    #[test]
    fn mutating_queries_require_read_scope_when_they_read_existing_graph() -> Result<()> {
        let read_write = parse(
            "USE project USE LAYER OBSERVED WRITE LAYER OBSERVED \
             MATCH (n) SET n.value = 1 RETURN n",
        )?;
        assert_eq!(
            required_query_operations(&read_write.statement)?,
            vec![OperationScope::READ, OperationScope::WRITE]
        );

        let create_only = parse(
            "USE project USE LAYER OBSERVED WRITE LAYER OBSERVED \
             CREATE (n:Item {value: 1}) RETURN n",
        )?;
        assert_eq!(
            required_query_operations(&create_only.statement)?,
            vec![OperationScope::WRITE]
        );

        let read = parse("USE project USE LAYER OBSERVED MATCH (n) RETURN n")?;
        assert_eq!(
            required_query_operations(&read.statement)?,
            vec![OperationScope::READ]
        );
        Ok(())
    }

    #[test]
    fn remote_credentials_cannot_administer_the_project_catalog() -> Result<()> {
        let statement = parse("SHOW PROJECTS")?;
        assert_eq!(
            required_query_operations(&statement.statement)
                .err()
                .map(|error| error.code),
            Some(ErrorCode::AuthorizationDenied)
        );
        Ok(())
    }

    #[test]
    fn leading_use_and_layer_masks_cannot_escape_the_credential_scope() -> Result<()> {
        let allowed = ProjectId(uuid::Uuid::new_v4());
        validate_remote_project(None, allowed)?;
        validate_remote_project(Some(allowed), allowed)?;
        assert_eq!(
            validate_remote_project(Some(ProjectId(uuid::Uuid::new_v4())), allowed)
                .err()
                .map(|error| error.code),
            Some(ErrorCode::AuthorizationDenied)
        );

        let authority = parse("USE project MATCH (n) RETURN n")?;
        assert_eq!(
            required_query_layers(&authority)?,
            LayerScope::OBSERVED | LayerScope::KNOWLEDGE
        );
        let knowledge =
            parse("USE project USE LAYER KNOWLEDGE WRITE LAYER KNOWLEDGE MATCH (n) RETURN n")?;
        assert_eq!(required_query_layers(&knowledge)?, LayerScope::KNOWLEDGE);
        Ok(())
    }

    #[tokio::test]
    async fn remote_http_rejects_unsupported_routes() -> Result<()> {
        let (_directory, database) = test_database()?;
        let application = remote_query_application(database)?;
        for path in ["/query", "/unsupported"] {
            let response = application
                .clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri(path)
                        .body(Body::empty())
                        .map_err(|error| Error::internal(error.to_string()))?,
                )
                .await
                .map_err(|error| Error::internal(error.to_string()))?;
            assert_eq!(response.status(), StatusCode::NOT_FOUND, "{path}");
        }
        Ok(())
    }

    #[tokio::test]
    async fn capacity_and_authentication_failures_do_not_stop_the_listener() -> Result<()> {
        let (_directory, database) = test_database()?;
        let listener = AuthenticatedTlsListener::bind(
            "127.0.0.1:0"
                .parse()
                .map_err(|error: std::net::AddrParseError| Error::internal(error.to_string()))?,
            test_server_tls()?,
            database,
            ProtocolScope::BOLT,
            Duration::from_millis(100),
            1,
        )
        .await;
        let mut listener = match listener {
            Ok(listener) => listener,
            Err(error)
                if error.code == ErrorCode::Io
                    && (error.message.contains("Operation not permitted")
                        || error.message.contains("Permission denied")) =>
            {
                return Ok(());
            }
            Err(error) => return Err(error),
        };
        let address = listener.listener.local_addr()?;
        let held = Arc::clone(&listener.permits)
            .acquire_owned()
            .await
            .map_err(|error| Error::internal(error.to_string()))?;
        let accept = tokio::spawn(async move { listener.accept_authenticated().await });

        let first = TcpStream::connect(address).await?;
        drop(first);
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(!accept.is_finished());
        drop(held);

        for _ in 0..2 {
            let mut invalid = TcpStream::connect(address).await?;
            invalid.write_all(b"not-tls").await?;
            invalid.shutdown().await?;
            tokio::time::sleep(Duration::from_millis(20)).await;
            assert!(!accept.is_finished());
        }

        accept.abort();
        let result = accept.await;
        assert!(result.is_err_and(|error| error.is_cancelled()));
        Ok(())
    }
}
