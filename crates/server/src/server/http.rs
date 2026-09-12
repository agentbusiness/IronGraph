use std::{env, net::SocketAddr, path::PathBuf, process::Command, sync::Arc, time::Duration};

use axum::{
    Router,
    body::Body,
    extract::{DefaultBodyLimit, Path, State},
    http::{HeaderValue, Request, StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Redirect, Response},
    routing::{get, post},
};
use rust_embed::RustEmbed;
use tokio::{net::TcpListener, task::JoinSet};
use tokio_util::sync::CancellationToken;
use tower::limit::ConcurrencyLimitLayer;

use super::{
    Database,
    remote::{
        AuthenticatedTlsListener, run_remote_amqp, run_remote_bolt, run_remote_kafka,
        run_remote_query,
    },
};
use crate::{
    Bookmark, CommitAcknowledgement, Error, ErrorCode, ProjectId, Result,
    broker::{BrokerCommand, BrokerCoordinator, QueueServer, StreamServer},
    config::Config,
    engine::{
        ExecutionClass, SingleNodeBootstrapConfig, WriteStorageLimits, load_existing_node_identity,
        load_or_generate_genesis_identity, open_standalone,
    },
    gpu::{
        BackendKind, DeviceMemoryGovernor, ResolvedComputeDevice,
        create_execution_backend_with_governor,
    },
    protocol::{
        BoltServer, QueryExecutor, QueryIngressAdmission, QueryRequest, QueryStreamEvent,
        ndjson_channel,
    },
};

#[derive(RustEmbed)]
#[folder = "../../web/dist"]
struct WebAssets;

#[derive(Clone)]
struct AppState {
    database: Database,
    query_admission: Arc<QueryIngressAdmission>,
    maximum_result_bytes: u64,
}

/// Opens durable state, loads the text encoder, and runs the standalone database protocols.
pub async fn run(config: Config) -> Result<()> {
    config.validate()?;
    let existing_identity = load_existing_node_identity(&config.data_dir)?;
    if existing_identity.is_none() {
        let _ = load_or_generate_genesis_identity(&config.data_dir)?;
    }

    let execution_device = configured_execution_device(&config)?;
    let memory_governor = DeviceMemoryGovernor::new(
        config.device_memory_limit_bytes,
        config.device_reserved_bytes,
    );
    let execution_backend =
        create_execution_backend_with_governor(execution_device, memory_governor.clone())?;
    let bootstrap_options = SingleNodeBootstrapConfig {
        execution_class: execution_class(execution_backend.kind()),
        startup_timeout: config.startup_timeout(),
        storage_limits: WriteStorageLimits {
            max_log_record_bytes: config.wal_max_record_bytes,
            max_log_entries_per_read: 4_096,
            max_snapshot_bytes: 64 * 1024 * 1024 * 1024,
        },
    };
    let max_write_bytes = config.max_write_bytes;
    let request_timeout = config.request_timeout();
    let boot = open_standalone(
        &config.data_dir,
        bootstrap_options,
        move |directory, identity| {
            Ok(Arc::new(Database::open_backend(
                directory,
                max_write_bytes,
                request_timeout,
                identity,
            )?))
        },
    )
    .await?;
    boot.backend()
        .bind_runtime(Arc::downgrade(boot.runtime()))?;
    boot.backend().bind_execution_backend(execution_backend)?;
    let database = boot.backend().as_ref().clone();

    let snapshot_directory = config.data_dir.join("standalone-snapshots");
    let recovered = database.standalone_recover(&snapshot_directory).await?;
    // The snapshot restores a prefix of the committed history; mutations written after it exist
    // only in the standalone WAL. Replaying must finish before any listener or maintenance task
    // can submit new writes, or the first write reuses an index the WAL already holds and the
    // durable write path shuts down.
    let replayed = boot.runtime().replay_standalone_wal().await?;
    if let Some(bookmark) = recovered {
        tracing::info!(
            index = bookmark.index,
            replayed_through = replayed.index,
            "recovered standalone snapshot and WAL"
        );
    } else if replayed != Bookmark::default() {
        tracing::info!(
            replayed_through = replayed.index,
            "replayed standalone WAL without a snapshot"
        );
    }

    let embedding_device = config.embedding_device();
    let embedding_governor = memory_governor.clone();
    let embedding_database = database.clone();
    let embedding = tokio::task::spawn_blocking(move || {
        let artifacts = crate::embeddings::ensure_default_embedding_model()?;
        let embedding = Arc::new(crate::embeddings::LocalEmbeddingModel::load(
            artifacts,
            embedding_device,
        )?);
        embedding.bind_memory_governor(embedding_governor)?;
        embedding.warm_up()?;
        embedding_database.bind_text_embedding(embedding.clone())?;
        Ok::<_, Error>(embedding)
    })
    .await
    .map_err(|error| Error::internal(format!("embedding startup task failed: {error}")))??;

    let shutdown = CancellationToken::new();
    let snapshot_task = spawn_snapshot_maintenance(
        database.clone(),
        Arc::clone(boot.runtime()),
        snapshot_directory,
        shutdown.clone(),
    );

    let broker_project = config.broker_project.map(ProjectId);
    if let Some(project) = broker_project {
        database.ensure_project(project, CommitAcknowledgement::Published)?;
    }

    let state = AppState {
        database: database.clone(),
        query_admission: Arc::new(QueryIngressAdmission::new(256, 256 * 1024 * 1024)?),
        maximum_result_bytes: u64::try_from(config.max_result_bytes).unwrap_or(u64::MAX),
    };
    let http_listener = TcpListener::bind(config.http_addr).await?;
    let local_address = http_listener.local_addr()?;
    let app = Router::new()
        .route("/api/query", post(query))
        .route("/system/local-ai-integrations", get(local_ai_integrations))
        .route(
            "/system/local-ai-integrations/{host}/{action}",
            post(change_local_ai_integration),
        )
        .route("/", get(web_root))
        .fallback(static_asset)
        .layer(DefaultBodyLimit::max(24 * 1024 * 1024))
        .layer(ConcurrencyLimitLayer::new(config.max_connections))
        .layer(middleware::from_fn_with_state(
            local_address,
            guard_local_http,
        ))
        .with_state(state);

    let remote_tls_paths = remote_tls_paths(&config)?;
    let remote_query = bind_remote_listener(
        config.remote_query_addr,
        remote_tls_paths,
        &database,
        crate::engine::ProtocolScope::QUERY_HTTP,
        &config,
        vec![b"h2".to_vec(), b"http/1.1".to_vec()],
    )
    .await?;
    let remote_bolt = bind_remote_listener(
        config.remote_bolt_addr,
        remote_tls_paths,
        &database,
        crate::engine::ProtocolScope::BOLT,
        &config,
        Vec::new(),
    )
    .await?;
    let remote_kafka = bind_remote_listener(
        config.remote_stream_addr,
        remote_tls_paths,
        &database,
        crate::engine::ProtocolScope::KAFKA,
        &config,
        Vec::new(),
    )
    .await?;
    let remote_amqp = bind_remote_listener(
        config.remote_queue_addr,
        remote_tls_paths,
        &database,
        crate::engine::ProtocolScope::AMQP,
        &config,
        Vec::new(),
    )
    .await?;

    let signal_shutdown = shutdown.clone();
    let signal = tokio::spawn(async move {
        await_stop_signal().await;
        signal_shutdown.cancel();
    });

    let query_executor: Arc<dyn QueryExecutor> = Arc::new(database.clone());
    let broker: Arc<dyn BrokerCoordinator> = Arc::new(database.clone());
    let mut components = JoinSet::new();
    let http_shutdown = shutdown.clone();
    components.spawn(run_component("HTTP", shutdown.clone(), async move {
        axum::serve(http_listener, app)
            .with_graceful_shutdown(http_shutdown.cancelled_owned())
            .await
            .map_err(Error::from)
    }));
    components.spawn(run_component(
        "Bolt",
        shutdown.clone(),
        BoltServer::new(config.bolt_addr, query_executor).run(shutdown.clone()),
    ));
    if let Some(listener) = remote_query {
        components.spawn(run_component(
            "remote query mTLS",
            shutdown.clone(),
            run_remote_query(listener, database.clone(), shutdown.clone()),
        ));
    }
    if let Some(listener) = remote_bolt {
        components.spawn(run_component(
            "remote Bolt mTLS",
            shutdown.clone(),
            run_remote_bolt(listener, database.clone(), shutdown.clone()),
        ));
    }
    if let (Some(listener), Some(address)) = (remote_kafka, config.remote_stream_addr) {
        components.spawn(run_component(
            "remote Kafka mTLS",
            shutdown.clone(),
            run_remote_kafka(
                listener,
                database.clone(),
                address.ip().to_string(),
                address.port(),
                shutdown.clone(),
            ),
        ));
    }
    if let Some(listener) = remote_amqp {
        components.spawn(run_component(
            "remote AMQP mTLS",
            shutdown.clone(),
            run_remote_amqp(listener, database.clone(), shutdown.clone()),
        ));
    }
    if let Some(project) = broker_project {
        components.spawn(run_component(
            "Kafka",
            shutdown.clone(),
            StreamServer::new(config.stream_addr, project, Arc::clone(&broker))
                .run(shutdown.clone()),
        ));
        components.spawn(run_component(
            "AMQP",
            shutdown.clone(),
            QueueServer::new(config.queue_addr, project, Arc::clone(&broker)).run(shutdown.clone()),
        ));
    }
    components.spawn(run_component(
        "broker retention",
        shutdown.clone(),
        broker_retention(
            Arc::clone(&broker),
            config.broker_retention_interval(),
            shutdown.clone(),
        ),
    ));
    components.spawn(run_component(
        "vector maintenance",
        shutdown.clone(),
        vector_maintenance(database.clone(), shutdown.clone()),
    ));
    components.spawn(run_component(
        "resident graph maintenance",
        shutdown.clone(),
        resident_maintenance(database.clone(), shutdown.clone()),
    ));
    components.spawn(run_component(
        "optimizer statistics maintenance",
        shutdown.clone(),
        statistics_maintenance(database.clone(), shutdown.clone()),
    ));

    let outcome = tokio::select! {
        () = shutdown.cancelled() => Ok(()),
        completed = components.join_next() => match completed {
            Some(Ok(result)) => result,
            Some(Err(error)) => Err(Error::internal(format!("server component panicked: {error}"))),
            None => Err(Error::internal("server has no running components")),
        },
    };
    shutdown.cancel();
    signal.abort();
    while let Some(completed) = components.join_next().await {
        if let Err(error) = completed {
            tracing::warn!(%error, "server component did not shut down cleanly");
        }
    }
    snapshot_task.abort();
    let _ = snapshot_task.await;
    drop(embedding);
    outcome.and(boot.runtime().shutdown().await)
}

/// Loopback transport alone does not establish a browser's authority to use this listener.
async fn guard_local_http(
    State(address): State<SocketAddr>,
    request: Request<Body>,
    next: Next,
) -> Response {
    if !local_http_request_allowed(&request, address) {
        return (
            StatusCode::FORBIDDEN,
            "Local HTTP authority or origin is not allowed",
        )
            .into_response();
    }
    next.run(request).await
}

fn local_http_request_allowed(request: &Request<Body>, address: SocketAddr) -> bool {
    use axum::http::uri::Authority;

    let headers = request.headers();
    let mut hosts = headers.get_all(header::HOST).iter();
    let host = match hosts.next() {
        Some(value) => value
            .to_str()
            .ok()
            .and_then(|value| value.parse::<Authority>().ok()),
        None => request.uri().authority().cloned(),
    };
    let Some(host) = host else { return false };
    if hosts.next().is_some()
        || !local_http_authority(&host, address)
        || request
            .uri()
            .scheme_str()
            .is_some_and(|scheme| scheme != "http")
    {
        return false;
    }
    // HTTP/2 and absolute-form requests can carry authority separately from Host.
    if request
        .uri()
        .authority()
        .is_some_and(|authority| !same_http_authority(authority, &host))
    {
        return false;
    }
    let mut origins = headers.get_all(header::ORIGIN).iter();
    let Some(origin) = origins.next() else {
        return !(request.method() == axum::http::Method::POST
            && request
                .uri()
                .path()
                .starts_with("/system/local-ai-integrations/"));
    };
    if origins.next().is_some() {
        return false;
    }
    origin
        .to_str()
        .ok()
        .and_then(|value| value.strip_prefix("http://"))
        .and_then(|value| value.parse::<Authority>().ok())
        .is_some_and(|origin| {
            local_http_authority(&origin, address) && same_http_authority(&origin, &host)
        })
}

fn local_http_authority(authority: &axum::http::uri::Authority, address: SocketAddr) -> bool {
    let host = authority.host();
    !authority.as_str().contains('@')
        && local_http_port(authority) == Some(address.port())
        && (host.eq_ignore_ascii_case("localhost")
            || host
                .trim_start_matches('[')
                .trim_end_matches(']')
                .parse::<std::net::IpAddr>()
                .is_ok_and(|ip| ip == address.ip() && ip.is_loopback()))
}

fn same_http_authority(
    left: &axum::http::uri::Authority,
    right: &axum::http::uri::Authority,
) -> bool {
    left.host().eq_ignore_ascii_case(right.host())
        && local_http_port(left).is_some()
        && local_http_port(left) == local_http_port(right)
}

fn local_http_port(authority: &axum::http::uri::Authority) -> Option<u16> {
    match authority.as_str().strip_prefix(authority.host())? {
        "" => Some(80),
        suffix => suffix.strip_prefix(':')?.parse().ok(),
    }
}

const LOCAL_AI_HOSTS: &[&str] = &[
    "codex",
    "claude",
    "cursor",
    "copilot",
    "gemini",
    "kiro",
    "cline",
    "opencode",
    "roo",
    "continue",
    "windsurf",
    "zed",
    "lm-studio",
    "warp",
    "goose",
    "hermes",
    "pi",
    "openclaw",
    "unsloth",
];

async fn local_ai_integrations() -> Response {
    run_local_ai_command(vec!["integrations", "list", "--json"]).await
}

async fn change_local_ai_integration(Path((host, action)): Path<(String, String)>) -> Response {
    let command = match local_ai_operation(&host, &action) {
        Ok(command) => command,
        Err((status, detail)) => return local_ai_error(status, detail),
    };
    run_local_ai_command(vec!["integrations", command, &host, "--json"]).await
}

fn local_ai_operation(
    host: &str,
    action: &str,
) -> std::result::Result<&'static str, (StatusCode, String)> {
    if !LOCAL_AI_HOSTS.contains(&host) {
        return Err((
            StatusCode::NOT_FOUND,
            format!("unsupported AI host: {host}"),
        ));
    }
    match action {
        "install" | "repair" => Ok("install"),
        "update" => Ok("update"),
        _ => Err((
            StatusCode::BAD_REQUEST,
            format!("unsupported integration action: {action}"),
        )),
    }
}

async fn run_local_ai_command(arguments: Vec<&str>) -> Response {
    let arguments = arguments.into_iter().map(str::to_owned).collect::<Vec<_>>();
    match tokio::task::spawn_blocking(move || integration_command(&arguments)).await {
        Ok(Ok(value)) => axum::Json(value).into_response(),
        Ok(Err(detail)) => local_ai_error(StatusCode::INTERNAL_SERVER_ERROR, detail),
        Err(error) => local_ai_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("integration task failed: {error}"),
        ),
    }
}

fn integration_command(arguments: &[String]) -> std::result::Result<serde_json::Value, String> {
    let binary = integration_binary();
    let output = Command::new(&binary)
        .args(arguments)
        .output()
        .map_err(|error| format!("could not start {}: {error}", binary.display()))?;
    if !output.status.success() {
        let detail = String::from_utf8_lossy(&output.stderr);
        return Err(if detail.trim().is_empty() {
            format!("{} exited with {}", binary.display(), output.status)
        } else {
            detail.trim().to_owned()
        });
    }
    serde_json::from_slice(&output.stdout)
        .map_err(|error| format!("integration manager returned invalid JSON: {error}"))
}

fn integration_binary() -> PathBuf {
    if let Some(path) = env::var_os("IRONGRAPH_MCP_BINARY") {
        return PathBuf::from(path);
    }
    env::current_exe()
        .ok()
        .map(|path| path.with_file_name("irongraph-mcp"))
        .filter(|path| path.is_file())
        .unwrap_or_else(|| PathBuf::from("irongraph-mcp"))
}

fn local_ai_error(status: StatusCode, detail: String) -> Response {
    (status, axum::Json(serde_json::json!({ "detail": detail }))).into_response()
}

async fn web_root() -> Redirect {
    Redirect::permanent("/web/")
}

async fn query(
    State(state): State<AppState>,
    axum::Json(mut request): axum::Json<QueryRequest>,
) -> Response {
    request.cancellation = CancellationToken::new();
    request.deadline = Some(std::time::Instant::now() + Duration::from_secs(120));
    request.connection_id = crate::storage::ConnectionId::new();
    request.limits = request
        .limits
        .clamped_to_server_policy(state.maximum_result_bytes);
    let ingress = match state.query_admission.try_admit(&request) {
        Ok(permit) => permit,
        Err(error) => return query_error(request.request_id, error).await,
    };
    let request_id = request.request_id;
    let cancellation = request.cancellation.clone();
    let database = state.database;
    let (sender, body) = ndjson_channel::<QueryStreamEvent>(8);
    let body = body.cancel_on_drop(cancellation);
    tokio::task::spawn_blocking(move || {
        let _ingress = ingress;
        if let Err(error) = database.execute(request, &mut |event| sender.blocking_send(event)) {
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

async fn query_error(request_id: uuid::Uuid, error: Error) -> Response {
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
    ndjson_response(body.into_body())
}

fn ndjson_response(body: Body) -> Response {
    let mut response = Response::new(body);
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("application/x-ndjson; charset=utf-8"),
    );
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    response
}

async fn static_asset(request: Request<Body>) -> Response {
    if request.method() != axum::http::Method::GET && request.method() != axum::http::Method::HEAD {
        return StatusCode::METHOD_NOT_ALLOWED.into_response();
    }
    let path = match request.uri().path() {
        "/web" | "/web/" => "",
        path if path.starts_with("/web/") => &path[5..],
        _ => return StatusCode::NOT_FOUND.into_response(),
    };
    let requested = if path.is_empty() { "index.html" } else { path };
    let (asset_path, asset) = match WebAssets::get(requested) {
        Some(asset) => (requested, asset),
        None if !requested.contains('.') => match WebAssets::get("index.html") {
            Some(asset) => ("index.html", asset),
            None => return StatusCode::NOT_FOUND.into_response(),
        },
        None => return StatusCode::NOT_FOUND.into_response(),
    };
    let mut response = Response::new(Body::from(asset.data.into_owned()));
    response.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static(content_type(asset_path)),
    );
    response.headers_mut().insert(
        header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_static(
            "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; worker-src 'self'; font-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'",
        ),
    );
    response.headers_mut().insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    response.headers_mut().insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static(if asset_path == "index.html" {
            "no-cache"
        } else {
            "public, max-age=31536000, immutable"
        }),
    );
    response
}

fn content_type(path: &str) -> &'static str {
    if path.ends_with(".html") {
        "text/html; charset=utf-8"
    } else if path.ends_with(".js") {
        "text/javascript; charset=utf-8"
    } else if path.ends_with(".css") {
        "text/css; charset=utf-8"
    } else if path.ends_with(".svg") {
        "image/svg+xml"
    } else if path.ends_with(".json") || path.ends_with(".map") {
        "application/json"
    } else {
        "application/octet-stream"
    }
}

fn configured_execution_device(config: &Config) -> Result<ResolvedComputeDevice> {
    use crate::config::BackendSelection;
    match config.execution_backend {
        BackendSelection::Cpu => Ok(ResolvedComputeDevice {
            backend: BackendKind::Cpu,
            ordinal: 0,
        }),
        BackendSelection::Metal => Ok(ResolvedComputeDevice {
            backend: BackendKind::Metal,
            ordinal: config.execution_device,
        }),
        BackendSelection::Cuda => Ok(ResolvedComputeDevice {
            backend: BackendKind::Cuda,
            ordinal: config.execution_device,
        }),
        BackendSelection::Auto => automatic_execution_device(),
    }
}

fn automatic_execution_device() -> Result<ResolvedComputeDevice> {
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    {
        return Ok(ResolvedComputeDevice {
            backend: BackendKind::Metal,
            ordinal: 0,
        });
    }
    #[cfg(all(feature = "cuda", not(any(target_os = "macos", target_os = "ios"))))]
    {
        return Ok(ResolvedComputeDevice {
            backend: BackendKind::Cuda,
            ordinal: 0,
        });
    }
    #[allow(unreachable_code)]
    Err(Error::new(
        ErrorCode::GpuAdmissionFailure,
        "automatic database-device selection found no compiled accelerator; select CPU explicitly",
    ))
}

const fn execution_class(backend: BackendKind) -> ExecutionClass {
    match backend {
        BackendKind::Cpu => ExecutionClass::Cpu,
        BackendKind::Metal => ExecutionClass::Metal,
        BackendKind::Cuda => ExecutionClass::Cuda,
    }
}

type RemoteTlsPaths<'a> = Option<(
    &'a std::path::Path,
    &'a std::path::Path,
    &'a std::path::Path,
)>;

fn remote_tls_paths(config: &Config) -> Result<RemoteTlsPaths<'_>> {
    match (
        config.remote_tls_cert_path.as_deref(),
        config.remote_tls_key_path.as_deref(),
        config.remote_tls_trust_path.as_deref(),
    ) {
        (Some(certificate), Some(key), Some(trust)) => Ok(Some((certificate, key, trust))),
        (None, None, None) => Ok(None),
        _ => Err(Error::new(
            ErrorCode::AuthenticationFailed,
            "remote TLS certificate, key, and trust paths must be supplied together",
        )),
    }
}

async fn bind_remote_listener(
    address: Option<SocketAddr>,
    paths: RemoteTlsPaths<'_>,
    database: &Database,
    scope: crate::engine::ProtocolScope,
    config: &Config,
    alpn: Vec<Vec<u8>>,
) -> Result<Option<AuthenticatedTlsListener>> {
    let Some(address) = address else {
        return Ok(None);
    };
    let (certificate, key, trust) = paths.ok_or_else(|| {
        Error::new(
            ErrorCode::AuthenticationFailed,
            "remote listener has no TLS material",
        )
    })?;
    let tls = super::tls::load_remote_service_tls(certificate, key, trust, alpn)?;
    Ok(Some(
        AuthenticatedTlsListener::bind(
            address,
            tls,
            database.clone(),
            scope,
            config.protocol_handshake_timeout(),
            config.max_connections,
        )
        .await?,
    ))
}

fn spawn_snapshot_maintenance(
    database: Database,
    runtime: Arc<crate::engine::WriteRuntime>,
    directory: std::path::PathBuf,
    shutdown: CancellationToken,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let interval = Duration::from_secs(30);
        let mut ticker = tokio::time::interval_at(tokio::time::Instant::now() + interval, interval);
        ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        let mut last = None;
        loop {
            tokio::select! {
                () = shutdown.cancelled() => break,
                _ = ticker.tick() => {
                    if last == Some(database.bookmark().index) { continue; }
                    match database.standalone_snapshot(&directory).await {
                        Ok(bookmark) => match database.standalone_wal_compaction_bookmark(&directory, bookmark).await {
                            Ok(prefix) if prefix != Bookmark::default() => {
                                let runtime = Arc::clone(&runtime);
                                match tokio::task::spawn_blocking(move || runtime.compact_wal_through(prefix)).await {
                                    Ok(Ok(())) => last = Some(bookmark.index),
                                    Ok(Err(error)) => tracing::warn!(code = ?error.code, message = %error.message, "WAL compaction failed"),
                                    Err(error) => tracing::warn!(%error, "WAL compaction task failed"),
                                }
                            }
                            Ok(_) => last = Some(bookmark.index),
                            Err(error) => tracing::warn!(code = ?error.code, message = %error.message, "snapshot prefix selection failed"),
                        },
                        Err(error) => tracing::warn!(code = ?error.code, message = %error.message, "periodic snapshot failed"),
                    }
                }
            }
        }
    })
}

async fn broker_retention(
    broker: Arc<dyn BrokerCoordinator>,
    interval: Duration,
    shutdown: CancellationToken,
) -> Result<()> {
    let mut ticker = tokio::time::interval(interval);
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    loop {
        tokio::select! {
            () = shutdown.cancelled() => return Ok(()),
            _ = ticker.tick() => {
                let coordinator = Arc::clone(&broker);
                let outcome = tokio::task::spawn_blocking(move || {
                    coordinator.reclaim_payload_storage()?;
                    let resolved_time_ms = chrono::Utc::now().timestamp_millis();
                    for project in coordinator.projects_with_state()? {
                        coordinator.submit(
                            BrokerCommand::Retain { project, resolved_time_ms },
                            CommitAcknowledgement::Published,
                        )?;
                    }
                    Ok::<_, Error>(())
                }).await;
                // Background upkeep degrades and retries on its next tick; it never ends the
                // serving process.
                match outcome {
                    Ok(Ok(())) => {}
                    Ok(Err(error)) => tracing::warn!(
                        code = ?error.code,
                        message = %error.message,
                        "broker retention failed; retrying at the next interval"
                    ),
                    Err(error) => tracing::warn!(
                        %error,
                        "broker retention task failed; retrying at the next interval"
                    ),
                }
            }
        }
    }
}

async fn vector_maintenance(database: Database, shutdown: CancellationToken) -> Result<()> {
    let mut ticker = tokio::time::interval(Duration::from_millis(100));
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut failing = false;
    loop {
        tokio::select! {
            () = shutdown.cancelled() => return Ok(()),
            _ = ticker.tick() => {
                let database = database.clone();
                let outcome = tokio::task::spawn_blocking(move || {
                    database.maintain_local_artifact_readiness()
                })
                .await;
                // This ticks every 100ms, so a persistent fault is logged once per outage
                // instead of ten times a second.
                match outcome {
                    Ok(Ok(_)) => failing = false,
                    Ok(Err(error)) => {
                        if !failing {
                            tracing::warn!(
                                code = ?error.code,
                                message = %error.message,
                                "vector maintenance failed; retrying until it recovers"
                            );
                        }
                        failing = true;
                    }
                    Err(error) => {
                        if !failing {
                            tracing::warn!(
                                %error,
                                "vector maintenance task failed; retrying until it recovers"
                            );
                        }
                        failing = true;
                    }
                }
            }
        }
    }
}

async fn resident_maintenance(database: Database, shutdown: CancellationToken) -> Result<()> {
    let mut ticker = tokio::time::interval(Duration::from_secs(2));
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut previous = std::collections::HashMap::<ProjectId, u64>::new();
    loop {
        tokio::select! {
            () = shutdown.cancelled() => return Ok(()),
            _ = ticker.tick() => {
                let deferred = database.deferred_resident_projects();
                let mut current = std::collections::HashMap::new();
                for (project, revision) in deferred {
                    if previous.get(&project) == Some(&revision) {
                        let database = database.clone();
                        let outcome = tokio::task::spawn_blocking(move || {
                            database.republish_resident_project(project, revision)
                        })
                        .await;
                        match outcome {
                            Ok(Ok(_)) => {}
                            Ok(Err(error)) => tracing::warn!(
                                %project,
                                code = ?error.code,
                                message = %error.message,
                                "resident republish failed; retrying at the next interval"
                            ),
                            Err(error) => tracing::warn!(
                                %project,
                                %error,
                                "resident maintenance task failed; retrying at the next interval"
                            ),
                        }
                    }
                    current.insert(project, revision);
                }
                previous = current;
            }
        }
    }
}

async fn statistics_maintenance(database: Database, shutdown: CancellationToken) -> Result<()> {
    let mut ticker = tokio::time::interval(Duration::from_secs(2));
    ticker.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
    let mut previous = std::collections::HashMap::<ProjectId, u64>::new();
    loop {
        tokio::select! {
            () = shutdown.cancelled() => return Ok(()),
            _ = ticker.tick() => {
                let cold = database.cold_statistics_projects();
                let mut current = std::collections::HashMap::new();
                for (project, revision) in cold {
                    if previous.get(&project) == Some(&revision) {
                        let database = database.clone();
                        if let Err(error) = tokio::task::spawn_blocking(move || {
                            database.prewarm_optimizer_statistics(project, revision)
                        })
                        .await
                        {
                            tracing::warn!(
                                %project,
                                %error,
                                "statistics maintenance task failed; retrying at the next interval"
                            );
                        }
                    }
                    current.insert(project, revision);
                }
                previous = current;
            }
        }
    }
}

async fn run_component(
    name: &'static str,
    shutdown: CancellationToken,
    component: impl std::future::Future<Output = Result<()>>,
) -> Result<()> {
    let result = component.await;
    if shutdown.is_cancelled() {
        return result;
    }
    match result {
        Ok(()) => Err(Error::internal(format!(
            "{name} component exited before shutdown"
        ))),
        Err(error) => Err(error),
    }
}

async fn await_stop_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};
        let mut terminate = signal(SignalKind::terminate()).ok();
        let mut hangup = signal(SignalKind::hangup()).ok();
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {},
            _ = async { if let Some(stream) = terminate.as_mut() { let _ = stream.recv().await; } } => {},
            _ = async { if let Some(stream) = hangup.as_mut() { let _ = stream.recv().await; } } => {},
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

#[cfg(test)]
mod tests {
    use axum::http::Method;

    use super::*;

    #[tokio::test]
    async fn local_http_guard_preserves_local_clients_and_stops_untrusted_requests() {
        use tower::ServiceExt as _;
        let address: SocketAddr = "127.0.0.1:18484".parse().unwrap();
        let app = Router::new()
            .fallback(|| async { StatusCode::NO_CONTENT })
            .layer(middleware::from_fn_with_state(address, guard_local_http));
        for (host, origin, path, expected) in [
            (
                "127.0.0.1:18484",
                None,
                "/api/query",
                StatusCode::NO_CONTENT,
            ),
            (
                "LOCALHOST:18484",
                Some("http://localhost:18484"),
                "/api/query",
                StatusCode::NO_CONTENT,
            ),
            (
                "127.0.0.1:18484",
                Some("http://127.0.0.1:18484"),
                "/system/local-ai-integrations/cline/install",
                StatusCode::NO_CONTENT,
            ),
            (
                "localhost:18484",
                Some("http://localhost:18484"),
                "/system/local-ai-integrations/pi/update",
                StatusCode::NO_CONTENT,
            ),
            (
                "attacker.example:18484",
                Some("http://attacker.example:18484"),
                "/api/query",
                StatusCode::FORBIDDEN,
            ),
            (
                "localhost.attacker.example:18484",
                None,
                "/api/query",
                StatusCode::FORBIDDEN,
            ),
            ("localhost:18485", None, "/api/query", StatusCode::FORBIDDEN),
            ("127.0.0.2:18484", None, "/api/query", StatusCode::FORBIDDEN),
            (
                "localhost:18484",
                None,
                "/system/local-ai-integrations/cline/install",
                StatusCode::FORBIDDEN,
            ),
            (
                "localhost:18484",
                Some("null"),
                "/system/local-ai-integrations/cline/install",
                StatusCode::FORBIDDEN,
            ),
            (
                "localhost:18484",
                Some("http://attacker.example"),
                "/system/local-ai-integrations/cline/install",
                StatusCode::FORBIDDEN,
            ),
            (
                "localhost:18484",
                Some("http://localhost:18485"),
                "/system/local-ai-integrations/cline/install",
                StatusCode::FORBIDDEN,
            ),
            (
                "localhost:18484",
                Some("https://localhost:18484"),
                "/api/query",
                StatusCode::FORBIDDEN,
            ),
            (
                "localhost:18484",
                Some("http://localhost:18484/path"),
                "/api/query",
                StatusCode::FORBIDDEN,
            ),
            (
                "user@localhost:18484",
                None,
                "/api/query",
                StatusCode::FORBIDDEN,
            ),
        ] {
            let mut request = Request::builder()
                .method("POST")
                .uri(path)
                .header(header::HOST, host);
            if let Some(origin) = origin {
                request = request.header(header::ORIGIN, origin);
            }
            let response = app
                .clone()
                .oneshot(request.body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), expected, "{host} {origin:?} {path}");
        }
    }

    #[test]
    fn local_http_authority_handles_ipv6_default_ports_and_ambiguous_headers() {
        let ipv6: SocketAddr = "[::1]:18484".parse().unwrap();
        let request = Request::builder()
            .uri("/api/query")
            .header(header::HOST, "[::1]:18484")
            .header(header::ORIGIN, "http://[::1]:18484")
            .body(Body::empty())
            .unwrap();
        assert!(local_http_request_allowed(&request, ipv6));
        let address: SocketAddr = "127.0.0.1:80".parse().unwrap();
        for host in ["localhost", "localhost:80"] {
            let request = Request::builder()
                .uri("/api/query")
                .header(header::HOST, host)
                .header(header::ORIGIN, "http://localhost")
                .body(Body::empty())
                .unwrap();
            assert!(local_http_request_allowed(&request, address));
        }
        for host in [
            "localhost:",
            "localhost:99999",
            "localhost:bad",
            "localhost.",
            "user@localhost",
        ] {
            let request = Request::builder()
                .uri("/api/query")
                .header(header::HOST, host)
                .body(Body::empty())
                .unwrap();
            assert!(!local_http_request_allowed(&request, address), "{host}");
        }
        for uri in [
            "http://attacker.example/api/query",
            "https://localhost/api/query",
        ] {
            let request = Request::builder()
                .uri(uri)
                .header(header::HOST, "localhost")
                .body(Body::empty())
                .unwrap();
            assert!(!local_http_request_allowed(&request, address));
        }
        let mut request = Request::builder()
            .uri("http://localhost/api/query")
            .body(Body::empty())
            .unwrap();
        assert!(local_http_request_allowed(&request, address));
        request
            .headers_mut()
            .append(header::HOST, HeaderValue::from_static("localhost"));
        request
            .headers_mut()
            .append(header::HOST, HeaderValue::from_static("localhost"));
        assert!(!local_http_request_allowed(&request, address));
        request.headers_mut().remove(header::HOST);
        request
            .headers_mut()
            .append(header::ORIGIN, HeaderValue::from_static("http://localhost"));
        request
            .headers_mut()
            .append(header::ORIGIN, HeaderValue::from_static("http://localhost"));
        assert!(!local_http_request_allowed(&request, address));
        let request = Request::builder()
            .uri("/api/query")
            .body(Body::empty())
            .unwrap();
        assert!(!local_http_request_allowed(&request, address));
    }

    #[test]
    fn local_ai_integrations_accept_only_declared_hosts_and_actions() {
        assert_eq!(LOCAL_AI_HOSTS.len(), 19);
        assert_eq!(local_ai_operation("hermes", "install"), Ok("install"));
        assert_eq!(local_ai_operation("pi", "repair"), Ok("install"));
        assert_eq!(local_ai_operation("openclaw", "update"), Ok("update"));
        assert_eq!(local_ai_operation("unsloth", "install"), Ok("install"));
        assert_eq!(
            local_ai_operation("unknown", "install").unwrap_err().0,
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            local_ai_operation("hermes", "delete").unwrap_err().0,
            StatusCode::BAD_REQUEST
        );
    }

    async fn asset_response(method: Method, path: &str) -> Response {
        static_asset(
            Request::builder()
                .method(method)
                .uri(path)
                .body(Body::empty())
                .expect("static asset request must be valid"),
        )
        .await
    }

    #[tokio::test]
    async fn web_assets_and_spa_fallback_are_scoped_below_web() {
        for path in ["/web", "/web/", "/web/query", "/web/graph", "/web/streams"] {
            let response = asset_response(Method::GET, path).await;
            assert_eq!(response.status(), StatusCode::OK, "{path}");
            assert_eq!(
                response.headers().get(header::CONTENT_TYPE),
                Some(&HeaderValue::from_static("text/html; charset=utf-8")),
                "{path}"
            );
        }

        assert_eq!(
            asset_response(Method::GET, "/outside").await.status(),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            asset_response(Method::GET, "/api/unknown").await.status(),
            StatusCode::NOT_FOUND
        );
        assert_eq!(
            asset_response(Method::POST, "/web/streams").await.status(),
            StatusCode::METHOD_NOT_ALLOWED
        );
    }

    #[tokio::test]
    async fn root_redirects_to_the_web_namespace() {
        let response = web_root().await.into_response();
        assert_eq!(response.status(), StatusCode::PERMANENT_REDIRECT);
        assert_eq!(
            response.headers().get(header::LOCATION),
            Some(&HeaderValue::from_static("/web/"))
        );
    }
}
