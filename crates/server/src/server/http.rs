use std::{env, net::SocketAddr, path::PathBuf, process::Command, sync::Arc, time::Duration};

use axum::{
    Router,
    body::Body,
    extract::{DefaultBodyLimit, Path, State},
    http::{HeaderMap, HeaderValue, Request, StatusCode, header},
    middleware::{self, Next},
    response::{IntoResponse, Redirect, Response},
    routing::{get, post},
    serve::ListenerExt,
};
#[cfg(irongraph_web_bundle)]
use rust_embed::RustEmbed;
use tokio::{net::TcpListener, task::JoinSet};
use tokio_util::sync::CancellationToken;

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
    protocol::{BoltServer, QueryExecutor, QueryRequest, QueryStreamEvent, ndjson_channel},
};

#[cfg_attr(irongraph_web_bundle, derive(RustEmbed))]
#[cfg_attr(irongraph_web_bundle, folder = "../../web/dist")]
struct WebAssets;

#[cfg(not(irongraph_web_bundle))]
impl WebAssets {
    fn get(_path: &str) -> Option<rust_embed::EmbeddedFile> {
        None
    }
}

#[derive(Clone)]
struct AppState {
    database: Database,
    startup: Arc<StartupProgress>,
}

struct StartupProgress {
    install: irongraph_embedding::ModelInstallProgress,
    phase: parking_lot::Mutex<&'static str>,
    error: parking_lot::Mutex<Option<String>>,
}

impl Default for StartupProgress {
    fn default() -> Self {
        Self {
            install: irongraph_embedding::ModelInstallProgress::default(),
            phase: parking_lot::Mutex::new("downloading"),
            error: parking_lot::Mutex::new(None),
        }
    }
}

async fn startup_progress(State(state): State<AppState>) -> impl IntoResponse {
    let phase = *state.startup.phase.lock();
    (
        [(header::CACHE_CONTROL, "no-store")],
        axum::Json(serde_json::json!({
            "phase": phase,
            "downloaded_bytes": state.startup.install.written(),
            "total_bytes": state.startup.install.total_bytes(),
            "error": state.startup.error.lock().clone(),
        })),
    )
}

/// Opens durable state, loads the text encoder, and runs the standalone database protocols.
pub async fn run(config: Config) -> Result<()> {
    run_with_startup(config, CancellationToken::new(), initialize_embedding).await
}

fn initialize_embedding(
    progress: Arc<StartupProgress>,
    database: Database,
    device: crate::embeddings::EmbeddingDevice,
) -> Result<()> {
    let artifacts =
        irongraph_embedding::ensure_default_embedding_model_with_progress(&progress.install)?;
    *progress.phase.lock() = "loading";
    let embedding = Arc::new(crate::embeddings::LocalEmbeddingModel::load(
        artifacts, device,
    )?);
    *progress.phase.lock() = "warming";
    embedding.warm_up()?;
    database.bind_text_embedding(embedding)?;
    *progress.phase.lock() = "ready";
    Ok(())
}

async fn run_with_startup(
    config: Config,
    shutdown: CancellationToken,
    initialize: impl FnOnce(
        Arc<StartupProgress>,
        Database,
        crate::embeddings::EmbeddingDevice,
    ) -> Result<()>
    + Send
    + 'static,
) -> Result<()> {
    config.validate()?;
    let identity_directory = config.data_dir.clone();
    tokio::task::spawn_blocking(move || {
        if load_existing_node_identity(&identity_directory)?.is_none() {
            let _ = load_or_generate_genesis_identity(&identity_directory)?;
        }
        Ok::<_, Error>(())
    })
    .await
    .map_err(|error| {
        Error::internal(format!("identity initialization worker failed: {error}"))
    })??;

    let bootstrap_options = SingleNodeBootstrapConfig {
        execution_class: ExecutionClass::Cpu,
        startup_timeout: config.startup_timeout(),
        storage_limits: WriteStorageLimits {
            max_log_record_bytes: usize::MAX,
            max_log_entries_per_read: 4_096,
            max_snapshot_bytes: usize::MAX,
        },
    };
    let max_write_bytes = usize::MAX;
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

    let startup = Arc::new(StartupProgress::default());
    let state = AppState {
        database: database.clone(),
        startup: Arc::clone(&startup),
    };
    let http_listener = TcpListener::bind(config.http_addr).await?;
    let local_address = http_listener.local_addr()?;
    let http_listener = http_listener.tap_io(|socket| {
        if let Err(error) = socket.set_nodelay(true) {
            tracing::warn!(%error, "could not enable immediate HTTP response writes");
        }
    });
    let app = Router::new()
        .route("/api/query", post(query))
        .route("/system/startup", get(startup_progress))
        .route("/system/local-ai-integrations", get(local_ai_integrations))
        .route(
            "/system/local-ai-integrations/{host}/{action}",
            post(change_local_ai_integration),
        )
        .route("/", get(web_root))
        .fallback(static_asset)
        .layer(DefaultBodyLimit::disable())
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
    let embedding_device = config.embedding_device();
    let embedding_database = database.clone();
    let embedding_shutdown = shutdown.clone();
    components.spawn(async move {
        let progress = Arc::clone(&startup);
        let mut initializer = tokio::task::spawn_blocking(move || {
            initialize(progress, embedding_database, embedding_device)
        });
        let initialized = tokio::select! {
            result = &mut initializer => result,
            () = embedding_shutdown.cancelled() => {
                startup.install.cancel();
                initializer.await
            }
        };
        let failure = match initialized {
            Ok(Ok(())) => None,
            Ok(Err(error)) => Some(error.to_string()),
            Err(error) => Some(format!("embedding startup worker failed: {error}")),
        };
        if let Some(error) = failure {
            tracing::error!(%error, "embedding startup failed");
            *startup.error.lock() = Some(error);
            *startup.phase.lock() = "failed";
        }
        embedding_shutdown.cancelled().await;
        Ok(())
    });
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
    let embedding_shutdown = database.shutdown_embedding_jobs().await;
    outcome
        .and(embedding_shutdown)
        .and(boot.runtime().shutdown().await)
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
    headers: HeaderMap,
    request: Request<Body>,
) -> Response {
    let json = headers
        .get(header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.split(';').next())
        .and_then(|value| value.trim().split_once('/'))
        .is_some_and(|(kind, subtype)| {
            kind.eq_ignore_ascii_case("application")
                && subtype
                    .rsplit('+')
                    .next()
                    .is_some_and(|suffix| suffix.eq_ignore_ascii_case("json"))
        });
    if !json {
        return StatusCode::UNSUPPORTED_MEDIA_TYPE.into_response();
    }
    let body = match axum::body::to_bytes(request.into_body(), usize::MAX).await {
        Ok(body) => body,
        Err(error) => return (StatusCode::BAD_REQUEST, error.to_string()).into_response(),
    };
    let cancellation = CancellationToken::new();
    let database = state.database;
    let (sender, stream) = ndjson_channel::<QueryStreamEvent>(8);
    let stream = stream.cancel_on_drop(cancellation.clone());
    let (accepted, acceptance) = tokio::sync::oneshot::channel();
    tokio::task::spawn_blocking(move || {
        let mut request = match axum::Json::<QueryRequest>::from_bytes(&body) {
            Ok(axum::Json(request)) => request,
            Err(error) => {
                let _ = accepted.send(Err(error.into_response()));
                return;
            }
        };
        request.cancellation = cancellation;
        request.deadline = None;
        request.connection_id = crate::storage::ConnectionId::new();
        let request_id = request.request_id;
        // Return the original HTTP decode error status; accepted queries continue on this
        // same producer worker. Dropping the request before acceptance prevents execution.
        if accepted.send(Ok(())).is_err() {
            return;
        }
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
    match acceptance.await {
        Ok(Ok(())) => ndjson_response(stream.into_body()),
        Ok(Err(response)) => response,
        Err(error) => (StatusCode::INTERNAL_SERVER_ERROR, error.to_string()).into_response(),
    }
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
    _config: &Config,
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
        AuthenticatedTlsListener::bind(address, tls, database.clone(), scope).await?,
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

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn console_and_queries_answer_while_model_preparation_is_blocked() -> Result<()> {
        use clap::Parser as _;
        use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
        let directory = tempfile::tempdir()?;
        let reservations = (0..4)
            .map(|_| std::net::TcpListener::bind("127.0.0.1:0"))
            .collect::<std::io::Result<Vec<_>>>()?;
        let addresses = reservations
            .iter()
            .map(|listener| listener.local_addr().unwrap().to_string())
            .collect::<Vec<_>>();
        let config = Config::try_parse_from([
            "irongraph",
            "--data-dir",
            directory.path().to_str().unwrap(),
            "--http-addr",
            &addresses[0],
            "--bolt-addr",
            &addresses[1],
            "--stream-addr",
            &addresses[2],
            "--queue-addr",
            &addresses[3],
        ])
        .map_err(|error| Error::invalid_data(error.to_string()))?;
        drop(reservations);
        let shutdown = CancellationToken::new();
        let release = shutdown.clone();
        let server = tokio::spawn(run_with_startup(
            config,
            shutdown.clone(),
            move |_progress, _database, _device| {
                while !release.is_cancelled() {
                    std::thread::sleep(Duration::from_millis(5));
                }
                Err(Error::new(
                    ErrorCode::Cancelled,
                    "controlled preparation cancelled",
                ))
            },
        ));
        let check = async {
            let deadline = tokio::time::Instant::now() + Duration::from_secs(5);
            let mut stream = loop {
                if let Ok(stream) = tokio::net::TcpStream::connect(&addresses[0]).await {
                    break stream;
                }
                if tokio::time::Instant::now() >= deadline {
                    return Err(Error::internal(
                        "HTTP listener did not start before model initialization",
                    ));
                }
                tokio::time::sleep(Duration::from_millis(5)).await;
            };
            stream
                .write_all(
                    format!(
                        "GET /system/startup HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n",
                        addresses[0]
                    )
                    .as_bytes(),
                )
                .await?;
            let mut response = String::new();
            stream.read_to_string(&mut response).await?;
            if !response.starts_with("HTTP/1.1 200")
                || !response.contains("\"phase\":\"downloading\"")
            {
                return Err(Error::internal(format!(
                    "startup progress did not answer while preparation was blocked: {response}"
                )));
            }
            let mut stream = tokio::net::TcpStream::connect(&addresses[0]).await?;
            stream
                .write_all(
                    format!(
                        "GET /web/ HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n\r\n",
                        addresses[0]
                    )
                    .as_bytes(),
                )
                .await?;
            response.clear();
            stream.read_to_string(&mut response).await?;
            if !response.starts_with("HTTP/1.1 200") || !response.contains("id=\"root\"") {
                return Err(Error::internal(format!(
                    "console did not load while preparation was blocked: {response}"
                )));
            }
            let request = serde_json::json!({"request_id":uuid::Uuid::new_v4(),"project_id":null,"query":"SHOW PROJECTS","bookmark":null}).to_string();
            let mut stream = tokio::net::TcpStream::connect(&addresses[0]).await?;
            stream.write_all(format!("POST /api/query HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{request}", addresses[0], request.len()).as_bytes()).await?;
            response.clear();
            stream.read_to_string(&mut response).await?;
            if !response.starts_with("HTTP/1.1 200") || !response.contains("summary") {
                return Err(Error::internal(format!(
                    "query did not complete while preparation was blocked: {response}"
                )));
            }
            for (content_type, body, expected_status) in [
                ("application/json", "{", "400"),
                ("application/json", "{}", "422"),
                ("text/plain", "{}", "415"),
            ] {
                let mut stream = tokio::net::TcpStream::connect(&addresses[0]).await?;
                stream.write_all(format!("POST /api/query HTTP/1.1\r\nHost: {}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}", addresses[0], body.len()).as_bytes()).await?;
                response.clear();
                stream.read_to_string(&mut response).await?;
                assert!(
                    response.starts_with(&format!("HTTP/1.1 {expected_status}")),
                    "{response}"
                );
            }
            Ok::<_, Error>(())
        };
        let outcome = tokio::time::timeout(Duration::from_secs(10), check).await;
        shutdown.cancel();
        server
            .await
            .map_err(|error| Error::internal(error.to_string()))??;
        outcome
            .map_err(|_| Error::internal("startup requests stalled during model preparation"))??;
        println!("CONSOLE AND QUERY API RESPOND BEFORE MODEL PREPARATION COMPLETES");
        Ok(())
    }

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
