//! Shared remote client and transport-neutral query result model.

use std::{
    collections::BTreeMap,
    io::{BufRead as _, BufReader, Cursor},
    net::IpAddr,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use irongraph_server::protocol::{
    BatchColumn, CatalogEvent, QueryColumn, QueryLimits, QueryRequest, QueryStatistics,
    QueryStreamEvent, RelationshipValue, ResultNode, TypedValue,
};
use irongraph_types::{Bookmark, CommitAcknowledgement, ProjectId};
use neo4j::{ValueReceive, ValueSend};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(120);
const MAXIMUM_EVENT_BYTES: usize = 64 * 1024 * 1024;
const MAXIMUM_TLS_FILE_BYTES: u64 = 16 * 1024 * 1024;

/// PEM files used for mutual TLS. The certificate file may include intermediate certificates.
#[derive(Clone, Debug)]
pub struct MutualTls {
    pub certificate: PathBuf,
    pub private_key: PathBuf,
    pub certificate_authority: PathBuf,
}

impl MutualTls {
    #[must_use]
    pub fn new(
        certificate: impl Into<PathBuf>,
        private_key: impl Into<PathBuf>,
        certificate_authority: impl Into<PathBuf>,
    ) -> Self {
        Self {
            certificate: certificate.into(),
            private_key: private_key.into(),
            certificate_authority: certificate_authority.into(),
        }
    }
}

/// Errors produced before, during, or after remote query execution.
#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error("invalid client configuration: {0}")]
    Configuration(String),
    #[error("HTTP query failed: {0}")]
    Http(#[from] reqwest::Error),
    #[error("Bolt query failed: {0}")]
    Bolt(#[source] Box<neo4j::Neo4jError>),
    #[error("query event decoding failed: {0}")]
    Json(#[from] serde_json::Error),
    #[error("query stream I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("database query failed with {code}: {message}")]
    Database {
        code: String,
        message: String,
        retryable: bool,
        retry_after_ms: Option<u64>,
    },
    #[error("unsupported Bolt value: {0}")]
    UnsupportedBoltValue(String),
    #[error("query result is malformed: {0}")]
    MalformedResult(String),
}

pub type Result<T> = std::result::Result<T, ClientError>;

impl From<neo4j::Neo4jError> for ClientError {
    fn from(error: neo4j::Neo4jError) -> Self {
        Self::Bolt(Box::new(error))
    }
}

/// One transport-neutral Cypher request.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Query {
    pub cypher: String,
    #[serde(default)]
    pub parameters: BTreeMap<String, serde_json::Value>,
    pub project_id: Option<ProjectId>,
    pub bookmark: Option<Bookmark>,
    #[serde(default)]
    pub consistency: CommitAcknowledgement,
    #[serde(default)]
    pub limits: QueryLimits,
}

impl Query {
    #[must_use]
    pub fn new(cypher: impl Into<String>) -> Self {
        Self {
            cypher: cypher.into(),
            parameters: BTreeMap::new(),
            project_id: None,
            bookmark: None,
            consistency: CommitAcknowledgement::default(),
            limits: QueryLimits::default(),
        }
    }

    #[must_use]
    pub fn with_project(mut self, project: ProjectId) -> Self {
        self.project_id = Some(project);
        self
    }

    #[must_use]
    pub fn with_parameter(mut self, name: impl Into<String>, value: serde_json::Value) -> Self {
        self.parameters.insert(name.into(), value);
        self
    }

    pub fn validate(&self) -> Result<()> {
        if self.cypher.trim().is_empty() {
            return Err(ClientError::Configuration("Cypher is empty".to_owned()));
        }
        self.limits
            .validate()
            .map_err(|error| ClientError::Configuration(error.message.to_string()))
    }

    #[must_use]
    pub fn into_protocol(self) -> QueryRequest {
        QueryRequest {
            request_id: Uuid::new_v4(),
            project_id: self.project_id,
            query: self.cypher,
            parameters: self.parameters,
            consistency: self.consistency,
            bookmark: self.bookmark,
            limits: self.limits,
            cancellation: Default::default(),
            deadline: None,
            connection_id: Default::default(),
        }
    }
}

/// Terminal query metadata shared by embedded, API, and Bolt callers.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct QuerySummary {
    pub bookmark: Option<Bookmark>,
    pub statistics: QueryStatistics,
    pub truncated: bool,
    pub truncation_reason: Option<String>,
}

/// A complete transport-neutral query result.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct QueryResult {
    pub catalog: Option<CatalogEvent>,
    pub columns: Vec<QueryColumn>,
    pub rows: Vec<Vec<TypedValue>>,
    pub summary: QuerySummary,
}

impl QueryResult {
    pub fn from_events(events: impl IntoIterator<Item = QueryStreamEvent>) -> Result<Self> {
        let mut result = Self::default();
        for event in events {
            match event {
                QueryStreamEvent::Catalog { catalog } => result.catalog = Some(catalog),
                QueryStreamEvent::Schema { columns, .. } => result.columns = columns,
                QueryStreamEvent::Batch {
                    row_count, columns, ..
                } => append_batch(&mut result, row_count, columns)?,
                QueryStreamEvent::Summary {
                    bookmark,
                    statistics,
                    truncated,
                    truncation_reason,
                    ..
                } => {
                    result.summary = QuerySummary {
                        bookmark: Some(bookmark),
                        statistics,
                        truncated,
                        truncation_reason,
                    };
                }
                QueryStreamEvent::Error {
                    code,
                    message,
                    retryable,
                    retry_after_ms,
                    ..
                } => {
                    return Err(ClientError::Database {
                        code: format!("{code:?}"),
                        message,
                        retryable,
                        retry_after_ms,
                    });
                }
            }
        }
        Ok(result)
    }
}

fn append_batch(result: &mut QueryResult, row_count: u64, columns: Vec<BatchColumn>) -> Result<()> {
    let row_count = usize::try_from(row_count)
        .map_err(|_| ClientError::MalformedResult("batch row count exceeds usize".to_owned()))?;
    if columns
        .iter()
        .any(|column| column.values.len() != row_count)
    {
        return Err(ClientError::MalformedResult(
            "column lengths do not match the batch row count".to_owned(),
        ));
    }
    for row_index in 0..row_count {
        result.rows.push(
            columns
                .iter()
                .map(|column| column.values[row_index].clone())
                .collect(),
        );
    }
    Ok(())
}

/// Client for IronGraph's sole browser-facing data endpoint.
pub struct ApiClient {
    endpoint: url::Url,
    http: reqwest::blocking::Client,
}

impl ApiClient {
    /// Connect to a plain local listener. Non-loopback HTTP is rejected.
    pub fn new(base_url: &str) -> Result<Self> {
        let mut endpoint = parse_local_endpoint(base_url, "http")?;
        endpoint.set_path("/api/query");
        endpoint.set_query(None);
        endpoint.set_fragment(None);
        let http = reqwest::blocking::Client::builder()
            .timeout(DEFAULT_TIMEOUT)
            .build()?;
        Ok(Self { endpoint, http })
    }

    /// Connect to a remote HTTPS listener using a client identity and an explicit server CA.
    pub fn with_mtls(base_url: &str, tls: &MutualTls) -> Result<Self> {
        let mut endpoint = url::Url::parse(base_url)
            .map_err(|error| ClientError::Configuration(error.to_string()))?;
        if endpoint.scheme() != "https" {
            return Err(ClientError::Configuration(
                "mutual-TLS API URLs must use https".to_owned(),
            ));
        }
        endpoint.set_path("/api/query");
        endpoint.set_query(None);
        endpoint.set_fragment(None);

        let mut identity = read_tls_file(&tls.certificate)?;
        identity.extend_from_slice(&read_tls_file(&tls.private_key)?);
        let identity = reqwest::Identity::from_pem(&identity)?;
        let authorities =
            reqwest::Certificate::from_pem_bundle(&read_tls_file(&tls.certificate_authority)?)?;
        let mut builder = reqwest::blocking::Client::builder()
            .timeout(DEFAULT_TIMEOUT)
            .min_tls_version(reqwest::tls::Version::TLS_1_3)
            .identity(identity);
        for authority in authorities {
            builder = builder.add_root_certificate(authority);
        }
        Ok(Self {
            endpoint,
            http: builder.build()?,
        })
    }

    pub fn query(&self, query: Query) -> Result<QueryResult> {
        query.validate()?;
        let response = self
            .http
            .post(self.endpoint.clone())
            .header(reqwest::header::ACCEPT, "application/x-ndjson")
            .json(&query.into_protocol())
            .send()?
            .error_for_status()?;
        let mut events = Vec::new();
        let mut reader = std::io::BufReader::new(response);
        let mut line = Vec::new();
        loop {
            line.clear();
            let read = reader.read_until(b'\n', &mut line)?;
            if read == 0 {
                break;
            }
            if line.len() > MAXIMUM_EVENT_BYTES {
                return Err(ClientError::MalformedResult(
                    "one query event exceeds 64 MiB".to_owned(),
                ));
            }
            while matches!(line.last(), Some(b'\n' | b'\r')) {
                line.pop();
            }
            if !line.is_empty() {
                events.push(serde_json::from_slice::<QueryStreamEvent>(&line)?);
            }
        }
        QueryResult::from_events(events)
    }
}

/// Client for IronGraph's Bolt listener.
pub struct BoltClient {
    driver: neo4j::driver::Driver,
}

impl BoltClient {
    /// Connect to a plain local Bolt listener. Non-loopback Bolt is rejected.
    pub fn new(uri: &str) -> Result<Self> {
        use neo4j::driver::{ConnectionConfig, DriverConfig, auth::AuthToken};

        parse_local_endpoint(uri, "bolt")?;
        let connection: ConnectionConfig =
            uri.parse()
                .map_err(|error: neo4j::driver::ConnectionConfigParseError| {
                    ClientError::Configuration(error.to_string())
                })?;
        let config = DriverConfig::new().with_auth(Arc::new(AuthToken::new_none_auth()));
        Ok(Self {
            driver: neo4j::driver::Driver::new(connection, config),
        })
    }

    /// Connect to a remote Bolt listener using a client identity and an explicit server CA.
    pub fn with_mtls(uri: &str, tls: &MutualTls) -> Result<Self> {
        use neo4j::driver::{ConnectionConfig, DriverConfig, auth::AuthToken};

        let parsed =
            url::Url::parse(uri).map_err(|error| ClientError::Configuration(error.to_string()))?;
        if !matches!(parsed.scheme(), "bolt+s" | "neo4j+s") {
            return Err(ClientError::Configuration(
                "mutual-TLS Bolt URLs must use bolt+s or neo4j+s".to_owned(),
            ));
        }
        let connection: ConnectionConfig = uri
            .parse::<ConnectionConfig>()
            .map_err(|error: neo4j::driver::ConnectionConfigParseError| {
                ClientError::Configuration(error.to_string())
            })?
            .with_encryption_custom_tls_config(bolt_tls_config(tls)?);
        let config = DriverConfig::new().with_auth(Arc::new(AuthToken::new_none_auth()));
        Ok(Self {
            driver: neo4j::driver::Driver::new(connection, config),
        })
    }

    pub fn query(&self, query: Query) -> Result<QueryResult> {
        use neo4j::bookmarks::{Bookmarks, bookmark_managers};

        query.validate()?;
        let project = query.project_id.ok_or_else(|| {
            ClientError::Configuration("Bolt queries require a project ID".to_owned())
        })?;
        let limits = query.limits;
        let initial_bookmarks = query.bookmark.map(|bookmark| {
            Arc::new(Bookmarks::from_raw(std::iter::once(format!(
                "ig:{}:{}",
                bookmark.term, bookmark.index
            ))))
        });
        let bookmark_manager: Arc<dyn neo4j::bookmarks::BookmarkManager> =
            Arc::new(bookmark_managers::simple(initial_bookmarks));
        let parameters = query
            .parameters
            .iter()
            .map(|(name, value)| json_to_bolt(value).map(|value| (name.clone(), value)))
            .collect::<Result<std::collections::HashMap<_, _>>>()?;
        let eager = self
            .driver
            .execute_query(&query.cypher)
            .with_database(Arc::new(project.0.to_string()))
            .with_parameters(&parameters)
            .with_bookmark_manager(Arc::clone(&bookmark_manager))
            .run()?;
        let summary = eager.summary.clone();
        let columns = eager
            .keys
            .iter()
            .map(|name| QueryColumn {
                name: name.to_string(),
                value_type: "ANY".to_owned(),
                nullable: true,
            })
            .collect();
        let rows = eager
            .records
            .into_iter()
            .map(|record| record.into_values().map(bolt_to_typed).collect())
            .collect::<Result<Vec<Vec<_>>>>()?;
        let bookmark = bookmark_manager
            .get_bookmarks()
            .map_err(|error| ClientError::MalformedResult(error.to_string()))?
            .raw()
            .filter_map(parse_bolt_bookmark)
            .max_by_key(|bookmark| (bookmark.term, bookmark.index));
        let elapsed = summary
            .result_available_after
            .unwrap_or_default()
            .saturating_add(summary.result_consumed_after.unwrap_or_default());
        let updates = bolt_update_count(&summary.counters);
        let mut result = QueryResult {
            catalog: None,
            columns,
            rows,
            summary: QuerySummary {
                bookmark,
                statistics: QueryStatistics {
                    elapsed_ms: u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX),
                    elapsed_us: u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX),
                    rows: 0,
                    nodes: 0,
                    edges: 0,
                    updates,
                },
                truncated: false,
                truncation_reason: None,
            },
        };
        let (rows, nodes, edges) = enforce_result_limits(&result, limits)?;
        result.summary.statistics.rows = rows;
        result.summary.statistics.nodes = nodes;
        result.summary.statistics.edges = edges;
        Ok(result)
    }
}

/// One package-level entry point for either supported remote query transport.
pub enum RemoteClient {
    Api(ApiClient),
    Bolt(Box<BoltClient>),
}

impl RemoteClient {
    pub fn api(base_url: &str) -> Result<Self> {
        ApiClient::new(base_url).map(Self::Api)
    }

    pub fn bolt(uri: &str) -> Result<Self> {
        BoltClient::new(uri).map(Box::new).map(Self::Bolt)
    }

    pub fn api_mtls(base_url: &str, tls: &MutualTls) -> Result<Self> {
        ApiClient::with_mtls(base_url, tls).map(Self::Api)
    }

    pub fn bolt_mtls(uri: &str, tls: &MutualTls) -> Result<Self> {
        BoltClient::with_mtls(uri, tls)
            .map(Box::new)
            .map(Self::Bolt)
    }

    pub fn query(&self, query: Query) -> Result<QueryResult> {
        match self {
            Self::Api(client) => client.query(query),
            Self::Bolt(client) => client.query(query),
        }
    }
}

fn parse_local_endpoint(value: &str, expected_scheme: &str) -> Result<url::Url> {
    let endpoint =
        url::Url::parse(value).map_err(|error| ClientError::Configuration(error.to_string()))?;
    if endpoint.scheme() != expected_scheme {
        return Err(ClientError::Configuration(format!(
            "plain local endpoint must use {expected_scheme}"
        )));
    }
    let host = endpoint
        .host_str()
        .ok_or_else(|| ClientError::Configuration("endpoint has no host".to_owned()))?;
    let loopback = host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<IpAddr>()
            .is_ok_and(|address| address.is_loopback());
    if !loopback {
        return Err(ClientError::Configuration(
            "plain remote connections are forbidden; use mutual TLS".to_owned(),
        ));
    }
    Ok(endpoint)
}

fn read_tls_file(path: &Path) -> Result<Vec<u8>> {
    let metadata = std::fs::symlink_metadata(path).map_err(|error| {
        ClientError::Configuration(format!(
            "cannot inspect TLS file {}: {error}",
            path.display()
        ))
    })?;
    if !metadata.file_type().is_file()
        || metadata.len() == 0
        || metadata.len() > MAXIMUM_TLS_FILE_BYTES
    {
        return Err(ClientError::Configuration(format!(
            "TLS material is not a bounded regular file: {}",
            path.display()
        )));
    }
    std::fs::read(path).map_err(|error| {
        ClientError::Configuration(format!("cannot read TLS file {}: {error}", path.display()))
    })
}

fn bolt_tls_config(tls: &MutualTls) -> Result<rustls::ClientConfig> {
    let provider = Arc::new(rustls::crypto::aws_lc_rs::default_provider());
    let mut roots = rustls::RootCertStore::empty();
    let mut authority_reader =
        BufReader::new(Cursor::new(read_tls_file(&tls.certificate_authority)?));
    let authorities = rustls_pemfile::certs(&mut authority_reader)
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|error| ClientError::Configuration(format!("invalid server CA: {error}")))?;
    if authorities.is_empty() {
        return Err(ClientError::Configuration(
            "server CA file contains no certificates".to_owned(),
        ));
    }
    for authority in authorities {
        roots
            .add(authority)
            .map_err(|error| ClientError::Configuration(format!("invalid server CA: {error}")))?;
    }

    let mut certificate_reader = BufReader::new(Cursor::new(read_tls_file(&tls.certificate)?));
    let certificates = rustls_pemfile::certs(&mut certificate_reader)
        .collect::<std::result::Result<Vec<_>, _>>()
        .map_err(|error| {
            ClientError::Configuration(format!("invalid client certificate: {error}"))
        })?;
    if certificates.is_empty() {
        return Err(ClientError::Configuration(
            "client certificate file contains no certificates".to_owned(),
        ));
    }
    let mut key_reader = BufReader::new(Cursor::new(read_tls_file(&tls.private_key)?));
    let key = rustls_pemfile::private_key(&mut key_reader)
        .map_err(|error| {
            ClientError::Configuration(format!("invalid client private key: {error}"))
        })?
        .ok_or_else(|| {
            ClientError::Configuration("client private-key file contains no key".to_owned())
        })?;
    rustls::ClientConfig::builder_with_provider(provider)
        .with_protocol_versions(&[&rustls::version::TLS13])
        .map_err(|error| ClientError::Configuration(format!("cannot enable TLS 1.3: {error}")))?
        .with_root_certificates(roots)
        .with_client_auth_cert(certificates, key)
        .map_err(|error| ClientError::Configuration(format!("invalid client identity: {error}")))
}

fn parse_bolt_bookmark(value: &str) -> Option<Bookmark> {
    let mut fields = value.split(':');
    if fields.next()? != "ig" {
        return None;
    }
    let term = fields.next()?.parse().ok()?;
    let index = fields.next()?.parse().ok()?;
    if fields.next().is_some() {
        return None;
    }
    Some(Bookmark { term, index })
}

fn bolt_update_count(counters: &neo4j::summary::Counters) -> u64 {
    [
        counters.nodes_created,
        counters.nodes_deleted,
        counters.relationships_created,
        counters.relationships_deleted,
        counters.properties_set,
        counters.labels_added,
        counters.labels_removed,
        counters.indexes_added,
        counters.indexes_removed,
        counters.constraints_added,
        counters.constraints_removed,
        counters.system_updates,
    ]
    .into_iter()
    .map(|value| u64::try_from(value).unwrap_or_default())
    .sum()
}

fn enforce_result_limits(result: &QueryResult, limits: QueryLimits) -> Result<(u64, u64, u64)> {
    let rows = u64::try_from(result.rows.len()).unwrap_or(u64::MAX);
    let bytes = u64::try_from(serde_json::to_vec(result)?.len()).unwrap_or(u64::MAX);
    let (nodes, edges) = result
        .rows
        .iter()
        .flatten()
        .map(typed_graph_counts)
        .try_fold((0_u64, 0_u64), |(nodes, edges), counts| {
            let (value_nodes, value_edges) = counts?;
            Some((
                nodes.checked_add(value_nodes)?,
                edges.checked_add(value_edges)?,
            ))
        })
        .ok_or_else(|| {
            ClientError::MalformedResult("graph result accounting overflow".to_owned())
        })?;
    for (kind, observed, limit) in [
        ("rows", rows, limits.rows),
        ("bytes", bytes, limits.bytes),
        ("nodes", nodes, limits.nodes),
        ("relationships", edges, limits.edges),
    ] {
        if observed > limit {
            return Err(ClientError::Database {
                code: "RESULT_BUDGET_EXCEEDED".to_owned(),
                message: format!(
                    "query result exceeds the caller's {kind} limit ({observed}/{limit})"
                ),
                retryable: false,
                retry_after_ms: None,
            });
        }
    }
    Ok((rows, nodes, edges))
}

fn typed_graph_counts(value: &TypedValue) -> Option<(u64, u64)> {
    match value {
        TypedValue::Node(_) => Some((1, 0)),
        TypedValue::Relationship(_) => Some((0, 1)),
        TypedValue::Path(path) => Some((
            u64::try_from(path.nodes.len()).ok()?,
            u64::try_from(path.relationships.len()).ok()?,
        )),
        TypedValue::List(values) => values.iter().map(typed_graph_counts).try_fold(
            (0_u64, 0_u64),
            |(nodes, edges), counts| {
                let (value_nodes, value_edges) = counts?;
                Some((
                    nodes.checked_add(value_nodes)?,
                    edges.checked_add(value_edges)?,
                ))
            },
        ),
        TypedValue::Map(values) => values.values().map(typed_graph_counts).try_fold(
            (0_u64, 0_u64),
            |(nodes, edges), counts| {
                let (value_nodes, value_edges) = counts?;
                Some((
                    nodes.checked_add(value_nodes)?,
                    edges.checked_add(value_edges)?,
                ))
            },
        ),
        TypedValue::Null
        | TypedValue::Boolean(_)
        | TypedValue::Integer(_)
        | TypedValue::Float(_)
        | TypedValue::String(_)
        | TypedValue::Bytes(_)
        | TypedValue::Date(_)
        | TypedValue::Time { .. }
        | TypedValue::DateTime { .. }
        | TypedValue::Duration { .. }
        | TypedValue::Vector(_) => Some((0, 0)),
    }
}

fn json_to_bolt(value: &serde_json::Value) -> Result<ValueSend> {
    Ok(match value {
        serde_json::Value::Null => ValueSend::Null,
        serde_json::Value::Bool(value) => ValueSend::Boolean(*value),
        serde_json::Value::Number(value) => {
            if let Some(value) = value.as_i64() {
                ValueSend::Integer(value)
            } else if let Some(value) = value.as_u64() {
                ValueSend::Integer(i64::try_from(value).map_err(|_| {
                    ClientError::Configuration("unsigned parameter exceeds i64".to_owned())
                })?)
            } else {
                ValueSend::Float(value.as_f64().ok_or_else(|| {
                    ClientError::Configuration("invalid numeric parameter".to_owned())
                })?)
            }
        }
        serde_json::Value::String(value) => ValueSend::String(value.clone()),
        serde_json::Value::Array(values) => {
            ValueSend::List(values.iter().map(json_to_bolt).collect::<Result<_>>()?)
        }
        serde_json::Value::Object(values) => ValueSend::Map(
            values
                .iter()
                .map(|(key, value)| json_to_bolt(value).map(|value| (key.clone(), value)))
                .collect::<Result<_>>()?,
        ),
    })
}

#[allow(clippy::too_many_lines)]
fn bolt_to_typed(value: ValueReceive) -> Result<TypedValue> {
    use chrono::Timelike as _;
    use neo4j::value::graph::RelationshipDirection;

    Ok(match value {
        ValueReceive::Null => TypedValue::Null,
        ValueReceive::Boolean(value) => TypedValue::Boolean(value),
        ValueReceive::Integer(value) => TypedValue::Integer(value.to_string()),
        ValueReceive::Float(value) => TypedValue::Float(value),
        ValueReceive::Bytes(value) => TypedValue::Bytes(value),
        ValueReceive::String(value) => TypedValue::String(value),
        ValueReceive::List(values) => TypedValue::List(
            values
                .into_iter()
                .map(bolt_to_typed)
                .collect::<Result<_>>()?,
        ),
        ValueReceive::Map(values) => TypedValue::Map(
            values
                .into_iter()
                .map(|(key, value)| bolt_to_typed(value).map(|value| (key, value)))
                .collect::<Result<_>>()?,
        ),
        ValueReceive::Node(node) => TypedValue::Node(ResultNode {
            id: node.element_id,
            labels: node.labels,
            properties: node
                .properties
                .into_iter()
                .map(|(key, value)| bolt_to_typed(value).map(|value| (key, value)))
                .collect::<Result<_>>()?,
        }),
        ValueReceive::Relationship(relationship) => TypedValue::Relationship(RelationshipValue {
            id: relationship.element_id,
            source: relationship.start_node_element_id,
            target: relationship.end_node_element_id,
            relationship_type: relationship.type_,
            properties: relationship
                .properties
                .into_iter()
                .map(|(key, value)| bolt_to_typed(value).map(|value| (key, value)))
                .collect::<Result<_>>()?,
        }),
        ValueReceive::Path(path) => {
            let (start, hops) = path.traverse();
            let mut nodes = vec![bolt_node(start)?];
            let mut relationships = Vec::with_capacity(hops.len());
            let mut previous = start.element_id.clone();
            for (direction, relationship, next) in hops {
                let (source, target) = match direction {
                    RelationshipDirection::To => (previous.clone(), next.element_id.clone()),
                    RelationshipDirection::From => (next.element_id.clone(), previous.clone()),
                };
                relationships.push(RelationshipValue {
                    id: relationship.element_id.clone(),
                    source,
                    target,
                    relationship_type: relationship.type_.clone(),
                    properties: relationship
                        .properties
                        .clone()
                        .into_iter()
                        .map(|(key, value)| bolt_to_typed(value).map(|value| (key, value)))
                        .collect::<Result<_>>()?,
                });
                nodes.push(bolt_node(next)?);
                previous = next.element_id.clone();
            }
            TypedValue::Path(irongraph_server::protocol::PathValue {
                nodes,
                relationships,
            })
        }
        ValueReceive::Date(value) => {
            let epoch = chrono::NaiveDate::from_ymd_opt(1970, 1, 1)
                .ok_or_else(|| ClientError::MalformedResult("invalid epoch".to_owned()))?;
            TypedValue::Date(value.signed_duration_since(epoch).num_days())
        }
        ValueReceive::LocalTime(value) => TypedValue::Time {
            nanos: i64::from(value.num_seconds_from_midnight()) * 1_000_000_000
                + i64::from(value.nanosecond()),
            offset_seconds: None,
        },
        ValueReceive::Time(value) => TypedValue::Time {
            nanos: i64::from(value.time.num_seconds_from_midnight()) * 1_000_000_000
                + i64::from(value.time.nanosecond()),
            offset_seconds: Some(value.offset.local_minus_utc()),
        },
        ValueReceive::LocalDateTime(value) => TypedValue::DateTime {
            seconds: value.and_utc().timestamp(),
            nanos: value.nanosecond(),
            timezone: None,
        },
        ValueReceive::DateTime(value) => TypedValue::DateTime {
            seconds: value.timestamp(),
            nanos: value.nanosecond(),
            timezone: Some(value.timezone().name().to_owned()),
        },
        ValueReceive::DateTimeFixed(value) => TypedValue::DateTime {
            seconds: value.timestamp(),
            nanos: value.nanosecond(),
            timezone: Some(value.offset().to_string()),
        },
        ValueReceive::Duration(value) => TypedValue::Duration {
            months: value.months(),
            days: value.days(),
            seconds: value.seconds(),
            nanos: value.nanoseconds(),
        },
        ValueReceive::Cartesian2D(_)
        | ValueReceive::Cartesian3D(_)
        | ValueReceive::WGS84_2D(_)
        | ValueReceive::WGS84_3D(_)
        | ValueReceive::BrokenValue(_) => {
            return Err(ClientError::UnsupportedBoltValue(format!("{value:?}")));
        }
        _ => return Err(ClientError::UnsupportedBoltValue(format!("{value:?}"))),
    })
}

fn bolt_node(node: &neo4j::value::graph::Node) -> Result<ResultNode> {
    Ok(ResultNode {
        id: node.element_id.clone(),
        labels: node.labels.clone(),
        properties: node
            .properties
            .clone()
            .into_iter()
            .map(|(key, value)| bolt_to_typed(value).map(|value| (key, value)))
            .collect::<Result<_>>()?,
    })
}
