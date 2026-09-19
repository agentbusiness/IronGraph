//! Lean stdio MCP server for host access to an IronGraph database.

use std::{
    collections::BTreeMap,
    io::{BufRead as _, Write as _},
    net::{IpAddr, SocketAddr},
    sync::Arc,
};

use axum::{
    Json, Router,
    extract::State,
    http::{HeaderMap, StatusCode, header::ORIGIN},
    response::{IntoResponse, Response},
    routing::post,
};
use irongraph_client::{ApiClient, Query, QueryResult};
use irongraph_server::protocol::{
    PathValue, QueryLimits, RelationshipValue, ResultNode, TypedValue,
};
use serde::Deserialize;
use serde_json::{Value, json};
use uuid::Uuid;

pub mod integrations;

const PROTOCOL_VERSION: &str = "2025-06-18";
const RESOURCE_PREFIX: &str = "irongraph://cypher/";
const DEFAULT_MAX_ROWS: u64 = 200;
const MAX_MAX_ROWS: u64 = 10_000;
const MAX_RESULT_BYTES: u64 = 4 * 1024 * 1024;

/// One embedded public Cypher reference page.
pub struct CypherDoc {
    path: &'static str,
    markdown: &'static str,
}

include!(concat!(env!("OUT_DIR"), "/cypher_docs.rs"));

/// Query boundary used by the MCP protocol layer.
pub trait Database {
    fn query(&self, query: Query) -> irongraph_client::Result<QueryResult>;
}

/// Database adapter backed by IronGraph's canonical Query API client.
pub struct ApiDatabase {
    client: ApiClient,
}

/// Query adapter for async MCP transports; the blocking client lives only on the request worker.
pub struct OnDemandApiDatabase {
    url: String,
    mutual_tls: Option<irongraph_client::MutualTls>,
}

impl OnDemandApiDatabase {
    /// Configure a plain loopback Query API connection.
    #[must_use]
    pub fn new(url: impl Into<String>) -> Self {
        Self {
            url: url.into(),
            mutual_tls: None,
        }
    }

    /// Configure a remote mutual-TLS Query API connection.
    #[must_use]
    pub fn with_mtls(url: impl Into<String>, mutual_tls: irongraph_client::MutualTls) -> Self {
        Self {
            url: url.into(),
            mutual_tls: Some(mutual_tls),
        }
    }
}

impl Database for OnDemandApiDatabase {
    fn query(&self, query: Query) -> irongraph_client::Result<QueryResult> {
        match &self.mutual_tls {
            Some(mutual_tls) => ApiClient::with_mtls(&self.url, mutual_tls)?.query(query),
            None => ApiClient::new(&self.url)?.query(query),
        }
    }
}

impl ApiDatabase {
    #[must_use]
    pub const fn new(client: ApiClient) -> Self {
        Self { client }
    }
}

impl Database for ApiDatabase {
    fn query(&self, query: Query) -> irongraph_client::Result<QueryResult> {
        self.client.query(query)
    }
}

/// MCP protocol server. Transport framing is one JSON-RPC object per stdio line.
pub struct McpServer<D> {
    database: D,
}

impl<D: Database> McpServer<D> {
    #[must_use]
    pub const fn new(database: D) -> Self {
        Self { database }
    }

    /// Serve MCP until the host closes stdin.
    pub fn run_stdio(&self) -> anyhow::Result<()> {
        let stdin = std::io::stdin();
        let mut stdout = std::io::BufWriter::new(std::io::stdout().lock());
        for line in stdin.lock().lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let response = match serde_json::from_str::<RpcRequest>(&line) {
                Ok(request) => self.handle(request),
                Err(error) => Some(rpc_error(
                    Value::Null,
                    -32700,
                    format!("invalid JSON-RPC request: {error}"),
                )),
            };
            if let Some(response) = response {
                serde_json::to_writer(&mut stdout, &response)?;
                stdout.write_all(b"\n")?;
                stdout.flush()?;
            }
        }
        Ok(())
    }

    fn handle_value(&self, value: Value) -> Option<Value> {
        match serde_json::from_value::<RpcRequest>(value) {
            Ok(request) => self.handle(request),
            Err(error) => Some(rpc_error(
                Value::Null,
                -32600,
                format!("invalid JSON-RPC request: {error}"),
            )),
        }
    }

    fn handle(&self, request: RpcRequest) -> Option<Value> {
        let id = request.id?;
        let result = match request.method.as_str() {
            "initialize" => Ok(initialize(&request.params)),
            "ping" => Ok(json!({})),
            "tools/list" => Ok(json!({ "tools": tool_definitions() })),
            "tools/call" => self.call_tool(&request.params),
            "resources/list" => Ok(list_resources()),
            "resources/read" => read_resource(&request.params),
            method => {
                return Some(rpc_error(
                    id,
                    -32601,
                    format!("unknown MCP method: {method}"),
                ));
            }
        };
        Some(match result {
            Ok(result) => json!({ "jsonrpc": "2.0", "id": id, "result": result }),
            Err(error) => rpc_error(id, -32602, error),
        })
    }

    fn call_tool(&self, params: &Value) -> Result<Value, String> {
        let call: ToolCall = serde_json::from_value(params.clone())
            .map_err(|error| format!("invalid tools/call parameters: {error}"))?;
        let result = match call.name.as_str() {
            "irongraph_run_cypher" => self.run_cypher(call.arguments),
            "irongraph_get_schema" => self.get_schema(call.arguments),
            "irongraph_save_document" => self.save_document(call.arguments),
            "irongraph_search" => self.search(call.arguments),
            "irongraph_search_cypher_docs" => search_cypher_docs(call.arguments),
            name => return Err(format!("unknown IronGraph tool: {name}")),
        };
        Ok(match result {
            Ok(structured) => tool_success(structured),
            Err(error) => tool_failure(error),
        })
    }

    fn run_cypher(&self, arguments: Value) -> Result<Value, String> {
        let input: RunCypherInput = parse_arguments(arguments)?;
        let max_rows = bounded_rows(input.max_rows)?;
        let mut query = Query::new(input.cypher);
        query.parameters = input.parameters;
        query.limits = result_limits(max_rows);
        self.database
            .query(query)
            .map(query_result_json)
            .map_err(|error| format!("IronGraph rejected the Cypher: {error}"))
    }

    fn get_schema(&self, arguments: Value) -> Result<Value, String> {
        let input: GetSchemaInput = parse_arguments(arguments)?;
        let cypher = match input.project {
            Some(project) => format!(
                "USE {} RETURN 'catalog' AS status",
                cypher_identifier(&project)?
            ),
            None => "SHOW PROJECTS".to_owned(),
        };
        let mut query = Query::new(cypher);
        query.limits = result_limits(DEFAULT_MAX_ROWS);
        self.database
            .query(query)
            .map(query_result_json)
            .map_err(|error| format!("IronGraph schema discovery failed: {error}"))
    }

    fn save_document(&self, arguments: Value) -> Result<Value, String> {
        let input: SaveDocumentInput = parse_arguments(arguments)?;
        let document_id = input
            .document_id
            .unwrap_or_else(|| Uuid::new_v4().to_string());
        let cypher = format!(
            "USE {} MERGE (document:Document {{id: $document_id}}) SET document.title = $title, document.body = $body, document.source = $source RETURN document",
            cypher_identifier(&input.project)?
        );
        let mut query = Query::new(cypher);
        query.parameters = BTreeMap::from([
            ("document_id".to_owned(), json!(document_id)),
            ("title".to_owned(), json!(input.title)),
            ("body".to_owned(), json!(input.body)),
            ("source".to_owned(), json!(input.source)),
        ]);
        query.limits = result_limits(1);
        self.database
            .query(query)
            .map(|result| {
                json!({
                    "document_id": document_id,
                    "result": query_result_json(result)
                })
            })
            .map_err(|error| format!("IronGraph could not save the document: {error}"))
    }

    fn search(&self, arguments: Value) -> Result<Value, String> {
        let input: SearchInput = parse_arguments(arguments)?;
        let limit = bounded_rows(Some(input.limit.unwrap_or(10)))?.min(100);
        let project = cypher_identifier(&input.project)?;
        let index_name = input.index.as_deref().unwrap_or("graph_semantic");
        let index = cypher_identifier(index_name)?;
        if input.label.is_some() && input.index.is_none() {
            return Err("label requires the name of its declared embedding index; omit both to search all nodes and relationships".to_owned());
        }
        if input.label.is_none()
            && !matches!(
                index_name,
                "graph_semantic" | "semantic_nodes" | "semantic_relationships"
            )
        {
            return Err("a declared node embedding index requires its label; use graph_semantic, semantic_nodes, or semantic_relationships without a label".to_owned());
        }
        let bound = input
            .label
            .as_deref()
            .map(cypher_identifier)
            .transpose()?
            .map(|label| format!("MATCH (entity:{label}) "))
            .unwrap_or_default();
        let continuation = if input.include_connections && input.label.is_some() {
            "OPTIONAL MATCH (entity)-[relationship]-(neighbor) RETURN entity, score, collect(relationship) AS relationships, collect(neighbor) AS neighbors ORDER BY score DESC"
        } else {
            "RETURN entity, score ORDER BY score DESC"
        };
        let cypher = format!(
            "USE {project} {bound}SEARCH entity IN (EMBEDDING INDEX {index} FOR TEXT $query LIMIT {limit}) SCORE AS score {continuation}"
        );
        let mut query = Query::new(cypher);
        query.parameters = BTreeMap::from([("query".to_owned(), json!(input.query))]);
        query.limits = result_limits(limit);
        let result = self
            .database
            .query(query)
            .map_err(|error| format!("IronGraph semantic graph search failed: {error}"))?;
        if !input.include_connections || input.label.is_some() {
            return Ok(query_result_json(result));
        }
        // The semantic result can contain both entity kinds. Traverse each kind through its
        // proper Cypher pattern, retaining the original ranking and each context read's limits.
        let mut node_ids = Vec::new();
        let mut relationship_ids = Vec::new();
        let entity_column = result
            .columns
            .iter()
            .position(|column| column.name == "entity");
        if let Some(column) = entity_column {
            for row in &result.rows {
                let (id, ids) = match row.get(column) {
                    Some(TypedValue::Node(node)) => (&node.id, &mut node_ids),
                    Some(TypedValue::Relationship(relationship)) => {
                        (&relationship.id, &mut relationship_ids)
                    }
                    _ => continue,
                };
                ids.push(
                    id.parse::<u64>()
                        .map_err(|_| "IronGraph returned an invalid graph identity".to_owned())?,
                );
            }
        }
        let mut connections = serde_json::Map::new();
        for (kind, mut ids, pattern) in [
            (
                "nodes",
                node_ids,
                "MATCH (entity) WHERE id(entity) IN $ids OPTIONAL MATCH (entity)-[relationship]-(neighbor) RETURN entity, collect(relationship) AS relationships, collect(neighbor) AS neighbors",
            ),
            (
                "relationships",
                relationship_ids,
                "MATCH (source)-[entity]->(target) WHERE id(entity) IN $ids RETURN entity, [entity] AS relationships, [source, target] AS neighbors",
            ),
        ] {
            if ids.is_empty() {
                continue;
            }
            ids.sort_unstable();
            ids.dedup();
            let mut query = Query::new(format!("USE {project} {pattern}"));
            query.parameters.insert("ids".to_owned(), json!(ids));
            query.bookmark = result.summary.bookmark;
            query.limits = result_limits(limit);
            let context = self
                .database
                .query(query)
                .map_err(|error| format!("IronGraph semantic search context failed: {error}"))?;
            connections.insert(kind.to_owned(), query_result_json(context));
        }
        let mut output = query_result_json(result);
        output["connections"] = Value::Object(connections);
        Ok(output)
    }
}

/// Build the stateless Streamable HTTP MCP router used by local URL-based hosts.
pub fn http_router<D>(database: D) -> Router
where
    D: Database + Send + Sync + 'static,
{
    Router::new()
        .route("/mcp", post(http_mcp::<D>))
        .with_state(Arc::new(McpServer::new(database)))
}

/// Serve IronGraph MCP over a loopback-only Streamable HTTP listener.
pub async fn serve_http<D>(database: D, address: SocketAddr) -> anyhow::Result<()>
where
    D: Database + Send + Sync + 'static,
{
    if !address.ip().is_loopback() {
        anyhow::bail!("IronGraph MCP HTTP must bind to a loopback address");
    }
    let listener = tokio::net::TcpListener::bind(address).await?;
    axum::serve(listener, http_router(database)).await?;
    Ok(())
}

async fn http_mcp<D>(
    State(server): State<Arc<McpServer<D>>>,
    headers: HeaderMap,
    Json(request): Json<Value>,
) -> Response
where
    D: Database + Send + Sync + 'static,
{
    if !origin_is_local(&headers) {
        return (
            StatusCode::FORBIDDEN,
            "MCP HTTP accepts only loopback origins",
        )
            .into_response();
    }
    match tokio::task::spawn_blocking(move || server.handle_value(request)).await {
        Ok(Some(response)) => Json(response).into_response(),
        Ok(None) => StatusCode::ACCEPTED.into_response(),
        Err(error) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("MCP request task failed: {error}"),
        )
            .into_response(),
    }
}

fn origin_is_local(headers: &HeaderMap) -> bool {
    let Some(origin) = headers.get(ORIGIN) else {
        return true;
    };
    let Ok(origin) = origin.to_str() else {
        return false;
    };
    let Ok(uri) = origin.parse::<axum::http::Uri>() else {
        return false;
    };
    uri.host().is_some_and(|host| {
        host.eq_ignore_ascii_case("localhost")
            || host
                .parse::<IpAddr>()
                .is_ok_and(|address| address.is_loopback())
    })
}

#[derive(Debug, Deserialize)]
struct RpcRequest {
    #[allow(dead_code)]
    jsonrpc: Option<String>,
    id: Option<Value>,
    method: String,
    #[serde(default)]
    params: Value,
}

#[derive(Debug, Deserialize)]
struct ToolCall {
    name: String,
    #[serde(default)]
    arguments: Value,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RunCypherInput {
    cypher: String,
    #[serde(default)]
    parameters: BTreeMap<String, Value>,
    max_rows: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct GetSchemaInput {
    project: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SaveDocumentInput {
    project: String,
    body: String,
    #[serde(default)]
    title: Option<String>,
    #[serde(default)]
    source: Option<String>,
    document_id: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchInput {
    project: String,
    index: Option<String>,
    label: Option<String>,
    query: String,
    limit: Option<u64>,
    #[serde(default = "default_true")]
    include_connections: bool,
}

const fn default_true() -> bool {
    true
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct SearchDocsInput {
    query: String,
    limit: Option<usize>,
}

fn initialize(params: &Value) -> Value {
    let requested = params.get("protocolVersion").and_then(Value::as_str);
    let version = match requested {
        Some("2025-06-18" | "2025-03-26" | "2024-11-05") => requested.unwrap_or(PROTOCOL_VERSION),
        _ => PROTOCOL_VERSION,
    };
    json!({
        "protocolVersion": version,
        "serverInfo": { "name": "irongraph-second-brain", "version": env!("CARGO_PKG_VERSION") },
        "capabilities": {
            "tools": { "listChanged": false },
            "resources": { "subscribe": false, "listChanged": false }
        },
        "instructions": concat!(
            "Use IronGraph proactively as the user's durable private second brain and graph computation surface. ",
            "Before asking the user to repeat durable context, search the relevant project. Save facts, decisions, documents, entities and relationships when the user asks to remember them or when durable reuse is clearly intended. ",
            "IronGraph has no implicit default project: call irongraph_get_schema and name the project in every operation. Never guess labels, relationship types, properties, indexes or Cypher syntax. ",
            "Use irongraph_search_cypher_docs when syntax is uncertain, then read the returned irongraph:// resource. Use irongraph_run_cypher for every direct read, write, administration statement, graph algorithm and exact calculation over stored data; do not split reads from writes or export stored data into host-side scripts. ",
            "Use irongraph_save_document for durable source text and irongraph_search for automatic semantic recall across nodes and relationships, including documents and their graph context. No index or label is required for default semantic search. Confirm destructive Cypher with the user before execution. Exact query rows are authoritative; summarize them without inventing missing facts."
        )
    })
}

fn tool_definitions() -> Vec<Value> {
    vec![
        json!({
            "name": "irongraph_run_cypher",
            "title": "Run Cypher in IronGraph",
            "description": "The single direct database tool for IronGraph. Use it for reads, writes, administration, graph traversal, graph algorithms, temporal analysis, vector work, Streams and Queues—never invent a read/write tool split. Run computation where the stored graph lives instead of copying rows into host code. Every query must select a project with USE except SHOW PROJECTS and project administration. Call irongraph_get_schema first against unfamiliar data; call irongraph_search_cypher_docs before guessing syntax. Parameters are strongly preferred over interpolating values. The result is bounded but otherwise exact and includes typed rows, catalog, update counts, timing, bookmark and truncation state. Confirm CLEAR, PURGE, DROP, DELETE and other difficult-to-reverse statements with the user before calling.",
            "inputSchema": {
                "type": "object",
                "required": ["cypher"],
                "properties": {
                    "cypher": { "type": "string", "description": "Exactly one IronGraph Cypher statement. Include USE <project> unless the statement manages or lists projects." },
                    "parameters": { "type": "object", "description": "Named $parameter values. Use these for all user data and document text." },
                    "max_rows": { "type": "integer", "minimum": 1, "maximum": MAX_MAX_ROWS, "default": DEFAULT_MAX_ROWS, "description": "Maximum rows returned to the host. This bounds output, not execution intermediates." }
                },
                "additionalProperties": false
            },
            "annotations": { "readOnlyHint": false, "destructiveHint": true, "idempotentHint": false, "openWorldHint": false }
        }),
        json!({
            "name": "irongraph_get_schema",
            "title": "Read the live IronGraph catalog",
            "description": "Call this before writing Cypher against an unfamiliar IronGraph project. With no project it returns the real project list. With a project it returns that project's current labels, relationship types, property names, functions, indexes, schema revision and stable project identity directly from the query catalog. Never guess these names and never assume a default project. Follow with SHOW INDEXES or a small bounded MATCH through irongraph_run_cypher when index state, property types or connection patterns matter.",
            "inputSchema": {
                "type": "object",
                "properties": { "project": { "type": "string", "description": "Exact project display name. Omit only to list projects." } },
                "additionalProperties": false
            },
            "annotations": { "readOnlyHint": true, "destructiveHint": false, "idempotentHint": true, "openWorldHint": false }
        }),
        json!({
            "name": "irongraph_save_document",
            "title": "Save durable source text",
            "description": "Persist or update complete source text as an ordinary (:Document) graph node in the named project. Use this when the user asks IronGraph to remember a note, decision, document, source excerpt or reusable context. Reusing document_id updates the same document; omitting it creates a stable UUID. The full body remains canonical graph data in the normal WAL/snapshot path. When the local embedding model is enabled, meaningful document content is embedded automatically and kept current. No placeholder vector or index declaration is required before using irongraph_search.",
            "inputSchema": {
                "type": "object",
                "required": ["project", "body"],
                "properties": {
                    "project": { "type": "string", "description": "Exact existing project display name." },
                    "body": { "type": "string", "description": "Complete canonical source text. Do not summarize unless the user asked for a summary." },
                    "title": { "type": ["string", "null"] },
                    "source": { "type": ["string", "null"], "description": "Human-readable provenance such as a filename, URL or conversation." },
                    "document_id": { "type": "string", "description": "Stable identity to update. Omit to create a new UUID." }
                },
                "additionalProperties": false
            },
            "annotations": { "readOnlyHint": false, "destructiveHint": false, "idempotentHint": true, "openWorldHint": false }
        }),
        json!({
            "name": "irongraph_search",
            "title": "Search the graph by meaning",
            "description": "Search graph meaning using only project and query. Automatic embeddings cover nodes and relationships, including document text and other meaningful properties; metadata and opaque identifiers are excluded. Results contain complete matched entities and similarity scores in descending order. Use this proactively before asking the user to repeat stored knowledge. Optionally select semantic_nodes or semantic_relationships, or supply both a declared node embedding index and its label. With include_connections, mixed search adds bounded connections.nodes and connections.relationships query results: node hits include incident relationships and neighbors; relationship hits include their endpoints. Each context result reports its own truncation and may reflect newer committed data. Declared node searches include context in their result rows. Local embedding must be enabled.",
            "inputSchema": {
                "type": "object",
                "required": ["project", "query"],
                "properties": {
                    "project": { "type": "string", "description": "Exact project display name." },
                    "index": { "type": "string", "description": "Omit for all nodes and relationships. Use semantic_nodes or semantic_relationships to select one entity kind, or provide a declared node embedding index together with label." },
                    "label": { "type": "string", "description": "Node label for an explicitly declared embedding index. Omit for automatic semantic search." },
                    "query": { "type": "string", "description": "Natural-language meaning to retrieve." },
                    "limit": { "type": "integer", "minimum": 1, "maximum": 100, "default": 10 },
                    "include_connections": { "type": "boolean", "default": true, "description": "Include bounded node adjacency and relationship endpoint context. Mixed search returns separate connections query results; each reports truncation." }
                },
                "additionalProperties": false
            },
            "annotations": { "readOnlyHint": true, "destructiveHint": false, "idempotentHint": true, "openWorldHint": false }
        }),
        json!({
            "name": "irongraph_search_cypher_docs",
            "title": "Search the IronGraph Cypher reference",
            "description": "Search IronGraph's complete bundled Cypher reference in plain words before inventing syntax. It covers the exact implemented dialect: projects, layers, temporal clauses, constraints, indexes, text and embedding search, vector functions, aggregates and graph procedures. Results return focused summaries and irongraph:// resource URIs; read the best matching resource before composing unfamiliar Cypher. Standard Cypher knowledge is not enough for IronGraph extensions, and examples in this reference are executable product examples.",
            "inputSchema": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": { "type": "string", "description": "Capability, syntax name, error phrase or goal in natural language." },
                    "limit": { "type": "integer", "minimum": 1, "maximum": 10, "default": 5 }
                },
                "additionalProperties": false
            },
            "annotations": { "readOnlyHint": true, "destructiveHint": false, "idempotentHint": true, "openWorldHint": false }
        }),
    ]
}

fn list_resources() -> Value {
    let resources = CYPHER_DOCS
        .iter()
        .map(|doc| {
            json!({
                "uri": format!("{RESOURCE_PREFIX}{}", doc.path),
                "name": doc.path,
                "title": doc_title(doc),
                "description": doc_summary(doc),
                "mimeType": "text/markdown"
            })
        })
        .collect::<Vec<_>>();
    json!({ "resources": resources })
}

fn read_resource(params: &Value) -> Result<Value, String> {
    let uri = params
        .get("uri")
        .and_then(Value::as_str)
        .ok_or_else(|| "resources/read requires a string uri".to_owned())?;
    let path = uri
        .strip_prefix(RESOURCE_PREFIX)
        .ok_or_else(|| "resource URI is not an IronGraph Cypher page".to_owned())?;
    let doc = CYPHER_DOCS
        .iter()
        .find(|doc| doc.path == path)
        .ok_or_else(|| format!("unknown IronGraph Cypher resource: {uri}"))?;
    Ok(json!({
        "contents": [{ "uri": uri, "mimeType": "text/markdown", "text": doc.markdown }]
    }))
}

fn search_cypher_docs(arguments: Value) -> Result<Value, String> {
    let input: SearchDocsInput = parse_arguments(arguments)?;
    let terms = input
        .query
        .split(|character: char| !character.is_alphanumeric() && character != '.')
        .filter(|term| term.len() > 1)
        .map(str::to_ascii_lowercase)
        .collect::<Vec<_>>();
    if terms.is_empty() {
        return Err("documentation search query has no searchable terms".to_owned());
    }
    let limit = input.limit.unwrap_or(5).clamp(1, 10);
    let mut matches = CYPHER_DOCS
        .iter()
        .filter_map(|doc| {
            let title = doc_title(doc);
            let title_lower = title.to_ascii_lowercase();
            let path_lower = doc.path.to_ascii_lowercase();
            let body_lower = doc.markdown.to_ascii_lowercase();
            let score = terms.iter().fold(0_u64, |score, term| {
                score
                    + u64::from(title_lower.contains(term)) * 20
                    + u64::from(path_lower.contains(term)) * 10
                    + u64::try_from(body_lower.matches(term).count())
                        .unwrap_or(u64::MAX)
                        .min(20)
            });
            (score > 0).then_some((score, doc, title))
        })
        .collect::<Vec<_>>();
    matches.sort_by(|left, right| {
        right
            .0
            .cmp(&left.0)
            .then_with(|| left.1.path.cmp(right.1.path))
    });
    let matches = matches
        .into_iter()
        .take(limit)
        .map(|(score, doc, title)| {
            json!({
                "title": title,
                "path": doc.path,
                "summary": doc_summary(doc),
                "resource_uri": format!("{RESOURCE_PREFIX}{}", doc.path),
                "relevance": score
            })
        })
        .collect::<Vec<_>>();
    Ok(json!({
        "query": input.query,
        "matches": matches,
        "instruction": "Read the best matching resource with resources/read before writing unfamiliar IronGraph Cypher."
    }))
}

fn parse_arguments<T: for<'de> Deserialize<'de>>(arguments: Value) -> Result<T, String> {
    serde_json::from_value(arguments).map_err(|error| format!("invalid tool arguments: {error}"))
}

fn bounded_rows(value: Option<u64>) -> Result<u64, String> {
    let value = value.unwrap_or(DEFAULT_MAX_ROWS);
    if value == 0 || value > MAX_MAX_ROWS {
        return Err(format!("max_rows must be between 1 and {MAX_MAX_ROWS}"));
    }
    Ok(value)
}

const fn result_limits(rows: u64) -> QueryLimits {
    QueryLimits {
        rows,
        bytes: MAX_RESULT_BYTES,
        nodes: rows.saturating_mul(10),
        edges: rows.saturating_mul(20),
    }
}

fn cypher_identifier(value: &str) -> Result<String, String> {
    if value.is_empty() || value.len() > 255 || value.chars().any(char::is_control) {
        return Err(
            "project and index names must be 1-255 characters without control characters"
                .to_owned(),
        );
    }
    Ok(format!("`{}`", value.replace('`', "``")))
}

fn query_result_json(result: QueryResult) -> Value {
    let rows = result
        .rows
        .into_iter()
        .map(|row| {
            result
                .columns
                .iter()
                .zip(row)
                .map(|(column, value)| (column.name.clone(), typed_value_json(value)))
                .collect::<serde_json::Map<_, _>>()
        })
        .map(Value::Object)
        .collect::<Vec<_>>();
    json!({
        "catalog": result.catalog,
        "columns": result.columns,
        "rows": rows,
        "summary": result.summary
    })
}

fn typed_value_json(value: TypedValue) -> Value {
    match value {
        TypedValue::Null => Value::Null,
        TypedValue::Boolean(value) => json!(value),
        TypedValue::Integer(value) => value
            .parse::<i64>()
            .map_or_else(|_| json!(value), |value| json!(value)),
        TypedValue::Float(value) => json!(value),
        TypedValue::String(value) => json!(value),
        TypedValue::Bytes(value) => json!({ "_type": "bytes", "value": value }),
        TypedValue::Date(days) => json!({ "_type": "date", "days_since_epoch": days }),
        TypedValue::Time {
            nanos,
            offset_seconds,
        } => json!({ "_type": "time", "nanos": nanos, "offset_seconds": offset_seconds }),
        TypedValue::DateTime {
            seconds,
            nanos,
            timezone,
        } => {
            json!({ "_type": "datetime", "seconds": seconds, "nanos": nanos, "timezone": timezone })
        }
        TypedValue::Duration {
            months,
            days,
            seconds,
            nanos,
        } => {
            json!({ "_type": "duration", "months": months, "days": days, "seconds": seconds, "nanos": nanos })
        }
        TypedValue::Vector(value) => json!({ "_type": "vector", "value": value }),
        TypedValue::Node(value) => node_json(value),
        TypedValue::Relationship(value) => relationship_json(value),
        TypedValue::Path(value) => path_json(value),
        TypedValue::List(values) => {
            Value::Array(values.into_iter().map(typed_value_json).collect())
        }
        TypedValue::Map(values) => Value::Object(
            values
                .into_iter()
                .map(|(key, value)| (key, typed_value_json(value)))
                .collect(),
        ),
    }
}

fn node_json(node: ResultNode) -> Value {
    json!({
        "_type": "node",
        "id": node.id,
        "labels": node.labels,
        "properties": node.properties.into_iter().map(|(key, value)| (key, typed_value_json(value))).collect::<serde_json::Map<_, _>>()
    })
}

fn relationship_json(relationship: RelationshipValue) -> Value {
    json!({
        "_type": "relationship",
        "id": relationship.id,
        "source": relationship.source,
        "target": relationship.target,
        "relationship_type": relationship.relationship_type,
        "properties": relationship.properties.into_iter().map(|(key, value)| (key, typed_value_json(value))).collect::<serde_json::Map<_, _>>()
    })
}

fn path_json(path: PathValue) -> Value {
    json!({
        "_type": "path",
        "nodes": path.nodes.into_iter().map(node_json).collect::<Vec<_>>(),
        "relationships": path.relationships.into_iter().map(relationship_json).collect::<Vec<_>>()
    })
}

fn doc_title(doc: &CypherDoc) -> String {
    doc.markdown
        .lines()
        .find_map(|line| line.strip_prefix("# "))
        .unwrap_or(doc.path)
        .replace('`', "")
}

fn doc_summary(doc: &CypherDoc) -> String {
    doc.markdown
        .lines()
        .find_map(|line| line.strip_prefix("> "))
        .or_else(|| {
            doc.markdown.lines().find(|line| {
                let line = line.trim();
                !line.is_empty() && !line.starts_with('#') && !line.starts_with('|')
            })
        })
        .unwrap_or("IronGraph Cypher reference page")
        .trim()
        .to_owned()
}

fn tool_success(structured: Value) -> Value {
    let text = serde_json::to_string_pretty(&structured).unwrap_or_else(|_| structured.to_string());
    json!({
        "content": [{ "type": "text", "text": text }],
        "structuredContent": structured,
        "isError": false
    })
}

fn tool_failure(message: String) -> Value {
    json!({
        "content": [{ "type": "text", "text": message }],
        "isError": true
    })
}

fn rpc_error(id: Value, code: i64, message: String) -> Value {
    json!({ "jsonrpc": "2.0", "id": id, "error": { "code": code, "message": message } })
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use axum::{
        body::{Body, to_bytes},
        http::Request,
    };
    use tower::ServiceExt as _;

    use super::*;

    struct FakeDatabase {
        queries: Mutex<Vec<Query>>,
    }

    impl FakeDatabase {
        fn new() -> Self {
            Self {
                queries: Mutex::new(Vec::new()),
            }
        }
    }

    impl Database for FakeDatabase {
        fn query(&self, query: Query) -> irongraph_client::Result<QueryResult> {
            self.queries
                .lock()
                .map_err(|error| irongraph_client::ClientError::MalformedResult(error.to_string()))?
                .push(query);
            Ok(QueryResult::default())
        }
    }

    fn request(method: &str, params: Value) -> RpcRequest {
        RpcRequest {
            jsonrpc: Some("2.0".to_owned()),
            id: Some(json!(1)),
            method: method.to_owned(),
            params,
        }
    }

    async fn http_request(app: Router, method: &str, origin: Option<&str>) -> Response {
        let mut builder = Request::post("/mcp").header("content-type", "application/json");
        if let Some(origin) = origin {
            builder = builder.header("origin", origin);
        }
        app.oneshot(
            builder
                .body(Body::from(
                    json!({"jsonrpc":"2.0","id":1,"method":method,"params":{}}).to_string(),
                ))
                .expect("HTTP MCP request"),
        )
        .await
        .expect("HTTP MCP response")
    }

    #[tokio::test]
    async fn http_transport_serves_initialize_and_tools() {
        let app = http_router(FakeDatabase::new());
        let initialize = http_request(app.clone(), "initialize", None).await;
        assert_eq!(initialize.status(), StatusCode::OK);
        let initialize = to_bytes(initialize.into_body(), 1024 * 1024)
            .await
            .expect("initialize body");
        let initialize: Value = serde_json::from_slice(&initialize).expect("initialize JSON");
        assert_eq!(
            initialize["result"]["serverInfo"]["name"],
            "irongraph-second-brain"
        );

        let tools = http_request(app, "tools/list", Some("http://127.0.0.1:8888")).await;
        assert_eq!(tools.status(), StatusCode::OK);
        let tools = to_bytes(tools.into_body(), 1024 * 1024)
            .await
            .expect("tools body");
        let tools: Value = serde_json::from_slice(&tools).expect("tools JSON");
        assert_eq!(tools["result"]["tools"].as_array().map(Vec::len), Some(5));
    }

    #[tokio::test]
    async fn http_transport_rejects_non_loopback_browser_origins() {
        let response = http_request(
            http_router(FakeDatabase::new()),
            "initialize",
            Some("https://example.com"),
        )
        .await;
        assert_eq!(response.status(), StatusCode::FORBIDDEN);
    }

    #[test]
    fn initialize_commits_the_host_to_the_second_brain_workflow() {
        let server = McpServer::new(FakeDatabase::new());
        let response = server
            .handle(request("initialize", json!({})))
            .expect("response");
        let instructions = response["result"]["instructions"]
            .as_str()
            .expect("instructions");
        assert!(instructions.contains("proactively"));
        assert!(instructions.contains("irongraph_run_cypher"));
        assert!(instructions.contains("no implicit default project"));
    }

    #[test]
    fn exposes_one_unified_cypher_tool_and_focused_memory_tools() {
        let definitions = tool_definitions();
        assert_eq!(definitions.len(), 5);
        assert!(
            definitions
                .iter()
                .any(|tool| tool["name"] == "irongraph_run_cypher")
        );
        assert!(
            !definitions
                .iter()
                .any(|tool| tool["name"] == "irongraph_read_cypher")
        );
        assert!(
            !definitions
                .iter()
                .any(|tool| tool["name"] == "irongraph_write_cypher")
        );
        assert!(
            definitions
                .iter()
                .any(|tool| tool["name"] == "irongraph_search")
        );
        assert!(definitions.iter().all(|tool| {
            tool["description"]
                .as_str()
                .is_some_and(|description| description.len() > 180)
        }));
    }

    #[test]
    fn documentation_search_returns_readable_resources() {
        let result =
            search_cypher_docs(json!({ "query": "semantic embedding search", "limit": 3 }))
                .expect("search");
        let uri = result["matches"][0]["resource_uri"]
            .as_str()
            .expect("resource URI");
        let read = read_resource(&json!({ "uri": uri })).expect("resource");
        assert!(
            read["contents"][0]["text"]
                .as_str()
                .is_some_and(|text| text.contains("SEARCH"))
        );
    }

    #[test]
    fn semantic_reference_examples_use_parseable_current_cypher() {
        let paths = [
            "clauses/search/create-embedding-index.md",
            "clauses/search/search.md",
            "statements/indexes/show-indexes.md",
        ];
        let mut examples = 0;
        for path in paths {
            let doc = CYPHER_DOCS
                .iter()
                .find(|doc| doc.path == path)
                .expect("reference page");
            for block in doc.markdown.split("```cypher\n").skip(1) {
                let cypher = block.split("```").next().expect("Cypher example");
                assert!(
                    irongraph_cypher::parse(cypher).is_ok(),
                    "invalid example in {path}: {cypher}"
                );
                examples += 1;
            }
            assert!(!doc.markdown.contains("90%"));
            assert!(!doc.markdown.contains("recall floor"));
        }
        assert_eq!(examples, 6);
    }

    #[test]
    fn document_save_is_parameterized_and_uses_the_canonical_document_label() {
        let database = FakeDatabase::new();
        let server = McpServer::new(database);
        let result = server.save_document(
            json!({ "project": "memory", "body": "Keep this", "title": "Decision" }),
        );
        assert!(result.is_ok());
        let queries = server.database.queries.lock().expect("queries");
        assert_eq!(queries.len(), 1);
        assert!(queries[0].cypher.contains("MERGE (document:Document"));
        assert_eq!(queries[0].parameters["body"], json!("Keep this"));
        assert!(!queries[0].cypher.contains("Keep this"));
        assert!(!queries[0].cypher.contains("embedding"));
        assert!(!queries[0].parameters.contains_key("embedding"));
    }

    #[test]
    fn semantic_search_returns_nodes_edges_and_neighbors_without_interpolating_text() {
        let database = FakeDatabase::new();
        let server = McpServer::new(database);
        let result = server.search(json!({ "project": "memory", "index": "knowledge_semantic", "label": "Knowledge", "query": "project decision" }));
        assert!(result.is_ok());
        let queries = server.database.queries.lock().expect("queries");
        assert!(queries[0].cypher.contains(
            "MATCH (entity:`Knowledge`) SEARCH entity IN (EMBEDDING INDEX `knowledge_semantic` FOR TEXT $query LIMIT 10)"
        ));
        assert!(
            queries[0]
                .cypher
                .contains("OPTIONAL MATCH (entity)-[relationship]-(neighbor)")
        );
        assert!(
            queries[0]
                .cypher
                .contains("collect(relationship) AS relationships")
        );
        assert!(irongraph_cypher::parse(&queries[0].cypher).is_ok());
        assert_eq!(queries[0].parameters["query"], json!("project decision"));
    }

    #[test]
    fn semantic_search_can_return_only_ranked_nodes() {
        let database = FakeDatabase::new();
        let server = McpServer::new(database);
        let result = server.search(json!({
            "project": "memory",
            "index": "document_semantic",
            "label": "Document",
            "query": "project decision",
            "include_connections": false
        }));
        assert!(result.is_ok());
        let queries = server.database.queries.lock().expect("queries");
        assert!(!queries[0].cypher.contains("OPTIONAL MATCH"));
        assert!(
            queries[0]
                .cypher
                .ends_with("RETURN entity, score ORDER BY score DESC")
        );
        assert!(irongraph_cypher::parse(&queries[0].cypher).is_ok());
    }

    #[test]
    fn semantic_search_defaults_to_ranked_mixed_entities_without_setup() {
        let server = McpServer::new(FakeDatabase::new());
        let result = server.search(
            json!({"project": "memory", "query": "launch plans", "include_connections": false}),
        );
        assert!(result.is_ok());
        let queries = server.database.queries.lock().expect("queries");
        assert_eq!(queries.len(), 1);
        assert_eq!(
            queries[0].cypher,
            "USE `memory` SEARCH entity IN (EMBEDDING INDEX `graph_semantic` FOR TEXT $query LIMIT 10) SCORE AS score RETURN entity, score ORDER BY score DESC"
        );
        assert_eq!(queries[0].parameters["query"], json!("launch plans"));
        assert!(irongraph_cypher::parse(&queries[0].cypher).is_ok());
        let definitions = tool_definitions();
        let tool = definitions
            .iter()
            .find(|tool| tool["name"] == "irongraph_search")
            .expect("search tool");
        assert_eq!(tool["inputSchema"]["required"], json!(["project", "query"]));
    }

    #[test]
    fn semantic_search_supports_entity_kind_selection_and_rejects_incomplete_field_selection() {
        for index in ["semantic_nodes", "semantic_relationships"] {
            let server = McpServer::new(FakeDatabase::new());
            assert!(server.search(json!({"project": "memory", "index": index, "query": "launch", "include_connections": false})).is_ok());
            let queries = server.database.queries.lock().expect("queries");
            assert!(
                queries[0]
                    .cypher
                    .contains(&format!("EMBEDDING INDEX `{index}`"))
            );
            assert!(!queries[0].cypher.contains("MATCH"));
            assert!(irongraph_cypher::parse(&queries[0].cypher).is_ok());
        }
        let server = McpServer::new(FakeDatabase::new());
        assert!(
            server
                .search(json!({"project": "memory", "label": "Document", "query": "launch"}))
                .is_err()
        );
        assert!(
            server
                .search(json!({"project": "memory", "index": "document_body", "query": "launch"}))
                .is_err()
        );
        assert!(server.database.queries.lock().expect("queries").is_empty());
    }

    #[test]
    fn mixed_semantic_connections_use_entity_specific_patterns_and_preserve_ranking() {
        use irongraph_server::protocol::QueryColumn;

        struct MixedDatabase {
            queries: Mutex<Vec<Query>>,
        }
        impl Database for MixedDatabase {
            fn query(&self, query: Query) -> irongraph_client::Result<QueryResult> {
                let mut queries = self.queries.lock().expect("queries");
                queries.push(query);
                if queries.len() != 1 {
                    return Ok(QueryResult::default());
                }
                let mut result = QueryResult {
                    columns: vec![
                        QueryColumn {
                            name: "entity".into(),
                            value_type: "ANY".into(),
                            nullable: false,
                        },
                        QueryColumn {
                            name: "score".into(),
                            value_type: "FLOAT".into(),
                            nullable: false,
                        },
                    ],
                    rows: vec![
                        vec![
                            TypedValue::Relationship(RelationshipValue {
                                id: "11".into(),
                                source: "1".into(),
                                target: "2".into(),
                                relationship_type: "OWNS".into(),
                                properties: BTreeMap::new(),
                            }),
                            TypedValue::Float(0.9),
                        ],
                        vec![
                            TypedValue::Node(ResultNode {
                                id: "11".into(),
                                labels: vec!["Person".into()],
                                properties: BTreeMap::new(),
                            }),
                            TypedValue::Float(0.8),
                        ],
                    ],
                    ..QueryResult::default()
                };
                result.summary.bookmark =
                    Some(serde_json::from_value(json!({"term": 1, "index": 7})).expect("bookmark"));
                Ok(result)
            }
        }
        let server = McpServer::new(MixedDatabase {
            queries: Mutex::new(Vec::new()),
        });
        let result = server
            .search(json!({"project": "memory", "query": "launch"}))
            .expect("mixed search");
        assert_eq!(result["rows"][0]["entity"]["_type"], "relationship");
        assert_eq!(result["rows"][0]["score"], 0.9);
        assert_eq!(result["rows"][1]["entity"]["_type"], "node");
        assert!(result["connections"]["nodes"].is_object());
        assert!(result["connections"]["relationships"].is_object());
        let queries = server.database.queries.lock().expect("queries");
        assert_eq!(queries.len(), 3);
        assert!(
            queries[1]
                .cypher
                .contains("MATCH (entity) WHERE id(entity) IN $ids")
        );
        assert!(
            queries[2]
                .cypher
                .contains("MATCH (source)-[entity]->(target)")
        );
        assert!(queries[2].cypher.contains("[source, target] AS neighbors"));
        for query in queries.iter().skip(1) {
            assert_eq!(query.parameters["ids"], json!([11]));
            assert_eq!(query.limits.rows, 10);
            assert_eq!(query.bookmark.map(|bookmark| bookmark.index), Some(7));
            assert!(irongraph_cypher::parse(&query.cypher).is_ok());
        }
        let server = McpServer::new(MixedDatabase {
            queries: Mutex::new(Vec::new()),
        });
        let result = server
            .search(json!({"project": "memory", "query": "launch", "include_connections": false}))
            .expect("ranked search");
        assert_eq!(result["rows"].as_array().map(Vec::len), Some(2));
        assert!(result.get("connections").is_none());
        assert_eq!(server.database.queries.lock().expect("queries").len(), 1);
    }

    #[test]
    fn project_and_index_names_are_quoted_as_cypher_identifiers() {
        assert_eq!(
            cypher_identifier("team memory").expect("identifier"),
            "`team memory`"
        );
        assert_eq!(cypher_identifier("a`b").expect("identifier"), "`a``b`");
        assert!(cypher_identifier("bad\nname").is_err());
    }
}
