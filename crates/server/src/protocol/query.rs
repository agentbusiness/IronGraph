use std::{collections::BTreeMap, sync::Arc, time::Instant};

use serde::{Deserialize, Serialize};
use tokio::sync::{OwnedSemaphorePermit, Semaphore};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

use crate::{Bookmark, CommitAcknowledgement, ProjectId, Result, storage::ConnectionId};

/// Browser/Bolt-neutral query request.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueryRequest {
    pub request_id: Uuid,
    pub project_id: Option<ProjectId>,
    pub query: String,
    #[serde(default)]
    pub parameters: BTreeMap<String, serde_json::Value>,
    #[serde(default)]
    pub consistency: CommitAcknowledgement,
    pub bookmark: Option<Bookmark>,
    #[serde(default)]
    pub limits: QueryLimits,
    /// In-process cancellation is assigned by the transport and never deserialized from a client.
    #[serde(skip, default)]
    pub cancellation: CancellationToken,
    /// Absolute in-process deadline assigned by the transport.
    #[serde(skip, default)]
    pub deadline: Option<Instant>,
    /// Transport-assigned connection identity for fair in-memory admission. It is never accepted
    /// from JSON and is unrelated to the idempotency request ID.
    #[serde(skip, default)]
    pub connection_id: ConnectionId,
}

impl QueryRequest {
    pub fn validate(&self) -> Result<()> {
        if self.query.trim().is_empty() || self.query.len() > 4 * 1024 * 1024 {
            return Err(crate::Error::invalid_data(
                "query is empty or exceeds 4 MiB",
            ));
        }
        if self.parameters.len() > 16_384 {
            return Err(crate::Error::invalid_data("query has too many parameters"));
        }
        let parameter_bytes = serde_json::to_vec(&self.parameters).map_err(|error| {
            crate::Error::invalid_data(format!("parameter encoding failed: {error}"))
        })?;
        if parameter_bytes.len() > 16 * 1024 * 1024 {
            return Err(crate::Error::invalid_data("query parameters exceed 16 MiB"));
        }
        self.limits.validate()
    }
}

const QUERY_ADMISSION_BYTE_QUANTUM: usize = 64 * 1024;

/// Bounds transport workers that outlive their HTTP handler while streaming results. The byte
/// semaphore accounts decoded query text and parameters before a blocking executor is spawned.
pub(crate) struct QueryIngressAdmission {
    requests: Arc<Semaphore>,
    byte_units: Arc<Semaphore>,
    retry_after_ms: u64,
}

impl QueryIngressAdmission {
    pub(crate) fn new(maximum_requests: usize, maximum_bytes: usize) -> Result<Self> {
        let units = maximum_bytes
            .checked_add(QUERY_ADMISSION_BYTE_QUANTUM - 1)
            .map(|bytes| bytes / QUERY_ADMISSION_BYTE_QUANTUM)
            .and_then(|units| u32::try_from(units).ok())
            .ok_or_else(|| crate::Error::invalid_data("query admission byte limit is invalid"))?;
        if maximum_requests == 0 || units == 0 {
            return Err(crate::Error::invalid_data(
                "query admission limits must be positive",
            ));
        }
        Ok(Self {
            requests: Arc::new(Semaphore::new(maximum_requests)),
            byte_units: Arc::new(Semaphore::new(units as usize)),
            retry_after_ms: 25,
        })
    }

    pub(crate) fn try_admit(&self, request: &QueryRequest) -> Result<QueryIngressPermit> {
        request.validate()?;
        let parameter_bytes = serde_json::to_vec(&request.parameters)
            .map_err(|error| {
                crate::Error::invalid_data(format!("parameter encoding failed: {error}"))
            })?
            .len();
        let bytes = request
            .query
            .len()
            .checked_add(parameter_bytes)
            .and_then(|bytes| bytes.checked_add(256))
            .ok_or_else(|| crate::Error::invalid_data("query admission byte size overflow"))?;
        let units = bytes
            .checked_add(QUERY_ADMISSION_BYTE_QUANTUM - 1)
            .map(|bytes| bytes / QUERY_ADMISSION_BYTE_QUANTUM)
            .and_then(|units| u32::try_from(units).ok())
            .ok_or_else(|| crate::Error::invalid_data("query admission byte units overflow"))?
            .max(1);
        let request_permit = Arc::clone(&self.requests)
            .try_acquire_owned()
            .map_err(|_| self.full("query request admission is full"))?;
        let byte_permit = Arc::clone(&self.byte_units)
            .try_acquire_many_owned(units)
            .map_err(|_| self.full("query byte admission is full"))?;
        Ok(QueryIngressPermit {
            _request: request_permit,
            _bytes: byte_permit,
        })
    }

    fn full(&self, message: &'static str) -> crate::Error {
        crate::Error::retryable(
            crate::ErrorCode::WriteAdmissionFull,
            message,
            Some(self.retry_after_ms),
        )
    }
}

pub(crate) struct QueryIngressPermit {
    _request: OwnedSemaphorePermit,
    _bytes: OwnedSemaphorePermit,
}

const UNBOUNDED_QUERY_RESULT_LIMIT: u64 = u64::MAX;

const fn unbounded_query_result_limit() -> u64 {
    UNBOUNDED_QUERY_RESULT_LIMIT
}

fn query_result_limit_is_unbounded(limit: &u64) -> bool {
    *limit == UNBOUNDED_QUERY_RESULT_LIMIT
}

/// Optional response limits supplied by the caller. An omitted field is unbounded; the transport
/// and database do not replace it with a server policy. Explicit limits apply only to emitted
/// result events and never become query-planning or intermediate-relation bounds.
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct QueryLimits {
    #[serde(
        default = "unbounded_query_result_limit",
        skip_serializing_if = "query_result_limit_is_unbounded"
    )]
    pub rows: u64,
    #[serde(
        default = "unbounded_query_result_limit",
        skip_serializing_if = "query_result_limit_is_unbounded"
    )]
    pub bytes: u64,
    #[serde(
        default = "unbounded_query_result_limit",
        skip_serializing_if = "query_result_limit_is_unbounded"
    )]
    pub nodes: u64,
    #[serde(
        default = "unbounded_query_result_limit",
        skip_serializing_if = "query_result_limit_is_unbounded"
    )]
    pub edges: u64,
}

impl Default for QueryLimits {
    fn default() -> Self {
        Self {
            rows: UNBOUNDED_QUERY_RESULT_LIMIT,
            bytes: UNBOUNDED_QUERY_RESULT_LIMIT,
            nodes: UNBOUNDED_QUERY_RESULT_LIMIT,
            edges: UNBOUNDED_QUERY_RESULT_LIMIT,
        }
    }
}

impl QueryLimits {
    pub fn validate(&self) -> Result<()> {
        if self.rows == 0 || self.bytes == 0 || self.nodes == 0 || self.edges == 0 {
            return Err(crate::Error::invalid_data(
                "query result limits must be non-zero",
            ));
        }
        Ok(())
    }

    /// Applies the operator's ceiling on top of whatever the client asked for.
    ///
    /// A client may leave a result limit unset, which means unbounded, so a single query could
    /// materialize an arbitrarily large result on the server before anything noticed. The
    /// `--max-result-bytes` option exists to bound exactly that, but nothing read it, so the knob
    /// was advertised and inert. Clamping here keeps the client's own limit authoritative whenever
    /// it is stricter and applies the operator's otherwise.
    #[must_use]
    pub fn clamped_to_server_policy(mut self, maximum_result_bytes: u64) -> Self {
        if maximum_result_bytes != UNBOUNDED_QUERY_RESULT_LIMIT {
            self.bytes = self.bytes.min(maximum_result_bytes);
        }
        self
    }
}

/// Typed values shared by NDJSON and Bolt mapping.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", content = "value", rename_all = "snake_case")]
pub enum TypedValue {
    Null,
    Boolean(bool),
    Integer(String),
    Float(f64),
    String(String),
    Bytes(Vec<u8>),
    Date(i64),
    Time {
        nanos: i64,
        offset_seconds: Option<i32>,
    },
    DateTime {
        seconds: i64,
        nanos: u32,
        timezone: Option<String>,
    },
    Duration {
        months: i64,
        days: i64,
        seconds: i64,
        nanos: i32,
    },
    Vector(Vec<f32>),
    Node(ResultNode),
    Relationship(RelationshipValue),
    Path(PathValue),
    List(Vec<TypedValue>),
    Map(BTreeMap<String, TypedValue>),
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ResultNode {
    pub id: String,
    pub labels: Vec<String>,
    pub properties: BTreeMap<String, TypedValue>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RelationshipValue {
    pub id: String,
    pub source: String,
    pub target: String,
    pub relationship_type: String,
    pub properties: BTreeMap<String, TypedValue>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PathValue {
    pub nodes: Vec<ResultNode>,
    pub relationships: Vec<RelationshipValue>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueryColumn {
    pub name: String,
    pub value_type: String,
    pub nullable: bool,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BatchColumn {
    pub name: String,
    pub value_type: String,
    pub values: Vec<TypedValue>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CatalogEvent {
    pub project_id: Option<ProjectId>,
    pub schema_revision: u64,
    pub labels: Vec<String>,
    pub relationship_types: Vec<String>,
    pub properties: Vec<String>,
    pub functions: Vec<String>,
    pub indexes: Vec<String>,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct QueryStatistics {
    /// Whole milliseconds, kept because Bolt publishes it as an Integer and clients read it.
    /// Almost every read on this engine finishes inside one, so on its own it reports `0`.
    pub elapsed_ms: u64,
    /// The same measurement at the resolution the engine actually works at. Filled by
    /// [`QueryResultBudget::settle`], which holds the only clock that spans a whole request.
    #[serde(default)]
    pub elapsed_us: u64,
    pub rows: u64,
    /// Node and relationship *values carried by the result* — the accounting the byte and entity
    /// budgets are enforced against. A node appearing in three rows counts three times, so this is
    /// not the number of distinct entities a viewer would draw, and the two disagree on purpose.
    pub nodes: u64,
    pub edges: u64,
    pub updates: u64,
}

/// Strict NDJSON event union. A summary or error is terminal.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum QueryStreamEvent {
    Catalog {
        #[serde(flatten)]
        catalog: CatalogEvent,
    },
    Schema {
        request_id: Uuid,
        columns: Vec<QueryColumn>,
    },
    Batch {
        request_id: Uuid,
        sequence: u64,
        row_count: u64,
        columns: Vec<BatchColumn>,
    },
    Summary {
        request_id: Uuid,
        bookmark: Bookmark,
        statistics: QueryStatistics,
        truncated: bool,
        truncation_reason: Option<String>,
    },
    Error {
        request_id: Uuid,
        code: crate::ErrorCode,
        message: String,
        retryable: bool,
        retry_after_ms: Option<u64>,
    },
}

/// Per-request result accounting shared by every query transport. The executor emits through this
/// guard before a protocol adapter can buffer or encode an event.
pub(crate) struct QueryResultBudget {
    limits: QueryLimits,
    rows: u64,
    bytes: u64,
    nodes: u64,
    edges: u64,
    /// When this request started. The guard already spans exactly the work worth timing — it is
    /// constructed before the statement runs and observes the terminal summary — so the clock
    /// lives here rather than being threaded through every transport separately.
    started: Instant,
}

impl QueryResultBudget {
    #[must_use]
    pub(crate) fn new(limits: QueryLimits) -> Self {
        Self {
            limits,
            rows: 0,
            bytes: 0,
            nodes: 0,
            edges: 0,
            started: Instant::now(),
        }
    }

    /// Stamps a terminal summary with what the request actually did.
    ///
    /// `elapsed_ms`, `nodes` and `edges` were never written by any emitter — every summary was
    /// built as `QueryStatistics { rows, updates, ..default() }`, so all three arrived as a
    /// structural zero. Read as "this took no time and touched nothing", which is what the Graph
    /// screen printed: `0 ms` against four hundred rows. The measurement is taken here because this
    /// is the one place that sees a request from before it is planned to after its last row.
    pub(crate) fn settle(&self, event: &mut QueryStreamEvent) {
        let QueryStreamEvent::Summary { statistics, .. } = event else {
            return;
        };
        let elapsed = self.started.elapsed();
        statistics.elapsed_us = u64::try_from(elapsed.as_micros()).unwrap_or(u64::MAX);
        statistics.elapsed_ms = u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX);
        statistics.nodes = self.nodes;
        statistics.edges = self.edges;
    }

    pub(crate) fn observe(&mut self, event: &QueryStreamEvent) -> Result<()> {
        let encoded_bytes = serde_json::to_vec(event)
            .map_err(|error| crate::Error::internal(format!("result encoding failed: {error}")))?
            .len()
            .checked_add(1)
            .and_then(|bytes| u64::try_from(bytes).ok())
            .ok_or_else(|| {
                crate::Error::new(
                    crate::ErrorCode::ResultBudgetExceeded,
                    "query result byte accounting overflow",
                )
            })?;
        let bytes = self.bytes.checked_add(encoded_bytes).ok_or_else(|| {
            crate::Error::new(
                crate::ErrorCode::ResultBudgetExceeded,
                "query result byte accounting overflow",
            )
        })?;
        let (batch_rows, batch_nodes, batch_edges) = match event {
            QueryStreamEvent::Batch {
                row_count, columns, ..
            } => {
                let mut nodes = 0_u64;
                let mut edges = 0_u64;
                for value in columns.iter().flat_map(|column| &column.values) {
                    let (value_nodes, value_edges) = typed_graph_counts(value)?;
                    nodes = nodes.checked_add(value_nodes).ok_or_else(|| {
                        crate::Error::new(
                            crate::ErrorCode::ResultBudgetExceeded,
                            "query node accounting overflow",
                        )
                    })?;
                    edges = edges.checked_add(value_edges).ok_or_else(|| {
                        crate::Error::new(
                            crate::ErrorCode::ResultBudgetExceeded,
                            "query relationship accounting overflow",
                        )
                    })?;
                }
                (*row_count, nodes, edges)
            }
            QueryStreamEvent::Catalog { .. }
            | QueryStreamEvent::Schema { .. }
            | QueryStreamEvent::Summary { .. }
            | QueryStreamEvent::Error { .. } => (0, 0, 0),
        };
        let rows = self.rows.checked_add(batch_rows).ok_or_else(|| {
            crate::Error::new(
                crate::ErrorCode::ResultBudgetExceeded,
                "query row accounting overflow",
            )
        })?;
        let nodes = self.nodes.checked_add(batch_nodes).ok_or_else(|| {
            crate::Error::new(
                crate::ErrorCode::ResultBudgetExceeded,
                "query node accounting overflow",
            )
        })?;
        let edges = self.edges.checked_add(batch_edges).ok_or_else(|| {
            crate::Error::new(
                crate::ErrorCode::ResultBudgetExceeded,
                "query relationship accounting overflow",
            )
        })?;
        enforce_explicit_result_limit("rows", rows, self.limits.rows)?;
        enforce_explicit_result_limit("bytes", bytes, self.limits.bytes)?;
        enforce_explicit_result_limit("nodes", nodes, self.limits.nodes)?;
        enforce_explicit_result_limit("relationships", edges, self.limits.edges)?;
        self.rows = rows;
        self.bytes = bytes;
        self.nodes = nodes;
        self.edges = edges;
        Ok(())
    }
}

fn enforce_explicit_result_limit(kind: &str, observed: u64, limit: u64) -> Result<()> {
    if limit != UNBOUNDED_QUERY_RESULT_LIMIT && observed > limit {
        return Err(crate::Error::new(
            crate::ErrorCode::ResultBudgetExceeded,
            format!("query result exceeds the caller's {kind} limit ({observed}/{limit})"),
        ));
    }
    Ok(())
}

fn typed_graph_counts(value: &TypedValue) -> Result<(u64, u64)> {
    match value {
        TypedValue::Node(node) => {
            let (nested_nodes, nested_edges) = graph_counts_iter(node.properties.values())?;
            Ok((
                nested_nodes.checked_add(1).ok_or_else(|| {
                    crate::Error::new(
                        crate::ErrorCode::ResultBudgetExceeded,
                        "query node accounting overflow",
                    )
                })?,
                nested_edges,
            ))
        }
        TypedValue::Relationship(edge) => {
            let (nested_nodes, nested_edges) = graph_counts_iter(edge.properties.values())?;
            Ok((
                nested_nodes,
                nested_edges.checked_add(1).ok_or_else(|| {
                    crate::Error::new(
                        crate::ErrorCode::ResultBudgetExceeded,
                        "query relationship accounting overflow",
                    )
                })?,
            ))
        }
        TypedValue::Path(path) => {
            let mut nodes = u64::try_from(path.nodes.len()).map_err(|_| {
                crate::Error::new(
                    crate::ErrorCode::ResultBudgetExceeded,
                    "path node count does not fit result accounting",
                )
            })?;
            let mut edges = u64::try_from(path.relationships.len()).map_err(|_| {
                crate::Error::new(
                    crate::ErrorCode::ResultBudgetExceeded,
                    "path relationship count does not fit result accounting",
                )
            })?;
            for node in &path.nodes {
                let (nested_nodes, nested_edges) = graph_counts_iter(node.properties.values())?;
                nodes = nodes.checked_add(nested_nodes).ok_or_else(|| {
                    crate::Error::new(
                        crate::ErrorCode::ResultBudgetExceeded,
                        "path node accounting overflow",
                    )
                })?;
                edges = edges.checked_add(nested_edges).ok_or_else(|| {
                    crate::Error::new(
                        crate::ErrorCode::ResultBudgetExceeded,
                        "path relationship accounting overflow",
                    )
                })?;
            }
            for relationship in &path.relationships {
                let (nested_nodes, nested_edges) =
                    graph_counts_iter(relationship.properties.values())?;
                nodes = nodes.checked_add(nested_nodes).ok_or_else(|| {
                    crate::Error::new(
                        crate::ErrorCode::ResultBudgetExceeded,
                        "path node accounting overflow",
                    )
                })?;
                edges = edges.checked_add(nested_edges).ok_or_else(|| {
                    crate::Error::new(
                        crate::ErrorCode::ResultBudgetExceeded,
                        "path relationship accounting overflow",
                    )
                })?;
            }
            Ok((nodes, edges))
        }
        TypedValue::List(values) => graph_counts_iter(values),
        TypedValue::Map(values) => graph_counts_iter(values.values()),
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
        | TypedValue::Vector(_) => Ok((0, 0)),
    }
}

fn graph_counts_iter<'a>(values: impl IntoIterator<Item = &'a TypedValue>) -> Result<(u64, u64)> {
    let mut nodes = 0_u64;
    let mut edges = 0_u64;
    for value in values {
        let (value_nodes, value_edges) = typed_graph_counts(value)?;
        nodes = nodes.checked_add(value_nodes).ok_or_else(|| {
            crate::Error::new(
                crate::ErrorCode::ResultBudgetExceeded,
                "query node accounting overflow",
            )
        })?;
        edges = edges.checked_add(value_edges).ok_or_else(|| {
            crate::Error::new(
                crate::ErrorCode::ResultBudgetExceeded,
                "query relationship accounting overflow",
            )
        })?;
    }
    Ok((nodes, edges))
}

/// Synchronous query execution contract used by HTTP and Bolt adapters.
pub trait QueryExecutor: Send + Sync {
    /// Resolves Bolt's `db` selector to the immutable project boundary. Implementations backed by
    /// the database accept both the display name and immutable UUID; scoped wrappers must reject
    /// selectors outside their credential project. The default keeps lightweight test executors
    /// useful without inventing a catalog.
    fn resolve_project(&self, selector: &str) -> Result<ProjectId> {
        let id = Uuid::parse_str(selector).map_err(|_| {
            crate::Error::new(
                crate::ErrorCode::ProjectNotFound,
                "Bolt db does not identify a project",
            )
        })?;
        Ok(ProjectId(id))
    }

    fn execute(
        &self,
        request: QueryRequest,
        emit: &mut dyn FnMut(QueryStreamEvent) -> Result<()>,
    ) -> Result<()>;

    fn begin(
        &self,
        project: Option<ProjectId>,
        bookmark: Option<Bookmark>,
        consistency: CommitAcknowledgement,
    ) -> Result<Box<dyn QueryTransaction>>;

    fn begin_on_connection(
        &self,
        connection: ConnectionId,
        project: Option<ProjectId>,
        bookmark: Option<Bookmark>,
        consistency: CommitAcknowledgement,
    ) -> Result<Box<dyn QueryTransaction>> {
        let _ = connection;
        self.begin(project, bookmark, consistency)
    }
}

/// Serializable explicit transaction used by Bolt. Implementations pin and release snapshots on
/// every terminal path.
pub trait QueryTransaction: Send {
    fn run(
        &mut self,
        request: QueryRequest,
        emit: &mut dyn FnMut(QueryStreamEvent) -> Result<()>,
    ) -> Result<()>;

    fn commit(self: Box<Self>) -> Result<Bookmark>;

    fn rollback(self: Box<Self>) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: &str) -> TypedValue {
        TypedValue::Node(ResultNode {
            id: id.to_owned(),
            labels: vec!["Item".to_owned()],
            properties: BTreeMap::new(),
        })
    }

    #[test]
    fn query_limits_default_to_unbounded_and_omit_synthetic_limits_from_json() {
        let limits = QueryLimits::default();
        assert_eq!(limits.rows, u64::MAX);
        assert_eq!(limits.bytes, u64::MAX);
        assert_eq!(limits.nodes, u64::MAX);
        assert_eq!(limits.edges, u64::MAX);
        assert_eq!(serde_json::to_value(limits).unwrap(), serde_json::json!({}));

        let decoded: QueryLimits = serde_json::from_value(serde_json::json!({})).unwrap();
        assert_eq!(decoded.rows, u64::MAX);
        assert_eq!(decoded.bytes, u64::MAX);
        assert_eq!(decoded.nodes, u64::MAX);
        assert_eq!(decoded.edges, u64::MAX);
    }

    #[test]
    fn default_result_budget_crosses_former_server_clamps() {
        let request_id = Uuid::nil();
        let event = QueryStreamEvent::Batch {
            request_id,
            sequence: 0,
            row_count: 1,
            columns: vec![BatchColumn {
                name: "item".to_owned(),
                value_type: "NODE".to_owned(),
                values: vec![node("1")],
            }],
        };
        let mut budget = QueryResultBudget::new(QueryLimits::default());
        budget.rows = 1_000_000;
        budget.bytes = 64 * 1024 * 1024;
        budget.nodes = 100_000;
        budget.edges = 250_000;

        budget.observe(&event).unwrap();
        assert_eq!(budget.rows, 1_000_001);
        assert!(budget.bytes > 64 * 1024 * 1024);
        assert_eq!(budget.nodes, 100_001);
        assert_eq!(budget.edges, 250_000);
    }

    #[test]
    fn a_summary_carries_the_time_it_took_and_the_graph_it_carried() {
        let request_id = Uuid::nil();
        let mut budget = QueryResultBudget::new(QueryLimits::default());
        budget
            .observe(&QueryStreamEvent::Batch {
                request_id,
                sequence: 0,
                row_count: 2,
                columns: vec![BatchColumn {
                    name: "item".to_owned(),
                    value_type: "NODE".to_owned(),
                    values: vec![node("1"), node("2")],
                }],
            })
            .unwrap();

        let mut summary = QueryStreamEvent::Summary {
            request_id,
            bookmark: Bookmark::default(),
            // What every emitter builds: rows and updates, and a structural zero for the rest.
            statistics: QueryStatistics {
                rows: 2,
                ..QueryStatistics::default()
            },
            truncated: false,
            truncation_reason: None,
        };
        budget.observe(&summary).unwrap();
        budget.settle(&mut summary);

        let QueryStreamEvent::Summary { statistics, .. } = &summary else {
            panic!("summary");
        };
        // Not asserted as a bound on how fast the engine is — only that the field is now a
        // measurement rather than a default. A request that reaches here has run.
        assert!(
            statistics.elapsed_us > 0,
            "elapsed_us stayed at its default: {statistics:?}"
        );
        assert!(statistics.elapsed_us >= statistics.elapsed_ms * 1_000);
        assert_eq!(statistics.nodes, 2);
        assert_eq!(statistics.edges, 0);
        assert_eq!(statistics.rows, 2);
    }

    #[test]
    fn a_summary_that_never_met_the_budget_reports_no_measurement() {
        // Nothing settles an event that is not terminal, and a summary the guard never saw keeps
        // its defaults — a client reading `elapsed_us == 0` is reading "not measured".
        let mut batch = QueryStreamEvent::Batch {
            request_id: Uuid::nil(),
            sequence: 0,
            row_count: 0,
            columns: Vec::new(),
        };
        let budget = QueryResultBudget::new(QueryLimits::default());
        budget.settle(&mut batch);
        assert!(matches!(batch, QueryStreamEvent::Batch { .. }));
    }

    #[test]
    fn the_server_ceiling_applies_only_when_it_is_stricter_than_the_caller() {
        let unbounded = QueryLimits::default();
        assert_eq!(unbounded.bytes, UNBOUNDED_QUERY_RESULT_LIMIT);

        // An omitted client limit means unbounded, which is the case the operator ceiling exists
        // to bound: without it a single query could materialize an arbitrarily large result.
        let clamped = QueryLimits::default().clamped_to_server_policy(1_024);
        assert_eq!(clamped.bytes, 1_024);

        // A stricter client limit stays authoritative.
        let strict = QueryLimits {
            bytes: 512,
            ..QueryLimits::default()
        }
        .clamped_to_server_policy(1_024);
        assert_eq!(strict.bytes, 512);

        // A looser client limit is cut down to the operator ceiling.
        let loose = QueryLimits {
            bytes: 8_192,
            ..QueryLimits::default()
        }
        .clamped_to_server_policy(1_024);
        assert_eq!(loose.bytes, 1_024);

        // An operator who disables the ceiling changes nothing, and the other budgets are never
        // touched by this policy.
        let disabled = QueryLimits {
            bytes: 8_192,
            ..QueryLimits::default()
        }
        .clamped_to_server_policy(UNBOUNDED_QUERY_RESULT_LIMIT);
        assert_eq!(disabled.bytes, 8_192);
        assert_eq!(clamped.rows, UNBOUNDED_QUERY_RESULT_LIMIT);
        assert_eq!(clamped.nodes, UNBOUNDED_QUERY_RESULT_LIMIT);
        assert_eq!(clamped.edges, UNBOUNDED_QUERY_RESULT_LIMIT);
    }

    #[test]
    fn result_budget_preserves_explicit_caller_limits() {
        let request_id = Uuid::nil();
        let event = QueryStreamEvent::Batch {
            request_id,
            sequence: 0,
            row_count: 1,
            columns: vec![BatchColumn {
                name: "items".to_owned(),
                value_type: "LIST".to_owned(),
                values: vec![TypedValue::List(vec![node("1"), node("2")])],
            }],
        };
        let mut node_limited = QueryResultBudget::new(QueryLimits {
            rows: 1,
            bytes: 1 << 20,
            nodes: 1,
            edges: 1,
        });
        assert_eq!(
            node_limited.observe(&event).err().map(|error| error.code),
            Some(crate::ErrorCode::ResultBudgetExceeded)
        );

        let mut byte_limited = QueryResultBudget::new(QueryLimits {
            rows: 1,
            bytes: 1,
            nodes: 10,
            edges: 10,
        });
        assert_eq!(
            byte_limited.observe(&event).err().map(|error| error.code),
            Some(crate::ErrorCode::ResultBudgetExceeded)
        );
    }

    #[test]
    fn ingress_admission_holds_request_and_byte_permits_until_worker_completion() {
        let admission = QueryIngressAdmission::new(1, QUERY_ADMISSION_BYTE_QUANTUM).unwrap();
        let request = QueryRequest {
            request_id: Uuid::new_v4(),
            project_id: Some(ProjectId::random()),
            query: "RETURN 1".to_owned(),
            parameters: BTreeMap::new(),
            consistency: CommitAcknowledgement::Published,
            bookmark: None,
            limits: QueryLimits::default(),
            cancellation: CancellationToken::new(),
            deadline: None,
            connection_id: ConnectionId::new(),
        };
        let permit = admission.try_admit(&request).unwrap();
        assert_eq!(
            admission.try_admit(&request).err().map(|error| error.code),
            Some(crate::ErrorCode::WriteAdmissionFull)
        );
        drop(permit);
        assert!(admission.try_admit(&request).is_ok());

        let byte_limited = QueryIngressAdmission::new(2, QUERY_ADMISSION_BYTE_QUANTUM).unwrap();
        let oversized = QueryRequest {
            query: "x".repeat(QUERY_ADMISSION_BYTE_QUANTUM),
            ..request
        };
        assert_eq!(
            byte_limited
                .try_admit(&oversized)
                .err()
                .map(|error| error.code),
            Some(crate::ErrorCode::WriteAdmissionFull)
        );
    }
}
