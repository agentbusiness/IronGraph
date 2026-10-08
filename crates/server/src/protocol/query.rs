use std::{collections::BTreeMap, time::Instant};

use serde::{Deserialize, Serialize};
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
        if self.query.trim().is_empty() {
            return Err(crate::Error::invalid_data("query is empty"));
        }
        Ok(())
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
    /// [`QueryResultStatistics::settle`], which holds the clock spanning a whole request.
    #[serde(default)]
    pub elapsed_us: u64,
    pub rows: u64,
    /// Node and relationship values carried by the result. A node appearing in three rows counts
    /// three times, so this is
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

/// Per-request result statistics shared by every query transport, without output quotas.
pub(crate) struct QueryResultStatistics {
    rows: u64,
    nodes: u64,
    edges: u64,
    /// When this request started. The accounting spans exactly the work worth timing — it is
    /// constructed before the statement runs and observes the terminal summary — so the clock
    /// lives here rather than being threaded through every transport separately.
    started: Instant,
}

impl QueryResultStatistics {
    #[must_use]
    pub(crate) fn new() -> Self {
        Self {
            rows: 0,
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
        self.rows = rows;
        self.nodes = nodes;
        self.edges = edges;
        Ok(())
    }
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

/// Explicit transaction used by Bolt. Reads use the live canonical store and may observe
/// overlapping writes. Implementations release transaction resources on every terminal path.
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
    fn query_requests_have_no_result_quota_fields() {
        let mut value = serde_json::json!({"request_id":Uuid::nil(), "query":"RETURN 1"});
        let request: QueryRequest = serde_json::from_value(value.clone()).unwrap();
        request.validate().unwrap();
        assert!(
            serde_json::to_value(request)
                .unwrap()
                .get("limits")
                .is_none()
        );
        value["limits"] = serde_json::json!({"bytes":1});
        assert!(serde_json::from_value::<QueryRequest>(value).is_err());
    }

    #[test]
    fn result_statistics_cross_former_server_clamps() {
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
        let mut budget = QueryResultStatistics::new();
        budget.rows = 1_000_000;
        budget.nodes = 100_000;
        budget.edges = 250_000;

        budget.observe(&event).unwrap();
        assert_eq!(budget.rows, 1_000_001);
        assert_eq!(budget.nodes, 100_001);
        assert_eq!(budget.edges, 250_000);
    }

    #[test]
    fn a_summary_carries_the_time_it_took_and_the_graph_it_carried() {
        let request_id = Uuid::nil();
        let mut budget = QueryResultStatistics::new();
        // Give the clock a known interval; a real submicrosecond request may round to zero.
        budget.started = Instant::now() - std::time::Duration::from_millis(5);
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
        let before_settle = budget.started.elapsed().as_micros() as u64;
        budget.settle(&mut summary);
        let after_settle = budget.started.elapsed().as_micros() as u64;

        let QueryStreamEvent::Summary { statistics, .. } = &summary else {
            panic!("summary");
        };
        assert!((before_settle..=after_settle).contains(&statistics.elapsed_us));
        assert!(statistics.elapsed_us >= statistics.elapsed_ms * 1_000);
        assert_eq!(statistics.nodes, 2);
        assert_eq!(statistics.edges, 0);
        assert_eq!(statistics.rows, 2);
    }

    #[test]
    fn a_summary_that_never_met_the_budget_reports_no_measurement() {
        // Nothing settles an event that is not terminal, and a summary the guard never saw keeps
        // its defaults. Zero also represents a measured duration below one microsecond.
        let mut batch = QueryStreamEvent::Batch {
            request_id: Uuid::nil(),
            sequence: 0,
            row_count: 0,
            columns: Vec::new(),
        };
        let budget = QueryResultStatistics::new();
        budget.settle(&mut batch);
        assert!(matches!(batch, QueryStreamEvent::Batch { .. }));
    }

    #[test]
    fn nested_result_statistics_do_not_reject_graph_values() {
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
        let mut statistics = QueryResultStatistics::new();
        statistics.observe(&event).unwrap();
        assert_eq!(statistics.rows, 1);
        assert_eq!(statistics.nodes, 2);
    }
}
