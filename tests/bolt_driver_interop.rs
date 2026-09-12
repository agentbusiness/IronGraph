// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{
    collections::BTreeMap,
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
};

use irongraph::{
    Bookmark, CommitAcknowledgement, Error, ErrorCode, ProjectId, Result,
    protocol::{
        BatchColumn, BoltSession, QueryColumn, QueryExecutor, QueryRequest, QueryStatistics,
        QueryStreamEvent, QueryTransaction, ResultNode, TypedValue,
    },
};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

struct DriverExecutor {
    project: ProjectId,
    next_bookmark: Arc<AtomicU64>,
}

impl QueryExecutor for DriverExecutor {
    fn resolve_project(&self, selector: &str) -> Result<ProjectId> {
        if selector == self.project.0.to_string() || selector == "driver-project" {
            Ok(self.project)
        } else {
            Err(Error::new(
                ErrorCode::ProjectNotFound,
                "driver test project does not exist",
            ))
        }
    }

    fn execute(
        &self,
        request: QueryRequest,
        emit: &mut dyn FnMut(QueryStreamEvent) -> Result<()>,
    ) -> Result<()> {
        emit_driver_result(
            &request,
            emit,
            self.project,
            self.next_bookmark.fetch_add(1, Ordering::SeqCst),
        )
    }

    fn begin(
        &self,
        project: Option<ProjectId>,
        _bookmark: Option<Bookmark>,
        consistency: CommitAcknowledgement,
    ) -> Result<Box<dyn QueryTransaction>> {
        if project != Some(self.project) || consistency != CommitAcknowledgement::Published {
            return Err(Error::new(
                ErrorCode::ProtocolViolation,
                "driver transaction binding changed",
            ));
        }
        Ok(Box::new(DriverTransaction {
            project: self.project,
            next_bookmark: Arc::clone(&self.next_bookmark),
        }))
    }
}

struct DriverTransaction {
    project: ProjectId,
    next_bookmark: Arc<AtomicU64>,
}

impl QueryTransaction for DriverTransaction {
    fn run(
        &mut self,
        request: QueryRequest,
        emit: &mut dyn FnMut(QueryStreamEvent) -> Result<()>,
    ) -> Result<()> {
        emit_driver_result(
            &request,
            emit,
            self.project,
            self.next_bookmark.fetch_add(1, Ordering::SeqCst),
        )
    }

    fn commit(self: Box<Self>) -> Result<Bookmark> {
        Ok(Bookmark {
            term: 9,
            index: self.next_bookmark.fetch_add(1, Ordering::SeqCst),
        })
    }

    fn rollback(self: Box<Self>) -> Result<()> {
        Ok(())
    }
}

fn emit_driver_result(
    request: &QueryRequest,
    emit: &mut dyn FnMut(QueryStreamEvent) -> Result<()>,
    project: ProjectId,
    bookmark_index: u64,
) -> Result<()> {
    if request.project_id != Some(project)
        || request.parameters.get("x") != Some(&serde_json::json!(42))
        || request
            .parameters
            .get("day")
            .and_then(|value| value.get("$irongraph_type"))
            != Some(&serde_json::json!("date"))
    {
        return Err(Error::new(
            ErrorCode::ProtocolViolation,
            "official driver parameters did not preserve project/scalar/date semantics",
        ));
    }
    emit(QueryStreamEvent::Schema {
        request_id: request.request_id,
        columns: vec![
            QueryColumn {
                name: "value".to_owned(),
                value_type: "INTEGER".to_owned(),
                nullable: false,
            },
            QueryColumn {
                name: "node".to_owned(),
                value_type: "NODE".to_owned(),
                nullable: false,
            },
            QueryColumn {
                name: "when".to_owned(),
                value_type: "DATETIME".to_owned(),
                nullable: false,
            },
        ],
    })?;
    emit(QueryStreamEvent::Batch {
        request_id: request.request_id,
        sequence: 0,
        row_count: 1,
        columns: vec![
            BatchColumn {
                name: "value".to_owned(),
                value_type: "INTEGER".to_owned(),
                values: vec![TypedValue::Integer("42".to_owned())],
            },
            BatchColumn {
                name: "node".to_owned(),
                value_type: "NODE".to_owned(),
                values: vec![TypedValue::Node(ResultNode {
                    id: "11".to_owned(),
                    labels: vec!["Item".to_owned()],
                    properties: BTreeMap::from([(
                        "name".to_owned(),
                        TypedValue::String("driver-node".to_owned()),
                    )]),
                })],
            },
            BatchColumn {
                name: "when".to_owned(),
                value_type: "DATETIME".to_owned(),
                values: vec![TypedValue::DateTime {
                    seconds: 0,
                    nanos: 0,
                    timezone: Some("UTC".to_owned()),
                }],
            },
        ],
    })?;
    emit(QueryStreamEvent::Summary {
        request_id: request.request_id,
        bookmark: Bookmark {
            term: 9,
            index: bookmark_index,
        },
        statistics: QueryStatistics {
            rows: 1,
            ..QueryStatistics::default()
        },
        truncated: false,
        truncation_reason: None,
    })
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "manual gate: requires the official Python neo4j driver"]
async fn official_python_driver_runs_autocommit_and_explicit_transaction() -> Result<()> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let address = listener.local_addr()?;
    let project = ProjectId(Uuid::new_v4());
    let executor: Arc<dyn QueryExecutor> = Arc::new(DriverExecutor {
        project,
        next_bookmark: Arc::new(AtomicU64::new(20)),
    });
    let shutdown = CancellationToken::new();
    let server_shutdown = shutdown.clone();
    let server: tokio::task::JoinHandle<Result<()>> = tokio::spawn(async move {
        loop {
            tokio::select! {
                () = server_shutdown.cancelled() => return Ok(()),
                accepted = listener.accept() => {
                    let (stream, _) = accepted?;
                    let executor = Arc::clone(&executor);
                    tokio::spawn(async move {
                        let _ignored = BoltSession::new(executor).serve(stream).await;
                    });
                }
            }
        }
    });
    let uri = format!("bolt://{address}");
    let project_name = project.0.to_string();
    let driver = tokio::task::spawn_blocking(move || {
        Command::new("python3")
            .arg("-c")
            .arg(
                r#"
import datetime
import neo4j
import os
from neo4j import GraphDatabase

print(f"neo4j-driver={neo4j.__version__}")
driver = GraphDatabase.driver(os.environ["IRONGRAPH_BOLT_URI"], auth=None)
driver.verify_connectivity()
with driver.session(database=os.environ["IRONGRAPH_PROJECT"]) as session:
    record = session.run(
        "RETURN $x AS value",
        x=42,
        day=datetime.date(2024, 1, 2),
    ).single(strict=True)
    assert record["value"] == 42
    assert record["node"].element_id == "11"
    assert set(record["node"].labels) == {"Item"}
    assert record["node"]["name"] == "driver-node"
    assert str(record["when"].tzinfo) == "UTC"
    tx = session.begin_transaction()
    record = tx.run(
        "RETURN $x AS value",
        x=42,
        day=datetime.date(2024, 1, 2),
    ).single(strict=True)
    assert record["value"] == 42
    tx.commit()
driver.close()
"#,
            )
            .env("IRONGRAPH_BOLT_URI", uri)
            .env("IRONGRAPH_PROJECT", project_name)
            .output()
    })
    .await
    .map_err(|error| Error::internal(format!("Python driver task failed: {error}")))??;
    shutdown.cancel();
    server
        .await
        .map_err(|error| Error::internal(format!("Bolt accept loop failed: {error}")))??;
    if !driver.status.success() {
        return Err(Error::new(
            ErrorCode::ProtocolViolation,
            format!(
                "official Python driver failed\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&driver.stdout),
                String::from_utf8_lossy(&driver.stderr),
            ),
        ));
    }
    eprintln!("{}", String::from_utf8_lossy(&driver.stdout));
    Ok(())
}
