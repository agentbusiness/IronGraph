use std::{
    net::{SocketAddr, TcpListener, TcpStream},
    sync::Arc,
    thread::JoinHandle,
    time::{Duration, Instant},
};

use axum::{Json, Router, http::header, response::IntoResponse, routing::post};
use irongraph_client::{Query, RemoteClient};
use irongraph_server::protocol::{
    BatchColumn, BoltServer, QueryColumn, QueryExecutor, QueryRequest, QueryStatistics,
    QueryStreamEvent, QueryTransaction, TypedValue,
};
use irongraph_types::{Bookmark, CommitAcknowledgement, ProjectId, Result};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

struct StaticExecutor;
struct StaticTransaction;

impl QueryTransaction for StaticTransaction {
    fn run(
        &mut self,
        request: QueryRequest,
        emit: &mut dyn FnMut(QueryStreamEvent) -> Result<()>,
    ) -> Result<()> {
        for event in result_events(request.request_id) {
            emit(event)?;
        }
        Ok(())
    }

    fn commit(self: Box<Self>) -> Result<Bookmark> {
        Ok(Bookmark { term: 1, index: 1 })
    }

    fn rollback(self: Box<Self>) -> Result<()> {
        Ok(())
    }
}

impl QueryExecutor for StaticExecutor {
    fn execute(
        &self,
        request: QueryRequest,
        emit: &mut dyn FnMut(QueryStreamEvent) -> Result<()>,
    ) -> Result<()> {
        for event in result_events(request.request_id) {
            emit(event)?;
        }
        Ok(())
    }

    fn begin(
        &self,
        _project: Option<ProjectId>,
        _bookmark: Option<Bookmark>,
        _consistency: CommitAcknowledgement,
    ) -> Result<Box<dyn QueryTransaction>> {
        Ok(Box::new(StaticTransaction))
    }
}

fn result_events(request_id: Uuid) -> Vec<QueryStreamEvent> {
    vec![
        QueryStreamEvent::Schema {
            request_id,
            columns: vec![QueryColumn {
                name: "answer".to_owned(),
                value_type: "INTEGER".to_owned(),
                nullable: false,
            }],
        },
        QueryStreamEvent::Batch {
            request_id,
            sequence: 0,
            row_count: 1,
            columns: vec![BatchColumn {
                name: "answer".to_owned(),
                value_type: "INTEGER".to_owned(),
                values: vec![TypedValue::Integer("42".to_owned())],
            }],
        },
        QueryStreamEvent::Summary {
            request_id,
            bookmark: Bookmark { term: 1, index: 1 },
            statistics: QueryStatistics {
                rows: 1,
                ..QueryStatistics::default()
            },
            truncated: false,
            truncation_reason: None,
        },
    ]
}

async fn api_query(Json(request): Json<QueryRequest>) -> impl IntoResponse {
    let body = result_events(request.request_id)
        .into_iter()
        .map(|event| serde_json::to_string(&event).expect("serialize query event"))
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    ([(header::CONTENT_TYPE, "application/x-ndjson")], body)
}

fn spawn_api() -> (SocketAddr, CancellationToken, JoinHandle<()>) {
    let (address_sender, address_receiver) = std::sync::mpsc::sync_channel(1);
    let shutdown = CancellationToken::new();
    let serving_shutdown = shutdown.clone();
    let thread = std::thread::spawn(move || {
        tokio::runtime::Runtime::new()
            .expect("API runtime")
            .block_on(async move {
                let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
                    .await
                    .expect("bind API listener");
                address_sender
                    .send(listener.local_addr().expect("API local address"))
                    .expect("publish API address");
                axum::serve(listener, Router::new().route("/api/query", post(api_query)))
                    .with_graceful_shutdown(serving_shutdown.cancelled_owned())
                    .await
                    .expect("serve API");
            });
    });
    (
        address_receiver.recv().expect("receive API address"),
        shutdown,
        thread,
    )
}

fn spawn_bolt() -> (SocketAddr, CancellationToken, JoinHandle<()>) {
    let listener = TcpListener::bind("127.0.0.1:0").expect("reserve Bolt address");
    let address = listener.local_addr().expect("Bolt local address");
    drop(listener);
    let shutdown = CancellationToken::new();
    let serving_shutdown = shutdown.clone();
    let thread = std::thread::spawn(move || {
        tokio::runtime::Runtime::new()
            .expect("Bolt runtime")
            .block_on(async move {
                BoltServer::new(address, Arc::new(StaticExecutor))
                    .run(serving_shutdown)
                    .await
                    .expect("serve Bolt");
            });
    });
    let deadline = Instant::now() + Duration::from_secs(5);
    while TcpStream::connect(address).is_err() {
        assert!(Instant::now() < deadline, "Bolt listener did not start");
        std::thread::sleep(Duration::from_millis(10));
    }
    (address, shutdown, thread)
}

#[test]
fn api_and_bolt_return_the_same_transport_neutral_rows() {
    let (api_address, api_shutdown, api_thread) = spawn_api();
    let (bolt_address, bolt_shutdown, bolt_thread) = spawn_bolt();
    let project = ProjectId(Uuid::new_v4());

    let api = RemoteClient::api(&format!("http://{api_address}"));
    let bolt = RemoteClient::bolt(&format!("bolt://{bolt_address}"));
    let api_result = api
        .expect("create API client")
        .query(Query::new("RETURN 42 AS answer").with_project(project))
        .expect("API query");
    let bolt_result = bolt
        .expect("create Bolt client")
        .query(Query::new("RETURN 42 AS answer").with_project(project))
        .expect("Bolt query");

    assert_eq!(api_result.columns[0].name, "answer");
    assert_eq!(bolt_result.columns[0].name, "answer");
    assert!(matches!(api_result.rows[0][0], TypedValue::Integer(ref value) if value == "42"));
    assert!(matches!(bolt_result.rows[0][0], TypedValue::Integer(ref value) if value == "42"));
    assert_eq!(
        bolt_result.summary.bookmark,
        Some(Bookmark { term: 1, index: 1 })
    );

    api_shutdown.cancel();
    bolt_shutdown.cancel();
    api_thread.join().expect("join API server");
    bolt_thread.join().expect("join Bolt server");
}

#[test]
fn plain_remote_connections_are_rejected() {
    assert!(RemoteClient::api("http://example.com").is_err());
    assert!(RemoteClient::bolt("bolt://192.0.2.1:18485").is_err());
}
