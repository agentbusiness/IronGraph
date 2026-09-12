// Test-only target. See the note in any `tests/*.rs` file.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Interop gate for the Stream surface, driven by real librdkafka.
//!
//! `rdkafka` vendors and builds the actual C library that backs the Python, Go, Node, and .NET
//! Kafka clients, so this measures what those clients do rather than what the wire code appears to
//! allow. Nothing here pins an API version or a message format: the client negotiates through
//! `ApiVersions` exactly as it would against a real broker, which is the only way to find out what
//! it actually selects.
//!
//! The previous Kafka gate shelled out to `kafka-python`, which tolerates far more than librdkafka
//! and was configured down to a legacy protocol, so it could pass while saying nothing about
//! whether a mainstream client works.

use std::{sync::Arc, time::Duration};

use crate::broker_harness::{LocalCoordinator, apply, reserve_loopback_port, wait_until_listening};
use irongraph::{
    ProjectId, Result,
    broker::{BrokerCommand, BrokerCoordinator, RetentionPolicy, StreamServer},
};
use rdkafka::{
    ClientConfig, Message,
    consumer::{Consumer, StreamConsumer},
    message::{Header, Headers, OwnedHeaders},
    producer::{FutureProducer, FutureRecord},
};

/// Starts the shipping Stream listener and returns its bootstrap address.
async fn start_stream(
    project: ProjectId,
) -> Result<
    Option<(
        String,
        Arc<LocalCoordinator>,
        tokio_util::sync::CancellationToken,
    )>,
> {
    let coordinator = Arc::new(LocalCoordinator::new()?);
    let address = match reserve_loopback_port().await {
        Ok(address) => address,
        Err(error) if error.message.contains("Operation not permitted") => return Ok(None),
        Err(error) => return Err(error),
    };
    let shutdown = tokio_util::sync::CancellationToken::new();
    let server = StreamServer::new(
        address,
        project,
        Arc::clone(&coordinator) as Arc<dyn BrokerCoordinator>,
    );
    let running = shutdown.clone();
    tokio::spawn(async move {
        let _ignored = server.run(running).await;
    });
    wait_until_listening(address).await?;
    Ok(Some((address.to_string(), coordinator, shutdown)))
}

fn declare_topic(
    coordinator: &Arc<LocalCoordinator>,
    project: ProjectId,
    name: &str,
    partitions: u16,
) -> Result<()> {
    apply(
        coordinator,
        BrokerCommand::CreateTopic {
            project,
            name: name.to_owned(),
            partitions,
            retention: RetentionPolicy {
                max_age_ms: None,
                max_bytes: None,
            },
        },
    )?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stock_librdkafka_producer_and_consumer_round_trip() -> Result<()> {
    tokio::time::timeout(Duration::from_mins(2), round_trip())
        .await
        .map_err(|_| irongraph::Error::internal("librdkafka round trip did not finish"))?
}

async fn round_trip() -> Result<()> {
    // Only `bootstrap.servers` and `group.id` are set. Everything else — protocol version, message
    // format, fetch and commit behaviour — is whatever librdkafka negotiates on its own.
    let project = ProjectId::random();
    let Some((bootstrap, coordinator, shutdown)) = start_stream(project).await? else {
        return Ok(());
    };
    declare_topic(&coordinator, project, "events", 1)?;

    let producer: FutureProducer = ClientConfig::new()
        .set("bootstrap.servers", &bootstrap)
        // Fail fast rather than retrying for the default five minutes: this is a gate, and a
        // delivery that does not complete promptly is the defect being measured.
        .set("message.timeout.ms", "8000")
        .set("socket.timeout.ms", "4000")
        .create()
        .map_err(|error| {
            irongraph::Error::internal(format!("librdkafka producer setup failed: {error}"))
        })?;

    for index in 0..8_i32 {
        let payload = format!("event-{index}");
        let key = format!("key-{index}");
        producer
            .send(
                FutureRecord::to("events").payload(&payload).key(&key),
                Duration::from_secs(10),
            )
            .await
            .map_err(|(error, _)| {
                irongraph::Error::internal(format!("librdkafka produce failed: {error}"))
            })?;
    }

    let consumer: StreamConsumer = ClientConfig::new()
        .set("bootstrap.servers", &bootstrap)
        .set("group.id", "interop")
        .set("auto.offset.reset", "earliest")
        .set("socket.timeout.ms", "4000")
        .set("session.timeout.ms", "6000")
        .create()
        .map_err(|error| {
            irongraph::Error::internal(format!("librdkafka consumer setup failed: {error}"))
        })?;
    consumer.subscribe(&["events"]).map_err(|error| {
        irongraph::Error::internal(format!("librdkafka subscribe failed: {error}"))
    })?;

    let mut received = Vec::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    while received.len() < 8 && tokio::time::Instant::now() < deadline {
        match tokio::time::timeout(Duration::from_secs(5), consumer.recv()).await {
            Ok(Ok(message)) => {
                let payload = message
                    .payload()
                    .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
                    .unwrap_or_default();
                received.push(payload);
            }
            Ok(Err(error)) => {
                return Err(irongraph::Error::internal(format!(
                    "librdkafka consume failed: {error}"
                )));
            }
            Err(_) => break,
        }
    }

    close_consumer(consumer).await;
    shutdown.cancel();

    assert_eq!(
        received,
        (0..8).map(|i| format!("event-{i}")).collect::<Vec<_>>(),
        "a stock librdkafka consumer must read back what a stock producer wrote, in order"
    );
    Ok(())
}

/// Closes a group consumer while the listener is still accepting.
///
/// librdkafka's consumer destructor runs a synchronous group close, and it is not bounded by any
/// of the timeouts above. Dropping the consumer after the listener has stopped — including on the
/// unwind from a failed assertion — leaves it retrying a refused connection until the test process
/// is killed, which turns every failure in this file into a hang instead of a report. Closing on a
/// blocking thread under a deadline keeps that from happening even if the close itself sticks.
async fn close_consumer(consumer: StreamConsumer) {
    let closed = tokio::task::spawn_blocking(move || drop(consumer));
    let _ignored = tokio::time::timeout(Duration::from_secs(15), closed).await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn librdkafka_uses_v2_record_batches_and_preserves_headers() -> Result<()> {
    tokio::time::timeout(Duration::from_mins(2), record_batch_round_trip())
        .await
        .map_err(|_| {
            irongraph::Error::internal("librdkafka record batch round trip did not finish")
        })?
}

/// Header names and values of a consumed message, ordered by name.
///
/// The broker stores headers by name, so insertion order is not preserved and the comparison has
/// to be order independent.
fn sorted_headers<M: Message>(message: &M) -> Vec<(String, String)> {
    let mut headers = message
        .headers()
        .map(|headers| {
            headers
                .iter()
                .map(|header| {
                    (
                        header.key.to_owned(),
                        header
                            .value
                            .map(|value| String::from_utf8_lossy(value).into_owned())
                            .unwrap_or_default(),
                    )
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    headers.sort();
    headers
}

/// Forces the modern wire path and checks that headers survive it.
///
/// librdkafka picks the v2 record batch format purely from what `ApiVersions` advertises for
/// Produce, so nothing here pins a format: `api.version.request=true` makes it ask, and record
/// headers are only representable in v2, so a broker that fell back to the legacy message set
/// would either drop them or refuse the produce outright.
async fn record_batch_round_trip() -> Result<()> {
    let project = ProjectId::random();
    let Some((bootstrap, coordinator, shutdown)) = start_stream(project).await? else {
        return Ok(());
    };
    declare_topic(&coordinator, project, "headed", 1)?;

    let producer: FutureProducer = ClientConfig::new()
        .set("bootstrap.servers", &bootstrap)
        .set("api.version.request", "true")
        // The delivery future is not bounded by the queue timeout passed to `send`, so a failure
        // has to be bounded here or it stalls for the default five minutes per message.
        .set("message.timeout.ms", "8000")
        .set("socket.timeout.ms", "4000")
        .create()
        .map_err(|error| {
            irongraph::Error::internal(format!("librdkafka producer setup failed: {error}"))
        })?;

    for index in 0..4_i32 {
        let payload = format!("headed-{index}");
        let key = format!("key-{index}");
        let sequence = index.to_string();
        let headers = OwnedHeaders::new()
            .insert(Header {
                key: "trace",
                value: Some("abc123"),
            })
            .insert(Header {
                key: "sequence",
                value: Some(sequence.as_str()),
            });
        producer
            .send(
                FutureRecord::to("headed")
                    .payload(&payload)
                    .key(&key)
                    .headers(headers),
                Duration::from_secs(10),
            )
            .await
            .map_err(|(error, _)| {
                irongraph::Error::internal(format!("librdkafka produce failed: {error}"))
            })?;
    }

    let consumer: StreamConsumer = ClientConfig::new()
        .set("bootstrap.servers", &bootstrap)
        .set("group.id", "interop-headers")
        .set("auto.offset.reset", "earliest")
        .set("api.version.request", "true")
        .set("socket.timeout.ms", "4000")
        .set("session.timeout.ms", "6000")
        .create()
        .map_err(|error| {
            irongraph::Error::internal(format!("librdkafka consumer setup failed: {error}"))
        })?;
    consumer.subscribe(&["headed"]).map_err(|error| {
        irongraph::Error::internal(format!("librdkafka subscribe failed: {error}"))
    })?;

    let mut received = Vec::new();
    let deadline = tokio::time::Instant::now() + Duration::from_secs(30);
    while received.len() < 4 && tokio::time::Instant::now() < deadline {
        match tokio::time::timeout(Duration::from_secs(5), consumer.recv()).await {
            Ok(Ok(message)) => {
                let payload = message
                    .payload()
                    .map(|bytes| String::from_utf8_lossy(bytes).into_owned())
                    .unwrap_or_default();
                received.push((payload, sorted_headers(&message)));
            }
            Ok(Err(error)) => {
                return Err(irongraph::Error::internal(format!(
                    "librdkafka consume failed: {error}"
                )));
            }
            Err(_) => break,
        }
    }

    close_consumer(consumer).await;
    shutdown.cancel();

    assert_eq!(
        received,
        (0..4)
            .map(|index| (
                format!("headed-{index}"),
                vec![
                    ("sequence".to_owned(), index.to_string()),
                    ("trace".to_owned(), "abc123".to_owned()),
                ]
            ))
            .collect::<Vec<_>>(),
        "a stock librdkafka consumer must read back both payloads and headers, in order"
    );
    Ok(())
}
