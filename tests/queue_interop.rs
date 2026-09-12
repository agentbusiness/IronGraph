// Test-only target. See the note in any `tests/*.rs` file.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

//! Interop gate for the Queue surface, driven by a real AMQP 0-9-1 client.
//!
//! Every assertion here uses stock client configuration. That is the point: the repository's
//! previous AMQP gate configured the client *around* the gaps — `ExternalCredentials`, a disabled
//! heartbeat — so it passed while a default client could not connect at all. Anything this suite
//! has to special-case is a real interoperability defect, not a test detail.
//!
//! `lapin` is a pure-Rust client with no C or system dependency, so this runs anywhere the rest of
//! the suite does.

use std::sync::Arc;

use crate::broker_harness::{LocalCoordinator, apply, reserve_loopback_port, wait_until_listening};
use futures::StreamExt;
use irongraph::{
    ProjectId, Result,
    broker::{AmqpExchangeKind, BrokerCommand, BrokerCoordinator, QueueKind, QueueServer},
};
use lapin::{
    BasicProperties, Connection, ConnectionProperties,
    options::{
        BasicAckOptions, BasicCancelOptions, BasicConsumeOptions, BasicGetOptions,
        BasicPublishOptions, ExchangeDeleteOptions, QueueDeclareOptions, QueueDeleteOptions,
        QueuePurgeOptions,
    },
    types::FieldTable,
};

/// Starts the shipping Queue listener and returns its URI plus a shutdown handle.
async fn start_queue(
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
    let server = QueueServer::new(
        address,
        project,
        Arc::clone(&coordinator) as Arc<dyn BrokerCoordinator>,
    );
    let running = shutdown.clone();
    tokio::spawn(async move {
        let _ignored = server.run(running).await;
    });
    wait_until_listening(address).await?;
    Ok(Some((
        format!("amqp://guest:guest@{address}/%2f"),
        coordinator,
        shutdown,
    )))
}

fn declare_topology(coordinator: &Arc<LocalCoordinator>, project: ProjectId) -> Result<()> {
    apply(
        coordinator,
        BrokerCommand::CreateExchange {
            project,
            name: "orders".to_owned(),
            kind: AmqpExchangeKind::Direct,
            durable: true,
            passive: false,
        },
    )?;
    apply(
        coordinator,
        BrokerCommand::CreateQueue {
            project,
            name: "orders.eu".to_owned(),
            kind: QueueKind::Classic,
            durable: true,
            passive: false,
            dead_letter_exchange: None,
            dead_letter_routing_key: None,
            retention: irongraph::broker::RetentionPolicy {
                max_age_ms: None,
                max_bytes: None,
            },
            exclusive_owner: None,
            auto_delete: false,
        },
    )?;
    apply(
        coordinator,
        BrokerCommand::BindQueue {
            project,
            exchange: "orders".to_owned(),
            queue: "orders.eu".to_owned(),
            routing_key: "eu".to_owned(),
        },
    )?;
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stock_client_connects_publishes_and_consumes() -> Result<()> {
    // The whole gate. `ConnectionProperties::default()` with a `guest:guest` URI is the first line
    // of every AMQP tutorial, and it is what the shipped listener has to accept.
    let project = ProjectId::random();
    let Some((uri, coordinator, shutdown)) = start_queue(project).await? else {
        return Ok(());
    };
    declare_topology(&coordinator, project)?;

    let connection = Connection::connect(&uri, ConnectionProperties::default())
        .await
        .map_err(|error| {
            irongraph::Error::internal(format!("stock AMQP client could not connect: {error}"))
        })?;
    let channel = connection
        .create_channel()
        .await
        .map_err(|error| irongraph::Error::internal(error.to_string()))?;

    channel
        .basic_publish(
            "orders",
            "eu",
            BasicPublishOptions::default(),
            b"first order",
            BasicProperties::default(),
        )
        .await
        .map_err(|error| irongraph::Error::internal(format!("publish failed: {error}")))?;

    let mut consumer = channel
        .basic_consume(
            "orders.eu",
            "interop",
            BasicConsumeOptions::default(),
            FieldTable::default(),
        )
        .await
        .map_err(|error| irongraph::Error::internal(format!("consume failed: {error}")))?;

    let delivery = tokio::time::timeout(std::time::Duration::from_secs(10), consumer.next())
        .await
        .map_err(|_| irongraph::Error::internal("no delivery arrived within ten seconds"))?
        .ok_or_else(|| irongraph::Error::internal("consumer stream ended"))?
        .map_err(|error| irongraph::Error::internal(format!("delivery failed: {error}")))?;
    assert_eq!(delivery.data, b"first order");

    delivery
        .ack(BasicAckOptions::default())
        .await
        .map_err(|error| irongraph::Error::internal(format!("ack failed: {error}")))?;

    shutdown.cancel();
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_stock_client_can_purge_unbind_and_delete_topology() -> Result<()> {
    // `queue.purge`, `queue.delete`, `queue.unbind` and `exchange.delete` used to answer
    // NOT_IMPLEMENTED, which a stock client reports as a channel-level error. Every management
    // path in every mainstream client library goes through these four methods, so a broker without
    // them can be published to but never cleaned up.
    let project = ProjectId::random();
    let Some((uri, coordinator, shutdown)) = start_queue(project).await? else {
        return Ok(());
    };
    declare_topology(&coordinator, project)?;

    let connection = Connection::connect(&uri, ConnectionProperties::default())
        .await
        .map_err(|error| irongraph::Error::internal(error.to_string()))?;
    let channel = connection
        .create_channel()
        .await
        .map_err(|error| irongraph::Error::internal(error.to_string()))?;

    for index in 0..3_u8 {
        channel
            .basic_publish(
                "orders",
                "eu",
                BasicPublishOptions::default(),
                &[b'o', index],
                BasicProperties::default(),
            )
            .await
            .map_err(|error| irongraph::Error::internal(format!("publish failed: {error}")))?;
    }

    // queue.purge reports how many ready messages it discarded.
    let purged = channel
        .queue_purge("orders.eu", QueuePurgeOptions::default())
        .await
        .map_err(|error| irongraph::Error::internal(format!("queue.purge failed: {error}")))?;
    assert_eq!(
        purged, 3,
        "queue.purge must report the ready messages it discarded"
    );

    // A purged queue delivers nothing, which is the assertion that the count above was not merely
    // a plausible number written into the reply frame.
    let drained = channel
        .basic_get("orders.eu", BasicGetOptions::default())
        .await
        .map_err(|error| irongraph::Error::internal(format!("basic.get failed: {error}")))?;
    assert!(
        drained.is_none(),
        "a purged queue must have no ready messages"
    );

    // A second purge finds nothing left.
    let again = channel
        .queue_purge("orders.eu", QueuePurgeOptions::default())
        .await
        .map_err(|error| irongraph::Error::internal(format!("second purge failed: {error}")))?;
    assert_eq!(again, 0);

    channel
        .queue_unbind("orders.eu", "orders", "eu", FieldTable::default())
        .await
        .map_err(|error| irongraph::Error::internal(format!("queue.unbind failed: {error}")))?;

    // Unbinding is what makes exchange.delete with if-unused succeed, so the two assertions are
    // coupled: if the unbind silently did nothing, this call fails PRECONDITION_FAILED.
    channel
        .exchange_delete(
            "orders",
            ExchangeDeleteOptions {
                if_unused: true,
                nowait: false,
            },
        )
        .await
        .map_err(|error| irongraph::Error::internal(format!("exchange.delete failed: {error}")))?;

    let deleted = channel
        .queue_delete("orders.eu", QueueDeleteOptions::default())
        .await
        .map_err(|error| irongraph::Error::internal(format!("queue.delete failed: {error}")))?;
    assert_eq!(deleted, 0, "the queue was already purged");

    shutdown.cancel();
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn queue_delete_if_empty_refuses_a_queue_with_ready_messages() -> Result<()> {
    // The precondition has to be enforced on canonical state rather than trusted from the client,
    // and a refused delete must leave the queue and its messages intact.
    let project = ProjectId::random();
    let Some((uri, coordinator, shutdown)) = start_queue(project).await? else {
        return Ok(());
    };
    declare_topology(&coordinator, project)?;

    let connection = Connection::connect(&uri, ConnectionProperties::default())
        .await
        .map_err(|error| irongraph::Error::internal(error.to_string()))?;
    let channel = connection
        .create_channel()
        .await
        .map_err(|error| irongraph::Error::internal(error.to_string()))?;
    channel
        .basic_publish(
            "orders",
            "eu",
            BasicPublishOptions::default(),
            b"keep me",
            BasicProperties::default(),
        )
        .await
        .map_err(|error| irongraph::Error::internal(error.to_string()))?;

    let refused = channel
        .queue_delete(
            "orders.eu",
            QueueDeleteOptions {
                if_unused: false,
                if_empty: true,
                nowait: false,
            },
        )
        .await;
    assert!(
        refused.is_err(),
        "queue.delete if-empty must refuse a queue that still holds a ready message"
    );

    // AMQP closes the channel on a channel-level exception, so the survival check needs a new one.
    let channel = connection
        .create_channel()
        .await
        .map_err(|error| irongraph::Error::internal(error.to_string()))?;
    let survivor = channel
        .basic_get("orders.eu", BasicGetOptions::default())
        .await
        .map_err(|error| irongraph::Error::internal(format!("basic.get failed: {error}")))?
        .ok_or_else(|| irongraph::Error::internal("the refused delete lost the message"))?;
    assert_eq!(survivor.data, b"keep me");

    shutdown.cancel();
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_exclusive_server_named_queue_is_private_and_self_cleaning() -> Result<()> {
    // `queue_declare(queue='', exclusive=True)` is the first line of the RPC pattern in every AMQP
    // tutorial, and it used to be answered NOT_IMPLEMENTED. Three properties make it usable: the
    // broker names it, a second connection cannot touch it, and it disappears on disconnect.
    let project = ProjectId::random();
    let Some((uri, _coordinator, shutdown)) = start_queue(project).await? else {
        return Ok(());
    };

    let owner = Connection::connect(&uri, ConnectionProperties::default())
        .await
        .map_err(|error| irongraph::Error::internal(error.to_string()))?;
    let channel = owner
        .create_channel()
        .await
        .map_err(|error| irongraph::Error::internal(error.to_string()))?;
    let declared = channel
        .queue_declare(
            "",
            QueueDeclareOptions {
                exclusive: true,
                auto_delete: true,
                ..QueueDeclareOptions::default()
            },
            FieldTable::default(),
        )
        .await
        .map_err(|error| {
            irongraph::Error::internal(format!("exclusive declare failed: {error}"))
        })?;
    let name = declared.name().as_str().to_owned();
    assert!(
        name.starts_with("amq.gen-"),
        "the broker must generate a name for an empty queue.declare, got {name:?}"
    );

    // A different connection must be refused, which is the whole point of `exclusive`.
    let intruder = Connection::connect(&uri, ConnectionProperties::default())
        .await
        .map_err(|error| irongraph::Error::internal(error.to_string()))?;
    let intruder_channel = intruder
        .create_channel()
        .await
        .map_err(|error| irongraph::Error::internal(error.to_string()))?;
    let refused = intruder_channel
        .basic_consume(
            &name,
            "intruder",
            BasicConsumeOptions::default(),
            FieldTable::default(),
        )
        .await;
    assert!(
        refused.is_err(),
        "another connection must not be able to consume from an exclusive queue"
    );

    // Closing the owning connection collects the queue, so a passive redeclare no longer finds it.
    owner
        .close(200, "done")
        .await
        .map_err(|error| irongraph::Error::internal(error.to_string()))?;

    let checker = Connection::connect(&uri, ConnectionProperties::default())
        .await
        .map_err(|error| irongraph::Error::internal(error.to_string()))?;
    let checker_channel = checker
        .create_channel()
        .await
        .map_err(|error| irongraph::Error::internal(error.to_string()))?;
    let mut gone = false;
    for _ in 0..50 {
        let probe = checker_channel
            .queue_declare(
                &name,
                QueueDeclareOptions {
                    passive: true,
                    ..QueueDeclareOptions::default()
                },
                FieldTable::default(),
            )
            .await;
        if probe.is_err() {
            gone = true;
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
    assert!(
        gone,
        "an exclusive queue must be deleted when its owning connection closes"
    );

    shutdown.cancel();
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn an_auto_delete_queue_survives_until_its_last_consumer_cancels() -> Result<()> {
    // auto-delete is defined on the LAST consumer leaving, not the first. A broker that collects
    // the queue when any consumer cancels breaks every multi-worker deployment, so this asserts
    // the queue is still there after one of two consumers goes away.
    let project = ProjectId::random();
    let Some((uri, _coordinator, shutdown)) = start_queue(project).await? else {
        return Ok(());
    };

    let connection = Connection::connect(&uri, ConnectionProperties::default())
        .await
        .map_err(|error| irongraph::Error::internal(error.to_string()))?;
    let channel = connection
        .create_channel()
        .await
        .map_err(|error| irongraph::Error::internal(error.to_string()))?;
    channel
        .queue_declare(
            "work",
            QueueDeclareOptions {
                auto_delete: true,
                ..QueueDeclareOptions::default()
            },
            FieldTable::default(),
        )
        .await
        .map_err(|error| irongraph::Error::internal(error.to_string()))?;

    // Dropping a `lapin::Consumer` sends basic.cancel, so both handles have to stay alive for the
    // queue to still have two consumers by the time the explicit cancel below runs.
    let mut consumers = Vec::new();
    for tag in ["worker-a", "worker-b"] {
        consumers.push(
            channel
                .basic_consume(
                    "work",
                    tag,
                    BasicConsumeOptions::default(),
                    FieldTable::default(),
                )
                .await
                .map_err(|error| irongraph::Error::internal(format!("consume failed: {error}")))?,
        );
    }

    channel
        .basic_cancel("worker-a", BasicCancelOptions::default())
        .await
        .map_err(|error| irongraph::Error::internal(format!("cancel failed: {error}")))?;
    channel
        .queue_declare(
            "work",
            QueueDeclareOptions {
                passive: true,
                ..QueueDeclareOptions::default()
            },
            FieldTable::default(),
        )
        .await
        .map_err(|error| {
            irongraph::Error::internal(format!(
                "an auto-delete queue was collected while a consumer was still attached: {error}"
            ))
        })?;

    channel
        .basic_cancel("worker-b", BasicCancelOptions::default())
        .await
        .map_err(|error| irongraph::Error::internal(format!("cancel failed: {error}")))?;
    drop(consumers);

    // AMQP closes the channel on the NOT_FOUND that a passive declare of a missing queue raises,
    // so this needs its own channel to avoid poisoning anything that follows.
    let probe_channel = connection
        .create_channel()
        .await
        .map_err(|error| irongraph::Error::internal(error.to_string()))?;
    let probe = probe_channel
        .queue_declare(
            "work",
            QueueDeclareOptions {
                passive: true,
                ..QueueDeclareOptions::default()
            },
            FieldTable::default(),
        )
        .await;
    assert!(
        probe.is_err(),
        "an auto-delete queue must be collected once its last consumer cancels"
    );

    shutdown.cancel();
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn a_malformed_plain_response_is_refused() -> Result<()> {
    // Credentials carry no authority on a loopback transport, but a blob that is not a PLAIN
    // response at all indicates a confused client rather than a valid anonymous login, so the
    // shape is still checked. `lapin` cannot send a malformed response, so this drives the
    // handshake directly.
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    let project = ProjectId::random();
    let Some((uri, _coordinator, shutdown)) = start_queue(project).await? else {
        return Ok(());
    };
    let address = uri
        .rsplit('@')
        .next()
        .and_then(|rest| rest.split('/').next())
        .ok_or_else(|| irongraph::Error::internal("queue URI has no authority"))?
        .to_owned();

    let mut stream = tokio::net::TcpStream::connect(&address).await?;
    stream.write_all(b"AMQP\x00\x00\x09\x01").await?;
    let mut header = [0_u8; 7];
    stream.read_exact(&mut header).await?;

    // connection.start-ok with mechanism PLAIN and a response missing its NUL separators.
    let mut payload = Vec::new();
    payload.extend_from_slice(&10_u16.to_be_bytes());
    payload.extend_from_slice(&11_u16.to_be_bytes());
    payload.extend_from_slice(&0_u32.to_be_bytes()); // empty client properties
    payload.push(5);
    payload.extend_from_slice(b"PLAIN");
    let response = b"not-a-sasl-blob";
    payload.extend_from_slice(&u32::try_from(response.len()).unwrap_or(0).to_be_bytes());
    payload.extend_from_slice(response);
    payload.push(5);
    payload.extend_from_slice(b"en_US");

    let mut frame = vec![1_u8];
    frame.extend_from_slice(&0_u16.to_be_bytes());
    frame.extend_from_slice(&u32::try_from(payload.len()).unwrap_or(0).to_be_bytes());
    frame.extend_from_slice(&payload);
    frame.push(0xCE);
    stream.write_all(&frame).await?;

    // The server answers with connection.close rather than proceeding to tune.
    let mut kind = [0_u8; 1];
    let refused = tokio::time::timeout(
        std::time::Duration::from_secs(5),
        stream.read_exact(&mut kind),
    )
    .await;
    assert!(
        refused.is_ok(),
        "a malformed PLAIN response must be answered, not ignored"
    );

    shutdown.cancel();
    Ok(())
}
