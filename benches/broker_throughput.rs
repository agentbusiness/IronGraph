#![allow(unsafe_code)]
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]
#![allow(
    clippy::cast_lossless,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::similar_names,
    clippy::too_many_lines
)]

//! Isolated real-client throughput cell for the Kafka-compatible and AMQP-compatible surfaces.
//!
//! One process measures one cell. This keeps `getrusage` CPU and peak-RSS evidence attributable to
//! exactly one protocol/direction/payload combination; the matrix runner launches and aggregates
//! these cells. Setup and correctness verification sit outside the timed interval.

#[path = "../tests/broker_harness/mod.rs"]
mod broker_harness;

use std::{
    collections::BTreeMap,
    env,
    sync::Arc,
    time::{Duration, Instant},
};

use broker_harness::{LocalCoordinator, apply, reserve_loopback_port, wait_until_listening};
use futures::{StreamExt, TryStreamExt, stream};
use irongraph::{
    ProjectId, Result,
    broker::{
        AmqpExchangeKind, BrokerCommand, BrokerCoordinator, KafkaBatchRecord, QueueKind,
        QueueServer, RetentionPolicy, StreamServer,
    },
};
use lapin::{
    BasicProperties, Connection, ConnectionProperties, ExchangeKind,
    options::{
        BasicConsumeOptions, BasicPublishOptions, ConfirmSelectOptions, ExchangeDeclareOptions,
        QueueBindOptions, QueueDeclareOptions,
    },
    types::FieldTable,
};
use rdkafka::{
    ClientConfig,
    admin::{AdminClient, AdminOptions, NewTopic, TopicReplication},
    client::DefaultClientContext,
    consumer::{Consumer, StreamConsumer},
    producer::{FutureProducer, FutureRecord},
    topic_partition_list::{Offset, TopicPartitionList},
};
use serde::Serialize;
use tokio_util::sync::CancellationToken;

const TOPIC: &str = "throughput";
const EXCHANGE: &str = "throughput";
const QUEUE: &str = "throughput.queue";
const ROUTING_KEY: &str = "throughput";

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum Protocol {
    Kafka,
    Amqp,
}

#[derive(Clone, Copy, Debug, Serialize)]
#[serde(rename_all = "snake_case")]
enum Direction {
    Ingress,
    Egress,
    FullDuplex,
}

#[derive(Debug)]
struct Config {
    protocol: Protocol,
    direction: Direction,
    payload_bytes: usize,
    messages: usize,
    pipeline_depth: usize,
    sample: usize,
    external_endpoint: Option<String>,
    existing_kafka_topic: Option<String>,
}

#[derive(Clone, Copy)]
struct Usage {
    user_seconds: f64,
    system_seconds: f64,
    peak_rss_bytes: u64,
}

#[derive(Serialize)]
struct CellResult {
    schema_version: u32,
    protocol: Protocol,
    direction: Direction,
    payload_bytes: usize,
    messages: usize,
    pipeline_depth: usize,
    concurrency: usize,
    sample: usize,
    wall_seconds: f64,
    cpu_user_seconds: f64,
    cpu_system_seconds: f64,
    peak_rss_bytes: u64,
    messages_per_second: f64,
    mib_per_second: f64,
    latency_p50_us: u64,
    latency_p95_us: u64,
    latency_p99_us: u64,
    verified_messages: usize,
    verified_bytes: u64,
    environment: &'static str,
}

struct TimedResult {
    wall: Duration,
    latencies: Vec<Duration>,
    verified_messages: usize,
    verified_bytes: u64,
}

fn parse_config() -> std::result::Result<Config, String> {
    let protocol = match env::var("IRONGRAPH_BROKER_PROTOCOL").as_deref() {
        Ok("kafka") | Err(_) => Protocol::Kafka,
        Ok("amqp") => Protocol::Amqp,
        Ok(other) => return Err(format!("unsupported IRONGRAPH_BROKER_PROTOCOL={other}")),
    };
    let direction = match env::var("IRONGRAPH_BROKER_DIRECTION").as_deref() {
        Ok("ingress") | Err(_) => Direction::Ingress,
        Ok("egress") => Direction::Egress,
        Ok("full_duplex") => Direction::FullDuplex,
        Ok(other) => return Err(format!("unsupported IRONGRAPH_BROKER_DIRECTION={other}")),
    };
    let payload_bytes = env_usize("IRONGRAPH_BROKER_PAYLOAD_BYTES", 1024)?;
    let pipeline_depth = env_usize("IRONGRAPH_BROKER_PIPELINE_DEPTH", 64)?;
    let target_bytes = env_usize("IRONGRAPH_BROKER_TARGET_BYTES", 32 * 1024 * 1024)?;
    let default_messages = (target_bytes / payload_bytes).clamp(8, 100_000);
    let messages = env_usize("IRONGRAPH_BROKER_MESSAGES", default_messages)?;
    let sample = env_usize("IRONGRAPH_BROKER_SAMPLE", 1)?;
    let external_endpoint = match protocol {
        Protocol::Kafka => env::var("IRONGRAPH_BROKER_KAFKA_ENDPOINT").ok(),
        Protocol::Amqp => env::var("IRONGRAPH_BROKER_AMQP_ENDPOINT").ok(),
    };
    let existing_kafka_topic = env::var("IRONGRAPH_BROKER_EXISTING_KAFKA_TOPIC").ok();
    if existing_kafka_topic.is_some()
        && (!matches!(protocol, Protocol::Kafka) || !matches!(direction, Direction::Egress))
    {
        return Err(
            "IRONGRAPH_BROKER_EXISTING_KAFKA_TOPIC requires Kafka egress so setup cannot republish"
                .to_owned(),
        );
    }
    if payload_bytes > 7 * 1024 * 1024 {
        return Err("payload exceeds the common 7 MiB Kafka/AMQP benchmark ceiling".to_owned());
    }
    Ok(Config {
        protocol,
        direction,
        payload_bytes,
        messages,
        pipeline_depth,
        sample,
        external_endpoint,
        existing_kafka_topic,
    })
}

fn env_usize(name: &str, default: usize) -> std::result::Result<usize, String> {
    env::var(name).map_or(Ok(default), |value| {
        value
            .parse::<usize>()
            .ok()
            .filter(|value| *value > 0)
            .ok_or_else(|| format!("{name} must be a positive integer"))
    })
}

fn payload(bytes: usize) -> Arc<[u8]> {
    (0..bytes)
        .map(|index| ((index.wrapping_mul(31).wrapping_add(17)) & 0xff) as u8)
        .collect::<Vec<_>>()
        .into()
}

fn usage() -> Usage {
    let mut raw = std::mem::MaybeUninit::<libc::rusage>::zeroed();
    // SAFETY: `getrusage` initializes the complete `rusage` output on success and the pointer is
    // valid for one writable value. A failed call returns zeroed evidence rather than reading it.
    let status = unsafe { libc::getrusage(libc::RUSAGE_SELF, raw.as_mut_ptr()) };
    if status != 0 {
        return Usage {
            user_seconds: 0.0,
            system_seconds: 0.0,
            peak_rss_bytes: 0,
        };
    }
    // SAFETY: success from `getrusage` initialized the output.
    let raw = unsafe { raw.assume_init() };
    let seconds = |time: libc::timeval| time.tv_sec as f64 + (time.tv_usec as f64 / 1_000_000.0);
    #[cfg(any(target_os = "macos", target_os = "ios"))]
    let peak_rss_bytes = raw.ru_maxrss.max(0) as u64;
    #[cfg(not(any(target_os = "macos", target_os = "ios")))]
    let peak_rss_bytes = (raw.ru_maxrss.max(0) as u64).saturating_mul(1024);
    Usage {
        user_seconds: seconds(raw.ru_utime),
        system_seconds: seconds(raw.ru_stime),
        peak_rss_bytes,
    }
}

fn percentile(latencies: &mut [Duration], percentile: usize) -> u64 {
    if latencies.is_empty() {
        return 0;
    }
    latencies.sort_unstable();
    let index = (latencies.len() - 1).saturating_mul(percentile) / 100;
    latencies[index].as_micros().min(u128::from(u64::MAX)) as u64
}

fn direct_kafka_publish(
    coordinator: &Arc<LocalCoordinator>,
    project: ProjectId,
    body: &[u8],
    messages: usize,
    batch: usize,
) -> Result<()> {
    for start in (0..messages).step_by(batch.max(1)) {
        let count = batch.min(messages - start);
        let records = (0..count)
            .map(|_| KafkaBatchRecord {
                create_time_ms: None,
                key: None,
                headers: BTreeMap::new(),
                payload: body.to_vec(),
                value_is_null: false,
            })
            .collect();
        apply(
            coordinator,
            BrokerCommand::PublishKafkaBatch {
                project,
                topic: TOPIC.to_owned(),
                partition: 0,
                resolved_time_ms: 0,
                records,
            },
        )?;
    }
    Ok(())
}

fn verify_kafka_payloads(
    coordinator: &Arc<LocalCoordinator>,
    project: ProjectId,
    expected_messages: usize,
) -> Result<(usize, u64)> {
    let mut offset = 0_u64;
    let mut messages = 0_usize;
    let mut bytes = 0_u64;
    while messages < expected_messages {
        let records = coordinator.fetch_partition(project, TOPIC, 0, offset, 128 * 1024 * 1024)?;
        if records.is_empty() {
            break;
        }
        for (record_offset, record) in records {
            offset = record_offset.saturating_add(1);
            messages = messages.saturating_add(1);
            bytes = bytes.saturating_add(record.payload.len() as u64);
        }
    }
    Ok((messages, bytes))
}

fn direct_amqp_publish(
    coordinator: &Arc<LocalCoordinator>,
    project: ProjectId,
    body: &[u8],
    messages: usize,
) -> Result<()> {
    for _ in 0..messages {
        apply(
            coordinator,
            BrokerCommand::PublishAmqp {
                project,
                exchange: EXCHANGE.to_owned(),
                routing_key: ROUTING_KEY.to_owned(),
                mandatory: false,
                resolved_time_ms: 0,
                properties: BTreeMap::new(),
                headers: BTreeMap::new(),
                payload: body.to_vec(),
            },
        )?;
    }
    Ok(())
}

async fn kafka_cell(config: &Config, body: Arc<[u8]>) -> Result<TimedResult> {
    let project = ProjectId::random();
    let coordinator = Arc::new(LocalCoordinator::new()?);
    apply(
        &coordinator,
        BrokerCommand::CreateTopic {
            project,
            name: TOPIC.to_owned(),
            partitions: 1,
            retention: RetentionPolicy {
                max_age_ms: None,
                max_bytes: None,
            },
        },
    )?;
    if matches!(config.direction, Direction::Egress) {
        direct_kafka_publish(
            &coordinator,
            project,
            &body,
            config.messages,
            config.pipeline_depth,
        )?;
    }
    let address = reserve_loopback_port().await?;
    let shutdown = CancellationToken::new();
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
    let bootstrap = address.to_string();
    let producer: FutureProducer = ClientConfig::new()
        .set("bootstrap.servers", &bootstrap)
        .set("message.timeout.ms", "30000")
        .set("socket.timeout.ms", "30000")
        .set("queue.buffering.max.ms", "0")
        .set("message.max.bytes", (8 * 1024 * 1024).to_string())
        .set("batch.num.messages", config.pipeline_depth.to_string())
        .create()
        .map_err(|error| irongraph::Error::internal(error.to_string()))?;
    let consumer: StreamConsumer = ClientConfig::new()
        .set("bootstrap.servers", &bootstrap)
        .set("group.id", format!("throughput-{}", config.sample))
        .set("enable.auto.commit", "false")
        .set("socket.timeout.ms", "30000")
        .set("fetch.message.max.bytes", (8 * 1024 * 1024).to_string())
        .set("fetch.max.bytes", (8 * 1024 * 1024).to_string())
        .set("receive.message.max.bytes", (16 * 1024 * 1024).to_string())
        .create()
        .map_err(|error| irongraph::Error::internal(error.to_string()))?;
    let mut assignment = TopicPartitionList::new();
    assignment
        .add_partition_offset(TOPIC, 0, Offset::Beginning)
        .map_err(|error| irongraph::Error::internal(error.to_string()))?;
    consumer
        .assign(&assignment)
        .map_err(|error| irongraph::Error::internal(error.to_string()))?;

    let produce = || async {
        stream::iter(0..config.messages)
            .map(|_| {
                let producer = &producer;
                let body = Arc::clone(&body);
                async move {
                    let started = Instant::now();
                    producer
                        .send(
                            FutureRecord::<[u8], [u8]>::to(TOPIC).payload(&body),
                            Duration::from_secs(30),
                        )
                        .await
                        .map_err(|(error, _)| irongraph::Error::internal(error.to_string()))?;
                    Ok::<Duration, irongraph::Error>(started.elapsed())
                }
            })
            .buffer_unordered(config.pipeline_depth)
            .try_collect::<Vec<_>>()
            .await
    };
    let consume = || async {
        let mut latencies = Vec::with_capacity(config.messages);
        let mut bytes = 0_u64;
        for _ in 0..config.messages {
            let started = Instant::now();
            let message = tokio::time::timeout(Duration::from_secs(30), consumer.recv())
                .await
                .map_err(|_| irongraph::Error::internal("Kafka receive timed out"))?
                .map_err(|error| irongraph::Error::internal(error.to_string()))?;
            latencies.push(started.elapsed());
            bytes = bytes.saturating_add(message.payload_len() as u64);
        }
        Ok::<(Vec<Duration>, u64), irongraph::Error>((latencies, bytes))
    };

    let (wall, latencies, verified_messages, verified_bytes) = match config.direction {
        Direction::Ingress => {
            let started = Instant::now();
            let latencies = produce().await?;
            let wall = started.elapsed();
            let (messages, bytes) = verify_kafka_payloads(&coordinator, project, config.messages)?;
            (wall, latencies, messages, bytes)
        }
        Direction::Egress => {
            let started = Instant::now();
            let (latencies, bytes) = consume().await?;
            (started.elapsed(), latencies, config.messages, bytes)
        }
        Direction::FullDuplex => {
            let started = Instant::now();
            let (produced, consumed) = tokio::join!(produce(), consume());
            let mut latencies = produced?;
            let (consume_latencies, bytes) = consumed?;
            latencies.extend(consume_latencies);
            (started.elapsed(), latencies, config.messages, bytes)
        }
    };
    shutdown.cancel();
    Ok(TimedResult {
        wall,
        latencies,
        verified_messages,
        verified_bytes,
    })
}

async fn amqp_cell(config: &Config, body: Arc<[u8]>) -> Result<TimedResult> {
    let project = ProjectId::random();
    let coordinator = Arc::new(LocalCoordinator::new()?);
    apply(
        &coordinator,
        BrokerCommand::CreateExchange {
            project,
            name: EXCHANGE.to_owned(),
            kind: AmqpExchangeKind::Direct,
            durable: true,
            passive: false,
        },
    )?;
    apply(
        &coordinator,
        BrokerCommand::CreateQueue {
            project,
            name: QUEUE.to_owned(),
            kind: QueueKind::Classic,
            durable: true,
            passive: false,
            dead_letter_exchange: None,
            dead_letter_routing_key: None,
            retention: RetentionPolicy {
                max_age_ms: None,
                max_bytes: None,
            },
            exclusive_owner: None,
            auto_delete: false,
        },
    )?;
    apply(
        &coordinator,
        BrokerCommand::BindQueue {
            project,
            exchange: EXCHANGE.to_owned(),
            queue: QUEUE.to_owned(),
            routing_key: ROUTING_KEY.to_owned(),
        },
    )?;
    if matches!(config.direction, Direction::Egress) {
        direct_amqp_publish(&coordinator, project, &body, config.messages)?;
    }
    let address = reserve_loopback_port().await?;
    let shutdown = CancellationToken::new();
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
    let uri = format!("amqp://guest:guest@{address}/%2f");
    let connection = Connection::connect(&uri, ConnectionProperties::default())
        .await
        .map_err(|error| irongraph::Error::internal(error.to_string()))?;
    let channel = connection
        .create_channel()
        .await
        .map_err(|error| irongraph::Error::internal(error.to_string()))?;
    channel
        .queue_declare(
            QUEUE,
            QueueDeclareOptions {
                passive: true,
                durable: true,
                ..QueueDeclareOptions::default()
            },
            FieldTable::default(),
        )
        .await
        .map_err(|error| irongraph::Error::internal(error.to_string()))?;
    channel
        .confirm_select(ConfirmSelectOptions::default())
        .await
        .map_err(|error| irongraph::Error::internal(error.to_string()))?;
    let mut consumer = if matches!(config.direction, Direction::Ingress) {
        None
    } else {
        Some(
            channel
                .basic_consume(
                    QUEUE,
                    "throughput",
                    BasicConsumeOptions {
                        no_ack: true,
                        ..BasicConsumeOptions::default()
                    },
                    FieldTable::default(),
                )
                .await
                .map_err(|error| irongraph::Error::internal(error.to_string()))?,
        )
    };
    let publish = || async {
        stream::iter(0..config.messages)
            .map(|_| {
                let channel = &channel;
                let body = Arc::clone(&body);
                async move {
                    let started = Instant::now();
                    let confirm = channel
                        .basic_publish(
                            EXCHANGE,
                            ROUTING_KEY,
                            BasicPublishOptions::default(),
                            &body,
                            BasicProperties::default(),
                        )
                        .await
                        .map_err(|error| irongraph::Error::internal(error.to_string()))?;
                    let confirmation = confirm
                        .await
                        .map_err(|error| irongraph::Error::internal(error.to_string()))?;
                    if !confirmation.is_ack() {
                        return Err(irongraph::Error::internal(
                            "AMQP publisher confirm was not an acknowledgement",
                        ));
                    }
                    Ok::<Duration, irongraph::Error>(started.elapsed())
                }
            })
            .buffer_unordered(config.pipeline_depth)
            .try_collect::<Vec<_>>()
            .await
    };
    let consume = async {
        let consumer = consumer
            .as_mut()
            .ok_or_else(|| irongraph::Error::internal("AMQP consumer was not created"))?;
        let mut latencies = Vec::with_capacity(config.messages);
        let mut bytes = 0_u64;
        for _ in 0..config.messages {
            let started = Instant::now();
            let delivery = tokio::time::timeout(Duration::from_secs(30), consumer.next())
                .await
                .map_err(|_| irongraph::Error::internal("AMQP receive timed out"))?
                .ok_or_else(|| irongraph::Error::internal("AMQP consumer ended"))?
                .map_err(|error| irongraph::Error::internal(error.to_string()))?;
            latencies.push(started.elapsed());
            bytes = bytes.saturating_add(delivery.data.len() as u64);
        }
        Ok::<(Vec<Duration>, u64), irongraph::Error>((latencies, bytes))
    };

    let (wall, latencies, verified_bytes) = match config.direction {
        Direction::Ingress => {
            let started = Instant::now();
            let latencies = publish().await?;
            let wall = started.elapsed();
            let info = coordinator
                .queue_info(project, QUEUE)?
                .ok_or_else(|| irongraph::Error::internal("benchmark queue disappeared"))?;
            if info.message_count != config.messages as u64 {
                return Err(irongraph::Error::internal(
                    "AMQP ingress count differs from confirmed publishes",
                ));
            }
            (
                wall,
                latencies,
                (config.messages as u64).saturating_mul(config.payload_bytes as u64),
            )
        }
        Direction::Egress => {
            let started = Instant::now();
            let (latencies, bytes) = consume.await?;
            (started.elapsed(), latencies, bytes)
        }
        Direction::FullDuplex => {
            let started = Instant::now();
            let (produced, consumed) = tokio::join!(publish(), consume);
            let mut latencies = produced?;
            let (consume_latencies, bytes) = consumed?;
            latencies.extend(consume_latencies);
            (started.elapsed(), latencies, bytes)
        }
    };
    shutdown.cancel();
    Ok(TimedResult {
        wall,
        latencies,
        verified_messages: config.messages,
        verified_bytes,
    })
}

fn kafka_clients(
    endpoint: &str,
    group: &str,
    pipeline_depth: usize,
) -> Result<(FutureProducer, StreamConsumer)> {
    let buffering_ms = if pipeline_depth == 1 { "0" } else { "5" };
    let producer: FutureProducer = ClientConfig::new()
        .set("bootstrap.servers", endpoint)
        .set("message.timeout.ms", "30000")
        .set("socket.timeout.ms", "30000")
        .set("queue.buffering.max.ms", buffering_ms)
        .set("batch.num.messages", pipeline_depth.to_string())
        .set("queue.buffering.max.messages", "1000000")
        .set("message.max.bytes", (8 * 1024 * 1024).to_string())
        .create()
        .map_err(|error| irongraph::Error::internal(error.to_string()))?;
    let consumer: StreamConsumer = ClientConfig::new()
        .set("bootstrap.servers", endpoint)
        .set("group.id", group)
        .set("enable.auto.commit", "false")
        .set("socket.timeout.ms", "30000")
        .set("fetch.message.max.bytes", (8 * 1024 * 1024).to_string())
        .set("fetch.max.bytes", (32 * 1024 * 1024).to_string())
        .set("receive.message.max.bytes", (64 * 1024 * 1024).to_string())
        .create()
        .map_err(|error| irongraph::Error::internal(error.to_string()))?;
    Ok((producer, consumer))
}

async fn external_kafka_cell(
    config: &Config,
    body: Arc<[u8]>,
    endpoint: &str,
) -> Result<TimedResult> {
    let suffix = format!("{}-{}", std::process::id(), config.sample);
    let topic = config
        .existing_kafka_topic
        .clone()
        .unwrap_or_else(|| format!("throughput-{suffix}"));
    if config.existing_kafka_topic.is_none() {
        let admin: AdminClient<DefaultClientContext> = ClientConfig::new()
            .set("bootstrap.servers", endpoint)
            .create()
            .map_err(|error| irongraph::Error::internal(error.to_string()))?;
        let declarations = [NewTopic::new(&topic, 1, TopicReplication::Fixed(1))];
        let created = admin
            .create_topics(&declarations, &AdminOptions::new())
            .await
            .map_err(|error| irongraph::Error::internal(error.to_string()))?;
        if created.iter().any(Result::is_err) {
            return Err(irongraph::Error::internal(format!(
                "Kafka topic declaration failed: {created:?}"
            )));
        }
    }
    let (producer, consumer) = kafka_clients(
        endpoint,
        &format!("throughput-{suffix}"),
        config.pipeline_depth,
    )?;
    let mut assignment = TopicPartitionList::new();
    assignment
        .add_partition_offset(&topic, 0, Offset::Beginning)
        .map_err(|error| irongraph::Error::internal(error.to_string()))?;
    consumer
        .assign(&assignment)
        .map_err(|error| irongraph::Error::internal(error.to_string()))?;

    let produce = || async {
        stream::iter(0..config.messages)
            .map(|_| {
                let producer = &producer;
                let body = Arc::clone(&body);
                let topic = &topic;
                async move {
                    let started = Instant::now();
                    producer
                        .send(
                            FutureRecord::<[u8], [u8]>::to(topic).payload(&body),
                            Duration::from_secs(30),
                        )
                        .await
                        .map_err(|(error, _)| irongraph::Error::internal(error.to_string()))?;
                    Ok::<Duration, irongraph::Error>(started.elapsed())
                }
            })
            .buffer_unordered(config.pipeline_depth)
            .try_collect::<Vec<_>>()
            .await
    };
    let consume = || async {
        let mut latencies = Vec::with_capacity(config.messages);
        let mut bytes = 0_u64;
        for _ in 0..config.messages {
            let started = Instant::now();
            let message = tokio::time::timeout(Duration::from_secs(30), consumer.recv())
                .await
                .map_err(|_| irongraph::Error::internal("Kafka receive timed out"))?
                .map_err(|error| irongraph::Error::internal(error.to_string()))?;
            latencies.push(started.elapsed());
            bytes = bytes.saturating_add(message.payload_len() as u64);
        }
        Ok::<(Vec<Duration>, u64), irongraph::Error>((latencies, bytes))
    };

    let (wall, latencies, verified_bytes) = match config.direction {
        Direction::Ingress => {
            let started = Instant::now();
            let latencies = produce().await?;
            let wall = started.elapsed();
            let (_, bytes) = consume().await?;
            (wall, latencies, bytes)
        }
        Direction::Egress => {
            if config.existing_kafka_topic.is_none() {
                let _ = produce().await?;
            }
            let started = Instant::now();
            let (latencies, bytes) = consume().await?;
            (started.elapsed(), latencies, bytes)
        }
        Direction::FullDuplex => {
            let started = Instant::now();
            let (produced, consumed) = tokio::join!(produce(), consume());
            let mut latencies = produced?;
            let (consume_latencies, bytes) = consumed?;
            latencies.extend(consume_latencies);
            (started.elapsed(), latencies, bytes)
        }
    };
    Ok(TimedResult {
        wall,
        latencies,
        verified_messages: config.messages,
        verified_bytes,
    })
}

async fn external_amqp_cell(
    config: &Config,
    body: Arc<[u8]>,
    endpoint: &str,
) -> Result<TimedResult> {
    let suffix = format!("{}-{}", std::process::id(), config.sample);
    let exchange = format!("throughput-{suffix}");
    let queue = format!("throughput-{suffix}.queue");
    let connection = Connection::connect(endpoint, ConnectionProperties::default())
        .await
        .map_err(|error| irongraph::Error::internal(error.to_string()))?;
    let channel = connection
        .create_channel()
        .await
        .map_err(|error| irongraph::Error::internal(error.to_string()))?;
    channel
        .exchange_declare(
            &exchange,
            ExchangeKind::Direct,
            ExchangeDeclareOptions {
                durable: true,
                ..ExchangeDeclareOptions::default()
            },
            FieldTable::default(),
        )
        .await
        .map_err(|error| irongraph::Error::internal(error.to_string()))?;
    channel
        .queue_declare(
            &queue,
            QueueDeclareOptions {
                durable: true,
                ..QueueDeclareOptions::default()
            },
            FieldTable::default(),
        )
        .await
        .map_err(|error| irongraph::Error::internal(error.to_string()))?;
    channel
        .queue_bind(
            &queue,
            &exchange,
            ROUTING_KEY,
            QueueBindOptions::default(),
            FieldTable::default(),
        )
        .await
        .map_err(|error| irongraph::Error::internal(error.to_string()))?;
    channel
        .confirm_select(ConfirmSelectOptions::default())
        .await
        .map_err(|error| irongraph::Error::internal(error.to_string()))?;

    let publish = || async {
        stream::iter(0..config.messages)
            .map(|_| {
                let channel = &channel;
                let body = Arc::clone(&body);
                let exchange = &exchange;
                async move {
                    let started = Instant::now();
                    let confirm = channel
                        .basic_publish(
                            exchange,
                            ROUTING_KEY,
                            BasicPublishOptions::default(),
                            &body,
                            BasicProperties::default(),
                        )
                        .await
                        .map_err(|error| irongraph::Error::internal(error.to_string()))?;
                    let confirmation = confirm
                        .await
                        .map_err(|error| irongraph::Error::internal(error.to_string()))?;
                    if !confirmation.is_ack() {
                        return Err(irongraph::Error::internal(
                            "AMQP publisher confirm was not an acknowledgement",
                        ));
                    }
                    Ok::<Duration, irongraph::Error>(started.elapsed())
                }
            })
            .buffer_unordered(config.pipeline_depth)
            .try_collect::<Vec<_>>()
            .await
    };
    let consume = || async {
        let mut consumer = channel
            .basic_consume(
                &queue,
                &format!("throughput-{suffix}"),
                BasicConsumeOptions {
                    no_ack: true,
                    ..BasicConsumeOptions::default()
                },
                FieldTable::default(),
            )
            .await
            .map_err(|error| irongraph::Error::internal(error.to_string()))?;
        let mut latencies = Vec::with_capacity(config.messages);
        let mut bytes = 0_u64;
        for _ in 0..config.messages {
            let started = Instant::now();
            let delivery = tokio::time::timeout(Duration::from_secs(30), consumer.next())
                .await
                .map_err(|_| irongraph::Error::internal("AMQP receive timed out"))?
                .ok_or_else(|| irongraph::Error::internal("AMQP consumer ended"))?
                .map_err(|error| irongraph::Error::internal(error.to_string()))?;
            latencies.push(started.elapsed());
            bytes = bytes.saturating_add(delivery.data.len() as u64);
        }
        Ok::<(Vec<Duration>, u64), irongraph::Error>((latencies, bytes))
    };

    let (wall, latencies, verified_bytes) = match config.direction {
        Direction::Ingress => {
            let started = Instant::now();
            let latencies = publish().await?;
            let wall = started.elapsed();
            let (_, bytes) = consume().await?;
            (wall, latencies, bytes)
        }
        Direction::Egress => {
            let _ = publish().await?;
            let started = Instant::now();
            let (latencies, bytes) = consume().await?;
            (started.elapsed(), latencies, bytes)
        }
        Direction::FullDuplex => {
            let started = Instant::now();
            let (produced, consumed) = tokio::join!(publish(), consume());
            let mut latencies = produced?;
            let (consume_latencies, bytes) = consumed?;
            latencies.extend(consume_latencies);
            (started.elapsed(), latencies, bytes)
        }
    };
    Ok(TimedResult {
        wall,
        latencies,
        verified_messages: config.messages,
        verified_bytes,
    })
}

fn main() {
    if let Err(error) = run() {
        eprintln!("broker throughput cell failed: {error}");
        std::process::exit(1);
    }
}

fn run() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let config = parse_config()?;
    let body = payload(config.payload_bytes);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(4)
        .enable_all()
        .build()?;
    let before = usage();
    let measured = runtime.block_on(async {
        match (config.protocol, config.external_endpoint.as_deref()) {
            (Protocol::Kafka, Some(endpoint)) => external_kafka_cell(&config, body, endpoint).await,
            (Protocol::Amqp, Some(endpoint)) => external_amqp_cell(&config, body, endpoint).await,
            (Protocol::Kafka, None) => kafka_cell(&config, body).await,
            (Protocol::Amqp, None) => amqp_cell(&config, body).await,
        }
    })?;
    let after = usage();
    if measured.verified_messages != config.messages
        || measured.verified_bytes
            != (config.messages as u64).saturating_mul(config.payload_bytes as u64)
    {
        return Err("verified message or byte total differs from the requested cell".into());
    }
    let transferred_messages = if matches!(config.direction, Direction::FullDuplex) {
        config.messages.saturating_mul(2)
    } else {
        config.messages
    };
    let transferred_bytes =
        (transferred_messages as u64).saturating_mul(config.payload_bytes as u64);
    let wall_seconds = measured.wall.as_secs_f64();
    let mut p50 = measured.latencies.clone();
    let mut p95 = measured.latencies.clone();
    let mut p99 = measured.latencies;
    let result = CellResult {
        schema_version: 1,
        protocol: config.protocol,
        direction: config.direction,
        payload_bytes: config.payload_bytes,
        messages: config.messages,
        pipeline_depth: config.pipeline_depth,
        concurrency: 1,
        sample: config.sample,
        wall_seconds,
        cpu_user_seconds: (after.user_seconds - before.user_seconds).max(0.0),
        cpu_system_seconds: (after.system_seconds - before.system_seconds).max(0.0),
        peak_rss_bytes: after.peak_rss_bytes.max(before.peak_rss_bytes),
        messages_per_second: transferred_messages as f64 / wall_seconds,
        mib_per_second: transferred_bytes as f64 / (1024.0 * 1024.0) / wall_seconds,
        latency_p50_us: percentile(&mut p50, 50),
        latency_p95_us: percentile(&mut p95, 95),
        latency_p99_us: percentile(&mut p99, 99),
        verified_messages: measured.verified_messages,
        verified_bytes: measured.verified_bytes,
        environment: if config.external_endpoint.is_some() {
            "external_server"
        } else {
            "component_fixture"
        },
    };
    println!(
        "IRONGRAPH_BROKER_RESULT={}",
        serde_json::to_string(&result)?
    );
    Ok(())
}
