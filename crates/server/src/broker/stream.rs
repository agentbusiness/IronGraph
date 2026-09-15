use std::{borrow::Cow, collections::BTreeMap, io::Read as _, net::SocketAddr, sync::Arc};

use bytes::Bytes;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::{sync::Semaphore, task::JoinSet};

use crate::{CommitAcknowledgement, Error, ErrorCode, ProjectId, Result, engine::ApplicationWait};

use super::engine::{
    BrokerCommand, BrokerCoordinator, BrokerReply, ConnectionBrokerCoordinator, IngressMetadata,
    KafkaBatchRecord, RetentionPolicy,
};
use super::memory::process_broker_memory;

const MAX_FRAME: usize = 128 * 1024 * 1024;
const MAX_RESPONSE_BODY: usize = MAX_FRAME - std::mem::size_of::<i32>();
const REQUEST_WORKSPACE_OVERHEAD: usize = 4 * 1024 * 1024;
const WRITER_ALLOCATION_QUANTUM: usize = 64 * 1024;
const MAX_CONNECTIONS: usize = 512;
const MAX_FETCH_WAIT_MILLIS: u32 = 30_000;
const MAX_FETCH_WAKEUPS: usize = 64;
const NONE: i16 = 0;
const UNKNOWN_SERVER_ERROR: i16 = -1;
const OFFSET_OUT_OF_RANGE: i16 = 1;
const UNKNOWN_TOPIC_OR_PARTITION: i16 = 3;
const REQUEST_TIMED_OUT: i16 = 7;
const MESSAGE_TOO_LARGE: i16 = 10;
const NOT_ENOUGH_REPLICAS: i16 = 19;
const ILLEGAL_GENERATION: i16 = 22;
const INCONSISTENT_GROUP_PROTOCOL: i16 = 23;
const UNKNOWN_MEMBER_ID: i16 = 25;
const UNSUPPORTED_VERSION: i16 = 35;
const TOPIC_ALREADY_EXISTS: i16 = 36;
const INVALID_REQUEST: i16 = 42;
/// Produce v3 and Fetch v4 are the first version of each API that carries the v2 `RecordBatch`
/// format. A client that negotiates below them gets the legacy message set it can parse, and one
/// that reaches them gets record batches with headers. The advertised maxima and the versions the
/// `handle` dispatch accepts are the same constants so the two cannot drift apart.
const MAX_PRODUCE_VERSION: i16 = 3;
const MAX_FETCH_VERSION: i16 = 4;
const PRODUCE_RECORD_BATCH_VERSION: i16 = 3;
const FETCH_RECORD_BATCH_VERSION: i16 = 4;
const RECORD_BATCH_MAGIC: i8 = 2;
/// A v2 batch header, `baseOffset` through `recordCount` inclusive, before any record bytes.
const RECORD_BATCH_HEADER_BYTES: usize = 61;
/// `batchLength` counts the bytes after itself, so it excludes `baseOffset` and its own field.
const RECORD_BATCH_LENGTH_PREFIX_BYTES: usize = 12;
/// Offset inside a batch body at which the CRC-32C coverage starts: after the leader epoch, the
/// magic byte, and the crc field itself.
const RECORD_BATCH_CRC_END: usize = 9;
/// Smallest legal `batchLength`: a header with no records at all.
const RECORD_BATCH_BODY_MINIMUM: usize =
    RECORD_BATCH_HEADER_BYTES - RECORD_BATCH_LENGTH_PREFIX_BYTES;
/// A record is at least seven bytes: its length varint, attributes, and the five varints for
/// timestamp delta, offset delta, null key, null value, and an empty header array.
const RECORD_MINIMUM_BYTES: usize = 7;
/// A header is at least two varints: an empty name and a null value.
const RECORD_HEADER_MINIMUM_BYTES: usize = 2;
/// Ceiling on the records one produce request may carry, matched to the legacy message-set bound.
const MAXIMUM_RECORDS: usize = 1_000_000;
/// AMQP transport properties surface to a Stream consumer as record headers under this namespace.
///
/// A publisher's own AMQP headers pass through unprefixed so the two surfaces agree on them, which
/// leaves the properties needing a namespace of their own. Kafka record headers are an ordered
/// list that permits duplicate names, so an application header that happens to use this prefix
/// still arrives intact alongside the property rather than displacing it.
const AMQP_PROPERTY_HEADER_PREFIX: &str = "amqp.property.";

fn validate_advertised_endpoint(host: &str, port: u16) -> Result<()> {
    if host.is_empty()
        || host.len() > 253
        || host.bytes().any(|byte| byte.is_ascii_control())
        || port == 0
    {
        return Err(Error::invalid_data("Kafka advertised endpoint is invalid"));
    }
    Ok(())
}

/// Kafka listener with flexible ApiVersions negotiation and an explicitly bounded legacy API set.
pub struct StreamServer {
    address: SocketAddr,
    advertised_host: String,
    advertised_port: u16,
    project: ProjectId,
    coordinator: Arc<dyn BrokerCoordinator>,
}

impl StreamServer {
    #[must_use]
    pub fn new(
        address: SocketAddr,
        project: ProjectId,
        coordinator: Arc<dyn BrokerCoordinator>,
    ) -> Self {
        Self {
            address,
            advertised_host: address.ip().to_string(),
            advertised_port: address.port(),
            project,
            coordinator,
        }
    }

    pub fn with_advertised_endpoint(mut self, host: String, port: u16) -> Result<Self> {
        validate_advertised_endpoint(&host, port)?;
        self.advertised_host = host;
        self.advertised_port = port;
        Ok(self)
    }

    pub async fn run(self, shutdown: tokio_util::sync::CancellationToken) -> Result<()> {
        if !self.address.ip().is_loopback() {
            return Err(Error::new(
                crate::ErrorCode::AuthenticationFailed,
                "plain Kafka listener is restricted to loopback; remote Kafka requires mTLS",
            ));
        }
        let listener = tokio::net::TcpListener::bind(self.address).await?;
        let permits = Arc::new(Semaphore::new(MAX_CONNECTIONS));
        let mut connections = JoinSet::new();
        loop {
            tokio::select! {
                _ = shutdown.cancelled() => break,
                accepted = listener.accept() => {
                    let (stream, _) = accepted?;
                    let Ok(permit) = Arc::clone(&permits).try_acquire_owned() else {
                        drop(stream);
                        continue;
                    };
                    let coordinator = Arc::clone(&self.coordinator);
                    let project = self.project;
                    let endpoint = KafkaEndpoint {
                        host: self.advertised_host.clone(),
                        port: self.advertised_port,
                    };
                    connections.spawn(async move {
                        let _permit = permit;
                        if let Err(error) = serve(stream, project, coordinator, endpoint).await {
                            tracing::warn!(stage = "connection", category = ?error.code, "Kafka connection failed");
                        }
                    });
                }
                _ = connections.join_next(), if !connections.is_empty() => {}
            }
        }
        connections.abort_all();
        while connections.join_next().await.is_some() {}
        Ok(())
    }

    /// Serves one transport after the owning TLS acceptor has authenticated its immutable
    /// project-scoped Kafka credential.
    pub async fn serve_authenticated_transport<T>(
        transport: T,
        project: ProjectId,
        coordinator: Arc<dyn BrokerCoordinator>,
        advertised_host: String,
        advertised_port: u16,
    ) -> Result<()>
    where
        T: AsyncRead + AsyncWrite + Unpin + Send + 'static,
    {
        validate_advertised_endpoint(&advertised_host, advertised_port)?;
        serve(
            transport,
            project,
            coordinator,
            KafkaEndpoint {
                host: advertised_host,
                port: advertised_port,
            },
        )
        .await
    }
}

#[derive(Clone)]
struct KafkaEndpoint {
    host: String,
    port: u16,
}

async fn serve<T>(
    mut stream: T,
    project: ProjectId,
    coordinator: Arc<dyn BrokerCoordinator>,
    endpoint: KafkaEndpoint,
) -> Result<()>
where
    T: AsyncRead + AsyncWrite + Unpin,
{
    let coordinator: Arc<dyn BrokerCoordinator> =
        Arc::new(ConnectionBrokerCoordinator::new(coordinator));
    loop {
        let length = match stream.read_i32().await {
            Ok(value) => value,
            Err(error) if error.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(()),
            Err(error) => return Err(error.into()),
        };
        if length < 8 || length as usize > MAX_FRAME {
            return Err(Error::new(
                crate::ErrorCode::ProtocolViolation,
                "Kafka frame size is invalid",
            ));
        }
        let mut prefix = [0_u8; 8];
        stream.read_exact(&mut prefix).await?;
        let api_key = i16::from_be_bytes([prefix[0], prefix[1]]);
        let api_version = i16::from_be_bytes([prefix[2], prefix[3]]);
        let correlation = i32::from_be_bytes(prefix[4..8].try_into().map_err(|_| {
            Error::internal("Kafka fixed request header length changed unexpectedly")
        })?);
        let body_length = length as usize - prefix.len();
        let fixed_workspace = match api_key {
            // Produce and Fetch are sized from their bounded request bodies after those bytes are
            // available. Until then, only the request allocation itself is admitted.
            0 | 1 => 0,
            _ => MAX_RESPONSE_BODY,
        };
        let initial_bytes = body_length
            .checked_add(fixed_workspace)
            .and_then(|bytes| bytes.checked_add(REQUEST_WORKSPACE_OVERHEAD))
            .ok_or_else(|| Error::internal("Kafka request workspace overflow"))?;
        let governor = process_broker_memory();
        let mut memory = governor.reserve(initial_bytes).await?;
        let mut body = vec![0u8; body_length];
        stream.read_exact(&mut body).await?;
        if matches!(api_key, 0 | 1) {
            let request_workspace = if api_key == 0 {
                produce_workspace_bytes(api_version, &body)?
            } else {
                fetch_workspace_bytes(api_version, &body)?
            };
            memory.try_grow_to(body_length.checked_add(request_workspace).ok_or_else(|| {
                Error::internal("Kafka request workspace reservation overflow")
            })?)?;
        }
        // `Bytes::from(Vec)` transfers the request allocation without copying its full contents.
        // Clones handed to blocking request execution then increment only the shared owner.
        let body = Bytes::from(body);
        let mut changes = (api_key == 1).then(|| coordinator.subscribe_changes());
        let request_started = tokio::time::Instant::now();
        let mut fetch_deadline = None;
        let mut wakeups = 0_usize;
        let mut final_fetch_after_deadline = false;
        let handled = loop {
            let request_coordinator = Arc::clone(&coordinator);
            let request_endpoint = endpoint.clone();
            let request_body = body.clone();
            let handled = execute_request_blocking(
                &memory,
                api_key,
                api_version,
                correlation,
                project,
                request_coordinator,
                request_endpoint,
                request_body,
            )
            .await?;
            let Some(progress) = handled.fetch else {
                break handled;
            };
            if progress.data_bytes >= progress.minimum_bytes
                || progress.maximum_wait_millis == 0
                || progress.terminal_error
                || final_fetch_after_deadline
            {
                break handled;
            }
            let wait_deadline = *fetch_deadline.get_or_insert_with(|| {
                request_started
                    + std::time::Duration::from_millis(u64::from(
                        progress.maximum_wait_millis.min(MAX_FETCH_WAIT_MILLIS),
                    ))
            });
            let Some(change_receiver) = changes.as_mut() else {
                break handled;
            };
            let response = handled.response;
            if wakeups >= MAX_FETCH_WAKEUPS {
                // A hot partition must not force unbounded re-encoding, but the defensive wakeup
                // cap must not turn max_wait/min_bytes into an arbitrary early response either.
                // Coalesce all remaining notifications until the deadline, then execute exactly
                // one final fetch against the latest applied broker state.
                drop(response);
                tokio::time::sleep_until(wait_deadline).await;
                final_fetch_after_deadline = true;
                continue;
            }
            match tokio::time::timeout_at(wait_deadline, change_receiver.changed()).await {
                Ok(Ok(())) => {
                    drop(response);
                    wakeups += 1;
                }
                Ok(Err(_)) | Err(_) => {
                    break HandledRequest {
                        response,
                        should_respond: handled.should_respond,
                        fetch: handled.fetch,
                    };
                }
            }
        };
        if handled.should_respond {
            let framed_length = handled
                .response
                .len()
                .checked_add(std::mem::size_of::<i32>())
                .ok_or_else(|| Error::internal("Kafka response length overflow"))?;
            if framed_length > MAX_FRAME {
                return Err(Error::new(
                    ErrorCode::Backpressure,
                    "Kafka response exceeds the process frame bound",
                ));
            }
            stream
                .write_i32(
                    i32::try_from(framed_length)
                        .map_err(|_| Error::internal("Kafka response exceeds i32"))?,
                )
                .await?;
            stream.write_i32(correlation).await?;
            stream.write_all(&handled.response).await?;
            stream.flush().await?;
        }
        drop(memory);
    }
}

async fn execute_request_blocking(
    memory: &super::memory::BrokerMemoryReservation,
    api_key: i16,
    api_version: i16,
    correlation: i32,
    project: ProjectId,
    coordinator: Arc<dyn BrokerCoordinator>,
    endpoint: KafkaEndpoint,
    body: Bytes,
) -> Result<HandledRequest> {
    memory
        .spawn_blocking(move || {
            let span = tracing::warn_span!("kafka_request", api_key, api_version, correlation);
            span.in_scope(|| handle_request(
                api_key,
                api_version,
                project,
                &coordinator,
                &endpoint,
                &body,
            ).inspect_err(|error| {
                tracing::warn!(stage = "request", category = ?error.code, kafka_code = produce_error_code(error), "Kafka request failed before response");
            }))
        })
        .await
        .map_err(|error| Error::internal(format!("Kafka request task failed: {error}")))?
}

fn handle_request(
    api_key: i16,
    api_version: i16,
    project: ProjectId,
    coordinator: &Arc<dyn BrokerCoordinator>,
    endpoint: &KafkaEndpoint,
    body: &[u8],
) -> Result<HandledRequest> {
    let mut reader = Reader::new(body);
    if request_header_is_flexible(api_key, api_version) {
        let _client_id = reader.nullable_string()?;
        reader.tagged_fields()?;
    } else {
        let _client_id = reader.nullable_string()?;
    }
    let outcome = handle(
        api_key,
        api_version,
        project,
        coordinator,
        endpoint,
        &mut reader,
    )?;
    reader.finish()?;
    Ok(outcome)
}

#[derive(Clone, Copy, Debug)]
struct FetchProgress {
    maximum_wait_millis: u32,
    minimum_bytes: usize,
    data_bytes: usize,
    terminal_error: bool,
}

struct HandledRequest {
    response: Vec<u8>,
    should_respond: bool,
    fetch: Option<FetchProgress>,
}

fn handle(
    api_key: i16,
    version: i16,
    project: ProjectId,
    coordinator: &Arc<dyn BrokerCoordinator>,
    endpoint: &KafkaEndpoint,
    reader: &mut Reader<'_>,
) -> Result<HandledRequest> {
    let mut response = Writer::with_limit(MAX_RESPONSE_BODY);
    let mut fetch_progress = None;
    match api_key {
        18 if version >= 0 => api_versions(version, reader, &mut response)?,
        3 if (0..=1).contains(&version) => metadata(
            version,
            project,
            coordinator,
            endpoint,
            reader,
            &mut response,
        )?,
        19 if version == 0 => create_topics(project, coordinator, reader, &mut response)?,
        20 if version == 0 => delete_topics(project, coordinator, reader, &mut response)?,
        0 if (0..=MAX_PRODUCE_VERSION).contains(&version) => {
            let respond = produce(version, project, coordinator, reader, &mut response)?;
            return Ok(HandledRequest {
                response: response.try_finish()?,
                should_respond: respond,
                fetch: None,
            });
        }
        1 if (0..=MAX_FETCH_VERSION).contains(&version) => {
            fetch_progress = Some(fetch(version, project, coordinator, reader, &mut response)?)
        }
        2 if (0..=1).contains(&version) => {
            list_offsets(version, project, coordinator, reader, &mut response)?
        }
        10 if version == 0 => find_coordinator(endpoint, reader, &mut response)?,
        11 if version == 0 => join_group(project, coordinator, reader, &mut response)?,
        14 if version == 0 => sync_group(project, coordinator, reader, &mut response)?,
        12 if version == 0 => heartbeat(project, coordinator, reader, &mut response)?,
        13 if version == 0 => leave_group(project, coordinator, reader, &mut response)?,
        8 if (0..=1).contains(&version) => {
            offset_commit(version, project, coordinator, reader, &mut response)?
        }
        9 if (0..=1).contains(&version) => {
            offset_fetch(project, coordinator, reader, &mut response)?
        }
        _ => {
            return Err(Error::new(
                crate::ErrorCode::ProtocolViolation,
                format!("Kafka API {api_key} version {version} is not advertised"),
            ));
        }
    }
    Ok(HandledRequest {
        response: response.try_finish()?,
        should_respond: true,
        fetch: fetch_progress,
    })
}

const fn request_header_is_flexible(api_key: i16, version: i16) -> bool {
    api_key == 18 && version >= 3
}

fn api_versions(version: i16, reader: &mut Reader<'_>, response: &mut Writer) -> Result<()> {
    if version >= 3 {
        let software_name = reader.compact_string()?;
        let software_version = reader.compact_string()?;
        if software_name.is_empty() || software_version.is_empty() {
            return Err(kafka_protocol_error(
                "Kafka client software name and version must be non-empty",
            ));
        }
        reader.tagged_fields()?;
    }
    response.i16(if version > 3 {
        UNSUPPORTED_VERSION
    } else {
        NONE
    });
    let versions = [
        (0, 0, MAX_PRODUCE_VERSION),
        (1, 0, MAX_FETCH_VERSION),
        (2, 0, 1),
        (3, 0, 1),
        (8, 0, 1),
        (9, 0, 1),
        (10, 0, 0),
        (11, 0, 0),
        (12, 0, 0),
        (13, 0, 0),
        (14, 0, 0),
        (18, 0, 3),
        (19, 0, 0),
        (20, 0, 0),
    ];
    if version > 3 {
        response.array_len(versions.len());
        for (key, minimum, maximum) in versions {
            response.i16(key);
            response.i16(minimum);
            response.i16(maximum);
        }
    } else if version >= 3 {
        response.compact_array_len(versions.len())?;
        for (key, minimum, maximum) in versions {
            response.i16(key);
            response.i16(minimum);
            response.i16(maximum);
            response.empty_tagged_fields();
        }
        response.i32(0);
        response.empty_tagged_fields();
    } else {
        response.array_len(versions.len());
        for (key, minimum, maximum) in versions {
            response.i16(key);
            response.i16(minimum);
            response.i16(maximum);
        }
        if version >= 1 {
            response.i32(0);
        }
    }
    Ok(())
}

fn metadata(
    version: i16,
    project: ProjectId,
    coordinator: &Arc<dyn BrokerCoordinator>,
    endpoint: &KafkaEndpoint,
    reader: &mut Reader<'_>,
    response: &mut Writer,
) -> Result<()> {
    let requested = reader.nullable_string_array()?;
    let all = coordinator.topic_metadata(project)?;
    let topics = match requested {
        None => all,
        Some(names) => names
            .into_iter()
            .map(|name| {
                let partitions = all
                    .iter()
                    .find(|(candidate, _)| candidate == &name)
                    .map_or(0, |(_, count)| *count);
                (name, partitions)
            })
            .collect(),
    };
    response.array_len(1);
    response.i32(0);
    response.string(&endpoint.host)?;
    response.i32(i32::from(endpoint.port));
    if version >= 1 {
        response.nullable_string(None)?;
        response.i32(0);
    }
    response.array_len(topics.len());
    for (name, count) in topics {
        response.i16(if count == 0 {
            UNKNOWN_TOPIC_OR_PARTITION
        } else {
            NONE
        });
        response.string(&name)?;
        if version >= 1 {
            response.i8(0);
        }
        response.array_len(count);
        for partition in 0..count {
            response.i16(NONE);
            response.i32(partition as i32);
            response.i32(0);
            response.array_len(1);
            response.i32(0);
            response.array_len(1);
            response.i32(0);
        }
    }
    Ok(())
}

fn create_topics(
    project: ProjectId,
    coordinator: &Arc<dyn BrokerCoordinator>,
    reader: &mut Reader<'_>,
    response: &mut Writer,
) -> Result<()> {
    let count = reader.array_count()?;
    let mut declarations = Vec::with_capacity(count);
    for _ in 0..count {
        let name = reader.string()?.to_owned();
        let partitions = reader.i32()?;
        let requested_copy_count = reader.i16()?;
        let assignments = reader.array_count()?;
        for _ in 0..assignments {
            let _partition = reader.i32()?;
            let brokers = reader.array_count()?;
            for _ in 0..brokers {
                let _ = reader.i32()?;
            }
        }
        let configs = reader.array_count()?;
        let mut max_age_ms = None;
        let mut max_bytes = None;
        let mut invalid_config = false;
        for _ in 0..configs {
            let key = reader.string()?;
            let value = reader.nullable_string()?;
            match (key, value) {
                ("retention.ms", Some(value)) => match value.parse() {
                    Ok(value) => max_age_ms = Some(value),
                    Err(_) => invalid_config = true,
                },
                ("retention.bytes", Some(value)) => match value.parse() {
                    Ok(value) => max_bytes = Some(value),
                    Err(_) => invalid_config = true,
                },
                _ => invalid_config = true,
            }
        }
        let command = if partitions <= 0 || requested_copy_count <= 0 || invalid_config {
            Err(Error::invalid_data("invalid topic declaration"))
        } else {
            Ok(BrokerCommand::CreateTopic {
                project,
                name: name.clone(),
                partitions: u16::try_from(partitions)
                    .map_err(|_| Error::invalid_data("partition count exceeds u16"))?,
                retention: RetentionPolicy {
                    max_age_ms,
                    max_bytes,
                },
            })
        };
        declarations.push((name, command));
    }
    let timeout = reader.i32()?;
    if timeout < 0 {
        return Err(kafka_protocol_error(
            "Kafka create-topics timeout is negative",
        ));
    }
    let timeout = u32::try_from(timeout.max(1))
        .map_err(|_| kafka_protocol_error("Kafka create-topics timeout exceeds u32"))?;
    let replies = declarations
        .into_iter()
        .map(|(name, command)| {
            let error = match command.and_then(|command| {
                coordinator.submit_with_timeout(command, CommitAcknowledgement::Published, timeout)
            }) {
                Ok(commit) if matches!(commit.reply, BrokerReply::TopicAlreadyExists) => {
                    TOPIC_ALREADY_EXISTS
                }
                Ok(_) => NONE,
                Err(error) => admin_error_code(&error),
            };
            (name, error)
        })
        .collect::<Vec<_>>();
    response.array_len(replies.len());
    for (name, error) in replies {
        response.string(&name)?;
        response.i16(error);
    }
    Ok(())
}

fn delete_topics(
    project: ProjectId,
    coordinator: &Arc<dyn BrokerCoordinator>,
    reader: &mut Reader<'_>,
    response: &mut Writer,
) -> Result<()> {
    let count = reader.array_count()?;
    let mut names = Vec::with_capacity(count);
    for _ in 0..count {
        names.push(reader.string()?.to_owned());
    }
    let timeout = reader.i32()?;
    if timeout < 0 {
        return Err(kafka_protocol_error(
            "Kafka delete-topics timeout is negative",
        ));
    }
    let timeout = u32::try_from(timeout.max(1))
        .map_err(|_| kafka_protocol_error("Kafka delete-topics timeout exceeds u32"))?;
    let replies = names
        .into_iter()
        .map(|name| {
            let error = coordinator
                .submit_with_timeout(
                    BrokerCommand::DeleteTopic {
                        project,
                        name: name.clone(),
                    },
                    CommitAcknowledgement::Published,
                    timeout,
                )
                .map(|_| NONE)
                .unwrap_or_else(|error| admin_error_code(&error));
            (name, error)
        })
        .collect::<Vec<_>>();
    response.array_len(replies.len());
    for (name, error) in replies {
        response.string(&name)?;
        response.i16(error);
    }
    Ok(())
}

fn produce(
    version: i16,
    project: ProjectId,
    coordinator: &Arc<dyn BrokerCoordinator>,
    reader: &mut Reader<'_>,
    response: &mut Writer,
) -> Result<bool> {
    if version >= PRODUCE_RECORD_BATCH_VERSION && reader.nullable_string()?.is_some() {
        // A transactional produce is only meaningful alongside InitProducerId, AddPartitionsToTxn,
        // and EndTxn, none of which is advertised. Accepting the id would let a client believe its
        // writes participate in a transaction this broker would never commit or abort.
        return Err(kafka_protocol_error(
            "Kafka transactional produce is not supported",
        ));
    }
    let required_acks = reader.i16()?;
    let timeout = reader.i32()?;
    if timeout < 0 {
        return Err(kafka_protocol_error("Kafka produce timeout is negative"));
    }
    let timeout = u32::try_from(timeout.max(1))
        .map_err(|_| kafka_protocol_error("Kafka produce timeout exceeds u32"))?;
    let wait = match required_acks {
        -1..=1 => CommitAcknowledgement::Published,
        _ => {
            return Err(Error::new(
                crate::ErrorCode::ProtocolViolation,
                "Kafka acks is invalid",
            ));
        }
    };
    let topic_count = reader.array_count()?;
    let mut replies = Vec::new();
    for _ in 0..topic_count {
        let topic = reader.string()?.to_owned();
        let partition_count = reader.array_count()?;
        for _ in 0..partition_count {
            let partition = reader.i32()?;
            let message_set = reader.bytes()?.ok_or_else(|| {
                Error::new(
                    crate::ErrorCode::ProtocolViolation,
                    "produce message set is null",
                )
            })?;
            let decoded = if version >= PRODUCE_RECORD_BATCH_VERSION {
                decode_record_batches(message_set)
            } else {
                decode_message_set(message_set, version)
            };
            let (error, first_offset) = match decoded {
                Ok(messages) if !messages.is_empty() => {
                    let records = messages
                        .into_iter()
                        .map(|message| KafkaBatchRecord {
                            create_time_ms: message.create_time_ms,
                            key: message.key,
                            // The canonical envelope models a header as a named value, so repeated
                            // names collapse to the last occurrence and an explicitly null value
                            // is retained as an empty one. Both are wire shapes Kafka permits but
                            // no mainstream client emits.
                            headers: message
                                .headers
                                .into_iter()
                                .map(|(name, value)| (name, value.unwrap_or_default()))
                                .collect(),
                            value_is_null: message.value.is_none(),
                            payload: message.value.unwrap_or_default(),
                        })
                        .collect();
                    match coordinator.submit_with_timeout(
                        BrokerCommand::PublishKafkaBatch {
                            project,
                            topic: topic.clone(),
                            partition,
                            resolved_time_ms: 0,
                            records,
                        },
                        wait,
                        timeout,
                    ) {
                        Ok(commit) => match commit.application {
                            ApplicationWait::Complete => match commit.reply {
                                BrokerReply::KafkaBatchPublished { first_offset, .. } => {
                                    (NONE, i64::try_from(first_offset).unwrap_or(-1))
                                }
                                _ => {
                                    let error = Error::internal(
                                        "broker returned an unexpected publish reply",
                                    );
                                    log_produce_failure(
                                        "acknowledgement",
                                        version,
                                        partition,
                                        &error,
                                    );
                                    (UNKNOWN_SERVER_ERROR, -1)
                                }
                            },
                        },
                        Err(error) => {
                            log_produce_failure("submission", version, partition, &error);
                            (produce_error_code(&error), -1)
                        }
                    }
                }
                Ok(_) => {
                    let error = kafka_protocol_error("produce batch contains no records");
                    log_produce_failure("validation", version, partition, &error);
                    (INVALID_REQUEST, -1)
                }
                Err(error) => {
                    log_produce_failure("decoding", version, partition, &error);
                    (produce_error_code(&error), -1)
                }
            };
            replies.push((topic.clone(), partition, error, first_offset));
        }
    }
    if required_acks == 0 {
        return Ok(false);
    }
    let grouped = group_partition_replies(replies);
    response.array_len(grouped.len());
    for (topic, partitions) in grouped {
        response.string(&topic)?;
        response.array_len(partitions.len());
        for (partition, error, offset) in partitions {
            response.i32(partition);
            response.i16(error);
            response.i64(offset);
            if version >= 2 {
                response.i64(-1);
            }
        }
    }
    if version >= 1 {
        response.i32(0);
    }
    Ok(true)
}

fn fetch(
    version: i16,
    project: ProjectId,
    coordinator: &Arc<dyn BrokerCoordinator>,
    reader: &mut Reader<'_>,
    response: &mut Writer,
) -> Result<FetchProgress> {
    let _replica_id = reader.i32()?;
    let maximum_wait_millis = reader.i32()?;
    let minimum_bytes = reader.i32()?;
    if maximum_wait_millis < 0 || minimum_bytes < 0 {
        return Err(kafka_protocol_error(
            "Kafka fetch wait and minimum bytes must be non-negative",
        ));
    }
    read_fetch_request_limits(version, reader)?;
    let topic_count = reader.array_count()?;
    let request_start = *reader;
    let limits = measure_fetch_request(version, topic_count, request_start)?;
    if version >= 1 {
        response.i32(0);
    }
    response.array_len(topic_count);
    let mut remaining_data_bytes = limits.maximum_data_bytes;
    let mut returned_data_bytes = 0_usize;
    let mut terminal_error = false;
    for _ in 0..topic_count {
        let topic = reader.string()?.to_owned();
        response.string(&topic)?;
        let partition_count = reader.array_count()?;
        response.array_len(partition_count);
        for _ in 0..partition_count {
            let partition = reader.i32()?;
            let offset = reader.i64()?;
            let maximum = reader.i32()?;
            response.i32(partition);
            let records = if offset < 0 || maximum <= 0 {
                Err(Error::invalid_data("invalid fetch offset/maximum"))
            } else if remaining_data_bytes == 0 {
                Ok(Vec::new())
            } else {
                let partition_maximum = usize::try_from(maximum)
                    .map_err(|_| kafka_protocol_error("Kafka fetch maximum exceeds usize"))?
                    .min(remaining_data_bytes);
                coordinator.fetch_partition(
                    project,
                    &topic,
                    partition,
                    offset as u64,
                    partition_maximum,
                )
            };
            match records {
                Ok(records) => {
                    response.i16(NONE);
                    let (high_watermark, _) = coordinator
                        .list_offset(project, &topic, partition, -1)?
                        .ok_or_else(|| Error::internal("Kafka latest offset is unavailable"))?;
                    let high_watermark = i64::try_from(high_watermark)
                        .map_err(|_| Error::internal("Kafka high watermark exceeds i64"))?;
                    response.i64(high_watermark);
                    if version >= FETCH_RECORD_BATCH_VERSION {
                        // No transactional batch is ever admitted, so every committed record is
                        // already stable and no transaction can be aborted.
                        response.i64(high_watermark);
                        response.array_len(0);
                    }
                    let length_position = response.len();
                    response.i32(0);
                    let message_set_start = response.len();
                    let client_partition_limit = usize::try_from(maximum.max(0))
                        .unwrap_or(0)
                        .min(remaining_data_bytes);
                    let mut partition_bytes = 0_usize;
                    for (index, (record_offset, payload)) in records.into_iter().enumerate() {
                        let encoded_length = encoded_message_length(&payload, version)?;
                        let aggregate_remaining =
                            remaining_data_bytes.saturating_sub(partition_bytes);
                        let partition_remaining =
                            client_partition_limit.saturating_sub(partition_bytes);
                        let permitted = if index == 0 {
                            // Legacy Fetch must return the first complete message even when it is
                            // larger than the partition byte hint. The process-wide response cap
                            // remains absolute, so the message is emitted only when it fits the
                            // aggregate response budget.
                            aggregate_remaining
                        } else {
                            aggregate_remaining.min(partition_remaining)
                        };
                        if encoded_length > permitted {
                            break;
                        }
                        let before = response.len();
                        encode_message(response, record_offset as i64, &payload, version)?;
                        debug_assert_eq!(response.len().saturating_sub(before), encoded_length);
                        partition_bytes = partition_bytes.saturating_add(encoded_length);
                        if partition_bytes >= client_partition_limit {
                            break;
                        }
                    }
                    response.patch_i32(
                        length_position,
                        i32::try_from(response.len().saturating_sub(message_set_start))
                            .map_err(|_| Error::internal("Kafka message set exceeds i32"))?,
                    )?;
                    remaining_data_bytes = remaining_data_bytes.saturating_sub(partition_bytes);
                    returned_data_bytes = returned_data_bytes.saturating_add(partition_bytes);
                }
                Err(error) => {
                    terminal_error = true;
                    response.i16(if error.code == ErrorCode::RetentionExpired {
                        OFFSET_OUT_OF_RANGE
                    } else {
                        UNKNOWN_TOPIC_OR_PARTITION
                    });
                    response.i64(-1);
                    if version >= FETCH_RECORD_BATCH_VERSION {
                        response.i64(-1);
                        response.array_len(0);
                    }
                    response.bytes(Some(&[]))?;
                }
            }
        }
    }
    Ok(FetchProgress {
        maximum_wait_millis: maximum_wait_millis as u32,
        minimum_bytes: minimum_bytes as usize,
        data_bytes: returned_data_bytes,
        terminal_error,
    })
}

#[derive(Clone, Copy, Debug)]
struct FetchLimits {
    minimum_response_bytes: usize,
    maximum_data_bytes: usize,
}

/// Consumes the request-level limits Fetch gained after v2 and validates them.
///
/// The v3 aggregate `max_bytes` is deliberately not folded into the response budget: storage must
/// return the first record of a partition even when that record exceeds a byte hint, so a small
/// aggregate hint would otherwise stall a consumer forever instead of merely shortening its
/// response. The per-partition maxima the client sends still bound each partition, and the frame
/// limit remains the absolute ceiling.
fn read_fetch_request_limits(version: i16, reader: &mut Reader<'_>) -> Result<()> {
    if version >= 3 {
        let maximum_bytes = reader.i32()?;
        if maximum_bytes < 0 {
            return Err(kafka_protocol_error(
                "Kafka fetch maximum bytes must be non-negative",
            ));
        }
    }
    if version >= FETCH_RECORD_BATCH_VERSION {
        let isolation_level = reader.i8()?;
        // read_committed and read_uncommitted are indistinguishable here because a transactional
        // batch is rejected at produce time, so nothing uncommitted can ever be readable.
        if !(0..=1).contains(&isolation_level) {
            return Err(kafka_protocol_error(
                "Kafka fetch isolation level is invalid",
            ));
        }
    }
    Ok(())
}

fn produce_workspace_bytes(version: i16, body: &[u8]) -> Result<usize> {
    let mut reader = Reader::new(body);
    if request_header_is_flexible(0, version) {
        let _client_id = reader.nullable_string()?;
        reader.tagged_fields()?;
    } else {
        let _client_id = reader.nullable_string()?;
    }
    if version >= PRODUCE_RECORD_BATCH_VERSION {
        let _transactional_id = reader.nullable_string()?;
    }
    let _required_acks = reader.i16()?;
    let _timeout = reader.i32()?;
    let topics = reader.array_count()?;
    let mut compressed = false;
    for _ in 0..topics {
        let _topic = reader.string()?;
        let partitions = reader.array_count()?;
        for _ in 0..partitions {
            let _partition = reader.i32()?;
            let message_set = reader.bytes()?.ok_or_else(|| {
                Error::new(ErrorCode::ProtocolViolation, "produce message set is null")
            })?;
            compressed |= if version >= PRODUCE_RECORD_BATCH_VERSION {
                record_batches_use_compression(message_set)?
            } else {
                message_set_uses_compression(message_set)?
            };
        }
    }
    reader.finish()?;
    let workspace = if compressed {
        // One standard compression envelope may retain its compressed value, bounded decoded
        // set, decoded record values, and record descriptors at the same time.
        MAX_FRAME
            .checked_mul(4)
            .ok_or_else(|| Error::internal("Kafka compressed produce workspace overflow"))?
    } else {
        body.len()
            .checked_mul(4)
            .ok_or_else(|| Error::internal("Kafka produce workspace overflow"))?
    };
    workspace
        .checked_add(REQUEST_WORKSPACE_OVERHEAD)
        .ok_or_else(|| Error::internal("Kafka produce workspace overflow"))
}

fn message_set_uses_compression(bytes: &[u8]) -> Result<bool> {
    let mut reader = Reader::new(bytes);
    let mut compressed = false;
    while reader.remaining() >= 12 {
        let _offset = reader.i64()?;
        let size = reader.i32()?;
        if size < 6 || size as usize > reader.remaining() {
            return Err(kafka_protocol_error("Kafka message size is invalid"));
        }
        let message = reader.take(size as usize)?;
        compressed |= message[5] & 0x07 != 0;
    }
    reader.finish()?;
    Ok(compressed)
}

fn fetch_workspace_bytes(version: i16, body: &[u8]) -> Result<usize> {
    let mut reader = Reader::new(body);
    if request_header_is_flexible(1, version) {
        let _client_id = reader.nullable_string()?;
        reader.tagged_fields()?;
    } else {
        let _client_id = reader.nullable_string()?;
    }
    let _replica_id = reader.i32()?;
    let _max_wait = reader.i32()?;
    let _min_bytes = reader.i32()?;
    read_fetch_request_limits(version, &mut reader)?;
    let topic_count = reader.array_count()?;
    let limits = measure_fetch_request(version, topic_count, reader)?;
    limits
        .minimum_response_bytes
        .checked_add(limits.maximum_data_bytes)
        .and_then(|bytes| bytes.checked_add(WRITER_ALLOCATION_QUANTUM))
        // Storage intentionally returns the first record even when it exceeds a partition byte
        // hint. Segment decoding can simultaneously retain one raw framed record and its decoded
        // payload while the bounded wire response is resident, so both full-record copies are
        // admitted independently before fetch begins.
        .and_then(|bytes| bytes.checked_add(MAX_FRAME.saturating_mul(2)))
        .and_then(|bytes| bytes.checked_add(REQUEST_WORKSPACE_OVERHEAD))
        .ok_or_else(|| Error::internal("Kafka fetch workspace overflow"))
}

fn measure_fetch_request(
    version: i16,
    topic_count: usize,
    reader: Reader<'_>,
) -> Result<FetchLimits> {
    measure_fetch_request_with_limit(version, topic_count, reader, MAX_RESPONSE_BODY)
}

fn measure_fetch_request_with_limit(
    version: i16,
    topic_count: usize,
    mut reader: Reader<'_>,
    response_limit: usize,
) -> Result<FetchLimits> {
    let mut minimum = std::mem::size_of::<i32>();
    if version >= 1 {
        minimum = minimum
            .checked_add(std::mem::size_of::<i32>())
            .ok_or_else(|| Error::internal("Kafka fetch response size overflow"))?;
    }
    for _ in 0..topic_count {
        let topic = reader.string()?;
        minimum = minimum
            .checked_add(std::mem::size_of::<i16>())
            .and_then(|bytes| bytes.checked_add(topic.len()))
            .and_then(|bytes| bytes.checked_add(std::mem::size_of::<i32>()))
            .ok_or_else(|| Error::internal("Kafka fetch response size overflow"))?;
        let partition_count = reader.array_count()?;
        // Partition index, error, and high watermark, plus the record set length prefix. From v4
        // each partition also carries the last stable offset and an aborted-transaction array.
        let partition_response_bytes = if version >= FETCH_RECORD_BATCH_VERSION {
            30
        } else {
            18
        };
        minimum = minimum
            .checked_add(
                partition_count
                    .checked_mul(partition_response_bytes)
                    .ok_or_else(|| {
                        Error::new(
                            ErrorCode::ProtocolViolation,
                            "Kafka fetch partition count overflows the response bound",
                        )
                    })?,
            )
            .ok_or_else(|| Error::internal("Kafka fetch response size overflow"))?;
        for _ in 0..partition_count {
            let _partition = reader.i32()?;
            let offset = reader.i64()?;
            let maximum = reader.i32()?;
            if offset < 0 || maximum <= 0 {
                continue;
            }
        }
    }
    if minimum > response_limit {
        return Err(Error::new(
            ErrorCode::ProtocolViolation,
            "Kafka fetch request requires an oversized minimum response",
        ));
    }
    // A legacy message-set Fetch must make progress by returning the first complete record even
    // when that record exceeds the partition byte hint. Consequently the aggregate data budget
    // cannot be derived from the sum of those hints (a maximum of one byte must still admit one
    // bounded record). The response frame limit is the sole aggregate ceiling; later records in
    // each partition continue to obey the client's partition hint.
    let maximum_data_bytes = response_limit - minimum;
    Ok(FetchLimits {
        minimum_response_bytes: minimum,
        maximum_data_bytes,
    })
}

fn list_offsets(
    version: i16,
    project: ProjectId,
    coordinator: &Arc<dyn BrokerCoordinator>,
    reader: &mut Reader<'_>,
    response: &mut Writer,
) -> Result<()> {
    let _replica_id = reader.i32()?;
    let topics = reader.array_count()?;
    response.array_len(topics);
    for _ in 0..topics {
        let topic = reader.string()?.to_owned();
        response.string(&topic)?;
        let partitions = reader.array_count()?;
        response.array_len(partitions);
        for _ in 0..partitions {
            let partition = reader.i32()?;
            let timestamp = reader.i64()?;
            if version == 0 {
                let _max_offsets = reader.i32()?;
            }
            let selected = coordinator.list_offset(project, &topic, partition, timestamp);
            response.i32(partition);
            match selected {
                Ok(Some((selected_offset, selected_timestamp))) => {
                    response.i16(NONE);
                    if version == 0 {
                        response.array_len(1);
                        response.i64(
                            i64::try_from(selected_offset).map_err(|_| {
                                Error::internal("Kafka selected offset exceeds i64")
                            })?,
                        );
                    } else {
                        response.i64(selected_timestamp);
                        response.i64(
                            i64::try_from(selected_offset).map_err(|_| {
                                Error::internal("Kafka selected offset exceeds i64")
                            })?,
                        );
                    }
                }
                Ok(None) => {
                    response.i16(NONE);
                    if version == 0 {
                        response.array_len(0);
                    } else {
                        response.i64(-1);
                        response.i64(-1);
                    }
                }
                Err(_) => {
                    response.i16(UNKNOWN_TOPIC_OR_PARTITION);
                    if version == 0 {
                        response.array_len(0);
                    } else {
                        response.i64(-1);
                        response.i64(-1);
                    }
                }
            }
        }
    }
    Ok(())
}

fn find_coordinator(
    endpoint: &KafkaEndpoint,
    reader: &mut Reader<'_>,
    response: &mut Writer,
) -> Result<()> {
    let _group = reader.string()?;
    response.i16(NONE);
    response.i32(0);
    response.string(&endpoint.host)?;
    response.i32(i32::from(endpoint.port));
    Ok(())
}

fn join_group(
    project: ProjectId,
    coordinator: &Arc<dyn BrokerCoordinator>,
    reader: &mut Reader<'_>,
    response: &mut Writer,
) -> Result<()> {
    let group = reader.string()?.to_owned();
    let session_timeout = reader.i32()?;
    if session_timeout <= 0 {
        return Err(kafka_protocol_error(
            "Kafka group session timeout must be positive",
        ));
    }
    let mut member = reader.string()?.to_owned();
    let protocol_type = reader.string()?.to_owned();
    let protocols = reader.array_count()?;
    let mut member_protocols = BTreeMap::new();
    for _ in 0..protocols {
        let name = reader.string()?.to_owned();
        let value = reader.bytes()?.unwrap_or_default().to_vec();
        if member_protocols.insert(name, value).is_some() {
            return Err(kafka_protocol_error(
                "Kafka group member repeated an assignment protocol",
            ));
        }
    }
    if protocol_type != "consumer" {
        return Err(kafka_protocol_error(
            "Kafka group protocol type must be consumer",
        ));
    }
    if member.is_empty() {
        member = format!("member-{}", uuid::Uuid::new_v4());
    }
    let reply = coordinator.submit(
        BrokerCommand::JoinGroup {
            project,
            group,
            member: member.clone(),
            session_timeout_ms: u32::try_from(session_timeout)
                .map_err(|_| kafka_protocol_error("Kafka session timeout exceeds u32"))?,
            resolved_time_ms: 0,
            protocols: member_protocols,
        },
        CommitAcknowledgement::Published,
    );
    let reply = match reply {
        Ok(reply) => reply,
        Err(error) => {
            response.i16(group_error_code(&error));
            response.i32(-1);
            response.string("")?;
            response.string("")?;
            response.string(&member)?;
            response.array_len(0);
            return Ok(());
        }
    };
    let BrokerReply::Group {
        generation,
        leader,
        members,
        protocol,
        metadata,
    } = reply.reply
    else {
        return Err(Error::internal("join group returned wrong reply"));
    };
    response.i16(NONE);
    response.i32(generation);
    response.string(&protocol)?;
    response.string(&leader)?;
    response.string(&member)?;
    response.array_len(members.len());
    for item in members {
        response.string(&item)?;
        response.bytes(Some(metadata.get(&item).map_or(&[], Vec::as_slice)))?;
    }
    Ok(())
}

fn sync_group(
    project: ProjectId,
    coordinator: &Arc<dyn BrokerCoordinator>,
    reader: &mut Reader<'_>,
    response: &mut Writer,
) -> Result<()> {
    let group = reader.string()?.to_owned();
    let generation = reader.i32()?;
    let member = reader.string()?.to_owned();
    let count = reader.array_count()?;
    let mut assignments = BTreeMap::new();
    for _ in 0..count {
        assignments.insert(
            reader.string()?.to_owned(),
            reader.bytes()?.unwrap_or_default().to_vec(),
        );
    }
    if !assignments.is_empty()
        && coordinator
            .group_leader(project, &group, generation)?
            .as_deref()
            != Some(member.as_str())
    {
        response.i16(ILLEGAL_GENERATION);
        response.bytes(Some(&[]))?;
        return Ok(());
    }
    if let Err(error) = coordinator.submit(
        BrokerCommand::SyncGroup {
            project,
            group: group.clone(),
            generation,
            assignments,
        },
        CommitAcknowledgement::Published,
    ) {
        response.i16(group_error_code(&error));
        response.bytes(Some(&[]))?;
        return Ok(());
    }
    let assignment = coordinator.group_assignment(project, &group, generation, &member)?;
    response.i16(if assignment.is_some() {
        NONE
    } else {
        ILLEGAL_GENERATION
    });
    response.bytes(Some(assignment.as_deref().unwrap_or_default()))?;
    Ok(())
}

fn heartbeat(
    project: ProjectId,
    coordinator: &Arc<dyn BrokerCoordinator>,
    reader: &mut Reader<'_>,
    response: &mut Writer,
) -> Result<()> {
    let group = reader.string()?.to_owned();
    let generation = reader.i32()?;
    let member = reader.string()?.to_owned();
    let result = coordinator.submit(
        BrokerCommand::HeartbeatGroup {
            project,
            group,
            generation,
            member,
            resolved_time_ms: 0,
        },
        CommitAcknowledgement::Published,
    );
    response.i16(match result {
        Ok(_) => NONE,
        Err(error) if error.code == ErrorCode::TransactionConflict => ILLEGAL_GENERATION,
        Err(_) => UNKNOWN_MEMBER_ID,
    });
    Ok(())
}

fn leave_group(
    project: ProjectId,
    coordinator: &Arc<dyn BrokerCoordinator>,
    reader: &mut Reader<'_>,
    response: &mut Writer,
) -> Result<()> {
    let group = reader.string()?.to_owned();
    let member = reader.string()?.to_owned();
    let result = coordinator.submit(
        BrokerCommand::LeaveGroup {
            project,
            group,
            member,
        },
        CommitAcknowledgement::Published,
    );
    response.i16(result.as_ref().map_or_else(group_error_code, |_| NONE));
    Ok(())
}

fn offset_commit(
    version: i16,
    project: ProjectId,
    coordinator: &Arc<dyn BrokerCoordinator>,
    reader: &mut Reader<'_>,
    response: &mut Writer,
) -> Result<()> {
    let group = reader.string()?.to_owned();
    let (generation, member) = if version >= 1 {
        (Some(reader.i32()?), Some(reader.string()?.to_owned()))
    } else {
        (None, None)
    };
    let topics = reader.array_count()?;
    response.array_len(topics);
    for _ in 0..topics {
        let topic = reader.string()?.to_owned();
        response.string(&topic)?;
        let partitions = reader.array_count()?;
        response.array_len(partitions);
        for _ in 0..partitions {
            let partition = reader.i32()?;
            let offset = reader.i64()?;
            let _metadata = reader.nullable_string()?;
            let result = if offset < 0 {
                Err(Error::invalid_data("negative committed offset"))
            } else {
                coordinator.submit(
                    BrokerCommand::CommitOffset {
                        project,
                        group: group.clone(),
                        topic: topic.clone(),
                        partition,
                        offset: offset as u64,
                        generation,
                        member: member.clone(),
                    },
                    CommitAcknowledgement::Published,
                )
            };
            response.i32(partition);
            response.i16(match result {
                Ok(_) => NONE,
                Err(error) if error.code == ErrorCode::TransactionConflict => ILLEGAL_GENERATION,
                Err(error) => admin_error_code(&error),
            });
        }
    }
    Ok(())
}

fn offset_fetch(
    project: ProjectId,
    coordinator: &Arc<dyn BrokerCoordinator>,
    reader: &mut Reader<'_>,
    response: &mut Writer,
) -> Result<()> {
    let group = reader.string()?.to_owned();
    let topics = reader.array_count()?;
    response.array_len(topics);
    for _ in 0..topics {
        let topic = reader.string()?.to_owned();
        response.string(&topic)?;
        let partitions = reader.array_count()?;
        response.array_len(partitions);
        for _ in 0..partitions {
            let partition = reader.i32()?;
            let offset = coordinator.committed_offset(project, &group, &topic, partition)?;
            response.i32(partition);
            response.i64(offset.map_or(-1, |value| value as i64));
            response.nullable_string(None)?;
            response.i16(NONE);
        }
    }
    Ok(())
}

struct Message {
    create_time_ms: Option<i64>,
    key: Option<Vec<u8>>,
    value: Option<Vec<u8>>,
    /// Only the v2 record format carries headers; a legacy message set decodes to none.
    headers: Vec<(String, Option<Vec<u8>>)>,
}

/// Decodes the v2 `RecordBatch` sequence a Produce v3 partition carries.
///
/// One partition may carry several batches back to back, so the sequence is drained rather than a
/// single batch parsed. Every length in the batch is attacker controlled, so each is checked
/// against the bytes actually present before it is used to bound any loop or allocation.
fn decode_record_batches(bytes: &[u8]) -> Result<Vec<Message>> {
    let mut reader = Reader::new(bytes);
    let mut messages = Vec::new();
    while reader.remaining() > 0 {
        let _base_offset = reader.i64()?;
        let length = reader.i32()?;
        let length = usize::try_from(length)
            .map_err(|_| kafka_protocol_error("Kafka record batch length is negative"))?;
        if length < RECORD_BATCH_BODY_MINIMUM || length > reader.remaining() {
            return Err(kafka_protocol_error("Kafka record batch length is invalid"));
        }
        decode_record_batch(reader.take(length)?, &mut messages)?;
        if messages.len() > MAXIMUM_RECORDS {
            return Err(kafka_protocol_error(
                "Kafka record batch contains too many records",
            ));
        }
    }
    reader.finish()?;
    Ok(messages)
}

fn decode_record_batch(batch: &[u8], messages: &mut Vec<Message>) -> Result<()> {
    let mut reader = Reader::new(batch);
    let _partition_leader_epoch = reader.i32()?;
    let magic = reader.i8()?;
    if magic != RECORD_BATCH_MAGIC {
        return Err(kafka_protocol_error(
            "Kafka produce v3 requires record batch magic 2",
        ));
    }
    let expected_crc = reader.u32()?;
    // v2 is checksummed with CRC-32C (Castagnoli) over everything after the crc field, where v0
    // and v1 messages use CRC-32 (IEEE). The wrong polynomial would accept corrupt batches.
    let covered = batch
        .get(RECORD_BATCH_CRC_END..)
        .ok_or_else(|| kafka_protocol_error("Kafka record batch header is truncated"))?;
    if crc32c::crc32c(covered) != expected_crc {
        return Err(kafka_protocol_error("Kafka record batch CRC mismatch"));
    }
    let attributes = reader.i16()?;
    if attributes & !0x7f != 0 {
        return Err(kafka_protocol_error(
            "Kafka record batch attributes use reserved bits",
        ));
    }
    if attributes & 0x10 != 0 {
        return Err(kafka_protocol_error(
            "Kafka transactional record batches are not supported",
        ));
    }
    if attributes & 0x20 != 0 {
        // Control batches mark transaction boundaries and are broker-authored; a producer that
        // sends one is either confused or probing.
        return Err(kafka_protocol_error(
            "Kafka control record batches cannot be produced",
        ));
    }
    let codec = u8::try_from(attributes & 0x07)
        .map_err(|_| Error::internal("Kafka record batch codec exceeds u8"))?;
    let last_offset_delta = reader.i32()?;
    let base_timestamp = reader.i64()?;
    let _maximum_timestamp = reader.i64()?;
    let _producer_id = reader.i64()?;
    let _producer_epoch = reader.i16()?;
    let _base_sequence = reader.i32()?;
    let record_count = reader.i32()?;
    let record_count = usize::try_from(record_count)
        .map_err(|_| kafka_protocol_error("Kafka record count is negative"))?;
    if record_count > 0
        && i64::from(last_offset_delta) != i64::try_from(record_count).unwrap_or(i64::MAX) - 1
    {
        // The delta is one less than the count, not the count. A batch that disagrees would have
        // its offsets assigned inconsistently by any broker that trusted either field.
        return Err(kafka_protocol_error(
            "Kafka record batch last offset delta does not match its record count",
        ));
    }
    let remaining = reader.remaining();
    let compressed = reader.take(remaining)?;
    let records = if codec == 0 {
        Cow::Borrowed(compressed)
    } else {
        Cow::Owned(decompress_message_value(codec, compressed)?)
    };
    // The count is only trusted once it could physically fit, so a lying count cannot drive a
    // large allocation or a long loop before the first truncated read fails.
    if record_count > MAXIMUM_RECORDS || record_count > records.len() / RECORD_MINIMUM_BYTES {
        return Err(kafka_protocol_error(
            "Kafka record count exceeds the batch payload",
        ));
    }
    let mut records = Reader::new(&records);
    for _ in 0..record_count {
        messages.push(decode_record(&mut records, base_timestamp)?);
    }
    records.finish()
}

fn decode_record(reader: &mut Reader<'_>, base_timestamp: i64) -> Result<Message> {
    let length = reader.signed_varint()?;
    let length = usize::try_from(length)
        .map_err(|_| kafka_protocol_error("Kafka record length is negative"))?;
    if length > reader.remaining() {
        return Err(kafka_protocol_error(
            "Kafka record length exceeds its batch",
        ));
    }
    let mut record = Reader::new(reader.take(length)?);
    let _attributes = record.i8()?;
    let timestamp_delta = record.signed_varlong()?;
    let _offset_delta = record.signed_varint()?;
    let key = record.varint_bytes()?.map(ToOwned::to_owned);
    let value = record.varint_bytes()?.map(ToOwned::to_owned);
    let header_count = record.signed_varint()?;
    let header_count = usize::try_from(header_count)
        .map_err(|_| kafka_protocol_error("Kafka record header count is negative"))?;
    if header_count > record.remaining() / RECORD_HEADER_MINIMUM_BYTES {
        return Err(kafka_protocol_error(
            "Kafka record header count exceeds the record payload",
        ));
    }
    // Grown as headers actually decode rather than reserved from the declared count: the count is
    // only bounded by the record bytes, and reserving from it would let a small record ask for a
    // large allocation.
    let mut headers = Vec::new();
    for _ in 0..header_count {
        let name = record.varint_string()?.to_owned();
        headers.push((name, record.varint_bytes()?.map(ToOwned::to_owned)));
    }
    record.finish()?;
    Ok(Message {
        // A base timestamp of -1 marks a batch whose producer supplied no create time, so the
        // record inherits the broker's append time rather than a negative one.
        create_time_ms: (base_timestamp >= 0)
            .then(|| base_timestamp.saturating_add(timestamp_delta)),
        key,
        value,
        headers,
    })
}

/// Reports whether any batch in the sequence is compressed, without decoding the records.
///
/// The produce admission path needs this before it reserves a decode workspace, so it walks only
/// the batch headers.
fn record_batches_use_compression(bytes: &[u8]) -> Result<bool> {
    let mut reader = Reader::new(bytes);
    let mut compressed = false;
    while reader.remaining() > 0 {
        let _base_offset = reader.i64()?;
        let length = reader.i32()?;
        let length = usize::try_from(length)
            .map_err(|_| kafka_protocol_error("Kafka record batch length is negative"))?;
        if length < RECORD_BATCH_BODY_MINIMUM || length > reader.remaining() {
            return Err(kafka_protocol_error("Kafka record batch length is invalid"));
        }
        let batch = reader.take(length)?;
        let mut header = Reader::new(batch);
        let _partition_leader_epoch = header.i32()?;
        if header.i8()? != RECORD_BATCH_MAGIC {
            return Err(kafka_protocol_error(
                "Kafka produce v3 requires record batch magic 2",
            ));
        }
        let _crc = header.u32()?;
        compressed |= header.i16()? & 0x07 != 0;
    }
    reader.finish()?;
    Ok(compressed)
}

fn decode_message_set(bytes: &[u8], request_version: i16) -> Result<Vec<Message>> {
    let _ = request_version;
    decode_message_set_inner(bytes, 0)
}

fn decode_message_set_inner(bytes: &[u8], depth: usize) -> Result<Vec<Message>> {
    // Kafka message-set compression is one envelope around an ordinary message set. Rejecting a
    // second compressed envelope prevents nested decompression from multiplying the bounded
    // frame workspace while preserving the standard producer wire shape.
    const MAXIMUM_COMPRESSION_DEPTH: usize = 1;
    const MAXIMUM_MESSAGES: usize = 1_000_000;
    if depth > MAXIMUM_COMPRESSION_DEPTH {
        return Err(kafka_protocol_error(
            "Kafka message compression nesting is excessive",
        ));
    }
    let mut reader = Reader::new(bytes);
    let mut messages = Vec::new();
    while reader.remaining() >= 12 {
        let _offset = reader.i64()?;
        let size = reader.i32()?;
        if size < 6 || size as usize > reader.remaining() {
            return Err(Error::new(
                crate::ErrorCode::ProtocolViolation,
                "Kafka message size is invalid",
            ));
        }
        let message_bytes = reader.take(size as usize)?;
        let mut message = Reader::new(message_bytes);
        let expected_crc = message.i32()? as u32;
        let body = &message_bytes[4..];
        let actual_crc = crc32fast::hash(body);
        if expected_crc != actual_crc {
            return Err(Error::new(
                crate::ErrorCode::ProtocolViolation,
                "Kafka message CRC mismatch",
            ));
        }
        let magic = message.i8()?;
        let attributes = message.i8()? as u8;
        if !(0..=1).contains(&magic)
            || (magic == 0 && attributes & !0x07 != 0)
            || (magic == 1 && attributes & !0x0f != 0)
        {
            return Err(kafka_protocol_error(
                "Kafka message magic or attributes are invalid",
            ));
        }
        let create_time_ms = (magic == 1).then(|| message.i64()).transpose()?;
        let key = message.bytes()?.map(ToOwned::to_owned);
        let value = message.bytes()?.map(ToOwned::to_owned);
        message.finish()?;
        let compression = attributes & 0x07;
        if compression == 0 {
            messages.push(Message {
                create_time_ms,
                key,
                value,
                headers: Vec::new(),
            });
        } else {
            if depth >= MAXIMUM_COMPRESSION_DEPTH {
                return Err(kafka_protocol_error(
                    "Kafka message compression nesting is excessive",
                ));
            }
            let compressed = value.as_deref().ok_or_else(|| {
                kafka_protocol_error("Kafka compressed message value cannot be null")
            })?;
            let decompressed = decompress_message_value(compression, compressed)?;
            let mut inner = decode_message_set_inner(&decompressed, depth.saturating_add(1))?;
            if attributes & 0x08 != 0 {
                for message in &mut inner {
                    message.create_time_ms = create_time_ms;
                }
            }
            messages.extend(inner);
        }
        if messages.len() > MAXIMUM_MESSAGES {
            return Err(kafka_protocol_error(
                "Kafka message set contains too many records",
            ));
        }
    }
    reader.finish()?;
    Ok(messages)
}

fn decompress_message_value(codec: u8, bytes: &[u8]) -> Result<Vec<u8>> {
    match codec {
        1 => read_bounded_decompressed(flate2::read::GzDecoder::new(bytes)),
        2 => decompress_snappy(bytes),
        3 => read_bounded_decompressed(lz4_flex::frame::FrameDecoder::new(bytes)),
        4 => {
            let decoder = zstd::stream::read::Decoder::new(bytes)
                .map_err(|_| kafka_protocol_error("Kafka zstd payload is invalid"))?;
            read_bounded_decompressed(decoder)
        }
        _ => Err(kafka_protocol_error(
            "Kafka message compression codec is invalid",
        )),
    }
}

fn read_bounded_decompressed(mut reader: impl std::io::Read) -> Result<Vec<u8>> {
    let maximum =
        u64::try_from(MAX_FRAME).map_err(|_| Error::internal("Kafka frame bound exceeds u64"))?;
    let mut output = Vec::new();
    reader
        .by_ref()
        .take(maximum.saturating_add(1))
        .read_to_end(&mut output)
        .map_err(|_| kafka_protocol_error("Kafka compressed payload is invalid"))?;
    if output.len() > MAX_FRAME {
        return Err(kafka_protocol_error(
            "Kafka decompressed message set exceeds the frame limit",
        ));
    }
    Ok(output)
}

fn decompress_snappy(bytes: &[u8]) -> Result<Vec<u8>> {
    const XERIAL_HEADER: &[u8; 8] = b"\x82SNAPPY\0";
    if !bytes.starts_with(XERIAL_HEADER) {
        return decompress_snappy_block(bytes, MAX_FRAME);
    }
    if bytes.len() < 16 {
        return Err(kafka_protocol_error(
            "Kafka xerial snappy header is truncated",
        ));
    }
    if bytes[8..12] != 1_i32.to_be_bytes() || bytes[12..16] != 1_i32.to_be_bytes() {
        return Err(kafka_protocol_error(
            "Kafka xerial snappy stream version is invalid",
        ));
    }
    let mut cursor = 16_usize;
    let mut output = Vec::new();
    while cursor < bytes.len() {
        let length_bytes = bytes
            .get(cursor..cursor.saturating_add(4))
            .ok_or_else(|| kafka_protocol_error("Kafka xerial snappy chunk is truncated"))?;
        let length = i32::from_be_bytes(
            length_bytes
                .try_into()
                .map_err(|_| Error::internal("snappy chunk length slice"))?,
        );
        if length < 0 {
            return Err(kafka_protocol_error(
                "Kafka xerial snappy chunk length is invalid",
            ));
        }
        cursor = cursor.saturating_add(4);
        let end = cursor
            .checked_add(length as usize)
            .ok_or_else(|| kafka_protocol_error("Kafka xerial snappy chunk length overflow"))?;
        let chunk = bytes
            .get(cursor..end)
            .ok_or_else(|| kafka_protocol_error("Kafka xerial snappy chunk is truncated"))?;
        let decompressed = decompress_snappy_block(chunk, MAX_FRAME - output.len())?;
        output = append_bounded(output, &decompressed)?;
        cursor = end;
    }
    Ok(output)
}

fn decompress_snappy_block(bytes: &[u8], maximum: usize) -> Result<Vec<u8>> {
    let decompressed_len = snap::raw::decompress_len(bytes)
        .map_err(|_| kafka_protocol_error("Kafka snappy payload is invalid"))?;
    if decompressed_len > maximum {
        return Err(kafka_protocol_error(
            "Kafka decompressed message set exceeds the frame limit",
        ));
    }
    snap::raw::Decoder::new()
        .decompress_vec(bytes)
        .map_err(|_| kafka_protocol_error("Kafka snappy payload is invalid"))
}

fn append_bounded(mut output: Vec<u8>, bytes: &[u8]) -> Result<Vec<u8>> {
    let total = output
        .len()
        .checked_add(bytes.len())
        .ok_or_else(|| kafka_protocol_error("Kafka decompressed size overflow"))?;
    if total > MAX_FRAME {
        return Err(kafka_protocol_error(
            "Kafka decompressed message set exceeds the frame limit",
        ));
    }
    output.extend_from_slice(bytes);
    Ok(output)
}

fn kafka_protocol_error(message: &'static str) -> Error {
    Error::new(crate::ErrorCode::ProtocolViolation, message)
}

const fn produce_error_code(error: &Error) -> i16 {
    match error.code {
        ErrorCode::DeadlineExceeded => REQUEST_TIMED_OUT,
        ErrorCode::Backpressure
        | ErrorCode::WriteAdmissionFull
        | ErrorCode::ResultBudgetExceeded => MESSAGE_TOO_LARGE,
        ErrorCode::ProjectNotFound | ErrorCode::ProjectFenced => UNKNOWN_TOPIC_OR_PARTITION,
        ErrorCode::InvalidData | ErrorCode::ProtocolViolation => INVALID_REQUEST,
        _ => UNKNOWN_SERVER_ERROR,
    }
}

fn log_produce_failure(stage: &'static str, api_version: i16, partition: i32, error: &Error) {
    // Do not record topic names or any key, header, value, credential, or request bytes.
    tracing::warn!(stage, api_version, partition, category = ?error.code,
        kafka_code = produce_error_code(error), reason = %error.message,
        "Kafka produce rejected");
}

const fn admin_error_code(error: &Error) -> i16 {
    match error.code {
        ErrorCode::DeadlineExceeded => REQUEST_TIMED_OUT,
        ErrorCode::Backpressure | ErrorCode::WriteAdmissionFull => NOT_ENOUGH_REPLICAS,
        ErrorCode::ProjectNotFound | ErrorCode::ProjectFenced => UNKNOWN_TOPIC_OR_PARTITION,
        _ => INVALID_REQUEST,
    }
}

const fn group_error_code(error: &Error) -> i16 {
    match error.code {
        ErrorCode::TransactionConflict => ILLEGAL_GENERATION,
        ErrorCode::InvalidData | ErrorCode::ProtocolViolation => INCONSISTENT_GROUP_PROTOCOL,
        ErrorCode::DeadlineExceeded => REQUEST_TIMED_OUT,
        ErrorCode::Backpressure | ErrorCode::WriteAdmissionFull => NOT_ENOUGH_REPLICAS,
        _ => UNKNOWN_MEMBER_ID,
    }
}

/// One stored record projected onto the Kafka record model, whatever surface published it.
struct RecordView<'a> {
    create_time_ms: i64,
    key: Option<&'a [u8]>,
    value: Option<&'a [u8]>,
    headers: Vec<(Cow<'a, str>, Option<&'a [u8]>)>,
}

/// Projects a stored record onto the Kafka record model.
///
/// A record published through the Queue surface has to arrive at a Stream consumer as a usable
/// Kafka record rather than a bare payload, so its AMQP routing key becomes the record key and its
/// AMQP headers become record headers. Headers only exist from the v2 record format onwards, so
/// `with_headers` is false when the caller is about to emit a legacy message; the key maps either
/// way because a v0 message already has one.
fn record_view(payload: &super::engine::PayloadRecord, with_headers: bool) -> RecordView<'_> {
    let (create_time_ms, key, headers) = match &payload.ingress {
        IngressMetadata::Kafka {
            create_time_ms,
            key,
            headers,
            ..
        } => (
            create_time_ms.unwrap_or(payload.resolved_time_ms),
            key.as_deref(),
            if with_headers {
                headers
                    .iter()
                    .map(|(name, value)| (Cow::Borrowed(name.as_str()), Some(value.as_slice())))
                    .collect()
            } else {
                Vec::new()
            },
        ),
        IngressMetadata::Amqp {
            routing_key,
            properties,
            headers,
            ..
        } => (
            payload.resolved_time_ms,
            // An empty routing key is the absence of one, which is a null Kafka key rather than a
            // present empty one.
            (!routing_key.is_empty()).then_some(routing_key.as_bytes()),
            if with_headers {
                headers
                    .iter()
                    .map(|(name, value)| (Cow::Borrowed(name.as_str()), Some(value.as_slice())))
                    .chain(properties.iter().map(|(name, value)| {
                        (
                            Cow::Owned(format!("{AMQP_PROPERTY_HEADER_PREFIX}{name}")),
                            Some(value.as_slice()),
                        )
                    }))
                    .collect()
            } else {
                Vec::new()
            },
        ),
    };
    RecordView {
        create_time_ms,
        key,
        value: match &payload.ingress {
            IngressMetadata::Kafka { value_is_null, .. } if *value_is_null => None,
            _ => Some(payload.payload.as_ref()),
        },
        headers,
    }
}

fn encode_message(
    output: &mut Writer,
    offset: i64,
    payload: &super::engine::PayloadRecord,
    version: i16,
) -> Result<()> {
    if version >= FETCH_RECORD_BATCH_VERSION {
        return encode_record_batch(output, offset, &record_view(payload, true));
    }
    let view = record_view(payload, false);
    let magic = if version >= 1 { 1 } else { 0 };
    let create_time_ms = (magic == 1).then_some(view.create_time_ms);
    let key = view.key;
    let value = view.value;
    let total_length = encoded_message_length(payload, version)?;
    let message_length = total_length
        .checked_sub(12)
        .ok_or_else(|| Error::internal("Kafka message envelope length underflow"))?;

    let mut crc = crc32fast::Hasher::new();
    crc.update(&[magic as u8, 0]);
    if let Some(create_time_ms) = create_time_ms {
        crc.update(&create_time_ms.to_be_bytes());
    }
    update_crc_bytes(&mut crc, key)?;
    update_crc_bytes(&mut crc, value)?;

    output.i64(offset);
    output.i32(
        i32::try_from(message_length).map_err(|_| Error::internal("Kafka message exceeds i32"))?,
    );
    output.i32(crc.finalize() as i32);
    output.i8(magic);
    output.i8(0);
    if let Some(create_time_ms) = create_time_ms {
        output.i64(create_time_ms);
    }
    output.bytes(key)?;
    output.bytes(value)?;
    Ok(())
}

fn update_crc_bytes(crc: &mut crc32fast::Hasher, value: Option<&[u8]>) -> Result<()> {
    let length = match value {
        Some(bytes) => {
            i32::try_from(bytes.len()).map_err(|_| Error::internal("Kafka bytes exceed i32"))?
        }
        None => -1,
    };
    crc.update(&length.to_be_bytes());
    if let Some(value) = value {
        crc.update(value);
    }
    Ok(())
}

fn encoded_message_length(payload: &super::engine::PayloadRecord, version: i16) -> Result<usize> {
    if version >= FETCH_RECORD_BATCH_VERSION {
        return record_batch_length(&record_view(payload, true));
    }
    let view = record_view(payload, false);
    let fixed = if version >= 1 { 34_usize } else { 26_usize };
    fixed
        .checked_add(view.key.map_or(0, <[u8]>::len))
        .and_then(|bytes| bytes.checked_add(view.value.map_or(0, <[u8]>::len)))
        .ok_or_else(|| Error::internal("Kafka encoded message length overflow"))
}

/// Emits one record as a self-contained v2 `RecordBatch`.
///
/// Records are stored individually rather than as the batches a producer sent, so a fetch has to
/// build batches. One batch per record keeps the caller's incremental byte budget exact — the
/// response can stop at any record without ever truncating a batch a consumer would then reject —
/// at the cost of a fixed 61-byte header per record.
fn encode_record_batch(output: &mut Writer, offset: i64, view: &RecordView<'_>) -> Result<()> {
    let body_length = record_body_length(view)?;
    output.i64(offset);
    let length_position = output.len();
    output.i32(0);
    output.i32(-1);
    output.i8(RECORD_BATCH_MAGIC);
    let crc_position = output.len();
    output.i32(0);
    let crc_start = output.len();
    // Uncompressed, create-time timestamps, neither transactional nor a control batch.
    output.i16(0);
    // lastOffsetDelta is the delta of the final record, so one record makes it zero rather than
    // the record count.
    output.i32(0);
    output.i64(view.create_time_ms);
    output.i64(view.create_time_ms);
    // No idempotent producer state is tracked, which the protocol spells as an absent producer.
    output.i64(-1);
    output.i16(-1);
    output.i32(-1);
    output.i32(1);
    output.signed_varint(
        i32::try_from(body_length).map_err(|_| Error::internal("Kafka record exceeds i32"))?,
    );
    output.i8(0);
    output.signed_varlong(0);
    output.signed_varint(0);
    output.varint_bytes(view.key)?;
    output.varint_bytes(view.value)?;
    output.signed_varint(
        i32::try_from(view.headers.len())
            .map_err(|_| Error::internal("Kafka record header count exceeds i32"))?,
    );
    for (name, value) in &view.headers {
        output.varint_string(name)?;
        output.varint_bytes(*value)?;
    }
    let Some(crc) = output.bytes_from(crc_start).map(crc32c::crc32c) else {
        // The bounded writer overflowed, so the response is discarded before it is framed and the
        // placeholders it never wrote must not be patched.
        return Ok(());
    };
    let batch_length = output
        .len()
        .checked_sub(length_position.saturating_add(std::mem::size_of::<i32>()))
        .ok_or_else(|| Error::internal("Kafka record batch length underflow"))?;
    output.patch_i32(
        length_position,
        i32::try_from(batch_length)
            .map_err(|_| Error::internal("Kafka record batch exceeds i32"))?,
    )?;
    output.patch_i32(crc_position, i32::from_be_bytes(crc.to_be_bytes()))
}

fn record_batch_length(view: &RecordView<'_>) -> Result<usize> {
    let body_length = record_body_length(view)?;
    RECORD_BATCH_HEADER_BYTES
        .checked_add(signed_varint_length(
            i32::try_from(body_length).map_err(|_| Error::internal("Kafka record exceeds i32"))?,
        ))
        .and_then(|bytes| bytes.checked_add(body_length))
        .ok_or_else(|| Error::internal("Kafka record batch length overflow"))
}

/// Bytes of one record after its own length varint.
fn record_body_length(view: &RecordView<'_>) -> Result<usize> {
    // Attributes, plus the zero timestamp and offset deltas a single-record batch always has.
    let mut length = 3_usize;
    for value in [view.key, view.value] {
        length = length
            .checked_add(varint_bytes_length(value)?)
            .ok_or_else(|| Error::internal("Kafka record length overflow"))?;
    }
    length = length
        .checked_add(signed_varint_length(
            i32::try_from(view.headers.len())
                .map_err(|_| Error::internal("Kafka record header count exceeds i32"))?,
        ))
        .ok_or_else(|| Error::internal("Kafka record length overflow"))?;
    for (name, value) in &view.headers {
        let header = varint_bytes_length(Some(name.as_bytes()))?
            .checked_add(varint_bytes_length(*value)?)
            .ok_or_else(|| Error::internal("Kafka record length overflow"))?;
        length = length
            .checked_add(header)
            .ok_or_else(|| Error::internal("Kafka record length overflow"))?;
    }
    Ok(length)
}

fn varint_bytes_length(value: Option<&[u8]>) -> Result<usize> {
    let Some(bytes) = value else {
        return Ok(signed_varint_length(-1));
    };
    let length =
        i32::try_from(bytes.len()).map_err(|_| Error::internal("Kafka record bytes exceed i32"))?;
    signed_varint_length(length)
        .checked_add(bytes.len())
        .ok_or_else(|| Error::internal("Kafka record length overflow"))
}

/// Zigzag maps a signed value onto an unsigned varint so small magnitudes stay one byte.
const fn zigzag_encode_i32(value: i32) -> u32 {
    let doubled = value.unsigned_abs() << 1;
    if value < 0 {
        doubled.wrapping_sub(1)
    } else {
        doubled
    }
}

const fn zigzag_encode_i64(value: i64) -> u64 {
    let doubled = value.unsigned_abs().wrapping_mul(2);
    if value < 0 {
        doubled.wrapping_sub(1)
    } else {
        doubled
    }
}

fn zigzag_decode_i32(value: u32) -> Result<i32> {
    let magnitude = i32::try_from(value >> 1)
        .map_err(|_| Error::internal("Kafka zigzag magnitude exceeds i32"))?;
    Ok(if value & 1 == 0 {
        magnitude
    } else {
        (-1_i32).wrapping_sub(magnitude)
    })
}

fn zigzag_decode_i64(value: u64) -> Result<i64> {
    let magnitude = i64::try_from(value >> 1)
        .map_err(|_| Error::internal("Kafka zigzag magnitude exceeds i64"))?;
    Ok(if value & 1 == 0 {
        magnitude
    } else {
        (-1_i64).wrapping_sub(magnitude)
    })
}

const fn unsigned_varint_length(value: u32) -> usize {
    match u32::BITS - value.leading_zeros() {
        0..=7 => 1,
        8..=14 => 2,
        15..=21 => 3,
        22..=28 => 4,
        _ => 5,
    }
}

const fn signed_varint_length(value: i32) -> usize {
    unsigned_varint_length(zigzag_encode_i32(value))
}

fn group_partition_replies(
    replies: Vec<(String, i32, i16, i64)>,
) -> BTreeMap<String, Vec<(i32, i16, i64)>> {
    let mut grouped = BTreeMap::new();
    for (topic, partition, error, offset) in replies {
        grouped
            .entry(topic)
            .or_insert_with(Vec::new)
            .push((partition, error, offset));
    }
    grouped
}

#[derive(Clone, Copy)]
struct Reader<'a> {
    input: &'a [u8],
    offset: usize,
}

impl<'a> Reader<'a> {
    const fn new(input: &'a [u8]) -> Self {
        Self { input, offset: 0 }
    }
    fn remaining(&self) -> usize {
        self.input.len().saturating_sub(self.offset)
    }
    fn finish(&self) -> Result<()> {
        if self.offset == self.input.len() {
            Ok(())
        } else {
            Err(Error::new(
                crate::ErrorCode::ProtocolViolation,
                "Kafka request has trailing bytes",
            ))
        }
    }
    fn take(&mut self, length: usize) -> Result<&'a [u8]> {
        let end = self
            .offset
            .checked_add(length)
            .ok_or_else(|| Error::invalid_data("Kafka length overflow"))?;
        let value = self.input.get(self.offset..end).ok_or_else(|| {
            Error::new(
                crate::ErrorCode::ProtocolViolation,
                "Kafka request is truncated",
            )
        })?;
        self.offset = end;
        Ok(value)
    }
    fn i8(&mut self) -> Result<i8> {
        Ok(self.take(1)?[0] as i8)
    }
    fn i16(&mut self) -> Result<i16> {
        Ok(i16::from_be_bytes(
            self.take(2)?
                .try_into()
                .map_err(|_| Error::internal("i16 slice"))?,
        ))
    }
    fn i32(&mut self) -> Result<i32> {
        Ok(i32::from_be_bytes(
            self.take(4)?
                .try_into()
                .map_err(|_| Error::internal("i32 slice"))?,
        ))
    }
    fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_be_bytes(
            self.take(8)?
                .try_into()
                .map_err(|_| Error::internal("i64 slice"))?,
        ))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_be_bytes(
            self.take(4)?
                .try_into()
                .map_err(|_| Error::internal("u32 slice"))?,
        ))
    }
    fn unsigned_varint(&mut self) -> Result<u32> {
        let mut value = 0_u32;
        for shift in (0..35).step_by(7) {
            let byte = self.take(1)?[0];
            if shift == 28 && byte & 0xf0 != 0 {
                return Err(kafka_protocol_error("Kafka unsigned varint overflows u32"));
            }
            value |= u32::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(kafka_protocol_error("Kafka unsigned varint is too long"))
    }
    fn unsigned_varlong(&mut self) -> Result<u64> {
        let mut value = 0_u64;
        for shift in (0..70).step_by(7) {
            let byte = self.take(1)?[0];
            if shift == 63 && byte & 0xfe != 0 {
                return Err(kafka_protocol_error("Kafka unsigned varlong overflows u64"));
            }
            value |= u64::from(byte & 0x7f) << shift;
            if byte & 0x80 == 0 {
                return Ok(value);
            }
        }
        Err(kafka_protocol_error("Kafka unsigned varlong is too long"))
    }
    /// Record fields are zigzag-signed varints, unlike the unsigned varints flexible versions use
    /// for tagged fields and compact lengths.
    fn signed_varint(&mut self) -> Result<i32> {
        zigzag_decode_i32(self.unsigned_varint()?)
    }
    fn signed_varlong(&mut self) -> Result<i64> {
        zigzag_decode_i64(self.unsigned_varlong()?)
    }
    fn varint_bytes(&mut self) -> Result<Option<&'a [u8]>> {
        let length = self.signed_varint()?;
        if length == -1 {
            return Ok(None);
        }
        let length = usize::try_from(length)
            .map_err(|_| kafka_protocol_error("Kafka record byte length is invalid"))?;
        self.take(length).map(Some)
    }
    fn varint_string(&mut self) -> Result<&'a str> {
        let bytes = self
            .varint_bytes()?
            .ok_or_else(|| kafka_protocol_error("Kafka record header name is null"))?;
        std::str::from_utf8(bytes)
            .map_err(|_| kafka_protocol_error("Kafka record header name is not UTF-8"))
    }
    fn compact_string(&mut self) -> Result<&'a str> {
        let encoded = self.unsigned_varint()?;
        if encoded == 0 {
            return Err(kafka_protocol_error("Kafka compact string is null"));
        }
        let length = usize::try_from(encoded - 1)
            .map_err(|_| kafka_protocol_error("Kafka compact string length exceeds usize"))?;
        std::str::from_utf8(self.take(length)?).map_err(|_| {
            Error::new(
                crate::ErrorCode::ProtocolViolation,
                "Kafka compact string is not UTF-8",
            )
        })
    }
    fn tagged_fields(&mut self) -> Result<()> {
        let count = self.unsigned_varint()?;
        if count > 10_000 {
            return Err(kafka_protocol_error(
                "Kafka tagged-field count is excessive",
            ));
        }
        let mut previous = None;
        for _ in 0..count {
            let tag = self.unsigned_varint()?;
            if previous.is_some_and(|previous| tag <= previous) {
                return Err(kafka_protocol_error(
                    "Kafka tagged fields are not strictly ordered",
                ));
            }
            let size = usize::try_from(self.unsigned_varint()?)
                .map_err(|_| kafka_protocol_error("Kafka tagged-field size exceeds usize"))?;
            let _ = self.take(size)?;
            previous = Some(tag);
        }
        Ok(())
    }
    fn array_count(&mut self) -> Result<usize> {
        let value = self.i32()?;
        if value < 0 || value as usize > 1_000_000 {
            return Err(Error::new(
                crate::ErrorCode::ProtocolViolation,
                "Kafka array count is invalid",
            ));
        }
        Ok(value as usize)
    }
    fn string(&mut self) -> Result<&'a str> {
        let length = self.i16()?;
        if length < 0 {
            return Err(Error::new(
                crate::ErrorCode::ProtocolViolation,
                "Kafka string is null",
            ));
        }
        std::str::from_utf8(self.take(length as usize)?).map_err(|_| {
            Error::new(
                crate::ErrorCode::ProtocolViolation,
                "Kafka string is not UTF-8",
            )
        })
    }
    fn nullable_string(&mut self) -> Result<Option<&'a str>> {
        let length = self.i16()?;
        if length == -1 {
            return Ok(None);
        }
        if length < -1 {
            return Err(Error::new(
                crate::ErrorCode::ProtocolViolation,
                "Kafka nullable string length is invalid",
            ));
        }
        Ok(Some(
            std::str::from_utf8(self.take(length as usize)?).map_err(|_| {
                Error::new(
                    crate::ErrorCode::ProtocolViolation,
                    "Kafka string is not UTF-8",
                )
            })?,
        ))
    }
    fn nullable_string_array(&mut self) -> Result<Option<Vec<String>>> {
        let count = self.i32()?;
        if count == -1 {
            return Ok(None);
        }
        if count < -1 || count as usize > 1_000_000 {
            return Err(Error::new(
                crate::ErrorCode::ProtocolViolation,
                "Kafka array count is invalid",
            ));
        }
        (0..count)
            .map(|_| self.string().map(str::to_owned))
            .collect::<Result<Vec<_>>>()
            .map(Some)
    }
    fn bytes(&mut self) -> Result<Option<&'a [u8]>> {
        let length = self.i32()?;
        if length == -1 {
            return Ok(None);
        }
        if length < -1 {
            return Err(Error::new(
                crate::ErrorCode::ProtocolViolation,
                "Kafka bytes length is invalid",
            ));
        }
        self.take(length as usize).map(Some)
    }
}

struct Writer {
    bytes: Vec<u8>,
    limit: usize,
    overflowed: bool,
}

impl Writer {
    #[cfg(test)]
    fn new() -> Self {
        Self {
            bytes: Vec::new(),
            limit: usize::MAX,
            overflowed: false,
        }
    }
    fn with_limit(limit: usize) -> Self {
        Self {
            bytes: Vec::new(),
            limit,
            overflowed: false,
        }
    }
    fn len(&self) -> usize {
        self.bytes.len()
    }
    #[cfg(test)]
    fn as_slice(&self) -> &[u8] {
        &self.bytes
    }
    #[cfg(test)]
    fn finish(self) -> Vec<u8> {
        debug_assert!(!self.overflowed, "unbounded Kafka writer overflowed");
        self.bytes
    }
    fn try_finish(self) -> Result<Vec<u8>> {
        if self.overflowed {
            Err(Error::retryable(
                ErrorCode::Backpressure,
                "Kafka response exceeds the bounded response workspace",
                Some(25),
            ))
        } else {
            Ok(self.bytes)
        }
    }
    fn raw(&mut self, bytes: &[u8]) {
        if self.overflowed {
            return;
        }
        let Some(next) = self.bytes.len().checked_add(bytes.len()) else {
            self.overflowed = true;
            return;
        };
        if next > self.limit {
            self.overflowed = true;
            return;
        }
        if next > self.bytes.capacity() {
            let target = if self.limit == usize::MAX {
                self.bytes.capacity().max(1024).saturating_mul(2).max(next)
            } else {
                next.checked_add(WRITER_ALLOCATION_QUANTUM - 1)
                    .map(|rounded| {
                        (rounded / WRITER_ALLOCATION_QUANTUM) * WRITER_ALLOCATION_QUANTUM
                    })
                    .unwrap_or(self.limit)
                    .max(next)
                    .min(self.limit)
            };
            let additional = target.saturating_sub(self.bytes.len());
            if self.bytes.try_reserve_exact(additional).is_err() {
                self.overflowed = true;
                return;
            }
        }
        self.bytes.extend_from_slice(bytes);
    }
    fn i8(&mut self, value: i8) {
        self.raw(&[value as u8]);
    }
    fn i16(&mut self, value: i16) {
        self.raw(&value.to_be_bytes());
    }
    fn i32(&mut self, value: i32) {
        self.raw(&value.to_be_bytes());
    }
    fn i64(&mut self, value: i64) {
        self.raw(&value.to_be_bytes());
    }
    fn unsigned_varint(&mut self, mut value: u32) {
        loop {
            let mut byte = (value & 0x7f) as u8;
            value >>= 7;
            if value != 0 {
                byte |= 0x80;
            }
            self.raw(&[byte]);
            if value == 0 {
                break;
            }
        }
    }
    fn unsigned_varlong(&mut self, mut value: u64) {
        loop {
            let mut byte = u8::try_from(value & 0x7f).unwrap_or(0);
            value >>= 7;
            if value != 0 {
                byte |= 0x80;
            }
            self.raw(&[byte]);
            if value == 0 {
                break;
            }
        }
    }
    /// Record fields are zigzag-signed varints, unlike the unsigned varints flexible versions use
    /// for tagged fields and compact lengths.
    fn signed_varint(&mut self, value: i32) {
        self.unsigned_varint(zigzag_encode_i32(value));
    }
    fn signed_varlong(&mut self, value: i64) {
        self.unsigned_varlong(zigzag_encode_i64(value));
    }
    fn varint_bytes(&mut self, value: Option<&[u8]>) -> Result<()> {
        match value {
            None => self.signed_varint(-1),
            Some(bytes) => {
                self.signed_varint(
                    i32::try_from(bytes.len())
                        .map_err(|_| Error::invalid_data("Kafka record bytes exceed i32"))?,
                );
                self.raw(bytes);
            }
        }
        Ok(())
    }
    fn varint_string(&mut self, value: &str) -> Result<()> {
        self.varint_bytes(Some(value.as_bytes()))
    }
    /// Written bytes from `offset`, or `None` once the bounded writer has overflowed and the
    /// response it holds will be discarded rather than framed.
    fn bytes_from(&self, offset: usize) -> Option<&[u8]> {
        if self.overflowed {
            return None;
        }
        self.bytes.get(offset..)
    }
    fn patch_i32(&mut self, offset: usize, value: i32) -> Result<()> {
        let end = offset
            .checked_add(std::mem::size_of::<i32>())
            .ok_or_else(|| Error::internal("Kafka response patch offset overflow"))?;
        let target = self
            .bytes
            .get_mut(offset..end)
            .ok_or_else(|| Error::internal("Kafka response length placeholder was not written"))?;
        target.copy_from_slice(&value.to_be_bytes());
        Ok(())
    }
    fn compact_array_len(&mut self, value: usize) -> Result<()> {
        let encoded = u32::try_from(value)
            .map_err(|_| Error::invalid_data("Kafka compact array exceeds u32"))?
            .checked_add(1)
            .ok_or_else(|| Error::invalid_data("Kafka compact array length overflow"))?;
        self.unsigned_varint(encoded);
        Ok(())
    }
    #[cfg(test)]
    fn compact_string(&mut self, value: &str) -> Result<()> {
        let encoded = u32::try_from(value.len())
            .map_err(|_| Error::invalid_data("Kafka compact string exceeds u32"))?
            .checked_add(1)
            .ok_or_else(|| Error::invalid_data("Kafka compact string length overflow"))?;
        self.unsigned_varint(encoded);
        self.raw(value.as_bytes());
        Ok(())
    }
    fn empty_tagged_fields(&mut self) {
        self.unsigned_varint(0);
    }
    fn array_len(&mut self, value: usize) {
        self.i32(value as i32);
    }
    fn string(&mut self, value: &str) -> Result<()> {
        self.i16(
            i16::try_from(value.len())
                .map_err(|_| Error::invalid_data("Kafka string exceeds i16"))?,
        );
        self.raw(value.as_bytes());
        Ok(())
    }
    fn nullable_string(&mut self, value: Option<&str>) -> Result<()> {
        if let Some(value) = value {
            self.string(value)
        } else {
            self.i16(-1);
            Ok(())
        }
    }
    fn bytes(&mut self, value: Option<&[u8]>) -> Result<()> {
        if let Some(value) = value {
            self.i32(
                i32::try_from(value.len())
                    .map_err(|_| Error::invalid_data("Kafka bytes exceed i32"))?,
            );
            self.raw(value);
        } else {
            self.i32(-1);
        }
        Ok(())
    }
}

#[cfg(test)]
mod compression_tests {
    use std::{
        io::Write as _,
        process::Command,
        sync::{
            Arc,
            atomic::{AtomicU64, Ordering},
        },
    };

    use parking_lot::Mutex;

    use super::*;

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn real_database_produce_v0_v3_survives_restart() -> Result<()> {
        use crate::{
            engine::{
                ExecutionClass, SingleNodeBootstrapConfig, WriteStorageLimits, open_standalone,
            },
            gpu::{
                BackendKind, DeviceMemoryGovernor, ResolvedComputeDevice,
                create_execution_backend_with_governor,
            },
            protocol::{QueryExecutor, QueryRequest, QueryStreamEvent},
            server::Database,
        };
        let directory = tempfile::tempdir()?;
        for round in 0..2 {
            let boot = open_standalone(
                directory.path(),
                SingleNodeBootstrapConfig {
                    execution_class: ExecutionClass::Cpu,
                    startup_timeout: std::time::Duration::from_secs(30),
                    storage_limits: WriteStorageLimits {
                        max_log_record_bytes: 8 * 1024 * 1024,
                        max_log_entries_per_read: 4096,
                        max_snapshot_bytes: 64 * 1024 * 1024,
                    },
                },
                |path, identity| {
                    Ok(Arc::new(Database::open_backend(
                        path,
                        4 * 1024 * 1024,
                        std::time::Duration::from_secs(10),
                        identity,
                    )?))
                },
            )
            .await?;
            let database = boot.backend().clone();
            database.bind_runtime(Arc::downgrade(boot.runtime()))?;
            let snapshot = directory.path().join("standalone-snapshots");
            database.standalone_recover(&snapshot).await?;
            database.bind_execution_backend(create_execution_backend_with_governor(
                ResolvedComputeDevice {
                    backend: BackendKind::Cpu,
                    ordinal: 0,
                },
                DeviceMemoryGovernor::new(usize::MAX, 0),
            )?)?;
            boot.runtime().replay_standalone_wal().await?;
            let query = |statement: &str| -> Result<Option<ProjectId>> {
                let request: QueryRequest = serde_json::from_value(
                    serde_json::json!({ "request_id":uuid::Uuid::new_v4(), "query":statement }),
                )
                .map_err(|error| Error::internal(error.to_string()))?;
                let mut project = None;
                database.execute(request, &mut |event| {
                    match event {
                        QueryStreamEvent::Catalog { catalog } => project = catalog.project_id,
                        QueryStreamEvent::Error { code, message, .. } => {
                            return Err(Error::new(code, message));
                        }
                        _ => {}
                    }
                    Ok(())
                })?;
                Ok(project)
            };
            if round == 0 {
                query("CREATE PROJECT protocol_restart")?;
                query("USE protocol_restart CREATE TOPIC before_restart PARTITIONS 1")?;
            } else {
                query("USE protocol_restart CREATE TOPIC after_restart PARTITIONS 1")?;
            }
            let project = query("USE protocol_restart RETURN 1")?
                .ok_or_else(|| Error::internal("project identity absent"))?;
            let (server_io, mut client_io) = tokio::io::duplex(2 * 1024 * 1024);
            let server = tokio::spawn(serve(
                server_io,
                project,
                database.clone(),
                KafkaEndpoint {
                    host: "127.0.0.1".into(),
                    port: 18486,
                },
            ));
            let topics = if round == 0 {
                vec!["before_restart"]
            } else {
                vec!["before_restart", "after_restart"]
            };
            for topic in topics {
                let mut expected = if round == 1 && topic == "before_restart" {
                    2
                } else {
                    0
                };
                let prior = database.fetch_partition(project, topic, 0, 0, 4096)?;
                assert_eq!(prior.len(), expected as usize);
                for version in [0, 3] {
                    let value = format!("distinct-{round}-{version}-{topic}");
                    let encoded = if version == 0 {
                        // Magic 0 record with independently constructed CRC and lengths.
                        let mut body = Writer::new();
                        body.i8(0);
                        body.i8(0);
                        body.bytes(None)?;
                        body.bytes(Some(value.as_bytes()))?;
                        let mut message = Writer::new();
                        message.i64(0);
                        message.i32((body.len() + 4) as i32);
                        message.i32(crc32fast::hash(body.as_slice()) as i32);
                        message.raw(body.as_slice());
                        message.finish()
                    } else {
                        encoded_batch(&RecordView {
                            create_time_ms: 1234,
                            key: Some(b"key"),
                            value: Some(value.as_bytes()),
                            headers: vec![header("source", Some(b"protocol"))],
                        })?
                    };
                    let mut produce = Writer::new();
                    if version == 3 {
                        produce.nullable_string(None)?;
                    }
                    produce.i16(-1);
                    produce.i32(5000);
                    produce.array_len(1);
                    produce.string(topic)?;
                    produce.array_len(1);
                    produce.i32(0);
                    produce.bytes(Some(&encoded))?;
                    let response = request(
                        &mut client_io,
                        0,
                        version,
                        expected as i32 + 10,
                        produce.as_slice(),
                    )
                    .await?;
                    let mut response = Reader::new(&response);
                    assert_eq!(response.array_count()?, 1);
                    assert_eq!(response.string()?, topic);
                    assert_eq!(response.array_count()?, 1);
                    assert_eq!(response.i32()?, 0);
                    assert_eq!(response.i16()?, NONE, "produce v{version}, round {round}");
                    assert_eq!(response.i64()?, expected);
                    let read =
                        database.fetch_partition(project, topic, 0, expected as u64, 4096)?;
                    assert_eq!(read.len(), 1);
                    assert_eq!(&*read[0].1.payload, value.as_bytes());
                    expected += 1;
                }
                assert_eq!(
                    database
                        .list_offset(project, topic, 0, -1)?
                        .map(|value| value.0),
                    Some(expected as u64)
                );
            }
            drop(client_io);
            server
                .await
                .map_err(|error| Error::internal(error.to_string()))??;
            database.standalone_snapshot(&snapshot).await?;
            boot.runtime().shutdown().await?;
        }
        Ok(())
    }

    fn nullable_message_set(
        value: Option<&[u8]>,
        attributes: u8,
        timestamp: i64,
    ) -> Result<Vec<u8>> {
        let mut body = Writer::new();
        body.i8(1);
        body.i8(attributes as i8);
        body.i64(timestamp);
        body.bytes(None)?;
        body.bytes(value)?;
        let mut message = Writer::new();
        message.i32(crc32fast::hash(body.as_slice()) as i32);
        message.raw(body.as_slice());
        let mut set = Writer::new();
        set.i64(0);
        set.i32(
            i32::try_from(message.len())
                .map_err(|_| Error::internal("test Kafka message exceeds i32"))?,
        );
        set.raw(message.as_slice());
        Ok(set.finish())
    }

    fn message_set(value: &[u8], attributes: u8, timestamp: i64) -> Result<Vec<u8>> {
        nullable_message_set(Some(value), attributes, timestamp)
    }

    fn gzip(bytes: &[u8]) -> Result<Vec<u8>> {
        let mut encoder = flate2::write::GzEncoder::new(Vec::new(), flate2::Compression::default());
        encoder.write_all(bytes)?;
        Ok(encoder.finish()?)
    }

    fn lz4(bytes: &[u8]) -> Result<Vec<u8>> {
        let mut encoder = lz4_flex::frame::FrameEncoder::new(Vec::new());
        encoder.write_all(bytes)?;
        encoder
            .finish()
            .map_err(|_| Error::internal("test lz4 compression failed"))
    }

    fn xerial_snappy(bytes: &[u8]) -> Result<Vec<u8>> {
        let compressed = snap::raw::Encoder::new()
            .compress_vec(bytes)
            .map_err(|_| Error::internal("test snappy compression failed"))?;
        let mut output = Vec::from(*b"\x82SNAPPY\0");
        output.extend_from_slice(&1_i32.to_be_bytes());
        output.extend_from_slice(&1_i32.to_be_bytes());
        output.extend_from_slice(
            &i32::try_from(compressed.len())
                .map_err(|_| Error::internal("test snappy chunk exceeds i32"))?
                .to_be_bytes(),
        );
        output.extend_from_slice(&compressed);
        Ok(output)
    }

    #[test]
    fn legacy_message_sets_accept_every_standard_compression_codec() -> Result<()> {
        let inner = message_set(b"payload", 0, 17)?;
        let codecs = [
            (1_u8, gzip(&inner)?),
            (2_u8, xerial_snappy(&inner)?),
            (
                2_u8,
                snap::raw::Encoder::new()
                    .compress_vec(&inner)
                    .map_err(|_| Error::internal("test raw snappy compression failed"))?,
            ),
            (3_u8, lz4(&inner)?),
            (
                4_u8,
                zstd::stream::encode_all(inner.as_slice(), 1)
                    .map_err(|_| Error::internal("test zstd compression failed"))?,
            ),
        ];
        for (codec, compressed) in codecs {
            let outer = message_set(&compressed, codec | 0x08, 99)?;
            let decoded = decode_message_set(&outer, 2)?;
            assert_eq!(decoded.len(), 1);
            assert_eq!(decoded[0].value.as_deref(), Some(b"payload".as_slice()));
            assert_eq!(decoded[0].create_time_ms, Some(99));
        }
        Ok(())
    }

    #[test]
    fn null_and_empty_record_values_remain_distinct_through_fetch_encoding() -> Result<()> {
        let decoded_null = decode_message_set(&nullable_message_set(None, 0, 7)?, 2)?;
        let decoded_empty = decode_message_set(&nullable_message_set(Some(&[]), 0, 7)?, 2)?;
        assert_eq!(decoded_null[0].value, None);
        assert_eq!(decoded_empty[0].value, Some(Vec::new()));

        for (value_is_null, expected) in [(true, None), (false, Some(Vec::new()))] {
            let payload = super::super::engine::PayloadRecord {
                id: crate::types::MessageId(1),
                resolved_time_ms: 7,
                ingress: IngressMetadata::Kafka {
                    create_time_ms: Some(7),
                    key: None,
                    headers: BTreeMap::new(),
                    value_is_null,
                },
                payload: Arc::from([]),
                checksum: [0; 32],
            };
            let mut encoded = Writer::new();
            encode_message(&mut encoded, 0, &payload, 2)?;
            assert_eq!(
                encoded.len(),
                encoded_message_length(&payload, 2)?,
                "fetch sizing must exactly match wire encoding"
            );
            let decoded = decode_message_set(encoded.as_slice(), 2)?;
            assert_eq!(decoded[0].value, expected);
        }
        Ok(())
    }

    /// Offsets of the two v2 batch header fields the truncation tests rewrite.
    const BATCH_CRC_OFFSET: usize = 17;
    const BATCH_RECORD_COUNT_OFFSET: usize = 57;

    fn encoded_batch(view: &RecordView<'_>) -> Result<Vec<u8>> {
        let mut output = Writer::new();
        encode_record_batch(&mut output, 7, view)?;
        assert_eq!(
            output.len(),
            record_batch_length(view)?,
            "fetch sizing must exactly match the encoded batch"
        );
        Ok(output.finish())
    }

    fn header(
        name: &str,
        value: Option<&'static [u8]>,
    ) -> (Cow<'static, str>, Option<&'static [u8]>) {
        (Cow::Owned(name.to_owned()), value)
    }

    /// Repairs the CRC after a test has rewritten a header field, so the rewritten field is what
    /// the decoder rejects rather than the checksum that no longer covers it.
    fn reseal_batch(batch: &mut [u8]) -> Result<()> {
        let covered = batch
            .get(BATCH_CRC_OFFSET + 4..)
            .ok_or_else(|| Error::internal("test record batch is truncated"))?;
        let crc = crc32c::crc32c(covered).to_be_bytes();
        batch
            .get_mut(BATCH_CRC_OFFSET..BATCH_CRC_OFFSET + 4)
            .ok_or_else(|| Error::internal("test record batch is truncated"))?
            .copy_from_slice(&crc);
        Ok(())
    }

    #[test]
    fn record_batches_round_trip_null_keys_values_and_headers() -> Result<()> {
        let cases = [
            RecordView {
                create_time_ms: 1_700_000_000_123,
                key: Some(b"order-42".as_slice()),
                value: Some(b"payload".as_slice()),
                headers: vec![
                    header("trace", Some(b"abc".as_slice())),
                    header("retry", Some(b"".as_slice())),
                    header("tombstone", None),
                ],
            },
            RecordView {
                create_time_ms: 0,
                key: None,
                value: None,
                headers: Vec::new(),
            },
            RecordView {
                create_time_ms: -1,
                key: Some(b"".as_slice()),
                value: Some(b"".as_slice()),
                headers: vec![header("only", None)],
            },
        ];
        for view in &cases {
            let decoded = decode_record_batches(&encoded_batch(view)?)?;
            assert_eq!(decoded.len(), 1);
            assert_eq!(decoded[0].key.as_deref(), view.key);
            assert_eq!(decoded[0].value.as_deref(), view.value);
            assert_eq!(
                decoded[0].headers,
                view.headers
                    .iter()
                    .map(|(name, value)| (name.to_string(), value.map(<[u8]>::to_vec)))
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                decoded[0].create_time_ms,
                (view.create_time_ms >= 0).then_some(view.create_time_ms)
            );
        }

        let mut concatenated = encoded_batch(&cases[0])?;
        concatenated.extend_from_slice(&encoded_batch(&cases[1])?);
        assert_eq!(
            decode_record_batches(&concatenated)?.len(),
            2,
            "one partition may carry several batches back to back"
        );
        Ok(())
    }

    #[test]
    fn record_batches_are_checksummed_with_castagnoli() -> Result<()> {
        // The published CRC-32C check value for the nine ASCII digits. CRC-32 (IEEE) answers
        // 0xcbf4_3926 for the same input, so a wrong polynomial cannot pass this.
        assert_eq!(crc32c::crc32c(b"123456789"), 0xe306_9283);
        assert_ne!(crc32fast::hash(b"123456789"), 0xe306_9283);

        let batch = encoded_batch(&RecordView {
            create_time_ms: 5,
            key: Some(b"k".as_slice()),
            value: Some(b"v".as_slice()),
            headers: vec![header("h", Some(b"1".as_slice()))],
        })?;
        let mut reader = Reader::new(&batch);
        let _base_offset = reader.i64()?;
        let _length = reader.i32()?;
        let _leader_epoch = reader.i32()?;
        assert_eq!(reader.i8()?, RECORD_BATCH_MAGIC);
        let written = reader.u32()?;
        let covered = batch
            .get(BATCH_CRC_OFFSET + 4..)
            .ok_or_else(|| Error::internal("test record batch is truncated"))?;
        assert_eq!(written, crc32c::crc32c(covered));
        assert_ne!(written, crc32fast::hash(covered));
        Ok(())
    }

    #[test]
    fn lying_record_batch_lengths_and_counts_are_rejected() -> Result<()> {
        let view = RecordView {
            create_time_ms: 11,
            key: None,
            value: Some(b"payload".as_slice()),
            headers: Vec::new(),
        };
        let valid = encoded_batch(&view)?;

        let mut truncated = valid.clone();
        truncated.truncate(valid.len() - 3);
        assert_eq!(
            decode_record_batches(&truncated)
                .err()
                .ok_or_else(|| Error::internal("truncated batch decoded"))?
                .code,
            ErrorCode::ProtocolViolation
        );

        for claimed in [i32::MAX, 1, -1] {
            let mut lying = valid.clone();
            lying
                .get_mut(8..12)
                .ok_or_else(|| Error::internal("test record batch is truncated"))?
                .copy_from_slice(&claimed.to_be_bytes());
            assert_eq!(
                decode_record_batches(&lying)
                    .err()
                    .ok_or_else(|| Error::internal("lying batch length decoded"))?
                    .code,
                ErrorCode::ProtocolViolation,
                "batch length {claimed} must be rejected before anything is sized from it"
            );
        }

        for (count, delta) in [(i32::MAX, i32::MAX - 1), (1_000_000, 999_999), (-1, -2)] {
            let mut lying = valid.clone();
            lying
                .get_mut(23..27)
                .ok_or_else(|| Error::internal("test record batch is truncated"))?
                .copy_from_slice(&delta.to_be_bytes());
            lying
                .get_mut(BATCH_RECORD_COUNT_OFFSET..BATCH_RECORD_COUNT_OFFSET + 4)
                .ok_or_else(|| Error::internal("test record batch is truncated"))?
                .copy_from_slice(&count.to_be_bytes());
            reseal_batch(&mut lying)?;
            assert_eq!(
                decode_record_batches(&lying)
                    .err()
                    .ok_or_else(|| Error::internal("lying record count decoded"))?
                    .code,
                ErrorCode::ProtocolViolation,
                "record count {count} must be rejected before any record is allocated"
            );
        }

        let mut corrupted = valid;
        let last = corrupted.len().saturating_sub(1);
        corrupted[last] ^= 0xff;
        assert_eq!(
            decode_record_batches(&corrupted)
                .err()
                .ok_or_else(|| Error::internal("corrupted batch decoded"))?
                .code,
            ErrorCode::ProtocolViolation
        );
        Ok(())
    }

    #[test]
    fn record_batch_magic_and_unsupported_attributes_are_rejected() -> Result<()> {
        let valid = encoded_batch(&RecordView {
            create_time_ms: 3,
            key: None,
            value: Some(b"payload".as_slice()),
            headers: Vec::new(),
        })?;

        let mut legacy_magic = valid.clone();
        legacy_magic[16] = 1;
        assert_eq!(
            decode_record_batches(&legacy_magic)
                .err()
                .ok_or_else(|| Error::internal("magic 1 decoded as a record batch"))?
                .code,
            ErrorCode::ProtocolViolation
        );

        // Transactional, control, and an unimplemented compression codec must each fail closed
        // rather than be read as an ordinary uncompressed batch.
        for attributes in [0x10_u16, 0x20, 0x07, 0x80] {
            let mut rejected = valid.clone();
            rejected
                .get_mut(21..23)
                .ok_or_else(|| Error::internal("test record batch is truncated"))?
                .copy_from_slice(&attributes.to_be_bytes());
            reseal_batch(&mut rejected)?;
            assert_eq!(
                decode_record_batches(&rejected)
                    .err()
                    .ok_or_else(|| Error::internal("unsupported attributes decoded"))?
                    .code,
                ErrorCode::ProtocolViolation,
                "attributes {attributes:#x} must be rejected"
            );
        }
        Ok(())
    }

    #[test]
    fn signed_varints_round_trip_at_their_boundaries() -> Result<()> {
        for value in [0, -1, 1, 63, -64, 64, -65, i32::MIN, i32::MAX] {
            let mut output = Writer::new();
            output.signed_varint(value);
            assert_eq!(output.len(), signed_varint_length(value));
            let encoded = output.finish();
            let mut reader = Reader::new(&encoded);
            assert_eq!(reader.signed_varint()?, value);
            reader.finish()?;
        }
        for value in [0_i64, -1, 1, i64::from(i32::MIN), i64::MIN, i64::MAX] {
            let mut output = Writer::new();
            output.signed_varlong(value);
            let encoded = output.finish();
            let mut reader = Reader::new(&encoded);
            assert_eq!(reader.signed_varlong()?, value);
            reader.finish()?;
        }
        // Zigzag is not the unsigned varint the flexible-version fields use; -1 must be one byte.
        let mut output = Writer::new();
        output.signed_varint(-1);
        assert_eq!(output.finish(), vec![1]);
        Ok(())
    }

    #[test]
    fn amqp_ingress_reaches_a_stream_consumer_as_a_keyed_record_with_headers() -> Result<()> {
        let payload = super::super::engine::PayloadRecord {
            id: crate::types::MessageId(3),
            resolved_time_ms: 91,
            ingress: IngressMetadata::Amqp {
                exchange: "orders".to_owned(),
                routing_key: "orders.eu".to_owned(),
                properties: BTreeMap::from([("content_type".to_owned(), b"text/plain".to_vec())]),
                headers: BTreeMap::from([("tenant".to_owned(), b"acme".to_vec())]),
                death_count: 0,
            },
            payload: Arc::from(*b"amqp-published"),
            checksum: [0; 32],
        };

        let mut batch = Writer::new();
        encode_message(&mut batch, 4, &payload, FETCH_RECORD_BATCH_VERSION)?;
        assert_eq!(
            batch.len(),
            encoded_message_length(&payload, FETCH_RECORD_BATCH_VERSION)?
        );
        let decoded = decode_record_batches(&batch.finish())?;
        assert_eq!(decoded.len(), 1);
        assert_eq!(decoded[0].key.as_deref(), Some(b"orders.eu".as_slice()));
        assert_eq!(
            decoded[0].value.as_deref(),
            Some(b"amqp-published".as_slice())
        );
        assert_eq!(
            decoded[0].headers,
            vec![
                ("tenant".to_owned(), Some(b"acme".to_vec())),
                (
                    format!("{AMQP_PROPERTY_HEADER_PREFIX}content_type"),
                    Some(b"text/plain".to_vec())
                ),
            ]
        );

        // A legacy message set has no header field, but it does have a key, so the routing key
        // still maps there.
        let mut legacy = Writer::new();
        encode_message(&mut legacy, 4, &payload, 1)?;
        assert_eq!(legacy.len(), encoded_message_length(&payload, 1)?);
        let legacy = decode_message_set(legacy.as_slice(), 1)?;
        assert_eq!(legacy[0].key.as_deref(), Some(b"orders.eu".as_slice()));
        assert_eq!(legacy[0].create_time_ms, Some(91));
        Ok(())
    }

    #[test]
    fn fetch_partition_maxima_share_one_aggregate_response_budget() -> Result<()> {
        let mut topics = Writer::new();
        topics.string("events")?;
        topics.array_len(2);
        for partition in 0..2 {
            topics.i32(partition);
            topics.i64(0);
            topics.i32(1_000_000);
        }
        let minimum = 4 + 4 + 2 + "events".len() + 4 + (2 * 18);
        let limits =
            measure_fetch_request_with_limit(1, 1, Reader::new(topics.as_slice()), minimum + 17)?;
        assert_eq!(limits.minimum_response_bytes, minimum);
        assert_eq!(limits.maximum_data_bytes, 17);
        Ok(())
    }

    #[test]
    fn oversized_fetch_partition_array_is_rejected_before_response_allocation() -> Result<()> {
        let mut topics = Writer::new();
        topics.string("events")?;
        topics.array_len(8);
        for partition in 0..8 {
            topics.i32(partition);
            topics.i64(0);
            topics.i32(1);
        }
        let error = measure_fetch_request_with_limit(1, 1, Reader::new(topics.as_slice()), 64)
            .expect_err("minimum response must be bounded independently of partition maxima");
        assert_eq!(error.code, ErrorCode::ProtocolViolation);
        Ok(())
    }

    #[test]
    fn malformed_and_excessively_nested_compression_fails_closed() -> Result<()> {
        let malformed = message_set(b"not-gzip", 1, 0)?;
        assert!(decode_message_set(&malformed, 2).is_err());

        let mut nested = message_set(b"payload", 0, 0)?;
        for _ in 0..10 {
            nested = message_set(&gzip(&nested)?, 1, 0)?;
        }
        let error = decode_message_set(&nested, 2)
            .err()
            .ok_or_else(|| Error::internal("excessively nested compression succeeded"))?;
        assert_eq!(error.code, crate::ErrorCode::ProtocolViolation);
        Ok(())
    }

    struct TestCoordinator {
        broker: Mutex<super::super::engine::BrokerStateMachine>,
        segments: crate::storage::SegmentStore,
        _directory: tempfile::TempDir,
        index: AtomicU64,
        changes: tokio::sync::watch::Sender<u64>,
    }

    impl TestCoordinator {
        fn new() -> Result<Self> {
            let directory = tempfile::tempdir()?;
            let segments = crate::storage::SegmentStore::open(directory.path(), 64 * 1024 * 1024)?;
            let (changes, _) = tokio::sync::watch::channel(0);
            Ok(Self {
                broker: Mutex::new(super::super::engine::BrokerStateMachine::default()),
                segments,
                _directory: directory,
                index: AtomicU64::new(0),
                changes,
            })
        }

        fn submissions(&self) -> u64 {
            self.index.load(Ordering::Acquire)
        }
    }

    impl BrokerCoordinator for TestCoordinator {
        fn submit(
            &self,
            command: BrokerCommand,
            wait: CommitAcknowledgement,
        ) -> Result<super::super::engine::BrokerCommit> {
            self.submit_with_timeout(command, wait, 5_000)
        }

        fn submit_with_timeout(
            &self,
            command: BrokerCommand,
            wait: CommitAcknowledgement,
            _timeout_millis: u32,
        ) -> Result<super::super::engine::BrokerCommit> {
            let reply = self.broker.lock().apply(command, &self.segments)?;
            let index = self.index.fetch_add(1, Ordering::AcqRel).saturating_add(1);
            self.changes.send_replace(index);
            let _ = wait;
            Ok(super::super::engine::BrokerCommit {
                bookmark: crate::Bookmark { term: 1, index },
                reply,
                application: ApplicationWait::Complete,
            })
        }

        fn snapshot(&self) -> Result<super::super::engine::BrokerStateMachine> {
            Ok(self.broker.lock().clone())
        }

        fn subscribe_changes(&self) -> tokio::sync::watch::Receiver<u64> {
            self.changes.subscribe()
        }

        fn fetch_partition(
            &self,
            project: ProjectId,
            topic: &str,
            partition: i32,
            offset: u64,
            maximum_bytes: usize,
        ) -> Result<Vec<(u64, Arc<super::super::engine::PayloadRecord>)>> {
            self.broker.lock().fetch_partition(
                project,
                topic,
                partition,
                offset,
                maximum_bytes,
                &self.segments,
            )
        }

        fn list_offset(
            &self,
            project: ProjectId,
            topic: &str,
            partition: i32,
            timestamp: i64,
        ) -> Result<Option<(u64, i64)>> {
            self.broker
                .lock()
                .list_offset(project, topic, partition, timestamp)
        }

        fn queue_info(
            &self,
            project: ProjectId,
            name: &str,
        ) -> Result<Option<super::super::engine::QueueInfo>> {
            Ok(self.broker.lock().queue_info(project, name))
        }

        fn fetch_stream_queue(
            &self,
            project: ProjectId,
            queue: &str,
            offset: super::super::engine::StreamOffset,
            maximum: usize,
            consumer: u64,
            automatic_ack: bool,
        ) -> Result<Vec<super::super::engine::Delivery>> {
            self.broker.lock().read_stream_queue(
                project,
                queue,
                offset,
                maximum,
                consumer,
                automatic_ack,
                &self.segments,
            )
        }
    }

    struct BlockingFetchCoordinator {
        inner: TestCoordinator,
        entered: Mutex<Option<tokio::sync::oneshot::Sender<()>>>,
        release: Mutex<std::sync::mpsc::Receiver<()>>,
    }

    impl BrokerCoordinator for BlockingFetchCoordinator {
        fn submit(
            &self,
            command: BrokerCommand,
            wait: CommitAcknowledgement,
        ) -> Result<super::super::engine::BrokerCommit> {
            self.inner.submit(command, wait)
        }

        fn snapshot(&self) -> Result<super::super::engine::BrokerStateMachine> {
            self.inner.snapshot()
        }

        fn fetch_partition(
            &self,
            _project: ProjectId,
            _topic: &str,
            _partition: i32,
            _offset: u64,
            _maximum_bytes: usize,
        ) -> Result<Vec<(u64, Arc<super::super::engine::PayloadRecord>)>> {
            if let Some(entered) = self.entered.lock().take() {
                let _ = entered.send(());
            }
            self.release
                .lock()
                .recv()
                .map_err(|_| Error::internal("Kafka blocking test release channel closed"))?;
            Ok(Vec::new())
        }

        fn list_offset(
            &self,
            _project: ProjectId,
            _topic: &str,
            _partition: i32,
            _timestamp: i64,
        ) -> Result<Option<(u64, i64)>> {
            Ok(Some((0, -1)))
        }

        fn queue_info(
            &self,
            project: ProjectId,
            name: &str,
        ) -> Result<Option<super::super::engine::QueueInfo>> {
            self.inner.queue_info(project, name)
        }

        fn fetch_stream_queue(
            &self,
            project: ProjectId,
            queue: &str,
            offset: super::super::engine::StreamOffset,
            maximum: usize,
            consumer: u64,
            automatic_ack: bool,
        ) -> Result<Vec<super::super::engine::Delivery>> {
            self.inner
                .fetch_stream_queue(project, queue, offset, maximum, consumer, automatic_ack)
        }
    }

    #[tokio::test]
    async fn aborted_fetch_waiter_holds_memory_until_blocking_work_finishes() -> Result<()> {
        const QUANTUM: usize = 64 * 1024;
        let governor = Arc::new(super::super::memory::BrokerMemoryGovernor::new(
            4 * QUANTUM,
        )?);
        let reservation = governor.reserve(QUANTUM).await?;
        let (entered_sender, entered_receiver) = tokio::sync::oneshot::channel();
        let (release_sender, release_receiver) = std::sync::mpsc::channel();
        let coordinator: Arc<dyn BrokerCoordinator> = Arc::new(BlockingFetchCoordinator {
            inner: TestCoordinator::new()?,
            entered: Mutex::new(Some(entered_sender)),
            release: Mutex::new(release_receiver),
        });

        let mut body = Writer::new();
        body.nullable_string(Some("abort-test"))?;
        body.i32(-1);
        body.i32(0);
        body.i32(0);
        body.array_len(1);
        body.string("events")?;
        body.array_len(1);
        body.i32(0);
        body.i64(0);
        body.i32(1);
        let body = Bytes::from(body.finish());

        let waiter = tokio::spawn(async move {
            execute_request_blocking(
                &reservation,
                1,
                2,
                1,
                ProjectId::random(),
                coordinator,
                KafkaEndpoint {
                    host: "127.0.0.1".to_owned(),
                    port: 9_092,
                },
                body,
            )
            .await
        });
        tokio::time::timeout(std::time::Duration::from_secs(2), entered_receiver)
            .await
            .map_err(|_| Error::internal("Kafka blocking fetch did not start"))?
            .map_err(|_| Error::internal("Kafka blocking fetch entry channel closed"))?;
        assert_eq!(governor.available_bytes(), 3 * QUANTUM);

        waiter.abort();
        let cancellation = match waiter.await {
            Err(error) => error,
            Ok(_) => return Err(Error::internal("Kafka request waiter was not cancelled")),
        };
        assert!(cancellation.is_cancelled());
        assert_eq!(
            governor.available_bytes(),
            3 * QUANTUM,
            "async abort must not release a reservation still protecting blocking work"
        );

        release_sender
            .send(())
            .map_err(|_| Error::internal("Kafka blocking fetch already exited"))?;
        tokio::time::timeout(std::time::Duration::from_secs(2), async {
            while governor.available_bytes() != 4 * QUANTUM {
                tokio::task::yield_now().await;
            }
        })
        .await
        .map_err(|_| Error::internal("Kafka blocking fetch did not release memory"))?;
        Ok(())
    }

    async fn request(
        stream: &mut tokio::io::DuplexStream,
        api_key: i16,
        version: i16,
        correlation: i32,
        body: &[u8],
    ) -> Result<Vec<u8>> {
        let mut frame = Writer::new();
        frame.i16(api_key);
        frame.i16(version);
        frame.i32(correlation);
        if request_header_is_flexible(api_key, version) {
            frame.nullable_string(Some("wire-test"))?;
            frame.empty_tagged_fields();
        } else {
            frame.nullable_string(Some("wire-test"))?;
        }
        frame.raw(body);
        stream
            .write_i32(
                i32::try_from(frame.len())
                    .map_err(|_| Error::internal("Kafka test request exceeds i32"))?,
            )
            .await?;
        stream.write_all(frame.as_slice()).await?;
        stream.flush().await?;
        let length = stream.read_i32().await?;
        if length < 4 || length as usize > MAX_FRAME {
            return Err(kafka_protocol_error(
                "Kafka test response length is invalid",
            ));
        }
        let mut response = vec![0; length as usize];
        stream.read_exact(&mut response).await?;
        let mut reader = Reader::new(&response);
        if reader.i32()? != correlation {
            return Err(kafka_protocol_error(
                "Kafka test response correlation changed",
            ));
        }
        let remaining = reader.remaining();
        Ok(reader.take(remaining)?.to_vec())
    }

    #[tokio::test]
    #[allow(clippy::too_many_lines)]
    async fn flexible_negotiation_and_atomic_partition_produce_work_on_the_wire() -> Result<()> {
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            let project = ProjectId::random();
            let coordinator = Arc::new(TestCoordinator::new()?);
            let broker: Arc<dyn BrokerCoordinator> = coordinator.clone();
            let (server_io, mut client_io) = tokio::io::duplex(2 * 1024 * 1024);
            let server = tokio::spawn(serve(
                server_io,
                project,
                broker,
                KafkaEndpoint {
                    host: "127.0.0.1".to_owned(),
                    port: 9_092,
                },
            ));

            let mut versions = Writer::new();
            versions.compact_string("wire-client")?;
            versions.compact_string("1.0")?;
            versions.empty_tagged_fields();
            let unsupported = request(&mut client_io, 18, 4, 0, versions.as_slice()).await?;
            let mut unsupported = Reader::new(&unsupported);
            assert_eq!(unsupported.i16()?, UNSUPPORTED_VERSION);
            let advertised = unsupported.array_count()?;
            assert!(advertised > 0);
            for _ in 0..advertised {
                let _key = unsupported.i16()?;
                let _minimum = unsupported.i16()?;
                let _maximum = unsupported.i16()?;
            }
            unsupported.finish()?;

            let response = request(&mut client_io, 18, 3, 1, versions.as_slice()).await?;
            let mut response = Reader::new(&response);
            assert_eq!(response.i16()?, NONE);
            let api_count = response.unsigned_varint()?.saturating_sub(1);
            let mut api_versions_max = None;
            for _ in 0..api_count {
                let key = response.i16()?;
                let _minimum = response.i16()?;
                let maximum = response.i16()?;
                response.tagged_fields()?;
                if key == 18 {
                    api_versions_max = Some(maximum);
                }
            }
            assert_eq!(api_versions_max, Some(3));
            assert_eq!(response.i32()?, 0);
            response.tagged_fields()?;
            response.finish()?;

            let mut create = Writer::new();
            create.array_len(1);
            create.string("events")?;
            create.i32(1);
            create.i16(1);
            create.array_len(0);
            create.array_len(0);
            create.i32(5_000);
            let response = request(&mut client_io, 19, 0, 2, create.as_slice()).await?;
            let mut response = Reader::new(&response);
            assert_eq!(response.array_count()?, 1);
            assert_eq!(response.string()?, "events");
            assert_eq!(response.i16()?, NONE);
            response.finish()?;

            let response = request(&mut client_io, 19, 0, 15, create.as_slice()).await?;
            let mut response = Reader::new(&response);
            assert_eq!(response.array_count()?, 1);
            assert_eq!(response.string()?, "events");
            assert_eq!(response.i16()?, TOPIC_ALREADY_EXISTS);
            response.finish()?;

            let mut set = message_set(b"one", 0, 1)?;
            set.extend_from_slice(&message_set(b"two", 0, 2)?);
            let mut produce = Writer::new();
            produce.i16(1);
            produce.i32(5_000);
            produce.array_len(1);
            produce.string("events")?;
            produce.array_len(1);
            produce.i32(0);
            produce.bytes(Some(&set))?;
            let response = request(&mut client_io, 0, 2, 3, produce.as_slice()).await?;
            let mut response = Reader::new(&response);
            assert_eq!(response.array_count()?, 1);
            assert_eq!(response.string()?, "events");
            assert_eq!(response.array_count()?, 1);
            assert_eq!(response.i32()?, 0);
            assert_eq!(response.i16()?, NONE);
            assert_eq!(response.i64()?, 0);
            assert_eq!(response.i64()?, -1);
            assert_eq!(response.i32()?, 0);
            response.finish()?;
            assert_eq!(coordinator.submissions(), 3);

            let mut pending = Writer::new();
            pending.i16(-1);
            pending.i32(1);
            pending.array_len(1);
            pending.string("events")?;
            pending.array_len(1);
            pending.i32(0);
            pending.bytes(Some(&message_set(b"maybe-committed", 0, 3)?))?;
            let response = request(&mut client_io, 0, 2, 4, pending.as_slice()).await?;
            let mut response = Reader::new(&response);
            assert_eq!(response.array_count()?, 1);
            assert_eq!(response.string()?, "events");
            assert_eq!(response.array_count()?, 1);
            assert_eq!(response.i32()?, 0);
            assert_eq!(response.i16()?, NONE);
            assert_eq!(response.i64()?, 2);
            assert_eq!(response.i64()?, -1);
            assert_eq!(response.i32()?, 0);
            response.finish()?;
            assert_eq!(
                coordinator.list_offset(project, "events", 0, -1)?,
                Some((3, -1))
            );

            let mut tiny_fetch = Writer::new();
            tiny_fetch.i32(-1);
            tiny_fetch.i32(0);
            tiny_fetch.i32(0);
            tiny_fetch.array_len(1);
            tiny_fetch.string("events")?;
            tiny_fetch.array_len(1);
            tiny_fetch.i32(0);
            tiny_fetch.i64(0);
            tiny_fetch.i32(1);
            let response = request(&mut client_io, 1, 2, 17, tiny_fetch.as_slice()).await?;
            let mut response = Reader::new(&response);
            assert_eq!(response.i32()?, 0);
            assert_eq!(response.array_count()?, 1);
            assert_eq!(response.string()?, "events");
            assert_eq!(response.array_count()?, 1);
            assert_eq!(response.i32()?, 0);
            assert_eq!(response.i16()?, NONE);
            assert_eq!(response.i64()?, 3);
            let records = response
                .bytes()?
                .ok_or_else(|| kafka_protocol_error("Kafka tiny fetch records are null"))?;
            let records = decode_message_set(records, 2)?;
            assert_eq!(records.len(), 1);
            assert_eq!(records[0].value.as_deref(), Some(b"one".as_slice()));
            response.finish()?;

            let mut fetch = Writer::new();
            fetch.i32(-1);
            fetch.i32(0);
            fetch.i32(1);
            fetch.array_len(1);
            fetch.string("events")?;
            fetch.array_len(1);
            fetch.i32(0);
            fetch.i64(0);
            fetch.i32(1024 * 1024);
            let response = request(&mut client_io, 1, 2, 5, fetch.as_slice()).await?;
            let mut response = Reader::new(&response);
            assert_eq!(response.i32()?, 0);
            assert_eq!(response.array_count()?, 1);
            assert_eq!(response.string()?, "events");
            assert_eq!(response.array_count()?, 1);
            assert_eq!(response.i32()?, 0);
            assert_eq!(response.i16()?, NONE);
            assert_eq!(response.i64()?, 3);
            let records = response
                .bytes()?
                .ok_or_else(|| kafka_protocol_error("Kafka fetch records are null"))?;
            let records = decode_message_set(records, 2)?;
            assert_eq!(
                records
                    .iter()
                    .map(|record| record.value.as_deref().unwrap_or_default())
                    .collect::<Vec<_>>(),
                vec![
                    b"one".as_slice(),
                    b"two".as_slice(),
                    b"maybe-committed".as_slice()
                ]
            );
            response.finish()?;

            let mut metadata = Writer::new();
            metadata.i32(1);
            metadata.string("events")?;
            let response = request(&mut client_io, 3, 1, 6, metadata.as_slice()).await?;
            let mut response = Reader::new(&response);
            assert_eq!(response.array_count()?, 1);
            assert_eq!(response.i32()?, 0);
            assert_eq!(response.string()?, "127.0.0.1");
            assert_eq!(response.i32()?, 9_092);
            assert_eq!(response.nullable_string()?, None);
            assert_eq!(response.i32()?, 0);
            assert_eq!(response.array_count()?, 1);
            assert_eq!(response.i16()?, NONE);
            assert_eq!(response.string()?, "events");
            assert_eq!(response.i8()?, 0);
            assert_eq!(response.array_count()?, 1);
            assert_eq!(response.i16()?, NONE);
            assert_eq!(response.i32()?, 0);
            assert_eq!(response.i32()?, 0);
            assert_eq!(response.array_count()?, 1);
            assert_eq!(response.i32()?, 0);
            assert_eq!(response.array_count()?, 1);
            assert_eq!(response.i32()?, 0);
            response.finish()?;

            let mut offsets = Writer::new();
            offsets.i32(-1);
            offsets.array_len(1);
            offsets.string("events")?;
            offsets.array_len(3);
            for timestamp in [-1, 2, 999] {
                offsets.i32(0);
                offsets.i64(timestamp);
            }
            let response = request(&mut client_io, 2, 1, 7, offsets.as_slice()).await?;
            let mut response = Reader::new(&response);
            assert_eq!(response.array_count()?, 1);
            assert_eq!(response.string()?, "events");
            assert_eq!(response.array_count()?, 3);
            assert_eq!(response.i32()?, 0);
            assert_eq!(response.i16()?, NONE);
            assert_eq!(response.i64()?, -1);
            assert_eq!(response.i64()?, 3);
            assert_eq!(response.i32()?, 0);
            assert_eq!(response.i16()?, NONE);
            assert_eq!(response.i64()?, 2);
            assert_eq!(response.i64()?, 1);
            assert_eq!(response.i32()?, 0);
            assert_eq!(response.i16()?, NONE);
            assert_eq!(response.i64()?, -1);
            assert_eq!(response.i64()?, -1);
            response.finish()?;

            let mut find = Writer::new();
            find.string("workers")?;
            let response = request(&mut client_io, 10, 0, 8, find.as_slice()).await?;
            let mut response = Reader::new(&response);
            assert_eq!(response.i16()?, NONE);
            assert_eq!(response.i32()?, 0);
            assert_eq!(response.string()?, "127.0.0.1");
            assert_eq!(response.i32()?, 9_092);
            response.finish()?;

            let mut join = Writer::new();
            join.string("workers")?;
            join.i32(30_000);
            join.string("")?;
            join.string("consumer")?;
            join.array_len(1);
            join.string("range")?;
            join.bytes(Some(b"subscription"))?;
            let response = request(&mut client_io, 11, 0, 9, join.as_slice()).await?;
            let mut response = Reader::new(&response);
            assert_eq!(response.i16()?, NONE);
            let generation = response.i32()?;
            assert_eq!(response.string()?, "range");
            let leader = response.string()?.to_owned();
            let member = response.string()?.to_owned();
            assert_eq!(leader, member);
            assert_eq!(response.array_count()?, 1);
            assert_eq!(response.string()?, member);
            assert_eq!(response.bytes()?, Some(b"subscription".as_slice()));
            response.finish()?;

            let mut sync = Writer::new();
            sync.string("workers")?;
            sync.i32(generation);
            sync.string(&member)?;
            sync.array_len(1);
            sync.string(&member)?;
            sync.bytes(Some(b"assignment"))?;
            let response = request(&mut client_io, 14, 0, 10, sync.as_slice()).await?;
            let mut response = Reader::new(&response);
            assert_eq!(response.i16()?, NONE);
            assert_eq!(response.bytes()?, Some(b"assignment".as_slice()));
            response.finish()?;

            let mut heartbeat = Writer::new();
            heartbeat.string("workers")?;
            heartbeat.i32(generation);
            heartbeat.string(&member)?;
            let response = request(&mut client_io, 12, 0, 11, heartbeat.as_slice()).await?;
            let mut response = Reader::new(&response);
            assert_eq!(response.i16()?, NONE);
            response.finish()?;

            let mut commit = Writer::new();
            commit.string("workers")?;
            commit.i32(generation);
            commit.string(&member)?;
            commit.array_len(1);
            commit.string("events")?;
            commit.array_len(1);
            commit.i32(0);
            commit.i64(3);
            commit.nullable_string(None)?;
            let response = request(&mut client_io, 8, 1, 12, commit.as_slice()).await?;
            let mut response = Reader::new(&response);
            assert_eq!(response.array_count()?, 1);
            assert_eq!(response.string()?, "events");
            assert_eq!(response.array_count()?, 1);
            assert_eq!(response.i32()?, 0);
            assert_eq!(response.i16()?, NONE);
            response.finish()?;

            let mut offset_fetch = Writer::new();
            offset_fetch.string("workers")?;
            offset_fetch.array_len(1);
            offset_fetch.string("events")?;
            offset_fetch.array_len(1);
            offset_fetch.i32(0);
            let response = request(&mut client_io, 9, 1, 13, offset_fetch.as_slice()).await?;
            let mut response = Reader::new(&response);
            assert_eq!(response.array_count()?, 1);
            assert_eq!(response.string()?, "events");
            assert_eq!(response.array_count()?, 1);
            assert_eq!(response.i32()?, 0);
            assert_eq!(response.i64()?, 3);
            assert_eq!(response.nullable_string()?, None);
            assert_eq!(response.i16()?, NONE);
            response.finish()?;

            let mut leave = Writer::new();
            leave.string("workers")?;
            leave.string(&member)?;
            let response = request(&mut client_io, 13, 0, 14, leave.as_slice()).await?;
            let mut response = Reader::new(&response);
            assert_eq!(response.i16()?, NONE);
            response.finish()?;

            let mut delete = Writer::new();
            delete.array_len(1);
            delete.string("events")?;
            delete.i32(5_000);
            let response = request(&mut client_io, 20, 0, 16, delete.as_slice()).await?;
            let mut response = Reader::new(&response);
            assert_eq!(response.array_count()?, 1);
            assert_eq!(response.string()?, "events");
            assert_eq!(response.i16()?, NONE);
            response.finish()?;

            drop(client_io);
            server
                .await
                .map_err(|error| Error::internal(format!("Kafka test server failed: {error}")))??;
            Ok(())
        })
        .await
        .map_err(|_| Error::new(ErrorCode::DeadlineExceeded, "Kafka wire test timed out"))?
    }

    #[tokio::test]
    #[allow(clippy::too_many_lines)]
    async fn produce_v3_and_fetch_v4_carry_record_batches_with_headers() -> Result<()> {
        tokio::time::timeout(std::time::Duration::from_secs(5), async {
            let project = ProjectId::random();
            let coordinator = Arc::new(TestCoordinator::new()?);
            let broker: Arc<dyn BrokerCoordinator> = coordinator.clone();
            let (server_io, mut client_io) = tokio::io::duplex(2 * 1024 * 1024);
            let server = tokio::spawn(serve(
                server_io,
                project,
                broker,
                KafkaEndpoint {
                    host: "127.0.0.1".to_owned(),
                    port: 9_092,
                },
            ));

            let mut versions = Writer::new();
            versions.compact_string("record-batch-client")?;
            versions.compact_string("1.0")?;
            versions.empty_tagged_fields();
            let response = request(&mut client_io, 18, 3, 0, versions.as_slice()).await?;
            let mut response = Reader::new(&response);
            assert_eq!(response.i16()?, NONE);
            let api_count = response.unsigned_varint()?.saturating_sub(1);
            let mut advertised = BTreeMap::new();
            for _ in 0..api_count {
                let key = response.i16()?;
                let _minimum = response.i16()?;
                let maximum = response.i16()?;
                response.tagged_fields()?;
                advertised.insert(key, maximum);
            }
            // A client picks the highest advertised version, so anything advertised must be a
            // version the dispatch below actually accepts.
            assert_eq!(advertised.get(&0), Some(&MAX_PRODUCE_VERSION));
            assert_eq!(advertised.get(&1), Some(&MAX_FETCH_VERSION));

            let mut create = Writer::new();
            create.array_len(1);
            create.string("events")?;
            create.i32(1);
            create.i16(1);
            create.array_len(0);
            create.array_len(0);
            create.i32(5_000);
            let _created = request(&mut client_io, 19, 0, 1, create.as_slice()).await?;

            let mut batches = encoded_batch(&RecordView {
                create_time_ms: 1_700_000_000_000,
                key: Some(b"first".as_slice()),
                value: Some(b"one".as_slice()),
                headers: vec![
                    header("trace", Some(b"abc".as_slice())),
                    header("tenant", Some(b"acme".as_slice())),
                ],
            })?;
            batches.extend_from_slice(&encoded_batch(&RecordView {
                create_time_ms: 1_700_000_000_001,
                key: None,
                value: None,
                headers: Vec::new(),
            })?);
            let mut produce = Writer::new();
            produce.nullable_string(None)?;
            produce.i16(-1);
            produce.i32(5_000);
            produce.array_len(1);
            produce.string("events")?;
            produce.array_len(1);
            produce.i32(0);
            produce.bytes(Some(&batches))?;
            let response = request(
                &mut client_io,
                0,
                MAX_PRODUCE_VERSION,
                2,
                produce.as_slice(),
            )
            .await?;
            let mut response = Reader::new(&response);
            assert_eq!(response.array_count()?, 1);
            assert_eq!(response.string()?, "events");
            assert_eq!(response.array_count()?, 1);
            assert_eq!(response.i32()?, 0);
            assert_eq!(response.i16()?, NONE);
            assert_eq!(response.i64()?, 0);
            assert_eq!(response.i64()?, -1);
            assert_eq!(response.i32()?, 0);
            response.finish()?;

            let mut fetch = Writer::new();
            fetch.i32(-1);
            fetch.i32(0);
            fetch.i32(1);
            fetch.i32(1024 * 1024);
            fetch.i8(0);
            fetch.array_len(1);
            fetch.string("events")?;
            fetch.array_len(1);
            fetch.i32(0);
            fetch.i64(0);
            fetch.i32(1024 * 1024);
            let response =
                request(&mut client_io, 1, MAX_FETCH_VERSION, 3, fetch.as_slice()).await?;
            let mut response = Reader::new(&response);
            assert_eq!(response.i32()?, 0);
            assert_eq!(response.array_count()?, 1);
            assert_eq!(response.string()?, "events");
            assert_eq!(response.array_count()?, 1);
            assert_eq!(response.i32()?, 0);
            assert_eq!(response.i16()?, NONE);
            assert_eq!(response.i64()?, 2);
            assert_eq!(
                response.i64()?,
                2,
                "last stable offset tracks the high watermark"
            );
            assert_eq!(response.array_count()?, 0, "no transaction can be aborted");
            let records = response
                .bytes()?
                .ok_or_else(|| kafka_protocol_error("Kafka v4 fetch records are null"))?;
            let records = decode_record_batches(records)?;
            assert_eq!(
                records
                    .iter()
                    .map(|record| (
                        record.key.clone(),
                        record.value.clone(),
                        record.headers.clone()
                    ))
                    .collect::<Vec<_>>(),
                vec![
                    (
                        Some(b"first".to_vec()),
                        Some(b"one".to_vec()),
                        vec![
                            ("tenant".to_owned(), Some(b"acme".to_vec())),
                            ("trace".to_owned(), Some(b"abc".to_vec())),
                        ]
                    ),
                    (None, None, Vec::new()),
                ],
                "a v4 fetch must return record batches with their headers intact"
            );
            response.finish()?;

            drop(client_io);
            server
                .await
                .map_err(|error| Error::internal(format!("Kafka test server failed: {error}")))??;
            Ok(())
        })
        .await
        .map_err(|_| {
            Error::new(
                ErrorCode::DeadlineExceeded,
                "Kafka record batch wire test timed out",
            )
        })?
    }

    #[tokio::test]
    async fn fetch_long_poll_wakes_from_change_notification_without_polling() -> Result<()> {
        tokio::time::timeout(std::time::Duration::from_secs(3), async {
            let project = ProjectId::random();
            let coordinator = Arc::new(TestCoordinator::new()?);
            coordinator.submit(
                BrokerCommand::CreateTopic {
                    project,
                    name: "events".to_owned(),
                    partitions: 1,
                    retention: RetentionPolicy {
                        max_age_ms: None,
                        max_bytes: None,
                    },
                },
                CommitAcknowledgement::Published,
            )?;
            let broker: Arc<dyn BrokerCoordinator> = coordinator.clone();
            let (server_io, mut client_io) = tokio::io::duplex(2 * 1024 * 1024);
            let server = tokio::spawn(serve(
                server_io,
                project,
                broker,
                KafkaEndpoint {
                    host: "127.0.0.1".to_owned(),
                    port: 9_092,
                },
            ));

            let mut fetch = Writer::new();
            fetch.i32(-1);
            fetch.i32(1_000);
            fetch.i32(1);
            fetch.array_len(1);
            fetch.string("events")?;
            fetch.array_len(1);
            fetch.i32(0);
            fetch.i64(0);
            fetch.i32(1024 * 1024);
            let fetch = fetch.finish();
            let client =
                tokio::spawn(async move { request(&mut client_io, 1, 2, 77, &fetch).await });
            tokio::time::sleep(std::time::Duration::from_millis(40)).await;
            assert!(
                !client.is_finished(),
                "empty fetch returned before min-bytes or max-wait"
            );

            coordinator.submit(
                BrokerCommand::PublishKafkaBatch {
                    project,
                    topic: "events".to_owned(),
                    partition: 0,
                    resolved_time_ms: 0,
                    records: vec![KafkaBatchRecord {
                        create_time_ms: Some(17),
                        key: None,
                        headers: BTreeMap::new(),
                        value_is_null: false,
                        payload: b"notified".to_vec(),
                    }],
                },
                CommitAcknowledgement::Published,
            )?;

            let response = client
                .await
                .map_err(|error| Error::internal(format!("Kafka client task failed: {error}")))??;
            let mut response = Reader::new(&response);
            assert_eq!(response.i32()?, 0);
            assert_eq!(response.array_count()?, 1);
            assert_eq!(response.string()?, "events");
            assert_eq!(response.array_count()?, 1);
            assert_eq!(response.i32()?, 0);
            assert_eq!(response.i16()?, NONE);
            assert_eq!(response.i64()?, 1);
            let records = response
                .bytes()?
                .ok_or_else(|| kafka_protocol_error("Kafka fetch records are null"))?;
            let records = decode_message_set(records, 2)?;
            assert_eq!(records.len(), 1);
            assert_eq!(records[0].value.as_deref(), Some(b"notified".as_slice()));
            response.finish()?;
            server
                .await
                .map_err(|error| Error::internal(format!("Kafka server task failed: {error}")))??;
            Ok(())
        })
        .await
        .map_err(|_| {
            Error::new(
                ErrorCode::DeadlineExceeded,
                "Kafka notification long-poll test timed out",
            )
        })?
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    #[ignore = "manual gate: requires the kafka-python package and loopback socket permission"]
    async fn kafka_python_preserves_null_and_empty_values() -> Result<()> {
        tokio::time::timeout(std::time::Duration::from_secs(30), async {
            let project = ProjectId::random();
            let coordinator = Arc::new(TestCoordinator::new()?);
            coordinator.submit(
                BrokerCommand::CreateTopic {
                    project,
                    name: "events".to_owned(),
                    partitions: 1,
                    retention: RetentionPolicy {
                        max_age_ms: None,
                        max_bytes: None,
                    },
                },
                CommitAcknowledgement::Published,
            )?;

            let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
            let address = listener.local_addr()?;
            let shutdown = tokio_util::sync::CancellationToken::new();
            let server_shutdown = shutdown.clone();
            let broker: Arc<dyn BrokerCoordinator> = coordinator;
            let server = tokio::spawn(async move {
                loop {
                    tokio::select! {
                        _ = server_shutdown.cancelled() => return Ok::<_, Error>(()),
                        accepted = listener.accept() => {
                            let (stream, _) = accepted?;
                            let broker = Arc::clone(&broker);
                            tokio::spawn(async move {
                                if let Err(error) = serve(
                                    stream,
                                    project,
                                    broker,
                                    KafkaEndpoint {
                                        host: address.ip().to_string(),
                                        port: address.port(),
                                    },
                                )
                                .await
                                {
                                    eprintln!("Kafka interoperability peer failed: {error}");
                                }
                            });
                        }
                    }
                }
            });

            let endpoint = address.to_string();
            let driver = tokio::task::spawn_blocking(move || {
                Command::new("python3")
                    .env("IG_KAFKA_ENDPOINT", endpoint)
                    .arg("-c")
                    .arg(
                        r#"
import os
import time
import kafka
import logging
from kafka import KafkaConsumer, KafkaProducer, TopicPartition

logging.basicConfig(level=logging.INFO)
endpoint = os.environ["IG_KAFKA_ENDPOINT"]
producer = KafkaProducer(
    bootstrap_servers=[endpoint],
    acks="all",
    api_version_auto_timeout_ms=5000,
    max_block_ms=10000,
    request_timeout_ms=10000,
)
first = producer.send("events", key=b"null", value=None).get(timeout=10)
second = producer.send("events", key=b"empty", value=b"").get(timeout=10)
assert (first.partition, first.offset) == (0, 0)
assert (second.partition, second.offset) == (0, 1)
producer.close(timeout=10)

consumer = KafkaConsumer(
    bootstrap_servers=[endpoint],
    group_id=None,
    enable_auto_commit=False,
    api_version_auto_timeout_ms=5000,
    request_timeout_ms=10000,
)
partition = TopicPartition("events", 0)
consumer.assign([partition])
consumer.seek(partition, 0)
records = []
deadline = time.monotonic() + 10
while len(records) < 2 and time.monotonic() < deadline:
    for batch in consumer.poll(timeout_ms=500, max_records=2).values():
        records.extend(batch)
consumer.close()
assert [(item.key, item.value) for item in records[:2]] == [
    (b"null", None),
    (b"empty", b""),
]
print(f"kafka-python={kafka.__version__}")
"#,
                    )
                    .output()
            })
            .await
            .map_err(|error| Error::internal(format!("Kafka Python task failed: {error}")))??;
            shutdown.cancel();
            server
                .await
                .map_err(|error| Error::internal(format!("Kafka test server failed: {error}")))??;
            if !driver.status.success() {
                return Err(Error::new(
                    ErrorCode::ProtocolViolation,
                    format!(
                        "kafka-python interoperability failed\nstdout:\n{}\nstderr:\n{}",
                        String::from_utf8_lossy(&driver.stdout),
                        String::from_utf8_lossy(&driver.stderr),
                    ),
                ));
            }
            eprintln!("{}", String::from_utf8_lossy(&driver.stdout));
            Ok(())
        })
        .await
        .map_err(|_| {
            Error::new(
                ErrorCode::DeadlineExceeded,
                "kafka-python interoperability test timed out",
            )
        })?
    }
}
