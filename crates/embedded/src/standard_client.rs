use super::*;

#[test]
fn standard_kafka_client_and_native_stream_share_the_real_database()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    use rdkafka::{
        ClientConfig, Message,
        consumer::{BaseConsumer, Consumer},
        producer::{FutureProducer, FutureRecord},
    };
    let _guard = acceptance::TESTS.lock().unwrap();
    let directory = tempfile::tempdir()?;
    let options = EmbeddedOptions::new(directory.path())
        .with_execution_device(ExecutionDevice::Cpu)
        .with_embedding_policy(EmbeddingPolicy::Disabled);
    let database = EmbeddedDatabase::open(options.clone())?;
    database.query(Query::new("CREATE PROJECT interop"))?;
    database.query(Query::new(
        "USE interop CREATE (:Document {body: 'shared engine'})",
    ))?;
    let result = database.query(Query::new("USE interop RETURN 1"))?;
    let project = result
        .catalog
        .and_then(|catalog| catalog.project_id)
        .ok_or("project identity missing")?;
    database.query(Query::new("CREATE TOPIC events PARTITIONS 1").with_project(project))?;
    let address = std::net::TcpListener::bind("127.0.0.1:0")?.local_addr()?;
    let shutdown = CancellationToken::new();
    let server = irongraph_server::broker::StreamServer::new(
        address,
        project,
        std::sync::Arc::new(database.core.as_ref().unwrap().database.clone()),
    );
    let serving = database.runtime.spawn(server.run(shutdown.clone()));
    let producer: FutureProducer = ClientConfig::new()
        .set("bootstrap.servers", address.to_string())
        .set("message.timeout.ms", "10000")
        .set("enable.idempotence", "false")
        .create()?;
    let delivery = database
        .runtime
        .block_on(
            producer.send(
                FutureRecord::to("events")
                    .key("source")
                    .payload("from-librdkafka"),
                Duration::from_secs(10),
            ),
        )
        .map_err(|(error, _)| error)?;
    assert_eq!(delivery.offset, 0);
    let fetch = StreamFetch {
        project_id: project,
        topic: "events".into(),
        partition: 0,
        offset: 0,
        max_records: 2,
        max_bytes: 4096,
    };
    let page = database.stream_fetch(fetch.clone(), Default::default())?;
    assert_eq!(&*page.records[0].1.payload, b"from-librdkafka");
    let ack = database.stream_append(
        StreamAppend {
            project_id: project,
            topic: "events".into(),
            partition: 0,
            records: vec![StreamRecord {
                key: None,
                headers: Default::default(),
                value: Some(b"from-native".to_vec()),
                create_time_ms: None,
            }],
        },
        Default::default(),
    )?;
    assert_eq!(ack.first_offset, 1);
    let consumer: BaseConsumer = ClientConfig::new()
        .set("bootstrap.servers", address.to_string())
        .set("group.id", "native-acceptance")
        .set("enable.auto.commit", "false")
        .create()?;
    let mut partitions = rdkafka::TopicPartitionList::new();
    partitions.add_partition_offset("events", 0, rdkafka::Offset::Beginning)?;
    consumer.assign(&partitions)?;
    for (offset, value) in [b"from-librdkafka".as_slice(), b"from-native".as_slice()]
        .into_iter()
        .enumerate()
    {
        let record = consumer
            .poll(Duration::from_secs(10))
            .ok_or("consumer deadline")??;
        assert_eq!(record.offset(), offset as i64);
        assert_eq!(record.payload(), Some(value));
    }
    drop(consumer);
    drop(producer);
    shutdown.cancel();
    database.runtime.block_on(serving)??;
    database.flush()?;
    database.close()?;
    let reopened = EmbeddedDatabase::open(options)?;
    assert_eq!(
        reopened
            .stream_fetch(fetch, Default::default())?
            .high_watermark,
        2
    );
    assert_eq!(
        reopened
            .query(Query::new("USE interop MATCH (d:Document) RETURN d.body"))?
            .rows
            .len(),
        1
    );
    reopened.close()?;
    Ok(())
}
