use irongraph_sdk::{
    EmbeddedDatabase, EmbeddedOptions, EmbeddingPolicy, ExecutionDevice, Query, RemoteClient,
};

#[test]
fn packaged_sdk_preserves_documents_and_integer_precision() -> Result<(), Box<dyn std::error::Error>>
{
    let directory = tempfile::tempdir()?;
    let options = EmbeddedOptions::new(directory.path())
        .with_execution_device(ExecutionDevice::Cpu)
        .with_embedding_policy(EmbeddingPolicy::Disabled);
    let database = EmbeddedDatabase::open(options.clone())?;
    database.query(Query::new("CREATE PROJECT sdk"))?;
    database.query(
        Query::new("USE sdk CREATE (:Document {body: $body, count: 9007199254740993})")
            .with_parameter("body", "Cargo binary persistence"),
    )?;
    database.snapshot()?;
    let project_id = database
        .query(Query::new("USE sdk RETURN 1"))?
        .catalog
        .and_then(|catalog| catalog["project_id"].as_str().map(str::to_owned))
        .ok_or("missing project ID")?;
    database.query(
        Query::new("CREATE TOPIC native_stream PARTITIONS 1").with_project(project_id.clone()),
    )?;
    let ack = database.stream_append(
        irongraph_sdk::StreamAppend {
            project_id: project_id.clone(),
            topic: "native_stream".into(),
            partition: 0,
            records: vec![irongraph_sdk::StreamRecord {
                key: Some(vec![0, 255]),
                headers: Default::default(),
                value: Some(vec![1, 2, 3]),
                create_time_ms: None,
            }],
        },
        Default::default(),
    )?;
    assert_eq!((ack.first_offset, ack.record_count), (0, 1));
    let fetch = irongraph_sdk::StreamFetch {
        project_id,
        topic: "native_stream".into(),
        partition: 0,
        offset: 0,
        max_records: 1,
        max_bytes: 4096,
    };
    let page = database.stream_fetch(fetch.clone(), Default::default())?;
    assert_eq!(page.records[0].1.payload, vec![1, 2, 3]);
    assert_eq!(database.status()?.data_dir, directory.path());
    database.flush()?;
    database.close()?;
    let reopened = EmbeddedDatabase::open(options)?;
    let persisted = reopened.stream_fetch(fetch, Default::default())?;
    assert_eq!(persisted.high_watermark, 1);
    assert_eq!(persisted.records[0].1.id, page.records[0].1.id);
    let result = reopened.query(Query::new(
        "USE sdk MATCH (d:Document) RETURN d.body, d.count",
    ))?;
    assert_eq!(result.rows.len(), 1);
    let encoded = serde_json::to_string(&result.rows)?;
    assert!(encoded.contains("Cargo binary persistence"));
    assert!(encoded.contains("9007199254740993"));
    reopened.close()?;
    Ok(())
}

#[test]
fn remote_connections_preserve_loopback_boundary() {
    match RemoteClient::api("http://example.com") {
        Ok(_) => panic!("plain remote connection was accepted"),
        Err(error) => assert_eq!(error.code, "CONFIGURATION"),
    }
}
