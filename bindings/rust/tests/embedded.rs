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
    database.close()?;
    let reopened = EmbeddedDatabase::open(options)?;
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
