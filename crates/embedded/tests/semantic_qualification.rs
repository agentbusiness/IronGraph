use irongraph_client::{ApiClient, Query};
use irongraph_embedded::{EmbeddedDatabase, EmbeddedOptions, EmbeddingDevice, ExecutionDevice};
use irongraph_server::protocol::TypedValue;

#[test]
#[ignore = "release qualification: loads the pinned local embedding model"]
fn real_model_automatic_semantic_native_and_remote() -> Result<(), Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let device = match std::env::var("IRONGRAPH_QUALIFY_DEVICE").as_deref() {
        Ok("metal") => EmbeddingDevice::Metal(0),
        Ok("cuda") => EmbeddingDevice::Cuda(0),
        _ => EmbeddingDevice::Cpu,
    };
    let database = EmbeddedDatabase::open(
        EmbeddedOptions::new(directory.path())
            .with_execution_device(ExecutionDevice::Cpu)
            .with_embedding_device(device),
    )?;
    database.query(Query::new("CREATE PROJECT qualification"))?;
    database.query(Query::new("USE qualification CREATE (p:Person {name:'Ada'}), (t:Task {title:'Arrange lessons'}), (t)-[:ASSIGNED_TO {description:'guitar music tuition'}]->(p)"))?;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    let result = loop {
        let result = database.query(Query::new("USE qualification SEARCH entity IN (EMBEDDING INDEX graph_semantic FOR TEXT 'guitar music tuition' LIMIT 10) SCORE AS score RETURN entity, score"))?;
        if result.rows.len() == 3 {
            break result;
        }
        if std::time::Instant::now() >= deadline {
            return Err(format!(
                "asynchronous embedding did not publish all owners: {} rows",
                result.rows.len()
            )
            .into());
        }
        std::thread::sleep(std::time::Duration::from_millis(25));
    };
    assert_eq!(result.rows.len(), 3);
    assert!(matches!(result.rows[0][0], TypedValue::Relationship(_)));
    assert!(result.rows.windows(2).all(|pair| matches!((&pair[0][1], &pair[1][1]), (TypedValue::Float(left), TypedValue::Float(right)) if left >= right)));
    database.close()?;
    if let Ok(url) = std::env::var("IRONGRAPH_QUALIFY_API") {
        let client = ApiClient::new(&url)?;
        let result = client.query(Query::new("USE acceptance SEARCH entity IN (EMBEDDING INDEX graph_semantic FOR TEXT 'contract negotiation supplier agreements' LIMIT 8) SCORE AS score RETURN entity, score"))?;
        assert_eq!(result.rows.len(), 8);
        assert!(matches!(result.rows[0][0], TypedValue::Relationship(_)));
    }
    Ok(())
}
