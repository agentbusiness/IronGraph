use irongraph_client::{ApiClient, Query};
use irongraph_embedded::{EmbeddedDatabase, EmbeddedOptions, ExecutionDevice};
use irongraph_server::protocol::TypedValue;

#[test]
#[ignore = "release qualification: loads the pinned local embedding model"]
fn real_model_automatic_semantic_native_and_remote() -> Result<(), Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let device = match std::env::var("IRONGRAPH_QUALIFY_DEVICE").as_deref() {
        Ok("metal") => ExecutionDevice::Metal(0),
        Ok("cuda") => ExecutionDevice::Cuda(0),
        _ => ExecutionDevice::Cpu,
    };
    let database = EmbeddedDatabase::open(
        EmbeddedOptions::new(directory.path()).with_execution_device(device),
    )?;
    database.query(Query::new("CREATE PROJECT qualification"))?;
    database.query(Query::new("USE qualification CREATE (p:Person {name:'Ada'}), (t:Task {title:'Arrange lessons'}), (t)-[:ASSIGNED_TO {description:'guitar music tuition'}]->(p)"))?;
    let result = database.query(Query::new("USE qualification SEARCH entity IN (EMBEDDING INDEX graph_semantic FOR TEXT 'guitar music tuition' LIMIT 10) SCORE AS score RETURN entity, score"))?;
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
