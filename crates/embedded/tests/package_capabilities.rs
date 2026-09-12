use irongraph_client::Query;
use irongraph_embedded::{EmbeddedDatabase, EmbeddedOptions, EmbeddingPolicy, ExecutionDevice};
use irongraph_server::protocol::TypedValue;

#[test]
fn canonical_query_covers_embedded_package_capabilities() -> Result<(), Box<dyn std::error::Error>>
{
    let directory = tempfile::tempdir()?;
    let options = || {
        EmbeddedOptions::new(directory.path())
            .with_execution_device(ExecutionDevice::Cpu)
            .with_embedding_policy(EmbeddingPolicy::Disabled)
    };
    let database = EmbeddedDatabase::open(options())?;
    // Projects are explicit, including for document and search operations.
    assert!(database.query(Query::new("MATCH (n) RETURN n")).is_err());
    database.query(Query::new("CREATE PROJECT package_app"))?;
    database.query(Query::new("CREATE PROJECT package_other"))?;
    database.query(Query::new(
        "USE package_app CREATE (a:Document {id: 'first', body: 'Original body', embedding: [1.0, 0.0]}), \
         (b:Document {id: 'second', body: 'Other body', embedding: [0.0, 1.0]}) CREATE (a)-[:CITES]->(b)",
    ))?;
    database.query(Query::new(
        "USE package_app CREATE INDEX document_id FOR (d:Document) ON (d.id)",
    ))?;
    database.query(Query::new(
        "USE package_app CREATE TEXT INDEX document_body FOR (d:Document) ON (d.body)",
    ))?;
    database.query(Query::new(
        "USE package_app MATCH (d:Document {id: 'first'}) SET d.body = 'Revised body'",
    ))?;
    let text = database.query(Query::new(
        "USE package_app MATCH (d:Document) WHERE d.body CONTAINS 'Revised' RETURN d.id",
    ))?;
    assert!(
        matches!(text.rows.as_slice(), [row] if matches!(row.as_slice(), [TypedValue::String(id)] if id == "first"))
    );
    let similarity = database.query(Query::new(
        "USE package_app MATCH (d:Document) \
         RETURN d.id, vector.cosine(d.embedding, [1.0, 0.0]) AS score ORDER BY score DESC",
    ))?;
    assert!(
        matches!(similarity.rows[0].as_slice(), [TypedValue::String(id), TypedValue::Float(score)] if id == "first" && (*score - 1.0).abs() < 1e-6)
    );
    // Indexed search requires an activated model profile. This fixture deliberately never
    // downloads a model; the release's real-model gate qualifies successful indexed search.
    assert!(
        database
            .query(Query::new(
                "USE package_app MATCH (d:Document) SEARCH d IN (EMBEDDING INDEX unavailable \
         FOR VECTOR [1.0, 0.0] LIMIT 1) SCORE AS score RETURN d.id, score",
            ))
            .is_err()
    );
    let paths = database.query(Query::new(
        "USE package_app MATCH p = (:Document {id: 'first'})-[:CITES]->(:Document) RETURN p",
    ))?;
    assert!(
        matches!(paths.rows.as_slice(), [row] if matches!(row.as_slice(), [TypedValue::Path(_)]))
    );
    let components = database.query(Query::new(
        "USE package_app CALL graph.wcc() YIELD node, component RETURN node, component",
    ))?;
    assert_eq!(components.rows.len(), 2);
    for layer in ["KNOWLEDGE", "WORKSPACE"] {
        database.query(Query::new(format!(
            "USE package_app USE LAYER {layer} WRITE LAYER {layer} \
             CREATE (:Document {{id: '{layer}', body: 'Layer document'}})"
        )))?;
        let result = database.query(Query::new(format!(
            "USE package_app USE LAYER {layer} MATCH (d:Document) RETURN d.id"
        )))?;
        assert!(
            matches!(result.rows.as_slice(), [row] if matches!(row.as_slice(), [TypedValue::String(id)] if id == layer))
        );
    }
    assert!(
        database
            .query(Query::new("USE package_other MATCH (n) RETURN n"))?
            .rows
            .is_empty()
    );
    database.query(Query::new(
        "USE package_app CREATE (:Sensor {temperature: 20.0})",
    ))?;
    database.query(Query::new(
        "USE package_app ALTER NODE PROPERTY Sensor.temperature SET TEMPORAL FLOAT RETENTION duration('P7D')",
    ))?;
    database.query(Query::new(
        "USE package_app MATCH (s:Sensor) SET s.temperature = 21.0 AT TIME datetime()",
    ))?;
    let historical = database.query(Query::new(
        "USE package_app AT TIME datetime() MATCH (s:Sensor) RETURN s.temperature",
    ))?;
    assert!(
        matches!(historical.rows.as_slice(), [row] if matches!(row.as_slice(), [TypedValue::Float(value)] if *value == 21.0))
    );
    // A failing statement cannot leave the writes that preceded its failing expression.
    assert!(
        database
            .query(Query::new(
                "USE package_app CREATE (:RollbackProbe) WITH 1 / 0 AS value RETURN value",
            ))
            .is_err()
    );
    assert!(
        database
            .query(Query::new(
                "USE package_app MATCH (n:RollbackProbe) RETURN n"
            ))?
            .rows
            .is_empty()
    );
    for statement in [
        "CREATE TOPIC activity PARTITIONS 2 RETENTION 7 DAYS",
        "CREATE EXCHANGE routing TYPE TOPIC",
        "CREATE QUEUE jobs STREAM RETENTION 7 DAYS",
        "BIND QUEUE jobs TO EXCHANGE routing KEY documents",
    ] {
        database.query(Query::new(format!("USE package_app {statement}")))?;
    }
    for statement in [
        "SHOW TOPICS",
        "SHOW QUEUES",
        "SHOW EXCHANGES",
        "SHOW INDEXES",
    ] {
        assert!(
            !database
                .query(Query::new(format!("USE package_app {statement}")))?
                .rows
                .is_empty(),
            "{statement}"
        );
    }
    database.query(Query::new("USE package_app SHOW CONSUMER LAG"))?;
    database.snapshot()?;
    database.close()?;

    let reopened = EmbeddedDatabase::open(options())?;
    let document = reopened.query(Query::new(
        "USE package_app MATCH (d:Document {id: 'first'}) RETURN d.body",
    ))?;
    assert!(
        matches!(document.rows.as_slice(), [row] if matches!(row.as_slice(), [TypedValue::String(body)] if body == "Revised body"))
    );
    for statement in [
        "SHOW TOPICS",
        "SHOW QUEUES",
        "SHOW EXCHANGES",
        "SHOW INDEXES",
    ] {
        assert!(
            !reopened
                .query(Query::new(format!("USE package_app {statement}")))?
                .rows
                .is_empty(),
            "persisted {statement}"
        );
    }
    reopened.query(Query::new(
        "USE package_app MATCH (d:Document {id: 'first'}) DETACH DELETE d",
    ))?;
    reopened.close()?;

    let deleted = EmbeddedDatabase::open(options())?;
    assert!(
        deleted
            .query(Query::new(
                "USE package_app MATCH (d:Document {id: 'first'}) RETURN d",
            ))?
            .rows
            .is_empty()
    );
    deleted.close()?;
    Ok(())
}
