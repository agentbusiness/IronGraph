use irongraph_client::Query;
use irongraph_embedded::{EmbeddedDatabase, EmbeddedOptions, EmbeddingPolicy, ExecutionDevice};
use irongraph_server::protocol::TypedValue;

fn test_guard() -> std::sync::MutexGuard<'static, ()> {
    static TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());
    TESTS
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

#[test]
fn bundled_fraud_dataset_makes_the_features_path_query_runnable()
-> Result<(), Box<dyn std::error::Error>> {
    let _guard = test_guard();
    let directory = tempfile::tempdir()?;
    let database = EmbeddedDatabase::open(
        EmbeddedOptions::new(directory.path())
            .with_execution_device(ExecutionDevice::Cpu)
            .with_embedding_policy(EmbeddingPolicy::Disabled),
    )?;

    database.query(Query::new("IMPORT DATASET fraud"))?;
    let result = database.query(Query::new(
        "USE fraud \
         MATCH path = (account:Account)-[:TRANSFERRED_TO*1..4]->(destination:Account) \
         WHERE account.id = 'acct-100' \
         RETURN path",
    ))?;
    assert!(!result.rows.is_empty());
    assert!(
        result
            .rows
            .iter()
            .all(|row| matches!(row.as_slice(), [TypedValue::Path(_)]))
    );
    database.close()?;
    Ok(())
}

#[test]
fn creations_add_created_at_unless_the_user_supplies_it() -> Result<(), Box<dyn std::error::Error>>
{
    let _guard = test_guard();
    let directory = tempfile::tempdir()?;
    let database = EmbeddedDatabase::open(
        EmbeddedOptions::new(directory.path())
            .with_execution_device(ExecutionDevice::Cpu)
            .with_embedding_policy(EmbeddingPolicy::Disabled),
    )?;
    database.query(Query::new("CREATE PROJECT app"))?;

    let before = i64::try_from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs(),
    )?;
    let first_result = database.query(Query::new(
        "USE app CREATE (probe:Probe) RETURN probe.created_at AS created_at",
    ))?;
    assert!(matches!(
        first_result.rows.as_slice(),
        [row] if matches!(
            row.as_slice(),
            [TypedValue::DateTime { seconds, timezone: Some(timezone), .. }]
                if *seconds >= before && timezone == "UTC"
        )
    ));
    database.query(Query::new(
        "USE app \
         CREATE (a:Item {id: 1, created_at: 123}), (b:Item {id: 2}) \
         CREATE (a)-[:LINK {created_at: 456}]->(b), (b)-[:LINK]->(a)",
    ))?;
    let after = i64::try_from(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs(),
    )?;

    let result = database.query(Query::new(
        "USE app \
         MATCH (a:Item {id: 1})-[supplied:LINK]->(b:Item {id: 2})-[generated:LINK]->(a) \
         RETURN a.created_at AS supplied_node, b.created_at AS generated_node, \
                supplied.created_at AS supplied_edge, generated.created_at AS generated_edge",
    ))?;
    assert_eq!(result.rows.len(), 1);
    assert!(matches!(
        result.rows[0].as_slice(),
        [
            TypedValue::Integer(node) ,
            TypedValue::DateTime { seconds: node_seconds, timezone: Some(node_timezone), .. },
            TypedValue::Integer(edge),
            TypedValue::DateTime { seconds: edge_seconds, timezone: Some(edge_timezone), .. },
        ] if node == "123"
            && edge == "456"
            && (before..=after).contains(node_seconds)
            && (before..=after).contains(edge_seconds)
            && node_timezone == "UTC"
            && edge_timezone == "UTC"
    ));
    database.close()?;
    Ok(())
}

#[test]
fn caller_selected_directory_persists_across_clean_reopen() -> Result<(), Box<dyn std::error::Error>>
{
    let _guard = test_guard();
    let directory = tempfile::tempdir()?;
    let options = || {
        EmbeddedOptions::new(directory.path())
            .with_execution_device(ExecutionDevice::Cpu)
            .with_embedding_policy(EmbeddingPolicy::Disabled)
    };

    let database = EmbeddedDatabase::open(options())?;
    database.query(Query::new("CREATE PROJECT app"))?;
    database.query(Query::new("USE app CREATE (:Item {value: 42})"))?;
    database.close()?;

    let reopened = EmbeddedDatabase::open(options())?;
    let result = reopened.query(Query::new("USE app MATCH (n:Item) RETURN n.value AS value"))?;
    assert_eq!(result.rows.len(), 1);
    assert!(matches!(
        result.rows[0].as_slice(),
        [TypedValue::Integer(value)] if value == "42"
    ));
    reopened.close()?;
    Ok(())
}

#[test]
fn snapshot_maintenance_compacts_the_standalone_wal() -> Result<(), Box<dyn std::error::Error>> {
    let _guard = test_guard();
    let directory = tempfile::tempdir()?;
    let interval = std::time::Duration::from_millis(250);
    let options = || {
        let mut options = EmbeddedOptions::new(directory.path())
            .with_execution_device(ExecutionDevice::Cpu)
            .with_embedding_policy(EmbeddingPolicy::Disabled);
        options.snapshot_interval = interval;
        options
    };
    let wal_path = directory.path().join("write").join("standalone.wal");

    let database = EmbeddedDatabase::open(options())?;
    database.query(Query::new("CREATE PROJECT app"))?;

    // Several write bursts spread across snapshot intervals; retaining every burst would keep the
    // WAL at least as large as the total payload.
    let payload = "x".repeat(2_048);
    let mut total_payload_bytes = 0u64;
    for round in 0..3u32 {
        for item in 0..50u32 {
            database.query(Query::new(format!(
                "USE app CREATE (:Item {{round: {round}, item: {item}, payload: '{payload}'}})"
            )))?;
            total_payload_bytes += payload.len() as u64;
        }
        std::thread::sleep(interval * 2);
    }

    // A snapshot at the latest bookmark plus one further write lets the next maintenance tick
    // compact the WAL through that snapshot, whatever the earlier snapshot cadence was.
    database.snapshot()?;
    database.query(Query::new("USE app CREATE (:Item {round: 3, item: 0})"))?;

    let bound = total_payload_bytes / 4;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
    let mut wal_bytes = u64::MAX;
    while std::time::Instant::now() < deadline {
        wal_bytes = std::fs::metadata(&wal_path)?.len();
        if wal_bytes < bound {
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(50));
    }
    assert!(
        wal_bytes < bound,
        "standalone WAL held {wal_bytes} bytes after {total_payload_bytes} payload bytes and repeated snapshots"
    );
    database.close()?;

    // Compaction may only drop entries the retained snapshots already cover.
    let reopened = EmbeddedDatabase::open(options())?;
    let result = reopened.query(Query::new("USE app MATCH (n:Item) RETURN n.item AS item"))?;
    assert_eq!(result.rows.len(), 151);
    reopened.close()?;
    Ok(())
}

#[test]
fn one_process_cannot_open_two_embedded_nodes() -> Result<(), Box<dyn std::error::Error>> {
    let _guard = test_guard();
    let first_directory = tempfile::tempdir()?;
    let second_directory = tempfile::tempdir()?;
    let options = |path: &std::path::Path| {
        EmbeddedOptions::new(path)
            .with_execution_device(ExecutionDevice::Cpu)
            .with_embedding_policy(EmbeddingPolicy::Disabled)
    };
    let first = EmbeddedDatabase::open(options(first_directory.path()))?;
    assert!(EmbeddedDatabase::open(options(second_directory.path())).is_err());
    first.close()?;
    Ok(())
}
