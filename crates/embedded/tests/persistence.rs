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
fn batched_merge_duplicate_keys_and_property_updates_survive_reopen()
-> Result<(), Box<dyn std::error::Error>> {
    let _guard = test_guard();
    let directory = tempfile::tempdir()?;
    let options =
        || EmbeddedOptions::new(directory.path()).with_embedding_policy(EmbeddingPolicy::Disabled);
    let database = EmbeddedDatabase::open(options())?;
    database.query(Query::new("CREATE PROJECT batch"))?;
    database.query(Query::new(
        "USE batch CREATE (:Code:Sym:Function {sid:'existing', rank:0})",
    ))?;
    database.query(Query::new(
        "USE batch CREATE INDEX code_sid FOR (n:Code) ON (n.sid)",
    ))?;
    let body = "complete source text ".repeat(4096);
    let rows = serde_json::json!([
        {"sid":"new", "p":{"sid":"new","rank":1,"body":body}},
        {"sid":"existing", "p":{"rank":2,"body":body}},
        {"sid":"new", "p":{"rank":3,"body":body}}
    ]);
    database.query(
        Query::new(
            "USE batch UNWIND $rows AS r MERGE (n:Code:Sym:Function {sid:r.sid}) SET n += r.p",
        )
        .with_parameter("rows", rows),
    )?;
    database.query(Query::new(
        "USE batch MATCH (n:Code {sid:'existing'}) SET n.sid='renamed', n.rank=4",
    ))?;
    database.query(Query::new("USE batch UNWIND ['renamed','new','renamed'] AS sid MERGE (n:Code:Sym:Function {sid:sid}) SET n.checked=true"))?;
    let statement =
        "USE batch MATCH (n:Code) RETURN n.sid, n.rank, n.body, n.checked ORDER BY n.sid";
    let before = database.query(Query::new(statement))?;
    assert_eq!(before.rows.len(), 2);
    for row in &before.rows {
        assert!(matches!(&row[2], TypedValue::String(text) if text == &body));
        assert!(matches!(&row[3], TypedValue::Boolean(true)));
    }
    assert!(matches!(&before.rows[0][0], TypedValue::String(sid) if sid == "new"));
    assert!(matches!(&before.rows[0][1], TypedValue::Integer(rank) if rank == "3"));
    assert!(matches!(&before.rows[1][0], TypedValue::String(sid) if sid == "renamed"));
    assert!(matches!(&before.rows[1][1], TypedValue::Integer(rank) if rank == "4"));
    database.flush()?;
    database.close()?;
    let database = EmbeddedDatabase::open(options())?;
    let after = database.query(Query::new(statement))?;
    assert_eq!(
        serde_json::to_value(&before.rows)?,
        serde_json::to_value(&after.rows)?
    );
    let old = database.query(Query::new(
        "USE batch MATCH (n:Code {sid:'existing'}) RETURN n",
    ))?;
    assert!(old.rows.is_empty());
    database.close()?;
    Ok(())
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

fn assert_record_files_only(root: &std::path::Path) -> Result<(), Box<dyn std::error::Error>> {
    assert!(!root.join("write").exists());
    assert!(!root.join("standalone-snapshots").exists());
    let directory = root.join("database");
    let names = std::fs::read_dir(directory)?
        .map(|entry| entry.map(|entry| entry.file_name()))
        .collect::<std::io::Result<Vec<_>>>()?;
    assert_eq!(names, [std::ffi::OsString::from("database.redb")]);
    Ok(())
}

#[test]
fn current_records_restore_indexes_and_large_untouched_properties()
-> Result<(), Box<dyn std::error::Error>> {
    let _guard = test_guard();
    let directory = tempfile::tempdir()?;
    let options = || {
        EmbeddedOptions::new(directory.path())
            .with_execution_device(ExecutionDevice::Cpu)
            .with_embedding_policy(EmbeddingPolicy::Disabled)
    };
    let database = EmbeddedDatabase::open(options())?;
    database.query(Query::new("CREATE PROJECT app"))?;
    let body = "unaltered source text ".repeat(16_384);
    for id in 0..16 {
        database.query(Query::new(format!(
            "USE app CREATE (:Item {{id: {id}, revision: 0, body: '{body}'}})"
        )))?;
    }
    database.query(Query::new(
        "USE app CREATE INDEX item_id FOR (n:Item) ON (n.id)",
    ))?;
    database.flush()?;
    let initial = std::fs::metadata(directory.path().join("database/database.redb"))?.len();
    for revision in 1..=100 {
        database.query(Query::new(format!(
            "USE app MATCH (n:Item {{id: 0}}) SET n.revision = {revision}"
        )))?;
    }
    database.query(Query::new("USE app MATCH (n:Item {id: 1}) DETACH DELETE n"))?;
    database.query(Query::new(
        "USE app CREATE (:Item {id: 99, revision: 7, body: 'replacement'})",
    ))?;
    database.flush()?;
    let after = std::fs::metadata(directory.path().join("database/database.redb"))?.len();
    assert!(
        after < initial * 3,
        "tiny updates repeatedly expanded the complete persisted graph"
    );
    assert_record_files_only(directory.path())?;
    database.close()?;
    let database = EmbeddedDatabase::open(options())?;
    let result = database.query(Query::new(
        "USE app MATCH (n:Item {id: 0}) RETURN n.revision, n.body",
    ))?;
    assert!(
        matches!(result.rows[0].as_slice(), [TypedValue::Integer(revision), TypedValue::String(text)] if revision == "100" && text == &body)
    );
    let result = database.query(Query::new("USE app MATCH (n:Item {id: 1}) RETURN n"))?;
    assert!(result.rows.is_empty());
    let result = database.query(Query::new(
        "USE app MATCH (n:Item {id: 99}) RETURN n.revision",
    ))?;
    assert!(
        matches!(result.rows[0].as_slice(), [TypedValue::Integer(revision)] if revision == "7")
    );
    database.close()?;
    assert_record_files_only(directory.path())?;
    Ok(())
}

#[test]
fn durable_records_recover_after_killed_process_without_close()
-> Result<(), Box<dyn std::error::Error>> {
    use std::io::{BufRead, Write};
    const CHILD_DIRECTORY: &str = "IRONGRAPH_RECORD_CRASH_DIRECTORY";
    let options = |path: &std::path::Path| {
        EmbeddedOptions::new(path)
            .with_execution_device(ExecutionDevice::Cpu)
            .with_embedding_policy(EmbeddingPolicy::Disabled)
    };
    if let Some(directory) = std::env::var_os(CHILD_DIRECTORY) {
        let database = EmbeddedDatabase::open(options(std::path::Path::new(&directory)))?;
        database.query(Query::new("CREATE PROJECT app"))?;
        database.query(Query::new("USE app CREATE (:Item {value: 1})"))?;
        database.query(Query::new("USE app MATCH (n:Item) SET n.value = 2"))?;
        database.flush()?;
        println!("durable-records-ready");
        std::io::stdout().flush()?;
        loop {
            std::thread::park();
        }
    }
    let _guard = test_guard();
    let directory = tempfile::tempdir()?;
    let mut child = std::process::Command::new(std::env::current_exe()?)
        .args([
            "--exact",
            "durable_records_recover_after_killed_process_without_close",
            "--nocapture",
        ])
        .env(CHILD_DIRECTORY, directory.path())
        .stdout(std::process::Stdio::piped())
        .spawn()?;
    let output = child.stdout.take().ok_or("missing child output")?;
    let mut ready = false;
    for line in std::io::BufReader::new(output).lines() {
        if line?.contains("durable-records-ready") {
            ready = true;
            break;
        }
    }
    if !ready {
        let status = child.wait()?;
        return Err(format!("child exited before durability: {status}").into());
    }
    child.kill()?;
    child.wait()?;
    let database = EmbeddedDatabase::open(options(directory.path()))?;
    let result = database.query(Query::new("USE app MATCH (n:Item) RETURN n.value"))?;
    assert!(matches!(result.rows[0].as_slice(), [TypedValue::Integer(value)] if value == "2"));
    database.close()?;
    assert_record_files_only(directory.path())?;
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
