use std::collections::BTreeMap;

use irongraph_client::Query;
use irongraph_embedded::{
    EmbeddedDatabase, EmbeddedOptions, EmbeddingPolicy, ExecutionDevice, OperationOptions,
    StreamAppend, StreamFetch, StreamRecord,
};
use irongraph_server::protocol::TypedValue;
use irongraph_types::ProjectId;

static NATIVE_TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
fn metal_empty_label_after_delete_and_restart() -> Result<(), Box<dyn std::error::Error>> {
    let _guard = NATIVE_TESTS.lock().unwrap();
    let directory = tempfile::tempdir()?;
    let options = EmbeddedOptions::new(directory.path())
        .with_execution_device(ExecutionDevice::Metal(0))
        .with_embedding_policy(EmbeddingPolicy::Disabled);
    for round in 0..2 {
        let database = EmbeddedDatabase::open(options.clone())?;
        if round == 0 {
            database.query(Query::new("CREATE PROJECT empty_label"))?;
        }
        for statement in [
            "USE empty_label MATCH (n:DocumentChange) RETURN n",
            "USE empty_label MATCH (n) WHERE $label IN labels(n) RETURN n",
        ] {
            assert!(
                database
                    .query(Query::new(statement).with_parameter("label", "DocumentChange".into()))?
                    .rows
                    .is_empty()
            );
        }
        if round == 0 {
            database.query(Query::new(
                "USE empty_label CREATE (:DocumentChange {body: 'complete'})",
            ))?;
            database.query(Query::new(
                "USE empty_label MATCH (n:DocumentChange) DETACH DELETE n",
            ))?;
            assert!(
                database
                    .query(Query::new(
                        "USE empty_label MATCH (n:DocumentChange) RETURN n"
                    ))?
                    .rows
                    .is_empty()
            );
        }
        database.close()?;
    }
    Ok(())
}

fn project(
    database: &EmbeddedDatabase,
    name: &str,
) -> Result<ProjectId, Box<dyn std::error::Error>> {
    database.query(Query::new(format!("CREATE PROJECT {name}")))?;
    let result = database.query(Query::new(format!("USE {name} RETURN 1")))?;
    result
        .catalog
        .and_then(|catalog| catalog.project_id)
        .ok_or_else(|| "project ID absent".into())
}

#[test]
fn native_graph_stream_bounds_isolation_and_reopen() -> Result<(), Box<dyn std::error::Error>> {
    let _guard = NATIVE_TESTS.lock().unwrap();
    let directory = tempfile::tempdir()?;
    let mut options = EmbeddedOptions::new(directory.path())
        .with_execution_device(ExecutionDevice::Cpu)
        .with_embedding_policy(EmbeddingPolicy::Disabled);
    options.worker_threads = 1;
    options.max_concurrent_operations = 2;
    let database = EmbeddedDatabase::open(options.clone())?;
    assert_eq!(database.status()?.data_dir, directory.path());
    assert!(database.status()?.ready);
    assert!(EmbeddedDatabase::open(options.clone()).is_err());
    let first = project(&database, "first")?;
    let second = project(&database, "second")?;
    database.query(
        Query::new("CHECK READ ONLY")
            .with_project(first)
            .with_parameter("statement", "MATCH (n) RETURN n".into()),
    )?;
    assert!(
        database
            .query(
                Query::new("CHECK READ ONLY")
                    .with_project(first)
                    .with_parameter("statement", "CREATE (:Rejected)".into())
            )
            .is_err()
    );
    for id in [first, second] {
        database.query(Query::new("CREATE TOPIC events PARTITIONS 1").with_project(id))?;
        for query in [
            "MATCH (n:DocumentChange) RETURN n",
            "MATCH (n) WHERE $label IN labels(n) RETURN n",
        ] {
            assert!(
                database
                    .query(
                        Query::new(query)
                            .with_project(id)
                            .with_parameter("label", "DocumentChange".into())
                    )?
                    .rows
                    .is_empty()
            );
        }
    }
    let written = database.query(
        Query::new("CREATE (:DocumentChange {body: $body, count: $count})")
            .with_project(first)
            .with_parameter("body", "complete source".into())
            .with_parameter("count", 9007199254740993i64.into()),
    )?;
    let mut read =
        Query::new("MATCH (d:DocumentChange) RETURN d.body, d.count").with_project(first);
    read.bookmark = written.summary.bookmark;
    let result = database.query(read)?;
    assert!(
        matches!(&result.rows[0][1], TypedValue::Integer(value) if value == "9007199254740993")
    );
    let record = StreamRecord {
        key: Some(vec![0, 255]),
        headers: BTreeMap::from([("source".into(), b"native".to_vec())]),
        value: Some(b"distinct record".to_vec()),
        create_time_ms: Some(1234),
    };
    let append = StreamAppend {
        project_id: first,
        topic: "events".into(),
        partition: 0,
        records: vec![record.clone()],
    };
    let ack = database.stream_append(append.clone(), OperationOptions::default())?;
    assert_eq!((ack.first_offset, ack.record_count), (0, 1));
    let fetch = StreamFetch {
        project_id: first,
        topic: "events".into(),
        partition: 0,
        offset: 0,
        max_records: 1,
        max_bytes: 4096,
    };
    let page = database.stream_fetch(fetch.clone(), OperationOptions::default())?;
    assert_eq!(page.high_watermark, 1);
    assert_eq!(&*page.records[0].1.payload, b"distinct record");
    let identity = page.records[0].1.id;
    assert!(!page.truncated);
    assert!(
        database
            .stream_fetch(
                StreamFetch {
                    project_id: second,
                    ..fetch.clone()
                },
                OperationOptions::default()
            )?
            .records
            .is_empty()
    );
    assert!(
        database
            .stream_fetch(
                StreamFetch {
                    max_bytes: 1,
                    ..fetch.clone()
                },
                OperationOptions::default()
            )
            .is_err()
    );
    assert!(
        database
            .stream_append(
                append.clone(),
                OperationOptions {
                    timeout_ms: Some(0),
                    ..Default::default()
                }
            )
            .is_err()
    );
    assert_eq!(
        database
            .stream_fetch(fetch.clone(), OperationOptions::default())?
            .high_watermark,
        1
    );
    database.close()?;
    for round in 0..3 {
        let database = EmbeddedDatabase::open(options.clone())?;
        let page = database.stream_fetch(fetch.clone(), OperationOptions::default())?;
        assert_eq!(page.records[0].1.id, identity);
        assert_eq!(page.high_watermark, 1 + round);
        let ack = database.stream_append(append.clone(), OperationOptions::default())?;
        assert_eq!(ack.first_offset, 1 + round);
        database.close()?;
    }
    Ok(())
}

#[test]
fn crash_child() -> Result<(), Box<dyn std::error::Error>> {
    let Some(path) = std::env::var_os("IRONGRAPH_CRASH_TEST_DIRECTORY") else {
        return Ok(());
    };
    let options = EmbeddedOptions::new(path)
        .with_execution_device(ExecutionDevice::Cpu)
        .with_embedding_policy(EmbeddingPolicy::Disabled);
    if std::env::var_os("IRONGRAPH_LOCK_PROBE").is_some() {
        assert!(EmbeddedDatabase::open(options).is_err());
        return Ok(());
    }
    let database = EmbeddedDatabase::open(options)?;
    let id = project(&database, "crash")?;
    database.query(
        Query::new("CREATE (:Recovered {value: $value})")
            .with_project(id)
            .with_parameter("value", 42.into()),
    )?;
    database.query(Query::new("CREATE TOPIC recovered PARTITIONS 1").with_project(id))?;
    database.stream_append(
        StreamAppend {
            project_id: id,
            topic: "recovered".into(),
            partition: 0,
            records: vec![StreamRecord {
                key: None,
                headers: Default::default(),
                value: Some(vec![42]),
                create_time_ms: None,
            }],
        },
        Default::default(),
    )?;
    database.flush()?;
    // Exit without destructors, snapshots, or runtime close. Recovery must read the flushed WAL.
    std::process::exit(86);
}

#[test]
fn flushed_wal_survives_abrupt_host_exit_and_directory_lock_is_exclusive()
-> Result<(), Box<dyn std::error::Error>> {
    let _guard = NATIVE_TESTS.lock().unwrap();
    let directory = tempfile::tempdir()?;
    let executable = std::env::current_exe()?;
    let status = std::process::Command::new(&executable)
        .args(["--exact", "crash_child", "--nocapture"])
        .env("IRONGRAPH_CRASH_TEST_DIRECTORY", directory.path())
        .status()?;
    assert_eq!(status.code(), Some(86));
    let host_workload = vec![19u8; 32 * 1024 * 1024];
    let database = EmbeddedDatabase::open(
        EmbeddedOptions::new(directory.path())
            .with_execution_device(ExecutionDevice::Cpu)
            .with_embedding_policy(EmbeddingPolicy::Disabled),
    )?;
    let recovered = database.query(Query::new("USE crash MATCH (n:Recovered) RETURN n.value"))?;
    assert!(matches!(&recovered.rows[0][0], TypedValue::Integer(value) if value == "42"));
    let id = recovered
        .catalog
        .and_then(|catalog| catalog.project_id)
        .ok_or("missing project")?;
    let page = database.stream_fetch(
        StreamFetch {
            project_id: id,
            topic: "recovered".into(),
            partition: 0,
            offset: 0,
            max_records: 1,
            max_bytes: 4096,
        },
        Default::default(),
    )?;
    assert_eq!(page.high_watermark, 1);
    assert_eq!(&*page.records[0].1.payload, &[42]);
    assert!(
        std::process::Command::new(executable)
            .args(["--exact", "crash_child"])
            .env("IRONGRAPH_CRASH_TEST_DIRECTORY", directory.path())
            .env("IRONGRAPH_LOCK_PROBE", "1")
            .status()?
            .success()
    );
    database.close()?;
    assert!(host_workload.iter().all(|value| *value == 19));
    Ok(())
}
