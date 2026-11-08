//! Abrupt termination applies only to child processes owning disposable databases.
use irongraph_client::Query;
use irongraph_embedded::{
    EmbeddedDatabase, EmbeddedOptions, EmbeddingPolicy, OperationOptions, StreamAppend,
    StreamFetch, StreamRecord,
};
use irongraph_server::protocol::TypedValue;
use serde_json::json;
use std::{
    path::Path,
    process::{Child, Command, Stdio},
    time::{Duration, Instant},
};
type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;

fn open(path: &Path) -> Result<EmbeddedDatabase> {
    Ok(EmbeddedDatabase::open(
        EmbeddedOptions::new(path).with_embedding_policy(EmbeddingPolicy::Disabled),
    )?)
}

struct OwnedChild(Child);
impl Drop for OwnedChild {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn wait_marker(child: &mut OwnedChild, path: &Path, name: &str) -> Result<()> {
    let deadline = Instant::now() + Duration::from_secs(30);
    while !path.join(name).exists() {
        if let Some(status) = child.0.try_wait()? {
            return Err(format!("child exited before {name}: {status}").into());
        }
        if Instant::now() >= deadline {
            return Err(format!("child missed {name} deadline").into());
        }
        std::thread::sleep(Duration::from_millis(1));
    }
    Ok(())
}

#[test]
#[ignore = "helper invoked only by the durability scorecard"]
fn durability_crash_child() -> Result<()> {
    let Some(directory) = std::env::var_os("IG_SCORECARD_CRASH_CHILD") else {
        return Ok(());
    };
    let path = Path::new(&directory);
    let db = open(path)?;
    db.query(Query::new("CREATE PROJECT durable"))?;
    db.query(Query::new("USE durable CREATE (a:Record {id:1,version:0,body:$body}), (b:Record {id:2,version:0}), (a)-[:LINK {marker:'flushed'}]->(b)").with_parameter("body",json!("durable complete text ".repeat(4096))))?;
    db.query(Query::new(
        "USE durable CREATE INDEX record_id FOR (n:Record) ON (n.id)",
    ))?;
    db.query(Query::new("USE durable MATCH (n:Record) SET n.version=100"))?;
    for statement in [
        "CREATE (:Sensor {temperature:20.0})",
        "ALTER NODE PROPERTY Sensor.temperature SET TEMPORAL FLOAT RETENTION duration('P7D')",
        "MATCH (s:Sensor) SET s.temperature=21.0 AT TIME datetime()",
        "CREATE TOPIC events PARTITIONS 1",
        "CREATE EXCHANGE routing TYPE TOPIC",
        "CREATE QUEUE jobs STREAM RETENTION 7 DAYS",
        "BIND QUEUE jobs TO EXCHANGE routing KEY documents",
    ] {
        db.query(Query::new(format!("USE durable {statement}")))?;
    }
    let project = db
        .query(Query::new("USE durable RETURN 1"))?
        .catalog
        .and_then(|catalog| catalog.project_id)
        .ok_or("project identity absent")?;
    db.stream_append(
        StreamAppend {
            project_id: project,
            topic: "events".into(),
            partition: 0,
            records: vec![StreamRecord {
                key: None,
                headers: Default::default(),
                value: Some(b"flushed event".to_vec()),
                create_time_ms: None,
            }],
        },
        OperationOptions::default(),
    )?;
    db.flush()?;
    std::fs::write(path.join("flushed-ready"), b"100")?;
    let phase: usize = std::env::var("IG_CRASH_PHASE")
        .unwrap_or_else(|_| "0".into())
        .parse()?;
    let first = if matches!(phase, 2 | 3) {
        db.query(Query::new("USE durable MATCH (n:Record) SET n.version=101"))?;
        std::fs::write(path.join("flush-entered"), b"101")?;
        db.flush()?;
        std::fs::write(path.join("flushed-next"), b"101")?;
        102
    } else {
        101
    };
    for revision in first..10_000 {
        db.query(
            Query::new("USE durable MATCH (n:Record) SET n.version=$revision")
                .with_parameter("revision", json!(revision)),
        )?;
    }
    loop {
        std::thread::park();
    }
}

#[test]
#[ignore = "100 owned child-process crash schedules; no active database is touched"]
fn durability_scorecard() -> Result<()> {
    let rounds: usize = std::env::var("IG_DURABILITY_ROUNDS")
        .unwrap_or_else(|_| "100".into())
        .parse()?;
    assert!(rounds > 0);
    let mut latencies = Vec::new();
    for seed in 0..rounds {
        let directory = tempfile::tempdir()?;
        let mut child = OwnedChild(
            Command::new(std::env::current_exe()?)
                .args([
                    "--exact",
                    "durability_crash_child",
                    "--ignored",
                    "--nocapture",
                ])
                .env("IG_SCORECARD_CRASH_CHILD", directory.path())
                .env("IG_CRASH_PHASE", (seed % 5).to_string())
                .stdout(Stdio::null())
                .stderr(Stdio::inherit())
                .spawn()?,
        );
        wait_marker(&mut child, directory.path(), "flushed-ready")?;
        match seed % 5 {
            0 => {}
            1 => std::thread::sleep(Duration::from_millis((seed as u64 * 17) % 31)),
            2 => wait_marker(&mut child, directory.path(), "flush-entered")?,
            3 => wait_marker(&mut child, directory.path(), "flushed-next")?,
            _ => std::thread::sleep(Duration::from_millis(1050)),
        }
        child.0.kill()?;
        assert!(!child.0.wait()?.success());
        let started = Instant::now();
        let db = open(directory.path())?;
        // The equality predicate exercises recovered scalar postings, not only a raw scan.
        let mut versions = Vec::new();
        for id in [1, 2] {
            let rows = db.query(
                Query::new("USE durable MATCH (n:Record {id:$id}) RETURN n.version")
                    .with_parameter("id", json!(id)),
            )?;
            let [row] = rows.rows.as_slice() else {
                return Err("flushed owner or index missing".into());
            };
            let Some(TypedValue::Integer(value)) = row.first() else {
                return Err("revision type changed".into());
            };
            let version = value.parse::<i64>()?;
            let minimum = if seed % 5 == 3 { 101 } else { 100 };
            assert!(
                (minimum..10_000).contains(&version),
                "acknowledged flush lost"
            );
            versions.push(version);
        }
        assert_eq!(
            versions[0], versions[1],
            "partial record batch survived recovery"
        );
        let bodies = db.query(Query::new(
            "USE durable MATCH (n:Record {id:1}) RETURN n.body",
        ))?;
        assert!(
            matches!(&bodies.rows[0][0],TypedValue::String(body) if body == &"durable complete text ".repeat(4096))
        );
        let edge = db.query(Query::new(
            "USE durable MATCH (:Record {id:1})-[r:LINK]->(:Record {id:2}) RETURN r.marker",
        ))?;
        assert!(matches!(&edge.rows[0][0],TypedValue::String(value) if value == "flushed"));
        let temporal = db.query(Query::new(
            "USE durable AT TIME datetime() MATCH (s:Sensor) RETURN s.temperature",
        ))?;
        assert!(matches!(&temporal.rows[0][0],TypedValue::Float(value) if *value==21.0));
        for (statement, name) in [
            ("SHOW TOPICS", "events"),
            ("SHOW QUEUES", "jobs"),
            ("SHOW EXCHANGES", "routing"),
        ] {
            let rows = db.query(Query::new(format!("USE durable {statement}")))?;
            assert!(
                rows.rows
                    .iter()
                    .flatten()
                    .any(|value| matches!(value,TypedValue::String(value) if value==name))
            );
        }
        let project = db
            .query(Query::new("USE durable RETURN 1"))?
            .catalog
            .and_then(|catalog| catalog.project_id)
            .ok_or("recovered project identity absent")?;
        let page = db.stream_fetch(
            StreamFetch {
                project_id: project,
                topic: "events".into(),
                partition: 0,
                offset: 0,
                max_records: 1,
                max_bytes: 4096,
            },
            OperationOptions::default(),
        )?;
        assert_eq!(page.high_watermark, 1);
        assert_eq!(&*page.records[0].1.payload, b"flushed event");
        db.close()?;
        latencies.push(started.elapsed().as_secs_f64() * 1000.0);
    }
    latencies.sort_by(f64::total_cmp);
    println!(
        "{}",
        json!({"event":"durability_scorecard","rounds":rounds,"reopen_and_verified_queries_p50_ms":latencies[rounds/2],"p95_ms":latencies[(rounds*95).div_ceil(100)-1],"boundary":"flushed graph/relationship/body/scalar index/temporal value/broker descriptors/stream payload survive process kill; unflushed versions may survive; not hardware power failure"})
    );
    println!("DURABILITY_SCORECARD_OK");
    Ok(())
}
