//! Release qualification through the real embedded query and persistence paths.
use irongraph_client::Query;
use irongraph_embedded::{EmbeddedDatabase, EmbeddedOptions, EmbeddingPolicy};
use irongraph_server::protocol::TypedValue;
use serde_json::json;
use std::{
    io::{BufRead, Write},
    path::Path,
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error + Send + Sync>>;
const CRASH_DIRECTORY: &str = "IRONGRAPH_STRESS_CRASH_DIRECTORY";
const DURABLE_REVISION: i64 = 999_999;

struct StopReaders(Arc<AtomicBool>);
impl Drop for StopReaders {
    fn drop(&mut self) {
        self.0.store(true, Ordering::Release);
    }
}

fn open(path: &Path) -> Result<EmbeddedDatabase> {
    Ok(EmbeddedDatabase::open(
        EmbeddedOptions::new(path).with_embedding_policy(EmbeddingPolicy::Disabled),
    )?)
}

fn integer(value: &TypedValue) -> Result<i64> {
    match value {
        TypedValue::Integer(value) => Ok(value.parse()?),
        _ => Err(format!("expected integer, got {value:?}").into()),
    }
}

fn emit(event: serde_json::Value) -> Result<()> {
    println!("{event}");
    std::io::stdout().flush()?;
    Ok(())
}

fn timings(mut values: Vec<f64>) -> serde_json::Value {
    values.sort_by(f64::total_cmp);
    let at = |fraction: usize| {
        values
            .get(values.len().saturating_sub(1) * fraction / 100)
            .copied()
    };
    json!({"count":values.len(),"p50_ms":at(50),"p95_ms":at(95),"p99_ms":at(99),"max_ms":values.last()})
}

fn phase(name: &str, started: Instant, extra: serde_json::Value) -> Result<()> {
    emit(json!({"stage":name,"ms":started.elapsed().as_secs_f64()*1000.0,"details":extra}))
}

fn crash_child(path: &Path) -> Result<()> {
    let db = open(path)?;
    let write = |revision| -> Result<()> {
        db.query(
            Query::new(
                "USE stress UNWIND [0,1] AS id MATCH (n:Stress {id:id}) SET n.revision=$revision",
            )
            .with_parameter("revision", json!(revision)),
        )?;
        Ok(())
    };
    write(DURABLE_REVISION)?;
    db.flush()?;
    emit(json!({"event":"crash_durable_ready"}))?;
    for revision in DURABLE_REVISION + 1..=DURABLE_REVISION + 10_000 {
        write(revision)?;
    }
    loop {
        std::thread::park();
    }
}

#[test]
#[ignore = "release persistence stress with concurrent queries and abrupt-process recovery"]
fn persistence_stress() -> Result<()> {
    if let Some(path) = std::env::var_os(CRASH_DIRECTORY) {
        return crash_child(Path::new(&path));
    }
    let nodes: usize = std::env::var("IG_STRESS_NODES")
        .unwrap_or_else(|_| "20000".into())
        .parse()?;
    let transactions: usize = std::env::var("IG_STRESS_TRANSACTIONS")
        .unwrap_or_else(|_| "2000".into())
        .parse()?;
    let flush_interval_ms: u64 = std::env::var("IG_STRESS_FLUSH_INTERVAL_MS")
        .unwrap_or_else(|_| "0".into())
        .parse()?;
    assert!(nodes >= 1_000 && nodes.is_multiple_of(4));
    assert!(transactions > 0);
    let directory = tempfile::tempdir()?;
    emit(
        json!({"event":"configuration","pid":std::process::id(),"directory":directory.path(),
        "nodes":nodes,"body_bytes":4096,"writers":4,"readers":2,"transactions_per_writer":transactions,
        "embedding":false,"profile":"release","forced_flush_interval_ms":flush_interval_ms}),
    )?;
    let options_path = directory.path();
    let db = Arc::new(open(options_path)?);
    db.query(Query::new("CREATE PROJECT stress"))?;
    let start = Instant::now();
    for first in (0..nodes).step_by(512) {
        let last = (first + 512).min(nodes) - 1;
        db.query(Query::new("USE stress UNWIND range($first,$last) AS id CREATE (:Stress {id:id, revision:0, body:$body})")
            .with_parameter("first", json!(first)).with_parameter("last", json!(last))
            .with_parameter("body", json!("x".repeat(4096))))?;
    }
    db.query(Query::new(
        "USE stress CREATE INDEX stress_id FOR (n:Stress) ON (n.id)",
    ))?;
    for first in (0..nodes).step_by(512) {
        db.query(Query::new("USE stress UNWIND range($first,$last) AS id MATCH (a:Stress {id:id}), (b:Stress {id:(id+1)%$nodes}) CREATE (a)-[:NEXT]->(b)")
            .with_parameter("first",json!(first)).with_parameter("last",json!((first+512).min(nodes)-1))
            .with_parameter("nodes",json!(nodes)))?;
    }
    db.flush()?;
    phase(
        "ingest_and_durable_flush",
        start,
        json!({"nodes":nodes,"relationships":nodes}),
    )?;
    let initial_file = std::fs::metadata(options_path.join("database/database.redb"))?.len();
    emit(json!({"event":"initial_file","bytes":initial_file}))?;
    let database_file = options_path.join("database/database.redb");
    let idle_modified = std::fs::metadata(&database_file)?.modified()?;
    emit(json!({"event":"idle_begin","duration_ms":1200}))?;
    let idle_started = Instant::now();
    std::thread::sleep(Duration::from_millis(1200));
    assert_eq!(
        std::fs::metadata(&database_file)?.modified()?,
        idle_modified
    );
    phase(
        "idle_no_work",
        idle_started,
        json!({"database_file_unchanged":true}),
    )?;

    let stop = Arc::new(AtomicBool::new(false));
    let _stop_on_failure = StopReaders(Arc::clone(&stop));
    let accepted = Arc::new(AtomicUsize::new(0));
    let mut readers = Vec::new();
    for _ in 0..2 {
        let db = Arc::clone(&db);
        let stop = Arc::clone(&stop);
        readers.push(std::thread::spawn(move || -> Result<Vec<f64>> {
            let mut values = Vec::new();
            while !stop.load(Ordering::Acquire) {
                let start = Instant::now();
                let row = db.query(Query::new(
                    "USE stress MATCH (n:Stress {id:0}) RETURN n.id, n.revision, size(n.body)",
                ))?;
                assert_eq!(row.rows.len(), 1);
                assert_eq!(integer(&row.rows[0][0])?, 0);
                assert_eq!(integer(&row.rows[0][2])?, 4096);
                values.push(start.elapsed().as_secs_f64() * 1000.0);
                std::thread::sleep(Duration::from_millis(2));
            }
            Ok(values)
        }));
    }
    let flush_db = Arc::clone(&db);
    let flush_stop = Arc::clone(&stop);
    let flushed_accepted = Arc::clone(&accepted);
    let flusher = std::thread::spawn(move || -> Result<Vec<f64>> {
        let mut values = Vec::new();
        if flush_interval_ms == 0 {
            return Ok(values);
        }
        while !flush_stop.load(Ordering::Acquire) {
            let before = flushed_accepted.load(Ordering::Acquire);
            let start = Instant::now();
            flush_db.flush()?;
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            values.push(ms);
            emit(json!({"event":"flush_progress","accepted_before":before,
                "accepted_after":flushed_accepted.load(Ordering::Acquire),"ms":ms}))?;
            std::thread::sleep(Duration::from_millis(flush_interval_ms));
        }
        Ok(values)
    });
    let start = Instant::now();
    let mut writers = Vec::new();
    for writer in 0..4 {
        let db = Arc::clone(&db);
        let accepted = Arc::clone(&accepted);
        writers.push(std::thread::spawn(move || -> Result<Vec<f64>> {
            let mut values = Vec::new();
            for sequence in 0..transactions {
                let id = writer + (sequence % (nodes / 4)) * 4;
                let revision = sequence / (nodes / 4) + 1;
                let start = Instant::now();
                db.query(
                    Query::new("USE stress MATCH (n:Stress {id:$id}) SET n.revision=$revision")
                        .with_parameter("id", json!(id))
                        .with_parameter("revision", json!(revision)),
                )?;
                accepted.fetch_add(1, Ordering::Release);
                values.push(start.elapsed().as_secs_f64() * 1000.0);
            }
            Ok(values)
        }));
    }
    let mut acknowledgements = Vec::new();
    let mut failure = None;
    for writer in writers {
        match writer.join().map_err(|_| "writer panicked")? {
            Ok(values) => acknowledgements.extend(values),
            Err(error) => {
                failure.get_or_insert(error);
            }
        }
    }
    stop.store(true, Ordering::Release);
    let mut reads = Vec::new();
    for reader in readers {
        reads.extend(reader.join().map_err(|_| "reader panicked")??);
    }
    let flushes = flusher.join().map_err(|_| "flusher panicked")??;
    if let Some(error) = failure {
        return Err(error);
    }
    let tail = Instant::now();
    db.flush()?;
    let tail_ms = tail.elapsed().as_secs_f64() * 1000.0;
    phase(
        "concurrent_small_updates",
        start,
        json!({"transactions":4*transactions,
        "acknowledgements":timings(acknowledgements),"reads":timings(reads),"flushes":timings(flushes),"tail_flush_ms":tail_ms}),
    )?;
    let small_update_file = std::fs::metadata(options_path.join("database/database.redb"))?.len();
    emit(
        json!({"event":"small_update_file","bytes":small_update_file,"growth_bytes":small_update_file.saturating_sub(initial_file)}),
    )?;

    let start = Instant::now();
    let mut last_body = String::new();
    for version in 0..32 {
        let mut body = format!("large durable value {version:08}:");
        body.push_str(&"abcdefgh01234567".repeat(65_536));
        db.query(
            Query::new("USE stress MATCH (n:Stress {id:0}) SET n.body=$body")
                .with_parameter("body", json!(body)),
        )?;
        last_body = body;
    }
    db.flush()?;
    phase(
        "large_value_rewrites",
        start,
        json!({"rewrites":32,"bytes_per_value":last_body.len()}),
    )?;
    db.query(Query::new(
        "USE stress MATCH (n:Stress) WHERE n.id>=100 AND n.id<200 DETACH DELETE n",
    ))?;
    db.query(Query::new("USE stress UNWIND range(100,199) AS id CREATE (:Stress {id:id,revision:-1,body:$body})")
        .with_parameter("body",json!("x".repeat(4096))))?;
    let start = Instant::now();
    Arc::try_unwrap(db)
        .map_err(|_| "database still borrowed")?
        .close()?;
    phase("close", start, json!({}))?;
    let start = Instant::now();
    let db = open(options_path)?;
    phase("reopen", start, json!({}))?;
    let rows = db.query(Query::new(
        "USE stress MATCH (n:Stress) RETURN n.id,n.revision,n.body",
    ))?;
    assert_eq!(rows.rows.len(), nodes);
    let mut seen = vec![false; nodes];
    let unchanged_body = "x".repeat(4096);
    for row in &rows.rows {
        let id = usize::try_from(integer(&row[0])?)?;
        assert!(id < nodes && !seen[id]);
        seen[id] = true;
        let local = id / 4;
        let expected = if (100..200).contains(&id) {
            -1
        } else if local < transactions {
            i64::try_from((transactions - 1 - local) / (nodes / 4) + 1)?
        } else {
            0
        };
        assert_eq!(integer(&row[1])?, expected, "revision for key {id}");
        let TypedValue::String(body) = &row[2] else {
            return Err("body changed type".into());
        };
        assert_eq!(body, if id == 0 { &last_body } else { &unchanged_body });
    }
    let edges = db.query(Query::new(
        "USE stress MATCH ()-[r:NEXT]->() RETURN count(r)",
    ))?;
    assert_eq!(integer(&edges.rows[0][0])?, i64::try_from(nodes - 101)?);
    db.close()?;
    emit(json!({"event":"clean_recovery_verified","nodes":nodes,"relationships":nodes-101}))?;

    let mut child = std::process::Command::new(std::env::current_exe()?)
        .args(["--exact", "persistence_stress", "--ignored", "--nocapture"])
        .env(CRASH_DIRECTORY, options_path)
        .stdout(std::process::Stdio::piped())
        .spawn()?;
    let output = child.stdout.take().ok_or("child stdout missing")?;
    let ready = std::io::BufReader::new(output)
        .lines()
        .any(|line| line.is_ok_and(|line| line.contains("crash_durable_ready")));
    if !ready {
        let status = child.wait()?;
        return Err(format!("child failed before flush: {status}").into());
    }
    std::thread::sleep(Duration::from_millis(25));
    child.kill()?;
    assert!(
        !child.wait()?.success(),
        "crash child must terminate abruptly"
    );
    let start = Instant::now();
    let db = open(options_path)?;
    let recovered = db.query(Query::new(
        "USE stress MATCH (n:Stress) WHERE n.id=0 OR n.id=1 RETURN n.revision",
    ))?;
    assert_eq!(recovered.rows.len(), 2);
    let revision = integer(&recovered.rows[0][0])?;
    assert_eq!(revision, integer(&recovered.rows[1][0])?);
    assert!((DURABLE_REVISION..=DURABLE_REVISION + 10_000).contains(&revision));
    db.close()?;
    phase(
        "crash_recovery",
        start,
        json!({"consistent_revision":revision,"minimum_durable_revision":DURABLE_REVISION}),
    )?;
    assert_eq!(std::fs::read_dir(options_path.join("database"))?.count(), 1);
    emit(
        json!({"event":"final_file","bytes":std::fs::metadata(options_path.join("database/database.redb"))?.len()}),
    )?;
    emit(json!({"event":"PERSISTENCE_STRESS_OK"}))?;
    Ok(())
}
