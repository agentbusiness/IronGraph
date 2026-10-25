//! Release-only reproduction of concurrent code graph ingestion, reads and checkpoints.
use irongraph_client::Query;
use irongraph_embedded::{EmbeddedDatabase, EmbeddedOptions, EmbeddingPolicy};
use irongraph_server::protocol::TypedValue;
use serde_json::json;
use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, Ordering},
    },
    time::{Duration, Instant},
};

fn integer(value: &TypedValue) -> i64 {
    match value {
        TypedValue::Integer(value) => value.parse().unwrap(),
        _ => panic!("expected integer: {value:?}"),
    }
}

fn record(stage: &str, started: Instant) {
    println!(
        "{}",
        json!({"stage":stage,"ms":started.elapsed().as_secs_f64()*1000.0})
    );
}

#[test]
fn empty_writes_preserve_bookmark_and_map_updates_skip_unchanged_values()
-> Result<(), Box<dyn std::error::Error>> {
    let directory = tempfile::tempdir()?;
    let database = EmbeddedDatabase::open(
        EmbeddedOptions::new(directory.path()).with_embedding_policy(EmbeddingPolicy::Disabled),
    )?;
    database.query(Query::new("CREATE PROJECT empty_write"))?;
    let created = database.query(Query::new("USE empty_write CREATE (:Code:Sym {sid:'one', summary:'complete source text', body_hash:'hash'})"))?;
    let bookmark = created.summary.bookmark;
    for statement in [
        "MATCH (n:Code) WHERE n.code IS NOT NULL REMOVE n.code",
        "MATCH (d:Description) DETACH DELETE d",
        "MATCH (n:Code {sid:'missing'}) SET n.summary = 'unused'",
        "MATCH (n:Code {sid:'one'}) SET n += {summary:'complete source text', body_hash:'hash'} RETURN n.sid",
    ] {
        let result = database.query(Query::new(format!("USE empty_write {statement}")))?;
        assert_eq!(
            result.summary.bookmark, bookmark,
            "empty statement advanced WAL: {statement}"
        );
        assert_eq!(result.summary.statistics.updates, 0);
        if statement.ends_with("RETURN n.sid") {
            assert_eq!(result.rows.len(), 1);
        }
    }
    let changed=database.query(Query::new("USE empty_write MATCH (n:Code {sid:'one'}) SET n += {summary:'complete source text', body_hash:'changed'} RETURN n.body_hash"))?;
    assert!(changed.summary.bookmark > bookmark);
    assert_eq!(changed.summary.statistics.updates, 1);
    std::thread::scope(|scope| -> Result<(), Box<dyn std::error::Error>> {
        let first = scope.spawn(|| {
            for _ in 0..4 {
                database.snapshot()?;
            }
            Ok::<_, irongraph_embedded::EmbeddedError>(())
        });
        let second = scope.spawn(|| {
            for _ in 0..4 {
                database.snapshot()?;
            }
            Ok::<_, irongraph_embedded::EmbeddedError>(())
        });
        for value in 0..64 {
            database.query(
                Query::new("USE empty_write MATCH (n:Code {sid:'one'}) SET n.updated_ms=$value")
                    .with_parameter("value", json!(value)),
            )?;
        }
        first.join().map_err(|_| "first snapshot panicked")??;
        second.join().map_err(|_| "second snapshot panicked")??;
        Ok(())
    })?;
    database.close()?;
    let database = EmbeddedDatabase::open(
        EmbeddedOptions::new(directory.path()).with_embedding_policy(EmbeddingPolicy::Disabled),
    )?;
    let result = database.query(Query::new(
        "USE empty_write MATCH (n:Code) RETURN n.body_hash",
    ))?;
    assert!(matches!(&result.rows[0][0],TypedValue::String(value) if value=="changed"));
    database.close()?;
    Ok(())
}

#[test]
#[ignore = "release qualification: code graph load with concurrent reads and memory sampling"]
fn code_graph_load() -> Result<(), Box<dyn std::error::Error>> {
    let nodes: usize = std::env::var("IG_LOAD_NODES")
        .unwrap_or_else(|_| "100000".into())
        .parse()?;
    let batch = 4000;
    let directory = tempfile::tempdir()?;
    println!(
        "{}",
        json!({"pid":std::process::id(),"nodes":nodes,"batch":batch,"directory":directory.path()})
    );
    let mut options = EmbeddedOptions::new(directory.path())
        .with_embedding_cache_dir(directory.path().join("embedding-cache"));
    let embedding = std::env::var_os("IG_LOAD_EMBEDDING").is_some();
    if !embedding {
        options = options.with_embedding_policy(EmbeddingPolicy::Disabled);
    }
    options.snapshot_interval = Duration::from_secs(30);
    let database = Arc::new(EmbeddedDatabase::open(options.clone())?);
    database.query(Query::new("CREATE PROJECT code_load"))?;
    let text = "code graph function documentation signature parameters return values ".repeat(10);
    let started = Instant::now();
    for start in (0..nodes).step_by(batch) {
        let rows = (start..(start+batch).min(nodes)).map(|id| json!({"sid":format!("sym:{id}"),"name":format!("function{id}"),"path":format!("src/{}.rs",id%1626),"summary":text,"doc":text,"signature":format!("fn function{id}(request: Request) -> Response"),"body_hash":"operational-hash","line":id,"kind":"function","optional":false})).collect::<Vec<_>>();
        let tick = Instant::now();
        database.query(
            Query::new("USE code_load UNWIND $rows AS r CREATE (n:Code:Sym) SET n += r")
                .with_parameter("rows", json!(rows)),
        )?;
        record("create_nodes", tick);
        if start == 0 {
            database.query(Query::new(
                "USE code_load CREATE INDEX code_sid IF NOT EXISTS FOR (n:Code) ON (n.sid)",
            ))?;
            database.query(Query::new(
                "USE code_load CREATE INDEX sym_sid IF NOT EXISTS FOR (n:Sym) ON (n.sid)",
            ))?;
        }
    }
    record("nodes_total", started);
    if std::env::var_os("IG_LOAD_EXTRA").is_some() {
        database.query(Query::new(
            "USE code_load MATCH (n:Code) WHERE n.line < 1626 SET n:File",
        ))?;
        let rows = (0..4000_usize)
            .map(
                |id| json!({"sid":format!("sym:{}",id%nodes.min(1626)),"hash":"changed-edge-hash"}),
            )
            .collect::<Vec<_>>();
        let tick = Instant::now();
        database.query(
            Query::new(
                "USE code_load UNWIND $rows AS r MATCH (f:File {sid:r.sid}) SET f.edge_hash=r.hash",
            )
            .with_parameter("rows", json!(rows)),
        )?;
        record("unindexed_file_update", tick);
        database.query(Query::new(
            "USE code_load USE LAYER WORKSPACE WRITE LAYER WORKSPACE CREATE (:Settings {scope:'code'})",
        ))?;
        database.query(Query::new(
            "USE code_load CREATE INDEX settings_scope IF NOT EXISTS FOR (n:Settings) ON (n.scope)",
        ))?;
        for value in 0..8 {
            let tick = Instant::now();
            database.query(Query::new("USE code_load USE LAYER OBSERVED, KNOWLEDGE, WORKSPACE WRITE LAYER WORKSPACE MERGE (s:Settings {scope:'code'}) SET s.values=$values, s.updated_ms=$now").with_parameter("values",json!({"theme":"plate","version":value})).with_parameter("now",json!(value)))?;
            record("settings_merge", tick);
        }
    }
    let stop = Arc::new(AtomicBool::new(false));
    let reader_database = Arc::clone(&database);
    let reader_stop = Arc::clone(&stop);
    let reader = std::thread::spawn(move || -> Result<Vec<f64>, String> {
        let mut times = Vec::new();
        while !reader_stop.load(Ordering::Acquire) {
            let tick = Instant::now();
            let result = reader_database
                .query(
                    Query::new("USE code_load MATCH (n:Sym {sid:$sid}) RETURN n.sid")
                        .with_parameter("sid", json!("sym:1")),
                )
                .map_err(|e| e.to_string())?;
            if result.rows.len() != 1 {
                return Err("point read lost its owner".into());
            }
            times.push(tick.elapsed().as_secs_f64() * 1000.0);
            std::thread::sleep(Duration::from_millis(10));
        }
        Ok(times)
    });
    let result = (|| -> Result<(), Box<dyn std::error::Error>> {
        for cycle in 0..3 {
            for start in (0..nodes * 3 / 2).step_by(batch) {
                let rows = (start..(start+batch).min(nodes*3/2)).map(|id| json!({"a":format!("sym:{}",id%nodes),"b":format!("sym:{}",(id+1)%nodes),"p":{"line":id,"confidence":"high","args":["request","parameters"]}})).collect::<Vec<_>>();
                let tick = Instant::now();
                database.query(Query::new("USE code_load UNWIND $rows AS r MATCH (a:Code {sid:r.a}), (b:Code {sid:r.b}) CREATE (a)-[e:CALLS]->(b) SET e += r.p").with_parameter("rows",json!(rows)))?;
                record("create_edges", tick);
            }
            let count = database.query(Query::new(
                "USE code_load MATCH ()-[r:CALLS]->() RETURN count(r)",
            ))?;
            assert_eq!(integer(&count.rows[0][0]), (nodes * 3 / 2) as i64);
            let tick = Instant::now();
            database.snapshot()?;
            record("checkpoint", tick);
            for statement in [
                "MATCH (n:Code) WHERE n.code IS NOT NULL REMOVE n.code",
                "MATCH (d:Description) DETACH DELETE d",
                "MATCH (n:Sym) RETURN n.sid LIMIT 1",
                "MATCH (n:Sym) WHERE n.description IS NOT NULL RETURN count(n)",
            ] {
                let tick = Instant::now();
                database.query(Query::new(format!("USE code_load {statement}")))?;
                record(statement, tick);
            }
            for start in (0..nodes).step_by(batch) {
                let rows=(start..(start+batch).min(nodes)).map(|id|json!({"sid":format!("sym:{id}"),"p":{"summary":text,"body_hash":format!("hash-{cycle}"),"sig_hash":"same","updated_ms":cycle}})).collect::<Vec<_>>();
                let tick = Instant::now();
                database.query(Query::new("USE code_load UNWIND $rows AS r MERGE (n:Code:Sym {sid:r.sid}) SET n += r.p").with_parameter("rows",json!(rows)))?;
                record("merge_nodes", tick);
                let sids = (start..(start + batch).min(nodes))
                    .map(|id| format!("sym:{id}"))
                    .collect::<Vec<_>>();
                let tick = Instant::now();
                database.query(
                    Query::new(
                        "USE code_load UNWIND $sids AS s MATCH (a:Code {sid:s})-[r]->() DELETE r",
                    )
                    .with_parameter("sids", json!(sids)),
                )?;
                record("delete_edges", tick);
            }
            let count = database.query(Query::new(
                "USE code_load MATCH ()-[r:CALLS]->() RETURN count(r)",
            ))?;
            assert_eq!(integer(&count.rows[0][0]), 0);
            println!("{}", json!({"cycle_done":cycle}));
        }
        Ok(())
    })();
    stop.store(true, Ordering::Release);
    let mut reads = reader.join().map_err(|_| "reader panicked")??;
    result?;
    reads.sort_by(f64::total_cmp);
    assert!(!reads.is_empty());
    println!(
        "{}",
        json!({"reads":reads.len(),"p50_ms":reads[reads.len()/2],"p99_ms":reads[reads.len()*99/100],"max_ms":reads.last()})
    );
    if std::env::var_os("IG_LOAD_ASSERT_LATENCY").is_some() {
        assert!(
            reads[reads.len() - 1] < 1000.0,
            "point read stalled for at least one second"
        );
    }
    if embedding {
        let deadline = Instant::now() + Duration::from_secs(120);
        loop {
            let found = database.query(Query::new("USE code_load SEARCH n IN (EMBEDDING INDEX graph_semantic FOR TEXT 'function documentation parameters' LIMIT 5) SCORE AS score RETURN n, score"))?;
            if !found.rows.is_empty() {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "encoder did not publish searchable owners under load"
            );
            std::thread::sleep(Duration::from_millis(100));
        }
    }
    Arc::try_unwrap(database)
        .map_err(|_| "database still borrowed")?
        .close()?;
    let database = EmbeddedDatabase::open(options)?;
    let count = database.query(Query::new("USE code_load MATCH (n:Code) RETURN count(n)"))?;
    assert_eq!(integer(&count.rows[0][0]), nodes as i64);
    database.close()?;
    println!("CODE_GRAPH_LOAD_OK");
    Ok(())
}
