use super::*;
use crate::{
    engine::{
        BootstrappedNode, ExecutionClass, SingleNodeBootstrapConfig, WriteStorageLimits,
        open_standalone,
    },
    graph::{EmbeddingDType, EmbeddingProfile, Similarity},
    protocol::{QueryExecutor, QueryStreamEvent},
};

/// Measures execution and the complete NDJSON pipeline on an imported, canonical flights graph.
/// Inference is paused identically across comparisons; no rows or properties are omitted.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[ignore = "release throughput measurement; imports the bundled flights dataset"]
async fn full_flights_result_throughput() -> Result<()> {
    use futures::StreamExt;
    let directory = tempfile::tempdir()?;
    let encoder = Arc::new(ContentEncoder::new()?);
    encoder.paused.store(true, Ordering::Release);
    let boot = open(directory.path(), encoder).await?;
    let database = boot.backend().as_ref().clone();
    let result = async {
        query_async(&database, "IMPORT DATASET flights").await?;
        for (name, statement, expected_rows) in [
            ("nodes", "USE flights MATCH (n) RETURN n", 13_859),
            ("route_ids", "USE flights MATCH (source)-[relationship]->(target) RETURN id(source), id(relationship), id(target)", 66_770),
            ("full_routes", "USE flights MATCH (source)-[relationship]->(target) RETURN source, relationship, target", 66_770),
        ] {
            let mut timings = Vec::new();
            let mut sizes = Vec::new();
            for repetition in 0..6 {
                let request: QueryRequest = serde_json::from_value(serde_json::json!({
                    "request_id": Uuid::nil(), "query": statement
                })).map_err(|error| Error::internal(error.to_string()))?;
                let (sender, mut body) = crate::protocol::ndjson_channel::<QueryStreamEvent>(8);
                let producer = database.clone();
                let started = Instant::now();
                let job = tokio::task::spawn_blocking(move || {
                    let mut rows = 0;
                    let mut values = 0;
                    producer.execute(request, &mut |event| {
                        if let QueryStreamEvent::Batch { row_count, columns, .. } = &event {
                            rows += row_count;
                            values += columns.iter().map(|column| column.values.len()).sum::<usize>();
                        }
                        sender.blocking_send(event)
                    })?;
                    Ok::<_, Error>((rows, values))
                });
                let mut bytes = 0;
                while let Some(chunk) = body.next().await {
                    bytes += chunk.expect("NDJSON body is infallible").len();
                }
                let elapsed = started.elapsed().as_micros();
                let (rows, values) = job.await.map_err(|error| Error::internal(error.to_string()))??;
                assert_eq!(rows, expected_rows);
                assert_eq!(values, expected_rows as usize * if name == "nodes" { 1 } else { 3 });
                if repetition > 0 {
                    timings.push(elapsed);
                    sizes.push(bytes);
                }
            }
            timings.sort_unstable();
            println!("FLIGHTS_THROUGHPUT name={name} rows={expected_rows} median_us={} min_us={} max_us={} bytes={}", timings[2], timings[0], timings[4], sizes[2]);
        }
        Ok::<_, Error>(())
    }.await;
    database.shutdown_embedding_jobs().await?;
    boot.runtime().shutdown().await?;
    result?;
    println!("FULL FLIGHTS RESULTS VERIFIED");
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn concurrent_create_then_set_keeps_canonical_apply_healthy() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let encoder = Arc::new(ContentEncoder::new()?);
    encoder.paused.store(true, Ordering::Release);
    let boot = open(directory.path(), encoder).await?;
    let database = boot.backend().as_ref().clone();
    query_async(&database, "CREATE PROJECT concurrent_set").await?;
    let writes = (0..16).map(|value| {
        let database = database.clone();
        tokio::spawn(async move {
            query_async(
                &database,
                &format!("USE concurrent_set UNWIND range(1,32) AS step CREATE (n:Parallel) SET n.value={value}, n.step=step RETURN count(*)"),
            ).await
        })
    }).collect::<Vec<_>>();
    for write in writes {
        write
            .await
            .map_err(|error| Error::internal(error.to_string()))??;
    }
    let rows = query_async(
        &database,
        "USE concurrent_set MATCH (n:Parallel) RETURN count(n)",
    )
    .await?;
    assert_eq!(rows[0][0]["value"], "512");
    database.shutdown_embedding_jobs().await?;
    boot.runtime().shutdown().await?;
    Ok(())
}

struct ContentEncoder {
    profile: EmbeddingProfile,
    inputs: parking_lot::Mutex<Vec<String>>,
    paused: AtomicBool,
    entered: AtomicBool,
}

impl ContentEncoder {
    fn new() -> Result<Self> {
        Ok(Self {
            profile: EmbeddingProfile::new(
                [21; 32],
                [22; 32],
                4,
                EmbeddingDType::F16,
                true,
                Similarity::Cosine,
            )?,
            inputs: parking_lot::Mutex::new(Vec::new()),
            paused: AtomicBool::new(false),
            entered: AtomicBool::new(false),
        })
    }
}

impl TextEmbedding for ContentEncoder {
    fn profile(&self) -> &EmbeddingProfile {
        &self.profile
    }
    fn embed(&self, text: &str) -> Result<Vec<f32>> {
        let lower = text.to_lowercase();
        let mut vector = vec![
            0.1,
            f32::from(lower.contains("contract")),
            f32::from(lower.contains("guitar")),
            f32::from(lower.contains("table")),
        ];
        let norm = vector.iter().map(|v| v * v).sum::<f32>().sqrt();
        for value in &mut vector {
            *value /= norm;
        }
        Ok(vector)
    }
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        self.inputs.lock().extend_from_slice(texts);
        texts.iter().map(|text| self.embed(text)).collect()
    }
    fn embed_batch_cancellable(
        &self,
        texts: &[String],
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Vec<Vec<f32>>> {
        self.entered.store(true, Ordering::Release);
        while self.paused.load(Ordering::Acquire) {
            if cancelled() {
                return Err(Error::new(ErrorCode::Cancelled, "test inference cancelled"));
            }
            std::thread::sleep(Duration::from_millis(1));
        }
        if cancelled() {
            return Err(Error::new(ErrorCode::Cancelled, "test inference cancelled"));
        }
        self.embed_batch(texts)
    }
}

async fn open(path: &Path, encoder: Arc<ContentEncoder>) -> Result<BootstrappedNode<Database>> {
    let boot = open_standalone(
        path,
        SingleNodeBootstrapConfig {
            execution_class: ExecutionClass::Cpu,
            startup_timeout: Duration::from_secs(30),
            storage_limits: WriteStorageLimits {
                max_log_record_bytes: usize::MAX,
                max_log_entries_per_read: 4096,
                max_snapshot_bytes: usize::MAX,
            },
        },
        |path, identity| {
            Ok(Arc::new(Database::open_backend(
                path,
                usize::MAX,
                Duration::ZERO,
                identity,
            )?))
        },
    )
    .await?;
    boot.backend()
        .bind_runtime(Arc::downgrade(boot.runtime()))?;
    boot.backend()
        .standalone_recover(&path.join("standalone-snapshots"))
        .await?;
    boot.runtime().replay_standalone_wal().await?;
    boot.backend().bind_text_embedding(encoder)?;
    Ok(boot)
}

fn query(database: &Database, statement: &str) -> Result<Vec<Vec<serde_json::Value>>> {
    let request: QueryRequest = serde_json::from_value(
        serde_json::json!({ "request_id": Uuid::new_v4(), "query": statement }),
    )
    .map_err(|error| Error::internal(error.to_string()))?;
    let mut rows = Vec::new();
    database
        .execute(request, &mut |event| {
            match event {
                QueryStreamEvent::Batch {
                    row_count, columns, ..
                } => {
                    for row in 0..row_count as usize {
                        rows.push(
                            columns
                                .iter()
                                .map(|column| {
                                    serde_json::to_value(&column.values[row])
                                        .map_err(|error| Error::internal(error.to_string()))
                                })
                                .collect::<Result<Vec<_>>>()?,
                        );
                    }
                }
                QueryStreamEvent::Error { code, message, .. } => {
                    return Err(Error::new(code, message));
                }
                _ => {}
            }
            Ok(())
        })
        .map_err(|error| Error::new(error.code, format!("{statement}: {}", error.message)))?;
    Ok(rows)
}

async fn wait_until(mut ready: impl FnMut() -> Result<bool>) -> Result<()> {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if ready()? {
                return Ok(());
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .map_err(|_| Error::internal("asynchronous semantic completion timed out"))?
}

async fn query_async(database: &Database, statement: &str) -> Result<Vec<Vec<serde_json::Value>>> {
    let database = database.clone();
    let statement = statement.to_owned();
    tokio::task::spawn_blocking(move || {
        std::thread::spawn(move || query(&database, &statement))
            .join()
            .map_err(|_| Error::internal("external synchronous query caller panicked"))?
    })
    .await
    .map_err(|e| Error::internal(format!("query worker failed: {e}")))?
}

async fn wait_embeddings(database: &Database, name: &str, encoder: &ContentEncoder) -> Result<()> {
    let project = database.resolve_project_name(name)?;
    database.initialize_automatic_semantic(project)?;
    wait_until(|| {
        let (live, _) = database.capture_canonical_project(project)?;
        for (kind, index, count) in [
            (
                crate::types::EntityKind::Node,
                crate::graph::SEMANTIC_NODE_INDEX,
                live.graph.node_count(),
            ),
            (
                crate::types::EntityKind::Relationship,
                crate::graph::SEMANTIC_RELATIONSHIP_INDEX,
                live.graph.edge_count(),
            ),
        ] {
            let Some((vectors, _)) = live.indexes.vector_search_source(index) else {
                return Ok(false);
            };
            if vectors.len() != count {
                return Ok(false);
            }
            let ids = match kind {
                crate::types::EntityKind::Node => {
                    live.graph.nodes().map(|row| row.id().0).collect::<Vec<_>>()
                }
                crate::types::EntityKind::Relationship => {
                    live.graph.edges().map(|row| row.id().0).collect::<Vec<_>>()
                }
            };
            for id in ids {
                let Some(text) = crate::graph::semantic_owner_text(&live.graph, kind, id)? else {
                    continue;
                };
                let Some(vector) = vectors.vector_for(id) else {
                    return Ok(false);
                };
                let expected = encoder.embed(&text)?;
                if vector.len() != expected.len()
                    || vector
                        .iter()
                        .zip(expected)
                        .any(|(actual, expected)| (actual - expected).abs() > 0.002)
                {
                    return Ok(false);
                }
            }
        }
        Ok(true)
    })
    .await
    .map_err(|error| {
        let details = database
            .capture_canonical_project(project)
            .map(|(live, _)| {
                let nodes = live
                    .indexes
                    .vector_search_source(crate::graph::SEMANTIC_NODE_INDEX)
                    .map_or(0, |(column, _)| column.len());
                let edges = live
                    .indexes
                    .vector_search_source(crate::graph::SEMANTIC_RELATIONSHIP_INDEX)
                    .map_or(0, |(column, _)| column.len());
                format!(
                    "{} nodes / {} vectors, {} relationships / {} vectors",
                    live.graph.node_count(),
                    nodes,
                    live.graph.edge_count(),
                    edges
                )
            })
            .unwrap_or_else(|error| error.to_string());
        Error::internal(format!(
            "{name}: {error}; {details}; {} encoder inputs",
            encoder.inputs.lock().len()
        ))
    })
}

async fn wait_embedding_idle(database: &Database) -> Result<()> {
    wait_until(|| {
        Ok(database
            .0
            .embedding_jobs
            .get()
            .is_some_and(|queue| queue.is_idle()))
    })
    .await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn every_bundled_example_imports_with_paused_inference_and_parallel_reads() -> Result<()> {
    for dataset in super::super::dataset_loader::DATASETS {
        let directory = tempfile::tempdir()?;
        let encoder = Arc::new(ContentEncoder::new()?);
        encoder.paused.store(true, Ordering::Release);
        let boot = open(directory.path(), encoder.clone()).await?;
        let database = boot.backend().as_ref().clone();
        let importer = database.clone();
        let statement = format!("IMPORT DATASET {}", dataset.name);
        let started = Instant::now();
        let import = tokio::spawn(async move { query_async(&importer, &statement).await });
        let mut reads = 0;
        while !import.is_finished() {
            tokio::time::timeout(
                Duration::from_secs(2),
                query_async(&database, "SHOW PROJECTS"),
            )
            .await
            .map_err(|_| {
                Error::internal("parallel catalog read stalled during dataset ingestion")
            })??;
            reads += 1;
            tokio::task::yield_now().await;
        }
        let outcome = import
            .await
            .map_err(|error| Error::internal(error.to_string()))?;
        let verification = outcome.and_then(|_| {
            let project = QueryExecutor::resolve_project(&database, dataset.name)?;
            let (live, _) = database.capture_canonical_project(project)?;
            let nodes = live.graph.node_count() as u64;
            let relationships = live.graph.edge_count() as u64;
            if nodes != dataset.nodes || relationships != dataset.relationships || reads == 0 {
                return Err(Error::internal(format!(
                    "{} import: {nodes} nodes / {relationships} relationships / {reads} concurrent reads; expected {} / {}",
                    dataset.name, dataset.nodes, dataset.relationships
                )));
            }
            println!(
                "EXAMPLE IMPORT PASS {}: {} nodes, {} relationships, {} concurrent reads, {:.3}s",
                dataset.name,
                dataset.nodes,
                dataset.relationships,
                reads,
                started.elapsed().as_secs_f64()
            );
            Ok(())
        });
        database.shutdown_embedding_jobs().await?;
        boot.runtime().shutdown().await?;
        verification?;
    }
    println!("ALL BUNDLED EXAMPLES IMPORTED WITH PAUSED INFERENCE AND PARALLEL READS");
    Ok(())
}

async fn verify() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let encoder = Arc::new(ContentEncoder::new()?);
    let boot = open(directory.path(), encoder.clone()).await?;
    let database = boot.backend();
    query(database, "CREATE PROJECT automatic")?;
    query(
        database,
        "USE automatic CREATE (p:Person {name:'Ada'}), (d:Document {title:'Handbook', body:'complete document prose'}), (e:Email {subject:'Greetings', body:'meeting notes'}), (t:Table {title:'table inventory', rows:['chair','desk'], quantities:[12,7]}), (c:CalendarPlan {title:'Workshop', starts_at:date('2026-09-20')}), (task:Task {title:'Prepare workshop', metadata:'guitar secret noise'}), (n:Place {name:'Studio'}), (task)-[:ASSIGNED_TO {description:'contract negotiation responsibility'}]->(p), (e)-[:ABOUT]->(c), (c)-[:LOCATED_IN]->(n)",
    )?;
    wait_embeddings(database, "automatic", &encoder).await?;
    let all = query(
        database,
        "USE automatic SEARCH entity IN (EMBEDDING INDEX graph_semantic FOR TEXT 'contract' LIMIT 100) SCORE AS score RETURN entity, score",
    )?;
    assert_eq!(
        all.len(),
        10,
        "all seven nodes and three relationships appear exactly once: {all:?}"
    );
    assert_eq!(all[0][0]["type"], "relationship");
    assert!(all[0][0].to_string().contains("contract negotiation"));
    let scores = all
        .iter()
        .map(|row| row[1]["value"].as_f64().unwrap())
        .collect::<Vec<_>>();
    assert!(scores.windows(2).all(|pair| pair[0] >= pair[1]));
    let reranked = query(
        database,
        "USE automatic SEARCH entity IN (EMBEDDING INDEX graph_semantic FOR TEXT 'table' LIMIT 100) SCORE AS initial SEARCH entity IN (EMBEDDING INDEX graph_semantic FOR TEXT 'contract' LIMIT 100) SCORE AS score RETURN entity, score",
    )?;
    assert_eq!(
        reranked.len(),
        10,
        "repeated mixed search preserves both identity namespaces"
    );
    assert_eq!(reranked[0][0]["type"], "relationship");
    let written = query(
        database,
        "USE automatic SEARCH entity IN (EMBEDDING INDEX graph_semantic FOR TEXT 'contract' LIMIT 1) SCORE AS score SET entity.reviewed = true RETURN entity, score",
    )?;
    assert_eq!(written.len(), 1);
    assert_eq!(written[0][0]["type"], "relationship");
    assert!(
        encoder
            .inputs
            .lock()
            .iter()
            .all(|text| !text.contains("secret noise") && !text.contains("created_at"))
    );
    let matched = query(
        database,
        "USE automatic MATCH (n) SEARCH n IN (EMBEDDING INDEX semantic_nodes FOR TEXT 'table' LIMIT 3) SCORE AS score RETURN n, score",
    )?;
    assert_eq!(
        matched.len(),
        3,
        "MATCH must not duplicate the top-k for every input node"
    );
    assert!(matched[0][0].to_string().contains("inventory"));
    let edges = query(
        database,
        "USE automatic MATCH ()-[r]->() SEARCH r IN (EMBEDDING INDEX semantic_relationships FOR TEXT 'contract' LIMIT 2) SCORE AS score RETURN r, score",
    )?;
    assert_eq!(edges.len(), 2);
    wait_embeddings(database, "automatic", &encoder).await?;
    wait_embedding_idle(database).await?;
    let before = encoder.inputs.lock().len();
    query(
        database,
        "USE automatic MATCH (n:Task) SET n.updated_at = 'guitar secret metadata'",
    )?;
    wait_embedding_idle(database).await?;
    assert_eq!(
        encoder.inputs.lock().len(),
        before,
        "metadata-only changes must not invoke encoder"
    );
    query(
        database,
        "USE automatic MATCH ()-[r:ASSIGNED_TO]->() SET r.description = 'guitar lesson responsibility'",
    )?;
    wait_embeddings(database, "automatic", &encoder).await?;
    let top = query(
        database,
        "USE automatic SEARCH entity IN (EMBEDDING INDEX graph_semantic FOR TEXT 'guitar' LIMIT 1) SCORE AS score RETURN entity, score",
    )?;
    assert!(top[0][0].to_string().contains("guitar lesson"));
    let project = database.resolve_project_name("automatic")?;
    let mut transaction = database.begin(Some(project), None, CommitAcknowledgement::Published)?;
    for (statement, expected_rows) in [
        (
            "MATCH (p:Person) CREATE (draft:Draft {title:'contract draft'})-[:FOR]->(p)",
            0,
        ),
        (
            "SEARCH entity IN (EMBEDDING INDEX graph_semantic FOR TEXT 'contract' LIMIT 100) SCORE AS score RETURN entity, score",
            10,
        ),
        ("MATCH (n:Draft) RETURN n.title", 1),
    ] {
        let request = serde_json::from_value(serde_json::json!({"request_id":Uuid::new_v4(), "project_id":project, "query":statement}))
            .map_err(|error| Error::internal(error.to_string()))?;
        let mut rows = 0;
        transaction.run(request, &mut |event| {
            match event {
                QueryStreamEvent::Batch { row_count, .. } => rows += row_count,
                QueryStreamEvent::Error { code, message, .. } => {
                    return Err(Error::new(code, message));
                }
                _ => {}
            }
            Ok(())
        })?;
        assert_eq!(
            rows, expected_rows,
            "transaction reads its graph journal while vectors update only after commit"
        );
    }
    transaction.rollback()?;
    assert_eq!(query(database, "USE automatic SEARCH entity IN (EMBEDDING INDEX graph_semantic FOR TEXT 'contract' LIMIT 100) SCORE AS score RETURN entity")?.len(), 10, "rolled-back vectors never become visible");
    let (canonical, _) = database.capture_canonical_project(project)?;
    assert_eq!(
        canonical.graph.node_count() + canonical.graph.edge_count(),
        10
    );
    database
        .standalone_snapshot(&directory.path().join("standalone-snapshots"))
        .await?;
    query(
        database,
        "USE automatic MATCH (n:Document) SET n.body = 'contract WAL content'",
    )?;
    wait_embeddings(database, "automatic", &encoder).await?;
    boot.runtime().shutdown().await?;
    drop(boot);
    let before_reopen = encoder.inputs.lock().len();
    encoder.paused.store(true, Ordering::Release);
    let restored = open(directory.path(), encoder.clone()).await?;
    let found = query(
        restored.backend(),
        "USE automatic SEARCH entity IN (EMBEDDING INDEX graph_semantic FOR TEXT 'contract' LIMIT 1) SCORE AS score RETURN entity, score",
    )?;
    assert!(found[0][0].to_string().contains("WAL content"));
    assert_eq!(
        encoder.inputs.lock().len(),
        before_reopen,
        "persisted vectors are readable before asynchronous recovery inference completes"
    );
    encoder.paused.store(false, Ordering::Release);
    wait_embeddings(restored.backend(), "automatic", &encoder).await?;
    wait_embedding_idle(restored.backend()).await?;
    assert_eq!(
        encoder.inputs.lock().len(),
        before_reopen,
        "completed recovery cold pass reuses all current persisted vectors"
    );
    query(
        restored.backend(),
        "USE automatic MATCH (p:Person) DETACH DELETE p",
    )?;
    wait_embeddings(restored.backend(), "automatic", &encoder).await?;
    let remaining = query(
        restored.backend(),
        "USE automatic SEARCH entity IN (EMBEDDING INDEX graph_semantic FOR TEXT 'guitar' LIMIT 100) SCORE AS score RETURN entity, score",
    )?;
    assert_eq!(remaining.len(), 8);
    assert!(
        remaining
            .iter()
            .all(|row| !row[0].to_string().contains("guitar lesson"))
    );
    restored.runtime().shutdown().await?;
    println!(
        "automatic semantic integration verified on canonical CPU: all owner kinds, ranked limits, edits, metadata, WAL/snapshot and detach deletion"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn automatic_semantic_cpu() -> Result<()> {
    verify().await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn automatic_semantic_backfills_existing_graph() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let encoder = Arc::new(ContentEncoder::new()?);
    let boot = open(directory.path(), encoder.clone()).await?;
    *boot.backend().0.text_embedding.write() = None;
    query(boot.backend(), "CREATE PROJECT existing")?;
    query(
        boot.backend(),
        "USE existing CREATE (a:Person {name:'Ada'}), (b:Task {title:'Plan'}), (b)-[:ASSIGNED_TO {description:'contract'}]->(a)",
    )?;
    assert!(encoder.inputs.lock().is_empty());
    boot.backend().bind_text_embedding(encoder.clone())?;
    wait_embeddings(boot.backend(), "existing", &encoder).await?;
    let hits = query(
        boot.backend(),
        "USE existing SEARCH entity IN (EMBEDDING INDEX graph_semantic FOR TEXT 'contract' LIMIT 10) SCORE AS score RETURN entity, score",
    )?;
    assert_eq!(hits.len(), 3);
    assert_eq!(hits[0][0]["type"], "relationship");
    assert_eq!(encoder.inputs.lock().len(), 3);
    boot.runtime().shutdown().await?;
    Ok(())
}

#[tokio::test(flavor = "current_thread")]
async fn paused_embedding_keeps_graph_queries_responsive_and_fences_endpoint_changes() -> Result<()>
{
    let directory = tempfile::tempdir()?;
    let encoder = Arc::new(ContentEncoder::new()?);
    let boot = open(directory.path(), encoder.clone()).await?;
    let database = boot.backend();
    query_async(database, "CREATE PROJECT paused").await?;
    query_async(
        database,
        "USE paused CREATE (a:Document {body:'contract original'}), (b:Person {name:'Ada'}), (a)-[:ABOUT {description:'contract original relationship'}]->(b)",
    ).await?;
    wait_embeddings(database, "paused", &encoder)
        .await
        .map_err(|e| Error::internal(format!("initial paused fixture: {e}")))?;
    encoder.entered.store(false, Ordering::Release);
    encoder.paused.store(true, Ordering::Release);
    query_async(
        database,
        "USE paused MATCH ()-[r:ABOUT]->() SET r.description = 'guitar obsolete relationship'",
    )
    .await?;
    wait_until(|| Ok(encoder.entered.load(Ordering::Acquire)))
        .await
        .map_err(|e| Error::internal(format!("paused inference admission: {e}")))?;
    let complete_body = format!(
        "{}table latest endpoint",
        "complete source paragraph ".repeat(12_000)
    );
    // There is only one async runtime thread. This timer and canonical read must make progress
    // while the encoder's blocking worker is paused, and another graph write must also complete.
    tokio::time::timeout(Duration::from_millis(250), async {
        tokio::time::sleep(Duration::from_millis(2)).await;
        let rows = query_async(database, "USE paused MATCH (n:Document) RETURN n.body").await?;
        assert_eq!(rows.len(), 1);
        query_async(
            database,
            &format!("USE paused MATCH (n:Document) SET n.body = '{complete_body}'"),
        )
        .await?;
        query_async(
            database,
            "USE paused MATCH ()-[r:ABOUT]->() SET r.description = 'table latest relationship'",
        )
        .await?;
        Result::<()>::Ok(())
    })
    .await
    .map_err(|_| Error::internal("paused embedding blocked graph/runtime progress"))??;
    let project = database.resolve_project_name("paused")?;
    let (live, _) = database.capture_canonical_project(project)?;
    let complete = query_async(database, "USE paused MATCH (n:Document) RETURN n.body").await?;
    assert_eq!(
        complete[0][0]["value"], complete_body,
        "the canonical owner retains complete large source content"
    );
    let edge = live
        .graph
        .edges()
        .next()
        .ok_or_else(|| Error::internal("test relationship absent"))?;
    let (vectors, _) = live
        .indexes
        .vector_search_source(crate::graph::SEMANTIC_RELATIONSHIP_INDEX)
        .ok_or_else(|| Error::internal("test semantic matrix absent"))?;
    let previous = vectors
        .vector_for(edge.id().0)
        .ok_or_else(|| Error::internal("test original vector absent"))?;
    assert!(
        previous[1] > previous[2],
        "paused inference leaves the previously published vector readable"
    );
    encoder.paused.store(false, Ordering::Release);
    wait_embeddings(database, "paused", &encoder)
        .await
        .map_err(|e| {
            Error::internal(format!(
                "latest endpoint completion: {e}; encoder batches: {}",
                encoder.inputs.lock().len()
            ))
        })?;
    let latest = vectors
        .vector_for(edge.id().0)
        .ok_or_else(|| Error::internal("latest relationship vector absent"))?;
    assert!(
        latest[3] > latest[1] && latest[3] > latest[2],
        "only the latest endpoint/relationship content is published"
    );
    assert!(
        encoder
            .inputs
            .lock()
            .iter()
            .all(|text| !text.contains("guitar obsolete")),
        "superseded work is cancelled before completing inference"
    );
    assert!(
        encoder
            .inputs
            .lock()
            .iter()
            .any(|text| text.contains(&complete_body)),
        "encoder receives complete owner text including its trailing content"
    );
    boot.runtime().shutdown().await?;
    Ok(())
}
