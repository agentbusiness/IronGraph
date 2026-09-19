use super::*;
use crate::{
    engine::{
        BootstrappedNode, ExecutionClass, SingleNodeBootstrapConfig, WriteStorageLimits,
        open_standalone,
    },
    gpu::{
        BackendKind, DeviceMemoryGovernor, ResolvedComputeDevice,
        create_execution_backend_with_governor,
    },
    graph::{EmbeddingDType, EmbeddingProfile, Similarity},
    protocol::{QueryExecutor, QueryStreamEvent},
};

struct ContentEncoder {
    profile: EmbeddingProfile,
    inputs: parking_lot::Mutex<Vec<String>>,
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
}

async fn open(
    path: &Path,
    kind: BackendKind,
    encoder: Arc<ContentEncoder>,
) -> Result<BootstrappedNode<Database>> {
    let boot = open_standalone(
        path,
        SingleNodeBootstrapConfig {
            execution_class: if kind == BackendKind::Metal {
                ExecutionClass::Metal
            } else {
                ExecutionClass::Cpu
            },
            startup_timeout: Duration::from_secs(30),
            storage_limits: WriteStorageLimits {
                max_log_record_bytes: 16 * 1024 * 1024,
                max_log_entries_per_read: 4096,
                max_snapshot_bytes: 64 * 1024 * 1024,
            },
        },
        |path, identity| {
            Ok(Arc::new(Database::open_backend(
                path,
                8 * 1024 * 1024,
                Duration::from_secs(30),
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
    boot.backend()
        .bind_execution_backend(create_execution_backend_with_governor(
            ResolvedComputeDevice {
                backend: kind,
                ordinal: 0,
            },
            DeviceMemoryGovernor::new(512 * 1024 * 1024, 0),
        )?)?;
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

async fn verify(kind: BackendKind) -> Result<()> {
    let directory = tempfile::tempdir()?;
    let encoder = Arc::new(ContentEncoder::new()?);
    let boot = open(directory.path(), kind, encoder.clone()).await?;
    let database = boot.backend();
    query(database, "CREATE PROJECT automatic")?;
    query(
        database,
        "USE automatic CREATE (p:Person {name:'Ada'}), (d:Document {title:'Handbook', body:'complete document prose'}), (e:Email {subject:'Greetings', body:'meeting notes'}), (t:Table {title:'table inventory', rows:['chair','desk'], quantities:[12,7]}), (c:CalendarPlan {title:'Workshop', starts_at:date('2026-09-20')}), (task:Task {title:'Prepare workshop', metadata:'guitar secret noise'}), (n:Place {name:'Studio'}), (task)-[:ASSIGNED_TO {description:'contract negotiation responsibility'}]->(p), (e)-[:ABOUT]->(c), (c)-[:LOCATED_IN]->(n)",
    )?;
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
    let before = encoder.inputs.lock().len();
    query(
        database,
        "USE automatic MATCH (n:Task) SET n.updated_at = 'guitar secret metadata'",
    )?;
    assert_eq!(
        encoder.inputs.lock().len(),
        before,
        "metadata-only changes must not invoke encoder"
    );
    query(
        database,
        "USE automatic MATCH ()-[r:ASSIGNED_TO]->() SET r.description = 'guitar lesson responsibility'",
    )?;
    let top = query(
        database,
        "USE automatic SEARCH entity IN (EMBEDDING INDEX graph_semantic FOR TEXT 'guitar' LIMIT 1) SCORE AS score RETURN entity, score",
    )?;
    assert!(top[0][0].to_string().contains("guitar lesson"));
    let project = database.resolve_project_name("automatic")?;
    let mut transaction = database.begin(Some(project), None, CommitAcknowledgement::Published)?;
    tokio::time::sleep(TRANSACTION_MAINTENANCE_INTERVAL * 2).await;
    for (statement, expected_rows) in [
        (
            "MATCH (p:Person) CREATE (draft:Draft {title:'contract draft'})-[:FOR]->(p)",
            0,
        ),
        (
            "SEARCH entity IN (EMBEDDING INDEX graph_semantic FOR TEXT 'contract' LIMIT 100) SCORE AS score RETURN entity, score",
            12,
        ),
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
            "transaction reads its own automatic vectors"
        );
    }
    transaction.rollback()?;
    assert_eq!(query(database, "USE automatic SEARCH entity IN (EMBEDDING INDEX graph_semantic FOR TEXT 'contract' LIMIT 100) SCORE AS score RETURN entity")?.len(), 10, "rolled-back vectors never become visible");
    let (_, _, resident) = database.capture_semantic_execution(project)?;
    assert_eq!(resident.as_ref().map(|backend| backend.kind()), Some(kind));
    database
        .standalone_snapshot(&directory.path().join("standalone-snapshots"))
        .await?;
    query(
        database,
        "USE automatic MATCH (n:Document) SET n.body = 'contract WAL content'",
    )?;
    boot.runtime().shutdown().await?;
    drop(resident);
    drop(boot);
    let before_reopen = encoder.inputs.lock().len();
    let restored = open(directory.path(), kind, encoder.clone()).await?;
    let found = query(
        restored.backend(),
        "USE automatic SEARCH entity IN (EMBEDDING INDEX graph_semantic FOR TEXT 'contract' LIMIT 1) SCORE AS score RETURN entity, score",
    )?;
    assert!(found[0][0].to_string().contains("WAL content"));
    assert_eq!(
        encoder.inputs.lock().len(),
        before_reopen,
        "recovery reuses persisted vectors"
    );
    query(
        restored.backend(),
        "USE automatic MATCH (p:Person) DETACH DELETE p",
    )?;
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
        "automatic semantic integration verified on {kind:?}: all owner kinds, ranked limits, edits, metadata, WAL/snapshot and detach deletion"
    );
    Ok(())
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn automatic_semantic_cpu() -> Result<()> {
    verify(BackendKind::Cpu).await
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn automatic_semantic_backfills_existing_graph() -> Result<()> {
    let directory = tempfile::tempdir()?;
    let encoder = Arc::new(ContentEncoder::new()?);
    let boot = open(directory.path(), BackendKind::Cpu, encoder.clone()).await?;
    *boot.backend().0.text_embedding.write() = None;
    query(boot.backend(), "CREATE PROJECT existing")?;
    query(
        boot.backend(),
        "USE existing CREATE (a:Person {name:'Ada'}), (b:Task {title:'Plan'}), (b)-[:ASSIGNED_TO {description:'contract'}]->(a)",
    )?;
    assert!(encoder.inputs.lock().is_empty());
    boot.backend().bind_text_embedding(encoder.clone())?;
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

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn automatic_semantic_metal() -> Result<()> {
    let _guard = crate::gpu::metal_test_guard();
    verify(BackendKind::Metal).await
}
