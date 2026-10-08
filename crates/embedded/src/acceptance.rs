use super::*;
pub(super) static TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct PausedEncoder {
    profile: irongraph_server::graph::EmbeddingProfile,
    entered: std::sync::mpsc::Sender<()>,
    release: Mutex<Option<std::sync::mpsc::Receiver<()>>>,
}

impl irongraph_server::execution::TextEmbedding for PausedEncoder {
    fn profile(&self) -> &irongraph_server::graph::EmbeddingProfile {
        &self.profile
    }
    fn embed(&self, _: &str) -> irongraph_types::Result<Vec<f32>> {
        Ok(vec![1.0, 0.0])
    }
    fn embed_batch(&self, texts: &[String]) -> irongraph_types::Result<Vec<Vec<f32>>> {
        let release = self.release.lock().unwrap().take();
        if let Some(release) = release {
            self.entered.send(()).unwrap();
            let _ = release.recv();
        }
        Ok(texts.iter().map(|_| vec![1.0, 0.0]).collect())
    }
}

#[test]
fn automatic_semantic_definitions_are_ready_while_encoder_is_paused()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let _guard = TESTS.lock().unwrap();
    let directory = tempfile::tempdir()?;
    let database = EmbeddedDatabase::open(
        EmbeddedOptions::new(directory.path()).with_embedding_policy(EmbeddingPolicy::Disabled),
    )?;
    let (entered, observed) = std::sync::mpsc::channel();
    let (release, blocked) = std::sync::mpsc::channel();
    let encoder = std::sync::Arc::new(PausedEncoder {
        profile: irongraph_server::graph::EmbeddingProfile::new(
            [1; 32],
            [2; 32],
            2,
            irongraph_server::graph::EmbeddingDType::F16,
            true,
            irongraph_server::graph::Similarity::Cosine,
        )?,
        entered,
        release: Mutex::new(Some(blocked)),
    });
    {
        let _runtime = database.runtime.enter();
        database
            .core
            .as_ref()
            .unwrap()
            .database
            .bind_text_embedding(encoder)?;
    }
    database.query(Query::new("CREATE PROJECT semantic_ready"))?;
    database.query(Query::new(
        "USE semantic_ready CREATE (:Document {body:'current meaningful text'})",
    ))?;
    observed.recv_timeout(Duration::from_secs(5))?;
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    let result = runtime.block_on(async {
        let result = database.query_async(Query::new("USE semantic_ready SEARCH entity IN (EMBEDDING INDEX graph_semantic FOR TEXT 'current meaningful text' LIMIT 10) SCORE AS score RETURN entity, score")).await?;
        assert!(result.rows.is_empty());
        let count = database.query_async(Query::new("USE semantic_ready MATCH (n:Document) RETURN count(n)")).await?;
        assert!(matches!(&count.rows[0][0], irongraph_server::protocol::TypedValue::Integer(value) if value == "1"));
        Ok::<_, EmbeddedError>(())
    });
    release.send(())?;
    result?;
    database.close()?;
    Ok(())
}

#[test]
fn async_large_parameters_and_results_keep_current_thread_responsive()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let _guard = TESTS.lock().unwrap();
    let directory = tempfile::tempdir()?;
    let database = EmbeddedDatabase::open(
        EmbeddedOptions::new(directory.path()).with_embedding_policy(EmbeddingPolicy::Disabled),
    )?;
    database.query(Query::new("CREATE PROJECT async_worker"))?;
    let query = Query::new("USE async_worker UNWIND range(1, 32) AS row RETURN $body AS body")
        .with_parameter("body", "x".repeat(1024 * 1024).into());
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async {
        let ticks = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let observed = ticks.clone();
        let heartbeat = tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_millis(1)).await;
                observed.fetch_add(1, Ordering::Relaxed);
            }
        });
        let result = database.query_async(query).await?;
        heartbeat.abort();
        assert_eq!(result.rows.len(), 32);
        assert!(
            ticks.load(Ordering::Relaxed) > 0,
            "current-thread timer did not run during large query"
        );
        Ok::<_, EmbeddedError>(())
    })?;
    database.close()?;
    Ok(())
}

#[test]
fn async_worker_capacity_survives_cancelled_waiter()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let _guard = TESTS.lock().unwrap();
    let directory = tempfile::tempdir()?;
    let mut options =
        EmbeddedOptions::new(directory.path()).with_embedding_policy(EmbeddingPolicy::Disabled);
    options.worker_threads = 1;
    let database = EmbeddedDatabase::open(options)?;
    let (entered, waiting) = std::sync::mpsc::channel();
    let mut blockers = Vec::new();
    let mut releases = Vec::new();
    for _ in 0..4 + BLOCKING_CALLBACK_HEADROOM {
        let entered = entered.clone();
        let (release, blocked) = std::sync::mpsc::channel();
        releases.push(release);
        blockers.push(database.runtime.handle().spawn_blocking(move || {
            entered.send(()).unwrap();
            blocked.recv().unwrap();
        }));
    }
    for _ in 0..blockers.len() {
        waiting.recv()?;
    }
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()?;
    runtime.block_on(async {
        let mut pending = Box::pin(database.query_async(Query::new("SHOW PROJECTS")));
        std::future::poll_fn(|context| {
            assert!(pending.as_mut().poll(context).is_pending());
            std::task::Poll::Ready(())
        })
        .await;
        drop(pending);
        assert_eq!(database.status()?.active_operations, 0);
        let mut queued = Box::pin(database.query_async(Query::new("SHOW PROJECTS")));
        std::future::poll_fn(|context| {
            assert!(queued.as_mut().poll(context).is_pending());
            std::task::Poll::Ready(())
        })
        .await;
        for release in releases {
            release.send(()).unwrap();
        }
        for blocker in blockers {
            blocker.await.unwrap();
        }
        queued.await?;
        drop(database.async_workers.acquire().await.unwrap());
        database.query_async(Query::new("SHOW PROJECTS")).await?;
        Ok::<_, EmbeddedError>(())
    })?;
    database.close()?;
    Ok(())
}

#[test]
fn saturated_async_writes_leave_runtime_callback_headroom()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let _guard = TESTS.lock().unwrap();
    for budget in [1, 32] {
        let directory = tempfile::tempdir()?;
        let mut options =
            EmbeddedOptions::new(directory.path()).with_embedding_policy(EmbeddingPolicy::Disabled);
        options.worker_threads = 1;
        let database = std::sync::Arc::new(EmbeddedDatabase::open(options)?);
        database.query(Query::new("CREATE PROJECT pool_progress"))?;
        database.query(Query::new("USE pool_progress CREATE (:Work {value: -1})"))?;
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?;
        runtime.block_on(async {
            let ticks = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
            let observed = ticks.clone();
            let heartbeat = tokio::spawn(async move {
                loop {
                    observed.fetch_add(1, Ordering::Relaxed);
                    tokio::time::sleep(Duration::from_millis(1)).await;
                }
            });
            let mut writers = Vec::new();
            for value in 0..budget {
                let database = database.clone();
                writers.push(tokio::spawn(async move {
                    database
                        .query_with_options_async(
                            Query::new("USE pool_progress CREATE (:Work {value: $value})")
                                .with_parameter("value", value.into()),
                            OperationOptions {
                                timeout_ms: Some(5_000),
                                ..Default::default()
                            },
                        )
                        .await
                }));
            }
            tokio::time::timeout(Duration::from_secs(10), async {
                for writer in writers {
                    writer.await.unwrap()?;
                }
                Ok::<_, EmbeddedError>(())
            })
            .await
            .unwrap()?;
            heartbeat.abort();
            assert!(ticks.load(Ordering::Relaxed) > 0);
            let count = database
                .query_async(Query::new(
                    "USE pool_progress MATCH (n:Work) RETURN count(n)",
                ))
                .await?;
            assert!(matches!(
                &count.rows[0][0],
                irongraph_server::protocol::TypedValue::Integer(value) if value == &(budget + 1).to_string()
            ));
            Ok::<_, EmbeddedError>(())
        })?;
        match std::sync::Arc::try_unwrap(database) {
            Ok(database) => database.close()?,
            Err(_) => panic!("async writer retained canonical database after completion"),
        }
    }
    Ok(())
}

#[test]
fn operation_admission_cancellation_and_release()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let _guard = TESTS.lock().unwrap();
    let directory = tempfile::tempdir()?;
    let options = EmbeddedOptions::new(directory.path())
        .with_execution_device(ExecutionDevice::Cpu)
        .with_embedding_policy(EmbeddingPolicy::Disabled);
    let database = EmbeddedDatabase::open(options)?;
    let active = database.begin_operation(OperationOptions {
        operation_id: Some("host-operation".into()),
        timeout_ms: Some(60_000),
    })?;
    assert_eq!(database.status()?.active_operations, 1);
    database.query(Query::new("SHOW PROJECTS"))?;
    assert!(database.cancel("host-operation"));
    assert!(
        matches!(active.check(), Err(EmbeddedError::Engine(error)) if error.code == irongraph_types::ErrorCode::Cancelled)
    );
    drop(active);
    assert_eq!(database.status()?.active_operations, 0);
    assert!(!database.cancel("host-operation"));
    database.query(Query::new("SHOW PROJECTS"))?;
    assert!(
        database
            .query_with_options(
                Query::new("CREATE PROJECT rejected"),
                OperationOptions {
                    timeout_ms: Some(0),
                    ..Default::default()
                }
            )
            .is_err()
    );
    assert!(database.query(Query::new("USE rejected RETURN 1")).is_err());
    database.close()?;
    Ok(())
}
