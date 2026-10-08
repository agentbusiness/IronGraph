use std::{sync::Arc, time::Duration};

use irongraph_client::{ClientError, Query};
use irongraph_embedded::{
    EmbeddedDatabase, EmbeddedError, EmbeddedOptions, EmbeddingPolicy, ExecutionDevice,
    OperationOptions,
};
use irongraph_server::protocol::TypedValue;

type TestResult = Result<(), Box<dyn std::error::Error>>;
static TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn open(directory: &std::path::Path) -> irongraph_embedded::Result<EmbeddedDatabase> {
    let mut options = EmbeddedOptions::new(directory)
        .with_execution_device(ExecutionDevice::Cpu)
        .with_embedding_policy(EmbeddingPolicy::Disabled);
    options.worker_threads = 1;
    let database = EmbeddedDatabase::open(options)?;
    database.query(Query::new("CREATE PROJECT async_cpu"))?;
    Ok(database)
}

fn current_thread() -> std::io::Result<tokio::runtime::Runtime> {
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
}

#[test]
fn current_thread_timer_progresses_during_cpu_query() -> TestResult {
    let _guard = TESTS.lock().map_err(|_| "test lock poisoned")?;
    let directory = tempfile::tempdir()?;
    let database = open(directory.path())?;
    let caller = current_thread()?;
    caller.block_on(async {
        let query = database.query_async(Query::new(
            "USE async_cpu UNWIND range(1, 1000000) AS value RETURN sum(value)",
        ));
        tokio::pin!(query);
        tokio::select! {
            result = &mut query => {
                result?;
                return Err::<(), Box<dyn std::error::Error>>("CPU fixture completed before the progress timer".into());
            }
            _ = tokio::time::sleep(Duration::from_millis(2)) => {}
        }
        assert_eq!(database.status()?.active_operations, 1);
        let result = tokio::time::timeout(Duration::from_secs(10), query).await??;
        assert!(matches!(&result.rows[0][0], TypedValue::Integer(value) if value == "500000500000"));
        assert_eq!(database.status()?.active_operations, 0);
        TestResult::Ok(())
    })?;
    database.close()?;
    Ok(())
}

#[test]
fn cancellation_releases_cpu_work_and_operation_admission() -> TestResult {
    let _guard = TESTS.lock().map_err(|_| "test lock poisoned")?;
    let directory = tempfile::tempdir()?;
    let database = Arc::new(open(directory.path())?);
    let caller = current_thread()?;
    caller.block_on(async {
        let worker_database = Arc::clone(&database);
        let worker = tokio::spawn(async move {
            worker_database.query_with_options_async(
                Query::new("USE async_cpu UNWIND range(1, 5000000) AS value RETURN sum(value)"),
                OperationOptions { operation_id: Some("cancel-cpu".to_owned()), timeout_ms: Some(10_000) },
            ).await
        });
        tokio::time::timeout(Duration::from_secs(1), async {
            while database.status().map(|status| status.active_operations).unwrap_or(0) == 0 {
                tokio::task::yield_now().await;
            }
        }).await?;
        tokio::time::sleep(Duration::from_millis(2)).await;
        assert!(database.cancel("cancel-cpu"));
        let error = tokio::time::timeout(Duration::from_secs(2), worker).await??.err()
            .ok_or("cancelled CPU query unexpectedly succeeded")?;
        assert!(matches!(error,
            EmbeddedError::Engine(ref error) if error.code == irongraph_types::ErrorCode::Cancelled
        ) || matches!(error,
            EmbeddedError::Query(ClientError::Database { ref code, .. }) if code == "Cancelled" || code == "CANCELLED"
        ), "unexpected cancellation error: {error}");
        assert_eq!(database.status()?.active_operations, 0);
        assert!(!database.cancel("cancel-cpu"));
        let result = database.query_async(Query::new("USE async_cpu RETURN 7")).await?;
        assert!(matches!(&result.rows[0][0], TypedValue::Integer(value) if value == "7"));
        TestResult::Ok(())
    })?;
    Arc::try_unwrap(database)
        .map_err(|_| "CPU worker retained database handle")?
        .close()?;
    Ok(())
}

#[test]
fn parallel_async_calls_share_one_database_and_observe_committed_values() -> TestResult {
    let _guard = TESTS.lock().map_err(|_| "test lock poisoned")?;
    let directory = tempfile::tempdir()?;
    let database = Arc::new(open(directory.path())?);
    let caller = current_thread()?;
    caller.block_on(async {
        database.query_async(Query::new("USE async_cpu CREATE (:Shared {value: 1})")).await?;
        for expected in [1, 2] {
            if expected == 2 {
                database.query_async(Query::new("USE async_cpu MATCH (n:Shared) SET n.value = 2")).await?;
            }
            let mut workers = tokio::task::JoinSet::new();
            for _ in 0..32 {
                let shared = Arc::clone(&database);
                assert!(Arc::ptr_eq(&database, &shared));
                workers.spawn(async move {
                    shared.query_async(Query::new("USE async_cpu MATCH (n:Shared) RETURN n.value")).await
                });
            }
            while let Some(result) = workers.join_next().await {
                let result = result??;
                assert_eq!(result.rows.len(), 1);
                assert!(matches!(&result.rows[0][0], TypedValue::Integer(value) if value == &expected.to_string()));
            }
            assert_eq!(database.status()?.active_operations, 0);
        }
        TestResult::Ok(())
    })?;
    Arc::try_unwrap(database)
        .map_err(|_| "parallel query retained database handle")?
        .close()?;
    Ok(())
}
