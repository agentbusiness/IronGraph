use super::*;
pub(super) static TESTS: std::sync::Mutex<()> = std::sync::Mutex::new(());

#[test]
fn operation_admission_cancellation_and_release()
-> std::result::Result<(), Box<dyn std::error::Error>> {
    let _guard = TESTS.lock().unwrap();
    let directory = tempfile::tempdir()?;
    let mut options = EmbeddedOptions::new(directory.path())
        .with_execution_device(ExecutionDevice::Cpu)
        .with_embedding_policy(EmbeddingPolicy::Disabled);
    options.max_concurrent_operations = 1;
    let database = EmbeddedDatabase::open(options)?;
    let active = database.begin_operation(OperationOptions {
        operation_id: Some("host-operation".into()),
        timeout_ms: Some(60_000),
    })?;
    assert_eq!(database.status()?.active_operations, 1);
    let rejected = database.query(Query::new("SHOW PROJECTS")).unwrap_err();
    assert!(
        matches!(rejected, EmbeddedError::Engine(error) if error.code == irongraph_types::ErrorCode::Backpressure)
    );
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
