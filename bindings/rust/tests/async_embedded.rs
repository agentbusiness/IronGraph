use std::sync::Arc;
use std::time::Duration;

use irongraph_sdk::{
    EmbeddedDatabase, EmbeddedOptions, EmbeddingPolicy, ExecutionDevice, OperationOptions, Query,
    StreamAppend, StreamFetch, StreamRecord,
};

#[test]
fn actual_native_async_lifecycle_timers_parallelism_and_cancellation()
-> Result<(), Box<dyn std::error::Error>> {
    let runtime = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .max_blocking_threads(2)
        .build()?;
    runtime.block_on(async {
        let directory = tempfile::tempdir()?;
        let options = EmbeddedOptions::new(directory.path()).with_execution_device(ExecutionDevice::Cpu).with_embedding_policy(EmbeddingPolicy::Disabled);
        let database = Arc::new(EmbeddedDatabase::open_async(options.clone()).await?);
        database.query_async(Query::new("CREATE PROJECT asyncsdk")).await?;
        let project = database.query_async(Query::new("USE asyncsdk RETURN 1")).await?.catalog.and_then(|catalog| catalog["project_id"].as_str().map(str::to_owned)).ok_or("project ID missing")?;
        database.query_async(Query::new("USE asyncsdk CREATE TOPIC events PARTITIONS 1")).await?;
        let ack = database.stream_append_async(StreamAppend { project_id: project.clone(), topic: "events".into(), partition: 0, records: vec![StreamRecord { key: None, headers: Default::default(), value: Some(vec![1,2,3]), create_time_ms: None }] }, OperationOptions::default()).await?;
        assert_eq!(ack.record_count, 1);
        let fetch = StreamFetch { project_id: project, topic: "events".into(), partition: 0, offset: 0, max_records: 1, max_bytes: 4096 };
        assert_eq!(database.stream_fetch_async(fetch.clone(), OperationOptions::default()).await?.records[0].1.payload, vec![1,2,3]);

        // Hold both native worker slots. Async polling must still run, excess callers
        // must await scheduling, and dropping queued operations must prevent publication.
        let (arrived, mut arrivals) = tokio::sync::mpsc::unbounded_channel();
        let mut releases = Vec::new();
        let mut blockers = Vec::new();
        for _ in 0..2 {
            let (release, wait) = std::sync::mpsc::channel();
            releases.push(release);
            let arrived = arrived.clone();
            blockers.push(tokio::task::spawn_blocking(move || { arrived.send(()).unwrap(); let _ = wait.recv_timeout(Duration::from_secs(5)); }));
        }
        for _ in 0..2 { arrivals.recv().await.ok_or("worker did not start")?; }
        let mut pending = Vec::new();
        for id in 0..64 {
            let database = database.clone();
            pending.push(tokio::spawn(async move { database.query_async(Query::new(format!("USE asyncsdk CREATE (:Cancelled {{value: {id}}})"))).await }));
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
        let queued_database = database.clone();
        let queued = tokio::spawn(async move { queued_database.query_async(Query::new("USE asyncsdk RETURN 1")).await });
        tokio::time::sleep(Duration::from_millis(20)).await;
        assert!(!queued.is_finished(), "saturated workers rejected or ran the queued query");
        for task in &pending { task.abort(); }
        for task in pending { assert!(task.await.unwrap_err().is_cancelled()); }
        for release in releases { release.send(())?; }
        for blocker in blockers { blocker.await?; }
        tokio::time::timeout(Duration::from_secs(5), queued).await???;
        let untouched = tokio::time::timeout(Duration::from_secs(5), database.query_async(Query::new("USE asyncsdk MATCH (n:Cancelled) RETURN count(n)"))).await??;
        assert_eq!(untouched.rows[0][0]["value"].as_str(), Some("0"));

        database.query_async(Query::new("USE asyncsdk UNWIND range(1,256) AS id CREATE (:Work {value:id})")).await?;
        let worker_database = database.clone();
        let active = tokio::spawn(async move { worker_database.query_with_options_async(Query::new("USE asyncsdk MATCH (a:Work), (b:Work), (c:Work) WHERE a.value+b.value+c.value > 0 RETURN sum(a.value)"), OperationOptions { operation_id: Some("active-sdk-cancel".into()), timeout_ms: Some(10_000) }).await });
        let deadline = tokio::time::Instant::now() + Duration::from_secs(3);
        loop {
            if database.status_async().await?.active_operations > 0 { break; }
            if active.is_finished() || tokio::time::Instant::now() >= deadline { return Err("native query did not become active".into()); }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
        let parallel = database.query_async(Query::new("USE asyncsdk MATCH (n:Work) RETURN count(n)")).await?;
        assert_eq!(parallel.rows[0][0]["value"].as_str(), Some("256"));
        active.abort();
        assert!(active.await.unwrap_err().is_cancelled());
        tokio::time::timeout(Duration::from_secs(3), async {
            while database.status_async().await?.active_operations != 0 { tokio::time::sleep(Duration::from_millis(2)).await; }
            Ok::<_,irongraph_sdk::Error>(())
        }).await??;
        database.flush_async().await?;
        database.snapshot_async().await?;
        Arc::try_unwrap(database).map_err(|_| "SDK owner remains borrowed")?.close_async().await?;
        let reopened = EmbeddedDatabase::open_async(options).await?;
        assert_eq!(reopened.stream_fetch_async(fetch, OperationOptions::default()).await?.high_watermark, 1);
        assert_eq!(reopened.query_async(Query::new("USE asyncsdk MATCH (n:Work) RETURN count(n)")).await?.rows[0][0]["value"].as_str(), Some("256"));
        reopened.close_async().await?;
        Ok::<_,Box<dyn std::error::Error>>(())
    })
}
