//! Release benchmark of the single canonical CPU store. Full Cypher workloads match the
//! stored performance matrix; scalar/metadata measurements use separately named timing boundaries.
// This executable deliberately uses a sequential workload driver rather than a library API.
#![allow(clippy::all, clippy::too_many_lines)]
use irongraph::{
    Bookmark, EdgeId, Layer, NodeId, ProjectId, Result, ScalarValue,
    cypher::{BindCapabilities, ExecutionContext, ExecutionOutput, QueryEngine},
    graph::{EdgeInput, GraphMutation, GraphStore, NodeInput, StatisticsSnapshot},
};
use serde::Serialize;
use std::{
    collections::BTreeMap,
    fs,
    hint::black_box,
    path::PathBuf,
    process::Command,
    sync::{Arc, mpsc},
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio_util::sync::CancellationToken;
use uuid::Uuid;
const PROJECT: ProjectId = ProjectId(Uuid::nil());
const BUCKETS: i64 = 64;
fn required<T>(value: Result<T>, context: &str) -> Result<T> {
    value.map_err(|error| irongraph::Error::internal(format!("{context}: {error}")))
}
#[derive(Debug, Serialize)]
struct Measurement {
    backend: String,
    nodes: usize,
    edges: usize,
    category: &'static str,
    operation: String,
    status: &'static str,
    error: Option<String>,
    warmups: usize,
    samples: usize,
    unit: &'static str,
    min: Option<f64>,
    p50: Option<f64>,
    p95: Option<f64>,
    p99: Option<f64>,
    max: Option<f64>,
    mean: Option<f64>,
    throughput_per_second: Option<f64>,
    elements_per_sample: usize,
    resident_bytes: Option<usize>,
    durability: &'static str,
    result_rows: Option<usize>,
}

#[derive(Clone, Copy)]
struct GraphIds {
    label: irongraph::types::LabelId,
    value: irongraph::types::PropertyId,
    bucket: irongraph::types::PropertyId,
    body: irongraph::types::PropertyId,
    relationship: irongraph::types::RelationshipTypeId,
}

fn build_graph(
    nodes: usize,
    fanout: usize,
    dirty_body_bytes: usize,
) -> Result<(GraphStore, GraphIds, Duration, Duration)> {
    let graph = GraphStore::default();
    let ids = GraphIds {
        label: required(graph.catalog_mut().intern_label("Node"), "intern label")?,
        value: required(
            graph.catalog_mut().intern_property("value"),
            "intern value property",
        )?,
        bucket: required(
            graph.catalog_mut().intern_property("bucket"),
            "intern bucket property",
        )?,
        body: required(
            graph.catalog_mut().intern_property("body"),
            "intern body property",
        )?,
        relationship: required(
            graph.catalog_mut().intern_relationship_type("R"),
            "intern relationship type",
        )?,
    };
    let dirty_stride = (nodes / 100).max(1);
    let dirty = "x".repeat(dirty_body_bytes);
    let node_started = Instant::now();
    for row in 0..nodes {
        let mut properties = vec![
            (
                ids.value,
                ScalarValue::Integer(
                    i64::try_from(row % 1_000)
                        .map_err(|error| irongraph::Error::internal(error.to_string()))?,
                ),
            ),
            (
                ids.bucket,
                ScalarValue::Integer(
                    i64::try_from(
                        row % usize::try_from(BUCKETS)
                            .map_err(|error| irongraph::Error::internal(error.to_string()))?,
                    )
                    .map_err(|error| irongraph::Error::internal(error.to_string()))?,
                ),
            ),
        ];
        if row % dirty_stride == 0 {
            properties.push((
                ids.body,
                ScalarValue::String(format!("{row}:{dirty}").into()),
            ));
        }
        required(
            graph.insert_node(NodeInput {
                id: NodeId(row as u64 + 1),
                layer: Layer::Observed,
                revision: 1,
                labels: vec![ids.label],
                properties,
            }),
            "insert benchmark node",
        )?;
    }
    let node_elapsed = node_started.elapsed();
    let edge_started = Instant::now();
    let mut edge_id = 1_u64;
    for source in 0..nodes {
        for step in 1..=fanout {
            let target = (source + step.saturating_mul(7_919)) % nodes;
            required(
                graph.insert_edge(EdgeInput {
                    id: EdgeId(edge_id),
                    source: NodeId(source as u64 + 1),
                    target: NodeId(target as u64 + 1),
                    relationship_type: ids.relationship,
                    layer: Layer::Observed,
                    revision: 1,
                    properties: Vec::new(),
                }),
                "insert benchmark relationship",
            )?;
            edge_id += 1;
        }
    }
    Ok((graph, ids, node_elapsed, edge_started.elapsed()))
}

#[allow(
    clippy::cast_precision_loss,
    clippy::needless_pass_by_value,
    clippy::too_many_arguments
)]
fn distribution(
    backend: &str,
    nodes: usize,
    edges: usize,
    category: &'static str,
    operation: impl Into<String>,
    samples: Vec<Duration>,
    warmups: usize,
    elements: usize,
    resident_bytes: Option<usize>,
    durability: &'static str,
    result_rows: Option<usize>,
) -> Measurement {
    let mut micros = samples
        .iter()
        .map(|sample| sample.as_secs_f64() * 1_000_000.0)
        .collect::<Vec<_>>();
    micros.sort_by(f64::total_cmp);
    let percentile = |percent: usize| {
        let index = micros
            .len()
            .saturating_sub(1)
            .saturating_mul(percent)
            .div_ceil(100);
        micros.get(index).copied()
    };
    let mean = (!micros.is_empty()).then(|| micros.iter().sum::<f64>() / micros.len() as f64);
    let p50 = percentile(50);
    Measurement {
        backend: backend.to_owned(),
        nodes,
        edges,
        category,
        operation: operation.into(),
        status: "ok",
        error: None,
        warmups,
        samples: micros.len(),
        unit: "microseconds",
        min: micros.first().copied(),
        p50,
        p95: percentile(95),
        p99: percentile(99),
        max: micros.last().copied(),
        mean,
        throughput_per_second: p50
            .filter(|value| *value > 0.0)
            .map(|value| elements as f64 * 1_000_000.0 / value),
        elements_per_sample: elements,
        resident_bytes,
        durability,
        result_rows,
    }
}

const fn query_workloads() -> &'static [(&'static str, &'static str, &'static str)] {
    &[
        (
            "read",
            "point_lookup",
            "MATCH (n:Node) WHERE n.value = 42 RETURN n.value LIMIT 1",
        ),
        (
            "read",
            "range_count",
            "MATCH (n:Node) WHERE n.value >= 400 AND n.value < 600 RETURN count(n)",
        ),
        (
            "read",
            "projection",
            "MATCH (n:Node) RETURN n.value LIMIT 1000",
        ),
        (
            "traversal",
            "one_hop",
            "MATCH (n:Node)-[:R]->(m) WHERE n.value = 42 RETURN count(m)",
        ),
        (
            "traversal",
            "two_hop",
            "MATCH (n:Node)-[:R]->()-[:R]->(m) WHERE n.value = 42 RETURN count(m)",
        ),
        (
            "traversal",
            "variable_1_3",
            "MATCH (n:Node)-[:R*1..3]->(m) WHERE n.value = 42 RETURN count(DISTINCT m)",
        ),
        ("aggregate", "count_nodes", "MATCH (n:Node) RETURN count(n)"),
        (
            "aggregate",
            "count_edges",
            "MATCH ()-[r:R]->() RETURN count(r)",
        ),
        ("aggregate", "sum", "MATCH (n:Node) RETURN sum(n.value)"),
        ("aggregate", "avg", "MATCH (n:Node) RETURN avg(n.value)"),
        (
            "aggregate",
            "min_max",
            "MATCH (n:Node) RETURN min(n.value), max(n.value)",
        ),
        (
            "aggregate",
            "group_count",
            "MATCH (n:Node) RETURN n.bucket, count(*)",
        ),
        (
            "aggregate",
            "distinct",
            "MATCH (n:Node) RETURN count(DISTINCT n.bucket)",
        ),
        (
            "aggregate",
            "top_k",
            "MATCH (n:Node) RETURN n.value ORDER BY n.value DESC LIMIT 10",
        ),
    ]
}

fn adaptive_algorithm_workloads(nodes: usize) -> Vec<(String, String)> {
    let target = nodes.max(1);
    vec![
        (
            String::from("adaptive_degree"),
            String::from("CALL graph.degree() YIELD node, degree RETURN count(*)"),
        ),
        (
            String::from("adaptive_bfs"),
            String::from("CALL graph.bfs(1) YIELD node, distance RETURN count(*)"),
        ),
        (
            String::from("adaptive_dfs"),
            String::from("CALL graph.dfs(1) YIELD node, order RETURN count(*)"),
        ),
        (
            String::from("adaptive_shortest_path"),
            format!("CALL graph.shortestpath(1, {target}) YIELD path, cost RETURN count(*)"),
        ),
        (
            String::from("adaptive_dijkstra"),
            String::from("CALL graph.dijkstra(1) YIELD node, cost RETURN count(*)"),
        ),
        (
            String::from("adaptive_wcc"),
            String::from("CALL graph.wcc() YIELD node, component RETURN count(*)"),
        ),
        (
            String::from("adaptive_scc"),
            String::from("CALL graph.scc() YIELD node, component RETURN count(*)"),
        ),
        (
            String::from("adaptive_pagerank"),
            String::from("CALL graph.pagerank() YIELD node, score RETURN count(*)"),
        ),
        (
            String::from("adaptive_triangle_count"),
            String::from("CALL graph.trianglecount() YIELD triangleCount RETURN triangleCount"),
        ),
        (
            String::from("adaptive_clustering"),
            String::from(
                "CALL graph.clusteringcoefficient() YIELD node, coefficient RETURN count(*)",
            ),
        ),
        (
            String::from("adaptive_k_core"),
            String::from("CALL graph.kcore() YIELD node, core RETURN count(*)"),
        ),
        (
            String::from("adaptive_louvain"),
            String::from("CALL graph.louvain() YIELD node, community RETURN count(*)"),
        ),
    ]
}

fn context<'a>(graph: &'a GraphStore, statistics: &'a StatisticsSnapshot) -> ExecutionContext<'a> {
    ExecutionContext {
        project_id: PROJECT,
        graph,
        binding_catalog: graph.catalog(),
        prior_graph_mutations: &[],
        temporal: None,
        prior_temporal_mutations: &[],
        vector_indexes: BTreeMap::new(),
        scalar_indexes: None,
        text_embedding: None,
        parameters: BTreeMap::new(),
        bookmark: Bookmark { term: 1, index: 1 },
        mutation_revision: graph.revision() + 1,
        resolved_time_nanos: 0,
        next_node_id: graph.node_slot_count() as u64 + 1,
        next_edge_id: graph.edge_slot_count() as u64 + 1,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities::default(),
        max_result_rows: 8_000_000,
        max_batch_rows: 65_536,
        optimizer_statistics: Some(statistics),
        backend: None,
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_hours(1)),
        resolved_query_at_time_nanos: None,
    }
}
fn query(
    graph: &GraphStore,
    statistics: &StatisticsSnapshot,
    source: &str,
) -> Result<ExecutionOutput> {
    QueryEngine.execute(source, &mut context(graph, statistics))
}
trait MeasurementValue {
    fn row_count(&self) -> usize;
}
impl MeasurementValue for usize {
    fn row_count(&self) -> usize {
        *self
    }
}
impl MeasurementValue for ExecutionOutput {
    fn row_count(&self) -> usize {
        self.result
            .batches
            .iter()
            .map(|batch| batch.row_count)
            .sum()
    }
}
fn measure<T: MeasurementValue>(
    warmups: usize,
    samples: usize,
    mut operation: impl FnMut() -> Result<T>,
) -> Result<(Vec<Duration>, usize)> {
    for _ in 0..warmups {
        black_box(operation()?);
    }
    let mut times = Vec::new();
    let mut rows = 0;
    for _ in 0..samples {
        let start = Instant::now();
        let value = black_box(operation()?);
        times.push(start.elapsed());
        // Historical full-query measurements stop before inspecting or dropping the output.
        rows = value.row_count();
    }
    Ok((times, rows))
}
fn shell(program: &str, args: &[&str]) -> String {
    Command::new(program)
        .args(args)
        .output()
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_owned())
        .unwrap_or_default()
}
fn rss() -> usize {
    shell(
        "/bin/ps",
        &["-o", "rss=", "-p", &std::process::id().to_string()],
    )
    .parse::<usize>()
    .unwrap_or(0)
        * 1024
}
/// Persistent bounded readers execute on shared handles to the same canonical allocation.
struct ReadPool {
    senders: Vec<mpsc::SyncSender<usize>>,
    results: mpsc::Receiver<Result<usize>>,
    workers: Vec<std::thread::JoinHandle<()>>,
}
impl ReadPool {
    fn new(graph: &GraphStore, statistics: &Arc<StatisticsSnapshot>, threads: usize) -> Self {
        let (result_tx, results) = mpsc::channel();
        let mut senders = Vec::new();
        let mut workers = Vec::new();
        for _ in 0..threads {
            let (send, recv) = mpsc::sync_channel(1);
            let graph = graph.clone();
            let statistics = Arc::clone(statistics);
            let result_tx = result_tx.clone();
            workers.push(std::thread::spawn(move || {
                while let Ok(count) = recv.recv() {
                    if count == 0 {
                        break;
                    }
                    let result = (0..count).try_fold(0, |sum, _| {
                        query(&graph, &statistics, "MATCH (n:Node) RETURN count(n)")
                            .map(|_| sum + 1)
                    });
                    if result_tx.send(result).is_err() {
                        break;
                    }
                }
            }));
            senders.push(send);
        }
        Self {
            senders,
            results,
            workers,
        }
    }
    fn count_queries(&self, count: usize) -> Result<usize> {
        for worker in &self.senders {
            worker
                .send(count)
                .map_err(|_| irongraph::Error::internal("read worker closed"))?;
        }
        let mut rows = 0;
        for _ in &self.senders {
            rows += self
                .results
                .recv()
                .map_err(|_| irongraph::Error::internal("read worker result closed"))??;
        }
        Ok(rows)
    }
}
impl Drop for ReadPool {
    fn drop(&mut self) {
        for worker in &self.senders {
            let _ = worker.send(0);
        }
        for worker in self.workers.drain(..) {
            let _ = worker.join();
        }
    }
}
fn main() -> std::process::ExitCode {
    match run() {
        Ok(()) => std::process::ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("canonical performance: {error}");
            std::process::ExitCode::FAILURE
        }
    }
}
fn run() -> Result<()> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let get = |key: &str| {
        args.windows(2)
            .find(|pair| pair[0] == key)
            .map(|pair| pair[1].clone())
    };
    let sizes = get("--sizes")
        .unwrap_or_else(|| "100,10000,100000,1000000,2000000".into())
        .split(',')
        .map(|n| {
            n.parse::<usize>()
                .map_err(|error| irongraph::Error::internal(error.to_string()))
        })
        .collect::<Result<Vec<_>>>()?;
    let parse_number = |key: &str, default| {
        get(key).map_or(Ok(default), |value| {
            value
                .parse::<usize>()
                .map_err(|error| irongraph::Error::internal(error.to_string()))
        })
    };
    let samples = parse_number("--samples", 5)?;
    let warmups = parse_number("--warmups", 2)?;
    if samples == 0 || sizes.iter().any(|size| *size == 0) {
        return Err(irongraph::Error::internal(
            "benchmark sizes and samples must be positive",
        ));
    }
    let operations = get("--operations")
        .unwrap_or_default()
        .split(',')
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let wants = |name: &str| operations.is_empty() || operations.iter().any(|n| n == name);
    let output = PathBuf::from(
        get("--output")
            .unwrap_or_else(|| "performance-results/cpu-canonical/current/results.json".into()),
    );
    let mut measurements = Vec::new();
    let mut memory = Vec::new();
    for &nodes in &sizes {
        eprintln!("CANONICAL fixture nodes={nodes} fanout=4 dirty_body_bytes=2048");
        let before = rss();
        let (graph, ids, node_time, edge_time) = build_graph(nodes, 4, 2048)?;
        let edges = graph.edge_count();
        let stored = graph.resident_bytes();
        memory.push(serde_json::json!({"nodes":nodes,"before_fixture_rss":before,"loaded_rss":rss(),"accounted_canonical_bytes":stored,"node_slots":graph.node_slot_count(),"edge_slots":graph.edge_slot_count()}));
        for (name, time, elements) in [
            ("initial_node_ingest", node_time, nodes),
            ("initial_edge_ingest", edge_time, edges),
        ] {
            measurements.push(distribution(
                "cpu-canonical",
                nodes,
                edges,
                "write",
                name,
                vec![time],
                0,
                elements,
                Some(stored),
                "memory",
                Some(elements),
            ));
        }
        let statistics = Arc::new(StatisticsSnapshot::collect(&graph));
        for (category, name, source) in query_workloads()
            .iter()
            .map(|(a, b, c)| (*a, (*b).to_owned(), (*c).to_owned()))
            .chain(
                adaptive_algorithm_workloads(nodes)
                    .into_iter()
                    .map(|(a, b)| ("adaptive_algorithm", a, b)),
            )
        {
            if !wants(&name) {
                continue;
            }
            eprintln!("CANONICAL nodes={nodes} operation={name}");
            let measured = measure(warmups, samples, || query(&graph, &statistics, &source));
            match measured {
                Ok((times, rows)) => measurements.push(distribution(
                    "cpu-canonical",
                    nodes,
                    edges,
                    category,
                    &name,
                    times,
                    warmups,
                    nodes + edges,
                    Some(stored),
                    "none",
                    Some(rows),
                )),
                Err(error) => return Err(error),
            }
            memory.push(
                serde_json::json!({"nodes":nodes,"operation":name,"after_operation_rss":rss()}),
            );
        }
        memory.push(serde_json::json!({"nodes":nodes,"stage":"after_queries_algorithms_before_threads","rss":rss()}));
        for name in ["canonical_metadata_capture", "canonical_scalar_point"] {
            if !wants(name) {
                continue;
            }
            let (times, rows) = measure(warmups, samples, || {
                for _ in 0..10000 {
                    if name == "canonical_metadata_capture" {
                        black_box((
                            graph.catalog().optimizer_generation(),
                            graph.optimizer_count_generations(),
                            graph.resident_bytes(),
                            graph.revision(),
                        ));
                    } else {
                        black_box(
                            graph
                                .node(NodeId(43.min(nodes as u64)))
                                .and_then(|node| node.property(ids.value)),
                        );
                    }
                }
                Ok(10000)
            })?;
            measurements.push(distribution(
                "cpu-canonical",
                nodes,
                edges,
                "primitive",
                name,
                times.into_iter().map(|t| t / 10000).collect(),
                warmups,
                1,
                Some(stored),
                "memory",
                Some(rows),
            ));
        }
        if wants("parallel_full_count_queries") {
            let threads = std::thread::available_parallelism()
                .map_or(1, usize::from)
                .min(8);
            let pool = ReadPool::new(&graph, &statistics, threads);
            let (times, rows) = measure(warmups, samples, || pool.count_queries(100))?;
            measurements.push(distribution(
                "cpu-canonical",
                nodes,
                edges,
                "concurrency",
                "parallel_full_count_queries",
                times,
                warmups,
                threads * 100,
                Some(stored),
                "memory",
                Some(rows),
            ));
        }
        if wants("reader_while_writer_paused") {
            let mut times = Vec::new();
            for _ in 0..samples {
                let revision = graph.revision() + 1;
                let (paused_tx, paused_rx) = mpsc::channel();
                let (resume_tx, resume_rx) = mpsc::channel();
                let (read_tx, read_rx) = mpsc::channel();
                std::thread::scope(|scope| -> Result<()> {
                    let writer_graph = &graph;
                    let writer = scope.spawn(move || -> Result<()> {
                        writer_graph.apply(GraphMutation::SetNodeProperty {
                            node: NodeId(1),
                            property: ids.value,
                            value: ScalarValue::Integer(7),
                            revision,
                        })?;
                        paused_tx
                            .send(())
                            .map_err(|error| irongraph::Error::internal(error.to_string()))?;
                        resume_rx
                            .recv()
                            .map_err(|error| irongraph::Error::internal(error.to_string()))?;
                        writer_graph.apply(GraphMutation::SetNodeProperty {
                            node: NodeId(1),
                            property: ids.value,
                            value: ScalarValue::Integer(0),
                            revision: revision + 1,
                        })?;
                        Ok(())
                    });
                    paused_rx
                        .recv()
                        .map_err(|error| irongraph::Error::internal(error.to_string()))?;
                    let reader = scope.spawn(|| {
                        let started = Instant::now();
                        let result = query(&graph, &statistics, "MATCH (n:Node) RETURN count(n)");
                        read_tx
                            .send((started.elapsed(), result))
                            .map_err(|error| irongraph::Error::internal(error.to_string()))
                    });
                    let result = read_rx.recv_timeout(Duration::from_secs(2));
                    resume_tx
                        .send(())
                        .map_err(|error| irongraph::Error::internal(error.to_string()))?;
                    writer
                        .join()
                        .map_err(|_| irongraph::Error::internal("paused writer panicked"))??;
                    reader
                        .join()
                        .map_err(|_| irongraph::Error::internal("paused reader panicked"))??;
                    let (time, rows) = result.map_err(|error| {
                        irongraph::Error::internal(format!(
                            "reader did not complete before writer resumed: {error}"
                        ))
                    })?;
                    if rows?.row_count() != 1 {
                        return Err(irongraph::Error::internal(
                            "paused reader returned incorrect row count",
                        ));
                    }
                    times.push(time);
                    Ok(())
                })?;
            }
            measurements.push(distribution(
                "cpu-canonical",
                nodes,
                edges,
                "concurrency",
                "reader_while_writer_paused",
                times,
                0,
                1,
                Some(stored),
                "memory",
                Some(1),
            ));
        }
        if wants("canonical_batch_insert") {
            let mut id = graph.node_slot_count() as u64 + 1;
            let revision = graph.revision() + 1;
            let (times, rows) = measure(warmups, samples, || {
                for offset in 0..256 {
                    graph.insert_node(NodeInput {
                        id: NodeId(id),
                        layer: Layer::Observed,
                        revision,
                        labels: vec![ids.label],
                        properties: vec![(ids.value, ScalarValue::Integer(offset))],
                    })?;
                    id += 1;
                }
                Ok(256)
            })?;
            measurements.push(distribution(
                "cpu-canonical",
                nodes,
                edges,
                "write",
                "canonical_batch_insert",
                times,
                warmups,
                256,
                Some(graph.resident_bytes()),
                "memory",
                Some(rows),
            ));
        }
        memory.push(serde_json::json!({"nodes":nodes,"after_queries_rss":rss(),"final_accounted_canonical_bytes":graph.resident_bytes()}));
        let report = serde_json::json!({"schema":"irongraph.performance-matrix.v1","generated_unix_millis":SystemTime::now().duration_since(UNIX_EPOCH).map_err(|error| irongraph::Error::internal(error.to_string()))?.as_millis(),"git_revision":shell("git", &["rev-parse","HEAD"]),"git_dirty":!shell("git", &["status","--porcelain"]).is_empty(),"os":std::env::consts::OS,"arch":std::env::consts::ARCH,"cpu_model":shell("/usr/sbin/sysctl", &["-n","machdep.cpu.brand_string"]),"physical_memory_bytes":shell("/usr/sbin/sysctl", &["-n","hw.memsize"]).parse::<u64>().ok(),"metal_devices":[],"parallelism":std::thread::available_parallelism().map_or(1,usize::from),"config":{"sizes":sizes,"fanout":4,"samples":samples,"warmups":warmups,"backends":["cpu-canonical"],"dirty_body_bytes":2048,"batch_rows":256,"operations":operations},"timing_boundaries":{"queries":"full QueryEngine execute including context construction, cached plan lookup, canonical CPU execution, and result batch construction; elapsed stops with ExecutionOutput alive, before row counting and output destruction, matching stored full-query timer; fixture and statistics outside timer","primitives":"10000 live canonical scalar reads or metadata captures per sample, elapsed divided by10000; these are not full Cypher query latency","batch_insert":"256 direct canonical inserts with no graph clone; each iteration allocates new IDs","concurrency":"full Cypher count queries sharing the same canonical store; persistent bounded worker pool created outside timer; threads<=8;100 queries per thread","paused_writer":"full Cypher count started after writer changed first record and stopped before second change; reader must complete before writer resumes"},"memory":memory,"measurements":measurements});
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| irongraph::Error::internal(error.to_string()))?;
        }
        let bytes = serde_json::to_vec_pretty(&report)
            .map_err(|error| irongraph::Error::internal(error.to_string()))?;
        fs::write(&output, bytes).map_err(|error| irongraph::Error::internal(error.to_string()))?;
    }
    Ok(())
}
