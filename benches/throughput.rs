#![allow(clippy::all, clippy::nursery, clippy::pedantic)]

//! Read/write/aggregation throughput across graph sizes. Custom harness (not criterion): builds a
//! synthetic labelled graph with an integer property, measures write (ingest) throughput, then runs
//! representative Cypher reads and aggregations and reports their latency/throughput.

use std::{
    collections::BTreeMap,
    time::{Duration, Instant},
};

use irongraph::{
    Bookmark, EdgeId, Layer, ProjectId, Result, ScalarValue,
    cypher::{BindCapabilities, ExecutionContext, ExecutionOutput, QueryEngine},
    gpu::{CpuBackend, ExecutionBackend, ResidentProjectImage},
    graph::{EdgeInput, GraphStore, IndexCatalog, NodeInput, StatisticsSnapshot, TemporalStore},
    types::NodeId,
};
use tokio_util::sync::CancellationToken;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());
const BOOKMARK: Bookmark = Bookmark { term: 1, index: 1 };
const BUCKETS: i64 = 16;

fn die<T>(r: Result<T>) -> T {
    match r {
        Ok(v) => v,
        Err(e) => {
            eprintln!("bench failed: {e}");
            std::process::exit(2);
        }
    }
}

/// Builds a labelled graph of `nodes` nodes (each with `value` + `bucket` integer properties) and
/// `nodes * fanout` edges. Returns the graph and (nodes/sec, edges/sec) write throughput.
fn build_graph(nodes: usize, fanout: usize) -> (GraphStore, f64, f64) {
    let mut g = GraphStore::default();
    let label = die(g.catalog_mut().intern_label("Node"));
    let value = die(g.catalog_mut().intern_property("value"));
    let bucket = die(g.catalog_mut().intern_property("bucket"));
    let rel = die(g.catalog_mut().intern_relationship_type("R"));

    let t0 = Instant::now();
    for i in 0..nodes {
        die(g.insert_node(NodeInput {
            id: NodeId(i as u64 + 1),
            layer: Layer::Observed,
            revision: 1,
            labels: vec![label],
            properties: vec![
                (value, ScalarValue::Integer((i as i64) % 1000)),
                (bucket, ScalarValue::Integer((i as i64) % BUCKETS)),
            ],
        }));
    }
    let write_nodes = nodes as f64 / t0.elapsed().as_secs_f64();

    let edges = nodes.saturating_mul(fanout);
    let t1 = Instant::now();
    let mut e = 0u64;
    for s in 0..nodes {
        for k in 1..=fanout {
            let target = (s + k * 7919) % nodes;
            die(g.insert_edge(EdgeInput {
                id: EdgeId(e + 1),
                source: NodeId(s as u64 + 1),
                target: NodeId(target as u64 + 1),
                relationship_type: rel,
                layer: Layer::Observed,
                revision: 1,
                properties: Vec::new(),
            }));
            e += 1;
        }
    }
    let write_edges = edges as f64 / t1.elapsed().as_secs_f64();
    (g, write_nodes, write_edges)
}

fn context<'a>(
    graph: &'a GraphStore,
    backend: &'a dyn ExecutionBackend,
    statistics: &'a StatisticsSnapshot,
) -> ExecutionContext<'a> {
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
        bookmark: BOOKMARK,
        mutation_revision: graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id: 1,
        next_edge_id: 1,
        predicate_versions: BTreeMap::new(),
        // Production mode: allow host fallback so we measure realistic latency, not native-only.
        capabilities: BindCapabilities::default(),
        max_result_rows: 8_000_000,
        max_batch_rows: 65_536,
        // Production supplies a pre-warmed statistics snapshot (Database::execute); mirror that here
        // so the benchmark measures the same planner + fast-path behavior production runs.
        optimizer_statistics: Some(statistics),
        backend: Some(backend),
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(120)),
        resolved_query_at_time_nanos: None,
    }
}

fn run(
    graph: &GraphStore,
    backend: &dyn ExecutionBackend,
    statistics: &StatisticsSnapshot,
    query: &str,
) -> std::result::Result<(Duration, usize), String> {
    let start = Instant::now();
    let out: ExecutionOutput = QueryEngine
        .execute(query, &mut context(graph, backend, statistics))
        .map_err(|e| e.to_string())?;
    let elapsed = start.elapsed();
    let rows: usize = out.result.batches.iter().map(|b| b.row_count).sum();
    Ok((elapsed, rows))
}

/// (label, cypher) read queries spanning aggregation, distinct, ordering/top-K, point/range
/// predicates, projection, and multi-hop traversal. Kept non-explosive so 1M stays tractable.
fn queries() -> Vec<(&'static str, &'static str)> {
    vec![
        ("count", "MATCH (n:Node) RETURN count(n) AS c"),
        ("count(*)", "MATCH (n:Node) RETURN count(*) AS c"),
        ("min", "MATCH (n:Node) RETURN min(n.value) AS m"),
        ("max", "MATCH (n:Node) RETURN max(n.value) AS m"),
        ("sum", "MATCH (n:Node) RETURN sum(n.value) AS s"),
        ("avg", "MATCH (n:Node) RETURN avg(n.value) AS a"),
        ("stdev", "MATCH (n:Node) RETURN stDev(n.value) AS d"),
        (
            "groupby",
            "MATCH (n:Node) RETURN n.bucket AS b, count(*) AS c",
        ),
        (
            "distinct_ct",
            "MATCH (n:Node) RETURN count(DISTINCT n.bucket) AS c",
        ),
        (
            "distinct_rows",
            "MATCH (n:Node) RETURN DISTINCT n.bucket AS b",
        ),
        (
            "where_eq",
            "MATCH (n:Node) WHERE n.value = 42 RETURN count(n) AS c",
        ),
        (
            "where_range",
            "MATCH (n:Node) WHERE n.value >= 400 AND n.value < 600 RETURN count(n) AS c",
        ),
        (
            "scan_lt",
            "MATCH (n:Node) WHERE n.value < 100 RETURN count(n) AS c",
        ),
        (
            "topk",
            "MATCH (n:Node) RETURN n.value AS v ORDER BY v DESC LIMIT 10",
        ),
        (
            "proj_limit",
            "MATCH (n:Node) RETURN n.value AS v LIMIT 1000",
        ),
        (
            "proj_math",
            "MATCH (n:Node) RETURN n.value * 2 + 1 AS v LIMIT 1000",
        ),
        ("count_r", "MATCH ()-[r:R]->() RETURN count(r) AS c"),
        ("count_r_any", "MATCH ()-[r]->() RETURN count(r) AS c"),
        ("1hop", "MATCH (n:Node)-[:R]->(m) RETURN count(m) AS c"),
        (
            "1hop_filter",
            "MATCH (n:Node)-[:R]->(m) WHERE m.value < 100 RETURN count(m) AS c",
        ),
        (
            "2hop",
            "MATCH (n:Node)-[:R]->()-[:R]->(m) RETURN count(m) AS c",
        ),
        (
            "reach2",
            "MATCH (n:Node)-[:R*1..2]->(m) RETURN count(DISTINCT m) AS c",
        ),
        (
            "reach3",
            "MATCH (n:Node)-[:R*1..3]->(m) RETURN count(DISTINCT m) AS c",
        ),
    ]
}

fn main() {
    let sizes: Vec<usize> = std::env::var("IGBENCH_SIZES")
        .ok()
        .map(|s| {
            s.split(',')
                .filter_map(|v| v.trim().parse::<usize>().ok())
                .collect()
        })
        .unwrap_or_else(|| vec![1_000, 10_000, 100_000, 1_000_000]);
    let fanout: usize = std::env::var("IGBENCH_FANOUT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4);

    println!(
        "backend=cpu-reference os={} arch={} fanout={}",
        std::env::consts::OS,
        std::env::consts::ARCH,
        fanout
    );

    let queries = queries();
    // times[query_index] = per-size microseconds
    let mut times: Vec<Vec<f64>> = vec![Vec::with_capacity(sizes.len()); queries.len()];
    let mut write_nodes = Vec::new();
    let mut write_edges = Vec::new();

    for &nodes in &sizes {
        let (graph, wn, we) = build_graph(nodes, fanout);
        write_nodes.push(wn);
        write_edges.push(we);
        let image = die(ResidentProjectImage::build(
            PROJECT,
            BOOKMARK,
            &graph,
            &TemporalStore::default(),
            &IndexCatalog::default(),
        ));
        // Mirror production: unbounded device-memory ceiling (no artificial sub-hardware cap), with
        // no host reserve for the isolated benchmark process.
        let mut cpu = CpuBackend::new(irongraph::config::UNBOUNDED_DEVICE_MEMORY_BYTES, 0);
        die(cpu.admit_project(image));

        // Pre-warmed statistics snapshot, exactly as production maintains one per project.
        let statistics = StatisticsSnapshot::collect_project(
            &graph,
            Some(&TemporalStore::default()),
            Some(&IndexCatalog::default()),
        );

        // warm the live node- and edge-count caches so per-query timings reflect the O(1) amortized
        // steady state production runs in, not the one-time lazy build.
        let _ = run(
            &graph,
            &cpu,
            &statistics,
            "MATCH (n:Node) RETURN count(n) AS c",
        );
        let _ = run(
            &graph,
            &cpu,
            &statistics,
            "MATCH ()-[r]->() RETURN count(r) AS c",
        );

        for (index, (name, q)) in queries.iter().enumerate() {
            match run(&graph, &cpu, &statistics, q) {
                Ok((dt, _)) => times[index].push(dt.as_secs_f64() * 1e6),
                Err(msg) => {
                    times[index].push(f64::NAN);
                    eprintln!("  ERR {name} @ {nodes} nodes: {msg}");
                }
            }
        }
    }

    // Transposed table: one row per query, one column per size (best for spotting O(1) vs O(N)).
    print!("\n{:>14}", "query \\ nodes");
    for &nodes in &sizes {
        print!("{:>13}", nodes);
    }
    println!();
    print!("{:>14}", "write N/s");
    for wn in &write_nodes {
        print!("{:>13.0}", wn);
    }
    println!();
    print!("{:>14}", "write E/s");
    for we in &write_edges {
        print!("{:>13.0}", we);
    }
    println!();
    for (index, (name, _)) in queries.iter().enumerate() {
        print!("{name:>14}");
        for value in &times[index] {
            print!("{:>11.0}µs", value);
        }
        println!();
    }
}
