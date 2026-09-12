#![allow(clippy::all, clippy::nursery, clippy::pedantic)]

//! Reproducible CPU/Metal performance matrix. This is a custom harness because the required graph
//! sizes are too large for Criterion's per-benchmark fixture model and because failures, device
//! admission, durability behavior, and resident bytes must be recorded beside distributions.

use std::{
    collections::BTreeMap,
    fs,
    hint::black_box,
    path::{Path, PathBuf},
    process::Command,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

use async_trait::async_trait;
use futures::future::join_all;
use irongraph::{
    Bookmark, EdgeId, Layer, NodeId, ProjectId, Result, ScalarValue,
    cypher::{BindCapabilities, ExecutionContext, ExecutionOutput, QueryEngine},
    engine::{
        BackendSnapshot, CommandReservation, ExecutionClass, MutationApplyResult,
        MutationStateBackend, NodeIdentity, StandaloneNode, WriteCommand, WriteRequest,
        WriteRuntime, WriteStorageLimits,
    },
    execution::{
        ResidentGraphProcedure, ResidentGraphProcedureRequest, ResidentGraphProcedureResult,
    },
    gpu::{CpuBackend, ExecutionBackend, MetalBackend, ResidentProjectDelta, ResidentProjectImage},
    graph::{
        EdgeInput, GraphSnapshot, GraphStore, IndexCatalog, NodeInput, PageRankConfig,
        StatisticsSnapshot, TemporalStore, bfs, clustering_coefficients, dfs, dijkstra, k_core,
        louvain_communities, page_rank, shortest_path, strongly_connected_components,
        triangle_count, weakly_connected_components,
    },
    storage::{AdmissionClass, ConnectionId, DurableLog, MutationEntry, MutationKind},
};
use serde::Serialize;
use tokio_util::sync::CancellationToken;
use uuid::Uuid;

const PROJECT: ProjectId = ProjectId(Uuid::nil());
const TERM: u64 = 1;
const BUCKETS: i64 = 64;
const DEFAULT_SIZES: &[usize] = &[100, 10_000, 100_000, 1_000_000, 2_000_000];

#[derive(Clone, Debug)]
struct Config {
    sizes: Vec<usize>,
    fanout: usize,
    samples: usize,
    warmups: usize,
    backends: Vec<String>,
    output: PathBuf,
    dirty_body_bytes: usize,
    batch_rows: usize,
    durable_batch_rows: usize,
    operations: Vec<String>,
}

impl Config {
    fn wants(&self, operation: &str) -> bool {
        self.operations.is_empty() || self.operations.iter().any(|value| value == operation)
    }

    fn wants_any(&self, operations: &[&str]) -> bool {
        self.operations.is_empty() || operations.iter().any(|operation| self.wants(operation))
    }
}

#[derive(Debug, Serialize)]
struct Report {
    schema: &'static str,
    generated_unix_millis: u128,
    git_revision: String,
    git_dirty: bool,
    os: &'static str,
    arch: &'static str,
    cpu_model: String,
    physical_memory_bytes: Option<u64>,
    metal_devices: Vec<String>,
    parallelism: usize,
    config: ReportConfig,
    measurements: Vec<Measurement>,
}

#[derive(Debug, Serialize)]
struct ReportConfig {
    sizes: Vec<usize>,
    fanout: usize,
    samples: usize,
    warmups: usize,
    backends: Vec<String>,
    dirty_body_bytes: usize,
    batch_rows: usize,
    durable_batch_rows: usize,
    operations: Vec<String>,
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

#[derive(Default)]
struct AckBenchmarkBackend {
    applied: AtomicU64,
}

#[async_trait]
impl MutationStateBackend for AckBenchmarkBackend {
    async fn reserve_command(
        &self,
        _command: &WriteCommand,
        position: Bookmark,
    ) -> Result<CommandReservation> {
        CommandReservation::pipelined(position)
    }

    async fn apply_mutation(&self, mutation: &MutationEntry) -> Result<MutationApplyResult> {
        let previous = self.applied.fetch_add(1, Ordering::AcqRel);
        if mutation.index() != previous + 1 {
            return Err(irongraph::Error::internal(
                "ack benchmark observed a mutation-order gap",
            ));
        }
        Ok(MutationApplyResult::default())
    }

    async fn applied_bookmark(&self) -> Bookmark {
        Bookmark {
            term: TERM,
            index: self.applied.load(Ordering::Acquire),
        }
    }

    async fn build_snapshot(
        &self,
        _bookmark: Bookmark,
        _destination: &Path,
    ) -> Result<BackendSnapshot> {
        Err(irongraph::Error::internal(
            "snapshot is outside the acknowledgement benchmark",
        ))
    }

    async fn install_snapshot(
        &self,
        _bookmark: Bookmark,
        _snapshot: &BackendSnapshot,
    ) -> Result<()> {
        Err(irongraph::Error::internal(
            "snapshot is outside the acknowledgement benchmark",
        ))
    }
}

fn required<T>(result: Result<T>, context: &str) -> T {
    match result {
        Ok(value) => value,
        Err(error) => {
            eprintln!("{context}: {error}");
            std::process::exit(2);
        }
    }
}

fn parse_list(name: &str, defaults: &[usize]) -> Vec<usize> {
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .split(',')
                .filter_map(|part| part.trim().parse::<usize>().ok())
                .filter(|value| *value > 0)
                .collect::<Vec<_>>()
        })
        .filter(|values| !values.is_empty())
        .unwrap_or_else(|| defaults.to_vec())
}

fn env_usize(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(default)
}

fn config() -> Config {
    let backends = std::env::var("IGPERF_BACKENDS")
        .unwrap_or_else(|_| String::from("cpu,metal"))
        .split(',')
        .map(str::trim)
        .filter(|backend| matches!(*backend, "cpu" | "metal"))
        .map(str::to_owned)
        .collect::<Vec<_>>();
    let output = std::env::var_os("IGPERF_OUTPUT")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("performance-results/latest/results.json"));
    Config {
        sizes: parse_list("IGPERF_SIZES", DEFAULT_SIZES),
        fanout: env_usize("IGPERF_FANOUT", 4),
        samples: env_usize("IGPERF_SAMPLES", 5),
        warmups: env_usize("IGPERF_WARMUPS", 2),
        backends,
        output,
        dirty_body_bytes: env_usize("IGPERF_DIRTY_BODY_BYTES", 2_048),
        batch_rows: env_usize("IGPERF_BATCH_ROWS", 256),
        durable_batch_rows: env_usize("IGPERF_DURABLE_BATCH_ROWS", 8_192),
        operations: std::env::var("IGPERF_OPERATIONS")
            .unwrap_or_default()
            .split(',')
            .map(str::trim)
            .filter(|operation| !operation.is_empty())
            .map(str::to_owned)
            .collect(),
    }
}

fn build_graph(
    nodes: usize,
    fanout: usize,
    dirty_body_bytes: usize,
) -> (GraphStore, GraphIds, Duration, Duration) {
    let mut graph = GraphStore::default();
    let ids = GraphIds {
        label: required(graph.catalog_mut().intern_label("Node"), "intern label"),
        value: required(
            graph.catalog_mut().intern_property("value"),
            "intern value property",
        ),
        bucket: required(
            graph.catalog_mut().intern_property("bucket"),
            "intern bucket property",
        ),
        body: required(
            graph.catalog_mut().intern_property("body"),
            "intern body property",
        ),
        relationship: required(
            graph.catalog_mut().intern_relationship_type("R"),
            "intern relationship type",
        ),
    };
    let dirty_stride = (nodes / 100).max(1);
    let dirty = "x".repeat(dirty_body_bytes);
    let node_started = Instant::now();
    for row in 0..nodes {
        let mut properties = vec![
            (ids.value, ScalarValue::Integer((row % 1_000) as i64)),
            (ids.bucket, ScalarValue::Integer((row as i64) % BUCKETS)),
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
        );
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
            );
            edge_id += 1;
        }
    }
    (graph, ids, node_elapsed, edge_started.elapsed())
}

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
    let percentile = |fraction: f64| {
        let index = ((micros.len().saturating_sub(1)) as f64 * fraction).ceil() as usize;
        micros.get(index).copied()
    };
    let mean = (!micros.is_empty()).then(|| micros.iter().sum::<f64>() / micros.len() as f64);
    let p50 = percentile(0.50);
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
        p95: percentile(0.95),
        p99: percentile(0.99),
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

fn failure(
    backend: &str,
    nodes: usize,
    edges: usize,
    category: &'static str,
    operation: impl Into<String>,
    error: impl Into<String>,
) -> Measurement {
    Measurement {
        backend: backend.to_owned(),
        nodes,
        edges,
        category,
        operation: operation.into(),
        status: "error",
        error: Some(error.into()),
        warmups: 0,
        samples: 0,
        unit: "microseconds",
        min: None,
        p50: None,
        p95: None,
        p99: None,
        max: None,
        mean: None,
        throughput_per_second: None,
        elements_per_sample: 0,
        resident_bytes: None,
        durability: "none",
        result_rows: None,
    }
}

fn measure<T>(
    warmups: usize,
    samples: usize,
    mut operation: impl FnMut() -> Result<T>,
) -> std::result::Result<(Vec<Duration>, T), String> {
    for _ in 0..warmups {
        black_box(operation().map_err(|error| error.to_string())?);
    }
    let mut times = Vec::with_capacity(samples);
    let mut last = None;
    for _ in 0..samples {
        let started = Instant::now();
        let value = operation().map_err(|error| error.to_string())?;
        times.push(started.elapsed());
        last = Some(value);
    }
    last.map(|last| (times, last))
        .ok_or_else(|| String::from("measurement requires at least one sample"))
}

/// Measures a state-changing operation against a fresh immutable generation per attempt. Setup is
/// deliberately outside the timed interval: pinning a resident generation is benchmark plumbing,
/// while the operation is the production publication whose distribution we want. Returning the
/// final fixture lets a dependent delta (the topology write) start from the exact measured batch
/// generation without replaying or rebuilding it.
fn measure_isolated<S, T>(
    warmups: usize,
    samples: usize,
    mut setup: impl FnMut() -> Result<S>,
    mut operation: impl FnMut(&mut S) -> Result<T>,
) -> std::result::Result<(Vec<Duration>, S, T), String> {
    for _ in 0..warmups {
        let mut fixture = setup().map_err(|error| error.to_string())?;
        black_box(operation(&mut fixture).map_err(|error| error.to_string())?);
    }
    let mut times = Vec::with_capacity(samples);
    let mut last = None;
    for _ in 0..samples {
        let mut fixture = setup().map_err(|error| error.to_string())?;
        let started = Instant::now();
        let value = operation(&mut fixture).map_err(|error| error.to_string())?;
        times.push(started.elapsed());
        last = Some((fixture, value));
    }
    last.map(|(fixture, value)| (times, fixture, value))
        .ok_or_else(|| String::from("measurement requires at least one sample"))
}

fn context<'a>(
    graph: &'a GraphStore,
    backend: &'a dyn ExecutionBackend,
    statistics: &'a StatisticsSnapshot,
    bookmark: Bookmark,
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
        bookmark,
        mutation_revision: graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id: graph.node_slot_count() as u64 + 1,
        next_edge_id: graph.edge_slot_count() as u64 + 1,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities::default(),
        max_result_rows: 8_000_000,
        max_batch_rows: 65_536,
        optimizer_statistics: Some(statistics),
        backend: Some(backend),
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(3_600)),
        resolved_query_at_time_nanos: None,
    }
}

fn query_workloads() -> &'static [(&'static str, &'static str, &'static str)] {
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

fn run_query(
    graph: &GraphStore,
    backend: &dyn ExecutionBackend,
    statistics: &StatisticsSnapshot,
    query: &str,
) -> Result<ExecutionOutput> {
    QueryEngine.execute(
        query,
        &mut context(
            graph,
            backend,
            statistics,
            Bookmark {
                term: TERM,
                index: 1,
            },
        ),
    )
}

fn run_native_query(
    graph: &GraphStore,
    backend: &dyn ExecutionBackend,
    statistics: &StatisticsSnapshot,
    query: &str,
) -> Result<ExecutionOutput> {
    let mut execution_context = context(
        graph,
        backend,
        statistics,
        Bookmark {
            term: TERM,
            index: 1,
        },
    );
    execution_context.capabilities.require_native_execution = true;
    QueryEngine.execute(query, &mut execution_context)
}

fn output_rows(output: &ExecutionOutput) -> usize {
    output
        .result
        .batches
        .iter()
        .map(|batch| batch.row_count)
        .sum()
}

fn result_rows(result: &ResidentGraphProcedureResult) -> usize {
    match result {
        ResidentGraphProcedureResult::Degree { node_rows, .. }
        | ResidentGraphProcedureResult::BreadthFirst { node_rows, .. }
        | ResidentGraphProcedureResult::DepthFirst { node_rows, .. }
        | ResidentGraphProcedureResult::ShortestPath { node_rows, .. }
        | ResidentGraphProcedureResult::Dijkstra { node_rows, .. }
        | ResidentGraphProcedureResult::Components { node_rows, .. }
        | ResidentGraphProcedureResult::Louvain { node_rows, .. }
        | ResidentGraphProcedureResult::PageRank { node_rows, .. }
        | ResidentGraphProcedureResult::ClusteringCoefficient { node_rows, .. }
        | ResidentGraphProcedureResult::KCore { node_rows, .. } => node_rows.len(),
        ResidentGraphProcedureResult::TriangleCount { .. } => 1,
    }
}

fn algorithm_workloads(nodes: usize) -> Vec<(&'static str, ResidentGraphProcedure)> {
    let last = u32::try_from(nodes.saturating_sub(1)).unwrap_or(u32::MAX);
    vec![
        ("degree", ResidentGraphProcedure::Degree),
        (
            "bfs",
            ResidentGraphProcedure::BreadthFirst { source_dense: 0 },
        ),
        (
            "dfs",
            ResidentGraphProcedure::DepthFirst { source_dense: 0 },
        ),
        (
            "shortest_path",
            ResidentGraphProcedure::ShortestPath {
                source_dense: 0,
                target_dense: last,
            },
        ),
        (
            "dijkstra",
            ResidentGraphProcedure::DijkstraUnit { source_dense: 0 },
        ),
        ("wcc", ResidentGraphProcedure::WeaklyConnectedComponents),
        ("scc", ResidentGraphProcedure::StronglyConnectedComponents),
        (
            "pagerank",
            ResidentGraphProcedure::PageRank {
                damping: 0.85,
                tolerance: 1e-7,
                max_iterations: 20,
            },
        ),
        ("triangle_count", ResidentGraphProcedure::TriangleCount),
        ("clustering", ResidentGraphProcedure::ClusteringCoefficient),
        ("k_core", ResidentGraphProcedure::KCore),
        ("louvain", ResidentGraphProcedure::Louvain),
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

fn cpu_algorithm(graph: &GraphSnapshot, procedure: ResidentGraphProcedure) -> Result<usize> {
    let outgoing = &graph.outgoing;
    let incoming = &graph.incoming;
    match procedure {
        ResidentGraphProcedure::Degree => {
            let degree = (0..graph.node_ids.len())
                .map(|row| {
                    let row = row as u32;
                    (
                        outgoing.row(row).map_or(0, Iterator::count),
                        incoming.row(row).map_or(0, Iterator::count),
                    )
                })
                .collect::<Vec<_>>();
            black_box(&degree);
            Ok(degree.len())
        }
        ResidentGraphProcedure::BreadthFirst { source_dense } => {
            bfs(outgoing, source_dense).map(|rows| rows.len())
        }
        ResidentGraphProcedure::DepthFirst { source_dense } => {
            dfs(outgoing, source_dense).map(|rows| rows.len())
        }
        ResidentGraphProcedure::ShortestPath {
            source_dense,
            target_dense,
        } => shortest_path(outgoing, source_dense, target_dense)
            .map(|path| path.map_or(0, |path| path.len())),
        ResidentGraphProcedure::DijkstraUnit { source_dense } => {
            dijkstra(outgoing, source_dense, |_| Ok(1.0)).map(|rows| rows.distance.len())
        }
        ResidentGraphProcedure::WeaklyConnectedComponents => {
            weakly_connected_components(outgoing, incoming).map(|rows| rows.component.len())
        }
        ResidentGraphProcedure::StronglyConnectedComponents => {
            strongly_connected_components(outgoing, incoming).map(|rows| rows.component.len())
        }
        ResidentGraphProcedure::PageRank {
            damping,
            tolerance,
            max_iterations,
        } => page_rank(
            outgoing,
            PageRankConfig {
                damping,
                tolerance,
                max_iterations,
            },
        )
        .map(|rows| rows.len()),
        ResidentGraphProcedure::TriangleCount => triangle_count(outgoing, incoming).map(|_| 1),
        ResidentGraphProcedure::ClusteringCoefficient => {
            clustering_coefficients(outgoing, incoming).map(|rows| rows.len())
        }
        ResidentGraphProcedure::KCore => k_core(outgoing, incoming).map(|rows| rows.len()),
        ResidentGraphProcedure::Louvain => {
            louvain_communities(outgoing, incoming).map(|rows| rows.component.len())
        }
        ResidentGraphProcedure::DijkstraWeighted { .. } => Err(irongraph::Error::invalid_data(
            "weighted benchmark requires an edge property",
        )),
    }
}

fn backend(name: &str) -> Result<Box<dyn ExecutionBackend>> {
    match name {
        "cpu" => Ok(Box::new(CpuBackend::new(
            irongraph::config::UNBOUNDED_DEVICE_MEMORY_BYTES,
            0,
        ))),
        "metal" => Ok(Box::new(MetalBackend::new(
            0,
            irongraph::config::UNBOUNDED_DEVICE_MEMORY_BYTES,
            0,
        )?)),
        _ => Err(irongraph::Error::invalid_data("unknown benchmark backend")),
    }
}

fn benchmark_backend(
    config: &Config,
    graph: &GraphStore,
    nodes: usize,
    ids: GraphIds,
    backend_name: &str,
    output: &mut Vec<Measurement>,
) {
    let edges = graph.edge_count();
    let mut execution = match backend(backend_name) {
        Ok(backend) => backend,
        Err(error) => {
            output.push(failure(
                backend_name,
                nodes,
                edges,
                "admission",
                "construct_backend",
                error.to_string(),
            ));
            return;
        }
    };
    let image_started = Instant::now();
    let image = match ResidentProjectImage::build(
        PROJECT,
        Bookmark {
            term: TERM,
            index: 1,
        },
        graph,
        &TemporalStore::default(),
        &IndexCatalog::default(),
    ) {
        Ok(image) => image,
        Err(error) => {
            output.push(failure(
                backend_name,
                nodes,
                edges,
                "admission",
                "build_image",
                error.to_string(),
            ));
            return;
        }
    };
    let image_elapsed = image_started.elapsed();
    let admit_started = Instant::now();
    if let Err(error) = execution.admit_project(image) {
        output.push(failure(
            backend_name,
            nodes,
            edges,
            "admission",
            "admit_project",
            error.to_string(),
        ));
        return;
    }
    let resident = execution.resident_project_bytes(PROJECT);
    output.push(distribution(
        backend_name,
        nodes,
        edges,
        "admission",
        "build_image",
        vec![image_elapsed],
        0,
        nodes + edges,
        resident,
        "none",
        None,
    ));
    output.push(distribution(
        backend_name,
        nodes,
        edges,
        "admission",
        "admit_project",
        vec![admit_started.elapsed()],
        0,
        nodes + edges,
        resident,
        "none",
        None,
    ));

    let statistics = StatisticsSnapshot::collect_project(
        graph,
        Some(&TemporalStore::default()),
        Some(&IndexCatalog::default()),
    );
    let algorithm_graph = match graph.snapshot() {
        Ok(snapshot) => snapshot,
        Err(error) => {
            output.push(failure(
                backend_name,
                nodes,
                edges,
                "algorithm",
                "snapshot",
                error.to_string(),
            ));
            return;
        }
    };
    for &(category, name, query) in query_workloads() {
        if !config.wants(name) {
            continue;
        }
        match measure(config.warmups, config.samples, || {
            run_query(graph, execution.as_ref(), &statistics, query)
        }) {
            Ok((times, value)) => output.push(distribution(
                backend_name,
                nodes,
                edges,
                category,
                name,
                times,
                config.warmups,
                nodes + edges,
                resident,
                "none",
                Some(output_rows(&value)),
            )),
            Err(error) => output.push(failure(backend_name, nodes, edges, category, name, error)),
        }
    }

    for (name, query) in adaptive_algorithm_workloads(nodes) {
        if !config.wants(&name) {
            continue;
        }
        match measure(config.warmups, config.samples, || {
            run_query(graph, execution.as_ref(), &statistics, &query)
        }) {
            Ok((times, value)) => output.push(distribution(
                backend_name,
                nodes,
                edges,
                "adaptive_algorithm",
                &name,
                times,
                config.warmups,
                nodes + edges,
                resident,
                "none",
                Some(output_rows(&value)),
            )),
            Err(error) => output.push(failure(
                backend_name,
                nodes,
                edges,
                "adaptive_algorithm",
                &name,
                error,
            )),
        }

        let native_name = name.replacen("adaptive_", "native_", 1);
        if backend_name != "metal" || !config.wants(&native_name) {
            continue;
        }
        match measure(config.warmups, config.samples, || {
            run_native_query(graph, execution.as_ref(), &statistics, &query)
        }) {
            Ok((times, value)) => output.push(distribution(
                backend_name,
                nodes,
                edges,
                "native_algorithm",
                &native_name,
                times,
                config.warmups,
                nodes + edges,
                resident,
                "none",
                Some(output_rows(&value)),
            )),
            Err(error) => output.push(failure(
                backend_name,
                nodes,
                edges,
                "native_algorithm",
                &native_name,
                error,
            )),
        }
    }

    for (name, procedure) in algorithm_workloads(nodes) {
        if !config.wants(name) {
            continue;
        }
        let measured = if backend_name == "cpu" {
            measure(config.warmups, config.samples, || {
                cpu_algorithm(&algorithm_graph, procedure)
            })
            .map(|(times, rows)| (times, rows))
        } else {
            let request = ResidentGraphProcedureRequest {
                project: PROJECT,
                layers: irongraph::graph::LayerMask::ALL,
                procedure,
                max_output_rows: nodes,
                deadline: Some(Instant::now() + Duration::from_secs(3_600)),
            };
            measure(config.warmups, config.samples, || {
                execution.execute_graph_procedure(&request, &CancellationToken::new())
            })
            .map(|(times, result)| (times, result_rows(&result)))
        };
        match measured {
            Ok((times, rows)) => output.push(distribution(
                backend_name,
                nodes,
                edges,
                "algorithm",
                name,
                times,
                config.warmups,
                nodes + edges,
                resident,
                "none",
                Some(rows),
            )),
            Err(error) => output.push(failure(
                backend_name,
                nodes,
                edges,
                "algorithm",
                name,
                error,
            )),
        }
    }

    if !config.wants_any(&[
        "canonical_batch_insert",
        "resident_batch_publish",
        "resident_edge_insert_publish",
    ]) {
        return;
    }

    let revision = graph.revision().saturating_add(1);
    let first_id = graph.node_slot_count() as u64 + 1;
    let (canonical_times, mut changed) = match measure(config.warmups, config.samples, || {
        let mut candidate = graph.clone();
        for offset in 0..config.batch_rows {
            candidate.insert_node(NodeInput {
                id: NodeId(first_id + offset as u64),
                layer: Layer::Observed,
                revision,
                labels: vec![ids.label],
                properties: vec![(ids.value, ScalarValue::Integer(offset as i64))],
            })?;
        }
        Ok(candidate)
    }) {
        Ok(measured) => measured,
        Err(error) => {
            output.push(failure(
                backend_name,
                nodes,
                edges,
                "write",
                "canonical_batch_insert",
                error,
            ));
            return;
        }
    };
    output.push(distribution(
        backend_name,
        nodes,
        edges,
        "write",
        "canonical_batch_insert",
        canonical_times,
        config.warmups,
        config.batch_rows,
        resident,
        "memory",
        Some(config.batch_rows),
    ));
    let graph_delta = match changed.device_delta(revision) {
        Ok(delta) => delta,
        Err(error) => {
            output.push(failure(
                backend_name,
                nodes,
                edges,
                "write",
                "resident_batch_publish",
                error.to_string(),
            ));
            return;
        }
    };
    let batch_delta = ResidentProjectDelta {
        project: PROJECT,
        bookmark: Bookmark {
            term: TERM,
            index: revision,
        },
        graph: graph_delta,
        temporal: Vec::new(),
        vectors: Vec::new(),
        invalidate_derived: false,
    };
    let batch_execution = match measure_isolated(
        config.warmups,
        config.samples,
        || execution.pin_project(PROJECT),
        |candidate| candidate.apply_project_delta(batch_delta.clone()),
    ) {
        Ok((times, candidate, ())) => {
            output.push(distribution(
                backend_name,
                nodes,
                edges,
                "write",
                "resident_batch_publish",
                times,
                config.warmups,
                config.batch_rows,
                candidate.resident_project_bytes(PROJECT),
                "memory",
                Some(config.batch_rows),
            ));
            candidate
        }
        Err(error) => {
            output.push(failure(
                backend_name,
                nodes,
                edges,
                "write",
                "resident_batch_publish",
                error.to_string(),
            ));
            return;
        }
    };

    // Measure the topology-changing delta independently. Appended edgeless nodes only extend
    // empty offset rows; this edge insertion changes a non-tail source row and therefore exposes
    // any resident backend that still rematerializes adjacency proportional to the whole graph.
    let topology_revision = revision.saturating_add(1);
    if let Err(error) = changed.insert_edge(EdgeInput {
        id: EdgeId(graph.edge_slot_count() as u64 + 1),
        source: NodeId(1),
        target: NodeId(first_id),
        relationship_type: ids.relationship,
        layer: Layer::Observed,
        revision: topology_revision,
        properties: Vec::new(),
    }) {
        output.push(failure(
            backend_name,
            nodes,
            edges,
            "write",
            "resident_edge_insert_publish",
            error.to_string(),
        ));
        return;
    }
    let topology_delta = match changed.device_delta(topology_revision) {
        Ok(delta) => delta,
        Err(error) => {
            output.push(failure(
                backend_name,
                nodes,
                edges,
                "write",
                "resident_edge_insert_publish",
                error.to_string(),
            ));
            return;
        }
    };
    let topology_delta = ResidentProjectDelta {
        project: PROJECT,
        bookmark: Bookmark {
            term: TERM,
            index: topology_revision,
        },
        graph: topology_delta,
        temporal: Vec::new(),
        vectors: Vec::new(),
        invalidate_derived: false,
    };
    match measure_isolated(
        config.warmups,
        config.samples,
        || batch_execution.pin_project(PROJECT),
        |candidate| candidate.apply_project_delta(topology_delta.clone()),
    ) {
        Ok((times, candidate, ())) => output.push(distribution(
            backend_name,
            nodes,
            edges,
            "write",
            "resident_edge_insert_publish",
            times,
            config.warmups,
            1,
            candidate.resident_project_bytes(PROJECT),
            "memory",
            Some(1),
        )),
        Err(error) => output.push(failure(
            backend_name,
            nodes,
            edges,
            "write",
            "resident_edge_insert_publish",
            error.to_string(),
        )),
    }
}

fn eventual_write_acknowledgements(
    config: &Config,
    nodes: usize,
    edges: usize,
    output: &mut Vec<Measurement>,
) {
    let measured = (|| -> std::result::Result<(Vec<Duration>, Vec<Duration>), String> {
        let directory = tempfile::tempdir().map_err(|error| error.to_string())?;
        let async_runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .map_err(|error| error.to_string())?;
        async_runtime.block_on(async {
            let identity = NodeIdentity::generate_genesis().public();
            let runtime = WriteRuntime::start(
                directory.path(),
                StandaloneNode {
                    identity,
                    execution_class: ExecutionClass::Cpu,
                },
                Arc::new(AckBenchmarkBackend::default()),
                WriteStorageLimits {
                    max_log_record_bytes: 1024 * 1024,
                    max_log_entries_per_read: 8_192,
                    max_snapshot_bytes: 1024 * 1024,
                },
            )
            .await
            .map_err(|error| error.to_string())?;
            let request = || WriteRequest {
                command: WriteCommand {
                    kind: MutationKind::Graph,
                    project_id: Some(PROJECT),
                    request_id: Some(Uuid::new_v4()),
                    commit_time_millis: 0,
                    payload: vec![7_u8; 256],
                },
                timeout_millis: 5_000,
                connection_id: ConnectionId::new(),
                admission_class: AdmissionClass::Client,
                transaction_fence: None,
            };

            let mut single_samples = Vec::with_capacity(config.samples);
            for sample in 0..config.warmups + config.samples {
                let write = request();
                let started = Instant::now();
                runtime
                    .write(write)
                    .await
                    .map_err(|error| error.to_string())?;
                if sample >= config.warmups {
                    single_samples.push(started.elapsed());
                }
            }

            let mut batch_samples = Vec::with_capacity(config.samples);
            for sample in 0..config.warmups + config.samples {
                let writes = (0..config.batch_rows)
                    .map(|_| runtime.write(request()))
                    .collect::<Vec<_>>();
                let started = Instant::now();
                for result in join_all(writes).await {
                    result.map_err(|error| error.to_string())?;
                }
                if sample >= config.warmups {
                    batch_samples.push(started.elapsed());
                }
            }
            runtime
                .shutdown()
                .await
                .map_err(|error| error.to_string())?;
            Ok((single_samples, batch_samples))
        })
    })();

    match measured {
        Ok((single, batch)) => {
            output.push(distribution(
                "standalone",
                nodes,
                edges,
                "write",
                "eventual_write_ack_single",
                single,
                config.warmups,
                1,
                None,
                "eventual",
                Some(1),
            ));
            output.push(distribution(
                "standalone",
                nodes,
                edges,
                "write",
                "eventual_write_ack_concurrent",
                batch,
                config.warmups,
                config.batch_rows,
                None,
                "eventual",
                Some(config.batch_rows),
            ));
        }
        Err(error) => {
            output.push(failure(
                "standalone",
                nodes,
                edges,
                "write",
                "eventual_write_ack_single",
                error.clone(),
            ));
            output.push(failure(
                "standalone",
                nodes,
                edges,
                "write",
                "eventual_write_ack_concurrent",
                error,
            ));
        }
    }
}

fn durable_writes(config: &Config, nodes: usize, edges: usize, output: &mut Vec<Measurement>) {
    let directory = match tempfile::tempdir() {
        Ok(directory) => directory,
        Err(error) => {
            output.push(failure(
                "storage",
                nodes,
                edges,
                "write",
                "wal_fsync",
                error.to_string(),
            ));
            return;
        }
    };
    let mut log = match DurableLog::open(
        directory.path().join("benchmark.wal"),
        16 * 1024 * 1024,
        Bookmark::default(),
    ) {
        Ok(log) => log,
        Err(error) => {
            output.push(failure(
                "storage",
                nodes,
                edges,
                "write",
                "wal_fsync",
                error.to_string(),
            ));
            return;
        }
    };
    let payloads = [
        ("wal_background_append_single", 1_usize, false),
        (
            "wal_background_append_batch",
            config.durable_batch_rows,
            false,
        ),
        ("wal_fsync_single", 1_usize, true),
        ("wal_fsync_batch", config.durable_batch_rows, true),
    ];
    for (operation, rows, sync) in payloads {
        let mut samples = Vec::with_capacity(config.samples);
        let mut failed = None;
        for sample in 0..config.samples {
            let first_index = log.last_bookmark().index + 1;
            let entries = (0..rows)
                .map(|offset| {
                    MutationEntry::new(
                        TERM,
                        first_index + offset as u64,
                        MutationKind::Graph,
                        Some(PROJECT),
                        Some(Uuid::new_v4()),
                        sample as i64 + 1,
                        vec![7_u8; 256],
                    )
                })
                .collect::<Result<Vec<_>>>();
            let entries = match entries {
                Ok(entries) => entries,
                Err(error) => {
                    failed = Some(error.to_string());
                    break;
                }
            };
            let started = Instant::now();
            let append = if rows == 1 {
                match entries.into_iter().next() {
                    Some(entry) if sync => log.append(entry),
                    Some(entry) => log.append_buffered(entry),
                    None => Err(irongraph::Error::internal(
                        "single WAL benchmark produced no entry",
                    )),
                }
            } else if sync {
                log.append_batch(entries)
            } else {
                log.append_batch_buffered(entries)
            };
            if let Err(error) = append {
                failed = Some(error.to_string());
                break;
            }
            samples.push(started.elapsed());
        }
        if failed.is_none()
            && !sync
            && let Err(error) = log.sync()
        {
            failed = Some(error.to_string());
        }
        if let Some(error) = failed {
            output.push(failure("storage", nodes, edges, "write", operation, error));
        } else {
            output.push(distribution(
                "storage",
                nodes,
                edges,
                "write",
                operation,
                samples,
                0,
                rows,
                None,
                if sync { "background-fsync" } else { "eventual" },
                Some(rows),
            ));
        }
    }
}

fn command_output(program: &str, arguments: &[&str]) -> String {
    Command::new(program)
        .args(arguments)
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .unwrap_or_else(|| String::from("unknown"))
}

fn physical_memory_bytes() -> Option<u64> {
    command_output("sysctl", &["-n", "hw.memsize"])
        .parse::<u64>()
        .ok()
}

fn metal_devices() -> Vec<String> {
    command_output("system_profiler", &["SPDisplaysDataType"])
        .lines()
        .filter_map(|line| line.trim().strip_prefix("Chipset Model:"))
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(str::to_owned)
        .collect()
}

fn write_report(path: &Path, report: &Report) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let temporary = path.with_extension("json.tmp");
    let bytes = serde_json::to_vec_pretty(report)?;
    fs::write(&temporary, bytes)?;
    fs::rename(temporary, path)
}

pub fn main() {
    let config = config();
    if config.backends.is_empty() {
        eprintln!("IGPERF_BACKENDS selected no supported backend");
        std::process::exit(2);
    }
    let mut measurements = Vec::new();
    for &nodes in &config.sizes {
        if nodes > u32::MAX as usize
            || nodes
                .checked_mul(config.fanout)
                .is_none_or(|edges| edges > u32::MAX as usize)
        {
            measurements.push(failure(
                "fixture",
                nodes,
                0,
                "fixture",
                "build",
                "graph shape exceeds stable u32 dense ordinals",
            ));
            continue;
        }
        eprintln!(
            "IGPERF fixture nodes={nodes} fanout={} backends={}",
            config.fanout,
            config.backends.join(",")
        );
        let (graph, ids, node_time, edge_time) =
            build_graph(nodes, config.fanout, config.dirty_body_bytes);
        let edges = graph.edge_count();
        let canonical_resident_bytes = graph.resident_bytes();
        measurements.push(distribution(
            "cpu-canonical",
            nodes,
            edges,
            "write",
            "initial_node_ingest",
            vec![node_time],
            0,
            nodes,
            Some(canonical_resident_bytes),
            "memory",
            Some(nodes),
        ));
        measurements.push(distribution(
            "cpu-canonical",
            nodes,
            edges,
            "write",
            "initial_edge_ingest",
            vec![edge_time],
            0,
            edges,
            Some(canonical_resident_bytes),
            "memory",
            Some(edges),
        ));
        if config.wants_any(&["eventual_write_ack_single", "eventual_write_ack_concurrent"]) {
            eventual_write_acknowledgements(&config, nodes, edges, &mut measurements);
        }
        if config.wants_any(&[
            "wal_background_append_single",
            "wal_background_append_batch",
            "wal_fsync_single",
            "wal_fsync_batch",
        ]) {
            durable_writes(&config, nodes, edges, &mut measurements);
        }
        for backend in &config.backends {
            benchmark_backend(&config, &graph, nodes, ids, backend, &mut measurements);
        }
    }
    let report = Report {
        schema: "irongraph.performance-matrix.v1",
        generated_unix_millis: SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |value| value.as_millis()),
        git_revision: command_output("git", &["rev-parse", "HEAD"]),
        git_dirty: !command_output("git", &["status", "--porcelain"]).is_empty(),
        os: std::env::consts::OS,
        arch: std::env::consts::ARCH,
        cpu_model: {
            let model = command_output("sysctl", &["-n", "machdep.cpu.brand_string"]);
            if model == "unknown" {
                command_output("sysctl", &["-n", "hw.model"])
            } else {
                model
            }
        },
        physical_memory_bytes: physical_memory_bytes(),
        metal_devices: metal_devices(),
        parallelism: std::thread::available_parallelism().map_or(1, usize::from),
        config: ReportConfig {
            sizes: config.sizes.clone(),
            fanout: config.fanout,
            samples: config.samples,
            warmups: config.warmups,
            backends: config.backends.clone(),
            dirty_body_bytes: config.dirty_body_bytes,
            batch_rows: config.batch_rows,
            durable_batch_rows: config.durable_batch_rows,
            operations: config.operations.clone(),
        },
        measurements,
    };
    if let Err(error) = write_report(&config.output, &report) {
        eprintln!("write performance report: {error}");
        std::process::exit(2);
    }
    println!("performance matrix written to {}", config.output.display());
}
