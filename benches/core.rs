#![allow(clippy::all, clippy::nursery, clippy::pedantic)]

use std::{hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use irongraph::{
    Result,
    gpu::{CompareOp, CpuBackend, ExecutionBackend},
    graph::{
        Csr, PageRankConfig, bfs, clustering_coefficients, dfs, dijkstra, k_core,
        louvain_communities, page_rank, shortest_path, strongly_connected_components,
        triangle_count, weakly_connected_components,
    },
};
use tokio_util::sync::CancellationToken;

fn required<T>(result: Result<T>) -> T {
    match result {
        Ok(value) => value,
        Err(error) => {
            eprintln!("benchmark setup or operator failed: {error}");
            std::process::exit(2);
        }
    }
}

fn configured_size(name: &str, default: usize, maximum: usize) -> usize {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0 && *value <= maximum)
        .unwrap_or(default)
}

fn u32_bound(value: usize, label: &str) -> u32 {
    match u32::try_from(value) {
        Ok(value) => value,
        Err(_) => {
            eprintln!("benchmark {label} exceeds u32");
            std::process::exit(2);
        }
    }
}

fn u64_bound(value: usize, label: &str) -> u64 {
    match u64::try_from(value) {
        Ok(value) => value,
        Err(_) => {
            eprintln!("benchmark {label} exceeds u64");
            std::process::exit(2);
        }
    }
}

fn u32_from_u64(value: u64, label: &str) -> u32 {
    match u32::try_from(value) {
        Ok(value) => value,
        Err(_) => {
            eprintln!("benchmark {label} exceeds u32");
            std::process::exit(2);
        }
    }
}

fn synthetic_graph(node_count: usize, fanout: usize) -> (Vec<(u32, u32, u32)>, Csr, Csr) {
    let node_count_u32 = u32_bound(node_count, "node count");
    let fanout_u32 = u32_bound(fanout, "fanout");
    if node_count
        .checked_mul(fanout)
        .is_none_or(|edges| edges > u32::MAX as usize)
    {
        eprintln!("benchmark edge count exceeds stable edge ordinal capacity");
        std::process::exit(2);
    }
    let mut triples = Vec::with_capacity(node_count.saturating_mul(fanout));
    let mut edge = 0_u32;
    for source in 0..node_count_u32 {
        for step in 1..=fanout_u32 {
            let target = u32_from_u64(
                (u64::from(source) + u64::from(step) * 7_919) % u64::from(node_count_u32),
                "target node",
            );
            triples.push((source, target, edge));
            edge = edge.saturating_add(1);
        }
    }
    let outgoing = required(Csr::build(node_count, &triples));
    let incoming_triples = triples
        .iter()
        .map(|(source, target, edge)| (*target, *source, *edge))
        .collect::<Vec<_>>();
    let incoming = required(Csr::build(node_count, &incoming_triples));
    (triples, outgoing, incoming)
}

fn graph_and_analytics(c: &mut Criterion) {
    let node_count = configured_size("IRONGRAPH_BENCH_NODES", 50_000, 10_000_000);
    let fanout = configured_size("IRONGRAPH_BENCH_FANOUT", 8, 1_024);
    let (triples, outgoing, incoming) = synthetic_graph(node_count, fanout);
    let edge_count = triples.len();
    eprintln!(
        "benchmark_context os={} arch={} backend=cpu-reference nodes={} edges={} state=warm correctness=exact",
        std::env::consts::OS,
        std::env::consts::ARCH,
        node_count,
        edge_count,
    );

    let mut group = c.benchmark_group("graph_cpu_reference");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(u64_bound(edge_count, "edge count")));
    group.bench_function(BenchmarkId::new("csr_build", edge_count), |b| {
        b.iter(|| black_box(required(Csr::build(node_count, black_box(&triples)))))
    });
    group.bench_function(BenchmarkId::new("one_hop_expand", fanout), |b| {
        b.iter(|| {
            black_box(
                outgoing
                    .row(black_box(u32_bound(node_count / 2, "middle node")))
                    .into_iter()
                    .flatten()
                    .count(),
            )
        })
    });
    group.bench_function("bfs", |b| {
        b.iter(|| black_box(required(bfs(black_box(&outgoing), 0))))
    });
    group.bench_function("dfs", |b| {
        b.iter(|| black_box(required(dfs(black_box(&outgoing), 0))))
    });
    group.bench_function("shortest_path", |b| {
        b.iter(|| {
            black_box(required(shortest_path(
                black_box(&outgoing),
                0,
                u32_bound(node_count - 1, "last node"),
            )))
        })
    });
    group.bench_function("dijkstra", |b| {
        b.iter(|| black_box(required(dijkstra(black_box(&outgoing), 0, |_| Ok(1.0)))))
    });
    group.bench_function("weak_components", |b| {
        b.iter(|| {
            black_box(required(weakly_connected_components(
                black_box(&outgoing),
                black_box(&incoming),
            )))
        })
    });
    group.bench_function("strong_components", |b| {
        b.iter(|| {
            black_box(required(strongly_connected_components(
                black_box(&outgoing),
                black_box(&incoming),
            )))
        })
    });
    group.bench_function("pagerank_20", |b| {
        let config = PageRankConfig {
            max_iterations: 20,
            ..PageRankConfig::default()
        };
        b.iter(|| black_box(required(page_rank(black_box(&outgoing), config))))
    });
    group.bench_function("triangle_count", |b| {
        b.iter(|| {
            black_box(required(triangle_count(
                black_box(&outgoing),
                black_box(&incoming),
            )))
        })
    });
    group.bench_function("clustering_coefficient", |b| {
        b.iter(|| {
            black_box(required(clustering_coefficients(
                black_box(&outgoing),
                black_box(&incoming),
            )))
        })
    });
    group.bench_function("k_core", |b| {
        b.iter(|| black_box(required(k_core(black_box(&outgoing), black_box(&incoming)))))
    });
    group.bench_function("louvain", |b| {
        b.iter(|| {
            black_box(required(louvain_communities(
                black_box(&outgoing),
                black_box(&incoming),
            )))
        })
    });
    group.finish();
}

fn vectorized_primitives(c: &mut Criterion) {
    let rows = configured_size("IRONGRAPH_BENCH_VECTOR_ROWS", 32_768, 2_000_000);
    let dimension = configured_size("IRONGRAPH_BENCH_VECTOR_DIM", 128, 8_192);
    let mut matrix = Vec::with_capacity(rows.saturating_mul(dimension));
    for row in 0..rows {
        for coordinate in 0..dimension {
            matrix.push(((row.wrapping_mul(17) ^ coordinate.wrapping_mul(31)) & 1023) as f32);
        }
    }
    let query = (0..dimension)
        .map(|coordinate| ((coordinate * 13) & 1023) as f32)
        .collect::<Vec<_>>();
    let row_bound = required(
        i64::try_from(rows)
            .map_err(|_| irongraph::Error::invalid_data("benchmark row count exceeds i64")),
    );
    let values = (0..row_bound).collect::<Vec<_>>();
    let validity = vec![true; rows];
    let cancellation = CancellationToken::new();
    let backend = CpuBackend::new(usize::MAX / 4, 0);

    let mut group = c.benchmark_group("resident_primitives_cpu_reference");
    group.sample_size(10);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(u64_bound(rows, "vector rows")));
    group.bench_function(BenchmarkId::new("filter_i64", rows), |b| {
        b.iter(|| {
            black_box(required(backend.filter_i64(
                black_box(&values),
                black_box(&validity),
                CompareOp::GreaterOrEqual,
                row_bound / 2,
                &cancellation,
            )))
        })
    });
    group.bench_function(BenchmarkId::new("exact_l2", dimension), |b| {
        b.iter(|| {
            black_box(required(backend.exact_l2(
                black_box(&matrix),
                rows,
                dimension,
                black_box(&query),
                &cancellation,
            )))
        })
    });
    group.finish();
}

criterion_group!(benches, graph_and_analytics, vectorized_primitives);
criterion_main!(benches);
