//! Differential and fail-closed acceptance for resident graph algorithms.
//!
//! These tests deliberately exercise the public Cypher boundary. A matching answer from a CPU
//! helper is not sufficient: the fake accelerator below proves that an active accelerator is
//! routed through `execute_graph_procedure` and that backend failures cannot fall through to any
//! host adjacency method. The real-Metal test uses a matching resident bookmark so it cannot
//! accidentally exercise the CPU path.

#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::{
    collections::BTreeMap,
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use irongraph::{
    Bookmark, EdgeId, Error, ErrorCode, Layer, NodeId, ProjectId, Result, ScalarValue,
    cypher::{BindCapabilities, ExecutionContext, ExecutionOutput, QueryEngine, ResultValue},
    gpu::{
        BackendKind, CompareOp, CpuBackend, DistanceBatch, ExecutionBackend,
        ResidentGraphProcedureRequest, ResidentGraphProcedureResult, ResidentGroup,
        ResidentGroupRequest, ResidentJoinPair, ResidentJoinRequest, ResidentNodePipelineRequest,
        ResidentNodePipelineResult, ResidentProjectDelta, ResidentProjectImage,
        ResidentSortRequest, ResidentSortResult, ResidentVectorQuery, ResidentVectorResult,
        ScratchReservation,
    },
    graph::{
        EdgeInput, GraphMutation, GraphStore, IndexCatalog, LayerMask, NodeInput, TemporalStore,
    },
    types::{LabelId, PropertyId},
};
use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::gpu::{MetalBackend, ResidentGraphProcedure};
#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::graph::{Csr, PageRankConfig, bfs, page_rank};

const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());
const MEMORY_LIMIT_BYTES: usize = 256 * 1024 * 1024;
const RESERVED_BYTES: usize = 8 * 1024 * 1024;

/// Sparse fixture containing a self-loop, a parallel edge, a directed cycle, dangling and
/// disconnected nodes, and a second durable layer. Stable IDs intentionally differ from dense
/// row ordinals so the test also covers result remapping at the Cypher boundary.
fn adversarial_graph() -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    let vertex = graph.catalog_mut().intern_label("Vertex")?;
    let connected = graph.catalog_mut().intern_relationship_type("CONNECTED")?;
    let weight = graph.catalog_mut().intern_property("weight")?;

    for (id, layer) in [
        (10, Layer::Observed),
        (20, Layer::Observed),
        (30, Layer::Observed),
        (40, Layer::Observed),
        (50, Layer::Observed), // dangling
        (60, Layer::Observed), // disconnected source component
        (70, Layer::Knowledge),
        (80, Layer::Knowledge),
    ] {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer,
            revision: id,
            labels: vec![vertex],
            properties: Vec::new(),
        })?;
    }

    for (id, source, target, layer) in [
        (101, 10, 10, Layer::Observed), // self-loop
        (102, 10, 20, Layer::Observed),
        (103, 10, 20, Layer::Observed), // parallel edge
        (104, 20, 30, Layer::Observed),
        (105, 30, 40, Layer::Observed),
        (106, 40, 20, Layer::Observed), // cycle
        (107, 60, 40, Layer::Observed), // separate directed source component
        (108, 70, 80, Layer::Knowledge),
        (109, 80, 70, Layer::Knowledge),
        (110, 80, 80, Layer::Knowledge), // knowledge-only self-loop
    ] {
        graph.insert_edge(EdgeInput {
            id: EdgeId(id),
            source: NodeId(source),
            target: NodeId(target),
            relationship_type: connected,
            layer,
            revision: id,
            properties: vec![(weight, ScalarValue::Integer(1))],
        })?;
    }
    Ok(graph)
}

/// Four-node Louvain quality counterexample for lower-community-only BSP moving. The correct
/// deterministic matching partition is `{100, 102}` and `{101, 103}`; a monotone-ID shortcut
/// collapses all four nodes and loses positive modularity. Parallel and reverse relationships must
/// still collapse to the same unweighted undirected topology.
#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn louvain_higher_id_move_graph() -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    let vertex = graph.catalog_mut().intern_label("Vertex")?;
    let connected = graph.catalog_mut().intern_relationship_type("CONNECTED")?;
    for id in [100_u64, 101, 102, 103] {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: id,
            labels: vec![vertex],
            properties: Vec::new(),
        })?;
    }
    for (id, source, target) in [
        (201, 100, 101),
        (202, 101, 100), // reverse duplicate
        (203, 100, 101), // parallel duplicate
        (204, 100, 102),
        (205, 101, 103),
        (206, 103, 103), // ignored self-loop
    ] {
        graph.insert_edge(EdgeInput {
            id: EdgeId(id),
            source: NodeId(source),
            target: NodeId(target),
            relationship_type: connected,
            layer: Layer::Observed,
            revision: id,
            properties: Vec::new(),
        })?;
    }
    Ok(graph)
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn randomized_louvain_graph(seed: u64) -> Result<GraphStore> {
    let node_count = 9 + usize::try_from(seed % 12).unwrap();
    let mut graph = GraphStore::default();
    let vertex = graph.catalog_mut().intern_label("RandomLouvainVertex")?;
    let connected = graph
        .catalog_mut()
        .intern_relationship_type("RANDOM_LOUVAIN_EDGE")?;
    for node in 0..node_count {
        let id = u64::try_from(node + 1).unwrap();
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: id,
            labels: vec![vertex],
            properties: Vec::new(),
        })?;
    }

    let mut state = seed ^ 0xa409_3822_299f_31d0;
    let mut edge_id = u64::try_from(node_count).unwrap() + 1;
    for left in 0..node_count {
        for right in left..node_count {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            if state % 7 >= 3 {
                continue;
            }
            let copies = if state & 8 == 0 { 1 } else { 2 };
            for copy in 0..copies {
                let reverse = left != right && ((state >> (copy + 9)) & 1) != 0;
                let (source, target) = if reverse {
                    (right, left)
                } else {
                    (left, right)
                };
                graph.insert_edge(EdgeInput {
                    id: EdgeId(edge_id),
                    source: NodeId(u64::try_from(source + 1).unwrap()),
                    target: NodeId(u64::try_from(target + 1).unwrap()),
                    relationship_type: connected,
                    layer: Layer::Observed,
                    revision: edge_id,
                    properties: Vec::new(),
                })?;
                edge_id += 1;
            }
        }
    }
    Ok(graph)
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn randomized_components_metrics_graph(seed: u64) -> Result<GraphStore> {
    const PER_LAYER: usize = 18;
    let mut graph = GraphStore::default();
    let vertex = graph.catalog_mut().intern_label("RandomGraphVertex")?;
    let connected = graph
        .catalog_mut()
        .intern_relationship_type("RANDOM_GRAPH_EDGE")?;
    let base = 60_000_u64 + seed * 10_000;
    for row in 0..PER_LAYER * 2 {
        let id = base + u64::try_from(row).unwrap();
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: if row < PER_LAYER {
                Layer::Observed
            } else {
                Layer::Knowledge
            },
            revision: id,
            labels: vec![vertex],
            properties: Vec::new(),
        })?;
    }

    let mut edge_id = base + 1_000;
    let mut insert = |source: usize, target: usize, layer: Layer| -> Result<()> {
        graph.insert_edge(EdgeInput {
            id: EdgeId(edge_id),
            source: NodeId(base + u64::try_from(source).unwrap()),
            target: NodeId(base + u64::try_from(target).unwrap()),
            relationship_type: connected,
            layer,
            revision: edge_id,
            properties: Vec::new(),
        })?;
        edge_id += 1;
        Ok(())
    };
    let mut state = seed ^ 0x6a09_e667_f3bc_c909;
    for (offset, layer) in [(0, Layer::Observed), (PER_LAYER, Layer::Knowledge)] {
        // Last two rows stay empty. The first row is a skewed hub; reciprocal and parallel
        // duplicates deliberately collapse in undirected algorithms but remain physical degree.
        for target in 1..PER_LAYER - 2 {
            let copies: usize = if target == 1 { 256 } else { 3 };
            for copy in 0..copies {
                if copy.is_multiple_of(2) {
                    insert(offset, offset + target, layer)?;
                } else {
                    insert(offset + target, offset, layer)?;
                }
            }
        }
        for node in 0..3 {
            insert(offset + node, offset + node, layer)?;
        }
        for _ in 0..180 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let source = usize::try_from(state % u64::try_from(PER_LAYER - 2).unwrap()).unwrap();
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(0x1405_7b7e_f767_814f ^ seed);
            let target = usize::try_from(state % u64::try_from(PER_LAYER - 2).unwrap()).unwrap();
            let copies = 1 + usize::try_from((state >> 17) % 4).unwrap();
            for copy in 0..copies {
                if copy & 1 == 0 {
                    insert(offset + source, offset + target, layer)?;
                } else {
                    insert(offset + target, offset + source, layer)?;
                }
            }
        }
    }
    Ok(graph)
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn randomized_path_graph(seed: u64) -> Result<(GraphStore, [u64; 4])> {
    const NODES_PER_LAYER: usize = 8;
    let mut graph = GraphStore::default();
    let vertex = graph.catalog_mut().intern_label("RandomPathVertex")?;
    let connected = graph
        .catalog_mut()
        .intern_relationship_type("RANDOM_PATH_EDGE")?;
    let weight = graph.catalog_mut().intern_property("weight")?;
    let base = 20_000_u64 + seed * 100;
    for node in 0..NODES_PER_LAYER * 2 {
        let id = base + u64::try_from(node).unwrap();
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: if node < NODES_PER_LAYER {
                Layer::Observed
            } else {
                Layer::Knowledge
            },
            revision: id,
            labels: vec![vertex],
            properties: Vec::new(),
        })?;
    }

    let mut state = seed ^ 0x243f_6a88_85a3_08d3;
    let mut edge_id = base + 50;
    let mut insert = |source: usize, target: usize, layer: Layer, value: i64| -> Result<()> {
        graph.insert_edge(EdgeInput {
            id: EdgeId(edge_id),
            source: NodeId(base + u64::try_from(source).unwrap()),
            target: NodeId(base + u64::try_from(target).unwrap()),
            relationship_type: connected,
            layer,
            revision: edge_id,
            properties: vec![(weight, ScalarValue::Integer(value))],
        })?;
        edge_id += 1;
        Ok(())
    };

    for (offset, layer) in [(0, Layer::Observed), (NODES_PER_LAYER, Layer::Knowledge)] {
        // A guaranteed path plus a zero-cost tie, parallel relationships, cycles, and self-loops.
        for node in 0..NODES_PER_LAYER - 2 {
            insert(offset + node, offset + node + 1, layer, 1)?;
        }
        insert(offset, offset, layer, 0)?;
        insert(offset, offset + 1, layer, 1)?;
        insert(offset + 2, offset + 1, layer, 0)?;
        insert(offset + 1, offset + 3, layer, 2)?;
        insert(offset + 2, offset + 3, layer, 1)?;

        // The final node in each layer stays disconnected. Remaining edges are seeded and sparse.
        for source in 0..NODES_PER_LAYER - 1 {
            for target in 0..NODES_PER_LAYER - 1 {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                if state % 11 >= 2 {
                    continue;
                }
                let copies = 1 + usize::from((state >> 9) & 1 != 0);
                for copy in 0..copies {
                    let value = i64::try_from((state >> (copy * 3 + 13)) % 5).unwrap();
                    insert(offset + source, offset + target, layer, value)?;
                }
            }
        }
    }
    Ok((
        graph,
        [
            base,
            base + u64::try_from(NODES_PER_LAYER - 2).unwrap(),
            base + u64::try_from(NODES_PER_LAYER).unwrap(),
            base + u64::try_from(NODES_PER_LAYER * 2 - 2).unwrap(),
        ],
    ))
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn high_degree_star_graph(leaf_count: usize) -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    let vertex = graph.catalog_mut().intern_label("PowerLawVertex")?;
    let connected = graph
        .catalog_mut()
        .intern_relationship_type("POWER_LAW_EDGE")?;
    let weight = graph.catalog_mut().intern_property("weight")?;
    for node in 0..=leaf_count {
        let id = u64::try_from(node + 1).unwrap();
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: id,
            labels: vec![vertex],
            properties: Vec::new(),
        })?;
    }
    for leaf in 1..=leaf_count {
        let id = u64::try_from(leaf_count + leaf + 1).unwrap();
        graph.insert_edge(EdgeInput {
            id: EdgeId(id),
            source: NodeId(1),
            target: NodeId(u64::try_from(leaf + 1).unwrap()),
            relationship_type: connected,
            layer: Layer::Observed,
            revision: id,
            properties: vec![(weight, ScalarValue::Integer(1))],
        })?;
    }
    Ok(graph)
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn exact_path_graph() -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    let vertex = graph.catalog_mut().intern_label("Vertex")?;
    let connected = graph.catalog_mut().intern_relationship_type("CONNECTED")?;
    let integer_weight = graph.catalog_mut().intern_property("integerWeight")?;
    let float_weight = graph.catalog_mut().intern_property("floatWeight")?;
    let mixed_weight = graph.catalog_mut().intern_property("mixedWeight")?;
    let negative_weight = graph.catalog_mut().intern_property("negativeWeight")?;
    let nan_weight = graph.catalog_mut().intern_property("nanWeight")?;
    let infinite_weight = graph.catalog_mut().intern_property("infiniteWeight")?;
    let missing_weight = graph.catalog_mut().intern_property("missingWeight")?;
    let string_weight = graph.catalog_mut().intern_property("stringWeight")?;
    for id in 1..=10_u64 {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: id,
            labels: vec![vertex],
            properties: Vec::new(),
        })?;
    }
    let edges = [
        // Equal-hop paths 1-2-9-6 and 1-3-4-6. BFS's lexicographically first path goes through
        // 2 even though its final predecessor 9 is larger than the alternative predecessor 4.
        (201_u64, 1_u64, 2_u64),
        (202, 1, 3),
        (203, 2, 9),
        (204, 3, 4),
        (205, 9, 6),
        (206, 4, 6),
        (207, 6, 7),
        (208, 7, 8),
    ];
    for (position, (id, source, target)) in edges.into_iter().enumerate() {
        let integer_offset = i64::try_from(position).expect("fixture position fits i64");
        let floating_offset =
            f64::from(u32::try_from(position).expect("fixture position fits u32"));
        let mixed = if position & 1 == 0 {
            ScalarValue::Integer((1_i64 << 53) + 1 + integer_offset)
        } else {
            ScalarValue::Float(ordered_float::OrderedFloat(
                floating_offset.mul_add(0.5, 0.25),
            ))
        };
        let mut properties = vec![
            (
                integer_weight,
                ScalarValue::Integer((1_i64 << 53) + 1 + integer_offset),
            ),
            (
                float_weight,
                ScalarValue::Float(ordered_float::OrderedFloat(match position {
                    0 => f64::from_bits(1),
                    1 => f64::MIN_POSITIVE,
                    2 => -0.0,
                    3 => 2.0_f64.powi(-53),
                    6 => f64::MAX,
                    _ => 0.5 + floating_offset,
                })),
            ),
            (mixed_weight, mixed),
            (
                negative_weight,
                ScalarValue::Float(ordered_float::OrderedFloat(if position == 0 {
                    -1.0
                } else {
                    1.0
                })),
            ),
            (
                nan_weight,
                ScalarValue::Float(ordered_float::OrderedFloat(if position == 0 {
                    f64::NAN
                } else {
                    1.0
                })),
            ),
            (
                infinite_weight,
                ScalarValue::Float(ordered_float::OrderedFloat(if position == 0 {
                    f64::INFINITY
                } else {
                    1.0
                })),
            ),
            (string_weight, ScalarValue::String("not numeric".into())),
        ];
        // The property exists in the catalog and on other edges, but is absent on the first
        // reachable edge. Both CPU and Metal must reject it when that edge is examined.
        if position != 0 {
            properties.push((missing_weight, ScalarValue::Integer(1)));
        }
        graph.insert_edge(EdgeInput {
            id: EdgeId(id),
            source: NodeId(source),
            target: NodeId(target),
            relationship_type: connected,
            layer: Layer::Observed,
            revision: id,
            properties,
        })?;
    }
    Ok(graph)
}

fn image(graph: &GraphStore) -> Result<ResidentProjectImage> {
    ResidentProjectImage::build(
        PROJECT,
        Bookmark {
            term: 0,
            index: graph.revision(),
        },
        graph,
        &TemporalStore::default(),
        &IndexCatalog::default(),
    )
}

fn context<'a>(
    graph: &'a GraphStore,
    backend: Option<&'a dyn ExecutionBackend>,
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
        bookmark: Bookmark {
            // `ResidentProjectImage::build` above uses this same bookmark. Keeping it exact is
            // essential: a mismatch intentionally disables the resident backend.
            term: 0,
            index: graph.revision(),
        },
        mutation_revision: graph.revision().saturating_add(1),
        resolved_time_nanos: 0,
        next_node_id: 1_000,
        next_edge_id: 2_000,
        predicate_versions: BTreeMap::new(),
        capabilities: BindCapabilities {
            // A KNOWLEDGE-only read must name KNOWLEDGE as its syntactic write layer because the
            // parser requires the write layer to be visible, even though these CALLs are read-only.
            knowledge_write: true,
            require_native_execution: backend
                .is_some_and(|backend| backend.kind() != BackendKind::Cpu),
            ..BindCapabilities::default()
        },
        max_result_rows: 10_000,
        max_batch_rows: 4_096,
        optimizer_statistics: None,
        backend,
        cancellation: CancellationToken::new(),
        deadline: Some(Instant::now() + Duration::from_secs(20)),
        resolved_query_at_time_nanos: None,
    }
}

fn rows(output: &ExecutionOutput) -> Vec<Vec<ResultValue>> {
    output
        .result
        .batches
        .iter()
        .flat_map(|batch| {
            (0..batch.row_count).map(|row| {
                batch
                    .columns
                    .iter()
                    .map(|column| column.values[row].clone())
                    .collect()
            })
        })
        .collect()
}

fn integer(value: &ResultValue) -> i64 {
    match value {
        ResultValue::Scalar(ScalarValue::Integer(value)) => *value,
        other => panic!("expected INTEGER, received {other:?}"),
    }
}

fn float(value: &ResultValue) -> f64 {
    match value {
        ResultValue::Scalar(ScalarValue::Float(value)) => value.into_inner(),
        other => panic!("expected FLOAT, received {other:?}"),
    }
}

fn execute(
    graph: &GraphStore,
    backend: Option<&dyn ExecutionBackend>,
    query: &str,
) -> Result<ExecutionOutput> {
    let mut execution_context = context(graph, backend);
    QueryEngine.execute(query, &mut execution_context)
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn execute_with_row_budget(
    graph: &GraphStore,
    backend: Option<&dyn ExecutionBackend>,
    query: &str,
    max_result_rows: usize,
) -> Result<ExecutionOutput> {
    let mut execution_context = context(graph, backend);
    execution_context.max_result_rows = max_result_rows;
    QueryEngine.execute(query, &mut execution_context)
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn assert_exact_differential(
    graph: &GraphStore,
    accelerator: &dyn ExecutionBackend,
    query: &str,
) -> Result<()> {
    let cpu = execute(graph, None, query)?;
    let device = execute(graph, Some(accelerator), query)?;
    assert_eq!(device.result.schema, cpu.result.schema, "schema: {query}");
    assert_eq!(device.result.batches, cpu.result.batches, "rows: {query}");
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn assert_pagerank_differential(
    graph: &GraphStore,
    accelerator: &dyn ExecutionBackend,
    query: &str,
) -> Result<Vec<Vec<ResultValue>>> {
    let cpu = execute(graph, None, query)?;
    let device = execute(graph, Some(accelerator), query)?;
    let cpu_rows = rows(&cpu);
    let device_rows = rows(&device);
    assert_eq!(device.result.schema, cpu.result.schema, "schema: {query}");
    assert_eq!(device_rows.len(), cpu_rows.len(), "row count: {query}");
    for (device, cpu) in device_rows.iter().zip(&cpu_rows) {
        assert_eq!(
            integer(&device[0]),
            integer(&cpu[0]),
            "node identity: {query}"
        );
        let device_score = float(&device[1]);
        let cpu_score = float(&cpu[1]);
        assert_eq!(
            device_score.to_bits(),
            cpu_score.to_bits(),
            "PageRank binary64 differs: device={device_score:.17e} ({:#018x}), cpu={cpu_score:.17e} ({:#018x}), query={query}",
            device_score.to_bits(),
            cpu_score.to_bits(),
        );
    }
    Ok(device_rows)
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn assert_clustering_differential(
    graph: &GraphStore,
    accelerator: &dyn ExecutionBackend,
    query: &str,
) -> Result<()> {
    let cpu = execute(graph, None, query)?;
    let device = execute(graph, Some(accelerator), query)?;
    let cpu_rows = rows(&cpu);
    let device_rows = rows(&device);
    assert_eq!(device.result.schema, cpu.result.schema, "schema: {query}");
    assert_eq!(device_rows.len(), cpu_rows.len(), "row count: {query}");
    for (device, cpu) in device_rows.iter().zip(&cpu_rows) {
        assert_eq!(integer(&device[0]), integer(&cpu[0]), "node: {query}");
        assert_eq!(
            float(&device[1]).to_bits(),
            float(&cpu[1]).to_bits(),
            "clustering coefficient binary64 differs: device={device:?}, cpu={cpu:?}"
        );
    }
    Ok(())
}

#[test]
#[allow(clippy::too_many_lines)]
fn cpu_sparse_algorithm_contract_covers_parallel_self_loop_disconnected_and_layers() -> Result<()> {
    let graph = adversarial_graph()?;

    let degree = execute(
        &graph,
        None,
        "USE LAYER OBSERVED\n\
         CALL graph.degree() YIELD node, outDegree, inDegree, degree\n\
         RETURN id(node) AS id, outDegree, inDegree, degree ORDER BY id",
    )?;
    let degree = rows(&degree);
    assert_eq!(
        degree
            .iter()
            .map(|row| row.iter().map(integer).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        vec![
            vec![10, 3, 1, 4],
            vec![20, 1, 3, 4],
            vec![30, 1, 1, 2],
            vec![40, 1, 2, 3],
            vec![50, 0, 0, 0],
            vec![60, 1, 0, 1],
        ]
    );

    let bfs = execute(
        &graph,
        None,
        "USE LAYER OBSERVED\n\
         CALL graph.bfs(10) YIELD node, distance\n\
         RETURN id(node) AS id, distance ORDER BY id",
    )?;
    assert_eq!(
        rows(&bfs)
            .iter()
            .map(|row| row.iter().map(integer).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        vec![vec![10, 0], vec![20, 1], vec![30, 2], vec![40, 3]]
    );

    let knowledge = execute(
        &graph,
        None,
        "USE LAYER KNOWLEDGE\n\
         WRITE LAYER KNOWLEDGE\n\
         CALL graph.bfs(70) YIELD node, distance\n\
         RETURN id(node) AS id, distance ORDER BY id",
    )?;
    assert_eq!(
        rows(&knowledge)
            .iter()
            .map(|row| row.iter().map(integer).collect::<Vec<_>>())
            .collect::<Vec<_>>(),
        vec![vec![70, 0], vec![80, 1]]
    );
    for query in [
        "USE LAYER KNOWLEDGE\n\
         WRITE LAYER KNOWLEDGE\n\
         CALL graph.dfs(70) YIELD node, order RETURN id(node) AS id, order ORDER BY order",
        "USE LAYER KNOWLEDGE\n\
         WRITE LAYER KNOWLEDGE\n\
         CALL graph.shortestpath(70, 80) YIELD path, cost RETURN path, cost",
        "USE LAYER KNOWLEDGE\n\
         WRITE LAYER KNOWLEDGE\n\
         CALL graph.dijkstra(70, 'weight') YIELD node, cost, predecessor\n\
         RETURN id(node) AS id, cost, id(predecessor) AS predecessor ORDER BY id",
        "USE LAYER KNOWLEDGE\n\
         WRITE LAYER KNOWLEDGE\n\
         CALL graph.wcc() YIELD node, component RETURN id(node) AS id, component ORDER BY id",
        "USE LAYER KNOWLEDGE\n\
         WRITE LAYER KNOWLEDGE\n\
         CALL graph.scc() YIELD node, component RETURN id(node) AS id, component ORDER BY id",
        "USE LAYER KNOWLEDGE\n\
         WRITE LAYER KNOWLEDGE\n\
         CALL graph.louvain() YIELD node, community RETURN id(node) AS id, community ORDER BY id",
        "USE LAYER KNOWLEDGE\n\
         WRITE LAYER KNOWLEDGE\n\
         CALL graph.trianglecount() YIELD triangleCount RETURN triangleCount",
        "USE LAYER KNOWLEDGE\n\
         WRITE LAYER KNOWLEDGE\n\
         CALL graph.kcore() YIELD node, core RETURN id(node) AS id, core ORDER BY id",
    ] {
        execute(&graph, None, query)?;
    }

    let dijkstra = execute(
        &graph,
        None,
        "USE LAYER OBSERVED\n\
         CALL graph.dijkstra(10) YIELD node, cost, predecessor\n\
         RETURN id(node) AS id, cost, id(predecessor) AS predecessor ORDER BY id",
    )?;
    assert_eq!(
        rows(&dijkstra),
        vec![
            vec![
                ResultValue::Scalar(ScalarValue::Integer(10)),
                ResultValue::Scalar(ScalarValue::Float(ordered_float::OrderedFloat(0.0))),
                ResultValue::Scalar(ScalarValue::Null),
            ],
            vec![
                ResultValue::Scalar(ScalarValue::Integer(20)),
                ResultValue::Scalar(ScalarValue::Float(ordered_float::OrderedFloat(1.0))),
                ResultValue::Scalar(ScalarValue::Integer(10)),
            ],
            vec![
                ResultValue::Scalar(ScalarValue::Integer(30)),
                ResultValue::Scalar(ScalarValue::Float(ordered_float::OrderedFloat(2.0))),
                ResultValue::Scalar(ScalarValue::Integer(20)),
            ],
            vec![
                ResultValue::Scalar(ScalarValue::Integer(40)),
                ResultValue::Scalar(ScalarValue::Float(ordered_float::OrderedFloat(3.0))),
                ResultValue::Scalar(ScalarValue::Integer(30)),
            ],
        ]
    );

    let pagerank_query = "USE LAYER OBSERVED\n\
        CALL graph.pagerank(0.85, 0.0000001, 100) YIELD node, score\n\
        RETURN id(node) AS id, score ORDER BY id";
    let first = rows(&execute(&graph, None, pagerank_query)?);
    let second = rows(&execute(&graph, None, pagerank_query)?);
    assert_eq!(first, second, "CPU reference must be deterministic");
    let rank_sum = first.iter().map(|row| float(&row[1])).sum::<f64>();
    assert!((rank_sum - 1.0).abs() <= 1.0e-10, "PageRank sum={rank_sum}");
    Ok(())
}

#[test]
fn graph_algorithm_cancellation_deadline_and_result_budgets_fail_before_output() -> Result<()> {
    let graph = adversarial_graph()?;

    let mut cancelled = context(&graph, None);
    cancelled.cancellation.cancel();
    let error = QueryEngine
        .execute(
            "CALL graph.pagerank() YIELD node, score RETURN node, score",
            &mut cancelled,
        )
        .expect_err("pre-cancelled graph algorithm must fail");
    assert_eq!(error.code, ErrorCode::Cancelled);

    let mut expired = context(&graph, None);
    expired.deadline = Some(
        Instant::now()
            .checked_sub(Duration::from_millis(1))
            .ok_or_else(|| Error::internal("test deadline underflow"))?,
    );
    let error = QueryEngine
        .execute(
            "CALL graph.bfs(10) YIELD node, distance RETURN node, distance",
            &mut expired,
        )
        .expect_err("expired graph algorithm must fail");
    assert_eq!(error.code, ErrorCode::DeadlineExceeded);

    for query in [
        "CALL graph.degree() YIELD node RETURN node",
        "CALL graph.bfs(10) YIELD node RETURN node",
        "CALL graph.dfs(10) YIELD node RETURN node",
        "CALL graph.dijkstra(10) YIELD node RETURN node",
        "CALL graph.dijkstra(10, 'weight') YIELD node RETURN node",
        "CALL graph.wcc() YIELD node RETURN node",
        "CALL graph.scc() YIELD node RETURN node",
        "CALL graph.louvain() YIELD node RETURN node",
        "CALL graph.pagerank() YIELD node RETURN node",
        "CALL graph.clusteringcoefficient() YIELD node RETURN node",
        "CALL graph.kcore() YIELD node RETURN node",
    ] {
        let mut bounded = context(&graph, None);
        bounded.max_result_rows = 3;
        let error = QueryEngine
            .execute(query, &mut bounded)
            .expect_err("graph procedure must honor the output row budget");
        assert_eq!(error.code, ErrorCode::ResultBudgetExceeded, "{query}");
    }
    for query in [
        "CALL graph.shortestpath(10, 40) YIELD path RETURN path",
        "CALL graph.trianglecount() YIELD triangleCount RETURN triangleCount",
    ] {
        let mut bounded = context(&graph, None);
        bounded.max_result_rows = 0;
        let error = QueryEngine
            .execute(query, &mut bounded)
            .expect_err("single-row graph procedure must honor a zero output budget");
        assert_eq!(error.code, ErrorCode::ResultBudgetExceeded, "{query}");
    }
    Ok(())
}

/// Accelerator-kind adapter over a CPU resident image. It exposes only lifecycle metadata and
/// intentionally rejects every query operator. This lets a platform-independent test distinguish
/// fail-closed accelerator dispatch from accidental execution of the host graph algorithms.
struct RejectingAccelerator {
    inner: Box<dyn ExecutionBackend>,
    graph_procedure_calls: Arc<AtomicUsize>,
    forbidden_host_calls: Arc<AtomicUsize>,
    overlay_calls: Arc<AtomicUsize>,
}

impl RejectingAccelerator {
    fn new(inner: CpuBackend) -> Self {
        Self {
            inner: Box::new(inner),
            graph_procedure_calls: Arc::new(AtomicUsize::new(0)),
            forbidden_host_calls: Arc::new(AtomicUsize::new(0)),
            overlay_calls: Arc::new(AtomicUsize::new(0)),
        }
    }

    fn reject_host<T>(&self, route: &str) -> Result<T> {
        self.forbidden_host_calls.fetch_add(1, Ordering::SeqCst);
        Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            format!("forbidden host graph route {route}"),
        ))
    }
}

impl ExecutionBackend for RejectingAccelerator {
    fn kind(&self) -> BackendKind {
        BackendKind::Metal
    }

    fn available_query_scratch_bytes(&self) -> usize {
        self.inner.available_query_scratch_bytes()
    }

    fn reserve_query_scratch(&self, bytes: usize) -> Result<ScratchReservation> {
        self.inner.reserve_query_scratch(bytes)
    }

    fn resident_project_bytes(&self, project: ProjectId) -> Option<usize> {
        self.inner.resident_project_bytes(project)
    }

    fn pin_project(&self, project: ProjectId) -> Result<Box<dyn ExecutionBackend>> {
        Ok(Box::new(Self {
            inner: self.inner.pin_project(project)?,
            graph_procedure_calls: Arc::clone(&self.graph_procedure_calls),
            forbidden_host_calls: Arc::clone(&self.forbidden_host_calls),
            overlay_calls: Arc::clone(&self.overlay_calls),
        }))
    }

    fn admit_project(&mut self, image: ResidentProjectImage) -> Result<()> {
        self.inner.admit_project(image)
    }

    fn replace_all_projects(&mut self, images: Vec<ResidentProjectImage>) -> Result<()> {
        self.inner.replace_all_projects(images)
    }

    fn apply_project_overlay(&mut self, delta: ResidentProjectDelta) -> Result<()> {
        self.overlay_calls.fetch_add(1, Ordering::SeqCst);
        self.inner.apply_project_overlay(delta)
    }

    fn evict_project(&mut self, project: ProjectId) -> Result<()> {
        self.inner.evict_project(project)
    }

    fn advance_bookmark(&mut self, bookmark: Bookmark) {
        self.inner.advance_bookmark(bookmark);
    }

    fn resident_revision(&self) -> Option<u64> {
        self.inner.resident_revision()
    }

    fn resident_graph_revision(&self, project: ProjectId) -> Option<u64> {
        self.inner.resident_graph_revision(project)
    }

    fn resident_bookmark(&self, project: ProjectId) -> Option<Bookmark> {
        self.inner.resident_bookmark(project)
    }

    fn scan_nodes(
        &self,
        _project: ProjectId,
        _label: Option<LabelId>,
        _layers: LayerMask,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.reject_host("scan_nodes")
    }

    fn filter_node_i64(
        &self,
        _project: ProjectId,
        _property: PropertyId,
        _operation: CompareOp,
        _operand: i64,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.reject_host("filter_node_i64")
    }

    fn expand_project_out(
        &self,
        _project: ProjectId,
        _sources: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.reject_host("expand_project_out")
    }

    fn expand_project_in(
        &self,
        _project: ProjectId,
        _targets: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.reject_host("expand_project_in")
    }

    fn search_vectors(
        &self,
        _request: &ResidentVectorQuery,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentVectorResult> {
        self.reject_host("search_vectors")
    }

    fn sort_rows(
        &self,
        _request: &ResidentSortRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentSortResult> {
        self.reject_host("sort_rows")
    }

    fn join_node_i64(
        &self,
        _request: &ResidentJoinRequest,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentJoinPair>> {
        self.reject_host("join_node_i64")
    }

    fn group_node_i64(
        &self,
        _request: &ResidentGroupRequest,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<ResidentGroup>> {
        self.reject_host("group_node_i64")
    }

    fn execute_node_pipeline(
        &self,
        _request: &ResidentNodePipelineRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentNodePipelineResult> {
        self.reject_host("execute_node_pipeline")
    }

    fn execute_graph_procedure(
        &self,
        _request: &ResidentGraphProcedureRequest,
        _cancellation: &CancellationToken,
    ) -> Result<ResidentGraphProcedureResult> {
        self.graph_procedure_calls.fetch_add(1, Ordering::SeqCst);
        Err(Error::new(
            ErrorCode::GpuAdmissionFailure,
            "resident graph procedure deliberately rejected by test accelerator",
        ))
    }

    fn filter_i64(
        &self,
        _values: &[i64],
        _validity: &[bool],
        _operation: CompareOp,
        _operand: i64,
        _cancellation: &CancellationToken,
    ) -> Result<Vec<u32>> {
        self.reject_host("filter_i64")
    }

    fn expand_out(
        &self,
        _sources: &[u32],
        _cancellation: &CancellationToken,
    ) -> Result<Vec<(u32, u32, u32)>> {
        self.reject_host("expand_out")
    }

    fn exact_l2(
        &self,
        _matrix: &[f32],
        _rows: usize,
        _dimension: usize,
        _query: &[f32],
        _cancellation: &CancellationToken,
    ) -> Result<DistanceBatch> {
        self.reject_host("exact_l2")
    }
}

fn direct_scan_channels_graph() -> Result<GraphStore> {
    let mut graph = GraphStore::default();
    let channel = graph.catalog_mut().intern_label("Channel")?;
    let document = graph.catalog_mut().intern_label("Document")?;
    let selected = graph.catalog_mut().intern_label("Selected")?;
    let body = graph.catalog_mut().intern_property("body")?;
    let fields = [
        "channel_id",
        "external_id",
        "source_id",
        "name",
        "stream_id",
        "extract_policy",
        "item_count",
        "last_at",
        "purpose",
        "kind",
    ]
    .map(|name| graph.catalog_mut().intern_property(name))
    .into_iter()
    .collect::<Result<Vec<_>>>()?;
    for id in 1..=256 {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Workspace,
            revision: id,
            labels: vec![document, selected],
            properties: vec![(
                body,
                ScalarValue::String(Arc::from(format!("{id}:{}", "body ".repeat(2048)))),
            )],
        })?;
    }
    for id in 257..=263 {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: if id == 263 {
                Layer::Observed
            } else {
                Layer::Workspace
            },
            revision: id,
            labels: if id == 258 || id == 260 {
                vec![channel, selected]
            } else {
                vec![channel]
            },
            properties: fields
                .iter()
                .enumerate()
                .filter(|(index, _)| *index != 7)
                .map(|(index, property)| {
                    (
                        *property,
                        if index == 6 {
                            ScalarValue::Integer(id as i64)
                        } else {
                            ScalarValue::String(Arc::from(format!("{id}-{index}")))
                        },
                    )
                })
                .collect(),
        })?;
    }
    graph.delete_node(NodeId(262), false, 264)?;
    let link = graph.catalog_mut().intern_relationship_type("LINK")?;
    for id in 1..=255 {
        graph.insert_edge(EdgeInput {
            id: EdgeId(id),
            source: NodeId(id),
            target: NodeId(id + 1),
            relationship_type: link,
            layer: Layer::Workspace,
            revision: 264 + id,
            properties: Vec::new(),
        })?;
    }
    Ok(graph)
}

const CHANNELS_QUERY: &str = "USE LAYER WORKSPACE MATCH (c:Channel) RETURN \
    c.channel_id AS channel_id, c.external_id AS external_id, c.source_id AS source_id, \
    c.name AS name, c.stream_id AS stream_id, c.extract_policy AS extract_policy, \
    c.item_count AS item_count, c.last_at AS last_at, c.purpose AS purpose, c.kind AS kind";

fn assert_direct_scan_channels(graph: &GraphStore, backend: &dyn ExecutionBackend) -> Result<()> {
    let expected = execute(graph, None, CHANNELS_QUERY)?;
    let expected_rows = rows(&expected);
    assert_eq!(expected_rows.len(), 5);
    assert_eq!(
        expected_rows[0][0],
        ResultValue::Scalar(ScalarValue::String(Arc::from("257-0")))
    );
    assert_eq!(
        expected_rows[0][6],
        ResultValue::Scalar(ScalarValue::Integer(257))
    );
    assert_eq!(expected_rows[0][7], ResultValue::Scalar(ScalarValue::Null));
    for _ in 0..2 {
        // Repeat exact query texts to cover cached as well as newly prepared route decisions.
        for (suffix, start, end) in [
            ("", 0, 5),
            (" LIMIT 9223372036854775807", 0, 5),
            (" LIMIT 0", 0, 0),
            (" LIMIT 2", 0, 2),
            (" SKIP 2", 2, 5),
            (" SKIP 2 LIMIT 2", 2, 4),
            (" SKIP 10 LIMIT 2", 5, 5),
        ] {
            let mut execution_context = context(graph, Some(backend));
            execution_context.capabilities.require_native_execution = false;
            let actual = QueryEngine
                .execute(&format!("{CHANNELS_QUERY}{suffix}"), &mut execution_context)?;
            assert_eq!(rows(&actual), expected_rows[start..end], "{suffix}");
            assert_eq!(actual.result.bookmark, expected.result.bookmark);
            assert!(!actual.result.truncated);
            assert!(actual.graph_mutations.is_empty());
            if start != end {
                assert_eq!(actual.result.schema, expected.result.schema);
            }
        }
    }
    for limit in [0, 2, 10] {
        let mut execution_context = context(graph, Some(backend));
        execution_context.capabilities.require_native_execution = false;
        execution_context.parameters.insert(
            "limit".to_owned(),
            ResultValue::Scalar(ScalarValue::Integer(limit)),
        );
        let actual = QueryEngine.execute(
            &format!("{CHANNELS_QUERY} LIMIT $limit"),
            &mut execution_context,
        )?;
        assert_eq!(rows(&actual), expected_rows[..(limit as usize).min(5)]);
    }
    for (suffix, ids) in [
        ("", vec![258, 260]),
        (" LIMIT 1", vec![258]),
        (" SKIP 1 LIMIT 1", vec![260]),
    ] {
        let mut execution_context = context(graph, Some(backend));
        execution_context.capabilities.require_native_execution = false;
        let actual = QueryEngine.execute(
            &format!("USE LAYER WORKSPACE MATCH (c:Channel:Selected) RETURN id(c) AS id{suffix}"),
            &mut execution_context,
        )?;
        assert_eq!(
            rows(&actual)
                .iter()
                .map(|row| integer(&row[0]))
                .collect::<Vec<_>>(),
            ids
        );
    }
    for suffix in ["", " LIMIT 9223372036854775807"] {
        let mut execution_context = context(graph, Some(backend));
        execution_context.capabilities.require_native_execution = false;
        execution_context.max_result_rows = 4;
        let error = QueryEngine
            .execute(&format!("{CHANNELS_QUERY}{suffix}"), &mut execution_context)
            .expect_err("an unlimited direct scan must preserve the result budget");
        assert_eq!(error.code, ErrorCode::ResultBudgetExceeded);
    }
    let prior = [GraphMutation::InsertNode(NodeInput {
        id: NodeId(1000),
        layer: Layer::Workspace,
        revision: graph.revision() + 1,
        labels: vec![graph.catalog().label("Channel").expect("fixture label")],
        properties: Vec::new(),
    })];
    for suffix in ["", " LIMIT 9223372036854775807"] {
        let mut execution_context = context(graph, Some(backend));
        execution_context.capabilities.require_native_execution = false;
        execution_context.prior_graph_mutations = &prior;
        let actual =
            QueryEngine.execute(&format!("{CHANNELS_QUERY}{suffix}"), &mut execution_context)?;
        let actual_rows = rows(&actual);
        assert_eq!(&actual_rows[..5], expected_rows.as_slice());
        assert_eq!(
            actual_rows[5],
            vec![ResultValue::Scalar(ScalarValue::Null); 10]
        );
        assert_eq!(actual_rows.len(), 6);
    }
    let mut cancelled = context(graph, Some(backend));
    cancelled.capabilities.require_native_execution = false;
    cancelled.cancellation.cancel();
    assert_eq!(
        QueryEngine
            .execute(CHANNELS_QUERY, &mut cancelled)
            .expect_err("direct scan must honor cancellation")
            .code,
        ErrorCode::Cancelled
    );
    Ok(())
}

#[test]
fn direct_node_scan_channels_need_no_dummy_limit_on_an_accelerator() -> Result<()> {
    let graph = direct_scan_channels_graph()?;
    let mut cpu = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
    cpu.admit_project(image(&graph)?)?;
    let accelerator = RejectingAccelerator::new(cpu);
    assert_direct_scan_channels(&graph, &accelerator)?;
    assert_eq!(accelerator.forbidden_host_calls.load(Ordering::SeqCst), 0);
    assert_eq!(accelerator.graph_procedure_calls.load(Ordering::SeqCst), 0);
    // This adapter deliberately cannot run a native query. Strict conformance must still fail.
    for query in [CHANNELS_QUERY.to_owned(), format!(" {CHANNELS_QUERY} ")] {
        // Exercise both exact-source and normalized cache lookups after ordinary execution.
        let error = execute(&graph, Some(&accelerator), &query)
            .expect_err("native conformance must not reuse a cached direct scan route");
        assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_direct_node_scan_channels_need_no_dummy_limit() -> Result<()> {
    let graph = direct_scan_channels_graph()?;
    let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
    metal.admit_project(image(&graph)?)?;
    assert_direct_scan_channels(&graph, &metal)
}

fn direct_filter_contacts_graph() -> Result<GraphStore> {
    let mut graph = direct_scan_channels_graph()?;
    let contact = graph.catalog_mut().intern_label("Contact")?;
    let card_id = graph.catalog_mut().intern_property("card_id")?;
    let display = graph.catalog_mut().intern_property("display")?;
    let value = graph.catalog_mut().intern_property("value")?;
    let kind = graph.catalog_mut().intern_property("kind")?;
    for index in 0..444 {
        let mut properties = vec![
            (
                display,
                ScalarValue::String(Arc::from(format!("Person {index}"))),
            ),
            (
                value,
                ScalarValue::String(Arc::from(format!("{index}@example.test"))),
            ),
            (kind, ScalarValue::String(Arc::from("email"))),
        ];
        if index >= 78 {
            properties.push((
                card_id,
                ScalarValue::String(Arc::from(format!("card-{index}"))),
            ));
        } else if index < 26 {
            properties.push((card_id, ScalarValue::String(Arc::from(""))));
        } else if index < 52 {
            properties.push((card_id, ScalarValue::Null));
        }
        graph.insert_node(NodeInput {
            id: NodeId(1000 + index),
            layer: Layer::Observed,
            revision: graph.revision() + 1,
            labels: vec![contact],
            properties,
        })?;
    }
    Ok(graph)
}

const CONTACT_RETURN: &str =
    "RETURN c.card_id AS card_id, c.display AS display, c.value AS value, c.kind AS kind";

fn assert_direct_filter_contacts(graph: &GraphStore, backend: &dyn ExecutionBackend) -> Result<()> {
    let all = execute(graph, None, &format!("MATCH (c:Contact) {CONTACT_RETURN}"))?;
    assert_eq!(rows(&all).len(), 444);
    let expected = rows(&all)[78..].to_vec();
    assert_eq!(expected.len(), 366);
    for _ in 0..2 {
        for predicate in [
            "c.card_id <> ''",
            "c.card_id <> $empty",
            "c.card_id IS NOT NULL AND c.card_id <> ''",
            "NOT (c.card_id = '')",
            "coalesce(c.card_id, '') <> ''",
            "c.card_id > ''",
            "c.card_id STARTS WITH 'card-'",
        ] {
            for (suffix, start, end) in [
                ("", 0, 366),
                (" LIMIT 9223372036854775807", 0, 366),
                (" LIMIT 2", 0, 2),
                (" SKIP 2 LIMIT 3", 2, 5),
                (" LIMIT 0", 0, 0),
            ] {
                let mut ctx = context(graph, Some(backend));
                ctx.capabilities.require_native_execution = false;
                ctx.parameters.insert(
                    "empty".to_owned(),
                    ResultValue::Scalar(ScalarValue::String(Arc::from(""))),
                );
                // Rejected contacts and skipped rows must not consume the final output budget.
                ctx.max_result_rows = (end - start).max(1);
                let query = format!("MATCH (c:Contact) WHERE {predicate} {CONTACT_RETURN}{suffix}");
                let actual = QueryEngine.execute(&query, &mut ctx)?;
                assert_eq!(rows(&actual), expected[start..end], "{query}");
                assert_eq!(actual.result.bookmark, all.result.bookmark);
                assert!(!actual.result.truncated);
                assert!(actual.graph_mutations.is_empty());
            }
        }
    }
    for (query, start, end) in [
        (
            format!(
                "MATCH (c:Contact) WITH c LIMIT 80 WITH c WHERE c.card_id <> '' {CONTACT_RETURN}"
            ),
            0,
            2,
        ),
        (
            format!(
                "MATCH (c:Contact) WHERE c.card_id <> '' WITH c LIMIT 5 WITH c WHERE c.card_id <> 'card-78' {CONTACT_RETURN}"
            ),
            1,
            5,
        ),
        (
            format!("MATCH (c:Contact) WHERE NULL {CONTACT_RETURN}"),
            0,
            0,
        ),
        (
            format!("MATCH (c:Contact {{card_id: 'card-78'}}) {CONTACT_RETURN}"),
            0,
            1,
        ),
    ] {
        let mut ctx = context(graph, Some(backend));
        ctx.capabilities.require_native_execution = false;
        ctx.max_result_rows = (end - start).max(1);
        let actual = QueryEngine.execute(&query, &mut ctx)?;
        assert_eq!(rows(&actual), expected[start..end], "{query}");
        assert_eq!(
            actual
                .result
                .schema
                .iter()
                .map(|(name, _)| name.as_str())
                .collect::<Vec<_>>(),
            ["card_id", "display", "value", "kind"]
        );
    }
    let mut ctx = context(graph, Some(backend));
    ctx.capabilities.require_native_execution = false;
    ctx.max_result_rows = 365;
    let query = format!("MATCH (c:Contact) WHERE c.card_id <> '' {CONTACT_RETURN}");
    assert_eq!(
        QueryEngine
            .execute(&query, &mut ctx)
            .expect_err("output budget must be enforced")
            .code,
        ErrorCode::ResultBudgetExceeded
    );
    let mut ctx = context(graph, Some(backend));
    ctx.capabilities.require_native_execution = false;
    ctx.max_result_rows = 2;
    ctx.parameters.insert(
        "take".to_owned(),
        ResultValue::Scalar(ScalarValue::Integer(2)),
    );
    assert_eq!(
        rows(&QueryEngine.execute(&format!("{query} LIMIT $take"), &mut ctx)?),
        expected[..2]
    );
    let aliased = QueryEngine.execute(
        "MATCH (c:Contact) WITH c.card_id AS id WHERE id <> '' RETURN id LIMIT 2",
        &mut ctx,
    )?;
    assert_eq!(
        rows(&aliased),
        expected[..2]
            .iter()
            .map(|row| vec![row[0].clone()])
            .collect::<Vec<_>>()
    );
    let prior = [
        GraphMutation::SetNodeProperty {
            node: NodeId(1078),
            property: graph
                .catalog()
                .property("card_id")
                .expect("fixture property"),
            value: ScalarValue::String(Arc::from("")),
            revision: graph.revision() + 1,
        },
        GraphMutation::DeleteNode {
            node: NodeId(1079),
            detach: false,
            revision: graph.revision() + 2,
        },
    ];
    ctx.prior_graph_mutations = &prior;
    assert_eq!(
        rows(&QueryEngine.execute(&format!("{query} LIMIT 2"), &mut ctx)?),
        expected[2..4]
    );
    ctx.prior_graph_mutations = &[];
    assert_eq!(
        QueryEngine
            .execute(
                "MATCH (c:Contact) WHERE c.card_id RETURN c.card_id",
                &mut ctx
            )
            .expect_err("a non-boolean predicate must not be treated as true")
            .code,
        ErrorCode::QueryType
    );
    ctx.cancellation.cancel();
    assert_eq!(
        QueryEngine
            .execute(&query, &mut ctx)
            .expect_err("cancelled filtered read must stop")
            .code,
        ErrorCode::Cancelled
    );
    Ok(())
}

#[test]
fn direct_filter_contacts_do_not_enter_resident_preparation() -> Result<()> {
    let graph = direct_filter_contacts_graph()?;
    let mut cpu = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
    cpu.admit_project(image(&graph)?)?;
    let accelerator = RejectingAccelerator::new(cpu);
    assert_direct_filter_contacts(&graph, &accelerator)?;
    assert_eq!(accelerator.forbidden_host_calls.load(Ordering::SeqCst), 0);
    assert_eq!(accelerator.graph_procedure_calls.load(Ordering::SeqCst), 0);
    let strict = format!("MATCH (c:Contact) WHERE c.card_id <> '' {CONTACT_RETURN}");
    assert_eq!(
        execute(&graph, Some(&accelerator), &strict)
            .expect_err("native-only requests must retain their execution contract")
            .code,
        ErrorCode::GpuAdmissionFailure
    );
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_direct_filter_contacts_preserve_results() -> Result<()> {
    let graph = direct_filter_contacts_graph()?;
    let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
    metal.admit_project(image(&graph)?)?;
    assert_direct_filter_contacts(&graph, &metal)
}

#[test]
#[allow(clippy::too_many_lines)]
fn active_accelerator_dispatches_every_graph_algorithm_without_host_fallback() -> Result<()> {
    let graph = adversarial_graph()?;
    let mut cpu = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
    cpu.admit_project(image(&graph)?)?;
    let accelerator = RejectingAccelerator::new(cpu);
    let graph_calls = Arc::clone(&accelerator.graph_procedure_calls);
    let host_calls = Arc::clone(&accelerator.forbidden_host_calls);
    let overlay_calls = Arc::clone(&accelerator.overlay_calls);

    let error = execute(
        &graph,
        Some(&accelerator),
        "CALL graph.degree() YIELD node, degree RETURN node, degree",
    )
    .expect_err("supported resident procedure must enter the accelerator adapter");
    assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
    assert!(error.message.contains("deliberately rejected"));
    assert_eq!(graph_calls.load(Ordering::SeqCst), 1);
    assert_eq!(host_calls.load(Ordering::SeqCst), 0);

    for (expected_calls, query) in [
        (
            2,
            "CALL graph.bfs(10) YIELD node, distance RETURN node, distance",
        ),
        (3, "CALL graph.dfs(10) YIELD node, order RETURN node, order"),
        (
            4,
            "CALL graph.shortestpath(10, 40) YIELD path, cost RETURN path, cost",
        ),
        (
            5,
            "CALL graph.dijkstra(10) YIELD node, cost RETURN node, cost",
        ),
        (
            6,
            "CALL graph.dijkstra(10, 'weight') YIELD node, cost RETURN node, cost",
        ),
        (
            7,
            "CALL graph.wcc() YIELD node, component RETURN node, component",
        ),
        (
            8,
            "CALL graph.scc() YIELD node, component RETURN node, component",
        ),
        (
            9,
            "CALL graph.louvain() YIELD node, community RETURN node, community",
        ),
        (
            10,
            "CALL graph.pagerank() YIELD node, score RETURN node, score",
        ),
        (
            11,
            "CALL graph.trianglecount() YIELD triangleCount RETURN triangleCount",
        ),
        (
            12,
            "CALL graph.clusteringcoefficient() YIELD node, coefficient RETURN node, coefficient",
        ),
        (13, "CALL graph.kcore() YIELD node, core RETURN node, core"),
    ] {
        let error = execute(&graph, Some(&accelerator), query)
            .expect_err("every graph procedure must enter the accelerator adapter");
        assert_eq!(error.code, ErrorCode::GpuAdmissionFailure, "{query}");
        assert!(error.message.contains("deliberately rejected"), "{query}");
        assert_eq!(
            graph_calls.load(Ordering::SeqCst),
            expected_calls,
            "{query}"
        );
    }
    assert_eq!(
        host_calls.load(Ordering::SeqCst),
        0,
        "accelerator execution must never reconstruct or traverse host adjacency"
    );

    let mut stale = context(&graph, Some(&accelerator));
    stale.bookmark.index = stale.bookmark.index.saturating_add(1);
    let error = QueryEngine
        .execute(
            "CALL graph.degree() YIELD node, degree RETURN node, degree",
            &mut stale,
        )
        .expect_err("stale accelerator residency must fail closed");
    assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
    assert_eq!(
        graph_calls.load(Ordering::SeqCst),
        13,
        "a stale resident generation must be rejected before dispatch"
    );
    assert_eq!(
        host_calls.load(Ordering::SeqCst),
        0,
        "stale accelerator residency must not fall through to host adjacency"
    );

    let prior = [GraphMutation::InsertNode(NodeInput {
        id: NodeId(90),
        layer: Layer::Observed,
        revision: graph.revision().saturating_add(1),
        labels: Vec::new(),
        properties: Vec::new(),
    })];
    let mut overlaid = context(&graph, Some(&accelerator));
    overlaid.prior_graph_mutations = &prior;
    let error = QueryEngine
        .execute(
            "CALL graph.degree() YIELD node, degree RETURN node, degree",
            &mut overlaid,
        )
        .expect_err("prior transaction mutations must refresh the resident graph before dispatch");
    assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
    assert!(error.message.contains("deliberately rejected"));
    assert_eq!(overlay_calls.load(Ordering::SeqCst), 1);
    assert_eq!(graph_calls.load(Ordering::SeqCst), 14);
    assert_eq!(host_calls.load(Ordering::SeqCst), 0);
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
#[allow(clippy::too_many_lines)]
fn real_metal_all_graph_algorithms_match_cpu_on_adversarial_sparse_graph() -> Result<()> {
    let graph = adversarial_graph()?;
    let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
    metal.admit_project(image(&graph)?)?;

    for query in [
        "USE LAYER OBSERVED\n\
         CALL graph.degree() YIELD node, outDegree, inDegree, degree\n\
         RETURN id(node) AS id, outDegree, inDegree, degree ORDER BY id",
        "USE LAYER OBSERVED\n\
         CALL graph.bfs(10) YIELD node, distance\n\
         RETURN id(node) AS id, distance ORDER BY id",
        "USE LAYER OBSERVED\n\
         CALL graph.dijkstra(10) YIELD node, cost, predecessor\n\
         RETURN id(node) AS id, cost, id(predecessor) AS predecessor ORDER BY id",
        "USE LAYER OBSERVED\n\
         CALL graph.dijkstra(10, 'weight') YIELD node, cost, predecessor\n\
         RETURN id(node) AS id, cost, id(predecessor) AS predecessor ORDER BY id",
        "USE LAYER OBSERVED\n\
         CALL graph.dfs(10) YIELD node, order\n\
         RETURN id(node) AS id, order ORDER BY order",
        "USE LAYER OBSERVED\n\
         CALL graph.shortestpath(10, 40) YIELD path, cost RETURN path, cost",
        "USE LAYER OBSERVED\n\
         CALL graph.wcc() YIELD node, component\n\
         RETURN id(node) AS id, component ORDER BY id",
        "USE LAYER OBSERVED\n\
         CALL graph.scc() YIELD node, component\n\
         RETURN id(node) AS id, component ORDER BY id",
        "USE LAYER OBSERVED\n\
         CALL graph.louvain() YIELD node, community\n\
         RETURN id(node) AS id, community ORDER BY id",
        "USE LAYER OBSERVED\n\
         CALL graph.trianglecount() YIELD triangleCount RETURN triangleCount",
        "USE LAYER OBSERVED\n\
         CALL graph.kcore() YIELD node, core RETURN id(node) AS id, core ORDER BY id",
        "USE LAYER KNOWLEDGE\n\
         WRITE LAYER KNOWLEDGE\n\
         CALL graph.degree() YIELD node, outDegree, inDegree, degree\n\
         RETURN id(node) AS id, outDegree, inDegree, degree ORDER BY id",
        "USE LAYER KNOWLEDGE\n\
         WRITE LAYER KNOWLEDGE\n\
         CALL graph.bfs(70) YIELD node, distance\n\
         RETURN id(node) AS id, distance ORDER BY id",
        "USE LAYER KNOWLEDGE\n\
         WRITE LAYER KNOWLEDGE\n\
         CALL graph.dfs(70) YIELD node, order RETURN id(node) AS id, order ORDER BY order",
        "USE LAYER KNOWLEDGE\n\
         WRITE LAYER KNOWLEDGE\n\
         CALL graph.shortestpath(70, 80) YIELD path, cost RETURN path, cost",
        "USE LAYER KNOWLEDGE\n\
         WRITE LAYER KNOWLEDGE\n\
         CALL graph.dijkstra(70, 'weight') YIELD node, cost, predecessor\n\
         RETURN id(node) AS id, cost, id(predecessor) AS predecessor ORDER BY id",
        "USE LAYER KNOWLEDGE\n\
         WRITE LAYER KNOWLEDGE\n\
         CALL graph.wcc() YIELD node, component RETURN id(node) AS id, component ORDER BY id",
        "USE LAYER KNOWLEDGE\n\
         WRITE LAYER KNOWLEDGE\n\
         CALL graph.scc() YIELD node, component RETURN id(node) AS id, component ORDER BY id",
        "USE LAYER KNOWLEDGE\n\
         WRITE LAYER KNOWLEDGE\n\
         CALL graph.louvain() YIELD node, community RETURN id(node) AS id, community ORDER BY id",
        "USE LAYER KNOWLEDGE\n\
         WRITE LAYER KNOWLEDGE\n\
         CALL graph.trianglecount() YIELD triangleCount RETURN triangleCount",
        "USE LAYER KNOWLEDGE\n\
         WRITE LAYER KNOWLEDGE\n\
         CALL graph.kcore() YIELD node, core RETURN id(node) AS id, core ORDER BY id",
    ] {
        assert_exact_differential(&graph, &metal, query)?;
    }

    assert_clustering_differential(
        &graph,
        &metal,
        "USE LAYER OBSERVED\n\
         CALL graph.clusteringcoefficient() YIELD node, coefficient\n\
         RETURN id(node) AS id, coefficient ORDER BY id",
    )?;
    assert_clustering_differential(
        &graph,
        &metal,
        "USE LAYER KNOWLEDGE\n\
         WRITE LAYER KNOWLEDGE\n\
         CALL graph.clusteringcoefficient() YIELD node, coefficient\n\
         RETURN id(node) AS id, coefficient ORDER BY id",
    )?;

    for query in [
        "USE LAYER OBSERVED\n\
         CALL graph.pagerank(0.85, 0.0000001, 100) YIELD node, score\n\
         RETURN id(node) AS id, score ORDER BY id",
        "USE LAYER KNOWLEDGE\n\
         WRITE LAYER KNOWLEDGE\n\
         CALL graph.pagerank(0.9, 0.0000001, 80) YIELD node, score\n\
         RETURN id(node) AS id, score ORDER BY id",
        "USE LAYER OBSERVED\n\
         CALL graph.pagerank(0.9999999999999999, 1.0e300, 4) YIELD node, score\n\
         RETURN id(node) AS id, score ORDER BY id",
        "USE LAYER OBSERVED\n\
         CALL graph.pagerank(0.85, 1.0e-300, 2) YIELD node, score\n\
         RETURN id(node) AS id, score ORDER BY id",
    ] {
        let first = assert_pagerank_differential(&graph, &metal, query)?;
        let second = rows(&execute(&graph, Some(&metal), query)?);
        assert_eq!(
            first, second,
            "Metal PageRank must be deterministic: {query}"
        );
    }

    let mut cancelled = context(&graph, Some(&metal));
    cancelled.cancellation.cancel();
    let error = QueryEngine
        .execute(
            "CALL graph.bfs(10) YIELD node, distance RETURN node, distance",
            &mut cancelled,
        )
        .expect_err("pre-cancelled Metal traversal must fail");
    assert_eq!(error.code, ErrorCode::Cancelled);

    for query in [
        "CALL graph.degree() YIELD node RETURN node",
        "CALL graph.bfs(10) YIELD node RETURN node",
        "CALL graph.dfs(10) YIELD node RETURN node",
        "CALL graph.dijkstra(10) YIELD node RETURN node",
        "CALL graph.dijkstra(10, 'weight') YIELD node RETURN node",
        "CALL graph.wcc() YIELD node RETURN node",
        "CALL graph.scc() YIELD node RETURN node",
        "CALL graph.louvain() YIELD node RETURN node",
        "CALL graph.pagerank() YIELD node RETURN node",
        "CALL graph.clusteringcoefficient() YIELD node RETURN node",
        "CALL graph.kcore() YIELD node RETURN node",
    ] {
        let mut bounded = context(&graph, Some(&metal));
        bounded.max_result_rows = 3;
        let error = QueryEngine
            .execute(query, &mut bounded)
            .expect_err("Metal graph procedure must honor the output row budget");
        assert_eq!(error.code, ErrorCode::ResultBudgetExceeded, "{query}");
    }
    for query in [
        "CALL graph.shortestpath(10, 40) YIELD path RETURN path",
        "CALL graph.trianglecount() YIELD triangleCount RETURN triangleCount",
    ] {
        let mut bounded = context(&graph, Some(&metal));
        bounded.max_result_rows = 0;
        let error = QueryEngine
            .execute(query, &mut bounded)
            .expect_err("Metal single-row graph procedure must honor a zero output budget");
        assert_eq!(error.code, ErrorCode::ResultBudgetExceeded, "{query}");
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_scc_matches_cpu_across_random_directed_graphs() -> Result<()> {
    const NODES: u64 = 48;
    const EDGES: u64 = 320;
    let mut random = 0xd1b5_4a32_d192_ed03_u64;
    for fixture in 0_u64..16 {
        let mut graph = GraphStore::default();
        let vertex = graph.catalog_mut().intern_label("Vertex")?;
        let connected = graph.catalog_mut().intern_relationship_type("CONNECTED")?;
        for node in 0..NODES {
            graph.insert_node(NodeInput {
                id: NodeId(10_000 + node),
                layer: Layer::Observed,
                revision: node + 1,
                labels: vec![vertex],
                properties: Vec::new(),
            })?;
        }
        for edge in 0..EDGES {
            random = random
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(0x1405_7b7e_f767_814f ^ fixture);
            let source = random % NODES;
            random = random
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            let target = random % NODES;
            graph.insert_edge(EdgeInput {
                id: EdgeId(100_000 + fixture * EDGES + edge),
                source: NodeId(10_000 + source),
                target: NodeId(10_000 + target),
                relationship_type: connected,
                layer: Layer::Observed,
                revision: NODES + edge + 1,
                properties: Vec::new(),
            })?;
        }
        let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
        metal.admit_project(image(&graph)?)?;
        assert_exact_differential(
            &graph,
            &metal,
            "CALL graph.scc() YIELD node, component\n\
             RETURN id(node) AS id, component ORDER BY id",
        )?;
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_components_metrics_and_pagerank_match_seeded_layered_graphs_exactly() -> Result<()> {
    for seed in 0_u64..3 {
        let graph = randomized_components_metrics_graph(seed)?;
        let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
        metal.admit_project(image(&graph)?)?;
        for (layer, prelude) in [
            ("OBSERVED", "USE LAYER OBSERVED\n"),
            ("KNOWLEDGE", "USE LAYER KNOWLEDGE\nWRITE LAYER KNOWLEDGE\n"),
        ] {
            for body in [
                "CALL graph.degree() YIELD node, outDegree, inDegree, degree \
                 RETURN id(node) AS id, outDegree, inDegree, degree ORDER BY id",
                "CALL graph.wcc() YIELD node, component \
                 RETURN id(node) AS id, component ORDER BY id",
                "CALL graph.scc() YIELD node, component \
                 RETURN id(node) AS id, component ORDER BY id",
                "CALL graph.trianglecount() YIELD triangleCount RETURN triangleCount",
                "CALL graph.kcore() YIELD node, core \
                 RETURN id(node) AS id, core ORDER BY id",
            ] {
                let query = format!("{prelude}{body}");
                assert_exact_differential(&graph, &metal, &query)?;
                let first = rows(&execute(&graph, Some(&metal), &query)?);
                let second = rows(&execute(&graph, Some(&metal), &query)?);
                assert_eq!(
                    first, second,
                    "Metal result is not repeatable: seed={seed}, layer={layer}, query={body}"
                );
            }

            let clustering = format!(
                "{prelude}CALL graph.clusteringcoefficient() YIELD node, coefficient \
                 RETURN id(node) AS id, coefficient ORDER BY id"
            );
            assert_clustering_differential(&graph, &metal, &clustering)?;
            assert_eq!(
                rows(&execute(&graph, Some(&metal), &clustering)?),
                rows(&execute(&graph, Some(&metal), &clustering)?),
                "Metal clustering is not repeatable: seed={seed}, layer={layer}"
            );

            let (damping, tolerance, iterations) = match seed {
                0 => ("0.85", "1.0e-12", 80),
                1 => ("0.9999999999999999", "1.0e300", 4),
                _ => ("0.0000000000000001", "1.0e-300", 7),
            };
            let pagerank = format!(
                "{prelude}CALL graph.pagerank({damping}, {tolerance}, {iterations}) \
                 YIELD node, score RETURN id(node) AS id, score ORDER BY id"
            );
            let first = assert_pagerank_differential(&graph, &metal, &pagerank)?;
            let second = rows(&execute(&graph, Some(&metal), &pagerank)?);
            assert_eq!(
                first, second,
                "Metal PageRank is not repeatable: seed={seed}, layer={layer}"
            );
        }
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "manual Apple-GPU sparse SCC throughput gate"]
fn metal_scc_many_singleton_dag_performance_gate() -> Result<()> {
    const NODES: usize = 20_000;
    const WIDTH: usize = 320;
    const FANOUT: usize = 4;
    let mut graph = GraphStore::default();
    let vertex = graph.catalog_mut().intern_label("Vertex")?;
    let connected = graph.catalog_mut().intern_relationship_type("CONNECTED")?;
    for node in 0..NODES {
        graph.insert_node(NodeInput {
            id: NodeId(u64::try_from(node + 1).expect("node ID fits u64")),
            layer: Layer::Observed,
            revision: u64::try_from(node + 1).expect("revision fits u64"),
            labels: vec![vertex],
            properties: Vec::new(),
        })?;
    }
    let mut edge_id = 1_u64;
    for source in 0..NODES.saturating_sub(WIDTH) {
        let source_column = source % WIDTH;
        for offset in 0..FANOUT {
            let target = source - source_column + WIDTH + (source_column + offset * 73) % WIDTH;
            if target >= NODES {
                continue;
            }
            graph.insert_edge(EdgeInput {
                id: EdgeId(1_000_000 + edge_id),
                source: NodeId(u64::try_from(source + 1).expect("source fits u64")),
                target: NodeId(u64::try_from(target + 1).expect("target fits u64")),
                relationship_type: connected,
                layer: Layer::Observed,
                revision: u64::try_from(NODES).expect("node count fits u64") + edge_id,
                properties: Vec::new(),
            })?;
            edge_id += 1;
        }
    }
    let mut metal = MetalBackend::new(0, 512 * 1024 * 1024, RESERVED_BYTES)?;
    metal.admit_project(image(&graph)?)?;
    let query = "CALL graph.scc() YIELD node, component\n\
                 RETURN id(node) AS id, component ORDER BY id";

    let mut cpu_context = context(&graph, None);
    cpu_context.max_result_rows = NODES + 1;
    cpu_context.deadline = Some(Instant::now() + Duration::from_mins(1));
    let cpu = QueryEngine.execute(query, &mut cpu_context)?;

    let started = Instant::now();
    let mut metal_context = context(&graph, Some(&metal));
    metal_context.max_result_rows = NODES + 1;
    metal_context.deadline = Some(Instant::now() + Duration::from_mins(1));
    let device = QueryEngine.execute(query, &mut metal_context)?;
    let elapsed = started.elapsed();
    assert_eq!(device.result.schema, cpu.result.schema);
    assert_eq!(device.result.batches, cpu.result.batches);
    assert!(
        elapsed < Duration::from_secs(10),
        "20k-node/approximately-80k-edge DAG SCC took {elapsed:?}"
    );
    eprintln!(
        "Metal SCC many-singleton DAG: nodes={NODES} edges={} elapsed={elapsed:?}",
        edge_id - 1
    );
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "manual Apple-GPU long-cycle SCC throughput gate"]
fn metal_scc_long_directed_cycle_performance_gate() -> Result<()> {
    const NODES: usize = 20_000;
    let mut graph = GraphStore::default();
    let vertex = graph.catalog_mut().intern_label("CycleVertex")?;
    let connected = graph
        .catalog_mut()
        .intern_relationship_type("DIRECTED_CYCLE")?;
    for node in 0..NODES {
        graph.insert_node(NodeInput {
            id: NodeId(u64::try_from(node + 1).expect("node ID fits u64")),
            layer: Layer::Observed,
            revision: u64::try_from(node + 1).expect("revision fits u64"),
            labels: vec![vertex],
            properties: Vec::new(),
        })?;
    }
    for source in 0..NODES {
        let target = (source + 1) % NODES;
        graph.insert_edge(EdgeInput {
            id: EdgeId(1_000_000 + u64::try_from(source).expect("edge fits u64")),
            source: NodeId(u64::try_from(source + 1).expect("source fits u64")),
            target: NodeId(u64::try_from(target + 1).expect("target fits u64")),
            relationship_type: connected,
            layer: Layer::Observed,
            revision: u64::try_from(NODES + source + 1).expect("revision fits u64"),
            properties: Vec::new(),
        })?;
    }
    let query = "CALL graph.scc() YIELD node, component \
                 RETURN id(node) AS id, component ORDER BY id";
    let mut cpu_context = context(&graph, None);
    cpu_context.max_result_rows = NODES + 1;
    cpu_context.deadline = Some(Instant::now() + Duration::from_mins(1));
    let cpu = QueryEngine.execute(query, &mut cpu_context)?;
    let mut metal = MetalBackend::new(0, 512 * 1024 * 1024, RESERVED_BYTES)?;
    metal.admit_project(image(&graph)?)?;
    let started = Instant::now();
    let mut metal_context = context(&graph, Some(&metal));
    metal_context.max_result_rows = NODES + 1;
    metal_context.deadline = Some(Instant::now() + Duration::from_mins(1));
    let device = QueryEngine.execute(query, &mut metal_context)?;
    let elapsed = started.elapsed();
    assert_eq!(device.result.schema, cpu.result.schema);
    assert_eq!(device.result.batches, cpu.result.batches);
    eprintln!("Metal SCC long directed cycle: nodes={NODES} elapsed={elapsed:?}");
    assert!(
        elapsed < Duration::from_secs(5),
        "20k-node directed-cycle SCC took {elapsed:?}"
    );
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "manual Apple-GPU WCC-chain and k-core-cascade throughput gate"]
fn metal_wcc_chain_and_kcore_cascade_performance_gate() -> Result<()> {
    const NODES: usize = 20_000;
    let mut graph = GraphStore::default();
    let vertex = graph.catalog_mut().intern_label("ChainVertex")?;
    let connected = graph.catalog_mut().intern_relationship_type("CHAIN_EDGE")?;
    for node in 0..NODES {
        graph.insert_node(NodeInput {
            id: NodeId(u64::try_from(node + 1).expect("node ID fits u64")),
            layer: Layer::Observed,
            revision: u64::try_from(node + 1).expect("revision fits u64"),
            labels: vec![vertex],
            properties: Vec::new(),
        })?;
    }
    for left in 0..NODES - 1 {
        let (source, target) = if left.is_multiple_of(2) {
            (left, left + 1)
        } else {
            (left + 1, left)
        };
        graph.insert_edge(EdgeInput {
            id: EdgeId(1_000_000 + u64::try_from(left).expect("edge fits u64")),
            source: NodeId(u64::try_from(source + 1).expect("source fits u64")),
            target: NodeId(u64::try_from(target + 1).expect("target fits u64")),
            relationship_type: connected,
            layer: Layer::Observed,
            revision: u64::try_from(NODES + left + 1).expect("revision fits u64"),
            properties: Vec::new(),
        })?;
    }
    let mut metal = MetalBackend::new(0, 512 * 1024 * 1024, RESERVED_BYTES)?;
    metal.admit_project(image(&graph)?)?;
    for (name, query) in [
        (
            "WCC chain",
            "CALL graph.wcc() YIELD node, component \
             RETURN id(node) AS id, component ORDER BY id",
        ),
        (
            "k-core path cascade",
            "CALL graph.kcore() YIELD node, core \
             RETURN id(node) AS id, core ORDER BY id",
        ),
    ] {
        let mut cpu_context = context(&graph, None);
        cpu_context.max_result_rows = NODES + 1;
        cpu_context.deadline = Some(Instant::now() + Duration::from_mins(1));
        let cpu = QueryEngine.execute(query, &mut cpu_context)?;
        let started = Instant::now();
        let mut metal_context = context(&graph, Some(&metal));
        metal_context.max_result_rows = NODES + 1;
        metal_context.deadline = Some(Instant::now() + Duration::from_mins(1));
        let device = QueryEngine.execute(query, &mut metal_context)?;
        let elapsed = started.elapsed();
        assert_eq!(device.result.schema, cpu.result.schema, "{name}");
        assert_eq!(device.result.batches, cpu.result.batches, "{name}");
        eprintln!("Metal {name}: nodes={NODES} elapsed={elapsed:?}");
        assert!(
            elapsed < Duration::from_secs(5),
            "20k-node {name} took {elapsed:?}"
        );
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn insert_metrics_edge(
    graph: &mut GraphStore,
    relationship_type: irongraph::types::RelationshipTypeId,
    source: usize,
    target: usize,
    ordinal: &mut u64,
    node_count: usize,
) -> Result<()> {
    graph.insert_edge(EdgeInput {
        id: EdgeId(1_000_000 + *ordinal),
        source: NodeId(u64::try_from(source + 1).expect("source ID fits u64")),
        target: NodeId(u64::try_from(target + 1).expect("target ID fits u64")),
        relationship_type,
        layer: Layer::Observed,
        revision: u64::try_from(node_count).expect("node count fits u64") + *ordinal,
        properties: Vec::new(),
    })?;
    *ordinal += 1;
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn power_law_metrics_graph() -> Result<(GraphStore, u64)> {
    const NODES: usize = 8_000;
    const HUBS: usize = 64;
    let mut graph = GraphStore::default();
    let vertex = graph.catalog_mut().intern_label("PowerLawVertex")?;
    let connected = graph
        .catalog_mut()
        .intern_relationship_type("POWER_LAW_EDGE")?;
    for node in 0..NODES {
        graph.insert_node(NodeInput {
            id: NodeId(u64::try_from(node + 1).expect("node ID fits u64")),
            layer: Layer::Observed,
            revision: u64::try_from(node + 1).expect("revision fits u64"),
            labels: vec![vertex],
            properties: Vec::new(),
        })?;
    }

    // Put the high-degree vertices at the end of dense row order. This prevents an accidental
    // early-exit advantage for row-serial intersections and reflects stable IDs that are not
    // correlated with degree. The four tiers produce a deliberately skewed degree distribution.
    let first_hub = NODES - HUBS;
    let mut edge = 1_u64;
    for left in 0..HUBS {
        for right in left + 1..HUBS {
            insert_metrics_edge(
                &mut graph,
                connected,
                first_hub + left,
                first_hub + right,
                &mut edge,
                NODES,
            )?;
        }
    }
    for leaf in 0..first_hub {
        let hubs = [
            first_hub,
            first_hub + 1 + leaf % 3,
            first_hub + 4 + (leaf.wrapping_mul(17) + 5) % 12,
            first_hub + 16 + (leaf.wrapping_mul(29) + 11) % 48,
        ];
        for (slot, hub) in hubs.into_iter().enumerate() {
            let (source, target) = if (leaf + slot).is_multiple_of(2) {
                (leaf, hub)
            } else {
                (hub, leaf)
            };
            insert_metrics_edge(&mut graph, connected, source, target, &mut edge, NODES)?;
        }
    }
    let leaves = u64::try_from(first_hub).expect("leaf count fits u64");
    let hubs = u64::try_from(HUBS).expect("hub count fits u64");
    let triangles = hubs * (hubs - 1) * (hubs - 2) / 6 + leaves * 6;
    Ok((graph, triangles))
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn dense_metrics_graph() -> Result<(GraphStore, u64)> {
    const GROUPS: usize = 8;
    const GROUP_SIZE: usize = 192;
    const NODES: usize = GROUPS * GROUP_SIZE;
    let mut graph = GraphStore::default();
    let vertex = graph.catalog_mut().intern_label("DenseVertex")?;
    let connected = graph.catalog_mut().intern_relationship_type("DENSE_EDGE")?;
    for node in 0..NODES {
        graph.insert_node(NodeInput {
            id: NodeId(u64::try_from(node + 1).expect("node ID fits u64")),
            layer: Layer::Observed,
            revision: u64::try_from(node + 1).expect("revision fits u64"),
            labels: vec![vertex],
            properties: Vec::new(),
        })?;
    }
    let mut edge = 1_u64;
    for group in 0..GROUPS {
        let begin = group * GROUP_SIZE;
        for left in 0..GROUP_SIZE {
            for right in left + 1..GROUP_SIZE {
                insert_metrics_edge(
                    &mut graph,
                    connected,
                    begin + left,
                    begin + right,
                    &mut edge,
                    NODES,
                )?;
            }
        }
    }
    let size = u64::try_from(GROUP_SIZE).expect("group size fits u64");
    let triangles =
        u64::try_from(GROUPS).expect("group count fits u64") * size * (size - 1) * (size - 2) / 6;
    Ok((graph, triangles))
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn isolated_high_degree_triangle_row_graph(spokes_per_hub: usize) -> Result<GraphStore> {
    let node_count = 2_usize
        .checked_add(spokes_per_hub.checked_mul(2).ok_or_else(|| {
            Error::new(ErrorCode::ResultBudgetExceeded, "hub fixture size overflow")
        })?)
        .ok_or_else(|| Error::new(ErrorCode::ResultBudgetExceeded, "hub fixture size overflow"))?;
    let mut graph = GraphStore::default();
    let vertex = graph.catalog_mut().intern_label("HighDegreeVertex")?;
    let connected = graph
        .catalog_mut()
        .intern_relationship_type("HIGH_DEGREE_EDGE")?;
    for node in 0..node_count {
        graph.insert_node(NodeInput {
            id: NodeId(u64::try_from(node + 1).expect("node ID fits u64")),
            layer: Layer::Observed,
            revision: u64::try_from(node + 1).expect("revision fits u64"),
            labels: vec![vertex],
            properties: Vec::new(),
        })?;
    }
    let mut edge = 1_u64;
    insert_metrics_edge(&mut graph, connected, 0, 1, &mut edge, node_count)?;
    for spoke in 0..spokes_per_hub {
        let left_leaf = 2 + spoke;
        let right_leaf = 2 + spokes_per_hub + spoke;
        let (left_source, left_target) = if spoke.is_multiple_of(2) {
            (0, left_leaf)
        } else {
            (left_leaf, 0)
        };
        let (right_source, right_target) = if spoke.is_multiple_of(2) {
            (right_leaf, 1)
        } else {
            (1, right_leaf)
        };
        insert_metrics_edge(
            &mut graph,
            connected,
            left_source,
            left_target,
            &mut edge,
            node_count,
        )?;
        insert_metrics_edge(
            &mut graph,
            connected,
            right_source,
            right_target,
            &mut edge,
            node_count,
        )?;
    }
    Ok(graph)
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn timed_metrics_queries(
    name: &str,
    graph: &GraphStore,
    expected_triangles: u64,
) -> Result<(Duration, Duration, Duration, Duration)> {
    let mut metal = MetalBackend::new(0, 512 * 1024 * 1024, RESERVED_BYTES)?;
    metal.admit_project(image(graph)?)?;

    let triangle_query = "CALL graph.trianglecount() YIELD triangleCount RETURN triangleCount";
    let cpu_triangle_started = Instant::now();
    let cpu_triangle = execute(graph, None, triangle_query)?;
    let cpu_triangle_elapsed = cpu_triangle_started.elapsed();
    let triangle_started = Instant::now();
    let triangle = execute(graph, Some(&metal), triangle_query)?;
    let triangle_elapsed = triangle_started.elapsed();
    assert_eq!(triangle.result.schema, cpu_triangle.result.schema);
    assert_eq!(triangle.result.batches, cpu_triangle.result.batches);
    assert_eq!(
        integer(&rows(&triangle)[0][0]),
        i64::try_from(expected_triangles).expect("triangle count fits i64")
    );

    let clustering_query = "CALL graph.clusteringcoefficient() YIELD node, coefficient \
         RETURN id(node) AS id, coefficient ORDER BY id";
    let cpu_clustering_started = Instant::now();
    let cpu_clustering = execute(graph, None, clustering_query)?;
    let cpu_clustering_elapsed = cpu_clustering_started.elapsed();
    let clustering_started = Instant::now();
    let clustering = execute(graph, Some(&metal), clustering_query)?;
    let clustering_elapsed = clustering_started.elapsed();
    assert_eq!(clustering.result.schema, cpu_clustering.result.schema);
    assert_eq!(clustering.result.batches, cpu_clustering.result.batches);
    assert_eq!(rows(&clustering).len(), graph.node_count());
    eprintln!(
        "{name} metrics: nodes={} edges={} triangle CPU={cpu_triangle_elapsed:?} Metal={triangle_elapsed:?} speedup={:.2}x; clustering CPU={cpu_clustering_elapsed:?} Metal={clustering_elapsed:?} speedup={:.2}x",
        graph.node_count(),
        graph.edge_count(),
        cpu_triangle_elapsed.as_secs_f64() / triangle_elapsed.as_secs_f64(),
        cpu_clustering_elapsed.as_secs_f64() / clustering_elapsed.as_secs_f64(),
    );
    Ok((
        cpu_triangle_elapsed,
        triangle_elapsed,
        cpu_clustering_elapsed,
        clustering_elapsed,
    ))
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "manual Apple-GPU triangle/clustering throughput gate"]
fn metal_triangle_clustering_power_law_and_dense_performance_gate() -> Result<()> {
    for (name, fixture) in [
        ("power-law", power_law_metrics_graph()?),
        ("dense-cliques", dense_metrics_graph()?),
    ] {
        let (graph, triangles) = fixture;
        let (cpu_triangle, triangle, cpu_clustering, clustering) =
            timed_metrics_queries(name, &graph, triangles)?;
        assert!(
            triangle < Duration::from_secs(5),
            "{name} triangle count took {triangle:?}"
        );
        assert!(
            clustering < Duration::from_secs(5),
            "{name} clustering coefficient took {clustering:?}"
        );
        assert!(
            triangle < cpu_triangle,
            "{name} Metal triangle count did not beat CPU: Metal={triangle:?} CPU={cpu_triangle:?}"
        );
        assert!(
            clustering < cpu_clustering,
            "{name} Metal clustering did not beat CPU: Metal={clustering:?} CPU={cpu_clustering:?}"
        );
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "manual Apple-GPU high-degree triangle cancellation gate"]
fn metal_triangle_single_high_degree_row_has_bounded_cancellation() -> Result<()> {
    const SPOKES_PER_HUB: usize = 40_000;
    let graph = isolated_high_degree_triangle_row_graph(SPOKES_PER_HUB)?;
    let query = "CALL graph.trianglecount() YIELD triangleCount RETURN triangleCount";
    let cpu_started = Instant::now();
    let cpu = execute(&graph, None, query)?;
    let cpu_elapsed = cpu_started.elapsed();
    let mut metal = MetalBackend::new(0, 512 * 1024 * 1024, RESERVED_BYTES)?;
    metal.admit_project(image(&graph)?)?;
    let metal_started = Instant::now();
    let device = execute(&graph, Some(&metal), query)?;
    let metal_elapsed = metal_started.elapsed();
    assert_eq!(device.result.schema, cpu.result.schema);
    assert_eq!(device.result.batches, cpu.result.batches);
    assert_eq!(integer(&rows(&device)[0][0]), 0);

    let cancellation = CancellationToken::new();
    let trigger = cancellation.clone();
    let canceller = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(10));
        trigger.cancel();
    });
    let request = ResidentGraphProcedureRequest {
        project: PROJECT,
        layers: LayerMask::OBSERVED,
        procedure: ResidentGraphProcedure::TriangleCount,
        max_output_rows: graph.node_count(),
        deadline: None,
    };
    let cancelled_started = Instant::now();
    let error = metal
        .execute_graph_procedure(&request, &cancellation)
        .expect_err("high-degree triangle execution must observe cancellation");
    let cancelled_elapsed = cancelled_started.elapsed();
    canceller
        .join()
        .map_err(|_| Error::internal("triangle cancellation thread panicked"))?;
    assert_eq!(error.code, ErrorCode::Cancelled);
    assert!(
        cancelled_elapsed < Duration::from_secs(2),
        "high-degree triangle cancellation took {cancelled_elapsed:?}"
    );
    eprintln!(
        "single high-degree triangle row: nodes={} edges={} row_degree={} CPU={cpu_elapsed:?} Metal={metal_elapsed:?} cancellation={cancelled_elapsed:?}",
        graph.node_count(),
        graph.edge_count(),
        SPOKES_PER_HUB + 1,
    );
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "manual Apple-GPU direct-degree throughput gate"]
fn metal_direct_degree_skew_performance_gate() -> Result<()> {
    let (graph, _) = power_law_metrics_graph()?;
    let query = "CALL graph.degree() YIELD node, outDegree, inDegree, degree \
                 RETURN id(node) AS id, outDegree, inDegree, degree ORDER BY id";
    let cpu_started = Instant::now();
    let cpu = execute(&graph, None, query)?;
    let cpu_elapsed = cpu_started.elapsed();
    let mut metal = MetalBackend::new(0, 512 * 1024 * 1024, RESERVED_BYTES)?;
    metal.admit_project(image(&graph)?)?;
    let cold_started = Instant::now();
    let cold = execute(&graph, Some(&metal), query)?;
    let cold_elapsed = cold_started.elapsed();
    let warm_started = Instant::now();
    let warm = execute(&graph, Some(&metal), query)?;
    let warm_elapsed = warm_started.elapsed();
    assert_eq!(cold.result.schema, cpu.result.schema);
    assert_eq!(cold.result.batches, cpu.result.batches);
    assert_eq!(warm.result.batches, cpu.result.batches);
    eprintln!(
        "direct degree skew: nodes={} edges={} CPU={cpu_elapsed:?} Metal-cold={cold_elapsed:?} Metal-warm={warm_elapsed:?} warm-speedup={:.2}x",
        graph.node_count(),
        graph.edge_count(),
        cpu_elapsed.as_secs_f64() / warm_elapsed.as_secs_f64(),
    );
    assert!(
        warm_elapsed < Duration::from_secs(2),
        "direct degree warm execution took {warm_elapsed:?}"
    );
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_exact_path_edge_cases_match_public_cypher_contract() -> Result<()> {
    let graph = exact_path_graph()?;
    let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
    metal.admit_project(image(&graph)?)?;

    for property in ["integerWeight", "floatWeight", "mixedWeight"] {
        let query = format!(
            "CALL graph.dijkstra(1, '{property}') YIELD node, cost, predecessor \
             RETURN id(node) AS id, cost, id(predecessor) AS predecessor ORDER BY id"
        );
        assert_exact_differential(&graph, &metal, &query)?;
        let first = execute(&graph, Some(&metal), &query)?.result.batches;
        let second = execute(&graph, Some(&metal), &query)?.result.batches;
        assert_eq!(
            first, second,
            "weighted Dijkstra is not deterministic: {property}"
        );
    }

    for query in [
        // The first path is lexicographically smaller, while its final predecessor has the larger
        // dense row. This catches reconstruction that incorrectly minimizes only predecessor ID.
        "CALL graph.shortestpath(1, 6) YIELD path, cost RETURN path, cost",
        "CALL graph.shortestpath(1, 1) YIELD path, cost RETURN path, cost",
        "CALL graph.shortestpath(1, 10) YIELD path, cost RETURN path, cost",
        "CALL graph.dfs(1) YIELD node, order RETURN id(node) AS id, order ORDER BY order",
    ] {
        assert_exact_differential(&graph, &metal, query)?;
        let first = execute(&graph, Some(&metal), query)?.result.batches;
        let second = execute(&graph, Some(&metal), query)?.result.batches;
        assert_eq!(
            first, second,
            "path procedure is not deterministic: {query}"
        );
    }
    assert!(
        rows(&execute(
            &graph,
            Some(&metal),
            "CALL graph.shortestpath(1, 10) YIELD path, cost RETURN path, cost",
        )?)
        .is_empty(),
        "unreachable shortest path must publish no row"
    );
    let same = rows(&execute(
        &graph,
        Some(&metal),
        "CALL graph.shortestpath(1, 1) YIELD path, cost RETURN path, cost",
    )?);
    assert_eq!(
        same.len(),
        1,
        "source==target must publish one zero-hop path"
    );
    assert_eq!(integer(&same[0][1]), 0);

    for property in [
        "missingWeight",
        "stringWeight",
        "negativeWeight",
        "nanWeight",
        "infiniteWeight",
    ] {
        let query = format!(
            "CALL graph.dijkstra(1, '{property}') YIELD node, cost RETURN id(node) AS id, cost"
        );
        let cpu = execute(&graph, None, &query)
            .expect_err("CPU reference accepted an invalid reachable edge weight");
        let device = execute(&graph, Some(&metal), &query)
            .expect_err("Metal accepted an invalid reachable edge weight");
        assert_eq!(device.code, ErrorCode::QueryType, "{property}: {device:?}");
        assert_eq!(device.code, cpu.code, "error-code parity: {property}");
        let unreachable_query = format!(
            "CALL graph.dijkstra(10, '{property}') YIELD node, cost \
             RETURN id(node) AS id, cost ORDER BY id"
        );
        assert_exact_differential(&graph, &metal, &unreachable_query)?;
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_randomized_paths_match_cpu_and_repeat_through_public_cypher() -> Result<()> {
    for seed in 0..12_u64 {
        let (
            graph,
            [
                observed_source,
                observed_target,
                knowledge_source,
                knowledge_target,
            ],
        ) = randomized_path_graph(seed)?;
        let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
        metal.admit_project(image(&graph)?)?;
        for (preamble, source, target) in [
            ("USE LAYER OBSERVED\n", observed_source, observed_target),
            (
                "USE LAYER KNOWLEDGE\nWRITE LAYER KNOWLEDGE\n",
                knowledge_source,
                knowledge_target,
            ),
        ] {
            for query in [
                format!(
                    "{preamble}CALL graph.bfs({source}) YIELD node, distance \
                     RETURN id(node) AS id, distance ORDER BY id"
                ),
                format!(
                    "{preamble}CALL graph.dfs({source}) YIELD node, order \
                     RETURN id(node) AS id, order ORDER BY order"
                ),
                format!(
                    "{preamble}CALL graph.shortestpath({source}, {target}) YIELD path, cost \
                     RETURN path, cost"
                ),
                format!(
                    "{preamble}CALL graph.dijkstra({source}) YIELD node, cost, predecessor \
                     RETURN id(node) AS id, cost, id(predecessor) AS predecessor ORDER BY id"
                ),
                format!(
                    "{preamble}CALL graph.dijkstra({source}, 'weight') \
                     YIELD node, cost, predecessor \
                     RETURN id(node) AS id, cost, id(predecessor) AS predecessor ORDER BY id"
                ),
            ] {
                assert_exact_differential(&graph, &metal, &query)?;
                let first = execute(&graph, Some(&metal), &query)?.result.batches;
                let second = execute(&graph, Some(&metal), &query)?.result.batches;
                assert_eq!(
                    first, second,
                    "seeded Metal path output is not deterministic: seed={seed}, query={query}"
                );
            }
        }
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_adaptive_heap_dijkstra_matches_cpu_and_is_repeatable() -> Result<()> {
    // The expensive direct labels create a broad first wave, then the unit chain leaves one
    // improving target per round. This intentionally crosses the production frontier probe into
    // the persistent device heap instead of testing only the grid-wide relaxation engine.
    let (graph, _) = weighted_re_relaxation_graph(128)?;
    let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
    metal.admit_project(image(&graph)?)?;
    let query = "CALL graph.dijkstra(1, 'weight') YIELD node, cost, predecessor \
                 RETURN id(node) AS id, cost, id(predecessor) AS predecessor ORDER BY id";
    assert_exact_differential(&graph, &metal, query)?;
    let first = execute(&graph, Some(&metal), query)?.result.batches;
    let second = execute(&graph, Some(&metal), query)?.result.batches;
    assert_eq!(first, second, "adaptive heap Dijkstra is not repeatable");
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_persistent_bfs_unit_and_shortest_match_cpu_and_repeat() -> Result<()> {
    // A directed chain leaves one node in every measured frontier. BFS, unit Dijkstra, and both
    // the forward and reverse halves of shortest path therefore necessarily restart through the
    // shared persistent FIFO after the eight-level probe.
    let graph = directed_chain_graph(128)?;
    let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
    metal.admit_project(image(&graph)?)?;
    for query in [
        "CALL graph.bfs(1) YIELD node, distance \
         RETURN id(node) AS id, distance ORDER BY id",
        "CALL graph.dijkstra(1) YIELD node, cost, predecessor \
         RETURN id(node) AS id, cost, id(predecessor) AS predecessor ORDER BY id",
        "CALL graph.shortestpath(1, 128) YIELD path, cost RETURN path, cost",
    ] {
        assert_exact_differential(&graph, &metal, query)?;
        let first = execute(&graph, Some(&metal), query)?.result.batches;
        let second = execute(&graph, Some(&metal), query)?.result.batches;
        assert_eq!(
            first, second,
            "persistent traversal is not repeatable: {query}"
        );
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_path_hub_rows_observe_active_cancellation_and_deadline_within_two_seconds()
-> Result<()> {
    const LEAVES: usize = 100_000;
    let mut graph = high_degree_star_graph(LEAVES)?;
    let weight_property = graph.catalog_mut().intern_property("weight")?;
    let mut metal = MetalBackend::new(0, 512 * 1024 * 1024, RESERVED_BYTES)?;
    metal.admit_project(image(&graph)?)?;

    let expired = Instant::now()
        .checked_sub(Duration::from_millis(1))
        .ok_or_else(|| Error::internal("test deadline underflow"))?;
    let error = metal
        .execute_graph_procedure(
            &ResidentGraphProcedureRequest {
                project: PROJECT,
                layers: LayerMask::OBSERVED,
                procedure: ResidentGraphProcedure::Degree,
                max_output_rows: graph.node_count(),
                deadline: Some(expired),
            },
            &CancellationToken::new(),
        )
        .expect_err("an expired direct Metal graph request must fail");
    assert_eq!(error.code, ErrorCode::DeadlineExceeded);

    for procedure in [
        ResidentGraphProcedure::BreadthFirst { source_dense: 0 },
        ResidentGraphProcedure::DepthFirst { source_dense: 0 },
        ResidentGraphProcedure::ShortestPath {
            source_dense: 0,
            target_dense: u32::try_from(LEAVES).unwrap(),
        },
        ResidentGraphProcedure::DijkstraUnit { source_dense: 0 },
        ResidentGraphProcedure::DijkstraWeighted {
            source_dense: 0,
            weight_property,
        },
    ] {
        let cancellation = CancellationToken::new();
        let trigger = cancellation.clone();
        let canceller = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(1));
            trigger.cancel();
        });
        let started = Instant::now();
        let error = metal
            .execute_graph_procedure(
                &ResidentGraphProcedureRequest {
                    project: PROJECT,
                    layers: LayerMask::OBSERVED,
                    procedure,
                    max_output_rows: graph.node_count(),
                    deadline: None,
                },
                &cancellation,
            )
            .expect_err("an actively cancelled high-degree traversal must not complete normally");
        let elapsed = started.elapsed();
        canceller
            .join()
            .map_err(|_| Error::internal("path cancellation thread panicked"))?;
        assert_eq!(error.code, ErrorCode::Cancelled, "{procedure:?}");
        assert!(
            elapsed < Duration::from_secs(2),
            "high-degree {procedure:?} cancellation took {elapsed:?}"
        );
    }

    let deadline = Instant::now()
        .checked_add(Duration::from_millis(1))
        .ok_or_else(|| Error::internal("test deadline overflow"))?;
    assert!(
        deadline > Instant::now(),
        "test deadline was not initially live"
    );
    let started = Instant::now();
    let error = metal
        .execute_graph_procedure(
            &ResidentGraphProcedureRequest {
                project: PROJECT,
                layers: LayerMask::OBSERVED,
                procedure: ResidentGraphProcedure::DepthFirst { source_dense: 0 },
                max_output_rows: graph.node_count(),
                deadline: Some(deadline),
            },
            &CancellationToken::new(),
        )
        .expect_err("a live deadline must interrupt the high-degree Metal traversal");
    let elapsed = started.elapsed();
    assert_eq!(error.code, ErrorCode::DeadlineExceeded);
    assert!(
        elapsed < Duration::from_secs(2),
        "high-degree deadline observation took {elapsed:?}"
    );
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_louvain_permits_beneficial_higher_id_moves_deterministically() -> Result<()> {
    let graph = louvain_higher_id_move_graph()?;
    let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
    metal.admit_project(image(&graph)?)?;
    let query = "USE LAYER OBSERVED\n\
        CALL graph.louvain() YIELD node, community\n\
        RETURN id(node) AS id, community ORDER BY id";
    assert_exact_differential(&graph, &metal, query)?;
    let first = rows(&execute(&graph, Some(&metal), query)?);
    let second = rows(&execute(&graph, Some(&metal), query)?);
    assert_eq!(first, second, "Metal Louvain must be repeatable");
    assert_eq!(
        first
            .iter()
            .map(|row| (integer(&row[0]), integer(&row[1])))
            .collect::<Vec<_>>(),
        vec![(100, 0), (101, 1), (102, 0), (103, 1)]
    );
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_louvain_randomized_exact_differential_and_repeatability() -> Result<()> {
    let query = "USE LAYER OBSERVED\n\
        CALL graph.louvain() YIELD node, community\n\
        RETURN id(node) AS id, community ORDER BY id";
    for seed in 1..=24 {
        let graph = randomized_louvain_graph(seed)?;
        let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
        metal.admit_project(image(&graph)?)?;
        assert_exact_differential(&graph, &metal, query)?;
        let first = rows(&execute(&graph, Some(&metal), query)?);
        let second = rows(&execute(&graph, Some(&metal), query)?);
        assert_eq!(
            first, second,
            "seed {seed}: Metal Louvain is not repeatable"
        );
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_louvain_high_degree_power_law_and_dense_fallback_is_exact() -> Result<()> {
    let query = "USE LAYER OBSERVED\n\
        CALL graph.louvain() YIELD node, community\n\
        RETURN id(node) AS id, community ORDER BY id";
    let fixtures = [
        ("power-law-star", high_degree_star_graph(160)?, 1_usize),
        ("dense-planted", planted_louvain_graph(3, 130, 64)?, 3_usize),
    ];
    for (name, graph, expected_communities) in fixtures {
        let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
        metal.admit_project(image(&graph)?)?;
        let cpu_start = Instant::now();
        let cpu = execute(&graph, None, query)?;
        let cpu_elapsed = cpu_start.elapsed().as_secs_f64();
        let metal_start = Instant::now();
        let device = execute(&graph, Some(&metal), query)?;
        let metal_elapsed = metal_start.elapsed().as_secs_f64();
        assert_eq!(device.result.batches, cpu.result.batches, "{name}");
        let repeated = execute(&graph, Some(&metal), query)?;
        assert_eq!(repeated.result.batches, device.result.batches, "{name}");
        let communities = rows(&device).into_iter().map(|row| integer(&row[1])).fold(
            BTreeMap::new(),
            |mut counts, community| {
                *counts.entry(community).or_insert(0_usize) += 1;
                counts
            },
        );
        assert_eq!(communities.len(), expected_communities, "{name}");
        eprintln!(
            "adaptive Louvain {name}: nodes={}, edges={}, communities={}, \
             cpu_elapsed={cpu_elapsed:.6}s, metal_elapsed={metal_elapsed:.6}s, speedup={:.3}x",
            graph.node_count(),
            graph.edge_count(),
            communities.len(),
            cpu_elapsed / metal_elapsed,
        );
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_components_metrics_and_louvain_match_cpu_independently() -> Result<()> {
    let graph = adversarial_graph()?;
    let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
    metal.admit_project(image(&graph)?)?;
    for query in [
        "USE LAYER OBSERVED\n\
         CALL graph.wcc() YIELD node, component RETURN id(node) AS id, component ORDER BY id",
        "USE LAYER OBSERVED\n\
         CALL graph.scc() YIELD node, component RETURN id(node) AS id, component ORDER BY id",
        "USE LAYER OBSERVED\n\
         CALL graph.trianglecount() YIELD triangleCount RETURN triangleCount",
        "USE LAYER OBSERVED\n\
         CALL graph.kcore() YIELD node, core RETURN id(node) AS id, core ORDER BY id",
        "USE LAYER OBSERVED\n\
         CALL graph.louvain() YIELD node, community RETURN id(node) AS id, community ORDER BY id",
    ] {
        assert_exact_differential(&graph, &metal, query)?;
    }
    assert_clustering_differential(
        &graph,
        &metal,
        "USE LAYER OBSERVED\n\
         CALL graph.clusteringcoefficient() YIELD node, coefficient\n\
         RETURN id(node) AS id, coefficient ORDER BY id",
    )
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_empty_graph_degree_and_pagerank_match_cpu() -> Result<()> {
    let graph = GraphStore::default();
    let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
    metal.admit_project(image(&graph)?)?;
    for query in [
        "CALL graph.degree() YIELD node, degree RETURN node, degree",
        "CALL graph.wcc() YIELD node, component RETURN node, component",
        "CALL graph.scc() YIELD node, component RETURN node, component",
        "CALL graph.louvain() YIELD node, community RETURN node, community",
        "CALL graph.pagerank() YIELD node, score RETURN node, score",
        "CALL graph.trianglecount() YIELD triangleCount RETURN triangleCount",
        "CALL graph.clusteringcoefficient() YIELD node, coefficient RETURN node, coefficient",
        "CALL graph.kcore() YIELD node, core RETURN node, core",
    ] {
        assert_exact_differential(&graph, &metal, query)?;
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_isolated_node_dijkstra_is_exact_and_honors_direct_budget() -> Result<()> {
    let mut graph = GraphStore::default();
    let vertex = graph.catalog_mut().intern_label("Vertex")?;
    graph.insert_node(NodeInput {
        id: NodeId(1),
        layer: Layer::Observed,
        revision: 1,
        labels: vec![vertex],
        properties: Vec::new(),
    })?;
    let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
    metal.admit_project(image(&graph)?)?;

    assert_exact_differential(
        &graph,
        &metal,
        "CALL graph.dijkstra(1) YIELD node, cost, predecessor\n\
         RETURN id(node) AS id, cost, id(predecessor) AS predecessor ORDER BY id",
    )?;

    let error = metal
        .execute_graph_procedure(
            &ResidentGraphProcedureRequest {
                project: PROJECT,
                layers: LayerMask::OBSERVED,
                procedure: ResidentGraphProcedure::DijkstraUnit { source_dense: 0 },
                max_output_rows: 0,
                deadline: None,
            },
            &CancellationToken::new(),
        )
        .expect_err("direct isolated-node Dijkstra must honor the result budget");
    assert_eq!(error.code, ErrorCode::ResultBudgetExceeded);
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn benchmark_environment_usize(name: &str, default: usize) -> Result<usize> {
    let Some(value) = std::env::var_os(name) else {
        return Ok(default);
    };
    value
        .to_string_lossy()
        .parse::<usize>()
        .map_err(|error| Error::new(ErrorCode::QueryType, format!("invalid {name}: {error}")))
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn benchmark_environment_f64(name: &str) -> Result<Option<f64>> {
    let Some(value) = std::env::var_os(name) else {
        return Ok(None);
    };
    let value = value
        .to_string_lossy()
        .parse::<f64>()
        .map_err(|error| Error::new(ErrorCode::QueryType, format!("invalid {name}: {error}")))?;
    if !value.is_finite() || value < 0.0 {
        return Err(Error::new(
            ErrorCode::QueryType,
            format!("{name} must be finite and non-negative"),
        ));
    }
    Ok(Some(value))
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn benchmark_count_as_f64(value: usize, label: &str) -> Result<f64> {
    let exact = u32::try_from(value).map_err(|_| {
        Error::new(
            ErrorCode::ResultBudgetExceeded,
            format!("{label} exceeds exact benchmark accounting"),
        )
    })?;
    Ok(f64::from(exact))
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn sparse_ring_graph(node_count: usize, fanout: usize) -> Result<(GraphStore, Csr)> {
    if node_count < 2 || fanout == 0 || fanout >= node_count {
        return Err(Error::new(
            ErrorCode::QueryType,
            "GPU graph benchmark requires nodes >= 2 and 1 <= fanout < nodes",
        ));
    }
    let mut graph = GraphStore::default();
    let vertex = graph.catalog_mut().intern_label("BenchmarkVertex")?;
    let connected = graph
        .catalog_mut()
        .intern_relationship_type("BENCHMARK_EDGE")?;
    for node in 0..node_count {
        let stable = u64::try_from(node)
            .ok()
            .and_then(|node| node.checked_add(1))
            .ok_or_else(|| Error::new(ErrorCode::ResultBudgetExceeded, "node ID overflow"))?;
        graph.insert_node(NodeInput {
            id: NodeId(stable),
            layer: Layer::Observed,
            revision: stable,
            labels: vec![vertex],
            properties: Vec::new(),
        })?;
    }
    let edge_count = node_count.checked_mul(fanout).ok_or_else(|| {
        Error::new(
            ErrorCode::ResultBudgetExceeded,
            "GPU graph benchmark edge count overflow",
        )
    })?;
    let mut triples = Vec::with_capacity(edge_count);
    for source in 0..node_count {
        for offset in 1..=fanout {
            let ordinal = source
                .checked_mul(fanout)
                .and_then(|base| base.checked_add(offset))
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::ResultBudgetExceeded,
                        "GPU graph benchmark edge ordinal overflow",
                    )
                })?;
            let edge_id = u64::try_from(ordinal).map_err(|_| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "GPU graph benchmark edge ID overflow",
                )
            })?;
            let revision = u64::try_from(node_count)
                .ok()
                .and_then(|nodes| nodes.checked_add(edge_id))
                .ok_or_else(|| {
                    Error::new(
                        ErrorCode::ResultBudgetExceeded,
                        "GPU graph benchmark revision overflow",
                    )
                })?;
            graph.insert_edge(EdgeInput {
                id: EdgeId(edge_id),
                source: NodeId(u64::try_from(source).unwrap() + 1),
                target: NodeId(u64::try_from((source + offset) % node_count).unwrap() + 1),
                relationship_type: connected,
                layer: Layer::Observed,
                revision,
                properties: Vec::new(),
            })?;
            triples.push((
                u32::try_from(source).map_err(|_| {
                    Error::new(
                        ErrorCode::ResultBudgetExceeded,
                        "GPU graph benchmark source ordinal overflow",
                    )
                })?,
                u32::try_from((source + offset) % node_count).map_err(|_| {
                    Error::new(
                        ErrorCode::ResultBudgetExceeded,
                        "GPU graph benchmark target ordinal overflow",
                    )
                })?,
                u32::try_from(ordinal - 1).map_err(|_| {
                    Error::new(
                        ErrorCode::ResultBudgetExceeded,
                        "GPU graph benchmark edge ordinal overflow",
                    )
                })?,
            ));
        }
    }
    assert_eq!(graph.node_count(), node_count);
    assert_eq!(graph.edge_count(), edge_count);
    Ok((graph, Csr::build(node_count, &triples)?))
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn planted_louvain_graph(
    community_count: usize,
    nodes_per_community: usize,
    intra_fanout: usize,
) -> Result<GraphStore> {
    if community_count < 3
        || nodes_per_community < 3
        || intra_fanout == 0
        || intra_fanout * 2 >= nodes_per_community
    {
        return Err(Error::new(
            ErrorCode::QueryType,
            "Louvain benchmark requires at least three communities and fanout < community_size/2",
        ));
    }
    let node_count = community_count
        .checked_mul(nodes_per_community)
        .ok_or_else(|| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "Louvain planted node count overflow",
            )
        })?;
    let mut graph = GraphStore::default();
    let vertex = graph.catalog_mut().intern_label("PlantedVertex")?;
    let connected = graph
        .catalog_mut()
        .intern_relationship_type("PLANTED_EDGE")?;
    for node in 0..node_count {
        let stable = u64::try_from(node + 1).map_err(|_| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "Louvain planted node ID overflow",
            )
        })?;
        graph.insert_node(NodeInput {
            id: NodeId(stable),
            layer: Layer::Observed,
            revision: stable,
            labels: vec![vertex],
            properties: Vec::new(),
        })?;
    }
    let mut edge = 1_u64;
    for community in 0..community_count {
        let base = community * nodes_per_community;
        for source in 0..nodes_per_community {
            for offset in 1..=intra_fanout {
                let target = (source + offset) % nodes_per_community;
                graph.insert_edge(EdgeInput {
                    id: EdgeId(edge),
                    source: NodeId(u64::try_from(base + source + 1).unwrap()),
                    target: NodeId(u64::try_from(base + target + 1).unwrap()),
                    relationship_type: connected,
                    layer: Layer::Observed,
                    revision: u64::try_from(node_count).unwrap() + edge,
                    properties: Vec::new(),
                })?;
                edge += 1;
            }
        }
        let next = ((community + 1) % community_count) * nodes_per_community;
        graph.insert_edge(EdgeInput {
            id: EdgeId(edge),
            source: NodeId(u64::try_from(base + 1).unwrap()),
            target: NodeId(u64::try_from(next + 1).unwrap()),
            relationship_type: connected,
            layer: Layer::Observed,
            revision: u64::try_from(node_count).unwrap() + edge,
            properties: Vec::new(),
        })?;
        edge += 1;
    }
    Ok(graph)
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn weighted_re_relaxation_graph(node_count: usize) -> Result<(GraphStore, PropertyId)> {
    if node_count < 2 {
        return Err(Error::new(
            ErrorCode::QueryType,
            "weighted path benchmark requires at least two nodes",
        ));
    }
    let mut graph = GraphStore::default();
    let vertex = graph
        .catalog_mut()
        .intern_label("WeightedBenchmarkVertex")?;
    let connected = graph
        .catalog_mut()
        .intern_relationship_type("WEIGHTED_BENCHMARK_EDGE")?;
    let weight = graph.catalog_mut().intern_property("weight")?;
    for node in 1..=node_count {
        let stable = u64::try_from(node).map_err(|_| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "benchmark node ID overflow",
            )
        })?;
        graph.insert_node(NodeInput {
            id: NodeId(stable),
            layer: Layer::Observed,
            revision: stable,
            labels: vec![vertex],
            properties: Vec::new(),
        })?;
    }
    let mut edge_id = 1_u64;
    for target in 2..=node_count {
        // Every node is discovered immediately with an expensive direct label. The unit chain
        // then improves exactly one additional depth per synchronous round, stressing the path
        // engine's sparse active-frontier behavior instead of an easy one-pass DAG.
        for (source, cost) in [(1_usize, 1_000_000.0_f64), (target - 1, 1.0)] {
            graph.insert_edge(EdgeInput {
                id: EdgeId(edge_id),
                source: NodeId(u64::try_from(source).unwrap()),
                target: NodeId(u64::try_from(target).unwrap()),
                relationship_type: connected,
                layer: Layer::Observed,
                revision: u64::try_from(node_count).unwrap() + edge_id,
                properties: vec![(
                    weight,
                    ScalarValue::Float(ordered_float::OrderedFloat(cost)),
                )],
            })?;
            edge_id = edge_id.checked_add(1).ok_or_else(|| {
                Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "weighted benchmark edge ID overflow",
                )
            })?;
        }
    }
    Ok((graph, weight))
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
fn directed_chain_graph(node_count: usize) -> Result<GraphStore> {
    if node_count < 2 {
        return Err(Error::new(
            ErrorCode::QueryType,
            "directed chain requires at least two nodes",
        ));
    }
    let mut graph = GraphStore::default();
    let vertex = graph.catalog_mut().intern_label("PathBenchmarkVertex")?;
    let connected = graph
        .catalog_mut()
        .intern_relationship_type("PATH_BENCHMARK_EDGE")?;
    for node in 0..node_count {
        let id = u64::try_from(node + 1).map_err(|_| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "path benchmark node ID overflow",
            )
        })?;
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: id,
            labels: vec![vertex],
            properties: Vec::new(),
        })?;
    }
    for source in 0..node_count - 1 {
        let edge = u64::try_from(source + 1).unwrap();
        graph.insert_edge(EdgeInput {
            id: EdgeId(edge),
            source: NodeId(u64::try_from(source + 1).unwrap()),
            target: NodeId(u64::try_from(source + 2).unwrap()),
            relationship_type: connected,
            layer: Layer::Observed,
            revision: u64::try_from(node_count).unwrap() + edge,
            properties: Vec::new(),
        })?;
    }
    Ok(graph)
}

/// Manual high-diameter gate for exact ordered DFS and both forward/reverse shortest-path BFS.
#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "manual real-Metal long-chain path throughput gate"]
fn real_metal_long_chain_path_throughput() -> Result<()> {
    let node_count = benchmark_environment_usize("IRONGRAPH_GPU_PATH_NODES", 5_000)?;
    let iterations = benchmark_environment_usize("IRONGRAPH_GPU_PATH_ITERATIONS", 3)?;
    if iterations == 0 {
        return Err(Error::new(
            ErrorCode::QueryType,
            "IRONGRAPH_GPU_PATH_ITERATIONS must be positive",
        ));
    }
    let graph = directed_chain_graph(node_count)?;
    let edge_count = graph.edge_count();
    let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
    metal.admit_project(image(&graph)?)?;
    let dfs_query = "CALL graph.dfs(1) YIELD node, order \
                     RETURN id(node) AS id, order ORDER BY order";
    let shortest_query =
        format!("CALL graph.shortestpath(1, {node_count}) YIELD path, cost RETURN path, cost");
    let verify_dfs = |output: &ExecutionOutput| -> Result<()> {
        let result_rows = rows(output);
        assert_eq!(result_rows.len(), node_count);
        assert_eq!(integer(&result_rows[0][0]), 1);
        assert_eq!(integer(&result_rows[0][1]), 0);
        assert_eq!(
            integer(&result_rows[node_count - 1][0]),
            i64::try_from(node_count).unwrap()
        );
        assert_eq!(
            integer(&result_rows[node_count - 1][1]),
            i64::try_from(node_count - 1).unwrap()
        );
        Ok(())
    };
    let verify_shortest = |output: &ExecutionOutput| -> Result<()> {
        let result_rows = rows(output);
        assert_eq!(result_rows.len(), 1);
        assert_eq!(
            usize::try_from(integer(&result_rows[0][1])).expect("path cost fits usize"),
            edge_count
        );
        Ok(())
    };
    let cpu_dfs_warmup = execute(&graph, None, dfs_query)?;
    let metal_dfs_warmup = execute(&graph, Some(&metal), dfs_query)?;
    assert_eq!(metal_dfs_warmup.result.schema, cpu_dfs_warmup.result.schema);
    assert_eq!(
        metal_dfs_warmup.result.batches,
        cpu_dfs_warmup.result.batches
    );
    verify_dfs(&metal_dfs_warmup)?;
    let cpu_shortest_warmup = execute(&graph, None, &shortest_query)?;
    let metal_shortest_warmup = execute(&graph, Some(&metal), &shortest_query)?;
    assert_eq!(
        metal_shortest_warmup.result.schema,
        cpu_shortest_warmup.result.schema
    );
    assert_eq!(
        metal_shortest_warmup.result.batches,
        cpu_shortest_warmup.result.batches
    );
    verify_shortest(&metal_shortest_warmup)?;

    let cpu_dfs_started = Instant::now();
    for _ in 0..iterations {
        let output = execute(&graph, None, dfs_query)?;
        assert_eq!(output.result.batches, cpu_dfs_warmup.result.batches);
    }
    let cpu_dfs_elapsed = cpu_dfs_started.elapsed().as_secs_f64();
    let cpu_shortest_started = Instant::now();
    for _ in 0..iterations {
        let output = execute(&graph, None, &shortest_query)?;
        assert_eq!(output.result.batches, cpu_shortest_warmup.result.batches);
    }
    let cpu_shortest_elapsed = cpu_shortest_started.elapsed().as_secs_f64();

    let dfs_started = Instant::now();
    for _ in 0..iterations {
        let output = execute(&graph, Some(&metal), dfs_query)?;
        assert_eq!(output.result.batches, cpu_dfs_warmup.result.batches);
    }
    let dfs_elapsed = dfs_started.elapsed().as_secs_f64();
    let shortest_started = Instant::now();
    for _ in 0..iterations {
        let output = execute(&graph, Some(&metal), &shortest_query)?;
        assert_eq!(output.result.batches, cpu_shortest_warmup.result.batches);
    }
    let shortest_elapsed = shortest_started.elapsed().as_secs_f64();
    let iteration_count = benchmark_count_as_f64(iterations, "path benchmark iteration count")?;
    let logical_edge_count = benchmark_count_as_f64(edge_count, "path benchmark edge count")?;
    eprintln!(
        "same-boundary long-chain paths: nodes={node_count}, edges={edge_count}, \
         iterations={iterations}, Metal-dfs_ms={:.3}, Metal-shortest_ms={:.3}, \
         CPU-dfs_ms={:.3}, CPU-shortest_ms={:.3}, dfs_edges/s={:.3}, \
         shortest_edges/s={:.3}, dfs-speedup={:.2}x, shortest-speedup={:.2}x",
        dfs_elapsed * 1_000.0 / iteration_count,
        shortest_elapsed * 1_000.0 / iteration_count,
        cpu_dfs_elapsed * 1_000.0 / iteration_count,
        cpu_shortest_elapsed * 1_000.0 / iteration_count,
        logical_edge_count * iteration_count / dfs_elapsed,
        logical_edge_count * iteration_count / shortest_elapsed,
        cpu_dfs_elapsed / dfs_elapsed,
        cpu_shortest_elapsed / shortest_elapsed,
    );
    Ok(())
}

/// Manual regression for the exact weighted frontier on a high-diameter graph with repeated
/// distance improvements. This shape previously collapses a whole-graph Bellman-Ford pull toward
/// O(VE); the production edge/node-tiled target pull should remain proportional to touched rows.
#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "manual real-Metal exact weighted-path throughput gate"]
fn real_metal_weighted_high_diameter_throughput() -> Result<()> {
    let node_count = benchmark_environment_usize("IRONGRAPH_GPU_WEIGHTED_NODES", 5_000)?;
    let iterations = benchmark_environment_usize("IRONGRAPH_GPU_WEIGHTED_ITERATIONS", 3)?;
    let memory_bytes =
        benchmark_environment_usize("IRONGRAPH_GPU_BENCH_MEMORY_BYTES", MEMORY_LIMIT_BYTES)?;
    if iterations == 0 {
        return Err(Error::new(
            ErrorCode::QueryType,
            "IRONGRAPH_GPU_WEIGHTED_ITERATIONS must be positive",
        ));
    }
    let (graph, _weight_property) = weighted_re_relaxation_graph(node_count)?;
    let edge_count = graph.edge_count();
    let mut metal = MetalBackend::new(0, memory_bytes, RESERVED_BYTES)?;
    metal.admit_project(image(&graph)?)?;
    let query = "CALL graph.dijkstra(1, 'weight') YIELD node, cost, predecessor \
                 RETURN id(node) AS id, cost, id(predecessor) AS predecessor ORDER BY id";
    let expected_last_cost =
        benchmark_count_as_f64(node_count - 1, "weighted benchmark path cost")?;
    let verify = |output: &ExecutionOutput| -> Result<()> {
        let result_rows = rows(output);
        assert_eq!(result_rows.len(), node_count);
        assert_eq!(integer(&result_rows[0][0]), 1);
        assert_eq!(
            float(&result_rows[0][1]).to_bits(),
            0.0_f64.to_bits(),
            "weighted source cost must be exact positive zero"
        );
        assert_eq!(
            float(&result_rows[node_count - 1][1]).to_bits(),
            expected_last_cost.to_bits(),
            "weighted terminal cost must be bit-exact"
        );
        Ok(())
    };
    let cpu_warmup = execute(&graph, None, query)?;
    let metal_warmup = execute(&graph, Some(&metal), query)?;
    assert_eq!(metal_warmup.result.schema, cpu_warmup.result.schema);
    assert_eq!(metal_warmup.result.batches, cpu_warmup.result.batches);
    verify(&metal_warmup)?;
    let cpu_start = Instant::now();
    for _ in 0..iterations {
        let output = execute(&graph, None, query)?;
        assert_eq!(output.result.batches, cpu_warmup.result.batches);
    }
    let cpu_elapsed = cpu_start.elapsed().as_secs_f64();
    let start = Instant::now();
    for _ in 0..iterations {
        let output = execute(&graph, Some(&metal), query)?;
        assert_eq!(output.result.batches, cpu_warmup.result.batches);
    }
    let elapsed = start.elapsed().as_secs_f64();
    let iteration_count = benchmark_count_as_f64(iterations, "weighted benchmark iteration count")?;
    let logical_edge_count = benchmark_count_as_f64(edge_count, "weighted benchmark edge count")?;
    let traversals_per_second = iteration_count / elapsed;
    let logical_edges_per_second = logical_edge_count * iteration_count / elapsed;
    eprintln!(
        "same-boundary exact weighted high-diameter benchmark: nodes={node_count}, \
         edges={edge_count}, iterations={iterations}, elapsed={elapsed:.6}s, \
         CPU-elapsed={cpu_elapsed:.6}s, traversals/s={traversals_per_second:.3}, \
         logical edges/s={logical_edges_per_second:.3}, speedup={:.2}x",
        cpu_elapsed / elapsed,
    );
    if let Some(minimum) =
        benchmark_environment_f64("IRONGRAPH_GPU_BENCH_MIN_WEIGHTED_TRAVERSALS_PER_SECOND")?
    {
        assert!(
            traversals_per_second >= minimum,
            "weighted traversals/s {traversals_per_second:.3} below required {minimum:.3}"
        );
    }
    Ok(())
}

/// Manual same-boundary CPU/Metal throughput and quality gate for deterministic multilevel
/// Louvain. Both measurements execute the same public Cypher query and include undirected
/// deduplication, every exact-gain local pass, deterministic proposal matching, coarsening, result
/// transfer, and result encoding. Set `IRONGRAPH_GPU_BENCH_MIN_LOUVAIN_EDGES_PER_SECOND` or
/// `IRONGRAPH_GPU_BENCH_MIN_LOUVAIN_SPEEDUP` to enforce machine-specific gates.
#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "manual real-Metal Louvain throughput gate"]
#[allow(clippy::too_many_lines)]
fn real_metal_louvain_throughput() -> Result<()> {
    let community_count = benchmark_environment_usize("IRONGRAPH_GPU_LOUVAIN_COMMUNITIES", 128)?;
    let nodes_per_community =
        benchmark_environment_usize("IRONGRAPH_GPU_LOUVAIN_COMMUNITY_SIZE", 64)?;
    let fanout = benchmark_environment_usize("IRONGRAPH_GPU_LOUVAIN_FANOUT", 24)?;
    let iterations = benchmark_environment_usize("IRONGRAPH_GPU_LOUVAIN_ITERATIONS", 2)?;
    let memory_bytes =
        benchmark_environment_usize("IRONGRAPH_GPU_BENCH_MEMORY_BYTES", MEMORY_LIMIT_BYTES)?;
    if iterations == 0 {
        return Err(Error::new(
            ErrorCode::QueryType,
            "IRONGRAPH_GPU_LOUVAIN_ITERATIONS must be positive",
        ));
    }
    let node_count = community_count
        .checked_mul(nodes_per_community)
        .ok_or_else(|| {
            Error::new(
                ErrorCode::ResultBudgetExceeded,
                "Louvain benchmark node count overflow",
            )
        })?;
    let graph = planted_louvain_graph(community_count, nodes_per_community, fanout)?;
    let edge_count = graph.edge_count();
    let mut metal = MetalBackend::new(0, memory_bytes, RESERVED_BYTES)?;
    metal.admit_project(image(&graph)?)?;
    let query = "USE LAYER OBSERVED\n\
        CALL graph.louvain() YIELD node, community\n\
        RETURN id(node) AS id, community ORDER BY id";
    let verify = |output: &ExecutionOutput| -> Result<()> {
        let output_rows = rows(output);
        assert_eq!(output_rows.len(), node_count);
        for (node, row) in output_rows.iter().enumerate() {
            assert_eq!(integer(&row[0]), i64::try_from(node + 1).unwrap());
            assert_eq!(
                integer(&row[1]),
                i64::try_from(node / nodes_per_community).unwrap(),
                "Louvain did not recover the planted partition at node {node}"
            );
        }
        Ok(())
    };

    let cpu_warmup = execute(&graph, None, query)?;
    verify(&cpu_warmup)?;
    let metal_warmup = execute(&graph, Some(&metal), query)?;
    verify(&metal_warmup)?;
    assert_eq!(
        metal_warmup.result.batches, cpu_warmup.result.batches,
        "planted Louvain Metal/CPU warmup differential"
    );

    let cpu_start = Instant::now();
    for _ in 0..iterations {
        let output = execute(&graph, None, query)?;
        verify(&output)?;
        assert_eq!(
            output.result.batches, cpu_warmup.result.batches,
            "CPU Louvain is not deterministic"
        );
    }
    let cpu_elapsed = cpu_start.elapsed().as_secs_f64();

    let metal_start = Instant::now();
    for _ in 0..iterations {
        let output = execute(&graph, Some(&metal), query)?;
        verify(&output)?;
        assert_eq!(
            output.result.batches, metal_warmup.result.batches,
            "Metal Louvain is not deterministic"
        );
    }
    let metal_elapsed = metal_start.elapsed().as_secs_f64();
    let iteration_count = benchmark_count_as_f64(iterations, "Louvain benchmark iteration count")?;
    let logical_edge_count = benchmark_count_as_f64(edge_count, "Louvain benchmark edge count")?;
    let cpu_edges_per_second = logical_edge_count * iteration_count / cpu_elapsed;
    let metal_traversals_per_second = iteration_count / metal_elapsed;
    let metal_edges_per_second = logical_edge_count * iteration_count / metal_elapsed;
    let speedup = cpu_elapsed / metal_elapsed;
    let verified_runs_per_backend = iterations.saturating_add(1);
    let internal_edge_count = nodes_per_community.checked_mul(fanout).ok_or_else(|| {
        Error::new(
            ErrorCode::ResultBudgetExceeded,
            "Louvain benchmark internal edge count overflow",
        )
    })?;
    let internal_edges =
        benchmark_count_as_f64(internal_edge_count, "Louvain benchmark internal edge count")?;
    let community_count_f64 =
        benchmark_count_as_f64(community_count, "Louvain benchmark community count")?;
    let modularity = internal_edges / (internal_edges + 1.0) - 1.0 / community_count_f64;
    eprintln!(
        "same-boundary planted Louvain benchmark: communities={community_count}, \
         nodes={node_count}, edges={edge_count}, modularity={modularity:.9}, \
         recovered_communities={community_count}, \
         verified_runs_per_backend={verified_runs_per_backend}, \
         cpu_elapsed={cpu_elapsed:.6}s, cpu_stored_edges/s={cpu_edges_per_second:.3}, \
         metal_elapsed={metal_elapsed:.6}s, metal_traversals/s={metal_traversals_per_second:.3}, \
         metal_stored_edges/s={metal_edges_per_second:.3}, metal_speedup={speedup:.3}x"
    );
    if let Some(minimum) =
        benchmark_environment_f64("IRONGRAPH_GPU_BENCH_MIN_LOUVAIN_EDGES_PER_SECOND")?
    {
        assert!(
            metal_edges_per_second >= minimum,
            "Louvain throughput {metal_edges_per_second:.3} edges/s is below {minimum:.3}"
        );
    }
    if let Some(minimum) = benchmark_environment_f64("IRONGRAPH_GPU_BENCH_MIN_LOUVAIN_SPEEDUP")? {
        assert!(
            speedup >= minimum,
            "Louvain Metal speedup {speedup:.3}x is below {minimum:.3}x"
        );
    }
    Ok(())
}

/// Manual end-to-end throughput gate for the resident sparse paths. It includes backend launch,
/// traversal/iteration, synchronization, and bounded final result transfer. The `PageRank` request
/// is pinned to exactly one iteration, making its reported rate one real sparse edge update per
/// stored edge rather than an estimate of an unobservable convergence count.
///
/// Run with:
/// `cargo test --test gpu_graph_algorithm_contract real_metal_sparse_algorithm_throughput -- --ignored --nocapture`
///
/// Shape and optional gates are controlled by `IRONGRAPH_GPU_BENCH_NODES`,
/// `IRONGRAPH_GPU_BENCH_FANOUT`, `IRONGRAPH_GPU_BENCH_ITERATIONS`,
/// `IRONGRAPH_GPU_BENCH_MEMORY_BYTES`, `IRONGRAPH_GPU_BENCH_MIN_BFS_EDGES_PER_SECOND`, and
/// `IRONGRAPH_GPU_BENCH_MIN_PAGERANK_EDGES_PER_SECOND`.
#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "manual real-Metal throughput gate; size and thresholds are environment-controlled"]
#[allow(clippy::too_many_lines)]
fn real_metal_sparse_algorithm_throughput() -> Result<()> {
    let node_count = benchmark_environment_usize("IRONGRAPH_GPU_BENCH_NODES", 20_000)?;
    let fanout = benchmark_environment_usize("IRONGRAPH_GPU_BENCH_FANOUT", 8)?;
    let iterations = benchmark_environment_usize("IRONGRAPH_GPU_BENCH_ITERATIONS", 5)?;
    let memory_bytes =
        benchmark_environment_usize("IRONGRAPH_GPU_BENCH_MEMORY_BYTES", MEMORY_LIMIT_BYTES)?;
    if iterations == 0 {
        return Err(Error::new(
            ErrorCode::QueryType,
            "IRONGRAPH_GPU_BENCH_ITERATIONS must be positive",
        ));
    }

    let (graph, cpu_csr) = sparse_ring_graph(node_count, fanout)?;
    let edge_count = graph.edge_count();
    let mut metal = MetalBackend::new(0, memory_bytes, RESERVED_BYTES)?;
    metal.admit_project(image(&graph)?)?;
    let bfs_query = "CALL graph.bfs(1) YIELD node, distance \
                     RETURN id(node) AS id, distance ORDER BY id";
    let pagerank_query = "CALL graph.pagerank(0.85, 1.0e-30, 1) YIELD node, score \
                          RETURN id(node) AS id, score ORDER BY id";
    let max_result_rows = node_count.max(1);

    // One unmeasured execution forces lazy pipeline/kernel setup and validates the complete
    // public Cypher output before timing begins.
    let cpu_bfs_warmup = execute_with_row_budget(&graph, None, bfs_query, max_result_rows)?;
    let metal_bfs_warmup =
        execute_with_row_budget(&graph, Some(&metal), bfs_query, max_result_rows)?;
    assert_eq!(metal_bfs_warmup.result.schema, cpu_bfs_warmup.result.schema);
    assert_eq!(
        metal_bfs_warmup.result.batches,
        cpu_bfs_warmup.result.batches
    );
    let bfs_rows = rows(&metal_bfs_warmup);
    assert_eq!(bfs_rows.len(), node_count);
    assert_eq!(integer(&bfs_rows[0][0]), 1);
    assert_eq!(integer(&bfs_rows[0][1]), 0);

    let cpu_pagerank_warmup =
        execute_with_row_budget(&graph, None, pagerank_query, max_result_rows)?;
    let metal_pagerank_warmup =
        execute_with_row_budget(&graph, Some(&metal), pagerank_query, max_result_rows)?;
    assert_eq!(
        metal_pagerank_warmup.result.schema,
        cpu_pagerank_warmup.result.schema
    );
    assert_eq!(
        metal_pagerank_warmup.result.batches,
        cpu_pagerank_warmup.result.batches
    );
    let pagerank_rows = rows(&metal_pagerank_warmup);
    assert_eq!(pagerank_rows.len(), node_count);
    let score_sum = pagerank_rows.iter().map(|row| float(&row[1])).sum::<f64>();
    assert!(
        (score_sum - 1.0).abs() <= 2.0e-3,
        "PageRank benchmark score sum={score_sum}"
    );

    // Also time the resident backend boundary. Comparing this with the public Cypher timings
    // distinguishes graph execution plus bounded publication from result encoding and ORDER BY.
    let direct_cancellation = CancellationToken::new();
    let direct_bfs_request = ResidentGraphProcedureRequest {
        project: PROJECT,
        layers: LayerMask::OBSERVED,
        procedure: ResidentGraphProcedure::BreadthFirst { source_dense: 0 },
        max_output_rows: max_result_rows,
        deadline: None,
    };
    let direct_pagerank_request = ResidentGraphProcedureRequest {
        project: PROJECT,
        layers: LayerMask::OBSERVED,
        procedure: ResidentGraphProcedure::PageRank {
            damping: 0.85,
            tolerance: 1.0e-30,
            max_iterations: 1,
        },
        max_output_rows: max_result_rows,
        deadline: None,
    };
    let ResidentGraphProcedureResult::BreadthFirst { node_rows, .. } =
        metal.execute_graph_procedure(&direct_bfs_request, &direct_cancellation)?
    else {
        return Err(Error::internal(
            "direct Metal BFS benchmark returned the wrong shape",
        ));
    };
    assert_eq!(node_rows.len(), node_count);
    let ResidentGraphProcedureResult::PageRank { node_rows, .. } =
        metal.execute_graph_procedure(&direct_pagerank_request, &direct_cancellation)?
    else {
        return Err(Error::internal(
            "direct Metal PageRank benchmark returned the wrong shape",
        ));
    };
    assert_eq!(node_rows.len(), node_count);
    assert_eq!(bfs(&cpu_csr, 0)?.len(), node_count);
    let direct_cpu_pagerank_config = PageRankConfig {
        damping: 0.85,
        tolerance: 1.0e-30,
        max_iterations: 1,
    };
    assert_eq!(
        page_rank(&cpu_csr, direct_cpu_pagerank_config)?.len(),
        node_count
    );

    let cpu_bfs_start = Instant::now();
    for _ in 0..iterations {
        let output = execute_with_row_budget(&graph, None, bfs_query, max_result_rows)?;
        assert_eq!(output.result.batches, cpu_bfs_warmup.result.batches);
    }
    let cpu_bfs_elapsed = cpu_bfs_start.elapsed();

    let cpu_pagerank_start = Instant::now();
    for _ in 0..iterations {
        let output = execute_with_row_budget(&graph, None, pagerank_query, max_result_rows)?;
        assert_eq!(output.result.batches, cpu_pagerank_warmup.result.batches);
    }
    let cpu_pagerank_elapsed = cpu_pagerank_start.elapsed();

    let direct_cpu_bfs_start = Instant::now();
    for _ in 0..iterations {
        assert_eq!(bfs(&cpu_csr, 0)?.len(), node_count);
    }
    let direct_cpu_bfs_elapsed = direct_cpu_bfs_start.elapsed();

    let direct_cpu_pagerank_start = Instant::now();
    for _ in 0..iterations {
        assert_eq!(
            page_rank(&cpu_csr, direct_cpu_pagerank_config)?.len(),
            node_count
        );
    }
    let direct_cpu_pagerank_elapsed = direct_cpu_pagerank_start.elapsed();

    let bfs_start = Instant::now();
    for _ in 0..iterations {
        let output = execute_with_row_budget(&graph, Some(&metal), bfs_query, max_result_rows)?;
        assert_eq!(output.result.batches, cpu_bfs_warmup.result.batches);
    }
    let bfs_elapsed = bfs_start.elapsed();

    let pagerank_start = Instant::now();
    for _ in 0..iterations {
        let output =
            execute_with_row_budget(&graph, Some(&metal), pagerank_query, max_result_rows)?;
        assert_eq!(output.result.batches, cpu_pagerank_warmup.result.batches);
    }
    let pagerank_elapsed = pagerank_start.elapsed();

    let direct_bfs_start = Instant::now();
    for _ in 0..iterations {
        let output = metal.execute_graph_procedure(&direct_bfs_request, &direct_cancellation)?;
        let ResidentGraphProcedureResult::BreadthFirst { node_rows, .. } = output else {
            return Err(Error::internal(
                "direct Metal BFS benchmark returned the wrong shape",
            ));
        };
        assert_eq!(node_rows.len(), node_count);
    }
    let direct_bfs_elapsed = direct_bfs_start.elapsed();

    let direct_pagerank_start = Instant::now();
    for _ in 0..iterations {
        let output =
            metal.execute_graph_procedure(&direct_pagerank_request, &direct_cancellation)?;
        let ResidentGraphProcedureResult::PageRank { node_rows, .. } = output else {
            return Err(Error::internal(
                "direct Metal PageRank benchmark returned the wrong shape",
            ));
        };
        assert_eq!(node_rows.len(), node_count);
    }
    let direct_pagerank_elapsed = direct_pagerank_start.elapsed();

    let processed_edges = f64::from(u32::try_from(edge_count).map_err(|_| {
        Error::new(
            ErrorCode::ResultBudgetExceeded,
            "GPU graph benchmark edge count exceeds exact throughput accounting",
        )
    })?) * f64::from(u32::try_from(iterations).map_err(|_| {
        Error::new(
            ErrorCode::ResultBudgetExceeded,
            "GPU graph benchmark iteration count exceeds exact throughput accounting",
        )
    })?);
    let bfs_edges_per_second = processed_edges / bfs_elapsed.as_secs_f64();
    let cpu_bfs_edges_per_second = processed_edges / cpu_bfs_elapsed.as_secs_f64();
    let pagerank_edges_per_second = processed_edges / pagerank_elapsed.as_secs_f64();
    let cpu_pagerank_edges_per_second = processed_edges / cpu_pagerank_elapsed.as_secs_f64();
    let direct_bfs_edges_per_second = processed_edges / direct_bfs_elapsed.as_secs_f64();
    let direct_pagerank_edges_per_second = processed_edges / direct_pagerank_elapsed.as_secs_f64();
    let direct_cpu_bfs_edges_per_second = processed_edges / direct_cpu_bfs_elapsed.as_secs_f64();
    let direct_cpu_pagerank_edges_per_second =
        processed_edges / direct_cpu_pagerank_elapsed.as_secs_f64();
    eprintln!(
        "same-boundary sparse graph benchmark: nodes={node_count}, edges={edge_count}, \
         fanout={fanout}, iterations={iterations}, CPU-bfs={cpu_bfs_edges_per_second:.0} edges/s \
         ({cpu_bfs_elapsed:?}), Metal-bfs={bfs_edges_per_second:.0} edges/s ({bfs_elapsed:?}), \
         bfs-speedup={:.2}x, resident-Metal-bfs={direct_bfs_edges_per_second:.0} edges/s \
         ({direct_bfs_elapsed:?}), direct-CPU-bfs={direct_cpu_bfs_edges_per_second:.0} edges/s \
         ({direct_cpu_bfs_elapsed:?}), CPU-pagerank-one-iteration=\
         {cpu_pagerank_edges_per_second:.0} \
         edges/s ({cpu_pagerank_elapsed:?}), Metal-pagerank-one-iteration=\
         {pagerank_edges_per_second:.0} edges/s ({pagerank_elapsed:?}), pagerank-speedup={:.2}x, \
         resident-Metal-pagerank={direct_pagerank_edges_per_second:.0} edges/s \
         ({direct_pagerank_elapsed:?}), direct-CPU-pagerank=\
         {direct_cpu_pagerank_edges_per_second:.0} edges/s ({direct_cpu_pagerank_elapsed:?})",
        cpu_bfs_elapsed.as_secs_f64() / bfs_elapsed.as_secs_f64(),
        cpu_pagerank_elapsed.as_secs_f64() / pagerank_elapsed.as_secs_f64(),
    );

    if let Some(minimum) =
        benchmark_environment_f64("IRONGRAPH_GPU_BENCH_MIN_BFS_EDGES_PER_SECOND")?
    {
        assert!(
            bfs_edges_per_second >= minimum,
            "BFS throughput {bfs_edges_per_second:.0} edges/s is below configured minimum {minimum:.0}"
        );
    }
    if let Some(minimum) =
        benchmark_environment_f64("IRONGRAPH_GPU_BENCH_MIN_PAGERANK_EDGES_PER_SECOND")?
    {
        assert!(
            pagerank_edges_per_second >= minimum,
            "PageRank throughput {pagerank_edges_per_second:.0} edges/s is below configured minimum {minimum:.0}"
        );
    }
    Ok(())
}
