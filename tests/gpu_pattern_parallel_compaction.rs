// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]
#![cfg(all(feature = "accelerator", target_os = "macos"))]

use std::sync::{Mutex, MutexGuard};

use irongraph::{
    Bookmark, EdgeId, ErrorCode, Layer, NodeId, ProjectId, Result,
    gpu::{
        BackendKind, CpuBackend, ExecutionBackend, MetalBackend, ResidentDeviceCompletion,
        ResidentDirection, ResidentExecutionId, ResidentExecutionObligation,
        ResidentObligationKind, ResidentObligationScope, ResidentPatternBooleanInstruction,
        ResidentPatternPairPredicateLeaf, ResidentPatternPairPredicateLength,
        ResidentPatternPairPredicateProgram, ResidentPatternPairPredicateRequest,
        ResidentPatternPairVisibleNodeScans, ResidentPatternPredicateInput,
        ResidentPatternPredicateLeaf, ResidentPatternPredicateLength,
        ResidentPatternPredicateProgram, ResidentPatternPredicateRequest,
        ResidentPatternRelationshipTypes, ResidentProjectImage,
    },
    graph::{EdgeInput, GraphStore, IndexCatalog, LayerMask, NodeInput, TemporalStore},
    types::RelationshipTypeId,
};
use tokio_util::sync::CancellationToken;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());
const MEMORY_LIMIT_BYTES: usize = 512 * 1024 * 1024;
const RESERVED_BYTES: usize = 64 * 1024 * 1024;
const TILE_ROWS: usize = 1_024;

fn metal_test_guard() -> MutexGuard<'static, ()> {
    static METAL_TEST: Mutex<()> = Mutex::new(());
    match METAL_TEST.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

fn obligation(
    id: u64,
    kind: ResidentObligationKind,
    scope: ResidentObligationScope,
) -> ResidentExecutionObligation {
    ResidentExecutionObligation { id, kind, scope }
}

fn image(graph: &GraphStore, bookmark: Bookmark) -> Result<ResidentProjectImage> {
    ResidentProjectImage::build(
        PROJECT,
        bookmark,
        graph,
        &TemporalStore::default(),
        &IndexCatalog::default(),
    )
}

fn admitted_backends(graph: &GraphStore, bookmark: Bookmark) -> Result<(CpuBackend, MetalBackend)> {
    let image = image(graph, bookmark)?;
    let mut cpu = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
    let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
    cpu.admit_project(image.clone())?;
    metal.admit_project(image)?;
    Ok((cpu, metal))
}

fn one_request(
    graph: &GraphStore,
    bookmark: Bookmark,
    node_slots: usize,
    seed: u64,
    relationship_types: ResidentPatternRelationshipTypes,
    output_capacity: usize,
) -> ResidentPatternPredicateRequest {
    ResidentPatternPredicateRequest {
        project: PROJECT,
        expected_bookmark: bookmark,
        expected_graph_revision: graph.revision(),
        expected_layout_version: graph.layout_version(),
        layers: LayerMask::OBSERVED,
        input: ResidentPatternPredicateInput::VisibleNodeScan {
            node_slots,
            obligation: obligation(
                seed * 10 + 1,
                ResidentObligationKind::PatternScan,
                ResidentObligationScope::PatternScan,
            ),
        },
        program: ResidentPatternPredicateProgram {
            execution: ResidentExecutionId {
                high: 0x5041_5241_4c4c_454c,
                low: seed,
            },
            leaves: vec![ResidentPatternPredicateLeaf {
                direction: ResidentDirection::Outgoing,
                relationship_types,
                length: ResidentPatternPredicateLength::One,
                property_equality: None,
                minimum_witnesses: 1,
                maximum_witnesses: u32::MAX,
                obligation: obligation(
                    seed * 10 + 2,
                    ResidentObligationKind::PatternTraversal,
                    ResidentObligationScope::PatternLeaf(0),
                ),
            }],
            instructions: vec![ResidentPatternBooleanInstruction::Leaf(0)],
            final_obligation: obligation(
                seed * 10 + 3,
                ResidentObligationKind::PatternFilter,
                ResidentObligationScope::PatternFinal,
            ),
        },
        max_output_rows: output_capacity,
    }
}

fn pair_request(
    graph: &GraphStore,
    bookmark: Bookmark,
    node_slots: usize,
    seed: u64,
    relationship_types: ResidentPatternRelationshipTypes,
    output_capacity: usize,
) -> ResidentPatternPairPredicateRequest {
    ResidentPatternPairPredicateRequest {
        project: PROJECT,
        expected_bookmark: bookmark,
        expected_graph_revision: graph.revision(),
        expected_layout_version: graph.layout_version(),
        layers: LayerMask::OBSERVED,
        input: ResidentPatternPairVisibleNodeScans {
            node_slots,
            n_obligation: obligation(
                seed * 10 + 1,
                ResidentObligationKind::PatternScan,
                ResidentObligationScope::PatternScanN,
            ),
            m_obligation: obligation(
                seed * 10 + 2,
                ResidentObligationKind::PatternScan,
                ResidentObligationScope::PatternScanM,
            ),
            cartesian_obligation: obligation(
                seed * 10 + 3,
                ResidentObligationKind::PatternCartesian,
                ResidentObligationScope::PatternCartesian,
            ),
        },
        program: ResidentPatternPairPredicateProgram {
            execution: ResidentExecutionId {
                high: 0x5041_4952_5343_414e,
                low: seed,
            },
            leaf: ResidentPatternPairPredicateLeaf {
                direction: ResidentDirection::Outgoing,
                relationship_types,
                length: ResidentPatternPairPredicateLength::One,
                obligation: obligation(
                    seed * 10 + 4,
                    ResidentObligationKind::PatternTraversal,
                    ResidentObligationScope::PatternLeaf(0),
                ),
            },
            final_obligation: obligation(
                seed * 10 + 5,
                ResidentObligationKind::PatternFilter,
                ResidentObligationScope::PatternFinal,
            ),
        },
        max_output_pairs: output_capacity,
    }
}

fn insert_nodes(graph: &mut GraphStore, count: usize) -> Result<()> {
    for row in 0..count {
        let id = u64::try_from(row + 1).map_err(|_| irongraph::Error::internal("node ID"))?;
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: id,
            labels: Vec::new(),
            properties: Vec::new(),
        })?;
    }
    Ok(())
}

#[test]
fn parallel_compaction_logical_domains_cross_physical_chunk_boundaries() -> Result<()> {
    let graph = GraphStore::default();
    let bookmark = Bookmark { term: 31, index: 1 };
    let above_former_fixed_boundary = (1_usize << 20) + 1;
    let one = one_request(
        &graph,
        bookmark,
        above_former_fixed_boundary,
        1,
        ResidentPatternRelationshipTypes::Never,
        0,
    );
    one.validate()?;
    assert_eq!(one.input.row_count().div_ceil(TILE_ROWS), 1_025);

    let pair = pair_request(
        &graph,
        bookmark,
        1_025,
        2,
        ResidentPatternRelationshipTypes::Never,
        0,
    );
    pair.validate()?;
    assert_eq!(pair.logical_pair_domain()?, 1_050_625);
    assert_eq!(
        pair.logical_pair_domain()?.div_ceil(TILE_ROWS as u64),
        1_027
    );
    Ok(())
}

#[test]
fn cpu_pair_execution_consumes_every_chunk_past_one_million_positions() -> Result<()> {
    const NODE_SLOTS: usize = 1_025;
    let mut graph = GraphStore::default();
    insert_nodes(&mut graph, NODE_SLOTS)?;
    let bookmark = Bookmark {
        term: 31,
        index: graph.revision(),
    };
    let image = image(&graph, bookmark)?;
    let mut cpu = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
    cpu.admit_project(image)?;
    let request = pair_request(
        &graph,
        bookmark,
        NODE_SLOTS,
        3,
        ResidentPatternRelationshipTypes::Never,
        0,
    );
    let result = cpu
        .execute_pattern_predicate_pairs(&request, &CancellationToken::new())?
        .validate(&request, BackendKind::Cpu)?;
    assert!(result.pair_positions().is_empty());
    assert_eq!(result.receipts()[2].input_cardinality, 1_050_625);
    assert_eq!(result.receipts()[3].output_cardinality, 0);
    Ok(())
}

#[test]
#[ignore = "requires an available physical Metal device"]
fn parallel_compaction_empty_domains_publish_zero_receipts_on_real_metal() -> Result<()> {
    let _metal = metal_test_guard();
    let graph = GraphStore::default();
    let bookmark = Bookmark { term: 32, index: 0 };
    let (cpu, metal) = admitted_backends(&graph, bookmark)?;
    let cancellation = CancellationToken::new();

    let one = one_request(
        &graph,
        bookmark,
        0,
        10,
        ResidentPatternRelationshipTypes::Any,
        0,
    );
    for (backend, kind) in [
        (&cpu as &dyn ExecutionBackend, BackendKind::Cpu),
        (&metal as &dyn ExecutionBackend, BackendKind::Metal),
    ] {
        let result = backend
            .execute_pattern_predicate(&one, &cancellation)?
            .validate(&one, kind)?;
        assert!(result.rows().is_empty());
        assert!(
            result
                .receipts()
                .iter()
                .all(|receipt| receipt.input_cardinality == 0 && receipt.output_cardinality == 0)
        );
    }

    let pair = pair_request(
        &graph,
        bookmark,
        0,
        11,
        ResidentPatternRelationshipTypes::Any,
        0,
    );
    for (backend, kind) in [
        (&cpu as &dyn ExecutionBackend, BackendKind::Cpu),
        (&metal as &dyn ExecutionBackend, BackendKind::Metal),
    ] {
        let result = backend
            .execute_pattern_predicate_pairs(&pair, &cancellation)?
            .validate(&pair, kind)?;
        assert!(result.pair_positions().is_empty());
        assert!(
            result
                .receipts()
                .iter()
                .all(|receipt| receipt.input_cardinality == 0 && receipt.output_cardinality == 0)
        );
    }
    Ok(())
}

#[test]
#[ignore = "requires an available physical Metal device"]
fn parallel_compaction_multitile_rows_are_stable_and_cpu_metal_identical() -> Result<()> {
    let _metal = metal_test_guard();
    const NODE_SLOTS: usize = 4 * TILE_ROWS + 17;
    let mut graph = GraphStore::default();
    let relationship_type = graph.catalog_mut().intern_relationship_type("REL")?;
    insert_nodes(&mut graph, NODE_SLOTS)?;
    let mut expected = Vec::new();
    let mut edge_id = 1_000_000_u64;
    for source in 0..NODE_SLOTS {
        if source % 3 == 1 {
            continue;
        }
        expected.push(source as u32);
        graph.insert_edge(EdgeInput {
            id: EdgeId(edge_id),
            source: NodeId((source + 1) as u64),
            target: NodeId(((source + 1) % NODE_SLOTS + 1) as u64),
            relationship_type,
            layer: Layer::Observed,
            revision: edge_id,
            properties: Vec::new(),
        })?;
        edge_id += 1;
    }
    assert_eq!(NODE_SLOTS.div_ceil(TILE_ROWS), 5);
    let bookmark = Bookmark {
        term: 33,
        index: graph.revision(),
    };
    let (cpu, metal) = admitted_backends(&graph, bookmark)?;
    let request = one_request(
        &graph,
        bookmark,
        NODE_SLOTS,
        20,
        ResidentPatternRelationshipTypes::Any,
        expected.len(),
    );
    let cancellation = CancellationToken::new();
    let cpu_result = cpu
        .execute_pattern_predicate(&request, &cancellation)?
        .validate(&request, BackendKind::Cpu)?;
    assert_eq!(cpu_result.rows(), expected);

    for _ in 0..3 {
        let metal_result = metal
            .execute_pattern_predicate(&request, &cancellation)?
            .validate(&request, BackendKind::Metal)?;
        assert_eq!(metal_result.rows(), cpu_result.rows());
        assert!(metal_result.rows().windows(2).all(|rows| rows[0] < rows[1]));
        for (cpu_receipt, metal_receipt) in
            cpu_result.receipts().iter().zip(metal_result.receipts())
        {
            assert_eq!(cpu_receipt.obligation, metal_receipt.obligation);
            assert_eq!(
                cpu_receipt.input_cardinality,
                metal_receipt.input_cardinality
            );
            assert_eq!(
                cpu_receipt.output_cardinality,
                metal_receipt.output_cardinality
            );
            assert_eq!(metal_receipt.completion, ResidentDeviceCompletion::Metal);
        }
    }

    let zero = one_request(
        &graph,
        bookmark,
        NODE_SLOTS,
        21,
        ResidentPatternRelationshipTypes::Never,
        0,
    );
    for (backend, kind) in [
        (&cpu as &dyn ExecutionBackend, BackendKind::Cpu),
        (&metal as &dyn ExecutionBackend, BackendKind::Metal),
    ] {
        let result = backend
            .execute_pattern_predicate(&zero, &cancellation)?
            .validate(&zero, kind)?;
        assert!(result.rows().is_empty());
    }

    let overflowing = one_request(
        &graph,
        bookmark,
        NODE_SLOTS,
        22,
        ResidentPatternRelationshipTypes::Any,
        0,
    );
    for backend in [
        &cpu as &dyn ExecutionBackend,
        &metal as &dyn ExecutionBackend,
    ] {
        let error = backend
            .execute_pattern_predicate(&overflowing, &cancellation)
            .expect_err("zero capacity must not truncate selected rows");
        assert_eq!(error.code, ErrorCode::ResultBudgetExceeded);
    }
    Ok(())
}

#[test]
#[ignore = "requires an available physical Metal device"]
fn parallel_compaction_crosses_a_physical_pair_chunk_in_row_major_order() -> Result<()> {
    let _metal = metal_test_guard();
    const NODE_SLOTS: usize = 1_025;
    let mut graph = GraphStore::default();
    let relationship_type: RelationshipTypeId =
        graph.catalog_mut().intern_relationship_type("REL")?;
    insert_nodes(&mut graph, NODE_SLOTS)?;
    let mut edge_id = 2_000_000_u64;
    // Reverse insertion order ensures result order comes from pair-position compaction, not CSR
    // construction order. Each source has two distinct selected targets.
    for source in (0..NODE_SLOTS).rev() {
        for target in [(source + 17) % NODE_SLOTS, (source + 1) % NODE_SLOTS] {
            graph.insert_edge(EdgeInput {
                id: EdgeId(edge_id),
                source: NodeId((source + 1) as u64),
                target: NodeId((target + 1) as u64),
                relationship_type,
                layer: Layer::Observed,
                revision: edge_id,
                properties: Vec::new(),
            })?;
            edge_id += 1;
        }
    }
    let mut expected = Vec::with_capacity(NODE_SLOTS * 2);
    for source in 0..NODE_SLOTS {
        let mut targets = [(source + 1) % NODE_SLOTS, (source + 17) % NODE_SLOTS];
        targets.sort_unstable();
        expected.extend(targets.map(|target| (source as u32, target as u32)));
    }
    let bookmark = Bookmark {
        term: 34,
        index: graph.revision(),
    };
    let (cpu, metal) = admitted_backends(&graph, bookmark)?;
    let request = pair_request(
        &graph,
        bookmark,
        NODE_SLOTS,
        30,
        ResidentPatternRelationshipTypes::Any,
        expected.len(),
    );
    assert_eq!(request.logical_pair_domain()?, 1_050_625);
    let cancellation = CancellationToken::new();
    let cpu_result = cpu
        .execute_pattern_predicate_pairs(&request, &cancellation)?
        .validate(&request, BackendKind::Cpu)?;
    let cpu_pairs = cpu_result
        .n_rows()
        .iter()
        .copied()
        .zip(cpu_result.m_rows().iter().copied())
        .collect::<Vec<_>>();
    assert_eq!(cpu_pairs, expected);

    for _ in 0..3 {
        let metal_result = metal
            .execute_pattern_predicate_pairs(&request, &cancellation)?
            .validate(&request, BackendKind::Metal)?;
        let metal_pairs = metal_result
            .n_rows()
            .iter()
            .copied()
            .zip(metal_result.m_rows().iter().copied())
            .collect::<Vec<_>>();
        assert_eq!(metal_pairs, cpu_pairs);
        assert_eq!(metal_result.pair_positions(), cpu_result.pair_positions());
        assert!(
            metal_result
                .pair_positions()
                .windows(2)
                .all(|positions| positions[0] < positions[1])
        );
    }

    let zero = pair_request(
        &graph,
        bookmark,
        NODE_SLOTS,
        31,
        ResidentPatternRelationshipTypes::Never,
        0,
    );
    for (backend, kind) in [
        (&cpu as &dyn ExecutionBackend, BackendKind::Cpu),
        (&metal as &dyn ExecutionBackend, BackendKind::Metal),
    ] {
        let result = backend
            .execute_pattern_predicate_pairs(&zero, &cancellation)?
            .validate(&zero, kind)?;
        assert!(result.pair_positions().is_empty());
    }

    let overflowing = pair_request(
        &graph,
        bookmark,
        NODE_SLOTS,
        32,
        ResidentPatternRelationshipTypes::Any,
        0,
    );
    for backend in [
        &cpu as &dyn ExecutionBackend,
        &metal as &dyn ExecutionBackend,
    ] {
        let error = backend
            .execute_pattern_predicate_pairs(&overflowing, &cancellation)
            .expect_err("zero capacity must not truncate selected pairs");
        assert_eq!(error.code, ErrorCode::ResultBudgetExceeded);
    }
    Ok(())
}
