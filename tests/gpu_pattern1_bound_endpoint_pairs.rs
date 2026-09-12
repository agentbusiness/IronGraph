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
        ResidentObligationKind, ResidentObligationScope, ResidentPatternPairPredicateLeaf,
        ResidentPatternPairPredicateLength, ResidentPatternPairPredicateProgram,
        ResidentPatternPairPredicateRequest, ResidentPatternPairPredicateResult,
        ResidentPatternPairVisibleNodeScans, ResidentPatternRelationshipTypes,
        ResidentProjectImage,
    },
    graph::{EdgeInput, GraphStore, IndexCatalog, LayerMask, NodeInput, TemporalStore},
    types::RelationshipTypeId,
};
use tokio_util::sync::CancellationToken;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());
const MEMORY_LIMIT_BYTES: usize = 256 * 1024 * 1024;
const RESERVED_BYTES: usize = 32 * 1024 * 1024;
const ABOVE_FORMER_PAIR_NODE_SLOTS: usize = 257;

fn metal_test_guard() -> MutexGuard<'static, ()> {
    static METAL_TEST: Mutex<()> = Mutex::new(());
    match METAL_TEST.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

struct Fixture {
    graph: GraphStore,
    bookmark: Bookmark,
    node_slots: usize,
    rel1: RelationshipTypeId,
    rel2: RelationshipTypeId,
}

impl Fixture {
    fn image(&self) -> Result<ResidentProjectImage> {
        ResidentProjectImage::build(
            PROJECT,
            self.bookmark,
            &self.graph,
            &TemporalStore::default(),
            &IndexCatalog::default(),
        )
    }

    fn graph_revision(&self) -> u64 {
        self.graph.revision()
    }

    fn layout_version(&self) -> u64 {
        self.graph.layout_version()
    }
}

fn admitted_backends(fixture: &Fixture) -> Result<(CpuBackend, MetalBackend)> {
    let image = fixture.image()?;
    let mut cpu = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
    let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
    cpu.admit_project(image.clone())?;
    metal.admit_project(image)?;
    Ok((cpu, metal))
}

fn obligation(
    id: u64,
    kind: ResidentObligationKind,
    scope: ResidentObligationScope,
) -> ResidentExecutionObligation {
    ResidentExecutionObligation { id, kind, scope }
}

fn known(mut types: Vec<RelationshipTypeId>) -> ResidentPatternRelationshipTypes {
    types.sort_unstable();
    types.dedup();
    ResidentPatternRelationshipTypes::Known(types)
}

fn request(
    fixture: &Fixture,
    seed: u64,
    layers: LayerMask,
    direction: ResidentDirection,
    relationship_types: ResidentPatternRelationshipTypes,
    length: ResidentPatternPairPredicateLength,
    max_output_pairs: usize,
) -> ResidentPatternPairPredicateRequest {
    let obligation_base = seed * 10;
    ResidentPatternPairPredicateRequest {
        project: PROJECT,
        expected_bookmark: fixture.bookmark,
        expected_graph_revision: fixture.graph_revision(),
        expected_layout_version: fixture.layout_version(),
        layers,
        input: ResidentPatternPairVisibleNodeScans {
            node_slots: fixture.node_slots,
            n_obligation: obligation(
                obligation_base + 1,
                ResidentObligationKind::PatternScan,
                ResidentObligationScope::PatternScanN,
            ),
            m_obligation: obligation(
                obligation_base + 2,
                ResidentObligationKind::PatternScan,
                ResidentObligationScope::PatternScanM,
            ),
            cartesian_obligation: obligation(
                obligation_base + 3,
                ResidentObligationKind::PatternCartesian,
                ResidentObligationScope::PatternCartesian,
            ),
        },
        program: ResidentPatternPairPredicateProgram {
            execution: ResidentExecutionId {
                high: 0x5041_4952_5445_5354,
                low: seed,
            },
            leaf: ResidentPatternPairPredicateLeaf {
                direction,
                relationship_types,
                length,
                obligation: obligation(
                    obligation_base + 4,
                    ResidentObligationKind::PatternTraversal,
                    ResidentObligationScope::PatternLeaf(0),
                ),
            },
            final_obligation: obligation(
                obligation_base + 5,
                ResidentObligationKind::PatternFilter,
                ResidentObligationScope::PatternFinal,
            ),
        },
        max_output_pairs,
    }
}

fn assert_cpu_metal_pairs(
    cpu: &CpuBackend,
    metal: &MetalBackend,
    request: &ResidentPatternPairPredicateRequest,
    visible_nodes: usize,
    expected: &[(u32, u32)],
) -> Result<()> {
    request.validate()?;
    assert!(expected.windows(2).all(|pair| pair[0] < pair[1]));

    let cancellation = CancellationToken::new();
    let cpu_raw: ResidentPatternPairPredicateResult =
        cpu.execute_pattern_predicate_pairs(request, &cancellation)?;
    let metal_raw: ResidentPatternPairPredicateResult =
        metal.execute_pattern_predicate_pairs(request, &cancellation)?;
    let cpu_result = cpu_raw.validate(request, BackendKind::Cpu)?;
    let metal_result = metal_raw.validate(request, BackendKind::Metal)?;

    let expected_positions = expected
        .iter()
        .map(|(n, m)| u64::from(*n) * request.input.node_slots as u64 + u64::from(*m))
        .collect::<Vec<_>>();
    let cpu_pairs = cpu_result
        .n_rows()
        .iter()
        .copied()
        .zip(cpu_result.m_rows().iter().copied())
        .collect::<Vec<_>>();
    let metal_pairs = metal_result
        .n_rows()
        .iter()
        .copied()
        .zip(metal_result.m_rows().iter().copied())
        .collect::<Vec<_>>();

    assert_eq!(cpu_pairs, expected);
    assert_eq!(metal_pairs, expected);
    assert_eq!(metal_pairs, cpu_pairs);
    assert_eq!(cpu_result.pair_positions(), expected_positions);
    assert_eq!(metal_result.pair_positions(), expected_positions);
    assert!(
        metal_result
            .pair_positions()
            .windows(2)
            .all(|positions| positions[0] < positions[1])
    );

    for result in [&cpu_result, &metal_result] {
        assert_eq!(result.execution(), request.program.execution);
        assert_eq!(result.bookmark(), request.expected_bookmark);
        assert_eq!(result.graph_revision(), request.expected_graph_revision);
        assert_eq!(result.layout_version(), request.expected_layout_version);
        assert_eq!(result.receipts().len(), 5);
    }

    let obligations = request.obligations();
    let visible_pairs = (visible_nodes as u64) * (visible_nodes as u64);
    for (index, ((cpu_receipt, metal_receipt), expected_obligation)) in cpu_result
        .receipts()
        .iter()
        .zip(metal_result.receipts())
        .zip(obligations)
        .enumerate()
    {
        assert_eq!(cpu_receipt.execution, request.program.execution);
        assert_eq!(metal_receipt.execution, request.program.execution);
        assert_eq!(cpu_receipt.obligation, expected_obligation);
        assert_eq!(metal_receipt.obligation, expected_obligation);
        assert_eq!(
            cpu_receipt.completion,
            ResidentDeviceCompletion::CpuReference
        );
        assert_eq!(metal_receipt.completion, ResidentDeviceCompletion::Metal);
        assert_eq!(
            cpu_receipt.input_cardinality,
            metal_receipt.input_cardinality
        );
        assert_eq!(
            cpu_receipt.output_cardinality,
            metal_receipt.output_cardinality
        );
        match index {
            0 | 1 => {
                assert_eq!(
                    cpu_receipt.input_cardinality,
                    request.input.node_slots as u64
                );
                assert_eq!(cpu_receipt.output_cardinality, visible_nodes as u64);
            }
            2 => {
                assert_eq!(cpu_receipt.input_cardinality, visible_pairs);
                assert_eq!(cpu_receipt.output_cardinality, visible_pairs);
            }
            3 => {
                assert_eq!(cpu_receipt.input_cardinality, visible_pairs);
                assert_eq!(cpu_receipt.output_cardinality, expected.len() as u64);
            }
            4 => {
                assert_eq!(cpu_receipt.input_cardinality, visible_pairs);
                assert_eq!(cpu_receipt.output_cardinality, expected.len() as u64);
            }
            _ => unreachable!(),
        }
    }
    Ok(())
}

fn direction_and_visibility_fixture() -> Result<Fixture> {
    let mut graph = GraphStore::default();
    let rel1 = graph.catalog_mut().intern_relationship_type("REL1")?;
    let rel2 = graph.catalog_mut().intern_relationship_type("REL2")?;
    for (id, layer) in [
        (1, Layer::Observed),
        (2, Layer::Observed),
        (3, Layer::Knowledge),
        (4, Layer::Observed),
        (5, Layer::Observed),
        (6, Layer::Observed),
        (7, Layer::Observed),
    ] {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer,
            revision: id,
            labels: Vec::new(),
            properties: Vec::new(),
        })?;
    }
    for (id, revision, source, target, relationship_type, layer) in [
        (101, 8, 1, 2, rel1, Layer::Observed),
        (102, 9, 2, 1, rel2, Layer::Observed),
        (103, 10, 1, 5, rel1, Layer::Observed),
        (104, 11, 5, 6, rel2, Layer::Observed),
        (105, 12, 6, 5, rel1, Layer::Observed),
        // Both endpoints remain in the physical image, but node 3 is hidden by its layer.
        (106, 13, 1, 3, rel1, Layer::Observed),
        // Deleting node 4 below tombstones this relationship without reclaiming either slot.
        (107, 14, 1, 4, rel1, Layer::Observed),
        // Both endpoints are visible, but this relationship is hidden by its layer.
        (108, 15, 6, 7, rel1, Layer::Knowledge),
    ] {
        graph.insert_edge(EdgeInput {
            id: EdgeId(id),
            source: NodeId(source),
            target: NodeId(target),
            relationship_type,
            layer,
            revision,
            properties: Vec::new(),
        })?;
    }
    graph.delete_node(NodeId(4), true, 16)?;
    let node_slots = graph.snapshot()?.node_ids.len();
    assert_eq!(node_slots, 7);
    Ok(Fixture {
        graph,
        bookmark: Bookmark { term: 7, index: 16 },
        node_slots,
        rel1,
        rel2,
    })
}

/// Four disconnected components expose the non-empty relationship-trail rules directly:
/// one ordinary edge, two parallel edges, a directed three-edge cycle, and one self-loop.
fn trail_fixture() -> Result<Fixture> {
    let mut graph = GraphStore::default();
    let rel1 = graph.catalog_mut().intern_relationship_type("REL1")?;
    let rel2 = graph.catalog_mut().intern_relationship_type("REL2")?;
    for id in 1..=8 {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: id,
            labels: Vec::new(),
            properties: Vec::new(),
        })?;
    }
    for (id, revision, source, target) in [
        (201, 9, 1, 2),
        (202, 10, 3, 4),
        (203, 11, 3, 4),
        (204, 12, 5, 6),
        (205, 13, 6, 7),
        (206, 14, 7, 5),
        (207, 15, 8, 8),
    ] {
        graph.insert_edge(EdgeInput {
            id: EdgeId(id),
            source: NodeId(source),
            target: NodeId(target),
            relationship_type: rel1,
            layer: Layer::Observed,
            revision,
            properties: Vec::new(),
        })?;
    }
    Ok(Fixture {
        graph,
        bookmark: Bookmark {
            term: 11,
            index: 15,
        },
        node_slots: 8,
        rel1,
        rel2,
    })
}

fn above_former_pair_domain_fixture() -> Result<Fixture> {
    let mut graph = GraphStore::default();
    let rel1 = graph.catalog_mut().intern_relationship_type("REL1")?;
    let rel2 = graph.catalog_mut().intern_relationship_type("REL2")?;
    for id in 1..=ABOVE_FORMER_PAIR_NODE_SLOTS as u64 {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: id,
            labels: Vec::new(),
            properties: Vec::new(),
        })?;
    }
    Ok(Fixture {
        graph,
        bookmark: Bookmark {
            term: 13,
            index: ABOVE_FORMER_PAIR_NODE_SLOTS as u64,
        },
        node_slots: ABOVE_FORMER_PAIR_NODE_SLOTS,
        rel1,
        rel2,
    })
}

#[test]
fn cpu_pair_domain_crosses_the_former_256_by_256_boundary() -> Result<()> {
    let fixture = above_former_pair_domain_fixture()?;
    let image = fixture.image()?;
    let mut cpu = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
    cpu.admit_project(image)?;
    let request = request(
        &fixture,
        99,
        LayerMask::OBSERVED,
        ResidentDirection::Undirected,
        ResidentPatternRelationshipTypes::Never,
        ResidentPatternPairPredicateLength::OneOrMoreReachable,
        0,
    );
    assert_eq!(request.logical_pair_domain()?, 66_049);
    let result = cpu
        .execute_pattern_predicate_pairs(&request, &CancellationToken::new())?
        .validate(&request, BackendKind::Cpu)?;
    assert!(result.pair_positions().is_empty());
    assert_eq!(result.receipts()[2].input_cardinality, 66_049);
    assert_eq!(result.receipts()[3].output_cardinality, 0);
    Ok(())
}

#[test]
#[ignore = "requires an available physical Metal device"]
fn paired_one_directions_types_visibility_and_row_major_receipts_match_cpu_and_metal() -> Result<()>
{
    let _metal = metal_test_guard();
    let fixture = direction_and_visibility_fixture()?;
    let (cpu, metal) = admitted_backends(&fixture)?;
    let capacity = fixture.node_slots * fixture.node_slots;

    let outgoing_any = request(
        &fixture,
        101,
        LayerMask::OBSERVED,
        ResidentDirection::Outgoing,
        ResidentPatternRelationshipTypes::Any,
        ResidentPatternPairPredicateLength::One,
        capacity,
    );
    assert_cpu_metal_pairs(
        &cpu,
        &metal,
        &outgoing_any,
        5,
        &[(0, 1), (0, 4), (1, 0), (4, 5), (5, 4)],
    )?;

    let incoming_known = request(
        &fixture,
        102,
        LayerMask::OBSERVED,
        ResidentDirection::Incoming,
        known(vec![fixture.rel1]),
        ResidentPatternPairPredicateLength::One,
        capacity,
    );
    assert_cpu_metal_pairs(&cpu, &metal, &incoming_known, 5, &[(1, 0), (4, 0), (4, 5)])?;

    let undirected_known = request(
        &fixture,
        103,
        LayerMask::OBSERVED,
        ResidentDirection::Undirected,
        known(vec![fixture.rel2]),
        ResidentPatternPairPredicateLength::One,
        capacity,
    );
    assert_cpu_metal_pairs(
        &cpu,
        &metal,
        &undirected_known,
        5,
        &[(0, 1), (1, 0), (4, 5), (5, 4)],
    )?;

    let never = request(
        &fixture,
        104,
        LayerMask::OBSERVED,
        ResidentDirection::Undirected,
        ResidentPatternRelationshipTypes::Never,
        ResidentPatternPairPredicateLength::One,
        0,
    );
    assert_cpu_metal_pairs(&cpu, &metal, &never, 5, &[])?;
    Ok(())
}

#[test]
#[ignore = "requires an available physical Metal device"]
fn paired_reachability_and_exact_two_trails_match_cpu_and_real_metal() -> Result<()> {
    let _metal = metal_test_guard();
    let fixture = trail_fixture()?;
    let (cpu, metal) = admitted_backends(&fixture)?;
    let capacity = fixture.node_slots * fixture.node_slots;

    let direct = request(
        &fixture,
        201,
        LayerMask::OBSERVED,
        ResidentDirection::Outgoing,
        known(vec![fixture.rel1]),
        ResidentPatternPairPredicateLength::One,
        capacity,
    );
    assert_cpu_metal_pairs(
        &cpu,
        &metal,
        &direct,
        8,
        &[(0, 1), (2, 3), (4, 5), (5, 6), (6, 4), (7, 7)],
    )?;

    let directed_reachable = request(
        &fixture,
        202,
        LayerMask::OBSERVED,
        ResidentDirection::Outgoing,
        known(vec![fixture.rel1]),
        ResidentPatternPairPredicateLength::OneOrMoreReachable,
        capacity,
    );
    assert_cpu_metal_pairs(
        &cpu,
        &metal,
        &directed_reachable,
        8,
        &[
            (0, 1),
            (2, 3),
            (4, 4),
            (4, 5),
            (4, 6),
            (5, 4),
            (5, 5),
            (5, 6),
            (6, 4),
            (6, 5),
            (6, 6),
            (7, 7),
        ],
    )?;

    let incoming_reachable = request(
        &fixture,
        203,
        LayerMask::OBSERVED,
        ResidentDirection::Incoming,
        known(vec![fixture.rel1]),
        ResidentPatternPairPredicateLength::OneOrMoreReachable,
        capacity,
    );
    assert_cpu_metal_pairs(
        &cpu,
        &metal,
        &incoming_reachable,
        8,
        &[
            (1, 0),
            (3, 2),
            (4, 4),
            (4, 5),
            (4, 6),
            (5, 4),
            (5, 5),
            (5, 6),
            (6, 4),
            (6, 5),
            (6, 6),
            (7, 7),
        ],
    )?;

    let undirected_reachable = request(
        &fixture,
        204,
        LayerMask::OBSERVED,
        ResidentDirection::Undirected,
        known(vec![fixture.rel1]),
        ResidentPatternPairPredicateLength::OneOrMoreReachable,
        capacity,
    );
    let undirected_reachable_pairs = [
        (0, 1),
        (1, 0),
        (2, 2),
        (2, 3),
        (3, 2),
        (3, 3),
        (4, 4),
        (4, 5),
        (4, 6),
        (5, 4),
        (5, 5),
        (5, 6),
        (6, 4),
        (6, 5),
        (6, 6),
        (7, 7),
    ];
    assert_cpu_metal_pairs(
        &cpu,
        &metal,
        &undirected_reachable,
        8,
        &undirected_reachable_pairs,
    )?;
    assert!(!undirected_reachable_pairs.contains(&(0, 0)));
    assert!(!undirected_reachable_pairs.contains(&(1, 1)));
    assert!(undirected_reachable_pairs.contains(&(2, 2)));
    assert!(undirected_reachable_pairs.contains(&(3, 3)));
    assert!(undirected_reachable_pairs.contains(&(4, 4)));
    assert!(undirected_reachable_pairs.contains(&(7, 7)));

    let outgoing_exact_two = request(
        &fixture,
        205,
        LayerMask::OBSERVED,
        ResidentDirection::Outgoing,
        known(vec![fixture.rel1]),
        ResidentPatternPairPredicateLength::ExactTwo,
        capacity,
    );
    assert_cpu_metal_pairs(
        &cpu,
        &metal,
        &outgoing_exact_two,
        8,
        &[(4, 6), (5, 4), (6, 5)],
    )?;

    let incoming_exact_two = request(
        &fixture,
        206,
        LayerMask::OBSERVED,
        ResidentDirection::Incoming,
        known(vec![fixture.rel1]),
        ResidentPatternPairPredicateLength::ExactTwo,
        capacity,
    );
    assert_cpu_metal_pairs(
        &cpu,
        &metal,
        &incoming_exact_two,
        8,
        &[(4, 5), (5, 6), (6, 4)],
    )?;

    let undirected_exact_two = request(
        &fixture,
        207,
        LayerMask::OBSERVED,
        ResidentDirection::Undirected,
        known(vec![fixture.rel1]),
        ResidentPatternPairPredicateLength::ExactTwo,
        capacity,
    );
    assert_cpu_metal_pairs(
        &cpu,
        &metal,
        &undirected_exact_two,
        8,
        &[
            (2, 2),
            (3, 3),
            (4, 5),
            (4, 6),
            (5, 4),
            (5, 6),
            (6, 4),
            (6, 5),
        ],
    )?;
    Ok(())
}

#[test]
#[ignore = "requires an available physical Metal device"]
fn paired_identity_domain_output_budget_and_cancellation_fail_closed() -> Result<()> {
    let _metal = metal_test_guard();

    let fixture = direction_and_visibility_fixture()?;
    let (cpu, metal) = admitted_backends(&fixture)?;
    let base = request(
        &fixture,
        301,
        LayerMask::OBSERVED,
        ResidentDirection::Outgoing,
        ResidentPatternRelationshipTypes::Any,
        ResidentPatternPairPredicateLength::One,
        fixture.node_slots * fixture.node_slots,
    );
    for stale in [
        {
            let mut stale = base.clone();
            stale.expected_bookmark.index += 1;
            stale
        },
        {
            let mut stale = base.clone();
            stale.expected_graph_revision += 1;
            stale
        },
        {
            let mut stale = base.clone();
            stale.expected_layout_version += 1;
            stale
        },
    ] {
        for backend in [
            &cpu as &dyn ExecutionBackend,
            &metal as &dyn ExecutionBackend,
        ] {
            let error = backend
                .execute_pattern_predicate_pairs(&stale, &CancellationToken::new())
                .expect_err("stale paired graph identity must fail before publication");
            assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
        }
    }

    let mut zero_output_budget = base.clone();
    zero_output_budget.max_output_pairs = 0;
    for backend in [
        &cpu as &dyn ExecutionBackend,
        &metal as &dyn ExecutionBackend,
    ] {
        let error = backend
            .execute_pattern_predicate_pairs(&zero_output_budget, &CancellationToken::new())
            .expect_err("a non-empty paired result must not be truncated to zero rows");
        assert_eq!(error.code, ErrorCode::ResultBudgetExceeded);
    }

    let cancelled = CancellationToken::new();
    cancelled.cancel();
    for backend in [
        &cpu as &dyn ExecutionBackend,
        &metal as &dyn ExecutionBackend,
    ] {
        let error = backend
            .execute_pattern_predicate_pairs(&base, &cancelled)
            .expect_err("pre-cancelled paired work must not complete");
        assert_eq!(error.code, ErrorCode::Cancelled);
    }

    drop(metal);
    drop(cpu);

    let maximum = above_former_pair_domain_fixture()?;
    let (maximum_cpu, maximum_metal) = admitted_backends(&maximum)?;
    let maximum_request = request(
        &maximum,
        302,
        LayerMask::OBSERVED,
        ResidentDirection::Undirected,
        ResidentPatternRelationshipTypes::Never,
        ResidentPatternPairPredicateLength::OneOrMoreReachable,
        0,
    );
    assert_eq!(maximum_request.logical_pair_domain()?, 66_049);
    assert_cpu_metal_pairs(
        &maximum_cpu,
        &maximum_metal,
        &maximum_request,
        ABOVE_FORMER_PAIR_NODE_SLOTS,
        &[],
    )?;

    let mut different_resident_size = maximum_request.clone();
    different_resident_size.input.node_slots += 1;
    different_resident_size.validate()?;
    for backend in [
        &maximum_cpu as &dyn ExecutionBackend,
        &maximum_metal as &dyn ExecutionBackend,
    ] {
        let error = backend
            .execute_pattern_predicate_pairs(&different_resident_size, &CancellationToken::new())
            .expect_err("a backend must reject a request for a different resident image size");
        assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
    }

    let mut one_output_over = maximum_request;
    one_output_over.max_output_pairs = one_output_over
        .pair_domain()?
        .checked_add(1)
        .ok_or_else(|| irongraph::Error::internal("test pair output capacity overflow"))?;
    let error = one_output_over
        .validate()
        .expect_err("output capacity cannot exceed the admitted Cartesian domain");
    assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
    Ok(())
}

#[test]
#[ignore = "requires an available physical Metal device"]
fn public_result_validation_rejects_cross_request_provenance_and_cardinality() -> Result<()> {
    let _metal = metal_test_guard();
    let fixture = direction_and_visibility_fixture()?;
    let (cpu, metal) = admitted_backends(&fixture)?;
    let base = request(
        &fixture,
        401,
        LayerMask::OBSERVED,
        ResidentDirection::Outgoing,
        known(vec![fixture.rel1]),
        ResidentPatternPairPredicateLength::One,
        fixture.node_slots * fixture.node_slots,
    );
    let raw = cpu.execute_pattern_predicate_pairs(&base, &CancellationToken::new())?;
    raw.clone().validate(&base, BackendKind::Cpu)?;

    let error = raw
        .clone()
        .validate(&base, BackendKind::Metal)
        .expect_err("CPU completion receipts must not masquerade as Metal");
    assert_eq!(error.code, ErrorCode::CorruptStorage);

    let mut wrong_execution = base.clone();
    wrong_execution.program.execution.low += 1;
    let error = raw
        .clone()
        .validate(&wrong_execution, BackendKind::Cpu)
        .expect_err("a result from another execution must be rejected");
    assert_eq!(error.code, ErrorCode::CorruptStorage);

    let mut wrong_obligation = base.clone();
    wrong_obligation.program.final_obligation.id += 1_000_000;
    let error = raw
        .clone()
        .validate(&wrong_obligation, BackendKind::Cpu)
        .expect_err("receipt coverage from another obligation manifest must be rejected");
    assert_eq!(error.code, ErrorCode::CorruptStorage);

    let mut wrong_cardinality = base.clone();
    wrong_cardinality.input.node_slots += 1;
    let error = raw
        .clone()
        .validate(&wrong_cardinality, BackendKind::Cpu)
        .expect_err("scan receipts cannot certify a different physical node domain");
    assert_eq!(error.code, ErrorCode::CorruptStorage);

    let mut too_small_output = base.clone();
    too_small_output.max_output_pairs = 1;
    let error = raw
        .validate(&too_small_output, BackendKind::Cpu)
        .expect_err("validated results cannot exceed the request output budget");
    assert_eq!(error.code, ErrorCode::CorruptStorage);

    let metal_raw = metal.execute_pattern_predicate_pairs(&base, &CancellationToken::new())?;
    let error = metal_raw
        .validate(&base, BackendKind::Cpu)
        .expect_err("Metal completion receipts must not masquerade as CPU reference work");
    assert_eq!(error.code, ErrorCode::CorruptStorage);
    Ok(())
}
