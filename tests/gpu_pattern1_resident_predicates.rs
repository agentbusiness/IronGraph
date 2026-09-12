// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]
#![cfg(all(feature = "accelerator", target_os = "macos"))]

use std::sync::{Arc, Mutex, MutexGuard};

use irongraph::{
    Bookmark, EdgeId, ErrorCode, Layer, NodeId, ProjectId, ScalarValue,
    gpu::{
        BackendKind, CpuBackend, ExecutionBackend, MetalBackend, ResidentDeviceCompletion,
        ResidentDirection, ResidentExecutionId, ResidentExecutionObligation,
        ResidentObligationKind, ResidentObligationScope, ResidentPatternBooleanInstruction,
        ResidentPatternPredicateInput, ResidentPatternPredicateLeaf,
        ResidentPatternPredicateLength, ResidentPatternPredicateProgram,
        ResidentPatternPredicatePropertyEquality, ResidentPatternPredicateRequest,
        ResidentPatternRelationshipTypes,
    },
    graph::{EdgeInput, GraphStore, LayerMask, NodeInput},
    types::{PropertyId, RelationshipTypeId},
};
use tokio_util::sync::CancellationToken;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());

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

fn leaf(
    seed: u64,
    index: u16,
    direction: ResidentDirection,
    relationship_types: ResidentPatternRelationshipTypes,
    length: ResidentPatternPredicateLength,
) -> ResidentPatternPredicateLeaf {
    ResidentPatternPredicateLeaf {
        direction,
        relationship_types,
        length,
        property_equality: None,
        minimum_witnesses: 1,
        maximum_witnesses: u32::MAX,
        obligation: obligation(
            seed * 1_000 + u64::from(index) + 1,
            ResidentObligationKind::PatternTraversal,
            ResidentObligationScope::PatternLeaf(index),
        ),
    }
}

fn known(mut types: Vec<RelationshipTypeId>) -> ResidentPatternRelationshipTypes {
    types.sort_unstable();
    types.dedup();
    ResidentPatternRelationshipTypes::Known(types)
}

fn program(
    seed: u64,
    leaves: Vec<ResidentPatternPredicateLeaf>,
    instructions: Vec<ResidentPatternBooleanInstruction>,
) -> ResidentPatternPredicateProgram {
    ResidentPatternPredicateProgram {
        execution: ResidentExecutionId {
            high: 0x5041_5454_4552_4e31,
            low: seed,
        },
        leaves,
        instructions,
        final_obligation: obligation(
            seed * 1_000 + 999,
            ResidentObligationKind::PatternFilter,
            ResidentObligationScope::PatternFinal,
        ),
    }
}

fn request(
    graph_revision: u64,
    input_rows: Vec<u32>,
    layers: LayerMask,
    program: ResidentPatternPredicateProgram,
    max_output_rows: Option<usize>,
) -> ResidentPatternPredicateRequest {
    let capacity = max_output_rows.unwrap_or(input_rows.len());
    ResidentPatternPredicateRequest {
        project: PROJECT,
        expected_bookmark: Bookmark {
            term: 0,
            index: graph_revision,
        },
        expected_graph_revision: graph_revision,
        expected_layout_version: 0,
        layers,
        input: ResidentPatternPredicateInput::Rows(input_rows),
        program,
        max_output_rows: capacity,
    }
}

fn scan_request(
    graph_revision: u64,
    layout_version: u64,
    node_slots: usize,
    layers: LayerMask,
    program: ResidentPatternPredicateProgram,
    max_output_rows: Option<usize>,
) -> ResidentPatternPredicateRequest {
    ResidentPatternPredicateRequest {
        project: PROJECT,
        expected_bookmark: Bookmark {
            term: 0,
            index: graph_revision,
        },
        expected_graph_revision: graph_revision,
        expected_layout_version: layout_version,
        layers,
        input: ResidentPatternPredicateInput::VisibleNodeScan {
            node_slots,
            obligation: obligation(
                program.execution.low * 1_000 + 998,
                ResidentObligationKind::PatternScan,
                ResidentObligationScope::PatternScan,
            ),
        },
        program,
        max_output_rows: max_output_rows.unwrap_or(node_slots),
    }
}

fn assert_cpu_metal_rows(
    cpu: &CpuBackend,
    metal: &MetalBackend,
    request: &ResidentPatternPredicateRequest,
    expected: &[u32],
) -> irongraph::Result<()> {
    let cancellation = CancellationToken::new();
    let cpu_result = cpu
        .execute_pattern_predicate(request, &cancellation)?
        .validate(request, BackendKind::Cpu)?;
    let metal_result = metal
        .execute_pattern_predicate(request, &cancellation)?
        .validate(request, BackendKind::Metal)?;
    assert_eq!(cpu_result.rows(), expected);
    assert_eq!(metal_result.rows(), expected);
    assert_eq!(metal_result.rows(), cpu_result.rows());
    assert_eq!(cpu_result.bookmark(), request.expected_bookmark);
    assert_eq!(metal_result.bookmark(), request.expected_bookmark);
    assert_eq!(cpu_result.graph_revision(), request.expected_graph_revision);
    assert_eq!(
        metal_result.graph_revision(),
        request.expected_graph_revision
    );
    assert_eq!(cpu_result.layout_version(), request.expected_layout_version);
    assert_eq!(
        metal_result.layout_version(),
        request.expected_layout_version
    );
    assert_eq!(metal_result.receipts().len(), request.obligations().count());
    for (index, (cpu_receipt, metal_receipt)) in cpu_result
        .receipts()
        .iter()
        .zip(metal_result.receipts())
        .enumerate()
    {
        assert_eq!(metal_receipt.execution, request.program.execution);
        assert_eq!(metal_receipt.obligation, cpu_receipt.obligation);
        assert_eq!(
            metal_receipt.input_cardinality,
            cpu_receipt.input_cardinality
        );
        assert_eq!(
            metal_receipt.output_cardinality,
            cpu_receipt.output_cardinality
        );
        assert_eq!(
            cpu_receipt.completion,
            ResidentDeviceCompletion::CpuReference
        );
        assert_eq!(metal_receipt.completion, ResidentDeviceCompletion::Metal);
        let scan_offset = usize::from(request.input.is_fused_visible_node_scan());
        if index < scan_offset {
            assert_eq!(
                metal_receipt.obligation.scope,
                ResidentObligationScope::PatternScan
            );
        } else if index < scan_offset + request.program.leaves.len() {
            assert_eq!(
                metal_receipt.obligation.scope,
                ResidentObligationScope::PatternLeaf((index - scan_offset) as u16)
            );
        } else {
            assert_eq!(
                metal_receipt.obligation.scope,
                ResidentObligationScope::PatternFinal
            );
        }
    }
    if request.input.is_fused_visible_node_scan() {
        let scan = &metal_result.receipts()[0];
        assert_eq!(scan.input_cardinality, request.input.row_count() as u64);
        for receipt in &metal_result.receipts()[1..] {
            assert_eq!(receipt.input_cardinality, scan.output_cardinality);
        }
    }
    Ok(())
}

fn resident_pattern_graph() -> irongraph::Result<(
    GraphStore,
    RelationshipTypeId,
    RelationshipTypeId,
    RelationshipTypeId,
)> {
    let mut graph = GraphStore::default();
    let knows = graph.catalog_mut().intern_relationship_type("KNOWS")?;
    let likes = graph.catalog_mut().intern_relationship_type("LIKES")?;
    let follows = graph.catalog_mut().intern_relationship_type("FOLLOWS")?;
    for id in 1..=6_u64 {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: id,
            labels: Vec::new(),
            properties: Vec::new(),
        })?;
    }
    for (id, source, target, relationship_type) in [
        (10, 1, 2, knows),
        // A parallel witness must not duplicate outer row zero.
        (11, 1, 2, knows),
        (12, 1, 3, likes),
        (13, 2, 1, knows),
        // This self-loop is present in both CSR orientations.
        (14, 5, 5, knows),
        (15, 1, 5, follows),
        (16, 3, 5, follows),
    ] {
        graph.insert_edge(EdgeInput {
            id: EdgeId(id),
            source: NodeId(source),
            target: NodeId(target),
            relationship_type,
            layer: Layer::Observed,
            revision: id,
            properties: Vec::new(),
        })?;
    }
    Ok((graph, knows, likes, follows))
}

#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_existential_leaf_matches_cpu_for_property_equality_and_witness_bounds()
-> irongraph::Result<()> {
    let _metal = metal_test_guard();
    let mut graph = GraphStore::default();
    let property = graph.catalog_mut().intern_property("prop")?;
    let relationship_type = graph.catalog_mut().intern_relationship_type("R")?;
    for (id, value) in [(1, 1), (2, 1), (3, 2), (4, 3)] {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: id,
            labels: Vec::new(),
            properties: vec![(property, ScalarValue::Integer(value))],
        })?;
    }
    for (id, source, target) in [(10, 1, 2), (11, 1, 3), (12, 1, 4), (13, 2, 4)] {
        graph.insert_edge(EdgeInput {
            id: EdgeId(id),
            source: NodeId(source),
            target: NodeId(target),
            relationship_type,
            layer: Layer::Observed,
            revision: id,
            properties: Vec::new(),
        })?;
    }
    let snapshot = Arc::new(graph.snapshot()?);
    let revision = snapshot.revision;
    let layout_version = snapshot.layout_version;
    let node_slots = snapshot.node_ids.len();
    let mut cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let mut metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    cpu.admit_graph(Arc::clone(&snapshot))?;
    metal.admit_graph(snapshot)?;

    let constrained = |seed: u64,
                       relationship_types: ResidentPatternRelationshipTypes,
                       equality: Option<(PropertyId, PropertyId)>,
                       minimum_witnesses: u32,
                       maximum_witnesses: u32| {
        let mut leaf = leaf(
            seed,
            0,
            ResidentDirection::Outgoing,
            relationship_types,
            ResidentPatternPredicateLength::One,
        );
        leaf.property_equality =
            equality.map(|(start, end)| ResidentPatternPredicatePropertyEquality { start, end });
        leaf.minimum_witnesses = minimum_witnesses;
        leaf.maximum_witnesses = maximum_witnesses;
        scan_request(
            revision,
            layout_version,
            node_slots,
            LayerMask::OBSERVED,
            program(
                seed,
                vec![leaf],
                vec![ResidentPatternBooleanInstruction::Leaf(0)],
            ),
            None,
        )
    };
    for (request, expected) in [
        (
            constrained(
                71,
                ResidentPatternRelationshipTypes::Any,
                Some((property, property)),
                1,
                u32::MAX,
            ),
            vec![0],
        ),
        (
            constrained(72, ResidentPatternRelationshipTypes::Any, None, 3, 3),
            vec![0],
        ),
        (
            constrained(73, known(vec![relationship_type]), None, 2, u32::MAX),
            vec![0],
        ),
        (
            constrained(
                74,
                ResidentPatternRelationshipTypes::Never,
                None,
                1,
                u32::MAX,
            ),
            Vec::new(),
        ),
    ] {
        assert_cpu_metal_rows(&cpu, &metal, &request, &expected)?;
    }

    // Admission is column-wide on both backends. An unrelated, unvisited STRING value sharing
    // the property token must not let CPU execute a shape that Metal rejects.
    graph.insert_node(NodeInput {
        id: NodeId(5),
        layer: Layer::Observed,
        revision: 20,
        labels: Vec::new(),
        properties: vec![(property, ScalarValue::String("not-an-integer".into()))],
    })?;
    let mixed_snapshot = Arc::new(graph.snapshot()?);
    let mut mixed_cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let mut mixed_metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    mixed_cpu.admit_graph(Arc::clone(&mixed_snapshot))?;
    mixed_metal.admit_graph(Arc::clone(&mixed_snapshot))?;
    let mixed_request = {
        let mut leaf = leaf(
            75,
            0,
            ResidentDirection::Outgoing,
            ResidentPatternRelationshipTypes::Any,
            ResidentPatternPredicateLength::One,
        );
        leaf.property_equality = Some(ResidentPatternPredicatePropertyEquality {
            start: property,
            end: property,
        });
        scan_request(
            mixed_snapshot.revision,
            mixed_snapshot.layout_version,
            mixed_snapshot.node_ids.len(),
            LayerMask::OBSERVED,
            program(
                75,
                vec![leaf],
                vec![ResidentPatternBooleanInstruction::Leaf(0)],
            ),
            None,
        )
    };
    for backend in [&mixed_cpu as &dyn ExecutionBackend, &mixed_metal] {
        let error = backend
            .execute_pattern_predicate(&mixed_request, &CancellationToken::new())
            .expect_err("mixed property column must fail explicit existential admission");
        assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
        assert!(error.message.contains("requires integer node columns"));
    }
    Ok(())
}

fn exact_two_pattern_graph()
-> irongraph::Result<(GraphStore, RelationshipTypeId, RelationshipTypeId)> {
    let mut graph = GraphStore::default();
    let rel1 = graph.catalog_mut().intern_relationship_type("REL1")?;
    let rel2 = graph.catalog_mut().intern_relationship_type("REL2")?;
    let rel3 = graph.catalog_mut().intern_relationship_type("REL3")?;
    for id in 1..=27_u64 {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: if matches!(id, 10 | 12 | 23) {
                Layer::Knowledge
            } else {
                Layer::Observed
            },
            revision: id,
            labels: Vec::new(),
            properties: Vec::new(),
        })?;
    }
    for (id, source, target, relationship_type, layer) in [
        // Official Pattern1 fixture component: exact-two undirected REL1 selects B and D only.
        (100, 1, 2, rel1, Layer::Observed),
        (101, 2, 1, rel2, Layer::Observed),
        (102, 1, 3, rel3, Layer::Observed),
        (103, 1, 4, rel1, Layer::Observed),
        // A three-edge cycle plus a parallel edge creates many witnesses per outer row.
        (104, 5, 6, rel1, Layer::Observed),
        (105, 6, 7, rel1, Layer::Observed),
        (106, 7, 5, rel1, Layer::Observed),
        (107, 5, 6, rel1, Layer::Observed),
        // One physical loop cannot be reused; two distinct loops can form a two-edge path.
        (108, 8, 8, rel1, Layer::Observed),
        (109, 9, 9, rel1, Layer::Observed),
        (110, 9, 9, rel1, Layer::Observed),
        // Hidden source and hidden final endpoint contracts.
        (111, 10, 11, rel1, Layer::Observed),
        (112, 11, 5, rel1, Layer::Observed),
        (113, 13, 14, rel1, Layer::Observed),
        (114, 14, 12, rel1, Layer::Observed),
        // Both hops must match the relationship type predicate.
        (115, 15, 16, rel1, Layer::Observed),
        (116, 16, 17, rel2, Layer::Observed),
        // A hidden second relationship cannot complete a visible two-hop path.
        (117, 18, 19, rel1, Layer::Observed),
        (118, 19, 20, rel1, Layer::Knowledge),
        // A complete physical path whose middle node alone is hidden.
        (119, 22, 23, rel1, Layer::Observed),
        (120, 23, 24, rel1, Layer::Observed),
        // A complete physical path whose first relationship alone is hidden.
        (121, 25, 26, rel1, Layer::Knowledge),
        (122, 26, 27, rel1, Layer::Observed),
    ] {
        graph.insert_edge(EdgeInput {
            id: EdgeId(id),
            source: NodeId(source),
            target: NodeId(target),
            relationship_type,
            layer,
            revision: id,
            properties: Vec::new(),
        })?;
    }
    Ok((graph, rel1, rel2))
}

fn official_pattern1_graph() -> irongraph::Result<(
    GraphStore,
    RelationshipTypeId,
    RelationshipTypeId,
    RelationshipTypeId,
)> {
    let mut graph = GraphStore::default();
    let rel1 = graph.catalog_mut().intern_relationship_type("REL1")?;
    let rel2 = graph.catalog_mut().intern_relationship_type("REL2")?;
    let rel3 = graph.catalog_mut().intern_relationship_type("REL3")?;
    for id in 1..=4_u64 {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: id,
            labels: Vec::new(),
            properties: Vec::new(),
        })?;
    }
    for (id, source, target, relationship_type) in [
        (10, 1, 2, rel1),
        (11, 2, 1, rel2),
        (12, 1, 3, rel3),
        (13, 1, 4, rel1),
    ] {
        graph.insert_edge(EdgeInput {
            id: EdgeId(id),
            source: NodeId(source),
            target: NodeId(target),
            relationship_type,
            layer: Layer::Observed,
            revision: id,
            properties: Vec::new(),
        })?;
    }
    Ok((graph, rel1, rel2, rel3))
}

#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_pattern1_fused_visible_scan_certifies_official_scenarios_1_to_10_and_19_to_21()
-> irongraph::Result<()> {
    let _metal = metal_test_guard();
    let (graph, rel1, rel2, rel3) = official_pattern1_graph()?;
    let snapshot = Arc::new(graph.snapshot()?);
    let revision = snapshot.revision;
    let layout_version = snapshot.layout_version;
    let node_slots = snapshot.node_ids.len();
    let mut cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let mut metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    cpu.admit_graph(Arc::clone(&snapshot))?;
    metal.admit_graph(snapshot)?;

    let one_leaf = |seed, direction, relationship_types, length| {
        scan_request(
            revision,
            layout_version,
            node_slots,
            LayerMask::OBSERVED,
            program(
                seed,
                vec![leaf(seed, 0, direction, relationship_types, length)],
                vec![ResidentPatternBooleanInstruction::Leaf(0)],
            ),
            None,
        )
    };

    for (request, expected) in [
        (
            one_leaf(
                101,
                ResidentDirection::Outgoing,
                ResidentPatternRelationshipTypes::Any,
                ResidentPatternPredicateLength::One,
            ),
            vec![0, 1],
        ),
        (
            one_leaf(
                102,
                ResidentDirection::Undirected,
                ResidentPatternRelationshipTypes::Any,
                ResidentPatternPredicateLength::One,
            ),
            vec![0, 1, 2, 3],
        ),
        (
            one_leaf(
                103,
                ResidentDirection::Incoming,
                ResidentPatternRelationshipTypes::Any,
                ResidentPatternPredicateLength::One,
            ),
            vec![0, 1, 2, 3],
        ),
        (
            one_leaf(
                104,
                ResidentDirection::Outgoing,
                known(vec![rel1]),
                ResidentPatternPredicateLength::One,
            ),
            vec![0],
        ),
        (
            one_leaf(
                105,
                ResidentDirection::Undirected,
                known(vec![rel1]),
                ResidentPatternPredicateLength::One,
            ),
            vec![0, 1, 3],
        ),
        (
            one_leaf(
                106,
                ResidentDirection::Incoming,
                known(vec![rel1]),
                ResidentPatternPredicateLength::One,
            ),
            vec![1, 3],
        ),
        (
            one_leaf(
                107,
                ResidentDirection::Outgoing,
                known(vec![rel1]),
                ResidentPatternPredicateLength::OneOrMoreWitness,
            ),
            vec![0],
        ),
        (
            one_leaf(
                108,
                ResidentDirection::Undirected,
                known(vec![rel1]),
                ResidentPatternPredicateLength::OneOrMoreWitness,
            ),
            vec![0, 1, 3],
        ),
        (
            one_leaf(
                109,
                ResidentDirection::Incoming,
                known(vec![rel1]),
                ResidentPatternPredicateLength::OneOrMoreWitness,
            ),
            vec![1, 3],
        ),
        (
            one_leaf(
                110,
                ResidentDirection::Undirected,
                known(vec![rel1]),
                ResidentPatternPredicateLength::ExactTwo,
            ),
            vec![1, 3],
        ),
    ] {
        assert_cpu_metal_rows(&cpu, &metal, &request, &expected)?;
    }

    let negated = scan_request(
        revision,
        layout_version,
        node_slots,
        LayerMask::OBSERVED,
        program(
            119,
            vec![leaf(
                119,
                0,
                ResidentDirection::Undirected,
                known(vec![rel2]),
                ResidentPatternPredicateLength::One,
            )],
            vec![
                ResidentPatternBooleanInstruction::Leaf(0),
                ResidentPatternBooleanInstruction::Not,
            ],
        ),
        None,
    );
    assert_cpu_metal_rows(&cpu, &metal, &negated, &[2, 3])?;

    let conjoined = scan_request(
        revision,
        layout_version,
        node_slots,
        LayerMask::OBSERVED,
        program(
            120,
            vec![
                leaf(
                    120,
                    0,
                    ResidentDirection::Undirected,
                    known(vec![rel1]),
                    ResidentPatternPredicateLength::One,
                ),
                leaf(
                    120,
                    1,
                    ResidentDirection::Undirected,
                    known(vec![rel3]),
                    ResidentPatternPredicateLength::One,
                ),
            ],
            vec![
                ResidentPatternBooleanInstruction::Leaf(0),
                ResidentPatternBooleanInstruction::Leaf(1),
                ResidentPatternBooleanInstruction::And,
            ],
        ),
        None,
    );
    assert_cpu_metal_rows(&cpu, &metal, &conjoined, &[0])?;

    let disjoined = scan_request(
        revision,
        layout_version,
        node_slots,
        LayerMask::OBSERVED,
        program(
            121,
            vec![
                leaf(
                    121,
                    0,
                    ResidentDirection::Undirected,
                    known(vec![rel1]),
                    ResidentPatternPredicateLength::One,
                ),
                leaf(
                    121,
                    1,
                    ResidentDirection::Undirected,
                    known(vec![rel2]),
                    ResidentPatternPredicateLength::One,
                ),
            ],
            vec![
                ResidentPatternBooleanInstruction::Leaf(0),
                ResidentPatternBooleanInstruction::Leaf(1),
                ResidentPatternBooleanInstruction::Or,
            ],
        ),
        None,
    );
    assert_cpu_metal_rows(&cpu, &metal, &disjoined, &[0, 1, 3])?;
    Ok(())
}

#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_pattern1_fused_scan_visibility_never_identity_capacity_and_empty_are_bounded()
-> irongraph::Result<()> {
    let _metal = metal_test_guard();
    let mut graph = GraphStore::default();
    let rel = graph.catalog_mut().intern_relationship_type("REL")?;
    for (id, layer) in [
        (1, Layer::Observed),
        (2, Layer::Knowledge),
        (3, Layer::Observed),
        (4, Layer::Observed),
    ] {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer,
            revision: id,
            labels: Vec::new(),
            properties: Vec::new(),
        })?;
    }
    graph.insert_edge(EdgeInput {
        id: EdgeId(10),
        source: NodeId(1),
        target: NodeId(4),
        relationship_type: rel,
        layer: Layer::Observed,
        revision: 5,
        properties: Vec::new(),
    })?;
    graph.insert_edge(EdgeInput {
        id: EdgeId(11),
        source: NodeId(2),
        target: NodeId(4),
        relationship_type: rel,
        layer: Layer::Observed,
        revision: 6,
        properties: Vec::new(),
    })?;
    graph.delete_node(NodeId(3), true, 7)?;
    let snapshot = Arc::new(graph.snapshot()?);
    let revision = snapshot.revision;
    let layout_version = snapshot.layout_version;
    let node_slots = snapshot.node_ids.len();
    let mut cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let mut metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    cpu.admit_graph(Arc::clone(&snapshot))?;
    metal.admit_graph(snapshot)?;

    let visible_witness = scan_request(
        revision,
        layout_version,
        node_slots,
        LayerMask::OBSERVED,
        program(
            130,
            vec![leaf(
                130,
                0,
                ResidentDirection::Outgoing,
                known(vec![rel]),
                ResidentPatternPredicateLength::One,
            )],
            vec![ResidentPatternBooleanInstruction::Leaf(0)],
        ),
        None,
    );
    assert_cpu_metal_rows(&cpu, &metal, &visible_witness, &[0])?;

    let never = scan_request(
        revision,
        layout_version,
        node_slots,
        LayerMask::OBSERVED,
        program(
            131,
            vec![leaf(
                131,
                0,
                ResidentDirection::Undirected,
                ResidentPatternRelationshipTypes::Never,
                ResidentPatternPredicateLength::One,
            )],
            vec![ResidentPatternBooleanInstruction::Leaf(0)],
        ),
        Some(0),
    );
    assert_cpu_metal_rows(&cpu, &metal, &never, &[])?;

    let not_never = scan_request(
        revision,
        layout_version,
        node_slots,
        LayerMask::OBSERVED,
        program(
            132,
            vec![leaf(
                132,
                0,
                ResidentDirection::Undirected,
                ResidentPatternRelationshipTypes::Never,
                ResidentPatternPredicateLength::One,
            )],
            vec![
                ResidentPatternBooleanInstruction::Leaf(0),
                ResidentPatternBooleanInstruction::Not,
            ],
        ),
        None,
    );
    // Only rows zero and three are active OBSERVED nodes. NOT must not revive the hidden or
    // tombstoned slots at rows one and two.
    assert_cpu_metal_rows(&cpu, &metal, &not_never, &[0, 3])?;

    for mismatched in [
        {
            let mut request = visible_witness.clone();
            request.expected_bookmark.index += 1;
            request
        },
        {
            let mut request = visible_witness.clone();
            request.expected_graph_revision += 1;
            request
        },
        {
            let mut request = visible_witness.clone();
            request.expected_layout_version += 1;
            request
        },
    ] {
        for backend in [
            &cpu as &dyn ExecutionBackend,
            &metal as &dyn ExecutionBackend,
        ] {
            let error = backend
                .execute_pattern_predicate(&mismatched, &CancellationToken::new())
                .expect_err("stale resident image identity must fail before execution");
            assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
        }
    }

    let zero_capacity_true = scan_request(
        revision,
        layout_version,
        node_slots,
        LayerMask::OBSERVED,
        not_never.program.clone(),
        Some(0),
    );
    for backend in [
        &cpu as &dyn ExecutionBackend,
        &metal as &dyn ExecutionBackend,
    ] {
        let error = backend
            .execute_pattern_predicate(&zero_capacity_true, &CancellationToken::new())
            .expect_err("visible true scan rows cannot fit zero output capacity");
        assert_eq!(error.code, ErrorCode::ResultBudgetExceeded);
    }

    let empty = Arc::new(GraphStore::default().snapshot()?);
    let mut empty_cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let mut empty_metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    empty_cpu.admit_graph(Arc::clone(&empty))?;
    empty_metal.admit_graph(Arc::clone(&empty))?;
    let empty_request = scan_request(
        empty.revision,
        empty.layout_version,
        0,
        LayerMask::OBSERVED,
        program(
            133,
            vec![leaf(
                133,
                0,
                ResidentDirection::Outgoing,
                ResidentPatternRelationshipTypes::Any,
                ResidentPatternPredicateLength::One,
            )],
            vec![ResidentPatternBooleanInstruction::Leaf(0)],
        ),
        Some(0),
    );
    assert_cpu_metal_rows(&empty_cpu, &empty_metal, &empty_request, &[])?;
    Ok(())
}

#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_pattern1_fused_scan_rejects_same_revision_compacted_layout_replay() -> irongraph::Result<()>
{
    let _metal = metal_test_guard();
    let mut graph = GraphStore::default();
    let rel = graph.catalog_mut().intern_relationship_type("REL")?;
    for (id, revision) in [(30, 1), (10, 2), (20, 3)] {
        graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision,
            labels: Vec::new(),
            properties: Vec::new(),
        })?;
    }
    graph.insert_edge(EdgeInput {
        id: EdgeId(40),
        source: NodeId(30),
        target: NodeId(10),
        relationship_type: rel,
        layer: Layer::Observed,
        revision: 4,
        properties: Vec::new(),
    })?;
    let before = Arc::new(graph.snapshot()?);
    assert_eq!(before.layout_version, 0);
    assert_eq!(before.node_ids, [NodeId(30), NodeId(10), NodeId(20)]);

    let mut cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let mut metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    cpu.admit_graph(Arc::clone(&before))?;
    metal.admit_graph(Arc::clone(&before))?;
    let old_request = scan_request(
        before.revision,
        before.layout_version,
        before.node_ids.len(),
        LayerMask::OBSERVED,
        program(
            140,
            vec![leaf(
                140,
                0,
                ResidentDirection::Outgoing,
                known(vec![rel]),
                ResidentPatternPredicateLength::One,
            )],
            vec![ResidentPatternBooleanInstruction::Leaf(0)],
        ),
        None,
    );
    assert_cpu_metal_rows(&cpu, &metal, &old_request, &[0])?;

    let revision_before_compaction = graph.revision();
    graph.compact()?;
    assert_eq!(graph.revision(), revision_before_compaction);
    assert_eq!(graph.layout_version(), 1);
    let after = Arc::new(graph.snapshot()?);
    assert_eq!(after.revision, before.revision);
    assert_eq!(after.node_ids.len(), before.node_ids.len());
    assert_eq!(after.node_ids, [NodeId(10), NodeId(20), NodeId(30)]);
    cpu.admit_graph(Arc::clone(&after))?;
    metal.admit_graph(Arc::clone(&after))?;

    for backend in [
        &cpu as &dyn ExecutionBackend,
        &metal as &dyn ExecutionBackend,
    ] {
        let error = backend
            .execute_pattern_predicate(&old_request, &CancellationToken::new())
            .expect_err("same-revision request from pre-compaction layout must be rejected");
        assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
    }

    let new_request = scan_request(
        after.revision,
        after.layout_version,
        after.node_ids.len(),
        LayerMask::OBSERVED,
        program(
            141,
            vec![leaf(
                141,
                0,
                ResidentDirection::Outgoing,
                known(vec![rel]),
                ResidentPatternPredicateLength::One,
            )],
            vec![ResidentPatternBooleanInstruction::Leaf(0)],
        ),
        None,
    );
    assert_cpu_metal_rows(&cpu, &metal, &new_request, &[2])?;
    Ok(())
}

#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_pattern1_directions_types_boolean_and_semijoin_match_cpu() -> irongraph::Result<()> {
    let _metal = metal_test_guard();
    let (graph, knows, _likes, follows) = resident_pattern_graph()?;
    let snapshot = Arc::new(graph.snapshot()?);
    let revision = snapshot.revision;
    let mut cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let mut metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    cpu.admit_graph(Arc::clone(&snapshot))?;
    metal.admit_graph(snapshot)?;
    let rows = vec![0, 1, 2, 3, 4, 5];

    for (seed, direction) in [
        (1, ResidentDirection::Outgoing),
        (2, ResidentDirection::Incoming),
        (3, ResidentDirection::Undirected),
    ] {
        let request = request(
            revision,
            rows.clone(),
            LayerMask::OBSERVED,
            program(
                seed,
                vec![leaf(
                    seed,
                    0,
                    direction,
                    ResidentPatternRelationshipTypes::Any,
                    ResidentPatternPredicateLength::One,
                )],
                vec![ResidentPatternBooleanInstruction::Leaf(0)],
            ),
            None,
        );
        assert_cpu_metal_rows(&cpu, &metal, &request, &[0, 1, 2, 4])?;
    }

    for (seed, direction, expected) in [
        (4, ResidentDirection::Outgoing, vec![0, 1, 2, 4]),
        (5, ResidentDirection::Incoming, vec![0, 1, 4]),
        (6, ResidentDirection::Undirected, vec![0, 1, 2, 4]),
    ] {
        let request = request(
            revision,
            rows.clone(),
            LayerMask::OBSERVED,
            program(
                seed,
                vec![leaf(
                    seed,
                    0,
                    direction,
                    known(vec![knows, follows]),
                    ResidentPatternPredicateLength::One,
                )],
                vec![ResidentPatternBooleanInstruction::Leaf(0)],
            ),
            None,
        );
        assert_cpu_metal_rows(&cpu, &metal, &request, &expected)?;
    }

    let never_request = request(
        revision,
        rows.clone(),
        LayerMask::OBSERVED,
        program(
            7,
            vec![leaf(
                7,
                0,
                ResidentDirection::Outgoing,
                ResidentPatternRelationshipTypes::Never,
                ResidentPatternPredicateLength::One,
            )],
            vec![ResidentPatternBooleanInstruction::Leaf(0)],
        ),
        None,
    );
    assert_cpu_metal_rows(&cpu, &metal, &never_request, &[])?;

    let star_request = request(
        revision,
        rows.clone(),
        LayerMask::OBSERVED,
        program(
            8,
            vec![leaf(
                8,
                0,
                ResidentDirection::Outgoing,
                ResidentPatternRelationshipTypes::Any,
                ResidentPatternPredicateLength::OneOrMoreWitness,
            )],
            vec![ResidentPatternBooleanInstruction::Leaf(0)],
        ),
        None,
    );
    assert_cpu_metal_rows(&cpu, &metal, &star_request, &[0, 1, 2, 4])?;

    let boolean_request = request(
        revision,
        rows,
        LayerMask::OBSERVED,
        program(
            9,
            vec![
                leaf(
                    9,
                    0,
                    ResidentDirection::Outgoing,
                    known(vec![knows]),
                    ResidentPatternPredicateLength::One,
                ),
                leaf(
                    9,
                    1,
                    ResidentDirection::Outgoing,
                    known(vec![follows]),
                    ResidentPatternPredicateLength::One,
                ),
            ],
            // (KNOWS AND FOLLOWS) OR NOT KNOWS
            vec![
                ResidentPatternBooleanInstruction::Leaf(0),
                ResidentPatternBooleanInstruction::Leaf(1),
                ResidentPatternBooleanInstruction::And,
                ResidentPatternBooleanInstruction::Leaf(0),
                ResidentPatternBooleanInstruction::Not,
                ResidentPatternBooleanInstruction::Or,
            ],
        ),
        None,
    );
    assert_cpu_metal_rows(&cpu, &metal, &boolean_request, &[0, 2, 3, 5])?;

    let no_multiplication_request = request(
        revision,
        vec![0, 0, 4, 3, 0],
        LayerMask::OBSERVED,
        program(
            10,
            vec![leaf(
                10,
                0,
                ResidentDirection::Undirected,
                known(vec![knows]),
                ResidentPatternPredicateLength::One,
            )],
            vec![ResidentPatternBooleanInstruction::Leaf(0)],
        ),
        None,
    );
    // Three copies of row zero entered, so exactly three leave despite two parallel KNOWS edges.
    // The undirected self-loop contributes exactly one copy of outer row four.
    assert_cpu_metal_rows(&cpu, &metal, &no_multiplication_request, &[0, 0, 4, 0])?;
    Ok(())
}

#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_pattern1_exact_two_scenario10_directions_and_distinct_edges_match_cpu()
-> irongraph::Result<()> {
    let _metal = metal_test_guard();
    let (graph, rel1, rel2) = exact_two_pattern_graph()?;
    let snapshot = Arc::new(graph.snapshot()?);
    let revision = snapshot.revision;
    let mut cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let mut metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    cpu.admit_graph(Arc::clone(&snapshot))?;
    metal.admit_graph(snapshot)?;

    let exact_request = |seed, direction, rows, relationship_types, capacity| {
        request(
            revision,
            rows,
            LayerMask::OBSERVED,
            program(
                seed,
                vec![leaf(
                    seed,
                    0,
                    direction,
                    relationship_types,
                    ResidentPatternPredicateLength::ExactTwo,
                )],
                vec![ResidentPatternBooleanInstruction::Leaf(0)],
            ),
            capacity,
        )
    };

    let official = exact_request(
        30,
        ResidentDirection::Undirected,
        vec![0, 1, 2, 3],
        known(vec![rel1]),
        None,
    );
    assert_cpu_metal_rows(&cpu, &metal, &official, &[1, 3])?;

    let visible_rows = vec![
        0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 13, 14, 15, 16, 17, 18, 19, 20,
    ];
    for (seed, direction, expected) in [
        (31, ResidentDirection::Outgoing, vec![4, 5, 6, 8, 10]),
        (32, ResidentDirection::Incoming, vec![4, 5, 6, 8]),
        (
            33,
            ResidentDirection::Undirected,
            vec![1, 3, 4, 5, 6, 8, 10],
        ),
    ] {
        let direction_request = exact_request(
            seed,
            direction,
            visible_rows.clone(),
            known(vec![rel1]),
            None,
        );
        assert_cpu_metal_rows(&cpu, &metal, &direction_request, &expected)?;
    }

    let rel1_only = exact_request(
        34,
        ResidentDirection::Outgoing,
        vec![14],
        known(vec![rel1]),
        None,
    );
    assert_cpu_metal_rows(&cpu, &metal, &rel1_only, &[])?;
    let type_union = exact_request(
        35,
        ResidentDirection::Outgoing,
        vec![14],
        known(vec![rel1, rel2]),
        None,
    );
    assert_cpu_metal_rows(&cpu, &metal, &type_union, &[14])?;
    let any_type = exact_request(
        36,
        ResidentDirection::Outgoing,
        vec![14],
        ResidentPatternRelationshipTypes::Any,
        None,
    );
    assert_cpu_metal_rows(&cpu, &metal, &any_type, &[14])?;

    let no_multiplication = exact_request(
        37,
        ResidentDirection::Undirected,
        vec![1, 1, 3, 4, 4, 7, 8, 12],
        known(vec![rel1]),
        None,
    );
    // Cycles and parallel edges produce many physical paths, but each input position contributes
    // at most one output. The single loop is false; the two distinct loops retain row eight once.
    assert_cpu_metal_rows(&cpu, &metal, &no_multiplication, &[1, 1, 3, 4, 4, 8])?;
    Ok(())
}

#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_pattern1_exact_two_visibility_capacity_and_cancellation_are_bounded()
-> irongraph::Result<()> {
    let _metal = metal_test_guard();
    let (graph, rel1, _rel2) = exact_two_pattern_graph()?;
    let snapshot = Arc::new(graph.snapshot()?);
    let revision = snapshot.revision;
    let mut cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let mut metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    cpu.admit_graph(Arc::clone(&snapshot))?;
    metal.admit_graph(snapshot)?;

    let exact_request = |seed, rows, capacity| {
        request(
            revision,
            rows,
            LayerMask::OBSERVED,
            program(
                seed,
                vec![leaf(
                    seed,
                    0,
                    ResidentDirection::Undirected,
                    known(vec![rel1]),
                    ResidentPatternPredicateLength::ExactTwo,
                )],
                vec![ResidentPatternBooleanInstruction::Leaf(0)],
            ),
            capacity,
        )
    };

    let hidden_final_endpoint = exact_request(40, vec![12, 13], None);
    assert_cpu_metal_rows(&cpu, &metal, &hidden_final_endpoint, &[])?;
    let hidden_second_relationship = exact_request(45, vec![17, 18], None);
    assert_cpu_metal_rows(&cpu, &metal, &hidden_second_relationship, &[])?;
    // These are complete physical paths. Only the middle node or first relationship is hidden.
    let hidden_middle_node = exact_request(46, vec![21], None);
    assert_cpu_metal_rows(&cpu, &metal, &hidden_middle_node, &[])?;
    let hidden_first_relationship = exact_request(47, vec![24], None);
    assert_cpu_metal_rows(&cpu, &metal, &hidden_first_relationship, &[])?;

    let hidden_source = exact_request(41, vec![9], None);
    for backend in [
        &cpu as &dyn ExecutionBackend,
        &metal as &dyn ExecutionBackend,
    ] {
        let error = backend
            .execute_pattern_predicate(&hidden_source, &CancellationToken::new())
            .expect_err("an exact-two outer source hidden by its layer must fail closed");
        assert_eq!(error.code, ErrorCode::QueryType);
    }

    let empty = exact_request(42, Vec::new(), Some(0));
    assert_cpu_metal_rows(&cpu, &metal, &empty, &[])?;
    let zero_capacity_false = exact_request(43, vec![7], Some(0));
    assert_cpu_metal_rows(&cpu, &metal, &zero_capacity_false, &[])?;
    let zero_capacity_true = exact_request(44, vec![4], Some(0));
    for backend in [
        &cpu as &dyn ExecutionBackend,
        &metal as &dyn ExecutionBackend,
    ] {
        let error = backend
            .execute_pattern_predicate(&zero_capacity_true, &CancellationToken::new())
            .expect_err("a true exact-two row cannot fit a zero-row output budget");
        assert_eq!(error.code, ErrorCode::ResultBudgetExceeded);
    }

    let cancellation = CancellationToken::new();
    cancellation.cancel();
    for backend in [
        &cpu as &dyn ExecutionBackend,
        &metal as &dyn ExecutionBackend,
    ] {
        let error = backend
            .execute_pattern_predicate(&zero_capacity_true, &cancellation)
            .expect_err("pre-cancelled exact-two work must not dispatch");
        assert_eq!(error.code, ErrorCode::Cancelled);
    }
    Ok(())
}

#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_pattern1_empty_capacity_cancellation_and_source_visibility_are_bounded()
-> irongraph::Result<()> {
    let _metal = metal_test_guard();

    let empty = Arc::new(GraphStore::default().snapshot()?);
    let empty_revision = empty.revision;
    let mut empty_cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let mut empty_metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    empty_cpu.admit_graph(Arc::clone(&empty))?;
    empty_metal.admit_graph(empty)?;
    let empty_request = request(
        empty_revision,
        Vec::new(),
        LayerMask::OBSERVED,
        program(
            20,
            vec![leaf(
                20,
                0,
                ResidentDirection::Outgoing,
                ResidentPatternRelationshipTypes::Any,
                ResidentPatternPredicateLength::One,
            )],
            vec![ResidentPatternBooleanInstruction::Leaf(0)],
        ),
        Some(0),
    );
    // Both kernels still dispatch one command group and publish same-execution zero receipts.
    assert_cpu_metal_rows(&empty_cpu, &empty_metal, &empty_request, &[])?;

    let (graph, _knows, _likes, _follows) = resident_pattern_graph()?;
    let snapshot = Arc::new(graph.snapshot()?);
    let revision = snapshot.revision;
    let mut cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let mut metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    cpu.admit_graph(Arc::clone(&snapshot))?;
    metal.admit_graph(snapshot)?;

    let zero_capacity_false = request(
        revision,
        vec![3, 5],
        LayerMask::OBSERVED,
        program(
            21,
            vec![leaf(
                21,
                0,
                ResidentDirection::Outgoing,
                ResidentPatternRelationshipTypes::Any,
                ResidentPatternPredicateLength::One,
            )],
            vec![ResidentPatternBooleanInstruction::Leaf(0)],
        ),
        Some(0),
    );
    assert_cpu_metal_rows(&cpu, &metal, &zero_capacity_false, &[])?;

    let zero_capacity_true = request(
        revision,
        vec![0],
        LayerMask::OBSERVED,
        program(
            22,
            vec![leaf(
                22,
                0,
                ResidentDirection::Outgoing,
                ResidentPatternRelationshipTypes::Any,
                ResidentPatternPredicateLength::One,
            )],
            vec![ResidentPatternBooleanInstruction::Leaf(0)],
        ),
        Some(0),
    );
    for backend in [
        &cpu as &dyn ExecutionBackend,
        &metal as &dyn ExecutionBackend,
    ] {
        let error = backend
            .execute_pattern_predicate(&zero_capacity_true, &CancellationToken::new())
            .expect_err("a true row cannot fit a zero-row output budget");
        assert_eq!(error.code, ErrorCode::ResultBudgetExceeded);
    }

    let cancelled = CancellationToken::new();
    cancelled.cancel();
    for backend in [
        &cpu as &dyn ExecutionBackend,
        &metal as &dyn ExecutionBackend,
    ] {
        let error = backend
            .execute_pattern_predicate(&zero_capacity_true, &cancelled)
            .expect_err("pre-cancelled resident pattern work must not dispatch");
        assert_eq!(error.code, ErrorCode::Cancelled);
    }

    let mut layered_graph = GraphStore::default();
    let knows = layered_graph
        .catalog_mut()
        .intern_relationship_type("KNOWS")?;
    layered_graph.insert_node(NodeInput {
        id: NodeId(100),
        layer: Layer::Knowledge,
        revision: 1,
        labels: Vec::new(),
        properties: Vec::new(),
    })?;
    layered_graph.insert_node(NodeInput {
        id: NodeId(101),
        layer: Layer::Observed,
        revision: 2,
        labels: Vec::new(),
        properties: Vec::new(),
    })?;
    layered_graph.insert_edge(EdgeInput {
        id: EdgeId(100),
        source: NodeId(100),
        target: NodeId(101),
        relationship_type: knows,
        layer: Layer::Observed,
        revision: 3,
        properties: Vec::new(),
    })?;
    let layered = Arc::new(layered_graph.snapshot()?);
    let layered_revision = layered.revision;
    let mut layered_cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let mut layered_metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    layered_cpu.admit_graph(Arc::clone(&layered))?;
    layered_metal.admit_graph(layered)?;
    let hidden_source = request(
        layered_revision,
        vec![0],
        LayerMask::OBSERVED,
        program(
            23,
            vec![leaf(
                23,
                0,
                ResidentDirection::Outgoing,
                ResidentPatternRelationshipTypes::Any,
                ResidentPatternPredicateLength::One,
            )],
            vec![ResidentPatternBooleanInstruction::Leaf(0)],
        ),
        None,
    );
    for backend in [
        &layered_cpu as &dyn ExecutionBackend,
        &layered_metal as &dyn ExecutionBackend,
    ] {
        let error = backend
            .execute_pattern_predicate(&hidden_source, &CancellationToken::new())
            .expect_err("an outer source hidden by the requested layer must fail closed");
        assert_eq!(error.code, ErrorCode::QueryType);
    }
    Ok(())
}
