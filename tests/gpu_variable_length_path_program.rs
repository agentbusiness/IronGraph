// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use irongraph::{
    Bookmark, EdgeId, ErrorCode, Layer, NodeId, ProjectId, Result,
    gpu::{
        BackendKind, CpuBackend, ExecutionBackend, ResidentDeviceCompletion, ResidentDirection,
        ResidentExecutionId, ResidentExecutionObligation, ResidentObligationKind,
        ResidentObligationScope, ResidentProjectImage, ResidentVariablePathInput,
        ResidentVariablePathMultiplicityScan, ResidentVariablePathRequest,
        ResidentVariablePathSegment,
    },
    graph::{EdgeInput, GraphStore, IndexCatalog, LayerMask, NodeInput, TemporalStore},
    types::{LabelId, RelationshipTypeId},
};
use tokio_util::sync::CancellationToken;

#[cfg(all(feature = "accelerator", target_os = "macos"))]
use irongraph::gpu::MetalBackend;

const MEMORY_LIMIT_BYTES: usize = 512 * 1024 * 1024;
const RESERVED_BYTES: usize = 64 * 1024 * 1024;
const FRONTIER_CAPACITY: usize = 4_096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Shape {
    Baseline,
    Extended,
    ReversedTopExtended,
    ReversedBelowTopExtended,
}

impl Shape {
    const fn maximum_depth(self) -> usize {
        match self {
            Self::Baseline => 3,
            Self::Extended | Self::ReversedTopExtended | Self::ReversedBelowTopExtended => 4,
        }
    }

    const fn tag(self) -> u128 {
        match self {
            Self::Baseline => 101,
            Self::Extended => 102,
            Self::ReversedTopExtended => 103,
            Self::ReversedBelowTopExtended => 104,
        }
    }
}

struct Fixture {
    graph: GraphStore,
    project: ProjectId,
    bookmark: Bookmark,
    root_label: LabelId,
    likes: RelationshipTypeId,
}

fn dense(depth: usize, position: usize) -> u32 {
    ((1_usize << depth) - 1 + position) as u32
}

fn depth_rows(depth: usize) -> Vec<u32> {
    (0..(1_usize << depth))
        .map(|position| dense(depth, position))
        .collect()
}

fn depth_range_rows(minimum: usize, maximum: usize) -> Vec<u32> {
    (minimum..=maximum).flat_map(depth_rows).collect()
}

fn fixture(shape: Shape) -> Result<Fixture> {
    let mut graph = GraphStore::default();
    let labels = ["A", "B", "C", "D", "E"]
        .into_iter()
        .map(|name| graph.catalog_mut().intern_label(name))
        .collect::<Result<Vec<_>>>()?;
    let likes = graph.catalog_mut().intern_relationship_type("LIKES")?;
    for depth in 0..=shape.maximum_depth() {
        for position in 0..(1_usize << depth) {
            let row = dense(depth, position);
            graph.insert_node(NodeInput {
                id: NodeId(u64::from(row) + 1),
                layer: Layer::Observed,
                revision: u64::from(row) + 1,
                labels: vec![labels[depth]],
                properties: Vec::new(),
            })?;
        }
    }
    let mut edge_id = 1_u64;
    for depth in 1..=shape.maximum_depth() {
        for position in 0..(1_usize << depth) {
            let parent = dense(depth - 1, position / 2);
            let child = dense(depth, position);
            let reverse = match shape {
                Shape::Baseline | Shape::Extended => false,
                Shape::ReversedTopExtended => depth == 1,
                Shape::ReversedBelowTopExtended => (2..=3).contains(&depth),
            };
            let (source, target) = if reverse {
                (child, parent)
            } else {
                (parent, child)
            };
            graph.insert_edge(EdgeInput {
                id: EdgeId(edge_id),
                source: NodeId(u64::from(source) + 1),
                target: NodeId(u64::from(target) + 1),
                relationship_type: likes,
                layer: Layer::Observed,
                revision: 1_000 + edge_id,
                properties: Vec::new(),
            })?;
            edge_id += 1;
        }
    }
    Ok(Fixture {
        graph,
        project: ProjectId(uuid::Uuid::from_u128(shape.tag())),
        bookmark: Bookmark {
            term: 17,
            index: 1_000 + shape.tag() as u64,
        },
        root_label: labels[0],
        likes,
    })
}

fn fixture_with_two_independent_a_rows() -> Result<Fixture> {
    let mut fixture = fixture(Shape::Extended)?;
    fixture.graph.insert_node(NodeInput {
        id: NodeId(10_001),
        layer: Layer::Observed,
        revision: 10_001,
        labels: vec![fixture.root_label],
        properties: Vec::new(),
    })?;
    Ok(fixture)
}

fn image(fixture: &Fixture) -> Result<ResidentProjectImage> {
    ResidentProjectImage::build(
        fixture.project,
        fixture.bookmark,
        &fixture.graph,
        &TemporalStore::default(),
        &IndexCatalog::default(),
    )
}

fn obligation(
    id: u64,
    kind: ResidentObligationKind,
    scope: ResidentObligationScope,
) -> ResidentExecutionObligation {
    ResidentExecutionObligation { id, kind, scope }
}

#[derive(Clone, Copy)]
struct SegmentSpec {
    direction: ResidentDirection,
    minimum: u32,
    maximum: Option<u32>,
}

fn exact(direction: ResidentDirection, hops: u32) -> SegmentSpec {
    SegmentSpec {
        direction,
        minimum: hops,
        maximum: Some(hops),
    }
}

fn ranged(direction: ResidentDirection, minimum: u32, maximum: Option<u32>) -> SegmentSpec {
    SegmentSpec {
        direction,
        minimum,
        maximum,
    }
}

fn scenario(id: u8) -> (Shape, bool, Vec<SegmentSpec>, Vec<u32>) {
    use ResidentDirection::{Incoming, Outgoing, Undirected};
    match id {
        1 | 2 => (
            Shape::Baseline,
            true,
            vec![ranged(Outgoing, 1, None)],
            depth_range_rows(1, 3),
        ),
        3 => (
            Shape::Baseline,
            true,
            vec![exact(Outgoing, 0)],
            depth_rows(0),
        ),
        4 => (
            Shape::Baseline,
            true,
            vec![exact(Outgoing, 1)],
            depth_rows(1),
        ),
        5 => (
            Shape::Baseline,
            true,
            vec![exact(Outgoing, 2)],
            depth_rows(2),
        ),
        6 => (
            Shape::Baseline,
            true,
            vec![ranged(Outgoing, 0, Some(2))],
            depth_range_rows(0, 2),
        ),
        7 => (
            Shape::Baseline,
            true,
            vec![ranged(Outgoing, 1, Some(2))],
            depth_range_rows(1, 2),
        ),
        8 => (
            Shape::Baseline,
            true,
            vec![exact(Outgoing, 0)],
            depth_rows(0),
        ),
        9 => (
            Shape::Baseline,
            true,
            vec![exact(Outgoing, 1)],
            depth_rows(1),
        ),
        10 => (
            Shape::Baseline,
            true,
            vec![exact(Outgoing, 2)],
            depth_rows(2),
        ),
        11 => (
            Shape::Baseline,
            true,
            vec![ranged(Outgoing, 2, Some(1))],
            Vec::new(),
        ),
        12 => (
            Shape::Baseline,
            true,
            vec![ranged(Outgoing, 1, Some(0))],
            Vec::new(),
        ),
        13 => (
            Shape::Baseline,
            true,
            vec![ranged(Outgoing, 1, Some(0))],
            Vec::new(),
        ),
        14 => (
            Shape::Baseline,
            true,
            vec![exact(Outgoing, 1)],
            depth_rows(1),
        ),
        15 => (
            Shape::Baseline,
            true,
            vec![ranged(Outgoing, 1, Some(2))],
            depth_range_rows(1, 2),
        ),
        16 => (
            Shape::Baseline,
            true,
            vec![ranged(Outgoing, 0, None)],
            depth_range_rows(0, 3),
        ),
        17 => (
            Shape::Baseline,
            true,
            vec![ranged(Outgoing, 1, None)],
            depth_range_rows(1, 3),
        ),
        18 => (
            Shape::Baseline,
            true,
            vec![ranged(Outgoing, 2, None)],
            depth_range_rows(2, 3),
        ),
        19 => (
            Shape::Baseline,
            true,
            vec![exact(Outgoing, 0), exact(Outgoing, 1)],
            depth_rows(1),
        ),
        20 => (
            Shape::Baseline,
            true,
            vec![exact(Outgoing, 1), exact(Outgoing, 0)],
            depth_rows(1),
        ),
        21 | 22 => (
            Shape::Baseline,
            true,
            vec![exact(Outgoing, 1), exact(Outgoing, 1)],
            depth_rows(2),
        ),
        23 => (
            Shape::Baseline,
            true,
            vec![exact(Outgoing, 2), exact(Outgoing, 1)],
            depth_rows(3),
        ),
        24 => (
            Shape::Baseline,
            true,
            vec![exact(Outgoing, 1), exact(Outgoing, 2)],
            depth_rows(3),
        ),
        25 => (
            Shape::Extended,
            true,
            vec![exact(Outgoing, 1), exact(Outgoing, 3)],
            depth_rows(4),
        ),
        26 => (
            Shape::ReversedTopExtended,
            true,
            vec![exact(Incoming, 1), exact(Outgoing, 3)],
            depth_rows(4),
        ),
        27 => (
            Shape::ReversedBelowTopExtended,
            true,
            vec![exact(Outgoing, 1), exact(Undirected, 3)],
            depth_rows(4),
        ),
        28 => (
            Shape::Extended,
            false,
            vec![exact(Outgoing, 1), exact(Outgoing, 1), exact(Outgoing, 2)],
            depth_rows(4),
        ),
        29 => (
            Shape::Extended,
            false,
            vec![exact(Outgoing, 1), exact(Outgoing, 2), exact(Outgoing, 1)],
            depth_rows(4),
        ),
        _ => panic!("unknown Match5 scenario {id}"),
    }
}

fn request(
    fixture: &Fixture,
    id: u8,
    root_only: bool,
    specs: &[SegmentSpec],
    frontier_capacity: usize,
    output_capacity: usize,
) -> Result<ResidentVariablePathRequest> {
    let graph = fixture.graph.snapshot()?;
    let base = u64::from(id) * 1_000;
    let input = ResidentVariablePathInput::VisibleNodeScan {
        node_slots: graph.node_ids.len(),
        labels: if root_only {
            vec![fixture.root_label]
        } else {
            Vec::new()
        },
        predicates: Vec::new(),
        obligation: obligation(
            base + 1,
            ResidentObligationKind::PatternScan,
            ResidentObligationScope::PatternScan,
        ),
    };
    let multiplicity_scans = if root_only {
        Vec::new()
    } else {
        vec![ResidentVariablePathMultiplicityScan {
            labels: vec![fixture.root_label],
            predicates: Vec::new(),
            obligation: obligation(
                base + 2,
                ResidentObligationKind::PatternScan,
                ResidentObligationScope::PatternScan,
            ),
        }]
    };
    let segments = specs
        .iter()
        .enumerate()
        .map(|(index, spec)| ResidentVariablePathSegment {
            direction: spec.direction,
            relationship_types: vec![fixture.likes],
            relationship_types_known_empty: false,
            relationship_integer_predicates: Vec::new(),
            target_labels: Vec::new(),
            target_labels_known_empty: false,
            target_predicates: Vec::new(),
            target_equals_path_start: false,
            minimum_hops: spec.minimum,
            maximum_hops: spec.maximum,
            obligation: obligation(
                base + 10 + index as u64,
                ResidentObligationKind::PatternTraversal,
                ResidentObligationScope::PatternLeaf(index as u16),
            ),
        })
        .collect();
    let request = ResidentVariablePathRequest {
        project: fixture.project,
        expected_bookmark: fixture.bookmark,
        expected_graph_revision: graph.revision,
        expected_layout_version: graph.layout_version,
        expected_node_slots: graph.node_ids.len(),
        expected_edge_slots: graph.edge_ids.len(),
        layers: LayerMask::OBSERVED,
        multiplicity_scans,
        bound_terminal_scan: None,
        cartesian_obligation: (!root_only).then(|| {
            obligation(
                base + 3,
                ResidentObligationKind::PatternCartesian,
                ResidentObligationScope::PatternCartesian,
            )
        }),
        input,
        execution: ResidentExecutionId {
            high: 0x5650,
            low: u64::from(id),
        },
        segments,
        optional: false,
        output_limit: None,
        final_obligation: obligation(
            base + 999,
            ResidentObligationKind::PatternFilter,
            ResidentObligationScope::PatternFinal,
        ),
        final_projection: irongraph::gpu::ResidentVariablePathFinalProjection::Publications,
        maximum_frontier_paths: frontier_capacity,
        maximum_output_rows: output_capacity,
        distinct_endpoints: false,
    };
    request.validate()?;
    Ok(request)
}

fn execute_endpoints(
    backend: &dyn ExecutionBackend,
    request: &ResidentVariablePathRequest,
) -> Result<(Vec<u32>, Vec<irongraph::gpu::ResidentVariablePath>)> {
    assert!(backend.supports_native_variable_path());
    let result = backend.execute_variable_path(request, &CancellationToken::new())?;
    let validated = result.validate_for_publication(request, backend.kind())?;
    assert_eq!(validated.execution(), request.execution);
    assert_eq!(validated.request_fingerprint(), request.fingerprint()?);
    assert_eq!(
        validated.completion(),
        match backend.kind() {
            BackendKind::Cpu => ResidentDeviceCompletion::CpuReference,
            BackendKind::Metal => ResidentDeviceCompletion::Metal,
            BackendKind::Cuda => unreachable!(),
        }
    );
    let paths = validated
        .rows()
        .iter()
        .map(|row| row.path.as_ref().expect("mandatory path row is matched"))
        .cloned()
        .collect::<Vec<_>>();
    let mut endpoints = paths
        .iter()
        .map(|path| path.end().expect("validated path has an endpoint"))
        .collect::<Vec<_>>();
    endpoints.sort_unstable();
    Ok((endpoints, paths))
}

fn cpu(fixture: &Fixture) -> Result<CpuBackend> {
    let mut backend = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
    backend.admit_project(image(fixture)?)?;
    Ok(backend)
}

#[test]
fn cpu_native_stage_covers_all_29_match5_path_shapes_with_exact_receipts() -> Result<()> {
    for id in 1_u8..=29 {
        let (shape, root_only, specs, mut expected) = scenario(id);
        let fixture = fixture(shape)?;
        let backend = cpu(&fixture)?;
        let request = request(
            &fixture,
            id,
            root_only,
            &specs,
            FRONTIER_CAPACITY,
            FRONTIER_CAPACITY,
        )?;
        let (actual, paths) = execute_endpoints(&backend, &request)?;
        expected.sort_unstable();
        assert_eq!(actual, expected, "Match5 scenario {id} endpoints");
        assert!(
            paths.iter().all(|path| path
                .relationships
                .iter()
                .collect::<std::collections::BTreeSet<_>>()
                .len()
                == path.relationships.len()),
            "Match5 scenario {id} reused a relationship"
        );
    }
    Ok(())
}

#[test]
fn cpu_optional_path_null_extends_when_minimum_exceeds_the_resident_edge_domain() -> Result<()> {
    let fixture = fixture(Shape::Baseline)?;
    let backend = cpu(&fixture)?;
    let edge_slots = fixture.graph.snapshot()?.edge_ids.len();
    let minimum = u32::try_from(edge_slots + 1).expect("small fixture edge count");
    let mut request = request(
        &fixture,
        30,
        true,
        &[ranged(ResidentDirection::Outgoing, minimum, None)],
        16,
        16,
    )?;
    request.optional = true;
    request.validate()?;

    let validated = backend
        .execute_variable_path(&request, &CancellationToken::new())?
        .validate_for_publication(&request, BackendKind::Cpu)?;
    assert_eq!(validated.input_cardinality(), 1);
    assert_eq!(validated.rows().len(), 1);
    let row = &validated.rows()[0];
    assert_eq!(row.parent_row, 0);
    assert_eq!(row.start, 0);
    assert_eq!(row.bound_terminal, None);
    assert_eq!(row.path, None);

    let receipts = validated.receipts();
    assert_eq!(receipts.len(), 3);
    assert_eq!(
        (
            receipts[1].input_cardinality,
            receipts[1].output_cardinality
        ),
        (1, 0),
        "the valid empty segment must publish zero candidates"
    );
    assert_eq!(
        (
            receipts[2].input_cardinality,
            receipts[2].output_cardinality
        ),
        (0, 1),
        "OPTIONAL finalization must publish one null-extension row"
    );
    Ok(())
}

fn verify_two_a_cartesian_multiplicity(
    backend: &dyn ExecutionBackend,
    fixture: &Fixture,
) -> Result<Vec<Vec<irongraph::gpu::ResidentVariablePath>>> {
    let node_slots = fixture.graph.snapshot()?.node_ids.len() as u64;
    assert_eq!(
        node_slots, 32,
        "the adversarial fixture must contain 32 visible nodes"
    );
    let mut scenario_paths = Vec::new();
    for id in [28_u8, 29] {
        let (_, root_only, specs, _) = scenario(id);
        assert!(
            !root_only,
            "scenario {id} must retain its independent A scan"
        );
        let request = request(
            fixture,
            id,
            root_only,
            &specs,
            FRONTIER_CAPACITY,
            FRONTIER_CAPACITY,
        )?;
        let raw = backend.execute_variable_path(&request, &CancellationToken::new())?;
        let validated = raw.validate_for_publication(&request, backend.kind())?;
        assert_eq!(validated.input_cardinality(), 64);

        let receipts = validated.receipts();
        assert_eq!(receipts[0].input_cardinality, 32);
        assert_eq!(receipts[0].output_cardinality, 2, "scenario {id} A scan");
        assert_eq!(receipts[1].input_cardinality, 32);
        assert_eq!(receipts[1].output_cardinality, 32, "scenario {id} p scan");
        assert_eq!(receipts[2].input_cardinality, 64);
        assert_eq!(
            receipts[2].output_cardinality, 64,
            "scenario {id} Cartesian"
        );

        let paths = validated
            .rows()
            .iter()
            .map(|row| row.path.as_ref().expect("mandatory path row is matched"))
            .cloned()
            .collect::<Vec<_>>();
        let mut actual = paths
            .iter()
            .map(|path| path.end().expect("validated path has an endpoint"))
            .collect::<Vec<_>>();
        actual.sort_unstable();
        let mut expected = depth_rows(4);
        expected.extend(depth_rows(4));
        expected.sort_unstable();
        assert_eq!(
            actual, expected,
            "scenario {id} must preserve both A bindings"
        );
        assert_eq!(paths.len(), 32);
        scenario_paths.push(paths);
    }
    Ok(scenario_paths)
}

#[test]
fn cpu_preserves_independent_visible_scan_cartesian_multiplicity_for_match5_28_and_29() -> Result<()>
{
    let fixture = fixture_with_two_independent_a_rows()?;
    let backend = cpu(&fixture)?;
    verify_two_a_cartesian_multiplicity(&backend, &fixture)?;
    Ok(())
}

#[test]
fn cpu_stage_rejects_stale_generation_capacity_and_cancellation_without_publication() -> Result<()>
{
    let fixture = fixture(Shape::Baseline)?;
    let backend = cpu(&fixture)?;
    let specs = [exact(ResidentDirection::Outgoing, 1)];

    let mut stale = request(&fixture, 4, true, &specs, 16, 16)?;
    stale.expected_bookmark.index += 1;
    assert_eq!(
        backend
            .execute_variable_path(&stale, &CancellationToken::new())
            .expect_err("stale generation must fail")
            .code,
        ErrorCode::GpuAdmissionFailure
    );

    let capacity = request(&fixture, 4, true, &specs, 1, 1)?;
    assert_eq!(
        backend
            .execute_variable_path(&capacity, &CancellationToken::new())
            .expect_err("two children cannot fit one frontier slot")
            .code,
        ErrorCode::ResultBudgetExceeded
    );

    let cancelled = CancellationToken::new();
    cancelled.cancel();
    let normal = request(&fixture, 4, true, &specs, 16, 16)?;
    assert_eq!(
        backend
            .execute_variable_path(&normal, &cancelled)
            .expect_err("pre-cancelled path command must fail")
            .code,
        ErrorCode::Cancelled
    );
    Ok(())
}

#[test]
fn cpu_receipt_cannot_masquerade_as_metal_completion() -> Result<()> {
    let fixture = fixture(Shape::Baseline)?;
    let backend = cpu(&fixture)?;
    let request = request(
        &fixture,
        4,
        true,
        &[exact(ResidentDirection::Outgoing, 1)],
        16,
        16,
    )?;
    let raw = backend.execute_variable_path(&request, &CancellationToken::new())?;
    assert!(
        raw.validate_for_publication(&request, BackendKind::Metal)
            .is_err()
    );
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "hardware acceptance gate: requires a real Metal device"]
fn real_metal_matches_cpu_for_all_29_match5_path_shapes_without_fallback() -> Result<()> {
    for id in 1_u8..=29 {
        let (shape, root_only, specs, mut expected) = scenario(id);
        let fixture = fixture(shape)?;
        let cpu = cpu(&fixture)?;
        let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
        metal.admit_project(image(&fixture)?)?;
        let request = request(
            &fixture,
            id,
            root_only,
            &specs,
            FRONTIER_CAPACITY,
            FRONTIER_CAPACITY,
        )?;
        let (cpu_endpoints, cpu_paths) = execute_endpoints(&cpu, &request)?;
        let (metal_endpoints, metal_paths) = execute_endpoints(&metal, &request)?;
        expected.sort_unstable();
        assert_eq!(
            metal_endpoints, expected,
            "Match5 scenario {id} Metal endpoints"
        );
        assert_eq!(
            metal_endpoints, cpu_endpoints,
            "Match5 scenario {id} parity"
        );
        assert_eq!(
            metal_paths, cpu_paths,
            "Match5 scenario {id} stable path order"
        );
    }
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "hardware acceptance gate: requires a real Metal device"]
fn real_metal_preserves_two_a_cartesian_multiplicity_for_match5_28_and_29() -> Result<()> {
    let fixture = fixture_with_two_independent_a_rows()?;
    let cpu = cpu(&fixture)?;
    let cpu_paths = verify_two_a_cartesian_multiplicity(&cpu, &fixture)?;
    let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
    metal.admit_project(image(&fixture)?)?;
    let metal_paths = verify_two_a_cartesian_multiplicity(&metal, &fixture)?;
    assert_eq!(metal_paths, cpu_paths, "two-A stable CPU/Metal path parity");
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "hardware acceptance gate: requires a real Metal device"]
fn real_metal_null_extends_an_effectively_empty_optional_path() -> Result<()> {
    let fixture = fixture(Shape::Baseline)?;
    let cpu = cpu(&fixture)?;
    let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
    metal.admit_project(image(&fixture)?)?;
    let edge_slots = fixture.graph.snapshot()?.edge_ids.len();
    let minimum = u32::try_from(edge_slots + 1).expect("small fixture edge count");
    let mut request = request(
        &fixture,
        30,
        true,
        &[ranged(ResidentDirection::Outgoing, minimum, None)],
        16,
        16,
    )?;
    request.optional = true;
    request.validate()?;

    let cpu_result = cpu
        .execute_variable_path(&request, &CancellationToken::new())?
        .validate_for_publication(&request, BackendKind::Cpu)?;
    let metal_result = metal
        .execute_variable_path(&request, &CancellationToken::new())?
        .validate_for_publication(&request, BackendKind::Metal)?;
    assert_eq!(metal_result.rows(), cpu_result.rows());
    assert_eq!(metal_result.input_cardinality(), 1);
    assert_eq!(
        metal_result
            .receipts()
            .iter()
            .map(|receipt| (receipt.input_cardinality, receipt.output_cardinality))
            .collect::<Vec<_>>(),
        cpu_result
            .receipts()
            .iter()
            .map(|receipt| (receipt.input_cardinality, receipt.output_cardinality))
            .collect::<Vec<_>>()
    );
    Ok(())
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "hardware acceptance gate: requires a real Metal device"]
fn real_metal_rejects_capacity_stale_generation_and_pre_cancel() -> Result<()> {
    let fixture = fixture(Shape::Baseline)?;
    let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
    metal.admit_project(image(&fixture)?)?;
    let specs = [exact(ResidentDirection::Outgoing, 1)];
    let capacity = request(&fixture, 4, true, &specs, 1, 1)?;
    assert_eq!(
        metal
            .execute_variable_path(&capacity, &CancellationToken::new())
            .expect_err("Metal frontier overflow must fail")
            .code,
        ErrorCode::ResultBudgetExceeded
    );
    let mut stale = request(&fixture, 4, true, &specs, 16, 16)?;
    stale.expected_graph_revision += 1;
    assert_eq!(
        metal
            .execute_variable_path(&stale, &CancellationToken::new())
            .expect_err("stale Metal generation must fail")
            .code,
        ErrorCode::GpuAdmissionFailure
    );
    let cancelled = CancellationToken::new();
    cancelled.cancel();
    let normal = request(&fixture, 4, true, &specs, 16, 16)?;
    assert_eq!(
        metal
            .execute_variable_path(&normal, &cancelled)
            .expect_err("pre-cancelled Metal command must fail")
            .code,
        ErrorCode::Cancelled
    );
    Ok(())
}
