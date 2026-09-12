// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use irongraph::{
    Bookmark, EdgeId, ErrorCode, Layer, NodeId, ProjectId, Result, ScalarValue,
    gpu::{
        BackendKind, CpuBackend, ExecutionBackend, ResidentDeviceCompletion, ResidentDirection,
        ResidentExecutionId, ResidentExecutionObligation, ResidentObligationKind,
        ResidentObligationScope, ResidentProjectImage, ResidentQuantifierBinary,
        ResidentQuantifierEntityHandle, ResidentQuantifierEntityKind, ResidentQuantifierEntityList,
        ResidentQuantifierEntityProperty, ResidentQuantifierEntityPropertyShape,
        ResidentQuantifierExpression as Expr, ResidentQuantifierGeneration, ResidentQuantifierKind,
        ResidentQuantifierOutput, ResidentQuantifierProgram, ResidentQuantifierProgramRequest,
        ResidentQuantifierProjection, ResidentQuantifierSlot, ResidentQuantifierSource,
        ResidentQuantifierStage, ResidentQuantifierValue as Value, ResidentVariablePathInput,
        ResidentVariablePathRequest, ResidentVariablePathSegment,
    },
    graph::{EdgeInput, GraphStore, IndexCatalog, LayerMask, NodeInput, TemporalStore},
    types::{LabelId, PropertyId, RelationshipTypeId},
};
use tokio_util::sync::CancellationToken;

const MEMORY_LIMIT: usize = 256 * 1024 * 1024;
const RESERVED_MEMORY: usize = 16 * 1024 * 1024;

struct Fixture {
    graph: GraphStore,
    project: ProjectId,
    bookmark: Bookmark,
    node_start: LabelId,
    relationship_start: LabelId,
    name: PropertyId,
    leading: RelationshipTypeId,
    branch_a: RelationshipTypeId,
    branch_b: RelationshipTypeId,
}

impl Fixture {
    fn generation(&self) -> ResidentQuantifierGeneration {
        ResidentQuantifierGeneration {
            project: self.project,
            bookmark: self.bookmark,
            graph_revision: self.graph.revision(),
            layout_version: self.graph.layout_version(),
            catalog_generation: self.graph.catalog().optimizer_generation(),
        }
    }

    fn backend(&self) -> Result<CpuBackend> {
        let image = ResidentProjectImage::build(
            self.project,
            self.bookmark,
            &self.graph,
            &TemporalStore::default(),
            &IndexCatalog::default(),
        )?;
        let mut backend = CpuBackend::new(MEMORY_LIMIT, RESERVED_MEMORY);
        backend.admit_project(image)?;
        Ok(backend)
    }
}

fn node(
    graph: &mut GraphStore,
    id: u64,
    labels: Vec<LabelId>,
    properties: Vec<(PropertyId, ScalarValue)>,
) -> Result<()> {
    graph.insert_node(NodeInput {
        id: NodeId(id),
        layer: Layer::Observed,
        revision: 1,
        labels,
        properties,
    })?;
    Ok(())
}

fn edge(
    graph: &mut GraphStore,
    id: u64,
    source: u64,
    target: u64,
    relationship_type: RelationshipTypeId,
    properties: Vec<(PropertyId, ScalarValue)>,
) -> Result<()> {
    graph.insert_edge(EdgeInput {
        id: EdgeId(id),
        source: NodeId(source),
        target: NodeId(target),
        relationship_type,
        layer: Layer::Observed,
        revision: 1,
        properties,
    })?;
    Ok(())
}

fn binary_fixture() -> Result<Fixture> {
    let mut graph = GraphStore::default();
    let node_start = graph.catalog_mut().intern_label("SNodes")?;
    let relationship_start = graph.catalog_mut().intern_label("SRelationships")?;
    let name = graph.catalog_mut().intern_property("name")?;
    let leading = graph.catalog_mut().intern_relationship_type("I")?;
    let branch_a = graph.catalog_mut().intern_relationship_type("RA")?;
    let branch_b = graph.catalog_mut().intern_relationship_type("RB")?;

    node(&mut graph, 1, vec![relationship_start], Vec::new())?;
    node(&mut graph, 2, vec![node_start], Vec::new())?;
    edge(&mut graph, 1, 1, 2, leading, Vec::new())?;

    let mut parents = vec![2_u64];
    let mut next_node = 3_u64;
    let mut next_edge = 2_u64;
    for _depth in 1..=3 {
        let mut children = Vec::with_capacity(parents.len() * 2);
        for parent in parents {
            for (relationship_type, value) in [(branch_a, "a"), (branch_b, "b")] {
                let child = next_node;
                next_node += 1;
                node(
                    &mut graph,
                    child,
                    Vec::new(),
                    vec![(name, ScalarValue::String(value.into()))],
                )?;
                edge(
                    &mut graph,
                    next_edge,
                    parent,
                    child,
                    relationship_type,
                    vec![(name, ScalarValue::String(value.into()))],
                )?;
                next_edge += 1;
                children.push(child);
            }
        }
        parents = children;
    }
    let project = ProjectId(uuid::Uuid::from_u128(0x5155_454e_5449_5459_0001));
    let bookmark = Bookmark {
        term: 41,
        index: graph.revision(),
    };
    Ok(Fixture {
        graph,
        project,
        bookmark,
        node_start,
        relationship_start,
        name,
        leading,
        branch_a,
        branch_b,
    })
}

fn three_valued_chain_fixture() -> Result<Fixture> {
    let mut graph = GraphStore::default();
    let node_start = graph.catalog_mut().intern_label("SNodes")?;
    let relationship_start = graph.catalog_mut().intern_label("SRelationships")?;
    let name = graph.catalog_mut().intern_property("name")?;
    let leading = graph.catalog_mut().intern_relationship_type("I")?;
    let branch_a = graph.catalog_mut().intern_relationship_type("RA")?;
    let branch_b = graph.catalog_mut().intern_relationship_type("RB")?;

    node(&mut graph, 101, vec![relationship_start], Vec::new())?;
    node(&mut graph, 102, vec![node_start], Vec::new())?;
    node(
        &mut graph,
        103,
        Vec::new(),
        vec![(name, ScalarValue::String("a".into()))],
    )?;
    node(
        &mut graph,
        104,
        Vec::new(),
        vec![(name, ScalarValue::Integer(7))],
    )?;
    node(&mut graph, 105, Vec::new(), Vec::new())?;
    edge(&mut graph, 101, 101, 102, leading, Vec::new())?;
    edge(
        &mut graph,
        102,
        102,
        103,
        branch_a,
        vec![(name, ScalarValue::String("a".into()))],
    )?;
    edge(
        &mut graph,
        103,
        103,
        104,
        branch_a,
        vec![(name, ScalarValue::Integer(7))],
    )?;
    edge(&mut graph, 104, 104, 105, branch_a, Vec::new())?;
    let project = ProjectId(uuid::Uuid::from_u128(0x5155_454e_5449_5459_0002));
    let bookmark = Bookmark {
        term: 42,
        index: graph.revision(),
    };
    Ok(Fixture {
        graph,
        project,
        bookmark,
        node_start,
        relationship_start,
        name,
        leading,
        branch_a,
        branch_b,
    })
}

fn obligation(
    id: u64,
    kind: ResidentObligationKind,
    scope: ResidentObligationScope,
) -> ResidentExecutionObligation {
    ResidentExecutionObligation { id, kind, scope }
}

fn path_request(
    fixture: &Fixture,
    execution: ResidentExecutionId,
    entity_kind: ResidentQuantifierEntityKind,
    minimum_hops: u32,
    maximum_hops: u32,
) -> Result<ResidentVariablePathRequest> {
    let snapshot = fixture.graph.snapshot()?;
    let base = 0x2000_0000_u64 + execution.low * 16;
    let (label, mut relationship_types) = match entity_kind {
        ResidentQuantifierEntityKind::Node => {
            (fixture.node_start, vec![fixture.branch_a, fixture.branch_b])
        }
        ResidentQuantifierEntityKind::Relationship => (
            fixture.relationship_start,
            vec![fixture.leading, fixture.branch_a, fixture.branch_b],
        ),
    };
    relationship_types.sort_unstable_by_key(|relationship_type| relationship_type.0);
    relationship_types.dedup();
    let request = ResidentVariablePathRequest {
        project: fixture.project,
        expected_bookmark: fixture.bookmark,
        expected_graph_revision: snapshot.revision,
        expected_layout_version: snapshot.layout_version,
        expected_node_slots: snapshot.node_ids.len(),
        expected_edge_slots: snapshot.edge_ids.len(),
        layers: LayerMask::OBSERVED,
        multiplicity_scans: Vec::new(),
        bound_terminal_scan: None,
        cartesian_obligation: None,
        input: ResidentVariablePathInput::VisibleNodeScan {
            node_slots: snapshot.node_ids.len(),
            labels: vec![label],
            predicates: Vec::new(),
            obligation: obligation(
                base + 1,
                ResidentObligationKind::PatternScan,
                ResidentObligationScope::PatternScan,
            ),
        },
        execution,
        segments: vec![ResidentVariablePathSegment {
            direction: ResidentDirection::Outgoing,
            relationship_types,
            relationship_types_known_empty: false,
            relationship_integer_predicates: Vec::new(),
            target_labels: Vec::new(),
            target_labels_known_empty: false,
            target_predicates: Vec::new(),
            target_equals_path_start: false,
            minimum_hops,
            maximum_hops: Some(maximum_hops),
            obligation: obligation(
                base + 2,
                ResidentObligationKind::PatternTraversal,
                ResidentObligationScope::PatternLeaf(0),
            ),
        }],
        optional: false,
        output_limit: None,
        final_obligation: obligation(
            base + 3,
            ResidentObligationKind::PatternFilter,
            ResidentObligationScope::PatternFinal,
        ),
        final_projection: irongraph::gpu::ResidentVariablePathFinalProjection::Publications,
        maximum_frontier_paths: 256,
        maximum_output_rows: 64,
        distinct_endpoints: false,
    };
    request.validate()?;
    Ok(request)
}

fn slot(index: u16) -> Expr {
    Expr::Slot(ResidentQuantifierSlot(index))
}

fn quantifier(kind: ResidentQuantifierKind, list_slot: u16) -> Expr {
    Expr::Predicate {
        kind,
        variable: ResidentQuantifierSlot(15),
        list: Box::new(slot(list_slot)),
        predicate: Box::new(Expr::Binary {
            left: Box::new(Expr::Property {
                source: Box::new(slot(15)),
                key: "name".to_owned(),
            }),
            operation: ResidentQuantifierBinary::Equal,
            right: Box::new(Expr::Literal(Value::String("a".to_owned()))),
        }),
    }
}

fn program(entity_kind: ResidentQuantifierEntityKind) -> ResidentQuantifierProgram {
    let mut stages = Vec::new();
    let list_slot = match entity_kind {
        ResidentQuantifierEntityKind::Node => 0,
        ResidentQuantifierEntityKind::Relationship => {
            stages.push(ResidentQuantifierStage::GroupCount {
                groups: vec![ResidentQuantifierProjection {
                    output: ResidentQuantifierSlot(1),
                    expression: slot(0),
                }],
                count_outputs: vec![ResidentQuantifierSlot(2)],
            });
            1
        }
    };
    stages.push(ResidentQuantifierStage::Project {
        keep_scope: false,
        bindings: vec![
            ResidentQuantifierProjection {
                output: ResidentQuantifierSlot(3),
                expression: slot(list_slot),
            },
            ResidentQuantifierProjection {
                output: ResidentQuantifierSlot(4),
                expression: quantifier(ResidentQuantifierKind::All, list_slot),
            },
            ResidentQuantifierProjection {
                output: ResidentQuantifierSlot(5),
                expression: quantifier(ResidentQuantifierKind::Any, list_slot),
            },
            ResidentQuantifierProjection {
                output: ResidentQuantifierSlot(6),
                expression: quantifier(ResidentQuantifierKind::None, list_slot),
            },
            ResidentQuantifierProjection {
                output: ResidentQuantifierSlot(7),
                expression: quantifier(ResidentQuantifierKind::Single, list_slot),
            },
        ],
    });
    ResidentQuantifierProgram {
        slot_count: 16,
        stages,
        outputs: [
            ("entities", 3),
            ("all", 4),
            ("any", 5),
            ("none", 6),
            ("single", 7),
        ]
        .into_iter()
        .map(|(name, source)| ResidentQuantifierOutput {
            name: name.to_owned(),
            source: ResidentQuantifierSlot(source),
        })
        .collect(),
    }
}

fn request(
    fixture: &Fixture,
    low: u64,
    entity_kind: ResidentQuantifierEntityKind,
    minimum_hops: u32,
    maximum_hops: u32,
) -> Result<ResidentQuantifierProgramRequest> {
    let execution = ResidentExecutionId {
        high: 0x4350_5551_454e_5449,
        low,
    };
    let path = path_request(fixture, execution, entity_kind, minimum_hops, maximum_hops)?;
    let values = match entity_kind {
        ResidentQuantifierEntityKind::Node => fixture
            .graph
            .nodes()
            .filter_map(|node| node.property(fixture.name))
            .collect::<Vec<_>>(),
        ResidentQuantifierEntityKind::Relationship => fixture
            .graph
            .edges()
            .filter_map(|relationship| relationship.property(fixture.name))
            .collect::<Vec<_>>(),
    };
    let shape = ResidentQuantifierEntityPropertyShape::from_values(values)?
        .expect("fixture uses only resident quantifier scalar property shapes");
    ResidentQuantifierProgramRequest::build_with_source(
        fixture.generation(),
        execution,
        ResidentQuantifierSource::VariablePathEntityList {
            path,
            output: ResidentQuantifierSlot(0),
            entity_kind,
            skip: 1,
            properties: vec![ResidentQuantifierEntityProperty {
                key: "name".to_owned(),
                property: Some(fixture.name),
                shape,
            }],
            materialize_obligation: obligation(
                0x3000_0000 + low,
                ResidentObligationKind::Expression,
                ResidentObligationScope::Expression(u16::MAX),
            ),
        },
        program(entity_kind),
        64,
        16,
        0x5eed + low,
    )
}

fn boolean(value: &Value) -> Option<bool> {
    match value {
        Value::Boolean(value) => Some(*value),
        Value::Null => None,
        other => panic!("expected Boolean/null quantifier output, got {other:?}"),
    }
}

#[test]
fn cpu_fused_source_matches_the_exact_node_and_relationship_tck_distributions() -> Result<()> {
    let fixture = binary_fixture()?;
    let backend = fixture.backend()?;
    assert!(backend.supports_native_quantifier_program());
    assert!(backend.supports_native_quantifier_entity_source());

    for (low, entity_kind, maximum_hops, source_rows, dependency_rows) in [
        (1, ResidentQuantifierEntityKind::Node, 3, 15_u64, 43_usize),
        (
            2,
            ResidentQuantifierEntityKind::Relationship,
            4,
            16_u64,
            45_usize,
        ),
    ] {
        let request = request(&fixture, low, entity_kind, 0, maximum_hops)?;
        let source_receipts = request.source_receipt_count();
        let path_receipts = match &request.source {
            ResidentQuantifierSource::VariablePathEntityList { path, .. } => {
                path.obligations().len()
            }
            ResidentQuantifierSource::Unit | ResidentQuantifierSource::Range { .. } => {
                unreachable!()
            }
        };
        let result = backend.execute_quantifier_program(&request, &CancellationToken::new())?;
        let validated = result.validate(&request, BackendKind::Cpu)?;
        assert_eq!(validated.rows().len(), 15);
        assert_eq!(validated.entity_outputs().len(), 15);
        assert_eq!(validated.read_dependencies().len(), dependency_rows);
        assert_eq!(
            validated.receipts()[path_receipts].output_cardinality,
            source_rows,
            "tail-list materialization must retain one row per accepted path"
        );
        assert_eq!(
            validated.receipts()[path_receipts + 1].output_cardinality,
            dependency_rows as u64
        );
        if entity_kind == ResidentQuantifierEntityKind::Relationship {
            assert_eq!(
                (
                    validated.receipts()[source_receipts].input_cardinality,
                    validated.receipts()[source_receipts].output_cardinality,
                ),
                (16, 15),
                "the duplicate zero-hop/leading-I empty list must collapse only in GroupCount"
            );
        }

        let mut truth_counts = [0_usize; 4];
        let mut empty_lists = 0_usize;
        for row in 0..validated.rows().len() {
            assert_eq!(validated.rows()[row][0], Value::Null);
            let entities = validated
                .entity_list(row, 0)
                .expect("entity list sidecar must accompany its null scalar placeholder");
            if entities.values.is_empty() {
                empty_lists += 1;
                assert_eq!(boolean(&validated.rows()[row][1]), Some(true));
                assert_eq!(boolean(&validated.rows()[row][2]), Some(false));
                assert_eq!(boolean(&validated.rows()[row][3]), Some(true));
                assert_eq!(boolean(&validated.rows()[row][4]), Some(false));
            }
            for (index, value) in validated.rows()[row][1..].iter().enumerate() {
                truth_counts[index] += usize::from(boolean(value) == Some(true));
            }
            assert!(entities.values.iter().all(|entity| matches!(
                (entity_kind, entity),
                (
                    ResidentQuantifierEntityKind::Node,
                    ResidentQuantifierEntityHandle::Node(_)
                ) | (
                    ResidentQuantifierEntityKind::Relationship,
                    ResidentQuantifierEntityHandle::Relationship(_)
                )
            )));
        }
        assert_eq!(empty_lists, 1);
        assert_eq!(truth_counts, [4, 11, 4, 6]);
    }
    Ok(())
}

#[test]
fn cpu_entity_properties_preserve_null_type_and_three_valued_quantifier_semantics() -> Result<()> {
    let fixture = three_valued_chain_fixture()?;
    let backend = fixture.backend()?;
    for (low, entity_kind, hops, expected_dependencies) in [
        (11, ResidentQuantifierEntityKind::Node, 3, 10_usize),
        (12, ResidentQuantifierEntityKind::Relationship, 4, 12_usize),
    ] {
        let request = request(&fixture, low, entity_kind, hops, hops)?;
        let result = backend.execute_quantifier_program(&request, &CancellationToken::new())?;
        let validated = result.validate(&request, BackendKind::Cpu)?;
        assert_eq!(validated.rows().len(), 1);
        assert_eq!(validated.read_dependencies().len(), expected_dependencies);
        assert_eq!(validated.entity_list(0, 0).unwrap().values.len(), 3);
        assert_eq!(boolean(&validated.rows()[0][1]), Some(false));
        assert_eq!(boolean(&validated.rows()[0][2]), Some(true));
        assert_eq!(boolean(&validated.rows()[0][3]), Some(false));
        assert_eq!(boolean(&validated.rows()[0][4]), None);
    }
    Ok(())
}

#[test]
fn entity_source_request_rejects_bad_skip_scope_capacity_and_fingerprint() -> Result<()> {
    let fixture = binary_fixture()?;
    let valid = request(&fixture, 21, ResidentQuantifierEntityKind::Node, 0, 3)?;

    let mut bad_fingerprint = valid.clone();
    bad_fingerprint.random_seed ^= 1;
    assert_eq!(
        bad_fingerprint.validate().unwrap_err().code,
        ErrorCode::GpuAdmissionFailure
    );

    let mut altered_shape = valid.clone();
    if let ResidentQuantifierSource::VariablePathEntityList { properties, .. } =
        &mut altered_shape.source
    {
        properties[0].shape = ResidentQuantifierEntityPropertyShape::MixedScalar {
            maximum_string_bytes: 1,
        };
    }
    assert_eq!(
        altered_shape.validate().unwrap_err().code,
        ErrorCode::GpuAdmissionFailure,
        "the immutable request fingerprint must bind the exact canonical property shape"
    );

    let mut wrong_shape_source = valid.source.clone();
    if let ResidentQuantifierSource::VariablePathEntityList { properties, .. } =
        &mut wrong_shape_source
    {
        properties[0].shape = ResidentQuantifierEntityPropertyShape::MixedScalar {
            maximum_string_bytes: 1,
        };
    }
    let wrong_shape = ResidentQuantifierProgramRequest::build_with_source(
        valid.generation,
        valid.execution,
        wrong_shape_source,
        valid.program.clone(),
        valid.max_rows,
        valid.max_list_items,
        valid.random_seed,
    )?;
    let error = fixture
        .backend()?
        .execute_quantifier_program(&wrong_shape, &CancellationToken::new())
        .expect_err("the CPU reference must revalidate shape against the pinned generation");
    assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);

    let mut absent_with_type = valid.source.clone();
    if let ResidentQuantifierSource::VariablePathEntityList { properties, .. } =
        &mut absent_with_type
    {
        properties[0].property = None;
        properties[0].shape = ResidentQuantifierEntityPropertyShape::Boolean;
    }
    assert!(
        ResidentQuantifierProgramRequest::build_with_source(
            valid.generation,
            valid.execution,
            absent_with_type,
            valid.program.clone(),
            valid.max_rows,
            valid.max_list_items,
            valid.random_seed,
        )
        .is_err()
    );
    assert!(ResidentQuantifierEntityPropertyShape::from_values([ScalarValue::Date(0),])?.is_none());

    let mut bad_skip_source = valid.source.clone();
    if let ResidentQuantifierSource::VariablePathEntityList { skip, .. } = &mut bad_skip_source {
        *skip = 0;
    }
    assert!(
        ResidentQuantifierProgramRequest::build_with_source(
            valid.generation,
            valid.execution,
            bad_skip_source,
            valid.program.clone(),
            valid.max_rows,
            valid.max_list_items,
            valid.random_seed,
        )
        .is_err()
    );

    let mut bad_slot_source = valid.source.clone();
    if let ResidentQuantifierSource::VariablePathEntityList { output, .. } = &mut bad_slot_source {
        *output = ResidentQuantifierSlot(valid.program.slot_count);
    }
    assert!(
        ResidentQuantifierProgramRequest::build_with_source(
            valid.generation,
            valid.execution,
            bad_slot_source,
            valid.program.clone(),
            valid.max_rows,
            valid.max_list_items,
            valid.random_seed,
        )
        .is_err()
    );

    assert!(
        ResidentQuantifierProgramRequest::build_with_source(
            valid.generation,
            valid.execution,
            valid.source.clone(),
            valid.program.clone(),
            1,
            valid.max_list_items,
            valid.random_seed,
        )
        .is_err()
    );

    let mut foreign_path_source = valid.source.clone();
    if let ResidentQuantifierSource::VariablePathEntityList { path, .. } = &mut foreign_path_source
    {
        path.execution.low ^= 1;
    }
    assert!(
        ResidentQuantifierProgramRequest::build_with_source(
            valid.generation,
            valid.execution,
            foreign_path_source,
            valid.program,
            valid.max_rows,
            valid.max_list_items,
            valid.random_seed,
        )
        .is_err()
    );
    Ok(())
}

#[test]
fn entity_result_rejects_forged_handles_dependencies_paths_and_receipts() -> Result<()> {
    let fixture = binary_fixture()?;
    let backend = fixture.backend()?;
    let request = request(&fixture, 31, ResidentQuantifierEntityKind::Node, 0, 3)?;
    let result = backend.execute_quantifier_program(&request, &CancellationToken::new())?;
    let (parts, entity_parts) = result.into_untrusted_entity_parts();
    let entity_parts = entity_parts.expect("entity source must publish typed proof parts");

    let mut bad_handle = entity_parts.clone();
    let output = bad_handle
        .outputs
        .iter_mut()
        .find(|output| !output.value.values.is_empty())
        .expect("fixture must contain a non-empty entity list");
    output.value.values[0] = ResidentQuantifierEntityHandle::Node(
        irongraph::gpu::ResidentQuantifierNodeHandle(u32::MAX),
    );
    assert!(
        irongraph::gpu::ResidentQuantifierProgramResult::from_untrusted_entity_parts(
            parts.clone(),
            bad_handle,
        )
        .validate(&request, BackendKind::Cpu)
        .is_err()
    );

    let mut bad_dependencies = entity_parts.clone();
    bad_dependencies.read_dependencies.pop();
    let mut dependency_parts = parts.clone();
    let dependency_receipt = request.source_receipt_count() - 1;
    dependency_parts.receipts[dependency_receipt].output_cardinality =
        bad_dependencies.read_dependencies.len() as u64;
    assert!(
        irongraph::gpu::ResidentQuantifierProgramResult::from_untrusted_entity_parts(
            dependency_parts,
            bad_dependencies,
        )
        .validate(&request, BackendKind::Cpu)
        .is_err()
    );

    let mut bad_path = entity_parts.clone();
    bad_path.source_paths[0].nodes[0] = u32::MAX;
    assert!(
        irongraph::gpu::ResidentQuantifierProgramResult::from_untrusted_entity_parts(
            parts.clone(),
            bad_path,
        )
        .validate(&request, BackendKind::Cpu)
        .is_err()
    );

    let mut bad_receipt = parts.clone();
    bad_receipt.receipts[request.source_receipt_count() - 2].output_cardinality += 1;
    assert!(
        irongraph::gpu::ResidentQuantifierProgramResult::from_untrusted_entity_parts(
            bad_receipt,
            entity_parts.clone(),
        )
        .validate(&request, BackendKind::Cpu)
        .is_err()
    );

    let mut bad_provenance = parts;
    bad_provenance.receipts[0].completion = ResidentDeviceCompletion::Metal;
    assert!(
        irongraph::gpu::ResidentQuantifierProgramResult::from_untrusted_entity_parts(
            bad_provenance,
            entity_parts,
        )
        .validate(&request, BackendKind::Cpu)
        .is_err()
    );

    let one = ResidentQuantifierEntityList {
        values: vec![ResidentQuantifierEntityHandle::Node(
            irongraph::gpu::ResidentQuantifierNodeHandle(1),
        )],
    };
    let one_again = one.clone();
    let two = ResidentQuantifierEntityList {
        values: vec![ResidentQuantifierEntityHandle::Node(
            irongraph::gpu::ResidentQuantifierNodeHandle(2),
        )],
    };
    assert_eq!(one, one_again);
    assert_eq!(one.structural_hash(), one_again.structural_hash());
    assert_ne!(one.structural_hash(), two.structural_hash());
    Ok(())
}
