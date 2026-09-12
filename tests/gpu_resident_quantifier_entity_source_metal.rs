// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]
#![cfg(all(feature = "accelerator", target_os = "macos"))]

use std::sync::{Mutex, MutexGuard};

use irongraph::{
    Bookmark, EdgeId, Layer, NodeId, ProjectId, Result, ScalarValue,
    gpu::{
        BackendKind, CpuBackend, ExecutionBackend, MetalBackend, ResidentDirection,
        ResidentExecutionId, ResidentExecutionObligation, ResidentObligationKind,
        ResidentObligationScope, ResidentProjectImage, ResidentQuantifierBinary,
        ResidentQuantifierEntityKind, ResidentQuantifierEntityProperty,
        ResidentQuantifierEntityPropertyShape, ResidentQuantifierExpression as Expr,
        ResidentQuantifierGeneration, ResidentQuantifierKind, ResidentQuantifierOutput,
        ResidentQuantifierProgram, ResidentQuantifierProgramRequest, ResidentQuantifierProjection,
        ResidentQuantifierSlot, ResidentQuantifierSource, ResidentQuantifierStage,
        ResidentQuantifierValue as Value, ResidentVariablePathInput, ResidentVariablePathRequest,
        ResidentVariablePathSegment,
    },
    graph::{EdgeInput, GraphStore, IndexCatalog, LayerMask, NodeInput, TemporalStore},
    types::{LabelId, PropertyId, RelationshipTypeId},
};
use tokio_util::sync::CancellationToken;

const MEMORY_LIMIT: usize = 512 * 1024 * 1024;
const RESERVED_MEMORY: usize = 64 * 1024 * 1024;
static METAL_TEST_LOCK: Mutex<()> = Mutex::new(());

fn metal_guard() -> MutexGuard<'static, ()> {
    METAL_TEST_LOCK
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

struct Fixture {
    graph: GraphStore,
    project: ProjectId,
    bookmark: Bookmark,
    node_start: LabelId,
    relationship_start: LabelId,
    name: PropertyId,
    missing: PropertyId,
    leading: RelationshipTypeId,
    branch_a: RelationshipTypeId,
    branch_b: RelationshipTypeId,
}

impl Fixture {
    fn new() -> Result<Self> {
        let mut graph = GraphStore::default();
        let node_start = graph.catalog_mut().intern_label("SNodes")?;
        let relationship_start = graph.catalog_mut().intern_label("SRelationships")?;
        let name = graph.catalog_mut().intern_property("name")?;
        let missing = graph.catalog_mut().intern_property("missing")?;
        let leading = graph.catalog_mut().intern_relationship_type("I")?;
        let branch_a = graph.catalog_mut().intern_relationship_type("RA")?;
        let branch_b = graph.catalog_mut().intern_relationship_type("RB")?;

        insert_node(&mut graph, 1, vec![relationship_start], Vec::new())?;
        insert_node(&mut graph, 2, vec![node_start], Vec::new())?;
        insert_edge(&mut graph, 1, 1, 2, leading, Vec::new())?;
        let mut parents = vec![2_u64];
        let mut next_node = 3_u64;
        let mut next_edge = 2_u64;
        for _ in 0..2 {
            let mut children = Vec::new();
            for parent in parents {
                for (relationship_type, value) in [(branch_b, "b"), (branch_a, "a")] {
                    let child = next_node;
                    next_node += 1;
                    insert_node(
                        &mut graph,
                        child,
                        Vec::new(),
                        vec![(name, ScalarValue::String(value.to_owned().into()))],
                    )?;
                    insert_edge(
                        &mut graph,
                        next_edge,
                        parent,
                        child,
                        relationship_type,
                        vec![(name, ScalarValue::String(value.to_owned().into()))],
                    )?;
                    next_edge += 1;
                    children.push(child);
                }
            }
            parents = children;
        }
        let project = ProjectId(uuid::Uuid::from_u128(0x4d45_5441_4c51_5541_4e54_0001));
        let bookmark = Bookmark {
            term: 91,
            index: graph.revision(),
        };
        Ok(Self {
            graph,
            project,
            bookmark,
            node_start,
            relationship_start,
            name,
            missing,
            leading,
            branch_a,
            branch_b,
        })
    }

    fn generation(&self) -> ResidentQuantifierGeneration {
        ResidentQuantifierGeneration {
            project: self.project,
            bookmark: self.bookmark,
            graph_revision: self.graph.revision(),
            layout_version: self.graph.layout_version(),
            catalog_generation: self.graph.catalog().optimizer_generation(),
        }
    }

    fn image(&self) -> Result<ResidentProjectImage> {
        ResidentProjectImage::build(
            self.project,
            self.bookmark,
            &self.graph,
            &TemporalStore::default(),
            &IndexCatalog::default(),
        )
    }
}

fn insert_node(
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

fn insert_edge(
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
    kind: ResidentQuantifierEntityKind,
) -> Result<ResidentVariablePathRequest> {
    let snapshot = fixture.graph.snapshot()?;
    let (label, maximum_hops, mut relationship_types) = match kind {
        ResidentQuantifierEntityKind::Node => (
            fixture.node_start,
            2,
            vec![fixture.branch_a, fixture.branch_b],
        ),
        ResidentQuantifierEntityKind::Relationship => (
            fixture.relationship_start,
            3,
            vec![fixture.leading, fixture.branch_a, fixture.branch_b],
        ),
    };
    relationship_types.sort_unstable_by_key(|relationship_type| relationship_type.0);
    let base = 0x6100_0000 + execution.low * 16;
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
            minimum_hops: 0,
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
        maximum_frontier_paths: 16,
        maximum_output_rows: 16,
        distinct_endpoints: false,
    };
    request.validate()?;
    Ok(request)
}

fn slot(index: u16) -> Expr {
    Expr::Slot(ResidentQuantifierSlot(index))
}

fn none_missing(list_slot: u16) -> Expr {
    Expr::Predicate {
        kind: ResidentQuantifierKind::None,
        variable: ResidentQuantifierSlot(7),
        list: Box::new(slot(list_slot)),
        predicate: Box::new(Expr::Binary {
            left: Box::new(Expr::Property {
                source: Box::new(slot(7)),
                key: "missing".to_owned(),
            }),
            operation: ResidentQuantifierBinary::Equal,
            right: Box::new(Expr::Literal(Value::String("a".to_owned()))),
        }),
    }
}

fn request(
    fixture: &Fixture,
    low: u64,
    kind: ResidentQuantifierEntityKind,
    grouped: bool,
) -> Result<ResidentQuantifierProgramRequest> {
    let execution = ResidentExecutionId {
        high: 0x4d45_5441_4c5f_5155,
        low,
    };
    let path = path_request(fixture, execution, kind)?;
    let (properties, stages, outputs) = if grouped {
        (
            vec![ResidentQuantifierEntityProperty {
                key: "name".to_owned(),
                property: Some(fixture.name),
                shape: ResidentQuantifierEntityPropertyShape::String { maximum_bytes: 1 },
            }],
            vec![
                ResidentQuantifierStage::GroupCount {
                    groups: vec![ResidentQuantifierProjection {
                        output: ResidentQuantifierSlot(1),
                        expression: slot(0),
                    }],
                    count_outputs: vec![ResidentQuantifierSlot(2)],
                },
                ResidentQuantifierStage::Project {
                    keep_scope: false,
                    bindings: vec![
                        ResidentQuantifierProjection {
                            output: ResidentQuantifierSlot(3),
                            expression: slot(1),
                        },
                        ResidentQuantifierProjection {
                            output: ResidentQuantifierSlot(4),
                            expression: slot(2),
                        },
                    ],
                },
            ],
            vec![
                ResidentQuantifierOutput {
                    name: "entities".to_owned(),
                    source: ResidentQuantifierSlot(3),
                },
                ResidentQuantifierOutput {
                    name: "count".to_owned(),
                    source: ResidentQuantifierSlot(4),
                },
            ],
        )
    } else {
        (
            vec![ResidentQuantifierEntityProperty {
                key: "missing".to_owned(),
                property: Some(fixture.missing),
                shape: ResidentQuantifierEntityPropertyShape::Absent,
            }],
            vec![ResidentQuantifierStage::Project {
                keep_scope: false,
                bindings: vec![
                    ResidentQuantifierProjection {
                        output: ResidentQuantifierSlot(3),
                        expression: slot(0),
                    },
                    ResidentQuantifierProjection {
                        output: ResidentQuantifierSlot(4),
                        expression: none_missing(0),
                    },
                ],
            }],
            vec![
                ResidentQuantifierOutput {
                    name: "entities".to_owned(),
                    source: ResidentQuantifierSlot(3),
                },
                ResidentQuantifierOutput {
                    name: "none".to_owned(),
                    source: ResidentQuantifierSlot(4),
                },
            ],
        )
    };
    ResidentQuantifierProgramRequest::build_with_source(
        fixture.generation(),
        execution,
        ResidentQuantifierSource::VariablePathEntityList {
            path,
            output: ResidentQuantifierSlot(0),
            entity_kind: kind,
            skip: 1,
            properties,
            materialize_obligation: obligation(
                0x6200_0000 + low,
                ResidentObligationKind::Expression,
                ResidentObligationScope::Expression(u16::MAX),
            ),
        },
        ResidentQuantifierProgram {
            slot_count: 8,
            stages,
            outputs,
        },
        64,
        16,
        0x5eed + low,
    )
}

fn short_string_list_request(fixture: &Fixture) -> Result<ResidentQuantifierProgramRequest> {
    const ROWS: usize = 128;
    let execution = ResidentExecutionId {
        high: 0x4d45_5441_4c5f_4259,
        low: 0x5445_5f52_4143_4501,
    };
    let concatenate = |value: &str| Expr::Binary {
        left: Box::new(Expr::Literal(Value::String(value.to_owned()))),
        operation: ResidentQuantifierBinary::Add,
        right: Box::new(Expr::Literal(Value::String(String::new()))),
    };
    ResidentQuantifierProgramRequest::build(
        fixture.generation(),
        execution,
        ResidentQuantifierProgram {
            slot_count: 4,
            stages: vec![
                ResidentQuantifierStage::Project {
                    keep_scope: false,
                    bindings: vec![ResidentQuantifierProjection {
                        output: ResidentQuantifierSlot(0),
                        expression: Expr::List(
                            (0..ROWS)
                                .map(|row| Expr::Literal(Value::Integer(row as i64)))
                                .collect(),
                        ),
                    }],
                },
                ResidentQuantifierStage::Unwind {
                    expression: slot(0),
                    output: ResidentQuantifierSlot(1),
                },
                ResidentQuantifierStage::Project {
                    keep_scope: false,
                    bindings: vec![
                        ResidentQuantifierProjection {
                            output: ResidentQuantifierSlot(2),
                            expression: slot(1),
                        },
                        ResidentQuantifierProjection {
                            output: ResidentQuantifierSlot(3),
                            expression: Expr::List(vec![
                                concatenate("a"),
                                concatenate("b"),
                                concatenate("c"),
                            ]),
                        },
                    ],
                },
            ],
            outputs: vec![
                ResidentQuantifierOutput {
                    name: "row".to_owned(),
                    source: ResidentQuantifierSlot(2),
                },
                ResidentQuantifierOutput {
                    name: "short_strings".to_owned(),
                    source: ResidentQuantifierSlot(3),
                },
            ],
        },
        ROWS,
        ROWS,
        0x4259_5445_5241_4345,
    )
}

#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_entity_source_matches_cpu_order_counts_absent_shape_and_bounds() -> Result<()> {
    let _guard = metal_guard();
    let fixture = Fixture::new()?;
    let mut cpu = CpuBackend::new(MEMORY_LIMIT, RESERVED_MEMORY);
    cpu.admit_project(fixture.image()?)?;
    let mut metal = MetalBackend::new(0, MEMORY_LIMIT, RESERVED_MEMORY)?;
    metal.admit_project(fixture.image()?)?;
    assert!(metal.supports_native_quantifier_entity_source());

    for (low, kind, grouped) in [
        (1, ResidentQuantifierEntityKind::Node, false),
        (2, ResidentQuantifierEntityKind::Relationship, true),
    ] {
        let request = request(&fixture, low, kind, grouped)?;
        let ResidentQuantifierSource::VariablePathEntityList { path, .. } = &request.source else {
            unreachable!()
        };
        assert!(request.max_rows > path.maximum_output_rows);
        let cpu_result = cpu
            .execute_quantifier_program(&request, &CancellationToken::new())?
            .validate(&request, BackendKind::Cpu)?;
        let metal_result = metal
            .execute_quantifier_program(&request, &CancellationToken::new())?
            .validate(&request, BackendKind::Metal)?;
        assert_eq!(metal_result.rows(), cpu_result.rows());
        assert_eq!(metal_result.entity_outputs(), cpu_result.entity_outputs());
        assert_eq!(
            metal_result.read_dependencies(),
            cpu_result.read_dependencies()
        );
        let (_, cpu_entities) = cpu_result.into_entity_parts();
        let (metal_parts, metal_entities) = metal_result.into_entity_parts();
        let cpu_entities = cpu_entities.expect("CPU entity sidecar");
        let metal_entities = metal_entities.expect("Metal entity sidecar");
        assert_eq!(metal_entities.source_paths, cpu_entities.source_paths);

        if grouped {
            let lists = metal_entities
                .outputs
                .iter()
                .map(|output| &output.value)
                .collect::<Vec<_>>();
            assert!(lists.windows(2).all(|pair| pair[0] <= pair[1]));
            let empty_row = metal_entities
                .outputs
                .iter()
                .find(|output| output.value.values.is_empty())
                .expect("grouped relationship source must contain the duplicate empty list")
                .row as usize;
            assert_eq!(
                metal_entities.outputs[empty_row].value.values.len(),
                0,
                "entity sidecar order must align with canonical grouped rows"
            );
            assert_eq!(metal_parts.rows[empty_row][1], Value::Integer(2));
        }
    }

    let request = short_string_list_request(&fixture)?;
    let cpu_result = cpu
        .execute_quantifier_program(&request, &CancellationToken::new())?
        .validate(&request, BackendKind::Cpu)?;
    let metal_result = metal
        .execute_quantifier_program(&request, &CancellationToken::new())?
        .validate(&request, BackendKind::Metal)?;
    assert_eq!(metal_result.rows(), cpu_result.rows());
    assert_eq!(metal_result.rows().len(), 128);
    for (row, values) in metal_result.rows().iter().enumerate() {
        assert_eq!(values[0], Value::Integer(row as i64));
        assert_eq!(
            values[1],
            Value::List(vec![
                Value::String("a".to_owned()),
                Value::String("b".to_owned()),
                Value::String("c".to_owned()),
            ])
        );
    }
    Ok(())
}
