// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]
#![cfg(all(feature = "accelerator", target_os = "macos"))]

use std::sync::{Mutex, MutexGuard};

use irongraph::{
    Bookmark, Layer, NodeId, ProjectId, Result, ScalarValue,
    gpu::{
        BackendKind, CpuBackend, ExecutionBackend, MetalBackend, ResidentCreateNodeValueInput,
        ResidentDeviceCompletion, ResidentExecutionId, ResidentExecutionObligation,
        ResidentMutationIntentEntry, ResidentMutationReadDependency, ResidentObligationKind,
        ResidentObligationScope, ResidentProjectImage, ResidentRowCreateCommand,
        ResidentRowCreateEntityOrigin, ResidentRowCreateMatchNode, ResidentRowCreateOutput,
        ResidentRowCreateProgram, ResidentRowCreateProperty, ResidentRowCreateRelationship,
        ResidentRowCreateScalarColumn, ResidentRowCreateValueInput, ResidentRowMutationGeneration,
        ResidentRowMutationRequest, ResidentRowMutationWorkIntentAction,
        ValidatedResidentRowMutation,
    },
    graph::{GraphStore, IndexCatalog, LayerMask, NodeInput, TemporalStore},
    types::{EntityKind, LabelId, PropertyId, RelationshipTypeId},
};
use tokio_util::sync::CancellationToken;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::from_u128(
    0x726f_775f_6372_6561_7465_5f6d_6574_616c,
));
const MEMORY_LIMIT_BYTES: usize = 256 * 1024 * 1024;
const RESERVED_BYTES: usize = 8 * 1024 * 1024;
const FIRST_NODE_ID: u64 = 50_000;
const FIRST_EDGE_ID: u64 = 60_000;

static METAL_TEST_MUTEX: Mutex<()> = Mutex::new(());

fn metal_test_guard() -> MutexGuard<'static, ()> {
    METAL_TEST_MUTEX
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

fn obligation(
    id: u64,
    kind: ResidentObligationKind,
    scope: ResidentObligationScope,
) -> ResidentExecutionObligation {
    ResidentExecutionObligation { id, kind, scope }
}

struct Fixture {
    graph: GraphStore,
    bookmark: Bookmark,
    left: LabelId,
    right: LabelId,
    missing: LabelId,
    payload: PropertyId,
    unset: PropertyId,
    relationship_type: RelationshipTypeId,
}

impl Fixture {
    fn new() -> Result<Self> {
        let mut graph = GraphStore::default();
        let left = graph.catalog_mut().intern_label("Left")?;
        let right = graph.catalog_mut().intern_label("Right")?;
        let missing = graph.catalog_mut().intern_label("Missing")?;
        let payload = graph.catalog_mut().intern_property("payload")?;
        let unset = graph.catalog_mut().intern_property("unset")?;
        let relationship_type = graph.catalog_mut().intern_relationship_type("LINKS")?;

        for (id, label) in [
            (101, left),
            (103, left),
            (202, right),
            (204, right),
            (206, right),
        ] {
            graph.insert_node(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: graph.revision().saturating_add(1),
                labels: vec![label],
                properties: Vec::new(),
            })?;
        }

        let bookmark = Bookmark {
            term: 73,
            index: graph.revision(),
        };
        Ok(Self {
            graph,
            bookmark,
            left,
            right,
            missing,
            payload,
            unset,
            relationship_type,
        })
    }

    fn image(&self) -> Result<ResidentProjectImage> {
        ResidentProjectImage::build(
            PROJECT,
            self.bookmark,
            &self.graph,
            &TemporalStore::default(),
            &IndexCatalog::default(),
        )
    }

    fn generation(&self) -> ResidentRowMutationGeneration {
        ResidentRowMutationGeneration {
            project: PROJECT,
            bookmark: self.bookmark,
            graph_revision: self.graph.revision(),
            layout_version: self.graph.layout_version(),
            catalog_generation: self.graph.catalog().optimizer_generation(),
        }
    }

    fn request(
        &self,
        execution_low: u64,
        first_match_label: u16,
    ) -> Result<ResidentRowMutationRequest> {
        let base = execution_low
            .checked_mul(100)
            .and_then(|value| value.checked_add(1_000))
            .expect("test obligation IDs must fit u64");
        let program = ResidentRowCreateProgram {
            entity_slot_count: 3,
            input_keys: Vec::new(),
            property_names: vec!["payload".to_owned(), "unset".to_owned()],
            property_tokens: vec![Some(self.payload), Some(self.unset)],
            label_names: vec!["Left".to_owned(), "Right".to_owned(), "Missing".to_owned()],
            label_tokens: vec![Some(self.left), Some(self.right), Some(self.missing)],
            relationship_type_names: vec!["LINKS".to_owned()],
            relationship_type_tokens: vec![Some(self.relationship_type)],
            commands: vec![
                ResidentRowCreateCommand::MatchNode(ResidentRowCreateMatchNode {
                    output_entity: 0,
                    labels: vec![first_match_label],
                    probe_obligation: obligation(
                        base + 1,
                        ResidentObligationKind::PatternScan,
                        ResidentObligationScope::MutationCommand(0),
                    ),
                }),
                ResidentRowCreateCommand::MatchNode(ResidentRowCreateMatchNode {
                    output_entity: 1,
                    labels: vec![1],
                    probe_obligation: obligation(
                        base + 2,
                        ResidentObligationKind::PatternScan,
                        ResidentObligationScope::MutationCommand(1),
                    ),
                }),
                ResidentRowCreateCommand::CreateRelationship(ResidentRowCreateRelationship {
                    output_entity: 2,
                    source_entity: 0,
                    target_entity: 1,
                    relationship_type: 0,
                    properties: vec![
                        ResidentRowCreateProperty {
                            property_name: 0,
                            value: ResidentRowCreateValueInput::Constant(
                                ResidentCreateNodeValueInput::Scalar(ScalarValue::Integer(77)),
                            ),
                        },
                        ResidentRowCreateProperty {
                            property_name: 1,
                            value: ResidentRowCreateValueInput::Constant(
                                ResidentCreateNodeValueInput::Scalar(ScalarValue::Null),
                            ),
                        },
                    ],
                    expression_obligation: obligation(
                        base + 3,
                        ResidentObligationKind::Expression,
                        ResidentObligationScope::MutationCommand(2),
                    ),
                    effect_obligation: obligation(
                        base + 4,
                        ResidentObligationKind::MutationEffect,
                        ResidentObligationScope::MutationCommand(2),
                    ),
                }),
            ],
            outputs: vec![
                ResidentRowCreateOutput {
                    name: "payload".to_owned(),
                    entity: 2,
                    property_name: 0,
                },
                ResidentRowCreateOutput {
                    name: "unset".to_owned(),
                    entity: 2,
                    property_name: 1,
                },
            ],
            continuation: None,
            source_obligation: obligation(
                base,
                ResidentObligationKind::MutationSelect,
                ResidentObligationScope::Selection,
            ),
            read_dependency_obligation: obligation(
                base + 5,
                ResidentObligationKind::MutationReadSet,
                ResidentObligationScope::Selection,
            ),
            final_relation_obligation: obligation(
                base + 6,
                ResidentObligationKind::Expression,
                ResidentObligationScope::PatternFinal,
            ),
        };

        ResidentRowMutationRequest::build_create(
            self.generation(),
            ResidentExecutionId {
                high: 0x524f_575f_4352_4541,
                low: execution_low,
            },
            Vec::new(),
            program,
            FIRST_NODE_ID,
            FIRST_EDGE_ID,
            LayerMask::OBSERVED,
            Layer::Observed,
            u64::try_from(self.graph.node_slot_count())
                .expect("resident node-slot count must fit u64"),
        )
    }
}

fn receipt_semantics(
    result: &ValidatedResidentRowMutation,
) -> Vec<(ResidentExecutionObligation, u64, u64)> {
    result
        .receipts()
        .iter()
        .map(|receipt| {
            (
                receipt.obligation,
                receipt.input_cardinality,
                receipt.output_cardinality,
            )
        })
        .collect()
}

fn assert_equal_validated_payloads(
    request: &ResidentRowMutationRequest,
    cpu: &ValidatedResidentRowMutation,
    metal: &ValidatedResidentRowMutation,
) {
    assert_eq!(metal.create(), cpu.create());
    assert_eq!(receipt_semantics(metal), receipt_semantics(cpu));
    assert_eq!(cpu.receipts().len(), request.obligations().len());
    assert_eq!(metal.receipts().len(), request.obligations().len());
    assert_eq!(
        cpu.receipts()
            .iter()
            .map(|receipt| receipt.obligation)
            .collect::<Vec<_>>(),
        request.obligations()
    );
    assert!(cpu.receipts().iter().all(|receipt| {
        receipt.execution == request.execution
            && receipt.completion == ResidentDeviceCompletion::CpuReference
    }));
    assert!(metal.receipts().iter().all(|receipt| {
        receipt.execution == request.execution
            && receipt.completion == ResidentDeviceCompletion::Metal
    }));
}

fn execute_cpu(
    backend: &CpuBackend,
    request: &ResidentRowMutationRequest,
) -> Result<ValidatedResidentRowMutation> {
    backend
        .execute_row_mutation(request, &CancellationToken::new())?
        .validate_for_publication(request, BackendKind::Cpu)
}

fn execute_metal(
    backend: &MetalBackend,
    request: &ResidentRowMutationRequest,
) -> Result<ValidatedResidentRowMutation> {
    backend
        .execute_row_mutation(request, &CancellationToken::new())?
        .validate_for_publication(request, BackendKind::Metal)
}

fn admitted_backends(fixture: &Fixture) -> Result<(CpuBackend, MetalBackend)> {
    let mut cpu = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
    cpu.admit_project(fixture.image()?)?;
    let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
    metal.admit_project(fixture.image()?)?;
    Ok((cpu, metal))
}

#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_generalized_row_create_cartesian_matches_cpu() -> Result<()> {
    let _guard = metal_test_guard();
    let fixture = Fixture::new()?;
    let (cpu, metal) = admitted_backends(&fixture)?;

    let cartesian_request = fixture.request(1, 0)?;
    let cpu_cartesian = execute_cpu(&cpu, &cartesian_request)?;
    let metal_cartesian = execute_metal(&metal, &cartesian_request)?;
    assert_equal_validated_payloads(&cartesian_request, &cpu_cartesian, &metal_cartesian);

    let create = metal_cartesian
        .create()
        .expect("validated Cartesian CREATE payload must exist");
    let expected_rows = [
        (0, 2, 101, 202),
        (0, 3, 101, 204),
        (0, 4, 101, 206),
        (1, 2, 103, 202),
        (1, 3, 103, 204),
        (1, 4, 103, 206),
    ];
    assert_eq!(create.work_rows().len(), expected_rows.len());
    assert_eq!(create.intents().len(), expected_rows.len());
    for (position, &(left_dense, right_dense, left_id, right_id)) in
        expected_rows.iter().enumerate()
    {
        let work_row = u64::try_from(position).expect("test row position must fit u64");
        let row = &create.work_rows()[position];
        assert_eq!(row.work_row, work_row);
        assert_eq!(row.source_row, 0);
        assert_eq!(row.match_lineage, [left_dense, right_dense]);
        assert_eq!(row.entities.len(), 3);

        assert_eq!(row.entities[0].kind, EntityKind::Node);
        assert_eq!(row.entities[0].target_id, left_id);
        assert_eq!(row.entities[0].dense_row, Some(left_dense));
        assert_eq!(row.entities[0].created_by, None);

        assert_eq!(row.entities[1].kind, EntityKind::Node);
        assert_eq!(row.entities[1].target_id, right_id);
        assert_eq!(row.entities[1].dense_row, Some(right_dense));
        assert_eq!(row.entities[1].created_by, None);

        assert_eq!(row.entities[2].kind, EntityKind::Relationship);
        assert_eq!(row.entities[2].target_id, FIRST_EDGE_ID + work_row);
        assert_eq!(row.entities[2].dense_row, None);
        assert_eq!(
            row.entities[2].created_by,
            Some(ResidentRowCreateEntityOrigin {
                command: 2,
                work_row,
            })
        );

        let intent = &create.intents()[position];
        assert_eq!(intent.work_row, work_row);
        assert_eq!(intent.source_row, 0);
        assert_eq!(intent.command, 2);
        assert_eq!(intent.target_id, FIRST_EDGE_ID + work_row);
        match &intent.action {
            ResidentRowMutationWorkIntentAction::CreateRelationship {
                relationship_type,
                source_node_id,
                target_node_id,
                properties,
            } => {
                assert_eq!(*relationship_type, 0);
                assert_eq!(*source_node_id, left_id);
                assert_eq!(*target_node_id, right_id);
                assert_eq!(
                    properties,
                    &[
                        ResidentMutationIntentEntry {
                            property_name: 0,
                            value: ScalarValue::Integer(77),
                        },
                        ResidentMutationIntentEntry {
                            property_name: 1,
                            value: ScalarValue::Null,
                        },
                    ]
                );
            }
            ResidentRowMutationWorkIntentAction::CreateNode { .. } => {
                panic!("Cartesian CREATE emitted a node intent instead of a relationship")
            }
        }
    }
    assert_eq!(
        create.read_dependencies(),
        &[
            ResidentMutationReadDependency::Node(0),
            ResidentMutationReadDependency::Node(1),
            ResidentMutationReadDependency::Node(2),
            ResidentMutationReadDependency::Node(3),
            ResidentMutationReadDependency::Node(4),
        ]
    );
    assert_eq!(
        create.columns(),
        &[
            ResidentRowCreateScalarColumn {
                output: 0,
                values: vec![ScalarValue::Integer(77); 6],
                work_rows: (0_u64..6).collect(),
            },
            ResidentRowCreateScalarColumn {
                output: 1,
                values: vec![ScalarValue::Null; 6],
                work_rows: (0_u64..6).collect(),
            },
        ]
    );
    assert_eq!(
        metal_cartesian
            .receipts()
            .iter()
            .map(|receipt| (receipt.input_cardinality, receipt.output_cardinality))
            .collect::<Vec<_>>(),
        [(1, 1), (1, 2), (2, 6), (6, 6), (6, 6), (8, 5), (6, 6)]
    );

    Ok(())
}

#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_generalized_row_create_empty_match_matches_cpu() -> Result<()> {
    let _guard = metal_test_guard();
    let fixture = Fixture::new()?;
    let (cpu, metal) = admitted_backends(&fixture)?;

    let empty_request = fixture.request(2, 2)?;
    let cpu_empty = execute_cpu(&cpu, &empty_request)?;
    let metal_empty = execute_metal(&metal, &empty_request)?;
    assert_equal_validated_payloads(&empty_request, &cpu_empty, &metal_empty);

    let empty = metal_empty
        .create()
        .expect("validated empty-match CREATE payload must exist");
    assert!(empty.work_rows().is_empty());
    assert!(empty.intents().is_empty());
    assert!(empty.read_dependencies().is_empty());
    assert_eq!(
        empty.columns(),
        &[
            ResidentRowCreateScalarColumn {
                output: 0,
                values: Vec::new(),
                work_rows: Vec::new(),
            },
            ResidentRowCreateScalarColumn {
                output: 1,
                values: Vec::new(),
                work_rows: Vec::new(),
            },
        ]
    );
    assert_eq!(
        metal_empty
            .receipts()
            .iter()
            .map(|receipt| (receipt.input_cardinality, receipt.output_cardinality))
            .collect::<Vec<_>>(),
        [(1, 1), (1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 0)]
    );

    Ok(())
}
