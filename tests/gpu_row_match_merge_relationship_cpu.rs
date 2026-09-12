// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use irongraph::{
    Bookmark, EdgeId, ErrorCode, Layer, NodeId, ProjectId, Result, ScalarValue,
    gpu::{
        BackendKind, CpuBackend, ExecutionBackend, ResidentDeviceCompletion, ResidentExecutionId,
        ResidentExecutionObligation, ResidentMutationReadDependency, ResidentObligationKind,
        ResidentObligationScope, ResidentProjectImage, ResidentRowMutationAscendingOrder,
        ResidentRowMutationGeneration, ResidentRowMutationInputMap,
        ResidentRowMutationIntegerOutput, ResidentRowMutationMatchMergeRelationshipProgram,
        ResidentRowMutationMatchNode, ResidentRowMutationMergeKey, ResidentRowMutationMergeNode,
        ResidentRowMutationMergeRelationship, ResidentRowMutationOutput,
        ResidentRowMutationProgram, ResidentRowMutationRequest, ResidentRowMutationTarget,
        ResidentRowMutationWorkIntentAction, ValidatedResidentRowMatchMergeRelationship,
    },
    graph::{EdgeInput, GraphStore, IndexCatalog, LayerMask, NodeInput, TemporalStore},
    types::{LabelId, PropertyId, RelationshipTypeId},
};
use tokio_util::sync::CancellationToken;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());
const MEMORY_LIMIT_BYTES: usize = 64 * 1024 * 1024;
const RESERVED_BYTES: usize = 4 * 1024 * 1024;
const FIRST_NODE_ID: u64 = 10_000;
const FIRST_EDGE_ID: u64 = 20_000;
const MAXIMUM_OUTPUT_ROWS: usize = 256;

struct Fixture {
    graph: GraphStore,
    year_label: LabelId,
    event_label: LabelId,
    year_property: PropertyId,
    id_property: PropertyId,
    in_type: RelationshipTypeId,
    wrong_type: RelationshipTypeId,
}

impl Fixture {
    fn new(years: &[(u64, i64)]) -> Result<Self> {
        let mut graph = GraphStore::default();
        let year_label = graph.catalog_mut().intern_label("Year")?;
        let event_label = graph.catalog_mut().intern_label("Event")?;
        let year_property = graph.catalog_mut().intern_property("year")?;
        let id_property = graph.catalog_mut().intern_property("id")?;
        let in_type = graph.catalog_mut().intern_relationship_type("IN")?;
        let wrong_type = graph.catalog_mut().intern_relationship_type("WRONG")?;
        for (revision, (id, year)) in years.iter().copied().enumerate() {
            graph.insert_node(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: revision as u64 + 1,
                labels: vec![year_label],
                properties: vec![(year_property, ScalarValue::Integer(year))],
            })?;
        }
        Ok(Self {
            graph,
            year_label,
            event_label,
            year_property,
            id_property,
            in_type,
            wrong_type,
        })
    }

    fn add_event(&mut self, id: u64, event_id: i64) -> Result<()> {
        self.graph.insert_node(NodeInput {
            id: NodeId(id),
            layer: Layer::Observed,
            revision: self.graph.revision().saturating_add(1),
            labels: vec![self.event_label],
            properties: vec![(self.id_property, ScalarValue::Integer(event_id))],
        })?;
        Ok(())
    }

    fn add_edge(
        &mut self,
        id: u64,
        source: u64,
        target: u64,
        relationship_type: RelationshipTypeId,
    ) -> Result<()> {
        self.graph.insert_edge(EdgeInput {
            id: EdgeId(id),
            source: NodeId(source),
            target: NodeId(target),
            relationship_type,
            layer: Layer::Observed,
            revision: self.graph.revision().saturating_add(1),
            properties: Vec::new(),
        })?;
        Ok(())
    }

    fn generation(&self) -> ResidentRowMutationGeneration {
        ResidentRowMutationGeneration {
            project: PROJECT,
            bookmark: Bookmark {
                term: 61,
                index: self.graph.revision(),
            },
            graph_revision: self.graph.revision(),
            layout_version: self.graph.layout_version(),
            catalog_generation: self.graph.catalog().optimizer_generation(),
        }
    }

    fn backend(&self) -> Result<CpuBackend> {
        let generation = self.generation();
        let image = ResidentProjectImage::build(
            PROJECT,
            generation.bookmark,
            &self.graph,
            &TemporalStore::default(),
            &IndexCatalog::default(),
        )?;
        let mut backend = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
        backend.admit_project(image)?;
        Ok(backend)
    }

    fn request(
        &self,
        seed: u64,
        input: Vec<ResidentRowMutationInputMap>,
    ) -> Result<ResidentRowMutationRequest> {
        let base = seed
            .checked_mul(32)
            .and_then(|value| value.checked_add(10_000))
            .expect("test obligation IDs must fit u64");
        ResidentRowMutationRequest::build_match_merge_relationship(
            self.generation(),
            ResidentExecutionId {
                high: 0x554e_5749_4e44_4d52,
                low: seed,
            },
            input,
            ResidentRowMutationProgram {
                input_keys: vec!["year".to_owned(), "id".to_owned()],
                property_names: vec!["year".to_owned(), "id".to_owned()],
                property_tokens: vec![Some(self.year_property), Some(self.id_property)],
                label_names: vec!["Year".to_owned(), "Event".to_owned()],
                label_tokens: vec![Some(self.year_label), Some(self.event_label)],
                merge: ResidentRowMutationMergeNode {
                    output_entity: 1,
                    labels: vec![1],
                    keys: vec![ResidentRowMutationMergeKey {
                        property_name: 1,
                        input_key: 1,
                    }],
                    probe_obligation: obligation(
                        base + 3,
                        ResidentObligationKind::MutationSelect,
                        ResidentObligationScope::MutationCommand(0),
                    ),
                    effect_obligation: obligation(
                        base + 4,
                        ResidentObligationKind::MutationEffect,
                        ResidentObligationScope::MutationCommand(0),
                    ),
                },
                sets: Vec::new(),
                outputs: Vec::new(),
                source_obligation: obligation(
                    base + 1,
                    ResidentObligationKind::Expression,
                    ResidentObligationScope::Expression(u16::MAX - 2),
                ),
                read_dependency_obligation: obligation(
                    base + 7,
                    ResidentObligationKind::MutationReadSet,
                    ResidentObligationScope::Selection,
                ),
                final_relation_obligation: obligation(
                    base + 8,
                    ResidentObligationKind::Expression,
                    ResidentObligationScope::Expression(u16::MAX - 1),
                ),
            },
            ResidentRowMutationMatchMergeRelationshipProgram {
                matched_node: ResidentRowMutationMatchNode {
                    output_entity: 0,
                    labels: vec![0],
                    keys: vec![ResidentRowMutationMergeKey {
                        property_name: 0,
                        input_key: 0,
                    }],
                    probe_obligation: obligation(
                        base + 2,
                        ResidentObligationKind::PatternScan,
                        ResidentObligationScope::PatternScan,
                    ),
                },
                relationship_type_names: vec!["IN".to_owned()],
                relationship_type_tokens: vec![Some(self.in_type)],
                relationship: ResidentRowMutationMergeRelationship {
                    output_entity: 2,
                    source_entity: 1,
                    target_entity: 0,
                    relationship_type: 0,
                    keys: Vec::new(),
                    probe_obligation: obligation(
                        base + 5,
                        ResidentObligationKind::MutationSelect,
                        ResidentObligationScope::MutationCommand(1),
                    ),
                    effect_obligation: obligation(
                        base + 6,
                        ResidentObligationKind::MutationEffect,
                        ResidentObligationScope::MutationCommand(1),
                    ),
                },
                integer_outputs: vec![ResidentRowMutationIntegerOutput {
                    name: "x".to_owned(),
                    entity: 1,
                    property_name: 1,
                }],
                ascending_order: ResidentRowMutationAscendingOrder {
                    output: 0,
                    obligation: obligation(
                        base + 9,
                        ResidentObligationKind::Sort,
                        ResidentObligationScope::Expression(u16::MAX),
                    ),
                },
            },
            FIRST_NODE_ID,
            FIRST_EDGE_ID,
            LayerMask::OBSERVED,
            Layer::Observed,
            MAXIMUM_OUTPUT_ROWS,
        )
    }
}

fn obligation(
    id: u64,
    kind: ResidentObligationKind,
    scope: ResidentObligationScope,
) -> ResidentExecutionObligation {
    ResidentExecutionObligation { id, kind, scope }
}

fn event(year: i64, id: ScalarValue) -> ResidentRowMutationInputMap {
    ResidentRowMutationInputMap {
        entries: vec![
            ("id".to_owned(), id),
            ("year".to_owned(), ScalarValue::Integer(year)),
        ],
    }
}

fn execute(
    backend: &CpuBackend,
    request: &ResidentRowMutationRequest,
) -> Result<irongraph::gpu::ValidatedResidentRowMutation> {
    backend
        .execute_row_mutation(request, &CancellationToken::new())?
        .validate_for_publication(request, BackendKind::Cpu)
}

fn generalized(
    result: &irongraph::gpu::ValidatedResidentRowMutation,
) -> &ValidatedResidentRowMatchMergeRelationship {
    result
        .match_merge_relationship()
        .expect("v2 request must expose a v2 validated payload")
}

fn assert_integer_rows(result: &ValidatedResidentRowMatchMergeRelationship, expected: &[i64]) {
    assert_eq!(result.columns().len(), 1);
    assert_eq!(result.columns()[0].output, 0);
    assert_eq!(result.columns()[0].values, expected);
    assert_eq!(result.columns()[0].validity, vec![1; expected.len()]);
}

#[test]
fn cpu_exact_unwind1_scenario_6_creates_two_events_and_two_normalized_edges() -> Result<()> {
    let fixture = Fixture::new(&[(100, 2016)])?;
    let backend = fixture.backend()?;
    let request = fixture.request(
        1,
        vec![
            event(2016, ScalarValue::Integer(1)),
            event(2016, ScalarValue::Integer(2)),
        ],
    )?;

    let validated = execute(&backend, &request)?;
    let result = generalized(&validated);

    assert_integer_rows(result, &[1, 2]);
    assert_eq!(result.work_rows().len(), 2);
    assert_eq!(result.columns()[0].work_rows, &[0, 1]);
    assert_eq!(result.intents().len(), 4);
    for (work_row, pair) in result.intents().chunks_exact(2).enumerate() {
        assert_eq!(pair[0].work_row, work_row as u64);
        assert_eq!(pair[0].command, 0);
        assert_eq!(pair[1].work_row, work_row as u64);
        assert_eq!(pair[1].command, 1);
        assert!(matches!(
            &pair[1].action,
            ResidentRowMutationWorkIntentAction::CreateRelationship {
                relationship_type: 0,
                source_node_id,
                target_node_id: 100,
                properties,
            } if *source_node_id == FIRST_NODE_ID + work_row as u64 && properties.is_empty()
        ));
    }
    assert_eq!(
        result.read_dependencies(),
        &[ResidentMutationReadDependency::Node(0)]
    );
    assert!(
        validated
            .receipts()
            .iter()
            .all(|receipt| { receipt.completion == ResidentDeviceCompletion::CpuReference })
    );
    assert_eq!(
        validated
            .receipts()
            .iter()
            .map(|receipt| (receipt.input_cardinality, receipt.output_cardinality))
            .collect::<Vec<_>>(),
        vec![
            (1, 2),
            (2, 2),
            (2, 2),
            (2, 2),
            (2, 2),
            (2, 2),
            (2, 2),
            (2, 2),
            (2, 2),
        ]
    );
    Ok(())
}

#[test]
fn cpu_integer_projection_is_stably_sorted_ascending_with_work_lineage() -> Result<()> {
    let fixture = Fixture::new(&[(100, 2016)])?;
    let backend = fixture.backend()?;
    let request = fixture.request(
        11,
        vec![
            event(2016, ScalarValue::Integer(2)),
            event(2016, ScalarValue::Integer(1)),
        ],
    )?;

    let validated = execute(&backend, &request)?;
    let result = generalized(&validated);

    assert_integer_rows(result, &[1, 2]);
    assert_eq!(result.columns()[0].work_rows, &[1, 0]);
    Ok(())
}

#[test]
fn cpu_duplicate_parameter_rows_reuse_one_node_and_one_relationship_without_collapsing_rows()
-> Result<()> {
    let fixture = Fixture::new(&[(100, 2016)])?;
    let backend = fixture.backend()?;
    let request = fixture.request(
        2,
        vec![
            event(2016, ScalarValue::Integer(7)),
            event(2016, ScalarValue::Integer(7)),
        ],
    )?;

    let validated = execute(&backend, &request)?;
    let result = generalized(&validated);

    assert_integer_rows(result, &[7, 7]);
    assert_eq!(result.work_rows().len(), 2);
    assert_eq!(result.intents().len(), 2);
    assert_eq!(result.work_rows()[1].merged_node.target_id, FIRST_NODE_ID);
    assert_eq!(result.work_rows()[1].merged_node.created_by, Some(0));
    assert_eq!(
        result.work_rows()[1].merged_relationship.target_id,
        FIRST_EDGE_ID
    );
    assert_eq!(
        result.work_rows()[1].merged_relationship.created_by,
        Some(0)
    );
    Ok(())
}

#[test]
fn cpu_zero_correlated_match_has_no_effects_and_an_empty_sorted_relation() -> Result<()> {
    let fixture = Fixture::new(&[(100, 2015)])?;
    let backend = fixture.backend()?;
    let request = fixture.request(3, vec![event(2016, ScalarValue::Integer(1))])?;

    let validated = execute(&backend, &request)?;
    let result = generalized(&validated);

    assert_integer_rows(result, &[]);
    assert!(result.work_rows().is_empty());
    assert!(result.intents().is_empty());
    Ok(())
}

#[test]
fn cpu_matched_null_merge_key_fails_atomically() -> Result<()> {
    let fixture = Fixture::new(&[(100, 2016)])?;
    let backend = fixture.backend()?;
    let request = fixture.request(4, vec![event(2016, ScalarValue::Null)])?;

    let raw = backend.execute_row_mutation(&request, &CancellationToken::new())?;
    let error = raw
        .validate_for_publication(&request, BackendKind::Cpu)
        .expect_err("a surviving NULL node-MERGE key must fail");

    assert_eq!(error.code, ErrorCode::QueryType);
    assert!(error.message.contains("surviving source row 0"));
    assert!(error.message.contains("id"));
    Ok(())
}

#[test]
fn cpu_unmatched_null_merge_key_is_filtered_before_merge_and_does_not_error() -> Result<()> {
    let fixture = Fixture::new(&[(100, 2015)])?;
    let backend = fixture.backend()?;
    let request = fixture.request(5, vec![event(2016, ScalarValue::Null)])?;

    let validated = execute(&backend, &request)?;
    let result = generalized(&validated);

    assert_integer_rows(result, &[]);
    assert!(result.intents().is_empty());
    Ok(())
}

#[test]
fn cpu_existing_event_and_existing_normalized_edge_are_reused() -> Result<()> {
    let mut fixture = Fixture::new(&[(100, 2016)])?;
    fixture.add_event(200, 1)?;
    let in_type = fixture.in_type;
    fixture.add_edge(300, 200, 100, in_type)?;
    let backend = fixture.backend()?;
    let request = fixture.request(6, vec![event(2016, ScalarValue::Integer(1))])?;

    let validated = execute(&backend, &request)?;
    let result = generalized(&validated);

    assert_integer_rows(result, &[1]);
    assert!(result.intents().is_empty());
    assert_eq!(
        result.work_rows()[0].merged_node,
        ResidentRowMutationTarget {
            target_id: 200,
            dense_row: Some(1),
            created_by: None,
        }
    );
    assert_eq!(
        result.work_rows()[0].merged_relationship,
        ResidentRowMutationTarget {
            target_id: 300,
            dense_row: Some(0),
            created_by: None,
        }
    );
    assert_eq!(
        result.read_dependencies(),
        &[
            ResidentMutationReadDependency::Node(0),
            ResidentMutationReadDependency::Node(1),
            ResidentMutationReadDependency::Relationship(0),
        ]
    );
    Ok(())
}

#[test]
fn cpu_wrong_direction_and_wrong_type_edges_do_not_suppress_correct_relationship_creation()
-> Result<()> {
    let mut fixture = Fixture::new(&[(100, 2016)])?;
    fixture.add_event(200, 1)?;
    let in_type = fixture.in_type;
    let wrong_type = fixture.wrong_type;
    fixture.add_edge(300, 100, 200, in_type)?;
    fixture.add_edge(301, 200, 100, wrong_type)?;
    let backend = fixture.backend()?;
    let request = fixture.request(7, vec![event(2016, ScalarValue::Integer(1))])?;

    let validated = execute(&backend, &request)?;
    let result = generalized(&validated);

    assert_integer_rows(result, &[1]);
    assert_eq!(result.intents().len(), 1);
    assert!(matches!(
        &result.intents()[0].action,
        ResidentRowMutationWorkIntentAction::CreateRelationship {
            relationship_type: 0,
            source_node_id: 200,
            target_node_id: 100,
            properties,
        } if properties.is_empty()
    ));
    assert_eq!(result.intents()[0].target_id, FIRST_EDGE_ID);
    assert_eq!(
        result.read_dependencies(),
        &[
            ResidentMutationReadDependency::Node(0),
            ResidentMutationReadDependency::Node(1),
            ResidentMutationReadDependency::Relationship(0),
            ResidentMutationReadDependency::Relationship(1),
        ]
    );
    Ok(())
}

#[test]
fn cpu_multiple_year_matches_expand_work_rows_and_create_distinct_endpoint_edges() -> Result<()> {
    let fixture = Fixture::new(&[(100, 2016), (101, 2016)])?;
    let backend = fixture.backend()?;
    let request = fixture.request(8, vec![event(2016, ScalarValue::Integer(9))])?;

    let validated = execute(&backend, &request)?;
    let result = generalized(&validated);

    assert_integer_rows(result, &[9, 9]);
    assert_eq!(result.work_rows().len(), 2);
    assert_eq!(result.work_rows()[0].source_row, 0);
    assert_eq!(result.work_rows()[1].source_row, 0);
    assert_eq!(result.work_rows()[0].merged_node.target_id, FIRST_NODE_ID);
    assert_eq!(result.work_rows()[1].merged_node.target_id, FIRST_NODE_ID);
    let relationship_targets = result
        .intents()
        .iter()
        .filter_map(|intent| match &intent.action {
            ResidentRowMutationWorkIntentAction::CreateRelationship { target_node_id, .. } => {
                Some(*target_node_id)
            }
            ResidentRowMutationWorkIntentAction::CreateNode { .. } => None,
        })
        .collect::<Vec<_>>();
    assert_eq!(relationship_targets, vec![100, 101]);
    Ok(())
}

#[test]
fn v1_rejects_duplicate_merge_destination_properties_with_different_input_keys() -> Result<()> {
    let fixture = Fixture::new(&[(100, 2016)])?;
    let base = 90_000;
    let error = ResidentRowMutationRequest::build(
        fixture.generation(),
        ResidentExecutionId { high: 1, low: 9 },
        vec![event(2016, ScalarValue::Integer(1))],
        ResidentRowMutationProgram {
            input_keys: vec!["year".to_owned(), "id".to_owned()],
            property_names: vec!["id".to_owned()],
            property_tokens: vec![Some(fixture.id_property)],
            label_names: vec!["Event".to_owned()],
            label_tokens: vec![Some(fixture.event_label)],
            merge: ResidentRowMutationMergeNode {
                output_entity: 0,
                labels: vec![0],
                keys: vec![
                    ResidentRowMutationMergeKey {
                        property_name: 0,
                        input_key: 0,
                    },
                    ResidentRowMutationMergeKey {
                        property_name: 0,
                        input_key: 1,
                    },
                ],
                probe_obligation: obligation(
                    base + 2,
                    ResidentObligationKind::MutationSelect,
                    ResidentObligationScope::MutationCommand(0),
                ),
                effect_obligation: obligation(
                    base + 3,
                    ResidentObligationKind::MutationEffect,
                    ResidentObligationScope::MutationCommand(0),
                ),
            },
            sets: Vec::new(),
            outputs: vec![ResidentRowMutationOutput {
                name: "id".to_owned(),
                entity: 0,
                property_name: 0,
            }],
            source_obligation: obligation(
                base + 1,
                ResidentObligationKind::Expression,
                ResidentObligationScope::Expression(u16::MAX - 2),
            ),
            read_dependency_obligation: obligation(
                base + 4,
                ResidentObligationKind::MutationReadSet,
                ResidentObligationScope::Selection,
            ),
            final_relation_obligation: obligation(
                base + 5,
                ResidentObligationKind::Expression,
                ResidentObligationScope::Expression(u16::MAX - 1),
            ),
        },
        FIRST_NODE_ID,
        LayerMask::OBSERVED,
        Layer::Observed,
    )
    .expect_err("duplicate destination properties must be rejected even with distinct inputs");

    assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
    assert!(error.message.contains("duplicate"));
    Ok(())
}
