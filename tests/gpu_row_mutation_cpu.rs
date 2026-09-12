// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use irongraph::{
    Bookmark, ErrorCode, Layer, NodeId, ProjectId, Result, ScalarValue,
    gpu::{
        BackendKind, CpuBackend, ExecutionBackend, ResidentDeviceCompletion, ResidentExecutionId,
        ResidentExecutionObligation, ResidentMutationIntentEntry, ResidentObligationKind,
        ResidentObligationScope, ResidentProjectImage, ResidentRowMutationGeneration,
        ResidentRowMutationInputMap, ResidentRowMutationIntent, ResidentRowMutationIntentAction,
        ResidentRowMutationMergeKey, ResidentRowMutationMergeNode, ResidentRowMutationOutput,
        ResidentRowMutationProgram, ResidentRowMutationRequest, ResidentRowMutationSetProperty,
        ResidentRowMutationStringColumn, ResidentRowMutationTarget, ValidatedResidentRowMutation,
    },
    graph::{GraphStore, IndexCatalog, LayerMask, NodeInput, TemporalStore},
    types::{LabelId, PropertyId},
};
use tokio_util::sync::CancellationToken;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());
const MEMORY_LIMIT_BYTES: usize = 64 * 1024 * 1024;
const RESERVED_BYTES: usize = 4 * 1024 * 1024;
const FIRST_NODE_ID: u64 = 10_000;

struct Fixture {
    graph: GraphStore,
    bookmark: Bookmark,
    person: LabelId,
    login: PropertyId,
    name: PropertyId,
}

impl Fixture {
    fn new(existing: Option<(u64, &str, &str)>) -> Result<Self> {
        let mut graph = GraphStore::default();
        let person = graph.catalog_mut().intern_label("Person")?;
        let login = graph.catalog_mut().intern_property("login")?;
        let name = graph.catalog_mut().intern_property("name")?;
        if let Some((id, login_value, name_value)) = existing {
            graph.insert_node(NodeInput {
                id: NodeId(id),
                layer: Layer::Observed,
                revision: 1,
                labels: vec![person],
                properties: vec![(login, string(login_value)), (name, string(name_value))],
            })?;
        }
        let bookmark = Bookmark {
            term: 41,
            index: graph.revision(),
        };
        Ok(Self {
            graph,
            bookmark,
            person,
            login,
            name,
        })
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

    fn backend(&self) -> Result<CpuBackend> {
        let image = ResidentProjectImage::build(
            PROJECT,
            self.bookmark,
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
        self.request_for_generation(seed, input, self.generation())
    }

    fn request_for_generation(
        &self,
        seed: u64,
        input: Vec<ResidentRowMutationInputMap>,
        generation: ResidentRowMutationGeneration,
    ) -> Result<ResidentRowMutationRequest> {
        let base = seed
            .checked_mul(16)
            .and_then(|value| value.checked_add(1_000))
            .expect("test obligation IDs must fit in u64");
        ResidentRowMutationRequest::build(
            generation,
            ResidentExecutionId {
                high: 0x524f_574d_5554_4350,
                low: seed,
            },
            input,
            ResidentRowMutationProgram {
                input_keys: vec!["login".to_owned(), "name".to_owned()],
                property_names: vec!["login".to_owned(), "name".to_owned()],
                property_tokens: vec![Some(self.login), Some(self.name)],
                label_names: vec!["Person".to_owned()],
                label_tokens: vec![Some(self.person)],
                merge: ResidentRowMutationMergeNode {
                    output_entity: 0,
                    labels: vec![0],
                    keys: vec![ResidentRowMutationMergeKey {
                        property_name: 0,
                        input_key: 0,
                    }],
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
                sets: vec![ResidentRowMutationSetProperty {
                    target_entity: 0,
                    property_name: 1,
                    input_key: 1,
                    rhs_obligation: obligation(
                        base + 4,
                        ResidentObligationKind::MutationRhs,
                        ResidentObligationScope::MutationCommand(1),
                    ),
                    effect_obligation: obligation(
                        base + 5,
                        ResidentObligationKind::MutationEffect,
                        ResidentObligationScope::MutationCommand(1),
                    ),
                }],
                outputs: vec![
                    ResidentRowMutationOutput {
                        name: "login".to_owned(),
                        entity: 0,
                        property_name: 0,
                    },
                    ResidentRowMutationOutput {
                        name: "name".to_owned(),
                        entity: 0,
                        property_name: 1,
                    },
                ],
                source_obligation: obligation(
                    base + 1,
                    ResidentObligationKind::Expression,
                    ResidentObligationScope::Expression(u16::MAX - 2),
                ),
                read_dependency_obligation: obligation(
                    base + 6,
                    ResidentObligationKind::MutationReadSet,
                    ResidentObligationScope::Selection,
                ),
                final_relation_obligation: obligation(
                    base + 7,
                    ResidentObligationKind::Expression,
                    ResidentObligationScope::Expression(u16::MAX - 1),
                ),
            },
            FIRST_NODE_ID,
            LayerMask::OBSERVED,
            Layer::Observed,
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

fn string(value: &str) -> ScalarValue {
    ScalarValue::String(value.into())
}

fn input(login: ScalarValue, name: &str) -> ResidentRowMutationInputMap {
    ResidentRowMutationInputMap {
        entries: vec![
            ("login".to_owned(), login),
            ("name".to_owned(), string(name)),
        ],
    }
}

fn execute(
    backend: &CpuBackend,
    request: &ResidentRowMutationRequest,
) -> Result<ValidatedResidentRowMutation> {
    backend
        .execute_row_mutation(request, &CancellationToken::new())?
        .validate_for_publication(request, BackendKind::Cpu)
}

fn assert_provenance_and_receipts(
    request: &ResidentRowMutationRequest,
    result: &ValidatedResidentRowMutation,
) {
    let expected = request.obligations();
    let row_count = request.input.len() as u64;
    assert_eq!(result.receipts().len(), expected.len());
    for (index, (receipt, obligation)) in result.receipts().iter().zip(expected).enumerate() {
        assert_eq!(receipt.execution, request.execution);
        assert_eq!(receipt.obligation, obligation);
        assert_eq!(receipt.completion, ResidentDeviceCompletion::CpuReference);
        assert_eq!(
            receipt.input_cardinality,
            if index == 0 { 1 } else { row_count }
        );
        assert_eq!(receipt.output_cardinality, row_count);
    }
}

fn decode(column: &ResidentRowMutationStringColumn) -> Vec<Option<String>> {
    column
        .validity
        .iter()
        .enumerate()
        .map(|(row, valid)| {
            let start = usize::try_from(column.offsets[row]).expect("offset must fit usize");
            let end = usize::try_from(column.offsets[row + 1]).expect("offset must fit usize");
            (*valid != 0).then(|| {
                std::str::from_utf8(&column.bytes[start..end])
                    .expect("validated string column must contain UTF-8")
                    .to_owned()
            })
        })
        .collect()
}

fn assert_column(column: &ResidentRowMutationStringColumn, output: u16, expected: &[Option<&str>]) {
    assert_eq!(column.output, output);
    assert_eq!(column.offsets.first(), Some(&0));
    assert_eq!(
        column.offsets.last().copied(),
        Some(column.bytes.len() as u64)
    );
    assert_eq!(column.validity.len(), expected.len());
    assert_eq!(
        decode(column),
        expected
            .iter()
            .map(|value| value.map(str::to_owned))
            .collect::<Vec<_>>()
    );
}

#[test]
fn cpu_row_mutation_creates_two_distinct_string_keys_with_complete_provenance() -> Result<()> {
    let fixture = Fixture::new(None)?;
    let backend = fixture.backend()?;
    let request = fixture.request(
        1,
        vec![input(string("alice"), "Alice"), input(string("bob"), "Bob")],
    )?;

    let result = execute(&backend, &request)?;

    assert_provenance_and_receipts(&request, &result);
    assert!(result.read_dependencies().is_empty());
    assert_eq!(
        result.targets(),
        &[
            ResidentRowMutationTarget {
                target_id: FIRST_NODE_ID,
                dense_row: None,
                created_by: Some(0),
            },
            ResidentRowMutationTarget {
                target_id: FIRST_NODE_ID + 1,
                dense_row: None,
                created_by: Some(1),
            },
        ]
    );
    assert_eq!(
        result.intents(),
        &[
            ResidentRowMutationIntent {
                source_row: 0,
                command: 0,
                target_id: FIRST_NODE_ID,
                action: ResidentRowMutationIntentAction::CreateNode {
                    labels: vec![0],
                    properties: vec![ResidentMutationIntentEntry {
                        property_name: 0,
                        value: string("alice"),
                    }],
                },
            },
            ResidentRowMutationIntent {
                source_row: 0,
                command: 1,
                target_id: FIRST_NODE_ID,
                action: ResidentRowMutationIntentAction::SetProperty {
                    property_name: 1,
                    value: string("Alice"),
                },
            },
            ResidentRowMutationIntent {
                source_row: 1,
                command: 0,
                target_id: FIRST_NODE_ID + 1,
                action: ResidentRowMutationIntentAction::CreateNode {
                    labels: vec![0],
                    properties: vec![ResidentMutationIntentEntry {
                        property_name: 0,
                        value: string("bob"),
                    }],
                },
            },
            ResidentRowMutationIntent {
                source_row: 1,
                command: 1,
                target_id: FIRST_NODE_ID + 1,
                action: ResidentRowMutationIntentAction::SetProperty {
                    property_name: 1,
                    value: string("Bob"),
                },
            },
        ]
    );
    assert_eq!(result.columns().len(), 2);
    assert_column(&result.columns()[0], 0, &[Some("alice"), Some("bob")]);
    assert_column(&result.columns()[1], 1, &[Some("Alice"), Some("Bob")]);
    Ok(())
}

#[test]
fn cpu_row_mutation_duplicate_key_has_one_leader_and_final_overlay_for_both_rows() -> Result<()> {
    let fixture = Fixture::new(None)?;
    let backend = fixture.backend()?;
    let request = fixture.request(
        2,
        vec![
            input(string("same"), "first"),
            input(string("same"), "second"),
        ],
    )?;

    let result = execute(&backend, &request)?;

    assert_provenance_and_receipts(&request, &result);
    assert_eq!(
        result.targets(),
        &[
            ResidentRowMutationTarget {
                target_id: FIRST_NODE_ID,
                dense_row: None,
                created_by: Some(0),
            },
            ResidentRowMutationTarget {
                target_id: FIRST_NODE_ID,
                dense_row: None,
                created_by: Some(0),
            },
        ]
    );
    assert_eq!(
        result
            .intents()
            .iter()
            .filter(|intent| matches!(
                intent.action,
                ResidentRowMutationIntentAction::CreateNode { .. }
            ))
            .count(),
        1
    );
    assert_eq!(result.intents().len(), 3);
    assert_eq!(result.intents()[0].source_row, 0);
    assert_eq!(result.intents()[0].command, 0);
    assert_eq!(result.intents()[1].source_row, 0);
    assert_eq!(result.intents()[1].command, 1);
    assert_eq!(result.intents()[2].source_row, 1);
    assert_eq!(result.intents()[2].command, 1);
    assert_column(&result.columns()[0], 0, &[Some("same"), Some("same")]);
    assert_column(&result.columns()[1], 1, &[Some("second"), Some("second")]);
    Ok(())
}

#[test]
fn cpu_row_mutation_matches_existing_node_and_creates_missing_node() -> Result<()> {
    const EXISTING_ID: u64 = 77;
    let fixture = Fixture::new(Some((EXISTING_ID, "existing", "before")))?;
    let backend = fixture.backend()?;
    let request = fixture.request(
        3,
        vec![
            input(string("existing"), "updated"),
            input(string("new"), "created"),
        ],
    )?;

    let result = execute(&backend, &request)?;

    assert_provenance_and_receipts(&request, &result);
    assert_eq!(result.read_dependencies(), &[0]);
    assert_eq!(
        result.targets(),
        &[
            ResidentRowMutationTarget {
                target_id: EXISTING_ID,
                dense_row: Some(0),
                created_by: None,
            },
            ResidentRowMutationTarget {
                target_id: FIRST_NODE_ID + 1,
                dense_row: None,
                created_by: Some(1),
            },
        ]
    );
    assert_eq!(result.intents().len(), 3);
    assert!(matches!(
        &result.intents()[0],
        ResidentRowMutationIntent {
            source_row: 0,
            command: 1,
            target_id: EXISTING_ID,
            action: ResidentRowMutationIntentAction::SetProperty {
                property_name: 1,
                value,
            },
        } if value == &string("updated")
    ));
    assert!(matches!(
        &result.intents()[1],
        ResidentRowMutationIntent {
            source_row: 1,
            command: 0,
            target_id,
            action: ResidentRowMutationIntentAction::CreateNode { .. },
        } if *target_id == FIRST_NODE_ID + 1
    ));
    assert!(matches!(
        &result.intents()[2],
        ResidentRowMutationIntent {
            source_row: 1,
            command: 1,
            target_id,
            action: ResidentRowMutationIntentAction::SetProperty {
                property_name: 1,
                value,
            },
        } if *target_id == FIRST_NODE_ID + 1 && value == &string("created")
    ));
    assert_column(&result.columns()[0], 0, &[Some("existing"), Some("new")]);
    assert_column(&result.columns()[1], 1, &[Some("updated"), Some("created")]);
    Ok(())
}

#[test]
fn cpu_row_mutation_null_key_is_atomic_and_exposes_no_validated_partial_result() -> Result<()> {
    let fixture = Fixture::new(None)?;
    let backend = fixture.backend()?;
    let request = fixture.request(
        4,
        vec![
            input(string("would-create"), "first"),
            input(ScalarValue::Null, "invalid"),
        ],
    )?;

    let raw = backend.execute_row_mutation(&request, &CancellationToken::new())?;
    let error = raw
        .validate_for_publication(&request, BackendKind::Cpu)
        .expect_err("a null MERGE key must not produce a publishable partial result");

    assert_eq!(error.code, ErrorCode::QueryType);
    assert!(error.message.contains("login"));
    assert!(error.message.contains("source row 1"));
    Ok(())
}

#[test]
fn cpu_row_mutation_rejects_a_stale_generation_before_execution() -> Result<()> {
    let fixture = Fixture::new(None)?;
    let backend = fixture.backend()?;
    let mut stale = fixture.generation();
    stale.graph_revision = stale
        .graph_revision
        .checked_add(1)
        .expect("test graph revision must fit in u64");
    let request =
        fixture.request_for_generation(5, vec![input(string("stale"), "ignored")], stale)?;

    let error = backend
        .execute_row_mutation(&request, &CancellationToken::new())
        .expect_err("a stale row-mutation generation must be rejected");

    assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);
    assert!(error.message.contains("stale resident graph generation"));
    Ok(())
}
