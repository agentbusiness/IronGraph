// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]
#![cfg(all(feature = "accelerator", target_os = "macos"))]

use std::sync::{Arc, Mutex, MutexGuard};

use irongraph::{
    EdgeId, Layer, NodeId, ProjectId, ScalarValue,
    gpu::{
        BackendKind, CompareOp, CpuBackend, ExecutionBackend, MetalBackend, ResidentDirection,
        ResidentEntityBinding, ResidentExecutionId, ResidentExecutionObligation, ResidentExpansion,
        ResidentI64Predicate, ResidentMutationCommand, ResidentMutationIntentAction,
        ResidentMutationOperation, ResidentMutationProgram, ResidentMutationValueInstruction,
        ResidentNodeBinding, ResidentNodePipelineRequest, ResidentObligationKind,
        ResidentObligationScope,
    },
    graph::{EdgeInput, GraphStore, LayerMask, NodeInput},
};
use ordered_float::OrderedFloat;
use tokio_util::sync::CancellationToken;

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

#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_set_property_intents_match_cpu_and_validate_receipts() -> irongraph::Result<()> {
    let _metal = metal_test_guard();
    let mut graph = GraphStore::default();
    let person = graph.catalog_mut().intern_label("Person")?;
    let knows = graph.catalog_mut().intern_relationship_type("KNOWS")?;
    let count = graph.catalog_mut().intern_property("count")?;
    let name = graph.catalog_mut().intern_property("name")?;
    let score = graph.catalog_mut().intern_property("score")?;
    let weight = graph.catalog_mut().intern_property("weight")?;
    graph.insert_node(NodeInput {
        id: NodeId(41),
        layer: Layer::Observed,
        revision: 1,
        labels: vec![person],
        properties: vec![
            (count, ScalarValue::Integer(5)),
            (name, ScalarValue::String(Arc::from("Ada"))),
            (score, ScalarValue::Float(OrderedFloat(1.5))),
        ],
    })?;
    graph.insert_node(NodeInput {
        id: NodeId(42),
        layer: Layer::Observed,
        revision: 2,
        labels: vec![person],
        properties: Vec::new(),
    })?;
    graph.insert_edge(EdgeInput {
        id: EdgeId(91),
        source: NodeId(41),
        target: NodeId(42),
        relationship_type: knows,
        layer: Layer::Observed,
        revision: 3,
        properties: vec![(weight, ScalarValue::Integer(7))],
    })?;

    let execution = ResidentExecutionId {
        high: 0xfeed,
        low: 0xbeef,
    };
    let values = vec![
        // Parameter values are lowered by the resident compiler to this immutable Constant form.
        ResidentMutationValueInstruction::Constant(ScalarValue::Boolean(true)),
        ResidentMutationValueInstruction::Property {
            binding: ResidentEntityBinding::Node(ResidentNodeBinding::Start),
            property_name: 1,
            property: Some(count),
        },
        ResidentMutationValueInstruction::Constant(ScalarValue::Integer(1)),
        ResidentMutationValueInstruction::Add { left: 1, right: 2 },
        // This must observe command 1's device-produced count=6 intent, not canonical count=5.
        ResidentMutationValueInstruction::Property {
            binding: ResidentEntityBinding::Node(ResidentNodeBinding::Start),
            property_name: 1,
            property: Some(count),
        },
        ResidentMutationValueInstruction::Property {
            binding: ResidentEntityBinding::Relationship(0),
            property_name: 3,
            property: Some(weight),
        },
        ResidentMutationValueInstruction::Constant(ScalarValue::Integer(2)),
        ResidentMutationValueInstruction::Add { left: 5, right: 6 },
        ResidentMutationValueInstruction::Property {
            binding: ResidentEntityBinding::Node(ResidentNodeBinding::Start),
            property_name: 4,
            property: Some(name),
        },
        ResidentMutationValueInstruction::Constant(ScalarValue::String(Arc::from("!"))),
        ResidentMutationValueInstruction::Add { left: 8, right: 9 },
        ResidentMutationValueInstruction::Property {
            binding: ResidentEntityBinding::Node(ResidentNodeBinding::Start),
            property_name: 6,
            property: Some(score),
        },
    ];
    let command =
        |index: u16, value_start: u16, value_count: u16, operation: ResidentMutationOperation| {
            ResidentMutationCommand {
                rhs_obligation: obligation(
                    11 + u64::from(index),
                    ResidentObligationKind::MutationRhs,
                    ResidentObligationScope::MutationCommand(index),
                ),
                effect_obligation: obligation(
                    17 + u64::from(index),
                    ResidentObligationKind::MutationEffect,
                    ResidentObligationScope::MutationCommand(index),
                ),
                value_start,
                value_count,
                operation,
            }
        };
    let commands = vec![
        command(
            0,
            0,
            1,
            ResidentMutationOperation::SetProperty {
                target: ResidentEntityBinding::Node(ResidentNodeBinding::Start),
                property_name: 0,
                value: 0,
            },
        ),
        command(
            1,
            1,
            3,
            ResidentMutationOperation::SetProperty {
                target: ResidentEntityBinding::Node(ResidentNodeBinding::Start),
                property_name: 1,
                value: 3,
            },
        ),
        command(
            2,
            4,
            1,
            ResidentMutationOperation::SetProperty {
                target: ResidentEntityBinding::Node(ResidentNodeBinding::Start),
                property_name: 2,
                value: 4,
            },
        ),
        command(
            3,
            5,
            3,
            ResidentMutationOperation::SetProperty {
                target: ResidentEntityBinding::Relationship(0),
                property_name: 3,
                value: 7,
            },
        ),
        command(
            4,
            8,
            3,
            ResidentMutationOperation::SetProperty {
                target: ResidentEntityBinding::Relationship(0),
                property_name: 5,
                value: 10,
            },
        ),
        command(
            5,
            11,
            1,
            ResidentMutationOperation::SetProperty {
                target: ResidentEntityBinding::Node(ResidentNodeBinding::Start),
                property_name: 7,
                value: 11,
            },
        ),
    ];
    let semantic_registers = [1_u16, 3, 4, 5, 7, 8, 10, 11];
    let program = ResidentMutationProgram {
        execution,
        selection_obligation: obligation(
            1,
            ResidentObligationKind::MutationSelect,
            ResidentObligationScope::Selection,
        ),
        filter_obligations: vec![obligation(
            2,
            ResidentObligationKind::Filter,
            ResidentObligationScope::Filter(0),
        )],
        expression_obligations: semantic_registers
            .into_iter()
            .enumerate()
            .map(|(position, register)| {
                obligation(
                    3 + position as u64,
                    ResidentObligationKind::Expression,
                    ResidentObligationScope::Expression(register),
                )
            })
            .collect(),
        property_names: vec![
            "flag".into(),
            "count".into(),
            "mirror".into(),
            "weight".into(),
            "name".into(),
            "note".into(),
            "score".into(),
            "ratio".into(),
        ],
        property_tokens: vec![
            None,
            Some(count),
            None,
            Some(weight),
            Some(name),
            None,
            Some(score),
            None,
        ],
        label_names: Vec::new(),
        label_tokens: Vec::new(),
        values,
        commands,
        max_intents: 6,
        continuation: None,
    };
    let project = ProjectId(uuid::Uuid::nil());
    let request = ResidentNodePipelineRequest {
        project,
        labels: vec![person],
        layers: LayerMask::OBSERVED,
        initial_optional: false,
        expansion: Some(ResidentExpansion {
            direction: ResidentDirection::Outgoing,
            relationship_types: vec![knows],
            end_labels: Vec::new(),
            end_equals_start: false,
            optional: false,
            end_predicates: Vec::new(),
        }),
        continuations: Vec::new(),
        correlated_optional: None,
        relationship_null_filter: None,
        predicates: vec![ResidentI64Predicate {
            binding: ResidentNodeBinding::Start,
            property: count,
            operation: CompareOp::Greater,
            operand: 0,
        }],
        property_filters: Vec::new(),
        value_matrix: None,
        mutation: Some(program),
        orders: Vec::new(),
        offset: 0,
        limit: usize::MAX,
        integer_projections: Vec::new(),
        property_null_projections: Vec::new(),
        max_output_rows: 8,
    };
    request.validate_mutation()?;

    let snapshot = Arc::new(graph.snapshot()?);
    let mut cpu = CpuBackend::new(64 * 1024 * 1024, 1024 * 1024);
    let mut metal = MetalBackend::new(0, 64 * 1024 * 1024, 1024 * 1024)?;
    cpu.admit_graph(Arc::clone(&snapshot))?;
    metal.admit_graph(snapshot)?;
    let cancellation = CancellationToken::new();
    let cpu_result = cpu.execute_node_pipeline(&request, &cancellation)?;
    let metal_result = metal.execute_node_pipeline(&request, &cancellation)?;
    assert_eq!(metal_result.start_rows, cpu_result.start_rows);
    assert_eq!(metal_result.edge_rows, cpu_result.edge_rows);
    assert_eq!(metal_result.end_rows, cpu_result.end_rows);

    let cpu_raw = cpu_result
        .mutation
        .ok_or_else(|| irongraph::Error::internal("CPU mutation result is absent"))?;
    let metal_raw = metal_result
        .mutation
        .ok_or_else(|| irongraph::Error::internal("Metal mutation result is absent"))?;
    let forged_raw = metal_raw.clone();
    let cpu_batch = cpu_raw.validate_for_publication(&request, BackendKind::Cpu)?;
    let metal_batch = metal_raw.validate_for_publication(&request, BackendKind::Metal)?;
    assert_eq!(metal_batch.intents(), cpu_batch.intents());
    assert_eq!(metal_batch.intents().len(), 6);
    assert!(
        metal_batch.receipts().iter().all(|receipt| {
            receipt.completion == irongraph::gpu::ResidentDeviceCompletion::Metal
        })
    );

    let mut forged_request = request.clone();
    let forged_program = forged_request
        .mutation
        .as_mut()
        .ok_or_else(|| irongraph::Error::internal("forged mutation program is absent"))?;
    forged_program.selection_obligation.id += 1000;
    assert!(
        forged_raw
            .validate_for_publication(&forged_request, BackendKind::Metal)
            .is_err()
    );
    Ok(())
}

#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_null_relationship_target_skips_invalid_rhs_and_receipts_zero_work() -> irongraph::Result<()>
{
    let _metal = metal_test_guard();
    let mut graph = GraphStore::default();
    let person = graph.catalog_mut().intern_label("Person")?;
    let knows = graph.catalog_mut().intern_relationship_type("KNOWS")?;
    graph.insert_node(NodeInput {
        id: NodeId(77),
        layer: Layer::Observed,
        revision: 1,
        labels: vec![person],
        properties: Vec::new(),
    })?;
    let execution = ResidentExecutionId { high: 9, low: 17 };
    let program = ResidentMutationProgram {
        execution,
        selection_obligation: obligation(
            1,
            ResidentObligationKind::MutationSelect,
            ResidentObligationScope::Selection,
        ),
        filter_obligations: Vec::new(),
        expression_obligations: vec![obligation(
            2,
            ResidentObligationKind::Expression,
            ResidentObligationScope::Expression(2),
        )],
        property_names: vec!["impossible".into()],
        property_tokens: vec![None],
        label_names: Vec::new(),
        label_tokens: Vec::new(),
        values: vec![
            ResidentMutationValueInstruction::Constant(ScalarValue::Boolean(true)),
            ResidentMutationValueInstruction::Constant(ScalarValue::Integer(1)),
            // This is a runtime type error only if the RHS is incorrectly evaluated.
            ResidentMutationValueInstruction::Add { left: 0, right: 1 },
        ],
        commands: vec![ResidentMutationCommand {
            rhs_obligation: obligation(
                3,
                ResidentObligationKind::MutationRhs,
                ResidentObligationScope::MutationCommand(0),
            ),
            effect_obligation: obligation(
                4,
                ResidentObligationKind::MutationEffect,
                ResidentObligationScope::MutationCommand(0),
            ),
            value_start: 0,
            value_count: 3,
            operation: ResidentMutationOperation::SetProperty {
                target: ResidentEntityBinding::Relationship(0),
                property_name: 0,
                value: 2,
            },
        }],
        max_intents: 1,
        continuation: None,
    };
    let project = ProjectId(uuid::Uuid::nil());
    let request = ResidentNodePipelineRequest {
        project,
        labels: vec![person],
        layers: LayerMask::OBSERVED,
        initial_optional: false,
        expansion: Some(ResidentExpansion {
            direction: ResidentDirection::Outgoing,
            relationship_types: vec![knows],
            end_labels: Vec::new(),
            end_equals_start: false,
            optional: true,
            end_predicates: Vec::new(),
        }),
        continuations: Vec::new(),
        correlated_optional: None,
        relationship_null_filter: None,
        predicates: Vec::new(),
        property_filters: Vec::new(),
        value_matrix: None,
        mutation: Some(program),
        orders: Vec::new(),
        offset: 0,
        limit: usize::MAX,
        integer_projections: Vec::new(),
        property_null_projections: Vec::new(),
        max_output_rows: 4,
    };
    request.validate_mutation()?;
    let snapshot = Arc::new(graph.snapshot()?);
    let mut cpu = CpuBackend::new(32 * 1024 * 1024, 1024 * 1024);
    let mut metal = MetalBackend::new(0, 32 * 1024 * 1024, 1024 * 1024)?;
    cpu.admit_graph(Arc::clone(&snapshot))?;
    metal.admit_graph(snapshot)?;
    let cancellation = CancellationToken::new();
    let cpu_result = cpu.execute_node_pipeline(&request, &cancellation)?;
    let metal_result = metal.execute_node_pipeline(&request, &cancellation)?;
    let cpu_batch = cpu_result
        .mutation
        .ok_or_else(|| irongraph::Error::internal("CPU null-target result is absent"))?
        .validate_for_publication(&request, BackendKind::Cpu)?;
    let metal_batch = metal_result
        .mutation
        .ok_or_else(|| irongraph::Error::internal("Metal null-target result is absent"))?
        .validate_for_publication(&request, BackendKind::Metal)?;
    assert!(cpu_batch.intents().is_empty());
    assert_eq!(metal_batch.intents(), cpu_batch.intents());
    assert!(metal_batch.receipts().iter().any(|receipt| {
        receipt.obligation.kind == ResidentObligationKind::Expression
            && receipt.input_cardinality == 0
            && receipt.output_cardinality == 0
    }));
    assert!(metal_batch.receipts().iter().any(|receipt| {
        receipt.obligation.kind == ResidentObligationKind::MutationRhs
            && receipt.input_cardinality == 1
            && receipt.output_cardinality == 0
    }));
    Ok(())
}

#[test]
#[ignore = "requires an available physical Metal device"]
fn metal_duplicate_rows_observe_prior_row_property_writes() -> irongraph::Result<()> {
    let _metal = metal_test_guard();
    let mut graph = GraphStore::default();
    let person = graph.catalog_mut().intern_label("Person")?;
    let knows = graph.catalog_mut().intern_relationship_type("KNOWS")?;
    let count = graph.catalog_mut().intern_property("count")?;
    graph.insert_node(NodeInput {
        id: NodeId(101),
        layer: Layer::Observed,
        revision: 1,
        labels: vec![person],
        properties: vec![(count, ScalarValue::Integer(5))],
    })?;
    for (edge, node) in [(201, 102), (202, 103)] {
        graph.insert_node(NodeInput {
            id: NodeId(node),
            layer: Layer::Observed,
            revision: 1,
            labels: Vec::new(),
            properties: Vec::new(),
        })?;
        graph.insert_edge(EdgeInput {
            id: EdgeId(edge),
            source: NodeId(101),
            target: NodeId(node),
            relationship_type: knows,
            layer: Layer::Observed,
            revision: 1,
            properties: Vec::new(),
        })?;
    }

    let program = ResidentMutationProgram {
        execution: ResidentExecutionId {
            high: 0x5eed,
            low: 0xcafe,
        },
        selection_obligation: obligation(
            1,
            ResidentObligationKind::MutationSelect,
            ResidentObligationScope::Selection,
        ),
        filter_obligations: Vec::new(),
        expression_obligations: vec![
            obligation(
                2,
                ResidentObligationKind::Expression,
                ResidentObligationScope::Expression(0),
            ),
            obligation(
                3,
                ResidentObligationKind::Expression,
                ResidentObligationScope::Expression(2),
            ),
        ],
        property_names: vec!["count".into()],
        property_tokens: vec![Some(count)],
        label_names: Vec::new(),
        label_tokens: Vec::new(),
        values: vec![
            ResidentMutationValueInstruction::Property {
                binding: ResidentEntityBinding::Node(ResidentNodeBinding::Start),
                property_name: 0,
                property: Some(count),
            },
            ResidentMutationValueInstruction::Constant(ScalarValue::Integer(1)),
            ResidentMutationValueInstruction::Add { left: 0, right: 1 },
        ],
        commands: vec![ResidentMutationCommand {
            rhs_obligation: obligation(
                4,
                ResidentObligationKind::MutationRhs,
                ResidentObligationScope::MutationCommand(0),
            ),
            effect_obligation: obligation(
                5,
                ResidentObligationKind::MutationEffect,
                ResidentObligationScope::MutationCommand(0),
            ),
            value_start: 0,
            value_count: 3,
            operation: ResidentMutationOperation::SetProperty {
                target: ResidentEntityBinding::Node(ResidentNodeBinding::Start),
                property_name: 0,
                value: 2,
            },
        }],
        max_intents: 2,
        continuation: None,
    };
    let request = ResidentNodePipelineRequest {
        project: ProjectId(uuid::Uuid::nil()),
        labels: vec![person],
        layers: LayerMask::OBSERVED,
        initial_optional: false,
        expansion: Some(ResidentExpansion {
            direction: ResidentDirection::Outgoing,
            relationship_types: vec![knows],
            end_labels: Vec::new(),
            end_equals_start: false,
            optional: false,
            end_predicates: Vec::new(),
        }),
        continuations: Vec::new(),
        correlated_optional: None,
        relationship_null_filter: None,
        predicates: Vec::new(),
        property_filters: Vec::new(),
        value_matrix: None,
        mutation: Some(program),
        orders: Vec::new(),
        offset: 0,
        limit: usize::MAX,
        integer_projections: Vec::new(),
        property_null_projections: Vec::new(),
        max_output_rows: 4,
    };
    request.validate_mutation()?;

    let snapshot = Arc::new(graph.snapshot()?);
    let mut cpu = CpuBackend::new(32 * 1024 * 1024, 1024 * 1024);
    let mut metal = MetalBackend::new(0, 32 * 1024 * 1024, 1024 * 1024)?;
    cpu.admit_graph(Arc::clone(&snapshot))?;
    metal.admit_graph(snapshot)?;
    let cancellation = CancellationToken::new();
    let cpu_batch = cpu
        .execute_node_pipeline(&request, &cancellation)?
        .mutation
        .ok_or_else(|| irongraph::Error::internal("CPU duplicate-row result is absent"))?
        .validate_for_publication(&request, BackendKind::Cpu)?;
    let metal_batch = metal
        .execute_node_pipeline(&request, &cancellation)?
        .mutation
        .ok_or_else(|| irongraph::Error::internal("Metal duplicate-row result is absent"))?
        .validate_for_publication(&request, BackendKind::Metal)?;
    assert_eq!(metal_batch.intents(), cpu_batch.intents());
    let values = metal_batch
        .intents()
        .iter()
        .map(|intent| match &intent.action {
            ResidentMutationIntentAction::SetProperty { value, .. } => Ok(value.clone()),
            action => Err(irongraph::Error::internal(format!(
                "unexpected duplicate-row intent action: {action:?}"
            ))),
        })
        .collect::<irongraph::Result<Vec<_>>>()?;
    assert_eq!(
        values,
        vec![ScalarValue::Integer(6), ScalarValue::Integer(7)]
    );
    Ok(())
}
