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
        ResidentEntityBinding, ResidentExecutionId, ResidentExpansion, ResidentNodeBinding,
        ResidentNodePipelineRequest, ResidentObligationKind, ResidentProjectImage,
        ResidentRowColumn, ResidentRowInstruction, ResidentRowManifestFingerprint,
        ResidentRowOperation, ResidentRowProgram, ResidentRowProgramManifest,
        ResidentRowProgramRequest, ResidentRowProgramResult, ResidentRowSortKey,
        ResidentRowValueType,
    },
    graph::{EdgeInput, GraphStore, IndexCatalog, LayerMask, NodeInput, TemporalStore},
    types::{LabelId, PropertyId, RelationshipTypeId},
};
use ordered_float::OrderedFloat;
use tokio_util::sync::CancellationToken;

const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());
const MEMORY_LIMIT_BYTES: usize = 128 * 1024 * 1024;
const RESERVED_BYTES: usize = 8 * 1024 * 1024;
const ROWS: usize = 6;

struct Fixture {
    graph: GraphStore,
    bookmark: Bookmark,
    start_label: LabelId,
    mid_label: LabelId,
    end_label: LabelId,
    relationship_type: RelationshipTypeId,
    flag: PropertyId,
    gate: PropertyId,
    num: PropertyId,
    num2: PropertyId,
    ratio: PropertyId,
    float_order: PropertyId,
    mixed: PropertyId,
    absent: PropertyId,
}

impl Fixture {
    fn new() -> Result<Self> {
        let mut graph = GraphStore::default();
        let start_label = graph.catalog_mut().intern_label("Start")?;
        let mid_label = graph.catalog_mut().intern_label("Mid")?;
        let end_label = graph.catalog_mut().intern_label("End")?;
        let relationship_type = graph.catalog_mut().intern_relationship_type("NEXT")?;
        let flag = graph.catalog_mut().intern_property("flag")?;
        let gate = graph.catalog_mut().intern_property("gate")?;
        let num = graph.catalog_mut().intern_property("num")?;
        let num2 = graph.catalog_mut().intern_property("num2")?;
        let ratio = graph.catalog_mut().intern_property("ratio")?;
        let float_order = graph.catalog_mut().intern_property("float_order")?;
        let mixed = graph.catalog_mut().intern_property("mixed")?;
        let absent = graph.catalog_mut().intern_property("absent")?;

        let nums = [Some(1), Some(3), Some(2), Some(2), None, Some(4)];
        let num2s = [10, 5, 4, 4, 9, 0];
        let flags = [Some(true), Some(false), None, Some(true), None, Some(false)];
        let gates = [Some(true), None, Some(true), Some(false), None, Some(true)];
        let ratios = [
            Some(0.5),
            Some(-1.0),
            Some(2.25),
            Some(2.25),
            None,
            Some(-0.0),
        ];
        let float_orders = [
            Some(f64::NAN),
            Some(f64::INFINITY),
            Some(f64::NEG_INFINITY),
            Some(-0.0),
            Some(0.0),
            None,
        ];
        for row in 0..ROWS {
            let mut properties = vec![(num2, ScalarValue::Integer(num2s[row]))];
            if let Some(value) = nums[row] {
                properties.push((num, ScalarValue::Integer(value)));
            }
            if let Some(value) = flags[row] {
                properties.push((flag, ScalarValue::Boolean(value)));
            }
            if let Some(value) = gates[row] {
                properties.push((gate, ScalarValue::Boolean(value)));
            }
            if let Some(value) = ratios[row] {
                properties.push((ratio, ScalarValue::Float(OrderedFloat(value))));
            }
            if let Some(value) = float_orders[row] {
                properties.push((float_order, ScalarValue::Float(OrderedFloat(value))));
            }
            if row == 0 {
                properties.push((mixed, ScalarValue::Integer(1)));
            } else if row == 1 {
                properties.push((mixed, ScalarValue::Boolean(true)));
            }
            graph.insert_node(NodeInput {
                id: NodeId(100 + row as u64),
                layer: Layer::Observed,
                revision: 1 + row as u64,
                labels: vec![start_label],
                properties,
            })?;
        }
        for row in 0..ROWS {
            graph.insert_node(NodeInput {
                id: NodeId(200 + row as u64),
                layer: Layer::Observed,
                revision: 20 + row as u64,
                labels: vec![mid_label],
                properties: Vec::new(),
            })?;
        }
        for row in 0..ROWS {
            graph.insert_node(NodeInput {
                id: NodeId(300 + row as u64),
                layer: Layer::Observed,
                revision: 40 + row as u64,
                labels: vec![end_label],
                properties: Vec::new(),
            })?;
        }
        for row in 0..ROWS {
            graph.insert_edge(EdgeInput {
                id: EdgeId(400 + row as u64),
                source: NodeId(100 + row as u64),
                target: NodeId(200 + row as u64),
                relationship_type,
                layer: Layer::Observed,
                revision: 60 + row as u64,
                properties: vec![(num, ScalarValue::Integer(row as i64))],
            })?;
        }
        for row in 0..ROWS {
            graph.insert_edge(EdgeInput {
                id: EdgeId(500 + row as u64),
                source: NodeId(200 + row as u64),
                target: NodeId(300 + row as u64),
                relationship_type,
                layer: Layer::Observed,
                revision: 80 + row as u64,
                properties: Vec::new(),
            })?;
        }
        let bookmark = Bookmark {
            term: 7,
            index: graph.revision(),
        };
        Ok(Self {
            graph,
            bookmark,
            start_label,
            mid_label,
            end_label,
            relationship_type,
            flag,
            gate,
            num,
            num2,
            ratio,
            float_order,
            mixed,
            absent,
        })
    }

    fn backend(&self) -> Result<CpuBackend> {
        let image = ResidentProjectImage::build(
            PROJECT,
            self.bookmark,
            &self.graph,
            &TemporalStore::default(),
            &IndexCatalog::default(),
        )?;
        let mut cpu = CpuBackend::new(MEMORY_LIMIT_BYTES, RESERVED_BYTES);
        cpu.admit_project(image)?;
        Ok(cpu)
    }

    fn input(&self) -> ResidentNodePipelineRequest {
        ResidentNodePipelineRequest {
            project: PROJECT,
            labels: vec![self.start_label],
            layers: LayerMask::OBSERVED,
            initial_optional: false,
            expansion: Some(ResidentExpansion {
                direction: ResidentDirection::Outgoing,
                relationship_types: vec![self.relationship_type],
                end_labels: vec![self.mid_label],
                end_equals_start: false,
                optional: false,
                end_predicates: Vec::new(),
            }),
            continuations: vec![ResidentExpansion {
                direction: ResidentDirection::Outgoing,
                relationship_types: vec![self.relationship_type],
                end_labels: vec![self.end_label],
                end_equals_start: false,
                optional: false,
                end_predicates: Vec::new(),
            }],
            correlated_optional: None,
            relationship_null_filter: None,
            predicates: Vec::new(),
            property_filters: Vec::new(),
            value_matrix: None,
            mutation: None,
            orders: Vec::new(),
            offset: 0,
            limit: usize::MAX,
            integer_projections: Vec::new(),
            property_null_projections: Vec::new(),
            max_output_rows: ROWS,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn request(
        &self,
        seed: u64,
        program: ResidentRowProgram,
        sort_keys: Vec<ResidentRowSortKey>,
        offset: usize,
        limit: usize,
        max_output_rows: usize,
        final_registers: Vec<u16>,
    ) -> Result<ResidentRowProgramRequest> {
        let manifest = ResidentRowProgramManifest::build(
            &program,
            &sort_keys,
            offset,
            limit,
            max_output_rows,
            &final_registers,
            seed.checked_mul(1_000)
                .and_then(|value| value.checked_add(1))
                .ok_or_else(|| irongraph::Error::internal("test obligation ID overflow"))?,
        )?;
        Ok(ResidentRowProgramRequest {
            project: PROJECT,
            expected_bookmark: self.bookmark,
            expected_graph_revision: self.graph.revision(),
            expected_layout_version: self.graph.layout_version(),
            execution: ResidentExecutionId {
                high: 0x524f_5750_524f_4752,
                low: seed,
            },
            input: self.input(),
            program,
            manifest,
            sort_keys,
            offset,
            limit,
            max_output_rows,
            final_registers,
        })
    }
}

fn instruction(
    output_type: ResidentRowValueType,
    operation: ResidentRowOperation,
) -> ResidentRowInstruction {
    ResidentRowInstruction {
        output_type,
        operation,
    }
}

fn node(binding: ResidentNodeBinding) -> ResidentEntityBinding {
    ResidentEntityBinding::Node(binding)
}

fn execute(
    cpu: &CpuBackend,
    request: &ResidentRowProgramRequest,
) -> Result<irongraph::gpu::ValidatedResidentRowProgramResult> {
    cpu.execute_row_program(request, &CancellationToken::new())?
        .validate(request, BackendKind::Cpu)
}

fn integer_column(column: &ResidentRowColumn) -> (&[i64], &[u8]) {
    let ResidentRowColumn::Integer { values, validity } = column else {
        panic!("expected integer column");
    };
    (values, validity)
}

fn boolean_column(column: &ResidentRowColumn) -> (&[u8], &[u8]) {
    let ResidentRowColumn::Boolean { values, validity } = column else {
        panic!("expected Boolean column");
    };
    (values, validity)
}

fn float_column(column: &ResidentRowColumn) -> (&[u64], &[u8]) {
    let ResidentRowColumn::Float { bits, validity } = column else {
        panic!("expected float column");
    };
    (bits, validity)
}

fn assert_graph_alignment(
    result: &irongraph::gpu::ValidatedResidentRowProgramResult,
    source_positions: &[u64],
) {
    let rows = result.rows();
    assert_eq!(result.source_positions(), source_positions);
    let graph_positions = source_positions
        .iter()
        .map(|position| u32::try_from(*position).expect("fixture position exceeds u32"))
        .collect::<Vec<_>>();
    assert_eq!(rows.start_rows, graph_positions);
    assert_eq!(rows.intermediate_node_rows.len(), 1);
    assert_eq!(rows.intermediate_edge_rows.len(), 1);
    assert_eq!(
        rows.intermediate_node_rows[0],
        graph_positions
            .iter()
            .map(|position| position + ROWS as u32)
            .collect::<Vec<_>>()
    );
    assert_eq!(rows.intermediate_edge_rows[0], graph_positions);
    assert_eq!(
        rows.edge_rows,
        graph_positions
            .iter()
            .map(|position| position + ROWS as u32)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        rows.end_rows,
        graph_positions
            .iter()
            .map(|position| position + (ROWS * 2) as u32)
            .collect::<Vec<_>>()
    );
}

fn integer_expression_program(fixture: &Fixture) -> ResidentRowProgram {
    ResidentRowProgram {
        instructions: vec![
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::LoadIntegerProperty {
                    binding: node(ResidentNodeBinding::Start),
                    property: fixture.num,
                },
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::IntegerConstant(2),
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::NumericMultiply { left: 0, right: 1 },
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::LoadIntegerProperty {
                    binding: node(ResidentNodeBinding::Start),
                    property: fixture.num2,
                },
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::NumericAdd { left: 3, right: 2 },
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::NumericNegate { operand: 4 },
            ),
        ],
    }
}

#[test]
fn boolean_not_and_use_cypher_null_logic_and_stable_order() -> Result<()> {
    let fixture = Fixture::new()?;
    let cpu = fixture.backend()?;
    let program = ResidentRowProgram {
        instructions: vec![
            instruction(
                ResidentRowValueType::Boolean,
                ResidentRowOperation::LoadBooleanProperty {
                    binding: node(ResidentNodeBinding::Start),
                    property: fixture.flag,
                },
            ),
            instruction(
                ResidentRowValueType::Boolean,
                ResidentRowOperation::BooleanNot { operand: 0 },
            ),
            instruction(
                ResidentRowValueType::Boolean,
                ResidentRowOperation::LoadBooleanProperty {
                    binding: node(ResidentNodeBinding::Start),
                    property: fixture.gate,
                },
            ),
            instruction(
                ResidentRowValueType::Boolean,
                ResidentRowOperation::BooleanAnd { left: 0, right: 2 },
            ),
        ],
    };
    let request = fixture.request(
        1,
        program,
        vec![ResidentRowSortKey {
            register: 1,
            descending: false,
            nulls_first: false,
        }],
        0,
        usize::MAX,
        ROWS,
        vec![0, 1, 3],
    )?;
    let result = execute(&cpu, &request)?;
    assert_graph_alignment(&result, &[0, 3, 1, 5, 2, 4]);

    let (not_values, not_validity) = boolean_column(&result.projected_columns()[1].column);
    assert_eq!(not_values, &[0, 0, 1, 1, 1, 1]);
    assert_eq!(not_validity, &[1, 1, 1, 1, 0, 0]);
    let (and_values, and_validity) = boolean_column(&result.projected_columns()[2].column);
    assert_eq!(and_values, &[1, 0, 0, 0, 0, 0]);
    assert_eq!(and_validity, &[1, 1, 1, 1, 0, 0]);
    assert_eq!(result.receipts().len(), 5);
    assert!(result.receipts()[..4].iter().all(|receipt| {
        receipt.completion == ResidentDeviceCompletion::CpuReference
            && receipt.input_cardinality == ROWS as u64
            && receipt.output_cardinality == ROWS as u64
    }));
    assert_eq!(
        result.receipts()[4].obligation.kind,
        ResidentObligationKind::Sort
    );
    Ok(())
}

#[test]
fn integer_expression_sorts_both_directions_with_stable_ties_and_alignment() -> Result<()> {
    let fixture = Fixture::new()?;
    let cpu = fixture.backend()?;
    let program = integer_expression_program(&fixture);
    let ascending = fixture.request(
        2,
        program.clone(),
        vec![ResidentRowSortKey {
            register: 5,
            descending: false,
            nulls_first: false,
        }],
        0,
        usize::MAX,
        ROWS,
        vec![5],
    )?;
    let result = execute(&cpu, &ascending)?;
    assert_graph_alignment(&result, &[0, 1, 2, 3, 5, 4]);
    let (values, validity) = integer_column(&result.projected_columns()[0].column);
    assert_eq!(values, &[-12, -11, -8, -8, -8, 0]);
    assert_eq!(validity, &[1, 1, 1, 1, 1, 0]);
    assert_eq!(&result.source_positions()[2..5], &[2, 3, 5]);

    let descending = fixture.request(
        3,
        program,
        vec![ResidentRowSortKey {
            register: 5,
            descending: true,
            nulls_first: false,
        }],
        0,
        usize::MAX,
        ROWS,
        vec![5],
    )?;
    let result = execute(&cpu, &descending)?;
    assert_graph_alignment(&result, &[2, 3, 5, 1, 0, 4]);
    let (values, validity) = integer_column(&result.projected_columns()[0].column);
    assert_eq!(values, &[-8, -8, -8, -11, -12, 0]);
    assert_eq!(validity, &[1, 1, 1, 1, 1, 0]);
    Ok(())
}

#[test]
fn mixed_numeric_promotion_float_order_and_sign_bit_negation_are_exact() -> Result<()> {
    let fixture = Fixture::new()?;
    let cpu = fixture.backend()?;
    let mixed = ResidentRowProgram {
        instructions: vec![
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::LoadIntegerProperty {
                    binding: node(ResidentNodeBinding::Start),
                    property: fixture.num,
                },
            ),
            instruction(
                ResidentRowValueType::Float,
                ResidentRowOperation::LoadFloatProperty {
                    binding: node(ResidentNodeBinding::Start),
                    property: fixture.ratio,
                },
            ),
            instruction(
                ResidentRowValueType::Float,
                ResidentRowOperation::NumericAdd { left: 0, right: 1 },
            ),
            instruction(
                ResidentRowValueType::Float,
                ResidentRowOperation::FloatConstant(0.0_f64.to_bits()),
            ),
            instruction(
                ResidentRowValueType::Float,
                ResidentRowOperation::NumericNegate { operand: 3 },
            ),
        ],
    };
    let request = fixture.request(
        4,
        mixed,
        vec![ResidentRowSortKey {
            register: 2,
            descending: false,
            nulls_first: false,
        }],
        0,
        usize::MAX,
        ROWS,
        vec![2, 4],
    )?;
    let result = execute(&cpu, &request)?;
    assert_graph_alignment(&result, &[0, 1, 5, 2, 3, 4]);
    let (bits, validity) = float_column(&result.projected_columns()[0].column);
    assert_eq!(
        bits.iter()
            .map(|bits| f64::from_bits(*bits))
            .collect::<Vec<_>>(),
        vec![1.5, 2.0, 4.0, 4.25, 4.25, 0.0]
    );
    assert_eq!(validity, &[1, 1, 1, 1, 1, 0]);
    let (negated_zero, validity) = float_column(&result.projected_columns()[1].column);
    assert!(
        negated_zero
            .iter()
            .all(|bits| *bits == (-0.0_f64).to_bits())
    );
    assert!(validity.iter().all(|valid| *valid == 1));

    let float_order = ResidentRowProgram {
        instructions: vec![instruction(
            ResidentRowValueType::Float,
            ResidentRowOperation::LoadFloatProperty {
                binding: node(ResidentNodeBinding::Start),
                property: fixture.float_order,
            },
        )],
    };
    let request = fixture.request(
        5,
        float_order,
        vec![ResidentRowSortKey {
            register: 0,
            descending: false,
            nulls_first: false,
        }],
        0,
        usize::MAX,
        ROWS,
        vec![0],
    )?;
    let result = execute(&cpu, &request)?;
    assert_graph_alignment(&result, &[2, 3, 4, 1, 0, 5]);
    let (bits, validity) = float_column(&result.projected_columns()[0].column);
    assert_eq!(f64::from_bits(bits[0]), f64::NEG_INFINITY);
    assert_eq!(f64::from_bits(bits[1]), -0.0);
    assert_eq!(f64::from_bits(bits[2]), 0.0);
    assert_eq!(f64::from_bits(bits[3]), f64::INFINITY);
    assert!(f64::from_bits(bits[4]).is_nan());
    assert_eq!(validity, &[1, 1, 1, 1, 1, 0]);
    Ok(())
}

#[test]
fn sort_offset_limit_top_k_null_order_and_budget_never_truncate() -> Result<()> {
    let fixture = Fixture::new()?;
    let cpu = fixture.backend()?;
    let program = integer_expression_program(&fixture);
    let page = fixture.request(
        6,
        program.clone(),
        vec![ResidentRowSortKey {
            register: 5,
            descending: false,
            nulls_first: false,
        }],
        2,
        2,
        2,
        vec![5],
    )?;
    let result = execute(&cpu, &page)?;
    assert_graph_alignment(&result, &[2, 3]);
    assert_eq!(
        result.receipts().last().unwrap().input_cardinality,
        ROWS as u64
    );
    assert_eq!(result.receipts().last().unwrap().output_cardinality, 2);
    assert_eq!(result.scratch_bytes(), page.scratch_bytes(ROWS)?);

    let nulls_first = fixture.request(
        7,
        program.clone(),
        vec![ResidentRowSortKey {
            register: 5,
            descending: false,
            nulls_first: true,
        }],
        0,
        usize::MAX,
        ROWS,
        vec![5],
    )?;
    let result = execute(&cpu, &nulls_first)?;
    assert_graph_alignment(&result, &[4, 0, 1, 2, 3, 5]);

    let too_small = fixture.request(
        8,
        program,
        vec![ResidentRowSortKey {
            register: 5,
            descending: false,
            nulls_first: false,
        }],
        2,
        2,
        1,
        vec![5],
    )?;
    let error = cpu
        .execute_row_program(&too_small, &CancellationToken::new())
        .unwrap_err();
    assert_eq!(error.code, ErrorCode::ResultBudgetExceeded);
    Ok(())
}

#[test]
fn validation_fails_closed_for_overflow_types_fences_manifest_and_receipts() -> Result<()> {
    let fixture = Fixture::new()?;
    let cpu = fixture.backend()?;
    let valid = fixture.request(
        9,
        ResidentRowProgram {
            instructions: vec![instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::IntegerConstant(1),
            )],
        },
        Vec::new(),
        0,
        usize::MAX,
        ROWS,
        vec![0],
    )?;
    valid.validate()?;
    let duplicate = ResidentRowProgramManifest::build(
        &valid.program,
        &valid.sort_keys,
        valid.offset,
        valid.limit,
        valid.max_output_rows,
        &valid.final_registers,
        99,
    )?;
    assert_eq!(
        duplicate.fingerprint,
        ResidentRowProgramManifest::build(
            &valid.program,
            &valid.sort_keys,
            valid.offset,
            valid.limit,
            valid.max_output_rows,
            &valid.final_registers,
            99,
        )?
        .fingerprint
    );

    let mut forward = valid.clone();
    forward.program.instructions[0].operation = ResidentRowOperation::NumericNegate { operand: 0 };
    assert_eq!(forward.validate().unwrap_err().code, ErrorCode::QueryType);
    let mut mismatch = valid.clone();
    mismatch.program.instructions[0].output_type = ResidentRowValueType::Float;
    assert_eq!(mismatch.validate().unwrap_err().code, ErrorCode::QueryType);

    let overflow_program = ResidentRowProgram {
        instructions: vec![
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::IntegerConstant(i64::MAX),
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::IntegerConstant(1),
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::NumericAdd { left: 0, right: 1 },
            ),
        ],
    };
    let overflow = fixture.request(
        10,
        overflow_program,
        Vec::new(),
        0,
        usize::MAX,
        ROWS,
        vec![2],
    )?;
    assert_eq!(
        cpu.execute_row_program(&overflow, &CancellationToken::new())
            .unwrap_err()
            .code,
        ErrorCode::QueryType
    );

    for stale in 0..3 {
        let mut request = valid.clone();
        match stale {
            0 => request.expected_bookmark.index += 1,
            1 => request.expected_graph_revision += 1,
            2 => request.expected_layout_version += 1,
            _ => unreachable!(),
        }
        assert_eq!(
            cpu.execute_row_program(&request, &CancellationToken::new())
                .unwrap_err()
                .code,
            ErrorCode::GpuAdmissionFailure
        );
    }

    let mut bad_manifest = valid.clone();
    bad_manifest.manifest.fingerprint.0[0] ^= 1;
    assert_eq!(
        bad_manifest.validate().unwrap_err().code,
        ErrorCode::GpuAdmissionFailure
    );
    let mut duplicate_obligation = valid.clone();
    duplicate_obligation.manifest.sort_obligation.id =
        duplicate_obligation.manifest.instruction_obligations[0].id;
    assert_eq!(
        duplicate_obligation.validate().unwrap_err().code,
        ErrorCode::GpuAdmissionFailure
    );

    let raw = cpu.execute_row_program(&valid, &CancellationToken::new())?;
    let mut parts = raw.clone().into_untrusted_parts();
    parts.receipts.pop();
    assert_eq!(
        ResidentRowProgramResult::from_untrusted_parts(parts)
            .validate(&valid, BackendKind::Cpu)
            .unwrap_err()
            .code,
        ErrorCode::CorruptStorage
    );
    let mut parts = raw.clone().into_untrusted_parts();
    parts.receipts[0].completion = ResidentDeviceCompletion::Metal;
    assert_eq!(
        ResidentRowProgramResult::from_untrusted_parts(parts)
            .validate(&valid, BackendKind::Cpu)
            .unwrap_err()
            .code,
        ErrorCode::CorruptStorage
    );
    let mut parts = raw.clone().into_untrusted_parts();
    parts.manifest_fingerprint = ResidentRowManifestFingerprint([0; 32]);
    assert_eq!(
        ResidentRowProgramResult::from_untrusted_parts(parts)
            .validate(&valid, BackendKind::Cpu)
            .unwrap_err()
            .code,
        ErrorCode::CorruptStorage
    );
    let mut parts = raw.into_untrusted_parts();
    parts.projected_columns[0].column = ResidentRowColumn::Boolean {
        values: vec![0; ROWS],
        validity: vec![1; ROWS],
    };
    assert_eq!(
        ResidentRowProgramResult::from_untrusted_parts(parts)
            .validate(&valid, BackendKind::Cpu)
            .unwrap_err()
            .code,
        ErrorCode::CorruptStorage
    );
    Ok(())
}

#[test]
fn property_shape_mismatch_is_an_error_while_absent_properties_are_null() -> Result<()> {
    let fixture = Fixture::new()?;
    let cpu = fixture.backend()?;
    for (seed, property) in [(11, fixture.num), (12, fixture.mixed)] {
        let request = fixture.request(
            seed,
            ResidentRowProgram {
                instructions: vec![instruction(
                    if property == fixture.num {
                        ResidentRowValueType::Float
                    } else {
                        ResidentRowValueType::Integer
                    },
                    if property == fixture.num {
                        ResidentRowOperation::LoadFloatProperty {
                            binding: node(ResidentNodeBinding::Start),
                            property,
                        }
                    } else {
                        ResidentRowOperation::LoadIntegerProperty {
                            binding: node(ResidentNodeBinding::Start),
                            property,
                        }
                    },
                )],
            },
            Vec::new(),
            0,
            usize::MAX,
            ROWS,
            vec![0],
        )?;
        assert_eq!(
            cpu.execute_row_program(&request, &CancellationToken::new())
                .unwrap_err()
                .code,
            ErrorCode::QueryType
        );
    }

    let absent = fixture.request(
        13,
        ResidentRowProgram {
            instructions: vec![instruction(
                ResidentRowValueType::Boolean,
                ResidentRowOperation::LoadBooleanProperty {
                    binding: node(ResidentNodeBinding::Start),
                    property: fixture.absent,
                },
            )],
        },
        Vec::new(),
        0,
        usize::MAX,
        ROWS,
        vec![0],
    )?;
    let result = execute(&cpu, &absent)?;
    assert_graph_alignment(&result, &[0, 1, 2, 3, 4, 5]);
    let (values, validity) = boolean_column(&result.projected_columns()[0].column);
    assert_eq!(values, &[0; ROWS]);
    assert_eq!(validity, &[0; ROWS]);
    Ok(())
}

#[test]
fn relationship_property_load_and_sort_preserve_every_path_column() -> Result<()> {
    let fixture = Fixture::new()?;
    let cpu = fixture.backend()?;
    let request = fixture.request(
        14,
        ResidentRowProgram {
            instructions: vec![instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::LoadIntegerProperty {
                    binding: ResidentEntityBinding::Relationship(0),
                    property: fixture.num,
                },
            )],
        },
        vec![ResidentRowSortKey {
            register: 0,
            descending: true,
            nulls_first: false,
        }],
        0,
        usize::MAX,
        ROWS,
        vec![0],
    )?;
    let result = execute(&cpu, &request)?;
    assert_graph_alignment(&result, &[5, 4, 3, 2, 1, 0]);
    let (values, validity) = integer_column(&result.projected_columns()[0].column);
    assert_eq!(values, &[5, 4, 3, 2, 1, 0]);
    assert_eq!(validity, &[1; ROWS]);
    Ok(())
}

#[test]
fn wrapped_pipeline_cannot_own_sort_pagination_or_projection() -> Result<()> {
    let fixture = Fixture::new()?;
    let program = ResidentRowProgram {
        instructions: vec![instruction(
            ResidentRowValueType::Integer,
            ResidentRowOperation::IntegerConstant(1),
        )],
    };
    let base = fixture.request(15, program, Vec::new(), 0, usize::MAX, ROWS, vec![0])?;
    for boundary in 0..3 {
        let mut request = base.clone();
        match boundary {
            0 => request.input.offset = 1,
            1 => request.input.limit = 1,
            2 => request
                .input
                .orders
                .push(irongraph::gpu::ResidentNodeOrder {
                    binding: ResidentNodeBinding::Start,
                    property: fixture.num,
                    descending: false,
                    nulls_first: false,
                }),
            _ => unreachable!(),
        }
        assert_eq!(
            request.validate().unwrap_err().code,
            ErrorCode::GpuAdmissionFailure
        );
    }
    Ok(())
}
