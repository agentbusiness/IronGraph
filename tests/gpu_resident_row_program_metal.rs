// Test-only target. The crate denies `expect`, `unwrap`, and `panic` because a panic in production
// is a whole-node abort under `panic = "abort"`. In a test the opposite is true: a failed
// expectation is how the test reports, and clippy's in-test allowance does not reach the helper
// functions that fixtures are built from. Scoping the allowance here keeps the production gate
// enforceable instead of switched off globally.
#![allow(clippy::expect_used, clippy::panic, clippy::unwrap_used)]

use std::sync::{Mutex, MutexGuard, OnceLock};

use irongraph::{
    Bookmark, EdgeId, ErrorCode, Layer, NodeId, ProjectId, Result, ScalarValue,
    gpu::{
        BackendKind, ExecutionBackend, MetalBackend, ResidentDeviceCompletion, ResidentDirection,
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
const MEMORY_LIMIT_BYTES: usize = 256 * 1024 * 1024;
const RESERVED_BYTES: usize = 8 * 1024 * 1024;
const ROWS: usize = 6;

fn metal_test_guard() -> MutexGuard<'static, ()> {
    static GUARD: OnceLock<Mutex<()>> = OnceLock::new();
    GUARD
        .get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

struct Fixture {
    graph: GraphStore,
    bookmark: Bookmark,
    start_label: LabelId,
    end_label: LabelId,
    relationship_type: RelationshipTypeId,
    flag: PropertyId,
    gate: PropertyId,
    num: PropertyId,
    num2: PropertyId,
    ratio: PropertyId,
    float_order: PropertyId,
}

impl Fixture {
    fn new() -> Result<Self> {
        let mut graph = GraphStore::default();
        let start_label = graph.catalog_mut().intern_label("Start")?;
        let end_label = graph.catalog_mut().intern_label("End")?;
        let relationship_type = graph.catalog_mut().intern_relationship_type("NEXT")?;
        let flag = graph.catalog_mut().intern_property("flag")?;
        let gate = graph.catalog_mut().intern_property("gate")?;
        let num = graph.catalog_mut().intern_property("num")?;
        let num2 = graph.catalog_mut().intern_property("num2")?;
        let ratio = graph.catalog_mut().intern_property("ratio")?;
        let float_order = graph.catalog_mut().intern_property("float_order")?;
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
                labels: vec![end_label],
                properties: Vec::new(),
            })?;
        }
        for row in 0..ROWS {
            graph.insert_edge(EdgeInput {
                id: EdgeId(300 + row as u64),
                source: NodeId(100 + row as u64),
                target: NodeId(200 + row as u64),
                relationship_type,
                layer: Layer::Observed,
                revision: 40 + row as u64,
                properties: vec![(num, ScalarValue::Integer(row as i64))],
            })?;
        }
        let bookmark = Bookmark {
            term: 11,
            index: graph.revision(),
        };
        Ok(Self {
            graph,
            bookmark,
            start_label,
            end_label,
            relationship_type,
            flag,
            gate,
            num,
            num2,
            ratio,
            float_order,
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

    fn backend(&self) -> Result<MetalBackend> {
        let mut metal = MetalBackend::new(0, MEMORY_LIMIT_BYTES, RESERVED_BYTES)?;
        metal.admit_project(self.image()?)?;
        Ok(metal)
    }

    fn input(&self, label: LabelId) -> ResidentNodePipelineRequest {
        ResidentNodePipelineRequest {
            project: PROJECT,
            labels: vec![label],
            layers: LayerMask::OBSERVED,
            initial_optional: false,
            expansion: Some(ResidentExpansion {
                direction: ResidentDirection::Outgoing,
                relationship_types: vec![self.relationship_type],
                end_labels: vec![self.end_label],
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
        input_label: LabelId,
        program: ResidentRowProgram,
        sort_keys: Vec<ResidentRowSortKey>,
        offset: usize,
        limit: usize,
        max_output_rows: usize,
        final_registers: Vec<u16>,
    ) -> Result<ResidentRowProgramRequest> {
        let first_obligation = seed
            .checked_mul(1_000)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| irongraph::Error::internal("test obligation ID overflow"))?;
        let manifest = ResidentRowProgramManifest::build(
            &program,
            &sort_keys,
            offset,
            limit,
            max_output_rows,
            &final_registers,
            first_obligation,
        )?;
        Ok(ResidentRowProgramRequest {
            project: PROJECT,
            expected_bookmark: self.bookmark,
            expected_graph_revision: self.graph.revision(),
            expected_layout_version: self.graph.layout_version(),
            execution: ResidentExecutionId {
                high: 0x4d45_5441_4c52_4f57,
                low: seed,
            },
            input: self.input(input_label),
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

fn start() -> ResidentEntityBinding {
    ResidentEntityBinding::Node(ResidentNodeBinding::Start)
}

fn execute(
    metal: &MetalBackend,
    request: &ResidentRowProgramRequest,
) -> Result<irongraph::gpu::ValidatedResidentRowProgramResult> {
    metal
        .execute_row_program(request, &CancellationToken::new())?
        .validate(request, BackendKind::Metal)
}

fn scalar_placeholder_input(rows: usize) -> ResidentNodePipelineRequest {
    ResidentNodePipelineRequest {
        project: PROJECT,
        labels: Vec::new(),
        layers: LayerMask::OBSERVED,
        initial_optional: false,
        expansion: None,
        continuations: Vec::new(),
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
        max_output_rows: rows,
    }
}

fn assert_alignment(result: &irongraph::gpu::ValidatedResidentRowProgramResult, positions: &[u64]) {
    assert_eq!(result.source_positions(), positions);
    let graph_positions = positions
        .iter()
        .map(|position| u32::try_from(*position).expect("fixture position exceeds u32"))
        .collect::<Vec<_>>();
    assert_eq!(result.rows().start_rows, graph_positions);
    assert_eq!(result.rows().edge_rows, graph_positions);
    assert_eq!(
        result.rows().end_rows,
        graph_positions
            .iter()
            .map(|position| position + ROWS as u32)
            .collect::<Vec<_>>()
    );
}

#[cfg(all(feature = "accelerator", target_os = "macos"))]
#[test]
#[ignore = "real-Metal no-ceiling gate: allocates and validates more than one million typed rows"]
fn real_metal_typed_rows_cross_the_former_fixed_count_with_u64_lineage() -> Result<()> {
    const ROWS_ABOVE_FORMER_BOUND: usize = (1 << 20) + 1;
    const MEMORY_LIMIT: usize = 1024 * 1024 * 1024;
    let _guard = metal_test_guard();
    let fixture = Fixture::new()?;
    let mut metal = MetalBackend::new(0, MEMORY_LIMIT, RESERVED_BYTES)?;
    metal.admit_project(fixture.image()?)?;

    let column = ResidentRowColumn::Integer {
        values: (0..ROWS_ABOVE_FORMER_BOUND)
            .map(|row| i64::try_from(row).expect("test row fits i64"))
            .collect(),
        validity: vec![1; ROWS_ABOVE_FORMER_BOUND],
    };
    let program = ResidentRowProgram {
        instructions: vec![instruction(
            ResidentRowValueType::Integer,
            ResidentRowOperation::InputColumn(column),
        )],
    };
    let final_registers = vec![0];
    let manifest = ResidentRowProgramManifest::build(
        &program,
        &[],
        0,
        usize::MAX,
        ROWS_ABOVE_FORMER_BOUND,
        &final_registers,
        0x4e4f_4345_494c_0001,
    )?;
    let request = ResidentRowProgramRequest {
        project: PROJECT,
        expected_bookmark: fixture.bookmark,
        expected_graph_revision: fixture.graph.revision(),
        expected_layout_version: fixture.graph.layout_version(),
        execution: ResidentExecutionId {
            high: 0x4e4f_4345_494c_494e,
            low: 1,
        },
        input: scalar_placeholder_input(ROWS_ABOVE_FORMER_BOUND),
        program,
        manifest,
        sort_keys: Vec::new(),
        offset: 0,
        limit: usize::MAX,
        max_output_rows: ROWS_ABOVE_FORMER_BOUND,
        final_registers,
    };

    let result = execute(&metal, &request)?;
    assert_eq!(result.source_positions().len(), ROWS_ABOVE_FORMER_BOUND);
    assert_eq!(result.source_positions().first(), Some(&0));
    assert_eq!(
        result.source_positions().last(),
        Some(&u64::try_from(ROWS_ABOVE_FORMER_BOUND - 1).expect("test row fits u64"))
    );
    let ResidentRowColumn::Integer { values, validity } = &result.projected_columns()[0].column
    else {
        return Err(irongraph::Error::internal(
            "Metal no-ceiling gate returned the wrong typed column",
        ));
    };
    assert_eq!(values.len(), ROWS_ABOVE_FORMER_BOUND);
    assert_eq!(values.first(), Some(&0));
    assert_eq!(
        values.last(),
        Some(&i64::try_from(ROWS_ABOVE_FORMER_BOUND - 1).expect("test row fits i64"))
    );
    assert!(validity.iter().all(|valid| *valid == 1));
    Ok(())
}

#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_null_logic_stable_ties_alignment_and_device_receipts() -> Result<()> {
    let _guard = metal_test_guard();
    let fixture = Fixture::new()?;
    let metal = fixture.backend()?;
    let program = ResidentRowProgram {
        instructions: vec![
            instruction(
                ResidentRowValueType::Boolean,
                ResidentRowOperation::LoadBooleanProperty {
                    binding: start(),
                    property: fixture.flag,
                },
            ),
            instruction(
                ResidentRowValueType::Boolean,
                ResidentRowOperation::LoadBooleanProperty {
                    binding: start(),
                    property: fixture.gate,
                },
            ),
            instruction(
                ResidentRowValueType::Boolean,
                ResidentRowOperation::BooleanAnd { left: 0, right: 1 },
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::LoadIntegerProperty {
                    binding: start(),
                    property: fixture.num,
                },
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::IntegerConstant(2),
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::NumericMultiply { left: 3, right: 4 },
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::LoadIntegerProperty {
                    binding: start(),
                    property: fixture.num2,
                },
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::NumericAdd { left: 6, right: 5 },
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::NumericNegate { operand: 7 },
            ),
        ],
    };
    let request = fixture.request(
        1,
        fixture.start_label,
        program,
        vec![ResidentRowSortKey {
            register: 8,
            descending: false,
            nulls_first: false,
        }],
        0,
        usize::MAX,
        ROWS,
        vec![2, 8],
    )?;
    let available_before = metal.available_query_scratch_bytes();
    let result = execute(&metal, &request)?;
    assert_eq!(metal.available_query_scratch_bytes(), available_before);
    assert_alignment(&result, &[0, 1, 2, 3, 5, 4]);
    let ResidentRowColumn::Boolean { values, validity } = &result.projected_columns()[0].column
    else {
        panic!("expected Boolean projection");
    };
    assert_eq!(values, &[1, 0, 0, 0, 0, 0]);
    assert_eq!(validity, &[1, 1, 0, 1, 1, 0]);
    let ResidentRowColumn::Integer { values, validity } = &result.projected_columns()[1].column
    else {
        panic!("expected integer projection");
    };
    assert_eq!(values, &[-12, -11, -8, -8, -8, 0]);
    assert_eq!(validity, &[1, 1, 1, 1, 1, 0]);
    assert!(result.receipts().iter().all(|receipt| {
        receipt.completion == ResidentDeviceCompletion::Metal
            && receipt.execution == request.execution
    }));
    assert_eq!(
        result.receipts().last().unwrap().obligation.kind,
        ResidentObligationKind::Sort
    );
    assert_eq!(result.scratch_bytes(), request.scratch_bytes(ROWS)?);
    Ok(())
}

#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_mixed_numeric_binary64_float_order_overflow_and_empty_rows() -> Result<()> {
    let _guard = metal_test_guard();
    let fixture = Fixture::new()?;
    let metal = fixture.backend()?;
    let program = ResidentRowProgram {
        instructions: vec![
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::LoadIntegerProperty {
                    binding: start(),
                    property: fixture.num,
                },
            ),
            instruction(
                ResidentRowValueType::Float,
                ResidentRowOperation::LoadFloatProperty {
                    binding: start(),
                    property: fixture.ratio,
                },
            ),
            instruction(
                ResidentRowValueType::Float,
                ResidentRowOperation::NumericAdd { left: 0, right: 1 },
            ),
            instruction(
                ResidentRowValueType::Float,
                ResidentRowOperation::FloatConstant((-1.01_f64).to_bits()),
            ),
            instruction(
                ResidentRowValueType::Float,
                ResidentRowOperation::NumericMultiply { left: 2, right: 3 },
            ),
            instruction(
                ResidentRowValueType::Float,
                ResidentRowOperation::NumericNegate { operand: 4 },
            ),
        ],
    };
    let request = fixture.request(
        2,
        fixture.start_label,
        program,
        vec![ResidentRowSortKey {
            register: 2,
            descending: false,
            nulls_first: false,
        }],
        0,
        usize::MAX,
        ROWS,
        vec![2, 4, 5],
    )?;
    let result = execute(&metal, &request)?;
    assert_alignment(&result, &[0, 1, 5, 2, 3, 4]);
    let ResidentRowColumn::Float { bits, validity } = &result.projected_columns()[0].column else {
        panic!("expected float projection");
    };
    assert_eq!(
        bits.iter()
            .map(|bits| f64::from_bits(*bits))
            .collect::<Vec<_>>(),
        vec![1.5, 2.0, 4.0, 4.25, 4.25, 0.0]
    );
    assert_eq!(validity, &[1, 1, 1, 1, 1, 0]);
    let ResidentRowColumn::Float { bits, validity } = &result.projected_columns()[1].column else {
        panic!("expected float product projection");
    };
    let expected = [1.5, 2.0, 4.0, 4.25, 4.25]
        .into_iter()
        .map(|value| (value * -1.01_f64).to_bits())
        .collect::<Vec<_>>();
    assert_eq!(&bits[..5], expected);
    assert_eq!(validity, &[1, 1, 1, 1, 1, 0]);
    let ResidentRowColumn::Float { bits: negated, .. } = &result.projected_columns()[2].column
    else {
        panic!("expected negated float projection");
    };
    assert_eq!(
        &negated[..5],
        &bits[..5]
            .iter()
            .map(|bits| bits ^ (1_u64 << 63))
            .collect::<Vec<_>>()
    );

    let float_order = fixture.request(
        3,
        fixture.start_label,
        ResidentRowProgram {
            instructions: vec![instruction(
                ResidentRowValueType::Float,
                ResidentRowOperation::LoadFloatProperty {
                    binding: start(),
                    property: fixture.float_order,
                },
            )],
        },
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
    assert_alignment(&execute(&metal, &float_order)?, &[2, 3, 4, 1, 0, 5]);

    let overflow = fixture.request(
        4,
        fixture.start_label,
        ResidentRowProgram {
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
        },
        Vec::new(),
        0,
        usize::MAX,
        ROWS,
        vec![2],
    )?;
    assert_eq!(
        metal
            .execute_row_program(&overflow, &CancellationToken::new())
            .unwrap_err()
            .code,
        ErrorCode::QueryType
    );

    let mut empty = fixture.request(
        5,
        fixture.start_label,
        ResidentRowProgram {
            instructions: vec![instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::IntegerConstant(7),
            )],
        },
        Vec::new(),
        0,
        usize::MAX,
        ROWS,
        vec![0],
    )?;
    // Both labels exist in the resident image, but no node has both. This exercises a genuine
    // empty device scan without relying on publication of unused catalog-only labels.
    empty.input.labels = vec![fixture.start_label, fixture.end_label];
    empty.validate()?;
    let empty = execute(&metal, &empty)?;
    assert_eq!(empty.input_cardinality(), 0);
    assert!(empty.rows().start_rows.is_empty());
    assert!(empty.projected_columns()[0].column.is_empty());
    assert!(
        empty
            .receipts()
            .iter()
            .all(|receipt| receipt.completion == ResidentDeviceCompletion::Metal)
    );
    Ok(())
}

#[test]
#[ignore = "requires an available physical Metal device"]
fn real_metal_fences_cancellation_corruption_and_rounded_scratch_fail_closed() -> Result<()> {
    let _guard = metal_test_guard();
    let fixture = Fixture::new()?;
    let metal = fixture.backend()?;
    let request = fixture.request(
        6,
        fixture.start_label,
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
        1,
        3,
        3,
        vec![0],
    )?;
    let raw = metal.execute_row_program(&request, &CancellationToken::new())?;
    let validated = raw.clone().validate(&request, BackendKind::Metal)?;
    assert_alignment(&validated, &[4, 3, 2]);
    assert_eq!(validated.scratch_bytes(), request.scratch_bytes(ROWS)?);

    for stale in 0..3 {
        let mut request = request.clone();
        match stale {
            0 => request.expected_bookmark.index += 1,
            1 => request.expected_graph_revision += 1,
            2 => request.expected_layout_version += 1,
            _ => unreachable!(),
        }
        assert_eq!(
            metal
                .execute_row_program(&request, &CancellationToken::new())
                .unwrap_err()
                .code,
            ErrorCode::GpuAdmissionFailure
        );
    }
    let cancellation = CancellationToken::new();
    cancellation.cancel();
    assert_eq!(
        metal
            .execute_row_program(&request, &cancellation)
            .unwrap_err()
            .code,
        ErrorCode::Cancelled
    );

    let mut parts = raw.clone().into_untrusted_parts();
    parts.receipts.pop();
    assert_eq!(
        ResidentRowProgramResult::from_untrusted_parts(parts)
            .validate(&request, BackendKind::Metal)
            .unwrap_err()
            .code,
        ErrorCode::CorruptStorage
    );
    let mut parts = raw.clone().into_untrusted_parts();
    parts.receipts[0].completion = ResidentDeviceCompletion::CpuReference;
    assert_eq!(
        ResidentRowProgramResult::from_untrusted_parts(parts)
            .validate(&request, BackendKind::Metal)
            .unwrap_err()
            .code,
        ErrorCode::CorruptStorage
    );
    let mut parts = raw.into_untrusted_parts();
    parts.manifest_fingerprint = ResidentRowManifestFingerprint([0; 32]);
    assert_eq!(
        ResidentRowProgramResult::from_untrusted_parts(parts)
            .validate(&request, BackendKind::Metal)
            .unwrap_err()
            .code,
        ErrorCode::CorruptStorage
    );

    let logical = request.scratch_bytes(ROWS)?;
    let available = metal.available_query_scratch_bytes();
    assert!(available > logical);
    let blocker = metal.reserve_query_scratch(available - logical)?;
    assert_eq!(
        metal
            .execute_row_program(&request, &CancellationToken::new())
            .unwrap_err()
            .code,
        ErrorCode::GpuAdmissionFailure
    );
    drop(blocker);
    execute(&metal, &request)?;
    Ok(())
}
