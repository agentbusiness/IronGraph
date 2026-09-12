// Test-only module. Clippy's `allow-expect-in-tests` covers `#[test]` bodies but not the helper
// functions those tests call, and a failed expectation in a fixture is the intended way for a test
// to fail. The production denial of `expect` is unaffected.
#![allow(clippy::expect_used)]

use super::{
    MetalResidentRowScratchBreakdown, MetalRowSortStrategy,
    metal_resident_row_program_scratch_breakdown, metal_resident_row_program_scratch_bytes,
    metal_row_shared_buffer_bytes, metal_row_sort_strategy, metal_row_sort_workspace_bytes,
    metal_row_string_arena_bytes, metal_row_string_slots,
};
use crate::{
    Bookmark, ErrorCode, ProjectId, Result,
    execution::{
        DeviceMemoryGovernor, ResidentDirection, ResidentEntityBinding, ResidentExecutionId,
        ResidentExecutionReceipt, ResidentExpansion, ResidentNodeBinding,
        ResidentNodePipelineRequest, ResidentRowColumn, ResidentRowInstruction,
        ResidentRowOperation, ResidentRowProgram, ResidentRowProgramManifest,
        ResidentRowProgramRequest, ResidentRowProjectedColumn, ResidentRowSortKey,
        ResidentRowValueType,
    },
    graph::LayerMask,
    types::{LabelId, PropertyId, RelationshipTypeId},
};
use std::mem::size_of;

const INPUT_ROWS: usize = 2_049;
const TEMPORAL_OUTPUT_ROWS: usize = 137;
const TIMEZONE_WIDTH: usize = 31;
const PROJECT: ProjectId = ProjectId(uuid::Uuid::nil());

fn instruction(
    output_type: ResidentRowValueType,
    operation: ResidentRowOperation,
) -> ResidentRowInstruction {
    ResidentRowInstruction {
        output_type,
        operation,
    }
}

fn input_for_rows(with_expansion: bool, max_output_rows: usize) -> ResidentNodePipelineRequest {
    ResidentNodePipelineRequest {
        project: PROJECT,
        labels: vec![LabelId(1)],
        layers: LayerMask::ALL,
        initial_optional: false,
        expansion: with_expansion.then(|| ResidentExpansion {
            direction: ResidentDirection::Outgoing,
            relationship_types: vec![RelationshipTypeId(1)],
            end_labels: vec![LabelId(2)],
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
        max_output_rows,
    }
}

fn request(
    seed: u64,
    program: ResidentRowProgram,
    sort_keys: Vec<ResidentRowSortKey>,
    offset: usize,
    limit: usize,
    final_registers: Vec<u16>,
    with_expansion: bool,
) -> Result<ResidentRowProgramRequest> {
    request_for_rows(
        seed,
        program,
        sort_keys,
        offset,
        limit,
        final_registers,
        with_expansion,
        INPUT_ROWS,
    )
}

#[allow(clippy::too_many_arguments)]
fn request_for_rows(
    seed: u64,
    program: ResidentRowProgram,
    sort_keys: Vec<ResidentRowSortKey>,
    offset: usize,
    limit: usize,
    final_registers: Vec<u16>,
    with_expansion: bool,
    input_rows: usize,
) -> Result<ResidentRowProgramRequest> {
    let manifest = ResidentRowProgramManifest::build(
        &program,
        &sort_keys,
        offset,
        limit,
        input_rows,
        &final_registers,
        seed.checked_mul(100)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| crate::Error::internal("row scratch test obligation ID overflow"))?,
    )?;
    Ok(ResidentRowProgramRequest {
        project: PROJECT,
        expected_bookmark: Bookmark { term: 1, index: 1 },
        expected_graph_revision: 1,
        expected_layout_version: 1,
        execution: ResidentExecutionId {
            high: 0x5343_5241_5443_4854,
            low: seed,
        },
        input: input_for_rows(with_expansion, input_rows),
        program,
        manifest,
        sort_keys,
        offset,
        limit,
        max_output_rows: input_rows,
        final_registers,
    })
}

fn key(register: u16) -> ResidentRowSortKey {
    ResidentRowSortKey {
        register,
        descending: false,
        nulls_first: false,
    }
}

fn assert_exact_admission_boundary(
    request: &ResidentRowProgramRequest,
) -> Result<MetalResidentRowScratchBreakdown> {
    assert_exact_admission_boundary_for_rows(request, INPUT_ROWS)
}

fn assert_exact_admission_boundary_for_rows(
    request: &ResidentRowProgramRequest,
    input_rows: usize,
) -> Result<MetalResidentRowScratchBreakdown> {
    let output_rows = request.output_cardinality(input_rows)?;
    let strategy = metal_row_sort_strategy(request, input_rows, output_rows)?;
    let breakdown = metal_resident_row_program_scratch_breakdown(request, input_rows, strategy)?;
    let required = metal_resident_row_program_scratch_bytes(request, input_rows)?;
    assert_eq!(required, breakdown.required_bytes()?);
    assert!(required > 0);

    let one_byte_short = DeviceMemoryGovernor::new(required - 1, 0);
    let error = one_byte_short
        .reserve_scratch(required)
        .expect_err("one byte below the typed-row requirement must fail admission");
    assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);

    let exact = DeviceMemoryGovernor::new(required, 0);
    let reservation = exact
        .reserve_scratch(required)
        .expect("the exact typed-row requirement must pass admission");
    assert_eq!(reservation.bytes(), required);
    Ok(breakdown)
}

fn expected_metal_buffer_bytes(logical_bytes: usize) -> Result<usize> {
    if logical_bytes == 0 {
        return Ok(0);
    }
    logical_bytes
        .checked_next_power_of_two()
        .map(|bytes| bytes.max(4_096))
        .ok_or_else(|| {
            crate::Error::new(
                ErrorCode::ResultBudgetExceeded,
                "row scratch test Metal allocation overflow",
            )
        })
}

fn expected_product(left: usize, right: usize) -> Result<usize> {
    left.checked_mul(right).ok_or_else(|| {
        crate::Error::new(
            ErrorCode::ResultBudgetExceeded,
            "row scratch test allocation shape overflow",
        )
    })
}

fn expected_sum(values: impl IntoIterator<Item = usize>) -> Result<usize> {
    values.into_iter().try_fold(0_usize, |total, value| {
        total.checked_add(value).ok_or_else(|| {
            crate::Error::new(
                ErrorCode::ResultBudgetExceeded,
                "row scratch test allocation sum overflow",
            )
        })
    })
}

fn repeated_string_column(rows: usize, value: &str) -> Result<ResidentRowColumn> {
    let total_bytes = expected_product(rows, value.len())?;
    let mut offsets = Vec::with_capacity(rows.saturating_add(1));
    let mut bytes = Vec::with_capacity(total_bytes);
    offsets.push(0);
    for row in 0..rows {
        bytes.extend_from_slice(value.as_bytes());
        offsets.push(
            u32::try_from(expected_product(row.saturating_add(1), value.len())?).map_err(|_| {
                crate::Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "row scratch test STRING offsets exceed u32",
                )
            })?,
        );
    }
    Ok(ResidentRowColumn::String {
        offsets,
        bytes,
        validity: vec![1; rows],
    })
}

fn scalar_request(mut request: ResidentRowProgramRequest) -> Result<ResidentRowProgramRequest> {
    request.input.labels.clear();
    request.validate()?;
    Ok(request)
}

fn expected_fixed_evaluated_device(rows: usize, instruction_count: usize) -> Result<usize> {
    let words = expected_product(rows, instruction_count)?
        .checked_mul(2)
        .and_then(|words| words.checked_add(1))
        // One device-authored error/status word per input row follows the value/validity frame.
        .and_then(|words| words.checked_add(rows))
        .ok_or_else(|| {
            crate::Error::new(
                ErrorCode::ResultBudgetExceeded,
                "row scratch test evaluated register shape overflow",
            )
        })?;
    expected_metal_buffer_bytes(expected_product(words, size_of::<i64>())?)
}

fn expected_evaluated_device(
    rows: usize,
    instruction_count: usize,
    arena_bytes: usize,
) -> Result<usize> {
    let fixed_words = expected_product(rows, instruction_count)?
        .checked_mul(2)
        .and_then(|words| words.checked_add(1))
        .and_then(|words| words.checked_add(rows))
        .ok_or_else(|| {
            crate::Error::new(
                ErrorCode::ResultBudgetExceeded,
                "row scratch test evaluated register shape overflow",
            )
        })?;
    let words = fixed_words
        .checked_add(arena_bytes.div_ceil(size_of::<i64>()))
        .ok_or_else(|| {
            crate::Error::new(
                ErrorCode::ResultBudgetExceeded,
                "row scratch test evaluated STRING arena overflow",
            )
        })?;
    expected_metal_buffer_bytes(expected_product(words, size_of::<i64>())?)
}

fn expected_temporal_evaluated_device(
    rows: usize,
    instruction_count: usize,
    temporal_sidecar_lanes: usize,
    variable_arena_bytes: usize,
) -> Result<usize> {
    let fixed_words = expected_product(rows, instruction_count)?
        .checked_mul(2)
        .and_then(|words| words.checked_add(1))
        .and_then(|words| words.checked_add(rows))
        .ok_or_else(|| {
            crate::Error::new(
                ErrorCode::ResultBudgetExceeded,
                "row scratch test temporal register frame overflow",
            )
        })?;
    let sidecar_words = expected_product(rows, temporal_sidecar_lanes)?;
    let words = expected_sum([
        fixed_words,
        sidecar_words,
        variable_arena_bytes.div_ceil(size_of::<i64>()),
    ])?;
    expected_metal_buffer_bytes(expected_product(words, size_of::<i64>())?)
}

fn expected_packet_allocation(
    request: &ResidentRowProgramRequest,
    output_rows: usize,
    projection_words: usize,
) -> Result<usize> {
    let words = expected_sum([
        super::METAL_ROW_PACKET_HEADER_WORDS,
        output_rows,
        projection_words,
        expected_product(
            request.obligations().count(),
            super::METAL_ROW_PACKET_RECEIPT_WORDS,
        )?,
    ])?;
    expected_metal_buffer_bytes(expected_product(words, size_of::<i64>())?)
}

fn expected_decoded_host(
    request: &ResidentRowProgramRequest,
    output_rows: usize,
    projection_payloads: usize,
    projection_temporary: usize,
) -> Result<usize> {
    expected_sum([
        expected_metal_buffer_bytes(expected_product(output_rows, size_of::<u32>())?)?,
        expected_metal_buffer_bytes(expected_product(
            request.final_registers.len(),
            size_of::<ResidentRowProjectedColumn>(),
        )?)?,
        projection_payloads,
        projection_temporary,
        expected_metal_buffer_bytes(expected_product(
            request.obligations().count(),
            size_of::<ResidentExecutionReceipt>(),
        )?)?,
    ])
}

fn assert_expected_required_boundary(
    request: &ResidentRowProgramRequest,
    input_rows: usize,
    expected: MetalResidentRowScratchBreakdown,
) -> Result<()> {
    let required = expected.required_bytes()?;
    assert!(required > 0);

    let one_byte_short = DeviceMemoryGovernor::new(required - 1, 0);
    let error = one_byte_short
        .reserve_scratch(required)
        .expect_err("one byte below the exact physical row requirement must fail admission");
    assert_eq!(error.code, ErrorCode::GpuAdmissionFailure);

    let exact = DeviceMemoryGovernor::new(required, 0);
    let reservation = exact
        .reserve_scratch(required)
        .expect("the exact physical row requirement must pass admission");
    assert_eq!(reservation.bytes(), required);
    assert_eq!(
        metal_resident_row_program_scratch_bytes(request, input_rows)?,
        required,
        "physical row scratch admission must equal the sum of the independently rounded buffers"
    );
    Ok(())
}

fn assert_exact_fields(context: &str, fields: &[(&str, usize, usize)]) {
    let mismatches = fields
        .iter()
        .filter_map(|(name, actual, expected)| {
            (actual != expected).then(|| format!("{name}: accounted {actual}, physical {expected}"))
        })
        .collect::<Vec<_>>();
    assert!(
        mismatches.is_empty(),
        "{context} accounting mismatch:\n{}",
        mismatches.join("\n")
    );
}

#[test]
fn single_key_top_k_and_multi_key_radix_have_exact_admission_boundaries() -> Result<()> {
    let single_program = ResidentRowProgram {
        instructions: vec![instruction(
            ResidentRowValueType::Integer,
            ResidentRowOperation::IntegerConstant(7),
        )],
    };
    let single = request(1, single_program, vec![key(0)], 5, 64, vec![0], false)?;
    let single_breakdown = assert_exact_admission_boundary(&single)?;
    assert_eq!(
        metal_row_sort_strategy(&single, INPUT_ROWS, single.output_cardinality(INPUT_ROWS)?)?,
        MetalRowSortStrategy::SingleKeyTopK { window: 69 }
    );
    assert_eq!(
        single_breakdown.sort_workspace,
        metal_row_sort_workspace_bytes(
            MetalRowSortStrategy::SingleKeyTopK { window: 69 },
            INPUT_ROWS,
        )?
    );

    let multi_program = ResidentRowProgram {
        instructions: vec![
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::IntegerConstant(7),
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::IntegerConstant(8),
            ),
        ],
    };
    let multi = request(
        2,
        multi_program,
        vec![key(0), key(1)],
        5,
        64,
        vec![0, 1],
        false,
    )?;
    let multi_breakdown = assert_exact_admission_boundary(&multi)?;
    assert_eq!(
        metal_row_sort_strategy(&multi, INPUT_ROWS, multi.output_cardinality(INPUT_ROWS)?)?,
        MetalRowSortStrategy::Radix
    );
    assert_eq!(
        multi_breakdown.sort_workspace,
        metal_row_sort_workspace_bytes(MetalRowSortStrategy::Radix, INPUT_ROWS)?
    );
    assert!(
        multi_breakdown.sort_workspace > single_breakdown.sort_workspace,
        "a small multi-key window must not be admitted as top-k"
    );
    Ok(())
}

#[test]
fn multiple_property_load_buffers_and_temporaries_have_an_exact_boundary() -> Result<()> {
    let program = ResidentRowProgram {
        instructions: vec![
            instruction(
                ResidentRowValueType::Boolean,
                ResidentRowOperation::LoadBooleanProperty {
                    binding: ResidentEntityBinding::Node(ResidentNodeBinding::Start),
                    property: PropertyId(1),
                },
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::LoadIntegerProperty {
                    binding: ResidentEntityBinding::Node(ResidentNodeBinding::Start),
                    property: PropertyId(2),
                },
            ),
            instruction(
                ResidentRowValueType::Float,
                ResidentRowOperation::LoadFloatProperty {
                    binding: ResidentEntityBinding::Node(ResidentNodeBinding::Start),
                    property: PropertyId(3),
                },
            ),
        ],
    };
    let request = request(3, program, vec![key(1)], 0, 128, vec![0, 1, 2], false)?;
    let breakdown = assert_exact_admission_boundary(&request)?;
    assert!(breakdown.property_gather_device > 0);
    assert!(breakdown.property_gather_temporaries > 0);
    assert!(breakdown.property_image_device > 0);
    assert!(breakdown.property_gather_device > breakdown.property_image_device);
    Ok(())
}

#[test]
fn zero_window_uses_only_empty_positions_and_has_an_exact_boundary() -> Result<()> {
    let program = ResidentRowProgram {
        instructions: vec![instruction(
            ResidentRowValueType::Integer,
            ResidentRowOperation::IntegerConstant(7),
        )],
    };
    let request = request(4, program, vec![key(0)], 0, 0, vec![0], false)?;
    let breakdown = assert_exact_admission_boundary(&request)?;
    assert_eq!(
        metal_row_sort_strategy(
            &request,
            INPUT_ROWS,
            request.output_cardinality(INPUT_ROWS)?,
        )?,
        MetalRowSortStrategy::EmptyWindow
    );
    assert_eq!(breakdown.sort_workspace, 0);
    assert_eq!(breakdown.position_device, metal_row_shared_buffer_bytes(1)?);
    assert!(breakdown.finalize_position_device > 0);
    Ok(())
}

#[test]
fn graph_column_device_and_readback_buffers_have_an_exact_boundary() -> Result<()> {
    let program = ResidentRowProgram {
        instructions: vec![instruction(
            ResidentRowValueType::Integer,
            ResidentRowOperation::IntegerConstant(7),
        )],
    };
    let request = request(5, program, Vec::new(), 0, 128, vec![0], true)?;
    let breakdown = assert_exact_admission_boundary(&request)?;
    assert_eq!(
        metal_row_sort_strategy(
            &request,
            INPUT_ROWS,
            request.output_cardinality(INPUT_ROWS)?,
        )?,
        MetalRowSortStrategy::NoSort
    );
    assert_eq!(
        1 + request.input.expansion_steps().count() + request.input.relationship_column_count(),
        3
    );
    assert!(breakdown.graph_input_device > 0);
    assert!(breakdown.graph_output_device > 0);
    assert!(breakdown.graph_readback_staging > 0);
    assert!(breakdown.graph_host > 0);
    Ok(())
}

#[test]
fn typed_literal_input_uploads_have_an_exact_boundary() -> Result<()> {
    let validity = vec![1_u8; INPUT_ROWS];
    let program = ResidentRowProgram {
        instructions: vec![
            instruction(
                ResidentRowValueType::Boolean,
                ResidentRowOperation::InputColumn(ResidentRowColumn::Boolean {
                    values: vec![1_u8; INPUT_ROWS],
                    validity: validity.clone(),
                }),
            ),
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::InputColumn(ResidentRowColumn::Integer {
                    values: vec![7_i64; INPUT_ROWS],
                    validity: validity.clone(),
                }),
            ),
            instruction(
                ResidentRowValueType::Float,
                ResidentRowOperation::InputColumn(ResidentRowColumn::Float {
                    bits: vec![3.5_f64.to_bits(); INPUT_ROWS],
                    validity,
                }),
            ),
        ],
    };
    let mut request = request(6, program, vec![key(1)], 0, 128, vec![0, 1, 2], false)?;
    request.input.labels.clear();
    request.validate()?;

    let breakdown = assert_exact_admission_boundary(&request)?;
    assert!(!request.has_graph_input());
    assert!(breakdown.typed_input_device > 0);
    assert!(breakdown.typed_input_host > 0);
    assert_eq!(breakdown.graph_input_device, 0);
    assert_eq!(breakdown.graph_output_device, 0);
    assert_eq!(breakdown.property_gather_device, 0);
    assert_eq!(breakdown.property_gather_temporaries, 0);
    assert!(breakdown.property_image_device > 0);
    Ok(())
}

#[test]
fn direct_string_rank_path_never_allocates_a_materialized_arena() -> Result<()> {
    const VALUE: &str = "0123456789abcdefg";
    let string_program = ResidentRowProgram {
        instructions: vec![instruction(
            ResidentRowValueType::String,
            ResidentRowOperation::InputColumn(repeated_string_column(INPUT_ROWS, VALUE)?),
        )],
    };
    let string_request = scalar_request(request(
        7,
        string_program,
        vec![key(0)],
        0,
        128,
        Vec::new(),
        false,
    )?)?;
    let string_breakdown = assert_exact_admission_boundary(&string_request)?;

    let integer_program = ResidentRowProgram {
        instructions: vec![instruction(
            ResidentRowValueType::Integer,
            ResidentRowOperation::InputColumn(ResidentRowColumn::Integer {
                values: vec![7; INPUT_ROWS],
                validity: vec![1; INPUT_ROWS],
            }),
        )],
    };
    let integer_request = scalar_request(request(
        8,
        integer_program,
        vec![key(0)],
        0,
        128,
        Vec::new(),
        false,
    )?)?;
    let integer_breakdown = assert_exact_admission_boundary(&integer_request)?;

    let strategy = MetalRowSortStrategy::SingleKeyTopK { window: 128 };
    assert_eq!(
        metal_row_sort_strategy(&string_request, INPUT_ROWS, 128)?,
        strategy
    );
    assert_eq!(
        metal_row_string_arena_bytes(&string_request, INPUT_ROWS)?,
        0
    );
    assert_eq!(
        metal_row_string_slots(&string_request, INPUT_ROWS)?,
        vec![(0, 0)]
    );
    assert_eq!(
        string_breakdown.evaluated_device,
        expected_fixed_evaluated_device(INPUT_ROWS, 1)?
    );
    assert_eq!(
        string_breakdown.evaluated_device, integer_breakdown.evaluated_device,
        "a rank-only STRING key must not blanket-materialize its maximum UTF-8 width"
    );
    assert_eq!(
        string_breakdown.typed_input_device,
        integer_breakdown.typed_input_device
    );
    assert_eq!(
        string_breakdown.property_image_device,
        integer_breakdown.property_image_device
    );
    assert_eq!(
        string_breakdown.sort_workspace,
        metal_row_sort_workspace_bytes(strategy, INPUT_ROWS)?
    );
    Ok(())
}

#[test]
fn materialized_input_constant_and_concat_account_exact_uploads_and_arena() -> Result<()> {
    const INPUT: &str = "0123456789abcdefg";
    const SUFFIX: &str = "β\0";
    let input_bytes = expected_product(INPUT_ROWS, INPUT.len())?;
    let program = ResidentRowProgram {
        instructions: vec![
            instruction(
                ResidentRowValueType::String,
                ResidentRowOperation::InputColumn(repeated_string_column(INPUT_ROWS, INPUT)?),
            ),
            instruction(
                ResidentRowValueType::String,
                ResidentRowOperation::StringConstant(SUFFIX.to_owned()),
            ),
            instruction(
                ResidentRowValueType::String,
                ResidentRowOperation::StringConcat { left: 0, right: 1 },
            ),
        ],
    };
    let request = scalar_request(request(
        9,
        program,
        vec![key(2)],
        0,
        INPUT_ROWS,
        Vec::new(),
        false,
    )?)?;
    let breakdown = assert_exact_admission_boundary(&request)?;

    let concat_width = INPUT.len() + SUFFIX.len();
    let arena_width = INPUT.len() + SUFFIX.len() + concat_width;
    let arena_bytes = expected_product(INPUT_ROWS, arena_width)?;
    assert_eq!(
        metal_row_string_arena_bytes(&request, INPUT_ROWS)?,
        arena_bytes
    );
    assert_eq!(
        breakdown.evaluated_device,
        expected_evaluated_device(INPUT_ROWS, 3, arena_bytes)?,
        "the evaluated allocation must contain all three disjoint per-row STRING slots"
    );

    let fixed_bytes = expected_product(
        INPUT_ROWS
            .checked_mul(3)
            .and_then(|words| words.checked_mul(2))
            .and_then(|words| words.checked_add(1))
            .and_then(|words| words.checked_add(INPUT_ROWS))
            .ok_or_else(|| {
                crate::Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "row scratch test fixed STRING slot shape overflow",
                )
            })?,
        size_of::<i64>(),
    )?;
    let slots = metal_row_string_slots(&request, INPUT_ROWS)?;
    assert_eq!(slots.len(), 3);
    assert_eq!(
        slots[0],
        (u32::try_from(fixed_bytes).unwrap(), INPUT.len() as u32)
    );
    assert_eq!(slots[1].1, SUFFIX.len() as u32);
    assert_eq!(slots[2].1, concat_width as u32);
    assert_eq!(
        usize::try_from(slots[1].0).unwrap(),
        fixed_bytes + expected_product(INPUT_ROWS, INPUT.len())?
    );
    assert_eq!(
        usize::try_from(slots[2].0).unwrap(),
        fixed_bytes
            + expected_product(INPUT_ROWS, INPUT.len())?
            + expected_product(INPUT_ROWS, SUFFIX.len())?
    );

    let row_i64_bytes = expected_product(INPUT_ROWS, size_of::<i64>())?;
    let values_and_validity = expected_product(2, expected_metal_buffer_bytes(row_i64_bytes)?)?;
    let offsets_upload =
        expected_metal_buffer_bytes(expected_product(INPUT_ROWS + 1, size_of::<i64>())?)?;
    let input_bytes_upload =
        expected_metal_buffer_bytes(expected_product(input_bytes, size_of::<i64>())?)?;
    let literal_upload =
        expected_metal_buffer_bytes(expected_product(SUFFIX.len(), size_of::<i64>())?)?;
    let expected_uploads = expected_sum([
        values_and_validity,
        offsets_upload,
        input_bytes_upload,
        literal_upload,
    ])?;
    let input_host_peak = expected_sum([values_and_validity, offsets_upload, input_bytes_upload])?;
    let property_words = expected_sum([
        expected_product(INPUT_ROWS, 2)?,
        INPUT_ROWS + 1,
        input_bytes,
        SUFFIX.len(),
    ])?;
    let expected_property_image =
        expected_metal_buffer_bytes(expected_product(property_words, size_of::<i64>())?)?;
    assert_exact_fields(
        "materialized InputColumn/StringConstant/StringConcat",
        &[
            (
                "query-local device uploads",
                breakdown.typed_input_device,
                expected_uploads,
            ),
            (
                "host conversion peak",
                breakdown.typed_input_host,
                input_host_peak.max(literal_upload),
            ),
            (
                "concatenated property image",
                breakdown.property_image_device,
                expected_property_image,
            ),
        ],
    );
    Ok(())
}

#[test]
fn materialized_string_sort_accounts_digit_radix_and_proof_buffers() -> Result<()> {
    const LITERAL: &str = "0123456789abcdefg";
    let program = ResidentRowProgram {
        instructions: vec![
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::InputColumn(ResidentRowColumn::Integer {
                    values: vec![1; INPUT_ROWS],
                    validity: vec![1; INPUT_ROWS],
                }),
            ),
            instruction(
                ResidentRowValueType::String,
                ResidentRowOperation::StringConstant(LITERAL.to_owned()),
            ),
        ],
    };
    let request = scalar_request(request(
        10,
        program,
        vec![key(1)],
        0,
        INPUT_ROWS,
        vec![0],
        false,
    )?)?;
    let strategy = metal_row_sort_strategy(&request, INPUT_ROWS, INPUT_ROWS)?;
    assert_eq!(strategy, MetalRowSortStrategy::Radix);
    let breakdown = assert_exact_admission_boundary(&request)?;

    let digit_buffer = expected_metal_buffer_bytes(INPUT_ROWS)?;
    let expected_sort_workspace = metal_row_sort_workspace_bytes(strategy, INPUT_ROWS)?
        .checked_add(expected_product(2, digit_buffer)?)
        .ok_or_else(|| {
            crate::Error::new(
                ErrorCode::ResultBudgetExceeded,
                "row scratch test STRING sort workspace overflow",
            )
        })?;
    assert_eq!(breakdown.proof_position_device, 0);
    assert_eq!(
        breakdown.proof_graph_device,
        expected_metal_buffer_bytes(size_of::<u32>())?
    );
    assert_eq!(
        breakdown.proof_membership_device,
        expected_metal_buffer_bytes(expected_product(INPUT_ROWS, size_of::<u32>())?)?
    );
    assert_eq!(
        breakdown.proof_argument_bytes,
        expected_metal_buffer_bytes(expected_sum([
            expected_product(3, size_of::<super::MetalResidentRowProofArgs>())?,
            size_of::<super::MetalResidentRowFinalizeArgs>(),
        ])?)?
    );
    assert_eq!(
        breakdown.finalize_position_device,
        expected_metal_buffer_bytes(expected_product(
            super::METAL_ROW_PROOF_HEADER_WORDS + expected_product(INPUT_ROWS, 2)?,
            size_of::<u32>(),
        )?)?
    );
    assert_eq!(
        breakdown.sort_workspace, expected_sort_workspace,
        "byte and presence digit tensors coexist with one complete radix workspace"
    );
    Ok(())
}

#[test]
fn materialized_string_total_uses_the_independent_one_byte_admission_boundary() -> Result<()> {
    const LITERAL: &str = "0123456789abcdefg";
    let program = ResidentRowProgram {
        instructions: vec![
            instruction(
                ResidentRowValueType::Integer,
                ResidentRowOperation::InputColumn(ResidentRowColumn::Integer {
                    values: vec![1; INPUT_ROWS],
                    validity: vec![1; INPUT_ROWS],
                }),
            ),
            instruction(
                ResidentRowValueType::String,
                ResidentRowOperation::StringConstant(LITERAL.to_owned()),
            ),
        ],
    };
    let request = scalar_request(request(
        18,
        program,
        vec![key(1)],
        0,
        INPUT_ROWS,
        vec![0],
        false,
    )?)?;
    let strategy = metal_row_sort_strategy(&request, INPUT_ROWS, INPUT_ROWS)?;
    assert_eq!(strategy, MetalRowSortStrategy::Radix);
    let breakdown = metal_resident_row_program_scratch_breakdown(&request, INPUT_ROWS, strategy)?;

    let row_upload = expected_metal_buffer_bytes(expected_product(INPUT_ROWS, size_of::<i64>())?)?;
    let literal_upload =
        expected_metal_buffer_bytes(expected_product(LITERAL.len(), size_of::<i64>())?)?;
    let digit_buffers = expected_product(2, expected_metal_buffer_bytes(INPUT_ROWS)?)?;
    let expected = MetalResidentRowScratchBreakdown {
        typed_input_device: expected_sum([expected_product(2, row_upload)?, literal_upload])?,
        typed_input_host: expected_product(2, row_upload)?.max(literal_upload),
        property_image_device: expected_metal_buffer_bytes(expected_product(
            expected_product(INPUT_ROWS, 2)? + LITERAL.len(),
            size_of::<i64>(),
        )?)?,
        evaluated_device: expected_evaluated_device(
            INPUT_ROWS,
            2,
            expected_product(INPUT_ROWS, LITERAL.len())?,
        )?,
        sort_workspace: metal_row_sort_workspace_bytes(strategy, INPUT_ROWS)?
            .checked_add(digit_buffers)
            .ok_or_else(|| {
                crate::Error::new(
                    ErrorCode::ResultBudgetExceeded,
                    "row scratch test exact STRING total overflow",
                )
            })?,
        ..breakdown
    };
    assert_expected_required_boundary(&request, INPUT_ROWS, expected)
}

#[test]
fn direct_graph_string_rank_accounts_gather_temporaries_without_an_arena() -> Result<()> {
    let program = ResidentRowProgram {
        instructions: vec![instruction(
            ResidentRowValueType::String,
            ResidentRowOperation::LoadStringProperty {
                binding: ResidentEntityBinding::Node(ResidentNodeBinding::Start),
                property: PropertyId(41),
                maximum_bytes: 17,
            },
        )],
    };
    let request = request(11, program, vec![key(0)], 0, 128, Vec::new(), false)?;
    let breakdown = assert_exact_admission_boundary(&request)?;

    assert_eq!(metal_row_string_arena_bytes(&request, INPUT_ROWS)?, 0);
    assert_eq!(metal_row_string_slots(&request, INPUT_ROWS)?, vec![(0, 0)]);
    assert_eq!(
        breakdown.evaluated_device,
        expected_fixed_evaluated_device(INPUT_ROWS, 1)?
    );
    assert_eq!(
        metal_row_sort_strategy(&request, INPUT_ROWS, 128)?,
        MetalRowSortStrategy::SingleKeyTopK { window: 128 }
    );

    let rounded_i64 = expected_metal_buffer_bytes(expected_product(INPUT_ROWS, size_of::<i64>())?)?;
    let rounded_u32 = expected_metal_buffer_bytes(expected_product(INPUT_ROWS, size_of::<u32>())?)?;
    let rounded_u8 = expected_metal_buffer_bytes(INPUT_ROWS)?;
    let expected_gather = expected_product(2, rounded_i64)?;
    let expected_temporaries = expected_sum([
        expected_product(4, rounded_u8)?,
        expected_product(5, rounded_u32)?,
    ])?;
    let expected_property_image = expected_metal_buffer_bytes(expected_product(
        expected_product(INPUT_ROWS, 2)?,
        size_of::<i64>(),
    )?)?;
    assert_exact_fields(
        "direct graph STRING rank",
        &[
            (
                "rank and validity gathers",
                breakdown.property_gather_device,
                expected_gather,
            ),
            (
                "rank gather temporaries",
                breakdown.property_gather_temporaries,
                expected_temporaries,
            ),
            (
                "concatenated property image",
                breakdown.property_image_device,
                expected_property_image,
            ),
        ],
    );

    let expected = MetalResidentRowScratchBreakdown {
        property_gather_device: expected_gather,
        property_gather_temporaries: expected_temporaries,
        property_image_device: expected_property_image,
        ..breakdown
    };
    assert_expected_required_boundary(&request, INPUT_ROWS, expected)
}

#[test]
fn materialized_graph_string_accounts_gather_literal_arena_digits_and_proof() -> Result<()> {
    const SUFFIX: &str = "β\0";
    let program = ResidentRowProgram {
        instructions: vec![
            instruction(
                ResidentRowValueType::String,
                ResidentRowOperation::LoadStringProperty {
                    binding: ResidentEntityBinding::Node(ResidentNodeBinding::Start),
                    property: PropertyId(42),
                    maximum_bytes: 17,
                },
            ),
            instruction(
                ResidentRowValueType::String,
                ResidentRowOperation::StringConstant(SUFFIX.to_owned()),
            ),
            instruction(
                ResidentRowValueType::String,
                ResidentRowOperation::StringConcat { left: 0, right: 1 },
            ),
        ],
    };
    let request = request(12, program, vec![key(2)], 0, INPUT_ROWS, Vec::new(), false)?;
    let strategy = metal_row_sort_strategy(&request, INPUT_ROWS, INPUT_ROWS)?;
    assert_eq!(strategy, MetalRowSortStrategy::Radix);
    let breakdown = assert_exact_admission_boundary(&request)?;

    let rounded_i64 = expected_metal_buffer_bytes(expected_product(INPUT_ROWS, size_of::<i64>())?)?;
    let rounded_u32 = expected_metal_buffer_bytes(expected_product(INPUT_ROWS, size_of::<u32>())?)?;
    let rounded_u8 = expected_metal_buffer_bytes(INPUT_ROWS)?;
    let expected_gather = expected_product(2, rounded_i64)?;
    let expected_temporaries = expected_sum([
        expected_product(3, rounded_u8)?,
        expected_product(3, rounded_u32)?,
    ])?;
    let literal_upload =
        expected_metal_buffer_bytes(expected_product(SUFFIX.len(), size_of::<i64>())?)?;
    let arena_width = 17 + SUFFIX.len() + 17 + SUFFIX.len();
    let arena_bytes = expected_product(INPUT_ROWS, arena_width)?;
    let expected_sort_workspace = metal_row_sort_workspace_bytes(strategy, INPUT_ROWS)?
        .checked_add(expected_product(2, rounded_u8)?)
        .ok_or_else(|| {
            crate::Error::new(
                ErrorCode::ResultBudgetExceeded,
                "row scratch test graph STRING sort workspace overflow",
            )
        })?;
    let property_words = expected_product(INPUT_ROWS, 2)? + SUFFIX.len();

    assert_eq!(
        metal_row_string_arena_bytes(&request, INPUT_ROWS)?,
        arena_bytes
    );
    assert_eq!(
        breakdown.evaluated_device,
        expected_evaluated_device(INPUT_ROWS, 3, arena_bytes)?
    );
    let expected_property_image =
        expected_metal_buffer_bytes(expected_product(property_words, size_of::<i64>())?)?;
    let expected_proof_graph =
        expected_metal_buffer_bytes(expected_product(INPUT_ROWS * 2, size_of::<u32>())?)?;
    assert_exact_fields(
        "materialized graph STRING",
        &[
            (
                "ID and validity gathers",
                breakdown.property_gather_device,
                expected_gather,
            ),
            (
                "materialized gather temporaries",
                breakdown.property_gather_temporaries,
                expected_temporaries,
            ),
            (
                "literal device upload",
                breakdown.typed_input_device,
                literal_upload,
            ),
            (
                "literal host conversion",
                breakdown.typed_input_host,
                literal_upload,
            ),
            (
                "concatenated property image",
                breakdown.property_image_device,
                expected_property_image,
            ),
            (
                "STRING digit plus radix workspace",
                breakdown.sort_workspace,
                expected_sort_workspace,
            ),
            (
                "graph proof image",
                breakdown.proof_graph_device,
                expected_proof_graph,
            ),
        ],
    );

    let expected = MetalResidentRowScratchBreakdown {
        typed_input_device: literal_upload,
        typed_input_host: literal_upload,
        property_gather_device: expected_gather,
        property_gather_temporaries: expected_temporaries,
        property_image_device: expected_property_image,
        evaluated_device: expected_evaluated_device(INPUT_ROWS, 3, arena_bytes)?,
        sort_workspace: expected_sort_workspace,
        ..breakdown
    };
    assert_expected_required_boundary(&request, INPUT_ROWS, expected)
}

#[test]
fn empty_materialized_strings_have_offsets_but_no_arena_or_digit_buffers() -> Result<()> {
    let program = ResidentRowProgram {
        instructions: vec![
            instruction(
                ResidentRowValueType::String,
                ResidentRowOperation::InputColumn(repeated_string_column(INPUT_ROWS, "")?),
            ),
            instruction(
                ResidentRowValueType::String,
                ResidentRowOperation::StringConstant(String::new()),
            ),
            instruction(
                ResidentRowValueType::String,
                ResidentRowOperation::StringConcat { left: 0, right: 1 },
            ),
        ],
    };
    let request = scalar_request(request(
        13,
        program,
        vec![key(2)],
        0,
        INPUT_ROWS,
        Vec::new(),
        false,
    )?)?;
    let strategy = metal_row_sort_strategy(&request, INPUT_ROWS, INPUT_ROWS)?;
    assert_eq!(strategy, MetalRowSortStrategy::Radix);
    let breakdown = assert_exact_admission_boundary(&request)?;

    assert_eq!(metal_row_string_arena_bytes(&request, INPUT_ROWS)?, 0);
    assert_eq!(
        breakdown.evaluated_device,
        expected_fixed_evaluated_device(INPUT_ROWS, 3)?
    );
    assert_eq!(
        breakdown.sort_workspace,
        metal_row_sort_workspace_bytes(strategy, INPUT_ROWS)?,
        "zero-width STRING keys run no byte/presence digit pass"
    );

    let row_upload = expected_metal_buffer_bytes(expected_product(INPUT_ROWS, size_of::<i64>())?)?;
    let offsets_upload =
        expected_metal_buffer_bytes(expected_product(INPUT_ROWS + 1, size_of::<i64>())?)?;
    let expected_uploads = expected_sum([expected_product(2, row_upload)?, offsets_upload])?;
    let expected_property_image = expected_metal_buffer_bytes(expected_product(
        expected_product(INPUT_ROWS, 3)? + 1,
        size_of::<i64>(),
    )?)?;
    assert_exact_fields(
        "empty materialized STRING",
        &[
            (
                "device values/validity/offset uploads",
                breakdown.typed_input_device,
                expected_uploads,
            ),
            (
                "host values/validity/offset conversions",
                breakdown.typed_input_host,
                expected_uploads,
            ),
            (
                "concatenated property image",
                breakdown.property_image_device,
                expected_property_image,
            ),
        ],
    );

    let expected = MetalResidentRowScratchBreakdown {
        typed_input_device: expected_uploads,
        typed_input_host: expected_uploads,
        property_image_device: expected_property_image,
        ..breakdown
    };
    assert_expected_required_boundary(&request, INPUT_ROWS, expected)
}

#[test]
fn zero_rows_account_the_canonical_offset_upload_but_no_string_arena_or_sort() -> Result<()> {
    let program = ResidentRowProgram {
        instructions: vec![
            instruction(
                ResidentRowValueType::String,
                ResidentRowOperation::InputColumn(repeated_string_column(0, "")?),
            ),
            instruction(
                ResidentRowValueType::String,
                ResidentRowOperation::StringConstant(String::new()),
            ),
            instruction(
                ResidentRowValueType::String,
                ResidentRowOperation::StringConcat { left: 0, right: 1 },
            ),
        ],
    };
    let request = scalar_request(request_for_rows(
        14,
        program,
        vec![key(2)],
        0,
        usize::MAX,
        Vec::new(),
        false,
        0,
    )?)?;
    let strategy = metal_row_sort_strategy(&request, 0, 0)?;
    assert_eq!(strategy, MetalRowSortStrategy::EmptyWindow);
    let breakdown = assert_exact_admission_boundary_for_rows(&request, 0)?;

    assert_eq!(metal_row_string_arena_bytes(&request, 0)?, 0);
    assert_eq!(
        breakdown.evaluated_device,
        expected_fixed_evaluated_device(0, 3)?
    );
    assert_eq!(breakdown.sort_workspace, 0);
    let canonical_offset_upload = expected_metal_buffer_bytes(size_of::<i64>())?;
    let expected_property_image = expected_metal_buffer_bytes(size_of::<i64>())?;
    assert_exact_fields(
        "zero-row materialized STRING",
        &[
            (
                "canonical offset device upload",
                breakdown.typed_input_device,
                canonical_offset_upload,
            ),
            (
                "canonical offset host conversion",
                breakdown.typed_input_host,
                canonical_offset_upload,
            ),
            (
                "empty property placeholder",
                breakdown.property_image_device,
                expected_property_image,
            ),
        ],
    );

    let expected = MetalResidentRowScratchBreakdown {
        typed_input_device: canonical_offset_upload,
        typed_input_host: canonical_offset_upload,
        ..breakdown
    };
    assert_expected_required_boundary(&request, 0, expected)
}

#[test]
fn zero_input_string_projection_has_exact_placeholder_and_empty_position_scratch() -> Result<()> {
    let program = ResidentRowProgram {
        instructions: vec![instruction(
            ResidentRowValueType::String,
            ResidentRowOperation::InputColumn(repeated_string_column(0, "")?),
        )],
    };
    let request = scalar_request(request_for_rows(
        24,
        program,
        vec![key(0)],
        0,
        0,
        vec![0],
        false,
        0,
    )?)?;
    assert_eq!(request.scalar_input_rows()?, Some(0));
    assert_eq!(request.output_cardinality(0)?, 0);
    let strategy = metal_row_sort_strategy(&request, 0, 0)?;
    assert_eq!(strategy, MetalRowSortStrategy::EmptyWindow);
    let breakdown = assert_exact_admission_boundary_for_rows(&request, 0)?;

    let empty_buffer = expected_metal_buffer_bytes(1)?;
    let canonical_offset_upload = expected_metal_buffer_bytes(size_of::<i64>())?;
    assert_exact_fields(
        "zero-row scalar STRING projection",
        &[
            (
                "only canonical offset device upload",
                breakdown.typed_input_device,
                canonical_offset_upload,
            ),
            (
                "only canonical offset host conversion",
                breakdown.typed_input_host,
                canonical_offset_upload,
            ),
            (
                "one-word property placeholder",
                breakdown.property_image_device,
                canonical_offset_upload,
            ),
            ("no sort workspace", breakdown.sort_workspace, 0),
            (
                "empty output position placeholder",
                breakdown.position_device,
                empty_buffer,
            ),
            (
                "empty proof position placeholder",
                breakdown.proof_position_device,
                empty_buffer,
            ),
        ],
    );
    Ok(())
}

#[test]
fn maximum_string_slots_fail_scratch_admission_at_the_packed_reference_boundary() -> Result<()> {
    fn graph_concat_request(
        seed: u64,
        width: u32,
        rows: usize,
    ) -> Result<ResidentRowProgramRequest> {
        request_for_rows(
            seed,
            ResidentRowProgram {
                instructions: vec![
                    instruction(
                        ResidentRowValueType::String,
                        ResidentRowOperation::LoadStringProperty {
                            binding: ResidentEntityBinding::Node(ResidentNodeBinding::Start),
                            property: PropertyId(43),
                            maximum_bytes: width,
                        },
                    ),
                    instruction(
                        ResidentRowValueType::String,
                        ResidentRowOperation::StringConstant(String::new()),
                    ),
                    instruction(
                        ResidentRowValueType::String,
                        ResidentRowOperation::StringConcat { left: 0, right: 1 },
                    ),
                ],
            },
            Vec::new(),
            0,
            usize::MAX,
            Vec::new(),
            false,
            rows,
        )
    }

    // Header + three value/validity register pairs + the per-row status word.
    let fixed_bytes = expected_product(8, size_of::<i64>())?;
    let largest_width = ((0x8000_0000_usize - fixed_bytes) / 2) as u32;
    let fitting = graph_concat_request(15, largest_width, 1)?;
    metal_row_string_slots(&fitting, 1)?;
    metal_resident_row_program_scratch_bytes(&fitting, 1)?;

    let zero_rows = graph_concat_request(17, u32::MAX, 0)?;
    assert_eq!(metal_row_string_arena_bytes(&zero_rows, 0)?, 0);
    metal_row_string_slots(&zero_rows, 0)?;
    metal_resident_row_program_scratch_bytes(&zero_rows, 0)?;

    let overflowing_concat = ResidentRowProgram {
        instructions: vec![
            instruction(
                ResidentRowValueType::String,
                ResidentRowOperation::LoadStringProperty {
                    binding: ResidentEntityBinding::Node(ResidentNodeBinding::Start),
                    property: PropertyId(44),
                    maximum_bytes: u32::MAX,
                },
            ),
            instruction(
                ResidentRowValueType::String,
                ResidentRowOperation::StringConstant("x".to_owned()),
            ),
            instruction(
                ResidentRowValueType::String,
                ResidentRowOperation::StringConcat { left: 0, right: 1 },
            ),
        ],
    };
    let concat_error = overflowing_concat
        .validate()
        .expect_err("STRING concatenation capacity above u32 must fail before allocation");
    assert_eq!(concat_error.code, ErrorCode::ResultBudgetExceeded);

    let overflowing = graph_concat_request(16, largest_width + 1, 1)?;
    let slot_error = metal_row_string_slots(&overflowing, 1)
        .expect_err("one byte beyond the packed STRING reference domain must fail slot layout");
    assert_eq!(slot_error.code, ErrorCode::ResultBudgetExceeded);
    let scratch_error = metal_resident_row_program_scratch_bytes(&overflowing, 1)
        .expect_err("scratch admission must reject every request the physical slot layout rejects");
    assert_eq!(scratch_error.code, ErrorCode::ResultBudgetExceeded);
    Ok(())
}

#[test]
fn date_and_local_time_account_exact_native_property_sort_and_projection_buffers() -> Result<()> {
    let program = ResidentRowProgram {
        instructions: vec![
            instruction(
                ResidentRowValueType::Date,
                ResidentRowOperation::LoadDateProperty {
                    binding: ResidentEntityBinding::Node(ResidentNodeBinding::Start),
                    property: PropertyId(50),
                },
            ),
            instruction(
                ResidentRowValueType::LocalTime,
                ResidentRowOperation::LoadLocalTimeProperty {
                    binding: ResidentEntityBinding::Node(ResidentNodeBinding::Start),
                    property: PropertyId(51),
                },
            ),
        ],
    };
    let request = request(
        19,
        program,
        vec![key(0), key(1)],
        0,
        TEMPORAL_OUTPUT_ROWS,
        vec![0, 1],
        false,
    )?;
    let output_rows = request.output_cardinality(INPUT_ROWS)?;
    assert_eq!(output_rows, TEMPORAL_OUTPUT_ROWS);
    let strategy = metal_row_sort_strategy(&request, INPUT_ROWS, output_rows)?;
    assert_eq!(strategy, MetalRowSortStrategy::Radix);
    let breakdown = metal_resident_row_program_scratch_breakdown(&request, INPUT_ROWS, strategy)?;

    let rounded_i64 = expected_metal_buffer_bytes(expected_product(INPUT_ROWS, size_of::<i64>())?)?;
    let rounded_u32 = expected_metal_buffer_bytes(expected_product(INPUT_ROWS, size_of::<u32>())?)?;
    let rounded_u8 = expected_metal_buffer_bytes(INPUT_ROWS)?;
    let expected_gather = expected_product(4, rounded_i64)?;
    let expected_temporaries = expected_sum([
        expected_product(6, rounded_u8)?,
        expected_product(4, rounded_u32)?,
    ])?;
    let expected_property_image = expected_metal_buffer_bytes(expected_product(
        expected_product(INPUT_ROWS, 4)?,
        size_of::<i64>(),
    )?)?;
    let expected_evaluated = expected_temporal_evaluated_device(INPUT_ROWS, 2, 0, 0)?;
    let expected_sort = metal_row_sort_workspace_bytes(strategy, INPUT_ROWS)?;

    let projection_words = expected_product(
        2,
        expected_sum([
            super::METAL_ROW_PACKET_PROJECTION_HEADER_WORDS,
            expected_product(output_rows, 2)?,
        ])?,
    )?;
    let expected_packet = expected_packet_allocation(&request, output_rows, projection_words)?;
    let output_validity = expected_metal_buffer_bytes(output_rows)?;
    let output_i64 = expected_metal_buffer_bytes(expected_product(output_rows, size_of::<i64>())?)?;
    let projection_payloads = expected_product(2, expected_sum([output_validity, output_i64])?)?;
    let expected_decoded = expected_decoded_host(&request, output_rows, projection_payloads, 0)?;

    assert_exact_fields(
        "native Date and LocalTime",
        &[
            (
                "four gathered value/validity columns",
                breakdown.property_gather_device,
                expected_gather,
            ),
            (
                "two temporal gather pipelines",
                breakdown.property_gather_temporaries,
                expected_temporaries,
            ),
            (
                "four-column query-local property image",
                breakdown.property_image_device,
                expected_property_image,
            ),
            (
                "fixed temporal register frame",
                breakdown.evaluated_device,
                expected_evaluated,
            ),
            (
                "stable two-key radix workspace",
                breakdown.sort_workspace,
                expected_sort,
            ),
            (
                "projection packet",
                breakdown.packet_device,
                expected_packet,
            ),
            (
                "projection readback",
                breakdown.packet_readback_staging,
                expected_packet,
            ),
            (
                "owned packet host image",
                breakdown.packet_host,
                expected_packet,
            ),
            (
                "decoded temporal columns",
                breakdown.decoded_host,
                expected_decoded,
            ),
        ],
    );
    assert_eq!(metal_row_string_arena_bytes(&request, INPUT_ROWS)?, 0);

    let expected = MetalResidentRowScratchBreakdown {
        property_gather_device: expected_gather,
        property_gather_temporaries: expected_temporaries,
        property_image_device: expected_property_image,
        evaluated_device: expected_evaluated,
        sort_workspace: expected_sort,
        packet_device: expected_packet,
        packet_readback_staging: expected_packet,
        packet_host: expected_packet,
        decoded_host: expected_decoded,
        ..breakdown
    };
    assert_expected_required_boundary(&request, INPUT_ROWS, expected)
}

#[test]
fn zoned_time_accounts_its_extra_i64_component_and_two_lane_sidecar() -> Result<()> {
    let program = ResidentRowProgram {
        instructions: vec![instruction(
            ResidentRowValueType::ZonedTime,
            ResidentRowOperation::LoadZonedTimeProperty {
                binding: ResidentEntityBinding::Node(ResidentNodeBinding::Start),
                property: PropertyId(52),
            },
        )],
    };
    let request = request(
        20,
        program,
        vec![key(0)],
        0,
        TEMPORAL_OUTPUT_ROWS,
        vec![0],
        false,
    )?;
    let output_rows = request.output_cardinality(INPUT_ROWS)?;
    let strategy = metal_row_sort_strategy(&request, INPUT_ROWS, output_rows)?;
    assert_eq!(strategy, MetalRowSortStrategy::Radix);
    let breakdown = metal_resident_row_program_scratch_breakdown(&request, INPUT_ROWS, strategy)?;

    let rounded_i64 = expected_metal_buffer_bytes(expected_product(INPUT_ROWS, size_of::<i64>())?)?;
    let rounded_u32 = expected_metal_buffer_bytes(expected_product(INPUT_ROWS, size_of::<u32>())?)?;
    let rounded_u8 = expected_metal_buffer_bytes(INPUT_ROWS)?;
    let expected_gather = expected_product(3, rounded_i64)?;
    let expected_temporaries = expected_sum([
        expected_product(3, rounded_u8)?,
        expected_product(2, rounded_u32)?,
    ])?;
    let expected_property_image = expected_metal_buffer_bytes(expected_product(
        expected_product(INPUT_ROWS, 3)?,
        size_of::<i64>(),
    )?)?;
    let expected_evaluated = expected_temporal_evaluated_device(INPUT_ROWS, 1, 2, 0)?;
    let expected_sort = metal_row_sort_workspace_bytes(strategy, INPUT_ROWS)?;

    let projection_words = expected_sum([
        super::METAL_ROW_PACKET_PROJECTION_HEADER_WORDS,
        expected_product(output_rows, 3)?,
    ])?;
    let expected_packet = expected_packet_allocation(&request, output_rows, projection_words)?;
    let projection_payloads = expected_sum([
        expected_metal_buffer_bytes(output_rows)?,
        expected_metal_buffer_bytes(expected_product(output_rows, size_of::<i64>())?)?,
        expected_metal_buffer_bytes(expected_product(output_rows, size_of::<u32>())?)?,
    ])?;
    let expected_decoded = expected_decoded_host(&request, output_rows, projection_payloads, 0)?;

    assert_exact_fields(
        "native ZonedTime",
        &[
            (
                "nanos/validity/offset gathers",
                breakdown.property_gather_device,
                expected_gather,
            ),
            (
                "I64 component gather temporaries",
                breakdown.property_gather_temporaries,
                expected_temporaries,
            ),
            (
                "three-column query-local property image",
                breakdown.property_image_device,
                expected_property_image,
            ),
            (
                "normalized value plus local-nanos/offset sidecar",
                breakdown.evaluated_device,
                expected_evaluated,
            ),
            (
                "composite temporal radix workspace",
                breakdown.sort_workspace,
                expected_sort,
            ),
            (
                "projection packet",
                breakdown.packet_device,
                expected_packet,
            ),
            (
                "projection readback",
                breakdown.packet_readback_staging,
                expected_packet,
            ),
            (
                "owned packet host image",
                breakdown.packet_host,
                expected_packet,
            ),
            (
                "decoded ZonedTime column",
                breakdown.decoded_host,
                expected_decoded,
            ),
        ],
    );

    let expected = MetalResidentRowScratchBreakdown {
        property_gather_device: expected_gather,
        property_gather_temporaries: expected_temporaries,
        property_image_device: expected_property_image,
        evaluated_device: expected_evaluated,
        sort_workspace: expected_sort,
        packet_device: expected_packet,
        packet_readback_staging: expected_packet,
        packet_host: expected_packet,
        decoded_host: expected_decoded,
        ..breakdown
    };
    assert_expected_required_boundary(&request, INPUT_ROWS, expected)
}

#[test]
fn local_datetime_accounts_the_u32_nanos_conversion_and_sidecar() -> Result<()> {
    let program = ResidentRowProgram {
        instructions: vec![instruction(
            ResidentRowValueType::LocalDateTime,
            ResidentRowOperation::LoadLocalDateTimeProperty {
                binding: ResidentEntityBinding::Node(ResidentNodeBinding::Start),
                property: PropertyId(53),
            },
        )],
    };
    let request = request(
        21,
        program,
        vec![key(0)],
        0,
        TEMPORAL_OUTPUT_ROWS,
        vec![0],
        false,
    )?;
    let output_rows = request.output_cardinality(INPUT_ROWS)?;
    let strategy = metal_row_sort_strategy(&request, INPUT_ROWS, output_rows)?;
    assert_eq!(strategy, MetalRowSortStrategy::Radix);
    let breakdown = metal_resident_row_program_scratch_breakdown(&request, INPUT_ROWS, strategy)?;

    let rounded_i64 = expected_metal_buffer_bytes(expected_product(INPUT_ROWS, size_of::<i64>())?)?;
    let rounded_u32 = expected_metal_buffer_bytes(expected_product(INPUT_ROWS, size_of::<u32>())?)?;
    let rounded_u8 = expected_metal_buffer_bytes(INPUT_ROWS)?;
    let expected_gather = expected_product(3, rounded_i64)?;
    let expected_temporaries = expected_sum([
        expected_product(3, rounded_u8)?,
        expected_product(3, rounded_u32)?,
    ])?;
    let expected_property_image = expected_metal_buffer_bytes(expected_product(
        expected_product(INPUT_ROWS, 3)?,
        size_of::<i64>(),
    )?)?;
    let expected_evaluated = expected_temporal_evaluated_device(INPUT_ROWS, 1, 1, 0)?;
    let expected_sort = metal_row_sort_workspace_bytes(strategy, INPUT_ROWS)?;

    let projection_words = expected_sum([
        super::METAL_ROW_PACKET_PROJECTION_HEADER_WORDS,
        expected_product(output_rows, 3)?,
    ])?;
    let expected_packet = expected_packet_allocation(&request, output_rows, projection_words)?;
    let projection_payloads = expected_sum([
        expected_metal_buffer_bytes(output_rows)?,
        expected_metal_buffer_bytes(expected_product(output_rows, size_of::<i64>())?)?,
        expected_metal_buffer_bytes(expected_product(output_rows, size_of::<u32>())?)?,
    ])?;
    let expected_decoded = expected_decoded_host(&request, output_rows, projection_payloads, 0)?;

    assert_exact_fields(
        "native LocalDateTime",
        &[
            (
                "seconds/validity/nanos gathers",
                breakdown.property_gather_device,
                expected_gather,
            ),
            (
                "gather temporaries including U32 nanos conversion",
                breakdown.property_gather_temporaries,
                expected_temporaries,
            ),
            (
                "three-column query-local property image",
                breakdown.property_image_device,
                expected_property_image,
            ),
            (
                "seconds plus nanos sidecar",
                breakdown.evaluated_device,
                expected_evaluated,
            ),
            (
                "composite temporal radix workspace",
                breakdown.sort_workspace,
                expected_sort,
            ),
            (
                "projection packet",
                breakdown.packet_device,
                expected_packet,
            ),
            (
                "projection readback",
                breakdown.packet_readback_staging,
                expected_packet,
            ),
            (
                "owned packet host image",
                breakdown.packet_host,
                expected_packet,
            ),
            (
                "decoded LocalDateTime column",
                breakdown.decoded_host,
                expected_decoded,
            ),
        ],
    );

    let expected = MetalResidentRowScratchBreakdown {
        property_gather_device: expected_gather,
        property_gather_temporaries: expected_temporaries,
        property_image_device: expected_property_image,
        evaluated_device: expected_evaluated,
        sort_workspace: expected_sort,
        packet_device: expected_packet,
        packet_readback_staging: expected_packet,
        packet_host: expected_packet,
        decoded_host: expected_decoded,
        ..breakdown
    };
    assert_expected_required_boundary(&request, INPUT_ROWS, expected)
}

#[test]
fn zoned_datetime_accounts_components_timezone_digits_packet_and_decode_arenas() -> Result<()> {
    let program = ResidentRowProgram {
        instructions: vec![instruction(
            ResidentRowValueType::ZonedDateTime,
            ResidentRowOperation::LoadZonedDateTimeProperty {
                binding: ResidentEntityBinding::Node(ResidentNodeBinding::Start),
                property: PropertyId(54),
                maximum_timezone_bytes: u32::try_from(TIMEZONE_WIDTH).unwrap(),
            },
        )],
    };
    let request = request(
        22,
        program,
        vec![key(0)],
        0,
        TEMPORAL_OUTPUT_ROWS,
        vec![0],
        false,
    )?;
    let output_rows = request.output_cardinality(INPUT_ROWS)?;
    let strategy = metal_row_sort_strategy(&request, INPUT_ROWS, output_rows)?;
    assert_eq!(strategy, MetalRowSortStrategy::Radix);
    let breakdown = metal_resident_row_program_scratch_breakdown(&request, INPUT_ROWS, strategy)?;

    let rounded_i64 = expected_metal_buffer_bytes(expected_product(INPUT_ROWS, size_of::<i64>())?)?;
    let rounded_u32 = expected_metal_buffer_bytes(expected_product(INPUT_ROWS, size_of::<u32>())?)?;
    let rounded_u8 = expected_metal_buffer_bytes(INPUT_ROWS)?;
    let expected_gather = expected_product(4, rounded_i64)?;
    let expected_temporaries = expected_sum([
        expected_product(3, rounded_u8)?,
        expected_product(4, rounded_u32)?,
    ])?;
    let expected_property_image = expected_metal_buffer_bytes(expected_product(
        expected_product(INPUT_ROWS, 4)?,
        size_of::<i64>(),
    )?)?;
    let timezone_arena_bytes = expected_product(INPUT_ROWS, TIMEZONE_WIDTH)?;
    let expected_evaluated =
        expected_temporal_evaluated_device(INPUT_ROWS, 1, 2, timezone_arena_bytes)?;
    let expected_sort = expected_sum([
        metal_row_sort_workspace_bytes(strategy, INPUT_ROWS)?,
        expected_product(2, rounded_u8)?,
    ])?;

    assert_eq!(
        metal_row_string_arena_bytes(&request, INPUT_ROWS)?,
        timezone_arena_bytes
    );
    let fixed_and_sidecar_words = expected_sum([
        expected_sum([expected_product(INPUT_ROWS, 2)?, 1, INPUT_ROWS])?,
        expected_product(INPUT_ROWS, 2)?,
    ])?;
    assert_eq!(
        metal_row_string_slots(&request, INPUT_ROWS)?,
        vec![(
            u32::try_from(expected_product(fixed_and_sidecar_words, size_of::<i64>())?).unwrap(),
            u32::try_from(TIMEZONE_WIDTH).unwrap(),
        )],
        "timezone UTF-8 starts after both the fixed frame and the temporal sidecars"
    );

    let projected_timezone_slot_bytes = expected_product(output_rows, TIMEZONE_WIDTH)?;
    let packed_timezone_words = projected_timezone_slot_bytes.div_ceil(size_of::<i64>());
    let projection_words = expected_sum([
        super::METAL_ROW_PACKET_PROJECTION_HEADER_WORDS,
        expected_product(output_rows, 4)?,
        packed_timezone_words,
    ])?;
    let expected_packet = expected_packet_allocation(&request, output_rows, projection_words)?;

    let decoded_timezone_offsets =
        expected_metal_buffer_bytes(expected_product(output_rows + 1, size_of::<u32>())?)?;
    let decoded_timezone_bytes = expected_metal_buffer_bytes(projected_timezone_slot_bytes)?;
    let projection_payloads = expected_sum([
        expected_metal_buffer_bytes(output_rows)?,
        decoded_timezone_offsets,
        decoded_timezone_bytes,
        expected_metal_buffer_bytes(expected_product(output_rows, size_of::<i64>())?)?,
        expected_metal_buffer_bytes(expected_product(output_rows, size_of::<u32>())?)?,
    ])?;
    let packed_timezone_bytes = packed_timezone_words
        .checked_mul(size_of::<i64>())
        .ok_or_else(|| {
            crate::Error::new(
                ErrorCode::ResultBudgetExceeded,
                "row scratch test packed timezone decode overflow",
            )
        })?;
    let decoded_temporary = expected_metal_buffer_bytes(packed_timezone_bytes)?;
    let expected_decoded = expected_decoded_host(
        &request,
        output_rows,
        projection_payloads,
        decoded_temporary,
    )?;

    assert_exact_fields(
        "native ZonedDateTime",
        &[
            (
                "seconds/validity/nanos/timezone-ID gathers",
                breakdown.property_gather_device,
                expected_gather,
            ),
            (
                "gather temporaries including two U32 conversions",
                breakdown.property_gather_temporaries,
                expected_temporaries,
            ),
            (
                "four-column query-local property image",
                breakdown.property_image_device,
                expected_property_image,
            ),
            (
                "temporal sidecars and timezone UTF-8 arena",
                breakdown.evaluated_device,
                expected_evaluated,
            ),
            (
                "radix plus timezone UTF-8 digit tensors",
                breakdown.sort_workspace,
                expected_sort,
            ),
            (
                "variable-width projection packet",
                breakdown.packet_device,
                expected_packet,
            ),
            (
                "variable-width projection readback",
                breakdown.packet_readback_staging,
                expected_packet,
            ),
            (
                "owned variable-width packet image",
                breakdown.packet_host,
                expected_packet,
            ),
            (
                "decoded timezone offsets/bytes/packed temporary",
                breakdown.decoded_host,
                expected_decoded,
            ),
        ],
    );

    let expected = MetalResidentRowScratchBreakdown {
        property_gather_device: expected_gather,
        property_gather_temporaries: expected_temporaries,
        property_image_device: expected_property_image,
        evaluated_device: expected_evaluated,
        sort_workspace: expected_sort,
        packet_device: expected_packet,
        packet_readback_staging: expected_packet,
        packet_host: expected_packet,
        decoded_host: expected_decoded,
        ..breakdown
    };
    assert_expected_required_boundary(&request, INPUT_ROWS, expected)
}

#[test]
fn zero_rows_keep_temporal_placeholders_but_allocate_no_timezone_or_digit_arena() -> Result<()> {
    let program = ResidentRowProgram {
        instructions: vec![instruction(
            ResidentRowValueType::ZonedDateTime,
            ResidentRowOperation::LoadZonedDateTimeProperty {
                binding: ResidentEntityBinding::Node(ResidentNodeBinding::Start),
                property: PropertyId(55),
                maximum_timezone_bytes: u32::try_from(TIMEZONE_WIDTH).unwrap(),
            },
        )],
    };
    let request = request_for_rows(23, program, vec![key(0)], 0, usize::MAX, vec![0], false, 0)?;
    let strategy = metal_row_sort_strategy(&request, 0, 0)?;
    assert_eq!(strategy, MetalRowSortStrategy::EmptyWindow);
    let breakdown = metal_resident_row_program_scratch_breakdown(&request, 0, strategy)?;

    let empty_buffer = expected_metal_buffer_bytes(1)?;
    let expected_gather = expected_product(4, empty_buffer)?;
    let expected_property_image = expected_metal_buffer_bytes(size_of::<i64>())?;
    let expected_evaluated = expected_temporal_evaluated_device(0, 1, 2, 0)?;
    assert_eq!(metal_row_string_arena_bytes(&request, 0)?, 0);
    assert_eq!(
        metal_row_string_slots(&request, 0)?,
        vec![(size_of::<i64>() as u32, TIMEZONE_WIDTH as u32)]
    );

    let projection_words = super::METAL_ROW_PACKET_PROJECTION_HEADER_WORDS;
    let expected_packet = expected_packet_allocation(&request, 0, projection_words)?;
    let projection_payloads = expected_metal_buffer_bytes(size_of::<u32>())?;
    let expected_decoded = expected_decoded_host(&request, 0, projection_payloads, 0)?;

    assert_exact_fields(
        "zero-row native ZonedDateTime",
        &[
            (
                "four canonical empty gather tensors",
                breakdown.property_gather_device,
                expected_gather,
            ),
            (
                "no gather conversion temporaries",
                breakdown.property_gather_temporaries,
                0,
            ),
            (
                "one-word property placeholder",
                breakdown.property_image_device,
                expected_property_image,
            ),
            (
                "one-word evaluated placeholder",
                breakdown.evaluated_device,
                expected_evaluated,
            ),
            (
                "no sort or UTF-8 digit workspace",
                breakdown.sort_workspace,
                0,
            ),
            (
                "header-only temporal packet",
                breakdown.packet_device,
                expected_packet,
            ),
            (
                "header-only temporal packet readback",
                breakdown.packet_readback_staging,
                expected_packet,
            ),
            (
                "owned header-only packet",
                breakdown.packet_host,
                expected_packet,
            ),
            (
                "projection owner plus canonical timezone offset",
                breakdown.decoded_host,
                expected_decoded,
            ),
        ],
    );

    let expected = MetalResidentRowScratchBreakdown {
        property_gather_device: expected_gather,
        property_gather_temporaries: 0,
        property_image_device: expected_property_image,
        evaluated_device: expected_evaluated,
        sort_workspace: 0,
        packet_device: expected_packet,
        packet_readback_staging: expected_packet,
        packet_host: expected_packet,
        decoded_host: expected_decoded,
        ..breakdown
    };
    assert_expected_required_boundary(&request, 0, expected)
}
